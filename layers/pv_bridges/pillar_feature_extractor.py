import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from spconv.pytorch.utils import PointToVoxel


class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super().__init__()
        
        self.last_layer = last_layer
        self.use_norm = use_norm
        if not self.last_layer:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_layer:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarFeatureExtractor(nn.Module):
    def __init__(self, model_cfg, input_channels, point_cloud_range, training, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.point_cloud_range = point_cloud_range
        self.mode = 'train' if self.training else 'test'
        
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.model_cfg.PILLAR_SIZE)
        self.grid_size = np.round(grid_size).astype(np.int64)
        self.nx, self.ny, self.nz = self.grid_size
        
        self.pillar_x = self.model_cfg.PILLAR_SIZE[0]
        self.pillar_y = self.model_cfg.PILLAR_SIZE[1]
        self.pillar_z = self.model_cfg.PILLAR_SIZE[2]
        self.x_offset = self.pillar_x / 2 + self.point_cloud_range[0]
        self.y_offset = self.pillar_y / 2 + self.point_cloud_range[1]
        self.z_offset = self.pillar_z / 2 + self.point_cloud_range[2]
        
        self.extra_channels = 0
        self.extra_channels += 3 if self.model_cfg.USE_ABSOLUTE_XYZ else 0
        self.extra_channels += 3 if self.model_cfg.USE_RELATIVE_XYZ_TO_CLUSTER else 0
        self.extra_channels += 3 if self.model_cfg.USE_RELATIVE_XYZ_TO_CENTER else 0
        
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        
        num_filters = [self.input_channels + self.extra_channels] + list(self.num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, use_norm=True, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)
        
        self.num_pv_features = self.num_filters[-1]
    
    def forward(self, batch_dict, **kwargs):
        batch_points = batch_dict['colored_points'] # (N1 + N2 + ..., 8), Points of (batch_id, x, y, z, intensity, r, g, b)
        batch_rv_features = batch_dict['rv_features'] # (N1 + N2 + ..., input_channels)
        
        self.pillar_generator = PointToVoxel(
            vsize_xyz=self.model_cfg.PILLAR_SIZE,
            coors_range_xyz=self.point_cloud_range,
            num_point_features=self.input_channels + 3,
            max_num_voxels=self.model_cfg.MAX_NUMBER_OF_PILLARS[self.mode],
            max_num_points_per_voxel=self.model_cfg.MAX_POINTS_PER_PILLAR,
            device=batch_points.device
        )
        
        batch_pillars = []
        batch_coords = []
        batch_size = batch_points[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            batch_mask = batch_points[:, 0] == batch_idx
            this_points = batch_points[batch_mask, :] # (Ni, 8), Points of (batch_id, x, y, z, intensity, r, g, b)
            this_features = batch_rv_features[batch_mask, :] # (Ni, input_channels)
            
            # pillars: (num_pillars, max_points_per_pillar, input_channels + 3)
            # coords: (num_pillars, 3), Location of pillars, [zi, yi, xi], zi should be 0
            # num_points_per_pillar: (num_pillars), Number of points in each pillar
            pillars, coords, num_points_per_pillar = self.pillar_generator(torch.cat([this_points[:, 1:4], this_features], dim=-1))
            absolute_xyz = pillars[:, :, :3]
            pillars = pillars[:, :, 3:]
            
            if self.model_cfg.USE_ABSOLUTE_XYZ:
                pillars = torch.cat([absolute_xyz, pillars], dim=-1)
            
            if self.model_cfg.USE_RELATIVE_XYZ_TO_CLUSTER:
                mean_xyz = absolute_xyz.sum(dim=1, keepdim=True) / num_points_per_pillar.type_as(absolute_xyz).view(-1, 1, 1)
                xyz_to_cluster = absolute_xyz - mean_xyz
                pillars = torch.cat([pillars, xyz_to_cluster], dim=-1)
            
            if self.model_cfg.USE_RELATIVE_XYZ_TO_CENTER:
                xyz_to_center = torch.zeros_like(absolute_xyz)
                xyz_to_center[:, :, 0] = absolute_xyz[:, :, 0] - (coords[:, 2].type_as(absolute_xyz).unsqueeze(1) * self.pillar_x + self.x_offset)
                xyz_to_center[:, :, 1] = absolute_xyz[:, :, 1] - (coords[:, 1].type_as(absolute_xyz).unsqueeze(1) * self.pillar_y + self.y_offset)
                xyz_to_center[:, :, 2] = absolute_xyz[:, :, 2] - (coords[:, 0].type_as(absolute_xyz).unsqueeze(1) * self.pillar_z + self.z_offset)
                pillars = torch.cat([pillars, xyz_to_center], dim=-1)
            
            max_num = torch.arange(self.model_cfg.MAX_POINTS_PER_PILLAR, dtype=torch.int, device=num_points_per_pillar.device).view(1, -1)
            mask = num_points_per_pillar.unsqueeze(1).int() > max_num # (num_pillars, max_points_per_pillar)
            mask = torch.unsqueeze(mask, -1).type_as(pillars)
            pillars *= mask
            batch_pillars.append(pillars)
            
            coords = torch.cat([torch.ones((coords.shape[0], 1), dtype=coords.dtype, device=coords.device) * batch_idx, coords], dim=-1)
            batch_coords.append(coords)
            
        batch_pillars = torch.cat(batch_pillars, dim=0)
        batch_coords = torch.cat(batch_coords, dim=0)
        
        for pfn in self.pfn_layers:
            batch_pillars = pfn(batch_pillars)
        batch_pillars = batch_pillars.squeeze()
        
        batch_pv_features = []
        batch_size = batch_coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            batch_mask = batch_coords[:, 0] == batch_idx
            this_coords = batch_coords[batch_mask, :]
            this_pillars = batch_pillars[batch_mask, :]
            
            pv_features = torch.zeros((self.num_pv_features, self.nz * self.ny * self.nx), dtype=this_pillars.dtype, device=this_pillars.device)
            
            indices = (this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]).type(torch.long)
            pv_features[:, indices] = this_pillars.t()
            batch_pv_features.append(pv_features)
        
        batch_pv_features = torch.stack(batch_pv_features, dim=0)
        batch_pv_features = batch_pv_features.view(batch_size, self.num_pv_features * self.nz, self.ny, self.nx)
        batch_dict['pv_features'] = batch_pv_features
        
        return batch_dict
