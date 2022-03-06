import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from spconv.pytorch.utils import PointToVoxel


class PFNLayer(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super().__init__()

        self.linear = nn.Linear(inp_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, inputs):
        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        return torch.max(x, dim=1, keepdim=False)[0]


class AttentionalFusionModule(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        
        self.w_1 = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        )
        self.w_2 = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        )
        
    def forward(self, x_1, x_2):
        weight_1 = self.w_1(x_1)
        weight_2 = self.w_2(x_2)
        aw = torch.softmax(torch.cat([weight_1, weight_2], dim=1), dim=1)
        y = x_1 * aw[:, 0:1, :, :] + x_2 * aw[:, 1:2, :, :]
        return y.contiguous()


class DualPFE(nn.Module):
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
        
        self.base_channels = 4
        self.extra_channels = 0
        self.extra_channels += 3 if self.model_cfg.USE_RELATIVE_XYZ_TO_CLUSTER else 0
        self.extra_channels += 3 if self.model_cfg.USE_RELATIVE_XYZ_TO_CENTER else 0
        
        assert len(self.model_cfg.FILTERS) > 0
        self.filters = self.model_cfg.FILTERS
        
        self.num_class = self.input_channels - 1
        self.num_pv_features = self.filters[-1]
        
        self.pfn_layers1 = PFNLayer(self.base_channels + self.extra_channels, self.num_pv_features)
        self.pfn_layers2 = PFNLayer(self.base_channels + self.num_class, self.num_pv_features)
        
        self.fusion_layers = AttentionalFusionModule(self.num_pv_features)
    
    def forward(self, batch_dict, **kwargs):
        batch_points = batch_dict['colored_points'] # (N1 + N2 + ..., 8), Points of (batch_id, x, y, z, intensity, r, g, b)
        batch_rv_features = batch_dict['rv_features'] # (N1 + N2 + ..., input_channels)
        
        self.pillar_generator = PointToVoxel(
            vsize_xyz=self.model_cfg.PILLAR_SIZE,
            coors_range_xyz=self.point_cloud_range,
            num_point_features=self.base_channels + self.num_class,
            max_num_voxels=self.model_cfg.MAX_NUMBER_OF_PILLARS[self.mode],
            max_num_points_per_voxel=self.model_cfg.MAX_POINTS_PER_PILLAR,
            device=batch_points.device
        )
        
        batch_pv_features = []
        batch_pv_seg_features = []
        batch_coords = []
        batch_size = batch_points[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            batch_mask = batch_points[:, 0] == batch_idx
            this_points = batch_points[batch_mask, :] # (Ni, 8), Points of (batch_id, x, y, z, intensity, r, g, b)
            rv_features = batch_rv_features[batch_mask, :] # (Ni, input_channels)
            
            rv_features = torch.softmax(rv_features, dim=-1)[:, 1:] # (Ni, num_class)
            
            # pillars: (num_pillars, max_points_per_pillar, base_channels + num_class)
            # coords: (num_pillars, 3), Location of pillars, [zi, yi, xi], zi should be 0
            # num_points_per_pillar: (num_pillars), Number of points in each pillar
            pillars, coords, num_points_per_pillar = self.pillar_generator(torch.cat([this_points[:, 1:5], rv_features], dim=-1))
            pv_features = pillars[:, :, :4]
            pv_seg_features = pillars
            
            absolute_xyz = pv_features[:, :, :3]
            if self.model_cfg.USE_RELATIVE_XYZ_TO_CLUSTER:
                mean_xyz = absolute_xyz.sum(dim=1, keepdim=True) / num_points_per_pillar.type_as(absolute_xyz).view(-1, 1, 1)
                xyz_to_cluster = absolute_xyz - mean_xyz
                pv_features = torch.cat([pv_features, xyz_to_cluster], dim=-1)
            if self.model_cfg.USE_RELATIVE_XYZ_TO_CENTER:
                xyz_to_center = torch.zeros_like(absolute_xyz)
                xyz_to_center[:, :, 0] = absolute_xyz[:, :, 0] - (coords[:, 2].type_as(absolute_xyz).unsqueeze(1) * self.pillar_x + self.x_offset)
                xyz_to_center[:, :, 1] = absolute_xyz[:, :, 1] - (coords[:, 1].type_as(absolute_xyz).unsqueeze(1) * self.pillar_y + self.y_offset)
                xyz_to_center[:, :, 2] = absolute_xyz[:, :, 2] - (coords[:, 0].type_as(absolute_xyz).unsqueeze(1) * self.pillar_z + self.z_offset)
                pv_features = torch.cat([pv_features, xyz_to_center], dim=-1)
            
            batch_pv_features.append(pv_features)
            batch_pv_seg_features.append(pv_seg_features)
            
            coords = torch.cat([torch.ones((coords.shape[0], 1), dtype=coords.dtype, device=coords.device) * batch_idx, coords], dim=-1)
            batch_coords.append(coords)
            
        batch_pv_features = torch.cat(batch_pv_features, dim=0)
        batch_pv_seg_features = torch.cat(batch_pv_seg_features, dim=0)
        batch_coords = torch.cat(batch_coords, dim=0)
        
        batch_pv_features = self.pfn_layers1(batch_pv_features)
        batch_pv_seg_features = self.pfn_layers2(batch_pv_seg_features)
        
        batch_bev_features = []
        batch_bev_seg_features = []
        batch_size = batch_coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            batch_mask = batch_coords[:, 0] == batch_idx
            this_coords = batch_coords[batch_mask, :]
            pv_features = batch_pv_features[batch_mask, :]
            pv_seg_features = batch_pv_seg_features[batch_mask, :]
            
            bev_features = torch.zeros(
                (self.num_pv_features, self.nz * self.ny * self.nx),
                dtype=pv_features.dtype,
                device=pv_features.device
            )
            bev_seg_features = torch.zeros(
                (self.num_pv_features, self.nz * self.ny * self.nx),
                dtype=pv_seg_features.dtype,
                device=pv_seg_features.device
            )
            
            indices = (this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]).type(torch.long)
            bev_features[:, indices] = pv_features.t()
            bev_seg_features[:, indices] = pv_seg_features.t()
            
            # ~ import matplotlib.pyplot as plt
            # ~ import time
            # ~ fig = plt.figure(figsize=(16, 8))
            # ~ fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
            # ~ for i in range(0, 3):
                # ~ img = bev_seg_features[i:i + 1, :].view(self.nz, self.ny, self.nx).permute(1, 2, 0).cpu().numpy()
                # ~ img = img[::-1, :, :] # y -> -y
                # ~ plt.subplot(1, 3, i + 1)
                # ~ plt.imshow(img)
                # ~ plt.axis('off')
            # ~ plt.show()
            # ~ fig.savefig(time.asctime(time.localtime(time.time())), dpi=200)
            
            batch_bev_features.append(bev_features)
            batch_bev_seg_features.append(bev_seg_features)
        
        batch_bev_features = torch.stack(batch_bev_features, dim=0)
        batch_bev_seg_features = torch.stack(batch_bev_seg_features, dim=0)
        
        batch_bev_features = batch_bev_features.view(
            batch_size, self.num_pv_features * self.nz, self.ny, self.nx) # (B, num_pv_features, ny, nx)
        batch_bev_seg_features = batch_bev_seg_features.view(
            batch_size, self.num_pv_features * self.nz, self.ny, self.nx) # (B, num_pv_features, ny, nx)
        
        batch_bev_features = self.fusion_layers(batch_bev_features, batch_bev_seg_features)
        batch_dict['pv_features'] = batch_bev_features # (B, num_pv_features, ny, nx)
        
        return batch_dict
    
