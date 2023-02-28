import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spconv.pytorch.utils import PointToVoxel


class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        x = self.linear(x)
        torch.backends.cudnn.enabled = False
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        torch.backends.cudnn.enabled = True
        x = F.relu(x)

        return torch.max(x, dim=1, keepdim=False)[0]


class PFE(nn.Module):
    def __init__(self, cfg, in_channels, point_cloud_range):
        super().__init__()

        self.pillar_size = cfg['pillar_size']
        self.point_cloud_range = point_cloud_range
        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.pillar_size)
        ).astype(np.int64)

        self.nx, self.ny, self.nz = self.grid_size
        self.pillar_x, self.pillar_y, self.pillar_z = self.pillar_size

        self.x_offset = self.pillar_x / 2 + self.point_cloud_range[0]
        self.y_offset = self.pillar_y / 2 + self.point_cloud_range[1]
        self.z_offset = self.pillar_z / 2 + self.point_cloud_range[2]
        
        self.base_channels = 4
        self.extra_channels = 6
        self.in_channels = in_channels

        assert len(cfg['filters']) == 1
        self.num_pv_features = cfg['filters'][-1]
        
        self.pfn_layers = PFNLayer(self.base_channels + self.extra_channels + self.in_channels, self.num_pv_features)
    
    def forward(self, batch_dict):
        batch_points = batch_dict['colored_points'][:, 0:5]  # [N1 + N2 + ..., 5], (batch_id, x, y, z, i)
        batch_size = batch_dict['batch_size']
        device = batch_points.device

        if self.in_channels > 0:
            batch_rv_features = batch_dict['rv_features']  # [N1 + N2 + ..., in_channels]
            batch_points = torch.cat([batch_points, batch_rv_features], dim=-1)

        pillar_generator = PointToVoxel(
            vsize_xyz=self.pillar_size,
            coors_range_xyz=self.point_cloud_range,
            num_point_features=self.base_channels + self.in_channels,
            max_num_voxels=16000 if self.training else 40000,
            max_num_points_per_voxel=32,
            device=device
        )

        batch_pillars = []
        batch_coords = []
        for batch_idx in range(batch_size):
            mask = batch_points[:, 0] == batch_idx
            points = batch_points[mask]

            # pillars: [num_pillars, max_points_per_pillar, num_point_features]
            # coords: [num_pillars, 3], locations of pillars, (zi, yi, xi), where zi is 0
            # num_points_per_pillar: [num_pillars], number of points in each pillar
            pillars, coords, num_points_per_pillar = pillar_generator(points[:, 1:].contiguous())

            xyz = pillars[:, :, :3]

            xyz_mean = xyz.sum(dim=1, keepdim=True) / num_points_per_pillar.type_as(xyz).view(-1, 1, 1)
            xyz_to_cluster = xyz - xyz_mean
            pillars = torch.cat([pillars, xyz_to_cluster], dim=-1)

            xyz_to_center = torch.zeros_like(xyz)
            xyz_to_center[:, :, 0] = xyz[:, :, 0] - (coords[:, 2].type_as(xyz).unsqueeze(1) * self.pillar_x + self.x_offset)
            xyz_to_center[:, :, 1] = xyz[:, :, 1] - (coords[:, 1].type_as(xyz).unsqueeze(1) * self.pillar_y + self.y_offset)
            xyz_to_center[:, :, 2] = xyz[:, :, 2] - (coords[:, 0].type_as(xyz).unsqueeze(1) * self.pillar_z + self.z_offset)
            pillars = torch.cat([pillars, xyz_to_center], dim=-1)

            batch_pillars.append(pillars)

            coords = torch.cat([torch.ones((coords.shape[0], 1), dtype=coords.dtype, device=device) * batch_idx, coords], dim=-1)
            batch_coords.append(coords)

        batch_pv_features = self.pfn_layers(torch.cat(batch_pillars, dim=0))  # [N, C]
        batch_coords = torch.cat(batch_coords, dim=0)

        x = []
        for batch_idx in range(batch_size):
            mask = batch_coords[:, 0] == batch_idx
            coords = batch_coords[mask]
            pv_features = batch_pv_features[mask]
            feature_map = torch.zeros(
                (self.num_pv_features, self.nz * self.ny * self.nx), dtype=pv_features.dtype, device=device
            )
            indices = (coords[:, 1] + coords[:, 2] * self.nx + coords[:, 3]).type(torch.long)
            feature_map[:, indices] = pv_features.t()
            x.append(feature_map)

        x = torch.stack(x, dim=0)
        x = x.view(batch_size, self.num_pv_features * self.nz, self.ny, self.nx)
        batch_dict['pv_features'] = x  # [batch_size, num_pv_features, ny, nx]

        return batch_dict
