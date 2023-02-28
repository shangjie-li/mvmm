import numpy as np
import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel
import spconv.pytorch as spconv


def spconv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0, indice_key=None, conv_type='subm'):
    if conv_type == 'subm':
        return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
    elif conv_type == 'spconv':
        return spconv.SparseSequential(
            spconv.SparseConv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
    else:
        raise NotImplementedError


class VFE(nn.Module):
    def __init__(self, cfg, in_channels, point_cloud_range):
        super().__init__()

        self.voxel_size = cfg['voxel_size']
        self.point_cloud_range = point_cloud_range
        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        ).astype(np.int64)
        self.spatial_shape = self.grid_size[::-1] + [1, 0, 0]
        
        self.base_channels = 4
        self.in_channels = in_channels
        
        assert len(cfg['filters']) == 6
        filters = cfg['filters']

        self.conv0 = spconv.SparseSequential(
            spconv_block(self.base_channels + self.in_channels, filters[0], kernel_size=3, padding=1, indice_key='subm0'),
        )
        
        self.conv1 = spconv.SparseSequential(
            spconv_block(filters[0], filters[1], kernel_size=3, padding=1, indice_key='subm1'),
        )
        
        self.conv2 = spconv.SparseSequential(
            spconv_block(filters[1], filters[2], kernel_size=3, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            spconv_block(filters[2], filters[2], kernel_size=3, padding=1, indice_key='subm2'),
            spconv_block(filters[2], filters[2], kernel_size=3, padding=1, indice_key='subm2'),
        )
        
        self.conv3 = spconv.SparseSequential(
            spconv_block(filters[2], filters[3], kernel_size=3, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            spconv_block(filters[3], filters[3], kernel_size=3, padding=1, indice_key='subm3'),
            spconv_block(filters[3], filters[3], kernel_size=3, padding=1, indice_key='subm3'),
        )
        
        self.conv4 = spconv.SparseSequential(
            spconv_block(filters[3], filters[4], kernel_size=3, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            spconv_block(filters[4], filters[4], kernel_size=3, padding=1, indice_key='subm4'),
            spconv_block(filters[4], filters[4], kernel_size=3, padding=1, indice_key='subm4'),
        )
        
        self.conv5 = spconv.SparseSequential(
            spconv_block(filters[4], filters[5], kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=0, indice_key='spconv5', conv_type='spconv'),
        )
        
        self.num_pv_features = filters[5] * 2
    
    def forward(self, batch_dict):
        batch_points = batch_dict['colored_points'][:, 0:5]  # [N1 + N2 + ..., 5], (batch_id, x, y, z, i)
        batch_size = batch_dict['batch_size']
        device = batch_points.device

        if self.in_channels > 0:
            batch_rv_features = batch_dict['rv_features']  # [N1 + N2 + ..., in_channels]
            batch_points = torch.cat([batch_points, batch_rv_features], dim=-1)

        voxel_generator = PointToVoxel(
            vsize_xyz=self.voxel_size,
            coors_range_xyz=self.point_cloud_range,
            num_point_features=self.base_channels + self.in_channels,
            max_num_voxels=16000 if self.training else 40000,
            max_num_points_per_voxel=5,
            device=device
        )
        
        batch_voxels = []
        batch_coords = []
        for batch_idx in range(batch_size):
            mask = batch_points[:, 0] == batch_idx
            points = batch_points[mask]
            
            # voxels: [num_voxels, max_points_per_voxel, num_point_features]
            # coords: [num_voxels, 3], locations of voxels, (zi, yi, xi)
            # num_points_per_voxel: [num_voxels], number of points in each voxel
            voxels, coords, num_points_per_voxel = voxel_generator(points[:, 1:].contiguous())
            
            voxel_features = voxels.sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(num_points_per_voxel.view(-1, 1), min=1.0).type_as(voxel_features)
            voxel_features /= normalizer
            batch_voxels.append(voxel_features)  # voxel_features: [num_voxels, num_point_features]
            
            coords = torch.cat([torch.ones((coords.shape[0], 1), dtype=coords.dtype, device=device) * batch_idx, coords], dim=-1)
            batch_coords.append(coords)  # coords: [num_voxels, 4], (batch_id, zi, yi, xi)
        
        batch_voxels = torch.cat(batch_voxels, dim=0)
        batch_coords = torch.cat(batch_coords, dim=0)
        
        x = spconv.SparseConvTensor(
            features=batch_voxels,
            indices=batch_coords.int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size
        )
        
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = x.dense()
        B, C, D, H, W = x.shape
        x = x.view(B, C * D, H, W)
        batch_dict['pv_features'] = x  # [batch_size, num_pv_features, ny, nx]
        
        return batch_dict
