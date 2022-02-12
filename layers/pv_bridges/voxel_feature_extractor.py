import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from spconv.pytorch.utils import PointToVoxel
from utils.spconv_utils import spconv


def spconv_block(inp_channels, out_channels, kernel_size=1, stride=1, padding=0, indice_key=None, conv_type='subm'):
    if conv_type == 'subm':
        return spconv.SparseSequential(
            spconv.SubMConv3d(inp_channels, out_channels, kernel_size=kernel_size, padding=padding,
                bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
    elif conv_type == 'spconv':
        return spconv.SparseSequential(
            spconv.SparseConv3d(inp_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
    else:
        raise NotImplementedError


class VFE(nn.Module):
    def __init__(self, model_cfg, input_channels, point_cloud_range, training, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.point_cloud_range = point_cloud_range
        self.mode = 'train' if self.training else 'test'
        
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.model_cfg.VOXEL_SIZE)
        self.grid_size = np.round(grid_size).astype(np.int64)
        self.spatial_shape = self.grid_size[::-1] + [1, 0, 0]
        
        assert len(self.model_cfg.FILTERS) == 6
        filters = self.model_cfg.FILTERS

        self.conv0 = spconv.SparseSequential(
            spconv_block(self.input_channels, filters[0], kernel_size=3, padding=1, indice_key='subm0'),
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
    
    def forward(self, batch_dict, **kwargs):
        batch_points = batch_dict['colored_points'] # (N1 + N2 + ..., 8), Points of (batch_id, x, y, z, intensity, r, g, b)
        batch_rv_features = batch_dict['rv_features'] # (N1 + N2 + ..., input_channels)
        
        self.voxel_generator = PointToVoxel(
            vsize_xyz=self.model_cfg.VOXEL_SIZE,
            coors_range_xyz=self.point_cloud_range,
            num_point_features=self.input_channels + 3,
            max_num_voxels=self.model_cfg.MAX_NUMBER_OF_VOXELS[self.mode],
            max_num_points_per_voxel=self.model_cfg.MAX_POINTS_PER_VOXEL,
            device=batch_points.device
        )
        
        batch_voxels = []
        batch_coords = []
        batch_size = batch_points[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            batch_mask = batch_points[:, 0] == batch_idx
            this_points = batch_points[batch_mask, :] # (Ni, 8), Points of (batch_id, x, y, z, intensity, r, g, b)
            this_features = batch_rv_features[batch_mask, :] # (Ni, input_channels)
            
            # voxels: (num_voxels, max_points_per_voxel, input_channels + 3)
            # coords: (num_voxels, 3), Location of voxels, [zi, yi, xi], zi should be 0
            # num_points_per_voxel: (num_voxels), Number of points in each voxel
            voxels, coords, num_points_per_voxel = self.voxel_generator(torch.cat([this_points[:, 1:4], this_features], dim=-1))
            
            voxel_features = voxels[:, :, 3:].sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(num_points_per_voxel.view(-1, 1), min=1.0).type_as(voxel_features)
            voxel_features /= normalizer
            batch_voxels.append(voxel_features) # voxel_features: (num_voxels, input_channels)
            
            coords = torch.cat([torch.ones((coords.shape[0], 1), dtype=coords.dtype, device=coords.device) * batch_idx, coords], dim=-1)
            batch_coords.append(coords) # coords: (num_voxels, 4), [batch_id, zi, yi, xi]
            
        batch_voxels = torch.cat(batch_voxels, dim=0)
        batch_coords = torch.cat(batch_coords, dim=0)
        
        x = spconv.SparseConvTensor(
            features=batch_voxels,
            indices=batch_coords.int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size
        )
        
        x_conv0 = self.conv0(x)
        x_conv1 = self.conv1(x_conv0)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        
        batch_pv_features = x_conv5.dense()
        N, C, D, H, W = batch_pv_features.shape
        batch_pv_features = batch_pv_features.view(N, C * D, H, W)
        batch_dict['pv_features'] = batch_pv_features
        
        return batch_dict
