import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from spconv.pytorch.utils import PointToVoxel
from utils.spconv_utils import spconv


class BaseVoxelBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, point_cloud_range, training, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.point_cloud_range = point_cloud_range
        self.mode = 'train' if self.training else 'test'
        
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.model_cfg.VOXEL_SIZE)
        self.grid_size = np.round(grid_size).astype(np.int64)
        self.spatial_shape = self.grid_size[::-1] + [1, 0, 0]
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(self.input_channels, 16, kernel_size=3, padding=1, bias=False, indice_key='subm1'),
            nn.BatchNorm1d(16, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        
        self.conv1 = spconv.SparseSequential(
            spconv.SparseSequential(
                spconv.SubMConv3d(16, 16, kernel_size=3, padding=1, bias=False, indice_key='subm1'),
                nn.BatchNorm1d(16, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )
        
        self.conv2 = spconv.SparseSequential(
            spconv.SparseSequential(
                spconv.SparseConv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False, indice_key='spconv2'),
                nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(32, 32, kernel_size=3, padding=1, bias=False, indice_key='subm2'),
                nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(32, 32, kernel_size=3, padding=1, bias=False, indice_key='subm2'),
                nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )
        
        self.conv3 = spconv.SparseSequential(
            spconv.SparseSequential(
                spconv.SparseConv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False, indice_key='spconv3'),
                nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(64, 64, kernel_size=3, padding=1, bias=False, indice_key='subm3'),
                nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(64, 64, kernel_size=3, padding=1, bias=False, indice_key='subm3'),
                nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )
        
        self.conv4 = spconv.SparseSequential(
            spconv.SparseSequential(
                spconv.SparseConv3d(64, 64, kernel_size=3, stride=2, padding=(0, 1, 1), bias=False, indice_key='spconv4'),
                nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(64, 64, kernel_size=3, padding=1, bias=False, indice_key='subm4'),
                nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(64, 64, kernel_size=3, padding=1, bias=False, indice_key='subm4'),
                nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )
        
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(64, 128, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key='spconv_down2'),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        
        self.num_voxel_features = 128 * 2
    
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
        
        spconv_tensor = spconv.SparseConvTensor(
            features=batch_voxels,
            indices=batch_coords.int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size
        )
        
        x = self.conv_input(spconv_tensor)
        
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        out = self.conv_out(x_conv4)
        
        batch_voxel_features = out.dense()
        N, C, D, H, W = batch_voxel_features.shape
        batch_voxel_features = batch_voxel_features.view(N, C * D, H, W)
        batch_dict['voxel_features'] = batch_voxel_features
        
        return batch_dict
