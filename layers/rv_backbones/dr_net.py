import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class DilatedResidualBlock(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super().__init__()
        
        self.conv_1x1_1 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv_d1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn_d1 = nn.BatchNorm2d(out_channels)
        self.conv_d2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn_d2 = nn.BatchNorm2d(out_channels)
        self.conv_d3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.bn_d3 = nn.BatchNorm2d(out_channels)
        
        self.conv_1x1_2 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, inputs):
        x0 = self.bn1(self.conv_1x1_1(inputs))
        
        x1 = self.bn_d1(self.conv_d1(x0))
        x2 = self.bn_d2(self.conv_d2(x1))
        x3 = self.bn_d3(self.conv_d3(x2))
        
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.bn2(self.conv_1x1_2(x))
        x += x0
        
        return x


class DownsampleDilatedResidualBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, use_pool=True):
        super().__init__()
        
        self.drb = DilatedResidualBlock(inp_channels, out_channels)
        self.use_pool = use_pool
        
    def forward(self, inputs):
        x = self.drb(inputs)
        
        if self.use_pool:
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        
        return x


class UpsampleDilatedResidualBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, use_interpolate=True):
        super().__init__()
        
        self.drb = DilatedResidualBlock(inp_channels * 2, out_channels)
        self.use_interpolate = use_interpolate
    
    def forward(self, inputs, skip_features):
        if self.use_interpolate:
            x = F.interpolate(inputs, scale_factor=2, mode='bilinear')
        else:
            x = inputs
        
        x = self.drb(torch.cat([x, skip_features], dim=1))
        
        return x


class DRNet(nn.Module):
    def __init__(self, model_cfg, input_channels, range_convertor, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.range_convertor = range_convertor
        
        if self.model_cfg.get('DOWNSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.DOWNSAMPLE_STRIDES) == len(self.model_cfg.DOWNSAMPLE_FILTERS) == len(self.model_cfg.USE_POOL)
            downsample_strides = self.model_cfg.DOWNSAMPLE_STRIDES
            downsample_filters = self.model_cfg.DOWNSAMPLE_FILTERS
            use_pool = self.model_cfg.USE_POOL
        else:
            raise NotImplementedError
        
        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.UPSAMPLE_FILTERS) == len(self.model_cfg.USE_INTERPOLATE)
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
            upsample_filters = self.model_cfg.UPSAMPLE_FILTERS
            use_interpolate = self.model_cfg.USE_INTERPOLATE
        else:
            raise NotImplementedError
        
        num_downsample_blocks = len(downsample_filters)
        c_in_list = [self.input_channels, *downsample_filters[:-1]]
        self.downsample_blocks = nn.ModuleList()
        for idx in range(num_downsample_blocks):
            self.downsample_blocks.append(
                DownsampleDilatedResidualBlock(c_in_list[idx], downsample_filters[idx], use_pool[idx])
            )
        
        num_upsample_blocks = len(upsample_filters)
        c_in_list = [downsample_filters[-1], *upsample_filters[:-1]]
        self.upsample_blocks = nn.ModuleList()
        for idx in range(num_upsample_blocks):
            self.upsample_blocks.append(
                UpsampleDilatedResidualBlock(c_in_list[idx], upsample_filters[idx], use_interpolate[idx])
            )
        
        assert len(self.model_cfg.FILTERS) > 0
        self.num_rv_features = self.model_cfg.FILTERS[-1]
        self.conv_1x1 = nn.Conv2d(upsample_filters[-1], self.num_rv_features, kernel_size=1)
    
    def forward(self, batch_dict, **kwargs):
        # colored_points: (N1 + N2 + ..., 8), points of (batch_id, x, y, z, intensity, r, g, b)
        # range_image: (batch_size, used_point_features, 48, 512), front range image
        batch_points = batch_dict['colored_points']
        batch_range_image = batch_dict['range_image']
        batch_size = batch_dict['batch_size']
        
        x = batch_range_image
        skip_features = []
        for i in range(len(self.downsample_blocks)):
            x = self.downsample_blocks[i](x)
            skip_features.append(x)
        for i in range(len(self.upsample_blocks)):
            x = self.upsample_blocks[i](x, skip_features[-i - 2])
        batch_range_image = self.conv_1x1(x)
        
        batch_rv_features = []
        for batch_idx in range(batch_size):
            mask = batch_points[:, 0] == batch_idx
            points = batch_points[mask, :]
            range_image = batch_range_image[batch_idx, ...]
            range_features = self.range_convertor.get_range_features(points, range_image)
            batch_rv_features.append(range_features)
        
        batch_rv_features = torch.cat(batch_rv_features, dim=0)
        batch_dict['rv_features'] = batch_rv_features # (N1 + N2 + ..., num_rv_features)
        
        return batch_dict
