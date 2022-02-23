import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DilatedResidualBlock(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super().__init__()
        
        self.conv_1x1_1 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.conv_d1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.conv_d2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv_d3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        
        self.conv_1x1_2 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
    
    def forward(self, inputs):
        x0 = self.conv_1x1_1(inputs)
        
        x1 = self.conv_d1(x0)
        x2 = self.conv_d2(x1)
        x3 = self.conv_d3(x2)
        
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv_1x1_2(x)
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


class DualDRNet(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.num_class = num_class
        
        self.pi = 3.14159
        self.full_size = self.model_cfg.FULL_SIZE_OF_RANGE_IMAGE
        self.front_size = self.model_cfg.FRONT_SIZE_OF_RANGE_IMAGE
        self.lidar_fov_up = self.model_cfg.LIDAR_FOV_UP * self.pi / 180
        self.lidar_fov_down = self.model_cfg.LIDAR_FOV_DOWN * self.pi / 180
        
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
        
        self.channel_split = self.model_cfg.CHANNEL_SPLIT
        assert self.channel_split[0] + self.channel_split[1] == self.input_channels
        
        num_downsample_blocks = len(downsample_filters)
        c_in_list_1 = [self.channel_split[0], *downsample_filters[:-1]]
        self.downsample_blocks_1 = nn.ModuleList()
        for idx in range(num_downsample_blocks):
            self.downsample_blocks_1.append(
                DownsampleDilatedResidualBlock(c_in_list_1[idx], downsample_filters[idx], use_pool[idx])
            )
        c_in_list_2 = [self.channel_split[1], *downsample_filters[:-1]]
        self.downsample_blocks_2 = nn.ModuleList()
        for idx in range(num_downsample_blocks):
            self.downsample_blocks_2.append(
                DownsampleDilatedResidualBlock(c_in_list_2[idx], downsample_filters[idx], use_pool[idx])
            )
        
        num_upsample_blocks = len(upsample_filters)
        c_in_list_1 = [downsample_filters[-1], *upsample_filters[:-1]]
        self.upsample_blocks_1 = nn.ModuleList()
        for idx in range(num_upsample_blocks):
            self.upsample_blocks_1.append(
                UpsampleDilatedResidualBlock(c_in_list_1[idx], upsample_filters[idx], use_interpolate[idx])
            )
        c_in_list_2 = [downsample_filters[-1], *upsample_filters[:-1]]
        self.upsample_blocks_2 = nn.ModuleList()
        for idx in range(num_upsample_blocks):
            self.upsample_blocks_2.append(
                UpsampleDilatedResidualBlock(c_in_list_2[idx], upsample_filters[idx], use_interpolate[idx])
            )

        self.num_rv_features = self.num_class + 1
        self.conv_3x3_1 = nn.Conv2d(upsample_filters[-1], self.num_rv_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_3x3_2 = nn.Conv2d(upsample_filters[-1], self.num_rv_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.fusion_layers = AttentionalFusionModule(self.num_rv_features)
    
    def forward(self, batch_dict, **kwargs):
        batch_points = batch_dict['colored_points'] # (N1 + N2 + ..., 8), Points of (batch_id, x, y, z, intensity, r, g, b)
        batch_point_features = batch_dict['point_features'] # (N1 + N2 + ..., input_channels)
        
        batch_point_us = []
        batch_point_vs = []
        batch_range_images = []
        batch_size = batch_points[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            batch_mask = batch_points[:, 0] == batch_idx
            this_points = batch_points[batch_mask, :] # (Ni, 8), Points of (batch_id, x, y, z, intensity, r, g, b)
            this_features = batch_point_features[batch_mask, :] # (Ni, input_channels)
            
            xs = this_points[:, 1]
            ys = this_points[:, 2]
            zs = this_points[:, 3]
            rs = torch.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
            
            us = 0.5 * (1 - torch.atan2(ys, xs) / self.pi) * self.full_size[1]
            vs = (1 - (torch.arcsin(zs / rs) + self.lidar_fov_down) / (self.lidar_fov_up + self.lidar_fov_down)) * self.full_size[0]
            us = (torch.clip(us, min=0, max=self.full_size[1] - 1)).type(torch.long)
            vs = (torch.clip(vs, min=0, max=self.full_size[0] - 1)).type(torch.long)
            
            full_range_image = torch.zeros(
                (self.input_channels, self.full_size[0], self.full_size[1]),
                dtype=this_features.dtype,
                device=this_features.device
            )
            
            full_range_image[:, vs, us] = this_features.t()
            if batch_dict.get('random_world_rotation', None) is not None:
                extra_u = self.full_size[1] * batch_dict['random_world_rotation'][batch_idx] / (2 * self.pi)
            else:
                extra_u = 0
            min_u = int(self.full_size[1] / 2 - self.front_size[1] / 2 - extra_u)
            max_u = int(min_u + self.front_size[1])
            front_range_image = full_range_image[:, 0:self.front_size[0], min_u:max_u]
            
            # import matplotlib.pyplot as plt
            # plt.imshow(front_range_image[3:4, :, :].permute(1, 2, 0).cpu().numpy())
            # plt.show()
            
            batch_point_us.append(us[:, None])
            batch_point_vs.append(vs[:, None])
            batch_range_images.append(front_range_image)
        
        batch_point_us = torch.cat(batch_point_us, dim=0)
        batch_point_vs = torch.cat(batch_point_vs, dim=0)
        batch_range_images = torch.stack(batch_range_images, dim=0)
        
        x_1 = batch_range_images[:, :self.channel_split[0], :, :]
        skip_features_1 = []
        for i in range(len(self.downsample_blocks_1)):
            x_1 = self.downsample_blocks_1[i](x_1)
            skip_features_1.append(x_1)
            
        for i in range(len(self.upsample_blocks_1)):
            x_1 = self.upsample_blocks_1[i](x_1, skip_features_1[-i - 2])
        x_1 = self.conv_3x3_1(x_1)
        
        x_2 = batch_range_images[:, self.channel_split[0]:, :, :]
        skip_features_2 = []
        for i in range(len(self.downsample_blocks_2)):
            x_2 = self.downsample_blocks_2[i](x_2)
            skip_features_2.append(x_2)
            
        for i in range(len(self.upsample_blocks_2)):
            x_2 = self.upsample_blocks_2[i](x_2, skip_features_2[-i - 2])
        x_2 = self.conv_3x3_2(x_2)
        
        batch_range_images = self.fusion_layers(x_1, x_2)
        batch_range_images = F.softmax(batch_range_images, dim=1)
        
        batch_rv_features = []
        batch_size = batch_points[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            batch_mask = batch_points[:, 0] == batch_idx
            us = batch_point_us[batch_mask, :].squeeze()
            vs = batch_point_vs[batch_mask, :].squeeze()
            range_image = batch_range_images[batch_idx, ...]
            
            full_range_image = torch.zeros(
                (self.num_rv_features, self.full_size[0], self.full_size[1]),
                dtype=range_image.dtype,
                device=range_image.device
            )
            
            if batch_dict.get('random_world_rotation', None) is not None:
                extra_u = self.full_size[1] * batch_dict['random_world_rotation'][batch_idx] / (2 * self.pi)
            else:
                extra_u = 0
            min_u = int(self.full_size[1] / 2 - self.front_size[1] / 2 - extra_u)
            max_u = int(min_u + self.front_size[1])
            full_range_image[:, 0:self.front_size[0], min_u:max_u] = range_image
            rv_features = full_range_image[:, vs, us].t()
            
            batch_rv_features.append(rv_features)
        
        batch_rv_features = torch.cat(batch_rv_features, dim=0)
        batch_dict['rv_features'] = batch_rv_features
        
        return batch_dict