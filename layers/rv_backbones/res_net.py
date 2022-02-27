import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from utils import loss_utils


class BottleNeck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class Encoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.input_channels = input_channels
        
        self.num_blocks = [3, 4, 6, 3]
        self.in_planes = 64
        
        self.layers = nn.ModuleList()
        self.src_channels = []
        
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self._make_layers(BottleNeck, 64, self.num_blocks[0], stride=1)
        self._make_layers(BottleNeck, 128, self.num_blocks[1], stride=2)
        self._make_layers(BottleNeck, 256, self.num_blocks[2], stride=2)
        self._make_layers(BottleNeck, 512, self.num_blocks[3], stride=2)
    
    def _make_layers(self, block, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        
        self.layers.append(nn.Sequential(*layers))
        self.src_channels.append(planes * block.expansion)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        out = []
        for layer in self.layers:
            x = layer(x)
            out.append(x)
        
        return out


class Decoder(nn.Module):
    def __init__(self, src_channels):
        super().__init__()
        
        self.output_channels = 256
        self.lat_layers = nn.ModuleList()
        for c in reversed(src_channels):
            self.lat_layers.append(nn.Conv2d(c, self.output_channels, kernel_size=1))
    
    def forward(self, src_features):
        x = self.lat_layers[0](src_features[-1])
        num = len(src_features)
        for i in range(1, num):
            lat_features = self.lat_layers[i](src_features[-i - 1])
            x = F.interpolate(x, scale_factor=2, mode='bilinear') + lat_features
        
        return F.interpolate(x, scale_factor=2, mode='bilinear')


class ResNet(nn.Module):
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
        
        self.encoder = Encoder(self.input_channels)
        self.decoder = Decoder(self.encoder.src_channels)
        
        self.num_rv_features = self.num_class + 1
        self.conv_3x3 = nn.Conv2d(self.decoder.output_channels, self.num_rv_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.add_module('seg_loss_func', loss_utils.WeightedCrossEntropyLoss())
    
    def get_seg_loss(self, batch_dict):
        seg_preds = batch_dict['rv_features'].unsqueeze(0) # (1, N1 + N2 + ..., input_channels), float
        seg_labels = batch_dict['point_labels'].unsqueeze(0) # (1, N1 + N2 + ...), int
        
        batch_size = batch_dict['batch_size']
        positives = seg_labels > 0
        negatives = seg_labels == 0
        seg_weights = (1.0 * positives + 1.0 * negatives).float() # seg_weights consider both positives and negatives
        seg_weights /= torch.clamp(positives.sum(1, keepdim=True).float(), min=1.0)
        
        seg_one_hot_targets = torch.zeros(*list(seg_labels.shape), self.num_class + 1, dtype=seg_labels.dtype, device=seg_labels.device)
        seg_one_hot_targets.scatter_(dim=-1, index=seg_labels.unsqueeze(dim=-1).long(), value=1.0) # (1, N1 + N2 + ..., input_channels)
        
        seg_loss_src = self.seg_loss_func(seg_preds, seg_one_hot_targets, weights=seg_weights) # (1, N1 + N2 + ...)
        seg_loss = seg_loss_src.sum() / batch_size
        seg_loss = seg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['seg_weight']
        
        return seg_loss
    
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
            
            # ~ import matplotlib.pyplot as plt
            # ~ plt.imshow(front_range_image[:3, :, :].permute(1, 2, 0).cpu().numpy())
            # ~ plt.show()
            
            batch_point_us.append(us[:, None])
            batch_point_vs.append(vs[:, None])
            batch_range_images.append(front_range_image)
        
        batch_point_us = torch.cat(batch_point_us, dim=0)
        batch_point_vs = torch.cat(batch_point_vs, dim=0)
        batch_range_images = torch.stack(batch_range_images, dim=0)
        
        src_features = self.encoder(batch_range_images)
        batch_range_images = self.decoder(src_features)
        batch_range_images = self.conv_3x3(batch_range_images)
        
        batch_rv_features = []
        batch_size = batch_points[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            batch_mask = batch_points[:, 0] == batch_idx
            us = batch_point_us[batch_mask, :].squeeze()
            vs = batch_point_vs[batch_mask, :].squeeze()
            range_image = batch_range_images[batch_idx, ...]
            
            # ~ import matplotlib.pyplot as plt
            # ~ import time
            # ~ fig = plt.figure(figsize=(16, 3))
            # ~ fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
            # ~ range_image_sm = F.softmax(range_image.detach(), dim=0)
            # ~ for i in range(0, 4):
                # ~ img = range_image_sm[i:i + 1, :, :].permute(1, 2, 0).cpu().numpy()
                # ~ plt.subplot(2, 2, i + 1)
                # ~ plt.imshow(img)
                # ~ plt.axis('off')
            # ~ plt.show()
            # ~ fig.savefig(time.asctime(time.localtime(time.time())), dpi=200)
            
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
        batch_dict['rv_features'] = batch_rv_features # (N1 + N2 + ..., num_rv_features)
        
        return batch_dict
