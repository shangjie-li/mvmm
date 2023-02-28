import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        
        self.num_blocks = [3, 4, 6, 3]
        self.in_planes = 64
        
        self.layers = nn.ModuleList()
        self.src_channels = []
        
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
        
        self.out_channels = 64
        self.lat_layers = nn.ModuleList()
        for c in reversed(src_channels):
            self.lat_layers.append(nn.Conv2d(c, self.out_channels, kernel_size=1))
    
    def forward(self, src_features):
        x = self.lat_layers[0](src_features[-1])
        num = len(src_features)
        for i in range(1, num):
            lat_features = self.lat_layers[i](src_features[-i - 1])
            x = F.interpolate(x, scale_factor=2, mode='bilinear') + lat_features
        
        return F.interpolate(x, scale_factor=2, mode='bilinear')


class ResNet(nn.Module):
    def __init__(self, cfg, range_convertor):
        super().__init__()

        self.in_channels = 7  # (x, y, z, i, r, g, b)
        self.range_convertor = range_convertor
        
        self.encoder = Encoder(self.in_channels)
        self.decoder = Decoder(self.encoder.src_channels)

        assert len(cfg['filters']) == 1
        self.num_rv_features = cfg['filters'][-1]

        self.conv_1x1 = nn.Conv2d(self.decoder.out_channels, self.num_rv_features, kernel_size=1)
    
    def forward(self, batch_dict):
        batch_points = batch_dict['colored_points']  # [N1 + N2 + ..., 8], (batch_id, x, y, z, i, r, g, b)
        batch_range_image = batch_dict['range_image']   # [batch_size, 7, 48, 512]
        batch_size = batch_dict['batch_size']
        
        x = self.decoder(self.encoder(batch_range_image))
        x = self.conv_1x1(x)
        
        batch_rv_features = []
        for batch_idx in range(batch_size):
            mask = batch_points[:, 0] == batch_idx
            points = batch_points[mask]
            range_image = x[batch_idx, ...]
            range_features = self.range_convertor.get_range_features(points[:, 1:4], range_image)
            batch_rv_features.append(range_features)
        
        batch_rv_features = torch.cat(batch_rv_features, dim=0)
        batch_dict['rv_features'] = batch_rv_features  # [N1 + N2 + ..., num_rv_features]
        
        return batch_dict
