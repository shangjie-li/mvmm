import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class DualBaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        
        if self.model_cfg.get('NUM_LAYERS', None) is not None:
            assert len(self.model_cfg.NUM_LAYERS) == len(self.model_cfg.DOWNSAMPLE_STRIDES) == len(self.model_cfg.DOWNSAMPLE_FILTERS)
            num_layers = self.model_cfg.NUM_LAYERS
            downsample_strides = self.model_cfg.DOWNSAMPLE_STRIDES
            downsample_filters = self.model_cfg.DOWNSAMPLE_FILTERS
        else:
            num_layers = downsample_strides = downsample_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.UPSAMPLE_FILTERS)
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
            upsample_filters = self.model_cfg.UPSAMPLE_FILTERS
        else:
            upsample_strides = upsample_filters = []
        
        self.channel_split = self.model_cfg.CHANNEL_SPLIT
        assert self.channel_split[0] + self.channel_split[1] == self.input_channels
        
        num_levels = len(num_layers)
        c_in_list1 = [self.channel_split[0], *downsample_filters[:-1]]
        c_in_list2 = [self.channel_split[1], *downsample_filters[:-1]]
        self.blocks1 = nn.ModuleList()
        self.blocks2 = nn.ModuleList()
        self.deblocks1 = nn.ModuleList()
        self.deblocks2 = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers1 = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list1[idx], downsample_filters[idx], kernel_size=3,
                    stride=downsample_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(downsample_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            cur_layers2 = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list2[idx], downsample_filters[idx], kernel_size=3,
                    stride=downsample_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(downsample_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(num_layers[idx]):
                cur_layers1.extend([
                    nn.Conv2d(downsample_filters[idx], downsample_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(downsample_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            for k in range(num_layers[idx]):
                cur_layers2.extend([
                    nn.Conv2d(downsample_filters[idx], downsample_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(downsample_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks1.append(nn.Sequential(*cur_layers1))
            self.blocks2.append(nn.Sequential(*cur_layers2))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks1.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            downsample_filters[idx], upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.deblocks2.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            downsample_filters[idx], upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks1.append(nn.Sequential(
                        nn.Conv2d(
                            downsample_filters[idx], upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.deblocks2.append(nn.Sequential(
                        nn.Conv2d(
                            downsample_filters[idx], upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks1.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
            self.deblocks2.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        
        # self.conv_1x1 = nn.Conv2d(2 * self.num_bev_features, self.num_bev_features, kernel_size=1, bias=False)
        self.fusion_layers = AttentionalFusionModule(self.num_bev_features)
    
    def forward(self, batch_dict, **kwargs):
        batch_pv_features = batch_dict['pv_features'] # (B, input_channels, ny, nx)
        x1 = batch_pv_features[:, :self.channel_split[0], :, :]
        x2 = batch_pv_features[:, self.channel_split[0]:, :, :]
        
        batch_bev_features1 = []
        batch_bev_features2 = []
        for i in range(len(self.blocks1)):
            x1 = self.blocks1[i](x1)
            x2 = self.blocks2[i](x2)
            if len(self.deblocks1) > 0:
                batch_bev_features1.append(self.deblocks1[i](x1))
                batch_bev_features2.append(self.deblocks2[i](x2))
            else:
                batch_bev_features1.append(x1)
                batch_bev_features2.append(x2)

        if len(batch_bev_features1) > 1:
            batch_bev_features1 = torch.cat(batch_bev_features1, dim=1)
            batch_bev_features2 = torch.cat(batch_bev_features2, dim=1)
        elif len(batch_bev_features1) == 1:
            batch_bev_features1 = batch_bev_features1[0]
            batch_bev_features2 = batch_bev_features2[0]

        if len(self.deblocks1) > len(self.blocks1):
            batch_bev_features1 = self.deblocks1[-1](batch_bev_features1)
            batch_bev_features2 = self.deblocks2[-1](batch_bev_features2)
        
        # batch_bev_features = batch_bev_features1 + batch_bev_features2
        # batch_bev_features = self.conv_1x1(torch.cat([batch_bev_features1, batch_bev_features2], dim=1))
        batch_bev_features = self.fusion_layers(batch_bev_features1, batch_bev_features2)
        
        batch_dict['bev_features'] = batch_bev_features
        
        return batch_dict
