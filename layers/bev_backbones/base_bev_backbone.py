import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseBEVBackbone(nn.Module):
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
        
        num_levels = len(num_layers)
        c_in_list = [self.input_channels, *downsample_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], downsample_filters[idx], kernel_size=3,
                    stride=downsample_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(downsample_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(num_layers[idx]):
                cur_layers.extend([
                    nn.Conv2d(downsample_filters[idx], downsample_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(downsample_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
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
                    self.deblocks.append(nn.Sequential(
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
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
    
    def forward(self, batch_dict, **kwargs):
        batch_pv_features = batch_dict['pv_features']
        
        batch_bev_features = []
        x = batch_pv_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if len(self.deblocks) > 0:
                batch_bev_features.append(self.deblocks[i](x))
            else:
                batch_bev_features.append(x)

        if len(batch_bev_features) > 1:
            batch_bev_features = torch.cat(batch_bev_features, dim=1)
        elif len(batch_bev_features) == 1:
            batch_bev_features = batch_bev_features[0]

        if len(self.deblocks) > len(self.blocks):
            batch_bev_features = self.deblocks[-1](batch_bev_features)
        
        batch_dict['bev_features'] = batch_bev_features
        
        return batch_dict
