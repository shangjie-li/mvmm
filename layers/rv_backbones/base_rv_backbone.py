import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseRVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        
        self.num_rv_features = self.num_filters[-1]
    
    def forward(self, batch_dict, **kwargs):
        batch_point_features = batch_dict['point_features']
        
        x = batch_point_features
        batch_rv_features = x
        
        batch_dict['rv_features'] = batch_rv_features
        
        return batch_dict
