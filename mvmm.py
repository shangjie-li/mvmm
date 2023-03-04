import torch
import torch.nn as nn

from layers import rv_backbones
from layers import pv_bridges
from layers import bev_backbones
from layers import heads
from utils.nms_utils import nms


def build_model(cfg, dataset):
    if cfg['type'] == 'MVMM':
        return MVMM(cfg=cfg, dataset=dataset)
    else:
        raise NotImplementedError


class MVMM(nn.Module):
    def __init__(self, cfg, dataset):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.module_list = []

        name = 'rv_backbone'
        if cfg.get(name) is None:
            self.num_rv_features = 0
        else:
            rv_backbone = rv_backbones.__all__[cfg[name]['type']](
                cfg=cfg[name],
                range_convertor=self.dataset.range_convertor,
            )
            self.num_rv_features = rv_backbone.num_rv_features
            self.add_module(name, rv_backbone)
            self.module_list.append(rv_backbone)

        name = 'pv_bridge'
        pv_bridge = pv_bridges.__all__[cfg[name]['type']](
            cfg=cfg[name],
            in_channels=self.num_rv_features,
            point_cloud_range=self.dataset.point_cloud_range,
        )
        self.num_pv_features = pv_bridge.num_pv_features
        self.add_module(name, pv_bridge)
        self.module_list.append(pv_bridge)

        name = 'bev_backbone'
        bev_backbone = bev_backbones.__all__[cfg[name]['type']](
            cfg=cfg[name],
            in_channels=self.num_pv_features,
        )
        self.num_bev_features = bev_backbone.num_bev_features
        self.add_module(name, bev_backbone)
        self.module_list.append(bev_backbone)

        name = 'head'
        head = heads.__all__[cfg[name]['type']](
            cfg=cfg[name],
            in_channels=self.num_bev_features,
            class_names=self.dataset.class_names,
            grid_size=self.pv_bridge.grid_size,
            point_cloud_range=self.dataset.point_cloud_range,
        )
        self.add_module(name, head)
        self.module_list.append(head)

    def forward(self, batch_dict, score_thresh=0.1, nms_thresh=0.1):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            cls_loss = self.head.get_cls_loss(batch_dict)
            loc_loss = self.head.get_loc_loss(batch_dict)
            dir_loss = self.head.get_dir_loss(batch_dict)

            total_loss = cls_loss + loc_loss + dir_loss
            stats_dict = {
                'cls': cls_loss.item(),
                'loc': loc_loss.item(),
                'dir': dir_loss.item(),
            }

            return total_loss, stats_dict

        else:
            batch_size = batch_dict['batch_size']
            batch_boxes, batch_scores, batch_classes = [], [], []

            for batch_idx in range(batch_size):
                classes = batch_dict['pred_classes'][batch_idx]  # [num_anchors, num_classes]
                boxes = batch_dict['pred_boxes'][batch_idx]  # [num_anchors, 7]

                scores, classes = torch.max(torch.sigmoid(classes), dim=-1)
                classes += 1
                selected, selected_scores = nms(scores, boxes, score_thresh=score_thresh, nms_thresh=nms_thresh)

                batch_boxes.append(boxes[selected])
                batch_scores.append(selected_scores)
                batch_classes.append(classes[selected])

            batch_dict['pred_boxes'] = batch_boxes  # list of tensor, which has the shape of [M, 7]
            batch_dict['pred_scores'] = batch_scores  # list of tensor, which has the shape of [M]
            batch_dict['pred_classes'] = batch_classes  # list of tensor, which has the shape of [M]

            return batch_dict
