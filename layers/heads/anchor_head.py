import numpy as np
import torch
import torch.nn as nn

from utils.common_utils import limit_period
from utils.anchor_utils import AnchorGenerator
from utils.box_coder_utils import ResidualCoder
from utils.box_utils import boxes3d_nearest_bev_iou
from utils.loss_utils import SigmoidFocalClassificationLoss
from utils.loss_utils import WeightedSmoothL1Loss
from utils.loss_utils import WeightedCrossEntropyLoss


class AnchorHead(nn.Module):
    def __init__(self, cfg, in_channels, class_names, grid_size, point_cloud_range):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = len(class_names)
        self.class_names = np.array(class_names)
        self.loss_weights = cfg['loss_weights']
        
        self.anchor_generator = AnchorGenerator(cfg['anchor_list'], point_cloud_range)
        self.anchor_class_names = self.anchor_generator.anchor_class_names
        self.matched_thresholds = self.anchor_generator.matched_thresholds
        self.unmatched_thresholds = self.anchor_generator.unmatched_thresholds

        self.feature_map_stride = cfg['feature_map_stride']
        grid_size_per_class = [grid_size[:2] // self.feature_map_stride for _ in cfg['anchor_list']]

        # all_anchors: list of ndarray, which has the shape of [nz, ny, nx, num_sizes, num_rotations, 7]
        # num_anchors_per_location: list of int
        all_anchors, num_anchors_per_location = self.anchor_generator.generate_anchors(grid_size_per_class)

        self.anchors = [x.cuda() for x in all_anchors]
        self.num_anchors_per_location = sum(num_anchors_per_location)
        
        self.box_coder = ResidualCoder()
        
        self.add_module('cls_loss_func', SigmoidFocalClassificationLoss())
        self.add_module('loc_loss_func', WeightedSmoothL1Loss(code_weights=[1.0] * 7))
        self.add_module('dir_loss_func', WeightedCrossEntropyLoss())
        
        self.conv_cls = nn.Conv2d(in_channels, self.num_anchors_per_location * self.num_classes, kernel_size=1)
        self.conv_box = nn.Conv2d(in_channels, self.num_anchors_per_location * 7, kernel_size=1)
        self.conv_dir_cls = nn.Conv2d(in_channels, self.num_anchors_per_location * 2, kernel_size=1)
        
        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - 0.01) / 0.01))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_cls_loss(self, batch_dict):
        pred_classes = batch_dict['pred_classes']  # [batch_size, H, W, C1], C1 is 6 * 3 by default
        gt_classes = batch_dict['gt_classes_per_anchor']  # [batch_size, num_anchors]
        batch_size = batch_dict['batch_size']
        dtype = pred_classes.dtype
        device = pred_classes.device

        cared = gt_classes >= 0
        positives = gt_classes > 0
        negatives = gt_classes == 0
        weights = (1.0 * positives + 1.0 * negatives).float()
        weights /= torch.clamp(positives.sum(1, keepdim=True).float(), min=1.0)

        pred_classes = pred_classes.view(batch_size, -1, self.num_classes)  # [batch_size, num_anchors, 3]

        gt_classes = gt_classes * cared.type_as(gt_classes)
        one_hot_labels = torch.zeros(*list(gt_classes.shape), self.num_classes + 1, dtype=dtype, device=device)
        one_hot_labels.scatter_(dim=-1, index=gt_classes.unsqueeze(dim=-1).long(), value=1.0)
        one_hot_labels = one_hot_labels[..., 1:]  # [batch_size, num_anchors, 3]
        
        cls_loss = self.cls_loss_func(pred_classes, one_hot_labels, weights=weights)  # [batch_size, num_anchors, 3]
        cls_loss = cls_loss.sum() / batch_size
        cls_loss = cls_loss * self.loss_weights['cls_weight']
        
        return cls_loss

    def get_loc_loss(self, batch_dict):
        pred_boxes = batch_dict['pred_boxes']  # [batch_size, H, W, C2], C2 is 6 * 7 by default
        gt_classes = batch_dict['gt_classes_per_anchor']  # [batch_size, num_anchors]
        gt_boxes = batch_dict['gt_boxes_per_anchor']  # [batch_size, num_anchors, 7]
        batch_size = batch_dict['batch_size']

        positives = gt_classes > 0
        weights = positives.float()
        weights /= torch.clamp(positives.sum(1, keepdim=True).float(), min=1.0)

        pred_boxes = pred_boxes.view(batch_size, -1, 7)  # [batch_size, num_anchors, 7]

        x1 = torch.sin(pred_boxes[..., 6:7]) * torch.cos(gt_boxes[..., 6:7])
        x2 = torch.cos(pred_boxes[..., 6:7]) * torch.sin(gt_boxes[..., 6:7])
        pred_boxes = torch.cat([pred_boxes[..., :6], x1], dim=-1)  # [batch_size, num_anchors, 7]
        gt_boxes = torch.cat([gt_boxes[..., :6], x2], dim=-1)  # [batch_size, num_anchors, 7]

        loc_loss = self.loc_loss_func(pred_boxes, gt_boxes, weights=weights)  # [batch_size, num_anchors, 7]
        loc_loss = loc_loss.sum() / batch_size
        loc_loss = loc_loss * self.loss_weights['loc_weight']

        return loc_loss

    def get_dir_loss(self, batch_dict):
        pred_dir_classes = batch_dict['pred_dir_classes']  # [batch_size, H, W, C3], C3 is 6 * 2 by default
        gt_classes = batch_dict['gt_classes_per_anchor']  # [batch_size, num_anchors]
        gt_boxes = batch_dict['gt_boxes_per_anchor']  # [batch_size, num_anchors, 7]
        batch_size = batch_dict['batch_size']
        dtype = pred_dir_classes.dtype
        device = pred_dir_classes.device

        positives = gt_classes > 0
        weights = positives.float()
        weights /= torch.clamp(positives.sum(1, keepdim=True).float(), min=1.0)

        pred_dir_classes = pred_dir_classes.view(batch_size, -1, 2)  # [batch_size, num_anchors, 2]

        anchors = torch.cat(self.anchors, dim=-3)  # [nz, ny, nx, num_sizes * num_classes, num_rotations, 7]
        anchors = anchors.view(1, -1, 7).repeat(batch_size, 1, 1)  # [batch_size, num_anchors, 7]
        rot_offsets = limit_period(gt_boxes[..., 6] + anchors[..., 6] - np.pi / 4, 0, 2 * np.pi)
        gt_dir_classes = torch.clamp(torch.floor(rot_offsets / np.pi).long(), min=0, max=1)
        one_hot_labels = torch.zeros(*list(gt_dir_classes.shape), 2, dtype=dtype, device=device)
        one_hot_labels.scatter_(dim=-1, index=gt_dir_classes.unsqueeze(dim=-1).long(), value=1.0)  # [batch_size, num_anchors, 2]
        
        dir_loss = self.dir_loss_func(pred_dir_classes, one_hot_labels, weights=weights)  # [batch_size, num_anchors]
        dir_loss = dir_loss.sum() / batch_size
        dir_loss = dir_loss * self.loss_weights['dir_weight']

        return dir_loss

    def assign_targets(self, anchors, gt_boxes, gt_classes, matched_threshold, unmatched_threshold):
        num_anchors = anchors.shape[0]
        num_objects = gt_boxes.shape[0]
        device = anchors.device
        classes = torch.ones((num_anchors,), dtype=torch.int32, device=device) * -1

        if num_objects > 0:
            anchor_by_gt_overlap = boxes3d_nearest_bev_iou(anchors, gt_boxes)  # [num_anchors, num_objects]

            anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
            anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_anchors, device=device), anchor_to_gt_argmax]
            gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_objects, device=device)]

            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1

            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            classes[anchors_with_max_overlap] = gt_classes[gt_inds_force]

            pos_inds = anchor_to_gt_max >= matched_threshold
            classes[pos_inds] = gt_classes[anchor_to_gt_argmax[pos_inds]]
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]

        else:
            bg_inds = torch.arange(num_anchors, device=device)

        fg_inds = (classes > 0).nonzero()[:, 0]

        if num_objects == 0:
            classes[:] = 0
        else:
            classes[bg_inds] = 0
            classes[anchors_with_max_overlap] = gt_classes[gt_inds_force]

        boxes = anchors.new_zeros((num_anchors, 7))
        if num_objects > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            fg_anchors = anchors[fg_inds, :]
            boxes[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)

        ret_dict = {
            'classes': classes,  # [num_anchors]
            'boxes': boxes,  # [num_anchors, 7]
        }

        return ret_dict

    def forward(self, batch_dict):
        batch_bev_features = batch_dict['bev_features']  # [batch_size, num_bev_features, ny, nx]
        batch_size = batch_dict['batch_size']
        
        x = batch_bev_features
        x1 = self.conv_cls(x).permute(0, 2, 3, 1).contiguous()  # [batch_size, H, W, C1], C1 is 6 * 3 by default
        x2 = self.conv_box(x).permute(0, 2, 3, 1).contiguous()  # [batch_size, H, W, C2], C2 is 6 * 7 by default
        x3 = self.conv_dir_cls(x).permute(0, 2, 3, 1).contiguous()  # [batch_size, H, W, C3], C3 is 6 * 2 by default
        
        if self.training:
            batch_dict['pred_classes'] = x1
            batch_dict['pred_boxes'] = x2
            batch_dict['pred_dir_classes'] = x3

            gt_classes = batch_dict['gt_boxes'][:, :, -1]  # [batch_size, M]
            gt_boxes = batch_dict['gt_boxes'][:, :, :-1]  # [batch_size, M, 7]
            gt_classes_per_anchor = []
            gt_boxes_per_anchor = []

            for batch_idx in range(batch_size):
                classes = gt_classes[batch_idx]  # [M]
                boxes = gt_boxes[batch_idx]  # [M, 7]

                cnt = boxes.__len__() - 1
                while cnt > 0 and boxes[cnt].sum() == 0:
                    cnt -= 1
                classes = classes[:cnt + 1].int()
                boxes = boxes[:cnt + 1]
                
                assignments = []
                for name, anchors in zip(self.anchor_class_names, self.anchors):
                    anchors = anchors.view(-1, 7)
                    mask = torch.tensor([self.class_names[c - 1] == name for c in classes], dtype=torch.bool)
                    assignment = self.assign_targets(
                        anchors, boxes[mask], classes[mask], self.matched_thresholds[name], self.unmatched_thresholds[name]
                    )
                    assignments.append(assignment)
                
                map_size = self.anchors[0].shape[:3]  # [nz, ny, nx]

                temp_classes = [t['classes'].view(*map_size, -1) for t in assignments]
                temp_boxes = [t['boxes'].view(*map_size, -1, 7) for t in assignments]
                temp_classes = torch.cat(temp_classes, dim=-1).view(-1)
                temp_boxes = torch.cat(temp_boxes, dim=-2).view(-1, 7)
                
                gt_classes_per_anchor.append(temp_classes)
                gt_boxes_per_anchor.append(temp_boxes)
            
            batch_dict['gt_classes_per_anchor'] = torch.stack(gt_classes_per_anchor, dim=0)  # [batch_size, num_anchors]
            batch_dict['gt_boxes_per_anchor'] = torch.stack(gt_boxes_per_anchor, dim=0)  # [batch_size, num_anchors, 7]

        else:
            anchors = torch.cat(self.anchors, dim=-3)  # [nz, ny, nx, num_sizes * num_classes, num_rotations, 7]
            anchors = anchors.view(-1, 7)
            num_anchors = anchors.shape[0]
            batch_anchors = anchors.view(1, -1, 7).repeat(batch_size, 1, 1)
            
            x1 = x1.view(batch_size, num_anchors, -1).float()
            x2 = x2.view(batch_size, num_anchors, -1)
            x3 = x3.view(batch_size, num_anchors, -1)

            boxes = self.box_coder.decode_torch(x2, batch_anchors)
            dir_classes = torch.max(x3, dim=-1)[1]
            boxes[..., 6] = limit_period(boxes[..., 6] - np.pi / 4, 0.0, np.pi)
            boxes[..., 6] = boxes[..., 6] + np.pi / 4 + np.pi * dir_classes.to(boxes.dtype)
            
            batch_dict['pred_classes'] = x1  # [batch_size, num_anchors, num_classes]
            batch_dict['pred_boxes'] = boxes  # [batch_size, num_anchors, 7]

        return batch_dict
