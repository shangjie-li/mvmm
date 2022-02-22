import numpy as np
import torch
import torch.nn as nn

from ops.iou3d_nms import iou3d_nms_utils
from utils import box_coder_utils, common_utils, loss_utils, box_utils


class AnchorGenerator(object):
    def __init__(self, anchor_generator_cfg, anchor_range):
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_cfg
        self.anchor_range = anchor_range
        
        self.anchor_sizes = [config['anchor_sizes'] for config in self.anchor_generator_cfg]
        self.anchor_rotations = [config['anchor_rotations'] for config in self.anchor_generator_cfg]
        self.anchor_heights = [config['anchor_bottom_heights'] for config in self.anchor_generator_cfg]
        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        
        self.anchor_class_names = [config['class_name'] for config in self.anchor_generator_cfg]
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for config in self.anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']

    def generate_anchors(self, grid_sizes):
        all_anchors = []
        num_anchors_per_location = []
        for grid_size, anchor_size, anchor_rotation, anchor_height in zip(
                grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights):

            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
            y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
            x_offset, y_offset = 0, 0

            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            z_shifts = x_shifts.new_tensor(anchor_height)

            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ]) # [x_grid, y_grid, z_grid]
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1) # [x, y, z, 3]
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])
            anchors = torch.cat((anchors, anchor_rotation), dim=-1) # [x, y, z, num_size, num_rot, 7]

            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous() # [z, y, x, num_size, num_rot, 7]
            anchors[..., 2] += anchors[..., 5] / 2 # shift to box centers
            all_anchors.append(anchors)
            
        return all_anchors, num_anchors_per_location


class AnchorHeadSingle(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.num_class = num_class
        self.class_names = np.array(class_names)
        
        self.anchor_generator = AnchorGenerator(self.model_cfg.ANCHOR_GENERATOR_CONFIG, point_cloud_range)
        self.anchor_class_names = self.anchor_generator.anchor_class_names
        self.matched_thresholds = self.anchor_generator.matched_thresholds
        self.unmatched_thresholds = self.anchor_generator.unmatched_thresholds
        
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in self.model_cfg.ANCHOR_GENERATOR_CONFIG]
        all_anchors, num_anchors_per_location = self.anchor_generator.generate_anchors(feature_map_size)
        self.anchors = [x.cuda() for x in all_anchors]
        self.num_anchors_per_location = sum(num_anchors_per_location)
        
        self.box_coder = box_coder_utils.ResidualCoder(code_size=7)
        
        self.add_module('cls_loss_func', loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0))
        self.add_module('box_loss_func', loss_utils.WeightedSmoothL1Loss(code_weights=self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']))
        self.add_module('dir_cls_loss_func', loss_utils.WeightedCrossEntropyLoss())
        
        self.conv_cls = nn.Conv2d(input_channels, self.num_anchors_per_location * self.num_class, kernel_size=1)
        self.conv_box = nn.Conv2d(input_channels, self.num_anchors_per_location * self.box_coder.code_size, kernel_size=1)
        self.conv_dir_cls = nn.Conv2d(input_channels, self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_cls_loss(self, batch_dict):
        cls_preds = batch_dict['cls_preds_for_training']
        cls_labels = batch_dict['cls_labels']
        
        batch_size = int(cls_preds.shape[0])
        cared = cls_labels >= 0 # [N, num_anchors]
        positives = cls_labels > 0
        negatives = cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = cls_labels * cared.type_as(cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(*list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device)
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        
        return cls_loss

    def get_box_loss(self, batch_dict):
        def add_sin_difference(boxes1, boxes2):
            rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(boxes2[..., 6:7])
            rad_gt_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[..., 6:7])
            boxes1 = torch.cat([boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
            boxes2 = torch.cat([boxes2[..., :6], rad_gt_encoding, boxes2[..., 7:]], dim=-1)
            return boxes1, boxes2
        
        def get_direction_target(anchors, reg_targets, dir_offset=0, num_bins=2):
            batch_size = reg_targets.shape[0]
            anchors = anchors.view(batch_size, -1, anchors.shape[-1])
            rot_gt = reg_targets[..., 6] + anchors[..., 6]
            offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
            dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
            dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype, device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
            return dir_cls_targets
        
        box_preds = batch_dict['box_preds_for_training']
        dir_cls_preds = batch_dict['dir_cls_preds_for_training']
        
        cls_labels = batch_dict['cls_labels']
        box_targets = batch_dict['box_targets']
        
        batch_size = int(box_preds.shape[0])
        positives = cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        anchors = torch.cat(self.anchors, dim=-3) # [z, y, x, num_size * num_class, num_rot, 7]
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        
        box_preds = box_preds.view(batch_size, -1, box_preds.shape[-1] // self.num_anchors_per_location)
        box_preds_sin, reg_targets_sin = add_sin_difference(box_preds, box_targets) # sin(a - b) = sinacosb - cosasinb
        
        loc_loss_src = self.box_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights) # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size
        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

        dir_targets = get_direction_target(anchors, box_targets, self.model_cfg.DIR_OFFSET, self.model_cfg.NUM_DIR_BINS)
        dir_logits = dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
        weights = positives.type_as(dir_logits)
        weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
        
        dir_loss_src = self.dir_cls_loss_func(dir_logits, dir_targets, weights=weights)
        dir_loss = dir_loss_src.sum() / batch_size
        dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']

        return loc_loss, dir_loss

    def assign_targets(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45):
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7])

            anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
            anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]

            gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1

            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()

            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        fg_inds = (labels > 0).nonzero()[:, 0]

        if len(gt_boxes) == 0 or anchors.shape[0] == 0:
            labels[:] = 0
        else:
            labels[bg_inds] = 0
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

        box_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            fg_anchors = anchors[fg_inds, :]
            box_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)

        ret_dict = {
            'cls_labels': labels, # [num_anchors]
            'box_targets': box_targets, # [num_anchors, 7]
        }
        
        return ret_dict

    def forward(self, batch_dict):
        batch_bev_features = batch_dict['bev_features']
        batch_size = batch_dict['batch_size']

        cls_preds = self.conv_cls(batch_bev_features)
        box_preds = self.conv_box(batch_bev_features)
        dir_cls_preds = self.conv_dir_cls(batch_bev_features)
        
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous() # [B, H, W, C1]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous() # [B, H, W, C2]
        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous() # [B, H, W, C3]

        if self.training:
            all_cls_labels = []
            all_box_targets = []
            gt_classes = batch_dict['gt_boxes'][:, :, -1] # [B, M]
            gt_boxes = batch_dict['gt_boxes'][:, :, :-1] # [B, M, 7]
            
            for batch_idx in range(batch_size):
                cur_gt_classes = gt_classes[batch_idx] # [M]
                cur_gt_boxes = gt_boxes[batch_idx] # [M, 7]
                cnt = cur_gt_boxes.__len__() - 1
                while cnt > 0 and cur_gt_boxes[cnt].sum() == 0:
                    cnt -= 1
                cur_gt_classes = cur_gt_classes[:cnt + 1].int()
                cur_gt_boxes = cur_gt_boxes[:cnt + 1]
                
                assignment_list = []
                for anchor_class_name, anchors in zip(self.anchor_class_names, self.anchors):
                    if cur_gt_classes.shape[0] > 1:
                        mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)
                    else:
                        mask = torch.tensor([self.class_names[c - 1] == anchor_class_name for c in cur_gt_classes], dtype=torch.bool)
                    
                    assignment = self.assign_targets(
                        anchors.view(-1, anchors.shape[-1]), cur_gt_boxes[mask], cur_gt_classes[mask],
                        self.matched_thresholds[anchor_class_name], self.unmatched_thresholds[anchor_class_name]
                    )
                    assignment_list.append(assignment)
                
                map_size = self.anchors[0].shape[:3]
                
                temp_cls_labels = [t['cls_labels'].view(*map_size, -1) for t in assignment_list]
                temp_box_targets = [t['box_targets'].view(*map_size, -1, self.box_coder.code_size) for t in assignment_list]
                temp_cls_labels = torch.cat(temp_cls_labels, dim=-1).view(-1)
                temp_box_targets = torch.cat(temp_box_targets, dim=-2).view(-1, self.box_coder.code_size)
                
                all_cls_labels.append(temp_cls_labels)
                all_box_targets.append(temp_box_targets)
            
            batch_dict['cls_labels'] = torch.stack(all_cls_labels, dim=0) # [B, num_anchors * num_class]
            batch_dict['box_targets'] = torch.stack(all_box_targets, dim=0) # [B, num_anchors * num_class, 7]
            
            batch_dict['cls_preds_for_training'] = cls_preds # [B, H, W, C1]
            batch_dict['box_preds_for_training'] = box_preds # [B, H, W, C2]
            batch_dict['dir_cls_preds_for_training'] = dir_cls_preds # [B, H, W, C3]
            
        else:
            anchors = torch.cat(self.anchors, dim=-3) # [z, y, x, num_size * num_class, num_rot, 7]
            num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
            batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
            
            cls_preds = cls_preds.view(batch_size, num_anchors, -1).float()
            box_preds = box_preds.view(batch_size, num_anchors, -1)
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1)
            
            box_preds = self.box_coder.decode_torch(box_preds, batch_anchors)
            dir_cls_labels = torch.max(dir_cls_preds, dim=-1)[1]
            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            rotation = common_utils.limit_period(box_preds[..., 6] - self.model_cfg.DIR_OFFSET, self.model_cfg.DIR_LIMIT_OFFSET, period)
            box_preds[..., 6] = rotation + self.model_cfg.DIR_OFFSET + period * dir_cls_labels.to(box_preds.dtype)
            
            batch_dict['cls_preds_for_testing'] = cls_preds # [B, num_boxes, num_classes]
            batch_dict['box_preds_for_testing'] = box_preds # [B, num_boxes, 7 + C]

        return batch_dict
