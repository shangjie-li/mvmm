import pickle

import os
import copy
import numpy as np
import SharedArray
import torch.distributed as dist

from ops.iou3d_nms import iou3d_nms_utils
from utils import box_utils


class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, num_point_features, logger=None):
        self.root_path = root_path
        self.sampler_cfg = sampler_cfg
        self.class_names = class_names
        self.num_point_features = num_point_features
        self.logger = logger
        
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []
        
        db_info_path = self.root_path.resolve() / sampler_cfg.DB_INFO_PATH
        with open(str(db_info_path), 'rb') as f:
            infos = pickle.load(f)
            [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val) # filter the db_infos by min_points and difficulty

        self.sample_groups = {}
        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer + sample_num >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(sampled_boxes, road_planes, calib):
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(sampled_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = sampled_boxes[:, 2] - sampled_boxes[:, 5] / 2 - cur_lidar_height
        sampled_boxes[:, 2] -= mv_height # lidar view
        return sampled_boxes, mv_height

    def add_sampled_boxes_to_scene(self, data_dict, sampled_boxes, total_valid_sampled_dict):
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names']
        points = data_dict['points']
        
        sampled_boxes, mv_height = self.put_boxes_on_road_planes(
            sampled_boxes, data_dict['road_plane'], data_dict['calib']
        ) # adjust the height of sampled_boxes
        sampled_names = np.array([x['name'] for x in total_valid_sampled_dict])
        
        obj_points_list = []
        for idx, info in enumerate(total_valid_sampled_dict):
            file_path = self.root_path / info['path']
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, self.num_point_features)
            obj_points[:, :3] += info['box3d_lidar'][:3]
            obj_points[:, 2] -= mv_height[idx] # adjust the height of obj_points
            #~ points_new = (obj_points[:-1] + obj_points[1:]) / 2
            #~ obj_points = np.concatenate([obj_points, points_new], axis=0)
            obj_points_list.append(obj_points)
        obj_points = np.concatenate(obj_points_list, axis=0)
        
        enlarged_sampled_boxes = box_utils.enlarge_box3d(
            sampled_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        ) # adjust the [l, w, h] of sampled_boxes for box_utils.remove_points_in_boxes3d
        points = box_utils.remove_points_in_boxes3d(points, enlarged_sampled_boxes)
        points = np.concatenate([obj_points, points], axis=0)
        
        gt_names = np.concatenate([gt_names, sampled_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_boxes], axis=0)
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        return data_dict

    def __call__(self, data_dict):
        gt_boxes = data_dict['gt_boxes'] # (num_gt, 7)
        gt_names = data_dict['gt_names'].astype(str)
        
        existed_boxes = gt_boxes # for adding dynamically
        total_valid_sampled_dict = [] # for adding dynamically
        
        for class_name, sample_group in self.sample_groups.items():
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)
                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32) # (num_sampled, 7)

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask] # (num_sampled')
                valid_sampled_boxes = sampled_boxes[valid_mask] # (num_sampled', 7)

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        total_valid_sampled_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        if total_valid_sampled_dict.__len__() > 0:
            data_dict = self.add_sampled_boxes_to_scene(data_dict, total_valid_sampled_boxes, total_valid_sampled_dict)

        return data_dict
