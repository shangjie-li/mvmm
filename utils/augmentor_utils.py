import os
import copy
import numpy as np
import SharedArray
import torch.distributed as dist
import torch
import pickle
from functools import partial

from . import common_utils, box_utils
from ops.iou3d_nms import iou3d_nms_utils
from ops.roiaware_pool3d import roiaware_pool3d_utils


def random_flip_along_x(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes, points, enable


def random_flip_along_y(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points, enable


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points, noise_rotation


def global_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scaling = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scaling
    gt_boxes[:, :6] *= noise_scaling
    return gt_boxes, points, noise_scaling


class DatabaseSampler(object):
    def __init__(self, sampler_cfg, root_path, class_names, src_point_features,
        range_convertor, logger=None):
        self.sampler_cfg = sampler_cfg
        self.root_path = root_path
        self.class_names = class_names
        self.src_point_features = src_point_features
        self.range_convertor = range_convertor
        self.logger = logger
        
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []
        
        db_info_path = root_path.resolve() / sampler_cfg.DB_INFO_PATH
        with open(str(db_info_path), 'rb') as f:
            infos = pickle.load(f)
            [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]
        
        self.gt_points_thresh = {}
        for x in sampler_cfg.FILTER_BY_MIN_POINTS:
            class_name, min_points = x.split(':')
            if class_name not in class_names:
                continue
            self.gt_points_thresh[class_name] = int(min_points)
        self.db_infos = self.filter_by_min_points(self.db_infos, self.gt_points_thresh)
        
        self.removed_difficulty = sampler_cfg.FILTER_BY_DIFFICULTY
        self.db_infos = self.filter_by_difficulty(self.db_infos, self.removed_difficulty)
        
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
    
    def filter_by_min_points(self, db_infos, gt_points_thresh):
        for class_name, thresh in gt_points_thresh.items():
            if thresh > 0 and class_name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[class_name]:
                    if info['num_points_in_gt'] >= thresh:
                        filtered_infos.append(info)
                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' % \
                        (class_name, len(db_infos[class_name]), len(filtered_infos)))
                db_infos[class_name] = filtered_infos
        return db_infos
    
    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info(
                    'Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key]))
                )
        return new_db_infos
    
    def sample_with_fixed_number(self, class_name, sample_group, random_sampling=True):
        sample_num, pointer, indices = \
            int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer + sample_num >= len(self.db_infos[class_name]):
            if random_sampling:
                pointer, indices = 0, np.random.permutation(len(self.db_infos[class_name]))
            else:
                pointer, indices = 0, np.arange(len(self.db_infos[class_name]))
        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer:pointer + sample_num]]
        sample_group['pointer'], sample_group['indices'] = pointer + sample_num, indices
        return sampled_dict
    
    @staticmethod
    def put_boxes_on_road_plane(sampled_boxes, road_plane, calib):
        a, b, c, d = road_plane
        center_cam = calib.lidar_to_rect(sampled_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = sampled_boxes[:, 2] - sampled_boxes[:, 5] / 2 - cur_lidar_height
        sampled_boxes[:, 2] -= mv_height # lidar view
        return sampled_boxes, mv_height
    
    def remove_occluded_points(self, data_dict, sampled_boxes, sampled_names, sampled_obj_points):
        gt_boxes = data_dict['gt_boxes'] # (num_gt, 7)
        gt_names = data_dict['gt_names'] # (num_gt)
        all_boxes = np.concatenate([sampled_boxes, gt_boxes.reshape(-1, 7)], axis=0)
        all_names = np.concatenate([sampled_names, gt_names.reshape(-1)], axis=0)
        points = data_dict['colored_points']
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
        ).numpy() # (nboxes, npoints)
        all_obj_points = sampled_obj_points
        for i in range(gt_boxes.shape[0]):
            all_obj_points.append(points[point_indices[i] > 0])
        back_points = box_utils.remove_points_in_boxes3d(points, gt_boxes)
        back_us, back_vs, _, _ = self.range_convertor.get_pixel_coords(back_points)
        
        all_rv_boxes = self.range_convertor.get_range_boxes(all_boxes)
        all_distances = np.sqrt(all_boxes[:, 0] ** 2 + all_boxes[:, 1] ** 2)
        keep_indices = []
        for i in reversed(np.argsort(all_distances)):
            x1, y1, x2, y2 = all_rv_boxes[i].squeeze()
            mask = (back_us > x1) & (back_vs > y1) & (back_us < x2) & (back_vs < y2)
            pp = back_points[mask]
            d = np.sqrt(pp[:, 0] ** 2 + pp[:, 1] ** 2).mean() if pp.shape[0] > 0 else np.inf
            if all_distances[i] < d:
                keep_indices.append(i)
        all_boxes = all_boxes[keep_indices]
        all_names = all_names[keep_indices]
        all_obj_points = [all_obj_points[i] for i in keep_indices]
        
        rv_indicator = self.range_convertor.get_range_indicator().numpy()
        for i in range(all_boxes.shape[0]):
            if all_obj_points[i].shape[0] > 0:
                us, vs, _, _ = self.range_convertor.get_pixel_coords(all_obj_points[i])
                rv_indicator[vs, us] = i + 1
        back_points = back_points[rv_indicator[back_vs, back_us] == 0]
        for i in range(all_boxes.shape[0]):
            if all_obj_points[i].shape[0] > 0:
                us, vs, _, _ = self.range_convertor.get_pixel_coords(all_obj_points[i])
                all_obj_points[i] = all_obj_points[i][rv_indicator[vs, us] == i + 1]
        
        keep_indices = []
        for i in range(all_boxes.shape[0]):
            thresh = self.gt_points_thresh[all_names[i]]
            if all_obj_points[i].shape[0] >= thresh:
                keep_indices.append(i)
        all_boxes = all_boxes[keep_indices]
        all_names = all_names[keep_indices]
        all_obj_points = [all_obj_points[i] for i in keep_indices]
        
        data_dict['gt_boxes'] = all_boxes
        data_dict['gt_names'] = all_names
        data_dict['colored_points'] = np.concatenate([back_points, *all_obj_points], axis=0)
        return data_dict
    
    def __call__(self, data_dict):
        gt_boxes = data_dict['gt_boxes'] # (num_gt, 7)
        gt_names = data_dict['gt_names'] # (num_gt)
        
        all_boxes = gt_boxes # for adding dynamically
        all_names = gt_names # for adding dynamically
        total_sampled_obj_points = [] # for adding dynamically
        
        for class_name, sample_group in self.sample_groups.items():
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(
                    class_name, sample_group, self.sampler_cfg.RANDOM_SAMPLING
                )
                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0) # (sample_num, 7)
                sampled_names = np.array([x['name'] for x in sampled_dict]) # (sample_num)
                sampled_obj_points = []
                for idx, info in enumerate(sampled_dict):
                    file_path = self.root_path / info['path']
                    pts = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, self.src_point_features)
                    pts[:, :3] += info['box3d_lidar'][:3] # move the points to its box from (0, 0, 0)
                    sampled_obj_points.append(pts)
                
                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], all_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_indices = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                sampled_boxes = sampled_boxes[valid_indices] # (sample_num', 7)
                sampled_names = sampled_names[valid_indices] # (sample_num')
                sampled_obj_points = [sampled_obj_points[x] for x in valid_indices] # (sample_num')
                
                all_boxes = np.concatenate((all_boxes, sampled_boxes), axis=0)
                all_names = np.concatenate((all_names, sampled_names), axis=0)
                total_sampled_obj_points.extend(sampled_obj_points)
        
        total_sampled_boxes = all_boxes[gt_boxes.shape[0]:, :]
        total_sampled_names = all_names[gt_names.shape[0]:]
        if len(total_sampled_obj_points) > 0:
            total_sampled_boxes, mv_height = self.put_boxes_on_road_plane(
                total_sampled_boxes, data_dict['road_plane'], data_dict['calib']
            ) # adjust the height of sampled boxes
            for i in range(len(total_sampled_obj_points)):
                total_sampled_obj_points[i][:, 2] -= mv_height[i] # adjust the height of sampled points
            data_dict = self.remove_occluded_points(
                data_dict, total_sampled_boxes, total_sampled_names, total_sampled_obj_points
            )
        return data_dict


class DataAugmentor(object):
    def __init__(self, augmentor_list, root_path, class_names, src_point_features,
        range_convertor, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.src_point_features = src_point_features
        self.range_convertor = range_convertor
        self.logger = logger
        
        self.data_augmentor_queue = []
        if augmentor_list is not None:
            for cur_cfg in augmentor_list:
                cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
                self.data_augmentor_queue.append(cur_augmentor)
    
    def gt_sampling(self, config=None):
        db_sampler = DatabaseSampler(
            config, self.root_path, self.class_names, self.src_point_features, self.range_convertor,
            self.logger
        )
        return db_sampler
    
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d
    
    def __setstate__(self, d):
        self.__dict__.update(d)
    
    def random_world_flip_along_x(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip_along_x, config=config)
        data_dict['gt_boxes'], data_dict['colored_points'], _ = random_flip_along_x(
            data_dict['gt_boxes'], data_dict['colored_points']
        )
        return data_dict
    
    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        data_dict['gt_boxes'], data_dict['colored_points'], _ = global_rotation(
            data_dict['gt_boxes'], data_dict['colored_points'], config['WORLD_ROT_ANGLE']
        )
        return data_dict
    
    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        data_dict['gt_boxes'], data_dict['colored_points'], _ = global_scaling(
            data_dict['gt_boxes'], data_dict['colored_points'], config['WORLD_SCALE_RANGE']
        )
        return data_dict
    
    def forward(self, data_dict):
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
        return data_dict

