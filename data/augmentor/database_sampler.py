import pickle

import os
import copy
import numpy as np
import SharedArray
import torch.distributed as dist
import torch
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

from ops.iou3d_nms import iou3d_nms_utils
from ops.roiaware_pool3d import roiaware_pool3d_utils
from utils import box_utils
from utils import augmentor_utils


class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, src_point_features, logger=None):
        self.root_path = root_path
        self.sampler_cfg = sampler_cfg
        self.class_names = class_names
        self.src_point_features = src_point_features
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
        
        self.min_visible_points = sampler_cfg.MIN_VISIBLE_POINTS
        self.remove_occluded_points = sampler_cfg.REMOVE_OCCLUDED_POINTS
        self.change_object_position = sampler_cfg.CHANGE_OBJECT_POSITION

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

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer:pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
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

    def add_sampled_boxes_to_scene(self, data_dict, total_sampled_boxes, total_sampled_names, total_sampled_obj_points_list):
        points = data_dict['colored_points']
        road_plane = data_dict['road_plane']
        calib = data_dict['calib']
        
        sampled_boxes, mv_height = self.put_boxes_on_road_plane(total_sampled_boxes, road_plane, calib) # adjust the height of sampled_boxes
        sampled_names = total_sampled_names
        sampled_obj_points_list = []
        for i in range(len(total_sampled_obj_points_list)):
            pts = total_sampled_obj_points_list[i]
            pts[:, 2] -= mv_height[i] # adjust the height of sampled_obj_points
            sampled_obj_points_list.append(pts)
        
        existed_boxes = data_dict['gt_boxes']
        existed_names = data_dict['gt_names']
        existed_obj_points_list = []
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(existed_boxes)
        ).numpy()  # (nboxes, npoints)
        for i in range(existed_boxes.shape[0]):
            existed_obj_points = points[point_indices[i] > 0]
            existed_obj_points_list.append(existed_obj_points)
        
        background_points = box_utils.remove_points_in_boxes3d(points, existed_boxes)
        all_boxes = np.concatenate([existed_boxes, sampled_boxes], axis=0)
        all_names = np.concatenate([existed_names, sampled_names], axis=0)
        all_obj_points_list = existed_obj_points_list + sampled_obj_points_list
        
        if self.remove_occluded_points:
            mask = [pts.shape[0] >= self.min_visible_points for pts in all_obj_points_list]
            all_boxes = all_boxes[mask]
            all_names = all_names[mask]
            all_obj_points_list = [all_obj_points_list[idx] for idx in range(len(all_obj_points_list)) if mask[idx]]
            
            obj_occlusion_ratio_list = []
            for i in range(len(all_obj_points_list)):
                corners = box_utils.boxes_to_corners_3d(all_boxes[i:i + 1]).squeeze()
                corners_img, _ = calib.lidar_to_img(corners[:, 0:3])
                
                hull = ConvexHull(corners_img)
                pts_of_hull = corners_img[hull.vertices]
                tri = Delaunay(pts_of_hull)
                
                bkg_pts_img, _ = calib.lidar_to_img(background_points[:, 0:3])
                occluded_pts_mask = tri.find_simplex(bkg_pts_img) >= 0
                occluded_points = background_points[occluded_pts_mask]
                
                if occluded_points.shape[0] > 0:
                    dis_obj = np.mean(np.sqrt(corners[:, 0] ** 2 + corners[:, 1] ** 2 + corners[:, 2] ** 2))
                    dis_bkg = np.mean(np.sqrt(occluded_points[:, 0] ** 2 + occluded_points[:, 1] ** 2 + occluded_points[:, 2] ** 2))
                    if dis_bkg < dis_obj:
                        area_obj = (corners_img[:, 0].max() - corners_img[:, 0].min()) * (corners_img[:, 1].max() - corners_img[:, 1].min())
                        area_bkg = (bkg_pts_img[:, 0].max() - bkg_pts_img[:, 0].min()) * (bkg_pts_img[:, 1].max() - bkg_pts_img[:, 1].min())
                        obj_occlusion_ratio_list.append(area_bkg / area_obj)
                    else:
                        obj_occlusion_ratio_list.append(0.0)
                else:
                    obj_occlusion_ratio_list.append(0.0)
            
            mask = [ratio < 0.5 for ratio in obj_occlusion_ratio_list]
            all_boxes = all_boxes[mask]
            all_names = all_names[mask]
            all_obj_points_list = [all_obj_points_list[idx] for idx in range(len(all_obj_points_list)) if mask[idx]]
            
            obj_dis_list = [np.mean(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2 + pts[:, 2] ** 2)) for pts in all_obj_points_list]
            indices = list(np.argsort(obj_dis_list))
            ordered_boxes = all_boxes[indices]
            ordered_names = all_names[indices]
            ordered_obj_points_list = [all_obj_points_list[idx] for idx in indices]
            
            num_obj = len(ordered_obj_points_list)
            if num_obj > 1:
                for i in range(1, num_obj):
                    pts_i = ordered_obj_points_list[i]
                    pts_img_i, _ = calib.lidar_to_img(pts_i[:, 0:3])
                    
                    for j in range(0, i):
                        pts_j = ordered_obj_points_list[j]
                        pts_img_j, _ = calib.lidar_to_img(pts_j[:, 0:3])
                        
                        if pts_img_j.shape[0] >= self.min_visible_points:
                            hull = ConvexHull(pts_img_j)
                            pts_of_hull = pts_img_j[hull.vertices]
                            tri = Delaunay(pts_of_hull)
                            
                            if pts_img_i.shape[0] >= self.min_visible_points:
                                visible_pts_mask = tri.find_simplex(pts_img_i) < 0
                                ordered_obj_points_list[i] = pts_i[visible_pts_mask]
                                pts_i = ordered_obj_points_list[i]
                                pts_img_i, _ = calib.lidar_to_img(pts_i[:, 0:3])
            
            mask = [pts.shape[0] >= self.min_visible_points for pts in ordered_obj_points_list]
            all_boxes = ordered_boxes[mask]
            all_names = ordered_names[mask]
            all_obj_points_list = [ordered_obj_points_list[idx] for idx in range(len(ordered_obj_points_list)) if mask[idx]]
            
            for i in range(len(all_obj_points_list)):
                pts = all_obj_points_list[i]
                pts_img, _ = calib.lidar_to_img(pts[:, 0:3])
                
                hull = ConvexHull(pts_img)
                pts_of_hull = pts_img[hull.vertices]
                tri = Delaunay(pts_of_hull)
                
                bkg_pts_img, _ = calib.lidar_to_img(background_points[:, 0:3])
                visible_pts_mask = tri.find_simplex(bkg_pts_img) < 0
                background_points = background_points[visible_pts_mask]
        
        background_points = box_utils.remove_points_in_boxes3d(background_points, all_boxes)
        data_dict['gt_boxes'] = all_boxes
        data_dict['gt_names'] = all_names
        if len(all_obj_points_list) > 0:
            data_dict['colored_points'] = np.concatenate([background_points, np.concatenate(all_obj_points_list, axis=0)], axis=0)
        else:
            data_dict['colored_points'] = background_points
        
        return data_dict

    def __call__(self, data_dict):
        gt_boxes = data_dict['gt_boxes'] # (num_gt, 7)
        gt_names = data_dict['gt_names'].astype(str)
        
        existed_boxes = gt_boxes # for adding dynamically
        existed_names = gt_names # for adding dynamically
        total_sampled_obj_points_list = [] # for adding dynamically
        
        for class_name, sample_group in self.sample_groups.items():
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)
                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32) # (num_sampled, 7)
                sampled_names = np.array([x['name'] for x in sampled_dict]) # (num_sampled)
                sampled_obj_points_list = []
                for idx, info in enumerate(sampled_dict):
                    file_path = self.root_path / info['path']
                    sampled_obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, self.src_point_features)
                    sampled_obj_points[:, :3] += info['box3d_lidar'][:3] # move the sampled_obj_points to the location of its box from (0, 0, 0)
                    sampled_obj_points_list.append(sampled_obj_points)

                if self.change_object_position:
                    for i in range(len(sampled_obj_points_list)):
                        sampled_boxes[i:i + 1], sampled_obj_points_list[i], _ = augmentor_utils.random_flip_along_x(
                            sampled_boxes[i:i + 1], sampled_obj_points_list[i])
                        sampled_boxes[i:i + 1], sampled_obj_points_list[i], _ = augmentor_utils.global_rotation(
                            sampled_boxes[i:i + 1], sampled_obj_points_list[i], [0, -2 * np.arctan2(sampled_boxes[i, 1], sampled_boxes[i, 0])])
                        sampled_boxes[i:i + 1], sampled_obj_points_list[i], _ = augmentor_utils.global_scaling(
                            sampled_boxes[i:i + 1], sampled_obj_points_list[i], [0.95, 1.05])
                
                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_indices = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                sampled_boxes = sampled_boxes[valid_indices] # (num_sampled', 7)
                sampled_names = sampled_names[valid_indices] # (num_sampled', 7)
                sampled_obj_points_list = [sampled_obj_points_list[x] for x in valid_indices] # (num_sampled')

                existed_boxes = np.concatenate((existed_boxes, sampled_boxes), axis=0)
                existed_names = np.concatenate((existed_names, sampled_names), axis=0)
                total_sampled_obj_points_list.extend(sampled_obj_points_list)

        total_sampled_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        total_sampled_names = existed_names[gt_names.shape[0]:]
        if len(total_sampled_obj_points_list) > 0:
            data_dict = self.add_sampled_boxes_to_scene(data_dict, total_sampled_boxes, total_sampled_names, total_sampled_obj_points_list)

        return data_dict
