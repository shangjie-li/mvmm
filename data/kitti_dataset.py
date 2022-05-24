import sys
import copy
import pickle
from skimage import io
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
import torch.utils.data as torch_data

from ops.roiaware_pool3d import roiaware_pool3d_utils
from utils import box_utils, common_utils, object3d_kitti, calibration_kitti
from utils import range_image_utils, augmentor_utils


class KittiDataset(torch_data.Dataset):
    def __init__(self, dataset_cfg, class_names, training=True, logger=None, data_augmentation=True):
        self.class_names = class_names
        self.root_path = Path('data/kitti')
        if dataset_cfg:
            self.dataset_cfg = dataset_cfg
            self.training = training
            self.logger = logger
            if self.training:
                self.set_split(dataset_cfg.SPLIT['train'])
                self.data_augmentation = data_augmentation
                self.prepare_data()
            else:
                self.set_split(dataset_cfg.SPLIT['test'])
                self.data_augmentation = False
                self.prepare_data()

    def set_split(self, s):
        assert s in ['test.txt', 'train.txt', 'trainval.txt', 'val.txt']
        self.split = s
        self.sample_id_list = [
            x.strip() for x in open(self.root_path / 'ImageSets' / self.split).readlines()]
        self.data_path = self.root_path / ('testing' if self.split == 'test.txt' else 'training')
        self.info_file = self.root_path / ('kitti_infos_%s.pkl' % str(Path(self.split).stem))

    def prepare_data(self):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        
        self.kitti_infos = []
        with open(self.info_file, 'rb') as f:
            infos = pickle.load(f)
        self.kitti_infos.extend(infos)
        
        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(self.kitti_infos)))
        
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.src_feature_list = self.dataset_cfg.SRC_FEATURE_LIST
        self.src_point_features = len(self.src_feature_list)
        self.used_feature_list = self.dataset_cfg.USED_FEATURE_LIST
        self.used_point_features = len(self.used_feature_list)
        
        self.range_convertor = range_image_utils.RangeConvertor(
            self.dataset_cfg.RANGE_IMAGE_CONFIG
        )
        
        if self.data_augmentation and self.dataset_cfg.get('AUGMENTOR_LIST', None) is not None:
            self.data_augmentor = augmentor_utils.DataAugmentor(
                self.dataset_cfg.AUGMENTOR_LIST,
                self.root_path, self.class_names, self.src_point_features, self.range_convertor,
                logger=self.logger
            )
        else:
            self.data_augmentor = None
        
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

    def get_points(self, idx):
        """
        Args:
            idx: str, sample index
        Returns:
            points: (N, 4), points of (x, y, z, intensity)
        """
        pts_file = self.data_path / 'velodyne' / ('%s.bin' % idx)
        assert pts_file.exists(), 'File not found: %s' % pts_file
        return np.fromfile(str(pts_file), dtype=np.float32).reshape(-1, 4)

    def get_image(self, idx):
        """
        Args:
            idx: str, sample index
        Returns:
            img: (H, W, 3), RGB image
        """
        img_file = self.data_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists(), 'File not found: %s' % img_file
        img = io.imread(img_file)
        img = img.astype(np.float32)
        img /= 255.0
        return img

    def get_image_shape(self, idx):
        """
        Args:
            idx: str, sample index
        Returns:
            image_shape: (2), H * W
        """
        img_file = self.data_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists(), 'File not found: %s' % img_file
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        """
        Args:
            idx: str, sample index
        Returns:
            objects: list, [Object3d, Object3d, ...]
        """
        label_file = self.data_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists(), 'File not found: %s' % label_file
        return object3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, idx):
        """
        Args:
            idx: str, sample index
        Returns:
            calib: calibration_kitti.Calibration
        """
        calib_file = self.data_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists(), 'File not found: %s' % calib_file
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        """
        Args:
            idx: str, sample index
        Returns:
            plane: (4), [a, b, c, d]
        """
        plane_file = self.data_path / 'planes' / ('%s.txt' % idx)
        assert plane_file.exists(), 'File not found: %s' % plane_file

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinates
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(points, img_shape, calib):
        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        return pts_valid_flag

    def get_colored_points_in_fov(self, idx):
        """
        Args:
            idx: str, sample index
        Returns:
            colored_points: (N, 7), points of (x, y, z, intensity, r, g, b)
        """
        points = self.get_points(idx)
        calib = self.get_calib(idx)
        fov_flag = self.get_fov_flag(points, self.get_image_shape(idx), calib)
        points = points[fov_flag]
        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect) # [N', 2], [N']
        pts_img = pts_img.astype(np.int)
        rgb = self.get_image(idx)[pts_img[:, 1], pts_img[:, 0], :] # [N', 3]
        return np.concatenate([points, rgb], axis=1)

    def get_point_features(self, points, used_feature_list, normalize=False):
        xs = points[:, 0:1]
        ys = points[:, 1:2]
        zs = points[:, 2:3]
        ranges = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
        intensities = points[:, 3:4]
        colors = points[:, 4:7]
        
        if normalize:
            xs = (xs - 15.96) / 10.85
            ys = (ys - 0.11) / 7.19
            zs = (zs - (-1.04)) / 0.82
            ranges = (ranges - 17.29) / 11.35
            intensities = (intensities - 0.24) / 0.15
            colors = (colors - np.array((0.35, 0.36, 0.35))) / np.array((0.27, 0.26, 0.26))
        
        f = used_feature_list
        if f == ['x', 'y', 'z', 'intensity', 'r', 'g', 'b']:
            features = np.concatenate([xs, ys, zs, intensities, colors], axis=1)
        elif f == ['x', 'y', 'z', 'intensity']:
            features = np.concatenate([xs, ys, zs, intensities], axis=1)
        elif f == ['r', 'g', 'b']:
            features = colors
        elif f == ['x', 'y', 'z', 'range', 'intensity']:
            features = np.concatenate([xs, ys, zs, ranges, intensities], axis=1)
        elif f == ['x', 'y', 'z', 'range', 'intensity', 'r', 'g', 'b']:
            features = np.concatenate([xs, ys, zs, ranges, intensities, colors], axis=1)
        elif f == ['x', 'y', 'z']:
            features = np.concatenate([xs, ys, zs], axis=1)
        elif f == ['range']:
            features = ranges
        else:
            raise NotImplementedError
        return features

    def get_infos(self, has_label=True, count_inside_pts=True, num_workers=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('sample_idx: %s in %s' % (sample_idx, self.split))
            info = {}
            info['point_cloud'] = {'num_features': 4, 'lidar_idx': sample_idx}
            info['image'] = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            
            calib = self.get_calib(sample_idx)
            P2_4x4 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            info['calib'] = {'P2': P2_4x4, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
            
            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare']) # including Pedestrian, Truck, Car, Cyclist, Misc, etc.
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar # (M, 7), [x, y, z, l, w, h, heading] in lidar coordinates

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_points(sample_idx)
                    calib = self.get_calib(sample_idx)
                    fov_flag = self.get_fov_flag(points, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt
            
            return info
        
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, self.sample_id_list)
            
        return list(infos)

    def create_gt_database(self):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database_%s' % str(Path(self.split).stem))
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s.pkl' % str(Path(self.split).stem))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(self.info_file, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample in %s: %d/%d' % (self.split, k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            colored_points = self.get_colored_points_in_fov(sample_idx)
            annos = info['annos']
            names = annos['name']
            bbox = annos['bbox']
            score = annos['score']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(colored_points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = colored_points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3] # move the points of object to (0, 0, 0)
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                           'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                           'difficulty': difficulty[i], 'bbox': bbox[i], 'score': score[i]}
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
            pred_dicts: list, [{pred_boxes: (M, 7), pred_scores: (M), pred_labels: (M)}, {...}, ...]
            class_names:
            output_path:
        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx],
                                 common_utils.normalize_angle(single_pred_dict['alpha'][idx]),
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0],
                                 loc[idx][0], loc[idx][1], loc[idx][2],
                                 common_utils.normalize_angle(single_pred_dict['rotation_y'][idx]),
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __setstate__(self, d):
        self.__dict__.update(d)
    
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs
        return len(self.kitti_infos)

    def __getitem__(self, index):
        """
        Returns:
            data_dict:
                frame_id: str, sample index
                colored_points: (N, 7), points of (x, y, z, intensity, r, g, b)
                calib: calibration_kitti.Calibration
                image_shape: (2), H * W
                gt_boxes: (M, 8), [x, y, z, l, w, h, heading, class_id] in lidar coordinates
                range_image: (used_point_features, 48, 512), front range image
        """
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)
        
        info = copy.deepcopy(self.kitti_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        data_dict = {
            'frame_id': sample_idx,
            'colored_points': self.get_colored_points_in_fov(sample_idx), # (N, 7)
            'calib': self.get_calib(sample_idx),
            'image_shape': info['image']['image_shape'],
        }
        
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare') # exclude class: DontCare
            data_dict.update({
                'gt_boxes': annos['gt_boxes_lidar'],
                'gt_names': annos['name'],
                'road_plane': self.get_road_plane(sample_idx)
            })
        
        mask = common_utils.mask_points_by_range(data_dict['colored_points'], self.point_cloud_range)
        data_dict['colored_points'] = data_dict['colored_points'][mask]
        
        if data_dict.get('gt_boxes', None) is not None:
            # Filter by the class: Car, Pedestrian, Cyclist
            mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            data_dict['gt_names'] = data_dict['gt_names'][mask]
            
            # Augment data
            data_dict = self.data_augmentor.forward(data_dict) if self.data_augmentor else data_dict
            
            # Limit heading to [-pi, pi)
            data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
                data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
            )
            
            # Merge gt_boxes and gt_classes: Car - 1, Pedestrian - 2, Cyclist - 3
            gt_classes = [self.class_names.index(n) + 1 for n in data_dict['gt_names']]
            data_dict['gt_boxes'] = np.concatenate(
                [data_dict['gt_boxes'], np.array(gt_classes).reshape(-1, 1).astype(np.float32)], axis=1
            ) # (M, 8), [x, y, z, l, w, h, heading, class_id] in lidar coordinates
            
            # Mask boxes outside range
            mask = box_utils.mask_boxes_outside_range_numpy(data_dict['gt_boxes'], self.point_cloud_range)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        
        if self.training:
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
                
            shuffle_idx = np.random.permutation(data_dict['colored_points'].shape[0])
            data_dict['colored_points'] = data_dict['colored_points'][shuffle_idx]
        
        points = data_dict['colored_points']
        features = self.get_point_features(points, self.used_feature_list, normalize=True)
        data_dict['range_image'] = self.range_convertor.get_range_image(points, features)
        data_dict.pop('gt_names', None)
        data_dict.pop('road_plane', None)
        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        """
        Args:
            batch_list: list, [data_dict, data_dict, ...]
        Returns:
            ret:
                batch_size: int
                frame_id: (batch_size), str, sample index
                colored_points: (N1 + N2 + ..., 8), points of (batch_id, x, y, z, intensity, r, g, b)
                calib: (batch_size), calibration_kitti.Calibration
                image_shape: (batch_size, 2), H * W
                gt_boxes: (batch_size, M_max, 8), [x, y, z, l, w, h, heading, class_id] in lidar coordinates
                range_image: (batch_size, used_point_features, 48, 512), front range image
        """
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        ret['batch_size'] = batch_size
        
        for key, val in data_dict.items():
            try:
                if key in ['colored_points']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError
        return ret


if __name__ == '__main__':
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        root_dir = (Path(__file__).resolve().parent / '../').resolve() # ~/mvmm
        save_path = root_dir / 'data/kitti'
        class_names = ['Car', 'Pedestrian', 'Cyclist']
        dataset = KittiDataset(dataset_cfg=None, class_names=class_names)
        
        test_file_name = 'kitti_infos_test.pkl'
        train_file_name = 'kitti_infos_train.pkl'
        trainval_file_name = 'kitti_infos_trainval.pkl'
        val_file_name = 'kitti_infos_val.pkl'
        
        print('---------------Generating data infos---------------')
        
        dataset.set_split('train.txt')
        kitti_infos_train = dataset.get_infos(has_label=True, count_inside_pts=True)
        with open(save_path / train_file_name, 'wb') as f:
            pickle.dump(kitti_infos_train, f)
        print('Kitti info for train.txt is saved to %s' % train_file_name)
        
        dataset.set_split('val.txt')
        kitti_infos_val = dataset.get_infos(has_label=True, count_inside_pts=True)
        with open(save_path / val_file_name, 'wb') as f:
            pickle.dump(kitti_infos_val, f)
        print('Kitti info for val.txt is saved to %s' % val_file_name)
        
        dataset.set_split('test.txt')
        kitti_infos_test = dataset.get_infos(has_label=False, count_inside_pts=False)
        with open(save_path / test_file_name, 'wb') as f:
            pickle.dump(kitti_infos_test, f)
        print('Kitti info for test.txt is saved to %s' % test_file_name)
        
        with open(save_path / trainval_file_name, 'wb') as f:
            pickle.dump(kitti_infos_train + kitti_infos_val, f)
        print('Kitti info for trainval.txt is saved to %s' % trainval_file_name)
        
        print('---------------Generating ground truth databases---------------')
        
        dataset.set_split('train.txt')
        dataset.create_gt_database()
        
        dataset.set_split('trainval.txt')
        dataset.create_gt_database()
        
        print('---------------Data preparation Done---------------')

