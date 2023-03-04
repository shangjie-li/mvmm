import os
import sys
import copy
import pickle
from skimage import io
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from data.kitti_object_eval_python.kitti_common import get_label_annos
from data.kitti_object_eval_python.eval import get_official_eval_result
from utils.box_utils import boxes3d_to_corners3d
from utils.box_utils import in_hull
from utils.box_utils import mask_boxes3d_by_range
from utils.point_cloud_utils import mask_points_by_range
from utils.point_cloud_utils import get_fov_flag
from utils.kitti_calibration_utils import parse_calib
from utils.kitti_object3d_utils import parse_objects
from utils.common_utils import limit_period
from utils.range_image_utils import RangeConvertor
from utils.augmentor_utils import DataAugmentor
from ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu


class KITTIDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split, is_training=True, augment_data=True, create_kitti_infos=False):
        self.root_dir = 'data/kitti'
        self.split = split

        assert self.split in ['train', 'val', 'trainval', 'test']
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.id_list = [x.strip() for x in open(self.split_file).readlines()]

        self.data_dir = os.path.join(self.root_dir, 'testing' if self.split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.velodyne_dir = os.path.join(self.data_dir, 'velodyne')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')
        self.plane_dir = os.path.join(self.data_dir, 'planes')

        self.info_file = os.path.join(self.root_dir, 'kitti_infos_%s.pkl' % self.split)
        if create_kitti_infos:
            return

        self.kitti_infos = []
        with open(self.info_file, 'rb') as f:
            infos = pickle.load(f)
        self.kitti_infos.extend(infos)

        self.class_names = cfg['class_names']
        self.write_list = cfg['write_list']
        self.point_cloud_range = np.array(cfg['point_cloud_range'], dtype=np.float32)

        self.is_training = is_training
        self.augment_data = augment_data
        if self.split not in ['train', 'trainval']:
            self.augment_data = False

        self.range_convertor = RangeConvertor(cfg['range_image'])
        if self.augment_data and cfg.get('augmentor_list') is not None:
            self.data_augmentor = DataAugmentor(
                cfg['augmentor_list'], self.root_dir, self.class_names, self.range_convertor
            )
        else:
            self.data_augmentor = None

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % int(idx))
        assert os.path.exists(img_file)
        return io.imread(img_file).astype(np.float32) / 255.0  # ndarray of float32, [H, W, 3], RGB image

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % int(idx))
        assert os.path.exists(img_file)
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)  # ndarray of int, [2], H and W

    def get_points(self, idx):
        pts_file = os.path.join(self.velodyne_dir, '%06d.bin' % int(idx))
        assert os.path.exists(pts_file)
        return np.fromfile(pts_file, dtype=np.float32).reshape(-1, 4)  # ndarray of float32, [N, 4], (x, y, z, i)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % int(idx))
        assert os.path.exists(calib_file)
        return parse_calib(calib_file)  # kitti_calibration_utils.Calibration

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % int(idx))
        assert os.path.exists(label_file)
        return parse_objects(label_file)  # list of kitti_object3d_utils.Object3d

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.plane_dir, '%06d.txt' % int(idx))
        assert os.path.exists(plane_file)
        with open(plane_file, 'r') as f: lines = f.readlines()
        plane = np.asarray([float(i) for i in lines[3].split()])
        if plane[1] > 0: plane = -plane
        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane  # ndarray of float32, [4], (a, b, c, d)

    def get_colored_points_in_fov(self, idx):
        points = self.get_points(idx)
        image_shape = self.get_image_shape(idx)
        calib = self.get_calib(idx)
        fov_flag = get_fov_flag(points, image_shape, calib)
        points = points[fov_flag]
        pts_img, pts_depth = calib.lidar_to_img(points[:, 0:3])  # [N, 2], [N]
        pts_img = pts_img.astype(np.int)
        img = self.get_image(idx)
        rgb = img[pts_img[:, 1], pts_img[:, 0], :]  # [N, 3]
        return np.concatenate([points, rgb], axis=1)  # ndarray of float32, [N, 7], (x, y, z, i, r, g, b)

    def get_infos(self, has_label=True, count_inside_pts=True, num_workers=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('sample_idx: %s in %s.txt' % (sample_idx, self.split))
            info = {}
            info['sample_idx'] = sample_idx
            info['image_shape'] = self.get_image_shape(sample_idx)
            
            if has_label:
                obj_list = self.get_label(sample_idx)
                annos = {}
                annos['name'] = np.array([obj.cls_type for obj in obj_list])
                annos['truncated'] = np.array([obj.truncation for obj in obj_list])
                annos['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annos['alpha'] = np.array([obj.alpha for obj in obj_list])
                annos['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annos['dimensions'] = np.array([[obj.h, obj.w, obj.l] for obj in obj_list])
                annos['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annos['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annos['score'] = np.array([obj.score for obj in obj_list])
                annos['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                xyz = annos['location'][:num_objects]
                hwl = annos['dimensions'][:num_objects]
                rot_y = annos['rotation_y'][:num_objects]

                calib = self.get_calib(sample_idx)
                xyz_lidar = calib.rect_to_lidar(xyz)
                h, w, l = hwl[:, 0:1], hwl[:, 1:2], hwl[:, 2:3]
                xyz_lidar[:, 2] += h[:, 0] / 2
                annos['gt_box_lidar'] = np.concatenate(
                    [xyz_lidar, l, w, h, -(np.pi / 2 + rot_y[..., np.newaxis])], axis=1
                )  # [M, 7], (x, y, z, l, w, h, heading) in lidar coordinates

                info['annos'] = annos

                if count_inside_pts:
                    colored_points = self.get_colored_points_in_fov(sample_idx)
                    corners = boxes3d_to_corners3d(annos['gt_box_lidar'])
                    num_points_in_gt = np.zeros(num_objects, dtype=np.int32)
                    for k in range(num_objects):
                        flag = in_hull(colored_points[:, 0:3], corners[k])
                        num_points_in_gt[k] = flag.sum()
                    annos['num_points_in_gt'] = num_points_in_gt  # [M]
            
            return info
        
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, self.id_list)
            
        return list(infos)

    def create_gt_database(self):
        import torch

        database_dir = Path(self.root_dir) / ('gt_database_%s' % self.split)
        db_info_file = Path(self.root_dir) / ('kitti_dbinfos_%s.pkl' % self.split)

        database_dir.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(self.info_file, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample in %s.txt: %d/%d' % (self.split, k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['sample_idx']
            annos = info['annos']

            colored_points = self.get_colored_points_in_fov(sample_idx)
            names = annos['name']
            gt_boxes = annos['gt_box_lidar']
            difficulties = annos['difficulty']
            bboxes = annos['bbox']

            num_objects = gt_boxes.shape[0]
            point_indices = points_in_boxes_cpu(
                torch.from_numpy(colored_points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # [num_boxes, num_points]

            for i in range(num_objects):
                pts_file = database_dir / ('%s_%s_%d.bin' % (sample_idx, names[i], i))
                gt_points = colored_points[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]  # move the points of object to (0, 0, 0)
                with open(pts_file, 'w') as f:
                    gt_points.tofile(f)

                db_info = {
                    'path': str(pts_file.relative_to(self.root_dir)),  # gt_database/xxxxx.bin
                    'name': names[i],
                    'gt_box_lidar': gt_boxes[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulties[i],
                    'bbox': bboxes[i],
                }

                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

        for k, v in all_db_infos.items():
            print('Number of ground truths in the %s class: %d' % (k, len(v)))

        with open(db_info_file, 'wb') as f:
            pickle.dump(all_db_infos, f)

    def eval(self, result_dir, logger):
        logger.info('==> Loading detections and ground truths...')
        img_ids = [int(idx) for idx in self.id_list]
        dt_annos = get_label_annos(result_dir)
        gt_annos = get_label_annos(self.label_dir, img_ids)
        logger.info('==> Done.')

        logger.info('==> Evaluating...')
        test_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        for category in self.write_list:
            result_str = get_official_eval_result(
                gt_annos, dt_annos, test_id[category], use_ldf_eval=False, print_info=False)
            logger.info(result_str)

    def __len__(self):
        return len(self.kitti_infos)

    def __getitem__(self, idx):
        """

        Returns:
            data_dict:
                frame_id: str, sample index
                colored_points: ndarray of float, [N, 7], (x, y, z, intensity, r, g, b)
                calib: kitti_calibration_utils.Calibration
                image_shape: ndarray of int, [2], H and W
                gt_boxes: ndarray of float, [M, 8], (x, y, z, l, w, h, heading, class_id) in lidar coordinates
                range_image: ndarray of float, [7, 48, 512], front range image

        """
        info = copy.deepcopy(self.kitti_infos[idx])
        sample_idx = info['sample_idx']
        data_dict = {
            'frame_id': sample_idx,
            'colored_points': self.get_colored_points_in_fov(sample_idx),
            'calib': self.get_calib(sample_idx),
            'image_shape': info['image_shape'],
        }
        data_dict['colored_points'] = mask_points_by_range(data_dict['colored_points'], self.point_cloud_range)
        
        if 'annos' in info:
            annos = info['annos']
            keep_annos = {}
            keep_indices = [i for i, x in enumerate(annos['name']) if x in self.class_names]
            for key in annos.keys():
                keep_annos[key] = annos[key][keep_indices]

            data_dict.update({
                'gt_boxes': keep_annos['gt_box_lidar'],
                'gt_names': keep_annos['name'],
                'road_plane': self.get_road_plane(sample_idx)
            })

            data_dict = self.data_augmentor.forward(data_dict) if self.data_augmentor else data_dict

            gt_cls_ids = [self.class_names.index(n) + 1 for n in data_dict['gt_names']]
            data_dict['gt_boxes'] = np.concatenate(
                [data_dict['gt_boxes'], np.array(gt_cls_ids).reshape(-1, 1).astype(np.float32)], axis=1
            )  # [M, 8], (x, y, z, l, w, h, heading, class_id) in lidar coordinates

            data_dict.pop('gt_names', None)
            data_dict.pop('road_plane', None)

            data_dict['gt_boxes'][:, 6] = limit_period(data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi)

            data_dict['gt_boxes'] = mask_boxes3d_by_range(data_dict['gt_boxes'], self.point_cloud_range)
            data_dict['colored_points'] = mask_points_by_range(data_dict['colored_points'], self.point_cloud_range)
        
        if self.is_training:
            if len(data_dict['gt_boxes']) == 0:
                return self.__getitem__(np.random.randint(self.__len__()))
                
            shuffle_indices = np.random.permutation(data_dict['colored_points'].shape[0])
            data_dict['colored_points'] = data_dict['colored_points'][shuffle_indices]
        
        points = data_dict['colored_points']

        xs = points[:, 0:1]
        ys = points[:, 1:2]
        zs = points[:, 2:3]
        intensities = points[:, 3:4]
        colors = points[:, 4:7]

        xmin, ymin, zmin, xmax, ymax, zmax = self.point_cloud_range
        xs = (xs - xmin) / (xmax - xmin)
        ys = (ys - ymin) / (ymax - ymin)
        zs = (zs - zmin) / (zmax - zmin)

        point_features = np.concatenate([xs, ys, zs, intensities, colors], axis=1)
        data_dict['range_image'] = self.range_convertor.get_range_image(points, point_features)

        return data_dict

    @staticmethod
    def collate_batch(batch_list):
        """

        Args:
            batch_list: list of data_dict

        Returns:
            batch_dict:
                batch_size: int
                frame_id: ndarray of str, [batch_size], sample index
                colored_points: ndarray of float, [N1 + N2 + ..., 8], (batch_id, x, y, z, intensity, r, g, b)
                calib: ndarray of kitti_calibration_utils.Calibration, [batch_size]
                image_shape: ndarray of int, [batch_size, 2], H and W
                gt_boxes: ndarray of float [batch_size, M_max, 8], (x, y, z, l, w, h, heading, class_id) in lidar coordinates
                range_image: ndarray of float, [batch_size, 7, 48, 512], front range image

        """
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)

        batch_size = len(batch_list)
        batch_dict = {}
        batch_dict['batch_size'] = batch_size
        
        for key, val in data_dict.items():
            if key in ['colored_points']:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                batch_dict[key] = np.concatenate(coors, axis=0)
            elif key in ['gt_boxes']:
                max_gt = max([len(x) for x in val])
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                batch_dict[key] = batch_gt_boxes3d
            else:
                batch_dict[key] = np.stack(val, axis=0)

        return batch_dict

    @staticmethod
    def load_data_to_gpu(batch_dict, device):
        for key, val in batch_dict.items():
            if key in ['batch_size', 'frame_id', 'calib', 'image_shape']:
                continue
            else:
                batch_dict[key] = torch.from_numpy(val).float().to(device)

        return batch_dict


if __name__ == '__main__':
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        print('==> Generating data info files...')

        dataset = KITTIDataset(cfg={}, split='train', create_kitti_infos=True)
        train_infos = dataset.get_infos(has_label=True, count_inside_pts=True)
        with open(dataset.info_file, 'wb') as f:
            pickle.dump(train_infos, f)
        print('==> The info file for `train.txt` is saved to: %s' % dataset.info_file)

        dataset = KITTIDataset(cfg={}, split='val', create_kitti_infos=True)
        val_infos = dataset.get_infos(has_label=True, count_inside_pts=True)
        with open(dataset.info_file, 'wb') as f:
            pickle.dump(val_infos, f)
        print('==> The info file for `val.txt` is saved to: %s' % dataset.info_file)

        dataset = KITTIDataset(cfg={}, split='test', create_kitti_infos=True)
        test_infos = dataset.get_infos(has_label=False, count_inside_pts=False)
        with open(dataset.info_file, 'wb') as f:
            pickle.dump(test_infos, f)
        print('==> The info file for `test.txt` is saved to: %s' % dataset.info_file)

        dataset = KITTIDataset(cfg={}, split='trainval', create_kitti_infos=True)
        trainval_infos = train_infos + val_infos
        with open(dataset.info_file, 'wb') as f:
            pickle.dump(trainval_infos, f)
        print('==> The info file for `trainval.txt` is saved to: %s' % dataset.info_file)
        
        print('==> Generating ground truth databases...')

        dataset = KITTIDataset(cfg={}, split='train', create_kitti_infos=True)
        dataset.create_gt_database()

        dataset = KITTIDataset(cfg={}, split='trainval', create_kitti_infos=True)
        dataset.create_gt_database()

        print('==> Done.')
