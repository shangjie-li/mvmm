from functools import partial

from . import database_sampler
from utils import augmentor_utils


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, src_point_features, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.src_point_features = src_point_features
        self.logger = logger

        self.data_augmentor_queue = []
        if augmentor_configs is not None:
            for cur_cfg in augmentor_configs:
                cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
                self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            src_point_features=self.src_point_features,
            logger=self.logger
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
        gt_boxes, points, enable = augmentor_utils.random_flip_along_x(
            data_dict['gt_boxes'], data_dict['colored_points']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['colored_points'] = points
        data_dict['random_world_flip_along_x'] = enable
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        gt_boxes, points, noise_rotation = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['colored_points'], config['WORLD_ROT_ANGLE']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['colored_points'] = points
        data_dict['random_world_rotation'] = noise_rotation
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points, noise_scaling = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['colored_points'], config['WORLD_SCALE_RANGE']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['colored_points'] = points
        data_dict['random_world_scaling'] = noise_scaling
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                calib: calibration_kitti.Calibration
                gt_names: (M), str
                gt_boxes: (M, 7), [x, y, z, l, w, h, heading] in lidar coordinate system
                road_plane: (4), [a, b, c, d]
                colored_points: (N, 7), Points of (x, y, z, intensity, r, g, b)
                ...

        Returns:
            data_dict:
                calib: calibration_kitti.Calibration
                gt_names: (M'), str
                gt_boxes: (M', 7), [x, y, z, l, w, h, heading] in lidar coordinate system
                road_plane: (4), [a, b, c, d]
                colored_points: (N', 7), Points of (x, y, z, intensity, r, g, b)
                ...

        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
        return data_dict
