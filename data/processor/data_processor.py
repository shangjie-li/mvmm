from functools import partial
import numpy as np

from utils import box_utils, common_utils


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if self.training else 'test'

        self.data_processor_queue = []
        if processor_configs is not None:
            for cur_cfg in processor_configs:
                cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
                self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('colored_points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['colored_points'], self.point_cloud_range)
            data_dict['colored_points'] = data_dict['colored_points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['colored_points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['colored_points'] = points

        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (M, 8), [x, y, z, l, w, h, heading, class_id] in lidar coordinate system
                colored_points: (N, 7), Points of (x, y, z, intensity, r, g, b)
                ...

        Returns:
            data_dict:
                gt_boxes: (M', 8), [x, y, z, l, w, h, heading, class_id] in lidar coordinate system
                colored_points: (N', 7), Points of (x, y, z, intensity, r, g, b)
                ...
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)
        return data_dict
