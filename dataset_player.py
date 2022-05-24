import argparse
import glob
from pathlib import Path
import time
import copy

import numpy as np
import torch

from data import cfg, cfg_from_yaml_file
from data import KittiDataset
from utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/config/ResNet_VFE.yaml',
        help='specify the config for training')
    parser.add_argument('--training', action='store_true', default=False,
        help='whether to use training mode')
    parser.add_argument('--data_augmentation', action='store_true', default=False,
        help='whether to use data augmentation')
    parser.add_argument('--show_boxes', action='store_true', default=False,
        help='whether to show boxes')
    parser.add_argument('--onto_range_image', action='store_true', default=False,
        help='whether to show the range image')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


if __name__ == '__main__':
    args, cfg = parse_config()
    print(cfg)
    
    dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=args.training,
        data_augmentation=args.data_augmentation
    )
    
    for i in range(len(dataset)):
        print('\n--------[%d/%d]--------' % (i + 1, len(dataset)))
        data_dict = dataset[i]
        
        print()
        print('<<< data_dict >>>')
        for key, val in data_dict.items():
            if isinstance(val, np.ndarray):
                print(key, type(val), val.shape)
                print(val)
            else:
                print(key, type(val))
                print(val)
        print()
        
        points = data_dict['colored_points'][:, 0:3]
        f = dataset.used_feature_list
        if args.onto_range_image:
            from utils import opencv_vis_utils as V
            features = dataset.get_point_features(data_dict['colored_points'], f)
            img = dataset.range_convertor.get_range_image(points, features).transpose(1, 2, 0)
            _, _, cu, cv = dataset.range_convertor.get_pixel_coords(points)
            if set(['r', 'g', 'b']).issubset(set(f)):
                img = img[:, :, [f.index('b'), f.index('g'), f.index('r')]]
            elif 'intensity' in f:
                img = img[:, :, [f.index('intensity')]]
            else:
                img = img[:, :, [0]]
            V.draw_scenes(
                img,
                calib=None,
                ref_boxes2d=dataset.range_convertor.get_range_boxes(data_dict['gt_boxes'][:, :7], cu, cv) if args.show_boxes else None,
                ref_labels=[cfg.CLASS_NAMES[j - 1] for j in data_dict['gt_boxes'][:, 7].astype(np.int)],
                window_name=data_dict['frame_id'],
            )
        else:
            from utils import open3d_vis_utils as V
            if set(['r', 'g', 'b']).issubset(set(f)):
                point_colors = data_dict['colored_points'][:, -3:]
            else:
                point_colors = None
            V.draw_scenes(
                points,
                ref_boxes=data_dict['gt_boxes'][:, :7] if args.show_boxes else None,
                ref_labels=[cfg.CLASS_NAMES[j - 1] for j in data_dict['gt_boxes'][:, 7].astype(np.int)],
                point_colors=point_colors,
                window_name=data_dict['frame_id'],
            )
        
