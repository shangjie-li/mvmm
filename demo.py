import argparse
import glob
from pathlib import Path
import time
import os

import numpy as np
import torch

from data import cfg, cfg_from_yaml_file
from data import KittiDataset
from mvmm import build_network, load_data_to_gpu
from utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/config/ResNet_VFE.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='specify the pretrained model')
    parser.add_argument('--compute_latency', action='store_true', default=False,
                        help='whether to compute the latency')
    parser.add_argument('--show_gt_boxes', action='store_true', default=False,
                        help='whether to show ground truth boxes')
    parser.add_argument('--sample_idx', type=str, default=None,
                        help='specify the index of sample')
    parser.add_argument('--onto_image', action='store_true', default=False,
                        help='whether to project results onto the RGB image')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def print_info(data, des=None):
    if des is not None:
        print(des)
    if isinstance(data, dict):
        for key, val in data.items():
            try:
                print(key, type(val), val.shape, '\n', val)
            except:
                print(key, type(val), '\n', val)
    else:
        print(data)
    print()


def run(model, demo_dataset, data_dict, frame_id, args):
    print_info(data_dict, des='<<< data_dict >>>')
    
    load_data_to_gpu(data_dict)
    pred_dicts, _ = model.forward(data_dict) # 0
    print_info(pred_dicts[0], '<<< pred_dicts[0] >>>')
    
    if args.compute_latency:
        num_iter, time0 = 10, time.time()
        for _ in range(num_iter):
            pred_dicts, _ = model.forward(data_dict)
        print('Time cost per batch: %s' % (round((time.time() - time0) / num_iter, 3)))
    
    if set(['r', 'g', 'b']).issubset(set(demo_dataset.used_feature_list)):
        point_colors = data_dict['colored_points'][:, -3:]
    else:
        point_colors = None
    if args.show_gt_boxes:
        gt_boxes, ref_labels = data_dict['gt_boxes'][0, :, :7], None
    else:
        gt_boxes, ref_labels = None, [cfg.CLASS_NAMES[j - 1] for j in pred_dicts[0]['pred_labels']]
    
    if args.onto_image:
        from utils import opencv_vis_utils as V
        V.draw_scenes(
            img=demo_dataset.get_image(frame_id)[:, :, ::-1],
            calib=demo_dataset.get_calib(frame_id), gt_boxes=gt_boxes,
            ref_boxes=pred_dicts[0]['pred_boxes'], ref_labels=ref_labels,
            window_name=frame_id,
        )
    else:
        from utils import open3d_vis_utils as V
        V.draw_scenes(
            points=data_dict['colored_points'][:, 1:4], gt_boxes=gt_boxes,
            ref_boxes=pred_dicts[0]['pred_boxes'], ref_labels=ref_labels, point_colors=point_colors,
            window_name=frame_id,
        )
    print()


if __name__ == '__main__':
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of MVMM-------------------------')
    
    demo_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    
    print_info(model, des='\n<<< model >>>')
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        if args.sample_idx is not None:
            assert args.sample_idx in demo_dataset.sample_id_list, \
                'Invalid sample index: %s' % args.sample_idx
            data_dict = demo_dataset[demo_dataset.sample_id_list.index(args.sample_idx)]
            data_dict = demo_dataset.collate_batch([data_dict])
            frame_id = data_dict['frame_id'][0]
            logger.info(f'Visualizing sample index: \t{frame_id}')
            run(model, demo_dataset, data_dict, frame_id, args)
        else:
            for idx, data_dict in enumerate(demo_dataset):
                data_dict = demo_dataset.collate_batch([data_dict])
                frame_id = data_dict['frame_id'][0]
                logger.info(f'Visualizing sample index: \t{frame_id}')
                run(model, demo_dataset, data_dict, frame_id, args)
        logger.info('Demo done.')
