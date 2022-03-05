import argparse
import glob
from pathlib import Path
import time
import os

from utils import open3d_vis_utils as V

import numpy as np
import torch

from data import cfg, cfg_from_yaml_file
from data import KittiDataset
from mvmm import build_network, load_data_to_gpu
from utils import common_utils


class DemoDataset(KittiDataset):
    def __init__(self, dataset_cfg, class_names, training=True, data_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, logger=logger
        )
        self.data_path = data_path
        file_list = glob.glob(str(self.data_path / '*.bin')) if self.data_path.is_dir() else [self.data_path]
        file_list.sort()
        self.sample_file_list = file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        sample_idx = os.path.basename(self.sample_file_list[index])[:-4]

        input_dict = {
            'frame_id': sample_idx,
            'colored_points': self.get_colored_points_in_fov(sample_idx),
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/config/ResNet_PFE.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='data/kitti/training/velodyne/000008.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='specify the pretrained model')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of MVMM-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        data_path=Path(args.data_path), logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    
    print()
    print('<<< model >>>')
    print(model)
    print()
    
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            
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
            
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict) # 0
            
            time_start = time.time()
            pred_dicts, _ = model.forward(data_dict) # 1
            pred_dicts, _ = model.forward(data_dict) # 2
            pred_dicts, _ = model.forward(data_dict) # 3
            pred_dicts, _ = model.forward(data_dict) # 4
            pred_dicts, _ = model.forward(data_dict) # 5
            pred_dicts, _ = model.forward(data_dict) # 6
            pred_dicts, _ = model.forward(data_dict) # 7
            pred_dicts, _ = model.forward(data_dict) # 8
            pred_dicts, _ = model.forward(data_dict) # 9
            pred_dicts, _ = model.forward(data_dict) # 10
            time_end = time.time()
            
            print()
            print('<<< pred_dicts[0] >>>') # It seems that there is only one element in the list of pred_dicts.
            for key, val in pred_dicts[0].items():
                try:
                    print(key, type(val), val.shape)
                    print(val)
                except:
                    print(key, type(val))
                    print(val)
            print()

            # ~ if set(['r', 'g', 'b']).issubset(set(demo_dataset.used_feature_list)):
                # ~ V.draw_scenes(
                    # ~ points=data_dict['colored_points'][:, 1:4].cpu().numpy(),
                    # ~ ref_boxes=pred_dicts[0]['pred_boxes'],
                    # ~ ref_scores=pred_dicts[0]['pred_scores'],
                    # ~ ref_labels=pred_dicts[0]['pred_labels'],
                    # ~ point_colors=data_dict['colored_points'][:, -3:].cpu().numpy(),
                    # ~ point_size=4.0
                # ~ )
            # ~ else:
                # ~ V.draw_scenes(
                    # ~ points=data_dict['colored_points'][:, 1:4].cpu().numpy(),
                    # ~ ref_boxes=pred_dicts[0]['pred_boxes'],
                    # ~ ref_scores=pred_dicts[0]['pred_scores'],
                    # ~ ref_labels=pred_dicts[0]['pred_labels'],
                    # ~ point_colors=np.ones((data_dict['colored_points'].shape[0], 3)),
                    # ~ point_size=2.0
                # ~ )
            
            V.draw_scenes(
                points=data_dict['colored_points'][:, 1:4].cpu().numpy(),
                ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'],
                ref_labels=pred_dicts[0]['pred_labels'],
                point_labels=pred_dicts[0]['pred_seg_labels'],
                point_size=2.0
            )
            
            print('Time cost per batch: %s' % (round((time_end - time_start) / 10, 3)))

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
