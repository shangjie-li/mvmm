import argparse
import glob
from pathlib import Path
import time
import os

import open3d
from utils import open3d_vis_utils as V

import numpy as np
import torch

from data import cfg, cfg_from_yaml_file
from data import KittiDataset
from mvmm import build_network, load_data_to_gpu
from utils import common_utils


class DemoDataset(KittiDataset):
    def __init__(self, dataset_cfg, class_names, training=True, data_path=None, logger=None, ext='.bin'):
        """
        Args:
            dataset_cfg:
            class_names:
            training:
            data_path:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, logger=logger
        )
        self.data_path = data_path
        self.ext = ext
        file_list = glob.glob(str(data_path / f'*{self.ext}')) if self.data_path.is_dir() else [self.data_path]
        file_list.sort()
        self.sample_file_list = file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            sample_idx = os.path.basename(self.sample_file_list[index])[:-4]
            colored_points = self.get_colored_points_in_fov(sample_idx)
        else:
            raise NotImplementedError

        input_dict = {
            'points': colored_points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/config.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='data/kitti/training/velodyne/000008.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of PointPillars-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        data_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    
    print()
    print('<<< model >>>')
    print(model)
    print()
    
    """
    <<< model >>>
    PointPillar(
      (vfe): PillarVFE(
        (pfn_layers): ModuleList(
          (0): PFNLayer(
            (linear): Linear(in_features=13, out_features=64, bias=False)
            (norm): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          )
        )
      )
      (map_to_bev_module): PointPillarScatter()
      (backbone_2d): BaseBEVBackbone(
        (blocks): ModuleList(
          (0): Sequential(
            (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
            (1): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), bias=False)
            (2): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (3): ReLU()
            (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (5): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (6): ReLU()
            (7): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (8): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (9): ReLU()
            (10): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (11): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (12): ReLU()
          )
          (1): Sequential(
            (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
            (1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), bias=False)
            (2): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (3): ReLU()
            (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (5): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (6): ReLU()
            (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (8): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (9): ReLU()
            (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (11): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (12): ReLU()
            (13): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (14): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (15): ReLU()
            (16): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (17): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (18): ReLU()
          )
          (2): Sequential(
            (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
            (1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), bias=False)
            (2): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (3): ReLU()
            (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (5): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (6): ReLU()
            (7): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (8): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (9): ReLU()
            (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (11): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (12): ReLU()
            (13): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (14): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (15): ReLU()
            (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (17): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (18): ReLU()
          )
        )
        (deblocks): ModuleList(
          (0): Sequential(
            (0): ConvTranspose2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (1): Sequential(
            (0): ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (2): Sequential(
            (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(4, 4), bias=False)
            (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
      )
      (dense_head): AnchorHeadSingle(
        (cls_loss_func): SigmoidFocalClassificationLoss()
        (reg_loss_func): WeightedSmoothL1Loss()
        (dir_loss_func): WeightedCrossEntropyLoss()
        (conv_cls): Conv2d(384, 18, kernel_size=(1, 1), stride=(1, 1))
        (conv_box): Conv2d(384, 42, kernel_size=(1, 1), stride=(1, 1))
        (conv_dir_cls): Conv2d(384, 12, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    """
    
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
            
            """
            <<< data_dict >>>
            <<< data_dict >>>
            points <class 'numpy.ndarray'> (17089, 8)
            [[ 0.00000000e+00  2.15540009e+01  2.80000009e-02 ...  2.11764708e-01
               2.90196091e-01  1.25490203e-01]
             [ 0.00000000e+00  2.12399998e+01  9.39999968e-02 ...  1.17647061e-02
               1.01960786e-01  1.33333340e-01]
             [ 0.00000000e+00  2.10559998e+01  1.58999994e-01 ...  2.74509817e-01
               2.35294119e-01  1.37254909e-01]
             ...
             [ 0.00000000e+00  6.31500006e+00 -3.09999995e-02 ...  8.11764717e-01
               7.88235307e-01  6.39215708e-01]
             [ 0.00000000e+00  6.30900002e+00 -2.09999997e-02 ...  8.94117653e-01
               7.45098054e-01  6.58823550e-01]
             [ 0.00000000e+00  6.31099987e+00 -1.00000005e-03 ...  8.11764717e-01
               7.49019623e-01  8.19607854e-01]]
            frame_id <class 'numpy.ndarray'> (1,)
            [0]
            voxels <class 'numpy.ndarray'> (3941, 32, 7)
            [[[ 2.15540009e+01  2.80000009e-02  9.38000023e-01 ...  2.11764708e-01
                2.90196091e-01  1.25490203e-01]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              ...
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]]
            
             [[ 2.12399998e+01  9.39999968e-02  9.26999986e-01 ...  1.17647061e-02
                1.01960786e-01  1.33333340e-01]
              [ 2.11480007e+01  3.59999985e-02  7.90000021e-01 ...  2.70588249e-01
                2.35294119e-01  9.41176489e-02]
              [ 2.12159996e+01  1.70000009e-02  6.90999985e-01 ...  3.29411775e-01
                1.76470593e-01  5.88235296e-02]
              ...
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]]
            
             [[ 2.10559998e+01  1.58999994e-01  9.21000004e-01 ...  2.74509817e-01
                2.35294119e-01  1.37254909e-01]
              [ 2.10720005e+01  1.01999998e-01  7.87999988e-01 ...  2.50980407e-01
                2.07843140e-01  8.23529437e-02]
              [ 2.10979996e+01  1.15999997e-01  6.89000010e-01 ...  3.25490206e-01
                3.13725501e-01  1.37254909e-01]
              ...
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]]
            
             ...
            
             [[ 6.35799980e+00 -4.62000012e-01 -1.66499996e+00 ...  9.21568632e-01
                7.41176486e-01  7.13725507e-01]
              [ 6.34399986e+00 -4.41000015e-01 -1.66100001e+00 ...  8.27450991e-01
                8.00000012e-01  7.84313738e-01]
              [ 6.35300016e+00 -4.21000004e-01 -1.66299999e+00 ...  9.52941179e-01
                8.50980401e-01  7.92156875e-01]
              ...
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]]
            
             [[ 6.35400009e+00 -3.10999990e-01 -1.66199994e+00 ...  6.07843161e-01
                7.56862760e-01  6.43137276e-01]
              [ 6.34499979e+00 -2.91000009e-01 -1.65900004e+00 ...  7.45098054e-01
                7.09803939e-01  5.88235319e-01]
              [ 6.35099983e+00 -2.80999988e-01 -1.66100001e+00 ...  6.43137276e-01
                7.29411781e-01  5.92156887e-01]
              ...
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]]
            
             [[ 6.32700014e+00 -1.50000006e-01 -1.65199995e+00 ...  9.01960790e-01
                7.84313738e-01  6.90196097e-01]
              [ 6.32299995e+00 -1.29999995e-01 -1.65100002e+00 ...  8.27450991e-01
                7.60784328e-01  6.74509823e-01]
              [ 6.32600021e+00 -1.11000001e-01 -1.65199995e+00 ...  7.56862760e-01
                7.41176486e-01  6.11764729e-01]
              ...
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]]]
            voxel_coords <class 'numpy.ndarray'> (3941, 4)
            [[  0   0 248 134]
             [  0   0 248 132]
             [  0   0 248 131]
             ...
             [  0   0 245  39]
             [  0   0 246  39]
             [  0   0 247  39]]
            voxel_num_points <class 'numpy.ndarray'> (3941,)
            [ 1 10 11 ...  8  9  9]
            batch_size <class 'int'>
            1
            """
            
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
            
            """
            <<< pred_dicts[0] >>>
            pred_boxes <class 'torch.Tensor'> torch.Size([22, 7])
            tensor([[ 14.7927,  -1.1235,  -0.8008,   3.7601,   1.5825,   1.4473,   5.9594],
                    [  6.4745,  -3.8800,  -0.9971,   3.1546,   1.4373,   1.3902,   5.9903],
                    [  8.1327,   1.2404,  -0.8074,   3.6305,   1.5564,   1.5173,   2.8296],
                    [ 33.3620,  -7.0344,  -0.5197,   3.9533,   1.6519,   1.6176,   2.8461],
                    [ 20.1866,  -8.4656,  -0.8792,   2.2816,   1.4797,   1.4621,   5.9293],
                    [  3.8859,   2.5997,  -0.8412,   3.3783,   1.4558,   1.4905,   6.0394],
                    [ 40.9987,  -9.7844,  -0.5807,   3.7816,   1.5878,   1.4781,   5.9194],
                    [ 28.7435,  -1.5721,  -0.5425,   3.9912,   1.6049,   1.5834,   1.2417],
                    [ 55.5531, -20.2295,  -0.5304,   4.4419,   1.7828,   1.6399,   2.8215],
                    [ 24.9865, -10.1664,  -0.8357,   3.6846,   1.5918,   1.4164,   5.9303],
                    [ 29.8550, -14.0356,  -0.6830,   0.7794,   0.5863,   1.6656,   4.8024],
                    [ 37.2468,  -6.0335,  -0.4151,   0.7402,   0.5876,   1.7614,   6.1056],
                    [ 34.0729,  -4.9580,  -0.4143,   0.7952,   0.6141,   1.7755,   6.3121],
                    [ 40.5224,  -7.1451,  -0.4537,   1.7573,   0.5311,   1.6584,   2.9681],
                    [ 10.3686,   5.3558,  -0.4740,   0.6035,   0.7284,   1.7393,   3.0756],
                    [ 18.6827,   0.2598,  -0.7684,   0.7151,   0.6531,   1.6487,   6.7219],
                    [ 33.4353, -15.3233,  -0.6375,   1.5949,   0.4325,   1.6806,   6.1247],
                    [ 53.7474, -16.1937,  -0.3424,   1.5919,   0.6748,   1.7864,   2.9927],
                    [ 52.8253, -21.7382,  -0.4714,   3.8221,   1.5238,   1.5853,   2.9257],
                    [ 13.4324,   4.3186,  -0.4972,   1.7780,   0.5214,   1.7779,   5.8251],
                    [ 30.5718,  -3.7013,  -0.4829,   1.7198,   0.5163,   1.7141,   6.1299],
                    [ 56.1591,  -7.6231,   0.3299,   4.5915,   1.7687,   2.1597,   5.9250]],
                   device='cuda:0')
            pred_scores <class 'torch.Tensor'> torch.Size([22])
            tensor([0.9648, 0.9247, 0.9238, 0.8865, 0.7283, 0.6788, 0.6772, 0.5932, 0.5745,
                    0.5302, 0.3800, 0.2912, 0.2622, 0.2426, 0.2329, 0.2288, 0.2215, 0.1988,
                    0.1580, 0.1357, 0.1313, 0.1272], device='cuda:0')
            pred_labels <class 'torch.Tensor'> torch.Size([22])
            tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 2, 2, 3, 3, 1, 3, 3, 1],
                   device='cuda:0')
            """

            V.draw_scenes(
                points=data_dict['points'][:, 1:].cpu().numpy(),
                ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'],
                ref_labels=pred_dicts[0]['pred_labels'],
                point_colors=data_dict['points'][:, -3:].cpu().numpy()
            )
                
            print('Time cost per batch: %s' % (round((time_end - time_start) / 10, 3)))

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
