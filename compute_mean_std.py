import argparse
import numpy as np

from data import cfg, cfg_from_yaml_file
from data import KittiDataset


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/config/ResNet_VFE.yaml',
        help='specify the config for training')
    parser.add_argument('--training', action='store_true', default=False,
        help='whether to use training mode')
    parser.add_argument('--data_augmentation', action='store_true', default=False,
        help='whether to use data augmentation')

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
    
    mean_xs, mean_ys, mean_zs, mean_ranges, mean_intensities, mean_rs, mean_gs, mean_bs = \
        [], [], [], [], [], [], [], []
    std_xs, std_ys, std_zs, std_ranges, std_intensities, std_rs, std_gs, std_bs = \
        [], [], [], [], [], [], [], []
    
    for i in range(len(dataset)):
        print('Reading: %d / %d...' % (i + 1, len(dataset)))
        data_dict = dataset[i]
        
        xs = data_dict['colored_points'][:, 0:1]
        ys = data_dict['colored_points'][:, 1:2]
        zs = data_dict['colored_points'][:, 2:3]
        ranges = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
        intensities = data_dict['colored_points'][:, 3:4]
        rs = data_dict['colored_points'][:, 4:5]
        gs = data_dict['colored_points'][:, 5:6]
        bs = data_dict['colored_points'][:, 6:7]
        
        mean_xs.append(xs.mean())
        mean_ys.append(ys.mean())
        mean_zs.append(zs.mean())
        mean_ranges.append(ranges.mean())
        mean_intensities.append(intensities.mean())
        mean_rs.append(rs.mean())
        mean_gs.append(gs.mean())
        mean_bs.append(bs.mean())
        
        std_xs.append(xs.std())
        std_ys.append(ys.std())
        std_zs.append(zs.std())
        std_ranges.append(ranges.std())
        std_intensities.append(intensities.std())
        std_rs.append(rs.std())
        std_gs.append(gs.std())
        std_bs.append(bs.std())
    
    mean_x = round(np.mean(mean_xs), 2)
    mean_y = round(np.mean(mean_ys), 2)
    mean_z = round(np.mean(mean_zs), 2)
    mean_range = round(np.mean(mean_ranges), 2)
    mean_intensity = round(np.mean(mean_intensities), 2)
    mean_r = round(np.mean(mean_rs), 2)
    mean_g = round(np.mean(mean_gs), 2)
    mean_b = round(np.mean(mean_bs), 2)
    
    std_x = round(np.mean(std_xs), 2)
    std_y = round(np.mean(std_ys), 2)
    std_z = round(np.mean(std_zs), 2)
    std_range = round(np.mean(std_ranges), 2)
    std_intensity = round(np.mean(std_intensities), 2)
    std_r = round(np.mean(std_rs), 2)
    std_g = round(np.mean(std_gs), 2)
    std_b = round(np.mean(std_bs), 2)
    
    print()
    print('mean_x:', mean_x, 'std_x:', std_x)
    print('mean_y:', mean_y, 'std_y:', std_y)
    print('mean_z:', mean_z, 'std_z:', std_z)
    print('mean_range:', mean_range, 'std_range:', std_range)
    print('mean_intensity:', mean_intensity, 'std_intensity:', std_intensity)
    print('mean_r:', mean_r, 'std_r:', std_r)
    print('mean_g:', mean_g, 'std_g:', std_g)
    print('mean_b:', mean_b, 'std_b:', std_b)

