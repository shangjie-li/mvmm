import copy
import argparse
import pickle
import matplotlib.pyplot as plt

import numpy as np

from data import KittiDataset
from utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--split', type=str, default='trainval.txt',
        help='specify the split for statistics, e.g., train.txt, val.txt, or trainval.txt')
    parser.add_argument('--class_names', type=str, default='Car/Pedestrian/Cyclist',
        help='specify the class names, split by the slash')

    args = parser.parse_args()

    return args


class SimplifiedKittiDataset(KittiDataset):
    def __init__(self, class_names, split):
        super().__init__(dataset_cfg=None, class_names=class_names)

        self.set_split(split)
        self.kitti_infos = []
        with open(self.info_file, 'rb') as f:
            infos = pickle.load(f)
        self.kitti_infos.extend(infos)

    def __len__(self):
        return len(self.kitti_infos)

    def __getitem__(self, index):
        info = copy.deepcopy(self.kitti_infos[index])
        data_dict = {}

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')  # exclude class: DontCare
            data_dict.update({
                'gt_boxes': annos['gt_boxes_lidar'],
                'gt_names': annos['name'],
            })

        if data_dict.get('gt_boxes', None) is not None:
            # Filter by the class: Car, Pedestrian, Cyclist
            mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            data_dict['gt_names'] = data_dict['gt_names'][mask]

            # Limit heading to [-pi, pi)
            data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
                data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
            )

            # Merge gt_boxes and gt_classes: Car - 1, Pedestrian - 2, Cyclist - 3
            gt_classes = [self.class_names.index(n) + 1 for n in data_dict['gt_names']]
            data_dict['gt_boxes'] = np.concatenate(
                [data_dict['gt_boxes'], np.array(gt_classes).reshape(-1, 1).astype(np.float32)], axis=1
            )  # (M, 8), [x, y, z, l, w, h, heading, class_id] in lidar coordinates

        data_dict.pop('gt_names', None)
        return data_dict


if __name__ == '__main__':
    args = parse_config()
    class_names = args.class_names.split('/')

    dataset = SimplifiedKittiDataset(
        class_names=class_names,
        split=args.split
    )

    # Rows represent classes
    # Cols represent distances, index 0: 0m - 10m, index 1: 10m - 20m, ..., index 9: 90m - 100m, index 10: > 100m
    size = 11
    numbers = np.zeros((len(class_names), size), np.int)

    for i in range(len(dataset)):
        data_dict = dataset[i]
        if data_dict.get('gt_boxes', None) is not None:
            gt_boxes = data_dict['gt_boxes']
            for j in range(gt_boxes.shape[0]):
                d = (gt_boxes[j][0] ** 2 + gt_boxes[j][1] ** 2) ** 0.5
                col = int(d / 10) if d < 100 else 10
                row = int(gt_boxes[j][7] - 1)
                numbers[row][col] += 1

    print(numbers)
    plt.figure(figsize=(12, 5))
    xs = np.arange(size)
    width = 0.8 / len(class_names)
    for k in range(len(class_names)):
        plt.bar(xs + k * width, numbers[k, :], width=width, label=class_names[k])
    xticks = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '>100']
    plt.xticks(xs + (len(class_names) // 2) * width, xticks)
    plt.xlabel('Distances (Meters)')
    plt.ylabel('Numbers')
    plt.legend()
    plt.show()
