import copy
import argparse
import pickle
import matplotlib.pyplot as plt

import numpy as np

from data import KittiDataset
from utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--split', type=str, default='val.txt',
        help='specify the split for statistics, e.g., train.txt, val.txt, or trainval.txt')
    parser.add_argument('--class_names', type=str, default='Car,Pedestrian,Cyclist',
        help='specify the class names, split by the comma')

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
            mask = np.array([n in self.class_names for n in annos['name']], dtype=np.bool)
            data_dict['location'] = annos['location'][mask]
            data_dict['name'] = annos['name'][mask]
        return data_dict


if __name__ == '__main__':
    args = parse_config()
    class_names = args.class_names.split(',')

    dataset = SimplifiedKittiDataset(
        class_names=class_names,
        split=args.split
    )

    MIN_DISTANCE = [0, 10, 20, 30, 40, 50, 60, 70]
    MAX_DISTANCE = [10, 20, 30, 40, 50, 60, 70, 80]

    # Rows represent classes
    # Cols represent distances, index 0: 0m - 10m, index 1: 10m - 20m, ..., index 7: 70m - 80m
    numbers = np.zeros((len(class_names), len(MAX_DISTANCE)), np.int)

    for i in range(len(dataset)):
        data_dict = dataset[i]
        if data_dict.get('name', None) is not None:
            for j in range(data_dict['name'].shape[0]):
                row = class_names.index(data_dict['name'][j])
                d = (data_dict['location'][j, 0] ** 2 + data_dict['location'][j, 2] ** 2) ** 0.5
                for l in range(len(MAX_DISTANCE)):
                    if d >= MIN_DISTANCE[l] and d < MAX_DISTANCE[l]:
                        numbers[row, l] += 1
                        break

    print(numbers)
    plt.figure(figsize=(12, 6))
    xs = np.arange(len(MAX_DISTANCE))
    class_names_with_num = []
    for k in range(len(class_names)):
        text = class_names[k] + ' ({:d})'.format(np.sum(numbers[k, :]))
        class_names_with_num.append(text)
    width = 0.8 / len(class_names)
    for k in range(len(class_names)):
        plt.bar(xs + k * width, numbers[k, :], width=width, label=class_names_with_num[k])
        for m in range(len(MAX_DISTANCE)):
            plt.text(xs[m] + k * width, numbers[k, m], str(numbers[k, m]), ha='center', fontsize=8)
    xticks = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80']
    plt.xticks(xs + (len(class_names) // 2) * width, xticks)
    plt.xlabel('Distances (meters)')
    plt.ylabel('Numbers')
    plt.title('Distance Distribution of Objects')
    plt.legend()
    plt.show()
    
