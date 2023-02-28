import os
import argparse
import yaml
import tqdm
import matplotlib.pyplot as plt  # for WARNING: QApplication was not created in the main() thread.

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from data.kitti_dataset import KITTIDataset
from utils import opencv_vis_utils
from utils import open3d_vis_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/configs/ResNet_VFE.yaml',
                        help='path to the config file')
    parser.add_argument('--split', type=str, default='train',
                        help='must be chosen from ["train", "val", "trainval", "test"]')
    parser.add_argument('--augment_data', action='store_true', default=False,
                        help='whether to use random data augmentation')
    parser.add_argument('--show_boxes', action='store_true', default=False,
                        help='whether to show boxes')
    parser.add_argument('--onto_image', action='store_true', default=False,
                        help='whether to show the RGB image')
    parser.add_argument('--onto_range_image', action='store_true', default=False,
                        help='whether to show the range image')
    parser.add_argument('--sample_idx', type=str, default=None,
                        help='index of the sample')
    args = parser.parse_args()
    return args


def visualize(dataset, args, frame_id, image, range_image, colored_points, boxes, names):
    calib = dataset.get_calib(frame_id)

    if args.onto_image:
        image = image[:, :, ::-1]  # BGR image
        image = opencv_vis_utils.normalize_img(image)
        opencv_vis_utils.draw_scene(
            image,
            calib,
            boxes3d=boxes if args.show_boxes else None,
            names=names if args.show_boxes else None,
            window_name=frame_id,
        )
    elif args.onto_range_image:
        range_image = range_image.transpose(1, 2, 0)[:, :, -3:][:, :, ::-1]  # BGR image
        range_image = opencv_vis_utils.normalize_img(range_image)
        points = colored_points[:, 0:3]
        boxes2d = dataset.range_convertor.get_range_boxes_in_front_image(points, boxes)
        opencv_vis_utils.draw_scene(
            range_image,
            calib,
            boxes2d=boxes2d if args.show_boxes else None,
            names=names if args.show_boxes else None,
            window_name=frame_id,
        )
    else:
        points = colored_points[:, 0:3]
        point_colors = colored_points[:, 4:7]
        open3d_vis_utils.draw_scene(
            points,
            boxes3d=boxes if args.show_boxes else None,
            names=names if args.show_boxes else None,
            point_colors=point_colors,
            window_name=frame_id,
        )


def run(dataset, args, data_dict):
    frame_id = data_dict['frame_id']

    image = dataset.get_image(frame_id)
    range_image = data_dict['range_image']
    colored_points = data_dict['colored_points']

    if args.show_boxes:
        boxes = data_dict['gt_boxes'][:, :7]
        names = [dataset.class_names[int(k - 1)] for k in data_dict['gt_boxes'][:, 7]]
    else:
        boxes = None
        names = None

    visualize(
        dataset, args, frame_id, image, range_image, colored_points, boxes, names
    )


if __name__ == '__main__':
    args = parse_config()
    assert os.path.exists(args.cfg_file)
    cfg = yaml.load(open(args.cfg_file, 'r'), Loader=yaml.Loader)

    if args.onto_image and args.augment_data:
        args.augment_data = False
        print('Automatically set args.augment_data as False, '
              'because the augmented objects cannot be shown in the RGB image.')

    if cfg['dataset']['type'] == 'KITTI':
        dataset = KITTIDataset(cfg['dataset'], split=args.split, augment_data=args.augment_data)
    else:
        raise NotImplementedError

    if args.sample_idx is not None:
        assert args.sample_idx in dataset.id_list
        i = dataset.id_list.index(args.sample_idx)
        run(dataset, args, dataset[i])
    else:
        progress_bar = tqdm.tqdm(total=len(dataset), dynamic_ncols=True, leave=True, desc='samples')
        for i in range(len(dataset)):
            run(dataset, args, dataset[i])
            progress_bar.update()
        progress_bar.close()
