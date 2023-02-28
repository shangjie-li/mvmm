import os
import datetime
import argparse

import kitti_common
import eval


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='the path to your result folder')
    parser.add_argument('--label_dir', type=str, default='../kitti/training/label_2',
                        help='the path to the ground truth label folder')
    parser.add_argument('--label_split_file', type=str, default='../kitti/training/ImageSets/val.txt',
                        help='the path to the dataset split file')
    parser.add_argument('--current_classes', type=str, default='0,1,2',
                        help='a filter for desired classes, e.g., 0,1,2 (split by a comma)')
    parser.add_argument('--use_ldf_eval', action='store_true', default=False,
                        help='whether to use long-distance-focused evaluation')
    args = parser.parse_args()
    return args


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(result_dir, label_dir, label_split_file, current_classes, use_ldf_eval=False, score_thresh=-1):
    dt_annos = kitti_common.get_label_annos(result_dir)
    if score_thresh > 0:
        dt_annos = kitti_common.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti_common.get_label_annos(label_dir, val_image_ids)
    ap_result_str = eval.get_official_eval_result(gt_annos, dt_annos, current_classes,
                                             use_ldf_eval=use_ldf_eval,
                                             print_info=True)
    print(ap_result_str)

    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    log_file = 'log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    with open(os.path.join(save_dir, log_file), 'a') as f:
        f.write(ap_result_str)


if __name__ == '__main__':
    args = parse_config()
    assert os.path.isdir(args.result_dir)
    assert os.path.isdir(args.label_dir)
    assert os.path.isfile(args.label_split_file)
    args.current_classes = list(map(int, args.current_classes.split(',')))
    evaluate(args.result_dir, args.label_dir, args.label_split_file,
             args.current_classes, use_ldf_eval=args.use_ldf_eval)
