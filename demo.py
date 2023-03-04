import os
import argparse
import yaml
import tqdm
import torch
import matplotlib.pyplot as plt  # for WARNING: QApplication was not created in the main() thread.

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from data.kitti_dataset import KITTIDataset
from mvmm import build_model
from helpers.checkpoint_helper import load_checkpoint
from dataset_player import visualize


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/configs/ResNet_VFE.yaml',
                        help='path to the config file')
    parser.add_argument('--split', type=str, default=None,
                        help='must be chosen from ["train", "val", "trainval", "test"]')
    parser.add_argument('--score_thresh', type=float, default=None,
                        help='score threshold for filtering detections')
    parser.add_argument('--nms_thresh', type=float, default=None,
                        help='NMS threshold for filtering detections')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to the checkpoint')
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


def run(model, dataset, args, cfg, data_dict, device):
    frame_id = data_dict['frame_id']

    image = dataset.get_image(frame_id)
    range_image = data_dict['range_image']
    colored_points = data_dict['colored_points']

    batch_dict = dataset.collate_batch([data_dict])
    batch_dict = dataset.load_data_to_gpu(batch_dict, device)

    batch_dict = model(batch_dict, score_thresh=cfg['tester']['score_thresh'], nms_thresh=cfg['tester']['nms_thresh'])

    pred_boxes = batch_dict['pred_boxes'][0].cpu().numpy()  # [M, 7]
    pred_classes = batch_dict['pred_classes'][0].cpu().numpy()  # [M]
    pred_names = [dataset.class_names[int(k - 1)] for k in pred_classes]

    visualize(
        dataset, args, frame_id, image, range_image, colored_points, pred_boxes, pred_names
    )


if __name__ == '__main__':
    args = parse_config()
    assert os.path.exists(args.cfg_file)
    cfg = yaml.load(open(args.cfg_file, 'r'), Loader=yaml.Loader)

    if args.split is not None:
        cfg['tester']['split'] = args.split
    if args.score_thresh is not None:
        cfg['tester']['score_thresh'] = args.score_thresh
    if args.nms_thresh is not None:
        cfg['tester']['nms_thresh'] = args.nms_thresh
    if args.checkpoint is not None:
        cfg['tester']['checkpoint'] = args.checkpoint

    if cfg['dataset']['type'] == 'KITTI':
        dataset = KITTIDataset(cfg['dataset'], split=cfg['tester']['split'], is_training=False, augment_data=False)
    else:
        raise NotImplementedError

    model = build_model(cfg['model'], dataset)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    assert os.path.exists(cfg['tester']['checkpoint'])
    load_checkpoint(
        file_name=cfg['tester']['checkpoint'],
        model=model,
        optimizer=None,
        map_location=device,
        logger=None,
    )

    torch.set_grad_enabled(False)
    model.eval()

    if args.sample_idx is not None:
        assert args.sample_idx in dataset.id_list
        i = dataset.id_list.index(args.sample_idx)
        run(model, dataset, args, cfg, dataset[i], device)
    else:
        progress_bar = tqdm.tqdm(total=len(dataset), dynamic_ncols=True, leave=True, desc='samples')
        for i in range(len(dataset)):
            run(model, dataset, args, cfg, dataset[i], device)
            progress_bar.update()
        progress_bar.close()
