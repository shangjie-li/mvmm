import numpy as np

from utils.box_utils import boxes3d_lidar_to_camera
from utils.box_utils import boxes3d_lidar_to_image


def decode_detections(data_dict):
    det = {}
    batch_size = data_dict['batch_size']

    for i in range(batch_size):
        frame_id = data_dict['frame_id'][i]
        calib = data_dict['calib'][i]
        image_shape = data_dict['image_shape'][i]
        boxes3d_lidar = data_dict['pred_boxes'][i].cpu().numpy()
        scores = data_dict['pred_scores'][i].cpu().numpy().reshape(-1, 1)
        classes = data_dict['pred_classes'][i].cpu().numpy().reshape(-1, 1)

        boxes3d_camera = boxes3d_lidar_to_camera(boxes3d_lidar, calib)
        boxes2d = boxes3d_lidar_to_image(boxes3d_lidar, calib, image_shape)

        locs3d = boxes3d_camera[:, 0:3]
        sizes3d = boxes3d_camera[:, 3:6]
        rys = boxes3d_camera[:, 6:7]
        alphas = -np.arctan2(locs3d[:, 0:1], locs3d[:, 2:3]) + rys
        locs3d[:, 1] += sizes3d[:, 0] / 2

        classes -= 1
        det[frame_id] = np.concatenate(
            [classes, alphas, boxes2d, sizes3d, locs3d, rys, scores],
            axis=-1
        )

    return det
