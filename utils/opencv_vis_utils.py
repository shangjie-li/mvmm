import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

import torch
import numpy as np

from . import common_utils, box_utils


box_colormap = {
    'Car': (0, 1, 0),
    'Pedestrian': (1, 1, 0),
    'Cyclist': (0, 1, 1),
} # BGR


def draw_scenes(img, calib, gt_boxes=None, ref_boxes=None, ref_labels=None,
    window_name='result'):
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    
    img = img.copy()
    if gt_boxes is not None:
        img = draw_box(img, calib, gt_boxes, color=(0, 0, 1))
    if ref_boxes is not None:
        if ref_labels is not None:
            img = draw_box(img, calib, ref_boxes, ref_labels)
        else:
            img = draw_box(img, calib, ref_boxes, color=(0, 1, 0))
    
    cv2.imshow(window_name, img)
    
    # press 'Esc' to close this window, or 'Enter' to save the image
    key = cv2.waitKey(0)
    while key:
        if key == 27: # Esc:
            cv2.destroyWindow(window_name)
            break
        elif key == 13: # Enter
            cv2.imwrite(window_name + '.png', (img * 255.0).astype(np.uint8))
            cv2.destroyWindow(window_name)
            break
        else:
            key = cv2.waitKey(0)


def draw_box(img, calib, boxes, labels=None, color=(0, 0, 1), thickness=2):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        labels: [name]

    Returns:
    """
    for i in range(boxes.shape[0]):
        corners3d = box_utils.boxes_to_corners_3d(np.array([boxes[i]]))[0]
        pts_rect = calib.lidar_to_rect(corners3d)
        pts_img, _ = calib.rect_to_img(pts_rect) # [N', 2], [N']
        
        pts_img = pts_img.astype(np.int)
        color = box_colormap[labels[i]] if labels is not None else color
        
        cv2.line(img, (pts_img[0, 0], pts_img[0, 1]), (pts_img[5, 0], pts_img[5, 1]),color, thickness)
        cv2.line(img, (pts_img[1, 0], pts_img[1, 1]), (pts_img[4, 0], pts_img[4, 1]), color, thickness)
        
        for k in range(4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (pts_img[i, 0], pts_img[i, 1]), (pts_img[j, 0], pts_img[j, 1]), color, thickness)
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (pts_img[i, 0], pts_img[i, 1]), (pts_img[j, 0], pts_img[j, 1]), color, thickness)
            i, j = k, k + 4
            cv2.line(img, (pts_img[i, 0], pts_img[i, 1]), (pts_img[j, 0], pts_img[j, 1]), color, thickness)
    
    return img
