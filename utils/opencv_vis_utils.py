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
    'Car': (0, 255, 0),
    'Pedestrian': (255, 255, 0),
    'Cyclist': (0, 255, 255),
} # BGR


def draw_scenes(img, calib, gt_boxes=None, ref_boxes=None, ref_labels=None,
    gt_boxes2d=None, ref_boxes2d=None, window_name='result'):
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    if isinstance(gt_boxes2d, torch.Tensor):
        gt_boxes2d = gt_boxes2d.cpu().numpy()
    if isinstance(ref_boxes2d, torch.Tensor):
        ref_boxes2d = ref_boxes2d.cpu().numpy()
    
    img = img.copy()
    img = (np.clip(img / img.max(), a_min=0, a_max=1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET) if img.shape[2] == 1 else img
    
    if gt_boxes is not None:
        img = draw_box(img, calib, gt_boxes, color=(0, 0, 255))
    if ref_boxes is not None:
        if ref_labels is not None:
            img = draw_box(img, calib, ref_boxes, ref_labels)
        else:
            img = draw_box(img, calib, ref_boxes, color=(0, 255, 0))
    
    if gt_boxes2d is not None:
        img = draw_box2d(img, gt_boxes2d, color=(0, 0, 255))
    if ref_boxes2d is not None:
        if ref_labels is not None:
            img = draw_box2d(img, ref_boxes2d, ref_labels)
        else:
            img = draw_box2d(img, ref_boxes2d, color=(0, 255, 0))
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, img.shape[1] + 100, img.shape[0] + 100)
    cv2.imshow(window_name, img)
    
    # press 'Esc' to close this window, or 'Enter' to save the image
    key = cv2.waitKey(0)
    while key:
        if key == 27: # Esc:
            cv2.destroyWindow(window_name)
            break
        elif key == 13: # Enter
            cv2.imwrite(window_name + '.png', img)
            cv2.destroyWindow(window_name)
            break
        else:
            key = cv2.waitKey(0)


def draw_box(img, calib, boxes, labels=None, color=(0, 0, 255), thickness=2):
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
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect) # [8, 2], [8]
        if (pts_rect_depth > 0).sum() < 8:
            continue
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


def draw_box2d(img, boxes2d, labels=None, color=(0, 0, 255), thickness=2):
    for i in range(boxes2d.shape[0]):
        color = box_colormap[labels[i]] if labels is not None else color
        b = boxes2d[i].astype(np.int)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, thickness)
    return img
