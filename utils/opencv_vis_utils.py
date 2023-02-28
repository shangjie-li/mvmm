import cv2
import numpy as np

from utils.box_utils import boxes3d_to_corners3d

box_colormap = {
    'Car': (0, 255, 0),
    'Pedestrian': (255, 255, 0),
    'Cyclist': (0, 255, 255),
}  # BGR


def normalize_img(img):
    """
    Normalize the image data to np.uint8 and image shape to [H, W, 3].

    Args:
        img: ndarray, [H, W] or [H, W, 1] or [H, W, 3]

    Returns:
        img: ndarray of uint8, [H, W, 3]

    """
    img = img.copy()
    img += -img.min() if img.min() < 0 else 0
    img = np.clip(img / img.max(), a_min=0., a_max=1.) * 255.
    img = img.astype(np.uint8)
    img = img[:, :, None] if len(img.shape) == 2 else img

    assert len(img.shape) == 3
    if img.shape[-1] == 1:
        return cv2.applyColorMap(img, cv2.COLORMAP_JET)
    elif img.shape[-1] == 3:
        return img
    else:
        raise NotImplementedError


def draw_scene(img, calib, boxes2d=None, boxes3d=None, names=None, window_name='image',
               wait_key=True, enter_to_save=True):
    """
    Show the image with 2D boxes or 3D boxes.

    Args:
        img: ndarray of uint8, [H, W, 3], BGR image
        calib: kitti_calibration_utils.Calibration
        boxes2d: ndarray of float32, [N, 4], (x1, y1, x2, y2) of bounding boxes
        boxes3d: ndarray of float32, [N, 7], (x, y, z, l, w, h, heading] in lidar coordinates
        names: list of str, name of each object
        window_name: str
        wait_key: bool
        enter_to_save: bool

    Returns:

    """
    if boxes2d is not None:
        img = draw_boxes2d(img, boxes2d, names)

    if boxes3d is not None:
        img = draw_boxes3d(img, calib, boxes3d, names)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, img.shape[1] + 100, img.shape[0] + 100)
    cv2.imshow(window_name, img)

    if wait_key:
        # Press 'Esc' to close this window, or 'Enter' to save the image.
        key = cv2.waitKey(0)
        while key:
            if key == 27:  # Esc
                cv2.destroyWindow(window_name)
                break
            elif enter_to_save and key == 13:  # Enter
                cv2.imwrite(window_name + '.png', img)
                cv2.destroyWindow(window_name)
                break
            else:
                key = cv2.waitKey(0)


def draw_boxes2d(img, boxes2d, names=None, color=(0, 255, 0), thickness=2):
    """
    Draw 2D boxes in the image.

    Args:
        img: ndarray of uint8, [H, W, 3], BGR image
        boxes2d: ndarray of float32, [N, 4], (x1, y1, x2, y2) of bounding boxes
        names: list of str, name of each object
        color: tuple
        thickness: int

    Returns:
        img: ndarray of uint8, [H, W, 3], BGR image

    """
    for i in range(boxes2d.shape[0]):
        u1, v1, u2, v2 = boxes2d[i].astype(np.int)

        if names is not None:
            color = box_colormap[names[i]]

        cv2.rectangle(img, (u1, v1), (u2, v2), color, thickness)

    return img


def draw_boxes3d(img, calib, boxes3d, names=None, color=(0, 255, 0), thickness=2):
    """
    Draw 3D boxes in the image.

    Args:
        img: ndarray of uint8, [H, W, 3], BGR image
        calib: kitti_calibration_utils.Calibration
        boxes3d: ndarray of float32, [N, 7], (x, y, z, l, w, h, heading] in lidar coordinates
        names: list of str, name of each object
        color: tuple
        thickness: int

    Returns:
        img: ndarray of uint8, [H, W, 3], BGR image

    """
    corners = boxes3d_to_corners3d(boxes3d)  # [N, 8, 3]
    for i in range(boxes3d.shape[0]):
        pts = corners[i]
        pts_img, pts_depth = calib.lidar_to_img(pts)

        if (pts_depth > 0).sum() < 8:
            continue

        if names is not None:
            color = box_colormap[names[i]]

        pts_img = pts_img.astype(np.int)
        cv2.line(img, (pts_img[0, 0], pts_img[0, 1]), (pts_img[5, 0], pts_img[5, 1]), color, thickness)
        cv2.line(img, (pts_img[1, 0], pts_img[1, 1]), (pts_img[4, 0], pts_img[4, 1]), color, thickness)
        for k in range(4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (pts_img[i, 0], pts_img[i, 1]), (pts_img[j, 0], pts_img[j, 1]), color, thickness)
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (pts_img[i, 0], pts_img[i, 1]), (pts_img[j, 0], pts_img[j, 1]), color, thickness)
            i, j = k, k + 4
            cv2.line(img, (pts_img[i, 0], pts_img[i, 1]), (pts_img[j, 0], pts_img[j, 1]), color, thickness)

    return img
