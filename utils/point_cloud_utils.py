import numpy as np
import torch

from utils.common_utils import check_numpy_to_torch


def rotate_points_along_z(points, angle):
    """

    Args:
        points: ndarray, [B, N, 3 + C]
        angle: ndarray, [B], angle along z-axis, angle increases x ==> y

    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)

    return points_rot.numpy() if is_numpy else points_rot


def mask_points_by_range(points, limit_range):
    """

    Args:
        points: ndarray, [N, 3 + C]
        limit_range: (xmin, ymin, zmin, xmax, ymax, zmax)

    Returns:

    """
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])

    return points[mask]


def get_fov_flag(points, image_shape, calib):
    """

    Args:
        points: ndarray, [N, 3 + C]
        image_shape: ndarray, [2], H and W
        calib: kitti_calibration_utils.Calibration

    Returns:

    """
    pts_rect = calib.lidar_to_rect(points[:, 0:3])
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)

    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < image_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < image_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)

    mask = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return mask
