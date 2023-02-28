import numpy as np
import torch

from utils.box_utils import boxes3d_to_corners3d


class RangeConvertor():
    def __init__(self, cfg):
        self.full_size = cfg['full_size']  # W and H
        self.front_size = cfg['front_size']  # W and H
        self.fov_up = 2.0 * 3.14159 / 180
        self.fov_down = 24.8 * 3.14159 / 180
        self.fov = self.fov_up + self.fov_down
    
    def get_pixel_coords_torch(self, points):
        """
        Get pixel coordinates of lidar points in the full range image, and meantime, return the coordinates with
        the center location. By this way, it will be easier to get the front range image, especially when
        the lidar points are rotated along the z axis during the data augmentation.

        Args:
            points: tensor, [N, 3 + C]

        Returns:

        """
        xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
        rs = torch.sqrt(xs ** 2 + ys ** 2 + zs ** 2)

        us = 0.5 * (1 - torch.atan2(ys, xs) / torch.pi) * self.full_size[0]
        vs = (1 - (torch.arcsin(zs / rs) + self.fov_down) / self.fov) * self.full_size[1]
        us = torch.clip(us, min=0, max=self.full_size[0] - 1).long()
        vs = torch.clip(vs, min=0, max=self.full_size[1] - 1).long()

        center_u = int((us.max() + us.min()) / 2)
        center_v = int((vs.max() + vs.min()) / 2)

        return us, vs, center_u, center_v

    def get_pixel_coords_numpy(self, points):
        """
        Get pixel coordinates of lidar points in the full range image, and meantime, return the coordinates with
        the center location. By this way, it will be easier to get the front range image, especially when
        the lidar points are rotated along the z axis during the data augmentation.

        Args:
            points: ndarray, [N, 3 + C]

        Returns:

        """
        xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
        rs = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)

        us = 0.5 * (1 - np.arctan2(ys, xs) / np.pi) * self.full_size[0]
        vs = (1 - (np.arcsin(zs / rs) + self.fov_down) / self.fov) * self.full_size[1]
        us = np.clip(us, a_min=0, a_max=self.full_size[0] - 1).astype(np.int)
        vs = np.clip(vs, a_min=0, a_max=self.full_size[1] - 1).astype(np.int)

        center_u = int((us.max() + us.min()) / 2)
        center_v = int((vs.max() + vs.min()) / 2)

        return us, vs, center_u, center_v

    def get_front_image_origin(self, center_u, center_v):
        u0 = center_u - self.front_size[0] // 2
        u0 = min(max(0, u0), self.full_size[0] - self.front_size[0])

        v0 = center_v - self.front_size[1] // 2
        v0 = min(max(0, v0), self.full_size[1] - self.front_size[1])

        return u0, v0
    
    def get_range_image(self, points, features):
        assert len(points.shape) == 2 and len(features.shape) == 2
        assert points.shape[0] == features.shape[0]

        image_shape = (features.shape[1], self.full_size[1], self.full_size[0])
        full_image = np.zeros(image_shape, dtype=np.float32)

        us, vs, cu, cv = self.get_pixel_coords_numpy(points)
        full_image[:, vs, us] = features.transpose()

        u0, v0 = self.get_front_image_origin(cu, cv)
        front_image = full_image[:, v0:v0 + self.front_size[1], u0:u0 + self.front_size[0]]

        return front_image
    
    def get_range_features(self, points, front_image):
        image_shape = (front_image.shape[0], self.full_size[1], self.full_size[0])
        full_image = torch.zeros(image_shape, dtype=front_image.dtype, device=front_image.device)

        us, vs, cu, cv = self.get_pixel_coords_torch(points)
        u0, v0 = self.get_front_image_origin(cu, cv)

        full_image[:, v0:v0 + self.front_size[1], u0:u0 + self.front_size[0]] = front_image
        range_features = full_image[:, vs, us].t()

        return range_features

    def get_range_boxes_in_full_image(self, boxes_lidar):
        boxes = []
        corners = boxes3d_to_corners3d(boxes_lidar)
        num_boxes = boxes_lidar.shape[0]
        for i in range(num_boxes):
            us, vs, _, _ = self.get_pixel_coords_numpy(corners[i])
            box = [us.min(), vs.min(), us.max(), vs.max()]
            boxes.append(box)

        return np.array(boxes).reshape(-1, 4)

    def get_range_boxes_in_front_image(self, points, boxes_lidar):
        _, _, cu, cv = self.get_pixel_coords_numpy(points)
        u0, v0 = self.get_front_image_origin(cu, cv)

        boxes = []
        corners = boxes3d_to_corners3d(boxes_lidar)
        num_boxes = boxes_lidar.shape[0]
        for i in range(num_boxes):
            us, vs, _, _ = self.get_pixel_coords_numpy(corners[i])
            us -= u0
            vs -= v0
            box = [us.min(), vs.min(), us.max(), vs.max()]
            boxes.append(box)

        return np.array(boxes).reshape(-1, 4)
    
    def get_range_indicator(self):
        return np.zeros((self.full_size[1], self.full_size[0]))
