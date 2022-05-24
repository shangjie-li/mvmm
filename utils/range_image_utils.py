import torch
import numpy as np

from . import common_utils, box_utils


class RangeConvertor(object):
    def __init__(self, config):
        self.full_size = config.FULL_SIZE
        self.front_size = config.FRONT_SIZE
        self.fov_up = config.LIDAR_FOV_UP * torch.pi / 180
        self.fov_down = config.LIDAR_FOV_DOWN * torch.pi / 180
        self.fov = self.fov_up + self.fov_down
    
    def get_pixel_coords(self, points):
        assert len(points.shape) == 2 and points.shape[0] > 0
        points, is_numpy = common_utils.check_numpy_to_torch(points)
        xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
        rs = torch.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
        us = 0.5 * (1 - torch.atan2(ys, xs) / torch.pi) * self.full_size[1]
        vs = (1 - (torch.arcsin(zs / rs) + self.fov_down) / self.fov) * self.full_size[0]
        us = (torch.clip(us, min=0, max=self.full_size[1] - 1)).type(torch.long)
        vs = (torch.clip(vs, min=0, max=self.full_size[0] - 1)).type(torch.long)
        center_u = int((us.max() + us.min()) / 2)
        center_v = int((vs.max() + vs.min()) / 2)
        if is_numpy:
            return us.numpy(), vs.numpy(), center_u, center_v
        else:
            return us, vs, center_u, center_v
    
    def get_u0(self, cu):
        u0 = cu - self.front_size[1] / 2
        u0 = min(max(0, u0), self.full_size[1] - self.front_size[1])
        return int(u0)
    
    def get_v0(self, cv):
        v0 = cv - self.front_size[0] / 2
        v0 = min(max(0, v0), self.full_size[0] - self.front_size[0])
        return int(v0)
    
    def get_range_image(self, points, features):
        assert len(features.shape) == 2
        assert points.shape[0] == features.shape[0]
        points, _ = common_utils.check_numpy_to_torch(points)
        features, is_numpy = common_utils.check_numpy_to_torch(features)
        image_shape = (features.shape[1], self.full_size[0], self.full_size[1])
        if is_numpy:
            full_image = torch.zeros(image_shape).float()
        else:
            full_image = torch.zeros(image_shape, dtype=features.dtype, device=features.device)
        us, vs, cu, cv = self.get_pixel_coords(points)
        full_image[:, vs, us] = features.t()
        u0, v0 = self.get_u0(cu), self.get_v0(cv)
        front_image = full_image[:, v0:v0 + self.front_size[0], u0:u0 + self.front_size[1]]
        return front_image.numpy() if is_numpy else front_image
    
    def get_range_features(self, points, front_image):
        assert len(front_image.shape) == 3
        points, _ = common_utils.check_numpy_to_torch(points)
        front_image, is_numpy = common_utils.check_numpy_to_torch(front_image)
        image_shape = (front_image.shape[0], self.full_size[0], self.full_size[1])
        if is_numpy:
            full_image = torch.zeros(image_shape).float()
        else:
            full_image = torch.zeros(image_shape, dtype=front_image.dtype, device=front_image.device)
        us, vs, cu, cv = self.get_pixel_coords(points)
        u0, v0 = self.get_u0(cu), self.get_v0(cv)
        full_image[:, v0:v0 + self.front_size[0], u0:u0 + self.front_size[1]] = front_image
        range_features = full_image[:, vs, us].t()
        return range_features.numpy() if is_numpy else range_features
    
    def get_range_boxes(self, boxes_lidar, cu0=0, cv0=0):
        boxes_lidar, is_numpy = common_utils.check_numpy_to_torch(boxes_lidar)
        corners = box_utils.boxes_to_corners_3d(boxes_lidar)
        boxes = []
        for i in range(corners.shape[0]):
            us, vs, _, _ = self.get_pixel_coords(corners[i])
            us -= self.get_u0(cu0)
            vs -= self.get_v0(cv0)
            box = torch.tensor((us.min(), vs.min(), us.max(), vs.max())).float()
            boxes.append(box.reshape(1, 4))
        boxes = torch.cat(boxes, dim=0) if len(boxes) > 0 else torch.tensor([]).float().reshape(-1, 4)
        return boxes.numpy() if is_numpy else boxes
    
    def get_range_iou(self, box, existed_boxes):
        box, _ = common_utils.check_numpy_to_torch(box)
        existed_boxes, _ = common_utils.check_numpy_to_torch(existed_boxes)
        x1, y1, x2, y2 = existed_boxes[:, 0], existed_boxes[:, 1], existed_boxes[:, 2], existed_boxes[:, 3]
        area = (x2 - x1) * (y2 - y1)
        b = box.squeeze()
        xx1 = torch.clip(x1, min=b[0], max=torch.inf)
        yy1 = torch.clip(y1, min=b[1], max=torch.inf)
        xx2 = torch.clip(x2, min=0, max=b[2])
        yy2 = torch.clip(y2, min=0, max=b[3])
        w = torch.clip((xx2 - xx1), min=0, max=torch.inf)
        h = torch.clip((yy2 - yy1), min=0, max=torch.inf)
        inter = w * h
        union = (b[2] - b[0]) * (b[3] - b[1]) + area - inter + 1e-5
        iou = inter / union
        return iou
    
    def get_range_indicator(self):
        return torch.zeros((self.full_size[0], self.full_size[1]), dtype=torch.long)

