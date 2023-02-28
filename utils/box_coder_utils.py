import torch


class ResidualCoder():
    def __init__(self):
        super().__init__()

        self.code_size = 7

    def encode_torch(self, boxes, anchors):
        """

        Args:
            boxes: tensor, [num_anchors, 7], (x, y, z, dx, dy, dz, heading)
            anchors: tensor, [num_anchors, 7], (x, y, z, dx, dy, dz, heading)

        Returns:

        """
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)

        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        rt = rg - ra

        return torch.cat([xt, yt, zt, dxt, dyt, dzt, rt], dim=-1)

    def decode_torch(self, box_encodings, anchors):
        """

        Args:
            box_encodings: tensor, [batch_size, num_anchors, 7] or [num_anchors, 7] (x, y, z, dx, dy, dz, heading)
            anchors: tensor, [batch_size, num_anchors, 7] or [num_anchors, 7] (x, y, z, dx, dy, dz, heading)

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, dxt, dyt, dzt, rt = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)

        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za
        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza
        rg = rt + ra

        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg], dim=-1)
