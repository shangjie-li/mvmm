import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss with anchor-wise weighting.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(SigmoidFocalClassificationLoss, self).__init__()

        self.alpha = alpha  # for balancing loss in positive and negative examples
        self.gamma = gamma  # for balancing loss in hard and easy examples

    @staticmethod
    def sigmoid_cross_entropy_with_logits(inputs, targets):
        """

        Args:
            inputs: tensor of float, [batch_size, num_anchors, num_classes], predicted logits
            targets: tensor of float, [batch_size, num_anchors, num_classes], one-hot classification targets

        Returns:
            loss: tensor of float, [batch_size, num_anchors, num_classes]

        """
        loss = torch.clamp(inputs, min=0) - inputs * targets + torch.log1p(torch.exp(-torch.abs(inputs)))

        return loss

    def forward(self, inputs, targets, weights):
        """

        Args:
            inputs: tensor of float, [batch_size, num_anchors, num_classes], predicted logits
            targets: tensor of float, [batch_size, num_anchors, num_classes], one-hot classification targets
            weights: tensor of float, [batch_size, num_anchors], anchor-wise weights

        Returns:
            loss: tensor of float, [batch_size, num_anchors, num_classes]

        """
        pred_sigmoid = torch.sigmoid(inputs)
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        pt = targets * (1.0 - pred_sigmoid) + (1.0 - targets) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(inputs, targets)

        loss = focal_weight * bce_loss

        assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
        weights = weights.unsqueeze(-1)
        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss with code-wise and anchor-wise weighting modified based on fvcore.nn.smooth_l1_loss.
    Ref: https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta,
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta=1.0 / 9.0, code_weights=None):
        super(WeightedSmoothL1Loss, self).__init__()

        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, inputs, targets, weights):
        """

        Args:
            inputs: tensor of float, [batch_size, num_anchors, code_size], predicted logits
            targets: tensor of float, [batch_size, num_anchors, code_size], regression targets
            weights: tensor of float, [batch_size, num_anchors], anchor-wise weights

        Returns:
            loss: tensor of float, [batch_size, num_anchors, code_size]

        """
        targets = torch.where(torch.isnan(targets), inputs, targets)
        diff = inputs - targets

        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
        weights = weights.unsqueeze(-1)
        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross entropy loss with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, weights):
        """

        Args:
            inputs: tensor of float, [batch_size, num_anchors, num_classes], predicted logits
            targets: tensor of float, [batch_size, num_anchors, num_classes], one-hot classification targets
            weights: tensor of float, [batch_size, num_anchors], anchor-wise weights

        Returns:
            loss: tensor of float, [batch_size, num_anchors]

        """
        inputs = inputs.permute(0, 2, 1)
        targets = targets.argmax(dim=-1)

        loss = F.cross_entropy(inputs, targets, reduction='none') * weights

        return loss
