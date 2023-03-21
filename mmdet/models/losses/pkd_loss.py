# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from .utils import weighted_loss


def norm(feat: torch.Tensor) -> torch.Tensor:
    """Normalize the feature maps to have zero mean and unit variances.

    Args:
        feat (torch.Tensor): The original feature map with shape
            (N, C, H, W).
    """
    assert len(feat.shape) == 4
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    feat = (feat - mean) / (std + 1e-6)
    return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)


@weighted_loss
def pkd_loss(pred, target):
    pred = norm(pred)
    target = norm(target)
    return F.mse_loss(pred, target, reduction='none') / 2


@MODELS.register_module()
class PKDLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PKDLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None) -> torch.Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * pkd_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
