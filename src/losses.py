"""Loss functions for sponsored post detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Focal loss for binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for positive class (default 0.25).
        gamma: Focusing parameter (default 2.0).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits:  [B] unnormalized scores.
            targets: [B] binary float labels (0 or 1).

        Returns:
            Scalar loss.
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)   # p_t = sigmoid(logit) for y=1, 1-sigmoid for y=0

        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        return (focal_weight * bce).mean()


class CombinedLoss(nn.Module):
    """0.5 * BCE + 0.5 * FocalLoss.

    BCE uses pos_weight to handle the 86.2%/13.8% class imbalance:
        pos_weight = n_neg / n_pos ≈ 6.22 (= 1,379,364 / 221,710)

    Args:
        pos_weight:   Positive class weight for BCE (default 6.22).
        focal_alpha:  Focal loss alpha (default 0.25).
        focal_gamma:  Focal loss gamma (default 2.0).
        focal_weight: Mixing weight for focal loss (default 0.5).
    """

    def __init__(
        self,
        pos_weight: float = 6.22,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        focal_weight: float = 0.5,
    ):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.focal_weight = focal_weight
        self.bce_weight = 1.0 - focal_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits:  [B] unnormalized scores.
            targets: [B] binary float labels.

        Returns:
            Scalar combined loss.
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
        focal = self.focal(logits, targets)
        return self.bce_weight * bce + self.focal_weight * focal
