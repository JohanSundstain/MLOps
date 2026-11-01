"""Loss functions for binary water segmentation.

The loss is the weighted sum of a binary cross entropy and a soft Dice loss: BCE gives a
stable pixel-wise gradient, Dice directly optimises the overlap and prevents the network from
collapsing to the "all land" solution when the water fraction is small.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SoftDiceLoss(nn.Module):
    """``1 - 2 * |P n T| / (|P| + |T|)`` computed on probabilities."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probabilities = torch.sigmoid(logits)
        target = target.to(probabilities.dtype)
        dimensions = tuple(range(1, probabilities.ndim))
        intersection = torch.sum(probabilities * target, dim=dimensions)
        cardinality = torch.sum(probabilities, dim=dimensions) + torch.sum(target, dim=dimensions)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Weighted combination of BCE (with logits) and soft Dice."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        total = float(bce_weight) + float(dice_weight)
        if total <= 0:
            raise ValueError("At least one loss weight must be positive")
        self.bce_weight = float(bce_weight) / total
        self.dice_weight = float(dice_weight) / total
        self.dice = SoftDiceLoss(smooth=smooth)
        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight)) if pos_weight is not None else None,
        )

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        target = target.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        dice = self.dice(logits, target)
        return self.bce_weight * bce + self.dice_weight * dice

    def components(self, logits: Tensor, target: Tensor) -> dict[str, Tensor]:
        """Individual loss terms (logged as separate curves)."""

        target = target.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        dice = self.dice(logits, target)
        return {"bce": bce, "dice": dice}
