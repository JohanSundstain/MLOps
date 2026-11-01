"""Streaming metrics used inside the Lightning module.

``torchmetrics`` provides IoU, Dice, Precision and Recall; the relative water area error is a
small custom metric because it is defined per image and must skip images without any reference
water (where the relative error is undefined).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, Dice


class RelativeAreaError(Metric):
    """Mean ``|S_pred - S_true| / S_true * 100`` over images with ``S_true > 0``."""

    full_state_update = False

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.add_state("error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probabilities: Tensor, target: Tensor) -> None:
        predicted = (probabilities > self.threshold).to(torch.float64)
        reference = (target > 0.5).to(torch.float64)
        dimensions = tuple(range(1, predicted.ndim))
        predicted_area = predicted.sum(dim=dimensions)
        reference_area = reference.sum(dim=dimensions)
        valid = reference_area > 0
        if not torch.any(valid):
            return
        error = torch.abs(predicted_area[valid] - reference_area[valid]) / reference_area[valid]
        self.error_sum = self.error_sum + torch.sum(error) * 100.0
        self.count = self.count + torch.sum(valid.to(self.count.dtype))

    def compute(self) -> Tensor:
        if int(self.count) == 0:
            return torch.tensor(float("nan"), device=self.error_sum.device)
        return self.error_sum / self.count


def build_metric_collection(prefix: str, threshold: float = 0.5) -> MetricCollection:
    """IoU / Dice / Precision / Recall / Relative Area Error for one stage."""

    return MetricCollection(
        {
            "iou": BinaryJaccardIndex(threshold=threshold),
            "dice": Dice(threshold=threshold),
            "precision": BinaryPrecision(threshold=threshold),
            "recall": BinaryRecall(threshold=threshold),
            "area_error": RelativeAreaError(threshold=threshold),
        },
        prefix=prefix,
    )
