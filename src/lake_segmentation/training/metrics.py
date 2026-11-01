"""Streaming metrics used inside the Lightning module.

The metrics are computed from the pixel-level confusion matrix accumulated over the epoch.
They are implemented here instead of relying on ``torchmetrics.classification`` because that
API keeps changing between releases (``Dice`` was moved out of it in ``torchmetrics`` 1.9),
while the definitions in :mod:`lake_segmentation.metrics` are the reference for the project:

* IoU of two empty masks is ``1.0``;
* Precision/Recall are ``0.0`` when their denominator is zero;
* the relative water area error is defined per image and skips images without reference water.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection

METRIC_MODES = ("iou", "dice", "precision", "recall")


class BinarySegmentationScore(Metric):
    """IoU, Dice, Precision or Recall accumulated over a whole epoch."""

    full_state_update = False

    def __init__(self, mode: str = "iou", threshold: float = 0.5) -> None:
        super().__init__()
        if mode not in METRIC_MODES:
            raise ValueError(f"Unknown metric mode {mode!r}, expected one of {METRIC_MODES}")
        self.mode = mode
        self.threshold = float(threshold)
        self.add_state("true_positive", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_positive", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_negative", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions: Tensor, target: Tensor) -> None:
        predicted = predictions.to(torch.bool)
        reference = target.to(torch.bool)
        self.true_positive = self.true_positive + torch.sum(predicted & reference).to(
            self.true_positive.dtype
        )
        self.false_positive = self.false_positive + torch.sum(predicted & ~reference).to(
            self.false_positive.dtype
        )
        self.false_negative = self.false_negative + torch.sum(~predicted & reference).to(
            self.false_negative.dtype
        )

    def compute(self) -> Tensor:
        true_positive = self.true_positive
        false_positive = self.false_positive
        false_negative = self.false_negative
        if self.mode == "iou":
            union = true_positive + false_positive + false_negative
            if float(union) == 0:
                return torch.ones_like(union)
            return true_positive / union
        if self.mode == "dice":
            denominator = 2 * true_positive + false_positive + false_negative
            if float(denominator) == 0:
                return torch.ones_like(denominator)
            return 2 * true_positive / denominator
        if self.mode == "precision":
            denominator = true_positive + false_positive
            if float(denominator) == 0:
                return torch.zeros_like(denominator)
            return true_positive / denominator
        denominator = true_positive + false_negative
        if float(denominator) == 0:
            return torch.zeros_like(denominator)
        return true_positive / denominator


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
            "iou": BinarySegmentationScore("iou", threshold=threshold),
            "dice": BinarySegmentationScore("dice", threshold=threshold),
            "precision": BinarySegmentationScore("precision", threshold=threshold),
            "recall": BinarySegmentationScore("recall", threshold=threshold),
            "area_error": RelativeAreaError(threshold=threshold),
        },
        prefix=prefix,
    )
