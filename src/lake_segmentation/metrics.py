"""Binary segmentation metrics.

Two flavours are used in the project:

* streaming metrics inside the Lightning module (``torchmetrics``), computed on the GPU;
* this module, which works with NumPy masks and is shared by offline evaluation, the NDWI
  baseline and the inference report so that all numbers are produced by the same code.

Conventions for degenerate cases:

* IoU of two empty masks is ``1.0`` (the masks are identical);
* Precision with no predicted water pixels is ``0.0``;
* Relative Area Error is ``None`` when the reference area is zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BinaryConfusion:
    """Pixel level confusion matrix of a binary segmentation task."""

    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def union(self) -> int:
        return self.true_positive + self.false_positive + self.false_negative

    @property
    def iou(self) -> float:
        if self.union == 0:
            return 1.0
        return self.true_positive / self.union

    @property
    def dice(self) -> float:
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        if denominator == 0:
            return 1.0
        return 2 * self.true_positive / denominator

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        if denominator == 0:
            return 0.0
        return self.true_positive / denominator

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        if denominator == 0:
            return 0.0
        return self.true_positive / denominator

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.true_positive,
            "fp": self.false_positive,
            "fn": self.false_negative,
            "iou": self.iou,
            "dice": self.dice,
            "precision": self.precision,
            "recall": self.recall,
        }


def confusion_from_masks(prediction: np.ndarray, target: np.ndarray) -> BinaryConfusion:
    """Count TP / FP / FN pixels of two binary masks."""

    predicted = np.asarray(prediction).astype(bool)
    reference = np.asarray(target).astype(bool)
    if predicted.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {predicted.shape} vs {reference.shape}")
    true_positive = int(np.logical_and(predicted, reference).sum())
    false_positive = int(np.logical_and(predicted, ~reference).sum())
    false_negative = int(np.logical_and(~predicted, reference).sum())
    return BinaryConfusion(true_positive, false_positive, false_negative)


def relative_area_error_percent(prediction: np.ndarray, target: np.ndarray) -> float | None:
    """``|S_pred - S_true| / S_true * 100`` computed on pixel counts.

    Pixel counts are proportional to the ground area for rasters with a constant pixel size,
    which is the case for every preprocessed Sentinel-2 tile.
    """

    predicted_area = float(np.asarray(prediction).astype(bool).sum())
    reference_area = float(np.asarray(target).astype(bool).sum())
    if reference_area <= 0:
        return None
    return abs(predicted_area - reference_area) / reference_area * 100.0


def evaluate_binary_masks(
    predictions: Sequence[np.ndarray], targets: Sequence[np.ndarray]
) -> dict[str, float | int]:
    """Aggregate metrics over a set of masks.

    ``iou``/``dice``/``precision``/``recall`` are micro-averaged (pixel counts are summed over
    all images), the ``*_per_image_mean`` keys are the macro averages, and the area error is
    averaged over images that contain reference water.
    """

    if len(predictions) != len(targets):
        raise ValueError("Number of predictions and targets differ")
    if not predictions:
        raise ValueError("No masks to evaluate")

    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0
    per_image_iou: list[float] = []
    per_image_dice: list[float] = []
    per_image_precision: list[float] = []
    per_image_recall: list[float] = []
    area_errors: list[float] = []
    images_without_water = 0

    for prediction, target in zip(predictions, targets):
        confusion = confusion_from_masks(prediction, target)
        total_true_positive += confusion.true_positive
        total_false_positive += confusion.false_positive
        total_false_negative += confusion.false_negative
        per_image_iou.append(confusion.iou)
        per_image_dice.append(confusion.dice)
        per_image_precision.append(confusion.precision)
        per_image_recall.append(confusion.recall)
        error = relative_area_error_percent(prediction, target)
        if error is None:
            images_without_water += 1
        else:
            area_errors.append(error)

    aggregated = BinaryConfusion(
        total_true_positive, total_false_positive, total_false_negative
    ).as_dict()
    result: dict[str, float | int] = dict(aggregated)
    result.update(
        {
            "iou_per_image_mean": float(np.mean(per_image_iou)),
            "dice_per_image_mean": float(np.mean(per_image_dice)),
            "precision_per_image_mean": float(np.mean(per_image_precision)),
            "recall_per_image_mean": float(np.mean(per_image_recall)),
            "area_error_percent": float(np.mean(area_errors)) if area_errors else float("nan"),
            "area_error_percent_median": (
                float(np.median(area_errors)) if area_errors else float("nan")
            ),
            "num_images": len(predictions),
            "num_images_without_reference_water": images_without_water,
        }
    )
    return result


def format_metrics(metrics: dict[str, float | int], precision: int = 4) -> str:
    """Human readable one-line (well, multi-line) summary of a metrics dictionary."""

    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key:>34}: {value:.{precision}f}")
        else:
            lines.append(f"{key:>34}: {value}")
    return "\n".join(lines)
