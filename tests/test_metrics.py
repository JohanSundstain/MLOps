"""Tests for the segmentation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from lake_segmentation.metrics import (
    BinaryConfusion,
    confusion_from_masks,
    evaluate_binary_masks,
    relative_area_error_percent,
)


def test_confusion_counts():
    prediction = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    target = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    confusion = confusion_from_masks(prediction, target)
    assert confusion.true_positive == 1
    assert confusion.false_positive == 1
    assert confusion.false_negative == 1


def test_metrics_values():
    confusion = BinaryConfusion(true_positive=6, false_positive=2, false_negative=3)
    assert confusion.iou == pytest.approx(6 / 11)
    assert confusion.dice == pytest.approx(12 / 17)
    assert confusion.precision == pytest.approx(6 / 8)
    assert confusion.recall == pytest.approx(6 / 9)


def test_empty_masks_have_iou_one():
    confusion = BinaryConfusion(true_positive=0, false_positive=0, false_negative=0)
    assert confusion.iou == 1.0
    assert confusion.dice == 1.0
    assert confusion.precision == 0.0


def test_relative_area_error_percent():
    prediction = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    target = np.array([[1, 1, 1, 0]], dtype=np.uint8)
    # 2 predicted water pixels against 3 reference pixels.
    assert relative_area_error_percent(prediction, target) == pytest.approx(100.0 / 3.0)
    assert relative_area_error_percent(prediction, np.zeros((1, 4), dtype=np.uint8)) is None


def test_evaluate_binary_masks_aggregates():
    predictions = [
        np.array([[1, 1], [0, 0]], dtype=np.uint8),
        np.array([[0, 0], [1, 1]], dtype=np.uint8),
    ]
    targets = [
        np.array([[1, 1], [0, 0]], dtype=np.uint8),
        np.array([[0, 0], [1, 1]], dtype=np.uint8),
    ]
    metrics = evaluate_binary_masks(predictions, targets)
    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["area_error_percent"] == pytest.approx(0.0)
    assert metrics["num_images"] == 2
