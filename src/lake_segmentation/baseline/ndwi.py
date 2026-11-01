"""NDWI threshold baseline.

``NDWI = (B3 - B8) / (B3 + B8)`` is the classic water index of McFeeters (1996); the threshold
is tuned on the validation split only and then reported on the test split, exactly like the
network, so that the comparison is fair.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from omegaconf import DictConfig

from lake_segmentation.config import resolve_paths
from lake_segmentation.data.patches import load_patch
from lake_segmentation.data.pipeline import load_manifest
from lake_segmentation.logging_utils import get_logger
from lake_segmentation.metrics import evaluate_binary_masks

LOGGER = get_logger("baseline.ndwi")

EPSILON = 1e-6


def ndwi(image: np.ndarray, green_index: int = 1, nir_index: int = 3) -> np.ndarray:
    """Normalised Difference Water Index from an integer or float reflectance patch."""

    reflectance = image.astype(np.float32)
    green = reflectance[green_index]
    nir = reflectance[nir_index]
    return (green - nir) / np.maximum(green + nir, EPSILON)


def _load_split(
    processed_dir: Path,
    frame: Any,
    split: str,
    green_index: int,
    nir_index: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    records = frame[frame["split"] == split]
    ndwi_values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for record in records.to_dict("records"):
        image, mask = load_patch(processed_dir / str(record["path"]))
        ndwi_values.append(ndwi(image, green_index=green_index, nir_index=nir_index))
        masks.append(mask.astype(bool))
    if not ndwi_values:
        raise ValueError(f"Split {split!r} is empty, cannot run the baseline")
    return ndwi_values, masks


def run_ndwi_baseline(cfg: DictConfig) -> dict[str, Any]:
    """Tune the NDWI threshold on validation and report test metrics."""

    paths = resolve_paths(cfg)
    processed_dir = paths["data_processed"]
    frame = load_manifest(processed_dir)
    band_order = list(cfg.data.stac.band_order)
    green_index = band_order.index("green")
    nir_index = band_order.index("nir")

    val_ndwi, val_masks = _load_split(processed_dir, frame, "val", green_index, nir_index)
    test_ndwi, test_masks = _load_split(processed_dir, frame, "test", green_index, nir_index)

    baseline_cfg = cfg.baseline
    grid = np.linspace(
        float(baseline_cfg.threshold_min),
        float(baseline_cfg.threshold_max),
        int(baseline_cfg.threshold_steps),
    )
    table: list[dict[str, float]] = []
    best_threshold = float(grid[0])
    best_iou = -1.0
    for threshold in grid:
        predictions = [values > threshold for values in val_ndwi]
        metrics = evaluate_binary_masks(predictions, val_masks)
        iou = float(metrics["iou"])
        table.append({"threshold": float(threshold), "iou": iou, "dice": float(metrics["dice"])})
        if iou > best_iou:
            best_iou = iou
            best_threshold = float(threshold)

    LOGGER.info("NDWI best threshold on validation: %.3f (IoU %.4f)", best_threshold, best_iou)
    test_predictions = [values > best_threshold for values in test_ndwi]
    test_metrics = evaluate_binary_masks(test_predictions, test_masks)

    report = {
        "index": "NDWI = (B3 - B8) / (B3 + B8)",
        "band_order": band_order,
        "tuned_on": "val",
        "threshold": best_threshold,
        "validation_iou": best_iou,
        "threshold_search": table,
        "test": {key: float(value) for key, value in test_metrics.items()},
    }

    output_path = Path(str(baseline_cfg.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("NDWI baseline report written to %s", output_path)
    LOGGER.info(
        "NDWI test IoU=%.4f Dice=%.4f Precision=%.4f Recall=%.4f area error=%.2f%%",
        test_metrics["iou"],
        test_metrics["dice"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["area_error_percent"],
    )
    return report


def compare_with_model(
    baseline_metrics: dict[str, Any], model_metrics: dict[str, Any]
) -> dict[str, dict[str, float]]:
    """Absolute delta of every metric between the baseline and the model."""

    keys: Sequence[str] = ("iou", "dice", "precision", "recall")
    comparison: dict[str, dict[str, float]] = {}
    for key in keys:
        baseline_value = float(baseline_metrics.get(key, float("nan")))
        model_value = float(model_metrics.get(key, float("nan")))
        comparison[key] = {
            "baseline": baseline_value,
            "model": model_value,
            "delta": model_value - baseline_value,
        }
    return comparison
