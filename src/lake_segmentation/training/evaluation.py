"""Offline evaluation: probability collection, threshold tuning and figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from lake_segmentation.data.patches import load_patch
from lake_segmentation.logging_utils import get_logger
from lake_segmentation.metrics import evaluate_binary_masks
from lake_segmentation.plots import save_prediction_grid

LOGGER = get_logger("training.evaluation")


def collect_probabilities(
    module: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    max_batches: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run the model over a data loader and return per-patch probabilities and masks."""

    was_training = module.training
    module.eval()
    module.to(device)
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if max_batches is not None and index >= max_batches:
                break
            logits = module(batch["image"].to(device))
            batch_probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            probabilities.extend(np.squeeze(batch_probabilities, axis=1))
            targets.extend(np.squeeze(batch["mask"].numpy(), axis=1) > 0.5)
    module.train(was_training)
    return probabilities, targets


def tune_threshold(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    threshold_min: float = 0.2,
    threshold_max: float = 0.8,
    steps: int = 25,
) -> dict[str, Any]:
    """Choose the binarisation threshold that maximises IoU on the validation split."""

    grid = np.linspace(threshold_min, threshold_max, steps)
    table: list[dict[str, float]] = []
    best_threshold = 0.5
    best_iou = -1.0
    for threshold in grid:
        predictions = [probability > threshold for probability in probabilities]
        metrics = evaluate_binary_masks(predictions, targets)
        iou = float(metrics["iou"])
        table.append(
            {
                "threshold": float(threshold),
                "iou": iou,
                "dice": float(metrics["dice"]),
                "area_error_percent": float(metrics["area_error_percent"]),
            }
        )
        if iou > best_iou:
            best_iou = iou
            best_threshold = float(threshold)
    LOGGER.info("Best validation threshold: %.3f (IoU %.4f)", best_threshold, best_iou)
    return {
        "threshold": best_threshold,
        "iou": best_iou,
        "table": table,
        "threshold_min": float(threshold_min),
        "threshold_max": float(threshold_max),
        "steps": int(steps),
    }


def evaluate_on_split(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    """Metrics of a split at the given threshold."""

    predictions = [probability > threshold for probability in probabilities]
    return dict(evaluate_binary_masks(predictions, targets))


def save_validation_predictions(
    records: Sequence[Mapping[str, Any]],
    image_dir: Path,
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    output_path: Path,
    band_order: Sequence[str],
    threshold: float,
    max_items: int = 3,
) -> Path | None:
    """Figure with ``image | reference | prediction`` for a few validation patches."""

    count = min(len(records), len(probabilities), max_items)
    if count == 0:
        return None
    images = []
    masks = []
    predictions = []
    for index in range(count):
        image, mask = load_patch(Path(image_dir) / str(records[index]["path"]))
        images.append(image)
        masks.append(mask)
        predictions.append((probabilities[index] > threshold).astype(np.uint8))
    return save_prediction_grid(
        images=images,
        targets=masks,
        predictions=predictions,
        output_path=output_path,
        band_order=band_order,
        title=(
            f"Validation predictions, threshold={threshold:.2f}, "
            f"IoU={evaluate_binary_masks(predictions, masks)['iou']:.3f}"
        ),
    )


def summarize_predictions_by_aoi(
    records: Sequence[Mapping[str, Any]],
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    threshold: float,
) -> dict[str, dict[str, float]]:
    """Metrics per lake group (the project requires per-region numbers in addition to means)."""

    per_aoi: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        per_aoi.setdefault(str(record["aoi"]), []).append(index)

    summary: dict[str, dict[str, float]] = {}
    for aoi, indices in sorted(per_aoi.items()):
        predictions = [probabilities[index] > threshold for index in indices]
        masks = [targets[index] for index in indices]
        metrics = evaluate_binary_masks(predictions, masks)
        summary[aoi] = {
            "patches": len(indices),
            "iou": float(metrics["iou"]),
            "dice": float(metrics["dice"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "area_error_percent": float(metrics["area_error_percent"]),
        }
    return summary


def iteration_to_device(batches: Iterable[Mapping[str, Any]], device: torch.device) -> Iterable:
    """Move a batch of tensors to a device (kept for symmetry with the training loop)."""

    for batch in batches:
        yield {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
