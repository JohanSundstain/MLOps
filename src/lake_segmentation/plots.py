"""Figures: dataset previews, training curves and inference reports.

Every figure is written to the ``plots`` directory of the repository (and additionally
uploaded to MLflow by the training callbacks), as required by the assignment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless rendering on servers and in Colab

import matplotlib.pyplot as plt  # noqa: E402  (must come after the backend selection)
import numpy as np  # noqa: E402

from lake_segmentation.constants import REFLECTANCE_SCALE  # noqa: E402
from lake_segmentation.data.patches import load_patch  # noqa: E402
from lake_segmentation.logging_utils import get_logger  # noqa: E402

LOGGER = get_logger("plots")

DEFAULT_FIGURE_SIZE = (14.0, 4.0)
DPI = 110


def _display_stretch(
    channel: np.ndarray, low_reflectance: float = 0.0, high_reflectance: float = 0.30
) -> np.ndarray:
    """Convert a reflectance channel into a displayable ``[0, 1]`` image.

    ``uint16`` bands are scaled by 10 000 first, then a fixed linear stretch is applied: this
    keeps the colours of the same ground cover consistent between figures (a percentile
    stretch would make every patch look "well exposed" and hide real brightness differences).
    """

    values = np.asarray(channel, dtype=np.float32)
    if not np.isfinite(values).all():
        values = np.nan_to_num(values, nan=0.0)
    if float(values.max()) > 1.5:  # integer surface reflectance counts
        values = values / REFLECTANCE_SCALE
    return np.clip(
        (values - low_reflectance) / max(high_reflectance - low_reflectance, 1e-6), 0.0, 1.0
    ).astype(np.float32)


def rgb_preview(image: np.ndarray, band_order: Sequence[str]) -> np.ndarray:
    """Convert a ``(bands, height, width)`` reflectance patch to a displayable RGB image."""

    red_index = list(band_order).index("red")
    green_index = list(band_order).index("green")
    blue_index = list(band_order).index("blue")
    channels = [_display_stretch(image[index]) for index in (red_index, green_index, blue_index)]
    return np.stack(channels, axis=-1)


def mask_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a red water overlay over an RGB image."""

    overlay = rgb.copy()
    water = np.asarray(mask).astype(bool)
    overlay[water] = (1.0 - alpha) * overlay[water] + alpha * np.array([1.0, 0.15, 0.15])
    return np.clip(overlay, 0.0, 1.0)


def save_dataset_preview(
    records: Sequence[Mapping[str, Any]],
    data_dir: Path,
    output_path: Path,
    band_order: Sequence[str],
    max_items: int = 4,
) -> Path:
    """Grid with ``RGB | reference mask | overlay`` pairs to visually validate the labels."""

    if not records:
        raise ValueError("No records to preview")
    selected = list(records)[:max_items]
    figure, axes = plt.subplots(
        len(selected), 3, figsize=(12.0, 3.6 * len(selected)), squeeze=False
    )
    for row, record in enumerate(selected):
        image, mask = load_patch(Path(data_dir) / str(record["path"]))
        rgb = rgb_preview(image, band_order)
        water_fraction = 100.0 * float(np.asarray(mask).mean())
        axes[row][0].imshow(rgb)
        axes[row][0].set_title(f"Sentinel-2 B4/B3/B2\n{record['patch_id']}", fontsize=8)
        axes[row][1].imshow(mask, cmap="Blues", vmin=0, vmax=1)
        axes[row][1].set_title(f"WorldCover water mask ({water_fraction:.0f}%)", fontsize=8)
        axes[row][2].imshow(mask_overlay(rgb, mask))
        axes[row][2].set_title("Overlay", fontsize=8)
        for axis in axes[row]:
            axis.axis("off")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Dataset preview saved to %s", output_path)
    return output_path


def save_prediction_grid(
    images: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    output_path: Path,
    band_order: Sequence[str],
    title: str = "Validation predictions",
) -> Path:
    """Grid with ``RGB | reference | prediction`` rows for a few validation patches."""

    count = min(len(images), len(targets), len(predictions))
    if count == 0:
        raise ValueError("Nothing to plot")
    figure, axes = plt.subplots(count, 3, figsize=(12.0, 3.4 * count), squeeze=False)
    for row in range(count):
        rgb = rgb_preview(images[row], band_order)
        axes[row][0].imshow(rgb)
        axes[row][0].set_title("Image", fontsize=8)
        axes[row][1].imshow(targets[row], cmap="Blues", vmin=0, vmax=1)
        axes[row][1].set_title("Reference", fontsize=8)
        axes[row][2].imshow(predictions[row], cmap="Blues", vmin=0, vmax=1)
        axes[row][2].set_title("Prediction (thresholded)", fontsize=8)
        for axis in axes[row]:
            axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Prediction grid saved to %s", output_path)
    return output_path


def save_training_curves(
    history: Mapping[str, Sequence[float]], output_dir: Path
) -> dict[str, Path]:
    """Write loss / metric / area error / learning rate curves and the raw history as CSV."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    epochs = list(history.get("epoch", []))

    def _plot(series: Mapping[str, Sequence[float]], title: str, ylabel: str, name: str) -> None:
        figure, axis = plt.subplots(figsize=(7.0, 4.2))
        plotted = False
        for label, values in series.items():
            if not values:
                continue
            axis.plot(epochs[: len(values)], values, label=label)
            plotted = True
        if not plotted:
            plt.close(figure)
            return
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.legend()
        path = output_dir / name
        figure.tight_layout()
        figure.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(figure)
        written[name] = path

    _plot(
        {"train loss": history.get("train_loss", []), "val loss": history.get("val_loss", [])},
        "Training and validation loss",
        "loss",
        "losses.png",
    )
    _plot(
        {
            "IoU": history.get("val_iou", []),
            "Dice": history.get("val_dice", []),
            "Precision": history.get("val_precision", []),
            "Recall": history.get("val_recall", []),
        },
        "Validation segmentation metrics",
        "score",
        "metrics.png",
    )
    _plot(
        {"area error, %": history.get("val_area_error", [])},
        "Relative water area error on validation",
        "percent",
        "area_error.png",
    )
    _plot(
        {"learning rate": history.get("learning_rate", [])},
        "Learning rate schedule",
        "learning rate",
        "learning_rate.png",
    )

    keys = [key for key in history if history[key]]
    if keys and epochs:
        columns = ["epoch", *[key for key in keys if key != "epoch"]]
        longest = max(len(history[key]) for key in keys)
        rows = []
        for index in range(longest):
            row = [history["epoch"][index] if index < len(history["epoch"]) else ""]
            for key in columns[1:]:
                values = history[key]
                row.append(values[index] if index < len(values) else "")
            rows.append(row)
        csv_path = output_dir / "history.csv"
        with csv_path.open("w", encoding="utf-8") as handle:
            handle.write(",".join(columns) + "\n")
            for row in rows:
                handle.write(",".join(str(value) for value in row) + "\n")
        written["history.csv"] = csv_path

    return written
