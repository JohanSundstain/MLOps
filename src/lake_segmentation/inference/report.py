"""Visual report of a two-date comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lake_segmentation.plots import DPI, mask_overlay, rgb_preview  # noqa: E402

MAX_FIGURE_PIXELS = 900


def _downsample(array: np.ndarray, max_size: int = MAX_FIGURE_PIXELS) -> np.ndarray:
    """Stride-subsample a ``(bands, h, w)`` or ``(h, w)`` array for plotting."""

    height, width = array.shape[-2:]
    step = max(1, int(np.ceil(max(height, width) / max_size)))
    return array[..., ::step, ::step]


def save_inference_figure(
    panels: Sequence[Mapping[str, object]],
    output_path: Path,
    band_order: Sequence[str],
    title: str,
) -> Path:
    """One row per date: ``true colour | water mask | overlay``."""

    if not panels:
        raise ValueError("No panels to plot")
    figure, axes = plt.subplots(len(panels), 3, figsize=(13.5, 4.3 * len(panels)), squeeze=False)
    for row, panel in enumerate(panels):
        image = _downsample(np.asarray(panel["image"]))
        mask = _downsample(np.asarray(panel["mask"]))
        rgb = rgb_preview(image, band_order)
        axes[row][0].imshow(rgb)
        axes[row][0].set_title(str(panel.get("label", "")), fontsize=9)
        axes[row][1].imshow(mask, cmap="Blues", vmin=0, vmax=1)
        axes[row][1].set_title(
            f"water mask, {float(panel.get('area_km2', 0.0)):.3f} km²", fontsize=9
        )
        axes[row][2].imshow(mask_overlay(rgb, mask))
        axes[row][2].set_title("overlay", fontsize=9)
        for axis in axes[row]:
            axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return output_path
