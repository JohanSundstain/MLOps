"""Reflectance normalisation.

Sentinel-2 L2A pixels are stored as integers scaled by 10 000. Before they reach the network
they are converted to reflectance, clipped to a physically plausible range and standardised
with the per-band mean and standard deviation of the **training** split. The statistics are
stored next to the dataset and copied into the model bundle, so inference never has to
recompute them (and never leaks validation data).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from lake_segmentation.constants import REFLECTANCE_SCALE
from lake_segmentation.logging_utils import get_logger

LOGGER = get_logger("data.normalization")

MIN_STD = 1e-6


@dataclass(frozen=True)
class NormalizationStats:
    """Per-band mean and standard deviation of the (clipped) reflectance."""

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.mean) != len(self.std):
            raise ValueError("mean and std must have the same length")
        if not self.mean:
            raise ValueError("normalisation statistics must not be empty")

    @property
    def num_bands(self) -> int:
        return len(self.mean)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, values: dict[str, Sequence[float]]) -> "NormalizationStats":
        return cls(
            mean=tuple(float(value) for value in values["mean"]),
            std=tuple(float(value) for value in values["std"]),
        )

    @classmethod
    def identity(cls, num_bands: int) -> "NormalizationStats":
        return cls(
            mean=tuple(0.0 for _ in range(num_bands)), std=tuple(1.0 for _ in range(num_bands))
        )


def to_reflectance(
    image: np.ndarray,
    scale: float = REFLECTANCE_SCALE,
    clip_min: float = 0.0,
    clip_max: float = 0.4,
) -> np.ndarray:
    """Integer L2A counts -> clipped float32 reflectance."""

    reflectance = image.astype(np.float32) / float(scale)
    return np.clip(reflectance, clip_min, clip_max)


def apply_normalization(image: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Standardise a ``(bands, height, width)`` reflectance array."""

    mean = np.asarray(stats.mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(stats.std, dtype=np.float32).reshape(-1, 1, 1)
    return (image - mean) / np.maximum(std, MIN_STD)


def denormalize(image: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Inverse of :func:`apply_normalization` (handy for plots)."""

    mean = np.asarray(stats.mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(stats.std, dtype=np.float32).reshape(-1, 1, 1)
    return image * np.maximum(std, MIN_STD) + mean


def compute_stats(
    patches: Iterable[np.ndarray],
    scale: float = REFLECTANCE_SCALE,
    clip_min: float = 0.0,
    clip_max: float = 0.4,
) -> NormalizationStats:
    """Accumulate per-band statistics over an iterable of integer image patches."""

    counts: np.ndarray | None = None
    total = None
    total_squared = None

    for patch in patches:
        reflectance = to_reflectance(patch, scale, clip_min, clip_max)
        if reflectance.ndim != 3:
            raise ValueError(f"Expected (bands, height, width), got {reflectance.shape}")
        flat = reflectance.reshape(reflectance.shape[0], -1).astype(np.float64)
        if total is None:
            total = flat.sum(axis=1)
            total_squared = np.square(flat).sum(axis=1)
            counts = np.full(flat.shape[0], flat.shape[1], dtype=np.float64)
        else:
            total += flat.sum(axis=1)
            total_squared += np.square(flat).sum(axis=1)
            counts += flat.shape[1]

    if total is None or counts is None:
        raise ValueError("No patches were provided to compute normalisation statistics")

    mean = total / counts
    variance = np.maximum(total_squared / counts - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.maximum(std, MIN_STD)
    LOGGER.info(
        "Normalisation stats: mean=%s std=%s", np.round(mean, 4).tolist(), np.round(std, 4).tolist()
    )
    return NormalizationStats(mean=tuple(mean.tolist()), std=tuple(std.tolist()))


def save_stats(path: Path, stats: NormalizationStats) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")


def load_stats(path: Path) -> NormalizationStats:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Normalisation statistics not found at {path}. Run the preprocessing step first."
        )
    return NormalizationStats.from_dict(json.loads(path.read_text(encoding="utf-8")))
