"""Cutting an AOI into fixed size patches and selecting a balanced subset.

A patch is useful for training when it contains a shoreline: patches made of water only teach
nothing, and patches without any water at all are only useful as negatives. The selection
therefore keeps patches whose water fraction lies in a configured range and adds a share of
all-land negatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from lake_segmentation.tiling import Tile, iter_tiles


@dataclass(frozen=True)
class PatchCandidate:
    """One candidate patch of an AOI with the statistics used for filtering."""

    tile: Tile
    water_fraction: float
    valid_fraction: float

    def is_positive(self, min_water_fraction: float) -> bool:
        return self.water_fraction >= min_water_fraction


def fraction_in_tile(mask: np.ndarray, tile: Tile) -> float:
    """Mean value of ``mask`` inside the valid part of ``tile``."""

    rows, cols = tile.valid_slices()
    values = mask[rows, cols]
    if values.size == 0:
        return 0.0
    return float(np.asarray(values, dtype=np.float64).mean())


def iter_candidates(
    water: np.ndarray,
    valid: np.ndarray,
    patch_size: int,
    stride: int | None = None,
) -> Iterator[PatchCandidate]:
    """Yield every tile of one AOI with its water and validity fractions."""

    height, width = water.shape
    for tile in iter_tiles(height, width, patch_size, stride):
        yield PatchCandidate(
            tile=tile,
            water_fraction=fraction_in_tile(water, tile),
            valid_fraction=fraction_in_tile(valid, tile),
        )


def select_candidates(
    candidates: Sequence[PatchCandidate],
    min_water_fraction: float,
    max_water_fraction: float,
    min_valid_fraction: float,
    negative_ratio: float,
    rng: np.random.Generator,
    max_patches: int | None = None,
) -> list[PatchCandidate]:
    """Filter candidates and subsample the negatives to reach ``negative_ratio``.

    ``negative_ratio`` is the share of all-land patches among the selected ones.
    """

    if not 0.0 <= negative_ratio < 1.0:
        raise ValueError(f"negative_ratio must be in [0, 1), got {negative_ratio}")
    usable = [
        candidate for candidate in candidates if candidate.valid_fraction >= min_valid_fraction
    ]
    positives = [
        candidate
        for candidate in usable
        if min_water_fraction <= candidate.water_fraction <= max_water_fraction
    ]
    negatives = [candidate for candidate in usable if candidate.water_fraction < min_water_fraction]

    if not positives:
        return []

    if max_patches is not None and len(positives) > max_patches:
        positions = rng.choice(len(positives), size=max_patches, replace=False)
        positives = [positives[index] for index in sorted(positions)]

    if negative_ratio <= 0.0:
        wanted_negatives = 0
    else:
        wanted_negatives = int(round(len(positives) * negative_ratio / (1.0 - negative_ratio)))
    if wanted_negatives > 0 and negatives:
        positions = rng.choice(
            len(negatives), size=min(wanted_negatives, len(negatives)), replace=False
        )
        negatives = [negatives[index] for index in sorted(positions)]
    else:
        negatives = []

    selected = positives + negatives
    selected.sort(key=lambda candidate: (candidate.tile.row, candidate.tile.col))
    return selected


def cut_patch(image: np.ndarray, mask: np.ndarray, tile: Tile) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(bands, size, size)`` image patch and the ``(size, size)`` label patch."""

    rows, cols = tile.valid_slices()
    image_patch = np.ascontiguousarray(image[:, rows, cols])
    mask_patch = np.ascontiguousarray(mask[rows, cols].astype(np.uint8, copy=False))
    return image_patch, mask_patch


def patch_identifier(aoi_name: str, target_date: str, tile: Tile) -> str:
    """Stable, human readable identifier of a patch."""

    return f"{aoi_name}__{target_date}__r{tile.row:05d}_c{tile.col:05d}"


def save_patch(path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    """Store one patch as a compressed ``npz`` archive (``image`` + ``mask``)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, image=image.astype(np.uint16, copy=False), mask=mask)


def load_patch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a patch written by :func:`save_patch`."""

    with np.load(Path(path)) as archive:
        image = archive["image"]
        mask = archive["mask"]
    return image, mask
