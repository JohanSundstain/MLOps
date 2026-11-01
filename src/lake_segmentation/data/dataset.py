"""PyTorch dataset over the preprocessed patches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from lake_segmentation.data.normalization import (
    NormalizationStats,
    apply_normalization,
    to_reflectance,
)
from lake_segmentation.data.patches import load_patch


def augment_image_and_mask(
    image: np.ndarray, mask: np.ndarray, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random flips and 90° rotations.

    Only geometric transformations are used: any photometric augmentation would distort the
    physical meaning of the spectral bands, which is exactly what the project description
    asks to avoid.
    """

    if generator.random() < 0.5:
        image = image[:, :, ::-1]
        mask = mask[:, ::-1]
    if generator.random() < 0.5:
        image = image[:, ::-1, :]
        mask = mask[::-1, :]
    quarter_turns = int(generator.integers(0, 4))
    if quarter_turns:
        image = np.rot90(image, k=quarter_turns, axes=(1, 2))
        mask = np.rot90(mask, k=quarter_turns, axes=(0, 1))
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class LakePatchDataset(Dataset):
    """Returns normalised images and float masks ready for ``BCEWithLogitsLoss``."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        data_dir: str | Path,
        stats: NormalizationStats,
        reflectance_scale: float = 10_000.0,
        clip_min: float = 0.0,
        clip_max: float = 0.4,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        if not records:
            raise ValueError("The dataset is empty, check the preprocessing output")
        self.records = list(records)
        self.data_dir = Path(data_dir)
        self.stats = stats
        self.reflectance_scale = float(reflectance_scale)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.augment = augment
        self.seed = int(seed)
        self._generator: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        patch_path = self.data_dir / str(record["path"])
        image, mask = load_patch(patch_path)

        if self.augment:
            if self._generator is None:
                # One generator per worker keeps the augmentation reproducible.
                worker_info = torch.utils.data.get_worker_info()
                worker_seed = self.seed if worker_info is None else self.seed + worker_info.id
                self._generator = np.random.default_rng(worker_seed)
            image, mask = augment_image_and_mask(image, mask, self._generator)

        reflectance = to_reflectance(image, self.reflectance_scale, self.clip_min, self.clip_max)
        normalized = apply_normalization(reflectance, self.stats)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(normalized)).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0),
            "patch_id": record["patch_id"],
            "aoi": record["aoi"],
        }

    def image_for_preview(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the raw integer patch and its mask (used by plotting callbacks)."""

        record = self.records[index]
        return load_patch(self.data_dir / str(record["path"]))
