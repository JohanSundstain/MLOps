"""LightningDataModule over the preprocessed patch dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from lake_segmentation.data.dataset import LakePatchDataset
from lake_segmentation.data.normalization import NormalizationStats
from lake_segmentation.data.pipeline import load_manifest
from lake_segmentation.logging_utils import get_logger

LOGGER = get_logger("training.datamodule")


class LakeSegmentationDataModule(LightningDataModule):
    """Exposes the three splits of ``data/processed`` as data loaders."""

    def __init__(
        self,
        processed_dir: str | Path,
        stats: NormalizationStats,
        batch_size: int = 8,
        num_workers: int = 4,
        augment: bool = True,
        seed: int = 42,
        reflectance_scale: float = 10_000.0,
        clip_min: float = 0.0,
        clip_max: float = 0.4,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.stats = stats
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.reflectance_scale = float(reflectance_scale)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.datasets: dict[str, LakePatchDataset] = {}

    # ------------------------------------------------------------------ helpers
    def _build_dataset(self, split: str, augment: bool) -> LakePatchDataset:
        frame = load_manifest(self.processed_dir)
        records = frame[frame["split"] == split].to_dict("records")
        if not records:
            raise ValueError(f"Split {split!r} is empty in {self.processed_dir}")
        LOGGER.info("Split %-5s: %d patches", split, len(records))
        return LakePatchDataset(
            records=records,
            data_dir=self.processed_dir,
            stats=self.stats,
            reflectance_scale=self.reflectance_scale,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            augment=augment,
            seed=self.seed,
        )

    def setup(self, stage: str | None = None) -> None:
        if "train" not in self.datasets:
            self.datasets["train"] = self._build_dataset("train", augment=self.augment)
        if "val" not in self.datasets:
            self.datasets["val"] = self._build_dataset("val", augment=False)
        if "test" not in self.datasets:
            self.datasets["test"] = self._build_dataset("test", augment=False)

    # --------------------------------------------------------------- data loaders
    def _loader(self, dataset: LakePatchDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.datasets["train"], shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.datasets["val"], shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.datasets["test"], shuffle=False)

    # ------------------------------------------------------------------ metadata
    @property
    def split_records(self) -> dict[str, list[dict[str, Any]]]:
        return {split: dataset.records for split, dataset in self.datasets.items()}

    def class_balance(self) -> Mapping[str, float]:
        """Mean water fraction per split (a sanity check before training)."""

        balance: dict[str, float] = {}
        for split, dataset in self.datasets.items():
            fractions = [float(record["water_fraction"]) for record in dataset.records]
            balance[split] = sum(fractions) / max(len(fractions), 1)
        return balance
