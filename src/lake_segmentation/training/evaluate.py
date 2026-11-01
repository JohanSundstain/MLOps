"""Evaluate a trained model on the test split without retraining."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from lake_segmentation.config import find_project_root, resolve_paths
from lake_segmentation.constants import BUNDLE_FILENAME
from lake_segmentation.data.dvc_utils import ensure_processed_data
from lake_segmentation.data.normalization import NormalizationStats
from lake_segmentation.export.onnx import load_state_dict
from lake_segmentation.logging_utils import get_logger, setup_logging
from lake_segmentation.models.bundle import load_bundle
from lake_segmentation.training.datamodule import LakeSegmentationDataModule
from lake_segmentation.training.evaluation import (
    collect_probabilities,
    evaluate_on_split,
    summarize_predictions_by_aoi,
)
from lake_segmentation.training.module import LakeSegmentationModule
from lake_segmentation.utils import seed_everything

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOGGER = get_logger("training.evaluate")


def run_evaluation(cfg: DictConfig) -> dict[str, Any]:
    """Report test metrics (overall and per lake) of the checkpoint in ``weights``."""

    setup_logging(force=True)
    seed_everything(int(cfg.seed))
    project_root = find_project_root()
    paths = resolve_paths(cfg)
    processed_dir = ensure_processed_data(cfg, project_root)

    bundle_path = paths["weights"] / BUNDLE_FILENAME
    bundle = load_bundle(bundle_path)
    checkpoint_path = paths["weights"] / Path(bundle["artifacts"]["checkpoint"]).name
    module = LakeSegmentationModule(
        model=bundle["model"],
        loss=dict(cfg.train.loss),
        optimizer={
            "learning_rate": float(cfg.train.learning_rate),
            "weight_decay": float(cfg.train.weight_decay),
            "encoder_learning_rate_scale": float(cfg.train.encoder_learning_rate_scale),
        },
        threshold=float(bundle["postprocessing"]["threshold"]),
        normalization=bundle["preprocessing"]["normalization"],
        scheduler=dict(cfg.train.scheduler),
    )
    state_dict = load_state_dict(checkpoint_path)
    normalized = {
        key[len("model.") :] if key.startswith("model.") else key: value
        for key, value in state_dict.items()
    }
    module.model.load_state_dict(normalized, strict=False)
    module.eval()

    datamodule = LakeSegmentationDataModule(
        processed_dir=processed_dir,
        stats=NormalizationStats.from_dict(bundle["preprocessing"]["normalization"]),
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        augment=False,
        seed=int(cfg.seed),
        reflectance_scale=float(cfg.preprocess.reflectance_scale),
        clip_min=float(cfg.preprocess.clip_min),
        clip_max=float(cfg.preprocess.clip_max),
    )
    datamodule.setup()

    probabilities, targets = collect_probabilities(module, datamodule.test_dataloader(), DEVICE)
    threshold = float(bundle["postprocessing"]["threshold"])
    metrics = evaluate_on_split(probabilities, targets, threshold)
    per_lake = summarize_predictions_by_aoi(
        datamodule.split_records["test"], probabilities, targets, threshold
    )
    LOGGER.info("Test IoU=%.4f Dice=%.4f", metrics["iou"], metrics["dice"])
    return {
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "test_metrics": metrics,
        "test_metrics_per_lake": per_lake,
    }
