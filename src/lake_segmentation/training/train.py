"""Training entry point: dataset, model, Lightning trainer, evaluation and artefacts."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from omegaconf import DictConfig

from lake_segmentation.config import find_project_root, resolve_paths
from lake_segmentation.constants import BUNDLE_FILENAME, NORM_STATS_FILENAME
from lake_segmentation.data.dvc_utils import ensure_processed_data
from lake_segmentation.data.normalization import load_stats
from lake_segmentation.data.pipeline import run_preprocess
from lake_segmentation.export.onnx import export_onnx
from lake_segmentation.logging_utils import get_logger, setup_logging
from lake_segmentation.models.bundle import build_bundle, save_bundle
from lake_segmentation.training.callbacks import HistoryCallback
from lake_segmentation.training.datamodule import LakeSegmentationDataModule
from lake_segmentation.training.evaluation import (
    collect_probabilities,
    evaluate_on_split,
    save_validation_predictions,
    summarize_predictions_by_aoi,
    tune_threshold,
)
from lake_segmentation.training.loggers import build_logger, log_artifacts, log_run_metadata
from lake_segmentation.training.module import LakeSegmentationModule
from lake_segmentation.utils import get_git_commit, seed_everything

LOGGER = get_logger("training.train")

CHECKPOINT_FILENAME = "lake_unet.ckpt"
ONNX_FILENAME = "lake_unet.onnx"


def resolve_precision(precision: str) -> str:
    """Downgrade mixed precision to fp32 when no GPU is available."""

    requested = str(precision)
    if requested in {"16-mixed", "bf16-mixed", "16", "bf16"} and not torch.cuda.is_available():
        LOGGER.warning("Requested precision %s requires a GPU, using 32-true instead", requested)
        return "32-true"
    return requested


def prepare_dataset(cfg: DictConfig, project_root: Path) -> Path:
    """Return the processed dataset directory, rebuilding it when it is missing."""

    try:
        return ensure_processed_data(cfg, project_root)
    except FileNotFoundError as error:
        if not bool(cfg.train.get("auto_preprocess", True)):
            raise
        LOGGER.warning("%s", error)
        LOGGER.info("Rebuilding the dataset from the public sources")
        run_preprocess(cfg)
        return ensure_processed_data(cfg, project_root)


def build_trainer(
    cfg: DictConfig, paths: dict[str, Path], logger: Any, callbacks: list[Any]
) -> Trainer:
    train_cfg = cfg.train
    return Trainer(
        max_epochs=int(train_cfg.max_epochs),
        accelerator=str(train_cfg.accelerator),
        devices=train_cfg.devices,
        precision=resolve_precision(str(train_cfg.precision)),
        gradient_clip_val=float(train_cfg.get("gradient_clip_val", 0.0)),
        log_every_n_steps=int(train_cfg.log_every_n_steps),
        limit_train_batches=float(train_cfg.get("limit_train_batches", 1.0)),
        limit_val_batches=float(train_cfg.get("limit_val_batches", 1.0)),
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(paths["logs"]),
        deterministic=bool(train_cfg.get("deterministic", False)),
        enable_model_summary=True,
        num_sanity_val_steps=1,
    )


def run_training(cfg: DictConfig) -> dict[str, Any]:
    """Full training pipeline. Returns a summary dictionary for the CLI."""

    project_root = find_project_root()
    paths = resolve_paths(cfg)
    setup_logging(log_file=paths["plots"] / "train.log", force=True)
    seed_everything(int(cfg.seed))
    LOGGER.info("Starting training (git commit %s)", get_git_commit())

    processed_dir = prepare_dataset(cfg, project_root)
    stats = load_stats(processed_dir / NORM_STATS_FILENAME)

    datamodule = LakeSegmentationDataModule(
        processed_dir=processed_dir,
        stats=stats,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        augment=bool(cfg.train.augment),
        seed=int(cfg.seed),
        reflectance_scale=float(cfg.preprocess.reflectance_scale),
        clip_min=float(cfg.preprocess.clip_min),
        clip_max=float(cfg.preprocess.clip_max),
    )
    datamodule.setup()
    LOGGER.info("Mean water fraction per split: %s", datamodule.class_balance())

    run_name = f"{cfg.mlflow.run_name}-{dt.datetime.now():%Y%m%d-%H%M%S}"
    logger, is_mlflow = build_logger(cfg, save_dir=paths["logs"], run_name=run_name)
    log_run_metadata(logger, cfg)

    module = LakeSegmentationModule(
        model=dict(cfg.model),
        loss=dict(cfg.train.loss),
        optimizer={
            "learning_rate": float(cfg.train.learning_rate),
            "weight_decay": float(cfg.train.weight_decay),
            "encoder_learning_rate_scale": float(cfg.train.encoder_learning_rate_scale),
        },
        threshold=float(cfg.train.threshold),
        normalization=stats.to_dict(),
        scheduler=dict(cfg.train.scheduler),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(paths["weights"]),
        filename=str(cfg.train.checkpoint.filename),
        monitor=str(cfg.train.checkpoint.monitor),
        mode=str(cfg.train.checkpoint.mode),
        save_top_k=int(cfg.train.checkpoint.save_top_k),
        save_weights_only=True,
        auto_insert_metric_name=False,
    )
    callbacks: list[Any] = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    history_callback = HistoryCallback(plots_dir=paths["plots"])
    callbacks.append(history_callback)
    if bool(cfg.train.early_stopping.enabled):
        callbacks.append(
            EarlyStopping(
                monitor=str(cfg.train.early_stopping.monitor),
                mode=str(cfg.train.early_stopping.mode),
                patience=int(cfg.train.early_stopping.patience),
            )
        )

    trainer = build_trainer(cfg, paths, logger, callbacks)
    trainer.fit(module, datamodule=datamodule)

    best_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    if not best_path:
        raise RuntimeError("Training finished without producing a checkpoint")
    LOGGER.info(
        "Best checkpoint: %s (val/iou=%.4f)",
        best_path,
        float(checkpoint_callback.best_model_score or float("nan")),
    )

    # Evaluate and tune the binarisation threshold on the validation split.
    device = module.device
    checkpoint_state = torch.load(best_path, map_location="cpu", weights_only=False)
    module.load_state_dict(checkpoint_state["state_dict"])
    val_probabilities, val_targets = collect_probabilities(
        module, datamodule.val_dataloader(), device
    )
    tuning = tune_threshold(
        val_probabilities,
        val_targets,
        threshold_min=float(cfg.train.get("threshold_min", 0.2)),
        threshold_max=float(cfg.train.get("threshold_max", 0.8)),
        steps=int(cfg.train.get("threshold_steps", 25)),
    )
    module.threshold = float(tuning["threshold"])
    val_metrics = evaluate_on_split(val_probabilities, val_targets, tuning["threshold"])

    test_probabilities, test_targets = collect_probabilities(
        module, datamodule.test_dataloader(), device
    )
    test_metrics = evaluate_on_split(test_probabilities, test_targets, tuning["threshold"])
    test_by_aoi = summarize_predictions_by_aoi(
        datamodule.split_records["test"], test_probabilities, test_targets, tuning["threshold"]
    )
    LOGGER.info("Test metrics: IoU=%.4f Dice=%.4f", test_metrics["iou"], test_metrics["dice"])
    LOGGER.info("Test metrics per lake: %s", test_by_aoi)

    trainer.test(module, datamodule=datamodule)
    if is_mlflow:
        logger.log_metrics(
            {
                "test/iou": float(test_metrics["iou"]),
                "test/dice": float(test_metrics["dice"]),
                "test/precision": float(test_metrics["precision"]),
                "test/recall": float(test_metrics["recall"]),
                "test/area_error_percent": float(test_metrics["area_error_percent"]),
                "val/iou_at_best_threshold": float(val_metrics["iou"]),
                "val/best_threshold": float(tuning["threshold"]),
            }
        )

    # Persist the artefacts: weights, bundle and figures.
    checkpoint_path = paths["weights"] / CHECKPOINT_FILENAME
    torch.save(
        {
            "state_dict": module.model.state_dict(),
            "threshold": float(tuning["threshold"]),
            "git_commit": get_git_commit(),
        },
        checkpoint_path,
    )
    LOGGER.info(
        "Weights saved to %s (%.1f MiB)",
        checkpoint_path,
        checkpoint_path.stat().st_size / (1024 * 1024),
    )

    bundle = build_bundle(
        model=dict(cfg.model),
        normalization=stats.to_dict(),
        threshold=float(tuning["threshold"]),
        band_order=list(cfg.data.stac.band_order),
        reflectance_scale=float(cfg.preprocess.reflectance_scale),
        clip_min=float(cfg.preprocess.clip_min),
        clip_max=float(cfg.preprocess.clip_max),
        tile_size=int(cfg.data.patch.size),
        checkpoint=CHECKPOINT_FILENAME,
        metrics={
            "test": test_metrics,
            "validation": val_metrics,
            "per_lake_test": test_by_aoi,
            "threshold_search": tuning["table"],
        },
        extra={
            "run_name": run_name,
            "mlflow_tracking_uri": str(cfg.mlflow.tracking_uri),
            "best_val_iou_torchmetrics": float(checkpoint_callback.best_model_score or 0.0),
        },
    )
    bundle_path = save_bundle(paths["weights"] / BUNDLE_FILENAME, bundle)

    save_validation_predictions(
        records=datamodule.split_records["val"],
        image_dir=processed_dir,
        probabilities=val_probabilities,
        targets=val_targets,
        output_path=paths["plots"] / "val_predictions.png",
        band_order=list(cfg.data.stac.band_order),
        threshold=float(tuning["threshold"]),
    )
    artifacts = sorted(paths["plots"].glob("*.png")) + sorted(paths["plots"].glob("*.csv"))
    log_artifacts(logger, artifacts)

    summary: dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": str(checkpoint_path),
        "bundle": str(bundle_path),
        "threshold": float(tuning["threshold"]),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_per_lake": test_by_aoi,
        "figures": [str(path) for path in artifacts],
        "mlflow_run_id": getattr(logger, "run_id", None) if is_mlflow else None,
    }

    if bool(cfg.train.get("export_onnx_after_training", True)):
        onnx_path = export_onnx(
            cfg,
            checkpoint_path=checkpoint_path,
            bundle_path=bundle_path,
            output_path=paths["weights"] / ONNX_FILENAME,
        )
        summary["onnx_model"] = str(onnx_path)

    LOGGER.info("Training finished: %s", summary["checkpoint"])
    return summary
