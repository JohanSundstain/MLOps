"""Callbacks: metric history, figures and artefact upload."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

try:  # Lightning >= 2.6 renamed the class to ``MLFlowLogger`` (capital F)
    from lightning.pytorch.loggers import MLFlowLogger as MLflowLogger
except ImportError:  # pragma: no cover - older Lightning releases
    from lightning.pytorch.loggers import MLflowLogger  # type: ignore[assignment]

from lake_segmentation.logging_utils import get_logger
from lake_segmentation.plots import save_training_curves
from lake_segmentation.training.loggers import log_artifacts

LOGGER = get_logger("training.callbacks")

#: Series collected for the figures: name in the history -> possible metric keys.
HISTORY_KEYS: Mapping[str, tuple[str, ...]] = {
    "train_loss": ("train/loss_epoch", "train/loss"),
    "val_loss": ("val/loss", "val/loss_epoch"),
    "val_iou": ("val/iou",),
    "val_dice": ("val/dice",),
    "val_precision": ("val/precision",),
    "val_recall": ("val/recall",),
    "val_area_error": ("val/area_error",),
    "learning_rate": ("train/learning_rate",),
}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class HistoryCallback(Callback):
    """Collects epoch metrics, writes the figures and uploads them to MLflow."""

    def __init__(self, plots_dir: str | Path, prefix: str = "") -> None:
        super().__init__()
        self.plots_dir = Path(plots_dir)
        self.prefix = prefix
        self.history: dict[str, list[float]] = {key: [] for key in HISTORY_KEYS}
        self.history["epoch"] = []
        self.written: dict[str, Path] = {}

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # The sanity-check pass reports bogus metrics for epoch 0; it must not be recorded.
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        epoch = int(trainer.current_epoch)
        values: dict[str, float | None] = {}
        for name, candidates in HISTORY_KEYS.items():
            value = None
            for candidate in candidates:
                value = _as_float(metrics.get(candidate))
                if value is not None:
                    break
            values[name] = value

        if all(values[name] is None for name in HISTORY_KEYS):
            return

        self.history["epoch"].append(epoch)
        for name, value in values.items():
            self.history[name].append(value if value is not None else float("nan"))

        LOGGER.debug("epoch %d metrics collected: %s", epoch, values)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.history["epoch"]:
            LOGGER.warning("No metrics were collected, the figures are skipped")
            return
        self.written = save_training_curves(self.history, self.plots_dir)
        LOGGER.info("Training figures written: %s", sorted(self.written))
        if isinstance(trainer.logger, MLflowLogger):
            log_artifacts(trainer.logger, list(self.written.values()))
