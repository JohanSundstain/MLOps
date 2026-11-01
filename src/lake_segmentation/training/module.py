"""The Lightning module: model, loss, metrics, optimiser and scheduler."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from lightning.pytorch import LightningModule
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from lake_segmentation.logging_utils import get_logger
from lake_segmentation.metrics import evaluate_binary_masks
from lake_segmentation.models.losses import CombinedLoss
from lake_segmentation.models.unet import WaterSegmentationModel, split_parameter_groups
from lake_segmentation.training.metrics import build_metric_collection

LOGGER = get_logger("training.module")


def build_warmup_cosine_lambda(
    warmup_epochs: int, total_epochs: int, min_learning_rate: float, base_learning_rate: float
) -> Any:
    """Linear warm-up followed by a cosine decay down to ``min_learning_rate``."""

    minimum_factor = min(
        max(float(min_learning_rate) / max(float(base_learning_rate), 1e-12), 0.0), 1.0
    )
    warmup = max(int(warmup_epochs), 0)
    total = max(int(total_epochs), 1)

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = float(epoch - warmup) / float(max(total - warmup, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine

    return lr_lambda


class LakeSegmentationModule(LightningModule):
    """Binary water segmentation with a pretrained U-Net."""

    def __init__(
        self,
        model: Mapping[str, Any],
        loss: Mapping[str, Any],
        optimizer: Mapping[str, Any],
        threshold: float = 0.5,
        normalization: Mapping[str, Any] | None = None,
        scheduler: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = WaterSegmentationModel(model)
        self.loss_function = CombinedLoss(
            bce_weight=float(loss.get("bce_weight", 0.5)),
            dice_weight=float(loss.get("dice_weight", 0.5)),
            smooth=float(loss.get("smooth", 1.0)),
        )
        self.threshold = float(threshold)
        self.train_metrics = build_metric_collection("train/", threshold=self.threshold)
        self.val_metrics = build_metric_collection("val/", threshold=self.threshold)
        self.test_metrics = build_metric_collection("test/", threshold=self.threshold)

    # ------------------------------------------------------------------ forward
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    def predict_probabilities(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(image))

    # -------------------------------------------------------------------- steps
    def _shared_step(self, batch: Mapping[str, Any], stage: str) -> torch.Tensor:
        images = batch["image"]
        targets = batch["mask"]
        logits = self.model(images)
        loss = self.loss_function(logits, targets)
        probabilities = torch.sigmoid(logits)

        metrics = getattr(self, f"{stage}_metrics")
        predictions = (probabilities > self.threshold).to(torch.int32)
        metrics.update(predictions, targets.to(torch.int32))

        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
            sync_dist=False,
        )
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=stage == "val",
            batch_size=images.shape[0],
        )
        if stage == "train":
            components = self.loss_function.components(logits, targets)
            self.log(
                "train/bce",
                components["bce"],
                on_step=False,
                on_epoch=True,
                batch_size=images.shape[0],
            )
            self.log(
                "train/dice_loss",
                components["dice"],
                on_step=False,
                on_epoch=True,
                batch_size=images.shape[0],
            )
        return loss

    def training_step(self, batch: Mapping[str, Any], batch_index: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Mapping[str, Any], batch_index: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Mapping[str, Any], batch_index: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def predict_step(self, batch: Mapping[str, Any], batch_index: int) -> torch.Tensor:
        return self.predict_probabilities(batch["image"])

    def on_train_epoch_end(self) -> None:
        optimizer = self.optimizers()
        parameters = getattr(optimizer, "param_groups", None)
        if parameters:
            self.log(
                "train/learning_rate",
                float(parameters[0]["lr"]),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

    # ---------------------------------------------------------------- optimiser
    def configure_optimizers(self) -> dict[str, Any]:
        optimizer_cfg = self.hparams.optimizer
        base_learning_rate = float(optimizer_cfg.get("learning_rate", 3e-4))
        parameter_groups, parameter_counts = split_parameter_groups(
            self.model, float(optimizer_cfg.get("encoder_learning_rate_scale", 1.0))
        )
        groups = []
        for index, group in enumerate(parameter_groups):
            if not group["params"]:
                continue
            groups.append(
                {
                    "params": group["params"],
                    "lr": base_learning_rate * float(group["lr_scale"]),
                    "name": "encoder" if index == 1 else "decoder",
                }
            )
        LOGGER.info(
            "Optimiser groups: decoder=%d tensors, encoder=%d tensors (lr %.2e)",
            parameter_counts["head"],
            parameter_counts["encoder"],
            base_learning_rate,
        )
        optimizer = AdamW(
            groups,
            lr=base_learning_rate,
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": self._build_scheduler(optimizer, base_learning_rate),
        }

    def _build_scheduler(self, optimizer: Optimizer, base_learning_rate: float) -> dict[str, Any]:
        scheduler_cfg = self.hparams.get("scheduler") or {}
        name = str(scheduler_cfg.get("name", "cosine"))
        # ``self.trainer`` raises when the module is not attached yet; ``_trainer`` is None then.
        trainer = getattr(self, "_trainer", None)
        epochs = int(trainer.max_epochs) if trainer is not None else 1
        if name != "cosine":
            raise ValueError(f"Unknown scheduler {name!r}, only 'cosine' is implemented")
        lr_lambda = build_warmup_cosine_lambda(
            warmup_epochs=int(scheduler_cfg.get("warmup_epochs", 0)),
            total_epochs=epochs,
            min_learning_rate=float(scheduler_cfg.get("min_learning_rate", 1e-6)),
            base_learning_rate=base_learning_rate,
        )
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

    # ------------------------------------------------------------- evaluation
    def evaluate_probabilities(
        self, probabilities: list[torch.Tensor], targets: list[torch.Tensor]
    ) -> dict[str, float | int]:
        """Offline metrics with the same NumPy implementation as the inference report."""

        predictions_array = [
            (probability.detach().cpu().numpy().squeeze(0) > self.threshold)
            for probability in probabilities
        ]
        targets_array = [(target.detach().cpu().numpy().squeeze(0) > 0.5) for target in targets]
        return evaluate_binary_masks(predictions_array, targets_array)
