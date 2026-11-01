"""Model factory: U-Net with a ResNet34 encoder adapted to four Sentinel-2 bands."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from lake_segmentation.logging_utils import get_logger

LOGGER = get_logger("models")


def build_model(model_cfg: Mapping[str, Any] | DictConfig) -> nn.Module:
    """Instantiate the segmentation network described by ``cfg.model``.

    ``segmentation-models-pytorch`` adapts the pretrained RGB encoder to ``in_channels``
    channels: the weights of the first convolution are averaged over the three colour
    channels and replicated for the remaining bands. The adaptation happens inside the
    library, so training simply fine-tunes the adapted network.
    """

    import segmentation_models_pytorch as smp

    architecture = str(model_cfg["architecture"])
    if not hasattr(smp, architecture):
        raise ValueError(f"Unknown architecture {architecture!r} for segmentation_models_pytorch")
    model_class = getattr(smp, architecture)
    encoder_weights = model_cfg.get("encoder_weights")
    if encoder_weights in (None, "none", "None", ""):
        encoder_weights = None
    model = model_class(
        encoder_name=str(model_cfg["encoder_name"]),
        encoder_weights=encoder_weights,
        in_channels=int(model_cfg["in_channels"]),
        classes=int(model_cfg["classes"]),
        activation=model_cfg.get("activation"),
    )
    LOGGER.info(
        "Built %s with encoder %s (%s weights), %d input channels",
        architecture,
        model_cfg["encoder_name"],
        encoder_weights or "random",
        int(model_cfg["in_channels"]),
    )
    return model


class WaterSegmentationModel(nn.Module):
    """Thin wrapper that documents the input/output contract of the network.

    * input: ``float32`` tensor ``(batch, 4, height, width)`` of normalised reflectance;
    * output: ``float32`` logits ``(batch, 1, height, width)``.
    """

    def __init__(self, model_cfg: Mapping[str, Any] | DictConfig) -> None:
        super().__init__()
        self.config = dict(model_cfg)
        self.network = build_model(model_cfg)

    def forward(self, image: Tensor) -> Tensor:
        return self.network(image)

    def predict_probabilities(self, image: Tensor) -> Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.network(image))


class ProbabilityModel(nn.Module):
    """Export wrapper that bakes the sigmoid into the graph.

    The ONNX / TensorRT artefacts therefore return probabilities in ``[0, 1]`` and the client
    only has to compare them with the threshold from the model bundle.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: Tensor) -> Tensor:
        return torch.sigmoid(self.model(image))


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def split_parameter_groups(
    model: nn.Module, encoder_learning_rate_scale: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Separate the pretrained encoder from the rest of the network.

    The ImageNet encoder is fine-tuned with a smaller learning rate than the randomly
    initialised decoder.
    """

    encoder_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        # The wrapper module keeps the network in ``self.network``.
        encoder = getattr(getattr(model, "network", None), "encoder", None)
    encoder_ids = {id(parameter) for parameter in encoder.parameters()} if encoder else set()
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in encoder_ids:
            encoder_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    return [
        {"params": head_parameters, "lr_scale": 1.0},
        {"params": encoder_parameters, "lr_scale": float(encoder_learning_rate_scale)},
    ], {"encoder": len(encoder_parameters), "head": len(head_parameters)}
