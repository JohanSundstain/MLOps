"""ONNX export of the trained segmentation model.

The exported graph contains the sigmoid, so the artefact returns probabilities in ``[0, 1]``.
The export is validated against PyTorch with ``onnxruntime`` before it is accepted, and the
model metadata (threshold, bands, git commit) is embedded into the ONNX file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import DictConfig

from lake_segmentation.logging_utils import get_logger
from lake_segmentation.models.bundle import load_bundle
from lake_segmentation.models.unet import ProbabilityModel, WaterSegmentationModel
from lake_segmentation.utils import get_git_commit

LOGGER = get_logger("export.onnx")


def load_state_dict(checkpoint_path: Path) -> Mapping[str, Any]:
    """Load a state dict from a Lightning checkpoint or a plain ``torch.save`` file."""

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")
    return checkpoint


def build_model_from_bundle(
    bundle: Mapping[str, Any], checkpoint_path: Path
) -> WaterSegmentationModel:
    """Rebuild the network from the bundle and load the trained weights."""

    model = WaterSegmentationModel(bundle["model"])
    state_dict = load_state_dict(checkpoint_path)
    # Checkpoints saved by ModelCheckpoint may carry the "model." prefix of the wrapper.
    normalized = {}
    for key, value in state_dict.items():
        normalized[key[len("model.") :] if key.startswith("model.") else key] = value
    missing, unexpected = model.load_state_dict(normalized, strict=False)
    if missing:
        raise RuntimeError(f"Missing weights when loading the checkpoint: {missing[:5]}")
    if unexpected:
        LOGGER.warning("Unexpected keys in the checkpoint (ignored): %s", unexpected[:5])
    model.eval()
    return model


def _dynamic_axes(
    input_name: str, output_name: str, dynamic_batch: bool, dynamic_spatial: bool
) -> dict[str, dict[int, str]] | None:
    input_axes: dict[int, str] = {}
    output_axes: dict[int, str] = {}
    if dynamic_batch:
        input_axes[0] = "batch"
        output_axes[0] = "batch"
    if dynamic_spatial:
        input_axes[2] = "height"
        input_axes[3] = "width"
        output_axes[2] = "height"
        output_axes[3] = "width"
    if not input_axes and not output_axes:
        return None
    return {input_name: input_axes, output_name: output_axes}


def _attach_metadata(onnx_path: Path, metadata: Mapping[str, str]) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    existing = {entry.key for entry in model.metadata_props}
    for key, value in metadata.items():
        if key not in existing:
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = str(value)
    onnx.save(model, str(onnx_path))


def _validate_onnx(
    torch_model: torch.nn.Module,
    onnx_path: Path,
    sample_input: torch.Tensor,
    input_name: str,
    output_name: str,
    atol: float,
) -> float:
    import onnxruntime as ort

    with torch.no_grad():
        reference = torch_model(sample_input).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run([output_name], {input_name: sample_input.numpy()})
    difference = float(np.max(np.abs(outputs[0] - reference)))
    LOGGER.info("ONNX vs PyTorch max |difference| = %.3e (tolerance %.1e)", difference, atol)
    if not np.isfinite(difference) or difference > atol:
        raise RuntimeError(
            f"The exported ONNX model deviates from PyTorch by {difference:.3e} "
            f"(tolerance {atol:.1e})"
        )
    return difference


def export_onnx(
    cfg: DictConfig,
    checkpoint_path: Path | None = None,
    bundle_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Export the checkpoint to ONNX and validate the result."""

    export_cfg = cfg.export
    checkpoint_path = Path(checkpoint_path or export_cfg.checkpoint)
    bundle_path = Path(bundle_path or export_cfg.bundle)
    output_path = Path(output_path or export_cfg.output)

    bundle = load_bundle(bundle_path)
    model = build_model_from_bundle(bundle, checkpoint_path)
    export_model = ProbabilityModel(model).eval()

    sample_input = torch.randn(*[int(value) for value in export_cfg.sample_input])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = _dynamic_axes(
        str(export_cfg.input_name),
        str(export_cfg.output_name),
        bool(export_cfg.dynamic_batch),
        bool(export_cfg.dynamic_spatial),
    )

    LOGGER.info(
        "Exporting %s to %s (opset %s)", checkpoint_path.name, output_path, export_cfg.opset
    )
    export_kwargs = {
        "input_names": [str(export_cfg.input_name)],
        "output_names": [str(export_cfg.output_name)],
        "opset_version": int(export_cfg.opset),
        "dynamic_axes": dynamic_axes,
        "do_constant_folding": True,
    }
    try:
        torch.onnx.export(
            export_model, sample_input, str(output_path), dynamo=False, **export_kwargs
        )
    except TypeError:
        torch.onnx.export(export_model, sample_input, str(output_path), **export_kwargs)

    import onnx

    onnx.checker.check_model(str(output_path))
    _attach_metadata(
        output_path,
        {
            "threshold": f"{float(bundle['postprocessing']['threshold'])}",
            "band_order": ",".join(bundle["preprocessing"]["band_order"]),
            "reflectance_scale": f"{float(bundle['preprocessing']['reflectance_scale'])}",
            "git_commit": get_git_commit(),
            "input_layout": "NCHW",
            "output": "probabilities",
        },
    )

    if bool(export_cfg.validate):
        _validate_onnx(
            torch_model=export_model,
            onnx_path=output_path,
            sample_input=sample_input,
            input_name=str(export_cfg.input_name),
            output_name=str(export_cfg.output_name),
            atol=float(export_cfg.atol),
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    LOGGER.info("ONNX model written to %s (%.1f MiB)", output_path, size_mb)
    return output_path


def describe_onnx(path: Path) -> dict[str, Any]:
    """Inspect inputs and outputs of an ONNX model (used by the CLI and the README)."""

    import onnx

    model = onnx.load(str(Path(path)))
    graph = model.graph

    def _shape(value_info: Any) -> list[str]:
        dimensions = []
        for dimension in value_info.type.tensor_type.shape.dim:
            dimensions.append(dimension.dim_param or str(dimension.dim_value))
        return dimensions

    return {
        "inputs": [{"name": value.name, "shape": _shape(value)} for value in graph.input],
        "outputs": [{"name": value.name, "shape": _shape(value)} for value in graph.output],
        "opset": [f"{entry.domain or 'ai.onnx'}:{entry.version}" for entry in model.opset_import],
        "size_bytes": Path(path).stat().st_size,
        "metadata": {entry.key: entry.value for entry in model.metadata_props},
    }
