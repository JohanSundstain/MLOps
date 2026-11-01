"""Publish the ONNX model to MLflow and serve it with ``mlflow models serve``.

This is the "MLflow Serving" variant of the inference server. The model signature describes
the tensor contract of the network:

* input ``image``: ``float32`` array ``(N, 4, 256, 256)`` of normalised reflectance;
* output ``probabilities``: ``float32`` array ``(N, 1, 256, 256)`` in ``[0, 1]``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from lake_segmentation.logging_utils import get_logger
from lake_segmentation.models.bundle import load_bundle

LOGGER = get_logger("serving.mlflow")


def _example_input(sample_shape: tuple[int, int, int, int]) -> np.ndarray:
    generator = np.random.default_rng(0)
    return generator.random(sample_shape, dtype=np.float32)


def _reference_output(onnx_path: Path, example: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: example})[0]


def publish_onnx_model(cfg: DictConfig) -> str:
    """Log the ONNX model with a signature and return its ``runs:/...`` URI."""

    import mlflow
    import mlflow.onnx
    from mlflow.models import infer_signature

    serving_cfg = cfg.serving
    onnx_path = Path(serving_cfg.onnx_model)
    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}. Run `python commands.py export-onnx` first."
        )
    bundle = load_bundle(Path(serving_cfg.bundle))
    tile_size = int(bundle["postprocessing"]["tile_size"])
    sample_shape = (1, len(bundle["preprocessing"]["band_order"]), tile_size, tile_size)
    example = _example_input(sample_shape)
    reference = _reference_output(onnx_path, example)
    signature = infer_signature(example, reference)

    import onnx

    onnx_model = onnx.load(str(onnx_path))
    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))
    registered_name = str(serving_cfg.get("registered_model_name") or "").strip() or None
    with mlflow.start_run(run_name="onnx-serving") as run:
        mlflow.log_params(
            {
                "threshold": float(bundle["postprocessing"]["threshold"]),
                "band_order": ",".join(bundle["preprocessing"]["band_order"]),
                "tile_size": tile_size,
            }
        )
        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            artifact_path="model",
            signature=signature,
            registered_model_name=registered_name,
        )
        run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/model"
    uri_file = Path(str(serving_cfg.model_uri_file))
    uri_file.parent.mkdir(parents=True, exist_ok=True)
    uri_file.write_text(model_uri + "\n", encoding="utf-8")

    LOGGER.info("Model logged to MLflow as %s", model_uri)
    LOGGER.info(
        "Serve it with:\n  mlflow models serve -m %s --host %s --port %d --env-manager local",
        model_uri,
        serving_cfg.host,
        int(serving_cfg.port),
    )
    LOGGER.info(
        "Then call it with:\n  curl -X POST http://%s:%d/invocations -H 'Content-Type: "
        "application/json' -d @request.json\nwhere request.json contains "
        '{"inputs": [[[...]]]} with shape %s',
        serving_cfg.host,
        int(serving_cfg.port),
        list(sample_shape),
    )
    return model_uri


def write_request_example(cfg: DictConfig, path: Path | None = None) -> Path:
    """Write a ready-to-send ``request.json`` for the served model."""

    bundle = load_bundle(Path(cfg.serving.bundle))
    tile_size = int(bundle["postprocessing"]["tile_size"])
    bands = len(bundle["preprocessing"]["band_order"])
    example = _example_input((1, bands, tile_size, tile_size))
    path = Path(path or Path(cfg.serving.model_uri_file).parent / "request.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"inputs": example.tolist()}), encoding="utf-8")
    LOGGER.info("Example request written to %s", path)
    return path
