"""Water segmentation predictor with two interchangeable backends (PyTorch and ONNX)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from omegaconf import DictConfig

from lake_segmentation.data.normalization import NormalizationStats, apply_normalization
from lake_segmentation.export.onnx import build_model_from_bundle
from lake_segmentation.logging_utils import get_logger
from lake_segmentation.models.bundle import load_bundle
from lake_segmentation.tiling import count_tiles, extract_tile, iter_tiles, stitch_probabilities

LOGGER = get_logger("inference.predictor")

SUPPORTED_BACKENDS = ("onnx", "torch")


class WaterSegmenter:
    """Tiles a raster, runs the network and stitches the probabilities back."""

    def __init__(
        self,
        bundle: Mapping[str, Any],
        backend: str = "onnx",
        onnx_path: Path | None = None,
        checkpoint_path: Path | None = None,
        device: str = "cpu",
        tile_size: int | None = None,
        tile_overlap: int = 0,
        batch_size: int = 8,
        padding_mode: str = "reflect",
    ) -> None:
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}, choose from {SUPPORTED_BACKENDS}")
        self.bundle = bundle
        self.backend = backend
        self.batch_size = int(batch_size)
        self.padding_mode = padding_mode
        preprocessing = bundle["preprocessing"]
        self.band_order = list(preprocessing["band_order"])
        self.stats = NormalizationStats.from_dict(preprocessing["normalization"])
        self.clip_min = float(preprocessing.get("clip_min", 0.0))
        self.clip_max = float(preprocessing.get("clip_max", 0.4))
        self.reflectance_scale = float(preprocessing.get("reflectance_scale", 10_000.0))
        postprocessing = bundle["postprocessing"]
        self.tile_size = int(tile_size or postprocessing.get("tile_size", 256))
        self.tile_overlap = int(tile_overlap)
        if self.tile_overlap >= self.tile_size:
            raise ValueError("tile_overlap must be smaller than tile_size")
        self.threshold = float(postprocessing["threshold"])
        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._torch_model: torch.nn.Module | None = None
        self._onnx_session: Any = None
        self._input_name = "image"
        self._output_name = "logits"

        if backend == "onnx":
            if onnx_path is None or not Path(onnx_path).is_file():
                raise FileNotFoundError(
                    "ONNX model not found. Export it with `python commands.py export-onnx` "
                    "or pass --backend=torch to use the checkpoint."
                )
            self._load_onnx(Path(onnx_path))
        else:
            if checkpoint_path is None or not Path(checkpoint_path).is_file():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}. Train the model first."
                )
            self._torch_model = build_model_from_bundle(bundle, Path(checkpoint_path))
            self._torch_model.eval()
            self._torch_model.to(self.device)

    # ------------------------------------------------------------------ backends
    def _load_onnx(self, path: Path) -> None:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.type == "cuda"
            else ["CPUExecutionProvider"]
        )
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 0
        self._onnx_session = ort.InferenceSession(
            str(path), sess_options=session_options, providers=providers
        )
        self._input_name = self._onnx_session.get_inputs()[0].name
        self._output_name = self._onnx_session.get_outputs()[0].name
        LOGGER.info("ONNX backend loaded from %s (providers: %s)", path, providers[0])

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        """Run one batch of normalised tiles through the active backend."""

        if self._onnx_session is not None:
            outputs = self._onnx_session.run([self._output_name], {self._input_name: batch})
            return np.asarray(outputs[0], dtype=np.float32)
        assert self._torch_model is not None
        tensor = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            logits = self._torch_model(tensor)
            return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    # --------------------------------------------------------------- prediction
    def predict(self, image: np.ndarray, progress_description: str = "segmentation") -> np.ndarray:
        """Return probabilities ``(height, width)`` for a ``(4, height, width)`` raster."""

        if image.ndim != 3:
            raise ValueError(f"Expected (bands, height, width), got {image.shape}")
        if image.shape[0] != len(self.band_order):
            raise ValueError(
                f"Raster has {image.shape[0]} bands but the model expects "
                f"{len(self.band_order)} ({self.band_order})"
            )
        height, width = image.shape[1], image.shape[2]
        stride = self.tile_size - self.tile_overlap
        tiles = list(iter_tiles(height, width, self.tile_size, stride))
        LOGGER.info(
            "Segmenting %dx%d raster in %d tiles of %d px (overlap %d)",
            height,
            width,
            count_tiles(height, width, self.tile_size, stride),
            self.tile_size,
            self.tile_overlap,
        )

        probabilities: list[np.ndarray] = []
        batch: list[np.ndarray] = []
        from tqdm import tqdm

        for tile in tqdm(tiles, desc=progress_description, leave=False):
            tile_values = extract_tile(
                image, tile, self.tile_size, mode=self.padding_mode, fill_value=0.0
            )
            batch.append(tile_values)
            if len(batch) == self.batch_size:
                probabilities.extend(self._run_batch(np.stack(batch)))
                batch = []
        if batch:
            probabilities.extend(self._run_batch(np.stack(batch)))

        return stitch_probabilities(probabilities, tiles, height, width)

    def _run_batch(self, batch: np.ndarray) -> list[np.ndarray]:
        normalized = apply_normalization(
            np.clip(batch, self.clip_min, self.clip_max).astype(np.float32), self.stats
        )
        output = self._forward(np.ascontiguousarray(normalized))
        if output.ndim != 4 or output.shape[1] != 1:
            raise RuntimeError(f"Unexpected model output shape {output.shape}")
        return [output[index, 0] for index in range(output.shape[0])]

    def segment(self, image: np.ndarray, progress_description: str = "segmentation") -> np.ndarray:
        """Probabilities -> binary water mask at the bundle threshold."""

        return (self.predict(image, progress_description) > self.threshold).astype(np.uint8)


def build_segmenter(cfg: DictConfig, backend: str | None = None) -> WaterSegmenter:
    """Create a :class:`WaterSegmenter` from the inference config."""

    bundle_path = Path(cfg.infer.bundle)
    bundle = load_bundle(bundle_path)
    selected_backend = backend or str(cfg.infer.backend)
    device = str(cfg.infer.get("device", "cpu"))
    return WaterSegmenter(
        bundle=bundle,
        backend=selected_backend,
        onnx_path=Path(cfg.infer.onnx_model),
        checkpoint_path=Path(cfg.infer.checkpoint),
        device=device,
        tile_size=int(cfg.infer.tile_size),
        tile_overlap=int(cfg.infer.get("tile_overlap", 0)),
        batch_size=int(cfg.infer.batch_size),
        padding_mode=str(cfg.infer.get("padding_mode", "reflect")),
    )


def probability_to_mask(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Binarise a probability map."""

    return (probabilities > float(threshold)).astype(np.uint8)


def mask_bands(
    image: np.ndarray, band_order: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the green, red and NIR channels in the order needed for NDWI and RGB previews."""

    indices = {name: index for index, name in enumerate(band_order)}
    return image[indices["green"]], image[indices["red"]], image[indices["nir"]]
