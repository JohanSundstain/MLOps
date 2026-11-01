"""Model bundle: the single artefact that inference and export need.

A bundle is a small JSON file that lives next to the weights and describes everything required
to reproduce the preprocessing of a prediction: the model architecture, the normalisation
statistics of the training split, the band order, the reflectance scaling, the binarisation
threshold and the metrics of the run. It is the "model card" of the project.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping

from lake_segmentation.constants import BUNDLE_FILENAME
from lake_segmentation.logging_utils import get_logger
from lake_segmentation.utils import get_git_commit

LOGGER = get_logger("models.bundle")

BUNDLE_VERSION = 1


def build_bundle(
    model: Mapping[str, Any],
    normalization: Mapping[str, Any],
    threshold: float,
    band_order: list[str],
    reflectance_scale: float,
    clip_min: float,
    clip_max: float,
    tile_size: int,
    checkpoint: str,
    metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the bundle dictionary."""

    bundle: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "model": dict(model),
        "preprocessing": {
            "band_order": list(band_order),
            "reflectance_scale": float(reflectance_scale),
            "clip_min": float(clip_min),
            "clip_max": float(clip_max),
            "normalization": dict(normalization),
        },
        "postprocessing": {"threshold": float(threshold), "tile_size": int(tile_size)},
        "artifacts": {"checkpoint": checkpoint},
        "metrics": dict(metrics) if metrics else {},
    }
    if extra:
        bundle["extra"] = dict(extra)
    return bundle


def save_bundle(path: Path, bundle: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Model bundle written to %s", path)
    return path


def load_bundle(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Train the model first "
            "(`python commands.py train`)."
        )
    bundle = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model", "preprocessing", "postprocessing"):
        if key not in bundle:
            raise ValueError(f"Model bundle {path} has no {key!r} section")
    return bundle


def bundle_path_for(directory: Path) -> Path:
    return Path(directory) / BUNDLE_FILENAME
