"""Build a small example pair of four-band GeoTIFFs for the inference demo.

The files are cut from the same Sentinel-2 scenes the dataset is built from, so the example is
representative of the input format described in the README: four bands (B2, B3, B4, B8) in
``uint16`` surface reflectance, 10 m pixels, georeferenced in the UTM CRS of the scene.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from omegaconf import DictConfig
from rasterio.transform import Affine

from lake_segmentation.config import resolve_path
from lake_segmentation.constants import REFLECTANCE_SCALE
from lake_segmentation.data.gdal_env import configure_gdal
from lake_segmentation.data.scenes import read_aoi_stack
from lake_segmentation.data.stac import AreaOfInterest, aois_from_config, select_scene
from lake_segmentation.logging_utils import get_logger

LOGGER = get_logger("data.example")


def _write_geotiff(path: Path, image: np.ndarray, transform: Affine, crs: Any) -> Path:
    profile = {
        "driver": "GTiff",
        "height": image.shape[1],
        "width": image.shape[2],
        "count": image.shape[0],
        "dtype": "uint16",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(image.astype(np.uint16))
        destination.set_band_description(1, "B2 (blue)")
        destination.set_band_description(2, "B3 (green)")
        destination.set_band_description(3, "B4 (red)")
        destination.set_band_description(4, "B8 (NIR)")
    return path


def build_example_pair(
    cfg: DictConfig,
    aoi_name: str = "chany",
    size: int = 1024,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Cut a ``size`` x ``size`` window of the first two configured dates of one AOI."""

    configure_gdal()
    destination_dir = Path(output_dir) if output_dir else resolve_path("data/example")
    aois = {aoi.name: aoi for aoi in aois_from_config(cfg)}
    if aoi_name not in aois:
        raise KeyError(f"Unknown AOI {aoi_name!r}, available: {sorted(aois)}")
    aoi: AreaOfInterest = aois[aoi_name]
    dates = list(aoi.dates)[:2]
    if len(dates) < 2:
        raise ValueError(f"AOI {aoi_name} needs at least two dates to build a comparison example")

    written: list[dict[str, Any]] = []
    for target_date in dates:
        reference = select_scene(aoi, target_date, cfg)
        stack = read_aoi_stack(
            reference=reference,
            bbox_wgs84=aoi.bbox,
            band_order=list(cfg.data.stac.band_order),
            patch_size=int(cfg.data.patch.size),
        )
        height, width = stack.shape
        window = min(size, height, width)
        row_offset = (height - window) // 2
        col_offset = (width - window) // 2
        image = stack.image[:, row_offset : row_offset + window, col_offset : col_offset + window]
        transform = stack.transform * Affine.translation(col_offset, row_offset)
        path = destination_dir / f"{aoi_name}_{target_date}.tif"
        _write_geotiff(path, image, transform, stack.crs)
        written.append(
            {
                "path": str(path),
                "date": target_date,
                "scene_id": reference.item_id,
                "shape": [int(window), int(window)],
                "crs": stack.crs.to_string(),
                "size_mb": round(path.stat().st_size / 1e6, 2),
            }
        )
        LOGGER.info("Example raster written to %s (%d x %d)", path, window, window)

    manifest = {
        "description": "Example pair of Sentinel-2 GeoTIFFs for `python commands.py infer`",
        "aoi": aoi_name,
        "band_order": list(cfg.data.stac.band_order),
        "dtype": "uint16",
        "reflectance_scale": REFLECTANCE_SCALE,
        "files": written,
    }
    manifest_path = destination_dir / "example_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"files": written, "manifest": str(manifest_path)}
