"""Reading and writing georeferenced rasters for inference.

Input format (documented in the README and represented by the example in ``data/example``):
one GeoTIFF per date with exactly four bands in the order B2, B3, B4, B8 (blue, green, red,
NIR) and 10 m pixels, either ``uint16`` surface reflectance scaled by 10 000 or ``float32``
reflectance in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from lake_segmentation.constants import NUM_BANDS, REFLECTANCE_SCALE
from lake_segmentation.logging_utils import get_logger

LOGGER = get_logger("inference.raster")
WGS84 = "EPSG:4326"


@dataclass
class RasterData:
    """Four-band reflectance raster on its native grid."""

    image: np.ndarray
    transform: Affine
    crs: CRS
    path: Path
    band_names: tuple[str, ...] = ("blue", "green", "red", "nir")

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[2]

    @property
    def bounds_wgs84(self) -> tuple[float, float, float, float]:
        left = self.transform.c
        top = self.transform.f
        right = left + self.transform.a * self.shape[1]
        bottom = top + self.transform.e * self.shape[0]
        return transform_bounds(self.crs, WGS84, left, bottom, right, top, densify_pts=21)


def read_raster(path: str | Path, reflectance_scale: float = REFLECTANCE_SCALE) -> RasterData:
    """Read a four-band GeoTIFF and bring it to float32 reflectance in ``[0, 1]``."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Raster not found: {path}")
    with rasterio.open(path) as source:
        if source.count != NUM_BANDS:
            raise ValueError(
                f"{path.name} has {source.count} bands, expected {NUM_BANDS} "
                "(B2, B3, B4, B8 in this order)"
            )
        if source.crs is None:
            raise ValueError(f"{path.name} has no coordinate reference system (no georeference)")
        image = source.read().astype(np.float32)
        transform = source.transform
        crs = source.crs

    if image.dtype.kind in {"i", "u"} or float(np.nanmax(image)) > 1.5:
        LOGGER.info("Treating %s as integer reflectance scaled by %s", path.name, reflectance_scale)
        image = image / float(reflectance_scale)
    else:
        LOGGER.info("Treating %s as float reflectance in [0, 1]", path.name)
    return RasterData(image=image, transform=transform, crs=crs, path=path)


def assert_comparable(first: RasterData, second: RasterData, tolerance: float = 1e-6) -> None:
    """Ensure that two rasters describe the same grid (required to compare areas)."""

    if first.crs != second.crs:
        raise ValueError(
            f"Rasters use different CRS: {first.crs.to_string()} vs {second.crs.to_string()}. "
            "Reproject one of them before running the comparison."
        )
    if first.shape != second.shape:
        raise ValueError(
            f"Rasters have different sizes: {first.shape} vs {second.shape}. "
            "Crop both rasters to the same area."
        )
    if (
        abs(first.transform.a - second.transform.a) > tolerance
        or abs(first.transform.e - second.transform.e) > tolerance
        or abs(first.transform.c - second.transform.c) > tolerance
        or abs(first.transform.f - second.transform.f) > tolerance
    ):
        raise ValueError(
            "Rasters have different geotransforms (pixel size or origin). "
            "Align them to the same grid before comparing."
        )


def crop_to_roi(
    raster: RasterData, roi_wgs84: Sequence[float]
) -> tuple[RasterData, tuple[int, int, int, int]]:
    """Crop a raster to a WGS84 bounding box and return the window offsets."""

    bounds = transform_bounds(WGS84, raster.crs, *roi_wgs84, densify_pts=21)
    window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], transform=raster.transform)
    window = (
        window.round_offsets()
        .round_lengths()
        .intersection(Window(0, 0, raster.shape[1], raster.shape[0]))
    )
    if window.width <= 0 or window.height <= 0:
        raise ValueError(f"ROI {roi_wgs84} does not overlap the raster {raster.path.name}")
    row_offset = int(window.row_off)
    col_offset = int(window.col_off)
    height = int(window.height)
    width = int(window.width)
    image = raster.image[:, row_offset : row_offset + height, col_offset : col_offset + width]
    transform = raster.transform * Affine.translation(col_offset, row_offset)
    cropped = RasterData(
        image=image,
        transform=transform,
        crs=raster.crs,
        path=raster.path,
        band_names=raster.band_names,
    )
    return cropped, (row_offset, col_offset, height, width)


def write_mask(
    path: Path,
    mask: np.ndarray,
    transform: Affine,
    crs: CRS,
    dtype: str = "uint8",
    nodata: int | float | None = 0,
) -> Path:
    """Write a single band mask (or probability map) as a compressed GeoTIFF."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 2 if dtype.startswith("float") else 1,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(mask.astype(dtype), 1)
    LOGGER.info("Wrote %s", path)
    return path
