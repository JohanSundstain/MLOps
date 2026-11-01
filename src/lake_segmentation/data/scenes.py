"""Reading a Sentinel-2 stack (B2, B3, B4, B8 + SCL) for one area of interest.

The scenes are read window by window directly from the remote Cloud Optimized GeoTIFFs, so
nothing but the AOI itself is transferred. The result is an ``AOI x 4 bands`` array on the
native 10 m grid of the scene, aligned to whole pixels, together with a validity mask built
from the scene classification layer (clouds, shadows, no-data, ...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from lake_segmentation.constants import BAND_NAMES, SCL_INVALID_CLASSES
from lake_segmentation.data.gdal_env import configure_gdal
from lake_segmentation.data.stac import SceneReference
from lake_segmentation.logging_utils import get_logger

LOGGER = get_logger("data.scenes")

WGS84 = "EPSG:4326"


@dataclass
class BandStack:
    """Aligned multi-band raster of one AOI."""

    image: np.ndarray
    scl: np.ndarray | None
    valid: np.ndarray
    transform: Affine
    crs: CRS
    reference: SceneReference

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[2]

    @property
    def band_names(self) -> tuple[str, ...]:
        return BAND_NAMES


def snap_bounds_to_grid(
    bounds: Sequence[float], resolution_x: float, resolution_y: float
) -> tuple[float, float, float, float]:
    """Extend bounds to whole pixels of a grid with the given resolution."""

    left, bottom, right, top = bounds
    left = math.floor(left / resolution_x) * resolution_x
    right = math.ceil(right / resolution_x) * resolution_x
    bottom = math.floor(bottom / resolution_y) * resolution_y
    top = math.ceil(top / resolution_y) * resolution_y
    return left, bottom, right, top


def compute_aoi_window(
    source: rasterio.DatasetReader, bbox_wgs84: Sequence[float]
) -> tuple[Window, Affine, CRS]:
    """Window in ``source`` that covers ``bbox_wgs84`` and its transform."""

    bounds = transform_bounds(WGS84, source.crs, *bbox_wgs84, densify_pts=21)
    resolution_x = abs(source.transform.a)
    resolution_y = abs(source.transform.e)
    snapped = snap_bounds_to_grid(bounds, resolution_x, resolution_y)
    window = from_bounds(snapped[0], snapped[1], snapped[2], snapped[3], transform=source.transform)
    window = (
        window.round_offsets()
        .round_lengths()
        .intersection(Window(0, 0, source.width, source.height))
    )
    if window.width <= 0 or window.height <= 0:
        raise ValueError(
            "The AOI does not overlap the scene: "
            f"aoi bounds {tuple(round(value, 4) for value in bbox_wgs84)}"
        )
    return window, source.window_transform(window), source.crs


def pad_to_multiple(
    array: np.ndarray, multiple: int, mode: str = "edge", fill_value: float = 0.0
) -> np.ndarray:
    """Pad the two trailing dimensions of ``array`` to a multiple of ``multiple``."""

    height, width = array.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    if pad_height == 0 and pad_width == 0:
        return array
    pad_spec = [(0, 0)] * (array.ndim - 2) + [(0, pad_height), (0, pad_width)]
    if mode == "constant":
        return np.pad(array, pad_spec, mode="constant", constant_values=fill_value)
    return np.pad(array, pad_spec, mode=mode)


def _read_window(
    href: str,
    window: Window | None,
    out_shape: tuple[int, int] | None,
    resampling: Resampling,
    dtype: str,
) -> np.ndarray:
    """Read one band of a remote COG as a 2D array."""

    with rasterio.open(href) as source:
        array = source.read(
            1,
            window=window,
            out_shape=out_shape,
            resampling=resampling,
            boundless=True,
            fill_value=0,
        )
    return array.astype(dtype, copy=False)


def read_aoi_stack(
    reference: SceneReference,
    bbox_wgs84: Sequence[float],
    band_order: Sequence[str],
    patch_size: int,
    invalid_scl_classes: Sequence[int] = SCL_INVALID_CLASSES,
    max_nodata_fraction: float = 1.0,
) -> BandStack:
    """Read all bands of one scene for one AOI and stack them into a ``4 x H x W`` array."""

    configure_gdal()
    missing = [band for band in band_order if band not in reference.assets]
    if missing:
        raise KeyError(f"Scene {reference.item_id} has no assets for bands {missing}")

    with rasterio.open(reference.assets[band_order[0]]) as first:
        window, transform, crs = compute_aoi_window(first, bbox_wgs84)
        height, width = int(window.height), int(window.width)

    bands: list[np.ndarray] = []
    for band_name in band_order:
        array = _read_window(
            reference.assets[band_name], window, (height, width), Resampling.nearest, "uint16"
        )
        if array.shape != (height, width):
            raise RuntimeError(
                f"Band {band_name} of {reference.item_id} has shape {array.shape}, "
                f"expected {(height, width)}"
            )
        bands.append(array)
    image = np.stack(bands, axis=0)

    scl: np.ndarray | None = None
    scl_href = reference.assets.get("scl")
    if scl_href is not None:
        # The scene classification layer is delivered at 20 m: it needs its own window on its
        # own grid and is then resampled onto the 10 m grid of the spectral bands.
        with rasterio.open(scl_href) as scl_source:
            scl_window, _, _ = compute_aoi_window(scl_source, bbox_wgs84)
            scl = scl_source.read(
                1,
                window=scl_window,
                out_shape=(height, width),
                resampling=Resampling.nearest,
                boundless=True,
                fill_value=0,
            ).astype(np.uint8, copy=False)
        validity = ~np.isin(scl, np.asarray(invalid_scl_classes, dtype=np.uint8))
    else:
        LOGGER.warning(
            "Scene %s has no SCL asset, all pixels are treated as valid", reference.item_id
        )
        validity = np.ones((height, width), dtype=bool)

    # Reflectance of 0 means "no data" in the L2A product.
    validity &= np.all(image > 0, axis=0)

    padded_image = pad_to_multiple(image, patch_size, mode="edge")
    padded_validity = pad_to_multiple(validity, patch_size, mode="constant", fill_value=False)
    padded_scl = (
        pad_to_multiple(scl, patch_size, mode="constant", fill_value=0) if scl is not None else None
    )

    nodata_fraction = 1.0 - float(padded_validity.mean())
    if nodata_fraction > max_nodata_fraction:
        raise ValueError(
            f"AOI {reference.aoi_name} is mostly invalid in scene {reference.item_id} "
            f"({nodata_fraction:.1%} invalid pixels); pick another date or AOI"
        )

    return BandStack(
        image=padded_image,
        scl=padded_scl,
        valid=padded_validity,
        transform=transform,
        crs=crs,
        reference=reference,
    )


def save_stack_cache(path: Path, stack: BandStack) -> Path:
    """Cache an AOI stack on disk (used when ``data.download.cache_stacks`` is enabled)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = stack.transform
    np.savez_compressed(
        path,
        image=stack.image,
        scl=stack.scl if stack.scl is not None else np.zeros((0, 0), dtype=np.uint8),
        valid=stack.valid,
        transform=np.asarray(
            [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
            dtype=np.float64,
        ),
        crs=np.asarray([stack.crs.to_string()]),
        item_id=np.asarray([stack.reference.item_id]),
    )
    return path


def load_stack_cache(path: Path, reference: SceneReference) -> BandStack:
    """Load a stack cached by :func:`save_stack_cache`."""

    with np.load(Path(path)) as archive:
        image = archive["image"]
        scl_values = archive["scl"]
        scl = scl_values if scl_values.size else None
        valid = archive["valid"].astype(bool)
        transform_values = archive["transform"]
        crs = CRS.from_string(str(archive["crs"][0]))
    transform = Affine(*[float(value) for value in transform_values])
    return BandStack(
        image=image,
        scl=scl,
        valid=valid,
        transform=transform,
        crs=crs,
        reference=reference,
    )


def stack_to_float(image: np.ndarray, scale: float) -> np.ndarray:
    """Convert integer surface reflectance to float``[0, 1]`` reflectance."""

    return image.astype(np.float32) / float(scale)


def band_indices(band_order: Sequence[str]) -> Mapping[str, int]:
    """Map band name to its position in the stacked array."""

    return {band: index for index, band in enumerate(band_order)}
