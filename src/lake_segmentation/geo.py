"""Geometry helpers: pixel area, mask area and area change.

The pixel area of a raster depends on its coordinate reference system:

* in a projected CRS (e.g. UTM) every pixel has the same area in square metres;
* in a geographic CRS (e.g. EPSG:4326) a pixel covers different ground areas depending on
  its latitude, so the area is computed with the WGS84 ellipsoid (``pyproj.Geod``).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from pyproj import Geod
from rasterio.crs import CRS
from rasterio.transform import Affine

WGS84_GEOD = Geod(ellps="WGS84")


def crs_from_value(value: CRS | str | int | None) -> CRS:
    """Coerce a CRS description coming from config / a raster into ``rasterio.crs.CRS``."""

    if isinstance(value, CRS):
        return value
    if value is None:
        raise ValueError("Raster has no coordinate reference system")
    if isinstance(value, int):
        return CRS.from_epsg(value)
    text = str(value)
    if text.upper().startswith("EPSG:"):
        return CRS.from_string(text)
    if text.isdigit():
        return CRS.from_epsg(int(text))
    return CRS.from_user_input(text)


def pixel_area_m2(transform: Affine, crs: CRS | str | int) -> float:
    """Ground area of one pixel in square metres (projected CRS only)."""

    crs_object = crs_from_value(crs)
    if not crs_object.is_projected:
        raise ValueError(
            f"Pixel area in metres requires a projected CRS, got {crs_object.to_string()}. "
            "Use mask_area_m2() for geographic rasters."
        )
    return abs(transform.a * transform.e - transform.b * transform.d)


def row_pixel_areas_m2(
    transform: Affine, crs: CRS | str | int, rows: np.ndarray | int
) -> np.ndarray:
    """Per-row pixel areas in square metres for a north-up raster."""

    crs_object = crs_from_value(crs)
    row_indices = np.atleast_1d(np.asarray(rows, dtype=np.int64))
    if not crs_object.is_geographic:
        return np.full(row_indices.shape, pixel_area_m2(transform, crs_object), dtype=np.float64)

    if not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0):
        raise ValueError(
            "Rotated geographic rasters are not supported, reproject the raster to a "
            "projected CRS first."
        )

    # Pixel centres of the requested rows, then a one-pixel polygon around each centre.
    longitudes = transform.c + (np.zeros(row_indices.shape) + 0.5) * transform.a
    latitudes = transform.f + (row_indices + 0.5) * transform.e
    half_width = abs(transform.a) / 2.0
    half_height = abs(transform.e) / 2.0

    areas = np.empty(row_indices.shape, dtype=np.float64)
    for position, (longitude, latitude) in enumerate(zip(longitudes, latitudes)):
        # Counter-clockwise ring of the pixel: (lon-, lat-), (lon+, lat-), (lon+, lat+), (lon-, lat+).
        corners_lon = [
            longitude - half_width,
            longitude + half_width,
            longitude + half_width,
            longitude - half_width,
        ]
        corners_lat = [
            latitude - half_height,
            latitude - half_height,
            latitude + half_height,
            latitude + half_height,
        ]
        area, _ = WGS84_GEOD.polygon_area_perimeter(corners_lon, corners_lat)
        areas[position] = abs(area)
    return areas


def mask_area_m2(mask: np.ndarray, transform: Affine, crs: CRS | str | int) -> float:
    """Ground area (m²) covered by the ``True`` pixels of ``mask``."""

    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {binary.shape}")
    if not binary.any():
        return 0.0
    crs_object = crs_from_value(crs)
    if not crs_object.is_geographic:
        return float(binary.sum()) * pixel_area_m2(transform, crs_object)

    rows = np.flatnonzero(binary.any(axis=1))
    row_pixels = binary.sum(axis=1)[rows].astype(np.float64)
    return float(np.sum(row_pixels * row_pixel_areas_m2(transform, crs_object, rows)))


def mask_area_km2(mask: np.ndarray, transform: Affine, crs: CRS | str | int) -> float:
    """Same as :func:`mask_area_m2` but in square kilometres."""

    return mask_area_m2(mask, transform, crs) / 1_000_000.0


def relative_area_change(area_first: float, area_second: float) -> float | None:
    """Relative change of the water area in percent.

    ``None`` means "not defined" (no water during the first observation), which is exactly
    what the project description requires instead of raising or returning infinity.
    """

    if area_first <= 0:
        return None
    return (area_second - area_first) / area_first * 100.0


def area_change_report(area_first_m2: float, area_second_m2: float) -> Mapping[str, float | None]:
    """Absolute and relative area change with the conventions of the project."""

    delta = area_second_m2 - area_first_m2
    return {
        "area_first_m2": area_first_m2,
        "area_second_m2": area_second_m2,
        "area_first_km2": area_first_m2 / 1_000_000.0,
        "area_second_km2": area_second_m2 / 1_000_000.0,
        "delta_m2": delta,
        "delta_km2": delta / 1_000_000.0,
        "delta_percent": relative_area_change(area_first_m2, area_second_m2),
    }
