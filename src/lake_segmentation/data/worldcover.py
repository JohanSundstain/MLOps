"""Reference water masks from ESA WorldCover 2021 (v200).

WorldCover is distributed as 3° x 3° Cloud Optimized GeoTIFFs in EPSG:4326 with a resolution
of 1/12000 degree (about 10 m). The project uses the class of permanent water bodies
(class ``80``) as the reference mask and reprojects the window of interest onto the grid of
the Sentinel-2 scene, so that image and mask share exactly the same pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import rasterio
from omegaconf import DictConfig
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from lake_segmentation.data.gdal_env import configure_gdal
from lake_segmentation.logging_utils import get_logger

LOGGER = get_logger("data.worldcover")

WGS84 = "EPSG:4326"
TILE_EXTENT_DEGREES = 3


@dataclass(frozen=True)
class WorldCoverTile:
    """Identification of one WorldCover tile."""

    name: str
    url: str
    bounds: tuple[float, float, float, float]

    def contains(self, bbox: Sequence[float]) -> bool:
        tolerance = 1e-9
        return (
            self.bounds[0] - tolerance <= bbox[0]
            and self.bounds[1] - tolerance <= bbox[1]
            and self.bounds[2] + tolerance >= bbox[2]
            and self.bounds[3] + tolerance >= bbox[3]
        )


def tile_name(latitude: float, longitude: float) -> str:
    """WorldCover tile name (lower-left corner of the 3° tile)."""

    tile_latitude = math.floor(latitude / TILE_EXTENT_DEGREES) * TILE_EXTENT_DEGREES
    tile_longitude = math.floor(longitude / TILE_EXTENT_DEGREES) * TILE_EXTENT_DEGREES
    north_south = "N" if tile_latitude >= 0 else "S"
    east_west = "E" if tile_longitude >= 0 else "W"
    return f"{north_south}{abs(tile_latitude):02d}" f"{east_west}{abs(tile_longitude):03d}"


def resolve_tiles(bbox: Sequence[float], cfg: DictConfig) -> list[WorldCoverTile]:
    """All WorldCover tiles that intersect ``bbox`` (EPSG:4326).

    WorldCover is delivered as 3° x 3° tiles, so an AOI close to a multiple of 3° crosses a
    tile boundary and its mask has to be mosaicked from several files.
    """

    min_lon, min_lat, max_lon, max_lat = bbox
    first_latitude = math.floor(min_lat / TILE_EXTENT_DEGREES) * TILE_EXTENT_DEGREES
    last_latitude = math.floor(max_lat / TILE_EXTENT_DEGREES) * TILE_EXTENT_DEGREES
    first_longitude = math.floor(min_lon / TILE_EXTENT_DEGREES) * TILE_EXTENT_DEGREES
    last_longitude = math.floor(max_lon / TILE_EXTENT_DEGREES) * TILE_EXTENT_DEGREES

    tiles: list[WorldCoverTile] = []
    for tile_latitude in range(int(first_latitude), int(last_latitude) + 1, TILE_EXTENT_DEGREES):
        for tile_longitude in range(
            int(first_longitude), int(last_longitude) + 1, TILE_EXTENT_DEGREES
        ):
            name = tile_name(float(tile_latitude), float(tile_longitude))
            filename = str(cfg.data.worldcover.filename_template).format(tile=name)
            url = f"{str(cfg.data.worldcover.base_url).rstrip('/')}/{filename}"
            tiles.append(
                WorldCoverTile(
                    name=name,
                    url=url,
                    bounds=(
                        float(tile_longitude),
                        float(tile_latitude),
                        float(tile_longitude + TILE_EXTENT_DEGREES),
                        float(tile_latitude + TILE_EXTENT_DEGREES),
                    ),
                )
            )
    return tiles


def read_water_mask(
    transform: Affine,
    crs: CRS,
    shape: tuple[int, int],
    bbox_wgs84: Sequence[float],
    cfg: DictConfig,
) -> np.ndarray:
    """Reproject WorldCover onto ``(shape, transform, crs)`` and return the water mask.

    Several 3° tiles are mosaicked when the AOI crosses a tile boundary.
    """

    configure_gdal()
    water_class = int(cfg.data.worldcover.water_class)
    height, width = shape
    destination_bounds = array_bounds(height, width, transform)
    destination = np.zeros(shape, dtype=np.uint8)
    coverage = np.zeros(shape, dtype=np.uint16)

    for tile in resolve_tiles(bbox_wgs84, cfg):
        with rasterio.open(tile.url) as source:
            source_bounds = transform_bounds(crs, source.crs, *destination_bounds, densify_pts=21)
            window = from_bounds(
                source_bounds[0],
                source_bounds[1],
                source_bounds[2],
                source_bounds[3],
                transform=source.transform,
            )
            window = (
                window.round_offsets()
                .round_lengths()
                .intersection(Window(0, 0, source.width, source.height))
            )
            if window.width <= 0 or window.height <= 0:
                continue
            classes = source.read(1, window=window)
            source_transform = source.window_transform(window)
            source_crs = source.crs

        reproject(
            source=classes,
            destination=destination,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
            src_nodata=None,
            dst_nodata=0,
            init_dest_nodata=False,
        )
        tile_bounds_in_crs = transform_bounds(source_crs, crs, *tile.bounds, densify_pts=21)
        tile_window = from_bounds(*tile_bounds_in_crs, transform=transform)
        tile_window = (
            tile_window.round_offsets().round_lengths().intersection(Window(0, 0, width, height))
        )
        if tile_window.width > 0 and tile_window.height > 0:
            coverage[
                int(tile_window.row_off) : int(tile_window.row_off + tile_window.height),
                int(tile_window.col_off) : int(tile_window.col_off + tile_window.width),
            ] += 1
        LOGGER.debug("WorldCover tile %s reprojected onto the scene grid", tile.name)

    uncovered = int(np.count_nonzero(coverage == 0))
    if uncovered:
        LOGGER.warning(
            "%d destination pixels are not covered by any WorldCover tile and are treated as "
            "land",
            uncovered,
        )
    water = destination == water_class
    LOGGER.debug(
        "WorldCover mask built from %d tile(s): %.1f%% water",
        len(resolve_tiles(bbox_wgs84, cfg)),
        100.0 * float(water.mean()),
    )
    return water


def class_histogram(mask_classes: np.ndarray) -> dict[int, int]:
    """Count pixels per WorldCover class (useful for logs and sanity checks)."""

    values, counts = np.unique(mask_classes, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}
