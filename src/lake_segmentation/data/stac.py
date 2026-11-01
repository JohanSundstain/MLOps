"""Discovery of Sentinel-2 L2A scenes through a STAC API.

Every observation used by the project is described by a STAC item. The item is resolved
once, stored in the scenes manifest (for reproducibility) and afterwards only the band assets
are read window by window.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from omegaconf import DictConfig
from pystac_client import Client

from lake_segmentation.logging_utils import get_logger

LOGGER = get_logger("data.stac")


@dataclass(frozen=True)
class AreaOfInterest:
    """A lake area with the dates it should be observed at."""

    name: str
    bbox: tuple[float, float, float, float]
    dates: tuple[str, ...]
    group: str | None = None

    @property
    def split_group(self) -> str:
        """Key used for the group-wise train/val/test split (one lake = one group)."""

        return self.group or self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bbox": list(self.bbox),
            "dates": list(self.dates),
            "group": self.split_group,
        }


@dataclass(frozen=True)
class SceneReference:
    """All the information about one selected STAC item that the project needs."""

    item_id: str
    collection: str
    datetime: str
    cloud_cover: float
    bbox: tuple[float, float, float, float]
    epsg: int | None
    assets: Mapping[str, str]
    aoi_name: str
    target_date: str
    properties: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "collection": self.collection,
            "datetime": self.datetime,
            "cloud_cover": self.cloud_cover,
            "bbox": list(self.bbox),
            "epsg": self.epsg,
            "assets": dict(self.assets),
            "aoi_name": self.aoi_name,
            "target_date": self.target_date,
        }


def aois_from_config(cfg: DictConfig) -> list[AreaOfInterest]:
    """Build :class:`AreaOfInterest` objects from ``cfg.data.aois``."""

    aois: list[AreaOfInterest] = []
    seen_names: set[str] = set()
    for entry in cfg.data.aois:
        name = str(entry["name"])
        if name in seen_names:
            raise ValueError(f"Duplicate AOI name: {name}")
        seen_names.add(name)
        bbox = tuple(float(value) for value in entry["bbox"])
        if len(bbox) != 4:
            raise ValueError(f"AOI {name} must have a 4-element bbox, got {bbox}")
        min_lon, min_lat, max_lon, max_lat = bbox
        if not (min_lon < max_lon and min_lat < max_lat):
            raise ValueError(f"AOI {name} has an invalid bbox: {bbox}")
        dates = tuple(str(value) for value in entry["dates"])
        for date in dates:
            dt.date.fromisoformat(date)
        aois.append(
            AreaOfInterest(
                name=name,
                bbox=(min_lon, min_lat, max_lon, max_lat),
                dates=dates,
                group=str(entry["group"]) if entry.get("group") else None,
            )
        )
    if not aois:
        raise ValueError("No AOIs are configured in data.aois")
    return aois


def _open_client(cfg: DictConfig) -> Client:
    timeout = float(cfg.data.stac.get("request_timeout", 60))
    try:
        return Client.open(str(cfg.data.stac.url), timeout=timeout)
    except TypeError:  # older pystac-client releases have no timeout argument
        return Client.open(str(cfg.data.stac.url))


def _search_with_retries(
    client: Client, bbox: Sequence[float], datetime_range: str, cfg: DictConfig
) -> list[Any]:
    retries = int(cfg.data.stac.get("retries", 3))
    max_items = int(cfg.data.stac.get("max_candidates", 50))
    cloud_limit = float(cfg.data.stac.get("max_cloud_cover", 20))
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            search = client.search(
                collections=[str(cfg.data.stac.collection)],
                bbox=list(bbox),
                datetime=datetime_range,
                query={"eo:cloud_cover": {"lt": cloud_limit}},
                max_items=max_items,
                limit=max_items,
            )
            return list(search.items())
        except Exception as error:  # noqa: BLE001 - network errors are retried
            last_error = error
            LOGGER.warning("STAC search failed (attempt %d/%d): %s", attempt, retries, error)
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"STAC search failed after {retries} attempts") from last_error


def _bbox_covers(outer: Sequence[float], inner: Sequence[float], tolerance: float = 1e-6) -> bool:
    return (
        outer[0] <= inner[0] + tolerance
        and outer[1] <= inner[1] + tolerance
        and outer[2] >= inner[2] - tolerance
        and outer[3] >= inner[3] - tolerance
    )


def _date_distance(item_datetime: dt.datetime | str, target_date: str) -> int:
    if isinstance(item_datetime, dt.datetime):
        observed = item_datetime.date()
    else:
        observed = dt.datetime.fromisoformat(item_datetime.replace("Z", "+00:00")).date()
    target = dt.date.fromisoformat(target_date)
    return abs((observed - target).days)


def find_scenes(aoi: AreaOfInterest, target_date: str, cfg: DictConfig) -> list[Any]:
    """Search STAC items for one AOI and one target date."""

    client = _open_client(cfg)
    target = dt.date.fromisoformat(target_date)
    window_days = int(cfg.data.stac.get("date_window_days", 45))
    start = target - dt.timedelta(days=window_days)
    stop = target + dt.timedelta(days=window_days)
    datetime_range = f"{start:%Y-%m-%d}T00:00:00Z/{stop:%Y-%m-%d}T23:59:59Z"
    items = _search_with_retries(client, aoi.bbox, datetime_range, cfg)
    covering = [item for item in items if _bbox_covers(item.bbox, aoi.bbox)]
    # Prefer clear scenes close to the target date.
    covering.sort(
        key=lambda item: (
            item.properties.get("eo:cloud_cover", 100.0),
            _date_distance(item.datetime, target_date),
        )
    )
    LOGGER.debug(
        "AOI %s %s: %d candidates, %d covering the AOI",
        aoi.name,
        target_date,
        len(items),
        len(covering),
    )
    return covering


def select_scene(aoi: AreaOfInterest, target_date: str, cfg: DictConfig) -> SceneReference:
    """Select the best scene for one AOI and one date.

    The best scene is the one with the lowest cloud cover that fully covers the AOI and is
    closest to the requested date. A clear error is raised when nothing is available so that
    the AOI list can be fixed instead of silently producing an empty dataset.
    """

    candidates = find_scenes(aoi, target_date, cfg)
    if not candidates:
        raise LookupError(
            f"No Sentinel-2 scene covers AOI {aoi.name!r} within "
            f"{cfg.data.stac.date_window_days} days of {target_date}. "
            "Move the AOI inside a single Sentinel-2 tile or change the date."
        )
    item = candidates[0]
    band_assets = dict(cfg.data.stac.bands)
    assets: dict[str, str] = {}
    for band_name, asset_name in band_assets.items():
        asset = item.assets.get(asset_name)
        if asset is None:
            raise LookupError(
                f"STAC item {item.id} has no asset {asset_name!r} (needed for band "
                f"{band_name!r}); available assets: {sorted(item.assets)}"
            )
        assets[band_name] = asset.href
    scl_name = str(cfg.data.stac.get("scl_asset", "scl"))
    if scl_name in item.assets:
        assets["scl"] = item.assets[scl_name].href

    epsg = item.properties.get("proj:epsg")
    return SceneReference(
        item_id=item.id,
        collection=str(cfg.data.stac.collection),
        datetime=item.datetime.isoformat() if item.datetime else "",
        cloud_cover=float(item.properties.get("eo:cloud_cover", float("nan"))),
        bbox=tuple(float(value) for value in item.bbox),
        epsg=int(epsg) if epsg is not None else None,
        assets=assets,
        aoi_name=aoi.name,
        target_date=target_date,
        properties={
            "platform": item.properties.get("platform"),
            "processing:level": item.properties.get("processing:level"),
            "sat:orbit_state": item.properties.get("sat:orbit_state"),
        },
    )


def resolve_scenes(aois: Iterable[AreaOfInterest], cfg: DictConfig) -> list[SceneReference]:
    """Resolve one scene per AOI and date, skipping failures with a warning."""

    references: list[SceneReference] = []
    for aoi in aois:
        for target_date in aoi.dates:
            try:
                reference = select_scene(aoi, target_date, cfg)
            except (LookupError, RuntimeError) as error:
                if cfg.data.stac.get("strict", False):
                    raise
                LOGGER.warning("Skipping %s %s: %s", aoi.name, target_date, error)
                continue
            LOGGER.info(
                "AOI %-14s %s -> %s (cloud %.1f%%)",
                aoi.name,
                target_date,
                reference.item_id,
                reference.cloud_cover,
            )
            references.append(reference)
    if not references:
        raise RuntimeError("No scenes were resolved, check the AOI list and its dates")
    return references
