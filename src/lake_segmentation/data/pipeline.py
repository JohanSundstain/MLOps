"""The preprocessing pipeline: STAC -> scenes -> patches -> manifest.

Two commands are built on top of this module:

``download``
    Resolve the best Sentinel-2 scene for every AOI and date, verify that all band assets are
    readable and store the scenes manifest (``data/raw/scenes_manifest.json``). Nothing large
    is downloaded: the scenes are streamed window by window later.

``preprocess``
    Read every AOI, build the water mask from WorldCover, cut the selected patches, save them
    as compressed ``npz`` files, assign the group-wise split, compute the normalisation
    statistics of the training split and write the manifest and the summary.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import rasterio
from omegaconf import DictConfig
from rasterio.transform import Affine
from rasterio.windows import Window
from tqdm import tqdm

from lake_segmentation.config import resolve_path, resolve_paths, to_container
from lake_segmentation.constants import (
    BAND_NAMES,
    DATASET_SUMMARY_FILENAME,
    MANIFEST_FILENAME,
    NORM_STATS_FILENAME,
    PATCH_DIRNAME,
    SCL_INVALID_CLASSES,
    SPLIT_NAMES,
)
from lake_segmentation.data.gdal_env import configure_gdal
from lake_segmentation.data.normalization import compute_stats, save_stats
from lake_segmentation.data.patches import (
    cut_patch,
    iter_candidates,
    load_patch,
    patch_identifier,
    save_patch,
    select_candidates,
)
from lake_segmentation.data.scenes import load_stack_cache, read_aoi_stack, save_stack_cache
from lake_segmentation.data.splits import assign_splits
from lake_segmentation.data.stac import (
    AreaOfInterest,
    SceneReference,
    aois_from_config,
    resolve_scenes,
)
from lake_segmentation.data.worldcover import read_water_mask
from lake_segmentation.logging_utils import get_logger
from lake_segmentation.plots import save_dataset_preview
from lake_segmentation.utils import get_git_commit, seed_everything, stable_seed

LOGGER = get_logger("data.pipeline")

SCENES_MANIFEST_FILENAME = "scenes_manifest.json"


def _aoi_lookup(aois: Sequence[AreaOfInterest]) -> dict[str, AreaOfInterest]:
    return {aoi.name: aoi for aoi in aois}


def scenes_manifest_path(raw_dir: Path) -> Path:
    return Path(raw_dir) / SCENES_MANIFEST_FILENAME


def save_scenes_manifest(references: Sequence[SceneReference], raw_dir: Path) -> Path:
    path = scenes_manifest_path(raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "scenes": [reference.to_dict() for reference in references],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Scenes manifest written to %s (%d scenes)", path, len(references))
    return path


def load_scenes_manifest(raw_dir: Path) -> list[SceneReference] | None:
    path = scenes_manifest_path(raw_dir)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    references = []
    for entry in payload.get("scenes", []):
        references.append(
            SceneReference(
                item_id=entry["item_id"],
                collection=entry["collection"],
                datetime=entry["datetime"],
                cloud_cover=float(entry["cloud_cover"]),
                bbox=tuple(entry["bbox"]),
                epsg=entry.get("epsg"),
                assets=entry["assets"],
                aoi_name=entry["aoi_name"],
                target_date=entry["target_date"],
                properties=entry.get("properties", {}),
            )
        )
    return references or None


def verify_scene_access(reference: SceneReference, window_size: int = 16) -> None:
    """Read a small window of every asset to fail fast on expired links or credentials."""

    for asset_name, href in reference.assets.items():
        with rasterio.open(href) as source:
            window = Window(
                max(source.width // 2, 0),
                max(source.height // 2, 0),
                min(window_size, source.width),
                min(window_size, source.height),
            )
            source.read(1, window=window)


def run_download(cfg: DictConfig) -> dict[str, Any]:
    """Resolve scenes for all AOIs, verify access and store the scenes manifest."""

    configure_gdal()
    seed_everything(int(cfg.seed))
    paths = resolve_paths(cfg)
    aois = aois_from_config(cfg)
    references = resolve_scenes(aois, cfg)

    verify = bool(cfg.data.download.get("verify_access", True))
    if verify:
        for reference in tqdm(references, desc="verify scenes"):
            verify_scene_access(reference)

    save_scenes_manifest(references, paths["data_raw"])
    summary = {
        "num_scenes": len(references),
        "num_aois": len({reference.aoi_name for reference in references}),
        "items": [
            {
                "aoi": reference.aoi_name,
                "target_date": reference.target_date,
                "item_id": reference.item_id,
                "datetime": reference.datetime,
                "cloud_cover": reference.cloud_cover,
                "epsg": reference.epsg,
            }
            for reference in references
        ],
    }
    LOGGER.info("Download step finished: %d scenes resolved", len(references))
    return summary


def _load_or_read_stack(
    reference: SceneReference,
    aoi: AreaOfInterest,
    cfg: DictConfig,
    raw_dir: Path,
):
    cache_enabled = bool(cfg.data.download.get("cache_stacks", False))
    cache_path = Path(raw_dir) / "stacks" / f"{reference.aoi_name}__{reference.target_date}.npz"
    if cache_enabled and cache_path.is_file():
        return load_stack_cache(cache_path, reference)

    stack = read_aoi_stack(
        reference=reference,
        bbox_wgs84=aoi.bbox,
        band_order=list(cfg.data.stac.band_order),
        patch_size=int(cfg.data.patch.size),
        invalid_scl_classes=list(cfg.data.scl_invalid_classes or SCL_INVALID_CLASSES),
        max_nodata_fraction=float(cfg.data.patch.max_invalid_fraction),
    )
    if cache_enabled:
        save_stack_cache(cache_path, stack)
    return stack


def build_patch_records(
    cfg: DictConfig,
    paths: dict[str, Path],
    references: Sequence[SceneReference],
) -> pd.DataFrame:
    """Read all scenes, cut the patches and return the manifest rows."""

    aois = _aoi_lookup(aois_from_config(cfg))
    patch_size = int(cfg.data.patch.size)
    stride = int(cfg.data.patch.stride)
    patch_dir = paths["data_processed"] / PATCH_DIRNAME
    patch_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    per_scene_stats: list[dict[str, Any]] = []

    for reference in tqdm(references, desc="build patches"):
        aoi = aois[reference.aoi_name]
        try:
            stack = _load_or_read_stack(reference, aoi, cfg, paths["data_raw"])
            water = read_water_mask(
                transform=stack.transform,
                crs=stack.crs,
                shape=stack.shape,
                bbox_wgs84=aoi.bbox,
                cfg=cfg,
            )
            candidates = list(iter_candidates(water, stack.valid, patch_size, stride))
            generator = np.random.default_rng(
                stable_seed(int(cfg.seed), aoi.name, reference.target_date)
            )
            selected = select_candidates(
                candidates=candidates,
                min_water_fraction=float(cfg.data.patch.min_water_fraction),
                max_water_fraction=float(cfg.data.patch.max_water_fraction),
                min_valid_fraction=float(cfg.data.patch.min_valid_fraction),
                negative_ratio=float(cfg.data.patch.negative_ratio),
                rng=generator,
                max_patches=int(cfg.data.patch.max_patches_per_scene),
            )
        except Exception as error:  # noqa: BLE001 - one bad scene must not stop the run
            LOGGER.warning("Skipping scene %s: %s", reference.item_id, error)
            per_scene_stats.append(
                {"aoi": aoi.name, "date": reference.target_date, "status": f"failed: {error}"}
            )
            continue

        image = stack.image
        for candidate in selected:
            patch_id = patch_identifier(aoi.name, reference.target_date, candidate.tile)
            image_patch, mask_patch = cut_patch(image, water, candidate.tile)
            save_patch(patch_dir / f"{patch_id}.npz", image_patch, mask_patch)
            patch_transform = stack.transform * Affine.translation(
                candidate.tile.col, candidate.tile.row
            )
            records.append(
                {
                    "patch_id": patch_id,
                    "path": f"{PATCH_DIRNAME}/{patch_id}.npz",
                    "aoi": aoi.name,
                    "group": aoi.split_group,
                    "target_date": reference.target_date,
                    "scene_id": reference.item_id,
                    "scene_datetime": reference.datetime,
                    "cloud_cover": reference.cloud_cover,
                    "row": candidate.tile.row,
                    "col": candidate.tile.col,
                    "patch_size": patch_size,
                    "water_fraction": round(candidate.water_fraction, 6),
                    "valid_fraction": round(candidate.valid_fraction, 6),
                    "crs": stack.crs.to_string(),
                    "crs_epsg": stack.crs.to_epsg(),
                    "transform_a": patch_transform.a,
                    "transform_b": patch_transform.b,
                    "transform_c": patch_transform.c,
                    "transform_d": patch_transform.d,
                    "transform_e": patch_transform.e,
                    "transform_f": patch_transform.f,
                }
            )
        per_scene_stats.append(
            {
                "aoi": aoi.name,
                "date": reference.target_date,
                "scene_id": reference.item_id,
                "status": "ok",
                "candidates": len(candidates),
                "selected": len(selected),
                "positives": sum(
                    1
                    for candidate in selected
                    if candidate.water_fraction >= float(cfg.data.patch.min_water_fraction)
                ),
                "water_fraction_aoi": round(float(water.mean()), 4),
            }
        )
        LOGGER.info(
            "AOI %-14s %s: %d/%d patches kept (%d tiles) from %s",
            aoi.name,
            reference.target_date,
            len(selected),
            len(candidates),
            len(candidates),
            reference.item_id,
        )

    if not records:
        raise RuntimeError(
            "No patches were created. Check the AOI list, the cloud filter and the water "
            "fraction thresholds."
        )
    frame = pd.DataFrame.from_records(records)
    frame.attrs["per_scene_stats"] = per_scene_stats
    return frame


def _split_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    """Per-split patch counts and lakes (``train`` / ``val`` / ``test`` only)."""

    statistics: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        subset = frame[frame["split"] == split]
        statistics[split] = {
            "patches": int(len(subset)),
            "lakes": sorted(subset["group"].unique().tolist()) if len(subset) else [],
            "positive_patches": int((subset["water_fraction"] > 0).sum()) if len(subset) else 0,
            "water_fraction_mean": (
                round(float(subset["water_fraction"].mean()), 6) if len(subset) else 0.0
            ),
        }
    return statistics


def run_preprocess(cfg: DictConfig) -> dict[str, Any]:
    """Build the dataset and write ``manifest.csv``, ``norm_stats.json`` and the summary."""

    configure_gdal()
    seed_everything(int(cfg.seed))
    paths = resolve_paths(cfg)
    processed_dir = paths["data_processed"]
    band_order = list(cfg.data.stac.band_order)
    if tuple(band_order) != BAND_NAMES:
        LOGGER.warning(
            "Configured band order %s differs from the default %s", band_order, BAND_NAMES
        )

    references = load_scenes_manifest(paths["data_raw"])
    if references is None:
        LOGGER.info("No scenes manifest found, resolving scenes from STAC")
        references = resolve_scenes(aois_from_config(cfg), cfg)
        save_scenes_manifest(references, paths["data_raw"])

    frame = build_patch_records(cfg, paths, references)
    per_scene_stats = frame.attrs.get("per_scene_stats", [])

    split_fractions = {
        "train": float(cfg.preprocess.split.train),
        "val": float(cfg.preprocess.split.val),
        "test": float(cfg.preprocess.split.test),
    }
    splits = assign_splits(
        groups=frame["group"].unique().tolist(),
        fractions=split_fractions,
        seed=int(cfg.preprocess.split.seed),
        sizes=frame.groupby("group").size().to_dict(),
    )
    frame["split"] = frame["group"].map(splits)
    if frame["split"].isna().any():
        raise RuntimeError("Some patches were not assigned to a split")

    column_order = [
        "patch_id",
        "path",
        "split",
        "aoi",
        "group",
        "target_date",
        "scene_id",
        "scene_datetime",
        "cloud_cover",
        "row",
        "col",
        "patch_size",
        "water_fraction",
        "valid_fraction",
        "crs",
        "crs_epsg",
        "transform_a",
        "transform_b",
        "transform_c",
        "transform_d",
        "transform_e",
        "transform_f",
    ]
    frame = frame[column_order].sort_values(["split", "aoi", "patch_id"]).reset_index(drop=True)
    manifest_path = processed_dir / MANIFEST_FILENAME
    frame.to_csv(manifest_path, index=False)
    LOGGER.info("Manifest written to %s (%d patches)", manifest_path, len(frame))
    remove_stale_patches(processed_dir, frame)

    train_records = frame[frame["split"] == "train"].to_dict("records")
    patch_iterable = (load_patch(processed_dir / record["path"])[0] for record in train_records)
    stats = compute_stats(
        patch_iterable,
        scale=float(cfg.preprocess.reflectance_scale),
        clip_min=float(cfg.preprocess.clip_min),
        clip_max=float(cfg.preprocess.clip_max),
    )
    save_stats(processed_dir / NORM_STATS_FILENAME, stats)

    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "num_patches": int(len(frame)),
        "patch_size": int(cfg.data.patch.size),
        "band_order": band_order,
        "splits": _split_statistics(frame),
        "split_assignment": dict(sorted(splits.items())),
        "patches_per_aoi": frame.groupby("aoi").size().to_dict(),
        "normalization": stats.to_dict(),
        "scenes": per_scene_stats,
        "config": to_container(cfg),
    }
    summary_path = processed_dir / DATASET_SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    # Show shoreline patches: they are the informative case for a visual check.
    train_frame = frame[frame["split"] == "train"].copy()
    train_frame["shoreline_score"] = (train_frame["water_fraction"] - 0.5).abs()
    preview_records = (
        train_frame.sort_values("shoreline_score")
        .head(int(cfg.preprocess.preview_samples))
        .to_dict("records")
    )
    if preview_records:
        save_dataset_preview(
            records=preview_records,
            data_dir=processed_dir,
            output_path=paths["plots"] / "dataset_preview.png",
            band_order=band_order,
        )

    LOGGER.info(
        "Preprocessing finished: %d patches (train %d / val %d / test %d)",
        len(frame),
        summary["splits"]["train"]["patches"],
        summary["splits"]["val"]["patches"],
        summary["splits"]["test"]["patches"],
    )
    return summary


def remove_stale_patches(processed_dir: Path, frame: pd.DataFrame) -> int:
    """Delete ``*.npz`` files that are no longer referenced by the manifest."""

    patch_dir = Path(processed_dir) / PATCH_DIRNAME
    expected = {str(path) for path in frame["path"]}
    removed = 0
    for existing in patch_dir.glob("*.npz"):
        relative = f"{PATCH_DIRNAME}/{existing.name}"
        if relative not in expected:
            existing.unlink()
            removed += 1
    if removed:
        LOGGER.info("Removed %d patch files left over from a previous run", removed)
    return removed


def load_manifest(processed_dir: Path) -> pd.DataFrame:
    """Read the dataset manifest."""

    manifest_path = Path(processed_dir) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found at {manifest_path}. Run preprocessing first."
        )
    return pd.read_csv(manifest_path)


def records_for_split(processed_dir: Path, split: str) -> list[dict[str, Any]]:
    """Manifest rows of one split."""

    frame = load_manifest(processed_dir)
    subset = frame[frame["split"] == split]
    if subset.empty:
        raise ValueError(f"Split {split!r} is empty in {processed_dir}")
    return subset.to_dict("records")


def transform_from_record(record: dict[str, Any]) -> Affine:
    """Rebuild the affine transform of a patch from its manifest row."""

    return Affine(
        float(record["transform_a"]),
        float(record["transform_b"]),
        float(record["transform_c"]),
        float(record["transform_d"]),
        float(record["transform_e"]),
        float(record["transform_f"]),
    )


def resolve_data_dir(cfg: DictConfig, project_root: Path | None = None) -> Path:
    """Absolute path of the processed dataset."""

    return resolve_path(cfg.paths.data_processed, project_root)
