"""End-to-end inference: two georeferenced Sentinel-2 dates -> water areas and their change."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from omegaconf import DictConfig

from lake_segmentation.config import resolve_path, resolve_paths
from lake_segmentation.geo import area_change_report, mask_area_m2
from lake_segmentation.inference.predictor import build_segmenter
from lake_segmentation.inference.raster import (
    RasterData,
    assert_comparable,
    crop_to_roi,
    read_raster,
    write_mask,
)
from lake_segmentation.inference.report import save_inference_figure
from lake_segmentation.logging_utils import get_logger, setup_logging
from lake_segmentation.utils import get_git_commit

LOGGER = get_logger("inference.run")


def _describe_raster(raster: RasterData, date: str | None = None) -> dict[str, Any]:
    return {
        "path": str(raster.path),
        "date": date,
        "height": int(raster.shape[0]),
        "width": int(raster.shape[1]),
        "crs": raster.crs.to_string(),
        "pixel_size_m": [abs(float(raster.transform.a)), abs(float(raster.transform.e))],
        "bounds_wgs84": [round(value, 6) for value in raster.bounds_wgs84],
    }


def _process(
    segmenter: Any,
    raster: RasterData,
    threshold: float,
    output_dir: Path,
    stem: str,
    save_masks: bool,
    save_probabilities: bool,
) -> dict[str, Any]:
    probabilities = segmenter.predict(raster.image, progress_description=f"segmentation {stem}")
    mask = (probabilities > threshold).astype(np.uint8)
    area_m2 = mask_area_m2(mask, raster.transform, raster.crs)
    written: dict[str, str] = {}
    if save_masks:
        written["mask"] = str(
            write_mask(output_dir / f"{stem}_mask.tif", mask, raster.transform, raster.crs)
        )
    if save_probabilities:
        written["probabilities"] = str(
            write_mask(
                output_dir / f"{stem}_probabilities.tif",
                probabilities.astype(np.float32),
                raster.transform,
                raster.crs,
                dtype="float32",
                nodata=None,
            )
        )
    water_pixels = int(mask.sum())
    LOGGER.info(
        "%s: %d water pixels, %.4f km² (%.1f%% of the raster)",
        stem,
        water_pixels,
        area_m2 / 1e6,
        100.0 * water_pixels / mask.size,
    )
    return {
        "probabilities": probabilities,
        "mask": mask,
        "area_m2": area_m2,
        "water_pixels": water_pixels,
        "water_fraction": float(water_pixels / mask.size),
        "artifacts": written,
    }


def run_inference(
    cfg: DictConfig,
    image_a: str | Path,
    image_b: str | Path | None = None,
    output_dir: str | Path | None = None,
    date_a: str | None = None,
    date_b: str | None = None,
    band_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Segment one or two dates, compute the water area and the area change."""

    setup_logging(force=True)
    paths = resolve_paths(cfg)
    target_dir = Path(output_dir or cfg.infer.get("output_dir") or paths["predictions"])
    output_dir = target_dir if target_dir.is_absolute() else resolve_path(target_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segmenter = build_segmenter(cfg)
    threshold = float(segmenter.threshold)
    band_names = list(band_order or cfg.infer.band_order)
    reflectance_scale = float(cfg.infer.reflectance_scale)

    first = read_raster(image_a, reflectance_scale=reflectance_scale)
    roi = list(cfg.infer.get("roi") or [])
    if len(roi) == 4:
        LOGGER.info("Cropping the rasters to the ROI %s", roi)
        first, _ = crop_to_roi(first, roi)

    second: RasterData | None = None
    if image_b is not None:
        second = read_raster(image_b, reflectance_scale=reflectance_scale)
        if len(roi) == 4:
            second, _ = crop_to_roi(second, roi)
        assert_comparable(first, second)

    first_result = _process(
        segmenter,
        first,
        threshold,
        output_dir,
        stem=Path(image_a).stem,
        save_masks=bool(cfg.infer.save_masks),
        save_probabilities=bool(cfg.infer.save_probabilities),
    )
    panels: list[dict[str, Any]] = [
        {
            "image": first.image,
            "mask": first_result["mask"],
            "area_km2": first_result["area_m2"] / 1e6,
            "label": f"{date_a or 'first date'}\n{Path(image_a).name}",
        }
    ]

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "model": {
            "backend": segmenter.backend,
            "threshold": threshold,
            "band_order": segmenter.band_order,
            "tile_size": segmenter.tile_size,
        },
        "first": {
            **_describe_raster(first, date_a),
            "water_pixels": first_result["water_pixels"],
            "water_fraction": first_result["water_fraction"],
            "area_m2": first_result["area_m2"],
            "area_km2": first_result["area_m2"] / 1e6,
            "artifacts": first_result["artifacts"],
        },
    }

    title_parts = []
    if second is not None:
        second_result = _process(
            segmenter,
            second,
            threshold,
            output_dir,
            stem=Path(image_b).stem,
            save_masks=bool(cfg.infer.save_masks),
            save_probabilities=bool(cfg.infer.save_probabilities),
        )
        change = area_change_report(first_result["area_m2"], second_result["area_m2"])
        report["second"] = {
            **_describe_raster(second, date_b),
            "water_pixels": second_result["water_pixels"],
            "water_fraction": second_result["water_fraction"],
            "area_m2": second_result["area_m2"],
            "area_km2": second_result["area_m2"] / 1e6,
            "artifacts": second_result["artifacts"],
        }
        report["change"] = dict(change)
        panels.append(
            {
                "image": second.image,
                "mask": second_result["mask"],
                "area_km2": second_result["area_m2"] / 1e6,
                "label": f"{date_b or 'second date'}\n{Path(image_b).name}",
            }
        )
        relative = change["delta_percent"]
        relative_text = (
            f"{relative:+.2f}%" if relative is not None else "relative change not defined (S1 = 0)"
        )
        title_parts.append(
            f"water area {change['area_first_km2']:.3f} km² -> "
            f"{change['area_second_km2']:.3f} km², delta {change['delta_km2']:+.3f} km² "
            f"({relative_text})"
        )
        LOGGER.info(
            "Area change: %.3f -> %.3f km² (%s)",
            change["area_first_km2"],
            change["area_second_km2"],
            relative_text,
        )
    else:
        title_parts.append(f"water area {report['first']['area_km2']:.3f} km²")

    if bool(cfg.infer.save_figure):
        figure_path = save_inference_figure(
            panels=panels,
            output_path=output_dir / "water_report.png",
            band_order=segmenter.band_order if len(segmenter.band_order) == 4 else band_names,
            title=title_parts[0],
        )
        report["figure"] = str(figure_path)

    if bool(cfg.infer.save_report):
        report_path = output_dir / "water_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["report_path"] = str(report_path)
        LOGGER.info("Report written to %s", report_path)

    return report


def summarize_reference_metrics(
    report: Mapping[str, Any], reference_area_m2: float | None
) -> dict[str, float | None]:
    """Relative area error of a prediction against a known reference area."""

    if reference_area_m2 is None or reference_area_m2 <= 0:
        return {"area_error_percent": None}
    predicted = float(report.get("first", {}).get("area_m2", 0.0))
    return {"area_error_percent": abs(predicted - reference_area_m2) / reference_area_m2 * 100.0}
