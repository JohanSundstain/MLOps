"""Tests for dataset related helpers that do not touch the network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lake_segmentation.data.normalization import (
    NormalizationStats,
    apply_normalization,
    compute_stats,
    denormalize,
    load_stats,
    save_stats,
)
from lake_segmentation.data.patches import PatchCandidate, select_candidates
from lake_segmentation.data.worldcover import tile_name
from lake_segmentation.tiling import Tile


def _candidate(row: int, water_fraction: float, valid_fraction: float = 1.0) -> PatchCandidate:
    return PatchCandidate(
        tile=Tile(index=0, row=row, col=0, height=256, width=256),
        water_fraction=water_fraction,
        valid_fraction=valid_fraction,
    )


def test_worldcover_tile_names():
    assert tile_name(54.9, 77.5) == "N54E075"
    assert tile_name(51.6, 87.7) == "N51E087"
    assert tile_name(-3.2, -72.1) == "S06W075"


def test_normalization_round_trip():
    stats = NormalizationStats(mean=(0.1, 0.2, 0.3, 0.4), std=(0.01, 0.02, 0.03, 0.04))
    image = np.random.default_rng(0).random((4, 8, 8)).astype(np.float32)
    normalized = apply_normalization(image, stats)
    np.testing.assert_allclose(denormalize(normalized, stats), image, rtol=1e-5, atol=1e-6)


def test_compute_stats_recovers_mean_and_std():
    generator = np.random.default_rng(0)
    patches = [(generator.random((4, 16, 16)) * 4000).astype(np.uint16) for _ in range(5)]
    stats = compute_stats(patches, scale=10_000.0, clip_min=0.0, clip_max=1.0)
    reference = np.concatenate([patch.reshape(4, -1) for patch in patches], axis=1) / 10_000.0
    np.testing.assert_allclose(stats.mean, reference.mean(axis=1), rtol=1e-4)
    np.testing.assert_allclose(stats.std, reference.std(axis=1), rtol=1e-3)


def test_save_and_load_stats(tmp_path: Path):
    stats = NormalizationStats(mean=(0.1, 0.2), std=(0.01, 0.02))
    path = tmp_path / "norm_stats.json"
    save_stats(path, stats)
    assert load_stats(path) == stats


def test_select_candidates_balances_negatives():
    candidates = [_candidate(row, 0.5) for row in range(10)]
    candidates += [_candidate(row + 100, 0.0) for row in range(10)]
    candidates += [_candidate(row + 200, 0.99) for row in range(3)]
    selected = select_candidates(
        candidates=candidates,
        min_water_fraction=0.02,
        max_water_fraction=0.98,
        min_valid_fraction=0.95,
        negative_ratio=0.25,
        rng=np.random.default_rng(0),
    )
    positives = [item for item in selected if item.water_fraction >= 0.02]
    negatives = [item for item in selected if item.water_fraction < 0.02]
    assert len(positives) == 10
    assert len(negatives) == pytest.approx(round(10 * 0.25 / 0.75), abs=1)


def test_select_candidates_drops_cloudy_patches():
    candidates = [_candidate(0, 0.5, valid_fraction=0.5)]
    selected = select_candidates(
        candidates=candidates,
        min_water_fraction=0.02,
        max_water_fraction=0.98,
        min_valid_fraction=0.95,
        negative_ratio=0.0,
        rng=np.random.default_rng(0),
    )
    assert selected == []


def test_select_candidates_returns_empty_without_positives():
    candidates = [_candidate(0, 0.0)]
    assert (
        select_candidates(
            candidates=candidates,
            min_water_fraction=0.02,
            max_water_fraction=0.98,
            min_valid_fraction=0.95,
            negative_ratio=0.3,
            rng=np.random.default_rng(0),
        )
        == []
    )
