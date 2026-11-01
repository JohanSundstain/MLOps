"""Tests for tiling and stitching."""

from __future__ import annotations

import numpy as np
import pytest

from lake_segmentation.tiling import count_tiles, extract_tile, iter_tiles, stitch_probabilities


def test_tiles_cover_the_whole_raster():
    height, width, tile_size = 300, 700, 256
    tiles = list(iter_tiles(height, width, tile_size))
    coverage = np.zeros((height, width), dtype=int)
    for tile in tiles:
        rows, cols = tile.valid_slices()
        coverage[rows, cols] += 1
    assert coverage.min() == 1
    assert count_tiles(height, width, tile_size) == len(tiles)


def test_small_raster_gives_a_single_padded_tile():
    tiles = list(iter_tiles(100, 120, 256))
    assert len(tiles) == 1
    assert (tiles[0].height, tiles[0].width) == (100, 120)


def test_extract_tile_pads_to_square():
    array = np.arange(100 * 120, dtype=np.float32).reshape(1, 100, 120)
    tile = next(iter(iter_tiles(100, 120, 256)))
    padded = extract_tile(array, tile, 256, mode="constant")
    assert padded.shape == (1, 256, 256)
    np.testing.assert_allclose(padded[0, :100, :120], array[0])


def test_stitching_without_overlap_reconstructs_the_image():
    height, width = 300, 400
    image = np.random.default_rng(0).random((height, width)).astype(np.float32)
    tiles = list(iter_tiles(height, width, 256))
    probabilities = [extract_tile(image, tile, 256)[: tile.height, : tile.width] for tile in tiles]
    restored = stitch_probabilities(probabilities, tiles, height, width)
    np.testing.assert_allclose(restored, image, rtol=0, atol=1e-6)


def test_overlapping_tiles_are_averaged():
    height, width = 300, 300
    image = np.ones((height, width), dtype=np.float32)
    tiles = list(iter_tiles(height, width, 256, stride=128))
    probabilities = [np.full((256, 256), 0.5, dtype=np.float32) for _ in tiles]
    stitched = stitch_probabilities(probabilities, tiles, height, width)
    np.testing.assert_allclose(stitched, 0.5)
    assert len(tiles) > 1
    assert image.mean() == pytest.approx(1.0)
