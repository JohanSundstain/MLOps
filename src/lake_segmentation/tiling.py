"""Splitting a raster into tiles and stitching the predicted probabilities back.

The inference pipeline never sees the whole image at once: a GeoTIFF may be much larger
than the model input. Tiles are extracted with padding on the borders and the predicted
probabilities are averaged in the overlapping regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

PADDING_MODES = ("constant", "reflect", "edge", "symmetric")


@dataclass(frozen=True)
class Tile:
    """Position of one tile inside the full raster.

    ``row``/``col`` are the offsets of the *valid* (non padded) part of the tile,
    ``height``/``width`` are the sizes of that valid part.
    """

    index: int
    row: int
    col: int
    height: int
    width: int

    def valid_slices(self) -> tuple[slice, slice]:
        return slice(self.row, self.row + self.height), slice(self.col, self.col + self.width)


def iter_tiles(
    height: int, width: int, tile_size: int, stride: int | None = None
) -> Iterator[Tile]:
    """Yield tiles covering a ``height`` x ``width`` raster."""

    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    step = tile_size if stride is None else stride
    if step <= 0:
        raise ValueError(f"stride must be positive, got {step}")

    row_offsets = list(range(0, max(height - tile_size, 0) + 1, step))
    col_offsets = list(range(0, max(width - tile_size, 0) + 1, step))
    if not row_offsets or row_offsets[-1] + tile_size < height:
        row_offsets.append(max(height - tile_size, 0))
    if not col_offsets or col_offsets[-1] + tile_size < width:
        col_offsets.append(max(width - tile_size, 0))

    index = 0
    for row_offset in row_offsets:
        for col_offset in col_offsets:
            valid_height = min(tile_size, height - row_offset)
            valid_width = min(tile_size, width - col_offset)
            yield Tile(
                index=index,
                row=row_offset,
                col=col_offset,
                height=valid_height,
                width=valid_width,
            )
            index += 1


def count_tiles(height: int, width: int, tile_size: int, stride: int | None = None) -> int:
    """Number of tiles that :func:`iter_tiles` would produce."""

    return sum(1 for _ in iter_tiles(height, width, tile_size, stride))


def extract_tile(
    array: np.ndarray,
    tile: Tile,
    tile_size: int,
    mode: str = "reflect",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Cut ``tile`` out of ``array`` (channels first) and pad it to ``tile_size``.

    Padding is applied to the bottom and right borders only, the valid content always starts
    at ``[0, 0]``, which makes stitching trivial.
    """

    if mode not in PADDING_MODES:
        raise ValueError(f"Unknown padding mode {mode!r}, expected one of {PADDING_MODES}")
    if array.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got {array.ndim}")

    rows, cols = tile.valid_slices()
    tile_values = array[..., rows, cols]
    pad_width = [(0, 0)] * (tile_values.ndim - 2)
    pad_width.append((0, tile_size - tile.height))
    pad_width.append((0, tile_size - tile.width))
    if all(before == 0 and after == 0 for before, after in pad_width[-2:]):
        return np.ascontiguousarray(tile_values)

    padded = np.pad(tile_values, pad_width, mode=mode, constant_values=fill_value)
    return np.ascontiguousarray(padded)


def stitch_probabilities(
    probabilities: np.ndarray, tiles: list[Tile], height: int, width: int
) -> np.ndarray:
    """Average per-tile probabilities into a full size map."""

    accumulator = np.zeros((height, width), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)
    per_tile = list(probabilities) if len(probabilities) == len(tiles) else []
    if not per_tile:
        raise ValueError("The number of probability maps does not match the number of tiles")

    for tile, tile_probability in zip(tiles, per_tile):
        rows, cols = tile.valid_slices()
        cropped = np.asarray(tile_probability)[: tile.height, : tile.width]
        if cropped.shape != (tile.height, tile.width):
            raise ValueError(
                f"Tile {tile.index} has shape {cropped.shape}, expected "
                f"{(tile.height, tile.width)}"
            )
        accumulator[rows, cols] += cropped
        weights[rows, cols] += 1.0

    if np.any(weights == 0):
        raise ValueError("Some pixels were not covered by any tile")
    return accumulator / weights
