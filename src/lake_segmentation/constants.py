"""Project wide constants.

Everything that is a genuine constant of the physical problem lives here. Values that are
expected to be tuned (thresholds, hyper-parameters, AOIs, ...) belong to the Hydra configs.
"""

from __future__ import annotations

from typing import Final

# Sentinel-2 L2A surface reflectance is stored as integers scaled by 10_000.
REFLECTANCE_SCALE: Final[float] = 10_000.0

# Model input bands (10 m resolution): B2, B3, B4, B8.
BAND_NAMES: Final[tuple[str, ...]] = ("blue", "green", "red", "nir")
NUM_BANDS: Final[int] = len(BAND_NAMES)

# ESA WorldCover class of permanent water bodies.
WATER_CLASS: Final[int] = 80

# Sentinel-2 scene classification layer (SCL) classes that mark a pixel as unusable.
# 0 = no data, 1 = saturated/defective, 3 = cloud shadow, 8/9 = cloud, 10 = cirrus, 11 = snow.
SCL_INVALID_CLASSES: Final[tuple[int, ...]] = (0, 1, 3, 8, 9, 10, 11)

# Masks and probabilities are stored as uint8 in [0, 255] when written to GeoTIFF.
UINT8_MAX: Final[int] = 255

# Name of the manifest that describes the processed dataset.
MANIFEST_FILENAME: Final[str] = "manifest.csv"
NORM_STATS_FILENAME: Final[str] = "norm_stats.json"
DATASET_SUMMARY_FILENAME: Final[str] = "dataset_summary.json"
PATCH_DIRNAME: Final[str] = "patches"

# Name of the JSON file that describes a trained model next to its weights.
BUNDLE_FILENAME: Final[str] = "bundle.json"

# Directory (inside the repository) with the Hydra configuration tree.
CONFIGS_DIRNAME: Final[str] = "configs"

SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "val", "test")
