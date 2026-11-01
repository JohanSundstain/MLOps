"""GDAL / vsicurl tuning for reading remote Cloud Optimized GeoTIFFs.

Sentinel-2 L2A scenes from the Element84 Earth Search STAC and ESA WorldCover are stored as
COGs on public S3 buckets, so they can be read window by window over HTTPS without
downloading the whole scene. The settings below make those windowed reads fast and resilient
(HTTP range requests, multi-range GETs and an on-disk block cache).
"""

from __future__ import annotations

import os
from typing import Mapping

GDAL_ENVIRONMENT: Mapping[str, str] = {
    # Do not try to list the remote directory when opening a dataset.
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    # Merge consecutive byte ranges into a single multi-range HTTP request.
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    # Retry transient network errors instead of failing a whole preprocessing run.
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
    # Small local caches for headers and blocks of the remote files.
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(64 * 1024 * 1024),
    "CPL_VSIL_CURL_CACHE_SIZE": str(128 * 1024 * 1024),
}


def configure_gdal() -> None:
    """Apply :data:`GDAL_ENVIRONMENT` to the process environment.

    GDAL reads these variables when a dataset is opened, so this has to be called before the
    first ``rasterio.open`` of a remote file.
    """

    for key, value in GDAL_ENVIRONMENT.items():
        os.environ.setdefault(key, value)
