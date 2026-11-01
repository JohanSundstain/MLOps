"""Tests for the area computations (the core of the "area change" part of the project)."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from lake_segmentation.geo import (
    area_change_report,
    mask_area_km2,
    mask_area_m2,
    pixel_area_m2,
    relative_area_change,
)

UTM = CRS.from_epsg(32643)
GEOGRAPHIC = CRS.from_epsg(4326)


def test_pixel_area_projected():
    transform = from_origin(400_000, 6_100_000, 10, 10)
    assert pixel_area_m2(transform, UTM) == pytest.approx(100.0)


def test_pixel_area_rejects_geographic():
    transform = from_origin(77.0, 55.0, 0.0001, 0.0001)
    with pytest.raises(ValueError, match="projected CRS"):
        pixel_area_m2(transform, GEOGRAPHIC)


def test_mask_area_ignores_land():
    transform = from_origin(400_000, 6_100_000, 10, 10)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 1
    assert mask_area_m2(mask, transform, UTM) == pytest.approx(4 * 100.0)
    assert mask_area_km2(mask, transform, UTM) == pytest.approx(4 * 100.0 / 1e6)


def test_mask_area_empty_is_zero():
    transform = from_origin(400_000, 6_100_000, 10, 10)
    assert mask_area_m2(np.zeros((3, 3), dtype=bool), transform, UTM) == 0.0


def test_mask_area_geographic_matches_geodesic_order_of_magnitude():
    # A 1/12000 degree pixel at 55N is about 9.2 m (N-S) x 5.3 m (E-W), i.e. roughly 50 m².
    transform = from_origin(77.0, 55.0, 1 / 12000, 1 / 12000)
    area = mask_area_m2(np.ones((1, 1), dtype=np.uint8), transform, GEOGRAPHIC)
    assert 40.0 < area < 60.0


def test_relative_area_change():
    assert relative_area_change(100.0, 150.0) == pytest.approx(50.0)
    assert relative_area_change(200.0, 150.0) == pytest.approx(-25.0)
    assert relative_area_change(0.0, 150.0) is None


def test_area_change_report_fields():
    report = area_change_report(1_000_000.0, 1_200_000.0)
    assert report["delta_m2"] == pytest.approx(200_000.0)
    assert report["delta_km2"] == pytest.approx(0.2)
    assert report["delta_percent"] == pytest.approx(20.0)
    assert report["area_first_km2"] == pytest.approx(1.0)
