"""Tests for the group-wise split."""

from __future__ import annotations

import pytest

from lake_segmentation.data.splits import assign_splits

FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}


def test_split_is_reproducible_and_groups_are_disjoint():
    groups = [f"lake_{index}" for index in range(10)]
    first = assign_splits(groups, FRACTIONS, seed=42)
    second = assign_splits(list(reversed(groups)), FRACTIONS, seed=42)
    assert first == second
    assert set(first) == set(groups)
    assert set(first.values()) == {"train", "val", "test"}


def test_split_respects_fractions():
    groups = [f"lake_{index}" for index in range(10)]
    assignment = assign_splits(groups, FRACTIONS, seed=1)
    counts = {split: list(assignment.values()).count(split) for split in FRACTIONS}
    # 10 groups cannot be split exactly as 70/15/15: the largest remainder goes to train.
    assert counts["train"] >= 6
    assert counts["train"] / 10 >= FRACTIONS["train"] - 0.15
    assert counts["val"] >= 1
    assert counts["test"] >= 1
    assert sum(counts.values()) == 10


def test_too_few_groups_raise():
    with pytest.raises(ValueError):
        assign_splits(["one", "two"], FRACTIONS, seed=0)


def test_bad_fractions_raise():
    with pytest.raises(ValueError, match="sum to 1"):
        assign_splits(
            [f"lake_{index}" for index in range(5)], {"train": 0.5, "val": 0.1, "test": 0.1}, seed=0
        )


def test_size_balanced_split_keeps_patch_ratios():
    sizes = {
        "big_a": 52,
        "big_b": 50,
        "medium_a": 39,
        "medium_b": 28,
        "small_a": 16,
        "small_b": 14,
        "small_c": 12,
        "small_d": 10,
        "small_e": 9,
        "small_f": 8,
    }
    assignment = assign_splits(list(sizes), FRACTIONS, seed=0, sizes=sizes)
    counts = {
        split: sum(size for lake, size in sizes.items() if assignment[lake] == split)
        for split in FRACTIONS
    }
    total = sum(sizes.values())
    assert counts["train"] / total == pytest.approx(0.7, abs=0.12)
    assert counts["test"] >= 20
    assert counts["val"] >= 20
    # every lake is used exactly once, with at least one lake per split
    assert sorted(assignment) == sorted(sizes)
    assert set(assignment.values()) == {"train", "val", "test"}
