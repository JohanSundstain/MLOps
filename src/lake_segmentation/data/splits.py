"""Group-wise train / validation / test split.

Patches cut from the same lake (and even more so from the same scene) are highly correlated,
so a random split over patches would leak information and overestimate the quality. The split
is therefore performed over whole lakes (AOIs): all patches of one lake land in exactly one
subset.

Lakes differ a lot in size (a big lake gives an order of magnitude more patches than a
dissected salt lake), so the grouping alone is not enough: the lakes are also distributed over
the splits in a size-balanced greedy way. That keeps the patch ratio close to 70 / 15 / 15 and
prevents the test set from being a handful of patches of two small lakes.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from lake_segmentation.constants import SPLIT_NAMES


def _split_group_counts(group_count: int, fractions: Mapping[str, float]) -> dict[str, int]:
    """Number of lakes per split, with at least one lake in every split."""

    test_count = max(1, int(round(float(fractions["test"]) * group_count)))
    val_count = max(1, int(round(float(fractions["val"]) * group_count)))
    train_count = group_count - test_count - val_count
    if train_count < 1:
        train_count = 1
        overflow = test_count + val_count - (group_count - 1)
        if overflow > 0:
            if val_count > 1:
                val_count -= overflow
            else:
                test_count -= overflow
    return {"train": train_count, "val": val_count, "test": test_count}


def assign_splits(
    groups: Sequence[str],
    fractions: Mapping[str, float],
    seed: int,
    sizes: Mapping[str, float] | None = None,
) -> dict[str, str]:
    """Map every group (lake) to ``"train"``, ``"val"`` or ``"test"``.

    The order of the groups is shuffled with ``seed``, so the assignment is reproducible and
    independent of the order in which the AOIs are listed in the config. When ``sizes`` (the
    number of patches per lake) is given, the largest lakes are distributed first, each one to
    the split that is furthest below its target share.
    """

    unique_groups = sorted(set(groups))
    if len(unique_groups) < len(SPLIT_NAMES):
        raise ValueError(
            f"Group-wise splitting needs at least {len(SPLIT_NAMES)} lakes, "
            f"got {len(unique_groups)}"
        )
    missing = [name for name in SPLIT_NAMES if name not in fractions]
    if missing:
        raise ValueError(f"Split fractions are missing for {missing}")
    total_fraction = sum(float(fractions[name]) for name in SPLIT_NAMES)
    if not np.isclose(total_fraction, 1.0, atol=1e-6):
        raise ValueError(f"Split fractions must sum to 1, got {total_fraction}")

    generator = np.random.default_rng(seed)
    shuffled = list(unique_groups)
    generator.shuffle(shuffled)
    group_counts = _split_group_counts(len(shuffled), fractions)
    assignment: dict[str, str] = {}

    if sizes is None:
        for index, group in enumerate(shuffled):
            if index < group_counts["train"]:
                assignment[group] = "train"
            elif index < group_counts["train"] + group_counts["val"]:
                assignment[group] = "val"
            else:
                assignment[group] = "test"
        return assignment

    weights = {group: max(float(sizes.get(group, 0.0)), 0.0) for group in unique_groups}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        for group in unique_groups:
            weights[group] = 1.0
        total_weight = float(len(unique_groups))

    if len(shuffled) <= EXHAUSTIVE_SEARCH_MAX_GROUPS:
        return _assign_by_exhaustive_search(shuffled, weights, fractions, group_counts)
    return _assign_by_greedy_search(shuffled, weights, fractions, group_counts)


EXHAUSTIVE_SEARCH_MAX_GROUPS = 24


def _squared_deviation(
    assignment: Mapping[str, str], weights: Mapping[str, float], targets: Mapping[str, float]
) -> float:
    totals = {split: 0.0 for split in SPLIT_NAMES}
    for group, split in assignment.items():
        totals[split] += weights[group]
    return sum((totals[split] - targets[split]) ** 2 for split in SPLIT_NAMES)


def _assign_by_exhaustive_search(
    shuffled: list[str],
    weights: Mapping[str, float],
    fractions: Mapping[str, float],
    group_counts: Mapping[str, int],
) -> dict[str, str]:
    """Try every combination of lakes for the smaller splits and keep the best fit.

    With a dozen lakes this is only a few thousand candidates, so the split can match the
    requested proportions much better than a greedy heuristic.
    """

    from itertools import combinations

    total_weight = sum(weights.values())
    targets = {split: float(fractions[split]) * total_weight for split in SPLIT_NAMES}
    held_out_count = group_counts["val"] + group_counts["test"]
    val_count = group_counts["val"]

    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    for held_out in combinations(shuffled, held_out_count):
        remaining = [group for group in shuffled if group not in held_out]
        for val_groups in combinations(held_out, val_count):
            assignment: dict[str, str] = {group: "train" for group in remaining}
            for group in held_out:
                assignment[group] = "val" if group in val_groups else "test"
            score = _squared_deviation(assignment, weights, targets)
            if score < best_score - 1e-9:
                best_score = score
                best_assignment = assignment
    if best_assignment is None:  # pragma: no cover - defensive
        raise RuntimeError("Could not find a valid split")
    return best_assignment


def _assign_by_greedy_search(
    shuffled: list[str],
    weights: Mapping[str, float],
    fractions: Mapping[str, float],
    group_counts: Mapping[str, int],
) -> dict[str, str]:
    """Fallback for many lakes: assign the biggest lakes first to the hungriest split."""

    total_weight = sum(weights.values())
    targets = {split: float(fractions[split]) * total_weight for split in SPLIT_NAMES}
    assigned_weight = {split: 0.0 for split in SPLIT_NAMES}
    lake_counts = {split: 0 for split in SPLIT_NAMES}
    assignment: dict[str, str] = {}
    for group in sorted(shuffled, key=lambda item: -weights[item]):
        available = [split for split in SPLIT_NAMES if lake_counts[split] < group_counts[split]]
        chosen = max(
            available,
            key=lambda split: (targets[split] - assigned_weight[split]) / max(targets[split], 1e-9),
        )
        assignment[group] = chosen
        assigned_weight[chosen] += weights[group]
        lake_counts[chosen] += 1
    return assignment
