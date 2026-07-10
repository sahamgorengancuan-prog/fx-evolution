"""Combinatorially purged cross-validation split generation.

Extends the Phase-1 split machinery: the research region is cut into S
contiguous groups; every C(S, k) choice of k test groups forms one split.
Training data excludes the test groups, a purge window *before* each test
block, and an embargo window *after* each test block — the same leakage
rules the walk-forward generator enforces, now combinatorial.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

from evoquant.data.splits import IndexRange, subtract_ranges, total_len
from evoquant.errors import SplitConfigError, SplitInvariantError


@dataclass(frozen=True)
class CPCVConfig:
    n_groups: int
    n_test_groups: int
    purge_bars: int
    embargo_bars: int
    min_train_bars: int = 1

    def __post_init__(self) -> None:
        if self.n_groups < 2:
            raise SplitConfigError("CPCV needs >= 2 groups")
        if not (1 <= self.n_test_groups < self.n_groups):
            raise SplitConfigError(
                "n_test_groups must be in [1, n_groups)", value=self.n_test_groups
            )
        if self.purge_bars < 0 or self.embargo_bars < 0:
            raise SplitConfigError("purge/embargo must be >= 0")


@dataclass(frozen=True)
class CPCVSplit:
    index: int
    test_groups: tuple[int, ...]
    train: tuple[IndexRange, ...]
    test: tuple[IndexRange, ...]


def generate_cpcv(n_bars: int, config: CPCVConfig) -> list[CPCVSplit]:
    if n_bars < config.n_groups * 2:
        raise SplitConfigError("Too few bars for CPCV grouping", n_bars=n_bars)
    bounds = [round(i * n_bars / config.n_groups) for i in range(config.n_groups + 1)]
    groups = [IndexRange(bounds[i], bounds[i + 1]) for i in range(config.n_groups)]

    splits: list[CPCVSplit] = []
    for idx, combo in enumerate(combinations(range(config.n_groups), config.n_test_groups)):
        test_ranges = tuple(groups[g] for g in combo)
        cuts: list[IndexRange] = []
        for tr in test_ranges:
            cuts.append(tr)
            if config.purge_bars:
                cuts.append(IndexRange(max(0, tr.start - config.purge_bars), tr.start))
            if config.embargo_bars:
                cuts.append(IndexRange(tr.stop, min(n_bars, tr.stop + config.embargo_bars)))
        train_ranges = tuple(subtract_ranges([IndexRange(0, n_bars)], cuts))
        if total_len(train_ranges) < config.min_train_bars:
            raise SplitConfigError(
                "CPCV split has insufficient training data after purge/embargo",
                split=idx,
                train_bars=total_len(train_ranges),
            )
        splits.append(
            CPCVSplit(index=idx, test_groups=combo, train=train_ranges, test=test_ranges)
        )

    expected = comb(config.n_groups, config.n_test_groups)
    if len(splits) != expected:  # pragma: no cover - combinatorial identity
        raise SplitInvariantError("CPCV combination count mismatch")
    validate_cpcv(splits, n_bars, config)
    return splits


def validate_cpcv(splits: list[CPCVSplit], n_bars: int, config: CPCVConfig) -> None:
    for s in splits:
        for tr in s.train:
            for te in s.test:
                if tr.intersects(te):
                    raise SplitInvariantError(
                        "CPCV train intersects test", split=s.index
                    )
                purge_zone = IndexRange(max(0, te.start - config.purge_bars), te.start)
                if len(purge_zone) and tr.intersects(purge_zone):
                    raise SplitInvariantError(
                        "CPCV purge violated", split=s.index, test_group=te.to_dict()
                    )
                embargo_zone = IndexRange(
                    te.stop, min(n_bars, te.stop + config.embargo_bars)
                )
                if len(embargo_zone) and tr.intersects(embargo_zone):
                    raise SplitInvariantError(
                        "CPCV embargo violated", split=s.index, test_group=te.to_dict()
                    )
