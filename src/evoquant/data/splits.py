"""Nested walk-forward split generation with purge, embargo, and lockbox.

Layout (bar indices, half-open ranges):

```
[ ......................... research region ........................ |purge| lockbox ]
   outer fold k:
     train: [0, test_k.start - purge)  minus  embargo windows after
            every earlier outer test block
     test : block k of the contiguous outer-test tail
   inner fold j (inside outer train):
     train: [0, val_j.start - purge)   minus  embargo windows after
            earlier inner validation blocks (and earlier outer tests)
     val  : block j of the inner-validation tail of the outer train
```

* **Purge** removes training bars immediately *before* an evaluation block,
  so information from open positions / lookback windows cannot straddle the
  boundary. It must be >= the maximum holding horizon plus feature overlap.
* **Embargo** removes training bars immediately *after* an evaluation block,
  so serial correlation cannot leak an evaluated period into later training.
* The **lockbox** is the most recent segment. Nothing in the plan touches
  it; it is reachable only through the experiment lockbox guard.

Plans are deterministic pure functions of ``(n_bars, config)``, serialize
to canonical JSON, and carry a stable hash that experiments seal.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from evoquant.data.manifest import sha256_of_dict
from evoquant.errors import SplitConfigError, SplitInvariantError


@dataclass(frozen=True, order=True)
class IndexRange:
    """Half-open [start, stop) range of bar indices."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < self.start:
            raise SplitConfigError("Invalid IndexRange", start=self.start, stop=self.stop)

    def __len__(self) -> int:
        return self.stop - self.start

    def intersects(self, other: IndexRange) -> bool:
        if len(self) == 0 or len(other) == 0:
            return False  # empty ranges intersect nothing
        return self.start < other.stop and other.start < self.stop

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "stop": self.stop}


def subtract_ranges(base: Iterable[IndexRange], cuts: Iterable[IndexRange]) -> list[IndexRange]:
    """Set-subtract cut ranges from base ranges (all half-open)."""
    result = list(base)
    for cut in cuts:
        nxt: list[IndexRange] = []
        for r in result:
            if not r.intersects(cut):
                nxt.append(r)
                continue
            if r.start < cut.start:
                nxt.append(IndexRange(r.start, cut.start))
            if cut.stop < r.stop:
                nxt.append(IndexRange(cut.stop, r.stop))
        result = nxt
    return [r for r in result if len(r) > 0]


def total_len(ranges: Iterable[IndexRange]) -> int:
    return sum(len(r) for r in ranges)


@dataclass(frozen=True)
class InnerFold:
    train: tuple[IndexRange, ...]
    validation: IndexRange


@dataclass(frozen=True)
class OuterFold:
    index: int
    train: tuple[IndexRange, ...]
    test: IndexRange
    inner_folds: tuple[InnerFold, ...]


@dataclass(frozen=True)
class SplitConfig:
    n_outer_folds: int
    outer_test_bars: int
    n_inner_folds: int
    inner_val_bars: int
    #: >= max holding horizon + max feature lookback overlap, in bars.
    purge_bars: int
    embargo_bars: int
    lockbox_bars: int
    min_train_bars: int

    def __post_init__(self) -> None:
        positive = (
            "n_outer_folds",
            "outer_test_bars",
            "n_inner_folds",
            "inner_val_bars",
            "lockbox_bars",
            "min_train_bars",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise SplitConfigError(f"SplitConfig.{name} must be > 0", value=getattr(self, name))
        for name in ("purge_bars", "embargo_bars"):
            if getattr(self, name) < 0:
                raise SplitConfigError(f"SplitConfig.{name} must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitPlan:
    schema_version: str
    n_bars: int
    config: SplitConfig
    lockbox: IndexRange
    outer_folds: tuple[OuterFold, ...]
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def plan_hash(self) -> str:
        return sha256_of_dict(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "n_bars": self.n_bars,
            "config": self.config.to_dict(),
            "lockbox": self.lockbox.to_dict(),
            "outer_folds": [
                {
                    "index": f.index,
                    "train": [r.to_dict() for r in f.train],
                    "test": f.test.to_dict(),
                    "inner_folds": [
                        {
                            "train": [r.to_dict() for r in i.train],
                            "validation": i.validation.to_dict(),
                        }
                        for i in f.inner_folds
                    ],
                }
                for f in self.outer_folds
            ],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SplitPlan:
        def rng(x: dict[str, int]) -> IndexRange:
            return IndexRange(x["start"], x["stop"])
        return cls(
            schema_version=d["schema_version"],
            n_bars=d["n_bars"],
            config=SplitConfig(**d["config"]),
            lockbox=rng(d["lockbox"]),
            outer_folds=tuple(
                OuterFold(
                    index=f["index"],
                    train=tuple(rng(r) for r in f["train"]),
                    test=rng(f["test"]),
                    inner_folds=tuple(
                        InnerFold(
                            train=tuple(rng(r) for r in i["train"]),
                            validation=rng(i["validation"]),
                        )
                        for i in f["inner_folds"]
                    ),
                )
                for f in d["outer_folds"]
            ),
            notes=d.get("notes", {}),
        )


def _embargo_cuts(blocks: Iterable[IndexRange], embargo_bars: int) -> list[IndexRange]:
    if embargo_bars <= 0:
        return []
    return [IndexRange(b.stop, b.stop + embargo_bars) for b in blocks]


def generate_nested_walk_forward(n_bars: int, config: SplitConfig) -> SplitPlan:
    """Deterministically generate a nested walk-forward plan.

    Raises :class:`SplitConfigError` when the configuration cannot fit into
    ``n_bars`` without violating ``min_train_bars``.
    """
    lockbox = IndexRange(n_bars - config.lockbox_bars, n_bars)
    research_end = lockbox.start - config.purge_bars  # purge before lockbox
    tests_total = config.n_outer_folds * config.outer_test_bars
    first_test_start = research_end - tests_total
    if first_test_start < config.min_train_bars + config.purge_bars:
        raise SplitConfigError(
            "Split configuration does not fit: first outer fold would have "
            "less than min_train_bars of training data.",
            n_bars=n_bars,
            required=(
                config.min_train_bars
                + 2 * config.purge_bars
                + tests_total
                + config.lockbox_bars
            ),
            config=config.to_dict(),
        )

    tests = [
        IndexRange(
            first_test_start + k * config.outer_test_bars,
            first_test_start + (k + 1) * config.outer_test_bars,
        )
        for k in range(config.n_outer_folds)
    ]

    outer_folds: list[OuterFold] = []
    for k, test in enumerate(tests):
        train_end = test.start - config.purge_bars
        base = [IndexRange(0, train_end)]
        cuts = _embargo_cuts(tests[:k], config.embargo_bars)
        train_ranges = subtract_ranges(base, cuts)
        if total_len(train_ranges) < config.min_train_bars:
            raise SplitConfigError(
                "Outer fold training set below min_train_bars after purge/embargo",
                fold=k,
                train_bars=total_len(train_ranges),
            )
        inner = _generate_inner_folds(train_end, train_ranges, config)
        outer_folds.append(
            OuterFold(index=k, train=tuple(train_ranges), test=test, inner_folds=inner)
        )

    plan = SplitPlan(
        schema_version="1.0",
        n_bars=n_bars,
        config=config,
        lockbox=lockbox,
        outer_folds=tuple(outer_folds),
        notes={
            "layout": "expanding_walk_forward",
            "purge_before_lockbox_bars": config.purge_bars,
        },
    )
    validate_plan(plan)
    return plan


def _generate_inner_folds(
    train_end: int, outer_train_ranges: list[IndexRange], config: SplitConfig
) -> tuple[InnerFold, ...]:
    vals_total = config.n_inner_folds * config.inner_val_bars
    first_val_start = train_end - vals_total
    if first_val_start < config.purge_bars + 1:
        raise SplitConfigError(
            "Inner validation blocks do not fit inside outer training region",
            train_end=train_end,
            required_val_bars=vals_total,
        )
    vals = [
        IndexRange(
            first_val_start + j * config.inner_val_bars,
            first_val_start + (j + 1) * config.inner_val_bars,
        )
        for j in range(config.n_inner_folds)
    ]
    inner: list[InnerFold] = []
    for j, val in enumerate(vals):
        base = [IndexRange(0, val.start - config.purge_bars)]
        # keep inner training inside the outer training set (respect outer embargo cuts)
        base = [
            IndexRange(max(r.start, b.start), min(r.stop, b.stop))
            for b in base
            for r in outer_train_ranges
            if r.intersects(b)
        ]
        cuts = _embargo_cuts(vals[:j], config.embargo_bars)
        train_ranges = subtract_ranges(base, cuts)
        inner.append(InnerFold(train=tuple(train_ranges), validation=val))
    return tuple(inner)


def validate_plan(plan: SplitPlan) -> None:
    """Enforce every leakage invariant; raise :class:`SplitInvariantError`."""
    cfg = plan.config
    lb = plan.lockbox

    def _assert(cond: bool, msg: str, **details: Any) -> None:
        if not cond:
            raise SplitInvariantError(msg, **details)

    _assert(lb.stop == plan.n_bars, "Lockbox must end at the last bar")
    all_research: list[IndexRange] = []

    prev_test_stop = -1
    for f in plan.outer_folds:
        _assert(f.test.start > prev_test_stop - 1, "Outer tests must be ordered")
        _assert(f.test.start >= prev_test_stop, "Outer tests must not overlap")
        prev_test_stop = f.test.stop
        all_research.append(f.test)
        all_research.extend(f.train)

        for r in f.train:
            _assert(
                r.stop + cfg.purge_bars <= f.test.start,
                "Purge gap violated between outer train and test",
                fold=f.index,
                train_stop=r.stop,
                test_start=f.test.start,
            )
            _assert(not r.intersects(f.test), "Outer train intersects test", fold=f.index)

        # Embargo: no train bar within embargo window after any earlier test.
        for earlier in plan.outer_folds[: f.index] if cfg.embargo_bars > 0 else []:
            window = IndexRange(earlier.test.stop, earlier.test.stop + cfg.embargo_bars)
            for r in f.train:
                _assert(
                    not r.intersects(window),
                    "Embargo violated: training data inside embargo window "
                    "after an earlier outer test",
                    fold=f.index,
                    earlier_fold=earlier.index,
                )

        for i_idx, inner in enumerate(f.inner_folds):
            _assert(
                inner.validation.stop + cfg.purge_bars <= f.test.start,
                "Inner validation must stay clear of the outer test purge boundary",
                fold=f.index,
            )
            for r in inner.train:
                _assert(
                    r.stop + cfg.purge_bars <= inner.validation.start,
                    "Purge gap violated between inner train and validation",
                    fold=f.index,
                    inner=i_idx,
                )
                _assert(
                    not r.intersects(inner.validation),
                    "Inner train intersects validation",
                    fold=f.index,
                    inner=i_idx,
                )
                _assert(
                    not r.intersects(f.test),
                    "Inner train intersects outer test",
                    fold=f.index,
                    inner=i_idx,
                )
            for earlier_val in f.inner_folds[:i_idx] if cfg.embargo_bars > 0 else []:
                window = IndexRange(
                    earlier_val.validation.stop,
                    earlier_val.validation.stop + cfg.embargo_bars,
                )
                for r in inner.train:
                    _assert(
                        not r.intersects(window),
                        "Embargo violated between inner folds",
                        fold=f.index,
                        inner=i_idx,
                    )
            all_research.append(inner.validation)
            all_research.extend(inner.train)

    for r in all_research:
        _assert(
            not r.intersects(lb),
            "Research range intersects the lockbox",
            range=r.to_dict(),
            lockbox=lb.to_dict(),
        )
        _assert(0 <= r.start and r.stop <= plan.n_bars, "Range out of bounds")
        _assert(
            r.stop <= lb.start - cfg.purge_bars,
            "Research range reaches into the purge zone before the lockbox",
            range=r.to_dict(),
            lockbox_start=lb.start,
            purge_bars=cfg.purge_bars,
        )
