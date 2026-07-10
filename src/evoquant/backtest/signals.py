"""The executable strategy contract shared by both engines.

A :class:`SignalPolicy` carries **precomputed, per-bar, causal** arrays —
produced by the Phase-2 compiler, whose causality is test-enforced — plus
a risk block. Engines never call back into strategy code, which keeps
execution deterministic and identical across engines.

Timing convention (both engines): a signal on bar ``i`` (computed from
bar-``i`` close) is executed at bar ``i+1`` open (Tier A) / at the first
tick after bar ``i`` completes plus latency (Tier B). Same-bar execution
is impossible by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evoquant.errors import EvoquantError


class PolicyError(EvoquantError):
    code = "POLICY"


@dataclass(frozen=True)
class RiskBlock:
    sl_atr: float
    tp_atr: float
    risk_per_trade: float
    max_bars: int
    cooldown_bars: int
    atr_period: int

    def __post_init__(self) -> None:
        if self.sl_atr <= 0 or self.tp_atr <= 0:
            raise PolicyError("sl_atr and tp_atr must be > 0")
        if not (0 < self.risk_per_trade <= 0.1):
            raise PolicyError(
                "risk_per_trade must be in (0, 0.1]", value=self.risk_per_trade
            )
        if self.max_bars < 1 or self.cooldown_bars < 0 or self.atr_period < 1:
            raise PolicyError("max_bars >= 1, cooldown >= 0, atr_period >= 1 required")


@dataclass(frozen=True)
class SignalPolicy:
    """Per-bar causal signal arrays + risk parameters."""

    entry_long: np.ndarray  # bool
    entry_short: np.ndarray  # bool
    allow: np.ndarray  # bool regime/filter gate
    atr: np.ndarray  # float64, quote units, causal
    risk: RiskBlock
    #: provenance of the arrays (genome hash, compiler version, ...)
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        n = len(self.entry_long)
        for name in ("entry_short", "allow", "atr"):
            if len(getattr(self, name)) != n:
                raise PolicyError(f"SignalPolicy array length mismatch: {name}")
        for name in ("entry_long", "entry_short", "allow"):
            if getattr(self, name).dtype != np.bool_:
                raise PolicyError(f"SignalPolicy.{name} must be a boolean array")
        if self.atr.dtype != np.float64:
            raise PolicyError("SignalPolicy.atr must be float64")

    @property
    def n_bars(self) -> int:
        return len(self.entry_long)
