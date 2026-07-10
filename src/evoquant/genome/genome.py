"""Strategy genome for one pair-regime island.

Structure (entry/filter ASTs) + risk parameters + lineage metadata.
Identity is the semantic hash of the canonicalized ASTs plus the rounded
risk block — syntactic duplicates collapse (v8 D4/D5 fix).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from evoquant.backtest.signals import RiskBlock
from evoquant.data.manifest import sha256_of_dict
from evoquant.features.ast import Node
from evoquant.features.compiler import compile_feature
from evoquant.features.registry import CompileError, FeatureTypeError
from evoquant.features.simplifier import semantic_hash, simplify
from evoquant.features.types import Dimension
from evoquant.genome.generator import generate_bool


@dataclass(frozen=True)
class StrategyGenome:
    symbol: str
    entry_long: Node
    entry_short: Node
    allow: Node
    risk: RiskBlock
    origin: str = "random"  # random | mutation | crossover | local_refine
    parent_hashes: tuple[str, ...] = field(default_factory=tuple)
    created_generation: int = 0

    def genome_hash(self) -> str:
        return sha256_of_dict(
            {
                "entry_long": semantic_hash(self.entry_long),
                "entry_short": semantic_hash(self.entry_short),
                "allow": semantic_hash(self.allow),
                "risk": {
                    "sl_atr": round(self.risk.sl_atr, 4),
                    "tp_atr": round(self.risk.tp_atr, 4),
                    "risk_per_trade": round(self.risk.risk_per_trade, 6),
                    "max_bars": self.risk.max_bars,
                    "cooldown_bars": self.risk.cooldown_bars,
                    "atr_period": self.risk.atr_period,
                },
            }
        )

    def validate(self) -> None:
        """Compile all three expressions; raise if any is invalid."""
        for name, node in (
            ("entry_long", self.entry_long),
            ("entry_short", self.entry_short),
            ("allow", self.allow),
        ):
            compiled = compile_feature(node)
            if compiled.dimension is not Dimension.BOOL:
                raise FeatureTypeError(
                    f"genome.{name} must compile to BOOL", got=compiled.dimension.value
                )

    def complexity(self) -> int:
        return sum(
            compile_feature(n).complexity
            for n in (self.entry_long, self.entry_short, self.allow)
        )

    def simplified(self) -> StrategyGenome:
        return replace(
            self,
            entry_long=simplify(self.entry_long),
            entry_short=simplify(self.entry_short),
            allow=simplify(self.allow),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "entry_long": self.entry_long.to_dict(),
            "entry_short": self.entry_short.to_dict(),
            "allow": self.allow.to_dict(),
            "risk": {
                "sl_atr": self.risk.sl_atr,
                "tp_atr": self.risk.tp_atr,
                "risk_per_trade": self.risk.risk_per_trade,
                "max_bars": self.risk.max_bars,
                "cooldown_bars": self.risk.cooldown_bars,
                "atr_period": self.risk.atr_period,
            },
            "origin": self.origin,
            "parent_hashes": list(self.parent_hashes),
            "created_generation": self.created_generation,
            "genome_hash": self.genome_hash(),
        }


def random_risk(rng: np.random.Generator) -> RiskBlock:
    return RiskBlock(
        sl_atr=round(float(rng.uniform(1.0, 4.0)), 4),
        tp_atr=round(float(rng.uniform(1.0, 4.0)), 4),
        risk_per_trade=round(float(rng.uniform(0.002, 0.02)), 6),
        max_bars=int(rng.choice([12, 24, 36, 48])),
        cooldown_bars=int(rng.choice([0, 2, 4, 8])),
        atr_period=int(rng.choice([8, 14, 21])),
    )


def random_genome(
    rng: np.random.Generator, symbol: str, generation: int = 0, depth: int = 3
) -> StrategyGenome:
    """Generate a random, guaranteed-compilable genome."""
    for _ in range(20):  # generation is type-directed; retries are a safety net
        genome = StrategyGenome(
            symbol=symbol,
            entry_long=generate_bool(rng, depth),
            entry_short=generate_bool(rng, depth),
            allow=generate_bool(rng, depth),
            risk=random_risk(rng),
            origin="random",
            created_generation=generation,
        )
        try:
            genome.validate()
            return genome
        except (CompileError, FeatureTypeError):  # pragma: no cover - safety net
            continue
    raise CompileError("failed to generate a valid genome after 20 attempts")
