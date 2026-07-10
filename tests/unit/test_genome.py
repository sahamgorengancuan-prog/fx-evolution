"""Property tests: generated and mutated genomes always compile."""
from __future__ import annotations

import numpy as np
import pytest

from evoquant.backtest.signals import RiskBlock
from evoquant.features.ast import Node
from evoquant.features.types import Dimension
from evoquant.genome.generator import generate_bool, generate_series
from evoquant.genome.genome import StrategyGenome, random_genome
from evoquant.genome.mutations import OPERATOR_NAMES, crossover, mutate


def test_random_genomes_always_compile():
    rng = np.random.default_rng(7)
    for _ in range(50):
        g = random_genome(rng, "TESTUSDT")
        g.validate()  # raises on any type/structure error
        assert g.complexity() > 0


def test_generated_series_have_requested_dimension():
    from evoquant.features.compiler import infer_dimension

    rng = np.random.default_rng(11)
    for dim in (
        Dimension.PRICE,
        Dimension.RETURN,
        Dimension.ZSCORE,
        Dimension.OSCILLATOR_0_1,
        Dimension.OSCILLATOR_0_100,
        Dimension.VOLATILITY,
        Dimension.UNSCALED,
    ):
        for _ in range(20):
            assert infer_dimension(generate_series(rng, dim, depth=3)) is dim


def test_generated_bools_compile_to_bool():
    from evoquant.features.compiler import infer_dimension

    rng = np.random.default_rng(13)
    for _ in range(50):
        assert infer_dimension(generate_bool(rng, depth=3)) is Dimension.BOOL


def test_generation_deterministic_under_seed():
    a = [random_genome(np.random.default_rng(5), "X").genome_hash() for _ in range(1)]
    b = [random_genome(np.random.default_rng(5), "X").genome_hash() for _ in range(1)]
    assert a == b


@pytest.mark.parametrize("operator", [op for op in OPERATOR_NAMES if op != "crossover"])
def test_mutations_preserve_compilability(operator):
    rng = np.random.default_rng(17)
    for _ in range(15):
        parent = random_genome(rng, "TESTUSDT")
        child, applied = mutate(parent, operator, rng, generation=1)
        child.validate()
        if applied:
            assert child.genome_hash() != parent.genome_hash()
            assert child.parent_hashes == (parent.genome_hash(),)
            assert child.origin == "mutation"
        else:
            assert child.genome_hash() == parent.genome_hash()


def test_crossover_compiles_and_records_parents():
    rng = np.random.default_rng(19)
    a, b = random_genome(rng, "X"), random_genome(rng, "X")
    child, applied = crossover(a, b, rng, generation=2)
    child.validate()
    if applied:
        assert set(child.parent_hashes) == {a.genome_hash(), b.genome_hash()}
        assert child.origin == "crossover"


def test_unknown_operator_rejected():
    rng = np.random.default_rng(23)
    with pytest.raises(ValueError):
        mutate(random_genome(rng, "X"), "quantum_leap", rng, 0)


def test_semantic_duplicates_share_genome_hash():
    """abs(abs(x)) vs abs(x) in a component => identical genome identity."""
    macd = Node(
        op="sub",
        children=(
            Node(op="ema", period=3, children=(Node(op="close"),)),
            Node(op="ema", period=8, children=(Node(op="close"),)),
        ),
    )
    abs_abs = Node(op="abs", children=(Node(op="abs", children=(macd,)),))
    redundant = Node(
        op="gt",
        children=(
            Node(op="zscore", period=21, children=(abs_abs,)),
            Node(op="constant", value=1.0, unit="ZSCORE"),
        ),
    )
    clean = Node(
        op="gt",
        children=(
            Node(op="zscore", period=21, children=(Node(op="abs", children=(macd,)),)),
            Node(op="constant", value=1.0, unit="ZSCORE"),
        ),
    )
    other = Node(
        op="lt",
        children=(
            Node(op="chop", period=21),
            Node(op="constant", value=0.5, unit="OSCILLATOR_0_1"),
        ),
    )
    risk = RiskBlock(
        sl_atr=2.0, tp_atr=2.0, risk_per_trade=0.01, max_bars=24, cooldown_bars=2, atr_period=14
    )

    def genome(entry: Node) -> StrategyGenome:
        return StrategyGenome(
            symbol="X", entry_long=entry, entry_short=other, allow=other, risk=risk
        )

    assert genome(redundant).genome_hash() == genome(clean).genome_hash()
