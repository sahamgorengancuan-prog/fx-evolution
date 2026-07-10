from __future__ import annotations

from math import comb

import numpy as np
import pytest

import evoquant.validation as validation_pkg
from evoquant.errors import SplitConfigError
from evoquant.validation.bootstrap import (
    circular_block_bootstrap,
    drawdown_distribution,
    sharpe_confidence,
    white_reality_check,
)
from evoquant.validation.concentration import return_concentration
from evoquant.validation.cpcv import CPCVConfig, generate_cpcv, validate_cpcv


class TestBootstrap:
    def test_shapes_and_determinism(self):
        r = np.random.default_rng(0).normal(0.001, 0.01, 300)
        a = circular_block_bootstrap(r, block=24, n_samples=50, seed=7)
        b = circular_block_bootstrap(r, block=24, n_samples=50, seed=7)
        assert a.shape == (50, 300)
        assert np.array_equal(a, b)
        c = circular_block_bootstrap(r, block=24, n_samples=50, seed=8)
        assert not np.array_equal(a, c)

    def test_sharpe_ci_brackets_truth(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0.005, 0.01, 600)  # true per-period SR = 0.5
        ci = sharpe_confidence(r, block=20, n_samples=300, seed=3)
        assert ci.p05 < 0.5 < ci.p95
        assert ci.p05 < ci.p50 < ci.p95

    def test_drawdown_distribution_is_path_statistic(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0.0, 0.02, 400)
        dd = drawdown_distribution(r, block=20, n_samples=200, seed=5)
        assert 0.0 <= dd.p05 <= dd.p50 <= dd.p95 <= 1.0
        assert dd.p95 > 0.01  # volatile path must show drawdown


class TestRealityCheck:
    def test_noise_battery_high_median_p_value(self):
        """A single 5%-level test rejects pure noise ~5% of the time by
        definition, so assert over several independent noise batteries:
        the MEDIAN p-value must be comfortably high. Fully seeded."""
        ps = []
        for data_seed in range(5):
            m = np.random.default_rng(data_seed).normal(0.0, 0.01, (400, 15))
            ps.append(
                white_reality_check(m, block=20, n_samples=300, seed=11)["p_value"]
            )
        assert float(np.median(ps)) > 0.2

    def test_true_edge_low_p_value_across_seeds(self):
        for data_seed in range(3):
            m = np.random.default_rng(data_seed).normal(0.0, 0.01, (400, 15))
            m[:, 4] += 0.004  # genuine edge well above noise
            out = white_reality_check(m, block=20, n_samples=300, seed=11)
            assert out["p_value"] < 0.05

    def test_deterministic_under_seed(self):
        m = np.random.default_rng(5).normal(0, 0.01, (200, 6))
        a = white_reality_check(m, seed=1)
        b = white_reality_check(m, seed=1)
        assert a == b


class TestCPCV:
    CFG = CPCVConfig(n_groups=6, n_test_groups=2, purge_bars=10, embargo_bars=5)

    def test_combination_count(self):
        splits = generate_cpcv(1_200, self.CFG)
        assert len(splits) == comb(6, 2) == 15

    def test_invariants_hold(self):
        from evoquant.data.splits import IndexRange

        splits = generate_cpcv(1_200, self.CFG)
        validate_cpcv(splits, 1_200, self.CFG)  # raises on violation
        for s in splits:
            for tr in s.train:
                for te in s.test:
                    assert not tr.intersects(te)
                    purge_zone = IndexRange(max(0, te.start - 10), te.start)
                    embargo_zone = IndexRange(te.stop, min(1_200, te.stop + 5))
                    assert not tr.intersects(purge_zone)
                    assert not tr.intersects(embargo_zone)

    def test_every_group_appears_as_test(self):
        splits = generate_cpcv(1_200, self.CFG)
        seen = {g for s in splits for g in s.test_groups}
        assert seen == set(range(6))

    def test_too_small_configs_rejected(self):
        with pytest.raises(SplitConfigError):
            generate_cpcv(8, self.CFG)
        with pytest.raises(SplitConfigError):
            CPCVConfig(n_groups=4, n_test_groups=4, purge_bars=0, embargo_bars=0)

    def test_deterministic(self):
        a = generate_cpcv(1_200, self.CFG)
        b = generate_cpcv(1_200, self.CFG)
        assert [(s.test_groups, s.train) for s in a] == [
            (s.test_groups, s.train) for s in b
        ]


class TestConcentration:
    def test_hand_computed_shares(self):
        from evoquant.backtest.fast_engine import Trade

        def trade(pnl):
            return Trade(
                direction=1, entry_index=0, exit_index=1, entry_price=1, exit_price=1,
                qty_base=1, lots=1, sl_price=0, tp_price=2, pnl_quote=pnl,
                costs_quote=0, exit_reason="tp",
            )

        trades = [trade(80.0), trade(15.0), trade(5.0), trade(-30.0)]
        out = return_concentration(trades)
        assert out["top1_winner_share"] == pytest.approx(0.8)
        assert out["top3_winner_share"] == pytest.approx(1.0)
        assert out["winner_hhi"] == pytest.approx(0.8**2 + 0.15**2 + 0.05**2)
        assert out["net_pnl"] == pytest.approx(70.0)


class TestForbiddenApis:
    def test_terminal_return_permutation_api_does_not_exist(self):
        """Audit S2: permuting identical returns cannot produce a meaningful
        terminal-return percentile (compounding is order-invariant). The
        API must not exist anywhere in the validation package."""
        import importlib
        import pkgutil

        forbidden_fragments = ("mc_return_percentile", "terminal_return")
        for mod_info in pkgutil.iter_modules(validation_pkg.__path__):
            module = importlib.import_module(f"evoquant.validation.{mod_info.name}")
            for attr in dir(module):
                for frag in forbidden_fragments:
                    assert frag not in attr.lower(), (
                        f"forbidden API {attr!r} found in {module.__name__}"
                    )
