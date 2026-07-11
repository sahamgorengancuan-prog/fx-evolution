"""End-to-end experiment orchestration — pair-agnostic by construction.

The pipeline never mentions a symbol: it takes any ForexSB JSON file and
a configuration, and runs

    load -> QA (fail closed) -> manifest -> sealed splits -> fingerprint
    -> train-only regime fit -> island search on inner folds
    -> outer-OOS evaluation (data the search never touched)
    -> validation battery (PBO across candidates, DSR w/ effective N,
       bootstrap CI, cost stress, concentration)
    -> honest verdict (SHORTLISTED | NO_EDGE_FOUND | REJECTED)
    -> MQL5 bundle + model card + repro record.

A pass is never forced; the lockbox is NOT opened here — freezing and the
one-shot lockbox evaluation remain deliberate human actions via the
experiment state machine.
"""
from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from evoquant import repro
from evoquant.backtest.costs import ExecutionAssumptions, resolve_costs
from evoquant.backtest.fast_engine import run_backtest
from evoquant.data.loaders import load_fsb_json
from evoquant.data.manifest import build_manifest, write_manifest
from evoquant.data.quality import validate_or_raise
from evoquant.data.splits import SplitConfig, generate_nested_walk_forward
from evoquant.experiment.lockbox import research_view
from evoquant.experiment.state import Experiment, ExperimentState
from evoquant.genome.mutations import OPERATOR_NAMES
from evoquant.memory.bandit import ThompsonOperatorSelector, context_key
from evoquant.memory.store import MemoryStore
from evoquant.mql5.bundle import write_bundle
from evoquant.regimes.fingerprints import compute_fingerprint
from evoquant.regimes.router import REGIME_NAMES, QuantileRegimeRouter, Regime
from evoquant.search.evaluator import EvalContext, _compile_policy, evaluate_genome
from evoquant.search.island import IslandConfig, SearchIsland
from evoquant.search.objectives import TargetConfig
from evoquant.validation.bootstrap import sharpe_confidence, white_reality_check
from evoquant.validation.concentration import return_concentration
from evoquant.validation.pbo import pbo_cscv
from evoquant.validation.sharpe import deflated_sharpe, effective_trials
from evoquant.validation.stress import cost_stress

ProgressFn = Callable[[dict[str, Any]], None]


@dataclass
class RunConfig:
    """Everything a run needs. `data_file` may be ANY ForexSB export."""

    data_file: str
    out_dir: str
    seed: int = 20260710
    #: use only the most recent N bars (None = full history). Recorded in
    #: the report — a truncated run is labeled, never silent.
    max_bars: int | None = None
    # split plan (bars); defaults sized for H1 crypto history
    n_outer_folds: int = 4
    outer_test_bars: int = 2160
    n_inner_folds: int = 4
    inner_val_bars: int = 1080
    purge_bars: int = 96
    embargo_bars: int = 24
    lockbox_bars: int = 10800
    min_train_bars: int = 8000
    # search budget
    population_size: int = 10
    n_generations: int = 4
    outer_folds_to_search: int = 1
    inner_blocks_per_fold: int = 2
    eval_warmup_bars: int = 300
    # execution assumptions (ALL explicit — fail closed otherwise)
    initial_equity: float = 1_000_000.0
    commission_pct_notional: float = 0.001
    assumed_spread_points: float = 5.0
    funding_mode: str = "not_applicable_spot"
    min_notional: float = 10.0
    assumptions_label: str = (
        "ASSUMED (not historical): user-configured commission/spread; "
        "spot, no funding"
    )
    # targets (shared vector; never force-passed)
    targets: dict[str, Any] = field(default_factory=lambda: TargetConfig().to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_experiment(config: RunConfig, on_progress: ProgressFn | None = None) -> dict[str, Any]:
    """Run one pair end-to-end. Returns (and persists) the report dict."""

    def progress(stage: str, **detail: Any) -> None:
        event = {
            "stage": stage,
            "ts_utc": _dt.datetime.now(_dt.UTC).isoformat(),
            **detail,
        }
        if on_progress is not None:
            on_progress(event)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = TargetConfig(**config.targets)

    # ---- 1. data, QA, manifest (any pair) ----------------------------- #
    progress("loading_data", file=config.data_file)
    bars, spec, _header = load_fsb_json(config.data_file)
    if config.max_bars is not None and bars.n_bars > config.max_bars:
        bars = bars.slice(bars.n_bars - config.max_bars, bars.n_bars)
        progress("data_truncated", used_bars=bars.n_bars,
                 note="most recent max_bars only; recorded in report")
    qa = validate_or_raise(bars, spec)
    manifest = build_manifest(bars, spec, qa, config.data_file)
    write_manifest(manifest, out_dir / "data_manifest.json")
    progress("data_ready", symbol=bars.symbol, n_bars=bars.n_bars,
             qa_warnings=[c.name for c in qa.warnings])

    # ---- 2. sealed splits + experiment state -------------------------- #
    split_cfg = SplitConfig(
        n_outer_folds=config.n_outer_folds,
        outer_test_bars=config.outer_test_bars,
        n_inner_folds=config.n_inner_folds,
        inner_val_bars=config.inner_val_bars,
        purge_bars=config.purge_bars,
        embargo_bars=config.embargo_bars,
        lockbox_bars=config.lockbox_bars,
        min_train_bars=config.min_train_bars,
    )
    plan = generate_nested_walk_forward(bars.n_bars, split_cfg)
    (out_dir / "split_plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True)
    )
    experiment = Experiment.create(name=f"{bars.symbol}-e2e")
    experiment.seal(
        data_content_hash=manifest.content_hash,
        split_plan_hash=plan.plan_hash,
        config_hash=repro.config_hash(config.to_dict()),
        seed=config.seed,
    )
    experiment.transition(ExperimentState.SEARCH_ACTIVE, actor="orchestrator")
    research = research_view(bars, plan)
    progress("experiment_sealed", experiment_id=experiment.experiment_id,
             lockbox=plan.lockbox.to_dict())

    # ---- 3. costs, fingerprint, regime router (train-only) ------------ #
    assumptions = ExecutionAssumptions(
        label=config.assumptions_label,
        initial_equity=config.initial_equity,
        commission_pct_notional=config.commission_pct_notional,
        assumed_spread_points=config.assumed_spread_points,
        funding_mode=config.funding_mode,
        min_notional=config.min_notional,
    )
    costs = resolve_costs(spec, research, assumptions)

    fold0_train = plan.outer_folds[0].train[0]
    train_bars = research.slice(fold0_train.start, fold0_train.stop)
    fingerprint = compute_fingerprint(train_bars)
    router = QuantileRegimeRouter()
    router_params = router.fit(train_bars)
    regime_labels = router.infer(research)
    dominant = Regime(int(np.bincount(regime_labels[regime_labels > 0]).argmax())) \
        if np.any(regime_labels > 0) else Regime.UNKNOWN
    ctx_key = context_key(fingerprint, REGIME_NAMES[dominant])
    progress("characterized", fingerprint=fingerprint.to_dict(),
             dominant_regime=REGIME_NAMES[dominant], context=ctx_key)

    # ---- 4. island search per outer fold ------------------------------- #
    memory = MemoryStore(out_dir / "memory.sqlite")
    selector = ThompsonOperatorSelector(
        operators=tuple(op for op in OPERATOR_NAMES if op != "crossover"),
        seed=config.seed,
    )
    fold_champions: list[dict[str, Any]] = []
    n_folds = min(config.outer_folds_to_search, len(plan.outer_folds))
    for k in range(n_folds):
        fold = plan.outer_folds[k]
        inner_vals = tuple(
            i.validation for i in fold.inner_folds[-config.inner_blocks_per_fold:]
        )
        ctx = EvalContext(
            bars=research,
            spec=spec,
            costs=costs,
            eval_ranges=inner_vals,
            initial_equity=config.initial_equity,
            warmup_bars=config.eval_warmup_bars,
        )
        island = SearchIsland(
            IslandConfig(
                symbol=bars.symbol,
                population_size=config.population_size,
                n_generations=config.n_generations,
                seed=config.seed + k,
                stagnation_patience=2,
            ),
            ctx,
            targets,
            operator_selector=selector,
            memory=memory,
            context_key=ctx_key,
        )

        state = island.run()
        for entry in state.telemetry:
            progress("search_generation", fold=k, **entry)
        champ = island.champion()
        if champ is None:
            continue
        genome, inner_eval = champ
        fold_champions.append(
            {
                "fold": k,
                "genome": genome,
                "inner_eval": inner_eval,
                "island_outcome": state.outcome,
                "elites": [g for g in state.population],
            }
        )
        progress("fold_done", fold=k, outcome=state.outcome,
                 champion=genome.genome_hash(),
                 evaluated=len(island.lineage))

    if not fold_champions:
        experiment.transition(
            ExperimentState.CLOSED_NO_EDGE, actor="orchestrator",
            details={"reason": "no champion emerged"},
        )
        report = _finalize(
            out_dir, experiment, config, manifest, plan, "NO_EDGE_FOUND",
            reason="no champion emerged from any searched fold",
            extra={}, progress=progress,
        )
        return report

    # ---- 5. outer-OOS evaluation (untouched by search) ----------------- #
    best = fold_champions[0]
    fold = plan.outer_folds[best["fold"]]
    oos_ctx = EvalContext(
        bars=research,
        spec=spec,
        costs=costs,
        eval_ranges=(fold.test,),
        initial_equity=config.initial_equity,
        warmup_bars=config.eval_warmup_bars,
    )
    champion = best["genome"]
    oos_eval = evaluate_genome(champion, oos_ctx)
    progress("oos_evaluated", fold=best["fold"],
             oos_return=oos_eval.fold_returns, n_trades=oos_eval.n_trades_total)

    # per-bar OOS equity returns for champion + elites -> candidate battery
    start = max(0, fold.test.start - config.eval_warmup_bars)
    window = research.slice(start, fold.test.stop)
    offset = fold.test.start - start
    battery_cols: list[np.ndarray] = []
    battery_hashes: list[str] = []
    seen_hashes = {champion.genome_hash()}
    battery_genomes = [champion]
    for g in best["elites"]:
        h = g.genome_hash()
        if h not in seen_hashes:
            seen_hashes.add(h)
            battery_genomes.append(g)
        if len(battery_genomes) >= 10:
            break
    for g in battery_genomes:
        policy = _compile_policy(g, window)
        result = run_backtest(window, spec, policy, _slice(costs, start, fold.test.stop),
                              config.initial_equity)
        eq = result.equity[offset:]
        rets = np.diff(eq) / eq[:-1]
        battery_cols.append(rets)
        battery_hashes.append(g.genome_hash())
    battery = np.column_stack(battery_cols)
    champion_returns = battery_cols[0]
    champion_policy = _compile_policy(champion, window)
    champion_result = run_backtest(
        window, spec, champion_policy, _slice(costs, start, fold.test.stop),
        config.initial_equity,
    )

    # ---- 6. validation battery ----------------------------------------- #
    validation: dict[str, Any] = {"candidates_in_battery": battery_hashes}
    active = champion_returns[np.abs(champion_returns) > 0]
    if len(battery_hashes) >= 2 and battery.shape[0] >= 16:
        validation["pbo_cscv"] = pbo_cscv(battery, n_partitions=8).to_dict()
    else:
        validation["pbo_cscv"] = {"skipped": "needs >=2 candidates and >=16 rows"}
    if len(active) >= 8:
        n_eff = effective_trials(battery)
        validation["effective_trials"] = n_eff
        validation["deflated_sharpe"] = deflated_sharpe(champion_returns, n_trials=n_eff)
        validation["sharpe_ci"] = sharpe_confidence(
            champion_returns, block=24, n_samples=300, seed=config.seed
        ).to_dict()
        validation["reality_check"] = white_reality_check(
            battery, block=24, n_samples=300, seed=config.seed
        )
    else:
        validation["deflated_sharpe"] = {"skipped": "champion has <8 active OOS returns"}
    if champion_result.trades:
        validation["return_concentration"] = return_concentration(
            list(champion_result.trades)
        )
    validation["cost_stress"] = cost_stress(
        champion, oos_ctx, spread_multipliers=(1.0, 2.0, 3.0)
    )
    progress("validated", keys=sorted(validation))

    # ---- 7. honest verdict ---------------------------------------------- #
    oos_feasible = oos_eval.is_feasible(targets)
    dsr = validation.get("deflated_sharpe", {})
    dsr_value = dsr.get("deflated_sharpe") if isinstance(dsr, dict) else None
    verdict = "SHORTLISTED" if oos_feasible else "NO_EDGE_FOUND"
    verdict_reasons = []
    if not oos_feasible:
        verdict_reasons = [
            f"outer-OOS violation: {k}={v:.4f}"
            for k, v in oos_eval.violations(targets).items()
        ]
    if dsr_value is not None and dsr_value < 0.5:
        verdict_reasons.append(
            f"deflated Sharpe {dsr_value:.3f} < 0.5 (weak after multiplicity)"
        )
        if verdict == "SHORTLISTED":
            verdict = "REJECTED"

    if verdict == "SHORTLISTED":
        experiment.transition(ExperimentState.SHORTLISTED, actor="orchestrator",
                              details={"champion": champion.genome_hash()})
    else:
        experiment.transition(ExperimentState.CLOSED_NO_EDGE, actor="orchestrator",
                              details={"reasons": verdict_reasons})

    # ---- 8. artifacts ----------------------------------------------------- #
    bundle_paths = write_bundle(
        out_dir / "mql5_bundle", champion, spec, window, champion_policy,
        champion_result, deposit=config.initial_equity,
        skip_warmup_bars=config.eval_warmup_bars,
    )
    extra = {
        "champion": {
            "genome_hash": champion.genome_hash(),
            "genome": champion.to_dict(),
            "inner_eval": best["inner_eval"].to_dict(),
            "oos_eval": oos_eval.to_dict(),
        },
        "validation": validation,
        "verdict_reasons": verdict_reasons,
        "regime": {
            "params": router_params.to_dict(),
            "dominant": REGIME_NAMES[dominant],
        },
        "fingerprint": fingerprint.to_dict(),
        "bandit_posteriors": selector.to_dict(),
        "mql5_bundle": bundle_paths,
        "memory_events": memory.count(),
    }
    return _finalize(out_dir, experiment, config, manifest, plan, verdict,
                     reason="; ".join(verdict_reasons) or "outer-OOS targets met",
                     extra=extra, progress=progress)


def _slice(costs: Any, start: int, stop: int) -> Any:
    from dataclasses import replace

    return replace(costs, spread_points=costs.spread_points[start:stop])


def _finalize(
    out_dir: Path,
    experiment: Experiment,
    config: RunConfig,
    manifest: Any,
    plan: Any,
    verdict: str,
    reason: str,
    extra: dict[str, Any],
    progress: Callable[..., None],
) -> dict[str, Any]:
    record = repro.capture(
        seed=config.seed,
        config=config.to_dict(),
        data_content_hash=manifest.content_hash,
    )
    report = {
        "schema_version": "1.0",
        "kind": "evoquant_experiment_report",
        "engine_disclaimer": (
            "All performance numbers come from the BAR_APPROXIMATION reference "
            "engine under EXPLICIT cost assumptions; tick and MQL5 validation "
            "are PENDING. This is research telemetry, not a live-trading claim."
        ),
        "experiment_id": experiment.experiment_id,
        "symbol": manifest.symbol,
        "verdict": verdict,
        "verdict_reason": reason,
        "config": config.to_dict(),
        "data_manifest_id": manifest.manifest_id,
        "data_content_hash": manifest.content_hash,
        "split_plan_hash": plan.plan_hash,
        "lockbox": plan.lockbox.to_dict(),
        "lockbox_note": (
            "The lockbox has NOT been opened. Freezing and the one-shot "
            "lockbox evaluation are deliberate separate actions."
        ),
        "state": experiment.state.value,
        "repro": record.to_dict(),
        **extra,
    }
    experiment.save(out_dir / "experiment.json")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    _write_model_card(out_dir / "MODEL_CARD.md", report)
    progress("finished", verdict=verdict, report=str(out_dir / "report.json"))
    return report


def _write_model_card(path: Path, report: dict[str, Any]) -> None:
    champ = report.get("champion", {})
    lines = [
        f"# Model card — {report['symbol']} ({report['experiment_id']})",
        "",
        f"**Verdict: {report['verdict']}** — {report['verdict_reason']}",
        "",
        f"> {report['engine_disclaimer']}",
        "",
        "## Provenance",
        f"- data manifest: `{report['data_manifest_id']}`",
        f"- data hash: `{report['data_content_hash'][:16]}…`",
        f"- split plan: `{report['split_plan_hash'][:16]}…`",
        f"- seed: {report['config']['seed']}",
        f"- state: {report['state']}",
        f"- {report['lockbox_note']}",
        "",
        "## Champion",
    ]
    if champ:
        oos = champ.get("oos_eval", {})
        lines += [
            f"- genome: `{champ.get('genome_hash', 'n/a')}`",
            f"- outer-OOS fold returns: {oos.get('fold_returns')}",
            f"- outer-OOS trades: {oos.get('n_trades_total')}",
            f"- outer-OOS max drawdown: {oos.get('max_drawdown')}",
        ]
    else:
        lines.append("- none (no champion emerged)")
    lines += [
        "",
        "## Negative evidence & limitations",
        "- Tick/event validation: NOT RUN (no tick data).",
        "- MQL5 real-tick parity: PENDING (see mql5_bundle/).",
        "- See repository KNOWN_LIMITATIONS.md — it applies in full.",
    ]
    path.write_text("\n".join(lines) + "\n")
