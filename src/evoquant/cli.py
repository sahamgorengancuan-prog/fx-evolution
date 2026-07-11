"""Minimal Phase-1 CLI.

Usage:
    python -m evoquant.cli data-validate <fsb_json>
    python -m evoquant.cli data-manifest <fsb_json> --out <manifest.json>
    python -m evoquant.cli experiment-create <fsb_json> --config <split_config.json> \
        --out-dir <dir> [--seed N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evoquant import repro
from evoquant.data.loaders import load_fsb_json
from evoquant.data.manifest import build_manifest, write_manifest
from evoquant.data.quality import run_quality_checks, validate_or_raise
from evoquant.data.splits import SplitConfig, generate_nested_walk_forward
from evoquant.errors import EvoquantError
from evoquant.experiment.state import Experiment, ExperimentState


def _cmd_data_validate(args: argparse.Namespace) -> int:
    bars, spec, _ = load_fsb_json(args.data)
    report = run_quality_checks(bars, spec)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


def _cmd_data_manifest(args: argparse.Namespace) -> int:
    bars, spec, _ = load_fsb_json(args.data)
    report = validate_or_raise(bars, spec)
    manifest = build_manifest(bars, spec, report, args.data)
    path = write_manifest(manifest, args.out)
    print(json.dumps({"manifest_id": manifest.manifest_id, "path": str(path)}, indent=2))
    return 0


def _cmd_experiment_create(args: argparse.Namespace) -> int:
    bars, spec, _ = load_fsb_json(args.data)
    report = validate_or_raise(bars, spec)
    manifest = build_manifest(bars, spec, report, args.data)

    cfg_doc = json.loads(Path(args.config).read_text())
    split_cfg = SplitConfig(**cfg_doc)
    plan = generate_nested_walk_forward(bars.n_bars, split_cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest, out_dir / "data_manifest.json")
    (out_dir / "split_plan.json").write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True))

    exp = Experiment.create(name=args.name or bars.symbol)
    exp.seal(
        data_content_hash=manifest.content_hash,
        split_plan_hash=plan.plan_hash,
        config_hash=repro.config_hash(cfg_doc),
        seed=args.seed,
    )
    exp.transition(ExperimentState.SEARCH_ACTIVE, actor="cli")
    exp.save(out_dir / "experiment.json")

    record = repro.capture(
        seed=args.seed,
        config=cfg_doc,
        data_content_hash=manifest.content_hash,
        repo_dir=Path(__file__).resolve().parents[2],
    )
    record.write(out_dir / "repro.json")

    print(
        json.dumps(
            {
                "experiment_id": exp.experiment_id,
                "state": exp.state.value,
                "data_manifest": manifest.manifest_id,
                "split_plan_hash": plan.plan_hash,
                "lockbox": plan.lockbox.to_dict(),
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )
    return 0


def _cmd_mql5_parity(args: argparse.Namespace) -> int:
    from evoquant.mql5.parity import parity_from_bundle

    report = parity_from_bundle(
        args.bundle, args.signals, args.trades, price_atol=args.price_atol
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.status == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evoquant")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data-validate", help="Run QA checks on a ForexSB JSON file")
    p.add_argument("data")
    p.set_defaults(func=_cmd_data_validate)

    p = sub.add_parser("data-manifest", help="Build an immutable data manifest")
    p.add_argument("data")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_data_manifest)

    p = sub.add_parser("experiment-create", help="Create and seal an experiment")
    p.add_argument("data")
    p.add_argument("--config", required=True, help="Split config JSON")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=20260710)
    p.add_argument("--name", default="")
    p.set_defaults(func=_cmd_experiment_create)

    p = sub.add_parser(
        "mql5-parity", help="Diff real Strategy Tester CSVs against a bundle"
    )
    p.add_argument("--bundle", required=True)
    p.add_argument("--signals", required=True, help="EA evoquant_signals.csv")
    p.add_argument("--trades", required=True, help="EA evoquant_trades.csv")
    p.add_argument("--price-atol", type=float, default=1e-6)
    p.set_defaults(func=_cmd_mql5_parity)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except EvoquantError as exc:
        print(json.dumps(exc.to_artifact(), indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
