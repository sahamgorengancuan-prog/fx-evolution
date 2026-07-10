# fx-evolution / evoquant

Auditable rebuild of an evolutionary trading-strategy discovery platform.
A **universal search engine**, not a universal strategy: shared evaluation
protocol and governance, pair-specific policies, `NO_EDGE_FOUND` as a
first-class outcome.

**Status: Phase 2** — Phase 1 (data contracts, immutable manifests,
quality gates, nested walk-forward splits with purge/embargo, experiment
state machine, one-shot final lockbox) plus the dimension-typed causal
feature DSL: a compiler that rejects dimensionally invalid rules at
compile time, AST canonicalization/semantic-duplicate hashing, and
batch/incremental evaluators with causality and parity test coverage.
No search, no backtesting, no performance claims.

- Forensic audit of the previous (v8) system: [`docs/FORENSIC_AUDIT.md`](docs/FORENSIC_AUDIT.md)
- Phase plan & acceptance criteria: [`docs/PHASE_PLAN.md`](docs/PHASE_PLAN.md)
- Architecture decision records: [`docs/adr/`](docs/adr)
- Decision log: [`DECISIONS.md`](DECISIONS.md)
- What is *not* claimed to work: [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md)
- Claim-by-claim test evidence: [`VALIDATION_STATUS.md`](VALIDATION_STATUS.md)

## Install & test

```bash
pip install -e ".[dev]"
pytest
```

## Phase-1 CLI

```bash
# QA a ForexSB export
python -m evoquant.cli data-validate data/raw/BNBUSDT_H1.json

# Build the immutable data manifest
python -m evoquant.cli data-manifest data/raw/BNBUSDT_H1.json \
    --out artifacts/BNBUSDT_manifest.json

# Create + seal an experiment (manifest, split plan, lockbox, repro record)
python -m evoquant.cli experiment-create data/raw/BNBUSDT_H1.json \
    --config configs/splits_bnbusdt_h1.json --out-dir experiments/EXP-demo
```

## Layout

```
src/evoquant/
  errors.py            structured, artifact-writing errors (fail closed)
  repro.py             reproducibility records (data/config/code hashes, seed)
  data/
    contracts.py       InstrumentSpec — metadata-first, fail-closed
    loaders.py         ForexSB JSON loader (UTC-explicit timestamps)
    quality.py         hard/soft QA battery
    manifest.py        immutable content-addressed data manifests
    splits.py          nested walk-forward + purge/embargo + lockbox carve-out
  experiment/
    state.py           lifecycle state machine with append-only event log
    lockbox.py         one-shot lockbox guard; research_view()
  features/
    types.py           dimension system (PRICE, RETURN, ..., UNSCALED)
    ast.py             immutable typed expression nodes
    registry.py        operator specs (type rule, lookback, batch, stream)
    primitives.py      ~30 causal operators, batch + incremental
    compiler.py        type checking, lookback/complexity, evaluators
    simplifier.py      simplification, canonicalization, semantic hashing
    proposals.py       LLM indicator-proposal schema (validation only)
  cli.py
```

First end-to-end instrument: **BNBUSDT H1** (75,791 bars, 2017-11-06 →
2026-07-07, committed under `data/raw/`). Additional pairs only after the
one-pair pipeline passes causality, nested OOS, tick validation, and
Python–MQL5 parity (see phase plan).
