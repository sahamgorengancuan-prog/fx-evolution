# fx-evolution / evoquant

Auditable rebuild of an evolutionary trading-strategy discovery platform.
A **universal search engine**, not a universal strategy: shared evaluation
protocol and governance, pair-specific policies, `NO_EDGE_FOUND` as a
first-class outcome.

**Status: Phase 8** — sealed data governance (P1), dimension-typed
causal DSL (P2), two-tier backtesting with fail-closed labeled costs
(P3), train-only pair-regime layer (P4), constraint-dominance search core
with a logged escalation ladder and first-class `NO_EDGE_FOUND` (P5), and
the evolutionary memory + LLM layer (P6), and the statistical
validation framework (P7): CPCV, candidate-matrix PBO (CSCV),
probabilistic/deflated Sharpe with EFFECTIVE trial counts, block
bootstrap, White's Reality Check, cost/parameter/start stress, return
concentration (the order-invariant terminal-return permutation "test"
is absent by construction) — plus MQL5 export (P8): deterministic EA
transpilation mirroring Tier-A semantics, real-tick tester configs, CSV
parity contracts and a three-state differential comparator whose default
is PENDING_REAL_TICK, never assumed success. No performance claims.

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
  backtest/
    costs.py           fail-closed cost resolution + risk-based sizing
    signals.py         SignalPolicy — the executable strategy contract
    fast_engine.py     Tier A: BAR_APPROXIMATION bar engine
    event_engine.py    Tier B: TICK_EVENT engine (latency, rejection)
    metrics.py         plain metrics from MTM equity + trades
  regimes/
    fingerprints.py    slice-local causal pair fingerprints
    router.py          train-only quantile regime router (one-sided)
  genome/
    generator.py       type-directed random expression generation
    genome.py          StrategyGenome — semantic-hash identity
    mutations.py       type-safe mutation/crossover operators
  search/
    objectives.py      constraint vector + Pareto objectives (no scalar)
    nsga.py            constraint-dominance NSGA-II selection
    map_elites.py      behavioral quality-diversity archive
    stagnation.py      multi-signal detector + logged escalation ladder
    lineage.py         per-candidate lineage records
    evaluator.py       genome -> compiled signals -> Tier-A fold metrics
    island.py          the generational loop for one search island
  memory/
    store.py           append-only SQLite event store (versioned schema)
    failures.py        deterministic failure-label attribution
    bandit.py          contextual Thompson sampling over operators
    lessons.py         semantic lesson store (supersede, replay)
    retrieval.py       bounded LLM context building (lockbox-guarded)
  llm/
    contracts.py       strict agent JSON contracts + lockbox-leak guard
    client.py          OpenAI/OpenRouter chat client (env keys, fail closed)
  validation/
    cpcv.py            combinatorially purged CV split generation
    pbo.py             real PBO via CSCV over candidate matrices
    sharpe.py          PSR, DSR, effective trial counts
    bootstrap.py       block bootstrap CIs, DD distribution, Reality Check
    stress.py          cost / parameter / start-offset stress harnesses
    concentration.py   return-concentration diagnostics
  mql5/
    exporter.py        genome -> deterministic MQL5 EA + tester.ini
    parser.py          EA signal/trade CSV contracts
    parity.py          differential comparator (PENDING/PASS/FAIL)
    bundle.py          export bundle with manual run instructions
  cli.py
```

First end-to-end instrument: **BNBUSDT H1** (75,791 bars, 2017-11-06 →
2026-07-07, committed under `data/raw/`). Additional pairs only after the
one-pair pipeline passes causality, nested OOS, tick validation, and
Python–MQL5 parity (see phase plan).
