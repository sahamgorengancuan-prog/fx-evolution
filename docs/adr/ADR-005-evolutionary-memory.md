# ADR-005: Event-sourced evolutionary memory

**Status:** Accepted, 2026-07-10 (implementation: Phase 6)

## Context
v8 kept only a champion + hall of fame; nothing recorded lineage, operator
outcomes, or failure modes; the LLM received raw (and holdout-tainted)
results (audit D1, D11).

## Decision
Append-only empirical event store (DuckDB/Parquet): every evaluated
candidate persists genome + canonical hash, parents/lineage, operator used,
pair/regime, dataset/code/config hashes, per-fold metrics, trade lists,
signal density, cost decomposition, sensitivity, structured failure labels
(`NO_SIGNAL`, `OVERFIRE`, `COST_DOMINATED`, `PARAMETER_CLIFF`, …), compute
cost. Separate semantic lesson store (hypothesis, observed outcome,
evidence count, confidence, applicability, active flag). Contextual
operator posteriors per (pair fingerprint, regime). LLM interaction is
retrieval-based (nearest failures/successes, high-confidence lessons,
contradictions) — never the raw history, never lockbox metrics.

## Consequences
Storage is a first-class artifact with schema migrations; replaying the
event log must reconstruct search state (tested).
