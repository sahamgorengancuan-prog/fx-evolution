# ADR-004: Hybrid search — GP + constraint-dominance NSGA + MAP-Elites + CMA-ES + contextual bandit

**Status:** Accepted, 2026-07-10 (implementation: Phase 5)

## Context
v8 used a single scalar fitness (audit D3) and a top-N hall of fame that
collapsed to one lineage (D4).

## Decision
Responsibilities are split:
- **Grammar-guided GP** evolves structure.
- **Constraint-dominance NSGA-II** ranks: feasible ≻ infeasible; among
  infeasible, smaller violation ≻ larger; among feasible, Pareto rank +
  crowding. Objectives (median OOS CAGR, LCB Sharpe, PF, −maxDD, −turnover,
  −fold dispersion, −complexity, regime stability, cost-stress survival)
  are never scalarized before selection.
- **MAP-Elites** archive over behavior descriptors (trade frequency,
  holding period, long/short exposure, trend/reversion character, cost
  sensitivity, complexity, return concentration, tail profile) preserves
  diverse mechanisms.
- **CMA-ES / Bayesian refinement** runs only after a structure is stable
  (same canonical AST surviving k generations), on parameters only.
- **Contextual Thompson sampling** picks mutation operators per
  (pair-fingerprint, regime) from empirical outcome posteriors; reward =
  feasibility/Pareto/novelty/OOS-stability improvement, not best score.
- **Surrogate ranker** may schedule compute but never assigns final
  fitness; every promoted candidate is canonically re-backtested.
- Stagnation detection uses hypervolume, feasibility rate, novelty entropy,
  lineage concentration, and failure-cluster dominance, driving a logged
  escalation state machine ending in `NO_EDGE_FOUND`. No claim that more
  generations improve results.
