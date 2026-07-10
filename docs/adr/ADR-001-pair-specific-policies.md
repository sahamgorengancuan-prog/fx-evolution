# ADR-001: Pair-specific policies instead of a shared genome

**Status:** Accepted, 2026-07-10

## Context
v8 evaluated one genome across five pairs via a scalar `universal_score`.
Verified consequences (audit D2/D3): the champion over-fired on
BNBUSDT/DOGEUSDT (210/226 holdout trades) while starving on ETHUSDT (17
trades); feasibility (`n_passed`) froze at 2/5 from generation 4 while the
scalar kept climbing.

## Decision
The *engine*, target vector, and evaluation protocol are universal. The
*policy* is per pair: `PairPolicy{symbol → RegimeRouter, RegimePolicy[regime],
FallbackPolicy=FLAT}`. Regimes within a pair may use unrelated indicator
families. `NO_EDGE_FOUND` is a legal terminal outcome per pair.

## Consequences
- No cross-pair averaging in fitness; cross-pair transfer happens only via
  the memory/operator-prior layer (ADR-005), never via shared parameters.
- Search cost scales with pairs; mitigated by starting with one pair
  (BNBUSDT) end-to-end before generalizing.
