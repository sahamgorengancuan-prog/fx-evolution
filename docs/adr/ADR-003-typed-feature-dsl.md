# ADR-003: Dimension-typed causal feature DSL

**Status:** Accepted, 2026-07-10 (implementation: Phase 2)

## Context
v8's champion contains `cross_above(rank(...), ret(8))` — a [0,1] rank
crossed against a raw return — and `abs(abs(keltner(233)))` (audit D5).
The grammar accepted dimensionally meaningless and redundant expressions,
burning search budget on syntactic noise and producing degenerate rules.

## Decision
Typed AST with dimensions {PRICE, RETURN, VOLATILITY, VOLUME,
OSCILLATOR_0_1, OSCILLATOR_0_100, ZSCORE, TIME, BOOL}. Compile-time
rejection of cross-dimension comparison without an explicit normalizer.
Deterministic canonical serialization, constant folding, idempotent/
involutive simplification (`abs∘abs → abs`), commutative argument ordering,
semantic-duplicate hashing, complexity accounting. All primitives causal
and incrementally evaluable; prefix-invariance is a released-gate test.
LLM-proposed primitives compile through a whitelisted formula schema in a
sandbox — never arbitrary code in the main process.

## Consequences
Smaller reachable grammar space (intended); every genome hash is a
semantic identity usable by the memory layer (ADR-005).
