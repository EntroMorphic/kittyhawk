---
date: 2026-04-14
phase: NODES
topic: §14 "Seven Open Decisions" in m4t/docs/M4T_SUBSTRATE.md
---

# Nodes of interest

## N1. The list is not homogeneous
The seven items span three categories: (a) substrate-level invariant questions, (b) statements that are actually proven or trivially decidable, (c) consumer- or thesis-level concerns that don't belong in a substrate spec. Lumping them under one header distorts all three.

## N2. 14.5 is a theorem masquerading as an open question
Max SDOT output over MTFP4 inputs = 16 × 40 × 40 = 25,600 ≪ int32 max (2^31 − 1 ≈ 2.15 × 10⁹). Exactness is provable, not a judgment call.

## N3. 14.7 is outside the substrate's purview entirely
"Which benchmark validates the routing thesis?" is a question for the thesis, not the substrate. M4T serves consumers; it does not pick what the consumers prove.

## N4. 14.6 depends on 14.7
Whether to pull LUT-backed nonlinearities from archive depends on what routing architecture needs them, which depends on what consumer is being built, which depends on 14.7. So 14.6 is not independent; it's downstream of a question that's outside §14.

## N5. 14.2 may be vestigial
In the routing primitives we kept (`apply_signed`, `signature_update`, `distance_batch`), accumulation happens within exponent-uniform blocks established at write time. Cross-block add across different block exponents may never be exercised in routing workloads. If so, the question has no consumer and shouldn't be answered speculatively.

## N6. 14.3 has a natural answer
A zero mantissa is the additive and multiplicative identity for any block exponent: `0 × 3^e = 0`. Zero-padding the tail of a tensor extends length with identity elements — not a silent substrate action, a mathematical null. The apparent conflict with the "widen, don't round" invariant was a framing error.

## N7. 14.4 is cosmetic
INT8_MIN-as-sentinel vs parallel 1-byte status array express identical semantics. The substrate's integrity is unaffected by either choice; the question is stylistic (numeric purity vs storage compactness).

## N8. 14.1 is the only genuinely empirical open
Logical block size (aggregating hardware blocks under shared exponent) is a cache/prefetch tuning question that has no theoretical derivation. It wants workload data, not philosophy.

## N9. The throughline works as a sieve, not as a hammer
"Substrate guarantees invariants; consumer names preconditions" doesn't produce uniform answers when the questions aren't uniform. But applied as a *sorting mechanism*, it correctly separates the seven into their true categories (N1).

## Tensions

### T1. Elegant throughline vs heterogeneous list
The throughline felt like it explained everything. In fact, it was exposing a categorization failure. This is the method doing its job: the throughline's elegance was *diagnostic*, not conclusive. Apparent unity came from poor upstream triage.

### T2. "Widen, don't round" (§8.5) vs zero-pad (14.3)
Looks like a conflict: widen-don't-round says no silent semantic shifts; zero-padding looks like a silent act. Resolution: zero-padding inserts identity elements, which is not a semantic shift — it's a length extension with no numeric consequence for any sum, product, or reduction that respects identity. No conflict.

### T3. "Answer every open" vs "defer until a consumer asks"
If a question has no consumer driving it (14.2 may be here), answering it speculatively is exactly the kind of substrate overreach we're rebuilding away from. The discipline is: the substrate declines to answer questions it doesn't have to answer yet.

## Dependencies

- 14.6 → 14.7 (nonlinearity scope depends on thesis benchmark)
- 14.2 → consumer-driven (needs routing workload that exercises cross-block add)
- 14.1 → consumer-driven (needs workload to measure prefetch/cache stress)
