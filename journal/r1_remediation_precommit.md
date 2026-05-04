# Pre-Commit: R1 Red-Team Remediation

Per `journal/r1_signature_rule_redteam.md`. Each finding gets a disposition committed BEFORE any code. Discipline anchor.

## Per-finding disposition

| ID | Severity | Disposition | Verification |
|----|----------|-------------|--------------|
| **C1** | Critical | FIX — partition-change gate replaces byte-difference gate | New gate: ≥30% of random-expression pairs change relationship between rules (same→different OR different→same). Bites if dual-rule produces similar partition to sign-only. |
| **C2** | Critical | FIX — rule-difference probe set | Hand-construct probes specifically designed to route DIFFERENTLY under dual-rule than under sign-only. PASS if ≥70% of probes route to dual-expected class (not sign-expected). Bites if dual-rule doesn't actually differ behaviorally. |
| **H1** | High | FIX — constant-offset diagnostic | ~5 pairs of constant-offset equivalents (`x*x` vs `x*x+1`, etc.). Report whether they route to same class under dual-rule. Diagnostic only — outcome documented either way. |
| **H2** | High | FIX — granularity sweep | For 50 expressions with max_abs in [1, 20], measure signature distance between expressions with max_abs differing by 1. PASS if ≥80% of adjacent-max pairs have signature distance ≤ 2. Bites if integer-division granularity is causing real discontinuities. |
| **H3** | High | FIX — multi-seed wrapper | All measurements (partition-change, rule-difference, info-gain) run across 5 seeds. PASS if seed-mean ≥ committed thresholds AND stddev ≤ 15pp. |
| **H4** | High | FIX — timing measurement | Time confidence-weighted distance vs popcount distance over 10,000 routings. Diagnostic only. Document the ratio so R2 can plan. |
| **M1** | Medium | FIX — inter-class distance under new rule | Re-run §5 diagnostic from original remediation under dual-rule. PASS if min inter-class distance ≥ 4 for both curated banks (the original headroom threshold). Bites if dual-rule's conf bits add noise that compresses class separation. |
| **M2** | Medium | FIX — runtime regression check | At probe time, verify dual-signature is not bit-identical to a fake sign-only signature on the same expression. Catches silent fallthrough where the kernel call exists but doesn't add information. |
| **M3** | Medium | FIX — per-band distribution diagnostic | For each curated bank tile, report cell counts per band (strong-neg, weak-neg, zero, weak-pos, strong-pos). Diagnostic only — flag if zero-band exceeds 60% on average. |
| **M4** | Medium | FIX — unified seed list | All multi-seed measurements use the same seed list `{0xa1, 0xa2, 0xa3, 0xa4, 0xa5}`. Reports are cross-comparable. |
| **L1** | Low | DEFER with rationale | Runtime check (M2) gives the regression catch; a separate CI check is overkill for this cycle's scope. Discipline note recorded. |
| **L2** | Low | DEFER for P1-1 | Concern 7's full closure depends on exp/log primitives existing. Re-evaluate after P1-1. Documented. |
| **L3** | Low | DEFER for R2-watch | 1.5x per-tile storage is a scale concern, not a R1 concern. R2 must report bank memory footprint. Documented. |

**Total: 10 fixes + 3 deferred-with-rationale = 13/13.**

## Pre-committed gates summary

A R1-remediation **PASS** requires all of:

1. **C1 gate**: 5-seed mean partition-change rate ≥ 30%, stddev ≤ 15pp.
2. **C2 gate**: ≥70% of rule-difference probes route to dual-expected class (not sign-expected).
3. **H2 gate**: ≥80% of adjacent-max pairs have signature distance ≤ 2.
4. **H3 gate (subsumes C1, R1-B re-test, partition-change)**: mean ≥ thresholds AND stddev ≤ 15pp on all multi-seed measurements.
5. **M1 gate**: minimum inter-class distance ≥ 4 for both curated banks under new rule.
6. **M2 gate**: runtime regression check passes on all probes (no silent fallthrough detected).

A WEAK is any combination not meeting all six but not FAILing any.

A FAIL is any of:
- C1 mean < 15% (rule is barely changing partition)
- C2 < 50% (rule-difference probes mostly route to sign-expected, suggesting rule isn't actually different in operational terms)
- M1 min < 2 for either arity (worse than sign-only headroom)

## Diagnostic outcomes (no PASS/FAIL gate; report only)

- H1: constant-offset merger behavior under dual-rule — document outcome.
- H4: timing ratio — document for R2.
- M3: per-band distribution — flag if zero exceeds 60%.

## What this remediation deliberately does NOT do

- Does not change the dual-threshold rule itself (the rule under test is fixed).
- Does not re-run the original R1 verification binary's gates (those still hold; this remediation tests sharper claims).
- Does not address concerns about cross-arity or exp/log (P1).
- Does not commit to changing the rule based on findings — if a gate FAILs, that's data for a follow-on cycle, not an immediate redesign.

## Order of execution

1. Write this doc (in progress).
2. Write `gesh/bench/expr_routing_r1_remediation.c` with all sections.
3. Update CMakeLists.
4. Build, run, verify against gates.
5. Closeout doc.
