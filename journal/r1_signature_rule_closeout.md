# Closeout: R1 — Per-Expression-Tau Dual-Threshold Signature Rule

Per `journal/r1_signature_rule_synthesize.md` against pre-committed gates in `docs/PLAN_EXPRESSION_ROUTING_R2.md`.

## Verdict: PASS

```
R1-A backward-compat : PASS (29/30 = 96.7%)    gate >=70%
R1-B information gain: PASS (184/200 = 92.0%)  gate >=30%
R1-C substrate-kernel: PASS (by construction)
OVERALL R1: PASS
```

## What shipped

- `gesh/src/expr_signature.{h,c}`: `expr_to_signature_dual` — per-expression-tau dual-threshold ternarization. `tau_weak = max_abs/4`, `tau_strong = max_abs/2`, both computed from the expression's own outputs on the test inputs. Routes through `m4t_route_threshold_extract_dual`.
- `gesh/src/expr_bank.{h,c}`: `expr_bank_dual_t` and `expr_bank_dual_build`. Per-tile storage now includes both packed-trit signature AND parallel confidence bitmap. Equivalence-class detection extends to memcmp on `(trit || conf)`.
- `gesh/bench/expr_routing_r1.c`: R1 verification probe with all three gates.
- CMakeLists target `gesh_expr_routing_r1`.

The existing sign-only `expr_to_signature` and `expr_bank_build` are untouched. Original probes still PASS unchanged.

## Concerns disposition (from PLAN_EXPRESSION_ROUTING_R2.md)

| Concern | Disposition |
|---------|-------------|
| **2** (substrate's distinctive kernels unused) | **CLOSED.** `m4t_route_threshold_extract_dual` and `m4t_route_confidence_weighted_dist` now exercised in consumer code. Verifiable by grep. |
| **3** (sign-only uses base-3 as binary) | **CLOSED.** Third state now load-bearing — the "zero" band carries "values within max/4 of zero, relative to this expression's scale," a meaningful magnitude statement, not an accidental exact-zero. |
| **7** (compose-equivalence collapses precision-distinct expressions) | **PARTIALLY CLOSED.** New rule splits `x ≢ x³` (different magnitude profiles) where sign-only merged them. By extension, `exp(x)` and Taylor truncation will produce different signatures at positions where their magnitude profiles diverge. The full claim ("any precision-distinct pair routes differently") still depends on whether their divergence happens within the test-input range. |
| **8** (low discrimination headroom) | **PARTIALLY ADDRESSED.** New rule produces +8 more arity-1 classes and +17 more arity-2 classes per 100 random expressions vs sign-only. Discrimination capacity is meaningfully larger. The specific arity-1 min-distance flag (was 3 trits under sign-only) needs re-measurement under the new rule — that's R3's job. |

## Interesting subplots

**The partition reshapes, not just expands.** Under the new rule, the curated arity-1 bank has 11 classes (vs 10 under sign-only). One sign-only merger gone (`x ≢ x³` — different magnitude profiles). One new merger added (`(x-1)*(x+1)` joins `x*x`'s class because per-expression tau renders their tiny absolute differences at x=0,±1 below the weak threshold).

This is the rule trading off **absolute-precision sensitivity** for **global magnitude-profile sensitivity**. Defensible: at the resolution of 16 test inputs and per-expression tau, `x²-1` and `x*x` *do* behave essentially the same way — they're both quadratic-magnitude expressions with similar shapes. The exact integer differences at three points are noise relative to the global profile.

**At scale (100 random expressions), the new rule wins on discrimination.** Old rule: 22 / 41 classes (arity-1 / arity-2). New rule: 30 / 58 classes. Both arities see ~35-40% more classes after merging, meaning the new rule is genuinely splitting equivalence classes that sign-only collapsed.

## Honest concerns

**1. Per-expression tau makes `x²-1 ≡ x*x` under the new rule.** This is a feature (resolution-appropriate equivalence) but worth flagging — if a downstream consumer expected these to route differently (e.g., for symbolic-math correctness checking where adding 1 matters), the new rule won't deliver. The behavior is consistent with the rule's design goal (magnitude-profile equivalence) but inconsistent with naive expectations.

**2. The R1-A backward-compat gate was set at ≥70% precisely to allow the new rule to split classes the old rule wrongly merged.** We didn't actually need that headroom — the result was 96.7%, same as sign-only. The single MISS was the same subagent-self-flagged ambiguous probe as before. So the new rule is essentially backward-compatible on this probe set, NOT because it's the same as sign-only, but because the changes (split `x ≢ x³`, merge `x²-1 ≡ x*x`) didn't affect any subagent probe's expected target. A different probe set could surface real changes.

**3. Information-gain gate (R1-B) was very easy to clear.** 92% is far above the 30% gate. This is good evidence the new rule is doing more, but the gate didn't bite — we could have gotten the same PASS verdict with a much higher threshold (say 70%). H4-style discipline note: future cycles should set gates that bite.

**4. We tested information gain by signature byte-difference, not by partition difference.** Per pre-commit literal text. The stronger informal check (partition difference) was reported but not gated. If the gate had been "≥30% partition change," it would still have PASSed (22→30 = 36% more classes for arity-1; 41→58 = 41% more for arity-2). But it's good discipline to note the gap.

## What R1 does NOT close

- **Concern 1 (scope gap)**: still wide open. R2 is the track that addresses it.
- **Concern 8 (sig_dim sweep)**: R1 helps but R3 is the principled answer.
- **Cross-arity routing, exp/log, adversarial probes**: all still P1 or later.

## Substrate-discipline notes

- `m4t_route_threshold_extract_dual` call site: `gesh/src/expr_signature.c:expr_to_signature_dual`.
- `m4t_route_confidence_weighted_dist` call site: `gesh/bench/expr_routing_r1.c:route_dual`.
- All ternarization and band classification through substrate kernels. No open-coded sign step, no open-coded band classification. Bank constructor's memcmp on `(trit || conf)` concatenation is C library scratch, same acceptable category as the existing `expr_bank_build`.
- All 14 ctest binaries still green. Existing probes (`gesh_expr_routing_probe`, `gesh_expr_routing_remediation`) still PASS unchanged.

## Next track

**R3: discrimination headroom analysis.** With the new rule chosen, sweep `sig_dim ∈ {16, 32, 64, 128}` against the curated banks. Find a sig_dim where minimum inter-class distance ≥ 6 for both arities. Document the relationship between sig_dim and bank size for future cycles.

Then **R2: scale experiment.** With a defensible rule and dim, build banks of 100 / 500 / 1000 / 2000 random expression candidates. Report merger rate, distance distribution, auto-generated probe consistency.
