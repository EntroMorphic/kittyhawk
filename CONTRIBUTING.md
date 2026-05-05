# Contributing to Glyph

Glyph is a research codebase with strict discipline. Every contribution honors the invariants below.

## Non-negotiable invariants

1. **No binary floating point in runtime kernels.** `float`, `double`, `float16`, `bfloat16` are banned in every runtime kernel of `libm4t` and in every per-query / per-batch path of higher layers. Sanctioned non-runtime float sites are enumerated in [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md) §12.

2. **No random projections in image pipelines.** Direct ternary quantization of pixels and gradients is the correct representation. Random projection matrices destroy the spatial structure that downstream LSH ranking depends on.

3. **No random weights anywhere.** Random ternary weights are wrong in every layer (projections, FFN, mixing, future layers). Every dimension must represent something specific — a derived feature, a class anchor, a centroid, or a measured quantity.

4. **No dense resolvers in classification paths.** Cascades stay routed end-to-end. Pixel L1/L2/cosine/centroid resolvers are forbidden in the classifier path.

5. **No primitive without named consumer demand.** New kernels appear when a consumer asks for them. Speculative infrastructure does not earn its place. If you are tempted to build a kernel "because we'll need it," name the consumer first or wait.

6. **DELETE = never.** Superseded code moves to an archive directory (`01MAY26_archived/` for prior-cycle work, future archive directories for future cycles), or is suffixed `.archived`, or is otherwise preserved. Never `rm`.

7. **Substrate-level specs are upstream of kernel designs.** A kernel design that does not trace back to constraints in [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md) is suspect. Re-read the relevant spec section before any design memo phase. Spec amendments require a journal cycle; kernel designs that contradict the spec without amending it are a discipline violation. (Lifted to a general principle by `journal/xexpo_design_closeout.md` after an external review surfaced that a kernel design had not been pressure-tested against the spec section it claimed to amend.)

## Working pattern

The project runs an LMM cycle for substantive research questions:

```
raw → nodes → reflect → synthesize → closeout
```

Each phase is a markdown file in `journal/`. Negative results are findings; record them with the same rigor as positive results. Cycles that revise their own prior conclusions are a feature, not a bug — see the prior project's `journal/learned_routing_closeout.md` for the canonical example.

## Build hygiene

- `-Werror` is on by default. Warnings fail the build.
- All tests under `ctest --test-dir build` must pass before merge.
- Code that breaks an existing test is a regression, not a feature. Fix the regression or fix the test deliberately — never silently skip.

## Substrate spec

The canonical numeric-system contract is [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md). Substantive changes to it require a journal cycle and an explicit memo recording the decision.

## Pull requests

- Keep PRs scoped: one cycle of work, one logical change.
- The PR description should explain *why* the change is needed and what it would mean if it didn't land.
- Negative-result PRs are welcome. Document the cycle, the hypothesis, the measurement, and the falsification. These are findings.

## Post-commit doc-currency checklist

After any commit that lands a kernel, a spec change, or a status flip, sweep the documentation ensemble for staleness. Two prior end-to-end red-teams in this project caught ~10 cross-document drift issues each (stale test counts, stale section headings, broken path references, "ONLINE" sections still labeled "pending consumer"). Each issue was small in isolation; their accumulation across files produced a misleading first impression for any new reader.

Before merging, audit:

- [ ] **Top-level `README.md`** — does the Status section still describe the current state? Does the Documentation table reference any new docs?
- [ ] **`m4t/README.md`** — do test counts and section headings reflect the actual test surface? Are tier labels current?
- [ ] **`m4t/docs/M4T_SUBSTRATE.md`** — are §17 cross-reference rows still pointing to the right files? Do path references resolve in the *current* tree (not the archived tree)? If a section mentions `archive/...`, verify it should say `01MAY26_archived/...` or be rephrased to drop the path.
- [ ] **`docs/THESIS.md`** — are any "open questions" now closed? Have closed ones been moved to a "closed questions" section rather than left misleading?
- [ ] **`docs/REMEDIATION_PLAN.md` and similar plans** — does the document's status header match the executed reality? If not, add a status note at the top that points readers to the CHANGELOG for the actual narrative.
- [ ] **`docs/FINDINGS.md`** — are new measurements axis-recorded? Are stale "(none yet)" claims still accurate?
- [ ] **`CHANGELOG.md`** — is the most recent entry complete? Does it cite the journal cycle that produced it?
- [ ] **Source comments** — do any `/* see m4t/tools/foo.c */` or similar references still resolve? Path-broken comments mislead future readers.
- [ ] **Spec amendments → journal cycles.** Per principle 7, every substantive `M4T_SUBSTRATE.md` edit needs a `journal/*_spec_amend.md` cycle (lightweight is fine; full RAW→NODES→REFLECT→SYNTHESIZE only when amending substrate semantics).
- [ ] **Aliasing assertions on every writable output.** Pattern set by the m4t kernel red-teams: any function that writes to a caller-provided buffer asserts the output doesn't alias any const input. `assert((const void*)dst != (const void*)src)`. Cheap, catches a class of silent-corruption bugs at debug time. The Gesh Phase A.1 red-team caught this gap in consumer code; it transfers from substrate to consumer.
- [ ] **Multi-seed validation for any directional measurement claim.** Set by the Phase A.2 sweep red-team (C1). If a benchmark or measurement supports a claim with directional language — "peak gain", "anomaly", "winner", "outperforms" — it needs ≥3 seeds with averages reported, not single-seed numbers. Single-seed measurements are exploratory; multi-seed measurements are evidence. The Phase A.2 single-seed sweep produced a "+15pp peak" and a "−2pp anomaly" that both evaporated under multi-seed averaging.
- [ ] **Hypothesis vs finding distinction in measurement docs.** Mechanism explanations for empirical results ("implicit denoising via X", "training walks into worse basin") are *hypotheses* until tested by a mechanism-revealing follow-up measurement. Documenting them as findings rather than hypotheses overstates what the data shows. Phase A.2 docs flag "implicit denoising via random projection" as a hypothesis; pattern transfers.
- [ ] **Multi-config gates the story; multi-seed gates the cell.** Set by the Phase B probe red-team. Multi-seed validates that one measurement cell's number is not a seed artifact. Multi-config validates that one measurement cell's *interpretation* is not a config artifact. Any closeout that asserts a *causal mechanism* ("X is the bottleneck", "Y is responsible for the failure") needs measurements at multiple configurations of the variables being attributed — not just multiple seeds at one configuration. The Phase B closeout originally attributed Gate 1 failure to consumer architecture from a 2-cell measurement; an ablation across budget × n_train was needed before the causal claim earned support.
- [ ] **Kernel use gates the substrate-claim.** Set by the Phase B kernel-use audit. Any substrate-claim measurement must run through the substrate kernels it claims. Re-implementing kernel-shaped arithmetic (multiply-accumulate, sign-threshold) in consumer or bench code invalidates the substrate-claim semantics — the measurement timed and validated something other than libm4t. For every MAC or threshold in consumer/bench code, ask: is there a libm4t kernel that does this? If yes, why isn't it called? The Phase B audit found 7 substrate-bypass sites; cleanup is in `journal/gesh_substrate_discipline_cleanup.md`.
- [ ] **Match the scope of evidence to the scope of claim.** Set by the SDOT-finding3 red-team's meta-pattern across four prior red-teams. The unifying pattern: lower-N evidence supporting higher-N claim is methodology debt at every layer.
  - Single seed → cannot claim population (Phase A.2). Multi-seed validates that one cell's number isn't a seed artifact.
  - Single config → cannot claim mechanism (Phase B). Multi-config validates that one cell's *interpretation* isn't a config artifact.
  - Hand-coded loops → cannot claim substrate-shaped behavior (Phase B kernel-use audit). Substrate kernels are required to claim substrate semantics.
  - In-scope kernel → cannot claim out-of-scope behavior (SDOT-finding3 red-team C1). When a kernel is used outside its labeled input class, either amend the spec or add a wrapper that exposes the new class.
  - Outcome → cannot claim mechanism (SDOT-finding3 red-team C3). Demonstrating an outcome consistent with a hypothesis is not the same as demonstrating the mechanism — a probe that exposes the mechanism's signature (e.g., per-class confusion matrix to verify pigeonhole collisions) upgrades hypothesis to finding.
  - Single workload-shape → cannot claim general kernel performance (V4-residual-3 closeout + ternary_mac_routing R-G3). A "fast" or "no-delta" measurement on a carry-dependent workload doesn't generalize to pipelined workloads, and vice versa. Examples from prior cycles: `bench_m4t_tier2_perf` showed "no LTO benefit" while `bench_m4t_lto`'s pipelined variant showed 3× LTO speedup for the same target function; `m4t_mtfp_ternary_matmul_bt`'s vmlal vs scalar speedup ranged 4.2× to 17.6× across 5 BATCHED shapes (single-shape claim would have been misleading by 4×). Name the workload shape when reporting kernel timings; sweep at least 3-5 shapes within a claimed regime; report numbers as a range, not a point.

  The discipline rule at every layer: **the N (or scope) of the evidence must match the N (or scope) of the claim**. Each gap is a methodology debt that compounds across documents. Audit the debt at red-team time.
- [ ] **Substrate-novelty audit.** Set by the P0 remediation plan (`docs/REMEDIATION_PLAN_P0.md`). For every new measurement, every new claim, every new primitive: **does this work USE the substrate's distinct capabilities, or just live ON the substrate?** Name the only-base-3-can-do property being exercised. If the work would produce identical results on a base-2 substrate (with appropriate quantization), it's not substrate-claim work — it's correctness work. Both are valuable; only the former is substrate-claim evidence. The five prior rules catch *correctness drift*; this one catches *capability drift* — the failure mode that produced the rebuild this discipline is correcting.
- [ ] **Throughput microbench discipline (added 2026-05-04 + 2026-05-05).** Compiler optimizers will silently invalidate naive throughput measurements by constant-folding (e.g., `acc += K * (a*b)` becomes `add+branch` per iter when inputs are loop-invariant). Three patterns observed across V4-residual-3 LTO bench and ternary_mac_routing T-G1, both required iterative defenses. Apply ALL of these before trusting a throughput number:
  - **Disasm verification.** Inspect the binary (`otool -tv`) to confirm the inner loop emits the target instructions, not just `add+branch`. A bench that doesn't emit the op being measured is measuring loop control.
  - **Inputs from non-constant memory** (heap pool with run-time-derived addressing OR pid-derived seed values), not file-scope constants the compiler can prove static.
  - **Distinct inputs per call within a loop iteration**, not just per-iteration. A loop calling `vmlal(acc, a, b)` 8 times with the same `(a, b)` will be factored to `acc += 8*(a*b)` → one mul.
  - **`__attribute__((noinline))` on the bench function** so caller-side constants don't propagate.
  - **Min-of-N sampling** (typically N=5) per measurement to reduce thermal/scheduling noise.
  - **Workload-shape declared in output**, per the scope-match rule above.
  - **Report a range, not a point**, when the regime admits shape variation (R-G3 above).

This checklist is not a hook (the harness can't run it for you). It's a discipline reminder: code red-teams catch kernel issues; documentation red-teams catch *ensemble* drift; measurement red-teams catch *methodology* drift; kernel-use audits catch substrate-claim drift; scope-match audits catch claim-overreach; **substrate-novelty audits catch capability drift — the meta-failure where competent work fails to be substrate-claim work**. Six rules; the sixth subsumes the others under "is this work the substrate-claim, or is it adjacent to it?"
