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

This checklist is not a hook (the harness can't run it for you). It's a discipline reminder: code red-teams catch kernel issues; documentation red-teams catch *ensemble* drift. Both are needed.
