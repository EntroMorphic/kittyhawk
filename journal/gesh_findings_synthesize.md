---
cycle: gesh_findings
phase: SYNTHESIZE
date: 2026-05-02
scope: commit to the next-cycle scope, with measurement gates, based on what RAW/NODES/REFLECT surfaced
companions: gesh_findings_{raw,nodes,reflect}.md
status: build commitment
---

# SYNTHESIZE — gesh_findings

REFLECT surfaced three load-bearing facts about the Phase A.2 findings:

1. The synthetic benchmark's structure makes mechanism comparisons weakly informative.
2. The strongest finding (C2: random > identity) has the weakest mechanism story.
3. Phase A.2's findings *don't address the substrate-claim*; that's a feature of the scope, but it makes "more synthetic sweeping" the wrong next move.

This phase commits to next-cycle work that takes those facts seriously.

## Next-cycle scope

**Cycle name:** `gesh_phase_b_probe`.

**Primary commitment:** a real-data probe that puts Gesh's forward + lattice-update path on a benchmark with structure the synthetic doesn't have. Per `project_benchmark_pivot`, Go position evaluation is the primary substrate-claim benchmark, gated by a half-day probe. Image canon (MNIST/CIFAR) is the regression-guard.

**Probe choice:** start with **image canon (MNIST first)** because:
- The pixel-trit-quantization pipeline already exists in the prior-cycle archive (`01MAY26_archived/`) — substrate-legal, no random projections, no dense resolvers. Lift forward.
- Gesh's forward + bank + lattice-update transfers directly: project ternary pixels → top-k Hamming → class vote. Same surface as the synthetic, just a different input distribution.
- The Phase A pipeline can be exercised end-to-end on real data in days, not weeks.
- Per the LSH architecture memory: prior-cycle best (mnist_routed_bucket_multi M=32 SUM at 97.24%/1.92ms) sets a regression-guard threshold. Below that, Gesh has clearly regressed; above or near, it's at parity. Above is interesting.

**Why not Go positions first:** the Go pipeline doesn't exist yet. Image canon has the substrate (ternary pixel quantization, packed-trit handling) already proven in the archived code. Image canon → exercise the consumer pipeline on real data → if Gesh sits near 97% on MNIST without surgery, escalate to Go for the substrate-claim measurement; if Gesh stalls on MNIST, fix the consumer pipeline before staging Go work.

**Secondary commitment (parallel, cheap):** the H1 mechanism test for "implicit denoising via random ternary projection." ~20 lines added to a new `gesh/bench/denoise_probe.c`. Either upgrades C2 to a demonstrated mechanism or surfaces that the +7pp gap has a different (currently unknown) cause.

## Pre-committed gates

These are decisions made now, before the data lands, so the cycle's interpretation can't drift to fit whatever number shows up.

### Gate 1 — image canon parity
**Measurement:** Gesh forward + lattice-update on MNIST canonical split (60k train, 10k test), ternary pixel quantization (no random projections, no learned features beyond the lattice-update on R). Single sig_dim choice; multi-seed (≥3 seeds).

**Pass:** trained Gesh ≥ 95% on MNIST, beating untrained random ≥ +2pp on average across seeds.
- *Rationale:* prior-cycle best with the comparable architecture was 97.24%. 95% is "in the same building." +2pp gain over random is the lower bound of the synthetic's compression-regime gain; if the regime transfers at all, +2pp should be reachable.

**Fail:** trained Gesh < 90% on MNIST, OR trained ≤ random within seed noise.
- *Action:* the consumer pipeline does not transfer. Loop back to NODES — what about the synthetic was over-fit to. Specifically check: bank construction may need refresh schedules tuned to the real-data scale, top_k may need to grow with class count, projection budget may need re-thinking when D = 784 (MNIST pixels) instead of 64.

**Inconclusive zone (90–95%, gain in {0, +2pp}):** the pipeline is loose but not broken. Continue; document that this is between "transfer" and "fail." Don't escalate to Go until inconclusive resolves to pass.

### Gate 2 — H1 mechanism test
**Measurement:** sample N random ternary R matrices at sig_dim=64; for each output dim *j* of each R, score `informative_weight_j = sum_{i ∈ informative_dims} |R[j][i]|` and `noise_weight_j = sum_{i ∈ noise_dims} |R[j][i]|`; compute per-output-dim contribution to inter-class Hamming separability (e.g., variance of class-mean tile values); regress the contribution against `(informative_weight_j − noise_weight_j)`.

**Pass (H1 supported):** positive correlation, p < 0.01 across N ≥ 100 random R samples.
- *Action:* update `sweep_dims_results.md` § Hypotheses to mark H1 demonstrated. C2's story holds; the substrate-claim narrative has one more honest piece.

**Fail (H1 falsified):** zero or negative correlation.
- *Action:* C2 stands as observation without mechanism. Update docs to remove the "implicit denoising" framing. The +7pp gap becomes "robustly observed, mechanism unknown — open question for next cycle."

**Inconclusive (weak positive correlation, p ∈ [0.01, 0.1]):** mechanism partially supported but not load-bearing. Document as such; don't cite as a finding.

### Gate 3 — multi-seed methodology audit on the new probe
**Measurement:** before any Phase B claim cites a benchmark number, the measurement must:
- Run ≥ 3 seeds (per `CONTRIBUTING.md`'s Phase A.2 rule).
- Report mean ± stddev.
- Distinguish hypotheses from findings explicitly.
- For real-data benchmarks: if the dataset has a single canonical realization (MNIST does), document that "multi-seed" varies (init, train) only and doesn't capture dataset variance.

**Failing this gate is a discipline violation, not a measurement failure.** It means the cycle hasn't internalized A4.

## What the cycle is NOT committing to

- **Adding more sig_dims to the synthetic sweep.** Beyond 1024 buys nothing. REFLECT covered this.
- **Running more lattice-update variants on the synthetic.** Same reason.
- **Measuring against a base-2 baseline before the consumer pipeline transfers.** Wrong order — the substrate-claim comparison happens once Gesh has shown it can run on real data. Comparing a synthetic-only mechanism against PyTorch attention is comparing the wrong things.
- **Building a new training algorithm.** The lattice-update mechanism works on the synthetic. Whether it works on MNIST is the question the cycle is testing. Don't change the algorithm before the test runs.

## Surface area expectations

- **No new substrate primitives expected.** MNIST is large but the kernels in `libm4t` already cover ternary popcount, distance batching, packed-trit operations, and route_apply. If a primitive is needed, it earns its place via the Phase B probe (per principle 5: named consumer demand suffices).
- **One new bench tool:** `gesh/bench/mnist_probe.c` (Phase B Gate 1) and `gesh/bench/denoise_probe.c` (Phase B Gate 2). Possibly a shared `gesh/bench/image_canon.{h,c}` for the canonical pixel-trit-quantization pipeline lifted from the archive.
- **Possible doc additions:** `gesh/docs/phase_b_gate1_results.md` and `gesh/docs/phase_b_gate2_results.md`. Keep them under the same hypothesis/finding discipline as `sweep_dims_results.md`.
- **No spec changes expected.** If the substrate has to be amended, that's a separate journal cycle (per principle 7).

## Open questions surfacing for the cycle

### Q1 — Does intra-epoch refresh transfer to MNIST scale?
The synthetic benchmark has n_train=2000; MNIST has 60k. The current `bank_refresh_every` and `batch_refresh_every` defaults (set by sweep_dims) scale to "4 refreshes per epoch." Whether that's the right cadence at 30× the data is empirical. Phase B Gate 1 should sweep over a small range of refresh cadences if the default underperforms.

### Q2 — Does sig_dim selection generalize across input dim?
On the synthetic (D=64), the compression sweet spot is sig_dim ≈ K = 16. MNIST pixels are D = 784. K is harder to define for real data — there's no clean "informative count" — but a heuristic might be "sig_dim such that compression ratio matches the synthetic's K/D ≈ 0.25." That would suggest sig_dim ≈ 196 for MNIST. Test 64, 128, 196, 256, 512 in Gate 1.

### Q3 — Does the +7pp identity-vs-random gap appear on MNIST?
Gate 1 should include an "identity" reference: take ternary-quantized pixels directly, build class-mean banks, classify. If Gesh forward (with random R) at the same dim hits +7pp over identity on MNIST, the C2 finding transfers. If it doesn't, the finding was synthetic-specific.

### Q4 — What's a fair PyTorch baseline?
Eventually (Phase B+ once parity is shown). Not in this cycle. Note for future: a fair baseline for the substrate-claim has to use comparable training compute, comparable model capacity (parameter count or its ternary equivalent), and identical evaluation protocol. The substrate-claim is "base-3 routing matches base-2 attention at comparable scale," not "base-3 wins a fight base-2 wasn't trained for."

## Build sequencing

1. Lift `image_canon.{h,c}` (ternary pixel quantization) from `01MAY26_archived/` into `gesh/bench/`. Adapt to current `m4t_types.h` and `m4t_trit_pack` surface. Substrate-legal pass: no random projections, no dense paths.
2. Build `mnist_probe.c` (Gate 1). Use existing `gesh_bank_t`, `gesh_forward_classify`, `gesh_train_lattice_update`. Multi-seed harness mirrors `sweep_dims.c`.
3. Run Gate 1 on a few sig_dim choices. Decide pass / fail / inconclusive.
4. **In parallel** (since the harness is independent): build `denoise_probe.c` (Gate 2). Run on the synthetic.
5. Document both in journal/ and gesh/docs/. Update CHANGELOG.

Order matters because: (1) gates the rest. If Gate 1 fails outright, the cycle changes — we're debugging consumer-pipeline transfer before any further Gesh work.

## Dependencies on prior memory and discipline

- **`feedback_no_synthetic`:** scoping respects this. Real-data probe is the primary commitment; synthetic mechanism test is secondary.
- **`feedback_no_random_projections`:** ternary pixel quantization, not random projection of pixels. The R matrix in Gesh is over ternary pixels, not over real-valued pixels — the canonical pipeline.
- **`feedback_no_random_weights`:** R initialization is random ternary; this is the *baseline being compared against* (the random-R variant of the sweep). The trained variant updates R via lattice descent. Random init persists in the trained variant only as a starting point.
- **`feedback_ternary_supports_lsh`:** Gesh forward is the LSH filter-ranker structure (random R is the filter; class-mean bank is the ranker). Lattice update learns the filter. Doesn't replace the architecture; refines a piece.
- **`project_benchmark_pivot`:** image canon is the regression guard; Go is the substrate-claim primary. Cycle starts with the regression guard for pipeline-transfer reasons; if it passes, Go becomes the next cycle's commitment.

## What success looks like

This cycle ships if at the end of it:
- Gate 1 has a verdict (pass / fail / inconclusive) on documented multi-seed measurements.
- Gate 2 has a verdict (H1 supported / falsified / inconclusive) on documented samples.
- The findings docs and CHANGELOG are updated honestly, with hypothesis/finding distinctions and conditions on every cited number.
- A clear next-cycle commitment exists: either "Phase B substrate-claim measurement on Go" (if Gate 1 passes), or "consumer pipeline debug" (if Gate 1 fails), or "more probing" (if inconclusive).

This cycle is a failure if it leaves these in motion: gate verdicts, doc currency, next-cycle scope. Half-finishing is worse than not starting.
