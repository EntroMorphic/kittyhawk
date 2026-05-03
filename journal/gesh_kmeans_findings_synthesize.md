---
cycle: gesh_kmeans_findings
phase: SYNTHESIZE
date: 2026-05-02
scope: commit to next-cycle scope with measurement gates
companions: gesh_kmeans_findings_{raw,nodes,reflect}.md
status: build commitment
---

# SYNTHESIZE — gesh_kmeans_findings

REFLECT surfaced three load-bearing observations:

1. **Phase A's "lattice update earns its place" claim has narrower scope than the journals currently state.** It earns its place at *low bank capacity*; at higher bank capacity, it can hurt.
2. **Bank-vs-training is non-additive and possibly antagonistic.** The dominant lever is the bank.
3. **Single-seed risk on C3 is large.** Per our own meta-rule, the "training hurts" claim is currently OUTCOME, not FINDING.

This cycle commits to next-cycle work that takes those facts seriously, with **pre-committed gates** so the interpretation can't drift.

## Next-cycle scope

**Cycle name:** `gesh_kmeans_validation`.

**Three measurement workstreams, in priority order:**

### W1 — Multi-seed validation of C3 (the load-bearing finding)

Re-run trained-R + k-means at the same config (sig_dim=64, n_train=60K, k=8, T=80, 250K flip budget, 64 epochs, no early stop) with ≥ 3 seeds. Same comparison cells:

- Random R + k-means (3 seeds): mean ± stddev
- Trained R + k-means (3 seeds): mean ± stddev
- Per-seed paired gain: mean ± paired stddev, 95% CI

**Pre-committed gate:**
- **CONFIRMED**: paired gain mean is negative AND its 95% upper bound is < 0 → C3 promotes to FINDING. Doc-currency cascade triggered.
- **FALSIFIED**: paired gain CI includes 0 or is positive → C3 was a single-seed artifact. Re-evaluate.
- **INCONCLUSIVE**: paired gain CI straddles 0 (e.g., −1pp ± 2pp) → run with more seeds (5–10) or accept C3 as "directionally negative, magnitude uncertain."

### W2 — Mechanism test for whichever of H1/H2/H3 is cheapest

The cheapest probe with the clearest signal is **H2 (frozen bank during training)**: run trained-R with `bank_refresh_every` set so high it never refreshes mid-run. Same other config as W1. Single decisive run (or multi-seed if W1 already established the multi-seed harness).

**Pre-committed reading:**
- If frozen-bank trained accuracy ≥ refresh-bank trained accuracy: H2 supported. K-means refresh during training was the destabilizer. **Practical implication: don't refresh k-means during lattice-update training.**
- If frozen-bank trained accuracy ≈ refresh-bank trained accuracy: H2 ruled out. The destabilizer is something else (H1 or H3 territory).
- If frozen-bank trained accuracy < refresh-bank trained accuracy: H2 inverted. K-means refresh was actually helping; freezing made it worse. Surprising; would need a different mechanism story.

If time / interest permits in the same cycle:
- **H1 (budget sweep):** flip budget ∈ {25K, 50K, 100K, 250K}. Single seed each (cheap). If smaller budgets give larger trained accuracy, H1 (overtraining hypothesis) supported.
- **H3 (batch-size sweep):** batch_size ∈ {128, 512, 2048}. Single seed each. If larger batches narrow the train/test gap, H3 supported.

### W3 — Doc-currency scope-qualifier pass on Phase A claims

If W1 confirms C3 (and even if it doesn't, the OUTCOME is real), the existing journals carry statements that no longer accurately scope:

- `journal/gesh_design_closeout.md`: "the lattice IS the geometry; training walks it directly" — true, but should note that "walking it" can hurt at high bank capacity.
- `journal/gesh_findings_synthesize.md`: "Path A: richer consumer + lattice update" — should narrow to "richer consumer; lattice update is optional and may need a different objective at higher T."
- `journal/gesh_phase_b_probe_closeout.md` revision banner: "C1 (lattice update earns +4–8pp in compression regime)" → "C1 (transfers at low bank capacity; inverts at high bank capacity, see kmeans_findings)"
- `gesh/docs/sweep_dims_results.md` § Finding 1: keep as-is for the synthetic regime, but cross-reference the MNIST inversion.
- `CHANGELOG.md`: add a top-level note about the C3 outcome.

**Methodology rule applied:** all updated cite-sites must explicitly state the regime where the claim holds. No more "lattice update earns +8pp" without "(at single-prototype bank)" or equivalent qualifier.

## What this cycle is NOT committing to

- **Multi-table LSH composition.** That's a different consumer architecture, properly Phase B+ scope. Doing it here would muddy the bank-vs-training measurement we want.
- **Switching to a different training objective** (e.g., inter-class separation). Plausible follow-up but not in this cycle. First we want to know whether the *current* mechanism's inversion is real (multi-seed) and what causes it (mechanism test).
- **Substrate purification of the new k-means code.** Already queued as task #14; pursued separately.
- **Tuning the existing config to make trained-R + k-means score higher.** Tuning until trained beats random would defeat the diagnostic value of C3; we want the verdict, not a palatable number.

## Surface area expectations

- **No new substrate primitives needed** for W1/W2/W3.
- **Code changes:** none for W1 (use the existing `mnist_kmeans_trained.c` probe with multi-seed loop). Small change for W2 (set `cfg.bank_refresh_every = INT_MAX` or similar to disable mid-run refresh; existing knob handles this).
- **Doc updates:** scope-qualifier pass across ~5 documents. Mechanical once W1's verdict is in.

## Open questions surfacing for the cycle

### Q1 — What's the right training objective for multi-prototype banks?
The current loss is "per-batch classification error against the current bank." That's appropriate for the bank-shape it was designed for (single-prototype, deterministic class-mean). With multi-prototype k-means, the bank is a multimodal distribution per class; training R against it is harder.

Plausible alternatives:
- **Inter-class signature distance:** maximize Hamming distance between class-mean projected signatures. No bank refresh needed; one-shot.
- **Per-cluster classification accuracy:** require each cluster's training samples to dominate its assigned tile. More direct loss for k-means.
- **Contrastive Hamming:** pull within-class samples closer, push between-class farther.

These are all substrate-legal (integer Hamming, deterministic). None are implemented. **Out of scope for this cycle's validation work; flagged for a future cycle if W2's mechanism test rules out simpler explanations.**

### Q2 — Does Path A still mean what we thought it meant?
The Phase B closeout's Path A was "richer consumer (multi-table LSH composition) + lattice update." The data here suggests the "lattice update" part of Path A might need to be dropped or replaced. **Path A should be reconsidered post-W1**: if training inverts at high capacity, Path A reduces to "richer consumer, no training" — which we've already partially measured (70.1% at k=32 random R). Multi-table LSH on top might be the only remaining lever.

### Q3 — What does the substrate-claim story look like if lattice update is contraindicated at scale?
The substrate-claim was "ternary projections + lattice update + bank → matches base-2 attention." If lattice update drops out of this story, the claim becomes "ternary projections + multi-prototype bank + multi-table composition → matches base-2 attention." That's still a substrate-claim, but a simpler one — closer to what the archive's `mnist_routed_bucket_multi M=32 SUM` already demonstrated.

The *novelty* of the substrate-claim shifts from "trainable ternary projections" to "ternary substrate enables expressive multi-table consumers." The narrative needs a structural rewrite, not just scope-qualifiers.

This is too big to commit to in this cycle. **Flagged for the next-cycle SYNTHESIZE if W1 confirms C3.**

## Pre-committed methodology

- **W1's gate is paired-difference 95% CI**, not separate-samples. Per the SDOT-finding3 red-team's H6 lesson.
- **Multi-config qualifier:** if W1 produces a CONFIRMED verdict at sig_dim=64 only, the claim is *"training hurts at sig_dim=64 with k=8 k-means."* Generalizing to "training hurts at high T regardless of sig_dim" requires multi-config measurement (multiple sig_dims). That's deferred to a future cycle but explicitly named here so we don't generalize prematurely.
- **All measurements use permille precision** (the C2 fix from SDOT-finding3 red-team).
- **Random-R baseline runs deterministically; identity baseline runs deterministically** (M3 cross-check).

## What success looks like

This cycle ships if:
- W1 has a verdict on C3 with paired-CI methodology.
- W2 has a verdict on H2 (frozen bank vs refresh bank).
- The doc-currency cascade (W3) is done if W1 confirms C3, or deferred with a clear "pending W1 verdict" tag if W1 falsifies/inconclusives.
- Subsequent-cycle scope is named (likely Q1 or Q2 territory).

This cycle is a failure if it ends with C3 still single-seed, or with the "training hurts" framing in journal docs but no validation behind it.
