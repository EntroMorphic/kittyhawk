---
cycle: gesh_zero_signal_design (P0-1)
phase: SYNTHESIZE
date: 2026-05-02
scope: commit to which primitives get built, in what order, with which verification gates and substrate-spec amendments
companions: gesh_zero_signal_design_{raw,nodes,reflect}.md
status: build commitment
---

# SYNTHESIZE — gesh_zero_signal_design

REFLECT surfaced four load-bearing constraints:

1. **P1 and P6 must ship together.** The wildcard distance kernel makes operational sense only paired with a wildcard-by-construction bank.
2. **P3's speedup is smaller than initially predicted** (~1.3–1.5× at MNIST density, not 2.5×). Worth doing but not transformative.
3. **P2 and P5 are auxiliaries** that need consumer integration to demonstrate value.
4. **MNIST is the wrong primary benchmark** for substrate-novelty; a synthetic with explicit don't-care structure is needed.

## Build commitment

**P0-1's deliverable: P1 (`m4t_route_wildcard_dist`) + P6 (`gesh_bank_build_class_wildcard`) + a wildcard-aware forward variant + a benchmark that exhibits don't-care structure.**

Rationale: P1+P6 are the substrate-novelty story for zero-as-wildcard. They demonstrate the third state operationally. The other primitives (P2, P3, P5) are deferred to follow-on work; ship if cheap once P1+P6 land, otherwise defer to subsequent P0 cycles where they better fit (P3 → P0-4 multi-stage where activations are naturally sparser).

Specifically committing to:

### Substrate primitive 1: `m4t_route_wildcard_dist`
**Header:** `m4t/src/m4t_route.h`
**Signature:**
```c
/* Wildcard-Hamming distance: tile-zero is treated as match-anything
 * (wildcard / don't-care). Distinct from m4t_popcount_dist which costs
 * (q=±1, t=0) at 1; this kernel costs it at 0.
 *
 * Per-position cost table:
 *   (q=±1, t=0)  → 0   (wildcard match — KEY DISTINCTION)
 *   (q=0,  t=0)  → 0   (mutual abstention)
 *   (q=0,  t=±1) → 1   (query abstains, tile asserts)
 *   (q=+1, t=+1) → 0   (full match)
 *   (q=-1, t=-1) → 0   (full match)
 *   (q=±1, t=∓1) → 2   (full mismatch)
 *
 * Sanctioned input class:
 *   Tile signatures constructed by gesh_bank_build_class_wildcard or
 *   any constructor that emits zeros DELIBERATELY (sparse-coded,
 *   feature-pruned). NOT for use against class-mean banks where zeros
 *   emerge from sample-cancellation ties — there the wildcard
 *   interpretation over-promotes ambiguous matches. */
int32_t m4t_route_wildcard_dist(
    const uint8_t* query_packed,
    const uint8_t* tile_packed,
    const uint8_t* mask,
    int sig_dim);
```
**Implementation:** XOR-popcount on packed bytes with an additional pre-step that zeroes out mismatch positions where tile-trit == 0. Cost-table is achievable via two popcounts: one over standard XOR, minus one over (XOR & tile_zero_mask).
**Tests:** property test against an explicit reference loop covering all 9 (q,t) combinations × shapes.

### Substrate primitive 2: `gesh_bank_build_class_wildcard`
**Header:** `gesh/src/gesh_bank.h`
**Signature:**
```c
/* Build a class-mean bank with deliberate wildcards at low-SNR
 * positions. Algorithm:
 *   1. For each class c:
 *      a. Compute per-dim signed sum (as in class_mean).
 *      b. Compute per-dim sample count.
 *      c. Compute per-dim signal magnitude: |sum| / count.
 *      d. Compute per-dim noise: stddev across samples (or proxy:
 *         count of sign-flips).
 *      e. SNR = signal / noise.
 *      f. Threshold: if SNR < snr_threshold, force trit to 0
 *         (wildcard). Else sign(sum).
 *   2. Pack tile.
 *
 * The wildcards are DELIBERATE — positions known to be class-c-
 * irrelevant. Pair with m4t_route_wildcard_dist for substrate-native
 * decision-rule routing.
 *
 * Preconditions: bank->n_tiles == n_classes; bank->sig_dim > 0. */
void gesh_bank_build_class_wildcard(
    gesh_bank_t* bank,
    const m4t_trit_t* samples,
    const int* labels,
    int n_samples,
    int n_classes,
    int snr_threshold_permille);
```
**Implementation:** integer arithmetic only; SNR computed as `|sum| × 1000 / max(count_sign_changes, 1)` with permille threshold. Substrate-discipline-clean (no float, no random).
**Tests:** verify wildcard count scales with snr_threshold; verify output is bit-identical to class_mean when threshold is 0 (no positions zeroed).

### Consumer integration: `gesh_forward_classify_wildcard`
**Header:** `gesh/src/gesh_forward.h`
**Signature:**
```c
/* Forward classification using m4t_route_wildcard_dist for tile
 * matching. Same shape as gesh_forward_classify but the per-tile
 * distance kernel is the wildcard variant.
 *
 * Caller must use a bank constructed via gesh_bank_build_class_wildcard
 * (or any other deliberate-wildcard constructor) for the routing
 * semantics to be coherent. */
int gesh_forward_classify_wildcard(
    int* out_predictions,
    const m4t_trit_t* queries,
    int n_queries,
    const gesh_bank_t* bank,
    const gesh_projection_t* proj,
    int top_k);
```
**Implementation:** copy-edit of `gesh_forward_classify` swapping `m4t_popcount_dist` → `m4t_route_wildcard_dist`. Other steps (projection, top-k, vote) unchanged.

### Verification benchmark: `synth_wildcard.c`
A new bench under `gesh/bench/` that constructs a synthetic with explicit don't-care structure:
- D = 64 input dims.
- K = 16 always-informative dims (every class has a ±1 prototype here).
- M = 16 sometimes-informative dims (each class has either a ±1 prototype OR zero, randomly chosen at proto-gen time, drawn fresh per class).
- N = 32 never-informative dims (uniform random ternary noise; should be wildcard in any class signature).
- Class signatures by construction: K opinionated + variable M + N noise.

The data exhibits the don't-care structure substrate-novelty needs: 32 dims that the bank should learn to wildcard out, plus 16 per-class-irrelevant dims that should be wildcard in some classes' signatures and ±1 in others.

## Pre-committed verification gates

### Gate 1 — Wildcard semantics is operationally distinct
**Measurement:** wildcard-dist vs current Hamming on the synth_wildcard benchmark, both with the wildcard bank constructor.
**PASS:** wildcard-dist routing accuracy ≥ current Hamming routing accuracy + 5pp on synth_wildcard. The 5pp threshold is calibrated so that random noise (single-seed) doesn't trigger pass.
**FAIL:** wildcard-dist accuracy < current Hamming + 1pp. Zero-as-wildcard semantics did not provide measurable benefit on a benchmark designed to expose it. **Substrate-novelty for this primitive is not demonstrated; revisit framing.**
**INCONCLUSIVE:** in [+1pp, +5pp] range. Run multi-seed; if mean still inconclusive, the primitive is real but small.

### Gate 2 — Wildcard cost is no higher than current Hamming
**Measurement:** wall-clock per query for both kernels at sig_dim=64, T=10 bank.
**PASS:** wildcard-dist runtime ≤ 1.2× current Hamming runtime. (The kernel does extra work — zero-position detection — so some overhead is expected; 20% is the budget.)
**FAIL:** wildcard-dist runtime > 1.2× current Hamming. The substrate's "free third state" claim doesn't hold operationally; the wildcard semantics costs more than what base-2 with masks would pay. **Implementation needs optimization.**

### Gate 3 — Substrate-novelty audit
**Measurement:** comparison against base-2 with explicit mask bits.
**PASS:** wildcard-dist + wildcard bank produces same routing decisions as base-2 (mask + Hamming over ±1 trits) at the same accuracy, but the substrate version uses **half the storage** (no mask bitvector; trits encode three states in 2 bits each) and **the same compute** (no separate mask popcount).
**FAIL:** the storage advantage doesn't materialize at this consumer's scale. (E.g., if mask compression in base-2 produces equally compact encodings.) Then the substrate-novelty claim is academic, not operational.

### Gate 4 — No regression on MNIST baselines
**Measurement:** wildcard-dist + wildcard bank on MNIST vs current single-prototype + Hamming on MNIST.
**PASS:** wildcard variant accuracy within ±2pp of current. Not better; just not regressing on the regression-guard.
**FAIL:** wildcard variant > 2pp below current on MNIST. The wildcard semantics actively hurts on a base-2-friendly benchmark — that's a real regression signal.
**INCONCLUSIVE acceptable:** within ±2pp is fine. MNIST isn't where the substrate-novelty claim wins; the gain is on synth_wildcard.

## Substrate-spec amendments (per principle 7)

Two amendments to `m4t/docs/M4T_SUBSTRATE.md`:

### §X.Y — Wildcard semantics for zero-state in routing tiles
New subsection documenting:
- The wildcard interpretation of structural zero in deliberately-constructed tile signatures.
- Distinction from emergent zero in class-mean tiles (where zero means "ties cancel," not "wildcard").
- Sanctioned input class for `m4t_route_wildcard_dist`: tiles from constructors that emit zeros deliberately.
- Cost-table specification (matching the kernel).

### §X.Z — Bank constructor input-class extension
Document `gesh_bank_build_class_wildcard` as a sanctioned bank constructor; specify the SNR-threshold semantics; cross-reference the wildcard-dist kernel.

## Sequencing

1. **Substrate-spec amendments.** Write the §X.Y and §X.Z text. Commit.
2. **`m4t_route_wildcard_dist` kernel.** Implementation + property tests. Verify against reference loop. Commit.
3. **`gesh_bank_build_class_wildcard` constructor.** Implementation + tests. Verify wildcard count scales with threshold; verify class_mean equivalence at threshold=0. Commit.
4. **`gesh_forward_classify_wildcard` consumer.** Implementation + tests. Commit.
5. **`synth_wildcard.c` benchmark.** Implementation + small-scale verify. Commit.
6. **Run Gates 1–4.** Document results.
7. **Closeout** (this cycle's CLOSEOUT updated with verdict).
8. **Red-team pass** before proceeding to P0-2.

## What this cycle is NOT committing to

- **P3 (skip-zero matmul).** Deferred. The 1.3–1.5× speedup at MNIST scale is real but smaller than the substrate-novelty bar. Better-suited to P0-4 where multi-stage activations have higher zero density by construction.
- **P2 (zero_alignment) or P5 (zero_count).** Auxiliary; ship later if a consumer needs them.
- **MNIST as the primary verification benchmark.** Used as a regression-guard (Gate 4) only.
- **Multi-stage zero propagation (P4).** Properly P0-4 territory.
- **Training with the wildcard bank.** Lattice-update training over a wildcard bank is its own design question (does the lattice update ever produce wildcards? does it preserve them?); deferred to a follow-on cycle.

## Open questions surfacing for follow-on cycles

### Q1 — Does lattice-update training preserve wildcard tiles?
The current `gesh_train_lattice_update` flips R trits to minimize batch error. With a wildcard bank, the bank refresh would *re-derive* wildcards from the new R + samples. Whether wildcards stabilize or oscillate is unknown. **Test with: run training over a wildcard bank, log wildcard-count per epoch, verify it stabilizes.**

### Q2 — Should `gesh_init_random_projection_balanced` (with zeros) be the default for wildcard-bank work?
A balanced ternary R has more output-side zeros (some matmul outputs threshold to zero). This may interact with wildcard banks naturally. The current `gesh_init_random_projection` writes only ±1; using balanced init might be substrate-naturally aligned with wildcard semantics. **Test with: balanced init vs ±1-only init on synth_wildcard; compare.**

### Q3 — Do other distance variants (e.g., zero-disagreement count) carry routing signal?
P2 (zero_alignment) was deferred but the inverse (zero-disagreement: q=0 xor t=0) might carry asymmetric signal: high zero-disagreement means "one signature has opinions the other doesn't." **Defer; revisit if Q1 or Q2 surface a need.**

## Methodology checks applied

- **Multi-seed:** Gate 1 specifies `+5pp threshold to avoid single-seed false-positive pass`. If single-seed lands in [+1pp, +5pp] range, multi-seed validation is required.
- **Multi-config:** Gates 1, 2, 4 specify configs explicitly (sig_dim=64, T=10 for current Hamming comparison). Generalization to other configs would be a follow-up cycle if Gates pass.
- **Substrate-novelty audit:** Gate 3 is the explicit substrate-novelty gate.
- **Match scope of evidence to scope of claim:** the substrate-claim from this P0 is "zero-as-wildcard provides operational distinction over base-2 with mask bits." Gates 1+3 jointly address the claim's scope; either alone is insufficient.

## What success looks like

P0-1 ships if:
- Gates 1, 2, 4 produce verdicts (PASS, FAIL, or INCONCLUSIVE-acceptable).
- Gate 3 (substrate-novelty audit) PASSES — the substrate-claim story has its first piece of substrate-distinct measurement.
- The journal/doc trail records the wildcard semantics as a substrate primitive, not just a Hamming variant.
- P0-2 (exponent signal) is unblocked and can begin its design cycle.

P0-1 fails if:
- Gate 3 FAILS — the wildcard primitives don't demonstrate base-3-only capability.
- Or all of Gate 1 FAILs and Gate 4 FAILs simultaneously — the wildcard semantics regresses on the substrate-friendly benchmark AND fails on the regression guard. Then the framing was wrong; back to RAW.
