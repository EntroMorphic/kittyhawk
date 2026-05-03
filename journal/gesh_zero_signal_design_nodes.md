---
cycle: gesh_zero_signal_design (P0-1)
phase: NODES
date: 2026-05-02
scope: extract discrete units from RAW; map dependencies; tag by what's claim, what's hypothesis, what's anchor
companions: gesh_zero_signal_design_raw.md
status: structuring
---

# NODES — gesh_zero_signal_design

Discrete units. **C** = claim (data-supported or definitionally true). **H** = hypothesis (proposed; mechanism untested). **A** = anchor (fixed reference frame). **D** = dependency. **G** = gap (named substrate-claim deficit from RAW).

## Substrate-claim gaps from RAW (named for sequencing)

### G1 — Hamming distance treats zero symmetrically; can't distinguish "agree-on-opinion" from "agree-on-don't-care"
**Severity:** high. (0,0) and (+1,+1) cost zero in current Hamming; operationally distinct decision-rule semantics absent.

### G2 — No primitive interprets zero as wildcard
**Severity:** high. The simplest substrate-novel routing primitive (TCAM-shape match-with-don't-care) is missing.

### G3 — No primitive uses zero to skip per-cell computation
**Severity:** medium-high. Substrate-native sparsity benefit unrealized in matmul. ~60% of MNIST quantized trits are zero; SDOT pays 100% cost.

### G4 — No bank constructor produces zero-aware tiles by design
**Severity:** medium. Class-mean and k-means produce emergent zeros from sign-thresholding ties. No deliberate "this dim is class-c-irrelevant" prior.

### G5 — No primitive uses zero count as information-content signal
**Severity:** medium. Zero density distinguishes specialist signatures from general; routing could exploit it.

### G6 — No threshold-extract variant respects an input-side "skip these positions" mask
**Severity:** medium. Required for P0-4 multi-stage; not yet needed for P0-1 in isolation.

### G7 — No primitive distinguishes "zero in query" from "zero in tile"
**Severity:** medium. Query-zero means "I have no value here." Tile-zero means "this position is class-irrelevant" (in a wildcard bank). Currently symmetric.

## Candidate primitives (numbered for cross-reference)

### P1 — `m4t_route_wildcard_dist`
**Statement:** Hamming-style distance where tile-zero is wildcard (free match), query-zero is "abstain" (treated symmetrically; cost 1 against tile-±1).
**Signature (proposed):**
```c
int32_t m4t_route_wildcard_dist(
    const uint8_t* query_packed,      // packed trits
    const uint8_t* tile_packed,        // packed trits; zeros = wildcards
    const uint8_t* mask,
    int sig_dim);
```
**Per-position cost table:**
- (q=±1, t=0)  → 0  (wildcard match — DISTINCT FROM CURRENT HAMMING which costs 1)
- (q=0,  t=0)  → 0  (mutual abstention)
- (q=0,  t=±1) → 1  (query abstains, tile asserts; partial mismatch)
- (q=+1, t=+1) → 0  (full match)
- (q=+1, t=-1) → 2  (full mismatch)
- (q=-1, t=+1) → 2  (full mismatch)
- (q=-1, t=-1) → 0  (full match)
**Cures:** G1, G2, G7.
**Substrate-novelty test:** base-2 has no native wildcard state. To approximate, base-2 needs (a) twice the bit-width per position to encode {-1, 0, +1, mask} or (b) a separate mask bitvector. Both increase storage cost beyond the substrate's free third state.

### P2 — `m4t_route_zero_alignment`
**Statement:** Counts positions where both signatures equal zero. Returns this count as a separate signal alongside (or instead of) Hamming distance.
**Signature (proposed):**
```c
int32_t m4t_route_zero_alignment(
    const uint8_t* a_packed,
    const uint8_t* b_packed,
    const uint8_t* mask,
    int sig_dim);
```
**Returns:** count of positions where both packed codes are 0b00 (zero-state).
**Cures:** G1 partially (separates the agree-as-zero count from the agree-as-opinion count).
**Substrate-novelty test:** measures a base-3-only quantity. Base-2 has no analog.

### P3 — `m4t_mtfp4_sdot_matmul_bt_skip_zero_query`
**Statement:** Matmul variant that skips K-iterations where the activation trit is zero. Per-row: scan activations, build a list of nonzero positions, iterate only over those.
**Signature (proposed):**
```c
void m4t_mtfp4_sdot_matmul_bt_skip_zero_query(
    m4t_mtfp_t* Y,
    const m4t_trit_t* X,        // ternary activations; zeros are skipped
    const m4t_trit_t* W,        // ternary weights
    int M, int K, int N);
```
**Cures:** G3.
**Substrate-novelty test:** the speedup is proportional to the activation zero density. For MNIST at 60% zero density, expected ~2.5× speedup over dense SDOT. Base-2 has no zero state to skip natively; sparse base-2 architectures need explicit indexing structures (CSR, COO, etc.) which carry overhead the substrate-native variant doesn't.

### P4 — `m4t_threshold_extract_with_skip_mask`
**Statement:** Threshold-extract that takes an additional "skip mask" — bit-vector over output positions. For positions in the skip mask, output is forced to zero regardless of input magnitude.
**Cures:** G6.
**Substrate-novelty test:** required for multi-stage P0-4; in isolation a convenience primitive that doesn't yet exercise novel substrate capability. **Defer to P0-4.**

### P5 — `m4t_signature_zero_count`
**Statement:** Counts zero positions in a packed-trit signature. Returns the count as int.
**Signature (proposed):**
```c
int32_t m4t_signature_zero_count(
    const uint8_t* packed,
    const uint8_t* mask,
    int sig_dim);
```
**Cures:** G5.
**Substrate-novelty test:** measures a base-3-only quantity (zero density). The signal can drive routing decisions ("low zero count → confident query → standard routing; high zero count → uncertain query → fall-through").

### P6 — `gesh_bank_build_class_wildcard`
**Statement:** Bank constructor that produces tiles with deliberate zeros at "class-c-irrelevant" positions. Algorithm: for each class, compute per-dim signal-to-noise ratio; positions with SNR below threshold get zero (wildcard); above get sign-thresholded ±1.
**Cures:** G4.
**Substrate-novelty test:** the bank tiles use the wildcard state by design, not as emergent ties. Combined with P1 (`wildcard_dist`), this gives a bank-and-routing pair that exploits the third state operationally.

## Architectural anchors

### A1 — The substrate's three-state alphabet is the load-bearing distinction
Per the substrate spec and `m4t_types.h`: trits are {-1, 0, +1}, not {-1, +1}. The third state is what justifies "base-3" terminology and what differentiates from quantized base-2.

### A2 — The original GESH design's three Gs implicitly assumed three-state semantics
The "Geometric" G referenced manifold structure on the trit lattice. A two-state lattice (Z2^n) is the binary hypercube; a three-state lattice (Z3^n) has fundamentally different geometry — including isolated points (all zeros) and natural notions of "distance from informativeness."

### A3 — Existing kernels operate over packed trits with all three states represented
- `m4t_pack_trits_1d`/`m4t_unpack_trits_1d`: round-trip ternary, all three states preserved.
- `m4t_route_threshold_extract`: emits all three states per §18.
- `m4t_popcount_dist`: distinguishes states in cost table (though symmetrically).

The substrate has the *form*. What's missing is the *semantics* — primitives that distinguish zero operationally.

### A4 — Substrate-claim measurement requires base-2 comparison harness
For each new primitive, the substrate-novelty audit demands: would base-2 with appropriate quantization produce the same result? If yes, not substrate-claim work. We need an explicit base-2 baseline for at least one of P1/P3 to demonstrate substrate advantage in a measurement.

### A5 — All P1-P6 primitives are in libm4t scope (substrate-level), not libgesh scope (consumer-level)
This matters for principle 7 (substrate spec upstream of kernel design): each primitive needs a spec amendment in `M4T_SUBSTRATE.md` before implementation.

## Hypotheses

### H1 — `m4t_route_wildcard_dist` (P1) provides measurable accuracy benefit on benchmarks with class-irrelevant feature dimensions
**Predicts:** for a task where class signatures differ on a small subset of "informative" dims and don't-care on others, a bank constructed with deliberate wildcards (P6) + wildcard distance (P1) produces higher accuracy than the same task with current Hamming + class-mean bank.
**Mechanism test:** synthetic task with K=4 informative dims out of D=64; class signatures use trit values on informative dims, zeros on noise dims; compare wildcard-routing accuracy vs current routing.
**Falsifiable:** if wildcard accuracy ≤ current accuracy, the wildcard interpretation isn't the right substrate use.

### H2 — `skip_zero_query` matmul (P3) provides proportional speedup on high-zero-density inputs
**Predicts:** for MNIST quantized at density 0.60 (60% zeros), the skip-zero matmul runs at ~2.5× the throughput of dense SDOT (proportional to non-zero fraction).
**Mechanism test:** benchmark same matrix multiplication via SDOT and via skip-zero-query path; report wall-clock and speedup.
**Falsifiable:** if speedup < 1.5× at 60% zero density, the skip overhead negates the sparsity benefit; the primitive is not substrate-claim work.

### H3 — Zero-alignment count (P2) carries class-discriminative information that Hamming distance discards
**Predicts:** for a fixed bank, augmenting top-k tile selection with zero-alignment as a tiebreak (or weighted secondary signal) improves accuracy.
**Mechanism test:** modify forward pass to compute both Hamming and zero-alignment per (query, tile); pick top-k with combined score.
**Falsifiable:** if accuracy doesn't improve, zero-alignment is redundant with Hamming for our consumer.

### H4 — Wildcard banks (P6) reach higher effective capacity than k-means banks at the same tile count
**Predicts:** a P6-wildcard bank with T tiles covers a larger region of trit-signature space than a k-means bank with T tiles, because each wildcard tile represents a *region* (don't-care positions). Test: classification accuracy at fixed T should be higher with wildcards than k-means on tasks with class-relevant subsets.
**Mechanism test:** comparison on synthetic with controllable informative-dim count.
**Falsifiable:** if accuracy parity, wildcards aren't expanding effective capacity.

## Dependencies

- **P1 (wildcard_dist) depends on A3** (packed-trit format with three states). No new packing.
- **P2 (zero_alignment) depends on A3.** Standalone primitive; doesn't require P1.
- **P3 (skip_zero_query matmul) depends on the existing SDOT path** (`m4t_mtfp4_sdot_matmul_bt`); is a variant.
- **P4 (skip_mask threshold) is deferred** to P0-4.
- **P5 (zero_count) depends on A3** and is small.
- **P6 (wildcard bank) depends on P1** to be useful at runtime; standalone construction OK.

**Critical path for substrate-novelty demonstration:** P1 + P6 jointly demonstrate wildcard semantics end-to-end. P3 demonstrates substrate-native sparsity speedup. These are the two primary substrate-claim deliverables.

P2 and P5 are useful auxiliaries but not load-bearing for the substrate-claim. Defer or include based on SYNTHESIZE prioritization.

## Substrate-novelty audit applied to each primitive

| Primitive | Substrate-distinct capability used | Base-2 substitute cost |
|---|---|---|
| P1 wildcard_dist | Tile-zero as match-anything | Mask bit per position (2× storage) or multi-rule disjunction |
| P2 zero_alignment | Counting agree-as-zero | No native equivalent; would require separate "abstain" channel |
| P3 skip_zero_query | Matmul iteration skip on zero | Sparse matrix format (CSR/COO) with indexing overhead |
| P5 zero_count | Counting zero density | Separate mask bitvector + popcount |
| P6 wildcard bank | Bank tiles as decision rules with don't-cares | TCAM in hardware (specialized chip); learned dropout (probabilistic) |

All five exercise base-3-only capability. **None of them have a free base-2 substitute.**

## What's not in any node

- Specific syntactic API choices (parameter ordering, error handling). NODES is about semantic structure; specifics are SYNTHESIZE/code work.
- Choice of which subset to build first. SYNTHESIZE territory.
- Substrate-spec amendment text. SYNTHESIZE plus the spec amendment that follows.
- Test plan specifics. Comes after SYNTHESIZE picks the primitives.
- Comparison baselines on a specific benchmark. REFLECT will pressure-test what benchmark surface this needs.
