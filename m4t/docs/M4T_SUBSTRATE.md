---
title: M4T Substrate Specification
status: ground-zero rebuild (2026-04-14)
supersedes: archive/m4t/docs/{M4T_CONTRACT,M4T_PIPELINE,M4T_BEYOND,M4T_REDTEAM,TRIT_LATTICE_LSH}.md
scope: the ternary-float compute substrate for Apple M-series (NEON + SDOT + TBL)
---

# M4T Substrate Specification

This is the canonical design document for the rebuild. Every decision here is traceable to an instruction-set primitive. Where hardware doesn't pin the answer, the decision is marked **OPEN** and named.

---

## 0. Status

The prior M4T implementation collapsed Multi-Trit Floating Point into "multi-trit fixed-point with a shared global scale" and then adopted a zero-float crusade downstream of that collapse. This spec resets the substrate to its original intent: **MTFP is a base-3 floating-point system**, not a fixed-point integer representation. Files built under the fixed-point reading live in `archive/`. Routing primitives survive intact (`m4t_trit_*`, `m4t_route`, `m4t_ternary_matmul`, `m4t_mtfp4`). The numeric core (`m4t_mtfp.*`) will be rewritten to this spec.

---

## 1. Thesis

1. **Routing is a first-class primitive over dense computation.** Dense matmul bends hardware into shape; routing rides hardware shapes that already exist (TBL, masked-VCNT, SDOT).
2. **Binary float is out. Ternary float is in.** The enemy was never floating-point representation — it was binary floating-point (IEEE-754), which is a shape foreign to ternary-native silicon paths. MTFP is floating-point, just base-3.
3. **MTFP maintains 100% accuracy on the normal path.** Base-3 mantissas are exact integers; exponents are exact integers; arithmetic either produces an exact result or *widens*. Rounding is never silent.
4. **Hardware shapes drive design.** When a decision admits multiple answers, the one that aligns with an instruction-set primitive wins.

---

## 2. Vocabulary: MTFP is Multi-Trit Floating Point

An **MTFP cell** is a fixed-width mantissa in trits (base-3 digits). A **block** of MTFP cells shares one **exponent**. A **value** is `mantissa × 3^exponent`, where mantissa is an integer in trit units.

- **Trit.** A base-3 digit, value ∈ {-1, 0, +1}. Storage-layer encoding of packed trits lives in `m4t_trit_pack.{c,h}`.
- **Mantissa.** A signed integer representing some number of trits, stored in a 2-complement cell (int8/int16/int32/int64). The integer value of the cell IS the trit-weighted sum — no scale factor, no hidden encoding. Range: `|mantissa| ≤ (3^n − 1) / 2` for an n-trit cell.
- **Exponent.** A signed integer, base-3. One exponent per **block** (see §4). Final real value of a cell within a block is `cell_mantissa × 3^block_exponent`.
- **MTFPn.** An n-trit mantissa cell (MTFP4, MTFP9, MTFP19, MTFP39). The cell is the *mantissa container* — exponent is sidecar metadata on the block, not stolen from the cell.

The previous implementation used `m4t_mtfp_t = int32_t` with `M4T_MTFP_SCALE = 3^10 = 59049` applied uniformly to every cell. That is a special case of this spec: it corresponds to one global block with `block_exponent = -10`. Under the new spec the exponent is per-block, not per-tensor, and is not baked into the type.

---

## 3. Hardware anchor: SDOT

The atomic hardware transaction on M-series NEON that does ternary matmul work is:

```
sdot  v_dst.4s,  v_a.16b,  v_b.16b
```

- Consumes **16 int8** per input (32 int8 total).
- Produces **4 int32** outputs (accumulating 4 × int8 × int8 into each output lane).
- 1 op per cycle on M-series big cores.

Every design decision below follows from this. If SDOT is the hottest ternary primitive on M4, then the unit M4T transacts in must be the unit SDOT transacts in.

Secondary anchors (in priority order):
- **TBL** (`tbl v_dst.16b, {v_tab.16b}, v_idx.16b`) — 16-byte table lookup. Backs trit ops and LUT-indexed nonlinearities.
- **VCNT + masked reductions** — 8-bit popcount, feeds routing distances.
- **`vmull_s32`** — 2-lane int32 × int32 → int64. Backs MTFP19 × MTFP19 → MTFP39.

Anchors explicitly NOT used:
- **SME / ZA tiles** — dense-matmul-shaped, hundreds of lanes. Out of scope; M4T is routing-first.
- **IEEE-754 FP units** — binary float. Banned except in build-time LUT generation (`m4t_lut_gen.c`).

---

## 4. The block: atomic unit of the substrate

**A block is one 128-bit NEON vector = 16 bytes.**

Cell count per block falls out of cell width:

| Cell type | Mantissa trits | Storage | Cells per block | Bytes per block | NEON vectors per block |
|---|---|---|---|---|---|
| MTFP4 | 4 | int8 | 16 | 16 | 1 |
| MTFP9 | 9 | int16 | 8 | 16 | 1 |
| MTFP19 | 19 | int32 | 4 | 16 | 1 |
| MTFP39 | 39 | int64 | 2 | 16 | 1 |

Why 16 bytes and not 8, 32, 64:
- **SDOT consumes 16 int8 per input lane.** One MTFP4 block = one SDOT input. One MTFP19 block (4 int32) = one SDOT output. Exponent arithmetic is a single scalar add per block.
- **TBL consumes 16 bytes of indices.** One MTFP4 block = one TBL input.
- **Width conversions preserve byte count.** Reinterpreting MTFP4 as MTFP19 changes cell count from 16 to 4, but bytes stay at 16 and the exponent array stays identical in layout.
- **8 cells would waste half the vector.** No M4 instruction runs faster on half-vectors.
- **32 cells would span 2 SDOT outputs per 1 exponent** — ambiguous ownership, no hardware "paired SDOT" to match.

This makes 16 bytes the only choice that composes cleanly across SDOT, TBL, vector arithmetic, and width reinterpretation.

---

## 5. Cell types (mantissa containers)

| Cell | Mantissa range (integer) | Dynamic range at `exp = 0` | Primary role |
|---|---|---|---|
| MTFP4 | ±40 (≈ ±3^3.37) | ±40 | Routing weights, ternary projections, SDOT inputs |
| MTFP9 | ±9 841 (≈ ±3^8.37) | ≈ ±10⁴ | Narrow activations, intermediate scores |
| MTFP19 | ±581 130 733 (≈ ±3^18.37) | ≈ ±5.8·10⁸ | General activations, SDOT outputs (natural accumulator) |
| MTFP39 | ±1.72·10¹⁸ (≈ ±3^38.37) | ≈ ±10¹⁸ | MTFP19×MTFP19 accumulator, high-precision paths |

Exact bound: `|mantissa|_max = (3^n − 1) / 2` for n-trit cells. Cell values outside this range are out-of-spec and must trigger widen (§8.5).

MTFP4 is SDOT-native: 4-trit signed range ±40 fits comfortably in int8 (±127), leaving headroom for intermediate accumulation before saturation kicks in.

---

## 6. Storage layout

**Structure-of-Arrays (SoA).** A tensor of N blocks is stored as two contiguous arrays:

```
mantissas:  bytes[N * 16]            // N NEON vectors, 16B each
exponents:  int8[N]                   // one signed exponent per block
```

Rationale:
- **Prefetch independence.** Mantissa array and exponent array stream at different rates; the prefetcher sees two clean strides rather than one strided-then-gathered access.
- **Vector alignment.** Every block is 16-byte-aligned by construction. No interleaving means no byte-offsetting inside a vector load.
- **Width reinterpretation is free.** An MTFP4 tensor can be reinterpreted as MTFP19 without touching the exponent array; only the reader's stride changes.
- **Compression.** The exponent array is 1/16 the size of the mantissa array. A 128-byte cache line of exponents covers 16 cache lines of mantissas — exponents are effectively hot in L1 at all times.

Not considered: AoS `{mantissa, exponent}` per cell. This is per-cell exponent, not per-block, and was rejected in §7.

---

## 7. Exponent granularity: per-block

The three candidates, with outcomes:

| Granularity | Mantissa trits per cell | Exponent overhead | Dynamic range | Hardware alignment |
|---|---|---|---|---|
| Per-cell | n − k (k stolen for exp) | 0 (in-cell) | Max per cell | Every op needs exp align — heavy |
| **Per-block (chosen)** | n (all mantissa) | 1 byte / 16 bytes (6.25%) | Max per block | Scalar exp add, one per NEON op |
| Per-tensor | n | ~0 | Minimum | Collapses to fixed-point (the failure mode) |

**Per-block wins on all three principles:** hardware-aligned (one exp per vector op), maximum resolution (all n trits are mantissa), and it's the pattern that commodity AI silicon already validates in binary (microscaling / MX formats). M4T is the base-3 version.

### Exponent encoding

- **Container:** signed int8 per block.
- **Interpretation:** base-3 integer. Value `e` means `mantissa × 3^e`.
- **Range:** [-40, +40] by convention. Combined with MTFP19 mantissa (~3^18.37) this gives overall dynamic range ~3^58 ≈ 10^28. Range `[-128, +127]` is physically available if the convention needs to widen later.
- **Zero-value convention:** a block whose every mantissa is 0 has exponent = 0 (canonical zero). The arithmetic rules never produce a nonzero exponent with all-zero mantissas.
- **Sentinels (OPEN — see §14):** whether to reserve any exponent values for NaN/saturation/widen-pending flags, or carry those as a separate per-block status byte.

---

## 8. Arithmetic rules

### 8.1 Same-block add

When two blocks share an exponent:
- Compute `mantissa_out[i] = mantissa_a[i] + mantissa_b[i]` in the widened lane type (e.g. MTFP19 + MTFP19 in int64 lanes via `vaddl_s32`).
- If every `|mantissa_out[i]| ≤ cell_max`, result is exact; write back with `block_exp_out = block_exp_a = block_exp_b`.
- If any `|mantissa_out[i]| > cell_max`, **widen** (§8.5).

No rounding. Exact result guaranteed in the non-widen path.

### 8.2 Cross-block add

When exponents differ by Δ = `exp_larger − exp_smaller`:
- Align the smaller-exponent block by dividing each of its mantissas by `3^Δ` (integer division, exact or remainder-bearing).
- If all remainders are zero: exact. Proceed as same-block add.
- If any remainder is nonzero: **rounding is required**. This is the one legitimate lossy path in the substrate. The default policy is:
  - Round-to-nearest-even in base-3 (ties go to the mantissa whose least-significant trit is 0).
  - Record a `rounding_occurred` flag at tensor or call-site level.
- Callers that need exactness can widen one side to a larger cell type first (guaranteeing `3^Δ` fits) and repeat.

**OPEN — §14.2:** whether "align and round" is the default or whether the substrate refuses and forces the caller to widen explicitly.

### 8.3 Multiply

- `mantissa_out[i] = mantissa_a[i] × mantissa_b[i]` in a widened accumulator (MTFP4×MTFP4 → int16 → MTFP9 mantissa range; MTFP19×MTFP19 → int64 → MTFP39 mantissa range).
- `block_exp_out = block_exp_a + block_exp_b`.
- If the product mantissas fit the next-tier cell, result is **exact** at that tier.
- If the product would overflow the widened tier (rare: MTFP39×MTFP39 → overflows int64): the substrate flags; no silent truncation.

### 8.4 SDOT as ternary matmul (MTFP4 × MTFP4 → MTFP19)

One SDOT op on the mantissa array:
```
sdot v_out.4s, v_a.16b, v_b.16b
```
- 16 MTFP4 mantissas on each side → 4 MTFP19 mantissas.
- Exponent: `block_exp_out = block_exp_a + block_exp_b` (scalar, one add).
- **Exact by construction.** SDOT's int32 output holds the full sum `Σ a_i × b_i` without rounding; the max value is 16 × 40 × 40 = 25 600, well inside int32 and well inside MTFP19 mantissa range.

This is the hottest path in the routing substrate. It is exact, hardware-native, and metadata-light.

**Contract.** SDOT over MTFP4 inputs is declared exact by the substrate. The bound `|output| ≤ 25 600 ≪ 2³¹ − 1` is a theorem, not a configuration. Callers that compose wider inputs into SDOT shape (e.g. via `vmull_s32` on MTFP19 pairs) are responsible for their own overflow checks; the substrate provides no wrapper that re-proves SDOT exactness for wider cases because no such proof is available by construction.

### 8.5 Widen, saturate, round — the three resolutions (invariant)

When an operation's exact result does not fit the target cell type, exactly one of three resolutions applies. The substrate names which case each operation falls into; silent rounding is never one of them.

**Case W — widen.** Output type admits a wider cell: promote to it. The exact result is preserved at the wider tier.
- Scalar mul: `MTFP_n × MTFP_m → MTFP_{n+m}` (widened accumulator).
- SDOT: MTFP4 × MTFP4 → MTFP19 mantissa (exact by §8.4 contract).
- Any op whose output type is a substrate-internal choice, not a caller-prescribed buffer.

**Case S — saturate.** Output type is *caller-prescribed* and cannot widen (in-place vec ops, accumulators that must land in a fixed-width tensor). Saturate at ±cell-max; this is lossy at the boundary but informative, not silent. Consumers that opt into flag tracking (§14.4) see a `SAT` bit set on the affected block.
- In-place `m4t_mtfp_block_add` / `_sub`: dst cell is MTFP19; can't widen without changing the caller's buffer type. Saturate.
- Routing's `apply_signed` accumulator: caller provides the output buffer; accumulator saturates per cell.

**Case R — round (named opt-in only).** Cross-block add with different block exponents (§8.2). The only legitimate rounding path in the substrate, and it must be requested by name (`m4t_mtfp_vec_add_aligning`). Never the default. Also produces a `ROUND` status on affected blocks.

**Block-exponent increment as a widening alternative.** When every mantissa in a block has a trailing 0 trit (i.e., is divisible by 3), the block's exponent can increment by 1 and each mantissa is divided by 3 exactly. This is a zero-loss alternative to cell-widening. When not applicable, cell-widening is required.

**Overflow ladder (Case W):**
- MTFP4 overflow → MTFP9 (2× cells in same 16 B block).
- MTFP9 overflow → MTFP19.
- MTFP19 overflow → MTFP39.
- MTFP39 overflow → block-exponent increment (if divisible by 3) OR substrate flags `OVER`; caller must partition. No silent saturation at this tier — this is the end of the ladder.

**Which case an op falls into is part of the op's contract.** Cell-type-changing ops (mul, SDOT) are Case W. Fixed-output-type ops (in-place vec, caller-buffer accumulators) are Case S. The named alignment variant is Case R. No op is silently any of the three.

---

## 9. Cache and prefetch

- **M4 big-core L1D line = 128 bytes.**
- **128 B / 16 B per block = 8 blocks per cache line.**
- A prefetched mantissa line carries 8 blocks' worth of cells; the corresponding 8 exponent bytes fit in 1/16 of an exponent-array cache line (128 blocks' worth).
- Stride-based hardware prefetcher sees unit stride on the mantissa array (16B blocks packed) and unit stride on the exponent array (1B exponents packed). Both streams prefetch cleanly, no gather.
- SoA also means tensor dimensions that are not multiples of 16 mantissa-bytes waste the tail of the last block only — not the exponent.

**OPEN — §14.3:** tail-block padding policy for tensor dimensions that are not multiples of the cell count per block.

---

## 10. Width conversions

Width conversions are byte-preserving reinterpretations; exponent array is unchanged.

| From | To | What changes | What doesn't |
|---|---|---|---|
| MTFP4 | MTFP19 | cells/block: 16 → 4; reader stride ×4 | exponent array, mantissa bytes |
| MTFP19 | MTFP39 | cells/block: 4 → 2; reader stride ×2 | exponent array, mantissa bytes |
| MTFP19 | MTFP4 | cells/block: 4 → 16; reader stride /4 | exponent array, mantissa bytes |

Widening a *cell* within a block (i.e., reallocating each mantissa into more bits to hold an overflow) is a different operation — it copies the mantissa into a new tensor of the wider cell type. See §8.5.

---

## 11. What M4T exposes (routing-first)

Primitives that compose into routing and into the scalar/vector arithmetic routing requires:

**Trit layer**
- `m4t_trit_pack` — pack/unpack between {-1,0,+1} and 2-bit packed trits.
- `m4t_trit_ops` — TBL-backed trit ops (mul, sat_add, max, min, eq, neg).
- `m4t_trit_reducers` — popcount-based reducers (signed_sum, sparsity, counts).

**MTFP numeric core (to rebuild per §5–§8)**
- Mantissa arithmetic: add, sub, neg, mul, widen, saturate.
- Block-exponent arithmetic: add, sub, align.
- Vector ops over mantissa arrays: vec_zero, vec_add, vec_sub, vec_scale.
- Width conversion: MTFP4↔MTFP9↔MTFP19↔MTFP39.

**Routing primitives**
- `m4t_route_threshold_extract` — int64 values + tau → packed-trit three-state signatures.
- `m4t_route_distance_batch` — query sig × T tile sigs → T distances.
- `m4t_route_topk_abs` — T scores → k (tile, sign) decisions.
- `m4t_route_apply_signed` — k decisions × tile outputs → accumulated MTFP.
- `m4t_route_signature_update` — weight-derived signatures (setup-time).

**Ternary matmul**
- `m4t_ternary_matmul` — SDOT-native MTFP4 × MTFP4 → MTFP19. This is the LSH projection primitive (Law #7 in the SDF journal): ternary projections applied to MTFP data.

---

## 12. What is NOT in M4T

Explicitly out of scope. Consumer-layer concerns.

- **Dense MTFP × MTFP matmul** (`matmul`, `matmul_bt`). Bends hardware. Archived.
- **Bias add, fan-in normalize, LayerNorm.** Dense-transformer plumbing. Archived.
- **GELU, softmax, argmax.** Dense-transformer nonlinearities. Archived with their LUTs; may return as routing-consumer primitives if needed.
- **Training.** M4T is inference-substrate-only. Training artifacts live in the consumer.
- **Threading.** Single-threaded at the opcode level. Parallelism is a consumer concern (no libdispatch, no pthreads).
- **Binary floating-point.** Banned at runtime. Permitted only in build-time LUT generation (`m4t_lut_gen.c`).

---

## 13. File organization (post-archive reality)

```
m4t/
  src/
    m4t_types.h              — cell typedefs, constants
    m4t_internal.h           — NEON detection, private macros
    m4t_trit_pack.{c,h}      — packed-trit storage
    m4t_trit_ops.{c,h}       — TBL-backed trit ops
    m4t_trit_reducers.{c,h}  — popcount reducers
    m4t_route.{c,h}          — routing primitives
    m4t_mtfp4.{c,h}          — 4-trit mantissa cell, SDOT-facing
    m4t_ternary_matmul.{c,h} — SDOT-native ternary matmul
    m4t_mtfp.{c,h}           — TO REBUILD: mantissa + block-exponent core
  tests/
    test_m4t_trit_ops.c
    test_m4t_trit_reducers.c
    test_m4t_route.c
    test_m4t_mtfp4.c
  tools/
    m4t_trit_golden.c        — enumerated golden-value tests
    m4t_size_check.sh        — .text budget enforcement
    m4t_lut_gen.c            — offline LUT generator (ONLY sanctioned float)
  docs/
    M4T_SUBSTRATE.md         — this file

src/                         — glyph wrapper headers (aliases over m4t)
  glyph_types.h
  glyph_trit_pack.h
  glyph_route.h
  glyph_ternary_matmul.h

tests/                       — glyph wrapper tests (TO REBUILD)
tools/
  mnist_trit_lattice.c       — LSH routing tool (primary benchmark)

archive/                     — everything that was dense or followed the fixed-point reading
```

---

## 14. Open decisions

Substrate-level questions where hardware does not dictate the answer. The original list of seven was reduced to four by triage under the substrate/consumer throughline; the full cycle is in `journal/seven_open_decisions_{raw,nodes,reflect,synthesize}.md`. Former 14.5 was promoted to a contract in §8.4 (proven theorem). Former 14.6 and 14.7 were moved to `docs/THESIS.md` (consumer/thesis concerns, not substrate).

### 14.1 Logical block size (OPEN — empirical)
Hardware says 16 B for the mantissa-block. Whether a *logical block* — the unit at which we group for prefetch tuning and scheduling — should be 1, 2, 4, or 8 hardware blocks is a cache/workload question with no theoretical derivation. **Decision:** logical block = hardware block (1:1) as the rebuild default. Revisit only when a running consumer shows prefetch or cache stress. This is the one genuinely open item in §14.

### 14.2 Cross-block add policy (DEFERRED)
No routing primitive we kept (`apply_signed`, `signature_update`, `distance_batch`) exercises cross-block add across different block exponents; accumulation happens within exponent-uniform blocks established at write time. **Decision:** do not implement until a consumer drives it. If one emerges, provide a named opt-in variant `m4t_mtfp_vec_add_aligning` with an explicit rounding flag, never as a default path. The "widen, don't round" invariant (§8.5) stands; cross-block alignment is the only legitimate lossy path and must be requested by name.

### 14.3 Tail-block padding (DECIDED: zero-pad)
Tensors whose dimensions aren't multiples of a block's cell count have a partial last block. **Decision:** zero-pad the unused mantissas. A zero mantissa is the additive and multiplicative identity for any block exponent (`0 × 3^e = 0`), so zero-padding inserts identity elements rather than altering values — the §8.5 invariant is preserved. Tensors carry a cell-count field so length-aware reducers can ignore the tail.

### 14.4 Exponent status tracking (DECIDED: parallel array if needed)
**Decision:** no sentinels in the exponent. Keep the exponent numerically pure. When flag semantics are needed (widen-pending, overflow, rounding-occurred), allocate a parallel 1-byte status array per block, scoped to the caller that requested flag tracking. The default path carries no status array at all.

---

## 15. Glossary

- **MTFP** — Multi-Trit Floating Point. Ternary-mantissa, ternary-exponent, per-block.
- **Trit** — Base-3 digit, value ∈ {-1, 0, +1}.
- **Cell** — One mantissa container (MTFP4/9/19/39).
- **Block** — 16 bytes = 1 NEON vector = 16/8/4/2 cells depending on cell type = unit sharing one exponent.
- **Mantissa** — The integer value of a cell, interpreted as a base-3 signed integer.
- **Exponent** — Signed int8 per block; base-3 scaling factor.
- **SDOT** — `sdot Vd.4s, Vn.16b, Vm.16b`. The hardware anchor.
- **SoA** — Structure-of-Arrays. Mantissas contiguous, exponents contiguous.
- **Routing** — Compute path built on sign_extract → distance_batch → topk_abs → apply_signed.
- **Widen, don't round** — Invariant: every non-cross-block MTFP op is exact or widens cell type / increments exponent.

---

## 16. Traceability

Each numbered section maps to a conversation decision:

| § | Decision | Source |
|---|---|---|
| 1 | Routing > dense; binary float is the enemy, not float | user, "routing is meant to be first-class" |
| 2 | MTFP is floating-point, not fixed-point | user, "the aim isn't to eliminate float" |
| 3 | SDOT is the hardware anchor | derived from ISA |
| 4 | Block = 16 B = 1 NEON vector | derived from SDOT / TBL atomicity |
| 5 | Cell trits all mantissa | user, "maximum resolution" |
| 7 | Per-block exponent | derived from hardware-alignment principle |
| 8.5 | Widen, don't round | user, "100% accuracy throughout" |
| 11/12 | Routing-first surface | user, archive criterion |
| 8.4 | SDOT exactness contract (proven theorem, promoted from former 14.5) | LMM cycle on §14 |
| 14 | Reduced from seven opens to four substrate-real items; 14.6 and 14.7 moved to `docs/THESIS.md` | `journal/seven_open_decisions_*.md` (LMM cycle, 2026-04-14) |

---

## 17. Spec-to-code cross-reference

For readers tracing a spec section back to the code that realizes it.

| § | Topic | Primary file(s) |
|---|---|---|
| 2 | MTFP vocabulary, mantissa/exponent semantics | `m4t/src/m4t_types.h` (cell typedefs, SCALE/RADIX framed as default-convention) |
| 3 | Hardware anchor (SDOT, TBL, VCNT, vmull_s32) | `m4t/src/m4t_mtfp4.c` (SDOT), `m4t/src/m4t_trit_ops.c` (TBL), `m4t/src/m4t_trit_reducers.c` (VCNT) |
| 4 | Block geometry (16 B = one NEON vector) | `m4t/src/m4t_types.h` (`M4T_BLOCK_BYTES`, cells-per-block constants); `m4t/src/m4t_mtfp.h` (`_Static_assert` enforcing the invariant) |
| 5 | Cell types + mantissa ranges | `m4t/src/m4t_types.h` |
| 6 | SoA storage layout | Implicit in all vec-op APIs (mantissa arrays; exponent arrays deferred until a consumer drives them) |
| 7 | Per-block exponent encoding | Default-block-exponent convention documented in `m4t_types.h`; per-block sidecar deferred (M2 in `docs/REMEDIATION_PLAN.md`) |
| 8.1 | Same-block add | `m4t/src/m4t_mtfp.c` (`m4t_mtfp_block_add` + vec composition) |
| 8.2 | Cross-block add | Deferred (§14.2) |
| 8.3 | Multiply (Case W widen) | Currently only scalar `m4t_mtfp_clamp64` in `m4t_mtfp.h`; full widening mul lands with a consumer |
| 8.4 | SDOT ternary matmul (exact by contract) | `m4t/src/m4t_mtfp4.c` (`m4t_mtfp4_sdot_matmul_bt`); MTFP19 variant in `m4t/src/m4t_ternary_matmul.c` |
| 8.5 | Widen / saturate / round resolutions | Case S saturation: `m4t_mtfp_block_add`/`_sub`, `m4t_mtfp_clamp64`, `m4t_mtfp4_clamp`, ternary_matmul store. Case W widen: SDOT path (§8.4). Case R: not implemented (§14.2). |
| 9 | Cache and prefetch (128 B line = 8 blocks) | Design constraint; no explicit enforcement needed in code |
| 10 | Width conversions | `m4t/src/m4t_mtfp4.c` (`m4t_mtfp4_to_mtfp19`, `m4t_mtfp19_to_mtfp4`) |
| 11 | Routing-first surface | `m4t/src/m4t_route.{c,h}` (the five primitives), `m4t/src/m4t_ternary_matmul.{c,h}` (LSH-projection primitive) |
| 12 | What is NOT in M4T | `archive/` (see `archive/README.md`) |
| 13 | File organization | `m4t/src/`, `m4t/tests/`, `m4t/tools/` — see `m4t/README.md` |
| 14.1 | Logical block size (OPEN) | Deferred; logical block = hardware block (1:1) |
| 14.2 | Cross-block add policy (DEFERRED) | No code; caller uses same-block ops or requests opt-in by name |
| 14.3 | Tail-block padding (DECIDED: zero-pad) | Vec ops in `m4t/src/m4t_mtfp.c` process whole blocks + scalar tail with identical semantics |
| 14.4 | Exponent status tracking (DECIDED) | No status array allocated until consumer requests |
| 18 | Base-3 native criterion (emission coverage + review gate) | `journal/base3_native_criterion_*.md` + `journal/updated_model_scrutiny_*.md` (LMM cycles, 2026-04-14) |

---

## 18. Base-3 native: emission coverage and the review gate

A primitive is "base-3 native" not as an absolute property but as a property of a **(primitive, input-distribution) pair**. The substrate publishes primitives; consumers deploy them with specific input distributions; correctness of "base-3 native" claims lives in the pair, not in the primitive alone.

### 18.1 Criterion — emission coverage

A (primitive, input-class) pair passes **emission coverage** iff every three-state position the primitive's API exposes is non-trivially exercised under that input class.

"Non-trivially" means: each of the three states occurs with positive measure under the input distribution — not as a measure-zero coincidence of exact arithmetic equality.

### 18.2 Scope — where the criterion applies

§18 applies to primitives that interact with the trit trichotomy at their API boundary. Three distinct scopes exist, and each primitive falls in exactly one:

**Output-side coverage** — enumerable three-state outputs.
The primitive emits output states drawn from {+1, 0, -1} (trit codes, decision signs, etc.). Emission coverage is tested at the output: all three codes must be emitted non-trivially under the sanctioned input class.
- Example: `m4t_route_threshold_extract` emits packed trit codes.

**Input-side coverage** — three-state-dependent behavior over continuous outputs.
The primitive's output is continuous (int32, MTFP mantissa array, etc.), but its useful semantic depends on the inputs realizing all three trit states. Emission coverage is tested at the input: the consumer must supply inputs that realize all three states, so that the primitive's trichotomy-aware behavior is actually exercised.
- Example: `m4t_route_distance_batch` outputs int32 distances, but the lattice-geometric distance metric (0-vs-sign = 1 bit, ±1-vs-∓1 = 2 bits) only realizes its three-way character when packed-trit inputs span all three states.
- Example: `m4t_route_apply_signed` outputs MTFP accumulations, but three-way dispatch (+1 add / -1 sub / 0 skip) only exercises all three branches when the decision array contains all three sign values.

**Not applicable** — primitives outside the trichotomy.
The primitive performs pure arithmetic on integer mantissas, cell-width conversion, or other operations that don't encode a three-state semantic at either end. §18 is silent; no coverage assertion required.
- Example: `m4t_mtfp_block_add`, `m4t_mtfp_clamp64`, `m4t_mtfp19_to_mtfp4`.

### 18.3 Worked examples

- `m4t_route_threshold_extract(tau > 0)` + values spanning ±tau: **output-side pass.** All three codes emitted.
- `m4t_route_threshold_extract(tau = 0)` + integer col_sum − mean inputs (`signature_update` internal): **output-side pass.** Zero occurs with positive measure from exact equality.
- `m4t_route_threshold_extract(tau = 0)` + continuous MTFP projection outputs: **output-side FAIL.** Zero is measure-zero. Produces sign-only classification in practice. Consumers in this class should use tau > 0.
- `m4t_route_distance_batch` + signatures from threshold_extract(tau > 0) output: **input-side pass.** Inputs realize all three packed-trit states; the lattice-geometric distance metric is fully exercised.
- `m4t_mtfp_block_add` + any in-range MTFP inputs: **not applicable.** Integer arithmetic does not encode the trichotomy at the API boundary.

### 18.4 Review gate

Every primitive added to the substrate MUST ship with:

(a) **Enumerated three-state position.** The API locus at which the trichotomy lives — either the output codes (output-side) or the input codes the primitive's semantic consumes (input-side). For "not applicable" primitives, §18 is declared not applicable in the primitive's docstring.
(b) **Sanctioned input-class contract.** The input distribution class(es) for which emission coverage holds, stated in the primitive's docstring.
(c) **Coverage test.** A test in `m4t/tests/` that verifies emission coverage — either by exercising the full output code space (output-side) or by supplying inputs that realize all three states (input-side). Coverage tests are labeled with `/* §18 coverage test */` or similar so the audit trail is machine-greppable.

Primitives that fail to ship all three do not land.

### 18.5 Per-primitive audit (as of 2026-04-14)

Each live routing primitive, classified by §18 scope with a pointer to its coverage evidence.

| Primitive | Scope | Sanctioned input class | Coverage evidence |
|---|---|---|---|
| `m4t_route_threshold_extract` | Output-side (emits {+1, 0, -1} trits) | tau > 0 with values spanning ±tau, OR tau = 0 with integer inputs realizing exact zero | `test_threshold_extract_tau0`, `test_threshold_extract_tau5`, `test_threshold_extract_all_within_band` in `test_m4t_route.c` |
| `m4t_route_distance_batch` | Input-side (consumes packed-trit query and tile signatures) | Signatures from a passing-coverage extractor (so all three packed-trit codes present) | `test_distance_batch` uses three-state tile signatures covering identical / all-differ / one-differs cases |
| `m4t_route_topk_abs` | Output-side (decision `sign` field ∈ {+1, -1, 0-sentinel}) | Score arrays with mixed signs AND (k > #nonzero OR k ≤ #nonzero) to realize all three sign states | `test_topk_abs_basic` (+1, -1), `test_topk_abs_with_zeros` (+1, 0-sentinel), `test_topk_abs_all_tiles` (+1, -1); union covers all three |
| `m4t_route_apply_signed` | Input-side (consumes decision `sign` field) | Decisions from `topk_abs` output, exercising all three sign branches under realistic routing loads | `test_apply_signed` (+1 add, -1 sub), `test_apply_signed_sentinel` (0-sentinel skip) |
| `m4t_route_signature_update` | Compound — internally uses `threshold_extract(tau=0)` on col_sum-minus-mean | Integer weight matrices where col_sum-vs-mean can realize exact equality (typical for ternary weights across T tiles) | `test_signature_update` produces signatures with all three trit states realized |
| `m4t_mtfp_ternary_matmul_bt` | Input-side (consumes packed-trit weights; three-way branch per trit) | Ternary weight matrices with genuine {-1, 0, +1} sparsity; any MTFP activations | Covered by `test_signature_update` end-to-end and indirectly by `mnist_trit_lattice` consumer |
| `m4t_mtfp4_sdot_matmul_bt` | Input-side (consumes MTFP4 activations and int8 trit weights) | Ternary weights with genuine {-1, 0, +1} sparsity | `test_f4_sdot_small`, `_k32`, `_k17_tail`, `_sdot_saturation` — all exercise the zero-preserves-zero invariant |

**Primitives §18 does not apply to** (listed for completeness; §18 is silent on these): `m4t_mtfp_clamp64`, `m4t_mtfp_vec_zero`, `m4t_mtfp_block_add/_sub`, `m4t_mtfp_vec_add_inplace/_sub_inplace`, `m4t_mtfp4_clamp`, `m4t_mtfp4_to_mtfp19`, `m4t_mtfp19_to_mtfp4`, trit pack/unpack primitives, trit reducers. These are either pure integer arithmetic or container-level operations that do not encode the trichotomy at their API boundary.

### 18.6 Why this matters

A prior primitive (`m4t_route_sign_extract`, now removed) advertised three output codes in its type system but produced only two on realistic continuous-valued inputs. The type-level three-ness was theater; the behavioral two-ness was what downstream routing actually saw. The resulting consumer experiment lost 23 accuracy points vs. a dense baseline — a failure that was invisible until the primitive was exercised end-to-end.

The emission-coverage criterion is the specific defense against this class of failure. It asks the question the type system cannot: *under the inputs this primitive will actually see, does its three-way semantic actually get exercised?*

### 18.7 History

This section exists because two LMM cycles in sequence converged on it. The first cycle produced a two-part criterion (C-sub: substrate-side three-way capacity; C-con: consumer-side realization). The scrutiny meta-cycle found the two parts collapsed structurally — C-sub applied literally to `sign_extract` fails because collapsing its zero state yields a well-defined binary sign-test. The single-part "emission coverage" criterion subsumes both correctly. A subsequent red-team noted the criterion needed scope qualification (output-side / input-side / not applicable) and a per-primitive audit trail; both landed in §18.2 and §18.5.

Journal record: `base3_native_criterion_{raw,nodes,reflect,synthesize}.md` (first cycle); `updated_model_scrutiny_{raw,nodes,reflect,synthesize}.md` (scrutiny cycle).
