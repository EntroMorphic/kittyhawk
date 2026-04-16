---
date: 2026-04-16
phase: SYNTHESIZE
topic: Dynamic N_PROJ — resolution-adaptive routing cascade
---

# Dynamic N_PROJ — SYNTHESIZE

Executable specification. Someone else could build this.

---

## What to build

A **Re-Rank pass** that re-scores the Stage-1 candidate union
using wider signatures. Not a full cascade. No new bucket index,
no new probing, no confidence gate. Always-on for the measurement
phase.

## Why Re-Rank, not Full Cascade

The oracle is 100% at M≥8 on all three datasets. The correct
neighbor is always in the Stage-1 union. The failure is ranking,
not finding. Re-Rank fixes ranking directly. Full Cascade is
deferred — it adds value only when the union is incomplete, which
current data says it isn't.

## Why always-on, not gated

The confidence gate requires threshold calibration and introduces
false-positive risk (accepting wrong answers that re-rank would
have corrected). Always-on re-rank for V1 avoids both. Report
three accuracy numbers per M checkpoint:

1. `SUM_16` — Stage-1 SUM at N_PROJ=16 (current baseline)
2. `SUM_32` — Re-Rank SUM at N_PROJ=32 over Stage-1 union
3. `GATE_32` — SUM_16 when vote_margin ≥ K, else SUM_32

(3) is measured at several K values post-hoc from the same data.

## Implementation plan

### Step 1: build wider signature encoders at startup

Alongside the existing M builders at N_PROJ=16, build M₂ builders
at N_PROJ=32 (sig_bytes=8). Each uses the same seed derivation as
the narrow pass. Encode all N_train training prototypes at both
widths.

```c
int rerank_n_proj = 32;
int rerank_sig_bytes = M4T_TRIT_PACKED_BYTES(rerank_n_proj); /* 8 */
int M2 = cfg.m_max; /* same as M for V1 */

glyph_sig_builder_t* rr_builders = calloc(M2, sizeof(...));
uint8_t** rr_train_sigs = calloc(M2, sizeof(uint8_t*));
for (int m = 0; m < M2; m++) {
    uint32_t seeds[4];
    derive_seed(m, cfg.base_seed, seeds);
    glyph_sig_builder_init(&rr_builders[m], rerank_n_proj, ...);
    rr_train_sigs[m] = calloc(ds.n_train * rerank_sig_bytes, 1);
    glyph_sig_encode_batch(&rr_builders[m], ds.x_train, ...);
}
```

Test query signatures are NOT pre-encoded. They're encoded
on-the-fly during the query loop (one query at a time, M₂
signatures of 8 bytes each — negligible per-query cost).

### Step 2: per-query re-rank in the query loop

After Stage-1 SUM resolve, re-score the Stage-1 union:

```c
/* Encode this query's wider signatures. */
uint8_t rr_qsig_buf[M2][8];
for (int m = 0; m < M2; m++)
    glyph_sig_encode(&rr_builders[m], &ds.x_test[q * input_dim],
                      rr_qsig_buf[m]);

/* Re-rank: SUM at N_PROJ=32 over Stage-1 union. */
uint8_t rr_mask[8]; memset(rr_mask, 0xFF, 8);
int pred_rr = glyph_resolver_sum(
    &u, M2, rerank_sig_bytes,
    rr_train_sigs, rr_q_ptrs, rr_mask);
```

### Step 3: report three accuracy columns

```
M   SUM_16  SUM_32_RR  GATE_32(K=20)  oracle
64  35.32%   ?.??%       ?.??%        100.00%
```

### Step 4: run on all three datasets

- CIFAR-10 (--no_deskew): the primary target
- Fashion-MNIST (--no_deskew): secondary
- MNIST (deskew on): validation that re-rank doesn't regress

## What this does NOT include

- No uint64 bucket keys (not needed for Re-Rank).
- No new library modules.
- No confidence gate in V1 (report GATE_32 post-hoc from
  the same run data).
- No multi-seed (single seed for the proof-of-concept;
  multi-seed confirmation after the go/no-go on CIFAR-10).

## Go / no-go criteria

**Go (Re-Rank works):** CIFAR-10 SUM_32_RR ≥ 40% (≥5pp over
the 35.32% baseline). This would confirm that wider signatures
provide genuinely better ranking on natural images.

**Marginal (Re-Rank helps but not enough):** CIFAR-10 SUM_32_RR
in 37-40%. Re-Rank helps but the gain is small. Investigate
N_PROJ=64 re-rank or Full Cascade.

**No-go (Re-Rank doesn't help):** CIFAR-10 SUM_32_RR ≤ 37%.
Wider random projections don't improve ranking on 3072-dim RGB.
The fix requires non-random projections (learned features,
spatial blocks, grayscale preprocessing) rather than wider
random ones. Pivot accordingly.

## Files to create / modify

| file | action |
|---|---|
| `tools/mnist_routed_bucket_multi.c` | add re-rank pass + reporting |
| `docs/DYNAMIC_NPROJ.md` | already written (design doc) |
| `journal/dynamic_nproj_*.md` | this LMM cycle |

No changes to `src/glyph_*.{h,c}`.

## Estimated effort

- Step 1 (wider encoders): ~20 lines in the tool's build section.
- Step 2 (per-query re-rank): ~30 lines in the query loop.
- Step 3 (reporting): ~10 lines.
- Step 4 (three dataset runs): ~3 minutes MNIST + ~5 minutes
  Fashion-MNIST + ~10 minutes CIFAR-10.

Total: ~60 lines of code + ~20 minutes of measurements.

## What the LMM cycle changed

The design document proposed a full multi-stage cascade with
confidence gates, uint64 bucket keys, and per-stage table
management. RAW's doubt about whether N_PROJ=32 would help
CIFAR-10 survived and became the experiment's go/no-go gate.
NODES surfaced Re-Rank as a cheaper alternative. REFLECT
found that the oracle argument makes Re-Rank structurally
sufficient and that always-on re-rank (no gate) is the right
V1. The implementation collapsed from ~200+ lines of library
changes to ~60 lines of tool-level orchestration.

The full cascade remains as the design for V2, contingent
on Re-Rank V1 proving that wider signatures improve CIFAR-10
ranking at all.
