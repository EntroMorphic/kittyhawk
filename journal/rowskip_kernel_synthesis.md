---
cycle: rowskip kernel — full build/red-team/bench cycle
phase: ALL — design + impl + test + bench + red-team + re-bench + verdict
date: 2026-05-07
scope: build a row-skip dense kernel that exploits BitNet's empty
       K-rows (~10% in o_proj, up to 44% in down_proj). Per user
       directive: "Build the kernel, red-team it, bench it, red-team
       the results, bench again."
companions: commit (rowskip kernel + tests + first bench), this
            commit (red-team + v2 bench), journal/bitnet_dead_columns.md
            (where the empty-row finding originated).
---

# rowskip kernel — synthesis

## Cycle

User directive: "Build the kernel, red-team it, bench it, red-team
the results, bench again."

Each phase recorded:

  Phase 1 — design + API
  Phase 2 — implementation (encoder + kernel)
  Phase 3 — bit-exact tests (18 cases, ASAN+UBSAN clean)
  Phase 4 — first bench: revealed serious regressions
  Phase 5 — red-team kernel + bench: found K%80 scalar-tail bug
  Phase 6 — fix + re-test + re-bench v2: hardened with three-way
            decomposition to separate tile-alignment from row-skip
  Phase 7 — red-team v2 results: discovered most "win" is from
            tile-alignment side effect, not row-skipping
  Phase 8 — honest verdict (this synthesis)

## What rowskip is

Per BitNet's training, ~10% of layer-0 o_proj's input dims contribute
nothing across all output cols (cell magnitudes either truly zero in
bf16 or rounded to zero by absmean quantization). Layer-1 down_proj
has 43.6% empty K-rows. Concentrated at boundary layers; middle
layers are essentially dense.

The rowskip kernel:
  1. At pack time: build a list of non-empty K indices, repack W
     into 5-in-8 over only those positions, pad K_compressed up
     to next multiple of 80 (the dense kernel's NEON tile size).
  2. At call time: gather X[nonempty_indices] into a scratch
     buffer; dispatch to the existing dense kernel at the
     compressed+padded K.

Bit-exact to the dense kernel because empty K-rows contribute 0
to every output regardless of X.

## Phase 4 → Phase 5: the K%80 bug (red-team #1)

First bench showed bewildering pattern:

  L0 o_proj K_c=2163 skip=15.5% → 1.20× WIN
  L1 o_proj K_c=2474 skip=3.4%  → 0.37× REGRESSION (huge!)
  L2 o_proj K_c=2557 skip=0.1%  → 0.35× REGRESSION

The dense kernel times were stable (~0.077-0.080 ms). Rowskip times
varied 3× depending on K_compressed.

**Root cause:** the dense kernel uses a slow per-trit scalar tail for
K%80 != 0. K_c values like 2474 (K%80=74) and 2557 (K%80=77) hit the
tail HARD — 70+ scalar trits per output × 2560 outputs ≈ 200k scalar
ops per call. This wiped out any row-skip benefit.

**Fix:** pad K_compressed up to the next multiple of 80 in the
encoder. Padded positions contain zero trits, contributing 0 to the
dot product. Stays bit-exact, removes the scalar tail.

  Encoder: K_padded = ((K_c + 79) / 80) * 80
  Storage: pad W_packed to N × M4T_TRIT_PACKED5_BYTES(K_padded)
  Kernel: gather X into K_padded scratch (zero-init padding)

After the fix: aggregate +5.5% speedup applying rowskip uniformly
across all 210 BitLinear calls, with no regressions on the
problematic K%80 cases.

## Phase 6 → Phase 7: tile-alignment confound (red-team #2)

Closer reading of the v1 results revealed something off:

  L10 down_proj at 0.3% skip → 1.23× WIN

That ratio can't come from skipping 23 rows out of 6912. Something
else is helping.

**Root cause:** down_proj has K=6912, K%80 = 32 — meaning the dense
baseline ALREADY has a 32-trit scalar tail. Rowskip's K_padded is
always a multiple of 80, so it avoids the tail. The "win" on
low-skip down_proj cases is mostly tile-alignment, not row-skipping.

**Fix:** v2 bench adds a third variant — `rs_no_skip` — where W is
fabricated to have NO empty rows, so rowskip's encoder pads the
original K to the next 80-multiple but doesn't actually compress
anything. This isolates pure tile-alignment benefit.

## Phase 7 results: honest decomposition

After v2 bench (200 iter, mean ± stddev, deterministic seeds):

  Variant                              Time      vs dense
  ------------------------------------+--------+----------
  dense (baseline)                     27.98 ms  1.000×
  rs_no_skip (tile-aligned, 0% skip)   26.61 ms  1.052×
  rs (full row-skip)                   26.27 ms  1.065×
  smart-dispatch (rs only when ≥5%)    27.55 ms  1.016×

  Decomposition:
    Tile-align contribution:    +4.90% (from K%80 padding side effect)
    Pure row-skip contribution: +1.22% (rs vs rs_no_skip)
    Combined always-on:         +6.12%
    Smart-dispatch:             +1.55%

**~80% of the headline gain is the tile-alignment side effect**, not
row-skipping. This benefit is available to ANY kernel that pads K to
a multiple of 80 — it is not unique to rowskip and could be obtained
by patching the dense kernel directly.

## Per-BitLinear details

Where rowskip wins decisively:

  L1 down_proj K=6912 skip=43.6% — 2.13× over dense, 1.75× over rs_no_skip
                                   (53% of dense time saved)
  L2 down_proj K=6912 skip=27.7% — 1.67× over dense, 1.38× over rs_no_skip

Where rowskip wins from row-skip alone (K%80=0 BitLinears):

  L29 o_proj K=2560 skip=24.0% — 1.26× over dense (no tile-align bonus)
  L0  o_proj K=2560 skip=15.5% — 1.11× over dense

Where rowskip is essentially wash or slight loss:

  Most o_proj at <5% skip: 0.97-0.99× (gather overhead)
  Mid-layer down_proj at <5% skip: 1.20-1.28× (entirely tile-align)

## Red-team v2 findings (third red-team)

1. **Most of the "win" is from a side effect.** The honest "row-skip
   per se" benefit is +1.22% always-on, +1.55% with dispatch. Modest.

2. **A simple K%80 patch to the dense kernel** would capture most of
   the +4.9% tile-alignment win for ALL callers, not just rowskip.
   This is the bigger optimization opportunity surfaced by this work.

3. **L15 down_proj outlier** (rs=0.459±0.281 ms, 1 sample huge).
   Measurement noise, not a real regression. 200 iter helped but
   didn't fully kill it. Doesn't change conclusions.

4. **Smart-dispatch threshold of 5%** is empirical. At 1% threshold
   we'd capture more cases at risk of more gather overhead. Worth
   tuning if shipped.

5. **Per-call malloc** (X_compressed) for every kernel call adds
   ~100 ns. 210 calls/token × 100 ns = 21 µs/token. Material relative
   to the +1.55% benefit. Could be eliminated by stashing X_compressed
   in the packed handle — but only if single-threaded use is the
   contract (note in header).

## Disposition

**Rowskip ships as-is** with these honest claims:
  - Correct (18 tests, ASAN+UBSAN clean, bit-exact vs dense)
  - +6.12% aggregate when always-on across BitNet inference's 210 calls
  - +1.55% with smart dispatch (skip% ≥ 5%)
  - L1 down_proj is the only BitLinear that benefits decisively
    from row-skip alone (53% time reduction)
  - Most of the headline gain is tile-alignment side effect; this
    is documented honestly, not hidden.

**The bigger win available** is a K%80-aware patch to
m4t_ternary_5in8_matmul_bt's tail. That would:
  - Capture the +4.9% tile-alignment benefit for all callers, not
    just rowskip
  - Reduce rowskip's incremental value to +1.55% (smart dispatch)
  - Make the rowskip kernel marginal — worth keeping only for
    L1 down_proj's 53% case

**Filed for future cycle.** Not building now; out of this scope.

## What this teaches

The cycle worked. Each red-team caught a real issue:

  Red-team 1: K%80 scalar tail bug — would have shipped a kernel
              with massive regressions on most BitLinears. Caught
              by inspecting the timing pattern instead of just
              accepting the aggregate.

  Red-team 2: tile-alignment confound — would have claimed +6%
              speedup as a "row-skip" win when ~80% was a side
              effect. Caught by designing a controlled bench
              variant (rs_no_skip).

The methodical structure ("build, red-team, bench, red-team results,
bench again") forced these findings to surface. The first bench
result alone would have been wrong; the second bench alone would
have been misleading. Both red-teams produced material corrections.

The lesson generalizes: when a kernel's ratio is suspicious (too
good for the obvious mechanism), find a control that isolates the
mechanism. If the control matches the test, the test isn't measuring
what it claims.
