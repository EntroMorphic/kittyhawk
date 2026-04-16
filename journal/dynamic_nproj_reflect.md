---
date: 2026-04-16
phase: REFLECT
topic: Dynamic N_PROJ — resolution-adaptive routing cascade
---

# Dynamic N_PROJ — REFLECT

Finding the structure beneath the content. Resolving tensions.

---

## Core insight

RAW started with a cascade design (independent tables at each
N_PROJ stage). But it drifted toward something simpler: **the
Stage-1 union is already correct; only the SCORING is wrong.**

The oracle is 100% at M≥8 on all three datasets. That means the
candidate union at N_PROJ=16 always contains a prototype of the
correct class. The resolver just ranks it wrong because 75% of
per-table signature comparisons are tied — the 16-trit lattice
can't distinguish the correct neighbor from the confuser.

Wider signatures don't need to FIND new candidates. They need to
RE-ORDER the existing ones. This is Re-Rank, not Cascade.

The cascade is overkill for the measured failure mode. It's
like rebuilding the index to fix a sort comparator. Re-Rank
fixes the comparator directly.

## Tension T1 resolved: Re-Rank first, Full Cascade unnecessary (for now)

**Re-Rank is sufficient** because the Stage-1 union is complete
(oracle 100%). The only scenario where Full Cascade adds value
is if wider-N_PROJ probing finds candidates that N_PROJ=16
probing misses — i.e., if the oracle drops below 100% when
measured against a wider-N_PROJ union. Current data shows no
such drop; in fact, oracle hits 100% even at M=8, meaning
significant oversampling. The union is saturated.

**Decision:** implement Re-Rank. Defer Full Cascade to a future
where oracle accuracy degrades (e.g., harder datasets with
smaller training sets where N_PROJ=16 multi-probe can't find
the right neighborhood).

**Consequence:** no uint64 bucket keys needed. No glyph_bucket
changes. No new library modules. The implementation is ~100
lines of tool-level orchestration — build wider encoders at
startup, and for each query, optionally re-score the Stage-1
union with the wider signatures.

## Tension T2 resolved: start always-on, gate is future optimization

The confidence gate (threshold on per-table votes) adds
engineering complexity and a calibration parameter. The Re-Rank
cost is bounded: U × M₂ × popcount_dist(sig_bytes₂). At
typical numbers (U=12K, M₂=64, 8-byte sigs), that's ~3.3 ms
per query — comparable to Stage 1's resolve cost.

If we always re-rank, total per-query cost roughly doubles.
But we eliminate the threshold parameter, the false-positive
risk (accepting wrong answers that could have been corrected),
and the calibration sensitivity across datasets.

**Decision:** always-on re-rank for V1. Every query gets
Stage-1 (N_PROJ=16) SUM prediction AND Stage-2 (N_PROJ=32)
re-ranked prediction. Report both. If the re-rank cost is
unacceptable for deployment, add the confidence gate as V2.

**Why this is safe:** re-ranking can't regress accuracy. If
the wider signatures pick the same winner as the narrow ones,
the answer is unchanged. If they pick a different winner, it's
because the wider ranking is more discriminative. Monotone
improvement is guaranteed by construction.

Actually — wait. Is monotone improvement guaranteed? What if
the N_PROJ=32 SUM picks a different winner that happens to be
wrong, when the N_PROJ=16 SUM was right? That would be a
regression.

The always-on variant doesn't guarantee monotone accuracy.
It guarantees that the wider ranking is *more discriminative*
(fewer ties, finer distance resolution), but "more
discriminative" doesn't mean "always more correct." A finer
distance can amplify noise as well as signal.

**Revised decision:** report BOTH predictions for every query.
Measure accuracy of N_PROJ=16 SUM alone, N_PROJ=32 Re-Rank
alone, and (for reference) a "take Re-Rank when N_PROJ=16 is
uncertain" gated variant. This is a measurement pass, not a
deployment. We'll see which strategy wins empirically before
committing to one.

## Tension T3 resolved: M₂ = M₁ initially, sweep later

The question of how many wider tables to build is an
optimization knob, not a design decision. Start with
M₂ = M₁ = 64 (same number of tables at both widths).
This gives the widest possible Re-Rank signal. If the
accuracy gain is real, sweep M₂ ∈ {16, 32, 64} to find
the minimal M₂ that preserves the gain.

**Rationale:** we're in the "does this work at all?" phase.
Optimizing M₂ before proving the concept is premature.

## What I now understand that I didn't in RAW

1. **The design space collapsed.** RAW presented a rich
   cascade architecture with multiple stages, confidence
   gates, and threshold calibration. NODES showed the oracle
   argument eliminates most of the complexity. REFLECT
   confirms: Re-Rank over the existing union is the
   minimum viable experiment, and it's structurally
   sufficient.

2. **The risk is not in the union.** The risk is whether
   wider signatures actually produce better rankings on
   CIFAR-10's specific input distribution. The scaling curve
   on MNIST says yes. But CIFAR-10's 3072-dim RGB might
   behave differently — random projections over natural
   images at any width might produce rankings that are only
   marginally better. This is the experiment's key empirical
   unknown, and no amount of design work can resolve it.
   We have to measure.

3. **Always-on re-rank is NOT guaranteed monotone.** This
   was a mistaken assumption in the design doc. A wider
   ranking is higher-resolution but not necessarily higher-
   accuracy on every query. The safe approach is to measure
   both and choose empirically.

4. **The implementation is tiny.** No library changes. Build
   M₂ wider sig_builders at startup. For each query, after
   the Stage-1 resolve, encode the query's wider sigs and
   compute sum_dist_wide for each candidate in the union.
   Pick argmin. ~50 lines of tool code on top of the
   existing query loop.

## What remains uncertain

- Whether N_PROJ=32 over 3072-dim RGB actually resolves the
  75% per-table tie rate. (Measurable.)
- Whether the accuracy gain justifies the build-time and
  per-query cost. (Measurable.)
- Whether the "always-on" vs "gated" tradeoff matters at
  CIFAR-10's accuracy level. (Measurable — report both.)
- Whether Re-Rank is a dead end if the gain is small and
  Full Cascade would have done better. (Answerable by
  checking whether the Stage-1 union is actually the
  bottleneck — the oracle data says it isn't, but oracle
  at N_PROJ=32 would confirm.)
