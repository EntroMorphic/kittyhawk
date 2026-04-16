---
date: 2026-04-16
phase: NODES
topic: Dynamic N_PROJ — resolution-adaptive routing cascade
---

# Dynamic N_PROJ — NODES

Discrete ideas extracted from RAW. Tensions and dependencies mapped.
Not solving yet.

---

## Node 1 — Two variants emerged: Full Cascade vs Re-Rank

RAW surfaced a critical fork in the design space:

**Full Cascade:** independent bucket tables at each N_PROJ stage.
Each stage builds its own projection, encodes, indexes, probes,
and resolves from scratch. Maximum power. High infrastructure cost
(uint64 bucket keys, multi-stage table management, full probe pass
per escalated query).

**Re-Rank:** Stage 1 builds the candidate union at N_PROJ=16 as
today. Uncertain queries are re-scored by computing wider (N_PROJ=32)
signatures for just the union candidates and running SUM at the
wider width. No new bucket index, no new probing. Only signature
encoding + resolve over the existing union.

The oracle data (100% at M≥8) guarantees the correct answer is
already in the Stage-1 union. Re-ranking can't miss it.

**Tension:** Full Cascade is more powerful (new union could contain
candidates that N_PROJ=16 missed). Re-Rank is dramatically cheaper
(no uint64 bucket keys needed at all; no new table build; just
sig_encode + popcount_dist on the existing union). But Re-Rank
relies on the N_PROJ=16 union being sufficient — which the oracle
data says it is.

## Node 2 — The oracle argument is exact, not approximate

Oracle accuracy at M=64 is 100.00% on all three datasets (MNIST,
Fashion-MNIST, CIFAR-10) at M≥8. This means the Stage-1 union
ALWAYS contains a training prototype of the correct class. The
resolver failure is not "wrong candidates in the union" but "wrong
ordering of the candidates that ARE in the union."

This makes the Re-Rank variant structurally sound. The wider
signatures aren't needed to FIND better candidates — they're
needed to RANK the existing candidates correctly.

## Node 3 — The confidence gate distribution needs per-query data

RAW noted that the aggregate mean (24.4% vs 15.3% per-table
votes) is clean but the per-query distribution might overlap.
The 5.8% swing queries are exactly where the overlap lives.

**Need:** histogram of per-table votes for the winner on correct
vs incorrect queries at seed 0. If the distributions are
bimodal with little overlap, any reasonable threshold works. If
they overlap heavily, the gate is noisy and the cascade's
routing is unreliable.

**Already have the data** — s0_votes_w was computed in the overlap
tool. Could histogram it from the existing run with a small extension.

## Node 4 — Re-Rank resolves the 75% tie problem directly

The CIFAR-10 atomics showed 75.2% of per-table signature
comparisons are tied at N_PROJ=16. At N_PROJ=32, each table
has 2× the Hamming bits — the tie rate should drop dramatically
because the finer-grained signatures distinguish candidates that
16-trit signatures collapse.

Re-Rank at N_PROJ=32 over the Stage-1 union specifically targets
this: same candidates, more discriminative scoring. The union is
already good (oracle 100%); the scoring is the bottleneck.

## Node 5 — Re-Rank doesn't need uint64 bucket keys

This is a major infrastructure simplification. The full cascade
design requires generalizing glyph_bucket from uint32 to uint64
keys (~200 lines). Re-Rank never indexes wider signatures — it
only encodes them and computes popcount_dist. The existing
`glyph_sig_builder_init`, `glyph_sig_encode_batch`, and
`m4t_popcount_dist` all already work at any sig_bytes. No
library changes needed.

## Node 6 — The cost of Re-Rank is bounded and predictable

For an escalated query with union size U and M₂ tables at
N_PROJ=32 (sig_bytes=8):

  cost = U × M₂ × popcount_dist(8 bytes)

At U=12,850 (CIFAR-10 avg) and M₂=64: ~823K popcount_dist
calls at 8 bytes. Each is ~4 ns on NEON → ~3.3 ms per escalated
query. Comparable to the Stage-1 resolve cost. Acceptable.

If M₂ is reduced to 32: ~1.6 ms per escalated query.

## Node 7 — M₂ (number of tables at the wider N_PROJ) is a free variable

The Stage-1 pass uses M=64 tables at N_PROJ=16 (1024 total trits).
The Re-Rank pass computes M₂ tables at N_PROJ=32. But M₂ doesn't
have to be 64. If 32 trits per table are discriminative enough,
M₂=32 (also 1024 total trits) might suffice. Or M₂=16 might
work if 32-trit signatures are much richer per table.

**Tension:** larger M₂ = more accurate re-ranking but slower.
The atomics tool can diagnose the right M₂ by measuring the
per-table tie rate at N_PROJ=32.

## Node 8 — Build cost is dominated by signature encoding, not tables

Re-Rank still needs to encode N_train training signatures at
N_PROJ=32 for each of M₂ tables. At N_train=50K, N_PROJ=32,
dim=3072: the projection kernel (ternary matmul) dominates. This
takes ~40s for M₂=64 at N_PROJ=16; at N_PROJ=32 expect ~80s.

But this is a one-time cost at startup, not per-query. And it
doesn't require bucket building (no sorting, no index construction).
Just encode + store the wider signatures.

## Node 9 — The confidence gate could be implicit

Instead of an explicit threshold, the system could ALWAYS re-rank
with wider signatures. On easy queries, the wider SUM will agree
with the narrow SUM (same winner). On hard queries, the wider
SUM picks a different (hopefully better) winner.

Advantage: no threshold to calibrate. Disadvantage: every query
pays the re-rank cost. This is the "always-on re-rank" variant.

**Tension with cost-adaptive goal:** the whole point of the cascade
is that easy queries should be cheap. Always-on re-rank is accurate
but not cost-adaptive. If the re-rank cost is small relative to
the Stage-1 cost (e.g., 30% overhead), always-on may be the
pragmatic choice. The confidence gate adds value only when re-rank
cost is large enough that skipping it matters.

## Node 10 — Staged rollout: Re-Rank first, Full Cascade later

These aren't mutually exclusive. Re-Rank is a strict subset of
the full cascade design:

1. Ship Re-Rank first (no uint64 keys, no new library changes).
   Measure accuracy gains and per-query cost.
2. If Re-Rank closes most of the gap → stop. The cascade is
   "just use wider signatures for scoring."
3. If Re-Rank hits a ceiling (union is missing candidates that
   wider probing would find) → build the full cascade with
   independent N_PROJ=32 tables and bucket keys.

Step 3 is only needed if oracle accuracy drops at the wider
N_PROJ for some queries — which would mean the Stage-1 union
is incomplete. Current data says it isn't, but CIFAR-10 at
N_PROJ=32 might have a different union shape.

## Node 11 — The name "Dynamic N_PROJ" is about the vision, not the implementation

The user's framing — "what if N_PROJ were dynamic" — points to
an architecture where the system self-selects its resolution.
Re-Rank is the minimal implementation that tests this thesis.
Full Cascade is the maximal. Both embody the same principle:
the lattice adapts its granularity to the input.

## Dependencies

- Node 1 depends on Node 2 (oracle argument gates Re-Rank viability)
- Node 5 depends on Node 1 (Re-Rank branch avoids uint64 work)
- Node 6 depends on Node 7 (cost depends on M₂ choice)
- Node 9 depends on Node 3 (if gate is clean, use gate; if noisy, go always-on)
- Node 10 depends on everything (staged rollout integrates all nodes)

## Tensions to resolve in REFLECT

**T1:** Full Cascade vs Re-Rank — which first, and can Re-Rank suffice?
**T2:** Explicit confidence gate vs always-on re-rank — does the gate add enough value to justify its calibration cost?
**T3:** How to set M₂ — same as M₁, or smaller since wider signatures are richer?
