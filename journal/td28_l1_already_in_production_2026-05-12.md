# The production substrate distance kernel IS L1 — the entire Phase β/γ/δ/ε arc was comparing two Python implementations, not testing a production change

## What I just discovered

The production substrate sparse-attention kernel `m4t_route_distance_batch`
calls `m4t_popcount_dist` (defined in `m4t/src/m4t_trit_pack.c`). That
kernel **computes L1 path-graph distance on packed 2-bit trit codes**,
not categorical Hamming.

The mechanism, documented explicitly in `m4t_trit_pack.h` lines 71-94:

```
Trit encoding: +1 → 0b01      0 → 0b00      −1 → 0b10

XOR of two trit codes produces:
    same trit (any state)            → 0b00   popcount 0   cost 0 (agree)
    ±1 vs 0                          → 0b01   popcount 1   cost 1 (partial)
    +1 vs −1                         → 0b11   popcount 2   cost 2 (opposite)

So this function returns a ternary Hamming distance with max = 2·N trits,
not a binary Hamming distance with max = N.
```

**This is exactly the L1 path-graph distance** with 0 as the natural
center. Verified against my Python `pairwise_L1_int8` — all 9 (a, b)
pairs match exactly.

## What this means for the Phase β/γ/δ/ε arc

My Python `pairwise_hamming_int8` computes categorical Hamming
(`sum(a != b)`, giving 0/1 per cell regardless of magnitude). I called
this "Hamming-substrate" throughout the arc.

My Python `pairwise_L1_int8` computes L1 (`sum(|a - b|)`, giving 0/1/2
per cell). I called this "L1-substrate."

**Neither was the production substrate distance.** Production has been on
L1 (via `m4t_popcount_dist`) the entire time.

The "switch from Hamming to L1 for substrate eviction" recommendation
across Phases δ and ε is **empty**: production isn't on Python
categorical Hamming. It's already on L1.

## What the arc actually measured

Reinterpreted honestly, Phase β/γ/δ/ε compared:
- **A Python strawman baseline** (categorical Hamming on substrate
  signatures — what production WOULD look like if someone had made the
  wrong encoding choice, collapsing the 2-bit codes to 1-bit sign-only)
- **The production-equivalent metric** (L1 on the path graph — what the
  XOR-popcount-of-2-bit-codes kernel produces)

The "+37-62% relative reduction in attention-output L2 error from L1"
result, properly stated:

> If someone had implemented substrate distance as `popcount(XOR of
> 1-bit signs)` instead of `popcount(XOR of 2-bit codes)`, the
> attention-output quality would be 38-62% worse. The current design
> (`m4t_popcount_dist` on 2-bit codes) **was the right call**, and the
> margin is large.

This is a **validation of the substrate's existing design choice**, not
a recommendation for a production change.

## Why I missed this for an entire arc

Three compounding failures:

1. **Terminology drift.** `m4t_popcount_dist` is named after the
   implementation (XOR popcount) not the semantics (L1 path-graph
   distance). The header explicitly notes "ternary Hamming with max =
   2·N trits, not binary Hamming with max = N" — the "Hamming" label
   in the prose obscured that the math is L1.

2. **No cross-check against production semantics.** I built
   `pairwise_hamming_int8` and `pairwise_L1_int8` in Python without
   ever loading a production-packed signature and computing
   `m4t_popcount_dist` against either. Trivial cross-check, never run.

3. **The vision-claim narrative was satisfying.** "Production uses
   Hamming; we propose L1; L1 wins." A clean story. I didn't pressure-
   test the premise that "production uses Hamming."

The user's question "L1 mostly Python?" was the catalyst. Tracing what
the production substrate kernels actually compute — instead of
assuming — revealed the misalignment.

## What survives from the arc

Re-cast against the correct production semantics:

- **The vision claim is operationally instantiated.** "Base-3 IS the
  graph with 0 at center" is implemented in `m4t_popcount_dist` for
  the entire history of the substrate. The user's claim isn't just
  philosophy; it's compiled code that's been shipping.

- **The design choice of 2-bit packing is load-bearing.** Anyone who
  proposed "simplify to 1-bit sign-only encoding" would silently
  degrade substrate-eviction quality by 38-62% relative on
  attention-output L2 error. The 2-bit encoding's "guard against
  silent degradation" warning in the header (lines 87-94) is now
  empirically validated.

- **The path-graph property (P3 in Phase γ, the only ROBUST finding)
  is genuinely substrate-distinctive.** It survives because production
  USES it, not because we proposed adding it.

- **The 5-prompt long-context measurement (Phase ε, n=395,400 trials)**
  documents how much margin the design choice provides over the
  naive alternative.

- **The shuffled-K finding (ε-5)** says the path-graph advantage is
  metric-fundamental (uses sign + magnitude) rather than learned-
  structure-specific. This means the substrate's design is robust to
  changes in K-cache distribution — useful for guarding against
  regressions.

## What's reversed from prior commits

- **`d06eb48` ("Phase ε: production should use L1")** — empty
  recommendation. Production already uses L1.
- **`7196046` ("L1 cost concern reframed, 1.7-1.8× in production
  analog")** — partially correct but irrelevant: the L1 cost is whatever
  `m4t_popcount_dist` costs (NEON popcount, very fast), and there's no
  alternative Python or NumPy implementation needed.
- **`4028e05` ("RT-E: L1 8-29× slower than Hamming")** — comparing
  Python NumPy implementations of two metrics neither of which is the
  production path. Moot.
- **The production-readiness checklist in my response 2 messages ago
  ("write `m4t_route_l1_distance_batch.c`")** — that kernel **already
  exists** under the name `m4t_route_distance_batch` / `m4t_popcount_dist`.
  The 4-step list of "what to do to ship L1" is mostly already done.

## What's actually open for production

The arc was framed as if L1 needed to be added to production. It
doesn't. What's actually open:

1. **Substrate eviction quality at long context (seq_k > 64) needs
   testing** — the current substrate eviction works at production
   scale but we haven't measured the L2-error margin there.

2. **The "Hamming substrate" Python baseline doesn't correspond to any
   production component** — the Phase ε comparison is informative
   about design-choice safety but not about a switchable knob.

3. **The vision claim → production code link should be documented.**
   The fact that `m4t_popcount_dist` operationalizes the path-graph
   metric isn't in any project-level doc; only in the kernel header.
   The vision memory and the substrate-claim arc journals should
   point at this.

## Files

This journal documents the misalignment. The Python implementations
in `experiments/phase_*/` will be annotated in a follow-up to note
the equivalence to `m4t_popcount_dist` and prevent future confusion.

## Discipline log

19th caught misalignment of the arc. **Pattern:** I built measurement
infrastructure assuming a production state, never verified the
assumption against production code, and shipped an arc of findings
that re-cast cleanly once the assumption was corrected. The fix isn't
"do less measurement" — it's "verify what production actually does
before designing the measurement."

A memory entry on cross-checking research-Python against production-C
semantics would prevent this class of mistake. Adding one.
