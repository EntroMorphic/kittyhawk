---
cycle: xexpo_design
phase: REFLECT
date: 2026-05-01
scope: find structure beneath the nodes; resolve tensions; name the core insight
---

# Reflect — xexpo_design

## Core insight (one sentence)

**The design specifies a primitive whose correctness is mathematically forced but whose *shape* is consumer-determined — and the shape might be wrong.**

The mathematics of cross-exponent alignment leaves no real choice (Path A is forced). But the *signature* of the kernel — pairwise vs accumulator, per-tensor vs per-block, with vs without sat_flags — encodes assumptions about how consumers will call it. Those assumptions haven't been measured. The design is not a substrate specification; it is a substrate *hypothesis* that pretends to be a specification.

## Asking why three times

**Why was Path A right?**
Because aligning to the smaller exponent forces the larger operand to multiply its mantissa by `3^Δ`, which overflows for any non-trivial Δ. The larger operand carries the dominant magnitude; corrupting it is catastrophic. Path A preserves the dominant magnitude.

**Why does this matter for the design (vs being a routine engineering choice)?**
Because Path A's named cost — smaller operand truncates to zero when Δ is large — propagates into every consumer that uses this kernel. The cost is *inherent*, not implementation-detail. If a consumer cannot tolerate the loss, the consumer's architecture must change, not the kernel's.

**Why does this turn the design into a hypothesis rather than a specification?**
Because we don't yet know whether the consumers' use patterns tolerate the loss. The design's claim "matches IEEE-754 semantics" is true but does not address whether IEEE-754 semantics match what the multi-table SUM resolver actually wants. The design encoded an assumption (consumers will tolerate small-vanishes-when-Δ-large) that has not been tested against any consumer.

## Structure beneath the nodes

The nodes fall into three buckets — exactly the laundry-method partition:

### Bucket A: Forced by mathematics (no consumer dependency)
- Node 1: Path A vs B → Path A
- Node 11: IEEE-754 framing is the geometry, just stated in base-2 vocabulary
- Node 6 (partial): the half-trit error bound is derivable from integer truncation, even if I derived it sloppily

These survive the cycle regardless of what the consumer measurement says. They are substrate facts.

### Bucket B: Consumer-shape decisions (decisions that the cycle must validate)
- Node 3: pairwise vs accumulator API — *load-bearing*
- Node 4: per-tensor vs per-block granularity
- Node 8: sat_flags layout / aggregate vs per-cell
- Node 5: out-param vs derived (becomes moot under accumulator API)
- Node 9: NEON cost (consumer-determined: hot-path or not?)

These are the design's hypotheses. The cycle's job is to test them against real consumer call patterns, not to confirm them.

### Bucket C: Discipline / methodology questions
- Node 10: design ahead of measurement — discipline violation or healthy exploration?
- Node 12: design as spec amendment vs spec as design constraint
- Node 7: assertion vs graceful degradation at Δ=19

These are not about the kernel; they are about *how the rebuild should run cycles*. They have answers, but not in the kernel's design.

### The boundary items (where mistakes hide — per the Laundry Method)

**Node 3 sits on the bucket-A/bucket-B boundary.** The choice between pairwise and accumulator looks like a consumer-shape decision (B), but it might be mathematically forced once we examine the consumer's actual operation. If multi-table SUM is `for t: sum += distance(query, table[t])`, then alignment between `sum` and `distance` happens once per accumulation — and the running `sum`'s exponent might need to migrate as new distances arrive. *That's not pairwise add anymore; it's a fundamentally different shape.*

This is the boundary where the design failed to look carefully. The original design treats Node 3 as a "future work" extension; reflection says it's the load-bearing primitive choice.

## Resolving the tensions

### T1: Path A's vanish property vs consumer precision needs (Nodes 1, 2)
**Resolution:** Path A is not negotiable. Whether the consumer tolerates the loss is the cycle's measurement question, not the design's. If the cycle says NO, the consumer's architecture is the thing that changes — not Path A. (The "rage against the trodden" rule applies: don't bend the substrate to fit a base-2-comfortable consumer pattern.)

### T2: Pairwise vs running accumulator (Node 3) — **the load-bearing tension**
**Resolution:** The current design (`vec_add_aligning`) is provisional. The cycle's instrumentation must record whether consumers naturally call this kernel in a pairwise pattern (one shot, two operands) or an accumulator pattern (running sum, many shots). The accumulator API:

```c
/* Hypothetical, to be validated by the cycle */
void m4t_mtfp_vec_accum_aligning(
    m4t_mtfp_t* running, int8_t* running_exp,  /* in-out */
    const m4t_mtfp_t* new, int8_t new_exp,
    uint8_t* sat_flags,
    int n);
```

This kernel migrates `running_exp` upward as needed; it never migrates downward (which would lose precision). The pairwise kernel falls out as the special case where `running_exp = a_exp` and we discard the in-out semantics. Building only the accumulator is plausible; building only the pairwise is *not* plausible if accumulation is the natural pattern.

### T3: Per-tensor vs per-block granularity (Node 4)
**Resolution:** Per-tensor for the MVP. Per-block becomes its own kernel only if a *future* consumer surfaces a tensor whose internal scale legitimately varies across blocks. The substrate spec §7's per-block intent is preserved as the design space; the MVP picks the slice that the cycle's named consumers actually need.

### T4: Half-trit error bound (Node 6)
**Resolution:** Derive the bound rigorously, don't justify by analogy. For Path A with integer truncation:
- Smaller operand mantissa `m_s` divides by `3^Δ` to produce `m_s / 3^Δ` (truncated toward zero).
- Truncation error: at most `(3^Δ - 1) / 3^Δ` of one mantissa unit at `e_d`.
- In real numbers at `e_d`: error `< 3^e_d` per cell.
- Sum-of-two-aligned-operands error: at most `2 × 3^e_d`. But only the smaller-operand contributed truncation, so error `< 3^e_d` per cell.

So the natural bound is `1 × 3^e_d`, not `3^(e_d − 1)`. The design's bound was *too tight by a factor of 3*. The property test would have rejected legitimate kernels that produced errors in `[3^(e_d−1), 3^e_d)`. **This is a real bug in the design's test specification.**

### T5: Out-param vs derived (Node 5)
**Resolution:** Becomes moot if Node 3 resolves to accumulator API (running_exp is necessarily in-out). If pairwise survives, the derived approach is fine; the helper `align_exp` documents intent.

### T6: `|Δ| ≤ 19` assertion (Node 7)
**Resolution:** Soften to documented "at Δ ≥ 19, the smaller operand truncates to zero by the math; the operation is well-defined but degenerate." Drop the hard assertion. Consumers that genuinely produce Δ ≥ 19 will get correct (if uninformative) results. The hard assertion adds no value; it just rejects valid inputs.

### T7: sat_flags layout (Node 8)
**Resolution:** Defer the layout decision until the cycle measures saturation rate. If <0.1%, a single counter is sufficient. If >1%, per-cell flags pay their cost. The cycle's protocol already tracks saturation rate; the design just needs to wait.

### T8: NEON cost (Node 9)
**Resolution:** The cycle measures whether this kernel is in a hot path on any consumer. If no, scalar is the answer. If yes, NEON design is its own cycle. Honest framing in the design, not concealment.

### T9: IEEE-754 framing (Node 11)
**Resolution:** Replace "matches IEEE-754" with "alignment-to-larger preserves dominant magnitude — this is the only positional-arithmetic choice consistent with not catastrophically saturating the larger operand." Stating the geometry directly. IEEE-754 reference becomes a parenthetical historical anchor, not the justification.

### T10: Design as spec amendment vs spec as constraint (Node 12)
**Resolution:** Re-read `M4T_SUBSTRATE.md` §14.2 *before* the design memo phase of tier 3c. The current design treats §14.2 as a sketch to replace; reflection says it might be a constraint we missed. This is straightforward to do.

### T11: Discipline question (Node 10)
**Resolution:** The design is exploration ahead of measurement, not violation. The discipline test is "did we ship a built thing whose existence biases future decisions toward use?" — and the answer is "no kernel exists; the design is a critique target, not a momentum vector." But the answer becomes "yes, violation" if we let the design's vetted-ness substitute for the cycle's measurement. The cycle must test *the design's hypotheses*, not just measure the kernel's own correctness.

## Hidden assumptions surfaced

1. **Consumers will produce heterogeneous block_exp values.** Untested.
2. **The natural call pattern is pairwise, not accumulator.** Reflected as wrong; load-bearing for the design.
3. **Smaller-operand truncation is acceptable to consumers.** Untested.
4. **Saturation matters enough to track per-cell.** Untested.
5. **The kernel is not in a hot path.** Untested.
6. **§14.2's existing prose is provisional and the design supersedes it.** I haven't re-read §14.2 to verify.
7. **The half-trit error bound is correct.** **REJECTED by reflection.** Bound is `3^e_d`, not `3^(e_d − 1)`.

## What I now understand

The design is a useful exploration. Two of its components are *correct independent of the cycle*: Path A choice and the geometry-not-IEEE-754 framing. Several components are *consumer-dependent hypotheses*, properly framed, that the cycle will test. One component — the pairwise-vs-accumulator API choice — is a load-bearing decision the design quietly assumed and that needs the cycle's evidence before commit. One component — the half-trit error bound — is *just wrong* and would have caused the property tests to reject correct kernels.

The reflection has changed the cycle's job. The cycle is not just "does any consumer pay a cost?" — it is also "does the consumer's call pattern match the design's hypothesized API shape?" Both measurements feed into whether the kernel earns its design and what the design *should be*.

## Remaining questions

- If the cycle reveals accumulator semantics are right, does the original `vec_add_aligning` kernel still ship as a degenerate case, or does only `vec_accum_aligning` ship?
- The error bound is now `3^e_d` per cell. What's the bound for accumulator semantics across N consecutive accumulations? Naive bound: `N × 3^e_d`. Is that acceptable, or does accumulator need a tighter contract?
- How aggressively should the cycle hunt for tensors whose internal scale varies across blocks? If found, that's a third kernel (per-block aligning add); if not, the per-tensor MVP holds.
