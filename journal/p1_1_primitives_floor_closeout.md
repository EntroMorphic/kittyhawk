# Closeout: P1-1 Primitives Floor — superseded by elemental-floor reframing

## What happened

The cycle ran RAW → NODES → REFLECT → SYNTHESIZE on the question "Path A or Path B for adding exp/log to the substrate." SYNTHESIZE fired a loop-back trigger when it discovered the substrate has no division, so Taylor exp/log aren't buildable from the existing six.

After that loop-back, owner conversation surfaced two reframings:

1. **The "no consumer demand" framing** that the cycle's RAW/NODES had imported was wrong for foundational research. Owner directive: *"YES it is wrong for research! Eliminate the consumer barrier with extreme prejudice!"* (saved to memory at `feedback_no_consumer_barrier.md`).

2. **The elemental floor isn't what the cycle assumed.** Owner pushed: *"Mul, if made of two conditions, is composite."* Walking the analysis through with that pressure: **mul is composite** (iterated conditional-add via shift). **div is composite** (long division via iterated conditional-sub). **exp, log, max, min, eq, sub, abs all composite.** The truly elemental floor is 4–5 ops: add, neg, shift3, sign, select (or 4 if sign+select fuse).

The original P1-1 question ("how to add exp and log") was the wrong question. The right question is: **what is actually elemental, and what's missing from the substrate?**

## Disposition

- This cycle is **superseded**, not failed. The loop-back trigger functioned correctly — it surfaced that the question itself needed re-framing.
- The four LMM artifacts (raw/nodes/reflect/synthesize) stay in the journal as historical record.
- New cycle: `journal/elemental_floor_*.md` (next).
- The "no consumer demand" objection that contaminated this cycle's reasoning has been retired in memory.

## What carries forward

- The recognition that **iteration is not an operation, it's program structure.** Anything derivable by iterating elemental ops is composite, regardless of whether the substrate provides a fast kernel for it.
- The pattern of cheap-test-first to distinguish paths still applies; it just gets applied to the right question now.

## Status

CLOSED — superseded by `journal/elemental_floor_*.md`.
