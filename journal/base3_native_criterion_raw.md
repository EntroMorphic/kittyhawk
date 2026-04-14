---
date: 2026-04-14
phase: RAW
topic: What is the right criterion for "base-3 native," and what replaces sign_extract?
---

# Raw thoughts

Two entangled questions:
1. What criterion correctly identifies whether a primitive belongs in a base-3 substrate?
2. Given that criterion, what replaces `sign_extract`?

If I get (1) wrong, (2) is wrong automatically. So (1) comes first.

---

## The criterion

I've been using informally: "three-state domain, three-way semantics, all three states occur in realistic inputs." That's three sub-criteria (C1, C2, C3). Call it **C-A**.

C-A caught sign_extract via C3 (zero state is measure-zero for MTFP projections). It passed everything else in the live surface. Did it get it right, or did it get lucky?

### Alternative formulations I've considered

**C-B: "Impossible in base-2 without explicit ternary encoding."**
Too strict. Under C-B, SDOT fails (int8×int8 is a base-2 instruction), popcount fails (bit-count, not trit-count), TBL fails (byte lookup). Most of our surface would go. That's wrong — NORTH_STAR's point is that the silicon IS binary and M4T's job is to EXPOSE the ternary semantics that live on top. The criterion shouldn't demand ternary silicon; it should demand ternary *abstraction*.

**C-C: "No state can be distinguished from another only by external context."**
For sign_extract: the zero state is distinguishable from ±1 only by the happenstance that the input was exactly zero — an external coincidence. For threshold_extract: the zero state is "input was in band |v|<τ" — an intrinsic property determined by the operation itself. C-C correctly separates them. But I can't evaluate C-C mechanically for all primitives; it requires judgment about what's "context" vs "intrinsic."

**C-D: "Commutes with balanced-base-3 representation."**
Too technical. Not all useful primitives need to commute with rep changes. Skip.

**C-E: "Zero state is an informational carrier, not just an identity or absence."**
This is the deepest test. For `vec_add(x, 0) = x`, zero acts as identity — base-2-style. For `apply_signed` with a zero-sentinel decision, zero means "skip this tile" — an active instruction, not absence. For sign_extract's zero output, it's just "input was zero," no further meaning. C-E is strict but might invalidate neutral arithmetic ops that are fine.

**C-final candidate: "If you remove any one of the three states — collapsing it into another, or treating it as a measure-zero edge case — does the primitive's utility on realistic inputs materially degrade?"**
This is a counterfactual test. Applies mechanically. Captures the sign_extract failure cleanly.

Let me stress-test C-final:
- sign_extract: collapse zero→+1. On realistic inputs, essentially no change (zero is measure-zero). FAIL.
- threshold_extract: collapse zero→+1. Lose all "weak-signal" information. PASS.
- popcount_dist: treat zero trits as equivalent to sign trits. Distance metric changes materially. PASS.
- apply_signed: treat sentinel as equivalent to +1 or -1. Lose "skip" capability. PASS.
- SDOT: treat zero trit weights as equivalent to sign trit weights. Compute a different function entirely. PASS.
- vec_add over MTFP mantissas: treat zero mantissas as equivalent to ±1 mantissas. Break the number system. PASS.

C-final gives the same verdict as C-A but is tighter and more mechanical. I think this is the right criterion.

### What could still be wrong about C-final

- It's a counterfactual test; interpretation depends on what "materially degrade" means. Ambiguous edge cases possible.
- It treats three-way-ness as a property of the operation's semantic, not its silicon. That matches NORTH_STAR but I've been wrong before — NORTH_STAR is a compass, not proof. Is there any NORTH_STAR-aligned property I'm missing?
- NORTH_STAR says "base-2 fakes the zero-state." C-final tests whether the zero-state is faked (collapsible to measure-zero). Matches. OK.

Confidence in C-final: moderate-to-high. Biggest worry is that I've missed a fourth kind of failure mode beyond "collapses states, measure-zero, faking zero."

---

## The replacement

Three candidates from earlier turns. Test each against C-final.

### Option 1: `trit_extract(values, k, n)` — k-th trit of balanced-base-3

For int64 v, its balanced-base-3 representation is unique. t_k is the k-th trit.
- Domain: int64.
- Output: packed trit {-1, 0, +1}.
- For realistic MTFP projections, does t_k take all three values? Depends on k:
  - k too small: t_k is noise (low-order bits).
  - k matched to magnitude: t_k is the leading trit, which is sign-for-large-|v|, zero-for-small-|v|. Three-way meaningful.
  - k too large: t_k is almost always zero.
- So the primitive is C-final-passing only for the right k.
- The consumer has to choose k. That's a configuration burden that moves the "is it ternary-native" decision onto the caller.

**Is this really three-way structurally?** Mathematically yes — balanced-base-3 is genuinely ternary. Pragmatically, extracting a specific trit position exposes a coarse quantization at scale 3^k.

### Option 2: `threshold_extract(values, tau, n)` — three-way by magnitude

- Domain: int64.
- Output: packed trit.
- Rule: |v| < τ → 0, v ≥ τ → +1, v ≤ -τ → -1.
- C-final: all three states are meaningful for any τ > 0. PASS unconditionally.
- Consumer burden: choose τ.

More direct than Option 1. The consumer parameterizes the zero-band explicitly. τ=0 degenerates to sign_extract — so sign_extract IS the degenerate case of threshold_extract. That tells us something: sign_extract was always a special case of a more general primitive, and we had the special case only.

### Option 3: Don't extract — mantissas ARE the signature

The signature of a projection is the MTFP4 (or MTFP9, or any cell-width) mantissa itself. 4 trits per projection dim instead of 1.

- Consumer pipeline: project in MTFP4 (or narrow MTFP19→MTFP4), pack trits, compute Hamming.
- No "extraction" primitive needed. The packing is zero-information (just reformatting).
- Distance: popcount-Hamming over packed trits of mantissas gives trit-level Hamming — lattice-geometric for three-state data.
- Narrowing MTFP19→MTFP4 has rounding (§8.5 Case R or similar). Projecting directly in MTFP4 avoids that but requires different input representation.

This eliminates the entire "extraction" primitive category. The signature representation IS the data.

### Stress-test: which of Options 1, 2, 3 is "most base-3 native"?

Under C-final:
- All three pass (when used correctly).
- Option 1 passes conditionally on k.
- Option 2 passes unconditionally (for any τ > 0).
- Option 3 passes structurally (no extraction step to fail).

Under NORTH_STAR:
- Option 3 is the deepest move — eliminates a primitive category rather than replacing a binary one with a ternary one.
- Option 2 is the most incremental — targeted replacement of sign_extract.
- Option 1 is structurally elegant but impractical (consumer has to choose k).

Under "substrate/consumer boundary" (our throughline):
- Option 2 requires consumer to parameterize τ. Consumer-side decision. OK.
- Option 3 requires consumer to quantize to MTFP4. Also consumer-side. OK.
- Option 1 requires consumer to choose k. Also consumer-side.
- All three move the decision appropriately.

Under "minimum surface" (our discipline):
- Option 3 removes a primitive rather than adds. Smallest surface expansion.
- Option 2 adds one primitive.
- Option 1 adds one primitive.

Option 3 looks strongest on most axes. But it changes the shape of the consumer significantly — the LSH tool currently produces MTFP19 projections, not MTFP4.

---

## What I notice I'm uncertain about

1. **The criterion itself.** C-final is a judgment call. Someone smarter might see a better one.
2. **The realistic-inputs test.** "Zero state is measure-zero for MTFP projections" is true for the LSH tool's projection shape. Other consumers might produce values that hit exact zero often (e.g., sign-of-count reductions, counted sparse activations). Does C3 fail those consumers too? If not, sign_extract might be useful for SOME consumers even if not this one.
3. **Whether Option 3 is really the same operation as Options 1/2.** I've been treating them as parallel alternatives, but Option 3 is structurally different — it's a non-extraction. Maybe it's a separate architectural move that can coexist with an extraction primitive.
4. **Whether there's a FOURTH option I haven't seen.** E.g., "sign-AND-nonzero" two-bit encoding, or signature-from-top-k-trit-positions, or something I'm not thinking of.

---

## What scares me about getting this wrong

The sign_extract problem hid for weeks because its type system lied (three codes, two-in-practice). If I pick a replacement that has a similar hidden failure mode, we're back here in another week. The criterion and the replacement need to be tight enough that the SAME CLASS of failure can't happen again.

C-final's counterfactual test is one defense. "Can you collapse a state without material damage?" is checkable, not vibes-based. If the replacement passes C-final cleanly on realistic inputs for the intended consumer, it's more likely to stay passing.

But — "realistic inputs for the intended consumer" is itself a moving target. When a new consumer arrives, does C-final have to be re-evaluated? Probably yes. That's a process implication.

---

## Questions arising

1. Should the substrate commit to ONE extraction primitive, or admit that different consumers want different extraction shapes?
2. Is Option 3 (no extraction) compatible with the existing routing surface (sign_extract feeding distance_batch)?
3. If we pick Option 2 (threshold_extract), what happens to `signature_update` (which calls sign_extract internally)? Does it get a threshold parameter too?
4. Does picking the replacement change what the fully-routed MNIST experiment should do next time we run it?
5. Is the criterion C-final something I should commit to in the substrate spec, or is it a meta-criterion that lives in documentation about what we put in the substrate?

---

## First instincts to watch for

- "Pick the elegant one" — Option 3 is elegant but changes consumer shape significantly. Elegance isn't automatically right.
- "Pick the incremental one" — Option 2 is the smallest delta, which is comfortable but might not be the architecturally correct move.
- "Let consumers decide by providing all three" — invites the same base-2-default failure mode we saw with sign_extract. The principle of minimum surface argues against.
- "We're not sure; let's defer" — sometimes right, but we have a known broken primitive in the live surface. Deferring keeps the broken thing.
