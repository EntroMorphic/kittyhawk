---
cycle: gesh_findings
phase: CLOSEOUT
date: 2026-05-02
scope: post-synthesize observations on the LMM cycle deployed against Phase A.2 findings
companions: gesh_findings_{raw,nodes,reflect,synthesize}.md
status: COMPLETE — no loop-backs triggered yet; awaits next-cycle gate measurements
---

# Closeout — gesh_findings

The cycle ran to a clean SYNTHESIZE commitment. No mid-cycle reframings. Closeout is thin.

## What the LMM cycle revealed that the prior docs didn't

The Phase A.2 docs are technically correct: every claim is conditioned on the synthetic benchmark and the sweep parameters. The deployment of LMM against those findings surfaced two things the docs don't make load-bearing:

1. **The strongest finding (C2) has the weakest mechanism story.** The +7pp identity-vs-random gap at sig_dim=D is robust across seeds and suggestive of substrate-claim-relevant emergent behavior, but the "implicit denoising" mechanism is an unmeasured hypothesis. REFLECT named this; SYNTHESIZE committed to a cheap mechanism test (Gate 2) that resolves the question one way or the other.

2. **The synthetic benchmark's structural rigging makes mechanism comparisons weakly informative.** K-vs-(D-K) split with uniform-random noise dims is the easiest possible signal/noise structure. Compression-regime gains and expansion-saturation behavior are conditional on this structure. This was buried in the docs as a "Conditions" footer; LMM lifts it to "the load-bearing problem."

Neither of these is news to anyone who's read the data carefully. LMM's value here is making the implicit explicit, so next-cycle scoping can't drift away from the constraint.

## The reframings worth keeping

REFLECT's "wrong frame / right frame" pairs are worth carrying forward:
- "Lattice update earns +8pp" → "Lattice update earns +8pp on a benchmark where +8pp is what's left to take."
- "Random ternary has implicit denoising properties" → "Random ternary wins on benchmarks where noise dims are uncorrelated; the mechanism is hypothesized but not tested."
- "Phase A complete; ready for Phase B" → "Phase A's mechanism-validation goals met; the substrate-claim path requires real-data work that has not started."

These are not corrections of the docs — they're the conditions the docs already attach, surfaced into the foreground. Future cite-sites should use the right-frame phrasing.

## What did NOT surface

- No "the lattice IS the geometry"-level reframe. The Phase A design closeout already absorbed that observation; this cycle didn't surface a comparable structural shift. The findings cycle is normal-shaped LMM: data → claims → pressure-test → next move.
- No mechanism finding inverted. C2 still stands as a robust correlation. C1 stands as a small but real compression-regime gain. C3/C4 stand as expected expansion-regime behavior.
- No methodology error caught beyond what the Phase A.2 red-team already remediated. Multi-seed methodology held up.

## Loop-back triggers from this closeout

These are the conditions under which the next cycle should send work back to an earlier phase rather than continue forward:

- **Back to RAW** if Phase B Gate 1 (image canon parity) shows real-data behavior the synthetic findings can't anticipate — e.g., compression hurting, or expansion helping. The current node set would need new observations.
- **Back to NODES** if Phase B Gate 2 falsifies H1. C2 becomes a node without a story; documentation needs a rewrite to remove the "implicit denoising" framing.
- **Back to REFLECT** if Gate 1 inconclusive zone (90–95%, marginal gain) lands and the cycle doesn't have a clear pass/fail verdict. The reference frame may need re-pressure-testing — either the gate threshold is wrong, or the probe is asking the wrong question.
- **No loop-back** if Gate 1 passes (≥95% MNIST, ≥+2pp gain) and Gate 2 supports H1. Both findings transfer; the substrate-claim path advances to Go positions on the next cycle.

## Methodology note

This is the third LMM cycle in the post-reset codebase (`xexpo_design`, `gesh_design`, `gesh_findings`). The pattern across all three:

- RAW captures honestly; mechanism conjecture stays out of it.
- NODES forces discrete units, which forces honesty about what's claim vs hypothesis vs anchor.
- REFLECT is where the value gets created — pressure-testing the reference frame surfaces what the docs underweight.
- SYNTHESIZE has to commit to specifics with gates, or the cycle is worth nothing.

This cycle's REFLECT was generative (it produced the wrong-frame / right-frame reframings, the H1 mechanism test, the realization that more synthetic sweeping is the wrong move). That's what REFLECT is for. When REFLECT is shallow, the cycle's SYNTHESIZE just restates RAW.

The cycle that produces the strongest SYNTHESIZE is the one whose REFLECT is willing to make its own RAW observations uncomfortable. Worth recording for future cycles.

## What this cycle does not finish

- No new measurement runs. The cycle re-interprets existing data; it doesn't generate new data.
- No code changes. Build is unchanged.
- No documentation edits. The journal is the artifact.

The cycle's output is a commitment to next-cycle scope (Gate 1, Gate 2, real-data probe). Whether that commitment delivers depends on the next cycle, not this one.
