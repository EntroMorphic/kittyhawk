# RAW — Glyph gaps as of 2026-05-13

## Stream of consciousness

The substrate-claim arc just had its final reversal retracted. The
arc consumed maybe 11 journals in 36 hours: Phase α through ζ plus
plan A red-team, plan B, plan B red-team. The territory verdict
for substrate eviction is now "inconclusive with positive trend"
(qsigdist +6pp vs random at N=20, CI straddles zero). I want to
believe the substrate claim is back on the table, but with so many
reversals the right honest position is **the measurement
infrastructure for this single claim is far ahead of its actual
substantiation**.

Meanwhile the vision names THREE foundational claims and I've been
testing one of them (claim 3, base-3 IS the graph). Claims 1 and 2
have basically no measurement infrastructure:

- Claim 1 (six frozen primitives floor): the project has trit ops +
  MTFP add/sub/accumulate. Missing exp/log per the vision memo.
  Nobody is working on exp/log. No experiments testing whether the
  current primitive set actually closes over the math the project
  needs.
- Claim 2 (math as signatures via routing): no bridge from math
  expressions to signatures exists in code. The vision says "any
  math expression has a derived signature." Without a bridge, claim
  2 is just words.

This bothers me. The eviction work is shiny because there's a
production code path and a measurement loop. The other foundational
gaps don't have a measurement loop at all, so they don't generate
work that LOOKS like progress. Path of least resistance is to keep
running variations on eviction. That's not the same as the project
actually moving forward.

Other open items I keep noticing:

- The c_dump_v3 prompts whose origin I never verified. They may be
  natural language OR gibberish. The Phase α through ε oracle
  numbers all use them. If they're gibberish, the oracle's "L1 beats
  Hamming by 38-62%" is on OOD data and may not generalize.
- README.md and CONTRIBUTING.md got scope updates on 2026-05-12.
  Phase ζ's retraction on 2026-05-13 may invalidate those updates.
  I haven't reread them since the retraction.
- Plan B added qsigdist code in `bitnet_harness.c`. There are no
  unit tests for it. The C kernel works (verified by harness output)
  but bit-level correctness is not pinned down.
- The substrate-claim arc has produced 11+ journals. They have
  internal contradictions (Phase ζ said sigdist≈random; planB redteam
  partially retracted that). A reader coming fresh will struggle.
- Multiple memory entries reflect superseded findings. e.g.,
  `feedback_proxy_to_territory_pattern` was written when plan B's
  loss claim was thought to be load-bearing.
- The "no scalar in production" rule was upgraded to apply to
  `m4t_mtfp_attn_v_combine`. That's solid. But other production
  paths haven't been audited for the same rule recently.
- The harness battery infrastructure assumes a tokenizer that was
  silently wrong. How many other places in the codebase use
  hard-coded token IDs that may be from the wrong tokenizer? Quick
  grep would help.
- Generation quality on natural-language prompts at window=16 hits
  only 51-57% match-rate vs no_evict for ALL non-no_evict policies.
  That's a low baseline. Is the harness even producing coherent
  generations to begin with? I assumed yes from spot-decoding 1-2
  prompts. Should look at more.

## Questions arising

- What's the *next falsifiable claim* in the substrate-claim arc?
  "qsigdist's +6pp trend reaches significance at N=50" is one. Are
  there others?
- For claim 1 (primitives floor), what's the minimal experiment
  that would tell us whether the current primitives close over the
  math we need? Some kind of expression coverage test?
- For claim 2 (math-as-signatures), what would even a toy bridge
  look like? Take a tiny expression "x + y * z" → derive a signature
  for it deterministically. Does such a derivation exist? Where?
- Is the c_dump_v3 corpus the right oracle data, or should I
  regenerate it from natural-language prompts to be sure?
- Should I sunset some of the older substrate-claim journals now
  that they've been superseded? Or annotate them with retractions?
- The vision memo says "categorical Hamming destroys the alphabet's
  graph structure and is not a valid test of (3)." Are there places
  in the codebase still using categorical Hamming in ways that
  pretend to test claim (3)? I haven't audited.

## First instincts

- The eviction work has a clear next step (scale N to 50-100
  prompts, settle the +6pp trend). Do it. Don't let the
  inconclusive result rot in CI.
- Then deliberately pivot away from eviction for a cycle. Pick
  ONE other foundational gap (probably claim 2's bridge, since
  it's the most structurally weighty) and produce its first
  measurement loop.
- Audit the journals for outdated claims. One short "current
  state" doc that points at the right journals would help.
- Verify c_dump_v3 provenance. 10-minute cost; resolves a hanging
  uncertainty.
- Test coverage on qsigdist. The code was added with no tests.

## What scares me

The eviction arc has consumed a huge fraction of recent cycles for
a question that may turn out to be tangential. The "substrate is
useful for KV-eviction" claim isn't actually one of the three
foundational claims — it's a USE CASE for the path-graph metric.
Even if qsigdist beats random by 6pp robustly, that doesn't
validate the foundational claim — it just validates one application.

I'm scared that I've been measuring increasingly-narrow corollaries
of claim 3 while claims 1 and 2 sit untouched. The user has been
patient through all the reversals but the project's center of mass
hasn't moved.
