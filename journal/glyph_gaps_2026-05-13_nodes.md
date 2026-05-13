# NODES — Glyph gaps

Extracted from `glyph_gaps_2026-05-13_raw.md`. Distinct gaps,
numbered. Tensions and dependencies marked.

## Node 1: Foundational claim 1 — primitives floor is missing exp/log
The vision names ~6 frozen primitives. Substrate has trit ops + MTFP
add/sub/accumulate. Exp/log aren't in. No work is queued to add them.
No experiments test whether the current primitive set closes over the
math the project needs.
**Why it matters:** Without exp/log, claim 1 is structurally incomplete.
Softmax/log-likelihood/anything requiring exponential decay can't be
expressed by routing alone over the current floor.

## Node 2: Foundational claim 2 — math-as-signatures-via-routing has no bridge
The vision says "any math expression has a derived signature."
Currently signatures derive from DATA (threshold extraction of K, Q
vectors). There is no code path that takes an expression like
`f(x) = x² + 3x` and produces a signature. Claim 2 has zero
measurement infrastructure.
**Why it matters:** Claim 2 is the load-bearing connection between
the substrate (claim 3) and the primitives floor (claim 1). Without
the bridge, the three claims sit in isolation rather than as a system.

## Node 3: Phase ζ retraction is unfinished — qsigdist needs N=50-100
The substrate-eviction territory verdict is "inconclusive with
positive trend" (qsigdist +6pp vs random, CI [-5.6, +18.1]). 20
prompts is underpowered. Without more data, the substrate-claim arc
ends ambiguously.
**Why it matters:** Either commit to settling it (more prompts) or
explicitly accept "inconclusive" as the closing position. Letting it
rot is the worst option.

## Node 4: c_dump_v3 prompt provenance is unverified
Phase α through ε oracle numbers all use c_dump_v3 activations. The
prompts that generated those activations are not in version control;
their content (natural language or gibberish) is unknown. If
gibberish, the entire single-shot oracle is on OOD data.
**Why it matters:** This invalidates the OUT-OF-SAMPLE basis for the
"L1 beats Hamming 38-62%" claim if c_dump_v3 turns out to be from
gibberish prompts (as the original 5-prompt harness battery was).

## Node 5: Journal navigability — 11+ journals with internal contradictions
Phase α/β/γ/δ/ε/ζ + plan A red-team + plan B + plan B red-team is
11+ journals in 36 hours. Multiple contain claims that subsequent
journals retract or weaken. A reader landing in 2026-06 will
struggle to know what's current.
**Why it matters:** Loss of institutional memory. Future work
(possibly mine, possibly user's, possibly a successor's) needs a
single authoritative "current state" map.

## Node 6: README / CONTRIBUTING scope updates may be stale
README.md and CONTRIBUTING.md were edited on 2026-05-12 with
"foundational primitives exempt from consumer-demand rule" framing
and "sparse-attention production-eligible" notes. The Phase ζ
retraction on 2026-05-13 may invalidate or weaken some of those
framings.
**Why it matters:** Public-facing scope claims should match
internal state.

## Node 7: qsigdist code has no unit tests
Plan B added `BITNET_KV_EVICT_QSIGDIST` mode to `bitnet_harness.c`
including a new code path in `bitnet_kv_evict_pick_victim`. The
harness output is plausibly correct but no unit test pins the
bit-level behavior. A future refactor could silently break it.
**Why it matters:** Standard hygiene. Also: if qsigdist scales to
N=50 and becomes load-bearing, untested code carrying load is bad.

## Node 8: Eviction work has crowded out the rest of the project
The substrate-claim arc has consumed most recent cycles. Other
foundational gaps (Node 1, Node 2) have received zero attention.
**Tension with Node 3:** Node 3 wants more eviction cycles. Nodes
1/2 want a pivot.
**Why it matters:** Path-of-least-resistance argues for continuing
eviction; foundational coverage argues for pivot.

## Node 9: Substrate-eviction is a corollary, not a foundational claim
The vision's three claims do NOT include "substrate is useful for
KV-eviction." That's a USE CASE for claim 3's path-graph metric.
Validating eviction doesn't validate the foundational claim — it
validates one application.
**Why it matters:** Even a clean qsigdist win at N=100 doesn't close
the foundational claim. Claim 3's STRONGEST tests are still ahead.

## Node 10: Hard-coded token IDs may be wrong elsewhere
The 5-prompt eviction battery used token IDs from a different
tokenizer than the model expects. There may be other places in the
codebase with the same bug. Hasn't been audited.
**Why it matters:** Silent OOD-input bug class. Cheap audit.

## Node 11: Generation quality at window=16 maxes at 57% match
On 20 natural-language prompts, even the best eviction policy
(qsigdist) only matches no_evict on 57% of generated tokens. That's
a low ceiling. Either no_evict is the gold standard (and 43%
divergence is the cost of any eviction), or no_evict itself is not
that coherent and match-rate is a weak metric.
**Why it matters:** Need to verify whether no_evict generations are
high quality. Match-rate is meaningless if the baseline is also bad.

## Node 12: Stale memory entries reflect superseded findings
Several memory entries (`feedback_proxy_to_territory_pattern`,
`feedback_validate_input_before_mechanism`) were written during the
unstable substrate-claim arc. They may overcorrect or undercorrect
relative to the final retracted state.
**Why it matters:** Memory drives future-conversation behavior. Out
of date = wrong behavior baked in.

## Tensions and dependencies

- **T1: Eviction completion vs foundational pivot (Node 3 vs Nodes 1/2).**
  Settling the eviction trend takes ~1 day of harness time. Starting
  claim 2's bridge takes longer with less certainty. Pure resource
  competition.
- **T2: Trust in oracle measurements (Node 4 affects Nodes 3, 9).**
  If c_dump_v3 is gibberish, the oracle numbers Phase ε used to
  motivate qsigdist are on OOD data. The retraction may reach
  further than I thought.
- **T3: Test coverage vs cycle speed (Node 7 vs Node 3).**
  Adding qsigdist tests delays the N=50 battery. Skipping them
  ships untested production code if qsigdist eventually scales.
- **T4: Vision claims vs corollaries (Node 9).**
  Even completing Node 3 doesn't move the foundational claim. The
  foundational test for claim 3 is something else (path-graph
  showcase beyond eviction).
- **T5: Documentation accuracy vs work-in-flight (Nodes 5, 6, 12).**
  Journals/README/memory will keep being out of date as long as work
  is in flight. Some "audit and update" step is needed periodically.

## Emerging solution paths

- **A. Settle eviction first, pivot second.** Run N=50-100 battery,
  get statistical resolution, then pivot to claim 2's bridge.
- **B. Pivot now, leave eviction inconclusive.** Accept "inconclusive
  with positive trend" as a closing position; start claim 2.
- **C. Both, parallel.** Kick off the long battery as background;
  use foreground cycles on claim 2.
- **D. Audit first.** Verify c_dump_v3 provenance, audit token-ID
  usage, update README/memory, then choose A/B/C.
