# REFLECT: methodical falsification of the R1 claim

Cold-eye review of `r1_falsify_nodes.md`.

## Load-bearing nodes

- **N2** — R1's claim restated as a falsifiable proposition. This is the WHOLE cycle. If the proposition is mis-stated, the falsification doesn't reach the underlying claim.
- **N5, N7** — prior FAIL evidence. Documented; verifiable by reading the closeouts. Strong.
- **N8 through N12** — the 5-axis falsification matrix. Three axes (F-A2, F-A3, F-A4) already have falsifying data; one (F-A1) weakly supports; one (F-A5) is missing. The cycle needs to fill F-A5 and cleanly tabulate.

## Weak nodes

- **N8 (F-A1 weakly supports)** — dual produces ~40% more equivalence classes. This is real but ambiguous. Could be: (a) dual carrying genuinely more discriminative info; (b) dual fragmenting arbitrarily without correlating to any meaningful structure. Without a quality measure, "more classes" is just "more partitions" — could be noise. Needs a follow-on interpretation.
- **N12 (F-A5 missing)** — held-out routing accuracy. Construction is non-trivial: requires labeled training expressions, DIFFERENT held-out expressions with known equivalence to training classes. The expr_routing probe binaries already do something like this. Need to inspect.
- **N18 (no consumer-demand framing)** — easy to slip into "is there a consumer for the routing accuracy?" The user has explicitly disclaimed this for foundational research. But this cycle is testing a CLAIM, not building a primitive — different shape. Should re-read the rule and check if it applies.

## Tensions

- **N15 (verdict logic)** vs **N22 (results may be mixed)**: my pre-commit verdict logic says "R1 falsified iff F-A2, F-A3, F-A4, F-A5 all fail to support; F-A1 may weakly support but doesn't constitute SUPPORT." That's conservative — sets a high bar for "supported" (4 of 5 axes need to falsify, 1 may weakly support without changing verdict). What if 3 falsify, F-A5 PASSes weakly, F-A1 passes weakly? Then 2 axes weakly support; 3 falsify. Verdict?
  - Resolution: pre-commit each axis's individual gate clearly. The cumulative verdict is "falsified" if a STRICT majority of relevant axes (i.e., F-A2, F-A3, F-A4, F-A5 — the four substantive axes; F-A1 is weak by nature) falsify. If 3+ of those 4 falsify: methodically falsified. If 2-2 split: ambiguous, needs more axes. If 3+ support: claim has merit, prior FAIL was misframed.
  - Pre-commit: 4-axis verdict. F-A1 reported separately as a curiosity, not a load-bearing axis.

- **N21 (R1 is one expression of claim 3) vs the user's instruction "falsify the R1 claim"**: the user said R1, not "vision claim 3 generally." So scope is R1 specifically. The cycle's CLOSEOUT should explicitly note that falsifying R1 doesn't falsify claim 3 broadly — leaves room for future expressions of claim 3 to be tested.

## Missing information

- **M1.** What does F-A5 actually look like in code? The expression sets in `gesh/bench/expr_routing_*.c` need inspection. Is there already a held-out test pattern, or does this need new construction?
- **M2.** The prior measurements are reproducible — but what command/script? Need to identify which probe binary produces F-A2, F-A3, F-A4 numbers and verify it builds + runs cleanly.
- **M3.** Are there any code changes since R1 was archived that could affect the measurements? The R1 expression-routing code (`gesh/src/expr*.c`) probably hasn't been touched. Verify.

## What I'd want before declaring "ready to execute"

In rough priority:

1. **Inspect `gesh/bench/expr_routing_*.c`** — find the probe binaries that produce R1's measurements (F-A2, F-A3, F-A4). Run them to verify they still produce the prior numbers.
2. **Sketch F-A5** concretely. What's the test? What's the data? Can it reuse existing expression sets or does it need new ones?
3. **Pre-commit gates per axis** with explicit numerical thresholds for "supports R1" vs "fails to support R1" vs "ambiguous."

## What I might be wrong about

- **The "5 axes is enough" framing.** Falsification can always be extended (more axes, different axes). The user said "methodically" not "exhaustively." 5 axes spans signature properties, partition behavior, substrate-novelty, AND routing performance — that covers the bases. But there could be a 6th angle I'm missing.
- **F-A1 (more classes) interpretation.** I'm calling it weakly supportive. It might actually be NEUTRAL (more classes ≠ better discrimination unless paired with intra-class consistency). For methodical purity, F-A1 should be paired with an intra-class consistency check.
- **Whether replication is needed at all.** If I trust prior measurements, just adding F-A5 is sufficient. But "methodical" suggests verifying the prior data is still valid under current code. Cheap insurance.

## Honest framing

The verdict is essentially predetermined by prior evidence. The cycle's contribution is:
1. Verify prior data still holds (replicate F-A2, F-A3, F-A4)
2. Add the missing axis (F-A5 held-out routing accuracy)
3. State the verdict cleanly, axis by axis

If F-A5 falsifies (likely given prior trends), R1 is methodically falsified across 4 of 5 axes (F-A1 reported as curiosity). Outcome: clean verdict.

If F-A5 surprisingly supports R1, the verdict is mixed: signature-property axes fail, routing-performance axis supports. Honest result; would need follow-on to interpret.

Either way, the cycle moves R1 from "informally archived" to "methodically falsified with audit trail."

## Methodology check (against project rules)

CONTRIBUTING rules I should apply:
- **Substrate-novelty audit**: applies to F-A4 directly.
- **Multi-seed validation**: F-A2/F-A3/F-A4 should be multi-seed (prior cycle was 5 seeds). Apply same to F-A5.
- **Hypothesis vs finding**: each axis should produce a verdict, not a hypothesis. Numerical gate, observed result, PASS/FAIL on supporting R1.
- **Multi-config gates the story**: F-A5's setup has multiple knobs (sig_dim, expression set, test input set). At least 2 configs to gate the interpretation.
- **Match scope of evidence to scope of claim**: the claim is "R1 better than sign-only on at least one axis." 5 axes is the scope. Verdict per axis; cumulative across.

The methodology is already in CONTRIBUTING. I just need to apply it.
