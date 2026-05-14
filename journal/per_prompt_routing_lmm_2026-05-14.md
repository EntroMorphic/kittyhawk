---
cycle: per-prompt routing reframe LMM
phase: ALL (raw + nodes + reflect + synthesize)
date: 2026-05-14
scope: synthesize the per-prompt routing finding (oracle +22.96pp, CI
       [+12.83, +20.42] over qsigdist; entropy of win distribution
       0.902 of max) into a coherent picture and a principled next step.
       The mean-Δ ceiling I just shipped a closeout journal on is
       broken at the per-prompt level. What does this MEAN, and what
       should we do?
companions: per_prompt_routing_reopen_2026-05-14.md (the journal entry);
            meta_routing_arc_closeout_2026-05-14.md (the shipped
            closeout this partially overturns); meta_routing_arc_lmm_
            2026-05-14.md (the LMM cycle on the now-incomplete arc).
---

# Per-prompt routing reframe — LMM cycle

## RAW

What I actually think, unfiltered:

I committed and pushed a closeout to origin/main 15 minutes ago
saying "qsigdist is the family ceiling, empirical claim refuted,
two transferable findings, ship the architecture lessons." Then
Tripp asked one question and the closeout was partially wrong. The
arc isn't closed; the empirical question wasn't even the right
question; the +22.96pp oracle headroom is enormous and I had been
treating it as zero by virtue of always asking about mean Δ.

The thing that bothers me most: I had the data. The per-prompt
match-rates exist in every result dir. I never looked at them
because "mean Δ vs random" was the headline number from the N=100
closeout and I treated it as the metric. I never asked "is the
mean hiding per-prompt structure" — and the answer is yes, by a
factor of three.

Tripp's question was simple. "Would any of the other ideas we
have explored enhance it vs compete with it?" The framing is
load-bearing. I had been thinking competitively (rank policies on
mean Δ; pick the best) when I should have been thinking
combinatorially (which policy wins on which prompt; can we
combine them). The kernel + wall architecture I built has the
right epistemic shape for either question — I just pointed it at
the wrong target.

"Entropy is structure" was the operationalization. The win
distribution entropy at 0.902 of max — 11 policies, none winning
more than 25% of prompts — is the data signature of "real
per-prompt variance, route on it." If qsigdist won 90% of prompts
the entropy would be near 0 and routing would be dead. The fact
that even death-zone policies win on 4–6% of prompts (so they
aren't bad globally, they're specialized) is the kind of finding I
should be looking for on every problem, not the one Tripp had to
explicitly cue me to.

The closeout's "anisotropic refinement overfit" lesson was
correct and remains correct. The "structural separation works"
lesson is also correct. But the headline empirical conclusion —
"qsigdist is the ceiling, no champion in this family" — was
wrong at the level that actually matters for the substrate's
inference quality. A per-prompt router over the family could
theoretically achieve 3.6× qsigdist's gain over random.

What I'm avoiding looking at:

- I should have run this analysis BEFORE the closeout. The data was
  already on disk. The closeout was premature. Tripp's reframe
  caught it; without him, the wrong verdict would now be shipped.
  This is a pattern: I declare verdicts based on the metric in
  front of me without asking whether the metric is the right
  metric. Same family as the "spot-check before verdict" memory
  already saved.

- The +22.96pp oracle is OPTIMISTIC. It's selection-on-max over 11
  policies per prompt, which inflates the apparent ceiling. With
  per-prompt N=1 (one generation per prompt per policy), some
  "wins" are 1-token-noise lucky matches. A held-out
  cross-validation oracle (train on 50, test on 50) would be
  tighter, probably 60-80% of the +22.96pp. Even at 60%, it's
  +13.8pp over qsigdist — still massive.

- I haven't even checked whether the per-prompt winners cluster
  meaningfully. The journal entry asserts "technical prompts
  favor different cells from dialog or long-form" but I haven't
  verified that. It might be true; it might be the per-prompt
  noise creating apparent structure. The cluster analysis is
  read-only on the data I just generated; should run it before
  proposing a learned router.

- "Entropy is structure" — I read this as "high entropy of win
  distribution = routable signal." But there's a complementary
  reading: PER-PROMPT entropy of match-rates across policies
  tells us how DECIDED each prompt is. The mean per-prompt
  entropy (3.245 bits, near max 3.459) says most prompts are
  HIGHLY CONTESTED — no policy dominates the match-rate column.
  This is good news for routing (the second-best policy is
  usually nearly as good as the best, so misclassification has
  bounded cost) AND bad news (the win signal is noisy at the
  per-prompt level).

- The match-rate metric itself measures match against `no_evict`.
  That's the oracle baseline for "what would the model have
  generated with full KV." A policy that maximizes match-rate is
  best at preserving the no-evict generation. But coherent
  generation might be possible without bit-exact match to
  no-evict — and the per-prompt winners might be picking up
  flickering near-matches rather than systematic fidelity. This
  is the same issue as "coherence over bit-parity" already in
  memory, but applied to the routing target.

## NODES

Extracted tensions and constraints:

**N1 — Mean Δ vs per-prompt Δ are different metrics.** I conflated
them in the closeout. Mean Δ ranks policies in expectation; per-
prompt Δ is the loss signal a router would optimize against. They
agree on policies that uniformly dominate (none here); they
disagree on specialized policies (the death zone). The arc was
testing the wrong one.

**N2 — Oracle Δ is upper-bounded by selection bias.** Picking
max over 11 policies per prompt overestimates the realizable
ceiling. Some "wins" are 1-token lucky matches. Held-out CV
oracle would be tighter.

**N3 — Per-prompt match-rate has 1/24 granularity.** N=24 gen
tokens means single-token differences are 4.2pp of match-rate.
Wins of "1 more token matched" are inside measurement noise.
Multi-seed measurement would distinguish genuine specialization
from token-noise lucky matches.

**N4 — High per-prompt entropy is double-edged.** Mean per-prompt
H = 3.245 bits / 3.459 max says most prompts are highly contested.
Good: low cost for misclassifying a router. Bad: the winning
signal is noisy at the per-prompt level. The router would need to
exploit small but systematic differences.

**N5 — "Death zone" policies aren't categorically bad — they're
specialists.** meta(0,-1,0) at mean −13.88pp wins on 4 prompts;
meta(1,-1,0) at mean −13.67pp wins on 2 prompts. They fail
catastrophically on most but succeed on a few specialized cases.
The arc dismissed them; per-prompt analysis rehabilitates them.

**N6 — Premature closeout pattern.** I shipped a journal verdict
before running an analysis I could have run on the existing data.
Tripp's reframe caught it. Same pattern as "spot-check before
verdict" already memorized — declaring a verdict based on the
metric in front of me without asking whether the metric is the
right metric.

**N7 — Architecture is unchanged; target is.** Layer 3 (kernel
retrieval + structural wall) was predicting mean Δ per cell. To
serve routing, it should predict per-prompt winning policy from
prompt features. Same architecture, different loss signal, different
input space (per-prompt features instead of trit coordinates).

**N8 — "Entropy is structure" generalizes.** The slogan applies
beyond this specific analysis. Whenever I have a population of
policies/programs/predictors and I'm tempted to rank them by
mean, I should ALSO measure the entropy of who-wins-what to see
if there's routable structure. This is a discipline, not just an
observation.

## REFLECT

Structure, assumptions to challenge, leverage points:

**Structure.** Two layers of "is there a champion?" that I
conflated:

1. **Single-policy champion**: one cell in the 27-cell family
   uniformly best. *No* — qsigdist wins on 25% of prompts;
   nothing wins more.

2. **Router champion**: a mechanism that picks the right cell
   per prompt achieves headroom. *Yes, large* — oracle Δ +22.96pp,
   CI [+12.83, +20.42]pp over qsigdist.

The arc's empirical claim was about #1 (refuted). The interesting
empirical claim is #2 (open, with strong oracle evidence for
non-trivial headroom).

**Assumptions to challenge:**

1. **"Mean Δ is the right scoring metric."** It isn't, when the
   underlying population has high per-policy specialization. The
   right metric depends on the use case: if a single policy is
   chosen for all prompts (a constant policy), mean Δ is correct.
   If a router can choose per-prompt, mean Δ is misleading and
   per-prompt-win-distribution + oracle ceiling are the right
   targets.

2. **"qsigdist is the family ceiling."** It's the *single-policy*
   ceiling. A per-prompt router has a much higher ceiling. The
   closeout conflated these and the verdict was wrong at the
   per-prompt level.

3. **"Death-zone policies are bad."** They have catastrophic
   mean Δ but they win on 4–6% of prompts. They're not bad —
   they're specialized. Discarding them from a router would lose
   3–6pp of the achievable headroom.

4. **"Routing requires new Layer 2 features."** I had proposed
   β as "add a 4th feature." That was right for "find a
   single-policy champion in a richer family." For routing, the
   priority shifts: characterize what prompt features predict
   the winner, AT WHICH GRANULARITY (token-level features?
   prompt-level? type tags?), and how cheap a router can be.
   Adding a Layer 2 feature is one of many options; the simpler
   move is to build a router over the existing 11-policy
   inventory.

5. **"The arc's architecture finding is the takeaway."** It's
   *a* takeaway. The bigger takeaway is now: a router over the
   existing policies can achieve 3.6× qsigdist's gain. The
   architecture validation was about HOW to predict; the per-
   prompt finding is about WHAT to predict. Both matter.

**Leverage points:**

L1. **Cluster prompts by winning policy.** Read-only on data
already on disk. Tells us whether the win signal has coherent
structure (technical → policy A, code → policy B, dialog → policy
C) or is essentially random. If coherent, a simple feature-based
router (token-set or first-token classifier) can capture much of
the headroom cheaply.

L2. **Cheapest-possible router stress test.** "Prompt has code
tokens → sigdist; else qsigdist" is a 2-line rule that achieves
SOME fraction of the oracle's +16.58pp. The achieved Δ is the
realizable floor with zero learning. Estimate the floor and the
gap to the oracle ceiling.

L3. **Multi-seed re-measurement of top-headroom prompts.**
Run the top-10 oracle-headroom prompts with 3 RNG seeds × all
policies. Distinguish "this policy genuinely specialized on this
prompt" from "this policy happened to match by 1-token noise."
Cost: ~30 min per prompt; 10 prompts × 11 policies / parallel
budget = 1-2 hours.

L4. **Held-out oracle estimate.** Split 100 prompts into 50/50.
For each 50/50 split, the oracle on the train set is the
realizable ceiling. Aggregate over multiple splits for a tighter
CI than the bootstrap on the full-set oracle.

L5. **Architectural re-target.** The Layer 3 kernel + wall
predictor's loss signal becomes "per-prompt winning policy"
instead of "mean Δ per cell." The architecture is unchanged;
only the input space (prompt features) and target (winner) change.
The structural-separation lesson transfers.

## SYNTHESIZE

Concrete actionable output:

### What the per-prompt analysis actually delivered

1. **Refutation of the mean-Δ ceiling.** Oracle Δ = +22.96pp;
   headroom over qsigdist +16.58pp, CI [+12.83, +20.42]. Bigger
   than qsigdist's gain over random.

2. **Validation of "entropy is structure" as a discipline.** Win
   distribution entropy 0.902 of max = real per-prompt
   specialization. The death-zone policies aren't bad; they're
   specialists.

3. **A premature-closeout caught in time.** The closeout I shipped
   15 minutes before this analysis is partially wrong. Tripp's
   one-question reframe exposed it. The closeout's architectural
   findings stand; the empirical verdict needs the footnote
   added in the reopen journal.

### What to do next (priority order)

1. **L1 — Cluster prompts by winning policy.** Read-only on disk.
   ~5 min to write, ~10s to run. Tells us whether the win signal
   is coherently structured.

2. **L2 — Cheapest-possible router stress test.** Try a handful of
   1-feature routers (prompt has code? long? Has digits? Starts
   with a question?) and measure achieved Δ vs oracle ceiling.
   Read-only. ~30 min total.

3. **L4 — Held-out oracle CV.** Split-and-aggregate; tighter
   ceiling estimate than the bootstrap. Read-only. ~15 min.

4. **L3 — Multi-seed re-measurement of top-headroom prompts.**
   Cost: 1-2 hours of harness runs. Skip if L1–L2 indicate the
   structure is genuinely there (i.e., clustering is coherent
   and 1-feature routers capture > 30% of the headroom).

5. **L5 — Architectural re-target.** If L1–L4 confirm
   non-trivial structure, redirect Layer 3 to predict winner from
   prompt features. The kernel + wall architecture transfers
   directly; only the inputs and target change.

### What NOT to do

- Continue iterating in the 27-cell family searching for a
  single-policy champion. The per-prompt analysis says there
  isn't one.
- Treat the closeout as the final word on the arc. Update the
  closeout journal with the per-prompt finding.
- Run the β plan (add a 4th Layer 2 feature) until L1–L4 are
  done. The existing 11-policy inventory may already contain
  enough specialization.
- Skip the held-out CV. The +22.96pp oracle is upper-bounded by
  selection bias; we should report the realistic ceiling.
- Skip multi-seed re-measurement of suspicious wins. Single-seed
  per-prompt-winner is noisy.

### Honest framing for forward communication

The meta-routing arc closeout shipped 15 minutes ago was premature.
A per-prompt analysis on the existing data — prompted by Tripp's
"enhance vs compete" reframe and "entropy is structure"
operationalization — found that the 27-cell linear-score family
DOES contain a champion when viewed as a router: oracle Δ
+22.96pp, 3.6× qsigdist's gain. The architecture's findings
(structural separation, in-sample MAE overfit trap, death-zone
observation) remain correct. The empirical headline shifts: the
question is no longer "can Layer 3 beat qsigdist?" but "can Layer
3 build a router that captures a fraction of the +16.58pp
per-prompt headroom?"

The architecture I built was pointed at the wrong target. Tripp's
question rotated the target by one reframe. The next moves are
all read-only on existing data; we don't need new harness runs
until the routing-target version is built and validated against
held-out prompts.

## Status

LMM cycle complete. The arc isn't closed — Tripp's question
reopened it at the level that actually matters for substrate
inference quality. Forward plan distinguishes between (a) the
read-only structural analyses that should be done before any new
harness runs, (b) the cheap-router stress test that establishes
the realizable floor, and (c) the eventual architectural re-target.
