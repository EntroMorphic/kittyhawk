---
cycle: integrative qsig_filter LMM
phase: ALL (raw + nodes + reflect + synthesize)
date: 2026-05-14
scope: synthesize the qsig_filter integration result (K=1 +8.50pp Δ
       beats qsigdist +6.38pp by +2.12pp; full curve K∈{1,2,4,8} with
       K=1 the clear winner; selective routing on top fails held-out)
       into a coherent picture and a principled next step. Tripp's
       reframe (competing vs integrative) rotated the entire arc;
       one test answered. What does this MEAN, and what should we do?
companions: integrative_qsig_filter_2026-05-14.md (factual record);
            per_prompt_routing_lmm_2026-05-14.md (the LMM where the
            "wrong question" was diagnosed); l1_cluster_lmm_2026-05-14.md
            (the prior LMM where competition's ceiling was clear).
---

# Integrative qsig_filter — LMM cycle

## RAW

What I actually think, unfiltered:

The reframe was load-bearing in a way I hadn't anticipated. I
spent four prior LMMs analyzing why competition couldn't reach the
+16.58pp oracle ceiling. Each was honest about the failure mode.
None considered that the framing itself — "pick one policy per
prompt" — was the problem. Tripp asked one question and an
afternoon's test went past everything competition had achieved.

The K=1 result feels real, even though the CI on +2.12pp straddles
0 at single seed. The per-category mechanism is coherent: technical
writing has redundancy structure that K-K-similar slots preserve;
code/structured-questions are more independent per token and
protecting redundancy hurts. This isn't post-hoc rationalization —
the categorical pattern was visible in the per-prompt routing data
before I built the filter, but I couldn't read it as "integration
target" until the integrative frame was named.

What surprised me: the SELECTIVE version on top of K=1 actively
hurt held-out (−0.67pp, CI excludes 0). The story I would have
predicted — "always-on K=1 OK, selective K=1 better" — was
wrong by 0.67pp. Routing of any kind at this sample size and
feature granularity loses to just leaving the integration on
globally. The integration is robust; the selective add-on is not.

The TriX patterns I noted (CompiledDispatch + guard) didn't help
this round because our signal isn't strong enough at N=100 single-
seed for guards to fire correctly. They would help if multi-seed
shrinks the per-cell CI enough to make per-category specialization
distinguishable from noise. I'm reading TriX as "this is what we
build NEXT, after multi-seed validates K=1 is real."

What concerns me: the +2.12pp gain at single-seed N=100 has CI
[-1.33, +5.83]. That's a 7.2pp width on the central estimate. If
multi-seed shrinks the CI to e.g. [+0.5, +3.5] then the win is
clear. If multi-seed reveals the variance was BIGGER than single-
seed suggested (heteroskedastic per-prompt noise), the gain could
collapse. We have a clean mechanism story but the statistics are
weak at this N.

What I'm avoiding looking at:

- The K=1 wins are mostly on technical/factual prompts. The
  losses are on code/questions/poetry. If the model's deployment
  population is skewed toward code (which is plausible for many
  applications), the integration could hurt rather than help. The
  +2.12pp is on a balanced N=100 designed for category coverage,
  not a real workload distribution.

- "Match against no_evict" is the metric. K=1 might preserve more
  no_evict-like tokens by mechanically deferring eviction in some
  way that's not really better generation, just a different
  failure mode that happens to look like no_evict. Multi-seed
  wouldn't catch this; only direct quality eval would. (Connects
  to "coherence over bit-parity" memory.)

- The protection criterion ("K most-similar to current K") might
  be an accidentally-good heuristic for THIS prompt distribution
  but not generalize. A real proof-of-mechanism would be:
  perturb prompts toward TECH-like content, watch K=1 gain
  increase. We have category-correlation evidence, not
  experimental manipulation evidence.

- I built test #1 of the 5 integrative architectures on the menu.
  The one I picked (conjunctive filter) was the simplest. The
  others might be MORE effective OR might stack with K=1. I
  haven't tested any of them.

- The architectural pattern "always-on integration > selective
  routing" might be specific to this problem (KV eviction at
  window=16) and not generalize. Different problems with different
  per-prompt variance structures might have different answers.

## NODES

Extracted tensions and constraints:

**N1 — Real mechanism, weak statistics.** K=1 wins by +2.12pp on
mean with coherent per-category mechanism (technical content
benefits from K-K redundancy preservation). But CI straddles 0 at
single-seed N=100. The mechanism story carries the result; the
statistics don't.

**N2 — Integration ≠ Routing.** Always-on K=1 = +2.12pp gain;
selective K=1 = −0.67pp. The integrative move (apply uniformly)
won where routing failed. This contradicts my prior intuition
("more sophisticated = better") and validates "do less" when
routing-confidence is below noise floor.

**N3 — TriX's CompiledDispatch isn't applicable yet.** The
guard+fallback pattern requires routing-confidence above noise
floor. At N=100 single-seed, ours isn't. The pattern is correct
in principle but premature in practice.

**N4 — The integrative-architecture menu has 4 untested options.**
Conjunctive filter worked. Multi-policy consensus, conditional
handoff, soft eviction, and score-multiplicative haven't been
tried. K=1's success doesn't tell us which others work.

**N5 — Per-prompt routing arc's investment was over-allocated to
the wrong question.** Four LMM cycles + 7+ runs spent on
competition. One integration test went further. The COST
distribution doesn't reflect the VALUE distribution.

**N6 — N=100 single-seed is the measurement floor.** Single-
seed per-cell match-rate has 1/24 token granularity. Multi-seed
random baseline would shrink the per-prompt Δ noise. We've been
quoting CIs that include this measurement-noise contribution
unattributed.

**N7 — The integration's mechanism is K-K-redundancy
preservation, but only verified by category correlation.**
Technical content correlates with K=1-helps; code with K=1-hurts.
Causal verification (deliberately add redundancy → see K=1 gain
go up) hasn't been done.

**N8 — The deployment-population question.** +2.12pp on N=100
balanced category coverage may not reflect real-world workload
mixes. A code-heavy workload could see net negative. Not
addressed.

## REFLECT

Structure, assumptions to challenge, leverage points:

**Structure.** Three layers of finding:

1. **Architectural** (does integration beat competition?): YES,
   measurably. K=1 +2.12pp; selective −0.67pp.
2. **Mechanism** (why does K=1 work?): K-K redundancy preservation
   helps technical content; mechanism inferred from category
   correlation, not direct manipulation.
3. **Statistics** (is +2.12pp robust signal?): NOT YET. CI
   [−1.33, +5.83] at single-seed; mechanism is the load-bearing
   evidence.

The architectural finding is solid. The statistical finding needs
multi-seed. The mechanism is plausible but unverified causally.

**Assumptions to challenge:**

1. **"Mean Δ +2.12pp is the headline number."** It's the headline
   on N=100 single-seed at this prompt mix. Different N, different
   seeds, different prompt mix → different number. Should report
   confidence interval AND mechanism alongside the point estimate.

2. **"Always-on integration always beats selective routing."**
   True at this sample size. As N → ∞ and routing-features
   improve, eventually selective COULD beat always-on (if there
   are categories where K=1 hurts strongly enough that omitting
   them more than compensates for prediction error). The current
   finding is N=100-specific, not universal.

3. **"K=1 is the right K."** It's the best of {1, 2, 4, 8}. K=3,
   5, 6, 7 weren't tested. The non-monotonic curve (K=2 < K=4)
   suggests measurement noise at this N; the smooth-mechanism
   story would predict monotone with K=1 best, K=8 worst, intermediate
   K's intermediate. If multi-seed reveals K=2 ≈ K=3 ≈ K=4 (smooth
   plateau), the K=1 winner becomes more defensible.

4. **"The conjunctive filter is the best integrative architecture."**
   It's the first one tested. The 4 others on the menu might do
   better individually, OR stack with K=1. The "integration > competition"
   lesson generalizes; the specific architecture choice doesn't.

5. **"This is the right problem to be working on."** KV cache
   eviction at window=16 is one specific application. The
   integrative-routing principle might apply to many problems
   (FFN tile routing, embedded retrieval, etc.). The architectural
   pattern transfers; the specific application doesn't.

**Leverage points:**

L1. **Multi-seed L3 (deferred from earlier).** Run K=1 and
qsigdist with 3 random-baseline seeds. Compute paired Δs against
each seed; tighten CI. ~70 min harness. Should run before any
further integration tests.

L2. **Test integrative architecture #2 (multi-policy consensus).**
Each of {qsigdist, sigdist, fifo} proposes top-2 victims; evict
the slot with most votes. Different mechanism; might stack with
K=1 (consensus AMONG K=1's protected-set-respecting outputs?).

L3. **TRiX bundle/manifest pattern when N grows.** Once we have
multi-seed data + a few integration tests, the address-plane +
guard pattern from TriX could let us SHIP a routing decision
with drift detection. Premature now, right pattern to head toward.

L4. **Workload-distribution sensitivity check.** Compute K=1
gain on different prompt subsets (code-heavy, tech-heavy, mixed)
to estimate how the gain shifts with deployment population.
Read-only on existing data.

L5. **Causal mechanism check.** Deliberately construct prompts
with high vs low K-K redundancy (e.g., repetitive technical
keywords vs unique tokens); measure K=1 gain on each. Validates
or refutes the redundancy-preservation mechanism.

## SYNTHESIZE

Concrete actionable output:

### What the integrative test actually delivered

1. **Architectural proof-of-concept**: integration beats competition
   at this problem, this sample size. K=1 always-on +2.12pp;
   selective routing −0.67pp.

2. **A new best policy**: qsig_filter K=1 supersedes qsigdist as
   the best known KV-eviction policy at window=16. Mean Δ +8.50pp.

3. **A second confirmation of the in-sample-overfit trap**:
   selective routing's in-sample +3.62pp evaporated to held-out
   −0.67pp. Same trap as L1.1 (memorized as
   feedback_in_sample_overfit_trap.md).

4. **A coherent mechanism**: protect K-K-similar slots from
   eviction = preserve technical redundancy = wins on technical
   content, loses on code/questions/poetry. Category correlation
   pattern is mechanism-consistent.

### Forward plan (priority order)

1. **Multi-seed L3 on K=1 vs qsigdist.** Confirm or shrink the
   CI on +2.12pp. ~70 min harness. The mechanism is solid;
   statistics need backup.

2. **Workload-distribution sensitivity check (L4 above).**
   Read-only on existing data. Tells us how the gain shifts under
   different prompt mixes. ~10 min.

3. **Test integrative architecture #2 (multi-policy consensus).**
   Cheap to specify, ~45 min harness mod + battery. If it
   independently improves OR stacks with K=1, we have the
   integrative-architecture menu opening up.

4. **Causal mechanism check.** Construct prompts with controlled
   K-K redundancy. Measure K=1 gain. ~30 min for prompt design +
   battery. Validates the mechanism story.

5. **Defer TriX patterns until N grows.** CompiledDispatch +
   guard pattern is right for the future, premature now.

### What NOT to do

- Run another competition-frame analysis. The category routing
  is dead at this sample size; multi-seed first, then revisit if
  CI shrinks.
- Build complex routers on top of K=1. Always-on > selective in
  held-out CV. Adding sophistication adds prediction risk that
  exceeds the gain.
- Stop iterating just because K=1 won. The integration menu has
  4 untested options; the K=1 win argues we should test the rest
  to see what the integrative-architecture space looks like.
- Quote +2.12pp without the CI and mechanism qualification. The
  point estimate alone is misleading at N=100 single-seed.

### Honest framing for forward communication

The integration test succeeded. qsig_filter K=1 is the new best
known policy at +8.50pp (vs qsigdist +6.38pp). The +2.12pp gain
has CI straddling 0 at single-seed N=100, but the per-category
mechanism is coherent and the architectural lesson —
"always-on integration > selective routing > competition" —
holds in held-out CV across the routing strategies tested.

The next move is multi-seed validation of the +2.12pp gain.
After that, the other 4 integrative architectures on the menu
(consensus, handoff, soft eviction, multiplicative) deserve
testing to map the integration design space. The TriX patterns
(CompiledDispatch + guard, address ABI, drift policy) are
correct for the future when statistics support them.

The substrate's six-primitives floor is unchanged.

## Status

LMM cycle complete. The integration arc has its first concrete
win and its first concrete architectural lesson. Forward plan
distinguishes between (a) multi-seed validation of the K=1 gain,
(b) workload sensitivity check, (c) other integration tests, and
(d) deferred operationalization (TriX patterns). The cost of
arriving at this finding (one afternoon, one harness mode, four
batteries) is small relative to the prior 4-LMM competition arc
that didn't reach +0pp over qsigdist held-out.
