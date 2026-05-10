# Cycle 1 Plan v2 — Part-B Experiment Design (post-red-team)

Folds the red-team findings (`cycle1_plan_redteam.md`) into the plan.
Supersedes `cycle1_plan.md`. The original plan stays in journal as the
audit trail.

## Goal (unchanged)

Produce a scored list of 5-10 candidate experiments that could provide
direct evidence for or against thesis Part B, with the top 1-2 scoped
enough to start Cycle 2.

## Process changes from v1 (per red-team)

### RAW phase

- **Don't re-read `step_change_raw.md` before writing.** Treat as a fresh
  problem. The 10+ candidate target should pressure past the obvious three.
  *(R3)*
- Include a forced category: "candidates from areas I'm less familiar with"
  with at least 2 entries, even if vague. *(R5)*
- Include a forced sub-question: "what would actually constitute Part-B
  evidence on a real workload?" — i.e., start sharpening Part-B's
  operational definition during RAW. *(R1)*
- Generate at least 12 candidates (the v1 plan said 10+; padding upward
  to defeat anchoring). *(R3, R5)*

### NODES phase (mostly unchanged)

- Group candidates by category if natural groupings emerge.
- Mark tensions: tractability vs informativeness, training-vs-inference,
  existence-vs-mechanism-vs-trajectory.
- Substrate-distinctiveness is a tension, not a disqualifier. *(R2)*

### REFLECT phase

- **Primary task: sharpen Part-B's operational definition.** What
  constitutes "equal compute"? "Task richness"? "Routing advantage"?
  Without sharpening, the SYNTH scoring is uncalibrated. *(R1)*
- The big question: "what makes a Part-B experiment USEFUL?"
  Answer should categorize by:
  - EXISTENCE (does base-3 routing win on any real workload?)
  - TRAJECTORY (does the gap widen with structural complexity?)
  - MECHANISM (is the win actually about routing-essentiality, or just
    substrate efficiency?)
- Surface and challenge assumptions: what does Part B assume about the
  workload? About the alternative? About "compute parity"?

### SYNTHESIZE phase

- Scored list with explicit scoring rubric (so it's re-scorable). *(R6, R10)*
- Scoring axes (specifically called out in v2):
  1. **Tractability range** (1-week / 1-month / 1-quarter), with
     dependency notes. *(R4)*
  2. **Informativeness — positive case** (what we learn if Part-B
     evidence is found).
  3. **Informativeness — negative case** (what we learn if the
     experiment falsifies Part B for this workload). *(R6)*
  4. **Substrate-distinctiveness** (does this USE the substrate's distinct
     capabilities, or just live ON it?). *(R2)*
  5. **Operationalizability** (are the pre-commit gates concrete enough
     that we'd know if we'd succeeded or failed?).
- **Two output categories** rather than one ranked list:
  - **Scored confidently**: candidates I can score on all 5 axes.
  - **Needs more research before scoring**: candidates I can't score yet
    but want to record for future cycles. *(R9)*
- Pre-commit gates per top candidate (what observation = Part-B evidence;
  what observation = falsification). *(R6)*

### Failure modes the plan handles explicitly

If during RAW or REFLECT it becomes clear that:

1. **No inference-only Part-B candidate is tractable** → SYNTH outputs
   "the synthesis's mode-shift recommendation needs revision; training-first
   sequencing is the right call after all." Loop back to
   `step_change_synth.md`. *(R7)*

2. **No candidate has clear pre-commit gates** → SYNTH outputs "Part-B as
   currently defined is not operationalizable; thesis-amend cycle needed."
   Loop back to `docs/THESIS.md`. *(R6)*

3. **The scoring rubric is more valuable than the scored list** → SYNTH
   names the rubric as Cycle 1's primary product. *(R9)*

4. **Cycle 1 needs another design cycle before any candidate is
   actionable** → STOP. The framing is wrong; revisit the synthesis.
   Cycle 1 must produce something Cycle 2 can act on, or it's failed at
   its job. *(R10 — recursive LMM risk)*

## Sequencing (unchanged)

Strict LMM discipline. RAW → NODES → REFLECT → SYNTH. Loop back if any
phase reveals an earlier phase was wrong.

## What this plan is NOT (unchanged)

- Not a commitment to find a winning Part-B experiment in Cycle 1.
- Not a commitment to start Cycle 2 immediately.
- Not bound by my prior notes on Part-B candidates.

## Time

No time-box. Quality over speed (per project rule + R8). Estimate is
"hours to a day" but I'll let the work take what it takes.

## What "done" looks like

The deliverable is `partB_experiments_synth.md` containing:
1. The 5-axis scoring rubric.
2. A "scored confidently" table of N candidates.
3. A "needs more research" list of M candidates.
4. Top 1-2 candidates from the confident list with Cycle 2 launch criteria.
5. A "what we can't test yet and why" subsection.
6. An honest assessment of whether the synthesis's mode-shift framing
   survived contact with the candidate landscape (per R7).

If `partB_experiments_synth.md` doesn't exist by the end of Cycle 1, the
cycle hasn't completed.
