# Cycle 1 Plan — Part-B Experiment Design LMM

Per `journal/step_change_synth.md`. This is the plan for HOW to run Cycle 1,
before running it.

## Goal of Cycle 1

Produce a **scored list of 5-10 candidate experiments** that could provide
direct evidence for or against thesis Part B (routing is essential, gap
widens with task richness), with the **top 1-2 candidates scoped concretely
enough that Cycle 2 can begin without further design work**.

## Success criteria

After Cycle 1, the project has:
1. A journal record (RAW → NODES → REFLECT → SYNTH) of how the candidate
   list was generated and how candidates were scored.
2. A scored table of 5-10 candidate experiments. Each entry includes:
   - Concise description of the experiment (what's measured, how)
   - Pre-committed gates (what would constitute Part-B evidence; what
     would constitute falsification)
   - Tractability estimate (cycles needed; capability prerequisites)
   - Informativeness estimate (what we learn from a positive result; what
     we learn from a negative result)
   - Substrate-novelty audit (does this experiment USE the substrate's
     distinct capabilities, or just test something that could be tested
     on a base-2 substrate too?)
3. Top 1-2 candidates with a Cycle 2 design sketch (key questions,
   measurement plan, capability needs).
4. A "what we can't test yet and why" section — Part-B candidates we
   considered but couldn't scope. This is a deliverable, not a failure.

## What "executing Cycle 1" means concretely

Four files in `journal/`, in sequence:

1. **`partB_experiments_raw.md`** (~30 minutes of unfiltered writing)
   - Brain-dump candidate experiments. Aim for 10+ candidates.
   - Cover multiple categories: training-required, inference-only,
     architectural, post-hoc, sparsity-detection, structural-prediction.
   - Include candidates I'm SUSPICIOUS of (probably won't work but worth listing).
   - Write the doubts plainly. What scares me about each candidate?

2. **`partB_experiments_nodes.md`** (~25 minutes)
   - Extract candidates as discrete nodes.
   - Mark tensions: tractability vs informativeness, training-vs-inference,
     single-workload vs trajectory measurement, "tests existence" vs "tests
     mechanism" vs "tests trajectory."
   - Group candidates by category if natural groupings emerge.

3. **`partB_experiments_reflect.md`** (~30 minutes)
   - Find the structural insight beneath the candidates.
   - The big question to answer: **"What makes a Part-B experiment useful?"**
     Categorize by what it tests:
     - EXISTENCE: does base-3 routing win on any workload?
     - TRAJECTORY: does the gap widen with task complexity?
     - MECHANISM: is the win actually about routing-essentiality vs
       just substrate efficiency?
   - Resolve tensions from the nodes phase.
   - Surface assumptions I'm making about what "Part B" means.

4. **`partB_experiments_synth.md`** (~30 minutes)
   - Scored list of 5-10 candidates.
   - Scoring rubric explicit (so a future reader can re-score with different weights).
   - Top 1-2 candidates with Cycle 2 launch criteria.
   - Honest caveats: which candidates I scored low because they're hard to
     understand vs because they're actually low-value.

## Sequencing

Strict LMM discipline: don't write SYNTH until REFLECT is done; don't write
REFLECT until NODES is done. Each phase is allowed to surprise the next.

If during NODES or REFLECT I realize Cycle 1's PLAN is wrong (e.g.,
candidates can't be meaningfully scored without first defining what Part B
means more precisely), loop back and fix the plan. That's an LMM
loop-back-trigger, not a failure.

## What this plan is NOT

- It is not a commitment to find a winning Part-B experiment in Cycle 1.
  Cycle 1's job is to ENUMERATE and SCORE candidates honestly. Whether any
  are tractable is itself the finding.
- It is not a commitment to start Cycle 2 immediately after Cycle 1.
  If Cycle 1 reveals that no candidate is tractable in current scope, the
  honest output is "Part-B is harder than the substrate currently allows;
  here's what would have to change."
- It is not bound by my prior notes on what Part-B experiments might look
  like. The RAW phase should generate fresh thinking, not regurgitate
  step_change_raw.md's three-line list.

## Estimated effort

If Cycle 1 follows the LMM-quickstart timing (30 min/phase × 4 = 2 hours)
with deeper reflection where it earns it: **half a day to a full day**.
The 1-week estimate from the synthesis was upper-bound; if Cycle 1 stays
focused, it's hours.
