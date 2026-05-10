# Synthesis: next step-change is a MODE shift, not a BUILD

## Architecture (the proposal in one paragraph)

The next step-change for Glyph is a mode shift from substrate-building to substrate-testing — specifically, from accumulating Part-A evidence (kernels work, BitNet runs) to actively searching for Part-B evidence (routing is essential and gap widens with task richness). The first concrete work in this mode is a Part-B Experiment Design cycle: enumerate 5–10 candidate experiments that could provide Part-B evidence (or falsify it), score each on tractability + informativeness, pick 1–2 to execute. Capability work (training, second model port) is sequenced AFTER this cycle and is justified by what specific Part-B experiment each unlocks, not as the default mode.

## Why this is a step-change

Substrate-building has been the project's mode since the ground-zero rebuild. It produced real artifacts: 29 ctest binaries, BitNet b1.58-2B-4T inference at ~92% strict pass, the L1 strong-claim cycle, the Phase 2 BitNet primitives. That mode reached the end of its high-leverage phase the day BitNet inference landed coherent end-to-end output. Continuing to build (training, second model, broader characterization) is high-EV incremental work, but it doesn't move the needle on the project's central thesis question.

The step-change is reorientation. The project's vision (NORTH_STAR.md, THESIS.md) is the falsifiable claim that base-3 routing structurally outperforms base-2 alternatives in a regime where structure rises. Part A (base-3 is the natural shape of the hardware) has substantial evidence. Part B (routing is essential and the gap widens) has zero direct evidence. The project has been BUILDING evidence for one half of its thesis while LEAVING the other half untested. That's not a deficiency — it's the natural shape of building infrastructure first. But continuing it past the point where infrastructure justifies it would be substrate-novelty drift (CONTRIBUTING.md): doing competent work that is adjacent to, rather than IS, the substrate-claim.

## Key decisions

1. **Mode shift over build choice.** The recommendation is NOT "do gradient kernels next" or "do strong-claim L2 next." It's "shift the mode and let the next builds be derived from a Part-B-relevant question." This is a bigger ask than picking a build, because it changes how decisions get made.

2. **Part-B Experiment Design comes first.** Before any large build, run an LMM cycle on Part-B candidates. The cycle's RAW phase asks "what would constitute clean Part-B evidence on a real workload?" The output is a scored list of candidate experiments. This cycle is small (days to weeks, not months).

3. **R1's falsification is a model, not a deterrent.** The R1 cycle attempted Part-B evidence and got falsified. That's the discipline the project says it values. The mode shift makes R1-style cycles the primary mode of work, not exceptional.

4. **Capability builds get demand-justified.** Training, second model, broader characterization — these still happen, but each is justified by a specific Part-B experiment that requires it. "Training because the four-phase plan says so" is replaced by "training because experiment X requires backward-pass kernels for Y."

5. **The four-phase plan is reinterpreted, not abandoned.** Inference → fine-tune → train → productize remains a sensible capability progression. But its sequence is not a discipline obligation; the trigger for each phase is a Part-B experiment that requires it.

## Implementation spec — the next 2-3 weeks

**Cycle 1: Part-B Experiment Design (LMM, ~1 week)**

- `journal/partB_experiments_raw.md`: unfiltered candidate list. Include the obvious (routing-native architecture trained from scratch) AND the less obvious (post-hoc routing of a trained model, analytically-derived routing, routing-as-attention-replacement, sparsity-detection benchmark, structural-prediction benchmark). Aim for 10+ candidates.
- `journal/partB_experiments_nodes.md`: extract candidates as discrete nodes. Tensions: tractability vs informativeness, training-required vs inference-only, single-workload vs trajectory.
- `journal/partB_experiments_reflect.md`: the structural insight should be a categorization of "what makes a Part-B experiment USEFUL." E.g., does it test the EXISTENCE of base-3 advantage, the TRAJECTORY (gap widens), or the MECHANISM (routing-essentiality)?
- `journal/partB_experiments_synth.md`: scored list of 5-10 candidates with concrete next-cycle scoping for the top 1-2. Each top candidate gets enough detail to run a follow-on LMM cycle on its design.

**Cycle 2: Top Part-B candidate execution (LMM + measurement, ~2 weeks)**

- Run the LMM cycle on the top candidate from Cycle 1.
- The candidate's design phase identifies what capability is needed (e.g., "this experiment requires training a 50M-parameter model" → triggers gradient-kernel build).
- Execute the experiment. Pre-commit to gates (per the project's discipline). Report the result honestly even if it's negative.

**If Cycle 2's experiment requires training:** Build the gradient kernels. The build is justified by the experiment, scoped to what the experiment needs, and time-bounded by the experiment's deadline.

**If Cycle 2's experiment is inference-only and fails to find Part-B evidence:** That's substantial information. Continue with Cycle 3 (next candidate from Cycle 1's list). Three failed Part-B candidates in a row would be evidence that Part B as currently framed might not be true on real workloads — itself a finding worth recording.

## Success criteria

- [ ] After Cycle 1: a journal-recorded scored list of 5-10 Part-B candidate experiments, with the top 1-2 scoped enough to start Cycle 2.
- [ ] After Cycle 2: a Part-B experiment with a pre-committed gate has been executed and the result is recorded (positive, negative, or methodologically inconclusive — all valid).
- [ ] The project's CHANGELOG / FINDINGS reflects a new axis of work: "Axis 5: Part-B searches" with an entry per candidate experiment attempted.
- [ ] Build cycles (training, etc.) that happen in this period are tied to Part-B experiments, not to the four-phase plan as default.

## Quality check (per LMM)

- **Could someone else execute this from the synthesis?** Mostly. Cycle 1 is well-scoped. Cycle 2 depends on Cycle 1's output, which is correct sequencing.
- **Does it address all the key nodes?** Yes:
  - N1 (multiple meanings of step-change): explicitly answered — claim step-change over capability step-change.
  - N2 (four-phase plan): reinterpreted, not abandoned.
  - N3 (L2 strong-claim): valued but de-prioritized vs Part-B work.
  - N4 (Part-B is most thesis-relevant): central.
  - N5 (training is a prerequisite for Part-B): addressed by demand-justifying training.
  - N6 (step-change vs EV): addressed by noting Part-B's low probability is partly a function of low search effort.
  - N7 (second model): also de-prioritized vs Part-B work.
  - N8 (workflow infra): may emerge as needed for Cycle 2; not the primary recommendation.
  - N9 (single-track assumption): preserved.
  - N10 ("step-changeyness" assumption): replaced with "claim-relevance" criterion.
  - N11 (Part-A vs Part-B claim density): central tension, addressed.
  - N12 (claim drift risk): mitigated by sequencing (some Part-A confirmation if needed).
  - N13 (inference-only Part-B): explicitly invited as a candidate category.
- **Is it simpler than the raw thoughts suggested?** Yes. The raw thoughts explored 6+ option categories; the synthesis collapses them to a single mode shift with a 2-cycle execution plan.
- **Surprised by how clean it is?** Somewhat. I started thinking "training" was the answer; ending with "the answer is to stop defaulting to capability builds" is a real surprise.

## Honest caveats

- This recommendation is a bigger strategic ask than my initial "training next." If the user has external constraints I don't see (timeline, audience, deliverable shape), the right move may be smaller.
- The mode-shift framing might itself be wrong. If the project's actual goal is "ship a usable substrate someone else builds on," then capability and characterization matter more than thesis-testing. Mode shift to Part-B would be the wrong call there.
- Cycle 1's RAW phase will surface whether tractable Part-B candidates actually exist. If they don't — if every candidate requires multi-month training before it can be tested — then training-first is the right call after all, and this synthesis was the wrong strategic frame. The cycle would still produce useful information (a categorization of what Part-B tests look like and what they cost).

## Loop-back triggers (for me)

- **Back to NODES** if Cycle 1's RAW reveals categories I didn't consider.
- **Back to REFLECT** if a candidate's design phase shows the "mode shift" framing is wrong (e.g., if testing Part-B turns out to require capability builds we don't have, recursively).
- **Run a fresh LMM cycle** on the project's strategic posture if the user pushes back on the mode-shift framing.

---

*"Give me six hours to chop down a tree, and I will spend the first four sharpening the axe."*

The axe right now is "we have a substrate that works." Sharpening it for Part-B means searching for the experiment that exposes the thesis to falsification. That search is the next chop.
