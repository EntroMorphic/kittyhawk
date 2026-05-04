# Synthesis: Path Forward After R1 Remediation FAIL

## Architecture

**The next cycle is a focused sig_dim experiment** that distinguishes three framings of the R1 FAIL (wrong rule / wrong axis / wrong layer). After the experiment, the choice between the various "what to do next" options becomes data-driven rather than intuited.

In parallel, **P1-1 (close primitives floor with exp/log) begins design work**. Vision claim #1 is independent of R-track outcomes and is foundational for vision claim #2 to scale beyond the current expression vocabulary.

The R2 plan's original three-track structure (R1 → R3 → R2) is replaced by:

```
R1-fork    : sig_dim experiment at {16, 32, 64} with BOTH rules
              ↓ (data resolves F1/F2/F3)
              ↓
R1-decide  : revert / redesign / per-arity / pivot, per fork outcome
              ↓
R3 / R2    : continue per the resolved framing, OR redirect entirely

P1-1       : begin design cycle in parallel (independent of R-track)
```

## Key Decisions

**D1: The fork is F1 vs F2 vs F3, not A-through-H.** [from REFLECT core insight]
The eight surface options reduce to three structural framings of the failure. Pick the framing first; the option follows. Don't pick the option without picking the framing.

**D2: The sig_dim experiment is the cheapest test that distinguishes the framings.** [REFLECT T1, T2 resolutions]
Three days of work resolves a fork that would otherwise be picked on intuition. Pre-committed thresholds in REFLECT (D1).

**D3: P1-1 is independent of R-track outcomes.** [REFLECT A3, A5]
The R1 FAIL doesn't change whether exp/log primitives are needed for vision claim #2 to scale. P1-1 can begin design in parallel with any R-track choice.

**D4: Per-arity rules are an outcome of the experiment, not a strategy.** [REFLECT A2]
If sig_dim experiment shows arity-1 and arity-2 benefit from different rules, per-arity dispatch is a finding. If it shows uniform behavior, per-arity isn't adopted.

**D5: The R2 plan needs replanning, not just rerouting.** [REFLECT T1, A4; NODES Node 10]
The original R2 plan assumed R1 PASS. With R1 FAIL, the sequencing breaks. The R1-fork experiment is the new entry point; what comes after depends on its outcome.

## Implementation Spec

### R1-fork: sig_dim experiment

**Goal:** distinguish F1 (wrong rule) from F2 (wrong axis) from F3 (wrong layer).

**Experiment design:**
- Run BOTH rules (sign-only via `expr_to_signature` + `expr_bank_build`; dual via `expr_to_signature_dual` + `expr_bank_dual_build`).
- At three sig_dim values: 16 (current), 32, 64.
- For each (rule, sig_dim) pair, measure:
  - Curated arity-1 bank: number of equivalence classes, inter-class min/mean/max distance.
  - Curated arity-2 bank: same.
  - Random bank (100 random expressions per arity, single seed for now): merger rate, inter-class distances.
  - Subagent-probe match rate (the existing 30 probes).
- Multi-seed for the random-bank measurements: 3 seeds (cheaper than 5; primary aim is direction-of-effect, not statistical confidence).

**Test-input sets:**
- sig_dim=16: existing 16 test inputs (unchanged).
- sig_dim=32: 32 test inputs constructed by interpolation between existing values (e.g., adding {-25, -7, -4, -8, +25, +7, +4, +8, ...} between existing points; preserve symmetric / sign-flip-spanning property).
- sig_dim=64: 64 test inputs by further interpolation.
- Arity-2: 5×5 grid for sig_dim=25, 6×6 grid for sig_dim=36, 8×8 for sig_dim=64.

**Pre-committed framings (committed BEFORE running):**

| Framing | Wins iff | Implication |
|---------|----------|-------------|
| **F1 (wrong rule)** | dual at sig_dim=64 has strictly better arity-1 inter-class min distance than sign-only at sig_dim=64 | R1 v2 with refined dual rule (Option B) is justified |
| **F2 (wrong axis)** | sign-only at sig_dim=64 reaches arity-1 inter-class min distance ≥ 6 AND dual at sig_dim=64 doesn't add ≥ 2 over sign-only at same dim | Revert R1; proceed to R3 (sig_dim sweep) and R2 (scale) with sign-only as the primary (Option A + D) |
| **F3 (wrong layer)** | both rules plateau at sig_dim=64 with similar discrimination AND neither reaches min ≥ 6 | Pivot to P1-1; vision claim #3 doesn't manifest in this consumer (Option F) |

If results are mixed (e.g., F2 wins for arity-1 but F1 wins for arity-2), the per-arity outcome (Option H) is the data-driven answer.

**Required code work:**
- Extend test-input sets for sig_dim=32 and 64 in a new constants file.
- New benchmark binary `gesh_expr_routing_r1_fork.c` that runs all (rule, sig_dim) combinations and reports the framing-determining metrics.
- ~1.5 days code + 0.5 day for test-input design + 1 day for analysis.

**Substrate-discipline:** all measurements use existing kernels (`m4t_route_threshold_extract`, `m4t_route_threshold_extract_dual`, `m4t_popcount_dist`, `m4t_route_confidence_weighted_dist`). No new substrate primitives required.

### P1-1 design cycle (parallel track)

**Goal:** define the path to closing the primitives floor with exp/log.

**This synthesize does not commit to which P1-1 path** (Path A: substrate primitives; Path B: compositions). That's the LMM cycle's job for P1-1 itself.

**Required design work:**
- LMM cycle on P1-1 (at minimum: RAW + NODES + REFLECT + SYNTHESIZE).
- Pre-committed gates per H4 discipline rule.
- Risk register: substrate spec amendment likely under Path A; precision-bound documentation likely under Path B.

**Required code work:** none until SYNTHESIZE picks a path. Estimate: ~1 week for design + 2-4 weeks for implementation, depending on path.

**Independence:** can start whenever owner authorizes. Doesn't block on R1-fork or anything else.

## Pre-committed gates (R1-fork)

The R1-fork experiment is a falsification experiment, not a PASS/FAIL gate. Its success criterion is "produces evidence sufficient to choose between F1, F2, F3." Specifically:

- **R1-fork PASSes** if at sig_dim=64, the per-rule, per-arity inter-class min distance and partition-change rate distinguish at least two of F1/F2/F3 with > 2-trit margin in either direction.
- **R1-fork is INCONCLUSIVE** if the metrics cluster within 2 trits of each other, requiring a higher sig_dim or different test-input design.
- **R1-fork has no FAIL state.** Any data is informative.

After R1-fork:
- If F1 wins: write R1 v2 plan.
- If F2 wins: revert R1, write the corrected R3/R2 plan.
- If F3 wins: archive R1, write a P1-2 plan that proceeds without substrate-distinctive signatures.
- If mixed: write per-arity plan with explicit caveats about fragmentation.

## What this synthesis does NOT do

- Does not commit to any of A-through-H prematurely. The fork experiment is the precondition.
- Does not design R1 v2. That's the next cycle's job, contingent on F1 winning.
- Does not pre-design P1-1's path. That's an independent LMM cycle.
- Does not retire R1 code. Per project ship-with-FAIL discipline (P0-4 precedent), R1 stays in the codebase regardless of outcome.

## Loop-back triggers (per LMM)

- **Back to RAW** if the R1-fork experiment surfaces a fourth framing not in F1/F2/F3 (e.g., "the bank shape is wrong, not the signature rule"). Real possibility — R1's failure modes might point to deeper architectural issues.
- **Back to NODES** if the F1/F2/F3 framings turn out to overlap in ways that make the experiment's verdict ambiguous. Indicates the framings need refinement.
- **Back to REFLECT** if the R1-fork experiment is conclusive but the chosen option (A-H) feels wrong on review. Re-examine whether the framing-to-option mapping in this synthesis was correct.
- **Run a full new cycle** if R1-fork reveals that vision claim #3 itself needs reformulation. That's a vision-level change, not a cycle-level one.

## Action plan

1. **Now:** write `docs/PLAN_R1_FORK.md` with the experiment design and pre-committed framings (mirror the discipline of `PLAN_EXPRESSION_ROUTING_R2.md`).
2. **Next:** implement the fork experiment. ~3 days.
3. **Then:** run, analyze, write closeout that names the winning framing.
4. **In parallel:** if owner authorizes, begin P1-1 design cycle (RAW phase).
5. **After fork resolves:** write the next plan based on the winning framing.

The honest stopping criterion for R-track work: if F3 wins (the consumer can't deliver substrate-distinctness), the R-track is closed and the project pivots to vision claim #1 work. If F1 or F2 wins, the R-track continues with the right tool.
