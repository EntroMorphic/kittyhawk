---
date: 2026-04-14
phase: SYNTHESIZE
topic: §14 "Seven Open Decisions" in m4t/docs/M4T_SUBSTRATE.md
---

# Synthesis

## Outcome

Six of the seven dissolve under triage. One remains.

| § | Item | Category | Decision |
|---|---|---|---|
| 14.1 | Logical block size | Substrate-real, empirical | **Defer.** Logical block = hardware block (16 B, 1:1). Revisit only when a workload shows cache/prefetch stress. |
| 14.2 | Cross-block add policy | Substrate-real, vestigial | **Don't implement.** No routing consumer drives it. If one emerges, provide as a named opt-in (`m4t_mtfp_vec_add_aligning`) with explicit rounding flag, not as a default path. |
| 14.3 | Tail-block padding | Substrate-real, answered | **Zero-pad mantissas.** A zero mantissa is the additive/multiplicative identity regardless of block exponent, so zero-padding is semantically null — it preserves the widen-don't-round invariant. Tensor carries a cell-count field for length-aware consumers. |
| 14.4 | Exponent sentinels | Substrate-real, answered | **No sentinels in the exponent.** If flag semantics are needed, allocate a parallel 1-byte status array. Keep the exponent numerically pure. |
| 14.5 | SDOT output exactness | Proven contract | **Move to §8.4 as theorem.** Max output = 16 × 40 × 40 = 25 600 ≪ int32_max. Exact by construction. Not open; a contract. |
| 14.6 | LUT-backed nonlinearities | Consumer-level | **Move out of §14.** Pull back from archive only when a specific routing consumer requests GELU/softmax. Not a substrate open. |
| 14.7 | Benchmark bed | Thesis-level | **Move out of substrate spec entirely.** Create a separate document (`docs/THESIS.md` or equivalent) that tracks what the routing thesis has to beat and on what problems. M4T serves whatever thesis asks; it does not pick benchmarks. |

## Required edits to `m4t/docs/M4T_SUBSTRATE.md`

1. **§8.4 (SDOT as ternary matmul)** — append proof sentence: "SDOT over MTFP4 inputs is exact by construction: max output magnitude = 16 × 40 × 40 = 25 600, well inside int32. The substrate declares this as contract; compositions of widened inputs into SDOT shape are the caller's responsibility to bounds-check."

2. **§14 (Open decisions)** — reduce to four entries:
   - 14.1 Logical block size (open, deferred pending first workload)
   - 14.2 Cross-block add (deferred, not implemented)
   - 14.3 Tail-block padding (decided: zero-pad)
   - 14.4 Exponent status tracking (decided: parallel byte array if needed)

3. **§16 (Traceability)** — add row: "§14 triage resolved via LMM cycle (journal/seven_open_decisions_*.md, 2026-04-14)."

4. **New document** `docs/THESIS.md` — owns former 14.6 and 14.7:
   - Which consumer(s) M4T is being built for.
   - Which benchmarks validate the routing-over-dense thesis.
   - When LUT-backed nonlinearities are pulled back from archive (driven by consumer architecture).

## Success criteria

- §14 contains only substrate-level questions.
- §14 contains exactly one genuinely open item (14.1).
- The proven contract (former 14.5) lives where contracts live (§8.4).
- Consumer/thesis concerns (former 14.6, 14.7) live in a thesis document, not the substrate spec.
- The rebuild can proceed on settled answers without further philosophy.

## What this buys the rebuild

A substrate spec that names its own boundaries. §14 was a heat sink for every unresolved question; triage revealed most of those questions either had answers or weren't the substrate's to answer. After this synthesis, the rebuild's first code can start — there is no substrate-level ambiguity blocking it.
