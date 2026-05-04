---
status: P0 — owner directive 2026-05-04
authority: owner directive — close the three honest residuals from tier2_residuals_closeout
predecessor: journal/tier2_residuals_closeout.md
---

# Pre-Commit: Tier 2 Residuals v2 — Atomics

## Gates

**V2-G1 (LTO root cause):** Identify and fix the image_canon segfault under global `-flto`. PASS iff (a) root cause is named with file/line evidence; (b) fix is applied; (c) global LTO can be enabled and all 16 ctest binaries PASS; (d) bench binary still produces meaningful absolute timings.

**V2-G2 (cache-aliasing adversarial distributions):** Implement subagent-designed distributions 2 (run-length trap with page-aligned aliasing on a/b/d for select) and 5 (confidence-stripe cache thrasher with all 4 buffers in same L1 set for conf-dist). PASS iff both distributions implemented with verified address-aliasing (compute and assert that conflicting buffers share L1 cache set), and results reported.

**V2-G3 (cache-defeat saturation):** Run RES-1-style warm-vs-cold measurement at multiple working-set sizes spanning L1, L2, and L3+. PASS iff (a) at small sizes (L1-resident) warm/cold ratio ≈ 1; (b) at mid sizes (L1-spill, L2-resident) warm/cold ratio > 1.3; (c) at large sizes (memory-bound) warm/cold ratio approaches 1 again. The shape of the curve verifies the cache-defeat mechanism actually works when it should.

**V2-G4 (no regression):** All 16 ctest binaries PASS through every step.

## Order of execution

1. Read image_canon.c, find LTO-tripping pattern.
2. Apply minimal fix; re-enable global LTO; run ctest.
3. Implement V2-G2 cache-aliasing distributions.
4. Implement V2-G3 multi-scale cache-defeat sweep.
5. Build, run, verify.
6. Closeout.
