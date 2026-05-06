---
cycle: bitnet_phase1
phase: SYNTHESIZE
date: 2026-05-06
scope: actionable plan for Phase 1 — end-to-end BitNet b1.58-2B-4T inference on the m4t substrate. Output that someone (us) could pick up and execute against. Includes go/no-go gates, decisions made, and explicit handling of the tensions REFLECT surfaced.
companions: bitnet_phase1_{raw,nodes,reflect}.md
---

# Synthesize — bitnet_phase1

## What this cycle decided

### D1 — Architecture is now grounded
BitNet b1.58-2B-4T: 30 layers, hidden 2560, GQA 20:5 (head_dim 128), FFN 6912 with **ReLU²** activation, **subln** normalization (RMSNorm primitive shape), RoPE with `theta=500000`, A8 activations, ternary weights via absmean. No bias terms. Llama-3 tokenizer. 4096 context.

### D2 — Phase 1 fidelity gate: per-layer L2-relative tolerance
Not bit-exact match to HF. Not task-quality. The gate is: **per-layer L2-relative error vs HF reference is bounded and characterized.** ε is determined empirically during Phase B-thin and reported in CLOSEOUT. We do not pre-pick a value of ε. We pre-commit to: "ε is finite, doesn't grow across layers, and we explain its source."

### D3 — Activation format: Path α (match A8) for Phase 1
Add an `m4t_a8_t` primitive — int8 cell + per-token absmax scale — as a substrate extension. Quantize/de-quantize at layer boundaries to match HF's reference path. Path β (MTFP19 throughout) is **explicitly deferred to Phase 2** when we control the training pipeline and can measure whether A8 is necessary.

**Rationale**: Phase 1 is a validation step; matching the reference's data path makes per-layer comparison meaningful. Path β risks "we look different from HF and we don't know whether that's correct or wrong." Path α gives us a tight comparison gate; Phase 2 revisits with more freedom.

**Strategic note for the user**: Path β is more aligned with the project's vision claim 3 ("don't pretend ternary is base-2"). Choosing Path α for Phase 1 is a pragmatic concession to validation rigor, not a long-term substrate identity choice. If you'd rather take Path β even at the cost of harder validation, this decision is reversible — flag before EXECUTE.

### D4 — Sequencing: thin-B → D → full-B → A
- **Thin B (work-unit 1)**: load BitNet weights from HF format, run *one transformer block* forward pass with stubs for missing primitives, compare per-layer outputs vs HF reference. Surfaces actual kernel access patterns and what's missing.
- **D (work-units 2-5)**: substrate gap closure. Per-primitive cycles for rsqrt, RoPE, softmax LUT, A8 quantize/dequantize. Each gets a NEON path + scalar_ref + bit-exact tests.
- **Full B (work-unit 6)**: re-run thin B with real primitives instead of stubs. All 30 layers. Measure ε.
- **A (work-units 7-9)**: end-to-end inference. KV cache. Generation loop. Sanity checks on output text.

### D5 — Schedule: work-units, not calendar
Phase 1 estimate is **9 work-units, each scoped explicitly below.** Calendar time depends on per-unit difficulty, which we can't predict reliably. Pace unit-by-unit; revise after each.

## Action plan

### Work-unit 1 — Thin B (single-block harness)

**Inputs**: Glyph substrate (current state); HF BitNet b1.58-2B-4T model files.

**Tasks**:
1. New consumer module `gesh/bitnet/` (separate from existing gesh routing infrastructure).
2. Weight loader: read HF's `model.safetensors` (or sharded variants), extract weights for layer 0 only (Q, K, V, O projections + FFN gate/up/down + 2 norm layers).
3. Convert ternary weights from HF storage to substrate-native packed format. **Question to resolve here**: HF's BitNet weights are stored in what format? int8? Already 5-in-8 packed? If int8, repack to 5-in-8.
4. Implement single transformer block:
   - subln_in → attention → +residual → subln_out → ffn → +residual
   - Stub primitives: scalar reference rsqrt, scalar Python-style RoPE, scalar softmax, scalar A8 quantize/dequantize.
5. Reference comparison: run HF's `transformers` library on the same input through layer 0; compare per-output-tensor.
6. Capture: (a) per-layer L2 error, (b) which primitives the kernel actually called and with what shapes, (c) any composition surprises.

**Success criterion**: layer 0 forward pass runs without crashing. Per-layer comparison works at all (even if numerical values differ). The list of "primitives that need NEON paths" is concrete and bounded.

**Failure modes to budget for**:
- Weight format mismatch (HF stores ternary in some specific way; need to match)
- Tokenizer integration with libm4t-shaped buffers
- Bit-precision in HF's reference (bf16 has rounding quirks the substrate doesn't)
- Memory layout (HF uses contiguous tensors; substrate uses block-aligned)

### Work-unit 2 — Substrate primitive: rsqrt for RMSNorm

**Design**: Newton-Raphson with magic-number initial guess, MTFP19 fixed-point throughout. Bit-exact NEON-vs-`_scalar_ref` gate. Range characterized: input is `mean(x²) + ε` for hidden=2560 vector with int8 (after de-quant) inputs; bound the input range empirically from Thin B's captured tensors.

**Deliverables**: `m4t_mtfp_rsqrt(dst, src, n)` + `_scalar_ref` in `m4t_mtfp.h`. Tests: bit-exact NEON-vs-scalar across the empirical input range; aliasing test; n=0 boundary.

**Success criterion**: NEON and scalar_ref produce bit-exact identical output for ≥10K random inputs in the empirical range. Newton-Raphson converges in fixed iteration count (3-4 expected).

### Work-unit 3 — Substrate primitive: RoPE rotation

**Design**: Pre-computed `(cos, sin)` LUT sized to `max_position × head_dim/2 = 4096 × 64 = 262144` pairs. At MTFP19 precision (4 bytes/value × 2 values = 8 bytes/pair) → 2 MB LUT total. Acceptable storage. NEON-vectorized 2-pair rotation: `(q_even, q_odd) → (q_even*cos - q_odd*sin, q_even*sin + q_odd*cos)`.

**Deliverables**: `m4t_rope_apply(q, position_offset, head_dim, n_heads)` + `_scalar_ref`. The LUT is built at substrate-init time using `rope_theta=500000` from BitNet's config (parameterizable for future models).

**Success criterion**: NEON output matches scalar_ref bit-exact across all (position × head_dim) combinations.

### Work-unit 4 — Substrate primitive: softmax via LUT

**Design**: Restore TD-14's archived LUT generator. Generate `exp(x)` LUT bounded to attention-score range (typical: `x ∈ [-30, +30]` post-scaling). Softmax = `exp(x_i - max(x)) / sum(exp(...))`. The max-subtraction step uses existing reductions; the exp step is the LUT lookup; the divide uses existing magic-multiply or a reciprocal LUT.

**Deliverables**: `m4t_softmax(dst, src, n)` + `_scalar_ref`. Numerical-stability path verified (subtract-max; no overflow).

**Success criterion**: bit-exact NEON-vs-scalar across attention-score-range inputs. Sum of output = 1.0 within rounding (degenerate test for stability).

### Work-unit 5 — Substrate primitive: A8 quantize/dequantize

**Design**: New cell type `m4t_a8_t = int8` + per-tensor `m4t_a8_scale_t = m4t_mtfp_t` (or smaller) carrying the absmax-derived scale. Two functions: `m4t_a8_quantize(dst_int8, src_mtfp19, scale_out, n)` and `m4t_a8_dequantize(dst_mtfp19, src_int8, scale, n)`.

**Deliverables**: new header `m4t_a8.h` (or extension to existing). `_scalar_ref` for both directions.

**Question for SYNTHESIZE**: should A8 live in libm4t (extends substrate) or in `gesh/bitnet/` (consumer-side)? **Decision**: in libm4t. The format is general enough that other consumers might want it, AND keeping it substrate-side gets us the no-scalar audit's NEON discipline by default. If Phase 2 deprecates A8, removing it is simpler than promoting it later.

**Success criterion**: round-trip preserves int8 codes; quantize → dequantize within rounding error of input; bit-exact NEON-vs-scalar.

### Work-unit 6 — Full B (all 30 layers)

**Inputs**: Thin B harness + completed primitives from D.

**Tasks**:
1. Replace stubs with NEON primitives.
2. Extend weight loader to all 30 layers + embedding + LM head.
3. Run forward pass on one input through all 30 layers.
4. Per-layer comparison vs HF reference (both at bf16 precision).
5. Compute ε (per-layer L2 relative error).
6. Verify ε does not grow exponentially across depth.

**Success criterion (Phase 1's actual gate)**: ε is bounded across all 30 layers. We can characterize where it comes from (rounding mode mismatch, A8 quantization granularity, etc.). The plot of "per-layer ε vs layer index" is non-divergent.

### Work-unit 7 — KV cache integration

**Tasks**: state management for cached K and V across token positions; per-step append; per-step attention-with-cache.

**Deliverables**: `gesh/bitnet/kv_cache.{c,h}` — scoped to BitNet's exact shape (5 KV heads × head_dim × max_position). Memory: `30 layers × 5 heads × 128 head_dim × 4096 ctx × 4 bytes/cell × 2 (K+V) = 2 GB`. Acceptable.

**Success criterion**: forward pass at position N matches forward pass at position N when the prefix is replayed (i.e., cache produces identical results to no-cache).

### Work-unit 8 — Generation loop

**Tasks**: token-by-token generation with greedy sampling (top-k / nucleus is a Phase 2+ concern). Tokenizer integration (use HF's tokenizer at the C boundary; emit `int32` token IDs to libm4t).

**Deliverables**: `gesh/bitnet/generate.c` driver. Reads a prompt; produces tokens until EOS or max_length.

**Success criterion**: generated text is coherent on a fixed prompt. Coherence is judged by: (a) the output isn't gibberish, (b) the same prompt produces the same output deterministically (greedy), (c) the output approximately resembles HF's output on the same prompt with greedy sampling.

### Work-unit 9 — CLOSEOUT

**Tasks**: write `journal/bitnet_phase1_closeout.md` capturing:
- ε measured across the 30 layers
- The Path α decision in retrospect (was it the right call?)
- Substrate primitives added (count, sizes, perf)
- Schedule actuals vs work-unit estimates
- Which surprises landed where in the plan vs where the plan didn't predict them
- Phase 2's prerequisites (which Phase 1 met, which it didn't)
- Recommendation to user on Phase 2 framing (BitNet-style mixed-precision vs. base-3-native research cycle)

## Decisions deferred to the user

These showed up in REFLECT and the cycle won't resolve them on its own:

**U1 — Path α vs Path β.** SYNTHESIZE proposes Path α (match A8 spec) for Phase 1 with revisit-in-Phase-2. If you'd rather take Path β (MTFP19 throughout) even at the cost of looser validation, flag before work-unit 1 starts.

**U2 — Phase 4 framing.** Phase 4 (productization) was in my prior summary but you didn't endorse it explicitly. SYNTHESIZE doesn't plan against it. If Phase 4 is load-bearing for the project's mission, that affects how Phase 2/3's APIs get designed.

**U3 — Calendar bound.** Phase 1 is 9 work-units of unknown duration. If you want a hard calendar bound (e.g., "re-plan if Phase 1 hasn't produced a working forward pass in N weeks"), set N. If you want unbounded ("methodically until done"), no further action.

## Pre-EXECUTE red-team checklist

Before work-unit 1 begins, the red-team should ask:

- Have I read the BitNet paper (arxiv 2504.12285)? If yes: any architectural details config.json didn't capture? If no: do that first.
- Is there an existing `bitnet.cpp` reference implementation? Yes (Microsoft's). Does our work overlap or duplicate? **Likely overlaps**. We're not building the fastest BitNet runner; we're building "BitNet on top of base-3 substrate" as a substrate validation. The deliverable is structurally different even if the inputs/outputs match.
- Are there license issues with using BitNet weights in CI tests? They're MIT-licensed per HF model card. Probably fine; verify before committing weight blobs to repo (don't — point at HF and download lazily).
- Does the substrate's CMake build still link cleanly when we add `gesh/bitnet/`? Verify before committing the directory structure.
- Are the existing 22 ctest binaries still passing? Yes (verified at start of this cycle). Re-verify before each work-unit.

## What this cycle did NOT resolve

- The strategic shape of Phases 2-4. (Not in scope.)
- Whether the substrate is the right vehicle vs. PyTorch / JAX bindings. (Not in scope; the project rule and substrate identity already answer this.)
- Whether base-3-native training will work. (Phase 2/3 question, not Phase 1.)

## Status

**Ready for red-team review.** SYNTHESIZE proposes Path α + thin-B-first + work-unit-by-work-unit pacing. Awaiting (a) user input on U1/U2/U3, (b) red-team check on whether the work-unit list has hidden dependencies or missing nodes.

If both clear, work-unit 1 begins.
