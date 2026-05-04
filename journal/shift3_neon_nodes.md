# NODES: NEON magic-multiply divide-by-3^k prototype

Atomic claims/findings/concerns extracted from `shift3_neon_raw.md`. Each node is one independently-evaluable statement.

## Findings (verified by code)

- **N1.** Magic constants exist (M, N) per k ∈ [1, 19] such that `(int64_t)x*M + (1 << (N-1))) >> N` is bit-exact equal to `m4t_pow3_round_div(x, 3^k)` for every x in [-M4T_MTFP_MAX_VAL, +M4T_MTFP_MAX_VAL].
- **N2.** The constants were verified exhaustively (1.16 × 10⁹ x values per k, 19 k values) by `m4t/tools/gen_pow3_magic.c`.
- **N3.** A NEON kernel (`m4t_shift3_div_neon` in `m4t/tests/test_m4t_shift3_neon_proto.c`) using `vmull_s32 + vaddq_s64 + vshlq_s64 + vmovn_s64` matches the scalar reference `m4t_mtfp_shift3` for the divide direction (k ∈ [-1, -19]) bit-exact across 50K random samples + 12 corner cases per k.
- **N4.** The NEON kernel runs at ~0.18 ns/element on Apple Silicon (n=4096, LTO build). The scalar reference runs at ~1.67 ns/element on the same data. Speedup factor ~9.5–10× across k ∈ {1, 7, 13, 19}.
- **N5.** The cycle-count estimate at 3.5 GHz: NEON ~0.63 cycles/elem, scalar ~5.8 cycles/elem.

## Decisions (made during prototyping)

- **N6.** Pivoted from the user's original `vqrdmulhq + vshlq` two-stage rounding pipeline to a 64-bit-intermediate `vmull + bias + arith-shift` pipeline. The pivot was forced — the two-stage approach could not be made bit-exact across all k due to compound rounding error.
- **N7.** Within the 64-bit-intermediate approach, M is chosen as `round(2^N / d)` and N as the largest value ≤ 62 such that `M ≤ INT32_MAX`, then a small ±8 search around the theoretical M for the bit-exact winner.
- **N8.** Verification is two-stage: a 700K-sample "smart set" winnows candidates, then a 1.16 × 10⁹ exhaustive verify confirms each smart-set winner. Total generator runtime ~23 seconds.

## Concerns (flagged, not yet addressed)

- **N9.** No formal proof that `(x*M + bias) >> N` always fits in int32 before `vmovn_s64` narrows. The result is mathematically `≈ x/d` and bounded by `MAX_VAL/3 < INT32_MAX`, but I haven't constructed the formal bound.
- **N10.** Aliasing (`dst == src`) is not exercised by the property test. The kernel reads 4 lanes via `vld1q_s32` and writes via `vst1q_s32` within one iter, so in-place should work, but no test confirms it.
- **N11.** The kernel assumes `abs_k ∈ [1, 19]`. The substrate's scalar `m4t_mtfp_shift3` handles `abs_k ≥ 20` separately (memset to zero). The NEON kernel has no such guard; productionizing means the wrapper must enforce this.
- **N12.** The magic constants live in two places (generator + test source). Productionizing requires a single source of truth (build-time generated header, or committed constants with regeneration discipline).
- **N13.** The kernel takes ONE `k` for a whole batch. Consumers that need per-cell-varying k (notably `m4t_mtfp_vec_accum_aligning`, the cross-exponent accumulator) cannot use this kernel directly.
- **N14.** The original `vqrdmulhq + vshlq` proposal might be salvageable via per-k specialization with `vrshrq_n_s32` (constant-shift rounding-right). Not pursued. Estimated potential: 1.5–2× faster than the 64-bit path.

## Hypotheses / unverified

- **N15.** No current substrate consumer has `m4t_mtfp_shift3` on its hot path. (Believed; not measured.)
- **N16.** The cross-exponent accumulator's per-cell-varying-k divide is structurally analogous (and hot), and a vector-of-Ms variant of this kernel might apply there. (Plausible; not investigated.)
- **N17.** AppleClang's auto-vectorizer does not currently vectorize the scalar divide path under -O3 + -mcpu=native + -flto. (Implied by the 5.8 cycles/elem scalar measurement; not directly verified by disasm.)

## Errors I made and recovered from

- **N18.** First emulator of `vqrdmulhq` missed the "doubling" (the `2*` factor in `(2*a*b + 2³¹) >> 32`). Caught by an off-by-2× pattern in the verification output.
- **N19.** First emulator of `vshlq` was plain arithmetic right shift; `vrshlq` (rounding shift left with negative count = rounding right) is the correct partner for `vqrdmulh`. Even after correcting, the compound rounding still failed bit-exact.
- **N20.** First N_max computation overshot by 4 (computed 44 for k=11, true max was 48). Failed bit-exactness for k ≥ 11. Fixed with a different computation strategy.
- **N21.** First smart-set verifier had `step = d/8 → 1` for small d, ballooning the test set to 8e9 points. Killed the runaway, fixed with explicit step floor.

## Reframings during the cycle

- **N22.** Original perf estimate was ~40× speedup. Actual is ~10×. The gap: I assumed scalar was hardware `sdiv` (~12 cycles); it's mul-based (~5.8 cycles). The 40× estimate was correct against `sdiv`-based scalar; the 10× is correct against the existing substrate.
- **N23.** Original framing was "substitute for custom silicon." Refined: the magic-multiply approach is a 10× win over the substrate's already-optimized scalar, not just over naive sdiv. Still meaningful but the headroom is smaller.

## Project-context observations

- **N24.** The substrate's `m4t_pow3_round_div` is mul-based, not sdiv-based. This is a non-obvious fact; `M4T_SUBSTRATE.md` doesn't currently call it out.
- **N25.** The cross-exponent accumulator (Tier 3a) is the only known site that might benefit from a per-cell-varying-k version of this kernel. It's substrate-internal; consumer-side use of shift3 directly is not currently surfaced.
- **N26.** Per `CONTRIBUTING.md` non-negotiable #5 ("No primitive without named consumer demand"), this work is OPTIMIZATION of an existing primitive (not a new one), so the consumer-demand gate doesn't apply directly. The user's explicit request is the demand signal.

## Open questions

- **N27.** Should the prototype become production now, or sit as a documented option until a consumer surfaces?
- **N28.** Is the 10× kernel-level speedup observable at any consumer level today?
- **N29.** Is the per-cell-varying-k variant tractable, and how much consumer impact would it have?
- **N30.** Is the original `vqrdmulhq + per-k specialized vrshrq_n_s32` worth pursuing for the additional ~1.5–2×?
- **N31.** Does the multiply direction (k > 0) have a similar opportunity? It's currently scalar with a `(int64_t)src[i] * scale` + clamp.
