# M4T Beyond Pipeline — Future Work

Items that gate real use but are not on the 8-item pipeline. Tracked here for follow-up.

Status markers: `[ ]` todo · `[~]` in progress · `[x]` done

---

- [x] **LUT tables committed.** `src/m4t_mtfp_tables.c` generated and committed. GELU, softmax, and argmax runtime functions in `m4t_mtfp_nonlinear.c`.

- [ ] **Softmax and GELU runtime.** LUT-backed lookup functions in `m4t_mtfp.{h,c}`. Table lookup is pure integer — the LUTs are the only float-derived artifact, computed offline by the generator. Blocks any transformer forward pass.

- [ ] **Weight I/O tool.** `tools/m4t_io.c` — host-side boundary converter: reads float32 weight files, quantizes to MTFP cells (4/9/19/39), writes binary blobs with a header. Not linked into `libm4t.a`. Blocks inference on real data.

- [ ] **Transformer block.** Compose attention (Q/K/V projection, softmax, residual) + routed FFN (routing primitives + ternary matmul) into a single forward step. All primitives exist; the plumbing doesn't.

- [ ] **CI pipeline.** GitHub Actions: build `libm4t.a`, run all test binaries, run `m4t_size_check.sh`. Triggers on push to main.

- [ ] **M3 measurement.** Apple Instruments L1i hit-rate measurement under a realistic consumer workload. Can't run until a transformer forward pass exists. Gates contract publication (clause 5).

- [ ] **LayerNorm p99 jitter.** Bench showed p99/mean = 1.87 at LayerNorm (1×64). Fails contract clause 4 (predictable latency: p99 ≤ 1.5× mean). Likely the Newton-Raphson isqrt convergence varying by input. Not investigated.

- [ ] **MTFP19 dense matmul NEON vectorization.** Currently scalar with __int128 accumulator. NEON path would use vmull_s32 (int32→int64 widening multiply) + vaddq_s64 accumulation, with periodic __int128 spill for the rescale. Significant perf win for dense MTFP×MTFP paths.

- [ ] **Ternary matmul vmulq_s32 → branchless sign-select optimization.** R10 from M4T red-team. The current NEON kernel uses `vmulq_s32` to apply ternary signs; a branchless XOR+ADD conditional-negate pattern would avoid the multiply. Blocked on benchmarking to measure the actual gap.
