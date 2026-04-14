# archive/

Superseded code and documentation, retained for historical reference.

Nothing in this directory is on the build path. Headers here are not on any include path. Consumers of `libm4t.a` and `libglyph` cannot reach any symbol defined under `archive/`.

## What moved here and why

The ground-zero rebuild (2026-04-14) was triggered when the prior implementation was found to have collapsed Multi-Trit Floating Point into a fixed-point reading — a single shared global scale treated as a property of the type rather than per-block metadata. The "zero float" ideology followed from that collapse. Recovering the spec required moving everything that assumed the fixed-point model, or that shaped computation densely over base-3 hardware, out of the live surface.

What lives here now:

| Path | What it was | Why it moved |
|---|---|---|
| `m4t/src/m4t_mtfp.{c,h}` | MTFP19 numeric core | Dense matmul, bias, fan-in, LayerNorm bundled with element-wise arithmetic. Rewritten cleanly in the new substrate. |
| `m4t/src/m4t_mtfp_w.{c,h}` | MTFP39 wide-cell arithmetic | Dense matmul path; no routing consumer. |
| `m4t/src/m4t_mtfp_nonlinear.c` + `m4t_mtfp_tables.c` | GELU / softmax / argmax LUTs | Dense-transformer consumers. 118K-line LUT table. Returns if a routing consumer needs smooth nonlinearities. |
| `m4t/src/m4t_ops.{c,h}` | Function-pointer dispatch table | Mixed dense and routing opcodes; needs pruning before it can return. |
| `m4t/tests/test_m4t_smoke.c` | Omnibus substrate smoke test | Exercised the dense path; replaced by focused per-primitive tests. |
| `m4t/tests/test_m4t_mtfp_w.c` | MTFP39 tests | Moved with MTFP39 source. |
| `m4t/tests/test_m4t_ops.c` | Dispatch-table tests | Moved with dispatch table. |
| `m4t/tools/m4t_bench.c` | Per-opcode cycle counter | Benched the dense path; returns when a consumer needs measured numbers on the rebuilt primitives. |
| `m4t/docs/{M4T_BEYOND, M4T_CONTRACT, M4T_PIPELINE, M4T_REDTEAM, TRIT_LATTICE_LSH}.md` | Pre-rebuild design docs | Superseded by `m4t/docs/M4T_SUBSTRATE.md`. |
| `src/glyph_mtfp.h` | Glyph MTFP wrapper header | Aliased the archived dense surface. Returns when a glyph consumer is defined. |
| `tests/test_glyph_wrapper.c` | Glyph alias coverage | Moved with the wrapper. |
| `tools/mnist_knn_lattice.c` | Dense L2 k-NN on MNIST | Dense computation over MTFP storage. Not routing-native. |
| `tools/mnist_m4t_infer.c` | Dense transformer inference | Entirely dense-shape. |
| `tools/mnist_train_dump.c` | Float-side training artifact converter | Float in the training path. |
| `reference-code/` | Original trix-z C kernels | Contained float paths. Quarantined from the start; moved here for tidiness. |
| `docs/REMEDIATION_PLAN.md` (old) | Pre-rebuild trix-z triage | Superseded by the live `docs/REMEDIATION_PLAN.md`. |
| `docs/REDTEAM_FIXES.md` (old) | Pre-rebuild red-team | Historical only. |

## What might come back

An item leaves archive only under a named consumer demand. Not "we might need it" — *this consumer needs this primitive, and here is what it will do with it*. The archive exists partly to make that discipline cheap: the code is already written, so when demand materializes, the return is editorial, not from-scratch.

## What will not come back

- Any path that treats MTFP as fixed-point with a shared global scale.
- Any primitive whose natural shape is dense matmul over ternary data.
- Any IEEE-754 float at runtime. (Build-time LUT generation is sanctioned in one named place: `m4t/tools/m4t_lut_gen.c`.)

## Further reading

- [`NORTH_STAR.md`](../NORTH_STAR.md) — why the rebuild happened.
- [`m4t/docs/M4T_SUBSTRATE.md`](../m4t/docs/M4T_SUBSTRATE.md) — the spec the rebuild restored.
- [`journal/seven_open_decisions_synthesize.md`](../journal/seven_open_decisions_synthesize.md) — the LMM cycle that finalized the rebuild scope.
