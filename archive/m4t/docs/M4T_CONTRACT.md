# M4T Opcode Contract

A function is an M4T opcode if and only if it satisfies all seven clauses below. These clauses are the definition of "extends the silicon" — they are what makes the substrate caller-indistinguishable from native hardware instructions.

---

## 1. Vector-op granularity

Every opcode operates on contiguous buffers. Length is parameterized (`int n_trits`, `int n`, or `int M, K, N` for matmul-shaped ops). No per-element dispatch overhead. No per-layer monoliths.

## 2. No allocations

No `malloc`, no VLA, no stack arrays whose size depends on input length. All scratch space is caller-provided or absent. An opcode may use a fixed amount of stack for loop variables and NEON register spills.

## 3. No errors

No return codes. No `errno`. No exceptions. Preconditions (non-null pointers, non-negative dimensions, in-range cell values) are the caller's responsibility and are asserted in debug builds (`assert()`). In release builds, violating a precondition is undefined behavior — same as passing a bad address to a hardware instruction.

## 4. Predictable latency

Runtime is a closed-form function of the input dimensions. No data-dependent branches in the inner loop except where inherent to the operation (e.g., LayerNorm's Newton-Raphson converges in a bounded number of steps). The contract gating measurement: **p99 latency ≤ 1.5 × mean latency** over 10⁶ calls with hot data. Anything wider indicates a cold-cache path leaking through.

## 5. Cache-resident under load

Opcode bodies (`.text`) fit within the L1i budget (24 KB). Data tables (LUTs, splat constants) fit within the L1d budget (4 KB). Both budgets are enforced at link time by `tools/m4t_size_check.sh`. Cache residency under realistic workloads is verified by PMC-based measurement (M3) once a consumer forward pass exists.

## 6. Indexable

Every opcode is callable by direct C function name (`m4t_trit_mul(...)`) AND via a function-pointer table entry (`m4t_trit_ops[M4T_TOP_MUL](...)`). The table surfaces are the canonical external API; direct calls are an optimization for internal consumers.

**Status:** opcode tables are pipeline item 6 and not yet implemented. Direct C calls are available now.

## 7. Caller-indistinguishable from silicon

From the call site, the operation looks like an instruction: takes registers and memory, runs in known time, produces the result. No setup, no warm-up, no error path, no finalization. The caller does not need to know or care that the implementation is software.

---

## Measurement gating

The contract is not publishable as honored until the following measurements pass:

| ID | Measurement | Gating clause | Status |
|---|---|---|---|
| M1 | `.text` size ≤ 24 KB | Clause 5 | **Passing** (17.7 KB, 71%) |
| M2 | p99 ≤ 1.5 × mean per opcode | Clause 4 | **Partial** — LayerNorm at small N fails (p99/mean = 1.87) |
| M3 | L1i miss rate < 0.5% under load | Clause 5 | **Deferred** — no consumer forward pass yet |

## What this contract does NOT cover

- Threading: M4T is single-threaded at the opcode level. Parallelism is a consumer concern.
- Training: v0 is inference-only. No backward passes, no gradient ops.
- Float: no float anywhere in `libm4t.a`. Boundary conversions live in host-side tools.
- Portability beyond aarch64+NEON: not supported. CMake errors on other targets.
