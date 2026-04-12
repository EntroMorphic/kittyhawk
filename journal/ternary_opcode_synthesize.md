# Synthesis: Glyph Ternary Opcode Set

Output of an LMM cycle on "deploy a Ternary Opcode set that lives in L-Cache and extends the silicon." Reads `ternary_opcode_reflect.md` as the source of truth; this file is the actionable artifact.

---

## One-line answer

**Glyph's ternary opcode set is a contract layer over its existing NEON kernels, plus a small batch of new TBL- and masked-VCNT-based primitives, exposed as two indexable function-pointer tables (`glyph_trit_ops[]` and `glyph_mtfp_ops[]`) with a documented L1i/L1d size budget and a cycle-counting measurement harness that defends the cache-residency claim.**

---

## The contract

A function `f` is a glyph ternary opcode iff it satisfies all of:

1. **Vector-op granularity.** Operates on contiguous buffers, with length passed as `int n` (or `int M, K, N` for matmul-shaped ops). Never per-trit, never per-layer.
2. **No allocations.** No `malloc`, no stack arrays whose size depends on `n`. Scratch space is caller-provided.
3. **No errors.** No return codes, no `errno`. Preconditions are caller responsibility, asserted in debug builds.
4. **Predictable latency.** Runtime is a closed-form function of `n` (or `M·N·K`) plus a fixed setup cost. No data-dependent branches in the inner loop except where required by the operation itself.
5. **Cache-resident under load.** Body fits in the L1i budget (see §Budget); data tables (LUTs, splat constants) fit in the L1d budget; access pattern keeps both hot under realistic glyph workloads.
6. **Indexable.** Reachable via either a direct C call or an entry in one of the two opcode tables (see §Tables).
7. **Caller-indistinguishable from silicon.** From the call site, the operation looks like an instruction: takes registers + memory, runs in known time, produces the result. No setup, no warm-up, no error path.

These seven clauses *are* the "extends the silicon" abstraction. Honor all seven and the caller cannot tell whether they're calling silicon or software.

---

## Two tables, one contract

### `glyph_trit_ops[]` — packed-trit vector ops

Header: `src/glyph_trit_opcodes.h`

```c
/* Calling convention for every trit opcode:
 *   void op(uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits);
 * Inputs and outputs are packed 2-bit trit codes (4 trits/byte).
 * Length is in trits, not bytes; the body computes packed-byte count.
 * dst may alias a or b for in-place ops. */
typedef void (*glyph_trit_op_t)(
    uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits);

enum glyph_trit_opcode {
    GLYPH_TOP_MUL,        /* dst = a ⊗ b   (Galois F_3 multiply) */
    GLYPH_TOP_SAT_ADD,    /* dst = sat(a + b) clamped to {-1, 0, +1} */
    GLYPH_TOP_MAX,        /* dst = max(a, b) */
    GLYPH_TOP_MIN,        /* dst = min(a, b) */
    GLYPH_TOP_EQ,         /* dst[i] = (a[i] == b[i]) ? +1 : 0 */
    GLYPH_TOP_NEG,        /* dst = -a   (b ignored) */
    GLYPH_TOP_COUNT       /* sentinel: number of trit opcodes */
};

extern const glyph_trit_op_t glyph_trit_ops[GLYPH_TOP_COUNT];
```

Each opcode body is the same shape: load 16 trit codes from `a` and `b`, pack pair indices into a uint8x16_t, `vqtbl1q_u8` against the per-op LUT, store. ~5–8 NEON instructions plus a 16-byte LUT in `.rodata`. Estimated body size: 64–96 bytes. With 6 opcodes, total ~512 B of code + 96 B of LUT data. Fits trivially in L1i.

### `glyph_mtfp_ops[]` — MTFP-cell vector ops

Header: `src/glyph_mtfp_opcodes.h`

```c
/* Calling convention varies by op shape — see per-op signature.
 * Three shapes for v0:
 *   unary:   void op(mtfp_t* dst, const mtfp_t* a, int n);
 *   binary:  void op(mtfp_t* dst, const mtfp_t* a, const mtfp_t* b, int n);
 *   matmul:  void op(mtfp_t* Y, const mtfp_t* X, const mtfp_t* W, int M, int K, int N);
 *
 * The opcode table holds (function pointer, shape tag) pairs so callers
 * can dispatch correctly. */

enum glyph_mtfp_opcode {
    GLYPH_MOP_VEC_ADD,         /* binary  */
    GLYPH_MOP_VEC_ADD_INPLACE, /* binary, dst aliases first arg */
    GLYPH_MOP_VEC_SCALE,       /* unary, scale passed as scalar */
    GLYPH_MOP_BIAS_ADD,        /* binary, broadcast bias over batch */
    GLYPH_MOP_FAN_IN_NORMALIZE,/* unary, fan_in passed as scalar */
    GLYPH_MOP_LAYERNORM,       /* layernorm shape (4-buffer) */
    GLYPH_MOP_MATMUL,          /* matmul shape */
    GLYPH_MOP_MATMUL_BT,       /* matmul shape */
    GLYPH_MOP_TERNARY_MATMUL_BT,/* matmul shape, W is packed trits */
    GLYPH_MOP_COUNT
};
```

These are the existing glyph kernels, **promoted** to opcodes by the contract. No reimplementation. The promotion adds:
- An entry in the table.
- A size budget assertion at link time (see §Budget).
- A measurement target for the harness (see §Measurement).

The function-pointer-table value is just `&glyph_mtfp_vec_add` etc. — the literal address of the existing function. Direct callers continue to call them by name and get full inlining where the compiler chooses.

---

## New opcodes glyph needs (the small greenfield)

These don't exist in glyph today. They are the kernels TBL and masked-VCNT make trivially fast, identified during the M4/NEON deep-dive earlier in the conversation:

### Trit-vector ops (TBL-based, all uniform shape)

- `GLYPH_TOP_MUL` — F_3 multiply via 16-byte LUT.
- `GLYPH_TOP_SAT_ADD` — saturating ternary add (2-trit operand → trit output via LUT).
- `GLYPH_TOP_MAX`, `GLYPH_TOP_MIN` — pairwise max/min.
- `GLYPH_TOP_EQ` — ternary equality, returns +1 / 0.
- `GLYPH_TOP_NEG` — sign flip via single TBL on a swap-encoded LUT (`{0, -1, +1, 0}`).

### Trit-vector reducers (masked-VCNT-based)

- `glyph_trit_signed_sum(const uint8_t* a, int n_trits) -> int64_t`
  Two masked VCNTs (mask `0x55` for +1s, mask `0xAA` for -1s), widening accumulation, subtract. ~15 NEON instructions, no LUT, no LUT lookup. Used by `colsum` for weight-derived signatures — directly relevant to glyph's routing layer.
- `glyph_trit_sparsity(const uint8_t* a, int n_trits) -> int64_t`
  Single VCNT pass over packed bytes (popcount of each pair gives nonzero indicator under our encoding), widening sum. Used for sparsity stats and as a denominator for routing-distance normalization.

These are the two reducers that "leave hardware on the table" today. They are the most concrete payoff of the M4 deep-dive.

### MTFP-cell ops

None new. Glyph's existing MTFP kernels already cover the v0 surface. New MTFP opcodes can land later as the routing layer demands.

---

## Budget

### Code budget (L1i)

| Region | M4 P-core L1i | Glyph reservation | Headroom for callers |
|---|---|---|---|
| L1i total | 192 KB | — | — |
| Glyph opcode bodies + LUTs (instructions) | — | **16 KB hard cap** | 176 KB |

**Justification:** caller code (routing, transformer block, training driver) needs the bulk of L1i. 16 KB is ~8% of L1i, large enough for ~150 small opcodes at 96 bytes each, small enough that a hot caller still has room. We sit well below the budget at v0 launch and land new opcodes against the cap, not against the limit.

### Data budget (L1d)

| Region | M4 P-core L1d | Glyph reservation |
|---|---|---|
| L1d total | 128 KB | — |
| Trit-op LUTs | — | **2 KB hard cap** (16 bytes × 128 max LUTs) |
| MTFP-op constants (splats, masks) | — | **1 KB hard cap** |

These are dwarfed by the working set of activation buffers, which is the point — opcode data is essentially free in L1d.

### Measurement-defended

Both budgets are *enforced*, not aspirational. See §Measurement.

---

## Layout

### File structure

```
src/
  glyph_trit_opcodes.h          # public table + enum + signatures
  glyph_trit_opcodes.c          # all 6 trit-op bodies, dense in one TU
  glyph_mtfp_opcodes.h          # public table + enum, no new code
  glyph_mtfp_opcodes.c          # the table itself (pointer array)
  glyph_trit_reducers.h         # signed_sum, sparsity declarations
  glyph_trit_reducers.c         # masked-VCNT bodies
  ... (existing glyph_mtfp.c, glyph_ternary_matmul.c, etc.)
tools/
  glyph_opcode_size_check.c     # link-time size budget check
  glyph_opcode_bench.c          # cycle-count harness
docs/
  TERNARY_OPCODE_SET.md         # caller-facing contract documentation
```

### Function-table layout for cache density

The two opcode-body `.c` files (`glyph_trit_opcodes.c` and the existing kernel files) must compile to **adjacent text sections** so that one cache line load brings in multiple opcode bodies. Achieved by either:

- Compiling all opcode TUs with `-ffunction-sections` and using a linker script to place them in a single `.text.glyph_opcodes` section, ordered by table index, **or**
- Defining all opcode bodies in a single `.c` file (loses TU isolation but gives the compiler/linker the strongest layout guarantee).

I recommend the second for v0: one TU, `glyph_opcodes.c`, that `#include`s the bodies of the existing kernels via `#include "glyph_mtfp_internal.c"` style (not the public `.c` files — refactored entry points). Sounds ugly, gives optimal density. Revisit if the maintenance cost shows up.

---

## Measurement

The contract is not honored until measured. Three measurements, all in `tools/glyph_opcode_bench.c`:

### M1 — code size against the budget

At link time, parse the resulting `.text` section sizes for opcode TUs and assert they sum to ≤ 16 KB. This is a CMake `add_custom_command` that runs after `libglyph.a` is built and fails the build if the budget is exceeded.

```bash
otool -l libglyph.a | awk '/__text/{...}' | sum_check 16384
```

(Real implementation uses `llvm-size` or `nm` parsing; the principle is the same.)

### M2 — cycle counts per opcode

For each opcode, run it 10⁶ times in a tight loop with hot inputs. Measure cycles per call using `mach_absolute_time` (M4 has nanosecond precision) and convert to cycles via the published clock rate. Record:

- Mean cycles per call
- p99 cycles per call (catches occasional cache miss)
- Cycles per element (for vector ops, divides by `n`)

The contract clause "predictable latency" is honored iff p99 ≤ 1.5 × mean. Anything above that ratio means a cold path is leaking through and we need to investigate (cache miss, branch mispredict, dispatch overhead).

### M3 — L1i hit rate under realistic load

Use Apple Instruments' `Counters` template with `INST_CACHE_MISS` and `INST_CACHE_ACCESS` PMCs. Run a glyph forward pass (once it exists) and measure miss rate over the opcode body addresses.

The contract clause "lives in L-cache" is honored iff `INST_CACHE_MISS / INST_CACHE_ACCESS < 0.5%` over the opcode body address range during a steady-state forward pass.

If we cannot satisfy this, the design is wrong — most likely the opcode bodies are interleaved with cold code in the link order, and the fix is the layout work in §Layout.

---

## Dispatch

### Internal use (most glyph callers)

```c
glyph_mtfp_vec_add(dst, a, b, n);   /* direct C call */
```

The compiler inlines trivially small ops and emits a normal call for the rest. No table lookup. No indirect branch.

### External use (runtime opcode selection)

```c
glyph_trit_op_t op = glyph_trit_ops[opcode_idx];
op(dst, a, b, n);
```

One indirect call. M4's indirect-branch predictor handles monomorphic call sites perfectly (one opcode per call site), and even polymorphic sites do well as long as the working set of opcodes per call site is small.

### Future use (bytecode VM)

Computed-goto threaded code dispatching over an opcode stream. Not in v0. Would be implemented as a third API surface over the same bodies — no kernel changes required, just a new `glyph_opcode_run(const uint8_t* program, ...)` entry point.

---

## Concrete v0 deliverables

In dependency order:

1. **`docs/TERNARY_OPCODE_SET.md`** — caller-facing contract documentation. The seven contract clauses, written for someone who is going to call into the opcode set without reading the implementation.

2. **`src/glyph_trit_opcodes.{h,c}`** — the 6 TBL-based binary trit ops. New code, ~600 LOC including LUTs and tests.

3. **`src/glyph_trit_reducers.{h,c}`** — `signed_sum` and `sparsity`. New code, ~200 LOC.

4. **`src/glyph_mtfp_opcodes.h`** — the table and enum. No new bodies. Just `extern const glyph_mtfp_op_t glyph_mtfp_ops[GLYPH_MOP_COUNT];` and the enum.

5. **`src/glyph_mtfp_opcodes.c`** — the table contents (pointers to existing functions). ~30 LOC.

6. **`tests/test_glyph_opcodes.c`** — golden tests for every new opcode (LUTs in particular need exhaustive 16×16 coverage), and one round-trip test that calls every entry of both tables via the indirect path.

7. **`tools/glyph_opcode_size_check.c`** — link-time budget enforcement. ~80 LOC.

8. **`tools/glyph_opcode_bench.c`** — cycle-count harness. ~150 LOC. Lands without M3 (Instruments work) initially; M3 lands when the first realistic glyph forward pass exists.

9. **`CMakeLists.txt`** — wire steps 7 and 8 into the build.

10. **`docs/REDTEAM_FIXES.md`** — log this design as the resolution path for the deferred opcode/contract questions.

Total new C code: ~1100 LOC. Total promoted existing code: ~0 LOC (pointers only). Total budget: well under 16 KB of `.text`.

---

## What this synthesis explicitly does *not* commit to

- A bytecode VM (deferred until there's a real stream-of-opcodes use case).
- JIT or self-modifying opcode bodies (banned by Apple W^X; not needed anyway).
- A unified single-shape opcode model (rejected — two shapes is honest).
- Replacing existing glyph kernels (we promote them; we don't rewrite them).
- Float boundary conversions inside opcodes (ban remains; tools/ stays the place).
- Per-trit-granularity ops (rejected — dispatch overhead dwarfs work).
- Performance claims without measurement (the contract requires evidence).

---

## Open questions still requiring user input before any of this lands

These are the ones I can't resolve from inside the LMM cycle:

- **Q1.** Is the L-cache claim caller-facing (needs docs + measurement) or internal-budget (needs measurement only)?
- **Q2.** Is the opcode set glyph-internal or generic substrate?
- **Q3.** Is the indirect-dispatch surface for an external API or future VM?
- **Q4.** Confirm the 16 KB budget — too generous, too tight, just right?
- **Q5.** Which new opcodes does the routing layer actually need that I haven't anticipated?

---

## Honest assessment of this synthesis

What I am confident about: the contract framing, the two-table structure, the promotion-not-rewrite path for existing kernels, the TBL- and VCNT-based new opcodes, the measurement approach, the deferral list.

What I'm guessing about: the 16 KB budget number (it's defensible but not measured), M4's indirect-branch predictor quality (asserted, not measured), the exact size of each new opcode body (estimated from instruction count, not compiled and weighed).

What this synthesis does well: it converts a vague brief into a specific deliverable list with budgets, measurements, and an honest "what could go wrong" section.

What this synthesis does poorly: it commits to a contract before measuring whether the contract is achievable. That's intentional — the contract is the design's core claim and the measurement work is downstream — but it does mean the v0 deliverables include a plausible failure mode where M2 or M3 reveals the contract isn't achievable on M4 with this layout, forcing a redesign.

The single most useful next action: build the cycle-count harness (M2) on the *existing* glyph kernels, get baseline numbers, and use those numbers to calibrate the budget and the contract clauses *before* writing any new opcode code.
