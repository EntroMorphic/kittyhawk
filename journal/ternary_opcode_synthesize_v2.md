# Synthesis v2: Glyph Ternary Opcode Set

Supersedes `ternary_opcode_synthesize.md`. Triggered by user answers to Q1–Q5 from v1. v1 stays as a historical record; this file is the actionable artifact going forward.

---

## What changed since v1

The five open questions resolved as follows. Each answer is a real design constraint, not a refinement.

| Q | Answer | Consequence |
|---|---|---|
| **Q1.** L-cache claim: caller-facing or internal budget? | "Whichever gets us truer to an extension of the silicon." → **both** | Caller-facing docs *and* measurement gating. Contract is not published until M1+M2+M3 pass. |
| **Q2.** Generic substrate or glyph-internal? | **Generic.** Any application needing ternary compute on M4. | Rename out of `glyph_*` namespace. Substrate is its own thing; glyph is one consumer. |
| **Q3.** External API or future VM? | **External API for now.** | Function-pointer table is the canonical surface. Indirect-dispatch quality is critical, must measure. |
| **Q4.** What's optimal for Ternary/Multi-Trit/MTFP21? | (open — see analysis below) | Two cell widths shipped: MTFP19 (int32, fast) and MTFP31 (int64, wide). MTFP21 is a documented subset of MTFP31. |
| **Q5.** What new opcodes does glyph actually need? | **Ternary routing primitives.** | Third opcode table for routing-shaped ops. Five new primitives. |

---

## One-line v2 answer

**M4T (M4 Ternary Extensions) is a generic, cache-resident, externally-callable opcode substrate for ternary and multi-trit compute on Apple Silicon. It exposes three indexable function-pointer tables — `m4t_trit_ops[]`, `m4t_mtfp_ops[]`, `m4t_route_ops[]` — built on NEON's TBL/VCNT/SDOT instructions, with measured L1i residency and a documented seven-clause contract that makes the substrate caller-indistinguishable from native silicon. Glyph becomes the first consumer.**

---

## Naming and namespace

The substrate is **M4T** — short for *M4 Ternary*. Project name, library name, header prefix.

| Layer | Old (v1) | New (v2) |
|---|---|---|
| Namespace | `glyph_*` | `m4t_*` |
| Default cell type | `glyph_mtfp_t` (int32) | `m4t_mtfp_t` (int32, MTFP19) |
| Wide cell type | (none) | `m4t_mtfp_w_t` (int64, MTFP31; aliased as `m4t_mtfp21_t` for the strictly-21-trit subset) |
| Trit type | `glyph_trit_t` | `m4t_trit_t` |
| Trit-vector ops | (existing) | `m4t_trit_*` |
| MTFP-cell ops | `glyph_mtfp_*` | `m4t_mtfp_*` |
| Routing ops | (none) | `m4t_route_*` |
| Library artifact | `libglyph.a` | `libm4t.a` (+ `libglyph.a` becomes a thin layer that depends on `m4t`) |

Glyph is preserved as the consuming project — its existing tests, its documentation, its routed-FFN plans — but its numerical core *is* M4T. The rename is mechanical and one-time.

---

## Cell width — Q4 decision in detail

### MTFP19 — the default fast path (int32)

```c
typedef int32_t m4t_mtfp_t;
#define M4T_MTFP_RADIX        10
#define M4T_MTFP_SCALE        59049              /* 3^10 */
#define M4T_MTFP_TRITS        19
#define M4T_MTFP_MAX_VAL      ((m4t_mtfp_t)581130733)  /* (3^19 − 1)/2 */
```

- **Trit count:** exactly 19 (clean power-of-3 boundary).
- **Real range:** `±9842.something`.
- **NEON SIMD:** full `int32x4` lanes, 4 cells per 128-bit register.
- **Memory:** 4 bytes per cell.
- **NEW2 invariant preserved:** `2·MAX_VAL = 1,162,261,466 < INT32_MAX = 2,147,483,647`, so non-saturating `vaddq_s32` followed by `vminq/vmaxq` clamp is still safe.
- **Difference vs current glyph:** glyph today uses `MAX_VAL = INT32_MAX/2 ≈ 1.07e9`, which is between 19 and 20 trits. The v2 rename pins MAX_VAL to exactly `(3¹⁹−1)/2`, sacrificing ~1.85× of dynamic range for honesty about the trit count.

**This is the canonical M4T cell.** Most callers should use this.

### MTFP31 — the wide path (int64)

```c
typedef int64_t m4t_mtfp_w_t;
#define M4T_MTFPW_RADIX       10
#define M4T_MTFPW_SCALE       59049
#define M4T_MTFPW_TRITS       31
#define M4T_MTFPW_MAX_VAL     ((m4t_mtfp_w_t)308836698141971ll)  /* (3^31 − 1)/2 */
```

- **Trit count:** 31.
- **Real range:** `±5.23 × 10⁹` cells / 59049 ≈ `±5.23 × 10⁹` ... wait, real range = MAX/SCALE = `3.09 × 10¹⁴ / 59049 ≈ 5.23 × 10⁹` real units. Generous.
- **NEON SIMD:** half the throughput — `int64x2` is 2 lanes per register vs 4 for int32. Most arithmetic primitives have int64 variants (`vaddq_s64`, etc.) but multiply is more expensive.
- **Memory:** 8 bytes per cell, 2× bandwidth.
- **NEW2 equivalent:** `2·MAX_VAL = 6.18 × 10¹⁴ < INT64_MAX = 9.22 × 10¹⁸`, so the same non-saturating-add-then-clamp pattern works in int64.

### MTFP21 as a subset of MTFP31

The trix-z spec name "MTFP21" describes 21 trits, range `(3²¹−1)/2 ≈ 5.23 × 10⁹` cell range, real `±88.3`. This fits inside MTFP31 trivially. We expose MTFP21 as a *documented constraint* on MTFP31, not as a separate cell type:

```c
/* MTFP21: a 21-trit subset of MTFP31. Same storage type, narrower clamp. */
#define M4T_MTFP21_MAX_VAL    ((m4t_mtfp_w_t)5230176601ll)  /* (3^21 − 1)/2 */

static inline m4t_mtfp_w_t m4t_mtfp21_clamp(m4t_mtfp_w_t v) {
    if (v >  M4T_MTFP21_MAX_VAL) return  M4T_MTFP21_MAX_VAL;
    if (v < -M4T_MTFP21_MAX_VAL) return -M4T_MTFP21_MAX_VAL;
    return v;
}
```

Callers who need exactly the trix-z MTFP21 semantics use the wide cell type and apply this clamp. Callers who don't care about the exact trit count just use MTFP31. Less fragmentation, same coverage.

### Why two cells, not one or three

- **One cell (int32 only):** matches glyph today, fastest, but caps the substrate at ~10K real range. Applications needing wider dynamic range have nowhere to go. Fails the "generic substrate" goal of Q2.
- **One cell (int64 only):** matches the spec name, gives unlimited (within int64) range, but halves NEON throughput on every workload. Pessimizes the common case for the rare case.
- **Two cells (int32 + int64):** caller picks. Common case stays fast. Wide case is available. Two parallel opcode tables, modest extra implementation cost.
- **Three cells (int16 + int32 + int64):** int16 is banned by the policy. Out.

Two cells is the right answer. Both honor the same seven-clause contract.

---

## The seven-clause contract (unchanged from v1)

A function `f` is an M4T opcode iff it satisfies all of:

1. **Vector-op granularity.** Operates on contiguous buffers, length parameterized.
2. **No allocations.** Caller-provided scratch.
3. **No errors.** Asserted preconditions in debug builds.
4. **Predictable latency.** Closed-form runtime in `n` (or `M·N·K`).
5. **Cache-resident under load.** Body and data tables fit measured size budgets.
6. **Indexable.** Direct C call *and* function-pointer-table entry.
7. **Caller-indistinguishable from silicon.**

Q1 raises the bar on clause 5: residency must be **measured**, not asserted. The contract is not publishable until M1+M2+M3 pass.

---

## Three tables, one substrate

### Table 1 — `m4t_trit_ops[]`

Calling convention:
```c
typedef void (*m4t_trit_op_t)(
    uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits);
```

| Index | Name | Description | Implementation |
|---|---|---|---|
| `M4T_TOP_MUL` | `m4t_trit_mul` | F₃ multiply | TBL on 16-byte LUT |
| `M4T_TOP_SAT_ADD` | `m4t_trit_sat_add` | clamped ternary add | TBL on 16-byte LUT |
| `M4T_TOP_MAX` | `m4t_trit_max` | pairwise max | TBL on 16-byte LUT |
| `M4T_TOP_MIN` | `m4t_trit_min` | pairwise min | TBL on 16-byte LUT |
| `M4T_TOP_EQ` | `m4t_trit_eq` | equality → +1/0 | TBL on 16-byte LUT |
| `M4T_TOP_NEG` | `m4t_trit_neg` | sign flip (b ignored) | TBL on swap LUT |

All six bodies share one shape: load 16 trits from `a` and `b`, pack pair indices, single `vqtbl1q_u8` lookup, store. Estimated ~80 bytes per body, ~96 bytes per LUT (in `.rodata`). **Total: ~1 KB code + ~0.6 KB data.**

### Table 2 — `m4t_mtfp_ops[]` (MTFP19, int32)

Three sub-shapes (variant signatures stored in the table alongside the function pointer).

| Index | Name | Shape | Source |
|---|---|---|---|
| `M4T_MOP_VEC_ADD` | `m4t_mtfp_vec_add` | binary | promoted from glyph |
| `M4T_MOP_VEC_ADD_INPLACE` | `m4t_mtfp_vec_add_inplace` | binary | promoted |
| `M4T_MOP_VEC_SCALE` | `m4t_mtfp_vec_scale` | unary+scalar | promoted |
| `M4T_MOP_BIAS_ADD` | `m4t_mtfp_bias_add` | binary, broadcast | promoted |
| `M4T_MOP_FAN_IN_NORMALIZE` | `m4t_mtfp_fan_in_normalize` | unary+scalar | promoted |
| `M4T_MOP_LAYERNORM` | `m4t_mtfp_layernorm` | layernorm shape | promoted |
| `M4T_MOP_MATMUL` | `m4t_mtfp_matmul` | matmul shape | promoted |
| `M4T_MOP_MATMUL_BT` | `m4t_mtfp_matmul_bt` | matmul shape | promoted |
| `M4T_MOP_TERNARY_MATMUL_BT` | `m4t_mtfp_ternary_matmul_bt` | matmul shape, packed-trit weights | promoted |

Plus future entries for softmax and GELU once the LUT generator lands. **Total promoted: ~3 KB code + LUTs deferred.**

### Table 2-wide — `m4t_mtfp_w_ops[]` (MTFP31, int64)

Same enum / same names, wide cell variants. Implementation is a parallel rewrite using `int64x2_t` intrinsics. Lands in v0.1, after MTFP19 path is measured and stable.

### Table 3 — `m4t_route_ops[]`

Q5 says glyph needs ternary routing primitives. The five most concrete:

| Index | Name | Inputs | Output | Implementation |
|---|---|---|---|---|
| `M4T_ROP_SIGNATURE_UPDATE` | `m4t_route_signature_update` | `[T, in_dim]` packed-trit weights | `[T, in_dim]` packed-trit signatures | per-tile: column reduction via masked VCNT, mean subtract, sign extract |
| `M4T_ROP_DISTANCE_BATCH` | `m4t_route_distance_batch` | one query packed-trit signature, T tile signatures | `[T]` int32 distances | T parallel calls to existing `popcount_dist` (promoted from glyph) |
| `M4T_ROP_TOPK_ABS` | `m4t_route_topk_abs` | `[T]` int32 scores, k | `[k]` (index, sign) pairs | small selection sort or partial heapselect, T typically ≤ 16 |
| `M4T_ROP_APPLY_SIGNED` | `m4t_route_apply_signed` | `[k]` (index, sign) pairs, per-tile output buffers, accumulator | accumulator updated | inner loop: for each (index, sign), `vec_add_inplace` (sign=+1) or sub (sign=−1) |
| `M4T_ROP_SIGN_EXTRACT` | `m4t_route_sign_extract` | int32 score array | packed-trit sign array | NEON `vshrq_n_s32` for sign bit, pack to trits via TBL |

These five are the **routing layer of trix-z's `trix_ternary_route.c`, decomposed into reusable primitives**. The current 680-line monolith becomes a sequence of opcode calls. Each primitive is independently testable, measurable, and cache-resident.

**Critical observation:** primitive #1 (signature update) uses the masked-VCNT trick from the M4 deep-dive. That's the second-biggest win from the hardware analysis (after TBL-based binary trit ops). Without M4T, this primitive would land buried inside a routing implementation; with M4T, it's a first-class opcode that any consumer can call.

---

## Cache budget (Q1 impact)

Now caller-facing, must be measured.

| Region | M4 P-core L1 | M4T reservation | Slack |
|---|---|---|---|
| L1i | 192 KB | **24 KB hard cap** for opcode bodies | 168 KB caller code |
| L1d | 128 KB | **4 KB hard cap** for LUTs + constants | 124 KB activation buffers |

24 KB code budget breakdown (estimated):
- `m4t_trit_ops`: ~1 KB (6 ops × ~150 B average)
- `m4t_mtfp_ops` (MTFP19): ~6 KB (9 ops, varying sizes)
- `m4t_mtfp_w_ops` (MTFP31): ~8 KB (parallel of above, slightly bigger int64 bodies)
- `m4t_route_ops`: ~5 KB (5 ops, signature update is largest)
- Future softmax + GELU LUT loaders + dispatchers: ~3 KB
- Headroom for v0.x growth: ~1 KB

Sits at 23 KB, 1 KB headroom, fits the cap.

**Enforcement:** `tools/m4t_size_check.c` runs at link time, parses the `.text.m4t_*` section sizes from `libm4t.a` via `llvm-size`, asserts each table's bodies sum within budget, fails the build otherwise.

---

## Measurement (M1, M2, M3)

Q1 makes these gating, not optional. The contract is not published until all three pass.

### M1 — code size budget (link-time)

`tools/m4t_size_check.c`. Runs as a CMake `add_custom_command` after `libm4t.a` builds. Reads section sizes via `llvm-size --format=sysv libm4t.a`, sums bodies per table, asserts each ≤ its cap, fails the build on overage. **Lands in v0.**

### M2 — cycle-count harness

`tools/m4t_bench.c`. For each opcode in each table, runs the op 10⁶ times in a tight hot loop, measures via `mach_absolute_time`. Records:

- mean cycles per call
- p99 cycles per call
- mean cycles per element (for vector ops, divides by `n`)

Gating clause: **p99 ≤ 1.5 × mean**. Anything wider means a cold-cache leak. **Lands in v0.**

### M3 — L1i hit-rate under realistic load

Apple Instruments `Counters` template, PMCs `INST_CACHE_MISS` and `INST_CACHE_ACCESS`, restricted to the address range of M4T's `.text.m4t_*` section. Runs against a *real consumer workload*: glyph's first end-to-end transformer forward pass.

Gating clause: `INST_CACHE_MISS / INST_CACHE_ACCESS < 0.5%` over the M4T address range during steady-state.

**Lands when glyph has a real transformer forward pass.** Not gating for v0 of M4T itself (since the consumer doesn't exist yet), but gating for the *contract publication*.

### Indirect-dispatch measurement (extension to M2)

Q3 makes external API canonical. M2 must distinguish:

- **Direct-call cycles:** function called by name, compiler may inline.
- **Indexed-call cycles:** function called via `table[idx](args)`, indirect branch.

The contract clause "predictable latency" must hold for *both*. If indexed-call is significantly slower than direct-call (more than ~5 cycles per opcode call), we have to either accept the gap and document it, or flatten the dispatch (e.g., `static inline` thunks for small ops, real indirect-call for large ops).

---

## File structure

```
m4t/                            # the substrate (new top-level dir, or sibling to glyph)
  src/
    m4t_types.h                 # m4t_mtfp_t, m4t_mtfp_w_t, m4t_trit_t, constants
    m4t_internal.h              # private platform macros
    m4t_trit_pack.{h,c}         # promoted from glyph_trit_pack.{h,c}
    m4t_trit_ops.{h,c}          # NEW — 6 TBL-based binary trit ops
    m4t_trit_reducers.{h,c}     # NEW — masked-VCNT signed_sum, sparsity
    m4t_mtfp.{h,c}              # promoted from glyph_mtfp.{h,c}
    m4t_mtfp_ternary_matmul.{h,c} # promoted from glyph_ternary_matmul.{h,c}
    m4t_mtfp_w.{h,c}            # NEW — int64 wide variants (v0.1)
    m4t_mtfp_ops.{h,c}          # the table (function pointers + shape tags)
    m4t_route.{h,c}             # NEW — 5 routing primitives
    m4t_route_ops.{h,c}         # the route table
    m4t_contract.h              # the seven contract clauses as comments + helper macros
  tools/
    m4t_size_check.c            # M1 link-time budget enforcement
    m4t_bench.c                 # M2 cycle-count harness
  tests/
    test_m4t_trit.c             # (existing glyph trit tests, renamed)
    test_m4t_mtfp.c             # (existing glyph mtfp tests, renamed)
    test_m4t_trit_ops.c         # NEW — exhaustive 16-entry coverage of TBL LUTs
    test_m4t_route.c            # NEW — routing primitive tests
    test_m4t_dispatch.c         # NEW — round-trip via function-pointer tables
  docs/
    M4T_CONTRACT.md             # caller-facing seven-clause contract documentation
    M4T_OPCODES.md              # the three tables, each entry documented
    M4T_PERF.md                 # M2 + M3 results (populated post-measurement)
  CMakeLists.txt                # builds libm4t.a, runs M1 at link time

glyph/                          # the consumer
  src/
    (existing glyph code, after the rename, becomes thin wrappers around m4t)
  CMakeLists.txt                # depends on m4t
```

**Two paths for the m4t/ location:**

- **(P1) Sibling repo.** `m4t/` lives at `~/Projects/m4t/`, glyph depends on it via CMake `find_package` or git submodule. Fully separates the substrate from the consumer.
- **(P2) Subdirectory of glyph.** `glyph/m4t/` is a self-contained subtree, builds `libm4t.a` independently, glyph builds `libglyph.a` on top. One repo, clean separation by directory.

**Recommendation: P2 for v0**, then promote to P1 once a second consumer exists. Keeps everything in one place during the rename, preserves git history, lets us iterate quickly. P1 is the right end state but premature today.

---

## Migration plan — the rename

The work is mechanical but invasive. ~12 source files renamed, ~80 symbol references updated, all tests must pass after each step.

### Phase 1 — substrate extraction

1. Create `glyph/m4t/` subtree.
2. `git mv` files: `glyph_types.h → m4t_types.h`, `glyph_mtfp.{h,c} → m4t_mtfp.{h,c}`, `glyph_trit_pack.{h,c} → m4t_trit_pack.{h,c}`, `glyph_ternary_matmul.{h,c} → m4t_mtfp_ternary_matmul.{h,c}`, `glyph_internal.h → m4t_internal.h`.
3. Mass rename in moved files: `glyph_*` → `m4t_*` (regex, then audit).
4. Update `MAX_VAL` to `(3¹⁹ − 1)/2 = 581130733` (the cell-width tightening). This *will* break tests that use `INT32_MAX/2` directly — fix them.
5. Add `m4t/CMakeLists.txt` building `libm4t.a`.
6. Run all tests, fix breakage.

### Phase 2 — glyph wrapper layer

7. Create thin `glyph/src/glyph_*.h` headers that `#include <m4t_*.h>` and `#define glyph_mtfp_t m4t_mtfp_t` etc. Existing glyph callers (none yet outside tests) keep working.
8. Update top-level CMakeLists to depend on `libm4t.a`.

### Phase 3 — opcode set additions

9. Write `m4t_trit_ops.{h,c}` with the six TBL-based primitives. Land tests.
10. Write `m4t_trit_reducers.{h,c}` with `signed_sum` and `sparsity`. Land tests.
11. Write `m4t_mtfp_ops.{h,c}` and `m4t_route_ops.{h,c}` — the function-pointer tables. Mostly pointer assignments, no new bodies.
12. Write `m4t_route.{h,c}` with the five routing primitives. Land tests.

### Phase 4 — measurement infrastructure

13. Write `tools/m4t_size_check.c` (M1).
14. Wire M1 into CMake as a post-link custom command.
15. Write `tools/m4t_bench.c` (M2). Run it. Record baseline numbers in `M4T_PERF.md`.
16. M3 deferred until glyph has a transformer forward pass.

### Phase 5 — wide cell type (v0.1)

17. Write `m4t_mtfp_w.{h,c}` with int64 cell type and parallel arithmetic primitives.
18. Write `m4t_mtfp_w_ops.{h,c}` parallel table.
19. Tests, M1 + M2 against the new bodies.

### Phase 6 — contract publication

20. Write `M4T_CONTRACT.md` and `M4T_OPCODES.md`.
21. Verify M1 + M2 pass and document results.
22. Tag v0.

**Phases 1–4 are v0. Phase 5 is v0.1. Phase 6 gates the public release.**

---

## What v2 explicitly does *not* commit to

- A bytecode VM (deferred — Q3 says external API is enough).
- JIT or self-modifying opcode bodies (W^X precludes; not needed).
- Float boundary conversions inside `libm4t.a` (banned by glyph policy, inherited).
- Per-trit-granularity ops (rejected).
- Replacing existing glyph kernels (we promote them via rename).
- Performance claims without M1+M2+M3 evidence.
- Cell types narrower than int32 (banned by policy).
- Cell types between int32 and int64 (no clean fit; pick one or the other).

---

## Open questions for the next round

These genuinely require user input before phase work begins.

- **Q6 (was Q4 partial).** The MTFP19 cell rename loses ~1.85× of dynamic range vs the current MTFP19.5. Is this an acceptable cost for the clean trit boundary? If not, we can ship MTFP19.5 (current behavior) and document it as "approximately 19 trits" — less honest, more range.
- **Q7.** P1 (sibling repo) vs P2 (subdirectory) for `m4t/` — confirm P2 for v0?
- **Q8.** Should `libm4t.a` depend on libdispatch (current glyph behavior, Apple-only) or be libdispatch-free at the substrate level so it can run on any aarch64+NEON, with libdispatch as a glyph-layer concern? The latter is more honest about "extends *aarch64* silicon."
- **Q9.** v0 includes phases 1–4 (substrate + measurement). Phase 5 (MTFP31 wide cell) lands separately. Is that sequencing correct, or should MTFP31 land in v0 alongside MTFP19?
- **Q10.** Consumer naming — does "glyph" stay as the application layer, or does it get a new identity now that the numerical core has migrated?

---

## Honest assessment of v2

**What I'm confident about:**
- The naming move (m4t out of glyph) is right and overdue.
- The two-cell decision (MTFP19 + MTFP31) is the cleanest answer to "optimal for MTFP21."
- The third opcode table (`m4t_route_ops`) is the right shape for Q5.
- The seven-clause contract holds up under generic-substrate scope.
- Measurement gating from Q1 is the most important constraint added by this round.

**What I'm guessing about:**
- The 24 KB body budget is defensible but not measured. Could be too tight if route ops grow.
- The MTFP19 cell tightening (`MAX_VAL = (3¹⁹−1)/2`) feels right but I haven't checked which existing glyph callers depend on values in the `(3¹⁹/2, INT32_MAX/2]` range.
- The five routing primitives are decomposed from trix-z's monolith — they may need refinement once we actually try to compose them into a full routing path.
- M2 measurement of indirect-call vs direct-call will likely surface a real cost gap; I'm guessing 2–8 cycles per call but I have no data.

**The single most useful next action:**
Phase 1 step 4 — try the MTFP19 rename and run the existing tests. If they pass, the cell tightening is free and we proceed. If they fail, we've discovered a hidden range dependency and Q6 becomes urgent.
