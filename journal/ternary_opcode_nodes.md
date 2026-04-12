# Nodes of Interest: Ternary Opcode Set in L-Cache

Extracted from `ternary_opcode_raw.md`. Numbered for reference; tensions and dependencies marked explicitly. Not solving anything here — just mapping the grain.

---

## Node 1: TBL is the universal binary ternary op

`vqtbl1q_s8` against a 16-byte LUT can implement *any* 2-operand ternary function (mul, sat-add, max, min, equality, custom gates). Cost is one TBL plus the work to pack `(a_code << 2) | b_code` into a uint8 index. The LUT itself is 16 bytes per opcode.

**Why it matters:** the entire 2-operand ternary opcode space — every binary ternary function — collapses into a uniform shape: a 16-byte LUT plus a ~5-instruction stub. That's the cleanest, smallest, most cache-friendly opcode shape we could ask for.

---

## Node 2: Two opcode shapes, not one

Trit-vector ops (operating on packed `uint8_t*` containers) and MTFP-vector ops (operating on `int32_t*` cells) are fundamentally different sizes, calling conventions, and instruction mixes. A single uniform "opcode" abstraction will paper over this and either bloat the trit ops or cripple the MTFP ops.

**Why it matters:** the design will live or die on whether we accept the asymmetry. Forcing a single uniform shape is a premature commitment.

---

## Node 3: Granularity is the central design choice

Three plausible granularities:
- **Per-trit-op** (one trit at a time): dispatch overhead dwarfs work. Dead.
- **Per-vector-op** (one operation on a contiguous buffer of N trits or N MTFP cells): dispatch amortizes across the loop. Healthy.
- **Per-layer** (whole routed FFN, whole layernorm): not really an "opcode" — it's a function. Misses the point.

**Why it matters:** picking wrong gives you either a slow interpreter or a fancy synonym for a function library.

---

## Node 4: L-cache residency is a contract, not just a wish

"Lives in L-Cache" is a perf-binding promise: predictable latency, no warm-up surprises, no cache thrash from co-resident code. To deliver, the entire opcode set body + LUTs has to fit in a known fraction of L1i with margin, and the access pattern has to keep it hot under realistic glyph workloads.

**Why it matters:** without measurement, the L-cache claim is rhetoric. With measurement, it's a feature you can defend.

---

## Node 5: Dispatch shape determines everything downstream

Three dispatch styles:
- **Direct C call**: compiler can inline at known call sites; no dispatch overhead; no per-op decode.
- **Indirect call via opcode index** (function-pointer table): runtime selectable; one indirect-branch predictor entry per opcode; works well if the dispatch site is monomorphic per call site.
- **Computed-goto threaded code** (bytecode interpreter): smallest per-op overhead in the dispatcher loop; biggest cognitive cost; only worth it if there's a real "stream of opcodes" use case.

**Why it matters:** different dispatch shapes target different use cases. Direct call is for the static glyph internal kernels; indirect call is for an external API where the caller picks an op at runtime; threaded code is for a future ternary VM.

---

## Node 6: "Extends the silicon" is a contract framing, not an implementation framing

The phrase "extends the silicon" sounds like it's about how the code is laid out in memory or cache. It's actually about what the *caller* can assume: that ternary operations are always available, always fast, always cheap, with predictable latency. The implementation can be anything (inline, called, threaded) as long as the contract holds.

**Why it matters:** treating "extends the silicon" as a contract rather than a layout decouples the design from the implementation. We can pick whatever implementation makes the contract cheap to honor.

---

## Node 7: M4's branch predictor and indirect-call behavior is unknown territory

We don't have measurements for indirect-branch prediction quality on Apple M4. If indirect dispatch eats 10-cycle bubbles per opcode call, the entire concept of an "opcode set" callable by index is dead and we have to inline everything. If it predicts perfectly under realistic patterns, indirect dispatch is fine.

**Why it matters:** this is the single largest unknown that could invalidate the design. It needs an early measurement before we commit.

---

## Node 8: Inline vs out-of-line is a real tradeoff, not a free lunch

- Inline: zero call overhead, perfect inlining, but every call site duplicates the body. Many call sites = I-cache fragmentation.
- Out-of-line: one copy of each opcode body, dense in I-cache, but call/return overhead and possible indirect-branch misses.

**Why it matters:** modern CPUs are good at call/return for short hot functions, so the conventional wisdom "inline for speed" may not hold here. Worth measuring rather than assuming.

---

## Node 9: TBL-based opcodes are tiny and uniform; MTFP opcodes are bigger and varied

A trit-op opcode body is ~5 instructions plus a 16-byte LUT — call it ~80 bytes. An MTFP-op opcode body (e.g., vec_add with saturation) is ~12 instructions plus splat constants — call it ~80–150 bytes. So the magnitude is similar, but the trit ops are uniform shape (load LUT, TBL, store) while MTFP ops vary by operation. Uniform shape simplifies a dispatch table; variable shape means each opcode needs its own entry point.

**Why it matters:** if we want a clean, indexable opcode table, we get it for free for trit ops and have to work for MTFP ops.

---

## Node 10: Glyph already has functions that look like opcodes

`glyph_mtfp_vec_add`, `glyph_mtfp_vec_add_inplace`, `glyph_mtfp_vec_scale`, `glyph_mtfp_bias_add`, `glyph_mtfp_layernorm`, `glyph_mtfp_matmul_bt`, `glyph_mtfp_ternary_matmul_bt` — these are all already in the shape "operate on a contiguous buffer with a small uniform calling convention." They are *almost* an opcode set already.

**Why it matters:** this is the strongest signal that the right answer is incremental, not a from-scratch redesign. The "ternary opcode set" may turn out to be "the existing glyph kernels, organized into a discoverable table, with cache-residency guarantees and a name for the contract."

---

## Tensions

### Tension A: Two opcode shapes vs one (Node 2 vs Node 9)

A uniform opcode shape (everything is "load LUT, TBL, store") is conceptually clean and dispatch-friendly, but it forces MTFP ops into a model they don't fit. Two distinct shapes are honest but require two separate opcode tables and two separate dispatch contracts.

### Tension B: Dispatch style vs use case (Node 5)

Direct C calls are best for glyph's internal hot paths (we know which op runs at every call site, the compiler can inline it). Indirect-table dispatch is required for any external API where the user picks an op at runtime. Threaded-code is overkill for now but is the only way to get a true ternary VM later. Picking one *implementation* commits us to one *use case*.

### Tension C: L-cache contract vs measurement gap (Node 4 vs Node 7)

The contract requires evidence of L1i residency and indirect-branch prediction quality, neither of which we currently have. The contract cannot be honored without measurement infrastructure that doesn't yet exist in glyph. So either we build the measurement first (slows everything down) or we ship the design with a "to be measured" caveat (weakens the contract).

### Tension D: Greenfield design vs existing kernels (Node 10)

The existing glyph functions are already opcode-shaped. A clean-sheet "opcode set" would either replace them (churn, regression risk, no payoff if they're already good) or duplicate them (two parallel APIs, maintenance burden). The third path — *promote* the existing functions to opcodes by adding a contract layer on top — is the cheapest and lowest risk, but it's not "designing a new opcode set," it's "naming an existing one."

### Tension E: Inline vs out-of-line (Node 8)

These are mutually exclusive at the call site. We can't have both. We can offer both via duplicate APIs (`glyph_op_*` for out-of-line, `static inline` headers for inline) but that's API surface bloat.

---

## Dependencies

- **Node 1 → Node 9**: TBL universality is what makes trit-op uniform shape possible. Without TBL, even trit ops would be heterogeneous.
- **Node 4 → Node 7**: residency contract depends on indirect-branch behavior, because if dispatch is slow then "fits in L1i" doesn't help.
- **Node 6 → Node 5**: contract framing dictates dispatch flexibility. A pure-internal contract permits direct calls; an external contract forces indirect dispatch or computed-goto.
- **Node 10 → Tension D**: existence of the current kernels is the dependency that turns "design new" into "promote existing."
