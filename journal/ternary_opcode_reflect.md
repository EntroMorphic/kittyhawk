# Reflections: Ternary Opcode Set in L-Cache

Working from `ternary_opcode_nodes.md`. The job here is to find the structure beneath the nodes — the pattern that makes the tensions dissolve rather than get balanced.

---

## The "why" ladder

Asking *why* three times on the original brief:

1. **Why a "ternary opcode set"?** Because glyph needs a contract that ternary operations are first-class on M4, not an awkward emulation under the hood that occasionally peeks through.
2. **Why "lives in L-cache"?** Because the contract is meaningless if performance is unpredictable. Cache residency is the most concrete proxy for "predictable, fast, cheap."
3. **Why "extends the silicon"?** Because the user's mental model is hardware-first. They want a *thing they can program against* that behaves like new instructions, not a *library they call into* that behaves like a dependency.

Underneath all three: **the user is asking me to deliver a hardware-feeling abstraction over a software-only implementation.** The phrase "extends the silicon" is doing the most work in the brief. It's the constraint.

---

## Core insight

> **An opcode set is a contract, not a layout. The L-cache requirement is one term in the contract, and it constrains size and access pattern but not implementation. The "extend the silicon" framing means the caller cannot tell whether they're calling silicon or software — that is the *only* hard requirement.**

Everything else — direct call vs indirect dispatch vs threaded code, uniform vs split shape, inline vs out-of-line — is a question of *how to honor that contract for a given use case*, not *what the contract is*.

Once you see this, the tensions stop feeling like binary forks and start feeling like positions on a sliding scale where different glyph callers sit at different points.

---

## Resolved tensions

### Tension A — two shapes vs one (Node 2 vs Node 9)

**Resolution: two shapes, one contract.** The contract is the same for trit-vector ops and MTFP-vector ops: predictable latency, fits in L1i, indexable, no allocations, no errors. The implementation is two distinct kernel families because the underlying NEON instructions are different. We do not collapse them into a uniform shape because that would lie about the silicon. We *do* require both to satisfy the same contract terms.

Concretely: there are two opcode tables (`trit_ops[]` and `mtfp_ops[]`), each indexable, each with its own calling convention, and the contract clause "lives in L-cache" is checked against the *union* of both tables.

### Tension B — dispatch style vs use case (Node 5)

**Resolution: direct call is the implementation; indirect dispatch is one API surface among several.** Glyph's hot internal kernels (the ones called from routed-FFN inner loops, etc.) use direct C calls because the call site knows the op statically and the compiler can inline. The same kernel bodies are *also* exposed via a function-pointer table for callers who select an op at runtime — the table just holds pointers to the same direct-call functions. There is no second implementation; the table is a thin index over the existing entry points.

This collapses three apparent "dispatch styles" into one implementation with two API surfaces:
- **Internal use:** direct call, inlinable, zero overhead.
- **External use:** index into table → indirect call → same body.

Threaded-code (a bytecode interpreter) is deferred. It would be a third API surface over the same bodies, but only if there's a real "stream of opcodes" use case, which we don't have today.

### Tension C — contract vs measurement gap (Node 4 vs Node 7)

**Resolution: ship the contract with a measurement dependency, build the measurement *before* claiming the contract is honored.** The opcode set design can land without proven L1i residency, but the contract cannot be advertised until we have:
1. A measurement that the bodies fit in a documented size budget.
2. A measurement that indirect-branch prediction is fast enough for the indexed-call path.

Until those two measurements exist, the design is "an opcode set whose contract is not yet honored." That's an honest position. It's not "lives in L-cache" — it's "designed to live in L-cache, pending measurement."

This is the truth-vs-hype move: we don't get to claim the contract until we can defend it. We *do* get to design toward the contract from day one, which is what most of the rest of the design is about.

### Tension D — greenfield vs existing kernels (Node 10)

**Resolution: the existing kernels are the opcode set.** They are already in the right shape (vector-op granularity, contiguous-buffer calling convention, no allocations, no errors). What's missing is:
1. A naming convention that marks them as opcodes (the "extends silicon" contract layer).
2. An indexable table that exposes them as a unified set.
3. A size budget and a layout strategy that keeps them L1i-resident.
4. A small set of *new* opcodes that don't currently exist (notably TBL-based trit-vector ops, and the masked-VCNT ternary reducer we identified earlier).

So the work is mostly *organize and contract*, not *design new*. The greenfield piece is small: ~5 new TBL-based trit ops and ~2 new VCNT-based reducers. Everything else exists.

This is the cheapest possible answer to the brief, and it's also the right one because it preserves all the existing test coverage.

### Tension E — inline vs out-of-line (Node 8)

**Resolution: out-of-line bodies, inline-friendly headers, measurable.** The bodies live in `.c` files (one copy each, dense in I-cache). The headers expose `static inline` wrappers around the body addresses *only for trivial opcodes* where inlining is provably better. For everything bigger than ~5 instructions, we let the linker emit the call and trust M4's call/return prediction.

The decision rule: if the opcode body is smaller than the call-site call/return overhead (~6 cycles), inline it. Otherwise, out-of-line it. This is exactly what `static inline` already does for the trivial primitives in `glyph_mtfp.h` (`add`, `sub`, `mul_trit`, `clamp64`).

So nothing changes for the trivial ops; they stay inline. The medium and large ops go out-of-line in the opcode bodies. We measure both and don't second-guess the call/return predictor without evidence.

---

## Hidden assumptions surfaced

### Assumption 1: "Lives in L-cache" means L1 instruction cache

The brief said "L-Cache." I assumed L1i. But it could mean:
- L1i (192 KB on M4, ideal target for hot opcode bodies)
- L1d (128 KB, where the LUTs and constants live)
- L2 (16 MB, the shared cluster cache, easy target)
- The "system level cache" / SLC (variable size, not under our control)

The right answer is **all four matter**, but the binding constraint is L1i for opcode *code* and L1d for opcode *data* (LUTs, splat constants). The L2 and SLC are slack we use to absorb cold-start and contention.

### Assumption 2: The opcode set has a fixed size

I assumed we'd pick a fixed number of opcodes upfront. But the contract framing suggests it's open-ended: opcodes can be added as long as the *cumulative* size stays inside the budget. New opcodes can be promoted from "candidate kernel" to "set member" by passing the size + measurement check. So the set has a *budget*, not a *count*.

### Assumption 3: The opcode set is glyph-internal

I assumed it serves glyph's transformer/routing layers. But "extends the silicon" reads more universal — a substrate that any ternary computation can sit on top of, not just glyph's. If we make the opcode contract independent of glyph's higher-level types, it becomes a reusable foundation. That's a small extra cost in design discipline (no glyph-specific structs leak into opcode signatures) and a big gain in claim quality.

### Assumption 4: The opcode caller is C code

I assumed C. But "extends the silicon" hints at a more abstract caller — could be C, could be a future ternary VM bytecode, could be a generated kernel from a higher-level DSL. Designing the table-of-pointers API surface protects this abstraction: any caller that can compute an index and pass arguments through registers can drive the opcode set.

---

## What I now understand

The brief is asking for **a software-defined ternary instruction set with three properties**:

1. **Caller-indistinguishable from silicon** — predictable latency, no setup, no errors, no allocations, indexable.
2. **Cache-resident under realistic load** — bodies fit in L1i, data fits in L1d, with measured headroom.
3. **Buildable without inventing the kernel layer from scratch** — the existing glyph functions are 80% of the answer; we add a contract layer, an index table, a size budget, a measurement harness, and a small handful of new TBL- and VCNT-based ops.

The "opcode set" is not new code. It is **a contract layer over glyph's existing kernels**, plus a small set of new kernels that exploit TBL and masked-VCNT, plus a measurement harness that proves the contract is honored.

The single most surprising thing this exercise revealed: **`glyph_mtfp.c` already implements 60% of the ternary opcode set without anyone calling it that.** The work is mostly recognition, not creation.

---

## Remaining questions for the user

These can't be answered by reflection alone:

- **Q1.** Is the L-cache claim a marketing surface (caller-facing) or a perf budget (internal)? Answers shape whether we build the measurement harness before or after the design ships.
- **Q2.** Is the opcode set glyph-internal or generic? Answers shape whether glyph types leak into opcode signatures.
- **Q3.** Is the indirect-dispatch table for *external API* or *future VM*? Answers shape whether we design for stream-of-opcodes use cases now or defer.
- **Q4.** What's the size budget? L1i is 192 KB on M4 P-core but most of it is needed by the calling code. Is the opcode set's slice 8 KB? 32 KB? 64 KB? This determines how many opcodes we can ship before we have to start cutting.
- **Q5.** What new opcodes does glyph actually need? I identified TBL-based binary trit ops and masked-VCNT reducers as obvious gaps, but I don't know which ones the routing layer or layernorm path actually demands.

---

## What I'd commit to without further input

If I had to write the synthesis right now, with no further user input, I'd commit to:

- Two opcode tables (trit-vector, MTFP-vector), one contract.
- Out-of-line bodies, indexed via function-pointer tables, also callable directly by name.
- Size budget: 16 KB total (bodies + LUTs), targeting ~50% of M4 L1i to leave room for callers.
- The set seeded with the existing glyph kernels, promoted to opcodes by the new contract.
- A measurement harness as a follow-up (cycle counting, I-cache miss counting via Instruments) that lands before we publicly claim cache residency.
- A small batch of new TBL-based trit ops (sat-add, mul, max, min) and VCNT reducers (signed sum, sparsity count) added to fill obvious gaps.

I'd defer: bytecode VM, JIT, anything involving self-modifying code or `MAP_JIT`, and any opcode that requires more than vector-op granularity.
