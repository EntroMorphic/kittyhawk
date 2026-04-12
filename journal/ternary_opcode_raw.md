# Raw Thoughts: Ternary Opcode Set in L-Cache

## Stream of Consciousness

OK so the user wants a "Ternary Opcode set that lives in L-Cache and extends the silicon." Let me unpack what I think this even means before I run off and do something dumb.

"Ternary Opcode" — a set of operations that take ternary inputs and produce ternary outputs. These don't exist in M4's silicon. We've already established that. M4 is binary at every layer of its ISA. There's no instruction that says "multiply two trits" or "add two ternary cells with ternary carry." Everything we do is emulated on top of NEON.

"Lives in L-Cache" — this is the interesting bit. The idea seems to be: even if the hardware doesn't have these instructions natively, we can define small code sequences that implement them, and if those sequences are *resident* in instruction cache, they execute at near-native speed because there's no fetch penalty, no decode penalty beyond what the CPU already pays for any instruction. This is basically the trick interpreters use with computed goto + threaded code, or what JIT compilers do, or what BPF does, or what microcode does inside the CPU itself. We're proposing to create a "soft microcode" layer that defines a ternary ISA on top of the binary one.

"Extends the silicon" — this is the framing I find compelling. We're not just writing helper functions. We're building a logical instruction set that *augments* what M4 can do. The fact that the implementation is software is an implementation detail; the *abstraction* is that the chip now has ternary opcodes.

OK, what scares me about this:

1. I don't actually know how big M4's L1i is. I think 192 KB. If our opcode set is bigger than that, we lose. So the size budget matters a lot.

2. I don't know how the CPU's branch predictor and instruction prefetcher behave when you do indirect dispatch through a function pointer table. If every "ternary opcode call" is a branch mispredict, we eat a 10-cycle bubble per opcode and the whole thing is dead on arrival.

3. There's a question of granularity. If a "ternary opcode" is "one trit operation" then dispatch overhead dwarfs the actual work. If it's "one routed FFN layer" then it's not really an opcode, it's a function. The right granularity is somewhere in between and I don't know where.

4. Computed goto / threaded code is a known fast dispatch pattern but it requires GCC/Clang support (`&&label`). I'd want to verify Clang on M4 generates good code.

5. Apple Silicon's L2 is 16MB shared per cluster. Big. L1i is small. The opcode set definitely fits in L2; the question is whether it stays hot in L1i. That depends on access pattern, locking patterns, and what else the program is doing.

6. Self-modifying code is a non-starter on Apple Silicon (W^X enforcement, even for JIT requires `MAP_JIT` and per-thread mode switching). But we don't need self-modifying code for this — the opcode bodies are static, we just dispatch to them.

7. There's a tension between "the opcode set is universal and you call into it" and "the opcode set is inlined at the call site." Inlining is faster (no dispatch overhead) but bigger (every call site duplicates the body, which fragments I-cache). Out-of-line dispatch is smaller (one copy of each opcode) but pays dispatch overhead. Modern CPUs handle short call/return very well, so out-of-line might actually win even on perf if the dispatch is direct.

8. What's an "opcode" vs. an "intrinsic" vs. a "function"? Honestly I think the difference is mostly about size, calling convention, and how often you call them. An "opcode" is a tiny fast-path operation that you call millions of times. A "function" is a big complex thing you call occasionally. Where's the line?

Half-formed idea: What if we define the ternary opcode set as a *flat array* of code blocks, all aligned to cache lines, all the same size, all callable via `lea + indirect call` with the opcode index? That gets us a uniform calling convention, predictable cache behavior, and lets the CPU prefetch by index. Like a jump table but where the targets are actual code, not just labels.

Half-formed idea #2: TBL is the universal binary ternary op. Earlier I worked out that any 2-operand ternary function fits in a 16-byte LUT. So one approach to "ternary opcodes" is: each opcode IS a 16-byte LUT, plus a tiny stub that loads the LUT and calls TBL. The stub is ~5 NEON instructions. If we have 64 ternary opcodes, that's 64 × (~80 bytes) = 5KB of opcode bodies + 64 × 16 = 1KB of LUTs. Trivially fits in L1i. But this approach only handles binary ternary ops on packed-trit vectors. For MTFP arithmetic we'd need different opcode shapes.

Half-formed idea #3: There might be two opcode sets, not one. Set A: "trit ops" that work on packed-trit vectors (uint8_t* containers), use TBL, are tiny and uniform. Set B: "MTFP ops" that work on int32 cells, use full NEON SIMD, are bigger and have more variation. Ducktape them under one dispatch table or treat them as separate.

Doubts and questions:

- Q: Is this actually faster than just calling well-named C functions that the compiler inlines? The compiler is already doing this for `glyph_mtfp_mul` and similar inline functions. What does an "opcode set" buy that inlining doesn't?
- A: Maybe nothing, IF the compiler can see the call site and the opcode body. The opcode set framing matters when the call is *dynamic* — the opcode index is not known at compile time. Like an interpreter for a ternary VM.
- Q: Wait, are we building a ternary VM? Is that what "extends the silicon" means? An abstract ternary machine that the user can program against?
- A: Maybe. That would be a coherent answer. Glyph defines a ternary VM. The VM's instructions are the "opcodes." The implementation is small NEON sequences. The instructions live in L-cache because we keep them hot. Programs in this VM are sequences of opcodes that the dispatcher walks through. This is exactly what threaded-code interpreters like Forth do.
- Q: But do we want a VM? Or do we want C-callable functions that happen to be very fast?
- A: Don't know. Ask the user. But I lean toward C-callable for now and "VM" as a future stretch.
- Q: How does the user invoke the opcode set?
- A: Three options: (a) C function calls — `glyph_op_trit_mul(dst, a, b, n);`, (b) opcode index dispatch — `glyph_dispatch(OP_TRIT_MUL, dst, a, b, n);`, (c) bytecode interpreter — feed a stream of opcode bytes and let the interpreter walk them.
- Q: Which of these "lives in L-Cache" most naturally?
- A: All of them, if they're hot. The difference is granularity. Bytecode interpreter has the smallest per-call overhead in code (since the dispatcher loop is tiny) but the highest per-op overhead (decode + dispatch every op). C function calls have no decode but have call/return overhead. Index dispatch is in between.
- Q: What does "extends the silicon" really mean? Is it just rhetoric, or is there a concrete test?
- A: Concrete test: from the caller's perspective, can they treat ternary operations as if they were CPU instructions? That means: predictable latency, predictable throughput, no surprising allocations, no error returns, no exceptions. A "silicon extension" is a contract: "this operation is always available, always fast, always cheap." If we can deliver that, it's a silicon extension. If the user has to worry about cache state or warm-up, it's not.
- Q: How do we keep the opcode set warm in L-cache?
- A: Three techniques. (1) Touch it at startup: a "warm-up" function that calls every opcode once. (2) Pin pages: `mlock()` keeps memory in RAM but doesn't affect cache; cache is managed by the CPU and you can't pin it directly on aarch64 user-mode. (3) Periodic refresh: in a long-running program where the opcode set might fall out of cache, occasionally call cheap opcodes to refresh. Mostly the CPU's eviction policy will keep hot code hot.

This is getting long. Let me stop and identify what I actually know and what I'm guessing.

What I know:
- M4 NEON has TBL and VCNT, both 1-cycle, 16-byte-wide.
- TBL is a universal 2-operand binary ternary op via 16-byte LUTs.
- L1i is small (~192KB on M4 P-core), L2 is large (~16MB shared).
- aarch64 supports indirect branch via `BR Xn`, which can be predicted by the indirect branch predictor.
- C99/GCC extension `&&label` + `goto *ptr` enables computed-goto threaded code; Clang supports it.
- Apple's W^X means we can't easily JIT, but we don't need to — opcode bodies are static.

What I'm guessing:
- Branch prediction quality for indirect dispatch on M4 (probably good but I don't have numbers).
- L1i pressure under realistic workloads.
- Whether inline beats out-of-line for these op sizes.
- Whether the user wants a "VM" or a "fast C library."

Open questions for the user:
1. Is this a VM (bytecode interpreter) or a C-callable library?
2. What's the granularity of an "opcode" — single operation, vector op, layer?
3. Is "lives in L-cache" a real perf constraint or a design metaphor?
4. Should the opcode set be discoverable / extensible at runtime, or fully static?
5. Is this for glyph specifically, or a generic ternary substrate?

I'll proceed assuming: C-callable library, vector-op granularity (one opcode = one operation on a packed-trit or MTFP vector of any length), L-cache residency is a real perf goal we'll measure, fully static set, scoped to glyph.

## Questions Arising

- Granularity: per-trit op? per-vector op? per-layer?
- Dispatch: direct call, indirect call, computed-goto, bytecode VM?
- Inline vs out-of-line — which wins on perf, which wins on I-cache?
- How big is the opcode set actually going to be?
- How do we measure I-cache residency to verify we're meeting the goal?
- Should we have one opcode set, or two (trit-vector vs MTFP-vector)?
- What's the relationship to existing glyph functions? Are these opcodes a *replacement* or a *layer*?

## First Instincts

- Make it C-callable, not a VM. Save the VM for later.
- Vector-op granularity: one opcode = one operation on a contiguous buffer.
- Use TBL aggressively for trit-vector ops; the per-opcode body is tiny (~5 instructions + 16-byte LUT).
- Keep MTFP ops separate from trit ops — different shapes, different calling conventions.
- Measure L1i residency with `perf` (Linux) or Instruments (macOS). Don't claim residency without evidence.
- Aim for the entire opcode set (bodies + LUTs) to fit in 32KB so it lives comfortably in L1i with room for the calling code.
- Don't build the "extends the silicon" abstraction until the C-level kernels are profiled and proven fast.
