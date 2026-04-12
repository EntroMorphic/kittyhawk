# Raw Thoughts: Lingering Thoughts After Pipeline Completion

## Stream of Consciousness

The pipeline is done. All 8 items landed, red-teamed, committed. 19 commits, 73 tests, 17.7 KB of opcode bodies, zero float. The substrate works. The wrapper works. The contract is written. So why am I not at rest?

Because the thing we built is an engine without a car. M4T is a beautiful set of primitives — I genuinely believe the SDOT-as-ternary-matmul insight and the TBL-as-universal-trit-op insight are real contributions — but nobody has driven them yet. There's no forward pass. No token has ever flowed through this system. Every test is synthetic: hand-constructed inputs, hand-derived golden values. The tests prove the math is right. They don't prove the architecture works.

That worries me. Not because I think the math is wrong — we've red-teamed it eight times. But because the gap between "correct primitives" and "working system" is where the hardest bugs live. Integration bugs. Composition bugs. The kind where each piece works in isolation but the way they connect produces drift, saturation, or silence.

Specifically, what's nagging:

1. **The softmax/GELU hole.** We have a LUT generator that works. We have not committed its output. Until we do, there's no nonlinearity, which means no attention (softmax) and no FFN activation (GELU). These are the two things that make a transformer a transformer. Everything else is linear algebra, which our primitives handle. But linear algebra alone is a fancy matrix calculator, not a language model.

2. **The weight loading problem.** There is no way to get weights into the system. The IO tool doesn't exist. Without weights, inference is impossible. Without inference, the contract clause about "predictable latency under realistic load" (M3) is untestable. We designed toward a contract we can't yet defend.

3. **The attention mechanism.** trix-z's architecture uses dense causal multi-head attention for the Q/K/V computation, with ternary routing only on the FFN. Glyph's PRD says "route everything — QKV, attention heads, W_O, FFN, LM head." That's six routing points per layer. We have primitives for one of them (the FFN routing path). The other five need routed projections, routed attention heads, and a routed LM head. Each is a composition of existing M4T primitives, but the composition hasn't been designed, let alone implemented.

4. **MTFP4 isn't connected to routing yet.** We built the SDOT cell and the routing primitives as separate items. The routing primitives operate on packed-trit signatures (popcount distance), but the SDOT matmul operates on int8 MTFP4 cells. The bridge — converting packed-trit signatures to MTFP4, or using SDOT for the dot-product routing score itself — hasn't been written. It's not hard, but it's not done.

5. **The LayerNorm p99 jitter.** The bench showed p99/mean = 1.87 at small N. That fails the contract. We noted it, deferred it, and moved on. But if glyph's transformer uses LayerNorm at d_model=64 (the trix-z default), every layer hits this jitter. It's not a theoretical concern — it's the actual performance the first user will see.

6. **The MTFP19 range might be too narrow.** We tightened MAX_VAL from INT32_MAX/2 to (3^19-1)/2. That reduced the real range from ±18183 to ±9842. No test broke. But no test exercises a real model's activation distribution either. If a pre-trained model has activations outside ±9842 in real units (which is plausible for un-normalized residual streams), the MTFP19 cell will saturate silently. The MTFP39 path exists as a fallback, but it's 2.5× slower.

7. **Training.** Decision D4 locked us to inference-only. That was the right call for v0. But glyph can't learn anything. It can only run weights that were trained elsewhere (in float, in a different framework) and converted. That's not a glass-box AI — it's a glass-box inference runtime for models trained in an opaque box. The real prize is training in MTFP, where the gradients and optimizer state are also ternary. Nobody has done this. It's a research problem, not an engineering one. And we haven't even started thinking about it.

8. **The "extends the silicon" promise.** We designed toward it. We wrote a contract. We built opcode tables. We measured cycle counts and size budgets. But the one thing we haven't done is have an external caller — someone who isn't us — try to use M4T as a ternary instruction set. The abstraction only works if someone else can pick it up and use it without reading the implementation. That hasn't been tested.

What am I actually scared of? Honestly: that the system works in pieces but fails in composition. That the first real model we try to run hits a saturation cliff, or a precision issue, or a performance wall, and the fix requires revisiting a fundamental decision (like the MTFP19 cell width, or the ternary-only routing, or the no-float policy). The decisions we made are bold. Bold decisions have bold failure modes.

What would the expert do? Ship what we have. Get a real model running, even if it's a 4-layer MNIST classifier. Find the first composition bug. Fix it. That's worth more than another red-team pass.

What would the beginner notice? "Wait, this thing can't actually DO anything yet?"

They'd be right.

## Questions Arising

- What is the minimum viable forward pass? One layer? One token?
- Can we borrow trix-z's MNIST path (test_ternary_mnist.c in reference-code/) and port it to M4T?
- What's the fastest path from "primitives work" to "end-to-end inference works"?
- Should we commit the LUT tables now, even though they're 5.4 MB?
- Is MTFP19 range sufficient for MNIST-scale models?
- What does training look like in MTFP? Has anyone published on ternary backprop without STE?
- When should we invite an external user?

## First Instincts

- Commit the LUT tables. Remove the blocker.
- Port trix-z's MNIST test as the first end-to-end validation.
- Skip attention for now — the MNIST path is FFN-only (projection + routed FFN + head).
- Use pre-trained trix-z weights, converted via the IO tool, as the first real model.
- If MNIST works, write a 1-layer GPT-2 forward pass next.
- Training is a separate research effort; don't try to solve it in the same sprint.
- Don't invite external users until at least one model runs correctly.
