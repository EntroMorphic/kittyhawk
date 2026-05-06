---
cycle: bitnet_phase1
phase: RAW
date: 2026-05-06
scope: end-to-end BitNet b1.58-2B-4T inference on the m4t substrate. Phase 1 of the four-phase ternary-ML arc (inference → fine-tune → train-from-scratch → productize). Validates that the substrate's kernel surface composes into a real ternary LLM running in its native numeric system. Output: the substrate ran a recognized ternary model end-to-end, with quality measured against HF's reference implementation.
---

# Raw — bitnet_phase1

## Stream of consciousness

I sold this in the prior message with more confidence than I've earned. "9 weeks total." Where did that estimate come from? I made it up. I have no actual basis for it. I have a sequence of phases (D → B → A) and put plausible-feeling numbers next to them. The truth is I don't know how long this takes because I haven't done end-to-end LLM inference before — at least not at the substrate level. I've sketched the kernel surface. I haven't loaded a real model from HuggingFace. I haven't run a tokenizer through libm4t-shaped buffers. I haven't generated a single token of output. The estimate is theater.

The "phase D → B → A" framing is also convenient. It says: first close substrate gaps, then test composition, then go end-to-end. That's clean. It's also a little too clean — it presupposes that the substrate gaps (TD-14 LUTs, RMSNorm, RoPE) are independent puzzles each with bounded scope. What if RMSNorm specifically wants something the substrate doesn't have, like a square root primitive that itself needs a research cycle? Then phase D balloons. What if RoPE's rotation pairs require a primitive shape (complex multiply) that doesn't exist in the substrate? Then phase D balloons differently. I framed phase D as "2-3 weeks" of essentially mechanical work — but I haven't actually verified the underlying assumptions.

I cited TD-14's "archived prior cycle LUT generator" as a quick restoration. I don't know what that generator handled. It might have generated GELU-shaped LUTs for an old expression-routing experiment that has nothing to do with BitNet's silu activation. The shape, the input range, the table size — none of that is verified to match what BitNet wants. "Restore from archive" makes it sound like a packaging task. It could be a redesign.

Numerical fidelity. I waved at this and called it "a real risk." That was honest as far as it went, but it's also the most likely place where Phase 1 fails. We'll be running BitNet at MTFP19 mantissa precision while HF runs it at bf16. Different rounding semantics, different range. Across 30 transformer blocks, errors compound. The baseline question is "do we match HF on a single forward pass?" — and we may not. Then what? Do we say "close enough, ship"? Do we redesign the substrate to round differently? Do we discover that bf16's rounding is structurally different from any base-3 rounding rule and accept that base-3 inference always diverges from base-2 inference even when both are mathematically correct? That last possibility is the most interesting one and also the most dangerous — it would make "match HF" a fundamentally incoherent goal.

I haven't actually read the BitNet paper or model architecture file. I'm using "RMSNorm and RoPE" as placeholders for "the architecture has some normalization and some positional encoding." It probably has those — most modern transformers do — but I haven't verified that BitNet b1.58-2B-4T specifically uses them, or whether they have any quirks. I've never inspected the model's `config.json`. I should have. The fact that I'm planning against an architecture I haven't read should make me suspicious of my own confidence.

The "BitNet is the right consumer" framing has a recognition-borrowing quality I'm uncomfortable with. Microsoft validated ternary at 2B parameters; we're going to ride that validation by running their model. The strategic story is genuine — we DO want a recognized arbiter for vision claim 3, and BitNet is genuinely well-shaped for that — but it also means our success becomes "we ran Microsoft's model on our thing." That's load-bearing for credibility but it's not actually our contribution. Our contribution is what comes after Phase 1 (fine-tuning, training from scratch, productization). Phase 1 itself produces no novel ML — it's a port. The risk is investing a lot in the port and discovering the post-port phases are where the real work is.

I'm also uncomfortable with how cleanly I separated "matmul covered" from "what's missing." Real LLM inference has a lot of glue: the embedding layer, the residual connections, the LM head. The KV cache. Token-by-token generation. Each of those is "covered by existing primitives" only in the sense that nothing exotic is needed. The integration is real engineering. I haven't sized it.

Calendar-time vs work-unit. The user has been generous with pace — "take your time, methodically, accuracy is paramount." I've enjoyed that. But Phase 1 spans real time, and I don't actually have a model for predicting calendar time. Each cycle I've done has fit inside a session or two. End-to-end inference is much bigger. There may be a phase where I'm stuck on something for days. There may be a phase where the work is mechanical and goes fast. I can't predict the ratio.

The user said the END goal is base-3-native ML training. Phase 1 is a gate. If Phase 1 fails — for whatever reason: numerical fidelity, kernel shape mismatch, some RMSNorm-shaped surprise — we don't get to the goal. So Phase 1's risk tolerance is different than "demonstrating an inference path." It's "the gate to the actual research project." That makes me want to run Phase 1 with more rigor than I'd otherwise apply, AND it makes me want to budget for failure more honestly than I did.

Also — Phase 4 (productization) was in my prior message but it might not be load-bearing for the user's stated goal. The user said "productize for the open-source population" but that's an outcome of having "what it takes to do end-to-end ML in base-3," which is the Phase 3 deliverable. Phase 4 might be optional. Or Phase 3-with-an-extra-mile. I conflated.

I'm noticing momentum bias again. The substrate work has been on a roll. The benches landed cleanly. The doc audit closed gaps. The defensive-asserts patch was small and easy. I want to keep moving. That energy can carry me past important questions if I'm not careful. The LMM is a check against that. Good that the user asked for it.

The shape-mismatch risk I flagged in the prior message — "§20 was built against bench shapes, BitNet may not fit" — I dismissed it a little too easily. "Phase D would surface this." But Phase D is substrate-side work. The shape mismatch surfaces when you actually call the kernels with BitNet's shapes and access patterns. That's Phase B at the earliest. By then we've already invested in Phase D under the assumption that Phase D's work is reusable. What if it isn't?

Skill gap. I haven't done LLM inference engineering before. KV cache, generation loops, sampling strategies (greedy vs top-k vs nucleus), tokenizer integration — I'd be learning while building. That's not bad in itself but it's a hidden complexity factor in the timeline. Things will surprise me.

What bothers me most: I keep using the word "validates" to describe Phase 1. "Phase 1 validates the kernel surface." But validation against what? If we run BitNet and produce text that's somewhat similar to HF's output, what does that prove about the substrate? It proves the kernels compose without crashing. It proves nothing about whether we have "what it takes to do end-to-end ML in base-3" in any deep sense. The deep test is Phase 2 (gradient flow) and Phase 3 (training stability). Phase 1's "validation" is an engineering check, not a research validation. I should be honest about that.

## Questions arising

1. What is BitNet b1.58-2B-4T's actual architecture? (Read config.json + paper before planning concretely.)
2. Does BitNet's quantization protocol map cleanly onto the substrate's existing absmean ternarize, or does it have quirks?
3. What's the substrate's MTFP19 mantissa range vs bf16's range, and where do they diverge in BitNet's activation distribution?
4. Is the "archived prior cycle LUT generator" actually applicable to BitNet's silu/softmax shapes, or do we need a redesign?
5. RMSNorm needs `rsqrt(mean(x²) + ε)`. The substrate has no square-root primitive. What does that mean for the work?
6. RoPE's rotation pairs — does the substrate have a complex-multiply-shaped primitive, or do we need to design one?
7. What's the right end-to-end success criterion? "Bit-exact match HF" is unrealistic. "Acceptable quality on one benchmark" is fuzzy. What's the precise gate?
8. Does Phase 1 commit to a fork in the architecture's identity (transformer-shape) that closes off other directions?
9. What's the relationship between Phase 1's output and the user's stated goal? (Validation step vs. demonstration vs. proof?)
10. How do we budget for the failure modes I haven't thought of yet?
