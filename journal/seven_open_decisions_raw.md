---
date: 2026-04-14
phase: RAW
topic: §14 "Seven Open Decisions" in m4t/docs/M4T_SUBSTRATE.md
---

# Raw thoughts

I wrote §14 as a list of seven items that all felt like "things we haven't decided yet." When I step back and read them again, I'm suspicious. Something about the list feels too tidy for how genuinely open each question is. Let me write through the unease.

The "throughline" I just gave the user — *the substrate guarantees invariants; the consumer names preconditions* — is elegant. Too elegant. Elegant throughlines are often narrative devices that obscure heterogeneity. So my first doubt is that the throughline fit because I was pattern-matching across a list that isn't actually uniform.

Reading each one fresh:

**14.1 (logical block size).** This is a real tuning question. Hardware gives us 16B. Whether we want a *logical* block larger than that — 2× or 4× hardware blocks sharing one exponent — depends on prefetch behavior, workload variance, exponent-storage pressure. This feels genuinely empirical. I can't derive it; I need numbers from a running consumer. So it's open but in a specific way: it's open *pending data*, not open *pending philosophy*.

**14.2 (cross-block add).** Do we align-and-round or refuse? Ideological in how I framed it. But wait — does routing actually DO cross-block add with different block exponents? In the routing primitives we kept (`apply_signed`, `signature_update`, `distance_batch`), accumulation happens within a result tensor that has a uniform exponent structure *established at write time*. Cross-block-different-exponent add may be a problem the substrate doesn't actually face in routing workloads. If no consumer drives it, the decision is vestigial.

**14.3 (tail padding).** I flagged zero-pad as inconsistent with the throughline. But… a zero mantissa is the additive and multiplicative identity regardless of the block's exponent: `0 × 3^e = 0` for any e. So zero-padding doesn't inject values; it extends length with identity. The "silent substrate work" concern dissolves once you see this — there's nothing silent, the zeros are semantically null. Maybe I was wrong to file this as a throughline violation.

**14.4 (exponent sentinels).** Cosmetic. INT8_MIN as "widen-pending" vs a parallel byte array. Both express the same state. Parallel array keeps the exponent numerically clean. Weakly open.

**14.5 (SDOT saturation).** Max SDOT output on MTFP4 inputs is 16 × 40 × 40 = 25,600. That's ≪ int32's 2.1 billion. It's not open — it's *proven*. Why did I list it as open? Because I was filling a "concerns we considered" section rather than triaging.

**14.6 (LUT-backed nonlinearities).** This one bothers me. Is it actually a substrate decision? Or is it a consumer decision? The LUTs exist. Whether to expose them depends on whether any routing consumer needs a GELU or softmax. That's a *scope* question, not an *invariant* question. It doesn't belong next to 14.1–14.4.

**14.7 (benchmark bed).** Even more obviously not a substrate question. "What problem does the routing thesis have to beat?" is not M4T's question. M4T serves whatever consumer it's given.

Questions arising:
1. Is §14 conflating substrate-level opens with consumer-level and thesis-level concerns?
2. Are there any "opens" that are actually proven or decidable but I mis-filed them?
3. Does the throughline work as a *sorting* mechanism even when it doesn't work as a uniform *answer* mechanism?
4. What if the list is honest but the *framing* of the list was wrong — i.e., these aren't all the same kind of thing and shouldn't share a section header?

First instincts to watch:
- Instinct: "answer each one, list the answers." Watch for: that'd just produce seven isolated opinions without checking whether the questions are well-posed.
- Instinct: "force them all through the throughline." Watch for: already did that, already found 14.3 was an outlier in a suspicious way.
- Instinct: "refactor the section." Promising. Different kinds of question want different treatments.

Gut: the cycle should expose that most of these aren't really "seven open decisions" — some are proven, some are scope/thesis concerns that shouldn't be in the substrate spec, and a smaller core are genuine substrate opens.
