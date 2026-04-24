---
date: 2026-04-24
scope: LMM cycle — base-3-native benchmark selection
phase: NODES
---

# NODES: base3_benchmarks

## Discrete ideas

1. **Benchmark fit = substrate property alignment.** "Is it base-3 native?" decomposes cleanly into: (a) does the input admit ternary representation without quantization loss, (b) does the task structure reward routing specialization, (c) does evaluation credit inspectability or only top-1 accuracy? A benchmark that hits all three is a "Glyph-native" benchmark. Image classification hits zero of three.

2. **MNIST/Fashion/CIFAR were the wrong proving ground.** We chose them because they're universal, not because they reward our substrate. Every pp on CIFAR has been paid against a representation tax (continuous RGB → ternary) that SSTT never incurs. We have been benchmarking the wrong thing.

3. **The substrate was never really tested.** MNIST/Fashion near-saturation at the *signature* layer means *routing never became load-bearing in those experiments*. We've never had a benchmark where choice-of-tile actually had to specialize. `routed_autodiff` 10-class toy was the first artificial construction of that scenario, and it immediately revealed expert collapse. That's a training problem. But there's also a benchmarks problem: even with perfect routing, MNIST doesn't need routing.

4. **Ternary-native inputs eliminate the first tax.** Data already in {−1, 0, +1}: Go/chess position state (empty/own/opponent), finance signals (down/unchanged/up), version-control edit tags, mutation categories, survey Likert-3, directional sentiment. No quantization loss, no "base-2 ignores 1/3 of signal" tax on input.

5. **Routing-load-bearing tasks have qualitatively different computation per input.** Board games have phase structure (opening/middlegame/endgame); extreme classification has label clusters; multi-domain text has domain identity. Every expert-model architecture targets these kinds of tasks for a reason: dense nets waste FLOPs, routed nets specialize. Glyph's substrate is the *cheapest* way to do that specialization — if the task rewards it.

6. **Inspectability-rewarding evaluations are rare but real.** Regulated domains (medical, legal, finance, T&S) pay for "which tile fired, why" as a direct product feature. Accuracy-only benchmarks like CIFAR actively discount this. A benchmark that credits signature-as-explanation would let our substrate claim an axis SSTT cannot.

7. **Shortest-path benchmarks using current capabilities:**
    - **Tabular classification**: needs only a CSV loader + categorical→trit encoder. Mixed discrete/continuous is already our comfort zone (pyramid quantization generalizes). Routing helps long-tail. Inspectability matters in credit/medical/insurance verticals. **Effort: low. Claim surface: medium.**
    - **Board-game position evaluation / policy**: state is natively ternary; Go 19×19 is 361 trits per board with zero loss. AlphaZero-style value+policy head. **Effort: medium (needs policy/value head + self-play or fixed dataset). Claim surface: high (routing by phase is structural).**
    - **Sentiment-with-neutral / directional finance**: ternary labels, text input. **Blocked by lack of text embedding path.**
    - **Extreme classification**: routing-by-necessity, but NLP embedding blocked.
    - **Compositional / SCAN**: cleanest substrate claim, but seq2seq blocked.

8. **Longer-path candidates require substrate extension.** Each is its own cycle:
    - Text embedding (for sentiment, extreme classification, BIG-Bench, SCAN).
    - Sequence head (for SCAN/COGS).
    - Policy/value head + self-play (for board games — but a fixed dataset variant is shorter).
    - Structured output (trees/graphs).

9. **"Build one" is not a cop-out.** Custom benchmark with controlled properties: synthesized trinary features, modular class structure, tunable routing load. Proof-of-concept for "routing-first claim is real." But: defensible only if later mapped to a real-world task — otherwise it's a stunt.

10. **The XGBoost comparison matters.** If we go tabular, the incumbent is gradient-boosted trees — themselves a form of routing (decision-tree splits ARE learned routing). Beating XGBoost on tabular is a credible "routing architecture wins" claim. Losing to it would be a sharp negative that tells us something real.

11. **"Beat SSTT on CIFAR" was never the right north star for this substrate.** It was a proxy for "we're a real contender." But SSTT is designed for CIFAR's representation, and we aren't. The actual north star is NORTH_STAR: "routing is essential in base-3." That's a substrate claim, not a CIFAR claim. Different benchmarks validate different claims.

12. **Inspectability without a ready product is abstract.** We can SAY our signatures are explainable, but until an actual regulated-domain user chooses us BECAUSE of that, it's a marketing property. Currently theoretical. Worth biasing toward tasks where inspectability has a measurable eval metric (e.g., decision rationale recall against human-labeled rationales), not just a narrative.

13. **Load-bearing routing needs multiple classes AND specialization pressure.** 10-class with dense signal on every class (the 10-class toy) has no incentive for tiles to specialize — every tile sees every class. Long-tailed distributions or hierarchical label structures create specialization pressure naturally. A benchmark with uniform class frequency and full label coverage per tile is a benchmark where routing *cannot* help, regardless of trainer design.

14. **Current substrate handles fixed-dim feature vectors exceptionally well.** This is a hint: a benchmark where the input is already a fixed-dim vector (tabular, pre-extracted features, board positions) is a benchmark where we don't pay any substrate-mismatch cost. Raw text, raw audio, raw video all require an embedding we don't have. Raw images we've tried three times and each time paid a tax.

15. **Effort-weighted value:**
    - Tabular: low effort, medium claim. Probably two-week scope. Real-world relevance high.
    - Board games (fixed-dataset): medium effort, high claim. Probably four-week scope. Narrative strong ("ternary-native state").
    - Custom synth benchmark: low-medium effort, conditional claim (only if later mapped to real). One-week scope.
    - Everything else: blocked or long-term.

16. **"Can we run it in a day?" is a valid phase-gate.** Before committing to any benchmark, the cheapest sanity check is: load the data, quantize it with existing tools, run direct_lsh baseline. If the baseline is already interesting, the benchmark is reachable. If direct_lsh faceplants, the benchmark needs substrate work first.

17. **Multi-benchmark strategy may be better than single choice.** A slate of three:
    - One cheap (tabular — quick iteration, broad appeal).
    - One claim-rich (board games — substrate-native story).
    - One diagnostic (custom synth — measures whether routing is even doing what we think).
    Let data pick which advances; don't over-commit upfront.

18. **The routing-load-bearing criterion is the sharpest filter.** Anything that passes "routing specialization is structurally required" is worth evaluating. Anything that fails it is a substrate waste, regardless of how clean the ternary encoding is.

19. **Inspectability is a bonus, not a primary driver** — for now. Until we have a product context where inspectability is monetized, optimizing for it risks building a demo nobody evaluates by that axis. Keep it in mind; don't lead with it.

20. **Ternary-native input is necessary but not sufficient.** Go state is natively ternary — great. But a Go net that just memorizes positions via dense lookup doesn't vindicate routing. The task has to *also* reward specialization. Board-game phase structure does; a pure memorization task wouldn't.

21. **"SSTT representation tax" framing generalizes.** Any benchmark where continuous precision in the input is load-bearing will penalize us (ImageNet, audio spectrograms, high-dim embeddings). Any benchmark where inputs are fundamentally discrete or categorical will not (tabular, games, edit sequences, DNA). This is a rule: **if the input loses real information going to ternary, it's a bad fit.**

22. **The `/lsh_filter_ranker` architecture is benchmark-agnostic but data-sensitive.** Filter-ranker + multi-table composition is a shape, not a benchmark. It'll work on any data where distance signatures discriminate. The question is which data that's true for at scale.

23. **Long-tail is underweighted in academic benchmarks.** Real-world data is long-tailed; most benchmarks deliberately balance classes. A long-tail benchmark (iNaturalist, Amazon reviews, Wikipedia categories) would reward substrate properties our current toys don't test.

24. **Concrete first probe (anti-commitment):** pick one tabular dataset and one ternary-state board-game dataset, run direct_lsh on both, see where the baseline lands. A half-day of work tells us whether either is immediately reachable. If either hits "surprising baseline with no trainer" territory, the benchmark chose itself.

25. **NORTH_STAR discipline: do not invent benchmarks to hide weak claims.** If we build a custom synthetic benchmark, the contract is that it must be immediately and honestly mapped to a real-world task after the proof-of-concept. "We win our own benchmark" is worthless otherwise.
