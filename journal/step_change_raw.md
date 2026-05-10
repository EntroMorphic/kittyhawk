# Raw thoughts: next step-change for Glyph

Stream of consciousness. The point is to be unfiltered, not to produce conclusions yet.

## Stream

I gave a recommendation an hour ago: gradient kernels / training. Let me actually look at why that came out so fast and whether it's because it's right or because it's the easy answer.

The four-phase plan in `gesh/bitnet/README.md` — inference → fine-tune → train-from-scratch → productize — exists. So "training" is the named next phase. That's prima facie reason to recommend it. But four-phase plans are often what people write before they know what's actually hard. The plan was written when the substrate hadn't run a real LLM yet; now it has. The state of evidence is different.

Why does training feel obvious? Because it's:
- Concrete (we know what kernels are needed)
- Bounded (substantial work but specifiable)
- A clean capability extension (substrate becomes a producer of weights, not just a consumer)
- Aligned with the four-phase plan

But concrete + bounded + extension + aligned — none of those individually mean "step-change." A step-change should change what's POSSIBLE, not just what's BUILT.

What would actually be a step-change?

Different categories I'm circling:
1. **Capability change** — substrate gains something it couldn't do before. Training is one of these.
2. **Claim change** — substrate makes a thesis-relevant claim it couldn't before. Strong-claim L2/L4/L6 cycles are these. Finding a Part-B benchmark would be the biggest of these.
3. **Generality change** — substrate proves it's not specific to one model. Porting Mistral or Llama is this.
4. **Approach change** — substrate gains a new way of being USED. Routing-native consumer architecture (the dream of NORTH_STAR.md) is this. Hard to scope.
5. **Audience change** — substrate becomes accessible to others. Documentation overhaul, Python bindings, packaging. Non-technical step-change.
6. **Workflow change** — substrate enables faster iteration on substrate-claim work itself. Better measurement tooling, easier experimentation. Meta-step-change.

I keep gravitating to (1) Training because it's tangible and the path is clear. But "tangible + clear" is correlated with "low-variance, modest-reward." Step-changes are by definition higher-variance.

What scares me about each option:

- **Training**: months of foundational kernel work. The failure mode is "we get gradient kernels working but the resulting fine-tune quality is bad" — and at that point we'd have learned a lot about backward-pass MTFP arithmetic but landed no new claim. The risk profile is "long timeline to demonstrable progress."

- **Strong-claim L2/L4/L6**: I dismissed these as "incremental" but maybe I underweighted them. The L1 strong claim is one data point. If L2 (activations) ALSO shows base-3 structural advantage at sub-2-bit, that's two data points and the "advantage at L1 is generic" story strengthens. If L2 DOESN'T show it, that's important negative evidence — "the L1 win was load-specific." That's not incremental, that's information about the thesis.

- **Part-B benchmark**: most strategically valuable but I don't even know what it would look like. The honest answer might be "I don't know how to scope it, so I can't recommend it confidently." But that uncertainty is itself information — maybe the project's biggest gap right now is "we don't know what would test Part B."

- **Second model**: feels like it would be a 4× return on the BitNet investment. Once you've ported one LLM, the second is much easier (kernel surface is shared, harness pattern is reusable). And it directly addresses the "is this BitNet-specific or substrate-general?" question.

- **Workflow / measurement infrastructure**: easy to dismiss but maybe the biggest leverage. If we could run inference batteries 10× faster, sweep hyperparameters 10× more configurations, do cross-checks against HF in minutes instead of an hour — every other step-change gets accelerated. But this isn't a "step-change" in the conventional sense.

What if the real answer is sequencing? Not "pick one" but "which one FIRST, given the others depend on it"?

- Training enables Part-B (need to train routing-native models)
- Training enables productizing
- L2 strong-claim doesn't depend on training
- Second model doesn't depend on training (could port another inference-only)
- Workflow infrastructure doesn't depend on anything; enables everything

If I were honest about leverage: workflow / measurement infrastructure has the highest leverage but the lowest "step-change" appearance. Strong-claim L2 has moderate leverage and is a real claim move. Training has high leverage AND is high-risk.

Hidden assumption I'm making: "the next step-change" implies one big thing, not several small things in parallel or sequence. But the project's mode is mostly sequential single-track. So this assumption is probably right for the practical question.

Another hidden assumption: that I should recommend the option that maximizes "step-changeyness." Maybe what the user actually wants is the option that maximizes "expected value of progress on the thesis." Those are different. Step-changeyness is about size of impact; expected value weighs probability of success.

If I weigh by expected value:
- Training: medium-high impact × medium probability = moderate-high EV
- L2 strong-claim: medium impact × high probability = moderate EV
- Part-B benchmark: very high impact × low probability = ??? hard to estimate
- Second model: medium impact × high probability = moderate EV
- Workflow infrastructure: low-medium impact × very high probability = moderate EV

Training still comes out on top by EV, but the gap to L2 isn't huge.

What's REALLY scary about training: it pushes us into Part B territory without having established Part A robustly. The L1 strong claim is one layer. If we train without strengthening the L2/L6 claims, and then try to make Part-B claims with our training infrastructure, we'll be claiming "routing wins on this trained model" without having established that base-3 routing wins at multiple substrate layers. That's claim drift — Part-B without Part-A backing.

The discipline argument suggests: strong-claim L2 first (cheap, supports thesis), then training (expensive, opens new territory). Sequencing matters.

Counter: the substrate-claim cycles are kind of... done? L1 has the strong evidence. L2 would be valuable confirmation but the project's been mostly building since the strong-claim work landed in early May. Maybe the project has implicitly decided "we have enough Part-A evidence; let's build Part-B-relevant capability." Training fits that.

Hmm. I'm circling. Let me write down the open questions:

## Questions arising

1. Is the user implicitly committed to the four-phase plan (inference → fine-tune → train → productize)? If so, training is locked in by sequencing and the question is just "training next, but how scoped."
2. Is there a thesis-relevant inference-only experiment that could test Part B without needing training? If yes, it changes the answer.
3. What's the actual time-to-first-result for each option? My estimates are vague.
4. Does the project have a notion of "claim density" — how much we've validated per unit of substrate work? The L1 strong claim took a lot of cycles. Have we earned the right to move on, or should we deepen?
5. Am I biased toward training because it's the option whose mechanics I most clearly understand?

## First instincts (revised after writing)

My initial recommendation was training. After thinking, I'm less sure. The sequencing argument for L2 strong-claim is real. The Part-B-without-training option (a routing-native inference experiment) is undertheorized but interesting. And "second model" is a low-risk way to test BitNet-specificity.

The honest first-instinct revision: "training" might be the wrong frame entirely. Maybe the right frame is "what's the most informative next experiment" — which could be small. Or "what most strengthens the thesis" — which favors L2. Or "what most expands what's buildable" — which favors training.

I want to NODES this and see what falls out.
