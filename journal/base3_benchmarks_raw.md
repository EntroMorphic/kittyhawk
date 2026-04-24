---
date: 2026-04-24
scope: LMM cycle — identify a benchmark where base-3 routing geometry is intrinsically advantaged, rather than re-paying the SSTT representation tax on binary-legacy datasets.
phase: RAW
---

# RAW: what is a base-3-native benchmark?

`routed_autodiff` closed with a structural finding: frozen-U + selection-only routing collapses on multi-class. Before building the next layer of trainer (soft routing, load balance, learned U), we should ask a more upstream question: **which benchmarks are we even trying to win, and are they the right proving ground for a base-3 routing substrate?**

CIFAR-10 has been our CIFAR because MNIST was easy and everyone else uses it. But MNIST / Fashion / CIFAR are all designed *by and for* binary-native continuous float networks. They were canonized on architectures that assume dense float matmul and have no notion of routing. Every pp we claw back on CIFAR is a pp paid to a representation tax we did not choose.

The question: is there a benchmark where **base-3 structure is an advantage rather than a penalty**, where **routing is load-bearing by construction**, and where **winning is a substrate claim rather than an accuracy race against SSTT**?

## What "base-3 native" even means — first-principles drafts

### Candidate framing 1: ternary data already
Benchmarks where the input is naturally trinary:
- **Sentiment polarity with "neutral"**: {negative, neutral, positive} → trinary label AND the content-signal (up/unchanged/down) is naturally ternary in finance, reviews, survey responses.
- **DNA/RNA mutation state**: base-pair changes are discrete; nonsense/missense/silent.
- **Edit tags** in version control: added / unchanged / deleted.
- **Neuro spike data** with refractory: +1 spike, 0 quiescent, -1 hyperpolarized — rare but real.
- **Triplet ranking** (A > B, A = B, A < B): learning-to-rank problems.
- **Go / Chess board state**: empty/black/white or empty/own/opponent is exactly ternary.

### Candidate framing 2: routing-load-bearing tasks
Tasks where different inputs need qualitatively different computation, not just a different point in a continuous weight space:
- **Mixture-of-experts natural settings**: multi-domain text (news / code / legal / poetry), multi-modal (text+image+audio), multi-task benchmarks where the task identity is input-dependent.
- **Sparse / modular skill tasks**: BIG-Bench tasks where each item tests a distinct skill.
- **Conditional computation**: procedural games, dynamic graph neural net problems.
- **OoD generalization** under task shifts — routing can specialize.

### Candidate framing 3: inspectability-leveraged tasks
Places where our substrate's audit property matters more than raw accuracy:
- **Medical triage** where decisions must be traceable.
- **Trust & safety classification** where the signature (which tiles fired, why) *is* the explanation.
- **Regulatory compliance** where "show your work" is mandated.
- **Scientific data where the features must correspond to interpretable constructs**.

### Candidate framing 4: tasks dense networks handle badly
- **Very long-tailed class distributions** — dense FLOPs waste on common classes, starve rare classes. Routing concentrates tiles where needed.
- **Compositional generalization** — SCAN, COGS, CFQ. Dense nets memorize; routed nets might actually compose.
- **Extreme-class problems** — 100k+ classes (extreme classification, named entity linking). Dense output layers become the whole cost.
- **Energy- or memory-constrained deployments**: mobile, embedded, microcontroller. Our substrate's entire reason-for-being.

### Candidate framing 5: benchmarks that don't exist yet
The most honest answer might be: there isn't one, and we need to build it. A benchmark where:
- Input representation is natively ternary-signed (no quantization loss).
- Class structure is genuinely modular (expert specialization pays off).
- Evaluation rewards routing coherence, not just top-1 accuracy.
- Binary/float networks have no inherent advantage — in fact, a measurable disadvantage.

## Known benchmarks, re-examined through base-3-native lens

### MNIST — NOT base-3 native
Grayscale continuous 0..255. Classes are visually-similar digits with overlapping signatures. Everything SSTT does a dense float network does as well as or better than a ternary one. We hit 97.30% and "win" by not underperforming — a wash. Not a claim-producing benchmark.

### Fashion-MNIST — marginally interesting
Our +2.12pp over SSTT suggests the per-garment background/foreground structure happens to favor our signature approach. Still fundamentally a "we beat them at their own game" benchmark — doesn't prove routing is load-bearing.

### CIFAR-10 — anti-base-3 native
Continuous natural images in RGB. The representation tax is maximal here: quantizing to ternary at input time loses color continuity, texture gradients, and scale cues that SSTT exploits in every convolutional filter. −4.95pp. The 10-class toy convergence cap (47pp below plain-linear on routed) confirms: routing is NOT the bottleneck in our current CIFAR results — input representation and (soon) trainable routing both are. CIFAR is not a base-3-native target; it's a base-3-hostile target, and beating SSTT there would require fighting a more-or-less unwinnable battle on every axis.

### BIG-Bench / BBH — base-3-friendly by construction
200+ tasks, each measuring a qualitatively different skill. Mixture-of-experts architectures consistently show gains here because the task-routing signal is meaningful. Latent "task identity" is exactly the kind of signal routing should leverage. But: the inputs are text, and our substrate has no text embedding story yet. That's a substrate-extension cycle of its own.

### SCAN / COGS — compositional generalization
Simple input→output mappings that test whether a model composes primitives correctly. Dense nets famously fail these; modular / routed / program-synthesis approaches succeed. Would be a *clean* base-3 substrate claim if we had sequence-to-sequence infrastructure.

### Extreme classification (e.g., Amazon-670k, Wikipedia-500k)
Half-million output classes. Any dense output layer is infeasible. Every deployed solution is some form of tree, hash, or cluster — i.e., routing. Our substrate was *born* for this. But: NLP embeddings again.

### Go / Chess / board games
Ternary-native state (empty, own, opponent). Decision complexity is qualitatively different in different positions — opening theory vs tactical middle-game vs endgame tablebases. AlphaZero-style networks benefit from expert decomposition. Our substrate could represent positions natively and route decisions by phase.

### Tabular data with mixed discrete/continuous features
Most of applied ML. Ternary columns are common (sign flags, categorical indicators). Long-tail class distributions are common. Inspectability matters in regulated domains.

## Substrate capabilities we know we have

- Fast ternary distance (Hamming / weighted-Hamming / pair-IG) via NEON.
- Routed top-k selection, signed dispatch, sum-based re-aggregation.
- Hysteresis-aware re-quantization for trainable latents.
- Multi-scale signature pyramid for spatial data.
- LSH filter-ranker composition.
- All in C, no Python dependency, Apple-Silicon targeted.

What we **don't** have (yet):
- Text/sequence embedding story.
- Learned routing.
- Signature-producing trained head.
- NEON backward kernels (scalar only).
- Any experience with structured output spaces (trees, graphs, sequences).

## Gut-check on what "winning" looks like

For each candidate direction, what would "Glyph wins decisively" actually measure?

- **Tabular**: beats XGBoost/LightGBM on memory-constrained deployment with ≤ 5% accuracy penalty. Or beats them outright on interpretability + speed at scale.
- **Board games**: competitive policy/value net performance using <10% of the weight budget.
- **Compositional / SCAN**: near-perfect on length-generalized splits where dense networks fail.
- **Extreme classification**: top-k accuracy within X% of specialized routers at Y× the throughput.
- **BIG-Bench**: gains on routing-sensitive subtask categories where dense MoE baselines already show benefit.

## Uncomfortable question

Is the answer that we should leave image classification entirely? MNIST/Fashion/CIFAR are legacy benchmarks for legacy architectures. If routing-first, inspectability-first, base-3-first *actually* means something, maybe the right move is to stop measuring against SSTT on datasets SSTT was designed for, and instead measure against XGBoost on tabular / AlphaZero-lite on games / specialized routers on extreme classification.

(User has consistently said the substrate's end-game is unknowable. But we can still pick benchmarks aligned with substrate *properties* rather than benchmarks aligned with the *status quo*.)

## Residue for NODES

- "Base-3 native" decomposes into at least three independent criteria: ternary data, routing load, inspectability. A benchmark can hit one without the others.
- Legacy image benchmarks are the WORST fit for this substrate by all three criteria simultaneously. We've been swimming upstream.
- The cheapest benchmark swap that still uses current capabilities: tabular (we'd need a loader, no embedding work) or board-game state (directly trinarizable, no quantization loss).
- The most substrate-claiming swap: extreme classification (routing is load-bearing by necessity, inspectability is valuable, trinary labels or label groups are plausible).
- The most conceptually-clean swap: SCAN/COGS compositional — but it requires sequence-to-sequence infrastructure we do not have.
- **There is also the option of building one.** Custom benchmark with controlled ternary-signed features, modular class structure, and a routing-rewarding evaluation metric. Higher effort, cleaner claim.

NODES should structure these by: what does the substrate already do well, what's the shortest path from here to a defensible claim, and what does "claim" mean — accuracy, efficiency, inspectability, or some composite that specifically rewards base-3 structure.
