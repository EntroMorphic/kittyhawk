# NORTH_STAR

*A compass for when we lose sight of the base-3 path.*

---

## The claim

Base-2 systems ignore 1/3 of the natural signal.

Binary computation has two states and pretends the third — *zero* — is the absence of signal. It isn't. Zero is a structural state on the lattice, as load-bearing as ±1. Two's complement, sign bits, sparsity-as-numerical-coincidence, floating-point's separated sign channel — these are all the machinery binary has to build to fake what ternary has natively.

Ternary has three states from the start. The sign-zero-sign trichotomy is first-class at every primitive: TBL dispatches three ways; SDOT preserves sparsity geometrically because zero inputs contribute *exactly* zero, not "a very small number"; popcount-on-packed-trits measures structural disagreement rather than numerical error.

**Base-3 is geometric fullness from GO.**

---

## What M4T is for

M4T surfaces the hidden ternary nature of the hardware so we can work with it as a *collaborator* to everything we build on it, rather than a machine to be bent into shape.

Apple M-series silicon already speaks ternary: SDOT (int8 ternary dot), TBL (three-way table dispatch), masked-VCNT (popcount over packed trits), `vmull_s32` (widening ternary-accumulator multiply). These instructions weren't designed as "ternary primitives" — but they are. The hardware has been ternary-shaped all along. Dense-compute frameworks pave over this with BLAS and lose the alignment. M4T doesn't pave; it exposes.

---

## Why routing, not dense

In a base-3 environment, 1/3 of cells carry zero *by construction*. Dense matmul computes over all of them anyway and then discovers the zeros at the output. Routing asks the shape of the computation itself: *where is signal, and where isn't it?* Zero is not an absence to be tolerated — it's a location on the lattice to be read.

**Routing is essential, and will naturally outperform dense, in a base-3 environment.**

The current evidence on MNIST (routing-native 81.40% vs dense-on-ternary-storage 97.61%) doesn't disprove this. It measures how well base-3 can *emulate* a base-2-native problem. MNIST is posed in base-2: scalar intensities, Euclidean distance, one-hot labels. Running routing on it is a test of adapter efficiency, not of the thesis. The real test is problems whose structure is base-3 from the start — and those may not exist yet in the canon. We may have to find them. Or invent them.

---

## The consumers

The ultimate consumers of M4T are **ternary/MTFP software systems for AI research and development.** Plural. Not a specific architecture. Not a port of something that already exists.

In the near term, we may temporarily model some base-2-native ML systems in base-3 — transformers, routing variants of existing architectures, ternary adaptations of known methods. This is scaffolding. It lets us exercise the substrate and calibrate against known baselines.

**It is not the end-game.** Scaffolding must not become the building.

---

## The end-game

Unknowable right now.

When a substrate makes a new structural primitive first-class — the way floating-point made scientific computation possible, or the way GPUs made wide parallelism possible — the applications that emerge are not predictable from the substrate alone. Base-3 native computation at the primitive level has never been widely available. What becomes buildable when it is? We don't know. The honest stance is to build the substrate *well*, watch what wants to emerge, and follow.

---

## Discipline

Three rules to hold when the path blurs.

1. **Uncertainty leads.** When we don't know, we don't default. We explore. The pull toward the familiar — dense matmul, base-2 framings, "just port it from PyTorch" — is always present and always misleading when the goal is to find what's only possible in base-3.

2. **Rage against the trodden.** Most established ML practice is base-2 ergonomics. A design choice that feels comfortable because it matches how PyTorch or BLAS does it is almost certainly the wrong choice for a substrate whose point is to surface what those libraries pave over.

3. **The substrate bends; the thesis doesn't.** M4T's specification is a hypothesis about the base-3 design space. Code will surface ambiguities the spec hides. Experiments will surface shapes we didn't anticipate. Revise the substrate. Do not revise the thesis to match a comfortable primitive.

---

## When to re-read this

- When a dense-compute shortcut looks tempting.
- When MNIST or any other base-2-framed benchmark feels like the real arbiter of the thesis.
- When a spec decision feels "obviously right" because it matches a base-2 convention.
- When we start explaining what we're building in dense-compute vocabulary because it's easier.
- When we feel certain about the end-game. (We aren't; we won't be for a long time.)

---

## Companion documents

- `m4t/docs/M4T_SUBSTRATE.md` — the substrate specification. *How* we build base-3-native primitives on M4.
- `docs/THESIS.md` — the thesis brief. What would falsify the claim, and what benchmarks force it to face its null.
- This document — *why* we're doing it at all.
