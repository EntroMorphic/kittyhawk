---
date: 2026-04-14
phase: REFLECT
topic: §14 "Seven Open Decisions" in m4t/docs/M4T_SUBSTRATE.md
---

# Reflections

## Core insight

**§14 was an honest list written before triage.** The "seven open decisions" aren't seven of the same thing. Once the throughline (substrate/consumer boundary) is applied as a *sorting mechanism* instead of an answer-generator, the seven collapse to:

- **1 proven contract** (14.5) — doesn't belong in "open" at all.
- **2 outside-the-substrate** (14.6, 14.7) — consumer/thesis concerns, wrong section.
- **3 substrate-real with natural answers** (14.2 defer, 14.3 zero-pad, 14.4 parallel status array).
- **1 substrate-real and genuinely empirical** (14.1).

The emergent answer isn't "here are seven solutions." It's *"six of the seven dissolve once categorized properly, and one remains as a real empirical open."*

## Resolved tensions

### T1 (elegant throughline vs heterogeneous list) → RESOLVED
The throughline wasn't wrong; it was exposing a categorization bug. Apparent unity across the seven was a symptom of grouping disparate questions. The method worked: pushing on the elegance revealed the upstream error instead of papering over it.

### T2 (widen-don't-round vs zero-pad) → RESOLVED
Zero-padding inserts the additive/multiplicative identity. No values change; no rounding occurs; no information is lost. The §8.5 invariant holds. The apparent conflict was a surface-level word-match (both involve "silent" substrate behavior) with no actual semantic conflict.

### T3 (answer-everything vs defer-until-consumer) → RESOLVED in favor of defer
The substrate's discipline is to **only answer questions it has to answer**. 14.2 (cross-block add) is the exemplar: answering it speculatively would add substrate opinion without a consumer requirement driving it. Under the throughline, that's substrate overreach.

## Hidden assumptions challenged

1. **"Each decision is independent."** False. 14.6 is a corollary of 14.7.
2. **"§14 is all substrate."** False. 14.6 and 14.7 are consumer/thesis concerns.
3. **"All opens deserve answers now."** False. Some opens should sit until a consumer demands an answer — that's not negligence, it's respect for the substrate/consumer boundary.
4. **"An elegant throughline should produce uniform answers."** False. An elegant throughline *sorts*. It tells you which questions go in which pile. Uniform answers come only when the questions turn out to be uniform — which, here, they weren't.
5. **"Listing an open question and declaring a lean is enough."** False. Several of my "leans" were filling a table rather than doing the triage. The discipline demanded by the method is to first ask *"is this question even the substrate's to answer?"* before proposing an answer.

## What I now understand

The thesis from earlier still holds: **the substrate guarantees invariants; the consumer names preconditions.** But it's a sorting principle, not an answer principle. Applied consistently, it does three things to §14:

1. **Kicks out** questions that aren't the substrate's (14.6, 14.7).
2. **Resolves** questions where the principle has a clear downstream answer (14.2 defer, 14.3 zero-pad, 14.4 parallel array).
3. **Leaves in place** the one question that is both substrate-scope and not derivable from principle (14.1 — requires empirical workload data).

Six of the seven "opens" were either mis-filed or answerable by uniform application of the throughline. The remaining one is empirical, and answering it prematurely would be the same kind of substrate-swallows-consumer-concern mistake that the fixed-point collapse was.

## What the method surfaced that I missed on first pass

- 14.5 not being an open decision at all. I wrote it as open because I was filling a "considered alternatives" list. The method's NODES phase exposed this: it's a theorem.
- 14.3's apparent conflict with §8.5 being illusory. First-pass framing obscured the identity-element argument.
- 14.6's dependency on 14.7 — I had them as independent bullets. They aren't.

Each of those was a latent bug in the spec document itself. Running the cycle corrected the spec, not just my opinion about the decisions.

## What remains uncertain

Only 14.1: the logical block size. Genuinely empirical, cannot be answered without a workload. Documenting it as "deferred pending first consumer" is the honest position.
