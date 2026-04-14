---
date: 2026-04-14
phase: RAW
topic: Scrutinizing the updated model (C-sub + C-con + "keep sign_extract")
---

# Raw thoughts

The prior cycle produced an updated model:
1. Two-part criterion: C-sub (substrate: three-way structure) + C-con (consumer: inputs realize all three states).
2. Keep sign_extract, annotate contract, add threshold_extract.
3. Option 3 (MTFP4-native signatures) is a consumer pattern, no substrate change.
4. ~180 LOC scope.

Now apply LMM to that model. Not to confirm it — to see if it holds.

---

## Challenge 1: C-sub and C-con might not be cleanly separable

I wrote C-sub as "collapse any one state; if the definition becomes ill-defined or trivially binary, fail." Let me apply that to sign_extract *honestly*.

Sign_extract: `v > 0 → +1, v < 0 → -1, v == 0 → 0`.

Collapse zero → +1: `v >= 0 → +1, v < 0 → -1`. This is **a perfectly well-defined binary sign-test**. It's not ill-defined. And it IS trivially equivalent to a binary operation.

So by C-sub as I wrote it, **sign_extract FAILS C-sub**. But I said it passes. I rationalized.

Let me check signature_update's use. Inside signature_update, the input to sign_extract is `col_sum_t_d - mean_d`, an int64. Zero here means "this tile's column-sum equals the cross-tile mean." Is that "non-trivial semantic" or "happenstance of exact arithmetic equality"?

The honest answer: it's a meaningful semantic condition (the tile is mean-aligned in this dim) that occurs with non-trivial probability on integer-arithmetic inputs. So within signature_update, sign_extract IS being used in a way where the zero state is informative.

But that's a property of the consumer's input distribution — the same consumer-side check I called C-con. I can't cleanly locate "structural three-wayness" at the substrate level for sign_extract. Its three-wayness is a *conditional* property that activates when inputs realize zero meaningfully.

So: **C-sub and C-con collapse into one thing for this primitive.** The two-part separation I proposed was rhetorical, not structural.

That's a problem for the updated model. If C-sub and C-con aren't separable, the substrate-side guarantee is hollow; only the consumer-side check does real work.

---

## Challenge 2: The "keep sign_extract + rename contract" decision

Arguments for keeping sign_extract:
- signature_update uses it internally; deletion forces a rewrite.
- The relationship to threshold_extract (τ=0 degenerate) is documented.
- Minimal disruption to existing tests.

Arguments for deletion:
- Renaming a contract is weaker than removing the primitive. Consumers can ignore docstrings; they can't ignore missing symbols.
- Keeping sign_extract *invites the same failure mode* — future consumers will call it because it exists, without reading the docstring about C-con.
- Minimum surface: one extractor (threshold_extract) replaces two (sign_extract + threshold_extract).
- signature_update's rewrite is ~3 lines: `sign_extract(...)` → `threshold_extract(..., 0)`.

When I wrote the synthesize, I weighed "minimal disruption" heavier than "minimum surface." But the user's instruction was explicit: *accuracy now, not convenience now*. Minimum surface is the architecturally correct choice here. I picked the convenient one.

So the updated model's decision to keep sign_extract was ... incremental reasoning, not rigorous reasoning.

---

## Challenge 3: Option 3 is not zero-substrate-change

My synthesize said Option 3 (MTFP4-native signatures, no extraction) requires no substrate changes. That's wrong.

To use an MTFP4 mantissa array as a packed-trit signature, you need to **unpack the 4-trit mantissas into individual trits**, then pack those into the `uint8_t` buffer format that `m4t_popcount_dist` consumes.

Current primitives:
- `m4t_pack_trits_1d(dst_packed, src_trits, n)` — packs m4t_trit_t values (single {-1, 0, +1} each). Not MTFP4 mantissas.

What's missing:
- `m4t_mtfp4_unpack_to_trits(dst_trits, src_mtfp4, n)` — expand each 4-trit mantissa into 4 m4t_trit_t values.

So Option 3 DOES require a substrate addition. My synthesize missed this. The scope is larger than 180 LOC if we also enable Option 3 — call it ~240 LOC. Not catastrophic, but the synthesize was wrong.

Or: Option 3 is OUT of this pass. Defer until a consumer actually needs MTFP4 signatures. But then the "Option 3 is available as a pattern" claim is a lie until we add the primitive.

---

## Challenge 4: Risks the updated model doesn't address

Risk A: **Consumers will claim C-con without verifying it.** Easy to assert "my inputs realize all three states." The sign_extract bug happened because no one ran the test. What's the review gate?

Risk B: **Renaming sign_extract to communicate the contract is weaker than deletion.** A name can be ignored. An absence cannot.

Risk C: **Two similar primitives create ongoing confusion.** Future contributors will miss the distinction between sign_extract and threshold_extract. The distinction is a parameter value — that's naturally a parameter, not two primitives.

Risk D: **The criterion requires depth to apply.** Casual review will see "three output codes" and think "it's ternary." The sign_extract bug hid precisely because of this.

These risks are cumulative. Each one is small; together they suggest the updated model is too lenient on sign_extract.

---

## Challenge 5: What if the correct answer is to delete?

Let me steelman: delete sign_extract, add only threshold_extract, rewrite signature_update's internal call as `threshold_extract(τ=0)`.

Consequences:
- One extractor primitive, parameterized. Clearer architecture.
- signature_update rewrite: ~5 lines.
- Tests for sign_extract: replaced by threshold_extract tests with τ=0 case.
- `m4t_route_sign_extract` symbol gone. Any future consumer that wants sign-only has to write `threshold_extract(..., 0)`, which makes the "sign is a special case" structural truth visible at every call site.
- Option 3 still needs mtfp4_unpack_to_trits if we enable it; unchanged.

Scope: ~180 LOC in the updated model becomes ~160 LOC under deletion (less boilerplate for the legacy primitive). Actually smaller.

Why didn't I propose this? Because I was thinking "backward compatibility" in a pre-1.0 repo with no external consumers. That's exactly the kind of premature compatibility preservation NORTH_STAR warns about.

---

## Challenge 6: Is the criterion even right?

Beyond the C-sub/C-con separation issue, is the CORE criterion correct?

The thing we actually want to guard against: **binary-shape hidden in ternary vocabulary**. A primitive that looks three-state in its type but behaves two-state in practice.

Alternative criterion: **"every trit code the primitive's API can emit must be emitted by a non-trivial set of realistic inputs."** This is a behavioral criterion on the primitive-consumer pair. No separation into substrate/consumer. Just: under the intended input distribution, is the output space actually utilized?

For sign_extract + MTFP projections: output code 0 is emitted almost never. Criterion FAILS.

For sign_extract + integer-differences: output code 0 is emitted meaningfully. Criterion PASSES.

For threshold_extract(τ>0) + any realistic input: all three output codes can be emitted. Criterion PASSES.

This is simpler than the two-part version. It explicitly acknowledges that the criterion is per-deployment, not per-primitive.

And it suggests a clean operational protocol: **every primitive in the substrate must come with a statement of the input distributions for which it passes this criterion.** The substrate documentation is the set of sanctioned (primitive, input-class) pairs.

---

## What I notice I'm doing

I'm steelmanning the deletion path harder than I steelmanned the keep path in the prior cycle. That might be overcorrection. Let me check: the user explicitly said accuracy matters, mistakes compound, they don't have time to waste. Given that, the architecturally correct move is usually the simpler, more-disciplined one. Deletion is that move.

Also: the prior cycle ended with "validate with the user before any code." Now I'm evaluating whether the thing to validate was right. That IS the right meta-move.

---

## Questions arising

1. Is C-sub/C-con a real separation or rhetorical? (I now think: rhetorical.)
2. Is "keep sign_extract" the right call or incremental convenience? (Leaning: incremental.)
3. Is Option 3 zero-substrate-change as I claimed? (No — needs mtfp4_unpack_to_trits.)
4. What's the simplest criterion that prevents the hidden-binary failure? (Candidate: "every output state must be emitted non-trivially under the intended input distribution.")
5. Should the substrate spec list (primitive, input-class) pairs as sanctioned deployments? (Maybe — it's the operational form of the criterion.)
6. Does deleting sign_extract cascade in ways I haven't traced? (Need to audit: all callers, all tests, all docs that mention it.)

---

## What scares me

If I just validated the prior cycle's output without this scrutiny, we'd have committed to:
- A criterion that's less separable than I claimed.
- A decision (keep sign_extract) that I made incrementally.
- A scope that missed Option 3's substrate addition.

Three errors layered on top of each other. The cycle catches them; not running the cycle would have compounded them. The user was right to insist on rigor.

What I still don't know: whether THIS cycle catches everything, or whether a third meta-meta-cycle would find issues with this one. At some point you have to commit — but commit only when the current analysis stops surfacing substantive challenges. Am I there? Probably not quite.

---

## First instincts to watch

- "The prior cycle was careful, this is over-thinking" — no, the prior cycle had specific errors this is exposing. Trust the process.
- "Deletion is too aggressive" — might be, but it's the minimum-surface choice and the user asked for accuracy, not preservation.
- "Let's just try both and see" — risk-aversion masquerading as rigor. Pick one and defend it.
