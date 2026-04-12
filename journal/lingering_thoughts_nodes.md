# Nodes of Interest: Lingering Thoughts

Extracted from `lingering_thoughts_raw.md`.

---

## Node 1: The engine-without-a-car problem

M4T is a complete substrate. Glyph is a thin wrapper. But nothing flows through the system end-to-end. Every test is synthetic. The gap between "correct primitives" and "working system" is where integration bugs hide — drift, saturation, composition failure.

**Why it matters:** The substrate's value is zero until it runs a model. Correct primitives are necessary but not sufficient.

---

## Node 2: The softmax/GELU blockage is artificial

The LUT generator exists and works. The tables are 5.4 MB. The only reason they aren't committed is that we deferred it. This is not a technical problem — it's a sequencing choice that can be reversed in 5 minutes.

**Why it matters:** This is the single smallest action that unblocks the largest capability (nonlinear computation → transformer forward pass).

---

## Node 3: Weight loading is the second gate

Even with softmax and GELU, we can't run inference without weights. The IO tool (`tools/m4t_io.c`) was deferred. But trix-z already has trained weights in float32 — the MNIST k-of-T model that achieved 98.22%. If we can convert those weights to MTFP cells, we can validate glyph against a known-good result.

**Why it matters:** The first external validation of M4T's correctness under real model weights.

---

## Node 4: MNIST is the minimum viable forward pass

trix-z's MNIST path uses: projection (input pixels → hidden), routed FFN (ternary tiles, weight-derived signatures, k-of-T dispatch), classification head (hidden → 10 logits). No attention. No positional encoding. Every primitive this needs already exists in M4T.

**Why it matters:** It's the shortest path from "primitives work" to "model works." If we can reproduce 98.22% MNIST accuracy using M4T primitives with trix-z's trained weights, the substrate is validated.

---

## Node 5: The MTFP19 range question is empirical, not theoretical

We tightened MAX_VAL to (3^19-1)/2 = 581M. No test broke. But no test uses real model activations. The trix-z MNIST model at d=128 operates at a scale where activations are typically ±1–10 in real units (cells ≈ ±59049–590490). That's well within 581M. But residual streams in deeper models can grow larger, especially without careful normalization.

**Why it matters:** The question "is MTFP19 wide enough?" can only be answered by running a real model. MNIST is the cheapest experiment.

---

## Node 6: The LayerNorm jitter is a real contract failure

p99/mean = 1.87 at N=64. The contract requires ≤ 1.5. This isn't deferred — it's failed. If glyph uses d_model=64, every LayerNorm call violates the contract. The likely cause: Newton-Raphson isqrt convergence varies by input (some inputs need more iterations, causing latency spikes).

**Why it matters:** This is the only contract clause that's currently failing. Everything else passes or is deferred pending measurement. This one is measured and broken.

---

## Node 7: Training in MTFP is a research frontier, not an engineering task

Standard training uses float shadow weights + STE. Glyph bans float. So training requires: MTFP gradients, MTFP optimizer state (momentum, variance), and a quantization-aware update rule that doesn't use STE. Nobody has published this. It's genuinely novel research.

**Why it matters:** Inference-only glyph is a glass-box inference runtime. Training-capable glyph is a glass-box AI. The difference is the difference between a viewer and a thinker.

---

## Node 8: The SDOT-routing bridge is unmade

MTFP4 SDOT matmul and packed-trit routing signatures exist separately. The bridge — converting activation signatures to MTFP4 for SDOT-based routing score computation — hasn't been written. It's a conversion + matmul, using primitives that exist. Small gap, disproportionate value.

**Why it matters:** Until this bridge exists, the SDOT cell is fast but disconnected from the routing layer.

---

## Tensions

### Tension A: Perfect substrate vs. usable system (Node 1 vs Node 4)

We've been perfecting the substrate (8 pipeline items, 8 red-teams). The substrate is excellent. But it doesn't DO anything. Every additional red-team pass on the substrate has diminishing returns; the first end-to-end forward pass has infinite marginal value.

### Tension B: Purity vs. pragmatism (Node 2 vs Node 3)

The LUT tables are generated from float arithmetic. Committing them crosses a conceptual line: float-derived artifacts enter the repo. The IO tool converts float weights to MTFP. Both are "float at the boundary." The policy allows this (tools/ is not libm4t.a), but it feels different when the generated artifact lives in src/.

### Tension C: Contract enforcement vs. shipping (Node 6)

LayerNorm fails the p99 contract at small N. Do we fix it before shipping, or ship with a known failure? The contract is our own standard — we wrote it. Shipping with a known violation is honest if documented, dishonest if not.

### Tension D: Inference vs. training (Node 7)

Inference-only glyph is useful but incomplete. Training-capable glyph is the real goal but requires research we haven't done. Pursuing training now delays everything else. Deferring training means glyph remains a consumer of float-trained models.

---

## Dependencies

- **Node 2 → Node 4**: Softmax/GELU needed for the MNIST forward pass (classification uses softmax over logits, FFN uses GELU).
- **Node 3 → Node 4**: Weight loading needed for MNIST.
- **Node 4 → Node 5**: MNIST validates the MTFP19 range empirically.
- **Node 4 → Node 1**: MNIST is the first end-to-end integration test.
- **Node 8 → Node 4**: SDOT routing bridge needed if the MNIST path uses MTFP4 for routing scores.
