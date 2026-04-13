# Nodes of Interest: Trit Lattice LSH

Extracted from the session's experimental sequence. Each node is a discovery or turning point.

---

## Node 1: Dense MTFP matmul is 95% of compute

Profiling the MNIST forward pass revealed that the dense MTFP×MTFP matmul (projection + head) consumed 95.3% of cycles. The ternary FFN — the entire routed architecture — was 4%. The routing decision was 0.03%. The expensive operation was __int128 multiply-and-rescale per element.

**Why it matters:** The bottleneck isn't the ternary computation. It's the non-ternary layers that exist only because the weights were trained in float.

---

## Node 2: MTFP multiplication is the wrong operation

MTFP × MTFP requires rescaling (dividing by SCALE) because both operands have scale. Trit × MTFP requires no rescale because trits are dimensionless. The division is the cost. On the MTFP lattice, multiplication moves you off-grid; trit operations keep you on-grid.

**Why it matters:** The "geometric in nature" insight — the natural operations on the lattice are add/subtract/skip, not multiply. Every dense matmul is a symptom of non-ternary weights.

---

## Node 3: STE cannot train all-ternary from random init

Three experiments with all-ternary-in-the-loop training from random initialization: SGD at lr=0.0174, lr=0.1, lr=1.0, and AdamW at lr=0.0174. All stuck at 11.35% (predicting a single class). SGD updates are too small to flip trits; large LR oscillates without converging; AdamW accumulates but the gradient signal through random ternary layers is pure noise.

**Why it matters:** Gradient-based training of ternary networks from scratch is a dead end. The gradient highway requires smooth (continuous) layers to carry signal. Random ternary layers shatter gradients.

---

## Node 4: The routing IS locality-sensitive hashing

A ternary signature defines a hyperplane on the MTFP lattice. The dot product measures which side of the hyperplane the input falls on. Multiple signatures = multiple hyperplanes = a partition of the lattice into regions. This is exactly LSH — the popcount distance is hash agreement, the routing decision is bucket assignment, the tile computation is per-bucket transform.

**Why it matters:** Reframes the architecture from "neural network with quantized weights" to "geometric hash table on the trit lattice." The hash functions (signatures) are the intelligence. The tiles are the computation. Training = finding good hash functions.

---

## Node 5: Train the routing, freeze the weights

Inversion of the trix-z paradigm. Instead of learning weights and deriving routing, fix the weights (tiles as static tools) and learn the routing (which tools to apply). The routing space is tiny: C(T,K) × 2^K possibilities per token, vs 3^(D×D) weight configurations per tile.

**Why it matters:** Makes training a discrete search over a small combinatorial space, not gradient descent over a continuous high-dimensional space. No float needed.

---

## Node 6: Class centroids on the lattice give 59.50%

Ten ternary signatures derived from class centroids (sign of centroid minus global mean). One ternary matmul per test image. 59.50% accuracy — 6× above random, proving the lattice geometry carries discriminative structure.

**Why it matters:** First proof that zero-float, zero-gradient classification works on the trit lattice. The geometry is real.

---

## Node 7: Pairwise signatures don't help much (60.01%)

Ninety pairwise signatures (sign of centroid_i minus centroid_j) gave 60.01% — barely above the 10-template baseline. The centroids are too blurry; pairwise differences are nearly redundant with centroid-vs-global differences.

**Why it matters:** More templates of the same quality don't help. The bottleneck is template QUALITY (sign discards magnitude), not quantity.

---

## Node 8: Random ternary projections give 79.74%

256 random ternary vectors as LSH hash functions, projecting from 784-dim pixel space to 256-dim hash space. L1 nearest centroid in hash space. 79.74% — a 20-point jump from centroid signatures.

**Why it matters:** Random projections preserve distance structure on the lattice (Johnson-Lindenstrauss for ternary). The projections don't need to know about the classes — they capture geometric structure that class-specific templates miss.

---

## Tensions

### Tension A: Random projections vs data-dependent projections

Random ternary projections gave 79.74%. Data-dependent projections (centroid-derived) gave 59.50%. Counterintuitively, random is better. This is because random projections are DIVERSE (each captures a different slice of the geometry) while centroid projections are REDUNDANT (all capture the same class-mean structure).

The resolution: data-dependent projections should be diverse AND informed. Find ternary vectors that maximize class separation while being decorrelated from each other.

### Tension B: Zero-float purity vs accuracy

The float-trained model achieves 97.46% with M4T inference. The zero-float Trit Lattice LSH achieves 79.74%. The gap is 17.7 percentage points. Can the gap be closed within the zero-float framework, or is float training necessary for high accuracy?

### Tension C: Simple classifier vs deep architecture

The 79.74% result uses a single projection layer + L1 nearest centroid — no FFN, no routing, no GELU, no LayerNorm. The full architecture (projection → LN → route → FFN → head) adds complexity but also capacity. Can the LSH framework extend to multi-layer architectures while staying zero-float?

---

## Dependencies

- **Node 2 → Node 4**: The insight that MTFP is geometric leads to the LSH framing.
- **Node 3 → Node 5**: STE failure motivates "train routing, freeze weights."
- **Node 6 → Node 8**: Centroid signatures prove the concept; random projections improve it.
- **Node 4 → Node 8**: LSH theory predicts that random projections should work.
