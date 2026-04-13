# Reflections: Trit Lattice LSH

Working from `trit_lattice_lsh_nodes.md`.

---

## The "why" ladder

1. **Why does the float-trained model get 97% and the zero-float model get 79%?** Because the float model has 398K continuous parameters that were optimized by gradient descent; the zero-float model has 256 random ternary vectors and a nearest-centroid classifier. The float model has more capacity and a better optimization algorithm.

2. **Why can't we give the zero-float model more capacity?** We can. More projections, more layers, data-dependent projections. The framework supports it. We just haven't built it yet.

3. **Why did we start with the simplest thing?** To prove the concept. 79.74% with zero float proves the lattice geometry is real. Now we add structure.

---

## Core insight

> **The trit lattice is a geometric space where ternary operations are the natural metric. Classification is partitioning this space. Training is finding good partitions. Gradient descent is one way to find partitions; geometric construction is another. On the trit lattice, geometric construction is native — gradient descent is foreign.**

Float gradient descent is a general-purpose optimizer that works on any differentiable function. But the trit lattice isn't differentiable — it's discrete. Using gradient descent on it (via STE) is like using a screwdriver as a hammer: it works badly because it's the wrong tool for the substrate.

The right tool for a discrete geometric space is discrete geometric construction: compute statistics on the lattice, find separating hyperplanes, assign trits. Every step is a lattice operation.

---

## Resolved tensions

### Tension A — random vs data-dependent

**Resolution: both, in layers.** Random projections provide a diverse, distance-preserving base representation (like a hash table). Data-dependent projections (derived from class statistics in the random-projection space) provide a discriminative second layer. The random layer captures geometry; the data-dependent layer captures task structure.

This is a two-layer LSH:
1. Random ternary projection: 784-dim → N-dim (distance-preserving hash)
2. Data-dependent ternary projection: N-dim → 10-dim (class-discriminative hash)

Layer 1 is task-agnostic. Layer 2 is task-specific. Both are ternary. Both are computed from the data via integer statistics. Zero float.

### Tension B — purity vs accuracy

**Resolution: the gap is capacity, not float.** The float model has 398K parameters across 4 dense layers with nonlinearities. Our LSH model has 256 random vectors + 10 centroids. The gap isn't because float is fundamentally better — it's because the float model has more expressive power. Adding more ternary projections and a multi-layer structure within the zero-float framework should close the gap.

The test: can a 2-layer Trit Lattice LSH with 1024+ projections and data-dependent second-layer signatures match the float model's 97%?

### Tension C — simple classifier vs deep architecture

**Resolution: depth is about composing lattice operations, not about gradients.** A multi-layer LSH is: project → hash → re-project → hash → classify. Each layer is a ternary matmul followed by sign extraction or nearest-centroid. No GELU needed (the nonlinearity IS the sign function). No LayerNorm needed (the ternary projection is inherently normalized). No residual connections needed (each layer produces a fresh hash, not a correction).

The architecture simplifies radically:
- **Layer**: ternary matmul → sign
- **Stack N layers**: each refines the partition
- **Classify**: L1 nearest centroid in the final hash space

No GELU LUT (5.4 MB saved). No LayerNorm (no isqrt). No dense matmul (no __int128). Just ternary matmul and sign extraction, all the way down.

---

## Hidden assumptions surfaced

### Assumption 1: The random projections should be truly random

Not necessarily. Structured random matrices (e.g., sparse ternary with a fixed fraction of nonzeros, or Hadamard-like ternary matrices) might preserve distance better than i.i.d. random trits. The theory (Achlioptas 2003) says sparse random projections work as well as Gaussian, but the optimal sparsity and structure for the trit lattice is unexplored.

### Assumption 2: L1 distance is the right metric

L1 distance in the projection space treats all projections equally. But some projections are more informative than others (the ones where class centroids are most separated). Weighted L1 distance (weighting by discriminative power) might improve accuracy. The weights would be ternary-scale integers derived from the inter-class centroid distances.

### Assumption 3: More projections = better

Johnson-Lindenstrauss says: more projections → better distance preservation. But at some point, the representation becomes overcomplete and the nearest-centroid classifier can't exploit the extra dimensions. A different classifier (e.g., per-class ternary templates in the hash space, matched via ternary matmul) might scale better with projection count.

---

## What I now understand

The Trit Lattice LSH framework is sound. The 79.74% result proves the geometry works. The path to higher accuracy is:

1. **More projections** (512, 1024) to capture finer geometric structure.
2. **Data-dependent projections** for the second layer: compute ternary signatures that maximize class separation in the first layer's projection space.
3. **Multi-layer hashing**: project → sign → re-project → sign → classify. Each layer is a ternary matmul. Depth adds capacity without adding float.
4. **Better classifier**: replace L1 nearest centroid with a ternary matmul classifier (class-specific ternary templates in hash space).

All of this is integer. All of this uses M4T primitives that already exist. The substrate was built for this — we just didn't know it yet.

---

## What would make me feel at peace

One number: **Trit Lattice LSH ≥ 90% on MNIST with zero float.** That would prove the geometric framework can approach neural-network-level accuracy on a real task without gradient descent. The remaining gap to 97% is capacity and architectural refinement, not a fundamental limitation.

The 90% threshold is achievable with: 1024 random ternary projections + data-dependent second-layer signatures + ternary matmul classifier. All integer.
