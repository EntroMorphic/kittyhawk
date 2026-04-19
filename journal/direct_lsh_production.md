# Direct LSH + GSH — production configurations and corrected claims

Date: 2026-04-18

## Corrected production configurations

Each dataset has its own optimal settings. Gradients help on
texture-rich datasets (Fashion, CIFAR) but hurt on sparse
datasets (MNIST). Density is dataset-dependent.

### MNIST

```bash
./build/direct_lsh --data <mnist> --density 0.10 --m_max 64
```

No gradients. Low density (10% zeros): MNIST's sparse pen
strokes are almost entirely foreground signal. Higher density
or gradients add noise.

```
LSH k=5-rw:     97.23%    (overall, all 10K queries)
GSH 1-NN:        96.65%
Agreement:       98.20%   (9820 / 10000)
P(correct|agree): 98.01%  (9625 / 9820, subset)
```

### Fashion-MNIST

```bash
./build/direct_lsh --data <fashion> --no_deskew --density 0.395 --m_max 64 --gradients
```

Gradients ON. Density 0.395 (39.5% zeros): normalized clothing
textures benefit from gradient edge detail.

```
LSH k=5-rw:     87.78%    (overall, all 10K queries)
GSH 1-NN:        85.26%
Agreement:       90.99%   (9099 / 10000)
P(correct|agree): 91.06%  (8286 / 9099, subset)
```

### CIFAR-10

```bash
./build/direct_lsh --data <cifar> --no_deskew --density 0.395 --m_max 64 --gradients
```

Gradients ON. Density 0.395. Normalized natural images need
both intensity and gradient channels.

```
LSH k=5-rw:     44.68%    (overall, all 10K queries)
GSH 1-NN:        36.87%
Agreement:       50.00%   (5000 / 10000)
P(correct|agree): 56.36%  (2818 / 5000, subset)
```

## Corrected claims

### What P(correct|agree) means and doesn't mean

P(correct|agree) is the accuracy on the SUBSET of queries where
LSH and GSH agree. It is NOT the overall accuracy. Overall
accuracy is the LSH k=5-rw number.

The agreement filter identifies WHERE the system is confident.
On CIFAR-10, the confident 50% has 56.36% accuracy while the
uncertain 50% has 33.00%. The filter stratifies but does NOT
improve overall accuracy — the combined overall is still 44.68%.

To USE the agreement signal productively, the uncertain subset
needs a DIFFERENT treatment — escalation to wider resolution,
the FFN bridge, or a fallback classifier. Without that, the
agreement filter is diagnostic only.

### SSTT comparison (corrected)

SSTT achieves 53% on ALL 10K CIFAR-10 queries. Our LSH achieves
44.68% on all 10K queries. Our P(correct|agree) of 56.36%
applies to only 5K queries. The comparison is INVALID as stated
in the prior commit message. The correct comparison:

| system | scope | accuracy |
|---|---|---|
| SSTT | all 10K | 53% |
| Glyph LSH | all 10K | 44.68% |
| Glyph agree subset | 5K (50%) | 56.36% |

Glyph's overall CIFAR-10 accuracy (44.68%) is 8.3pp below SSTT.
The agreement filter identifies a high-confidence subset but
doesn't close the gap on the full dataset.

### Per-dataset findings

**Gradients:** help Fashion-MNIST (+1.15pp) and CIFAR-10
(required for competitive accuracy). Hurt MNIST (−0.69pp at
d=0.395). MNIST should be run WITHOUT --gradients.

**Density:** MNIST peaks at d=0.10 (97.23%), Fashion and CIFAR
peak at d=0.395 (87.78%, 44.68%). The optimal density reflects
the image sparsity — sparse pen strokes need low density (almost
all pixels are signal), dense natural images need moderate density
(filter noise).

## Architecture summary

```
pixels → normalize → quantize (intensity + optional gradients)
  → hierarchical spatial pooling → multi-table bucket key
  → multi-probe → union → full-signature Hamming k=5-NN
  → per-query routing pattern → GSH bucket → GSH resolve
  → agreement analysis
```

Zero random projections. Every trit is a specific pixel or
gradient. The structural zero (W_f[hidden]=0) filters noise
via the density-calibrated tau threshold.
