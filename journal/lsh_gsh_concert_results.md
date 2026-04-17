# LSH + GSH in concert — first light results

Date: 2026-04-17
Tool: `tools/layered_lsh.c`

## Architecture

Two routing instruments playing together:

**LSH (Local Signature Hash):** standard multi-table bucket-indexed
LSH on pixel signatures. M=64 tables, N_PROJ=16. Finds prototypes
the query LOOKS LIKE. Scores by summed Hamming distance (k=5
rank-weighted k-NN). This is the geometric voice.

**GSH (Global Signature Hash):** hashes the LSH's per-table 1-NN
vote pattern directly as a multi-trit signature — no random
projection. Each table's vote label (0-9) is encoded as 4 trits
(unique codeword per class). M=64 tables × 4 trits = 256 trits
= 64 bytes. Bucket-indexed on the first 16 trits (4 bytes).
Multi-probe explores routing-pattern neighborhoods. Finds training
images the query ROUTES LIKE. This is the topological voice.

Both fully routed: bucket index, multi-probe, k-NN resolve. The
GSH uses the SAME infrastructure as the LSH (glyph_bucket,
glyph_multiprobe, m4t_popcount_dist) on a different input —
routing patterns instead of pixel signatures.

## Evolution

Three versions were tested:

**V1 (one-hot + random projection):** encoded per-table labels
as a 640-dim one-hot binary vector, then projected through random
ternary weights to produce Layer 2 signatures. Results: GSH at
chance level. Root cause: random projection added noise to an
already-clean signal. The GSH became a noisy copy of the LSH.

**V2 (one-hot + distance enrichment):** added per-table 1-NN
distance to the one-hot encoding. Marginal improvement. Same
root cause: random projection of the routing pattern destroys
the signal.

**V3 (multi-trit direct encoding, no random projection):** each
vote label encoded as 4 trits via a fixed codebook. The routing
pattern IS the signature — no random projection, no τ calibration.
The GSH hashes vote patterns directly. This is the version that
works.

## Results — V3 (production)

### MNIST (deskew on, density=0.33)

```
  LSH k=5-NN:                  97.70%
  GSH k=5-NN:                  90.84%

  Agreement rate:               91.55%  (9155 / 10000)
  P(correct | agree):           98.74%  (9040 / 9155)
  P(LSH correct | disagree):    86.39%  (730 / 845)
  P(GSH correct | disagree):     5.21%  (44 / 845)
```

### Fashion-MNIST (no_deskew, density=0.33)

```
  LSH k=5-NN:                  85.73%
  GSH k=5-NN:                  79.95%

  Agreement rate:               82.23%  (8223 / 10000)
  P(correct | agree):           91.97%  (7563 / 8223)
  P(LSH correct | disagree):    56.84%  (1010 / 1777)
  P(GSH correct | disagree):    24.31%  (432 / 1777)
```

### CIFAR-10 (no_deskew, density=0.33)

```
  LSH k=5-NN:                  37.06%
  GSH k=5-NN:                  19.28%

  Agreement rate:               22.29%  (2229 / 10000)
  P(correct | agree):           46.34%  (1033 / 2229)
  P(LSH correct | disagree):    34.40%  (2673 / 7771)
  P(GSH correct | disagree):    11.52%  (895 / 7771)
```

## Key findings

### 1. The GSH's voice is CONFIRMATION, not override

The GSH is consistently weaker than the LSH in standalone accuracy
(90.84% vs 97.70% MNIST, 79.95% vs 85.73% Fashion, 19.28% vs
37.06% CIFAR). It never earns override authority — on disputed
queries, the LSH is always more often right.

But the GSH provides something the LSH cannot: a CONFIDENCE
SIGNAL. When LSH and GSH agree, accuracy jumps:

| dataset | LSH alone | P(correct \| agree) | Δ |
|---|---|---|---|
| MNIST | 97.70% | 98.74% | +1.04 |
| Fashion-MNIST | 85.73% | 91.97% | +6.24 |
| CIFAR-10 | 37.06% | 46.34% | +9.28 |

The agreement filter identifies the confident subset where
accuracy is significantly higher than the baseline.

### 2. Disagreement identifies the hard queries

On CIFAR-10, the 77.7% of queries where LSH and GSH disagree
have only 34.4% LSH accuracy — 2.7pp below the 37.06% baseline.
The disagreement subset is genuinely harder. The agreement/
disagreement split is a routing-native confidence gate that
separates easy from hard without any threshold parameter.

### 3. The GSH distance metric is vote disagreement

The GSH's Hamming distance counts how many tables' votes differ
between two images (weighted by the 4-trit codeword distance).
This is a TOPOLOGICAL distance: it measures how similarly the
trit lattice SEES two images, regardless of their pixel-space
distance.

Two images can be pixel-distant but route-similar (same class
surrounded by different backgrounds) or pixel-close but route-
different (similar textures, different classes). The GSH captures
the routing similarity that the LSH's pixel distance cannot.

### 4. The bucket index generalization was needed

The bucket index (glyph_bucket.c) was generalized from
`sig_bytes == 4` to `sig_bytes >= 4`. All signatures are keyed
on the first 4 bytes (16 trits). Wider signatures use the first
4 bytes for bucket routing and the full signature for Hamming
distance scoring. This unblocks N_PROJ=64 bucket indexing as
well as the 64-byte GSH signatures.

### 5. Agreement rate tracks dataset difficulty

| dataset | agreement rate | LSH accuracy |
|---|---|---|
| MNIST | 91.55% | 97.70% |
| Fashion-MNIST | 82.23% | 85.73% |
| CIFAR-10 | 22.29% | 37.06% |

Harder datasets produce more disagreement. The agreement rate
itself is a routing-native measure of dataset difficulty.

## Architecture insight

The GSH's training pass reveals the operational relationship:

1. Build LSH tables on pixels (standard).
2. Route ALL training images through LSH (probe + union).
3. Extract per-table 1-NN labels from each training image's
   LSH union (excluding self-match).
4. Encode the vote patterns as GSH signatures.
5. Build GSH bucket index on the vote-pattern signatures.

At test time:
1. Route query through LSH → get LSH prediction.
2. Extract query's vote pattern from LSH union.
3. Route the vote pattern through GSH → get GSH prediction.
4. LSH and GSH agree → high confidence.
   LSH and GSH disagree → low confidence, flag for escalation.

The two instruments share the LSH's infrastructure — the GSH
reads the LSH's routing pattern, not raw pixels. The GSH is
DOWNSTREAM of the LSH, not parallel. They're in series, not
in parallel — the bass line (LSH) plays first, and the melody
(GSH) harmonizes with it.

## Training routing signatures: LSH-routed, not brute-force

The training routing signatures are computed via LSH probing,
not brute-force. For each training image i:
- Probe M tables through the bucket index + multi-probe.
- Build union from the probe hits.
- Extract per-table 1-NN labels from the union (excluding self).

This is O(union_size × M) per training image — much faster than
brute-force O(N_train × M). The routing signatures reflect the
LSH's ROUTED view of each training image's neighborhood, which
is the correct input for the GSH.

## Brute-force ceiling comparison (CIFAR-10)

```
                                accuracy
Single-layer LSH k=5 M=64:      37.06%
Brute-force N_PROJ=64 M=64 k=5: 38.14%  (ceiling, no filter)
Multi-resolution combined k=5:  37.90%
GSH standalone k=5:              19.28%
P(correct | LSH+GSH agree):     46.34%  (22.3% of queries)
```

The agreement filter achieves 46.34% on its confident subset —
8pp above the brute-force ceiling. This is not a contradiction:
the agreement filter SELECTS the easy queries (22.3% of the
dataset) where accuracy is naturally higher. It doesn't improve
accuracy on ALL queries; it identifies WHERE the architecture
is confident.

## What comes next

The agreement signal opens the path to the dynamic cascade:
- Stage 1: LSH + GSH. If they agree, accept (high confidence).
- Stage 2: for disagreement queries, escalate to wider N_PROJ
  or more tables.
- Stage 3: for persistent disagreement, escalate further.

The GSH provides the confidence gate the cascade needs, computed
from routing measurements rather than arbitrary thresholds.
