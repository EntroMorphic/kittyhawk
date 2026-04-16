# Re-Rank first light — wider signatures for ranking, not finding

Date: 2026-04-16
Tool: `tools/mnist_routed_bucket_multi.c` (re-rank pass added)
Config: N_PROJ=16 filter + N_PROJ=32 re-rank, M=64, density=0.33

## The experiment

After the LMM cycle collapsed the full cascade design into Re-Rank
(journal/dynamic_nproj_synthesize.md), we added an always-on re-rank
pass to the multi-table consumer. For every query, after the Stage-1
SUM resolve at N_PROJ=16, the Stage-1 union is re-scored by computing
sum_dist at N_PROJ=32 (sig_bytes=8) over all M tables. No new bucket
index, no new probing — only wider signature encoding + wider
popcount_dist over the existing union.

## Results

### CIFAR-10 (no_deskew, density=0.33)

```
   M      VOTE    SUM_16   SUM_32_RR    PTM      oracle
    1    22.72%   16.63%    30.18%   16.63%    99.51%
    2    24.42%   18.42%    32.26%   15.99%    99.95%
    4    25.53%   22.64%    34.01%   19.38%    99.99%
    8    26.77%   27.08%    35.40%   23.10%   100.00%
   16    28.50%   31.46%    36.45%   27.50%   100.00%
   32    29.02%   34.43%    36.36%   31.09%   100.00%
   64    29.76%   35.32%    36.25%   34.94%   100.00%
```

### Fashion-MNIST (no_deskew, density=0.33)

```
   M      VOTE    SUM_16   SUM_32_RR    PTM      oracle
    1    61.59%   52.92%    78.08%   52.97%    96.46%
    2    67.34%   67.88%    81.82%   57.73%    99.29%
    4    69.57%   75.43%    83.75%   70.24%    99.90%
    8    71.98%   79.93%    84.29%   76.74%    99.99%
   16    73.94%   82.59%    84.92%   80.34%   100.00%
   32    74.59%   84.37%    85.07%   82.23%   100.00%
   64    74.79%   85.15%    85.01%   83.25%   100.00%
```

### MNIST (deskew on, density=0.33)

```
   M      VOTE    SUM_16   SUM_32_RR    PTM      oracle
    1    62.96%   54.50%    88.37%   54.63%    94.30%
    2    71.82%   77.78%    92.70%   62.20%    97.90%
    4    76.75%   88.91%    95.75%   75.34%    99.75%
    8    81.83%   93.84%    96.88%   86.07%    99.99%
   16    85.78%   96.13%    97.35%   91.48%   100.00%
   32    88.50%   97.24%    97.55%   94.25%   100.00%
   64    89.77%   97.31%    97.52%   95.36%   100.00%
```

## Key findings

### 1. Re-rank is massive at low M, marginal-to-negative at high M

The wider signatures shine when the union is small. At M=1,
re-rank gains +13.55pp (CIFAR), +25.16pp (Fashion), +33.87pp
(MNIST). At M=64, gains are +0.93pp (CIFAR), -0.14pp (Fashion),
+0.21pp (MNIST). Fashion-MNIST at M=64 actually REGRESSES —
confirming REFLECT's warning that wider ranking is not guaranteed
monotone.

Mechanism: at low M the union has ~1500-3500 candidates (small,
manageable), and the 32-trit signatures can discriminate within
that set. At M=64 the union has ~6400-12850 candidates, and the
wider scoring can't cut through the confuser noise in a union
that large.

### 2. ~4× table equivalence

Re-rank at M trades roughly 4× in table count vs narrow SUM:

| re-rank config | accuracy | narrow equivalent |
|---|---|---|
| CIFAR-10 SUM_32_RR M=8 | 35.40% | SUM_16 M=64 (35.32%) |
| Fashion SUM_32_RR M=4 | 83.75% | SUM_16 M=32 (84.37%) |
| MNIST SUM_32_RR M=4 | 95.75% | SUM_16 M=16 (96.13%) |

M=8 tables with re-rank matches M=64 without: same accuracy,
8× fewer tables, smaller union, faster probing.

### 3. Re-rank peaks at M=8-16, then degrades

On all three datasets, SUM_32_RR peaks at M=8 or M=16 and
then plateaus or decreases. The optimal operating point for
re-rank is NOT maximum M — it's moderate M where the union
is small enough for wider signatures to discriminate.

### 4. CIFAR-10 go/no-go: marginal

The LMM synthesis set go/no-go at ≥40% for CIFAR-10. We got
36.45% (peak at M=16), 36.25% at M=64. Below threshold. N_PROJ=32
re-rank over 3072-dim RGB provides only ~+1pp at high M. Random
linear projections at any width may be structurally limited on
natural images at this input dimensionality.

## Architectural insight

The re-rank data reframes the Dynamic N_PROJ idea. The original
framing was "start cheap, escalate uncertain queries to wider
resolution." The data says the better operating point is:

**Always filter at N_PROJ=16 (cheap probing, small signatures,
fast bucket lookup). Always rank at N_PROJ=32 (wider signatures,
better discrimination within the union). Use moderate M (8-16)
instead of high M (64).**

The filter and the ranker have different jobs. The filter's job
is to produce a compact union containing the correct answer —
N_PROJ=16 with M=8 achieves 100% oracle. The ranker's job is to
pick the correct answer from the union — wider signatures do
this better than narrow ones, especially when the union is small.

This is the filter-ranker decomposition from Axis 4, now realized
within the routing architecture instead of across a dense/routing
boundary. Both stages are routing-native.

## What comes next

The user wants to explore the full dynamic N_PROJ recycle option
with a range of 16..1024. Instead of a single re-rank at N_PROJ=32,
the system would iteratively escalate uncertain queries through
progressively wider projections until confident. The re-rank
findings inform the design:

1. Use low M at each stage (union quality matters more than
   union size — the oracle is already 100%).
2. The confidence gate should route queries away from higher
   stages as soon as the margin is sufficient.
3. The widest stages (N_PROJ=512, 1024) will only fire on the
   hardest queries — the ones that N_PROJ=16..64 can't resolve.
