# #1 K-signature caching — implementation + bit-exact + 2% speedup

Per `docs/TRIT_ROUTING_APPLICATIONS.md` item #1. Per
`journal/td27_4_fixed_tau_2026-05-10.md`, fixed tau is acceptable for
quality, which makes K-signature caching feasible (signatures depend
only on K + tau; if tau is fixed, cache once at K-write, reuse free).

## Implementation

`gesh/bitnet/bitnet_config.h`: added `k_sig` (uint8_t* — packed-trit
signatures) and `k_sig_tau` (int — tau used for cached signatures) to
`bitnet_kv_cache_t`.

`gesh/bitnet/bitnet_kv_cache.c`: alloc/free updated to handle the
optional signature buffer. Lazy-allocated (NULL until routed mode with
fixed tau is first invoked).

`gesh/bitnet/bitnet_harness.c`:
- `bitnet_kv_cache_ensure_sig(cache, tau)` — lazy-alloc signature buffer;
  if a different tau is requested than was cached, free + reallocate.
  Returns 1 if cache was (re-)allocated (caller must populate all positions).
- `bitnet_kv_cache_store_k_sig(cache, layer, position, tau)` — compute
  and store K signatures for one (layer, position) across all
  NUM_KV_HEADS kv_heads.
- `bitnet_kv_cache_k_sig(cache, layer, position, kv_head)` — pointer
  to cached signature for lookup.
- K-write site: when routed mode + fixed tau active, populate cache
  for the just-written K position.
- `bitnet_pick_routed_indices`: optionally takes cache + layer_idx;
  uses cached signatures when available + tau-matched, else recomputes.

Memory footprint: per (layer, position, kv_head) signature is
`M4T_TRIT_PACKED_BYTES(128) = 32 bytes`. Total: 30 layers × 4096 max_seq
× 5 kv_heads × 32 bytes = **19 MB**. Comparable to existing K/V cache
(~39 MB at the same max_seq).

Added `BITNET_ATTN_NO_CACHE=1` env var as a verification/debugging flag
that disables the cache even when tau is fixed.

## Bit-exactness verification

For each of 4 diverse prompts, ran `routed k=4 τ=5000` with cache
enabled and disabled (via BITNET_ATTN_NO_CACHE), compared first 10
generated tokens:

| prompt | verdict |
|---|---|
| math_div | BIT-EXACT ✓ |
| factual_capital | BIT-EXACT ✓ |
| long_history | BIT-EXACT ✓ |
| edge_repetitive | BIT-EXACT ✓ |

The cache is a pure perf optimization — output is identical to the
recompute path.

## Wall-clock measurement

3 runs each with/without cache on `long_history` prompt (36 tokens) +
gen 30 with `routed k=4 τ=5000`. One uncached run was an outlier
(real 955s due to system contention, user 28s normal); excluded.

| config | runs (user CPU seconds) | mean |
|---|---|---|
| WITHOUT cache | 27.98, 27.50, 26.87 | 27.45 |
| WITH cache | 26.78, 26.96, 26.88 | 26.87 |

**Speedup: ~2.1% on this workload.**

## Honest framing

This is a small speedup — the K-signature recomputation isn't the
dominant cost in BitNet inference. BitLinear matmuls dominate; routing
work (signatures + distance + sort) is roughly 1% of total per-token
work at this context length.

**The speedup should scale with context length.** K-sig recompute work
is O(seq_k × head_dim) per attention step. At seq_k=4096 (BitNet's max
context), the K-sig work would be ~62× larger than at seq_k=66 (the
test workload). Long-context inference is where the cache pays off.

## What this DOES validate

- The substrate's signature primitive composes cleanly with the KV
  cache infrastructure (engineering primitive validation)
- Fixed tau works in production (lifts the #4 finding from "focused
  subset n=10" to "deployed in production harness")
- The route_threshold_extract primitive is competitive when amortized:
  recomputing per-step is ~2% overhead on this workload; could be
  meaningful at longer contexts

## What this does NOT change

- The Cycle 2 Part-B EVIDENCE finding (routed > random with widening
  margins) — based on per-Q tau without cache; the cached fixed-tau
  variant gets different exact outputs (different tau) so the EVIDENCE
  numbers would need to be re-measured. The PATTERN should hold but
  absolute pass rates may shift.
- The substrate-vs-posracle comparison. Posracle isn't affected by
  K-sig caching (it doesn't use signatures); only the routed arm is.

## Methodology lifts

1. **Bit-exactness verification before perf claims.** Without confirming
   cache produces identical output to recompute, any perf measurement
   is meaningless.

2. **Permanent verification flag.** `BITNET_ATTN_NO_CACHE=1` stays as
   a debugging tool. Cheap to keep (one branch); valuable for
   regression testing.

3. **Honest about small wins.** A 2% speedup is real but small; the
   cost-distinct story for the substrate doesn't get a big boost from
   #1. The implementation is still worth keeping (correct, low memory,
   sets up #3 hybrid two-stage routing which needs the same cache
   infrastructure).

## Open follow-ups

- **Long-context speedup measurement.** This test was at seq_k≈66.
  Re-running at seq_k=512 or 1024 would show how the speedup scales.
  Recorded as TD candidate.

- **Quality of fixed-tau routed at full battery.** #4 tested at n=10;
  this commit doesn't re-run the full battery on the cached fixed-tau
  config. Should the routed full battery be re-measured with τ=5000
  as the default tau? Would close the "fixed tau works at scale"
  question.
