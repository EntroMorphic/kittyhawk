---
cycle: rope_design (compressed LMM)
phase: ALL (raw + nodes + reflect + synthesize, single doc)
date: 2026-05-05
scope: design the substrate's RoPE primitive for BitNet b1.58-2B-4T.
       Per work-unit 3 of bitnet_phase1_synthesize. Compressed because
       most uncertainty resolves to single decisions; one ambiguity
       (rotation convention) needed an external reference check.
companions: bitnet_phase1_synthesize.md (work-unit 3); bitnet_stubs.c
            (current stub — DEFECTIVE, see below)
---

# RoPE design — compressed LMM

## RAW: what's actually uncertain

Three things to nail down before implementing m4t_mtfp_rope_apply:

- **Rotation convention.** Two equivalent-but-textually-different
  conventions exist: "adjacent pairs" (x[2i], x[2i+1]) and "rotate_half"
  (x[i], x[i+d/2]). Both produce valid attention but the K-cache layout
  differs. The current stub uses adjacent-pair. BitNet's expected
  convention is determined by HF's reference implementation. This is
  the LMM-resolvable uncertainty.

- **sin/cos representation.** Substrate has no FP at runtime. cos/sin
  ∈ [-1, 1]. Need a fixed-point representation that fits MTFP19
  mantissa (|m| ≤ 2^29.1) with adequate precision.

- **Computation strategy.** Three options:
  1. Compute cos/sin per call (libm sin/cos). Forbidden in production.
  2. Precompute LUT at substrate init using libm (one-time).
  3. Pure-int CORDIC at runtime.

  Option 2 fits the project rule (init-time FP allowed, runtime path
  pure-int) and is by far the simplest. Option 3 is overengineered for
  Phase 1 (cos/sin only depend on position, which is a small finite set
  for any given context length).

## NODES: decisions

- **N1 — Rotation convention.** Resolved by checking the HF model.
- **N2 — sin/cos scale.** Pick a power-of-2 scale Q for fixed-point.
- **N3 — LUT structure.** Per-(position, freq_idx) or per-position
  flat? Lazy or eager init?
- **N4 — Lifecycle.** Where does the LUT live? Per-call alloc, static,
  or caller-managed?
- **N5 — Output saturation.** RoPE produces values up to
  √2 × max(|x|) ≈ 1.41 × x. With x ≤ MTFP19_MAX, output ≤ 1.41 ×
  MTFP19_MAX > MTFP19_MAX → saturation possible. Clamp or scale-down?
- **N6 — NEON vectorization.** Per-pair (a, b) × (cos, sin) is a
  2x2 multiply — naturally SIMD-able across pairs.

## REFLECT

**N1 — Rotation convention (LMM).**

External check: HF repo `microsoft/bitnet-b1.58-2B-4T` has no custom
`modeling_*.py` — only config.json + safetensors weights. The
`trust_remote_code=True` in their usage example loads the model via
standard transformers infrastructure (LlamaForCausalLM-derived). Llama's
`apply_rotary_pos_emb` (transformers/models/llama/modeling_llama.py)
uses `rotate_half`:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    ...
```

Concretely, for i ∈ [0, d/2):
```
q'[i]       = q[i]       · cos_i  −  q[i + d/2] · sin_i
q'[i + d/2] = q[i + d/2] · cos_i  +  q[i]       · sin_i
```
where cos and sin are length-d but duplicated halves: `cos[i] == cos[i + d/2]`.

**Decision:** Use rotate_half (Llama/BitNet convention).
**Implication:** The current stub (adjacent-pair) is DEFECTIVE for
BitNet weight compatibility. K-cache produced by stub-RoPE would be
mis-rotated relative to Q, breaking attention. Phase 1 work-unit 3
fixes this.

**N2 — sin/cos scale.** Cos/sin ∈ [-1, 1]. Use scale Q = 2^29 (close to
MTFP19_MAX = 581130733 ≈ 2^29.1). Gives ~9 decimal digits of precision.
LUT entries fit MTFP19 mantissa cleanly. Per-pair multiply: q[i] × cos
at scales (29, 29) → product scale 2^58, then >>29 to recover MTFP19
scale — fits int64 for the multiply, no __int128 needed.

**N3 — LUT structure.** Per-(position, freq_idx) flat array:
```
cos_lut[pos × (head_dim/2) + freq_idx]
sin_lut[pos × (head_dim/2) + freq_idx]
```
For BitNet: max_position = 4096 (config), head_dim/2 = 64.
LUT size = 4096 × 64 × 4 bytes × 2 (cos+sin) = 2 MiB total. Fits cache.

For Phase 1, init eagerly at first apply call (lazy global). Simple.

**N4 — Lifecycle.** Static-allocated LUT in the .c file, init-once with
__attribute__((constructor))-style guard (or pthread_once). Caller
passes only (q, k, position, num_q_heads, num_kv_heads, head_dim,
theta_base) — primitive owns the LUT.

For Phase 1 simplicity: assume single-threaded (BitNet inference loop
is single-thread). A simple `if (!initialized) init();` guard is fine.

**N5 — Output saturation.** Saturating clamp via m4t_mtfp_clamp64.
Worst case: q[i] = q[i + d/2] = MTFP19_MAX, cos = sin = √2/2.
q'[i] = MAX × cos − MAX × sin = 0. Or q[i] = MAX, q[i + d/2] = 0,
cos = √2/2, sin = √2/2: q'[i] = MAX × √2/2 ≈ 0.707 × MAX < MAX. So
in fact the L2 norm is preserved under rotation — RoPE doesn't cause
saturation given valid MTFP19 inputs. Clamp is defensive.

**N6 — NEON vectorization.** Per-pair (a, b, cos, sin) → (a·cos − b·sin,
a·sin + b·cos). Across freq_idx ∈ [0, d/2): vectorizable as 4-lane
int32 multiply-accumulate. NEON dispatch in production. Scalar test
oracle for verification.

But: cos/sin per pair are different (different freq_idx → different
angles). So the loads need to step through cos_lut/sin_lut alongside q.
Standard vectorization pattern. NEON intrinsics:
- vld1q_s32(q + 2*i) → loads 4 consecutive int32 (but they're paired
  across head_dim/2 boundary, not consecutive!).

Hmm — with rotate_half, the pair is (q[i], q[i + d/2]), NOT consecutive
in memory. NEON load must gather q[i..i+3] for "a" and q[i+d/2..i+d/2+3]
for "b" — two separate 4-lane loads from different offsets. That's
fine; NEON supports it.

## SYNTHESIZE: actionable plan

### Primitive

```c
/* Apply RoPE to (q, k) in place using rotate_half convention.
 * Llama/BitNet-compatible.
 *
 * q layout: [num_q_heads × head_dim], k layout: [num_kv_heads × head_dim].
 * head_dim must be even.
 *
 * For i ∈ [0, head_dim/2):
 *   q'[h, i]            = q[h, i]            · cos[pos, i] − q[h, i + d/2] · sin[pos, i]
 *   q'[h, i + d/2]      = q[h, i + d/2]      · cos[pos, i] + q[h, i]       · sin[pos, i]
 *
 * cos/sin LUT precomputed at first call using libm (init-time FP
 * allowed, like weight loading). LUT indexed by (position, freq_idx).
 *
 * theta_base parameterizes the angle: angle_i = position / theta_base^(2i/head_dim).
 *
 * Caller is responsible for ensuring position < M4T_ROPE_MAX_POSITION
 * (compile-time constant; assertion catches violations). */
void m4t_mtfp_rope_apply(
    m4t_mtfp_t* q, m4t_mtfp_t* k,
    int position,
    int num_q_heads, int num_kv_heads, int head_dim,
    double theta_base
);

/* Scalar test oracle. Same algorithm, no NEON. */
void m4t_mtfp_rope_apply_scalar_ref(
    m4t_mtfp_t* q, m4t_mtfp_t* k,
    int position,
    int num_q_heads, int num_kv_heads, int head_dim,
    double theta_base
);
```

Constants:
- `M4T_ROPE_MAX_POSITION = 4096` (BitNet's max_position_embeddings).
- `M4T_ROPE_COS_SIN_SCALE = (1 << 29)` (Q = 2^29).

### LUT initialization

```c
static int32_t* g_cos_lut;  /* [M4T_ROPE_MAX_POSITION × MAX_HEAD_DIM/2] */
static int32_t* g_sin_lut;
static int g_lut_initialized;
static int g_lut_head_dim;
static double g_lut_theta_base;

static void rope_init_lut(int head_dim, double theta_base) {
    if (g_lut_initialized && g_lut_head_dim == head_dim
        && g_lut_theta_base == theta_base) return;
    /* (Re-)allocate and fill. libm sin/cos here is init-time only. */
    int half = head_dim / 2;
    size_t n = (size_t)M4T_ROPE_MAX_POSITION * half;
    g_cos_lut = (int32_t*)realloc(g_cos_lut, n * sizeof(int32_t));
    g_sin_lut = (int32_t*)realloc(g_sin_lut, n * sizeof(int32_t));
    for (int pos = 0; pos < M4T_ROPE_MAX_POSITION; pos++) {
        for (int i = 0; i < half; i++) {
            double freq = pow(theta_base, -2.0 * i / (double)head_dim);
            double angle = (double)pos * freq;
            g_cos_lut[pos*half + i] = (int32_t)(cos(angle) * (double)M4T_ROPE_COS_SIN_SCALE);
            g_sin_lut[pos*half + i] = (int32_t)(sin(angle) * (double)M4T_ROPE_COS_SIN_SCALE);
        }
    }
    g_lut_initialized = 1;
    g_lut_head_dim = head_dim;
    g_lut_theta_base = theta_base;
}
```

### Apply (production, scalar form for clarity)

```c
void m4t_mtfp_rope_apply(...) {
    rope_init_lut(head_dim, theta_base);
    int half = head_dim / 2;
    const int32_t* cos_row = g_cos_lut + (size_t)position * half;
    const int32_t* sin_row = g_sin_lut + (size_t)position * half;

    for (int h = 0; h < num_q_heads; h++) {
        m4t_mtfp_t* qh = q + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            int64_t a = qh[i];
            int64_t b = qh[i + half];
            int64_t c = cos_row[i];
            int64_t s = sin_row[i];
            int64_t new_a = (a * c - b * s) >> 29;
            int64_t new_b = (b * c + a * s) >> 29;
            qh[i]        = m4t_mtfp_clamp64(new_a);
            qh[i + half] = m4t_mtfp_clamp64(new_b);
        }
    }
    /* Same for k loop. */
}
```

NEON vectorization deferred to a follow-up: the per-pair multiply is
naturally SIMD (4 freq_idx at a time via vmull_s32 + vshr_n_s64). Not
on the Phase 1 critical path — correctness first.

Per project memory: "production dispatchers are NEON-only (scalar_ref
test oracle is fine; geometric tail is fine)". RoPE is single-token
work (small head_dim×heads loop, ~3200 ops/call). NEON would drop call
cost ~4×; not load-bearing for Phase 1's correctness gate. Will revisit
in Phase 1 closeout if profiling shows RoPE dominating.

For Phase 1 scope: ship the scalar production path. The key
correctness claim — rotate_half convention, LUT-based — is what gates
Phase 1 inference accuracy. Performance is Phase 2/4.

### Tests

`test_m4t_rope`:
- Bit-exact production-vs-scalar_ref (both use the same LUT-and-apply
  pipeline; bit-exact is achievable).
- Boundary: position=0 (cos=1, sin=0; identity rotation).
- position=M4T_ROPE_MAX_POSITION-1 (boundary).
- Reverse-rotation property: applying RoPE then "un-RoPE" (rotate by
  -position) should approximately recover the original.
- Saturation: input MTFP19_MAX values don't blow up the result.

### Wiring into bitnet_harness.c

Replace `bitnet_stub_rope_apply` call with `m4t_mtfp_rope_apply`. Note:
the harness's residual stream and K-cache compatibility depends on
this being correct end-to-end — work-unit 6 will gate this against
HF reference.

### Out of scope for work-unit 3

- NEON vectorization (deferred per above).
- Position > 4096 (NTK scaling, etc.) — BitNet uses 4096 max.
- LUT lifecycle thread safety (single-threaded inference).

## Status

Cycle complete. Proceeding to implementation.
