/*
 * gesh/bitnet/bitnet_harness.c — single-block forward-pass harness for
 * BitNet b1.58-2B-4T inference on the m4t substrate. Per work-unit 1
 * of the bitnet_phase1 LMM cycle.
 *
 * This file is the SKELETON committed at the start of work-unit 1.
 * It compiles cleanly, allocates scratch, lays out the forward pass
 * structure, and stubs out the parts that need (a) actual weights
 * loaded from disk and (b) per-layer activation dumping. Subsequent
 * work-unit-1 commits fill those in.
 *
 * Execution model:
 *   build/gesh/bitnet/bitnet_harness <weights_blob.bin> <input_token_id>
 *
 * Output:
 *   per-layer activations dumped to stdout in a documented binary
 *   format (work-unit 1 fills this in alongside the Python comparison
 *   driver).
 */

#include "bitnet_config.h"
#include "bitnet_stubs.h"
#include "bitnet_weights.h"

#include "m4t_ternary_matmul.h"
#include "m4t_mtfp.h"
#include "m4t_route.h"        /* Cycle 2 Phase 2.4: substrate-routed sparse attention */
#include "m4t_trit_pack.h"    /* M4T_TRIT_PACKED_BYTES for signature sizing */
#include "m4t_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

/* ── Cycle 2 (Part-B sparse attention experiment) — runtime-selectable
 *    attention mode. Default DENSE = current production behavior; sparse
 *    arms (RANDOM, ROUTED, ORACLE) exist for the experiment per
 *    journal/cycle2_design.md and journal/partB_experiments_synth.md.
 *
 *    Set BITNET_ATTN_MODE ∈ {dense,random,routed,oracle} (case-insensitive)
 *    Set BITNET_ATTN_K ∈ positive int (default = head_dim, i.e. equivalent
 *    to dense for sparse modes).
 *
 *    When env vars are unset: bit-exact match to the production path. */
typedef enum {
    BITNET_ATTN_DENSE = 0,
    BITNET_ATTN_RANDOM,
    BITNET_ATTN_ROUTED,
    BITNET_ATTN_ORACLE,
    BITNET_ATTN_POSRACLE,  /* TD-27 H2 test: top-k by SIGNED Q·K (positives win)
                            * vs oracle's top-k by |Q·K|. If posracle ≈ routed,
                            * the routed-vs-oracle gap is explained by oracle
                            * "wasting" budget on high-|negative-score| positions
                            * that softmax suppresses anyway. */
} bitnet_attn_mode_t;

static bitnet_attn_mode_t g_attn_mode = BITNET_ATTN_DENSE;
static int                g_attn_k    = BITNET_HEAD_DIM;  /* trajectory knob */
static __attribute__((unused)) unsigned int g_attn_rng = 0xC0FFEE01u;  /* random-arm seed (used in Phase 2.2) */
static int64_t g_attn_fixed_tau = 0;  /* TD-27 follow-up #4: 0 = use per-Q 1/3-quantile;
                                       * positive = use this fixed tau in routed signature extraction.
                                       * Tests whether per-Q adaptiveness is load-bearing. */

static const char* bitnet_attn_mode_name(bitnet_attn_mode_t m) {
    switch (m) {
        case BITNET_ATTN_DENSE:    return "dense";
        case BITNET_ATTN_RANDOM:   return "random";
        case BITNET_ATTN_ROUTED:   return "routed";
        case BITNET_ATTN_ORACLE:   return "oracle";
        case BITNET_ATTN_POSRACLE: return "posracle";
    }
    return "unknown";
}

static void bitnet_attn_mode_init_from_env(void) {
    const char* m = getenv("BITNET_ATTN_MODE");
    if (m) {
        if      (!strcasecmp(m, "dense"))    g_attn_mode = BITNET_ATTN_DENSE;
        else if (!strcasecmp(m, "random"))   g_attn_mode = BITNET_ATTN_RANDOM;
        else if (!strcasecmp(m, "routed"))   g_attn_mode = BITNET_ATTN_ROUTED;
        else if (!strcasecmp(m, "oracle"))   g_attn_mode = BITNET_ATTN_ORACLE;
        else if (!strcasecmp(m, "posracle")) g_attn_mode = BITNET_ATTN_POSRACLE;
        else {
            fprintf(stderr, "[harness] unknown BITNET_ATTN_MODE=%s, using dense\n", m);
        }
    }
    const char* k = getenv("BITNET_ATTN_K");
    if (k) {
        int v = atoi(k);
        if (v > 0 && v <= BITNET_HEAD_DIM * 32) g_attn_k = v;
        else fprintf(stderr, "[harness] bad BITNET_ATTN_K=%s, using %d\n", k, g_attn_k);
    }
    const char* t = getenv("BITNET_ATTN_TAU");
    if (t) {
        long v = atol(t);
        if (v > 0) g_attn_fixed_tau = (int64_t)v;
        else fprintf(stderr, "[harness] bad BITNET_ATTN_TAU=%s, ignoring\n", t);
    }
    if (g_attn_mode != BITNET_ATTN_DENSE) {
        fprintf(stderr, "[harness] sparse attention mode = %s, k = %d",
                bitnet_attn_mode_name(g_attn_mode), g_attn_k);
        if (g_attn_fixed_tau > 0)
            fprintf(stderr, ", fixed_tau = %lld", (long long)g_attn_fixed_tau);
        fprintf(stderr, "\n");
    }
}

/* ── Phase 2 work-unit 1: bx-aware activation flow constants. ────────────
 * Target bx for normal (linear-magnitude) activations through the
 * network. Picked to give MTFP19_MAX/3^14 ≈ 35 of headroom — comfortable
 * for BitNet's value range (0.5–10 typical, 30–100 worst-case).
 *
 * The FFN intermediate path (between gate_proj and ffn_sub_norm) carries
 * SQUARED magnitudes (relu²(gate) × up). For gate_real ≤ 6 typical, the
 * squared product reaches ~200; bx=14 saturates at 35. We use a smaller
 * FFN_BX that gives wider range (real max 3^(29-FFN_BX) at MTFP19_MAX). */
#define BITNET_ACT_BX 8       /* MTFP19_MAX/3^8 ≈ 88573. Tuning history:
                               *
                               * Phase 2 wu1.4 red-team (pre-RMSNorm-fix):
                               * ACT_BX=10 saturated 0.5% of block_output
                               * cells from L4 onward. Lowered to 8 →
                               * "zero saturation across 30 layers" was
                               * the claim at the time.
                               *
                               * Post-RMSNorm-fix (commit 4d4c917): the
                               * "zero saturation" claim NO LONGER HOLDS.
                               * The bug was magnitude-collapsing post_attn_norm
                               * outputs by 6.5×; fixing it lets correctly-larger
                               * residuals propagate, and ~209 cells (0.034%
                               * of the 30L × 8pos × 2560-cell residual
                               * stream sweep) now saturate in late layers
                               * (L24-L29). See journal/saturation_audit_2026-05-09.md
                               * and journal/act_bx_sweep_2026-05-09.md.
                               *
                               * The ACT_BX sweep ∈ {6, 7, 8} (TD-19) showed:
                               * - BX=8 keeps 209 saturations BUT 8/8 prompts
                               *   coherent and math correct (12+7=19).
                               * - BX=7 → 0 saturations BUT one prompt loops.
                               * - BX=6 → 0 saturations BUT math regresses
                               *   (12+7=20; 1-trit precision loss flips argmax
                               *   on tight integer-token logit margins).
                               *
                               * Conscious tradeoff: keep BX=8. The ~0.034%
                               * residual-stream saturation is absorbed by
                               * downstream RMSNorm; coherence + correctness
                               * are the load-bearing metrics, not saturation
                               * count. End-to-end battery (8 + 24 prompts)
                               * confirms the substrate produces coherent
                               * English on factual / definitional / narrative
                               * / long-context tasks at this setting. */
#define BITNET_FFN_BX 6       /* MTFP19_MAX/3^6 ≈ 797K — gate, up.
                               * Sweep [4,6,8,10,12]: Pearson peaks at 6
                               * (0.6857). FFN_BX=8 ranks Paris higher
                               * (#8 vs #41) but loses overall correlation
                               * — picking 6 favors broader signal. */
#define BITNET_GATE_ACT_BX 1  /* MTFP19_MAX/3^1 ≈ 194M headroom for
                               * gate²×up products. Tuning history:
                               *
                               * Original choice (pre-RMSNorm-fix): 2,
                               * "Pearson invariant for [1,6]; picking 2
                               * to preserve ~17 fractional bits below 1.0."
                               * That sweep was on the buggy substrate
                               * (predates 4d4c917).
                               *
                               * Post-fix re-sweep (TD-20 remediation,
                               * 2026-05-10): swept GATE_ACT_BX ∈ {1, 2,
                               * 4, 6} on the 5 substrate-specific failure
                               * prompts. BX=1 went from baseline 1/5 to
                               * 5/5. Full 24-prompt battery confirmed:
                               * 5 prompts upgrade from loop/wrong → coherent
                               * (reason_word "2 hours" not "8", code_comment
                               * yields real def sort_array, json_format
                               * yields actual JSON), 1 regression
                               * (factual_hamlet gives a "Hint" instead
                               * of "Shakespeare"). Strict pass rate 15/24
                               * → ~19/24. Per journal/hp_sweep_2026-05-10.md. */

/* ── Cycle 2 sparse-attention helpers (experimental; not on production
 *    path unless BITNET_ATTN_MODE != dense). Per journal/cycle2_design.md. */

/* xorshift32 for deterministic random index selection (random arm). */
static inline unsigned int bitnet_xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

/* Pick k unique indices from [0, n) into out[]. Modifies *rng_state.
 * Uses partial Fisher-Yates: O(n) memory + time, fine for our n ≤ 4096. */
static void bitnet_pick_random_indices(int* out, int k, int n, unsigned int* rng_state) {
    if (k >= n) {
        for (int i = 0; i < n; i++) out[i] = i;
        return;
    }
    int* pool = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) pool[i] = i;
    for (int i = 0; i < k; i++) {
        unsigned int r = bitnet_xorshift32(rng_state);
        int j = i + (int)(r % (unsigned int)(n - i));
        int tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
        out[i] = pool[i];
    }
    free(pool);
}

/* qsort comparator for int64 ascending. */
static int bitnet_cmp_i64_asc(const void* a, const void* b) {
    int64_t av = *(const int64_t*)a, bv = *(const int64_t*)b;
    return (av > bv) - (av < bv);
}

/* Compute a percentile-based tau over |values[]| to feed
 * m4t_route_threshold_extract such that all three trit states are
 * realized non-trivially (per §18 input-class contract). Uses the
 * 1/3-quantile of absolute values: ~33% of trits will be 0, ~33%
 * will be +1, ~33% will be -1 in the resulting signature. Returns
 * tau in the same units as |values|. */
static int64_t bitnet_routed_pick_tau(const m4t_mtfp_t* values, int n) {
    int64_t* abs_vals = (int64_t*)malloc((size_t)n * sizeof(int64_t));
    for (int i = 0; i < n; i++) {
        int32_t v = values[i];
        abs_vals[i] = (int64_t)(v < 0 ? -v : v);
    }
    qsort(abs_vals, (size_t)n, sizeof(int64_t), bitnet_cmp_i64_asc);
    int64_t tau = abs_vals[n / 3];
    if (tau < 1) tau = 1;  /* floor at 1 to avoid all-trit-zero degenerate */
    free(abs_vals);
    return tau;
}

/* Pick the k positions in [0, seq_k) whose K signatures are CLOSEST to
 * the Q signature in popcount (Hamming-on-trits) distance. Uses
 * m4t_route_threshold_extract to build signatures and
 * m4t_route_distance_batch for distance computation. Manually sorts
 * distances for top-k selection (route_topk_abs has T ≤ 64 limit; we
 * want it for arbitrary seq_k).
 *
 * out[0..k) = chosen position indices. */
static int bitnet_pick_routed_compare(const void* a, const void* b) {
    /* Sort ascending by second-element-of-pair (distance). */
    int32_t av = ((const int32_t*)a)[1];
    int32_t bv = ((const int32_t*)b)[1];
    return (av > bv) - (av < bv);
}

static void bitnet_pick_routed_indices(
    int* out, int k, int seq_k,
    const m4t_mtfp_t* qh,
    const m4t_mtfp_t* k_cache, size_t k_row_size, int kv_head,
    int head_dim)
{
    int sig_bytes = M4T_TRIT_PACKED_BYTES(head_dim);

    /* Choose tau. Default: per-Q 1/3-quantile of |Q| (ensures all three trit
     * states realize per §18 input-class contract). Override via
     * BITNET_ATTN_TAU env var for the TD-27 follow-up #4 experiment
     * (does per-Q adaptiveness add anything over a fixed tau?). */
    int64_t tau = g_attn_fixed_tau > 0 ? g_attn_fixed_tau
                                        : bitnet_routed_pick_tau(qh, head_dim);

    /* Q signature. */
    int64_t* q_i64 = (int64_t*)malloc((size_t)head_dim * sizeof(int64_t));
    for (int d = 0; d < head_dim; d++) q_i64[d] = (int64_t)qh[d];
    uint8_t* q_sig = (uint8_t*)malloc((size_t)sig_bytes);
    m4t_route_threshold_extract(q_sig, q_i64, tau, head_dim);

    /* K signatures (one per position). */
    uint8_t* k_sigs = (uint8_t*)malloc((size_t)seq_k * sig_bytes);
    int64_t* tmp_i64 = q_i64;  /* reuse */
    for (int t = 0; t < seq_k; t++) {
        const m4t_mtfp_t* kh = k_cache + (size_t)t * k_row_size + (size_t)kv_head * head_dim;
        for (int d = 0; d < head_dim; d++) tmp_i64[d] = (int64_t)kh[d];
        m4t_route_threshold_extract(k_sigs + (size_t)t * sig_bytes,
                                     tmp_i64, tau, head_dim);
    }

    /* Distances. mask = all 0xFF (consider all positions). */
    uint8_t* mask = (uint8_t*)malloc((size_t)sig_bytes);
    memset(mask, 0xFF, (size_t)sig_bytes);
    int32_t* dists = (int32_t*)malloc((size_t)seq_k * sizeof(int32_t));
    m4t_route_distance_batch(dists, q_sig, k_sigs, mask, seq_k, head_dim);

    /* Top-k by SMALLEST distance — sort (idx, dist) pairs ascending. */
    int32_t* pairs = (int32_t*)malloc((size_t)seq_k * 2 * sizeof(int32_t));
    for (int t = 0; t < seq_k; t++) {
        pairs[2*t + 0] = (int32_t)t;
        pairs[2*t + 1] = dists[t];
    }
    qsort(pairs, (size_t)seq_k, 2 * sizeof(int32_t), bitnet_pick_routed_compare);
    for (int i = 0; i < k; i++) out[i] = (int)pairs[2*i + 0];

    free(pairs); free(dists); free(mask);
    free(k_sigs); free(q_sig); free(q_i64);
}

/* Pick top-k indices from scores[0..n) by absolute value (descending).
 * Used by ORACLE arm — picks the k positions with highest |Q·K| score
 * after dense scores are computed. O(n log n) sort; fine for n ≤ 4096. */
static int bitnet_pick_oracle_compare(const void* a, const void* b) {
    /* Sort descending by |second-element-of-pair|. Pair: (index, abs_score). */
    int64_t av = ((const int64_t*)a)[1];
    int64_t bv = ((const int64_t*)b)[1];
    if (av < bv) return  1;
    if (av > bv) return -1;
    return 0;
}

static void bitnet_pick_oracle_topk(int* out, int k, int n, const int64_t* scores) {
    if (k >= n) {
        for (int i = 0; i < n; i++) out[i] = i;
        return;
    }
    int64_t* pairs = (int64_t*)malloc((size_t)n * 2 * sizeof(int64_t));
    for (int i = 0; i < n; i++) {
        pairs[2*i + 0] = (int64_t)i;
        pairs[2*i + 1] = scores[i] < 0 ? -scores[i] : scores[i];
    }
    qsort(pairs, (size_t)n, 2 * sizeof(int64_t), bitnet_pick_oracle_compare);
    for (int i = 0; i < k; i++) out[i] = (int)pairs[2*i + 0];
    free(pairs);
}

/* TD-27 H2 test: top-k by SIGNED score (descending — highest positive wins).
 * If oracle's "wastes budget on high-|negative-score| positions softmax
 * suppresses anyway" hypothesis is right, this should approach routed quality. */
static int bitnet_pick_posracle_compare(const void* a, const void* b) {
    /* Sort descending by SIGNED score (no abs). Negatives sink to bottom. */
    int64_t av = ((const int64_t*)a)[1];
    int64_t bv = ((const int64_t*)b)[1];
    if (av < bv) return  1;
    if (av > bv) return -1;
    return 0;
}

static void bitnet_pick_posracle_topk(int* out, int k, int n, const int64_t* scores) {
    if (k >= n) {
        for (int i = 0; i < n; i++) out[i] = i;
        return;
    }
    int64_t* pairs = (int64_t*)malloc((size_t)n * 2 * sizeof(int64_t));
    for (int i = 0; i < n; i++) {
        pairs[2*i + 0] = (int64_t)i;
        pairs[2*i + 1] = scores[i];  /* SIGNED — negatives sink to bottom in sort */
    }
    qsort(pairs, (size_t)n, 2 * sizeof(int64_t), bitnet_pick_posracle_compare);
    for (int i = 0; i < k; i++) out[i] = (int)pairs[2*i + 0];
    free(pairs);
}

/* Sparse attn_v_combine: out[d] = clamp(Σ_i weights[i] · V[indices[i]][d] >> shift).
 * Scalar implementation — this path is experimental (Cycle 2 sparse arms);
 * NOT on the production hot path when BITNET_ATTN_MODE is unset/dense.
 * The dense path remains m4t_mtfp_attn_v_combine (NEON-routed). */
static void bitnet_sparse_attn_v_combine(
    m4t_mtfp_t* out, int shift,
    const m4t_mtfp_t* weights,
    const m4t_mtfp_t* V_base, size_t row_size,
    int k, int head_dim,
    const int* indices)
{
    int64_t* acc = (int64_t*)calloc((size_t)head_dim, sizeof(int64_t));
    for (int i = 0; i < k; i++) {
        int t = indices[i];
        const m4t_mtfp_t* V_t = V_base + (size_t)t * row_size;
        int64_t w = (int64_t)weights[i];
        for (int d = 0; d < head_dim; d++) {
            acc[d] += w * (int64_t)V_t[d];
        }
    }
    for (int d = 0; d < head_dim; d++) {
        int64_t r = acc[d] >> shift;
        if (r >  M4T_MTFP_MAX_VAL) r =  M4T_MTFP_MAX_VAL;
        if (r < -M4T_MTFP_MAX_VAL) r = -M4T_MTFP_MAX_VAL;
        out[d] = (m4t_mtfp_t)r;
    }
    free(acc);
}

/* ── BitLinear scale composition (work-unit 5) ───────────────────────────
 *
 * BitLinear forward:  y = (matmul_int_out) × α × absmax / 127
 * where:
 *   - α is the BitLinear's per-tensor scale, stored as MTFP19
 *     (mantissa × 3^(-block_exp)).
 *   - absmax is the activation's per-token absmax (from A8 quantize).
 *   - matmul_int_out is the raw int-mantissa output from
 *     m4t_ternary_5in8_matmul_bt.
 *
 * Composing: y = raw × (α_m / 3^bx) × absmax / 127
 *              = raw × (α_m × absmax) / (127 × 3^bx)
 *
 * num = α_m × absmax, den = 127 × 3^bx.
 *
 * Constraint: bx ≤ ~38 to keep den ≤ int64 max. For BitNet b1.58-2B-4T
 * α magnitudes (range ~1e-3 to ~1), typical block_exp ≤ 25. Documented.
 *
 * If α_mantissa == 0 (convert_weights.py's sentinel for "no α
 * available"), the apply is a no-op (skip vec_scale; output remains
 * raw matmul, useful for skeleton-mode debugging). */

/* pow3_int: replaced by m4t_mtfp_*'s internal pow3_i64 since work-unit-1
 * Phase 2 (bx-aware primitives consume the bx-shift inline). */

__attribute__((unused))
static void bitnet_apply_bitlinear_scale(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    const m4t_mtfp_t* alpha_ptr, int alpha_block_exp,
    m4t_mtfp_t absmax, int n)
{
    /* bx-aware path (Phase 2 work-unit 1): produce output at
     * BITNET_ACT_BX assuming the input x_norm was at BITNET_ACT_BX. */
    m4t_mtfp_bitlinear_scale_bx(y, x, alpha_ptr, alpha_block_exp,
                                 absmax,
                                 /*x_bx=*/BITNET_ACT_BX,
                                 /*target_bx=*/BITNET_ACT_BX, n);
}

/* Bit-faithful BitLinear: int32 × ternary matmul → int64 raw → α scale apply.
 * No a8 quantization; matches HF's bf16-everywhere precision (modulo MTFP19
 * vs bf16 storage of x and α). Uses the 4-in-8 packed weights produced by
 * bitnet_weights.c's repack pass. */
static void bitnet_bitlinear_no_a8(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    const uint8_t* W_packed_4in8,
    const m4t_mtfp_t* alpha_ptr, int alpha_block_exp,
    int x_bx, int target_bx,
    int K, int N)
{
    /* Stack scratch for int64 raw output. Largest N is INTERMEDIATE = 6912
     * → 55 KB. Comfortable on macOS' 8 MB main stack. */
    int64_t y_raw[BITNET_INTERMEDIATE_SIZE];
    assert(N <= BITNET_INTERMEDIATE_SIZE);
    m4t_mtfp_ternary_matmul_bt_route_i64(y_raw, x, W_packed_4in8, /*M=*/1, K, N);
    m4t_mtfp_bitlinear_scale_no_a8_bx(y, y_raw, alpha_ptr, alpha_block_exp,
                                       x_bx, target_bx, N);
}

/* ── Scratch alloc / free ────────────────────────────────────────── */

void bitnet_block_scratch_alloc(bitnet_block_scratch_t* s) {
    /* Single-token forward pass: each [HIDDEN] or [INTERMEDIATE] buffer
     * holds one row. Multi-token prefill widens these (work-unit 7+). */
    s->x              = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->residual       = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->x_norm         = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->x_norm_input   = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->q              = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->q_pre_rope     = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->k              = calloc(BITNET_KV_PROJ_DIM,         sizeof(m4t_mtfp_t));
    s->k_pre_rope     = calloc(BITNET_KV_PROJ_DIM,         sizeof(m4t_mtfp_t));
    s->v              = calloc(BITNET_KV_PROJ_DIM,         sizeof(m4t_mtfp_t));
    /* Single-token: attn_scores is just [num_heads × 1 × 1] = 20 cells.
     * Multi-token: would scale with seq_len. */
    s->attn_scores    = calloc(BITNET_NUM_ATTENTION_HEADS, sizeof(m4t_mtfp_t));
    s->attn_output    = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->attn_sub_norm  = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->gate           = calloc(BITNET_INTERMEDIATE_SIZE,   sizeof(m4t_mtfp_t));
    s->up             = calloc(BITNET_INTERMEDIATE_SIZE,   sizeof(m4t_mtfp_t));
    s->gate_act       = calloc(BITNET_INTERMEDIATE_SIZE,   sizeof(m4t_mtfp_t));
    s->ffn_sub_norm   = calloc(BITNET_INTERMEDIATE_SIZE,   sizeof(m4t_mtfp_t));
    s->q_int8         = calloc(BITNET_INTERMEDIATE_SIZE,   sizeof(int8_t));
    s->q_absmax       = 0;
}

void bitnet_block_scratch_free(bitnet_block_scratch_t* s) {
    free(s->x); free(s->residual); free(s->x_norm); free(s->x_norm_input);
    free(s->q); free(s->q_pre_rope);
    free(s->k); free(s->k_pre_rope);
    free(s->v);
    free(s->attn_scores); free(s->attn_output); free(s->attn_sub_norm);
    free(s->gate); free(s->up); free(s->gate_act); free(s->ffn_sub_norm);
    free(s->q_int8);
    memset(s, 0, sizeof(*s));
}

/* ── Forward pass through one transformer block ──────────────────── */

/*
 * Input: x[HIDDEN] (the token embedding for the first block; the
 * previous block's output for layers > 0).
 * Output: x[HIDDEN] (overwritten in place).
 *
 * Per the architecture (verified in bitnet_phase1_o1_findings.md):
 *
 *   residual = x
 *   x = input_layernorm(x)
 *   x = attention(x)            ← contains attn_sub_norm
 *   x = residual + x
 *   residual = x
 *   x = post_attention_layernorm(x)
 *   x = ffn(x)                  ← contains ffn_sub_norm
 *   x = residual + x
 */
void bitnet_forward_block(
    m4t_mtfp_t* x_io,
    const bitnet_layer_weights_t* w,
    bitnet_block_scratch_t* s,
    bitnet_kv_cache_t* cache,
    int layer_idx,
    int position)
{
    /* For now: x_io aliased into s->x at entry, copied back at exit.
     * Future cleanup may eliminate the copy. */
    memcpy(s->x, x_io, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));

    /* ── Attention sub-block ──────────────────────────────────────── */

    /* residual = x */
    memcpy(s->residual, s->x, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));

    /* x_norm = input_layernorm(x). bx-aware: x at BITNET_ACT_BX,
     * γ at its loaded bx, output at BITNET_ACT_BX. */
    m4t_mtfp_rmsnorm_bx(s->x_norm, s->x, w->gamma_input_norm,
                        /*x_bx=*/BITNET_ACT_BX,
                        /*gamma_bx=*/w->gamma_input_norm_block_exp,
                        /*target_bx=*/BITNET_ACT_BX,
                        /*eps_mantissa=*/1, BITNET_HIDDEN_SIZE);
    /* Snapshot for the dump — s->x_norm gets overwritten by
     * post_attention_layernorm later in the block. */
    memcpy(s->x_norm_input, s->x_norm,
           BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
    /* TODO(work-unit 1.5): empirically determine eps_mtfp19. The HF
     * eps is 1e-5 (FP); the substrate-side equivalent depends on
     * what scale the activations land in. Capture x's magnitude
     * during first run and pick eps to land at the same ratio. */

    /* QKV projections via packed-ternary matmul.
     * Activation goes through A8 quantize first per W1.58A8 spec. */
    /* TODO: A8 quantize x_norm → q_int8 + scale; pass to BitLinear-equivalent.
     * For now, the substrate's existing matmul takes int8 X directly,
     * and we approximate by treating x_norm's MSB as int8. This is
     * NOT the BitNet protocol — work-unit 5 builds the actual A8 path. */

    /* QKV projections via packed-ternary matmul (gated on weights-loaded;
     * zero output if w->w_q is NULL — keeps harness runnable without
     * actual weights for skeleton testing).
     *
     * BitNet's W1.58A8 spec: x_norm gets A8-quantized to int8 (per-token
     * absmax), then matmul receives the int8. Q/K/V share x_norm as input
     * — we A8-quantize ONCE and reuse (RC-11 fix from work-unit 1 red-team:
     * computing the absmax separately for each projection wastes work
     * and risks numerical divergence if they produced different absmaxes).
     *
     * Note (RC-6): we call `m4t_ternary_5in8_matmul_bt` with non-ternary X
     * (int8 in [-127, 127] from A8 quantize, not {-1, 0, +1}). The kernel
     * uses SDOT which handles full int8 range correctly, but the documented
     * contract says X is ternary. This usage is intentional; future
     * cleanup (work-unit 5+) may add a substrate variant with a wider X
     * contract. */
    if (w->w_q != NULL) {
        /* Bit-faithful BitLinear: no a8 quantization. Q, K, V from x_norm. */
        bitnet_bitlinear_no_a8(s->q, s->x_norm, w->w_q,
                                w->alpha_q, w->alpha_q_block_exp,
                                BITNET_ACT_BX, BITNET_ACT_BX,
                                BITNET_HIDDEN_SIZE, BITNET_HIDDEN_SIZE);
        bitnet_bitlinear_no_a8(s->k, s->x_norm, w->w_k,
                                w->alpha_k, w->alpha_k_block_exp,
                                BITNET_ACT_BX, BITNET_ACT_BX,
                                BITNET_HIDDEN_SIZE, BITNET_KV_PROJ_DIM);
        bitnet_bitlinear_no_a8(s->v, s->x_norm, w->w_v,
                                w->alpha_v, w->alpha_v_block_exp,
                                BITNET_ACT_BX, BITNET_ACT_BX,
                                BITNET_HIDDEN_SIZE, BITNET_KV_PROJ_DIM);
    } else {
        /* No weights loaded — skeleton mode. Zero outputs. */
        memset(s->q, 0, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
        memset(s->k, 0, BITNET_KV_PROJ_DIM * sizeof(m4t_mtfp_t));
        memset(s->v, 0, BITNET_KV_PROJ_DIM * sizeof(m4t_mtfp_t));
    }

    /* Snapshot pre-RoPE Q, K for the dump (HF's q_proj/k_proj hooks
     * capture pre-RoPE; this lets the comparison be apples-to-apples). */
    memcpy(s->q_pre_rope, s->q, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
    memcpy(s->k_pre_rope, s->k, BITNET_KV_PROJ_DIM * sizeof(m4t_mtfp_t));

    /* RoPE on Q, K. */
    m4t_mtfp_rope_apply(s->q, s->k, position,
                        BITNET_NUM_ATTENTION_HEADS,
                        BITNET_NUM_KV_HEADS,
                        BITNET_HEAD_DIM,
                        BITNET_ROPE_THETA);

    /* Attention with KV cache (work-unit 7).
     *
     * Per Q head h:
     *   kv_head_idx = h / (num_q_heads / num_kv_heads)
     *   scores[t]   = Q[h] · K_cache[layer][t][kv_head_idx] / sqrt(head_dim)
     *                 for t ∈ [0, position+1)
     *   weights     = softmax(scores)        [length = position+1]
     *   attn_out[h] = Σ_t weights[t] · V_cache[layer][t][kv_head_idx]
     *
     * Score → softmax scale: softmax expects "1 LSB = 1 nat". Q · K
     * dot products in raw int can be enormous (up to 2^63ish). To bring
     * them into the LUT range [-30, 0] post-max-subtraction, we
     * right-shift the scores by enough bits to fit. This is a temperature
     * change relative to HF (softmax(x / T) ≠ softmax(x)) — the per-layer
     * ε comparison (work-unit 6) measures the resulting mismatch.
     *
     * 1/sqrt(head_dim) factor is folded into the rescale shift (it's a
     * constant downscale ≈ /11.3 = ~3.5 bits; absorbed into the heuristic). */
    {
        const int q_per_kv = BITNET_NUM_ATTENTION_HEADS / BITNET_NUM_KV_HEADS;

        if (cache == NULL) {
            /* No-cache fallback: degenerate seq=1 (attn_out = V). */
            for (int h = 0; h < BITNET_NUM_ATTENTION_HEADS; h++) {
                int kv_head = h / q_per_kv;
                memcpy(s->attn_output + (size_t)h * BITNET_HEAD_DIM,
                       s->v + (size_t)kv_head * BITNET_HEAD_DIM,
                       BITNET_HEAD_DIM * sizeof(m4t_mtfp_t));
            }
            memset(s->attn_scores, 0,
                   BITNET_NUM_ATTENTION_HEADS * sizeof(m4t_mtfp_t));
        } else {
            /* Write current K, V to cache at this layer's slot for `position`. */
            assert(layer_idx >= 0 && layer_idx < cache->n_layers);
            assert(position >= 0 && position < cache->max_seq_len);
            size_t row_size = (size_t)BITNET_NUM_KV_HEADS * BITNET_HEAD_DIM;
            size_t base = (size_t)layer_idx * cache->per_layer_stride
                          + (size_t)position * row_size;
            memcpy(cache->k + base, s->k, row_size * sizeof(m4t_mtfp_t));
            memcpy(cache->v + base, s->v, row_size * sizeof(m4t_mtfp_t));

            int seq_k = position + 1;

            /* Per-head scratch hoisted out of the head loop (RC-2). */
            int64_t* scores_i64    = (int64_t*)   malloc((size_t)seq_k * sizeof(int64_t));
            m4t_mtfp_t* scores_int = (m4t_mtfp_t*)malloc((size_t)seq_k * sizeof(m4t_mtfp_t));
            m4t_mtfp_t* weights    = (m4t_mtfp_t*)malloc((size_t)seq_k * sizeof(m4t_mtfp_t));
            assert(scores_i64 && scores_int && weights);

            /* Cycle 2: branch by attention mode. Dense path is bit-exact
             * unchanged from production. Sparse arms (random/routed/oracle)
             * use a separate code path with bitnet_sparse_attn_v_combine
             * (scalar — experimental, not on production hot path). */
            int sparse_active = (g_attn_mode != BITNET_ATTN_DENSE && g_attn_k < seq_k);

            for (int h = 0; h < BITNET_NUM_ATTENTION_HEADS; h++) {
                int kv_head = h / q_per_kv;
                const m4t_mtfp_t* qh = s->q + (size_t)h * BITNET_HEAD_DIM;
                m4t_mtfp_t* out_h = s->attn_output + (size_t)h * BITNET_HEAD_DIM;
                const m4t_mtfp_t* V_base = cache->v
                    + (size_t)layer_idx * cache->per_layer_stride
                    + (size_t)kv_head * BITNET_HEAD_DIM;

                if (!sparse_active) {
                    /* ── DENSE PATH (production) — unchanged from pre-Cycle 2. */
                    /* Compute scores[t] = dot(qh, K_cache[layer][t][kv_head]). */
                    int64_t max_abs = 1;
                    for (int t = 0; t < seq_k; t++) {
                        size_t k_row_base = (size_t)layer_idx * cache->per_layer_stride
                                            + (size_t)t * row_size
                                            + (size_t)kv_head * BITNET_HEAD_DIM;
                        const m4t_mtfp_t* kh = cache->k + k_row_base;
                        /* V14.A: NEON int32×int32→int64 dot via libm4t helper.
                         * Same semantics as the prior scalar loop, but the
                         * production code path is NEON-only per condition (5)
                         * of the pure-ternary directive. */
                        int64_t acc = m4t_mtfp_vec_dot_i64(qh, kh, BITNET_HEAD_DIM);
                        scores_i64[t] = acc;
                        int64_t a = acc < 0 ? -acc : acc;
                        if (a > max_abs) max_abs = a;
                    }

                    /* Rescale scores; see tuning history at score_shift below.
                     * Per journal/math_div_atomics_2026-05-10.md (gate1+fudge2). */
                    int score_shift = 0;
                    while ((max_abs >> score_shift) > 30) score_shift++;
                    score_shift += 2;

                    for (int t = 0; t < seq_k; t++) {
                        int64_t r;
                        if (scores_i64[t] >= 0) r = scores_i64[t] >> score_shift;
                        else                    r = -((-scores_i64[t]) >> score_shift);
                        if (r >  M4T_MTFP_MAX_VAL) r =  M4T_MTFP_MAX_VAL;
                        if (r < -M4T_MTFP_MAX_VAL) r = -M4T_MTFP_MAX_VAL;
                        scores_int[t] = (m4t_mtfp_t)r;
                    }

                    m4t_mtfp_softmax(weights, scores_int, seq_k);

                    /* V14.B: NEON outer-product accumulate via libm4t helper.
                     * Loop order swapped (t outer, d inner) for NEON
                     * vmlal_s32 broadcast pattern. */
                    m4t_mtfp_attn_v_combine(out_h, 30, weights, V_base, row_size,
                                            seq_k, BITNET_HEAD_DIM);
                } else {
                    /* ── SPARSE PATH (Cycle 2 experimental) ─────────────── */
                    int k_eff = g_attn_k;  /* sparse_active gate ensures < seq_k */

                    int* indices = (int*)malloc((size_t)k_eff * sizeof(int));
                    int64_t* sub_scores_i64 = (int64_t*)malloc((size_t)k_eff * sizeof(int64_t));
                    m4t_mtfp_t* sub_scores_int = (m4t_mtfp_t*)malloc((size_t)k_eff * sizeof(m4t_mtfp_t));
                    m4t_mtfp_t* sub_weights = (m4t_mtfp_t*)malloc((size_t)k_eff * sizeof(m4t_mtfp_t));
                    assert(indices && sub_scores_i64 && sub_scores_int && sub_weights);

                    /* Pick indices per arm. RANDOM and ORACLE implemented in
                     * Phases 2.2/2.3. ROUTED is Phase 2.4 (TODO). */
                    int64_t max_abs = 1;
                    if (g_attn_mode == BITNET_ATTN_ORACLE ||
                        g_attn_mode == BITNET_ATTN_POSRACLE) {
                        /* ORACLE / POSRACLE: compute dense scores first, then pick top-k.
                         * ORACLE picks by |score|; POSRACLE by signed score (positives win). */
                        for (int t = 0; t < seq_k; t++) {
                            size_t k_row_base = (size_t)layer_idx * cache->per_layer_stride
                                                + (size_t)t * row_size
                                                + (size_t)kv_head * BITNET_HEAD_DIM;
                            const m4t_mtfp_t* kh = cache->k + k_row_base;
                            scores_i64[t] = m4t_mtfp_vec_dot_i64(qh, kh, BITNET_HEAD_DIM);
                        }
                        if (g_attn_mode == BITNET_ATTN_ORACLE)
                            bitnet_pick_oracle_topk  (indices, k_eff, seq_k, scores_i64);
                        else
                            bitnet_pick_posracle_topk(indices, k_eff, seq_k, scores_i64);
                        /* Pull the chosen scores into the sub buffer. */
                        for (int i = 0; i < k_eff; i++) {
                            sub_scores_i64[i] = scores_i64[indices[i]];
                            int64_t a = sub_scores_i64[i] < 0 ? -sub_scores_i64[i] : sub_scores_i64[i];
                            if (a > max_abs) max_abs = a;
                        }
                    } else {
                        /* RANDOM and ROUTED — pick indices, then compute true scores. */
                        if (g_attn_mode == BITNET_ATTN_RANDOM) {
                            bitnet_pick_random_indices(indices, k_eff, seq_k, &g_attn_rng);
                        } else if (g_attn_mode == BITNET_ATTN_ROUTED) {
                            const m4t_mtfp_t* k_cache_layer =
                                cache->k + (size_t)layer_idx * cache->per_layer_stride;
                            bitnet_pick_routed_indices(
                                indices, k_eff, seq_k,
                                qh, k_cache_layer, row_size, kv_head,
                                BITNET_HEAD_DIM);
                        }
                        /* Compute true Q·K dot at the chosen indices only. */
                        for (int i = 0; i < k_eff; i++) {
                            int t = indices[i];
                            size_t k_row_base = (size_t)layer_idx * cache->per_layer_stride
                                                + (size_t)t * row_size
                                                + (size_t)kv_head * BITNET_HEAD_DIM;
                            const m4t_mtfp_t* kh = cache->k + k_row_base;
                            int64_t acc = m4t_mtfp_vec_dot_i64(qh, kh, BITNET_HEAD_DIM);
                            sub_scores_i64[i] = acc;
                            int64_t a = acc < 0 ? -acc : acc;
                            if (a > max_abs) max_abs = a;
                        }
                    }

                    int score_shift = 0;
                    while ((max_abs >> score_shift) > 30) score_shift++;
                    score_shift += 2;

                    for (int i = 0; i < k_eff; i++) {
                        int64_t r;
                        if (sub_scores_i64[i] >= 0) r = sub_scores_i64[i] >> score_shift;
                        else                        r = -((-sub_scores_i64[i]) >> score_shift);
                        if (r >  M4T_MTFP_MAX_VAL) r =  M4T_MTFP_MAX_VAL;
                        if (r < -M4T_MTFP_MAX_VAL) r = -M4T_MTFP_MAX_VAL;
                        sub_scores_int[i] = (m4t_mtfp_t)r;
                    }

                    m4t_mtfp_softmax(sub_weights, sub_scores_int, k_eff);

                    bitnet_sparse_attn_v_combine(out_h, 30, sub_weights,
                                                  V_base, row_size,
                                                  k_eff, BITNET_HEAD_DIM,
                                                  indices);

                    free(indices); free(sub_scores_i64);
                    free(sub_scores_int); free(sub_weights);
                }
            }

            /* Stash one debug score per head (last position's score),
             * for the activation dump. */
            for (int h = 0; h < BITNET_NUM_ATTENTION_HEADS; h++) {
                s->attn_scores[h] = 0;  /* not load-bearing in tests */
            }

            free(scores_i64); free(scores_int); free(weights);
        }
    }

    /* attn_sub_norm: y = γ · y · rsqrt(mean(y²) + ε) */
    m4t_mtfp_rmsnorm_bx(s->attn_sub_norm, s->attn_output,
                        w->gamma_attn_sub_norm,
                        BITNET_ACT_BX, w->gamma_attn_sub_norm_block_exp,
                        BITNET_ACT_BX, /*eps=*/1, BITNET_HIDDEN_SIZE);

    /* O projection: y = attn_sub_norm @ W_o^T (BitLinear, A8-quantized).
     * Per-projection input — own A8 quantize. */
    if (w->w_o != NULL) {
        bitnet_bitlinear_no_a8(s->x, s->attn_sub_norm, w->w_o,
                                w->alpha_o, w->alpha_o_block_exp,
                                BITNET_ACT_BX, BITNET_ACT_BX,
                                BITNET_HIDDEN_SIZE, BITNET_HIDDEN_SIZE);
    } else {
        memcpy(s->x, s->attn_sub_norm, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
    }

    /* DEBUG: dump o_proj output (pre-residual-add) for layer-by-layer
     * substrate-vs-HF localization. Off by default; set DEBUG_DUMP_OPROJ env. */
    {
        const char* dbg = getenv("DEBUG_DUMP_OPROJ");
        if (dbg && layer_idx == 0) {
            FILE* f = fopen(dbg, "wb");
            if (f) { fwrite(s->x, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f); fclose(f); }
        }
    }

    /* x = residual + x (V13.A: NEON via libm4t's saturating-add
     * primitive — same MTFP19 clamp semantics as the previous scalar
     * loop, no scalar production code per condition (5) of the
     * pure-ternary directive). */
    m4t_mtfp_vec_add_inplace(s->x, s->residual, BITNET_HIDDEN_SIZE);

    /* DEBUG: dump post-residual-add (= input to post_attn_norm). */
    {
        const char* dbg = getenv("DEBUG_DUMP_PREPN");
        if (dbg && layer_idx == 0) {
            FILE* f = fopen(dbg, "wb");
            if (f) { fwrite(s->x, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f); fclose(f); }
        }
    }

    /* ── FFN sub-block ────────────────────────────────────────────── */

    /* residual = x */
    memcpy(s->residual, s->x, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));

    /* x_norm = post_attention_layernorm(x) */
    m4t_mtfp_rmsnorm_bx(s->x_norm, s->x, w->gamma_post_attn_norm,
                        BITNET_ACT_BX, w->gamma_post_attn_norm_block_exp,
                        BITNET_ACT_BX, /*eps=*/1, BITNET_HIDDEN_SIZE);

    /* gate, up = x_norm projected (BitLinear, A8). They share x_norm as
     * input — A8-quantize ONCE and reuse (RC-11 fix). */
    if (w->w_gate != NULL && w->w_up != NULL) {
        /* gate, up at FFN_BX (wider headroom; gate values typical ~80). */
        bitnet_bitlinear_no_a8(s->gate, s->x_norm, w->w_gate,
                                w->alpha_gate, w->alpha_gate_block_exp,
                                BITNET_ACT_BX, BITNET_FFN_BX,
                                BITNET_HIDDEN_SIZE, BITNET_INTERMEDIATE_SIZE);
        bitnet_bitlinear_no_a8(s->up, s->x_norm, w->w_up,
                                w->alpha_up, w->alpha_up_block_exp,
                                BITNET_ACT_BX, BITNET_FFN_BX,
                                BITNET_HIDDEN_SIZE, BITNET_INTERMEDIATE_SIZE);
    } else {
        memset(s->gate, 0, BITNET_INTERMEDIATE_SIZE * sizeof(m4t_mtfp_t));
        memset(s->up,   0, BITNET_INTERMEDIATE_SIZE * sizeof(m4t_mtfp_t));
    }

    /* gate_act = relu²(gate) × up. gate at FFN_BX, relu² stays at FFN_BX
     * (gate²_real ≤ ~10K fits MTFP19_MAX/3^8 ≈ 88K). Mul with up (FFN_BX)
     * produces products up to gate²×up ≈ 6M for outliers — needs the wider
     * GATE_ACT_BX. */
    m4t_mtfp_relu2_inplace_bx(s->gate, BITNET_FFN_BX, BITNET_FFN_BX,
                              BITNET_INTERMEDIATE_SIZE);
    m4t_mtfp_elementwise_mul_bx(s->gate_act,
                                s->gate, BITNET_FFN_BX,
                                s->up,   BITNET_FFN_BX,
                                BITNET_GATE_ACT_BX,
                                BITNET_INTERMEDIATE_SIZE);

    /* ffn_sub_norm(gate_act). gate_act is at GATE_ACT_BX. Output at
     * ACT_BX (resumes linear-magnitude flow for down_proj). */
    m4t_mtfp_rmsnorm_bx(s->ffn_sub_norm, s->gate_act,
                        w->gamma_ffn_sub_norm,
                        BITNET_GATE_ACT_BX, w->gamma_ffn_sub_norm_block_exp,
                        BITNET_ACT_BX, /*eps=*/1, BITNET_INTERMEDIATE_SIZE);

    /* down = ffn_sub_norm @ W_down^T (BitLinear, A8). Different input than
     * gate/up — own A8 quantize. ffn_sub_norm output is at ACT_BX. */
    if (w->w_down != NULL) {
        bitnet_bitlinear_no_a8(s->x, s->ffn_sub_norm, w->w_down,
                                w->alpha_down, w->alpha_down_block_exp,
                                BITNET_ACT_BX, BITNET_ACT_BX,
                                BITNET_INTERMEDIATE_SIZE, BITNET_HIDDEN_SIZE);
    } else {
        memset(s->x, 0, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
    }

    /* x = residual + x (V13.A: NEON via libm4t's saturating-add
     * primitive — same MTFP19 clamp semantics as the previous scalar
     * loop, no scalar production code per condition (5) of the
     * pure-ternary directive). */
    m4t_mtfp_vec_add_inplace(s->x, s->residual, BITNET_HIDDEN_SIZE);

    /* Copy back to caller's buffer. */
    memcpy(x_io, s->x, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
}

/* ── main ─────────────────────────────────────────────────────────── */

/* Per-layer activation dump: writes captured tensors to a binary file
 * compatible with the comparison driver (scripts/compare_activations.py).
 *
 * Format: one .bin per (layer, sublayer) capture site, containing raw
 * int32 mantissas. A small JSON sidecar lists shapes + block_exps (TBD).
 *
 * Phase 1 simplification: just dump a few key sites to one consolidated
 * file. Detailed sublayer dumps are work-unit 6+ scope. */
static int dump_activations_to_file(
    const char* path,
    const bitnet_block_scratch_t* s,
    int layer)
{
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[harness] cannot open %s for write\n", path);
        return 1;
    }
    /* Header: magic "ACTV2" (bumped from ACTV in red-team), layer,
     * sizes. v2 adds x_norm_input, q_pre_rope, k_pre_rope captures. */
    fwrite("ACTV2", 1, 5, f);
    /* 3-byte pad to keep 4-byte alignment for the int32s that follow. */
    char pad[3] = {0, 0, 0}; fwrite(pad, 1, 3, f);
    int32_t li = layer;
    fwrite(&li, sizeof(int32_t), 1, f);
    int32_t hidden = BITNET_HIDDEN_SIZE;
    int32_t intermediate = BITNET_INTERMEDIATE_SIZE;
    int32_t kv_proj = BITNET_KV_PROJ_DIM;
    fwrite(&hidden, sizeof(int32_t), 1, f);
    fwrite(&intermediate, sizeof(int32_t), 1, f);
    fwrite(&kv_proj, sizeof(int32_t), 1, f);

    /* Capture order (matches scripts/compare_activations.py CAPTURE_ORDER): */
    /* 1. x_norm_input — true post-input_layernorm output */
    fwrite(s->x_norm_input, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    /* 2. q_pre_rope — post-q_proj+α scale, before RoPE */
    fwrite(s->q_pre_rope, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    /* 3. k_pre_rope — post-k_proj+α scale, before RoPE */
    fwrite(s->k_pre_rope, sizeof(m4t_mtfp_t), BITNET_KV_PROJ_DIM, f);
    /* 4. v — post-v_proj+α scale (no RoPE on V) */
    fwrite(s->v, sizeof(m4t_mtfp_t), BITNET_KV_PROJ_DIM, f);
    /* 5. q_post_rope — post-RoPE Q (no HF analog hooked) */
    fwrite(s->q, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    /* 6. k_post_rope — post-RoPE K */
    fwrite(s->k, sizeof(m4t_mtfp_t), BITNET_KV_PROJ_DIM, f);
    /* 7. attn_sub_norm output */
    fwrite(s->attn_sub_norm, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    /* 8. x_norm — post-attn-residual rmsnorm input (post_attention_layernorm output) */
    fwrite(s->x_norm, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    /* 9. gate (NB: post-relu²-inplace; raw gate_proj NOT captured) */
    fwrite(s->gate, sizeof(m4t_mtfp_t), BITNET_INTERMEDIATE_SIZE, f);
    /* 10. up (post-up_proj+α scale, no relu²) */
    fwrite(s->up, sizeof(m4t_mtfp_t), BITNET_INTERMEDIATE_SIZE, f);
    /* 11. ffn_sub_norm output */
    fwrite(s->ffn_sub_norm, sizeof(m4t_mtfp_t), BITNET_INTERMEDIATE_SIZE, f);
    /* 12. block_output */
    fwrite(s->x, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    fclose(f);
    return 0;
}

/* Embedding lookup: x[i] = embedding[token_id × HIDDEN + i] as MTFP19
 * mantissas. The block_exp of the embedding is set at conversion time
 * (per-tensor). Caller must hold block_exp tracking externally if needed.
 *
 * Phase 1 simplification: we read mantissas directly as activation
 * values, treating everything as "block_exp 0" through the forward
 * pass. This loses precision relative to HF's bf16/fp32 but produces
 * a consistent, deterministic substrate output. Per-layer ε
 * comparison (work-unit 6's gate) measures how the discrepancy
 * accumulates. */
static void bitnet_embed(
    m4t_mtfp_t* x_out,
    const m4t_mtfp_t* embedding,
    int embedding_bx,
    int token_id)
{
    if (embedding == NULL) {
        for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) x_out[i] = 0;
        return;
    }
    const m4t_mtfp_t* row = embedding + (size_t)token_id * BITNET_HIDDEN_SIZE;
    /* Phase 2 work-unit 1: rescale embedding row from its stored bx
     * into the activation flow's BITNET_ACT_BX. */
    m4t_mtfp_rescale_bx(x_out, row, embedding_bx, BITNET_ACT_BX,
                        BITNET_HIDDEN_SIZE);
}

/* LM head: logits = x @ lm_head^T. lm_head is [VOCAB × HIDDEN] MTFP19
 * mantissas. Output: logits at scale (lm_head_block_exp + x_block_exp)
 * in mantissa units; magnitudes can grow; we don't compute argmax in C
 * (the comparison driver does that against HF reference output). */
static void bitnet_lm_head(
    m4t_mtfp_t* logits_out,
    const m4t_mtfp_t* x,
    const m4t_mtfp_t* lm_head,
    int top_n)
{
    /* For comparison purposes we only need logits[0..top_n) — full vocab
     * (128256) is dumped to file by the comparison driver if needed.
     *
     * V13.B of pure-ternary audit: dot product via libm4t's NEON
     * helper (vmlal_s32 chain), no scalar production code per
     * condition (5) of the directive. */
    for (int v = 0; v < top_n; v++) {
        const m4t_mtfp_t* row = lm_head + (size_t)v * BITNET_HIDDEN_SIZE;
        int64_t acc = m4t_mtfp_vec_dot_i64(x, row, BITNET_HIDDEN_SIZE);
        /* Crude scale-down to fit MTFP19; the comparison driver consumes
         * raw acc values for ε measurement so we expose them at int64
         * in a separate accumulator buffer (caller supplied). */
        int64_t scaled = acc >> 30;  /* somewhat arbitrary; rescaled by Python comparison */
        logits_out[v] = m4t_mtfp_clamp64(scaled);
    }
}

/* Argmax over full vocabulary using raw int64 logits (no shift).
 * Phase 1 greedy decoding: returns the token id with highest logit.
 * Returns -1 if lm_head is NULL. */
static int bitnet_argmax_full_vocab(
    const m4t_mtfp_t* x,
    const m4t_mtfp_t* lm_head)
{
    if (lm_head == NULL) return -1;
    int64_t best_acc = INT64_MIN;
    int     best_v   = 0;
    /* V13.B of pure-ternary audit: per-vocab dot product via libm4t's
     * NEON helper. No scalar production code per condition (5). */
    for (int v = 0; v < BITNET_VOCAB_SIZE; v++) {
        const m4t_mtfp_t* row = lm_head + (size_t)v * BITNET_HIDDEN_SIZE;
        int64_t acc = m4t_mtfp_vec_dot_i64(x, row, BITNET_HIDDEN_SIZE);
        if (acc > best_acc) { best_acc = acc; best_v = v; }
    }
    return best_v;
}

int main(int argc, char** argv) {
    /* Cycle 2: read sparse-attention mode from env. No-op when env unset. */
    bitnet_attn_mode_init_from_env();

    int token_id = 1;          /* default: BOS-like token */
    int n_layers = -1;         /* -1 = all loaded layers */
    int n_positions = 1;       /* number of positions to forward (work-unit 7 cache) */
    int prompt_tokens[256] = {0};
    int n_prompt_tokens = 0;   /* if >0, overrides --token + --positions */
    int n_generate = 0;        /* generation steps after the prompt (work-unit 8) */
    const char* weights_arg = NULL;
    const char* dump_path = NULL;

    /* Simple positional + flag parsing. */
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <weights_blob.bin|-> "
            "[--token <id>] [--layers <n>] [--positions <p>] [--dump <path>]\n"
            "  weights_blob.bin — produced by scripts/convert_weights.py.\n"
            "                     '-' for skeleton mode.\n"
            "  --token <id>     — input token id (default: 1).\n"
            "  --layers <n>     — number of transformer layers to run\n"
            "                     (default: all loaded).\n"
            "  --positions <p>  — number of forward passes to run\n"
            "                     (positions 0..p-1; default 1). Exercises\n"
            "                     the KV cache across multiple decode steps.\n"
            "  --gen <n>        — greedy-generate <n> tokens after the\n"
            "                     prompt (work-unit 8). Default 0 (no\n"
            "                     generation). Each step: forward, argmax\n"
            "                     over LM head, embed, repeat.\n"
            "  --dump <path>    — write per-layer activation snapshots to\n"
            "                     <path>.layer<N>.bin (last position) for\n"
            "                     ε comparison.\n",
            argv[0]);
        return 1;
    }
    weights_arg = argv[1];
    for (int i = 2; i + 1 < argc; i += 2) {
        if      (strcmp(argv[i], "--token")     == 0) token_id    = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--layers")    == 0) n_layers    = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--positions") == 0) n_positions = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--prompt-tokens") == 0) {
            /* Parse comma-separated token ids. */
            const char* p = argv[i+1];
            n_prompt_tokens = 0;
            while (*p && n_prompt_tokens < 256) {
                prompt_tokens[n_prompt_tokens++] = atoi(p);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
            n_positions = n_prompt_tokens;
        }
        else if (strcmp(argv[i], "--gen")       == 0) n_generate  = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--dump")      == 0) dump_path   = argv[i+1];
        else { fprintf(stderr, "[harness] unknown flag: %s\n", argv[i]); return 1; }
    }
    if (n_positions < 1) n_positions = 1;
    if (n_generate  < 0) n_generate  = 0;

    bitnet_weights_t weights = {0};
    bitnet_weights_loaded_t handle = {0};
    int loaded_ok = 0;
    if (strcmp(weights_arg, "-") != 0) {
        if (bitnet_weights_load(weights_arg, &weights, &handle) == 0) {
            loaded_ok = 1;
            fprintf(stderr, "[harness] loaded weights from %s\n", weights_arg);
        } else {
            fprintf(stderr, "[harness] weight load failed; running skeleton mode\n");
        }
    } else {
        fprintf(stderr, "[harness] skeleton mode (no weights)\n");
    }

    /* Allocate scratch. */
    bitnet_block_scratch_t s = {0};
    bitnet_block_scratch_alloc(&s);

    /* Input: embed token_id (or fallback pattern in skeleton mode). */
    m4t_mtfp_t x[BITNET_HIDDEN_SIZE];
    if (loaded_ok && weights.embedding != NULL) {
        bitnet_embed(x, weights.embedding, weights.embedding_block_exp, token_id);
    } else {
        for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) x[i] = (i % 7) - 3;
    }

    /* Determine layer count to run. */
    int layers_to_run = (n_layers >= 0) ? n_layers : BITNET_NUM_LAYERS;
    if (layers_to_run > BITNET_NUM_LAYERS) layers_to_run = BITNET_NUM_LAYERS;

    /* Skeleton fallback weights for layers without loaded weights. */
    bitnet_layer_weights_t w_dummy = {0};
    static m4t_mtfp_t gamma_dummy[BITNET_INTERMEDIATE_SIZE];
    static m4t_mtfp_t alpha_dummy = 0;  /* triggers no-op scale in apply helper */
    for (int i = 0; i < BITNET_INTERMEDIATE_SIZE; i++) gamma_dummy[i] = 1;
    w_dummy.gamma_input_norm     = gamma_dummy;
    w_dummy.gamma_post_attn_norm = gamma_dummy;
    w_dummy.gamma_attn_sub_norm  = gamma_dummy;
    w_dummy.gamma_ffn_sub_norm   = gamma_dummy;
    w_dummy.alpha_q = w_dummy.alpha_k = w_dummy.alpha_v = w_dummy.alpha_o
        = w_dummy.alpha_gate = w_dummy.alpha_up = w_dummy.alpha_down = &alpha_dummy;

    /* KV cache (work-unit 7). Sized for prompt-positions + generation. */
    bitnet_kv_cache_t cache = {0};
    int total_positions = n_positions + n_generate;
    int max_seq = total_positions > 256 ? total_positions : 256;
    if (bitnet_kv_cache_alloc(&cache, max_seq, layers_to_run) != 0) {
        fprintf(stderr, "[harness] KV cache alloc failed\n");
        return 1;
    }

    /* Inner forward pass for one position. Mutates x in place; assumes
     * cache, scratch, weights, and x are set up by the caller. */
    #define FORWARD_ONE(pos_, dump_this) do {                              \
        for (int l = 0; l < layers_to_run; l++) {                          \
            const bitnet_layer_weights_t* w_l;                             \
            if (loaded_ok && weights.layers[l].w_q != NULL)                \
                w_l = &weights.layers[l];                                  \
            else                                                            \
                w_l = &w_dummy;                                            \
            bitnet_forward_block(x, w_l, &s, &cache, l, (pos_));           \
            if (dump_path && (dump_this)) {                                \
                char path[1024];                                           \
                snprintf(path, sizeof(path), "%s.layer%d.bin", dump_path, l); \
                dump_activations_to_file(path, &s, l);                     \
            }                                                              \
        }                                                                  \
        cache.current_pos = (pos_) + 1;                                    \
    } while (0)

    /* Re-embed scratch — used to re-feed the same prompt token at each
     * --positions step (skeleton mode without generation). */
    m4t_mtfp_t x_init[BITNET_HIDDEN_SIZE];
    memcpy(x_init, x, sizeof(x_init));

    /* Phase 1: prompt-position forward.
     * If --prompt-tokens is given, embed each prompt token in turn.
     * Otherwise re-feed the same --token at each --positions step. */
    int last_dump_pos = (n_generate > 0) ? -1 : (n_positions - 1);
    for (int pos = 0; pos < n_positions; pos++) {
        if (n_prompt_tokens > 0) {
            int tok = prompt_tokens[pos];
            if (loaded_ok && weights.embedding != NULL) {
                bitnet_embed(x, weights.embedding, weights.embedding_block_exp, tok);
            } else {
                for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) x[i] = (i % 7) - 3;
            }
        } else if (pos > 0) {
            memcpy(x, x_init, sizeof(x_init));
        }
        /* Per-position dump if --prompt-tokens used (each position gets its own
         * dump file). For single-token mode, only the last position dumps. */
        int dump_this = (n_prompt_tokens > 0)
                        ? (dump_path != NULL)
                        : (pos == last_dump_pos);
        /* Inline FORWARD_ONE with a per-position dump suffix. */
        for (int l = 0; l < layers_to_run; l++) {
            const bitnet_layer_weights_t* w_l;
            if (loaded_ok && weights.layers[l].w_q != NULL) w_l = &weights.layers[l];
            else                                            w_l = &w_dummy;
            bitnet_forward_block(x, w_l, &s, &cache, l, pos);
            if (dump_this) {
                char path[1024];
                if (n_prompt_tokens > 0) {
                    snprintf(path, sizeof(path), "%s.pos%d.layer%d.bin", dump_path, pos, l);
                } else {
                    snprintf(path, sizeof(path), "%s.layer%d.bin", dump_path, l);
                }
                dump_activations_to_file(path, &s, l);
            }
        }
        cache.current_pos = pos + 1;
    }

    /* Generated tokens accumulator. */
    int generated_tokens[2048];
    int n_generated = 0;

    /* Phase 2: greedy generation (--gen N). After each forward, take
     * argmax over LM head logits; that token id becomes the next input
     * embedding. Stops at n_generate or when sequence_len fills the cache. */
    for (int g = 0; g < n_generate; g++) {
        int pos = n_positions + g;
        if (pos >= max_seq) break;

        /* Apply final norm + argmax to the current x (output of last
         * prompt/gen forward). */
        m4t_mtfp_t x_finalnorm[BITNET_HIDDEN_SIZE];
        if (loaded_ok && weights.gamma_final_norm != NULL) {
            m4t_mtfp_rmsnorm_bx(x_finalnorm, x, weights.gamma_final_norm,
                                BITNET_ACT_BX, weights.gamma_final_norm_block_exp,
                                BITNET_ACT_BX, /*eps=*/1, BITNET_HIDDEN_SIZE);
        } else {
            memcpy(x_finalnorm, x, sizeof(x_finalnorm));
        }
        int next_tok = bitnet_argmax_full_vocab(x_finalnorm, weights.lm_head);
        if (next_tok < 0) {
            /* Skeleton mode (no LM head): pick a fixed dummy token. */
            next_tok = (token_id + g + 1) % BITNET_VOCAB_SIZE;
        }
        if (n_generated < (int)(sizeof(generated_tokens)/sizeof(int))) {
            generated_tokens[n_generated++] = next_tok;
        }

        /* Embed the new token; forward at pos. */
        if (loaded_ok && weights.embedding != NULL) {
            bitnet_embed(x, weights.embedding, weights.embedding_block_exp, next_tok);
        } else {
            for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) x[i] = (i % 7) - 3;
        }
        int dump_this = (g == n_generate - 1) && dump_path != NULL;
        FORWARD_ONE(pos, dump_this);
    }

    #undef FORWARD_ONE

    /* Final norm + LM head on the last x (post all forwards). */
    if (loaded_ok && weights.gamma_final_norm != NULL) {
        m4t_mtfp_t x_norm[BITNET_HIDDEN_SIZE];
        m4t_mtfp_rmsnorm_bx(x_norm, x, weights.gamma_final_norm,
                            BITNET_ACT_BX, weights.gamma_final_norm_block_exp,
                            BITNET_ACT_BX, /*eps=*/1, BITNET_HIDDEN_SIZE);
        memcpy(x, x_norm, sizeof(x));
    }

    int top_n = 16;
    m4t_mtfp_t logits[16];
    if (loaded_ok && weights.lm_head != NULL) {
        bitnet_lm_head(logits, x, weights.lm_head, top_n);
        /* Also compute argmax over full vocab + dump raw int64 logits
         * (red-team #6: cross-check substrate's predicted next-token
         * against HF reference). */
        int argmax_tok = bitnet_argmax_full_vocab(x, weights.lm_head);
        fprintf(stderr, "     argmax over full vocab        = %d\n", argmax_tok);
        if (dump_path) {
            char logits_path[1024];
            snprintf(logits_path, sizeof(logits_path), "%s.logits.bin", dump_path);
            FILE* lf = fopen(logits_path, "wb");
            if (lf) {
                int32_t vocab = BITNET_VOCAB_SIZE;
                fwrite(&vocab, sizeof(int32_t), 1, lf);
                int64_t* full_logits = (int64_t*)malloc((size_t)vocab * sizeof(int64_t));
                for (int v = 0; v < vocab; v++) {
                    const m4t_mtfp_t* row = weights.lm_head + (size_t)v * BITNET_HIDDEN_SIZE;
                    int64_t acc = 0;
                    for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) {
                        acc += (int64_t)x[i] * (int64_t)row[i];
                    }
                    full_logits[v] = acc;
                }
                fwrite(full_logits, sizeof(int64_t), (size_t)vocab, lf);
                free(full_logits);
                fclose(lf);
                fprintf(stderr, "     dumped full logits → %s\n", logits_path);
            }
        }
    } else {
        memset(logits, 0, sizeof(logits));
    }

    fprintf(stderr,
        "[ok] %d layer(s) forward pass completed (token_id=%d, "
            "positions=%d, generated=%d).\n"
        "     post-final-norm x[0..3]      = %d %d %d %d\n"
        "     logits[0..3]                  = %d %d %d %d\n",
        layers_to_run, token_id, n_positions, n_generated,
        x[0], x[1], x[2], x[3],
        logits[0], logits[1], logits[2], logits[3]);
    if (n_generated > 0) {
        fprintf(stderr, "     generated tokens             =");
        for (int i = 0; i < n_generated; i++) {
            fprintf(stderr, " %d", generated_tokens[i]);
        }
        fprintf(stderr, "\n");
    }

    bitnet_block_scratch_free(&s);
    bitnet_kv_cache_free(&cache);
    if (loaded_ok) bitnet_weights_unload(&handle);
    return 0;
}
