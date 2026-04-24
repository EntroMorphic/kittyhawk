/*
 * direct_lsh.c — direct ternary quantization + Trit Lattice LSH.
 *
 * Each trit represents a SPECIFIC input dimension (pixel or gradient),
 * not a random mixture. Normalized pixels are quantized to {-1, 0, +1}
 * via per-value thresholding. Optionally appends horizontal and vertical
 * gradients as additional trit channels.
 *
 * The quantized image IS the signature. The LSH infrastructure (bucket
 * index, multi-probe, k-NN resolve) operates on these direct signatures.
 *
 * NO RANDOM PROJECTIONS. Each trit = one pixel or one gradient.
 */

#include "glyph_config.h"
#include "glyph_dataset.h"
#include "glyph_rng.h"
#include "glyph_sig.h"
#include "glyph_bucket.h"
#include "glyph_multiprobe.h"
#include "glyph_probe.h"
#include "glyph_resolver.h"
#include "m4t_trit_pack.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#  include <arm_neon.h>
#endif

/* SDOT-based similarity score — consumer-local variant of the
 * substrate SDOT matmul kernel, but returning a raw int32 instead
 * of clamping to the MTFP4 range. Used by the E1 distance-function
 * experiment (see journal/distance_function_synthesize.md). If this
 * primitive proves useful across consumers, promote to libm4t as
 * m4t_sdot_score under the "no primitive without named consumer
 * demand" rule. */
static int32_t sdot_score(const int8_t* a, const int8_t* b, int n) {
    int32_t acc = 0;
    int k = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t vacc = vdupq_n_s32(0);
    for (; k + 16 <= n; k += 16) {
        int8x16_t va = vld1q_s8(a + k);
        int8x16_t vb = vld1q_s8(b + k);
        vacc = vdotq_s32(vacc, va, vb);
    }
    acc = vaddvq_s32(vacc);
#endif
    for (; k < n; k++) {
        acc += (int32_t)a[k] * (int32_t)b[k];
    }
    return acc;
}

/* F-stat dim selection: sortable (f, d) pair. Used in main() by the
 * --fstat_K encoder path. File scope so the qsort comparator can be
 * a plain static function (nested functions are a GCC extension). */
typedef struct { double f; int d; } fstat_entry_t;
static int fstat_cmp_desc(const void* a, const void* b) {
    double fa = ((const fstat_entry_t*)a)->f;
    double fb = ((const fstat_entry_t*)b)->f;
    return (fa < fb) - (fa > fb);
}

/* Integer 2× downsample by 2×2-block average, per channel.
 * src: n_ch × H × W (H, W assumed even). dst: n_ch × H/2 × W/2. */
static void downsample_2x_mtfp(
    const m4t_mtfp_t* src, m4t_mtfp_t* dst,
    int img_w, int img_h, int n_ch)
{
    int out_w = img_w / 2;
    int out_h = img_h / 2;
    for (int c = 0; c < n_ch; c++) {
        const m4t_mtfp_t* src_ch = src + (size_t)c * img_h * img_w;
        m4t_mtfp_t* dst_ch = dst + (size_t)c * out_h * out_w;
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                int sx = x * 2, sy = y * 2;
                int64_t sum = (int64_t)src_ch[ sy    * img_w + sx]
                            + (int64_t)src_ch[ sy    * img_w + sx + 1]
                            + (int64_t)src_ch[(sy+1) * img_w + sx]
                            + (int64_t)src_ch[(sy+1) * img_w + sx + 1];
                dst_ch[y * out_w + x] = (m4t_mtfp_t)(sum >> 2);
            }
        }
    }
}

/* Pair-IG mismatch-weighted sum (NEON).
 *
 * Computes Σ_d pw[d] × (q[d] != t[d]) for ternary q, t ∈ {-1, 0, +1}
 * stored as int8. The mismatch indicator (q[d] != t[d]) is expressed as
 *     min(|q[d] - t[d]|, 1) ∈ {0, 1}
 * which lets the whole sum land inside a single vdotq_s32 — 16 ops per
 * NEON iteration. Pair-IG weights pw[d] ∈ [1, 16] fit int8.
 *
 * Replaces the per-dim scalar glyph_read_trit loops in pair-IG re-rank
 * and filtered pair-IG. Requires int8 ternary buffers (now built
 * unconditionally on direct_lsh startup). */
static int32_t pair_ig_mismatch_score(
    const int8_t* q_i8, const int8_t* t_i8,
    const uint8_t* pw, int n)
{
    int32_t acc = 0;
    int k = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t vacc = vdupq_n_s32(0);
    int8x16_t one = vdupq_n_s8(1);
    for (; k + 16 <= n; k += 16) {
        int8x16_t qv   = vld1q_s8(q_i8 + k);
        int8x16_t tv   = vld1q_s8(t_i8 + k);
        int8x16_t diff = vabsq_s8(vsubq_s8(qv, tv));           /* {0, 1, 2} */
        int8x16_t mask = vminq_s8(diff, one);                   /* {0, 1}    */
        int8x16_t wv   = vreinterpretq_s8_u8(vld1q_u8(pw + k));
        vacc = vdotq_s32(vacc, mask, wv);
    }
    acc = vaddvq_s32(vacc);
#endif
    for (; k < n; k++) {
        if (q_i8[k] != t_i8[k]) acc += (int32_t)pw[k];
    }
    return acc;
}

/* E3: 4-trit block distance with threshold-count aggregator.
 *
 * Block = one packed byte = 4 trits (TBL-native dispatch unit).
 * Per-block cost = popcount(q_byte ^ t_byte) ∈ [0, 8].
 * Score = count of blocks whose per-block cost > T. Smaller score
 * = closer. This is a structurally different distance from per-trit
 * Hamming: it measures "how many local neighborhoods disagreed
 * sufficiently," not "how many individual trits disagreed."
 *
 * NEON: process 16 bytes per iteration. XOR → vcntq_u8 (per-byte
 * popcount) → vcgtq_u8 against T → pairwise-widen-accumulate the
 * 0xFF-per-mismatching-block pattern.
 *
 * No mask parameter — padded trits in the final byte always XOR
 * to zero (pack functions zero-code padding), so unmasked popcount
 * is correct. */
static int32_t block_threshold_score(
    const uint8_t* q, const uint8_t* t, int n_bytes, int T)
{
    int32_t count = 0;
    int i = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
    uint8x16_t t_dup = vdupq_n_u8((uint8_t)T);
    uint8x16_t one   = vdupq_n_u8(1);
    uint32x4_t acc   = vdupq_n_u32(0);
    for (; i + 16 <= n_bytes; i += 16) {
        uint8x16_t qv  = vld1q_u8(q + i);
        uint8x16_t tv  = vld1q_u8(t + i);
        uint8x16_t xr  = veorq_u8(qv, tv);
        uint8x16_t cnt = vcntq_u8(xr);              /* per-byte popcount, [0, 8] */
        uint8x16_t msk = vcgtq_u8(cnt, t_dup);      /* 0xFF where cost > T */
        uint8x16_t ind = vandq_u8(msk, one);        /* 1 per mismatching block */
        acc = vaddq_u32(acc, vpaddlq_u16(vpaddlq_u8(ind)));
    }
    count += (int32_t)vaddvq_u32(acc);
#endif
    for (; i < n_bytes; i++) {
        int c = __builtin_popcount((unsigned int)(q[i] ^ t[i]));
        if (c > T) count++;
    }
    return count;
}

/* Calibrate block-threshold T ∈ {0..4} by measuring argmin-class
 * accuracy on a leave-one-out split of a 1000-sample training subset.
 * Returns the best T. Called once at startup. */
static int calibrate_block_threshold(
    const uint8_t* train_sigs, const int* y_train,
    int n_train, int sig_bytes)
{
    int n_sample = n_train < 1000 ? n_train : 1000;
    int best_T = 0;
    int best_correct = -1;
    printf("  Calibrating block_threshold T ∈ {0..4} on %d samples:\n", n_sample);
    for (int T = 0; T <= 4; T++) {
        int correct = 0;
        for (int i = 0; i < n_sample; i++) {
            const uint8_t* qs = train_sigs + (size_t)i * sig_bytes;
            int32_t best_d = INT32_MAX;
            int best_l = -1;
            for (int j = 0; j < n_sample; j++) {
                if (j == i) continue;
                int32_t d = block_threshold_score(
                    qs, train_sigs + (size_t)j * sig_bytes, sig_bytes, T);
                if (d < best_d) { best_d = d; best_l = y_train[j]; }
            }
            if (best_l == y_train[i]) correct++;
        }
        printf("    T=%d: train-subset acc = %.2f%% (%d/%d)\n",
               T, 100.0 * correct / n_sample, correct, n_sample);
        if (correct > best_correct) { best_correct = correct; best_T = T; }
    }
    return best_T;
}

/* E2: global per-dim-weighted ternary Hamming distance.
 * cost(q, t) per trit pair (same as m4t_popcount_dist semantics):
 *   agree (any state)    → 0
 *   ±1 vs 0 / 0 vs ±1    → 1
 *   +1 vs −1 / −1 vs +1  → 2
 * Score = Σ_d w[d] × cost(q[d], t[d]).
 *
 * Key identity for int8-stored trits in {-1, 0, +1}:
 *     cost(q, t) = |q - t|
 * So the full score becomes:
 *     score = Σ_d w[d] × |q[d] - t[d]|
 * which is exactly the substrate's SDOT-shape after a vsubq_s8 + vabsq_s8
 * preprocessing step. Uses vdotq_s32 to 16-way-parallel accumulate the
 * (cost × weight) products into int32.
 *
 * Weights are uint8 in [1, 16]; abs-diff is int8 in [0, 2]. Their
 * product fits in int8 ([0, 32]), so vdotq_s32 on the pair is safe.
 * Requires int8 ternary query/target buffers (already built when
 * --distance sdot or --distance weighted_hamming is selected). */
static int32_t weighted_hamming_score(
    const int8_t* q_i8, const int8_t* t_i8,
    const uint8_t* w_per_dim, int n)
{
    int32_t acc = 0;
    int k = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t vacc = vdupq_n_s32(0);
    for (; k + 16 <= n; k += 16) {
        int8x16_t qv   = vld1q_s8(q_i8 + k);
        int8x16_t tv   = vld1q_s8(t_i8 + k);
        int8x16_t diff = vabsq_s8(vsubq_s8(qv, tv));           /* ∈ {0, 1, 2} */
        int8x16_t wv   = vreinterpretq_s8_u8(vld1q_u8(w_per_dim + k));
        vacc = vdotq_s32(vacc, diff, wv);
    }
    acc = vaddvq_s32(vacc);
#endif
    for (; k < n; k++) {
        int d = (int)q_i8[k] - (int)t_i8[k];
        if (d < 0) d = -d;
        acc += (int32_t)d * (int32_t)w_per_dim[k];
    }
    return acc;
}

#define N_CLASSES 10
#define KNN_K 5
#define TRITS_PER_VOTE 4

typedef struct { int32_t score; int label; } topk_entry_t;

static inline void topk_insert(topk_entry_t* tk, int* n, int k,
                                int32_t score, int label) {
    if (*n < k) {
        int pos = *n;
        while (pos > 0 && tk[pos-1].score > score) { tk[pos] = tk[pos-1]; pos--; }
        tk[pos].score = score; tk[pos].label = label; (*n)++;
    } else if (score < tk[k-1].score) {
        int pos = k - 1;
        while (pos > 0 && tk[pos-1].score > score) { tk[pos] = tk[pos-1]; pos--; }
        tk[pos].score = score; tk[pos].label = label;
    }
}

static inline int topk_vote(const topk_entry_t* tk, int n, int k) {
    int cv[N_CLASSES] = {0};
    for (int i = 0; i < n; i++) cv[tk[i].label] += (k - i);
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++) if (cv[c] > cv[best]) best = c;
    return best;
}

static const int8_t vote_trits[10][TRITS_PER_VOTE] = {
    {-1,-1,-1,-1}, {-1,-1,-1, 0}, {-1,-1,-1,+1}, {-1,-1, 0,-1},
    {-1,-1, 0, 0}, {-1,-1, 0,+1}, {-1,-1,+1,-1}, {-1,-1,+1, 0},
    {-1,-1,+1,+1}, {-1, 0,-1,-1},
};

static void encode_gsh_sig(const int* labels, int n_tables,
                           uint8_t* out, int gsh_sb) {
    memset(out, 0, gsh_sb);
    for (int m = 0; m < n_tables; m++) {
        int lbl = labels[m];
        if (lbl < 0 || lbl >= N_CLASSES) lbl = 0;
        for (int t = 0; t < TRITS_PER_VOTE; t++)
            glyph_write_trit(out, m * TRITS_PER_VOTE + t, vote_trits[lbl][t]);
    }
}

/* Probe infrastructure provided by glyph_probe.h */

/* Extract top-M nearest labels from the union by full-sig Hamming
 * distance. With direct quantization there's one signature per image,
 * not M per-table sigs — so we take the top-M nearest by distance
 * and use their labels as the M-dim routing pattern. */
static void union_top_m_labels(
    const glyph_probe_state_t* st, int M_labels, int sig_bytes,
    const uint8_t* train_sigs, const uint8_t* q_sig,
    const uint8_t* mask, const int* y_train,
    int exclude_idx, int* out_labels)
{
    typedef struct { int32_t d; int label; } dl_t;
    dl_t topk[256];
    int ntk = 0;
    int mlim = (M_labels < 256) ? M_labels : 256;

    for (int j = 0; j < st->n_hit; j++) {
        int idx = st->hit_list[j];
        if (idx == exclude_idx) continue;
        int32_t d = m4t_popcount_dist(
            q_sig, train_sigs + (size_t)idx * sig_bytes, mask, sig_bytes);
        int lbl = y_train[idx];
        if (ntk < mlim) {
            int pos = ntk;
            while (pos > 0 && topk[pos-1].d > d) { topk[pos]=topk[pos-1]; pos--; }
            topk[pos].d = d; topk[pos].label = lbl; ntk++;
        } else if (d < topk[mlim-1].d) {
            int pos = mlim - 1;
            while (pos > 0 && topk[pos-1].d > d) { topk[pos]=topk[pos-1]; pos--; }
            topk[pos].d = d; topk[pos].label = lbl;
        }
    }
    for (int i = 0; i < mlim; i++)
        out_labels[i] = (i < ntk) ? topk[i].label : 0;
}

/* NOTE: trit-native transitions (quantize first, gradients second)
 * were tested and HURT accuracy on all three datasets (CIFAR −10.67pp,
 * Fashion −3.53pp, MNIST −1.37pp). The transitions produce 81-91%
 * zeros because adjacent pixels often share the same ternary state.
 * Hamming distance treats those zeros as dead weight.
 *
 * The float gradient with separate tau calibration (below) preserves
 * continuous magnitude information and is quantized to keep 90%
 * non-zero. This is the correct design for Hamming-distance scoring.
 * SSTT's quantize-first approach works with IG-weighted inverted
 * index scoring but not with uniform Hamming. */
/* Gradient computation provided by glyph_dataset_gradients() */

/* Integer-arithmetic pair-IG weights.
 *
 * Policy: binary float is confined to a one-shot startup LUT — T[i] =
 * round(i × log2(i) × IG_T_SCALE) for i in [0, n_train]. Same shape as
 * the archived m4t_lut_gen.c: float at build/setup, integer at runtime.
 * Per-class-pair IG computation, per-dim scoring, and final quantization
 * to pw[d] ∈ [1, 16] are all integer arithmetic.
 *
 * Math equivalence (see journal entry): with T as above,
 *   h_ab × n_ab × S = T[n_ab] − T[n_a] − T[n_b]
 *   hc   × n_ab × S = Σ_v (T[n_v] − T[va] − T[vb])
 *   IG   × n_ab × S = (h_ab − hc) × n_ab × S
 * The common n_ab × S factor cancels in the final IG/max_IG ratio, so
 * pw[d] = IG_scaled[d] × 15 / max_IG_scaled + 1 matches the double-precision
 * ranking up to last-digit quantization rounding. */

#define IG_T_SCALE (1 << 20)

static int64_t* build_ig_log_table(int n_max) {
    int64_t* T = malloc((size_t)(n_max + 1) * sizeof(int64_t));
    if (!T) return NULL;
    T[0] = 0;
    for (int i = 1; i <= n_max; i++) {
        T[i] = (int64_t)llround((double)i * log2((double)i) * (double)IG_T_SCALE);
    }
    return T;
}

static void build_pair_ig(const uint8_t* train_sigs, const int* y_train,
                          int n_train, int total_dim, int sig_bytes,
                          uint8_t** pair_ig) {
    uint16_t* ig_hot = calloc((size_t)total_dim * 3 * N_CLASSES, sizeof(uint16_t));
    #define IG_HOT(d, v, c) ig_hot[(size_t)(d)*3*N_CLASSES + (size_t)(v)*N_CLASSES + (c)]
    for (int i = 0; i < n_train; i++) {
        int lbl = y_train[i];
        const uint8_t* sig = train_sigs + (size_t)i * sig_bytes;
        for (int d = 0; d < total_dim; d++) {
            int8_t t = glyph_read_trit(sig, d);
            int v = (t < 0) ? 0 : (t == 0) ? 1 : 2;
            IG_HOT(d, v, lbl)++;
        }
    }
    int ig_cc[N_CLASSES] = {0};
    for (int i = 0; i < n_train; i++) ig_cc[y_train[i]]++;

    int64_t* T = build_ig_log_table(n_train);
    int64_t* ig_tmp = malloc((size_t)total_dim * sizeof(int64_t));
    for (int a = 0; a < N_CLASSES; a++) {
        for (int b = a + 1; b < N_CLASSES; b++) {
            uint8_t* pw = malloc((size_t)total_dim);
            int n_ab = ig_cc[a] + ig_cc[b];
            int64_t h_ab_scaled = T[n_ab] - T[ig_cc[a]] - T[ig_cc[b]];
            int64_t max_ig = 0;
            for (int d = 0; d < total_dim; d++) {
                int64_t hc_scaled = 0;
                for (int v = 0; v < 3; v++) {
                    int va = IG_HOT(d, v, a);
                    int vb = IG_HOT(d, v, b);
                    int vt = va + vb;
                    if (!vt) continue;
                    hc_scaled += T[vt] - T[va] - T[vb];
                }
                int64_t ig = h_ab_scaled - hc_scaled;
                if (ig < 0) ig = 0;
                ig_tmp[d] = ig;
                if (ig > max_ig) max_ig = ig;
            }
            for (int d = 0; d < total_dim; d++)
                pw[d] = max_ig > 0
                        ? (uint8_t)(ig_tmp[d] * 15 / max_ig + 1)
                        : 1;
            pair_ig[a*N_CLASSES+b] = pw;
            pair_ig[b*N_CLASSES+a] = pw;
        }
        pair_ig[a*N_CLASSES+a] = NULL;
    }
    free(ig_tmp); free(T); free(ig_hot);
    #undef IG_HOT
}

static void build_spatial_summary(const uint8_t* sigs, int n_imgs, int sig_bytes,
                                  int img_w, int img_h, int n_ch,
                                  int blk_w, int blk_h, int summary_bytes,
                                  uint8_t* summaries) {
    int sum_w = img_w / blk_w;
    int sum_h = img_h / blk_h;
    int ppc = img_w * img_h;
    for (int i = 0; i < n_imgs; i++) {
        const uint8_t* sig = sigs + (size_t)i * sig_bytes;
        uint8_t* sum_sig = summaries + (size_t)i * summary_bytes;
        int si = 0;
        for (int ch = 0; ch < n_ch; ch++) {
            for (int by = 0; by < sum_h; by++) {
                for (int bx = 0; bx < sum_w; bx++) {
                    int pos_count = 0, neg_count = 0;
                    for (int dy = 0; dy < blk_h; dy++) {
                        for (int dx = 0; dx < blk_w; dx++) {
                            int px = bx * blk_w + dx;
                            int py = by * blk_h + dy;
                            if (px >= img_w || py >= img_h) continue;
                            int trit_pos = ch * ppc + py * img_w + px;
                            int8_t t = glyph_read_trit(sig, trit_pos);
                            if (t > 0) pos_count++;
                            else if (t < 0) neg_count++;
                        }
                    }
                    int8_t summary_trit = 0;
                    if (pos_count > neg_count) summary_trit = +1;
                    else if (neg_count > pos_count) summary_trit = -1;
                    glyph_write_trit(sum_sig, si, summary_trit);
                    si++;
                }
            }
        }
    }
}

static void print_emission_coverage(const uint8_t* train_sigs, int n_train,
                                    int sig_bytes, int total_dim,
                                    int intensity_dim, int use_gradients) {
    long n_pos = 0, n_neg = 0, n_zero = 0;
    int sample_n = (n_train < 1000) ? n_train : 1000;
    for (int i = 0; i < sample_n; i++) {
        const uint8_t* sig = train_sigs + (size_t)i * sig_bytes;
        for (int d = 0; d < total_dim; d++) {
            int8_t t = glyph_read_trit(sig, d);
            if (t > 0) n_pos++;
            else if (t < 0) n_neg++;
            else n_zero++;
        }
    }
    long sampled = (long)sample_n * total_dim;
    printf("  Emission coverage (first %d images):\n", sample_n);
    printf("    +1: %.1f%%  0: %.1f%%  -1: %.1f%%\n",
           100.0 * n_pos / sampled,
           100.0 * n_zero / sampled,
           100.0 * n_neg / sampled);
    if (use_gradients) {
        long ip = 0, iz = 0, in_ = 0, gp = 0, gz = 0, gn = 0;
        for (int i = 0; i < sample_n; i++) {
            const uint8_t* sig = train_sigs + (size_t)i * sig_bytes;
            for (int d = 0; d < intensity_dim; d++) {
                int8_t t = glyph_read_trit(sig, d);
                if (t > 0) ip++; else if (t < 0) in_++; else iz++;
            }
            for (int d = intensity_dim; d < total_dim; d++) {
                int8_t t = glyph_read_trit(sig, d);
                if (t > 0) gp++; else if (t < 0) gn++; else gz++;
            }
        }
        long it = (long)sample_n * intensity_dim;
        long gt = (long)sample_n * (total_dim - intensity_dim);
        printf("    intensity: +1=%.1f%% 0=%.1f%% -1=%.1f%%\n",
               100.0*ip/it, 100.0*iz/it, 100.0*in_/it);
        printf("    gradient:  +1=%.1f%% 0=%.1f%% -1=%.1f%%\n",
               100.0*gp/gt, 100.0*gz/gt, 100.0*gn/gt);
    }
}

int main(int argc, char** argv) {
    /* Strip --gradients and --distance before glyph_config sees them. */
    int use_gradients = 0;
    int use_multiscale = 0; /* S1: append 2× downsampled intensity + gradients */
    int use_multiscale4 = 0; /* S1 extension: also append 4× downsampled */
    double grad_density = 0.10; /* gradient-channel tau density; CLI --grad_density */
    int region_tau = 0; /* S4: 0 = global tau; N > 0 = N×N per-region tau grid */
    int region_tau_auto = 0; /* --region_tau auto: heuristic-selected R based on spatial class-variance */
    int fstat_K = 0;        /* S7 fallback: select top-K dims by F-stat (0 = disabled) */
    int use_brute_1nn = 0;  /* S7 control: bypass filter/resolver, use brute-force 1-NN */
    const char* dump_preds_path = NULL; /* optional per-query prediction dump */
    int use_sdot = 0;      /* E1 distance-function experiment: --distance sdot */
    int use_weighted = 0;  /* E2 distance-function experiment: --distance weighted_hamming */
    int use_block = 0;     /* E3 distance-function experiment: --distance block_threshold */
    int block_T_override = -1; /* -1 = calibrate; ≥0 = use given T directly */
    int new_argc = 0;
    char** new_argv = malloc((size_t)argc * sizeof(char*));
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--gradients") == 0) { use_gradients = 1; continue; }
        if (strcmp(argv[i], "--multi_scale") == 0) { use_multiscale = 1; continue; }
        if (strcmp(argv[i], "--multi_scale4") == 0) { use_multiscale = 1; use_multiscale4 = 1; continue; }
        if (strcmp(argv[i], "--grad_density") == 0 && i + 1 < argc) {
            grad_density = atof(argv[i+1]);
            if (grad_density < 0.001) grad_density = 0.001;
            if (grad_density > 0.999) grad_density = 0.999;
            i++; continue;
        }
        if (strcmp(argv[i], "--region_tau") == 0 && i + 1 < argc) {
            if (strcmp(argv[i+1], "auto") == 0) {
                region_tau_auto = 1;
                region_tau = 0; /* will be set by heuristic */
            } else {
                region_tau = atoi(argv[i+1]);
                if (region_tau < 0) region_tau = 0;
            }
            i++; continue;
        }
        if (strcmp(argv[i], "--dump_preds") == 0 && i + 1 < argc) {
            dump_preds_path = argv[i+1];
            i++; continue;
        }
        if (strcmp(argv[i], "--fstat_K") == 0 && i + 1 < argc) {
            fstat_K = atoi(argv[i+1]);
            if (fstat_K < 0) fstat_K = 0;
            i++; continue;
        }
        if (strcmp(argv[i], "--brute_1nn") == 0) {
            use_brute_1nn = 1; continue;
        }
        if (strcmp(argv[i], "--distance") == 0 && i + 1 < argc) {
            if (strcmp(argv[i+1], "sdot") == 0) use_sdot = 1;
            else if (strcmp(argv[i+1], "weighted_hamming") == 0) use_weighted = 1;
            else if (strcmp(argv[i+1], "block_threshold") == 0) use_block = 1;
            /* "hamming" is the default; other values silently ignored */
            i++; continue;
        }
        if (strcmp(argv[i], "--block_threshold_T") == 0 && i + 1 < argc) {
            block_T_override = atoi(argv[i+1]);
            i++; continue;
        }
        new_argv[new_argc++] = argv[i];
    }

    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, new_argc, new_argv);
    free(new_argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, cfg.data_dir) != 0) return 1;
    if (!cfg.no_deskew) glyph_dataset_deskew(&ds);
    if (cfg.normalize) glyph_dataset_normalize(&ds);

    int n_ch = (ds.input_dim > 784) ? 3 : 1;
    int img_w = ds.img_w > 0 ? ds.img_w : (n_ch == 3 ? 32 : 28);
    int img_h = ds.img_h > 0 ? ds.img_h : (n_ch == 3 ? 32 : 28);

    int intensity_dim = ds.input_dim;
    int hgrad_dim = n_ch * img_h * (img_w - 1);
    int vgrad_dim = n_ch * (img_h - 1) * img_w;
    /* S1 multi-scale: 2× downsampled intensity + gradients of that
     * downsampled image. Only enabled when --multi_scale flag is set.
     * Requires img_w and img_h even (true for MNIST 28×28 and CIFAR 32×32). */
    int ms_w = img_w / 2;
    int ms_h = img_h / 2;
    int ms_intensity_dim = use_multiscale ? (n_ch * ms_h * ms_w) : 0;
    int ms_hgrad_dim     = use_multiscale ? (n_ch * ms_h * (ms_w - 1)) : 0;
    int ms_vgrad_dim     = use_multiscale ? (n_ch * (ms_h - 1) * ms_w) : 0;
    int ms4_w = img_w / 4;
    int ms4_h = img_h / 4;
    int ms4_intensity_dim = use_multiscale4 ? (n_ch * ms4_h * ms4_w) : 0;
    int ms4_hgrad_dim     = use_multiscale4 ? (n_ch * ms4_h * (ms4_w - 1)) : 0;
    int ms4_vgrad_dim     = use_multiscale4 ? (n_ch * (ms4_h - 1) * ms4_w) : 0;
    int total_dim = intensity_dim
                  + (use_gradients   ? (hgrad_dim + vgrad_dim) : 0)
                  + (use_multiscale  ? (ms_intensity_dim + ms_hgrad_dim + ms_vgrad_dim) : 0)
                  + (use_multiscale4 ? (ms4_intensity_dim + ms4_hgrad_dim + ms4_vgrad_dim) : 0);
    int sig_bytes = M4T_TRIT_PACKED_BYTES(total_dim);

    printf("direct_lsh: direct ternary quantization + Trit Lattice LSH\n");
    printf("  distance=%s\n",
           use_sdot     ? "sdot (int8 ternary inner-product; E1)" :
           use_weighted ? "weighted_hamming (global per-dim weights; E2)" :
           use_block    ? "block_threshold (4-trit block, T-threshold count; E3)" :
                          "hamming (popcount_dist)");
    printf("  data=%s  deskew=%s  gradients=%s  multi_scale=%s\n",
           cfg.data_dir, cfg.no_deskew ? "off" : "on",
           use_gradients ? "on" : "off",
           use_multiscale ? "on" : "off");
    printf("  image: %dx%dx%d  intensity_dim=%d\n", img_w, img_h, n_ch, intensity_dim);
    if (use_gradients)
        printf("  hgrad_dim=%d  vgrad_dim=%d\n", hgrad_dim, vgrad_dim);
    if (use_multiscale)
        printf("  ms_intensity_dim=%d  ms_hgrad_dim=%d  ms_vgrad_dim=%d  (2× downsampled)\n",
               ms_intensity_dim, ms_hgrad_dim, ms_vgrad_dim);
    if (use_multiscale4)
        printf("  ms4_intensity_dim=%d  ms4_hgrad_dim=%d  ms4_vgrad_dim=%d  (4× downsampled)\n",
               ms4_intensity_dim, ms4_hgrad_dim, ms4_vgrad_dim);
    printf("  total_dim=%d\n", total_dim);
    /* NOTE: --density for direct quantization means "fraction of pixel
     * values that map to zero (structural zero)." This is different from
     * the random-projection meaning ("fraction of projection WEIGHTS
     * that are zero"). For normalized CIFAR-10, --density 0.395
     * produces tau≈0.6×SCALE which matches the empirically optimal
     * threshold. */
    printf("  sig_bytes=%d  density=%.3f (%.1f%% of intensity trits will be zero)\n",
           sig_bytes, cfg.density, cfg.density * 100.0);

    /* Multi-table: each table uses a DIFFERENT permutation of the
     * first 16 trits as its bucket key. The full signature is shared;
     * only the key subset differs per table. */
    const int M = cfg.m_max;
    const int KEY_TRITS = 16;
    printf("  M=%d tables (different 16-trit key subsets)  knn_k=%d\n",
           M, KNN_K);
    printf("  n_train=%d  n_test=%d\n\n", ds.n_train, ds.n_test);

    clock_t t0 = clock();

    /* Build feature vectors: intensity + optional float gradients. */
    printf("Building feature vectors...\n");
    m4t_mtfp_t* train_feat = malloc((size_t)ds.n_train * total_dim * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* test_feat  = malloc((size_t)ds.n_test  * total_dim * sizeof(m4t_mtfp_t));

    /* Temporary buffers for gradient + multi-scale computation. */
    m4t_mtfp_t* hg    = use_gradients  ? malloc((size_t)hgrad_dim * sizeof(m4t_mtfp_t))    : NULL;
    m4t_mtfp_t* vg    = use_gradients  ? malloc((size_t)vgrad_dim * sizeof(m4t_mtfp_t))    : NULL;
    m4t_mtfp_t* ms_im = use_multiscale ? malloc((size_t)ms_intensity_dim * sizeof(m4t_mtfp_t)) : NULL;
    m4t_mtfp_t* ms_hg = use_multiscale ? malloc((size_t)ms_hgrad_dim * sizeof(m4t_mtfp_t)) : NULL;
    m4t_mtfp_t* ms_vg = use_multiscale ? malloc((size_t)ms_vgrad_dim * sizeof(m4t_mtfp_t)) : NULL;
    m4t_mtfp_t* ms4_im = use_multiscale4 ? malloc((size_t)ms4_intensity_dim * sizeof(m4t_mtfp_t)) : NULL;
    m4t_mtfp_t* ms4_hg = use_multiscale4 ? malloc((size_t)ms4_hgrad_dim * sizeof(m4t_mtfp_t)) : NULL;
    m4t_mtfp_t* ms4_vg = use_multiscale4 ? malloc((size_t)ms4_vgrad_dim * sizeof(m4t_mtfp_t)) : NULL;

    for (int pass = 0; pass < 2; pass++) {
        int n_imgs = (pass == 0) ? ds.n_train : ds.n_test;
        const m4t_mtfp_t* src_imgs = (pass == 0) ? ds.x_train : ds.x_test;
        m4t_mtfp_t* dst_feat       = (pass == 0) ? train_feat : test_feat;
        for (int i = 0; i < n_imgs; i++) {
            const m4t_mtfp_t* img = src_imgs + (size_t)i * ds.input_dim;
            m4t_mtfp_t* out = dst_feat + (size_t)i * total_dim;
            int off = 0;
            memcpy(out + off, img, (size_t)intensity_dim * sizeof(m4t_mtfp_t));
            off += intensity_dim;
            if (use_gradients) {
                glyph_dataset_gradients(img, img_w, img_h, n_ch, hg, vg);
                memcpy(out + off, hg, (size_t)hgrad_dim * sizeof(m4t_mtfp_t)); off += hgrad_dim;
                memcpy(out + off, vg, (size_t)vgrad_dim * sizeof(m4t_mtfp_t)); off += vgrad_dim;
            }
            if (use_multiscale) {
                downsample_2x_mtfp(img, ms_im, img_w, img_h, n_ch);
                memcpy(out + off, ms_im, (size_t)ms_intensity_dim * sizeof(m4t_mtfp_t));
                off += ms_intensity_dim;
                glyph_dataset_gradients(ms_im, ms_w, ms_h, n_ch, ms_hg, ms_vg);
                memcpy(out + off, ms_hg, (size_t)ms_hgrad_dim * sizeof(m4t_mtfp_t)); off += ms_hgrad_dim;
                memcpy(out + off, ms_vg, (size_t)ms_vgrad_dim * sizeof(m4t_mtfp_t)); off += ms_vgrad_dim;
            }
            if (use_multiscale4) {
                /* Chain 2× → 4× from the already-computed ms_im buffer. */
                downsample_2x_mtfp(ms_im, ms4_im, ms_w, ms_h, n_ch);
                memcpy(out + off, ms4_im, (size_t)ms4_intensity_dim * sizeof(m4t_mtfp_t));
                off += ms4_intensity_dim;
                glyph_dataset_gradients(ms4_im, ms4_w, ms4_h, n_ch, ms4_hg, ms4_vg);
                memcpy(out + off, ms4_hg, (size_t)ms4_hgrad_dim * sizeof(m4t_mtfp_t)); off += ms4_hgrad_dim;
                memcpy(out + off, ms4_vg, (size_t)ms4_vgrad_dim * sizeof(m4t_mtfp_t)); off += ms4_vgrad_dim;
            }
        }
    }
    free(hg); free(vg); free(ms_im); free(ms_hg); free(ms_vg);
    free(ms4_im); free(ms4_hg); free(ms4_vg);

    /* S7 fallback: F-statistic dim selection. Compute per-dim F-stat
     * from training features, select top-K most class-discriminative,
     * compact feature buffers to only those dims.
     *
     * F-stat = (between-class variance) / (within-class variance).
     * Computed in one pass over training features (unquantized MTFP values).
     * No gradient training — this is class-conditional feature selection,
     * not end-to-end learning. Tests whether uniform-dim selection is
     * leaving signal on the table, as the closest in-C proxy to the
     * question "does learned encoding help?". */
    int* fstat_selected = NULL;
    if (fstat_K > 0 && fstat_K < total_dim) {
        printf("F-stat encoder: selecting top-%d of %d dims...\n", fstat_K, total_dim);
        int n_sample = (ds.n_train < 5000) ? ds.n_train : 5000;
        double* class_sum = calloc((size_t)N_CLASSES * total_dim, sizeof(double));
        double* class_sqsum = calloc((size_t)N_CLASSES * total_dim, sizeof(double));
        int class_cnt[N_CLASSES] = {0};
        for (int i = 0; i < n_sample; i++) {
            int lbl = ds.y_train[i];
            if (lbl < 0 || lbl >= N_CLASSES) continue;
            class_cnt[lbl]++;
            const m4t_mtfp_t* f = train_feat + (size_t)i * total_dim;
            double* cs = class_sum + (size_t)lbl * total_dim;
            double* cq = class_sqsum + (size_t)lbl * total_dim;
            for (int d = 0; d < total_dim; d++) {
                double v = (double)f[d];
                cs[d] += v;
                cq[d] += v * v;
            }
        }
        /* F-stat per dim. */
        double* fstat = malloc((size_t)total_dim * sizeof(double));
        for (int d = 0; d < total_dim; d++) {
            double grand_mean = 0;
            int grand_n = 0;
            for (int c = 0; c < N_CLASSES; c++) {
                grand_mean += class_sum[(size_t)c * total_dim + d];
                grand_n += class_cnt[c];
            }
            if (grand_n == 0) { fstat[d] = 0; continue; }
            grand_mean /= grand_n;
            double between = 0, within = 0;
            for (int c = 0; c < N_CLASSES; c++) {
                if (class_cnt[c] == 0) continue;
                double cm = class_sum[(size_t)c * total_dim + d] / class_cnt[c];
                double cv = class_sqsum[(size_t)c * total_dim + d] / class_cnt[c] - cm * cm;
                between += class_cnt[c] * (cm - grand_mean) * (cm - grand_mean);
                within  += class_cnt[c] * (cv > 0 ? cv : 0);
            }
            fstat[d] = (within > 1e-9) ? (between / within) : 0;
        }
        free(class_sum); free(class_sqsum);
        /* Select top-K by F-stat: full sort descending, take top K.
         * total_dim ≤ 12000 so O(N log N) is fine. */
        fstat_selected = malloc((size_t)fstat_K * sizeof(int));
        fstat_entry_t* fd = malloc((size_t)total_dim * sizeof(fstat_entry_t));
        for (int d = 0; d < total_dim; d++) { fd[d].f = fstat[d]; fd[d].d = d; }
        qsort(fd, total_dim, sizeof(fstat_entry_t), fstat_cmp_desc);
        double f_top = fd[0].f, f_kth = fd[fstat_K-1].f, f_bottom = fd[total_dim-1].f;
        for (int k = 0; k < fstat_K; k++) fstat_selected[k] = fd[k].d;
        free(fd); free(fstat);
        printf("  F-stat range: top=%.3f  Kth=%.3f  bottom=%.6f\n", f_top, f_kth, f_bottom);
        /* Compact train_feat and test_feat to only the selected K dims. */
        int new_total = fstat_K;
        m4t_mtfp_t* nt = malloc((size_t)ds.n_train * new_total * sizeof(m4t_mtfp_t));
        m4t_mtfp_t* ne = malloc((size_t)ds.n_test  * new_total * sizeof(m4t_mtfp_t));
        for (int i = 0; i < ds.n_train; i++) {
            const m4t_mtfp_t* src = train_feat + (size_t)i * total_dim;
            m4t_mtfp_t* dst = nt + (size_t)i * new_total;
            for (int k = 0; k < fstat_K; k++) dst[k] = src[fstat_selected[k]];
        }
        for (int i = 0; i < ds.n_test; i++) {
            const m4t_mtfp_t* src = test_feat + (size_t)i * total_dim;
            m4t_mtfp_t* dst = ne + (size_t)i * new_total;
            for (int k = 0; k < fstat_K; k++) dst[k] = src[fstat_selected[k]];
        }
        free(train_feat); free(test_feat);
        train_feat = nt; test_feat = ne;
        total_dim = new_total;
        sig_bytes = M4T_TRIT_PACKED_BYTES(total_dim);
        /* Intensity and gradient boundary info is now invalidated for
         * downstream quantization — we'll use global tau_intensity for
         * all selected dims. The per-channel split no longer applies. */
        intensity_dim = total_dim;
        hgrad_dim = vgrad_dim = 0;
        ms_intensity_dim = ms_hgrad_dim = ms_vgrad_dim = 0;
        ms4_intensity_dim = ms4_hgrad_dim = ms4_vgrad_dim = 0;
        use_gradients = 0;  /* turn off channel-aware quantization */
        use_multiscale = use_multiscale4 = 0;
        printf("  F-stat encoder: total_dim=%d  sig_bytes=%d\n", total_dim, sig_bytes);
    }

    /* Calibrate tau: separate thresholds for intensity and gradients.
     * Extract into contiguous buffers for correct stride. */
    int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;
    m4t_mtfp_t* intensity_sample = malloc((size_t)n_calib * intensity_dim * sizeof(m4t_mtfp_t));
    for (int i = 0; i < n_calib; i++)
        memcpy(intensity_sample + (size_t)i * intensity_dim,
               train_feat + (size_t)i * total_dim,
               (size_t)intensity_dim * sizeof(m4t_mtfp_t));
    int64_t tau_intensity = glyph_sig_quantize_tau(
        intensity_sample, n_calib, intensity_dim, cfg.density);
    free(intensity_sample);

    int64_t tau_gradient = 0;
    if (use_gradients) {
        int grad_dim = hgrad_dim + vgrad_dim;
        m4t_mtfp_t* grad_sample = malloc((size_t)n_calib * grad_dim * sizeof(m4t_mtfp_t));
        for (int i = 0; i < n_calib; i++)
            memcpy(grad_sample + (size_t)i * grad_dim,
                   train_feat + (size_t)i * total_dim + intensity_dim,
                   (size_t)grad_dim * sizeof(m4t_mtfp_t));
        tau_gradient = glyph_sig_quantize_tau(grad_sample, n_calib, grad_dim, grad_density);
        free(grad_sample);
    }
    /* Multi-scale uses the SAME tau for intensity and gradient channels
     * as the base scale — the distribution of pixel intensities and
     * gradients is similar across scales under the 2×2-block average. */

    /* S4 auto-gating: EXPERIMENTAL. Reports per-class COM spread as a
     * diagnostic but does NOT reliably predict whether R4 helps.
     *
     * Mechanism depth problem: R4 helps when different classes produce
     * different per-region pixel distributions (e.g., CIFAR sky-background
     * classes vs ground-subject classes differ in per-region tau).
     * R4 hurts when all classes share a common layout (Fashion garments
     * centered on uniform black background). Simple scalar heuristics
     * (COM spread, tau spread, inter-class variance) fail to separate
     * these regimes — COM spread ranks Fashion HIGHER than CIFAR because
     * Fashion garments have distinct shapes, even though R4 hurts Fashion.
     *
     * Deployment guidance: default R=0. Measure empirically on your
     * dataset: compare `--region_tau 4` vs `--region_tau 0` Selective
     * accuracy. Enable R4 only where it helps. CIFAR-10-class natural
     * images tend to benefit; centered-object datasets tend not to. */
    if (region_tau_auto) {
        int probe_R = 4;
        /* Per-class center-of-mass (intensity-weighted). Use absolute value
         * of normalized pixel values so dark/bright backgrounds contribute
         * uniformly. */
        double cx[N_CLASSES] = {0}, cy[N_CLASSES] = {0}, wsum[N_CLASSES] = {0};
        for (int i = 0; i < n_calib; i++) {
            int lbl = ds.y_train[i];
            if (lbl < 0 || lbl >= N_CLASSES) continue;
            const m4t_mtfp_t* feat = train_feat + (size_t)i * total_dim;
            for (int c = 0; c < n_ch; c++) {
                for (int y = 0; y < img_h; y++) {
                    for (int x = 0; x < img_w; x++) {
                        int64_t v = (int64_t)feat[c * img_h * img_w + y * img_w + x];
                        if (v < 0) v = -v;
                        cx[lbl]   += (double)x * (double)v;
                        cy[lbl]   += (double)y * (double)v;
                        wsum[lbl] += (double)v;
                    }
                }
            }
        }
        double mean_cx = 0, mean_cy = 0;
        int n_classes_observed = 0;
        for (int k = 0; k < N_CLASSES; k++) {
            if (wsum[k] > 0) {
                cx[k] /= wsum[k];
                cy[k] /= wsum[k];
                mean_cx += cx[k];
                mean_cy += cy[k];
                n_classes_observed++;
            }
        }
        mean_cx /= n_classes_observed;
        mean_cy /= n_classes_observed;
        double var_cx = 0, var_cy = 0;
        for (int k = 0; k < N_CLASSES; k++) {
            if (wsum[k] > 0) {
                double dx = cx[k] - mean_cx;
                double dy = cy[k] - mean_cy;
                var_cx += dx * dx;
                var_cy += dy * dy;
            }
        }
        var_cx /= n_classes_observed;
        var_cy /= n_classes_observed;
        double dim_mean = 0.5 * (img_w + img_h);
        double com_spread = sqrt(var_cx + var_cy) / dim_mean;
        printf("  --region_tau auto: per-class COM spread=%.4f  [σcx=%.2f σcy=%.2f img=%dx%d]\n",
               com_spread, sqrt(var_cx), sqrt(var_cy), img_w, img_h);
        printf("  --region_tau auto: metric is EXPERIMENTAL and does not reliably\n"
               "                     predict R4 benefit. Defaulting to DISABLED.\n"
               "                     Opt in with --region_tau N after empirical check.\n");
        region_tau = 0;
        (void)probe_R;
    }

    /* S4: per-region tau arrays for the base intensity and gradient
     * channels. Region = (img_h/R)×(img_w/R) block of the image; each
     * region gets its own density-percentile tau computed from the
     * calibration subset. Keeps multi-scale channels on global tau. */
    int R_regions = region_tau;
    int n_regions = R_regions * R_regions;
    int64_t* rtau_int  = NULL;
    int64_t* rtau_grad = NULL;
    if (R_regions > 0) {
        rtau_int = malloc((size_t)n_regions * sizeof(int64_t));
        for (int r = 0; r < n_regions; r++) rtau_int[r] = tau_intensity;
        /* Max region footprint must cover BOTH intensity and
         * (hgrad + vgrad) per region, across n_calib samples.
         * Gradient layout doubles the per-region element count vs
         * intensity alone. Use the full per-image dim budget divided
         * by region count plus a slack margin. */
        int max_per_region_img = (intensity_dim
                                  + (use_gradients ? (hgrad_dim + vgrad_dim) : 0))
                                / n_regions + 8 * n_ch;
        m4t_mtfp_t* rbuf = malloc((size_t)n_calib * max_per_region_img * sizeof(m4t_mtfp_t));
        for (int rrow = 0; rrow < R_regions; rrow++) {
            for (int rcol = 0; rcol < R_regions; rcol++) {
                int cnt = 0;
                for (int i = 0; i < n_calib; i++) {
                    const m4t_mtfp_t* feat = train_feat + (size_t)i * total_dim;
                    for (int c = 0; c < n_ch; c++) {
                        for (int y = 0; y < img_h; y++) {
                            int ry = (y * R_regions) / img_h;
                            if (ry != rrow) continue;
                            for (int x = 0; x < img_w; x++) {
                                int rx = (x * R_regions) / img_w;
                                if (rx != rcol) continue;
                                rbuf[cnt++] = feat[c * img_h * img_w + y * img_w + x];
                            }
                        }
                    }
                }
                rtau_int[rrow * R_regions + rcol] =
                    glyph_sig_quantize_tau(rbuf, 1, cnt, cfg.density);
            }
        }
        if (use_gradients) {
            rtau_grad = malloc((size_t)n_regions * sizeof(int64_t));
            for (int r = 0; r < n_regions; r++) rtau_grad[r] = tau_gradient;
            /* Gradients: hgrad at (c, y, x) for y∈[0,H), x∈[0,W-1); vgrad at
             * (c, y, x) for y∈[0,H-1), x∈[0,W). Region mapping uses the
             * gradient's own extent so regions balance. */
            for (int rrow = 0; rrow < R_regions; rrow++) {
                for (int rcol = 0; rcol < R_regions; rcol++) {
                    int cnt = 0;
                    for (int i = 0; i < n_calib; i++) {
                        const m4t_mtfp_t* feat =
                            train_feat + (size_t)i * total_dim + intensity_dim;
                        /* hgrad */
                        for (int c = 0; c < n_ch; c++) {
                            for (int y = 0; y < img_h; y++) {
                                int ry = (y * R_regions) / img_h;
                                if (ry != rrow) continue;
                                for (int x = 0; x < img_w - 1; x++) {
                                    int rx = (x * R_regions) / (img_w - 1);
                                    if (rx != rcol) continue;
                                    rbuf[cnt++] =
                                        feat[c * img_h * (img_w-1) + y * (img_w-1) + x];
                                }
                            }
                        }
                        /* vgrad */
                        const m4t_mtfp_t* vgp = feat + hgrad_dim;
                        for (int c = 0; c < n_ch; c++) {
                            for (int y = 0; y < img_h - 1; y++) {
                                int ry = (y * R_regions) / (img_h - 1);
                                if (ry != rrow) continue;
                                for (int x = 0; x < img_w; x++) {
                                    int rx = (x * R_regions) / img_w;
                                    if (rx != rcol) continue;
                                    rbuf[cnt++] =
                                        vgp[c * (img_h-1) * img_w + y * img_w + x];
                                }
                            }
                        }
                    }
                    rtau_grad[rrow * R_regions + rcol] =
                        glyph_sig_quantize_tau(rbuf, 1, cnt, grad_density);
                }
            }
        }
        free(rbuf);
        printf("  region_tau R=%d calibrated (%d regions):\n", R_regions, n_regions);
        printf("    intensity tau range: %lld .. %lld\n",
               (long long)rtau_int[0], (long long)rtau_int[n_regions-1]);
        if (rtau_grad)
            printf("    gradient tau range:  %lld .. %lld\n",
                   (long long)rtau_grad[0], (long long)rtau_grad[n_regions-1]);
    }
    printf("  tau_intensity=%lld (%.3f × SCALE)  tau_gradient=%lld (%.3f × SCALE)\n",
           (long long)tau_intensity, (double)tau_intensity / M4T_MTFP_SCALE,
           (long long)tau_gradient, (double)tau_gradient / M4T_MTFP_SCALE);

    /* Quantize all images to trit signatures. */
    printf("Quantizing signatures (%d trits = %d bytes)...\n", total_dim, sig_bytes);
    uint8_t* train_sigs = calloc((size_t)ds.n_train * sig_bytes, 1);
    uint8_t* test_sigs  = calloc((size_t)ds.n_test  * sig_bytes, 1);

    for (int pass = 0; pass < 2; pass++) {
        int n_imgs = (pass == 0) ? ds.n_train : ds.n_test;
        const m4t_mtfp_t* feat = (pass == 0) ? train_feat : test_feat;
        uint8_t* sigs = (pass == 0) ? train_sigs : test_sigs;
        for (int i = 0; i < n_imgs; i++) {
            const m4t_mtfp_t* f = feat + (size_t)i * total_dim;
            uint8_t* sig = sigs + (size_t)i * sig_bytes;
            int off = 0;
            /* Base intensity — per-region tau if S4 enabled. */
            if (R_regions > 0) {
                for (int c = 0; c < n_ch; c++) {
                    for (int y = 0; y < img_h; y++) {
                        int ry = (y * R_regions) / img_h;
                        for (int x = 0; x < img_w; x++) {
                            int rx = (x * R_regions) / img_w;
                            int64_t tau_r = rtau_int[ry * R_regions + rx];
                            int pos = off + c * img_h * img_w + y * img_w + x;
                            int64_t v = (int64_t)f[pos];
                            if (v >  tau_r) glyph_write_trit(sig, pos, +1);
                            else if (v < -tau_r) glyph_write_trit(sig, pos, -1);
                        }
                    }
                }
            } else {
                for (int d = 0; d < intensity_dim; d++) {
                    int64_t v = (int64_t)f[off + d];
                    if (v > tau_intensity) glyph_write_trit(sig, off + d, +1);
                    else if (v < -tau_intensity) glyph_write_trit(sig, off + d, -1);
                }
            }
            off += intensity_dim;
            /* Base gradients — per-region tau if S4 enabled. */
            if (use_gradients) {
                if (R_regions > 0 && rtau_grad) {
                    for (int c = 0; c < n_ch; c++) {
                        for (int y = 0; y < img_h; y++) {
                            int ry = (y * R_regions) / img_h;
                            for (int x = 0; x < img_w - 1; x++) {
                                int rx = (x * R_regions) / (img_w - 1);
                                int64_t tau_r = rtau_grad[ry * R_regions + rx];
                                int pos = off + c * img_h * (img_w-1) + y * (img_w-1) + x;
                                int64_t v = (int64_t)f[pos];
                                if (v >  tau_r) glyph_write_trit(sig, pos, +1);
                                else if (v < -tau_r) glyph_write_trit(sig, pos, -1);
                            }
                        }
                    }
                    int voff = off + hgrad_dim;
                    for (int c = 0; c < n_ch; c++) {
                        for (int y = 0; y < img_h - 1; y++) {
                            int ry = (y * R_regions) / (img_h - 1);
                            for (int x = 0; x < img_w; x++) {
                                int rx = (x * R_regions) / img_w;
                                int64_t tau_r = rtau_grad[ry * R_regions + rx];
                                int pos = voff + c * (img_h-1) * img_w + y * img_w + x;
                                int64_t v = (int64_t)f[pos];
                                if (v >  tau_r) glyph_write_trit(sig, pos, +1);
                                else if (v < -tau_r) glyph_write_trit(sig, pos, -1);
                            }
                        }
                    }
                } else {
                    int grad_dim = hgrad_dim + vgrad_dim;
                    for (int d = 0; d < grad_dim; d++) {
                        int64_t v = (int64_t)f[off + d];
                        if (v > tau_gradient) glyph_write_trit(sig, off + d, +1);
                        else if (v < -tau_gradient) glyph_write_trit(sig, off + d, -1);
                    }
                }
                off += hgrad_dim + vgrad_dim;
            }
            /* Multi-scale (2× downsampled) intensity + gradients. */
            if (use_multiscale) {
                for (int d = 0; d < ms_intensity_dim; d++) {
                    int64_t v = (int64_t)f[off + d];
                    if (v > tau_intensity) glyph_write_trit(sig, off + d, +1);
                    else if (v < -tau_intensity) glyph_write_trit(sig, off + d, -1);
                }
                off += ms_intensity_dim;
                int ms_grad_dim = ms_hgrad_dim + ms_vgrad_dim;
                for (int d = 0; d < ms_grad_dim; d++) {
                    int64_t v = (int64_t)f[off + d];
                    if (v > tau_gradient) glyph_write_trit(sig, off + d, +1);
                    else if (v < -tau_gradient) glyph_write_trit(sig, off + d, -1);
                }
                off += ms_grad_dim;
            }
            /* Multi-scale 4× downsampled intensity + gradients. */
            if (use_multiscale4) {
                for (int d = 0; d < ms4_intensity_dim; d++) {
                    int64_t v = (int64_t)f[off + d];
                    if (v > tau_intensity) glyph_write_trit(sig, off + d, +1);
                    else if (v < -tau_intensity) glyph_write_trit(sig, off + d, -1);
                }
                off += ms4_intensity_dim;
                int ms4_grad_dim = ms4_hgrad_dim + ms4_vgrad_dim;
                for (int d = 0; d < ms4_grad_dim; d++) {
                    int64_t v = (int64_t)f[off + d];
                    if (v > tau_gradient) glyph_write_trit(sig, off + d, +1);
                    else if (v < -tau_gradient) glyph_write_trit(sig, off + d, -1);
                }
                off += ms4_grad_dim;
            }
        }
    }
    free(train_feat); free(test_feat);

    /* Build int8 ternary buffers parallel to the packed-trit signatures.
     * Used by:
     *   - Pair-IG re-rank and filtered pair-IG (NEON pair_ig_mismatch_score),
     *     which are on every query path.
     *   - --distance sdot (E1) via sdot_score.
     *   - --distance weighted_hamming (E2) via weighted_hamming_score.
     * Cost: total_dim bytes per image × (n_train + n_test). For CIFAR-10
     * with gradients: ~540 MB total. Acceptable on 16 GB unified memory. */
    int8_t* train_i8 = NULL;
    int8_t* test_i8  = NULL;
    {
        printf("Building int8 ternary buffers (%.1f MB + %.1f MB)...\n",
               (double)ds.n_train * total_dim / (1024.0 * 1024.0),
               (double)ds.n_test  * total_dim / (1024.0 * 1024.0));
        train_i8 = malloc((size_t)ds.n_train * total_dim);
        test_i8  = malloc((size_t)ds.n_test  * total_dim);
        if (!train_i8 || !test_i8) {
            fprintf(stderr, "direct_lsh: failed to allocate int8 ternary buffers\n");
            return 1;
        }
        for (int i = 0; i < ds.n_train; i++) {
            m4t_unpack_trits_1d(train_i8 + (size_t)i * total_dim,
                                train_sigs + (size_t)i * sig_bytes,
                                total_dim);
        }
        for (int i = 0; i < ds.n_test; i++) {
            m4t_unpack_trits_1d(test_i8 + (size_t)i * total_dim,
                                test_sigs + (size_t)i * sig_bytes,
                                total_dim);
        }
    }

    printf("Computing IG weights...\n");
    uint8_t** pair_ig = malloc((size_t)N_CLASSES * N_CLASSES * sizeof(uint8_t*));
    build_pair_ig(train_sigs, ds.y_train, ds.n_train, total_dim, sig_bytes, pair_ig);
    printf("  Pair-IG weights computed for %d pairs.\n", N_CLASSES*(N_CLASSES-1)/2);

    /* E3 threshold calibration: choose T ∈ {0..4} before the sweep. */
    int block_T = block_T_override;
    if (use_block && block_T < 0) {
        block_T = calibrate_block_threshold(train_sigs, ds.y_train,
                                             ds.n_train, sig_bytes);
        printf("  block_threshold T calibrated → T=%d\n", block_T);
    } else if (use_block) {
        printf("  block_threshold T=%d (user-overridden)\n", block_T);
    }

    /* E2: global per-dim weight vector = mean of pair-IG weights across
     * class pairs. Same range as pair-IG weights ([1, 16] per dim).
     * This is the global-approximation-to-pair-IG derivation: directly
     * tests how much of pair-IG's gain comes from per-dim weighting vs
     * per-pair specialization (T2 from distance_function_synthesize). */
    uint8_t* global_w = NULL;
    if (use_weighted) {
        global_w = malloc((size_t)total_dim);
        int n_pairs = N_CLASSES * (N_CLASSES - 1) / 2;
        for (int d = 0; d < total_dim; d++) {
            int32_t acc = 0;
            for (int a = 0; a < N_CLASSES; a++) {
                for (int b = a + 1; b < N_CLASSES; b++) {
                    acc += pair_ig[a * N_CLASSES + b][d];
                }
            }
            int32_t avg = (acc + n_pairs / 2) / n_pairs;
            if (avg < 1) avg = 1;
            if (avg > 16) avg = 16;
            global_w[d] = (uint8_t)avg;
        }
        /* Sparsity diagnostic — distribution of global weights. */
        int wh[17] = {0};
        for (int d = 0; d < total_dim; d++) wh[global_w[d]]++;
        printf("  Global weight histogram [1..16]:");
        for (int i = 1; i <= 16; i++) printf(" %d", wh[i]);
        printf("\n");
    }

    /* Hierarchical Trit Lattice LSH: spatial pooling builds the bucket
     * key by reducing blocks of trits via majority vote.
     *
     * For CIFAR-10 (32×32×3 = 3072 intensity trits):
     *   Level 0: 4×4×n_ch summary trits from 8×8 spatial blocks
     * For MNIST (28×28 = 784 intensity trits):
     *   Level 0: 7×7 summary trits from 4×4 spatial blocks
     *
     * Each summary trit = majority of the block's trits:
     *   more +1 than -1 → +1, more -1 → -1, balanced → 0.
     *
     * The bucket key is the first 16 of the summary trits. Each
     * table uses a different PERMUTATION of the summary trits so
     * different tables key on different spatial regions.
     *
     * No random projections — the key is a spatial summary of
     * directly quantized pixels. */

    /* Compute block summary dimensions. */
    /* Block size for hierarchical summary. Smaller blocks = more
     * summary trits = finer spatial key = better filtering. */
    int blk_w, blk_h;
    if (img_w == 32) { blk_w = 4; blk_h = 4; }      /* CIFAR: 8×8×3=192 summary */
    else             { blk_w = 2; blk_h = 2; }       /* MNIST: 14×14=196 summary */
    int sum_w = img_w / blk_w;
    int sum_h = img_h / blk_h;
    int summary_dim = sum_w * sum_h * n_ch;
    printf("Hierarchical key: %dx%d blocks → %dx%dx%d = %d summary trits\n",
           blk_w, blk_h, sum_w, sum_h, n_ch, summary_dim);

    int summary_bytes = M4T_TRIT_PACKED_BYTES(summary_dim);
    uint8_t* train_summary = calloc((size_t)ds.n_train * summary_bytes, 1);
    uint8_t* test_summary  = calloc((size_t)ds.n_test  * summary_bytes, 1);
    build_spatial_summary(train_sigs, ds.n_train, sig_bytes,
                          img_w, img_h, n_ch, blk_w, blk_h,
                          summary_bytes, train_summary);
    build_spatial_summary(test_sigs, ds.n_test, sig_bytes,
                          img_w, img_h, n_ch, blk_w, blk_h,
                          summary_bytes, test_summary);

    print_emission_coverage(train_sigs, ds.n_train, sig_bytes,
                           total_dim, intensity_dim, use_gradients);
    printf("\n");

    /* Build M bucket tables. Each table uses a different permutation
     * of the summary trits for its 16-trit key. */
    printf("Building %d bucket tables...\n", M);
    glyph_bucket_table_t* tables = calloc((size_t)M, sizeof(glyph_bucket_table_t));
    uint8_t** table_train_keys = calloc((size_t)M, sizeof(uint8_t*));
    uint8_t** table_test_keys  = calloc((size_t)M, sizeof(uint8_t*));

    /* Generate a per-table permutation of summary trit indices. */
    int* perm = malloc((size_t)summary_dim * sizeof(int));

    for (int m = 0; m < M; m++) {
        /* Fisher-Yates shuffle seeded per table for diverse keys. */
        for (int t = 0; t < summary_dim; t++) perm[t] = t;
        glyph_rng_t prng;
        uint32_t ps[4];
        ps[0] = cfg.base_seed[0] + (uint32_t)m * 9973u;
        ps[1] = cfg.base_seed[1] + (uint32_t)m * 7919u;
        ps[2] = cfg.base_seed[2] + (uint32_t)m * 6271u;
        ps[3] = cfg.base_seed[3] + (uint32_t)m * 5381u;
        if ((ps[0]|ps[1]|ps[2]|ps[3]) == 0) ps[0] = 1;
        glyph_rng_seed(&prng, ps[0], ps[1], ps[2], ps[3]);
        for (int t = summary_dim - 1; t > 0; t--) {
            int j = (int)(glyph_rng_next(&prng) % (uint32_t)(t + 1));
            int tmp = perm[t]; perm[t] = perm[j]; perm[j] = tmp;
        }

        table_train_keys[m] = calloc((size_t)ds.n_train * 4, 1);
        table_test_keys[m]  = calloc((size_t)ds.n_test  * 4, 1);

        for (int i = 0; i < ds.n_train; i++) {
            const uint8_t* sum_sig = train_summary + (size_t)i * summary_bytes;
            uint8_t* key = table_train_keys[m] + (size_t)i * 4;
            for (int t = 0; t < KEY_TRITS && t < summary_dim; t++)
                glyph_write_trit(key, t, glyph_read_trit(sum_sig, perm[t]));
        }
        for (int i = 0; i < ds.n_test; i++) {
            const uint8_t* sum_sig = test_summary + (size_t)i * summary_bytes;
            uint8_t* key = table_test_keys[m] + (size_t)i * 4;
            for (int t = 0; t < KEY_TRITS && t < summary_dim; t++)
                glyph_write_trit(key, t, glyph_read_trit(sum_sig, perm[t]));
        }
        glyph_bucket_build(&tables[m], table_train_keys[m], ds.n_train, 4);
    }
    free(perm);

    printf("LSH tables built.\n");

    /* ============================================================
     * GSH: compute training routing signatures via LSH probing,
     * encode as multi-trit vote patterns, build GSH bucket index.
     * ============================================================ */
    const int GSH_NTRITS = M * TRITS_PER_VOTE;
    const int GSH_SB = M4T_TRIT_PACKED_BYTES(GSH_NTRITS);
    printf("Building GSH (%d trits = %d bytes)...\n", GSH_NTRITS, GSH_SB);

    glyph_probe_state_t gsh_build_st = {0};
    gsh_build_st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    gsh_build_st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    gsh_build_st.max_union = cfg.max_union;
    int* vote_labels = malloc((size_t)M * sizeof(int));
    uint8_t* gsh_train = calloc((size_t)ds.n_train * GSH_SB, 1);
    uint8_t gsh_build_scratch[4];
    uint8_t* gsh_build_mask = malloc(sig_bytes);
    memset(gsh_build_mask, 0xFF, sig_bytes);

    for (int i = 0; i < ds.n_train; i++) {
        const uint8_t* q = train_sigs + (size_t)i * sig_bytes;
        glyph_probe_reset(&gsh_build_st);
        for (int m = 0; m < M; m++) {
            const uint8_t* qk = table_train_keys[m] + (size_t)i * 4;
            glyph_probe_table(&tables[m], qk, KEY_TRITS, 4,
                        cfg.max_radius, cfg.min_cands, &gsh_build_st, gsh_build_scratch);
        }
        union_top_m_labels(&gsh_build_st, M, sig_bytes,
                           train_sigs, q, gsh_build_mask,
                           ds.y_train, i, vote_labels);
        encode_gsh_sig(vote_labels, M, gsh_train + (size_t)i * GSH_SB, GSH_SB);

        if ((i + 1) % 10000 == 0)
            printf("  %d/%d training GSH sigs\n", i + 1, ds.n_train);
    }
    free(gsh_build_st.votes); free(gsh_build_st.hit_list);
    free(gsh_build_mask);

    glyph_bucket_table_t gsh_table;
    glyph_bucket_build(&gsh_table, gsh_train, ds.n_train, GSH_SB);
    printf("  GSH: %d distinct buckets.\n", glyph_bucket_count_distinct(&gsh_table));

    double build_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Total build: %.1fs.\n\n", build_sec);

    /* Classify. The resolver scores by Hamming distance on the FULL
     * signature (all total_dim trits), not on the 16-trit key. The
     * bucket key is for FILTERING only. */
    glyph_probe_state_t st = {0};
    st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    st.max_union = cfg.max_union;
    uint8_t key_scratch[4];
    uint8_t* full_mask = malloc(sig_bytes); memset(full_mask, 0xFF, sig_bytes);

    const uint8_t* qs_ptr;

    glyph_union_t u = {0};
    u.y_train = ds.y_train; u.n_classes = N_CLASSES;

    int m_sweep[] = {1, 2, 4, 8, 16, 32, 64};
    int n_sweep = 0;
    for (int i = 0; i < 7; i++) if (m_sweep[i] <= M) n_sweep = i + 1;

    int oracle_c[7]={0}, sum_c[7]={0}, knn_c[7]={0}, maj_c[7]={0};
    long union_sum[7]={0};
    int* final_pred = malloc((size_t)ds.n_test * sizeof(int));
    memset(final_pred, 0xFF, (size_t)ds.n_test * sizeof(int));

    /* Routing footprint experiment counters (Experiment 1). */
    int rf_resolver_miss = 0;
    int rf_route_discriminates = 0;
    int rf_route_tied = 0;
    long rf_droute_correct_sum = 0, rf_droute_wrong_sum = 0;

    /* GSH query-time state. */
    glyph_probe_state_t gst = {0};
    gst.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    gst.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    gst.max_union = cfg.max_union;
    uint8_t* q_gsh = calloc(GSH_SB, 1);
    uint8_t* gsh_mask = malloc(GSH_SB); memset(gsh_mask, 0xFF, GSH_SB);

    int lsh_total_correct = 0, gsh_total_correct = 0;
    int agree_count = 0, agree_correct = 0;
    int disagree_count = 0, disagree_lsh_correct = 0, disagree_gsh_correct = 0;
    int pair_ig_correct = 0, selective_correct = 0;
    int filtered_ig_correct = 0, filtered_selective_correct = 0;

    /* S7 control: brute-force 1-NN classifier, bypassing filter+resolver.
     * Scores each test query against all n_train training signatures via
     * popcount_dist, classifies by argmin. The same int8 buffers built for
     * pair-IG are reused via weighted_hamming_score is NOT the right kernel
     * here; uniform Hamming is correct. */
    if (use_brute_1nn) {
        printf("Brute-force 1-NN classification (bypassing filter+resolver)...\n");
        clock_t t_brute = clock();
        int correct = 0;
        int per_class_total[N_CLASSES] = {0};
        int per_class_correct[N_CLASSES] = {0};
        for (int qi = 0; qi < ds.n_test; qi++) {
            int y = ds.y_test[qi];
            per_class_total[y]++;
            const uint8_t* qs = test_sigs + (size_t)qi * sig_bytes;
            int32_t best = INT32_MAX;
            int best_l = 0;
            for (int j = 0; j < ds.n_train; j++) {
                int32_t d = m4t_popcount_dist(
                    qs, train_sigs + (size_t)j * sig_bytes, full_mask, sig_bytes);
                if (d < best) { best = d; best_l = ds.y_train[j]; }
            }
            if (best_l == y) { correct++; per_class_correct[y]++; }
        }
        double brute_sec = (double)(clock() - t_brute) / CLOCKS_PER_SEC;
        printf("Brute 1-NN sweep: %.1fs (%.1f μs/query)\n",
               brute_sec, 1e6 * brute_sec / ds.n_test);
        printf("\n=== Brute-force 1-NN ===\n");
        printf("  Accuracy: %.2f%% (%d/%d)\n", 100.0 * correct / ds.n_test, correct, ds.n_test);
        printf("  Per-class:\n");
        for (int c = 0; c < N_CLASSES; c++)
            printf("    class %d: %d/%d = %.2f%%\n", c,
                   per_class_correct[c], per_class_total[c],
                   per_class_total[c] > 0 ? 100.0 * per_class_correct[c] / per_class_total[c] : 0.0);
        free(train_sigs); free(test_sigs);
        free(train_i8); free(test_i8);
        free(global_w);
        free(rtau_int); free(rtau_grad);
        free(fstat_selected);
        free(train_summary); free(test_summary);
        for (int m = 0; m < M; m++) {
            glyph_bucket_table_free(&tables[m]);
            free(table_train_keys[m]); free(table_test_keys[m]);
        }
        free(tables); free(table_train_keys); free(table_test_keys);
        glyph_dataset_free(&ds);
        return 0;
    }

    printf("Classifying %d queries...\n", ds.n_test);
    /* Open the per-query prediction dump if requested. One line per
     * query: `qi y lsh_pred sel_pred pig_pred gsh_pred`. Used to diff
     * configurations and identify which queries a layer flipped. */
    FILE* dump_f = NULL;
    if (dump_preds_path) {
        dump_f = fopen(dump_preds_path, "w");
        if (!dump_f) {
            fprintf(stderr, "warning: cannot open dump_preds=%s (continuing without dump)\n",
                    dump_preds_path);
        } else {
            fprintf(dump_f, "# qi y lsh sel pig gsh\n");
        }
    }

    clock_t t_sweep = clock();

    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];
        qs_ptr = test_sigs + (size_t)qi * sig_bytes;

        glyph_probe_reset(&st);
        int prev = 0;
        for (int si = 0; si < n_sweep; si++) {
            int Mt = m_sweep[si];
            for (int m = prev; m < Mt; m++) {
                const uint8_t* q_key = table_test_keys[m] + (size_t)qi * 4;
                glyph_probe_table(&tables[m], q_key, KEY_TRITS, 4,
                            cfg.max_radius, cfg.min_cands, &st, key_scratch);
            }

            for (int j = 0; j < st.n_hit; j++)
                if (ds.y_train[st.hit_list[j]] == y) { oracle_c[si]++; break; }
            union_sum[si] += st.n_hit;

            u.hit_list = st.hit_list; u.n_hit = st.n_hit; u.votes = st.votes;

            /* Score on FULL signature (1 "table"). Distance kernels:
             *   hamming (default)  : m4t_popcount_dist             → argmin
             *   sdot   (E1)        : −sdot_score                   → argmin
             *   weighted_hamming   : Σ w[d] × cost(q[d], t[d])     → argmin
             */
            int32_t best_d = INT32_MAX; int best_l = -1;
            topk_entry_t topk[64]; int ntk = 0;
            const int8_t* q_i8 = (use_sdot || use_weighted)
                ? (test_i8 + (size_t)qi * total_dim) : NULL;
            for (int j = 0; j < st.n_hit; j++) {
                int idx = st.hit_list[j];
                int32_t d;
                if (use_sdot) {
                    d = -sdot_score(q_i8, train_i8 + (size_t)idx * total_dim, total_dim);
                } else if (use_weighted) {
                    d = weighted_hamming_score(
                        q_i8, train_i8 + (size_t)idx * total_dim,
                        global_w, total_dim);
                } else if (use_block) {
                    d = block_threshold_score(
                        qs_ptr, train_sigs + (size_t)idx * sig_bytes,
                        sig_bytes, block_T);
                } else {
                    d = m4t_popcount_dist(
                        qs_ptr, train_sigs + (size_t)idx * sig_bytes,
                        full_mask, sig_bytes);
                }
                if (d < best_d) { best_d = d; best_l = ds.y_train[idx]; }
                topk_insert(topk, &ntk, KNN_K, d, ds.y_train[idx]);
            }
            if (best_l == y) sum_c[si]++;
            int kpred = topk_vote(topk, ntk, KNN_K);
            if (kpred == y) knn_c[si]++;
            if (m_sweep[si] == M) {
                final_pred[qi] = kpred;

                /* Experiment 1: routing footprint independence.
                 * For resolver-miss cases where correct class IS in
                 * union, compare d_route to best-correct vs best-wrong. */
                if (kpred != y) {
                    int has_correct = 0;
                    int best_correct_idx = -1, best_wrong_idx = -1;
                    int32_t best_correct_ham = INT32_MAX;
                    int32_t best_wrong_ham = INT32_MAX;
                    for (int j = 0; j < st.n_hit; j++) {
                        int idx = st.hit_list[j];
                        int32_t d = m4t_popcount_dist(
                            qs_ptr, train_sigs + (size_t)idx * sig_bytes,
                            full_mask, sig_bytes);
                        if (ds.y_train[idx] == y) {
                            has_correct = 1;
                            if (d < best_correct_ham) {
                                best_correct_ham = d;
                                best_correct_idx = idx;
                            }
                        } else if (d < best_wrong_ham) {
                            best_wrong_ham = d;
                            best_wrong_idx = idx;
                        }
                    }
                    if (has_correct && best_correct_idx >= 0 && best_wrong_idx >= 0) {
                        int d_route_correct = 0, d_route_wrong = 0;
                        for (int m = 0; m < M; m++) {
                            uint32_t qk = glyph_sig_to_key_u32(
                                table_test_keys[m] + (size_t)qi * 4);
                            uint32_t ck = glyph_sig_to_key_u32(
                                table_train_keys[m] + (size_t)best_correct_idx * 4);
                            uint32_t wk = glyph_sig_to_key_u32(
                                table_train_keys[m] + (size_t)best_wrong_idx * 4);
                            if (qk != ck) d_route_correct++;
                            if (qk != wk) d_route_wrong++;
                        }
                        rf_resolver_miss++;
                        rf_droute_correct_sum += d_route_correct;
                        rf_droute_wrong_sum += d_route_wrong;
                        if (d_route_correct < d_route_wrong)
                            rf_route_discriminates++;
                        else if (d_route_correct == d_route_wrong)
                            rf_route_tied++;
                    }
                }
            }
            /* Majority vote (for comparison with brute-force baseline). */
            int mv[N_CLASSES] = {0};
            for (int i = 0; i < ntk; i++) mv[topk[i].label]++;
            int mpred = 0;
            for (int c = 1; c < N_CLASSES; c++) if (mv[c] > mv[mpred]) mpred = c;
            if (mpred == y) maj_c[si]++;

            prev = Mt;
        }

        /* GSH pass at max M. */
        int lsh_pred = final_pred[qi];
        if (lsh_pred == y) lsh_total_correct++;

        union_top_m_labels(&st, M, sig_bytes, train_sigs, qs_ptr,
                           full_mask, ds.y_train, -1, vote_labels);
        encode_gsh_sig(vote_labels, M, q_gsh, GSH_SB);

        glyph_probe_reset(&gst);
        glyph_probe_table(&gsh_table, q_gsh, KEY_TRITS, 4,
                    cfg.max_radius, cfg.min_cands, &gst, key_scratch);

        int gsh_pred = -1;
        {
            int32_t best = INT32_MAX;
            for (int j = 0; j < gst.n_hit; j++) {
                int idx = gst.hit_list[j];
                int32_t d = m4t_popcount_dist(
                    q_gsh, gsh_train + (size_t)idx * GSH_SB,
                    gsh_mask, GSH_SB);
                if (d < best) { best = d; gsh_pred = ds.y_train[idx]; }
            }
        }
        if (gsh_pred == y) gsh_total_correct++;

        /* Pair-IG re-rank on LSH union.
         * Identify top-2 classes from Hamming k-NN, re-rank union
         * with pair-specific IG weights. */
        int pig_pred = lsh_pred;
        {
            int cv2[N_CLASSES] = {0};
            for (int j = 0; j < st.n_hit && j < 5; j++)
                cv2[ds.y_train[st.hit_list[j]]]++;
            /* Actually use the Hamming topk labels. */
            /* Re-derive top-2 from the k-NN prediction. */
            int c1 = lsh_pred, c2 = -1;
            if (gsh_pred >= 0 && gsh_pred != c1) c2 = gsh_pred;
            else {
                /* Find runner-up from vote labels. */
                int vl_cv[N_CLASSES] = {0};
                for (int m = 0; m < M && m < 64; m++)
                    if (vote_labels[m] >= 0 && vote_labels[m] < N_CLASSES)
                        vl_cv[vote_labels[m]]++;
                for (int c = 0; c < N_CLASSES; c++)
                    if (c != c1 && (c2 < 0 || vl_cv[c] > vl_cv[c2])) c2 = c;
            }
            if (c2 < 0) c2 = (c1 + 1) % N_CLASSES;
            const uint8_t* pw = pair_ig[c1 * N_CLASSES + c2];
            if (pw && st.n_hit > 0) {
                topk_entry_t ptopk[64]; int pntk = 0;
                const int8_t* q_pig = test_i8 + (size_t)qi * total_dim;
                for (int j = 0; j < st.n_hit; j++) {
                    int idx = st.hit_list[j];
                    int32_t dig = pair_ig_mismatch_score(
                        q_pig, train_i8 + (size_t)idx * total_dim,
                        pw, total_dim);
                    topk_insert(ptopk, &pntk, KNN_K, dig, ds.y_train[idx]);
                }
                pig_pred = topk_vote(ptopk, pntk, KNN_K);
            }
        }
        if (pig_pred == y) pair_ig_correct++;

        /* FILTERED pair-IG: re-rank only candidates labeled c1 or c2.
         * Tests whether PURITY of the re-ranking set helps. */
        int fpig_pred = lsh_pred;
        if (lsh_pred != gsh_pred) {
            int fc1 = lsh_pred, fc2 = (gsh_pred >= 0) ? gsh_pred : (fc1+1)%N_CLASSES;
            const uint8_t* fpw = pair_ig[fc1 * N_CLASSES + fc2];
            if (fpw && st.n_hit > 0) {
                topk_entry_t fptopk[64]; int fpntk = 0;
                const int8_t* q_fpig = test_i8 + (size_t)qi * total_dim;
                for (int j = 0; j < st.n_hit; j++) {
                    int idx = st.hit_list[j];
                    int lbl = ds.y_train[idx];
                    if (lbl != fc1 && lbl != fc2) continue;
                    int32_t dig = pair_ig_mismatch_score(
                        q_fpig, train_i8 + (size_t)idx * total_dim,
                        fpw, total_dim);
                    topk_insert(fptopk, &fpntk, KNN_K, dig, lbl);
                }
                if (fpntk > 0)
                    fpig_pred = topk_vote(fptopk, fpntk, KNN_K);
            }
        }
        if (fpig_pred == y) filtered_ig_correct++;
        int fsel_pred = (lsh_pred == gsh_pred) ? lsh_pred : fpig_pred;
        if (fsel_pred == y) filtered_selective_correct++;

        /* Selective: Hamming when agree, pair-IG when disagree. */
        int sel_pred = (lsh_pred == gsh_pred) ? lsh_pred : pig_pred;
        if (sel_pred == y) selective_correct++;

        if (dump_f) {
            fprintf(dump_f, "%d %d %d %d %d %d\n",
                    qi, y, lsh_pred, sel_pred, pig_pred, gsh_pred);
        }

        if (lsh_pred == gsh_pred) {
            agree_count++;
            if (lsh_pred == y) agree_correct++;
        } else {
            disagree_count++;
            if (lsh_pred == y) disagree_lsh_correct++;
            if (gsh_pred == y) disagree_gsh_correct++;
        }
    }
    double sweep_sec = (double)(clock() - t_sweep) / CLOCKS_PER_SEC;

    if (dump_f) { fclose(dump_f); dump_f = NULL; }

    printf("Sweep: %.1fs\n\n", sweep_sec);
    printf("   M    oracle    avg_union   1-NN      k=%d-maj   k=%d-rw\n", KNN_K, KNN_K);
    for (int si = 0; si < n_sweep; si++)
        printf("  %3d   %6.2f%%   %7.1f   %6.2f%%   %6.2f%%   %6.2f%%\n",
               m_sweep[si],
               100.0 * oracle_c[si] / ds.n_test,
               (double)union_sum[si] / ds.n_test,
               100.0 * sum_c[si] / ds.n_test,
               100.0 * maj_c[si] / ds.n_test,
               100.0 * knn_c[si] / ds.n_test);
    printf("\n");

    printf("=== LSH + GSH + pair-IG ===\n");
    printf("  LSH k=%d-rw (Hamming):    %6.2f%%\n", KNN_K, 100.0 * lsh_total_correct / ds.n_test);
    printf("  Pair-IG re-rank:          %6.2f%%\n", 100.0 * pair_ig_correct / ds.n_test);
    printf("  GSH 1-NN:                 %6.2f%%\n", 100.0 * gsh_total_correct / ds.n_test);
    printf("  Selective (agree→Ham, disagree→pair-IG): %6.2f%%\n",
           100.0 * selective_correct / ds.n_test);
    printf("  Filtered pair-IG (c1/c2 only):         %6.2f%%\n",
           100.0 * filtered_ig_correct / ds.n_test);
    printf("  Filtered selective:                     %6.2f%%\n",
           100.0 * filtered_selective_correct / ds.n_test);
    printf("\n  Agreement:                %6.2f%%  (%d / %d)\n",
           100.0 * agree_count / ds.n_test, agree_count, ds.n_test);
    printf("  P(correct | agree):       %6.2f%%  (%d / %d)\n",
           agree_count ? 100.0 * agree_correct / agree_count : 0.0,
           agree_correct, agree_count);
    printf("  P(LSH correct | disagree):%6.2f%%  (%d / %d)\n",
           disagree_count ? 100.0 * disagree_lsh_correct / disagree_count : 0.0,
           disagree_lsh_correct, disagree_count);
    printf("  P(GSH correct | disagree):%6.2f%%  (%d / %d)\n",
           disagree_count ? 100.0 * disagree_gsh_correct / disagree_count : 0.0,
           disagree_gsh_correct, disagree_count);
    printf("\n");

    /* Per-class at max M (from stored predictions — no double sweep). */
    int pc_t[N_CLASSES]={0}, pc_c[N_CLASSES]={0};
    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];
        if (y < 0 || y >= N_CLASSES) continue;
        pc_t[y]++;
        if (final_pred[qi] == y) pc_c[y]++;
    }
    printf("Per-class k=%d at M=%d:\n", KNN_K, M);
    printf("  class   count   correct   accuracy\n");
    for (int c = 0; c < N_CLASSES; c++)
        if (pc_t[c] > 0)
            printf("   %2d    %5d   %5d     %6.2f%%\n",
                   c, pc_t[c], pc_c[c], 100.0 * pc_c[c] / pc_t[c]);

    /* Routing footprint experiment results. */
    printf("\n=== Routing Footprint Experiment 1: Independence ===\n");
    printf("  Resolver-miss cases (wrong k-NN, correct in union): %d\n", rf_resolver_miss);
    if (rf_resolver_miss > 0) {
        int rf_wrong_wins = rf_resolver_miss - rf_route_discriminates - rf_route_tied;
        printf("  Route discriminates (d_route_correct < d_route_wrong): %d (%.1f%%)\n",
               rf_route_discriminates, 100.0 * rf_route_discriminates / rf_resolver_miss);
        printf("  Route tied: %d (%.1f%%)\n",
               rf_route_tied, 100.0 * rf_route_tied / rf_resolver_miss);
        printf("  Route wrong (d_route_correct >= d_route_wrong): %d (%.1f%%)\n",
               rf_wrong_wins, 100.0 * rf_wrong_wins / rf_resolver_miss);
        printf("  Avg d_route to correct: %.2f / %d tables\n",
               (double)rf_droute_correct_sum / rf_resolver_miss, M);
        printf("  Avg d_route to wrong:   %.2f / %d tables\n",
               (double)rf_droute_wrong_sum / rf_resolver_miss, M);
        printf("  Verdict: %s\n",
               rf_route_discriminates > rf_resolver_miss / 2
                   ? "ROUTING CARRIES INDEPENDENT SIGNAL"
                   : "Routing redundant with Hamming");
    }
    printf("\n");

    /* Cleanup. */
    for (int a = 0; a < N_CLASSES; a++)
        for (int b = a + 1; b < N_CLASSES; b++)
            free(pair_ig[a * N_CLASSES + b]);
    free(pair_ig);
    free(full_mask); free(st.votes); free(st.hit_list);
    free(final_pred); free(vote_labels);
    free(q_gsh); free(gsh_mask); free(gsh_train);
    glyph_bucket_table_free(&gsh_table);
    free(gst.votes); free(gst.hit_list);
    free(train_sigs); free(test_sigs);
    free(train_i8); free(test_i8);
    free(global_w);
    free(rtau_int); free(rtau_grad);
    free(fstat_selected);
    free(train_summary); free(test_summary);
    for (int m = 0; m < M; m++) {
        glyph_bucket_table_free(&tables[m]);
        free(table_train_keys[m]); free(table_test_keys[m]);
    }
    free(tables); free(table_train_keys); free(table_test_keys);
    glyph_dataset_free(&ds);
    return 0;
}
