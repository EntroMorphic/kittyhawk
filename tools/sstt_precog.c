/*
 * sstt_precog.c — SSTT feature extraction + Glyph Hierarchical LSH + GSH.
 *
 * SSTT pre-processes: RGB interleave → fixed ternary quantization
 * (>170 → +1, <85 → -1, else → 0) → gradients → the ternary
 * features become the input to Glyph's hierarchical LSH.
 *
 * Glyph routes: hierarchical spatial pooling keys → multi-table
 * bucket index → multi-probe → full-signature Hamming k-NN → GSH
 * agreement filter.
 *
 * SSTT does what it's good at (spatial feature extraction).
 * Glyph does what it's good at (multi-table routing + GSH).
 *
 * NO RANDOM PROJECTIONS. SSTT's ternary features are the signatures.
 */

#include "glyph_config.h"
#include "glyph_dataset.h"
#include "glyph_rng.h"
#include "glyph_sig.h"
#include "glyph_bucket.h"
#include "glyph_multiprobe.h"
#include "glyph_resolver.h"
#include "m4t_trit_pack.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_CLASSES 10
#define KNN_K 5
#define TRITS_PER_VOTE 4
#define KEY_TRITS 16

/* SSTT quantization thresholds on raw [0,255] byte values.
 * These are the FIXED thirds that SSTT uses — no per-image
 * normalization, no calibration. Third of the 0-255 range. */
#define SSTT_HI 170
#define SSTT_LO 85

/* Image dims after RGB interleave: 32×96 for CIFAR-10 (3ch). */
#define CIFAR_H 32
#define CIFAR_FLAT_W 96  /* 32 pixels × 3 channels interleaved */
#define CIFAR_FLAT_PIX (CIFAR_H * CIFAR_FLAT_W)  /* 3072 */

static const int8_t vote_trits_tbl[10][TRITS_PER_VOTE] = {
    {-1,-1,-1,-1}, {-1,-1,-1, 0}, {-1,-1,-1,+1}, {-1,-1, 0,-1},
    {-1,-1, 0, 0}, {-1,-1, 0,+1}, {-1,-1,+1,-1}, {-1,-1,+1, 0},
    {-1,-1,+1,+1}, {-1, 0,-1,-1},
};

typedef struct {
    uint16_t* votes;
    int32_t*  hit_list;
    int       n_hit;
    int       max_union;
    int       n_probes;
    int       per_table_cands;
} probe_state_t;

typedef struct {
    const glyph_bucket_table_t* table;
    probe_state_t* state;
} probe_ctx_t;

static int probe_cb(const uint8_t* probe_sig, void* vctx) {
    probe_ctx_t* pc = (probe_ctx_t*)vctx;
    probe_state_t* st = pc->state;
    const glyph_bucket_table_t* bt = pc->table;
    st->n_probes++;
    uint32_t key = glyph_sig_to_key_u32(probe_sig);
    int lb = glyph_bucket_lower_bound(bt, key);
    if (lb >= bt->n_entries || bt->entries[lb].key != key) return 0;
    for (int i = lb; i < bt->n_entries && bt->entries[i].key == key; i++) {
        int idx = bt->entries[i].proto_idx;
        if (st->votes[idx] == 0) {
            if (st->n_hit >= st->max_union) return 1;
            st->hit_list[st->n_hit++] = idx;
        }
        st->votes[idx]++;
        st->per_table_cands++;
        if (st->n_hit >= st->max_union) return 1;
    }
    return 0;
}

static void probe_state_reset(probe_state_t* st) {
    for (int j = 0; j < st->n_hit; j++) st->votes[st->hit_list[j]] = 0;
    st->n_hit = 0; st->n_probes = 0;
}

static void probe_table(const glyph_bucket_table_t* bt, const uint8_t* q_sig,
                        int n_proj, int sig_bytes, int max_radius, int min_cands,
                        probe_state_t* st, uint8_t* scratch) {
    probe_ctx_t pc = { bt, st };
    st->per_table_cands = 0;
    for (int r = 0; r <= max_radius; r++) {
        if (st->per_table_cands >= min_cands && r > 0) break;
        glyph_multiprobe_enumerate(q_sig, n_proj, sig_bytes, r, scratch, probe_cb, &pc);
        if (st->n_hit >= st->max_union) break;
    }
}

/* SSTT-style feature extraction on MTFP normalized data.
 * RGB interleave → density-calibrated quantize → gradients.
 * Input: MTFP normalized pixels, channel-first [R_HW, G_HW, B_HW].
 * Output: packed trit signature (intensity + hgrad + vgrad).
 * Uses tau thresholds instead of fixed 85/170. */
static void sstt_extract_mtfp(const m4t_mtfp_t* pixels, int orig_w, int orig_h,
                              int n_ch, int64_t tau_i, int64_t tau_g,
                              uint8_t* out_sig, int total_trits) {
    int flat_w = orig_w * n_ch;
    int flat_pix = orig_h * flat_w;
    int hgrad_dim = orig_h * (flat_w - 1);
    int vgrad_dim = (orig_h - 1) * flat_w;

    /* Interleave RGB into a flat MTFP buffer. */
    m4t_mtfp_t* flat = malloc((size_t)flat_pix * sizeof(m4t_mtfp_t));
    for (int y = 0; y < orig_h; y++)
        for (int x = 0; x < orig_w; x++)
            for (int c = 0; c < n_ch; c++)
                flat[y * flat_w + x * n_ch + c] =
                    pixels[c * orig_w * orig_h + y * orig_w + x];

    /* Compute gradients on the MTFP interleaved image. */
    m4t_mtfp_t* hg = calloc(flat_pix, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* vg = calloc(flat_pix, sizeof(m4t_mtfp_t));
    for (int y = 0; y < orig_h; y++)
        for (int x = 0; x < flat_w - 1; x++)
            hg[y * flat_w + x] = flat[y * flat_w + x + 1] - flat[y * flat_w + x];
    for (int y = 0; y < orig_h - 1; y++)
        for (int x = 0; x < flat_w; x++)
            vg[y * flat_w + x] = flat[(y + 1) * flat_w + x] - flat[y * flat_w + x];

    /* Quantize and pack: intensity + hgrad + vgrad. */
    memset(out_sig, 0, M4T_TRIT_PACKED_BYTES(total_trits));
    int pos = 0;
    for (int i = 0; i < flat_pix; i++) {
        int64_t v = (int64_t)flat[i];
        if (v > tau_i) glyph_write_trit(out_sig, pos, +1);
        else if (v < -tau_i) glyph_write_trit(out_sig, pos, -1);
        pos++;
    }
    for (int i = 0; i < hgrad_dim; i++) {
        int64_t v = (int64_t)hg[i];
        if (v > tau_g) glyph_write_trit(out_sig, pos, +1);
        else if (v < -tau_g) glyph_write_trit(out_sig, pos, -1);
        pos++;
    }
    for (int i = 0; i < vgrad_dim; i++) {
        int64_t v = (int64_t)vg[i];
        if (v > tau_g) glyph_write_trit(out_sig, pos, +1);
        else if (v < -tau_g) glyph_write_trit(out_sig, pos, -1);
        pos++;
    }

    free(flat); free(hg); free(vg);
}

static void encode_gsh_sig(const int* labels, int n_tables,
                           uint8_t* out, int gsh_sb) {
    memset(out, 0, gsh_sb);
    for (int m = 0; m < n_tables; m++) {
        int lbl = labels[m];
        if (lbl < 0 || lbl >= N_CLASSES) lbl = 0;
        for (int t = 0; t < TRITS_PER_VOTE; t++)
            glyph_write_trit(out, m * TRITS_PER_VOTE + t, vote_trits_tbl[lbl][t]);
    }
}

static void union_top_m_labels(
    const probe_state_t* st, int M_labels, int sig_bytes,
    const uint8_t* train_sigs, const uint8_t* q_sig,
    const uint8_t* mask, const int* y_train,
    int exclude_idx, int* out_labels)
{
    typedef struct { int32_t d; int label; } dl_t;
    dl_t topk[256]; int ntk = 0;
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

int main(int argc, char** argv) {
    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, argc, argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    /* Load raw CIFAR-10 as uint8 for SSTT quantization. */
    printf("sstt_precog: SSTT features + Glyph Hierarchical LSH + GSH\n");
    printf("  data=%s\n\n", cfg.data_dir);

    /* Load float data and convert back to uint8 for SSTT thresholds. */
    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, cfg.data_dir) != 0) return 1;

    int n_ch = (ds.input_dim > 784) ? 3 : 1;
    int img_w = ds.img_w > 0 ? ds.img_w : (n_ch == 3 ? 32 : 28);
    int img_h = ds.img_h > 0 ? ds.img_h : (n_ch == 3 ? 32 : 28);

    /* Normalize per-image BEFORE interleaving, so the density-
     * calibrated tau produces balanced emission. */
    glyph_dataset_normalize(&ds);

    /* SSTT feature dimensions. */
    int flat_w = img_w * n_ch;
    int flat_pix = img_h * flat_w;
    int hgrad_dim = img_h * (flat_w - 1);
    int vgrad_dim = (img_h - 1) * flat_w;
    int total_trits = flat_pix + hgrad_dim + vgrad_dim;
    int sig_bytes = M4T_TRIT_PACKED_BYTES(total_trits);

    printf("  SSTT features: %dx%d interleaved → %d trits (%d bytes)\n",
           flat_w, img_h, total_trits, sig_bytes);
    printf("    intensity=%d  hgrad=%d  vgrad=%d\n", flat_pix, hgrad_dim, vgrad_dim);
    printf("    thresholds: lo=%d hi=%d (fixed thirds)\n", SSTT_LO, SSTT_HI);

    const int M = cfg.m_max;
    printf("  M=%d  knn_k=%d  n_train=%d  n_test=%d\n\n", M, KNN_K, ds.n_train, ds.n_test);

    /* Calibrate tau on the interleaved normalized training data. */
    clock_t t0 = clock();

    /* Build interleaved calibration sample for tau computation. */
    int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;
    m4t_mtfp_t* calib_flat = malloc((size_t)n_calib * flat_pix * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* calib_grad = malloc((size_t)n_calib * (hgrad_dim + vgrad_dim) * sizeof(m4t_mtfp_t));
    for (int i = 0; i < n_calib; i++) {
        const m4t_mtfp_t* px = ds.x_train + (size_t)i * ds.input_dim;
        m4t_mtfp_t* fl = calib_flat + (size_t)i * flat_pix;
        for (int y = 0; y < img_h; y++)
            for (int x = 0; x < img_w; x++)
                for (int c = 0; c < n_ch; c++)
                    fl[y * flat_w + x * n_ch + c] = px[c * img_w * img_h + y * img_w + x];
        m4t_mtfp_t* gr = calib_grad + (size_t)i * (hgrad_dim + vgrad_dim);
        int gi = 0;
        for (int y = 0; y < img_h; y++)
            for (int x = 0; x < flat_w - 1; x++)
                gr[gi++] = fl[y * flat_w + x + 1] - fl[y * flat_w + x];
        for (int y = 0; y < img_h - 1; y++)
            for (int x = 0; x < flat_w; x++)
                gr[gi++] = fl[(y + 1) * flat_w + x] - fl[y * flat_w + x];
    }
    int64_t tau_i = glyph_sig_quantize_tau(calib_flat, n_calib, flat_pix, 0.395);
    int64_t tau_g = glyph_sig_quantize_tau(calib_grad, n_calib, hgrad_dim + vgrad_dim, 0.10);
    free(calib_flat); free(calib_grad);
    printf("  tau_intensity=%lld (%.3f×S)  tau_gradient=%lld (%.3f×S)\n",
           (long long)tau_i, (double)tau_i / M4T_MTFP_SCALE,
           (long long)tau_g, (double)tau_g / M4T_MTFP_SCALE);

    /* Extract features for all images. */
    printf("Extracting features...\n");
    uint8_t* train_sigs = calloc((size_t)ds.n_train * sig_bytes, 1);
    uint8_t* test_sigs  = calloc((size_t)ds.n_test  * sig_bytes, 1);
    for (int i = 0; i < ds.n_train; i++)
        sstt_extract_mtfp(ds.x_train + (size_t)i * ds.input_dim, img_w, img_h, n_ch,
                          tau_i, tau_g,
                          train_sigs + (size_t)i * sig_bytes, total_trits);
    for (int i = 0; i < ds.n_test; i++)
        sstt_extract_mtfp(ds.x_test + (size_t)i * ds.input_dim, img_w, img_h, n_ch,
                          tau_i, tau_g,
                          test_sigs + (size_t)i * sig_bytes, total_trits);
    printf("  Features extracted in %.1fs.\n",
           (double)(clock() - t0) / CLOCKS_PER_SEC);

    /* Emission coverage diagnostic. */
    {
        long np = 0, nz = 0, nn = 0;
        int sn = (ds.n_train < 1000) ? ds.n_train : 1000;
        for (int i = 0; i < sn; i++) {
            const uint8_t* sig = train_sigs + (size_t)i * sig_bytes;
            for (int d = 0; d < total_trits; d++) {
                int8_t t = glyph_read_trit(sig, d);
                if (t > 0) np++; else if (t < 0) nn++; else nz++;
            }
        }
        long tot = (long)sn * total_trits;
        printf("  Emission: +1=%.1f%% 0=%.1f%% -1=%.1f%%\n\n",
               100.0*np/tot, 100.0*nz/tot, 100.0*nn/tot);
    }

    /* Hierarchical spatial pooling for bucket keys. */
    int blk_w = (n_ch == 3) ? 4 : 2;
    int blk_h = 4;
    /* Summary operates on the INTERLEAVED image (flat_w × img_h). */
    int sum_w = flat_w / blk_w;
    int sum_h = img_h / blk_h;
    int summary_dim = sum_w * sum_h;
    int summary_bytes = M4T_TRIT_PACKED_BYTES(summary_dim);
    printf("Hierarchical key: %dx%d blocks → %dx%d = %d summary trits\n",
           blk_w, blk_h, sum_w, sum_h, summary_dim);

    uint8_t* train_summary = calloc((size_t)ds.n_train * summary_bytes, 1);
    uint8_t* test_summary  = calloc((size_t)ds.n_test  * summary_bytes, 1);

    for (int pass = 0; pass < 2; pass++) {
        int n_imgs = (pass == 0) ? ds.n_train : ds.n_test;
        const uint8_t* sigs = (pass == 0) ? train_sigs : test_sigs;
        uint8_t* summaries = (pass == 0) ? train_summary : test_summary;
        for (int i = 0; i < n_imgs; i++) {
            const uint8_t* sig = sigs + (size_t)i * sig_bytes;
            uint8_t* ss = summaries + (size_t)i * summary_bytes;
            int si = 0;
            for (int by = 0; by < sum_h; by++) {
                for (int bx = 0; bx < sum_w; bx++) {
                    int pc = 0, nc = 0;
                    for (int dy = 0; dy < blk_h; dy++)
                        for (int dx = 0; dx < blk_w; dx++) {
                            int py = by * blk_h + dy;
                            int px = bx * blk_w + dx;
                            if (py >= img_h || px >= flat_w) continue;
                            int8_t t = glyph_read_trit(sig, py * flat_w + px);
                            if (t > 0) pc++; else if (t < 0) nc++;
                        }
                    int8_t st = 0;
                    if (pc > nc) st = +1; else if (nc > pc) st = -1;
                    glyph_write_trit(ss, si++, st);
                }
            }
        }
    }

    /* Build bucket tables. */
    printf("Building %d bucket tables...\n", M);
    glyph_bucket_table_t* tables = calloc((size_t)M, sizeof(glyph_bucket_table_t));
    uint8_t** tkeys = calloc((size_t)M, sizeof(uint8_t*));
    uint8_t** qkeys = calloc((size_t)M, sizeof(uint8_t*));
    int* perm = malloc((size_t)summary_dim * sizeof(int));

    for (int m = 0; m < M; m++) {
        for (int t = 0; t < summary_dim; t++) perm[t] = t;
        glyph_rng_t prng;
        uint32_t ps[4] = {
            cfg.base_seed[0]+(uint32_t)m*9973u, cfg.base_seed[1]+(uint32_t)m*7919u,
            cfg.base_seed[2]+(uint32_t)m*6271u, cfg.base_seed[3]+(uint32_t)m*5381u};
        if (!(ps[0]|ps[1]|ps[2]|ps[3])) ps[0]=1;
        glyph_rng_seed(&prng, ps[0], ps[1], ps[2], ps[3]);
        for (int t = summary_dim-1; t > 0; t--) {
            int j = (int)(glyph_rng_next(&prng) % (uint32_t)(t+1));
            int tmp = perm[t]; perm[t] = perm[j]; perm[j] = tmp;
        }
        tkeys[m] = calloc((size_t)ds.n_train * 4, 1);
        qkeys[m] = calloc((size_t)ds.n_test  * 4, 1);
        for (int i = 0; i < ds.n_train; i++) {
            const uint8_t* ss = train_summary + (size_t)i * summary_bytes;
            uint8_t* k = tkeys[m] + (size_t)i * 4;
            for (int t = 0; t < KEY_TRITS && t < summary_dim; t++)
                glyph_write_trit(k, t, glyph_read_trit(ss, perm[t]));
        }
        for (int i = 0; i < ds.n_test; i++) {
            const uint8_t* ss = test_summary + (size_t)i * summary_bytes;
            uint8_t* k = qkeys[m] + (size_t)i * 4;
            for (int t = 0; t < KEY_TRITS && t < summary_dim; t++)
                glyph_write_trit(k, t, glyph_read_trit(ss, perm[t]));
        }
        glyph_bucket_build(&tables[m], tkeys[m], ds.n_train, 4);
    }
    free(perm);

    /* GSH training sigs. */
    const int GSH_NT = M * TRITS_PER_VOTE;
    const int GSH_SB = M4T_TRIT_PACKED_BYTES(GSH_NT);
    printf("Building GSH (%d trits)...\n", GSH_NT);
    probe_state_t bst;
    bst.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    bst.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    bst.max_union = cfg.max_union; bst.n_hit = 0;
    uint8_t bscratch[4];
    uint8_t* fmask = malloc(sig_bytes); memset(fmask, 0xFF, sig_bytes);
    int* vlabels = malloc((size_t)M * sizeof(int));
    uint8_t* gsh_train = calloc((size_t)ds.n_train * GSH_SB, 1);

    for (int i = 0; i < ds.n_train; i++) {
        probe_state_reset(&bst);
        for (int m = 0; m < M; m++)
            probe_table(&tables[m], tkeys[m]+(size_t)i*4, KEY_TRITS, 4,
                        cfg.max_radius, cfg.min_cands, &bst, bscratch);
        union_top_m_labels(&bst, M, sig_bytes, train_sigs,
                           train_sigs+(size_t)i*sig_bytes, fmask,
                           ds.y_train, i, vlabels);
        encode_gsh_sig(vlabels, M, gsh_train+(size_t)i*GSH_SB, GSH_SB);
        if ((i+1) % 10000 == 0) printf("  %d/%d\n", i+1, ds.n_train);
    }
    glyph_bucket_table_t gsh_table;
    glyph_bucket_build(&gsh_table, gsh_train, ds.n_train, GSH_SB);
    printf("  GSH: %d distinct buckets.\n", glyph_bucket_count_distinct(&gsh_table));

    double build_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Total build: %.1fs\n\n", build_sec);

    /* Classify. */
    probe_state_t st, gst;
    st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    st.max_union = cfg.max_union; st.n_hit = 0;
    gst.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    gst.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    gst.max_union = cfg.max_union; gst.n_hit = 0;
    uint8_t* qgsh = calloc(GSH_SB, 1);
    uint8_t* gmask = malloc(GSH_SB); memset(gmask, 0xFF, GSH_SB);

    int lsh_c = 0, gsh_c = 0, agree_n = 0, agree_c = 0;
    int disagree_n = 0, disagree_lsh = 0, disagree_gsh = 0;
    int* fpred = malloc((size_t)ds.n_test * sizeof(int));

    printf("Classifying %d queries...\n", ds.n_test);
    clock_t t_sweep = clock();

    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];
        const uint8_t* qs = test_sigs + (size_t)qi * sig_bytes;

        probe_state_reset(&st);
        for (int m = 0; m < M; m++)
            probe_table(&tables[m], qkeys[m]+(size_t)qi*4, KEY_TRITS, 4,
                        cfg.max_radius, cfg.min_cands, &st, bscratch);

        /* k-NN on full SSTT signatures. */
        typedef struct { int32_t s; int l; } tk_t;
        tk_t topk[64]; int ntk = 0;
        for (int j = 0; j < st.n_hit; j++) {
            int idx = st.hit_list[j];
            int32_t d = m4t_popcount_dist(qs, train_sigs+(size_t)idx*sig_bytes, fmask, sig_bytes);
            int lbl = ds.y_train[idx];
            if (ntk < KNN_K) {
                int pos = ntk;
                while (pos > 0 && topk[pos-1].s > d) { topk[pos]=topk[pos-1]; pos--; }
                topk[pos].s = d; topk[pos].l = lbl; ntk++;
            } else if (d < topk[KNN_K-1].s) {
                int pos = KNN_K-1;
                while (pos > 0 && topk[pos-1].s > d) { topk[pos]=topk[pos-1]; pos--; }
                topk[pos].s = d; topk[pos].l = lbl;
            }
        }
        int cv[N_CLASSES] = {0};
        for (int i = 0; i < ntk; i++) cv[topk[i].l] += (KNN_K - i);
        int lsh_pred = 0;
        for (int c = 1; c < N_CLASSES; c++) if (cv[c] > cv[lsh_pred]) lsh_pred = c;
        if (lsh_pred == y) lsh_c++;
        fpred[qi] = lsh_pred;

        /* GSH. */
        union_top_m_labels(&st, M, sig_bytes, train_sigs, qs, fmask,
                           ds.y_train, -1, vlabels);
        encode_gsh_sig(vlabels, M, qgsh, GSH_SB);
        probe_state_reset(&gst);
        probe_table(&gsh_table, qgsh, KEY_TRITS, 4,
                    cfg.max_radius, cfg.min_cands, &gst, bscratch);
        int gsh_pred = -1;
        { int32_t bd = INT32_MAX;
          for (int j = 0; j < gst.n_hit; j++) {
              int idx = gst.hit_list[j];
              int32_t d = m4t_popcount_dist(qgsh, gsh_train+(size_t)idx*GSH_SB, gmask, GSH_SB);
              if (d < bd) { bd = d; gsh_pred = ds.y_train[idx]; }
          }
        }
        if (gsh_pred == y) gsh_c++;
        if (lsh_pred == gsh_pred) { agree_n++; if (lsh_pred == y) agree_c++; }
        else { disagree_n++; if (lsh_pred == y) disagree_lsh++; if (gsh_pred == y) disagree_gsh++; }
    }
    double sweep_sec = (double)(clock() - t_sweep) / CLOCKS_PER_SEC;

    printf("Sweep: %.1fs\n\n", sweep_sec);
    printf("=== Results ===\n");
    printf("  LSH k=%d-rw:              %6.2f%%\n", KNN_K, 100.0 * lsh_c / ds.n_test);
    printf("  GSH 1-NN:                 %6.2f%%\n", 100.0 * gsh_c / ds.n_test);
    printf("  Agreement:                %6.2f%%  (%d / %d)\n",
           100.0 * agree_n / ds.n_test, agree_n, ds.n_test);
    printf("  P(correct | agree):       %6.2f%%  (%d / %d)\n",
           agree_n ? 100.0 * agree_c / agree_n : 0.0, agree_c, agree_n);
    printf("  P(LSH correct | disagree):%6.2f%%\n",
           disagree_n ? 100.0 * disagree_lsh / disagree_n : 0.0);
    printf("  P(GSH correct | disagree):%6.2f%%\n\n",
           disagree_n ? 100.0 * disagree_gsh / disagree_n : 0.0);

    /* Per-class. */
    int pt[N_CLASSES]={0}, pc[N_CLASSES]={0};
    for (int qi = 0; qi < ds.n_test; qi++) {
        int yy = ds.y_test[qi]; if (yy<0||yy>=N_CLASSES) continue;
        pt[yy]++; if (fpred[qi]==yy) pc[yy]++;
    }
    printf("Per-class k=%d:\n  class  count  correct  accuracy\n", KNN_K);
    for (int c = 0; c < N_CLASSES; c++)
        if (pt[c]) printf("   %2d   %5d   %5d    %6.2f%%\n", c, pt[c], pc[c], 100.0*pc[c]/pt[c]);

    /* Cleanup. */
    free(fpred); free(vlabels); free(qgsh); free(gmask);
    free(fmask); free(gsh_train); glyph_bucket_table_free(&gsh_table);
    free(st.votes); free(st.hit_list); free(gst.votes); free(gst.hit_list);
    free(bst.votes); free(bst.hit_list);
    free(train_sigs); free(test_sigs);
    free(train_summary); free(test_summary);
    for (int m = 0; m < M; m++) {
        glyph_bucket_table_free(&tables[m]); free(tkeys[m]); free(qkeys[m]);
    }
    free(tables); free(tkeys); free(qkeys);
    glyph_dataset_free(&ds);
    return 0;
}
