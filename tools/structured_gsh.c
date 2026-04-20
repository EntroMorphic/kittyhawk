/*
 * structured_gsh.c — structured GSH via class-vote profile.
 *
 * The GSH signature is a 20-trit class-vote profile: for each of
 * 10 classes, two trits encoding (a) vote count and (b) average
 * distance. Each trit has a SPECIFIC MEANING — no random weights,
 * no matmul, no mixing.
 *
 * Composes with the direct_lsh pipeline: hierarchical LSH finds
 * candidates, Hamming k-NN produces LSH prediction, structured GSH
 * hashes the class-vote profile to find routing-similar images,
 * agreement + pair-IG provides the selective scorer.
 *
 * NO RANDOM WEIGHTS ANYWHERE.
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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_CLASSES 10
#define KNN_K 5
#define KEY_TRITS 16
#define GSH_DIM (N_CLASSES * 2)  /* 10 vote counts + 10 avg distances = 20 */

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

static void compute_gradients(const m4t_mtfp_t* img, int W, int H, int n_ch,
                              m4t_mtfp_t* hgrad, m4t_mtfp_t* vgrad) {
    int ppc = W * H;
    for (int ch = 0; ch < n_ch; ch++) {
        const m4t_mtfp_t* c = img + ch * ppc;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W - 1; x++)
                *hgrad++ = c[y * W + x + 1] - c[y * W + x];
        for (int y = 0; y < H - 1; y++)
            for (int x = 0; x < W; x++)
                *vgrad++ = c[(y + 1) * W + x] - c[y * W + x];
    }
}

/* Compute the class-vote profile from the union's top-K nearest.
 * out[0..9] = vote count per class (scaled to MTFP)
 * out[10..19] = avg distance per class (relative to median, scaled)
 * Each dimension has a SPECIFIC MEANING. No random weights. */
static void class_vote_profile(const probe_state_t* st, int K_top,
                               int sig_bytes, const uint8_t* train_sigs,
                               const uint8_t* q_sig, const uint8_t* mask,
                               const int* y_train, int exclude_idx,
                               m4t_mtfp_t* out) {
    typedef struct { int32_t d; int label; } dl_t;
    dl_t topk[256];
    int ntk = 0;
    int klim = (K_top < 256) ? K_top : 256;

    for (int j = 0; j < st->n_hit; j++) {
        int idx = st->hit_list[j];
        if (idx == exclude_idx) continue;
        int32_t d = m4t_popcount_dist(q_sig, train_sigs + (size_t)idx * sig_bytes,
                                       mask, sig_bytes);
        int lbl = y_train[idx];
        if (ntk < klim) {
            int pos = ntk;
            while (pos > 0 && topk[pos-1].d > d) { topk[pos]=topk[pos-1]; pos--; }
            topk[pos].d = d; topk[pos].label = lbl; ntk++;
        } else if (d < topk[klim-1].d) {
            int pos = klim - 1;
            while (pos > 0 && topk[pos-1].d > d) { topk[pos]=topk[pos-1]; pos--; }
            topk[pos].d = d; topk[pos].label = lbl;
        }
    }

    int votes[N_CLASSES] = {0};
    int64_t dist_sum[N_CLASSES] = {0};
    int median_d = (ntk > 0) ? topk[ntk / 2].d : 1;
    if (median_d == 0) median_d = 1;

    for (int i = 0; i < ntk; i++) {
        int c = topk[i].label;
        if (c >= 0 && c < N_CLASSES) {
            votes[c]++;
            dist_sum[c] += topk[i].d;
        }
    }

    for (int c = 0; c < N_CLASSES; c++) {
        out[c] = (m4t_mtfp_t)((int64_t)votes[c] * M4T_MTFP_SCALE / (klim > 0 ? klim : 1));
        out[N_CLASSES + c] = votes[c] > 0
            ? (m4t_mtfp_t)(((int64_t)(median_d * votes[c] - dist_sum[c]) * M4T_MTFP_SCALE)
                           / (median_d * votes[c]))
            : 0;
    }
}

int main(int argc, char** argv) {
    int use_gradients = 0;
    int new_argc = 0;
    char** new_argv = malloc((size_t)argc * sizeof(char*));
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--gradients") == 0) { use_gradients = 1; continue; }
        new_argv[new_argc++] = argv[i];
    }
    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, new_argc, new_argv);
    free(new_argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, cfg.data_dir) != 0) return 1;
    if (!cfg.no_deskew) glyph_dataset_deskew(&ds);
    glyph_dataset_normalize(&ds);

    int n_ch = (ds.input_dim > 784) ? 3 : 1;
    int img_w = ds.img_w > 0 ? ds.img_w : (n_ch == 3 ? 32 : 28);
    int img_h = ds.img_h > 0 ? ds.img_h : (n_ch == 3 ? 32 : 28);
    int intensity_dim = ds.input_dim;
    int hgrad_dim = n_ch * img_h * (img_w - 1);
    int vgrad_dim = n_ch * (img_h - 1) * img_w;
    int total_dim = intensity_dim + (use_gradients ? (hgrad_dim + vgrad_dim) : 0);
    int sig_bytes = M4T_TRIT_PACKED_BYTES(total_dim);
    int gsh_sb = M4T_TRIT_PACKED_BYTES(GSH_DIM);

    const int M = cfg.m_max;
    const int K_PROFILE = 64;

    printf("structured_gsh: class-vote-profile GSH\n");
    printf("  data=%s  gradients=%s  density=%.3f  M=%d\n",
           cfg.data_dir, use_gradients ? "on" : "off", cfg.density, M);
    printf("  GSH: %d dims (10 votes + 10 dists) = %d trits, %d bytes\n",
           GSH_DIM, GSH_DIM, gsh_sb);
    printf("  K_profile=%d  knn_k=%d  n_train=%d  n_test=%d\n\n",
           K_PROFILE, KNN_K, ds.n_train, ds.n_test);

    clock_t t0 = clock();

    /* Quantize signatures (same as direct_lsh). */
    printf("Quantizing...\n");
    int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;
    int64_t tau_i = glyph_sig_quantize_tau(ds.x_train, n_calib, intensity_dim, cfg.density);
    int64_t tau_g = 0;
    m4t_mtfp_t* train_feat = NULL;
    if (use_gradients) {
        train_feat = malloc((size_t)ds.n_train * total_dim * sizeof(m4t_mtfp_t));
        m4t_mtfp_t* hg = malloc((size_t)hgrad_dim * sizeof(m4t_mtfp_t));
        m4t_mtfp_t* vg = malloc((size_t)vgrad_dim * sizeof(m4t_mtfp_t));
        for (int i = 0; i < ds.n_train; i++) {
            m4t_mtfp_t* out = train_feat + (size_t)i * total_dim;
            memcpy(out, ds.x_train + (size_t)i * intensity_dim,
                   (size_t)intensity_dim * sizeof(m4t_mtfp_t));
            compute_gradients(ds.x_train + (size_t)i * intensity_dim,
                              img_w, img_h, n_ch, hg, vg);
            memcpy(out + intensity_dim, hg, (size_t)hgrad_dim * sizeof(m4t_mtfp_t));
            memcpy(out + intensity_dim + hgrad_dim, vg, (size_t)vgrad_dim * sizeof(m4t_mtfp_t));
        }
        m4t_mtfp_t* gs = malloc((size_t)n_calib * (hgrad_dim+vgrad_dim) * sizeof(m4t_mtfp_t));
        for (int i = 0; i < n_calib; i++)
            memcpy(gs + (size_t)i*(hgrad_dim+vgrad_dim),
                   train_feat + (size_t)i*total_dim + intensity_dim,
                   (size_t)(hgrad_dim+vgrad_dim)*sizeof(m4t_mtfp_t));
        tau_g = glyph_sig_quantize_tau(gs, n_calib, hgrad_dim+vgrad_dim, 0.10);
        free(gs); free(hg); free(vg);
    }

    uint8_t* train_sigs = calloc((size_t)ds.n_train * sig_bytes, 1);
    uint8_t* test_sigs  = calloc((size_t)ds.n_test  * sig_bytes, 1);
    for (int pass = 0; pass < 2; pass++) {
        int ni = (pass==0) ? ds.n_train : ds.n_test;
        const m4t_mtfp_t* src = (pass==0)
            ? (use_gradients ? train_feat : ds.x_train)
            : ds.x_test;
        uint8_t* sigs = (pass==0) ? train_sigs : test_sigs;
        int stride = use_gradients ? total_dim : intensity_dim;

        for (int i = 0; i < ni; i++) {
            const m4t_mtfp_t* f;
            m4t_mtfp_t* tmp_feat = NULL;
            if (pass == 1 && use_gradients) {
                tmp_feat = malloc((size_t)total_dim * sizeof(m4t_mtfp_t));
                memcpy(tmp_feat, ds.x_test + (size_t)i*intensity_dim,
                       (size_t)intensity_dim*sizeof(m4t_mtfp_t));
                m4t_mtfp_t* hg2 = malloc((size_t)hgrad_dim*sizeof(m4t_mtfp_t));
                m4t_mtfp_t* vg2 = malloc((size_t)vgrad_dim*sizeof(m4t_mtfp_t));
                compute_gradients(ds.x_test + (size_t)i*intensity_dim,
                                  img_w, img_h, n_ch, hg2, vg2);
                memcpy(tmp_feat+intensity_dim, hg2, (size_t)hgrad_dim*sizeof(m4t_mtfp_t));
                memcpy(tmp_feat+intensity_dim+hgrad_dim, vg2, (size_t)vgrad_dim*sizeof(m4t_mtfp_t));
                free(hg2); free(vg2);
                f = tmp_feat;
            } else {
                f = src + (size_t)i * stride;
            }
            uint8_t* sig = sigs + (size_t)i * sig_bytes;
            for (int d = 0; d < intensity_dim; d++) {
                int64_t v = (int64_t)f[d];
                if (v > tau_i) glyph_write_trit(sig, d, +1);
                else if (v < -tau_i) glyph_write_trit(sig, d, -1);
            }
            if (use_gradients) {
                for (int d = 0; d < hgrad_dim+vgrad_dim; d++) {
                    int64_t v = (int64_t)f[intensity_dim+d];
                    if (v > tau_g) glyph_write_trit(sig, intensity_dim+d, +1);
                    else if (v < -tau_g) glyph_write_trit(sig, intensity_dim+d, -1);
                }
            }
            if (tmp_feat) free(tmp_feat);
        }
    }
    free(train_feat);

    /* Build hierarchical bucket tables (same as direct_lsh). */
    printf("Building LSH tables...\n");
    int blk_w = (n_ch == 3) ? 4 : 2, blk_h = 4;
    int sum_w = img_w / blk_w, sum_h = img_h / blk_h;
    int summary_dim = sum_w * sum_h * n_ch;
    int summary_bytes = M4T_TRIT_PACKED_BYTES(summary_dim);
    int ppc = img_w * img_h;

    uint8_t* train_summary = calloc((size_t)ds.n_train * summary_bytes, 1);
    uint8_t* test_summary  = calloc((size_t)ds.n_test  * summary_bytes, 1);
    for (int pass = 0; pass < 2; pass++) {
        int ni = (pass==0) ? ds.n_train : ds.n_test;
        const uint8_t* sigs = (pass==0) ? train_sigs : test_sigs;
        uint8_t* sums = (pass==0) ? train_summary : test_summary;
        for (int i = 0; i < ni; i++) {
            const uint8_t* sig = sigs + (size_t)i * sig_bytes;
            uint8_t* ss = sums + (size_t)i * summary_bytes;
            int si = 0;
            for (int ch = 0; ch < n_ch; ch++)
                for (int by = 0; by < sum_h; by++)
                    for (int bx = 0; bx < sum_w; bx++) {
                        int pc2=0, nc2=0;
                        for (int dy=0; dy<blk_h; dy++)
                            for (int dx=0; dx<blk_w; dx++) {
                                int px=bx*blk_w+dx, py=by*blk_h+dy;
                                if (px>=img_w||py>=img_h) continue;
                                int8_t t = glyph_read_trit(sig, ch*ppc+py*img_w+px);
                                if (t>0) pc2++; else if (t<0) nc2++;
                            }
                        glyph_write_trit(ss, si++, pc2>nc2 ? +1 : nc2>pc2 ? -1 : 0);
                    }
        }
    }

    glyph_bucket_table_t* tables = calloc((size_t)M, sizeof(glyph_bucket_table_t));
    uint8_t** tkeys = calloc((size_t)M, sizeof(uint8_t*));
    uint8_t** qkeys = calloc((size_t)M, sizeof(uint8_t*));
    int* perm = malloc((size_t)summary_dim * sizeof(int));
    for (int m = 0; m < M; m++) {
        for (int t = 0; t < summary_dim; t++) perm[t] = t;
        glyph_rng_t prng;
        uint32_t ps[4] = {cfg.base_seed[0]+(uint32_t)m*9973u, cfg.base_seed[1]+(uint32_t)m*7919u,
                          cfg.base_seed[2]+(uint32_t)m*6271u, cfg.base_seed[3]+(uint32_t)m*5381u};
        if (!(ps[0]|ps[1]|ps[2]|ps[3])) ps[0]=1;
        glyph_rng_seed(&prng, ps[0], ps[1], ps[2], ps[3]);
        for (int t = summary_dim-1; t > 0; t--) {
            int j = (int)(glyph_rng_next(&prng) % (uint32_t)(t+1));
            int tmp = perm[t]; perm[t] = perm[j]; perm[j] = tmp;
        }
        tkeys[m] = calloc((size_t)ds.n_train*4, 1);
        qkeys[m] = calloc((size_t)ds.n_test*4, 1);
        for (int i = 0; i < ds.n_train; i++) {
            const uint8_t* ss = train_summary + (size_t)i*summary_bytes;
            uint8_t* k = tkeys[m] + (size_t)i*4;
            for (int t = 0; t < KEY_TRITS && t < summary_dim; t++)
                glyph_write_trit(k, t, glyph_read_trit(ss, perm[t]));
        }
        for (int i = 0; i < ds.n_test; i++) {
            const uint8_t* ss = test_summary + (size_t)i*summary_bytes;
            uint8_t* k = qkeys[m] + (size_t)i*4;
            for (int t = 0; t < KEY_TRITS && t < summary_dim; t++)
                glyph_write_trit(k, t, glyph_read_trit(ss, perm[t]));
        }
        glyph_bucket_build(&tables[m], tkeys[m], ds.n_train, 4);
    }
    free(perm);

    /* Build GSH: class-vote profiles for training images. */
    printf("Building structured GSH (class-vote profiles)...\n");
    probe_state_t bst;
    bst.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    bst.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    bst.max_union = cfg.max_union; bst.n_hit = 0;
    uint8_t bscratch[4];
    uint8_t* fmask = malloc(sig_bytes); memset(fmask, 0xFF, sig_bytes);

    /* Calibrate GSH tau. */
    m4t_mtfp_t profile[GSH_DIM];
    m4t_mtfp_t* gsh_calib = malloc((size_t)n_calib * GSH_DIM * sizeof(m4t_mtfp_t));
    for (int i = 0; i < n_calib; i++) {
        probe_state_reset(&bst);
        for (int m = 0; m < M; m++)
            probe_table(&tables[m], tkeys[m]+(size_t)i*4, KEY_TRITS, 4,
                        cfg.max_radius, cfg.min_cands, &bst, bscratch);
        class_vote_profile(&bst, K_PROFILE, sig_bytes, train_sigs,
                           train_sigs+(size_t)i*sig_bytes, fmask,
                           ds.y_train, i, profile);
        memcpy(gsh_calib + (size_t)i*GSH_DIM, profile, sizeof(profile));
    }
    int64_t gsh_tau = glyph_sig_quantize_tau(gsh_calib, n_calib, GSH_DIM, 0.33);
    free(gsh_calib);
    printf("  GSH tau=%lld (%.3f×S)\n", (long long)gsh_tau, (double)gsh_tau/M4T_MTFP_SCALE);

    uint8_t* gsh_train = calloc((size_t)ds.n_train * gsh_sb, 1);
    for (int i = 0; i < ds.n_train; i++) {
        probe_state_reset(&bst);
        for (int m = 0; m < M; m++)
            probe_table(&tables[m], tkeys[m]+(size_t)i*4, KEY_TRITS, 4,
                        cfg.max_radius, cfg.min_cands, &bst, bscratch);
        class_vote_profile(&bst, K_PROFILE, sig_bytes, train_sigs,
                           train_sigs+(size_t)i*sig_bytes, fmask,
                           ds.y_train, i, profile);
        glyph_sig_quantize(profile, GSH_DIM, gsh_tau, gsh_train+(size_t)i*gsh_sb);
        if ((i+1) % 10000 == 0) printf("  %d/%d\n", i+1, ds.n_train);
    }
    glyph_bucket_table_t gsh_table;
    glyph_bucket_build(&gsh_table, gsh_train, ds.n_train, gsh_sb);
    printf("  GSH: %d distinct buckets.\n", glyph_bucket_count_distinct(&gsh_table));

    double build_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Build: %.1fs\n\n", build_sec);

    /* Classify. */
    probe_state_t st, gst;
    st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    st.max_union = cfg.max_union; st.n_hit = 0;
    gst.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    gst.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    gst.max_union = cfg.max_union; gst.n_hit = 0;
    uint8_t* q_gsh = calloc(gsh_sb, 1);
    uint8_t* gsh_mask = malloc(gsh_sb); memset(gsh_mask, 0xFF, gsh_sb);

    int lsh_c=0, gsh_c=0, agree_n=0, agree_c=0;

    printf("Classifying %d queries...\n", ds.n_test);
    clock_t t_sweep = clock();

    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];
        const uint8_t* qs = test_sigs + (size_t)qi * sig_bytes;

        /* LSH. */
        probe_state_reset(&st);
        for (int m = 0; m < M; m++)
            probe_table(&tables[m], qkeys[m]+(size_t)qi*4, KEY_TRITS, 4,
                        cfg.max_radius, cfg.min_cands, &st, bscratch);

        /* Hamming k-NN. */
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

        /* Structured GSH: class-vote profile → quantize → probe. */
        class_vote_profile(&st, K_PROFILE, sig_bytes, train_sigs, qs,
                           fmask, ds.y_train, -1, profile);
        glyph_sig_quantize(profile, GSH_DIM, gsh_tau, q_gsh);

        probe_state_reset(&gst);
        probe_table(&gsh_table, q_gsh, KEY_TRITS, 4,
                    cfg.max_radius, cfg.min_cands, &gst, bscratch);

        int gsh_pred = -1;
        { int32_t bd = INT32_MAX;
          for (int j = 0; j < gst.n_hit; j++) {
              int idx = gst.hit_list[j];
              int32_t d = m4t_popcount_dist(q_gsh, gsh_train+(size_t)idx*gsh_sb, gsh_mask, gsh_sb);
              if (d < bd) { bd = d; gsh_pred = ds.y_train[idx]; }
          }
        }
        if (gsh_pred == y) gsh_c++;

        if (lsh_pred == gsh_pred) { agree_n++; if (lsh_pred == y) agree_c++; }

        if ((qi+1) % 2000 == 0)
            printf("  %d/%d  LSH=%.2f%%  GSH=%.2f%%  agree=%.1f%%\n",
                   qi+1, ds.n_test,
                   100.0*lsh_c/(qi+1), 100.0*gsh_c/(qi+1),
                   100.0*agree_n/(qi+1));
    }
    double sweep_sec = (double)(clock() - t_sweep) / CLOCKS_PER_SEC;

    printf("\n=== Results ===\n");
    printf("  LSH k=%d-rw:           %6.2f%%\n", KNN_K, 100.0*lsh_c/ds.n_test);
    printf("  Structured GSH 1-NN:  %6.2f%%\n", 100.0*gsh_c/ds.n_test);
    printf("  Agreement:             %6.2f%%  (%d / %d)\n",
           100.0*agree_n/ds.n_test, agree_n, ds.n_test);
    printf("  P(correct|agree):      %6.2f%%\n",
           agree_n ? 100.0*agree_c/agree_n : 0.0);
    printf("  Sweep: %.1fs\n", sweep_sec);

    /* Cleanup. */
    free(q_gsh); free(gsh_mask); free(gsh_train);
    glyph_bucket_table_free(&gsh_table);
    free(bst.votes); free(bst.hit_list);
    free(st.votes); free(st.hit_list);
    free(gst.votes); free(gst.hit_list);
    free(fmask);
    free(train_sigs); free(test_sigs);
    free(train_summary); free(test_summary);
    for (int m = 0; m < M; m++) {
        glyph_bucket_table_free(&tables[m]); free(tkeys[m]); free(qkeys[m]);
    }
    free(tables); free(tkeys); free(qkeys);
    glyph_dataset_free(&ds);
    return 0;
}
