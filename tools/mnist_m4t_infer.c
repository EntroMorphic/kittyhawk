/*
 * mnist_m4t_infer.c — MNIST inference using M4T primitives only.
 *
 * Loads weights from mnist_train_dump output (float+ternary), converts
 * to MTFP19 cells, runs the forward pass in pure M4T, reports accuracy.
 *
 * This is the substrate hypothesis test: if M4T produces the same
 * (or close) accuracy as the float+ternary reference, the substrate works.
 *
 * Usage: ./mnist_m4t_infer <mnist_dir> <weights.bin>
 */

#include "m4t_types.h"
#include "m4t_mtfp.h"
#include "m4t_trit_pack.h"
#include "m4t_ternary_matmul.h"
#include "m4t_route.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>  /* only for the float→MTFP conversion in the loader */

/* ── Helpers ───────────────────────────────────────────────────────────── */

static m4t_mtfp_t float_to_mtfp(float x) {
    float scaled = x * (float)M4T_MTFP_SCALE;
    int32_t r = (int32_t)lrintf(scaled);
    if (r >  (int32_t)M4T_MTFP_MAX_VAL) r =  M4T_MTFP_MAX_VAL;
    if (r < -(int32_t)M4T_MTFP_MAX_VAL) r = -M4T_MTFP_MAX_VAL;
    return (m4t_mtfp_t)r;
}

/* IDX format reader — returns MTFP19 cells (pixels / 255 * SCALE) */
static uint32_t read_u32_be(FILE* f) {
    uint8_t b[4]; fread(b,1,4,f);
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];
}
static m4t_mtfp_t* load_idx_images_mtfp(const char* p, int* n) {
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}
    read_u32_be(f); *n=(int)read_u32_be(f);
    int rows=(int)read_u32_be(f), cols=(int)read_u32_be(f);
    int dim=rows*cols; size_t t=(size_t)(*n)*dim;
    uint8_t* r=malloc(t); fread(r,1,t,f); fclose(f);
    m4t_mtfp_t* d=malloc(t*sizeof(m4t_mtfp_t));
    for(size_t i=0;i<t;i++) d[i] = float_to_mtfp((float)r[i]/255.0f);
    free(r); return d;
}
static int* load_idx_labels(const char* p, int* n) {
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}
    read_u32_be(f); *n=(int)read_u32_be(f);
    uint8_t* r=malloc(*n); fread(r,1,*n,f); fclose(f);
    int* l=malloc(*n*sizeof(int));
    for(int i=0;i<*n;i++) l[i]=(int)r[i];
    free(r); return l;
}

/* Load float array from weight file, convert to MTFP19 */
static m4t_mtfp_t* load_floats_as_mtfp(FILE* f, int* count) {
    int32_t c; fread(&c, 4, 1, f);
    *count = (int)c;
    float* buf = malloc((size_t)c * sizeof(float));
    fread(buf, sizeof(float), (size_t)c, f);
    m4t_mtfp_t* out = malloc((size_t)c * sizeof(m4t_mtfp_t));
    for (int i = 0; i < c; i++) out[i] = float_to_mtfp(buf[i]);
    free(buf);
    return out;
}

/* Load ternary array from weight file as m4t_trit_t */
static m4t_trit_t* load_ternary(FILE* f, int* count) {
    int32_t c; fread(&c, 4, 1, f);
    *count = (int)c;
    int8_t* buf = malloc((size_t)c);
    fread(buf, 1, (size_t)c, f);
    return (m4t_trit_t*)buf;  /* already int8 {-1,0,+1} */
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <mnist_dir> <weights.bin>\n", argv[0]);
        return 1;
    }

    /* Load MNIST test data as MTFP19 */
    char path[512]; int nv;
    snprintf(path,512,"%s/t10k-images-idx3-ubyte", argv[1]);
    m4t_mtfp_t* xv = load_idx_images_mtfp(path, &nv);
    snprintf(path,512,"%s/t10k-labels-idx1-ubyte", argv[1]);
    int* yv = load_idx_labels(path, &nv);
    printf("Loaded %d test images\n", nv);

    /* Load weights */
    FILE* fw = fopen(argv[2], "rb");
    if (!fw) { fprintf(stderr, "Cannot open %s\n", argv[2]); return 1; }

    int32_t magic; fread(&magic, 4, 1, fw);
    if (magic != 0x4D345457) { fprintf(stderr, "Bad magic\n"); return 1; }

    int32_t cfg[6]; fread(cfg, 4, 6, fw);
    int input_dim=cfg[0], D=cfg[1], T=cfg[2], K=cfg[3], H_proj=cfg[4], n_classes=cfg[5];
    printf("Config: input=%d D=%d T=%d K=%d H_proj=%d classes=%d\n",
           input_dim, D, T, K, H_proj, n_classes);

    int cnt;
    /* Projection */
    m4t_mtfp_t* pW1 = load_floats_as_mtfp(fw, &cnt); /* [H_proj, input_dim] */
    m4t_mtfp_t* pb1 = load_floats_as_mtfp(fw, &cnt); /* [H_proj] */
    m4t_mtfp_t* pW2 = load_floats_as_mtfp(fw, &cnt); /* [D, H_proj] */
    m4t_mtfp_t* pb2 = load_floats_as_mtfp(fw, &cnt); /* [D] */

    /* FFN LayerNorm */
    m4t_mtfp_t* ln_w = load_floats_as_mtfp(fw, &cnt); /* [D] */
    m4t_mtfp_t* ln_b = load_floats_as_mtfp(fw, &cnt); /* [D] */

    /* FFN tiles */
    m4t_trit_t** tile_W1 = malloc(T * sizeof(m4t_trit_t*));
    m4t_mtfp_t** tile_b1 = malloc(T * sizeof(m4t_mtfp_t*));
    m4t_trit_t** tile_W2 = malloc(T * sizeof(m4t_trit_t*));
    m4t_mtfp_t** tile_b2 = malloc(T * sizeof(m4t_mtfp_t*));
    uint8_t** tile_W1_packed = malloc(T * sizeof(uint8_t*));
    uint8_t** tile_W2_packed = malloc(T * sizeof(uint8_t*));

    int Dp1 = M4T_TRIT_PACKED_BYTES(D);  /* tile_hidden == D */
    for (int t = 0; t < T; t++) {
        tile_W1[t] = load_ternary(fw, &cnt); /* [D, D] ternary */
        tile_b1[t] = load_floats_as_mtfp(fw, &cnt); /* [D] */
        tile_W2[t] = load_ternary(fw, &cnt); /* [D, D] ternary */
        tile_b2[t] = load_floats_as_mtfp(fw, &cnt); /* [D] */

        /* Pack ternary weights for m4t_mtfp_ternary_matmul_bt */
        tile_W1_packed[t] = malloc((size_t)D * Dp1);
        m4t_pack_trits_rowmajor(tile_W1_packed[t], tile_W1[t], D, D);
        tile_W2_packed[t] = malloc((size_t)D * Dp1);
        m4t_pack_trits_rowmajor(tile_W2_packed[t], tile_W2[t], D, D);
    }

    /* FFN output_scale */
    m4t_mtfp_t* os_f = load_floats_as_mtfp(fw, &cnt);
    m4t_mtfp_t output_scale = os_f[0];
    free(os_f);

    /* Head */
    m4t_mtfp_t* hW1 = load_floats_as_mtfp(fw, &cnt); /* [D*2, D] */
    m4t_mtfp_t* hb1 = load_floats_as_mtfp(fw, &cnt); /* [D*2] */
    m4t_mtfp_t* hW2 = load_floats_as_mtfp(fw, &cnt); /* [n_classes, D*2] */
    m4t_mtfp_t* hb2 = load_floats_as_mtfp(fw, &cnt); /* [n_classes] */

    fclose(fw);
    printf("Weights loaded and converted to MTFP19\n");

    /* Compute routing signatures */
    /* For now, use simple column-sum signatures from tile_W1 */
    uint8_t* tile_sigs = malloc((size_t)T * Dp1);
    {
        int64_t scratch[(T + 1) * D];
        /* Build a weight buffer: [T * D, D] = T tiles of [D, D] */
        uint8_t* all_W1 = malloc((size_t)T * D * Dp1);
        for (int t = 0; t < T; t++)
            memcpy(all_W1 + (size_t)t * D * Dp1, tile_W1_packed[t], (size_t)D * Dp1);
        m4t_route_signature_update(tile_sigs, all_W1, scratch, T, D, D);
        free(all_W1);
    }
    printf("Signatures computed\n");

    /* Scratch buffers */
    m4t_mtfp_t* z1 = malloc((size_t)H_proj * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* h1 = malloc((size_t)H_proj * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* po = malloc((size_t)D * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* ln_out = malloc((size_t)D * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* tile_out = malloc((size_t)D * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* tile_h = malloc((size_t)D * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* fo = malloc((size_t)D * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* hz = malloc((size_t)D * 2 * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* hh = malloc((size_t)D * 2 * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* lo = malloc((size_t)n_classes * sizeof(m4t_mtfp_t));
    (void)Dp1;  /* no longer needed for Hamming routing */

    /* Inference loop */
    int correct = 0;
    m4t_mtfp_t eps = 1;  /* ~1e-5 in MTFP real units */

    for (int s = 0; s < nv; s++) {
        m4t_mtfp_t* x = xv + (size_t)s * input_dim;

        /* Projection: z1 = x @ pW1^T + pb1 */
        m4t_mtfp_matmul_bt(z1, x, pW1, 1, input_dim, H_proj);
        m4t_mtfp_bias_add(z1, pb1, 1, H_proj);
        m4t_mtfp_gelu(h1, z1, H_proj);
        m4t_mtfp_matmul_bt(po, h1, pW2, 1, H_proj, D);
        m4t_mtfp_bias_add(po, pb2, 1, D);

        /* LayerNorm */
        m4t_mtfp_layernorm(ln_out, po, ln_w, ln_b, eps, 1, D);

        /* Routing: score[t] = dot(ln_out, sig_t) via ternary matmul.
         * This is the correct approach — trix-z uses MTFP dot products
         * for routing, not Hamming distance on sign-extracted signatures. */
        m4t_mtfp_t scores_mtfp[T];
        m4t_mtfp_ternary_matmul_bt(scores_mtfp, ln_out, tile_sigs, 1, D, T);

        int32_t scores[T];
        for (int t = 0; t < T; t++) scores[t] = (int32_t)scores_mtfp[t];

        m4t_route_decision_t decisions[K];
        m4t_route_topk_abs(decisions, scores, T, K);

        /* FFN: apply selected tiles */
        m4t_mtfp_vec_zero(fo, D);
        for (int sel = 0; sel < K; sel++) {
            int tidx = decisions[sel].tile_idx;
            m4t_trit_t sign = decisions[sel].sign;
            if (tidx < 0) continue;

            /* tile FFN: W1 @ ln_out → fan_in_normalize → bias → GELU → W2 → bias */
            m4t_mtfp_ternary_matmul_bt(tile_h, ln_out, tile_W1_packed[tidx], 1, D, D);
            m4t_mtfp_fan_in_normalize(tile_h, D, D);  /* prevent GELU saturation */
            m4t_mtfp_bias_add(tile_h, tile_b1[tidx], 1, D);
            m4t_mtfp_gelu(tile_h, tile_h, D);
            m4t_mtfp_ternary_matmul_bt(tile_out, tile_h, tile_W2_packed[tidx], 1, D, D);
            m4t_mtfp_bias_add(tile_out, tile_b2[tidx], 1, D);

            /* Signed accumulation */
            if (sign == 1)
                m4t_mtfp_vec_add_inplace(fo, tile_out, D);
            else if (sign == -1)
                m4t_mtfp_vec_sub_inplace(fo, tile_out, D);
        }

        /* Scale and residual */
        m4t_mtfp_vec_scale(fo, fo, output_scale, D);
        m4t_mtfp_vec_add_inplace(fo, po, D);  /* residual */

        /* Head: fo → hz → GELU → lo */
        m4t_mtfp_matmul_bt(hz, fo, hW1, 1, D, D * 2);
        m4t_mtfp_bias_add(hz, hb1, 1, D * 2);
        m4t_mtfp_gelu(hh, hz, D * 2);
        m4t_mtfp_matmul_bt(lo, hh, hW2, 1, D * 2, n_classes);
        m4t_mtfp_bias_add(lo, hb2, 1, n_classes);

        /* Argmax */
        int pred;
        m4t_mtfp_argmax(&pred, lo, 1, n_classes);
        if (pred == yv[s]) correct++;

        if (s > 0 && s % 1000 == 0) {
            printf("  %d/%d — running accuracy: %.2f%%\n",
                   s, nv, (float)correct / (float)s * 100.0f);
        }
    }

    float accuracy = (float)correct / (float)nv;
    printf("\nM4T MNIST inference: %d/%d correct = %.2f%%\n", correct, nv, accuracy * 100.0f);
    printf("trix-z reference:    95.42%%\n");
    printf("Difference:          %+.2f%%\n", (accuracy - 0.9542f) * 100.0f);

    /* Cleanup */
    free(xv); free(yv);
    free(pW1); free(pb1); free(pW2); free(pb2);
    free(ln_w); free(ln_b);
    for (int t = 0; t < T; t++) {
        free(tile_W1[t]); free(tile_b1[t]);
        free(tile_W2[t]); free(tile_b2[t]);
        free(tile_W1_packed[t]); free(tile_W2_packed[t]);
    }
    free(tile_W1); free(tile_b1); free(tile_W2); free(tile_b2);
    free(tile_W1_packed); free(tile_W2_packed);
    free(tile_sigs);
    free(hW1); free(hb1); free(hW2); free(hb2);
    free(z1); free(h1); free(po); free(ln_out);
    free(tile_out); free(tile_h); free(fo);
    free(hz); free(hh); free(lo);

    return (accuracy >= 0.90f) ? 0 : 1;
}
