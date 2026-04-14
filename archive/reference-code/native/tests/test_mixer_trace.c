/*
 * test_mixer_trace.c — Frame-by-frame trace through the routed mixer
 *
 * Runs one forward pass with a tiny config (D=4, T=4, K=2, seq=4)
 * and prints every intermediate value from the struct's saved state.
 */

#include "trix_routed_mixer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void pv(const char* label, const float* v, int n) {
    printf("  %-16s [", label);
    for (int i = 0; i < n; i++) printf("%s%+8.4f", i ? " " : "", v[i]);
    printf(" ]\n");
}

static void pi(const char* label, const int* v, int n) {
    printf("  %-16s [", label);
    for (int i = 0; i < n; i++) printf("%s%+d", i ? " " : "", v[i]);
    printf(" ]\n");
}

static void pm(const char* label, const mtfp_t* v, int n) {
    printf("  %-16s [", label);
    for (int i = 0; i < n; i++) printf("%s%+8.4f", i ? " " : "", mtfp_to_float(v[i]));
    printf(" ]\n");
}

int main(void) {
    int D = 4, T = 4, H = 8, K = 2, seq = 4;

    printf("=== ROUTED MIXER TRACE ===\n");
    printf("D=%d  T=%d  H=%d  K=%d  seq=%d\n", D, T, H, K, seq);

    TrixRoutedMixerConfig cfg = {
        .d_model = D, .num_tiles = T, .tile_hidden = H,
        .active_k = K, .output_scale_init = 0.1f, .ln_eps = 1e-5f
    };
    TrixRoutedMixer* rm = trix_routed_mixer_create(cfg, 42);

    float x[16] = {
         1.0f,  0.5f,  0.2f,  0.1f,   /* pos 0: positive cluster */
        -0.8f, -0.6f, -0.3f, -0.1f,   /* pos 1: negative cluster */
         0.5f, -0.5f,  0.5f, -0.5f,   /* pos 2: alternating */
         0.1f,  0.1f,  0.1f,  0.1f,   /* pos 3: uniform */
    };
    float out[16];

    printf("\n── 0. INPUT ──\n");
    for (int s = 0; s < seq; s++) {
        char l[32]; snprintf(l, 32, "x[%d]", s);
        pv(l, x + s * D, D);
    }

    /* Run full forward — populates all scratch in the struct */
    trix_routed_mixer_forward(rm, x, out, seq);

    /* Now read the struct to trace every step */

    printf("\n── 1. LAYERNORM ──\n");
    for (int s = 0; s < seq; s++) {
        char l[32]; snprintf(l, 32, "x_norm[%d]", s);
        pv(l, rm->x_norm + s * D, D);
    }

    printf("\n── 2. SIGNATURES (ternary, weight-derived) ──\n");
    for (int t = 0; t < T; t++) {
        char l[32]; snprintf(l, 32, "sig[%d]", t);
        pv(l, rm->signatures + t * D, D);
    }

    printf("\n── 3. ROUTING SCORES (x_norm @ sig^T, MTFP x ternary = add/sub) ──\n");
    for (int s = 0; s < seq; s++) {
        char l[32]; snprintf(l, 32, "scores[%d]", s);
        pv(l, rm->scores + s * T, T);
    }

    printf("\n── 4. ROUTE (top-%d by |score|, ternary threshold) ──\n", K);
    for (int s = 0; s < seq; s++) {
        char l[32]; snprintf(l, 32, "route[%d]", s);
        pi(l, rm->route + s * T, T);
    }

    printf("\n── 5. SCATTER (pool tokens by route destination) ──\n");
    printf("  counts:          [");
    for (int t = 0; t < T; t++) printf("%s%d", t ? " " : "", rm->pool_counts[t]);
    printf(" ]\n");
    for (int t = 0; t < T; t++) {
        if (rm->pool_counts[t] == 0) { printf("  pool[%d]           EMPTY\n", t); continue; }
        char l[32]; snprintf(l, 32, "pool[%d] (n=%d)", t, rm->pool_counts[t]);
        /* pool was normalized in-place, show saved_pool (post-normalize, pre-FFN input) */
        pv(l, rm->saved_pool + t * D, D);
    }

    printf("\n── 6. TILE FFN (ternary matmul → GELU → ternary matmul) ──\n");
    for (int t = 0; t < T; t++) {
        if (rm->pool_counts[t] == 0) { printf("  tile %d: SKIPPED\n", t); continue; }
        printf("  tile %d:\n", t);
        char l[32];
        snprintf(l, 32, "  z1[%d]", t);
        pv(l, rm->saved_z1 + t * H, H);
        snprintf(l, 32, "  h1[%d]", t);
        pv(l, rm->saved_h1 + t * H, H);
        snprintf(l, 32, "  mixed[%d]", t);
        pv(l, rm->saved_mixed + t * D, D);
    }

    printf("\n── 7. GATHER (distribute tile outputs back to positions) ──\n");
    for (int s = 0; s < seq; s++) {
        char l[32]; snprintf(l, 32, "gathered[%d]", s);
        pm(l, rm->mgathered + s * D, D);
    }

    printf("\n── 8. SCALE + RESIDUAL (output_scale=%.4f) ──\n", rm->output_scale);
    for (int s = 0; s < seq; s++) {
        char l[32]; snprintf(l, 32, "out[%d]", s);
        pv(l, out + s * D, D);
    }

    printf("\n── 9. CORRECTION (out - x) ──\n");
    for (int s = 0; s < seq; s++) {
        printf("  pos %d:           [", s);
        for (int d = 0; d < D; d++)
            printf("%s%+8.4f", d ? " " : "", out[s * D + d] - x[s * D + d]);
        printf(" ]\n");
    }

    /* Which positions contributed to which tiles? */
    printf("\n── 10. ROUTING MAP ──\n");
    for (int t = 0; t < T; t++) {
        printf("  tile %d ←", t);
        for (int s = 0; s < seq; s++) {
            int r = rm->route[s * T + t];
            if (r == 1) printf(" pos%d(+)", s);
            else if (r == -1) printf(" pos%d(-)", s);
        }
        if (rm->pool_counts[t] == 0) printf(" (empty)");
        printf("\n");
    }
    printf("\n");
    for (int s = 0; s < seq; s++) {
        printf("  pos %d →", s);
        for (int t = 0; t < T; t++) {
            int r = rm->route[s * T + t];
            if (r == 1) printf(" tile%d(+)", t);
            else if (r == -1) printf(" tile%d(-)", t);
        }
        printf("\n");
    }

    trix_routed_mixer_destroy(rm);
    return 0;
}
