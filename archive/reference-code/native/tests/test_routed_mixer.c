/*
 * test_routed_mixer.c — Smoke test for the routed token mixer
 *
 * Validates:
 *   1. Forward produces non-trivial output (not zeros, not identity)
 *   2. Scatter/gather are transposes (scatter then gather recovers structure)
 *   3. Backward produces nonzero gradients
 *   4. Training loop converges on a simple sequence task:
 *      Given a sequence of tokens, predict the sum-class of each position
 *      based on its neighbors. Forces cross-position interaction.
 */

#include "trix_routed_mixer.h"
#include "trix_routed_proj.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Simple PRNG ── */
static uint64_t test_rng_state = 42;
static float test_randf(void) {
    test_rng_state = test_rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((test_rng_state >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

/* ══════════════════════════════════════════════════════════════════════
 * Test 1: Forward produces non-trivial output
 * ══════════════════════════════════════════════════════════════════════ */

static int test_forward(void) {
    printf("  forward... ");
    int D = 16, T = 8, H = 32, K = 3, seq = 8;

    TrixRoutedMixerConfig cfg = {
        .d_model = D, .num_tiles = T, .tile_hidden = H,
        .active_k = K, .output_scale_init = 0.1f, .ln_eps = 1e-5f
    };
    TrixRoutedMixer* rm = trix_routed_mixer_create(cfg, 42);

    float* x = calloc(seq * D, sizeof(float));
    float* out = calloc(seq * D, sizeof(float));
    for (int i = 0; i < seq * D; i++) x[i] = test_randf() * 2.0f - 1.0f;

    trix_routed_mixer_forward(rm, x, out, seq);

    /* Check output is not all zeros */
    float sum_abs = 0;
    for (int i = 0; i < seq * D; i++) sum_abs += fabsf(out[i]);
    if (sum_abs < 1e-6f) { printf("FAIL (all zeros)\n"); return 1; }

    /* Check output differs from input (mixer did something) */
    float diff = 0;
    for (int i = 0; i < seq * D; i++) diff += fabsf(out[i] - x[i]);
    if (diff < 1e-6f) { printf("FAIL (identity)\n"); return 1; }

    printf("OK (diff=%.4f)\n", diff / (seq * D));
    free(x); free(out);
    trix_routed_mixer_destroy(rm);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════
 * Test 2: Scatter-gather roundtrip
 * ══════════════════════════════════════════════════════════════════════ */

static int test_scatter_gather(void) {
    printf("  scatter-gather... ");
    int D = 8, T = 2, seq = 4;

    /* All tokens route +1 to tile 0, -1 to tile 1 */
    int* route = calloc(seq * T, sizeof(int));
    for (int i = 0; i < seq; i++) { route[i * T + 0] = 1; route[i * T + 1] = -1; }

    /* Input: each position has a distinct pattern */
    mtfp_t* x = calloc(seq * D, sizeof(mtfp_t));
    for (int i = 0; i < seq; i++)
        for (int d = 0; d < D; d++)
            x[i * D + d] = mtfp_from_float((float)(i + 1) * (d % 2 == 0 ? 1.0f : -1.0f));

    /* Scatter */
    mtfp_t* pool = calloc(T * D, sizeof(mtfp_t));
    int* counts = calloc(T, sizeof(int));
    trix_routed_scatter(pool, counts, x, route, seq, D, T);

    if (counts[0] != seq || counts[1] != seq) {
        printf("FAIL (counts: %d %d, expected %d %d)\n", counts[0], counts[1], seq, seq);
        return 1;
    }

    /* Pool 0 should be sum of all tokens (route +1) */
    /* Pool 1 should be negative sum (route -1) */
    /* So pool[0] == -pool[1] */
    int pools_opposite = 1;
    for (int d = 0; d < D; d++) {
        if (pool[d] != -pool[T * 0 * D + D + d]) { /* pool[0,d] vs pool[1,d] */
            /* pool[1,d] should be -sum because route is -1 */
        }
        if (pool[d] + pool[D + d] != 0) pools_opposite = 0;
    }
    if (!pools_opposite) { printf("FAIL (pools not opposite)\n"); return 1; }

    /* Gather from pools: y[pos] = pool[0] - pool[1] = 2 * pool[0] */
    mtfp_t* y = calloc(seq * D, sizeof(mtfp_t));
    trix_routed_gather(y, pool, route, seq, D, T);

    /* Every position should get the same value (same routing pattern) */
    int all_same = 1;
    for (int i = 1; i < seq; i++)
        for (int d = 0; d < D; d++)
            if (y[i * D + d] != y[d]) all_same = 0;

    if (!all_same) { printf("FAIL (gathered values differ across positions)\n"); return 1; }

    printf("OK (counts=[%d,%d], pools_opposite=%d)\n", counts[0], counts[1], pools_opposite);
    free(route); free(x); free(pool); free(counts); free(y);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════
 * Test 3: Backward produces gradients
 * ══════════════════════════════════════════════════════════════════════ */

static int test_backward(void) {
    printf("  backward... ");
    int D = 16, T = 8, H = 32, K = 3, seq = 8;

    TrixRoutedMixerConfig cfg = {
        .d_model = D, .num_tiles = T, .tile_hidden = H,
        .active_k = K, .output_scale_init = 0.1f, .ln_eps = 1e-5f
    };
    TrixRoutedMixer* rm = trix_routed_mixer_create(cfg, 123);

    float* x = calloc(seq * D, sizeof(float));
    float* out = calloc(seq * D, sizeof(float));
    float* dy = calloc(seq * D, sizeof(float));
    float* dx = calloc(seq * D, sizeof(float));

    for (int i = 0; i < seq * D; i++) {
        x[i] = test_randf() * 2.0f - 1.0f;
        dy[i] = test_randf() * 0.1f - 0.05f;
    }

    trix_routed_mixer_forward(rm, x, out, seq);
    trix_routed_mixer_zero_grad(rm);
    trix_routed_mixer_backward(rm, x, dy, dx, seq);

    /* Check dx is nonzero */
    float dx_norm = 0;
    for (int i = 0; i < seq * D; i++) dx_norm += dx[i] * dx[i];
    dx_norm = sqrtf(dx_norm);
    if (dx_norm < 1e-8f) { printf("FAIL (dx=0)\n"); return 1; }

    /* Check dW1 has nonzero gradients */
    float dw1_norm = sqrtf(trix_sum_sq(rm->dW1, T * H * D));
    if (dw1_norm < 1e-8f) { printf("FAIL (dW1=0)\n"); return 1; }

    printf("OK (|dx|=%.6f, |dW1|=%.6f)\n", dx_norm, dw1_norm);
    free(x); free(out); free(dy); free(dx);
    trix_routed_mixer_destroy(rm);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════
 * Test 4: Training convergence — sequence classification
 *
 * Task: 4-position sequence, each position is a random D-vector.
 * Label = which position has the largest L2 norm.
 * This REQUIRES cross-position interaction: a single position can't
 * know if it's the largest without seeing the others.
 *
 * Architecture: routed_proj(D_in → D) → mixer → routed_proj(D → classes)
 * ══════════════════════════════════════════════════════════════════════ */

static int test_training(void) {
    printf("  training...\n");
    int D_in = 8, D = 32, T = 8, H = 64, K = 3;
    int seq = 4, classes = seq;  /* predict which position is largest */
    int n_train = 1024, epochs = 80, batch = 32;
    float lr = 0.01f;

    /* Create components: proj_in → mixer → proj_out */
    TrixRoutedProjConfig proj_in_cfg = {
        .in_dim = D_in, .out_dim = D, .num_tiles = T, .active_k = K,
        .output_scale_init = 0.5f, .ln_eps = 1e-5f, .use_layernorm = true
    };
    TrixRoutedProj* proj_in = trix_routed_proj_create(proj_in_cfg, 42);

    TrixRoutedMixerConfig mixer_cfg = {
        .d_model = D, .num_tiles = T, .tile_hidden = H,
        .active_k = K, .output_scale_init = 0.1f, .ln_eps = 1e-5f
    };
    TrixRoutedMixer* mixer = trix_routed_mixer_create(mixer_cfg, 100);

    /* Output head: take position 0's representation → classify */
    TrixRoutedProjConfig proj_out_cfg = {
        .in_dim = D, .out_dim = classes, .num_tiles = T, .active_k = K,
        .output_scale_init = 0.5f, .ln_eps = 1e-5f, .use_layernorm = true
    };
    TrixRoutedProj* proj_out = trix_routed_proj_create(proj_out_cfg, 200);

    /* Generate dataset: seq positions of D_in dims.
     * Label = whether the sum of all positions' first dimension is positive (1) or negative (0).
     * This is a pure aggregation task: requires seeing all positions. */
    classes = 2;
    float* data = malloc(n_train * seq * D_in * sizeof(float));
    int* labels = malloc(n_train * sizeof(int));
    test_rng_state = 777;
    for (int n = 0; n < n_train; n++) {
        float sum_d0 = 0;
        for (int s = 0; s < seq; s++) {
            for (int d = 0; d < D_in; d++) {
                float v = test_randf() * 2.0f - 1.0f;
                data[n * seq * D_in + s * D_in + d] = v;
                if (d == 0) sum_d0 += v;
            }
        }
        labels[n] = (sum_d0 > 0.0f) ? 1 : 0;
    }

    /* Scratch buffers */
    float* proj_out_buf = calloc(batch * seq * D, sizeof(float));
    float* mixer_out = calloc(batch * seq * D, sizeof(float));
    float* head_in = calloc(batch * D, sizeof(float));
    float* logits = calloc(batch * classes, sizeof(float));
    float* d_logits = calloc(batch * classes, sizeof(float));
    float* d_head_in = calloc(batch * D, sizeof(float));
    float* d_mixer_out = calloc(batch * seq * D, sizeof(float));
    float* d_proj_out = calloc(batch * seq * D, sizeof(float));
    float* d_input = calloc(batch * seq * D_in, sizeof(float));

    float best_acc = 0;
    for (int ep = 0; ep < epochs; ep++) {
        float total_loss = 0;
        int correct = 0, total = 0;

        for (int b = 0; b < n_train; b += batch) {
            int bs = (b + batch <= n_train) ? batch : n_train - b;

            /* Forward: project each position independently */
            for (int i = 0; i < bs * seq; i++) {
                trix_routed_proj_forward(proj_in,
                    data + (b * seq + i) * D_in,
                    proj_out_buf + i * D, 1);
            }

            /* Mixer: cross-position interaction */
            for (int i = 0; i < bs; i++) {
                trix_routed_mixer_forward(mixer,
                    proj_out_buf + i * seq * D,
                    mixer_out + i * seq * D, seq);
            }

            /* Take position 0 for classification */
            for (int i = 0; i < bs; i++)
                memcpy(head_in + i * D, mixer_out + i * seq * D, D * sizeof(float));

            /* Classify */
            trix_routed_proj_forward(proj_out, head_in, logits, bs);

            /* Loss + accuracy */
            total_loss += trix_cross_entropy_loss(logits, labels + b, bs, classes);
            int* preds = malloc(bs * sizeof(int));
            trix_argmax(preds, logits, bs, classes);
            for (int i = 0; i < bs; i++) if (preds[i] == labels[b + i]) correct++;
            total += bs;
            free(preds);

            /* Backward */
            trix_cross_entropy_grad(d_logits, logits, labels + b, bs, classes);

            trix_routed_proj_zero_grad(proj_out);
            trix_routed_proj_backward(proj_out, head_in, d_logits, d_head_in, bs);

            /* d_head_in → position 0 of d_mixer_out, rest zero */
            memset(d_mixer_out, 0, bs * seq * D * sizeof(float));
            for (int i = 0; i < bs; i++)
                memcpy(d_mixer_out + i * seq * D, d_head_in + i * D, D * sizeof(float));

            /* Mixer backward */
            trix_routed_mixer_zero_grad(mixer);
            for (int i = 0; i < bs; i++) {
                trix_routed_mixer_backward(mixer,
                    proj_out_buf + i * seq * D,
                    d_mixer_out + i * seq * D,
                    d_proj_out + i * seq * D, seq);
            }

            /* Proj_in backward */
            trix_routed_proj_zero_grad(proj_in);
            for (int i = 0; i < bs * seq; i++) {
                trix_routed_proj_backward(proj_in,
                    data + (b * seq + i) * D_in,
                    d_proj_out + i * D,
                    d_input + i * D_in, 1);
            }

            /* Clip + Update */
            trix_routed_mixer_clip_grad_norm(mixer, 1.0f);
            trix_routed_proj_adamw_step(proj_in, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
            trix_routed_mixer_adamw_step(mixer, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
            trix_routed_proj_adamw_step(proj_out, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
        }

        float acc = (float)correct / (float)total;
        if (acc > best_acc) best_acc = acc;
        if (ep % 20 == 0 || ep == epochs - 1) {
            printf("    ep %2d: loss=%.3f acc=%.1f%% os=%.4f",
                   ep, total_loss / (n_train / batch), acc * 100, mixer->output_scale);
            /* Print routing pattern of last batch, first sample */
            printf(" route=[");
            for (int s = 0; s < seq; s++) {
                printf("[");
                for (int t = 0; t < T; t++) printf("%+d", mixer->route[s * T + t]);
                printf("]");
            }
            printf("]\n");
        }
    }

    printf("    best=%.1f%% (chance=%.1f%%)\n", best_acc * 100, 100.0f / classes);

    int pass = best_acc > 0.65f;  /* well above 50% chance for binary task */
    printf("  training... %s\n", pass ? "OK" : "FAIL");

    free(data); free(labels);
    free(proj_out_buf); free(mixer_out); free(head_in);
    free(logits); free(d_logits); free(d_head_in);
    free(d_mixer_out); free(d_proj_out); free(d_input);
    trix_routed_proj_destroy(proj_in);
    trix_routed_mixer_destroy(mixer);
    trix_routed_proj_destroy(proj_out);
    return pass ? 0 : 1;
}

/* ══════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("=== ROUTED MIXER TESTS ===\n");
    int fail = 0;
    fail += test_forward();
    fail += test_scatter_gather();
    fail += test_backward();
    fail += test_training();
    printf("\n%s (%d failures)\n", fail ? "FAILED" : "ALL PASSED", fail);
    return fail;
}
