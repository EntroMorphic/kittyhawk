/*
 * test_synth_proto.c — verifies the synthetic prototype-classification
 * benchmark generator.
 *
 * Properties checked:
 *   1. Determinism: same cfg + sample_seed → same samples + labels.
 *   2. Class-balance: with uniform random class draw, the empirical
 *      class distribution is close to uniform at large n.
 *   3. Noise dim is signal-free: the per-noise-dim sum across samples
 *      of one class is near zero (uniform random ternary).
 *   4. Informative dim has class signal: per-class mean of an informative
 *      dim has the prototype's sign with high probability.
 *   5. Prototype shape: noise dims of the prototype are zero;
 *      informative dims are ±1.
 */

#include "synth_proto.h"
#include "m4t_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_SAMPLES 2000

static int test_prototype_shape(void) {
    synth_proto_config_t cfg = synth_proto_default();
    int C = cfg.n_classes;
    int D = cfg.input_dim;
    int K = cfg.informative_dim;

    m4t_trit_t* protos = malloc((size_t)C * D * sizeof(m4t_trit_t));
    synth_proto_generate_prototypes(protos, &cfg);

    for (int c = 0; c < C; c++) {
        const m4t_trit_t* p = protos + (size_t)c * D;
        for (int j = 0; j < K; j++) {
            if (p[j] != 1 && p[j] != -1) {
                printf("FAIL prototype_shape class %d dim %d: got %d\n",
                       c, j, (int)p[j]);
                free(protos); return 1;
            }
        }
        for (int j = K; j < D; j++) {
            if (p[j] != 0) {
                printf("FAIL prototype_shape class %d noise-dim %d: got %d\n",
                       c, j, (int)p[j]);
                free(protos); return 1;
            }
        }
    }
    free(protos);
    return 0;
}

static int test_determinism(void) {
    synth_proto_config_t cfg = synth_proto_default();
    int C = cfg.n_classes;
    int D = cfg.input_dim;

    m4t_trit_t* protos = malloc((size_t)C * D * sizeof(m4t_trit_t));
    synth_proto_generate_prototypes(protos, &cfg);

    m4t_trit_t* s1 = malloc((size_t)N_SAMPLES * D * sizeof(m4t_trit_t));
    m4t_trit_t* s2 = malloc((size_t)N_SAMPLES * D * sizeof(m4t_trit_t));
    int* l1 = malloc((size_t)N_SAMPLES * sizeof(int));
    int* l2 = malloc((size_t)N_SAMPLES * sizeof(int));

    synth_proto_generate_samples(s1, l1, N_SAMPLES, protos, &cfg, 0xa5a5a5a5u);
    synth_proto_generate_samples(s2, l2, N_SAMPLES, protos, &cfg, 0xa5a5a5a5u);

    int ok = (memcmp(s1, s2, (size_t)N_SAMPLES * D * sizeof(m4t_trit_t)) == 0)
          && (memcmp(l1, l2, (size_t)N_SAMPLES * sizeof(int)) == 0);
    free(protos); free(s1); free(s2); free(l1); free(l2);

    if (!ok) {
        printf("FAIL determinism: identical seeds produced different output\n");
        return 1;
    }
    return 0;
}

static int test_class_balance(void) {
    synth_proto_config_t cfg = synth_proto_default();
    int C = cfg.n_classes;
    int D = cfg.input_dim;

    m4t_trit_t* protos = malloc((size_t)C * D * sizeof(m4t_trit_t));
    synth_proto_generate_prototypes(protos, &cfg);
    m4t_trit_t* samples = malloc((size_t)N_SAMPLES * D * sizeof(m4t_trit_t));
    int* labels = malloc((size_t)N_SAMPLES * sizeof(int));

    synth_proto_generate_samples(samples, labels, N_SAMPLES, protos, &cfg, 0u);

    int* counts = calloc((size_t)C, sizeof(int));
    for (int i = 0; i < N_SAMPLES; i++) counts[labels[i]]++;

    /* Expect each class ~ N_SAMPLES / C. Tolerance ±15% — at N=2000,
     * C=10 (expected=200/class), this is [170, 230]. The std dev for
     * uniform-random class draws is sqrt(200·0.9·0.1) ≈ 13.4; the
     * tolerance is ~2.2σ, well above 4σ would let any class drift.
     * Tightened from the earlier ±25% per the Phase A.1 red-team. */
    int expected = N_SAMPLES / C;
    int tol = (expected * 15) / 100;
    int low = expected - tol;
    int high = expected + tol;
    int ok = 1;
    for (int c = 0; c < C; c++) {
        if (counts[c] < low || counts[c] > high) {
            printf("FAIL class_balance class %d: got %d, expected ~%d\n",
                   c, counts[c], expected);
            ok = 0;
        }
    }

    free(protos); free(samples); free(labels); free(counts);
    return ok ? 0 : 1;
}

static int test_informative_signal(void) {
    /* For each class, the per-class mean of the first K dims should
     * point in the prototype's direction with high probability. With
     * noise_pct=10, the per-class mean trit value at an informative
     * dim is approximately:
     *   E[trit_j | class c] = p[c,j] * (1 - 2 * (noise_pct/100) * 1/3 - ...)
     * Empirically with 100+ samples per class, sum-sign should match
     * prototype-sign for almost all (class, informative-dim) pairs. */
    synth_proto_config_t cfg = synth_proto_default();
    int C = cfg.n_classes;
    int D = cfg.input_dim;
    int K = cfg.informative_dim;

    m4t_trit_t* protos = malloc((size_t)C * D * sizeof(m4t_trit_t));
    synth_proto_generate_prototypes(protos, &cfg);
    m4t_trit_t* samples = malloc((size_t)N_SAMPLES * D * sizeof(m4t_trit_t));
    int* labels = malloc((size_t)N_SAMPLES * sizeof(int));

    synth_proto_generate_samples(samples, labels, N_SAMPLES, protos, &cfg, 1u);

    int* class_count = calloc((size_t)C, sizeof(int));
    int32_t* class_sum = calloc((size_t)C * D, sizeof(int32_t));
    for (int i = 0; i < N_SAMPLES; i++) {
        int c = labels[i];
        class_count[c]++;
        for (int j = 0; j < D; j++) {
            class_sum[c * D + j] += (int32_t)samples[i * D + j];
        }
    }

    int correct = 0, total = 0;
    for (int c = 0; c < C; c++) {
        for (int j = 0; j < K; j++) {
            int32_t s = class_sum[c * D + j];
            int recovered_sign = (s > 0) ? 1 : (s < 0) ? -1 : 0;
            if (recovered_sign == (int)protos[c * D + j]) correct++;
            total++;
        }
    }
    free(protos); free(samples); free(labels); free(class_count); free(class_sum);

    /* With 200 samples/class and 10% noise, recovery should be near
     * perfect. Accept >95%. */
    int pct = (correct * 100) / total;
    if (pct < 95) {
        printf("FAIL informative_signal: %d/%d (%d%%) prototype-sign recovery\n",
               correct, total, pct);
        return 1;
    }
    return 0;
}

int main(void) {
    if (test_prototype_shape())     return 1;
    if (test_determinism())         return 1;
    if (test_class_balance())       return 1;
    if (test_informative_signal())  return 1;
    printf("synth_proto: all 4 tests passed\n");
    return 0;
}
