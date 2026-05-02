/*
 * denoise_probe.c — Phase B Gate 2: mechanism test for H1.
 *
 * H1 ("implicit denoising via random ternary projection"): random ternary
 * projection at sig_dim = D outperforms identity at sig_dim = D because
 * informative input dims carry class-correlated signal that survives the
 * mix, while noise input dims contribute incoherent (zero-mean) terms
 * that average out across the class-mean bank construction.
 *
 * Mechanism prediction: per-output-dim of random R, the class-discrimination
 * the dim provides is positively correlated with how "well-aligned" R[j]
 * is with the informative subspace.
 *
 *   x[j] = stddev_c (R[j] · P_c)        — across classes c
 *           where P_c is the class-c prototype (zero outside informative
 *           dims). This measures how strongly R[j] separates classes
 *           via prototype alignment. R[j] · P_c = 0 for all c if R[j]
 *           is random-walk-canceling on informative dims; large spread
 *           if R[j] aligns with the informative subspace's class-distinct
 *           structure.
 *
 *   y[j] = inter-class spread of class-mean projection at output dim j,
 *           measured from training data. Range of per-class average
 *           projection value, scaled to permille to avoid integer-division
 *           truncation that bit a previous version of this probe.
 *
 * If H1 holds, Pearson r(x, y) > 0 with p < 0.01.
 *
 * Configuration: synthetic prototype benchmark, D=64 (16 informative + 48
 * noise), C=10, sig_dim = D = 64, 100 random R samples.
 */

#include "synth_proto.h"
#include "gesh_train.h"
#include "m4t_types.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_R_SAMPLES 100
#define N_TRAIN     2000
#define SIG_DIM     64

/* x[j] = stddev across classes of (R[j] · P_c). Returns squared sum
 * for variance computation — caller divides by C. */
static double prototype_alignment_stddev(
    const m4t_trit_t* R_row,
    const m4t_trit_t* prototypes,  /* [C × D] */
    int C, int D)
{
    int proj_per_class[64];
    int sum = 0;
    for (int c = 0; c < C; c++) {
        const m4t_trit_t* Pc = prototypes + (size_t)c * D;
        int acc = 0;
        for (int i = 0; i < D; i++) {
            acc += (int)R_row[i] * (int)Pc[i];
        }
        proj_per_class[c] = acc;
        sum += acc;
    }
    double mean = (double)sum / (double)C;
    double sq = 0.0;
    for (int c = 0; c < C; c++) {
        double d = (double)proj_per_class[c] - mean;
        sq += d * d;
    }
    return sqrt(sq / (double)C);
}

/* y[j] = max-min spread of per-class average projected accumulator at
 * output dim j (in permille of n_train, to keep precision under integer
 * arithmetic). Per-class-average is computed BEFORE sign-quantization. */
static int per_dim_spread_permille(
    const int32_t* projected_accs,  /* [n_train × sig_dim] raw int32 accs */
    const int* train_lbl,
    int n_train, int sig_dim, int n_classes, int j)
{
    long long per_class_sum[16] = {0};
    int per_class_count[16] = {0};
    for (int i = 0; i < n_train; i++) {
        int c = train_lbl[i];
        per_class_sum[c] += (long long)projected_accs[(size_t)i * sig_dim + j];
        per_class_count[c]++;
    }
    /* per-class mean as permille of average per-sample acc magnitude. */
    long long max_v = LLONG_MIN, min_v = LLONG_MAX;
    for (int c = 0; c < n_classes; c++) {
        if (per_class_count[c] == 0) continue;
        /* scale by 1000 before division so we keep two sig digits. */
        long long avg = per_class_sum[c] * 1000 / per_class_count[c];
        if (avg > max_v) max_v = avg;
        if (avg < min_v) min_v = avg;
    }
    return (int)(max_v - min_v);
}

/* Project the training set; keep RAW int32 accumulators (no quantize). */
static void project_train_acc(
    int32_t* out_accs, const m4t_trit_t* R,
    const m4t_trit_t* train, int n_train,
    int sig_dim, int input_dim)
{
    for (int i = 0; i < n_train; i++) {
        const m4t_trit_t* x = train + (size_t)i * input_dim;
        int32_t* s = out_accs + (size_t)i * sig_dim;
        for (int oi = 0; oi < sig_dim; oi++) {
            const m4t_trit_t* r = R + (size_t)oi * input_dim;
            int32_t acc = 0;
            for (int j = 0; j < input_dim; j++)
                acc += (int32_t)r[j] * (int32_t)x[j];
            s[oi] = acc;
        }
    }
}

int main(void) {
    synth_proto_config_t cfg = synth_proto_default();
    int D = cfg.input_dim;        /* 64 */
    int K = cfg.informative_dim;  /* 16 */
    int C = cfg.n_classes;        /* 10 */
    (void)K;

    m4t_trit_t* protos = malloc((size_t)C * D * sizeof(m4t_trit_t));
    synth_proto_generate_prototypes(protos, &cfg);
    m4t_trit_t* train = malloc((size_t)N_TRAIN * D * sizeof(m4t_trit_t));
    int* train_lbl = malloc((size_t)N_TRAIN * sizeof(int));
    synth_proto_generate_samples(train, train_lbl, N_TRAIN, protos,
                                   &cfg, 0x11111111u);

    long long n_obs = 0;
    double sum_x = 0, sum_y = 0;
    double sum_xx = 0, sum_yy = 0, sum_xy = 0;

    int bins_lo = 0, bins_mid = 0, bins_hi = 0;
    double sep_lo = 0, sep_mid = 0, sep_hi = 0;

    m4t_trit_t* R = malloc((size_t)SIG_DIM * D * sizeof(m4t_trit_t));
    int32_t* projected = malloc((size_t)N_TRAIN * SIG_DIM * sizeof(int32_t));

    printf("# Phase B Gate 2: H1 mechanism test (denoising via random ternary R)\n");
    printf("# D=%d (K=%d informative + %d noise), C=%d, sig_dim=%d, n_train=%d\n",
           D, K, D - K, C, SIG_DIM, N_TRAIN);
    printf("# R samples: %d. Per sample: %d output dims scored.\n",
           N_R_SAMPLES, SIG_DIM);
    printf("# x[j] = stddev_c (R[j] · P_c)  — alignment with informative subspace's\n");
    printf("#                                  class-distinct structure\n");
    printf("# y[j] = max-min spread of per-class proj-acc-mean at output dim j (permille)\n");
    fflush(stdout);

    uint32_t seeds[8] = {
        0xc0ffeebbu, 0xa5a5a5a5u, 0xfeedfaceu, 0xb16b00b5u,
        0x13579bdfu, 0xc7c7c7c7u, 0xb22bd00du, 0xdeadc0deu };

    for (int rs = 0; rs < N_R_SAMPLES; rs++) {
        uint32_t seed = seeds[rs % 8] ^ ((uint32_t)rs * 0x9e3779b9u);
        gesh_init_random_projection(R, SIG_DIM, D, seed);
        project_train_acc(projected, R, train, N_TRAIN, SIG_DIM, D);

        for (int j = 0; j < SIG_DIM; j++) {
            double x = prototype_alignment_stddev(R + (size_t)j * D,
                                                    protos, C, D);
            int y = per_dim_spread_permille(projected, train_lbl,
                                              N_TRAIN, SIG_DIM, C, j);

            n_obs++;
            sum_x += x;
            sum_y += (double)y;
            sum_xx += x * x;
            sum_yy += (double)y * (double)y;
            sum_xy += x * (double)y;

            /* Stratify x into low/mid/hi tertiles for sanity check. */
            if      (x < 1.5) { bins_lo++;  sep_lo  += y; }
            else if (x < 3.0) { bins_mid++; sep_mid += y; }
            else              { bins_hi++;  sep_hi  += y; }
        }
    }

    double n = (double)n_obs;
    double mean_x = sum_x / n;
    double mean_y = sum_y / n;
    double cov_xy = sum_xy / n - mean_x * mean_y;
    double var_x  = sum_xx / n - mean_x * mean_x;
    double var_y  = sum_yy / n - mean_y * mean_y;
    double r = (var_x > 0 && var_y > 0)
                 ? cov_xy / (sqrt(var_x) * sqrt(var_y)) : 0.0;
    double t = (fabs(r) < 1.0)
                 ? r * sqrt((n - 2.0) / (1.0 - r * r))
                 : 0.0;

    printf("\nObservations: %lld\n", n_obs);
    printf("mean(x)=%.3f   stddev(x)=%.3f\n", mean_x, sqrt(var_x));
    printf("mean(y)=%.1f   stddev(y)=%.1f (permille spread)\n",
           mean_y, sqrt(var_y));
    printf("\n");
    printf("Pearson r(x, y) = %+.4f\n", r);
    printf("t-statistic     = %+.2f  (df = %lld)\n", t, n_obs - 2);
    printf("Approx |t| > 2.58 → p < 0.01;  |t| > 3.29 → p < 0.001\n");

    printf("\nStratification (mean y by tertile of x):\n");
    if (bins_lo)  printf("  x ∈ [0.0, 1.5)   (low alignment)  : n=%-6d  mean(y)=%.1f\n",
                          bins_lo,  sep_lo / bins_lo);
    if (bins_mid) printf("  x ∈ [1.5, 3.0)   (mid alignment)  : n=%-6d  mean(y)=%.1f\n",
                          bins_mid, sep_mid / bins_mid);
    if (bins_hi)  printf("  x ∈ [3.0, +∞)    (high alignment) : n=%-6d  mean(y)=%.1f\n",
                          bins_hi,  sep_hi / bins_hi);

    printf("\nVerdict: ");
    if (r > 0 && fabs(t) > 2.58) {
        printf("H1 SUPPORTED — positive correlation with p < 0.01.\n");
    } else if (r > 0 && fabs(t) > 1.96) {
        printf("H1 INCONCLUSIVE — weak positive correlation (p in [0.01, 0.05]).\n");
    } else if (r > 0) {
        printf("H1 INCONCLUSIVE — directional but not significant (p > 0.05).\n");
    } else {
        printf("H1 FALSIFIED — zero or negative correlation.\n");
    }

    free(R);
    free(projected);
    free(train);
    free(train_lbl);
    free(protos);
    return 0;
}
