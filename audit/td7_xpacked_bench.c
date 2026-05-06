/*
 * tcd/audit/td7_xpacked_bench.c — wall-clock comparison of the three §20
 * matmul variants. Closes the wall-clock comparison TD-7's closeout
 * deferred ("primitive ships per project rule; wall-clock comparison vs
 * §20 deferred until a consumer demands it").
 *
 * Three kernels:
 *   m4t_ternary_dot_matmul_bt          — unpacked X (8 b/c) × unpacked W (8 b/c)
 *   m4t_ternary_5in8_matmul_bt         — unpacked X (8 b/c) × 5-in-8 W (1.6 b/c)
 *   m4t_ternary_5in8_matmul_xpacked_bt — 5-in-8 X (1.6 b/c) × 5-in-8 W (1.6 b/c)
 *
 * Question: at what (M, K, N) regime does X-packing pay off?
 *   - Small batch (inference): X is L1-resident; X-decode adds cost without
 *     saving meaningful bandwidth. X-packed should LOSE.
 *   - Large batch / KV cache (training): X grows with M*K bytes; eventually
 *     spills L1, then L2. X-packed reduces X bandwidth 5×. X-packed should
 *     break even, then WIN as M grows.
 *
 * Pre-committed gate: X-packed/§20 ratio should be < 1.0 at some M; the
 * crossover M is the headline.
 *
 * Bit-exactness verified per call (memcmp Y_xpacked vs Y_unpacked).
 *
 * Cache-flush + warmup discipline mirrored from tristate_strong_bench.
 */

#include "m4t_trit_pack.h"
#include "m4t_ternary_matmul.h"
#include "m4t_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

typedef struct { uint32_t s; } rng_t;
static void rng_init(rng_t* r, uint32_t seed) { r->s = seed ? seed : 0xdeadbeefu; }
static uint32_t rng_u32(rng_t* r) {
    uint32_t x = r->s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    r->s = x;
    return x;
}
static int rng_sign(rng_t* r) { return (rng_u32(r) & 1u) ? 1 : -1; }
static int rng_lt(rng_t* r, double p) {
    return ((double)(rng_u32(r) & 0xFFFFFFu) / (double)0x1000000) < p;
}
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void flush_caches(uint8_t* buf, size_t n) {
    volatile uint32_t sink = 0;
    for (size_t i = 0; i < n; i += 64) sink ^= buf[i];
    (void)sink;
}

typedef struct {
    int M, K, N, reps;
    double bytes_X_unp;     /* X size if unpacked (M*K bytes) */
    double bytes_X_pkd;     /* X size if 5-in-8 packed (M*Kp5 bytes) */
    double bytes_W_pkd;     /* W size 5-in-8 packed (N*Kp5 bytes) */
    const char* note;
} Config;

static Config make_cfg(int M, int K, int N, int reps, const char* note) {
    int Kp5 = (K + 4) / 5;
    Config c = { M, K, N, reps,
        (double)(M * K)   / (1024.0 * 1024.0),
        (double)(M * Kp5) / (1024.0 * 1024.0),
        (double)(N * Kp5) / (1024.0 * 1024.0),
        note
    };
    return c;
}

int main(void) {
    const size_t FLUSH_SIZE = 64 * 1024 * 1024;
    uint8_t* flush_buf = (uint8_t*)calloc(FLUSH_SIZE, 1);
    if (!flush_buf) { fprintf(stderr, "OOM flush buf\n"); return 1; }

    /* Sweep batch M from 1 (single-token inference) to large training shape.
     * K covers typical hidden sizes; N=64 is fixed (single output projection). */
    Config cfgs[] = {
        /* Inference shapes: small M, X is L1-resident. */
        make_cfg(   1,  1280,  64, 5000, "M=1,    K=1280  inference (single token)"),
        make_cfg(   8,  1280,  64, 2000, "M=8,    K=1280  inference (small batch)"),
        make_cfg(  64,  1280,  64,  500, "M=64,   K=1280  fine-tune (mid batch)"),
        make_cfg( 256,  1280,  64,  100, "M=256,  K=1280  training (X≈320 KB)"),
        make_cfg(1024,  1280,  64,   20, "M=1024, K=1280  training (X≈1.3 MB)"),
        /* Larger K, sweep M. */
        make_cfg(   1,  4480,  64, 1000, "M=1,    K=4480  inference"),
        make_cfg(  64,  4480,  64,  100, "M=64,   K=4480  X≈287 KB"),
        make_cfg( 256,  4480,  64,   50, "M=256,  K=4480  X≈1.1 MB (past L1)"),
        make_cfg(1024,  4480,  64,   10, "M=1024, K=4480  X≈4.6 MB (in L2)"),
        make_cfg(4096,  4480,  64,    3, "M=4096, K=4480  X≈18 MB (past L2)"),
        /* Very large K (LLM-scale hidden). */
        make_cfg(   1, 12800,  64,  500, "M=1,    K=12800 inference"),
        make_cfg(  64, 12800,  64,   30, "M=64,   K=12800 X≈820 KB"),
        make_cfg( 256, 12800,  64,   10, "M=256,  K=12800 X≈3.3 MB"),
        make_cfg(1024, 12800,  64,    3, "M=1024, K=12800 X≈13 MB (at L2)"),
    };
    int n_cfgs = (int)(sizeof(cfgs) / sizeof(cfgs[0]));

    printf("# TD-7 X-packed vs §20 wall-clock comparison\n");
    printf("# 3 kernels: unpacked-W (m4t_ternary_dot_matmul_bt),\n"
           "#            §20 (5-in-8 W, unpacked X),\n"
           "#            §20-xp (5-in-8 W AND X — TD-7 new).\n\n");
    printf("%-50s X_unp  X_pkd  W_pkd  reps   ms_dot   ms_§20   ms_xp    xp/§20  xp/dot   §20/dot\n",
        "config");

    for (int c = 0; c < n_cfgs; c++) {
        Config* cfg = &cfgs[c];
        int M = cfg->M, K = cfg->K, N = cfg->N;
        if (K % 80 != 0 || N % 4 != 0) {
            printf("[skip] cfg requires K%%80==0 N%%4==0 (K=%d N=%d)\n", K, N);
            continue;
        }
        int Kp5 = (K + 4) / 5;

        m4t_trit_t* X_unp = (m4t_trit_t*)calloc((size_t)M*K, sizeof(m4t_trit_t));
        m4t_trit_t* W_unp = (m4t_trit_t*)calloc((size_t)N*K, sizeof(m4t_trit_t));
        uint8_t*    X_pkd = (uint8_t*)calloc((size_t)M*Kp5, 1);
        uint8_t*    W_pkd = (uint8_t*)calloc((size_t)N*Kp5, 1);
        m4t_mtfp_t* Y_dot = (m4t_mtfp_t*)calloc((size_t)M*N, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* Y_20  = (m4t_mtfp_t*)calloc((size_t)M*N, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* Y_xp  = (m4t_mtfp_t*)calloc((size_t)M*N, sizeof(m4t_mtfp_t));

        if (!X_unp || !W_unp || !X_pkd || !W_pkd || !Y_dot || !Y_20 || !Y_xp) {
            fprintf(stderr, "OOM cfg %d\n", c);
            free(X_unp); free(W_unp); free(X_pkd); free(W_pkd);
            free(Y_dot); free(Y_20); free(Y_xp);
            continue;
        }

        rng_t rng; rng_init(&rng, (uint32_t)(c + 1) * 0x9E3779B1u);
        for (int i = 0; i < M*K; i++)
            X_unp[i] = rng_lt(&rng, 0.4) ? 0 : (m4t_trit_t)rng_sign(&rng);
        for (int i = 0; i < N*K; i++)
            W_unp[i] = rng_lt(&rng, 0.4) ? 0 : (m4t_trit_t)rng_sign(&rng);
        for (int i = 0; i < M; i++)
            m4t_pack_trits_5in8_1d(X_pkd + (size_t)i * Kp5, X_unp + (size_t)i * K, K);
        for (int j = 0; j < N; j++)
            m4t_pack_trits_5in8_1d(W_pkd + (size_t)j * Kp5, W_unp + (size_t)j * K, K);

        /* Bit-exactness gate. */
        m4t_ternary_dot_matmul_bt          (Y_dot, X_unp, W_unp, M, K, N);
        m4t_ternary_5in8_matmul_bt         (Y_20,  X_unp, W_pkd, M, K, N);
        m4t_ternary_5in8_matmul_xpacked_bt (Y_xp,  X_pkd, W_pkd, M, K, N);
        if (memcmp(Y_dot, Y_20, (size_t)M*N*sizeof(m4t_mtfp_t)) != 0 ||
            memcmp(Y_dot, Y_xp, (size_t)M*N*sizeof(m4t_mtfp_t)) != 0) {
            fprintf(stderr, "[ERROR] cfg %d: kernel mismatch\n", c);
            free(X_unp); free(W_unp); free(X_pkd); free(W_pkd);
            free(Y_dot); free(Y_20); free(Y_xp);
            continue;
        }

        /* Warm one rep. */
        flush_caches(flush_buf, FLUSH_SIZE);
        m4t_ternary_dot_matmul_bt(Y_dot, X_unp, W_unp, M, K, N);
        flush_caches(flush_buf, FLUSH_SIZE);
        m4t_ternary_5in8_matmul_bt(Y_20, X_unp, W_pkd, M, K, N);
        flush_caches(flush_buf, FLUSH_SIZE);
        m4t_ternary_5in8_matmul_xpacked_bt(Y_xp, X_pkd, W_pkd, M, K, N);

        /* Time. */
        double sum_dot = 0, sum_20 = 0, sum_xp = 0;
        for (int r = 0; r < cfg->reps; r++) {
            flush_caches(flush_buf, FLUSH_SIZE);
            double t0 = now_ms();
            m4t_ternary_dot_matmul_bt(Y_dot, X_unp, W_unp, M, K, N);
            sum_dot += now_ms() - t0;

            flush_caches(flush_buf, FLUSH_SIZE);
            t0 = now_ms();
            m4t_ternary_5in8_matmul_bt(Y_20, X_unp, W_pkd, M, K, N);
            sum_20 += now_ms() - t0;

            flush_caches(flush_buf, FLUSH_SIZE);
            t0 = now_ms();
            m4t_ternary_5in8_matmul_xpacked_bt(Y_xp, X_pkd, W_pkd, M, K, N);
            sum_xp += now_ms() - t0;
        }
        double m_dot = sum_dot / cfg->reps;
        double m_20  = sum_20  / cfg->reps;
        double m_xp  = sum_xp  / cfg->reps;
        double r_xp_20  = (m_20  > 0) ? m_xp / m_20  : 0;
        double r_xp_dot = (m_dot > 0) ? m_xp / m_dot : 0;
        double r_20_dot = (m_dot > 0) ? m_20 / m_dot : 0;

        printf("%-50s %4.2fM  %4.2fM  %4.2fM  %4d  %7.3f  %7.3f  %7.3f  %.3f   %.3f    %.3f\n",
            cfg->note, cfg->bytes_X_unp, cfg->bytes_X_pkd, cfg->bytes_W_pkd,
            cfg->reps, m_dot, m_20, m_xp, r_xp_20, r_xp_dot, r_20_dot);

        free(X_unp); free(W_unp); free(X_pkd); free(W_pkd);
        free(Y_dot); free(Y_20); free(Y_xp);
    }

    free(flush_buf);

    printf("\n=== Reading the table ===\n");
    printf("xp/§20  : §20-xpacked / §20. < 1.0 means X-packing pays off.\n");
    printf("xp/dot  : §20-xpacked / unpacked. < 1.0 means full packing pays off.\n");
    printf("§20/dot : §20 (W-packed only) / unpacked. < 1.0 means W-packing pays off.\n\n");
    printf("Crossover M: lowest M where xp/§20 < 1.0. That's the consumer\n"
           "regime where TD-7's X-packed kernel begins to beat the unpacked-X\n"
           "§20. Below that M, ship §20 (unpacked X). Above, ship §20-xpacked.\n");

    return 0;
}
