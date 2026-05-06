/*
 * tristate_dram_regime.c — TD-9 DRAM-bound regime test.
 *
 * Per docs/TECHNICAL_DEBT.md TD-9 + journal/tristate_strong_membw_addendum.md.
 *
 * Question: does sub-2-bit base-3's density advantage manifest as wall-clock
 * crossover at TRUE DRAM-bound regimes (W substantially exceeds L2)? The
 * existing strong-claim membw addendum tested up to W = 25.6 MB (just past
 * L2 = 12-16 MB on M-series). It found a PLATEAU not a crossover within
 * that range. TD-9 extends the sweep into N=2048+ / K=large territory.
 *
 * Apple Silicon cache hierarchy (M-series, approximate):
 *   L1 data cache      192 KB per P-core
 *   L2 cache (shared)  12-16 MB
 *   No discrete L3; system-level cache (SLC) ≈ 8-32 MB depending on chip.
 *   DRAM bandwidth     ~70-200 GB/s unified.
 *
 * For W > L2 substantially (≥ 50 MB), data must come from SLC or DRAM each
 * call. Sub-2-bit packing's 5×-vs-unpacked storage advantage should help
 * proportionally MORE in this regime — IF the bandwidth difference is the
 * bottleneck.
 *
 * Configs span W ∈ {0.4 MB, 1 MB, 6.4 MB, 25.6 MB, 51.2 MB, 102.4 MB,
 * 204.8 MB} via N=2048..16384, K=12800..51200. Compares:
 *   Path A (4-in-8 packed)   — 2.0 bits/cell W, dense decode + SDOT
 *   Path D (5-in-8 packed)   — 1.6 bits/cell W, split-LUT decode + SDOT
 *
 * Pre-committed gate: if Path D's wall-clock ratio (D/A) drops below
 * 1.0 at W > 50 MB, sub-2-bit base-3 advantage manifests at DRAM-bound.
 * If ratio stays > 1.0 throughout, the audit's plateau finding extends.
 *
 * Cache-flush + warmup discipline is mirrored from tristate_strong_bench
 * (R-G1 cold-cache buffer; per-rep flush; small N_REPS to bound runtime).
 */

#include "b2b_matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* ── RNG ────────────────────────────────────────────────────────────────── */
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

/* ── Time ──────────────────────────────────────────────────────────────── */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ── Cache-flush ───────────────────────────────────────────────────────── */
static void flush_caches(uint8_t* buf, size_t n) {
    /* Walk a buffer larger than L2; touch each cacheline. */
    volatile uint32_t sink = 0;
    for (size_t i = 0; i < n; i += 64) sink ^= buf[i];
    (void)sink;
}

/* ── Configs ────────────────────────────────────────────────────────────── */
typedef struct {
    int M, K, N;
    double bytes_W_path_A;   /* bytes for Path A (2 b/c) W storage */
    double bytes_W_path_D;   /* bytes for Path D (1.6 b/c) W storage */
    int reps;
    const char* note;
} Config;

#define MB_2BC(n,k) ((double)((n) * (((k) + 3) / 4)) / (1024.0 * 1024.0))
#define MB_5in8(n,k) ((double)((n) * (((k) + 4) / 5)) / (1024.0 * 1024.0))

static Config make_cfg(int M, int K, int N, int reps, const char* note) {
    Config c = { M, K, N, MB_2BC(N, K), MB_5in8(N, K), reps, note };
    return c;
}

int main(void) {
    /* 64 MB cache flush buffer — exceeds M-series L2. */
    const size_t FLUSH_SIZE = 64 * 1024 * 1024;
    uint8_t* flush_buf = (uint8_t*)calloc(FLUSH_SIZE, 1);
    if (!flush_buf) { fprintf(stderr, "OOM flush buf\n"); return 1; }

    /* RC-5/RC-8 remediation: realistic-K configs (K ≤ 12800 per typical ML
     * shapes); reps increased at deep-DRAM. K=51200 sanity-check kept but
     * marked clearly as a synthetic shape. */
    Config cfgs[] = {
        /* Reference: L1/L2-resident at realistic K. */
        make_cfg(8,   1280,    64,  200, "L1-resident                 (realistic K)"),
        make_cfg(8,  12800,    64,  100, "L2-resident                 (realistic K)"),
        /* L2-overflow at realistic K. */
        make_cfg(8,  12800,  1024,   40, "W ≈  3.2 MB  near L2        (realistic K)"),
        make_cfg(8,  12800,  4096,   20, "W ≈ 12.8 MB  at L2          (realistic K)"),
        make_cfg(8,  12800,  8192,   10, "W ≈ 25.6 MB  past L2        (realistic K)"),
        /* DRAM-bound at realistic K — RC-8: increased reps from 2-3 to 5-10. */
        make_cfg(8,  12800, 16384,   10, "W ≈ 51.2 MB  DRAM-bound     (realistic K)"),
        make_cfg(8,  12800, 32768,    5, "W ≈102.4 MB  deep DRAM      (realistic K)"),
        /* Synthetic-K sanity check (RC-5: K=25600 is unusual; K=51200 even
         * more so. Kept for trajectory observation but NOT load-bearing
         * configs for the verdict.) */
        make_cfg(8,  25600,  8192,    5, "W ≈ 51.2 MB  alt shape      (sanity, K=25600)"),
        make_cfg(8,  25600, 16384,    5, "W ≈102.4 MB  deep DRAM      (sanity, K=25600)"),
        make_cfg(8,  51200, 16384,    3, "W ≈204.8 MB  far past DRAM  (sanity, K=51200)"),
    };
    int n_cfgs = (int)(sizeof(cfgs) / sizeof(cfgs[0]));

    printf("# TD-9: DRAM-bound regime test (v2 — RC-4/RC-5/RC-8 remediation)\n");
    printf("# Compares Path A (4-in-8, 2.0 b/c) vs Path D (5-in-8, 1.6 b/c)\n");
    printf("# Per-cell density savings: Path D = 0.8 × Path A storage.\n\n");
    printf("# RC-4 fix (tightened pre-committed gate):\n");
    printf("#   TRUE crossover requires D/A at deep-DRAM (W ≥ 50 MB) to be\n");
    printf("#   ≤ 0.8 × D/A at L1-resident (i.e., the ratio MUST IMPROVE\n");
    printf("#   monotonically with W). Bandwidth-driven advantage compounds\n");
    printf("#   with W, so a true bandwidth crossover should show this.\n");
    printf("#   v1's gate (D/A < 1.0 at any DRAM-bound config) was trivially\n");
    printf("#   met because D was already winning at L1.\n\n");
    printf("# RC-5 fix: realistic-K configs are the load-bearing measurement;\n");
    printf("#           K=25600 / K=51200 rows are sanity-check shapes only.\n");
    printf("# RC-8 fix: deep-DRAM reps increased from 2 to 5-10.\n\n");
    printf("%-50s W_A       W_D       reps   ms_A      ms_D      D/A\n",
        "config");

    for (int c = 0; c < n_cfgs; c++) {
        Config* cfg = &cfgs[c];
        int M = cfg->M, K = cfg->K, N = cfg->N;
        if (K % 80 != 0 || N % 4 != 0) {
            printf("[skip] cfg requires K%%80==0 N%%4==0 (K=%d N=%d)\n", K, N);
            continue;
        }
        int Kp4 = (K + 3) / 4;
        int Kp5 = (K + 4) / 5;

        /* Allocate. */
        int8_t*  X       = (int8_t*)calloc((size_t)M * K, sizeof(int8_t));
        int8_t*  W_unp   = (int8_t*)calloc((size_t)N * K, sizeof(int8_t));
        uint8_t* W_a     = (uint8_t*)calloc((size_t)N * Kp4, 1);
        uint8_t* W_d     = (uint8_t*)calloc((size_t)N * Kp5, 1);
        int32_t* Y_a     = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
        int32_t* Y_d     = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));

        if (!X || !W_unp || !W_a || !W_d || !Y_a || !Y_d) {
            fprintf(stderr, "OOM cfg %d\n", c);
            free(X); free(W_unp); free(W_a); free(W_d); free(Y_a); free(Y_d);
            continue;
        }

        rng_t rng; rng_init(&rng, (uint32_t)(c + 1) * 0x9E3779B1u);
        for (int i = 0; i < M*K; i++)
            X[i] = rng_lt(&rng, 0.4) ? 0 : (int8_t)rng_sign(&rng);
        for (int i = 0; i < N*K; i++)
            W_unp[i] = rng_lt(&rng, 0.4) ? 0 : (int8_t)rng_sign(&rng);
        for (int j = 0; j < N; j++) {
            base3_pack     (W_a + (size_t)j * Kp4, W_unp + (size_t)j * K, K);
            base3_5in8_pack(W_d + (size_t)j * Kp5, W_unp + (size_t)j * K, K);
        }

        /* Verify equivalence on first iteration. */
        base3_packed_matmul_neon (Y_a, X, W_a, M, K, N);
        base3_5in8_matmul_neon   (Y_d, X, W_d, M, K, N);
        if (memcmp(Y_a, Y_d, (size_t)M * N * sizeof(int32_t)) != 0) {
            fprintf(stderr, "[ERROR] cfg %d: Path A vs Path D mismatch\n", c);
            free(X); free(W_unp); free(W_a); free(W_d); free(Y_a); free(Y_d);
            continue;
        }

        /* Warm one rep. */
        flush_caches(flush_buf, FLUSH_SIZE);
        base3_packed_matmul_neon(Y_a, X, W_a, M, K, N);
        flush_caches(flush_buf, FLUSH_SIZE);
        base3_5in8_matmul_neon  (Y_d, X, W_d, M, K, N);

        /* Time. Per-rep cache flush. */
        double sum_ms_a = 0, sum_ms_d = 0;
        for (int r = 0; r < cfg->reps; r++) {
            flush_caches(flush_buf, FLUSH_SIZE);
            double t0 = now_ms();
            base3_packed_matmul_neon(Y_a, X, W_a, M, K, N);
            sum_ms_a += now_ms() - t0;

            flush_caches(flush_buf, FLUSH_SIZE);
            t0 = now_ms();
            base3_5in8_matmul_neon  (Y_d, X, W_d, M, K, N);
            sum_ms_d += now_ms() - t0;
        }
        double mean_a = sum_ms_a / cfg->reps;
        double mean_d = sum_ms_d / cfg->reps;
        double ratio  = (mean_a > 0) ? mean_d / mean_a : 0.0;

        printf("%-50s %6.2f MB %6.2f MB  %3d   %8.3f  %8.3f  %.3f\n",
               cfg->note, cfg->bytes_W_path_A, cfg->bytes_W_path_D,
               cfg->reps, mean_a, mean_d, ratio);

        free(X); free(W_unp); free(W_a); free(W_d); free(Y_a); free(Y_d);
    }

    free(flush_buf);

    /* Pre-committed gate evaluation. We need the ratio at L1 and at the
     * deepest realistic-K DRAM config. Hardcoded indices: cfg 0 = L1
     * realistic, cfg 6 = "W ≈102.4 MB deep DRAM (realistic K)". */
    /* (Don't try to recover from arrays we already freed; just print the
     * gate semantics; user reads ratios above.) */
    printf("\n=== Pre-committed gate (RC-4 tightened) ===\n");
    printf("TRUE bandwidth-driven crossover requires:\n"
           "  D/A at deep-DRAM realistic-K (cfg \"W≈102.4 MB realistic K\")\n"
           "  ≤ 0.8 × D/A at L1-resident realistic-K (cfg \"L1-resident\").\n");
    printf("If gate FAILS (ratio at deep-DRAM ≥ 0.8 × ratio at L1): there's\n"
           "no bandwidth-driven crossover — Path D's advantage is constant,\n"
           "not compounding with W. Membw addendum's PLATEAU finding extends.\n");
    printf("If gate PASSES: bandwidth bottleneck contributes additively to\n"
           "Path D's advantage; the advantage compounds with W.\n\n");

    printf("=== Verdict template ===\n");
    printf("Read the D/A column across configs:\n");
    printf("  - REALISTIC-K rows are the load-bearing measurement.\n");
    printf("  - K=25600 / K=51200 rows are sanity-check shapes (ML workloads\n"
           "    don't have these K values; included only to confirm trajectory).\n");
    printf("  - If realistic-K D/A is roughly constant across W: PLATEAU.\n");
    printf("  - If realistic-K D/A drops monotonically with W: TRUE crossover.\n");
    printf("  - If realistic-K D/A stays flat or rises slightly: SDOT-amortization\n"
           "    advantage dominates; bandwidth not the bottleneck.\n");

    return 0;
}
