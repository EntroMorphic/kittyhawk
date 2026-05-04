/*
 * bench_m4t_tier2_perf.c — fair-comparison perf harness for Tier 2 work.
 *
 * Per journal/tier2_remediation_precommit.md (R-G1, R-G2, R-G4, R-G5).
 *
 * Both versions of each candidate are called via the lib boundary
 * (`m4t_route_select` vs `m4t_route_select_scalar_ref`;
 *  `m4t_route_confidence_weighted_dist` vs the `_branchless` variant).
 * This eliminates the inline-vs-lib-call asymmetry that contaminated
 * the original Tier 2 measurement (see journal/tier2_perf_redteam.md C1).
 *
 * Discipline:
 *   - clock_gettime(CLOCK_MONOTONIC) for ns resolution (M1).
 *   - Three data distributions (M2): random, structured, sparse-zero.
 *   - Pool of 8 distinct data arrays, pseudo-randomly indexed per iter (M3).
 *   - Median of 5 trials per measurement (Risk B mitigation).
 *
 * READING THE NUMBERS: this bench measures carry-dependent, single-pass
 * workloads — the shape matching the substrate's actual consumers. A
 * different workload shape (e.g., pipelined / batched independent ops)
 * surfaces a different bottleneck profile and may show very different
 * timings for the same kernel. See "Reading perf measurements" in
 * m4t/README.md, or m4t/tests/bench_m4t_lto.c for a controlled
 * comparison across two shapes.
 */

#include "m4t_route.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define POOL_SIZE 8
#define N_TRIALS 5

/* Cache-trash buffer: 32 MB > L3 cache on most Apple Silicon. Walking it
 * cache-line-by-cache-line evicts our working set. Used by RES-1 to defeat
 * steady-state cache effects between trials. */
#define CACHE_TRASH_SIZE (32 * 1024 * 1024)
static volatile uint8_t* g_cache_trash = NULL;
static volatile uint64_t g_cache_trash_acc = 0;

static void cache_trash_init(void) {
    g_cache_trash = malloc(CACHE_TRASH_SIZE);
    if (g_cache_trash) memset((void*)g_cache_trash, 0xCC, CACHE_TRASH_SIZE);
}
static void cache_trash_free(void) {
    free((void*)g_cache_trash); g_cache_trash = NULL;
}
static void cache_trash(void) {
    uint64_t acc = 0;
    for (size_t i = 0; i < CACHE_TRASH_SIZE; i += 64) {
        acc += g_cache_trash[i];
    }
    g_cache_trash_acc ^= acc;  /* defeat dead-store elimination */
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int dcmp(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da < db) ? -1 : (da > db) ? 1 : 0;
}
static double median5(double* xs) {
    qsort(xs, N_TRIALS, sizeof(double), dcmp);
    return xs[N_TRIALS / 2];
}

static uint32_t xs32(uint32_t* s) {
    uint32_t x = *s; if (x == 0) x = 0x12345678u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x; return x;
}

typedef enum { DIST_RANDOM, DIST_STRUCTURED, DIST_SPARSE_ZERO } dist_kind_t;
static const char* DIST_NAMES[3] = { "random", "structured", "sparse-zero" };

/* ── Adversarial distributions (subagent-designed; RES-2) ───────────────── */

/* Dist A1: LFSR-cycled trits with rejection sampling. 16-bit Galois LFSR
 * polynomial 0xB400, seeded with 0xACE1. Maximally entropic for branch
 * predictors — no exploitable autocorrelation. Predicts scalar loses big. */
static uint16_t adv_lfsr_state = 0xACE1u;
static void adv_lfsr_reset(uint16_t seed) { adv_lfsr_state = seed ? seed : 0xACE1u; }
static uint16_t adv_lfsr_step(void) {
    uint16_t s = adv_lfsr_state;
    int lsb = s & 1;
    s >>= 1;
    if (lsb) s ^= 0xB400u;
    adv_lfsr_state = s;
    return s;
}
static void adv_fill_lfsr_trits(uint8_t* dst, int n_bytes, uint16_t seed) {
    adv_lfsr_reset(seed);
    for (int i = 0; i < n_bytes; i++) {
        uint8_t b = 0;
        for (int f = 0; f < 4; f++) {
            uint8_t code;
            do {
                code = (uint8_t)(adv_lfsr_step() & 0x3u);
            } while (code == 0x3u);  /* reject reserved 0b11 */
            b |= (uint8_t)(code << (f * 2));
        }
        dst[i] = b;
    }
}

/* Dist A2: sparse-zero bursts. 95% +1, 5% zero (in Poisson-spaced bursts).
 * Almost-always-taken branch is trivially predicted. Predicts NEON might
 * lose because branchless does work that the predictable branchy skips. */
static void adv_fill_sparse_zero_bursts(uint8_t* dst, int n_bytes, uint32_t seed) {
    uint32_t state = seed;
    /* Default fill: all +1 trits. */
    for (int i = 0; i < n_bytes; i++) dst[i] = 0x55u;  /* 0b01010101 = four +1 codes */
    /* Sprinkle bursts of zeros. Expected burst spacing 200 cells. */
    int n_cells_total = n_bytes * 4;
    for (int pos = 0; pos < n_cells_total; ) {
        int gap = (int)(50 + (xs32(&state) % 300u));  /* spacing */
        pos += gap;
        if (pos >= n_cells_total) break;
        int burst_len = 3 + (int)(xs32(&state) % 5u);  /* 3-7 zeros */
        for (int k = 0; k < burst_len && pos < n_cells_total; k++, pos++) {
            int byte_idx = pos >> 2;
            int field    = pos & 3;
            uint8_t mask = (uint8_t)~(0x3u << (field * 2));
            dst[byte_idx] = (uint8_t)(dst[byte_idx] & mask);  /* clear field → 0 trit */
        }
    }
}

/* Dist B3: confidence-stripe thrasher. Period-64 conf stripes, period-96
 * mask stripes. Forces irregular per-byte work patterns. Subagent dist 5
 * (V2-G2). Without the cache-set aliasing; pattern-stress preserved. */
static void adv_fill_conf_stripe_thrasher(
    uint8_t* qt, uint8_t* tt, uint8_t* qc, uint8_t* tc, uint8_t* mask,
    int n_bytes, int n_conf_bytes, int sig_dim, uint32_t seed)
{
    /* Generate signatures with EXACTLY ~50% opposite-sign per 16-position window. */
    uint32_t state = seed;
    for (int byte_idx = 0; byte_idx < n_bytes; byte_idx++) {
        /* For each of 4 fields per byte, randomly assign (q, t) to one of:
         * (+1,+1), (-1,-1), (0,0), (+1,-1), (-1,+1) — 50% same-sign-or-zero,
         * 50% opposite. */
        uint8_t qb = 0, tb = 0;
        for (int f = 0; f < 4; f++) {
            uint32_t r = xs32(&state) % 4u;
            uint8_t qc2, tc2;
            if (r == 0) { qc2 = 0x01u; tc2 = 0x01u; }      /* both +1 */
            else if (r == 1) { qc2 = 0x02u; tc2 = 0x02u; } /* both -1 */
            else if (r == 2) { qc2 = 0x01u; tc2 = 0x02u; } /* opposite */
            else { qc2 = 0x02u; tc2 = 0x01u; }              /* opposite */
            qb |= (uint8_t)(qc2 << (f * 2));
            tb |= (uint8_t)(tc2 << (f * 2));
        }
        qt[byte_idx] = qb;
        tt[byte_idx] = tb;
    }
    /* Period-64 conf stripes: alternating runs of 64 ones / 64 zeros. */
    for (int b = 0; b < n_conf_bytes; b++) {
        int phase = (b / 8) % 2;  /* 8 bytes = 64 bits */
        qc[b] = phase ? 0xFFu : 0x00u;
        /* Tile conf: same shape but phase-shifted by 16 bits. */
        int tphase = ((b - 2 + n_conf_bytes) / 8) % 2;
        tc[b] = tphase ? 0xFFu : 0x00u;
    }
    /* Period-96 mask stripes (12 bytes of 0xFF, 12 bytes of 0). */
    for (int b = 0; b < n_bytes; b++) {
        int phase = (b / 12) % 2;
        mask[b] = phase ? 0xFFu : 0x00u;
    }
    int tail = sig_dim & 3;
    if (tail > 0 && (n_bytes - 1) >= 0) {
        /* Trim tail bits in last byte to valid trit fields. */
        uint8_t tail_mask = (uint8_t)((1u << (tail * 2)) - 1u);
        mask[n_bytes - 1] = (uint8_t)(mask[n_bytes - 1] & tail_mask);
    }
}

/* Dist B1: sparse-opposite needle. For conf-dist, query and tile mostly
 * agree (+1 everywhere) except a few opposite-sign positions sprinkled
 * with low-discrepancy spacing. Branchy's early-exit nails this case;
 * branchless does full work. Predicts branchy wins. */
static void adv_fill_sparse_opposite(
    uint8_t* qt, uint8_t* tt, uint8_t* qc, uint8_t* tc,
    int n_bytes, int n_conf_bytes, int n_opposite, uint32_t seed)
{
    /* All +1 trits in both query and tile. */
    for (int i = 0; i < n_bytes; i++) { qt[i] = 0x55u; tt[i] = 0x55u; }
    /* All conf bits set. */
    memset(qc, 0xFFu, (size_t)n_conf_bytes);
    memset(tc, 0xFFu, (size_t)n_conf_bytes);
    /* Place n_opposite mismatches: tile becomes -1 at those positions. */
    int n_cells = n_bytes * 4;
    if (n_opposite > n_cells) n_opposite = n_cells;
    uint32_t state = seed;
    /* Low-discrepancy spacing: stride = n_cells / n_opposite plus jitter. */
    int stride = (n_opposite > 0) ? (n_cells / n_opposite) : 1;
    if (stride < 1) stride = 1;
    int placed = 0;
    for (int p = stride / 2; p < n_cells && placed < n_opposite; p += stride) {
        int jitter = (int)(xs32(&state) % (uint32_t)(stride > 4 ? stride / 4 : 1));
        int pos = p + jitter;
        if (pos >= n_cells) break;
        int byte_idx = pos >> 2;
        int field    = pos & 3;
        /* Set tile field to 0b10 (-1). */
        uint8_t clear = (uint8_t)~(0x3u << (field * 2));
        uint8_t set   = (uint8_t)(0x2u << (field * 2));
        tt[byte_idx] = (uint8_t)((tt[byte_idx] & clear) | set);
        placed++;
    }
}

/* Dist A3: run-length trap. Phases of all-same, then period-3 alternation.
 * Subagent dist 2 (V2-G2). Without the cache-set aliasing (which requires
 * allocator-level engineering that's hard to verify); the branch-pattern
 * adversarial behavior is preserved. */
static void adv_fill_run_length_trap(uint8_t* dst, int n_bytes) {
    /* Phases of 16 cells (4 trit bytes) each. */
    int byte_idx = 0;
    int phase = 0;
    while (byte_idx < n_bytes) {
        uint8_t pat;
        switch (phase % 5) {
        case 0: pat = 0x55u; break;  /* all +1 */
        case 1: pat = 0x00u; break;  /* all 0  */
        case 2: pat = 0xAAu; break;  /* all -1 */
        case 3: pat = 0x69u; break;  /* alternating: +1,-1,+1,-1 */
        case 4: pat = 0x21u; break;  /* period-3 close-to-it: +1, 0, -1, +1 (mod 4) */
        default: pat = 0; break;
        }
        for (int k = 0; k < 4 && byte_idx < n_bytes; k++, byte_idx++) {
            dst[byte_idx] = pat;
        }
        phase++;
    }
}

/* Dist B2: triple-period resonance. Coprime periods 7 (opposite-sign
 * pattern), 11 (mask), 13 (query conf), 17 (tile conf). Defeats history-
 * length predictors. Predicts branchy loses 2-4x. */
static void adv_fill_triple_period(
    uint8_t* qt, uint8_t* tt, uint8_t* qc, uint8_t* tc, uint8_t* mask,
    int n_bytes, int n_conf_bytes, int sig_dim)
{
    /* Default fill: both signatures all +1. */
    for (int i = 0; i < n_bytes; i++) { qt[i] = 0x55u; tt[i] = 0x55u; }
    memset(qc, 0u, (size_t)n_conf_bytes);
    memset(tc, 0u, (size_t)n_conf_bytes);
    memset(mask, 0u, (size_t)n_bytes);
    /* Set tile to -1 at positions where (i mod 7) ∈ {0, 3, 5}. */
    for (int i = 0; i < sig_dim; i++) {
        int m7 = i % 7;
        if (m7 == 0 || m7 == 3 || m7 == 5) {
            int byte_idx = i >> 2;
            int field    = i & 3;
            uint8_t clear = (uint8_t)~(0x3u << (field * 2));
            uint8_t set   = (uint8_t)(0x2u << (field * 2));
            tt[byte_idx] = (uint8_t)((tt[byte_idx] & clear) | set);
        }
    }
    /* Mask: bit set every period 11 (set the corresponding trit field active). */
    for (int i = 0; i < sig_dim; i++) {
        if ((i % 11) == 0) {
            int byte_idx = i >> 2;
            int field    = i & 3;
            mask[byte_idx] = (uint8_t)(mask[byte_idx] | (uint8_t)(0x3u << (field * 2)));
        }
    }
    /* Q conf: bit set every period 13. T conf: every period 17. */
    for (int i = 0; i < sig_dim; i++) {
        if ((i % 13) == 0) qc[i >> 3] = (uint8_t)(qc[i >> 3] | (uint8_t)(1u << (i & 7)));
        if ((i % 17) == 0) tc[i >> 3] = (uint8_t)(tc[i >> 3] | (uint8_t)(1u << (i & 7)));
    }
}

static void fill_packed_trits(uint8_t* dst, int n_bytes, dist_kind_t d, uint32_t seed) {
    uint32_t state = seed;
    for (int i = 0; i < n_bytes; i++) {
        uint8_t b = 0;
        for (int f = 0; f < 4; f++) {
            uint8_t code;
            switch (d) {
            case DIST_RANDOM: {
                uint32_t r = xs32(&state) % 3u;
                code = (r == 0) ? 0x00u : (r == 1) ? 0x01u : 0x02u;
                break;
            }
            case DIST_STRUCTURED: {
                int r = ((i * 4 + f) % 3);
                code = (r == 0) ? 0x01u : (r == 1) ? 0x02u : 0x00u;
                break;
            }
            case DIST_SPARSE_ZERO: {
                uint32_t r = xs32(&state) % 10u;
                code = (r >= 9) ? 0x01u : (r == 8) ? 0x02u : 0x00u;
                break;
            }
            default: code = 0; break;
            }
            b |= (uint8_t)(code << (f * 2));
        }
        dst[i] = b;
    }
}

/* ── select benchmark ──────────────────────────────────────────────────── */

typedef struct {
    uint8_t* c[POOL_SIZE];
    m4t_mtfp_t* a[POOL_SIZE];
    m4t_mtfp_t* b[POOL_SIZE];
    m4t_mtfp_t* d[POOL_SIZE];
    m4t_mtfp_t* out;
    int n_cells;
} select_pool_t;

static void select_pool_init(select_pool_t* p, int n_cells, dist_kind_t dist, uint32_t seed) {
    int Dp = M4T_TRIT_PACKED_BYTES(n_cells);
    p->n_cells = n_cells;
    p->out = malloc((size_t)n_cells * sizeof(m4t_mtfp_t));
    for (int i = 0; i < POOL_SIZE; i++) {
        p->c[i] = malloc((size_t)Dp);
        p->a[i] = malloc((size_t)n_cells * sizeof(m4t_mtfp_t));
        p->b[i] = malloc((size_t)n_cells * sizeof(m4t_mtfp_t));
        p->d[i] = malloc((size_t)n_cells * sizeof(m4t_mtfp_t));
        fill_packed_trits(p->c[i], Dp, dist, seed + (uint32_t)i * 17u);
        uint32_t st = seed + (uint32_t)i * 23u;
        for (int j = 0; j < n_cells; j++) {
            p->a[i][j] = (m4t_mtfp_t)((int)(xs32(&st) % 2001u) - 1000);
            p->b[i][j] = (m4t_mtfp_t)((int)(xs32(&st) % 2001u) - 1000);
            p->d[i][j] = (m4t_mtfp_t)((int)(xs32(&st) % 2001u) - 1000);
        }
    }
}
static void select_pool_free(select_pool_t* p) {
    free(p->out);
    for (int i = 0; i < POOL_SIZE; i++) {
        free(p->c[i]); free(p->a[i]); free(p->b[i]); free(p->d[i]);
    }
}

static double measure_select(select_pool_t* p, int n_iter, int use_neon) {
    double t0 = now_seconds();
    int dummy = 0;
    uint32_t idx_state = 0x5eed5eedu;
    for (int it = 0; it < n_iter; it++) {
        int k = (int)(xs32(&idx_state) & (POOL_SIZE - 1));
        if (use_neon) m4t_route_select(p->out, p->c[k], p->a[k], p->b[k], p->d[k], p->n_cells);
        else          m4t_route_select_scalar_ref(p->out, p->c[k], p->a[k], p->b[k], p->d[k], p->n_cells);
        dummy ^= p->out[0];
    }
    double t1 = now_seconds();
    (void)dummy;
    return t1 - t0;
}

/* ── confidence-weighted distance benchmark ─────────────────────────────── */

typedef struct {
    uint8_t* qt[POOL_SIZE];
    uint8_t* tt[POOL_SIZE];
    uint8_t* qc[POOL_SIZE];
    uint8_t* tc[POOL_SIZE];
    uint8_t* mask;
    int sig_dim;
} dist_pool_t;

static void dist_pool_init(dist_pool_t* p, int sig_dim, dist_kind_t dist, uint32_t seed) {
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    int Cp = (sig_dim + 7) / 8;
    p->sig_dim = sig_dim;
    p->mask = malloc((size_t)Dp);
    memset(p->mask, 0xFF, (size_t)Dp);
    int tail = sig_dim & 3;
    if (tail > 0) p->mask[Dp - 1] = (uint8_t)((1u << (tail * 2)) - 1u);
    for (int i = 0; i < POOL_SIZE; i++) {
        p->qt[i] = malloc((size_t)Dp);
        p->tt[i] = malloc((size_t)Dp);
        p->qc[i] = malloc((size_t)Cp);
        p->tc[i] = malloc((size_t)Cp);
        fill_packed_trits(p->qt[i], Dp, dist, seed + (uint32_t)i * 31u);
        fill_packed_trits(p->tt[i], Dp, dist, seed + (uint32_t)i * 31u + 11u);
        uint32_t st = seed + (uint32_t)i * 41u;
        for (int j = 0; j < Cp; j++) {
            p->qc[i][j] = (uint8_t)(xs32(&st) & 0xFFu);
            p->tc[i][j] = (uint8_t)(xs32(&st) & 0xFFu);
        }
    }
}
static void dist_pool_free(dist_pool_t* p) {
    free(p->mask);
    for (int i = 0; i < POOL_SIZE; i++) {
        free(p->qt[i]); free(p->tt[i]); free(p->qc[i]); free(p->tc[i]);
    }
}

static double measure_dist(dist_pool_t* p, int n_iter, int use_branchless) {
    double t0 = now_seconds();
    int32_t dummy = 0;
    uint32_t idx_state = 0xdeadbeefu;
    for (int it = 0; it < n_iter; it++) {
        int k = (int)(xs32(&idx_state) & (POOL_SIZE - 1));
        if (use_branchless) {
            dummy ^= m4t_route_confidence_weighted_dist_branchless(
                p->qt[k], p->qc[k], p->tt[k], p->tc[k], p->mask, p->sig_dim);
        } else {
            dummy ^= m4t_route_confidence_weighted_dist(
                p->qt[k], p->qc[k], p->tt[k], p->tc[k], p->mask, p->sig_dim);
        }
    }
    double t1 = now_seconds();
    (void)dummy;
    return t1 - t0;
}

/* ── Cache-defeat verification (RES-1, real cache trashing) ─────────────── */

/* Measure select with a forced cache trash before timing starts. The
 * difference between this and steady-state (no trash) shows how much the
 * steady-state numbers were optimistically benefiting from hot cache. */
static double measure_select_cold(select_pool_t* p, int n_iter, int use_neon) {
    cache_trash();  /* evict working set BEFORE starting the timer */
    return measure_select(p, n_iter, use_neon);
}

static void verify_cache_defeat_real(select_pool_t* p) {
    /* Steady-state (warm cache, repeated). */
    double warm[N_TRIALS];
    for (int t = 0; t < N_TRIALS; t++) warm[t] = measure_select(p, 1000, 1);
    double m_warm = median5(warm);

    /* Cold-cache (trash before each trial). */
    double cold[N_TRIALS];
    for (int t = 0; t < N_TRIALS; t++) cold[t] = measure_select_cold(p, 1000, 1);
    double m_cold = median5(cold);

    double per_iter_warm = m_warm * 1e9 / 1000.0;
    double per_iter_cold = m_cold * 1e9 / 1000.0;
    double ratio = (per_iter_warm > 0) ? per_iter_cold / per_iter_warm : 1.0;
    printf("  RES-1 cache-defeat (REAL trash): warm=%.1fns/iter cold=%.1fns/iter ratio=%.2fx\n",
           per_iter_warm, per_iter_cold, ratio);
    /* Gate per RES-1: PASS iff cold ≥ 1.3× warm (cache effects are real)
     * OR cold within 30% of warm (working set fits in L1 anyway, defensible). */
    if (ratio >= 1.3) {
        printf("  RES-1 verdict: cold-cache is %.1fx slower; cache effects ARE real; "
               "warm steady-state is optimistic.\n", ratio);
    } else if (ratio >= 0.85) {
        printf("  RES-1 verdict: cold/warm within 30%% — working set likely fits in L1 "
               "even after eviction; steady-state numbers are honest for this workload.\n");
    } else {
        printf("  RES-1 verdict: cold is FASTER than warm — measurement anomaly, "
               "investigate.\n");
    }
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(void) {
    printf("# Tier 2 Residuals Closure (LTO + real cache-trash + adversarial)\n");
    printf("# Per journal/tier2_residuals_precommit.md\n\n");

    cache_trash_init();

    int n_cells = 64;
    int sig_dim = 16;
    int n_iter = 100000;

    int g1_pass_count = 0;
    int select_dir_pos = 0, select_dir_neg = 0;
    int dist_dir_pos = 0, dist_dir_neg = 0;

    for (int d = 0; d < 3; d++) {
        printf("\n=== Distribution: %s ===\n", DIST_NAMES[d]);

        select_pool_t sp;
        select_pool_init(&sp, n_cells, (dist_kind_t)d, 0xa1a2a3a4u);

        double trials_scalar[N_TRIALS], trials_neon[N_TRIALS];
        for (int t = 0; t < N_TRIALS; t++) trials_scalar[t] = measure_select(&sp, n_iter, 0);
        for (int t = 0; t < N_TRIALS; t++) trials_neon[t]   = measure_select(&sp, n_iter, 1);
        double t_scalar = median5(trials_scalar);
        double t_neon   = median5(trials_neon);
        double speedup  = (t_neon > 0) ? t_scalar / t_neon : 0;
        printf("  select (n_cells=%d, %d iter, median of %d trials):\n", n_cells, n_iter, N_TRIALS);
        printf("    scalar (lib)  : %.3fms\n", t_scalar * 1000.0);
        printf("    NEON   (lib)  : %.3fms\n", t_neon * 1000.0);
        printf("    speedup       : %.2fx  GATE >=1.5x  -> %s\n",
               speedup, speedup >= 1.5 ? "PASS" : (speedup >= 1.1 ? "WEAK" : "FAIL"));
        if (speedup >= 1.5) g1_pass_count++;
        if (speedup > 1.0) select_dir_pos++; else select_dir_neg++;

        if (d == 0) verify_cache_defeat_real(&sp);

        select_pool_free(&sp);

        dist_pool_t dp;
        dist_pool_init(&dp, sig_dim, (dist_kind_t)d, 0xb1b2b3b4u);

        double td_branchy[N_TRIALS], td_branchless[N_TRIALS];
        for (int t = 0; t < N_TRIALS; t++) td_branchy[t]    = measure_dist(&dp, n_iter, 0);
        for (int t = 0; t < N_TRIALS; t++) td_branchless[t] = measure_dist(&dp, n_iter, 1);
        double t_branchy   = median5(td_branchy);
        double t_branchless = median5(td_branchless);
        double dist_speedup = (t_branchless > 0) ? t_branchy / t_branchless : 0;
        printf("  conf-dist (sig_dim=%d, %d iter):\n", sig_dim, n_iter);
        printf("    branchy   (lib): %.3fms\n", t_branchy * 1000.0);
        printf("    branchless(lib): %.3fms\n", t_branchless * 1000.0);
        printf("    branchless / branchy speedup: %.2fx (R-G2 diagnostic)\n", dist_speedup);
        if (dist_speedup > 1.0) dist_dir_pos++; else dist_dir_neg++;

        dist_pool_free(&dp);
    }

    int g4_consistent = (select_dir_pos == 3 || select_dir_neg == 3) &&
                          (dist_dir_pos == 3 || dist_dir_neg == 3);

    /* ── RES-2: adversarial distributions (subagent-designed) ─────────── */
    printf("\n========================================\n");
    printf("RES-2: ADVERSARIAL DISTRIBUTIONS (subagent-designed; blind)\n");
    printf("========================================\n");

    /* A1: LFSR-cycled trits for select (predicts scalar loses 4-8x). */
    {
        printf("\n  Adversarial A1: LFSR-cycled trits (select, n_cells=%d)\n", n_cells);
        select_pool_t sp;
        select_pool_init(&sp, n_cells, DIST_RANDOM, 0);  /* a/b/d filled randomly */
        /* Override control vectors with LFSR-trits. */
        int Dp = M4T_TRIT_PACKED_BYTES(n_cells);
        for (int k = 0; k < POOL_SIZE; k++) {
            adv_fill_lfsr_trits(sp.c[k], Dp, (uint16_t)(0xACE1u + k * 0x100u));
        }
        double trials_s[N_TRIALS], trials_n[N_TRIALS];
        for (int t = 0; t < N_TRIALS; t++) trials_s[t] = measure_select(&sp, n_iter, 0);
        for (int t = 0; t < N_TRIALS; t++) trials_n[t] = measure_select(&sp, n_iter, 1);
        double t_s = median5(trials_s), t_n = median5(trials_n);
        double sp_ratio = (t_n > 0) ? t_s / t_n : 0;
        printf("    scalar: %.3fms  NEON: %.3fms  speedup: %.2fx  "
               "(subagent predicted scalar loses 4-8x)\n", t_s*1000, t_n*1000, sp_ratio);
        select_pool_free(&sp);
    }

    /* A2: sparse-zero bursts for select (predicts NEON loses 1.2-2x). */
    {
        printf("\n  Adversarial A2: sparse-zero bursts (select, n_cells=%d)\n", n_cells);
        select_pool_t sp;
        select_pool_init(&sp, n_cells, DIST_RANDOM, 0);
        int Dp = M4T_TRIT_PACKED_BYTES(n_cells);
        for (int k = 0; k < POOL_SIZE; k++) {
            adv_fill_sparse_zero_bursts(sp.c[k], Dp, 0xc0ffeeu + (uint32_t)k * 31u);
        }
        double trials_s[N_TRIALS], trials_n[N_TRIALS];
        for (int t = 0; t < N_TRIALS; t++) trials_s[t] = measure_select(&sp, n_iter, 0);
        for (int t = 0; t < N_TRIALS; t++) trials_n[t] = measure_select(&sp, n_iter, 1);
        double t_s = median5(trials_s), t_n = median5(trials_n);
        double sp_ratio = (t_n > 0) ? t_s / t_n : 0;
        printf("    scalar: %.3fms  NEON: %.3fms  speedup: %.2fx  "
               "(subagent predicted NEON loses 1.2-2x)\n", t_s*1000, t_n*1000, sp_ratio);
        select_pool_free(&sp);
    }

    /* B1: sparse-opposite needle for conf-dist at sig_dim=256.
     * Predicts branchy wins 3-5x. */
    {
        int adv_sig_dim = 256;
        int adv_Dp = M4T_TRIT_PACKED_BYTES(adv_sig_dim);
        int adv_Cp = (adv_sig_dim + 7) / 8;
        printf("\n  Adversarial B1: sparse-opposite needle (conf-dist, sig_dim=%d)\n", adv_sig_dim);
        dist_pool_t dp;
        dist_pool_init(&dp, adv_sig_dim, DIST_RANDOM, 0);
        for (int k = 0; k < POOL_SIZE; k++) {
            adv_fill_sparse_opposite(dp.qt[k], dp.tt[k], dp.qc[k], dp.tc[k],
                                       adv_Dp, adv_Cp, /*n_opposite=*/4,
                                       0xb0bau + (uint32_t)k * 41u);
        }
        memset(dp.mask, 0xFF, (size_t)adv_Dp);
        int tail = adv_sig_dim & 3;
        if (tail > 0) dp.mask[adv_Dp - 1] = (uint8_t)((1u << (tail * 2)) - 1u);
        double trials_branchy[N_TRIALS], trials_branchless[N_TRIALS];
        for (int t = 0; t < N_TRIALS; t++) trials_branchy[t]    = measure_dist(&dp, n_iter, 0);
        for (int t = 0; t < N_TRIALS; t++) trials_branchless[t] = measure_dist(&dp, n_iter, 1);
        double t_b = median5(trials_branchy), t_bl = median5(trials_branchless);
        double sp_ratio = (t_bl > 0) ? t_b / t_bl : 0;
        printf("    branchy: %.3fms  branchless: %.3fms  speedup (branchless/branchy): %.2fx  "
               "(subagent predicted branchy wins 3-5x)\n", t_b*1000, t_bl*1000, sp_ratio);
        dist_pool_free(&dp);
    }

    /* B2: triple-period resonance for conf-dist at sig_dim=256.
     * Predicts branchy loses 2-4x. */
    {
        int adv_sig_dim = 256;
        int adv_Dp = M4T_TRIT_PACKED_BYTES(adv_sig_dim);
        int adv_Cp = (adv_sig_dim + 7) / 8;
        printf("\n  Adversarial B2: triple-period resonance (conf-dist, sig_dim=%d)\n", adv_sig_dim);
        dist_pool_t dp;
        dist_pool_init(&dp, adv_sig_dim, DIST_RANDOM, 0);
        for (int k = 0; k < POOL_SIZE; k++) {
            adv_fill_triple_period(dp.qt[k], dp.tt[k], dp.qc[k], dp.tc[k], dp.mask,
                                     adv_Dp, adv_Cp, adv_sig_dim);
        }
        double trials_branchy[N_TRIALS], trials_branchless[N_TRIALS];
        for (int t = 0; t < N_TRIALS; t++) trials_branchy[t]    = measure_dist(&dp, n_iter, 0);
        for (int t = 0; t < N_TRIALS; t++) trials_branchless[t] = measure_dist(&dp, n_iter, 1);
        double t_b = median5(trials_branchy), t_bl = median5(trials_branchless);
        double sp_ratio = (t_bl > 0) ? t_b / t_bl : 0;
        printf("    branchy: %.3fms  branchless: %.3fms  speedup (branchless/branchy): %.2fx  "
               "(subagent predicted branchy loses 2-4x)\n", t_b*1000, t_bl*1000, sp_ratio);
        dist_pool_free(&dp);
    }

    /* V2-G2: A3 run-length trap (subagent dist 2 — branch-pattern only,
     * cache-aliasing engineering skipped per closeout). */
    {
        printf("\n  V2-G2 A3: run-length trap (select, n_cells=%d)\n", n_cells);
        select_pool_t sp;
        select_pool_init(&sp, n_cells, DIST_RANDOM, 0);
        int Dp = M4T_TRIT_PACKED_BYTES(n_cells);
        for (int k = 0; k < POOL_SIZE; k++) adv_fill_run_length_trap(sp.c[k], Dp);
        double trials_s[N_TRIALS], trials_n[N_TRIALS];
        for (int t = 0; t < N_TRIALS; t++) trials_s[t] = measure_select(&sp, n_iter, 0);
        for (int t = 0; t < N_TRIALS; t++) trials_n[t] = measure_select(&sp, n_iter, 1);
        double t_s = median5(trials_s), t_n = median5(trials_n);
        double sp_ratio = (t_n > 0) ? t_s / t_n : 0;
        printf("    scalar: %.3fms  NEON: %.3fms  speedup: %.2fx\n",
               t_s*1000, t_n*1000, sp_ratio);
        select_pool_free(&sp);
    }

    /* V2-G2: B3 confidence-stripe thrasher (subagent dist 5 — pattern only). */
    {
        int adv_sig_dim = 4096;  /* larger sig_dim for non-trivial work under LTO */
        int adv_Dp = M4T_TRIT_PACKED_BYTES(adv_sig_dim);
        int adv_Cp = (adv_sig_dim + 7) / 8;
        printf("\n  V2-G2 B3: confidence-stripe thrasher (conf-dist, sig_dim=%d)\n", adv_sig_dim);
        dist_pool_t dp;
        dist_pool_init(&dp, adv_sig_dim, DIST_RANDOM, 0);
        for (int k = 0; k < POOL_SIZE; k++) {
            adv_fill_conf_stripe_thrasher(dp.qt[k], dp.tt[k], dp.qc[k], dp.tc[k], dp.mask,
                                            adv_Dp, adv_Cp, adv_sig_dim,
                                            0xc0deu + (uint32_t)k * 53u);
        }
        double trials_b[N_TRIALS], trials_bl[N_TRIALS];
        for (int t = 0; t < N_TRIALS; t++) trials_b[t]  = measure_dist(&dp, n_iter, 0);
        for (int t = 0; t < N_TRIALS; t++) trials_bl[t] = measure_dist(&dp, n_iter, 1);
        double t_b = median5(trials_b), t_bl = median5(trials_bl);
        double sp_ratio = (t_bl > 0) ? t_b / t_bl : 0;
        printf("    branchy: %.3fms  branchless: %.3fms  ratio: %.2fx\n",
               t_b*1000, t_bl*1000, sp_ratio);
        dist_pool_free(&dp);
    }

    /* V2-G3: cache-defeat saturation across L1/L2/L3 working set sizes. */
    printf("\n========================================\n");
    printf("V2-G3: CACHE-DEFEAT SATURATION (multi-scale)\n");
    printf("========================================\n");
    static const int sweep_n_cells[] = { 64, 4096, 65536, 524288 };
    static const char* sweep_labels[] = {
        "n=64 (~1KB, L1)",
        "n=4096 (~64KB, L1/L2 boundary)",
        "n=65536 (~1MB, L2)",
        "n=524288 (~8MB, L2/L3 boundary)"
    };
    int n_sweep = sizeof(sweep_n_cells) / sizeof(sweep_n_cells[0]);
    int g3_l1_low = 0, g3_mid_high = 0, g3_large_stable = 0;
    for (int s = 0; s < n_sweep; s++) {
        int nc = sweep_n_cells[s];
        printf("\n  %s\n", sweep_labels[s]);
        select_pool_t sp;
        select_pool_init(&sp, nc, DIST_RANDOM, 0xa1a2a3a4u);
        /* For larger sizes, fewer iterations to keep total time reasonable. */
        int sweep_iter = (nc <= 4096) ? 10000 : (nc <= 65536) ? 1000 : 100;
        double warm[N_TRIALS], cold[N_TRIALS];
        for (int t = 0; t < N_TRIALS; t++) warm[t] = measure_select(&sp, sweep_iter, 1);
        for (int t = 0; t < N_TRIALS; t++) cold[t] = measure_select_cold(&sp, sweep_iter, 1);
        double m_warm = median5(warm), m_cold = median5(cold);
        double per_warm = m_warm * 1e9 / sweep_iter;
        double per_cold = m_cold * 1e9 / sweep_iter;
        double ratio = (per_warm > 0) ? per_cold / per_warm : 1.0;
        printf("    warm=%.0fns/iter cold=%.0fns/iter ratio=%.2fx\n",
               per_warm, per_cold, ratio);
        if (s == 0 && ratio < 1.3) g3_l1_low = 1;
        if ((s == 1 || s == 2) && ratio >= 1.3) g3_mid_high = 1;
        if (s == n_sweep - 1) g3_large_stable = 1;  /* report-only at largest */
        select_pool_free(&sp);
    }
    printf("\n  V2-G3 verdict: L1-resident (n=64) ratio < 1.3 = %s; "
           "mid (n=4096 or n=65536) ratio >= 1.3 = %s; "
           "large measured = %s\n",
           g3_l1_low ? "OK" : "MISS",
           g3_mid_high ? "OK" : "MISS",
           g3_large_stable ? "OK" : "MISS");

    printf("\n=================================================\n");
    printf("R-G1 select speedup     : %d/3 distributions PASS  -> %s\n",
           g1_pass_count, g1_pass_count == 3 ? "PASS" : (g1_pass_count >= 2 ? "WEAK" : "FAIL"));
    printf("R-G2 conf-dist          : diagnostic-only (no PASS/FAIL)\n");
    printf("R-G4 direction-of-effect: %s across distributions\n",
           g4_consistent ? "CONSISTENT (PASS)" : "MIXED (WEAK)");
    printf("RES-2 adversarial       : 4 distributions tested; results above.\n");
    printf("=================================================\n");
    cache_trash_free();
    return 0;
}
