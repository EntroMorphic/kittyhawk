/*
 * audit/no_scalar_audit_bench.c — quantify setup-time wins from the
 * no-scalar audit remediation (2026-05-06).
 *
 * Compares the NEON public dispatcher vs the _scalar_ref test oracle
 * for each function NEON-ized in this audit:
 *
 *   m4t_pack_trits_1d         (4-in-8 pack)
 *   m4t_unpack_trits_1d       (4-in-8 unpack)
 *   m4t_pack_trits_5in8_1d    (5-in-8 pack)
 *   m4t_unpack_trits_5in8_1d  (5-in-8 unpack)
 *   m4t_mtfp19_to_mtfp4       (cell-width narrow)
 *   m4t_mtfp4_to_mtfp19       (cell-width widen)
 *   m4t_mtfp_shift3 k>0       (multiply-by-3^k)
 *
 * Cache-flush + warm-rep discipline mirrored from tristate_strong_bench.
 * Bit-exactness verified per config (memcmp NEON vs scalar_ref) before
 * timing — guards against compiler/optimization regressions across the
 * test surface.
 *
 * Pre-committed expectation: NEON should be ≥3× scalar_ref at L1-
 * resident N for SIMD-friendly ops; ≥2× at DRAM-bound N (memory
 * bandwidth caps the asymptote regardless of compute ratio). Failures
 * to clear these would warrant investigation.
 */

#include "m4t_trit_pack.h"
#include "m4t_mtfp.h"
#include "m4t_mtfp4.h"
#include "m4t_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* ── Helpers ─────────────────────────────────────────────────────────── */
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

typedef struct { uint32_t s; } rng_t;
static void rng_init(rng_t* r, uint32_t seed) { r->s = seed ? seed : 0xdeadbeefu; }
static uint32_t rng_u32(rng_t* r) {
    uint32_t x = r->s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    r->s = x;
    return x;
}

/* ── Bench harness ───────────────────────────────────────────────────── */

typedef struct {
    int n;
    int reps;
    const char* note;
} Cfg;

/* Reps tuned so each row takes O(100ms) — flush overhead (64 MB walk per
 * rep) dominates at tiny N, so reps don't scale linearly with N. */
static const Cfg CFGS[] = {
    {       64,    1000, "tiny  (per-call overhead)" },
    {     1024,    1000, "L1-resident small        " },
    {    16384,     500, "L1-resident large        " },
    {   262144,     100, "L2-resident              " },
    {  2097152,      20, "DRAM-bound               " },
    /* Tail-exercise rows: deliberately non-aligned. */
    {     1025,    1000, "tail (4-in-8 n%16=1)     " },
    {     1041,    1000, "tail (5-in-8 n%80=1)     " },
};
#define N_CFGS ((int)(sizeof(CFGS) / sizeof(CFGS[0])))
#define MAX_N 2097152

static const size_t FLUSH_SIZE = 64 * 1024 * 1024;

/* ── Per-function bench bodies ───────────────────────────────────────── */

#define BENCH_HEAD(name) \
    printf("\n=== %s ===\n", name); \
    printf("%-26s   N      reps      ns/elem_NEON   ns/elem_scalar   speedup\n", \
        "config")

#define BENCH_ROW(note, n, reps, neon_ns_per, scalar_ns_per) \
    printf("%-26s %7d  %7d   %10.3f    %10.3f      %5.2fx\n", \
        (note), (n), (reps), (neon_ns_per), (scalar_ns_per), \
        (neon_ns_per) > 0 ? (scalar_ns_per) / (neon_ns_per) : 0.0)

static void bench_pack_1d(uint8_t* flush_buf,
                          m4t_trit_t* src_buf, uint8_t* dst_neon, uint8_t* dst_scalar) {
    BENCH_HEAD("m4t_pack_trits_1d (4-in-8 pack)");
    for (int c = 0; c < N_CFGS; c++) {
        const Cfg* cfg = &CFGS[c];
        int n = cfg->n;
        rng_t r; rng_init(&r, (uint32_t)(c + 1) * 0x9E3779B1u);
        for (int i = 0; i < n; i++)
            src_buf[i] = (m4t_trit_t)((int)(rng_u32(&r) % 3u) - 1);

        /* Bit-exactness gate. */
        m4t_pack_trits_1d           (dst_neon,   src_buf, n);
        m4t_pack_trits_1d_scalar_ref(dst_scalar, src_buf, n);
        if (memcmp(dst_neon, dst_scalar, (size_t)M4T_TRIT_PACKED_BYTES(n)) != 0) {
            fprintf(stderr, "[ERROR] bit-exactness fail: pack_1d n=%d\n", n);
            continue;
        }
        flush_caches(flush_buf, FLUSH_SIZE);
        m4t_pack_trits_1d(dst_neon, src_buf, n);  /* warm */

        double sum_neon = 0, sum_scalar = 0;
        for (int rep = 0; rep < cfg->reps; rep++) {
            flush_caches(flush_buf, FLUSH_SIZE);
            double t0 = now_ms();
            m4t_pack_trits_1d(dst_neon, src_buf, n);
            sum_neon += now_ms() - t0;

            flush_caches(flush_buf, FLUSH_SIZE);
            t0 = now_ms();
            m4t_pack_trits_1d_scalar_ref(dst_scalar, src_buf, n);
            sum_scalar += now_ms() - t0;
        }
        double ne = (sum_neon / cfg->reps) * 1e6 / n;
        double sc = (sum_scalar / cfg->reps) * 1e6 / n;
        BENCH_ROW(cfg->note, n, cfg->reps, ne, sc);
    }
}

static void bench_unpack_1d(uint8_t* flush_buf,
                            uint8_t* src_buf, m4t_trit_t* dst_neon, m4t_trit_t* dst_scalar) {
    BENCH_HEAD("m4t_unpack_trits_1d (4-in-8 unpack)");
    for (int c = 0; c < N_CFGS; c++) {
        const Cfg* cfg = &CFGS[c];
        int n = cfg->n;
        int nb = M4T_TRIT_PACKED_BYTES(n);
        rng_t r; rng_init(&r, (uint32_t)(c + 1) * 0x85EBCA6Bu);
        for (int i = 0; i < nb; i++) {
            /* Random bytes; reserved 0b11 codes decode to 0 — both paths agree. */
            src_buf[i] = (uint8_t)rng_u32(&r);
        }

        m4t_unpack_trits_1d           (dst_neon,   src_buf, n);
        m4t_unpack_trits_1d_scalar_ref(dst_scalar, src_buf, n);
        if (memcmp(dst_neon, dst_scalar, (size_t)n * sizeof(m4t_trit_t)) != 0) {
            fprintf(stderr, "[ERROR] bit-exactness fail: unpack_1d n=%d\n", n);
            continue;
        }
        flush_caches(flush_buf, FLUSH_SIZE);
        m4t_unpack_trits_1d(dst_neon, src_buf, n);

        double sum_neon = 0, sum_scalar = 0;
        for (int rep = 0; rep < cfg->reps; rep++) {
            flush_caches(flush_buf, FLUSH_SIZE);
            double t0 = now_ms();
            m4t_unpack_trits_1d(dst_neon, src_buf, n);
            sum_neon += now_ms() - t0;

            flush_caches(flush_buf, FLUSH_SIZE);
            t0 = now_ms();
            m4t_unpack_trits_1d_scalar_ref(dst_scalar, src_buf, n);
            sum_scalar += now_ms() - t0;
        }
        double ne = (sum_neon / cfg->reps) * 1e6 / n;
        double sc = (sum_scalar / cfg->reps) * 1e6 / n;
        BENCH_ROW(cfg->note, n, cfg->reps, ne, sc);
    }
}

static void bench_pack_5in8(uint8_t* flush_buf,
                            m4t_trit_t* src_buf, uint8_t* dst_neon, uint8_t* dst_scalar) {
    BENCH_HEAD("m4t_pack_trits_5in8_1d (5-in-8 pack)");
    for (int c = 0; c < N_CFGS; c++) {
        const Cfg* cfg = &CFGS[c];
        int n = cfg->n;
        rng_t r; rng_init(&r, (uint32_t)(c + 1) * 0xC2B2AE3Du);
        for (int i = 0; i < n; i++)
            src_buf[i] = (m4t_trit_t)((int)(rng_u32(&r) % 3u) - 1);

        m4t_pack_trits_5in8_1d           (dst_neon,   src_buf, n);
        m4t_pack_trits_5in8_1d_scalar_ref(dst_scalar, src_buf, n);
        if (memcmp(dst_neon, dst_scalar, (size_t)M4T_TRIT_PACKED5_BYTES(n)) != 0) {
            fprintf(stderr, "[ERROR] bit-exactness fail: pack_5in8 n=%d\n", n);
            continue;
        }
        flush_caches(flush_buf, FLUSH_SIZE);
        m4t_pack_trits_5in8_1d(dst_neon, src_buf, n);

        double sum_neon = 0, sum_scalar = 0;
        for (int rep = 0; rep < cfg->reps; rep++) {
            flush_caches(flush_buf, FLUSH_SIZE);
            double t0 = now_ms();
            m4t_pack_trits_5in8_1d(dst_neon, src_buf, n);
            sum_neon += now_ms() - t0;

            flush_caches(flush_buf, FLUSH_SIZE);
            t0 = now_ms();
            m4t_pack_trits_5in8_1d_scalar_ref(dst_scalar, src_buf, n);
            sum_scalar += now_ms() - t0;
        }
        double ne = (sum_neon / cfg->reps) * 1e6 / n;
        double sc = (sum_scalar / cfg->reps) * 1e6 / n;
        BENCH_ROW(cfg->note, n, cfg->reps, ne, sc);
    }
}

static void bench_unpack_5in8(uint8_t* flush_buf,
                              uint8_t* src_buf, m4t_trit_t* dst_neon, m4t_trit_t* dst_scalar) {
    BENCH_HEAD("m4t_unpack_trits_5in8_1d (5-in-8 unpack)");
    for (int c = 0; c < N_CFGS; c++) {
        const Cfg* cfg = &CFGS[c];
        int n = cfg->n;
        int nb = M4T_TRIT_PACKED5_BYTES(n);
        /* Generate via pack (so input bytes are valid 5-in-8 codes). */
        rng_t r; rng_init(&r, (uint32_t)(c + 1) * 0x27D4EB2Fu);
        for (int i = 0; i < n; i++) {
            ((m4t_trit_t*)dst_neon)[i] = (m4t_trit_t)((int)(rng_u32(&r) % 3u) - 1);
        }
        m4t_pack_trits_5in8_1d(src_buf, (const m4t_trit_t*)dst_neon, n);

        m4t_unpack_trits_5in8_1d           (dst_neon,   src_buf, n);
        m4t_unpack_trits_5in8_1d_scalar_ref(dst_scalar, src_buf, n);
        if (memcmp(dst_neon, dst_scalar, (size_t)n * sizeof(m4t_trit_t)) != 0) {
            fprintf(stderr, "[ERROR] bit-exactness fail: unpack_5in8 n=%d\n", n);
            continue;
        }
        flush_caches(flush_buf, FLUSH_SIZE);
        m4t_unpack_trits_5in8_1d(dst_neon, src_buf, n);
        (void)nb;

        double sum_neon = 0, sum_scalar = 0;
        for (int rep = 0; rep < cfg->reps; rep++) {
            flush_caches(flush_buf, FLUSH_SIZE);
            double t0 = now_ms();
            m4t_unpack_trits_5in8_1d(dst_neon, src_buf, n);
            sum_neon += now_ms() - t0;

            flush_caches(flush_buf, FLUSH_SIZE);
            t0 = now_ms();
            m4t_unpack_trits_5in8_1d_scalar_ref(dst_scalar, src_buf, n);
            sum_scalar += now_ms() - t0;
        }
        double ne = (sum_neon / cfg->reps) * 1e6 / n;
        double sc = (sum_scalar / cfg->reps) * 1e6 / n;
        BENCH_ROW(cfg->note, n, cfg->reps, ne, sc);
    }
}

static void bench_narrow(uint8_t* flush_buf,
                         m4t_mtfp_t* src_buf, m4t_mtfp4_t* dst_neon, m4t_mtfp4_t* dst_scalar) {
    BENCH_HEAD("m4t_mtfp19_to_mtfp4 (cell-width narrow)");
    for (int c = 0; c < N_CFGS; c++) {
        const Cfg* cfg = &CFGS[c];
        int n = cfg->n;
        rng_t r; rng_init(&r, (uint32_t)(c + 1) * 0x165667B1u);
        for (int i = 0; i < n; i++) {
            uint32_t u = rng_u32(&r);
            int32_t v = (int32_t)(u % (uint32_t)(2 * M4T_MTFP_MAX_VAL + 1)) - M4T_MTFP_MAX_VAL;
            src_buf[i] = v;
        }

        m4t_mtfp19_to_mtfp4           (dst_neon,   src_buf, NULL, n);
        m4t_mtfp19_to_mtfp4_scalar_ref(dst_scalar, src_buf, NULL, n);
        if (memcmp(dst_neon, dst_scalar, (size_t)n) != 0) {
            fprintf(stderr, "[ERROR] bit-exactness fail: mtfp19_to_mtfp4 n=%d\n", n);
            continue;
        }
        flush_caches(flush_buf, FLUSH_SIZE);
        m4t_mtfp19_to_mtfp4(dst_neon, src_buf, NULL, n);

        double sum_neon = 0, sum_scalar = 0;
        for (int rep = 0; rep < cfg->reps; rep++) {
            flush_caches(flush_buf, FLUSH_SIZE);
            double t0 = now_ms();
            m4t_mtfp19_to_mtfp4(dst_neon, src_buf, NULL, n);
            sum_neon += now_ms() - t0;

            flush_caches(flush_buf, FLUSH_SIZE);
            t0 = now_ms();
            m4t_mtfp19_to_mtfp4_scalar_ref(dst_scalar, src_buf, NULL, n);
            sum_scalar += now_ms() - t0;
        }
        double ne = (sum_neon / cfg->reps) * 1e6 / n;
        double sc = (sum_scalar / cfg->reps) * 1e6 / n;
        BENCH_ROW(cfg->note, n, cfg->reps, ne, sc);
    }
}

static void bench_widen(uint8_t* flush_buf,
                        m4t_mtfp4_t* src_buf, m4t_mtfp_t* dst_neon, m4t_mtfp_t* dst_scalar) {
    BENCH_HEAD("m4t_mtfp4_to_mtfp19 (cell-width widen)");
    for (int c = 0; c < N_CFGS; c++) {
        const Cfg* cfg = &CFGS[c];
        int n = cfg->n;
        rng_t r; rng_init(&r, (uint32_t)(c + 1) * 0xD3163865u);
        for (int i = 0; i < n; i++) {
            int v = (int)(rng_u32(&r) % 81u) - 40;  /* [-40, +40] = MTFP4 range */
            src_buf[i] = (m4t_mtfp4_t)v;
        }

        m4t_mtfp4_to_mtfp19           (dst_neon,   src_buf, n);
        m4t_mtfp4_to_mtfp19_scalar_ref(dst_scalar, src_buf, n);
        if (memcmp(dst_neon, dst_scalar, (size_t)n * sizeof(m4t_mtfp_t)) != 0) {
            fprintf(stderr, "[ERROR] bit-exactness fail: mtfp4_to_mtfp19 n=%d\n", n);
            continue;
        }
        flush_caches(flush_buf, FLUSH_SIZE);
        m4t_mtfp4_to_mtfp19(dst_neon, src_buf, n);

        double sum_neon = 0, sum_scalar = 0;
        for (int rep = 0; rep < cfg->reps; rep++) {
            flush_caches(flush_buf, FLUSH_SIZE);
            double t0 = now_ms();
            m4t_mtfp4_to_mtfp19(dst_neon, src_buf, n);
            sum_neon += now_ms() - t0;

            flush_caches(flush_buf, FLUSH_SIZE);
            t0 = now_ms();
            m4t_mtfp4_to_mtfp19_scalar_ref(dst_scalar, src_buf, n);
            sum_scalar += now_ms() - t0;
        }
        double ne = (sum_neon / cfg->reps) * 1e6 / n;
        double sc = (sum_scalar / cfg->reps) * 1e6 / n;
        BENCH_ROW(cfg->note, n, cfg->reps, ne, sc);
    }
}

static void bench_shift3_mul(uint8_t* flush_buf,
                             m4t_mtfp_t* src_buf, m4t_mtfp_t* dst_neon, m4t_mtfp_t* dst_scalar) {
    BENCH_HEAD("m4t_mtfp_shift3 k>0 multiply (k=3 fixed)");
    int k = 3;  /* representative: |scale|=27, comfortably non-trivial. */
    for (int c = 0; c < N_CFGS; c++) {
        const Cfg* cfg = &CFGS[c];
        int n = cfg->n;
        rng_t r; rng_init(&r, (uint32_t)(c + 1) * 0xA0761D65u);
        for (int i = 0; i < n; i++) {
            uint32_t u = rng_u32(&r);
            int32_t v = (int32_t)(u % (uint32_t)(2 * M4T_MTFP_MAX_VAL + 1)) - M4T_MTFP_MAX_VAL;
            /* Bound to keep most values from saturating at k=3 (|scale|=27);
             * unbounded inputs would have ~all post-multiply saturate, which
             * isn't representative of real consumer usage. */
            src_buf[i] = (m4t_mtfp_t)(v / 27);
        }

        m4t_mtfp_shift3           (dst_neon,   src_buf, k, n);
        m4t_mtfp_shift3_scalar_ref(dst_scalar, src_buf, k, n);
        if (memcmp(dst_neon, dst_scalar, (size_t)n * sizeof(m4t_mtfp_t)) != 0) {
            fprintf(stderr, "[ERROR] bit-exactness fail: shift3 mul n=%d\n", n);
            continue;
        }
        flush_caches(flush_buf, FLUSH_SIZE);
        m4t_mtfp_shift3(dst_neon, src_buf, k, n);

        double sum_neon = 0, sum_scalar = 0;
        for (int rep = 0; rep < cfg->reps; rep++) {
            flush_caches(flush_buf, FLUSH_SIZE);
            double t0 = now_ms();
            m4t_mtfp_shift3(dst_neon, src_buf, k, n);
            sum_neon += now_ms() - t0;

            flush_caches(flush_buf, FLUSH_SIZE);
            t0 = now_ms();
            m4t_mtfp_shift3_scalar_ref(dst_scalar, src_buf, k, n);
            sum_scalar += now_ms() - t0;
        }
        double ne = (sum_neon / cfg->reps) * 1e6 / n;
        double sc = (sum_scalar / cfg->reps) * 1e6 / n;
        BENCH_ROW(cfg->note, n, cfg->reps, ne, sc);
    }
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(void) {
    uint8_t* flush_buf = (uint8_t*)calloc(FLUSH_SIZE, 1);
    if (!flush_buf) { fprintf(stderr, "OOM flush\n"); return 1; }

    /* Allocate buffers sized for the largest cfg, reused across benches.
     * Byte buffers must be sized for the worst-case packing density:
     * 4-in-8 needs ⌈N/4⌉ bytes (denser than 5-in-8's ⌈N/5⌉). Use the
     * 4-in-8 size for all byte buffers. */
    int max_packed_bytes = (MAX_N + 3) / 4;
    m4t_trit_t* trit_src = (m4t_trit_t*)calloc((size_t)MAX_N, sizeof(m4t_trit_t));
    uint8_t*    byte_src = (uint8_t*)calloc((size_t)max_packed_bytes, 1);
    uint8_t*    byte_dst_neon = (uint8_t*)calloc((size_t)max_packed_bytes, 1);
    uint8_t*    byte_dst_scalar = (uint8_t*)calloc((size_t)max_packed_bytes, 1);
    m4t_trit_t* trit_dst_neon = (m4t_trit_t*)calloc((size_t)MAX_N, sizeof(m4t_trit_t));
    m4t_trit_t* trit_dst_scalar = (m4t_trit_t*)calloc((size_t)MAX_N, sizeof(m4t_trit_t));
    m4t_mtfp_t* mtfp_src = (m4t_mtfp_t*)calloc((size_t)MAX_N, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* mtfp_dst_neon = (m4t_mtfp_t*)calloc((size_t)MAX_N, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* mtfp_dst_scalar = (m4t_mtfp_t*)calloc((size_t)MAX_N, sizeof(m4t_mtfp_t));
    m4t_mtfp4_t* mtfp4_src = (m4t_mtfp4_t*)calloc((size_t)MAX_N, sizeof(m4t_mtfp4_t));
    m4t_mtfp4_t* mtfp4_dst_neon = (m4t_mtfp4_t*)calloc((size_t)MAX_N, sizeof(m4t_mtfp4_t));
    m4t_mtfp4_t* mtfp4_dst_scalar = (m4t_mtfp4_t*)calloc((size_t)MAX_N, sizeof(m4t_mtfp4_t));

    if (!trit_src || !byte_src || !byte_dst_neon || !byte_dst_scalar ||
        !trit_dst_neon || !trit_dst_scalar || !mtfp_src ||
        !mtfp_dst_neon || !mtfp_dst_scalar || !mtfp4_src ||
        !mtfp4_dst_neon || !mtfp4_dst_scalar)
    {
        fprintf(stderr, "OOM bench buffers\n");
        return 1;
    }

    printf("# No-scalar audit setup-time bench (2026-05-06 remediation)\n");
    printf("# NEON public dispatcher vs _scalar_ref test oracle.\n");
    printf("# ns/elem = (mean ms per call × 1e6) / N. Per-rep cache flush.\n");

    bench_pack_1d   (flush_buf, trit_src, byte_dst_neon, byte_dst_scalar);
    bench_unpack_1d (flush_buf, byte_src, trit_dst_neon, trit_dst_scalar);
    bench_pack_5in8 (flush_buf, trit_src, byte_dst_neon, byte_dst_scalar);
    bench_unpack_5in8(flush_buf, byte_src, trit_dst_neon, trit_dst_scalar);
    bench_narrow    (flush_buf, mtfp_src, mtfp4_dst_neon, mtfp4_dst_scalar);
    bench_widen     (flush_buf, mtfp4_src, mtfp_dst_neon, mtfp_dst_scalar);
    bench_shift3_mul(flush_buf, mtfp_src, mtfp_dst_neon, mtfp_dst_scalar);

    printf("\n=== Done. Reading the table ===\n");
    printf("speedup column = ns/elem_scalar / ns/elem_NEON. Higher = bigger win.\n");
    printf("Tail rows test n%%16==1 (4-in-8) and n%%80==1 (5-in-8) — the\n"
           "geometric scalar tail's worst case (1 trit's worth of remainder).\n");

    free(flush_buf);
    free(trit_src); free(byte_src); free(byte_dst_neon); free(byte_dst_scalar);
    free(trit_dst_neon); free(trit_dst_scalar);
    free(mtfp_src); free(mtfp_dst_neon); free(mtfp_dst_scalar);
    free(mtfp4_src); free(mtfp4_dst_neon); free(mtfp4_dst_scalar);
    return 0;
}
