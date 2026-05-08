/*
 * test_m4t_mtfp4.c — tests for the SDOT ternary matmul + cell-width
 * conversions. Hand-derived golden values where small enough; bit-exact
 * int64 reference for the matmul (since SDOT must be exact per §8.4).
 *
 * Coverage:
 *   1.  mtfp4_clamp_basic        — clamping at MTFP4 boundaries
 *   2.  sdot_matmul_small        — golden-value 2x4x4 matmul
 *   3.  sdot_matmul_large        — random K up to 1024 vs int64 reference
 *   4.  sdot_matmul_high_mag     — extreme inputs at K=4096, verify exact
 *   5.  sdot_matmul_long_k       — K=1M vs int64 reference (exercises real
 *                                  K-bound regime, partway to K_MAX_EXACT)
 *   6.  sdot_zero_dim            — M=0/N=0/K=0 edge cases
 *   7.  widen_exact              — mtfp4_to_mtfp19 boundary values
 *   8.  narrow_round             — mtfp19_to_mtfp4 round-to-nearest cases
 *   9.  narrow_saturate          — mtfp19_to_mtfp4 saturation cases
 *  10.  narrow_flags             — ROUNDED / SATURATED bits exact
 *  11.  narrow_property          — 10k random src vs int64 reference
 *  12.  roundtrip_widen_narrow   — widen then narrow recovers exact value
 */

#include "m4t_mtfp4.h"
#include "m4t_mtfp.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT_EQ_I32(got, want, msg) do { \
    if ((int32_t)(got) != (int32_t)(want)) { \
        printf("FAIL %s: got=%d want=%d\n", (msg), (int)(got), (int)(want)); \
        return 1; \
    } \
} while (0)

/* ── xorshift32 RNG ─────────────────────────────────────────────────────── */
static uint32_t g_rng = 0xdeadbeefu;
static uint32_t xs32(void) {
    uint32_t x = g_rng; x ^= x << 13; x ^= x >> 17; x ^= x << 5; g_rng = x; return x;
}
static int8_t rand_mtfp4(void) {
    /* Uniform in [-40, 40]. */
    return (int8_t)((int)(xs32() % 81u) - 40);
}
static int8_t rand_trit(void) {
    /* Uniform in {-1, 0, +1}. */
    return (int8_t)((int)(xs32() % 3u) - 1);
}
static int32_t rand_mtfp19(void) {
    int64_t span = (int64_t)M4T_MTFP_MAX_VAL * 2 + 1;
    return (int32_t)((int64_t)(xs32() % (uint64_t)span) - (int64_t)M4T_MTFP_MAX_VAL);
}
static int rand_int(int lo, int hi) {
    return lo + (int)(xs32() % (uint32_t)(hi - lo + 1));
}

/* Bit-exact int64 reference for the narrow conversion. Mirrors the kernel
 * algorithm in m4t_mtfp4.c structurally (round-to-nearest-even by 6561,
 * then saturating clamp), used as the property-test oracle. */
static int8_t narrow_reference(int32_t v, int* out_rounded, int* out_saturated) {
    int32_t s = 6561;
    int32_t q = v / s;
    int32_t rem = v - q * s;
    *out_rounded = (rem != 0) ? 1 : 0;
    if (rem > 0 && 2 * rem > s) q += 1;
    else if (rem < 0 && 2 * (-rem) > s) q -= 1;
    int8_t out;
    *out_saturated = 0;
    if (q >  M4T_MTFP4_MAX_VAL) { out =  M4T_MTFP4_MAX_VAL; *out_saturated = 1; }
    else if (q < -M4T_MTFP4_MAX_VAL) { out = -M4T_MTFP4_MAX_VAL; *out_saturated = 1; }
    else out = (int8_t)q;
    return out;
}

/* ── Tests ──────────────────────────────────────────────────────────────── */

static int test_clamp_basic(void) {
    ASSERT_EQ_I32(m4t_mtfp4_clamp(0), 0, "clamp(0)");
    ASSERT_EQ_I32(m4t_mtfp4_clamp(40), 40, "clamp(40)");
    ASSERT_EQ_I32(m4t_mtfp4_clamp(-40), -40, "clamp(-40)");
    ASSERT_EQ_I32(m4t_mtfp4_clamp(41), 40, "clamp(41)");
    ASSERT_EQ_I32(m4t_mtfp4_clamp(-41), -40, "clamp(-41)");
    ASSERT_EQ_I32(m4t_mtfp4_clamp(1000000), 40, "clamp(huge)");
    return 0;
}

/* Golden-value matmul: 2 rows × 4 cols × K=4. Computed by hand. */
static int test_sdot_matmul_small(void) {
    /* X = [[1, 2, 3, 4], [-1, -2, -3, -4]]  (M=2, K=4) */
    int8_t X[8] = { 1, 2, 3, 4,  -1, -2, -3, -4 };
    /* W = [[+1,-1, 0,+1], [-1, 0,+1,+1], [ 0, 0,+1, 0], [+1,+1,+1,+1]]  (N=4, K=4) */
    int8_t W[16] = {
         1, -1,  0,  1,
        -1,  0,  1,  1,
         0,  0,  1,  0,
         1,  1,  1,  1
    };
    /* Y = X @ W^T:
     *   Y[0,0] = 1·1 + 2·-1 + 3·0 + 4·1 =  3
     *   Y[0,1] = 1·-1 + 2·0 + 3·1 + 4·1 =  6
     *   Y[0,2] = 1·0 + 2·0 + 3·1 + 4·0 =  3
     *   Y[0,3] = 1·1 + 2·1 + 3·1 + 4·1 = 10
     *   Y[1,*] = -Y[0,*] = [-3, -6, -3, -10] */
    int32_t Y[8];
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, 2, 4, 4);
    int32_t expected[8] = { 3, 6, 3, 10,  -3, -6, -3, -10 };
    for (int i = 0; i < 8; i++) {
        if (Y[i] != expected[i]) {
            printf("FAIL sdot_small[%d]: got=%d want=%d\n",
                   i, Y[i], expected[i]);
            return 1;
        }
    }
    return 0;
}

/* Random large-K matmul: bit-exact vs int64 reference. */
static int test_sdot_matmul_large(void) {
    g_rng = 0xc001d00du;
    for (int trial = 0; trial < 200; trial++) {
        int M = rand_int(1, 4);
        int N = rand_int(1, 4);
        int K = rand_int(16, 1024);    /* exercises the NEON loop and tail */
        int8_t* X = malloc((size_t)M * K);
        int8_t* W = malloc((size_t)N * K);
        int32_t* Y = malloc((size_t)M * N * sizeof(int32_t));
        int32_t* Yref = malloc((size_t)M * N * sizeof(int32_t));

        for (int i = 0; i < M * K; i++) X[i] = rand_mtfp4();
        for (int i = 0; i < N * K; i++) W[i] = rand_trit();

        m4t_mtfp4_sdot_matmul_bt(Y, X, W, M, K, N);

        /* int64 reference. */
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int64_t acc = 0;
                for (int k = 0; k < K; k++) {
                    acc += (int64_t)X[i*K+k] * (int64_t)W[j*K+k];
                }
                Yref[i*N+j] = (int32_t)acc;
            }
        }

        for (int i = 0; i < M*N; i++) {
            if (Y[i] != Yref[i]) {
                printf("FAIL sdot_large trial %d cell %d: got=%d ref=%d (M=%d K=%d N=%d)\n",
                       trial, i, Y[i], Yref[i], M, K, N);
                free(X); free(W); free(Y); free(Yref);
                return 1;
            }
        }
        free(X); free(W); free(Y); free(Yref);
    }
    return 0;
}

/* High-magnitude inputs at K=4096: verify exact (no overflow into MTFP19). */
static int test_sdot_matmul_high_mag(void) {
    int K = 4096;
    int8_t* X = malloc((size_t)K);
    int8_t* W = malloc((size_t)K);
    int32_t Y[1];

    /* All X = +40, all W = +1: Y = K · 40 = 163840 */
    for (int k = 0; k < K; k++) { X[k] = 40; W[k] = 1; }
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, 1, K, 1);
    if (Y[0] != K * 40) {
        printf("FAIL high_mag +1: got=%d want=%d\n", Y[0], K * 40);
        free(X); free(W); return 1;
    }

    /* All X = +40, all W = -1: Y = -K · 40 */
    for (int k = 0; k < K; k++) W[k] = -1;
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, 1, K, 1);
    if (Y[0] != -K * 40) {
        printf("FAIL high_mag -1: got=%d want=%d\n", Y[0], -K * 40);
        free(X); free(W); return 1;
    }

    /* Verify result is in MTFP19 range. */
    if (Y[0] < -M4T_MTFP_MAX_VAL || Y[0] > M4T_MTFP_MAX_VAL) {
        printf("FAIL high_mag: |Y|=%d outside MTFP19\n", Y[0]);
        free(X); free(W); return 1;
    }

    free(X); free(W);
    return 0;
}

/* Long-K stress at K=1M, partway to M4T_SDOT_K_MAX_EXACT (~14.5M). Verifies
 * accumulator behavior under realistic large-K workloads and exercises both
 * the NEON loop body and the bit-exact correspondence with int64 reference.
 * Worst-case |Y| at K=1M is K · 40 = 40,000,000 — well within MTFP19 range. */
static int test_sdot_matmul_long_k(void) {
    int K = 1000000;
    int8_t* X = malloc((size_t)K);
    int8_t* W = malloc((size_t)K);
    int32_t Y, Yref;

    /* Adversarial: all X = ±40, all W = ±1, mixed signs to avoid trivial
     * cancellation. Worst-case sum dominates. */
    g_rng = 0xa5a5a5a5u;
    for (int k = 0; k < K; k++) {
        X[k] = (xs32() & 1) ? 40 : -40;
        W[k] = (int8_t)((int)(xs32() % 3u) - 1);
    }

    m4t_mtfp4_sdot_matmul_bt(&Y, X, W, 1, K, 1);

    /* int64 reference. */
    int64_t acc = 0;
    for (int k = 0; k < K; k++) acc += (int64_t)X[k] * (int64_t)W[k];
    Yref = (int32_t)acc;

    if (Y != Yref) {
        printf("FAIL long_k K=%d: got=%d ref=%d\n", K, Y, Yref);
        free(X); free(W); return 1;
    }
    if (Y < -M4T_MTFP_MAX_VAL || Y > M4T_MTFP_MAX_VAL) {
        printf("FAIL long_k: |Y|=%d outside MTFP19\n", Y);
        free(X); free(W); return 1;
    }

    free(X); free(W);
    return 0;
}

static int test_sdot_zero_dim(void) {
    int32_t Y[4] = { 99, 99, 99, 99 };  /* sentinel */
    int8_t X[4] = { 1, 1, 1, 1 };
    int8_t W[4] = { 1, 1, 1, 1 };

    /* M=0 — kernel must not touch Y. */
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, 0, 4, 1);
    if (Y[0] != 99) { printf("FAIL zero_dim M=0\n"); return 1; }

    /* N=0 — ditto. */
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, 1, 4, 0);
    if (Y[0] != 99) { printf("FAIL zero_dim N=0\n"); return 1; }

    /* K=0 — Y[0,0] = empty sum = 0. */
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, 1, 0, 1);
    if (Y[0] != 0) { printf("FAIL zero_dim K=0: got=%d\n", Y[0]); return 1; }

    return 0;
}

static int test_widen_exact(void) {
    int8_t src[5] = { 0, 1, -1, 40, -40 };
    int32_t dst[5];
    m4t_mtfp4_to_mtfp19(dst, src, 5);
    int32_t expected[5] = { 0, 6561, -6561, 40 * 6561, -40 * 6561 };
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ_I32(dst[i], expected[i], "widen");
    }
    return 0;
}

static int test_narrow_round(void) {
    /* Verify base-3 round-to-nearest by examining values straddling the
     * rescale boundary. SCALE = 6561; halfway point is 3280.5 (integer
     * remainders never hit it).
     *   src  =     0:        q=0,    rem=0           → 0,    not rounded
     *   src  =  3280:        q=0,    rem=3280  <half → 0,    rounded
     *   src  =  3281:        q=0,    rem=3281  >half → 1,    rounded
     *   src  =  6561:        q=1,    rem=0           → 1,    not rounded
     *   src  = -3280:        q=0,    rem=-3280 <half → 0,    rounded
     *   src  = -3281:        q=0,    rem=-3281 >half →-1,    rounded */
    int32_t src[6] = { 0, 3280, 3281, 6561, -3280, -3281 };
    int8_t dst[6];
    m4t_mtfp19_to_mtfp4(dst, src, NULL, 6);
    int8_t expected[6] = { 0, 0, 1, 1, 0, -1 };
    for (int i = 0; i < 6; i++) {
        ASSERT_EQ_I32(dst[i], expected[i], "narrow_round");
    }
    return 0;
}

static int test_narrow_saturate(void) {
    /* Inputs that exceed MTFP4 range after rescale clamp at ±40. */
    int32_t src[3] = {
        40 * 6561 + 6561,   /* rounds to 41, saturates to 40 */
        -40 * 6561 - 6561,  /* rounds to -41, saturates to -40 */
        100 * 6561,         /* rounds to 100, saturates to 40 */
    };
    int8_t dst[3];
    m4t_mtfp19_to_mtfp4(dst, src, NULL, 3);
    ASSERT_EQ_I32(dst[0],  40, "sat +");
    ASSERT_EQ_I32(dst[1], -40, "sat -");
    ASSERT_EQ_I32(dst[2],  40, "sat huge");
    return 0;
}

static int test_narrow_flags(void) {
    /* Mix of rounded and saturated cells. */
    int32_t src[8] = {
        0,           /* exact zero, no flags */
        3280,        /* rounded, not saturated */
        6561,        /* exact, no flags */
        100 * 6561,  /* rounded ×0 (exact div), saturated */
        40 * 6561,   /* exact div, no flags (exactly 40 = MAX) */
        40 * 6561 + 1, /* rounded (rem=1), not saturated (still rounds to 40) */
        50 * 6561 + 10, /* rounded AND saturated */
        -3281,       /* rounded, not saturated */
    };
    int n = 8;
    size_t fb = M4T_FLAG_BYTES(n);
    int8_t dst[8];
    uint8_t flags[2];
    memset(flags, 0, fb);
    m4t_mtfp19_to_mtfp4(dst, src, flags, n);

    /* Per-cell expectations:
     *   0: no flags
     *   1: ROUNDED only
     *   2: no flags
     *   3: SATURATED only (100*6561 / 6561 = 100 exact, then clamp to 40)
     *   4: no flags
     *   5: ROUNDED only (40*6561+1 / 6561 = 40 with rem=1, no sat)
     *   6: ROUNDED + SATURATED
     *   7: ROUNDED only */
    uint8_t expected[8] = {
        0,
        M4T_FLAG_ROUNDED,
        0,
        M4T_FLAG_SATURATED,
        0,
        M4T_FLAG_ROUNDED,
        M4T_FLAG_ROUNDED | M4T_FLAG_SATURATED,
        M4T_FLAG_ROUNDED,
    };
    for (int i = 0; i < n; i++) {
        uint8_t got_sat = m4t_flag_test(flags, i, M4T_FLAG_SATURATED);
        uint8_t got_rnd = m4t_flag_test(flags, i, M4T_FLAG_ROUNDED);
        uint8_t got = (got_sat ? M4T_FLAG_SATURATED : 0) |
                      (got_rnd ? M4T_FLAG_ROUNDED   : 0);
        if (got != expected[i]) {
            printf("FAIL narrow_flags cell %d: got=0x%02x want=0x%02x\n",
                   i, got, expected[i]);
            return 1;
        }
    }
    return 0;
}

static int test_roundtrip_widen_narrow(void) {
    /* For values in MTFP4 range, widen then narrow recovers exact. */
    int8_t src[5] = { 0, 1, -1, 40, -40 };
    int32_t mid[5];
    int8_t back[5];
    m4t_mtfp4_to_mtfp19(mid, src, 5);
    m4t_mtfp19_to_mtfp4(back, mid, NULL, 5);
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ_I32(back[i], src[i], "roundtrip");
    }
    return 0;
}

/* Random property test for narrow conversion: 10k random src values,
 * compare kernel output (mantissa + flag bits) bit-exactly against an
 * int64 reference. Catches off-by-one rounding bugs and flag-encoding
 * mistakes that the hand-derived tests above might miss. */
static int test_narrow_property(void) {
    int n = 64;
    int32_t* src = malloc(sizeof(int32_t) * (size_t)n);
    int8_t*  kernel_dst = malloc((size_t)n);
    uint8_t* kernel_flags = malloc(M4T_FLAG_BYTES(n));

    g_rng = 0xc0ffeebbu;
    for (int s = 0; s < 10000; s++) {
        /* Mix uniformly random values with values targeted at boundaries
         * (3280, 3281, 6561, MAX*6561) to exercise rounding + saturation. */
        for (int i = 0; i < n; i++) {
            uint32_t r = xs32();
            if ((r % 8) == 0) {
                /* Boundary-targeted: near a rescale halfway or saturation edge. */
                int variant = (int)((r >> 3) % 6u);
                int32_t base;
                switch (variant) {
                    case 0: base = 0; break;
                    case 1: base = 3280; break;
                    case 2: base = 3281; break;
                    case 3: base = 6561; break;
                    case 4: base = (int32_t)M4T_MTFP4_MAX_VAL * 6561; break;
                    default: base = (int32_t)(M4T_MTFP4_MAX_VAL * 6561 + 100);
                }
                src[i] = ((r >> 16) & 1) ? base : -base;
            } else {
                src[i] = rand_mtfp19();
            }
        }
        memset(kernel_flags, 0, M4T_FLAG_BYTES(n));
        m4t_mtfp19_to_mtfp4(kernel_dst, src, kernel_flags, n);

        for (int i = 0; i < n; i++) {
            int ref_rounded = 0, ref_saturated = 0;
            int8_t ref_dst = narrow_reference(src[i], &ref_rounded, &ref_saturated);
            if (kernel_dst[i] != ref_dst) {
                printf("FAIL narrow_property s=%d cell %d: src=%d kernel=%d ref=%d\n",
                       s, i, src[i], kernel_dst[i], ref_dst);
                free(src); free(kernel_dst); free(kernel_flags); return 1;
            }
            int got_sat = m4t_flag_test(kernel_flags, i, M4T_FLAG_SATURATED) ? 1 : 0;
            int got_rnd = m4t_flag_test(kernel_flags, i, M4T_FLAG_ROUNDED)   ? 1 : 0;
            if (got_sat != ref_saturated) {
                printf("FAIL narrow_property s=%d cell %d sat: src=%d kernel=%d ref=%d\n",
                       s, i, src[i], got_sat, ref_saturated);
                free(src); free(kernel_dst); free(kernel_flags); return 1;
            }
            if (got_rnd != ref_rounded) {
                printf("FAIL narrow_property s=%d cell %d rnd: src=%d kernel=%d ref=%d\n",
                       s, i, src[i], got_rnd, ref_rounded);
                free(src); free(kernel_dst); free(kernel_flags); return 1;
            }
        }
    }

    free(src); free(kernel_dst); free(kernel_flags);
    return 0;
}

/* NEON vs scalar_ref bit-exact for the conversion functions. Per
 * no-scalar-audit remediation 2026-05-06: these were scalar in production
 * until we added NEON paths. Ensure NEON output bit-exact matches the
 * reference scalar implementation across diverse N values. */
static int test_conversions_neon_vs_scalar_ref(void) {
    g_rng = 0xfeed1234u;
    const int Ns[] = { 1, 3, 4, 5, 8, 15, 16, 17, 31, 32, 33,
                       63, 64, 65, 127, 128, 129, 255, 256, 257 };
    for (size_t k = 0; k < sizeof(Ns)/sizeof(Ns[0]); k++) {
        int n = Ns[k];

        /* Test mtfp19_to_mtfp4 NEON vs scalar_ref. */
        int32_t* src19 = malloc(sizeof(int32_t) * (size_t)n);
        int8_t*  d_neon = malloc((size_t)n);
        int8_t*  d_ref  = malloc((size_t)n);
        uint8_t* f_neon = malloc(M4T_FLAG_BYTES(n));
        uint8_t* f_ref  = malloc(M4T_FLAG_BYTES(n));
        for (int s = 0; s < 100; s++) {
            for (int i = 0; i < n; i++) src19[i] = rand_mtfp19();
            memset(f_neon, 0, M4T_FLAG_BYTES(n));
            memset(f_ref,  0, M4T_FLAG_BYTES(n));
            m4t_mtfp19_to_mtfp4(d_neon, src19, f_neon, n);
            m4t_mtfp19_to_mtfp4_scalar_ref(d_ref, src19, f_ref, n);
            if (memcmp(d_neon, d_ref, (size_t)n) != 0 ||
                memcmp(f_neon, f_ref, M4T_FLAG_BYTES(n)) != 0) {
                fprintf(stderr, "FAIL mtfp19_to_mtfp4 NEON-vs-scalar n=%d sample=%d\n", n, s);
                free(src19); free(d_neon); free(d_ref); free(f_neon); free(f_ref);
                return 1;
            }
        }
        free(src19); free(d_neon); free(d_ref); free(f_neon); free(f_ref);

        /* Test mtfp4_to_mtfp19 NEON vs scalar_ref. */
        int8_t*  src4  = malloc((size_t)n);
        int32_t* w_neon = malloc(sizeof(int32_t) * (size_t)n);
        int32_t* w_ref  = malloc(sizeof(int32_t) * (size_t)n);
        for (int s = 0; s < 100; s++) {
            for (int i = 0; i < n; i++) src4[i] = rand_mtfp4();
            m4t_mtfp4_to_mtfp19(w_neon, src4, n);
            m4t_mtfp4_to_mtfp19_scalar_ref(w_ref, src4, n);
            if (memcmp(w_neon, w_ref, sizeof(int32_t) * (size_t)n) != 0) {
                fprintf(stderr, "FAIL mtfp4_to_mtfp19 NEON-vs-scalar n=%d sample=%d\n", n, s);
                free(src4); free(w_neon); free(w_ref);
                return 1;
            }
        }
        free(src4); free(w_neon); free(w_ref);
    }
    return 0;
}

/* Per journal/k80_fix_lmm.md (mirror for K%16): explicit sweep covering
 * every boundary-tile pattern. K = 16 + km for km ∈ {1..15} hits every
 * K%16 mod with both main loop AND boundary tile active. K<16 cases
 * exercise the boundary-tile-only path. */
static int test_sdot_matmul_k16_sweep(void) {
    g_rng = 0xb04e5e7au;
    for (int km = 1; km < 16; km++) {
        for (int K_base = 16; K_base <= 64; K_base += 16) {
            int K = K_base + km;
            int M = 2, N = 5;  /* 5 = 4 j_tile + 1 j_tail */
            int8_t* X = malloc((size_t)M * K);
            int8_t* W = malloc((size_t)N * K);
            int32_t* Y = malloc((size_t)M * N * sizeof(int32_t));
            int32_t* Yref = malloc((size_t)M * N * sizeof(int32_t));

            for (int i = 0; i < M * K; i++) X[i] = rand_mtfp4();
            for (int i = 0; i < N * K; i++) W[i] = rand_trit();

            m4t_mtfp4_sdot_matmul_bt(Y, X, W, M, K, N);

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    int64_t acc = 0;
                    for (int k = 0; k < K; k++) {
                        acc += (int64_t)X[i*K+k] * (int64_t)W[j*K+k];
                    }
                    Yref[i*N+j] = (int32_t)acc;
                }
            }
            for (int i = 0; i < M*N; i++) {
                if (Y[i] != Yref[i]) {
                    printf("FAIL k16_sweep K=%d cell %d: got=%d ref=%d\n",
                           K, i, Y[i], Yref[i]);
                    free(X); free(W); free(Y); free(Yref);
                    return 1;
                }
            }
            free(X); free(W); free(Y); free(Yref);
        }
    }
    /* K<16 (entire kernel is boundary tile). */
    int K_lt[] = {1, 5, 8, 15};
    for (size_t kli = 0; kli < sizeof(K_lt)/sizeof(K_lt[0]); kli++) {
        int K = K_lt[kli];
        int M = 2, N = 5;
        int8_t* X = malloc((size_t)M * K);
        int8_t* W = malloc((size_t)N * K);
        int32_t* Y = malloc((size_t)M * N * sizeof(int32_t));
        int32_t* Yref = malloc((size_t)M * N * sizeof(int32_t));
        for (int i = 0; i < M * K; i++) X[i] = rand_mtfp4();
        for (int i = 0; i < N * K; i++) W[i] = rand_trit();
        m4t_mtfp4_sdot_matmul_bt(Y, X, W, M, K, N);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int64_t acc = 0;
                for (int k = 0; k < K; k++) acc += (int64_t)X[i*K+k] * (int64_t)W[j*K+k];
                Yref[i*N+j] = (int32_t)acc;
            }
        }
        for (int i = 0; i < M*N; i++) {
            if (Y[i] != Yref[i]) {
                printf("FAIL K<16 K=%d cell %d: got=%d ref=%d\n", K, i, Y[i], Yref[i]);
                free(X); free(W); free(Y); free(Yref);
                return 1;
            }
        }
        free(X); free(W); free(Y); free(Yref);
    }
    /* K=0 explicit. */
    {
        int M = 4, N = 4;
        int32_t Y[16];
        for (int i = 0; i < 16; i++) Y[i] = 0xDEADBEEF;
        m4t_mtfp4_sdot_matmul_bt(Y, NULL, NULL, M, 0, N);
        for (int i = 0; i < 16; i++) {
            if (Y[i] != 0) {
                printf("FAIL K=0 expected 0, got %d at %d\n", Y[i], i);
                return 1;
            }
        }
    }
    return 0;
}

/* V3 (pure-ternary audit): route variant bit-exact vs the int64
 * reference. Mirrors the random-K and K%16-sweep coverage. */
static int test_sdot_matmul_route_bit_exact(void) {
    g_rng = 0x0007E2A0u;
    /* Random shapes */
    for (int trial = 0; trial < 100; trial++) {
        int M = rand_int(1, 4);
        int N = rand_int(1, 4);
        int K = rand_int(16, 1024);
        int8_t* X = malloc((size_t)M * K);
        int8_t* W = malloc((size_t)N * K);
        int32_t* Y = malloc((size_t)M * N * sizeof(int32_t));
        int32_t* Yref = malloc((size_t)M * N * sizeof(int32_t));
        for (int i = 0; i < M * K; i++) X[i] = rand_mtfp4();
        for (int i = 0; i < N * K; i++) W[i] = rand_trit();

        m4t_mtfp4_sdot_matmul_bt_route(Y, X, W, M, K, N);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int64_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int64_t)X[i*K+k] * (int64_t)W[j*K+k];
                Yref[i*N+j] = (int32_t)acc;
            }
        }
        for (int i = 0; i < M*N; i++) {
            if (Y[i] != Yref[i]) {
                printf("FAIL route trial %d cell %d: got=%d ref=%d (M=%d K=%d N=%d)\n",
                       trial, i, Y[i], Yref[i], M, K, N);
                free(X); free(W); free(Y); free(Yref); return 1;
            }
        }
        free(X); free(W); free(Y); free(Yref);
    }
    /* K%16 sweep on route. */
    for (int km = 1; km < 16; km++) {
        for (int K_base = 16; K_base <= 64; K_base += 16) {
            int K = K_base + km;
            int M = 2, N = 5;
            int8_t* X = malloc((size_t)M * K);
            int8_t* W = malloc((size_t)N * K);
            int32_t* Y = malloc((size_t)M * N * sizeof(int32_t));
            int32_t* Yref = malloc((size_t)M * N * sizeof(int32_t));
            for (int i = 0; i < M * K; i++) X[i] = rand_mtfp4();
            for (int i = 0; i < N * K; i++) W[i] = rand_trit();
            m4t_mtfp4_sdot_matmul_bt_route(Y, X, W, M, K, N);
            for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
                int64_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int64_t)X[i*K+k] * (int64_t)W[j*K+k];
                Yref[i*N+j] = (int32_t)acc;
            }
            for (int i = 0; i < M*N; i++) {
                if (Y[i] != Yref[i]) {
                    printf("FAIL route k16 sweep K=%d cell %d: got=%d ref=%d\n",
                           K, i, Y[i], Yref[i]);
                    free(X); free(W); free(Y); free(Yref); return 1;
                }
            }
            free(X); free(W); free(Y); free(Yref);
        }
    }
    /* K=0 explicit. */
    {
        int M = 4, N = 4;
        int32_t Y[16];
        for (int i = 0; i < 16; i++) Y[i] = 0xCAFEBABE;
        m4t_mtfp4_sdot_matmul_bt_route(Y, NULL, NULL, M, 0, N);
        for (int i = 0; i < 16; i++) {
            if (Y[i] != 0) {
                printf("FAIL route K=0 expected 0, got %d at %d\n", Y[i], i);
                return 1;
            }
        }
    }
    /* Extreme X = ±40 (MTFP4 max magnitude). */
    {
        int K = 80, M = 2, N = 4;
        int8_t* X = malloc((size_t)M * K);
        int8_t* W = malloc((size_t)N * K);
        int32_t* Y = malloc((size_t)M * N * sizeof(int32_t));
        int32_t* Yref = malloc((size_t)M * N * sizeof(int32_t));
        for (int i = 0; i < M * K; i++) X[i] = (i & 1) ? 40 : -40;
        for (int i = 0; i < N * K; i++) W[i] = rand_trit();
        m4t_mtfp4_sdot_matmul_bt_route(Y, X, W, M, K, N);
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
            int64_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int64_t)X[i*K+k] * (int64_t)W[j*K+k];
            Yref[i*N+j] = (int32_t)acc;
        }
        for (int i = 0; i < M*N; i++) {
            if (Y[i] != Yref[i]) {
                printf("FAIL route extreme X=±40 cell %d: got=%d ref=%d\n",
                       i, Y[i], Yref[i]);
                free(X); free(W); free(Y); free(Yref); return 1;
            }
        }
        free(X); free(W); free(Y); free(Yref);
    }
    return 0;
}

int main(void) {
    if (test_clamp_basic())                return 1;
    if (test_sdot_matmul_small())          return 1;
    if (test_sdot_matmul_large())          return 1;
    if (test_sdot_matmul_high_mag())       return 1;
    if (test_sdot_matmul_long_k())         return 1;
    if (test_sdot_matmul_k16_sweep())      return 1;
    if (test_sdot_matmul_route_bit_exact()) return 1;
    if (test_sdot_zero_dim())              return 1;
    if (test_widen_exact())                return 1;
    if (test_narrow_round())               return 1;
    if (test_narrow_saturate())            return 1;
    if (test_narrow_flags())               return 1;
    if (test_narrow_property())            return 1;
    if (test_roundtrip_widen_narrow())     return 1;
    if (test_conversions_neon_vs_scalar_ref()) return 1;
    printf("m4t_mtfp4: all 15 tests passed\n");
    return 0;
}
