/*
 * test_m4t_trit_pack.c — direct unit tests for m4t_trit_pack primitives.
 *
 * Covers m4t_pack_trits_1d, m4t_unpack_trits_1d, and in particular
 * m4t_popcount_dist across every packed_bytes regime after the
 * multi-tier optimization (NEON 16B, scalar 8B, scalar 4B, byte tail).
 *
 * The popcount_dist tests pin the implementation against a reference
 * scalar oracle computed independently in-test — so future refactors
 * of the kernel (wider SIMD, different chunk sizes, different
 * compiler builtins) will fail loudly if they diverge bit-for-bit
 * from the reference at any packed_bytes length.
 */

#include "m4t_trit_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__); \
        g_failed++; \
        return; \
    } \
} while (0)

#define CHECK_EQ(actual, expected, msg) do { \
    long _a = (long)(actual); long _e = (long)(expected); \
    if (_a != _e) { \
        fprintf(stderr, "FAIL: %s — got %ld, expected %ld (line %d)\n", \
                (msg), _a, _e, __LINE__); \
        g_failed++; \
        return; \
    } \
} while (0)

/* ── Reference implementation: slowest-possible scalar popcount_dist
 *    used purely as an oracle against the optimized kernel. Kept byte-
 *    by-byte with Kernighan so no SIMD path can sneak in via the
 *    compiler's autovectorizer. */
static int32_t ref_popcount_dist(
    const uint8_t* a, const uint8_t* b, const uint8_t* mask, int n)
{
    int32_t total = 0;
    for (int i = 0; i < n; i++) {
        volatile uint8_t x = (uint8_t)((a[i] ^ b[i]) & mask[i]);
        while (x) {
            total++;
            x = (uint8_t)(x & (uint8_t)(x - 1u));
        }
    }
    return total;
}

/* ── pack/unpack roundtrip ────────────────────────────────────────── */

static void test_pack_unpack_roundtrip_various_n(void) {
    const int Ns[] = {1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128};
    for (size_t idx = 0; idx < sizeof(Ns)/sizeof(Ns[0]); idx++) {
        int n = Ns[idx];
        m4t_trit_t* src = malloc((size_t)n * sizeof(m4t_trit_t));
        m4t_trit_t* dst = malloc((size_t)n * sizeof(m4t_trit_t));
        uint8_t*    pk  = malloc((size_t)M4T_TRIT_PACKED_BYTES(n));
        for (int i = 0; i < n; i++) src[i] = (m4t_trit_t)((i % 3) - 1);
        m4t_pack_trits_1d(pk, src, n);
        m4t_unpack_trits_1d(dst, pk, n);
        for (int i = 0; i < n; i++) {
            if (dst[i] != src[i]) {
                fprintf(stderr, "FAIL: pack/unpack mismatch at n=%d i=%d\n", n, i);
                g_failed++;
                free(src); free(dst); free(pk);
                return;
            }
        }
        free(src); free(dst); free(pk);
    }
}

/* ── popcount_dist edge cases ─────────────────────────────────────── */

static void test_popcount_dist_empty(void) {
    uint8_t a[1] = {0}, b[1] = {0}, m[1] = {0xff};
    CHECK_EQ(m4t_popcount_dist(a, b, m, 0), 0, "empty packed_bytes");
}

static void test_popcount_dist_identical_inputs(void) {
    uint8_t a[64]; for (int i = 0; i < 64; i++) a[i] = (uint8_t)(i * 17 + 3);
    uint8_t m[64]; memset(m, 0xff, 64);
    for (int n = 1; n <= 64; n++) {
        int32_t got = m4t_popcount_dist(a, a, m, n);
        if (got != 0) {
            fprintf(stderr, "FAIL: identical inputs at n=%d got %d\n", n, got);
            g_failed++;
            return;
        }
    }
}

static void test_popcount_dist_known_values(void) {
    /* 1 byte, single differing bit. */
    {
        uint8_t a = 0x00, b = 0x01, m = 0xff;
        CHECK_EQ(m4t_popcount_dist(&a, &b, &m, 1), 1, "1 byte 1 diff bit");
    }
    /* 4 bytes, all bits flipped. */
    {
        uint8_t a[4] = {0x00,0x00,0x00,0x00};
        uint8_t b[4] = {0xff,0xff,0xff,0xff};
        uint8_t m[4] = {0xff,0xff,0xff,0xff};
        CHECK_EQ(m4t_popcount_dist(a, b, m, 4), 32, "4 bytes all bits flipped");
    }
    /* 4 bytes, mask zeros out half. */
    {
        uint8_t a[4] = {0x00,0x00,0x00,0x00};
        uint8_t b[4] = {0xff,0xff,0xff,0xff};
        uint8_t m[4] = {0x0f,0x0f,0x0f,0x0f};
        CHECK_EQ(m4t_popcount_dist(a, b, m, 4), 16, "4 bytes half masked");
    }
    /* 8 bytes, alternating. */
    {
        uint8_t a[8] = {0xaa,0xaa,0xaa,0xaa,0xaa,0xaa,0xaa,0xaa};
        uint8_t b[8] = {0x55,0x55,0x55,0x55,0x55,0x55,0x55,0x55};
        uint8_t m[8]; memset(m, 0xff, 8);
        CHECK_EQ(m4t_popcount_dist(a, b, m, 8), 64, "8 bytes alternating all diff");
    }
    /* 16 bytes (exercises NEON path), all flipped. */
    {
        uint8_t a[16], b[16], m[16];
        memset(a, 0x00, 16); memset(b, 0xff, 16); memset(m, 0xff, 16);
        CHECK_EQ(m4t_popcount_dist(a, b, m, 16), 128, "16 bytes NEON path");
    }
    /* 17 bytes (NEON + byte tail). */
    {
        uint8_t a[17], b[17], m[17];
        memset(a, 0x00, 17); memset(b, 0xff, 17); memset(m, 0xff, 17);
        CHECK_EQ(m4t_popcount_dist(a, b, m, 17), 17*8, "17 bytes NEON+tail");
    }
    /* 20 bytes (NEON + 4-byte path). */
    {
        uint8_t a[20], b[20], m[20];
        memset(a, 0x00, 20); memset(b, 0xff, 20); memset(m, 0xff, 20);
        CHECK_EQ(m4t_popcount_dist(a, b, m, 20), 20*8, "20 bytes NEON+4byte");
    }
    /* 24 bytes (NEON + 8-byte path). */
    {
        uint8_t a[24], b[24], m[24];
        memset(a, 0x00, 24); memset(b, 0xff, 24); memset(m, 0xff, 24);
        CHECK_EQ(m4t_popcount_dist(a, b, m, 24), 24*8, "24 bytes NEON+8byte");
    }
    /* 32 bytes (2x NEON loop). */
    {
        uint8_t a[32], b[32], m[32];
        memset(a, 0x00, 32); memset(b, 0xff, 32); memset(m, 0xff, 32);
        CHECK_EQ(m4t_popcount_dist(a, b, m, 32), 32*8, "32 bytes 2x NEON");
    }
}

/* ── bit-exact equivalence against the reference on random inputs ── */

static uint32_t xorshift_state = 0x12345678u;
static uint32_t rnd(void) {
    uint32_t x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

static void test_popcount_dist_random_equivalence(void) {
    /* Cover every packed_bytes regime that hits each tier of the
     * implementation: 0, 1-3 (tail only), 4, 5-7 (4+tail), 8, 9-11,
     * 12 (8+4), 13-15 (8+4+tail), 16 (NEON only), 17, 20, 23, 24, 32, 64. */
    const int sizes[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 19, 20, 23, 24, 31, 32, 48, 63, 64, 127, 128
    };
    uint8_t a[256], b[256], m[256];
    for (int trial = 0; trial < 200; trial++) {
        for (int i = 0; i < 256; i++) {
            uint32_t r = rnd();
            a[i] = (uint8_t)(r & 0xff);
            b[i] = (uint8_t)((r >> 8) & 0xff);
            m[i] = (uint8_t)((r >> 16) & 0xff);
        }
        for (size_t idx = 0; idx < sizeof(sizes)/sizeof(sizes[0]); idx++) {
            int n = sizes[idx];
            int32_t got = m4t_popcount_dist(a, b, m, n);
            int32_t ref = ref_popcount_dist(a, b, m, n);
            if (got != ref) {
                fprintf(stderr,
                    "FAIL: random trial %d n=%d got %d expected %d\n",
                    trial, n, got, ref);
                g_failed++;
                return;
            }
        }
    }
}

static void test_popcount_dist_mask_all_zero(void) {
    /* All-zero mask: output must be 0 regardless of a, b. Exercises
     * the fast paths' AND-with-mask correctness. */
    uint8_t a[64], b[64], m[64];
    for (int i = 0; i < 64; i++) {
        a[i] = (uint8_t)(i * 31 + 7);
        b[i] = (uint8_t)(i * 13 + 5);
    }
    memset(m, 0x00, 64);
    for (int n = 0; n <= 64; n++) {
        int32_t got = m4t_popcount_dist(a, b, m, n);
        if (got != 0) {
            fprintf(stderr, "FAIL: all-zero mask n=%d got %d expected 0\n", n, got);
            g_failed++;
            return;
        }
    }
}

static void test_popcount_dist_unaligned_offset(void) {
    /* The memcpy-based loads should work at any byte alignment.
     * Allocate a buffer + 1 so the inputs start at an odd offset. */
    uint8_t buf_a[33], buf_b[33], buf_m[33];
    for (int i = 0; i < 33; i++) {
        buf_a[i] = (uint8_t)(i * 11);
        buf_b[i] = (uint8_t)(i * 7 + 1);
        buf_m[i] = 0xff;
    }
    for (int off = 0; off < 3; off++) {
        for (int n = 0; n + off <= 32; n++) {
            int32_t got = m4t_popcount_dist(buf_a + off, buf_b + off, buf_m + off, n);
            int32_t ref = ref_popcount_dist(buf_a + off, buf_b + off, buf_m + off, n);
            if (got != ref) {
                fprintf(stderr,
                    "FAIL: unaligned off=%d n=%d got %d expected %d\n",
                    off, n, got, ref);
                g_failed++;
                return;
            }
        }
    }
}

int main(void) {
    test_pack_unpack_roundtrip_various_n();

    test_popcount_dist_empty();
    test_popcount_dist_identical_inputs();
    test_popcount_dist_known_values();
    test_popcount_dist_random_equivalence();
    test_popcount_dist_mask_all_zero();
    test_popcount_dist_unaligned_offset();

    if (g_failed > 0) {
        fprintf(stderr, "test_m4t_trit_pack: %d FAILURES\n", g_failed);
        return 1;
    }
    fprintf(stderr, "test_m4t_trit_pack: all tests passed\n");
    return 0;
}
