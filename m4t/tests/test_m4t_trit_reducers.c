/*
 * test_m4t_trit_reducers.c — tests for masked-VCNT ternary reductions.
 *
 * Golden values are hand-counted. No float.
 */

#include "m4t_types.h"
#include "m4t_trit_pack.h"
#include "m4t_trit_reducers.h"

#include <stdio.h>
#include <string.h>

#define ASSERT_EQ_I64(actual, expected, msg) do { \
    if ((actual) != (expected)) { \
        fprintf(stderr, "FAIL: %s — got %lld, expected %lld (line %d)\n", \
                (msg), (long long)(actual), (long long)(expected), __LINE__); \
        return 1; \
    } \
} while (0)

/* ── All zeros ─────────────────────────────────────────────────────────── */

static int test_all_zeros(void) {
    m4t_trit_t z[20];
    memset(z, 0, sizeof(z));
    uint8_t packed[M4T_TRIT_PACKED_BYTES(20)];
    m4t_pack_trits_1d(packed, z, 20);

    ASSERT_EQ_I64(m4t_trit_signed_sum(packed, 20), 0, "zeros signed_sum");
    ASSERT_EQ_I64(m4t_trit_sparsity(packed, 20), 0, "zeros sparsity");

    int64_t pos, neg;
    m4t_trit_counts(packed, 20, &pos, &neg);
    ASSERT_EQ_I64(pos, 0, "zeros pos");
    ASSERT_EQ_I64(neg, 0, "zeros neg");
    return 0;
}

/* ── All +1 ────────────────────────────────────────────────────────────── */

static int test_all_pos(void) {
    m4t_trit_t t[12];
    for (int i = 0; i < 12; i++) t[i] = 1;
    uint8_t packed[M4T_TRIT_PACKED_BYTES(12)];
    m4t_pack_trits_1d(packed, t, 12);

    ASSERT_EQ_I64(m4t_trit_signed_sum(packed, 12), 12, "all+1 signed_sum");
    ASSERT_EQ_I64(m4t_trit_sparsity(packed, 12), 12, "all+1 sparsity");

    int64_t pos, neg;
    m4t_trit_counts(packed, 12, &pos, &neg);
    ASSERT_EQ_I64(pos, 12, "all+1 pos");
    ASSERT_EQ_I64(neg, 0, "all+1 neg");
    return 0;
}

/* ── All -1 ────────────────────────────────────────────────────────────── */

static int test_all_neg(void) {
    m4t_trit_t t[12];
    for (int i = 0; i < 12; i++) t[i] = -1;
    uint8_t packed[M4T_TRIT_PACKED_BYTES(12)];
    m4t_pack_trits_1d(packed, t, 12);

    ASSERT_EQ_I64(m4t_trit_signed_sum(packed, 12), -12, "all-1 signed_sum");
    ASSERT_EQ_I64(m4t_trit_sparsity(packed, 12), 12, "all-1 sparsity");
    return 0;
}

/* ── Mixed pattern ─────────────────────────────────────────────────────── */

static int test_mixed(void) {
    /* 9 trits: [-1, 0, +1, -1, +1, +1, 0, -1, +1]
     * pos = 4 (+1 at indices 2,4,5,8)
     * neg = 3 (-1 at indices 0,3,7)
     * signed_sum = 4 - 3 = 1
     * sparsity = 4 + 3 = 7
     */
    m4t_trit_t t[9] = { -1, 0, 1, -1, 1, 1, 0, -1, 1 };
    uint8_t packed[M4T_TRIT_PACKED_BYTES(9)];
    m4t_pack_trits_1d(packed, t, 9);

    ASSERT_EQ_I64(m4t_trit_signed_sum(packed, 9), 1, "mixed signed_sum");
    ASSERT_EQ_I64(m4t_trit_sparsity(packed, 9), 7, "mixed sparsity");

    int64_t pos, neg;
    m4t_trit_counts(packed, 9, &pos, &neg);
    ASSERT_EQ_I64(pos, 4, "mixed pos");
    ASSERT_EQ_I64(neg, 3, "mixed neg");
    return 0;
}

/* ── NEON + tail boundary (65 trits) ───────────────────────────────────── */

static int test_neon_boundary(void) {
    /* 65 trits: cycling {+1, -1, 0}.
     * 65 / 3 = 21 full cycles + 2 extra.
     * pos:  indices 0, 3, 6, ..., 63 → every 3rd starting at 0 → 22 values
     * neg:  indices 1, 4, 7, ..., 64 → every 3rd starting at 1 → 22 values
     * zero: indices 2, 5, 8, ..., 62 → every 3rd starting at 2 → 21 values
     * signed_sum = 22 - 22 = 0
     * sparsity = 22 + 22 = 44
     */
    const int N = 65;
    m4t_trit_t t[65];
    static const m4t_trit_t cycle[3] = { 1, -1, 0 };
    for (int i = 0; i < N; i++) t[i] = cycle[i % 3];

    uint8_t packed[M4T_TRIT_PACKED_BYTES(65)];
    m4t_pack_trits_1d(packed, t, N);

    ASSERT_EQ_I64(m4t_trit_signed_sum(packed, N), 0, "n65 signed_sum");
    ASSERT_EQ_I64(m4t_trit_sparsity(packed, N), 44, "n65 sparsity");

    int64_t pos, neg;
    m4t_trit_counts(packed, N, &pos, &neg);
    ASSERT_EQ_I64(pos, 22, "n65 pos");
    ASSERT_EQ_I64(neg, 22, "n65 neg");
    return 0;
}

/* ── Large vector (256 trits = exact NEON, no tail) ────────────────────── */

static int test_256_exact(void) {
    /* 256 trits = 64 bytes = 4 NEON blocks, zero tail.
     * All +1 → signed_sum = 256, sparsity = 256. */
    const int N = 256;
    m4t_trit_t t[256];
    for (int i = 0; i < N; i++) t[i] = 1;

    uint8_t packed[M4T_TRIT_PACKED_BYTES(256)];
    m4t_pack_trits_1d(packed, t, N);

    ASSERT_EQ_I64(m4t_trit_signed_sum(packed, N), 256, "n256 signed_sum");
    ASSERT_EQ_I64(m4t_trit_sparsity(packed, N), 256, "n256 sparsity");
    return 0;
}

/* ── Large vector, all -1 (F4: exercises odd-bit NEON path) ────────────── */

static int test_256_all_neg(void) {
    const int N = 256;
    m4t_trit_t t[256];
    for (int i = 0; i < N; i++) t[i] = -1;

    uint8_t packed[M4T_TRIT_PACKED_BYTES(256)];
    m4t_pack_trits_1d(packed, t, N);

    ASSERT_EQ_I64(m4t_trit_signed_sum(packed, N), -256, "n256 neg signed_sum");
    ASSERT_EQ_I64(m4t_trit_sparsity(packed, N), 256, "n256 neg sparsity");

    int64_t pos, neg;
    m4t_trit_counts(packed, N, &pos, &neg);
    ASSERT_EQ_I64(pos, 0, "n256 neg pos");
    ASSERT_EQ_I64(neg, 256, "n256 neg neg");
    return 0;
}

/* ── Edge: n_trits = 0 ─────────────────────────────────────────────────── */

static int test_zero_length(void) {
    uint8_t dummy[1] = { 0xFF };
    ASSERT_EQ_I64(m4t_trit_signed_sum(dummy, 0), 0, "n0 signed_sum");
    ASSERT_EQ_I64(m4t_trit_sparsity(dummy, 0), 0, "n0 sparsity");
    return 0;
}

/* ── Edge: n_trits = 1 ─────────────────────────────────────────────────── */

static int test_single_trit(void) {
    m4t_trit_t t_pos[1] = { 1 };
    m4t_trit_t t_neg[1] = { -1 };
    m4t_trit_t t_zero[1] = { 0 };

    uint8_t pp[1], pn[1], pz[1];
    m4t_pack_trits_1d(pp, t_pos, 1);
    m4t_pack_trits_1d(pn, t_neg, 1);
    m4t_pack_trits_1d(pz, t_zero, 1);

    ASSERT_EQ_I64(m4t_trit_signed_sum(pp, 1), 1, "single +1");
    ASSERT_EQ_I64(m4t_trit_signed_sum(pn, 1), -1, "single -1");
    ASSERT_EQ_I64(m4t_trit_signed_sum(pz, 1), 0, "single 0");
    ASSERT_EQ_I64(m4t_trit_sparsity(pp, 1), 1, "single +1 sparsity");
    ASSERT_EQ_I64(m4t_trit_sparsity(pn, 1), 1, "single -1 sparsity");
    ASSERT_EQ_I64(m4t_trit_sparsity(pz, 1), 0, "single 0 sparsity");
    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    if (test_all_zeros())    return 1;
    if (test_all_pos())      return 1;
    if (test_all_neg())      return 1;
    if (test_mixed())        return 1;
    if (test_neon_boundary()) return 1;
    if (test_256_exact())    return 1;
    if (test_256_all_neg())  return 1;
    if (test_zero_length())  return 1;
    if (test_single_trit())  return 1;
    printf("m4t_trit_reducers: all tests passed\n");
    return 0;
}
