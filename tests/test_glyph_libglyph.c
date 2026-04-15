/*
 * test_glyph_libglyph.c — unit tests for libglyph primitive modules.
 *
 * Covers glyph_rng, glyph_bucket, glyph_multiprobe, and glyph_resolver.
 * The dataset loader, sig builder, and config parser are intentionally
 * excluded here: they either require real MNIST files on disk (dataset),
 * pull in the full ternary matmul pipeline (sig builder), or test
 * primarily through argv manipulation (config). Each can get its own
 * coverage file when it's worth the build cost.
 *
 * This suite aims for fast, deterministic, dependency-free tests that
 * catch the kinds of bugs red-teaming raised: OOB reads, ternary-Hamming
 * miscounts, RNG regressions, bucket lookup edge cases.
 */

#include "glyph_rng.h"
#include "glyph_bucket.h"
#include "glyph_multiprobe.h"
#include "glyph_resolver.h"
#include "m4t_trit_pack.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failed = 0;

#define TEST_FAIL(msg) do { \
    fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__); \
    g_failed++; \
    return; \
} while (0)

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) TEST_FAIL(msg); \
} while (0)

#define TEST_ASSERT_EQ(actual, expected, msg) do { \
    long _a = (long)(actual); \
    long _e = (long)(expected); \
    if (_a != _e) { \
        fprintf(stderr, "FAIL: %s — got %ld, expected %ld (line %d)\n", \
                (msg), _a, _e, __LINE__); \
        g_failed++; \
        return; \
    } \
} while (0)

/* ── glyph_rng ────────────────────────────────────────────────────── */

static void test_rng_determinism(void) {
    glyph_rng_t a, b;
    glyph_rng_seed(&a, 42, 123, 456, 789);
    glyph_rng_seed(&b, 42, 123, 456, 789);
    for (int i = 0; i < 1000; i++) {
        uint32_t va = glyph_rng_next(&a);
        uint32_t vb = glyph_rng_next(&b);
        TEST_ASSERT_EQ(va, vb, "rng determinism drift");
    }
}

static void test_rng_different_seeds_diverge(void) {
    glyph_rng_t a, b;
    glyph_rng_seed(&a, 42, 123, 456, 789);
    glyph_rng_seed(&b, 42, 123, 456, 790);  /* one bit different */
    int matches = 0;
    for (int i = 0; i < 64; i++) {
        uint32_t va = glyph_rng_next(&a);
        uint32_t vb = glyph_rng_next(&b);
        if (va == vb) matches++;
    }
    /* 64 draws from a good RNG should virtually never match between
     * different seeds. Tolerate zero matches; flag many. */
    TEST_ASSERT(matches < 4, "rng different seeds produced too many matches");
}

static void test_rng_uniform_mod3(void) {
    /* Coarse chi-square-ish check: 30000 draws mod 3 should be
     * roughly balanced. Tolerate a wide band. This catches gross
     * RNG breakage (e.g., always returning 0). */
    glyph_rng_t r;
    glyph_rng_seed(&r, 42, 123, 456, 789);
    int counts[3] = {0};
    const int N = 30000;
    for (int i = 0; i < N; i++) counts[glyph_rng_next(&r) % 3]++;
    for (int k = 0; k < 3; k++) {
        int expected = N / 3;
        int deviation = counts[k] - expected;
        if (deviation < 0) deviation = -deviation;
        TEST_ASSERT(deviation < 500, "rng mod 3 distribution skewed");
    }
}

/* ── glyph_bucket ─────────────────────────────────────────────────── */

/* Helper: write a 4-byte signature from a uint32 (little-endian, matches
 * glyph_sig_to_key_u32's decoder). */
static void sig_from_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)(v & 0xff);
    out[1] = (uint8_t)((v >> 8) & 0xff);
    out[2] = (uint8_t)((v >> 16) & 0xff);
    out[3] = (uint8_t)((v >> 24) & 0xff);
}

static void test_bucket_empty(void) {
    glyph_bucket_table_t bt = {0};
    TEST_ASSERT_EQ(glyph_bucket_build(&bt, NULL, 0, 4), 0, "build empty");
    TEST_ASSERT_EQ(bt.n_entries, 0, "empty n_entries");
    TEST_ASSERT_EQ(glyph_bucket_lower_bound(&bt, 0u), 0, "lower_bound empty");
    TEST_ASSERT_EQ(glyph_bucket_lower_bound(&bt, 0xffffffffu), 0, "lower_bound empty upper");
    TEST_ASSERT_EQ(glyph_bucket_count_distinct(&bt), 0, "distinct empty");
    glyph_bucket_table_free(&bt);
}

static void test_bucket_single_entry(void) {
    uint8_t sigs[4];
    sig_from_u32(sigs, 0x12345678u);
    glyph_bucket_table_t bt = {0};
    TEST_ASSERT_EQ(glyph_bucket_build(&bt, sigs, 1, 4), 0, "build single");
    TEST_ASSERT_EQ(bt.n_entries, 1, "single n_entries");
    TEST_ASSERT_EQ(bt.entries[0].key, 0x12345678u, "single key");
    TEST_ASSERT_EQ(bt.entries[0].proto_idx, 0, "single proto_idx");
    TEST_ASSERT_EQ(glyph_bucket_lower_bound(&bt, 0x12345678u), 0, "lb hit");
    TEST_ASSERT_EQ(glyph_bucket_lower_bound(&bt, 0u), 0, "lb below");
    TEST_ASSERT_EQ(glyph_bucket_lower_bound(&bt, 0xffffffffu), 1, "lb above");
    TEST_ASSERT_EQ(glyph_bucket_count_distinct(&bt), 1, "distinct single");
    glyph_bucket_table_free(&bt);
}

static void test_bucket_collisions_form_runs(void) {
    /* Four entries, two with the same key. After sorting, same-key
     * entries should be a contiguous run. */
    uint8_t sigs[16];
    sig_from_u32(sigs + 0,  0x00000100u);   /* proto 0 */
    sig_from_u32(sigs + 4,  0x00000200u);   /* proto 1 */
    sig_from_u32(sigs + 8,  0x00000100u);   /* proto 2 — same key as 0 */
    sig_from_u32(sigs + 12, 0x00000300u);   /* proto 3 */
    glyph_bucket_table_t bt = {0};
    TEST_ASSERT_EQ(glyph_bucket_build(&bt, sigs, 4, 4), 0, "build 4");
    TEST_ASSERT_EQ(bt.n_entries, 4, "n=4");
    /* The 0x100 bucket should be at position 0..1 (two proto_idx) */
    int lb = glyph_bucket_lower_bound(&bt, 0x100u);
    TEST_ASSERT(lb < bt.n_entries && bt.entries[lb].key == 0x100u, "lb on 0x100");
    TEST_ASSERT(bt.entries[lb + 1].key == 0x100u, "run continues");
    TEST_ASSERT(bt.entries[lb + 2].key != 0x100u, "run ends");
    /* Distinct bucket count is 3 (0x100, 0x200, 0x300). */
    TEST_ASSERT_EQ(glyph_bucket_count_distinct(&bt), 3, "distinct=3");
    glyph_bucket_table_free(&bt);
}

static void test_bucket_rejects_wrong_sig_bytes(void) {
    uint8_t sigs[8] = {0};
    glyph_bucket_table_t bt = {0};
    TEST_ASSERT(glyph_bucket_build(&bt, sigs, 1, 8) != 0, "reject 8-byte");
    TEST_ASSERT_EQ(bt.n_entries, 0, "n_entries unchanged");
    /* No need to free — build failure left entries NULL. */
}

static void test_bucket_lower_bound_gap(void) {
    /* Three entries with a gap; lower_bound on a missing key returns
     * the position where it would be inserted. */
    uint8_t sigs[12];
    sig_from_u32(sigs + 0,  10u);
    sig_from_u32(sigs + 4,  20u);
    sig_from_u32(sigs + 8,  30u);
    glyph_bucket_table_t bt = {0};
    TEST_ASSERT_EQ(glyph_bucket_build(&bt, sigs, 3, 4), 0, "build gap");
    TEST_ASSERT_EQ(glyph_bucket_lower_bound(&bt, 15u), 1, "lb in gap");
    TEST_ASSERT_EQ(glyph_bucket_lower_bound(&bt, 25u), 2, "lb in gap 2");
    TEST_ASSERT_EQ(glyph_bucket_lower_bound(&bt, 31u), 3, "lb past end");
    glyph_bucket_table_free(&bt);
}

static void test_sig_to_key_u32_endianness(void) {
    /* A 4-byte signature 0xde 0xad 0xbe 0xef should decode as
     * little-endian 0xefbeadde. Matches the internal convention. */
    uint8_t sig[4] = {0xde, 0xad, 0xbe, 0xef};
    TEST_ASSERT_EQ(glyph_sig_to_key_u32(sig), 0xefbeaddeu, "LE decode");
}

/* ── glyph_multiprobe ─────────────────────────────────────────────── */

/* Helper to build a 16-trit signature from an array of trits. */
static void pack_trits_16(uint8_t sig[4], const int8_t trits[16]) {
    for (int i = 0; i < 4; i++) sig[i] = 0;
    for (int i = 0; i < 16; i++) {
        glyph_write_trit(sig, i, trits[i]);
    }
}

static void test_read_write_trit_roundtrip(void) {
    uint8_t sig[4] = {0};
    int8_t values[16] = {0,1,-1,0, 1,-1,0,1, -1,0,1,-1, 0,1,-1,0};
    for (int i = 0; i < 16; i++) glyph_write_trit(sig, i, values[i]);
    for (int i = 0; i < 16; i++) {
        TEST_ASSERT_EQ(glyph_read_trit(sig, i), values[i], "roundtrip");
    }
}

static void test_write_trit_does_not_disturb_neighbors(void) {
    /* Writing trit j should not change any other trit. */
    uint8_t sig[4] = {0};
    for (int i = 0; i < 16; i++) glyph_write_trit(sig, i, +1);
    glyph_write_trit(sig, 7, -1);
    for (int i = 0; i < 16; i++) {
        int expected = (i == 7) ? -1 : 1;
        TEST_ASSERT_EQ(glyph_read_trit(sig, i), expected, "non-neighbor preserved");
    }
}

/* Multi-probe callback context: counts probes and their ternary
 * Hamming distance from the reference signature. */
typedef struct {
    const uint8_t* reference;
    int n_probes;
    int total_cost;
    int max_cost_seen;
    int min_cost_seen;
} probe_counter_ctx_t;

static int ternary_hamming(const uint8_t* a, const uint8_t* b, int n_trits) {
    int cost = 0;
    for (int i = 0; i < n_trits; i++) {
        int8_t ta = glyph_read_trit(a, i);
        int8_t tb = glyph_read_trit(b, i);
        if (ta == tb) continue;
        if (ta == 0 || tb == 0) cost += 1;        /* zero vs ±1 */
        else                    cost += 2;        /* +1 vs -1 */
    }
    return cost;
}

static int count_probe_cb(const uint8_t* probe, void* vctx) {
    probe_counter_ctx_t* c = (probe_counter_ctx_t*)vctx;
    int cost = ternary_hamming(c->reference, probe, 16);
    c->n_probes++;
    c->total_cost += cost;
    if (cost > c->max_cost_seen) c->max_cost_seen = cost;
    if (cost < c->min_cost_seen) c->min_cost_seen = cost;
    return 0;    /* continue enumeration */
}

static void test_multiprobe_radius_0(void) {
    int8_t query_trits[16] = {0,1,-1,0, 1,-1,0,1, -1,0,1,-1, 0,1,-1,0};
    uint8_t query_sig[4];
    pack_trits_16(query_sig, query_trits);
    uint8_t scratch[4];

    probe_counter_ctx_t ctx = {query_sig, 0, 0, -1, 999};
    glyph_multiprobe_enumerate(query_sig, 16, 4, 0, scratch, count_probe_cb, &ctx);
    TEST_ASSERT_EQ(ctx.n_probes, 1, "r=0 produces 1 probe");
    TEST_ASSERT_EQ(ctx.total_cost, 0, "r=0 cost 0");
}

static void test_multiprobe_radius_1_all_zeros(void) {
    /* All-zero query. Every radius-1 neighbor changes exactly one trit
     * from 0 to ±1 (cost 1). 16 positions × 2 outcomes = 32 probes. */
    int8_t query_trits[16] = {0};
    uint8_t query_sig[4];
    pack_trits_16(query_sig, query_trits);
    uint8_t scratch[4];

    probe_counter_ctx_t ctx = {query_sig, 0, 0, -1, 999};
    glyph_multiprobe_enumerate(query_sig, 16, 4, 1, scratch, count_probe_cb, &ctx);
    TEST_ASSERT_EQ(ctx.n_probes, 32, "r=1 all-zero produces 32 probes");
    TEST_ASSERT_EQ(ctx.total_cost, 32, "r=1 all-zero total cost 32");
    TEST_ASSERT_EQ(ctx.min_cost_seen, 1, "r=1 min cost is 1");
    TEST_ASSERT_EQ(ctx.max_cost_seen, 1, "r=1 max cost is 1");
}

static void test_multiprobe_radius_1_all_plus(void) {
    /* Query with every trit at +1. Each position's only cost-1 move is
     * +1 → 0 (the sign flip +1 → -1 is cost 2). 16 probes total. */
    int8_t query_trits[16];
    for (int i = 0; i < 16; i++) query_trits[i] = 1;
    uint8_t query_sig[4];
    pack_trits_16(query_sig, query_trits);
    uint8_t scratch[4];

    probe_counter_ctx_t ctx = {query_sig, 0, 0, -1, 999};
    glyph_multiprobe_enumerate(query_sig, 16, 4, 1, scratch, count_probe_cb, &ctx);
    TEST_ASSERT_EQ(ctx.n_probes, 16, "r=1 all-plus produces 16 probes");
    TEST_ASSERT_EQ(ctx.total_cost, 16, "r=1 all-plus total cost 16");
    TEST_ASSERT_EQ(ctx.max_cost_seen, 1, "r=1 max cost is 1");
}

static void test_multiprobe_radius_2_all_zeros(void) {
    /* All-zero query. Radius-2 enumeration has no sign flips (no
     * non-zero positions), only the (b) branch with two cost-1 moves
     * on distinct positions. C(16,2) pairs × 4 combinations each
     * (each position has 2 cost-1 targets) = 120 × 4 = 480 probes. */
    int8_t query_trits[16] = {0};
    uint8_t query_sig[4];
    pack_trits_16(query_sig, query_trits);
    uint8_t scratch[4];

    probe_counter_ctx_t ctx = {query_sig, 0, 0, -1, 999};
    glyph_multiprobe_enumerate(query_sig, 16, 4, 2, scratch, count_probe_cb, &ctx);
    TEST_ASSERT_EQ(ctx.n_probes, 480, "r=2 all-zero produces 480 probes");
    TEST_ASSERT_EQ(ctx.min_cost_seen, 2, "r=2 min cost is 2");
    TEST_ASSERT_EQ(ctx.max_cost_seen, 2, "r=2 max cost is 2");
}

static void test_multiprobe_radius_2_single_plus(void) {
    /* Query with exactly one +1 trit at position 0, rest zero.
     *
     * Radius-2 (a) sign-flip branch: one position (position 0) is
     * non-zero, so 1 sign-flip probe (+1 → -1 at position 0).
     *
     * Radius-2 (b) two-cost-1-moves branch: pairs (j,k), j<k.
     *   - pairs not involving position 0: both positions are zero,
     *     so each has 2 cost-1 targets. C(15,2) = 105 pairs × 4 = 420.
     *   - pairs involving position 0: position 0 is +1 so only 1
     *     cost-1 target (→0); the other position is zero so 2 targets.
     *     15 pairs × (1 × 2) = 30.
     *
     * Total: 1 (sign) + 420 + 30 = 451 probes. */
    int8_t query_trits[16] = {0};
    query_trits[0] = 1;
    uint8_t query_sig[4];
    pack_trits_16(query_sig, query_trits);
    uint8_t scratch[4];

    probe_counter_ctx_t ctx = {query_sig, 0, 0, -1, 999};
    glyph_multiprobe_enumerate(query_sig, 16, 4, 2, scratch, count_probe_cb, &ctx);
    TEST_ASSERT_EQ(ctx.n_probes, 1 + 420 + 30, "r=2 single-plus probe count");
    TEST_ASSERT(ctx.max_cost_seen == 2, "r=2 max cost is 2");
    TEST_ASSERT(ctx.min_cost_seen == 2, "r=2 min cost is 2");
}

/* Early-stop: callback returns 1 on the first probe. The enumerator
 * should return without invoking cb again. */
static int early_stop_cb(const uint8_t* probe, void* vctx) {
    (void)probe;
    int* count = (int*)vctx;
    (*count)++;
    return 1;
}

static void test_multiprobe_early_stop(void) {
    int8_t query_trits[16] = {0,1,-1,0, 1,-1,0,1, -1,0,1,-1, 0,1,-1,0};
    uint8_t query_sig[4];
    pack_trits_16(query_sig, query_trits);
    uint8_t scratch[4];

    int count = 0;
    glyph_multiprobe_enumerate(query_sig, 16, 4, 1, scratch, early_stop_cb, &count);
    TEST_ASSERT_EQ(count, 1, "early stop at first probe");

    count = 0;
    glyph_multiprobe_enumerate(query_sig, 16, 4, 2, scratch, early_stop_cb, &count);
    TEST_ASSERT_EQ(count, 1, "early stop at first probe (r=2)");
}

/* ── glyph_resolver ───────────────────────────────────────────────── */

static void test_resolver_vote_simple_majority(void) {
    /* Two candidates in the union: proto 0 label 3 with 1 vote, proto
     * 1 label 7 with 2 votes. VOTE picks class 7. */
    const int n_train = 2;
    int32_t hit_list[] = {0, 1};
    uint16_t votes[2] = {1, 2};
    int y_train[] = {3, 7};
    glyph_union_t u = {hit_list, 2, votes, y_train, 10};
    TEST_ASSERT_EQ(glyph_resolver_vote(&u), 7, "vote picks majority");
    (void)n_train;
}

static void test_resolver_vote_tiebreaks_to_lower_class(void) {
    /* Tie at equal votes — argmax tiebreaks to the first class scanned,
     * which is the lower class index. */
    int32_t hit_list[] = {0, 1};
    uint16_t votes[2] = {1, 1};
    int y_train[] = {2, 5};
    glyph_union_t u = {hit_list, 2, votes, y_train, 10};
    TEST_ASSERT_EQ(glyph_resolver_vote(&u), 2, "vote tiebreak to lower class");
}

static void test_resolver_vote_weighted(void) {
    /* Weighted vote — multiple prototypes per class, vote counts
     * sum per class. Class 0 has 3+2=5, class 1 has 4. Class 0 wins. */
    int32_t hit_list[] = {0, 1, 2};
    uint16_t votes[3] = {3, 2, 4};
    int y_train[] = {0, 0, 1};
    glyph_union_t u = {hit_list, 3, votes, y_train, 10};
    TEST_ASSERT_EQ(glyph_resolver_vote(&u), 0, "vote class-weighted sum");
}

static void test_resolver_sum_one_table(void) {
    /* One table, two candidates. SUM resolver = single-table 1-NN
     * within the union. popcount_dist picks the closer candidate. */
    const int n_train = 2;
    const int sig_bytes = 4;
    uint8_t train_sig[2 * 4];
    uint8_t query_sig[4];
    /* Candidate 0: all zero trits (Hamming 0 from all-zero query). */
    memset(train_sig + 0, 0, 4);
    /* Candidate 1: trit 0 = +1, rest zero. Hamming 1 from all-zero query. */
    memset(train_sig + 4, 0, 4);
    glyph_write_trit(train_sig + 4, 0, +1);
    memset(query_sig, 0, 4);

    int32_t hit_list[] = {0, 1};
    uint16_t votes[2] = {1, 1};   /* both in the union */
    int y_train[] = {5, 9};
    glyph_union_t u = {hit_list, 2, votes, y_train, 10};

    uint8_t* table_sigs[1];
    const uint8_t* q_sigs[1];
    table_sigs[0] = train_sig;
    q_sigs[0] = query_sig;

    uint8_t mask[4];
    memset(mask, 0xff, 4);

    int pred = glyph_resolver_sum(&u, 1, sig_bytes, table_sigs, q_sigs, mask);
    TEST_ASSERT_EQ(pred, 5, "sum picks closer candidate");
    (void)n_train;
}

/* ── driver ───────────────────────────────────────────────────────── */

int main(void) {
    test_rng_determinism();
    test_rng_different_seeds_diverge();
    test_rng_uniform_mod3();

    test_bucket_empty();
    test_bucket_single_entry();
    test_bucket_collisions_form_runs();
    test_bucket_rejects_wrong_sig_bytes();
    test_bucket_lower_bound_gap();
    test_sig_to_key_u32_endianness();

    test_read_write_trit_roundtrip();
    test_write_trit_does_not_disturb_neighbors();
    test_multiprobe_radius_0();
    test_multiprobe_radius_1_all_zeros();
    test_multiprobe_radius_1_all_plus();
    test_multiprobe_radius_2_all_zeros();
    test_multiprobe_radius_2_single_plus();
    test_multiprobe_early_stop();

    test_resolver_vote_simple_majority();
    test_resolver_vote_tiebreaks_to_lower_class();
    test_resolver_vote_weighted();
    test_resolver_sum_one_table();

    if (g_failed > 0) {
        fprintf(stderr, "test_glyph_libglyph: %d FAILURES\n", g_failed);
        return 1;
    }
    fprintf(stderr, "test_glyph_libglyph: all tests passed\n");
    return 0;
}
