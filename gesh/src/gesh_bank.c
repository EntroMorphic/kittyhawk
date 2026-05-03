/*
 * gesh_bank.c — class-conditional ternary mean bank construction.
 *
 * Pure integer arithmetic. Per-class per-dim sum across samples,
 * sign-thresholded to ternary, packed into bank tiles.
 */

#include "gesh_bank.h"
#include "gesh_project.h"
#include "m4t_route.h"
#include "m4t_trit_pack.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void gesh_bank_build_class_mean(
    gesh_bank_t* bank,
    const m4t_trit_t* samples,
    const int* labels,
    int n_samples,
    int n_classes)
{
    assert(bank && samples && labels);
    assert(n_samples >= 0 && n_classes > 0);
    assert(bank->n_tiles == n_classes);
    assert(bank->sig_dim > 0);
    assert(bank->tiles_packed && bank->labels);

    int D = bank->sig_dim;
    int Dp = M4T_TRIT_PACKED_BYTES(D);

    /* Per-class per-dim sum. int32 is enough for typical n_samples
     * (max |sum| = n_samples). Sign-thresholding doesn't need
     * per-class counts, so we don't track them — sum > 0 ⇒ +1, etc.,
     * regardless of how many samples contributed. */
    int32_t* class_sums = calloc((size_t)n_classes * (size_t)D, sizeof(int32_t));

    for (int i = 0; i < n_samples; i++) {
        int c = labels[i];
        assert(c >= 0 && c < n_classes);
        const m4t_trit_t* s = samples + (size_t)i * D;
        int32_t* row = class_sums + (size_t)c * D;
        for (int j = 0; j < D; j++) {
            row[j] += (int32_t)s[j];
        }
    }

    /* Sign-threshold the per-class sums via the kernel-routed wrapper
     * (gesh_threshold_int32_to_trit → m4t_route_threshold_extract).
     * Substrate-discipline: no open-coded sign threshold. Zero sum
     * (or no samples for class) → zero trit, per the kernel's tau=0
     * sanctioned input class for integer inputs. */
    m4t_trit_t* tile_unpacked = malloc((size_t)D * sizeof(m4t_trit_t));
    for (int c = 0; c < n_classes; c++) {
        const int32_t* row = class_sums + (size_t)c * D;
        gesh_threshold_int32_to_trit(tile_unpacked, row, D);
        m4t_pack_trits_1d(
            bank->tiles_packed + (size_t)c * Dp,
            tile_unpacked, D);
        bank->labels[c] = c;
    }

    free(tile_unpacked);
    free(class_sums);
}

/* ── Wildcard bank constructor ──────────────────────────────────────────
 *
 * Per-class signed sum + per-class sample count + signal-to-noise
 * threshold. Positions below threshold receive deliberate zero
 * (§19 (II) Wildcard); above, standard sign.
 *
 * The "signal" is |sum| / count = mean magnitude. This is a coarse
 * SNR proxy — for ternary samples whose values are ±1 or 0, the per-dim
 * mean magnitude IS the within-class consistency on that dim. A class
 * whose samples uniformly assert +1 on dim d gives signal=1.0 there;
 * a class whose samples are evenly split on dim d gives signal≈0.
 * Permille scale: signal_pm = |sum| × 1000 / count, range [0, 1000]. */
void gesh_bank_build_class_wildcard(
    gesh_bank_t* bank,
    const m4t_trit_t* samples,
    const int* labels,
    int n_samples,
    int n_classes,
    int snr_threshold_permille)
{
    assert(bank && samples && labels);
    assert(n_samples >= 0 && n_classes > 0);
    assert(bank->n_tiles == n_classes);
    assert(bank->sig_dim > 0);
    assert(bank->tiles_packed && bank->labels);
    assert(snr_threshold_permille >= 0);

    int D = bank->sig_dim;
    int Dp = M4T_TRIT_PACKED_BYTES(D);

    int32_t* class_sums   = calloc((size_t)n_classes * (size_t)D, sizeof(int32_t));
    int32_t* class_counts = calloc((size_t)n_classes,             sizeof(int32_t));
    assert(class_sums && class_counts);

    /* Accumulate sums and counts per class. */
    for (int i = 0; i < n_samples; i++) {
        int c = labels[i];
        assert(c >= 0 && c < n_classes);
        const m4t_trit_t* s = samples + (size_t)i * D;
        int32_t* row = class_sums + (size_t)c * D;
        for (int j = 0; j < D; j++) {
            row[j] += (int32_t)s[j];
        }
        class_counts[c]++;
    }

    m4t_trit_t* tile_unpacked = malloc((size_t)D * sizeof(m4t_trit_t));
    assert(tile_unpacked);

    for (int c = 0; c < n_classes; c++) {
        const int32_t* row = class_sums + (size_t)c * D;
        int32_t cnt = class_counts[c];

        for (int j = 0; j < D; j++) {
            int32_t v = row[j];
            int32_t abs_v = (v >= 0) ? v : -v;
            int32_t signal_pm;
            if (cnt > 0) {
                /* signal_pm = |sum| × 1000 / count; integer arithmetic. */
                signal_pm = (int32_t)((int64_t)abs_v * 1000 / cnt);
            } else {
                signal_pm = 0;
            }

            if (signal_pm < snr_threshold_permille) {
                /* DELIBERATE wildcard. (II) Wildcard interpretation. */
                tile_unpacked[j] = 0;
            } else {
                tile_unpacked[j] = (v > 0) ?  1
                                 : (v < 0) ? -1
                                 :            0;
            }
        }

        m4t_pack_trits_1d(
            bank->tiles_packed + (size_t)c * Dp,
            tile_unpacked, D);
        bank->labels[c] = c;
    }

    free(tile_unpacked);
    free(class_counts);
    free(class_sums);
}

/* ── Confidence-aware bank constructor (P0-2) ─────────────────────────── */
void gesh_bank_build_class_mean_with_confidence(
    gesh_bank_t* bank,
    uint8_t* conf_bits,
    const m4t_trit_t* samples,
    const int* labels,
    int n_samples,
    int n_classes,
    int tau_strong_permille)
{
    assert(bank && conf_bits && samples && labels);
    assert(bank->n_tiles == n_classes);
    int D = bank->sig_dim;
    int Dp = M4T_TRIT_PACKED_BYTES(D);
    int conf_bytes_per_class = (D + 7) / 8;

    int32_t* sums = calloc((size_t)n_classes * D, sizeof(int32_t));
    int32_t* counts = calloc((size_t)n_classes, sizeof(int32_t));
    for (int i = 0; i < n_samples; i++) {
        int c = labels[i];
        int32_t* row = sums + (size_t)c * D;
        const m4t_trit_t* s = samples + (size_t)i * D;
        for (int j = 0; j < D; j++) row[j] += (int32_t)s[j];
        counts[c]++;
    }

    m4t_trit_t* tile = malloc((size_t)D * sizeof(m4t_trit_t));
    memset(conf_bits, 0, (size_t)n_classes * (size_t)conf_bytes_per_class);

    for (int c = 0; c < n_classes; c++) {
        const int32_t* row = sums + (size_t)c * D;
        int32_t cnt = counts[c];
        uint8_t* conf_row = conf_bits + (size_t)c * (size_t)conf_bytes_per_class;
        for (int j = 0; j < D; j++) {
            int32_t v = row[j];
            int32_t abs_v = (v >= 0) ? v : -v;
            tile[j] = (v > 0) ? 1 : (v < 0) ? -1 : 0;
            int32_t signal_pm = (cnt > 0)
                ? (int32_t)((int64_t)abs_v * 1000 / cnt) : 0;
            if (signal_pm > tau_strong_permille) {
                conf_row[j >> 3] |= (uint8_t)(1u << (j & 7));
            }
        }
        m4t_pack_trits_1d(bank->tiles_packed + (size_t)c * Dp, tile, D);
        bank->labels[c] = c;
    }
    free(tile); free(counts); free(sums);
}

/* ── k-means per-class bank constructor ──────────────────────────────────
 *
 * Multi-prototype bank: k > 1 tiles per class. See header for full
 * algorithm description. Substrate-legal: integer arithmetic, ternary
 * outputs, deterministic stratified init (no random weights).
 *
 * Performance note: this routine still uses per-iteration scalar
 * orchestration (assignment loop, sum update, comparison). The
 * substrate purification task is queued to lift these to libm4t
 * kernels. Numerical results are bit-identical pre/post purification
 * because the operations are deterministic integer arithmetic. */
void gesh_bank_build_kmeans_per_class(
    gesh_bank_t* bank,
    const m4t_trit_t* samples,
    const int* labels,
    int n_samples,
    int n_classes,
    int k_per_class,
    int max_iters,
    uint32_t seed)
{
    (void)seed;  /* reserved for future tie-breaking variants */

    assert(bank && samples && labels);
    assert(n_samples >= 0 && n_classes > 0);
    assert(k_per_class >= 1 && max_iters >= 1);
    assert(bank->n_tiles == k_per_class * n_classes);
    assert(bank->sig_dim > 0);
    assert(bank->tiles_packed && bank->labels);

    int D = bank->sig_dim;
    int Dp = M4T_TRIT_PACKED_BYTES(D);

    /* Mask: all-ones for valid trits, tail bits zeroed. Same construction
     * as gesh_forward_classify uses for popcount_dist. */
    uint8_t* mask = malloc((size_t)Dp);
    memset(mask, 0xFF, (size_t)Dp);
    int tail_trits = D & 3;
    if (tail_trits > 0) {
        uint8_t tail_mask = (uint8_t)((1u << (tail_trits * 2)) - 1u);
        mask[Dp - 1] = tail_mask;
    }

    /* Pre-pack all samples once (used by Hamming distance per iteration). */
    uint8_t* samples_packed = malloc((size_t)n_samples * (size_t)Dp);
    for (int i = 0; i < n_samples; i++) {
        m4t_pack_trits_1d(samples_packed + (size_t)i * Dp,
                          samples + (size_t)i * D, D);
    }

    /* Per-class scratch buffers (sized for the largest class; reuse). */
    int* class_idx = malloc((size_t)n_samples * sizeof(int));
    int* assignments = malloc((size_t)n_samples * sizeof(int));
    int32_t* sums = malloc((size_t)k_per_class * (size_t)D * sizeof(int32_t));
    int* counts = malloc((size_t)k_per_class * sizeof(int));
    m4t_trit_t* centroids_unpacked = malloc((size_t)k_per_class * (size_t)D
                                              * sizeof(m4t_trit_t));
    uint8_t* centroids_packed = malloc((size_t)k_per_class * (size_t)Dp);
    m4t_trit_t* tile_buf = malloc((size_t)D * sizeof(m4t_trit_t));

    for (int c = 0; c < n_classes; c++) {
        /* Gather indices of class-c samples. */
        int n_c = 0;
        for (int i = 0; i < n_samples; i++) {
            if (labels[i] == c) {
                class_idx[n_c++] = i;
            }
        }
        if (n_c == 0) {
            /* No samples for this class — leave tiles zero-packed and
             * labels set. The forward path will compute Hamming distances
             * against zero tiles; not meaningful but doesn't crash. */
            for (int j = 0; j < k_per_class; j++) {
                memset(bank->tiles_packed
                         + (size_t)(c * k_per_class + j) * Dp, 0, (size_t)Dp);
                bank->labels[c * k_per_class + j] = c;
            }
            continue;
        }

        /* Stratified mod-k init: assignments[s] = s % k_per_class for the
         * per-class sample list. Deterministic, substrate-clean. */
        for (int s = 0; s < n_c; s++) {
            assignments[s] = s % k_per_class;
        }

        /* Initial centroid computation from stratified assignments. */
        for (int it = 0; it <= max_iters; it++) {
            /* (Re)compute centroids from current assignments. */
            memset(sums, 0, (size_t)k_per_class * (size_t)D * sizeof(int32_t));
            memset(counts, 0, (size_t)k_per_class * sizeof(int));
            for (int s = 0; s < n_c; s++) {
                int cluster = assignments[s];
                const m4t_trit_t* sample = samples + (size_t)class_idx[s] * D;
                int32_t* sum_row = sums + (size_t)cluster * D;
                for (int d = 0; d < D; d++) {
                    sum_row[d] += (int32_t)sample[d];
                }
                counts[cluster]++;
            }
            for (int kk = 0; kk < k_per_class; kk++) {
                m4t_trit_t* cent = centroids_unpacked + (size_t)kk * D;
                if (counts[kk] == 0) {
                    /* Empty cluster — keep prior centroid (no update). */
                    continue;
                }
                gesh_threshold_int32_to_trit(cent, sums + (size_t)kk * D, D);
            }

            if (it == max_iters) break;

            /* Pack centroids for Hamming distance computation. */
            for (int kk = 0; kk < k_per_class; kk++) {
                m4t_pack_trits_1d(centroids_packed + (size_t)kk * Dp,
                                  centroids_unpacked + (size_t)kk * D, D);
            }

            /* Reassign each sample to nearest centroid. */
            int n_changed = 0;
            for (int s = 0; s < n_c; s++) {
                const uint8_t* sample_packed =
                    samples_packed + (size_t)class_idx[s] * Dp;
                int best_cluster = 0;
                int32_t best_dist = m4t_popcount_dist(
                    sample_packed, centroids_packed, mask, Dp);
                for (int kk = 1; kk < k_per_class; kk++) {
                    int32_t d = m4t_popcount_dist(
                        sample_packed,
                        centroids_packed + (size_t)kk * Dp,
                        mask, Dp);
                    if (d < best_dist) {
                        best_dist = d;
                        best_cluster = kk;
                    }
                }
                if (assignments[s] != best_cluster) {
                    assignments[s] = best_cluster;
                    n_changed++;
                }
            }
            if (n_changed == 0) break;
        }

        /* Pack final centroids into bank tiles for class c. */
        for (int kk = 0; kk < k_per_class; kk++) {
            m4t_pack_trits_1d(
                bank->tiles_packed + (size_t)(c * k_per_class + kk) * Dp,
                centroids_unpacked + (size_t)kk * D, D);
            bank->labels[c * k_per_class + kk] = c;
        }
    }

    free(mask);
    free(samples_packed);
    free(class_idx);
    free(assignments);
    free(sums);
    free(counts);
    free(centroids_unpacked);
    free(centroids_packed);
    free(tile_buf);
}

/* ── Hierarchical bank (P0-4 compositional routing) ────────────────────── */

int gesh_bank_hier_alloc(
    gesh_bank_hier_t* hbank,
    int n_classes,
    int sig_dim)
{
    assert(hbank && n_classes > 0 && sig_dim > 0);
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);

    hbank->n_stage1 = n_classes;
    hbank->sig_dim  = sig_dim;

    hbank->stage1.n_tiles      = n_classes;
    hbank->stage1.sig_dim      = sig_dim;
    hbank->stage1.tiles_packed = calloc((size_t)n_classes, (size_t)Dp);
    hbank->stage1.labels       = calloc((size_t)n_classes, sizeof(int));

    hbank->stage2 = calloc((size_t)n_classes, sizeof(gesh_bank_t));
    hbank->stage2_masks = calloc((size_t)n_classes, (size_t)Dp);

    if (!hbank->stage1.tiles_packed || !hbank->stage1.labels
        || !hbank->stage2 || !hbank->stage2_masks) return -1;

    for (int c = 0; c < n_classes; c++) {
        hbank->stage2[c].n_tiles      = n_classes;
        hbank->stage2[c].sig_dim      = sig_dim;
        hbank->stage2[c].tiles_packed = calloc((size_t)n_classes, (size_t)Dp);
        hbank->stage2[c].labels       = calloc((size_t)n_classes, sizeof(int));
        if (!hbank->stage2[c].tiles_packed || !hbank->stage2[c].labels) return -1;
        for (int k = 0; k < n_classes; k++) hbank->stage2[c].labels[k] = k;
    }
    return 0;
}

void gesh_bank_hier_free(gesh_bank_hier_t* hbank) {
    if (!hbank) return;
    free(hbank->stage1.tiles_packed); hbank->stage1.tiles_packed = NULL;
    free(hbank->stage1.labels);       hbank->stage1.labels = NULL;
    if (hbank->stage2) {
        for (int c = 0; c < hbank->n_stage1; c++) {
            free(hbank->stage2[c].tiles_packed);
            free(hbank->stage2[c].labels);
        }
        free(hbank->stage2); hbank->stage2 = NULL;
    }
    free(hbank->stage2_masks); hbank->stage2_masks = NULL;
    hbank->n_stage1 = 0;
}

void gesh_bank_build_hierarchical(
    gesh_bank_hier_t* hbank,
    const m4t_trit_t* samples,
    const int* labels,
    int n_samples,
    int n_classes,
    int snr_threshold_permille)
{
    assert(hbank && samples && labels);
    assert(n_classes == hbank->n_stage1);
    int D  = hbank->sig_dim;
    int Dp = M4T_TRIT_PACKED_BYTES(D);

    /* Step 1: build stage-1 wildcard bank in-place. */
    gesh_bank_build_class_wildcard(&hbank->stage1, samples, labels,
                                      n_samples, n_classes,
                                      snr_threshold_permille);

    /* Step 2a: precompute per-stage-1-tile dim-selector masks via the
     * substrate kernel. Wildcard positions in stage-1 tile c become
     * the active dims for stage-2 sub-bank c's distance compare. */
    for (int c = 0; c < n_classes; c++) {
        m4t_route_wildcard_select_mask(
            hbank->stage2_masks + (size_t)c * Dp,
            hbank->stage1.tiles_packed + (size_t)c * Dp,
            D);
    }

    /* Step 2b: assign each sample to its nearest stage-1 tile (by
     * wildcard distance). Stable tie-breaker: lowest tile index wins.
     * Build a full-active mask once for stage-1 routing. */
    uint8_t* full_mask = malloc((size_t)Dp);
    memset(full_mask, 0xFF, (size_t)Dp);
    int tail_trits = D - (D / 4) * 4;
    if (tail_trits > 0) {
        uint8_t tail = (uint8_t)((1u << (tail_trits * 2)) - 1u);
        full_mask[Dp - 1] = tail;
    }

    int* assign = malloc((size_t)n_samples * sizeof(int));
    uint8_t* sample_packed = malloc((size_t)Dp);
    m4t_trit_t* sample_buf = malloc((size_t)D * sizeof(m4t_trit_t));

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < D; j++) sample_buf[j] = samples[(size_t)i * D + j];
        m4t_pack_trits_1d(sample_packed, sample_buf, D);
        int best_t = 0;
        int32_t best_d = INT32_MAX;
        for (int t = 0; t < n_classes; t++) {
            int32_t d = m4t_route_wildcard_dist(
                sample_packed,
                hbank->stage1.tiles_packed + (size_t)t * Dp,
                full_mask, D);
            if (d < best_d) { best_d = d; best_t = t; }
        }
        assign[i] = best_t;
    }
    free(sample_packed);
    free(sample_buf);
    free(full_mask);

    /* Step 2c: build stage-2 sub-bank per stage-1 bucket.
     * Sub-bank is class-mean over the bucket's samples (full sig_dim).
     * The mask gates compare-time, not build-time. */
    int* bucket_labels = malloc((size_t)n_samples * sizeof(int));
    m4t_trit_t* bucket_samples = NULL;
    int bucket_cap = 0;

    for (int c = 0; c < n_classes; c++) {
        int n_in = 0;
        for (int i = 0; i < n_samples; i++) if (assign[i] == c) n_in++;

        if (n_in == 0) {
            /* Empty bucket: stage2[c] tiles already zero from calloc;
             * forward will fall back to stage-1 tile label. */
            continue;
        }

        if (n_in > bucket_cap) {
            free(bucket_samples);
            bucket_samples = malloc((size_t)n_in * (size_t)D * sizeof(m4t_trit_t));
            bucket_cap = n_in;
        }
        int idx = 0;
        for (int i = 0; i < n_samples; i++) {
            if (assign[i] != c) continue;
            memcpy(bucket_samples + (size_t)idx * D,
                   samples + (size_t)i * D,
                   (size_t)D * sizeof(m4t_trit_t));
            bucket_labels[idx] = labels[i];
            idx++;
        }
        gesh_bank_build_class_mean(&hbank->stage2[c],
                                      bucket_samples, bucket_labels,
                                      n_in, n_classes);
    }

    free(bucket_samples);
    free(bucket_labels);
    free(assign);
}
