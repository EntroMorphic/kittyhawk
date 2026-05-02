/*
 * gesh_bank.c — class-conditional ternary mean bank construction.
 *
 * Pure integer arithmetic. Per-class per-dim sum across samples,
 * sign-thresholded to ternary, packed into bank tiles.
 */

#include "gesh_bank.h"
#include "m4t_trit_pack.h"

#include <assert.h>
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

    /* Sign-threshold the per-class sums. Zero sum (or no samples for
     * class) → zero trit. Substrate-legal: pure integer. */
    m4t_trit_t* tile_unpacked = malloc((size_t)D * sizeof(m4t_trit_t));
    for (int c = 0; c < n_classes; c++) {
        const int32_t* row = class_sums + (size_t)c * D;
        for (int j = 0; j < D; j++) {
            int32_t v = row[j];
            tile_unpacked[j] = (v > 0) ?  1
                             : (v < 0) ? -1
                             :            0;
        }
        m4t_pack_trits_1d(
            bank->tiles_packed + (size_t)c * Dp,
            tile_unpacked, D);
        bank->labels[c] = c;
    }

    free(tile_unpacked);
    free(class_sums);
}
