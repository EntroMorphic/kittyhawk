/*
 * expr_bank.c — implementation of expr_bank_build.
 *
 * Equivalence-class detection: two candidates are in the same class iff
 * their signatures are byte-equal. We compute each candidate's signature
 * once, then sweep linearly to assign each candidate to either an existing
 * class (if its signature matches a representative we've already kept) or
 * a new class (if not).
 *
 * Linear sweep is O(n_candidates × n_classes_so_far) — fine for the small
 * banks P0 needs. If the bank ever grows large, swap for a hash on the
 * packed signature.
 */

#include "expr_bank.h"
#include "expr_signature.h"
#include "m4t_trit_pack.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

void expr_bank_build(
    expr_bank_t* bank,
    const expr_t* const* candidates,
    int n_candidates,
    const m4t_mtfp_t* test_inputs,
    int n_test_inputs,
    int n_vars)
{
    assert(bank && candidates && test_inputs);
    assert(n_candidates > 0 && n_test_inputs > 0 && n_vars >= 1);
    assert(bank->base.sig_dim == n_test_inputs);
    assert(bank->base.tiles_packed && bank->base.labels);
    assert(bank->candidate_to_class);

    int Dp = M4T_TRIT_PACKED_BYTES(n_test_inputs);

    bank->n_candidates  = n_candidates;
    bank->n_vars        = n_vars;
    bank->n_test_inputs = n_test_inputs;

    /* Per-candidate signature workspace. */
    uint8_t* sig_buf = (uint8_t*)malloc((size_t)Dp);
    assert(sig_buf);

    int n_classes = 0;

    for (int c = 0; c < n_candidates; c++) {
        expr_to_signature(sig_buf, candidates[c], test_inputs,
                            n_test_inputs, n_vars);

        /* Search existing classes for a byte-equal signature. */
        int matched_class = -1;
        for (int k = 0; k < n_classes; k++) {
            const uint8_t* tile = bank->base.tiles_packed + (size_t)k * Dp;
            if (memcmp(tile, sig_buf, (size_t)Dp) == 0) {
                matched_class = k;
                break;
            }
        }

        if (matched_class >= 0) {
            /* Merge: candidate joins existing class. */
            bank->candidate_to_class[c] = matched_class;
        } else {
            /* New class: candidate becomes the representative. */
            int k = n_classes++;
            uint8_t* tile = bank->base.tiles_packed + (size_t)k * Dp;
            memcpy(tile, sig_buf, (size_t)Dp);
            bank->base.labels[k] = c;          /* representative = this candidate */
            bank->candidate_to_class[c] = k;
        }
    }

    bank->base.n_tiles = n_classes;

    free(sig_buf);
}

/* ── R1 dual-signature bank constructor ─────────────────────────────────── */

void expr_bank_dual_build(
    expr_bank_dual_t* bank,
    const expr_t* const* candidates,
    int n_candidates,
    const m4t_mtfp_t* test_inputs,
    int n_test_inputs,
    int n_vars)
{
    assert(bank && candidates && test_inputs);
    assert(n_candidates > 0 && n_test_inputs > 0 && n_vars >= 1);
    assert(bank->base.sig_dim == n_test_inputs);
    assert(bank->base.tiles_packed && bank->base.labels);
    assert(bank->conf_bits_per_tile);
    assert(bank->candidate_to_class);

    int Dp = M4T_TRIT_PACKED_BYTES(n_test_inputs);
    int Cp = (n_test_inputs + 7) / 8;

    bank->n_candidates  = n_candidates;
    bank->n_vars        = n_vars;
    bank->n_test_inputs = n_test_inputs;

    /* Per-candidate signature workspace (trit + conf). */
    uint8_t* trit_buf = (uint8_t*)malloc((size_t)Dp);
    uint8_t* conf_buf = (uint8_t*)malloc((size_t)Cp);
    assert(trit_buf && conf_buf);

    int n_classes = 0;

    for (int c = 0; c < n_candidates; c++) {
        expr_to_signature_dual(trit_buf, conf_buf, candidates[c],
                                 test_inputs, n_test_inputs, n_vars);

        /* Equivalence-class detection: byte-equal on BOTH trit_sig AND
         * conf_bits with an existing representative. */
        int matched_class = -1;
        for (int k = 0; k < n_classes; k++) {
            const uint8_t* tile_trit = bank->base.tiles_packed + (size_t)k * Dp;
            const uint8_t* tile_conf = bank->conf_bits_per_tile + (size_t)k * Cp;
            if (memcmp(tile_trit, trit_buf, (size_t)Dp) == 0 &&
                memcmp(tile_conf, conf_buf, (size_t)Cp) == 0) {
                matched_class = k;
                break;
            }
        }

        if (matched_class >= 0) {
            bank->candidate_to_class[c] = matched_class;
        } else {
            int k = n_classes++;
            uint8_t* tile_trit = bank->base.tiles_packed + (size_t)k * Dp;
            uint8_t* tile_conf = bank->conf_bits_per_tile + (size_t)k * Cp;
            memcpy(tile_trit, trit_buf, (size_t)Dp);
            memcpy(tile_conf, conf_buf, (size_t)Cp);
            bank->base.labels[k] = c;
            bank->candidate_to_class[c] = k;
        }
    }

    bank->base.n_tiles = n_classes;

    free(trit_buf);
    free(conf_buf);
}
