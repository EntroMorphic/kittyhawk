/*
 * trix_compiled_dispatch.c — Compiled Dispatch implementation
 *
 * Ported from compiled_dispatch.py. Operates on the C ternary route engine.
 */

#include "trix_compiled_dispatch.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ══════════════════════════════════════════════════════════════════════
 * Lifecycle
 * ══════════════════════════════════════════════════════════════════════ */

TrixCompiledDispatch* trix_dispatch_create(int num_classes, int num_tiles) {
    if (num_classes <= 0 || num_tiles <= 0) return NULL;

    TrixCompiledDispatch* cd = calloc(1, sizeof(TrixCompiledDispatch));
    if (!cd) return NULL;

    cd->num_classes = num_classes;
    cd->num_tiles = num_tiles;

    cd->entries = calloc((size_t)num_classes, sizeof(TrixCompiledEntry));
    if (!cd->entries) { free(cd); return NULL; }

    /* All entries start invalid */
    for (int i = 0; i < num_classes; i++) {
        cd->entries[i].valid = false;
    }

    return cd;
}

void trix_dispatch_destroy(TrixCompiledDispatch* cd) {
    if (!cd) return;
    free(cd->entries);
    free(cd);
}

/* ══════════════════════════════════════════════════════════════════════
 * Profiling
 * ══════════════════════════════════════════════════════════════════════ */

TrixProfileStats trix_dispatch_profile(const TrixCompiledDispatch* cd,
                                       const TrixSurgeryCtx* surgery,
                                       int class_id)
{
    TrixProfileStats stats = {0};
    stats.class_id = class_id;
    stats.mode_tile = -1;

    if (!cd || !surgery) return stats;
    if (class_id < 0 || class_id >= surgery->num_classes) return stats;

    int T = surgery->num_tiles;
    int C = surgery->num_classes;

    /* Count how many samples of this class went to each tile */
    int64_t total = 0;
    int64_t best_count = 0;
    int best_tile = -1;

    for (int t = 0; t < T; t++) {
        int64_t count = surgery->claims[t * C + class_id];
        total += count;
        if (count > best_count) {
            best_count = count;
            best_tile = t;
        }
    }

    if (total == 0) return stats;

    stats.total_samples = (int)total;
    stats.mode_tile = best_tile;
    stats.mode_frequency = (float)best_count / (float)total;

    /* Purity: fraction of mode_tile's total traffic that is this class */
    int64_t tile_total = 0;
    for (int c = 0; c < C; c++) {
        tile_total += surgery->claims[best_tile * C + c];
    }
    stats.purity = (tile_total > 0)
                   ? (float)best_count / (float)tile_total
                   : 0.0f;

    stats.compilability = stats.mode_frequency * stats.purity;

    return stats;
}

/* ══════════════════════════════════════════════════════════════════════
 * Compilation
 * ══════════════════════════════════════════════════════════════════════ */

int trix_dispatch_compile(TrixCompiledDispatch* cd,
                          int class_id, int tile_idx,
                          float frequency, float purity,
                          float min_confidence)
{
    if (!cd) return -1;
    if (class_id < 0 || class_id >= cd->num_classes) return -1;
    if (tile_idx < 0 || tile_idx >= cd->num_tiles) return -1;

    TrixCompiledEntry* e = &cd->entries[class_id];
    e->tile_idx       = tile_idx;
    e->frequency      = frequency;
    e->purity         = purity;
    e->min_confidence = min_confidence;
    e->version        = cd->version;
    e->valid          = true;

    return 0;
}

int trix_dispatch_compile_stable(TrixCompiledDispatch* cd,
                                 const TrixSurgeryCtx* surgery,
                                 float threshold,
                                 float min_confidence)
{
    if (!cd || !surgery) return 0;

    int compiled = 0;
    int C = surgery->num_classes;
    if (C > cd->num_classes) C = cd->num_classes;

    for (int c = 0; c < C; c++) {
        TrixProfileStats stats = trix_dispatch_profile(cd, surgery, c);
        if (stats.compilability >= threshold && stats.mode_tile >= 0) {
            trix_dispatch_compile(cd, c, stats.mode_tile,
                                  stats.mode_frequency, stats.purity,
                                  min_confidence);
            compiled++;
        }
    }

    cd->version++;
    return compiled;
}

void trix_dispatch_decompile(TrixCompiledDispatch* cd, int class_id) {
    if (!cd || class_id < 0 || class_id >= cd->num_classes) return;
    cd->entries[class_id].valid = false;
}

void trix_dispatch_decompile_all(TrixCompiledDispatch* cd) {
    if (!cd) return;
    for (int i = 0; i < cd->num_classes; i++) {
        cd->entries[i].valid = false;
    }
    cd->version++;
}

/* ══════════════════════════════════════════════════════════════════════
 * Execution
 * ══════════════════════════════════════════════════════════════════════ */

int trix_dispatch_lookup(TrixCompiledDispatch* cd,
                         int class_hint, float confidence)
{
    if (!cd || class_hint < 0 || class_hint >= cd->num_classes) {
        if (cd) cd->dynamic_calls++;
        return -1;
    }

    TrixCompiledEntry* e = &cd->entries[class_hint];
    if (!e->valid) {
        cd->dynamic_calls++;
        return -1;
    }

    if (confidence >= e->min_confidence) {
        cd->compiled_hits++;
        return e->tile_idx;
    }

    cd->compiled_misses++;
    return -1;
}

int trix_dispatch_forward(TrixCompiledDispatch* cd,
                          TrixTernaryRoutedFFN* tr,
                          const float* x, float* out,
                          int class_hint, float confidence)
{
    if (!cd || !tr || !x || !out) return 0;

    int tile_idx = trix_dispatch_lookup(cd, class_hint, confidence);

    if (tile_idx < 0) {
        /* Dynamic fallback: full forward with routing */
        trix_ternary_route_forward(tr, x, out, 1);
        return 0;
    }

    /* Compiled path: bypass routing, run single tile directly.
     * This is the O(1) win: no signature matching, no score computation. */
    int D = tr->cfg.d_model;
    int H = tr->cfg.tile_hidden;
    int T = tr->cfg.num_tiles;

    /* Ensure scratch is allocated for batch=1 */
    if (tr->batch_cap < 1) {
        trix_ternary_route_forward(tr, x, out, 1);
        return 0;  /* First call initializes scratch; use dynamic */
    }

    /* LayerNorm */
    trix_layernorm_forward_save(tr->x_norm, tr->ln_mean, tr->ln_rstd,
        x, tr->ln_weight, tr->ln_bias, 1, D, tr->cfg.ln_eps);

    /* Single tile: z1 = x_norm @ W1_t^T + b1 */
    int8_t* W1t = tr->W1_tern + tile_idx * H * D;
    int8_t* W2t = tr->W2_tern + tile_idx * D * H;

    /* Use MTFP path for consistency with training */
    mtfp_from_float_batch(tr->mx_norm, tr->x_norm, D);

    mtfp_ternary_matmul_bt(tr->mz1, tr->mx_norm, W1t, 1, D, H);
    mtfp_fan_in_normalize(tr->mz1, H, D);
    mtfp_bias_add(tr->mz1, tr->mb1 + tile_idx * H, 1, H);
    mtfp_gelu(tr->mh1, tr->mz1, H);

    mtfp_ternary_matmul_bt(tr->mtile_out, tr->mh1, W2t, 1, H, D);
    mtfp_bias_add(tr->mtile_out, tr->mb2 + tile_idx * D, 1, D);

    /* Convert to float, apply output_scale + residual.
     * NOTE: compiled forward is inference-only. It does NOT save z1/h1 for
     * backward. Running backward after compiled forward will use stale
     * saved activations — do not mix compiled forward with training. */
    mtfp_to_float_batch(tr->combined, tr->mtile_out, D);

    for (int d = 0; d < D; d++) {
        out[d] = x[d] + tr->output_scale * tr->combined[d];
    }

    /* Set route array for diagnostics (only this tile active as +1) */
    memset(tr->route, 0, (size_t)T * sizeof(int));
    tr->route[tile_idx] = 1;

    return 1;
}

/* ══════════════════════════════════════════════════════════════════════
 * Monitoring
 * ══════════════════════════════════════════════════════════════════════ */

TrixDispatchStats trix_dispatch_get_stats(const TrixCompiledDispatch* cd) {
    TrixDispatchStats s = {0};
    if (!cd) return s;

    s.compiled_hits  = cd->compiled_hits;
    s.compiled_misses = cd->compiled_misses;
    s.dynamic_calls  = cd->dynamic_calls;
    s.total_calls    = s.compiled_hits + s.compiled_misses + s.dynamic_calls;
    s.hit_rate       = (s.total_calls > 0)
                       ? (float)s.compiled_hits / (float)s.total_calls
                       : 0.0f;
    s.version = cd->version;

    int count = 0;
    for (int i = 0; i < cd->num_classes; i++) {
        if (cd->entries[i].valid) count++;
    }
    s.num_compiled = count;

    return s;
}

void trix_dispatch_reset_stats(TrixCompiledDispatch* cd) {
    if (!cd) return;
    cd->compiled_hits = 0;
    cd->compiled_misses = 0;
    cd->dynamic_calls = 0;
}

/* ══════════════════════════════════════════════════════════════════════
 * Drift Detection
 * ══════════════════════════════════════════════════════════════════════ */

int trix_dispatch_check_drift(const TrixCompiledDispatch* cd,
                              const TrixSurgeryCtx* surgery,
                              float threshold)
{
    if (!cd || !surgery) return 0;

    int drifted = 0;
    for (int c = 0; c < cd->num_classes; c++) {
        if (!cd->entries[c].valid) continue;

        TrixProfileStats stats = trix_dispatch_profile(cd, surgery, c);
        TrixCompiledEntry* e = &cd->entries[c];

        /* Mode tile changed */
        if (stats.mode_tile != e->tile_idx) { drifted++; continue; }

        /* Frequency dropped significantly */
        if (stats.mode_frequency < e->frequency - threshold) { drifted++; }
    }

    return drifted;
}

int trix_dispatch_recompile_drifted(TrixCompiledDispatch* cd,
                                    const TrixSurgeryCtx* surgery,
                                    float threshold,
                                    float min_confidence)
{
    if (!cd || !surgery) return 0;

    int recompiled = 0;
    for (int c = 0; c < cd->num_classes; c++) {
        if (!cd->entries[c].valid) continue;

        TrixProfileStats stats = trix_dispatch_profile(cd, surgery, c);
        TrixCompiledEntry* e = &cd->entries[c];

        bool drifted = false;
        if (stats.mode_tile != e->tile_idx) drifted = true;
        if (stats.mode_frequency < e->frequency - threshold) drifted = true;

        if (!drifted) continue;

        if (stats.compilability >= threshold && stats.mode_tile >= 0) {
            trix_dispatch_compile(cd, c, stats.mode_tile,
                                  stats.mode_frequency, stats.purity,
                                  min_confidence);
            recompiled++;
        } else {
            trix_dispatch_decompile(cd, c);
        }
    }

    if (recompiled > 0) cd->version++;
    return recompiled;
}
