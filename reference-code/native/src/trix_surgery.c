/*
 * trix_surgery.c — Signature Surgery API implementation
 *
 * Ported from sparse_lookup_v2.py surgery API + claim tracking.
 */

#include "trix_surgery.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ══════════════════════════════════════════════════════════════════════
 * Lifecycle
 * ══════════════════════════════════════════════════════════════════════ */

TrixSurgeryCtx* trix_surgery_create(int num_tiles, int d_model,
                                     int num_classes, int history_cap)
{
    if (num_tiles <= 0 || d_model <= 0 || num_classes <= 0 || history_cap <= 0)
        return NULL;

    TrixSurgeryCtx* ctx = calloc(1, sizeof(TrixSurgeryCtx));
    if (!ctx) return NULL;

    ctx->num_tiles = num_tiles;
    ctx->d_model   = d_model;
    ctx->num_classes = num_classes;

    ctx->frozen = calloc((size_t)num_tiles, sizeof(bool));
    if (!ctx->frozen) { free(ctx); return NULL; }

    ctx->claims = calloc((size_t)num_tiles * (size_t)num_classes, sizeof(int64_t));
    if (!ctx->claims) { free(ctx->frozen); free(ctx); return NULL; }

    ctx->history_cap = history_cap;
    ctx->history_len = 0;
    ctx->history = calloc((size_t)history_cap, sizeof(TrixSurgeryEntry));
    if (!ctx->history) {
        free(ctx->claims); free(ctx->frozen); free(ctx); return NULL;
    }

    return ctx;
}

void trix_surgery_destroy(TrixSurgeryCtx* ctx) {
    if (!ctx) return;
    free(ctx->frozen);
    free(ctx->claims);
    free(ctx->history);
    free(ctx);
}

/* ══════════════════════════════════════════════════════════════════════
 * Internal: append to history ring buffer
 * ══════════════════════════════════════════════════════════════════════ */

static void surgery_record(TrixSurgeryCtx* ctx, TrixSurgeryAction action,
                           int tile_idx, int frozen_state, const char* tag)
{
    TrixSurgeryEntry* e;
    if (ctx->history_len < ctx->history_cap) {
        e = &ctx->history[ctx->history_len++];
    } else {
        /* Ring: overwrite oldest. Shift everything left by one. */
        memmove(ctx->history, ctx->history + 1,
                (size_t)(ctx->history_cap - 1) * sizeof(TrixSurgeryEntry));
        e = &ctx->history[ctx->history_cap - 1];
    }
    e->action   = action;
    e->tile_idx = tile_idx;
    e->frozen   = frozen_state;
    memset(e->tag, 0, sizeof(e->tag));
    if (tag) {
        strncpy(e->tag, tag, sizeof(e->tag) - 1);
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Signature Operations
 * ══════════════════════════════════════════════════════════════════════ */

int trix_surgery_insert(TrixSurgeryCtx* ctx, TrixTernaryRoutedFFN* tr,
                        int tile_idx, const float* sig, bool freeze,
                        const char* tag)
{
    if (!ctx || !tr || !sig) return -1;
    if (tile_idx < 0 || tile_idx >= ctx->num_tiles) return -1;
    if (ctx->d_model != tr->cfg.d_model) return -1;

    int D = ctx->d_model;

    /* Copy signature into the FFN's signature array.
     * Signatures are stored as float {-1, 0, +1}. */
    memcpy(tr->signatures + tile_idx * D, sig, (size_t)D * sizeof(float));

    if (freeze) {
        ctx->frozen[tile_idx] = true;
    }

    surgery_record(ctx, TRIX_SURGERY_INSERT, tile_idx, freeze ? 1 : 0, tag);
    return 0;
}

void trix_surgery_freeze(TrixSurgeryCtx* ctx, int tile_idx) {
    if (!ctx || tile_idx < 0 || tile_idx >= ctx->num_tiles) return;
    ctx->frozen[tile_idx] = true;
    surgery_record(ctx, TRIX_SURGERY_FREEZE, tile_idx, 1, NULL);
}

void trix_surgery_unfreeze(TrixSurgeryCtx* ctx, int tile_idx) {
    if (!ctx || tile_idx < 0 || tile_idx >= ctx->num_tiles) return;
    ctx->frozen[tile_idx] = false;
    surgery_record(ctx, TRIX_SURGERY_UNFREEZE, tile_idx, 0, NULL);
}

bool trix_surgery_is_frozen(const TrixSurgeryCtx* ctx, int tile_idx) {
    if (!ctx || tile_idx < 0 || tile_idx >= ctx->num_tiles) return false;
    return ctx->frozen[tile_idx];
}

/* ══════════════════════════════════════════════════════════════════════
 * Claim Tracking
 * ══════════════════════════════════════════════════════════════════════ */

void trix_surgery_update_claims(TrixSurgeryCtx* ctx,
                                const int* route, const int* labels,
                                int batch, int num_tiles)
{
    if (!ctx || !route || !labels) return;

    int T = num_tiles;
    int C = ctx->num_classes;

    for (int i = 0; i < batch; i++) {
        int label = labels[i];
        if (label < 0 || label >= C) continue;

        for (int t = 0; t < T && t < ctx->num_tiles; t++) {
            if (route[i * T + t] != 0) {
                ctx->claims[t * C + label]++;
            }
        }
    }
}

void trix_surgery_reset_claims(TrixSurgeryCtx* ctx) {
    if (!ctx || !ctx->claims) return;
    memset(ctx->claims, 0,
           (size_t)ctx->num_tiles * (size_t)ctx->num_classes * sizeof(int64_t));
}

int64_t trix_surgery_get_claim(const TrixSurgeryCtx* ctx,
                                int tile_idx, int class_id)
{
    if (!ctx || tile_idx < 0 || tile_idx >= ctx->num_tiles) return 0;
    if (class_id < 0 || class_id >= ctx->num_classes) return 0;
    return ctx->claims[tile_idx * ctx->num_classes + class_id];
}

int trix_surgery_mode_tile(const TrixSurgeryCtx* ctx, int class_id,
                           float* frequency_out)
{
    if (!ctx || class_id < 0 || class_id >= ctx->num_classes) {
        if (frequency_out) *frequency_out = 0.0f;
        return -1;
    }

    int C = ctx->num_classes;
    int best_tile = -1;
    int64_t best_count = 0;
    int64_t total = 0;

    for (int t = 0; t < ctx->num_tiles; t++) {
        int64_t count = ctx->claims[t * C + class_id];
        total += count;
        if (count > best_count) {
            best_count = count;
            best_tile = t;
        }
    }

    if (frequency_out) {
        *frequency_out = (total > 0) ? (float)best_count / (float)total : 0.0f;
    }
    return best_tile;
}

/* ══════════════════════════════════════════════════════════════════════
 * Analysis
 * ══════════════════════════════════════════════════════════════════════ */

TrixSignatureAnalysis trix_surgery_analyze(const TrixSurgeryCtx* ctx,
                                           const TrixTernaryRoutedFFN* tr,
                                           int tile_idx)
{
    TrixSignatureAnalysis a = {0};
    a.tile_idx = tile_idx;

    if (!ctx || !tr || tile_idx < 0 || tile_idx >= ctx->num_tiles) return a;

    int D = tr->cfg.d_model;
    const float* sig = tr->signatures + tile_idx * D;

    for (int d = 0; d < D; d++) {
        if (sig[d] > 0.5f)       a.num_positive++;
        else if (sig[d] < -0.5f) a.num_negative++;
        else                     a.num_zero++;
    }
    a.frozen = ctx->frozen[tile_idx] ? 1 : 0;

    return a;
}

const TrixSurgeryEntry* trix_surgery_get_history(const TrixSurgeryCtx* ctx,
                                                  int* len)
{
    if (!ctx) { if (len) *len = 0; return NULL; }
    if (len) *len = ctx->history_len;
    return ctx->history;
}

/* ══════════════════════════════════════════════════════════════════════
 * Freeze-Aware Optimizer Hook
 * ══════════════════════════════════════════════════════════════════════ */

void trix_surgery_save_signatures(const TrixSurgeryCtx* ctx,
                                  const TrixTernaryRoutedFFN* tr,
                                  float* saved_sigs)
{
    if (!ctx || !tr || !saved_sigs) return;
    int D = tr->cfg.d_model;
    int T = tr->cfg.num_tiles;
    memcpy(saved_sigs, tr->signatures, (size_t)T * (size_t)D * sizeof(float));
}

void trix_surgery_restore_frozen(const TrixSurgeryCtx* ctx,
                                 TrixTernaryRoutedFFN* tr,
                                 const float* saved_sigs)
{
    if (!ctx || !tr || !saved_sigs) return;
    int D = tr->cfg.d_model;

    for (int t = 0; t < ctx->num_tiles; t++) {
        if (ctx->frozen[t]) {
            memcpy(tr->signatures + t * D,
                   saved_sigs + t * D,
                   (size_t)D * sizeof(float));
        }
    }
}
