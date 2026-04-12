/*
 * trix_surgery.h — Signature Surgery API
 *
 * Runtime inspection and editing of ternary routing signatures with
 * full audit trail. Operates on TrixTernaryRoutedFFN instances.
 *
 * Capabilities:
 *   - Insert hand-designed signatures into tiles
 *   - Freeze/unfreeze individual tile signatures (gradient control)
 *   - Track which classes route to which tiles (claim matrix)
 *   - Query surgery history and signature analysis
 *
 * Ported from sparse_lookup_v2.py surgery API.
 */

#ifndef TRIX_SURGERY_H
#define TRIX_SURGERY_H

#include "trix_ternary_route.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Surgery History Entry ── */

typedef enum {
    TRIX_SURGERY_INSERT = 0,
    TRIX_SURGERY_FREEZE = 1,
    TRIX_SURGERY_UNFREEZE = 2,
} TrixSurgeryAction;

typedef struct {
    TrixSurgeryAction action;
    int               tile_idx;
    int               frozen;    /* 1 if tile was frozen after this op */
    char              tag[64];   /* user-provided label */
} TrixSurgeryEntry;

/* ── Signature Analysis ── */

typedef struct {
    int  tile_idx;
    int  num_positive;      /* dims where sig > 0 */
    int  num_negative;      /* dims where sig < 0 */
    int  num_zero;          /* dims where sig == 0 */
    int  frozen;
} TrixSignatureAnalysis;

/* ── Surgery Context ──
 *
 * Attached to a TrixTernaryRoutedFFN. Manages freeze state, claim matrix,
 * and audit history. One context per FFN instance. */

typedef struct {
    int   num_tiles;
    int   d_model;
    int   num_classes;       /* columns of claim matrix (set at init or grow) */

    /* Freeze mask: frozen[t] == 1 means tile t's signature is frozen */
    bool* frozen;            /* [num_tiles] */

    /* Claim matrix: claims[t * num_classes + c] = count of class c routed to tile t */
    int64_t* claims;         /* [num_tiles * num_classes] */

    /* Surgery history (ring buffer) */
    TrixSurgeryEntry* history;
    int history_cap;
    int history_len;
} TrixSurgeryCtx;

/* ── Lifecycle ── */

/* Create a surgery context for an FFN with the given tile/dim counts.
 * num_classes: number of distinct classes for claim tracking (e.g. 10 for MNIST).
 * history_cap: max history entries to retain (oldest are dropped). */
TrixSurgeryCtx* trix_surgery_create(int num_tiles, int d_model,
                                     int num_classes, int history_cap);
void trix_surgery_destroy(TrixSurgeryCtx* ctx);

/* ── Signature Operations ── */

/* Insert a ternary signature into tile_idx. sig must be [d_model] with
 * values in {-1, 0, +1}. If freeze is true, the tile is frozen after insert.
 * tag is an optional label (up to 63 chars, NULL for none).
 * Returns 0 on success, -1 on invalid args.
 *
 * NOTE: This updates the float signature array but does NOT re-quantize
 * or re-pack ternary tile weights. Routing scores (which use signatures,
 * not tile weights) will reflect the change immediately. Tile weights
 * are re-packed at the next trix_ternary_route_adamw_step call. */
int trix_surgery_insert(TrixSurgeryCtx* ctx, TrixTernaryRoutedFFN* tr,
                        int tile_idx, const float* sig, bool freeze,
                        const char* tag);

/* Freeze a tile's signature: trix_ternary_route_adamw_step will skip
 * gradient updates for this tile's signature row. */
void trix_surgery_freeze(TrixSurgeryCtx* ctx, int tile_idx);

/* Unfreeze a tile's signature. */
void trix_surgery_unfreeze(TrixSurgeryCtx* ctx, int tile_idx);

/* Check if a tile is frozen. */
bool trix_surgery_is_frozen(const TrixSurgeryCtx* ctx, int tile_idx);

/* ── Claim Tracking ── */

/* Update claim matrix from a batch of routing results and labels.
 * route: [batch * num_tiles] ternary route from last forward pass.
 * labels: [batch] class labels (0-based).
 * For each sample, every tile with route != 0 gets a claim for that label. */
void trix_surgery_update_claims(TrixSurgeryCtx* ctx,
                                const int* route, const int* labels,
                                int batch, int num_tiles);

/* Reset all claim counts to zero. */
void trix_surgery_reset_claims(TrixSurgeryCtx* ctx);

/* Get claim count for a specific tile and class. */
int64_t trix_surgery_get_claim(const TrixSurgeryCtx* ctx, int tile_idx, int class_id);

/* Get the tile with the highest claim count for a class (mode tile).
 * Returns tile index, or -1 if no claims exist.
 * If frequency_out is non-NULL, stores mode_count / total_count. */
int trix_surgery_mode_tile(const TrixSurgeryCtx* ctx, int class_id,
                           float* frequency_out);

/* ── Analysis ── */

/* Analyze a tile's current signature in the FFN. */
TrixSignatureAnalysis trix_surgery_analyze(const TrixSurgeryCtx* ctx,
                                           const TrixTernaryRoutedFFN* tr,
                                           int tile_idx);

/* Get surgery history. Returns pointer to internal array; len is set to
 * the number of valid entries. Do not free. */
const TrixSurgeryEntry* trix_surgery_get_history(const TrixSurgeryCtx* ctx,
                                                  int* len);

/* ── Freeze-Aware Optimizer Hook ──
 *
 * Call this AFTER trix_ternary_route_adamw_step to zero out signature
 * changes for frozen tiles (restores pre-step values).
 * saved_sigs: buffer of [num_tiles * d_model] holding pre-step signatures.
 * Call trix_surgery_save_signatures before the optimizer step. */
void trix_surgery_save_signatures(const TrixSurgeryCtx* ctx,
                                  const TrixTernaryRoutedFFN* tr,
                                  float* saved_sigs);

void trix_surgery_restore_frozen(const TrixSurgeryCtx* ctx,
                                 TrixTernaryRoutedFFN* tr,
                                 const float* saved_sigs);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_SURGERY_H */
