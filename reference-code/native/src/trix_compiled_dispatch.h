/*
 * trix_compiled_dispatch.h — Compiled Dispatch for deterministic inference
 *
 * Converts dynamic routing into frozen dispatch tables for O(1) inference.
 * Workflow: train → profile (via surgery claim matrix) → compile → execute.
 *
 * A compiled entry maps a class_id to a specific tile. During inference,
 * if the class is known, routing is bypassed entirely: the input goes
 * directly to the assigned tile.
 *
 * Includes confidence guards, drift detection, and monitoring.
 *
 * Ported from compiled_dispatch.py.
 */

#ifndef TRIX_COMPILED_DISPATCH_H
#define TRIX_COMPILED_DISPATCH_H

#include "trix_ternary_route.h"
#include "trix_surgery.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Compiled Entry ── */

typedef struct {
    int   tile_idx;
    float frequency;       /* how often this tile was chosen for this class */
    float purity;          /* fraction of tile's traffic that is this class */
    float min_confidence;  /* guard threshold */
    int   version;
    bool  valid;           /* false = slot is empty */
} TrixCompiledEntry;

/* ── Profile Stats ── */

typedef struct {
    int   class_id;
    int   total_samples;
    int   mode_tile;       /* tile with highest count for this class */
    float mode_frequency;  /* mode_count / total */
    float purity;          /* mode_count / tile_total */
    float compilability;   /* frequency * purity */
} TrixProfileStats;

/* ── Dispatch Table ── */

typedef struct {
    int num_classes;          /* size of dispatch table */
    int num_tiles;
    TrixCompiledEntry* entries;  /* [num_classes] — one per class */

    /* Monitoring counters */
    int64_t compiled_hits;
    int64_t compiled_misses;  /* guard failed */
    int64_t dynamic_calls;    /* no compiled entry */
    int     version;
} TrixCompiledDispatch;

/* ── Lifecycle ── */

TrixCompiledDispatch* trix_dispatch_create(int num_classes, int num_tiles);
void trix_dispatch_destroy(TrixCompiledDispatch* cd);

/* ── Profiling ── */

/* Profile a single class using the surgery context's claim matrix.
 * Returns stats; compilability = frequency * purity. */
TrixProfileStats trix_dispatch_profile(const TrixCompiledDispatch* cd,
                                       const TrixSurgeryCtx* surgery,
                                       int class_id);

/* ── Compilation ── */

/* Compile a single class → tile mapping. */
int trix_dispatch_compile(TrixCompiledDispatch* cd,
                          int class_id, int tile_idx,
                          float frequency, float purity,
                          float min_confidence);

/* Auto-compile all classes above the compilability threshold.
 * Returns number of classes compiled. */
int trix_dispatch_compile_stable(TrixCompiledDispatch* cd,
                                 const TrixSurgeryCtx* surgery,
                                 float threshold,
                                 float min_confidence);

/* Remove a class from the dispatch table (return to dynamic routing). */
void trix_dispatch_decompile(TrixCompiledDispatch* cd, int class_id);

/* Remove all compiled entries. */
void trix_dispatch_decompile_all(TrixCompiledDispatch* cd);

/* ── Execution ── */

/* Look up the compiled tile for a class. Returns the tile index if
 * the class has a compiled entry and confidence >= guard threshold.
 * Returns -1 if no compiled path (caller should use dynamic routing).
 * Updates monitoring counters. */
int trix_dispatch_lookup(TrixCompiledDispatch* cd,
                         int class_hint, float confidence);

/* Execute compiled forward: runs the FFN for a single sample, bypassing
 * routing if a compiled path exists for class_hint.
 * x: [d_model] input, out: [d_model] output.
 * If no compiled path, falls back to full forward.
 * Returns 1 if compiled path was used, 0 if dynamic. */
int trix_dispatch_forward(TrixCompiledDispatch* cd,
                          TrixTernaryRoutedFFN* tr,
                          const float* x, float* out,
                          int class_hint, float confidence);

/* ── Monitoring ── */

typedef struct {
    int64_t compiled_hits;
    int64_t compiled_misses;
    int64_t dynamic_calls;
    int64_t total_calls;
    float   hit_rate;
    int     num_compiled;
    int     version;
} TrixDispatchStats;

TrixDispatchStats trix_dispatch_get_stats(const TrixCompiledDispatch* cd);
void trix_dispatch_reset_stats(TrixCompiledDispatch* cd);

/* ── Drift Detection ── */

/* Check how many compiled classes have drifted (mode tile changed or
 * frequency dropped by more than threshold). Returns count. */
int trix_dispatch_check_drift(const TrixCompiledDispatch* cd,
                              const TrixSurgeryCtx* surgery,
                              float threshold);

/* Recompile drifted classes. Returns number recompiled. */
int trix_dispatch_recompile_drifted(TrixCompiledDispatch* cd,
                                    const TrixSurgeryCtx* surgery,
                                    float threshold,
                                    float min_confidence);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_COMPILED_DISPATCH_H */
