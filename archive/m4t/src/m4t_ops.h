/*
 * m4t_ops.h — function-pointer opcode tables for M4T
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Three indexable tables exposing M4T primitives as an external API.
 * Each entry is a function pointer to the corresponding kernel. Callers
 * can dispatch by opcode index at runtime or call functions directly by
 * name — both reach the same body.
 *
 * This satisfies contract clause 6 (indexable): every opcode is
 * reachable via both a direct C call and a table entry.
 */

#ifndef M4T_OPS_H
#define M4T_OPS_H

#include "m4t_types.h"
#include "m4t_trit_ops.h"
#include "m4t_trit_pack.h"
#include "m4t_trit_reducers.h"
#include "m4t_mtfp.h"
#include "m4t_mtfp_w.h"
#include "m4t_mtfp4.h"
#include "m4t_ternary_matmul.h"
#include "m4t_route.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Trit ops table ────────────────────────────────────────────────────── */

typedef void (*m4t_trit_op_fn)(
    uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits);

enum m4t_trit_opcode {
    M4T_TOP_MUL,
    M4T_TOP_SAT_ADD,
    M4T_TOP_MAX,
    M4T_TOP_MIN,
    M4T_TOP_EQ,
    M4T_TOP_NEG,
    M4T_TOP_COUNT
};

extern const m4t_trit_op_fn m4t_trit_ops[M4T_TOP_COUNT];

/* ── MTFP19 ops table ──────────────────────────────────────────────────── */

/* MTFP ops have varying signatures. The table stores generic function
 * pointers cast to void(*)(void); callers cast back to the correct
 * signature using the shape tag. Calling through a cast function pointer
 * whose type differs from the original is technically UB per C99
 * §6.3.2.3¶8, but is universally supported on aarch64 (all arguments
 * pass through registers per the calling convention) and is the standard
 * pattern for C dispatch tables (GLib, SQLite, etc.).
 *
 * MTFP39 and MTFP4 ops are intentionally absent from this table for v0.
 * Those paths are direct-call-only. Tables for the wide and narrow cell
 * types land when a consumer needs indexed dispatch to them.
 *
 * Trit reducers (signed_sum, sparsity, counts) have a different return
 * type (int64_t, not void) and are also direct-call-only. */

enum m4t_mtfp_op_shape {
    M4T_MOP_SHAPE_VEC_BINARY,    /* void(mtfp*, const mtfp*, const mtfp*, int) */
    M4T_MOP_SHAPE_VEC_INPLACE,   /* void(mtfp*, const mtfp*, int) */
    M4T_MOP_SHAPE_MATMUL,        /* void(mtfp*, const mtfp*, const mtfp*, int, int, int) */
    M4T_MOP_SHAPE_TERNARY_MATMUL,/* void(mtfp*, const mtfp*, const uint8_t*, int, int, int) */
    M4T_MOP_SHAPE_LAYERNORM,     /* void(mtfp*, const mtfp*, const mtfp*, const mtfp*, mtfp, int, int) */
    M4T_MOP_SHAPE_OTHER
};

typedef struct {
    void (*fn)(void);  /* cast to correct signature per shape */
    enum m4t_mtfp_op_shape shape;
} m4t_mtfp_op_entry_t;

enum m4t_mtfp_opcode {
    M4T_MOP_VEC_ADD,
    M4T_MOP_VEC_ADD_INPLACE,
    M4T_MOP_VEC_SUB_INPLACE,
    M4T_MOP_VEC_ZERO,
    M4T_MOP_VEC_SCALE,
    M4T_MOP_BIAS_ADD,
    M4T_MOP_FAN_IN_NORMALIZE,
    M4T_MOP_LAYERNORM,
    M4T_MOP_MATMUL,
    M4T_MOP_MATMUL_BT,
    M4T_MOP_TERNARY_MATMUL_BT,
    M4T_MOP_COUNT
};

extern const m4t_mtfp_op_entry_t m4t_mtfp_ops[M4T_MOP_COUNT];

/* ── Route ops table ───────────────────────────────────────────────────── */

/* Route ops also have varying signatures. Same pattern as MTFP ops. */

enum m4t_route_opcode {
    M4T_ROP_SIGN_EXTRACT,
    M4T_ROP_DISTANCE_BATCH,
    M4T_ROP_TOPK_ABS,
    M4T_ROP_APPLY_SIGNED,
    M4T_ROP_SIGNATURE_UPDATE,
    M4T_ROP_COUNT
};

typedef struct {
    void (*fn)(void);
    const char* name;
} m4t_route_op_entry_t;

extern const m4t_route_op_entry_t m4t_route_ops[M4T_ROP_COUNT];

#ifdef __cplusplus
}
#endif

#endif /* M4T_OPS_H */
