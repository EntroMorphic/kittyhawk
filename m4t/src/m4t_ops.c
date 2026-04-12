/*
 * m4t_ops.c — function-pointer opcode tables for M4T
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * No new kernel code. Just pointer assignments to existing functions.
 */

#include "m4t_ops.h"

/* ── Trit ops table ────────────────────────────────────────────────────── */

const m4t_trit_op_fn m4t_trit_ops[M4T_TOP_COUNT] = {
    [M4T_TOP_MUL]     = m4t_trit_mul,
    [M4T_TOP_SAT_ADD] = m4t_trit_sat_add,
    [M4T_TOP_MAX]     = m4t_trit_max,
    [M4T_TOP_MIN]     = m4t_trit_min,
    [M4T_TOP_EQ]      = m4t_trit_eq,
    [M4T_TOP_NEG]     = m4t_trit_neg,
};

/* ── MTFP19 ops table ──────────────────────────────────────────────────── */

const m4t_mtfp_op_entry_t m4t_mtfp_ops[M4T_MOP_COUNT] = {
    [M4T_MOP_VEC_ADD]           = { (void(*)(void))m4t_mtfp_vec_add,           M4T_MOP_SHAPE_VEC_BINARY },
    [M4T_MOP_VEC_ADD_INPLACE]   = { (void(*)(void))m4t_mtfp_vec_add_inplace,   M4T_MOP_SHAPE_VEC_INPLACE },
    [M4T_MOP_VEC_SUB_INPLACE]   = { (void(*)(void))m4t_mtfp_vec_sub_inplace,   M4T_MOP_SHAPE_VEC_INPLACE },
    [M4T_MOP_VEC_ZERO]          = { (void(*)(void))m4t_mtfp_vec_zero,          M4T_MOP_SHAPE_OTHER },
    [M4T_MOP_VEC_SCALE]         = { (void(*)(void))m4t_mtfp_vec_scale,         M4T_MOP_SHAPE_OTHER },
    [M4T_MOP_BIAS_ADD]          = { (void(*)(void))m4t_mtfp_bias_add,          M4T_MOP_SHAPE_OTHER },
    [M4T_MOP_FAN_IN_NORMALIZE]  = { (void(*)(void))m4t_mtfp_fan_in_normalize,  M4T_MOP_SHAPE_OTHER },
    [M4T_MOP_LAYERNORM]         = { (void(*)(void))m4t_mtfp_layernorm,         M4T_MOP_SHAPE_LAYERNORM },
    [M4T_MOP_MATMUL]            = { (void(*)(void))m4t_mtfp_matmul,            M4T_MOP_SHAPE_MATMUL },
    [M4T_MOP_MATMUL_BT]         = { (void(*)(void))m4t_mtfp_matmul_bt,         M4T_MOP_SHAPE_MATMUL },
    [M4T_MOP_TERNARY_MATMUL_BT] = { (void(*)(void))m4t_mtfp_ternary_matmul_bt, M4T_MOP_SHAPE_TERNARY_MATMUL },
};

/* ── Route ops table ───────────────────────────────────────────────────── */

const m4t_route_op_entry_t m4t_route_ops[M4T_ROP_COUNT] = {
    [M4T_ROP_SIGN_EXTRACT]      = { (void(*)(void))m4t_route_sign_extract,      "sign_extract" },
    [M4T_ROP_DISTANCE_BATCH]    = { (void(*)(void))m4t_route_distance_batch,    "distance_batch" },
    [M4T_ROP_TOPK_ABS]          = { (void(*)(void))m4t_route_topk_abs,          "topk_abs" },
    [M4T_ROP_APPLY_SIGNED]      = { (void(*)(void))m4t_route_apply_signed,      "apply_signed" },
    [M4T_ROP_SIGNATURE_UPDATE]  = { (void(*)(void))m4t_route_signature_update,  "signature_update" },
};
