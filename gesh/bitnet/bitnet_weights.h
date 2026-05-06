/*
 * gesh/bitnet/bitnet_weights.h — load + map a substrate-format BitNet
 * weights blob produced by `scripts/convert_weights.py`.
 *
 * Contract:
 *   bitnet_weights_load() opens the blob via mmap, validates the header,
 *   populates a bitnet_weights_t with pointers into the mapped region.
 *   The caller must keep the bitnet_weights_loaded_t handle alive for as
 *   long as it holds the bitnet_weights_t pointers.
 *
 *   bitnet_weights_unload() munmaps and clears.
 */

#ifndef GESH_BITNET_WEIGHTS_H
#define GESH_BITNET_WEIGHTS_H

#include "bitnet_config.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Lifetime handle. Returned from load; passed to unload. Caller treats
 * as opaque. */
typedef struct {
    void*  base;          /* mmap base pointer */
    size_t size;          /* mmap size */
    int    fd;            /* file descriptor (kept for munmap) */
    int    lm_head_tied;  /* 1 if lm_head reuses embedding */
    /* Per-tensor block exponents (read at load time; pointer into mmap). */
    const int32_t*  block_exps;
    int32_t         n_tensors;
} bitnet_weights_loaded_t;

/* Load a weights blob. Populates `weights` with pointers into the
 * mmap'd region. Returns 0 on success, non-zero on error (prints to
 * stderr).
 *
 * Layout matches scripts/convert_weights.py output:
 *   [magic "M4T1"][version: u32][lm_head_tied: u32][n_tensors: u32]
 *   [block_exps: i32 × n_tensors]
 *   [offsets:    u64 × n_tensors]
 *   [sizes:      u64 × n_tensors]
 *   [tensor_data]
 *
 * The order of tensors in the blob matches:
 *   embedding,
 *   layer0_w_q, w_k, w_v, w_o, w_gate, w_up, w_down,
 *   layer0_alpha_q, alpha_k, alpha_v, alpha_o, alpha_gate, alpha_up, alpha_down,
 *   layer0_gamma_input_norm, gamma_post_attn_norm, gamma_attn_sub_norm,
 *   gamma_ffn_sub_norm,
 *   ... (next layer same shape) ...
 *   final_norm.gamma,
 *   [lm_head if !lm_head_tied]
 *
 * On success: `weights` has pointers to all loaded tensors. Layers beyond
 * those converted (per the script's `--layers N` flag) have NULL pointers
 * and the loader prints how many layers it actually loaded. */
int bitnet_weights_load(
    const char* path,
    bitnet_weights_t* weights,
    bitnet_weights_loaded_t* handle
);

/* Unload (munmap, close fd). Safe to call with a zeroed handle. */
void bitnet_weights_unload(bitnet_weights_loaded_t* handle);

/* Block-exponent accessor: per-tensor int32 read from the loaded header.
 * Returns 0 if i out of bounds. */
int32_t bitnet_weights_block_exp(const bitnet_weights_loaded_t* h, int i);

#ifdef __cplusplus
}
#endif

#endif /* GESH_BITNET_WEIGHTS_H */
