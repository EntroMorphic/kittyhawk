/*
 * gesh/bitnet/bitnet_weights.c — mmap + validate the weights blob.
 *
 * Read-only mmap. The blob layout is described in the Python converter
 * (`scripts/convert_weights.py`) and bitnet_weights.h.
 *
 * The order of tensors must match between Python and C exactly. We
 * encode it once here and once there; if they diverge, the load will
 * succeed (no integrity check), but the harness will read the wrong
 * pointers. To mitigate: a per-tensor checksum or name-table is future
 * work; for Phase 1, we rely on the order being stable.
 */

#include "bitnet_weights.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* Tensor order — MUST MATCH scripts/convert_weights.py exactly. The Python
 * writer and this C reader share an implicit contract: same tensor ordering
 * in the same fixed sequence. Divergence between them produces silent
 * corruption (the load succeeds but pointers point to wrong tensors).
 *
 * Future hardening: a tensor-name table in the blob header with C-side
 * verification on load. For Phase 1 we rely on the order being stable and
 * call out the coupling here for any future maintainer.
 *
 * Endianness: the blob format is little-endian (Python pack '<' codes);
 * C side reads via memcpy + native struct alignment. aarch64 is LE so
 * this works. Non-aarch64 is excluded by m4t/src/m4t_internal.h's #error. */
#define TENSORS_PER_LAYER  18  /* 7 weights + 7 scales + 4 norm γ */

/* Index helpers — return the tensor index for a given (layer, slot).
 * Slot 0..6 = w_q..w_down, 7..13 = α_q..α_down, 14..17 = norm γ × 4. */
static int idx_embedding(void) { return 0; }
static int idx_layer_base(int layer) { return 1 + layer * TENSORS_PER_LAYER; }
static int idx_layer_w(int layer, int slot)   { return idx_layer_base(layer) + slot; }      /* slot 0..6 */
/* idx_layer_alpha kept for future use when α scales are wired into the
 * forward pass (work-unit 5). Currently unreferenced. */
__attribute__((unused))
static int idx_layer_alpha(int layer, int slot) { return idx_layer_base(layer) + 7 + slot; } /* slot 0..6 */
static int idx_layer_gamma(int layer, int slot) { return idx_layer_base(layer) + 14 + slot; } /* slot 0..3 */
static int idx_final_norm(int layer_count) { return 1 + layer_count * TENSORS_PER_LAYER; }
static int idx_lm_head(int layer_count) { return 2 + layer_count * TENSORS_PER_LAYER; }

int bitnet_weights_load(
    const char* path,
    bitnet_weights_t* weights,
    bitnet_weights_loaded_t* handle)
{
    memset(handle, 0, sizeof(*handle));
    memset(weights, 0, sizeof(*weights));

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[bitnet_weights] cannot open %s: %s\n", path, strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "[bitnet_weights] fstat failed: %s\n", strerror(errno));
        close(fd); return 2;
    }
    size_t size = (size_t)st.st_size;

    void* base = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "[bitnet_weights] mmap failed: %s\n", strerror(errno));
        close(fd); return 3;
    }
    handle->base = base;
    handle->size = size;
    handle->fd = fd;

    /* Parse header. */
    const uint8_t* p = (const uint8_t*)base;
    if (size < 16) {
        fprintf(stderr, "[bitnet_weights] file too small (%zu bytes)\n", size);
        bitnet_weights_unload(handle); return 4;
    }
    if (memcmp(p, "M4T1", 4) != 0) {
        fprintf(stderr, "[bitnet_weights] bad magic (expected 'M4T1')\n");
        bitnet_weights_unload(handle); return 5;
    }
    uint32_t version, lm_head_tied, n_tensors;
    memcpy(&version,      p + 4,  4);
    memcpy(&lm_head_tied, p + 8,  4);
    memcpy(&n_tensors,    p + 12, 4);
    if (version != 1) {
        fprintf(stderr, "[bitnet_weights] unsupported version %u\n", version);
        bitnet_weights_unload(handle); return 6;
    }
    handle->lm_head_tied = (int)lm_head_tied;
    handle->n_tensors = (int32_t)n_tensors;

    /* Block exponents, offsets, sizes. */
    const int32_t*  block_exps = (const int32_t*) (p + 16);
    const uint64_t* offsets    = (const uint64_t*)(p + 16 + 4 * n_tensors);
    const uint64_t* sizes      = (const uint64_t*)(p + 16 + 4 * n_tensors + 8 * n_tensors);
    handle->block_exps = block_exps;

    size_t header_end = 16 + 20ULL * n_tensors;
    if (size < header_end) {
        fprintf(stderr, "[bitnet_weights] header truncated\n");
        bitnet_weights_unload(handle); return 7;
    }

    /* Helper: tensor pointer from index. */
    #define TENSOR_PTR(idx, type) \
        ((const type*)((const uint8_t*)base + offsets[(idx)]))

    /* Determine layers actually present. The script may have produced
     * a partial blob (--layers N). Compute by inspecting tensor count:
     *   n_tensors = 1 (embedding) + L*18 + 1 (final norm) + (0 or 1) (lm_head)
     * → L = (n_tensors - 1 - 1 - (1 - lm_head_tied)) / 18 */
    int extra = (lm_head_tied ? 1 : 2);  /* 1 = just final_norm; 2 = + lm_head */
    int layers_present = ((int)n_tensors - 1 - extra) / TENSORS_PER_LAYER;
    if (layers_present < 0 || layers_present > BITNET_NUM_LAYERS) {
        fprintf(stderr, "[bitnet_weights] cannot infer layer count from n_tensors=%u, "
                "lm_head_tied=%u\n", n_tensors, lm_head_tied);
        bitnet_weights_unload(handle); return 8;
    }
    fprintf(stderr, "[bitnet_weights] loaded %d layers (lm_head %s)\n",
            layers_present, lm_head_tied ? "tied to embedding" : "untied");

    /* Embedding. */
    weights->embedding = TENSOR_PTR(idx_embedding(), m4t_mtfp_t);

    /* Per-layer pointers. Layers beyond `layers_present` stay NULL. */
    static const int proj_to_field_offset_w[7] = { 0, 1, 2, 3, 4, 5, 6 };
    (void)proj_to_field_offset_w;
    for (int l = 0; l < layers_present; l++) {
        bitnet_layer_weights_t* lw = &weights->layers[l];
        lw->w_q     = TENSOR_PTR(idx_layer_w(l, 0), uint8_t);
        lw->w_k     = TENSOR_PTR(idx_layer_w(l, 1), uint8_t);
        lw->w_v     = TENSOR_PTR(idx_layer_w(l, 2), uint8_t);
        lw->w_o     = TENSOR_PTR(idx_layer_w(l, 3), uint8_t);
        lw->w_gate  = TENSOR_PTR(idx_layer_w(l, 4), uint8_t);
        lw->w_up    = TENSOR_PTR(idx_layer_w(l, 5), uint8_t);
        lw->w_down  = TENSOR_PTR(idx_layer_w(l, 6), uint8_t);
        /* α scales (slots 7..13) — not stored on bitnet_layer_weights_t in
         * its current shape; the harness fetches them via block_exps and
         * tensor index when applying scale-vector multiply. Future
         * cleanup: extend bitnet_layer_weights_t with α pointers. */
        /* γ vectors (slots 14..17). */
        lw->gamma_input_norm     = TENSOR_PTR(idx_layer_gamma(l, 0), m4t_mtfp_t);
        lw->gamma_post_attn_norm = TENSOR_PTR(idx_layer_gamma(l, 1), m4t_mtfp_t);
        lw->gamma_attn_sub_norm  = TENSOR_PTR(idx_layer_gamma(l, 2), m4t_mtfp_t);
        lw->gamma_ffn_sub_norm   = TENSOR_PTR(idx_layer_gamma(l, 3), m4t_mtfp_t);
    }

    /* Final norm. */
    weights->gamma_final_norm = TENSOR_PTR(idx_final_norm(layers_present), m4t_mtfp_t);

    /* LM head: tied means reuse embedding. Else point to the loaded buffer. */
    if (lm_head_tied) {
        weights->lm_head = weights->embedding;
    } else {
        weights->lm_head = TENSOR_PTR(idx_lm_head(layers_present), m4t_mtfp_t);
    }

    /* Sanity: spot-check the first tensor's offset vs header_end. */
    if (offsets[0] != header_end) {
        fprintf(stderr, "[bitnet_weights] WARN: first offset %llu != header_end %zu\n",
                (unsigned long long)offsets[0], header_end);
    }
    /* Spot-check: final offset+size must not exceed file size. */
    uint64_t last_end = offsets[n_tensors - 1] + sizes[n_tensors - 1];
    if (last_end > size) {
        fprintf(stderr, "[bitnet_weights] last tensor extends past file end "
                "(%llu > %zu)\n", (unsigned long long)last_end, size);
        bitnet_weights_unload(handle); return 9;
    }

    return 0;
    #undef TENSOR_PTR
}

void bitnet_weights_unload(bitnet_weights_loaded_t* handle) {
    if (handle->base && handle->base != MAP_FAILED) {
        munmap(handle->base, handle->size);
    }
    if (handle->fd > 0) close(handle->fd);
    memset(handle, 0, sizeof(*handle));
}

int32_t bitnet_weights_block_exp(const bitnet_weights_loaded_t* h, int i) {
    if (!h || !h->block_exps || i < 0 || i >= h->n_tensors) return 0;
    return h->block_exps[i];
}
