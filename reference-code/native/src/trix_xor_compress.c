/*
 * trix_xor_compress.c — XOR Superposition Signature Compression
 *
 * Ported from xor_superposition.py.
 */

#include "trix_xor_compress.h"
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define TRIX_HAS_NEON 1
#else
#define TRIX_HAS_NEON 0
#endif

/* ── Popcount LUT for scalar path ── */

static const uint8_t popcount_lut[256] = {
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,
};

/* ══════════════════════════════════════════════════════════════════════
 * 2-Bit Packing
 * ══════════════════════════════════════════════════════════════════════ */

static inline uint8_t encode_ternary(float v) {
    if (v > 0.5f) return 0x01;   /* +1 */
    if (v < -0.5f) return 0x02;  /* -1 */
    return 0x00;                  /*  0 */
}

void trix_xor_pack(const float* ternary, uint8_t* packed, int n) {
    int packed_dim = (n + 3) / 4;
    memset(packed, 0, (size_t)packed_dim);

    for (int i = 0; i < n; i++) {
        uint8_t code = encode_ternary(ternary[i]);
        packed[i / 4] |= (uint8_t)(code << ((i % 4) * 2));
    }
}

void trix_xor_unpack(const uint8_t* packed, float* out, int dim) {
    for (int i = 0; i < dim; i++) {
        uint8_t code = (packed[i / 4] >> ((i % 4) * 2)) & 0x03;
        if (code == 0x01) out[i] = 1.0f;
        else if (code == 0x02) out[i] = -1.0f;
        else out[i] = 0.0f;
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * XOR+POPCNT Distance
 * ══════════════════════════════════════════════════════════════════════ */

int trix_xor_distance(const uint8_t* a, const uint8_t* b, int packed_dim) {
    int dist = 0;
#if TRIX_HAS_NEON
    /* Process 16 bytes at a time. Reduce every 31 iterations to avoid
     * uint8 overflow (max 8 bits per vcntq_u8 result, 31*8 = 248 < 255). */
    int i = 0;
    while (i + 16 <= packed_dim) {
        uint8x16_t vacc = vdupq_n_u8(0);
        int chunk_end = i + 31 * 16;
        if (chunk_end > packed_dim) chunk_end = packed_dim;
        for (; i + 16 <= chunk_end; i += 16) {
            uint8x16_t va = vld1q_u8(a + i);
            uint8x16_t vb = vld1q_u8(b + i);
            uint8x16_t vxor = veorq_u8(va, vb);
            vacc = vaddq_u8(vacc, vcntq_u8(vxor));
        }
        dist += vaddlvq_u8(vacc);
    }
    for (; i < packed_dim; i++) {
        dist += popcount_lut[a[i] ^ b[i]];
    }
#else
    for (int i = 0; i < packed_dim; i++) {
        dist += popcount_lut[a[i] ^ b[i]];
    }
#endif
    return dist;
}

/* Build mask for non-zero 2-bit groups in query.
 * For each group: 00 -> mask 00, anything else -> mask 11. */
static void build_nonzero_mask(const uint8_t* query, uint8_t* mask, int packed_dim) {
    for (int i = 0; i < packed_dim; i++) {
        uint8_t q = query[i];
        uint8_t m = 0;
        if (q & 0x03) m |= 0x03;
        if (q & 0x0C) m |= 0x0C;
        if (q & 0x30) m |= 0x30;
        if (q & 0xC0) m |= 0xC0;
        mask[i] = m;
    }
}

void trix_xor_distances(const uint8_t* query, const uint8_t* sigs,
                        int num_sigs, int packed_dim,
                        int* dists_out, int mask_query_zeros)
{
    uint8_t* mask = NULL;
    if (mask_query_zeros) {
        mask = malloc((size_t)packed_dim);
        if (!mask) {
            /* Fallback: no masking */
            mask_query_zeros = 0;
        } else {
            build_nonzero_mask(query, mask, packed_dim);
        }
    }

    for (int s = 0; s < num_sigs; s++) {
        const uint8_t* sig = sigs + s * packed_dim;
        int dist = 0;

        if (mask_query_zeros && mask) {
            for (int i = 0; i < packed_dim; i++) {
                dist += popcount_lut[(query[i] ^ sig[i]) & mask[i]];
            }
        } else {
            dist = trix_xor_distance(query, sig, packed_dim);
        }

        dists_out[s] = dist;
    }

    free(mask);
}

int trix_xor_nearest(const uint8_t* query, const uint8_t* sigs,
                     int num_sigs, int packed_dim,
                     int mask_query_zeros)
{
    if (num_sigs <= 0) return -1;

    /* Avoid heap allocation for small num_sigs */
    int stack_dists[256];
    int* dists = (num_sigs <= 256) ? stack_dists
                                   : malloc((size_t)num_sigs * sizeof(int));
    if (!dists) return -1;

    trix_xor_distances(query, sigs, num_sigs, packed_dim, dists,
                       mask_query_zeros);

    int best = 0;
    int best_dist = dists[0];
    for (int s = 1; s < num_sigs; s++) {
        if (dists[s] < best_dist) {
            best_dist = dists[s];
            best = s;
        }
    }

    if (num_sigs > 256) free(dists);
    return best;
}

/* ══════════════════════════════════════════════════════════════════════
 * Compression
 * ══════════════════════════════════════════════════════════════════════ */

static int8_t float_to_ternary(float v) {
    if (v > 0.5f) return 1;
    if (v < -0.5f) return -1;
    return 0;
}

TrixCompressedSigs* trix_xor_compress(const float* sigs, int num_sigs, int dim) {
    if (!sigs || num_sigs <= 0 || dim <= 0) return NULL;

    TrixCompressedSigs* cs = calloc(1, sizeof(TrixCompressedSigs));
    if (!cs) return NULL;

    cs->num_sigs = num_sigs;
    cs->dim = dim;
    cs->packed_dim = (dim + 3) / 4;

    /* Compute base signature as sign(sum of all signatures per dimension).
     * This is the centroid in ternary space. */
    cs->base = calloc((size_t)dim, sizeof(int8_t));
    if (!cs->base) { trix_xor_compress_destroy(cs); return NULL; }

    float* col_sums = calloc((size_t)dim, sizeof(float));
    if (!col_sums) { trix_xor_compress_destroy(cs); return NULL; }

    for (int s = 0; s < num_sigs; s++) {
        for (int d = 0; d < dim; d++) {
            col_sums[d] += float_to_ternary(sigs[s * dim + d]);
        }
    }
    for (int d = 0; d < dim; d++) {
        if (col_sums[d] > 0.0f)       cs->base[d] = 1;
        else if (col_sums[d] < 0.0f)  cs->base[d] = -1;
        else                           cs->base[d] = 0;
    }
    free(col_sums);

    /* Pack base */
    cs->base_packed = calloc((size_t)cs->packed_dim, sizeof(uint8_t));
    if (!cs->base_packed) { trix_xor_compress_destroy(cs); return NULL; }
    {
        float* base_f = malloc((size_t)dim * sizeof(float));
        if (!base_f) { trix_xor_compress_destroy(cs); return NULL; }
        for (int d = 0; d < dim; d++) base_f[d] = (float)cs->base[d];
        trix_xor_pack(base_f, cs->base_packed, dim);
        free(base_f);
    }

    /* Compute sparse deltas */
    cs->deltas = calloc((size_t)num_sigs, sizeof(TrixSparseDelta));
    if (!cs->deltas) { trix_xor_compress_destroy(cs); return NULL; }

    for (int s = 0; s < num_sigs; s++) {
        /* First pass: count differences */
        int nnz = 0;
        for (int d = 0; d < dim; d++) {
            int8_t val = float_to_ternary(sigs[s * dim + d]);
            if (val != cs->base[d]) nnz++;
        }

        cs->deltas[s].nnz = nnz;
        if (nnz == 0) {
            cs->deltas[s].positions = NULL;
            cs->deltas[s].values = NULL;
            continue;
        }

        cs->deltas[s].positions = malloc((size_t)nnz * sizeof(int16_t));
        cs->deltas[s].values    = malloc((size_t)nnz * sizeof(int8_t));
        if (!cs->deltas[s].positions || !cs->deltas[s].values) {
            trix_xor_compress_destroy(cs);
            return NULL;
        }

        /* Second pass: fill */
        int idx = 0;
        for (int d = 0; d < dim; d++) {
            int8_t val = float_to_ternary(sigs[s * dim + d]);
            if (val != cs->base[d]) {
                cs->deltas[s].positions[idx] = (int16_t)d;
                cs->deltas[s].values[idx]    = val;
                idx++;
            }
        }
    }

    return cs;
}

void trix_xor_decompress(const TrixCompressedSigs* cs, int index, float* out) {
    if (!cs || !out || index < 0 || index >= cs->num_sigs) return;

    int dim = cs->dim;

    /* Start from base */
    for (int d = 0; d < dim; d++) {
        out[d] = (float)cs->base[d];
    }

    /* Apply delta */
    TrixSparseDelta* delta = &cs->deltas[index];
    for (int i = 0; i < delta->nnz; i++) {
        int pos = delta->positions[i];
        if (pos >= 0 && pos < dim) {
            out[pos] = (float)delta->values[i];
        }
    }
}

void trix_xor_decompress_all(const TrixCompressedSigs* cs, float* out) {
    if (!cs || !out) return;
    for (int s = 0; s < cs->num_sigs; s++) {
        trix_xor_decompress(cs, s, out + s * cs->dim);
    }
}

void trix_xor_compress_destroy(TrixCompressedSigs* cs) {
    if (!cs) return;
    free(cs->base);
    free(cs->base_packed);
    if (cs->deltas) {
        for (int s = 0; s < cs->num_sigs; s++) {
            free(cs->deltas[s].positions);
            free(cs->deltas[s].values);
        }
        free(cs->deltas);
    }
    free(cs);
}

TrixCompressionStats trix_xor_compress_stats(const TrixCompressedSigs* cs) {
    TrixCompressionStats stats = {0};
    if (!cs) return stats;

    stats.num_signatures = cs->num_sigs;
    stats.dim = cs->dim;

    /* Original: num_sigs * dim * sizeof(float) */
    stats.original_bytes = cs->num_sigs * cs->dim * 4;

    /* Compressed: base_packed + sum of (3 bytes per delta position/value pair) */
    int base_bytes = cs->packed_dim;
    int delta_bytes = 0;
    float max_sparsity = 0.0f;
    float sum_sparsity = 0.0f;

    for (int s = 0; s < cs->num_sigs; s++) {
        int nnz = cs->deltas[s].nnz;
        /* 2 bytes per position + 1 byte per value */
        delta_bytes += nnz * 3;

        float sp = (cs->dim > 0) ? (float)nnz / (float)cs->dim : 0.0f;
        sum_sparsity += sp;
        if (sp > max_sparsity) max_sparsity = sp;
    }

    stats.compressed_bytes = base_bytes + delta_bytes;
    stats.compression_ratio = (stats.compressed_bytes > 0)
        ? (float)stats.original_bytes / (float)stats.compressed_bytes
        : 0.0f;
    stats.mean_delta_sparsity = (cs->num_sigs > 0)
        ? sum_sparsity / (float)cs->num_sigs
        : 0.0f;
    stats.max_delta_sparsity = max_sparsity;

    return stats;
}
