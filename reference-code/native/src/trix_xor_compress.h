/*
 * trix_xor_compress.h — XOR Superposition Signature Compression
 *
 * Lossless compression for ternary signatures using base + sparse deltas.
 * When signatures are similar (common early in training), most deltas are
 * very sparse, giving significant memory savings.
 *
 * Storage: one base signature (centroid) + per-signature sparse deltas
 * (positions + values where sig differs from base).
 *
 * Also provides packed 2-bit ternary utilities and XOR+POPCNT distance
 * computations at the batch level.
 *
 * Ported from xor_superposition.py.
 */

#ifndef TRIX_XOR_COMPRESS_H
#define TRIX_XOR_COMPRESS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Sparse Delta ── */

typedef struct {
    int16_t* positions;   /* indices where sig differs from base */
    int8_t*  values;      /* ternary values at those positions */
    int      nnz;         /* number of differing positions */
} TrixSparseDelta;

/* ── Compression Stats ── */

typedef struct {
    int   original_bytes;
    int   compressed_bytes;
    float compression_ratio;
    float mean_delta_sparsity;  /* mean(nnz / dim) across signatures */
    float max_delta_sparsity;
    int   num_signatures;
    int   dim;
} TrixCompressionStats;

/* ── Compressed Signatures ── */

typedef struct {
    int8_t*         base;       /* [dim] base signature {-1, 0, +1} */
    uint8_t*        base_packed;/* [packed_dim] 2-bit packed base */
    TrixSparseDelta* deltas;    /* [num_sigs] sparse deltas from base */
    int             num_sigs;
    int             dim;
    int             packed_dim; /* (dim + 3) / 4 */
} TrixCompressedSigs;

/* ── Lifecycle ── */

/* Compress an array of ternary signatures.
 * sigs: [num_sigs * dim] ternary values {-1, 0, +1} as float.
 * Returns NULL on failure. */
TrixCompressedSigs* trix_xor_compress(const float* sigs, int num_sigs, int dim);

/* Decompress a single signature by index. out must be [dim]. */
void trix_xor_decompress(const TrixCompressedSigs* cs, int index, float* out);

/* Decompress all signatures. out must be [num_sigs * dim]. */
void trix_xor_decompress_all(const TrixCompressedSigs* cs, float* out);

void trix_xor_compress_destroy(TrixCompressedSigs* cs);

/* Get compression statistics. */
TrixCompressionStats trix_xor_compress_stats(const TrixCompressedSigs* cs);

/* ── 2-Bit Packing ──
 *
 * Encoding per ternary value:
 *   0  -> 00
 *  +1  -> 01
 *  -1  -> 10
 *
 * 4 values per byte: byte = v0 | (v1<<2) | (v2<<4) | (v3<<6) */

/* Pack ternary float array to 2-bit uint8.
 * ternary: [n] values in {-1, 0, +1} as float.
 * packed: [(n+3)/4] output bytes. */
void trix_xor_pack(const float* ternary, uint8_t* packed, int n);

/* Unpack 2-bit uint8 back to ternary float.
 * packed: [packed_n] input bytes.
 * out: [dim] output floats. */
void trix_xor_unpack(const uint8_t* packed, float* out, int dim);

/* ── XOR+POPCNT Distance ── */

/* Compute XOR+POPCNT distance between two packed signature vectors.
 * Returns total bit distance (sum of popcounts of XOR bytes). */
int trix_xor_distance(const uint8_t* a, const uint8_t* b, int packed_dim);

/* Compute distances from one query to multiple signatures.
 * query: [packed_dim], sigs: [num_sigs * packed_dim].
 * dists_out: [num_sigs] output distances.
 * If mask_query_zeros is true, positions where query has 00 contribute 0. */
void trix_xor_distances(const uint8_t* query, const uint8_t* sigs,
                        int num_sigs, int packed_dim,
                        int* dists_out, int mask_query_zeros);

/* Find the nearest signature by XOR+POPCNT distance.
 * Returns index of nearest signature. */
int trix_xor_nearest(const uint8_t* query, const uint8_t* sigs,
                     int num_sigs, int packed_dim,
                     int mask_query_zeros);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_XOR_COMPRESS_H */
