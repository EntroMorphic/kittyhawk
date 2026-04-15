/*
 * glyph_multiprobe.c — ternary Hamming multi-probe enumeration.
 */

#include "glyph_multiprobe.h"

#include <string.h>

int8_t glyph_read_trit(const uint8_t* sig, int j) {
    uint8_t code = (sig[j >> 2] >> ((j & 3) * 2)) & 0x3u;
    return (code == 0x01u) ? 1 : (code == 0x02u) ? -1 : 0;
}

void glyph_write_trit(uint8_t* sig, int j, int8_t t) {
    uint8_t code = (t == 1) ? 0x01u : (t == -1) ? 0x02u : 0x00u;
    int shift = (j & 3) * 2;
    sig[j >> 2] = (uint8_t)((sig[j >> 2] & ~(0x3u << shift)) | (code << shift));
}

int glyph_multiprobe_enumerate(
    const uint8_t* query_sig,
    int n_proj,
    int sig_bytes,
    int radius,
    uint8_t* scratch,
    glyph_probe_cb cb,
    void* ctx)
{
    if (radius == 0) {
        memcpy(scratch, query_sig, (size_t)sig_bytes);
        return cb(scratch, ctx);
    }
    if (radius == 1) {
        for (int j = 0; j < n_proj; j++) {
            int8_t orig = glyph_read_trit(query_sig, j);
            memcpy(scratch, query_sig, (size_t)sig_bytes);
            if (orig == 0) {
                /* 0 → +1 (cost 1) */
                glyph_write_trit(scratch, j, +1);
                if (cb(scratch, ctx)) return 1;
                /* 0 → -1 (cost 1) */
                memcpy(scratch, query_sig, (size_t)sig_bytes);
                glyph_write_trit(scratch, j, -1);
                if (cb(scratch, ctx)) return 1;
            } else {
                /* ±1 → 0 (cost 1); the sign flip would be cost 2 */
                glyph_write_trit(scratch, j, 0);
                if (cb(scratch, ctx)) return 1;
            }
        }
        return 0;
    }
    if (radius == 2) {
        /* (a) single sign-flip at each non-zero position: cost 2. */
        for (int j = 0; j < n_proj; j++) {
            int8_t orig = glyph_read_trit(query_sig, j);
            if (orig == 0) continue;
            memcpy(scratch, query_sig, (size_t)sig_bytes);
            glyph_write_trit(scratch, j, (int8_t)(-orig));
            if (cb(scratch, ctx)) return 1;
        }
        /* (b) two cost-1 moves on distinct positions: cost 1 + 1 = 2. */
        for (int j = 0; j < n_proj; j++) {
            int8_t oj = glyph_read_trit(query_sig, j);
            for (int k = j + 1; k < n_proj; k++) {
                int8_t ok = glyph_read_trit(query_sig, k);
                int8_t tj[2]; int nj = 0;
                int8_t tk[2]; int nk = 0;
                if (oj == 0) { tj[nj++] = +1; tj[nj++] = -1; } else { tj[nj++] = 0; }
                if (ok == 0) { tk[nk++] = +1; tk[nk++] = -1; } else { tk[nk++] = 0; }
                for (int a = 0; a < nj; a++) {
                    for (int b = 0; b < nk; b++) {
                        memcpy(scratch, query_sig, (size_t)sig_bytes);
                        glyph_write_trit(scratch, j, tj[a]);
                        glyph_write_trit(scratch, k, tk[b]);
                        if (cb(scratch, ctx)) return 1;
                    }
                }
            }
        }
        return 0;
    }
    return 0;
}
