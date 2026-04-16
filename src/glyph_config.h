/*
 * glyph_config.h — hyperparameter struct + CLI argument parsing for
 * routed k-NN consumer tools.
 *
 * The config struct captures every hyperparameter that previously
 * required source edits. CLI parsing is a minimal hand-written long-
 * option loop (no getopt_long dependency) so tools stay portable.
 */

#ifndef GLYPH_CONFIG_H
#define GLYPH_CONFIG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* data_dir;       /* MNIST IDX directory                     */
    int         n_proj;         /* signature dimension in trits            */
    double      density;        /* tau calibration density                 */
    int         m_max;          /* number of bucket tables                 */
    int         max_radius;     /* ternary multi-probe radius budget       */
    int         min_cands;      /* per-table early-stop candidate thresh.  */
    int         max_union;      /* cap on per-query candidate union size   */
    uint32_t    base_seed[4];   /* seed quadruple for table 0              */
    const char* mode;           /* "oracle" or "full"                      */
    int         verbose;

    /* Optional: restrict the M sweep to a single value. 0 = full sweep. */
    int         single_m;

    /* Set via --no_deskew. Default 0 = apply integer-moment shear
     * deskewing after loading the dataset. The deskew pass is optimal
     * for MNIST digits (aligns the vertical stroke axis) but may
     * distort datasets without a canonical shear axis (e.g. clothing
     * items in Fashion-MNIST). */
    int         no_deskew;

    /* SUM resolver implementation selector. Options:
     *   "scalar"       — general-purpose post-Fix-1 scalar path
     *   "neon4"        — NEON-batched variant for sig_bytes=4
     *   "voteweighted" — score = sum_dist / (1 + votes)
     *   "radiusaware"  — score = sum_dist + lambda × min_radius
     */
    const char* resolver_sum;

    /* Radius-aware SUM penalty coefficient. Only consulted when
     * resolver_sum == "radiusaware". Default 8 (one radius step
     * costs 8 Hamming-distance units, roughly equivalent to a
     * single-byte popcount mismatch). Larger values shrink the
     * ranking toward strict radius-0 preference; smaller values
     * reduce toward scalar SUM. */
    int         radius_lambda;

    /* Routed k-NN: top-K candidates scored by summed Hamming distance,
     * rank-weighted majority vote. Only consulted when
     * resolver_sum == "knn". Default 5. */
    int         knn_k;

    /* Density schedule across the M multi-tables.
     *   "fixed" (default) — every table uses --density.
     *   "mixed"           — round-robin over --density_triple.
     */
    const char* density_schedule;

    /* Density triple for --density_schedule mixed. Table m uses
     * density_triple[m % 3]. Default {0.25, 0.33, 0.40}. Only
     * consulted when density_schedule == "mixed". */
    double      density_triple[3];
} glyph_config_t;

/* Fill with project defaults (matches the values from Phase 3 run). */
void glyph_config_defaults(glyph_config_t* cfg);

/* Parse argv. Returns:
 *    0 on success
 *    1 on usage error (prints message to stderr)
 *   -1 when --help is requested (usage printed to stdout)
 */
int glyph_config_parse_argv(glyph_config_t* cfg, int argc, char** argv);

/* Print usage to stdout (or stderr on error). */
void glyph_config_print_usage(const char* progname);

#ifdef __cplusplus
}
#endif

#endif /* GLYPH_CONFIG_H */
