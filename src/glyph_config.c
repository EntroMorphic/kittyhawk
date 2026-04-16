/*
 * glyph_config.c — CLI argument parsing for Glyph consumer tools.
 *
 * Hand-written long-option loop. Supports:
 *   --data <path>           default: ./data/mnist
 *   --n_proj <int>          default: 16
 *   --density <float>       default: 0.33
 *   --m_max <int>           default: 64
 *   --max_radius <int>      default: 2
 *   --min_cands <int>       default: 50
 *   --max_union <int>       default: 16384
 *   --base_seed <a,b,c,d>   default: 42,123,456,789
 *   --mode <str>            oracle | full      default: oracle
 *   --single_m <int>        default: 0 (full sweep)
 *   --verbose               default: off
 *   --help                  print usage and exit
 */

#include "glyph_config.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void glyph_config_defaults(glyph_config_t* cfg) {
    cfg->data_dir    = "./data/mnist";
    cfg->n_proj      = 16;
    cfg->density     = 0.33;
    cfg->m_max       = 64;
    cfg->max_radius  = 2;
    cfg->min_cands   = 50;
    cfg->max_union   = 16384;
    cfg->base_seed[0] = 42;
    cfg->base_seed[1] = 123;
    cfg->base_seed[2] = 456;
    cfg->base_seed[3] = 789;
    cfg->mode        = "oracle";
    cfg->verbose     = 0;
    cfg->single_m    = 0;
    cfg->no_deskew   = 0;
    cfg->resolver_sum = "scalar";
    cfg->radius_lambda = 8;
}

void glyph_config_print_usage(const char* progname) {
    printf(
        "Usage: %s [options]\n\n"
        "Hyperparameter options (all values are optional; defaults in brackets):\n"
        "  --data <path>           MNIST IDX directory [./data/mnist]\n"
        "  --n_proj <int>          signature dimension in trits [16]\n"
        "  --density <float>       tau calibration density in (0,1) [0.33]\n"
        "  --m_max <int>           number of bucket tables [64]\n"
        "  --max_radius <int>      ternary multi-probe radius budget in {0,1,2} [2]\n"
        "  --min_cands <int>       per-table candidate early-stop threshold [50]\n"
        "  --max_union <int>       cap on per-query candidate union size [16384]\n"
        "  --base_seed <a,b,c,d>   seed quadruple for table 0 ONLY [42,123,456,789]\n"
        "                          (tables m>=1 use a fixed deterministic derivation\n"
        "                           independent of base_seed so that the default run\n"
        "                           reproduces the Phase 3 measurement byte-for-byte)\n"
        "  --mode <str>            'oracle' (default) or 'full' (+ resolvers)\n"
        "  --single_m <int>        restrict sweep to a single M value [0 = full sweep]\n"
        "  --no_deskew             skip the integer image-moment shear preprocessing step\n"
        "                          (default: deskew; recommended ON for MNIST digits,\n"
        "                           OFF for datasets without a canonical shear axis like\n"
        "                           Fashion-MNIST clothing or CIFAR-10 natural images)\n"
        "  --resolver_sum <mode>   SUM resolver implementation. Options:\n"
        "                            'scalar' (default) — general-purpose post-Fix-1 scalar\n"
        "                                path using builtin popcount on 4-byte sigs.\n"
        "                            'neon4' — NEON-batched variant for sig_bytes=4, 4\n"
        "                                candidates per 16-byte vector. Bit-exact equivalent\n"
        "                                to 'scalar' — chosen for speed only.\n"
        "                            'voteweighted' — scores each candidate as\n"
        "                                sum_dist / (1 + filter_votes[c]), folding the\n"
        "                                filter-stage vote count into the resolver ranking.\n"
        "                                DIFFERENT algorithm (not bit-exact to 'scalar'),\n"
        "                                intended to recover the Fashion-MNIST resolver\n"
        "                                gap by preferring candidates that more tables\n"
        "                                agreed on.\n"
        "                            'radiusaware' — scores each candidate as\n"
        "                                sum_dist + lambda*min_radius[c], penalizing\n"
        "                                candidates only reachable via deep multi-probe\n"
        "                                expansion. lambda is --radius_lambda.\n"
        "  --radius_lambda <int>   radius penalty coefficient for --resolver_sum radiusaware\n"
        "                          (default 8; 0 reduces to scalar SUM; higher values\n"
        "                           shrink toward strict radius-0 preference)\n"
        "  --verbose               print extra diagnostic information\n"
        "  --help                  print this message and exit\n"
        "\n"
        "Notes:\n"
        "  * Numeric arguments reject non-numeric input. Out-of-range values are\n"
        "    reported with a diagnostic to stderr; no silent clamping.\n"
        "  * --base_seed must not be all zero; xoshiro requires a non-degenerate seed.\n"
        "\n"
        "Examples:\n"
        "  %s --data ~/mnist --mode full\n"
        "  %s --n_proj 16 --m_max 16 --mode full --single_m 16\n"
        "  %s --density 0.50 --min_cands 100 --max_radius 1\n",
        progname, progname, progname, progname);
}

/* Strict integer parse: whole arg must be a valid long, no trailing
 * garbage. Writes to *out on success, returns 0; returns 1 on any
 * parse error with a message to stderr naming the offending option. */
static int parse_int(const char* opt, const char* val, int* out) {
    if (!val || !*val) {
        fprintf(stderr, "glyph_config: %s requires an integer value\n", opt);
        return 1;
    }
    errno = 0;
    char* endp = NULL;
    long v = strtol(val, &endp, 10);
    if (errno != 0 || endp == val || (endp && *endp != '\0')) {
        fprintf(stderr, "glyph_config: %s got non-integer value '%s'\n", opt, val);
        return 1;
    }
    if (v < INT_MIN || v > INT_MAX) {
        fprintf(stderr, "glyph_config: %s value %ld out of int range\n", opt, v);
        return 1;
    }
    *out = (int)v;
    return 0;
}

/* Strict double parse. Same rules. */
static int parse_double(const char* opt, const char* val, double* out) {
    if (!val || !*val) {
        fprintf(stderr, "glyph_config: %s requires a numeric value\n", opt);
        return 1;
    }
    errno = 0;
    char* endp = NULL;
    double v = strtod(val, &endp);
    if (errno != 0 || endp == val || (endp && *endp != '\0')) {
        fprintf(stderr, "glyph_config: %s got non-numeric value '%s'\n", opt, val);
        return 1;
    }
    *out = v;
    return 0;
}

static int parse_seed_quad(const char* arg, uint32_t out[4]) {
    /* Accepts "a,b,c,d" with optional whitespace. */
    unsigned long v[4];
    int n = sscanf(arg, "%lu,%lu,%lu,%lu", &v[0], &v[1], &v[2], &v[3]);
    if (n != 4) return 1;
    for (int i = 0; i < 4; i++) out[i] = (uint32_t)v[i];
    return 0;
}

int glyph_config_parse_argv(glyph_config_t* cfg, int argc, char** argv) {
    glyph_config_defaults(cfg);

    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            glyph_config_print_usage(argv[0]);
            return -1;
        }
        if (strcmp(arg, "--verbose") == 0)   { cfg->verbose = 1;   continue; }
        if (strcmp(arg, "--no_deskew") == 0) { cfg->no_deskew = 1; continue; }

        /* All other options take a value in argv[i+1]. */
        if (i + 1 >= argc) {
            fprintf(stderr, "glyph_config: option %s requires a value\n", arg);
            return 1;
        }
        const char* val = argv[++i];

        if      (strcmp(arg, "--data")       == 0) cfg->data_dir   = val;
        else if (strcmp(arg, "--mode")       == 0) cfg->mode       = val;
        else if (strcmp(arg, "--resolver_sum") == 0) cfg->resolver_sum = val;
        else if (strcmp(arg, "--n_proj")     == 0) {
            if (parse_int(arg, val, &cfg->n_proj)) return 1;
        }
        else if (strcmp(arg, "--density")    == 0) {
            if (parse_double(arg, val, &cfg->density)) return 1;
        }
        else if (strcmp(arg, "--m_max")      == 0) {
            if (parse_int(arg, val, &cfg->m_max)) return 1;
        }
        else if (strcmp(arg, "--max_radius") == 0) {
            if (parse_int(arg, val, &cfg->max_radius)) return 1;
        }
        else if (strcmp(arg, "--min_cands")  == 0) {
            if (parse_int(arg, val, &cfg->min_cands)) return 1;
        }
        else if (strcmp(arg, "--max_union")  == 0) {
            if (parse_int(arg, val, &cfg->max_union)) return 1;
        }
        else if (strcmp(arg, "--single_m")   == 0) {
            if (parse_int(arg, val, &cfg->single_m)) return 1;
        }
        else if (strcmp(arg, "--radius_lambda") == 0) {
            if (parse_int(arg, val, &cfg->radius_lambda)) return 1;
        }
        else if (strcmp(arg, "--base_seed")  == 0) {
            if (parse_seed_quad(val, cfg->base_seed) != 0) {
                fprintf(stderr, "glyph_config: --base_seed expects 'a,b,c,d'\n");
                return 1;
            }
        }
        else {
            fprintf(stderr, "glyph_config: unknown option %s\n", arg);
            return 1;
        }
    }

    /* Additional validation beyond the range checks at the bottom. */
    if ((cfg->base_seed[0] | cfg->base_seed[1] |
         cfg->base_seed[2] | cfg->base_seed[3]) == 0u) {
        fprintf(stderr,
            "glyph_config: --base_seed must not be all zero "
            "(xoshiro128+ needs at least one non-zero state word)\n");
        return 1;
    }

    /* Validation. */
    if (cfg->n_proj <= 0 || cfg->n_proj > 64) {
        fprintf(stderr, "glyph_config: --n_proj out of range (1..64)\n");
        return 1;
    }
    if (cfg->density <= 0.0 || cfg->density >= 1.0) {
        fprintf(stderr, "glyph_config: --density must be in (0, 1)\n");
        return 1;
    }
    if (cfg->m_max <= 0 || cfg->m_max > 256) {
        fprintf(stderr, "glyph_config: --m_max out of range (1..256)\n");
        return 1;
    }
    if (cfg->max_radius < 0 || cfg->max_radius > 2) {
        fprintf(stderr, "glyph_config: --max_radius must be in {0, 1, 2}\n");
        return 1;
    }
    if (cfg->min_cands < 0) {
        fprintf(stderr, "glyph_config: --min_cands must be non-negative\n");
        return 1;
    }
    if (cfg->max_union < 1) {
        fprintf(stderr, "glyph_config: --max_union must be positive\n");
        return 1;
    }
    if (strcmp(cfg->mode, "oracle") != 0 && strcmp(cfg->mode, "full") != 0) {
        fprintf(stderr, "glyph_config: --mode must be 'oracle' or 'full'\n");
        return 1;
    }
    if (strcmp(cfg->resolver_sum, "scalar")       != 0 &&
        strcmp(cfg->resolver_sum, "neon4")        != 0 &&
        strcmp(cfg->resolver_sum, "voteweighted") != 0 &&
        strcmp(cfg->resolver_sum, "radiusaware")  != 0) {
        fprintf(stderr,
            "glyph_config: --resolver_sum must be 'scalar', 'neon4', "
            "'voteweighted', or 'radiusaware'\n");
        return 1;
    }
    if (cfg->radius_lambda < 0) {
        fprintf(stderr, "glyph_config: --radius_lambda must be non-negative\n");
        return 1;
    }
    return 0;
}
