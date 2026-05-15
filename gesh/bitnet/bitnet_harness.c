/*
 * gesh/bitnet/bitnet_harness.c — single-block forward-pass harness for
 * BitNet b1.58-2B-4T inference on the m4t substrate. Per work-unit 1
 * of the bitnet_phase1 LMM cycle.
 *
 * This file is the SKELETON committed at the start of work-unit 1.
 * It compiles cleanly, allocates scratch, lays out the forward pass
 * structure, and stubs out the parts that need (a) actual weights
 * loaded from disk and (b) per-layer activation dumping. Subsequent
 * work-unit-1 commits fill those in.
 *
 * Execution model:
 *   build/gesh/bitnet/bitnet_harness <weights_blob.bin> <input_token_id>
 *
 * Output:
 *   per-layer activations dumped to stdout in a documented binary
 *   format (work-unit 1 fills this in alongside the Python comparison
 *   driver).
 */

#include "bitnet_config.h"
#include "bitnet_stubs.h"
#include "bitnet_weights.h"

#include "m4t_ternary_matmul.h"
#include "m4t_mtfp.h"
#include "m4t_route.h"        /* Cycle 2 Phase 2.4: substrate-routed sparse attention */
#include "m4t_trit_pack.h"    /* M4T_TRIT_PACKED_BYTES for signature sizing */
#include "m4t_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>  /* clock_gettime for gen-loop timing (cost measurement) */

/* Attention-only time accumulator for cost measurement. Reset before each
 * generation loop, printed alongside gen_loop_seconds. Lets the
 * substrate-vs-dense comparison attribute wall-clock differences to the
 * attention block specifically vs other path differences. */
static double g_attn_seconds = 0.0;
#include <string.h>
#include <stdint.h>
#include <assert.h>

/* ── Cycle 2 (Part-B sparse attention experiment) — runtime-selectable
 *    attention mode. Default DENSE = current production behavior; sparse
 *    arms (RANDOM, ROUTED, ORACLE) exist for the experiment per
 *    journal/cycle2_design.md and journal/partB_experiments_synth.md.
 *
 *    Set BITNET_ATTN_MODE ∈ {dense,random,routed,oracle} (case-insensitive)
 *    Set BITNET_ATTN_K ∈ positive int (default = head_dim, i.e. equivalent
 *    to dense for sparse modes).
 *
 *    When env vars are unset: bit-exact match to the production path. */
typedef enum {
    BITNET_ATTN_DENSE = 0,
    BITNET_ATTN_RANDOM,
    BITNET_ATTN_ROUTED,
    BITNET_ATTN_ORACLE,
    BITNET_ATTN_POSRACLE,  /* TD-27 H2 test: top-k by SIGNED Q·K (positives win)
                            * vs oracle's top-k by |Q·K|. If posracle ≈ routed,
                            * the routed-vs-oracle gap is explained by oracle
                            * "wasting" budget on high-|negative-score| positions
                            * that softmax suppresses anyway. */
    BITNET_ATTN_HYBRID,    /* TRIT_ROUTING #3: two-stage. Stage 1: signature
                            * distance picks top-k1 candidates (cheap filter).
                            * Stage 2: signed Q·K picks top-k2 < k1 from the
                            * shortlist (precise refinement). Combines
                            * direction-awareness from signatures with score
                            * precision from posracle. k1 from BITNET_ATTN_K1
                            * env (default k1 = 4 × k2). */
} bitnet_attn_mode_t;

static bitnet_attn_mode_t g_attn_mode = BITNET_ATTN_DENSE;
static int                g_attn_k    = BITNET_HEAD_DIM;  /* trajectory knob */
static __attribute__((unused)) unsigned int g_attn_rng = 0xC0FFEE01u;  /* random-arm seed (used in Phase 2.2) */
static int64_t g_attn_fixed_tau = 0;  /* TD-27 follow-up #4: 0 = use per-Q 1/3-quantile;
                                       * positive = use this fixed tau in routed signature extraction.
                                       * Tests whether per-Q adaptiveness is load-bearing. */
static int g_attn_no_k_cache = 0;     /* TRIT_ROUTING #1 verification: when 1, skip
                                       * K-signature caching even if fixed tau set.
                                       * Used to verify cache bit-exactness vs uncached
                                       * fixed-tau path. */

static const char* bitnet_attn_mode_name(bitnet_attn_mode_t m) {
    switch (m) {
        case BITNET_ATTN_DENSE:    return "dense";
        case BITNET_ATTN_RANDOM:   return "random";
        case BITNET_ATTN_ROUTED:   return "routed";
        case BITNET_ATTN_ORACLE:   return "oracle";
        case BITNET_ATTN_POSRACLE: return "posracle";
        case BITNET_ATTN_HYBRID:   return "hybrid";
    }
    return "unknown";
}

static int g_attn_k1 = 0;  /* TRIT_ROUTING #3: hybrid stage-1 candidate count.
                            * 0 = use default (4 × g_attn_k). */

static void bitnet_attn_mode_init_from_env(void) {
    const char* m = getenv("BITNET_ATTN_MODE");
    if (m) {
        if      (!strcasecmp(m, "dense"))    g_attn_mode = BITNET_ATTN_DENSE;
        else if (!strcasecmp(m, "random"))   g_attn_mode = BITNET_ATTN_RANDOM;
        else if (!strcasecmp(m, "routed"))   g_attn_mode = BITNET_ATTN_ROUTED;
        else if (!strcasecmp(m, "oracle"))   g_attn_mode = BITNET_ATTN_ORACLE;
        else if (!strcasecmp(m, "posracle")) g_attn_mode = BITNET_ATTN_POSRACLE;
        else if (!strcasecmp(m, "hybrid"))   g_attn_mode = BITNET_ATTN_HYBRID;
        else {
            fprintf(stderr, "[harness] unknown BITNET_ATTN_MODE=%s, using dense\n", m);
        }
    }
    const char* k = getenv("BITNET_ATTN_K");
    if (k) {
        int v = atoi(k);
        if (v > 0 && v <= BITNET_HEAD_DIM * 32) g_attn_k = v;
        else fprintf(stderr, "[harness] bad BITNET_ATTN_K=%s, using %d\n", k, g_attn_k);
    }
    const char* t = getenv("BITNET_ATTN_TAU");
    if (t) {
        long v = atol(t);
        if (v > 0) g_attn_fixed_tau = (int64_t)v;
        else fprintf(stderr, "[harness] bad BITNET_ATTN_TAU=%s, ignoring\n", t);
    }
    if (getenv("BITNET_ATTN_NO_CACHE")) g_attn_no_k_cache = 1;
    const char* k1 = getenv("BITNET_ATTN_K1");
    if (k1) {
        int v = atoi(k1);
        if (v > 0 && v <= BITNET_HEAD_DIM * 32) g_attn_k1 = v;
    }
    if (g_attn_mode != BITNET_ATTN_DENSE) {
        fprintf(stderr, "[harness] sparse attention mode = %s, k = %d",
                bitnet_attn_mode_name(g_attn_mode), g_attn_k);
        if (g_attn_fixed_tau > 0)
            fprintf(stderr, ", fixed_tau = %lld", (long long)g_attn_fixed_tau);
        fprintf(stderr, "\n");
    }
}

/* ── TRIT_ROUTING #10: KV cache eviction. ──────────────────────────────
 * When `BITNET_KV_WINDOW=N` is set and `current_pos > N` for a layer's
 * cache, evict positions until alive_count = N. Policies:
 *   - none     : no-op (default)
 *   - fifo     : evict the oldest non-evicted position
 *   - random   : evict a uniformly-random non-evicted position
 *   - sigdist  : evict the position whose K-signature is most distant
 *                (popcount XOR with most-recent Q-sig, summed over
 *                kv_heads) from the recent Q direction. Substrate-routed
 *                eviction; the test #10 was designed to answer.
 *
 * Attention path masks evicted positions to a large negative score before
 * softmax so weight=0 and they don't contribute to V combine.
 *
 * sigdist requires the K-sig cache populated (the #1 work). When eviction
 * mode is sigdist and BITNET_ATTN_TAU is unset, a default tau of 5000 is
 * used (per the #4 finding that fixed tau is acceptable).
 */
typedef enum {
    BITNET_KV_EVICT_NONE = 0,
    BITNET_KV_EVICT_FIFO,
    BITNET_KV_EVICT_RANDOM,
    BITNET_KV_EVICT_SIGDIST,
    BITNET_KV_EVICT_QSIGDIST,  /* Plan B (2026-05-12): Q-aware eviction.
                                 * Uses L1(Q-sig per Q-head, K-sig) summed
                                 * over all Q-heads × KV-heads as the
                                 * eviction cost. Implements the operation
                                 * Phase ε measured (Q-K oracle) — distinct
                                 * from SIGDIST's K-K proxy. */
    BITNET_KV_EVICT_META,      /* Meta-routing prototype (2026-05-14): a
                                 * parameterized policy combining recency,
                                 * K-K distance, and Q-K distance with
                                 * trit weights {-1, 0, +1} per component
                                 * (BITNET_KV_EVICT_W_R/W_KK/W_QK env vars,
                                 * Python convention from meta_routing.py).
                                 * The four hand-coded policies (fifo,
                                 * sigdist, qsigdist, random) are recoverable
                                 * by setting the appropriate weights;
                                 * non-anchor combinations are what layer 3
                                 * is searching to validate. */
    BITNET_KV_EVICT_QSIG_FILTER, /* Integrative eviction (2026-05-14):
                                  * qsigdist with a K-K-similarity FILTER.
                                  * Among candidate slots, mark the K
                                  * most-similar to current K (lowest
                                  * KK_dist) as PROTECTED. Among
                                  * unprotected, evict argmax(QK_dist) —
                                  * same as qsigdist. Fallback: if all
                                  * candidates protected (window ≤ K+1),
                                  * use plain qsigdist. K=0 reproduces
                                  * qsigdist exactly.
                                  *
                                  * Parameter: BITNET_KV_EVICT_KK_PROTECT_K
                                  * (integer ≥ 0).
                                  *
                                  * Tests CONJUNCTIVE integration: the
                                  * linear-additive trit family can't
                                  * express "filter then rank."
                                  *
                                  * Workload guidance (N=100 closeout,
                                  * window=16, gen=24, single-seed):
                                  *
                                  *   workload     | recommended K | gain vs qsigdist
                                  *   tech/science |       1       | +9.21pp [+1.97, +18.42] ✓
                                  *   logic/reason |       1       | +3.06pp (trending)
                                  *   dialog/Q&A   |       1       | +2.43pp (trending)
                                  *   short fact   |       0       | -0.30pp (neutral)
                                  *   code-heavy   |       0       | -2.18pp (avoid K=1)
                                  *   long-form    |       0       | -3.33pp (avoid K=1)
                                  *
                                  * Per-prompt routing among policies
                                  * fails held-out CV (-2.42pp); the right
                                  * granularity is per-WORKLOAD config
                                  * chosen at deploy time, not per-prompt
                                  * routing.
                                  *
                                  * See journal/integrative_qsig_filter_2026-05-14.md
                                  * and journal/integrative_qsig_filter_lmm_2026-05-14.md
                                  * for full arc + LMM. */
} bitnet_kv_evict_mode_t;

static bitnet_kv_evict_mode_t g_kv_evict_mode = BITNET_KV_EVICT_NONE;
static int g_kv_window = 0;
static unsigned int g_kv_evict_rng = 0xC0FFEE10u;
/* TRIT_ROUTING #10 amendment: M-step running-mean direction proxy for
 * sigdist. M=1 reproduces original "current K-sig as direction" probe.
 * M>1 averages K vectors over last M alive positions per kv_head and
 * extracts signature from the mean. Per #10 red-team finding M5. */
static int g_kv_evict_m = 1;

/* Meta-routing trit weights (Python convention). Score(slot) =
 * w_r·age + w_kk·KK_sim + w_qk·QK_sim, evict slot with LOWEST score.
 * Set to {-1, 0, +1} via BITNET_KV_EVICT_W_R / _W_KK / _W_QK. The
 * existing modes correspond to specific weight tuples:
 *   fifo:     (-1,  0,  0)
 *   sigdist:  ( 0, +1,  0)
 *   qsigdist: ( 0,  0, +1)
 *   random:   ( 0,  0,  0)
 * Other combinations (e.g., (0, -1, +1)) are layer-3 predictions. */
static int g_kv_evict_w_r  = 0;
static int g_kv_evict_w_kk = 0;
static int g_kv_evict_w_qk = 0;

/* QSIG_FILTER: number of K-K-similar slots to PROTECT from eviction.
 * Set via BITNET_KV_EVICT_KK_PROTECT_K. K=0 reproduces qsigdist. */
static int g_kv_evict_kk_protect_k = 0;

/* Trit Lattice LSH FFN — phase η validation (2026-05-14). Dumps the
 * FFN-input activation (post-attention layernorm output, the input
 * to gate/up projections) per (prompt, position, layer) so the
 * synthetic LSH protocol can be replayed on real activations.
 *
 * Env vars:
 *   BITNET_DUMP_FFN_INPUTS_DIR  — output directory (created if missing)
 *   BITNET_DUMP_FFN_INPUTS_LAYERS — comma-separated layer ids, or "all"
 *   BITNET_DUMP_LABEL           — prompt label for filename prefix
 *
 * Format: raw binary, hidden_size × m4t_mtfp_t (int16), no header.
 * Python reads via np.fromfile(path, dtype=np.int16).
 *
 * Filename: {dir}/{label}_p{position}_l{layer}.bin
 *
 * Read-only on the inference path: no semantic effect, just I/O. */
#define BITNET_DUMP_MAX_LAYERS 64
static const char* g_dump_ffn_inputs_dir = NULL;
static int  g_dump_ffn_inputs_layer_mask[BITNET_DUMP_MAX_LAYERS] = {0};
static int  g_dump_ffn_inputs_any = 0;  /* 1 if any layer to dump */
static const char* g_dump_label = "unlabeled";

/* Routed FFN — Step 1 PoC of "routed compute throughout" arc (2026-05-14).
 *
 * Replaces the dense FFN compute (gate/up/relu²/mul/sub_norm/down) with a
 * routed lookup: input signature → bucket → atom-composition tile → output.
 *
 * Calibration (offline): build_lsh_dict.py reads dump corpus, fits per-layer
 * atom dictionary + per-bucket recipes, serializes to a binary dict file.
 *
 * Env vars:
 *   BITNET_FFN_LSH_DICT     — path to dict file
 *   BITNET_FFN_LSH_LAYERS   — comma-separated layer ids to route, or "all"
 *
 * The architecture is "fully routed" in the substrate-vision sense: the
 * dispatch is signature-based; the tile content is shared atoms (ternary)
 * + per-bucket recipe (sparse coefficients). No dense matmul in the FFN
 * compute path at routed layers.
 *
 * NOTE Step 1 PoC: this skips the dense FFN compute entirely at routed
 * layers; subsequent steps will integrate routed BitLinear projections,
 * routed attention, etc. See journal/path_forward_2026-05-15.md (TBD). */
typedef struct {
    uint32_t recipe_len;
    uint32_t* atom_idx;     /* recipe_len entries */
    double*   scale;        /* recipe_len entries */
} bitnet_lsh_recipe_t;

typedef struct {
    int32_t  layer_idx;
    int32_t* mu;            /* d_model entries (int32) */
    int8_t*  atoms;         /* m_atoms × d_model trits */
    /* Bucket recipes indexed by bucket_id ∈ [0, 3^k_lsh).
     * Sparse: most buckets unset (recipe_len = 0). */
    bitnet_lsh_recipe_t* by_bucket;
    uint32_t  num_possible_buckets; /* 3^k_lsh */
} bitnet_lsh_layer_t;

typedef struct {
    int loaded;
    uint32_t num_layers;
    uint32_t d_model;
    uint32_t k_lsh;
    uint32_t m_atoms;
    uint32_t k_recipe_max;
    int32_t  tau;
    /* Layer index → pointer into layers[] (NULL if not in dict) */
    bitnet_lsh_layer_t* by_layer_idx[BITNET_DUMP_MAX_LAYERS];
    bitnet_lsh_layer_t* layers;
} bitnet_lsh_dict_t;

static bitnet_lsh_dict_t g_lsh_dict = {0};
static int g_lsh_active_layers[BITNET_DUMP_MAX_LAYERS] = {0};
static int g_lsh_any_active = 0;

static const char* bitnet_kv_evict_mode_name(bitnet_kv_evict_mode_t m) {
    switch (m) {
        case BITNET_KV_EVICT_NONE:    return "none";
        case BITNET_KV_EVICT_FIFO:    return "fifo";
        case BITNET_KV_EVICT_RANDOM:  return "random";
        case BITNET_KV_EVICT_SIGDIST: return "sigdist";
        case BITNET_KV_EVICT_QSIGDIST: return "qsigdist";
        case BITNET_KV_EVICT_META:    return "meta";
        case BITNET_KV_EVICT_QSIG_FILTER: return "qsig_filter";
    }
    return "unknown";
}

static void bitnet_kv_evict_init_from_env(void) {
    const char* m = getenv("BITNET_KV_EVICT_MODE");
    if (m) {
        if      (!strcasecmp(m, "none"))    g_kv_evict_mode = BITNET_KV_EVICT_NONE;
        else if (!strcasecmp(m, "fifo"))    g_kv_evict_mode = BITNET_KV_EVICT_FIFO;
        else if (!strcasecmp(m, "random"))  g_kv_evict_mode = BITNET_KV_EVICT_RANDOM;
        else if (!strcasecmp(m, "sigdist")) g_kv_evict_mode = BITNET_KV_EVICT_SIGDIST;
        else if (!strcasecmp(m, "qsigdist")) g_kv_evict_mode = BITNET_KV_EVICT_QSIGDIST;
        else if (!strcasecmp(m, "meta"))    g_kv_evict_mode = BITNET_KV_EVICT_META;
        else if (!strcasecmp(m, "qsig_filter")) g_kv_evict_mode = BITNET_KV_EVICT_QSIG_FILTER;
        else fprintf(stderr, "[harness] unknown BITNET_KV_EVICT_MODE=%s, using none\n", m);
    }
    /* Meta-routing trit weights. Validated to {-1, 0, +1}; out-of-range
     * values are clamped to 0 with a warning. */
    const char* wr_s  = getenv("BITNET_KV_EVICT_W_R");
    const char* wkk_s = getenv("BITNET_KV_EVICT_W_KK");
    const char* wqk_s = getenv("BITNET_KV_EVICT_W_QK");
    if (wr_s)  { int v = atoi(wr_s);
                 if (v >= -1 && v <= 1) g_kv_evict_w_r = v;
                 else fprintf(stderr, "[harness] BITNET_KV_EVICT_W_R out of {-1,0,+1}, using 0\n"); }
    if (wkk_s) { int v = atoi(wkk_s);
                 if (v >= -1 && v <= 1) g_kv_evict_w_kk = v;
                 else fprintf(stderr, "[harness] BITNET_KV_EVICT_W_KK out of {-1,0,+1}, using 0\n"); }
    if (wqk_s) { int v = atoi(wqk_s);
                 if (v >= -1 && v <= 1) g_kv_evict_w_qk = v;
                 else fprintf(stderr, "[harness] BITNET_KV_EVICT_W_QK out of {-1,0,+1}, using 0\n"); }
    /* QSIG_FILTER: number of K-K-similar slots to PROTECT. */
    const char* kpk_s = getenv("BITNET_KV_EVICT_KK_PROTECT_K");
    if (kpk_s) { int v = atoi(kpk_s);
                 if (v >= 0) g_kv_evict_kk_protect_k = v;
                 else fprintf(stderr, "[harness] BITNET_KV_EVICT_KK_PROTECT_K must be ≥0, using 0\n"); }

    /* Routed FFN — Step 1 PoC. Load LSH dict + active-layer set. */
    const char* lsh_dict_path = getenv("BITNET_FFN_LSH_DICT");
    const char* lsh_layers_s  = getenv("BITNET_FFN_LSH_LAYERS");
    if (lsh_dict_path && lsh_layers_s) {
        FILE* df = fopen(lsh_dict_path, "rb");
        if (!df) {
            fprintf(stderr, "[harness] WARN: failed to open LSH dict %s\n", lsh_dict_path);
        } else {
            char magic[4];
            if (fread(magic, 1, 4, df) != 4 || memcmp(magic, "GLFF", 4) != 0) {
                fprintf(stderr, "[harness] WARN: LSH dict bad magic\n");
                fclose(df);
            } else {
                uint32_t hdr[6]; int32_t tau;
                fread(hdr, sizeof(uint32_t), 6, df);  /* version, num_layers, d_model, k_lsh, m_atoms, k_recipe_max */
                fread(&tau, sizeof(int32_t), 1, df);
                g_lsh_dict.num_layers = hdr[1];
                g_lsh_dict.d_model = hdr[2];
                g_lsh_dict.k_lsh = hdr[3];
                g_lsh_dict.m_atoms = hdr[4];
                g_lsh_dict.k_recipe_max = hdr[5];
                g_lsh_dict.tau = tau;
                if (g_lsh_dict.d_model != BITNET_HIDDEN_SIZE) {
                    fprintf(stderr, "[harness] WARN: LSH dict d_model %u != %d\n",
                            g_lsh_dict.d_model, BITNET_HIDDEN_SIZE);
                }
                /* 3^k_lsh — compute */
                uint32_t n_buckets_possible = 1;
                for (uint32_t i = 0; i < g_lsh_dict.k_lsh; i++) n_buckets_possible *= 3;
                g_lsh_dict.layers = (bitnet_lsh_layer_t*)calloc(g_lsh_dict.num_layers,
                                                                  sizeof(bitnet_lsh_layer_t));
                for (uint32_t li = 0; li < g_lsh_dict.num_layers; li++) {
                    uint32_t lh[2];
                    fread(lh, sizeof(uint32_t), 2, df);  /* layer_idx, num_buckets */
                    bitnet_lsh_layer_t* L = &g_lsh_dict.layers[li];
                    L->layer_idx = (int32_t)lh[0];
                    L->num_possible_buckets = n_buckets_possible;
                    L->mu = (int32_t*)malloc(g_lsh_dict.d_model * sizeof(int32_t));
                    L->atoms = (int8_t*)malloc((size_t)g_lsh_dict.m_atoms * g_lsh_dict.d_model);
                    L->by_bucket = (bitnet_lsh_recipe_t*)calloc(n_buckets_possible,
                                                                  sizeof(bitnet_lsh_recipe_t));
                    fread(L->mu, sizeof(int32_t), g_lsh_dict.d_model, df);
                    fread(L->atoms, 1, (size_t)g_lsh_dict.m_atoms * g_lsh_dict.d_model, df);
                    uint32_t num_buckets_in_dict = lh[1];
                    for (uint32_t bi = 0; bi < num_buckets_in_dict; bi++) {
                        uint32_t bh[2];
                        fread(bh, sizeof(uint32_t), 2, df);  /* bucket_id, recipe_len */
                        bitnet_lsh_recipe_t* R = &L->by_bucket[bh[0]];
                        R->recipe_len = bh[1];
                        if (R->recipe_len > 0) {
                            R->atom_idx = (uint32_t*)malloc(R->recipe_len * sizeof(uint32_t));
                            R->scale = (double*)malloc(R->recipe_len * sizeof(double));
                            fread(R->atom_idx, sizeof(uint32_t), R->recipe_len, df);
                            fread(R->scale, sizeof(double), R->recipe_len, df);
                        }
                    }
                    if (L->layer_idx >= 0 && L->layer_idx < BITNET_DUMP_MAX_LAYERS) {
                        g_lsh_dict.by_layer_idx[L->layer_idx] = L;
                    }
                }
                fclose(df);
                g_lsh_dict.loaded = 1;
                fprintf(stderr, "[harness] LSH dict loaded: %u layers, d_model=%u, "
                        "k_lsh=%u, m_atoms=%u, tau=%d\n",
                        g_lsh_dict.num_layers, g_lsh_dict.d_model,
                        g_lsh_dict.k_lsh, g_lsh_dict.m_atoms, g_lsh_dict.tau);
            }
        }
        /* Parse active layers */
        if (g_lsh_dict.loaded) {
            if (!strcasecmp(lsh_layers_s, "all")) {
                for (uint32_t li = 0; li < g_lsh_dict.num_layers; li++) {
                    int32_t lid = g_lsh_dict.layers[li].layer_idx;
                    if (lid >= 0 && lid < BITNET_DUMP_MAX_LAYERS) {
                        g_lsh_active_layers[lid] = 1;
                        g_lsh_any_active = 1;
                    }
                }
            } else {
                const char* p = lsh_layers_s;
                while (*p) {
                    int v = atoi(p);
                    if (v >= 0 && v < BITNET_DUMP_MAX_LAYERS &&
                        g_lsh_dict.by_layer_idx[v] != NULL) {
                        g_lsh_active_layers[v] = 1;
                        g_lsh_any_active = 1;
                    }
                    while (*p && *p != ',') p++;
                    if (*p == ',') p++;
                }
            }
            fprintf(stderr, "[harness] LSH FFN active on layers: %s\n", lsh_layers_s);
        }
    }

    /* FFN-input activation dump (Trit Lattice LSH FFN validation) */
    g_dump_ffn_inputs_dir = getenv("BITNET_DUMP_FFN_INPUTS_DIR");
    const char* dlbl = getenv("BITNET_DUMP_LABEL");
    if (dlbl) g_dump_label = dlbl;
    const char* dlayers = getenv("BITNET_DUMP_FFN_INPUTS_LAYERS");
    if (g_dump_ffn_inputs_dir != NULL) {
        if (!dlayers || !strcasecmp(dlayers, "all")) {
            for (int i = 0; i < BITNET_DUMP_MAX_LAYERS; i++) g_dump_ffn_inputs_layer_mask[i] = 1;
            g_dump_ffn_inputs_any = 1;
        } else {
            /* Parse comma-separated list. */
            const char* p = dlayers;
            while (*p) {
                int v = atoi(p);
                if (v >= 0 && v < BITNET_DUMP_MAX_LAYERS) {
                    g_dump_ffn_inputs_layer_mask[v] = 1;
                    g_dump_ffn_inputs_any = 1;
                }
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        }
        fprintf(stderr, "[harness] FFN input dump → %s, label=%s, layers=%s\n",
                g_dump_ffn_inputs_dir, g_dump_label, dlayers ? dlayers : "all");
    }
    const char* w = getenv("BITNET_KV_WINDOW");
    if (w) {
        int v = atoi(w);
        if (v > 0) g_kv_window = v;
    }
    /* TRIT_ROUTING #10 amendment: M-step running-mean direction proxy. */
    const char* mm = getenv("BITNET_KV_EVICT_M");
    if (mm) { int v = atoi(mm); if (v >= 1) g_kv_evict_m = v; }
    /* TRIT_ROUTING #10 amendment: configurable random seed for multi-seed
     * baseline (per red-team finding M4 — single-seed random has unmeasured
     * variance). */
    const char* sd = getenv("BITNET_KV_EVICT_SEED");
    if (sd) { unsigned int v = (unsigned int)strtoul(sd, NULL, 0);
              if (v != 0) g_kv_evict_rng = v; }
    /* sigdist requires a tau for K signatures. Use 5000 default if unset
     * (per #4 finding: fixed tau is acceptable for quality). qsigdist
     * (Plan B) uses the same regime — K-sigs at fixed tau, Q-sig computed
     * per-Q-head on the fly. */
    if ((g_kv_evict_mode == BITNET_KV_EVICT_SIGDIST ||
         g_kv_evict_mode == BITNET_KV_EVICT_QSIGDIST ||
         g_kv_evict_mode == BITNET_KV_EVICT_META ||
         g_kv_evict_mode == BITNET_KV_EVICT_QSIG_FILTER) && g_attn_fixed_tau == 0) {
        g_attn_fixed_tau = 5000;
    }
    if (g_kv_evict_mode != BITNET_KV_EVICT_NONE) {
        fprintf(stderr, "[harness] KV eviction mode = %s, window = %d\n",
                bitnet_kv_evict_mode_name(g_kv_evict_mode), g_kv_window);
    }
    if (g_kv_evict_mode == BITNET_KV_EVICT_META) {
        fprintf(stderr, "[harness] meta weights: w_r=%+d, w_kk=%+d, w_qk=%+d\n",
                g_kv_evict_w_r, g_kv_evict_w_kk, g_kv_evict_w_qk);
    }
    if (g_kv_evict_mode == BITNET_KV_EVICT_QSIG_FILTER) {
        fprintf(stderr, "[harness] qsig_filter: kk_protect_k=%d\n",
                g_kv_evict_kk_protect_k);
    }
}

/* ── TRIT_ROUTING #8 falsification probe: synthetic-MoE slice masking. ──
 * BitNet's FFN intermediate (6912) is partitioned into N equal slices.
 * Per-token, pick top-k slices and zero out the rest in gate_act before
 * ffn_sub_norm sees it. Modes:
 *   - oracle : top-k by sum |gate_act[slice]| (uses ground truth)
 *   - random : top-k random slices (deterministic via xorshift, seeded)
 *   - dense  : no masking (default; behavior unchanged)
 * Substrate-routed gating is the eventual goal of #8 but requires
 * offline slice characteristic precomputation; this probe answers the
 * prerequisite question — does the FFN tolerate slice masking at all? */
typedef enum {
    BITNET_FFN_DENSE  = 0,
    BITNET_FFN_ORACLE,
    BITNET_FFN_RANDOM,
} bitnet_ffn_mode_t;

static bitnet_ffn_mode_t g_ffn_mode = BITNET_FFN_DENSE;
static int g_ffn_num_experts = 4;
static int g_ffn_k           = 2;
static unsigned int g_ffn_rng = 0xC0FFEE08u;

/* TRIT_ROUTING #9 falsification probe: cell-level FFN sparse activation.
 * Independent of #8 slice mode. Mask gate_act cells (not slices). */
typedef enum {
    BITNET_FFN_CELL_DENSE = 0,
    BITNET_FFN_CELL_ORACLE,
    BITNET_FFN_CELL_RANDOM,
} bitnet_ffn_cell_mode_t;

static bitnet_ffn_cell_mode_t g_ffn_cell_mode = BITNET_FFN_CELL_DENSE;
static int g_ffn_cell_keep = 0;          /* 0 = keep all (no mask) */
static unsigned int g_ffn_cell_rng = 0xC0FFEE09u;

static void bitnet_ffn_mode_init_from_env(void) {
    const char* m = getenv("BITNET_FFN_MODE");
    if (m) {
        if      (!strcasecmp(m, "dense"))  g_ffn_mode = BITNET_FFN_DENSE;
        else if (!strcasecmp(m, "oracle")) g_ffn_mode = BITNET_FFN_ORACLE;
        else if (!strcasecmp(m, "random")) g_ffn_mode = BITNET_FFN_RANDOM;
        else fprintf(stderr, "[harness] unknown BITNET_FFN_MODE=%s, using dense\n", m);
    }
    const char* n = getenv("BITNET_FFN_NUM_EXPERTS");
    if (n) { int v = atoi(n); if (v > 0 && BITNET_INTERMEDIATE_SIZE % v == 0) g_ffn_num_experts = v; }
    const char* k = getenv("BITNET_FFN_K");
    if (k) { int v = atoi(k); if (v > 0) g_ffn_k = v; }
    if (g_ffn_mode != BITNET_FFN_DENSE) {
        fprintf(stderr, "[harness] sparse FFN mode = %s, num_experts = %d, k = %d\n",
                m ? m : "dense", g_ffn_num_experts, g_ffn_k);
    }
    /* TRIT_ROUTING #9: cell-level mode. */
    const char* cm = getenv("BITNET_FFN_CELL_MODE");
    if (cm) {
        if      (!strcasecmp(cm, "dense"))  g_ffn_cell_mode = BITNET_FFN_CELL_DENSE;
        else if (!strcasecmp(cm, "oracle")) g_ffn_cell_mode = BITNET_FFN_CELL_ORACLE;
        else if (!strcasecmp(cm, "random")) g_ffn_cell_mode = BITNET_FFN_CELL_RANDOM;
        else fprintf(stderr, "[harness] unknown BITNET_FFN_CELL_MODE=%s, using dense\n", cm);
    }
    const char* ck = getenv("BITNET_FFN_CELL_KEEP");
    if (ck) { int v = atoi(ck); if (v > 0 && v <= BITNET_INTERMEDIATE_SIZE) g_ffn_cell_keep = v; }
    if (g_ffn_cell_mode != BITNET_FFN_CELL_DENSE && g_ffn_cell_keep > 0) {
        fprintf(stderr, "[harness] FFN cell mode = %s, keep = %d/%d\n",
                cm ? cm : "dense", g_ffn_cell_keep, BITNET_INTERMEDIATE_SIZE);
    }
}

/* Apply slice-mask to gate_act IN PLACE. Defined later (after xorshift32
 * and posracle_compare); forward declaration here. */
static void bitnet_ffn_apply_slice_mask(m4t_mtfp_t* gate_act, int N, int k);
/* TRIT_ROUTING #9: apply cell-mask to gate_act IN PLACE. Forward declared. */
static void bitnet_ffn_apply_cell_mask(m4t_mtfp_t* gate_act, int keep);

/* ── Phase 2 work-unit 1: bx-aware activation flow constants. ────────────
 * Target bx for normal (linear-magnitude) activations through the
 * network. Picked to give MTFP19_MAX/3^14 ≈ 35 of headroom — comfortable
 * for BitNet's value range (0.5–10 typical, 30–100 worst-case).
 *
 * The FFN intermediate path (between gate_proj and ffn_sub_norm) carries
 * SQUARED magnitudes (relu²(gate) × up). For gate_real ≤ 6 typical, the
 * squared product reaches ~200; bx=14 saturates at 35. We use a smaller
 * FFN_BX that gives wider range (real max 3^(29-FFN_BX) at MTFP19_MAX). */
#define BITNET_ACT_BX 8       /* MTFP19_MAX/3^8 ≈ 88573. Tuning history:
                               *
                               * Phase 2 wu1.4 red-team (pre-RMSNorm-fix):
                               * ACT_BX=10 saturated 0.5% of block_output
                               * cells from L4 onward. Lowered to 8 →
                               * "zero saturation across 30 layers" was
                               * the claim at the time.
                               *
                               * Post-RMSNorm-fix (commit 4d4c917): the
                               * "zero saturation" claim NO LONGER HOLDS.
                               * The bug was magnitude-collapsing post_attn_norm
                               * outputs by 6.5×; fixing it lets correctly-larger
                               * residuals propagate, and ~209 cells (0.034%
                               * of the 30L × 8pos × 2560-cell residual
                               * stream sweep) now saturate in late layers
                               * (L24-L29). See journal/saturation_audit_2026-05-09.md
                               * and journal/act_bx_sweep_2026-05-09.md.
                               *
                               * The ACT_BX sweep ∈ {6, 7, 8} (TD-19) showed:
                               * - BX=8 keeps 209 saturations BUT 8/8 prompts
                               *   coherent and math correct (12+7=19).
                               * - BX=7 → 0 saturations BUT one prompt loops.
                               * - BX=6 → 0 saturations BUT math regresses
                               *   (12+7=20; 1-trit precision loss flips argmax
                               *   on tight integer-token logit margins).
                               *
                               * Conscious tradeoff: keep BX=8. The ~0.034%
                               * residual-stream saturation is absorbed by
                               * downstream RMSNorm; coherence + correctness
                               * are the load-bearing metrics, not saturation
                               * count. End-to-end battery (8 + 24 prompts)
                               * confirms the substrate produces coherent
                               * English on factual / definitional / narrative
                               * / long-context tasks at this setting. */
#define BITNET_FFN_BX 6       /* MTFP19_MAX/3^6 ≈ 797K — gate, up.
                               * Sweep [4,6,8,10,12]: Pearson peaks at 6
                               * (0.6857). FFN_BX=8 ranks Paris higher
                               * (#8 vs #41) but loses overall correlation
                               * — picking 6 favors broader signal. */
#define BITNET_GATE_ACT_BX 1  /* MTFP19_MAX/3^1 ≈ 194M headroom for
                               * gate²×up products. Tuning history:
                               *
                               * Original choice (pre-RMSNorm-fix): 2,
                               * "Pearson invariant for [1,6]; picking 2
                               * to preserve ~17 fractional bits below 1.0."
                               * That sweep was on the buggy substrate
                               * (predates 4d4c917).
                               *
                               * Post-fix re-sweep (TD-20 remediation,
                               * 2026-05-10): swept GATE_ACT_BX ∈ {1, 2,
                               * 4, 6} on the 5 substrate-specific failure
                               * prompts. BX=1 went from baseline 1/5 to
                               * 5/5. Full 24-prompt battery confirmed:
                               * 5 prompts upgrade from loop/wrong → coherent
                               * (reason_word "2 hours" not "8", code_comment
                               * yields real def sort_array, json_format
                               * yields actual JSON), 1 regression
                               * (factual_hamlet gives a "Hint" instead
                               * of "Shakespeare"). Strict pass rate 15/24
                               * → ~19/24. Per journal/hp_sweep_2026-05-10.md. */

/* ── Cycle 2 sparse-attention helpers. Production-eligible per the
 *    "no scalar in production" foundational rule (2026-05-12 — see
 *    line 1139 below). Off by default (BITNET_ATTN_MODE=dense); opt
 *    into routed/oracle/etc. via env var. Per journal/cycle2_design.md. */

/* xorshift32 for deterministic random index selection (random arm). */
static inline unsigned int bitnet_xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

/* Pick k unique indices from [0, n) into out[]. Modifies *rng_state.
 * Uses partial Fisher-Yates: O(n) memory + time, fine for our n ≤ 4096. */
static void bitnet_pick_random_indices(int* out, int k, int n, unsigned int* rng_state) {
    if (k >= n) {
        for (int i = 0; i < n; i++) out[i] = i;
        return;
    }
    int* pool = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) pool[i] = i;
    for (int i = 0; i < k; i++) {
        unsigned int r = bitnet_xorshift32(rng_state);
        int j = i + (int)(r % (unsigned int)(n - i));
        int tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
        out[i] = pool[i];
    }
    free(pool);
}

/* qsort comparator for int64 ascending. */
static int bitnet_cmp_i64_asc(const void* a, const void* b) {
    int64_t av = *(const int64_t*)a, bv = *(const int64_t*)b;
    return (av > bv) - (av < bv);
}

/* Compute a percentile-based tau over |values[]| to feed
 * m4t_route_threshold_extract such that all three trit states are
 * realized non-trivially (per §18 input-class contract). Uses the
 * 1/3-quantile of absolute values: ~33% of trits will be 0, ~33%
 * will be +1, ~33% will be -1 in the resulting signature. Returns
 * tau in the same units as |values|. */
static int64_t bitnet_routed_pick_tau(const m4t_mtfp_t* values, int n) {
    int64_t* abs_vals = (int64_t*)malloc((size_t)n * sizeof(int64_t));
    for (int i = 0; i < n; i++) {
        int32_t v = values[i];
        abs_vals[i] = (int64_t)(v < 0 ? -v : v);
    }
    qsort(abs_vals, (size_t)n, sizeof(int64_t), bitnet_cmp_i64_asc);
    int64_t tau = abs_vals[n / 3];
    if (tau < 1) tau = 1;  /* floor at 1 to avoid all-trit-zero degenerate */
    free(abs_vals);
    return tau;
}

/* TRIT_ROUTING #1: ensure K-signature cache is allocated, with the
 * requested tau. If cache->k_sig is NULL, allocate. If allocated with
 * a different tau, free and re-allocate (caller must repopulate).
 * Returns 1 if cache was (re-)allocated (caller should populate); 0 if
 * cache already valid for this tau (caller can use existing). */
static int bitnet_kv_cache_ensure_sig(bitnet_kv_cache_t* cache, int64_t tau) {
    int sig_bytes = M4T_TRIT_PACKED_BYTES(BITNET_HEAD_DIM);
    size_t per_layer_sig_bytes = (size_t)cache->max_seq_len *
                                  (size_t)BITNET_NUM_KV_HEADS *
                                  (size_t)sig_bytes;
    size_t total_bytes = per_layer_sig_bytes * (size_t)cache->n_layers;
    if (cache->k_sig && cache->k_sig_tau == (int)tau) return 0;
    free(cache->k_sig);
    cache->k_sig = (uint8_t*)calloc(total_bytes, 1);
    cache->k_sig_tau = (int)tau;
    assert(cache->k_sig);
    return 1;  /* caller should populate signatures for all positions 0..current_pos */
}

/* TRIT_ROUTING #1: compute and store K signatures for one position
 * in the K cache. Called after K is written. */
static void bitnet_kv_cache_store_k_sig(
    bitnet_kv_cache_t* cache, int layer_idx, int position, int64_t tau)
{
    int sig_bytes = M4T_TRIT_PACKED_BYTES(BITNET_HEAD_DIM);
    size_t row_size = (size_t)BITNET_NUM_KV_HEADS * BITNET_HEAD_DIM;
    size_t k_row_base = (size_t)layer_idx * cache->per_layer_stride
                        + (size_t)position * row_size;
    size_t sig_row_base = (size_t)layer_idx * (size_t)cache->max_seq_len *
                          (size_t)BITNET_NUM_KV_HEADS * (size_t)sig_bytes
                        + (size_t)position * (size_t)BITNET_NUM_KV_HEADS *
                          (size_t)sig_bytes;
    int64_t k_i64[BITNET_HEAD_DIM];
    for (int kvh = 0; kvh < BITNET_NUM_KV_HEADS; kvh++) {
        const m4t_mtfp_t* k = cache->k + k_row_base + (size_t)kvh * BITNET_HEAD_DIM;
        for (int d = 0; d < BITNET_HEAD_DIM; d++) k_i64[d] = (int64_t)k[d];
        m4t_route_threshold_extract(
            cache->k_sig + sig_row_base + (size_t)kvh * (size_t)sig_bytes,
            k_i64, tau, BITNET_HEAD_DIM);
    }
}

/* TRIT_ROUTING #1: pointer to cached K signature for (layer, position, kv_head). */
static const uint8_t* bitnet_kv_cache_k_sig(
    const bitnet_kv_cache_t* cache, int layer_idx, int position, int kv_head)
{
    int sig_bytes = M4T_TRIT_PACKED_BYTES(BITNET_HEAD_DIM);
    size_t off = (size_t)layer_idx * (size_t)cache->max_seq_len *
                 (size_t)BITNET_NUM_KV_HEADS * (size_t)sig_bytes
               + (size_t)position * (size_t)BITNET_NUM_KV_HEADS *
                 (size_t)sig_bytes
               + (size_t)kv_head * (size_t)sig_bytes;
    return cache->k_sig + off;
}

/* TRIT_ROUTING #10: KV cache eviction helpers. ────────────────────────
 * After K-write, if alive_count for the layer exceeds the configured
 * window, mark non-alive positions until alive_count == window. */

static uint8_t* bitnet_kv_cache_evicted_row(bitnet_kv_cache_t* cache, int layer_idx) {
    return cache->evicted + (size_t)layer_idx * (size_t)cache->max_seq_len;
}

static int bitnet_kv_cache_ensure_evicted(bitnet_kv_cache_t* cache) {
    if (cache->evicted != NULL) return 0;
    size_t total = (size_t)cache->n_layers * (size_t)cache->max_seq_len;
    cache->evicted = (uint8_t*)calloc(total, 1);
    return cache->evicted ? 0 : 1;
}

/* Pick a victim alive position for layer_idx per the configured policy.
 * Returns -1 if no eviction is needed or no alive position exists.
 * current_position is the position just written (used by sigdist as the
 * "direction proxy" — K-sig of the current token approximates Q-direction). */
static int bitnet_kv_evict_pick_victim(
    bitnet_kv_cache_t* cache, int layer_idx, int seq_k, int current_position,
    const m4t_mtfp_t* q_all_heads /* (NUM_ATTENTION_HEADS, HEAD_DIM) or NULL */ )
{
    uint8_t* row = bitnet_kv_cache_evicted_row(cache, layer_idx);

    if (g_kv_evict_mode == BITNET_KV_EVICT_FIFO) {
        for (int p = 0; p < seq_k; p++) {
            if (!row[p] && p != current_position) return p;
        }
        return -1;
    }

    if (g_kv_evict_mode == BITNET_KV_EVICT_RANDOM) {
        /* Count alive positions excluding current_position. */
        int alive = 0;
        for (int p = 0; p < seq_k; p++) if (!row[p] && p != current_position) alive++;
        if (alive == 0) return -1;
        unsigned int r = bitnet_xorshift32(&g_kv_evict_rng);
        int pick = (int)(r % (unsigned int)alive);
        int seen = 0;
        for (int p = 0; p < seq_k; p++) {
            if (row[p] || p == current_position) continue;
            if (seen == pick) return p;
            seen++;
        }
        return -1;
    }

    if (g_kv_evict_mode == BITNET_KV_EVICT_SIGDIST) {
        /* For each alive p (excluding current_position), compute distance
         * from p's K-sig to the DIRECTION PROXY's signature.
         *
         * Direction proxy options (per #10 amendment, M5 finding):
         *  - M=1: signature of current_position's K (original probe behavior;
         *         uses pre-cached k_sig directly).
         *  - M>1: signature of the mean of K vectors over the last M alive
         *         positions (current_position + up to M-1 prior alive).
         *
         * Evict the position with MAX distance. */
        if (cache->k_sig == NULL) return -1;  /* sig cache must be populated */
        int sig_bytes = M4T_TRIT_PACKED_BYTES(BITNET_HEAD_DIM);

        /* Build M-step running-mean direction signature per kv_head.
         * For M=1 we can take a shortcut (use cached current K-sig);
         * for M>1 we recompute against an averaged K. */
        uint8_t* dir_sigs = NULL;
        if (g_kv_evict_m > 1) {
            dir_sigs = (uint8_t*)malloc((size_t)BITNET_NUM_KV_HEADS * (size_t)sig_bytes);
            if (!dir_sigs) return -1;
            /* Gather last M alive positions (including current_position),
             * walking backward from current_position. */
            int* m_positions = (int*)malloc((size_t)g_kv_evict_m * sizeof(int));
            int m_found = 0;
            for (int p = current_position; p >= 0 && m_found < g_kv_evict_m; p--) {
                if (p == current_position || !row[p]) m_positions[m_found++] = p;
            }
            /* For each kv_head: average K vectors over those positions,
             * threshold-extract a signature, store in dir_sigs[h]. */
            int64_t* k_mean = (int64_t*)malloc((size_t)BITNET_HEAD_DIM * sizeof(int64_t));
            size_t row_size = (size_t)BITNET_NUM_KV_HEADS * BITNET_HEAD_DIM;
            for (int h = 0; h < BITNET_NUM_KV_HEADS; h++) {
                for (int d = 0; d < BITNET_HEAD_DIM; d++) k_mean[d] = 0;
                for (int i = 0; i < m_found; i++) {
                    size_t base = (size_t)layer_idx * cache->per_layer_stride
                                  + (size_t)m_positions[i] * row_size
                                  + (size_t)h * BITNET_HEAD_DIM;
                    const m4t_mtfp_t* k = cache->k + base;
                    for (int d = 0; d < BITNET_HEAD_DIM; d++) k_mean[d] += (int64_t)k[d];
                }
                /* Mean — integer division is OK; threshold-extract is invariant
                 * to positive scaling. */
                if (m_found > 1)
                    for (int d = 0; d < BITNET_HEAD_DIM; d++) k_mean[d] /= m_found;
                /* Use same tau used for cached K signatures, so distances are
                 * commensurable. */
                int64_t tau = (int64_t)cache->k_sig_tau;
                m4t_route_threshold_extract(
                    dir_sigs + (size_t)h * sig_bytes,
                    k_mean, tau, BITNET_HEAD_DIM);
            }
            free(k_mean); free(m_positions);
        }

        int worst_p = -1; int worst_d = -1;
        for (int p = 0; p < seq_k; p++) {
            if (row[p] || p == current_position) continue;
            int dsum = 0;
            for (int h = 0; h < BITNET_NUM_KV_HEADS; h++) {
                const uint8_t* sa = bitnet_kv_cache_k_sig(cache, layer_idx, p, h);
                const uint8_t* sb;
                if (g_kv_evict_m > 1) sb = dir_sigs + (size_t)h * sig_bytes;
                else sb = bitnet_kv_cache_k_sig(cache, layer_idx, current_position, h);
                /* popcount of (sa XOR sb) over sig_bytes. */
                for (int i = 0; i < sig_bytes; i++) {
                    uint8_t x = (uint8_t)(sa[i] ^ sb[i]);
                    /* popcount4 lookup is cheaper but inline popcount8 is fine. */
                    x = (uint8_t)(x - ((x >> 1) & 0x55));
                    x = (uint8_t)((x & 0x33) + ((x >> 2) & 0x33));
                    dsum += (int)(((x + (x >> 4)) & 0x0F));
                }
            }
            if (dsum > worst_d) { worst_d = dsum; worst_p = p; }
        }
        free(dir_sigs);
        return worst_p;
    }

    if (g_kv_evict_mode == BITNET_KV_EVICT_QSIGDIST) {
        /* Plan B (2026-05-12): Q-aware eviction. For each alive
         * candidate position p, sum L1 distance between each Q-head's
         * signature and the K-signature at (p, that Q-head's KV-head).
         * Evict the position with MAX total distance — implementing the
         * operation Phase ε measured (Q-K oracle), aggregated across
         * the full GQA structure (20 Q-heads × 5 KV-heads).
         *
         * Q-sigs are computed per Q-head at the same fixed tau used
         * for K-sigs (g_attn_fixed_tau, forced to 5000 in this mode).
         * Distance is popcount(XOR on packed 2-bit codes) = L1 on the
         * underlying ternary signatures, matching SIGDIST's distance
         * loop bit-for-bit. */
        if (cache->k_sig == NULL || q_all_heads == NULL) return -1;
        int sig_bytes = M4T_TRIT_PACKED_BYTES(BITNET_HEAD_DIM);
        const int q_per_kv = BITNET_NUM_ATTENTION_HEADS / BITNET_NUM_KV_HEADS;

        /* Precompute Q-sigs per Q-head once. */
        uint8_t* q_sigs = (uint8_t*)malloc(
            (size_t)BITNET_NUM_ATTENTION_HEADS * (size_t)sig_bytes);
        if (!q_sigs) return -1;
        int64_t* q_tmp = (int64_t*)malloc((size_t)BITNET_HEAD_DIM * sizeof(int64_t));
        if (!q_tmp) { free(q_sigs); return -1; }
        for (int qh = 0; qh < BITNET_NUM_ATTENTION_HEADS; qh++) {
            const m4t_mtfp_t* q = q_all_heads + (size_t)qh * BITNET_HEAD_DIM;
            for (int d = 0; d < BITNET_HEAD_DIM; d++) q_tmp[d] = (int64_t)q[d];
            int64_t tau = (int64_t)cache->k_sig_tau;
            m4t_route_threshold_extract(
                q_sigs + (size_t)qh * sig_bytes, q_tmp, tau, BITNET_HEAD_DIM);
        }
        free(q_tmp);

        int worst_p = -1; int worst_d = -1;
        for (int p = 0; p < seq_k; p++) {
            if (row[p] || p == current_position) continue;
            int dsum = 0;
            for (int qh = 0; qh < BITNET_NUM_ATTENTION_HEADS; qh++) {
                int kvh = qh / q_per_kv;
                const uint8_t* sa = bitnet_kv_cache_k_sig(cache, layer_idx, p, kvh);
                const uint8_t* sb = q_sigs + (size_t)qh * sig_bytes;
                for (int i = 0; i < sig_bytes; i++) {
                    uint8_t x = (uint8_t)(sa[i] ^ sb[i]);
                    x = (uint8_t)(x - ((x >> 1) & 0x55));
                    x = (uint8_t)((x & 0x33) + ((x >> 2) & 0x33));
                    dsum += (int)(((x + (x >> 4)) & 0x0F));
                }
            }
            if (dsum > worst_d) { worst_d = dsum; worst_p = p; }
        }
        free(q_sigs);
        return worst_p;
    }

    if (g_kv_evict_mode == BITNET_KV_EVICT_QSIG_FILTER) {
        /* QSIG_FILTER (2026-05-14, integrative test): qsigdist with
         * a K-K-similarity FILTER. The hypothesis: qsigdist
         * sometimes evicts slots that are K-K-similar to current K
         * (representing the same context as the current write).
         * Removing them loses redundant-but-coherent signal. The
         * conjunctive filter prevents this:
         *   1. Compute KK_dist for every candidate slot.
         *   2. Mark the K candidates with LOWEST KK_dist as PROTECTED
         *      (most similar to current K).
         *   3. Among unprotected, evict argmax(QK_dist) — qsigdist's
         *      criterion.
         *   4. Fallback: if all candidates are protected (e.g.
         *      window-1 ≤ K), evict argmax(QK_dist) over ALL
         *      candidates — i.e., behave like qsigdist.
         *
         * This is structurally distinct from the linear-additive
         * trit-weight family: the filter is a CONJUNCTION (slot
         * survives IF KK_dist > threshold AND eviction-by-QK
         * applies), which can't be expressed as
         *   c_score = w_kk·KK_dist + w_qk·QK_dist
         * for any choice of weights.
         *
         * K = g_kv_evict_kk_protect_k. K=0 ≡ qsigdist exactly. */
        if (cache->k_sig == NULL || q_all_heads == NULL) return -1;
        int sig_bytes = M4T_TRIT_PACKED_BYTES(BITNET_HEAD_DIM);
        const int q_per_kv = BITNET_NUM_ATTENTION_HEADS / BITNET_NUM_KV_HEADS;

        /* Precompute Q-sigs per Q-head once. */
        uint8_t* q_sigs = (uint8_t*)malloc(
            (size_t)BITNET_NUM_ATTENTION_HEADS * (size_t)sig_bytes);
        if (!q_sigs) return -1;
        int64_t* q_tmp = (int64_t*)malloc((size_t)BITNET_HEAD_DIM * sizeof(int64_t));
        if (!q_tmp) { free(q_sigs); return -1; }
        for (int qh = 0; qh < BITNET_NUM_ATTENTION_HEADS; qh++) {
            const m4t_mtfp_t* q = q_all_heads + (size_t)qh * BITNET_HEAD_DIM;
            for (int d = 0; d < BITNET_HEAD_DIM; d++) q_tmp[d] = (int64_t)q[d];
            int64_t tau = (int64_t)cache->k_sig_tau;
            m4t_route_threshold_extract(
                q_sigs + (size_t)qh * sig_bytes, q_tmp, tau, BITNET_HEAD_DIM);
        }
        free(q_tmp);

        /* Pass 1: collect candidates and their KK_dists + QK_dists. */
        int max_cands = seq_k;
        int* cand_p     = (int*)malloc((size_t)max_cands * sizeof(int));
        int* cand_kk    = (int*)malloc((size_t)max_cands * sizeof(int));
        int* cand_qk    = (int*)malloc((size_t)max_cands * sizeof(int));
        if (!cand_p || !cand_kk || !cand_qk) {
            free(q_sigs); free(cand_p); free(cand_kk); free(cand_qk);
            return -1;
        }
        int n_cand = 0;
        for (int p = 0; p < seq_k; p++) {
            if (row[p] || p == current_position) continue;
            int kk_dist = 0;
            for (int h = 0; h < BITNET_NUM_KV_HEADS; h++) {
                const uint8_t* sa = bitnet_kv_cache_k_sig(cache, layer_idx, p, h);
                const uint8_t* sb = bitnet_kv_cache_k_sig(cache, layer_idx,
                                                          current_position, h);
                for (int i = 0; i < sig_bytes; i++) {
                    uint8_t x = (uint8_t)(sa[i] ^ sb[i]);
                    x = (uint8_t)(x - ((x >> 1) & 0x55));
                    x = (uint8_t)((x & 0x33) + ((x >> 2) & 0x33));
                    kk_dist += (int)(((x + (x >> 4)) & 0x0F));
                }
            }
            int qk_dist = 0;
            for (int qh = 0; qh < BITNET_NUM_ATTENTION_HEADS; qh++) {
                int kvh = qh / q_per_kv;
                const uint8_t* sa = bitnet_kv_cache_k_sig(cache, layer_idx, p, kvh);
                const uint8_t* sb = q_sigs + (size_t)qh * sig_bytes;
                for (int i = 0; i < sig_bytes; i++) {
                    uint8_t x = (uint8_t)(sa[i] ^ sb[i]);
                    x = (uint8_t)(x - ((x >> 1) & 0x55));
                    x = (uint8_t)((x & 0x33) + ((x >> 2) & 0x33));
                    qk_dist += (int)(((x + (x >> 4)) & 0x0F));
                }
            }
            cand_p[n_cand] = p;
            cand_kk[n_cand] = kk_dist;
            cand_qk[n_cand] = qk_dist;
            n_cand++;
        }
        free(q_sigs);

        int K = g_kv_evict_kk_protect_k;
        int worst_p = -1; int worst_qk = -1;

        if (K <= 0 || n_cand <= K) {
            /* No filtering, or all candidates would be protected.
             * Fallback to plain qsigdist over ALL candidates. */
            for (int i = 0; i < n_cand; i++) {
                if (cand_qk[i] > worst_qk) {
                    worst_qk = cand_qk[i];
                    worst_p = cand_p[i];
                }
            }
        } else {
            /* Find K smallest kk_dists via partial selection. n_cand
             * is small (≤ window), so O(K·n_cand) is fine. Tie-break:
             * earliest-encountered (deterministic). */
            uint8_t* protected_arr = (uint8_t*)calloc((size_t)n_cand, sizeof(uint8_t));
            if (!protected_arr) {
                free(cand_p); free(cand_kk); free(cand_qk);
                return -1;
            }
            for (int k = 0; k < K; k++) {
                int min_i = -1; int min_v = INT_MAX;
                for (int i = 0; i < n_cand; i++) {
                    if (protected_arr[i]) continue;
                    if (cand_kk[i] < min_v) { min_v = cand_kk[i]; min_i = i; }
                }
                if (min_i >= 0) protected_arr[min_i] = 1;
            }
            /* Among unprotected, evict argmax(QK_dist). Tie-break: first. */
            for (int i = 0; i < n_cand; i++) {
                if (protected_arr[i]) continue;
                if (cand_qk[i] > worst_qk) {
                    worst_qk = cand_qk[i];
                    worst_p = cand_p[i];
                }
            }
            free(protected_arr);
        }
        free(cand_p); free(cand_kk); free(cand_qk);
        return worst_p;
    }

    if (g_kv_evict_mode == BITNET_KV_EVICT_META) {
        /* Meta-routing: parameterized scoring policy with trit weights.
         *
         * Python convention (matching meta_routing.py): score = w_r·age
         * + w_kk·KK_sim + w_qk·QK_sim, evict slot with LOWEST score.
         * Translated to the C convention used here (distance, evict
         * argmax): we evict the slot with HIGHEST
         *
         *   c_score(p) = -w_r·age(p) + w_kk·KK_dist(p) + w_qk·QK_dist(p)
         *
         * The negation on w_r comes from age (Python) ↔ -recency
         * (similarity-like, where younger = higher); on w_kk/w_qk the
         * sign carries through because sim = -dist + const and the
         * constant drops out under argmax.
         *
         * Existing modes recovered:
         *   fifo:     w_r=-1 (python) → c_score = +age, evict oldest.    ✓
         *   sigdist:  w_kk=+1 → c_score = +KK_dist, evict max-dist.       ✓
         *   qsigdist: w_qk=+1 → c_score = +QK_dist, evict max-dist.       ✓
         *
         * Candidate from the layer-3 prediction (2026-05-14):
         *   (w_r=0, w_kk=-1, w_qk=+1) — "qsigdist but KEEP K-K-similar
         *   slots." c_score = -KK_dist + QK_dist, evict slot with HIGH
         *   QK_dist AND LOW KK_dist. Semantic: "redundant with current
         *   K AND irrelevant to Q." */
        if (cache->k_sig == NULL) return -1;
        int sig_bytes = M4T_TRIT_PACKED_BYTES(BITNET_HEAD_DIM);
        const int q_per_kv = BITNET_NUM_ATTENTION_HEADS / BITNET_NUM_KV_HEADS;

        /* Precompute Q-sigs once if w_qk != 0 and Q is available. */
        uint8_t* q_sigs = NULL;
        if (g_kv_evict_w_qk != 0 && q_all_heads != NULL) {
            q_sigs = (uint8_t*)malloc(
                (size_t)BITNET_NUM_ATTENTION_HEADS * (size_t)sig_bytes);
            if (!q_sigs) return -1;
            int64_t* q_tmp = (int64_t*)malloc((size_t)BITNET_HEAD_DIM * sizeof(int64_t));
            if (!q_tmp) { free(q_sigs); return -1; }
            for (int qh = 0; qh < BITNET_NUM_ATTENTION_HEADS; qh++) {
                const m4t_mtfp_t* q = q_all_heads + (size_t)qh * BITNET_HEAD_DIM;
                for (int d = 0; d < BITNET_HEAD_DIM; d++) q_tmp[d] = (int64_t)q[d];
                int64_t tau = (int64_t)cache->k_sig_tau;
                m4t_route_threshold_extract(
                    q_sigs + (size_t)qh * sig_bytes, q_tmp, tau, BITNET_HEAD_DIM);
            }
            free(q_tmp);
        }

        int best_p = -1;
        int64_t best_score = INT64_MIN;
        /* If all weights are zero, behave like random (any alive slot
         * has score 0, ties broken by first-encountered iteration order
         * — caller can layer randomization). */
        for (int p = 0; p < seq_k; p++) {
            if (row[p] || p == current_position) continue;

            int kk_dist = 0;
            if (g_kv_evict_w_kk != 0) {
                for (int h = 0; h < BITNET_NUM_KV_HEADS; h++) {
                    const uint8_t* sa = bitnet_kv_cache_k_sig(cache, layer_idx, p, h);
                    const uint8_t* sb = bitnet_kv_cache_k_sig(cache, layer_idx,
                                                              current_position, h);
                    for (int i = 0; i < sig_bytes; i++) {
                        uint8_t x = (uint8_t)(sa[i] ^ sb[i]);
                        x = (uint8_t)(x - ((x >> 1) & 0x55));
                        x = (uint8_t)((x & 0x33) + ((x >> 2) & 0x33));
                        kk_dist += (int)(((x + (x >> 4)) & 0x0F));
                    }
                }
            }

            int qk_dist = 0;
            if (g_kv_evict_w_qk != 0 && q_sigs) {
                for (int qh = 0; qh < BITNET_NUM_ATTENTION_HEADS; qh++) {
                    int kvh = qh / q_per_kv;
                    const uint8_t* sa = bitnet_kv_cache_k_sig(cache, layer_idx, p, kvh);
                    const uint8_t* sb = q_sigs + (size_t)qh * sig_bytes;
                    for (int i = 0; i < sig_bytes; i++) {
                        uint8_t x = (uint8_t)(sa[i] ^ sb[i]);
                        x = (uint8_t)(x - ((x >> 1) & 0x55));
                        x = (uint8_t)((x & 0x33) + ((x >> 2) & 0x33));
                        qk_dist += (int)(((x + (x >> 4)) & 0x0F));
                    }
                }
            }

            int age = current_position - p;
            int64_t score = (int64_t)(-g_kv_evict_w_r) * (int64_t)age
                          + (int64_t)g_kv_evict_w_kk  * (int64_t)kk_dist
                          + (int64_t)g_kv_evict_w_qk  * (int64_t)qk_dist;

            if (score > best_score) { best_score = score; best_p = p; }
        }
        free(q_sigs);
        return best_p;
    }

    return -1;
}

/* Apply eviction at the given layer to bring alive_count down to window.
 * Called after K-write + K-sig store. q_all_heads is required for
 * QSIGDIST mode; safely NULL for other modes. */
static void bitnet_kv_evict_apply(
    bitnet_kv_cache_t* cache, int layer_idx, int seq_k, int current_position,
    const m4t_mtfp_t* q_all_heads)
{
    if (g_kv_evict_mode == BITNET_KV_EVICT_NONE || g_kv_window <= 0) return;
    if (bitnet_kv_cache_ensure_evicted(cache)) return;
    uint8_t* row = bitnet_kv_cache_evicted_row(cache, layer_idx);

    /* Count currently alive in [0, seq_k). */
    int alive = 0;
    for (int p = 0; p < seq_k; p++) if (!row[p]) alive++;

    while (alive > g_kv_window) {
        int victim = bitnet_kv_evict_pick_victim(cache, layer_idx, seq_k,
                                                  current_position, q_all_heads);
        if (victim < 0) break;
        row[victim] = 1;
        alive--;
    }
}

/* Pick the k positions in [0, seq_k) whose K signatures are CLOSEST to
 * the Q signature in popcount (Hamming-on-trits) distance. Uses
 * m4t_route_threshold_extract to build signatures and
 * m4t_route_distance_batch for distance computation. Manually sorts
 * distances for top-k selection (route_topk_abs has T ≤ 64 limit; we
 * want it for arbitrary seq_k).
 *
 * out[0..k) = chosen position indices. */
static int bitnet_pick_routed_compare(const void* a, const void* b) {
    /* Sort ascending by second-element-of-pair (distance). */
    int32_t av = ((const int32_t*)a)[1];
    int32_t bv = ((const int32_t*)b)[1];
    return (av > bv) - (av < bv);
}

static void bitnet_pick_routed_indices(
    int* out, int k, int seq_k,
    const m4t_mtfp_t* qh,
    const m4t_mtfp_t* k_cache, size_t k_row_size, int kv_head,
    int head_dim,
    /* TRIT_ROUTING #1: optional cached K signatures. If non-NULL and tau
     * matches cache->k_sig_tau, skip K signature recomputation. */
    const bitnet_kv_cache_t* cache, int layer_idx)
{
    int sig_bytes = M4T_TRIT_PACKED_BYTES(head_dim);

    /* Choose tau. Default: per-Q 1/3-quantile of |Q| (ensures all three trit
     * states realize per §18 input-class contract). Override via
     * BITNET_ATTN_TAU env var (TRIT_ROUTING #4: per-Q not load-bearing
     * for aggregate quality; fixed tau enables K-signature caching). */
    int64_t tau = g_attn_fixed_tau > 0 ? g_attn_fixed_tau
                                        : bitnet_routed_pick_tau(qh, head_dim);

    /* Use cached K signatures if available with matching tau. */
    int use_cache = (cache && cache->k_sig && cache->k_sig_tau == (int)tau);

    /* Q signature (always per-step — depends on current Q). */
    int64_t* q_i64 = (int64_t*)malloc((size_t)head_dim * sizeof(int64_t));
    for (int d = 0; d < head_dim; d++) q_i64[d] = (int64_t)qh[d];
    uint8_t* q_sig = (uint8_t*)malloc((size_t)sig_bytes);
    m4t_route_threshold_extract(q_sig, q_i64, tau, head_dim);

    /* K signatures: read from cache if available, else compute on the fly.
     * Either way, gather into a contiguous buffer for distance_batch. */
    uint8_t* k_sigs = (uint8_t*)malloc((size_t)seq_k * sig_bytes);
    if (use_cache) {
        for (int t = 0; t < seq_k; t++) {
            const uint8_t* src = bitnet_kv_cache_k_sig(cache, layer_idx, t, kv_head);
            memcpy(k_sigs + (size_t)t * sig_bytes, src, (size_t)sig_bytes);
        }
    } else {
        int64_t* tmp_i64 = q_i64;  /* reuse */
        for (int t = 0; t < seq_k; t++) {
            const m4t_mtfp_t* kh = k_cache + (size_t)t * k_row_size + (size_t)kv_head * head_dim;
            for (int d = 0; d < head_dim; d++) tmp_i64[d] = (int64_t)kh[d];
            m4t_route_threshold_extract(k_sigs + (size_t)t * sig_bytes,
                                         tmp_i64, tau, head_dim);
        }
    }

    /* Distances. mask = all 0xFF (consider all positions). */
    uint8_t* mask = (uint8_t*)malloc((size_t)sig_bytes);
    memset(mask, 0xFF, (size_t)sig_bytes);
    int32_t* dists = (int32_t*)malloc((size_t)seq_k * sizeof(int32_t));
    m4t_route_distance_batch(dists, q_sig, k_sigs, mask, seq_k, head_dim);

    /* Top-k by SMALLEST distance — sort (idx, dist) pairs ascending. */
    int32_t* pairs = (int32_t*)malloc((size_t)seq_k * 2 * sizeof(int32_t));
    for (int t = 0; t < seq_k; t++) {
        pairs[2*t + 0] = (int32_t)t;
        pairs[2*t + 1] = dists[t];
    }
    qsort(pairs, (size_t)seq_k, 2 * sizeof(int32_t), bitnet_pick_routed_compare);
    for (int i = 0; i < k; i++) out[i] = (int)pairs[2*i + 0];

    free(pairs); free(dists); free(mask);
    free(k_sigs); free(q_sig); free(q_i64);
}

/* Pick top-k indices from scores[0..n) by absolute value (descending).
 * Used by ORACLE arm — picks the k positions with highest |Q·K| score
 * after dense scores are computed. O(n log n) sort; fine for n ≤ 4096. */
static int bitnet_pick_oracle_compare(const void* a, const void* b) {
    /* Sort descending by |second-element-of-pair|. Pair: (index, abs_score). */
    int64_t av = ((const int64_t*)a)[1];
    int64_t bv = ((const int64_t*)b)[1];
    if (av < bv) return  1;
    if (av > bv) return -1;
    return 0;
}

static void bitnet_pick_oracle_topk(int* out, int k, int n, const int64_t* scores) {
    if (k >= n) {
        for (int i = 0; i < n; i++) out[i] = i;
        return;
    }
    int64_t* pairs = (int64_t*)malloc((size_t)n * 2 * sizeof(int64_t));
    for (int i = 0; i < n; i++) {
        pairs[2*i + 0] = (int64_t)i;
        pairs[2*i + 1] = scores[i] < 0 ? -scores[i] : scores[i];
    }
    qsort(pairs, (size_t)n, 2 * sizeof(int64_t), bitnet_pick_oracle_compare);
    for (int i = 0; i < k; i++) out[i] = (int)pairs[2*i + 0];
    free(pairs);
}

/* TD-27 H2 test: top-k by SIGNED score (descending — highest positive wins).
 * If oracle's "wastes budget on high-|negative-score| positions softmax
 * suppresses anyway" hypothesis is right, this should approach routed quality. */
static int bitnet_pick_posracle_compare(const void* a, const void* b) {
    /* Sort descending by SIGNED score (no abs). Negatives sink to bottom. */
    int64_t av = ((const int64_t*)a)[1];
    int64_t bv = ((const int64_t*)b)[1];
    if (av < bv) return  1;
    if (av > bv) return -1;
    return 0;
}

static void bitnet_pick_posracle_topk(int* out, int k, int n, const int64_t* scores) {
    if (k >= n) {
        for (int i = 0; i < n; i++) out[i] = i;
        return;
    }
    int64_t* pairs = (int64_t*)malloc((size_t)n * 2 * sizeof(int64_t));
    for (int i = 0; i < n; i++) {
        pairs[2*i + 0] = (int64_t)i;
        pairs[2*i + 1] = scores[i];  /* SIGNED — negatives sink to bottom in sort */
    }
    qsort(pairs, (size_t)n, 2 * sizeof(int64_t), bitnet_pick_posracle_compare);
    for (int i = 0; i < k; i++) out[i] = (int)pairs[2*i + 0];
    free(pairs);
}

/* TRIT_ROUTING #8 falsification probe: slice masking on FFN intermediate.
 * See bitnet_ffn_mode_t comment above for design rationale. */
static void bitnet_ffn_apply_slice_mask(m4t_mtfp_t* gate_act, int N, int k) {
    if (N <= 1 || k >= N) return;
    int S = BITNET_INTERMEDIATE_SIZE / N;
    if (S * N != BITNET_INTERMEDIATE_SIZE) return;  /* misalignment guard */

    int64_t* slice_scores = (int64_t*)malloc((size_t)N * sizeof(int64_t));
    for (int e = 0; e < N; e++) {
        int64_t s = 0;
        if (g_ffn_mode == BITNET_FFN_ORACLE) {
            for (int j = e*S; j < (e+1)*S; j++) {
                m4t_mtfp_t v = gate_act[j];
                s += v < 0 ? -(int64_t)v : (int64_t)v;
            }
        } else if (g_ffn_mode == BITNET_FFN_RANDOM) {
            s = (int64_t)bitnet_xorshift32(&g_ffn_rng);
        }
        slice_scores[e] = s;
    }

    int64_t* pairs = (int64_t*)malloc((size_t)N * 2 * sizeof(int64_t));
    for (int e = 0; e < N; e++) {
        pairs[2*e + 0] = (int64_t)e;
        pairs[2*e + 1] = slice_scores[e];
    }
    qsort(pairs, (size_t)N, 2 * sizeof(int64_t), bitnet_pick_posracle_compare);

    uint8_t* keep = (uint8_t*)calloc((size_t)N, 1);
    for (int i = 0; i < k; i++) keep[(int)pairs[2*i + 0]] = 1;

    for (int e = 0; e < N; e++) {
        if (keep[e]) continue;
        for (int j = e*S; j < (e+1)*S; j++) gate_act[j] = 0;
    }

    free(keep); free(pairs); free(slice_scores);
}

/* TRIT_ROUTING #9: cell-level mask. Keep top-`keep` cells by score
 * (oracle = |gate_act[j]|; random = xorshift). Zero the rest. */
static void bitnet_ffn_apply_cell_mask(m4t_mtfp_t* gate_act, int keep) {
    int N = BITNET_INTERMEDIATE_SIZE;
    if (keep <= 0 || keep >= N) return;

    int64_t* pairs = (int64_t*)malloc((size_t)N * 2 * sizeof(int64_t));
    for (int j = 0; j < N; j++) {
        pairs[2*j + 0] = (int64_t)j;
        int64_t s = 0;
        if (g_ffn_cell_mode == BITNET_FFN_CELL_ORACLE) {
            m4t_mtfp_t v = gate_act[j];
            s = v < 0 ? -(int64_t)v : (int64_t)v;
        } else if (g_ffn_cell_mode == BITNET_FFN_CELL_RANDOM) {
            s = (int64_t)bitnet_xorshift32(&g_ffn_cell_rng);
        }
        pairs[2*j + 1] = s;
    }
    qsort(pairs, (size_t)N, 2 * sizeof(int64_t), bitnet_pick_posracle_compare);

    uint8_t* keep_bits = (uint8_t*)calloc((size_t)N, 1);
    for (int i = 0; i < keep; i++) keep_bits[(int)pairs[2*i + 0]] = 1;

    for (int j = 0; j < N; j++) {
        if (!keep_bits[j]) gate_act[j] = 0;
    }

    free(keep_bits); free(pairs);
}

/* Sparse attn_v_combine: out[d] = clamp(Σ_i weights[i] · V[indices[i]][d] >> shift).
 *
 * NEON via gather: copy V[indices[i]] into a contiguous scratch buffer, then
 * call the NEON `m4t_mtfp_attn_v_combine` with tight stride. Per the
 * foundational "no scalar in production" rule, the gather is data movement
 * (libc memcpy → SIMD on modern targets) and the computational kernel is the
 * existing NEON V combine.
 *
 * Scratch buffer is file-scope static, sized for the worst case (k_max =
 * 4096, head_dim_max = BITNET_HEAD_DIM = 128). Inference is single-threaded
 * so no concurrency issues. This avoids 900+ malloc/free pairs per gen step
 * that an alloc-per-call version had — caught in red-team 2026-05-12. */
#define BITNET_SPARSE_GATHER_MAX_K  4096
static m4t_mtfp_t s_sparse_v_gather[BITNET_SPARSE_GATHER_MAX_K * BITNET_HEAD_DIM];

static void bitnet_sparse_attn_v_combine(
    m4t_mtfp_t* out, int shift,
    const m4t_mtfp_t* weights,
    const m4t_mtfp_t* V_base, size_t row_size,
    int k, int head_dim,
    const int* indices)
{
    if (k == 0 || head_dim == 0) return;
    assert(k <= BITNET_SPARSE_GATHER_MAX_K);
    assert(head_dim <= BITNET_HEAD_DIM);
    /* Gather V[indices[i]] into the scratch buffer. */
    for (int i = 0; i < k; i++) {
        const m4t_mtfp_t* V_t = V_base + (size_t)indices[i] * row_size;
        memcpy(s_sparse_v_gather + (size_t)i * (size_t)head_dim, V_t,
               (size_t)head_dim * sizeof(m4t_mtfp_t));
    }
    /* NEON V combine on the gathered buffer with tight stride. */
    m4t_mtfp_attn_v_combine(out, shift, weights, s_sparse_v_gather, (size_t)head_dim, k, head_dim);
}

/* ── BitLinear scale composition (work-unit 5) ───────────────────────────
 *
 * BitLinear forward:  y = (matmul_int_out) × α × absmax / 127
 * where:
 *   - α is the BitLinear's per-tensor scale, stored as MTFP19
 *     (mantissa × 3^(-block_exp)).
 *   - absmax is the activation's per-token absmax (from A8 quantize).
 *   - matmul_int_out is the raw int-mantissa output from
 *     m4t_ternary_5in8_matmul_bt.
 *
 * Composing: y = raw × (α_m / 3^bx) × absmax / 127
 *              = raw × (α_m × absmax) / (127 × 3^bx)
 *
 * num = α_m × absmax, den = 127 × 3^bx.
 *
 * Constraint: bx ≤ ~38 to keep den ≤ int64 max. For BitNet b1.58-2B-4T
 * α magnitudes (range ~1e-3 to ~1), typical block_exp ≤ 25. Documented.
 *
 * If α_mantissa == 0 (convert_weights.py's sentinel for "no α
 * available"), the apply is a no-op (skip vec_scale; output remains
 * raw matmul, useful for skeleton-mode debugging). */

/* pow3_int: replaced by m4t_mtfp_*'s internal pow3_i64 since work-unit-1
 * Phase 2 (bx-aware primitives consume the bx-shift inline). */

__attribute__((unused))
static void bitnet_apply_bitlinear_scale(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    const m4t_mtfp_t* alpha_ptr, int alpha_block_exp,
    m4t_mtfp_t absmax, int n)
{
    /* bx-aware path (Phase 2 work-unit 1): produce output at
     * BITNET_ACT_BX assuming the input x_norm was at BITNET_ACT_BX. */
    m4t_mtfp_bitlinear_scale_bx(y, x, alpha_ptr, alpha_block_exp,
                                 absmax,
                                 /*x_bx=*/BITNET_ACT_BX,
                                 /*target_bx=*/BITNET_ACT_BX, n);
}

/* Bit-faithful BitLinear: int32 × ternary matmul → int64 raw → α scale apply.
 * No a8 quantization; matches HF's bf16-everywhere precision (modulo MTFP19
 * vs bf16 storage of x and α). Uses the 4-in-8 packed weights produced by
 * bitnet_weights.c's repack pass. */
static void bitnet_bitlinear_no_a8(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    const uint8_t* W_packed_4in8,
    const m4t_mtfp_t* alpha_ptr, int alpha_block_exp,
    int x_bx, int target_bx,
    int K, int N)
{
    /* Stack scratch for int64 raw output. Largest N is INTERMEDIATE = 6912
     * → 55 KB. Comfortable on macOS' 8 MB main stack. */
    int64_t y_raw[BITNET_INTERMEDIATE_SIZE];
    assert(N <= BITNET_INTERMEDIATE_SIZE);
    m4t_mtfp_ternary_matmul_bt_route_i64(y_raw, x, W_packed_4in8, /*M=*/1, K, N);
    m4t_mtfp_bitlinear_scale_no_a8_bx(y, y_raw, alpha_ptr, alpha_block_exp,
                                       x_bx, target_bx, N);
}

/* ── Scratch alloc / free ────────────────────────────────────────── */

void bitnet_block_scratch_alloc(bitnet_block_scratch_t* s) {
    /* Single-token forward pass: each [HIDDEN] or [INTERMEDIATE] buffer
     * holds one row. Multi-token prefill widens these (work-unit 7+). */
    s->x              = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->residual       = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->x_norm         = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->x_norm_input   = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->q              = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->q_pre_rope     = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->k              = calloc(BITNET_KV_PROJ_DIM,         sizeof(m4t_mtfp_t));
    s->k_pre_rope     = calloc(BITNET_KV_PROJ_DIM,         sizeof(m4t_mtfp_t));
    s->v              = calloc(BITNET_KV_PROJ_DIM,         sizeof(m4t_mtfp_t));
    /* Single-token: attn_scores is just [num_heads × 1 × 1] = 20 cells.
     * Multi-token: would scale with seq_len. */
    s->attn_scores    = calloc(BITNET_NUM_ATTENTION_HEADS, sizeof(m4t_mtfp_t));
    s->attn_output    = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->attn_sub_norm  = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->gate           = calloc(BITNET_INTERMEDIATE_SIZE,   sizeof(m4t_mtfp_t));
    s->up             = calloc(BITNET_INTERMEDIATE_SIZE,   sizeof(m4t_mtfp_t));
    s->gate_act       = calloc(BITNET_INTERMEDIATE_SIZE,   sizeof(m4t_mtfp_t));
    s->ffn_sub_norm   = calloc(BITNET_INTERMEDIATE_SIZE,   sizeof(m4t_mtfp_t));
    s->q_int8         = calloc(BITNET_INTERMEDIATE_SIZE,   sizeof(int8_t));
    s->q_absmax       = 0;
}

void bitnet_block_scratch_free(bitnet_block_scratch_t* s) {
    free(s->x); free(s->residual); free(s->x_norm); free(s->x_norm_input);
    free(s->q); free(s->q_pre_rope);
    free(s->k); free(s->k_pre_rope);
    free(s->v);
    free(s->attn_scores); free(s->attn_output); free(s->attn_sub_norm);
    free(s->gate); free(s->up); free(s->gate_act); free(s->ffn_sub_norm);
    free(s->q_int8);
    memset(s, 0, sizeof(*s));
}

/* ── Forward pass through one transformer block ──────────────────── */

/*
 * Input: x[HIDDEN] (the token embedding for the first block; the
 * previous block's output for layers > 0).
 * Output: x[HIDDEN] (overwritten in place).
 *
 * Per the architecture (verified in bitnet_phase1_o1_findings.md):
 *
 *   residual = x
 *   x = input_layernorm(x)
 *   x = attention(x)            ← contains attn_sub_norm
 *   x = residual + x
 *   residual = x
 *   x = post_attention_layernorm(x)
 *   x = ffn(x)                  ← contains ffn_sub_norm
 *   x = residual + x
 */
void bitnet_forward_block(
    m4t_mtfp_t* x_io,
    const bitnet_layer_weights_t* w,
    bitnet_block_scratch_t* s,
    bitnet_kv_cache_t* cache,
    int layer_idx,
    int position)
{
    /* For now: x_io aliased into s->x at entry, copied back at exit.
     * Future cleanup may eliminate the copy. */
    memcpy(s->x, x_io, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));

    /* ── Attention sub-block ──────────────────────────────────────── */

    /* residual = x */
    memcpy(s->residual, s->x, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));

    /* x_norm = input_layernorm(x). bx-aware: x at BITNET_ACT_BX,
     * γ at its loaded bx, output at BITNET_ACT_BX. */
    m4t_mtfp_rmsnorm_bx(s->x_norm, s->x, w->gamma_input_norm,
                        /*x_bx=*/BITNET_ACT_BX,
                        /*gamma_bx=*/w->gamma_input_norm_block_exp,
                        /*target_bx=*/BITNET_ACT_BX,
                        /*eps_mantissa=*/1, BITNET_HIDDEN_SIZE);
    /* Snapshot for the dump — s->x_norm gets overwritten by
     * post_attention_layernorm later in the block. */
    memcpy(s->x_norm_input, s->x_norm,
           BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
    /* TODO(work-unit 1.5): empirically determine eps_mtfp19. The HF
     * eps is 1e-5 (FP); the substrate-side equivalent depends on
     * what scale the activations land in. Capture x's magnitude
     * during first run and pick eps to land at the same ratio. */

    /* QKV projections via packed-ternary matmul.
     * Activation goes through A8 quantize first per W1.58A8 spec. */
    /* TODO: A8 quantize x_norm → q_int8 + scale; pass to BitLinear-equivalent.
     * For now, the substrate's existing matmul takes int8 X directly,
     * and we approximate by treating x_norm's MSB as int8. This is
     * NOT the BitNet protocol — work-unit 5 builds the actual A8 path. */

    /* QKV projections via packed-ternary matmul (gated on weights-loaded;
     * zero output if w->w_q is NULL — keeps harness runnable without
     * actual weights for skeleton testing).
     *
     * BitNet's W1.58A8 spec: x_norm gets A8-quantized to int8 (per-token
     * absmax), then matmul receives the int8. Q/K/V share x_norm as input
     * — we A8-quantize ONCE and reuse (RC-11 fix from work-unit 1 red-team:
     * computing the absmax separately for each projection wastes work
     * and risks numerical divergence if they produced different absmaxes).
     *
     * Note (RC-6): we call `m4t_ternary_5in8_matmul_bt` with non-ternary X
     * (int8 in [-127, 127] from A8 quantize, not {-1, 0, +1}). The kernel
     * uses SDOT which handles full int8 range correctly, but the documented
     * contract says X is ternary. This usage is intentional; future
     * cleanup (work-unit 5+) may add a substrate variant with a wider X
     * contract. */
    if (w->w_q != NULL) {
        /* Bit-faithful BitLinear: no a8 quantization. Q, K, V from x_norm. */
        bitnet_bitlinear_no_a8(s->q, s->x_norm, w->w_q,
                                w->alpha_q, w->alpha_q_block_exp,
                                BITNET_ACT_BX, BITNET_ACT_BX,
                                BITNET_HIDDEN_SIZE, BITNET_HIDDEN_SIZE);
        bitnet_bitlinear_no_a8(s->k, s->x_norm, w->w_k,
                                w->alpha_k, w->alpha_k_block_exp,
                                BITNET_ACT_BX, BITNET_ACT_BX,
                                BITNET_HIDDEN_SIZE, BITNET_KV_PROJ_DIM);
        bitnet_bitlinear_no_a8(s->v, s->x_norm, w->w_v,
                                w->alpha_v, w->alpha_v_block_exp,
                                BITNET_ACT_BX, BITNET_ACT_BX,
                                BITNET_HIDDEN_SIZE, BITNET_KV_PROJ_DIM);
    } else {
        /* No weights loaded — skeleton mode. Zero outputs. */
        memset(s->q, 0, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
        memset(s->k, 0, BITNET_KV_PROJ_DIM * sizeof(m4t_mtfp_t));
        memset(s->v, 0, BITNET_KV_PROJ_DIM * sizeof(m4t_mtfp_t));
    }

    /* Snapshot pre-RoPE Q, K for the dump (HF's q_proj/k_proj hooks
     * capture pre-RoPE; this lets the comparison be apples-to-apples). */
    memcpy(s->q_pre_rope, s->q, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
    memcpy(s->k_pre_rope, s->k, BITNET_KV_PROJ_DIM * sizeof(m4t_mtfp_t));

    /* RoPE on Q, K. */
    m4t_mtfp_rope_apply(s->q, s->k, position,
                        BITNET_NUM_ATTENTION_HEADS,
                        BITNET_NUM_KV_HEADS,
                        BITNET_HEAD_DIM,
                        BITNET_ROPE_THETA);

    /* Attention with KV cache (work-unit 7).
     *
     * Per Q head h:
     *   kv_head_idx = h / (num_q_heads / num_kv_heads)
     *   scores[t]   = Q[h] · K_cache[layer][t][kv_head_idx] / sqrt(head_dim)
     *                 for t ∈ [0, position+1)
     *   weights     = softmax(scores)        [length = position+1]
     *   attn_out[h] = Σ_t weights[t] · V_cache[layer][t][kv_head_idx]
     *
     * Score → softmax scale: softmax expects "1 LSB = 1 nat". Q · K
     * dot products in raw int can be enormous (up to 2^63ish). To bring
     * them into the LUT range [-30, 0] post-max-subtraction, we
     * right-shift the scores by enough bits to fit. This is a temperature
     * change relative to HF (softmax(x / T) ≠ softmax(x)) — the per-layer
     * ε comparison (work-unit 6) measures the resulting mismatch.
     *
     * 1/sqrt(head_dim) factor is folded into the rescale shift (it's a
     * constant downscale ≈ /11.3 = ~3.5 bits; absorbed into the heuristic). */
    {
        const int q_per_kv = BITNET_NUM_ATTENTION_HEADS / BITNET_NUM_KV_HEADS;

        if (cache == NULL) {
            /* No-cache fallback: degenerate seq=1 (attn_out = V). */
            for (int h = 0; h < BITNET_NUM_ATTENTION_HEADS; h++) {
                int kv_head = h / q_per_kv;
                memcpy(s->attn_output + (size_t)h * BITNET_HEAD_DIM,
                       s->v + (size_t)kv_head * BITNET_HEAD_DIM,
                       BITNET_HEAD_DIM * sizeof(m4t_mtfp_t));
            }
            memset(s->attn_scores, 0,
                   BITNET_NUM_ATTENTION_HEADS * sizeof(m4t_mtfp_t));
        } else {
            /* Write current K, V to cache at this layer's slot for `position`. */
            assert(layer_idx >= 0 && layer_idx < cache->n_layers);
            assert(position >= 0 && position < cache->max_seq_len);
            size_t row_size = (size_t)BITNET_NUM_KV_HEADS * BITNET_HEAD_DIM;
            size_t base = (size_t)layer_idx * cache->per_layer_stride
                          + (size_t)position * row_size;
            memcpy(cache->k + base, s->k, row_size * sizeof(m4t_mtfp_t));
            memcpy(cache->v + base, s->v, row_size * sizeof(m4t_mtfp_t));

            /* TRIT_ROUTING #1: if routed mode is active AND a fixed tau is
             * configured, lazy-allocate K signature cache and populate this
             * position's signatures. Per #4 finding: per-Q tau not load-
             * bearing for aggregate quality, so fixed-tau caching is safe.
             * TRIT_ROUTING #10: also populate when sigdist eviction is on. */
            int want_sig_cache = (g_attn_mode == BITNET_ATTN_ROUTED
                                  || g_kv_evict_mode == BITNET_KV_EVICT_SIGDIST
                                  || g_kv_evict_mode == BITNET_KV_EVICT_QSIGDIST
                                  || g_kv_evict_mode == BITNET_KV_EVICT_META
                                  || g_kv_evict_mode == BITNET_KV_EVICT_QSIG_FILTER)
                                  && g_attn_fixed_tau > 0
                                  && !g_attn_no_k_cache;
            if (want_sig_cache) {
                if (bitnet_kv_cache_ensure_sig(cache, g_attn_fixed_tau)) {
                    /* Cache was just (re-)allocated. Populate ALL positions
                     * up to and including the current one (current_pos may
                     * have positions already written by prior K-write before
                     * the cache was active). */
                    for (int p = 0; p <= position; p++) {
                        bitnet_kv_cache_store_k_sig(cache, layer_idx, p, g_attn_fixed_tau);
                    }
                } else {
                    bitnet_kv_cache_store_k_sig(cache, layer_idx, position, g_attn_fixed_tau);
                }
            }

            /* TRIT_ROUTING #10: trigger eviction if window exceeded.
             * Plan B (2026-05-12): pass s->q so QSIGDIST can use the
             * Q-direction; other modes ignore it. */
            if (g_kv_evict_mode != BITNET_KV_EVICT_NONE && g_kv_window > 0) {
                bitnet_kv_evict_apply(cache, layer_idx, position + 1, position, s->q);
            }

            int seq_k = position + 1;

            /* Per-head scratch hoisted out of the head loop (RC-2). */
            int64_t* scores_i64    = (int64_t*)   malloc((size_t)seq_k * sizeof(int64_t));
            m4t_mtfp_t* scores_int = (m4t_mtfp_t*)malloc((size_t)seq_k * sizeof(m4t_mtfp_t));
            m4t_mtfp_t* weights    = (m4t_mtfp_t*)malloc((size_t)seq_k * sizeof(m4t_mtfp_t));
            assert(scores_i64 && scores_int && weights);

            /* Cycle 2: branch by attention mode. Dense path is bit-exact
             * unchanged from production. Sparse arms (random/routed/oracle/
             * posracle/hybrid) use bitnet_sparse_attn_v_combine which gathers
             * V[indices] into a contiguous buffer and calls the NEON
             * m4t_mtfp_attn_v_combine on it. Production-eligible per the
             * "no scalar in production" foundational rule (2026-05-12). */
            int sparse_active = (g_attn_mode != BITNET_ATTN_DENSE && g_attn_k < seq_k);

            /* Attention-only timing: accumulate per-head-loop wall-clock into
             * g_attn_seconds. Lets us attribute substrate vs dense diffs to
             * attention specifically. */
            struct timespec attn_t0, attn_t1;
            clock_gettime(CLOCK_MONOTONIC, &attn_t0);

            for (int h = 0; h < BITNET_NUM_ATTENTION_HEADS; h++) {
                int kv_head = h / q_per_kv;
                const m4t_mtfp_t* qh = s->q + (size_t)h * BITNET_HEAD_DIM;
                m4t_mtfp_t* out_h = s->attn_output + (size_t)h * BITNET_HEAD_DIM;
                const m4t_mtfp_t* V_base = cache->v
                    + (size_t)layer_idx * cache->per_layer_stride
                    + (size_t)kv_head * BITNET_HEAD_DIM;

                if (!sparse_active) {
                    /* ── DENSE PATH (production) — unchanged from pre-Cycle 2. */
                    /* Compute scores[t] = dot(qh, K_cache[layer][t][kv_head]). */
                    int64_t max_abs = 1;
                    for (int t = 0; t < seq_k; t++) {
                        size_t k_row_base = (size_t)layer_idx * cache->per_layer_stride
                                            + (size_t)t * row_size
                                            + (size_t)kv_head * BITNET_HEAD_DIM;
                        const m4t_mtfp_t* kh = cache->k + k_row_base;
                        /* V14.A: NEON int32×int32→int64 dot via libm4t helper.
                         * Same semantics as the prior scalar loop, but the
                         * production code path is NEON-only per condition (5)
                         * of the pure-ternary directive. */
                        int64_t acc = m4t_mtfp_vec_dot_i64(qh, kh, BITNET_HEAD_DIM);
                        scores_i64[t] = acc;
                        int64_t a = acc < 0 ? -acc : acc;
                        if (a > max_abs) max_abs = a;
                    }

                    /* Rescale scores; see tuning history at score_shift below.
                     * Per journal/math_div_atomics_2026-05-10.md (gate1+fudge2). */
                    int score_shift = 0;
                    while ((max_abs >> score_shift) > 30) score_shift++;
                    score_shift += 2;

                    for (int t = 0; t < seq_k; t++) {
                        int64_t r;
                        if (scores_i64[t] >= 0) r = scores_i64[t] >> score_shift;
                        else                    r = -((-scores_i64[t]) >> score_shift);
                        if (r >  M4T_MTFP_MAX_VAL) r =  M4T_MTFP_MAX_VAL;
                        if (r < -M4T_MTFP_MAX_VAL) r = -M4T_MTFP_MAX_VAL;
                        scores_int[t] = (m4t_mtfp_t)r;
                    }

                    /* TRIT_ROUTING #10: mask evicted positions to large
                     * negative score so softmax weight ≈ 0. */
                    if (cache->evicted != NULL) {
                        const uint8_t* evrow = cache->evicted
                            + (size_t)layer_idx * (size_t)cache->max_seq_len;
                        for (int t = 0; t < seq_k; t++) {
                            if (evrow[t]) scores_int[t] = (m4t_mtfp_t)(-M4T_MTFP_MAX_VAL);
                        }
                    }

                    m4t_mtfp_softmax(weights, scores_int, seq_k);

                    /* V14.B: NEON outer-product accumulate via libm4t helper.
                     * Loop order swapped (t outer, d inner) for NEON
                     * vmlal_s32 broadcast pattern. */
                    m4t_mtfp_attn_v_combine(out_h, 30, weights, V_base, row_size,
                                            seq_k, BITNET_HEAD_DIM);
                } else {
                    /* ── SPARSE PATH (Cycle 2, production-eligible, opt-in) ── */
                    int k_eff = g_attn_k;  /* sparse_active gate ensures < seq_k */

                    int* indices = (int*)malloc((size_t)k_eff * sizeof(int));
                    int64_t* sub_scores_i64 = (int64_t*)malloc((size_t)k_eff * sizeof(int64_t));
                    m4t_mtfp_t* sub_scores_int = (m4t_mtfp_t*)malloc((size_t)k_eff * sizeof(m4t_mtfp_t));
                    m4t_mtfp_t* sub_weights = (m4t_mtfp_t*)malloc((size_t)k_eff * sizeof(m4t_mtfp_t));
                    assert(indices && sub_scores_i64 && sub_scores_int && sub_weights);

                    /* Pick indices per arm. RANDOM, ROUTED, ORACLE, POSRACLE,
                     * HYBRID. */
                    int64_t max_abs = 1;
                    if (g_attn_mode == BITNET_ATTN_HYBRID) {
                        /* TRIT_ROUTING #3: HYBRID two-stage routing.
                         * Stage 1: signature distance picks top-k1 candidates.
                         * Stage 2: signed Q·K picks top-k2 = k_eff < k1 from
                         *          the shortlist (precise refinement). */
                        int k1 = g_attn_k1 > 0 ? g_attn_k1 : (4 * k_eff);
                        if (k1 > seq_k) k1 = seq_k;
                        if (k1 < k_eff) k1 = k_eff;

                        /* Stage 1: routed candidate set. */
                        int* candidates = (int*)malloc((size_t)k1 * sizeof(int));
                        const m4t_mtfp_t* k_cache_layer =
                            cache->k + (size_t)layer_idx * cache->per_layer_stride;
                        bitnet_pick_routed_indices(
                            candidates, k1, seq_k,
                            qh, k_cache_layer, row_size, kv_head,
                            BITNET_HEAD_DIM,
                            cache, layer_idx);

                        /* Stage 2: compute true Q·K on the k1 candidates. */
                        int64_t* cand_scores = (int64_t*)malloc((size_t)k1 * sizeof(int64_t));
                        for (int i = 0; i < k1; i++) {
                            int t = candidates[i];
                            size_t k_row_base = (size_t)layer_idx * cache->per_layer_stride
                                                + (size_t)t * row_size
                                                + (size_t)kv_head * BITNET_HEAD_DIM;
                            const m4t_mtfp_t* kh = cache->k + k_row_base;
                            cand_scores[i] = m4t_mtfp_vec_dot_i64(qh, kh, BITNET_HEAD_DIM);
                        }

                        /* Stage 2 sort: top-k_eff by signed score (positives first). */
                        int* relative_top = (int*)malloc((size_t)k1 * sizeof(int));
                        bitnet_pick_posracle_topk(relative_top, k_eff, k1, cand_scores);

                        /* Translate relative indices in [0, k1) → absolute in [0, seq_k). */
                        for (int i = 0; i < k_eff; i++) {
                            indices[i] = candidates[relative_top[i]];
                            sub_scores_i64[i] = cand_scores[relative_top[i]];
                            int64_t a = sub_scores_i64[i] < 0 ? -sub_scores_i64[i] : sub_scores_i64[i];
                            if (a > max_abs) max_abs = a;
                        }
                        free(relative_top); free(cand_scores); free(candidates);
                    } else if (g_attn_mode == BITNET_ATTN_ORACLE ||
                        g_attn_mode == BITNET_ATTN_POSRACLE) {
                        /* ORACLE / POSRACLE: compute dense scores first, then pick top-k.
                         * ORACLE picks by |score|; POSRACLE by signed score (positives win). */
                        for (int t = 0; t < seq_k; t++) {
                            size_t k_row_base = (size_t)layer_idx * cache->per_layer_stride
                                                + (size_t)t * row_size
                                                + (size_t)kv_head * BITNET_HEAD_DIM;
                            const m4t_mtfp_t* kh = cache->k + k_row_base;
                            scores_i64[t] = m4t_mtfp_vec_dot_i64(qh, kh, BITNET_HEAD_DIM);
                        }
                        if (g_attn_mode == BITNET_ATTN_ORACLE)
                            bitnet_pick_oracle_topk  (indices, k_eff, seq_k, scores_i64);
                        else
                            bitnet_pick_posracle_topk(indices, k_eff, seq_k, scores_i64);
                        /* Pull the chosen scores into the sub buffer. */
                        for (int i = 0; i < k_eff; i++) {
                            sub_scores_i64[i] = scores_i64[indices[i]];
                            int64_t a = sub_scores_i64[i] < 0 ? -sub_scores_i64[i] : sub_scores_i64[i];
                            if (a > max_abs) max_abs = a;
                        }
                    } else {
                        /* RANDOM and ROUTED — pick indices, then compute true scores. */
                        if (g_attn_mode == BITNET_ATTN_RANDOM) {
                            bitnet_pick_random_indices(indices, k_eff, seq_k, &g_attn_rng);
                        } else if (g_attn_mode == BITNET_ATTN_ROUTED) {
                            const m4t_mtfp_t* k_cache_layer =
                                cache->k + (size_t)layer_idx * cache->per_layer_stride;
                            bitnet_pick_routed_indices(
                                indices, k_eff, seq_k,
                                qh, k_cache_layer, row_size, kv_head,
                                BITNET_HEAD_DIM,
                                cache, layer_idx);
                        }
                        /* Compute true Q·K dot at the chosen indices only. */
                        for (int i = 0; i < k_eff; i++) {
                            int t = indices[i];
                            size_t k_row_base = (size_t)layer_idx * cache->per_layer_stride
                                                + (size_t)t * row_size
                                                + (size_t)kv_head * BITNET_HEAD_DIM;
                            const m4t_mtfp_t* kh = cache->k + k_row_base;
                            int64_t acc = m4t_mtfp_vec_dot_i64(qh, kh, BITNET_HEAD_DIM);
                            sub_scores_i64[i] = acc;
                            int64_t a = acc < 0 ? -acc : acc;
                            if (a > max_abs) max_abs = a;
                        }
                    }

                    int score_shift = 0;
                    while ((max_abs >> score_shift) > 30) score_shift++;
                    score_shift += 2;

                    for (int i = 0; i < k_eff; i++) {
                        int64_t r;
                        if (sub_scores_i64[i] >= 0) r = sub_scores_i64[i] >> score_shift;
                        else                        r = -((-sub_scores_i64[i]) >> score_shift);
                        if (r >  M4T_MTFP_MAX_VAL) r =  M4T_MTFP_MAX_VAL;
                        if (r < -M4T_MTFP_MAX_VAL) r = -M4T_MTFP_MAX_VAL;
                        sub_scores_int[i] = (m4t_mtfp_t)r;
                    }

                    m4t_mtfp_softmax(sub_weights, sub_scores_int, k_eff);

                    bitnet_sparse_attn_v_combine(out_h, 30, sub_weights,
                                                  V_base, row_size,
                                                  k_eff, BITNET_HEAD_DIM,
                                                  indices);

                    free(indices); free(sub_scores_i64);
                    free(sub_scores_int); free(sub_weights);
                }
            }

            clock_gettime(CLOCK_MONOTONIC, &attn_t1);
            g_attn_seconds += (double)(attn_t1.tv_sec - attn_t0.tv_sec)
                            + (double)(attn_t1.tv_nsec - attn_t0.tv_nsec) / 1e9;

            /* Stash one debug score per head (last position's score),
             * for the activation dump. */
            for (int h = 0; h < BITNET_NUM_ATTENTION_HEADS; h++) {
                s->attn_scores[h] = 0;  /* not load-bearing in tests */
            }

            free(scores_i64); free(scores_int); free(weights);
        }
    }

    /* attn_sub_norm: y = γ · y · rsqrt(mean(y²) + ε) */
    m4t_mtfp_rmsnorm_bx(s->attn_sub_norm, s->attn_output,
                        w->gamma_attn_sub_norm,
                        BITNET_ACT_BX, w->gamma_attn_sub_norm_block_exp,
                        BITNET_ACT_BX, /*eps=*/1, BITNET_HIDDEN_SIZE);

    /* O projection: y = attn_sub_norm @ W_o^T (BitLinear, A8-quantized).
     * Per-projection input — own A8 quantize. */
    if (w->w_o != NULL) {
        bitnet_bitlinear_no_a8(s->x, s->attn_sub_norm, w->w_o,
                                w->alpha_o, w->alpha_o_block_exp,
                                BITNET_ACT_BX, BITNET_ACT_BX,
                                BITNET_HIDDEN_SIZE, BITNET_HIDDEN_SIZE);
    } else {
        memcpy(s->x, s->attn_sub_norm, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
    }

    /* DEBUG: dump o_proj output (pre-residual-add) for layer-by-layer
     * substrate-vs-HF localization. Off by default; set DEBUG_DUMP_OPROJ env. */
    {
        const char* dbg = getenv("DEBUG_DUMP_OPROJ");
        if (dbg && layer_idx == 0) {
            FILE* f = fopen(dbg, "wb");
            if (f) { fwrite(s->x, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f); fclose(f); }
        }
    }

    /* x = residual + x (V13.A: NEON via libm4t's saturating-add
     * primitive — same MTFP19 clamp semantics as the previous scalar
     * loop, no scalar production code per condition (5) of the
     * pure-ternary directive). */
    m4t_mtfp_vec_add_inplace(s->x, s->residual, BITNET_HIDDEN_SIZE);

    /* DEBUG: dump post-residual-add (= input to post_attn_norm). */
    {
        const char* dbg = getenv("DEBUG_DUMP_PREPN");
        if (dbg && layer_idx == 0) {
            FILE* f = fopen(dbg, "wb");
            if (f) { fwrite(s->x, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f); fclose(f); }
        }
    }

    /* ── FFN sub-block ────────────────────────────────────────────── */

    /* residual = x */
    memcpy(s->residual, s->x, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));

    /* x_norm = post_attention_layernorm(x) */
    m4t_mtfp_rmsnorm_bx(s->x_norm, s->x, w->gamma_post_attn_norm,
                        BITNET_ACT_BX, w->gamma_post_attn_norm_block_exp,
                        BITNET_ACT_BX, /*eps=*/1, BITNET_HIDDEN_SIZE);

    /* Trit Lattice LSH FFN validation: dump x_norm (FFN input) per
     * (prompt, position, layer) when enabled. Read-only on the
     * inference path; semantic effect is zero. See BITNET_DUMP_FFN_INPUTS_*
     * env-var docs above. */
    if (g_dump_ffn_inputs_any &&
        g_dump_ffn_inputs_dir != NULL &&
        layer_idx >= 0 && layer_idx < BITNET_DUMP_MAX_LAYERS &&
        g_dump_ffn_inputs_layer_mask[layer_idx]) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s_p%04d_l%02d.bin",
                 g_dump_ffn_inputs_dir, g_dump_label, position, layer_idx);
        FILE* f = fopen(path, "wb");
        if (f) {
            fwrite(s->x_norm, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
            fclose(f);
        } else {
            fprintf(stderr, "[harness] WARN: failed to open %s for FFN dump\n", path);
        }
    }

    /* Routed FFN — Step 1 PoC. If LSH FFN is active for this layer AND
     * the bucket has a calibrated recipe (recipe_len > 0): compute
     * s->x = mu + sum(scale_j × atoms[idx_j]); skip dense FFN.
     * Otherwise (uncalibrated bucket OR layer not active): fall through
     * to the dense compute below.
     *
     * Step 1.5a hybrid: buckets whose recipe was skipped at calibration
     * (n < n_min) have recipe_len = 0 in the dict → harness falls back
     * to dense. This preserves dense compute for poorly-populated
     * buckets while routing well-populated ones. */
    if (g_lsh_any_active && layer_idx >= 0 && layer_idx < BITNET_DUMP_MAX_LAYERS &&
        g_lsh_active_layers[layer_idx] && g_lsh_dict.by_layer_idx[layer_idx] != NULL) {
        const bitnet_lsh_layer_t* L = g_lsh_dict.by_layer_idx[layer_idx];
        /* Threshold-extract first k_lsh trits of x_norm to compute bucket. */
        uint64_t bucket = 0; uint64_t pow3 = 1;
        const int32_t tau = g_lsh_dict.tau;
        for (uint32_t i = 0; i < g_lsh_dict.k_lsh; i++) {
            int32_t v = (int32_t)s->x_norm[i];
            int trit;
            if (v > tau)       trit = 2;  /* +1 → digit 2 */
            else if (v < -tau) trit = 0;  /* -1 → digit 0 */
            else               trit = 1;  /*  0 → digit 1 */
            bucket += (uint64_t)trit * pow3;
            pow3 *= 3;
        }
        const bitnet_lsh_recipe_t* R = &L->by_bucket[bucket];
        /* If recipe_len == 0, this bucket was uncalibrated (or skipped
         * for low sample count). Fall through to dense FFN. */
        if (R->recipe_len > 0) {
            /* Predict s->x = mu + sum(scale_j × atoms[atom_idx_j]). */
            for (uint32_t d = 0; d < g_lsh_dict.d_model; d++) {
                double acc = (double)L->mu[d];
                for (uint32_t j = 0; j < R->recipe_len; j++) {
                    int8_t a = L->atoms[R->atom_idx[j] * g_lsh_dict.d_model + d];
                    if (a != 0) acc += R->scale[j] * (double)a;
                }
                /* Clamp to mtfp range; round to nearest. */
                if (acc > 2147483647.0) acc = 2147483647.0;
                if (acc < -2147483647.0) acc = -2147483647.0;
                s->x[d] = (m4t_mtfp_t)(acc + (acc >= 0 ? 0.5 : -0.5));
            }
            /* Skip the dense FFN compute; jump to residual add. */
            goto bitnet_block_ffn_residual;
        }
        /* else: fall through to dense compute */
    }

    /* gate, up = x_norm projected (BitLinear, A8). They share x_norm as
     * input — A8-quantize ONCE and reuse (RC-11 fix). */
    if (w->w_gate != NULL && w->w_up != NULL) {
        /* gate, up at FFN_BX (wider headroom; gate values typical ~80). */
        bitnet_bitlinear_no_a8(s->gate, s->x_norm, w->w_gate,
                                w->alpha_gate, w->alpha_gate_block_exp,
                                BITNET_ACT_BX, BITNET_FFN_BX,
                                BITNET_HIDDEN_SIZE, BITNET_INTERMEDIATE_SIZE);
        bitnet_bitlinear_no_a8(s->up, s->x_norm, w->w_up,
                                w->alpha_up, w->alpha_up_block_exp,
                                BITNET_ACT_BX, BITNET_FFN_BX,
                                BITNET_HIDDEN_SIZE, BITNET_INTERMEDIATE_SIZE);
    } else {
        memset(s->gate, 0, BITNET_INTERMEDIATE_SIZE * sizeof(m4t_mtfp_t));
        memset(s->up,   0, BITNET_INTERMEDIATE_SIZE * sizeof(m4t_mtfp_t));
    }

    /* gate_act = relu²(gate) × up. gate at FFN_BX, relu² stays at FFN_BX
     * (gate²_real ≤ ~10K fits MTFP19_MAX/3^8 ≈ 88K). Mul with up (FFN_BX)
     * produces products up to gate²×up ≈ 6M for outliers — needs the wider
     * GATE_ACT_BX. */
    m4t_mtfp_relu2_inplace_bx(s->gate, BITNET_FFN_BX, BITNET_FFN_BX,
                              BITNET_INTERMEDIATE_SIZE);
    m4t_mtfp_elementwise_mul_bx(s->gate_act,
                                s->gate, BITNET_FFN_BX,
                                s->up,   BITNET_FFN_BX,
                                BITNET_GATE_ACT_BX,
                                BITNET_INTERMEDIATE_SIZE);

    /* TRIT_ROUTING #8 falsification probe: slice-mask gate_act in place
     * when BITNET_FFN_MODE ∈ {oracle, random}. Default dense = no-op. */
    if (g_ffn_mode != BITNET_FFN_DENSE) {
        bitnet_ffn_apply_slice_mask(s->gate_act, g_ffn_num_experts, g_ffn_k);
    }
    /* TRIT_ROUTING #9 falsification probe: cell-mask gate_act in place
     * when BITNET_FFN_CELL_MODE ∈ {oracle, random}. Default dense = no-op. */
    if (g_ffn_cell_mode != BITNET_FFN_CELL_DENSE && g_ffn_cell_keep > 0) {
        bitnet_ffn_apply_cell_mask(s->gate_act, g_ffn_cell_keep);
    }

    /* ffn_sub_norm(gate_act). gate_act is at GATE_ACT_BX. Output at
     * ACT_BX (resumes linear-magnitude flow for down_proj). */
    m4t_mtfp_rmsnorm_bx(s->ffn_sub_norm, s->gate_act,
                        w->gamma_ffn_sub_norm,
                        BITNET_GATE_ACT_BX, w->gamma_ffn_sub_norm_block_exp,
                        BITNET_ACT_BX, /*eps=*/1, BITNET_INTERMEDIATE_SIZE);

    /* down = ffn_sub_norm @ W_down^T (BitLinear, A8). Different input than
     * gate/up — own A8 quantize. ffn_sub_norm output is at ACT_BX. */
    if (w->w_down != NULL) {
        bitnet_bitlinear_no_a8(s->x, s->ffn_sub_norm, w->w_down,
                                w->alpha_down, w->alpha_down_block_exp,
                                BITNET_ACT_BX, BITNET_ACT_BX,
                                BITNET_INTERMEDIATE_SIZE, BITNET_HIDDEN_SIZE);
    } else {
        memset(s->x, 0, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
    }

    /* B1.6 — FFN output dump (post-down_proj, pre-residual). Mirror of
     * the FFN-input dump: same env vars, same per-(prompt, position,
     * layer) file naming with "_out" suffix. Captures the dense FFN's
     * contribution that an LSH FFN tile would need to predict. */
    if (g_dump_ffn_inputs_any &&
        g_dump_ffn_inputs_dir != NULL &&
        layer_idx >= 0 && layer_idx < BITNET_DUMP_MAX_LAYERS &&
        g_dump_ffn_inputs_layer_mask[layer_idx]) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s_p%04d_l%02d_out.bin",
                 g_dump_ffn_inputs_dir, g_dump_label, position, layer_idx);
        FILE* f = fopen(path, "wb");
        if (f) {
            fwrite(s->x, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
            fclose(f);
        }
    }

bitnet_block_ffn_residual:
    /* x = residual + x (V13.A: NEON via libm4t's saturating-add
     * primitive — same MTFP19 clamp semantics as the previous scalar
     * loop, no scalar production code per condition (5) of the
     * pure-ternary directive). */
    m4t_mtfp_vec_add_inplace(s->x, s->residual, BITNET_HIDDEN_SIZE);

    /* Copy back to caller's buffer. */
    memcpy(x_io, s->x, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
}

/* ── main ─────────────────────────────────────────────────────────── */

/* Per-layer activation dump: writes captured tensors to a binary file
 * compatible with the comparison driver (scripts/compare_activations.py).
 *
 * Format: one .bin per (layer, sublayer) capture site, containing raw
 * int32 mantissas. A small JSON sidecar lists shapes + block_exps (TBD).
 *
 * Phase 1 simplification: just dump a few key sites to one consolidated
 * file. Detailed sublayer dumps are work-unit 6+ scope. */
static int dump_activations_to_file(
    const char* path,
    const bitnet_block_scratch_t* s,
    int layer)
{
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[harness] cannot open %s for write\n", path);
        return 1;
    }
    /* Header: magic "ACTV2" (bumped from ACTV in red-team), layer,
     * sizes. v2 adds x_norm_input, q_pre_rope, k_pre_rope captures. */
    fwrite("ACTV2", 1, 5, f);
    /* 3-byte pad to keep 4-byte alignment for the int32s that follow. */
    char pad[3] = {0, 0, 0}; fwrite(pad, 1, 3, f);
    int32_t li = layer;
    fwrite(&li, sizeof(int32_t), 1, f);
    int32_t hidden = BITNET_HIDDEN_SIZE;
    int32_t intermediate = BITNET_INTERMEDIATE_SIZE;
    int32_t kv_proj = BITNET_KV_PROJ_DIM;
    fwrite(&hidden, sizeof(int32_t), 1, f);
    fwrite(&intermediate, sizeof(int32_t), 1, f);
    fwrite(&kv_proj, sizeof(int32_t), 1, f);

    /* Capture order (matches scripts/compare_activations.py CAPTURE_ORDER): */
    /* 1. x_norm_input — true post-input_layernorm output */
    fwrite(s->x_norm_input, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    /* 2. q_pre_rope — post-q_proj+α scale, before RoPE */
    fwrite(s->q_pre_rope, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    /* 3. k_pre_rope — post-k_proj+α scale, before RoPE */
    fwrite(s->k_pre_rope, sizeof(m4t_mtfp_t), BITNET_KV_PROJ_DIM, f);
    /* 4. v — post-v_proj+α scale (no RoPE on V) */
    fwrite(s->v, sizeof(m4t_mtfp_t), BITNET_KV_PROJ_DIM, f);
    /* 5. q_post_rope — post-RoPE Q (no HF analog hooked) */
    fwrite(s->q, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    /* 6. k_post_rope — post-RoPE K */
    fwrite(s->k, sizeof(m4t_mtfp_t), BITNET_KV_PROJ_DIM, f);
    /* 7. attn_sub_norm output */
    fwrite(s->attn_sub_norm, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    /* 8. x_norm — post-attn-residual rmsnorm input (post_attention_layernorm output) */
    fwrite(s->x_norm, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    /* 9. gate (NB: post-relu²-inplace; raw gate_proj NOT captured) */
    fwrite(s->gate, sizeof(m4t_mtfp_t), BITNET_INTERMEDIATE_SIZE, f);
    /* 10. up (post-up_proj+α scale, no relu²) */
    fwrite(s->up, sizeof(m4t_mtfp_t), BITNET_INTERMEDIATE_SIZE, f);
    /* 11. ffn_sub_norm output */
    fwrite(s->ffn_sub_norm, sizeof(m4t_mtfp_t), BITNET_INTERMEDIATE_SIZE, f);
    /* 12. block_output */
    fwrite(s->x, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    fclose(f);
    return 0;
}

/* Embedding lookup: x[i] = embedding[token_id × HIDDEN + i] as MTFP19
 * mantissas. The block_exp of the embedding is set at conversion time
 * (per-tensor). Caller must hold block_exp tracking externally if needed.
 *
 * Phase 1 simplification: we read mantissas directly as activation
 * values, treating everything as "block_exp 0" through the forward
 * pass. This loses precision relative to HF's bf16/fp32 but produces
 * a consistent, deterministic substrate output. Per-layer ε
 * comparison (work-unit 6's gate) measures how the discrepancy
 * accumulates. */
static void bitnet_embed(
    m4t_mtfp_t* x_out,
    const m4t_mtfp_t* embedding,
    int embedding_bx,
    int token_id)
{
    if (embedding == NULL) {
        for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) x_out[i] = 0;
        return;
    }
    const m4t_mtfp_t* row = embedding + (size_t)token_id * BITNET_HIDDEN_SIZE;
    /* Phase 2 work-unit 1: rescale embedding row from its stored bx
     * into the activation flow's BITNET_ACT_BX. */
    m4t_mtfp_rescale_bx(x_out, row, embedding_bx, BITNET_ACT_BX,
                        BITNET_HIDDEN_SIZE);
}

/* LM head: logits = x @ lm_head^T. lm_head is [VOCAB × HIDDEN] MTFP19
 * mantissas. Output: logits at scale (lm_head_block_exp + x_block_exp)
 * in mantissa units; magnitudes can grow; we don't compute argmax in C
 * (the comparison driver does that against HF reference output). */
static void bitnet_lm_head(
    m4t_mtfp_t* logits_out,
    const m4t_mtfp_t* x,
    const m4t_mtfp_t* lm_head,
    int top_n)
{
    /* For comparison purposes we only need logits[0..top_n) — full vocab
     * (128256) is dumped to file by the comparison driver if needed.
     *
     * V13.B of pure-ternary audit: dot product via libm4t's NEON
     * helper (vmlal_s32 chain), no scalar production code per
     * condition (5) of the directive. */
    for (int v = 0; v < top_n; v++) {
        const m4t_mtfp_t* row = lm_head + (size_t)v * BITNET_HIDDEN_SIZE;
        int64_t acc = m4t_mtfp_vec_dot_i64(x, row, BITNET_HIDDEN_SIZE);
        /* Crude scale-down to fit MTFP19; the comparison driver consumes
         * raw acc values for ε measurement so we expose them at int64
         * in a separate accumulator buffer (caller supplied). */
        int64_t scaled = acc >> 30;  /* somewhat arbitrary; rescaled by Python comparison */
        logits_out[v] = m4t_mtfp_clamp64(scaled);
    }
}

/* Argmax over full vocabulary using raw int64 logits (no shift).
 * Phase 1 greedy decoding: returns the token id with highest logit.
 * Returns -1 if lm_head is NULL. */
static int bitnet_argmax_full_vocab(
    const m4t_mtfp_t* x,
    const m4t_mtfp_t* lm_head)
{
    if (lm_head == NULL) return -1;
    int64_t best_acc = INT64_MIN;
    int     best_v   = 0;
    /* V13.B of pure-ternary audit: per-vocab dot product via libm4t's
     * NEON helper. No scalar production code per condition (5). */
    for (int v = 0; v < BITNET_VOCAB_SIZE; v++) {
        const m4t_mtfp_t* row = lm_head + (size_t)v * BITNET_HIDDEN_SIZE;
        int64_t acc = m4t_mtfp_vec_dot_i64(x, row, BITNET_HIDDEN_SIZE);
        if (acc > best_acc) { best_acc = acc; best_v = v; }
    }
    return best_v;
}

/* Top-2 over full vocab. Used by per-step telemetry to expose argmax
 * margin (top1_acc - top2_acc) so downstream analysis can tell whether
 * eviction perturbations are within the safety margin of the argmax
 * decision (Phase ζ atomic investigation). Returns 0 on success, -1
 * if lm_head is NULL. */
static int bitnet_top2_full_vocab(
    const m4t_mtfp_t* x,
    const m4t_mtfp_t* lm_head,
    int* top1_tok, int64_t* top1_acc,
    int* top2_tok, int64_t* top2_acc)
{
    if (lm_head == NULL) return -1;
    int64_t a1 = INT64_MIN, a2 = INT64_MIN;
    int v1 = 0, v2 = 0;
    for (int v = 0; v < BITNET_VOCAB_SIZE; v++) {
        const m4t_mtfp_t* row = lm_head + (size_t)v * BITNET_HIDDEN_SIZE;
        int64_t acc = m4t_mtfp_vec_dot_i64(x, row, BITNET_HIDDEN_SIZE);
        if (acc > a1) {
            a2 = a1; v2 = v1;
            a1 = acc; v1 = v;
        } else if (acc > a2) {
            a2 = acc; v2 = v;
        }
    }
    *top1_tok = v1; *top1_acc = a1;
    *top2_tok = v2; *top2_acc = a2;
    return 0;
}

int main(int argc, char** argv) {
    /* Cycle 2: read sparse-attention mode from env. No-op when env unset. */
    bitnet_attn_mode_init_from_env();
    /* TRIT_ROUTING #8 falsification probe init. */
    bitnet_ffn_mode_init_from_env();
    /* TRIT_ROUTING #10: KV eviction init. */
    bitnet_kv_evict_init_from_env();

    int token_id = 1;          /* default: BOS-like token */
    int n_layers = -1;         /* -1 = all loaded layers */
    int n_positions = 1;       /* number of positions to forward (work-unit 7 cache) */
    /* Sized to BitNet's max_seq = 4096 so long-context measurements aren't
     * silently truncated. Was 256; caught 2026-05-12 during cost re-test. */
    int prompt_tokens[4096] = {0};
    int n_prompt_tokens = 0;   /* if >0, overrides --token + --positions */
    int n_generate = 0;        /* generation steps after the prompt (work-unit 8) */
    const char* weights_arg = NULL;
    const char* dump_path = NULL;

    /* Simple positional + flag parsing. */
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <weights_blob.bin|-> "
            "[--token <id>] [--layers <n>] [--positions <p>] [--dump <path>]\n"
            "  weights_blob.bin — produced by scripts/convert_weights.py.\n"
            "                     '-' for skeleton mode.\n"
            "  --token <id>     — input token id (default: 1).\n"
            "  --layers <n>     — number of transformer layers to run\n"
            "                     (default: all loaded).\n"
            "  --positions <p>  — number of forward passes to run\n"
            "                     (positions 0..p-1; default 1). Exercises\n"
            "                     the KV cache across multiple decode steps.\n"
            "  --gen <n>        — greedy-generate <n> tokens after the\n"
            "                     prompt (work-unit 8). Default 0 (no\n"
            "                     generation). Each step: forward, argmax\n"
            "                     over LM head, embed, repeat.\n"
            "  --dump <path>    — write per-layer activation snapshots to\n"
            "                     <path>.layer<N>.bin (last position) for\n"
            "                     ε comparison.\n",
            argv[0]);
        return 1;
    }
    weights_arg = argv[1];
    for (int i = 2; i + 1 < argc; i += 2) {
        if      (strcmp(argv[i], "--token")     == 0) token_id    = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--layers")    == 0) n_layers    = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--positions") == 0) n_positions = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--prompt-tokens") == 0) {
            /* Parse comma-separated token ids. Hard error if input exceeds
             * the buffer — silent truncation was the v1 cost-measurement bug
             * (caught 2026-05-12). */
            const int max_prompt = (int)(sizeof(prompt_tokens)/sizeof(prompt_tokens[0]));
            const char* p = argv[i+1];
            n_prompt_tokens = 0;
            while (*p && n_prompt_tokens < max_prompt) {
                prompt_tokens[n_prompt_tokens++] = atoi(p);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
            if (*p) {
                fprintf(stderr,
                    "[harness] ERROR: --prompt-tokens exceeds buffer (max %d tokens); "
                    "remaining input not parsed. Rebuild with larger prompt_tokens[] or "
                    "split the prompt.\n", max_prompt);
                return 2;
            }
            n_positions = n_prompt_tokens;
        }
        else if (strcmp(argv[i], "--gen")       == 0) n_generate  = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--dump")      == 0) dump_path   = argv[i+1];
        else { fprintf(stderr, "[harness] unknown flag: %s\n", argv[i]); return 1; }
    }
    if (n_positions < 1) n_positions = 1;
    if (n_generate  < 0) n_generate  = 0;

    bitnet_weights_t weights = {0};
    bitnet_weights_loaded_t handle = {0};
    int loaded_ok = 0;
    if (strcmp(weights_arg, "-") != 0) {
        if (bitnet_weights_load(weights_arg, &weights, &handle) == 0) {
            loaded_ok = 1;
            fprintf(stderr, "[harness] loaded weights from %s\n", weights_arg);
        } else {
            fprintf(stderr, "[harness] weight load failed; running skeleton mode\n");
        }
    } else {
        fprintf(stderr, "[harness] skeleton mode (no weights)\n");
    }

    /* Allocate scratch. */
    bitnet_block_scratch_t s = {0};
    bitnet_block_scratch_alloc(&s);

    /* Input: embed token_id (or fallback pattern in skeleton mode). */
    m4t_mtfp_t x[BITNET_HIDDEN_SIZE];
    if (loaded_ok && weights.embedding != NULL) {
        bitnet_embed(x, weights.embedding, weights.embedding_block_exp, token_id);
    } else {
        for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) x[i] = (i % 7) - 3;
    }

    /* Determine layer count to run. */
    int layers_to_run = (n_layers >= 0) ? n_layers : BITNET_NUM_LAYERS;
    if (layers_to_run > BITNET_NUM_LAYERS) layers_to_run = BITNET_NUM_LAYERS;

    /* Skeleton fallback weights for layers without loaded weights. */
    bitnet_layer_weights_t w_dummy = {0};
    static m4t_mtfp_t gamma_dummy[BITNET_INTERMEDIATE_SIZE];
    static m4t_mtfp_t alpha_dummy = 0;  /* triggers no-op scale in apply helper */
    for (int i = 0; i < BITNET_INTERMEDIATE_SIZE; i++) gamma_dummy[i] = 1;
    w_dummy.gamma_input_norm     = gamma_dummy;
    w_dummy.gamma_post_attn_norm = gamma_dummy;
    w_dummy.gamma_attn_sub_norm  = gamma_dummy;
    w_dummy.gamma_ffn_sub_norm   = gamma_dummy;
    w_dummy.alpha_q = w_dummy.alpha_k = w_dummy.alpha_v = w_dummy.alpha_o
        = w_dummy.alpha_gate = w_dummy.alpha_up = w_dummy.alpha_down = &alpha_dummy;

    /* KV cache (work-unit 7). Sized for prompt-positions + generation. */
    bitnet_kv_cache_t cache = {0};
    int total_positions = n_positions + n_generate;
    int max_seq = total_positions > 256 ? total_positions : 256;
    if (bitnet_kv_cache_alloc(&cache, max_seq, layers_to_run) != 0) {
        fprintf(stderr, "[harness] KV cache alloc failed\n");
        return 1;
    }

    /* Inner forward pass for one position. Mutates x in place; assumes
     * cache, scratch, weights, and x are set up by the caller. */
    #define FORWARD_ONE(pos_, dump_this) do {                              \
        for (int l = 0; l < layers_to_run; l++) {                          \
            const bitnet_layer_weights_t* w_l;                             \
            if (loaded_ok && weights.layers[l].w_q != NULL)                \
                w_l = &weights.layers[l];                                  \
            else                                                            \
                w_l = &w_dummy;                                            \
            bitnet_forward_block(x, w_l, &s, &cache, l, (pos_));           \
            if (dump_path && (dump_this)) {                                \
                char path[1024];                                           \
                snprintf(path, sizeof(path), "%s.layer%d.bin", dump_path, l); \
                dump_activations_to_file(path, &s, l);                     \
            }                                                              \
        }                                                                  \
        cache.current_pos = (pos_) + 1;                                    \
    } while (0)

    /* Re-embed scratch — used to re-feed the same prompt token at each
     * --positions step (skeleton mode without generation). */
    m4t_mtfp_t x_init[BITNET_HIDDEN_SIZE];
    memcpy(x_init, x, sizeof(x_init));

    /* Phase 1: prompt-position forward.
     * If --prompt-tokens is given, embed each prompt token in turn.
     * Otherwise re-feed the same --token at each --positions step. */
    int last_dump_pos = (n_generate > 0) ? -1 : (n_positions - 1);
    for (int pos = 0; pos < n_positions; pos++) {
        if (n_prompt_tokens > 0) {
            int tok = prompt_tokens[pos];
            if (loaded_ok && weights.embedding != NULL) {
                bitnet_embed(x, weights.embedding, weights.embedding_block_exp, tok);
            } else {
                for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) x[i] = (i % 7) - 3;
            }
        } else if (pos > 0) {
            memcpy(x, x_init, sizeof(x_init));
        }
        /* Per-position dump if --prompt-tokens used (each position gets its own
         * dump file). For single-token mode, only the last position dumps. */
        int dump_this = (n_prompt_tokens > 0)
                        ? (dump_path != NULL)
                        : (pos == last_dump_pos);
        /* Inline FORWARD_ONE with a per-position dump suffix. */
        for (int l = 0; l < layers_to_run; l++) {
            const bitnet_layer_weights_t* w_l;
            if (loaded_ok && weights.layers[l].w_q != NULL) w_l = &weights.layers[l];
            else                                            w_l = &w_dummy;
            bitnet_forward_block(x, w_l, &s, &cache, l, pos);
            if (dump_this) {
                char path[1024];
                if (n_prompt_tokens > 0) {
                    snprintf(path, sizeof(path), "%s.pos%d.layer%d.bin", dump_path, pos, l);
                } else {
                    snprintf(path, sizeof(path), "%s.layer%d.bin", dump_path, l);
                }
                dump_activations_to_file(path, &s, l);
            }
        }
        cache.current_pos = pos + 1;
    }

    /* Generated tokens accumulator. */
    int generated_tokens[2048];
    int n_generated = 0;

    /* Phase 2: greedy generation (--gen N). After each forward, take
     * argmax over LM head logits; that token id becomes the next input
     * embedding. Stops at n_generate or when sequence_len fills the cache. */
    struct timespec gen_t0, gen_t1;
    g_attn_seconds = 0.0;  /* reset for this gen loop */
    clock_gettime(CLOCK_MONOTONIC, &gen_t0);
    for (int g = 0; g < n_generate; g++) {
        int pos = n_positions + g;
        if (pos >= max_seq) break;

        /* Apply final norm + argmax to the current x (output of last
         * prompt/gen forward). */
        m4t_mtfp_t x_finalnorm[BITNET_HIDDEN_SIZE];
        if (loaded_ok && weights.gamma_final_norm != NULL) {
            m4t_mtfp_rmsnorm_bx(x_finalnorm, x, weights.gamma_final_norm,
                                BITNET_ACT_BX, weights.gamma_final_norm_block_exp,
                                BITNET_ACT_BX, /*eps=*/1, BITNET_HIDDEN_SIZE);
        } else {
            memcpy(x_finalnorm, x, sizeof(x_finalnorm));
        }
        int next_tok;
        {
            int t1 = -1, t2 = -1;
            int64_t a1 = 0, a2 = 0;
            if (weights.lm_head != NULL &&
                bitnet_top2_full_vocab(x_finalnorm, weights.lm_head,
                                       &t1, &a1, &t2, &a2) == 0) {
                next_tok = t1;
                const char* persp = getenv("BITNET_LOG_PERSTEP");
                if (persp && *persp && *persp != '0') {
                    fprintf(stderr,
                        "[perstep] pos=%d gen=%d top1=%d top1_acc=%lld "
                        "top2=%d top2_acc=%lld margin=%lld\n",
                        pos, g, t1, (long long)a1, t2, (long long)a2,
                        (long long)(a1 - a2));
                }
            } else {
                next_tok = bitnet_argmax_full_vocab(x_finalnorm, weights.lm_head);
            }
        }
        if (next_tok < 0) {
            /* Skeleton mode (no LM head): pick a fixed dummy token. */
            next_tok = (token_id + g + 1) % BITNET_VOCAB_SIZE;
        }
        if (n_generated < (int)(sizeof(generated_tokens)/sizeof(int))) {
            generated_tokens[n_generated++] = next_tok;
        }

        /* Embed the new token; forward at pos. */
        if (loaded_ok && weights.embedding != NULL) {
            bitnet_embed(x, weights.embedding, weights.embedding_block_exp, next_tok);
        } else {
            for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) x[i] = (i % 7) - 3;
        }
        int dump_this = (g == n_generate - 1) && dump_path != NULL;
        FORWARD_ONE(pos, dump_this);
    }
    clock_gettime(CLOCK_MONOTONIC, &gen_t1);
    double gen_seconds = (double)(gen_t1.tv_sec - gen_t0.tv_sec)
                       + (double)(gen_t1.tv_nsec - gen_t0.tv_nsec) / 1e9;
    if (n_generate > 0) {
        double attn_per_token = g_attn_seconds / n_generate;
        double attn_frac = (gen_seconds > 0) ? (g_attn_seconds / gen_seconds) : 0.0;
        fprintf(stderr,
                "[harness] gen_loop_seconds = %.4f  n_generate = %d  seconds_per_token = %.4f  "
                "prompt_tokens = %d  attn_seconds = %.4f  attn_seconds_per_token = %.4f  "
                "attn_fraction = %.3f\n",
                gen_seconds, n_generate, gen_seconds / n_generate,
                n_positions, g_attn_seconds, attn_per_token, attn_frac);
    }

    #undef FORWARD_ONE

    /* Final norm + LM head on the last x (post all forwards). */
    if (loaded_ok && weights.gamma_final_norm != NULL) {
        m4t_mtfp_t x_norm[BITNET_HIDDEN_SIZE];
        m4t_mtfp_rmsnorm_bx(x_norm, x, weights.gamma_final_norm,
                            BITNET_ACT_BX, weights.gamma_final_norm_block_exp,
                            BITNET_ACT_BX, /*eps=*/1, BITNET_HIDDEN_SIZE);
        memcpy(x, x_norm, sizeof(x));
    }

    int top_n = 16;
    m4t_mtfp_t logits[16];
    if (loaded_ok && weights.lm_head != NULL) {
        bitnet_lm_head(logits, x, weights.lm_head, top_n);
        /* Also compute argmax over full vocab + dump raw int64 logits
         * (red-team #6: cross-check substrate's predicted next-token
         * against HF reference). */
        int argmax_tok = bitnet_argmax_full_vocab(x, weights.lm_head);
        fprintf(stderr, "     argmax over full vocab        = %d\n", argmax_tok);
        if (dump_path) {
            char logits_path[1024];
            snprintf(logits_path, sizeof(logits_path), "%s.logits.bin", dump_path);
            FILE* lf = fopen(logits_path, "wb");
            if (lf) {
                int32_t vocab = BITNET_VOCAB_SIZE;
                fwrite(&vocab, sizeof(int32_t), 1, lf);
                int64_t* full_logits = (int64_t*)malloc((size_t)vocab * sizeof(int64_t));
                for (int v = 0; v < vocab; v++) {
                    const m4t_mtfp_t* row = weights.lm_head + (size_t)v * BITNET_HIDDEN_SIZE;
                    int64_t acc = 0;
                    for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) {
                        acc += (int64_t)x[i] * (int64_t)row[i];
                    }
                    full_logits[v] = acc;
                }
                fwrite(full_logits, sizeof(int64_t), (size_t)vocab, lf);
                free(full_logits);
                fclose(lf);
                fprintf(stderr, "     dumped full logits → %s\n", logits_path);
            }
        }
    } else {
        memset(logits, 0, sizeof(logits));
    }

    fprintf(stderr,
        "[ok] %d layer(s) forward pass completed (token_id=%d, "
            "positions=%d, generated=%d).\n"
        "     post-final-norm x[0..3]      = %d %d %d %d\n"
        "     logits[0..3]                  = %d %d %d %d\n",
        layers_to_run, token_id, n_positions, n_generated,
        x[0], x[1], x[2], x[3],
        logits[0], logits[1], logits[2], logits[3]);
    if (n_generated > 0) {
        fprintf(stderr, "     generated tokens             =");
        for (int i = 0; i < n_generated; i++) {
            fprintf(stderr, " %d", generated_tokens[i]);
        }
        fprintf(stderr, "\n");
    }

    bitnet_block_scratch_free(&s);
    bitnet_kv_cache_free(&cache);
    if (loaded_ok) bitnet_weights_unload(&handle);
    return 0;
}
