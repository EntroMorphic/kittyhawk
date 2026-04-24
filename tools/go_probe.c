/*
 * go_probe.c — base3 benchmark / substrate-distance probe on 19×19 Go positions.
 *
 * Cycle 1 (base3_benchmarks → base3_go_probe): raw Hamming on raw trits
 *   classified phase at 40.40%; density-only baseline at 98.28%. RED.
 *
 * Cycle 2 (substrate_distance_refinement): this file extends the probe
 *   with two orthogonal fixes and one density-controlled task, so we can
 *   say whether a substrate-level distance fix can close the gap.
 *
 * Axes:
 *   --encoding {raw, contrast3}
 *       raw       : own=+1, empty=0, opp=-1 per cell (original probe).
 *       contrast3 : per cell, sign of 3×3-neighborhood (own − opp) sum.
 *                   Same 361 shape, local-structure content.
 *
 *   --metric   {hamming, hamming_norm}
 *       hamming      : count of trit mismatches.
 *       hamming_norm : hamming · C / (density(a) + density(b) + eps).
 *                      Compensates for the sparse-vector attractor that
 *                      caused the original probe's opening-cluster bias.
 *
 *   --task     {phase, same_game}
 *       phase     : 3-class k-NN classification by move-number bin.
 *       same_game : k-NN retrieval; report fraction of top-k drawn from
 *                   the query's own source game. Density-controlled since
 *                   adjacent positions have similar density.
 *
 * Defaults: --encoding raw --metric hamming --task phase, which reproduces
 * the cycle-1 baseline numbers exactly.
 */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#  include <arm_neon.h>
#endif

#define BOARD   19
#define BSQ     (BOARD * BOARD)  /* 361 */
#define MAX_MOVES 720

static uint32_t rng_state = 0xdeadbeefu;
static uint32_t xor32(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x; return x;
}

typedef struct {
    int8_t cells[BSQ];
} board_t;

typedef struct {
    int stack[BSQ];
    int top;
    int8_t visited[BSQ];
} flood_t;

static void flood_reset(flood_t* f) {
    f->top = 0;
    memset(f->visited, 0, sizeof(f->visited));
}

static int group_has_liberty(const board_t* b, int idx, int* group, int* n_group, flood_t* f) {
    int color = b->cells[idx];
    if (color == 0) { if (n_group) *n_group = 0; return 1; }
    flood_reset(f);
    f->stack[f->top++] = idx;
    f->visited[idx] = 1;
    int n = 0;
    int has_lib = 0;
    while (f->top > 0) {
        int p = f->stack[--f->top];
        if (group) group[n] = p;
        n++;
        int r = p / BOARD, c = p % BOARD;
        int d[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
        for (int k = 0; k < 4; k++) {
            int nr = r + d[k][0], nc = c + d[k][1];
            if (nr < 0 || nr >= BOARD || nc < 0 || nc >= BOARD) continue;
            int q = nr * BOARD + nc;
            if (f->visited[q]) continue;
            if (b->cells[q] == 0) { has_lib = 1; f->visited[q] = 1; continue; }
            if (b->cells[q] == color) {
                f->visited[q] = 1;
                f->stack[f->top++] = q;
            }
        }
    }
    if (n_group) *n_group = n;
    return has_lib;
}

static int play_move(board_t* b, int idx, int color, flood_t* f) {
    if (idx < 0) return 0;
    if (idx >= BSQ) return -1;
    if (b->cells[idx] != 0) return -1;
    b->cells[idx] = (int8_t)color;
    int r = idx / BOARD, c = idx % BOARD;
    int d[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
    int group[BSQ];
    int n_group;
    for (int k = 0; k < 4; k++) {
        int nr = r + d[k][0], nc = c + d[k][1];
        if (nr < 0 || nr >= BOARD || nc < 0 || nc >= BOARD) continue;
        int q = nr * BOARD + nc;
        if (b->cells[q] == -color) {
            if (!group_has_liberty(b, q, group, &n_group, f)) {
                for (int i = 0; i < n_group; i++) b->cells[group[i]] = 0;
            }
        }
    }
    if (!group_has_liberty(b, idx, NULL, NULL, f)) {
        b->cells[idx] = 0;
        return -1;
    }
    return 0;
}

static int parse_sgf(const char* path, int* out_sz, int* out_handicap,
                     int* moves_idx, int* moves_col, int max)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0 || sz > (long)(2 << 20)) { fclose(fp); return -1; }
    char* buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return -1; }
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) { free(buf); fclose(fp); return -1; }
    buf[sz] = 0;
    fclose(fp);

    *out_sz = 19;
    *out_handicap = 0;
    int n = 0;
    int depth = 0;
    int main_line_only = 1;
    int main_depth = 0;

    for (long i = 0; i < sz && n < max; i++) {
        char ch = buf[i];
        if (ch == '(') {
            depth++;
            if (depth == 1) main_depth = 1;
            else if (main_line_only && depth > main_depth) {
                int skip_depth = 1;
                long j = i + 1;
                while (j < sz && skip_depth > 0) {
                    if (buf[j] == '[') {
                        j++;
                        while (j < sz && buf[j] != ']') {
                            if (buf[j] == '\\' && j + 1 < sz) j++;
                            j++;
                        }
                    } else if (buf[j] == '(') skip_depth++;
                    else if (buf[j] == ')') skip_depth--;
                    j++;
                }
                i = j - 1;
                depth--;
                continue;
            }
        } else if (ch == ')') {
            depth--;
        } else if (ch == 'S' && i + 3 < sz && buf[i+1] == 'Z' && buf[i+2] == '[') {
            int v = 0;
            long j = i + 3;
            while (j < sz && isdigit((unsigned char)buf[j])) { v = v * 10 + (buf[j] - '0'); j++; }
            *out_sz = v;
            i = j;
        } else if (ch == 'H' && i + 3 < sz && buf[i+1] == 'A' && buf[i+2] == '[') {
            int v = 0;
            long j = i + 3;
            while (j < sz && isdigit((unsigned char)buf[j])) { v = v * 10 + (buf[j] - '0'); j++; }
            *out_handicap = v;
            i = j;
        } else if ((ch == 'B' || ch == 'W') && i + 1 < sz && buf[i+1] == '[') {
            long back = i - 1;
            while (back >= 0 && isspace((unsigned char)buf[back])) back--;
            if (back < 0 || buf[back] != ';') continue;
            long j = i + 2;
            int idx = -1;
            if (j < sz && buf[j] != ']') {
                if (j + 1 < sz && buf[j] >= 'a' && buf[j] <= 's' &&
                                  buf[j+1] >= 'a' && buf[j+1] <= 's')
                {
                    int c = buf[j] - 'a';
                    int r = buf[j+1] - 'a';
                    idx = r * BOARD + c;
                } else {
                    idx = -1;
                }
            }
            moves_idx[n] = idx;
            moves_col[n] = (ch == 'B') ? +1 : -1;
            n++;
            while (j < sz && buf[j] != ']') j++;
            i = j;
        }
    }
    free(buf);
    return n;
}

/* Encoding selector. */
typedef enum { ENC_RAW = 0, ENC_CONTRAST3 = 1 } encoding_t;

/* encode_raw: own=+1, empty=0, opp=-1 from `mover`'s perspective. */
static void encode_raw(const board_t* b, int mover, int8_t* out) {
    for (int i = 0; i < BSQ; i++) {
        int v = b->cells[i];
        if (v == 0) out[i] = 0;
        else if (v == mover) out[i] = +1;
        else out[i] = -1;
    }
}

/* encode_contrast3: for each cell, sign of (own − opp) summed over its
 * 3×3 neighborhood (clamped to board). Output trits encode local balance,
 * not raw stone presence. */
static void encode_contrast3(const board_t* b, int mover, int8_t* out) {
    for (int r = 0; r < BOARD; r++) {
        for (int c = 0; c < BOARD; c++) {
            int s = 0;
            for (int dr = -1; dr <= 1; dr++) {
                int nr = r + dr;
                if (nr < 0 || nr >= BOARD) continue;
                for (int dc = -1; dc <= 1; dc++) {
                    int nc = c + dc;
                    if (nc < 0 || nc >= BOARD) continue;
                    int v = b->cells[nr * BOARD + nc];
                    if (v == 0) continue;
                    if (v == mover) s++; else s--;
                }
            }
            int8_t trit = (s > 0) ? +1 : (s < 0) ? -1 : 0;
            out[r * BOARD + c] = trit;
        }
    }
}

/* Dataset with phase, move_num, and game_id (source SGF index). */
typedef struct {
    int8_t* trits;
    int8_t* phase;
    int*    move_num;
    int*    game_id;
    int     n;
    int     cap;
} dataset_t;

static void ds_init(dataset_t* d) { memset(d, 0, sizeof(*d)); }
static void ds_free(dataset_t* d) {
    free(d->trits); free(d->phase); free(d->move_num); free(d->game_id);
    ds_init(d);
}
static void ds_reserve(dataset_t* d, int extra) {
    if (d->n + extra <= d->cap) return;
    int newcap = d->cap ? d->cap * 2 : 16;
    while (newcap < d->n + extra) newcap *= 2;
    d->trits = realloc(d->trits, (size_t)newcap * BSQ);
    d->phase = realloc(d->phase, (size_t)newcap);
    d->move_num = realloc(d->move_num, (size_t)newcap * sizeof(int));
    d->game_id = realloc(d->game_id, (size_t)newcap * sizeof(int));
    d->cap = newcap;
}

static void emit_position(dataset_t* d, const board_t* b, int mover,
                          int move_num, int game_id, encoding_t enc)
{
    ds_reserve(d, 1);
    int8_t* out = d->trits + (size_t)d->n * BSQ;
    if (enc == ENC_CONTRAST3) encode_contrast3(b, mover, out);
    else                       encode_raw(b, mover, out);
    int p;
    if (move_num < 60) p = 0;
    else if (move_num < 150) p = 1;
    else p = 2;
    d->phase[d->n] = (int8_t)p;
    d->move_num[d->n] = move_num;
    d->game_id[d->n] = game_id;
    d->n++;
}

static void ds_shuffle(dataset_t* d) {
    for (int i = d->n - 1; i > 0; i--) {
        int j = (int)(xor32() % (uint32_t)(i + 1));
        if (j == i) continue;
        int8_t tmp_row[BSQ];
        memcpy(tmp_row, d->trits + (size_t)i * BSQ, BSQ);
        memcpy(d->trits + (size_t)i * BSQ, d->trits + (size_t)j * BSQ, BSQ);
        memcpy(d->trits + (size_t)j * BSQ, tmp_row, BSQ);
        int8_t tp = d->phase[i]; d->phase[i] = d->phase[j]; d->phase[j] = tp;
        int tm = d->move_num[i]; d->move_num[i] = d->move_num[j]; d->move_num[j] = tm;
        int tg = d->game_id[i]; d->game_id[i] = d->game_id[j]; d->game_id[j] = tg;
    }
}

/* Hamming distance on int8 trit vectors. */
static int hamming_trit(const int8_t* a, const int8_t* b, int n) {
    int d = 0;
    int k = 0;
#if defined(__ARM_NEON)
    int32x4_t vacc = vdupq_n_s32(0);
    for (; k + 16 <= n; k += 16) {
        int8x16_t va = vld1q_s8(a + k);
        int8x16_t vb = vld1q_s8(b + k);
        uint8x16_t neq = vmvnq_u8(vceqq_s8(va, vb));
        int8x16_t ones = vandq_s8(vreinterpretq_s8_u8(neq), vdupq_n_s8(1));
        int16x8_t lo = vpaddlq_s8(ones);
        vacc = vpadalq_s16(vacc, lo);
    }
    d = vaddvq_s32(vacc);
#endif
    for (; k < n; k++) if (a[k] != b[k]) d++;
    return d;
}

static int trit_density(const int8_t* v, int n) {
    int d = 0;
    for (int i = 0; i < n; i++) if (v[i] != 0) d++;
    return d;
}

/* Scoring modes for metric. Return an int where smaller = closer. */
typedef enum { MET_HAMMING = 0, MET_HAMMING_NORM = 1 } metric_t;

static int score_pair(const int8_t* a, int dens_a,
                      const int8_t* b, int dens_b,
                      metric_t metric)
{
    int h = hamming_trit(a, b, BSQ);
    if (metric == MET_HAMMING) return h;
    /* hamming_norm: h * 1024 / (dens_a + dens_b + 1). */
    int denom = dens_a + dens_b + 1;
    return (h * 1024) / denom;
}

typedef struct { int d; int cls; int src_game; } neighbor_t;

static int cmp_neigh(const void* a, const void* b) {
    return ((const neighbor_t*)a)->d - ((const neighbor_t*)b)->d;
}

static double knn_phase_accuracy(
    const int8_t* X_tr, const int8_t* Y_tr, const int* dens_tr, int n_tr,
    const int8_t* X_te, const int8_t* Y_te, const int* dens_te, int n_te,
    int k, metric_t metric)
{
    neighbor_t* neigh = malloc((size_t)n_tr * sizeof(neighbor_t));
    int* conf = calloc(9, sizeof(int));
    int correct = 0;
    for (int t = 0; t < n_te; t++) {
        const int8_t* q = X_te + (size_t)t * BSQ;
        int dq = dens_te[t];
        for (int i = 0; i < n_tr; i++) {
            neigh[i].d = score_pair(q, dq, X_tr + (size_t)i * BSQ, dens_tr[i], metric);
            neigh[i].cls = (int)Y_tr[i];
        }
        qsort(neigh, (size_t)n_tr, sizeof(neighbor_t), cmp_neigh);
        int votes[3] = {0, 0, 0};
        int kk = k < n_tr ? k : n_tr;
        for (int i = 0; i < kk; i++) votes[neigh[i].cls]++;
        int pred = 0;
        if (votes[1] > votes[pred]) pred = 1;
        if (votes[2] > votes[pred]) pred = 2;
        int truth = (int)Y_te[t];
        if (pred == truth) correct++;
        conf[truth * 3 + pred]++;
    }
    printf("  confusion (rows=truth, cols=pred):\n");
    const char* names[3] = {"open", "mid ", "end "};
    printf("         open   mid   end\n");
    for (int r = 0; r < 3; r++) {
        printf("   %s  ", names[r]);
        for (int c = 0; c < 3; c++) printf(" %5d", conf[r * 3 + c]);
        printf("\n");
    }
    double acc = (double)correct / (double)n_te;
    free(neigh); free(conf);
    return acc;
}

/* Same-game retrieval: for each test position q, find top-k nearest in
 * train, count how many come from the same source game as q. Report mean
 * and compare to the random baseline (k / n_tr). */
static void knn_same_game_retrieval(
    const int8_t* X_tr, const int* game_tr, const int* dens_tr, int n_tr,
    const int8_t* X_te, const int* game_te, const int* dens_te, int n_te,
    int k, metric_t metric)
{
    neighbor_t* neigh = malloc((size_t)n_tr * sizeof(neighbor_t));
    double total_frac = 0.0;
    double total_random_frac = 0.0;  /* per-query g_q / n_train averaged */
    int queries_with_same_game_train = 0;
    for (int t = 0; t < n_te; t++) {
        const int8_t* q = X_te + (size_t)t * BSQ;
        int dq = dens_te[t];
        int gq = game_te[t];
        int g_companions = 0;
        for (int i = 0; i < n_tr; i++) {
            neigh[i].d = score_pair(q, dq, X_tr + (size_t)i * BSQ, dens_tr[i], metric);
            neigh[i].src_game = game_tr[i];
            if (game_tr[i] == gq) g_companions++;
        }
        if (g_companions == 0) continue;  /* query's game has no train companions */
        queries_with_same_game_train++;
        qsort(neigh, (size_t)n_tr, sizeof(neighbor_t), cmp_neigh);
        int kk = k < n_tr ? k : n_tr;
        int hits = 0;
        for (int i = 0; i < kk; i++) if (neigh[i].src_game == gq) hits++;
        total_frac += (double)hits / (double)kk;
        /* Per-query random baseline: random kNN draw expected same-game
         * fraction = g_companions / n_tr. */
        total_random_frac += (double)g_companions / (double)n_tr;
    }
    double mean_frac = queries_with_same_game_train
                        ? total_frac / queries_with_same_game_train : 0.0;
    double mean_random = queries_with_same_game_train
                        ? total_random_frac / queries_with_same_game_train : 0.0;
    double lift = (mean_random > 0) ? mean_frac / mean_random : 0.0;
    printf("  mean same-game hit rate = %.4f%%  (random = %.4f%%, lift = %.1f×)\n",
           100.0 * mean_frac, 100.0 * mean_random, lift);
    printf("  queries with ≥1 same-game train companion: %d / %d\n",
           queries_with_same_game_train, n_te);
    free(neigh);
}

static int list_sgf(const char* dir, char*** out) {
    DIR* d = opendir(dir);
    if (!d) { fprintf(stderr, "opendir(%s): %s\n", dir, strerror(errno)); return -1; }
    int cap = 128, n = 0;
    char** arr = malloc((size_t)cap * sizeof(char*));
    struct dirent* e;
    while ((e = readdir(d)) != NULL) {
        size_t L = strlen(e->d_name);
        if (L < 4 || strcmp(e->d_name + L - 4, ".sgf") != 0) continue;
        if (n == cap) { cap *= 2; arr = realloc(arr, (size_t)cap * sizeof(char*)); }
        size_t plen = strlen(dir) + 1 + L + 1;
        arr[n] = malloc(plen);
        snprintf(arr[n], plen, "%s/%s", dir, e->d_name);
        n++;
    }
    closedir(d);
    *out = arr;
    return n;
}

typedef enum { TASK_PHASE = 0, TASK_SAME_GAME = 1 } task_t;
typedef enum { SPLIT_POSITION = 0, SPLIT_GAME = 1 } split_t;

/* Game-wise train/test split: all positions from a given game go to the
 * same side. Eliminates within-game leakage (adjacent-move positions
 * sharing >95% of trits). */
static void ds_split_by_game(dataset_t* d, int* out_n_tr, int* out_n_te,
                             int test_frac_pct)
{
    /* Count distinct games. */
    int n_games = 0;
    for (int i = 0; i < d->n; i++) if (d->game_id[i] > n_games) n_games = d->game_id[i];
    n_games++;  /* IDs are 0-indexed */

    /* Shuffle game IDs, pick last test_frac_pct as test games. */
    int* order = malloc((size_t)n_games * sizeof(int));
    for (int i = 0; i < n_games; i++) order[i] = i;
    for (int i = n_games - 1; i > 0; i--) {
        int j = (int)(xor32() % (uint32_t)(i + 1));
        int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }
    int n_test_games = (n_games * test_frac_pct) / 100;
    int* is_test = calloc((size_t)n_games, sizeof(int));
    for (int i = n_games - n_test_games; i < n_games; i++) is_test[order[i]] = 1;
    free(order);

    /* Stable partition: train positions first, then test positions. Swap
     * to keep memory layout contiguous for brute-force k-NN. */
    int write_tr = 0;
    int write_te = d->n - 1;
    /* Use a simple two-pointer partition. O(n) swaps. */
    int i = 0;
    while (i <= write_te) {
        int gid = d->game_id[i];
        if (!is_test[gid]) {
            /* stays in train */
            if (i != write_tr) {
                int8_t row_tmp[BSQ];
                memcpy(row_tmp, d->trits + (size_t)i * BSQ, BSQ);
                memcpy(d->trits + (size_t)i * BSQ, d->trits + (size_t)write_tr * BSQ, BSQ);
                memcpy(d->trits + (size_t)write_tr * BSQ, row_tmp, BSQ);
                int8_t tp = d->phase[i]; d->phase[i] = d->phase[write_tr]; d->phase[write_tr] = tp;
                int tm = d->move_num[i]; d->move_num[i] = d->move_num[write_tr]; d->move_num[write_tr] = tm;
                int tg = d->game_id[i]; d->game_id[i] = d->game_id[write_tr]; d->game_id[write_tr] = tg;
            }
            write_tr++;
            i++;
        } else {
            /* goes to test — swap to the end */
            if (i != write_te) {
                int8_t row_tmp[BSQ];
                memcpy(row_tmp, d->trits + (size_t)i * BSQ, BSQ);
                memcpy(d->trits + (size_t)i * BSQ, d->trits + (size_t)write_te * BSQ, BSQ);
                memcpy(d->trits + (size_t)write_te * BSQ, row_tmp, BSQ);
                int8_t tp = d->phase[i]; d->phase[i] = d->phase[write_te]; d->phase[write_te] = tp;
                int tm = d->move_num[i]; d->move_num[i] = d->move_num[write_te]; d->move_num[write_te] = tm;
                int tg = d->game_id[i]; d->game_id[i] = d->game_id[write_te]; d->game_id[write_te] = tg;
            }
            write_te--;
        }
    }
    free(is_test);
    *out_n_tr = write_tr;
    *out_n_te = d->n - write_tr;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
          "usage: %s <sgf_dir> [--max_games N] [--sample_every K]\n"
          "                    [--encoding {raw,contrast3}]\n"
          "                    [--metric   {hamming,hamming_norm}]\n"
          "                    [--task     {phase,same_game}]\n", argv[0]);
        return 1;
    }
    const char* dir = argv[1];
    int max_games = 2000;
    int sample_every = 5;
    encoding_t enc = ENC_RAW;
    metric_t   metric = MET_HAMMING;
    task_t     task = TASK_PHASE;
    split_t    split = SPLIT_POSITION;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--max_games") && i + 1 < argc) max_games = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sample_every") && i + 1 < argc) sample_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--encoding") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "raw")) enc = ENC_RAW;
            else if (!strcmp(argv[i], "contrast3")) enc = ENC_CONTRAST3;
            else { fprintf(stderr, "unknown encoding: %s\n", argv[i]); return 1; }
        }
        else if (!strcmp(argv[i], "--metric") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "hamming")) metric = MET_HAMMING;
            else if (!strcmp(argv[i], "hamming_norm")) metric = MET_HAMMING_NORM;
            else { fprintf(stderr, "unknown metric: %s\n", argv[i]); return 1; }
        }
        else if (!strcmp(argv[i], "--task") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "phase")) task = TASK_PHASE;
            else if (!strcmp(argv[i], "same_game")) task = TASK_SAME_GAME;
            else { fprintf(stderr, "unknown task: %s\n", argv[i]); return 1; }
        }
        else if (!strcmp(argv[i], "--split") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "position")) split = SPLIT_POSITION;
            else if (!strcmp(argv[i], "game")) split = SPLIT_GAME;
            else { fprintf(stderr, "unknown split: %s\n", argv[i]); return 1; }
        }
    }
    const char* enc_name = (enc == ENC_CONTRAST3) ? "contrast3" : "raw";
    const char* met_name = (metric == MET_HAMMING_NORM) ? "hamming_norm" : "hamming";
    const char* task_name = (task == TASK_SAME_GAME) ? "same_game" : "phase";
    const char* split_name = (split == SPLIT_GAME) ? "game" : "position";
    printf("== go_probe encoding=%s metric=%s task=%s split=%s ==\n",
           enc_name, met_name, task_name, split_name);

    char** files;
    int nf = list_sgf(dir, &files);
    if (nf < 0) return 1;
    printf("scanned %d sgf files in %s\n", nf, dir);
    if (nf > max_games) nf = max_games;

    dataset_t ds;
    ds_init(&ds);

    int* moves_idx = malloc(sizeof(int) * MAX_MOVES);
    int* moves_col = malloc(sizeof(int) * MAX_MOVES);
    flood_t f;

    int n_games_ok = 0, n_games_handicap = 0, n_games_wrong_size = 0;
    int n_games_illegal_move = 0;
    int phase_counts[3] = {0, 0, 0};
    clock_t t0 = clock();

    int game_id_counter = 0;
    for (int g = 0; g < nf; g++) {
        int sz, ha;
        int nm = parse_sgf(files[g], &sz, &ha, moves_idx, moves_col, MAX_MOVES);
        if (nm <= 10) { free(files[g]); continue; }
        if (sz != 19) { n_games_wrong_size++; free(files[g]); continue; }
        if (ha > 1)   { n_games_handicap++;   free(files[g]); continue; }

        board_t b;
        memset(b.cells, 0, sizeof(b.cells));
        int illegal_here = 0;
        int played = 0;
        int this_game_id = game_id_counter++;
        for (int m = 0; m < nm; m++) {
            if (play_move(&b, moves_idx[m], moves_col[m], &f) < 0) { illegal_here = 1; break; }
            played++;
            int mover_next = -moves_col[m];
            if ((played % sample_every) == 0) {
                emit_position(&ds, &b, mover_next, played, this_game_id, enc);
                phase_counts[ds.phase[ds.n - 1]]++;
            }
        }
        free(files[g]);
        if (illegal_here) { n_games_illegal_move++; continue; }
        n_games_ok++;
    }
    free(files); free(moves_idx); free(moves_col);

    double t_parse = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("games ok=%d handicap_skip=%d wrong_size=%d illegal_move=%d\n",
           n_games_ok, n_games_handicap, n_games_wrong_size, n_games_illegal_move);
    printf("positions=%d  phase counts: open=%d mid=%d end=%d  (parse+encode %.2fs)\n",
           ds.n, phase_counts[0], phase_counts[1], phase_counts[2], t_parse);

    if (ds.n < 500) {
        fprintf(stderr, "too few positions for a meaningful probe\n");
        ds_free(&ds);
        return 1;
    }

    /* Shuffle first (breaks intra-game storage adjacency so position-split
     * gets a random 80/20 mix). Game-wise split then partitions BY game_id,
     * ignoring position order. */
    ds_shuffle(&ds);
    int n_tr, n_te;
    if (split == SPLIT_GAME) {
        ds_split_by_game(&ds, &n_tr, &n_te, 20);
        printf("game-wise split: train=%d positions  test=%d positions  (no game leaks across)\n",
               n_tr, n_te);
    } else {
        n_te = ds.n / 5;
        n_tr = ds.n - n_te;
    }
    const int8_t* X_tr = ds.trits;
    const int8_t* Y_tr = ds.phase;
    const int*    G_tr = ds.game_id;
    const int8_t* X_te = ds.trits + (size_t)n_tr * BSQ;
    const int8_t* Y_te = ds.phase + n_tr;
    const int*    G_te = ds.game_id + n_tr;

    /* Precompute density for current encoding. */
    int* dens_tr = malloc((size_t)n_tr * sizeof(int));
    int* dens_te = malloc((size_t)n_te * sizeof(int));
    for (int i = 0; i < n_tr; i++) dens_tr[i] = trit_density(X_tr + (size_t)i * BSQ, BSQ);
    for (int i = 0; i < n_te; i++) dens_te[i] = trit_density(X_te + (size_t)i * BSQ, BSQ);

    if (task == TASK_PHASE) {
        /* Majority-class baseline. */
        int tc[3] = {0, 0, 0};
        for (int i = 0; i < n_tr; i++) tc[Y_tr[i]]++;
        int maj = 0;
        if (tc[1] > tc[maj]) maj = 1;
        if (tc[2] > tc[maj]) maj = 2;
        int correct_maj = 0;
        for (int i = 0; i < n_te; i++) if ((int)Y_te[i] == maj) correct_maj++;
        printf("majority-class baseline (cls=%d): %.2f%%\n",
               maj, 100.0 * correct_maj / n_te);

        /* Density-only baseline (k=200): yardstick, not competitor. */
        int correct_dens = 0;
        int cap_te = n_te < 500 ? n_te : 500;
        for (int t = 0; t < cap_te; t++) {
            int q = dens_te[t];
            neighbor_t* dn = malloc((size_t)n_tr * sizeof(neighbor_t));
            for (int i = 0; i < n_tr; i++) {
                int diff = dens_tr[i] - q;
                dn[i].d = diff < 0 ? -diff : diff;
                dn[i].cls = (int)Y_tr[i];
            }
            qsort(dn, (size_t)n_tr, sizeof(neighbor_t), cmp_neigh);
            int votes[3] = {0, 0, 0};
            for (int i = 0; i < 200 && i < n_tr; i++) votes[dn[i].cls]++;
            int pred = 0;
            if (votes[1] > votes[pred]) pred = 1;
            if (votes[2] > votes[pred]) pred = 2;
            if (pred == (int)Y_te[t]) correct_dens++;
            free(dn);
        }
        printf("density-only k=200 baseline: %.2f%%  (on first %d test positions)\n",
               100.0 * correct_dens / cap_te, cap_te);
    }

    int cap_te = n_te < 500 ? n_te : 500;
    printf("split: train=%d test=%d (capped at %d for brute-force k-NN)  k ∈ {50, 100, 200}\n",
           n_tr, n_te, cap_te);

    int ks[] = {50, 100, 200};
    for (int ki = 0; ki < 3; ki++) {
        int k = ks[ki];
        clock_t kt0 = clock();
        if (task == TASK_PHASE) {
            double acc = knn_phase_accuracy(X_tr, Y_tr, dens_tr, n_tr,
                                             X_te, Y_te, dens_te, cap_te,
                                             k, metric);
            double dt = (double)(clock() - kt0) / CLOCKS_PER_SEC;
            printf("k=%3d  phase-ID acc = %.2f%%  (%.2fs)\n", k, 100.0 * acc, dt);
        } else {
            knn_same_game_retrieval(X_tr, G_tr, dens_tr, n_tr,
                                     X_te, G_te, dens_te, cap_te,
                                     k, metric);
            double dt = (double)(clock() - kt0) / CLOCKS_PER_SEC;
            printf("k=%3d  same-game retrieval  (%.2fs)\n", k, dt);
        }
    }

    free(dens_tr); free(dens_te);
    ds_free(&ds);
    return 0;
}
