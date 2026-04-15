/*
 * STATUS: research scaffolding, not production architecture.
 * Runs routing primitives inside an O(N_train) dense outer loop.
 * Pre-Axis-5 headline tool; produced 97.79% / 97.86% / 97.99% across
 * the N=2048/4096 configurations under "routed kernels vs dense
 * kernels inside the same dense application shape" — a compression
 * win, not a routing-architecture win.
 * For production routed k-NN use tools/mnist_routed_bucket{,_multi}.c
 * on libglyph. See docs/FINDINGS.md Axis 5.
 *
 * mnist_routed_knn.c — Trit Lattice LSH with k-NN; fair comparison edition.
 *
 * Four classifier tracks, all k-NN:
 *   1. Deskewed-pixel L1           (dense baseline, journal historic best)
 *   2. Raw-projection L1           (NEON-vectorized; controls for the
 *                                   classifier while holding features
 *                                   equal to the routed path)
 *   3. Deskewed-projection routed  (routed path with deskewing)
 *   4. Raw-projection routed       (routed path without deskewing)
 *
 * Multi-seed sweep for statistical significance: each projection-dependent
 * track runs with several RNG seeds and reports mean ± stddev.
 *
 * Per-side tau_q calibrated from the empirical projection distribution →
 * symmetric §18-passing deployment. Full three-way trit distribution
 * reported (%+1 / %0 / %-1) for signatures, not just zero-density.
 *
 * Usage: ./mnist_routed_knn <mnist_dir>
 */

#include "m4t_types.h"
#include "m4t_mtfp.h"
#include "m4t_trit_pack.h"
#include "m4t_ternary_matmul.h"
#include "m4t_route.h"
#include "m4t_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define INPUT_DIM 784
#define IMG_W 28
#define IMG_H 28
#define N_CLASSES 10
#define MAX_K 7

/* ── Deskewing (ported from archive/tools/mnist_knn_lattice.c) ──────────
 *
 * Integer image-moment deskew. Computes per-row horizontal shift that
 * aligns the image along its principal inertial axis, applies shift
 * per row. No float; all arithmetic in int64 over MTFP mantissas. */
static void deskew_image(m4t_mtfp_t* dst, const m4t_mtfp_t* src) {
    int64_t sum_p=0,sum_xp=0,sum_yp=0;
    for(int y=0;y<IMG_H;y++)
        for(int x=0;x<IMG_W;x++){
            int64_t p=(int64_t)src[y*IMG_W+x];
            sum_p+=p; sum_xp+=(int64_t)x*p; sum_yp+=(int64_t)y*p;
        }
    if(sum_p==0){memcpy(dst,src,INPUT_DIM*sizeof(m4t_mtfp_t));return;}

    int64_t Mxy=0,Myy=0;
    for(int y=0;y<IMG_H;y++){
        int64_t dy=(int64_t)y*sum_p-sum_yp;
        for(int x=0;x<IMG_W;x++){
            int64_t p=(int64_t)src[y*IMG_W+x];
            int64_t dx=(int64_t)x*sum_p-sum_xp;
            Mxy+=dx*dy/sum_p*p/sum_p;
            Myy+=dy*dy/sum_p*p/sum_p;
        }
    }

    memset(dst,0,INPUT_DIM*sizeof(m4t_mtfp_t));
    for(int y=0;y<IMG_H;y++){
        int32_t shift=0;
        if(Myy!=0){
            int64_t dy=(int64_t)y*sum_p-sum_yp;
            shift=(int32_t)(-(dy*Mxy)/(Myy*sum_p));
        }
        for(int x=0;x<IMG_W;x++){
            int nx=x+shift;
            if(nx>=0&&nx<IMG_W) dst[y*IMG_W+nx]=src[y*IMG_W+x];
        }
    }
}

static void deskew_all(m4t_mtfp_t* images, int n) {
    m4t_mtfp_t buf[INPUT_DIM];
    for(int i=0;i<n;i++){
        deskew_image(buf, images+(size_t)i*INPUT_DIM);
        memcpy(images+(size_t)i*INPUT_DIM, buf, INPUT_DIM*sizeof(m4t_mtfp_t));
    }
}

/* ── Data loaders ──────────────────────────────────────────────────────── */

static uint32_t read_u32_be(FILE* f) {
    uint8_t b[4]; fread(b,1,4,f);
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];
}
static m4t_mtfp_t* load_images_mtfp(const char* path, int* n) {
    FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",path);exit(1);}
    read_u32_be(f); *n=(int)read_u32_be(f);
    int rows=(int)read_u32_be(f),cols=(int)read_u32_be(f);
    int dim=rows*cols; size_t total=(size_t)(*n)*dim;
    uint8_t* raw=malloc(total); fread(raw,1,total,f); fclose(f);
    m4t_mtfp_t* data=malloc(total*sizeof(m4t_mtfp_t));
    for(size_t i=0;i<total;i++)
        data[i]=(m4t_mtfp_t)(((int32_t)raw[i]*M4T_MTFP_SCALE+127)/255);
    free(raw); return data;
}
static int* load_labels(const char* path, int* n) {
    FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",path);exit(1);}
    read_u32_be(f); *n=(int)read_u32_be(f);
    uint8_t* raw=malloc(*n); fread(raw,1,*n,f); fclose(f);
    int* l=malloc(*n*sizeof(int));
    for(int i=0;i<*n;i++) l[i]=(int)raw[i];
    free(raw); return l;
}

/* Same RNG seed as the other MNIST tools → identical projection matrices. */
static uint32_t rng_s[4];
static uint32_t rng_next(void) {
    uint32_t result=rng_s[0]+rng_s[3];
    uint32_t t=rng_s[1]<<9;
    rng_s[2]^=rng_s[0]; rng_s[3]^=rng_s[1];
    rng_s[1]^=rng_s[2]; rng_s[0]^=rng_s[3];
    rng_s[2]^=t; rng_s[3]=(rng_s[3]<<11)|(rng_s[3]>>21);
    return result;
}

/* ── Percentile-based tau calibration ─────────────────────────────────── */

static int cmp_i64(const void* a, const void* b) {
    int64_t x = *(const int64_t*)a, y = *(const int64_t*)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}
static int64_t tau_for_density(int64_t* abs_values, size_t n, double density) {
    if (n == 0 || density <= 0.0) return 0;
    if (density >= 1.0) return abs_values[n - 1] + 1;
    qsort(abs_values, n, sizeof(int64_t), cmp_i64);
    size_t idx = (size_t)(density * (double)n);
    if (idx >= n) idx = n - 1;
    return abs_values[idx];
}

/* ── NEON-vectorized L1 distance over MTFP cells ─────────────────────── */

/* Sum |q[p] - r[p]| for p in [0, n). NEON path processes 4 lanes at a
 * time with widening accumulation to int64. Addresses H-RT1D: L1 baseline
 * must use the same SIMD shape as the routed popcount path so the speed
 * comparison reflects algorithm, not SIMD deployment asymmetry. */
static int64_t l1_distance_mtfp(
    const m4t_mtfp_t* q, const m4t_mtfp_t* r, int n)
{
    int p = 0;
    int64_t d = 0;
#if M4T_HAS_NEON
    int64x2_t acc_lo = vdupq_n_s64(0);
    int64x2_t acc_hi = vdupq_n_s64(0);
    for (; p + 4 <= n; p += 4) {
        int32x4_t vq = vld1q_s32(q + p);
        int32x4_t vr = vld1q_s32(r + p);
        int32x4_t va = vabdq_s32(vq, vr);
        acc_lo = vaddw_s32(acc_lo, vget_low_s32(va));
        acc_hi = vaddw_s32(acc_hi, vget_high_s32(va));
    }
    d = vgetq_lane_s64(acc_lo, 0) + vgetq_lane_s64(acc_lo, 1)
      + vgetq_lane_s64(acc_hi, 0) + vgetq_lane_s64(acc_hi, 1);
#endif
    for (; p < n; p++) {
        int64_t x = (int64_t)q[p] - (int64_t)r[p];
        d += (x >= 0) ? x : -x;
    }
    return d;
}

/* ── Top-k helpers ────────────────────────────────────────────────────── */

/* Maintain k smallest distances (and their labels) via insertion sort.
 * dists[] is sorted ascending; dists[0] is the nearest. */

static void topk_insert_i32(int32_t* dists, int* labels, int k,
                             int32_t new_dist, int new_label)
{
    if (new_dist >= dists[k-1]) return;
    dists[k-1] = new_dist;
    labels[k-1] = new_label;
    for (int j = k-2; j >= 0; j--) {
        if (dists[j+1] < dists[j]) {
            int32_t d = dists[j]; dists[j] = dists[j+1]; dists[j+1] = d;
            int l = labels[j]; labels[j] = labels[j+1]; labels[j+1] = l;
        } else break;
    }
}

static void topk_insert_i64(int64_t* dists, int* labels, int k,
                             int64_t new_dist, int new_label)
{
    if (new_dist >= dists[k-1]) return;
    dists[k-1] = new_dist;
    labels[k-1] = new_label;
    for (int j = k-2; j >= 0; j--) {
        if (dists[j+1] < dists[j]) {
            int64_t d = dists[j]; dists[j] = dists[j+1]; dists[j+1] = d;
            int l = labels[j]; labels[j] = labels[j+1]; labels[j+1] = l;
        } else break;
    }
}

static int majority_vote(const int* labels, int k) {
    int counts[N_CLASSES] = {0};
    for (int i = 0; i < k; i++) counts[labels[i]]++;
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (counts[c] > counts[best]) best = c;
    return best;
}

/* ── Mean and stddev helpers ──────────────────────────────────────────── */

static double mean_pct(const int* correct, int n_runs, int n_test) {
    double sum = 0.0;
    for (int i = 0; i < n_runs; i++) sum += 100.0 * (double)correct[i] / (double)n_test;
    return sum / (double)n_runs;
}
static double stddev_pct(const int* correct, int n_runs, int n_test) {
    if (n_runs < 2) return 0.0;
    double m = mean_pct(correct, n_runs, n_test);
    double s = 0.0;
    for (int i = 0; i < n_runs; i++) {
        double x = 100.0 * (double)correct[i] / (double)n_test;
        s += (x - m) * (x - m);
    }
    return sqrt(s / (double)(n_runs - 1));
}

/* ── Experiment driver ────────────────────────────────────────────────── */

#define N_SEEDS 3
#define N_NPROJS 2
#define N_MODES 2        /* 0: raw pixels, 1: deskewed pixels */
#define N_KS 3           /* k = 1, 3, 5 */

/* Run L1 k-NN over projections for one (seed, N_PROJ, mode) cell.
 * Returns correct counts at k=1, k=3, k=5. */
static void l1_knn_projections(
    const m4t_mtfp_t* train_proj, const m4t_mtfp_t* test_proj,
    int n_train, int n_test, int N_PROJ,
    const int* y_train, const int* y_test,
    int out_correct[N_KS], double* out_ms)
{
    out_correct[0] = out_correct[1] = out_correct[2] = 0;
    clock_t t0 = clock();
    for (int s = 0; s < n_test; s++) {
        const m4t_mtfp_t* q = test_proj + (size_t)s * N_PROJ;
        int64_t dists[MAX_K]; int labels[MAX_K];
        for (int j = 0; j < 5; j++) { dists[j] = INT64_MAX; labels[j] = -1; }
        for (int i = 0; i < n_train; i++) {
            const m4t_mtfp_t* r = train_proj + (size_t)i * N_PROJ;
            int64_t d = l1_distance_mtfp(q, r, N_PROJ);
            topk_insert_i64(dists, labels, 5, d, y_train[i]);
        }
        if (labels[0] == y_test[s]) out_correct[0]++;
        if (majority_vote(labels, 3) == y_test[s]) out_correct[1]++;
        if (majority_vote(labels, 5) == y_test[s]) out_correct[2]++;
    }
    *out_ms = 1000.0 * (double)(clock() - t0) / CLOCKS_PER_SEC;
}

/* Run routed k-NN over packed-trit signatures. */
static void routed_knn_signatures(
    const uint8_t* train_sigs, const uint8_t* test_sigs,
    int n_train, int n_test, int Sp,
    const int* y_train, const int* y_test,
    int out_correct[N_KS], double* out_ms)
{
    out_correct[0] = out_correct[1] = out_correct[2] = 0;
    uint8_t* mask = malloc(Sp); memset(mask, 0xFF, Sp);
    clock_t t0 = clock();
    for (int s = 0; s < n_test; s++) {
        const uint8_t* q_sig = test_sigs + (size_t)s * Sp;
        int32_t dists[MAX_K]; int labels[MAX_K];
        for (int j = 0; j < 5; j++) { dists[j] = INT32_MAX; labels[j] = -1; }
        for (int i = 0; i < n_train; i++) {
            const uint8_t* r_sig = train_sigs + (size_t)i * Sp;
            int32_t d = m4t_popcount_dist(q_sig, r_sig, mask, Sp);
            topk_insert_i32(dists, labels, 5, d, y_train[i]);
        }
        if (labels[0] == y_test[s]) out_correct[0]++;
        if (majority_vote(labels, 3) == y_test[s]) out_correct[1]++;
        if (majority_vote(labels, 5) == y_test[s]) out_correct[2]++;
    }
    *out_ms = 1000.0 * (double)(clock() - t0) / CLOCKS_PER_SEC;
    free(mask);
}

/* Run dense pixel L1 k-NN (no projection). */
static void l1_knn_pixels(
    const m4t_mtfp_t* train_pix, const m4t_mtfp_t* test_pix,
    int n_train, int n_test,
    const int* y_train, const int* y_test,
    int out_correct[N_KS], double* out_ms)
{
    out_correct[0] = out_correct[1] = out_correct[2] = 0;
    clock_t t0 = clock();
    for (int s = 0; s < n_test; s++) {
        const m4t_mtfp_t* q = test_pix + (size_t)s * INPUT_DIM;
        int64_t dists[MAX_K]; int labels[MAX_K];
        for (int j = 0; j < 5; j++) { dists[j] = INT64_MAX; labels[j] = -1; }
        for (int i = 0; i < n_train; i++) {
            const m4t_mtfp_t* r = train_pix + (size_t)i * INPUT_DIM;
            int64_t d = l1_distance_mtfp(q, r, INPUT_DIM);
            topk_insert_i64(dists, labels, 5, d, y_train[i]);
        }
        if (labels[0] == y_test[s]) out_correct[0]++;
        if (majority_vote(labels, 3) == y_test[s]) out_correct[1]++;
        if (majority_vote(labels, 5) == y_test[s]) out_correct[2]++;
    }
    *out_ms = 1000.0 * (double)(clock() - t0) / CLOCKS_PER_SEC;
}

/* Count ±1 / 0 / -1 trits in a packed-trit buffer. */
static void count_trits(
    const uint8_t* sigs, int n_sigs, int Sp, int n_trits_per_sig,
    long* out_pos, long* out_zero, long* out_neg)
{
    long pos = 0, zero = 0, neg = 0;
    for (int i = 0; i < n_sigs; i++) {
        const uint8_t* sig = sigs + (size_t)i * Sp;
        for (int p = 0; p < n_trits_per_sig; p++) {
            uint8_t code = (sig[p >> 2] >> ((p & 3) * 2)) & 0x3u;
            if (code == 0x01u) pos++;
            else if (code == 0x02u) neg++;
            else zero++;  /* 0x00 and (reserved 0x11) both counted as zero */
        }
    }
    *out_pos = pos; *out_zero = zero; *out_neg = neg;
}

/* ── Main ─────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    if(argc<2){fprintf(stderr,"Usage: %s <mnist_dir>\n",argv[0]);return 1;}

    char path[512]; int n_train, n_test;
    snprintf(path,512,"%s/train-images-idx3-ubyte",argv[1]);
    m4t_mtfp_t* x_train_raw = load_images_mtfp(path, &n_train);
    snprintf(path,512,"%s/train-labels-idx1-ubyte",argv[1]);
    int* y_train = load_labels(path, &n_train);
    snprintf(path,512,"%s/t10k-images-idx3-ubyte",argv[1]);
    m4t_mtfp_t* x_test_raw = load_images_mtfp(path, &n_test);
    snprintf(path,512,"%s/t10k-labels-idx1-ubyte",argv[1]);
    int* y_test = load_labels(path, &n_test);

    /* Build deskewed copies of both train and test pixels. */
    m4t_mtfp_t* x_train_deskew = malloc((size_t)n_train * INPUT_DIM * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* x_test_deskew  = malloc((size_t)n_test  * INPUT_DIM * sizeof(m4t_mtfp_t));
    memcpy(x_train_deskew, x_train_raw, (size_t)n_train * INPUT_DIM * sizeof(m4t_mtfp_t));
    memcpy(x_test_deskew,  x_test_raw,  (size_t)n_test  * INPUT_DIM * sizeof(m4t_mtfp_t));
    clock_t t_deskew = clock();
    deskew_all(x_train_deskew, n_train);
    deskew_all(x_test_deskew,  n_test);
    double deskew_ms = 1000.0 * (double)(clock() - t_deskew) / CLOCKS_PER_SEC;

    printf("Trit Lattice LSH — k-NN MNIST (fair-comparison edition)\n");
    printf("Loaded %d train, %d test  (deskew %.0f ms)\n\n", n_train, n_test, deskew_ms);
    printf("Sweep: %d seeds × %d N_PROJ × %d modes (raw/deskewed) × 2 paths\n",
           N_SEEDS, N_NPROJS, N_MODES);
    printf("k ∈ {1, 3, 5}; density target 0.33; tau_q calibrated per (seed, N_PROJ, mode)\n\n");

    /* ── Dense pixel k-NN baseline (deskewed, the journal historic best) ── */
    printf("=== Dense pixel k-NN baseline ===\n");
    int correct_dense_raw[N_KS], correct_dense_deskew[N_KS];
    double ms_dense_raw = 0.0, ms_dense_deskew = 0.0;

    l1_knn_pixels(x_train_raw, x_test_raw, n_train, n_test,
                  y_train, y_test, correct_dense_raw, &ms_dense_raw);
    printf("  Raw pixels (L1 k-NN, NEON):\n");
    printf("    k=1: %.2f%%   k=3: %.2f%%   k=5: %.2f%%   time: %.0f ms\n",
           correct_dense_raw[0]*100.0/n_test, correct_dense_raw[1]*100.0/n_test,
           correct_dense_raw[2]*100.0/n_test, ms_dense_raw);

    l1_knn_pixels(x_train_deskew, x_test_deskew, n_train, n_test,
                  y_train, y_test, correct_dense_deskew, &ms_dense_deskew);
    printf("  Deskewed pixels (L1 k-NN, NEON):\n");
    printf("    k=1: %.2f%%   k=3: %.2f%%   k=5: %.2f%%   time: %.0f ms\n\n",
           correct_dense_deskew[0]*100.0/n_test, correct_dense_deskew[1]*100.0/n_test,
           correct_dense_deskew[2]*100.0/n_test, ms_dense_deskew);

    /* ── Projection-dependent sweep: multi-seed × N_PROJ × mode × path ── */

    const int N_PROJS[N_NPROJS] = { 512, 2048 };
    const uint32_t SEEDS[N_SEEDS][4] = {
        { 42,   123,  456,  789  },
        { 137,  271,  331,  983  },
        { 1009, 2017, 3041, 5059 }
    };
    const double DENSITY = 0.33;

    /* results[mode][np_idx][k_idx][seed_idx] = correct count for L1 or routed */
    int correct_l1    [N_MODES][N_NPROJS][N_KS][N_SEEDS];
    int correct_routed[N_MODES][N_NPROJS][N_KS][N_SEEDS];
    double trit_pos[N_MODES][N_NPROJS][N_SEEDS], trit_zero[N_MODES][N_NPROJS][N_SEEDS], trit_neg[N_MODES][N_NPROJS][N_SEEDS];
    double ms_l1[N_MODES][N_NPROJS][N_SEEDS], ms_routed[N_MODES][N_NPROJS][N_SEEDS];

    for (int mode = 0; mode < N_MODES; mode++) {
        const m4t_mtfp_t* x_train_cur = (mode == 0) ? x_train_raw : x_train_deskew;
        const m4t_mtfp_t* x_test_cur  = (mode == 0) ? x_test_raw  : x_test_deskew;
        const char* mode_name = (mode == 0) ? "raw" : "deskewed";

        for (int np_idx = 0; np_idx < N_NPROJS; np_idx++) {
            int N_PROJ = N_PROJS[np_idx];
            int Sp = M4T_TRIT_PACKED_BYTES(N_PROJ);

            for (int seed_idx = 0; seed_idx < N_SEEDS; seed_idx++) {
                for (int i = 0; i < 4; i++) rng_s[i] = SEEDS[seed_idx][i];

                printf("  [mode=%s N_PROJ=%d seed#%d] ", mode_name, N_PROJ, seed_idx);
                fflush(stdout);

                /* Projection matrix for this seed. */
                m4t_trit_t* proj_w = malloc((size_t)N_PROJ * INPUT_DIM);
                for (int i = 0; i < N_PROJ * INPUT_DIM; i++) {
                    uint32_t r = rng_next() % 3;
                    proj_w[i] = (r == 0) ? -1 : (r == 1) ? 0 : 1;
                }
                int proj_Dp = M4T_TRIT_PACKED_BYTES(INPUT_DIM);
                uint8_t* proj_packed = malloc((size_t)N_PROJ * proj_Dp);
                m4t_pack_trits_rowmajor(proj_packed, proj_w, N_PROJ, INPUT_DIM);
                free(proj_w);

                /* Project train + test. */
                m4t_mtfp_t* train_proj = malloc((size_t)n_train * N_PROJ * sizeof(m4t_mtfp_t));
                m4t_mtfp_t* test_proj  = malloc((size_t)n_test  * N_PROJ * sizeof(m4t_mtfp_t));
                for (int i = 0; i < n_train; i++)
                    m4t_mtfp_ternary_matmul_bt(
                        train_proj + (size_t)i*N_PROJ,
                        x_train_cur + (size_t)i*INPUT_DIM,
                        proj_packed, 1, INPUT_DIM, N_PROJ);
                for (int i = 0; i < n_test; i++)
                    m4t_mtfp_ternary_matmul_bt(
                        test_proj + (size_t)i*N_PROJ,
                        x_test_cur + (size_t)i*INPUT_DIM,
                        proj_packed, 1, INPUT_DIM, N_PROJ);

                /* NEON-vectorized L1 k-NN over projections. */
                int l1_correct[N_KS]; double l1_ms;
                l1_knn_projections(train_proj, test_proj, n_train, n_test, N_PROJ,
                                   y_train, y_test, l1_correct, &l1_ms);
                for (int k = 0; k < N_KS; k++)
                    correct_l1[mode][np_idx][k][seed_idx] = l1_correct[k];
                ms_l1[mode][np_idx][seed_idx] = l1_ms;

                /* Calibrate tau_q from a sample of train projection |values|. */
                const int TAU_SAMPLE = 1000;
                int n_sample = (TAU_SAMPLE < n_train) ? TAU_SAMPLE : n_train;
                int64_t tau_q;
                {
                    size_t total = (size_t)n_sample * (size_t)N_PROJ;
                    int64_t* buf = malloc(total * sizeof(int64_t));
                    for (int i = 0; i < n_sample; i++)
                        for (int p = 0; p < N_PROJ; p++) {
                            int64_t v = train_proj[(size_t)i*N_PROJ + p];
                            buf[(size_t)i*N_PROJ + p] = (v >= 0) ? v : -v;
                        }
                    tau_q = tau_for_density(buf, total, DENSITY);
                    free(buf);
                }

                /* Extract signatures. */
                uint8_t* train_sigs = calloc((size_t)n_train * Sp, 1);
                uint8_t* test_sigs  = calloc((size_t)n_test  * Sp, 1);
                int64_t* tmp_i64 = malloc((size_t)N_PROJ * sizeof(int64_t));
                for (int i = 0; i < n_train; i++) {
                    for (int p = 0; p < N_PROJ; p++)
                        tmp_i64[p] = (int64_t)train_proj[(size_t)i*N_PROJ + p];
                    m4t_route_threshold_extract(train_sigs + (size_t)i*Sp,
                                                 tmp_i64, tau_q, N_PROJ);
                }
                for (int i = 0; i < n_test; i++) {
                    for (int p = 0; p < N_PROJ; p++)
                        tmp_i64[p] = (int64_t)test_proj[(size_t)i*N_PROJ + p];
                    m4t_route_threshold_extract(test_sigs + (size_t)i*Sp,
                                                 tmp_i64, tau_q, N_PROJ);
                }
                free(tmp_i64);

                /* Count trits on train signatures (representative; test is
                 * iid from the same distribution). */
                long t_pos, t_zero, t_neg;
                count_trits(train_sigs, n_train, Sp, N_PROJ, &t_pos, &t_zero, &t_neg);
                double total_trits = (double)t_pos + (double)t_zero + (double)t_neg;
                trit_pos [mode][np_idx][seed_idx] = 100.0 * (double)t_pos  / total_trits;
                trit_zero[mode][np_idx][seed_idx] = 100.0 * (double)t_zero / total_trits;
                trit_neg [mode][np_idx][seed_idx] = 100.0 * (double)t_neg  / total_trits;

                /* Free projections once signatures are extracted. */
                free(train_proj); free(test_proj); free(proj_packed);

                /* Routed k-NN over signatures. */
                int r_correct[N_KS]; double r_ms;
                routed_knn_signatures(train_sigs, test_sigs, n_train, n_test, Sp,
                                      y_train, y_test, r_correct, &r_ms);
                for (int k = 0; k < N_KS; k++)
                    correct_routed[mode][np_idx][k][seed_idx] = r_correct[k];
                ms_routed[mode][np_idx][seed_idx] = r_ms;

                printf("L1 %.2f/%.2f/%.2f  R %.2f/%.2f/%.2f  trits +/0/- %.1f/%.1f/%.1f  "
                       "time L1 %.0fs  R %.1fs\n",
                       l1_correct[0]*100.0/n_test, l1_correct[1]*100.0/n_test, l1_correct[2]*100.0/n_test,
                       r_correct[0]*100.0/n_test, r_correct[1]*100.0/n_test, r_correct[2]*100.0/n_test,
                       trit_pos[mode][np_idx][seed_idx],
                       trit_zero[mode][np_idx][seed_idx],
                       trit_neg[mode][np_idx][seed_idx],
                       l1_ms/1000.0, r_ms/1000.0);

                free(train_sigs); free(test_sigs);
            }
        }
    }

    /* ── Summary: mean ± stddev for each cell ──────────────────────────── */

    printf("\n=== Summary — mean ± stddev over %d seeds ===\n\n", N_SEEDS);
    printf("Dense baselines (single run, no RNG):\n");
    printf("  Raw pixel L1 k-NN          k=1 %.2f%%  k=3 %.2f%%  k=5 %.2f%%\n",
           correct_dense_raw[0]*100.0/n_test, correct_dense_raw[1]*100.0/n_test,
           correct_dense_raw[2]*100.0/n_test);
    printf("  Deskewed pixel L1 k-NN     k=1 %.2f%%  k=3 %.2f%%  k=5 %.2f%%\n\n",
           correct_dense_deskew[0]*100.0/n_test, correct_dense_deskew[1]*100.0/n_test,
           correct_dense_deskew[2]*100.0/n_test);

    printf("Projection-based (NEON-vectorized L1 baseline + routed path):\n");
    for (int mode = 0; mode < N_MODES; mode++) {
        const char* mn = (mode == 0) ? "raw" : "deskewed";
        for (int np_idx = 0; np_idx < N_NPROJS; np_idx++) {
            int N_PROJ = N_PROJS[np_idx];
            printf("\n  mode=%s  N_PROJ=%d\n", mn, N_PROJ);
            printf("    trit distribution: +%.1f%% / 0 %.1f%% / -%.1f%%\n",
                   (trit_pos[mode][np_idx][0]+trit_pos[mode][np_idx][1]+trit_pos[mode][np_idx][2])/3.0,
                   (trit_zero[mode][np_idx][0]+trit_zero[mode][np_idx][1]+trit_zero[mode][np_idx][2])/3.0,
                   (trit_neg[mode][np_idx][0]+trit_neg[mode][np_idx][1]+trit_neg[mode][np_idx][2])/3.0);
            for (int k = 0; k < N_KS; k++) {
                int k_val = (k == 0) ? 1 : (k == 1) ? 3 : 5;
                int l1_runs[N_SEEDS], r_runs[N_SEEDS];
                for (int s = 0; s < N_SEEDS; s++) {
                    l1_runs[s] = correct_l1    [mode][np_idx][k][s];
                    r_runs[s]  = correct_routed[mode][np_idx][k][s];
                }
                double l1_mean = mean_pct(l1_runs, N_SEEDS, n_test);
                double l1_sd   = stddev_pct(l1_runs, N_SEEDS, n_test);
                double r_mean  = mean_pct(r_runs, N_SEEDS, n_test);
                double r_sd    = stddev_pct(r_runs, N_SEEDS, n_test);
                printf("    k=%d  L1 %.2f ± %.2f%%   Routed %.2f ± %.2f%%   Δ %+.2f%%\n",
                       k_val, l1_mean, l1_sd, r_mean, r_sd, r_mean - l1_mean);
            }
            double l1_ms_mean = 0.0, r_ms_mean = 0.0;
            for (int s = 0; s < N_SEEDS; s++) {
                l1_ms_mean += ms_l1[mode][np_idx][s] / N_SEEDS;
                r_ms_mean  += ms_routed[mode][np_idx][s] / N_SEEDS;
            }
            printf("    mean time: L1 %.1f s   Routed %.1f s   speedup %.1fx\n",
                   l1_ms_mean/1000.0, r_ms_mean/1000.0, l1_ms_mean / r_ms_mean);
        }
    }

    printf("\nZero float at runtime. Real Trit Lattice LSH, fair comparison.\n");

    free(x_train_raw); free(x_test_raw);
    free(x_train_deskew); free(x_test_deskew);
    free(y_train); free(y_test);
    return 0;
}
