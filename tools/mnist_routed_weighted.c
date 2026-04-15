/*
 * STATUS: research scaffolding, not production architecture.
 * Runs routing primitives inside an O(N_train) dense outer loop.
 * Failure-guided adaptation test using distance-weighted voting.
 * Produced the rank-weighted k=5 adaptation finding that motivated
 * the "mechanism that worked" LMM cycle.
 * For production routed k-NN use tools/mnist_routed_bucket{,_multi}.c
 * on libglyph. See docs/FINDINGS.md Axis 5.
 *
 * mnist_routed_weighted.c — failure-guided adaptation test: does
 * distance-weighted voting recover the NARROW MISS cases the audit
 * trail exposed?
 *
 * Same configuration as the best routed k-NN result (deskewed pixels,
 * N_PROJ=2048, symmetric balanced base-3). Same projections, same
 * training, same signatures — only the VOTE RULE differs.
 *
 * Three vote rules compared per (seed, k):
 *   majority:     count labels in top-k, argmax. The baseline.
 *   distance-wt:  weight = max_dist - d;  nearer prototypes count more.
 *   rank-wt:      weight = k - rank;       top-1 counts k, top-k counts 1.
 *
 * Hypothesis (from journal/routed_inspectability_trace.md):
 *   ~25-30 of the 74 NARROW MISS cases at k=3 have correct top-1 but
 *   wrong-class cluster at ranks 2..5. Distance- or rank-weighted
 *   voting should recover most of those, moving accuracy toward
 *   97.79% + ~0.3% ≈ 98.1%.
 *
 * Usage: ./mnist_routed_weighted <mnist_dir>
 */

#include "m4t_types.h"
#include "m4t_mtfp.h"
#include "m4t_trit_pack.h"
#include "m4t_ternary_matmul.h"
#include "m4t_route.h"

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
#define N_PROJ 2048
#define DENSITY 0.33
#define MAX_K 5
#define N_SEEDS 3

/* ── Data + deskew + RNG + τ helpers (copy of the minimal set) ──────── */

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

static uint32_t rng_s[4];
static uint32_t rng_next(void) {
    uint32_t result=rng_s[0]+rng_s[3];
    uint32_t t=rng_s[1]<<9;
    rng_s[2]^=rng_s[0]; rng_s[3]^=rng_s[1];
    rng_s[1]^=rng_s[2]; rng_s[0]^=rng_s[3];
    rng_s[2]^=t; rng_s[3]=(rng_s[3]<<11)|(rng_s[3]>>21);
    return result;
}
static int cmp_i64(const void* a, const void* b) {
    int64_t x=*(const int64_t*)a, y=*(const int64_t*)b;
    return (x<y)?-1:(x>y)?1:0;
}
static int64_t tau_for_density(int64_t* v, size_t n, double d) {
    if (n==0||d<=0.0) return 0;
    if (d>=1.0) return v[n-1]+1;
    qsort(v, n, sizeof(int64_t), cmp_i64);
    size_t idx = (size_t)(d * (double)n);
    if (idx >= n) idx = n-1;
    return v[idx];
}

/* ── Top-k with insertion sort ────────────────────────────────────────── */

static void topk_insert(int32_t* dists, int* labels, int k,
                         int32_t new_d, int new_l)
{
    if (new_d >= dists[k-1]) return;
    dists[k-1] = new_d;
    labels[k-1] = new_l;
    for (int j=k-2; j>=0; j--) {
        if (dists[j+1] < dists[j]) {
            int32_t d=dists[j]; dists[j]=dists[j+1]; dists[j+1]=d;
            int l=labels[j]; labels[j]=labels[j+1]; labels[j+1]=l;
        } else break;
    }
}

/* ── Three vote rules ─────────────────────────────────────────────────── */

static int vote_majority(const int* labels, int k) {
    int counts[N_CLASSES] = {0};
    for (int i = 0; i < k; i++) counts[labels[i]]++;
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (counts[c] > counts[best]) best = c;
    return best;
}

/* Distance-weighted: weight = max_dist - d. Nearer prototypes count more.
 * max_dist = 2 * N_PROJ (the popcount-dist upper bound). */
static int vote_distance_weighted(
    const int* labels, const int32_t* dists, int k, int32_t max_dist)
{
    int32_t scores[N_CLASSES] = {0};
    for (int i = 0; i < k; i++) {
        int32_t w = max_dist - dists[i];
        if (w < 1) w = 1;  /* floor for the pathological d = max_dist case */
        scores[labels[i]] += w;
    }
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (scores[c] > scores[best]) best = c;
    return best;
}

/* Rank-weighted: top-1 counts k, top-2 counts k-1, ..., top-k counts 1.
 * Simpler than distance-weighted; doesn't require max_dist. */
static int vote_rank_weighted(const int* labels, int k) {
    int scores[N_CLASSES] = {0};
    for (int i = 0; i < k; i++) scores[labels[i]] += (k - i);
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (scores[c] > scores[best]) best = c;
    return best;
}

/* ── Mean/stddev over seeds ───────────────────────────────────────────── */

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

/* ── Main ─────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    if(argc<2){fprintf(stderr,"Usage: %s <mnist_dir>\n",argv[0]);return 1;}

    char path[512]; int n_train, n_test;
    snprintf(path,512,"%s/train-images-idx3-ubyte",argv[1]);
    m4t_mtfp_t* x_train=load_images_mtfp(path,&n_train);
    snprintf(path,512,"%s/train-labels-idx1-ubyte",argv[1]);
    int* y_train=load_labels(path,&n_train);
    snprintf(path,512,"%s/t10k-images-idx3-ubyte",argv[1]);
    m4t_mtfp_t* x_test=load_images_mtfp(path,&n_test);
    snprintf(path,512,"%s/t10k-labels-idx1-ubyte",argv[1]);
    int* y_test=load_labels(path,&n_test);

    deskew_all(x_train, n_train);
    deskew_all(x_test,  n_test);

    printf("Failure-guided adaptation: distance-weighted + rank-weighted voting\n");
    printf("Deskewed MNIST, N_PROJ=%d, density=%.2f, %d seeds\n", N_PROJ, DENSITY, N_SEEDS);
    printf("Baseline (majority vote at k=3): 97.79 ± 0.05%% from journal/routed_knn_mnist.md\n\n");

    const uint32_t SEEDS[N_SEEDS][4] = {
        { 42,   123,  456,  789  },
        { 137,  271,  331,  983  },
        { 1009, 2017, 3041, 5059 }
    };

    /* Results: [vote_rule][k_idx][seed_idx] = correct count.
     * vote_rule: 0=majority, 1=distance-weighted, 2=rank-weighted.
     * k_idx:     0=k3, 1=k5. */
    int correct[3][2][N_SEEDS];

    for (int seed_idx = 0; seed_idx < N_SEEDS; seed_idx++) {
        for (int i = 0; i < 4; i++) rng_s[i] = SEEDS[seed_idx][i];

        printf("  [seed #%d] ", seed_idx); fflush(stdout);
        clock_t t0 = clock();

        /* Projection matrix. */
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
            m4t_mtfp_ternary_matmul_bt(train_proj + (size_t)i*N_PROJ,
                                        x_train + (size_t)i*INPUT_DIM,
                                        proj_packed, 1, INPUT_DIM, N_PROJ);
        for (int i = 0; i < n_test; i++)
            m4t_mtfp_ternary_matmul_bt(test_proj + (size_t)i*N_PROJ,
                                        x_test + (size_t)i*INPUT_DIM,
                                        proj_packed, 1, INPUT_DIM, N_PROJ);

        /* Calibrate τ_q from 1000-image sample. */
        int64_t tau_q;
        {
            size_t total = (size_t)1000 * N_PROJ;
            int64_t* buf = malloc(total * sizeof(int64_t));
            for (int i = 0; i < 1000; i++)
                for (int p = 0; p < N_PROJ; p++) {
                    int64_t v = train_proj[(size_t)i*N_PROJ + p];
                    buf[(size_t)i*N_PROJ + p] = (v >= 0) ? v : -v;
                }
            tau_q = tau_for_density(buf, total, DENSITY);
            free(buf);
        }

        int Sp = M4T_TRIT_PACKED_BYTES(N_PROJ);
        uint8_t* train_sigs = calloc((size_t)n_train * Sp, 1);
        uint8_t* test_sigs  = calloc((size_t)n_test  * Sp, 1);
        int64_t* tmp = malloc((size_t)N_PROJ * sizeof(int64_t));
        for (int i = 0; i < n_train; i++) {
            for (int p = 0; p < N_PROJ; p++)
                tmp[p] = (int64_t)train_proj[(size_t)i*N_PROJ + p];
            m4t_route_threshold_extract(train_sigs + (size_t)i*Sp, tmp, tau_q, N_PROJ);
        }
        for (int i = 0; i < n_test; i++) {
            for (int p = 0; p < N_PROJ; p++)
                tmp[p] = (int64_t)test_proj[(size_t)i*N_PROJ + p];
            m4t_route_threshold_extract(test_sigs + (size_t)i*Sp, tmp, tau_q, N_PROJ);
        }
        free(tmp);
        free(train_proj); free(test_proj); free(proj_packed);

        uint8_t* mask = malloc(Sp); memset(mask, 0xFF, Sp);

        /* Inference: keep top-5, compute all six vote-rule outputs per query. */
        int c[3][2] = {{0,0},{0,0},{0,0}};  /* [rule][k_idx] */
        int32_t max_dist = 2 * N_PROJ;

        for (int s = 0; s < n_test; s++) {
            const uint8_t* q_sig = test_sigs + (size_t)s*Sp;
            int32_t dists[MAX_K]; int labels[MAX_K];
            for (int j = 0; j < MAX_K; j++) {
                dists[j] = INT32_MAX; labels[j] = -1;
            }
            for (int i = 0; i < n_train; i++) {
                const uint8_t* r_sig = train_sigs + (size_t)i*Sp;
                int32_t d = m4t_popcount_dist(q_sig, r_sig, mask, Sp);
                topk_insert(dists, labels, MAX_K, d, y_train[i]);
            }

            /* Six predictions from the same top-5: 3 vote rules × 2 k values. */
            int pred_maj3 = vote_majority(labels, 3);
            int pred_maj5 = vote_majority(labels, 5);
            int pred_dw3  = vote_distance_weighted(labels, dists, 3, max_dist);
            int pred_dw5  = vote_distance_weighted(labels, dists, 5, max_dist);
            int pred_rw3  = vote_rank_weighted(labels, 3);
            int pred_rw5  = vote_rank_weighted(labels, 5);

            if (pred_maj3 == y_test[s]) c[0][0]++;
            if (pred_maj5 == y_test[s]) c[0][1]++;
            if (pred_dw3  == y_test[s]) c[1][0]++;
            if (pred_dw5  == y_test[s]) c[1][1]++;
            if (pred_rw3  == y_test[s]) c[2][0]++;
            if (pred_rw5  == y_test[s]) c[2][1]++;
        }

        for (int r = 0; r < 3; r++)
            for (int kk = 0; kk < 2; kk++)
                correct[r][kk][seed_idx] = c[r][kk];

        double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("maj_k3 %.2f / dw_k3 %.2f / rw_k3 %.2f  |  maj_k5 %.2f / dw_k5 %.2f / rw_k5 %.2f  (%.0fs)\n",
               c[0][0]*100.0/n_test, c[1][0]*100.0/n_test, c[2][0]*100.0/n_test,
               c[0][1]*100.0/n_test, c[1][1]*100.0/n_test, c[2][1]*100.0/n_test,
               elapsed);

        free(train_sigs); free(test_sigs); free(mask);
    }

    /* Summary table. */
    printf("\n=== Summary — mean ± stddev over %d seeds ===\n\n", N_SEEDS);
    const char* rule_names[3] = {"majority", "distance-weighted", "rank-weighted"};

    for (int kk = 0; kk < 2; kk++) {
        int k_val = (kk == 0) ? 3 : 5;
        printf("  k=%d:\n", k_val);
        double baseline = mean_pct(correct[0][kk], N_SEEDS, n_test);
        for (int r = 0; r < 3; r++) {
            double m = mean_pct(correct[r][kk], N_SEEDS, n_test);
            double sd = stddev_pct(correct[r][kk], N_SEEDS, n_test);
            printf("    %-20s  %.2f ± %.2f%%   Δ from majority: %+.3f%%\n",
                   rule_names[r], m, sd, m - baseline);
        }
        printf("\n");
    }

    printf("Prediction from trace analysis: distance-weighted or rank-weighted\n");
    printf("should recover 25-30 of the 74 NARROW MISS cases (~0.25-0.30%% gain).\n");

    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
