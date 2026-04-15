/*
 * mnist_full_sweep.c — full matrix sweep over (N_PROJ × density × k × vote_rule).
 *
 * Deskewed MNIST, three RNG seeds. For each (N_PROJ, density, seed) the
 * signatures are built once; the top-7 is computed once per query; all
 * 9 combinations of (k ∈ {3, 5, 7}) × (vote ∈ {majority, rank-wt, exp-wt})
 * are derived from the same top-7 by fast integer passes.
 *
 * Matrix:
 *   N_PROJ     ∈ {1024, 2048, 4096}     (3)
 *   density    ∈ {0.25, 0.33, 0.50}     (3)
 *   k          ∈ {3, 5, 7}              (3)
 *   vote_rule  ∈ {majority, rank, exp}  (3)
 *   seeds      ∈ 3 master seeds         (3)
 * Total: 81 cells × 3 seeds = 243 measurements.
 *
 * Runtime: ~20-25 minutes on M-series single-threaded.
 *
 * Usage: ./mnist_full_sweep <mnist_dir>
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
#define MAX_K 7          /* top-7 kept per query; k ∈ {3,5,7} derived from it */
#define N_SEEDS 3

/* ── Data + deskew + RNG + τ helpers (shared with other tools) ───────── */

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

/* ── Top-k maintenance ────────────────────────────────────────────────── */

static void topk_insert_i32(int32_t* dists, int* labels, int k,
                             int32_t new_d, int new_l)
{
    if (new_d >= dists[k-1]) return;
    dists[k-1] = new_d; labels[k-1] = new_l;
    for (int j=k-2; j>=0; j--) {
        if (dists[j+1] < dists[j]) {
            int32_t d=dists[j]; dists[j]=dists[j+1]; dists[j+1]=d;
            int l=labels[j]; labels[j]=labels[j+1]; labels[j+1]=l;
        } else break;
    }
}

/* ── Vote rules (three, all integer-arithmetic) ──────────────────────── */

static int vote_majority(const int* labels, int k) {
    int counts[N_CLASSES] = {0};
    for (int i = 0; i < k; i++) counts[labels[i]]++;
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (counts[c] > counts[best]) best = c;
    return best;
}
static int vote_rank_weighted(const int* labels, int k) {
    int scores[N_CLASSES] = {0};
    for (int i = 0; i < k; i++) scores[labels[i]] += (k - i);
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (scores[c] > scores[best]) best = c;
    return best;
}
/* Exponential: weight = 2^(k - rank - 1). At k=5: {16, 8, 4, 2, 1}.
 * Top-1's weight is always ~50% of total — tests the "too-steep" hypothesis. */
static int vote_exponential_weighted(const int* labels, int k) {
    int scores[N_CLASSES] = {0};
    for (int i = 0; i < k; i++) scores[labels[i]] += (1 << (k - i - 1));
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (scores[c] > scores[best]) best = c;
    return best;
}

/* ── Mean/stddev ──────────────────────────────────────────────────────── */

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

    /* Matrix axes.
     *
     * Canonical sweep covers {1024, 2048, 4096} — the accuracy range where
     * the substrate's operating points live. Spot probes at other N_PROJ
     * values are documented in journal/full_scaling_curve.md; rerun those
     * by editing this array. */
    const int    N_PROJ_VALUES[]  = {1024, 2048, 4096};
    const int    N_N_PROJ         = 3;
    const double DENSITY_VALUES[] = {0.25, 0.33, 0.50};
    const int    N_DENSITIES      = 3;
    const int    K_VALUES[]       = {3, 5, 7};
    const int    N_KS             = 3;
    const int    N_VOTES          = 3;
    const char*  VOTE_NAMES[]     = {"majority", "rank-wt", "exp-wt"};

    const uint32_t SEEDS[N_SEEDS][4] = {
        { 42,   123,  456,  789  },
        { 137,  271,  331,  983  },
        { 1009, 2017, 3041, 5059 }
    };

    /* Results array. Index: [n_proj][density][k][vote][seed]. */
    int results[16][3][3][3][N_SEEDS];
    memset(results, 0, sizeof(results));
    double actual_density[16][3][N_SEEDS];

    printf("Full matrix sweep — deskewed MNIST, %d seeds per cell\n", N_SEEDS);
    printf("N_PROJ ∈ {1024, 2048, 4096}   density ∈ {0.25, 0.33, 0.50}\n");
    printf("k ∈ {3, 5, 7}   vote ∈ {majority, rank-wt, exp-wt}\n");
    printf("Total: 3×3×3×3 = 81 cells × 3 seeds = 243 measurements\n\n");

    clock_t t_global = clock();

    for (int np_idx = 0; np_idx < N_N_PROJ; np_idx++) {
        int N_PROJ = N_PROJ_VALUES[np_idx];
        int Sp = M4T_TRIT_PACKED_BYTES(N_PROJ);

        for (int seed_idx = 0; seed_idx < N_SEEDS; seed_idx++) {
            for (int i = 0; i < 4; i++) rng_s[i] = SEEDS[seed_idx][i];

            /* Projection matrix (once per seed regardless of density, since
             * the matrix is determined by the seed). */
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
            clock_t t_proj = clock();
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
            double proj_s = (double)(clock() - t_proj) / CLOCKS_PER_SEC;

            /* Pre-sample the train-projection magnitudes once per seed for
             * fast tau computation across densities. */
            const int TAU_SAMPLE = 1000;
            size_t tau_total = (size_t)TAU_SAMPLE * N_PROJ;
            int64_t* tau_buf = malloc(tau_total * sizeof(int64_t));
            for (int i = 0; i < TAU_SAMPLE; i++)
                for (int p = 0; p < N_PROJ; p++) {
                    int64_t v = train_proj[(size_t)i*N_PROJ + p];
                    tau_buf[(size_t)i*N_PROJ + p] = (v >= 0) ? v : -v;
                }

            for (int d_idx = 0; d_idx < N_DENSITIES; d_idx++) {
                double density = DENSITY_VALUES[d_idx];

                /* Fresh buffer copy for each density's qsort. */
                int64_t* work = malloc(tau_total * sizeof(int64_t));
                memcpy(work, tau_buf, tau_total * sizeof(int64_t));
                int64_t tau_q = tau_for_density(work, tau_total, density);
                free(work);

                /* Extract signatures. */
                uint8_t* train_sigs = calloc((size_t)n_train * Sp, 1);
                uint8_t* test_sigs  = calloc((size_t)n_test  * Sp, 1);
                int64_t* tmp = malloc((size_t)N_PROJ * sizeof(int64_t));
                long zero_count = 0, total_count = 0;
                for (int i = 0; i < n_train; i++) {
                    for (int p = 0; p < N_PROJ; p++)
                        tmp[p] = (int64_t)train_proj[(size_t)i*N_PROJ + p];
                    uint8_t* sig = train_sigs + (size_t)i*Sp;
                    m4t_route_threshold_extract(sig, tmp, tau_q, N_PROJ);
                    for (int p = 0; p < N_PROJ; p++) {
                        uint8_t code = (sig[p >> 2] >> ((p & 3) * 2)) & 0x3u;
                        total_count++;
                        if (code == 0) zero_count++;
                    }
                }
                for (int i = 0; i < n_test; i++) {
                    for (int p = 0; p < N_PROJ; p++)
                        tmp[p] = (int64_t)test_proj[(size_t)i*N_PROJ + p];
                    m4t_route_threshold_extract(test_sigs + (size_t)i*Sp,
                                                 tmp, tau_q, N_PROJ);
                }
                free(tmp);
                actual_density[np_idx][d_idx][seed_idx] =
                    (double)zero_count / (double)total_count;

                /* Run k-NN with top-MAX_K; derive all (k, vote) outputs. */
                uint8_t* mask = malloc(Sp); memset(mask, 0xFF, Sp);
                clock_t t_knn = clock();

                int cell_correct[N_KS][N_VOTES] = {{0}};
                for (int s = 0; s < n_test; s++) {
                    const uint8_t* q_sig = test_sigs + (size_t)s * Sp;
                    int32_t dists[MAX_K]; int labels[MAX_K];
                    for (int j = 0; j < MAX_K; j++) {
                        dists[j] = INT32_MAX; labels[j] = -1;
                    }
                    for (int i = 0; i < n_train; i++) {
                        const uint8_t* r_sig = train_sigs + (size_t)i*Sp;
                        int32_t d = m4t_popcount_dist(q_sig, r_sig, mask, Sp);
                        topk_insert_i32(dists, labels, MAX_K, d, y_train[i]);
                    }
                    /* Derive predictions for all (k, vote). */
                    for (int k_idx = 0; k_idx < N_KS; k_idx++) {
                        int kval = K_VALUES[k_idx];
                        int preds[N_VOTES];
                        preds[0] = vote_majority(labels, kval);
                        preds[1] = vote_rank_weighted(labels, kval);
                        preds[2] = vote_exponential_weighted(labels, kval);
                        for (int v_idx = 0; v_idx < N_VOTES; v_idx++)
                            if (preds[v_idx] == y_test[s])
                                cell_correct[k_idx][v_idx]++;
                    }
                }
                double knn_s = (double)(clock() - t_knn) / CLOCKS_PER_SEC;

                for (int k_idx = 0; k_idx < N_KS; k_idx++)
                    for (int v_idx = 0; v_idx < N_VOTES; v_idx++)
                        results[np_idx][d_idx][k_idx][v_idx][seed_idx] =
                            cell_correct[k_idx][v_idx];

                printf("  N=%d d=%.2f seed#%d  actual_zero=%.1f%%  "
                       "[proj %.1fs knn %.1fs]\n",
                       N_PROJ, density, seed_idx,
                       100.0 * actual_density[np_idx][d_idx][seed_idx],
                       proj_s, knn_s);

                free(train_sigs); free(test_sigs); free(mask);
            }

            free(tau_buf);
            free(train_proj); free(test_proj); free(proj_packed);
        }
    }

    double t_total = (double)(clock() - t_global) / CLOCKS_PER_SEC;
    printf("\n=== Total sweep time: %.1f min ===\n\n", t_total / 60.0);

    /* ── Summary table: per cell, mean ± stddev ──────────────────────── */

    printf("Full matrix (mean accuracy %% over %d seeds):\n\n", N_SEEDS);
    printf("                    | majority     | rank-wt      | exp-wt\n");
    printf("  N_PROJ  d     k   |  mean   sd   |  mean   sd   |  mean   sd\n");
    printf("  ------  ----  --  |  ----- ----- |  ----- ----- |  ----- -----\n");

    /* Track overall top-10. */
    double best_score[10] = {0};
    int    best_idx[10][4];
    int n_stored = 0;

    for (int np_idx = 0; np_idx < N_N_PROJ; np_idx++) {
        int N_PROJ = N_PROJ_VALUES[np_idx];
        for (int d_idx = 0; d_idx < N_DENSITIES; d_idx++) {
            double density = DENSITY_VALUES[d_idx];
            for (int k_idx = 0; k_idx < N_KS; k_idx++) {
                int kval = K_VALUES[k_idx];
                printf("   %4d   %.2f  %2d  |", N_PROJ, density, kval);
                for (int v_idx = 0; v_idx < N_VOTES; v_idx++) {
                    int runs[N_SEEDS];
                    for (int s = 0; s < N_SEEDS; s++)
                        runs[s] = results[np_idx][d_idx][k_idx][v_idx][s];
                    double m = mean_pct(runs, N_SEEDS, n_test);
                    double sd = stddev_pct(runs, N_SEEDS, n_test);
                    printf(" %6.2f %5.2f |", m, sd);

                    /* Track top-10. */
                    if (n_stored < 10) {
                        best_score[n_stored] = m;
                        best_idx[n_stored][0] = np_idx;
                        best_idx[n_stored][1] = d_idx;
                        best_idx[n_stored][2] = k_idx;
                        best_idx[n_stored][3] = v_idx;
                        n_stored++;
                    } else {
                        /* Find worst in current top-10; replace if this is better. */
                        int worst = 0;
                        for (int i = 1; i < 10; i++)
                            if (best_score[i] < best_score[worst]) worst = i;
                        if (m > best_score[worst]) {
                            best_score[worst] = m;
                            best_idx[worst][0] = np_idx;
                            best_idx[worst][1] = d_idx;
                            best_idx[worst][2] = k_idx;
                            best_idx[worst][3] = v_idx;
                        }
                    }
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    /* Sort top-10 descending. */
    for (int i = 0; i < 10; i++) {
        int best = i;
        for (int j = i+1; j < 10; j++)
            if (best_score[j] > best_score[best]) best = j;
        if (best != i) {
            double ts = best_score[i]; best_score[i] = best_score[best]; best_score[best] = ts;
            for (int c = 0; c < 4; c++) {
                int tmp = best_idx[i][c];
                best_idx[i][c] = best_idx[best][c];
                best_idx[best][c] = tmp;
            }
        }
    }

    printf("\n=== Top 10 configurations by mean accuracy ===\n\n");
    printf("  rank  N_PROJ  d     k   vote       mean   sd      actual_zero\n");
    for (int i = 0; i < 10; i++) {
        int np = best_idx[i][0], d = best_idx[i][1];
        int k = best_idx[i][2], v = best_idx[i][3];
        int runs[N_SEEDS];
        for (int s = 0; s < N_SEEDS; s++) runs[s] = results[np][d][k][v][s];
        double m = mean_pct(runs, N_SEEDS, n_test);
        double sd = stddev_pct(runs, N_SEEDS, n_test);
        double ad = 0.0;
        for (int s = 0; s < N_SEEDS; s++) ad += actual_density[np][d][s];
        ad /= N_SEEDS;
        printf("  #%-3d  %4d    %.2f  %2d  %-10s %6.2f %5.2f    %.1f%%\n",
               i+1, N_PROJ_VALUES[np], DENSITY_VALUES[d], K_VALUES[k],
               VOTE_NAMES[v], m, sd, 100.0 * ad);
    }

    printf("\n=== Per-N_PROJ best ===\n\n");
    for (int np_idx = 0; np_idx < N_N_PROJ; np_idx++) {
        double best_m = 0.0; int bd=0, bk=0, bv=0;
        for (int d_idx = 0; d_idx < N_DENSITIES; d_idx++)
            for (int k_idx = 0; k_idx < N_KS; k_idx++)
                for (int v_idx = 0; v_idx < N_VOTES; v_idx++) {
                    int runs[N_SEEDS];
                    for (int s = 0; s < N_SEEDS; s++)
                        runs[s] = results[np_idx][d_idx][k_idx][v_idx][s];
                    double m = mean_pct(runs, N_SEEDS, n_test);
                    if (m > best_m) { best_m = m; bd=d_idx; bk=k_idx; bv=v_idx; }
                }
        int runs[N_SEEDS];
        for (int s = 0; s < N_SEEDS; s++)
            runs[s] = results[np_idx][bd][bk][bv][s];
        double sd = stddev_pct(runs, N_SEEDS, n_test);
        printf("  N_PROJ=%4d  best: d=%.2f  k=%d  %-10s  →  %.2f ± %.2f%%\n",
               N_PROJ_VALUES[np_idx], DENSITY_VALUES[bd],
               K_VALUES[bk], VOTE_NAMES[bv], best_m, sd);
    }

    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
