/*
 * mnist_routed_amplified.c — multi-projection ensemble with
 * audit-triggered routed fallback.
 *
 * Two amplification paths stacked:
 *   (1) K independent random ternary projections, each producing its
 *       own 60K-signature set. Each projection gives a rank-weighted
 *       k=5 prediction per test image. Ensemble = majority vote
 *       across the K per-projection predictions.
 *   (2) Audit-triggered routed fallback: when fewer than AGREE_THRESHOLD
 *       projections agree on the winning class, flatten the K per-head
 *       top-5 routed neighbors and re-vote over that routed evidence.
 *
 * The sixth-round remediation removes the dense pixel fallback entirely.
 * Uncertain queries are still detected from routed audit signals, but the
 * resolver stays inside ternary signature space.
 *
 * Reports solo per-projection accuracies, ensemble-only, and
 * ensemble+routed-fallback at several agreement thresholds.
 *
 * Usage: ./mnist_routed_amplified <mnist_dir>
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
#define K_PROJS 5          /* ensemble size */
#define N_MASTER_SEEDS 3

/* ── Data, deskew, RNG, τ (copied helpers) ────────────────────────────── */

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

/* ── Top-k and voting ─────────────────────────────────────────────────── */

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
/* Rank-weighted vote at k=5: weights {5, 4, 3, 2, 1}. */
static int vote_rank_weighted_k5(const int* labels) {
    int scores[N_CLASSES] = {0};
    for (int i = 0; i < 5; i++) scores[labels[i]] += (5 - i);
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (scores[c] > scores[best]) best = c;
    return best;
}

/* Routed fallback: flatten K heads x top-5 routed neighbors and re-score by
 * per-head rank. This keeps the resolver in signature space while consuming
 * more routed evidence than the coarse per-head winner vote. */
static int vote_flattened_rank_weighted(const int labels[K_PROJS][MAX_K]) {
    int scores[N_CLASSES] = {0};
    for (int pk = 0; pk < K_PROJS; pk++)
        for (int rank = 0; rank < MAX_K; rank++)
            scores[labels[pk][rank]] += (MAX_K - rank);
    {
        int best = 0;
        for (int c = 1; c < N_CLASSES; c++)
            if (scores[c] > scores[best]) best = c;
        return best;
    }
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

    /* Deskew both train and test. Deskewing remains an input transform; the
     * remediation removes the pixel-space fallback from the decision path. */
    deskew_all(x_train, n_train);
    deskew_all(x_test,  n_test);

    printf("Amplified routed k-NN — ensemble(K=%d) + audit-triggered routed fallback\n",
           K_PROJS);
    printf("Deskewed MNIST, N_PROJ=%d, density=%.2f, %d master seeds\n\n",
           N_PROJ, DENSITY, N_MASTER_SEEDS);

    /* Master seeds drive each ensemble run as a whole. Within a run,
     * the K projections get deterministic offsets. */
    const uint32_t MASTER_SEEDS[N_MASTER_SEEDS][4] = {
        { 42,   123,  456,  789  },
        { 137,  271,  331,  983  },
        { 1009, 2017, 3041, 5059 }
    };

    /* Per-master-seed results. Columns:
     *   0: best solo projection
     *   1: ensemble-only (K-majority vote of per-proj rank-k5 predictions)
     *   2: ensemble + routed fallback @ agree≥5 (when unanimity fails)
     *   3: ensemble + routed fallback @ agree≥4
     *   4: ensemble + routed fallback @ agree≥3
     */
    int results[5][N_MASTER_SEEDS];
    int fallback_triggers[3][N_MASTER_SEEDS];       /* triggers for agree≥5,4,3 */
    int fallback_correct[3][N_MASTER_SEEDS];        /* of triggers, how many ended correct */
    int solo_runs[K_PROJS][N_MASTER_SEEDS];         /* per-projection solo accuracy */

    for (int ms_idx = 0; ms_idx < N_MASTER_SEEDS; ms_idx++) {
        printf("=== Master seed #%d ===\n", ms_idx);
        clock_t t_seed = clock();

        /* Build K projections' signature sets. */
        int Sp = M4T_TRIT_PACKED_BYTES(N_PROJ);
        uint8_t* train_sigs[K_PROJS];
        uint8_t* test_sigs [K_PROJS];

        for (int pk = 0; pk < K_PROJS; pk++) {
            /* Seed this projection distinctly. */
            rng_s[0] = MASTER_SEEDS[ms_idx][0] + pk * 997;
            rng_s[1] = MASTER_SEEDS[ms_idx][1] + pk * 1009;
            rng_s[2] = MASTER_SEEDS[ms_idx][2] + pk * 1013;
            rng_s[3] = MASTER_SEEDS[ms_idx][3] + pk * 1019;

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

            /* Project train+test. */
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

            /* Calibrate tau_q. */
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

            /* Extract signatures. */
            train_sigs[pk] = calloc((size_t)n_train * Sp, 1);
            test_sigs [pk] = calloc((size_t)n_test  * Sp, 1);
            int64_t* tmp = malloc((size_t)N_PROJ * sizeof(int64_t));
            for (int i = 0; i < n_train; i++) {
                for (int p = 0; p < N_PROJ; p++)
                    tmp[p] = (int64_t)train_proj[(size_t)i*N_PROJ + p];
                m4t_route_threshold_extract(train_sigs[pk] + (size_t)i*Sp,
                                             tmp, tau_q, N_PROJ);
            }
            for (int i = 0; i < n_test; i++) {
                for (int p = 0; p < N_PROJ; p++)
                    tmp[p] = (int64_t)test_proj[(size_t)i*N_PROJ + p];
                m4t_route_threshold_extract(test_sigs[pk] + (size_t)i*Sp,
                                             tmp, tau_q, N_PROJ);
            }
            free(tmp);
            free(train_proj); free(test_proj); free(proj_packed);

            printf("  projection %d ready (tau_q=%lld)\n", pk, (long long)tau_q);
        }

        uint8_t* mask = malloc(Sp); memset(mask, 0xFF, Sp);

        /* ── Inference: per-projection top-5, per-projection rank-k=5
         *    prediction, K-agreement count, ensemble vote, optional routed
         *    fallback over flattened per-head routed evidence. */

        int solo_correct[K_PROJS] = {0};
        int ens_correct = 0;
        int ens_fb_correct[3] = {0,0,0};    /* at agree thresholds 5, 4, 3 */
        int fb_triggers_here[3] = {0,0,0};
        int fb_correct_here[3] = {0,0,0};

        for (int s = 0; s < n_test; s++) {
            /* Per-projection top-5 and rank-k=5 prediction. */
            int per_proj_pred[K_PROJS];
            int per_proj_labels[K_PROJS][MAX_K];
            for (int pk = 0; pk < K_PROJS; pk++) {
                const uint8_t* q_sig = test_sigs[pk] + (size_t)s * Sp;
                int32_t dists[MAX_K]; int labels[MAX_K];
                for (int j = 0; j < MAX_K; j++) {
                    dists[j] = INT32_MAX; labels[j] = -1;
                }
                for (int i = 0; i < n_train; i++) {
                    const uint8_t* r_sig = train_sigs[pk] + (size_t)i * Sp;
                    int32_t d = m4t_popcount_dist(q_sig, r_sig, mask, Sp);
                    topk_insert_i32(dists, labels, MAX_K, d, y_train[i]);
                }
                for (int j = 0; j < MAX_K; j++) per_proj_labels[pk][j] = labels[j];
                per_proj_pred[pk] = vote_rank_weighted_k5(labels);
                if (per_proj_pred[pk] == y_test[s]) solo_correct[pk]++;
            }

            /* Ensemble: majority vote across K projections. */
            int agreement[N_CLASSES] = {0};
            for (int pk = 0; pk < K_PROJS; pk++) agreement[per_proj_pred[pk]]++;
            int ens_pred = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (agreement[c] > agreement[ens_pred]) ens_pred = c;
            int ens_agreement = agreement[ens_pred];

            if (ens_pred == y_test[s]) ens_correct++;

            /* Routed fallback evaluation at three agreement thresholds. A
             * higher threshold is stricter (more reroutes); a lower threshold
             * trusts the ensemble more. */
            const int AGREE_THRESH[3] = {5, 4, 3};

            int fallback_pred = -1;  /* lazy compute: only if any threshold triggers */

            for (int ti = 0; ti < 3; ti++) {
                int thresh = AGREE_THRESH[ti];
                int use_fallback = (ens_agreement < thresh);
                int final_pred;

                if (use_fallback) {
                    /* Compute routed fallback once across all thresholds that
                     * triggered. */
                    if (fallback_pred < 0) {
                        fallback_pred = vote_flattened_rank_weighted(per_proj_labels);
                    }
                    final_pred = fallback_pred;
                    fb_triggers_here[ti]++;
                    if (final_pred == y_test[s]) fb_correct_here[ti]++;
                } else {
                    final_pred = ens_pred;
                }

                if (final_pred == y_test[s]) ens_fb_correct[ti]++;
            }

            if (s > 0 && s % 2000 == 0)
                printf("    progress %d/%d (ens so far: %d = %.2f%%)\n",
                       s, n_test, ens_correct, 100.0*ens_correct/s);
        }

        for (int pk = 0; pk < K_PROJS; pk++) solo_runs[pk][ms_idx] = solo_correct[pk];
        results[0][ms_idx] = solo_correct[0];
        for (int pk = 1; pk < K_PROJS; pk++)
            if (solo_correct[pk] > results[0][ms_idx]) results[0][ms_idx] = solo_correct[pk];
        results[1][ms_idx] = ens_correct;
        for (int ti = 0; ti < 3; ti++) {
            results[2+ti][ms_idx] = ens_fb_correct[ti];
            fallback_triggers[ti][ms_idx] = fb_triggers_here[ti];
            fallback_correct [ti][ms_idx] = fb_correct_here [ti];
        }

        double t_elapsed = (double)(clock() - t_seed) / CLOCKS_PER_SEC;
        printf("  solo accuracies:");
        for (int pk = 0; pk < K_PROJS; pk++)
            printf(" %.2f", solo_correct[pk]*100.0/n_test);
        printf("  →  best %.2f\n", results[0][ms_idx]*100.0/n_test);
        printf("  ensemble (no fallback):   %.2f%%\n",
               ens_correct*100.0/n_test);
        printf("  ensemble+routed-fb (agree≥5): %.2f%%   (fb triggered %d, fb-correct %d)\n",
               ens_fb_correct[0]*100.0/n_test, fb_triggers_here[0], fb_correct_here[0]);
        printf("  ensemble+routed-fb (agree≥4): %.2f%%   (fb triggered %d, fb-correct %d)\n",
               ens_fb_correct[1]*100.0/n_test, fb_triggers_here[1], fb_correct_here[1]);
        printf("  ensemble+routed-fb (agree≥3): %.2f%%   (fb triggered %d, fb-correct %d)\n",
               ens_fb_correct[2]*100.0/n_test, fb_triggers_here[2], fb_correct_here[2]);
        printf("  time: %.0f s\n\n", t_elapsed);

        free(mask);
        for (int pk = 0; pk < K_PROJS; pk++) {
            free(train_sigs[pk]);
            free(test_sigs [pk]);
        }
    }

    /* ── Summary ──────────────────────────────────────────────────────── */

    printf("=== Summary — mean ± stddev over %d master seeds ===\n\n",
           N_MASTER_SEEDS);
    const char* row_names[5] = {
        "Best solo projection (rank-k=5)",
        "Ensemble (K=5, no fallback)",
        "Ensemble+routed-FB (agree≥5 → unanimous-only trust)",
        "Ensemble+routed-FB (agree≥4 → ≥4-of-5 trust)",
        "Ensemble+routed-FB (agree≥3 → simple majority trust)"
    };
    for (int r = 0; r < 5; r++) {
        double m = mean_pct(results[r], N_MASTER_SEEDS, n_test);
        double sd = stddev_pct(results[r], N_MASTER_SEEDS, n_test);
        printf("  %-48s  %.2f ± %.2f%%\n", row_names[r], m, sd);
    }

    printf("\n  Fallback trigger rates and recovery:\n");
    const char* thr_names[3] = {"agree≥5", "agree≥4", "agree≥3"};
    for (int ti = 0; ti < 3; ti++) {
        double trig_mean = 0.0, rec_mean = 0.0;
        for (int ms = 0; ms < N_MASTER_SEEDS; ms++) {
            trig_mean += 100.0 * fallback_triggers[ti][ms] / n_test;
            if (fallback_triggers[ti][ms] > 0)
                rec_mean += 100.0 * fallback_correct[ti][ms] / fallback_triggers[ti][ms];
        }
        trig_mean /= N_MASTER_SEEDS;
        rec_mean  /= N_MASTER_SEEDS;
        printf("    %-8s  triggered on %.2f%% of queries, correct on %.1f%% of triggers\n",
               thr_names[ti], trig_mean, rec_mean);
    }

    printf("\nReference routed baselines (from prior journal entries):\n");
    printf("  rank-weighted k=5 single projection: 97.86 ± 0.01%% (journal/weighted_voting_adaptation.md)\n");
    printf("  majority k=3 single projection:      97.79 ± 0.05%%\n");

    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
