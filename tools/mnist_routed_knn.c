/*
 * mnist_routed_knn.c — Trit Lattice LSH with k-NN, fully routed.
 *
 * The real LSH consumer the repo has been implicitly pointing at since
 * the reset: 60K training signatures as prototypes (not 10 class
 * centroids), k-NN classification by Hamming distance over packed-trit
 * signatures. Compares side-by-side against MTFP19 L1 k-NN as the
 * information-fidelity baseline.
 *
 * Per-side tau_q is calibrated from the empirical projection
 * distribution, so train and test signatures have matching zero
 * density (symmetric §18-passing deployment by construction).
 *
 * Usage: ./mnist_routed_knn <mnist_dir>
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
#include <time.h>

#define INPUT_DIM 784
#define N_CLASSES 10

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

/* ── Top-k helpers ────────────────────────────────────────────────────── */

/* Maintain k smallest distances (and their labels) via insertion sort.
 * dists[] is sorted ascending; dists[0] is the nearest. */
#define MAX_K 7

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

    printf("Trit Lattice LSH — k-NN MNIST (real LSH; %d prototypes per class)\n",
           n_train / N_CLASSES);
    printf("Loaded %d train, %d test\n\n", n_train, n_test);

    const int N_PROJS[] = { 512, 2048 };
    const int N_SIZES = (int)(sizeof(N_PROJS)/sizeof(N_PROJS[0]));
    const double DENSITY = 0.33;  /* balanced base-3 */

    for (int si = 0; si < N_SIZES; si++) {
        int N_PROJ = N_PROJS[si];
        rng_s[0]=42; rng_s[1]=123; rng_s[2]=456; rng_s[3]=789;

        printf("=== N_PROJ = %d ===\n", N_PROJ);

        /* Projection matrix (identical seed → identical matrix as other tools). */
        m4t_trit_t* proj_w=malloc((size_t)N_PROJ*INPUT_DIM);
        for(int i=0;i<N_PROJ*INPUT_DIM;i++){
            uint32_t r=rng_next()%3;
            proj_w[i]=(r==0)?-1:(r==1)?0:1;
        }
        int proj_Dp=M4T_TRIT_PACKED_BYTES(INPUT_DIM);
        uint8_t* proj_packed=malloc((size_t)N_PROJ*proj_Dp);
        m4t_pack_trits_rowmajor(proj_packed,proj_w,N_PROJ,INPUT_DIM);
        free(proj_w);

        /* Project all train AND test images. Keep both in memory for the
         * L1 k-NN baseline; signatures derived from these for the routed
         * k-NN path. */
        clock_t t_proj = clock();
        m4t_mtfp_t* train_proj = malloc((size_t)n_train*N_PROJ*sizeof(m4t_mtfp_t));
        m4t_mtfp_t* test_proj  = malloc((size_t)n_test *N_PROJ*sizeof(m4t_mtfp_t));
        for (int i = 0; i < n_train; i++)
            m4t_mtfp_ternary_matmul_bt(
                train_proj + (size_t)i*N_PROJ,
                x_train    + (size_t)i*INPUT_DIM,
                proj_packed, 1, INPUT_DIM, N_PROJ);
        for (int i = 0; i < n_test; i++)
            m4t_mtfp_ternary_matmul_bt(
                test_proj + (size_t)i*N_PROJ,
                x_test    + (size_t)i*INPUT_DIM,
                proj_packed, 1, INPUT_DIM, N_PROJ);
        double proj_ms = 1000.0*(double)(clock()-t_proj)/CLOCKS_PER_SEC;
        printf("  Projection phase (%d+%d images): %.0f ms\n",
               n_train, n_test, proj_ms);

        /* ── MTFP19 L1 k-NN baseline (fidelity reference) ────────────────
         * For each test: L1 over full MTFP19 projections against all n_train
         * training projections; top-k vote.  k=1, k=3, k=5 tracked. */
        int correct_l1[3] = {0,0,0};   /* k=1, k=3, k=5 */
        clock_t t_l1 = clock();
        for (int s = 0; s < n_test; s++) {
            const m4t_mtfp_t* q = test_proj + (size_t)s*N_PROJ;
            int64_t dists[MAX_K]; int labels[MAX_K];
            for (int j = 0; j < 5; j++) { dists[j] = INT64_MAX; labels[j] = -1; }

            for (int i = 0; i < n_train; i++) {
                const m4t_mtfp_t* r = train_proj + (size_t)i*N_PROJ;
                int64_t d = 0;
                for (int p = 0; p < N_PROJ; p++) {
                    int64_t x = (int64_t)q[p] - (int64_t)r[p];
                    d += (x >= 0) ? x : -x;
                }
                topk_insert_i64(dists, labels, 5, d, y_train[i]);
            }
            if (labels[0] == y_test[s]) correct_l1[0]++;
            if (majority_vote(labels, 3) == y_test[s]) correct_l1[1]++;
            if (majority_vote(labels, 5) == y_test[s]) correct_l1[2]++;

            if (s > 0 && s % 2000 == 0)
                printf("    L1 k-NN progress: %d/%d\n", s, n_test);
        }
        double l1_ms = 1000.0*(double)(clock()-t_l1)/CLOCKS_PER_SEC;

        printf("  L1 k-NN (MTFP19 projections, L1 distance):\n");
        printf("    k=1: %d/%d = %d.%02d%%\n", correct_l1[0], n_test,
               correct_l1[0]*100/n_test, (correct_l1[0]*10000/n_test)%100);
        printf("    k=3: %d/%d = %d.%02d%%\n", correct_l1[1], n_test,
               correct_l1[1]*100/n_test, (correct_l1[1]*10000/n_test)%100);
        printf("    k=5: %d/%d = %d.%02d%%\n", correct_l1[2], n_test,
               correct_l1[2]*100/n_test, (correct_l1[2]*10000/n_test)%100);
        printf("    time: %.0f ms\n", l1_ms);

        /* ── Routed k-NN (packed-trit signatures, Hamming distance) ──────
         *
         * Calibrate tau_q once from a sample of training projection |values|.
         * Since train and test projections come from the same distribution
         * (random ternary projection of MNIST pixels), the same tau produces
         * symmetric zero density on both sides by construction. */
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

        int Sp = M4T_TRIT_PACKED_BYTES(N_PROJ);

        /* Extract all training and test signatures. */
        clock_t t_ext = clock();
        uint8_t* train_sigs = calloc((size_t)n_train * Sp, 1);
        uint8_t* test_sigs  = calloc((size_t)n_test  * Sp, 1);
        int64_t* tmp_i64 = malloc((size_t)N_PROJ * sizeof(int64_t));

        long train_zero_count = 0, train_total = 0;
        for (int i = 0; i < n_train; i++) {
            for (int p = 0; p < N_PROJ; p++)
                tmp_i64[p] = (int64_t)train_proj[(size_t)i*N_PROJ + p];
            uint8_t* sig = train_sigs + (size_t)i*Sp;
            m4t_route_threshold_extract(sig, tmp_i64, tau_q, N_PROJ);
            for (int p = 0; p < N_PROJ; p++) {
                uint8_t code = (sig[p >> 2] >> ((p & 3) * 2)) & 0x3u;
                train_total++;
                if (code == 0) train_zero_count++;
            }
        }
        long test_zero_count = 0, test_total = 0;
        for (int i = 0; i < n_test; i++) {
            for (int p = 0; p < N_PROJ; p++)
                tmp_i64[p] = (int64_t)test_proj[(size_t)i*N_PROJ + p];
            uint8_t* sig = test_sigs + (size_t)i*Sp;
            m4t_route_threshold_extract(sig, tmp_i64, tau_q, N_PROJ);
            for (int p = 0; p < N_PROJ; p++) {
                uint8_t code = (sig[p >> 2] >> ((p & 3) * 2)) & 0x3u;
                test_total++;
                if (code == 0) test_zero_count++;
            }
        }
        free(tmp_i64);
        double ext_ms = 1000.0*(double)(clock()-t_ext)/CLOCKS_PER_SEC;

        /* Free training projections; we only need the test projections for
         * (already-completed) L1 and the signatures are self-contained. */
        free(train_proj);

        /* All-ones mask for popcount distance. */
        uint8_t* mask = malloc(Sp);
        memset(mask, 0xFF, Sp);

        /* Routed k-NN inference. */
        int correct_r[3] = {0,0,0};
        clock_t t_knn = clock();
        for (int s = 0; s < n_test; s++) {
            const uint8_t* q_sig = test_sigs + (size_t)s*Sp;
            int32_t dists[MAX_K]; int labels[MAX_K];
            for (int j = 0; j < 5; j++) { dists[j] = INT32_MAX; labels[j] = -1; }

            for (int i = 0; i < n_train; i++) {
                const uint8_t* r_sig = train_sigs + (size_t)i*Sp;
                int32_t d = m4t_popcount_dist(q_sig, r_sig, mask, Sp);
                topk_insert_i32(dists, labels, 5, d, y_train[i]);
            }
            if (labels[0] == y_test[s]) correct_r[0]++;
            if (majority_vote(labels, 3) == y_test[s]) correct_r[1]++;
            if (majority_vote(labels, 5) == y_test[s]) correct_r[2]++;

            if (s > 0 && s % 2000 == 0)
                printf("    Routed k-NN progress: %d/%d\n", s, n_test);
        }
        double knn_ms = 1000.0*(double)(clock()-t_knn)/CLOCKS_PER_SEC;

        int train_pct100 = (int)((100 * train_zero_count * 100) / train_total);
        int test_pct100  = (int)((100 * test_zero_count  * 100) / test_total);
        printf("  Routed k-NN (Hamming over packed-trit signatures):\n");
        printf("    tau_q = %lld  target density = %.2f  train %%zero = %d.%02d%%  test %%zero = %d.%02d%%\n",
               (long long)tau_q, DENSITY,
               train_pct100/100, train_pct100%100,
               test_pct100/100,  test_pct100%100);
        printf("    k=1: %d/%d = %d.%02d%%\n", correct_r[0], n_test,
               correct_r[0]*100/n_test, (correct_r[0]*10000/n_test)%100);
        printf("    k=3: %d/%d = %d.%02d%%\n", correct_r[1], n_test,
               correct_r[1]*100/n_test, (correct_r[1]*10000/n_test)%100);
        printf("    k=5: %d/%d = %d.%02d%%\n", correct_r[2], n_test,
               correct_r[2]*100/n_test, (correct_r[2]*10000/n_test)%100);
        printf("    extract time: %.0f ms   inference time: %.0f ms\n\n", ext_ms, knn_ms);

        free(train_sigs); free(test_sigs); free(mask);
        free(test_proj); free(proj_packed);
    }

    printf("Zero float. Zero gradients. Real Trit Lattice LSH, fully routed.\n");
    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
