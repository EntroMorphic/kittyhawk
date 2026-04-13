/*
 * mnist_knn_lattice.c — k-NN on the trit lattice with deskewing. Zero float.
 *
 * Steps:
 *   1. Load images as MTFP19 cells (integer)
 *   2. Deskew via integer image moments + horizontal shear
 *   3. Project via random ternary matmul
 *   4. k-NN in pixel-space, projection-space, and combined
 *
 * Usage: ./mnist_knn_lattice <mnist_dir>
 */

#include "m4t_types.h"
#include "m4t_mtfp.h"
#include "m4t_trit_pack.h"
#include "m4t_ternary_matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define INPUT_DIM 784
#define IMG_W 28
#define IMG_H 28
#define N_CLASSES 10
#define MAX_K 7

static uint32_t read_u32_be(FILE* f) {
    uint8_t b[4]; fread(b,1,4,f);
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];
}
static m4t_mtfp_t* load_images_mtfp(const char* path, int* n) {
    FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",path);exit(1);}
    read_u32_be(f); *n=(int)read_u32_be(f);
    int rows=(int)read_u32_be(f),cols=(int)read_u32_be(f);
    (void)rows; (void)cols;
    size_t total=(size_t)(*n)*INPUT_DIM;
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

static uint32_t rng_s[4];
static uint32_t rng_next(void) {
    uint32_t result=rng_s[0]+rng_s[3];
    uint32_t t=rng_s[1]<<9;
    rng_s[2]^=rng_s[0]; rng_s[3]^=rng_s[1];
    rng_s[1]^=rng_s[2]; rng_s[0]^=rng_s[3];
    rng_s[2]^=t; rng_s[3]=(rng_s[3]<<11)|(rng_s[3]>>21);
    return result;
}

/* ── Integer deskewing ─────────────────────────────────────────────────── */

static void deskew_image(m4t_mtfp_t* dst, const m4t_mtfp_t* src) {
    /* Compute intensity-weighted moments — all integer. */
    int64_t sum_p = 0, sum_xp = 0, sum_yp = 0;
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            int64_t p = (int64_t)src[y * IMG_W + x];
            sum_p += p;
            sum_xp += (int64_t)x * p;
            sum_yp += (int64_t)y * p;
        }
    }

    if (sum_p == 0) { memcpy(dst, src, INPUT_DIM * sizeof(m4t_mtfp_t)); return; }

    /* Second moment: Mxy and Myy, scaled by sum_p to avoid division.
     * We compute (x - cx)*sum_p = x*sum_p - sum_xp per pixel. */
    int64_t Mxy = 0, Myy = 0;
    for (int y = 0; y < IMG_H; y++) {
        int64_t dy = (int64_t)y * sum_p - sum_yp;
        for (int x = 0; x < IMG_W; x++) {
            int64_t p = (int64_t)src[y * IMG_W + x];
            int64_t dx = (int64_t)x * sum_p - sum_xp;
            Mxy += dx * dy / sum_p * p / sum_p;
            Myy += dy * dy / sum_p * p / sum_p;
        }
    }

    /* Shear: for each row y, shift by -(y - cy) * Mxy / Myy pixels.
     * Equivalent: shift = -(y * sum_p - sum_yp) * Mxy / (Myy * sum_p)
     * We compute in integer with rounding. */
    memset(dst, 0, INPUT_DIM * sizeof(m4t_mtfp_t));

    for (int y = 0; y < IMG_H; y++) {
        int32_t shift = 0;
        if (Myy != 0) {
            int64_t dy = (int64_t)y * sum_p - sum_yp;
            shift = (int32_t)(-(dy * Mxy) / (Myy * sum_p));
        }

        for (int x = 0; x < IMG_W; x++) {
            int nx = x + shift;
            if (nx >= 0 && nx < IMG_W)
                dst[y * IMG_W + nx] = src[y * IMG_W + x];
        }
    }
}

static void deskew_all(m4t_mtfp_t* images, int n) {
    m4t_mtfp_t buf[INPUT_DIM];
    for (int i = 0; i < n; i++) {
        m4t_mtfp_t* img = images + (size_t)i * INPUT_DIM;
        deskew_image(buf, img);
        memcpy(img, buf, INPUT_DIM * sizeof(m4t_mtfp_t));
    }
}

/* ── k-NN engine ───────────────────────────────────────────────────────── */

static int knn_classify(
    const m4t_mtfp_t* query, int feat_dim,
    const m4t_mtfp_t* train_feats, const int* y_train, int n_train,
    int K)
{
    int64_t knn_dist[MAX_K];
    int knn_label[MAX_K];
    for (int j = 0; j < K; j++) { knn_dist[j] = INT64_MAX; knn_label[j] = -1; }

    for (int i = 0; i < n_train; i++) {
        const m4t_mtfp_t* ref = train_feats + (size_t)i * feat_dim;
        int64_t dist = 0;
        for (int p = 0; p < feat_dim; p++) {
            int64_t d = (int64_t)query[p] - (int64_t)ref[p];
            dist += d * d;  /* L2 squared */
        }

        if (dist < knn_dist[K-1]) {
            knn_dist[K-1] = dist;
            knn_label[K-1] = y_train[i];
            for (int j = K-2; j >= 0; j--) {
                if (knn_dist[j+1] < knn_dist[j]) {
                    int64_t td=knn_dist[j]; knn_dist[j]=knn_dist[j+1]; knn_dist[j+1]=td;
                    int tl=knn_label[j]; knn_label[j]=knn_label[j+1]; knn_label[j+1]=tl;
                } else break;
            }
        }
    }

    int votes[N_CLASSES];
    memset(votes, 0, sizeof(votes));
    for (int j = 0; j < K; j++)
        if (knn_label[j] >= 0) votes[knn_label[j]]++;
    int pred = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (votes[c] > votes[pred]) pred = c;
    return pred;
}

static void run_experiment(const char* label,
    const m4t_mtfp_t* test_feats, int n_test, const int* y_test,
    const m4t_mtfp_t* train_feats, int n_train, const int* y_train,
    int feat_dim, int K)
{
    int correct = 0;
    for (int s = 0; s < n_test; s++) {
        int pred = knn_classify(test_feats + (size_t)s * feat_dim, feat_dim,
                                train_feats, y_train, n_train, K);
        if (pred == y_test[s]) correct++;
        if (s > 0 && s % 2000 == 0)
            printf("    %d/%d — %d.%02d%%\n", s, n_test,
                   correct*100/s, (correct*10000/s)%100);
    }
    printf("  %s: %d/%d = %d.%02d%%\n\n", label, correct, n_test,
           correct*100/n_test, (correct*10000/n_test)%100);
}

/* ── Main ──────────────────────────────────────────────────────────────── */

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

    printf("k-NN on the Trit Lattice — MNIST (zero float)\n");
    printf("Loaded %d train, %d test\n\n", n_train, n_test);

    int K = 5;
    int N_PROJ = 512;

    /* ── Step 1: Pixel-space k-NN (no projection, no deskewing) ────────── */

    printf("=== Step 1: Pixel-space k-NN (784-dim, L2, k=%d) ===\n", K);
    run_experiment("pixel-space raw", x_test, n_test, y_test,
                   x_train, n_train, y_train, INPUT_DIM, K);

    /* ── Step 2: Deskew, then pixel-space k-NN ─────────────────────────── */

    printf("=== Step 2: Deskewing all images (integer moments + shear) ===\n");
    deskew_all(x_train, n_train);
    deskew_all(x_test, n_test);
    printf("  Deskewed %d + %d images\n\n", n_train, n_test);

    printf("=== Step 2b: Pixel-space k-NN after deskewing ===\n");
    run_experiment("pixel-space deskewed", x_test, n_test, y_test,
                   x_train, n_train, y_train, INPUT_DIM, K);

    /* ── Step 3: Project deskewed images, then k-NN ────────────────────── */

    printf("=== Step 3: Random ternary projection (%d dims) + k-NN ===\n", N_PROJ);
    rng_s[0]=42; rng_s[1]=123; rng_s[2]=456; rng_s[3]=789;

    m4t_trit_t* proj_w = malloc((size_t)N_PROJ * INPUT_DIM);
    for (int i = 0; i < N_PROJ * INPUT_DIM; i++) {
        uint32_t r = rng_next() % 3;
        proj_w[i] = (r==0) ? -1 : (r==1) ? 0 : 1;
    }
    int proj_Dp = M4T_TRIT_PACKED_BYTES(INPUT_DIM);
    uint8_t* proj_packed = malloc((size_t)N_PROJ * proj_Dp);
    m4t_pack_trits_rowmajor(proj_packed, proj_w, N_PROJ, INPUT_DIM);

    m4t_mtfp_t* train_proj = malloc((size_t)n_train * N_PROJ * sizeof(m4t_mtfp_t));
    for (int i = 0; i < n_train; i++)
        m4t_mtfp_ternary_matmul_bt(train_proj + (size_t)i * N_PROJ,
            x_train + (size_t)i * INPUT_DIM, proj_packed, 1, INPUT_DIM, N_PROJ);

    m4t_mtfp_t* test_proj = malloc((size_t)n_test * N_PROJ * sizeof(m4t_mtfp_t));
    for (int i = 0; i < n_test; i++)
        m4t_mtfp_ternary_matmul_bt(test_proj + (size_t)i * N_PROJ,
            x_test + (size_t)i * INPUT_DIM, proj_packed, 1, INPUT_DIM, N_PROJ);

    run_experiment("proj-space deskewed", test_proj, n_test, y_test,
                   train_proj, n_train, y_train, N_PROJ, K);

    /* ── Step 4: Combined pixel + projection ───────────────────────────── */

    printf("=== Step 4: Combined pixel+projection (%d dims) + k-NN ===\n",
           INPUT_DIM + N_PROJ);

    int combined_dim = INPUT_DIM + N_PROJ;
    m4t_mtfp_t* train_combined = malloc((size_t)n_train * combined_dim * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* test_combined = malloc((size_t)n_test * combined_dim * sizeof(m4t_mtfp_t));

    for (int i = 0; i < n_train; i++) {
        memcpy(train_combined + (size_t)i * combined_dim,
               x_train + (size_t)i * INPUT_DIM, INPUT_DIM * sizeof(m4t_mtfp_t));
        memcpy(train_combined + (size_t)i * combined_dim + INPUT_DIM,
               train_proj + (size_t)i * N_PROJ, N_PROJ * sizeof(m4t_mtfp_t));
    }
    for (int i = 0; i < n_test; i++) {
        memcpy(test_combined + (size_t)i * combined_dim,
               x_test + (size_t)i * INPUT_DIM, INPUT_DIM * sizeof(m4t_mtfp_t));
        memcpy(test_combined + (size_t)i * combined_dim + INPUT_DIM,
               test_proj + (size_t)i * N_PROJ, N_PROJ * sizeof(m4t_mtfp_t));
    }

    run_experiment("combined deskewed", test_combined, n_test, y_test,
                   train_combined, n_train, y_train, combined_dim, K);

    printf("Zero float. Zero gradients. Pure lattice geometry.\n");

    free(x_train); free(y_train); free(x_test); free(y_test);
    free(proj_w); free(proj_packed); free(train_proj); free(test_proj);
    free(train_combined); free(test_combined);
    return 0;
}
