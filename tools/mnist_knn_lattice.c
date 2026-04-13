/*
 * mnist_knn_lattice.c — k-NN on the trit lattice. Zero float.
 *
 * Experiments: projection width, distance metric, voting scheme.
 * Includes confusion matrix for error analysis.
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

static uint32_t rng_s[4];
static uint32_t rng_next(void) {
    uint32_t result=rng_s[0]+rng_s[3];
    uint32_t t=rng_s[1]<<9;
    rng_s[2]^=rng_s[0]; rng_s[3]^=rng_s[1];
    rng_s[1]^=rng_s[2]; rng_s[0]^=rng_s[3];
    rng_s[2]^=t; rng_s[3]=(rng_s[3]<<11)|(rng_s[3]>>21);
    return result;
}

static void run_knn(const char* label, int K,
    const m4t_mtfp_t* test_proj, int n_test, const int* y_test,
    const m4t_mtfp_t* train_proj, int n_train, const int* y_train,
    int N_PROJ, int weighted, int use_l2)
{
    int correct = 0;
    int confusion[N_CLASSES][N_CLASSES];
    memset(confusion, 0, sizeof(confusion));

    for (int s = 0; s < n_test; s++) {
        const m4t_mtfp_t* timg = test_proj + (size_t)s * N_PROJ;

        int64_t knn_dist[MAX_K];
        int knn_label[MAX_K];
        for (int j = 0; j < K; j++) { knn_dist[j] = INT64_MAX; knn_label[j] = -1; }

        for (int i = 0; i < n_train; i++) {
            const m4t_mtfp_t* tref = train_proj + (size_t)i * N_PROJ;
            int64_t dist = 0;

            if (use_l2) {
                for (int p = 0; p < N_PROJ; p++) {
                    int64_t d = (int64_t)timg[p] - (int64_t)tref[p];
                    dist += d * d;
                }
            } else {
                for (int p = 0; p < N_PROJ; p++) {
                    int64_t d = (int64_t)timg[p] - (int64_t)tref[p];
                    dist += (d >= 0) ? d : -d;
                }
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

        int pred;
        if (weighted) {
            /* Distance-weighted voting: weight = 1 / (dist + 1) */
            int64_t votes[N_CLASSES];
            memset(votes, 0, sizeof(votes));
            for (int j = 0; j < K; j++) {
                if (knn_label[j] >= 0) {
                    int64_t w = 1000000 / (knn_dist[j] / 1000 + 1);
                    votes[knn_label[j]] += w;
                }
            }
            pred = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (votes[c] > votes[pred]) pred = c;
        } else {
            int votes[N_CLASSES];
            memset(votes, 0, sizeof(votes));
            for (int j = 0; j < K; j++)
                if (knn_label[j] >= 0) votes[knn_label[j]]++;
            pred = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (votes[c] > votes[pred]) pred = c;
        }

        if (pred == y_test[s]) correct++;
        confusion[y_test[s]][pred]++;
    }

    printf("  %s: %d/%d = %d.%02d%%\n",
           label, correct, n_test,
           correct*100/n_test, (correct*10000/n_test)%100);

    /* Print confusion matrix for the best run */
    if (correct > 9600) {
        printf("  Confusion matrix (rows=true, cols=predicted):\n");
        printf("       ");
        for (int c = 0; c < N_CLASSES; c++) printf(" %4d", c);
        printf("  | err\n");
        for (int r = 0; r < N_CLASSES; r++) {
            printf("    %d: ", r);
            int row_total = 0, row_err = 0;
            for (int c = 0; c < N_CLASSES; c++) {
                printf(" %4d", confusion[r][c]);
                row_total += confusion[r][c];
                if (c != r) row_err += confusion[r][c];
            }
            printf("  | %d/%d\n", row_err, row_total);
        }
        printf("\n");
    }
}

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

    printf("k-NN Atomics — MNIST (zero float)\n");
    printf("Loaded %d train, %d test\n\n", n_train, n_test);

    int proj_sizes[] = {256, 512};
    int n_proj_count = 2;

    for (int pi = 0; pi < n_proj_count; pi++) {
        int N_PROJ = proj_sizes[pi];
        rng_s[0]=42; rng_s[1]=123; rng_s[2]=456; rng_s[3]=789;

        printf("=== N_PROJ = %d ===\n", N_PROJ);

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

        printf("  Projected. Running k-NN variants...\n");

        /* L1, majority vote */
        run_knn("k=3 L1 majority", 3, test_proj, n_test, y_test,
                train_proj, n_train, y_train, N_PROJ, 0, 0);
        run_knn("k=5 L1 majority", 5, test_proj, n_test, y_test,
                train_proj, n_train, y_train, N_PROJ, 0, 0);

        /* L1, distance-weighted */
        run_knn("k=5 L1 weighted", 5, test_proj, n_test, y_test,
                train_proj, n_train, y_train, N_PROJ, 1, 0);

        /* L2, majority vote */
        run_knn("k=5 L2 majority", 5, test_proj, n_test, y_test,
                train_proj, n_train, y_train, N_PROJ, 0, 1);

        /* L2, distance-weighted */
        run_knn("k=5 L2 weighted", 5, test_proj, n_test, y_test,
                train_proj, n_train, y_train, N_PROJ, 1, 1);

        printf("\n");
        free(proj_w); free(proj_packed); free(train_proj); free(test_proj);
    }

    printf("Zero float. Zero gradients. Pure lattice geometry.\n");
    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
