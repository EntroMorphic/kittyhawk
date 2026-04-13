/*
 * mnist_knn_lattice.c — k-NN on the trit lattice. Zero float.
 *
 * Projects all images via random ternary matmul, then classifies by
 * k-nearest-neighbors in projection space using L1 distance.
 *
 * This is the ceiling test: if k-NN gets 95%+, the representation is
 * already good enough and the gap to 81% is entirely the classifier.
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

static uint32_t rng_s[4]={42,123,456,789};
static uint32_t rng_next(void) {
    uint32_t result=rng_s[0]+rng_s[3];
    uint32_t t=rng_s[1]<<9;
    rng_s[2]^=rng_s[0]; rng_s[3]^=rng_s[1];
    rng_s[1]^=rng_s[2]; rng_s[0]^=rng_s[3];
    rng_s[2]^=t; rng_s[3]=(rng_s[3]<<11)|(rng_s[3]>>21);
    return result;
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

    printf("k-NN on the Trit Lattice — MNIST (zero float)\n");
    printf("Loaded %d train, %d test\n\n", n_train, n_test);

    /* Generate random ternary projection */
    int N_PROJ = 256;
    printf("Projecting to %d dims via random ternary matmul...\n", N_PROJ);

    m4t_trit_t* proj_w = malloc((size_t)N_PROJ * INPUT_DIM);
    for (int i = 0; i < N_PROJ * INPUT_DIM; i++) {
        uint32_t r = rng_next() % 3;
        proj_w[i] = (r==0) ? -1 : (r==1) ? 0 : 1;
    }
    int proj_Dp = M4T_TRIT_PACKED_BYTES(INPUT_DIM);
    uint8_t* proj_packed = malloc((size_t)N_PROJ * proj_Dp);
    m4t_pack_trits_rowmajor(proj_packed, proj_w, N_PROJ, INPUT_DIM);

    /* Project all training images */
    printf("Projecting %d training images...\n", n_train);
    m4t_mtfp_t* train_proj = malloc((size_t)n_train * N_PROJ * sizeof(m4t_mtfp_t));
    for (int i = 0; i < n_train; i++)
        m4t_mtfp_ternary_matmul_bt(train_proj + (size_t)i * N_PROJ,
            x_train + (size_t)i * INPUT_DIM, proj_packed, 1, INPUT_DIM, N_PROJ);

    /* Project all test images */
    printf("Projecting %d test images...\n", n_test);
    m4t_mtfp_t* test_proj = malloc((size_t)n_test * N_PROJ * sizeof(m4t_mtfp_t));
    for (int i = 0; i < n_test; i++)
        m4t_mtfp_ternary_matmul_bt(test_proj + (size_t)i * N_PROJ,
            x_test + (size_t)i * INPUT_DIM, proj_packed, 1, INPUT_DIM, N_PROJ);

    /* k-NN for various k values */
    int k_vals[] = {1, 3, 5, 7};
    int n_k = 4;

    printf("\nRunning k-NN (L1 distance in %d-dim projection space)...\n", N_PROJ);
    printf("This is %d × %d × %d = ~%.0f billion integer ops.\n\n",
           n_test, n_train, N_PROJ,
           (double)n_test * n_train * N_PROJ / 1e9);

    for (int ki = 0; ki < n_k; ki++) {
        int K = k_vals[ki];
        int correct = 0;

        for (int s = 0; s < n_test; s++) {
            const m4t_mtfp_t* timg = test_proj + (size_t)s * N_PROJ;

            /* Find K nearest neighbors by L1 distance */
            /* Keep a small sorted list of (distance, label) pairs */
            int64_t knn_dist[7];  /* max K=7 */
            int knn_label[7];
            for (int j = 0; j < K; j++) { knn_dist[j] = INT64_MAX; knn_label[j] = -1; }

            for (int i = 0; i < n_train; i++) {
                const m4t_mtfp_t* tref = train_proj + (size_t)i * N_PROJ;
                int64_t dist = 0;
                for (int p = 0; p < N_PROJ; p++) {
                    int64_t d = (int64_t)timg[p] - (int64_t)tref[p];
                    dist += (d >= 0) ? d : -d;
                }

                /* Insert into sorted knn list if closer than the farthest */
                if (dist < knn_dist[K-1]) {
                    knn_dist[K-1] = dist;
                    knn_label[K-1] = y_train[i];
                    /* Bubble up */
                    for (int j = K-2; j >= 0; j--) {
                        if (knn_dist[j+1] < knn_dist[j]) {
                            int64_t td = knn_dist[j]; knn_dist[j] = knn_dist[j+1]; knn_dist[j+1] = td;
                            int tl = knn_label[j]; knn_label[j] = knn_label[j+1]; knn_label[j+1] = tl;
                        } else break;
                    }
                }
            }

            /* Majority vote */
            int votes[N_CLASSES];
            memset(votes, 0, sizeof(votes));
            for (int j = 0; j < K; j++)
                if (knn_label[j] >= 0) votes[knn_label[j]]++;
            int pred = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (votes[c] > votes[pred]) pred = c;

            if (pred == y_test[s]) correct++;

            if (s > 0 && s % 1000 == 0)
                printf("  k=%d: %d/%d — running: %d.%02d%%\n",
                       K, s, n_test, correct*100/s, (correct*10000/s)%100);
        }

        printf("k=%d: %d/%d = %d.%02d%%\n\n",
               K, correct, n_test,
               correct*100/n_test, (correct*10000/n_test)%100);
    }

    printf("Zero float. Zero gradients. Pure lattice geometry.\n");
    free(x_train); free(y_train); free(x_test); free(y_test);
    free(proj_w); free(proj_packed); free(train_proj); free(test_proj);
    return 0;
}
