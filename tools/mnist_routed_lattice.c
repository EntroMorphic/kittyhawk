/*
 * mnist_routed_lattice.c — Trit Lattice classification, genuinely routed.
 *
 * Same pipeline prefix as mnist_trit_lattice.c (random ternary projection
 * via m4t_mtfp_ternary_matmul_bt), but the CLASSIFICATION step uses the
 * m4t_route primitives instead of scalar L1 over MTFP mantissas:
 *
 *   projection → sign_extract → packed-trit query signature
 *   centroid   → mean-subtract → sign_extract → packed-trit class signature
 *   classify   via m4t_route_distance_batch (popcount-Hamming) + topk_abs
 *
 * Side-by-side with the L1-over-mantissa path so the comparison is honest:
 * same projections, same training, same test set, different decision step.
 *
 * Usage: ./mnist_routed_lattice <mnist_dir>
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

/* ── Data loaders (identical to mnist_trit_lattice.c) ─────────────────── */

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

/* Same RNG seed as mnist_trit_lattice.c → identical projection matrices. */
static uint32_t rng_s[4];
static uint32_t rng_next(void) {
    uint32_t result=rng_s[0]+rng_s[3];
    uint32_t t=rng_s[1]<<9;
    rng_s[2]^=rng_s[0]; rng_s[3]^=rng_s[1];
    rng_s[1]^=rng_s[2]; rng_s[0]^=rng_s[3];
    rng_s[2]^=t; rng_s[3]=(rng_s[3]<<11)|(rng_s[3]>>21);
    return result;
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

    printf("Trit Lattice ROUTED — MNIST (zero float, full routing surface)\n");
    printf("Loaded %d train, %d test\n", n_train, n_test);
    printf("Routing primitives exercised: threshold_extract, distance_batch, topk_abs.\n\n");

    int class_counts[N_CLASSES]; memset(class_counts,0,sizeof(class_counts));
    for(int i=0;i<n_train;i++) class_counts[y_train[i]]++;

    int proj_sizes[]={256, 512, 1024, 2048};
    int n_sizes=4;

    for(int si=0;si<n_sizes;si++){
        int N_PROJ=proj_sizes[si];
        rng_s[0]=42; rng_s[1]=123; rng_s[2]=456; rng_s[3]=789;

        printf("=== N_PROJ = %d ===\n", N_PROJ);

        /* Build ternary projection matrix (identical seed → identical
         * matrix as mnist_trit_lattice.c for a direct A/B). */
        m4t_trit_t* proj_w=malloc((size_t)N_PROJ*INPUT_DIM);
        for(int i=0;i<N_PROJ*INPUT_DIM;i++){
            uint32_t r=rng_next()%3;
            proj_w[i]=(r==0)?-1:(r==1)?0:1;
        }
        int proj_Dp=M4T_TRIT_PACKED_BYTES(INPUT_DIM);
        uint8_t* proj_packed=malloc((size_t)N_PROJ*proj_Dp);
        m4t_pack_trits_rowmajor(proj_packed,proj_w,N_PROJ,INPUT_DIM);

        /* Project training images. */
        m4t_mtfp_t* train_proj=malloc((size_t)n_train*N_PROJ*sizeof(m4t_mtfp_t));
        for(int i=0;i<n_train;i++)
            m4t_mtfp_ternary_matmul_bt(train_proj+(size_t)i*N_PROJ,
                x_train+(size_t)i*INPUT_DIM,proj_packed,1,INPUT_DIM,N_PROJ);

        /* Per-class centroid (MTFP mantissas). */
        int64_t* class_sums=calloc((size_t)N_CLASSES*N_PROJ,sizeof(int64_t));
        for(int i=0;i<n_train;i++){
            int c=y_train[i];
            for(int p=0;p<N_PROJ;p++)
                class_sums[(size_t)c*N_PROJ+p]+=(int64_t)train_proj[(size_t)i*N_PROJ+p];
        }
        m4t_mtfp_t* centroids=malloc((size_t)N_CLASSES*N_PROJ*sizeof(m4t_mtfp_t));
        for(int c=0;c<N_CLASSES;c++)
            for(int p=0;p<N_PROJ;p++)
                centroids[(size_t)c*N_PROJ+p]=(m4t_mtfp_t)(class_sums[(size_t)c*N_PROJ+p]/class_counts[c]);

        /* Build per-class packed-trit signatures:
         *   sig[c][p] = sign(centroid[c][p] - mean_over_classes(centroid[*,p]))
         * Mean-subtract normalizes out dims where every class is consistently
         * positive or negative (they contribute zero discrimination). */
        int Sp=M4T_TRIT_PACKED_BYTES(N_PROJ);
        uint8_t* class_sigs=calloc((size_t)N_CLASSES*Sp,1);
        {
            int64_t* dim_mean=calloc((size_t)N_PROJ,sizeof(int64_t));
            for(int c=0;c<N_CLASSES;c++)
                for(int p=0;p<N_PROJ;p++)
                    dim_mean[p]+=(int64_t)centroids[(size_t)c*N_PROJ+p];
            for(int p=0;p<N_PROJ;p++) dim_mean[p]/=N_CLASSES;

            int64_t* diff=malloc((size_t)N_PROJ*sizeof(int64_t));
            for(int c=0;c<N_CLASSES;c++){
                for(int p=0;p<N_PROJ;p++)
                    diff[p]=(int64_t)centroids[(size_t)c*N_PROJ+p]-dim_mean[p];
                /* tau=0 here is the sign-only degenerate of threshold_extract.
                 * The centroid - mean inputs are integer differences that can
                 * hit exact zero, so emission coverage is borderline-sanctioned
                 * on this deployment. Switching to tau>0 is a candidate for
                 * follow-up experimentation. */
                m4t_route_threshold_extract(class_sigs+(size_t)c*Sp,diff,0,N_PROJ);
            }
            free(diff); free(dim_mean);
        }

        /* Active-bit mask: all trits participate. */
        uint8_t* mask=malloc(Sp); memset(mask,0xFF,Sp);

        /* ── Inference loop: L1-over-mantissa vs routed-popcount, same imgs ── */

        m4t_mtfp_t* test_proj_buf=malloc((size_t)N_PROJ*sizeof(m4t_mtfp_t));
        int64_t* query_i64=malloc((size_t)N_PROJ*sizeof(int64_t));
        uint8_t* query_sig=malloc(Sp);
        int32_t dists[N_CLASSES];

        int correct_l1=0, correct_routed=0;

        /* Time projection step (shared) */
        clock_t t0=clock();
        for(int s=0;s<n_test;s++){
            const m4t_mtfp_t* img=x_test+(size_t)s*INPUT_DIM;
            m4t_mtfp_ternary_matmul_bt(test_proj_buf,img,proj_packed,1,INPUT_DIM,N_PROJ);

            /* L1-over-mantissa classification (dense decision step). */
            int pred_l1=0; int64_t best_l1=INT64_MAX;
            for(int c=0;c<N_CLASSES;c++){
                int64_t d=0;
                const m4t_mtfp_t* cc=centroids+(size_t)c*N_PROJ;
                for(int p=0;p<N_PROJ;p++){
                    int64_t x=(int64_t)test_proj_buf[p]-(int64_t)cc[p];
                    d+=(x>=0)?x:-x;
                }
                if(d<best_l1){best_l1=d; pred_l1=c;}
            }
            if(pred_l1==y_test[s]) correct_l1++;

            /* Routed classification: threshold_extract(tau=0) query → popcount
             * distance → topk_abs picks the class with the largest affinity
             * (= smallest distance, mapped via score = MAX - dist).
             *
             * Emission-coverage note: tau=0 on MTFP19 projection outputs
             * produces a sign-only classification in practice (zero state is
             * measure-zero for continuous-valued projections). This deployment
             * does NOT pass emission coverage per M4T_SUBSTRATE §18; the 58%
             * accuracy measured in journal/fully_routed_mnist.md is the
             * resulting binary-shape limit. A tau>0 variant is the candidate
             * follow-up experiment. */
            for(int p=0;p<N_PROJ;p++) query_i64[p]=(int64_t)test_proj_buf[p];
            m4t_route_threshold_extract(query_sig,query_i64,0,N_PROJ);
            m4t_route_distance_batch(dists,query_sig,class_sigs,mask,N_CLASSES,N_PROJ);

            /* Map distance → affinity score so topk_abs picks the minimum. */
            int32_t scores[N_CLASSES];
            int32_t max_dist=2*N_PROJ;  /* 2 bits per trit, all may differ */
            for(int c=0;c<N_CLASSES;c++) scores[c]=max_dist-dists[c];

            m4t_route_decision_t decision;
            m4t_route_topk_abs(&decision,scores,N_CLASSES,1);
            int pred_r = decision.tile_idx;
            if(pred_r<0) pred_r=0;  /* sentinel fallback (no nonzero score) */
            if(pred_r==y_test[s]) correct_routed++;
        }
        clock_t t1=clock();
        double total_ms=1000.0*(double)(t1-t0)/CLOCKS_PER_SEC;

        printf("  L1-over-mantissa (dense decision):       %d/%d = %d.%02d%%\n",
               correct_l1,n_test,correct_l1*100/n_test,(correct_l1*10000/n_test)%100);
        printf("  Routed (threshold_extract tau=0 + VCNT): %d/%d = %d.%02d%%\n",
               correct_routed,n_test,correct_routed*100/n_test,(correct_routed*10000/n_test)%100);
        printf("  Inference (both paths, %d images):  %.0f ms\n\n",
               n_test, total_ms);

        free(test_proj_buf); free(query_i64); free(query_sig);
        free(class_sigs); free(mask); free(centroids); free(class_sums); free(train_proj);
        free(proj_w); free(proj_packed);
    }

    printf("Zero float. Zero gradients. Full routing surface exercised.\n");
    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
