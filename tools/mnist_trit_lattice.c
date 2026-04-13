/*
 * mnist_trit_lattice.c — MNIST via Trit Lattice LSH. Zero float.
 *
 * Two-stage: random ternary projection → L1 coarse routing → pixel-space
 * L1 refinement among top-K candidates.
 *
 * Usage: ./mnist_trit_lattice <mnist_dir>
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

static uint32_t rng_s[4];
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

    printf("Trit Lattice LSH — Two-Stage MNIST (zero float)\n");
    printf("Loaded %d train, %d test\n\n", n_train, n_test);

    /* Class counts and pixel-space centroids */
    int class_counts[N_CLASSES];
    memset(class_counts,0,sizeof(class_counts));
    for(int i=0;i<n_train;i++) class_counts[y_train[i]]++;

    int64_t pixel_sums[N_CLASSES][INPUT_DIM];
    memset(pixel_sums,0,sizeof(pixel_sums));
    for(int i=0;i<n_train;i++){
        int c=y_train[i];
        const m4t_mtfp_t* img=x_train+(size_t)i*INPUT_DIM;
        for(int d=0;d<INPUT_DIM;d++) pixel_sums[c][d]+=(int64_t)img[d];
    }
    m4t_mtfp_t pixel_centroids[N_CLASSES][INPUT_DIM];
    for(int c=0;c<N_CLASSES;c++)
        for(int d=0;d<INPUT_DIM;d++)
            pixel_centroids[c][d]=(m4t_mtfp_t)(pixel_sums[c][d]/class_counts[c]);

    /* ── Pixel-space L1 baseline ───────────────────────────────────────── */
    {
        printf("=== Pixel-space L1 nearest centroid (no projection) ===\n");
        int correct=0;
        for(int s=0;s<n_test;s++){
            const m4t_mtfp_t* img=x_test+(size_t)s*INPUT_DIM;
            int pred=0; int64_t best=INT64_MAX;
            for(int c=0;c<N_CLASSES;c++){
                int64_t d=0;
                for(int p=0;p<INPUT_DIM;p++){
                    int64_t x=(int64_t)img[p]-(int64_t)pixel_centroids[c][p];
                    d+=(x>=0)?x:-x;
                }
                if(d<best){best=d;pred=c;}
            }
            if(pred==y_test[s]) correct++;
        }
        printf("  %d/%d = %d.%02d%%\n\n", correct, n_test,
               correct*100/n_test, (correct*10000/n_test)%100);
    }

    /* ── Projection-space experiments ──────────────────────────────────── */

    int proj_sizes[]={256, 512, 1024, 2048};
    int n_sizes=4;

    for(int si=0;si<n_sizes;si++){
        int N_PROJ=proj_sizes[si];
        rng_s[0]=42; rng_s[1]=123; rng_s[2]=456; rng_s[3]=789;

        printf("=== N_PROJ = %d ===\n", N_PROJ);

        /* Generate random ternary projections */
        m4t_trit_t* proj_w=malloc((size_t)N_PROJ*INPUT_DIM);
        for(int i=0;i<N_PROJ*INPUT_DIM;i++){
            uint32_t r=rng_next()%3;
            proj_w[i]=(r==0)?-1:(r==1)?0:1;
        }
        int proj_Dp=M4T_TRIT_PACKED_BYTES(INPUT_DIM);
        uint8_t* proj_packed=malloc((size_t)N_PROJ*proj_Dp);
        m4t_pack_trits_rowmajor(proj_packed,proj_w,N_PROJ,INPUT_DIM);

        /* Project training images */
        m4t_mtfp_t* train_proj=malloc((size_t)n_train*N_PROJ*sizeof(m4t_mtfp_t));
        for(int i=0;i<n_train;i++)
            m4t_mtfp_ternary_matmul_bt(train_proj+(size_t)i*N_PROJ,
                x_train+(size_t)i*INPUT_DIM,proj_packed,1,INPUT_DIM,N_PROJ);

        /* Class centroids in projection space */
        int64_t* proj_sums=calloc((size_t)N_CLASSES*N_PROJ,sizeof(int64_t));
        for(int i=0;i<n_train;i++){
            int c=y_train[i];
            for(int p=0;p<N_PROJ;p++)
                proj_sums[(size_t)c*N_PROJ+p]+=(int64_t)train_proj[(size_t)i*N_PROJ+p];
        }
        int32_t* proj_cents=malloc((size_t)N_CLASSES*N_PROJ*sizeof(int32_t));
        for(int c=0;c<N_CLASSES;c++)
            for(int p=0;p<N_PROJ;p++)
                proj_cents[(size_t)c*N_PROJ+p]=(int32_t)(proj_sums[(size_t)c*N_PROJ+p]/class_counts[c]);

        /* Inference */
        int correct_l1=0, correct_refine3=0, correct_refine5=0;

        for(int s=0;s<n_test;s++){
            const m4t_mtfp_t* img=x_test+(size_t)s*INPUT_DIM;
            m4t_mtfp_t* tproj=train_proj; /* reuse buffer */
            m4t_mtfp_t test_proj_buf[4096];
            m4t_mtfp_ternary_matmul_bt(test_proj_buf,img,proj_packed,1,INPUT_DIM,N_PROJ);

            /* Stage 1: L1 in projection space */
            int64_t dists_proj[N_CLASSES];
            for(int c=0;c<N_CLASSES;c++){
                int64_t d=0;
                const int32_t* cc=proj_cents+(size_t)c*N_PROJ;
                for(int p=0;p<N_PROJ;p++){
                    int64_t x=(int64_t)test_proj_buf[p]-(int64_t)cc[p];
                    d+=(x>=0)?x:-x;
                }
                dists_proj[c]=d;
            }

            /* Rank */
            int ranked[N_CLASSES];
            for(int c=0;c<N_CLASSES;c++) ranked[c]=c;
            for(int i=1;i<N_CLASSES;i++){
                int key=ranked[i]; int64_t kd=dists_proj[key]; int j=i-1;
                while(j>=0&&dists_proj[ranked[j]]>kd){ranked[j+1]=ranked[j];j--;}
                ranked[j+1]=key;
            }

            /* L1-only */
            if(ranked[0]==y_test[s]) correct_l1++;

            /* Stage 2: L1 refinement in PIXEL space among top-K */
            /* top-3 */
            {
                int pred=ranked[0]; int64_t best=INT64_MAX;
                for(int k=0;k<3;k++){
                    int c=ranked[k]; int64_t d=0;
                    for(int p=0;p<INPUT_DIM;p++){
                        int64_t x=(int64_t)img[p]-(int64_t)pixel_centroids[c][p];
                        d+=(x>=0)?x:-x;
                    }
                    if(d<best){best=d;pred=c;}
                }
                if(pred==y_test[s]) correct_refine3++;
            }
            /* top-5 */
            {
                int pred=ranked[0]; int64_t best=INT64_MAX;
                for(int k=0;k<5;k++){
                    int c=ranked[k]; int64_t d=0;
                    for(int p=0;p<INPUT_DIM;p++){
                        int64_t x=(int64_t)img[p]-(int64_t)pixel_centroids[c][p];
                        d+=(x>=0)?x:-x;
                    }
                    if(d<best){best=d;pred=c;}
                }
                if(pred==y_test[s]) correct_refine5++;
            }
        }

        printf("  L1 proj only:                 %d/%d = %d.%02d%%\n",
               correct_l1,n_test,correct_l1*100/n_test,(correct_l1*10000/n_test)%100);
        printf("  L1 proj top-3 → pixel refine: %d/%d = %d.%02d%%\n",
               correct_refine3,n_test,correct_refine3*100/n_test,(correct_refine3*10000/n_test)%100);
        printf("  L1 proj top-5 → pixel refine: %d/%d = %d.%02d%%\n\n",
               correct_refine5,n_test,correct_refine5*100/n_test,(correct_refine5*10000/n_test)%100);

        free(proj_w);free(proj_packed);free(train_proj);free(proj_sums);free(proj_cents);
    }

    printf("Zero float. Zero gradients. Pure lattice geometry.\n");
    free(x_train);free(y_train);free(x_test);free(y_test);
    return 0;
}
