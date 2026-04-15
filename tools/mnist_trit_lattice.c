/*
 * mnist_trit_lattice.c — fully routed Trit Lattice classification on MNIST.
 *
 * Projection stays the same as the original tool: random ternary routing
 * via m4t_mtfp_ternary_matmul_bt. The sixth-round remediation removes the
 * dense projection/pixel L1 classifier and keeps the decision inside the
 * routing surface end to end:
 *
 *   project -> threshold_extract -> route_distance_batch -> topk_abs
 *
 * Class centroids are still formed in projection space, but classification
 * consumes only class/query signatures and route scores. No pixel-space
 * refinement remains in the live path.
 *
 * Usage: ./mnist_trit_lattice <mnist_dir>
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
#define DENSITY 0.33

static uint32_t read_u32_be(FILE* f) {
    uint8_t b[4]; fread(b,1,4,f);
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];
}

static m4t_mtfp_t* load_images_mtfp(const char* path, int* n) {
    FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",path);exit(1);}
    read_u32_be(f); *n=(int)read_u32_be(f);
    {
        int rows=(int)read_u32_be(f), cols=(int)read_u32_be(f);
        int dim=rows*cols; size_t total=(size_t)(*n)*dim;
        uint8_t* raw=malloc(total); fread(raw,1,total,f); fclose(f);
        m4t_mtfp_t* data=malloc(total*sizeof(m4t_mtfp_t));
        for(size_t i=0;i<total;i++)
            data[i]=(m4t_mtfp_t)(((int32_t)raw[i]*M4T_MTFP_SCALE+127)/255);
        free(raw);
        return data;
    }
}

static int* load_labels(const char* path, int* n) {
    FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",path);exit(1);}
    read_u32_be(f); *n=(int)read_u32_be(f);
    {
        uint8_t* raw=malloc(*n); fread(raw,1,*n,f); fclose(f);
        int* labels=malloc(*n*sizeof(int));
        for(int i=0;i<*n;i++) labels[i]=(int)raw[i];
        free(raw);
        return labels;
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
    int64_t x = *(const int64_t*)a;
    int64_t y = *(const int64_t*)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

static int64_t tau_for_density(int64_t* abs_values, size_t n, double density) {
    if (n == 0 || density <= 0.0) return 0;
    if (density >= 1.0) return abs_values[n - 1] + 1;
    qsort(abs_values, n, sizeof(int64_t), cmp_i64);
    {
        size_t idx = (size_t)(density * (double)n);
        if (idx >= n) idx = n - 1;
        return abs_values[idx];
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

    printf("Trit Lattice LSH — fully routed MNIST (zero float)\n");
    printf("Loaded %d train, %d test\n\n", n_train, n_test);

    {
        int class_counts[N_CLASSES];
        int proj_sizes[] = {256, 512, 1024, 2048};
        int n_sizes = 4;

        memset(class_counts, 0, sizeof(class_counts));
        for(int i=0;i<n_train;i++) class_counts[y_train[i]]++;

        for(int si=0;si<n_sizes;si++){
            int N_PROJ=proj_sizes[si];
            int proj_Dp=M4T_TRIT_PACKED_BYTES(INPUT_DIM);
            int Sp=M4T_TRIT_PACKED_BYTES(N_PROJ);
            m4t_trit_t* proj_w;
            uint8_t* proj_packed;
            m4t_mtfp_t* train_proj;
            int64_t* proj_sums;
            m4t_mtfp_t* proj_cents;
            int64_t* dim_mean;
            int64_t* class_diff_abs;
            int64_t tau_c;
            int64_t tau_q;
            uint8_t* class_sigs;
            uint8_t* mask;
            long class_zero = 0, class_total = 0;
            long query_zero = 0, query_total = 0;
            int correct_routed = 0;
            clock_t t0 = clock();

            rng_s[0]=42; rng_s[1]=123; rng_s[2]=456; rng_s[3]=789;

            printf("=== N_PROJ = %d ===\n", N_PROJ);

            proj_w=malloc((size_t)N_PROJ*INPUT_DIM);
            for(int i=0;i<N_PROJ*INPUT_DIM;i++){
                uint32_t r=rng_next()%3;
                proj_w[i]=(r==0)?-1:(r==1)?0:1;
            }
            proj_packed=malloc((size_t)N_PROJ*proj_Dp);
            m4t_pack_trits_rowmajor(proj_packed,proj_w,N_PROJ,INPUT_DIM);
            free(proj_w);

            train_proj=malloc((size_t)n_train*N_PROJ*sizeof(m4t_mtfp_t));
            for(int i=0;i<n_train;i++)
                m4t_mtfp_ternary_matmul_bt(train_proj+(size_t)i*N_PROJ,
                                           x_train+(size_t)i*INPUT_DIM,
                                           proj_packed,1,INPUT_DIM,N_PROJ);

            proj_sums=calloc((size_t)N_CLASSES*N_PROJ,sizeof(int64_t));
            for(int i=0;i<n_train;i++){
                int c=y_train[i];
                for(int p=0;p<N_PROJ;p++)
                    proj_sums[(size_t)c*N_PROJ+p]+=(int64_t)train_proj[(size_t)i*N_PROJ+p];
            }
            proj_cents=malloc((size_t)N_CLASSES*N_PROJ*sizeof(m4t_mtfp_t));
            for(int c=0;c<N_CLASSES;c++)
                for(int p=0;p<N_PROJ;p++)
                    proj_cents[(size_t)c*N_PROJ+p]=(m4t_mtfp_t)(proj_sums[(size_t)c*N_PROJ+p]/class_counts[c]);

            dim_mean=calloc((size_t)N_PROJ,sizeof(int64_t));
            for(int c=0;c<N_CLASSES;c++)
                for(int p=0;p<N_PROJ;p++)
                    dim_mean[p]+=(int64_t)proj_cents[(size_t)c*N_PROJ+p];
            for(int p=0;p<N_PROJ;p++) dim_mean[p]/=N_CLASSES;

            class_diff_abs=malloc((size_t)N_CLASSES*N_PROJ*sizeof(int64_t));
            for(int c=0;c<N_CLASSES;c++)
                for(int p=0;p<N_PROJ;p++){
                    int64_t d=(int64_t)proj_cents[(size_t)c*N_PROJ+p]-dim_mean[p];
                    class_diff_abs[(size_t)c*N_PROJ+p]=(d>=0)?d:-d;
                }
            tau_c=tau_for_density(class_diff_abs,(size_t)N_CLASSES*N_PROJ,DENSITY);
            free(class_diff_abs);

            {
                int sample_n = (n_train < 1000) ? n_train : 1000;
                size_t total = (size_t)sample_n * (size_t)N_PROJ;
                int64_t* abs_buf = malloc(total * sizeof(int64_t));
                for(int i=0;i<sample_n;i++)
                    for(int p=0;p<N_PROJ;p++){
                        int64_t v=train_proj[(size_t)i*N_PROJ+p];
                        abs_buf[(size_t)i*N_PROJ+p]=(v>=0)?v:-v;
                    }
                tau_q=tau_for_density(abs_buf,total,DENSITY);
                free(abs_buf);
            }

            class_sigs=calloc((size_t)N_CLASSES*Sp,1);
            mask=malloc((size_t)Sp);
            memset(mask,0xFF,(size_t)Sp);
            {
                int64_t* diff=malloc((size_t)N_PROJ*sizeof(int64_t));
                for(int c=0;c<N_CLASSES;c++){
                    uint8_t* sig = class_sigs + (size_t)c*Sp;
                    for(int p=0;p<N_PROJ;p++)
                        diff[p]=(int64_t)proj_cents[(size_t)c*N_PROJ+p]-dim_mean[p];
                    m4t_route_threshold_extract(sig,diff,tau_c,N_PROJ);
                    for(int p=0;p<N_PROJ;p++){
                        uint8_t code=(sig[p>>2]>>((p&3)*2))&0x3u;
                        class_total++;
                        if(code==0) class_zero++;
                    }
                }
                free(diff);
            }

            {
                m4t_mtfp_t* test_proj_buf = malloc((size_t)N_PROJ * sizeof(m4t_mtfp_t));
                int64_t* query_i64 = malloc((size_t)N_PROJ * sizeof(int64_t));
                uint8_t* query_sig = malloc((size_t)Sp);

                for(int s=0;s<n_test;s++){
                    m4t_route_decision_t decision;
                    int32_t dists[N_CLASSES];
                    int32_t scores[N_CLASSES];
                    const m4t_mtfp_t* img=x_test+(size_t)s*INPUT_DIM;
                    m4t_mtfp_ternary_matmul_bt(test_proj_buf,img,proj_packed,1,INPUT_DIM,N_PROJ);
                    for(int p=0;p<N_PROJ;p++) query_i64[p]=(int64_t)test_proj_buf[p];
                    m4t_route_threshold_extract(query_sig,query_i64,tau_q,N_PROJ);
                    for(int p=0;p<N_PROJ;p++){
                        uint8_t code=(query_sig[p>>2]>>((p&3)*2))&0x3u;
                        query_total++;
                        if(code==0) query_zero++;
                    }
                    m4t_route_distance_batch(dists,query_sig,class_sigs,mask,N_CLASSES,N_PROJ);
                    for(int c=0;c<N_CLASSES;c++) scores[c]=2*N_PROJ-dists[c];
                    m4t_route_topk_abs(&decision,scores,N_CLASSES,1);
                    if(decision.tile_idx==y_test[s]) correct_routed++;
                }

                free(test_proj_buf);
                free(query_i64);
                free(query_sig);
            }

            printf("  routed class-signature acc:    %d/%d = %d.%02d%%\n",
                   correct_routed,n_test,correct_routed*100/n_test,
                   (correct_routed*10000/n_test)%100);
            printf("  tau_c=%lld  tau_q=%lld  class %%zero=%d.%02d%%  query %%zero=%d.%02d%%\n",
                   (long long)tau_c,(long long)tau_q,
                   (int)((10000*class_zero/class_total)/100),
                   (int)((10000*class_zero/class_total)%100),
                   (int)((10000*query_zero/query_total)/100),
                   (int)((10000*query_zero/query_total)%100));
            printf("  routed inference time:         %.0f ms\n\n",
                   1000.0*(double)(clock()-t0)/CLOCKS_PER_SEC);

            free(class_sigs);
            free(mask);
            free(dim_mean);
            free(proj_packed);
            free(train_proj);
            free(proj_sums);
            free(proj_cents);
        }
    }

    printf("Zero float. Zero gradients. Routed lattice geometry end to end.\n");
    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
