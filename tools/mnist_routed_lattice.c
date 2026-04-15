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

/* Percentile computation on an array of absolute values. Sorts in place
 * (caller-owned buffer) and returns the value at the given fraction. */
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

        /* ── Symmetric sweep: target zero density per side ─────────────────
         *
         * Prior version used the same tau on both sides, which produces
         * ASYMMETRIC zero densities because the two inputs have different
         * scales (class-side centroid-diffs ~1/3 the scale of query-side
         * projections). This version computes tau_c and tau_q separately
         * from the actual data distributions to hit a common target zero
         * density on each side — the architecturally symmetric deployment
         * we should have been running all along.
         *
         * At target=0.33 this is the balanced base-3 distribution NORTH_STAR
         * describes: ~1/3 zero, ~1/3 +1, ~1/3 -1 on both sides of the
         * distance comparison. */
        const double TARGET_DENSITIES[] = { 0.00, 0.20, 0.33, 0.50, 0.67 };
        const int N_TAUS = (int)(sizeof(TARGET_DENSITIES) / sizeof(TARGET_DENSITIES[0]));

        /* Mean-subtract centroids once (shared across all density targets). */
        int Sp=M4T_TRIT_PACKED_BYTES(N_PROJ);
        int64_t* dim_mean=calloc((size_t)N_PROJ,sizeof(int64_t));
        for(int c=0;c<N_CLASSES;c++)
            for(int p=0;p<N_PROJ;p++)
                dim_mean[p]+=(int64_t)centroids[(size_t)c*N_PROJ+p];
        for(int p=0;p<N_PROJ;p++) dim_mean[p]/=N_CLASSES;

        /* Compute tau_c per target density from the actual |centroid-diff|
         * distribution. Class-side has N_CLASSES * N_PROJ total trit-
         * positions; sort their absolute values and pick the percentile. */
        int64_t* class_diff_abs = malloc((size_t)N_CLASSES * N_PROJ * sizeof(int64_t));
        for(int c=0;c<N_CLASSES;c++){
            for(int p=0;p<N_PROJ;p++){
                int64_t d = (int64_t)centroids[(size_t)c*N_PROJ+p] - dim_mean[p];
                class_diff_abs[(size_t)c*N_PROJ+p] = d >= 0 ? d : -d;
            }
        }
        int64_t tau_c[16];  /* N_TAUS ≤ 16 */
        {
            int64_t* buf = malloc((size_t)N_CLASSES*N_PROJ*sizeof(int64_t));
            for(int di=0; di<N_TAUS; di++){
                memcpy(buf, class_diff_abs, (size_t)N_CLASSES*N_PROJ*sizeof(int64_t));
                tau_c[di] = tau_for_density(buf, (size_t)N_CLASSES*N_PROJ, TARGET_DENSITIES[di]);
            }
            free(buf);
        }
        free(class_diff_abs);

        /* Compute tau_q per target density from a sample of training-image
         * |projection| values. Sample avoids the 60K×N_PROJ full sort
         * (would be >100M entries at N_PROJ=2048; sample is 1000×N_PROJ). */
        const int TAU_Q_SAMPLE = 1000;
        int tau_q_samples = (TAU_Q_SAMPLE < n_train) ? TAU_Q_SAMPLE : n_train;
        int64_t tau_q[16];
        {
            size_t total = (size_t)tau_q_samples * (size_t)N_PROJ;
            int64_t* buf_stable = malloc(total * sizeof(int64_t));
            for(int i=0; i<tau_q_samples; i++){
                for(int p=0; p<N_PROJ; p++){
                    int64_t v = train_proj[(size_t)i*N_PROJ + p];
                    buf_stable[(size_t)i*N_PROJ + p] = v >= 0 ? v : -v;
                }
            }
            int64_t* buf = malloc(total * sizeof(int64_t));
            for(int di=0; di<N_TAUS; di++){
                memcpy(buf, buf_stable, total * sizeof(int64_t));
                tau_q[di] = tau_for_density(buf, total, TARGET_DENSITIES[di]);
            }
            free(buf);
            free(buf_stable);
        }

        /* Build class signatures per target density using the per-side tau_c.
         * Count actual zero-density produced (should match target closely). */
        uint8_t* class_sigs_per_tau=calloc((size_t)N_TAUS*N_CLASSES*Sp,1);
        long class_zero_count_per_tau[16] = {0};
        long class_total_trits_per_tau[16] = {0};
        {
            int64_t* diff=malloc((size_t)N_PROJ*sizeof(int64_t));
            for(int di=0; di<N_TAUS; di++){
                for(int c=0;c<N_CLASSES;c++){
                    for(int p=0;p<N_PROJ;p++)
                        diff[p]=(int64_t)centroids[(size_t)c*N_PROJ+p]-dim_mean[p];
                    uint8_t* sig = class_sigs_per_tau + ((size_t)di*N_CLASSES + c)*Sp;
                    m4t_route_threshold_extract(sig, diff, tau_c[di], N_PROJ);
                    for (int p = 0; p < N_PROJ; p++) {
                        uint8_t code = (sig[p >> 2] >> ((p & 3) * 2)) & 0x3u;
                        class_total_trits_per_tau[di]++;
                        if (code == 0) class_zero_count_per_tau[di]++;
                    }
                }
            }
            free(diff);
        }
        free(dim_mean);

        /* Active-bit mask: all trits participate. */
        uint8_t* mask=malloc(Sp); memset(mask,0xFF,Sp);

        /* ── Inference: L1 once + routed per-tau ──────────────────────────── */

        m4t_mtfp_t* test_proj_buf=malloc((size_t)N_PROJ*sizeof(m4t_mtfp_t));
        int64_t* query_i64=malloc((size_t)N_PROJ*sizeof(int64_t));
        uint8_t* query_sig=malloc(Sp);
        int32_t dists[N_CLASSES];

        int correct_l1=0;
        int* correct_routed=calloc((size_t)N_TAUS, sizeof(int));

        /* Query-side scale and zero-density (collect over ALL test images
         * for tau=0; sample 500 images for the other tau values). */
        int64_t query_abs_max = 0;
        int64_t query_abs_sum = 0;
        long query_abs_count = 0;
        long* query_zero_per_tau = calloc((size_t)N_TAUS, sizeof(long));
        long* query_total_per_tau = calloc((size_t)N_TAUS, sizeof(long));

        clock_t t0=clock();
        for(int s=0;s<n_test;s++){
            const m4t_mtfp_t* img=x_test+(size_t)s*INPUT_DIM;
            m4t_mtfp_ternary_matmul_bt(test_proj_buf,img,proj_packed,1,INPUT_DIM,N_PROJ);

            /* L1-over-mantissa (tau-independent). */
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

            /* Routed-per-density. Project once, query-extract per density. */
            for(int p=0;p<N_PROJ;p++) query_i64[p]=(int64_t)test_proj_buf[p];
            if (s == 0) {
                for(int p=0;p<N_PROJ;p++) {
                    int64_t a = query_i64[p] >= 0 ? query_i64[p] : -query_i64[p];
                    if (a > query_abs_max) query_abs_max = a;
                    query_abs_sum += a;
                    query_abs_count++;
                }
            }
            int32_t max_dist=2*N_PROJ;
            for(int di=0; di<N_TAUS; di++){
                m4t_route_threshold_extract(query_sig,query_i64,tau_q[di],N_PROJ);
                for (int p = 0; p < N_PROJ; p++) {
                    uint8_t code = (query_sig[p >> 2] >> ((p & 3) * 2)) & 0x3u;
                    query_total_per_tau[di]++;
                    if (code == 0) query_zero_per_tau[di]++;
                }
                m4t_route_distance_batch(
                    dists, query_sig,
                    class_sigs_per_tau + (size_t)di*N_CLASSES*Sp,
                    mask, N_CLASSES, N_PROJ);

                int32_t scores[N_CLASSES];
                for(int c=0;c<N_CLASSES;c++) scores[c]=max_dist-dists[c];

                m4t_route_decision_t decision;
                m4t_route_topk_abs(&decision,scores,N_CLASSES,1);
                int pred_r = decision.tile_idx;
                if(pred_r<0) pred_r=0;
                if(pred_r==y_test[s]) correct_routed[di]++;
            }
        }
        clock_t t1=clock();
        double total_ms=1000.0*(double)(t1-t0)/CLOCKS_PER_SEC;

        printf("  Query-side proj scale (image 0): max=%lld  mean|proj|=%lld\n",
               (long long)query_abs_max,
               (long long)(query_abs_count > 0 ? query_abs_sum/query_abs_count : 0));
        printf("\n");
        printf("  L1-over-mantissa (dense decision): %d/%d = %d.%02d%%\n",
               correct_l1,n_test,correct_l1*100/n_test,(correct_l1*10000/n_test)%100);
        printf("  %-7s  %-10s  %-10s  %-13s  %-13s  %s\n",
               "target", "tau_c", "tau_q", "class %zero", "query %zero", "routed acc");
        for(int di=0; di<N_TAUS; di++){
            int class_pct100 = (int)((100 * class_zero_count_per_tau[di] * 100) / class_total_trits_per_tau[di]);
            int query_pct100 = (int)((100 * query_zero_per_tau[di] * 100) / query_total_per_tau[di]);
            printf("  %.2f     %-10lld  %-10lld  %3d.%02d%%        %3d.%02d%%        %d.%02d%%\n",
                   TARGET_DENSITIES[di],
                   (long long)tau_c[di], (long long)tau_q[di],
                   class_pct100/100, class_pct100%100,
                   query_pct100/100, query_pct100%100,
                   correct_routed[di]*100/n_test,
                   (correct_routed[di]*10000/n_test)%100);
        }
        printf("  Inference (L1 + %d density targets, %d images):  %.0f ms\n\n",
               N_TAUS, n_test, total_ms);

        free(test_proj_buf); free(query_i64); free(query_sig);
        free(class_sigs_per_tau); free(correct_routed); free(mask);
        free(query_zero_per_tau); free(query_total_per_tau);
        free(centroids); free(class_sums); free(train_proj);
        free(proj_w); free(proj_packed);
    }

    printf("Zero float. Zero gradients. Full routing surface exercised.\n");
    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
