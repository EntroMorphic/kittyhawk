/*
 * mnist_local_v2.c — rerun P1 with fixed local architectures.
 *
 * Atomic decomposition of the P1 gate (journal/lvg_atomics_decomposition.md)
 * identified two structural failures in the local architecture that the
 * observable-signal meta-router cannot work around:
 *
 *   1. Filter miss: 7.4% of rescues have correct class outside H1's top-50
 *      entirely. No resolver reading the filtered pool can reach them.
 *   2. Filter rank destruction: H1 alone preserves 98.6% of neighborhood
 *      membership but only 55.5% of top-1 rank. Damages cluster where
 *      local fusion has had to dig deep (ranks 21-50) in H1's ordering.
 *
 * Two fixes, independently composable:
 *
 *   Fix A — widen K_RESOLVE from 50 to 100 / 200, so filter-miss rescues
 *     become filter-hit rescues.
 *   Fix B — fused filter: take top-K by (H1+H2) summed distance instead
 *     of H1 alone. Two independent hashes at the filter stage, before any
 *     resolver runs. Resolver stage uses H3+H4 instead of H2+H3+H4.
 *
 * Variant grid (all compared against the same Gq = H1+H2+H3+H4 over 60K):
 *
 *                K=50           K=100          K=200
 *   H1 filter    L50_H1         L100_H1        L200_H1
 *   H1+H2 filter L50_H12        L100_H12       L200_H12
 *
 * All six variants are cheap to compute per query if we cache the four
 * global hash distance arrays once. For each variant we report:
 *
 *   - aggregate accuracy
 *   - 2x2 contingency vs Gq (rescues, damages, net)
 *   - oracle ceiling (variant right ∪ Gq right)
 *
 * Usage: ./mnist_local_v2 <mnist_dir>
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
#define IMG_W 28
#define IMG_H 28
#define N_CLASSES 10
#define N_PROJ 16
#define DENSITY 0.33
#define K_MAX 200    /* widest filter we test */
#define N_VARIANTS 6

static const char* variant_names[N_VARIANTS] = {
    "L50_H1   (K=50,  H1 filter)   ",
    "L100_H1  (K=100, H1 filter)   ",
    "L200_H1  (K=200, H1 filter)   ",
    "L50_H12  (K=50,  H1+H2 filter)",
    "L100_H12 (K=100, H1+H2 filter)",
    "L200_H12 (K=200, H1+H2 filter)"
};
static const int variant_K[N_VARIANTS] = {50, 100, 200, 50, 100, 200};
static const int variant_fusedfilter[N_VARIANTS] = {0, 0, 0, 1, 1, 1};

/* ── Loaders/deskew/RNG/tau (mirrored) ───────────────────────────────── */

static uint32_t read_u32_be(FILE* f) {
    uint8_t b[4]; fread(b,1,4,f);
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];
}
static m4t_mtfp_t* load_images_mtfp(const char* path, int* n) {
    FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",path);exit(1);}
    read_u32_be(f); *n=(int)read_u32_be(f);
    {
        int rows=(int)read_u32_be(f),cols=(int)read_u32_be(f);
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
        int* l=malloc(*n*sizeof(int));
        for(int i=0;i<*n;i++) l[i]=(int)raw[i];
        free(raw);
        return l;
    }
}
static void deskew_image(m4t_mtfp_t* dst, const m4t_mtfp_t* src) {
    int64_t sum_p=0,sum_xp=0,sum_yp=0;
    for(int y=0;y<IMG_H;y++)
        for(int x=0;x<IMG_W;x++){
            int64_t p=(int64_t)src[y*IMG_W+x];
            sum_p+=p; sum_xp+=(int64_t)x*p; sum_yp+=(int64_t)y*p;
        }
    if(sum_p==0){memcpy(dst,src,INPUT_DIM*sizeof(m4t_mtfp_t));return;}
    {
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
    {
        size_t idx = (size_t)(d * (double)n);
        if (idx >= n) idx = n-1;
        return v[idx];
    }
}
static void build_signature_set(
    int N_proj,
    const m4t_mtfp_t* x_train, int n_train,
    const m4t_mtfp_t* x_test, int n_test,
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
    double density,
    uint8_t** out_train_sigs, uint8_t** out_test_sigs)
{
    int Sp=M4T_TRIT_PACKED_BYTES(N_proj);
    int proj_Dp=M4T_TRIT_PACKED_BYTES(INPUT_DIM);
    rng_s[0]=s0; rng_s[1]=s1; rng_s[2]=s2; rng_s[3]=s3;
    {
        m4t_trit_t* proj_w=malloc((size_t)N_proj*INPUT_DIM);
        uint8_t* proj_packed=malloc((size_t)N_proj*proj_Dp);
        m4t_mtfp_t* train_proj=malloc((size_t)n_train*N_proj*sizeof(m4t_mtfp_t));
        m4t_mtfp_t* test_proj =malloc((size_t)n_test *N_proj*sizeof(m4t_mtfp_t));
        int64_t tau_q;
        int64_t* tmp;
        for(int i=0;i<N_proj*INPUT_DIM;i++){
            uint32_t r=rng_next()%3;
            proj_w[i]=(r==0)?-1:(r==1)?0:1;
        }
        m4t_pack_trits_rowmajor(proj_packed,proj_w,N_proj,INPUT_DIM);
        free(proj_w);
        for(int i=0;i<n_train;i++)
            m4t_mtfp_ternary_matmul_bt(train_proj+(size_t)i*N_proj,
                                       x_train+(size_t)i*INPUT_DIM,
                                       proj_packed,1,INPUT_DIM,N_proj);
        for(int i=0;i<n_test;i++)
            m4t_mtfp_ternary_matmul_bt(test_proj+(size_t)i*N_proj,
                                       x_test+(size_t)i*INPUT_DIM,
                                       proj_packed,1,INPUT_DIM,N_proj);
        {
            size_t total=(size_t)1000*N_proj;
            int64_t* buf=malloc(total*sizeof(int64_t));
            for(int i=0;i<1000;i++)
                for(int p=0;p<N_proj;p++){
                    int64_t v=train_proj[(size_t)i*N_proj+p];
                    buf[(size_t)i*N_proj+p]=(v>=0)?v:-v;
                }
            tau_q=tau_for_density(buf,total,density);
            free(buf);
        }
        *out_train_sigs=calloc((size_t)n_train*Sp,1);
        *out_test_sigs =calloc((size_t)n_test *Sp,1);
        tmp=malloc((size_t)N_proj*sizeof(int64_t));
        for(int i=0;i<n_train;i++){
            for(int p=0;p<N_proj;p++) tmp[p]=(int64_t)train_proj[(size_t)i*N_proj+p];
            m4t_route_threshold_extract((*out_train_sigs)+(size_t)i*Sp,tmp,tau_q,N_proj);
        }
        for(int i=0;i<n_test;i++){
            for(int p=0;p<N_proj;p++) tmp[p]=(int64_t)test_proj[(size_t)i*N_proj+p];
            m4t_route_threshold_extract((*out_test_sigs)+(size_t)i*Sp,tmp,tau_q,N_proj);
        }
        free(tmp); free(train_proj); free(test_proj); free(proj_packed);
    }
}

/* Partial-sort top-K of dists[] by ascending distance, with tiebreak by
 * index. Fills out_idx[K] with the selected indices. */
static void top_k_of(const int32_t* dists, int n, int K,
                     int32_t* scratch_d, int* scratch_i)
{
    for (int j = 0; j < K; j++) { scratch_d[j] = INT32_MAX; scratch_i[j] = -1; }
    for (int i = 0; i < n; i++) {
        int32_t d = dists[i];
        if (d >= scratch_d[K-1]) continue;
        scratch_d[K-1] = d; scratch_i[K-1] = i;
        for (int j = K-2; j >= 0; j--) {
            if (scratch_d[j+1] < scratch_d[j]) {
                int32_t td = scratch_d[j]; scratch_d[j] = scratch_d[j+1]; scratch_d[j+1] = td;
                int ti = scratch_i[j]; scratch_i[j] = scratch_i[j+1]; scratch_i[j+1] = ti;
            } else break;
        }
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

    deskew_all(x_train, n_train);
    deskew_all(x_test, n_test);

    int Sp=M4T_TRIT_PACKED_BYTES(N_PROJ);
    uint8_t *trA,*teA,*trB,*teB,*trC,*teC,*trD,*teD;
    build_signature_set(N_PROJ, x_train, n_train, x_test, n_test,
                        42, 123, 456, 789, DENSITY, &trA, &teA);
    build_signature_set(N_PROJ, x_train, n_train, x_test, n_test,
                        1337, 2718, 3141, 5923, DENSITY, &trB, &teB);
    build_signature_set(N_PROJ, x_train, n_train, x_test, n_test,
                        1009, 2017, 3041, 5059, DENSITY, &trC, &teC);
    build_signature_set(N_PROJ, x_train, n_train, x_test, n_test,
                        9001, 9002, 9003, 9004, DENSITY, &trD, &teD);

    uint8_t* mask=malloc(Sp); memset(mask,0xFF,Sp);

    printf("mnist_local_v2 — P1 rerun with fixed local architectures\n");
    printf("N_PROJ=%d, density=%.2f, deskewed MNIST, single seed.\n\n",
           N_PROJ, DENSITY);

    /* Per-query globally cached distances. */
    int32_t* dA = malloc((size_t)n_train*sizeof(int32_t));
    int32_t* dB = malloc((size_t)n_train*sizeof(int32_t));
    int32_t* dC = malloc((size_t)n_train*sizeof(int32_t));
    int32_t* dD = malloc((size_t)n_train*sizeof(int32_t));
    int32_t* dAB = malloc((size_t)n_train*sizeof(int32_t));

    /* Per-variant counters. */
    int correct[N_VARIANTS] = {0};
    int Gq_correct = 0;

    /* Per-variant 2x2 vs Gq. */
    int v_LR_GR[N_VARIANTS] = {0};
    int v_LR_GW[N_VARIANTS] = {0};
    int v_LW_GR[N_VARIANTS] = {0};
    int v_LW_GW[N_VARIANTS] = {0};

    /* Per-variant oracle ceiling (variant right ∪ Gq right). */
    int v_oracle[N_VARIANTS] = {0};

    /* Filter-ceiling@K for both filters (H1 alone and H1+H2). */
    int ceiling_h1_K[N_VARIANTS] = {0};
    int ceiling_h12_K[N_VARIANTS] = {0};

    int32_t scratch_d[K_MAX]; int scratch_i[K_MAX];

    clock_t t0 = clock();
    for (int s = 0; s < n_test; s++) {
        const uint8_t* qA=teA+(size_t)s*Sp;
        const uint8_t* qB=teB+(size_t)s*Sp;
        const uint8_t* qC=teC+(size_t)s*Sp;
        const uint8_t* qD=teD+(size_t)s*Sp;
        int y = y_test[s];

        /* Compute all four global hash distances once per query. */
        for (int i = 0; i < n_train; i++) {
            dA[i] = m4t_popcount_dist(qA, trA+(size_t)i*Sp, mask, Sp);
            dB[i] = m4t_popcount_dist(qB, trB+(size_t)i*Sp, mask, Sp);
            dC[i] = m4t_popcount_dist(qC, trC+(size_t)i*Sp, mask, Sp);
            dD[i] = m4t_popcount_dist(qD, trD+(size_t)i*Sp, mask, Sp);
            dAB[i] = dA[i] + dB[i];
        }

        /* Gq reference: argmin of dA+dB+dC+dD over 60K. */
        int Gq_label = -1;
        {
            int32_t best = INT32_MAX;
            for (int i = 0; i < n_train; i++) {
                int32_t score = dA[i] + dB[i] + dC[i] + dD[i];
                if (score < best) { best = score; Gq_label = y_train[i]; }
            }
        }
        int Gq_right = (Gq_label == y);
        if (Gq_right) Gq_correct++;

        /* Per-variant evaluation. */
        for (int v = 0; v < N_VARIANTS; v++) {
            int K = variant_K[v];
            int fused = variant_fusedfilter[v];

            /* Filter step: take top-K by H1 alone or by H1+H2. */
            top_k_of(fused ? dAB : dA, n_train, K, scratch_d, scratch_i);

            /* Filter ceiling: is correct class in top-K? */
            int in_pool = 0;
            for (int j = 0; j < K; j++)
                if (y_train[scratch_i[j]] == y) { in_pool = 1; break; }
            if (in_pool) {
                if (fused) ceiling_h12_K[v]++;
                else       ceiling_h1_K[v]++;
            }

            /* Resolver step: summed distance of the NON-filter hashes
             * over the top-K. H1-filter variants sum H2+H3+H4; H1+H2-filter
             * variants sum H3+H4. */
            int L_label = -1;
            int32_t best = INT32_MAX;
            if (fused) {
                for (int j = 0; j < K; j++) {
                    int idx = scratch_i[j];
                    int32_t score = dC[idx] + dD[idx];
                    if (score < best) { best = score; L_label = y_train[idx]; }
                }
            } else {
                for (int j = 0; j < K; j++) {
                    int idx = scratch_i[j];
                    int32_t score = dB[idx] + dC[idx] + dD[idx];
                    if (score < best) { best = score; L_label = y_train[idx]; }
                }
            }
            int L_right = (L_label == y);
            if (L_right) correct[v]++;

            /* 2x2 vs Gq. */
            if ( L_right &&  Gq_right) v_LR_GR[v]++;
            if ( L_right && !Gq_right) v_LR_GW[v]++;
            if (!L_right &&  Gq_right) v_LW_GR[v]++;
            if (!L_right && !Gq_right) v_LW_GW[v]++;

            /* Oracle (variant ∪ Gq). */
            if (L_right || Gq_right) v_oracle[v]++;
        }
    }
    double secs = (double)(clock()-t0)/CLOCKS_PER_SEC;

    printf("Inference: %.1fs for %d test queries.\n\n", secs, n_test);

    printf("Gq reference:  %.2f%% (H1+H2+H3+H4 over 60K)\n\n",
           100.0*Gq_correct/n_test);

    printf("Variant accuracies:\n");
    for (int v = 0; v < N_VARIANTS; v++) {
        double acc = 100.0*correct[v]/n_test;
        printf("  %-32s  %6.2f%%   (Δ vs Gq: %+6.2f%%)\n",
               variant_names[v], acc, acc - 100.0*Gq_correct/n_test);
    }
    printf("\n");

    printf("Filter ceilings (correct class in filter's top-K):\n");
    printf("  K    H1 filter    H1+H2 filter    lift\n");
    for (int v = 0; v < N_VARIANTS; v++) {
        if (variant_fusedfilter[v]) continue;
        int K = variant_K[v];
        int h1 = ceiling_h1_K[v];
        /* find the matching H1+H2 variant at this K */
        int h12 = 0;
        for (int u = 0; u < N_VARIANTS; u++)
            if (variant_fusedfilter[u] && variant_K[u]==K) { h12 = ceiling_h12_K[u]; break; }
        printf("  %3d  %6.2f%%     %6.2f%%        %+6.2f%%\n",
               K, 100.0*h1/n_test, 100.0*h12/n_test,
               100.0*(h12-h1)/n_test);
    }
    printf("\n");

    printf("Per-variant contingency vs Gq:\n");
    printf("  %-32s  LR_GR  LR_GW  LW_GR  LW_GW   rescue  damage   net   oracle\n",
           "variant");
    for (int v = 0; v < N_VARIANTS; v++) {
        printf("  %-32s  %5d  %5d  %5d  %5d   %5d   %5d   %+5d  %6.2f%%\n",
               variant_names[v],
               v_LR_GR[v], v_LR_GW[v], v_LW_GR[v], v_LW_GW[v],
               v_LW_GR[v], v_LR_GW[v],
               v_LW_GR[v] - v_LR_GW[v],
               100.0*v_oracle[v]/n_test);
    }
    printf("\n");

    printf("Interpretation:\n");
    printf("  Each L variant lifts local accuracy. The shrinking rescue count\n");
    printf("  shows how much of the L-vs-Gq gap each fix closes.\n");
    printf("  The oracle ceiling (L right OR Gq right) bounds any meta-router\n");
    printf("  that routes between L and Gq - higher oracle means more room.\n");
    printf("  Compare to P1 baseline: L50_H1 oracle was 92.77%%.\n");

    free(dA); free(dB); free(dC); free(dD); free(dAB);
    free(mask);
    free(trA); free(teA); free(trB); free(teB);
    free(trC); free(teC); free(trD); free(teD);
    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
