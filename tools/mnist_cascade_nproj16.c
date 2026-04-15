/*
 * mnist_cascade_nproj16.c — cascade experiment at N_PROJ=16.
 *
 * LMM synthesize output: the 16-bit hash is a FILTER, not a classifier.
 * Use it to narrow 60K → top-K, then resolve locally with a cheap signal.
 *
 * Variants run in a single pass:
 *   E1/E2. Pixel-L1 1-NN within top-K for K ∈ {5,10,20,50,100}.
 *   E3.    Pixel-L2 (squared) 1-NN and pixel-L1 3-NN majority, K=20 fixed.
 *   E4.    Partition-aware pixel-L1: if tied_count==1, take top-1 label;
 *          else resolve via pixel-L1 over tied set (or top-K if larger).
 *   E5.    Secondary-hash (different seed) Hamming re-rank within top-K=20.
 *
 * All variants share one primary-hash pass, so cost is amortized.
 *
 * Usage: ./mnist_cascade_nproj16 <mnist_dir>
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
#define MAX_DIST (2 * N_PROJ)

/* ── Loaders, deskew, RNG, τ (mirrored from probe tool) ──────────────── */

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
static void rng_seed(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    rng_s[0]=a; rng_s[1]=b; rng_s[2]=c; rng_s[3]=d;
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

/* Build signatures under given RNG seeds. Returns packed sigs + τ. */
static void build_signatures(const m4t_mtfp_t* x_train, int n_train,
                             const m4t_mtfp_t* x_test,  int n_test,
                             uint32_t seed_a, uint32_t seed_b,
                             uint32_t seed_c, uint32_t seed_d,
                             uint8_t** out_train_sigs,
                             uint8_t** out_test_sigs,
                             int*      out_Sp)
{
    rng_seed(seed_a, seed_b, seed_c, seed_d);
    m4t_trit_t* proj_w=malloc((size_t)N_PROJ*INPUT_DIM);
    for(int i=0;i<N_PROJ*INPUT_DIM;i++){
        uint32_t r=rng_next()%3;
        proj_w[i]=(r==0)?-1:(r==1)?0:1;
    }
    int proj_Dp=M4T_TRIT_PACKED_BYTES(INPUT_DIM);
    uint8_t* proj_packed=malloc((size_t)N_PROJ*proj_Dp);
    m4t_pack_trits_rowmajor(proj_packed,proj_w,N_PROJ,INPUT_DIM);
    free(proj_w);

    m4t_mtfp_t* train_proj=malloc((size_t)n_train*N_PROJ*sizeof(m4t_mtfp_t));
    m4t_mtfp_t* test_proj =malloc((size_t)n_test *N_PROJ*sizeof(m4t_mtfp_t));
    for(int i=0;i<n_train;i++)
        m4t_mtfp_ternary_matmul_bt(train_proj+(size_t)i*N_PROJ,
                                    x_train+(size_t)i*INPUT_DIM,
                                    proj_packed,1,INPUT_DIM,N_PROJ);
    for(int i=0;i<n_test;i++)
        m4t_mtfp_ternary_matmul_bt(test_proj+(size_t)i*N_PROJ,
                                    x_test+(size_t)i*INPUT_DIM,
                                    proj_packed,1,INPUT_DIM,N_PROJ);

    int64_t tau_q;
    {
        size_t total=(size_t)1000*N_PROJ;
        int64_t* buf=malloc(total*sizeof(int64_t));
        for(int i=0;i<1000;i++)
            for(int p=0;p<N_PROJ;p++){
                int64_t v=train_proj[(size_t)i*N_PROJ+p];
                buf[(size_t)i*N_PROJ+p]=(v>=0)?v:-v;
            }
        tau_q=tau_for_density(buf,total,DENSITY);
        free(buf);
    }

    int Sp=M4T_TRIT_PACKED_BYTES(N_PROJ);
    uint8_t* train_sigs=calloc((size_t)n_train*Sp,1);
    uint8_t* test_sigs =calloc((size_t)n_test *Sp,1);
    int64_t* tmp=malloc((size_t)N_PROJ*sizeof(int64_t));
    for(int i=0;i<n_train;i++){
        for(int p=0;p<N_PROJ;p++) tmp[p]=(int64_t)train_proj[(size_t)i*N_PROJ+p];
        m4t_route_threshold_extract(train_sigs+(size_t)i*Sp,tmp,tau_q,N_PROJ);
    }
    for(int i=0;i<n_test;i++){
        for(int p=0;p<N_PROJ;p++) tmp[p]=(int64_t)test_proj[(size_t)i*N_PROJ+p];
        m4t_route_threshold_extract(test_sigs+(size_t)i*Sp,tmp,tau_q,N_PROJ);
    }
    free(tmp);
    free(train_proj); free(test_proj); free(proj_packed);

    *out_train_sigs = train_sigs;
    *out_test_sigs  = test_sigs;
    *out_Sp         = Sp;
}

/* Pixel L1 distance between two images. */
static int64_t pixel_l1(const m4t_mtfp_t* a, const m4t_mtfp_t* b) {
    int64_t s=0;
    for(int i=0;i<INPUT_DIM;i++){
        int32_t d=(int32_t)a[i]-(int32_t)b[i];
        s += (d<0)?-d:d;
    }
    return s;
}
static int64_t pixel_l2sq(const m4t_mtfp_t* a, const m4t_mtfp_t* b) {
    int64_t s=0;
    for(int i=0;i<INPUT_DIM;i++){
        int32_t d=(int32_t)a[i]-(int32_t)b[i];
        s += (int64_t)d*d;
    }
    return s;
}

/* Top-K extraction: given dists[n_train], fill indices[K] sorted by distance
 * ascending (ties broken by index order). */
static void top_k_indices(const int32_t* dists, int n, int K,
                          int32_t* out_d, int* out_idx) {
    for(int j=0;j<K;j++){ out_d[j]=INT32_MAX; out_idx[j]=-1; }
    for(int i=0;i<n;i++){
        int32_t d=dists[i];
        if(d>=out_d[K-1]) continue;
        out_d[K-1]=d; out_idx[K-1]=i;
        for(int j=K-2;j>=0;j--){
            if(out_d[j+1]<out_d[j]){
                int32_t td=out_d[j]; out_d[j]=out_d[j+1]; out_d[j+1]=td;
                int ti=out_idx[j]; out_idx[j]=out_idx[j+1]; out_idx[j+1]=ti;
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

    uint8_t *train_sigs_A, *test_sigs_A, *train_sigs_B, *test_sigs_B;
    int Sp;
    build_signatures(x_train,n_train,x_test,n_test,
                     42,123,456,789,
                     &train_sigs_A,&test_sigs_A,&Sp);
    build_signatures(x_train,n_train,x_test,n_test,
                     1337,2718,3141,5923,
                     &train_sigs_B,&test_sigs_B,&Sp);

    uint8_t* mask=malloc(Sp); memset(mask,0xFF,Sp);

    printf("N_PROJ=%d cascade experiment — deskewed MNIST, density=%.2f\n",
           N_PROJ, DENSITY);
    printf("Primary seed=42; secondary seed=1337.  Sig size=%d bytes.\n",Sp);
    printf("%d train prototypes, %d test queries.\n\n", n_train, n_test);

    /* Ks to report. */
    const int Ks[] = {5, 10, 20, 50, 100};
    const int nK   = sizeof(Ks)/sizeof(Ks[0]);
    const int Kmax = 100;

    /* Counters per K. */
    int pure_maj_correct[5] = {0};       /* baseline: k-NN majority on primary hash */
    int cascade_l1_correct[5] = {0};     /* E1/E2: pixel-L1 1-NN within top-K */
    int cascade_l2_correct[5] = {0};     /* E3: pixel-L2sq 1-NN within top-K */

    /* K=20 fixed variants (E3, E4, E5). */
    int cascade_l1_3nn_correct = 0;      /* E3: pixel-L1 3-NN majority within top-20 */
    int partition_aware_correct = 0;     /* E4: skip resolver if tied_count==1 */
    int secondary_hash_correct = 0;      /* E5: secondary-hash Hamming 1-NN within top-20 */

    /* Ceiling diagnostic. */
    int correct_in_topKmax = 0;

    int32_t* dists_A = malloc((size_t)n_train*sizeof(int32_t));

    clock_t t0 = clock();
    for (int s = 0; s < n_test; s++) {
        const uint8_t* q_sig_A = test_sigs_A + (size_t)s*Sp;
        const m4t_mtfp_t* q_img = x_test + (size_t)s*INPUT_DIM;

        /* Primary hash distances. */
        for (int i = 0; i < n_train; i++) {
            const uint8_t* r_sig = train_sigs_A + (size_t)i*Sp;
            dists_A[i] = m4t_popcount_dist(q_sig_A, r_sig, mask, Sp);
        }

        /* Top-Kmax by primary hash distance. */
        int32_t topd[100]; int topi[100];
        top_k_indices(dists_A, n_train, Kmax, topd, topi);

        int y = y_test[s];
        int correct_present = 0;
        for (int j=0;j<Kmax;j++) if (y_train[topi[j]]==y){correct_present=1;break;}
        if (correct_present) correct_in_topKmax++;

        /* Tied-at-top-1 count (for E4 partition awareness). */
        int min_d = topd[0];
        int tied_count = 0;
        for (int j=0;j<Kmax;j++) if (topd[j]==min_d) tied_count++;
        /* (Approximate: uses top-Kmax. If actual tied_count > Kmax this caps,
         * but tied-min avg was 4-6, max seen ~500 — rare. Acceptable for E4.) */

        /* For each K: compute pure-hash majority, pixel-L1 1-NN, pixel-L2 1-NN. */
        for (int ki = 0; ki < nK; ki++) {
            int K = Ks[ki];

            /* Pure-hash majority on top-K. */
            int counts[N_CLASSES] = {0};
            for (int j=0;j<K;j++) counts[y_train[topi[j]]]++;
            int pred = 0;
            for (int c=1;c<N_CLASSES;c++)
                if (counts[c]>counts[pred]) pred=c;
            if (pred==y) pure_maj_correct[ki]++;

            /* Pixel-L1 1-NN. */
            int64_t best_l1 = INT64_MAX; int best_l1_label = -1;
            int64_t best_l2 = INT64_MAX; int best_l2_label = -1;
            for (int j=0;j<K;j++) {
                const m4t_mtfp_t* r_img = x_train + (size_t)topi[j]*INPUT_DIM;
                int64_t l1 = pixel_l1(q_img, r_img);
                int64_t l2 = pixel_l2sq(q_img, r_img);
                if (l1 < best_l1) { best_l1 = l1; best_l1_label = y_train[topi[j]]; }
                if (l2 < best_l2) { best_l2 = l2; best_l2_label = y_train[topi[j]]; }
            }
            if (best_l1_label==y) cascade_l1_correct[ki]++;
            if (best_l2_label==y) cascade_l2_correct[ki]++;
        }

        /* K=20 fixed: E3 pixel-L1 3-NN majority; E4 partition-aware; E5 secondary. */
        const int K20 = 20;

        /* E3: pixel-L1 within top-20, take 3 nearest, majority vote. */
        int64_t pl1[20]; int pll[20];
        for (int j=0;j<K20;j++) {
            pl1[j] = pixel_l1(q_img, x_train + (size_t)topi[j]*INPUT_DIM);
            pll[j] = y_train[topi[j]];
        }
        /* insertion sort by pl1 ascending */
        for (int a=1;a<K20;a++){
            int64_t kd=pl1[a]; int kl=pll[a]; int b=a-1;
            while(b>=0 && pl1[b]>kd){ pl1[b+1]=pl1[b]; pll[b+1]=pll[b]; b--; }
            pl1[b+1]=kd; pll[b+1]=kl;
        }
        {
            int c3[N_CLASSES]={0};
            for(int j=0;j<3;j++) c3[pll[j]]++;
            int p=0; for(int c=1;c<N_CLASSES;c++) if(c3[c]>c3[p]) p=c;
            if (p==y) cascade_l1_3nn_correct++;
        }

        /* E4: partition-aware. If only one tied-at-top-1, use that label.
         * Otherwise apply pixel-L1 1-NN within top-20 (or tied set). */
        if (tied_count == 1) {
            if (y_train[topi[0]] == y) partition_aware_correct++;
        } else {
            int64_t best_l1 = INT64_MAX; int best_label = -1;
            int scan = (tied_count < K20) ? K20 : tied_count;
            if (scan > Kmax) scan = Kmax;
            for (int j=0;j<scan;j++) {
                int64_t l1 = pixel_l1(q_img, x_train + (size_t)topi[j]*INPUT_DIM);
                if (l1 < best_l1) { best_l1 = l1; best_label = y_train[topi[j]]; }
            }
            if (best_label==y) partition_aware_correct++;
        }

        /* E5: secondary-hash Hamming re-rank within top-20. */
        {
            const uint8_t* q_sig_B = test_sigs_B + (size_t)s*Sp;
            int32_t best_d = INT32_MAX; int best_label = -1;
            for (int j=0;j<K20;j++) {
                const uint8_t* r_sig = train_sigs_B + (size_t)topi[j]*Sp;
                int32_t d = m4t_popcount_dist(q_sig_B, r_sig, mask, Sp);
                if (d < best_d) { best_d = d; best_label = y_train[topi[j]]; }
            }
            if (best_label==y) secondary_hash_correct++;
        }
    }
    double elapsed = (double)(clock()-t0)/CLOCKS_PER_SEC;

    printf("Inference took %.1f s.\n\n", elapsed);

    printf("Ceiling: correct class in top-%d: %d / %d = %.2f%%\n\n",
           Kmax, correct_in_topKmax, n_test, 100.0*correct_in_topKmax/n_test);

    printf("Per-K results:\n");
    printf("  K    pure-hash-maj    cascade-L1-1NN    cascade-L2-1NN\n");
    for (int ki=0; ki<nK; ki++) {
        int K = Ks[ki];
        printf("  %3d  %6.2f%%         %6.2f%%           %6.2f%%\n",
               K,
               100.0*pure_maj_correct[ki]/n_test,
               100.0*cascade_l1_correct[ki]/n_test,
               100.0*cascade_l2_correct[ki]/n_test);
    }
    printf("\n");

    printf("K=20 fixed variants:\n");
    printf("  pixel-L1 3-NN majority:     %.2f%%\n",
           100.0*cascade_l1_3nn_correct/n_test);
    printf("  E4 partition-aware L1:      %.2f%%\n",
           100.0*partition_aware_correct/n_test);
    printf("  E5 secondary-hash Hamming:  %.2f%%\n",
           100.0*secondary_hash_correct/n_test);
    printf("\n");

    printf("Interpretation:\n");
    printf("  Pure-hash-maj is the original N_PROJ=16 baseline (~62%% at k=7).\n");
    printf("  Cascade-L1/L2 replaces voting with pixel 1-NN on the filtered pool.\n");
    printf("  E5 isolates the 'more bits vs pixel access' question: same\n");
    printf("  architecture but resolver is another 16-bit hash, not pixels.\n");

    free(dists_A);
    free(mask);
    free(train_sigs_A); free(test_sigs_A);
    free(train_sigs_B); free(test_sigs_B);
    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
