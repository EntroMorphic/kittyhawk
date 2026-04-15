/*
 * STATUS: research scaffolding, not production architecture.
 * Runs routing primitives inside an O(N_train) dense outer loop.
 * Atomic decomposition that identified the observability ceiling.
 * For production routed k-NN use tools/mnist_routed_bucket{,_multi}.c
 * on libglyph. See docs/FINDINGS.md Axis 4c.
 *
 * mnist_lvg_atomics.c — atomic decomposition of the local-vs-global
 * contingency at N_PROJ=16.
 *
 * P1 (tools/mnist_local_vs_global.c) reported:
 *   L  = local  quadruple cascade  (H1 top-50, then H2+H3+H4 fusion)   83.86%
 *   Gq = global quadruple          (H1+H2+H3+H4 summed over 60K)       89.46%
 *   rescues  (L wrong, Gq right): 891
 *   damages  (L right, Gq wrong):  331
 *   oracle ceiling (L ∨ Gq right):   92.77%
 *
 * That PASSED the meta-router gate. This probe digs into *what kind of
 * queries* live in each contingency cell, so the P2 meta-router design
 * can pick a routing-context signature that actually separates them.
 *
 * Signals computed per query:
 *   - H1 min distance (the primary filter's confidence)
 *   - H1 tied-min count
 *   - rank of the correct class in H1 top-50 (or "out-of-pool")
 *   - H2-H3 disagreement (do their top-1s within top-50 match?)
 *   - H3-H4 disagreement
 *   - Gq global rank of L's chosen prototype
 *
 * Reports per contingency cell (LR_GR, LR_GW, LW_GR, LW_GW):
 *   A. count
 *   B. H1 min distance histogram
 *   C. tied-min count histogram
 *   D. correct-class rank histogram (where is the right answer in H1's top-50?)
 *   E. H2-H3 disagreement fraction
 *   F. per-class counts
 *   G. top confusion pairs (true, pred) for L and Gq
 *
 * Usage: ./mnist_lvg_atomics <mnist_dir>
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
#define K_RESOLVE 50

/* ── Shared loaders/deskew/RNG/tau ───────────────────────────────────── */

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

/* Contingency cells. */
#define CELL_LR_GR 0  /* L right, Gq right — easy */
#define CELL_LR_GW 1  /* L right, Gq wrong — damage */
#define CELL_LW_GR 2  /* L wrong, Gq right — rescue */
#define CELL_LW_GW 3  /* L wrong, Gq wrong — double fail */
#define N_CELLS 4

static const char* cell_names[N_CELLS] = {
    "LR_GR (both right)",
    "LR_GW (damage)    ",
    "LW_GR (rescue)    ",
    "LW_GW (both wrong)"
};

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
    int32_t* dA=malloc((size_t)n_train*sizeof(int32_t));

    printf("mnist_lvg_atomics — decomposition of the L-vs-Gq contingency\n");
    printf("N_PROJ=%d, density=%.2f, K_RESOLVE=%d, seeds H1/H2/H3/H4 fixed.\n\n",
           N_PROJ, DENSITY, K_RESOLVE);

    /* Counters per cell. */
    int cell_count[N_CELLS] = {0};
    /* H1 min-distance histogram per cell. Max is 2*N_PROJ. */
    #define N_MIND_BUCKETS 8
    int mind_buckets[N_CELLS][N_MIND_BUCKETS] = {{0}};
    /* Tied-count histogram per cell. Buckets: 1, 2-4, 5-15, 16+. */
    #define N_TIED_BUCKETS 4
    int tied_buckets[N_CELLS][N_TIED_BUCKETS] = {{0}};
    /* Correct-rank-in-local-top-50 buckets: 1, 2-5, 6-10, 11-20, 21-50, >50. */
    #define N_RANK_BUCKETS 6
    int rank_buckets[N_CELLS][N_RANK_BUCKETS] = {{0}};
    /* H1-rank where local fusion's winning prototype sits. Same buckets as D
     * minus the "out of pool" bucket since fusion always picks from top-50. */
    #define N_FPICK_BUCKETS 5
    int fpick_buckets[N_CELLS][N_FPICK_BUCKETS] = {{0}};
    /* Fusion margin: (second-best summed distance) - (best summed distance),
     * bucketed. Zero = tied, larger = more confident. */
    #define N_FMARGIN_BUCKETS 5
    int fmargin_buckets[N_CELLS][N_FMARGIN_BUCKETS] = {{0}};
    /* Disagreement counts. */
    int disagree_h2h3[N_CELLS] = {0};
    int disagree_h3h4[N_CELLS] = {0};
    int disagree_h2h4[N_CELLS] = {0};
    /* Per-class count per cell. */
    int class_per_cell[N_CELLS][N_CLASSES] = {{0}};
    /* Confusion counts. */
    int conf_L[N_CLASSES][N_CLASSES] = {{0}};
    int conf_Gq[N_CLASSES][N_CLASSES] = {{0}};
    int conf_rescue_true_L[N_CLASSES][N_CLASSES] = {{0}};
    int conf_damage_true_Gq[N_CLASSES][N_CLASSES] = {{0}};

    clock_t t0 = clock();
    for (int s = 0; s < n_test; s++) {
        const uint8_t* qA=teA+(size_t)s*Sp;
        const uint8_t* qB=teB+(size_t)s*Sp;
        const uint8_t* qC=teC+(size_t)s*Sp;
        const uint8_t* qD=teD+(size_t)s*Sp;
        int y = y_test[s];

        /* Global H1 pass. */
        for (int i = 0; i < n_train; i++)
            dA[i] = m4t_popcount_dist(qA, trA+(size_t)i*Sp, mask, Sp);

        /* Top-50 by H1. */
        int32_t topd[K_RESOLVE]; int topi[K_RESOLVE];
        for(int j=0;j<K_RESOLVE;j++){ topd[j]=INT32_MAX; topi[j]=-1; }
        for(int i=0;i<n_train;i++){
            int32_t d=dA[i];
            if(d>=topd[K_RESOLVE-1]) continue;
            topd[K_RESOLVE-1]=d; topi[K_RESOLVE-1]=i;
            for(int j=K_RESOLVE-2;j>=0;j--){
                if(topd[j+1]<topd[j]){
                    int32_t td=topd[j]; topd[j]=topd[j+1]; topd[j+1]=td;
                    int ti=topi[j]; topi[j]=topi[j+1]; topi[j+1]=ti;
                } else break;
            }
        }

        /* Signals. */
        int min_d = topd[0];
        int tied_count = 0;
        for (int i = 0; i < n_train; i++) if (dA[i] == min_d) tied_count++;

        /* Correct-class rank in local top-50 (0-based, -1 if not in pool). */
        int correct_rank = -1;
        for (int j = 0; j < K_RESOLVE; j++) {
            if (y_train[topi[j]] == y) { correct_rank = j; break; }
        }

        /* Local quadruple prediction: H2+H3+H4 fusion over top-50.
         * Also records H1-rank of fusion's pick and fusion margin. */
        int L_label = -1;
        int L_pick_rank = -1;
        int32_t L_best = INT32_MAX, L_second = INT32_MAX;
        {
            for (int j = 0; j < K_RESOLVE; j++) {
                int idx=topi[j];
                int32_t score =
                    m4t_popcount_dist(qB, trB+(size_t)idx*Sp, mask, Sp) +
                    m4t_popcount_dist(qC, trC+(size_t)idx*Sp, mask, Sp) +
                    m4t_popcount_dist(qD, trD+(size_t)idx*Sp, mask, Sp);
                if (score < L_best) {
                    L_second = L_best;
                    L_best = score;
                    L_label = y_train[idx];
                    L_pick_rank = j;
                } else if (score < L_second) {
                    L_second = score;
                }
            }
        }
        int L_margin = (L_second < INT32_MAX) ? (L_second - L_best) : 0;

        /* H2 top-1 in local pool, H3 top-1, H4 top-1 (for disagreement). */
        int h2_top=-1, h3_top=-1, h4_top=-1;
        {
            int32_t bB=INT32_MAX, bC=INT32_MAX, bD=INT32_MAX;
            for (int j = 0; j < K_RESOLVE; j++) {
                int idx=topi[j];
                int32_t rB = m4t_popcount_dist(qB, trB+(size_t)idx*Sp, mask, Sp);
                int32_t rC = m4t_popcount_dist(qC, trC+(size_t)idx*Sp, mask, Sp);
                int32_t rD = m4t_popcount_dist(qD, trD+(size_t)idx*Sp, mask, Sp);
                if (rB < bB) { bB=rB; h2_top=y_train[idx]; }
                if (rC < bC) { bC=rC; h3_top=y_train[idx]; }
                if (rD < bD) { bD=rD; h4_top=y_train[idx]; }
            }
        }

        /* Global quadruple prediction. */
        int Gq_label = -1;
        {
            int32_t best = INT32_MAX;
            for (int i = 0; i < n_train; i++) {
                int32_t score = dA[i] +
                    m4t_popcount_dist(qB, trB+(size_t)i*Sp, mask, Sp) +
                    m4t_popcount_dist(qC, trC+(size_t)i*Sp, mask, Sp) +
                    m4t_popcount_dist(qD, trD+(size_t)i*Sp, mask, Sp);
                if (score < best) { best=score; Gq_label=y_train[i]; }
            }
        }

        int L_right  = (L_label == y);
        int Gq_right = (Gq_label == y);
        int cell = (L_right && Gq_right) ? CELL_LR_GR
                 : (L_right && !Gq_right) ? CELL_LR_GW
                 : (!L_right && Gq_right) ? CELL_LW_GR
                 :                          CELL_LW_GW;
        cell_count[cell]++;

        /* H1 min-distance bucket. Max distance is 2*N_PROJ = 32. */
        {
            int b;
            if      (min_d <= 1)  b = 0;
            else if (min_d <= 2)  b = 1;
            else if (min_d <= 4)  b = 2;
            else if (min_d <= 6)  b = 3;
            else if (min_d <= 10) b = 4;
            else if (min_d <= 14) b = 5;
            else if (min_d <= 20) b = 6;
            else                  b = 7;
            mind_buckets[cell][b]++;
        }
        /* Tied-count bucket. */
        {
            int b;
            if      (tied_count <= 1)  b = 0;
            else if (tied_count <= 4)  b = 1;
            else if (tied_count <= 15) b = 2;
            else                       b = 3;
            tied_buckets[cell][b]++;
        }
        /* Correct-rank bucket. */
        {
            int b;
            if      (correct_rank < 0)  b = 5;  /* out of pool */
            else if (correct_rank == 0) b = 0;
            else if (correct_rank <= 4) b = 1;
            else if (correct_rank <= 9) b = 2;
            else if (correct_rank <= 19)b = 3;
            else                        b = 4;
            rank_buckets[cell][b]++;
        }
        /* Ensemble disagreement (do the heads pick the same top-1?). */
        if (h2_top != h3_top) disagree_h2h3[cell]++;
        if (h3_top != h4_top) disagree_h3h4[cell]++;
        if (h2_top != h4_top) disagree_h2h4[cell]++;

        /* Fusion-pick H1-rank bucket. */
        {
            int b;
            if      (L_pick_rank == 0)  b = 0;
            else if (L_pick_rank <= 4)  b = 1;
            else if (L_pick_rank <= 9)  b = 2;
            else if (L_pick_rank <= 19) b = 3;
            else                        b = 4;
            fpick_buckets[cell][b]++;
        }
        /* Fusion margin bucket. */
        {
            int b;
            if      (L_margin == 0) b = 0;
            else if (L_margin <= 2) b = 1;
            else if (L_margin <= 5) b = 2;
            else if (L_margin <= 10)b = 3;
            else                    b = 4;
            fmargin_buckets[cell][b]++;
        }

        /* Per-class and confusion. */
        class_per_cell[cell][y]++;
        conf_L[y][L_label]++;
        conf_Gq[y][Gq_label]++;
        if (cell == CELL_LW_GR) conf_rescue_true_L[y][L_label]++;
        if (cell == CELL_LR_GW) conf_damage_true_Gq[y][Gq_label]++;
    }
    double secs = (double)(clock()-t0)/CLOCKS_PER_SEC;

    /* ── Reports ─────────────────────────────────────────────────────── */

    printf("Inference: %.1fs for %d queries.\n\n", secs, n_test);

    printf("A. Cell counts:\n");
    for (int c = 0; c < N_CELLS; c++)
        printf("  %-20s  %5d  (%.2f%%)\n",
               cell_names[c], cell_count[c], 100.0*cell_count[c]/n_test);
    printf("\n");

    printf("B. H1 min-distance histogram per cell:\n");
    printf("  cell              d<=1   d<=2   d<=4   d<=6   d<=10  d<=14  d<=20  d>20\n");
    for (int c = 0; c < N_CELLS; c++) {
        printf("  %-16s", cell_names[c]);
        int total = cell_count[c]; if (total==0) total=1;
        for (int b = 0; b < N_MIND_BUCKETS; b++)
            printf("  %5.1f%%", 100.0*mind_buckets[c][b]/total);
        printf("\n");
    }
    printf("\n");

    printf("C. H1 tied-min-count histogram per cell:\n");
    printf("  cell              tied=1  tied=2-4  tied=5-15  tied=16+\n");
    for (int c = 0; c < N_CELLS; c++) {
        printf("  %-16s", cell_names[c]);
        int total = cell_count[c]; if (total==0) total=1;
        for (int b = 0; b < N_TIED_BUCKETS; b++)
            printf("  %7.2f%%", 100.0*tied_buckets[c][b]/total);
        printf("\n");
    }
    printf("\n");

    printf("D. Correct-class rank in H1 top-50 per cell:\n");
    printf("  cell              rank=1   2-5    6-10   11-20  21-50  >50\n");
    for (int c = 0; c < N_CELLS; c++) {
        printf("  %-16s", cell_names[c]);
        int total = cell_count[c]; if (total==0) total=1;
        for (int b = 0; b < N_RANK_BUCKETS; b++)
            printf("  %6.2f%%", 100.0*rank_buckets[c][b]/total);
        printf("\n");
    }
    printf("\n");

    printf("E. Ensemble top-1 disagreement rates per cell:\n");
    printf("  cell              H2-H3   H3-H4   H2-H4\n");
    for (int c = 0; c < N_CELLS; c++) {
        int total = cell_count[c]; if (total==0) total=1;
        printf("  %-16s  %6.2f%%  %6.2f%%  %6.2f%%\n",
               cell_names[c],
               100.0*disagree_h2h3[c]/total,
               100.0*disagree_h3h4[c]/total,
               100.0*disagree_h2h4[c]/total);
    }
    printf("\n");

    printf("E2. H1-rank where local fusion's winning prototype sits:\n");
    printf("  cell              rank=1   2-5    6-10   11-20  21-50\n");
    for (int c = 0; c < N_CELLS; c++) {
        printf("  %-16s", cell_names[c]);
        int total = cell_count[c]; if (total==0) total=1;
        for (int b = 0; b < N_FPICK_BUCKETS; b++)
            printf("  %6.2f%%", 100.0*fpick_buckets[c][b]/total);
        printf("\n");
    }
    printf("\n");

    printf("E3. Fusion margin (second-best summed - best summed):\n");
    printf("  cell              m=0    m<=2    m<=5    m<=10   m>10\n");
    for (int c = 0; c < N_CELLS; c++) {
        printf("  %-16s", cell_names[c]);
        int total = cell_count[c]; if (total==0) total=1;
        for (int b = 0; b < N_FMARGIN_BUCKETS; b++)
            printf("  %6.2f%%", 100.0*fmargin_buckets[c][b]/total);
        printf("\n");
    }
    printf("\n");

    printf("F. Per-class counts per cell:\n");
    printf("  class:");
    for (int k = 0; k < N_CLASSES; k++) printf("  %4d", k);
    printf("    total\n");
    for (int c = 0; c < N_CELLS; c++) {
        printf("  %-16s", cell_names[c]);
        int total = 0;
        for (int k = 0; k < N_CLASSES; k++) { printf("  %4d", class_per_cell[c][k]); total += class_per_cell[c][k]; }
        printf("    %5d\n", total);
    }
    printf("\n");

    printf("G. Top class-pair confusions for L (rows=true, cols=L_pred).\n");
    printf("  Showing top 10 off-diagonal (true,pred) pairs by L error count:\n");
    {
        typedef struct { int t,p,count; } cf_t;
        cf_t all[N_CLASSES*N_CLASSES]; int n_all=0;
        for (int t=0;t<N_CLASSES;t++) for (int p=0;p<N_CLASSES;p++)
            if (t!=p && conf_L[t][p] > 0) {
                all[n_all].t=t; all[n_all].p=p; all[n_all].count=conf_L[t][p]; n_all++;
            }
        /* selection sort top-10 */
        for (int sel=0; sel<10 && sel<n_all; sel++) {
            int best=sel;
            for (int i=sel+1;i<n_all;i++) if (all[i].count>all[best].count) best=i;
            cf_t tmp=all[sel]; all[sel]=all[best]; all[best]=tmp;
        }
        printf("  true pred  L_err  Gq_err  delta(L-Gq)\n");
        int shown = (n_all<10)?n_all:10;
        for (int i=0;i<shown;i++)
            printf("   %d    %d    %4d   %4d    %+d\n",
                   all[i].t, all[i].p, all[i].count, conf_Gq[all[i].t][all[i].p],
                   all[i].count - conf_Gq[all[i].t][all[i].p]);
    }
    printf("\n");

    printf("H. Rescue-specific confusions (L wrong, Gq right): what did L say?\n");
    printf("  Showing top 10 (true, L_pred) pairs within rescues:\n");
    {
        typedef struct { int t,p,count; } cf_t;
        cf_t all[N_CLASSES*N_CLASSES]; int n_all=0;
        for (int t=0;t<N_CLASSES;t++) for (int p=0;p<N_CLASSES;p++)
            if (t!=p && conf_rescue_true_L[t][p] > 0) {
                all[n_all].t=t; all[n_all].p=p; all[n_all].count=conf_rescue_true_L[t][p]; n_all++;
            }
        for (int sel=0; sel<10 && sel<n_all; sel++) {
            int best=sel;
            for (int i=sel+1;i<n_all;i++) if (all[i].count>all[best].count) best=i;
            cf_t tmp=all[sel]; all[sel]=all[best]; all[best]=tmp;
        }
        printf("  true  L_pred   rescued_count\n");
        int shown = (n_all<10)?n_all:10;
        for (int i=0;i<shown;i++)
            printf("   %d     %d        %4d\n", all[i].t, all[i].p, all[i].count);
    }
    printf("\n");

    printf("I. Damage-specific confusions (L right, Gq wrong): what did Gq say?\n");
    printf("  Showing top 10 (true, Gq_pred) pairs within damages:\n");
    {
        typedef struct { int t,p,count; } cf_t;
        cf_t all[N_CLASSES*N_CLASSES]; int n_all=0;
        for (int t=0;t<N_CLASSES;t++) for (int p=0;p<N_CLASSES;p++)
            if (t!=p && conf_damage_true_Gq[t][p] > 0) {
                all[n_all].t=t; all[n_all].p=p; all[n_all].count=conf_damage_true_Gq[t][p]; n_all++;
            }
        for (int sel=0; sel<10 && sel<n_all; sel++) {
            int best=sel;
            for (int i=sel+1;i<n_all;i++) if (all[i].count>all[best].count) best=i;
            cf_t tmp=all[sel]; all[sel]=all[best]; all[best]=tmp;
        }
        printf("  true  Gq_pred  damaged_count\n");
        int shown = (n_all<10)?n_all:10;
        for (int i=0;i<shown;i++)
            printf("   %d     %d        %4d\n", all[i].t, all[i].p, all[i].count);
    }
    printf("\n");

    printf("Interpretation:\n");
    printf("  A gives the P1 contingency counts (for reference).\n");
    printf("  B, C, D test whether H1's own confidence signals separate\n");
    printf("    rescues from damages. If the rescue distribution skews\n");
    printf("    toward loose H1 (high min_d, high tied_count, deeper correct\n");
    printf("    rank) and damage skews toward tight H1, a single-signal\n");
    printf("    meta-router could work. If not, P2 needs multi-signal context.\n");
    printf("  E tests ensemble disagreement as a rescue-detection signal.\n");
    printf("  F-I identify which digits and which confusion pairs carry\n");
    printf("    the rescue/damage mass.\n");

    free(dA); free(mask);
    free(trA); free(teA); free(trB); free(teB);
    free(trC); free(teC); free(trD); free(teD);
    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
