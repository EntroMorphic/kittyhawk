/*
 * STATUS: research scaffolding, not production architecture.
 * Runs routing primitives inside an O(N_train) dense outer loop.
 * P1 gate for the meta-router LMM cycle; contingency-table diagnostic.
 * For production routed k-NN use tools/mnist_routed_bucket{,_multi}.c
 * on libglyph. See docs/FINDINGS.md Axis 4c.
 *
 * mnist_local_vs_global.c — prerequisite P1 gate for the meta-router LMM.
 *
 * See journal/meta_router_online_synthesize.md for the full design.
 *
 * Question: on queries where the local routed cascade (H1 filter +
 * H2+H3+H4 fusion over top-K=50) fails, does a global routed cascade
 * (H1+H2+H3+H4 fusion over all 60K prototypes) rescue them?
 *
 * If rescues (pure-local-wrong ∩ global-correct) significantly exceed
 * damages (pure-local-correct ∩ global-wrong), the meta-router has real
 * headroom and we build P2. If the two counts are near-equal, both
 * architectures hit the same ceiling and the meta-router cycle ends
 * as an honest negative.
 *
 * Variants reported per N_PROJ:
 *
 *   L  = local quadruple  (H1 top-50, then H2+H3+H4 summed distance 1-NN)
 *   Gt = global triple    (H2+H3+H4 summed distance over all 60K)
 *   Gq = global quadruple (H1+H2+H3+H4 summed distance over all 60K)
 *
 * Gq is the natural escalation target because it subsumes H1 into the
 * fusion. Gt is reported for contrast.
 *
 * Primary 2x2 contingency is (L vs Gq).
 *
 * Usage: ./mnist_local_vs_global <mnist_dir>
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
#define DENSITY 0.33
#define K_RESOLVE 50

/* ── Shared loaders/deskew/RNG/tau (mirrored from prior tools) ───────── */

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

/* Build one signature set (primary or secondary) at given seed, N_proj,
 * density. Allocates and returns train/test sig buffers. */
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

        free(tmp);
        free(train_proj);
        free(test_proj);
        free(proj_packed);
    }
}

static void eval_one(int N_proj,
                     const m4t_mtfp_t* x_train, int n_train, const int* y_train,
                     const m4t_mtfp_t* x_test,  int n_test,  const int* y_test)
{
    clock_t t0 = clock();
    int Sp=M4T_TRIT_PACKED_BYTES(N_proj);
    uint8_t *trA,*teA,*trB,*teB,*trC,*teC,*trD,*teD;

    build_signature_set(N_proj, x_train, n_train, x_test, n_test,
                        42, 123, 456, 789, DENSITY, &trA, &teA);
    build_signature_set(N_proj, x_train, n_train, x_test, n_test,
                        1337, 2718, 3141, 5923, DENSITY, &trB, &teB);
    build_signature_set(N_proj, x_train, n_train, x_test, n_test,
                        1009, 2017, 3041, 5059, DENSITY, &trC, &teC);
    build_signature_set(N_proj, x_train, n_train, x_test, n_test,
                        9001, 9002, 9003, 9004, DENSITY, &trD, &teD);

    uint8_t* mask=malloc(Sp); memset(mask,0xFF,Sp);

    /* Per-query distances for the primary hash (used by local cascade). */
    int32_t* dA=malloc((size_t)n_train*sizeof(int32_t));

    int L_correct=0, Gt_correct=0, Gq_correct=0;

    /* 2x2 contingency for (L vs Gq). */
    int L_r_Gq_r=0, L_r_Gq_w=0, L_w_Gq_r=0, L_w_Gq_w=0;

    /* Secondary 2x2 for (L vs Gt). */
    int L_r_Gt_r=0, L_r_Gt_w=0, L_w_Gt_r=0, L_w_Gt_w=0;

    for (int s = 0; s < n_test; s++) {
        const uint8_t* qA=teA+(size_t)s*Sp;
        const uint8_t* qB=teB+(size_t)s*Sp;
        const uint8_t* qC=teC+(size_t)s*Sp;
        const uint8_t* qD=teD+(size_t)s*Sp;
        int y = y_test[s];

        /* H1 distances over all 60K (shared between local and both globals). */
        for (int i = 0; i < n_train; i++)
            dA[i] = m4t_popcount_dist(qA, trA+(size_t)i*Sp, mask, Sp);

        /* ── Local quadruple: H1 top-50, then H2+H3+H4 summed 1-NN. ──── */
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
        int L_label = -1;
        {
            int32_t best = INT32_MAX;
            for (int j = 0; j < K_RESOLVE; j++) {
                int idx=topi[j];
                int32_t score =
                    m4t_popcount_dist(qB, trB+(size_t)idx*Sp, mask, Sp) +
                    m4t_popcount_dist(qC, trC+(size_t)idx*Sp, mask, Sp) +
                    m4t_popcount_dist(qD, trD+(size_t)idx*Sp, mask, Sp);
                if (score < best) { best=score; L_label=y_train[idx]; }
            }
        }
        int L_correct_flag = (L_label == y);
        if (L_correct_flag) L_correct++;

        /* ── Global triple: H2+H3+H4 summed over all 60K. ───────────── */
        int Gt_label = -1;
        {
            int32_t best = INT32_MAX;
            for (int i = 0; i < n_train; i++) {
                int32_t score =
                    m4t_popcount_dist(qB, trB+(size_t)i*Sp, mask, Sp) +
                    m4t_popcount_dist(qC, trC+(size_t)i*Sp, mask, Sp) +
                    m4t_popcount_dist(qD, trD+(size_t)i*Sp, mask, Sp);
                if (score < best) { best=score; Gt_label=y_train[i]; }
            }
        }
        int Gt_correct_flag = (Gt_label == y);
        if (Gt_correct_flag) Gt_correct++;

        /* ── Global quadruple: H1+H2+H3+H4 summed over all 60K. ─────── */
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
        int Gq_correct_flag = (Gq_label == y);
        if (Gq_correct_flag) Gq_correct++;

        /* Contingency: L vs Gq. */
        if ( L_correct_flag &&  Gq_correct_flag) L_r_Gq_r++;
        if ( L_correct_flag && !Gq_correct_flag) L_r_Gq_w++;
        if (!L_correct_flag &&  Gq_correct_flag) L_w_Gq_r++;
        if (!L_correct_flag && !Gq_correct_flag) L_w_Gq_w++;

        /* Contingency: L vs Gt. */
        if ( L_correct_flag &&  Gt_correct_flag) L_r_Gt_r++;
        if ( L_correct_flag && !Gt_correct_flag) L_r_Gt_w++;
        if (!L_correct_flag &&  Gt_correct_flag) L_w_Gt_r++;
        if (!L_correct_flag && !Gt_correct_flag) L_w_Gt_w++;
    }

    double secs = (double)(clock()-t0)/CLOCKS_PER_SEC;

    printf("─── N_PROJ = %d ──────────────────────────────────\n", N_proj);
    printf("(%.1fs)\n\n", secs);
    printf("Aggregate accuracy:\n");
    printf("  L  (local  H1 top-50 + H2+H3+H4 1-NN)   %.2f%%\n", 100.0*L_correct/n_test);
    printf("  Gt (global H2+H3+H4   1-NN over 60K)    %.2f%%\n", 100.0*Gt_correct/n_test);
    printf("  Gq (global H1+H2+H3+H4 1-NN over 60K)   %.2f%%\n", 100.0*Gq_correct/n_test);
    printf("\n");

    printf("Contingency (rows = L, cols = Gq):\n");
    printf("                 Gq_right    Gq_wrong\n");
    printf("  L_right        %-8d    %-8d    (L right: %d)\n",
           L_r_Gq_r, L_r_Gq_w, L_r_Gq_r+L_r_Gq_w);
    printf("  L_wrong        %-8d    %-8d    (L wrong: %d)\n",
           L_w_Gq_r, L_w_Gq_w, L_w_Gq_r+L_w_Gq_w);
    printf("\n");
    printf("  Rescues (L_wrong AND Gq_right):  %d  (%.2f%% of queries)\n",
           L_w_Gq_r, 100.0*L_w_Gq_r/n_test);
    printf("  Damages (L_right AND Gq_wrong):  %d  (%.2f%% of queries)\n",
           L_r_Gq_w, 100.0*L_r_Gq_w/n_test);
    printf("  Net Gq - L:                       %+d  (%+.2f%%)\n",
           L_w_Gq_r - L_r_Gq_w,
           100.0*(L_w_Gq_r - L_r_Gq_w)/n_test);
    printf("  Conditional P(Gq correct | L wrong) = %.2f%%\n",
           (L_w_Gq_r+L_w_Gq_w) ? 100.0*L_w_Gq_r/(L_w_Gq_r+L_w_Gq_w) : 0.0);
    printf("\n");

    printf("Contingency (rows = L, cols = Gt, contrast only):\n");
    printf("                 Gt_right    Gt_wrong\n");
    printf("  L_right        %-8d    %-8d\n", L_r_Gt_r, L_r_Gt_w);
    printf("  L_wrong        %-8d    %-8d\n", L_w_Gt_r, L_w_Gt_w);
    printf("  Rescues: %d  Damages: %d  Net: %+d\n",
           L_w_Gt_r, L_r_Gt_w, L_w_Gt_r - L_r_Gt_w);
    printf("\n");

    /* Gate decision. */
    int rescues = L_w_Gq_r;
    int damages = L_r_Gq_w;
    printf("GATE DECISION at N_PROJ=%d:\n", N_proj);
    if (rescues > damages + 10) {
        printf("  PASS — Gq rescues %d queries vs %d damages (net %+d).\n",
               rescues, damages, rescues-damages);
        printf("  Meta-router P2 has architectural headroom. Proceed.\n");
    } else if (rescues > damages) {
        printf("  MARGINAL — Gq rescues %d vs %d damages (net %+d).\n",
               rescues, damages, rescues-damages);
        printf("  Headroom exists but is thin. Reassess P2 scope.\n");
    } else {
        printf("  FAIL — Gq rescues %d vs %d damages (net %+d).\n",
               rescues, damages, rescues-damages);
        printf("  Both architectures hit the same ceiling. Meta-router\n");
        printf("  cannot recover the gap. Cycle ends negative.\n");
    }
    printf("\n");

    free(dA); free(mask);
    free(trA); free(teA);
    free(trB); free(teB);
    free(trC); free(teC);
    free(trD); free(teD);
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

    printf("mnist_local_vs_global — P1 prerequisite gate for meta-router cycle.\n");
    printf("Deskewed MNIST, density=%.2f, K_RESOLVE=%d, seeds H1/H2/H3/H4 fixed.\n",
           DENSITY, K_RESOLVE);
    printf("%d train prototypes, %d test queries.\n\n", n_train, n_test);

    int N_projs[] = {16, 32};
    int nN = (int)(sizeof(N_projs)/sizeof(N_projs[0]));
    for (int k = 0; k < nN; k++) {
        eval_one(N_projs[k], x_train, n_train, y_train, x_test, n_test, y_test);
    }

    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
