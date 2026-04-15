/*
 * mnist_resolver_sweep.c — routed resolver sweep over the cascade.
 *
 * Sixth-round remediation removes the dense resolver family and replaces it
 * with routed resolver variants over the same filtered top-K candidate pool.
 * The question becomes: which routed secondary view best resolves the
 * primary-hash shortlist?
 *
 * Resolvers evaluated on filtered top-K candidates (primary hash sorted):
 *
 *   R1: secondary-hash 1-NN
 *   R2: dual-hash (primary+secondary) 1-NN     <- baseline
 *   R3: secondary-hash 3-NN majority
 *   R4: secondary-hash 5-NN majority
 *   R5: secondary-hash 7-NN majority
 *   R6: secondary-hash 5-NN rank-weighted
 *   R7: secondary-hash 5-NN distance-weighted
 *   R8: tertiary-hash 1-NN
 *   R9: dual secondary+tertiary 1-NN
 *   R10: per-class nearest secondary-hash
 *   R11: primary+secondary rank hybrid
 *
 * Run at N_PROJ in {16,128,1024}.
 *
 * Usage: ./mnist_resolver_sweep <mnist_dir>
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
#define DENSITY_D50 0.50
#define DENSITY_D20 0.20
#define K_RESOLVE 50
#define N_RESOLVERS 15

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

static const char* resolver_names[N_RESOLVERS] = {
    "H2 1-NN              ",
    "H1+H2 1-NN (base)    ",
    "H2 3-NN majority     ",
    "H2 5-NN majority     ",
    "H2 7-NN majority     ",
    "H2 5-NN rank-wt      ",
    "H2 5-NN dist-wt      ",
    "H3 1-NN              ",
    "H2+H3 1-NN           ",
    "per-class nearest H2 ",
    "H1+H2 rank hybrid    ",
    "H2+H3+H4 1-NN        ",
    "H_D50 1-NN           ",
    "H_D20 1-NN           ",
    "H2+H_D50 1-NN        "
};

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

static int eval_n_proj(int N_proj,
                       const m4t_mtfp_t* x_train, int n_train, const int* y_train,
                       const m4t_mtfp_t* x_test,  int n_test,  const int* y_test,
                       int* out_correct_per_resolver,
                       int* out_confusion_3_8,
                       int* out_confusion_3_5,
                       int* out_confusion_6_8,
                       int* out_pure_top1,
                       int* out_ceiling_at_K,
                       double* out_seconds)
{
    clock_t t0 = clock();
    int Sp=M4T_TRIT_PACKED_BYTES(N_proj);
    uint8_t *train_sigs_A, *test_sigs_A, *train_sigs_B, *test_sigs_B, *train_sigs_C, *test_sigs_C;
    uint8_t *train_sigs_D, *test_sigs_D, *train_sigs_E, *test_sigs_E, *train_sigs_F, *test_sigs_F;
    uint8_t* mask;
    int32_t* dists;

    /* H1 primary, H2/H3 standard-density secondaries, H4 fourth seed,
     * H_D50 high-density seed, H_D20 low-density seed. */
    build_signature_set(N_proj, x_train, n_train, x_test, n_test,
                        42, 123, 456, 789, DENSITY, &train_sigs_A, &test_sigs_A);
    build_signature_set(N_proj, x_train, n_train, x_test, n_test,
                        1337, 2718, 3141, 5923, DENSITY, &train_sigs_B, &test_sigs_B);
    build_signature_set(N_proj, x_train, n_train, x_test, n_test,
                        1009, 2017, 3041, 5059, DENSITY, &train_sigs_C, &test_sigs_C);
    build_signature_set(N_proj, x_train, n_train, x_test, n_test,
                        9001, 9002, 9003, 9004, DENSITY, &train_sigs_D, &test_sigs_D);
    build_signature_set(N_proj, x_train, n_train, x_test, n_test,
                        5555, 6666, 7777, 8888, DENSITY_D50, &train_sigs_E, &test_sigs_E);
    build_signature_set(N_proj, x_train, n_train, x_test, n_test,
                        3141, 5926, 5358, 9793, DENSITY_D20, &train_sigs_F, &test_sigs_F);

    mask=malloc(Sp); memset(mask,0xFF,Sp);
    dists=malloc((size_t)n_train*sizeof(int32_t));

    memset(out_correct_per_resolver, 0, N_RESOLVERS*sizeof(int));
    memset(out_confusion_3_8, 0, N_RESOLVERS*sizeof(int));
    memset(out_confusion_3_5, 0, N_RESOLVERS*sizeof(int));
    memset(out_confusion_6_8, 0, N_RESOLVERS*sizeof(int));
    *out_pure_top1 = 0;
    *out_ceiling_at_K = 0;

    for (int s = 0; s < n_test; s++) {
        const uint8_t* q_sig_A = test_sigs_A + (size_t)s*Sp;
        const uint8_t* q_sig_B = test_sigs_B + (size_t)s*Sp;
        const uint8_t* q_sig_C = test_sigs_C + (size_t)s*Sp;
        const uint8_t* q_sig_D = test_sigs_D + (size_t)s*Sp;
        const uint8_t* q_sig_E = test_sigs_E + (size_t)s*Sp;
        const uint8_t* q_sig_F = test_sigs_F + (size_t)s*Sp;
        int y = y_test[s];
        int32_t topd[K_RESOLVE];
        int topi[K_RESOLVE];
        int32_t dB[K_RESOLVE], dC[K_RESOLVE], dD[K_RESOLVE], dE[K_RESOLVE], dF[K_RESOLVE];
        int pLbl[K_RESOLVE];
        int rankA[K_RESOLVE], rankB[K_RESOLVE];
        int byB[K_RESOLVE];
        int kcounts[N_CLASSES];

        for (int i = 0; i < n_train; i++) {
            const uint8_t* r_sig_A = train_sigs_A + (size_t)i*Sp;
            dists[i] = m4t_popcount_dist(q_sig_A, r_sig_A, mask, Sp);
        }

        for(int j=0;j<K_RESOLVE;j++){ topd[j]=INT32_MAX; topi[j]=-1; }
        for(int i=0;i<n_train;i++){
            int32_t d=dists[i];
            if(d>=topd[K_RESOLVE-1]) continue;
            topd[K_RESOLVE-1]=d; topi[K_RESOLVE-1]=i;
            for(int j=K_RESOLVE-2;j>=0;j--){
                if(topd[j+1]<topd[j]){
                    int32_t td=topd[j]; topd[j]=topd[j+1]; topd[j+1]=td;
                    int ti=topi[j]; topi[j]=topi[j+1]; topi[j+1]=ti;
                } else break;
            }
        }

        if (y_train[topi[0]] == y) (*out_pure_top1)++;
        for (int j=0;j<K_RESOLVE;j++) if (y_train[topi[j]]==y){(*out_ceiling_at_K)++; break;}

        for (int j = 0; j < K_RESOLVE; j++) {
            const uint8_t* rB = train_sigs_B + (size_t)topi[j]*Sp;
            const uint8_t* rC = train_sigs_C + (size_t)topi[j]*Sp;
            const uint8_t* rD = train_sigs_D + (size_t)topi[j]*Sp;
            const uint8_t* rE = train_sigs_E + (size_t)topi[j]*Sp;
            const uint8_t* rF = train_sigs_F + (size_t)topi[j]*Sp;
            dB[j] = m4t_popcount_dist(q_sig_B, rB, mask, Sp);
            dC[j] = m4t_popcount_dist(q_sig_C, rC, mask, Sp);
            dD[j] = m4t_popcount_dist(q_sig_D, rD, mask, Sp);
            dE[j] = m4t_popcount_dist(q_sig_E, rE, mask, Sp);
            dF[j] = m4t_popcount_dist(q_sig_F, rF, mask, Sp);
            pLbl[j] = y_train[topi[j]];
            rankA[j] = j;
            byB[j] = j;
        }

        for (int a=1;a<K_RESOLVE;a++) {
            int k=byB[a]; int32_t kv=dB[k]; int b=a-1;
            while(b>=0 && dB[byB[b]]>kv){ byB[b+1]=byB[b]; b--; }
            byB[b+1]=k;
        }
        for (int a=0;a<K_RESOLVE;a++) rankB[byB[a]] = a;

        /* R1: H2 1-NN. */
        {
            int32_t b=INT32_MAX; int bl=-1;
            for(int j=0;j<K_RESOLVE;j++) if(dB[j]<b){b=dB[j];bl=pLbl[j];}
            if(bl==y) out_correct_per_resolver[0]++;
        }
        /* R2: H1+H2 1-NN baseline. */
        {
            int32_t b=INT32_MAX; int bl=-1;
            for(int j=0;j<K_RESOLVE;j++) {
                int32_t score = topd[j] + dB[j];
                if(score < b){ b=score; bl=pLbl[j]; }
            }
            if(bl==y) out_correct_per_resolver[1]++;
        }
        /* R3/R4/R5: H2 k-NN majority. */
        for (int kchoice = 0; kchoice < 3; kchoice++) {
            int kk = (kchoice==0)?3:(kchoice==1)?5:7;
            memset(kcounts,0,sizeof(kcounts));
            for (int j=0;j<kk;j++) kcounts[pLbl[byB[j]]]++;
            {
                int p=0; for(int c=1;c<N_CLASSES;c++) if(kcounts[c]>kcounts[p]) p=c;
                if (p==y) out_correct_per_resolver[2+kchoice]++;
            }
        }
        /* R6: H2 5-NN rank-weighted. */
        {
            int rw[N_CLASSES]={0};
            for(int j=0;j<5;j++) rw[pLbl[byB[j]]] += (5-j);
            int p=0; for(int c=1;c<N_CLASSES;c++) if(rw[c]>rw[p]) p=c;
            if (p==y) out_correct_per_resolver[5]++;
        }
        /* R7: H2 5-NN distance-weighted. */
        {
            int dw[N_CLASSES]={0};
            int32_t max_dist = 2 * N_proj;
            for(int j=0;j<5;j++) dw[pLbl[byB[j]]] += (max_dist - dB[byB[j]]);
            int p=0; for(int c=1;c<N_CLASSES;c++) if(dw[c]>dw[p]) p=c;
            if (p==y) out_correct_per_resolver[6]++;
        }
        /* R8: H3 1-NN. */
        {
            int32_t b=INT32_MAX; int bl=-1;
            for(int j=0;j<K_RESOLVE;j++) if(dC[j]<b){b=dC[j];bl=pLbl[j];}
            if (bl==y) out_correct_per_resolver[7]++;
        }
        /* R9: H2+H3 1-NN. */
        {
            int32_t b=INT32_MAX; int bl=-1;
            for(int j=0;j<K_RESOLVE;j++) {
                int32_t score = dB[j] + dC[j];
                if(score < b){ b=score; bl=pLbl[j]; }
            }
            if (bl==y) out_correct_per_resolver[8]++;
        }
        /* R10: per-class nearest H2 within top-K. */
        {
            int32_t class_best[N_CLASSES];
            for (int c=0;c<N_CLASSES;c++) class_best[c] = INT32_MAX;
            for (int j=0;j<K_RESOLVE;j++)
                if (dB[j] < class_best[pLbl[j]]) class_best[pLbl[j]] = dB[j];
            {
                int32_t b=INT32_MAX; int bl=-1;
                for (int c=0;c<N_CLASSES;c++)
                    if (class_best[c] < b) { b = class_best[c]; bl = c; }
                if (bl==y) out_correct_per_resolver[9]++;
            }
        }
        /* R11: H1+H2 rank hybrid. */
        {
            int b_score = 1<<30; int bl = -1;
            for (int j=0;j<K_RESOLVE;j++) {
                int score = rankA[j] + rankB[j];
                if (score < b_score) { b_score=score; bl=pLbl[j]; }
            }
            if (bl==y) out_correct_per_resolver[10]++;
        }
        /* R12: H2+H3+H4 1-NN (triple secondary-hash fusion). */
        {
            int32_t b=INT32_MAX; int bl=-1;
            for(int j=0;j<K_RESOLVE;j++) {
                int32_t score = dB[j] + dC[j] + dD[j];
                if(score < b){ b=score; bl=pLbl[j]; }
            }
            if (bl==y) out_correct_per_resolver[11]++;
            if (y==3 && bl==8) out_confusion_3_8[11]++;
            if (y==3 && bl==5) out_confusion_3_5[11]++;
            if (y==6 && bl==8) out_confusion_6_8[11]++;
        }
        /* R13: H_D50 1-NN (new seed, density 0.50). */
        {
            int32_t b=INT32_MAX; int bl=-1;
            for(int j=0;j<K_RESOLVE;j++) if(dE[j]<b){b=dE[j];bl=pLbl[j];}
            if (bl==y) out_correct_per_resolver[12]++;
            if (y==3 && bl==8) out_confusion_3_8[12]++;
            if (y==3 && bl==5) out_confusion_3_5[12]++;
            if (y==6 && bl==8) out_confusion_6_8[12]++;
        }
        /* R14: H_D20 1-NN (new seed, density 0.20). */
        {
            int32_t b=INT32_MAX; int bl=-1;
            for(int j=0;j<K_RESOLVE;j++) if(dF[j]<b){b=dF[j];bl=pLbl[j];}
            if (bl==y) out_correct_per_resolver[13]++;
            if (y==3 && bl==8) out_confusion_3_8[13]++;
            if (y==3 && bl==5) out_confusion_3_5[13]++;
            if (y==6 && bl==8) out_confusion_6_8[13]++;
        }
        /* R15: H2+H_D50 1-NN (dual-density fusion). */
        {
            int32_t b=INT32_MAX; int bl=-1;
            for(int j=0;j<K_RESOLVE;j++) {
                int32_t score = dB[j] + dE[j];
                if(score < b){ b=score; bl=pLbl[j]; }
            }
            if (bl==y) out_correct_per_resolver[14]++;
            if (y==3 && bl==8) out_confusion_3_8[14]++;
            if (y==3 && bl==5) out_confusion_3_5[14]++;
            if (y==6 && bl==8) out_confusion_6_8[14]++;
        }

        /* Confusion tracking for original R1-R11 (recompute winner labels
         * with inline logic; cheap since we only need the final predicted
         * label for the three pair counters). */
        {
            int pred[11];
            /* R1: H2 1-NN */
            {
                int32_t b=INT32_MAX; int bl=-1;
                for(int j=0;j<K_RESOLVE;j++) if(dB[j]<b){b=dB[j];bl=pLbl[j];}
                pred[0]=bl;
            }
            /* R2: H1+H2 1-NN */
            {
                int32_t b=INT32_MAX; int bl=-1;
                for(int j=0;j<K_RESOLVE;j++) {
                    int32_t score = topd[j] + dB[j];
                    if(score < b){ b=score; bl=pLbl[j]; }
                }
                pred[1]=bl;
            }
            /* R3-R5: H2 k-NN majority */
            for (int kchoice=0;kchoice<3;kchoice++) {
                int kk = (kchoice==0)?3:(kchoice==1)?5:7;
                int cc[N_CLASSES]={0};
                for (int j=0;j<kk;j++) cc[pLbl[byB[j]]]++;
                int p=0; for(int c=1;c<N_CLASSES;c++) if(cc[c]>cc[p]) p=c;
                pred[2+kchoice]=p;
            }
            /* R6: H2 5-NN rank-wt */
            {
                int rw[N_CLASSES]={0};
                for(int j=0;j<5;j++) rw[pLbl[byB[j]]] += (5-j);
                int p=0; for(int c=1;c<N_CLASSES;c++) if(rw[c]>rw[p]) p=c;
                pred[5]=p;
            }
            /* R7: H2 5-NN dist-wt */
            {
                int dw[N_CLASSES]={0};
                int32_t max_dist = 2*N_proj;
                for(int j=0;j<5;j++) dw[pLbl[byB[j]]] += (max_dist - dB[byB[j]]);
                int p=0; for(int c=1;c<N_CLASSES;c++) if(dw[c]>dw[p]) p=c;
                pred[6]=p;
            }
            /* R8: H3 1-NN */
            {
                int32_t b=INT32_MAX; int bl=-1;
                for(int j=0;j<K_RESOLVE;j++) if(dC[j]<b){b=dC[j];bl=pLbl[j];}
                pred[7]=bl;
            }
            /* R9: H2+H3 1-NN */
            {
                int32_t b=INT32_MAX; int bl=-1;
                for(int j=0;j<K_RESOLVE;j++) {
                    int32_t score = dB[j]+dC[j];
                    if(score < b){ b=score; bl=pLbl[j]; }
                }
                pred[8]=bl;
            }
            /* R10: per-class nearest H2 */
            {
                int32_t class_best[N_CLASSES];
                for (int c=0;c<N_CLASSES;c++) class_best[c] = INT32_MAX;
                for (int j=0;j<K_RESOLVE;j++)
                    if (dB[j] < class_best[pLbl[j]]) class_best[pLbl[j]] = dB[j];
                int32_t b=INT32_MAX; int bl=-1;
                for (int c=0;c<N_CLASSES;c++)
                    if (class_best[c] < b) { b=class_best[c]; bl=c; }
                pred[9]=bl;
            }
            /* R11: H1+H2 rank hybrid */
            {
                int b_score=1<<30; int bl=-1;
                for (int j=0;j<K_RESOLVE;j++) {
                    int score = rankA[j] + rankB[j];
                    if (score < b_score) { b_score=score; bl=pLbl[j]; }
                }
                pred[10]=bl;
            }
            for (int r = 0; r < 11; r++) {
                if (y==3 && pred[r]==8) out_confusion_3_8[r]++;
                if (y==3 && pred[r]==5) out_confusion_3_5[r]++;
                if (y==6 && pred[r]==8) out_confusion_6_8[r]++;
            }
        }
    }

    free(dists); free(mask);
    free(train_sigs_A); free(test_sigs_A);
    free(train_sigs_B); free(test_sigs_B);
    free(train_sigs_C); free(test_sigs_C);
    free(train_sigs_D); free(test_sigs_D);
    free(train_sigs_E); free(test_sigs_E);
    free(train_sigs_F); free(test_sigs_F);
    *out_seconds = (double)(clock()-t0)/CLOCKS_PER_SEC;
    return n_test;
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

    {
        int N_projs[] = {16, 128, 1024};
        int nN = (int)(sizeof(N_projs)/sizeof(N_projs[0]));

        printf("Routed resolver sweep: K_RESOLVE=%d, density=%.2f, seeds=(42,1337,1009), deskewed MNIST.\n\n",
               K_RESOLVE, DENSITY);

        for (int k = 0; k < nN; k++) {
            int N_proj = N_projs[k];
            int correct[N_RESOLVERS];
            int conf_3_8[N_RESOLVERS], conf_3_5[N_RESOLVERS], conf_6_8[N_RESOLVERS];
            int pure_top1, ceiling_at_K;
            double secs;
            int n = eval_n_proj(N_proj, x_train, n_train, y_train, x_test, n_test, y_test,
                                correct, conf_3_8, conf_3_5, conf_6_8,
                                &pure_top1, &ceiling_at_K, &secs);
            printf("--- N_PROJ = %d --------------------------------------\n", N_proj);
            printf("pure top-1:  %.2f%%    ceiling@%d:  %.2f%%    (%.1fs)\n",
                   100.0*pure_top1/n, K_RESOLVE, 100.0*ceiling_at_K/n, secs);
            printf("  #   resolver                    accuracy    Delta vs H1+H2    3->8  3->5  6->8\n");
            {
                double baseline = 100.0*correct[1]/n;
                for (int r = 0; r < N_RESOLVERS; r++) {
                    double acc = 100.0*correct[r]/n;
                    double delta = acc - baseline;
                    char marker = (acc > baseline + 0.005) ? '+' : (acc < baseline - 0.005) ? '-' : '=';
                    printf("  R%-2d %s   %6.2f%%    %c %+.2f%%        %4d  %4d  %4d\n",
                           r+1, resolver_names[r], acc, marker, delta,
                           conf_3_8[r], conf_3_5[r], conf_6_8[r]);
                }
            }
            printf("\n");
            fflush(stdout);
        }
    }

    printf("Interpretation:\n");
    printf("  Baseline is R2 (primary+secondary dual-hash 1-NN).\n");
    printf("  Any resolver exceeding R2 at high N_PROJ improves the fully routed cascade.\n");

    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
