/*
 * mnist_cascade_sweep.c — cascade vs pure-hash across N_PROJ.
 *
 * Verifies the prediction from cascade atomics: cascade headroom shrinks
 * as N_PROJ grows, with crossover around N_PROJ≈256.
 *
 * For each N_PROJ in {8,16,32,64,128,256,512,1024,4096}:
 *   - pure-hash k=7 majority accuracy
 *   - pure-hash top-1 accuracy
 *   - ceiling at K=50 (correct class in top-50)
 *   - cascade K=50 pixel-L2 1-NN accuracy
 *   - Δ (cascade - pure-maj)
 *
 * Single seed (42) per N_PROJ — mechanism, not ±σ.
 *
 * Usage: ./mnist_cascade_sweep <mnist_dir>
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

/* ── Loaders, deskew, RNG, τ, pixel-L2 (mirrored from prior tools) ───── */

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
static int64_t pixel_l2sq(const m4t_mtfp_t* a, const m4t_mtfp_t* b) {
    int64_t s=0;
    for(int i=0;i<INPUT_DIM;i++){
        int32_t d=(int32_t)a[i]-(int32_t)b[i];
        s += (int64_t)d*d;
    }
    return s;
}

/* ── One-point evaluation at a given N_PROJ. ─────────────────────────── */

typedef struct {
    int N_proj;
    int pure_top1;
    int pure_maj7;
    int ceiling_at_K;
    int cascade_correct;
    double seconds;
} sweep_row_t;

static void eval_one(int N_proj,
                     const m4t_mtfp_t* x_train, int n_train, const int* y_train,
                     const m4t_mtfp_t* x_test,  int n_test,  const int* y_test,
                     sweep_row_t* out)
{
    clock_t t0 = clock();

    /* Build projections. */
    rng_s[0]=42; rng_s[1]=123; rng_s[2]=456; rng_s[3]=789;
    m4t_trit_t* proj_w=malloc((size_t)N_proj*INPUT_DIM);
    for(int i=0;i<N_proj*INPUT_DIM;i++){
        uint32_t r=rng_next()%3;
        proj_w[i]=(r==0)?-1:(r==1)?0:1;
    }
    int proj_Dp=M4T_TRIT_PACKED_BYTES(INPUT_DIM);
    uint8_t* proj_packed=malloc((size_t)N_proj*proj_Dp);
    m4t_pack_trits_rowmajor(proj_packed,proj_w,N_proj,INPUT_DIM);
    free(proj_w);

    m4t_mtfp_t* train_proj=malloc((size_t)n_train*N_proj*sizeof(m4t_mtfp_t));
    m4t_mtfp_t* test_proj =malloc((size_t)n_test *N_proj*sizeof(m4t_mtfp_t));
    for(int i=0;i<n_train;i++)
        m4t_mtfp_ternary_matmul_bt(train_proj+(size_t)i*N_proj,
                                    x_train+(size_t)i*INPUT_DIM,
                                    proj_packed,1,INPUT_DIM,N_proj);
    for(int i=0;i<n_test;i++)
        m4t_mtfp_ternary_matmul_bt(test_proj+(size_t)i*N_proj,
                                    x_test+(size_t)i*INPUT_DIM,
                                    proj_packed,1,INPUT_DIM,N_proj);

    int64_t tau_q;
    {
        size_t total=(size_t)1000*N_proj;
        int64_t* buf=malloc(total*sizeof(int64_t));
        for(int i=0;i<1000;i++)
            for(int p=0;p<N_proj;p++){
                int64_t v=train_proj[(size_t)i*N_proj+p];
                buf[(size_t)i*N_proj+p]=(v>=0)?v:-v;
            }
        tau_q=tau_for_density(buf,total,DENSITY);
        free(buf);
    }

    int Sp=M4T_TRIT_PACKED_BYTES(N_proj);
    uint8_t* train_sigs=calloc((size_t)n_train*Sp,1);
    uint8_t* test_sigs =calloc((size_t)n_test *Sp,1);
    int64_t* tmp=malloc((size_t)N_proj*sizeof(int64_t));
    for(int i=0;i<n_train;i++){
        for(int p=0;p<N_proj;p++) tmp[p]=(int64_t)train_proj[(size_t)i*N_proj+p];
        m4t_route_threshold_extract(train_sigs+(size_t)i*Sp,tmp,tau_q,N_proj);
    }
    for(int i=0;i<n_test;i++){
        for(int p=0;p<N_proj;p++) tmp[p]=(int64_t)test_proj[(size_t)i*N_proj+p];
        m4t_route_threshold_extract(test_sigs+(size_t)i*Sp,tmp,tau_q,N_proj);
    }
    free(tmp); free(train_proj); free(test_proj); free(proj_packed);

    uint8_t* mask=malloc(Sp); memset(mask,0xFF,Sp);

    int pure_top1=0, pure_maj7=0, ceiling=0, cascade=0;
    int32_t* dists = malloc((size_t)n_train*sizeof(int32_t));

    for (int s = 0; s < n_test; s++) {
        const uint8_t* q_sig = test_sigs + (size_t)s*Sp;
        const m4t_mtfp_t* q_img = x_test + (size_t)s*INPUT_DIM;
        int y = y_test[s];

        for (int i = 0; i < n_train; i++) {
            const uint8_t* r_sig = train_sigs + (size_t)i*Sp;
            dists[i] = m4t_popcount_dist(q_sig, r_sig, mask, Sp);
        }

        /* Top-K_RESOLVE by hash distance (partial insertion sort). */
        int32_t topd[K_RESOLVE]; int topi[K_RESOLVE];
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

        /* Pure top-1. */
        if (y_train[topi[0]] == y) pure_top1++;

        /* Pure majority k=7. */
        int counts[N_CLASSES] = {0};
        for (int j=0;j<7;j++) counts[y_train[topi[j]]]++;
        int pred = 0;
        for (int c=1;c<N_CLASSES;c++)
            if (counts[c]>counts[pred]) pred=c;
        if (pred == y) pure_maj7++;

        /* Ceiling at K=50. */
        for (int j=0;j<K_RESOLVE;j++) {
            if (y_train[topi[j]] == y) { ceiling++; break; }
        }

        /* Cascade: pixel-L2sq 1-NN within top-K_RESOLVE. */
        int64_t best_d = INT64_MAX; int best_label = -1;
        for (int j=0;j<K_RESOLVE;j++) {
            int64_t l2 = pixel_l2sq(q_img, x_train + (size_t)topi[j]*INPUT_DIM);
            if (l2 < best_d) { best_d = l2; best_label = y_train[topi[j]]; }
        }
        if (best_label == y) cascade++;
    }

    free(dists); free(mask); free(train_sigs); free(test_sigs);

    out->N_proj = N_proj;
    out->pure_top1 = pure_top1;
    out->pure_maj7 = pure_maj7;
    out->ceiling_at_K = ceiling;
    out->cascade_correct = cascade;
    out->seconds = (double)(clock()-t0)/CLOCKS_PER_SEC;
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

    int N_projs[] = {8, 16, 32, 64, 128, 256, 512, 1024, 4096};
    int nN = sizeof(N_projs)/sizeof(N_projs[0]);

    printf("Cascade-vs-pure sweep across N_PROJ.\n");
    printf("K_RESOLVE=%d, density=%.2f, seed=42, deskewed MNIST.\n", K_RESOLVE, DENSITY);
    printf("%d train, %d test.\n\n", n_train, n_test);

    printf(" N_PROJ   pure_top1   pure_maj7   ceiling@%d   cascade_L2   Δ(casc-maj)   Δ(casc-top1)   time\n",
           K_RESOLVE);

    sweep_row_t rows[9];
    for (int k = 0; k < nN; k++) {
        eval_one(N_projs[k], x_train, n_train, y_train, x_test, n_test, y_test, &rows[k]);
        double t1 = 100.0*rows[k].pure_top1/n_test;
        double m7 = 100.0*rows[k].pure_maj7/n_test;
        double cl = 100.0*rows[k].ceiling_at_K/n_test;
        double cc = 100.0*rows[k].cascade_correct/n_test;
        printf(" %6d   %6.2f%%    %6.2f%%    %6.2f%%     %6.2f%%      %+6.2f%%      %+6.2f%%       %.1fs\n",
               rows[k].N_proj, t1, m7, cl, cc, cc-m7, cc-t1, rows[k].seconds);
        fflush(stdout);
    }

    printf("\nInterpretation:\n");
    printf("  Δ(casc-maj) = cascade gain over pure-hash k=7 majority.\n");
    printf("  Δ(casc-top1) = cascade gain over pure-hash top-1 (hash-only 1-NN).\n");
    printf("  Crossover where cascade no longer helps should appear when\n");
    printf("  pure-hash accuracy approaches the ceiling — predicted near N_PROJ=256.\n");

    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
