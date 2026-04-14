/*
 * test_ternary_scale.c — Scaling the Ternary FFN on MNIST
 *
 * How does accuracy scale with model size?
 * D=32 (baseline), D=64, D=128. Same architecture, more capacity.
 * k=4 ternary routing, 64 epochs, LR=0.02.
 */

#include "trix_ternary_route.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

static uint32_t read_u32_be(FILE* f){uint8_t b[4];fread(b,1,4,f);return((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];}
static float* load_img(const char* p,int* n){FILE* f=fopen(p,"rb");if(!f){exit(1);}read_u32_be(f);*n=(int)read_u32_be(f);read_u32_be(f);read_u32_be(f);size_t t=(size_t)(*n)*784;uint8_t* r=malloc(t);fread(r,1,t,f);fclose(f);float* d=malloc(t*sizeof(float));for(size_t i=0;i<t;i++)d[i]=(float)r[i]/255.0f;free(r);return d;}
static int* load_lbl(const char* p,int* n){FILE* f=fopen(p,"rb");if(!f){exit(1);}read_u32_be(f);*n=(int)read_u32_be(f);uint8_t* r=malloc(*n);fread(r,1,*n,f);fclose(f);int* l=malloc(*n*sizeof(int));for(int i=0;i<*n;i++)l[i]=(int)r[i];free(r);return l;}
static void shuf(int* a,int n,uint64_t s){srand(s);for(int i=n-1;i>0;i--){int j=rand()%(i+1);int t=a[i];a[i]=a[j];a[j]=t;}}

static float run(int D, int T, int H_proj,
                  float* xt, int* yt, int nt,
                  float* xv, int* yv, int nv,
                  int epochs, int batch, float lr, uint64_t seed) {
    srand48(seed);

    /* Projection: 784→H_proj→D */
    float* W1=calloc(H_proj*784,sizeof(float)); float* b1=calloc(H_proj,sizeof(float));
    float* W2=calloc(D*H_proj,sizeof(float));   float* b2=calloc(D,sizeof(float));
    float l1=sqrtf(6.0f/(784.0f+H_proj)), l2=sqrtf(6.0f/(H_proj+D));
    for(int i=0;i<H_proj*784;i++) W1[i]=(2.0*drand48()-1.0)*l1;
    for(int i=0;i<D*H_proj;i++) W2[i]=(2.0*drand48()-1.0)*l2;

    /* Ternary FFN: k=T (all tiles active with signs) */
    TrixTernaryRouteConfig fc={.d_model=D,.num_tiles=T,.tile_hidden=D,
        .active_k=T,.output_scale_init=0.1f,.ln_eps=1e-5f};
    TrixTernaryRoutedFFN* ffn=trix_ternary_route_create(fc,seed+100);
    TrixAtomFFN* head=trix_atom_ffn_create(D,D*2,10,seed+200);

    int* idx=malloc(nt*sizeof(int));for(int i=0;i<nt;i++)idx[i]=i;
    float* z1=calloc(batch*H_proj,sizeof(float));
    float* h1=calloc(batch*H_proj,sizeof(float));
    float* po=calloc(batch*D,sizeof(float));
    float* fo=calloc(batch*D,sizeof(float));
    float* lo=calloc(batch*10,sizeof(float));
    float* dl=calloc(batch*10,sizeof(float));
    float* dfo=calloc(batch*D,sizeof(float));
    float* dpo=calloc(batch*D,sizeof(float));
    float* dh1=calloc(batch*H_proj,sizeof(float));
    float* dz1=calloc(batch*H_proj,sizeof(float));
    float* dW1=calloc(H_proj*784,sizeof(float));float* db1=calloc(H_proj,sizeof(float));
    float* dW2=calloc(D*H_proj,sizeof(float));  float* db2=calloc(D,sizeof(float));

    float best=0;
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);

    for(int ep=0;ep<epochs;ep++){
        shuf(idx,nt,seed+ep*1000);
        for(int b=0;b+batch<=nt;b+=batch){
            float* xb=malloc(batch*784*sizeof(float));int* yb=malloc(batch*sizeof(int));
            for(int i=0;i<batch;i++){memcpy(xb+i*784,xt+idx[b+i]*784,784*sizeof(float));yb[i]=yt[idx[b+i]];}

            trix_matmul_bt(z1,xb,W1,batch,784,H_proj);
            trix_bias_add(z1,b1,batch,H_proj);
            trix_gelu(h1,z1,batch*H_proj);
            trix_matmul_bt(po,h1,W2,batch,H_proj,D);
            trix_bias_add(po,b2,batch,D);

            trix_ternary_route_forward(ffn,po,fo,batch);
            trix_atom_ffn_forward(head,fo,lo,batch);
            trix_cross_entropy_grad(dl,lo,yb,batch,10);

            trix_atom_ffn_zero_grad(head);
            trix_atom_ffn_backward(head,fo,dl,dfo,batch);
            trix_ternary_route_zero_grad(ffn);
            trix_ternary_route_backward(ffn,po,dfo,dpo,batch);

            memset(dW2,0,D*H_proj*sizeof(float));memset(db2,0,D*sizeof(float));
            trix_matmul_at(dW2,dpo,h1,batch,D,H_proj);
            trix_bias_grad(db2,dpo,batch,D);
            trix_matmul(dh1,dpo,W2,batch,D,H_proj);
            trix_gelu_grad(dz1,dh1,z1,batch*H_proj);
            memset(dW1,0,H_proj*784*sizeof(float));memset(db1,0,H_proj*sizeof(float));
            trix_matmul_at(dW1,dz1,xb,batch,H_proj,784);
            trix_bias_grad(db1,dz1,batch,H_proj);

            trix_sgd_update(W1,dW1,lr,H_proj*784);trix_sgd_update(b1,db1,lr,H_proj);
            trix_sgd_update(W2,dW2,lr,D*H_proj);trix_sgd_update(b2,db2,lr,D);
            trix_ternary_route_clip_grad_norm(ffn,1.0f);
            trix_ternary_route_adamw_step(ffn,lr,0.9f,0.999f,1e-8f,0.01f);
            trix_atom_ffn_sgd_step(head,lr);
            free(xb);free(yb);
        }
        int cor=0;
        for(int o=0;o<nv;o+=batch){
            int n=(o+batch<=nv)?batch:nv-o;
            trix_matmul_bt(z1,xv+o*784,W1,n,784,H_proj);
            trix_bias_add(z1,b1,n,H_proj);trix_gelu(h1,z1,n*H_proj);
            trix_matmul_bt(po,h1,W2,n,H_proj,D);trix_bias_add(po,b2,n,D);
            trix_ternary_route_forward(ffn,po,fo,n);
            trix_atom_ffn_forward(head,fo,lo,n);
            int* pr=malloc(n*sizeof(int));trix_argmax(pr,lo,n,10);
            for(int i=0;i<n;i++)if(pr[i]==yv[o+i])cor++;
            free(pr);
        }
        float acc=(float)cor/(float)nv;
        if(acc>best)best=acc;
        if(ep%20==0||ep==epochs-1){
            struct timespec t1;clock_gettime(CLOCK_MONOTONIC,&t1);
            double s=(double)(t1.tv_sec-t0.tv_sec)+1e-9*(double)(t1.tv_nsec-t0.tv_nsec);
            printf("    ep %2d: acc=%.2f%% (%.0fs)\n",ep,acc*100,s);
        }
    }

    int params = H_proj*784+H_proj+D*H_proj+D + T*(D*D+D+D*D+D) + D*D*2+D*2+10;
    printf("    best=%.2f%% params=%dK\n", best*100, params/1000);

    free(W1);free(b1);free(W2);free(b2);
    free(idx);free(z1);free(h1);free(po);free(fo);free(lo);
    free(dl);free(dfo);free(dpo);free(dh1);free(dz1);
    free(dW1);free(db1);free(dW2);free(db2);
    trix_ternary_route_destroy(ffn);trix_atom_ffn_destroy(head);
    return best;
}

int main(int argc,char** argv){
    const char* dd="data/mnist";if(argc>1)dd=argv[1];
    char p[512];int nt,nv;
    snprintf(p,512,"%s/train-images-idx3-ubyte",dd);float* xt=load_img(p,&nt);
    snprintf(p,512,"%s/train-labels-idx1-ubyte",dd);int* yt=load_lbl(p,&nt);
    snprintf(p,512,"%s/t10k-images-idx3-ubyte",dd);float* xv=load_img(p,&nv);
    snprintf(p,512,"%s/t10k-labels-idx1-ubyte",dd);int* yv=load_lbl(p,&nv);

    int epochs=64, batch=128;
    float lr=0.02f;

    printf("=== TERNARY FFN SCALING (k=T, 64 epochs, LR=0.02) ===\n\n");

    struct { int D; int T; int H_proj; const char* label; } configs[] = {
        {32,  4, 64,  "D=32  T=4  H=64"},
        {64,  4, 128, "D=64  T=4  H=128"},
        {64,  8, 128, "D=64  T=8  H=128"},
        {128, 4, 256, "D=128 T=4  H=256"},
        {128, 8, 256, "D=128 T=8  H=256"},
    };
    int n_configs = sizeof(configs)/sizeof(configs[0]);

    float results[5];
    for (int c = 0; c < n_configs; c++) {
        printf("--- %s ---\n", configs[c].label);
        results[c] = run(configs[c].D, configs[c].T, configs[c].H_proj,
                          xt,yt,nt, xv,yv,nv, epochs,batch,lr, 42);
        printf("\n");
    }

    printf("=== SCALING SUMMARY ===\n");
    for (int c = 0; c < n_configs; c++) {
        printf("  %s: %.2f%%\n", configs[c].label, results[c]*100);
    }

    free(xt);free(yt);free(xv);free(yv);
    return 0;
}
