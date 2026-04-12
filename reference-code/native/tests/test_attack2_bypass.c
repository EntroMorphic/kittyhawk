/*
 * test_attack2_bypass.c — Red-team Attack 2 verification
 *
 * The original test_ternary_scale.c calls trix_ternary_route_backward()
 * which is an empty stub, producing dpo = zeros. This means:
 *   (a) the ternary FFN weights never update (adamw_step is also a stub)
 *   (b) the projection layers W1,b1,W2,b2 get zero gradients (dpo=0 propagates)
 *   (c) only the classifier head receives real gradients
 *
 * This test runs three variants to isolate what's actually contributing:
 *
 * A) ORIGINAL:   projection(frozen) → FFN(frozen) → head(trained)
 *    Same as test_ternary_scale.c — reproduces the claimed architecture.
 *    Projection and FFN get zero gradients due to stub backward.
 *
 * B) NO_FFN:     projection(frozen) → head(trained)
 *    Removes the FFN entirely. If accuracy matches A, the FFN adds nothing.
 *
 * C) FULL_GRAD:  projection(trained) → head(trained)
 *    No FFN, but fixes gradient flow: backward goes from head through to
 *    projection. Tests whether a trained projection + head beats the
 *    frozen-projection variants.
 */

#include "trix_ternary_route.h"
#include "trix_atoms.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

static uint32_t read_u32_be(FILE* f){uint8_t b[4];fread(b,1,4,f);return((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];}
static float* load_img(const char* p,int* n){FILE* f=fopen(p,"rb");if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}read_u32_be(f);*n=(int)read_u32_be(f);read_u32_be(f);read_u32_be(f);size_t t=(size_t)(*n)*784;uint8_t* r=malloc(t);fread(r,1,t,f);fclose(f);float* d=malloc(t*sizeof(float));for(size_t i=0;i<t;i++)d[i]=(float)r[i]/255.0f;free(r);return d;}
static int* load_lbl(const char* p,int* n){FILE* f=fopen(p,"rb");if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}read_u32_be(f);*n=(int)read_u32_be(f);uint8_t* r=malloc(*n);fread(r,1,*n,f);fclose(f);int* l=malloc(*n*sizeof(int));for(int i=0;i<*n;i++)l[i]=(int)r[i];free(r);return l;}
static void shuf(int* a,int n,uint64_t s){srand(s);for(int i=n-1;i>0;i--){int j=rand()%(i+1);int t=a[i];a[i]=a[j];a[j]=t;}}

typedef enum { MODE_ORIGINAL, MODE_NO_FFN, MODE_FULL_GRAD } Mode;

static float run(Mode mode, int D, int T, int H_proj,
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

    /* Ternary FFN (only created for MODE_ORIGINAL) */
    TrixTernaryRoutedFFN* ffn = NULL;
    if (mode == MODE_ORIGINAL) {
        TrixTernaryRouteConfig fc={.d_model=D,.num_tiles=T,.tile_hidden=D,
            .active_k=T,.output_scale_init=0.1f,.ln_eps=1e-5f};
        ffn=trix_ternary_route_create(fc,seed+100);
    }

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

            /* Forward: projection */
            trix_matmul_bt(z1,xb,W1,batch,784,H_proj);
            trix_bias_add(z1,b1,batch,H_proj);
            trix_gelu(h1,z1,batch*H_proj);
            trix_matmul_bt(po,h1,W2,batch,H_proj,D);
            trix_bias_add(po,b2,batch,D);

            /* Forward: FFN (only in ORIGINAL mode) */
            float* head_input;
            if (mode == MODE_ORIGINAL) {
                trix_ternary_route_forward(ffn,po,fo,batch);
                head_input = fo;
            } else {
                head_input = po;
            }

            /* Forward: classifier */
            trix_atom_ffn_forward(head,head_input,lo,batch);
            trix_cross_entropy_grad(dl,lo,yb,batch,10);

            /* Backward: classifier */
            trix_atom_ffn_zero_grad(head);
            trix_atom_ffn_backward(head,head_input,dl,dfo,batch);

            if (mode == MODE_ORIGINAL) {
                /* Original path: backward through FFN stub → dpo = zeros */
                trix_ternary_route_zero_grad(ffn);
                trix_ternary_route_backward(ffn,po,dfo,dpo,batch);
                /* dpo is all zeros because backward is a stub */
            } else if (mode == MODE_NO_FFN) {
                /* No FFN, and we mimic the original's broken gradient:
                   set dpo = zeros to match what the original does */
                memset(dpo,0,batch*D*sizeof(float));
            } else {
                /* FULL_GRAD: dfo goes straight back as dpo */
                memcpy(dpo,dfo,batch*D*sizeof(float));
            }

            /* Backward: projection (gets zero grads in ORIGINAL and NO_FFN) */
            memset(dW2,0,D*H_proj*sizeof(float));memset(db2,0,D*sizeof(float));
            trix_matmul_at(dW2,dpo,h1,batch,D,H_proj);
            trix_bias_grad(db2,dpo,batch,D);
            trix_matmul(dh1,dpo,W2,batch,D,H_proj);
            trix_gelu_grad(dz1,dh1,z1,batch*H_proj);
            memset(dW1,0,H_proj*784*sizeof(float));memset(db1,0,H_proj*sizeof(float));
            trix_matmul_at(dW1,dz1,xb,batch,H_proj,784);
            trix_bias_grad(db1,dz1,batch,H_proj);

            /* Update */
            trix_sgd_update(W1,dW1,lr,H_proj*784);trix_sgd_update(b1,db1,lr,H_proj);
            trix_sgd_update(W2,dW2,lr,D*H_proj);trix_sgd_update(b2,db2,lr,D);
            if (mode == MODE_ORIGINAL) {
                trix_ternary_route_clip_grad_norm(ffn,1.0f);
                trix_ternary_route_adamw_step(ffn,lr,0.9f,0.999f,1e-8f,0.01f);
            }
            trix_atom_ffn_sgd_step(head,lr);
            free(xb);free(yb);
        }

        /* Eval */
        int cor=0;
        for(int o=0;o<nv;o+=batch){
            int n=(o+batch<=nv)?batch:nv-o;
            trix_matmul_bt(z1,xv+o*784,W1,n,784,H_proj);
            trix_bias_add(z1,b1,n,H_proj);trix_gelu(h1,z1,n*H_proj);
            trix_matmul_bt(po,h1,W2,n,H_proj,D);trix_bias_add(po,b2,n,D);
            float* eval_input;
            if (mode == MODE_ORIGINAL) {
                trix_ternary_route_forward(ffn,po,fo,n);
                eval_input = fo;
            } else {
                eval_input = po;
            }
            trix_atom_ffn_forward(head,eval_input,lo,n);
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

    printf("    best=%.2f%%\n", best*100);

    free(W1);free(b1);free(W2);free(b2);
    free(idx);free(z1);free(h1);free(po);free(fo);free(lo);
    free(dl);free(dfo);free(dpo);free(dh1);free(dz1);
    free(dW1);free(db1);free(dW2);free(db2);
    if(ffn) trix_ternary_route_destroy(ffn);
    trix_atom_ffn_destroy(head);
    return best;
}

int main(int argc,char** argv){
    const char* dd="data/mnist";if(argc>1)dd=argv[1];
    char p[512];int nt,nv;
    snprintf(p,512,"%s/train-images-idx3-ubyte",dd);float* xt=load_img(p,&nt);
    snprintf(p,512,"%s/train-labels-idx1-ubyte",dd);int* yt=load_lbl(p,&nt);
    snprintf(p,512,"%s/t10k-images-idx3-ubyte",dd);float* xv=load_img(p,&nv);
    snprintf(p,512,"%s/t10k-labels-idx1-ubyte",dd);int* yv=load_lbl(p,&nv);

    /* Match the D=128 T=4 config that produced 98.22% */
    int D=128, T=4, H_proj=256;
    int epochs=64, batch=32;
    float lr=0.0092f;

    printf("=== ATTACK 2: TERNARY FFN BYPASS TEST (D=%d T=%d H=%d LR=%.4f) ===\n\n", D, T, H_proj, lr);

    printf("--- A) ORIGINAL: projection(frozen) + FFN(frozen) + head(trained) ---\n");
    printf("    (Reproduces test_ternary_scale.c — backward stub makes projection+FFN frozen)\n");
    float a = run(MODE_ORIGINAL, D, T, H_proj, xt,yt,nt, xv,yv,nv, epochs,batch,lr, 42);

    printf("\n--- B) NO_FFN: projection(frozen) + head(trained) ---\n");
    printf("    (FFN removed entirely; projection still frozen via zero grads)\n");
    float b = run(MODE_NO_FFN, D, T, H_proj, xt,yt,nt, xv,yv,nv, epochs,batch,lr, 42);

    printf("\n--- C) FULL_GRAD: projection(trained) + head(trained) ---\n");
    printf("    (No FFN; gradient flows back through projection)\n");
    float c = run(MODE_FULL_GRAD, D, T, H_proj, xt,yt,nt, xv,yv,nv, epochs,batch,lr, 42);

    printf("\n=== ATTACK 2 SUMMARY ===\n");
    printf("  A) ORIGINAL (proj frozen + FFN frozen + head): %.2f%%\n", a*100);
    printf("  B) NO_FFN   (proj frozen + head):              %.2f%%\n", b*100);
    printf("  C) FULL_GRAD (proj trained + head):            %.2f%%\n", c*100);
    printf("\n");
    float diff_ab = (a-b)*100;
    printf("  FFN contribution (A-B):                        %+.2f%%\n", diff_ab);
    printf("  Projection training value (C-B):               %+.2f%%\n", (c-b)*100);

    if (fabsf(diff_ab) < 0.5f) {
        printf("\n  VERDICT: FFN contributes <0.5%% — Attack 2 CONFIRMED.\n");
        printf("  The 98.22%% result is primarily from the random projection + trained head.\n");
    } else if (diff_ab > 0.5f) {
        printf("\n  VERDICT: FFN contributes +%.2f%% — Attack 2 PARTIALLY REFUTED.\n", diff_ab);
        printf("  The FFN's random features add measurable value, even without training.\n");
    } else {
        printf("\n  VERDICT: FFN HURTS by %.2f%% — the frozen FFN degrades accuracy.\n", -diff_ab);
    }

    free(xt);free(yt);free(xv);free(yv);
    return 0;
}
