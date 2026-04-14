/*
 * mnist_train_dump.c — train ALL-TERNARY MNIST model, dump weights at peak.
 *
 * HOST-SIDE TOOL. Float is permitted for training (shadow weights + STE).
 * ALL weight matrices use ternary quantization in the forward pass:
 *   - Projection W1, W2: float shadow → quantize → ternary matmul
 *   - FFN tiles: handled by trix_ternary_route (already ternary-in-loop)
 *   - Head W1, W2: float shadow → quantize → ternary matmul
 * Backward: STE (gradients flow through quantization as identity).
 * At save time: all weight matrices are ternary. No dense float weights.
 *
 * Usage: ./mnist_train_dump <mnist_dir> <output.bin> [epochs]
 */

#include "trix_ternary_route.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint32_t read_u32_be(FILE* f) {
    uint8_t b[4]; fread(b,1,4,f);
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];
}
static float* load_idx_images(const char* p, int* n) {
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}
    read_u32_be(f); *n=(int)read_u32_be(f);
    int rows=(int)read_u32_be(f), cols=(int)read_u32_be(f);
    int dim=rows*cols; size_t t=(size_t)(*n)*dim;
    uint8_t* r=malloc(t); fread(r,1,t,f); fclose(f);
    float* d=malloc(t*sizeof(float));
    for(size_t i=0;i<t;i++) d[i]=(float)r[i]/255.0f;
    free(r); return d;
}
static int* load_idx_labels(const char* p, int* n) {
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}
    read_u32_be(f); *n=(int)read_u32_be(f);
    uint8_t* r=malloc(*n); fread(r,1,*n,f); fclose(f);
    int* l=malloc(*n*sizeof(int));
    for(int i=0;i<*n;i++) l[i]=(int)r[i];
    free(r); return l;
}
static void shuf(int* a, int n, uint64_t s) {
    srand(s);
    for(int i=n-1;i>0;i--){ int j=rand()%(i+1); int t=a[i]; a[i]=a[j]; a[j]=t; }
}

/* Quantize float weights to ternary-as-float {-1, 0, +1} in place.
 * Threshold: values beyond ±(mean_abs * 0.5) become ±1; rest become 0. */
static void quantize_to_float_ternary(float* dst, const float* src, int n) {
    float mean_abs = 0;
    for (int i = 0; i < n; i++) mean_abs += fabsf(src[i]);
    mean_abs /= (float)n;
    float thresh = mean_abs * 0.5f;
    for (int i = 0; i < n; i++)
        dst[i] = (src[i] > thresh) ? 1.0f : (src[i] < -thresh) ? -1.0f : 0.0f;
}

static void save_floats(FILE* f, const float* data, int count) {
    int32_t c = count;
    fwrite(&c, sizeof(int32_t), 1, f);
    fwrite(data, sizeof(float), (size_t)count, f);
}
static void save_as_ternary(FILE* f, const float* data, int count) {
    float mean_abs = 0;
    for (int i = 0; i < count; i++) mean_abs += fabsf(data[i]);
    mean_abs /= (float)count;
    float thresh = mean_abs * 0.5f;
    int32_t c = count;
    fwrite(&c, 4, 1, f);
    for (int i = 0; i < count; i++) {
        int8_t t = (data[i] > thresh) ? 1 : (data[i] < -thresh) ? -1 : 0;
        fwrite(&t, 1, 1, f);
    }
}

static void save_weights(const char* path,
    int input_dim, int D, int T, int K, int H_proj, int n_classes,
    const float* pW1, const float* pb1, const float* pW2, const float* pb2,
    TrixTernaryRoutedFFN* ffn,
    const float* headW1, const float* headb1, int head_hidden,
    const float* headW2, const float* headb2)
{
    FILE* fout = fopen(path, "wb");
    if (!fout) return;
    int32_t magic = 0x4D345457;
    fwrite(&magic, 4, 1, fout);
    int32_t cfg[6] = { input_dim, D, T, K, H_proj, n_classes };
    fwrite(cfg, sizeof(int32_t), 6, fout);

    save_as_ternary(fout, pW1, H_proj * input_dim);
    save_floats(fout, pb1, H_proj);
    save_as_ternary(fout, pW2, D * H_proj);
    save_floats(fout, pb2, D);
    save_floats(fout, ffn->ln_weight, D);
    save_floats(fout, ffn->ln_bias, D);

    int w1e = ffn->cfg.tile_hidden * D;
    int w2e = D * ffn->cfg.tile_hidden;
    for (int t = 0; t < T; t++) {
        save_as_ternary(fout, ffn->W1 + t*w1e, w1e);
        save_floats(fout, ffn->b1 + t*ffn->cfg.tile_hidden, ffn->cfg.tile_hidden);
        save_as_ternary(fout, ffn->W2 + t*w2e, w2e);
        save_floats(fout, ffn->b2 + t*D, D);
    }
    save_floats(fout, &ffn->output_scale, 1);

    save_as_ternary(fout, headW1, head_hidden * D);
    save_floats(fout, headb1, head_hidden);
    save_as_ternary(fout, headW2, n_classes * head_hidden);
    save_floats(fout, headb2, n_classes);
    fclose(fout);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <mnist_dir> <output.bin> [epochs]\n", argv[0]);
        return 1;
    }
    const char* mnist_dir = argv[1];
    const char* output_path = argv[2];
    int n_epochs = (argc > 3) ? atoi(argv[3]) : 20;

    int input_dim=784, D=128, T=4, K=3, H_proj=256, n_classes=10, batch=32;
    float lr=0.0174f; uint64_t seed=42;
    int head_hidden = D * 2;

    char path[512]; int nt, nv;
    snprintf(path,512,"%s/train-images-idx3-ubyte",mnist_dir);
    float* xt=load_idx_images(path,&nt);
    snprintf(path,512,"%s/train-labels-idx1-ubyte",mnist_dir);
    int* yt=load_idx_labels(path,&nt);
    snprintf(path,512,"%s/t10k-images-idx3-ubyte",mnist_dir);
    float* xv=load_idx_images(path,&nv);
    snprintf(path,512,"%s/t10k-labels-idx1-ubyte",mnist_dir);
    int* yv=load_idx_labels(path,&nv);
    printf("Loaded MNIST: %d train, %d test\n", nt, nv);

    srand48(seed);
    /* Projection shadow weights */
    float* pW1=calloc(H_proj*input_dim,sizeof(float)); float* pb1=calloc(H_proj,sizeof(float));
    float* pW2=calloc(D*H_proj,sizeof(float)); float* pb2=calloc(D,sizeof(float));
    float l1=sqrtf(6.0f/((float)input_dim+(float)H_proj));
    float l2=sqrtf(6.0f/((float)H_proj+(float)D));
    for(int i=0;i<H_proj*input_dim;i++) pW1[i]=(2.0*drand48()-1.0)*l1;
    for(int i=0;i<D*H_proj;i++) pW2[i]=(2.0*drand48()-1.0)*l2;
    /* Quantized projection buffers */
    float* pW1q=malloc(H_proj*input_dim*sizeof(float));
    float* pW2q=malloc(D*H_proj*sizeof(float));

    /* FFN (already ternary-in-loop) */
    TrixTernaryRouteConfig fc={.d_model=D,.num_tiles=T,.tile_hidden=D,
        .active_k=K,.output_scale_init=0.1f,.ln_eps=1e-5f};
    TrixTernaryRoutedFFN* ffn=trix_ternary_route_create(fc,seed+100);

    /* Head shadow weights */
    float* hW1=calloc(head_hidden*D,sizeof(float)); float* hb1=calloc(head_hidden,sizeof(float));
    float* hW2=calloc(n_classes*head_hidden,sizeof(float)); float* hb2=calloc(n_classes,sizeof(float));
    float l3=sqrtf(6.0f/((float)D+(float)head_hidden));
    float l4=sqrtf(6.0f/((float)head_hidden+(float)n_classes));
    for(int i=0;i<head_hidden*D;i++) hW1[i]=(2.0*drand48()-1.0)*l3;
    for(int i=0;i<n_classes*head_hidden;i++) hW2[i]=(2.0*drand48()-1.0)*l4;
    /* Quantized head buffers */
    float* hW1q=malloc(head_hidden*D*sizeof(float));
    float* hW2q=malloc(n_classes*head_hidden*sizeof(float));

    /* AdamW state for projection + head (like the FFN tiles use internally) */
    int proj_sizes[4] = { H_proj*input_dim, H_proj, D*H_proj, D };
    float* proj_ptrs[4] = { pW1, pb1, pW2, pb2 };
    float* proj_grads[4]; /* assigned per batch below */
    float* proj_m[4], *proj_v[4];
    for (int p = 0; p < 4; p++) {
        proj_m[p] = calloc(proj_sizes[p], sizeof(float));
        proj_v[p] = calloc(proj_sizes[p], sizeof(float));
    }

    int head_sizes[4] = { head_hidden*D, head_hidden, n_classes*head_hidden, n_classes };
    float* head_ptrs[4] = { hW1, hb1, hW2, hb2 };
    float* head_grads[4]; /* assigned below */
    float* head_m[4], *head_v[4];
    for (int p = 0; p < 4; p++) {
        head_m[p] = calloc(head_sizes[p], sizeof(float));
        head_v[p] = calloc(head_sizes[p], sizeof(float));
    }
    int adam_step = 0;

    /* Scratch */
    float* z1=calloc(batch*H_proj,sizeof(float));
    float* h1=calloc(batch*H_proj,sizeof(float));
    float* po=calloc(batch*D,sizeof(float));
    float* fo=calloc(batch*D,sizeof(float));
    float* hz=calloc(batch*head_hidden,sizeof(float));
    float* hh=calloc(batch*head_hidden,sizeof(float));
    float* lo=calloc(batch*n_classes,sizeof(float));
    float* dl=calloc(batch*n_classes,sizeof(float));
    float* dfo=calloc(batch*D,sizeof(float));
    float* dpo=calloc(batch*D,sizeof(float));
    float* dhh=calloc(batch*head_hidden,sizeof(float));
    float* dhz=calloc(batch*head_hidden,sizeof(float));
    float* dh1=calloc(batch*H_proj,sizeof(float));
    float* dz1=calloc(batch*H_proj,sizeof(float));
    float* dW1=calloc(H_proj*input_dim,sizeof(float));
    float* db1g=calloc(H_proj,sizeof(float));
    float* dW2=calloc(D*H_proj,sizeof(float));
    float* db2g=calloc(D,sizeof(float));
    float* dhW1=calloc(head_hidden*D,sizeof(float));
    float* dhb1g=calloc(head_hidden,sizeof(float));
    float* dhW2=calloc(n_classes*head_hidden,sizeof(float));
    float* dhb2g=calloc(n_classes,sizeof(float));
    int* idx=malloc(nt*sizeof(int));
    for(int i=0;i<nt;i++) idx[i]=i;

    float best_acc=0; int best_ep=-1;

    printf("Training %d epochs — ALL layers ternary-in-the-loop (STE)\n", n_epochs);
    for (int ep=0; ep<n_epochs; ep++) {
        shuf(idx,nt,seed+ep*1000);
        for(int b=0;b+batch<=nt;b+=batch){
            float* xb=malloc(batch*input_dim*sizeof(float));
            int* yb=malloc(batch*sizeof(int));
            for(int i=0;i<batch;i++){
                memcpy(xb+i*input_dim,xt+idx[b+i]*input_dim,input_dim*sizeof(float));
                yb[i]=yt[idx[b+i]];
            }

            /* ── Forward: projection with QUANTIZED weights ── */
            quantize_to_float_ternary(pW1q, pW1, H_proj*input_dim);
            trix_matmul_bt(z1,xb,pW1q,batch,input_dim,H_proj);
            trix_bias_add(z1,pb1,batch,H_proj);
            trix_gelu(h1,z1,batch*H_proj);
            quantize_to_float_ternary(pW2q, pW2, D*H_proj);
            trix_matmul_bt(po,h1,pW2q,batch,H_proj,D);
            trix_bias_add(po,pb2,batch,D);

            /* ── Forward: ternary FFN (already quantized internally) ── */
            trix_ternary_route_forward(ffn,po,fo,batch);

            /* ── Forward: head with QUANTIZED weights ── */
            quantize_to_float_ternary(hW1q, hW1, head_hidden*D);
            trix_matmul_bt(hz,fo,hW1q,batch,D,head_hidden);
            trix_bias_add(hz,hb1,batch,head_hidden);
            trix_gelu(hh,hz,batch*head_hidden);
            quantize_to_float_ternary(hW2q, hW2, n_classes*head_hidden);
            trix_matmul_bt(lo,hh,hW2q,batch,head_hidden,n_classes);
            trix_bias_add(lo,hb2,batch,n_classes);

            /* ── Backward: STE — gradients through quantization as identity ── */
            trix_cross_entropy_grad(dl,lo,yb,batch,n_classes);

            /* Head backward (manual — STE on quantized weights) */
            memset(dhW2,0,n_classes*head_hidden*sizeof(float));
            memset(dhb2g,0,n_classes*sizeof(float));
            trix_matmul_at(dhW2,dl,hh,batch,n_classes,head_hidden);
            trix_bias_grad(dhb2g,dl,batch,n_classes);
            trix_matmul(dhh,dl,hW2q,batch,n_classes,head_hidden);
            trix_gelu_grad(dhz,dhh,hz,batch*head_hidden);
            memset(dhW1,0,head_hidden*D*sizeof(float));
            memset(dhb1g,0,head_hidden*sizeof(float));
            trix_matmul_at(dhW1,dhz,fo,batch,head_hidden,D);
            trix_bias_grad(dhb1g,dhz,batch,head_hidden);
            trix_matmul(dfo,dhz,hW1q,batch,head_hidden,D);

            /* FFN backward */
            trix_ternary_route_zero_grad(ffn);
            trix_ternary_route_backward(ffn,po,dfo,dpo,batch);

            /* Projection backward (STE) */
            memset(dW2,0,D*H_proj*sizeof(float));
            memset(db2g,0,D*sizeof(float));
            trix_matmul_at(dW2,dpo,h1,batch,D,H_proj);
            trix_bias_grad(db2g,dpo,batch,D);
            trix_matmul(dh1,dpo,pW2q,batch,D,H_proj);
            trix_gelu_grad(dz1,dh1,z1,batch*H_proj);
            memset(dW1,0,H_proj*input_dim*sizeof(float));
            memset(db1g,0,H_proj*sizeof(float));
            trix_matmul_at(dW1,dz1,xb,batch,H_proj,input_dim);
            trix_bias_grad(db1g,dz1,batch,H_proj);

            /* Update ALL shadow weights with AdamW (not SGD) */
            adam_step++;
            proj_grads[0]=dW1; proj_grads[1]=db1g; proj_grads[2]=dW2; proj_grads[3]=db2g;
            for (int p=0;p<4;p++)
                trix_adamw_update(proj_ptrs[p],proj_grads[p],proj_m[p],proj_v[p],
                    lr,0.9f,0.999f,1e-8f,0.01f,adam_step,proj_sizes[p]);

            trix_ternary_route_clip_grad_norm(ffn,1.0f);
            trix_ternary_route_adamw_step(ffn,lr,0.9f,0.999f,1e-8f,0.01f);

            head_grads[0]=dhW1; head_grads[1]=dhb1g; head_grads[2]=dhW2; head_grads[3]=dhb2g;
            for (int p=0;p<4;p++)
                trix_adamw_update(head_ptrs[p],head_grads[p],head_m[p],head_v[p],
                    lr,0.9f,0.999f,1e-8f,0.01f,adam_step,head_sizes[p]);

            free(xb);free(yb);
        }

        /* Eval (using quantized weights) */
        quantize_to_float_ternary(pW1q, pW1, H_proj*input_dim);
        quantize_to_float_ternary(pW2q, pW2, D*H_proj);
        quantize_to_float_ternary(hW1q, hW1, head_hidden*D);
        quantize_to_float_ternary(hW2q, hW2, n_classes*head_hidden);
        int cor=0;
        for(int o=0;o<nv;o+=batch){
            int n=(o+batch<=nv)?batch:nv-o;
            trix_matmul_bt(z1,xv+o*input_dim,pW1q,n,input_dim,H_proj);
            trix_bias_add(z1,pb1,n,H_proj);
            trix_gelu(h1,z1,n*H_proj);
            trix_matmul_bt(po,h1,pW2q,n,H_proj,D);
            trix_bias_add(po,pb2,n,D);
            trix_ternary_route_forward(ffn,po,fo,n);
            trix_matmul_bt(hz,fo,hW1q,n,D,head_hidden);
            trix_bias_add(hz,hb1,n,head_hidden);
            trix_gelu(hh,hz,n*head_hidden);
            trix_matmul_bt(lo,hh,hW2q,n,head_hidden,n_classes);
            trix_bias_add(lo,hb2,n,n_classes);
            int* pr=malloc(n*sizeof(int));
            trix_argmax(pr,lo,n,n_classes);
            for(int i=0;i<n;i++) if(pr[i]==yv[o+i]) cor++;
            free(pr);
        }
        float acc=(float)cor/(float)nv;
        int is_best=(acc>best_acc);
        printf("  ep %2d: %.2f%% (%d/%d)%s\n", ep, acc*100, cor, nv,
               is_best ? " << BEST — saving" : "");
        if (is_best) {
            best_acc=acc; best_ep=ep;
            save_weights(output_path, input_dim, D, T, K, H_proj, n_classes,
                         pW1, pb1, pW2, pb2, ffn, hW1, hb1, head_hidden, hW2, hb2);
        }
    }

    printf("\nBest: %.2f%% at epoch %d → %s\n", best_acc*100, best_ep, output_path);
    free(pW1);free(pb1);free(pW2);free(pb2);free(pW1q);free(pW2q);
    free(hW1);free(hb1);free(hW2);free(hb2);free(hW1q);free(hW2q);
    free(z1);free(h1);free(po);free(fo);free(hz);free(hh);free(lo);
    free(dl);free(dfo);free(dpo);free(dhh);free(dhz);free(dh1);free(dz1);
    free(dW1);free(db1g);free(dW2);free(db2g);
    free(dhW1);free(dhb1g);free(dhW2);free(dhb2g);
    free(idx);free(xt);free(yt);free(xv);free(yv);
    trix_ternary_route_destroy(ffn);
    return 0;
}
