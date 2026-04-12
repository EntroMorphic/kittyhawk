/*
 * test_ternary_vision.c — Ternary routed FFN on MNIST, Fashion-MNIST, and CIFAR-10
 *
 * Same architecture as test_ternary_scale.c (k-of-T ternary routing with
 * weight-derived signatures, full backward pass), applied to three datasets.
 *
 * MNIST / Fashion-MNIST: 784-dim (28x28 grayscale), IDX format
 * CIFAR-10: 3072-dim (32x32x3 float32), raw binary format
 */

#include "trix_ternary_route.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* -- IDX loaders (MNIST / Fashion-MNIST) -- */
static uint32_t read_u32_be(FILE* f) {
    uint8_t b[4]; fread(b,1,4,f);
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];
}
static float* load_idx_images(const char* p, int* n) {
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}
    read_u32_be(f); *n=(int)read_u32_be(f);
    int rows=(int)read_u32_be(f), cols=(int)read_u32_be(f);
    int dim = rows*cols;
    size_t t=(size_t)(*n)*dim;
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

/* -- Binary loaders (CIFAR-10 pre-extracted) -- */
static float* load_bin_f32(const char* p, int n_samples, int dim) {
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}
    size_t total = (size_t)n_samples * dim;
    float* d = malloc(total * sizeof(float));
    size_t rd = fread(d, sizeof(float), total, f); fclose(f);
    if (rd != total) { fprintf(stderr, "Short read %s: got %zu, expected %zu\n", p, rd, total); exit(1); }
    return d;
}
static int* load_bin_i32(const char* p, int n_samples) {
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}
    int32_t* raw = malloc((size_t)n_samples * sizeof(int32_t));
    fread(raw, sizeof(int32_t), n_samples, f); fclose(f);
    int* l = malloc((size_t)n_samples * sizeof(int));
    for (int i=0; i<n_samples; i++) l[i] = (int)raw[i];
    free(raw); return l;
}

static void shuf(int* a, int n, uint64_t s) {
    srand(s);
    for(int i=n-1;i>0;i--){ int j=rand()%(i+1); int t=a[i]; a[i]=a[j]; a[j]=t; }
}

/* -- Training -- */
static float run(int input_dim, int D, int T, int K, int H_proj, int n_classes,
                  float* xt, int* yt, int nt,
                  float* xv, int* yv, int nv,
                  int epochs, int batch, float lr, uint64_t seed) {
    srand48(seed);

    /* Projection: input_dim → H_proj → D */
    float* W1=calloc(H_proj*input_dim,sizeof(float)); float* b1=calloc(H_proj,sizeof(float));
    float* W2=calloc(D*H_proj,sizeof(float));          float* b2=calloc(D,sizeof(float));
    float l1=sqrtf(6.0f/((float)input_dim+(float)H_proj));
    float l2=sqrtf(6.0f/((float)H_proj+(float)D));
    for(int i=0;i<H_proj*input_dim;i++) W1[i]=(2.0*drand48()-1.0)*l1;
    for(int i=0;i<D*H_proj;i++) W2[i]=(2.0*drand48()-1.0)*l2;

    /* Ternary FFN: k=T (all tiles active with signs) */
    TrixTernaryRouteConfig fc = {
        .d_model = D, .num_tiles = T, .tile_hidden = D,
        .active_k = K, .output_scale_init = 0.1f, .ln_eps = 1e-5f
    };
    TrixTernaryRoutedFFN* ffn = trix_ternary_route_create(fc, seed+100);

    /* Classifier head: D → D*2 → n_classes */
    TrixAtomFFN* head = trix_atom_ffn_create(D, D*2, n_classes, seed+200);

    int* idx = malloc(nt*sizeof(int));
    for(int i=0;i<nt;i++) idx[i]=i;

    /* Scratch buffers */
    float* z1_buf  = calloc(batch*H_proj,sizeof(float));
    float* h1_buf  = calloc(batch*H_proj,sizeof(float));
    float* po      = calloc(batch*D,sizeof(float));
    float* fo      = calloc(batch*D,sizeof(float));
    float* lo      = calloc(batch*n_classes,sizeof(float));
    float* dl      = calloc(batch*n_classes,sizeof(float));
    float* dfo     = calloc(batch*D,sizeof(float));
    float* dpo     = calloc(batch*D,sizeof(float));
    float* dh1_buf = calloc(batch*H_proj,sizeof(float));
    float* dz1_buf = calloc(batch*H_proj,sizeof(float));
    float* dW1     = calloc(H_proj*input_dim,sizeof(float));
    float* db1_buf = calloc(H_proj,sizeof(float));
    float* dW2     = calloc(D*H_proj,sizeof(float));
    float* db2_buf = calloc(D,sizeof(float));

    float best = 0;
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int ep = 0; ep < epochs; ep++) {
        shuf(idx, nt, seed + ep*1000);
        for (int b = 0; b+batch <= nt; b += batch) {
            float* xb = malloc(batch*input_dim*sizeof(float));
            int* yb = malloc(batch*sizeof(int));
            for(int i=0;i<batch;i++){
                memcpy(xb+i*input_dim, xt+idx[b+i]*input_dim, input_dim*sizeof(float));
                yb[i] = yt[idx[b+i]];
            }

            /* Forward: projection */
            trix_matmul_bt(z1_buf, xb, W1, batch, input_dim, H_proj);
            trix_bias_add(z1_buf, b1, batch, H_proj);
            trix_gelu(h1_buf, z1_buf, batch*H_proj);
            trix_matmul_bt(po, h1_buf, W2, batch, H_proj, D);
            trix_bias_add(po, b2, batch, D);

            /* Forward: ternary routed FFN */
            trix_ternary_route_forward(ffn, po, fo, batch);

            /* Forward: classifier */
            trix_atom_ffn_forward(head, fo, lo, batch);
            trix_cross_entropy_grad(dl, lo, yb, batch, n_classes);

            /* Backward: classifier */
            trix_atom_ffn_zero_grad(head);
            trix_atom_ffn_backward(head, fo, dl, dfo, batch);

            /* Backward: ternary FFN (real backward pass) */
            trix_ternary_route_zero_grad(ffn);
            trix_ternary_route_backward(ffn, po, dfo, dpo, batch);

            /* Backward: projection */
            memset(dW2, 0, D*H_proj*sizeof(float));
            memset(db2_buf, 0, D*sizeof(float));
            trix_matmul_at(dW2, dpo, h1_buf, batch, D, H_proj);
            trix_bias_grad(db2_buf, dpo, batch, D);
            trix_matmul(dh1_buf, dpo, W2, batch, D, H_proj);
            trix_gelu_grad(dz1_buf, dh1_buf, z1_buf, batch*H_proj);
            memset(dW1, 0, H_proj*input_dim*sizeof(float));
            memset(db1_buf, 0, H_proj*sizeof(float));
            trix_matmul_at(dW1, dz1_buf, xb, batch, H_proj, input_dim);
            trix_bias_grad(db1_buf, dz1_buf, batch, H_proj);

            /* Update */
            trix_sgd_update(W1, dW1, lr, H_proj*input_dim);
            trix_sgd_update(b1, db1_buf, lr, H_proj);
            trix_sgd_update(W2, dW2, lr, D*H_proj);
            trix_sgd_update(b2, db2_buf, lr, D);
            trix_ternary_route_clip_grad_norm(ffn, 1.0f);
            trix_ternary_route_adamw_step(ffn, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
            trix_atom_ffn_sgd_step(head, lr);

            free(xb); free(yb);
        }

        /* Eval */
        int cor = 0;
        for (int o = 0; o < nv; o += batch) {
            int n = (o+batch <= nv) ? batch : nv-o;
            float* xb_eval = xv + o*input_dim;
            trix_matmul_bt(z1_buf, xb_eval, W1, n, input_dim, H_proj);
            trix_bias_add(z1_buf, b1, n, H_proj);
            trix_gelu(h1_buf, z1_buf, n*H_proj);
            trix_matmul_bt(po, h1_buf, W2, n, H_proj, D);
            trix_bias_add(po, b2, n, D);
            trix_ternary_route_forward(ffn, po, fo, n);
            trix_atom_ffn_forward(head, fo, lo, n);
            int* pr = malloc(n*sizeof(int));
            trix_argmax(pr, lo, n, n_classes);
            for(int i=0;i<n;i++) if(pr[i]==yv[o+i]) cor++;
            free(pr);
        }
        float acc = (float)cor / (float)nv;
        if (acc > best) best = acc;
        if (ep % 10 == 0 || ep == epochs-1) {
            struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
            double s = (double)(t1.tv_sec-t0.tv_sec) + 1e-9*(double)(t1.tv_nsec-t0.tv_nsec);
            printf("    ep %2d: acc=%.2f%% (%.0fs)\n", ep, acc*100, s);
        }
    }

    printf("    best=%.2f%%\n", best*100);

    free(W1);free(b1);free(W2);free(b2);
    free(idx);free(z1_buf);free(h1_buf);free(po);free(fo);free(lo);
    free(dl);free(dfo);free(dpo);free(dh1_buf);free(dz1_buf);
    free(dW1);free(db1_buf);free(dW2);free(db2_buf);
    trix_ternary_route_destroy(ffn);
    trix_atom_ffn_destroy(head);
    return best;
}

int main(int argc, char** argv) {
    const char* mnist_dir    = "data/mnist";
    const char* fashion_dir  = "data/fashion_mnist";
    const char* cifar_dir    = "data/cifar-10-batches-py";
    if (argc > 1) mnist_dir   = argv[1];
    if (argc > 2) fashion_dir = argv[2];
    if (argc > 3) cifar_dir   = argv[3];

    int D = 128, T = 4, K = 3, epochs = 64, batch = 32;
    float lr = 0.0174f;

    printf("=== TERNARY ROUTED FFN: VISION BENCHMARKS ===\n");
    printf("    D=%d T=%d K=%d epochs=%d batch=%d lr=%.4f seed=42\n\n", D, T, K, epochs, batch, lr);

    /* ── MNIST (784-dim, 10 classes) ── */
    {
        char p[512]; int nt, nv;
        snprintf(p,512,"%s/train-images-idx3-ubyte",mnist_dir);
        float* xt = load_idx_images(p, &nt);
        snprintf(p,512,"%s/train-labels-idx1-ubyte",mnist_dir);
        int* yt = load_idx_labels(p, &nt);
        snprintf(p,512,"%s/t10k-images-idx3-ubyte",mnist_dir);
        float* xv = load_idx_images(p, &nv);
        snprintf(p,512,"%s/t10k-labels-idx1-ubyte",mnist_dir);
        int* yv = load_idx_labels(p, &nv);

        int H_proj = 256;
        printf("--- MNIST (784→%d→%d, %d tiles) ---\n", H_proj, D, T);
        float acc = run(784, D, T, K, H_proj, 10, xt,yt,nt, xv,yv,nv, epochs,batch,lr, 42);
        printf("\n");
        free(xt);free(yt);free(xv);free(yv);
    }

    /* ── Fashion-MNIST (784-dim, 10 classes) ── */
    {
        char p[512]; int nt, nv;
        snprintf(p,512,"%s/train-images-idx3-ubyte",fashion_dir);
        float* xt = load_idx_images(p, &nt);
        snprintf(p,512,"%s/train-labels-idx1-ubyte",fashion_dir);
        int* yt = load_idx_labels(p, &nt);
        snprintf(p,512,"%s/t10k-images-idx3-ubyte",fashion_dir);
        float* xv = load_idx_images(p, &nv);
        snprintf(p,512,"%s/t10k-labels-idx1-ubyte",fashion_dir);
        int* yv = load_idx_labels(p, &nv);

        int H_proj = 256;
        printf("--- Fashion-MNIST (784→%d→%d, %d tiles) ---\n", H_proj, D, T);
        float acc = run(784, D, T, K, H_proj, 10, xt,yt,nt, xv,yv,nv, epochs,batch,lr, 42);
        printf("\n");
        free(xt);free(yt);free(xv);free(yv);
    }

    /* ── CIFAR-10 (3072-dim, 10 classes) ── */
    {
        char pi[512], pl[512];
        snprintf(pi,512,"%s/train_images.bin",cifar_dir);
        snprintf(pl,512,"%s/train_labels.bin",cifar_dir);
        int nt = 50000;
        float* xt = load_bin_f32(pi, nt, 3072);
        int* yt = load_bin_i32(pl, nt);

        snprintf(pi,512,"%s/test_images.bin",cifar_dir);
        snprintf(pl,512,"%s/test_labels.bin",cifar_dir);
        int nv = 10000;
        float* xv = load_bin_f32(pi, nv, 3072);
        int* yv = load_bin_i32(pl, nv);

        int H_proj = 512;  /* wider projection for higher-dim input */
        printf("--- CIFAR-10 (3072→%d→%d, %d tiles) ---\n", H_proj, D, T);
        float acc = run(3072, D, T, K, H_proj, 10, xt,yt,nt, xv,yv,nv, epochs,batch,lr, 42);
        printf("\n");
        free(xt);free(yt);free(xv);free(yv);
    }

    printf("=== DONE ===\n");
    return 0;
}
