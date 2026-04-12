/*
 * mnist_train_dump.c — train trix-z MNIST model and dump weights at peak.
 *
 * This is a HOST-SIDE TOOL that uses trix-z's float+ternary training engine.
 * It is NOT part of libm4t. Float arithmetic is permitted here.
 *
 * Usage: ./mnist_train_dump <mnist_dir> <output_weights.bin>
 *
 * Trains for 1 epoch (peak accuracy is at epoch 0 for this config),
 * then saves all model weights to a binary file for M4T inference.
 */

#include "trix_ternary_route.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* IDX format reader */
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

/* Save a float array to file with a header: [int32 count] [float32 data...] */
static void save_floats(FILE* f, const float* data, int count) {
    int32_t c = count;
    fwrite(&c, sizeof(int32_t), 1, f);
    fwrite(data, sizeof(float), (size_t)count, f);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <mnist_dir> <output_weights.bin>\n", argv[0]);
        return 1;
    }
    const char* mnist_dir = argv[1];
    const char* output_path = argv[2];

    /* Config matching the reference test */
    int input_dim = 784, D = 128, T = 4, K = 3;
    int H_proj = 256, n_classes = 10, batch = 32;
    float lr = 0.0174f;
    uint64_t seed = 42;

    /* Load data */
    char path[512]; int nt, nv;
    snprintf(path,512,"%s/train-images-idx3-ubyte", mnist_dir);
    float* xt = load_idx_images(path, &nt);
    snprintf(path,512,"%s/train-labels-idx1-ubyte", mnist_dir);
    int* yt = load_idx_labels(path, &nt);
    snprintf(path,512,"%s/t10k-images-idx3-ubyte", mnist_dir);
    float* xv = load_idx_images(path, &nv);
    snprintf(path,512,"%s/t10k-labels-idx1-ubyte", mnist_dir);
    int* yv = load_idx_labels(path, &nv);

    printf("Loaded MNIST: %d train, %d test\n", nt, nv);

    /* Init projection */
    srand48(seed);
    float* pW1=calloc(H_proj*input_dim,sizeof(float)); float* pb1=calloc(H_proj,sizeof(float));
    float* pW2=calloc(D*H_proj,sizeof(float));          float* pb2=calloc(D,sizeof(float));
    float l1=sqrtf(6.0f/((float)input_dim+(float)H_proj));
    float l2=sqrtf(6.0f/((float)H_proj+(float)D));
    for(int i=0;i<H_proj*input_dim;i++) pW1[i]=(2.0*drand48()-1.0)*l1;
    for(int i=0;i<D*H_proj;i++) pW2[i]=(2.0*drand48()-1.0)*l2;

    /* Init FFN */
    TrixTernaryRouteConfig fc = {
        .d_model=D, .num_tiles=T, .tile_hidden=D,
        .active_k=K, .output_scale_init=0.1f, .ln_eps=1e-5f
    };
    TrixTernaryRoutedFFN* ffn = trix_ternary_route_create(fc, seed+100);

    /* Init head */
    TrixAtomFFN* head = trix_atom_ffn_create(D, D*2, n_classes, seed+200);

    /* Scratch */
    float* z1=calloc(batch*H_proj,sizeof(float));
    float* h1=calloc(batch*H_proj,sizeof(float));
    float* po=calloc(batch*D,sizeof(float));
    float* fo=calloc(batch*D,sizeof(float));
    float* lo=calloc(batch*n_classes,sizeof(float));
    float* dl=calloc(batch*n_classes,sizeof(float));
    float* dfo=calloc(batch*D,sizeof(float));
    float* dpo=calloc(batch*D,sizeof(float));
    float* dh1=calloc(batch*H_proj,sizeof(float));
    float* dz1=calloc(batch*H_proj,sizeof(float));
    float* dW1=calloc(H_proj*input_dim,sizeof(float));
    float* db1_g=calloc(H_proj,sizeof(float));
    float* dW2=calloc(D*H_proj,sizeof(float));
    float* db2_g=calloc(D,sizeof(float));

    int* idx=malloc(nt*sizeof(int));
    for(int i=0;i<nt;i++) idx[i]=i;

    /* Train for 1 epoch */
    printf("Training 1 epoch...\n");
    shuf(idx, nt, seed);
    for (int b=0; b+batch<=nt; b+=batch) {
        float* xb=malloc(batch*input_dim*sizeof(float));
        int* yb=malloc(batch*sizeof(int));
        for(int i=0;i<batch;i++){
            memcpy(xb+i*input_dim, xt+idx[b+i]*input_dim, input_dim*sizeof(float));
            yb[i]=yt[idx[b+i]];
        }
        trix_matmul_bt(z1,xb,pW1,batch,input_dim,H_proj);
        trix_bias_add(z1,pb1,batch,H_proj);
        trix_gelu(h1,z1,batch*H_proj);
        trix_matmul_bt(po,h1,pW2,batch,H_proj,D);
        trix_bias_add(po,pb2,batch,D);
        trix_ternary_route_forward(ffn,po,fo,batch);
        trix_atom_ffn_forward(head,fo,lo,batch);
        trix_cross_entropy_grad(dl,lo,yb,batch,n_classes);
        trix_atom_ffn_zero_grad(head);
        trix_atom_ffn_backward(head,fo,dl,dfo,batch);
        trix_ternary_route_zero_grad(ffn);
        trix_ternary_route_backward(ffn,po,dfo,dpo,batch);
        memset(dW2,0,D*H_proj*sizeof(float));
        memset(db2_g,0,D*sizeof(float));
        trix_matmul_at(dW2,dpo,h1,batch,D,H_proj);
        trix_bias_grad(db2_g,dpo,batch,D);
        trix_matmul(dh1,dpo,pW2,batch,D,H_proj);
        trix_gelu_grad(dz1,dh1,z1,batch*H_proj);
        memset(dW1,0,H_proj*input_dim*sizeof(float));
        memset(db1_g,0,H_proj*sizeof(float));
        trix_matmul_at(dW1,dz1,xb,batch,H_proj,input_dim);
        trix_bias_grad(db1_g,dz1,batch,H_proj);
        trix_sgd_update(pW1,dW1,lr,H_proj*input_dim);
        trix_sgd_update(pb1,db1_g,lr,H_proj);
        trix_sgd_update(pW2,dW2,lr,D*H_proj);
        trix_sgd_update(pb2,db2_g,lr,D);
        trix_ternary_route_clip_grad_norm(ffn,1.0f);
        trix_ternary_route_adamw_step(ffn,lr,0.9f,0.999f,1e-8f,0.01f);
        trix_atom_ffn_sgd_step(head,lr);
        free(xb); free(yb);
    }

    /* Eval */
    int cor=0;
    for(int o=0;o<nv;o+=batch){
        int n=(o+batch<=nv)?batch:nv-o;
        float* xb_eval=xv+o*input_dim;
        trix_matmul_bt(z1,xb_eval,pW1,n,input_dim,H_proj);
        trix_bias_add(z1,pb1,n,H_proj);
        trix_gelu(h1,z1,n*H_proj);
        trix_matmul_bt(po,h1,pW2,n,H_proj,D);
        trix_bias_add(po,pb2,n,D);
        trix_ternary_route_forward(ffn,po,fo,n);
        trix_atom_ffn_forward(head,fo,lo,n);
        int* pr=malloc(n*sizeof(int));
        trix_argmax(pr,lo,n,n_classes);
        for(int i=0;i<n;i++) if(pr[i]==yv[o+i]) cor++;
        free(pr);
    }
    float acc=(float)cor/(float)nv;
    printf("After 1 epoch: accuracy = %.2f%% (%d/%d)\n", acc*100, cor, nv);

    /* Dump weights */
    FILE* fout = fopen(output_path, "wb");
    if (!fout) { fprintf(stderr, "Cannot open %s for writing\n", output_path); return 1; }

    /* Header: magic + config */
    int32_t magic = 0x4D345457;  /* "M4TW" */
    fwrite(&magic, 4, 1, fout);
    int32_t cfg[6] = { input_dim, D, T, K, H_proj, n_classes };
    fwrite(cfg, sizeof(int32_t), 6, fout);

    /* Projection */
    save_floats(fout, pW1, H_proj * input_dim);
    save_floats(fout, pb1, H_proj);
    save_floats(fout, pW2, D * H_proj);
    save_floats(fout, pb2, D);

    /* FFN: LN params */
    save_floats(fout, ffn->ln_weight, D);
    save_floats(fout, ffn->ln_bias, D);

    /* FFN: per-tile ternary weights (quantized from shadow) and biases */
    /* Quantize shadow weights to ternary for the dump */
    int w1_elems = D * D;  /* tile_hidden == D for this config */
    int w2_elems = D * D;
    for (int t = 0; t < T; t++) {
        /* Quantize W1_t to ternary */
        float* sw1 = ffn->W1 + t * w1_elems;
        float mean_abs = 0;
        for (int i = 0; i < w1_elems; i++) mean_abs += fabsf(sw1[i]);
        mean_abs /= (float)w1_elems;
        float thresh = mean_abs * 0.5f;
        int8_t* tern = malloc(w1_elems);
        for (int i = 0; i < w1_elems; i++)
            tern[i] = (sw1[i] > thresh) ? 1 : (sw1[i] < -thresh) ? -1 : 0;
        int32_t c = w1_elems;
        fwrite(&c, 4, 1, fout);
        fwrite(tern, 1, w1_elems, fout);
        free(tern);

        save_floats(fout, ffn->b1 + t * D, D);

        /* Quantize W2_t */
        float* sw2 = ffn->W2 + t * w2_elems;
        mean_abs = 0;
        for (int i = 0; i < w2_elems; i++) mean_abs += fabsf(sw2[i]);
        mean_abs /= (float)w2_elems;
        thresh = mean_abs * 0.5f;
        tern = malloc(w2_elems);
        for (int i = 0; i < w2_elems; i++)
            tern[i] = (sw2[i] > thresh) ? 1 : (sw2[i] < -thresh) ? -1 : 0;
        c = w2_elems;
        fwrite(&c, 4, 1, fout);
        fwrite(tern, 1, w2_elems, fout);
        free(tern);

        save_floats(fout, ffn->b2 + t * D, D);
    }

    /* FFN: output_scale */
    save_floats(fout, &ffn->output_scale, 1);

    /* Head */
    save_floats(fout, head->W1, head->hidden_dim * D);
    save_floats(fout, head->b1, head->hidden_dim);
    save_floats(fout, head->W2, n_classes * head->hidden_dim);
    save_floats(fout, head->b2, n_classes);

    fclose(fout);
    printf("Weights saved to %s\n", output_path);

    /* Cleanup */
    free(pW1);free(pb1);free(pW2);free(pb2);
    free(z1);free(h1);free(po);free(fo);free(lo);
    free(dl);free(dfo);free(dpo);free(dh1);free(dz1);
    free(dW1);free(db1_g);free(dW2);free(db2_g);
    free(idx);
    free(xt);free(yt);free(xv);free(yv);
    trix_ternary_route_destroy(ffn);
    trix_atom_ffn_destroy(head);
    return 0;
}
