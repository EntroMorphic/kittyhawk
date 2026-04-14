/*
 * mnist_knn_lattice.c — k-NN on the trit lattice. Zero float.
 *
 * Multi-channel features inspired by SSTT:
 *   - Pixel intensity (deskewed)
 *   - Horizontal gradient: sign(pixel[y][x+1] - pixel[y][x])
 *   - Vertical gradient: sign(pixel[y+1][x] - pixel[y][x])
 *   - Enclosed-region count via flood-fill (topological feature)
 *
 * Usage: ./mnist_knn_lattice <mnist_dir>
 */

#include "m4t_types.h"
#include "m4t_mtfp.h"
#include "m4t_trit_pack.h"
#include "m4t_ternary_matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define INPUT_DIM 784
#define IMG_W 28
#define IMG_H 28
#define N_CLASSES 10
#define MAX_K 7

/* Feature dimensions: pixel(784) + h_grad(784) + v_grad(784) + topo(1) */
#define FEAT_DIM_PIXEL  784
#define FEAT_DIM_GRAD   (784 + 784)
#define FEAT_DIM_TOPO   1
#define FEAT_DIM_FULL   (FEAT_DIM_PIXEL + FEAT_DIM_GRAD + FEAT_DIM_TOPO)

static uint32_t read_u32_be(FILE* f) {
    uint8_t b[4]; fread(b,1,4,f);
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];
}
static m4t_mtfp_t* load_images_mtfp(const char* path, int* n) {
    FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",path);exit(1);}
    read_u32_be(f); *n=(int)read_u32_be(f);
    int rows=(int)read_u32_be(f),cols=(int)read_u32_be(f);
    (void)rows;(void)cols;
    size_t total=(size_t)(*n)*INPUT_DIM;
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

/* ── Integer deskewing ─────────────────────────────────────────────────── */

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

/* ── Gradient channels ─────────────────────────────────────────────────── */

/* Compute horizontal and vertical gradients as MTFP values.
 * h_grad[y][x] = pixel[y][x+1] - pixel[y][x]  (0 at right edge)
 * v_grad[y][x] = pixel[y+1][x] - pixel[y][x]  (0 at bottom edge)
 * These are NOT sign-extracted — we keep full MTFP magnitude for L2 distance. */

static void compute_gradients(m4t_mtfp_t* h_grad, m4t_mtfp_t* v_grad,
                               const m4t_mtfp_t* pixels) {
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            int idx = y * IMG_W + x;
            /* Horizontal gradient */
            if (x + 1 < IMG_W)
                h_grad[idx] = pixels[idx + 1] - pixels[idx];
            else
                h_grad[idx] = 0;
            /* Vertical gradient */
            if (y + 1 < IMG_H)
                v_grad[idx] = pixels[idx + IMG_W] - pixels[idx];
            else
                v_grad[idx] = 0;
        }
    }
}

/* ── Flood-fill enclosed-region count ──────────────────────────────────── */

/* Count enclosed regions (holes) in the digit.
 * Binarize: pixel > threshold → foreground (1), else background (0).
 * Flood-fill from all border background pixels. Any background pixel
 * NOT reached by the flood is an enclosed region.
 * Count connected components of unreached background = enclosed regions.
 *
 * Digit 0: 1 hole. Digit 4: 0-1 holes. Digit 8: 2 holes.
 * Digit 9: 1 hole. Digit 6: 1 hole.
 *
 * All integer: threshold compare, queue-based flood, connected-component count. */

static int32_t count_enclosed_regions(const m4t_mtfp_t* pixels) {
    /* Binarize at 50% intensity. MTFP cell for 0.5 = SCALE/2 = 29524. */
    int32_t threshold = M4T_MTFP_SCALE / 2;
    uint8_t fg[IMG_H][IMG_W];  /* 1 = foreground, 0 = background */
    uint8_t reached[IMG_H][IMG_W];  /* 1 = reached by border flood */

    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++)
            fg[y][x] = (pixels[y * IMG_W + x] > threshold) ? 1 : 0;

    /* Flood-fill from all border background pixels. */
    memset(reached, 0, sizeof(reached));
    int16_t queue[IMG_H * IMG_W * 2];  /* (y, x) pairs */
    int qhead = 0, qtail = 0;

    /* Seed: all border pixels that are background */
    for (int x = 0; x < IMG_W; x++) {
        if (!fg[0][x] && !reached[0][x])
            { reached[0][x]=1; queue[qtail++]=0; queue[qtail++]=x; }
        if (!fg[IMG_H-1][x] && !reached[IMG_H-1][x])
            { reached[IMG_H-1][x]=1; queue[qtail++]=IMG_H-1; queue[qtail++]=x; }
    }
    for (int y = 1; y < IMG_H-1; y++) {
        if (!fg[y][0] && !reached[y][0])
            { reached[y][0]=1; queue[qtail++]=y; queue[qtail++]=0; }
        if (!fg[y][IMG_W-1] && !reached[y][IMG_W-1])
            { reached[y][IMG_W-1]=1; queue[qtail++]=y; queue[qtail++]=IMG_W-1; }
    }

    /* BFS flood */
    while (qhead < qtail) {
        int cy = queue[qhead++], cx = queue[qhead++];
        static const int dy4[] = {-1, 1, 0, 0};
        static const int dx4[] = {0, 0, -1, 1};
        for (int d = 0; d < 4; d++) {
            int ny = cy + dy4[d], nx = cx + dx4[d];
            if (ny >= 0 && ny < IMG_H && nx >= 0 && nx < IMG_W
                && !fg[ny][nx] && !reached[ny][nx]) {
                reached[ny][nx] = 1;
                queue[qtail++] = ny; queue[qtail++] = nx;
            }
        }
    }

    /* Count connected components of unreached background = enclosed regions. */
    int regions = 0;
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++) {
            if (!fg[y][x] && !reached[y][x]) {
                /* New enclosed region — flood-fill to mark it */
                regions++;
                qhead = qtail = 0;
                reached[y][x] = 1;
                queue[qtail++] = y; queue[qtail++] = x;
                while (qhead < qtail) {
                    int cy2 = queue[qhead++], cx2 = queue[qhead++];
                    static const int dy4[] = {-1, 1, 0, 0};
                    static const int dx4[] = {0, 0, -1, 1};
                    for (int d = 0; d < 4; d++) {
                        int ny = cy2+dy4[d], nx = cx2+dx4[d];
                        if (ny>=0 && ny<IMG_H && nx>=0 && nx<IMG_W
                            && !fg[ny][nx] && !reached[ny][nx]) {
                            reached[ny][nx] = 1;
                            queue[qtail++] = ny; queue[qtail++] = nx;
                        }
                    }
                }
            }
        }

    return (int32_t)regions;
}

/* ── Build feature vectors ─────────────────────────────────────────────── */

/* Full feature: [pixel(784), h_grad(784), v_grad(784), topo(1)] = 2353 dims.
 * Topo feature is scaled to be comparable magnitude to pixel features. */

#define TOPO_SCALE (M4T_MTFP_SCALE * 5)  /* scale enclosed-region count */

static void build_features(m4t_mtfp_t* features, const m4t_mtfp_t* pixels, int n) {
    m4t_mtfp_t h_grad[INPUT_DIM], v_grad[INPUT_DIM];
    for (int i = 0; i < n; i++) {
        const m4t_mtfp_t* px = pixels + (size_t)i * INPUT_DIM;
        m4t_mtfp_t* feat = features + (size_t)i * FEAT_DIM_FULL;

        /* Pixels */
        memcpy(feat, px, FEAT_DIM_PIXEL * sizeof(m4t_mtfp_t));

        /* Gradients */
        compute_gradients(h_grad, v_grad, px);
        memcpy(feat + FEAT_DIM_PIXEL, h_grad, INPUT_DIM * sizeof(m4t_mtfp_t));
        memcpy(feat + FEAT_DIM_PIXEL + INPUT_DIM, v_grad, INPUT_DIM * sizeof(m4t_mtfp_t));

        /* Topological: enclosed region count */
        int32_t regions = count_enclosed_regions(px);
        feat[FEAT_DIM_PIXEL + FEAT_DIM_GRAD] = (m4t_mtfp_t)(regions * TOPO_SCALE);
    }
}

/* ── k-NN engine ───────────────────────────────────────────────────────── */

/* Per-channel weighted k-NN.
 * feat_dim = FEAT_DIM_FULL (pixel 784 + grad 1568 + topo 1).
 * Weights are integer multipliers applied to each channel's squared L2.
 * A weight of 0 drops the channel entirely. */
static int knn_classify_weighted(
    const m4t_mtfp_t* query,
    const m4t_mtfp_t* train_feats, const int* y_train, int n_train,
    int K, int w_pix, int w_grad, int w_topo)
{
    const int feat_dim = FEAT_DIM_FULL;
    int64_t knn_dist[MAX_K];
    int knn_label[MAX_K];
    for(int j=0;j<K;j++){knn_dist[j]=INT64_MAX;knn_label[j]=-1;}

    for(int i=0;i<n_train;i++){
        const m4t_mtfp_t* ref=train_feats+(size_t)i*feat_dim;
        int64_t d_pix=0, d_grad=0, d_topo=0;
        for(int p=0;p<FEAT_DIM_PIXEL;p++){
            int64_t d=(int64_t)query[p]-(int64_t)ref[p];
            d_pix+=d*d;
        }
        for(int p=FEAT_DIM_PIXEL;p<FEAT_DIM_PIXEL+FEAT_DIM_GRAD;p++){
            int64_t d=(int64_t)query[p]-(int64_t)ref[p];
            d_grad+=d*d;
        }
        {
            int p=FEAT_DIM_PIXEL+FEAT_DIM_GRAD;
            int64_t d=(int64_t)query[p]-(int64_t)ref[p];
            d_topo+=d*d;
        }
        int64_t dist = (int64_t)w_pix*d_pix + (int64_t)w_grad*d_grad + (int64_t)w_topo*d_topo;
        if(dist<knn_dist[K-1]){
            knn_dist[K-1]=dist; knn_label[K-1]=y_train[i];
            for(int j=K-2;j>=0;j--){
                if(knn_dist[j+1]<knn_dist[j]){
                    int64_t td=knn_dist[j];knn_dist[j]=knn_dist[j+1];knn_dist[j+1]=td;
                    int tl=knn_label[j];knn_label[j]=knn_label[j+1];knn_label[j+1]=tl;
                }else break;
            }
        }
    }

    int votes[N_CLASSES]; memset(votes,0,sizeof(votes));
    for(int j=0;j<K;j++) if(knn_label[j]>=0) votes[knn_label[j]]++;
    int pred=0;
    for(int c=1;c<N_CLASSES;c++) if(votes[c]>votes[pred]) pred=c;
    return pred;
}

static void run_experiment(const char* label,
    const m4t_mtfp_t* test_feats, int n_test, const int* y_test,
    const m4t_mtfp_t* train_feats, int n_train, const int* y_train,
    int feat_dim, int K, int w_pix, int w_grad, int w_topo)
{
    (void)feat_dim;
    int correct=0;
    int confusion[N_CLASSES][N_CLASSES];
    memset(confusion,0,sizeof(confusion));

    for(int s=0;s<n_test;s++){
        int pred=knn_classify_weighted(test_feats+(size_t)s*FEAT_DIM_FULL,
                              train_feats,y_train,n_train,K,
                              w_pix,w_grad,w_topo);
        if(pred==y_test[s]) correct++;
        confusion[y_test[s]][pred]++;
        if(s>0 && s%2000==0)
            printf("    %d/%d — %d.%02d%%\n",s,n_test,
                   correct*100/s,(correct*10000/s)%100);
    }
    printf("  %s: %d/%d = %d.%02d%%\n",label,correct,n_test,
           correct*100/n_test,(correct*10000/n_test)%100);

    /* Print top confusions */
    printf("  Top confusions: ");
    for(int pass=0;pass<5;pass++){
        int best_r=-1,best_c=-1,best_v=0;
        for(int r=0;r<N_CLASSES;r++)
            for(int c=0;c<N_CLASSES;c++)
                if(r!=c && confusion[r][c]>best_v)
                    {best_v=confusion[r][c];best_r=r;best_c=c;}
        if(best_v>0){
            printf("%d→%d(%d) ",best_r,best_c,best_v);
            confusion[best_r][best_c]=0;
        }
    }
    printf("\n\n");
}

/* ── Main ──────────────────────────────────────────────────────────────── */

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

    printf("k-NN on the Trit Lattice — Multi-Channel (zero float)\n");
    printf("Loaded %d train, %d test\n\n", n_train, n_test);

    /* Deskew */
    printf("Deskewing...\n");
    deskew_all(x_train, n_train);
    deskew_all(x_test, n_test);

    printf("Building multi-channel features (pixel + h_grad + v_grad + topo)...\n");
    printf("  Feature dim: %d (pixel=%d + grad=%d + topo=%d)\n",
           FEAT_DIM_FULL, FEAT_DIM_PIXEL, FEAT_DIM_GRAD, FEAT_DIM_TOPO);

    m4t_mtfp_t* train_feat = malloc((size_t)n_train * FEAT_DIM_FULL * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* test_feat = malloc((size_t)n_test * FEAT_DIM_FULL * sizeof(m4t_mtfp_t));

    printf("  Computing train features...\n");
    build_features(train_feat, x_train, n_train);
    printf("  Computing test features...\n");
    build_features(test_feat, x_test, n_test);

    /* ── Per-channel weighted sweep ───────────────────────────────────────
     * Weights scale squared L2 per channel. Grad has 2x dims vs pixel,
     * so unweighted grad dominates. Hypothesis: deweight grad, weight topo up. */

    /* A: pixel-only baseline (reproduces 97.61%) */
    printf("\n=== A: pixel-only k=3 L2 (baseline) ===\n");
    run_experiment("pixel k=3", test_feat, n_test, y_test,
                   train_feat, n_train, y_train, FEAT_DIM_FULL, 3, 1, 0, 0);

    /* B: pixel + topo only (drop grad) */
    printf("=== B: pixel + topo k=3 L2 (w_pix=1, w_topo=1) ===\n");
    run_experiment("pix+topo k=3", test_feat, n_test, y_test,
                   train_feat, n_train, y_train, FEAT_DIM_FULL, 3, 1, 0, 1);

    /* C: pixel + grad/4 (grad weight = 1, pixel weight = 4 → grad effective 1/4) */
    printf("=== C: pixel + grad/4 k=3 L2 (w_pix=4, w_grad=1) ===\n");
    run_experiment("pix+grad/4 k=3", test_feat, n_test, y_test,
                   train_feat, n_train, y_train, FEAT_DIM_FULL, 3, 4, 1, 0);

    /* D: pixel + grad/8 */
    printf("=== D: pixel + grad/8 k=3 L2 (w_pix=8, w_grad=1) ===\n");
    run_experiment("pix+grad/8 k=3", test_feat, n_test, y_test,
                   train_feat, n_train, y_train, FEAT_DIM_FULL, 3, 8, 1, 0);

    /* E: pixel + grad/4 + topo */
    printf("=== E: pixel + grad/4 + topo k=3 L2 (4,1,4) ===\n");
    run_experiment("pix+grad/4+topo k=3", test_feat, n_test, y_test,
                   train_feat, n_train, y_train, FEAT_DIM_FULL, 3, 4, 1, 4);

    /* F: pixel + grad/8 + topo */
    printf("=== F: pixel + grad/8 + topo k=3 L2 (8,1,8) ===\n");
    run_experiment("pix+grad/8+topo k=3", test_feat, n_test, y_test,
                   train_feat, n_train, y_train, FEAT_DIM_FULL, 3, 8, 1, 8);

    printf("Zero float. Zero gradients. Pure lattice geometry.\n");

    free(x_train);free(y_train);free(x_test);free(y_test);
    free(train_feat);free(test_feat);
    return 0;
}
