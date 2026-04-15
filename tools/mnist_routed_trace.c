/*
 * mnist_routed_trace.c — inspectability demo for routed k-NN decisions.
 *
 * Dense k-NN produces a scalar L1 distance per prototype — a summed black
 * box you can't decompose. Routed k-NN produces a Hamming distance that
 * IS a per-trit sum of {0, 1, 2} cost contributions. Every classification
 * has a readable audit trail by construction.
 *
 * For each misclassified MNIST test image, this tool prints:
 *   - The top-5 nearest training prototypes (idx, label, Hamming distance).
 *   - The vote composition at k=3.
 *   - Per-trit breakdown vs the nearest prototype (agreements, sign flips,
 *     zero-vs-sign mismatches).
 *   - Per-class nearest-prototype distance across ALL 60 000 prototypes.
 *   - A failure-type classification from the above numbers.
 *
 * At the end: aggregate stats over the full misclassification set.
 *
 * Usage: ./mnist_routed_trace <mnist_dir>
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
#define N_PROJ 2048
#define TRACE_N 8           /* detailed traces to print */
#define DENSITY 0.33

/* ── Data + deskew (same as the main routed k-NN tool) ────────────────── */

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

/* Decode one 2-bit trit code at position p in a packed-trit buffer. */
static inline int trit_at(const uint8_t* packed, int p) {
    uint8_t code = (packed[p >> 2] >> ((p & 3) * 2)) & 0x3u;
    return (code == 0x01u) ?  1 :
           (code == 0x02u) ? -1 : 0;
}

/* ── Top-k with indices (for the full audit trail) ────────────────────── */

#define MAX_K 5
static void topk_insert(int32_t* dists, int* labels, int* indices, int k,
                         int32_t new_d, int new_l, int new_i)
{
    if (new_d >= dists[k-1]) return;
    dists[k-1]=new_d; labels[k-1]=new_l; indices[k-1]=new_i;
    for (int j=k-2; j>=0; j--) {
        if (dists[j+1]<dists[j]) {
            int32_t d=dists[j]; dists[j]=dists[j+1]; dists[j+1]=d;
            int l=labels[j]; labels[j]=labels[j+1]; labels[j+1]=l;
            int i=indices[j]; indices[j]=indices[j+1]; indices[j+1]=i;
        } else break;
    }
}

static int majority_vote(const int* labels, int k) {
    int counts[N_CLASSES]={0};
    for (int i=0;i<k;i++) counts[labels[i]]++;
    int best=0;
    for (int c=1;c<N_CLASSES;c++) if (counts[c]>counts[best]) best=c;
    return best;
}

/* ── Trit-by-trit breakdown ───────────────────────────────────────────── */

typedef struct {
    int agree_plus;    /* both +1 */
    int agree_zero;    /* both 0 */
    int agree_minus;   /* both -1 */
    int sign_flip;     /* +1 vs -1 or -1 vs +1  → 2 bits each */
    int zero_vs_pos;   /* 0 vs +1 or +1 vs 0   → 1 bit each */
    int zero_vs_neg;   /* 0 vs -1 or -1 vs 0   → 1 bit each */
    int32_t total_distance;
} trit_breakdown_t;

static trit_breakdown_t breakdown(const uint8_t* q, const uint8_t* p, int n) {
    trit_breakdown_t b = {0,0,0,0,0,0,0};
    for (int i = 0; i < n; i++) {
        int tq = trit_at(q, i);
        int tp = trit_at(p, i);
        if (tq == tp) {
            if (tq == 1) b.agree_plus++;
            else if (tq == -1) b.agree_minus++;
            else b.agree_zero++;
        } else if ((tq == 1 && tp == -1) || (tq == -1 && tp == 1)) {
            b.sign_flip++;
        } else if (tq == 0) {
            if (tp == 1) b.zero_vs_pos++; else b.zero_vs_neg++;
        } else {
            if (tq == 1) b.zero_vs_pos++; else b.zero_vs_neg++;
        }
    }
    b.total_distance = 2 * b.sign_flip + b.zero_vs_pos + b.zero_vs_neg;
    return b;
}

/* ── Classify the failure mode ────────────────────────────────────────── */

static const char* classify_failure(
    int true_label, int pred_label,
    const int32_t* class_best_dist, int top1_dist, int N_PROJ_)
{
    int32_t correct_dist = class_best_dist[true_label];
    int32_t pred_dist = class_best_dist[pred_label];
    int32_t max_possible = 2 * N_PROJ_;

    /* Outlier: even the nearest-class distance is high. */
    if (top1_dist > max_possible * 6 / 10)
        return "OUTLIER (no close prototype in any class)";

    /* Rescue: correct class was very close to being nearest. */
    if (correct_dist - pred_dist <= 10)
        return "NARROW MISS (correct-class prototype within 10 bits)";

    /* Visual confusion: pred class and true class both have close prototypes. */
    if (correct_dist <= pred_dist + 50 && correct_dist < max_possible / 2)
        return "VISUAL CONFUSION (both classes have near-lattice prototypes)";

    /* Default. */
    return "SEPARATED (correct-class prototypes far from this query)";
}

/* ── Main ─────────────────────────────────────────────────────────────── */

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

    /* Deskewed config — the strongest routed deployment. */
    deskew_all(x_train, n_train);
    deskew_all(x_test, n_test);

    printf("Routed k-NN inspectability trace — MNIST deskewed N_PROJ=%d k=3\n",
           N_PROJ);
    printf("%d train / %d test; density=%.2f\n\n", n_train, n_test, DENSITY);

    /* Single deterministic run. */
    rng_s[0]=42; rng_s[1]=123; rng_s[2]=456; rng_s[3]=789;
    m4t_trit_t* proj_w=malloc((size_t)N_PROJ*INPUT_DIM);
    for(int i=0;i<N_PROJ*INPUT_DIM;i++){
        uint32_t r=rng_next()%3;
        proj_w[i]=(r==0)?-1:(r==1)?0:1;
    }
    int proj_Dp=M4T_TRIT_PACKED_BYTES(INPUT_DIM);
    uint8_t* proj_packed=malloc((size_t)N_PROJ*proj_Dp);
    m4t_pack_trits_rowmajor(proj_packed,proj_w,N_PROJ,INPUT_DIM);
    free(proj_w);

    m4t_mtfp_t* train_proj=malloc((size_t)n_train*N_PROJ*sizeof(m4t_mtfp_t));
    m4t_mtfp_t* test_proj =malloc((size_t)n_test *N_PROJ*sizeof(m4t_mtfp_t));
    for(int i=0;i<n_train;i++)
        m4t_mtfp_ternary_matmul_bt(train_proj+(size_t)i*N_PROJ,
                                    x_train+(size_t)i*INPUT_DIM,
                                    proj_packed,1,INPUT_DIM,N_PROJ);
    for(int i=0;i<n_test;i++)
        m4t_mtfp_ternary_matmul_bt(test_proj+(size_t)i*N_PROJ,
                                    x_test+(size_t)i*INPUT_DIM,
                                    proj_packed,1,INPUT_DIM,N_PROJ);

    /* Calibrate tau_q. */
    int64_t tau_q;
    {
        size_t total=(size_t)1000*N_PROJ;
        int64_t* buf=malloc(total*sizeof(int64_t));
        for(int i=0;i<1000;i++)
            for(int p=0;p<N_PROJ;p++){
                int64_t v=train_proj[(size_t)i*N_PROJ+p];
                buf[(size_t)i*N_PROJ+p]=(v>=0)?v:-v;
            }
        tau_q=tau_for_density(buf,total,DENSITY);
        free(buf);
    }
    printf("tau_q = %lld (calibrated to density %.2f)\n\n",
           (long long)tau_q, DENSITY);

    /* Extract signatures. */
    int Sp=M4T_TRIT_PACKED_BYTES(N_PROJ);
    uint8_t* train_sigs=calloc((size_t)n_train*Sp,1);
    uint8_t* test_sigs =calloc((size_t)n_test *Sp,1);
    int64_t* tmp=malloc((size_t)N_PROJ*sizeof(int64_t));
    for(int i=0;i<n_train;i++){
        for(int p=0;p<N_PROJ;p++) tmp[p]=(int64_t)train_proj[(size_t)i*N_PROJ+p];
        m4t_route_threshold_extract(train_sigs+(size_t)i*Sp,tmp,tau_q,N_PROJ);
    }
    for(int i=0;i<n_test;i++){
        for(int p=0;p<N_PROJ;p++) tmp[p]=(int64_t)test_proj[(size_t)i*N_PROJ+p];
        m4t_route_threshold_extract(test_sigs+(size_t)i*Sp,tmp,tau_q,N_PROJ);
    }
    free(tmp);
    free(train_proj); free(test_proj); free(proj_packed);

    uint8_t* mask=malloc(Sp); memset(mask,0xFF,Sp);

    /* Inference loop with full per-class distance tracking. */
    int n_correct = 0, n_misclassified = 0, n_traced = 0;
    int failure_counts[4] = {0,0,0,0};  /* narrow, visual, separated, outlier */
    const char* failure_names[4] = {"NARROW MISS", "VISUAL CONFUSION",
                                     "SEPARATED", "OUTLIER"};

    int32_t* class_best_dist = malloc(N_CLASSES * sizeof(int32_t));

    clock_t t0 = clock();
    for (int s = 0; s < n_test; s++) {
        const uint8_t* q_sig = test_sigs + (size_t)s*Sp;
        int32_t dists[MAX_K]; int labels[MAX_K]; int indices[MAX_K];
        for (int j = 0; j < MAX_K; j++) {
            dists[j] = INT32_MAX; labels[j] = -1; indices[j] = -1;
        }
        for (int c = 0; c < N_CLASSES; c++) class_best_dist[c] = INT32_MAX;

        for (int i = 0; i < n_train; i++) {
            const uint8_t* r_sig = train_sigs + (size_t)i*Sp;
            int32_t d = m4t_popcount_dist(q_sig, r_sig, mask, Sp);
            topk_insert(dists, labels, indices, MAX_K, d, y_train[i], i);
            if (d < class_best_dist[y_train[i]]) class_best_dist[y_train[i]] = d;
        }

        int pred = majority_vote(labels, 3);
        if (pred == y_test[s]) {
            n_correct++;
        } else {
            n_misclassified++;
            const char* ftype = classify_failure(y_test[s], pred,
                                                  class_best_dist, dists[0], N_PROJ);
            if (strstr(ftype, "NARROW")) failure_counts[0]++;
            else if (strstr(ftype, "VISUAL")) failure_counts[1]++;
            else if (strstr(ftype, "OUTLIER")) failure_counts[3]++;
            else failure_counts[2]++;

            if (n_traced < TRACE_N) {
                n_traced++;
                const uint8_t* top1_sig = train_sigs + (size_t)indices[0] * Sp;
                trit_breakdown_t bd = breakdown(q_sig, top1_sig, N_PROJ);

                printf("── Misclassified test #%d ─────────────────────────\n", s);
                printf("  True label: %d    Predicted: %d    Failure: %s\n",
                       y_test[s], pred, ftype);
                printf("  Top-5 neighbors (train_idx, label, Hamming dist):\n");
                for (int j = 0; j < MAX_K; j++) {
                    int is_correct = (labels[j] == y_test[s]) ? 1 : 0;
                    printf("    #%d: idx=%5d  label=%d  dist=%4d%s\n",
                           j+1, indices[j], labels[j], dists[j],
                           is_correct ? "  ← correct class" : "");
                }
                int vote[N_CLASSES] = {0};
                for (int j = 0; j < 3; j++) vote[labels[j]]++;
                printf("  Vote at k=3:");
                for (int c = 0; c < N_CLASSES; c++)
                    if (vote[c] > 0) printf(" [%d]=%d", c, vote[c]);
                printf("\n");
                printf("  Per-trit breakdown vs top-1 (total trits=%d):\n", N_PROJ);
                printf("    agreements: +%d / 0:%d / -%d  (total %d, %.1f%%)\n",
                       bd.agree_plus, bd.agree_zero, bd.agree_minus,
                       bd.agree_plus + bd.agree_zero + bd.agree_minus,
                       100.0 * (bd.agree_plus+bd.agree_zero+bd.agree_minus) / N_PROJ);
                printf("    sign flips (±1 vs ∓1): %d  (cost %d)\n",
                       bd.sign_flip, 2*bd.sign_flip);
                printf("    zero-vs-sign: %d+ / %d-  (cost %d)\n",
                       bd.zero_vs_pos, bd.zero_vs_neg,
                       bd.zero_vs_pos + bd.zero_vs_neg);
                printf("    total Hamming distance: %d (=popcount_dist)\n",
                       bd.total_distance);
                printf("  Per-class nearest-prototype distance:\n   ");
                for (int c = 0; c < N_CLASSES; c++) {
                    const char* marker =
                        (c == y_test[s]) ? "*" :
                        (c == pred)      ? "→" : " ";
                    printf(" %s[%d]:%d", marker, c, class_best_dist[c]);
                }
                printf("\n\n");
            }
        }
    }
    double inf_s = (double)(clock()-t0)/CLOCKS_PER_SEC;

    /* Aggregate. */
    printf("=== Aggregate summary ===\n");
    printf("  Test set size: %d\n", n_test);
    printf("  Correct: %d (%.2f%%)    Misclassified: %d (%.2f%%)\n",
           n_correct, 100.0*n_correct/n_test,
           n_misclassified, 100.0*n_misclassified/n_test);
    printf("  Failure type distribution:\n");
    for (int i = 0; i < 4; i++)
        printf("    %-20s %d (%.1f%% of misclassified)\n",
               failure_names[i], failure_counts[i],
               100.0 * failure_counts[i] / (n_misclassified > 0 ? n_misclassified : 1));
    printf("  Inference time: %.1f s\n\n", inf_s);

    printf("This audit trail is IMPOSSIBLE with dense k-NN.\n");
    printf("Dense L1 distances are scalar sums — no per-position decomposition.\n");
    printf("Routed Hamming distances are compositional sums of {0, 1, 2} per trit.\n");
    printf("That's a structural property of the routing surface, not marketing.\n");

    free(class_best_dist);
    free(train_sigs); free(test_sigs); free(mask);
    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
