/*
 * mnist_probe_nproj16.c — atomic investigation of the vote-rule inversion
 * at N_PROJ=16.
 *
 * At N_PROJ=16, the full scaling curve shows majority beating rank-weighted:
 * a reversal of the pattern seen at N_PROJ ≥ 32. The mechanism cycle
 * predicted this: at small N_PROJ, the Hamming-distance space is small
 * (max = 2 × N_PROJ = 32 at N_PROJ=16), so 60K training prototypes cluster
 * on few distinct distance values. Top-k becomes dominated by ties;
 * rank-weighted amplifies the arbitrary tie-breaking noise.
 *
 * This tool measures the atomics of that regime:
 *   - For each test query, the full distribution of distances to 60K
 *     training prototypes.
 *   - Size of the tied-at-top-1 set per query.
 *   - Whether the correct class has prototypes in the tied set.
 *   - Conditional accuracy: majority vs rank-weighted recovery,
 *     split by "correct in tied top-1" vs "correct elsewhere."
 *
 * Output is plain text tables — this is diagnostic, not a benchmark.
 *
 * Usage: ./mnist_probe_nproj16 <mnist_dir>
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
#define N_PROJ 16
#define DENSITY 0.33
#define MAX_DIST (2 * N_PROJ)   /* 32 for N_PROJ=16 */

/* ── Data loaders, deskew, RNG, τ (minimal copies) ───────────────────── */

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

/* ── Main probe ───────────────────────────────────────────────────────── */

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

    int Sp=M4T_TRIT_PACKED_BYTES(N_PROJ);   /* 4 bytes at N_PROJ=16 */
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

    printf("N_PROJ=%d atomic probe — deskewed MNIST, density=%.2f, seed=42\n",
           N_PROJ, DENSITY);
    printf("Signature size: %d bytes.  Max Hamming distance: %d bits.\n",
           Sp, MAX_DIST);
    printf("%d train prototypes, %d test queries.\n\n", n_train, n_test);

    /* ── Aggregate statistics ────────────────────────────────────────── */

    /* Global distance distribution: count of (test, train) pairs at each distance. */
    long dist_count_global[MAX_DIST + 2] = {0};

    /* Per-query: min distance, count-at-min, count-at-min+1, count-at-min+2. */
    long min_dist_count[MAX_DIST + 2] = {0};   /* histogram of min distances */
    long tied_at_min_count[1001] = {0};         /* histogram: size of tied-min set */

    /* Conditional accuracy tracking:
     *   correct_in_tied_min: correct class has at least one prototype
     *     at the minimum distance.
     *   correct_elsewhere: correct class exists in top-10 but not at min. */
    int correct_in_tied_min = 0;
    int correct_elsewhere   = 0;
    int correct_nowhere_in_top10 = 0;

    /* Per-vote-rule recovery (majority k=7 and rank-wt k=7). */
    int maj_correct_total = 0, rank_correct_total = 0;
    int maj_correct_given_tied = 0, rank_correct_given_tied = 0;
    int maj_correct_given_elsewhere = 0, rank_correct_given_elsewhere = 0;

    /* Class distribution within tied-min sets (averaged across queries). */
    double avg_classes_in_tied = 0.0;

    /* Keep all 60K distances per query in a buffer (for the full probe). */
    int32_t* dists = malloc((size_t)n_train * sizeof(int32_t));
    int*     labels_by_idx = y_train;

    clock_t t0 = clock();
    for (int s = 0; s < n_test; s++) {
        const uint8_t* q_sig = test_sigs + (size_t)s*Sp;

        /* Compute all 60K distances, tally the global distribution. */
        for (int i = 0; i < n_train; i++) {
            const uint8_t* r_sig = train_sigs + (size_t)i*Sp;
            int32_t d = m4t_popcount_dist(q_sig, r_sig, mask, Sp);
            dists[i] = d;
            if (d >= 0 && d <= MAX_DIST) dist_count_global[d]++;
            else dist_count_global[MAX_DIST + 1]++;
        }

        /* Find min distance. */
        int32_t min_d = dists[0];
        for (int i = 1; i < n_train; i++)
            if (dists[i] < min_d) min_d = dists[i];
        if (min_d >= 0 && min_d <= MAX_DIST) min_dist_count[min_d]++;

        /* Count prototypes at exactly min_d, and build class histogram. */
        int tied_count = 0;
        int class_hist[N_CLASSES] = {0};
        int correct_class_in_tied = 0;
        for (int i = 0; i < n_train; i++) {
            if (dists[i] == min_d) {
                tied_count++;
                class_hist[labels_by_idx[i]]++;
                if (labels_by_idx[i] == y_test[s]) correct_class_in_tied = 1;
            }
        }
        if (tied_count < 1000) tied_at_min_count[tied_count]++;
        else tied_at_min_count[1000]++;   /* overflow bucket */

        int classes_present = 0;
        for (int c = 0; c < N_CLASSES; c++)
            if (class_hist[c] > 0) classes_present++;
        avg_classes_in_tied += (double)classes_present;

        /* Find the top-10 nearest training labels (insertion-sort). */
        int32_t top_d[10]; int top_l[10];
        for (int j = 0; j < 10; j++) { top_d[j] = INT32_MAX; top_l[j] = -1; }
        for (int i = 0; i < n_train; i++) {
            int32_t d = dists[i];
            if (d >= top_d[9]) continue;
            top_d[9] = d; top_l[9] = labels_by_idx[i];
            for (int j = 8; j >= 0; j--) {
                if (top_d[j+1] < top_d[j]) {
                    int32_t td = top_d[j]; top_d[j] = top_d[j+1]; top_d[j+1] = td;
                    int tl = top_l[j]; top_l[j] = top_l[j+1]; top_l[j+1] = tl;
                } else break;
            }
        }

        /* Classify the query by correct-class location. */
        int correct_in_top10 = 0;
        int correct_at_min = correct_class_in_tied;
        for (int j = 0; j < 10; j++)
            if (top_l[j] == y_test[s]) correct_in_top10 = 1;
        if (correct_at_min) correct_in_tied_min++;
        else if (correct_in_top10) correct_elsewhere++;
        else correct_nowhere_in_top10++;

        /* Majority k=7 vote. */
        int maj_counts[N_CLASSES] = {0};
        for (int j = 0; j < 7; j++) maj_counts[top_l[j]]++;
        int maj_pred = 0;
        for (int c = 1; c < N_CLASSES; c++)
            if (maj_counts[c] > maj_counts[maj_pred]) maj_pred = c;
        int maj_correct = (maj_pred == y_test[s]);
        maj_correct_total += maj_correct;
        if (correct_at_min) maj_correct_given_tied += maj_correct;
        else if (correct_in_top10) maj_correct_given_elsewhere += maj_correct;

        /* Rank-weighted k=7 vote. */
        int rank_scores[N_CLASSES] = {0};
        for (int j = 0; j < 7; j++) rank_scores[top_l[j]] += (7 - j);
        int rank_pred = 0;
        for (int c = 1; c < N_CLASSES; c++)
            if (rank_scores[c] > rank_scores[rank_pred]) rank_pred = c;
        int rank_correct = (rank_pred == y_test[s]);
        rank_correct_total += rank_correct;
        if (correct_at_min) rank_correct_given_tied += rank_correct;
        else if (correct_in_top10) rank_correct_given_elsewhere += rank_correct;
    }
    double elapsed = (double)(clock()-t0)/CLOCKS_PER_SEC;

    printf("Inference took %.1f s.\n\n", elapsed);

    /* ── Report: global distance distribution ────────────────────────── */

    printf("Global Hamming-distance distribution (test × train = %ld pairs):\n",
           (long)n_test * (long)n_train);
    printf("  dist  count         fraction\n");
    long total_pairs = (long)n_test * (long)n_train;
    for (int d = 0; d <= MAX_DIST; d++) {
        if (dist_count_global[d] == 0) continue;
        printf("  %3d   %-12ld  %.3f%%\n",
               d, dist_count_global[d],
               100.0 * dist_count_global[d] / total_pairs);
    }
    printf("\n");

    /* ── Report: min-distance histogram ──────────────────────────────── */

    printf("Min-distance-per-query histogram (%d queries):\n", n_test);
    printf("  min_d  queries       fraction\n");
    for (int d = 0; d <= MAX_DIST; d++) {
        if (min_dist_count[d] == 0) continue;
        printf("  %3d    %-12ld  %.2f%%\n",
               d, min_dist_count[d],
               100.0 * min_dist_count[d] / n_test);
    }
    printf("\n");

    /* ── Report: tied-at-min-size histogram ──────────────────────────── */

    printf("Tied-at-top-1 set size histogram:\n");
    printf("  size_bucket           queries       fraction\n");
    struct { int lo, hi; } buckets[] = {
        {1, 1}, {2, 5}, {6, 10}, {11, 25}, {26, 50},
        {51, 100}, {101, 250}, {251, 500}, {501, 1000}, {1001, 100000}
    };
    int n_buckets = sizeof(buckets)/sizeof(buckets[0]);
    for (int b = 0; b < n_buckets; b++) {
        long sum = 0;
        int upper = (buckets[b].hi >= 1000) ? 1000 : buckets[b].hi;
        for (int s = buckets[b].lo; s <= upper && s < 1001; s++)
            sum += tied_at_min_count[s];
        if (sum == 0) continue;
        if (buckets[b].lo == buckets[b].hi)
            printf("  exactly %-4d          %-12ld  %.2f%%\n",
                   buckets[b].lo, sum, 100.0 * sum / n_test);
        else
            printf("  [%4d .. %4d]         %-12ld  %.2f%%\n",
                   buckets[b].lo, buckets[b].hi, sum, 100.0 * sum / n_test);
    }
    printf("\n");

    avg_classes_in_tied /= n_test;
    printf("Average number of distinct classes in tied-top-1 set: %.2f / 10\n\n",
           avg_classes_in_tied);

    /* ── Report: conditional accuracy ────────────────────────────────── */

    printf("Correct-class location partition:\n");
    printf("  correct class in tied-min set:   %-5d  (%.2f%%)\n",
           correct_in_tied_min, 100.0*correct_in_tied_min/n_test);
    printf("  correct class elsewhere in top-10: %-5d  (%.2f%%)\n",
           correct_elsewhere, 100.0*correct_elsewhere/n_test);
    printf("  correct class nowhere in top-10: %-5d  (%.2f%%)\n",
           correct_nowhere_in_top10, 100.0*correct_nowhere_in_top10/n_test);
    printf("\n");

    printf("Accuracy by vote rule (k=7):\n");
    printf("  majority:      %d / %d = %.2f%%\n",
           maj_correct_total, n_test, 100.0*maj_correct_total/n_test);
    printf("  rank-weighted: %d / %d = %.2f%%\n",
           rank_correct_total, n_test, 100.0*rank_correct_total/n_test);
    printf("\n");

    printf("Conditional recovery (given correct class location):\n");
    if (correct_in_tied_min > 0)
        printf("  When correct IS in tied-min set:\n"
               "    majority k=7:     %d / %d = %.2f%%\n"
               "    rank-wt  k=7:     %d / %d = %.2f%%\n"
               "    Δ (rank - maj):   %+.2f%%\n",
               maj_correct_given_tied, correct_in_tied_min,
               100.0*maj_correct_given_tied/correct_in_tied_min,
               rank_correct_given_tied, correct_in_tied_min,
               100.0*rank_correct_given_tied/correct_in_tied_min,
               100.0*(rank_correct_given_tied - maj_correct_given_tied)/correct_in_tied_min);
    if (correct_elsewhere > 0)
        printf("  When correct is at ranks 2-10 (NOT in tied-min):\n"
               "    majority k=7:     %d / %d = %.2f%%\n"
               "    rank-wt  k=7:     %d / %d = %.2f%%\n"
               "    Δ (rank - maj):   %+.2f%%\n",
               maj_correct_given_elsewhere, correct_elsewhere,
               100.0*maj_correct_given_elsewhere/correct_elsewhere,
               rank_correct_given_elsewhere, correct_elsewhere,
               100.0*rank_correct_given_elsewhere/correct_elsewhere,
               100.0*(rank_correct_given_elsewhere - maj_correct_given_elsewhere)/correct_elsewhere);
    printf("\n");

    printf("Mechanism read-out:\n");
    printf("  The 'tied-min' partition captures WHERE the information is.\n");
    printf("  If majority recovers more tied-min cases than rank-wt,\n");
    printf("  rank-weighted's profile amplification is costing accuracy\n");
    printf("  in the ambiguous regime. Conversely if rank-wt wins on the\n");
    printf("  elsewhere partition (correct not in tied-min), its\n");
    printf("  profile helps when signal is concentrated at the front.\n");

    free(dists);
    free(mask);
    free(train_sigs); free(test_sigs);
    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
