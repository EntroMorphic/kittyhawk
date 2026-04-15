/*
 * STATUS: research scaffolding, not production architecture.
 * Runs routing primitives inside an O(N_train) dense outer loop.
 * Produced cascade atomics that motivated the bucket architecture.
 * For production routed k-NN use tools/mnist_routed_bucket{,_multi}.c
 * on libglyph. See docs/FINDINGS.md Axis 5 and
 * journal/routed_bucket_consumer.md.
 *
 * mnist_cascade_atomics.c — atomic decomposition of WHY the N_PROJ=16
 * routed cascade works.
 *
 * Sixth-round remediation removes the dense pixel resolver from this probe.
 * The cascade variant studied is now: primary-hash top-K filter followed by
 * secondary-hash 1-NN within top-K.
 *
 * This probe dissects the mechanism:
 *
 *   A. Ceiling-in-top-K curve (K = 1..200). How much does widening help?
 *   B. Conditional resolver accuracy: GIVEN correct is in top-K, does the
 *      secondary routed hash pick it?
 *   C. Rescue/damage matrix at K=50: 2x2 of {pure-hash-top1 right/wrong}
 *      x {cascade right/wrong}.
 *   D. Hash-rank distribution of cascade's correct picks.
 *   E. Per-partition (tied-min / elsewhere-top-10 / nowhere-top-10)
 *      cascade accuracy.
 *   F. Class-pair confusion delta (pure-hash vs cascade).
 *   G. Secondary-hash margin: correct-class routed distance vs nearest
 *      wrong-class routed distance within top-K, averaged.
 *
 * Usage: ./mnist_cascade_atomics <mnist_dir>
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
#define K_CEILING 200
#define K_FIXED 50

/* ── Shared loaders/deskew/RNG/τ (mirrored from prior tools). ────────── */

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

    int Sp=M4T_TRIT_PACKED_BYTES(N_PROJ);
    uint8_t *train_sigs_A, *test_sigs_A, *train_sigs_B, *test_sigs_B;
    {
        const uint32_t seeds[2][4] = {
            {42, 123, 456, 789},
            {1337, 2718, 3141, 5923}
        };
        uint8_t** train_sets[2] = {&train_sigs_A, &train_sigs_B};
        uint8_t** test_sets[2] = {&test_sigs_A, &test_sigs_B};

        for (int pass = 0; pass < 2; pass++) {
            rng_s[0]=seeds[pass][0]; rng_s[1]=seeds[pass][1];
            rng_s[2]=seeds[pass][2]; rng_s[3]=seeds[pass][3];
            {
                m4t_trit_t* proj_w=malloc((size_t)N_PROJ*INPUT_DIM);
                int proj_Dp=M4T_TRIT_PACKED_BYTES(INPUT_DIM);
                uint8_t* proj_packed=malloc((size_t)N_PROJ*proj_Dp);
                m4t_mtfp_t* train_proj=malloc((size_t)n_train*N_PROJ*sizeof(m4t_mtfp_t));
                m4t_mtfp_t* test_proj =malloc((size_t)n_test *N_PROJ*sizeof(m4t_mtfp_t));
                int64_t tau_q;
                int64_t* tmp;

                for(int i=0;i<N_PROJ*INPUT_DIM;i++){
                    uint32_t r=rng_next()%3;
                    proj_w[i]=(r==0)?-1:(r==1)?0:1;
                }
                m4t_pack_trits_rowmajor(proj_packed,proj_w,N_PROJ,INPUT_DIM);
                free(proj_w);

                for(int i=0;i<n_train;i++)
                    m4t_mtfp_ternary_matmul_bt(train_proj+(size_t)i*N_PROJ,
                                               x_train+(size_t)i*INPUT_DIM,
                                               proj_packed,1,INPUT_DIM,N_PROJ);
                for(int i=0;i<n_test;i++)
                    m4t_mtfp_ternary_matmul_bt(test_proj+(size_t)i*N_PROJ,
                                               x_test+(size_t)i*INPUT_DIM,
                                               proj_packed,1,INPUT_DIM,N_PROJ);

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

                *train_sets[pass]=calloc((size_t)n_train*Sp,1);
                *test_sets[pass] =calloc((size_t)n_test *Sp,1);
                tmp=malloc((size_t)N_PROJ*sizeof(int64_t));
                for(int i=0;i<n_train;i++){
                    for(int p=0;p<N_PROJ;p++) tmp[p]=(int64_t)train_proj[(size_t)i*N_PROJ+p];
                    m4t_route_threshold_extract((*train_sets[pass])+(size_t)i*Sp,tmp,tau_q,N_PROJ);
                }
                for(int i=0;i<n_test;i++){
                    for(int p=0;p<N_PROJ;p++) tmp[p]=(int64_t)test_proj[(size_t)i*N_PROJ+p];
                    m4t_route_threshold_extract((*test_sets[pass])+(size_t)i*Sp,tmp,tau_q,N_PROJ);
                }

                free(tmp);
                free(train_proj);
                free(test_proj);
                free(proj_packed);
            }
        }
    }

    uint8_t* mask=malloc(Sp); memset(mask,0xFF,Sp);

    printf("N_PROJ=%d routed-cascade atomics — deskewed MNIST, density=%.2f, seeds=(42,1337)\n\n",
           N_PROJ, DENSITY);

    /* ── Counters ────────────────────────────────────────────────────── */

    long ceiling_at_K[K_CEILING + 1] = {0};    /* # queries with correct in top-K */
    long conditional_resolver_at_K[K_CEILING + 1] = {0};   /* resolver picks correctly, given correct in top-K */

    /* Rescue/damage matrix at K=50. */
    int rr=0, rw=0, wr=0, ww=0;
    /* r = pure-hash top-1 matches truth; R = cascade matches truth */

    /* Hash-rank of cascade's correct pick (when it's correct). */
    long pick_hash_rank_hist[K_FIXED + 1] = {0};

    /* Per-partition cascade accuracy. Partition defined at K=10 (matches probe). */
    int tied_min_total=0, tied_min_cascade_correct=0;
    int elsewhere_total=0, elsewhere_cascade_correct=0;
    int nowhere_total=0, nowhere_cascade_correct=0;

    /* Class-pair confusion at K=50: conf_hash[t][p], conf_cascade[t][p]. */
    int conf_hash[N_CLASSES][N_CLASSES] = {0};
    int conf_cascade[N_CLASSES][N_CLASSES] = {0};

    /* Secondary-hash margin diagnostics. */
    double margin_sum = 0.0;
    int margin_n = 0;

    int32_t* dists = malloc((size_t)n_train*sizeof(int32_t));

    clock_t t0 = clock();
    for (int s = 0; s < n_test; s++) {
        const uint8_t* q_sig_A = test_sigs_A + (size_t)s*Sp;
        const uint8_t* q_sig_B = test_sigs_B + (size_t)s*Sp;
        int y = y_test[s];

        /* All 60K hash distances. */
        for (int i = 0; i < n_train; i++) {
            const uint8_t* r_sig_A = train_sigs_A + (size_t)i*Sp;
            dists[i] = m4t_popcount_dist(q_sig_A, r_sig_A, mask, Sp);
        }

        /* Top-K_CEILING by hash distance (partial insertion sort). */
        int32_t topd[K_CEILING]; int topi[K_CEILING];
        for(int j=0;j<K_CEILING;j++){ topd[j]=INT32_MAX; topi[j]=-1; }
        for(int i=0;i<n_train;i++){
            int32_t d=dists[i];
            if(d>=topd[K_CEILING-1]) continue;
            topd[K_CEILING-1]=d; topi[K_CEILING-1]=i;
            for(int j=K_CEILING-2;j>=0;j--){
                if(topd[j+1]<topd[j]){
                    int32_t td=topd[j]; topd[j]=topd[j+1]; topd[j+1]=td;
                    int ti=topi[j]; topi[j]=topi[j+1]; topi[j+1]=ti;
                } else break;
            }
        }

        /* Ceiling curve: cumulative presence of correct class by K. */
        int correct_first_rank = -1;
        for (int j = 0; j < K_CEILING; j++) {
            if (y_train[topi[j]] == y) {
                correct_first_rank = j;
                break;
            }
        }
        if (correct_first_rank >= 0) {
            for (int k = correct_first_rank; k < K_CEILING; k++)
                ceiling_at_K[k+1]++;
        }

        /* Precompute secondary-hash distances for top-K_CEILING. */
        int32_t resolver_d[K_CEILING];
        for (int j = 0; j < K_CEILING; j++) {
            const uint8_t* r_sig_B = train_sigs_B + (size_t)topi[j]*Sp;
            resolver_d[j] = m4t_popcount_dist(q_sig_B, r_sig_B, mask, Sp);
        }

        /* Conditional resolver: for each K, does secondary-hash 1-NN within top-K
         * pick the correct class? We tally only queries where correct is in top-K. */
        int32_t best_d = INT32_MAX; int best_j = -1;
        for (int k = 0; k < K_CEILING; k++) {
            if (resolver_d[k] < best_d) { best_d = resolver_d[k]; best_j = k; }
            /* best_j is the secondary-hash 1-NN within top-(k+1). */
            if (correct_first_rank >= 0 && correct_first_rank <= k) {
                if (y_train[topi[best_j]] == y)
                    conditional_resolver_at_K[k+1]++;
            }
        }

        /* Rescue/damage matrix at K=50. */
        int pure_top1_right = (y_train[topi[0]] == y);
        /* cascade @ K=50 = secondary-hash 1-NN within top-50. */
        int32_t c_best_d = INT32_MAX; int c_best_j = -1;
        for (int j = 0; j < K_FIXED; j++)
            if (resolver_d[j] < c_best_d) { c_best_d = resolver_d[j]; c_best_j = j; }
        int cascade_label = y_train[topi[c_best_j]];
        int cascade_right = (cascade_label == y);

        if (pure_top1_right && cascade_right)   rr++;
        else if (pure_top1_right && !cascade_right) rw++;
        else if (!pure_top1_right && cascade_right) wr++;
        else ww++;

        /* Hash-rank distribution of cascade's pick (for correct picks only). */
        if (cascade_right) pick_hash_rank_hist[c_best_j]++;

        /* Partition (via top-10 correct location + tied-min at hash-rank 0). */
        int min_hd = topd[0];
        int correct_in_tied_min = 0;
        int correct_in_top10 = 0;
        for (int j = 0; j < 10; j++) {
            if (y_train[topi[j]] == y) {
                correct_in_top10 = 1;
                if (topd[j] == min_hd) correct_in_tied_min = 1;
            }
        }
        if (correct_in_tied_min) {
            tied_min_total++;
            if (cascade_right) tied_min_cascade_correct++;
        } else if (correct_in_top10) {
            elsewhere_total++;
            if (cascade_right) elsewhere_cascade_correct++;
        } else {
            nowhere_total++;
            if (cascade_right) nowhere_cascade_correct++;
        }

        /* Confusion: pure-hash k=7 majority. */
        int counts7[N_CLASSES]={0};
        for(int j=0;j<7;j++) counts7[y_train[topi[j]]]++;
        int pure_pred=0;
        for(int c=1;c<N_CLASSES;c++) if(counts7[c]>counts7[pure_pred]) pure_pred=c;
        conf_hash[y][pure_pred]++;
        conf_cascade[y][cascade_label]++;

        /* Margin: within top-K_FIXED, secondary-hash distance of nearest correct
         * vs nearest wrong candidate. Positive means correct is routed-closer. */
        int32_t best_correct = INT32_MAX, best_wrong = INT32_MAX;
        for (int j = 0; j < K_FIXED; j++) {
            if (y_train[topi[j]] == y) {
                if (resolver_d[j] < best_correct) best_correct = resolver_d[j];
            } else {
                if (resolver_d[j] < best_wrong) best_wrong = resolver_d[j];
            }
        }
        if (best_correct < INT32_MAX && best_wrong < INT32_MAX) {
            double mc = (double)best_correct;
            double mw = (double)best_wrong;
            /* Positive = correct is routed-closer than nearest wrong candidate. */
            double rel = (mw - mc) / (mw + mc + 1.0);
            margin_sum += rel;
            margin_n++;
        }
    }
    double elapsed = (double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("Inference: %.1f s, %d test queries.\n\n", elapsed, n_test);

    /* ── A. Ceiling curve ─────────────────────────────────────────────── */

    printf("A. Ceiling curve: fraction of queries with correct class in top-K.\n");
    printf("     K    in_top_K     fraction    Δ from K-1\n");
    int check_ks[] = {1,2,3,5,7,10,15,20,30,50,75,100,150,200};
    long prev = 0;
    for (size_t z = 0; z < sizeof(check_ks)/sizeof(check_ks[0]); z++) {
        int K = check_ks[z];
        if (K > K_CEILING) continue;
        printf("   %4d    %-10ld   %6.2f%%      +%.2f%%\n",
               K, ceiling_at_K[K], 100.0*ceiling_at_K[K]/n_test,
               100.0*(ceiling_at_K[K]-prev)/n_test);
        prev = ceiling_at_K[K];
    }
    printf("\n");

    /* ── B. Conditional resolver accuracy ─────────────────────────────── */

    printf("B. Conditional resolver: P(secondary-hash 1-NN picks correct | correct in top-K).\n");
    printf("     K    conditional_correct   ceiling   conditional_rate\n");
    for (size_t z = 0; z < sizeof(check_ks)/sizeof(check_ks[0]); z++) {
        int K = check_ks[z];
        if (K > K_CEILING) continue;
        long c = conditional_resolver_at_K[K];
        long pool = ceiling_at_K[K];
        double rate = pool ? 100.0 * c / pool : 0.0;
        double global_rate = 100.0 * c / n_test;
        printf("   %4d    %-10ld      %-5ld     %6.2f%%   (global: %.2f%%)\n",
               K, c, pool, rate, global_rate);
    }
    printf("\n");

    /* ── C. Rescue/damage matrix at K=50 ──────────────────────────────── */

    printf("C. Rescue/damage matrix at K=%d (secondary-hash 1-NN cascade).\n", K_FIXED);
    printf("   Rows: pure-hash top-1.  Cols: cascade prediction.\n");
    printf("                  cascade_right    cascade_wrong\n");
    printf("   pure_right     %-14d   %-14d   (pure-top-1 correct: %d)\n",
           rr, rw, rr+rw);
    printf("   pure_wrong     %-14d   %-14d   (pure-top-1 wrong:   %d)\n",
           wr, ww, wr+ww);
    printf("\n");
    printf("   Rescued (pure wrong → cascade right): %d = %.2f%%\n",
           wr, 100.0*wr/n_test);
    printf("   Damaged (pure right → cascade wrong): %d = %.2f%%\n",
           rw, 100.0*rw/n_test);
    printf("   Net cascade advantage over pure-top-1: %+d = %+.2f%%\n\n",
           wr-rw, 100.0*(wr-rw)/n_test);

    /* ── D. Hash-rank of cascade's correct picks ──────────────────────── */

    printf("D. Hash-rank distribution of cascade's correct picks (K=%d).\n", K_FIXED);
    printf("   rank_bucket   correct_picks    fraction_of_correct\n");
    long total_correct = rr + wr;
    struct { int lo, hi; const char* name; } rbuckets[] = {
        {0,0,"rank 1 (top-1)"},
        {1,1,"rank 2"},
        {2,4,"ranks 3-5"},
        {5,9,"ranks 6-10"},
        {10,19,"ranks 11-20"},
        {20,49,"ranks 21-50"}
    };
    for (size_t b = 0; b < sizeof(rbuckets)/sizeof(rbuckets[0]); b++) {
        long sum = 0;
        for (int r = rbuckets[b].lo; r <= rbuckets[b].hi && r < K_FIXED; r++)
            sum += pick_hash_rank_hist[r];
        if (sum == 0) continue;
        printf("   %-14s  %-14ld   %.2f%%\n",
               rbuckets[b].name, sum, 100.0*sum/total_correct);
    }
    printf("\n");

    /* ── E. Per-partition cascade accuracy ────────────────────────────── */

    printf("E. Cascade accuracy by correct-class location partition (from top-10):\n");
    printf("                              count    cascade_correct   rate\n");
    printf("   correct in tied-min set:   %-5d    %-5d             %.2f%%\n",
           tied_min_total, tied_min_cascade_correct,
           tied_min_total ? 100.0*tied_min_cascade_correct/tied_min_total : 0);
    printf("   correct elsewhere top-10:  %-5d    %-5d             %.2f%%\n",
           elsewhere_total, elsewhere_cascade_correct,
           elsewhere_total ? 100.0*elsewhere_cascade_correct/elsewhere_total : 0);
    printf("   correct nowhere top-10:    %-5d    %-5d             %.2f%%\n",
           nowhere_total, nowhere_cascade_correct,
           nowhere_total ? 100.0*nowhere_cascade_correct/nowhere_total : 0);
    printf("\n");

    /* ── F. Class-pair confusion delta ────────────────────────────────── */

    printf("F. Class-pair errors: pure-hash k=7 majority vs routed cascade K=%d.\n", K_FIXED);
    printf("   Top off-diagonal deltas (positive = cascade improves).\n");
    printf("   true  pred   hash_err   cascade_err   Δ\n");
    /* Collect all pairs where t != p and sort by improvement. */
    typedef struct { int t,p,h,c; } pair_t;
    pair_t pairs[N_CLASSES*N_CLASSES];
    int np = 0;
    for (int t=0;t<N_CLASSES;t++)
        for (int p=0;p<N_CLASSES;p++) {
            if (t==p) continue;
            int h = conf_hash[t][p];
            int c = conf_cascade[t][p];
            if (h == 0 && c == 0) continue;
            pairs[np].t=t; pairs[np].p=p; pairs[np].h=h; pairs[np].c=c;
            np++;
        }
    /* Sort descending by (h - c). */
    for (int i=1;i<np;i++) {
        int d = pairs[i].h - pairs[i].c;
        int j = i;
        while (j>0 && (pairs[j-1].h - pairs[j-1].c) < d) {
            pair_t swap_pair = pairs[j-1];
            pairs[j-1] = pairs[j];
            pairs[j] = swap_pair;
            j--;
        }
    }
    int show = np < 12 ? np : 12;
    for (int i = 0; i < show; i++)
        printf("   %-5d %-5d  %-9d  %-12d  %+d\n",
               pairs[i].t, pairs[i].p, pairs[i].h, pairs[i].c,
               pairs[i].h - pairs[i].c);
    printf("\n");

    /* Also show worst regressions (cascade worse than hash). */
    printf("   Worst regressions (negative Δ = cascade worse):\n");
    printf("   true  pred   hash_err   cascade_err   Δ\n");
    int shown = 0;
    for (int i = np - 1; i >= 0 && shown < 6; i--) {
        int d = pairs[i].h - pairs[i].c;
        if (d >= 0) break;
        printf("   %-5d %-5d  %-9d  %-12d  %+d\n",
               pairs[i].t, pairs[i].p, pairs[i].h, pairs[i].c, d);
        shown++;
    }
    if (shown == 0) printf("   (no regressions — cascade does not introduce any new confusion pairs)\n");
    printf("\n");

    /* ── G. Routed margin ─────────────────────────────────────────────── */

    double avg_margin = margin_n ? margin_sum / margin_n : 0.0;
    printf("G. Secondary-hash margin in top-%d (correct vs nearest-wrong).\n", K_FIXED);
    printf("   Average relative margin (wrong - correct) / (wrong + correct + 1) = %+.4f\n",
           avg_margin);
    printf("   Sample size: %d queries with both correct and wrong class present in top-%d.\n",
           margin_n, K_FIXED);
    printf("   (Positive = correct class is on average routed-closer than nearest wrong.)\n");
    printf("\n");

    printf("Summary of mechanism:\n");
    printf("  A shows how much widening K buys at the filter stage.\n");
    printf("  B shows how reliably the routed secondary resolver picks correct when available.\n");
    printf("  C quantifies rescue vs damage over pure-hash-top-1.\n");
    printf("  D shows where in the hash ranking cascade's correct picks live.\n");
    printf("  E isolates cascade's gains by partition.\n");
    printf("  F shows which digit confusions cascade fixes (and any it creates).\n");
    printf("  G shows why the routed secondary hash discriminates at all — the margin.\n");

    free(dists);
    free(mask);
    free(train_sigs_A); free(test_sigs_A);
    free(train_sigs_B); free(test_sigs_B);
    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
