/*
 * mnist_routed_bucket.c — first genuinely routed Glyph consumer.
 *
 * Every prior cascade tool ran routing primitives inside an O(N_train)
 * dense outer scan: for every query, compute popcount_dist to all 60K
 * training signatures. That is dense application with routed kernels -
 * precisely the shape NORTH_STAR forbids.
 *
 * This consumer uses the signature as an index key. Training-time cost
 * is O(N_train log N_train) once to sort the bucket table. Query time
 * is O(1) amortized: look up the query signature in the sorted table
 * and read off all prototypes that share its code (exact Hamming-0
 * collision class). For queries whose exact bucket is empty or small,
 * multi-probe expands outward to neighbor trit codes within a Hamming
 * radius budget.
 *
 * Architecture:
 *
 *   TRAINING
 *     for each prototype i in 0..N_train:
 *         compute packed-trit sig[i] via H1
 *         append (sig[i], i) to entries[]
 *     qsort entries[] by sig key
 *     // entries[] is now a bucket index: prototypes with equal keys
 *     // form contiguous runs
 *
 *   QUERY
 *     compute query sig
 *     candidate_set = {}
 *     for radius r = 0 .. MAX_RADIUS (stop early if candidate_set full):
 *         for each probe_sig in neighbors(query_sig, radius r):
 *             binary-search entries[] for probe_sig
 *             append matching prototype indices to candidate_set
 *     resolver: route the candidate set via H2+H3+H4 popcount_dist sum
 *     return label of argmin
 *
 * For N_PROJ=16, sig = 4 bytes. Exact-match collisions (r=0) already
 * dominate: the Axis 4c atomic probe showed 52% of test queries have
 * a Hamming-0 match to at least one training prototype, 97% are at
 * Hamming <= 2. Multi-probe radius 2 should cover nearly all queries
 * without any dense scan.
 *
 * Usage: ./mnist_routed_bucket <mnist_dir>
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
#define SIG_BYTES 4            /* M4T_TRIT_PACKED_BYTES(16) = 4 */
#define MAX_RADIUS 2
#define MIN_CANDIDATES 5       /* expand radius until candidate set has this many */
#define MAX_CANDIDATES 1024    /* bucket-side cap; prevents pathological fan-out */

/* ── Shared loaders/deskew/RNG/tau (mirrored from prior tools) ───────── */

static uint32_t read_u32_be(FILE* f) {
    uint8_t b[4]; fread(b,1,4,f);
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];
}
static m4t_mtfp_t* load_images_mtfp(const char* path, int* n) {
    FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",path);exit(1);}
    read_u32_be(f); *n=(int)read_u32_be(f);
    {
        int rows=(int)read_u32_be(f),cols=(int)read_u32_be(f);
        int dim=rows*cols; size_t total=(size_t)(*n)*dim;
        uint8_t* raw=malloc(total); fread(raw,1,total,f); fclose(f);
        m4t_mtfp_t* data=malloc(total*sizeof(m4t_mtfp_t));
        for(size_t i=0;i<total;i++)
            data[i]=(m4t_mtfp_t)(((int32_t)raw[i]*M4T_MTFP_SCALE+127)/255);
        free(raw);
        return data;
    }
}
static int* load_labels(const char* path, int* n) {
    FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",path);exit(1);}
    read_u32_be(f); *n=(int)read_u32_be(f);
    {
        uint8_t* raw=malloc(*n); fread(raw,1,*n,f); fclose(f);
        int* l=malloc(*n*sizeof(int));
        for(int i=0;i<*n;i++) l[i]=(int)raw[i];
        free(raw);
        return l;
    }
}
static void deskew_image(m4t_mtfp_t* dst, const m4t_mtfp_t* src) {
    int64_t sum_p=0,sum_xp=0,sum_yp=0;
    for(int y=0;y<IMG_H;y++)
        for(int x=0;x<IMG_W;x++){
            int64_t p=(int64_t)src[y*IMG_W+x];
            sum_p+=p; sum_xp+=(int64_t)x*p; sum_yp+=(int64_t)y*p;
        }
    if(sum_p==0){memcpy(dst,src,INPUT_DIM*sizeof(m4t_mtfp_t));return;}
    {
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
    {
        size_t idx = (size_t)(d * (double)n);
        if (idx >= n) idx = n-1;
        return v[idx];
    }
}
static void build_signature_set(
    int N_proj,
    const m4t_mtfp_t* x_train, int n_train,
    const m4t_mtfp_t* x_test, int n_test,
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
    double density,
    uint8_t** out_train_sigs, uint8_t** out_test_sigs)
{
    int Sp=M4T_TRIT_PACKED_BYTES(N_proj);
    int proj_Dp=M4T_TRIT_PACKED_BYTES(INPUT_DIM);
    rng_s[0]=s0; rng_s[1]=s1; rng_s[2]=s2; rng_s[3]=s3;
    {
        m4t_trit_t* proj_w=malloc((size_t)N_proj*INPUT_DIM);
        uint8_t* proj_packed=malloc((size_t)N_proj*proj_Dp);
        m4t_mtfp_t* train_proj=malloc((size_t)n_train*N_proj*sizeof(m4t_mtfp_t));
        m4t_mtfp_t* test_proj =malloc((size_t)n_test *N_proj*sizeof(m4t_mtfp_t));
        int64_t tau_q;
        int64_t* tmp;
        for(int i=0;i<N_proj*INPUT_DIM;i++){
            uint32_t r=rng_next()%3;
            proj_w[i]=(r==0)?-1:(r==1)?0:1;
        }
        m4t_pack_trits_rowmajor(proj_packed,proj_w,N_proj,INPUT_DIM);
        free(proj_w);
        for(int i=0;i<n_train;i++)
            m4t_mtfp_ternary_matmul_bt(train_proj+(size_t)i*N_proj,
                                       x_train+(size_t)i*INPUT_DIM,
                                       proj_packed,1,INPUT_DIM,N_proj);
        for(int i=0;i<n_test;i++)
            m4t_mtfp_ternary_matmul_bt(test_proj+(size_t)i*N_proj,
                                       x_test+(size_t)i*INPUT_DIM,
                                       proj_packed,1,INPUT_DIM,N_proj);
        {
            size_t total=(size_t)1000*N_proj;
            int64_t* buf=malloc(total*sizeof(int64_t));
            for(int i=0;i<1000;i++)
                for(int p=0;p<N_proj;p++){
                    int64_t v=train_proj[(size_t)i*N_proj+p];
                    buf[(size_t)i*N_proj+p]=(v>=0)?v:-v;
                }
            tau_q=tau_for_density(buf,total,density);
            free(buf);
        }
        *out_train_sigs=calloc((size_t)n_train*Sp,1);
        *out_test_sigs =calloc((size_t)n_test *Sp,1);
        tmp=malloc((size_t)N_proj*sizeof(int64_t));
        for(int i=0;i<n_train;i++){
            for(int p=0;p<N_proj;p++) tmp[p]=(int64_t)train_proj[(size_t)i*N_proj+p];
            m4t_route_threshold_extract((*out_train_sigs)+(size_t)i*Sp,tmp,tau_q,N_proj);
        }
        for(int i=0;i<n_test;i++){
            for(int p=0;p<N_proj;p++) tmp[p]=(int64_t)test_proj[(size_t)i*N_proj+p];
            m4t_route_threshold_extract((*out_test_sigs)+(size_t)i*Sp,tmp,tau_q,N_proj);
        }
        free(tmp); free(train_proj); free(test_proj); free(proj_packed);
    }
}

/* ── Bucket index ────────────────────────────────────────────────────── */

typedef struct { uint32_t key; int proto_idx; } entry_t;

static int cmp_entry(const void* a, const void* b) {
    uint32_t x = ((const entry_t*)a)->key;
    uint32_t y = ((const entry_t*)b)->key;
    return (x<y) ? -1 : (x>y) ? 1 : 0;
}

static uint32_t sig_to_key(const uint8_t* sig) {
    /* SIG_BYTES = 4 at N_PROJ=16. Treat as little-endian uint32. */
    return (uint32_t)sig[0]
         | ((uint32_t)sig[1] << 8)
         | ((uint32_t)sig[2] << 16)
         | ((uint32_t)sig[3] << 24);
}

/* Binary search for the first entry with key >= target. Returns
 * lower-bound index in [0, n]. */
static int lower_bound(const entry_t* entries, int n, uint32_t target) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (entries[mid].key < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

/* ── Trit manipulation on packed sig bytes ───────────────────────────── */

/* Read trit j from a packed sig. Trits are 2-bit codes:
 *   0b00 = 0, 0b01 = +1, 0b10 = -1. */
static int8_t read_trit(const uint8_t* sig, int j) {
    uint8_t code = (sig[j >> 2] >> ((j & 3) * 2)) & 0x3u;
    return (code == 0x01u) ? 1 : (code == 0x02u) ? -1 : 0;
}

/* Write trit j into a packed sig (in place). */
static void write_trit(uint8_t* sig, int j, int8_t t) {
    uint8_t code = (t == 1) ? 0x01u : (t == -1) ? 0x02u : 0x00u;
    int shift = (j & 3) * 2;
    sig[j >> 2] = (uint8_t)((sig[j >> 2] & ~(0x3u << shift)) | (code << shift));
}

/* Enumerate neighbors of `query_sig` at EXACTLY ternary Hamming cost
 * equal to `radius`, for radius in {0, 1, 2}. Calls `callback` with
 * each probe signature; `callback` returns non-zero to stop early. */
typedef int (*probe_cb)(const uint8_t* probe, void* ctx);

static int enumerate_radius(const uint8_t* query_sig, int radius,
                            uint8_t* scratch, probe_cb cb, void* ctx)
{
    /* radius 0: just the query itself. */
    if (radius == 0) {
        memcpy(scratch, query_sig, SIG_BYTES);
        return cb(scratch, ctx);
    }
    if (radius == 1) {
        /* Cost-1 moves: at exactly one position, flip between 0 and ±1
         * in the direction that costs 1 per ternary Hamming rules. A
         * position holding 0 can move to +1 or -1 (cost 1 each); a
         * position holding ±1 can move to 0 (cost 1) — not to ∓1 (cost 2). */
        for (int j = 0; j < N_PROJ; j++) {
            int8_t orig = read_trit(query_sig, j);
            memcpy(scratch, query_sig, SIG_BYTES);
            if (orig == 0) {
                write_trit(scratch, j, +1);
                if (cb(scratch, ctx)) return 1;
                write_trit(scratch, j, -1);
                if (cb(scratch, ctx)) return 1;
            } else {
                write_trit(scratch, j, 0);
                if (cb(scratch, ctx)) return 1;
            }
            /* restore is implicit via next memcpy */
        }
        return 0;
    }
    if (radius == 2) {
        /* Cost-2 moves: either (a) one position sign-flipped (+1↔-1,
         * cost 2), or (b) two positions each with a cost-1 move. (b) has
         * many combinations; for retrieval budget we emit all of them. */
        /* (a) sign flips */
        for (int j = 0; j < N_PROJ; j++) {
            int8_t orig = read_trit(query_sig, j);
            if (orig == 0) continue;
            memcpy(scratch, query_sig, SIG_BYTES);
            write_trit(scratch, j, (int8_t)(-orig));
            if (cb(scratch, ctx)) return 1;
        }
        /* (b) two cost-1 moves on distinct positions */
        for (int j = 0; j < N_PROJ; j++) {
            int8_t oj = read_trit(query_sig, j);
            for (int k = j + 1; k < N_PROJ; k++) {
                int8_t ok = read_trit(query_sig, k);
                /* Enumerate per-position cost-1 targets at both j and k. */
                int8_t targets_j[2]; int nj = 0;
                int8_t targets_k[2]; int nk = 0;
                if (oj == 0) { targets_j[nj++] = +1; targets_j[nj++] = -1; }
                else         { targets_j[nj++] = 0; }
                if (ok == 0) { targets_k[nk++] = +1; targets_k[nk++] = -1; }
                else         { targets_k[nk++] = 0; }
                for (int a = 0; a < nj; a++) {
                    for (int b = 0; b < nk; b++) {
                        memcpy(scratch, query_sig, SIG_BYTES);
                        write_trit(scratch, j, targets_j[a]);
                        write_trit(scratch, k, targets_k[b]);
                        if (cb(scratch, ctx)) return 1;
                    }
                }
            }
        }
        return 0;
    }
    return 0;
}

/* ── Query-time state for a bucket lookup ────────────────────────────── */

typedef struct {
    const entry_t* entries;
    int n_entries;
    int candidates[MAX_CANDIDATES];
    int n_candidates;
    int probes_issued;
    int bucket_hits;
} lookup_state_t;

static int collect_bucket(const uint8_t* probe, void* ctx) {
    lookup_state_t* st = (lookup_state_t*)ctx;
    uint32_t key = sig_to_key(probe);
    st->probes_issued++;
    int lb = lower_bound(st->entries, st->n_entries, key);
    if (lb >= st->n_entries) return 0;
    if (st->entries[lb].key != key) return 0;
    /* Scan run of entries with matching key. */
    int found = 0;
    for (int i = lb; i < st->n_entries && st->entries[i].key == key; i++) {
        if (st->n_candidates >= MAX_CANDIDATES) return 1;  /* stop probing */
        st->candidates[st->n_candidates++] = st->entries[i].proto_idx;
        found = 1;
    }
    if (found) st->bucket_hits++;
    return 0;
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

    int Sp = M4T_TRIT_PACKED_BYTES(N_PROJ);
    if (Sp != SIG_BYTES) {
        fprintf(stderr, "Sig-bytes mismatch: got %d expected %d\n", Sp, SIG_BYTES);
        return 1;
    }

    /* Build four signature sets (H1 = primary bucket key, H2/H3/H4 = resolver). */
    uint8_t *trA,*teA,*trB,*teB,*trC,*teC,*trD,*teD;
    build_signature_set(N_PROJ, x_train, n_train, x_test, n_test,
                        42, 123, 456, 789, DENSITY, &trA, &teA);
    build_signature_set(N_PROJ, x_train, n_train, x_test, n_test,
                        1337, 2718, 3141, 5923, DENSITY, &trB, &teB);
    build_signature_set(N_PROJ, x_train, n_train, x_test, n_test,
                        1009, 2017, 3041, 5059, DENSITY, &trC, &teC);
    build_signature_set(N_PROJ, x_train, n_train, x_test, n_test,
                        9001, 9002, 9003, 9004, DENSITY, &trD, &teD);

    uint8_t* mask = malloc(Sp); memset(mask, 0xFF, Sp);

    /* ── Build bucket index on H1 sigs ───────────────────────────────── */

    clock_t t_build_start = clock();
    entry_t* entries = malloc((size_t)n_train * sizeof(entry_t));
    for (int i = 0; i < n_train; i++) {
        entries[i].key = sig_to_key(trA + (size_t)i * Sp);
        entries[i].proto_idx = i;
    }
    qsort(entries, n_train, sizeof(entry_t), cmp_entry);
    double t_build = (double)(clock() - t_build_start) / CLOCKS_PER_SEC;

    /* Count distinct buckets and bucket-size histogram. */
    int n_buckets = 0;
    int bucket_hist[8] = {0}; /* sizes 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64-127, 128+ */
    {
        int i = 0;
        while (i < n_train) {
            int j = i + 1;
            while (j < n_train && entries[j].key == entries[i].key) j++;
            int size = j - i;
            n_buckets++;
            int b = 0;
            if      (size == 1)    b = 0;
            else if (size <= 3)    b = 1;
            else if (size <= 7)    b = 2;
            else if (size <= 15)   b = 3;
            else if (size <= 31)   b = 4;
            else if (size <= 63)   b = 5;
            else if (size <= 127)  b = 6;
            else                   b = 7;
            bucket_hist[b]++;
            i = j;
        }
    }

    printf("mnist_routed_bucket — genuinely routed LSH consumer\n");
    printf("N_PROJ=%d, density=%.2f, SIG_BYTES=%d, seeds H1/H2/H3/H4 fixed.\n\n",
           N_PROJ, DENSITY, Sp);

    printf("Index build:\n");
    printf("  %d training prototypes -> %d distinct buckets (%.2fx compression)\n",
           n_train, n_buckets, (double)n_train / n_buckets);
    printf("  build time: %.3fs\n", t_build);
    printf("  bucket size histogram:\n");
    const char* size_labels[] = {"1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128+"};
    for (int b = 0; b < 8; b++)
        if (bucket_hist[b] > 0)
            printf("    %-8s  %6d buckets\n", size_labels[b], bucket_hist[b]);
    printf("\n");

    /* ── Query loop ──────────────────────────────────────────────────── */

    int correct = 0;
    int total_probes = 0;
    int total_candidates = 0;
    int total_bucket_hits = 0;
    int queries_needing_r0 = 0;
    int queries_needing_r1 = 0;
    int queries_needing_r2 = 0;
    int queries_empty = 0;

    uint8_t scratch[SIG_BYTES];

    clock_t t_query_start = clock();
    for (int s = 0; s < n_test; s++) {
        const uint8_t* qA = teA + (size_t)s * Sp;
        const uint8_t* qB = teB + (size_t)s * Sp;
        const uint8_t* qC = teC + (size_t)s * Sp;
        const uint8_t* qD = teD + (size_t)s * Sp;
        int y = y_test[s];

        lookup_state_t st = {0};
        st.entries = entries;
        st.n_entries = n_train;

        /* Radius 0 first. */
        int hit_radius = -1;
        enumerate_radius(qA, 0, scratch, collect_bucket, &st);
        if (st.n_candidates > 0) { hit_radius = 0; queries_needing_r0++; }
        if (st.n_candidates < MIN_CANDIDATES) {
            enumerate_radius(qA, 1, scratch, collect_bucket, &st);
            if (hit_radius < 0 && st.n_candidates > 0) { hit_radius = 1; }
            if (hit_radius == 1) queries_needing_r1++;
        }
        if (st.n_candidates < MIN_CANDIDATES) {
            enumerate_radius(qA, 2, scratch, collect_bucket, &st);
            if (hit_radius < 0 && st.n_candidates > 0) { hit_radius = 2; }
            if (hit_radius == 2) queries_needing_r2++;
        }
        if (st.n_candidates == 0) { queries_empty++; continue; }

        total_probes += st.probes_issued;
        total_candidates += st.n_candidates;
        total_bucket_hits += st.bucket_hits;

        /* Resolver: H2+H3+H4 summed Hamming 1-NN over the candidate set. */
        int best_label = -1;
        int32_t best_score = INT32_MAX;
        for (int c = 0; c < st.n_candidates; c++) {
            int idx = st.candidates[c];
            int32_t score =
                m4t_popcount_dist(qB, trB + (size_t)idx * Sp, mask, Sp) +
                m4t_popcount_dist(qC, trC + (size_t)idx * Sp, mask, Sp) +
                m4t_popcount_dist(qD, trD + (size_t)idx * Sp, mask, Sp);
            if (score < best_score) { best_score = score; best_label = y_train[idx]; }
        }
        if (best_label == y) correct++;
    }
    double t_query = (double)(clock() - t_query_start) / CLOCKS_PER_SEC;

    /* ── Report ──────────────────────────────────────────────────────── */

    printf("Query results (%d test queries):\n", n_test);
    printf("  accuracy:            %.2f%%   (%d / %d)\n",
           100.0 * correct / n_test, correct, n_test);
    printf("  avg candidates/qry:  %.2f\n",
           (double)total_candidates / (n_test - queries_empty));
    printf("  avg probes/qry:      %.2f\n",
           (double)total_probes / (n_test - queries_empty));
    printf("  avg bucket-hits/qry: %.2f\n",
           (double)total_bucket_hits / (n_test - queries_empty));
    printf("  queries empty:       %d\n", queries_empty);
    printf("  query time:          %.3fs   (%.1f us / query)\n",
           t_query, 1e6 * t_query / n_test);
    printf("\n");

    printf("Radius escalation profile:\n");
    printf("  stopped at r=0:      %d  (%.2f%%)\n",
           queries_needing_r0, 100.0*queries_needing_r0/n_test);
    printf("  escalated to r=1:    %d  (%.2f%%)\n",
           queries_needing_r1, 100.0*queries_needing_r1/n_test);
    printf("  escalated to r=2:    %d  (%.2f%%)\n",
           queries_needing_r2, 100.0*queries_needing_r2/n_test);
    printf("\n");

    /* Cost comparison: dense cascade would do 60K popcount_dists for H1
     * alone, plus K*3 for the resolver. We do ~avg_probes * log(n_train)
     * for binary searches (ignoring since lb is cheap) plus avg_candidates*3
     * popcount_dists for the resolver. The H1 pass is completely eliminated. */
    printf("Cost comparison vs dense L50_H1 baseline:\n");
    printf("  dense baseline:      %d H1 popcount_dists + %d resolver ops per query\n",
           n_train, 50 * 3);
    printf("  bucket consumer:     0 H1 popcount_dists + %.0f resolver ops per query\n",
           3.0 * total_candidates / (n_test - queries_empty));
    printf("  ratio of popcount_dist calls: %.3fx\n",
           (3.0 * total_candidates / (n_test - queries_empty))
           / (double)(n_train + 150));
    printf("\n");

    printf("Interpretation:\n");
    printf("  The H1 pass is gone. Query cost is dominated by a handful of\n");
    printf("  binary searches into the sorted bucket table plus a small\n");
    printf("  resolver pass over the candidate set. Accuracy should track\n");
    printf("  the dense L50_H1 baseline (83.86%%) because the candidate set\n");
    printf("  contains all prototypes at minimum Hamming distance plus any\n");
    printf("  added by multi-probe expansion. Gaps from the dense baseline\n");
    printf("  indicate MIN_CANDIDATES / MAX_RADIUS tuning room.\n");

    free(entries);
    free(mask);
    free(trA); free(teA); free(trB); free(teB);
    free(trC); free(teC); free(trD); free(teD);
    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
