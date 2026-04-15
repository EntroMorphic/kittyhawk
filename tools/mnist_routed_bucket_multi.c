/*
 * mnist_routed_bucket_multi.c — multi-table routed bucket LSH.
 *
 * Extension of tools/mnist_routed_bucket.c to M independent bucket
 * indexes. Each table has its own RNG seed, its own random ternary
 * projection, and its own sorted (sig_key, proto_idx) entries.
 * At query time, every table is probed independently with ternary
 * multi-probe; candidate sets are union-merged across tables; a
 * resolver scores the union.
 *
 * LMM synthesize for this tool: journal/break_97_nproj16_synthesize.md
 *
 * Architecture:
 *
 *   TRAINING (one-time, ~13 s for M=64 at N_train=60000)
 *     for m in 0..M_MAX:
 *         seed_m = derive_seed(m)
 *         W_m = random_ternary(seed_m, N_PROJ=16)
 *         tau_m = density_percentile(|W_m @ x|, 0.33)
 *         for i in 0..N_train:
 *             sig_m[i] = threshold_extract(W_m @ x_train[i], tau_m)
 *             entries[m][i] = (sig_to_key(sig_m[i]), i)
 *         qsort(entries[m], by sig_key)
 *
 *   QUERY (per test sample, O(1) amortized in N_train)
 *     clear_union()
 *     for m in 0..M_active:
 *         q_sig_m = threshold_extract(W_m @ query, tau_m)
 *         for r = 0..2:
 *             if |per_table_candidates_m| >= MIN_CANDS: break
 *             for each neighbor_key in ternary_neighbors(q_sig_m, r):
 *                 lookup_bucket(entries[m], neighbor_key)
 *                 add candidates to union (votes[] / hit_list[])
 *     resolver(union) -> predicted label
 *
 * Three resolvers (measured at each M):
 *   VOTE : argmax class by sum of vote counts per class
 *   SUM  : argmin candidate by sum of popcount_dist across all M hashes
 *   PTM  : per-table-majority. For each table, pick its 1-NN candidate
 *          from the union, vote across the M labels.
 *
 * Modes:
 *   ORACLE (default): run only the oracle ceiling pass across M sweep.
 *   FULL: run oracle + all three resolvers at each M value.
 *
 * Usage:
 *   ./mnist_routed_bucket_multi <mnist_dir> [--full]
 *
 * Red-teaming expected between Phase 1 (build this tool) and Phase 2
 * (run oracle pass). See journal/break_97_nproj16_synthesize.md.
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
#define SIG_BYTES 4
#define M_MAX 64
#define N_TRAIN_MAX 60000
#define MAX_UNION 16384
#define MAX_RADIUS 2
#define MIN_CANDS_PER_TABLE 50

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
    uint8_t* out_train_sigs, uint8_t* out_test_sigs)
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
        tmp=malloc((size_t)N_proj*sizeof(int64_t));
        for(int i=0;i<n_train;i++){
            for(int p=0;p<N_proj;p++) tmp[p]=(int64_t)train_proj[(size_t)i*N_proj+p];
            m4t_route_threshold_extract(out_train_sigs+(size_t)i*Sp,tmp,tau_q,N_proj);
        }
        for(int i=0;i<n_test;i++){
            for(int p=0;p<N_proj;p++) tmp[p]=(int64_t)test_proj[(size_t)i*N_proj+p];
            m4t_route_threshold_extract(out_test_sigs+(size_t)i*Sp,tmp,tau_q,N_proj);
        }
        free(tmp); free(train_proj); free(test_proj); free(proj_packed);
    }
}

/* ── Bucket entries and ternary multi-probe (mirrored from mnist_routed_bucket.c) */

typedef struct { uint32_t key; int proto_idx; } entry_t;

static int cmp_entry(const void* a, const void* b) {
    uint32_t x = ((const entry_t*)a)->key;
    uint32_t y = ((const entry_t*)b)->key;
    return (x<y) ? -1 : (x>y) ? 1 : 0;
}
static uint32_t sig_to_key(const uint8_t* sig) {
    return (uint32_t)sig[0]
         | ((uint32_t)sig[1] << 8)
         | ((uint32_t)sig[2] << 16)
         | ((uint32_t)sig[3] << 24);
}
static int lower_bound(const entry_t* entries, int n, uint32_t target) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (entries[mid].key < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}
static int8_t read_trit(const uint8_t* sig, int j) {
    uint8_t code = (sig[j >> 2] >> ((j & 3) * 2)) & 0x3u;
    return (code == 0x01u) ? 1 : (code == 0x02u) ? -1 : 0;
}
static void write_trit(uint8_t* sig, int j, int8_t t) {
    uint8_t code = (t == 1) ? 0x01u : (t == -1) ? 0x02u : 0x00u;
    int shift = (j & 3) * 2;
    sig[j >> 2] = (uint8_t)((sig[j >> 2] & ~(0x3u << shift)) | (code << shift));
}

/* ── Global union state (single query at a time) ─────────────────────── */

static uint16_t g_votes[N_TRAIN_MAX];
static int32_t  g_hit_list[MAX_UNION];
static int      g_n_hit;
static int      g_n_probes;
/* Per-table candidate counts for MIN_CANDS_PER_TABLE tracking. */
static int      g_this_table_cands;

static void union_reset(void) {
    for (int j = 0; j < g_n_hit; j++) g_votes[g_hit_list[j]] = 0;
    g_n_hit = 0;
    g_n_probes = 0;
}

static void union_add_candidate(int proto_idx) {
    if (g_votes[proto_idx] == 0) {
        if (g_n_hit >= MAX_UNION) return;  /* cap — rare */
        g_hit_list[g_n_hit++] = proto_idx;
    }
    g_votes[proto_idx]++;
    g_this_table_cands++;
}

/* Probe entries for one key. Returns 1 if at least one hit. */
static int probe_key(const entry_t* entries, int n_entries, uint32_t key) {
    g_n_probes++;
    int lb = lower_bound(entries, n_entries, key);
    if (lb >= n_entries || entries[lb].key != key) return 0;
    int hit = 0;
    for (int i = lb; i < n_entries && entries[i].key == key; i++) {
        union_add_candidate(entries[i].proto_idx);
        hit = 1;
        if (g_n_hit >= MAX_UNION) return hit;
    }
    return hit;
}

/* Enumerate ternary radius r around query_sig and probe each neighbor
 * in the given table's entries. Returns the number of buckets hit. */
static void probe_radius(const entry_t* entries, int n_entries,
                         const uint8_t* query_sig, int radius)
{
    uint8_t scratch[SIG_BYTES];
    if (radius == 0) {
        memcpy(scratch, query_sig, SIG_BYTES);
        probe_key(entries, n_entries, sig_to_key(scratch));
        return;
    }
    if (radius == 1) {
        for (int j = 0; j < N_PROJ; j++) {
            int8_t orig = read_trit(query_sig, j);
            memcpy(scratch, query_sig, SIG_BYTES);
            if (orig == 0) {
                write_trit(scratch, j, +1);
                probe_key(entries, n_entries, sig_to_key(scratch));
                if (g_n_hit >= MAX_UNION) return;
                memcpy(scratch, query_sig, SIG_BYTES);
                write_trit(scratch, j, -1);
                probe_key(entries, n_entries, sig_to_key(scratch));
            } else {
                write_trit(scratch, j, 0);
                probe_key(entries, n_entries, sig_to_key(scratch));
            }
            if (g_n_hit >= MAX_UNION) return;
        }
        return;
    }
    if (radius == 2) {
        /* (a) sign flips at non-zero positions */
        for (int j = 0; j < N_PROJ; j++) {
            int8_t orig = read_trit(query_sig, j);
            if (orig == 0) continue;
            memcpy(scratch, query_sig, SIG_BYTES);
            write_trit(scratch, j, (int8_t)(-orig));
            probe_key(entries, n_entries, sig_to_key(scratch));
            if (g_n_hit >= MAX_UNION) return;
        }
        /* (b) two cost-1 moves on distinct positions */
        for (int j = 0; j < N_PROJ; j++) {
            int8_t oj = read_trit(query_sig, j);
            for (int k = j + 1; k < N_PROJ; k++) {
                int8_t ok = read_trit(query_sig, k);
                int8_t tj[2]; int nj = 0;
                int8_t tk[2]; int nk = 0;
                if (oj == 0) { tj[nj++] = +1; tj[nj++] = -1; } else { tj[nj++] = 0; }
                if (ok == 0) { tk[nk++] = +1; tk[nk++] = -1; } else { tk[nk++] = 0; }
                for (int a = 0; a < nj; a++) {
                    for (int b = 0; b < nk; b++) {
                        memcpy(scratch, query_sig, SIG_BYTES);
                        write_trit(scratch, j, tj[a]);
                        write_trit(scratch, k, tk[b]);
                        probe_key(entries, n_entries, sig_to_key(scratch));
                        if (g_n_hit >= MAX_UNION) return;
                    }
                }
            }
        }
    }
}

/* Probe one table into the current union. Early-stop at per-table
 * candidate count threshold. */
static void probe_table_into_union(const entry_t* entries, int n_entries,
                                   const uint8_t* q_sig)
{
    g_this_table_cands = 0;
    probe_radius(entries, n_entries, q_sig, 0);
    if (g_this_table_cands < MIN_CANDS_PER_TABLE && g_n_hit < MAX_UNION)
        probe_radius(entries, n_entries, q_sig, 1);
    if (g_this_table_cands < MIN_CANDS_PER_TABLE && g_n_hit < MAX_UNION)
        probe_radius(entries, n_entries, q_sig, 2);
}

/* ── Resolvers ───────────────────────────────────────────────────────── */

/* VOTE: argmax class by vote total. */
static int resolver_vote(const int* y_train) {
    int class_votes[N_CLASSES] = {0};
    for (int j = 0; j < g_n_hit; j++) {
        int idx = g_hit_list[j];
        class_votes[y_train[idx]] += g_votes[idx];
    }
    int pred = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (class_votes[c] > class_votes[pred]) pred = c;
    return pred;
}

/* SUM: argmin candidate by sum_m popcount_dist across active tables. */
static int resolver_sum(int M_active, int Sp,
                        uint8_t* const* train_sigs,
                        uint8_t* const* query_sigs_per_table,
                        const uint8_t* mask, const int* y_train)
{
    int best_score = INT32_MAX;
    int best_label = -1;
    for (int j = 0; j < g_n_hit; j++) {
        int idx = g_hit_list[j];
        int32_t score = 0;
        for (int m = 0; m < M_active; m++) {
            score += m4t_popcount_dist(
                query_sigs_per_table[m],
                train_sigs[m] + (size_t)idx * Sp,
                mask, Sp);
        }
        if (score < best_score) { best_score = score; best_label = y_train[idx]; }
    }
    return best_label;
}

/* PTM: per-table majority. For each table, find the candidate in the
 * union with smallest popcount_dist in that table's signature; collect
 * M labels; majority vote. */
static int resolver_per_table_majority(int M_active, int Sp,
                                       uint8_t* const* train_sigs,
                                       uint8_t* const* query_sigs_per_table,
                                       const uint8_t* mask, const int* y_train)
{
    int label_votes[N_CLASSES] = {0};
    for (int m = 0; m < M_active; m++) {
        int32_t best_d = INT32_MAX;
        int best_label = -1;
        for (int j = 0; j < g_n_hit; j++) {
            int idx = g_hit_list[j];
            int32_t d = m4t_popcount_dist(
                query_sigs_per_table[m],
                train_sigs[m] + (size_t)idx * Sp,
                mask, Sp);
            if (d < best_d) { best_d = d; best_label = y_train[idx]; }
        }
        if (best_label >= 0) label_votes[best_label]++;
    }
    int pred = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (label_votes[c] > label_votes[pred]) pred = c;
    return pred;
}

/* ── Oracle check: is the true label present in the current union? ──── */

static int oracle_check(int y_true, const int* y_train) {
    for (int j = 0; j < g_n_hit; j++)
        if (y_train[g_hit_list[j]] == y_true) return 1;
    return 0;
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    if(argc<2){fprintf(stderr,"Usage: %s <mnist_dir> [--full]\n",argv[0]);return 1;}
    int mode_full = (argc >= 3 && strcmp(argv[2], "--full") == 0);

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

    if (n_train > N_TRAIN_MAX) {
        fprintf(stderr, "n_train=%d exceeds N_TRAIN_MAX=%d\n", n_train, N_TRAIN_MAX);
        return 1;
    }

    int Sp = M4T_TRIT_PACKED_BYTES(N_PROJ);
    if (Sp != SIG_BYTES) {
        fprintf(stderr, "Sig-bytes mismatch: %d vs %d\n", Sp, SIG_BYTES);
        return 1;
    }

    printf("mnist_routed_bucket_multi — multi-table routed LSH (Phase 1/2 tool)\n");
    printf("N_PROJ=%d, density=%.2f, M_MAX=%d, r_max=%d, MIN_CANDS_PER_TABLE=%d\n",
           N_PROJ, DENSITY, M_MAX, MAX_RADIUS, MIN_CANDS_PER_TABLE);
    printf("mode: %s\n", mode_full ? "FULL (oracle + all 3 resolvers)" : "ORACLE only");
    printf("%d train prototypes, %d test queries.\n\n", n_train, n_test);

    /* Allocate per-table storage. */
    uint8_t*  train_sigs[M_MAX];
    uint8_t*  test_sigs[M_MAX];
    entry_t*  entries_per_table[M_MAX];

    /* Seed families per table, derived from the table index with large
     * odd multipliers to get independent-looking 4-tuples. Seed 0 matches
     * the canonical (42,123,456,789) used by every other cascade tool. */
    uint32_t seed_bases[M_MAX][4];
    seed_bases[0][0] = 42;  seed_bases[0][1] = 123;
    seed_bases[0][2] = 456; seed_bases[0][3] = 789;
    for (int m = 1; m < M_MAX; m++) {
        uint32_t u = (uint32_t)m;
        seed_bases[m][0] = 2654435761u * u + 1013904223u;   /* golden ratio + LCG */
        seed_bases[m][1] = 1597334677u * u + 2246822519u;
        seed_bases[m][2] = 3266489917u * u +  668265263u;
        seed_bases[m][3]  = 374761393u * u + 3266489917u;
    }

    clock_t t_build_start = clock();
    for (int m = 0; m < M_MAX; m++) {
        train_sigs[m] = calloc((size_t)n_train * Sp, 1);
        test_sigs[m]  = calloc((size_t)n_test  * Sp, 1);
        build_signature_set(
            N_PROJ, x_train, n_train, x_test, n_test,
            seed_bases[m][0], seed_bases[m][1], seed_bases[m][2], seed_bases[m][3],
            DENSITY, train_sigs[m], test_sigs[m]);
        entries_per_table[m] = malloc((size_t)n_train * sizeof(entry_t));
        for (int i = 0; i < n_train; i++) {
            entries_per_table[m][i].key = sig_to_key(train_sigs[m] + (size_t)i * Sp);
            entries_per_table[m][i].proto_idx = i;
        }
        qsort(entries_per_table[m], n_train, sizeof(entry_t), cmp_entry);
    }
    double t_build = (double)(clock() - t_build_start) / CLOCKS_PER_SEC;
    printf("Built %d tables in %.2fs.\n\n", M_MAX, t_build);

    /* Distinct-bucket stats for the first table (sanity check — should match
     * the single-table bucket consumer). */
    {
        int distinct = 0;
        for (int i = 0; i < n_train; ) {
            int j = i + 1;
            while (j < n_train && entries_per_table[0][j].key == entries_per_table[0][i].key) j++;
            distinct++;
            i = j;
        }
        printf("Table 0 sanity: %d distinct buckets (single-table was 37906).\n\n",
               distinct);
    }

    uint8_t* mask = malloc(Sp);
    memset(mask, 0xFF, Sp);

    /* ── M sweep ─────────────────────────────────────────────────────── */

    int M_values[] = {1, 2, 4, 8, 16, 32, 64};
    int n_M = (int)(sizeof(M_values)/sizeof(M_values[0]));

    /* Per-M metrics. */
    int  oracle_correct[16] = {0};
    int  vote_correct[16]   = {0};
    int  sum_correct[16]    = {0};
    int  ptm_correct[16]    = {0};
    long total_union_size[16] = {0};
    long total_probes[16]     = {0};
    double wall_oracle[16] = {0};
    double wall_vote[16]   = {0};
    double wall_sum[16]    = {0};
    double wall_ptm[16]    = {0};

    /* Single buffer for per-query query-sigs (one per table, reused). */
    uint8_t* query_sigs_per_table[M_MAX];

    clock_t t_all_start = clock();
    for (int s = 0; s < n_test; s++) {
        int y = y_test[s];

        /* Initial per-table query sigs: pointers into test_sigs. */
        for (int m = 0; m < M_MAX; m++)
            query_sigs_per_table[m] = test_sigs[m] + (size_t)s * Sp;

        /* Reset union. */
        union_reset();

        /* Incrementally grow union and measure at each M checkpoint. */
        int prev_M = 0;
        for (int mi = 0; mi < n_M; mi++) {
            int M_target = M_values[mi];
            /* Probe tables [prev_M, M_target). */
            for (int m = prev_M; m < M_target; m++) {
                clock_t t0 = clock();
                probe_table_into_union(
                    entries_per_table[m], n_train, query_sigs_per_table[m]);
                wall_oracle[mi] += (double)(clock() - t0) / CLOCKS_PER_SEC;
            }

            /* Oracle check. */
            if (oracle_check(y, y_train)) oracle_correct[mi]++;
            total_union_size[mi] += g_n_hit;
            total_probes[mi]     += g_n_probes;

            /* Resolvers. */
            if (mode_full) {
                clock_t t0;

                t0 = clock();
                int pred_vote = resolver_vote(y_train);
                wall_vote[mi] += (double)(clock() - t0) / CLOCKS_PER_SEC;
                if (pred_vote == y) vote_correct[mi]++;

                t0 = clock();
                int pred_sum  = resolver_sum(
                    M_target, Sp, train_sigs, query_sigs_per_table, mask, y_train);
                wall_sum[mi] += (double)(clock() - t0) / CLOCKS_PER_SEC;
                if (pred_sum == y) sum_correct[mi]++;

                t0 = clock();
                int pred_ptm  = resolver_per_table_majority(
                    M_target, Sp, train_sigs, query_sigs_per_table, mask, y_train);
                wall_ptm[mi] += (double)(clock() - t0) / CLOCKS_PER_SEC;
                if (pred_ptm == y) ptm_correct[mi]++;
            }

            prev_M = M_target;
        }
    }
    double t_all = (double)(clock() - t_all_start) / CLOCKS_PER_SEC;

    /* ── Report ──────────────────────────────────────────────────────── */

    printf("Total wall: %.2fs for %d queries\n\n", t_all, n_test);

    printf("Phase 2 — oracle ceiling pass:\n");
    printf("   M   oracle_ceiling   avg_union   avg_probes\n");
    for (int mi = 0; mi < n_M; mi++) {
        printf("  %3d     %6.2f%%        %8.1f    %9.1f\n",
               M_values[mi],
               100.0 * oracle_correct[mi] / n_test,
               (double)total_union_size[mi] / n_test,
               (double)total_probes[mi] / n_test);
    }
    printf("\n");

    if (mode_full) {
        printf("Phase 3 — resolver sweep:\n");
        printf("   M       VOTE        SUM        PTM       oracle\n");
        for (int mi = 0; mi < n_M; mi++) {
            printf("  %3d   %6.2f%%   %6.2f%%   %6.2f%%    %6.2f%%\n",
                   M_values[mi],
                   100.0 * vote_correct[mi] / n_test,
                   100.0 * sum_correct[mi] / n_test,
                   100.0 * ptm_correct[mi] / n_test,
                   100.0 * oracle_correct[mi] / n_test);
        }
        printf("\n");
        printf("Wall time per resolver (cumulative over all queries, across whole M sweep):\n");
        printf("  VOTE: %.2fs\n", wall_vote[0] + wall_vote[1] + wall_vote[2] + wall_vote[3] + wall_vote[4] + wall_vote[5] + wall_vote[6]);
        printf("  SUM:  %.2fs\n", wall_sum[0] + wall_sum[1] + wall_sum[2] + wall_sum[3] + wall_sum[4] + wall_sum[5] + wall_sum[6]);
        printf("  PTM:  %.2fs\n", wall_ptm[0] + wall_ptm[1] + wall_ptm[2] + wall_ptm[3] + wall_ptm[4] + wall_ptm[5] + wall_ptm[6]);
        printf("\n");
    }

    printf("Gate decision: ");
    double ceiling_32 = 100.0 * oracle_correct[5] / n_test;  /* index 5 = M=32 */
    if (ceiling_32 >= 97.00) {
        printf("PASS. Oracle ceiling at M=32 = %.2f%% >= 97.00%%.\n", ceiling_32);
        printf("Phase 3 resolver sweep is justified (run with --full).\n");
    } else {
        printf("FAIL. Oracle ceiling at M=32 = %.2f%% < 97.00%%.\n", ceiling_32);
        printf("Multi-table LSH cannot reach 97%% with M<=32 at this N_PROJ.\n");
        printf("Try M=64 in ceiling below; if still below, hypothesis falsified.\n");
    }
    double ceiling_64 = 100.0 * oracle_correct[6] / n_test;
    printf("Oracle ceiling at M=64 = %.2f%%.\n", ceiling_64);

    /* Cleanup */
    for (int m = 0; m < M_MAX; m++) {
        free(entries_per_table[m]);
        free(train_sigs[m]);
        free(test_sigs[m]);
    }
    free(mask);
    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
