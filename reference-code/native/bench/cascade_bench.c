/*
 * cascade_bench.c — Benchmark for NEON-optimized Cascaded Tile Router
 *
 * Measures:
 * - Throughput (tokens/sec)
 * - Latency (ns/token)
 * - Early exit rate
 * - Memory bandwidth utilization
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Forward declarations */
typedef struct TrixCascadeEngine TrixCascadeEngine;

extern TrixCascadeEngine* trix_cascade_engine_create(
    int32_t d_model,
    int32_t num_sets,
    int32_t tiles_per_set,
    int32_t compress_hidden,
    int32_t grid_size,
    int32_t max_batch,
    float confidence_threshold);

extern void trix_cascade_engine_destroy(TrixCascadeEngine* e);

extern int trix_cascade_engine_forward(
    TrixCascadeEngine* e,
    const float* x,
    float* output,
    int32_t batch_size,
    int32_t* exit_sets,
    float* confidences,
    int32_t* tiles);

extern void trix_cascade_engine_set_threshold(TrixCascadeEngine* e, float threshold);
extern float trix_cascade_engine_get_threshold(TrixCascadeEngine* e);

/* ============================================================================
 * Timing Utilities
 * ============================================================================ */

static inline double get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* ============================================================================
 * Benchmark Configurations
 * ============================================================================ */

typedef struct {
    const char* name;
    int32_t d_model;
    int32_t num_sets;
    int32_t tiles_per_set;
    int32_t batch_size;
    int32_t warmup_iters;
    int32_t bench_iters;
    float threshold;
} BenchConfig;

static const BenchConfig CONFIGS[] = {
    /* Small model configs */
    {"small-batch1",    128, 2, 16, 1,    10, 100, 0.9f},
    {"small-batch32",   128, 2, 16, 32,   10, 100, 0.9f},
    {"small-batch256",  128, 2, 16, 256,  10, 100, 0.9f},
    {"small-batch1024", 128, 2, 16, 1024, 10, 100, 0.9f},
    
    /* Medium model configs */
    {"med-batch1",      256, 3, 32, 1,    10, 100, 0.9f},
    {"med-batch32",     256, 3, 32, 32,   10, 100, 0.9f},
    {"med-batch256",    256, 3, 32, 256,  10, 100, 0.9f},
    {"med-batch1024",   256, 3, 32, 1024, 10, 100, 0.9f},
    
    /* Large model configs */
    {"large-batch1",    512, 4, 64, 1,    10, 50,  0.9f},
    {"large-batch32",   512, 4, 64, 32,   10, 50,  0.9f},
    {"large-batch256",  512, 4, 64, 256,  10, 50,  0.9f},
    
    /* Threshold sweep (medium config) */
    {"med-thresh-0.5",  256, 3, 32, 256,  10, 100, 0.5f},
    {"med-thresh-0.7",  256, 3, 32, 256,  10, 100, 0.7f},
    {"med-thresh-0.8",  256, 3, 32, 256,  10, 100, 0.8f},
    {"med-thresh-0.9",  256, 3, 32, 256,  10, 100, 0.9f},
    {"med-thresh-0.95", 256, 3, 32, 256,  10, 100, 0.95f},
    {"med-thresh-1.0",  256, 3, 32, 256,  10, 100, 1.0f},
};

#define NUM_CONFIGS (sizeof(CONFIGS) / sizeof(CONFIGS[0]))

/* ============================================================================
 * Benchmark Runner
 * ============================================================================ */

static void run_benchmark(const BenchConfig* cfg) {
    printf("\n=== %s ===\n", cfg->name);
    printf("d_model=%d, num_sets=%d, tiles=%d, batch=%d, threshold=%.2f\n",
           cfg->d_model, cfg->num_sets, cfg->tiles_per_set, 
           cfg->batch_size, cfg->threshold);
    
    /* Create engine */
    int32_t compress_hidden = cfg->d_model / 4;
    int32_t grid_size = 16;
    
    TrixCascadeEngine* engine = trix_cascade_engine_create(
        cfg->d_model, cfg->num_sets, cfg->tiles_per_set,
        compress_hidden, grid_size, cfg->batch_size, cfg->threshold);
    
    if (!engine) {
        printf("ERROR: Failed to create engine\n");
        return;
    }
    
    /* Allocate input/output */
    size_t input_size = (size_t)cfg->batch_size * cfg->d_model * sizeof(float);
    size_t output_size = input_size;
    
    float* x = (float*)malloc(input_size);
    float* y = (float*)malloc(output_size);
    int32_t* exit_sets = (int32_t*)malloc(cfg->batch_size * sizeof(int32_t));
    float* confidences = (float*)malloc(cfg->batch_size * sizeof(float));
    int32_t* tiles = (int32_t*)malloc(cfg->batch_size * sizeof(int32_t));
    
    /* Initialize input with random data */
    srand(42);
    for (int i = 0; i < cfg->batch_size * cfg->d_model; i++) {
        x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    
    /* Warmup */
    for (int i = 0; i < cfg->warmup_iters; i++) {
        trix_cascade_engine_forward(engine, x, y, cfg->batch_size,
                                    exit_sets, confidences, tiles);
    }
    
    /* Benchmark */
    double total_time_ns = 0.0;
    double min_time_ns = 1e15;
    double max_time_ns = 0.0;
    
    int64_t total_early_exits = 0;
    
    for (int iter = 0; iter < cfg->bench_iters; iter++) {
        double start = get_time_ns();
        
        int rc = trix_cascade_engine_forward(engine, x, y, cfg->batch_size,
                                              exit_sets, confidences, tiles);
        
        double end = get_time_ns();
        double elapsed = end - start;
        
        if (rc != 0) {
            printf("ERROR: Forward pass failed with code %d\n", rc);
            break;
        }
        
        total_time_ns += elapsed;
        if (elapsed < min_time_ns) min_time_ns = elapsed;
        if (elapsed > max_time_ns) max_time_ns = elapsed;
        
        /* Count early exits (exit before last set) */
        for (int b = 0; b < cfg->batch_size; b++) {
            if (exit_sets[b] < cfg->num_sets - 1) {
                total_early_exits++;
            }
        }
    }
    
    /* Compute statistics */
    double avg_time_ns = total_time_ns / cfg->bench_iters;
    int64_t total_tokens = (int64_t)cfg->bench_iters * cfg->batch_size;
    double tokens_per_sec = total_tokens / (total_time_ns / 1e9);
    double ns_per_token = avg_time_ns / cfg->batch_size;
    double early_exit_rate = (double)total_early_exits / total_tokens;
    
    /* Memory bandwidth estimation */
    /* Per forward: read input + signatures + weights, write output */
    size_t bytes_per_forward = 
        input_size +                                          /* input read */
        output_size +                                         /* output write */
        cfg->num_sets * cfg->tiles_per_set * cfg->d_model +  /* signatures */
        cfg->num_sets * compress_hidden * cfg->d_model * 4 + /* compress_w1 */
        cfg->num_sets * cfg->tiles_per_set * cfg->d_model * 4; /* directions */
    
    double mem_bw_gbps = (bytes_per_forward * cfg->bench_iters) / total_time_ns;
    
    /* Print results */
    printf("\nResults:\n");
    printf("  Throughput:      %.2f M tokens/sec\n", tokens_per_sec / 1e6);
    printf("  Latency (avg):   %.1f ns/token\n", ns_per_token);
    printf("  Latency (min):   %.1f ns/batch\n", min_time_ns);
    printf("  Latency (max):   %.1f ns/batch\n", max_time_ns);
    printf("  Early exit rate: %.1f%%\n", early_exit_rate * 100.0);
    printf("  Est. mem BW:     %.2f GB/s\n", mem_bw_gbps);
    
    /* Sample output verification */
    float sum = 0.0f;
    for (int i = 0; i < cfg->d_model; i++) {
        sum += y[i];
    }
    printf("  Output checksum: %.6f (sample[0])\n", sum);
    
    /* Cleanup */
    free(x);
    free(y);
    free(exit_sets);
    free(confidences);
    free(tiles);
    trix_cascade_engine_destroy(engine);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    printf("=== NEON Cascade Router Benchmark ===\n");
    printf("Compiled: %s %s\n", __DATE__, __TIME__);
    
#ifdef __ARM_NEON
    printf("NEON: ENABLED\n");
#else
    printf("NEON: DISABLED (scalar fallback)\n");
#endif
    
#ifdef TRIX_THREAD_SAFE
    printf("Thread safety: ENABLED\n");
#else
    printf("Thread safety: DISABLED\n");
#endif
    
    /* Run specific config or all */
    if (argc > 1) {
        const char* name = argv[1];
        for (size_t i = 0; i < NUM_CONFIGS; i++) {
            if (strcmp(CONFIGS[i].name, name) == 0) {
                run_benchmark(&CONFIGS[i]);
                return 0;
            }
        }
        printf("Unknown config: %s\n", name);
        printf("Available configs:\n");
        for (size_t i = 0; i < NUM_CONFIGS; i++) {
            printf("  %s\n", CONFIGS[i].name);
        }
        return 1;
    }
    
    /* Run all benchmarks */
    for (size_t i = 0; i < NUM_CONFIGS; i++) {
        run_benchmark(&CONFIGS[i]);
    }
    
    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
