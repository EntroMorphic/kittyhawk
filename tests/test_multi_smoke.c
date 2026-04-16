/*
 * test_multi_smoke.c — smoke test for mnist_routed_bucket_multi.
 *
 * Writes a tiny synthetic MNIST dataset, runs the multi-table
 * consumer in both fixed and mixed density_schedule modes, and
 * asserts both complete successfully.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

static void write_u32_be(FILE* f, uint32_t v) {
    uint8_t b[4];
    b[0] = (uint8_t)(v >> 24);
    b[1] = (uint8_t)(v >> 16);
    b[2] = (uint8_t)(v >> 8);
    b[3] = (uint8_t)v;
    fwrite(b, 1, 4, f);
}

static int write_idx_images(const char* path, int n) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    write_u32_be(f, 2051);
    write_u32_be(f, (uint32_t)n);
    write_u32_be(f, 28);
    write_u32_be(f, 28);
    for (int img = 0; img < n; img++) {
        uint8_t pixels[28 * 28];
        memset(pixels, 0, sizeof(pixels));
        for (int y = 0; y < 28; y++) {
            int x = (img * 3 + y) % 28;
            pixels[y * 28 + x] = 255;
            pixels[y * 28 + ((x + img + 1) % 28)] = 96;
        }
        fwrite(pixels, 1, sizeof(pixels), f);
    }
    fclose(f);
    return 1;
}

static int write_idx_labels(const char* path, int n) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    write_u32_be(f, 2049);
    write_u32_be(f, (uint32_t)n);
    for (int i = 0; i < n; i++) {
        uint8_t label = (uint8_t)(i % 10);
        fwrite(&label, 1, 1, f);
    }
    fclose(f);
    return 1;
}

static int run_tool(const char* tool, const char* data_dir,
                    const char* extra_flags, const char* output_path) {
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
             "\"%s\" --data \"%s\" --mode full --m_max 3 --single_m 3 %s > \"%s\" 2>&1",
             tool, data_dir, extra_flags, output_path);
    return system(cmd);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <mnist_routed_bucket_multi_path>\n", argv[0]);
        return 1;
    }
    const char* tool = argv[1];

    char tmp_dir[1024];
    snprintf(tmp_dir, sizeof(tmp_dir), "/tmp/glyph-multi-smoke-%ld-%d",
             (long)time(NULL), (int)getpid());
    if (mkdir(tmp_dir, 0700) != 0) { perror("mkdir"); return 1; }

    char path[2048];
    snprintf(path, sizeof(path), "%s/train-images-idx3-ubyte", tmp_dir);
    if (!write_idx_images(path, 20)) return 1;
    snprintf(path, sizeof(path), "%s/train-labels-idx1-ubyte", tmp_dir);
    if (!write_idx_labels(path, 20)) return 1;
    snprintf(path, sizeof(path), "%s/t10k-images-idx3-ubyte", tmp_dir);
    if (!write_idx_images(path, 10)) return 1;
    snprintf(path, sizeof(path), "%s/t10k-labels-idx1-ubyte", tmp_dir);
    if (!write_idx_labels(path, 10)) return 1;

    char output[2048];

    /* Test 1: fixed schedule (default). */
    snprintf(output, sizeof(output), "%s/out_fixed.txt", tmp_dir);
    if (run_tool(tool, tmp_dir, "", output) != 0) {
        fprintf(stderr, "FAIL: fixed schedule returned non-zero\n");
        return 1;
    }
    printf("PASS: fixed schedule\n");

    /* Test 2: mixed schedule with explicit triple. */
    snprintf(output, sizeof(output), "%s/out_mixed.txt", tmp_dir);
    if (run_tool(tool, tmp_dir,
                 "--density_schedule mixed --density_triple 0.20,0.33,0.50",
                 output) != 0) {
        fprintf(stderr, "FAIL: mixed schedule returned non-zero\n");
        return 1;
    }
    printf("PASS: mixed schedule (0.20,0.33,0.50)\n");

    /* Test 3: mixed schedule with narrow triple. */
    snprintf(output, sizeof(output), "%s/out_narrow.txt", tmp_dir);
    if (run_tool(tool, tmp_dir,
                 "--density_schedule mixed --density_triple 0.25,0.33,0.40",
                 output) != 0) {
        fprintf(stderr, "FAIL: narrow mixed schedule returned non-zero\n");
        return 1;
    }
    printf("PASS: mixed schedule (0.25,0.33,0.40)\n");

    printf("All multi-table smoke tests passed.\n");
    return 0;
}
