/*
 * test_routed_tool_smoke.c — end-to-end smoke test for a routed consumer.
 *
 * Writes a tiny synthetic MNIST dataset (10 train + 10 test, one class each)
 * in IDX format, runs mnist_trit_lattice against it, and checks that the tool
 * executes to completion through the routed classification path.
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
        uint8_t label = (uint8_t)i;
        fwrite(&label, 1, 1, f);
    }
    fclose(f);
    return 1;
}

int main(int argc, char** argv) {
    char tmp_dir[1024];
    char cmd[4096];
    char output_path[2048];
    char path[2048];
    const char* tool_path;
    FILE* out;
    char line[512];
    int saw_routed = 0;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <tool_path>\n", argv[0]);
        return 1;
    }
    tool_path = argv[1];
    snprintf(tmp_dir, sizeof(tmp_dir), "/tmp/glyph-mnist-smoke-%ld-%d",
             (long)time(NULL), (int)getpid());
    if (mkdir(tmp_dir, 0700) != 0) {
        perror("mkdir");
        return 1;
    }

    snprintf(path, sizeof(path), "%s/train-images-idx3-ubyte", tmp_dir);
    if (!write_idx_images(path, 10)) return 1;
    snprintf(path, sizeof(path), "%s/train-labels-idx1-ubyte", tmp_dir);
    if (!write_idx_labels(path, 10)) return 1;
    snprintf(path, sizeof(path), "%s/t10k-images-idx3-ubyte", tmp_dir);
    if (!write_idx_images(path, 10)) return 1;
    snprintf(path, sizeof(path), "%s/t10k-labels-idx1-ubyte", tmp_dir);
    if (!write_idx_labels(path, 10)) return 1;

    snprintf(output_path, sizeof(output_path), "%s/output.txt", tmp_dir);
    snprintf(cmd, sizeof(cmd), "\"%s\" \"%s\" > \"%s\" 2>&1", tool_path, tmp_dir, output_path);
    if (system(cmd) != 0) {
        fprintf(stderr, "routed tool returned non-zero\n");
        return 1;
    }

    out = fopen(output_path, "r");
    if (!out) {
        perror("fopen");
        return 1;
    }
    while (fgets(line, sizeof(line), out)) {
        if (strstr(line, "Routed lattice geometry end to end.") ||
            strstr(line, "fully routed MNIST")) {
            saw_routed = 1;
            break;
        }
    }
    fclose(out);

    if (!saw_routed) {
        fprintf(stderr, "expected routed output marker not found\n");
        return 1;
    }
    return 0;
}
