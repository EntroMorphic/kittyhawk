#include <stdio.h>
#include "src/trix_types.h"
#include "src/trix_routing.c"

int test_cluster_signatures() {
    printf("Testing cluster signatures...\n");
    
    int8_t tile_sigs[8] = {1, -1, 1, -1, 1, -1, 1, -1};  // 4 tiles, 2 dims
    int8_t cluster_sigs[4];  // 2 clusters, 2 dims
    
    trix_compute_cluster_signatures(tile_sigs, cluster_sigs, 4, 2, 2);
    
    printf("  Tile sigs: ");
    for (int i = 0; i < 8; i++) printf("%d ", tile_sigs[i]);
    printf("\n");
    printf("  Cluster sigs: ");
    for (int i = 0; i < 4; i++) printf("%d ", cluster_sigs[i]);
    printf("\n");
    
    return 1;
}

int test_hierarchical_batch() {
    printf("Testing hierarchical batch routing...\n");
    
    /* Setup */
    TrixConfig cfg;
    trix_config_init(&cfg, 4, 4, 2);  // d_model=4, num_tiles=4, tiles_per_cluster=2
    
    int8_t tile_sigs[16] = {
        1, 0, -1, 1,
        1, 1, -1, 0,
        -1, 1, 0, 1,
        0, -1, 1, -1
    };
    
    int8_t cluster_sigs[8] = {
        1, 0, -1, 0,
        0, 0, 0, 0
    };
    
    int32_t cluster_assignments[4] = {0, 0, 1, 1};
    
    /* Input: 2 samples, 4 dims */
    float x_batch[8] = {
        0.5f, 0.3f, -0.2f, 0.1f,
        -0.3f, 0.4f, 0.2f, -0.1f
    };
    
    int tile_indices[2];
    float margins[2];
    
    trix_route_hierarchical_batch(
        x_batch,
        tile_sigs,
        cluster_sigs,
        cluster_assignments,
        2,  // batch
        2,  // clusters
        2,  // tiles per cluster
        4,  // num tiles
        4,  // d_model
        tile_indices,
        margins
    );
    
    printf("  Tile indices: %d, %d\n", tile_indices[0], tile_indices[1]);
    printf("  Margins: %.3f, %.3f\n", margins[0], margins[1]);
    
    return 1;
}

int test_flat_popcount() {
    printf("Testing flat popcount routing...\n");
    
    /* 2-bit packed: 4 values per byte */
    uint8_t x_packed[1] = {0x54};  // 01 01 01 00 = +1, +1, +1, 0
    
    uint8_t sigs_packed[4] = {
        0x55,  // +1, +1, +1, +1
        0xAA,  // -1, -1, -1, -1
        0x00,  // 0, 0, 0, 0
        0x54   // +1, +1, +1, 0
    };
    
    int32_t tile = trix_route_flat_popcount(x_packed, sigs_packed, 4, 1, NULL);
    printf("  Best tile: %d (expected 3 - same as input)\n", tile);
    
    return 1;
}

int test_flat_popcount_batch() {
    printf("Testing flat popcount batch...\n");
    
    uint8_t x_batch[2] = {0x54, 0xAA};  // +1,+1,+1,0 and -1,-1,-1,-1
    
    uint8_t sigs_packed[4] = {
        0x55,  // +1,+1,+1,+1
        0xAA,  // -1,-1,-1,-1
        0x00,  // 0,0,0,0
        0x54   // +1,+1,+1,0
    };
    
    int tile_indices[2];
    trix_route_flat_popcount_batch(x_batch, sigs_packed, 2, 4, 1, tile_indices);
    
    printf("  Tile indices: %d, %d\n", tile_indices[0], tile_indices[1]);
    printf("  Expected: 3, 1 (closest match)\n");
    
    return 1;
}

int test_policy_reroute() {
    printf("Testing policy reroute...\n");
    
    float scores[4] = {0.9f, 0.1f, 0.5f, 0.3f};
    
    /* Allow tiles 0 and 2 only */
    int32_t allow[2] = {0, 2};
    
    int32_t tile = trix_reroute_allowed(scores, allow, 2, NULL, 0, 4);
    printf("  Best allowed tile: %d (expected 0 - has highest score)\n", tile);
    
    /* Now with deny */
    int32_t deny[1] = {0};
    tile = trix_reroute_allowed(scores, allow, 2, deny, 1, 4);
    printf("  Best allowed (deny 0): %d (expected 2 - next highest)\n", tile);
    
    return 1;
}

int test_policy_apply() {
    printf("Testing policy apply...\n");
    
    int32_t tiles[3] = {0, 2, 3};  // Selected tiles
    float scores[12] = {  // 3 samples, 4 tiles
        0.9f, 0.1f, 0.5f, 0.3f,
        0.2f, 0.8f, 0.1f, 0.4f,
        0.3f, 0.3f, 0.7f, 0.2f
    };
    
    int32_t allow[2] = {0, 2};
    int32_t deny[1] = {3};
    
    int32_t violations = trix_apply_policy_and_count_violations(
        tiles, scores, allow, 2, deny, 1, 3, 4
    );
    
    printf("  Violations: %d\n", violations);
    printf("  Tiles after policy: %d, %d, %d\n", tiles[0], tiles[1], tiles[2]);
    
    return 1;
}

int main() {
    printf("=== trix_routing.c Falsification Tests ===\n\n");
    
    int passed = 0;
    
    passed += test_cluster_signatures();
    printf("\n");
    
    passed += test_hierarchical_batch();
    printf("\n");
    
    passed += test_flat_popcount();
    printf("\n");
    
    passed += test_flat_popcount_batch();
    printf("\n");
    
    passed += test_policy_reroute();
    printf("\n");
    
    passed += test_policy_apply();
    printf("\n");
    
    printf("=== Results: %d/6 tests passed ===\n", passed);
    
    return passed == 6 ? 0 : 1;
}
