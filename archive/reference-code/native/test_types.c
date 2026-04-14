#include <stdio.h>
#include "src/trix_types.h"

int main() {
    printf("Testing trix_types.h...\n");
    
    TrixConfig cfg;
    trix_config_init(&cfg, 512, 64, 8);
    
    printf("Config: d_model=%d, num_tiles=%d, tiles_per_cluster=%d\n", 
           cfg.d_model, cfg.num_tiles, cfg.tiles_per_cluster);
    printf("  compress_hidden=%d, grid_size=%d, num_clusters=%d\n",
           cfg.compress_hidden, cfg.grid_size, cfg.num_clusters);
    printf("  backend=%d, use_score_calibration=%d\n",
           cfg.backend, cfg.use_score_calibration);
    
    TrixWeights* w = trix_weights_create(&cfg);
    printf("Weights created: signatures=%p, directions=%p\n", 
           (void*)w->signatures, (void*)w->directions);
    
    TrixTrainState* s = trix_train_state_create(&cfg);
    printf("TrainState created: grad_signatures=%p, tile_counts=%p\n",
           (void*)s->grad_signatures, (void*)s->tile_counts);
    
    TrixPolicy* p = trix_policy_create();
    printf("Policy created: on_violation=%d\n", p->on_violation);
    
    TrixDiag* d = trix_diag_create(64);
    printf("Diag created: tile_indices=%p, batch_size=%d\n",
           (void*)d->tile_indices, d->batch_size);
    
    TrixAuxLosses l;
    trix_aux_losses_zero(&l);
    printf("Aux losses: total_aux=%.2f\n", l.total_aux);
    
    TrixEngine* e = trix_engine_create(&cfg);
    printf("Engine created\n");
    
    /* Test scratch allocation */
    trix_engine_ensure_scratch(e, 128);
    printf("Scratch capacity: %d\n", e->scratch_capacity);
    
    /* Test freeze */
    s->frozen_tiles[0] = true;
    s->num_frozen = 1;
    printf("Frozen tiles: %d\n", s->num_frozen);
    
    /* Cleanup */
    trix_engine_destroy(e);
    trix_diag_destroy(d);
    trix_policy_destroy(p);
    trix_train_state_destroy(s);
    trix_weights_destroy(w);
    
    printf("All tests passed!\n");
    return 0;
}
