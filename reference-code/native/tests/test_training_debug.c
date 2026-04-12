/*
 * test_training_debug.c — Diagnose training dynamics
 *
 * Runs a few training steps on a tiny corpus and prints:
 *   - Gradient norms per component (emb, mixer, ffn, head) per block
 *   - Output_scale trajectory for each mixer and FFN
 *   - Pool counts per tile (are tiles getting used?)
 *   - Logit distribution (is the model collapsing to one prediction?)
 *   - Routing pattern evolution
 */

#include "trix_routed_block.h"
#include "trix_routed_mixer_causal.h"
#include "trix_routed_proj.h"
#include "trix_atoms.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VOCAB   128
#define D       32
#define T       8
#define H       64
#define K       3
#define LAYERS  2
#define SEQ     16

static const char* CORPUS = "the cat sat on the mat. the dog sat on the log. ";

static void char_to_vec(float* out, int ch) {
    memset(out, 0, VOCAB * sizeof(float));
    if (ch >= 0 && ch < VOCAB) out[ch] = 1.0f;
}

int main(void) {
    int clen = (int)strlen(CORPUS);

    printf("=== TRAINING DEBUG ===\n");
    printf("D=%d T=%d H=%d K=%d L=%d seq=%d corpus=%d\n\n", D, T, H, K, LAYERS, SEQ, clen);

    /* Create model */
    TrixRoutedProjConfig emb_cfg = {
        .in_dim = VOCAB, .out_dim = D, .num_tiles = T, .active_k = K,
        .output_scale_init = 1.0f, .ln_eps = 1e-5f, .use_layernorm = false
    };
    TrixRoutedProj* emb = trix_routed_proj_create(emb_cfg, 1);

    TrixRoutedBlockConfig blk_cfg = {
        .d_model = D, .num_tiles = T, .tile_hidden = H, .active_k = K,
        .mixer_scale_init = 0.1f, .ffn_scale_init = 0.1f, .ln_eps = 1e-5f
    };
    TrixRoutedBlock* blocks[LAYERS];
    for (int l = 0; l < LAYERS; l++)
        blocks[l] = trix_routed_block_create(blk_cfg, 1000 + l * 1000);

    TrixRoutedProjConfig head_cfg = {
        .in_dim = D, .out_dim = VOCAB, .num_tiles = T, .active_k = K,
        .output_scale_init = 1.0f, .ln_eps = 1e-5f, .use_layernorm = true
    };
    TrixRoutedProj* head = trix_routed_proj_create(head_cfg, 9999);

    /* Scratch */
    float* x_oh = calloc(SEQ * VOCAB, sizeof(float));
    float* emb_out = calloc(SEQ * D, sizeof(float));
    float* logits = calloc(SEQ * VOCAB, sizeof(float));
    float* d_logits = calloc(SEQ * VOCAB, sizeof(float));
    float* d_head = calloc(SEQ * D, sizeof(float));
    float* d_layer = calloc(SEQ * D, sizeof(float));
    int* targets = calloc(SEQ, sizeof(int));

    int n_steps = 20;
    float lr = 0.003f;

    for (int step = 0; step < n_steps; step++) {
        int start = step % (clen - SEQ - 1);

        /* Encode */
        for (int i = 0; i < SEQ; i++)
            char_to_vec(x_oh + i * VOCAB, (int)(unsigned char)CORPUS[start + i]);
        for (int i = 0; i < SEQ; i++)
            targets[i] = (int)(unsigned char)CORPUS[start + i + 1];

        /* Forward: embed */
        for (int i = 0; i < SEQ; i++)
            trix_routed_proj_forward(emb, x_oh + i * VOCAB, emb_out + i * D, 1);

        /* Forward: blocks (save activations) */
        float* saved[LAYERS + 1];
        saved[0] = malloc(SEQ * D * sizeof(float));
        memcpy(saved[0], emb_out, SEQ * D * sizeof(float));
        for (int l = 0; l < LAYERS; l++) {
            saved[l + 1] = malloc(SEQ * D * sizeof(float));
            trix_routed_block_forward_causal(blocks[l], saved[l], saved[l + 1], SEQ);
        }

        /* Forward: head */
        for (int i = 0; i < SEQ; i++)
            trix_routed_proj_forward(head, saved[LAYERS] + i * D, logits + i * VOCAB, 1);

        /* Loss */
        float loss = trix_cross_entropy_loss(logits, targets, SEQ, VOCAB);

        /* Logit stats: check for collapse */
        float logit_max = -1e30f, logit_min = 1e30f, logit_mean = 0;
        for (int i = 0; i < SEQ * VOCAB; i++) {
            if (logits[i] > logit_max) logit_max = logits[i];
            if (logits[i] < logit_min) logit_min = logits[i];
            logit_mean += logits[i];
        }
        logit_mean /= (SEQ * VOCAB);

        /* Top prediction for each position */
        printf("step %2d: loss=%.3f logits=[%.2f,%.2f,%.2f] preds=\"", step, loss, logit_min, logit_mean, logit_max);
        for (int i = 0; i < SEQ; i++) {
            int best = 0; float bv = logits[i * VOCAB];
            for (int c = 1; c < VOCAB; c++) if (logits[i * VOCAB + c] > bv) { bv = logits[i * VOCAB + c]; best = c; }
            putchar((best >= 32 && best < 127) ? best : '?');
        }
        printf("\"\n");

        /* Backward */
        trix_routed_proj_zero_grad(emb);
        for (int l = 0; l < LAYERS; l++) trix_routed_block_zero_grad(blocks[l]);
        trix_routed_proj_zero_grad(head);

        trix_cross_entropy_grad(d_logits, logits, targets, SEQ, VOCAB);
        for (int i = 0; i < SEQ; i++)
            trix_routed_proj_backward(head, saved[LAYERS] + i * D,
                                      d_logits + i * VOCAB, d_head + i * D, 1);

        memcpy(d_layer, d_head, SEQ * D * sizeof(float));
        for (int l = LAYERS - 1; l >= 0; l--) {
            float* dx = calloc(SEQ * D, sizeof(float));
            trix_routed_block_backward(blocks[l], saved[l], d_layer, dx, SEQ);
            memcpy(d_layer, dx, SEQ * D * sizeof(float));
            free(dx);
        }
        for (int l = 0; l <= LAYERS; l++) free(saved[l]);

        /* Print gradient norms and output_scales */
        for (int l = 0; l < LAYERS; l++) {
            TrixRoutedMixer* rm = blocks[l]->mixer;
            TrixTernaryRoutedFFN* tr = blocks[l]->ffn;

            float mixer_gnorm = sqrtf(
                trix_sum_sq(rm->dW1, T*H*D) + trix_sum_sq(rm->dW2, T*D*H) +
                trix_sum_sq(rm->db1, T*H) + trix_sum_sq(rm->db2, T*D));
            float ffn_gnorm = sqrtf(
                trix_sum_sq(tr->dW1, T*H*D) + trix_sum_sq(tr->dW2, T*D*H) +
                trix_sum_sq(tr->db1, T*H) + trix_sum_sq(tr->db2, T*D));

            printf("  blk%d: mixer_gnorm=%.6f os=%.4f | ffn_gnorm=%.6f os=%.4f",
                   l, mixer_gnorm, rm->output_scale, ffn_gnorm, tr->output_scale);

            /* Pool counts from last causal forward */
            printf(" pools=[");
            for (int t = 0; t < T; t++) printf("%s%d", t ? "," : "", rm->pool_counts[t]);
            printf("]\n");
        }

        /* Update */
        for (int l = 0; l < LAYERS; l++)
            trix_routed_block_clip_grad_norm(blocks[l], 1.0f);
        trix_routed_proj_adamw_step(emb, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
        for (int l = 0; l < LAYERS; l++)
            trix_routed_block_adamw_step(blocks[l], lr, 0.9f, 0.999f, 1e-8f, 0.01f);
        trix_routed_proj_adamw_step(head, lr, 0.9f, 0.999f, 1e-8f, 0.01f);

        printf("\n");
    }

    free(x_oh); free(emb_out); free(logits); free(d_logits);
    free(d_head); free(d_layer); free(targets);
    trix_routed_proj_destroy(emb);
    for (int l = 0; l < LAYERS; l++) trix_routed_block_destroy(blocks[l]);
    trix_routed_proj_destroy(head);
    return 0;
}
