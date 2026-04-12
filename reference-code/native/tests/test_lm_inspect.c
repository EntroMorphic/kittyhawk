/*
 * test_lm_inspect.c — Inspect what the trained LM actually predicts
 *
 * Train for 50 epochs on embedded corpus, then:
 *   1. Show top-5 predictions for each position in a test sequence
 *   2. Show logit distribution (is it peaked or flat?)
 *   3. Try sampling (temperature + top-k) instead of greedy
 */

#include "trix_routed_block.h"
#include "trix_routed_mixer_causal.h"
#include "trix_routed_proj.h"
#include "trix_atoms.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define VOCAB   128
#define D       64
#define T       4
#define H       128
#define K       2
#define LAYERS  8
#define SEQ     32
#define BATCH   16
#define LR      0.00168f
#define EPOCHS  50

static const char* CORPUS =
    "the cat sat on the mat and the dog sat on the log. "
    "a bird in the hand is worth two in the bush. "
    "to be or not to be that is the question. "
    "all that glitters is not gold. "
    "the quick brown fox jumps over the lazy dog. "
    "she sells sea shells by the sea shore. "
    "how much wood would a wood chuck chuck. "
    "peter piper picked a peck of pickled peppers. "
    "the rain in spain falls mainly on the plain. "
    "jack and jill went up the hill to fetch a pail of water. "
    "humpty dumpty sat on a wall humpty dumpty had a great fall. "
    "mary had a little lamb its fleece was white as snow. "
    "twinkle twinkle little star how i wonder what you are. "
    "row row row your boat gently down the stream. "
    "old king cole was a merry old soul and a merry old soul was he. "
    "one fish two fish red fish blue fish. "
    "green eggs and ham i do not like green eggs and ham. "
    "the cat in the hat came back with a little cat under his hat. "
    "oh the places you will go today is your day. "
    "i meant what i said and i said what i meant. ";

static uint64_t rng = 42;
static int rand_int(int n) { rng = rng*6364136223846793005ULL+1442695040888963407ULL; return (int)((rng>>33)%(uint64_t)n); }

static void char_to_vec(float* out, int ch) {
    memset(out, 0, VOCAB * sizeof(float));
    if (ch >= 0 && ch < VOCAB) out[ch] = 1.0f;
}

int main(void) {
    int clen = (int)strlen(CORPUS);
    int train_len = (int)(clen * 0.8f);

    printf("=== LM INSPECT ===\n");
    printf("Training %d epochs on %d chars...\n", EPOCHS, train_len);

    /* Create model (same as test_routed_lm.c) */
    TrixRoutedProjConfig emb_cfg = { .in_dim=VOCAB, .out_dim=D, .num_tiles=T, .active_k=K, .output_scale_init=1.0f, .ln_eps=1e-5f, .use_layernorm=false };
    TrixRoutedProj* emb = trix_routed_proj_create(emb_cfg, 1);

    TrixRoutedBlockConfig blk_cfg = { .d_model=D, .num_tiles=T, .tile_hidden=H, .active_k=K, .mixer_scale_init=0.1f, .ffn_scale_init=0.1f, .ln_eps=1e-5f };
    TrixRoutedBlock* blocks[LAYERS];
    for (int l = 0; l < LAYERS; l++) blocks[l] = trix_routed_block_create(blk_cfg, 1000+l*1000);

    TrixRoutedProjConfig head_cfg = { .in_dim=D, .out_dim=VOCAB, .num_tiles=T, .active_k=K, .output_scale_init=1.0f, .ln_eps=1e-5f, .use_layernorm=true };
    TrixRoutedProj* head = trix_routed_proj_create(head_cfg, 9999);

    float* x_oh = calloc(SEQ*VOCAB, sizeof(float));
    mtfp_t* mx_oh = calloc(SEQ*VOCAB, sizeof(mtfp_t));
    mtfp_t* emb_out = calloc(SEQ*D, sizeof(mtfp_t));
    mtfp_t* buf_a = calloc(SEQ*D, sizeof(mtfp_t));
    mtfp_t* buf_b = calloc(SEQ*D, sizeof(mtfp_t));
    mtfp_t* mlogits = calloc(SEQ*VOCAB, sizeof(mtfp_t));
    float* logits = calloc(SEQ*VOCAB, sizeof(float));
    float* d_logits = calloc(SEQ*VOCAB, sizeof(float));
    float* d_head = calloc(SEQ*D, sizeof(float));
    float* d_layer = calloc(SEQ*D, sizeof(float));
    int* targets = calloc(SEQ, sizeof(int));

    /* Train */
    int n_batches = (train_len - SEQ - 1) / BATCH;
    if (n_batches < 1) n_batches = 1;

    for (int ep = 0; ep < EPOCHS; ep++) {
        float eloss = 0;
        for (int b = 0; b < n_batches; b++) {
            trix_routed_proj_zero_grad(emb);
            for (int l = 0; l < LAYERS; l++) trix_routed_block_zero_grad(blocks[l]);
            trix_routed_proj_zero_grad(head);

            for (int s = 0; s < BATCH; s++) {
                int start = rand_int(train_len - SEQ - 1);
                for (int i = 0; i < SEQ; i++) char_to_vec(x_oh+i*VOCAB, (int)(unsigned char)CORPUS[start+i]);
                for (int i = 0; i < SEQ; i++) targets[i] = (int)(unsigned char)CORPUS[start+i+1];

                /* Forward: embed — MTFP end to end */
                mtfp_from_float_batch(mx_oh, x_oh, SEQ*VOCAB);
                trix_routed_proj_forward_mtfp(emb, mx_oh, emb_out, SEQ);

                /* Forward: blocks — MTFP end to end (save activations) */
                mtfp_t* saved[LAYERS+1];
                saved[0] = malloc(SEQ*D*sizeof(mtfp_t)); memcpy(saved[0], emb_out, SEQ*D*sizeof(mtfp_t));
                for (int l = 0; l < LAYERS; l++) {
                    saved[l+1] = malloc(SEQ*D*sizeof(mtfp_t));
                    trix_routed_block_forward_causal_mtfp(blocks[l], saved[l], saved[l+1], SEQ);
                }

                /* Forward: head — MTFP, then convert to float for loss only */
                trix_routed_proj_forward_mtfp(head, saved[LAYERS], mlogits, SEQ);
                mtfp_to_float_batch(logits, mlogits, SEQ*VOCAB);
                eloss += trix_cross_entropy_loss(logits, targets, SEQ, VOCAB);

                /* Backward uses float (shadow weights + STE).
                 * Convert saved MTFP activations to float for backward. */
                trix_cross_entropy_grad(d_logits, logits, targets, SEQ, VOCAB);

                float* saved_f[LAYERS+1];
                for (int l = 0; l <= LAYERS; l++) {
                    saved_f[l] = malloc(SEQ*D*sizeof(float));
                    mtfp_to_float_batch(saved_f[l], saved[l], SEQ*D);
                }

                /* Convert head input (last layer MTFP) for backward */
                float* head_in_f = malloc(SEQ*VOCAB*sizeof(float));
                mtfp_to_float_batch(head_in_f, saved[LAYERS], SEQ*D);
                trix_routed_proj_backward(head, head_in_f, d_logits, d_head, SEQ);
                free(head_in_f);

                /* Backward: blocks (float path) */
                memcpy(d_layer, d_head, SEQ*D*sizeof(float));
                for (int l = LAYERS-1; l >= 0; l--) {
                    float* dx = calloc(SEQ*D, sizeof(float));
                    trix_routed_block_backward(blocks[l], saved_f[l], d_layer, dx, SEQ);
                    memcpy(d_layer, dx, SEQ*D*sizeof(float));
                    free(dx);
                }

                for (int l = 0; l <= LAYERS; l++) { free(saved[l]); free(saved_f[l]); }

                /* Backward: embed (float) */
                trix_routed_proj_backward(emb, x_oh, d_layer, NULL, SEQ);
            }

            for (int l = 0; l < LAYERS; l++) trix_routed_block_clip_grad_norm(blocks[l], 1.0f);
            trix_routed_proj_adamw_step(emb, LR, 0.9f, 0.999f, 1e-8f, 0.01f);
            for (int l = 0; l < LAYERS; l++) trix_routed_block_adamw_step(blocks[l], LR, 0.9f, 0.999f, 1e-8f, 0.01f);
            trix_routed_proj_adamw_step(head, LR, 0.9f, 0.999f, 1e-8f, 0.01f);
        }
        if (ep % 10 == 0 || ep == EPOCHS-1)
            printf("  ep %2d: loss=%.3f\n", ep, eloss / n_batches / BATCH);
    }

    /* ── Inspect: top-5 predictions per position ── */
    printf("\n── TOP-5 PREDICTIONS ──\n");
    const char* test_seq = "the cat sat on the mat";
    int tlen = (int)strlen(test_seq);
    if (tlen > SEQ) tlen = SEQ;

    for (int i = 0; i < tlen; i++) char_to_vec(x_oh+i*VOCAB, (int)(unsigned char)test_seq[i]);
    mtfp_from_float_batch(mx_oh, x_oh, tlen*VOCAB);
    trix_routed_proj_forward_mtfp(emb, mx_oh, emb_out, tlen);

    memcpy(buf_a, emb_out, tlen*D*sizeof(mtfp_t));
    for (int l = 0; l < LAYERS; l++) {
        trix_routed_block_forward_causal_mtfp(blocks[l], buf_a, buf_b, tlen);
        mtfp_t* tmp = buf_a; buf_a = buf_b; buf_b = tmp;
    }
    trix_routed_proj_forward_mtfp(head, buf_a, mlogits, tlen);
    mtfp_to_float_batch(logits, mlogits, tlen*VOCAB);

    for (int pos = 0; pos < tlen; pos++) {
        float* lg = logits + pos * VOCAB;
        char actual = (pos+1 < tlen) ? test_seq[pos+1] : '?';

        /* Softmax */
        float max_l = -1e30f;
        for (int c = 0; c < VOCAB; c++) if (lg[c] > max_l) max_l = lg[c];
        float sum_exp = 0;
        float probs[128];
        for (int c = 0; c < VOCAB; c++) { probs[c] = expf(lg[c]-max_l); sum_exp += probs[c]; }
        for (int c = 0; c < VOCAB; c++) probs[c] /= sum_exp;

        /* Top 5 */
        int top[5] = {0};
        for (int k = 0; k < 5; k++) {
            int best = -1; float bv = -1.0f;
            for (int c = 0; c < VOCAB; c++) {
                int skip = 0;
                for (int j = 0; j < k; j++) if (top[j] == c) skip = 1;
                if (!skip && probs[c] > bv) { bv = probs[c]; best = c; }
            }
            top[k] = best;
        }

        char in_ch = test_seq[pos];
        printf("  '%c' → ", (in_ch >= 32 && in_ch < 127) ? in_ch : '?');
        for (int k = 0; k < 5; k++) {
            char ch = (top[k] >= 32 && top[k] < 127) ? (char)top[k] : '?';
            printf("'%c'(%.1f%%) ", ch, probs[top[k]]*100.0f);
        }
        printf(" [actual='%c']\n", (actual >= 32 && actual < 127) ? actual : '?');
    }

    /* ── Logit range ── */
    printf("\n── LOGIT STATS (last position) ──\n");
    float* last_lg = logits + (tlen-1)*VOCAB;
    float lg_min=1e30f, lg_max=-1e30f, lg_mean=0;
    for (int c = 0; c < VOCAB; c++) {
        if (last_lg[c] < lg_min) lg_min = last_lg[c];
        if (last_lg[c] > lg_max) lg_max = last_lg[c];
        lg_mean += last_lg[c];
    }
    lg_mean /= VOCAB;
    printf("  min=%.3f mean=%.3f max=%.3f range=%.3f\n", lg_min, lg_mean, lg_max, lg_max-lg_min);

    /* ── Temperature sampling ── */
    printf("\n── SAMPLES ──\n");
    float temps[] = {0.5f, 1.0f, 2.0f};
    for (int ti = 0; ti < 3; ti++) {
        float temp = temps[ti];
        printf("  T=%.1f: \"", temp);

        mtfp_t* ctx = calloc(SEQ*D, sizeof(mtfp_t));
        float* gen_oh_f = calloc(VOCAB, sizeof(float));
        mtfp_t* gen_oh_m = calloc(VOCAB, sizeof(mtfp_t));
        mtfp_t* gen_emb_v = calloc(D, sizeof(mtfp_t));

        /* Seed with "the " */
        const char* seed = "the ";
        for (int i = 0; i < 4 && i < SEQ; i++) {
            char_to_vec(gen_oh_f, (int)(unsigned char)seed[i]);
            mtfp_from_float_batch(gen_oh_m, gen_oh_f, VOCAB);
            trix_routed_proj_forward_mtfp(emb, gen_oh_m, gen_emb_v, 1);
            memcpy(ctx+i*D, gen_emb_v, D*sizeof(mtfp_t));
        }
        int ctx_len = 4;

        for (int g = 0; g < 60; g++) {
            memcpy(buf_a, ctx, ctx_len*D*sizeof(mtfp_t));
            for (int l = 0; l < LAYERS; l++) {
                trix_routed_block_forward_causal_mtfp(blocks[l], buf_a, buf_b, ctx_len);
                mtfp_t* tmp = buf_a; buf_a = buf_b; buf_b = tmp;
            }

            mtfp_t gen_mlogits[128];
            float gen_logits[128];
            trix_routed_proj_forward_mtfp(head, buf_a+(ctx_len-1)*D, gen_mlogits, 1);
            mtfp_to_float_batch(gen_logits, gen_mlogits, VOCAB);

            /* Temperature + sampling */
            float max_l = -1e30f;
            for (int c = 0; c < VOCAB; c++) if (gen_logits[c] > max_l) max_l = gen_logits[c];
            float sum_exp = 0;
            float pr[128];
            for (int c = 0; c < VOCAB; c++) { pr[c] = expf((gen_logits[c]-max_l)/temp); sum_exp += pr[c]; }
            for (int c = 0; c < VOCAB; c++) pr[c] /= sum_exp;

            /* Sample from distribution */
            float r = (float)(rand_int(10000)) / 10000.0f;
            float cum = 0; int ch = 0;
            for (int c = 0; c < VOCAB; c++) { cum += pr[c]; if (cum >= r) { ch = c; break; } }

            char out_ch = (ch >= 32 && ch < 127) ? (char)ch : '?';
            putchar(out_ch);

            /* Shift context */
            char_to_vec(gen_oh_f, ch);
            mtfp_from_float_batch(gen_oh_m, gen_oh_f, VOCAB);
            trix_routed_proj_forward_mtfp(emb, gen_oh_m, gen_emb_v, 1);
            if (ctx_len < SEQ) {
                memcpy(ctx+ctx_len*D, gen_emb_v, D*sizeof(mtfp_t));
                ctx_len++;
            } else {
                memmove(ctx, ctx+D, (SEQ-1)*D*sizeof(mtfp_t));
                memcpy(ctx+(SEQ-1)*D, gen_emb_v, D*sizeof(mtfp_t));
            }
        }
        printf("\"\n");
        free(ctx); free(gen_oh_f); free(gen_oh_m); free(gen_emb_v);
    }

    /* Cleanup */
    free(x_oh); free(mx_oh); free(emb_out); free(buf_a); free(buf_b);
    free(mlogits); free(logits); free(d_logits); free(d_head); free(d_layer); free(targets);
    trix_routed_proj_destroy(emb);
    for (int l = 0; l < LAYERS; l++) trix_routed_block_destroy(blocks[l]);
    trix_routed_proj_destroy(head);
    return 0;
}
