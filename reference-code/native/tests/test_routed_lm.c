/*
 * test_routed_lm.c — Character-level language model with the fully routed stack
 *
 * Architecture:
 *   TrixRoutedProj (vocab → D)     input embedding as routed projection
 *   L x TrixRoutedBlock            mixer + FFN per block
 *   TrixRoutedProj (D → vocab)     output head as routed projection
 *
 * No dense matmul. No attention. All ternary-routed.
 *
 * Usage:
 *   ./trix_routed_lm                         (uses embedded corpus)
 *   ./trix_routed_lm path/to/textfile.txt    (uses file)
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

/* ── Config ── */

#define VOCAB       128     /* ASCII */
#define D_MODEL     64
#define NUM_TILES   8
#define TILE_HIDDEN 128
#define ACTIVE_K    3
#define N_LAYERS    4
#define SEQ_LEN     32
#define BATCH       16
#define LR          0.003f
#define EPOCHS      50
#define GRAD_CLIP   1.0f

/* ── Embedded corpus (used when no file is provided) ── */

static const char* EMBEDDED_CORPUS =
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

/* ── PRNG ── */
static uint64_t lm_rng = 42;
static int lm_rand_int(int n) {
    lm_rng = lm_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (int)((lm_rng >> 33) % (uint64_t)n);
}

/* ── One-hot encode a character into a D_in-sized vector ── */
static void char_to_vec(float* out, int ch) {
    memset(out, 0, VOCAB * sizeof(float));
    if (ch >= 0 && ch < VOCAB) out[ch] = 1.0f;
}

int main(int argc, char** argv) {
    /* Load corpus */
    char* corpus;
    int corpus_len;

    if (argc > 1) {
        FILE* f = fopen(argv[1], "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }
        fseek(f, 0, SEEK_END); corpus_len = (int)ftell(f); fseek(f, 0, SEEK_SET);
        corpus = malloc(corpus_len + 1);
        fread(corpus, 1, corpus_len, f); fclose(f);
        corpus[corpus_len] = 0;
    } else {
        corpus_len = (int)strlen(EMBEDDED_CORPUS);
        corpus = malloc(corpus_len + 1);
        memcpy(corpus, EMBEDDED_CORPUS, corpus_len + 1);
    }

    /* Clamp to ASCII */
    for (int i = 0; i < corpus_len; i++)
        if ((unsigned char)corpus[i] >= VOCAB) corpus[i] = '?';

    /* Train/val split: 80/20 */
    int train_len = (int)(corpus_len * 0.8f);
    int val_len = corpus_len - train_len;
    char* train_corpus = corpus;
    char* val_corpus = corpus + train_len;

    printf("=== FULLY ROUTED CHARACTER LM ===\n");
    printf("corpus: %d chars (train=%d val=%d), vocab: %d (ASCII)\n",
           corpus_len, train_len, val_len, VOCAB);
    printf("D=%d T=%d H=%d K=%d L=%d seq=%d batch=%d lr=%.4f\n",
           D_MODEL, NUM_TILES, TILE_HIDDEN, ACTIVE_K, N_LAYERS, SEQ_LEN, BATCH, LR);

    /* ── Create model ── */

    /* Input: one-hot [VOCAB] → [D_MODEL] via routed projection */
    TrixRoutedProjConfig emb_cfg = {
        .in_dim = VOCAB, .out_dim = D_MODEL,
        .num_tiles = NUM_TILES, .active_k = ACTIVE_K,
        .output_scale_init = 1.0f, .ln_eps = 1e-5f,
        .use_layernorm = false  /* one-hot input, no need */
    };
    TrixRoutedProj* emb = trix_routed_proj_create(emb_cfg, 1);

    /* Blocks */
    TrixRoutedBlockConfig blk_cfg = {
        .d_model = D_MODEL, .num_tiles = NUM_TILES,
        .tile_hidden = TILE_HIDDEN, .active_k = ACTIVE_K,
        .mixer_scale_init = 0.1f, .ffn_scale_init = 0.1f,
        .ln_eps = 1e-5f
    };
    TrixRoutedBlock* blocks[N_LAYERS];
    for (int l = 0; l < N_LAYERS; l++)
        blocks[l] = trix_routed_block_create(blk_cfg, 1000 + l * 1000);

    /* Output head: [D_MODEL] → [VOCAB] via routed projection */
    TrixRoutedProjConfig head_cfg = {
        .in_dim = D_MODEL, .out_dim = VOCAB,
        .num_tiles = NUM_TILES, .active_k = ACTIVE_K,
        .output_scale_init = 1.0f, .ln_eps = 1e-5f,
        .use_layernorm = true
    };
    TrixRoutedProj* head = trix_routed_proj_create(head_cfg, 9999);

    /* ── Scratch ── */
    float* x_onehot   = calloc(SEQ_LEN * VOCAB, sizeof(float));
    float* emb_out    = calloc(SEQ_LEN * D_MODEL, sizeof(float));
    float* layer_buf  = calloc(2 * SEQ_LEN * D_MODEL, sizeof(float));
    float* logits     = calloc(SEQ_LEN * VOCAB, sizeof(float));
    float* d_logits   = calloc(SEQ_LEN * VOCAB, sizeof(float));
    float* d_head_in  = calloc(SEQ_LEN * D_MODEL, sizeof(float));
    float* d_layer    = calloc(SEQ_LEN * D_MODEL, sizeof(float));
    float* d_emb      = calloc(SEQ_LEN * VOCAB, sizeof(float));
    float* blk_in     = layer_buf;
    float* blk_out    = layer_buf + SEQ_LEN * D_MODEL;

    int* targets = calloc(SEQ_LEN, sizeof(int));

    int n_batches_per_epoch = (train_len - SEQ_LEN - 1) / BATCH;
    if (n_batches_per_epoch < 1) n_batches_per_epoch = 1;

    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int ep = 0; ep < EPOCHS; ep++) {
        float epoch_loss = 0;
        int epoch_correct = 0, epoch_total = 0;

        for (int b = 0; b < n_batches_per_epoch; b++) {
            float batch_loss = 0;

            /* Zero grads */
            trix_routed_proj_zero_grad(emb);
            for (int l = 0; l < N_LAYERS; l++)
                trix_routed_block_zero_grad(blocks[l]);
            trix_routed_proj_zero_grad(head);

            for (int s = 0; s < BATCH; s++) {
                /* Sample a random window */
                int start = lm_rand_int(train_len - SEQ_LEN - 1);

                /* Encode input: chars [start..start+SEQ_LEN-1] */
                for (int i = 0; i < SEQ_LEN; i++)
                    char_to_vec(x_onehot + i * VOCAB, (int)(unsigned char)train_corpus[start + i]);

                /* Targets: next char for each position */
                for (int i = 0; i < SEQ_LEN; i++)
                    targets[i] = (int)(unsigned char)train_corpus[start + i + 1];

                /* Forward: embed each position */
                for (int i = 0; i < SEQ_LEN; i++)
                    trix_routed_proj_forward(emb, x_onehot + i * VOCAB, emb_out + i * D_MODEL, 1);

                /* Forward: blocks (causal — position i sees only 0..i-1)
                 * Save intermediate activations for backward. */
                float* saved[N_LAYERS + 1];
                saved[0] = malloc(SEQ_LEN * D_MODEL * sizeof(float));
                memcpy(saved[0], emb_out, SEQ_LEN * D_MODEL * sizeof(float));
                for (int l = 0; l < N_LAYERS; l++) {
                    saved[l + 1] = malloc(SEQ_LEN * D_MODEL * sizeof(float));
                    trix_routed_block_forward_causal(blocks[l], saved[l], saved[l + 1], SEQ_LEN);
                }
                float* blk_final = saved[N_LAYERS];

                /* Forward: head — project each position to logits */
                for (int i = 0; i < SEQ_LEN; i++)
                    trix_routed_proj_forward(head, blk_final + i * D_MODEL, logits + i * VOCAB, 1);

                /* Loss */
                batch_loss += trix_cross_entropy_loss(logits, targets, SEQ_LEN, VOCAB);

                /* Accuracy */
                int* preds = calloc(SEQ_LEN, sizeof(int));
                trix_argmax(preds, logits, SEQ_LEN, VOCAB);
                for (int i = 0; i < SEQ_LEN; i++)
                    if (preds[i] == targets[i]) epoch_correct++;
                epoch_total += SEQ_LEN;
                free(preds);

                /* Backward: head */
                trix_cross_entropy_grad(d_logits, logits, targets, SEQ_LEN, VOCAB);
                for (int i = 0; i < SEQ_LEN; i++)
                    trix_routed_proj_backward(head, blk_final + i * D_MODEL,
                                              d_logits + i * VOCAB,
                                              d_head_in + i * D_MODEL, 1);

                /* Backward: blocks (reverse order, using saved activations) */
                memcpy(d_layer, d_head_in, SEQ_LEN * D_MODEL * sizeof(float));

                for (int l = N_LAYERS - 1; l >= 0; l--) {
                    float* dx = calloc(SEQ_LEN * D_MODEL, sizeof(float));
                    trix_routed_block_backward(blocks[l], saved[l], d_layer, dx, SEQ_LEN);
                    memcpy(d_layer, dx, SEQ_LEN * D_MODEL * sizeof(float));
                    free(dx);
                }

                for (int l = 0; l <= N_LAYERS; l++) free(saved[l]);

                /* Backward: embedding */
                for (int i = 0; i < SEQ_LEN; i++)
                    trix_routed_proj_backward(emb, x_onehot + i * VOCAB,
                                              d_layer + i * D_MODEL,
                                              d_emb + i * VOCAB, 1);
            }

            epoch_loss += batch_loss / BATCH;

            /* Clip + update */
            for (int l = 0; l < N_LAYERS; l++)
                trix_routed_block_clip_grad_norm(blocks[l], GRAD_CLIP);

            trix_routed_proj_adamw_step(emb, LR, 0.9f, 0.999f, 1e-8f, 0.01f);
            for (int l = 0; l < N_LAYERS; l++)
                trix_routed_block_adamw_step(blocks[l], LR, 0.9f, 0.999f, 1e-8f, 0.01f);
            trix_routed_proj_adamw_step(head, LR, 0.9f, 0.999f, 1e-8f, 0.01f);
        }

        float avg_loss = epoch_loss / n_batches_per_epoch;
        float acc = (float)epoch_correct / (float)epoch_total * 100.0f;
        float ppl = expf(avg_loss);

        if (ep % 5 == 0 || ep == EPOCHS - 1) {
            struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);

            /* Print mixer routing diversity for block 0 */
            TrixRoutedMixer* rm = blocks[0]->mixer;
            int unique_patterns = 0;
            if (rm->route && rm->seq_cap >= SEQ_LEN) {
                /* Count unique routing patterns in last forward */
                int T = NUM_TILES;
                for (int i = 0; i < SEQ_LEN; i++) {
                    int is_unique = 1;
                    for (int j = 0; j < i; j++) {
                        int same = 1;
                        for (int t = 0; t < T; t++)
                            if (rm->route[i * T + t] != rm->route[j * T + t]) { same = 0; break; }
                        if (same) { is_unique = 0; break; }
                    }
                    unique_patterns += is_unique;
                }
            }

            /* Validation loss */
            float val_loss = 0; int val_steps = 0;
            if (val_len > SEQ_LEN + 1) {
                int n_val = (val_len - SEQ_LEN - 1);
                int val_samples = n_val < 8 ? n_val : 8;
                for (int v = 0; v < val_samples; v++) {
                    int vs = v * (n_val / val_samples);
                    for (int i = 0; i < SEQ_LEN; i++)
                        char_to_vec(x_onehot + i * VOCAB, (int)(unsigned char)val_corpus[vs + i]);
                    for (int i = 0; i < SEQ_LEN; i++)
                        targets[i] = (int)(unsigned char)val_corpus[vs + i + 1];
                    for (int i = 0; i < SEQ_LEN; i++)
                        trix_routed_proj_forward(emb, x_onehot + i * VOCAB, emb_out + i * D_MODEL, 1);
                    float* vbuf_a = malloc(SEQ_LEN * D_MODEL * sizeof(float));
                    float* vbuf_b = malloc(SEQ_LEN * D_MODEL * sizeof(float));
                    memcpy(vbuf_a, emb_out, SEQ_LEN * D_MODEL * sizeof(float));
                    for (int l = 0; l < N_LAYERS; l++) {
                        trix_routed_block_forward_causal(blocks[l], vbuf_a, vbuf_b, SEQ_LEN);
                        float* tmp = vbuf_a; vbuf_a = vbuf_b; vbuf_b = tmp;
                    }
                    for (int i = 0; i < SEQ_LEN; i++)
                        trix_routed_proj_forward(head, vbuf_a + i * D_MODEL, logits + i * VOCAB, 1);
                    val_loss += trix_cross_entropy_loss(logits, targets, SEQ_LEN, VOCAB);
                    val_steps++;
                    free(vbuf_a); free(vbuf_b);
                }
                val_loss /= val_steps;
            }

            printf("  ep %2d: loss=%.3f val=%.3f ppl=%.1f acc=%.1f%% routes=%d/%d (%.0fs)\n",
                   ep, avg_loss, val_loss, ppl, acc, unique_patterns, SEQ_LEN, elapsed);
        }
    }

    /* ── Generate sample ── */
    printf("\n── SAMPLE (greedy, 80 chars) ──\n  ");
    char seed_ch = 't';
    float* gen_emb = calloc(D_MODEL, sizeof(float));
    float* gen_buf_a = calloc(SEQ_LEN * D_MODEL, sizeof(float));
    float* gen_buf_b = calloc(SEQ_LEN * D_MODEL, sizeof(float));
    float* gen_logits = calloc(VOCAB, sizeof(float));
    float* gen_onehot = calloc(VOCAB, sizeof(float));

    /* Fill context with seed char */
    float* context = calloc(SEQ_LEN * D_MODEL, sizeof(float));
    char_to_vec(gen_onehot, (int)(unsigned char)seed_ch);
    trix_routed_proj_forward(emb, gen_onehot, gen_emb, 1);
    for (int i = 0; i < SEQ_LEN; i++)
        memcpy(context + i * D_MODEL, gen_emb, D_MODEL * sizeof(float));

    for (int g = 0; g < 80; g++) {
        /* Forward through blocks */
        memcpy(gen_buf_a, context, SEQ_LEN * D_MODEL * sizeof(float));
        for (int l = 0; l < N_LAYERS; l++) {
            trix_routed_block_forward(blocks[l], gen_buf_a, gen_buf_b, SEQ_LEN);
            float* tmp = gen_buf_a; gen_buf_a = gen_buf_b; gen_buf_b = tmp;
        }

        /* Head on last position */
        trix_routed_proj_forward(head, gen_buf_a + (SEQ_LEN - 1) * D_MODEL, gen_logits, 1);

        /* Greedy decode */
        int best = 0; float bv = gen_logits[0];
        for (int c = 1; c < VOCAB; c++)
            if (gen_logits[c] > bv) { bv = gen_logits[c]; best = c; }

        char ch = (best >= 32 && best < 127) ? (char)best : '?';
        putchar(ch);

        /* Shift context left, append new embedding */
        char_to_vec(gen_onehot, best);
        trix_routed_proj_forward(emb, gen_onehot, gen_emb, 1);
        memmove(context, context + D_MODEL, (SEQ_LEN - 1) * D_MODEL * sizeof(float));
        memcpy(context + (SEQ_LEN - 1) * D_MODEL, gen_emb, D_MODEL * sizeof(float));
    }
    printf("\n");

    /* Cleanup */
    free(corpus); free(x_onehot); free(emb_out); free(layer_buf);
    free(logits); free(d_logits); free(d_head_in);
    free(d_layer); free(d_emb); free(targets);
    free(gen_emb); free(gen_buf_a); free(gen_buf_b);
    free(gen_logits); free(gen_onehot); free(context);
    trix_routed_proj_destroy(emb);
    for (int l = 0; l < N_LAYERS; l++) trix_routed_block_destroy(blocks[l]);
    trix_routed_proj_destroy(head);

    return 0;
}
