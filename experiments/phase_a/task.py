"""Phase A task: sequence copy.

Input:  [BOS, x_1, ..., x_n, SEP, PAD, ..., PAD]
Target: [PAD, ..., PAD, x_1, ..., x_n] (shifted to align with positions after SEP)

The model predicts at positions where target != PAD; loss ignored elsewhere.

n is sampled uniformly from {4, ..., 12} per example.
x_i is sampled uniformly from {SYM_0, ..., SYM_31}.
Fixed sequence length 32 (pad to 32).

Vocab assignment:
  0 = PAD
  1 = BOS
  2 = SEP
  3..34 = symbols 0..31
  35..63 = reserve (unused in Phase A)
"""
import torch

PAD = 0
BOS = 1
SEP = 2
SYM_BASE = 3
N_SYMBOLS = 32
VOCAB = 64
SEQ_LEN = 32
# Phase A fixed-N=8 (default) OR Phase A.1 variable-length via VARIABLE_N=1 env.
# Variable-length requires rotary position encoding (use_rope=True in model).
import os as _os
VARIABLE_N = _os.environ.get("VARIABLE_N", "0") == "1"
if VARIABLE_N:
    MIN_N = 4
    MAX_N = 12
else:
    N = 8
    MIN_N = N
    MAX_N = N


def make_batch(batch_size: int, device: torch.device, rng: torch.Generator):
    """Return (input_ids, target_ids), each (batch_size, SEQ_LEN), dtype long.

    target_ids has PAD at positions where loss should be ignored.
    """
    input_ids = torch.full((batch_size, SEQ_LEN), PAD, dtype=torch.long, device=device)
    target_ids = torch.full((batch_size, SEQ_LEN), PAD, dtype=torch.long, device=device)

    for b in range(batch_size):
        if VARIABLE_N:
            n = int(torch.randint(MIN_N, MAX_N + 1, (1,), generator=rng, device=rng.device).item())
        else:
            n = N
        symbols = torch.randint(0, N_SYMBOLS, (n,), generator=rng, device=rng.device) + SYM_BASE

        # Standard teacher-forcing layout:
        #   Input:  [BOS, x_1, ..., x_n, SEP, x_1, ..., x_n, PAD...]
        #   Target: shift left by 1 → predict input[t+1] at position t.
        #
        # The model is asked to predict at position (1+n) onward, where the
        # answer is the copied symbols. Predictions before that (BOS→x_1,
        # x_1→x_2, ...) get PAD targets so loss ignores them. This isolates
        # the copy task from the (also-trivial) memorization of the input
        # half.
        input_ids[b, 0] = BOS
        input_ids[b, 1:1+n] = symbols.to(device)
        input_ids[b, 1+n] = SEP
        end = 1 + n + n  # last filled position in input (x_n at index 1+n+n)
        if end + 1 <= SEQ_LEN:
            input_ids[b, 1+n+1:end+1] = symbols.to(device)
        else:
            # truncate — shouldn't happen given MAX_N=12 and SEQ_LEN=32
            keep = SEQ_LEN - (1+n+1)
            input_ids[b, 1+n+1:1+n+1+keep] = symbols[:keep].to(device)

        # Target: predict the copy. target[1+n+i] = symbols[i] for i = 0..n-1
        # (At position 1+n, input is SEP → predict x_1.
        #  At position 1+n+i for i ≥ 1, input is x_i → predict x_(i+1).
        #  At position 1+n+n-1, input is x_(n-1) → predict x_n.)
        for i in range(n):
            tgt_pos = 1 + n + i
            if tgt_pos >= SEQ_LEN:
                break
            target_ids[b, tgt_pos] = symbols[i].to(device)

    return input_ids, target_ids


def exact_sequence_accuracy(pred_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
    """Fraction of sequences where every non-PAD target position is correctly predicted.

    pred_ids:   (batch, SEQ_LEN) long, argmax over vocab
    target_ids: (batch, SEQ_LEN) long, PAD where ignored
    """
    mask = (target_ids != PAD)
    # A sequence passes if all positions with mask=True predict correctly
    per_pos_correct = (pred_ids == target_ids) | (~mask)  # ignore non-target positions
    per_seq_pass = per_pos_correct.all(dim=-1)
    return per_seq_pass.float().mean().item()


if __name__ == "__main__":
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)
    x, y = make_batch(4, torch.device("cpu"), rng)
    print("input:", x[0].tolist())
    print("target:", y[0].tolist())
    print("(non-PAD positions):", (y[0] != PAD).nonzero().flatten().tolist())
