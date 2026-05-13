"""Decode c_dump_v3 prompts by reading layer-0 x_norm_input back to tokens.

Per the dump format in bitnet_harness.c, each .layerN.bin file contains
x_norm_input (BITNET_HIDDEN_SIZE int8/int16 values). At layer 0, this
is the post-norm embedding of the input token. Compare each position's
embedding against the model's embedding matrix to identify the token.

If c_dump_v3 was generated from natural-language prompts, the recovered
tokens should decode to coherent text. If from gibberish (like the
5-prompt harness battery was), they'll match the same gibberish.
"""
import numpy as np
import os
import struct

DUMP_DIR = "data/c_dump_v3"
PROMPTS  = ["long64", "long_a", "long_b", "long_c", "long_d"]


def read_actv2_first_field(path: str):
    """Read just the x_norm_input field from an ACTV2 dump."""
    with open(path, "rb") as f:
        magic = f.read(5)
        assert magic == b"ACTV2", f"Bad magic in {path}: {magic!r}"
        f.read(3)  # pad
        li, hidden, intermediate, kv_proj = struct.unpack("<iiii", f.read(16))
        # x_norm_input: hidden int16 values (m4t_mtfp_t = int16)
        x_norm_input = np.frombuffer(f.read(hidden * 2), dtype=np.int16)
    return x_norm_input.copy()


def main():
    # We don't have the model embedding matrix loaded; instead, we save
    # the x_norm_input vectors for each (prompt, position) at layer 0 and
    # then try to match them against a fresh harness run with a candidate
    # prompt. That tells us position-by-position what token reproduces the
    # same x_norm_input.

    # First pass: just print the dimensionality and a fingerprint of the
    # x_norm_input at each (prompt, position 0) so we can see whether
    # different prompts START with the same token (e.g., BOS=128000 vs
    # BOS=1).
    print(f"{'prompt':<10} {'pos':>4} {'hidden':>6} {'x_norm[0..7]'}")
    print("-" * 80)
    for prompt in PROMPTS:
        for pos in [0, 1, 5, 10, 30, 63]:
            path = os.path.join(DUMP_DIR, f"{prompt}.pos{pos}.layer0.bin")
            if not os.path.exists(path):
                continue
            x = read_actv2_first_field(path)
            sample = x[:8].tolist()
            print(f"{prompt:<10} {pos:>4} {len(x):>6} {sample}")
        print()

    # Compare BOS token by checking pos0 of each prompt — they should
    # share the same first-token x_norm_input if all prompts start with
    # the same BOS. Output cosine similarity / L1 between pos0 vectors.
    pos0 = {}
    for prompt in PROMPTS:
        path = os.path.join(DUMP_DIR, f"{prompt}.pos0.layer0.bin")
        if os.path.exists(path):
            pos0[prompt] = read_actv2_first_field(path).astype(np.float64)

    print(f"\nPairwise L1 distance of layer-0 x_norm_input at pos=0:")
    print(f"  (=0 means same BOS token; >0 means different first tokens)")
    print(f"  {'  ':>10}", end="")
    for p in pos0:
        print(f" {p:>10}", end="")
    print()
    for p1 in pos0:
        print(f"  {p1:>10}", end="")
        for p2 in pos0:
            d = float(np.abs(pos0[p1] - pos0[p2]).sum())
            print(f" {d:>10.0f}", end="")
        print()


if __name__ == "__main__":
    main()
