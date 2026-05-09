#!/usr/bin/env python3
"""
Post-RMSNorm-fix end-to-end inference battery.

Per prompt: tokenize correctly (BOS=128000), greedy-generate N tokens with
the substrate harness AND with HF bf16, compare token IDs, decoded text,
and report qualitative match.
"""
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "microsoft/bitnet-b1.58-2B-4T-bf16"
HARNESS = "/Users/aaronjosserand-austin/Projects/glyph/build/gesh/bitnet_harness"
WEIGHTS = "/Users/aaronjosserand-austin/Projects/glyph/data/bitnet_b158_2b4t.bin"
N_GEN = 30  # short enough to keep CPU-HF tractable, long enough to see drift

PROMPTS = [
    ("factual_capital",      "What is the capital of France?"),
    ("factual_who",          "Who wrote Hamlet?"),
    ("definition_photosynth", "Photosynthesis is"),
    ("continuation_once",    "Once upon a time"),
    ("math_simple",          "12 plus 7 equals"),
    ("reasoning_color",      "The color of the sky on a clear day is"),
    ("reflective",           "Hypothetically, might reflective recursion be a function of cognition?"),
    ("translate_hello",      "Translate to French: Hello, how are you?"),
]


def tokenize(tok, text):
    return tok.encode(text, add_special_tokens=True)


def run_substrate(prompt_ids, n_gen):
    args = [
        HARNESS, WEIGHTS,
        "--prompt-tokens", ",".join(str(t) for t in prompt_ids),
        "--gen", str(n_gen),
    ]
    t0 = time.time()
    out = subprocess.run(args, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0
    combined = out.stdout + out.stderr
    gen_line = next(
        (ln for ln in combined.splitlines() if "generated tokens" in ln),
        None,
    )
    if not gen_line:
        return None, elapsed, combined
    raw = gen_line.split("=", 1)[1].strip()
    return [int(t) for t in raw.split()], elapsed, None


def run_hf(model, prompt_ids, n_gen):
    inp = torch.tensor([prompt_ids])
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=n_gen, do_sample=False)
    elapsed = time.time() - t0
    return out[0].tolist()[len(prompt_ids):], elapsed


def coherence_label(text):
    # Heuristic: long-run repeating substring suggests degenerate loop
    if len(text) < 20:
        return "short"
    for win in (8, 12, 20):
        if win * 3 > len(text):
            continue
        for i in range(len(text) - win * 3):
            chunk = text[i:i + win]
            if not chunk.strip():
                continue
            if text.count(chunk) >= 4:
                return "loop"
    return "ok"


def main():
    print("Loading HF model (CPU bf16) ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16).eval()
    print(f"\n{'name':22s} {'sub_match':>10s} {'len':>4s} {'sub_t(s)':>8s} {'hf_t(s)':>8s} {'sub_coh':>8s}\n")

    overall_match = []
    rows = []
    for name, prompt in PROMPTS:
        ids = tokenize(tok, prompt)
        sub_gen, sub_t, sub_err = run_substrate(ids, N_GEN)
        if sub_err:
            print(f"{name:22s} ERROR: {sub_err[:100]}")
            continue
        hf_gen, hf_t = run_hf(model, ids, N_GEN)
        n = min(len(sub_gen), len(hf_gen))
        match = sum(1 for i in range(n) if sub_gen[i] == hf_gen[i])
        sub_text = tok.decode(sub_gen)
        hf_text  = tok.decode(hf_gen)
        coh = coherence_label(sub_text)
        overall_match.append(match / n if n else 0.0)
        rows.append((name, prompt, sub_gen, hf_gen, sub_text, hf_text, match, n, sub_t, hf_t, coh))
        print(f"{name:22s} {match}/{n:<5d}    {n:>4d} {sub_t:>8.2f} {hf_t:>8.2f} {coh:>8s}")

    print(f"\nMean token agreement: {sum(overall_match)/len(overall_match):.2%}")

    out_dir = "/Users/aaronjosserand-austin/Projects/glyph/journal/post_rmsnorm_fix_battery_2026-05-09"
    with open(f"{out_dir}/results.txt", "w") as f:
        for name, prompt, sub, hf, st, ht, m, n, st_t, ht_t, coh in rows:
            f.write(f"=== {name} ({coh}, match {m}/{n}, sub {st_t:.2f}s, hf {ht_t:.2f}s) ===\n")
            f.write(f"PROMPT: {prompt}\n")
            f.write(f"SUB:    {st}\n")
            f.write(f"HF:     {ht}\n\n")
    print(f"\nDetailed results → {out_dir}/results.txt")


if __name__ == "__main__":
    main()
