"""Phase A training loop.

Pre-registered protocol per journal/td27_7_phase_a_2026-05-11.md:
  - AdamW (lr 3e-4, betas (0.9, 0.95), wd 0.01)
  - Cosine LR schedule to 1e-5 over 10000 steps
  - Gradient clip 1.0
  - Batch size 32
  - Eval every 100 steps on 1024 held-out sequences
  - Stop at first eval ≥ 95% exact-sequence accuracy
  - Step cap 10000 (or 10× Variant A's pass-step if running Variant B)
"""
import argparse
import json
import math
import os
import sys
import time
import torch
import torch.nn.functional as F

# Allow running from experiments/phase_a or repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import TinyGPT, count_params
from task import make_batch, exact_sequence_accuracy, PAD, VOCAB, SEQ_LEN


def evaluate(model, device, gen, n_seqs=1024, batch_size=128):
    model.eval()
    correct_seqs = 0
    total_seqs = 0
    with torch.no_grad():
        for _ in range(n_seqs // batch_size):
            x, y = make_batch(batch_size, device, gen)
            logits = model(x)  # (B, T, V)
            pred = logits.argmax(dim=-1)
            mask = (y != PAD)
            per_pos_correct = (pred == y) | (~mask)
            per_seq_pass = per_pos_correct.all(dim=-1)
            correct_seqs += per_seq_pass.sum().item()
            total_seqs += per_seq_pass.numel()
    model.train()
    return correct_seqs / total_seqs


def cosine_lr(step, max_steps, lr_init, lr_min):
    if step >= max_steps:
        return lr_min
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + math.cos(math.pi * step / max_steps))


def train(variant: str, seed: int, max_steps: int, log_path: str,
          lr_init=3e-4, lr_min=1e-5, lr_cosine_horizon=10000,
          batch_size=32, eval_every=100, target_acc=0.95,
          device_name="cpu", verbose=True, use_rope=False):
    device = torch.device(device_name)
    torch.manual_seed(seed)
    data_gen = torch.Generator(device=device)
    data_gen.manual_seed(seed * 1000 + 7)
    eval_gen = torch.Generator(device=device)
    eval_gen.manual_seed(seed * 1000 + 99)

    model = TinyGPT(variant, use_rope=use_rope).to(device)
    n_params = count_params(model)

    opt = torch.optim.AdamW(model.parameters(), lr=lr_init, betas=(0.9, 0.95), weight_decay=0.01)

    log = {
        "variant": variant,
        "seed": seed,
        "n_params": n_params,
        "config": {
            "lr_init": lr_init, "lr_min": lr_min, "lr_cosine_horizon": lr_cosine_horizon,
            "batch_size": batch_size, "max_steps": max_steps, "target_acc": target_acc,
            "device": device_name,
        },
        "events": [],  # list of {step, train_loss, eval_acc, lr, wall_s}
        "pass_step": None,  # first step where eval ≥ target_acc
        "final_acc": None,
    }
    t0 = time.time()

    for step in range(1, max_steps + 1):
        lr = cosine_lr(step, lr_cosine_horizon, lr_init, lr_min)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = make_batch(batch_size, device, data_gen)
        logits = model(x)
        # Cross-entropy with PAD ignored
        loss = F.cross_entropy(
            logits.view(-1, VOCAB), y.view(-1), ignore_index=PAD
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_every == 0:
            acc = evaluate(model, device, eval_gen)
            wall = time.time() - t0
            ev = {"step": step, "train_loss": loss.item(), "eval_acc": acc, "lr": lr, "wall_s": wall}
            log["events"].append(ev)
            if verbose:
                print(f"[{variant} seed={seed}] step={step} loss={loss.item():.4f} acc={acc:.3f} lr={lr:.2e} wall={wall:.1f}s")
            if acc >= target_acc and log["pass_step"] is None:
                log["pass_step"] = step
                log["final_acc"] = acc
                if verbose:
                    print(f"[{variant} seed={seed}] PASS at step {step} (acc {acc:.3f})")
                break

    if log["pass_step"] is None:
        log["final_acc"] = log["events"][-1]["eval_acc"] if log["events"] else 0.0

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["dense", "substrate", "random"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--log", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--use-rope", action="store_true", help="use rotary position encoding (for variable-length)")
    args = ap.parse_args()

    log = train(args.variant, args.seed, args.max_steps, args.log, device_name=args.device, use_rope=args.use_rope)
    print(f"\nFinal: pass_step={log['pass_step']} final_acc={log['final_acc']:.3f}")
