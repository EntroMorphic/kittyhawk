"""Phase ζ atomic — per-step argmax-margin telemetry.

Hypothesis under test: a 35× per-Q-head L2-error advantage of sigdist
over random (Phase ε oracle) collapses to ~0 generation-quality
advantage (Phase ζ harness) because the argmax decision is robust to
the magnitude of perturbation eviction introduces. The runner-up
margin (top1_acc − top2_acc) is the natural ruler.

For each (prompt, window, mode), we run the harness with
BITNET_LOG_PERSTEP=1 and parse the per-step lines:
  pos, gen, top1_tok, top1_acc, top2_tok, top2_acc, margin

We compare modes per-position:
  1. Did the argmax flip vs no_evict?
  2. If so, by how much (margin sign change)?
  3. Did the no_evict-chosen token's logit RANK shift, even if argmax
     didn't flip? (We don't have rank directly, but we can check
     "no_evict's top1 token still ≥ this mode's top1 token in this mode's
     accumulators" by re-running with a specific token's logit asked.)

For atomic-probe purposes the (1)+(2) pair is enough: if margin is
typically much larger than the cross-mode top1_acc spread, the argmax
is in a stability basin and the per-Q-head L2 advantage is invisible
at the harness level.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from typing import Optional

HARNESS = "build/gesh/bitnet_harness"
DATA    = "data/bitnet_b158_2b4t.bin"
OUTDIR  = "experiments/phase_zeta/results/perstep"

PROMPTS = {
    "medium_11":      "1,1841,8085,341,9099,1735,374,279,3838,520,2728",
    "bos_only":       "",
}
WINDOWS = [16]
GEN_N   = 24

PERSTEP_RE = re.compile(
    r"\[perstep\] pos=(\d+) gen=(\d+) top1=(\d+) top1_acc=(-?\d+) "
    r"top2=(\d+) top2_acc=(-?\d+) margin=(-?\d+)"
)


def mode_env(mode: str, window: int) -> dict:
    if mode == "no_evict":
        return {"BITNET_KV_EVICT_MODE": "none"}
    if mode == "fifo":
        return {"BITNET_KV_EVICT_MODE": "fifo", "BITNET_KV_WINDOW": str(window)}
    if mode == "sigdist":
        return {"BITNET_KV_EVICT_MODE": "sigdist", "BITNET_KV_WINDOW": str(window),
                "BITNET_ATTN_FIXED_TAU": "5000"}
    if mode == "random":
        return {"BITNET_KV_EVICT_MODE": "random", "BITNET_KV_WINDOW": str(window),
                "BITNET_KV_EVICT_SEED": "42"}
    raise ValueError(mode)


def run_one(label: str, prompt: str, mode: str, window: int) -> list[dict]:
    env = os.environ.copy()
    env.update(mode_env(mode, window))
    env["BITNET_LOG_PERSTEP"] = "1"
    if prompt:
        cmd = [HARNESS, DATA, "--prompt-tokens", prompt, "--gen", str(GEN_N)]
    else:
        cmd = [HARNESS, DATA, "--token", "1", "--positions", "1", "--gen", str(GEN_N)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"{label}_w{window}_{mode}.log"), "w") as f:
        f.write(out)
    steps = []
    for line in out.splitlines():
        m = PERSTEP_RE.search(line)
        if m:
            steps.append({
                "pos": int(m.group(1)),
                "gen": int(m.group(2)),
                "top1": int(m.group(3)),
                "top1_acc": int(m.group(4)),
                "top2": int(m.group(5)),
                "top2_acc": int(m.group(6)),
                "margin": int(m.group(7)),
            })
    return steps


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    modes = ["no_evict", "fifo", "random", "sigdist"]
    all_data = {}
    for window in WINDOWS:
        for label, prompt in PROMPTS.items():
            print(f"\n=== {label}, window={window} ===")
            by_mode = {}
            for mode in modes:
                t0 = time.time()
                steps = run_one(label, prompt, mode, window)
                el = time.time() - t0
                print(f"  {mode:<9} {el:>6.1f}s  n_steps={len(steps)}")
                by_mode[mode] = steps
            all_data[f"{label}_w{window}"] = by_mode

            # Compare per-step vs no_evict
            base = by_mode["no_evict"]
            n = len(base)
            print(f"\n  per-step margins (no_evict baseline) and divergences:")
            print(f"  {'gen':>3} {'noevict':<10} {'ne_tok':>6} {'margin(ne)':>14}"
                  f"  {'fifo':>6} {'random':>6} {'sigdist':>6}"
                  f"  {'d_top1_acc_fifo':>16} {'d_top1_acc_rand':>16} {'d_top1_acc_sig':>16}")
            argmax_flip = {m: 0 for m in ("fifo", "random", "sigdist")}
            close_call_count = 0  # gen-steps where ne margin < 5 * cross-mode acc spread
            for g in range(n):
                ne = base[g]
                line = f"  {g:>3d}              {ne['top1']:>6d}  {ne['margin']:>14d}"
                row_diffs = {}
                for m in ("fifo", "random", "sigdist"):
                    if g >= len(by_mode[m]):
                        line += f"  {'?':>6}"
                        continue
                    s = by_mode[m][g]
                    flip = "X" if s["top1"] != ne["top1"] else "-"
                    if flip == "X":
                        argmax_flip[m] += 1
                    line += f"  {s['top1']:>5d}{flip}"
                    # Diff: |this_mode's top1_acc - no_evict's top1_acc on same token|
                    # Since we don't have arbitrary-token logits, use:
                    #   d = |s.top1_acc - ne.top1_acc| (raw top1 magnitude shift —
                    # noisy but captures hidden-state perturbation magnitude)
                    row_diffs[m] = s["top1_acc"] - ne["top1_acc"]
                for m in ("fifo", "random", "sigdist"):
                    d = row_diffs.get(m, 0)
                    line += f"  {d:>16d}"
                print(line)

            print(f"\n  argmax-flip counts (gen=0..{n-1}):")
            for m in ("fifo", "random", "sigdist"):
                print(f"    {m:<9} {argmax_flip[m]}/{n}")

    out_path = os.path.join(OUTDIR, "perstep_data.json")
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nRaw data: {out_path}")


if __name__ == "__main__":
    main()
