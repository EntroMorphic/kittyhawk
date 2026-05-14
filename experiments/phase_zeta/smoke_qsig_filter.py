"""Smoke test: BITNET_KV_EVICT_QSIG_FILTER mode.

Required:
  K=0  → bit-identical to qsigdist on a representative prompt.

Useful diagnostic:
  K=1, K=2 produce DIFFERENT outputs from qsigdist (eviction picks
  a different victim when the protected slot would have been the
  one qsigdist evicted).
"""
import os
import re
import subprocess
import sys

HARNESS = "build/gesh/bitnet_harness"
DATA = "data/bitnet_b158_2b4t.bin"
WINDOW = 16
GEN = 24
TAU = "5000"

# A medium-length prompt that triggers eviction
sys.path.insert(0, "experiments/phase_zeta")
from n50_battery import PROMPTS as OLD
from n100_battery_incremental import NEW_PROMPTS as NEW
ALL = {}; ALL.update(OLD); ALL.update(NEW)
LABEL = "long_storm"  # 9-prompt category, has eviction events
PROMPT = ALL[LABEL]

def run(mode, extra_env=None):
    env = os.environ.copy()
    env.update({
        "BITNET_KV_EVICT_MODE": mode,
        "BITNET_KV_WINDOW": str(WINDOW),
        "BITNET_ATTN_FIXED_TAU": TAU,
    })
    if extra_env:
        env.update(extra_env)
    cmd = [HARNESS, DATA, "--prompt-tokens", PROMPT, "--gen", str(GEN)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    out = proc.stdout + proc.stderr
    m = re.search(r"generated tokens\s*=\s*([\d\s\-]+)", out)
    return [int(t) for t in m.group(1).strip().split()] if m else None

print(f"Prompt: {LABEL}")
toks_qsig = run("qsigdist")
toks_k0   = run("qsig_filter", {"BITNET_KV_EVICT_KK_PROTECT_K": "0"})
toks_k1   = run("qsig_filter", {"BITNET_KV_EVICT_KK_PROTECT_K": "1"})
toks_k2   = run("qsig_filter", {"BITNET_KV_EVICT_KK_PROTECT_K": "2"})
toks_k4   = run("qsig_filter", {"BITNET_KV_EVICT_KK_PROTECT_K": "4"})
toks_k15  = run("qsig_filter", {"BITNET_KV_EVICT_KK_PROTECT_K": "15"})  # near full window=16, fallback to qsigdist

print(f"\nqsigdist: {toks_qsig[:12]}...")
print(f"K=0:      {toks_k0[:12]}...")
print(f"K=1:      {toks_k1[:12]}...")
print(f"K=2:      {toks_k2[:12]}...")
print(f"K=4:      {toks_k4[:12]}...")
print(f"K=15:     {toks_k15[:12]}...")

print()
print(f"K=0 ≡ qsigdist:   {toks_qsig == toks_k0}")
print(f"K=1 ≢ qsigdist:   {toks_qsig != toks_k1}")
print(f"K=2 ≢ qsigdist:   {toks_qsig != toks_k2}")
print(f"K=4 ≢ qsigdist:   {toks_qsig != toks_k4}")
print(f"K=15 ≡ qsigdist:  {toks_qsig == toks_k15}  (window=16 → fallback)")

if toks_qsig != toks_k0:
    print("\nSMOKE FAIL: K=0 should be bit-identical to qsigdist")
    sys.exit(1)
print("\nSmoke PASS")
