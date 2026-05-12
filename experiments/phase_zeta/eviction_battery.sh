#!/usr/bin/env bash
# Phase ζ — harness-level generation-quality benchmark for substrate eviction.
#
# The Phase ε attention-output L2 result said L1-substrate eviction
# reduces attention error by 38-62% vs Hamming-substrate eviction.
# But the production DEFAULT is no-eviction. The cold-eye audit
# (td28_cold_eye_audit) flagged that the substrate-claim arc measured
# properties of an opt-in mode, not the default's behavior.
#
# This script measures THE TERRITORY: does enabling substrate eviction
# (BITNET_KV_EVICT_MODE=sigdist) on real inference produce
# coherent / different / degraded generation vs no-eviction baseline?
#
# Three modes per prompt:
#   1. no-evict  — production default. Full K-cache retained.
#   2. fifo      — simplest eviction (drop oldest). Null baseline.
#   3. sigdist   — substrate L1 distance eviction. The claim under test.
#
# Compare:
#   - generated token sequences (exact match? divergence point?)
#   - per-position post-final-norm x (proxy for hidden state drift)
#   - argmax over LM head (next-token prediction)

set -e
HARNESS="${HARNESS:-build/gesh/bitnet_harness}"
DATA="${DATA:-data/bitnet_b158_2b4t.bin}"
OUTDIR="${OUTDIR:-experiments/phase_zeta/results}"
WINDOW="${WINDOW:-8}"      # cache size cap — chosen small enough to force eviction
GEN_N="${GEN_N:-16}"
mkdir -p "$OUTDIR"

if [ ! -x "$HARNESS" ]; then
  echo "Build the harness first: cmake -S . -B build && cmake --build build --target bitnet_harness" >&2
  exit 1
fi

# Battery — diverse prompt lengths, drawn from journal/v14_inference_battery
declare -A PROMPTS=(
  [capital_france]="1,1841,8085,341,9099,1735"
  [short_a]="1,464,2944,18"
  [short_b]="1,791,28036,9100"
  [medium_11]="1,1841,8085,341,9099,1735,374,279,3838,520,2728"
  [bos_only]=""   # single-token prompt, generates from BOS
)

run_one() {
  local label="$1"
  local prompt="$2"
  local mode="$3"
  local logfile="$OUTDIR/${label}_${mode}.log"

  local env_str=""
  case "$mode" in
    no_evict)
      env_str="BITNET_KV_EVICT_MODE=none"
      ;;
    fifo)
      env_str="BITNET_KV_EVICT_MODE=fifo BITNET_KV_WINDOW=$WINDOW"
      ;;
    sigdist)
      env_str="BITNET_KV_EVICT_MODE=sigdist BITNET_KV_WINDOW=$WINDOW BITNET_ATTN_FIXED_TAU=5000"
      ;;
  esac

  local cmd
  if [ -z "$prompt" ]; then
    cmd="env $env_str $HARNESS $DATA --token 1 --positions 1 --gen $GEN_N"
  else
    cmd="env $env_str $HARNESS $DATA --prompt-tokens $prompt --gen $GEN_N"
  fi

  eval "$cmd" >"$logfile" 2>&1 || true
}

extract_tokens() {
  # Output format: 'generated tokens             = 25 330 7063 1'
  local logfile="$1"
  grep "generated tokens" "$logfile" | head -1 | sed 's/.*= //' | tr -d '\n'
}

extract_logits() {
  local logfile="$1"
  grep -oE "logits\[0\.\.3\] *= [^ ]+ [^ ]+ [^ ]+ [^ ]+" "$logfile" | head -1
}

extract_x() {
  local logfile="$1"
  grep -oE "post-final-norm x\[0\.\.3\] *= [^ ]+ [^ ]+ [^ ]+ [^ ]+" "$logfile" | head -1
}

# Header
printf "%-18s | %-8s | %-50s | %-30s | %-30s\n" \
  "label" "mode" "generated tokens" "post-final-norm x[0..3]" "logits[0..3]"
printf -- "%.s-" {1..160}; echo

for label in "${!PROMPTS[@]}"; do
  prompt="${PROMPTS[$label]}"
  for mode in no_evict fifo sigdist; do
    run_one "$label" "$prompt" "$mode"
    logfile="$OUTDIR/${label}_${mode}.log"
    tokens=$(extract_tokens "$logfile" || echo "?")
    x_line=$(extract_x "$logfile" | sed 's/.*= //' || echo "?")
    logits=$(extract_logits "$logfile" | sed 's/.*= //' || echo "?")
    printf "%-18s | %-8s | %-50s | %-30s | %-30s\n" \
      "$label" "$mode" "${tokens:0:50}" "$x_line" "$logits"
  done
  echo
done
