#!/bin/bash
# Args: $1 = label_prefix (v14opt or v13)
prefix=$1
HARNESS=build/gesh/bitnet_harness
DATA=data/bitnet_b158_2b4t.bin
OUTDIR=/tmp/inference_test
mkdir -p "$OUTDIR"

run() {
    local label=$1; shift
    local desc=$1; shift
    # Warmup once (drop the first call's cold cache)
    "$HARNESS" "$DATA" "$@" >/dev/null 2>&1
    local start=$(date +%s%N)
    local out=$("$HARNESS" "$DATA" "$@" 2>&1)
    local end=$(date +%s%N)
    local elapsed_ms=$(( (end - start) / 1000000 ))
    local gentoks=$(echo "$out" | grep "generated tokens" | sed 's/.*=//; s/^ *//')
    local argmax=$(echo "$out" | grep "argmax over full vocab" | sed 's/.*=//; s/^ *//')
    local x_line=$(echo "$out" | grep "post-final-norm x" | sed 's/.*=//; s/^ *//')
    local logits=$(echo "$out" | grep "logits\[0..3\]" | sed 's/.*=//; s/^ *//')
    printf "%-20s | argmax=%s | gen='%s' | x=%s | logits=%s | time=%dms\n" \
        "$label" "$argmax" "$gentoks" "$x_line" "$logits" "$elapsed_ms"
    {
        echo "label=$label"
        echo "desc=$desc"
        echo "argmax=$argmax"
        echo "gen=$gentoks"
        echo "x=$x_line"
        echo "logits=$logits"
        echo "time_ms=$elapsed_ms"
    } > "$OUTDIR/${prefix}_${label}.txt"
}

# --- The battery ---
run capital_france "Capital of France →" --prompt-tokens 1,1841,8085,341,9099,1735 --gen 16
run bos_only      "BOS → 32 tokens"     --token 1 --positions 1 --gen 32
run token_5337    "tok 5337 → 16"       --token 5337 --positions 1 --gen 16
run short_a       "1,464,2944,18 → 16"  --prompt-tokens 1,464,2944,18 --gen 16
run short_b       "1,791,28036,9100 → 16" --prompt-tokens 1,791,28036,9100 --gen 16
run medium        "11-tok prompt → 16"  --prompt-tokens 1,1841,8085,341,9099,1735,374,279,3838,520,2728 --gen 16
run pos32_no_gen  "p=32 (bench shape)"  --token 1 --positions 32 --gen 0
