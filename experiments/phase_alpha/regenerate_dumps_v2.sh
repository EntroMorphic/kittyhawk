#!/bin/bash
# Regenerate data/c_dump_v2 — five diverse-prompt K-cache dumps used by
# Phase α remediation (journal/td27_phase_alpha_remediation_2026-05-12.md).
# Total output: ~150MB across 30 layers × 33 prompt-positions × 12 sites.

set -e
HARNESS="build/gesh/bitnet_harness"
DATA="data/bitnet_b158_2b4t.bin"
OUT="data/c_dump_v2"

if [ ! -x "$HARNESS" ]; then
  echo "Build the harness first: cd build && make bitnet_harness" >&2
  exit 1
fi
if [ ! -f "$DATA" ]; then
  echo "Missing weights blob $DATA" >&2
  exit 1
fi

mkdir -p "$OUT"
echo "Generating Phase α remediation corpus into $OUT …"

"$HARNESS" "$DATA" --prompt-tokens 1,1841,8085,341,9099,1735 --dump "$OUT/p1" >/dev/null 2>&1 &
"$HARNESS" "$DATA" --prompt-tokens 1,464,2944,18 --dump "$OUT/p2" >/dev/null 2>&1 &
"$HARNESS" "$DATA" --prompt-tokens 1,791,28036,9100 --dump "$OUT/p3" >/dev/null 2>&1 &
"$HARNESS" "$DATA" --prompt-tokens 1,1841,8085,341,9099,1735,374,279,3838,520,2728 --dump "$OUT/p4" >/dev/null 2>&1 &
"$HARNESS" "$DATA" --prompt-tokens 1,5337,7912,4192,11823,2731,9999,3145 --dump "$OUT/p5" >/dev/null 2>&1 &
wait

echo "Done. $(ls "$OUT" | wc -l | tr -d ' ') files."
