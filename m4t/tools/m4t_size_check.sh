#!/bin/bash
# m4t_size_check.sh — link-time budget enforcement for M4T opcode bodies.
# Parses __text section size from libm4t.a and fails if over budget.
# Wired into CMake as a post-link custom command.
#
# Usage: m4t_size_check.sh <path-to-libm4t.a> <budget-bytes>

set -euo pipefail

LIB="${1:?Usage: m4t_size_check.sh <libm4t.a> <budget>}"
BUDGET="${2:?Usage: m4t_size_check.sh <libm4t.a> <budget>}"

TOTAL=$(size -m "$LIB" 2>/dev/null \
    | grep '__TEXT.*__text' \
    | awk '{sum += $NF} END {print sum+0}')

if [ -z "$TOTAL" ] || [ "$TOTAL" -eq 0 ]; then
    TOTAL=$(size "$LIB" 2>/dev/null \
        | tail -n +2 \
        | awk '{sum += $1} END {print sum+0}')
fi

echo "M4T size check: __text = ${TOTAL} bytes, budget = ${BUDGET} bytes"

if [ "$TOTAL" -gt "$BUDGET" ]; then
    echo "FAIL: M4T opcode bodies exceed L1i budget (${TOTAL} > ${BUDGET})"
    exit 1
fi

echo "PASS: ${TOTAL} / ${BUDGET} bytes ($(( TOTAL * 100 / BUDGET ))% used)"
