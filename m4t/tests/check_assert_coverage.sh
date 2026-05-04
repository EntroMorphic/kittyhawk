#!/usr/bin/env bash
# check_assert_coverage.sh — V4-residual-3 #3 closure.
#
# Verifies test_m4t_assert_live.c's cases[] covers every substrate source
# file that has an assert() call site. Fails (exit 1) if either:
#
#   (a) a substrate .c gains an assert() but the test doesn't include a
#       case for it (silent coverage loss), or
#   (b) the test's cases[] references a source file that has no asserts
#       (stale case — still passes the meta-test trivially because no
#       precondition can be violated to surface it).
#
# Usage: check_assert_coverage.sh <substrate_src_dir> <test_file>
#
# Run as a ctest entry (m4t_assert_coverage); see m4t/CMakeLists.txt.

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <substrate_src_dir> <test_file>" >&2
    exit 2
fi

SRC_DIR="$1"
TEST_FILE="$2"

if [ ! -d "$SRC_DIR" ]; then
    echo "error: substrate src dir not found: $SRC_DIR" >&2
    exit 2
fi
if [ ! -f "$TEST_FILE" ]; then
    echo "error: test file not found: $TEST_FILE" >&2
    exit 2
fi

# List-A: substrate .c files with at least one runtime assert( call.
#
# Pattern: line-start (modulo leading whitespace) followed by `assert(`.
# This excludes false positives from:
#   - block-comment continuations (line begins with `*`)
#   - line comments (line begins with `//`)
#   - string literals containing `assert(` (literal can't span line start)
#   - mid-line non-call use (e.g., `_Static_assert(...)` — different token,
#     and doesn't appear at line start anyway)
#   - mentions in conditional code like `if (x) assert(y)` (mid-line)
#
# Audit confirmed (at the time of writing): every `assert(` call site in
# m4t/src/*.c is at line start modulo whitespace. If a future call site
# is written mid-line and not caught by this, the missing case will
# surface as a coverage failure, not a silent pass — visible failure mode.
files_with_asserts() {
    for f in "$SRC_DIR"/*.c; do
        if grep -qE '^[[:space:]]*assert\(' "$f"; then
            basename "$f"
        fi
    done | sort
}

# List-B: source files named in cases[].source_file string literals.
# Pattern: lines of the form { "m4t_xxx.c", ... }. Character class
# includes digits because some sources have them (m4t_mtfp4.c).
files_in_cases() {
    grep -oE '"m4t_[a-z0-9_]+\.c"' "$TEST_FILE" | tr -d '"' | sort -u
}

# Safety net: warn (not fail) if a file has any assert( token that the
# strict regex would miss. Catches the inverse failure mode of the
# strict regex: a mid-line assert (e.g., `if (x) assert(y);`) wouldn't
# be at line start, so the file would be excluded from List A and
# silently lack a case. Warning surfaces this; doesn't break the gate.
warn_mid_line_asserts() {
    for f in "$SRC_DIR"/*.c; do
        # _Static_assert is compile-time, not relevant here.
        loose=$(grep -cE '(^|[^A-Za-z_])assert\(' "$f" || true)
        static_assert=$(grep -cE '_Static_assert\(' "$f" || true)
        loose_runtime=$((loose - static_assert))
        strict=$(grep -cE '^[[:space:]]*assert\(' "$f" || true)
        if [ "$loose_runtime" -gt "$strict" ]; then
            echo "WARNING: $(basename "$f") has $((loose_runtime - strict)) assert( token(s) not at line start;" >&2
            echo "         coverage check may miss them. Restructure to line-start, or amend this script." >&2
        fi
    done
}
warn_mid_line_asserts

actual="$(files_with_asserts)"
covered="$(files_in_cases)"

if [ "$actual" = "$covered" ]; then
    n=$(echo "$actual" | wc -l | tr -d ' ')
    echo "PASS: $n substrate source(s) with asserts; all covered by cases[]"
    exit 0
fi

echo "FAIL: parameterized assert-live test coverage mismatch" >&2
echo "" >&2
echo "Substrate sources with asserts (truth):" >&2
echo "$actual" | sed 's/^/  /' >&2
echo "" >&2
echo "Sources covered by cases[] (test):" >&2
echo "$covered" | sed 's/^/  /' >&2
echo "" >&2
echo "Sources missing from cases[]:" >&2
comm -23 <(echo "$actual") <(echo "$covered") | sed 's/^/  - /' >&2 || true
echo "" >&2
echo "Sources in cases[] but no longer have asserts:" >&2
comm -13 <(echo "$actual") <(echo "$covered") | sed 's/^/  - /' >&2 || true
exit 1
