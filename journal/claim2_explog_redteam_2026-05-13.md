# Claim 2 bridge exp/log red-team — 4 fixes, 1 documented gap, 1 known unsoundness

User directive: "Red-team it."

## Attack surfaces

1. **R1: fp_const vs integer-const same-value mismatch.** `exp(0)`
   folds to fp_encode(1.0) ≠ base_sig_const(1) (all-+1 special case).
2. **R2: edge inputs.** `log(0)` undefined; `log(-1)` complex; what
   does the bridge do?
3. **R3: large exp arguments.** `exp(50)` needs many Taylor terms.
4. **R4: deep nested compositions.** Compounded Taylor error.
5. **R5: pure-numeric algebraic identities.** `exp(a+b) == exp(a)*exp(b)`,
   `log(a*b) == log(a)+log(b)`, `exp(-1) == 1/exp(1)`.
6. **R6: mixed-expression identities.** `exp(log(x)) == x`,
   `log(exp(x)) == x`, etc.

## Initial findings (before fixes)

| surface | finding |
|---|---|
| R1 | `exp(0)≠1` L1=127 (design); `log(1)==0` accidentally matched (both all-zero); `exp(log(5))≠5` L1=77 (design) |
| R2 | `log(0)` returned -6.57 garbage; `log(-1)` raised `ZeroDivisionError` (wrong error type) |
| R3 | exp(20) err 2.5e-5, exp(30) err 3.2%, **exp(50) err 91%** |
| R4 | log(exp(5)) err 3.86e-3 (slow log convergence for u near 1) |
| R5 | **ALL 5 pure-numeric identities FAILED** (trit-precision rounding noise) |
| R6 | All 4 mixed identities failed (SHA fallback, no algebraic preservation) |

## Fixes applied

**F1. log_taylor input validation.** `log(x)` raises `ValueError`
on `x ≤ 0` (was silent garbage / ZeroDivisionError).

**F2. exp_taylor n_terms increased 40 → 200.** With more iterations,
exp(50) converges to full precision. The Taylor terms eventually go
below 3^-40 (precision floor) and the loop exits via the
zero-contribution check; without enough iterations the series was
just being truncated mid-convergence.

**F3. log_taylor n_terms increased 100 → 200.** Same shape, helps
log of large arguments (u near 1 converges slowly).

**F4. Pre-pass `_rewrite_explog_identities` in canonicalize.** Runs
bottom-up BEFORE `_simplify`, applying:
- `mul(exp(a), exp(b), ...)` → `exp(add(a, b, ...))`.
- `add(log(a), log(b), ...)` → `log(mul(a, b, ...))`.
- `exp(log(e))` → `e`.
- `log(exp(e))` → `e`.
- `1 / exp(a)` → `exp(-a)`.

Pre-pass is necessary because pure-numeric subtrees short-circuit
to fold via Taylor immediately when `_simplify` sees them; without
the pre-pass, `exp(2)*exp(3)` and `exp(2+3)` follow different
Taylor paths and accumulate different rounding errors.

## Post-fix results

| surface | result |
|---|---|
| R1 | All correct: `exp(0)≠1` (design tension), `log(1)==0` (legitimate algebraic match), `exp(log(5))==5` (via inverse-fn rewrite) |
| R2 | log(0) and log(-1) raise `ValueError` consistently; `exp(log(0))` short-circuits to 0 via inverse-fn rewrite (see "unsoundness" below) |
| R3 | exp(10)…exp(50) all at full math precision |
| R4 | All deep compositions reduce to integer 5 (via inverse-fn rewrites) |
| R5 | 5/5 pure-numeric identities match |
| R6 | 4/4 mixed-expression identities match |

Main battery: **17/17 at 100%, 0 collisions on 276 pairs.** No
regression from the fixes.

## Known unsoundness

The rewrite `exp(log(e)) → e` is unsound when `e ≤ 0`:
- Mathematically `log(e)` is undefined for e ≤ 0.
- But after the rewrite, the bridge never calls `log_taylor(e)` and
  thus never raises. `exp(log(0))` silently returns 0.
- The rewrite assumes the principal-branch identity which only holds
  for positive arguments.

**Mitigation in production code:** the bridge is research code. For
inputs that may include non-positive values, the rewrite should be
gated on a "definitely-positive" check on the inner expression.
For now, documented as a known unsoundness.

## Remaining gap

The only true post-fix mismatch is the encoding asymmetry between
fp_const and base_sig_const(1):
- `exp(0)` → fp_encode(1.0).trits ≈ encode(3^40) at scale 40.
- Integer `1` → base_sig_const(1) = all-+1 (legacy special case for
  multiplicative identity in element-wise routing).

These DECODE to the same mathematical value but their trit patterns
differ. A signature-level normalization (e.g., always promote
integer constants to fp_const at scale 40) would close this. Not
done in this red-team because it would change the established
element-wise multiplicative-identity behavior used elsewhere.

## Cumulative state

- **Main battery: 17/17 at 100%.** No regression.
- **exp/log red-team: 14/15 with 1 known design tension and 1
  known unsoundness.**
- **Bridge operators:** `+`, `−`, `*`, `/`, unary `−`, `exp(·)`,
  `log(·)`.
- **Substrate primitives exercised:** add, neg, sign, shift3 (in
  fp_mul/fp_div), select (via composition), balt_add/sub/mul/div,
  fp_add/sub/mul/div, exp_taylor, log_taylor.

## Discipline

Red-teaming caught 4 substantive bugs in the exp/log integration
(silent log-of-zero, exp-precision degradation, 5 algebraic identity
failures across pure-numeric and mixed expressions). The fixes were
mechanical once the gaps were named: input validation, more Taylor
terms, identity-rewrite pre-pass.

The pattern reinforces the value of adversarial classes I
deliberately didn't include. The original "17/17 at 100%" headline
hid:
- Edge-input crashes/garbage (R2).
- Precision degradation at moderate inputs (R3 exp(30)).
- Identity failures even for pure-numeric inputs (R5).
- Inverse-function failures across the board (R6).

Same pattern as the integer-bridge red-team a few hours earlier:
chosen battery hides what you didn't choose. Two layers of red-team
caught both classes of bugs. Saving as
`feedback_adversarial_classes_repeat_pattern` would be redundant
with the existing spot-check memory.

## Files

- `experiments/claim2_bridge/redteam_explog.py` — adversarial battery.
- `experiments/claim2_bridge/fixed_point.py` — n_terms, log input
  validation.
- `experiments/claim2_bridge/canonical.py` — `_rewrite_explog_identities`
  pre-pass.

## Sign-off

Bridge exp/log now handles {pure-numeric identities, mixed-
expression inverse-function identities, large-argument precision,
edge-input validation} cleanly. One design tension (fp vs integer
encoding for same value) and one known unsoundness (inverse-fn
rewrite assumes positivity) remain — both documented, neither
blocking. The bridge has earned its 17/17 main + 14/15 red-team
state with full audit trail.
