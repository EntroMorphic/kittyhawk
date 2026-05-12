# Cold-eye audit — done first-hand, not by subagent

User directive: "Audit the codebase once more with a cold eye. Do not
use subagents this time. I want you to have first-hand knowledge."

Three things this finds that the four-agent audit missed. One memory-
to-code drift not previously caught. Two pieces of scoping that the
substrate-claim arc has been informally papering over.

## Eight passes I ran

1. **`#if !M4T_HAS_NEON` in production** → one instance
   (`m4t_ternary_rowskip.c:17`) and it's a compile-time `#error`
   enforcing the rule, not violating it. Clean.

2. **TODO/FIXME/XXX/experimental in production code** → four hits in
   `bitnet_harness.c`, three of which are real documentation drift
   (see "Findings" §1 below).

3. **Edge cases in L1 estimator** → `cell_pmf_from_data` silently
   renormalizes when input contains non-ternary values (line 80,
   `experiments/phase_beta/m1_l1_estimator.py`). All callers pass
   `threshold_extract` output (guaranteed ternary), so latent
   defense-in-depth only.

4. **Binary structures / bit-parity gates in the architecture** →
   none. The only `bf16/fp32` mention in `bitnet_harness.c` is in a
   comment explaining a precision tradeoff in `bitnet_embed`, not
   gating logic.

5. **Phase ε `trial_eviction` shuffle correctness** → the shuffle
   uses `np.random.default_rng(RNG_SEED + 1)` freshly inside each
   call. Same permutation patterns across calls. Functionally
   correct for the ε-5 control (destroys per-K-vector correlations)
   but a minor quirk — different K_caches still get the same row
   permutations.

6. **`cell_pmf_from_data` callers** → all 7 usages pass
   `threshold_extract` output. No contract violations in practice.

7. **Production τ vs experiments τ** → real divergence (see §2).

8. **Production eviction default** → real scoping issue (see §3).

## Findings — direct, not via subagent

### §1. Code-comment documentation drift (not just README)

`gesh/bitnet/bitnet_harness.c`:
- **Line 358**: "Cycle 2 sparse-attention helpers (experimental;
  not on production path unless BITNET_ATTN_MODE != dense)."
- **Line 1209**: "── SPARSE PATH (Cycle 2 experimental) ─────"

These contradict the explicit comment at **line 1139-1140**:
"Production-eligible per the 'no scalar in production' foundational
rule (2026-05-12)."

Audit-2 (subagent) flagged the README drift only. The same drift
exists in code comments. A single sweep across `bitnet_harness.c` to
unify the framing would close it. Same severity as the README issue
(LOW; functional code is correct).

### §2. Production Q-sig τ is per-query-adaptive; experiments use fixed 5000

Production reads `g_attn_fixed_tau` (default 0). When zero, Q-sig τ
is computed per-query via `bitnet_routed_pick_tau` — the
**1/3-quantile of |Q| values** — targeting ~33% zeros in the
Q-signature (`bitnet_harness.c:399-410`). When non-zero (or when
sigdist eviction is on), Q-sig τ is fixed at 5000.

K-sig τ is **always fixed at 5000** when sigdist eviction is active
(see line 209).

**Experiments code** (`load_k_signatures.py`) hardcodes
`THRESHOLD_TAU = 5000` for BOTH Q-sig and K-sig generation. So:

- Production default Q-sig: per-Q adaptive, ~67% nonzero.
- Experiments Q-sig: fixed τ=5000, ~62% nonzero (per Phase γ measure).
- Both K-sigs: fixed τ=5000, ~62% nonzero.

The Q-side differs. The L1>Hamming direction-of-effect should hold
across both regimes (both are in the "majority nonzero" range), but
absolute recall@k numbers from Phase δ/ε are for the fixed-τ
regime — not what production would produce.

This is the same shape as the td28 misalignment (Python doesn't
match production semantics), at a different layer. Adding to
`feedback_verify_production_semantics`.

### §3. Production DEFAULT does not exercise substrate eviction

`g_kv_evict_mode` default is **`BITNET_KV_EVICT_NONE`**
(`bitnet_harness.c:164`). `g_attn_mode` default is **`BITNET_ATTN_DENSE`**
(verified via grep — the sparse path branches only when
`g_attn_mode != BITNET_ATTN_DENSE && g_attn_k < seq_k`).

So:
- README's "~92% strict pass rate on 24-prompt battery" is for
  default settings: **dense attention, no eviction**. Substrate
  signatures don't fire on that path.
- The substrate-claim arc (Phase α/β/γ/δ/ε) measured properties of
  substrate signatures under modes that are **opt-in only**:
  `BITNET_KV_EVICT_MODE=sigdist` or `BITNET_ATTN_MODE=routed`.

**Implication for the arc's framing**: "L1 is in production"
(td28's discovery) is technically true — the L1 metric IS what
`m4t_popcount_dist` computes in the production code paths. But
those code paths are **off by default**. The default 24-prompt
battery quality (~92%) has zero substrate-eviction content.

The substrate-distinctive value sits in optional code that hasn't
been load-bearing for the production default's quality so far. The
properties measured by Phase δ/ε are for opt-in modes that the user
can enable; they aren't measured against the default production
quality.

This isn't wrong — it's just scope. Worth being explicit so we
don't drift into claiming the substrate's eviction work helps the
~92% number when it doesn't (because it's not on by default).

## What I verified is actually fine

- The substrate kernel's `m4t_popcount_dist` correctly computes L1
  path-graph distance on packed 2-bit codes. Already covered in
  td28.
- The production NEON paths have no scalar fallbacks (verified
  directly by grep).
- The L1 estimator math (cell PMF, convolution, CDF) is correct
  modulo the cold-eye edge case (defense-in-depth on input
  validation).
- `_scalar_ref` functions appear in non-test code only as test
  oracles, which the project rule permits.

## What still isn't audited (honest gap list)

- The `m4t_mtfp.c` MTFP arithmetic kernels (2400+ lines) — spot-
  checked, not exhaustively read.
- The `gesh/src/` consumer code outside the BitNet harness.
- The 100+ older journal files. Many may have claims that recent
  corrections contradict; I sampled only the recent.
- Test coverage analysis: I have not verified that the existing
  unit tests cover the code paths actually exercised in production.
- The CMake build configuration.
- The pre-rebuild material in `01MAY26_archived/` (gitignored).

## Pattern: subagents miss what memory-aware audits catch

The four-agent audit produced one false-critical finding (Audit-3's
`* 2` bug), one disclaimed-rule citation (Audit-4's "no primitive
without demand"), and otherwise clean reports. This first-hand
audit found three additional drift items the agents missed:

- The sparse-path comments at lines 358 and 1209 (Audit-2 caught
  only the README, not the code-comment instances).
- Production-vs-experiments τ divergence.
- Production-default mode question.

The common shape: each agent had a single slice and read only that
slice. Cross-slice findings (Python τ doesn't match production τ;
production default doesn't exercise the substrate-eviction code)
require seeing multiple parts at once. First-hand audit catches the
cross-slice gaps; agents catch within-slice details.

For future audits: combine first-hand + agent. Agents for breadth
within slices, first-hand for cross-slice integration and memory-
contradiction checks. The td28 discovery and the "no primitive
without demand" miss both needed cross-slice memory + code reading
— neither was the kind of finding an agent could surface.

## What to do with these findings

Two material, two LOW.

**Material (production scoping clarity):**

1. **Update the substrate-claim journals** to be explicit that the
   findings apply to opt-in eviction modes, not the production
   default. A single header on each journal pointing at this
   discovery would prevent future readers from inferring "the 92%
   battery uses substrate eviction" (it doesn't).

2. **Resolve the Q-sig τ mismatch**: either rerun Phase ε with
   per-Q-adaptive τ (matching production sigdist mode's input
   distribution to the experiments) OR document the gap. Magnitudes
   may shift; directions likely don't.

**Low (cheap doc cleanup):**

3. Sweep `bitnet_harness.c` for "experimental" / "not on production"
   comments that contradict line 1139-1140's "Production-eligible"
   note. Make them consistent.

4. Add an assertion or runtime check in `cell_pmf_from_data` that
   rejects non-ternary input. Currently silently renormalizes.
