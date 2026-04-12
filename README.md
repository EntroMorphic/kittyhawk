# GLYPH

A NEON-optimized C library for Apple M4.

## Numerical System — Non-Negotiable

**Glyph uses only three numeric types, all rooted in base-3:**

1. **Ternary** — single trits `{-1, 0, +1}`, packed 4 per byte (2-bit codes).
2. **Multi-Trit** — fixed-width trit words (e.g. 10-trit MTFP cells), used for weights, activations, and routing state.
3. **Multi-Trit Floating Point (MTFP)** — balanced ternary fixed-point with scale `3^10 = 59049` (real value = cell / 59049). Stored in an `int32` container, but treated as an opaque MTFP cell — never as a binary integer.

**Banned everywhere in glyph source:**

- `float`, `double`, `float16`, `bfloat16`, or any IEEE-754 type.
- `int8_t`, `int16_t`, `uint8_t`/`uint16_t` **as numeric quantization types** (byte/word buffers for packed trits are fine — they are containers, not numbers).
- STE shadow weights, float gradient accumulators, float optimizer state.
- Softmax over floats, GELU via `erff`, any transcendental called on a float.

If a computation needs a "real number," it goes through MTFP. Period. This is the core liberation of the project: once you give up binary FP, every operation becomes a trit-level add/subtract/skip, and the hardware maps directly to `XOR`+`VCNT`+`SDOT` on NEON.

See `reference-code/` for the trix-z source this project forks from. That code is retained verbatim for reference; it contains float paths that **do not** belong in glyph.
