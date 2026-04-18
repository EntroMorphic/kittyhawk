---
date: 2026-04-17
phase: RAW
topic: Per-image normalization and what the representation sweep reveals
---

# Normalization findings — RAW

---

The representation sweep just told us something I wasn't
expecting and I need to sit with it before reacting.

```
Raw pixels:           36.4%
Zero-mean:            38.6%
Normalized:           42.8%
Simple HOG:           17.4%
Gradient magnitude:   27.8%
Grayscale:            31.4%
```

The spatial features (gradients, HOG) that I spent hours trying
to build are WORSE than raw pixels. The thing that helps is the
simplest possible transform: subtract the mean and divide by the
standard deviation. Per image. No spatial structure. No block
encoding. No multi-channel fusion.

Why does this work? A cat in a dark room and a cat in sunlight
have very different raw pixel values but the same PATTERN after
normalization. The normalization removes illumination variation
— the single largest source of within-class variance in natural
images. Once illumination is factored out, the remaining pixel
pattern carries more class signal per pixel.

And why do gradients HURT? Because gradients are sensitive to
TEXTURE, not SHAPE. A grassy field and a leafy tree have similar
gradient distributions but belong to different contexts (deer
vs bird). Raw pixels after normalization preserve the global
color-shape pattern that gradients destroy.

But wait. SSTT uses gradients and block encodings and reaches
53%. Why do gradients work for SSTT but not for us?

Because SSTT doesn't use gradients as the SOLE representation.
SSTT uses THREE channels: intensity, horizontal gradient, and
vertical gradient. Each channel contributes to the ternary
signature. The intensity channel carries the global pattern
(like our raw pixels). The gradient channels add edge
information ON TOP of intensity. The combination is richer
than either alone.

And SSTT quantizes BEFORE computing nearest neighbors. The
ternary quantization collapses the continuous gradient values
into {-1, 0, +1}, which acts as a form of normalization —
strong gradients map to ±1, weak gradients map to 0. The
quantization itself is a noise filter.

So SSTT's advantage is not "gradients are good" — it's
"intensity + gradients + ternary quantization" is a richer
representation than raw pixels alone. The ternary quantization
is doing the heavy lifting by normalizing each feature to a
three-state symbol.

Now, what does 42.8% from per-image normalization mean for
our architecture?

It means the pixel-space k-NN ceiling is NOT 36%. It's at
least 42.8%. Our random projection LSH was matching the 36%
raw-pixel ceiling perfectly. If we normalize first, the
ceiling lifts and the LSH should track upward.

The implementation is trivial: before projection, normalize
each image vector. In MTFP integer arithmetic:
1. Compute mean = sum(pixels) / N_pixels
2. Subtract mean from each pixel
3. Compute variance = sum((pixel - mean)^2) / N_pixels
4. Divide each pixel by sqrt(variance)

Step 4 requires integer square root — available in MTFP via
iterative approximation, or approximated by right-shifting
the variance exponent.

Actually, step 4 is the hard part. Division by sqrt(variance)
in integer arithmetic without float is non-trivial. But maybe
we don't need full normalization. The sweep showed zero-mean
alone gives +2.2pp. The additional +4.2pp from variance
scaling might not be necessary — or might be approximable.

Let me think about what zero-mean does to the ternary
projection. The projection is w⋅x = Σ w_i × x_i. If x is
zero-mean, then Σ w_i × (x_i - μ) = Σ w_i × x_i - μ Σ w_i.
At density 0.33, the projection weights are {-1, 0, +1}
with equal probability. So Σ w_i ≈ 0 (equal +1 and -1
weights sum to near zero). Therefore w⋅(x - μ) ≈ w⋅x for
balanced ternary projections.

WAIT. If balanced ternary projections already approximately
subtract the mean (because Σ w_i ≈ 0), then zero-mean
preprocessing should have MINIMAL effect on our LSH. Yet
the raw pixel k-NN showed +2.2pp from zero-mean and +6.4pp
from normalization.

The difference is that k-NN uses L1 distance on ALL pixels,
not ternary-projected Hamming distance on 16 trits. Zero-mean
helps L1 because L1 treats every pixel equally — a bright
image is far from a dark image in L1 even if the pattern is
identical. Ternary projection with balanced weights already
cancels the brightness bias.

So zero-mean preprocessing might NOT help our LSH. The ternary
projection might already be doing implicit zero-mean. The
thing that would help is VARIANCE normalization — making each
image's contrast uniform.

Let me think about what variance normalization does to the
projection. If two images have the same pattern but different
contrast, their projections w⋅x are proportional (scaling all
x by a constant scales w⋅x by the same constant). The τ
threshold is calibrated on the training distribution, so a
low-contrast image whose projections are all small (below τ)
would produce an all-zero signature, while a high-contrast
image with the same pattern would produce non-zero trits.

THAT'S THE PROBLEM. Low-contrast images produce all-zero or
near-zero signatures regardless of their content, because the
projection values fall below τ. High-contrast images produce
rich signatures. The τ calibration is set for the average
contrast level, so low-contrast images are under-represented
in signature space.

Variance normalization fixes this: every image has the same
contrast after normalization, so the projection values have
the same distribution regardless of the original brightness
or contrast. The τ threshold works equally well for every
image.

This is a HUGE insight. The problem isn't the projection
DIRECTION (random vs structured). The problem is that
LOW-CONTRAST IMAGES PRODUCE DEGENERATE SIGNATURES. The
structural zero is being imposed by the τ threshold on
images that happen to have low contrast, not by design.

On MNIST, this isn't a problem — handwritten digits have
high contrast (white strokes on black background). Every
image produces rich signatures. On CIFAR-10, natural images
vary enormously in contrast — a foggy scene, a night photo,
a bright outdoor scene all have different contrast levels.
The low-contrast images produce degenerate (all-zero)
signatures that collide with each other regardless of class.

The fix is per-image variance normalization before
projection. Every image enters the lattice with the same
contrast. The structural zero (W_f[hidden]=0) means "this
direction doesn't discriminate" — not "this image is too
faint to measure."

Can this be done in MTFP integer arithmetic?

Mean: sum / N = integer division. Fine.
Variance: sum of (x - mean)^2 / N. Fine in int64.
Stddev: sqrt(variance). This requires integer square root.
Division by stddev: x / sqrt(var). This requires integer
division by a variable denominator.

Integer square root: Newton's method converges in ~5
iterations for 32-bit values. Approximation: find the
highest bit of the variance, shift right by half that many
bits. Exact sqrt isn't needed — an approximation that's
within 2× is sufficient for normalization.

Actually, do we even need division? What if we just multiply
by 1/stddev? Precompute 1/stddev as a fixed-point multiplier.
Or: scale all pixels by a common reference (e.g., stddev =
target_stddev) using integer multiply-shift.

Or the simplest approach: don't normalize pixel VALUES.
Instead, adjust τ PER IMAGE to compensate for contrast.
Instead of a fixed τ for all images, set τ_q = τ × stddev_q
/ stddev_avg. This scales the threshold proportionally to
the image's contrast. Low-contrast images get a lower τ
(more trits activate). High-contrast images get a higher τ.

Per-image τ adjustment is one integer multiply per query.
No modification to the training signatures — they use the
standard τ. Only the test query encoding changes.

But wait — if training and test use different τ values, the
signatures aren't comparable. The τ must be consistent
between training and test for the same image pattern to
produce the same signature.

So: normalize the PIXEL VALUES at load time (both train and
test), then use standard τ. This is a one-time preprocessing
step applied to ds.x_train and ds.x_test before any
projection.

The normalization is:
1. For each image i: compute mean_i and var_i from its pixels.
2. Compute a global target scale (e.g., mean of all stddevs).
3. Scale: pixel[j] = (pixel[j] - mean_i) × target / stddev_i

This requires one integer multiply and one integer divide per
pixel. In MTFP, the multiply is direct and the divide uses
the integer-division convention (truncation).

Can this be a library function? glyph_dataset_normalize(ds)?
Added alongside glyph_dataset_deskew(ds)? Same pattern —
per-image transform applied at load time, idempotent.

Yes. And it should be the DEFAULT for CIFAR-10, the way
deskew is the default for MNIST. Different datasets need
different preprocessing. MNIST needs deskew (aligns stroke
axis). CIFAR-10 needs contrast normalization (equalizes
illumination).

This is the missing step. Not better projections. Not better
resolvers. Not GSH or specialists. CONTRAST NORMALIZATION
so that every image enters the trit lattice with equal
signal strength.

And it's fully routing-native: integer arithmetic on MTFP
values, applied once at load time. The projections, signatures,
bucket index, multi-probe, and resolvers are all unchanged.
