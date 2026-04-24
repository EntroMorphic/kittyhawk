#include "requantize.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int cmp_float(const void* a, const void* b) {
    float x = *(const float*)a, y = *(const float*)b;
    return (x < y) - (x > y);  /* descending */
}

int requantize_hysteresis(
    int8_t* W, const float* W_latent, int n,
    double density, double hysteresis)
{
    /* τ = density-th percentile of |W_latent|. Density is the fraction
     * of |values| below τ (those land in the zero band). */
    float* abs_vals = malloc((size_t)n * sizeof(float));
    if (!abs_vals) return 0;
    for (int i = 0; i < n; i++) {
        abs_vals[i] = fabsf(W_latent[i]);
    }
    qsort(abs_vals, (size_t)n, sizeof(float), cmp_float);
    /* After descending sort, element at index (n * (1 - density)) is the
     * density-th percentile (0 < density < 1). */
    int idx = (int)((double)n * (1.0 - density));
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    float tau = abs_vals[idx];
    free(abs_vals);

    double h = (hysteresis < 0) ? 0 : hysteresis;
    float tau_in  = (float)(tau * (1.0 + h));   /* zero → ±1 requires |w| > tau_in */
    float tau_out = (float)(tau * (1.0 - h));   /* ±1 → zero requires |w| < tau_out */

    int flips = 0;
    for (int i = 0; i < n; i++) {
        float w = W_latent[i];
        int8_t cur = W[i];
        int8_t next = cur;
        if (cur == 0) {
            if (w >  tau_in) next = +1;
            else if (w < -tau_in) next = -1;
        } else if (cur == +1) {
            if (w < -tau_in)        next = -1;   /* full sign flip needs wider band */
            else if (w <  tau_out)  next = 0;
        } else { /* cur == -1 */
            if (w >  tau_in)        next = +1;
            else if (w > -tau_out)  next = 0;
        }
        if (next != cur) flips++;
        W[i] = next;
    }
    return flips;
}
