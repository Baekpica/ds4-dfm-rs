#ifndef DS4_LING3VL_ROPE_H
#define DS4_LING3VL_ROPE_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

enum { L3V_ROPE = 64, L3V_CONTEXT_ORIG = 131072,
       L3V_YARN_MAX_FACTOR = 2,
       L3V_CONTEXT = L3V_CONTEXT_ORIG * L3V_YARN_MAX_FACTOR };

/* Static per-session scaling: cached keys must keep the same frequencies
 * for their entire lifetime, including short prefixes in a 256K session. */
static inline uint32_t ling3vl_rope_factor(uint32_t ctx) {
    if (!ctx || ctx > L3V_CONTEXT) { return 0u; }
    return ctx <= L3V_CONTEXT_ORIG ? 1u : L3V_YARN_MAX_FACTOR;
}

/* Official Ling override: theta 6e6, 64 rotary dimensions, beta 32/1.
 * YaRN scales only the rotated Q/K tails, leaving MLA's latent path intact.
 * Equations: transformers v4.57.1 modeling_rope_utils._compute_yarn_parameters. */
static inline bool ling3vl_rope_table(float *inv, float *attn, uint32_t ctx) {
    const uint32_t factor = ling3vl_rope_factor(ctx);
    if (!inv || !attn || !factor) { return false; }
    const double theta = 6000000.0;
    const double beta_fast = 32.0, beta_slow = 1.0;
    const double tau = 2.0 * acos(-1.0);
    const double low = fmax(0.0, floor(L3V_ROPE *
        log(L3V_CONTEXT_ORIG / (beta_fast * tau)) / (2.0 * log(theta))));
    const double high = fmin(L3V_ROPE - 1.0, ceil(L3V_ROPE *
        log(L3V_CONTEXT_ORIG / (beta_slow * tau)) / (2.0 * log(theta))));
    *attn = factor == 1u ? 1.0f : (float)(1.0 + 0.1 * log((double)factor));
    for (uint32_t j = 0; j < L3V_ROPE / 2u; j++) {
        /* Preserve the original factor-one table bit for bit. */
        const float base = (float)pow(theta, -2.0 * j / L3V_ROPE);
        const float ramp = (float)fmin(1.0, fmax(0.0, (j - low) / (high - low)));
        inv[j] = factor == 1u ? base :
            base * (1.0f - ramp) + base / (float)factor * ramp;
    }
    return true;
}

#endif
