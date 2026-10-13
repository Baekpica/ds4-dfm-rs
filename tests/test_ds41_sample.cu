/* V4.1 device sampling kernel gate (tests/test_ds41_sample, the engine's
 * tests/cuda_sample_selftest.c, 2026-09-28): CUDA device only, no model.
 *
 * The gate is distribution-level, not byte-level: the sampling route has no
 * "which token should it emit" golden, it has "each token's frequency must
 * equal the target distribution".  Four passes, one recipe set (temperature /
 * top_k / top_p / min_p):
 *   (1) full-distribution samples: draw N times (positions 0..N-1 as the RNG
 *       counter); outside the kept set the frequency must be exactly 0 (the
 *       truncation is exact); inside it, the top-32 tokens' frequencies must
 *       be within 5 sigma of the renormalized truncated probability (sigma =
 *       sqrt(p(1-p)/N)), and the total-variation distance must be under a
 *       bound.
 *   (2) the speculative marginal: with a fixed draft d, emitted = accept ? d
 *       : the residual sample; its frequency must equal p too (that sentence
 *       IS the correctness of the rejection sampling); the accept rate must be
 *       about p(d).
 *   (3) determinism: same parameters, same position, drawn again -> all four
 *       ints per row bit-identical (the graph and direct routes run this same
 *       kernel, so this is "one request, two dispatch ways, one token
 *       stream").
 *   (4) distribution draft: the draft is drawn from q with stream 1 (the same
 *       kernel the engine's draft tower uses), fed as the next row's input,
 *       and verified against qlogits = q; the marginal must still be p and
 *       the accept rate about sum min(p,q).
 * What happens on error: a sample outside the kept set = a wrong threshold
 * key (radix select / key transform); a top-32 entry over 5 sigma = a Gumbel
 * key or RNG bias; a speculative marginal that is not p = a wrong accept coin
 * or residual draw (the speculative route would silently change the
 * distribution, invisible in text under sampling -- which is why this gate
 * exists).
 * The low-p token count (the engine's 2026-09-29 conviction): when the RNG
 * hits u == 1.0f, the Gumbel key is +inf and that token wins unconditionally
 * = a uniformly random word injected into the text, per slot |K|/2^24
 * (0.77% over the whole table).  The top-32 5-sigma and the 0.05 total
 * variation both absorb that uniform leak (each top-32 entry off by ~3 sigma,
 * total variation +0.008) and the gate stayed green while a 200k-token
 * production report carried 1,880 random words.  So one more count: how often
 * tokens with p < 1e-7 are drawn at all -- expected N * sum(p) (hundreds), a
 * uniform leak adds N * 0.0077 * |low-p set| / V (thousands), judged at 5
 * sigma of Poisson.
 *
 * Build: make tests/test_ds41_sample (or the test-ds41-sample runner). */

#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define V 129280u          /* the V4.1 vocab size, so the thread stride / bucket shapes match production */
#define ROWS 64u           /* one launch covers 64 rows (consecutive positions); accumulate N samples */
#define N 262144u          /* sample count */

typedef struct { float temperature, top_p, min_p; int top_k; const char *name; } recipe;

static uint64_t rs = 0x1234567ull;
static float rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 40) & 0xFFFFFFu) / 16777216.0f; }

/* Host golden: the kept set + renormalized probabilities in double, the same
 * semantics as ds4_sample_logits. */
static int cmp_desc(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (y > x) - (y < x); }
static void reference(const float *l, const recipe *r, double *p /* [V]: truncated renormalized probability, 0 = outside the kept set */) {
    double M = -INFINITY; for (uint32_t i = 0; i < V; i++) if (l[i] > M) M = l[i];
    static double m[V], sorted[V];
    double cut_k = -INFINITY;
    if (r->top_k > 0 && (uint32_t)r->top_k < V) {
        for (uint32_t i = 0; i < V; i++) sorted[i] = l[i];
        qsort(sorted, V, sizeof sorted[0], cmp_desc);
        cut_k = sorted[r->top_k - 1];
    }
    double Z = 0.0;
    for (uint32_t i = 0; i < V; i++) { m[i] = l[i] >= cut_k ? exp(((double)l[i] - M) / r->temperature) : 0.0; Z += m[i]; }
    double cut_p = -INFINITY;
    if (r->top_p < 1.0f) {   /* the item where the largest-to-smallest cumulative first reaches top_p*Z (ties kept together) */
        uint32_t n = 0; for (uint32_t i = 0; i < V; i++) if (m[i] > 0.0) sorted[n++] = l[i];
        qsort(sorted, n, sizeof sorted[0], cmp_desc);
        double cum = 0.0;
        for (uint32_t i = 0; i < n; i++) { cum += exp((sorted[i] - M) / r->temperature); if (cum >= (double)r->top_p * Z) { cut_p = sorted[i]; break; } }
    }
    const double cut_m = r->min_p > 0.0f ? M + r->temperature * log((double)r->min_p) : -INFINITY;
    double ZK = 0.0;
    for (uint32_t i = 0; i < V; i++) { if (m[i] <= 0.0 || l[i] < cut_p || l[i] < cut_m) m[i] = 0.0; ZK += m[i]; }
    for (uint32_t i = 0; i < V; i++) p[i] = m[i] / ZK;
}

static int check_freq(const char *what, const uint32_t *cnt, const double *p, uint32_t n_samples) {
    int bad = 0; double tv = 0.0; uint32_t outside = 0;
    for (uint32_t i = 0; i < V; i++) { tv += fabs((double)cnt[i] / n_samples - p[i]); if (p[i] == 0.0 && cnt[i]) outside += cnt[i]; }
    tv *= 0.5;
    /* top 32 by p, each at 5 sigma */
    uint32_t top[32]; uint32_t nt = 0;
    for (uint32_t i = 0; i < V; i++) {
        if (p[i] <= 0.0) continue;
        uint32_t j;
        if (nt < 32u) j = nt++;
        else { if (p[i] <= p[top[31]]) continue; j = 31u; }
        while (j > 0 && p[top[j - 1]] < p[i]) { top[j] = top[j - 1]; j--; }
        top[j] = i;
    }
    double worst = 0.0; uint32_t worst_i = 0;
    for (uint32_t t = 0; t < nt; t++) {
        const uint32_t i = top[t];
        const double f = (double)cnt[i] / n_samples, sd = sqrt(p[i] * (1.0 - p[i]) / n_samples);
        const double z = sd > 0.0 ? fabs(f - p[i]) / sd : 0.0;
        if (z > worst) { worst = z; worst_i = i; }
    }
    /* The very-low-probability token count (p < 1e-7, non-empty only in the
     * full-table recipes) vs its expectation: the uniform random injection
     * reads tens of sigma here but only ~3 sigma in the top 32. */
    double elow = 0.0; uint32_t low = 0;
    for (uint32_t i = 0; i < V; i++) if (p[i] > 0.0 && p[i] < 1e-7) { elow += p[i]; low += cnt[i]; }
    elow *= n_samples;
    const double zlow = fabs((double)low - elow) / (sqrt(elow) + 1.0);
    /* The total-variation expectation is roughly sum sqrt(p(1-p)/N)/2, large
     * over a 120k long tail; 0.05 is a coarse bound (a real mismatch reads
     * 0.3~1). */
    if (outside || worst > 5.0 || tv > 0.05 || zlow > 5.0) bad = 1;
    printf("  %-20s outside %u samples, top %u worst %.2fsigma (token %u: freq %.5f vs p %.5f), total variation %.4f, low-p %u vs expected %.0f (%.1fsigma) %s\n",
           what, outside, nt, worst, worst_i, (double)cnt[worst_i] / n_samples, p[worst_i], tv, low, elow, zlow, bad ? "RED" : "ok");
    return bad;
}

int main(void) {
    if (!ds4_gpu_init()) { fprintf(stderr, "no CUDA device\n"); return 1; }
    static float l[V]; static double p[V]; static uint32_t cnt[V], cnt_spec[V];
    /* Synthetic logits: a long tail + dozens of spikes, shaped like a real
     * head (max probability in the tens of percent). */
    for (uint32_t i = 0; i < V; i++) l[i] = -6.0f * rnd() - 4.0f;
    for (uint32_t t = 0; t < 40; t++) l[(uint32_t)(rnd() * V)] = 2.0f + 6.0f * rnd();
    l[777] = 9.5f; l[4242] = 8.8f; l[100000] = 8.0f;
    l[5] = -INFINITY;   /* a non-finite entry: both sides must skip it */
    const recipe R[] = {
        { 1.0f, 1.0f, 0.0f, 0, "temp1 full table" },
        { 0.7f, 0.9f, 0.05f, 0, "temp.7 p.9 m.05" },
        { 1.3f, 0.95f, 0.0f, 40, "temp1.3 p.95 k40" },
        { 1.0f, 1.0f, 0.0f, 1, "k1 (degenerate greedy)" },
    };
    /* The draft distribution q: the p logits plus noise (q != p but close,
     * like a draft tower): (4) draws the draft from q, the verify kernel
     * reads q, and the emitted marginal must still be p with an accept rate
     * about sum min(p,q). */
    static float lq[V]; static double q[V]; static uint32_t cnt_q[V];
    for (uint32_t i = 0; i < V; i++) lq[i] = isfinite(l[i]) ? l[i] + 1.5f * (rnd() - 0.5f) : l[i];
    ds4_gpu_tensor *tl = ds4_gpu_tensor_alloc((uint64_t)ROWS * V * 4u), *tq = ds4_gpu_tensor_alloc((uint64_t)ROWS * V * 4u);
    ds4_gpu_tensor *tpos = ds4_gpu_tensor_alloc(ROWS * 4u), *ttok = ds4_gpu_tensor_alloc(ROWS * 4u), *tout = ds4_gpu_tensor_alloc(ROWS * 16u);
    ds4_gpu_tensor *ttokq = ds4_gpu_tensor_alloc(ROWS * 4u), *toutq = ds4_gpu_tensor_alloc(ROWS * 16u);
    if (!tl || !tq || !tpos || !ttok || !tout || !ttokq || !toutq) { fprintf(stderr, "allocation failed\n"); return 1; }
    for (uint32_t r = 0; r < ROWS; r++)
        if (!ds4_gpu_tensor_write(tl, (uint64_t)r * V * 4u, l, (uint64_t)V * 4u) || !ds4_gpu_tensor_write(tq, (uint64_t)r * V * 4u, lq, (uint64_t)V * 4u)) return 1;
    int32_t pos[ROWS], tok[ROWS], out[ROWS * 4], out2[ROWS * 4], noneq[ROWS], outq[ROWS * 4];
    for (uint32_t i = 0; i < ROWS; i++) noneq[i] = -1;   /* the draft-drawing launch: no draft */
    if (!ds4_gpu_tensor_write(ttokq, 0, noneq, sizeof noneq)) return 1;
    int fail = 0;
    const int32_t d = 4242;   /* point-mass draft: the second-largest item (accept rate = p(d), tens of percent, so the reject path is measurable too) */
    for (size_t ri = 0; ri < sizeof R / sizeof R[0]; ri++) {
        const recipe *r = &R[ri];
        reference(l, r, p); reference(lq, r, q);
        ds4_gpu_sample_params sp = { r->temperature, r->top_p, r->min_p, r->top_k, 0x9E37ull + (uint64_t)ri, 0u };
        ds4_gpu_sample_params spq = sp; spq.stream = 1u;
        memset(cnt, 0, sizeof cnt); memset(cnt_spec, 0, sizeof cnt_spec); memset(cnt_q, 0, sizeof cnt_q);
        uint32_t nacc = 0, ndet = 0, naccq = 0;
        for (uint32_t s = 0; s < N; s += ROWS) {
            for (uint32_t i = 0; i < ROWS; i++) { pos[i] = (int32_t)(s + i); tok[i] = d; }   /* each row's draft = the next row's input = d (the last row has none) */
            if (!ds4_gpu_tensor_write(tpos, 0, pos, sizeof pos) || !ds4_gpu_tensor_write(ttok, 0, tok, sizeof tok)) return 1;
            if (!ds4_gpu_v41_sample_tensor(tout, tl, 0u, ROWS, V, tpos, ttok, &sp, NULL) || !ds4_gpu_synchronize() ||
                !ds4_gpu_tensor_read(tout, 0, out, sizeof out)) { fprintf(stderr, "sampling kernel failed\n"); return 1; }
            if (s == 0) {   /* (3) determinism: the same launch once more */
                if (!ds4_gpu_v41_sample_tensor(tout, tl, 0u, ROWS, V, tpos, ttok, &sp, NULL) || !ds4_gpu_synchronize() ||
                    !ds4_gpu_tensor_read(tout, 0, out2, sizeof out2)) return 1;
                ndet = memcmp(out, out2, sizeof out) == 0 ? 1u : 0u;
            }
            for (uint32_t i = 0; i < ROWS; i++) {
                const int32_t *o = out + 4u * i;
                if (o[0] < 0 || (uint32_t)o[0] >= V) { fprintf(stderr, "sample out of range %d\n", o[0]); return 1; }
                cnt[o[0]]++;
                if (i + 1u < ROWS) { cnt_spec[o[1] ? d : o[2]]++; nacc += o[1] ? 1u : 0u; }
            }
            /* (4) the draft is drawn from q (stream 1, the same kernel the
             * engine's draft tower runs), fed as the next row's input, then
             * verified against q. */
            if (!ds4_gpu_v41_sample_tensor(toutq, tq, 0u, ROWS, V, tpos, ttokq, &spq, NULL) || !ds4_gpu_synchronize() ||
                !ds4_gpu_tensor_read(toutq, 0, outq, sizeof outq)) return 1;
            tok[0] = -1; for (uint32_t i = 0; i + 1u < ROWS; i++) tok[i + 1u] = outq[4u * i];
            if (!ds4_gpu_tensor_write(ttok, 0, tok, sizeof tok)) return 1;
            if (!ds4_gpu_v41_sample_tensor(tout, tl, 0u, ROWS, V, tpos, ttok, &sp, tq) || !ds4_gpu_synchronize() ||
                !ds4_gpu_tensor_read(tout, 0, out, sizeof out)) return 1;
            for (uint32_t i = 0; i + 1u < ROWS; i++) { const int32_t *o = out + 4u * i; cnt_q[o[1] ? tok[i + 1u] : o[2]]++; naccq += o[1] ? 1u : 0u; }
        }
        const uint32_t n_spec = N / ROWS * (ROWS - 1u);
        printf("== %s: kept set %d items (kernel), determinism %s\n", r->name, out[3], ndet ? "ok" : "TWO RUNS DIFFER");
        fail |= !ndet;
        fail |= check_freq("(1) full distribution", cnt, p, N);
        fail |= check_freq("(2) point-mass marginal", cnt_spec, p, n_spec);
        const double acc = (double)nacc / n_spec, sd = sqrt(p[d] * (1.0 - p[d]) / n_spec);
        const int accbad = fabs(acc - p[d]) > 5.0 * sd + 1e-9;
        printf("  accept rate %.5f vs p(draft) %.5f (%.2fsigma) %s\n", acc, p[d], sd > 0 ? fabs(acc - p[d]) / sd : 0.0, accbad ? "RED" : "ok");
        fail |= accbad;
        fail |= check_freq("(4) distribution marginal", cnt_q, p, n_spec);
        double emin = 0.0; for (uint32_t i = 0; i < V; i++) emin += p[i] < q[i] ? p[i] : q[i];
        const double accq = (double)naccq / n_spec, sdq = sqrt(emin * (1.0 - emin) / n_spec);
        const int accqbad = fabs(accq - emin) > 5.0 * sdq + 1e-9;
        printf("  distribution-draft accept rate %.5f vs sum min(p,q) %.5f (%.2fsigma) %s\n", accq, emin, sdq > 0 ? fabs(accq - emin) / sdq : 0.0, accqbad ? "RED" : "ok");
        fail |= accqbad;
    }
    printf(fail ? "ds41 sample distribution gate: RED\n" : "ds41 sample distribution gate: all green\n");
    return fail;
}
