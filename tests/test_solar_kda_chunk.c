/* Solar Open 2 CUDA chunked-KDA prefill test at the production head width.
 *
 * The dim-16 boundary test keeps exercising the sequence kernel; this one
 * pins the 64-token UT-transform path (head_dim 128) against the same scalar
 * recurrence mirror: ragged chunks, sub-block edges, multi-launch
 * continuation through recurrent and convolution state, and agreement with
 * the generic sequence kernel on identical inputs.
 */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    T_HEAD = 4,
    T_DIM = 128,
    T_CONV = 4,
    T_VECTOR = T_HEAD * T_DIM,
    T_STATE = T_HEAD * T_DIM * T_DIM,
    T_CONV_STATE = T_VECTOR * T_CONV,
    T_MAX_TOKENS = 512,
};

typedef struct {
    float state[T_STATE];
    float q_conv[T_CONV_STATE];
    float k_conv[T_CONV_STATE];
    float v_conv[T_CONV_STATE];
} host_state;

static int failures;

#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", (m)); exit(1); } } while (0)

static float softplus_ref(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static float silu_ref(float x) {
    return x / (1.0f + expf(-x));
}

static void make_fixed(float *qw, float *kw, float *vw, float *decay, float *dt) {
    for (uint32_t i = 0; i < T_CONV_STATE; i++) {
        const float x = (float)(i + 1u);
        qw[i] = 0.10f * sinf(0.031f * x) + 0.03f;
        kw[i] = 0.09f * cosf(0.027f * x) - 0.02f;
        vw[i] = 0.08f * sinf(0.023f * x + 0.2f) + 0.01f;
    }
    for (uint32_t h = 0; h < T_HEAD; h++) decay[h] = -expf(-0.5f + 0.15f * h);
    for (uint32_t i = 0; i < T_VECTOR; i++) dt[i] = 0.12f * sinf(0.07f * i) - 0.05f;
}

/* Every 89th gate channel spikes to +11 so softplus saturates and the -5
 * clamp engages: a chunk then carries hard per-channel decay, the case the
 * factored exponents must survive. */
static void make_token(uint32_t token, float *q, float *k, float *v,
                       float *g, float *beta) {
    const float t = (float)(token + 1u);
    for (uint32_t i = 0; i < T_VECTOR; i++) {
        const float x = (float)(i + 1u);
        q[i] = 0.19f * sinf(0.009f * x * t) + 0.01f * (float)(i % 3u);
        k[i] = 0.17f * cosf(0.011f * x * (t + 0.25f));
        v[i] = 0.15f * sinf(0.013f * x + 0.03f * t);
        g[i] = ((i + token) % 89u == 0u) ? 11.0f : 0.6f * sinf(0.017f * x - 0.02f * t);
    }
    for (uint32_t h = 0; h < T_HEAD; h++) beta[h] = -0.8f + 0.35f * h + 0.002f * t;
}

static void conv(float *out, float *state, const float *raw, const float *weight) {
    for (uint32_t ch = 0; ch < T_VECTOR; ch++) {
        float *row = state + (size_t)ch * T_CONV;
        for (uint32_t i = 0; i + 1u < T_CONV; i++) row[i] = row[i + 1u];
        row[T_CONV - 1u] = raw[ch];
        float sum = 0.0f;
        for (uint32_t i = 0; i < T_CONV; i++) sum += row[i] * weight[(size_t)ch * T_CONV + i];
        out[ch] = silu_ref(sum);
    }
}

static void host_step(float *out, host_state *s,
                      const float *q_raw, const float *k_raw, const float *v_raw,
                      const float *g_raw, const float *beta_logits,
                      const float *qw, const float *kw, const float *vw,
                      const float *decay, const float *dt, int glm53) {
    float q[T_VECTOR], k[T_VECTOR], v[T_VECTOR];
    conv(q, s->q_conv, q_raw, qw);
    conv(k, s->k_conv, k_raw, kw);
    conv(v, s->v_conv, v_raw, vw);
    for (uint32_t h = 0; h < T_HEAD; h++) {
        const uint32_t base = h * T_DIM;
        float q2 = 1.0e-6f, k2 = 1.0e-6f;
        for (uint32_t d = 0; d < T_DIM; d++) {
            q2 += q[base + d] * q[base + d];
            k2 += k[base + d] * k[base + d];
        }
        const float qs = 1.0f / sqrtf(q2) / sqrtf((float)T_DIM);
        const float ks = 1.0f / sqrtf(k2);
        for (uint32_t d = 0; d < T_DIM; d++) { q[base + d] *= qs; k[base + d] *= ks; }
        const float beta = (glm53 ? 1.0f : 2.0f) /
            (1.0f + expf(-beta_logits[h]));
        const size_t sb = (size_t)h * T_DIM * T_DIM;
        for (uint32_t vd = 0; vd < T_DIM; vd++) {
            float memory = 0.0f;
            for (uint32_t kd = 0; kd < T_DIM; kd++) {
                const float raw = g_raw[base + kd] + dt[base + kd];
                float gate = glm53
                    ? -5.0f / (1.0f + expf(-expf(decay[h]) * raw))
                    : decay[h] * softplus_ref(raw);
                if (!glm53 && gate < -5.0f) gate = -5.0f;
                const size_t ix = sb + (size_t)kd * T_DIM + vd;
                s->state[ix] *= expf(gate);
                memory += s->state[ix] * k[base + kd];
            }
            const float delta = (v[base + vd] - memory) * beta;
            float result = 0.0f;
            for (uint32_t kd = 0; kd < T_DIM; kd++) {
                const size_t ix = sb + (size_t)kd * T_DIM + vd;
                s->state[ix] += k[base + kd] * delta;
                result += s->state[ix] * q[base + kd];
            }
            out[base + vd] = result;
        }
    }
}

static void compare(const char *label, const float *got, const float *want,
                    size_t n, double atol, double rtol) {
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double delta = fabs((double)got[i] - want[i]);
        const double scale = fmax(fabs((double)got[i]), fabs((double)want[i]));
        if (delta > max_abs) max_abs = delta;
        if (scale > 1.0e-8 && delta / scale > max_rel) max_rel = delta / scale;
    }
    const int ok = max_abs <= atol || max_rel <= rtol;
    if (!ok) failures++;
    printf("%-34s abs=%.3e rel=%.3e %s\n", label, max_abs, max_rel, ok ? "ok" : "FAIL");
}

typedef struct {
    ds4_gpu_tensor *q, *k, *v, *g, *beta, *out, *scratch;
    ds4_gpu_tensor *state, *qc, *kc, *vc;
    ds4_gpu_tensor *qw, *kw, *vw, *decay, *dt;
} device_set;

/* Feeds `total` tokens through the prefill entry in the given launch splits
 * and reads back every output row plus the final states. */
static void run_case(device_set *d, const float *q, const float *k,
                     const float *v, const float *g, const float *beta,
                     uint32_t total, const uint32_t *splits, uint32_t n_splits,
                     int glm53,
                     float *got, float *got_state,
                     float *got_qc, float *got_kc, float *got_vc) {
    CHECK(ds4_gpu_tensor_fill_f32(d->state, 0.0f, T_STATE), "state reset");
    CHECK(ds4_gpu_tensor_fill_f32(d->qc, 0.0f, T_CONV_STATE), "q conv reset");
    CHECK(ds4_gpu_tensor_fill_f32(d->kc, 0.0f, T_CONV_STATE), "k conv reset");
    CHECK(ds4_gpu_tensor_fill_f32(d->vc, 0.0f, T_CONV_STATE), "v conv reset");
    uint32_t done = 0;
    for (uint32_t part = 0; done < total; part++) {
        const uint32_t n = part < n_splits ? splits[part] : total - done;
        const uint64_t vb = (uint64_t)n * T_VECTOR * sizeof(float);
        const uint64_t bb = (uint64_t)n * T_HEAD * sizeof(float);
        CHECK(ds4_gpu_tensor_write(d->q, 0, q + (size_t)done * T_VECTOR, vb), "write q");
        CHECK(ds4_gpu_tensor_write(d->k, 0, k + (size_t)done * T_VECTOR, vb), "write k");
        CHECK(ds4_gpu_tensor_write(d->v, 0, v + (size_t)done * T_VECTOR, vb), "write v");
        CHECK(ds4_gpu_tensor_write(d->g, 0, g + (size_t)done * T_VECTOR, vb), "write g");
        CHECK(ds4_gpu_tensor_write(d->beta, 0, beta + (size_t)done * T_HEAD, bb), "write beta");
        CHECK((glm53 ? ds4_gpu_glm53_kda_prefill_tensor
                     : ds4_gpu_solar_kda_prefill_tensor)(
                    d->out, d->scratch, d->state, d->qc, d->kc, d->vc,
                    d->q, d->k, d->v, d->g, d->beta, d->qw, d->kw, d->vw,
                    d->decay, d->dt, n, T_HEAD, T_DIM, T_CONV, -5.0f),
              "prefill launch");
        CHECK(ds4_gpu_tensor_read(d->out, 0, got + (size_t)done * T_VECTOR, vb), "read out");
        done += n;
    }
    CHECK(ds4_gpu_tensor_read(d->state, 0, got_state, T_STATE * sizeof(float)), "read state");
    CHECK(ds4_gpu_tensor_read(d->qc, 0, got_qc, T_CONV_STATE * sizeof(float)), "read q conv");
    CHECK(ds4_gpu_tensor_read(d->kc, 0, got_kc, T_CONV_STATE * sizeof(float)), "read k conv");
    CHECK(ds4_gpu_tensor_read(d->vc, 0, got_vc, T_CONV_STATE * sizeof(float)), "read v conv");
}

int main(void) {
    unsetenv("DS4_SOLAR_KDA_STATE_PARTS");
    CHECK(ds4_gpu_init(), "CUDA init");
    const uint64_t scratch_bytes =
        ds4_gpu_solar_kda_prefill_scratch_bytes(
            T_MAX_TOKENS, T_HEAD, T_DIM);
    CHECK(scratch_bytes > 0u, "chunk scratch sizing");
    CHECK(ds4_gpu_solar_kda_prefill_scratch_bytes(1u, T_HEAD, T_DIM) == 0u,
          "decode needs no chunk scratch");
    CHECK(ds4_gpu_solar_kda_prefill_scratch_bytes(63u, T_HEAD, T_DIM) == 0u,
          "short prefill stays on recurrence");
    CHECK(ds4_gpu_solar_kda_prefill_scratch_bytes(64u, T_HEAD, T_DIM) > 0u,
          "full chunk selects transform workspace");
    CHECK(ds4_gpu_solar_kda_prefill_scratch_bytes(
              T_MAX_TOKENS, T_HEAD, 16u) == 0u,
          "non-production width uses sequence path");

    float *q = calloc((size_t)T_MAX_TOKENS * T_VECTOR, sizeof(float));
    float *k = calloc((size_t)T_MAX_TOKENS * T_VECTOR, sizeof(float));
    float *v = calloc((size_t)T_MAX_TOKENS * T_VECTOR, sizeof(float));
    float *g = calloc((size_t)T_MAX_TOKENS * T_VECTOR, sizeof(float));
    float *beta = calloc((size_t)T_MAX_TOKENS * T_HEAD, sizeof(float));
    float *want = calloc((size_t)T_MAX_TOKENS * T_VECTOR, sizeof(float));
    float *got = calloc((size_t)T_MAX_TOKENS * T_VECTOR, sizeof(float));
    float *got_state = calloc(T_STATE, sizeof(float));
    float *got_conv = calloc(3u * T_CONV_STATE, sizeof(float));
    CHECK(q && k && v && g && beta && want && got && got_state && got_conv,
          "host arrays");

    float qw[T_CONV_STATE], kw[T_CONV_STATE], vw[T_CONV_STATE];
    float decay[T_HEAD], dt[T_VECTOR];
    make_fixed(qw, kw, vw, decay, dt);

    device_set d;
    d.q = ds4_gpu_tensor_alloc((uint64_t)T_MAX_TOKENS * T_VECTOR * sizeof(float));
    d.k = ds4_gpu_tensor_alloc((uint64_t)T_MAX_TOKENS * T_VECTOR * sizeof(float));
    d.v = ds4_gpu_tensor_alloc((uint64_t)T_MAX_TOKENS * T_VECTOR * sizeof(float));
    d.g = ds4_gpu_tensor_alloc((uint64_t)T_MAX_TOKENS * T_VECTOR * sizeof(float));
    d.beta = ds4_gpu_tensor_alloc((uint64_t)T_MAX_TOKENS * T_HEAD * sizeof(float));
    d.out = ds4_gpu_tensor_alloc((uint64_t)T_MAX_TOKENS * T_VECTOR * sizeof(float));
    d.scratch = ds4_gpu_tensor_alloc(scratch_bytes);
    d.state = ds4_gpu_tensor_alloc((uint64_t)T_STATE * sizeof(float));
    d.qc = ds4_gpu_tensor_alloc((uint64_t)T_CONV_STATE * sizeof(float));
    d.kc = ds4_gpu_tensor_alloc((uint64_t)T_CONV_STATE * sizeof(float));
    d.vc = ds4_gpu_tensor_alloc((uint64_t)T_CONV_STATE * sizeof(float));
    d.qw = ds4_gpu_tensor_alloc(sizeof(qw));
    d.kw = ds4_gpu_tensor_alloc(sizeof(kw));
    d.vw = ds4_gpu_tensor_alloc(sizeof(vw));
    d.decay = ds4_gpu_tensor_alloc(sizeof(decay));
    d.dt = ds4_gpu_tensor_alloc(sizeof(dt));
    CHECK(d.q && d.k && d.v && d.g && d.beta && d.out && d.scratch &&
          d.state && d.qc &&
          d.kc && d.vc && d.qw && d.kw && d.vw && d.decay && d.dt,
          "device arrays");
    CHECK(ds4_gpu_tensor_write(d.qw, 0, qw, sizeof(qw)), "q weights");
    CHECK(ds4_gpu_tensor_write(d.kw, 0, kw, sizeof(kw)), "k weights");
    CHECK(ds4_gpu_tensor_write(d.vw, 0, vw, sizeof(vw)), "v weights");
    CHECK(ds4_gpu_tensor_write(d.decay, 0, decay, sizeof(decay)), "decay");
    CHECK(ds4_gpu_tensor_write(d.dt, 0, dt, sizeof(dt)), "dt");

    host_state hs;
    memset(&hs, 0, sizeof(hs));
    for (uint32_t t = 0; t < T_MAX_TOKENS; t++) {
        make_token(t, q + (size_t)t * T_VECTOR, k + (size_t)t * T_VECTOR,
                   v + (size_t)t * T_VECTOR, g + (size_t)t * T_VECTOR,
                   beta + (size_t)t * T_HEAD);
        host_step(want + (size_t)t * T_VECTOR, &hs,
                  q + (size_t)t * T_VECTOR, k + (size_t)t * T_VECTOR,
                  v + (size_t)t * T_VECTOR, g + (size_t)t * T_VECTOR,
                  beta + (size_t)t * T_HEAD, qw, kw, vw, decay, dt, 0);
    }

    puts("== Solar Open 2 CUDA chunked KDA prefill (head_dim 128) ==");
    /* Host mirror states are only kept for the full length, so every case
     * replays from token zero and compares against the host prefix. Final
     * state is asserted on the full-length cases below. */
    const uint32_t singles[] = {1, 2, 7, 63, 64, 65, 127, 128, 129, 200, 391};
    char label[80];
    for (size_t i = 0; i < sizeof(singles) / sizeof(singles[0]); i++) {
        const uint32_t total = singles[i];
        run_case(&d, q, k, v, g, beta, total, NULL, 0, 0, got, got_state,
                 got_conv, got_conv + T_CONV_STATE,
                 got_conv + 2u * T_CONV_STATE);
        snprintf(label, sizeof(label), "single %u output", total);
        compare(label, got, want, (size_t)total * T_VECTOR, 1.0e-4, 1.0e-3);
    }

    /* Full-length single call: output, recurrent state, conv states. */
    run_case(&d, q, k, v, g, beta, T_MAX_TOKENS, NULL, 0, 0, got, got_state,
             got_conv, got_conv + T_CONV_STATE, got_conv + 2u * T_CONV_STATE);
    compare("single 512 output", got, want, (size_t)T_MAX_TOKENS * T_VECTOR,
            1.0e-4, 1.0e-3);
    compare("single 512 state", got_state, hs.state, T_STATE, 2.0e-4, 2.0e-3);
    compare("single 512 q conv", got_conv, hs.q_conv, T_CONV_STATE, 0.0, 0.0);
    compare("single 512 k conv", got_conv + T_CONV_STATE, hs.k_conv,
            T_CONV_STATE, 0.0, 0.0);
    compare("single 512 v conv", got_conv + 2u * T_CONV_STATE, hs.v_conv,
            T_CONV_STATE, 0.0, 0.0);

    /* Multi-launch continuation across awkward boundaries: recurrent and
     * conv state must hand off exactly like the engine's 2048-token launch
     * chunking does. */
    static const uint32_t split_a[] = {37, 93, 130, 175, 77};
    static const uint32_t split_b[] = {64, 65, 71, 1, 2, 309};
    static const uint32_t split_c[] = {128, 7, 377};
    const struct { const uint32_t *s; uint32_t n; const char *name; } splits[] = {
        {split_a, 5, "37+93+130+175+77"},
        {split_b, 6, "64+65+71+1+2+309"},
        {split_c, 3, "128+7+377"},
    };
    for (size_t i = 0; i < sizeof(splits) / sizeof(splits[0]); i++) {
        run_case(&d, q, k, v, g, beta, T_MAX_TOKENS, splits[i].s, splits[i].n,
                 0, got, got_state, got_conv, got_conv + T_CONV_STATE,
                 got_conv + 2u * T_CONV_STATE);
        snprintf(label, sizeof(label), "split %s output", splits[i].name);
        compare(label, got, want, (size_t)T_MAX_TOKENS * T_VECTOR,
                1.0e-4, 1.0e-3);
        snprintf(label, sizeof(label), "split %s state", splits[i].name);
        compare(label, got_state, hs.state, T_STATE, 2.0e-4, 2.0e-3);
        snprintf(label, sizeof(label), "split %s conv", splits[i].name);
        compare(label, got_conv, hs.q_conv, T_CONV_STATE, 0.0, 0.0);
    }

    /* The sequence kernel on the same inputs, as the in-family drift
     * reference: chunked and generic must not diverge from each other any
     * further than each sits from the scalar mirror. */
    CHECK(setenv("DS4_SOLAR_KDA_STATE_PARTS", "1", 1) == 0, "select generic");
    float *generic_out = calloc((size_t)T_MAX_TOKENS * T_VECTOR, sizeof(float));
    float *generic_state = calloc(T_STATE, sizeof(float));
    CHECK(generic_out && generic_state, "generic scratch");
    run_case(&d, q, k, v, g, beta, T_MAX_TOKENS, NULL, 0, 0, generic_out,
             generic_state, got_conv, got_conv + T_CONV_STATE,
             got_conv + 2u * T_CONV_STATE);
    unsetenv("DS4_SOLAR_KDA_STATE_PARTS");
    run_case(&d, q, k, v, g, beta, T_MAX_TOKENS, NULL, 0, 0, got, got_state,
             got_conv, got_conv + T_CONV_STATE, got_conv + 2u * T_CONV_STATE);
    compare("chunked vs generic output", got, generic_out,
            (size_t)T_MAX_TOKENS * T_VECTOR, 1.0e-4, 1.0e-3);
    compare("chunked vs generic state", got_state, generic_state, T_STATE,
            2.0e-4, 2.0e-3);

    puts("== GLM 5.3 CUDA chunked KDA prefill (head_dim 128) ==");
    memset(&hs, 0, sizeof(hs));
    for (uint32_t t = 0; t < T_MAX_TOKENS; t++) {
        host_step(want + (size_t)t * T_VECTOR, &hs,
                  q + (size_t)t * T_VECTOR, k + (size_t)t * T_VECTOR,
                  v + (size_t)t * T_VECTOR, g + (size_t)t * T_VECTOR,
                  beta + (size_t)t * T_HEAD, qw, kw, vw, decay, dt, 1);
    }
    run_case(&d, q, k, v, g, beta, T_MAX_TOKENS, NULL, 0, 1,
             got, got_state, got_conv, got_conv + T_CONV_STATE,
             got_conv + 2u * T_CONV_STATE);
    compare("GLM single 512 output", got, want,
            (size_t)T_MAX_TOKENS * T_VECTOR, 1.0e-4, 1.0e-3);
    compare("GLM single 512 state", got_state, hs.state, T_STATE,
            2.0e-4, 2.0e-3);
    run_case(&d, q, k, v, g, beta, T_MAX_TOKENS,
             split_b, sizeof(split_b) / sizeof(split_b[0]), 1,
             got, got_state, got_conv, got_conv + T_CONV_STATE,
             got_conv + 2u * T_CONV_STATE);
    compare("GLM split output", got, want,
            (size_t)T_MAX_TOKENS * T_VECTOR, 1.0e-4, 1.0e-3);
    compare("GLM split state", got_state, hs.state, T_STATE,
            2.0e-4, 2.0e-3);
    CHECK(setenv("DS4_SOLAR_KDA_STATE_PARTS", "1", 1) == 0,
          "select GLM generic");
    run_case(&d, q, k, v, g, beta, T_MAX_TOKENS, NULL, 0, 1,
             generic_out, generic_state, got_conv,
             got_conv + T_CONV_STATE, got_conv + 2u * T_CONV_STATE);
    unsetenv("DS4_SOLAR_KDA_STATE_PARTS");
    compare("GLM chunked vs generic output", got, generic_out,
            (size_t)T_MAX_TOKENS * T_VECTOR, 1.0e-4, 1.0e-3);
    compare("GLM chunked vs generic state", got_state, generic_state,
            T_STATE, 2.0e-4, 2.0e-3);
    free(generic_out);
    free(generic_state);

    ds4_gpu_tensor_free(d.q); ds4_gpu_tensor_free(d.k); ds4_gpu_tensor_free(d.v);
    ds4_gpu_tensor_free(d.g); ds4_gpu_tensor_free(d.beta); ds4_gpu_tensor_free(d.out);
    ds4_gpu_tensor_free(d.scratch);
    ds4_gpu_tensor_free(d.state); ds4_gpu_tensor_free(d.qc); ds4_gpu_tensor_free(d.kc);
    ds4_gpu_tensor_free(d.vc); ds4_gpu_tensor_free(d.qw); ds4_gpu_tensor_free(d.kw);
    ds4_gpu_tensor_free(d.vw); ds4_gpu_tensor_free(d.decay); ds4_gpu_tensor_free(d.dt);
    free(q); free(k); free(v); free(g); free(beta); free(want); free(got);
    free(got_state); free(got_conv);
    ds4_gpu_cleanup();
    puts(failures ? "Solar/GLM chunked KDA checks FAILED"
                  : "all Solar/GLM chunked KDA checks passed");
    return failures ? 1 : 0;
}
