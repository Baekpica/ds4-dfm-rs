/* Solar Open 2 CUDA KDA decode kernel vs an independent host scalar mirror of
 * Upstage's torch_recurrent_kda + torch_causal_conv1d_update equations.
 *
 * This test uses Solar's real head dimension (128), exercises the -5 gate
 * clamp, runs a multi-token continuation, then checks reset and fork-by-copy
 * identity for recurrent and all three short-convolution states.
 */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    T_HEAD = 2,
    T_DIM = 128,
    T_CONV = 4,
    T_TOKENS = 7,
    T_VECTOR = T_HEAD * T_DIM,
    T_STATE = T_HEAD * T_DIM * T_DIM,
    T_CONV_STATE = T_VECTOR * T_CONV,
};

typedef struct {
    float state[T_STATE];
    float q_conv[T_CONV_STATE];
    float k_conv[T_CONV_STATE];
    float v_conv[T_CONV_STATE];
} cpu_kda_state;

typedef struct {
    ds4_gpu_tensor *out;
    ds4_gpu_tensor *state;
    ds4_gpu_tensor *q_conv;
    ds4_gpu_tensor *k_conv;
    ds4_gpu_tensor *v_conv;
} gpu_kda_state;

typedef struct {
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *k;
    ds4_gpu_tensor *v;
    ds4_gpu_tensor *g;
    ds4_gpu_tensor *beta;
    ds4_gpu_tensor *q_weight;
    ds4_gpu_tensor *k_weight;
    ds4_gpu_tensor *v_weight;
    ds4_gpu_tensor *decay;
    ds4_gpu_tensor *dt;
} gpu_kda_inputs;

static int failures = 0;

#define REQUIRE(condition, message)                                           \
    do {                                                                      \
        if (!(condition)) {                                                   \
            fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__);    \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

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
        qw[i] = 0.12f * sinf(0.013f * x) + 0.04f;
        kw[i] = 0.11f * cosf(0.017f * x) - 0.03f;
        vw[i] = 0.09f * sinf(0.019f * x + 0.4f) + 0.02f;
    }
    for (uint32_t h = 0; h < T_HEAD; h++) {
        /* GGUF stores -exp(A_log), not A_log. */
        decay[h] = -expf(-0.35f + 0.17f * (float)h);
    }
    for (uint32_t i = 0; i < T_VECTOR; i++) {
        dt[i] = 0.15f * sinf(0.031f * (float)i) - 0.08f;
    }
}

static void make_token(uint32_t token, float *q, float *k, float *v,
                       float *g, float *beta) {
    for (uint32_t i = 0; i < T_VECTOR; i++) {
        const float t = (float)(token + 1u);
        const float x = (float)(i + 3u);
        q[i] = 0.22f * sinf(0.011f * x * t) + 0.01f * (float)(i % 5u);
        k[i] = 0.19f * cosf(0.014f * x * (t + 0.5f)) - 0.012f * (float)(i % 7u);
        v[i] = 0.17f * sinf(0.009f * x + 0.2f * t);
        g[i] = ((i + 13u * token) % 61u == 0u)
            ? 12.0f /* forces the official lower-bound clamp */
            : 0.7f * sinf(0.023f * x - 0.3f * t);
    }
    for (uint32_t h = 0; h < T_HEAD; h++) {
        beta[h] = -0.8f + 0.55f * (float)h + 0.13f * (float)token;
    }
}

static void conv_one(float *out, float *state, const float *raw,
                     const float *weight) {
    for (uint32_t channel = 0; channel < T_VECTOR; channel++) {
        float *row = state + (size_t)channel * T_CONV;
        for (uint32_t tap = 0; tap + 1u < T_CONV; tap++) row[tap] = row[tap + 1u];
        row[T_CONV - 1u] = raw[channel];
        float sum = 0.0f;
        for (uint32_t tap = 0; tap < T_CONV; tap++) {
            sum += row[tap] * weight[(size_t)channel * T_CONV + tap];
        }
        out[channel] = silu_ref(sum);
    }
}

static void cpu_step(float *out, cpu_kda_state *cache,
                     const float *q_raw, const float *k_raw, const float *v_raw,
                     const float *g_raw, const float *beta_logits,
                     const float *q_weight, const float *k_weight,
                     const float *v_weight, const float *decay_scale,
                     const float *dt_bias, int glm53) {
    float q[T_VECTOR], k[T_VECTOR], v[T_VECTOR];
    conv_one(q, cache->q_conv, q_raw, q_weight);
    conv_one(k, cache->k_conv, k_raw, k_weight);
    conv_one(v, cache->v_conv, v_raw, v_weight);

    for (uint32_t head = 0; head < T_HEAD; head++) {
        float q_norm = 1.0e-6f, k_norm = 1.0e-6f;
        const uint32_t base = head * T_DIM;
        for (uint32_t d = 0; d < T_DIM; d++) {
            q_norm += q[base + d] * q[base + d];
            k_norm += k[base + d] * k[base + d];
        }
        const float q_scale = 1.0f / sqrtf(q_norm) / sqrtf((float)T_DIM);
        const float k_scale = 1.0f / sqrtf(k_norm);
        for (uint32_t d = 0; d < T_DIM; d++) {
            q[base + d] *= q_scale;
            k[base + d] *= k_scale;
        }

        const float beta = (glm53 ? 1.0f : 2.0f) /
            (1.0f + expf(-beta_logits[head]));
        const size_t state_base = (size_t)head * T_DIM * T_DIM;
        for (uint32_t value_dim = 0; value_dim < T_DIM; value_dim++) {
            float memory = 0.0f;
            for (uint32_t key_dim = 0; key_dim < T_DIM; key_dim++) {
                const float raw =
                    g_raw[base + key_dim] + dt_bias[base + key_dim];
                float gate = glm53
                    ? -5.0f / (1.0f + expf(-expf(decay_scale[head]) * raw))
                    : decay_scale[head] * softplus_ref(raw);
                if (!glm53 && gate < -5.0f) gate = -5.0f;
                const size_t index = state_base + (size_t)key_dim * T_DIM + value_dim;
                cache->state[index] *= expf(gate);
                memory += cache->state[index] * k[base + key_dim];
            }
            const float delta = (v[base + value_dim] - memory) * beta;
            float result = 0.0f;
            for (uint32_t key_dim = 0; key_dim < T_DIM; key_dim++) {
                const size_t index = state_base + (size_t)key_dim * T_DIM + value_dim;
                cache->state[index] += k[base + key_dim] * delta;
                result += cache->state[index] * q[base + key_dim];
            }
            out[base + value_dim] = result;
        }
    }
}

static void compare(const char *name, const float *got, const float *want,
                    size_t count, double abs_tolerance, double rel_tolerance) {
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < count; i++) {
        const double delta = fabs((double)got[i] - (double)want[i]);
        const double scale = fmax(fabs((double)got[i]), fabs((double)want[i]));
        if (delta > max_abs) max_abs = delta;
        if (scale > 1.0e-7 && delta / scale > max_rel) max_rel = delta / scale;
    }
    const int ok = max_abs <= abs_tolerance || max_rel <= rel_tolerance;
    if (!ok) failures++;
    printf("%-30s max_abs=%.3e max_rel=%.3e %s\n",
           name, max_abs, max_rel, ok ? "ok" : "FAIL");
}

static int gpu_state_alloc(gpu_kda_state *state) {
    memset(state, 0, sizeof(*state));
    state->out = ds4_gpu_tensor_alloc((uint64_t)T_VECTOR * sizeof(float));
    state->state = ds4_gpu_tensor_alloc((uint64_t)T_STATE * sizeof(float));
    state->q_conv = ds4_gpu_tensor_alloc((uint64_t)T_CONV_STATE * sizeof(float));
    state->k_conv = ds4_gpu_tensor_alloc((uint64_t)T_CONV_STATE * sizeof(float));
    state->v_conv = ds4_gpu_tensor_alloc((uint64_t)T_CONV_STATE * sizeof(float));
    REQUIRE(state->out && state->state && state->q_conv && state->k_conv && state->v_conv,
            "GPU KDA state allocation");
    REQUIRE(ds4_gpu_tensor_fill_f32(state->state, 0.0f, T_STATE), "state reset");
    REQUIRE(ds4_gpu_tensor_fill_f32(state->q_conv, 0.0f, T_CONV_STATE), "q conv reset");
    REQUIRE(ds4_gpu_tensor_fill_f32(state->k_conv, 0.0f, T_CONV_STATE), "k conv reset");
    REQUIRE(ds4_gpu_tensor_fill_f32(state->v_conv, 0.0f, T_CONV_STATE), "v conv reset");
    return 1;
}

static void gpu_state_free(gpu_kda_state *state) {
    ds4_gpu_tensor_free(state->out);
    ds4_gpu_tensor_free(state->state);
    ds4_gpu_tensor_free(state->q_conv);
    ds4_gpu_tensor_free(state->k_conv);
    ds4_gpu_tensor_free(state->v_conv);
    memset(state, 0, sizeof(*state));
}

static int gpu_inputs_alloc(gpu_kda_inputs *inputs) {
    memset(inputs, 0, sizeof(*inputs));
    inputs->q = ds4_gpu_tensor_alloc((uint64_t)T_VECTOR * sizeof(float));
    inputs->k = ds4_gpu_tensor_alloc((uint64_t)T_VECTOR * sizeof(float));
    inputs->v = ds4_gpu_tensor_alloc((uint64_t)T_VECTOR * sizeof(float));
    inputs->g = ds4_gpu_tensor_alloc((uint64_t)T_VECTOR * sizeof(float));
    inputs->beta = ds4_gpu_tensor_alloc((uint64_t)T_HEAD * sizeof(float));
    inputs->q_weight = ds4_gpu_tensor_alloc((uint64_t)T_CONV_STATE * sizeof(float));
    inputs->k_weight = ds4_gpu_tensor_alloc((uint64_t)T_CONV_STATE * sizeof(float));
    inputs->v_weight = ds4_gpu_tensor_alloc((uint64_t)T_CONV_STATE * sizeof(float));
    inputs->decay = ds4_gpu_tensor_alloc((uint64_t)T_HEAD * sizeof(float));
    inputs->dt = ds4_gpu_tensor_alloc((uint64_t)T_VECTOR * sizeof(float));
    REQUIRE(inputs->q && inputs->k && inputs->v && inputs->g && inputs->beta &&
            inputs->q_weight && inputs->k_weight && inputs->v_weight &&
            inputs->decay && inputs->dt, "GPU KDA input allocation");
    return 1;
}

static void gpu_inputs_free(gpu_kda_inputs *inputs) {
    ds4_gpu_tensor_free(inputs->q);
    ds4_gpu_tensor_free(inputs->k);
    ds4_gpu_tensor_free(inputs->v);
    ds4_gpu_tensor_free(inputs->g);
    ds4_gpu_tensor_free(inputs->beta);
    ds4_gpu_tensor_free(inputs->q_weight);
    ds4_gpu_tensor_free(inputs->k_weight);
    ds4_gpu_tensor_free(inputs->v_weight);
    ds4_gpu_tensor_free(inputs->decay);
    ds4_gpu_tensor_free(inputs->dt);
    memset(inputs, 0, sizeof(*inputs));
}

static int gpu_write_token(const gpu_kda_inputs *inputs, const float *q,
                           const float *k, const float *v, const float *g,
                           const float *beta) {
    const uint64_t vector_bytes = (uint64_t)T_VECTOR * sizeof(float);
    REQUIRE(ds4_gpu_tensor_write(inputs->q, 0, q, vector_bytes), "write q");
    REQUIRE(ds4_gpu_tensor_write(inputs->k, 0, k, vector_bytes), "write k");
    REQUIRE(ds4_gpu_tensor_write(inputs->v, 0, v, vector_bytes), "write v");
    REQUIRE(ds4_gpu_tensor_write(inputs->g, 0, g, vector_bytes), "write g");
    REQUIRE(ds4_gpu_tensor_write(inputs->beta, 0, beta,
                                 (uint64_t)T_HEAD * sizeof(float)), "write beta");
    return 1;
}

static int gpu_step(gpu_kda_state *state, const gpu_kda_inputs *inputs,
                    int glm53) {
    if (glm53) {
        return ds4_gpu_glm53_kda_decode_tensor(
            state->out, state->state, state->q_conv, state->k_conv,
            state->v_conv, inputs->q, inputs->k, inputs->v, inputs->g,
            inputs->beta, inputs->q_weight, inputs->k_weight,
            inputs->v_weight, inputs->decay, inputs->dt,
            T_HEAD, T_DIM, T_CONV, -5.0f);
    }
    return ds4_gpu_solar_kda_decode_tensor(
        state->out, state->state, state->q_conv, state->k_conv, state->v_conv,
        inputs->q, inputs->k, inputs->v, inputs->g, inputs->beta,
        inputs->q_weight, inputs->k_weight, inputs->v_weight,
        inputs->decay, inputs->dt, T_HEAD, T_DIM, T_CONV, -5.0f);
}

static int gpu_copy_state(gpu_kda_state *dst, const gpu_kda_state *src) {
    REQUIRE(ds4_gpu_tensor_copy(dst->state, 0, src->state, 0,
                                (uint64_t)T_STATE * sizeof(float)), "copy recurrent state");
    REQUIRE(ds4_gpu_tensor_copy(dst->q_conv, 0, src->q_conv, 0,
                                (uint64_t)T_CONV_STATE * sizeof(float)), "copy q conv state");
    REQUIRE(ds4_gpu_tensor_copy(dst->k_conv, 0, src->k_conv, 0,
                                (uint64_t)T_CONV_STATE * sizeof(float)), "copy k conv state");
    REQUIRE(ds4_gpu_tensor_copy(dst->v_conv, 0, src->v_conv, 0,
                                (uint64_t)T_CONV_STATE * sizeof(float)), "copy v conv state");
    return 1;
}

static int gpu_reset_state(gpu_kda_state *state) {
    REQUIRE(ds4_gpu_tensor_fill_f32(state->state, 0.0f, T_STATE), "reset recurrent state");
    REQUIRE(ds4_gpu_tensor_fill_f32(state->q_conv, 0.0f, T_CONV_STATE), "reset q conv state");
    REQUIRE(ds4_gpu_tensor_fill_f32(state->k_conv, 0.0f, T_CONV_STATE), "reset k conv state");
    REQUIRE(ds4_gpu_tensor_fill_f32(state->v_conv, 0.0f, T_CONV_STATE), "reset v conv state");
    return 1;
}

static void test_banked_decode(
        const gpu_kda_inputs *fixed,
        const float *q_weight,
        const float *k_weight,
        const float *v_weight,
        const float *decay,
        const float *dt,
        int glm53) {
    enum { T_BANKS = 2 };
    const uint64_t recurrent_bytes = (uint64_t)T_STATE * sizeof(float);
    const uint64_t conv_bytes = (uint64_t)T_CONV_STATE * sizeof(float);
    const uint64_t q_offset = recurrent_bytes;
    const uint64_t k_offset = q_offset + conv_bytes;
    const uint64_t v_offset = k_offset + conv_bytes;
    const uint64_t bank_stride = v_offset + conv_bytes;
    const uint64_t vector_bytes = (uint64_t)T_VECTOR * sizeof(float);
    const uint32_t bank_ids[T_BANKS] = {1u, 0u};

    float *q = calloc((size_t)T_BANKS * T_VECTOR, sizeof(*q));
    float *k = calloc((size_t)T_BANKS * T_VECTOR, sizeof(*k));
    float *v = calloc((size_t)T_BANKS * T_VECTOR, sizeof(*v));
    float *g = calloc((size_t)T_BANKS * T_VECTOR, sizeof(*g));
    float *beta = calloc((size_t)T_BANKS * T_HEAD, sizeof(*beta));
    float *gpu_out = calloc((size_t)T_BANKS * T_VECTOR, sizeof(*gpu_out));
    float *want_out = calloc((size_t)T_BANKS * T_VECTOR, sizeof(*want_out));
    float *bank_data = malloc((size_t)bank_stride);
    cpu_kda_state *cpu = calloc(T_BANKS, sizeof(*cpu));
    REQUIRE(q && k && v && g && beta && gpu_out && want_out && bank_data && cpu,
            "banked KDA host allocation");

    for (uint32_t row = 0; row < T_BANKS; row++) {
        const uint32_t token = row == 0u ? 3u : 5u;
        make_token(token,
                   q + (size_t)row * T_VECTOR,
                   k + (size_t)row * T_VECTOR,
                   v + (size_t)row * T_VECTOR,
                   g + (size_t)row * T_VECTOR,
                   beta + (size_t)row * T_HEAD);
        cpu_step(want_out + (size_t)row * T_VECTOR,
                 &cpu[bank_ids[row]],
                 q + (size_t)row * T_VECTOR,
                 k + (size_t)row * T_VECTOR,
                 v + (size_t)row * T_VECTOR,
                 g + (size_t)row * T_VECTOR,
                 beta + (size_t)row * T_HEAD,
                 q_weight, k_weight, v_weight, decay, dt, glm53);
    }

    ds4_gpu_tensor *state_slab = ds4_gpu_tensor_alloc(
        (uint64_t)T_BANKS * bank_stride);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
        (uint64_t)T_BANKS * vector_bytes);
    ds4_gpu_tensor *dq = ds4_gpu_tensor_alloc(
        (uint64_t)T_BANKS * vector_bytes);
    ds4_gpu_tensor *dk = ds4_gpu_tensor_alloc(
        (uint64_t)T_BANKS * vector_bytes);
    ds4_gpu_tensor *dv = ds4_gpu_tensor_alloc(
        (uint64_t)T_BANKS * vector_bytes);
    ds4_gpu_tensor *dg = ds4_gpu_tensor_alloc(
        (uint64_t)T_BANKS * vector_bytes);
    ds4_gpu_tensor *dbeta = ds4_gpu_tensor_alloc(
        (uint64_t)T_BANKS * T_HEAD * sizeof(float));
    ds4_gpu_tensor *dbanks = ds4_gpu_tensor_alloc(sizeof(bank_ids));
    REQUIRE(state_slab && out && dq && dk && dv && dg && dbeta && dbanks,
            "banked KDA device allocation");
    REQUIRE(ds4_gpu_tensor_fill_f32(
                state_slab, 0.0f,
                (uint64_t)T_BANKS * bank_stride / sizeof(float)),
            "banked KDA state reset");
    REQUIRE(ds4_gpu_tensor_write(dq, 0, q,
                (uint64_t)T_BANKS * vector_bytes), "banked write q");
    REQUIRE(ds4_gpu_tensor_write(dk, 0, k,
                (uint64_t)T_BANKS * vector_bytes), "banked write k");
    REQUIRE(ds4_gpu_tensor_write(dv, 0, v,
                (uint64_t)T_BANKS * vector_bytes), "banked write v");
    REQUIRE(ds4_gpu_tensor_write(dg, 0, g,
                (uint64_t)T_BANKS * vector_bytes), "banked write gate");
    REQUIRE(ds4_gpu_tensor_write(dbeta, 0, beta,
                (uint64_t)T_BANKS * T_HEAD * sizeof(float)),
            "banked write beta");
    REQUIRE(ds4_gpu_tensor_write(dbanks, 0, bank_ids, sizeof(bank_ids)),
            "banked write ids");
    REQUIRE((glm53 ? ds4_gpu_glm53_kda_decode_banks_tensor
                   : ds4_gpu_solar_kda_decode_banks_tensor)(
                    out, state_slab, bank_stride, 0u, q_offset, k_offset,
                    v_offset, dbanks, T_BANKS, T_BANKS,
                    dq, dk, dv, dg, dbeta,
                    fixed->q_weight, fixed->k_weight, fixed->v_weight,
                    fixed->decay, fixed->dt,
                    T_HEAD, T_DIM, T_CONV, -5.0f),
            "banked KDA kernel launch");
    REQUIRE(ds4_gpu_tensor_read(
                out, 0, gpu_out,
                (uint64_t)T_BANKS * vector_bytes),
            "read banked KDA output");
    compare("banked decode row 0", gpu_out, want_out,
            T_VECTOR, 3.0e-5, 3.0e-4);
    compare("banked decode row 1", gpu_out + T_VECTOR,
            want_out + T_VECTOR, T_VECTOR, 3.0e-5, 3.0e-4);

    for (uint32_t bank = 0; bank < T_BANKS; bank++) {
        REQUIRE(ds4_gpu_tensor_read(
                    state_slab, (uint64_t)bank * bank_stride,
                    bank_data, bank_stride),
                "read banked KDA state");
        char label[64];
        snprintf(label, sizeof(label), "%s bank %u recurrent state",
                 glm53 ? "GLM" : "Solar", bank);
        compare(label, bank_data, cpu[bank].state,
                T_STATE, 5.0e-5, 5.0e-4);
        snprintf(label, sizeof(label), "bank %u q conv state", bank);
        compare(label, bank_data + q_offset / sizeof(float),
                cpu[bank].q_conv, T_CONV_STATE, 1.0e-7, 1.0e-6);
        snprintf(label, sizeof(label), "bank %u k conv state", bank);
        compare(label, bank_data + k_offset / sizeof(float),
                cpu[bank].k_conv, T_CONV_STATE, 1.0e-7, 1.0e-6);
        snprintf(label, sizeof(label), "bank %u v conv state", bank);
        compare(label, bank_data + v_offset / sizeof(float),
                cpu[bank].v_conv, T_CONV_STATE, 1.0e-7, 1.0e-6);
    }

    ds4_gpu_tensor_free(dbanks);
    ds4_gpu_tensor_free(dbeta);
    ds4_gpu_tensor_free(dg);
    ds4_gpu_tensor_free(dv);
    ds4_gpu_tensor_free(dk);
    ds4_gpu_tensor_free(dq);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(state_slab);
    free(cpu);
    free(bank_data);
    free(want_out);
    free(gpu_out);
    free(beta);
    free(g);
    free(v);
    free(k);
    free(q);
}

int main(void) {
    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4_gpu_init failed\n");
        return 1;
    }

    float q_weight[T_CONV_STATE], k_weight[T_CONV_STATE], v_weight[T_CONV_STATE];
    float decay[T_HEAD], dt[T_VECTOR];
    float q[T_VECTOR], k[T_VECTOR], v[T_VECTOR], g[T_VECTOR], beta[T_HEAD];
    float gpu_out[T_VECTOR], cpu_out[T_VECTOR], first_out[T_VECTOR];
    float gpu_recurrent[T_STATE], gpu_conv[T_CONV_STATE];
    cpu_kda_state cpu_cache;
    memset(&cpu_cache, 0, sizeof(cpu_cache));
    make_fixed(q_weight, k_weight, v_weight, decay, dt);

    gpu_kda_inputs inputs;
    gpu_kda_state primary, forked;
    if (!gpu_inputs_alloc(&inputs) || !gpu_state_alloc(&primary) || !gpu_state_alloc(&forked)) {
        return 1;
    }
    REQUIRE(ds4_gpu_tensor_write(inputs.q_weight, 0, q_weight, sizeof(q_weight)), "write q weight");
    REQUIRE(ds4_gpu_tensor_write(inputs.k_weight, 0, k_weight, sizeof(k_weight)), "write k weight");
    REQUIRE(ds4_gpu_tensor_write(inputs.v_weight, 0, v_weight, sizeof(v_weight)), "write v weight");
    REQUIRE(ds4_gpu_tensor_write(inputs.decay, 0, decay, sizeof(decay)), "write decay scale");
    REQUIRE(ds4_gpu_tensor_write(inputs.dt, 0, dt, sizeof(dt)), "write dt bias");

    printf("== Solar Open 2 CUDA KDA decode ==\n");
    for (uint32_t token = 0; token < T_TOKENS; token++) {
        make_token(token, q, k, v, g, beta);
        REQUIRE(gpu_write_token(&inputs, q, k, v, g, beta), "write token");
        REQUIRE(gpu_step(&primary, &inputs, 0), "KDA kernel launch");
        REQUIRE(ds4_gpu_tensor_read(primary.out, 0, gpu_out, sizeof(gpu_out)), "read output");
        cpu_step(cpu_out, &cpu_cache, q, k, v, g, beta,
                 q_weight, k_weight, v_weight, decay, dt, 0);
        char label[64];
        snprintf(label, sizeof(label), "continuation token %u", token);
        compare(label, gpu_out, cpu_out, T_VECTOR, 3.0e-5, 3.0e-4);
        if (token == 0u) memcpy(first_out, gpu_out, sizeof(first_out));
    }

    REQUIRE(ds4_gpu_tensor_read(primary.state, 0, gpu_recurrent, sizeof(gpu_recurrent)),
            "read recurrent state");
    compare("final recurrent state", gpu_recurrent, cpu_cache.state, T_STATE, 5.0e-5, 5.0e-4);
    REQUIRE(ds4_gpu_tensor_read(primary.q_conv, 0, gpu_conv, sizeof(gpu_conv)), "read q conv");
    compare("final q conv state", gpu_conv, cpu_cache.q_conv, T_CONV_STATE, 1.0e-7, 1.0e-6);
    REQUIRE(ds4_gpu_tensor_read(primary.k_conv, 0, gpu_conv, sizeof(gpu_conv)), "read k conv");
    compare("final k conv state", gpu_conv, cpu_cache.k_conv, T_CONV_STATE, 1.0e-7, 1.0e-6);
    REQUIRE(ds4_gpu_tensor_read(primary.v_conv, 0, gpu_conv, sizeof(gpu_conv)), "read v conv");
    compare("final v conv state", gpu_conv, cpu_cache.v_conv, T_CONV_STATE, 1.0e-7, 1.0e-6);

    REQUIRE(gpu_reset_state(&primary), "state reset");
    make_token(0, q, k, v, g, beta);
    REQUIRE(gpu_write_token(&inputs, q, k, v, g, beta), "write reset token");
    REQUIRE(gpu_step(&primary, &inputs, 0), "reset decode");
    REQUIRE(ds4_gpu_tensor_read(primary.out, 0, gpu_out, sizeof(gpu_out)), "read reset output");
    compare("reset identity", gpu_out, first_out, T_VECTOR, 1.0e-7, 1.0e-6);

    REQUIRE(gpu_copy_state(&forked, &primary), "fork state copy");
    make_token(1, q, k, v, g, beta);
    REQUIRE(gpu_write_token(&inputs, q, k, v, g, beta), "write fork token");
    REQUIRE(gpu_step(&primary, &inputs, 0), "primary fork decode");
    REQUIRE(gpu_step(&forked, &inputs, 0), "copied fork decode");
    REQUIRE(ds4_gpu_tensor_read(primary.out, 0, gpu_out, sizeof(gpu_out)), "read primary fork");
    REQUIRE(ds4_gpu_tensor_read(forked.out, 0, cpu_out, sizeof(cpu_out)), "read copied fork");
    compare("fork output identity", gpu_out, cpu_out, T_VECTOR, 1.0e-7, 1.0e-6);
    REQUIRE(ds4_gpu_tensor_read(primary.state, 0, gpu_recurrent, sizeof(gpu_recurrent)),
            "read primary fork state");
    REQUIRE(ds4_gpu_tensor_read(forked.state, 0, cpu_cache.state, sizeof(cpu_cache.state)),
            "read copied fork state");
    compare("fork state identity", gpu_recurrent, cpu_cache.state, T_STATE, 1.0e-7, 1.0e-6);

    puts("== Solar Open 2 banked CUDA KDA decode ==");
    test_banked_decode(&inputs, q_weight, k_weight, v_weight, decay, dt, 0);

    puts("== GLM 5.3 banked CUDA KDA decode ==");
    test_banked_decode(&inputs, q_weight, k_weight, v_weight, decay, dt, 1);

    puts("== GLM 5.3 CUDA KDA decode ==");
    REQUIRE(gpu_reset_state(&primary), "GLM 5.3 state reset");
    memset(&cpu_cache, 0, sizeof(cpu_cache));
    for (uint32_t token = 0; token < T_TOKENS; token++) {
        make_token(token, q, k, v, g, beta);
        REQUIRE(gpu_write_token(&inputs, q, k, v, g, beta),
                "write GLM 5.3 token");
        REQUIRE(gpu_step(&primary, &inputs, 1), "GLM 5.3 KDA kernel launch");
        REQUIRE(ds4_gpu_tensor_read(primary.out, 0, gpu_out, sizeof(gpu_out)),
                "read GLM 5.3 output");
        cpu_step(cpu_out, &cpu_cache, q, k, v, g, beta,
                 q_weight, k_weight, v_weight, decay, dt, 1);
        char label[64];
        snprintf(label, sizeof(label), "GLM 5.3 token %u", token);
        compare(label, gpu_out, cpu_out, T_VECTOR, 3.0e-5, 3.0e-4);
    }
    REQUIRE(ds4_gpu_tensor_read(primary.state, 0, gpu_recurrent,
                                sizeof(gpu_recurrent)),
            "read GLM 5.3 recurrent state");
    compare("GLM 5.3 recurrent state", gpu_recurrent, cpu_cache.state,
            T_STATE, 5.0e-5, 5.0e-4);

    gpu_state_free(&forked);
    gpu_state_free(&primary);
    gpu_inputs_free(&inputs);
    ds4_gpu_cleanup();

    printf("%s\n", failures ? "KDA checks FAILED" : "all Solar/GLM KDA checks passed");
    return failures ? 1 : 0;
}
