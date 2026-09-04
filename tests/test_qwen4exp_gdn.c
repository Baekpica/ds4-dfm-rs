/* CUDA parity and state-lifecycle checks for Qwen3.8-Flash-Next's production
 * Gated DeltaNet geometry. Exercises the official four-slot convolution
 * cache, 48x128x128 recurrent state, arbitrary chunking, one-token decode,
 * control transforms, and gated per-head RMSNorm. */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    ROWS = 11,
    KEY_HEADS = 16,
    VALUE_HEADS = 48,
    HEAD_DIM = 128,
    KEY_DIM = KEY_HEADS * HEAD_DIM,
    VALUE_DIM = VALUE_HEADS * HEAD_DIM,
    CONV_DIM = 2 * KEY_DIM + VALUE_DIM,
    CONV_KERNEL = 4,
};

#define REQUIRE(condition, message) do {                                      \
    if (!(condition)) {                                                       \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (message), __FILE__, __LINE__); \
        exit(1);                                                              \
    }                                                                         \
} while (0)

static uint16_t f32_to_bf16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(bits >> 16u);
}

static float bf16_bits_to_f32(uint16_t value) {
    const uint32_t bits = (uint32_t)value << 16u;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint64_t fnv1a_bits(const float *values, uint64_t count) {
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (uint64_t i = 0; i < count; i++) {
        uint32_t bits;
        memcpy(&bits, &values[i], sizeof(bits));
        for (uint32_t b = 0; b < 4u; b++) {
            hash ^= (bits >> (8u * b)) & 0xffu;
            hash *= UINT64_C(0x100000001b3);
        }
    }
    return hash;
}

static float sigmoid_ref(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + expf(-x))
                     : expf(x) / (1.0f + expf(x));
}

static float softplus_ref(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

static void upload_f32(ds4_gpu_tensor *dst, const float *src, uint64_t count,
                       const char *what) {
    REQUIRE(ds4_gpu_tensor_write(dst, 0, src, count * sizeof(*src)), what);
}

static void download_f32(const ds4_gpu_tensor *src, float *dst,
                         uint64_t count, const char *what) {
    REQUIRE(ds4_gpu_tensor_read(src, 0, dst, count * sizeof(*dst)), what);
}

static void compare_f32(const char *name, const float *got, const float *want,
                        uint64_t count, float max_abs_limit,
                        double rel_rms_limit) {
    double err2 = 0.0;
    double ref2 = 0.0;
    float max_abs = 0.0f;
    uint64_t worst = 0u;
    for (uint64_t i = 0; i < count; i++) {
        const float error = fabsf(got[i] - want[i]);
        if (error > max_abs) {
            max_abs = error;
            worst = i;
        }
        err2 += (double)(got[i] - want[i]) * (got[i] - want[i]);
        ref2 += (double)want[i] * want[i];
    }
    const double rel_rms = sqrt(err2 / fmax(ref2, 1.0e-30));
    if (max_abs > max_abs_limit || rel_rms > rel_rms_limit) {
        fprintf(stderr,
                "FAIL: %s max abs %.9g at %llu (got %.9g want %.9g), "
                "rel RMS %.9g\n",
                name, max_abs, (unsigned long long)worst, got[worst],
                want[worst], rel_rms);
        exit(1);
    }
    printf("%-47s pass (max abs %.3g, rel RMS %.3g)\n",
           name, max_abs, rel_rms);
}

static void conv_reference(float *out, float *state, const float *input,
                           const float *weight, uint32_t rows) {
    for (uint32_t token = 0; token < rows; token++) {
        for (uint32_t channel = 0; channel < CONV_DIM; channel++) {
            const uint64_t sb = (uint64_t)channel * CONV_KERNEL;
            const uint64_t wb = (uint64_t)channel * CONV_KERNEL;
            float sum = state[sb + 1u] * weight[wb + 0u] +
                        state[sb + 2u] * weight[wb + 1u] +
                        state[sb + 3u] * weight[wb + 2u] +
                        input[(uint64_t)token * CONV_DIM + channel] *
                            weight[wb + 3u];
            out[(uint64_t)token * CONV_DIM + channel] =
                sum * sigmoid_ref(sum);
            state[sb + 0u] = state[sb + 1u];
            state[sb + 1u] = state[sb + 2u];
            state[sb + 2u] = state[sb + 3u];
            state[sb + 3u] = input[(uint64_t)token * CONV_DIM + channel];
        }
    }
}

static void recurrent_reference(float *out, float *state,
                                const float *mixed_qkv, const float *beta,
                                const float *g, uint32_t rows) {
    const uint32_t repeat = VALUE_HEADS / KEY_HEADS;
    for (uint32_t token = 0; token < rows; token++) {
        const uint64_t rb = (uint64_t)token * CONV_DIM;
        for (uint32_t vh = 0; vh < VALUE_HEADS; vh++) {
            const uint32_t kh = vh / repeat;
            const float *q = mixed_qkv + rb + (uint64_t)kh * HEAD_DIM;
            const float *k = mixed_qkv + rb + KEY_DIM +
                             (uint64_t)kh * HEAD_DIM;
            const float *value = mixed_qkv + rb + 2u * KEY_DIM +
                                 (uint64_t)vh * HEAD_DIM;
            float q2 = 0.0f;
            float k2 = 0.0f;
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                q2 += q[d] * q[d];
                k2 += k[d] * k[d];
            }
            const float q_inv = 1.0f / sqrtf(q2 + 1.0e-6f) /
                                sqrtf((float)HEAD_DIM);
            const float k_inv = 1.0f / sqrtf(k2 + 1.0e-6f);
            const float decay = expf(g[(uint64_t)token * VALUE_HEADS + vh]);
            const float bt = beta[(uint64_t)token * VALUE_HEADS + vh];
            const uint64_t sh = (uint64_t)vh * HEAD_DIM * HEAD_DIM;
            for (uint32_t v = 0; v < HEAD_DIM; v++) {
                float kv_mem = 0.0f;
                for (uint32_t d = 0; d < HEAD_DIM; d++) {
                    const uint64_t at = sh + (uint64_t)d * HEAD_DIM + v;
                    state[at] *= decay;
                    kv_mem += state[at] * (k[d] * k_inv);
                }
                const float delta = (value[v] - kv_mem) * bt;
                float result = 0.0f;
                for (uint32_t d = 0; d < HEAD_DIM; d++) {
                    const uint64_t at = sh + (uint64_t)d * HEAD_DIM + v;
                    state[at] += (k[d] * k_inv) * delta;
                    result += state[at] * (q[d] * q_inv);
                }
                out[((uint64_t)token * VALUE_HEADS + vh) * HEAD_DIM + v] =
                    result;
            }
        }
    }
}

static void gated_norm_reference(float *out, const float *core,
                                 const float *z, const float *weight,
                                 uint32_t rows) {
    for (uint64_t hr = 0; hr < (uint64_t)rows * VALUE_HEADS; hr++) {
        const uint64_t base = hr * HEAD_DIM;
        float sum = 0.0f;
        for (uint32_t d = 0; d < HEAD_DIM; d++)
            sum += core[base + d] * core[base + d];
        const float scale = 1.0f /
            sqrtf(sum / (float)HEAD_DIM + 1.0e-6f);
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            const float gate = z[base + d];
            out[base + d] = core[base + d] * scale * weight[d] *
                            sigmoid_ref(gate);
        }
    }
}

static void run_conv_chunks(ds4_gpu_tensor *out, ds4_gpu_tensor *state,
                            ds4_gpu_tensor *input, const uint32_t *chunks,
                            uint32_t n_chunks, const void *model_map,
                            uint64_t model_size, uint64_t weight_offset) {
    uint32_t row = 0u;
    for (uint32_t c = 0; c < n_chunks; c++) {
        const uint64_t offset = (uint64_t)row * CONV_DIM * sizeof(float);
        const uint64_t bytes =
            (uint64_t)chunks[c] * CONV_DIM * sizeof(float);
        ds4_gpu_tensor *iv = ds4_gpu_tensor_view(input, offset, bytes);
        ds4_gpu_tensor *ov = ds4_gpu_tensor_view(out, offset, bytes);
        REQUIRE(iv && ov, "GDN convolution chunk views");
        REQUIRE(ds4_gpu_qwen4exp_gdn_conv_tensor(
                    ov, state, iv, model_map, model_size, weight_offset,
                    chunks[c], CONV_DIM, CONV_KERNEL),
                "GDN convolution chunk launch");
        ds4_gpu_tensor_free(ov);
        ds4_gpu_tensor_free(iv);
        row += chunks[c];
    }
    REQUIRE(row == ROWS, "GDN convolution chunks cover sequence");
}

static void run_recurrent_chunks(ds4_gpu_tensor *out, ds4_gpu_tensor *state,
                                 ds4_gpu_tensor *mixed,
                                 ds4_gpu_tensor *beta, ds4_gpu_tensor *g,
                                 const uint32_t *chunks, uint32_t n_chunks) {
    uint32_t row = 0u;
    for (uint32_t c = 0; c < n_chunks; c++) {
        const uint64_t mixed_offset =
            (uint64_t)row * CONV_DIM * sizeof(float);
        const uint64_t mixed_bytes =
            (uint64_t)chunks[c] * CONV_DIM * sizeof(float);
        const uint64_t control_offset =
            (uint64_t)row * VALUE_HEADS * sizeof(float);
        const uint64_t control_bytes =
            (uint64_t)chunks[c] * VALUE_HEADS * sizeof(float);
        const uint64_t out_offset =
            (uint64_t)row * VALUE_DIM * sizeof(float);
        const uint64_t out_bytes =
            (uint64_t)chunks[c] * VALUE_DIM * sizeof(float);
        ds4_gpu_tensor *mv = ds4_gpu_tensor_view(mixed, mixed_offset,
                                                 mixed_bytes);
        ds4_gpu_tensor *bv = ds4_gpu_tensor_view(beta, control_offset,
                                                 control_bytes);
        ds4_gpu_tensor *gv = ds4_gpu_tensor_view(g, control_offset,
                                                 control_bytes);
        ds4_gpu_tensor *ov = ds4_gpu_tensor_view(out, out_offset, out_bytes);
        REQUIRE(mv && bv && gv && ov, "GDN recurrent chunk views");
        REQUIRE(ds4_gpu_qwen4exp_gdn_recurrent_tensor(
                    ov, state, mv, bv, gv, chunks[c], KEY_HEADS,
                    VALUE_HEADS, HEAD_DIM),
                "GDN recurrent chunk launch");
        ds4_gpu_tensor_free(ov);
        ds4_gpu_tensor_free(gv);
        ds4_gpu_tensor_free(bv);
        ds4_gpu_tensor_free(mv);
        row += chunks[c];
    }
    REQUIRE(row == ROWS, "GDN recurrent chunks cover sequence");
}

static void profile_gdn_recurrent(void) {
    enum { profile_rows = 8025 };
    const uint64_t mixed_count = (uint64_t)profile_rows * CONV_DIM;
    const uint64_t control_count = (uint64_t)profile_rows * VALUE_HEADS;
    const uint64_t state_count =
        (uint64_t)VALUE_HEADS * HEAD_DIM * HEAD_DIM;
    const uint64_t out_count = (uint64_t)profile_rows * VALUE_DIM;
    ds4_gpu_tensor *mixed = ds4_gpu_tensor_alloc(mixed_count * sizeof(float));
    ds4_gpu_tensor *beta = ds4_gpu_tensor_alloc(control_count * sizeof(float));
    ds4_gpu_tensor *g = ds4_gpu_tensor_alloc(control_count * sizeof(float));
    ds4_gpu_tensor *state = ds4_gpu_tensor_alloc(state_count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    REQUIRE(mixed && beta && g && state && out,
            "GDN profile GPU allocation");
    REQUIRE(ds4_gpu_tensor_fill_f32(mixed, 0.03125f, mixed_count) &&
                ds4_gpu_tensor_fill_f32(beta, 0.25f, control_count) &&
                ds4_gpu_tensor_fill_f32(g, -0.01f, control_count) &&
                ds4_gpu_tensor_fill_f32(state, 0.001f, state_count),
            "GDN profile input initialization");
    REQUIRE(ds4_gpu_qwen4exp_gdn_recurrent_tensor(
                out, state, mixed, beta, g, profile_rows, KEY_HEADS,
                VALUE_HEADS, HEAD_DIM),
            "GDN production profile launch");
    REQUIRE(ds4_gpu_synchronize(), "GDN production profile sync");
    float edge[2] = {0.0f, 0.0f};
    REQUIRE(ds4_gpu_tensor_read(out, 0, &edge[0], sizeof(float)) &&
                ds4_gpu_tensor_read(out, (out_count - 1u) * sizeof(float),
                                    &edge[1], sizeof(float)) &&
                isfinite(edge[0]) && isfinite(edge[1]),
            "GDN production profile output");
    printf("NCU Qwen GDN: rows=%u edge=%.6g/%.6g\n",
           profile_rows, edge[0], edge[1]);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(state);
    ds4_gpu_tensor_free(g);
    ds4_gpu_tensor_free(beta);
    ds4_gpu_tensor_free(mixed);
}

int main(void) {
    if (getenv("DS4_QWEN_PROFILE_GDN_RECURRENT")) {
        REQUIRE(ds4_gpu_init(), "CUDA init");
        profile_gdn_recurrent();
        ds4_gpu_cleanup();
        return 0;
    }
    const uint64_t conv_count = (uint64_t)ROWS * CONV_DIM;
    const uint64_t conv_weight_count = (uint64_t)CONV_DIM * CONV_KERNEL;
    const uint64_t conv_state_count = conv_weight_count;
    const uint64_t control_count = (uint64_t)ROWS * VALUE_HEADS;
    const uint64_t recurrent_count =
        (uint64_t)VALUE_HEADS * HEAD_DIM * HEAD_DIM;
    const uint64_t core_count = (uint64_t)ROWS * VALUE_DIM;

    const uint64_t conv_offset = 4096u;
    const uint64_t conv_bytes = conv_weight_count * sizeof(uint16_t);
    const uint64_t a_log_offset =
        (conv_offset + conv_bytes + 255u) & ~255ull;
    const uint64_t dt_bias_offset = a_log_offset + VALUE_HEADS * sizeof(float);
    const uint64_t norm_offset = dt_bias_offset + VALUE_HEADS * sizeof(float);
    const uint64_t model_size =
        (norm_offset + HEAD_DIM * sizeof(float) + 4095u) & ~4095ull;
    void *model_map = NULL;
    REQUIRE(posix_memalign(&model_map, 4096u, (size_t)model_size) == 0,
            "aligned GDN model fixture");
    memset(model_map, 0, (size_t)model_size);
    uint16_t *conv_bits =
        (uint16_t *)((unsigned char *)model_map + conv_offset);
    float *a_log = (float *)((unsigned char *)model_map + a_log_offset);
    float *dt_bias = (float *)((unsigned char *)model_map + dt_bias_offset);
    float *norm_weight = (float *)((unsigned char *)model_map + norm_offset);

    float *conv_weight = malloc(conv_weight_count * sizeof(float));
    float *conv_input = malloc(conv_count * sizeof(float));
    float *conv_want = malloc(conv_count * sizeof(float));
    float *conv_got = malloc(conv_count * sizeof(float));
    float *conv_chunk = malloc(conv_count * sizeof(float));
    float *conv_state_initial = malloc(conv_state_count * sizeof(float));
    float *conv_state_want = malloc(conv_state_count * sizeof(float));
    float *conv_state_got = malloc(conv_state_count * sizeof(float));
    float *a = malloc(control_count * sizeof(float));
    float *b = malloc(control_count * sizeof(float));
    float *beta_want = malloc(control_count * sizeof(float));
    float *g_want = malloc(control_count * sizeof(float));
    float *control_got = malloc(control_count * sizeof(float));
    float *mixed = malloc(conv_count * sizeof(float));
    float *state_initial = malloc(recurrent_count * sizeof(float));
    float *state_want = malloc(recurrent_count * sizeof(float));
    float *state_got = malloc(recurrent_count * sizeof(float));
    float *core_want = malloc(core_count * sizeof(float));
    float *core_got = malloc(core_count * sizeof(float));
    float *core_chunk = malloc(core_count * sizeof(float));
    float *mixed1 = malloc(conv_count * sizeof(float));
    float *beta1 = malloc(control_count * sizeof(float));
    float *g1 = malloc(control_count * sizeof(float));
    float *state1_initial = malloc(recurrent_count * sizeof(float));
    float *state1_want = malloc(recurrent_count * sizeof(float));
    float *state1_got = malloc(recurrent_count * sizeof(float));
    float *core1_want = malloc(core_count * sizeof(float));
    float *core1_got = malloc(core_count * sizeof(float));
    float *z = malloc(core_count * sizeof(float));
    float *norm_want = malloc(core_count * sizeof(float));
    float *norm_got = malloc(core_count * sizeof(float));
    REQUIRE(conv_weight && conv_input && conv_want && conv_got && conv_chunk &&
            conv_state_initial && conv_state_want && conv_state_got && a && b &&
            beta_want && g_want && control_got && mixed && state_initial &&
            state_want && state_got && core_want && core_got && core_chunk &&
            mixed1 && beta1 && g1 && state1_initial && state1_want &&
            state1_got && core1_want && core1_got && z && norm_want &&
            norm_got, "GDN host allocations");

    for (uint64_t i = 0; i < conv_weight_count; i++) {
        const float source = 0.16f * sinf((float)(i + 1u) * 0.017f) +
            (float)((int)(i % 9u) - 4) * 0.004f;
        conv_bits[i] = f32_to_bf16_bits(source);
        conv_weight[i] = bf16_bits_to_f32(conv_bits[i]);
    }
    for (uint32_t h = 0; h < VALUE_HEADS; h++) {
        a_log[h] = logf(0.015f + 0.019f * (float)(h + 1u));
        dt_bias[h] = -2.2f + 0.071f * (float)h;
    }
    for (uint32_t d = 0; d < HEAD_DIM; d++)
        norm_weight[d] = 0.78f + 0.003f * (float)(d % 41u);
    for (uint64_t i = 0; i < conv_count; i++) {
        conv_input[i] = 0.31f * sinf((float)(i + 7u) * 0.0037f) +
                        0.08f * cosf((float)(i + 19u) * 0.011f);
        mixed[i] = 0.19f * sinf((float)(i + 3u) * 0.0051f) +
                   0.07f * cosf((float)(i + 29u) * 0.0093f);
    }
    for (uint64_t i = 0; i < conv_state_count; i++)
        conv_state_initial[i] =
            0.025f * sinf((float)(i + 13u) * 0.012f);
    memcpy(conv_state_want, conv_state_initial,
           conv_state_count * sizeof(float));
    conv_reference(conv_want, conv_state_want, conv_input, conv_weight, ROWS);
    for (uint64_t i = 0; i < control_count; i++) {
        const uint32_t h = (uint32_t)(i % VALUE_HEADS);
        a[i] = -1.8f + 0.013f * (float)(i % 211u);
        b[i] = 1.4f * sinf((float)(i + 5u) * 0.027f);
        beta_want[i] = sigmoid_ref(b[i]);
        g_want[i] = -expf(a_log[h]) * softplus_ref(a[i] + dt_bias[h]);
    }
    for (uint64_t i = 0; i < recurrent_count; i++)
        state_initial[i] =
            0.0007f * sinf((float)(i + 17u) * 0.0017f);
    for (uint64_t i = 0; i < conv_count; i++)
        mixed1[i] = 0.17f * cosf((float)(i + 11u) * 0.0043f) -
                    0.06f * sinf((float)(i + 37u) * 0.0081f);
    for (uint64_t i = 0; i < control_count; i++) {
        beta1[i] = sigmoid_ref(-0.9f + 0.021f * (float)(i % 97u));
        g1[i] = -0.008f - 0.0003f * (float)(i % VALUE_HEADS);
    }
    for (uint64_t i = 0; i < recurrent_count; i++)
        state1_initial[i] =
            0.0005f * cosf((float)(i + 29u) * 0.0013f);
    memcpy(state_want, state_initial, recurrent_count * sizeof(float));
    recurrent_reference(core_want, state_want, mixed, beta_want, g_want,
                        ROWS);
    for (uint64_t i = 0; i < core_count; i++)
        z[i] = 0.55f * cosf((float)(i + 23u) * 0.0049f) - 0.1f;
    gated_norm_reference(norm_want, core_want, z, norm_weight, ROWS);

    REQUIRE(ds4_gpu_init(), "CUDA init");
    REQUIRE(unsetenv("DS4_CUDA_COPY_MODEL") == 0,
            "disable whole-map fixture copy");
    REQUIRE(ds4_gpu_set_model_map(model_map, model_size),
            "register GDN model fixture");
    puts("== Qwen3.8-Flash-Next Gated DeltaNet ==");
    printf("production state: %.3f MiB recurrent + %.3f MiB convolution\n",
           (double)(recurrent_count * sizeof(float)) / 1048576.0,
           (double)(conv_state_count * sizeof(float)) / 1048576.0);

    ds4_gpu_tensor *dconv_in = ds4_gpu_tensor_alloc(conv_count * sizeof(float));
    ds4_gpu_tensor *dconv_out = ds4_gpu_tensor_alloc(conv_count * sizeof(float));
    ds4_gpu_tensor *dconv_state =
        ds4_gpu_tensor_alloc(conv_state_count * sizeof(float));
    ds4_gpu_tensor *da = ds4_gpu_tensor_alloc(control_count * sizeof(float));
    ds4_gpu_tensor *db = ds4_gpu_tensor_alloc(control_count * sizeof(float));
    ds4_gpu_tensor *dbeta = ds4_gpu_tensor_alloc(control_count * sizeof(float));
    ds4_gpu_tensor *dg = ds4_gpu_tensor_alloc(control_count * sizeof(float));
    ds4_gpu_tensor *dmixed = ds4_gpu_tensor_alloc(conv_count * sizeof(float));
    ds4_gpu_tensor *dstate =
        ds4_gpu_tensor_alloc(recurrent_count * sizeof(float));
    ds4_gpu_tensor *dcore = ds4_gpu_tensor_alloc(core_count * sizeof(float));
    ds4_gpu_tensor *dmixed1 = ds4_gpu_tensor_alloc(conv_count * sizeof(float));
    ds4_gpu_tensor *dbeta1 = ds4_gpu_tensor_alloc(control_count * sizeof(float));
    ds4_gpu_tensor *dg1 = ds4_gpu_tensor_alloc(control_count * sizeof(float));
    ds4_gpu_tensor *dstate1 =
        ds4_gpu_tensor_alloc(recurrent_count * sizeof(float));
    ds4_gpu_tensor *dcore1 = ds4_gpu_tensor_alloc(core_count * sizeof(float));
    ds4_gpu_tensor *dz = ds4_gpu_tensor_alloc(core_count * sizeof(float));
    ds4_gpu_tensor *dnorm = ds4_gpu_tensor_alloc(core_count * sizeof(float));
    REQUIRE(dconv_in && dconv_out && dconv_state && da && db && dbeta && dg &&
            dmixed && dstate && dcore && dmixed1 && dbeta1 && dg1 && dstate1 &&
            dcore1 && dz && dnorm, "GDN GPU allocations");
    upload_f32(dconv_in, conv_input, conv_count, "GDN conv input upload");
    upload_f32(da, a, control_count, "GDN a upload");
    upload_f32(db, b, control_count, "GDN b upload");
    upload_f32(dmixed, mixed, conv_count, "GDN mixed QKV upload");
    upload_f32(dmixed1, mixed1, conv_count, "GDN bank1 mixed QKV upload");
    upload_f32(dbeta1, beta1, control_count, "GDN bank1 beta upload");
    upload_f32(dg1, g1, control_count, "GDN bank1 decay upload");
    upload_f32(dz, z, core_count, "GDN z upload");

    upload_f32(dconv_state, conv_state_initial, conv_state_count,
               "GDN conv state upload");
    REQUIRE(ds4_gpu_qwen4exp_gdn_conv_tensor(
                dconv_out, dconv_state, dconv_in, model_map, model_size,
                conv_offset, ROWS, CONV_DIM, CONV_KERNEL),
            "GDN full convolution launch");
    download_f32(dconv_out, conv_got, conv_count, "GDN conv output download");
    download_f32(dconv_state, conv_state_got, conv_state_count,
                 "GDN conv state download");
    compare_f32("GDN convolution full prefill", conv_got, conv_want,
                conv_count, 3.0e-6f, 3.0e-6);
    compare_f32("GDN official four-slot convolution state", conv_state_got,
                conv_state_want, conv_state_count, 0.0f, 0.0);

    const uint32_t chunks[] = {2u, 1u, 5u, 3u};
    upload_f32(dconv_state, conv_state_initial, conv_state_count,
               "GDN chunk conv state reset");
    run_conv_chunks(dconv_out, dconv_state, dconv_in, chunks, 4u,
                    model_map, model_size, conv_offset);
    download_f32(dconv_out, conv_chunk, conv_count,
                 "GDN chunk conv output download");
    compare_f32("GDN convolution arbitrary-chunk parity", conv_chunk,
                conv_got, conv_count, 0.0f, 0.0);
    uint32_t decode_chunks[ROWS];
    for (uint32_t i = 0; i < ROWS; i++) decode_chunks[i] = 1u;
    upload_f32(dconv_state, conv_state_initial, conv_state_count,
               "GDN decode conv state reset");
    run_conv_chunks(dconv_out, dconv_state, dconv_in, decode_chunks, ROWS,
                    model_map, model_size, conv_offset);
    download_f32(dconv_out, conv_chunk, conv_count,
                 "GDN decode conv output download");
    compare_f32("GDN convolution one-token decode parity", conv_chunk,
                conv_got, conv_count, 0.0f, 0.0);

    REQUIRE(ds4_gpu_qwen4exp_gdn_controls_tensor(
                dbeta, dg, db, da, model_map, model_size,
                a_log_offset, dt_bias_offset, ROWS, VALUE_HEADS),
            "GDN control launch");
    download_f32(dbeta, control_got, control_count, "GDN beta download");
    compare_f32("GDN beta = sigmoid(b)", control_got, beta_want,
                control_count, 3.0e-7f, 3.0e-7);
    download_f32(dg, control_got, control_count, "GDN decay download");
    compare_f32("GDN g = -exp(A_log)*softplus(a+dt)", control_got, g_want,
                control_count, 3.0e-6f, 3.0e-6);
    upload_f32(dbeta, b, control_count, "GDN in-place b reset");
    upload_f32(dg, a, control_count, "GDN in-place a reset");
    REQUIRE(ds4_gpu_qwen4exp_gdn_controls_tensor(
                dbeta, dg, dbeta, dg, model_map, model_size,
                a_log_offset, dt_bias_offset, ROWS, VALUE_HEADS),
            "GDN in-place control launch");
    download_f32(dbeta, control_got, control_count, "GDN in-place beta read");
    compare_f32("GDN in-place control scratch reuse", control_got, beta_want,
                control_count, 3.0e-7f, 3.0e-7);

    /* Restore both transformed controls for the recurrent tests. */
    REQUIRE(ds4_gpu_qwen4exp_gdn_controls_tensor(
                dbeta, dg, db, da, model_map, model_size,
                a_log_offset, dt_bias_offset, ROWS, VALUE_HEADS),
            "GDN control restore");
    upload_f32(dstate, state_initial, recurrent_count,
               "GDN recurrent state upload");
    REQUIRE(ds4_gpu_qwen4exp_gdn_recurrent_tensor(
                dcore, dstate, dmixed, dbeta, dg, ROWS, KEY_HEADS,
                VALUE_HEADS, HEAD_DIM),
            "GDN full recurrent launch");
    download_f32(dcore, core_got, core_count, "GDN core download");
    download_f32(dstate, state_got, recurrent_count, "GDN state download");
    compare_f32("GDN recurrent full prefill output", core_got, core_want,
                core_count, 5.0e-5f, 2.0e-4);
    compare_f32("GDN recurrent full final state", state_got, state_want,
                recurrent_count, 5.0e-5f, 2.0e-4);
    /* Cross-build digest: a kernel rewrite that keeps the arithmetic order
     * reproduces these bits exactly; compare against a previous binary. */
    printf("%-48s output %016llx state %016llx\n",
           "GDN recurrent bit digest",
           (unsigned long long)fnv1a_bits(core_got, core_count),
           (unsigned long long)fnv1a_bits(state_got, recurrent_count));

    upload_f32(dstate1, state1_initial, recurrent_count,
               "GDN bank1 scalar state upload");
    REQUIRE(ds4_gpu_qwen4exp_gdn_recurrent_tensor(
                dcore1, dstate1, dmixed1, dbeta1, dg1, ROWS, KEY_HEADS,
                VALUE_HEADS, HEAD_DIM),
            "GDN bank1 scalar recurrent launch");
    download_f32(dcore1, core1_want, core_count,
                 "GDN bank1 scalar output download");
    download_f32(dstate1, state1_want, recurrent_count,
                 "GDN bank1 scalar state download");

    upload_f32(dstate, state_initial, recurrent_count,
               "GDN bank0 paired state reset");
    upload_f32(dstate1, state1_initial, recurrent_count,
               "GDN bank1 paired state reset");
    REQUIRE(ds4_gpu_qwen4exp_gdn_recurrent_bank2_tensor(
                dcore, dstate, dmixed, dbeta, dg,
                dcore1, dstate1, dmixed1, dbeta1, dg1,
                ROWS, KEY_HEADS, VALUE_HEADS, HEAD_DIM),
            "GDN paired-bank recurrent launch");
    download_f32(dcore, core_chunk, core_count,
                 "GDN bank0 paired output download");
    download_f32(dstate, state_want, recurrent_count,
                 "GDN bank0 paired state download");
    download_f32(dcore1, core1_got, core_count,
                 "GDN bank1 paired output download");
    download_f32(dstate1, state1_got, recurrent_count,
                 "GDN bank1 paired state download");
    compare_f32("GDN paired bank0 output bit parity", core_chunk, core_got,
                core_count, 0.0f, 0.0);
    compare_f32("GDN paired bank0 state bit parity", state_want, state_got,
                recurrent_count, 0.0f, 0.0);
    compare_f32("GDN paired bank1 output bit parity", core1_got, core1_want,
                core_count, 0.0f, 0.0);
    compare_f32("GDN paired bank1 state bit parity", state1_got, state1_want,
                recurrent_count, 0.0f, 0.0);

    upload_f32(dstate, state_initial, recurrent_count,
               "GDN recurrent chunk state reset");
    run_recurrent_chunks(dcore, dstate, dmixed, dbeta, dg, chunks, 4u);
    download_f32(dcore, core_chunk, core_count,
                 "GDN recurrent chunk output download");
    compare_f32("GDN recurrent arbitrary-chunk bit parity", core_chunk,
                core_got, core_count, 0.0f, 0.0);
    upload_f32(dstate, state_initial, recurrent_count,
               "GDN recurrent decode state reset");
    run_recurrent_chunks(dcore, dstate, dmixed, dbeta, dg,
                         decode_chunks, ROWS);
    download_f32(dcore, core_chunk, core_count,
                 "GDN recurrent decode output download");
    compare_f32("GDN recurrent one-token decode bit parity", core_chunk,
                core_got, core_count, 0.0f, 0.0);

    /* Use the reference core here so this gate isolates the norm/gate math. */
    upload_f32(dcore, core_want, core_count, "GDN norm core upload");
    REQUIRE(ds4_gpu_qwen4exp_gdn_gated_rms_norm_tensor(
                dnorm, dcore, dz, model_map, model_size, norm_offset,
                ROWS, VALUE_HEADS, HEAD_DIM, 1.0e-6f),
            "GDN gated RMSNorm launch");
    download_f32(dnorm, norm_got, core_count, "GDN norm download");
    compare_f32("GDN per-head RMSNorm then sigmoid(z)", norm_got, norm_want,
                core_count, 8.0e-6f, 8.0e-6);

    REQUIRE(!ds4_gpu_qwen4exp_gdn_conv_tensor(
                dconv_in, dconv_state, dconv_in, model_map, model_size,
                conv_offset, ROWS, CONV_DIM, CONV_KERNEL),
            "GDN convolution rejects output/input alias");
    REQUIRE(!ds4_gpu_qwen4exp_gdn_recurrent_tensor(
                dmixed, dstate, dmixed, dbeta, dg, ROWS, KEY_HEADS,
                VALUE_HEADS, HEAD_DIM),
            "GDN recurrent rejects output/input alias");
    REQUIRE(!ds4_gpu_qwen4exp_gdn_recurrent_bank2_tensor(
                dcore, dstate, dmixed, dbeta, dg,
                dcore1, dstate, dmixed1, dbeta1, dg1,
                ROWS, KEY_HEADS, VALUE_HEADS, HEAD_DIM),
            "GDN paired recurrent rejects shared state");
    REQUIRE(!ds4_gpu_qwen4exp_gdn_controls_tensor(
                dbeta, dbeta, db, da, model_map, model_size,
                a_log_offset, dt_bias_offset, ROWS, VALUE_HEADS),
            "GDN controls reject aliased outputs");

    REQUIRE(ds4_gpu_synchronize(), "final GDN synchronization");
    ds4_gpu_tensor_free(dnorm);
    ds4_gpu_tensor_free(dz);
    ds4_gpu_tensor_free(dcore1);
    ds4_gpu_tensor_free(dstate1);
    ds4_gpu_tensor_free(dg1);
    ds4_gpu_tensor_free(dbeta1);
    ds4_gpu_tensor_free(dmixed1);
    ds4_gpu_tensor_free(dcore);
    ds4_gpu_tensor_free(dstate);
    ds4_gpu_tensor_free(dmixed);
    ds4_gpu_tensor_free(dg);
    ds4_gpu_tensor_free(dbeta);
    ds4_gpu_tensor_free(db);
    ds4_gpu_tensor_free(da);
    ds4_gpu_tensor_free(dconv_state);
    ds4_gpu_tensor_free(dconv_out);
    ds4_gpu_tensor_free(dconv_in);
    ds4_gpu_unregister_model_map(model_map);
    ds4_gpu_cleanup();

    free(norm_got); free(norm_want); free(z);
    free(core1_got); free(core1_want); free(state1_got); free(state1_want);
    free(state1_initial); free(g1); free(beta1); free(mixed1);
    free(core_chunk); free(core_got); free(core_want);
    free(state_got); free(state_want); free(state_initial);
    free(mixed); free(control_got); free(g_want); free(beta_want);
    free(b); free(a); free(conv_state_got); free(conv_state_want);
    free(conv_state_initial); free(conv_chunk); free(conv_got);
    free(conv_want); free(conv_input); free(conv_weight);
    free(model_map);
    puts("all Qwen3.8 Gated DeltaNet checks passed");
    return 0;
}
