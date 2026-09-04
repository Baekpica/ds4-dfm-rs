/* Projection-to-residual integration check for Qwen4Exp hyper-connections. */
#include "../ds4.c"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    ROWS = 2,
    HIDDEN = 2560,
    HC = 4,
    WIDTH = HIDDEN * HC,
    LOW = 320,
};

#define REQUIRE(c, m) do {                                                     \
    if (!(c)) { fprintf(stderr, "FAIL: %s (%s:%d)\n", m, __FILE__, __LINE__); exit(1); } \
} while (0)

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(bits >> 16u);
}

static uint64_t align4k(uint64_t value) {
    return (value + 4095u) & ~UINT64_C(4095);
}

static ds4_tensor make_weight(uint64_t *cursor, uint64_t d0, uint64_t d1) {
    ds4_tensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.ndim = d1 ? 2u : 1u;
    tensor.dim[0] = d0;
    tensor.dim[1] = d1;
    tensor.type = d1 ? DS4_TENSOR_BF16 : DS4_TENSOR_F32;
    tensor.elements = d0 * (d1 ? d1 : 1u);
    tensor.bytes = tensor.elements * (d1 ? sizeof(uint16_t) : sizeof(float));
    tensor.abs_offset = tensor.rel_offset = align4k(*cursor);
    *cursor = tensor.abs_offset + tensor.bytes;
    return tensor;
}

static float sigmoidf_ref(float value) {
    return 1.0f / (1.0f + expf(-value));
}

static float bf16_to_float(uint16_t bits) {
    const uint32_t wide = (uint32_t)bits << 16u;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

static float bf16_round(float value) {
    return bf16_to_float(bf16(value));
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Batched-path oracle for one row: identity-like mix_down (low = BF16 of
 * the first LOW normalised values, over hc_count), SiLU, BF16 rounding of
 * the low-rank rows (what the mix_up GEMM consumes), a double-precision
 * mix_up dot against an arbitrary BF16 weight, then the sigmoid mix on the
 * F32 normalised rows. */
static void batched_reference_row(
        const float *input_row, const uint16_t *up_data, float *mixed_out) {
    float normed[WIDTH];
    for (uint32_t lane = 0; lane < HC; lane++) {
        const uint64_t base = (uint64_t)lane * HIDDEN;
        double sum = 0.0;
        for (uint32_t d = 0; d < HIDDEN; d++)
            sum += (double)input_row[base + d] * input_row[base + d];
        const float scale = 1.0f / sqrtf((float)(sum / HIDDEN) + DS4_RMS_EPS);
        for (uint32_t d = 0; d < HIDDEN; d++)
            normed[base + d] = input_row[base + d] * scale;
    }
    float low_b[LOW];
    for (uint32_t i = 0; i < LOW; i++) {
        const float projected = bf16_round(normed[i]) / HC;
        low_b[i] = bf16_round(projected * sigmoidf_ref(projected));
    }
    for (uint32_t d = 0; d < HIDDEN; d++) {
        double mixed = 0.0;
        for (uint32_t source = 0; source < HC; source++) {
            const uint32_t at = source * HIDDEN + d;
            double logit = 0.0;
            for (uint32_t k = 0; k < LOW; k++)
                logit += (double)low_b[k] * bf16_to_float(up_data[(uint64_t)at * LOW + k]);
            mixed += (1.0 / (1.0 + exp(-logit))) * (double)normed[at];
        }
        mixed_out[d] = (float)(mixed / HC);
    }
}

int main(void) {
    g_ds4_shape = DS4_SHAPE_QWEN38_FLASH_NEXT;
    uint64_t cursor = 4096u;
    ds4_tensor norm = make_weight(&cursor, WIDTH, 0u);
    ds4_tensor down = make_weight(&cursor, WIDTH, LOW);
    ds4_tensor up = make_weight(&cursor, LOW, WIDTH);
    ds4_tensor inject = make_weight(&cursor, WIDTH, HC);
    ds4_tensor dense = make_weight(&cursor, WIDTH, LOW);
    ds4_tensor up_dense = make_weight(&cursor, LOW, WIDTH);
    const uint64_t model_size = align4k(cursor);
    unsigned char *map = NULL;
    REQUIRE(posix_memalign((void **)&map, 4096u, (size_t)model_size) == 0,
            "model fixture allocation");
    memset(map, 0, (size_t)model_size);

    uint16_t *down_data = (uint16_t *)(map + down.abs_offset);
    uint16_t *up_data = (uint16_t *)(map + up.abs_offset);
    uint16_t *inject_data = (uint16_t *)(map + inject.abs_offset);
    uint16_t *dense_data = (uint16_t *)(map + dense.abs_offset);
    for (uint32_t low = 0; low < LOW; low++)
        down_data[(uint64_t)low * WIDTH + low] = bf16(1.0f);
    for (uint32_t out = 0; out < WIDTH; out++)
        up_data[(uint64_t)out * LOW + out % LOW] = bf16(1.0f);
    for (uint32_t lane = 0; lane < HC; lane++)
        inject_data[(uint64_t)lane * WIDTH + (uint64_t)lane * HIDDEN] = bf16(1.0f);
    for (uint64_t i = 0u; i < dense.elements; i++)
        dense_data[i] = bf16((float)((int)(i % 29u) - 14) * 0.001f);
    uint16_t *up_dense_data = (uint16_t *)(map + up_dense.abs_offset);
    {
        uint32_t state = 0x9e3779b9u;
        for (uint64_t i = 0u; i < up_dense.elements; i++) {
            state = state * 1664525u + 1013904223u;
            up_dense_data[i] = bf16(((float)(state >> 8) / 16777216.0f - 0.5f) * 0.1f);
        }
    }

    ds4_model model;
    memset(&model, 0, sizeof(model));
    model.fd = -1;
    model.map = map;
    model.size = model_size;
    ds4_qwen_hc_weights weights = {
        .norm = &norm,
        .mix_down = &down,
        .mix_up = &up,
        .inject = &inject,
    };
    ds4_qwen_hc_weights dense_weights = {
        .norm = &norm,
        .mix_down = &down,
        .mix_up = &up_dense,
        .inject = &inject,
    };

    const uint64_t hc_values = (uint64_t)ROWS * WIDTH;
    const uint64_t block_values = (uint64_t)ROWS * HIDDEN;
    float *input = malloc(hc_values * sizeof(*input));
    float *block = malloc(block_values * sizeof(*block));
    float *got = malloc(hc_values * sizeof(*got));
    float *want = malloc(hc_values * sizeof(*want));
    float *mixed_got = malloc(block_values * sizeof(*mixed_got));
    float *mixed_want = malloc(block_values * sizeof(*mixed_want));
    float *scalar_got = malloc(hc_values * sizeof(*scalar_got));
    float *scalar_mixed = malloc(block_values * sizeof(*scalar_mixed));
    REQUIRE(input && block && got && want && mixed_got && mixed_want &&
            scalar_got && scalar_mixed,
            "host buffers");
    for (uint64_t i = 0; i < hc_values; i++)
        input[i] = 0.8f * sinf((float)(i + 1u) * 0.003f) + 0.2f;
    for (uint64_t i = 0; i < block_values; i++)
        block[i] = 0.4f * cosf((float)(i + 3u) * 0.007f);

    for (uint32_t row = 0; row < ROWS; row++) {
        float normed[WIDTH];
        for (uint32_t lane = 0; lane < HC; lane++) {
            const uint64_t base = (uint64_t)row * WIDTH + (uint64_t)lane * HIDDEN;
            double sum = 0.0;
            for (uint32_t d = 0; d < HIDDEN; d++) sum += (double)input[base + d] * input[base + d];
            const float scale = 1.0f / sqrtf((float)(sum / HIDDEN) + DS4_RMS_EPS);
            for (uint32_t d = 0; d < HIDDEN; d++) normed[(uint64_t)lane * HIDDEN + d] = input[base + d] * scale;
        }
        float low[LOW];
        for (uint32_t i = 0; i < LOW; i++) {
            const float projected = normed[i] / HC;
            low[i] = projected * sigmoidf_ref(projected);
        }
        float injection[HC];
        for (uint32_t lane = 0; lane < HC; lane++)
            injection[lane] = 2.0f * sigmoidf_ref(normed[(uint64_t)lane * HIDDEN] / HC);
        for (uint32_t lane = 0; lane < HC; lane++) {
            for (uint32_t d = 0; d < HIDDEN; d++) {
                float mixed = 0.0f;
                for (uint32_t source = 0; source < HC; source++) {
                    const uint32_t at = source * HIDDEN + d;
                    mixed += sigmoidf_ref(low[at % LOW]) * normed[at];
                }
                mixed /= HC;
                mixed_want[(uint64_t)row * HIDDEN + d] = mixed;
                const uint64_t at = (uint64_t)row * WIDTH + (uint64_t)lane * HIDDEN + d;
                want[at] = input[at] + block[(uint64_t)row * HIDDEN + d] * injection[lane];
            }
        }
    }

    REQUIRE(ds4_gpu_init(), "CUDA init");
    REQUIRE(ds4_gpu_set_model_map(map, model_size), "model map registration");
    ds4_gpu_tensor *dinput = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *dblock = ds4_gpu_tensor_alloc(block_values * sizeof(float));
    ds4_gpu_tensor *dout = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *dense_batch = ds4_gpu_tensor_alloc(
        (uint64_t)ROWS * LOW * sizeof(float));
    ds4_gpu_tensor *dense_scalar = ds4_gpu_tensor_alloc(
        (uint64_t)ROWS * LOW * sizeof(float));
    ds4_qwen_hc_ws ws;
    ds4_qwen_hc_ws scalar_ws;
    REQUIRE(dinput && dblock && dout && dense_batch && dense_scalar &&
            qwen4exp_hc_ws_alloc(&ws, ROWS, HIDDEN, HC, LOW) &&
            qwen4exp_hc_ws_alloc(&scalar_ws, 1u, HIDDEN, HC, LOW),
            "HC workspace");
    REQUIRE(ds4_gpu_tensor_write(dinput, 0, input, hc_values * sizeof(float)) &&
            ds4_gpu_tensor_write(dblock, 0, block, block_values * sizeof(float)),
            "input upload");
    REQUIRE(qwen4exp_hc_begin(
                &ws, &model, &weights, dinput, ROWS, true), "HC begin");
    REQUIRE(ds4_gpu_tensor_read(ws.mixed, 0, mixed_got,
                                block_values * sizeof(float)),
            "mixed-input readback");
    REQUIRE(qwen4exp_hc_finish(&ws, dinput, dblock, dout, ROWS), "HC finish");
    REQUIRE(ds4_gpu_tensor_read(dout, 0, got, hc_values * sizeof(float)), "output readback");
    for (uint32_t row = 0u; row < ROWS; row++) {
        ds4_gpu_tensor *input_row = ds4_gpu_tensor_view(
            dinput, (uint64_t)row * WIDTH * sizeof(float),
            (uint64_t)WIDTH * sizeof(float));
        ds4_gpu_tensor *block_row = ds4_gpu_tensor_view(
            dblock, (uint64_t)row * HIDDEN * sizeof(float),
            (uint64_t)HIDDEN * sizeof(float));
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
            dout, (uint64_t)row * WIDTH * sizeof(float),
            (uint64_t)WIDTH * sizeof(float));
        REQUIRE(input_row && block_row && out_row &&
                qwen4exp_hc_begin(
                    &scalar_ws, &model, &weights, input_row, 1u, true) &&
                ds4_gpu_tensor_read(
                    scalar_ws.mixed, 0,
                    scalar_mixed + (uint64_t)row * HIDDEN,
                    (uint64_t)HIDDEN * sizeof(float)) &&
                qwen4exp_hc_finish(
                    &scalar_ws, input_row, block_row, out_row, 1u),
                "scalar-row HC");
        ds4_gpu_tensor_free(out_row);
        ds4_gpu_tensor_free(block_row);
        ds4_gpu_tensor_free(input_row);
    }
    REQUIRE(ds4_gpu_tensor_read(
                dout, 0, scalar_got, hc_values * sizeof(float)),
            "scalar-row output readback");
    float mixed_worst = 0.0f, worst = 0.0f;
    for (uint64_t i = 0; i < block_values; i++) {
        const float error = fabsf(mixed_got[i] - mixed_want[i]);
        if (error > mixed_worst) mixed_worst = error;
    }
    for (uint64_t i = 0; i < hc_values; i++) {
        const float error = fabsf(got[i] - want[i]);
        if (error > worst) worst = error;
    }
    printf("Qwen HC projection/residual integration passed (max %.3g / %.3g)\n",
           mixed_worst, worst);
    REQUIRE(mixed_worst < 2.0e-4f, "integrated HC projection parity");
    REQUIRE(worst < 2.0e-4f, "integrated HC residual parity");
    REQUIRE(memcmp(mixed_got, scalar_mixed,
                   block_values * sizeof(float)) == 0,
            "two-row HC mixed output differs from scalar rows");
    REQUIRE(memcmp(got, scalar_got, hc_values * sizeof(float)) == 0,
            "two-row HC residual differs from scalar rows");

    REQUIRE(ds4_gpu_matmul_bf16_stable_rows_tensor(
                dense_batch, map, model_size, dense.abs_offset,
                WIDTH, LOW, dinput, ROWS),
            "dense two-row BF16 projection");
    for (uint32_t row = 0u; row < ROWS; row++) {
        ds4_gpu_tensor *input_row = ds4_gpu_tensor_view(
            dinput, (uint64_t)row * WIDTH * sizeof(float),
            (uint64_t)WIDTH * sizeof(float));
        ds4_gpu_tensor *output_row = ds4_gpu_tensor_view(
            dense_scalar, (uint64_t)row * LOW * sizeof(float),
            (uint64_t)LOW * sizeof(float));
        REQUIRE(input_row && output_row &&
                ds4_gpu_matmul_bf16_stable_rows_tensor(
                    output_row, map, model_size, dense.abs_offset,
                    WIDTH, LOW, input_row, 1u),
                "dense scalar-row BF16 projection");
        ds4_gpu_tensor_free(output_row);
        ds4_gpu_tensor_free(input_row);
    }
    float dense_batch_host[ROWS * LOW];
    float dense_scalar_host[ROWS * LOW];
    REQUIRE(ds4_gpu_tensor_read(
                dense_batch, 0u, dense_batch_host,
                sizeof(dense_batch_host)) &&
            ds4_gpu_tensor_read(
                dense_scalar, 0u, dense_scalar_host,
                sizeof(dense_scalar_host)),
            "dense BF16 projection readback");
    REQUIRE(memcmp(dense_batch_host, dense_scalar_host,
                   sizeof(dense_batch_host)) == 0,
            "dense two-row BF16 projection differs from scalar rows");

    /* Batched (BF16-rows) path: the fused mix against the cuBLAS + mix
     * pair.  On the identity-like fixture every logit is a single product,
     * so the two paths must agree bit for bit; on the dense mix_up the
     * accumulation order differs by a few ulp and both are checked against
     * the double-precision oracle.  200 rows exercise the 64-row tile
     * guard. */
    enum { BROWS = 200 };
    const uint64_t b_hc_values = (uint64_t)BROWS * WIDTH;
    const uint64_t b_block_values = (uint64_t)BROWS * HIDDEN;
    const uint64_t b_inject_values = (uint64_t)BROWS * HC;
    float *binput = malloc(b_hc_values * sizeof(*binput));
    float *b_old = malloc(b_block_values * sizeof(*b_old));
    float *b_new = malloc(b_block_values * sizeof(*b_new));
    float *b_ref = malloc(b_block_values * sizeof(*b_ref));
    float *b_inj_old = malloc(b_inject_values * sizeof(*b_inj_old));
    float *b_inj_new = malloc(b_inject_values * sizeof(*b_inj_new));
    REQUIRE(binput && b_old && b_new && b_ref && b_inj_old && b_inj_new,
            "batched host buffers");
    for (uint64_t i = 0; i < b_hc_values; i++)
        binput[i] = 0.7f * sinf((float)(i % 7919u) * 0.011f +
                                (float)(i / WIDTH) * 0.37f) + 0.1f;
    ds4_gpu_tensor *dbinput = ds4_gpu_tensor_alloc(b_hc_values * sizeof(float));
    ds4_qwen_hc_ws bws;
    REQUIRE(dbinput && qwen4exp_hc_ws_alloc(&bws, BROWS, HIDDEN, HC, LOW),
            "batched HC workspace");
    REQUIRE(ds4_gpu_tensor_write(dbinput, 0, binput, b_hc_values * sizeof(float)),
            "batched input upload");
    REQUIRE(ds4_gpu_qwen4exp_hc_mix_fused_applies(HIDDEN, HC, LOW),
            "fused mix applies to the Qwen shape");
    for (int dense_up = 0; dense_up < 2; dense_up++) {
        const ds4_qwen_hc_weights *w = dense_up ? &dense_weights : &weights;
        ds4_gpu_qwen4exp_hc_mix_fused_override(0);
        REQUIRE(qwen4exp_hc_begin(&bws, &model, w, dbinput, BROWS, false),
                "batched HC begin (cuBLAS mix)");
        REQUIRE(ds4_gpu_tensor_read(bws.mixed, 0, b_old,
                                    b_block_values * sizeof(float)) &&
                ds4_gpu_tensor_read(bws.injection, 0, b_inj_old,
                                    b_inject_values * sizeof(float)),
                "batched readback (cuBLAS mix)");
        ds4_gpu_qwen4exp_hc_mix_fused_override(1);
        REQUIRE(qwen4exp_hc_begin(&bws, &model, w, dbinput, BROWS, false),
                "batched HC begin (fused mix)");
        REQUIRE(ds4_gpu_tensor_read(bws.mixed, 0, b_new,
                                    b_block_values * sizeof(float)) &&
                ds4_gpu_tensor_read(bws.injection, 0, b_inj_new,
                                    b_inject_values * sizeof(float)),
                "batched readback (fused mix)");
        REQUIRE(memcmp(b_inj_old, b_inj_new,
                       b_inject_values * sizeof(float)) == 0,
                "fused path changed the injection gate");
        float pair_worst = 0.0f, old_ref_worst = 0.0f, new_ref_worst = 0.0f;
        for (uint32_t row = 0; row < BROWS; row++) {
            batched_reference_row(
                binput + (uint64_t)row * WIDTH,
                (const uint16_t *)(map + w->mix_up->abs_offset),
                b_ref + (uint64_t)row * HIDDEN);
        }
        for (uint64_t i = 0; i < b_block_values; i++) {
            const float pair = fabsf(b_old[i] - b_new[i]);
            const float old_ref = fabsf(b_old[i] - b_ref[i]);
            const float new_ref = fabsf(b_new[i] - b_ref[i]);
            if (pair > pair_worst) pair_worst = pair;
            if (old_ref > old_ref_worst) old_ref_worst = old_ref;
            if (new_ref > new_ref_worst) new_ref_worst = new_ref;
        }
        printf("Qwen HC batched mix (%s mix_up): fused vs cuBLAS %.3g, "
               "cuBLAS vs oracle %.3g, fused vs oracle %.3g\n",
               dense_up ? "dense" : "identity", pair_worst,
               old_ref_worst, new_ref_worst);
        if (!dense_up) {
            REQUIRE(pair_worst == 0.0f,
                    "fused mix differs from the cuBLAS + mix pair on "
                    "single-term logits");
        } else {
            REQUIRE(pair_worst < 5.0e-5f,
                    "fused mix differs from the cuBLAS + mix pair beyond "
                    "accumulation-order noise");
        }
        REQUIRE(old_ref_worst < 2.0e-4f, "cuBLAS mix vs batched oracle");
        REQUIRE(new_ref_worst < 2.0e-4f, "fused mix vs batched oracle");
    }
    /* Residual fused with the next norm (finish_begin) against finish then
     * begin, both on the fused chain and against the separate kernels:
     * the new state must match bit for bit and so must the mixed rows. */
    {
        ds4_gpu_tensor *dbblock = ds4_gpu_tensor_alloc(b_block_values * sizeof(float));
        ds4_gpu_tensor *dbout_a = ds4_gpu_tensor_alloc(b_hc_values * sizeof(float));
        ds4_gpu_tensor *dbout_b = ds4_gpu_tensor_alloc(b_hc_values * sizeof(float));
        float *bblock = malloc(b_block_values * sizeof(*bblock));
        float *out_a = malloc(b_hc_values * sizeof(*out_a));
        float *out_b = malloc(b_hc_values * sizeof(*out_b));
        REQUIRE(dbblock && dbout_a && dbout_b && bblock && out_a && out_b,
                "finish_begin buffers");
        for (uint64_t i = 0; i < b_block_values; i++)
            bblock[i] = 0.3f * cosf((float)(i % 6007u) * 0.019f);
        REQUIRE(ds4_gpu_tensor_write(dbblock, 0, bblock,
                                     b_block_values * sizeof(float)),
                "finish_begin block upload");
        for (int mode = 0; mode < 2; mode++) {
            /* Reference: separate finish + begin under this mode. */
            ds4_gpu_qwen4exp_hc_mix_fused_override(mode);
            REQUIRE(qwen4exp_hc_begin(&bws, &model, &weights, dbinput, BROWS, false) &&
                    qwen4exp_hc_finish(&bws, dbinput, dbblock, dbout_a, BROWS) &&
                    qwen4exp_hc_begin(&bws, &model, &dense_weights, dbout_a, BROWS, false),
                    "finish + begin");
            REQUIRE(ds4_gpu_tensor_read(dbout_a, 0, out_a, b_hc_values * sizeof(float)) &&
                    ds4_gpu_tensor_read(bws.mixed, 0, b_old, b_block_values * sizeof(float)),
                    "finish + begin readback");
            /* Candidate: the fused finish_begin (mode 1) or its fallback (mode 0). */
            REQUIRE(qwen4exp_hc_begin(&bws, &model, &weights, dbinput, BROWS, false) &&
                    qwen4exp_hc_finish_begin(&bws, &model, &dense_weights, dbinput,
                                             dbblock, dbout_b, BROWS, false),
                    "finish_begin");
            REQUIRE(ds4_gpu_tensor_read(dbout_b, 0, out_b, b_hc_values * sizeof(float)) &&
                    ds4_gpu_tensor_read(bws.mixed, 0, b_new, b_block_values * sizeof(float)),
                    "finish_begin readback");
            REQUIRE(memcmp(out_a, out_b, b_hc_values * sizeof(float)) == 0,
                    "finish_begin residual differs from finish");
            REQUIRE(memcmp(b_old, b_new, b_block_values * sizeof(float)) == 0,
                    "finish_begin mixed rows differ from finish + begin");
            printf("Qwen HC finish_begin (%s): residual and mixed rows bit-identical\n",
                   mode ? "fused" : "separate kernels");
        }
        free(out_b); free(out_a); free(bblock);
        ds4_gpu_tensor_free(dbout_b); ds4_gpu_tensor_free(dbout_a);
        ds4_gpu_tensor_free(dbblock);
    }
    ds4_gpu_qwen4exp_hc_mix_fused_override(-1);

    if (getenv("DS4_QWEN_PROFILE_HC")) {
        /* Production-shape probe: one 8,192-row hc_begin per path. */
        enum { PROWS = 8192 };
        const uint64_t p_hc_values = (uint64_t)PROWS * WIDTH;
        float *pinput = malloc(p_hc_values * sizeof(*pinput));
        REQUIRE(pinput, "probe host buffer");
        for (uint64_t i = 0; i < p_hc_values; i++)
            pinput[i] = 0.7f * sinf((float)(i % 7919u) * 0.011f) + 0.1f;
        ds4_gpu_tensor *dpinput = ds4_gpu_tensor_alloc(p_hc_values * sizeof(float));
        ds4_qwen_hc_ws pws;
        REQUIRE(dpinput && qwen4exp_hc_ws_alloc(&pws, PROWS, HIDDEN, HC, LOW),
                "probe workspace");
        REQUIRE(ds4_gpu_tensor_write(dpinput, 0, pinput, p_hc_values * sizeof(float)),
                "probe upload");
        /* The fixture map is host-registered; the served model's weights
         * are device-resident and L2-cached.  Cache the four HC weights so
         * the probe sees production-like weight reads (a host-mapped mix_up
         * costs the fused kernel 2.5x). */
        REQUIRE(ds4_gpu_cache_model_range(map, model_size, norm.abs_offset, norm.bytes, "hc norm") &&
                ds4_gpu_cache_model_range(map, model_size, down.abs_offset, down.bytes, "hc down") &&
                ds4_gpu_cache_model_range(map, model_size, up_dense.abs_offset, up_dense.bytes, "hc up") &&
                ds4_gpu_cache_model_range(map, model_size, inject.abs_offset, inject.bytes, "hc inject"),
                "probe weight caching");
        ds4_gpu_tensor *dpblock = ds4_gpu_tensor_alloc((uint64_t)PROWS * HIDDEN * sizeof(float));
        ds4_gpu_tensor *dpout = ds4_gpu_tensor_alloc(p_hc_values * sizeof(float));
        REQUIRE(dpblock && dpout && ds4_gpu_tensor_fill_f32(dpblock, 0.25f, (uint64_t)PROWS * HIDDEN),
                "probe block buffer");
        for (int fused = 0; fused < 2; fused++) {
            ds4_gpu_qwen4exp_hc_mix_fused_override(fused);
            const int iters = 10;
            for (int warm = 0; warm < 2; warm++) {
                REQUIRE(qwen4exp_hc_begin(&pws, &model, &dense_weights,
                                          dpinput, PROWS, false) &&
                        qwen4exp_hc_finish_begin(
                            &pws, &model, &dense_weights, dpinput,
                            dpblock, dpout, PROWS, false),
                        "probe warm-up");
            }
            /* Timed passes. */
            double ms_begin = 0.0, ms_finish_begin = 0.0;
            {
                REQUIRE(ds4_gpu_synchronize(), "probe sync");
                const double t0 = now_seconds();
                for (int it = 0; it < iters; it++)
                    REQUIRE(qwen4exp_hc_begin(&pws, &model, &dense_weights,
                                              dpinput, PROWS, false),
                            "probe HC begin");
                REQUIRE(ds4_gpu_synchronize(), "probe sync");
                ms_begin = (now_seconds() - t0) * 1e3 / iters;
                const double t1 = now_seconds();
                for (int it = 0; it < iters; it++)
                    REQUIRE(qwen4exp_hc_finish_begin(
                                &pws, &model, &dense_weights, dpinput,
                                dpblock, dpout, PROWS, false),
                            "probe HC finish_begin");
                REQUIRE(ds4_gpu_synchronize(), "probe sync");
                ms_finish_begin = (now_seconds() - t1) * 1e3 / iters;
            }
            printf("Qwen HC probe (%d rows, %s): begin %.3f ms, "
                   "finish+begin %.3f ms per call\n",
                   PROWS, fused ? "fused" : "separate kernels",
                   ms_begin, ms_finish_begin);
        }
        ds4_gpu_tensor_free(dpout);
        ds4_gpu_tensor_free(dpblock);
        ds4_gpu_qwen4exp_hc_mix_fused_override(-1);
        qwen4exp_hc_ws_free(&pws);
        ds4_gpu_tensor_free(dpinput);
        free(pinput);
    }

    qwen4exp_hc_ws_free(&bws);
    ds4_gpu_tensor_free(dbinput);
    free(b_inj_new); free(b_inj_old); free(b_ref); free(b_new); free(b_old);
    free(binput);

    qwen4exp_hc_ws_free(&scalar_ws);
    qwen4exp_hc_ws_free(&ws);
    ds4_gpu_tensor_free(dense_scalar);
    ds4_gpu_tensor_free(dense_batch);
    ds4_gpu_tensor_free(dout);
    ds4_gpu_tensor_free(dblock);
    ds4_gpu_tensor_free(dinput);
    ds4_gpu_unregister_model_map(map);
    ds4_gpu_cleanup();
    free(scalar_mixed); free(scalar_got);
    free(mixed_want); free(mixed_got); free(want); free(got);
    free(block); free(input); free(map);
    return 0;
}
