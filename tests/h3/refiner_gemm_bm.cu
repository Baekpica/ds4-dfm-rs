/*
 * What the the refiner refiner block's matmuls cost on their own, at the shape stage 2
 * runs them, so a per-block profile can be split into the part cuBLASLt owns
 * and the part the port's own kernels own.
 *
 * One call here is one projection of one block
 * (`crates/ds4-core/src/minimax_h3/the refiner module.rs`): thirty-four of them across
 * the six attentions and two feed-forwards, at 16,128 video rows, 126 audio
 * rows and 1,024 context rows. The list is the block's forward order, and every
 * row count is the one the block passes - `Attention::run` takes the query and
 * key rows separately, so the text cross-attention's K/V are 1,024 rows while
 * its Q is 16,128, and the A2V/V2A pair is 16,128 by 126 each way.
 *
 *   tests/h3/refiner_gemm_bm             # 16,128 video rows, the release tile
 *   tests/h3/refiner_gemm_bm 16128 5     # five passes for the mean
 *
 * It reports each projection's milliseconds and TFLOPS, then the block's GEMM
 * total, which is the ceiling the block could reach with zero elementwise work.
 * Compare it with `H3_BLOCK_PROFILE=1`'s per-layer milliseconds: whatever the
 * block spends beyond this figure is the port's kernels, not the GEMM.
 *
 * Both compute classes are priced on the same operands. `h3d_linear` (the
 * refiner's own path) asks for `CUBLAS_COMPUTE_32F`; the second arm asks for
 * `CUBLAS_COMPUTE_32F_FAST_16BF`, which is the class the video VAE decoder's
 * attribution found cuBLASLt selects for BF16 operands. If the two agree the
 * compute class is not a lever; if the FAST class is faster, it is.
 *
 * Build: `make tests/h3/refiner_gemm_bm`.
 */
#include "../../native/h3/h3_dit_gpu.cu"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* The release's geometry (`the refiner module.rs`: `VIDEO_TOKENS`, `AUDIO_TOKENS`). */
enum {
    VIDEO_ROWS = 16 * 24 * 42,
    AUDIO_ROWS = 126,
    CONTEXT_ROWS = 1024,
    VIDEO_DIM = 4096,
    AUDIO_DIM = 2048,
    MULTIPLIER = 4,
    HEADS = 32,
};

static void fail(const char *what, const char *detail) {
    std::fprintf(stderr, "%s: %s\n", what, detail);
    std::exit(2);
}

struct Call {
    const char *name;
    uint32_t rows;
    uint32_t input_dim;
    uint32_t output_dim;
};

/* The block's projections in `BasicAVTransformerBlock.forward` order
 * (`transformer.py:253-417`). Row `0` means "the video rows", the argument
 * this benchmark varies; the audio and prompt row counts are fixed. */
static const Call CALLS[] = {
    /* 1. Video self-attention (`attn1`). */
    {"v-self-q", 0, VIDEO_DIM, VIDEO_DIM},
    {"v-self-k", 0, VIDEO_DIM, VIDEO_DIM},
    {"v-self-v", 0, VIDEO_DIM, VIDEO_DIM},
    {"v-self-out", 0, VIDEO_DIM, VIDEO_DIM},
    {"v-self-gate", 0, VIDEO_DIM, HEADS},
    /* 2. Video text cross-attention (`attn2`): Q on the video rows, K/V on the
     *    prompt's. */
    {"v-text-q", 0, VIDEO_DIM, VIDEO_DIM},
    {"v-text-k", CONTEXT_ROWS, VIDEO_DIM, VIDEO_DIM},
    {"v-text-v", CONTEXT_ROWS, VIDEO_DIM, VIDEO_DIM},
    {"v-text-out", 0, VIDEO_DIM, VIDEO_DIM},
    {"v-text-gate", 0, VIDEO_DIM, HEADS},
    /* 3. Audio self- and text cross-attention. */
    {"a-self-q", AUDIO_ROWS, AUDIO_DIM, AUDIO_DIM},
    {"a-self-k", AUDIO_ROWS, AUDIO_DIM, AUDIO_DIM},
    {"a-self-v", AUDIO_ROWS, AUDIO_DIM, AUDIO_DIM},
    {"a-self-out", AUDIO_ROWS, AUDIO_DIM, AUDIO_DIM},
    {"a-self-gate", AUDIO_ROWS, AUDIO_DIM, HEADS},
    {"a-text-q", AUDIO_ROWS, AUDIO_DIM, AUDIO_DIM},
    {"a-text-k", CONTEXT_ROWS, AUDIO_DIM, AUDIO_DIM},
    {"a-text-v", CONTEXT_ROWS, AUDIO_DIM, AUDIO_DIM},
    {"a-text-out", AUDIO_ROWS, AUDIO_DIM, AUDIO_DIM},
    {"a-text-gate", AUDIO_ROWS, AUDIO_DIM, HEADS},
    /* 4. A2V then V2A: the projection shapes are the audio config's
     *    (`transformer.py:151-177`), so the inner width is `HEADS * 64`. */
    {"a2v-q", 0, VIDEO_DIM, AUDIO_DIM},
    {"a2v-k", AUDIO_ROWS, AUDIO_DIM, AUDIO_DIM},
    {"a2v-v", AUDIO_ROWS, AUDIO_DIM, AUDIO_DIM},
    {"a2v-out", 0, AUDIO_DIM, VIDEO_DIM},
    {"a2v-gate", 0, VIDEO_DIM, HEADS},
    {"v2a-q", AUDIO_ROWS, AUDIO_DIM, AUDIO_DIM},
    {"v2a-k", 0, VIDEO_DIM, AUDIO_DIM},
    {"v2a-v", 0, VIDEO_DIM, AUDIO_DIM},
    {"v2a-out", AUDIO_ROWS, AUDIO_DIM, AUDIO_DIM},
    {"v2a-gate", AUDIO_ROWS, AUDIO_DIM, HEADS},
    /* 5. The two feed-forwards. */
    {"v-ff-up", 0, VIDEO_DIM, VIDEO_DIM * MULTIPLIER},
    {"v-ff-down", 0, VIDEO_DIM * MULTIPLIER, VIDEO_DIM},
    {"a-ff-up", AUDIO_ROWS, AUDIO_DIM, AUDIO_DIM * MULTIPLIER},
    {"a-ff-down", AUDIO_ROWS, AUDIO_DIM * MULTIPLIER, AUDIO_DIM},
};

static const int CALL_COUNT = (int)(sizeof(CALLS) / sizeof(CALLS[0]));

/* Deterministic values in [-1, 1): every bit of the BF16 mantissa is used, so
 * the arithmetic is the same class as a real activation's. */
static void fill(std::vector<__nv_bfloat16> &host, uint32_t seed) {
    uint32_t state = seed * 2654435761u + 1u;
    for (auto &item : host) {
        state = state * 1664525u + 1013904223u;
        item = __float2bfloat16((float)(state >> 8) / 8388608.0f - 1.0f);
    }
}

/* One operand pool per distinct shape, so thirty-four projections need a
 * handful of buffers rather than thirty-four of each. */
struct Pool {
    std::vector<h3d_tensor *> tensors;
    std::vector<std::vector<uint32_t>> keys;
    h3d_context *gpu;

    h3d_tensor *get(uint32_t a, uint32_t b, char *error, size_t size) {
        for (size_t index = 0; index < tensors.size(); index++) {
            if (keys[index][0] == a && keys[index][1] == b) return tensors[index];
        }
        const size_t elements = (size_t)a * b;
        h3d_tensor *tensor = h3d_alloc(gpu, H3D_BF16, elements, error, size);
        if (!tensor) fail("alloc", error);
        std::vector<__nv_bfloat16> host;
        host.resize(elements);
        fill(host, (uint32_t)(tensors.size() * 7 + 3));
        if (!h3d_upload(gpu, tensor, host.data(), error, size)) fail("upload", error);
        tensors.push_back(tensor);
        keys.push_back({a, b});
        return tensor;
    }
};

static double tflops(uint32_t rows, uint32_t input_dim, uint32_t output_dim,
                     double milliseconds) {
    const double flops =
            2.0 * (double)rows * (double)input_dim * (double)output_dim;
    return milliseconds > 0.0 ? flops / (milliseconds * 1e9) : 0.0;
}

int main(int argc, char **argv) {
    const int calls = argc > 2 ? std::atoi(argv[2]) : 3;
    char error[512] = {};
    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) fail("create", error);

    Pool inputs, weights, outputs;
    inputs.gpu = gpu;
    weights.gpu = gpu;
    outputs.gpu = gpu;
    if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);

    const uint32_t video_rows =
            argc > 1 ? (uint32_t)std::atoi(argv[1]) : (uint32_t)VIDEO_ROWS;

    /* The same operand set under both compute classes, so a difference between
     * the arms is the class and nothing else. */
    std::vector<h3d_tensor *> input(CALL_COUNT), weight(CALL_COUNT),
            output(CALL_COUNT);
    for (int index = 0; index < CALL_COUNT; index++) {
        const Call &call = CALLS[index];
        const uint32_t rows = call.rows == 0 ? video_rows : call.rows;
        input[index] = inputs.get(rows, call.input_dim, error, sizeof(error));
        weight[index] = weights.get(call.output_dim, call.input_dim, error,
                                    sizeof(error));
        output[index] = outputs.get(rows, call.output_dim, error, sizeof(error));
    }
    if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);

    /* One call of one projection, through whichever class the arm asks for, so
     * the two arms share a code path and differ only in the compute type. */
    bool fast_class = false;
    auto project = [&](int index) -> bool {
        const Call &call = CALLS[index];
        const uint32_t rows = call.rows == 0 ? video_rows : call.rows;
        if (fast_class) {
            return h3d_linear_lt_compute(
                           gpu, output[index]->data, input[index]->data,
                           weight[index]->data, rows, call.input_dim,
                           call.output_dim, CUDA_R_16BF,
                           CUBLAS_COMPUTE_32F_FAST_16BF, error,
                           sizeof(error)) != 0;
        }
        return h3d_linear(gpu, output[index], input[index], weight[index], NULL,
                          rows, call.input_dim, call.output_dim, H3D_BF16, error,
                          sizeof(error)) != 0;
    };

    /* cuBLASLt pays a one-time cost on a shape's first call - algorithm
     * selection and, on some hosts, a JIT - so both classes get a warm-up pass
     * and the figure reported is a warmed call's. Each call is synced, so the
     * time is the call's own and not the queue's. */
    for (int arm = 0; arm < 2; arm++) {
        fast_class = arm == 1;
        for (int index = 0; index < CALL_COUNT; index++) {
            if (!project(index)) fail("warm-up", error);
        }
        if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);
    }

    const char *classes[2] = {"CUBLAS_COMPUTE_32F",
                              "CUBLAS_COMPUTE_32F_FAST_16BF"};
    double per_call[CALL_COUNT] = {};
    double arm_total[2] = {};
    double flops = 0.0;
    std::printf("arm=linear rows=%u projections=%d passes=%d\n", video_rows,
                CALL_COUNT, calls);
    for (int arm = 0; arm < 2; arm++) {
        fast_class = arm == 1;
        for (int index = 0; index < CALL_COUNT; index++) {
            const Call &call = CALLS[index];
            const uint32_t rows = call.rows == 0 ? video_rows : call.rows;
            double spent = 0.0;
            for (int pass = 0; pass < calls; pass++) {
                auto start = std::chrono::steady_clock::now();
                if (!project(index)) fail("linear", error);
                if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);
                auto finish = std::chrono::steady_clock::now();
                spent +=
                        std::chrono::duration<double, std::milli>(finish - start)
                                .count();
            }
            per_call[index] = spent / calls;
            if (arm == 0) {
                flops += 2.0 * (double)rows * (double)call.input_dim *
                         (double)call.output_dim;
                std::printf(
                        "GEMM %-12s rows=%5u in=%5u out=%5u ms=%9.3f tflops=%6.2f\n",
                        call.name, rows, call.input_dim, call.output_dim,
                        per_call[index],
                        tflops(rows, call.input_dim, call.output_dim,
                               per_call[index]));
            }
            arm_total[arm] += per_call[index];
        }
    }

    std::printf("BLOCK class=%s gemm_ms=%.3f tflops=%.2f\n", classes[0],
                arm_total[0], flops / (arm_total[0] * 1e9));
    std::printf("BLOCK class=%s gemm_ms=%.3f tflops=%.2f gain=%.3f\n", classes[1],
                arm_total[1], flops / (arm_total[1] * 1e9),
                arm_total[0] / arm_total[1]);
    return 0;
}
