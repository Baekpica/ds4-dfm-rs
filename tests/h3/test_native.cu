// Include the DiT implementation only to queue work on its private stream.
#include "../../native/h3/h3_dit_gpu.cu"
#include "../../native/h3/h3_text_gpu.h"
#include "../../native/h3/h3_vision_gpu.h"

#include <algorithm>
#include <chrono>
#include <vector>
#include <future>
#include <limits>

static int failures = 0;

static void check(bool passed, const char *name) {
    printf("%s %s\n", passed ? "PASS" : "FAIL", name);
    failures += !passed;
}

static void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(status));
        exit(2);
    }
}

struct StreamBlock {
    std::promise<void> entered;
    std::promise<void> release;
    std::future<void> released = release.get_future();
};

static void CUDART_CB hold_stream(void *data) {
    auto *block = static_cast<StreamBlock *>(data);
    block->entered.set_value();
    block->released.wait();
}

static void read_waits(h3d_context *gpu) {
    char error[512] = {};
    h3d_tensor *tensor = h3d_alloc(gpu, H3D_U32, 1, error, sizeof(error));
    if (!tensor) {
        fprintf(stderr, "%s\n", error);
        exit(2);
    }
    uint32_t source = 42;
    uint32_t *result = nullptr;
    cuda_check(cudaMallocHost(&result, sizeof(*result)));
    cuda_check(cudaMemcpy(tensor->data, &source, sizeof(source), cudaMemcpyHostToDevice));
    *result = 0;

    StreamBlock block;
    cuda_check(cudaLaunchHostFunc(gpu->stream, hold_stream, &block));
    block.entered.get_future().wait();
    std::promise<void> started;
    auto reader = std::async(std::launch::async, [&] {
        cuda_check(cudaSetDevice(0));
        started.set_value();
        return h3d_read(gpu, result, tensor, error, sizeof(error));
    });
    started.get_future().wait();
    bool returned_early = reader.wait_for(std::chrono::milliseconds(500)) ==
                          std::future_status::ready;
    // Always release and join before inspecting or freeing the host buffer.
    block.release.set_value();
    int ok = reader.get();
    cuda_check(cudaStreamSynchronize(gpu->stream));
    check(ok && !returned_early && *result == source, "read waits for host bytes");
    cuda_check(cudaFreeHost(result));
    h3d_free(tensor);
}

static void allocation_sizes(h3d_context *dit, h3t_context *text, h3v_context *vision) {
    char error[512] = {};
    constexpr size_t maximum = std::numeric_limits<size_t>::max();
    for (int dtype : {H3D_BF16, H3D_F32, H3D_U32}) {
        size_t item = dtype == H3D_BF16 ? 2 : 4;
        auto *tensor = h3d_alloc(dit, dtype, maximum / item + 2, error, sizeof(error));
        check(!tensor, "DiT allocation rejects byte overflow");
        h3d_free(tensor);
    }
    for (int dtype : {H3T_BF16, H3T_F32}) {
        size_t item = dtype == H3T_BF16 ? 2 : 4;
        auto *tensor = h3t_alloc(text, dtype, maximum / item + 2, error, sizeof(error));
        check(!tensor, "text allocation rejects byte overflow");
        h3t_free(tensor);
    }
    auto *tensor = h3v_alloc(vision, maximum / 2 + 2, error, sizeof(error));
    check(!tensor, "vision allocation rejects byte overflow");
    h3v_free(tensor);
}

static void gate_dtype(h3d_context *gpu) {
    char error[512] = {};
    auto *values = h3d_alloc(gpu, H3D_BF16, 1, error, sizeof(error));
    if (!values) {
        exit(2);
    }
    cuda_check(cudaMemset(values->data, 0, 2));
    for (int dtype : {H3D_BF16, H3D_F32, H3D_U32}) {
        // Even invalid types have four initialized bytes and a valid row 0.
        auto *map = h3d_alloc(gpu, dtype, dtype == H3D_BF16 ? 2 : 1, error, sizeof(error));
        if (!map) {
            exit(2);
        }
        cuda_check(cudaMemset(map->data, 0, 4));
        int ok = h3d_gate(gpu, values, values, values, values, map,
                          1, 1, 1, 0, error, sizeof(error));
        check(bool(ok) == (dtype == H3D_U32), "gate requires U32 row indices");
        cuda_check(cudaStreamSynchronize(gpu->stream));
        h3d_free(map);
    }
    h3d_free(values);
}

// 65,536 rows exceed CUDA grid.y/z; the same operations also cover multiple
// column/head tiles at a short row count without loading a checkpoint.
static void row_launches(h3d_context *gpu, uint32_t rows, uint32_t width) {
    char error[512] = {};
    const size_t count = (size_t)rows * width;
    std::vector<h3d_tensor *> tensors;
    auto alloc = [&](int dtype, size_t elements) {
        auto *tensor = h3d_alloc(gpu, dtype, elements, error, sizeof(error));
        if (!tensor) { fprintf(stderr, "%s\n", error); exit(2); }
        cuda_check(cudaMemsetAsync(tensor->data, 0xff, elements * h3d_dtype_size(dtype), gpu->stream));
        tensors.push_back(tensor);
        return tensor;
    };
    auto upload = [&](h3d_tensor *tensor, const void *data) {
        cuda_check(cudaMemcpyAsync(tensor->data, data,
            tensor->elements * h3d_dtype_size(tensor->dtype), cudaMemcpyHostToDevice, gpu->stream));
    };
    auto verify = [&](int ok, h3d_tensor *output, const std::vector<uint16_t> &expected,
                      const char *name) {
        int synced = h3d_sync(gpu, error, sizeof(error));
        std::vector<uint16_t> actual(expected.size());
        bool read = ok && synced && h3d_read(gpu, actual.data(), output, error, sizeof(error));
        if (!ok || !synced || !read) { fprintf(stderr, "%s: %s\n", name, error); }
        check(read && actual == expected, name);
    };
    auto *input = alloc(H3D_BF16, count);
    auto *output = alloc(H3D_BF16, count);
    auto *branch = alloc(H3D_BF16, count);
    auto *modulation = alloc(H3D_BF16, width);
    auto *map = alloc(H3D_U32, rows);
    auto *fused = alloc(H3D_BF16, count * 2);
    auto *original = alloc(H3D_BF16, count);
    auto *baseline = alloc(H3D_BF16, count);
    auto *indices = alloc(H3D_U32, rows);
    auto *pairs = alloc(H3D_U32, (size_t)rows * 2);

    std::vector<uint16_t> ones(count, 0x3f80), twos(count, 0x4000);
    std::vector<uint16_t> threes(width, 0x4040), sevens(count, 0x40e0);
    std::vector<uint16_t> zeros(count * 2, 0);
    std::vector<uint32_t> map_values(rows, 0), identity(rows), pair_values((size_t)rows * 2);
    for (uint32_t row = 0; row < rows; ++row) {
        identity[row] = row;
        pair_values[(size_t)row * 2] = pair_values[(size_t)row * 2 + 1] = row;
    }
    upload(input, ones.data()); upload(branch, twos.data());
    upload(modulation, threes.data()); upload(map, map_values.data());
    upload(fused, zeros.data()); upload(indices, identity.data()); upload(pairs, pair_values.data());

    int ok = h3d_gate(gpu, output, input, branch, modulation, map,
                      rows, width, 1, 0, error, sizeof(error));
    verify(ok, output, sevens, "gate covers every row and column");
    cuda_check(cudaMemsetAsync(output->data, 0xff, count * 2, gpu->stream));
    ok = h3d_swiglu(gpu, output, fused, rows, width, error, sizeof(error));
    verify(ok, output, std::vector<uint16_t>(count, 0), "SwiGLU covers every row and column");
    cuda_check(cudaMemsetAsync(output->data, 0xff, count * 2, gpu->stream));
    ok = h3d_token_pool(gpu, output, input, 0, original, 0, baseline, 0,
                        indices, pairs, rows, rows, rows, width, error, sizeof(error));
    verify(ok, output, ones, "token pool covers every row and column");
    cuda_check(cudaMemsetAsync(output->data, 0xff, count * 2, gpu->stream));
    ok = h3d_token_expand(gpu, output, original, 0, input, baseline, 0,
                          indices, indices, rows, rows, rows, width, 0, 1.0f,
                          error, sizeof(error));
    verify(ok, output, ones, "token expand covers every row and column");
    for (auto *tensor : tensors) { h3d_free(tensor); }
}

static void qkv_row_launch(h3d_context *gpu, uint32_t rows, uint32_t head_dim) {
    char error[512] = {};
    constexpr uint32_t heads = 2;
    const size_t count = (size_t)rows * heads * head_dim;
    std::vector<h3d_tensor *> tensors;
    auto alloc = [&](size_t elements) {
        auto *tensor = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
        if (!tensor) { fprintf(stderr, "%s\n", error); exit(2); }
        if (elements) { cuda_check(cudaMemsetAsync(tensor->data, 0xff, elements * 2, gpu->stream)); }
        tensors.push_back(tensor);
        return tensor;
    };
    auto *qkv = alloc(count * 3), *query = alloc(count), *key = alloc(count), *value = alloc(count);
    auto *norm = alloc(head_dim), *rope = alloc(0);
    std::vector<uint16_t> source(count * 3, 0), expected(count), actual(count);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t head = 0; head < heads; ++head) {
            for (uint32_t dim = 0; dim < head_dim; ++dim) {
                // Exact finite BF16 values distinguish every axis, including the final row.
                uint16_t bits = 0x3f80 + ((row + head * 17 + dim) % 128);
                source[((size_t)row * heads + head) * head_dim * 3 + 2 * head_dim + dim] = bits;
                expected[((size_t)head * rows + row) * head_dim + dim] = bits;
            }
        }
    }
    cuda_check(cudaMemcpyAsync(qkv->data, source.data(), source.size() * 2, cudaMemcpyHostToDevice, gpu->stream));
    cuda_check(cudaMemsetAsync(norm->data, 0, head_dim * 2, gpu->stream));
    int ok = h3d_qkv_rope(gpu, query, key, value, qkv, norm, norm, rope, rope,
                          rows, heads, head_dim, 0, 1e-5f, error, sizeof(error));
    int synced = h3d_sync(gpu, error, sizeof(error));
    bool values_match = ok && synced && h3d_read(gpu, actual.data(), value, error, sizeof(error)) &&
                        actual == expected;
    bool normalized = ok && synced;
    for (auto *tensor : {query, key}) {
        normalized = normalized && h3d_read(gpu, actual.data(), tensor, error, sizeof(error)) &&
                     std::all_of(actual.begin(), actual.end(), [](uint16_t x) { return x == 0; });
    }
    if (!ok || !synced) { fprintf(stderr, "QKV: %s\n", error); }
    check(values_match && normalized, "QKV covers every row, head and dimension");
    for (auto *tensor : tensors) { h3d_free(tensor); }
}

static void pad_plane_launch(h3d_context *gpu, uint32_t depth, uint32_t height,
                              uint32_t channels) {
    char error[512] = {};
    constexpr uint32_t width = 2, front = 2;
    const uint32_t od = depth + front, oh = height + 2, ow = width + 2;
    std::vector<float> input((size_t)depth * height * width * channels);
    std::vector<float> expected((size_t)od * oh * ow * channels, 0), actual(expected.size());
    for (size_t i = 0; i < input.size(); ++i) { input[i] = (float)(i % 31); }
    for (uint32_t t = front; t < od; ++t) {
        for (uint32_t y = 0; y < oh; ++y) {
            uint32_t sy = y == 0 ? 1 : y == oh - 1 ? height - 2 : y - 1;
            for (uint32_t x = 0; x < ow; ++x) {
                uint32_t sx = x == 0 ? 1 : x == ow - 1 ? width - 2 : x - 1;
                for (uint32_t c = 0; c < channels; ++c) {
                    expected[(((size_t)t * oh + y) * ow + x) * channels + c] =
                        input[(((size_t)(t - front) * height + sy) * width + sx) * channels + c];
                }
            }
        }
    }
    auto *source = h3d_alloc(gpu, H3D_F32, input.size(), error, sizeof(error));
    auto *output = h3d_alloc(gpu, H3D_F32, expected.size(), error, sizeof(error));
    if (!source || !output) { fprintf(stderr, "%s\n", error); exit(2); }
    cuda_check(cudaMemcpyAsync(source->data, input.data(), input.size() * 4, cudaMemcpyHostToDevice, gpu->stream));
    cuda_check(cudaMemsetAsync(output->data, 0xff, expected.size() * 4, gpu->stream));
    int ok = h3d_vae_encoder_pad(gpu, output, source, 1, depth, height, width, channels,
                                 front, 1, 1, 1, 1, error, sizeof(error));
    int synced = h3d_sync(gpu, error, sizeof(error));
    bool read = ok && synced && h3d_read(gpu, actual.data(), output, error, sizeof(error));
    if (!ok || !synced || !read) { fprintf(stderr, "VAE pad: %s\n", error); }
    check(read && actual == expected, "VAE padding covers every plane and channel");
    h3d_free(source); h3d_free(output);
}

static void short_attention_axes(h3d_context *gpu) {
    char error[512] = {};
    constexpr uint32_t rows = 17, heads = 2, width = 128;
    constexpr size_t elements = (size_t)rows * heads * width;
    auto *zero = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *value = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *output = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    if (!zero || !value || !output) { fprintf(stderr, "%s\n", error); exit(2); }
    std::vector<uint16_t> source(elements), expected(elements), actual(elements);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t head = 0; head < heads; ++head) {
            for (uint32_t dim = 0; dim < width; ++dim) {
                uint16_t bits = head == 0 ? 0x3f80 : 0x4000;
                source[((size_t)head * rows + row) * width + dim] = bits;
                expected[((size_t)row * heads + head) * width + dim] = bits;
            }
        }
    }
    cuda_check(cudaMemsetAsync(zero->data, 0, elements * 2, gpu->stream));
    cuda_check(cudaMemcpyAsync(value->data, source.data(), elements * 2, cudaMemcpyHostToDevice, gpu->stream));
    cuda_check(cudaMemsetAsync(output->data, 0xff, elements * 2, gpu->stream));
    int ok = h3d_sdpa(gpu, output, zero, zero, value, rows, heads, width, 0.125f,
                      error, sizeof(error));
    int synced = h3d_sync(gpu, error, sizeof(error));
    bool read = ok && synced && h3d_read(gpu, actual.data(), output, error, sizeof(error));
    if (!ok || !synced || !read) { fprintf(stderr, "attention: %s\n", error); }
    check(read && actual == expected, "tiled attention preserves head and row axes");
    h3d_free(zero); h3d_free(value); h3d_free(output);
}

static void short_attention_f32_axes(h3d_context *gpu) {
    char error[512] = {};
    /* width 64 is the video VAE's head_dim, so this drives the tiled F32
     * kernel rather than the generic one, and 17 rows leaves a partial final
     * block for the rows-per-warp tail. */
    constexpr uint32_t rows = 17, heads = 2, width = 64;
    constexpr size_t elements = (size_t)rows * heads * width;
    auto *zero = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    auto *value = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    auto *output = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    if (!zero || !value || !output) { fprintf(stderr, "%s\n", error); exit(2); }
    std::vector<float> source(elements), expected(elements), actual(elements);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t head = 0; head < heads; ++head) {
            for (uint32_t dim = 0; dim < width; ++dim) {
                // Powers of two retain an exact uniform mean for every row.
                float item = head == 0 ? 1.0f : 2.0f;
                source[((size_t)head * rows + row) * width + dim] = item;
                expected[((size_t)row * heads + head) * width + dim] = item;
            }
        }
    }
    cuda_check(cudaMemsetAsync(zero->data, 0, elements * 4, gpu->stream));
    cuda_check(cudaMemcpyAsync(value->data, source.data(), elements * 4, cudaMemcpyHostToDevice, gpu->stream));
    cuda_check(cudaMemsetAsync(output->data, 0xff, elements * 4, gpu->stream));
    int ok = h3d_sdpa_f32(gpu, output, zero, zero, value, rows, heads, width, 0.125f,
                          error, sizeof(error));
    int synced = h3d_sync(gpu, error, sizeof(error));
    bool read = ok && synced && h3d_read(gpu, actual.data(), output, error, sizeof(error));
    if (!ok || !synced || !read) { fprintf(stderr, "F32 attention: %s\n", error); }
    check(read && actual == expected, "tiled F32 attention preserves head and row axes");
    h3d_free(zero); h3d_free(value); h3d_free(output);
}

/* Deterministic high-entropy F32 values in [-1, 1): every bit of the mantissa
 * is used, so a reduced-precision arm has something to round. */
static void seed_values(std::vector<float> &values, uint32_t seed) {
    uint32_t state = seed * 2654435761u + 1u;
    for (auto &item : values) {
        state = state * 1664525u + 1013904223u;
        item = (float)(state >> 8) / 8388608.0f - 1.0f;
    }
}

/* Every option back to the reference path, so one test cannot leak into the
 * next: the context is shared by everything below. */
static void reset_vae_options(h3d_context *gpu) {
    bool reset = h3d_set_vae_option(gpu, "attention", "scalar") &&
                 h3d_set_vae_option(gpu, "sdpa_order", "heads") &&
                 h3d_set_vae_option(gpu, "gemm", "fp32");
    check(reset, "video VAE options reset to the reference path");
}

/* The F32 attention's CTA order is a cache experiment, not an arithmetic one:
 * `grid(query blocks, heads)` must reproduce the reference order bit for bit.
 * 37 rows is a partial final query block, and 32 heads covers the order swap. */
static void video_vae_attention_order(h3d_context *gpu) {
    char error[512] = {};
    constexpr uint32_t rows = 37, heads = 32, width = 64;
    constexpr size_t elements = (size_t)rows * heads * width;
    auto *query = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    auto *key = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    auto *value = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    auto *output = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    if (!query || !key || !value || !output) { fprintf(stderr, "%s\n", error); exit(2); }
    std::vector<float> host(elements), reference(elements), actual(elements);
    uint32_t seed = 1;
    bool uploaded = true;
    for (auto *tensor : {query, key, value}) {
        seed_values(host, seed++);
        uploaded = uploaded && h3d_upload(gpu, tensor, host.data(), error, sizeof(error));
    }
    if (!uploaded) { fprintf(stderr, "upload: %s\n", error); exit(2); }

    const float scale = 1.0f / (float)std::sqrt((double)width);
    bool queued = h3d_set_vae_option(gpu, "sdpa_order", "heads") &&
                  h3d_sdpa_f32(gpu, output, query, key, value, rows, heads, width, scale,
                               error, sizeof(error)) &&
                  h3d_sync(gpu, error, sizeof(error)) &&
                  h3d_read(gpu, reference.data(), output, error, sizeof(error));
    queued = queued && h3d_set_vae_option(gpu, "sdpa_order", "rows") &&
             h3d_sdpa_f32(gpu, output, query, key, value, rows, heads, width, scale,
                          error, sizeof(error)) &&
             h3d_sync(gpu, error, sizeof(error)) &&
             h3d_read(gpu, actual.data(), output, error, sizeof(error));
    if (!queued) { fprintf(stderr, "order arms: %s\n", error); }
    check(queued && actual == reference,
          "F32 attention CTA order is bit-identical to the reference order");
    reset_vae_options(gpu);
    h3d_free(query); h3d_free(key); h3d_free(value); h3d_free(output);
}

/* The tensor-core F32 attention is a numerical arm: it must differ from the
 * scalar kernel (otherwise the switch did nothing), stay finite, and land
 * inside a structural bound. The bound is loose on purpose — it catches a
 * mis-staged tile or a swapped axis, which move the output by order 1, not the
 * BF16 rounding this arm actually carries, which is measured and reported by
 * `tests/h3/attn_f32_bm --compare`. */
static void video_vae_attention_mma(h3d_context *gpu) {
    char error[512] = {};
    constexpr uint32_t rows = 37, heads = 32, width = 64;
    constexpr size_t elements = (size_t)rows * heads * width;
    auto *query = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    auto *key = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    auto *value = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    auto *output = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    if (!query || !key || !value || !output) { fprintf(stderr, "%s\n", error); exit(2); }
    std::vector<float> host(elements), reference(elements), actual(elements);
    uint32_t seed = 1;
    bool uploaded = true;
    for (auto *tensor : {query, key, value}) {
        seed_values(host, seed++);
        uploaded = uploaded && h3d_upload(gpu, tensor, host.data(), error, sizeof(error));
    }
    if (!uploaded) { fprintf(stderr, "upload: %s\n", error); exit(2); }

    const float scale = 1.0f / (float)std::sqrt((double)width);
    bool queued = h3d_set_vae_option(gpu, "attention", "scalar") &&
                  h3d_sdpa_f32(gpu, output, query, key, value, rows, heads, width, scale,
                               error, sizeof(error)) &&
                  h3d_sync(gpu, error, sizeof(error)) &&
                  h3d_read(gpu, reference.data(), output, error, sizeof(error));
    queued = queued && h3d_set_vae_option(gpu, "attention", "tensor") &&
             h3d_sdpa_f32(gpu, output, query, key, value, rows, heads, width, scale,
                          error, sizeof(error)) &&
             h3d_sync(gpu, error, sizeof(error)) &&
             h3d_read(gpu, actual.data(), output, error, sizeof(error));
    if (!queued) { fprintf(stderr, "attention arms: %s\n", error); }

    float worst = 0.0f, largest = 0.0f;
    bool finite = true;
    bool differs = false;
    for (size_t index = 0; index < elements; index++) {
        finite = finite && std::isfinite(actual[index]) && std::isfinite(reference[index]);
        differs = differs || actual[index] != reference[index];
        worst = std::max(worst, std::fabs(actual[index] - reference[index]));
        largest = std::max(largest, std::fabs(reference[index]));
    }
    check(queued && finite, "tensor-core F32 attention output is finite");
    check(differs, "tensor-core F32 attention is a distinct arm, not the scalar one");
    check(worst <= 0.05f * largest,
          "tensor-core F32 attention stays inside the structural bound");
    reset_vae_options(gpu);
    h3d_free(query); h3d_free(key); h3d_free(value); h3d_free(output);
}

/* The decoder's linear compute modes. The bias still goes through the
 * reference's separate kernel in every mode, so the check is the matmul: the
 * default path must equal the plain `h3d_linear` bit for bit, which is what
 * makes the option free to leave unset, and the reduced modes must stay finite
 * and inside a structural bound until the reproducer measures their deviation. */
static void video_vae_linear_options(h3d_context *gpu) {
    char error[512] = {};
    constexpr uint32_t rows = 5, input_dim = 11, output_dim = 7;
    constexpr size_t inputs = (size_t)rows * input_dim;
    constexpr size_t weights = (size_t)output_dim * input_dim;
    constexpr size_t outputs = (size_t)rows * output_dim;
    auto *input = h3d_alloc(gpu, H3D_F32, inputs, error, sizeof(error));
    auto *weight = h3d_alloc(gpu, H3D_F32, weights, error, sizeof(error));
    auto *bias = h3d_alloc(gpu, H3D_F32, output_dim, error, sizeof(error));
    auto *plain = h3d_alloc(gpu, H3D_F32, outputs, error, sizeof(error));
    auto *option = h3d_alloc(gpu, H3D_F32, outputs, error, sizeof(error));
    if (!input || !weight || !bias || !plain || !option) {
        fprintf(stderr, "%s\n", error);
        exit(2);
    }
    std::vector<float> input_host(inputs), weight_host(weights), bias_host(output_dim);
    seed_values(input_host, 3);
    seed_values(weight_host, 4);
    seed_values(bias_host, 5);
    bool uploaded = h3d_upload(gpu, input, input_host.data(), error, sizeof(error)) &&
                    h3d_upload(gpu, weight, weight_host.data(), error, sizeof(error)) &&
                    h3d_upload(gpu, bias, bias_host.data(), error, sizeof(error));
    if (!uploaded) { fprintf(stderr, "upload: %s\n", error); exit(2); }

    std::vector<float> reference(outputs), actual(outputs);
    auto run = [&](h3d_tensor *into) {
        return h3d_vae_linear(gpu, into, input, weight, bias, rows, input_dim,
                              output_dim, error, sizeof(error)) &&
               h3d_sync(gpu, error, sizeof(error)) &&
               h3d_read(gpu, actual.data(), into, error, sizeof(error));
    };

    /* The default path against the reference's own entry point. */
    bool queued = h3d_linear(gpu, plain, input, weight, bias, rows, input_dim,
                             output_dim, H3D_F32, error, sizeof(error)) &&
                  h3d_sync(gpu, error, sizeof(error)) &&
                  h3d_read(gpu, reference.data(), plain, error, sizeof(error)) &&
                  run(option);
    if (!queued) { fprintf(stderr, "fp32 arms: %s\n", error); }
    check(queued && actual == reference,
          "video VAE linear default path is bit-identical to the reference path");

    /* The reduced-precision compute modes: finite, inside the structural bound,
     * and reported by the reproducer's own deviation readout. */
    for (const char *mode : {"tf32", "bf16"}) {
        queued = h3d_set_vae_option(gpu, "gemm", mode) && run(option);
        if (!queued) { fprintf(stderr, "gemm %s arm: %s\n", mode, error); }
        float worst = 0.0f, largest = 0.0f;
        bool finite = true;
        for (size_t index = 0; index < outputs; index++) {
            finite = finite && std::isfinite(actual[index]);
            worst = std::max(worst, std::fabs(actual[index] - reference[index]));
            largest = std::max(largest, std::fabs(reference[index]));
        }
        char name[96];
        snprintf(name, sizeof(name), "video VAE %s matmul is finite and inside the bound", mode);
        check(queued && finite && worst <= 0.05f * largest, name);
        h3d_set_vae_option(gpu, "gemm", "fp32");
    }

    reset_vae_options(gpu);
    h3d_free(input); h3d_free(weight); h3d_free(bias);
    h3d_free(plain); h3d_free(option);
}

static void text_embedding_rows(h3t_context *gpu, uint32_t rows, uint32_t width) {    char error[512] = {};
    constexpr uint32_t vocab = 3;
    auto *weight = h3t_alloc(gpu, H3T_BF16, (size_t)vocab * width, error, sizeof(error));
    auto *output = h3t_alloc(gpu, H3T_BF16, (size_t)rows * width, error, sizeof(error));
    if (!weight || !output) { fprintf(stderr, "%s\n", error); exit(2); }
    std::vector<uint16_t> source((size_t)vocab * width);
    std::vector<uint16_t> expected((size_t)rows * width), actual(expected.size(), 0xffff);
    std::vector<uint32_t> ids(rows);
    for (size_t i = 0; i < source.size(); ++i) { source[i] = 0x3f80 + (i % 128); }
    for (uint32_t row = 0; row < rows; ++row) {
        ids[row] = row % vocab;
        for (uint32_t column = 0; column < width; ++column) {
            expected[(size_t)row * width + column] = source[(size_t)ids[row] * width + column];
        }
    }
    int prepared = h3t_upload(gpu, weight, source.data(), error, sizeof(error)) &&
                   h3t_upload(gpu, output, actual.data(), error, sizeof(error));
    if (!prepared) { fprintf(stderr, "%s\n", error); exit(2); }
    int ok = h3t_embed(gpu, output, weight, ids.data(), rows, vocab, width, error, sizeof(error));
    int synced = h3t_sync(gpu, error, sizeof(error));
    bool read = ok && synced && h3t_read(gpu, actual.data(), output, error, sizeof(error));
    if (!ok || !synced || !read) { fprintf(stderr, "text embedding: %s\n", error); }
    check(read && actual == expected, "text embedding covers every token and column");
    h3t_free(weight); h3t_free(output);
}

static void short_text_gqa_axes(h3t_context *gpu) {
    char error[512] = {};
    constexpr uint32_t rows = 17, query_heads = 4, kv_heads = 2, width = 4;
    constexpr size_t query_count = (size_t)rows * query_heads * width;
    constexpr size_t kv_count = (size_t)rows * kv_heads * width;
    auto *query = h3t_alloc(gpu, H3T_BF16, query_count, error, sizeof(error));
    auto *key = h3t_alloc(gpu, H3T_BF16, kv_count, error, sizeof(error));
    auto *value = h3t_alloc(gpu, H3T_BF16, kv_count, error, sizeof(error));
    auto *output = h3t_alloc(gpu, H3T_BF16, query_count, error, sizeof(error));
    if (!query || !key || !value || !output) { fprintf(stderr, "%s\n", error); exit(2); }
    std::vector<uint16_t> zeros(query_count, 0), source(kv_count);
    std::vector<uint16_t> expected(query_count), actual(query_count, 0xffff);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t head = 0; head < kv_heads; ++head) {
            for (uint32_t dimension = 0; dimension < width; ++dimension) {
                // Powers of two retain an exact causal mean for every row.
                source[((size_t)row * kv_heads + head) * width + dimension] =
                    0x3f80 + (head * width + dimension) * 128;
            }
        }
        for (uint32_t head = 0; head < query_heads; ++head) {
            for (uint32_t dimension = 0; dimension < width; ++dimension) {
                uint32_t kv_head = head / (query_heads / kv_heads);
                expected[((size_t)row * query_heads + head) * width + dimension] =
                    source[((size_t)row * kv_heads + kv_head) * width + dimension];
            }
        }
    }
    int prepared = h3t_upload(gpu, query, zeros.data(), error, sizeof(error)) &&
                   h3t_upload(gpu, key, zeros.data(), error, sizeof(error)) &&
                   h3t_upload(gpu, value, source.data(), error, sizeof(error)) &&
                   h3t_upload(gpu, output, actual.data(), error, sizeof(error));
    if (!prepared) { fprintf(stderr, "%s\n", error); exit(2); }
    int ok = h3t_gqa_causal(gpu, output, query, key, value, rows, query_heads,
                             kv_heads, width, 0.5f, error, sizeof(error));
    int synced = h3t_sync(gpu, error, sizeof(error));
    bool read = ok && synced && h3t_read(gpu, actual.data(), output, error, sizeof(error));
    if (!ok || !synced || !read) { fprintf(stderr, "text GQA: %s\n", error); }
    check(read && actual == expected, "causal text GQA preserves row and grouped-head axes");
    h3t_free(query); h3t_free(key); h3t_free(value); h3t_free(output);
}

static void vision_qkv_rows(h3v_context *gpu, uint32_t rows, uint32_t head_dim) {
    char error[512] = {};
    constexpr uint32_t heads = 2;
    const size_t count = (size_t)rows * heads * head_dim;
    auto *qkv = h3v_alloc(gpu, count * 3, error, sizeof(error));
    auto *query = h3v_alloc(gpu, count, error, sizeof(error));
    auto *key = h3v_alloc(gpu, count, error, sizeof(error));
    auto *value = h3v_alloc(gpu, count, error, sizeof(error));
    auto *rope = h3v_alloc(gpu, 0, error, sizeof(error));
    if (!qkv || !query || !key || !value || !rope) { fprintf(stderr, "%s\n", error); exit(2); }
    std::vector<uint16_t> source(count * 3), actual(count, 0xffff);
    std::vector<uint16_t> expected[3] = {
        std::vector<uint16_t>(count), std::vector<uint16_t>(count), std::vector<uint16_t>(count)
    };
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t part = 0; part < 3; ++part) {
            for (uint32_t head = 0; head < heads; ++head) {
                for (uint32_t dim = 0; dim < head_dim; ++dim) {
                    uint16_t bits = 0x3f80 + ((part * 31 + row + head * 17 + dim) % 128);
                    source[((size_t)row * 3 * heads + part * heads + head) * head_dim + dim] = bits;
                    expected[part][((size_t)head * rows + row) * head_dim + dim] = bits;
                }
            }
        }
    }
    int prepared = h3v_upload(gpu, qkv, source.data(), error, sizeof(error)) &&
                   h3v_upload(gpu, query, actual.data(), error, sizeof(error)) &&
                   h3v_upload(gpu, key, actual.data(), error, sizeof(error)) &&
                   h3v_upload(gpu, value, actual.data(), error, sizeof(error));
    if (!prepared) { fprintf(stderr, "%s\n", error); exit(2); }
    int ok = h3v_qkv_rope(gpu, query, key, value, qkv, rope, rope, rows, heads,
                          head_dim, 0, error, sizeof(error));
    int synced = h3v_sync(gpu, error, sizeof(error));
    bool matched = ok && synced;
    h3v_tensor *outputs[] = {query, key, value};
    for (uint32_t part = 0; part < 3; ++part) {
        matched = matched && h3v_read(gpu, actual.data(), outputs[part], error, sizeof(error)) &&
                  actual == expected[part];
    }
    if (!ok || !synced) { fprintf(stderr, "vision QKV: %s\n", error); }
    check(matched, "vision QKV covers every row, head and dimension");
    h3v_free(qkv); h3v_free(query); h3v_free(key); h3v_free(value); h3v_free(rope);
}

static void short_vision_sdpa_axes(h3v_context *gpu) {
    char error[512] = {};
    constexpr uint32_t rows = 17, heads = 2, width = 4;
    constexpr size_t count = (size_t)rows * heads * width;
    auto *zero = h3v_alloc(gpu, count, error, sizeof(error));
    auto *value = h3v_alloc(gpu, count, error, sizeof(error));
    auto *output = h3v_alloc(gpu, count, error, sizeof(error));
    if (!zero || !value || !output) { fprintf(stderr, "%s\n", error); exit(2); }
    std::vector<uint16_t> zeros(count, 0), source(count), expected(count), actual(count, 0xffff);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t head = 0; head < heads; ++head) {
            for (uint32_t dim = 0; dim < width; ++dim) {
                uint16_t bits = 0x3f80 + (head * width + dim) * 128;
                source[((size_t)head * rows + row) * width + dim] = bits;
                expected[((size_t)row * heads + head) * width + dim] = bits;
            }
        }
    }
    int prepared = h3v_upload(gpu, zero, zeros.data(), error, sizeof(error)) &&
                   h3v_upload(gpu, value, source.data(), error, sizeof(error)) &&
                   h3v_upload(gpu, output, actual.data(), error, sizeof(error));
    if (!prepared) { fprintf(stderr, "%s\n", error); exit(2); }
    int ok = h3v_sdpa(gpu, output, zero, zero, value, rows, heads, width, 0.5f,
                      error, sizeof(error));
    int synced = h3v_sync(gpu, error, sizeof(error));
    bool read = ok && synced && h3v_read(gpu, actual.data(), output, error, sizeof(error));
    if (!ok || !synced || !read) { fprintf(stderr, "vision SDPA: %s\n", error); }
    check(read && actual == expected, "vision SDPA preserves head and row axes");
    h3v_free(zero); h3v_free(value); h3v_free(output);
}


/* ---------------------------------------------------------------------------
 * Sol-Attn's search step against the reference's own rule, transcribed here as
 * the oracle. NVIDIA Sana `sol-engine` @ `6e4a440`:
 *   preprocess.py:280-322   kc = block mean of K, vc = the block SUM of V,
 *                           both stored BF16
 *   preprocess.py:96-135    kc_mean, kc_var_diag over blocks, from BF16 kc
 *   preprocess.py:143-186   threshold = mean + TAU*std in log2 space
 *   triton_ref/fwd.py:250-266  col_mean = the column mean of the route logits,
 *                           summed over the query block's rows
 *   selector.py:132-142     exact = col_mean > threshold or |q-n| <= 1
 *   interface.py:134-142    the sink block range, forced exact
 * ------------------------------------------------------------------------- */

static uint32_t sol_rand(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return (*state >> 8) & 0xffffffu;
}

static float sol_unit(uint32_t *state) {
    return (float)sol_rand(state) / 16777216.0f;
}

/* Round to nearest even, which is what `__float2bfloat16` does on the device;
 * the pooled summaries are compared bit for bit through this. */
static uint16_t sol_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static float sol_float(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

struct SolData {
    uint32_t sequence = 0, heads = 0, blocks = 0, words = 0;
    std::vector<uint16_t> query, key, value;
};

/* Blocks carry a per-block signal, so the route is a mix of exact and skipped
 * rather than degenerate; V is an order of magnitude above K, so a summary that
 * is a mean where the reference takes a sum cannot compare equal. */
static void sol_build(SolData &data, uint32_t sequence, uint32_t heads,
                      uint32_t seed) {
    data.sequence = sequence;
    data.heads = heads;
    data.blocks = (sequence + H3D_SOL_BLOCK - 1) / H3D_SOL_BLOCK;
    data.words = (data.blocks + H3D_SOL_GROUP - 1) / H3D_SOL_GROUP;
    const size_t elements = (size_t)sequence * heads * H3D_SOL_HEAD_DIM;
    data.query.assign(elements, 0);
    data.key.assign(elements, 0);
    data.value.assign(elements, 0);
    uint32_t state = seed;
    for (uint32_t head = 0; head < heads; ++head) {
        for (uint32_t block = 0; block < data.blocks; ++block) {
            const float signal = sol_unit(&state);
            const uint32_t first = block * H3D_SOL_BLOCK;
            const uint32_t rows = std::min<uint32_t>(H3D_SOL_BLOCK, sequence - first);
            for (uint32_t row = 0; row < rows; ++row) {
                for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
                    const size_t at =
                        ((size_t)head * sequence + first + row) * H3D_SOL_HEAD_DIM + dim;
                    data.key[at] = sol_bf16(0.5f * signal +
                                            0.02f * (sol_unit(&state) - 0.5f));
                    data.value[at] = sol_bf16(2.0f * sol_unit(&state));
                    data.query[at] = sol_bf16(0.5f + sol_unit(&state));
                }
            }
        }
    }
}

struct SolExpect {
    std::vector<float> kc, vc;      /* [heads][blocks][128] */
    std::vector<float> threshold;   /* [blocks][heads] */
    std::vector<float> columns;     /* [q][heads][n], the reference's row order */
    std::vector<float> centroid;    /* [q][heads][n], the port's centroid order */
    std::vector<uint32_t> mask, count;
};

static void sol_expect(const SolData &data, float scale, float tau,
                       uint32_t sink_start, uint32_t sink_tokens,
                       SolExpect &out) {
    const uint32_t sequence = data.sequence, heads = data.heads;
    const uint32_t blocks = data.blocks;
    const size_t summary = (size_t)heads * blocks * H3D_SOL_HEAD_DIM;
    out.kc.assign(summary, 0.0f);
    out.vc.assign(summary, 0.0f);
    for (uint32_t head = 0; head < heads; ++head) {
        for (uint32_t block = 0; block < blocks; ++block) {
            const uint32_t first = block * H3D_SOL_BLOCK;
            const uint32_t rows = std::min<uint32_t>(H3D_SOL_BLOCK, sequence - first);
            for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
                float key_sum = 0.0f, value_sum = 0.0f;
                for (uint32_t row = 0; row < rows; ++row) {
                    const size_t at =
                        ((size_t)head * sequence + first + row) * H3D_SOL_HEAD_DIM + dim;
                    key_sum += sol_float(data.key[at]);
                    value_sum += sol_float(data.value[at]);
                }
                const size_t into =
                    ((size_t)head * blocks + block) * H3D_SOL_HEAD_DIM + dim;
                out.kc[into] = sol_float(sol_bf16(key_sum / (float)rows));
                out.vc[into] = sol_float(sol_bf16(value_sum));
            }
        }
    }
    const float log2_scale = scale * 1.4426950408889634f;
    out.threshold.assign((size_t)blocks * heads, 0.0f);
    out.columns.assign((size_t)blocks * heads * blocks, 0.0f);
    out.centroid.assign((size_t)blocks * heads * blocks, 0.0f);
    for (uint32_t head = 0; head < heads; ++head) {
        std::vector<float> mean_d(H3D_SOL_HEAD_DIM, 0.0f);
        std::vector<float> var_d(H3D_SOL_HEAD_DIM, 0.0f);
        for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
            float total = 0.0f, total_sq = 0.0f;
            for (uint32_t block = 0; block < blocks; ++block) {
                const float value =
                    out.kc[((size_t)head * blocks + block) * H3D_SOL_HEAD_DIM + dim];
                total += value;
                total_sq += value * value;
            }
            const float mean = total / (float)blocks;
            float variance = total_sq / (float)blocks - mean * mean;
            mean_d[dim] = mean;
            var_d[dim] = variance > 0.0f ? variance : 0.0f;
        }
        for (uint32_t q_block = 0; q_block < blocks; ++q_block) {
            const uint32_t first = q_block * H3D_SOL_BLOCK;
            const uint32_t rows = std::min<uint32_t>(H3D_SOL_BLOCK, sequence - first);
            std::vector<float> q_bar(H3D_SOL_HEAD_DIM, 0.0f);
            for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
                float total = 0.0f;
                for (uint32_t row = 0; row < rows; ++row) {
                    total += sol_float(
                        data.query[((size_t)head * sequence + first + row) *
                                       H3D_SOL_HEAD_DIM + dim]);
                }
                q_bar[dim] = total / (float)rows;
            }
            float mean = 0.0f, variance = 0.0f;
            for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
                mean += q_bar[dim] * mean_d[dim];
                variance += q_bar[dim] * q_bar[dim] * var_d[dim];
            }
            const float std =
                sqrtf((variance > 0.0f ? variance : 0.0f) * log2_scale * log2_scale +
                      1.0e-6f);
            out.threshold[(size_t)q_block * heads + head] =
                mean * log2_scale + tau * std;
            for (uint32_t block = 0; block < blocks; ++block) {
                float column = 0.0f;
                for (uint32_t row = 0; row < rows; ++row) {
                    float dot = 0.0f;
                    for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
                        dot += sol_float(
                                   data.query[((size_t)head * sequence + first + row) *
                                                  H3D_SOL_HEAD_DIM + dim]) *
                               out.kc[((size_t)head * blocks + block) *
                                          H3D_SOL_HEAD_DIM + dim];
                    }
                    column += dot;
                }
                const size_t into = ((size_t)q_block * heads + head) * blocks + block;
                out.columns[into] = column * log2_scale / (float)rows;
                float centroid = 0.0f;
                for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
                    centroid += q_bar[dim] *
                                out.kc[((size_t)head * blocks + block) *
                                           H3D_SOL_HEAD_DIM + dim];
                }
                out.centroid[into] = centroid * log2_scale;
            }
        }
    }
    out.mask.assign((size_t)blocks * heads * data.words, 0u);
    out.count.assign((size_t)blocks * heads, 0u);
    /* `_sink_block_range` (`interface.py:134-142`): a zero sink is the empty
     * range, so no block is forced exact. */
    const uint32_t sink_first =
        sink_tokens ? sink_start / H3D_SOL_BLOCK : blocks;
    const uint32_t sink_last =
        sink_tokens ? (sink_start + sink_tokens + H3D_SOL_BLOCK - 1) / H3D_SOL_BLOCK
                    : blocks;
    for (uint32_t q_block = 0; q_block < blocks; ++q_block) {
        for (uint32_t head = 0; head < heads; ++head) {
            uint32_t total = 0;
            for (uint32_t block = 0; block < blocks; ++block) {
                const float column =
                    out.columns[((size_t)q_block * heads + head) * blocks + block];
                const uint32_t distance =
                    q_block > block ? q_block - block : block - q_block;
                const bool sink = sink_last > sink_first && block >= sink_first &&
                                  block < sink_last;
                if (distance <= 1 || sink ||
                    column > out.threshold[(size_t)q_block * heads + head]) {
                    out.mask[((size_t)q_block * heads + head) * data.words +
                             block / H3D_SOL_GROUP] |= 1u << (block % H3D_SOL_GROUP);
                    total += 1;
                }
            }
            out.count[(size_t)q_block * heads + head] = total;
        }
    }
}

struct SolPlan {
    h3d_sol_plan plan;
    h3d_tensor *kc, *vc, *mask, *count, *threshold, *query_bar, *kc_mean, *kc_var;
};

static bool sol_alloc(h3d_context *gpu, const SolData &data, SolPlan &out,
                      char *error, size_t size) {
    const size_t summary = (size_t)data.heads * data.blocks * H3D_SOL_HEAD_DIM;
    out.kc = h3d_alloc(gpu, H3D_BF16, summary, error, size);
    out.vc = h3d_alloc(gpu, H3D_BF16, summary, error, size);
    out.mask = h3d_alloc(gpu, H3D_U32, (size_t)data.blocks * data.heads * data.words,
                         error, size);
    out.count = h3d_alloc(gpu, H3D_U32, (size_t)data.blocks * data.heads, error, size);
    out.threshold = h3d_alloc(gpu, H3D_F32, (size_t)data.blocks * data.heads, error, size);
    out.query_bar = h3d_alloc(gpu, H3D_F32, summary, error, size);
    out.kc_mean = h3d_alloc(gpu, H3D_F32, (size_t)data.heads * H3D_SOL_HEAD_DIM,
                            error, size);
    out.kc_var = h3d_alloc(gpu, H3D_F32, (size_t)data.heads * H3D_SOL_HEAD_DIM,
                           error, size);
    out.plan = {out.kc, out.vc, out.mask, out.count, out.threshold,
                out.query_bar, out.kc_mean, out.kc_var};
    return out.kc && out.vc && out.mask && out.count && out.threshold &&
           out.query_bar && out.kc_mean && out.kc_var;
}

static void sol_free(SolPlan &plan) {
    h3d_tensor *all[] = {plan.kc, plan.vc, plan.mask, plan.count,
                         plan.threshold, plan.query_bar, plan.kc_mean, plan.kc_var};
    for (h3d_tensor *tensor : all) {
        if (tensor) h3d_free(tensor);
    }
}

/* Runs the search step and reads all of it back. */
struct SolActual {
    std::vector<float> kc, vc, threshold, query_bar;
    std::vector<uint32_t> mask, count;
};

static bool sol_run(h3d_context *gpu, const SolData &data, float scale, float tau,
                    uint32_t sink_start, uint32_t sink_tokens, SolActual &out,
                    char *error, size_t size) {
    SolPlan plan;
    if (!sol_alloc(gpu, data, plan, error, size)) return false;
    const size_t elements = (size_t)data.sequence * data.heads * H3D_SOL_HEAD_DIM;
    auto *query = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    auto *key = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    auto *value = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    const size_t summary = (size_t)data.heads * data.blocks * H3D_SOL_HEAD_DIM;
    out.kc.assign(summary, 0.0f);
    out.vc.assign(summary, 0.0f);
    out.threshold.assign((size_t)data.blocks * data.heads, 0.0f);
    out.query_bar.assign(summary, 0.0f);
    out.mask.assign((size_t)data.blocks * data.heads * data.words, 0u);
    out.count.assign((size_t)data.blocks * data.heads, 0u);
    bool ok = query && key && value &&
              h3d_upload(gpu, query, data.query.data(), error, size) &&
              h3d_upload(gpu, key, data.key.data(), error, size) &&
              h3d_upload(gpu, value, data.value.data(), error, size) &&
              h3d_sol_search(gpu, &plan.plan, query, key, value, data.sequence,
                             data.heads, scale, tau, sink_start, sink_tokens,
                             error, size) &&
              h3d_sync(gpu, error, size);
    std::vector<uint16_t> key_out(summary), value_out(summary);
    ok = ok && h3d_read(gpu, key_out.data(), plan.kc, error, size) &&
         h3d_read(gpu, value_out.data(), plan.vc, error, size);
    if (ok) {
        for (size_t at = 0; at < summary; ++at) {
            out.kc[at] = sol_float(key_out[at]);
            out.vc[at] = sol_float(value_out[at]);
        }
    }
    ok = ok && h3d_read(gpu, out.threshold.data(), plan.threshold, error, size) &&
         h3d_read(gpu, out.query_bar.data(), plan.query_bar, error, size) &&
         h3d_read(gpu, out.mask.data(), plan.mask, error, size) &&
         h3d_read(gpu, out.count.data(), plan.count, error, size);
    if (!ok) fprintf(stderr, "Sol-Attn search: %s\n", error);
    if (query) h3d_free(query);
    if (key) h3d_free(key);
    if (value) h3d_free(value);
    sol_free(plan);
    return ok;
}

/* Compared bit for bit where the port fixed the reference's order (the pooled
 * summaries), and as a reported margin where F32 order is the port's own (the
 * threshold). The route itself is compared as the reference's rule computes it
 * from the raw query rows. */
static void sol_compare(const char *label, const SolData &data,
                        const SolExpect &expect, const SolActual &actual,
                        float scale, float tau) {

    bool summaries = expect.kc == actual.kc && expect.vc == actual.vc;
    check(summaries, "search pools kc and vc exactly as the reference");
    double worst = 0.0;
    for (size_t at = 0; at < expect.threshold.size(); ++at) {
        const double delta = std::abs((double)expect.threshold[at] -
                                      (double)actual.threshold[at]);
        const double relative = delta / (1.0 + std::abs((double)expect.threshold[at]));
        if (relative > worst) worst = relative;
    }
    printf("  [%s tau=%.2f] threshold worst relative delta %.3e\n", label, tau,
           worst);
    check(worst < 1.0e-6, "search threshold matches the reference's formula");
    (void)scale;
    const bool mask = expect.mask == actual.mask;
    const bool count = expect.count == actual.count;
    if (!mask) {
        uint32_t shown = 0;
        for (uint32_t q_block = 0; q_block < data.blocks && shown < 8; ++q_block) {
            for (uint32_t head = 0; head < data.heads && shown < 8; ++head) {
                for (uint32_t block = 0; block < data.blocks && shown < 8; ++block) {
                    const size_t word =
                        ((size_t)q_block * data.heads + head) * data.words +
                        block / H3D_SOL_GROUP;
                    const uint32_t bit = 1u << (block % H3D_SOL_GROUP);
                    if ((expect.mask[word] & bit) == (actual.mask[word] & bit)) {
                        continue;
                    }
                    const size_t into =
                        ((size_t)q_block * data.heads + head) * data.blocks + block;
                    printf("  [%s] diff q=%u head=%u block=%u want=%u got=%u "
                           "row_column=%.6f centroid_column=%.6f threshold=%.6f\n",
                           label, q_block, head, block,
                           !!(expect.mask[word] & bit), !!(actual.mask[word] & bit),
                           (double)expect.columns[into], (double)expect.centroid[into],
                           (double)expect.threshold[(size_t)q_block * data.heads + head]);
                    shown += 1;
                }
            }
        }
    }
    uint32_t exact = 0;
    for (uint32_t value : expect.count) {
        exact += value;
    }
    const double share = 100.0 * (double)exact /
                         (double)((size_t)data.blocks * data.heads * data.blocks);
    printf("  [%s tau=%.2f] exact blocks %u (%.1f%% of the pairs)\n", label, tau,
           exact, share);
    check(mask, "search routes the reference's exact block set");
    check(count, "search counts the reference's exact blocks");
    /* The reference's row order against the port's centroid order: the same
     * column mean, so the same decision, measured rather than asserted. */
    float worst_column = 0.0f;
    float tightest = std::numeric_limits<float>::max();
    for (size_t at = 0; at < expect.columns.size(); ++at) {
        const float delta = std::abs(expect.columns[at] - expect.centroid[at]);
        if (delta > worst_column) worst_column = delta;
        const size_t head = (at / data.blocks) % data.heads;
        const uint32_t q_block = (uint32_t)(at / ((size_t)data.heads * data.blocks));
        const float margin =
            std::abs(expect.columns[at] - expect.threshold[(size_t)q_block * data.heads + head]);
        if (margin < tightest) tightest = margin;
    }
    printf("  [%s tau=%.2f] column mean row-vs-centroid worst delta %.3e, "
           "tightest threshold margin %.3e\n",
           label, tau, (double)worst_column, (double)tightest);
    check(worst_column < tightest,
          "the centroid order cannot move a route wider than the tightest margin");

    /* Bits past the last KV block are never set: the reference's `valid`. */
    uint32_t stray = 0;
    for (uint32_t q_block = 0; q_block < data.blocks; ++q_block) {
        for (uint32_t head = 0; head < data.heads; ++head) {
            for (uint32_t word = 0; word < data.words; ++word) {
                const uint32_t bits =
                    actual.mask[((size_t)q_block * data.heads + head) * data.words + word];
                for (uint32_t bit = 0; bit < H3D_SOL_GROUP; ++bit) {
                    const uint32_t block = word * H3D_SOL_GROUP + bit;
                    if (block < data.blocks || !(bits & (1u << bit))) continue;
                    stray += 1;
                }
            }
        }
    }
    check(stray == 0, "search never routes a block past the sequence");
    uint32_t popcount = 0, mismatch = 0;
    for (uint32_t q_block = 0; q_block < data.blocks; ++q_block) {
        for (uint32_t head = 0; head < data.heads; ++head) {
            uint32_t total = 0;
            for (uint32_t word = 0; word < data.words; ++word) {
                total += __builtin_popcount(
                    actual.mask[((size_t)q_block * data.heads + head) * data.words + word]);
            }
            popcount += total;
            mismatch += total != actual.count[(size_t)q_block * data.heads + head];
        }
    }
    check(mismatch == 0, "search count is the mask's own population");
}

/* The random-shape gate: pooled summaries bit for bit, threshold to a reported
 * margin, and the routed set equal to the reference's rule. T=200 leaves a
 * final block of 8 rows, so the block length and the query length are
 * exercised at the tail. */
static void sol_search_matches_reference(h3d_context *gpu) {
    char error[512] = {};
    SolData data;
    sol_build(data, 200, 3, 20261028u);
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    for (float tau : {0.5f, 1.0f, 1.5f}) {
        SolExpect expect;
        sol_expect(data, scale, tau, 0, 0, expect);
        SolActual actual;
        if (!sol_run(gpu, data, scale, tau, 0, 0, actual, error, sizeof(error))) {
            check(false, "search runs the reference's rule");
            return;
        }
        sol_compare("random", data, expect, actual, scale, tau);
    }
}

/* The sink range: text and audio tokens ride as sinks and are routed exactly
 * for every query block (`interface.py:134-142`). */
static void sol_search_sink(h3d_context *gpu) {
    char error[512] = {};
    SolData data;
    sol_build(data, 400, 3, 777u);
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    const uint32_t sink_start = 200, sink_tokens = 100;
    SolExpect expect;
    sol_expect(data, scale, 1.25f, sink_start, sink_tokens, expect);
    SolActual actual;
    if (!sol_run(gpu, data, scale, 1.25f, sink_start, sink_tokens, actual, error,
                 sizeof(error))) {
        check(false, "search runs with a sink range");
        return;
    }
    sol_compare("sink", data, expect, actual, scale, 1.25f);
    const uint32_t first = sink_start / H3D_SOL_BLOCK;
    const uint32_t last = (sink_start + sink_tokens + H3D_SOL_BLOCK - 1) / H3D_SOL_BLOCK;
    uint32_t missed = 0;
    for (uint32_t q_block = 0; q_block < data.blocks; ++q_block) {
        for (uint32_t head = 0; head < data.heads; ++head) {
            for (uint32_t block = first; block < last; ++block) {
                const uint32_t bits =
                    actual.mask[((size_t)q_block * data.heads + head) * data.words +
                                block / H3D_SOL_GROUP];
                missed += !(bits & (1u << (block % H3D_SOL_GROUP)));
            }
        }
    }
    check(missed == 0, "search routes every sink block exactly");
}

/* The empty sink range. `_sink_block_range` (`interface.py:134-142`) returns
 * `[blocks, blocks)` whenever `sink_tokens` is zero, so no block is forced
 * exact. Rounding the token range up without that case gives `[0, 1)` for any
 * `sink_start` below one block, which forces block 0 for every query block -
 * a decision the reference never makes.
 *
 * Constant summaries make the expected set arithmetic instead of a second
 * transcription: every kc is 1.0, so every column mean is 128*log2_scale and
 * the threshold is 128*log2_scale + tau*sqrt(1e-6), which no off-band block
 * clears. Block 0 appearing more than one block away from the diagonal is
 * therefore the rounding bug, not the rule. */
static void sol_search_empty_sink(h3d_context *gpu) {
    char error[512] = {};
    SolData data;
    data.sequence = 512;
    data.heads = 2;
    data.blocks = 8;
    data.words = 1;
    const size_t elements = (size_t)data.sequence * data.heads * H3D_SOL_HEAD_DIM;
    data.key.assign(elements, sol_bf16(1.0f));
    data.value.assign(elements, sol_bf16(0.5f));
    data.query.assign(elements, sol_bf16(1.0f));
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    SolActual actual;
    if (!sol_run(gpu, data, scale, 1.5f, 1, 0, actual, error, sizeof(error))) {
        check(false, "search runs on an empty sink range");
        return;
    }
    uint32_t mismatch = 0;
    for (uint32_t q_block = 0; q_block < data.blocks; ++q_block) {
        for (uint32_t head = 0; head < data.heads; ++head) {
            const uint32_t bits =
                actual.mask[((size_t)q_block * data.heads + head) * data.words];
            uint32_t set = 0;
            for (uint32_t block = 0; block < data.blocks; ++block) {
                const uint32_t distance =
                    q_block > block ? q_block - block : block - q_block;
                mismatch += (distance <= 1) != !!(bits & (1u << block));
                set += !!(bits & (1u << block));
            }
            mismatch += set != actual.count[(size_t)q_block * data.heads + head];
        }
    }
    check(mismatch == 0,
          "an empty sink range forces no block: the band is the whole route");

    /* The same call with no sink at all must give the identical mask: the
     * token range is empty either way. */
    SolActual plain;
    if (!sol_run(gpu, data, scale, 1.5f, 0, 0, plain, error, sizeof(error))) {
        check(false, "search runs with no sink at all");
        return;
    }
    check(plain.mask == actual.mask,
          "an empty sink range routes exactly as no sink at all");
}

/* The API boundary. The sink range lives inside the sequence
 * (`interface.py:134-142`), and the block count rounds `sequence` up by
 * `BLOCK - 1`, so a wrapping range or an overflowing shape must be refused
 * before a launch rather than reaching a kernel. The first case is the sum
 * `sink_start + sink_tokens` wrapping to a small number that would otherwise
 * pass an unchecked comparison. */
static void sol_search_rejects_bad_args(h3d_context *gpu) {
    char error[512] = {};
    SolData data;
    sol_build(data, 256, 1, 4242u);
    SolPlan plan;
    if (!sol_alloc(gpu, data, plan, error, sizeof(error))) {
        check(false, "the rejection cases allocate a plan");
        return;
    }
    const uint32_t sequence = data.sequence, heads = data.heads;
    const size_t elements = (size_t)sequence * heads * H3D_SOL_HEAD_DIM;
    auto *query = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *key = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *value = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    uint32_t refused = 0;
    refused += !h3d_sol_search(gpu, &plan.plan, query, key, value, sequence,
                               heads, scale, 1.0f, UINT32_MAX - 1, 4, error,
                               sizeof(error));
    refused += !h3d_sol_search(gpu, &plan.plan, query, key, value, sequence,
                               heads, scale, 1.0f, 0, sequence + 1, error,
                               sizeof(error));
    refused += !h3d_sol_search(gpu, &plan.plan, query, key, value, UINT32_MAX,
                               heads, scale, 1.0f, 0, 0, error,
                               sizeof(error));
    check(refused == 3, "the search step refuses a wrapping sink or shape");
    h3d_free(query);
    h3d_free(key);
    h3d_free(value);
    sol_free(plan);
}

/* A shape the suite never covered: 70 blocks is three mask words and two query
 * CTAs (H3D_SOL_QM is 64), and 4,446 tokens leaves the last block 30 rows long,
 * so the short tail sits inside the tile that also crosses the word boundary.
 * Both taus are compared against the oracle, and the oracle's row-order form
 * is what makes the second CTA and the third word independent evidence. */
static void sol_search_multiword(h3d_context *gpu) {
    char error[512] = {};
    SolData data;
    sol_build(data, 4446, 3, 20261028u);
    check(data.blocks == 70 && data.words == 3,
          "the multiword fixture is 70 blocks over three words");
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    for (float tau : {1.0f, 1.5f}) {
        SolExpect expect;
        sol_expect(data, scale, tau, 0, 0, expect);
        SolActual actual;
        if (!sol_run(gpu, data, scale, tau, 0, 0, actual, error, sizeof(error))) {
            check(false, "search runs across a second mask word");
            return;
        }
        sol_compare("multiword", data, expect, actual, scale, tau);
    }
}

/* The strict `>`. Constant K and Q made every off-band block clear the
 * threshold by a margin, so no case in this file could tell `>` from `>=`.
 * Here Q is one-hot on dimension 0 and the block summaries on that dimension
 * are [1, 0, 3, 0]: the across-block mean is exactly 1.0, so block 0's column
 * mean is exactly 1.0*log2_scale at tau=0, which is the threshold itself and
 * must not be routed. Block 2's 3.0 clears it for every query block, so the
 * expected set is the band plus that one block - arithmetic, not a second
 * transcription. */
static void sol_search_equal_threshold(h3d_context *gpu) {
    char error[512] = {};
    SolData data;
    data.sequence = 256;
    data.heads = 1;
    data.blocks = 4;
    data.words = 1;
    const size_t elements = (size_t)data.sequence * H3D_SOL_HEAD_DIM;
    data.key.assign(elements, sol_bf16(0.0f));
    data.value.assign(elements, sol_bf16(0.0f));
    data.query.assign(elements, sol_bf16(0.0f));
    const float signal[4] = {1.0f, 0.0f, 3.0f, 0.0f};
    for (uint32_t block = 0; block < data.blocks; ++block) {
        for (uint32_t row = 0; row < H3D_SOL_BLOCK; ++row) {
            const size_t at =
                ((size_t)block * H3D_SOL_BLOCK + row) * H3D_SOL_HEAD_DIM;
            data.key[at] = sol_bf16(signal[block]);
        }
    }
    for (uint32_t row = 0; row < data.sequence; ++row) {
        data.query[(size_t)row * H3D_SOL_HEAD_DIM] = sol_bf16(1.0f);
    }
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    SolActual actual;
    if (!sol_run(gpu, data, scale, 0.0f, 0, 0, actual, error, sizeof(error))) {
        check(false, "search runs with a column mean at the threshold");
        return;
    }
    uint32_t mismatch = 0;
    for (uint32_t q_block = 0; q_block < data.blocks; ++q_block) {
        const uint32_t bits = actual.mask[q_block * data.words];
        for (uint32_t block = 0; block < data.blocks; ++block) {
            const uint32_t distance =
                q_block > block ? q_block - block : block - q_block;
            const bool want = distance <= 1 || block == 2;
            mismatch += want != !!(bits & (1u << block));
        }
    }
    check(mismatch == 0,
          "a column mean equal to the threshold is not routed");
}

/* Constant K and constant Q: the summaries have no variance, so the threshold
 * is mean + tau*sqrt(1e-6) and no block off the diagonal band can clear it,
 * whatever tau is. The expected set is arithmetic, not a second transcription. */
static void sol_search_constant(h3d_context *gpu) {
    char error[512] = {};
    SolData data;
    data.sequence = 128;
    data.heads = 2;
    data.blocks = 2;
    data.words = 1;
    const size_t elements = 128 * 2 * H3D_SOL_HEAD_DIM;
    data.key.assign(elements, sol_bf16(1.0f));
    data.value.assign(elements, sol_bf16(0.5f));
    data.query.assign(elements, sol_bf16(1.0f));
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    SolActual actual;
    if (!sol_run(gpu, data, scale, 1.5f, 0, 0, actual, error, sizeof(error))) {
        check(false, "search runs on constant inputs");
        return;
    }
    uint32_t expected = 0, mismatch = 0;
    for (uint32_t q_block = 0; q_block < data.blocks; ++q_block) {
        for (uint32_t block = 0; block < data.blocks; ++block) {
            const uint32_t distance =
                q_block > block ? q_block - block : block - q_block;
            expected += distance <= 1;
        }
    }
    for (uint32_t q_block = 0; q_block < data.blocks; ++q_block) {
        for (uint32_t head = 0; head < data.heads; ++head) {
            const uint32_t bits =
                actual.mask[((size_t)q_block * data.heads + head) * data.words];
            uint32_t set = 0;
            for (uint32_t block = 0; block < data.blocks; ++block) {
                const uint32_t distance =
                    q_block > block ? q_block - block : block - q_block;
                set += !!(bits & (1u << block));
                mismatch += (distance <= 1) != !!(bits & (1u << block));
            }
            mismatch += set != actual.count[(size_t)q_block * data.heads + head];
        }
    }
    (void)expected;
    check(mismatch == 0,
          "constant summaries route the diagonal band and nothing else");
    /* The threshold itself: 128 * log2_scale + tau * sqrt(1e-6). */
    const float log2_scale = scale * 1.4426950408889634f;
    const float want = 128.0f * log2_scale + 1.5f * 1.0e-3f;
    float worst = 0.0f;
    for (float value : actual.threshold) {
        worst = std::max(worst, std::abs(value - want));
    }
    printf("  [constant] threshold vs 128*log2_scale + 1.5e-3: worst %.3e\n",
           (double)worst);
    check(worst < 2.0e-5f, "constant risk threshold matches its closed form");
}

/* One KV block whose summary spikes in dimension 0: with Q a one-hot row in the
 * same dimension, only that block's column mean clears mean + tau*std, so the
 * routed set is the band plus exactly that block. */
static void sol_search_spike(h3d_context *gpu) {
    char error[512] = {};
    SolData data;
    data.sequence = 512;
    data.heads = 1;
    data.blocks = 8;
    data.words = 1;
    const size_t elements = (size_t)data.sequence * H3D_SOL_HEAD_DIM;
    data.key.assign(elements, sol_bf16(0.0f));
    data.value.assign(elements, sol_bf16(0.0f));
    data.query.assign(elements, sol_bf16(0.0f));
    for (uint32_t row = 0; row < data.sequence; ++row) {
        data.query[(size_t)row * H3D_SOL_HEAD_DIM] = sol_bf16(1.0f);
    }
    for (uint32_t row = H3D_SOL_BLOCK; row < 2 * H3D_SOL_BLOCK; ++row) {
        data.key[(size_t)row * H3D_SOL_HEAD_DIM] = sol_bf16(1.0f);
    }
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    SolActual actual;
    if (!sol_run(gpu, data, scale, 1.0f, 0, 0, actual, error, sizeof(error))) {
        check(false, "search runs on a spiked summary");
        return;
    }
    uint32_t mismatch = 0;
    for (uint32_t q_block = 0; q_block < data.blocks; ++q_block) {
        const uint32_t bits = actual.mask[q_block * data.words];
        for (uint32_t block = 0; block < data.blocks; ++block) {
            const uint32_t distance =
                q_block > block ? q_block - block : block - q_block;
            const bool want = distance <= 1 || block == 1;
            mismatch += want != !!(bits & (1u << block));
        }
    }
    check(mismatch == 0,
          "a spiked summary is routed exactly and no other far block is");
}

/* Raising tau raises the threshold, so it can only remove blocks from the exact
 * set: a superset check across a sweep that needs no oracle at all. */
static void sol_search_tau_monotone(h3d_context *gpu) {
    char error[512] = {};
    SolData data;
    sol_build(data, 200, 3, 20261028u);
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    const float taus[] = {0.0f, 0.5f, 1.0f, 1.5f, 3.0f, 40.0f};
    SolActual previous;
    uint32_t added = 0, removed = 0;
    for (size_t step = 0; step < sizeof(taus) / sizeof(taus[0]); ++step) {
        SolActual actual;
        if (!sol_run(gpu, data, scale, taus[step], 0, 0, actual, error,
                     sizeof(error))) {
            check(false, "search runs the tau sweep");
            return;
        }
        uint32_t exact = 0;
        for (uint32_t value : actual.count) {
            exact += value;
        }
        float lowest = std::numeric_limits<float>::max(), highest = -lowest;
        for (float value : actual.threshold) {
            lowest = std::min(lowest, value);
            highest = std::max(highest, value);
        }
        printf("  [tau] %.1f routes %u of %u pairs, threshold %.4f..%.4f\n",
               (double)taus[step], exact,
               data.blocks * data.heads * data.blocks, (double)lowest,
               (double)highest);
        if (step) {
            for (size_t at = 0; at < actual.mask.size(); ++at) {
                added += __builtin_popcount(actual.mask[at] & ~previous.mask[at]);
                removed += __builtin_popcount(previous.mask[at] & ~actual.mask[at]);
            }
        }
        previous = actual;
    }
    printf("  [tau] the sweep removes %u bits and adds %u\n", removed, added);
    check(added == 0 && removed > 0,
          "raising tau only removes exact blocks");
}
/* ---------------------------------------------------------------------------
 * Sol-Attn's sparse forward against the reference's own walk, transcribed here
 * as the oracle. NVIDIA Sana `sol-engine` @ `6e4a440`, `triton_ref/fwd.py`:
 *   :88-108    per group of 32 KV blocks, the pooled scores
 *              q @ kc^T * scale*log2(e), and the group's exact/approximated
 *              split
 *   :109-131   the approximated blocks folded into the running softmax from
 *              the pooled summaries: o += P @ vc and row_sum += P *
 *              block_length, because vc is the block SUM of V
 *              (`preprocess.py:82`)
 *   :132-170   the group's exact blocks walked in increasing block order, one
 *              BLOCK = 64 key tile at a time: o += P @ v, row_sum += sum(P)
 *   :171-173   o / row_sum, stored BF16
 * The mask is an input here, not a rule this oracle recomputes: the split is
 * whatever the caller's mask states, so the consumer is testable apart from
 * the router.
 * ------------------------------------------------------------------------- */

/* Rounds to BF16 where the reference's `tl.dot(probability.to(v.dtype), v)`
 * does, so the only difference left between the two sides is the reduction
 * order of the F32 accumulators. */
static void sol_forward_expect(const SolData &data, const SolExpect &summary,
                               const std::vector<uint32_t> &mask, float scale,
                               std::vector<uint16_t> &out) {
    const uint32_t sequence = data.sequence, heads = data.heads;
    const uint32_t blocks = data.blocks;
    const float log2_scale = scale * 1.4426950408889634f;
    out.assign((size_t)sequence * heads * H3D_SOL_HEAD_DIM, 0);
    std::vector<float> output((size_t)H3D_SOL_BLOCK * H3D_SOL_HEAD_DIM);
    float row_sum[H3D_SOL_BLOCK];
    float row_max[H3D_SOL_BLOCK];
    float scores[H3D_SOL_BLOCK][H3D_SOL_GROUP];
    for (uint32_t head = 0; head < heads; ++head) {
        const uint16_t *query = data.query.data() +
                                (size_t)head * sequence * H3D_SOL_HEAD_DIM;
        const uint16_t *key = data.key.data() +
                              (size_t)head * sequence * H3D_SOL_HEAD_DIM;
        const uint16_t *value = data.value.data() +
                                (size_t)head * sequence * H3D_SOL_HEAD_DIM;
        for (uint32_t q_block = 0; q_block < blocks; ++q_block) {
            const uint32_t first = q_block * H3D_SOL_BLOCK;
            const uint32_t rows = std::min<uint32_t>(H3D_SOL_BLOCK, sequence - first);
            std::fill(output.begin(), output.end(), 0.0f);
            for (uint32_t row = 0; row < H3D_SOL_BLOCK; ++row) {
                row_sum[row] = 0.0f;
                row_max[row] = -INFINITY;
            }
            for (uint32_t group_start = 0; group_start < blocks;
                 group_start += H3D_SOL_GROUP) {
                const uint32_t bits =
                    mask[((size_t)q_block * heads + head) * data.words +
                         group_start / H3D_SOL_GROUP];
                bool exact[H3D_SOL_GROUP];
                uint32_t approximated = 0;
                for (uint32_t n = 0; n < H3D_SOL_GROUP; ++n) {
                    const uint32_t block = group_start + n;
                    exact[n] = block < blocks && (bits & (1u << n));
                    approximated += block < blocks && !exact[n];
                }
                /* `has_approximate` (`fwd.py:256-262`): a group with nothing
                 * approximated leaves the running maximum untouched. */
                if (approximated) {
                    for (uint32_t row = 0; row < rows; ++row) {
                        const uint16_t *q = query +
                            (size_t)(first + row) * H3D_SOL_HEAD_DIM;
                        for (uint32_t n = 0; n < H3D_SOL_GROUP; ++n) {
                            const uint32_t block = group_start + n;
                            if (block >= blocks || exact[n]) {
                                scores[row][n] = -INFINITY;
                                continue;
                            }
                            const float *kc = summary.kc.data() +
                                ((size_t)head * blocks + block) * H3D_SOL_HEAD_DIM;
                            float dot = 0.0f;
                            for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
                                dot += sol_float(q[dim]) * kc[dim];
                            }
                            scores[row][n] = dot * log2_scale;
                        }
                    }
                    for (uint32_t row = 0; row < rows; ++row) {
                        float next = row_max[row];
                        for (uint32_t n = 0; n < H3D_SOL_GROUP; ++n) {
                            next = std::max(next, scores[row][n]);
                        }
                        const float alpha = std::exp2(row_max[row] - next);
                        float *into = output.data() +
                                      (size_t)row * H3D_SOL_HEAD_DIM;
                        for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
                            into[dim] *= alpha;
                        }
                        for (uint32_t n = 0; n < H3D_SOL_GROUP; ++n) {
                            if (scores[row][n] == -INFINITY) continue;
                            const float probability =
                                std::exp2(scores[row][n] - next);
                            const float packed =
                                sol_float(sol_bf16(probability));
                            const uint32_t block = group_start + n;
                            const float length = (float)std::min<uint32_t>(
                                H3D_SOL_BLOCK, sequence - block * H3D_SOL_BLOCK);
                            const float *vc = summary.vc.data() +
                                ((size_t)head * blocks + block) * H3D_SOL_HEAD_DIM;
                            row_sum[row] += probability * length;
                            for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
                                into[dim] += packed * vc[dim];
                            }
                        }
                        row_max[row] = next;
                    }
                }
                for (uint32_t n = 0; n < H3D_SOL_GROUP; ++n) {
                    if (!exact[n]) continue;
                    const uint32_t kv_first = (group_start + n) * H3D_SOL_BLOCK;
                    for (uint32_t row = 0; row < rows; ++row) {
                        float column[H3D_SOL_BLOCK];
                        float next = row_max[row];
                        const uint16_t *q = query +
                            (size_t)(first + row) * H3D_SOL_HEAD_DIM;
                        for (uint32_t k = 0; k < H3D_SOL_BLOCK; ++k) {
                            const uint32_t token = kv_first + k;
                            if (token >= sequence) {
                                column[k] = -INFINITY;
                                continue;
                            }
                            const uint16_t *k_row = key +
                                (size_t)token * H3D_SOL_HEAD_DIM;
                            float dot = 0.0f;
                            for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
                                dot += sol_float(q[dim]) * sol_float(k_row[dim]);
                            }
                            column[k] = dot * log2_scale;
                            next = std::max(next, column[k]);
                        }
                        const float alpha = std::exp2(row_max[row] - next);
                        float *into = output.data() +
                                      (size_t)row * H3D_SOL_HEAD_DIM;
                        for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
                            into[dim] *= alpha;
                        }
                        row_sum[row] *= alpha;
                        for (uint32_t k = 0; k < H3D_SOL_BLOCK; ++k) {
                            if (column[k] == -INFINITY) continue;
                            const uint32_t token = kv_first + k;
                            const float probability = std::exp2(column[k] - next);
                            const float packed =
                                sol_float(sol_bf16(probability));
                            const uint16_t *v_row = value +
                                (size_t)token * H3D_SOL_HEAD_DIM;
                            row_sum[row] += probability;
                            for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
                                into[dim] += packed * sol_float(v_row[dim]);
                            }
                        }
                        row_max[row] = next;
                    }
                }
            }
            for (uint32_t row = 0; row < rows; ++row) {
                const float *from = output.data() +
                                    (size_t)row * H3D_SOL_HEAD_DIM;
                /* The port's output layout, `[token][head][dim]`, the same one
                 * `h3d_sdpa` writes (`short_attention_axes` gates it). */
                uint16_t *into = out.data() + ((size_t)(first + row) * heads +
                                               head) * H3D_SOL_HEAD_DIM;
                for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
                    into[dim] = sol_bf16(from[dim] / row_sum[row]);
                }
            }
        }
    }
}

/* The live elements only: a padded query row is never written by the kernel
 * and carries no meaning in the reference either. Both sides are in the port's
 * `[token][head][dim]` output layout. */
static void sol_forward_compare(const char *label, const SolData &data,
                                const std::vector<uint16_t> &expect,
                                const std::vector<uint16_t> &actual) {
    double worst = 0.0, squared = 0.0, energy = 0.0;
    size_t live = 0;
    uint32_t differing = 0;
    for (uint32_t head = 0; head < data.heads; ++head) {
        for (uint32_t block = 0; block < data.blocks; ++block) {
            const uint32_t first = block * H3D_SOL_BLOCK;
            const uint32_t rows =
                std::min<uint32_t>(H3D_SOL_BLOCK, data.sequence - first);
            for (uint32_t row = 0; row < rows; ++row) {
                const size_t at = ((size_t)(first + row) * data.heads + head) *
                                  H3D_SOL_HEAD_DIM;
                for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
                    const double want = sol_float(expect[at + dim]);
                    const double got = sol_float(actual[at + dim]);
                    const double delta = std::abs(want - got);
                    if (delta > worst) worst = delta;
                    squared += delta * delta;
                    energy += want * want;
                    differing += delta > 0.0;
                    live += 1;
                }
            }
        }
    }
    const double rms = std::sqrt(squared / (double)live);
    const double relative = rms / std::sqrt(energy / (double)live);
    printf("  [%s] worst %.3e, RMS %.3e, relative RMS %.3e, %u of %zu elements "
           "differ\n",
           label, worst, rms, relative, differing, live);
    check(relative < 1.0e-3, "the sparse forward matches the reference's walk");
}

/* The forward with the summaries and the mask the caller states: the search
 * step is not called, so a route the router cannot produce (no diagonal band
 * at all, nothing exact in a group) still reaches the consumer. */
static bool sol_forward_run(h3d_context *gpu, const SolData &data,
                            const SolExpect &summary,
                            const std::vector<uint32_t> &mask, float scale,
                            std::vector<uint16_t> &out, char *error,
                            size_t size) {
    SolPlan plan;
    if (!sol_alloc(gpu, data, plan, error, size)) return false;
    const size_t elements = (size_t)data.sequence * data.heads * H3D_SOL_HEAD_DIM;
    const size_t summaries = (size_t)data.heads * data.blocks * H3D_SOL_HEAD_DIM;
    auto *query = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    auto *key = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    auto *value = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    auto *output = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    std::vector<uint16_t> kc(summaries), vc(summaries);
    for (size_t at = 0; at < summaries; ++at) {
        kc[at] = sol_bf16(summary.kc[at]);
        vc[at] = sol_bf16(summary.vc[at]);
    }
    bool ok = query && key && value && output &&
              h3d_upload(gpu, query, data.query.data(), error, size) &&
              h3d_upload(gpu, key, data.key.data(), error, size) &&
              h3d_upload(gpu, value, data.value.data(), error, size) &&
              h3d_upload(gpu, plan.kc, kc.data(), error, size) &&
              h3d_upload(gpu, plan.vc, vc.data(), error, size) &&
              h3d_upload(gpu, plan.mask, mask.data(), error, size) &&
              h3d_sol_forward(gpu, output, &plan.plan, query, key, value,
                              data.sequence, data.heads, scale, error, size) &&
              h3d_sync(gpu, error, size);
    out.assign(elements, 0);
    ok = ok && h3d_read(gpu, out.data(), output, error, size);
    if (!ok) fprintf(stderr, "Sol-Attn forward: %s\n", error);
    if (query) h3d_free(query);
    if (key) h3d_free(key);
    if (value) h3d_free(value);
    if (output) h3d_free(output);
    sol_free(plan);
    return ok;
}

/* Search, then the forward, in the order the engine calls them; the mask is
 * returned so the oracle consumes the router's own decision. */
static bool sol_forward_search_run(h3d_context *gpu, const SolData &data,
                                   float scale, float tau, uint32_t sink_start,
                                   uint32_t sink_tokens, std::vector<uint16_t> &out,
                                   std::vector<uint32_t> &mask, char *error,
                                   size_t size) {
    SolPlan plan;
    if (!sol_alloc(gpu, data, plan, error, size)) return false;
    const size_t elements = (size_t)data.sequence * data.heads * H3D_SOL_HEAD_DIM;
    auto *query = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    auto *key = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    auto *value = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    auto *output = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    bool ok = query && key && value && output &&
              h3d_upload(gpu, query, data.query.data(), error, size) &&
              h3d_upload(gpu, key, data.key.data(), error, size) &&
              h3d_upload(gpu, value, data.value.data(), error, size) &&
              h3d_sol_search(gpu, &plan.plan, query, key, value, data.sequence,
                             data.heads, scale, tau, sink_start, sink_tokens,
                             error, size) &&
              h3d_sol_forward(gpu, output, &plan.plan, query, key, value,
                              data.sequence, data.heads, scale, error, size) &&
              h3d_sync(gpu, error, size);
    out.assign(elements, 0);
    mask.assign((size_t)data.blocks * data.heads * data.words, 0u);
    ok = ok && h3d_read(gpu, out.data(), output, error, size) &&
         h3d_read(gpu, mask.data(), plan.mask, error, size);
    if (!ok) fprintf(stderr, "Sol-Attn forward after search: %s\n", error);
    if (query) h3d_free(query);
    if (key) h3d_free(key);
    if (value) h3d_free(value);
    if (output) h3d_free(output);
    sol_free(plan);
    return ok;
}

static bool sol_dense_run(h3d_context *gpu, const SolData &data, float scale,
                          std::vector<uint16_t> &out, char *error, size_t size) {
    const size_t elements = (size_t)data.sequence * data.heads * H3D_SOL_HEAD_DIM;
    auto *query = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    auto *key = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    auto *value = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    auto *output = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    bool ok = query && key && value && output &&
              h3d_upload(gpu, query, data.query.data(), error, size) &&
              h3d_upload(gpu, key, data.key.data(), error, size) &&
              h3d_upload(gpu, value, data.value.data(), error, size) &&
              h3d_sdpa(gpu, output, query, key, value, data.sequence, data.heads,
                       H3D_SOL_HEAD_DIM, scale, error, size) &&
              h3d_sync(gpu, error, size);
    out.assign(elements, 0);
    ok = ok && h3d_read(gpu, out.data(), output, error, size);
    if (!ok) fprintf(stderr, "dense attention: %s\n", error);
    if (query) h3d_free(query);
    if (key) h3d_free(key);
    if (value) h3d_free(value);
    if (output) h3d_free(output);
    return ok;
}

static void sol_report_sparsity(const char *label, const SolData &data,
                                const std::vector<uint32_t> &mask) {
    uint64_t exact = 0, pairs = (uint64_t)data.blocks * data.heads * data.blocks;
    for (uint32_t head = 0; head < data.heads; ++head) {
        for (uint32_t q_block = 0; q_block < data.blocks; ++q_block) {
            for (uint32_t word = 0; word < data.words; ++word) {
                exact += __builtin_popcount(
                    mask[((size_t)q_block * data.heads + head) * data.words + word]);
            }
        }
    }
    printf("  [%s] %llu of %llu block pairs exact, %.1f%% folded\n", label,
           (unsigned long long)exact, (unsigned long long)pairs,
           100.0 - 100.0 * (double)exact / (double)pairs);
}

/* The end-to-end case: the router's own mask, both taus, against the oracle's
 * transcription of `fwd.py:88-173`. */
static void sol_forward_matches_reference(h3d_context *gpu) {
    char error[512] = {};
    SolData data;
    sol_build(data, 200, 3, 20261028u);
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    for (float tau : {0.5f, 1.5f}) {
        SolExpect expect;
        sol_expect(data, scale, tau, 0, 0, expect);
        std::vector<uint16_t> actual;
        std::vector<uint32_t> mask;
        if (!sol_forward_search_run(gpu, data, scale, tau, 0, 0, actual, mask,
                                    error, sizeof(error))) {
            check(false, "the sparse forward runs after the search");
            return;
        }
        std::vector<uint16_t> want;
        sol_forward_expect(data, expect, mask, scale, want);
        sol_report_sparsity("random", data, mask);
        sol_forward_compare("random", data, want, actual);
    }
}

/* Every block exact: the folded path contributes nothing and the walk is the
 * dense one, so the sparse forward must agree with the dense attention at the
 * same shape. This is the property that says the exact half is a plain
 * attention; the oracle case above says the same for the folded half. */
static void sol_forward_all_exact(h3d_context *gpu) {
    char error[512] = {};
    SolData data;
    sol_build(data, 200, 3, 20261028u);
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    SolExpect expect;
    sol_expect(data, scale, 1.0f, 0, 0, expect);
    std::vector<uint32_t> mask((size_t)data.blocks * data.heads * data.words, 0u);
    for (uint32_t word = 0; word < data.words; ++word) {
        const uint32_t first = word * H3D_SOL_GROUP;
        const uint32_t rest = data.blocks - first;
        const uint32_t bits = rest >= H3D_SOL_GROUP ? ~0u : (1u << rest) - 1u;
        for (uint32_t q_block = 0; q_block < data.blocks; ++q_block) {
            for (uint32_t head = 0; head < data.heads; ++head) {
                mask[((size_t)q_block * data.heads + head) * data.words + word] = bits;
            }
        }
    }
    std::vector<uint16_t> sparse, dense, want;
    if (!sol_forward_run(gpu, data, expect, mask, scale, sparse, error,
                         sizeof(error))) {
        check(false, "the sparse forward runs with every block exact");
        return;
    }
    sol_forward_expect(data, expect, mask, scale, want);
    sol_forward_compare("all-exact", data, want, sparse);
    if (!sol_dense_run(gpu, data, scale, dense, error, sizeof(error))) {
        check(false, "the dense attention runs at the fixture shape");
        return;
    }
    sol_forward_compare("all-exact vs dense", data, dense, sparse);
}

/* Two mask words, so the group loop crosses its 32-block boundary, and a tail
 * block of 4,446 - 2,048 = 52 rows that is shorter than the tile. */
static void sol_forward_multiword(h3d_context *gpu) {
    char error[512] = {};
    SolData data;
    sol_build(data, 2100, 2, 20261028u);
    check(data.blocks == 33 && data.words == 2,
          "the multiword forward fixture is 33 blocks over two words");
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    SolExpect expect;
    sol_expect(data, scale, 1.25f, 0, 0, expect);
    std::vector<uint16_t> actual;
    std::vector<uint32_t> mask;
    if (!sol_forward_search_run(gpu, data, scale, 1.25f, 0, 0, actual, mask, error,
                                sizeof(error))) {
        check(false, "the sparse forward runs across a second mask word");
        return;
    }
    std::vector<uint16_t> want;
    sol_forward_expect(data, expect, mask, scale, want);
    sol_report_sparsity("multiword", data, mask);
    sol_forward_compare("multiword", data, want, actual);
}

/* The consumer holds no policy of its own: the mask is read literally. Three
 * routes the router cannot produce, each against the oracle that consumes the
 * same bits.
 *   - the two far blocks of one word exact, with no diagonal band at all;
 *   - nothing exact, so every group folds;
 *   - one word entirely exact beside a word with nothing exact, which is the
 *     `has_approximate` skip and the fold in a single call. */
static void sol_forward_mask_is_literal(h3d_context *gpu) {
    char error[512] = {};
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    {
        SolData data;
        sol_build(data, 512, 1, 99u);
        SolExpect expect;
        sol_expect(data, scale, 1.0f, 0, 0, expect);
        const uint32_t far = (1u << 1) | (1u << 6);
        std::vector<uint32_t> mask((size_t)data.blocks * data.heads * data.words, far);
        std::vector<uint16_t> actual, want;
        if (!sol_forward_run(gpu, data, expect, mask, scale, actual, error,
                             sizeof(error))) {
            check(false, "the sparse forward runs on a hand-built mask");
            return;
        }
        sol_forward_expect(data, expect, mask, scale, want);
        sol_report_sparsity("far-blocks", data, mask);
        sol_forward_compare("far-blocks", data, want, actual);
    }
    {
        SolData data;
        sol_build(data, 512, 1, 100u);
        SolExpect expect;
        sol_expect(data, scale, 1.0f, 0, 0, expect);
        std::vector<uint32_t> mask((size_t)data.blocks * data.heads * data.words, 0u);
        std::vector<uint16_t> actual, want;
        if (!sol_forward_run(gpu, data, expect, mask, scale, actual, error,
                             sizeof(error))) {
            check(false, "the sparse forward runs with nothing exact");
            return;
        }
        sol_forward_expect(data, expect, mask, scale, want);
        sol_report_sparsity("all-folded", data, mask);
        sol_forward_compare("all-folded", data, want, actual);
    }
    {
        SolData data;
        sol_build(data, 2100, 1, 101u);
        SolExpect expect;
        sol_expect(data, scale, 1.0f, 0, 0, expect);
        std::vector<uint32_t> mask((size_t)data.blocks * data.heads * data.words, 0u);
        for (uint32_t q_block = 0; q_block < data.blocks; ++q_block) {
            mask[((size_t)q_block * data.heads + 0) * data.words + 0] = ~0u;
        }
        std::vector<uint16_t> actual, want;
        if (!sol_forward_run(gpu, data, expect, mask, scale, actual, error,
                             sizeof(error))) {
            check(false, "the sparse forward runs on a split word pair");
            return;
        }
        sol_forward_expect(data, expect, mask, scale, want);
        sol_report_sparsity("split", data, mask);
        sol_forward_compare("split", data, want, actual);
    }
}

/* The API boundary: the same refusals the search step's own kernel needs, plus
 * a plan whose summary buffers are absent. */
static void sol_forward_rejects_bad_args(h3d_context *gpu) {
    char error[512] = {};
    SolData data;
    sol_build(data, 256, 1, 4242u);
    SolPlan plan;
    if (!sol_alloc(gpu, data, plan, error, sizeof(error))) {
        check(false, "the forward's rejection cases allocate a plan");
        return;
    }
    const size_t elements = (size_t)data.sequence * data.heads * H3D_SOL_HEAD_DIM;
    auto *query = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *key = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *value = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *output = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *wide = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    uint32_t refused = 0;
    refused += !h3d_sol_forward(gpu, output, &plan.plan, query, key, value, 0,
                                data.heads, scale, error, sizeof(error));
    refused += !h3d_sol_forward(gpu, output, &plan.plan, query, key, value,
                                data.sequence, 0, scale, error, sizeof(error));
    refused += !h3d_sol_forward(gpu, output, &plan.plan, query, key, wide,
                                data.sequence, data.heads, scale, error,
                                sizeof(error));
    refused += !h3d_sol_forward(gpu, output, &plan.plan, query, key, value,
                                UINT32_MAX, data.heads, scale, error,
                                sizeof(error));
    h3d_sol_plan absent = plan.plan;
    absent.kc = nullptr;
    refused += !h3d_sol_forward(gpu, output, &absent, query, key, value,
                                data.sequence, data.heads, scale, error,
                                sizeof(error));
    check(refused == 5, "the sparse forward refuses a bad shape or a bad plan");
    h3d_free(query);
    h3d_free(key);
    h3d_free(value);
    h3d_free(output);
    h3d_free(wide);
    sol_free(plan);
}

/* ---------------------------------------------------------------------------
 * VSA, the rule the reference's H3 stage itself runs (the tau threshold above
 * is its the refiner refiner's). Read from the pinned FastVideo module:
 *   video_sparse_attn_h3.py:380-383   tile summary = the tile MEAN, F32
 *   video_sparse_attn_h3.py:629       scores = q_pool @ k_pool^T / sqrt(head_dim)
 *   video_sparse_attn.py:161-163      k_vid = max(1, min(ceil((1-s)*tiles), tiles))
 *   video_sparse_attn_h3.py:387-410   video columns compete for the k_vid
 *                                     largest scores, every prefix column and
 *                                     every prefix query row is forced, and
 *                                     s == 0 selects everything
 *
 * Deviations, named: `torch.topk`'s tie order is not documented, so this port
 * breaks ties by ascending column index; the prefix is this engine's own
 * text-and-audio span rounded up to whole 64-token blocks.
 * ------------------------------------------------------------------------- */

/* `compute_topk` (`video_sparse_attn.py:161-163`) in the reference's own double
 * precision, so a sparsity that lands exactly on a boundary keeps the same
 * count it does there. The sparsity arrives as a double for the same reason:
 * 0.9 rounded to F32 turns `ceil((1 - s) * 280)` into 29 where the reference's
 * own double gives 28. */
static uint32_t sol_vsa_keep(double sparsity, uint32_t tiles) {
    const double keep = std::ceil((1.0 - sparsity) * (double)tiles);
    if (keep < 1.0) return 1u;
    if (keep > (double)tiles) return tiles;
    return (uint32_t)keep;
}

/* The kernel's own reduction tree, so the gate compares the rule and not the
 * F32 order: lane `l` sums the row's dimensions l, l+32, l+64, l+96 in
 * sequence, then the butterfly adds the offsets 16, 8, 4, 2, 1 and lane 0
 * stores (`h3d_sol_vsa_kernel`). `fmaf` because nvcc contracts `p += q * k`
 * by default; the bench reports what the order moves against the reference's
 * raw-row means. */
static float sol_vsa_score(const float *query_bar, const float *kc,
                           float scale) {
    float part[H3D_SOL_WARP];
    for (uint32_t lane = 0; lane < H3D_SOL_WARP; ++lane) {
        float total = 0.0f;
        for (uint32_t step = 0; step < 4; ++step) {
            const uint32_t dim = step * H3D_SOL_WARP + lane;
            total = fmaf(query_bar[dim], kc[dim], total);
        }
        part[lane] = total;
    }
    for (uint32_t offset = H3D_SOL_WARP / 2; offset > 0; offset >>= 1) {
        float next[H3D_SOL_WARP];
        for (uint32_t lane = 0; lane < H3D_SOL_WARP; ++lane) {
            next[lane] = part[lane] + part[lane ^ offset];
        }
        for (uint32_t lane = 0; lane < H3D_SOL_WARP; ++lane) {
            part[lane] = next[lane];
        }
    }
    return part[0] * scale;
}

/* The selector alone, applied to the summaries the port's own pooling wrote:
 * a mask that differs is the selection's fault and not the pooling's, which
 * the solver's own tests gate separately. */
static void sol_vsa_expect(const SolData &data, const std::vector<float> &kc,
                           const std::vector<float> &query_bar, float scale,
                           double sparsity, uint32_t prefix_blocks,
                           SolExpect &out) {
    const uint32_t blocks = data.blocks, heads = data.heads;
    const uint32_t keep = sol_vsa_keep(sparsity, blocks - prefix_blocks);
    out.mask.assign((size_t)blocks * heads * data.words, 0u);
    out.count.assign((size_t)blocks * heads, 0u);
    for (uint32_t q_block = 0; q_block < blocks; ++q_block) {
        for (uint32_t head = 0; head < heads; ++head) {
            const float *row = &query_bar[((size_t)q_block * heads + head) *
                                          H3D_SOL_HEAD_DIM];
            std::vector<float> score(blocks, 0.0f);
            for (uint32_t block = 0; block < blocks; ++block) {
                score[block] = sol_vsa_score(
                    row, &kc[((size_t)head * blocks + block) * H3D_SOL_HEAD_DIM],
                    scale);
            }
            uint32_t total = 0;
            for (uint32_t block = 0; block < blocks; ++block) {
                /* `:387-410`, exempt mode: a prefix query row is dense, a
                 * prefix column is always selected, and the video columns
                 * keep the k_vid largest scores. */
                bool exact = q_block < prefix_blocks || block < prefix_blocks;
                if (!exact) {
                    uint32_t rank = 0;
                    for (uint32_t other = prefix_blocks; other < blocks; ++other) {
                        if (score[other] > score[block] ||
                            (score[other] == score[block] && other < block)) {
                            rank += 1;
                        }
                    }
                    exact = rank < keep;
                }
                if (exact) {
                    out.mask[((size_t)q_block * heads + head) * data.words +
                             block / H3D_SOL_GROUP] |= 1u << (block % H3D_SOL_GROUP);
                    total += 1;
                }
            }
            out.count[(size_t)q_block * heads + head] = total;
        }
    }
}

static bool sol_vsa_run(h3d_context *gpu, const SolData &data, float scale,
                        double sparsity, uint32_t prefix_blocks, SolActual &out,
                        char *error, size_t size) {
    SolPlan plan;
    if (!sol_alloc(gpu, data, plan, error, size)) return false;
    const size_t elements = (size_t)data.sequence * data.heads * H3D_SOL_HEAD_DIM;
    const size_t summary = (size_t)data.heads * data.blocks * H3D_SOL_HEAD_DIM;
    auto *query = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    auto *key = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    auto *value = h3d_alloc(gpu, H3D_BF16, elements, error, size);
    out.kc.assign(summary, 0.0f);
    out.vc.assign(summary, 0.0f);
    out.query_bar.assign(summary, 0.0f);
    out.mask.assign((size_t)data.blocks * data.heads * data.words, 0u);
    out.count.assign((size_t)data.blocks * data.heads, 0u);
    out.threshold.assign((size_t)data.blocks * data.heads, 0.0f);
    bool ok = query && key && value &&
              h3d_upload(gpu, query, data.query.data(), error, size) &&
              h3d_upload(gpu, key, data.key.data(), error, size) &&
              h3d_upload(gpu, value, data.value.data(), error, size) &&
              h3d_sol_vsa(gpu, &plan.plan, query, key, value, data.sequence,
                          data.heads, scale, sparsity, prefix_blocks, error,
                          size) &&
              h3d_sync(gpu, error, size);
    std::vector<uint16_t> key_out(summary), value_out(summary);
    ok = ok && h3d_read(gpu, key_out.data(), plan.kc, error, size) &&
         h3d_read(gpu, value_out.data(), plan.vc, error, size);
    if (ok) {
        for (size_t at = 0; at < summary; ++at) {
            out.kc[at] = sol_float(key_out[at]);
            out.vc[at] = sol_float(value_out[at]);
        }
    }
    ok = ok && h3d_read(gpu, out.query_bar.data(), plan.query_bar, error, size) &&
         h3d_read(gpu, out.mask.data(), plan.mask, error, size) &&
         h3d_read(gpu, out.count.data(), plan.count, error, size);
    if (!ok) fprintf(stderr, "VSA selector: %s\n", error);
    if (query) h3d_free(query);
    if (key) h3d_free(key);
    if (value) h3d_free(value);
    sol_free(plan);
    return ok;
}

static void sol_vsa_compare(const char *label, const SolData &data,
                            const SolExpect &expect, const SolActual &actual,
                            uint32_t prefix_blocks, double sparsity) {
    const bool mask = expect.mask == actual.mask;
    check(mask, "the VSA selector writes the reference's mask");
    check(expect.count == actual.count, "the VSA selector counts its exact blocks");
    uint32_t exact = 0;
    for (uint32_t word : actual.mask) exact += (uint32_t)__builtin_popcount(word);
    const double share = (double)exact /
                         ((double)data.blocks * data.heads * data.blocks);
    printf("  [%s s=%.2f prefix=%u] keep %u of %u video blocks, realized exact "
           "share %.4f\n",
           label, sparsity, prefix_blocks,
           sol_vsa_keep(sparsity, data.blocks - prefix_blocks),
           data.blocks - prefix_blocks, share);
    if (!mask) {
        for (size_t at = 0; at < actual.mask.size(); ++at) {
            if (expect.mask[at] != actual.mask[at]) {
                printf("    first mask word %zu: expected %08x, actual %08x\n", at,
                       expect.mask[at], actual.mask[at]);
                break;
            }
        }
    }
}

static void sol_vsa_matches_reference(h3d_context *gpu) {
    char error[512] = {};
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    const double levels[] = {0.9, 0.75, 0.5, 0.25, 0.0};
    SolData data;
    sol_build(data, 1024, 3, 20261028u);
    for (double sparsity : levels) {
        const uint32_t prefix = 2;   /* 128 rows of text and audio */
        SolActual actual;
        if (!sol_vsa_run(gpu, data, scale, sparsity, prefix, actual, error,
                         sizeof(error))) {
            check(false, "the VSA selector runs");
            return;
        }
        SolExpect expect;
        sol_vsa_expect(data, actual.kc, actual.query_bar, scale, sparsity, prefix,
                       expect);
        sol_vsa_compare("random", data, expect, actual, prefix, sparsity);
    }
}

/* The reference's own setting on the job's shape: sparsity 0.9 of the video
 * tiles, prefix exempt. */
static void sol_vsa_reference_setting(h3d_context *gpu) {
    char error[512] = {};
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    SolData data;
    sol_build(data, 18432, 2, 913u);
    const uint32_t prefix = 8;     /* 512 rows, text and audio */
    SolActual actual;
    if (!sol_vsa_run(gpu, data, scale, 0.9, prefix, actual, error,
                     sizeof(error))) {
        check(false, "the VSA selector runs at the job's shape");
        return;
    }
    SolExpect expect;
    sol_vsa_expect(data, actual.kc, actual.query_bar, scale, 0.9, prefix, expect);
    sol_vsa_compare("job shape", data, expect, actual, prefix, 0.9);
    /* 288 blocks, 8 of them prefix: `ceil(0.1 * 280)` = 28 video blocks kept,
     * plus the 8 prefix columns, on every non-prefix row. */
    const uint32_t keep = sol_vsa_keep(0.9, data.blocks - prefix);
    check(keep == 28, "the keep count follows the reference's own formula");
    bool structural = true;
    for (uint32_t q_block = 0; q_block < data.blocks; ++q_block) {
        for (uint32_t head = 0; head < data.heads; ++head) {
            const uint32_t want = q_block < prefix ? data.blocks : prefix + keep;
            structural = structural &&
                         actual.count[(size_t)q_block * data.heads + head] == want;
        }
    }
    check(structural, "a prefix row is dense and a video row keeps prefix+keep");
}

/* A keep of exactly one: `ceil((1 - 0.9) * 8)` is 1, the clamp's lower bound. */
static void sol_vsa_keeps_one(h3d_context *gpu) {
    char error[512] = {};
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    SolData data;
    sol_build(data, 640, 2, 77u);          /* 10 blocks, 8 of them video */
    const uint32_t prefix = 2;
    SolActual actual;
    if (!sol_vsa_run(gpu, data, scale, 0.9, prefix, actual, error,
                     sizeof(error))) {
        check(false, "the VSA selector runs with a keep of one");
        return;
    }
    SolExpect expect;
    sol_vsa_expect(data, actual.kc, actual.query_bar, scale, 0.9, prefix, expect);
    sol_vsa_compare("keep one", data, expect, actual, prefix, 0.9);
    bool one = true;
    for (uint32_t q_block = prefix; q_block < data.blocks; ++q_block) {
        for (uint32_t head = 0; head < data.heads; ++head) {
            one = one && actual.count[(size_t)q_block * data.heads + head] ==
                             prefix + 1;
        }
    }
    check(one, "a keep of one selects one video block per row");
}

/* Every score equal, which is where the reference leaves the order to
 * `torch.topk` and this port fixed it: ties go by ascending column index. */
static void sol_vsa_ties(h3d_context *gpu) {
    char error[512] = {};
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    SolData data;
    sol_build(data, 1024, 2, 5150u);
    std::fill(data.key.begin(), data.key.end(), sol_bf16(0.0f));
    std::fill(data.query.begin(), data.query.end(), sol_bf16(1.0f));
    const uint32_t prefix = 3;
    SolActual actual;
    if (!sol_vsa_run(gpu, data, scale, 0.75, prefix, actual, error,
                     sizeof(error))) {
        check(false, "the VSA selector survives an all-tie row");
        return;
    }
    SolExpect expect;
    sol_vsa_expect(data, actual.kc, actual.query_bar, scale, 0.75, prefix, expect);
    sol_vsa_compare("all ties", data, expect, actual, prefix, 0.75);
    const uint32_t keep = sol_vsa_keep(0.75, data.blocks - prefix);
    bool by_index = true;
    for (uint32_t q_block = prefix; q_block < data.blocks; ++q_block) {
        for (uint32_t head = 0; head < data.heads; ++head) {
            const uint32_t word0 =
                actual.mask[((size_t)q_block * data.heads + head) * data.words];
            /* The first `prefix + keep` blocks, in index order, and nothing
             * above them: an idempotent tie rule cannot skip a low index. */
            const uint32_t want = (1u << (prefix + keep)) - 1u;
            by_index = by_index && (word0 & 0xffffffffu) == want;
        }
    }
    check(by_index, "ties keep the lowest column indices");
}

/* The count is monotone in sparsity: lowering it can only add exact blocks. */
static void sol_vsa_monotone(h3d_context *gpu) {
    char error[512] = {};
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    SolData data;
    sol_build(data, 2048, 2, 31337u);
    const uint32_t prefix = 1;
    uint32_t previous = 0;
    bool monotone = true;
    for (int step = 10; step >= 0; --step) {
        const double sparsity = (double)step / 10.0;
        SolActual actual;
        if (!sol_vsa_run(gpu, data, scale, sparsity, prefix, actual, error,
                         sizeof(error))) {
            check(false, "the VSA selector runs across the sparsity sweep");
            return;
        }
        uint32_t total = 0;
        for (size_t at = 0; at < actual.count.size(); ++at) total += actual.count[at];
        monotone = monotone && total >= previous;
        previous = total;
    }
    check(monotone, "a lower sparsity never drops an exact block");
}

/* The API boundary: the same refusals the search step needs, plus a prefix
 * that covers the whole sequence, where the reference's own rule has no
 * video column to select and its `topk` would be asked for more than exist. */
static void sol_vsa_rejects_bad_args(h3d_context *gpu) {
    char error[512] = {};
    SolData data;
    sol_build(data, 256, 1, 4242u);       /* 4 blocks */
    SolPlan plan;
    if (!sol_alloc(gpu, data, plan, error, sizeof(error))) {
        check(false, "VSA's rejection cases allocate a plan");
        return;
    }
    const size_t elements = (size_t)data.sequence * data.heads * H3D_SOL_HEAD_DIM;
    auto *query = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *key = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *value = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *wide = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    uint32_t refused = 0;
    refused += !h3d_sol_vsa(gpu, &plan.plan, query, key, value, 0, data.heads,
                            scale, 0.9, 0, error, sizeof(error));
    refused += !h3d_sol_vsa(gpu, &plan.plan, query, key, value, data.sequence, 0,
                            scale, 0.9, 0, error, sizeof(error));
    refused += !h3d_sol_vsa(gpu, &plan.plan, query, key, wide, data.sequence,
                            data.heads, scale, 0.9, 0, error, sizeof(error));
    refused += !h3d_sol_vsa(gpu, &plan.plan, query, key, value, UINT32_MAX,
                            data.heads, scale, 0.9, 0, error, sizeof(error));
    refused += !h3d_sol_vsa(gpu, &plan.plan, query, key, value, data.sequence,
                            data.heads, scale, 0.9, data.blocks, error,
                            sizeof(error));
    refused += !h3d_sol_vsa(gpu, &plan.plan, query, key, value, data.sequence,
                            data.heads, scale, 0.9, data.blocks + 1, error,
                            sizeof(error));
    h3d_sol_plan absent = plan.plan;
    absent.query_bar = nullptr;
    refused += !h3d_sol_vsa(gpu, &absent, query, key, value, data.sequence,
                            data.heads, scale, 0.9, 0, error, sizeof(error));
    check(refused == 7, "the VSA selector refuses a bad shape, prefix or plan");
    h3d_free(query);
    h3d_free(key);
    h3d_free(value);
    h3d_free(wide);
    sol_free(plan);
}

int main() {
    char error[512] = {};
    auto *dit = h3d_create(error, sizeof(error));
    auto *text = h3t_create(error, sizeof(error));
    auto *vision = h3v_create(error, sizeof(error));
    if (!dit || !text || !vision) {
        fprintf(stderr, "%s\n", error);
        return 2;
    }
    read_waits(dit);
    allocation_sizes(dit, text, vision);
    gate_dtype(dit);
    vision_qkv_rows(vision, 65536, 4);
    vision_qkv_rows(vision, 3, 129);
    short_vision_sdpa_axes(vision);
    text_embedding_rows(text, 65536, 3);
    text_embedding_rows(text, 3, 257);
    short_text_gqa_axes(text);
    row_launches(dit, 65536, 1);
    row_launches(dit, 3, 257);
    qkv_row_launch(dit, 65536, 4);
    qkv_row_launch(dit, 3, 129);
    pad_plane_launch(dit, 254, 256, 1);
    pad_plane_launch(dit, 3, 2, 129);
    short_attention_axes(dit);
    short_attention_f32_axes(dit);
    video_vae_attention_order(dit);
    video_vae_attention_mma(dit);
    video_vae_linear_options(dit);
    sol_search_matches_reference(dit);
    sol_search_sink(dit);
    sol_search_empty_sink(dit);
    sol_search_rejects_bad_args(dit);
    sol_search_multiword(dit);
    sol_search_equal_threshold(dit);
    sol_search_constant(dit);
    sol_search_spike(dit);
    sol_search_tau_monotone(dit);
    sol_forward_matches_reference(dit);
    sol_forward_all_exact(dit);
    sol_forward_multiword(dit);
    sol_forward_mask_is_literal(dit);
    sol_forward_rejects_bad_args(dit);
    sol_vsa_matches_reference(dit);
    sol_vsa_reference_setting(dit);
    sol_vsa_keeps_one(dit);
    sol_vsa_ties(dit);
    sol_vsa_monotone(dit);
    sol_vsa_rejects_bad_args(dit);
    h3v_destroy(vision);
    h3t_destroy(text);
    h3d_destroy(dit);
    return failures ? 1 : 0;
}
