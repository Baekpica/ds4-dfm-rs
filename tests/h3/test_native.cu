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

static void text_embedding_rows(h3t_context *gpu, uint32_t rows, uint32_t width) {
    char error[512] = {};
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
    h3v_destroy(vision);
    h3t_destroy(text);
    h3d_destroy(dit);
    return failures ? 1 : 0;
}
