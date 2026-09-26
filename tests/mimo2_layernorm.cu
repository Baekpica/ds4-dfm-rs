// Actual LayerNorm scratch reuse at audio and final vision normalization shapes.
// nvcc -O3 -g -lineinfo --use_fast_math -std=c++17 -arch=sm_121a \
//   tests/mimo2_layernorm.cu -o /tmp/mimo2-layernorm
// compute-sanitizer --tool racecheck --error-exitcode 99 /tmp/mimo2-layernorm
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../cuda/mimo2_media.cuh"

namespace {
constexpr int AUDIO_ROWS = 557, AUDIO_DIM = 1024, VISION_ROWS = 31, VISION_DIM = 1280;
constexpr int THREADS = 256, REPEATS = 3;
enum class Kind { Audio, Vision };
constexpr size_t GUARD = 32;
constexpr uint32_t SENTINEL = 0x7fc0abcd;
constexpr float AUDIO_EPS = 1e-5f, VISION_EPS = 1e-6f, TOLERANCE = 3e-5f;

void check(cudaError_t rc) {
    if (rc == cudaSuccess) { return; }
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(rc));
    std::exit(1);
}

uint32_t mix(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

float input(uint32_t index) {
    const uint32_t hash = mix(index ^ 0x9fb14c83u);
    const uint32_t bits = (hash & 0x807fffffu) | ((125u + ((hash >> 24) % 5u)) << 23);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Match the production block reduction tree, with explicit FP32 rounding.
float tree(float *values) {
    for (int span = THREADS / 2; span > 0; span /= 2) {
        for (int tid = 0; tid < span; tid++) { values[tid] += values[tid + span]; }
    }
    return values[0];
}

void reference(std::vector<float> &out, const std::vector<float> &x,
               const std::vector<float> &weight, const std::vector<float> &bias,
               int rows, int dim, float eps) {
    for (int row = 0; row < rows; row++) {
        const float *src = x.data() + row * dim;
        float partial[THREADS] = {};
        for (int tid = 0; tid < THREADS; tid++) {
            for (int i = tid; i < dim; i += THREADS) { partial[tid] += src[i]; }
        }
        const float mean = tree(partial) / dim;
        std::fill(partial, partial + THREADS, 0.f);
        for (int tid = 0; tid < THREADS; tid++) {
            for (int i = tid; i < dim; i += THREADS) {
                const float delta = src[i] - mean;
                partial[tid] = std::fma(delta, delta, partial[tid]);
            }
        }
        const float inv = 1.f / std::sqrt(tree(partial) / dim + eps);
        for (int i = 0; i < dim; i++) {
            const float value = (src[i] - mean) * inv;
            out[row * dim + i] = bias.empty() ? value * weight[i] : std::fma(value, weight[i], bias[i]);
        }
    }
}

int run(Kind kind) {
    const int rows = kind == Kind::Audio ? AUDIO_ROWS : VISION_ROWS;
    const int dim = kind == Kind::Audio ? AUDIO_DIM : VISION_DIM;
    const float eps = kind == Kind::Audio ? AUDIO_EPS : VISION_EPS;
    const size_t count = (size_t)rows * dim;
    std::vector<float> x(count), weight(dim), bias(dim), expected(count), got(count), first(count);
    for (size_t i = 0; i < count; i++) { x[i] = input((uint32_t)i); }
    for (int i = 0; i < dim; i++) {
        weight[i] = 1.f + input((uint32_t)i + 0x31415926u) * 0.125f;
        bias[i] = input((uint32_t)i + 0x27182818u) * 0.03125f;
    }
    if (kind == Kind::Vision) { bias.clear(); }
    reference(expected, x, weight, bias, rows, dim, eps);

    float *dx, *dw, *db = nullptr, *base;
    check(cudaMalloc(&dx, count * sizeof(float)));
    check(cudaMalloc(&dw, dim * sizeof(float)));
    if (!bias.empty()) { check(cudaMalloc(&db, dim * sizeof(float))); }
    check(cudaMalloc(&base, (count + 2 * GUARD) * sizeof(float)));
    std::vector<uint32_t> guarded(count + 2 * GUARD, SENTINEL);
    check(cudaMemcpy(dx, x.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dw, weight.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
    if (!bias.empty()) { check(cudaMemcpy(db, bias.data(), dim * sizeof(float), cudaMemcpyHostToDevice)); }
    check(cudaMemcpy(base, guarded.data(), guarded.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    float max_error = 0.f;
    bool exact = true, numeric = true;
    for (int repeat = 0; repeat < REPEATS; repeat++) {
        mimo2_layernorm<<<rows, THREADS, THREADS * sizeof(float)>>>(base + GUARD, dx, dw, db, dim, eps);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(got.data(), base + GUARD, count * sizeof(float), cudaMemcpyDeviceToHost));
        if (repeat == 0) { first = got; }
        else { exact = exact && std::memcmp(first.data(), got.data(), count * sizeof(float)) == 0; }
        for (size_t i = 0; i < count; i++) {
            const float error = std::fabs(got[i] - expected[i]);
            max_error = std::max(max_error, error);
            numeric = numeric && std::isfinite(got[i]) && error <= TOLERANCE;
        }
    }
    check(cudaMemcpy(guarded.data(), base, guarded.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    bool guards = true;
    for (size_t i = 0; i < GUARD; i++) {
        guards = guards && guarded[i] == SENTINEL && guarded[count + GUARD + i] == SENTINEL;
    }
    std::vector<float> unchanged(count);
    check(cudaMemcpy(unchanged.data(), dx, count * sizeof(float), cudaMemcpyDeviceToHost));
    const bool input_exact = std::memcmp(unchanged.data(), x.data(), count * sizeof(float)) == 0;
    const bool pass = numeric && exact && guards && input_exact;
    std::printf("case=%s rows=%d dim=%d repeats=%d max_abs=%.9g repeat_exact=%s guards=%s input_exact=%s %s\n",
                kind == Kind::Audio ? "audio" : "vision", rows, dim, REPEATS, max_error, exact ? "true" : "false", guards ? "true" : "false",
                input_exact ? "true" : "false", pass ? "PASS" : "FAIL");
    check(cudaFree(dx)); check(cudaFree(dw)); check(cudaFree(db)); check(cudaFree(base));
    return pass ? 0 : 2;
}
}

int main() {
    const int audio = run(Kind::Audio);
    const int vision = run(Kind::Vision);
    return audio || vision ? 2 : 0;
}
