// Compare every ID and weight bit, including legacy malformed-input behavior.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "../cuda/iquest_router.cuh"

#define CUDA_OK(call) do { \
    const cudaError_t status = (call); \
    if (status != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); return 2; } \
} while (0)

static float from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

int main() {
    constexpr unsigned prefix_width = 5, prefix_cases = 1u << (2 * prefix_width);
    constexpr unsigned random_cases = 512, special_cases = 8;
    constexpr unsigned rows = IQ_EXPERTS + prefix_cases + random_cases + special_cases;
    constexpr uint32_t seed = 0x912fab73u;
    const float nan = from_bits(0x7fc01234u);
    const float alphabet[] = {-INFINITY, nan, -0.0f, 1.0f};
    const float special[] = {0.0f, -0.0f, INFINITY, -INFINITY, nan,
                            from_bits(1), from_bits(0x80000001u), from_bits(0x007fffffu)};
    std::vector<float> input((size_t)rows * IQ_EXPERTS);
    uint32_t random = seed;
    for (unsigned row = 0; row < rows; row++) {
        for (unsigned expert = 0; expert < IQ_EXPERTS; expert++) {
            float value;
            if (row < IQ_EXPERTS) {
                value = expert == row ? nan : (float)((expert * 73) % IQ_EXPERTS);
            } else if (row < IQ_EXPERTS + prefix_cases) {
                const unsigned pattern = row - IQ_EXPERTS;
                value = expert < prefix_width ? alphabet[(pattern >> (2 * expert)) & 3u] : -INFINITY;
            } else if (row < IQ_EXPERTS + prefix_cases + random_cases) {
                random ^= random << 13; random ^= random >> 17; random ^= random << 5;
                value = from_bits(random);
            } else {
                const unsigned kind = row - IQ_EXPERTS - prefix_cases - random_cases;
                value = special[(expert + kind) % special_cases];
            }
            input[(size_t)row * IQ_EXPERTS + expert] = value;
        }
    }
    constexpr size_t count = (size_t)rows * IQ_USED;
    float *scores = nullptr, *weight_a = nullptr, *weight_b = nullptr;
    unsigned *ids_a = nullptr, *ids_b = nullptr;
    CUDA_OK(cudaMalloc(&scores, input.size() * sizeof(float)));
    CUDA_OK(cudaMemcpy(scores, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&weight_a, count * sizeof(float)));
    CUDA_OK(cudaMalloc(&weight_b, count * sizeof(float)));
    CUDA_OK(cudaMalloc(&ids_a, count * sizeof(unsigned)));
    CUDA_OK(cudaMalloc(&ids_b, count * sizeof(unsigned)));
    CUDA_OK(cudaMemset(weight_a, 0x7f, count * sizeof(float)));
    CUDA_OK(cudaMemset(weight_b, 0xff, count * sizeof(float)));
    CUDA_OK(cudaMemset(ids_a, 0x7f, count * sizeof(unsigned)));
    CUDA_OK(cudaMemset(ids_b, 0xff, count * sizeof(unsigned)));
    iquest_router_kernel<<<rows, 1>>>(ids_a, weight_a, scores);
    CUDA_OK(cudaGetLastError());
    iq_router::select<<<rows, iq_router::WARP>>>(ids_b, weight_b, scores);
    CUDA_OK(cudaGetLastError());
    std::vector<unsigned> expected_ids(count), actual_ids(count);
    std::vector<float> expected_weights(count), actual_weights(count);
    CUDA_OK(cudaMemcpy(expected_ids.data(), ids_a, count * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(actual_ids.data(), ids_b, count * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(expected_weights.data(), weight_a, count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(actual_weights.data(), weight_b, count * sizeof(float), cudaMemcpyDeviceToHost));
    size_t id_mismatch = 0, weight_mismatch = 0, invalid = 0;
    for (size_t i = 0; i < count; i++) {
        id_mismatch += expected_ids[i] != actual_ids[i];
        weight_mismatch += std::memcmp(&expected_weights[i], &actual_weights[i], sizeof(float)) != 0;
        invalid += actual_ids[i] >= IQ_EXPERTS;
        for (size_t j = i / IQ_USED * IQ_USED; j < i; j++) { invalid += actual_ids[j] == actual_ids[i]; }
    }
    const bool passed = !id_mismatch && !weight_mismatch && !invalid;
    std::printf("{\"passed\":%s,\"rows\":%u,\"readback_per_output\":%zu,"
        "\"id_mismatch\":%zu,\"weight_bit_mismatch\":%zu,\"invalid_or_duplicate_ids\":%zu,"
        "\"oracle\":\"retained_cuda_serial_full_readback\",\"seed\":%u}\n",
        passed ? "true" : "false", rows, count, id_mismatch, weight_mismatch, invalid, seed);
    CUDA_OK(cudaFree(scores)); CUDA_OK(cudaFree(weight_a)); CUDA_OK(cudaFree(weight_b));
    CUDA_OK(cudaFree(ids_a)); CUDA_OK(cudaFree(ids_b));
    return passed ? 0 : 1;
}
