#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "../cuda/step37_primitives.cuh"
#include "../cuda/mimo2_primitives.cuh"

namespace {
constexpr unsigned WIDTH = 4096, USED = 8, THREADS = 256;

void check(cudaError_t rc) {
    if (rc == cudaSuccess) { return; }
    std::fprintf(stderr, "%s\n", cudaGetErrorString(rc));
    std::exit(1);
}

__global__ void add_residual(float *cur, const float *sum, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { cur[i] += sum[i]; }
}

// Compare against the existing ordered sum and its separate residual pass.
void compare(unsigned rows) {
    const size_t count = (size_t)rows * WIDTH;
    std::mt19937 rng(rows);
    std::uniform_real_distribution<float> value(-8.0f, 8.0f);
    std::uniform_real_distribution<float> weight(0.0f, 1.0f);
    std::vector<float> down(count * USED), weights(rows * USED), initial(count);
    for (auto &x : down) { x = value(rng); }
    for (auto &x : weights) { x = weight(rng); }
    for (auto &x : initial) { x = value(rng); }

    float *d, *w, *tmp, *a, *b;
    check(cudaMalloc(&d, down.size() * sizeof(float)));
    check(cudaMalloc(&w, weights.size() * sizeof(float)));
    check(cudaMalloc(&tmp, count * sizeof(float)));
    check(cudaMalloc(&a, count * sizeof(float)));
    check(cudaMalloc(&b, count * sizeof(float)));
    check(cudaMemcpy(d, down.data(), down.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(w, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(a, initial.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(b, initial.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    step37_expert_sum<<<(count + THREADS - 1) / THREADS, THREADS>>>(
        tmp, d, w, WIDTH, count);
    add_residual<<<(count + THREADS - 1) / THREADS, THREADS>>>(a, tmp, count);
    mimo2_sum_residual<<<(count + THREADS - 1) / THREADS, THREADS>>>(
        b, d, w, count);
    check(cudaGetLastError());

    std::vector<float> expected(count), actual(count);
    check(cudaMemcpy(expected.data(), a, count * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual.data(), b, count * sizeof(float), cudaMemcpyDeviceToHost));
    if (std::memcmp(expected.data(), actual.data(), count * sizeof(float)) != 0) {
        std::fprintf(stderr, "rows=%u: sum/residual parity failed\n", rows);
        std::exit(2);
    }
    for (float *p : {d, w, tmp, a, b}) { check(cudaFree(p)); }
    std::printf("rows=%u: byte-exact PASS\n", rows);
}
}

int main() {
    // Include decode-sized, dispatch-boundary, irregular and production rows.
    for (unsigned rows : {1u, 31u, 32u, 33u, 4096u}) { compare(rows); }
    return 0;
}
