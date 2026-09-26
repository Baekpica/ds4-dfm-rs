// Exercise the production wrapper, including refusals before any GPU work.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static bool m2_hmma_available = false;
static unsigned stream_calls = 0;
static int cuda_ok(cudaError_t err, const char *) { return err == cudaSuccess; }
static cudaStream_t ds4_current_stream(void) { ++stream_calls; return 0; }
static int ds4_capture_active(void) { return 0; }
static const char *cuda_model_range_ptr(const void *, uint64_t, uint64_t, const char *) { return nullptr; }
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; int owner; int memc; };
#include "../ds4_mimo2_gpu.cuh"
#include "../cuda/step37_primitives.cuh"

namespace {
constexpr unsigned WIDTH = 4096, USED = 8, THREADS = 256;
constexpr const char *SWITCH = "DS4_MIMO2_SUM_RESIDUAL";

void check(cudaError_t rc) {
    if (rc == cudaSuccess) { return; }
    std::fprintf(stderr, "%s\n", cudaGetErrorString(rc));
    std::exit(1);
}

void expect(const char *label, int got, int want) {
    const unsigned launches = want == 1 ? 1 : 0;
    if (got != want || stream_calls != launches) {
        std::fprintf(stderr, "%s: rc=%d want=%d stream_calls=%u want=%u\n",
                     label, got, want, stream_calls, launches);
        std::exit(2);
    }
    stream_calls = 0;
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
}

void refusals(ds4_gpu_tensor cur, ds4_gpu_tensor down, ds4_gpu_tensor weights) {
    constexpr unsigned ROWS = 32;
    unsetenv(SWITCH);
    expect("default enabled", ds4_gpu_mimo2_sum_add(nullptr, &down, &weights, ROWS), 0);
    for (const char *off : {"0", "", "10"}) {
        setenv(SWITCH, off, 1);
        expect("disabled", ds4_gpu_mimo2_sum_add(&cur, &down, &weights, ROWS), -1);
    }
    setenv(SWITCH, "1", 1);
    for (unsigned rows : {0u, 1u, 31u, 8193u}) {
        expect("unsupported rows", ds4_gpu_mimo2_sum_add(&cur, &down, &weights, rows), -1);
    }
    expect("null cur", ds4_gpu_mimo2_sum_add(nullptr, &down, &weights, ROWS), 0);
    expect("null down", ds4_gpu_mimo2_sum_add(&cur, nullptr, &weights, ROWS), 0);
    expect("null weights", ds4_gpu_mimo2_sum_add(&cur, &down, nullptr, ROWS), 0);

    // Each operand must have a live allocation and its full logical size.
    for (unsigned operand = 0; operand < 3; operand++) {
        ds4_gpu_tensor args[] = {cur, down, weights};
        args[operand].ptr = nullptr;
        expect("null storage", ds4_gpu_mimo2_sum_add(&args[0], &args[1], &args[2], ROWS), 0);
        args[operand] = operand == 0 ? cur : operand == 1 ? down : weights;
        --args[operand].bytes;
        expect("undersized", ds4_gpu_mimo2_sum_add(&args[0], &args[1], &args[2], ROWS), 0);
    }
    ds4_gpu_tensor unaligned = cur;
    unaligned.ptr = (char *)cur.ptr + 1;
    expect("unaligned cur", ds4_gpu_mimo2_sum_add(&unaligned, &down, &weights, ROWS), -1);
    unaligned = down;
    unaligned.ptr = (char *)down.ptr + 1;
    expect("unaligned down", ds4_gpu_mimo2_sum_add(&cur, &unaligned, &weights, ROWS), -1);
    unaligned = weights;
    unaligned.ptr = (char *)weights.ptr + 1;
    expect("unaligned weights", ds4_gpu_mimo2_sum_add(&cur, &down, &unaligned, ROWS), -1);
}

__global__ void add_residual(float *cur, const float *sum, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { cur[i] += sum[i]; }
}

void compare(unsigned rows) {
    const size_t count = (size_t)rows * WIDTH, bytes = count * sizeof(float);
    std::mt19937 rng(rows);
    std::uniform_real_distribution<float> value(-8.0f, 8.0f), weight(0.0f, 1.0f);
    std::vector<float> down(count * USED), weights(rows * USED), initial(count);
    for (float &x : down) { x = value(rng); }
    for (float &x : weights) { x = weight(rng); }
    for (float &x : initial) { x = value(rng); }
    float *d, *w, *tmp, *a, *b;
    check(cudaMalloc(&d, down.size() * sizeof(float)));
    check(cudaMalloc(&w, weights.size() * sizeof(float)));
    check(cudaMalloc(&tmp, bytes));
    check(cudaMalloc(&a, bytes));
    check(cudaMalloc(&b, bytes));
    check(cudaMemcpy(d, down.data(), down.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(w, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(a, initial.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(b, initial.data(), bytes, cudaMemcpyHostToDevice));
    ds4_gpu_tensor cur{b, bytes, 0, 0};
    ds4_gpu_tensor routed{d, down.size() * sizeof(float), 0, 0};
    ds4_gpu_tensor route_weights{w, weights.size() * sizeof(float), 0, 0};
    if (rows == 32) { refusals(cur, routed, route_weights); }

    std::vector<float> expected(count), actual(count);
    check(cudaMemcpy(actual.data(), b, bytes, cudaMemcpyDeviceToHost));
    if (std::memcmp(initial.data(), actual.data(), bytes) != 0) {
        std::fprintf(stderr, "refused dispatch modified the residual\n");
        std::exit(3);
    }
    step37_expert_sum<<<(count + THREADS - 1) / THREADS, THREADS>>>(
        tmp, d, w, WIDTH, count);
    add_residual<<<(count + THREADS - 1) / THREADS, THREADS>>>(a, tmp, count);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    if (rows == 33) { unsetenv(SWITCH); } else { setenv(SWITCH, "1", 1); }
    expect("enabled", ds4_gpu_mimo2_sum_add(&cur, &routed, &route_weights, rows), 1);
    check(cudaMemcpy(expected.data(), a, bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual.data(), b, bytes, cudaMemcpyDeviceToHost));
    if (std::memcmp(expected.data(), actual.data(), bytes) != 0) {
        std::fprintf(stderr, "rows=%u: wrapper parity failed\n", rows);
        std::exit(4);
    }
    for (float *p : {d, w, tmp, a, b}) { check(cudaFree(p)); }
    std::printf("rows=%u: wrapper byte-exact PASS\n", rows);
}
}

int main() {
    compare(32);
    compare(33);
    unsetenv(SWITCH);
    return 0;
}
