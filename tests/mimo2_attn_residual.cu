// Exercise the production wrapper and the original scale/add rounding.
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

// Original production arithmetic from ds4_cuda.cu.
__global__ static void dots3_scale_kernel(float *x, float scale, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { x[i] *= scale; }
}

__global__ static void exaone_add_kernel(float *x, const float *y, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { x[i] += y[i]; }
}

namespace {
constexpr unsigned WIDTH = 4096, THREADS = 256;
constexpr float SCALE = 0.707f;
constexpr const char *SWITCH = "DS4_MIMO2_ATTN_RESIDUAL";

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

void refusals(ds4_gpu_tensor cur, ds4_gpu_tensor attn) {
    constexpr unsigned ROWS = 32;
    setenv(SWITCH, "0", 1);
    expect("disabled", ds4_gpu_mimo2_attn_add(&cur, &attn, ROWS), -1);
    unsetenv(SWITCH);
    for (unsigned rows : {0u, 1u, 31u, 8193u}) {
        expect("unsupported rows", ds4_gpu_mimo2_attn_add(&cur, &attn, rows), -1);
    }
    expect("null cur", ds4_gpu_mimo2_attn_add(nullptr, &attn, ROWS), 0);
    expect("null attn", ds4_gpu_mimo2_attn_add(&cur, nullptr, ROWS), 0);
    for (unsigned operand = 0; operand < 2; operand++) {
        ds4_gpu_tensor args[] = {cur, attn};
        args[operand].ptr = nullptr;
        expect("null storage", ds4_gpu_mimo2_attn_add(&args[0], &args[1], ROWS), 0);
        args[operand] = operand == 0 ? cur : attn;
        --args[operand].bytes;
        expect("undersized", ds4_gpu_mimo2_attn_add(&args[0], &args[1], ROWS), 0);
        args[operand] = operand == 0 ? cur : attn;
        args[operand].ptr = (char *)args[operand].ptr + 1;
        expect("unaligned", ds4_gpu_mimo2_attn_add(&args[0], &args[1], ROWS), -1);
    }
}

void exact(const char *label, const std::vector<float> &expected, const float *device) {
    std::vector<float> actual(expected.size());
    const size_t bytes = expected.size() * sizeof(float);
    check(cudaMemcpy(actual.data(), device, bytes, cudaMemcpyDeviceToHost));
    if (std::memcmp(expected.data(), actual.data(), bytes) == 0) { return; }
    std::fprintf(stderr, "%s: byte parity failed\n", label);
    std::exit(3);
}

void compare(unsigned rows) {
    const size_t count = (size_t)rows * WIDTH, bytes = count * sizeof(float);
    std::mt19937 rng(rows);
    std::uniform_real_distribution<float> value(-8.0f, 8.0f);
    std::vector<float> attn(count), initial(count), expected(count);
    for (float &x : attn) { x = value(rng); }
    for (float &x : initial) { x = value(rng); }
    attn[0] = -0.0f; initial[0] = 0.0f;
    attn[1] = 0x1p-126f; initial[1] = -0x1p-127f;
    float *source, *scaled, *reference, *actual;
    check(cudaMalloc(&source, bytes));
    check(cudaMalloc(&scaled, bytes));
    check(cudaMalloc(&reference, bytes));
    check(cudaMalloc(&actual, bytes));
    check(cudaMemcpy(source, attn.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(scaled, attn.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(reference, initial.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(actual, initial.data(), bytes, cudaMemcpyHostToDevice));
    ds4_gpu_tensor cur{actual, bytes, 0, 0}, projection{source, bytes, 0, 0};
    if (rows == 32) { refusals(cur, projection); }
    exact("refused residual unchanged", initial, actual);
    exact("refused projection unchanged", attn, source);

    const unsigned blocks = (count + THREADS - 1) / THREADS;
    dots3_scale_kernel<<<blocks, THREADS>>>(scaled, SCALE, count);
    exaone_add_kernel<<<blocks, THREADS>>>(reference, scaled, count);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    if (rows == 33) { setenv(SWITCH, "1", 1); } else { unsetenv(SWITCH); }
    if (rows < 32) {
        expect("narrow fallback", ds4_gpu_mimo2_attn_add(&cur, &projection, rows), -1);
        exact("narrow residual unchanged", initial, actual);
        // Numerical parity also covers narrow shapes, without enabling dispatch.
        mimo2_attn_residual<<<blocks, THREADS>>>(actual, source, count);
        check(cudaGetLastError());
    } else {
        expect("enabled", ds4_gpu_mimo2_attn_add(&cur, &projection, rows), 1);
    }
    check(cudaMemcpy(expected.data(), reference, bytes, cudaMemcpyDeviceToHost));
    exact("scale/add result", expected, actual);
    exact("fused projection unchanged", attn, source);
    for (float *p : {source, scaled, reference, actual}) { check(cudaFree(p)); }
    std::printf("rows=%u: attention residual byte-exact PASS\n", rows);
}
}

int main() {
    for (unsigned rows : {1u, 31u, 32u, 33u, 4096u}) { compare(rows); }
    unsetenv(SWITCH);
    return 0;
}
