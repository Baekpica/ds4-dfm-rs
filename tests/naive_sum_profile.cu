/* Original pair and fused kernel share synthetic production geometry.
 * GPU bytes cover exceptional operands; CPU equations cover finite fixtures. */
#include <cuda_runtime.h>
#include "../cuda/naive_primitives.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call) do { if (!(call)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #call); exit(1); \
} } while (0)
#define CUDA(call) do { const cudaError_t rc = (call); if (rc != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(rc)); exit(1); \
} } while (0)

enum { THREADS = 256, REPEATS = 20 };
enum class SumPath { Separate, Fused };
enum class SumFixture { Original, Boundary, Exceptional };

template<class T> static T *device(size_t count) {
    T *out;
    CUDA(cudaMalloc(&out, count * sizeof(T)));
    return out;
}

template<class T> static T *upload(const std::vector<T> &values) {
    T *out = device<T>(values.size());
    CUDA(cudaMemcpy(out, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return out;
}

/* Finite F32-to-BF16 round-to-nearest-even, independent of CUDA conversion. */
static float bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    memcpy(&value, &bits, sizeof(bits));
    return value;
}

static float f32(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

__global__ static void reset_cur(float *cur, const float *input, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { cur[i] = input[i]; }
}

static void fill(std::vector<float> &down, std::vector<float> &weights,
                 std::vector<float> &cur, unsigned rows, SumFixture fixture) {
    for (size_t i = 0; i < down.size(); i++) {
        const int seed = (int)((i * 17 + i / N05_EMBED * 11) % 1023) - 511;
        const float base = bf16(std::ldexp(seed * .0078125f, -(int)(i % 5)));
        down[i] = base + ((int)(i % 3) - 1) * 0x1p-16f;
    }
    for (size_t i = 0; i < cur.size(); i++) {
        const int seed = (int)((i * 13 + i / N05_EMBED * 7) % 511) - 255;
        cur[i] = bf16(seed * .00390625f);
    }
    for (unsigned row = 0; row < rows; row++) {
        float total = 0;
        for (unsigned e = 0; e < N05_USED; e++) {
            const float value = 1.0f + (row * 7 + e * 3) % 19;
            weights[(size_t)row * N05_USED + e] = value;
            total += value;
        }
        for (unsigned e = 0; e < N05_USED; e++) {
            weights[(size_t)row * N05_USED + e] /= total;
        }
    }
    if (fixture == SumFixture::Original) { return; }

    if (fixture == SumFixture::Exceptional) {
        // GPU original-pair bytes are the oracle here. The finite CPU equation
        // does not model CUDA fast-math FTZ or NaN payload handling.
        const uint32_t bits[] = {0x00000000u, 0x80000000u, 0x7f7f0000u, 0xff7f0000u,
            0x00010000u, 0x80010000u, 0x00000001u, 0x80000001u,
            0x00800000u, 0x80800000u, 0x7f800000u, 0xff800000u,
            0x7fc10000u, 0x7fa10000u, 0x7f7fffffu, 0xff7fffffu};
        const size_t cases = sizeof(bits) / sizeof(bits[0]);
        for (size_t i = 0; i < down.size(); i++) {
            down[i] = f32(bits[(i + i / N05_EMBED) % cases]);
        }
        for (size_t i = 0; i < cur.size(); i++) { cur[i] = f32(bits[(i * 3) % cases]); }
        for (unsigned row = 0; row < rows; row++) {
            for (unsigned e = 0; e < N05_USED; e++) {
                weights[(size_t)row * N05_USED + e] = e % 4 ? 1.0f / 6.0f : 0.0f;
            }
        }
        return;
    }

    // Equal positive weights retain normalized routing. Alternating expert
    // signs and halfway down values exercise order and BF16 tie boundaries.
    for (unsigned row = 0; row < rows; row++) {
        for (unsigned e = 0; e < N05_USED; e++) {
            weights[(size_t)row * N05_USED + e] = 1.0f / N05_USED;
            for (unsigned col = 0; col < N05_EMBED; col++) {
                const float base = 1.0f + (col % 16) * .0078125f;
                const float near = base + ((int)(col % 3) - 1) * .00390625f;
                down[((size_t)row * N05_USED + e) * N05_EMBED + col] =
                    std::ldexp(e % 2 ? -near : near, (int)(e % 4) - 2);
            }
        }
        for (unsigned col = 0; col < N05_EMBED; col++) {
            cur[(size_t)row * N05_EMBED + col] =
                col % 2 ? -1.0f : 1.0f + (col % 16) * .0078125f;
        }
    }
}

static void equations(const std::vector<float> &down, const std::vector<float> &weights,
                      const std::vector<float> &cur, std::vector<float> &sum,
                      std::vector<float> &result) {
    for (size_t i = 0; i < cur.size(); i++) {
        const size_t row = i / N05_EMBED, col = i % N05_EMBED;
        float value = 0;
        for (unsigned e = 0; e < N05_USED; e++) {
            const float term = bf16(bf16(down[(row * N05_USED + e) * N05_EMBED + col]) *
                weights[row * N05_USED + e]);
            value = bf16(value + term);
        }
        sum[i] = value;
        result[i] = bf16(cur[i] + bf16(value));
    }
}

static void launch(SumPath path, float *cur, float *sum, const float *down,
                   const float *weights, uint64_t count, unsigned blocks) {
    if (path == SumPath::Fused) {
        naive_sum_add<<<blocks, THREADS>>>(cur, down, weights, count);
        return;
    }
    naive_sum<<<blocks, THREADS>>>(sum, down, weights, count);
    naive_add<<<blocks, THREADS>>>(cur, sum, count);
}

int main(int argc, char **argv) {
    if (argc > 4) { return 2; }
    const unsigned rows = argc >= 2 ? (unsigned)atoi(argv[1]) : N05_PREFILL;
    const auto path = argc >= 3 ? static_cast<SumPath>((unsigned)atoi(argv[2])) : SumPath::Separate;
    const auto fixture = argc >= 4 ? static_cast<SumFixture>((unsigned)atoi(argv[3])) : SumFixture::Original;
    if (!rows || rows > N05_PREFILL_MAX || path > SumPath::Fused ||
        fixture > SumFixture::Exceptional) { return 2; }
    const uint64_t count = (uint64_t)rows * N05_EMBED;
    const unsigned blocks = (unsigned)((count + THREADS - 1) / THREADS);
    const size_t bytes = (size_t)count * sizeof(float);
    std::vector<float> down((size_t)count * N05_USED), weights((size_t)rows * N05_USED);
    std::vector<float> cur((size_t)count), expected_sum((size_t)count), expected((size_t)count);
    fill(down, weights, cur, rows, fixture);
    const unsigned cpu_checked = fixture != SumFixture::Exceptional;
    if (cpu_checked) { equations(down, weights, cur, expected_sum, expected); }

    float *dd = upload(down), *dw = upload(weights), *input = upload(cur);
    float *dc = device<float>((size_t)count), *ds = device<float>((size_t)count);
    float *reference = device<float>((size_t)count);
    // The untimed original pair is a second oracle for both selected paths.
    reset_cur<<<blocks, THREADS>>>(reference, input, count);
    launch(SumPath::Separate, reference, ds, dd, dw, count, blocks);
    CUDA(cudaGetLastError());
    std::vector<float> gpu_sum((size_t)count), gpu_reference((size_t)count);
    CUDA(cudaMemcpy(gpu_sum.data(), ds, bytes, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(gpu_reference.data(), reference, bytes, cudaMemcpyDeviceToHost));
    if (cpu_checked) {
        CHECK(!memcmp(gpu_sum.data(), expected_sum.data(), bytes));
        CHECK(!memcmp(gpu_reference.data(), expected.data(), bytes));
    }

    cudaEvent_t start, end;
    CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
    std::vector<float> samples;
    for (unsigned repeat = 0; repeat <= REPEATS; repeat++) {
        // Reset is ordered before start, so only the selected path is timed.
        reset_cur<<<blocks, THREADS>>>(dc, input, count);
        CUDA(cudaGetLastError());
        CUDA(cudaEventRecord(start));
        launch(path, dc, ds, dd, dw, count, blocks);
        CUDA(cudaGetLastError());
        CUDA(cudaEventRecord(end));
        CUDA(cudaEventSynchronize(end));
        if (!repeat) { continue; }
        float elapsed;
        CUDA(cudaEventElapsedTime(&elapsed, start, end));
        samples.push_back(elapsed);
    }

    std::vector<float> got((size_t)count);
    CUDA(cudaMemcpy(got.data(), dc, bytes, cudaMemcpyDeviceToHost));
    if (cpu_checked) { CHECK(!memcmp(got.data(), expected.data(), bytes)); }
    CHECK(!memcmp(got.data(), gpu_reference.data(), bytes));
    if (cpu_checked) {
        for (float value : got) { CHECK(std::isfinite(value)); }
    }
    if (path == SumPath::Separate) {
        std::vector<float> got_sum((size_t)count);
        CUDA(cudaMemcpy(got_sum.data(), ds, bytes, cudaMemcpyDeviceToHost));
        CHECK(!memcmp(got_sum.data(), gpu_sum.data(), bytes));
        if (cpu_checked) { CHECK(!memcmp(got_sum.data(), expected_sum.data(), bytes)); }
    }
    double total = 0;
    for (float value : samples) { total += value; }
    const auto bounds = std::minmax_element(samples.begin(), samples.end());
    printf("sum_residual rows=%u path=%u fixture=%u experts=%u width=%u repeats=%u "
           "milliseconds=%.6f min=%.6f max=%.6f reset_timed=0 cpu_exact=%u gpu_exact=1\n",
        rows, (unsigned)path, (unsigned)fixture, N05_USED, N05_EMBED, REPEATS,
        total / samples.size(), *bounds.first, *bounds.second, cpu_checked);
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
    CUDA(cudaFree(dd)); CUDA(cudaFree(dw)); CUDA(cudaFree(input));
    CUDA(cudaFree(dc)); CUDA(cudaFree(ds)); CUDA(cudaFree(reference));
    return 0;
}
