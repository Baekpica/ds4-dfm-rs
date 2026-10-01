// Expose accumulation bias that the final BF16 store can hide. With unit V,
// the unrounded weighted average must stay near one across 8192 keys.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#include "../cuda/iquest_prefill.cuh"

#define CUDA_OK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        return 2; \
    } \
} while (0)

int main() {
    constexpr unsigned rows = IQ_PREFILL, keys = 8192;
    constexpr float tolerance = 32 * std::numeric_limits<float>::epsilon();
    std::vector<float> query((size_t)rows * IQ_HEADS * IQ_HEAD), output(query.size());
    std::vector<unsigned> positions(rows);
    std::vector<iquest_q8> cache((size_t)keys * IQ_Q8_ROW_BLOCKS);
    for (unsigned row = 0; row < rows; row++) {
        positions[row] = keys - rows + row;
        for (unsigned head = 0; head < IQ_HEADS; head++) {
            query[((size_t)row * IQ_HEADS + head) * IQ_HEAD] = 1;
        }
    }
    for (unsigned token = 0; token < keys; token++) {
        for (unsigned block = 0; block < IQ_Q8_ROW_BLOCKS; block++) {
            auto &part = cache[(size_t)token * IQ_Q8_ROW_BLOCKS + block];
            const bool value = block >= IQ_Q8_ROW_BLOCKS / 2;
            part.d = __half_as_ushort(__float2half_rn(value ? 1.0f : 0.125f));
            for (unsigned i = 0; i < IQ_Q8_BLOCK; i++) {
                part.qs[i] = value ? 1 : static_cast<int>((token * 73 + block * 31 + i * 7) % 255) - 127;
            }
            // A127 maximum makes scale1/unit values a realizable Q8 block.
            part.qs[IQ_Q8_BLOCK - 1] = 127;
        }
    }

    float *q = nullptr, *out = nullptr, *lse = nullptr;
    unsigned *pos = nullptr;
    iquest_q8 *kv = nullptr;
    CUDA_OK(cudaMalloc(&q, query.size() * sizeof(float)));
    CUDA_OK(cudaMalloc(&out, query.size() * sizeof(float)));
    CUDA_OK(cudaMalloc(&lse, (size_t)rows * IQ_HEADS * sizeof(float)));
    CUDA_OK(cudaMalloc(&pos, rows * sizeof(unsigned)));
    CUDA_OK(cudaMalloc(&kv, cache.size() * sizeof(iquest_q8)));
    CUDA_OK(cudaMemcpy(q, query.data(), query.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(pos, positions.data(), rows * sizeof(unsigned), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(kv, cache.data(), cache.size() * sizeof(iquest_q8), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(out, 0xff, query.size() * sizeof(float)));

    iq_prefill::prefill<iq_prefill::Output::DiagnosticF32>
        <<<dim3(rows / iq_prefill::TQ, IQ_HEADS), iq_prefill::THREADS>>>
        (out, lse, q, kv, pos, rows, keys, 0);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(output.data(), out, query.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double max_error = 0, mean_error = 0;
    size_t count = 0, nonfinite = 0;
    for (size_t i = 0; i < output.size(); i++) {
        nonfinite += !std::isfinite(output[i]);
        if (i % IQ_Q8_BLOCK == IQ_Q8_BLOCK - 1) { continue; }
        const double error = (double)output[i] - 1.0;
        max_error = std::fmax(max_error, std::fabs(error));
        mean_error += error;
        count++;
    }
    const bool passed = !nonfinite && max_error <= tolerance;
    std::printf("{\"passed\":%s,\"keys\":%u,\"unit_elements\":%zu,"
        "\"max_abs_error\":%.17g,\"mean_error\":%.17g,"
        "\"tolerance\":%.17g,\"nonfinite\":%zu}\n",
        passed ? "true" : "false", keys, count, max_error, mean_error / count,
        (double)tolerance, nonfinite);

    CUDA_OK(cudaFree(q)); CUDA_OK(cudaFree(out)); CUDA_OK(cudaFree(lse));
    CUDA_OK(cudaFree(pos)); CUDA_OK(cudaFree(kv));
    return passed ? 0 : 1;
}
