/* Routed Down producer only: production BF16 boundaries, geometry and D4
 * quantizer; synthetic finite values and stable expert-major row maps. */
#include "../cuda/naive_primitives.cuh"
#include "../cuda/mmq/quantize.cuh"
#include "../cuda/mmq/ds4_mimo2_swiglu.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA(call) do { const cudaError_t rc = (call); if (rc != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(rc)); exit(1); } } while (0)

enum class Routes { Uniform, Concentrated, Invalid };
enum class Producer { Separate, Fused, Mimo };

static __global__ void f32_swiglu(float *out, const float *gate,
                                 const float *up, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i]; }
}

template<class T> static T *upload(const std::vector<T> &values) {
    T *out;
    CUDA(cudaMalloc(&out, values.size() * sizeof(T)));
    CUDA(cudaMemcpy(out, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return out;
}

static void launch(float *mid, const float *gate, const float *up,
                   const int *ids, void *out, unsigned rows, Producer path,
                   SwiGLUOutput output) {
    if (path != Producer::Separate) {
        const dim3 grid(rows, (N05_FF + 511) / 512);
        if (output == SwiGLUOutput::NaiveBF16) {
            mimo2_swiglu_q8<SwiGLUOutput::NaiveBF16><<<grid, 128>>>(
                gate, up, ids, (block_q8_1_mmq *)out, N05_FF, rows);
        } else {
            mimo2_swiglu_q8<<<grid, 128>>>(gate, up, ids,
                (block_q8_1_mmq *)out, N05_FF, rows);
        }
        CUDA(cudaGetLastError());
        return;
    }
    const uint64_t count = (uint64_t)rows * N05_FF;
    if (output == SwiGLUOutput::NaiveBF16) {
        naive_swiglu<<<(count + 255) / 256, 256>>>(mid, gate, up, count);
    } else {
        f32_swiglu<<<(count + 255) / 256, 256>>>(mid, gate, up, count);
    }
    quantize_mmq_q8_1_cuda(mid, ids, out, GGML_TYPE_IQ2_XS,
        N05_FF, N05_FF, N05_FF, count, N05_FF, rows, 1, 1, nullptr);
    CUDA(cudaGetLastError());
}

int main(int argc, char **argv) {
    enum { REPEATS = 20 };
    const unsigned tokens = argc > 1 ? (unsigned)atoi(argv[1]) : N05_PREFILL;
    const auto path = argc > 3 ? static_cast<Producer>(atoi(argv[3])) : Producer::Separate;
    Routes routes = Routes::Uniform;
    if (argc > 2 && !strcmp(argv[2], "concentrated")) { routes = Routes::Concentrated; }
    else if (argc > 2 && !strcmp(argv[2], "invalid")) { routes = Routes::Invalid; }
    else if (argc > 2 && strcmp(argv[2], "uniform")) { return 2; }
    if (!tokens || tokens > N05_PREFILL || path > Producer::Mimo) { return 2; }
    const auto output = path == Producer::Mimo ? SwiGLUOutput::F32 : SwiGLUOutput::NaiveBF16;
    const unsigned rows = tokens * N05_USED;
    const uint64_t count = (uint64_t)rows * N05_FF;
    const size_t bytes = count / 128 * sizeof(block_q8_1_mmq);
    std::vector<float> values(count);
    for (uint64_t i = 0; i < count; i++) {
        values[i] = ((int)((i * 13 + i / N05_FF * 19) % 2047) - 1023) * .031257f;
        if (i % N05_FF < 32) { values[i] = 0; }
    }
    float *gate = upload(values);
    for (uint64_t i = 0; i < count; i++) {
        values[i] = ((int)((i * 7 + i / N05_FF * 23) % 997) - 498) * .125031f;
    }
    float *up = upload(values), *mid;
    std::vector<int> order;
    for (unsigned expert = 0; expert < N05_EXPERTS; expert++) {
        for (unsigned row = 0; row < rows; row++) {
            const unsigned token = row / N05_USED, slot = row % N05_USED;
            const unsigned id = routes == Routes::Concentrated ? slot
                : (token + slot * 37) % N05_EXPERTS;
            if (routes == Routes::Invalid && row % 113 == 0) { continue; }
            if (id == expert) { order.push_back(row); }
        }
    }
    // Dropped assignments leave a zero-initialized, unconsumed map tail.
    order.resize(rows, 0);
    int *ids = upload(order);
    void *out, *reference;
    CUDA(cudaMalloc(&mid, count * sizeof(float)));
    CUDA(cudaMalloc(&out, bytes));
    CUDA(cudaMalloc(&reference, bytes));
    cudaEvent_t start, end;
    CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
    launch(mid, gate, up, ids, reference, rows, Producer::Separate, output);
    launch(mid, gate, up, ids, out, rows, path, output);
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaEventRecord(start));
    for (unsigned repeat = 0; repeat < REPEATS; repeat++) {
        launch(mid, gate, up, ids, out, rows, path, output);
    }
    CUDA(cudaEventRecord(end)); CUDA(cudaEventSynchronize(end));
    float ms;
    CUDA(cudaEventElapsedTime(&ms, start, end));
    std::vector<char> actual(bytes), expected(bytes);
    CUDA(cudaMemcpy(actual.data(), out, bytes, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(expected.data(), reference, bytes, cudaMemcpyDeviceToHost));
    const bool exact = actual == expected;
    printf("tokens=%u rows=%u path=%u producer_ms=%.6f q8_exact=%s\n",
        tokens, rows, (unsigned)path, ms / REPEATS, exact ? "true" : "false");
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
    CUDA(cudaFree(reference)); CUDA(cudaFree(out)); CUDA(cudaFree(mid)); CUDA(cudaFree(ids));
    CUDA(cudaFree(up)); CUDA(cudaFree(gate));
    return exact ? 0 : 1;
}
