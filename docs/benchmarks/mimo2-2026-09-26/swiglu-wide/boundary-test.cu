// SPDX-License-Identifier: MIT
// Exercise the production MiMo SwiGLU/Q8 launch through its Down entry. Synthetic IQ2_XS
// weights and distinct top-8 routing preserve geometry, not model/cache values.
// Build after the CUDA objects, using the production compiler flags:
// nvcc -O3 -g -lineinfo --use_fast_math -std=c++17 -arch=sm_121a -Icuda/mmq \
//   tests/mimo2_swiglu_wide.cu cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o \
//   cuda/mmq/ds4_mmq_d2r.o cuda/mmq/quantize.o cuda/mmq/mmid.o \
//   cuda/mmq/mmvq.o -lcudart -lcublas -lcuda -o /tmp/mimo2-swiglu
// DS4_MIMO2_SWIGLU_WIDE=0 PROBE_DUMP=/tmp/swiglu-off.bin /tmp/mimo2-swiglu 100 4096
// DS4_MIMO2_SWIGLU_WIDE=1 PROBE_DUMP=/tmp/swiglu-on.bin /tmp/mimo2-swiglu 100 4096
// cmp /tmp/swiglu-off.bin /tmp/swiglu-on.bin
// Unset DS4_MIMO2_SWIGLU_WIDE also selects ON. Compare fresh processes.
// Repeat with 32 and 8192 tokens for the supported dispatch boundaries.
// Full Down parity includes routing, quantized D4 production and its consumer.

#include "ds4_mmq.h"
#include "quantize.cuh"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// This entry never consumes the unrelated dense Q8 fold cache.
extern "C" int ds4_cuda_q8_fold_take_q81(const void *, uint64_t, void *) {
    return 0;
}

namespace {
constexpr int EXPERTS = 256, OUTPUT = 4096, WIDTH = 2048, USED = 8;
constexpr int MIN_TOKENS = 32, MAX_TOKENS = 8192, DEFAULT_TOKENS = 4096;
constexpr int WARMUPS = 16, DEFAULT_REPS = 100, MAX_REPS = 1000000;
constexpr int QUANT_BLOCK = 256;

void check(cudaError_t rc) {
    if (rc == cudaSuccess) { return; }
    std::fprintf(stderr, "%s\n", cudaGetErrorString(rc));
    std::exit(1);
}

int argument(const char *text, int minimum, int maximum) {
    char *end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < minimum || value > maximum) {
        std::fprintf(stderr, "invalid argument: %s (expected %d..%d)\n", text, minimum, maximum);
        std::exit(1);
    }
    return (int)value;
}

void *upload(const void *data, size_t bytes) {
    void *device = nullptr;
    check(cudaMalloc(&device, bytes));
    check(cudaMemcpy(device, data, bytes, cudaMemcpyHostToDevice));
    return device;
}

float *activation(std::mt19937 &rng, size_t count) {
    std::vector<float> host(count);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    for (float &x : host) { x = value(rng); }
    return (float *)upload(host.data(), count * sizeof(float));
}

void dump(const float *output, size_t count) {
    const char *path = std::getenv("PROBE_DUMP");
    std::vector<float> host(count);
    check(cudaMemcpy(host.data(), output, count * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(host[i])) {
            std::fprintf(stderr, "non-finite output at %zu: %g\n", i, host[i]);
            std::exit(1);
        }
    }
    std::printf("full_output_finite=PASS count=%zu bytes=%zu\n", count, count * sizeof(float));
    if (!path) { return; }
    FILE *file = std::fopen(path, "wb");
    if (!file) { std::perror(path); std::exit(1); }
    const bool written = std::fwrite(host.data(), sizeof(float), count, file) == count;
    const int closed = std::fclose(file);
    if (!written || closed != 0) {
        std::fprintf(stderr, "output write failed: %s\n", path);
        std::exit(1);
    }
}

void check_refusal(const void *weights, const float *gate, const float *up,
                   const int32_t *ids, float *output) {
    for (int rows : {(MIN_TOKENS - 1) * USED, MIN_TOKENS * USED - 1, MIN_TOKENS * USED + 1,
                     MAX_TOKENS * USED + USED}) {
        // Unsupported shapes must return before looking at device buffers.
        const int rc = ds4_mmq_mimo2_down(weights, gate, up, ids, output, rows, 0);
        if (rc != -1) {
            std::fprintf(stderr, "bad refusal: rows=%d rc=%d\n", rows, rc);
            std::exit(1);
        }
    }
    const int rc = ds4_mmq_mimo2_down(weights, gate, nullptr, ids, output, MIN_TOKENS * USED, 0);
    if (rc != -1) {
        std::fprintf(stderr, "null-up refusal failed: %d\n", rc);
        std::exit(1);
    }
    std::puts("unsupported_shape_refusal=PASS");
}

void run(int reps, int tokens) {
    std::mt19937 rng(0x5a0au);
    const int rows = tokens * USED;
    const size_t blocks = (size_t)EXPERTS * OUTPUT * WIDTH / QUANT_BLOCK;
    std::vector<block_iq2_xs> weights(blocks);
    const size_t weight_bytes = weights.size() * sizeof(block_iq2_xs);
    // Fixed synthetic weights make fresh OFF/ON processes byte-comparable.
    for (size_t i = 0; i < weight_bytes / sizeof(uint32_t); i++) {
        const uint32_t word = rng();
        std::memcpy((char *)weights.data() + i * sizeof(word), &word, sizeof(word));
    }
    for (block_iq2_xs &block : weights) {
        const uint16_t mantissa = (uint16_t)(rng() & 0x03ffu);
        const uint16_t sign = (uint16_t)((rng() & 1u) << 15);
        block.d = __ushort_as_half((uint16_t)(sign | (14u << 10) | mantissa));
    }
    void *device_weights = upload(weights.data(), weight_bytes);
    float *gate = activation(rng, (size_t)rows * WIDTH);
    float *up = activation(rng, (size_t)rows * WIDTH);

    std::vector<int32_t> ids(rows);
    for (int token = 0; token < tokens; token++) {
        for (int slot = 0; slot < USED; slot++) {
            int expert;
            bool duplicate;
            do {
                expert = (int)(rng() % EXPERTS);
                duplicate = false;
                for (int prev = 0; prev < slot; prev++) {
                    duplicate |= ids[token * USED + prev] == expert;
                }
            } while (duplicate);
            ids[token * USED + slot] = expert;
        }
    }
    int32_t *device_ids = (int32_t *)upload(ids.data(), ids.size() * sizeof(int32_t));
    const size_t count = (size_t)rows * OUTPUT;
    float *output = nullptr;
    check(cudaMalloc(&output, count * sizeof(float)));
    check_refusal(device_weights, gate, up, device_ids, output);
    auto launch = [&] {
        const int rc = ds4_mmq_mimo2_down(device_weights, gate, up, device_ids, output, rows, 0);
        if (rc != 0) {
            std::fprintf(stderr, "MiMo Down entry failed: %d\n", rc);
            std::exit(1);
        }
        check(cudaGetLastError());
    };

    cudaEvent_t start, end;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&end));
    for (int i = 0; i < WARMUPS; i++) { launch(); }
    check(cudaDeviceSynchronize());
    check(cudaEventRecord(start));
    for (int i = 0; i < reps; i++) { launch(); }
    check(cudaEventRecord(end));
    check(cudaEventSynchronize(end));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, end));
    const char *mode = std::getenv("DS4_MIMO2_SWIGLU_WIDE");
    std::printf("mode=%s tokens=%d assignments=%d warmups=%d reps=%d entry_ms=%.6f\n",
                mode ? mode : "default", tokens, rows, WARMUPS, reps, ms / reps);
    dump(output, count);

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(end));
    for (void *buffer : {device_weights, (void *)gate, (void *)up, (void *)device_ids, (void *)output}) {
        check(cudaFree(buffer));
    }
}
} // namespace

int main(int argc, char **argv) {
    if (argc > 3) {
        std::fprintf(stderr, "usage: %s [iterations=100] [tokens=4096]\n", argv[0]);
        return 1;
    }
    const int reps = argc > 1 ? argument(argv[1], 1, MAX_REPS) : DEFAULT_REPS;
    const int tokens = argc > 2 ? argument(argv[2], MIN_TOKENS, MAX_TOKENS) : DEFAULT_TOKENS;
    run(reps, tokens);
    return 0;
}
