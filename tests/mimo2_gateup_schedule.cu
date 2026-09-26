// SPDX-License-Identifier: MIT
// Production IQ2 SoA entry; compare independent processes so cached switches
// match serving. Synthetic weights/routing preserve geometry, not model values.
// Build after the CUDA objects (use the same flags as the production build):
// nvcc -O3 -lineinfo --use_fast_math -std=c++17 -arch=sm_121a -Icuda/mmq \
//   tests/mimo2_gateup_schedule.cu cuda/mmq/ds4_ggml_stubs.o \
//   cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o cuda/mmq/quantize.o \
//   cuda/mmq/mmid.o cuda/mmq/mmvq.o -lcudart -lcublas -lcuda -o /tmp/m2-gateup
// DS4_MIMO2_GATEUP_BOUNDED=0 PROBE_DUMP=/tmp/m2-off /tmp/m2-gateup 100 4096
// DS4_MIMO2_GATEUP_BOUNDED=1 PROBE_DUMP=/tmp/m2-on /tmp/m2-gateup 100 4096
// cmp /tmp/m2-off-0.bin /tmp/m2-on-0.bin
// cmp /tmp/m2-off-1.bin /tmp/m2-on-1.bin
// Repeat with 255/256 tokens for the 2048-assignment dispatch boundary.
// A third argument of 6 checks other-family topology, e.g. "2 512 6".

#include "ds4_mmq.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// This entry never consumes the unrelated dense Q8 fold cache.
extern "C" int ds4_cuda_q8_fold_take_q81(const void *, uint64_t, void *) {
    return 0;
}

namespace {
constexpr int EXPERTS = 256, ROWS = 2048, COLUMNS = 4096, DEFAULT_USED = 8;
constexpr int OTHER_USED = 6;
constexpr int WARMUPS = 16, DEFAULT_REPS = 100, DEFAULT_TOKENS = 4096;
constexpr int MAX_TOKENS = 8192, MAX_REPS = 1000000, QUANT_BLOCK = 256;

void check(cudaError_t rc) {
    if (rc == cudaSuccess) { return; }
    std::fprintf(stderr, "%s\n", cudaGetErrorString(rc));
    std::exit(1);
}

int argument(const char *text, int maximum) {
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end || value < 1 || value > maximum) {
        std::fprintf(stderr, "invalid argument: %s (expected 1..%d)\n", text, maximum);
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

void *weights(std::mt19937 &rng) {
    const uint64_t bytes = ds4_mmq_iq2_xxs_aligned_bytes(ROWS, COLUMNS, EXPERTS);
    const uint64_t blocks = (uint64_t)EXPERTS * ROWS * COLUMNS / QUANT_BLOCK;
    if (!bytes || bytes % sizeof(uint32_t)) {
        std::fprintf(stderr, "invalid IQ2 SoA size\n");
        std::exit(1);
    }
    std::vector<uint32_t> host(bytes / sizeof(uint32_t));
    for (uint32_t &word : host) { word = rng(); }

    // The first SoA plane holds half scales; keep finite magnitudes [0.5, 1).
    for (uint64_t i = 0; i < blocks; i++) {
        const uint16_t mantissa = (uint16_t)(rng() & 0x03ffu);
        const uint16_t sign = (uint16_t)((rng() & 1u) << 15);
        const uint16_t scale = (uint16_t)(sign | (14u << 10) | mantissa);
        std::memcpy((char *)host.data() + i * sizeof(scale), &scale, sizeof(scale));
    }
    return upload(host.data(), bytes);
}

void dump(const float *output, size_t count, int leg) {
    const char *prefix = std::getenv("PROBE_DUMP");
    if (!prefix) { return; }
    std::vector<float> host(count);
    check(cudaMemcpy(host.data(), output, count * sizeof(float), cudaMemcpyDeviceToHost));
    const std::string path = std::string(prefix) + "-" + std::to_string(leg) + ".bin";
    FILE *file = std::fopen(path.c_str(), "wb");
    if (!file) { std::perror(path.c_str()); std::exit(1); }
    const bool written = std::fwrite(host.data(), sizeof(float), count, file) == count;
    const int closed = std::fclose(file);
    if (!written || closed != 0) {
        std::fprintf(stderr, "output write failed: %s\n", path.c_str());
        std::exit(1);
    }
}

void run(int reps, int tokens, int used) {
    std::mt19937 rng(0x5a0au);
    void *gate = weights(rng), *up = weights(rng);
    std::vector<float> host_x((size_t)tokens * COLUMNS);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    for (float &x : host_x) { x = value(rng); }
    float *x = (float *)upload(host_x.data(), host_x.size() * sizeof(float));

    // Each token chooses distinct experts, as in the production router.
    std::vector<int32_t> host_ids((size_t)tokens * used);
    for (int token = 0; token < tokens; token++) {
        for (int slot = 0; slot < used; slot++) {
            int expert;
            bool duplicate;
            do {
                expert = (int)(rng() % EXPERTS);
                duplicate = false;
                for (int prev = 0; prev < slot; prev++) {
                    duplicate |= host_ids[(size_t)token * used + prev] == expert;
                }
            } while (duplicate);
            host_ids[(size_t)token * used + slot] = expert;
        }
    }
    int32_t *ids = (int32_t *)upload(host_ids.data(), host_ids.size() * sizeof(int32_t));
    const size_t count = (size_t)tokens * used * ROWS;
    float *out[2] = {};
    for (float *&buffer : out) { check(cudaMalloc(&buffer, count * sizeof(float))); }

    auto launch = [&] {
        const int rc = ds4_mmq_iq2_xxs_moe_pair_soa(gate, up, x, ids, out[0], out[1],
                ROWS, COLUMNS, tokens, EXPERTS, used, 0, 0);
        if (rc != 0) {
            std::fprintf(stderr, "IQ2 SoA entry failed: %d\n", rc);
            std::exit(1);
        }
        check(cudaGetLastError());
    };
    for (int i = 0; i < WARMUPS; i++) { launch(); }
    check(cudaDeviceSynchronize());
    cudaEvent_t start, end;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&end));
    check(cudaEventRecord(start));
    for (int i = 0; i < reps; i++) { launch(); }
    check(cudaEventRecord(end));
    check(cudaEventSynchronize(end));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, end));
    std::printf("tokens=%d used=%d assignments=%d warmups=%d reps=%d entry_ms=%.6f\n",
                tokens, used, tokens * used, WARMUPS, reps, ms / reps);
    for (int leg = 0; leg < 2; leg++) { dump(out[leg], count, leg); }

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(end));
    for (void *buffer : {gate, up, (void *)x, (void *)ids, (void *)out[0], (void *)out[1]}) {
        check(cudaFree(buffer));
    }
}
} // namespace

int main(int argc, char **argv) {
    if (argc > 4) {
        std::fprintf(stderr, "usage: %s [iterations=100] [tokens=4096] [used=8]\n", argv[0]);
        return 1;
    }
    const int reps = argc > 1 ? argument(argv[1], MAX_REPS) : DEFAULT_REPS;
    const int tokens = argc > 2 ? argument(argv[2], MAX_TOKENS) : DEFAULT_TOKENS;
    const int used = argc > 3 ? argument(argv[3], DEFAULT_USED) : DEFAULT_USED;
    if (used != OTHER_USED && used != DEFAULT_USED) {
        std::fprintf(stderr, "used must be %d or %d\n", OTHER_USED, DEFAULT_USED);
        return 1;
    }
    run(reps, tokens, used);
    return 0;
}
