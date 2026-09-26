// Baseline only: include the production kernel and D4 type unchanged.
#include <cuda.h>
#include "mmq.cuh"
#include "ds4_mimo2_swiglu.cuh"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr int TOKENS = 4096, USED = 8, EXPERTS = 256;
constexpr int ROWS = TOKENS * USED, WIDTH = 2048, THREADS = 128;
constexpr int WARMUPS = 16, DEFAULT_REPS = 100, MAX_REPS = 1000000;
constexpr uint32_t GATE_SEED = 0x5a0a1234u, UP_SEED = 0x7b3d5678u;
constexpr int Q8_VALUES = 128, SCALES = 4;
static_assert(sizeof(block_q8_1_mmq) == 144, "Production D4 block size");
static_assert(WIDTH % (THREADS * 4) == 0, "Production launch geometry");

void check(cudaError_t status) {
    if (status == cudaSuccess) { return; }
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
    std::exit(1);
}

int parse_reps(const char *text) {
    char *end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < 1 || value > MAX_REPS) {
        std::fprintf(stderr, "invalid repetitions: %s\n", text);
        std::exit(2);
    }
    return (int)value;
}

uint32_t mix(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

float *input(uint32_t seed, float amplitude) {
    const size_t count = (size_t)ROWS * WIDTH;
    std::vector<float> host(count);
    for (size_t i = 0; i < count; ++i) {
        // Dyadic, finite inputs; each seed produces a different full tensor.
        const int value = (int)(mix((uint32_t)i ^ seed) & 0xffffu) - 32768;
        host[i] = (float)value * (amplitude / 32768.0f);
    }
    float *device = nullptr;
    check(cudaMalloc(&device, count * sizeof(float)));
    check(cudaMemcpy(device, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    return device;
}

int32_t *sorted_ids() {
    std::array<std::vector<int32_t>, EXPERTS> buckets;
    for (int token = 0; token < TOKENS; ++token) {
        const unsigned first = mix((uint32_t)token ^ GATE_SEED) % EXPERTS;
        for (int slot = 0; slot < USED; ++slot) {
            // Odd stride gives eight distinct experts, then stable expert sort.
            const unsigned expert = (first + 31u * (unsigned)slot) % EXPERTS;
            buckets[expert].push_back(token * USED + slot);
        }
    }
    std::vector<int32_t> host;
    host.reserve(ROWS);
    size_t min_rows = ROWS, max_rows = 0;
    for (const auto &bucket : buckets) {
        if (bucket.size() < min_rows) { min_rows = bucket.size(); }
        if (bucket.size() > max_rows) { max_rows = bucket.size(); }
        host.insert(host.end(), bucket.begin(), bucket.end());
    }
    if (host.size() != ROWS) { std::exit(3); }
    std::printf("routing=expert_major_stable_permutation min_bucket=%zu max_bucket=%zu\n",
                min_rows, max_rows);
    int32_t *device = nullptr;
    check(cudaMalloc(&device, host.size() * sizeof(int32_t)));
    check(cudaMemcpy(device, host.data(), host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    return device;
}

void verify_dump(const block_q8_1_mmq *device, size_t blocks) {
    std::vector<block_q8_1_mmq> host(blocks);
    check(cudaMemcpy(host.data(), device, blocks * sizeof(*device), cudaMemcpyDeviceToHost));
    float min_scale = INFINITY, max_scale = 0.0f;
    for (size_t i = 0; i < blocks; ++i) {
        for (int scale = 0; scale < SCALES; ++scale) {
            const float value = host[i].d4[scale];
            if (!std::isfinite(value) || value <= 0.0f) {
                std::fprintf(stderr, "invalid D4 scale: block=%zu scale=%d value=%g\n", i, scale, value);
                std::exit(4);
            }
            min_scale = std::fmin(min_scale, value);
            max_scale = std::fmax(max_scale, value);
        }
        for (int col = 0; col < Q8_VALUES; ++col) {
            if (host[i].qs[col] == -128) {
                std::fprintf(stderr, "invalid Q8 value: block=%zu col=%d\n", i, col);
                std::exit(4);
            }
        }
    }
    std::printf("finite_d4=PASS blocks=%zu scales=%zu q8_values=%zu scale_min=%.9g scale_max=%.9g\n",
                blocks, blocks * SCALES, blocks * Q8_VALUES, min_scale, max_scale);
    if (const char *path = std::getenv("PROBE_DUMP")) {
        FILE *file = std::fopen(path, "wb");
        if (!file || std::fwrite(host.data(), sizeof(*device), blocks, file) != blocks) {
            std::fprintf(stderr, "failed full D4 dump: %s\n", path);
            std::exit(5);
        }
        if (std::fclose(file) != 0) { std::exit(5); }
        std::printf("dump=%s bytes=%zu\n", path, blocks * sizeof(*device));
    }
}
}

int main(int argc, char **argv) {
    if (argc > 2) {
        std::fprintf(stderr, "usage: %s [repetitions=100]; PROBE_DUMP=full-output.bin\n", argv[0]);
        return 2;
    }
    const int reps = argc == 2 ? parse_reps(argv[1]) : DEFAULT_REPS;
    float *gate = input(GATE_SEED, 4.0f);
    float *up = input(UP_SEED, 2.0f);
    int32_t *ids = sorted_ids();
    const size_t blocks = (size_t)ROWS * (WIDTH / Q8_VALUES);
    block_q8_1_mmq *out = nullptr;
    check(cudaMalloc(&out, blocks * sizeof(*out)));
    check(cudaMemset(out, 0xa5, blocks * sizeof(*out)));
    const dim3 grid(ROWS, (WIDTH + 511) / 512);
    auto launch = [&] {
        mimo2_swiglu_q8<<<grid, THREADS>>>(gate, up, ids, out, WIDTH, ROWS);
        check(cudaGetLastError());
    };

    // Inputs are read-only, outputs overwritten; repeats have identical state.
    for (int i = 0; i < WARMUPS; ++i) { launch(); }
    check(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    check(cudaEventRecord(start));
    for (int i = 0; i < reps; ++i) { launch(); }
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, start, stop));
    std::printf("rows=%d width=%d grid=%u,%u,%u block=%d warmups=%d reps=%d kernel_ms=%.6f\n",
                ROWS, WIDTH, grid.x, grid.y, grid.z, THREADS, WARMUPS, reps, elapsed / reps);
    verify_dump(out, blocks);
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(gate));
    check(cudaFree(up));
    check(cudaFree(ids));
    check(cudaFree(out));
    return 0;
}
