// Current production OFF/ON; probe-original is the frozen baseline binary.
#include <cuda_runtime.h>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "../../../cuda/mimo2_primitives.cuh"
#include "../../../cuda/step37_primitives.cuh"

namespace {
constexpr unsigned ROWS = 4096;
constexpr unsigned Q_WIDTH = 64 * 192;
constexpr unsigned ROTARY = 64;
constexpr unsigned THREADS = 256;
constexpr unsigned TABLE_THREADS = 128;
constexpr unsigned DEFAULT_WARMUP = 16;
constexpr size_t READBACK_FLOATS = 1024 * 1024;
constexpr uint64_t FNV_OFFSET = UINT64_C(14695981039346656037);
constexpr uint64_t FNV_PRIME = UINT64_C(1099511628211);

void check(cudaError_t status) {
    if (status == cudaSuccess) { return; }
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(1);
}

unsigned parse_count(const char *value, unsigned minimum, unsigned maximum) {
    char *end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno || end == value || *end || value[0] == '-' ||
        parsed < minimum || parsed > maximum) {
        std::fprintf(stderr, "invalid argument: %s\n", value);
        std::exit(2);
    }
    return (unsigned)parsed;
}

// Hash every output byte and optionally retain complete Q/K/V files.
void inspect(const char *name, const float *device, size_t count) {
    const char *prefix = std::getenv("PROBE_DUMP");
    std::FILE *dump = nullptr;
    if (prefix && *prefix) {
        const std::string path = std::string(prefix) + "." + name + ".f32";
        dump = std::fopen(path.c_str(), "wb");
        if (!dump) { std::perror(path.c_str()); std::exit(1); }
    }
    uint64_t hash = FNV_OFFSET;
    std::vector<float> host(READBACK_FLOATS);
    float first = 0, last = 0;
    for (size_t offset = 0; offset < count; offset += host.size()) {
        const size_t left = count - offset;
        const size_t n = left < host.size() ? left : host.size();
        check(cudaMemcpy(host.data(), device + offset, n * sizeof(float), cudaMemcpyDeviceToHost));
        if (!offset) { first = host[0]; }
        last = host[n - 1];
        for (size_t i = 0; i < n; i++) {
            if (!std::isfinite(host[i])) {
                std::fprintf(stderr, "%s nonfinite at %zu\n", name, offset + i);
                std::exit(1);
            }
        }
        const auto *bytes = reinterpret_cast<const unsigned char *>(host.data());
        for (size_t i = 0; i < n * sizeof(float); i++) {
            hash = (hash ^ bytes[i]) * FNV_PRIME;
        }
        if (dump && std::fwrite(host.data(), sizeof(float), n, dump) != n) {
            std::perror("output write"); std::exit(1);
        }
    }
    if (dump && std::fclose(dump)) { std::perror("output close"); std::exit(1); }
    std::printf("output=%s floats=%zu fnv1a64=%016llx first=%.9g last=%.9g\n",
                name, count, (unsigned long long)hash, first, last);
}
}

int main(int argc, char **argv) {
    if (argc > 5) {
        std::fprintf(stderr, "usage: %s [kv_heads=8] [repetitions=1] [warmup=16] [pos0=4096]\n", argv[0]);
        return 2;
    }
    const unsigned heads = argc > 1 ? parse_count(argv[1], 4, 8) : 8;
    if (heads != 4 && heads != 8) { return 2; }
    const unsigned repeats = argc > 2 ? parse_count(argv[2], 1, UINT_MAX) : 1;
    const unsigned warmup = argc > 3 ? parse_count(argv[3], 0, UINT_MAX) : DEFAULT_WARMUP;
    const unsigned pos0 = argc > 4 ? parse_count(argv[4], 0, UINT_MAX - ROWS) : ROWS;
    const unsigned kw = heads * 192, vw = heads * 128;
    const unsigned stride = Q_WIDTH + kw + vw;
    const size_t count = (size_t)ROWS * stride;
    const size_t q_count = (size_t)ROWS * Q_WIDTH;
    const size_t k_count = (size_t)ROWS * kw, v_count = (size_t)ROWS * vw;
    const unsigned blocks = (unsigned)((count + THREADS - 1) / THREADS);
    const auto mapping = m2_rope_map(ROWS);
    float *qkv, *q, *k, *v, *frequency;
    float2 *table;
    unsigned *positions;
    check(cudaMalloc(&qkv, count * sizeof(float)));
    check(cudaMalloc(&q, q_count * sizeof(float)));
    check(cudaMalloc(&k, k_count * sizeof(float)));
    check(cudaMalloc(&v, v_count * sizeof(float)));
    check(cudaMalloc(&frequency, ROTARY / 2 * sizeof(float)));
    check(cudaMalloc(&positions, ROWS * sizeof(unsigned)));
    check(cudaMalloc(&table, (size_t)ROWS * ROTARY / 2 * sizeof(float2)));
    {
        std::vector<float> input(count);
        for (size_t i = 0; i < count; i++) {
            uint32_t bits = (uint32_t)i * UINT32_C(747796405) + UINT32_C(2891336453);
            bits = ((bits >> ((bits >> 28) + 4)) ^ bits) * UINT32_C(277803737);
            bits = (bits >> 22) ^ bits;
            input[i] = ((int)(bits & UINT32_C(0xffff)) - 32768) / 4096.0f;
        }
        check(cudaMemcpy(qkv, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    }
    // Match graph allocation: double pow -> float frequency; live row positions.
    std::vector<float> host_freq(ROTARY / 2);
    std::vector<unsigned> host_pos(ROWS);
    const double theta = heads == 4 ? 10000000.0 : 10000.0;
    for (unsigned d = 0; d < ROTARY / 2; d++) {
        host_freq[d] = (float)std::pow(theta, -2.0 * d / ROTARY);
    }
    for (unsigned row = 0; row < ROWS; row++) { host_pos[row] = pos0 + row; }
    check(cudaMemcpy(frequency, host_freq.data(), host_freq.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(positions, host_pos.data(), host_pos.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
    step37_rope_table<<<(ROWS * ROTARY / 2 + TABLE_THREADS - 1) / TABLE_THREADS, TABLE_THREADS>>>(
        table, frequency, positions, ROTARY / 2, ROWS);
    check(cudaGetLastError());
    auto launch = [&] {
        if (mapping == Mimo2RopeMap::Rows) {
            mimo2_split_rope<Mimo2RopeMap::Rows>
                <<<dim3((stride + THREADS - 1) / THREADS, ROWS), THREADS>>>(
                    q, k, v, qkv, table, heads, ROWS);
        } else {
            mimo2_split_rope<<<blocks, THREADS>>>(q, k, v, qkv, table, heads, ROWS);
        }
        check(cudaGetLastError());
    };
    for (unsigned i = 0; i < warmup; i++) { launch(); }
    check(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    double total_ms = 0;
    for (unsigned i = 0; i < repeats; i++) {
        check(cudaEventRecord(start));
        launch();
        check(cudaEventRecord(stop));
        check(cudaEventSynchronize(stop));
        float ms = 0;
        check(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }
    std::printf("candidate mapping=%s rows=%u kv_heads=%u pos0=%u stride=%u blocks=%u block=%u "
                "repetitions=%u warmup=%u kernel_ms=%.6f\n",
                mapping == Mimo2RopeMap::Rows ? "rows" : "linear",
                ROWS, heads, pos0, stride, blocks, THREADS, repeats, warmup, total_ms / repeats);
    inspect("q", q, q_count);
    inspect("k", k, k_count);
    inspect("v", v, v_count);
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    for (void *ptr : {(void *)qkv, (void *)q, (void *)k, (void *)v,
                     (void *)frequency, (void *)positions, (void *)table}) {
        check(cudaFree(ptr));
    }
    return 0;
}
