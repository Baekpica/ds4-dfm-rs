/* GLM Q2_K down target: production public MMQ API, synthetic routing/mid.
 * Demand VMM preserves all3389 cache slots without loading unused weights. */
#include "../ds4_gpu.h"
#include "../cuda/mmq/ds4_mmq.h"
extern "C" {
#include "../ds4.h"
}
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#define GGML_COMMON_DECL_C
#include "../cuda/mmq/ggml-common.h"

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "GLM Q2 FAIL %d: %s\n", __LINE__, #x); std::exit(1); \
} } while (0)
#define CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "GLM Q2 CUDA FAIL %d: %s\n", __LINE__, \
                 cudaGetErrorString(e)); std::exit(2); \
} } while (0)

#ifndef GLM_Q2_ROWS
#define GLM_Q2_ROWS 128
#endif
enum { M = 4096, K = 2048, ROWS = GLM_Q2_ROWS, USED = 8, ASSIGN = ROWS * USED,
       EXPERTS = 288, CACHE = 3389, QK = 256, GUARD = 16 };
static constexpr uint64_t UNIT = uint64_t(K / QK) * M * sizeof(block_q2_K);
static constexpr uint64_t STRIDE = 2753688;
static constexpr float SENTINEL = 12345.0f;
static constexpr unsigned char PAD = 0x5a;
static_assert(sizeof(block_q2_K) == 84 && UNIT == 2752512, "canonical Q2_K shape");
static_assert(ROWS >= 32 && ROWS <= 128, "bounded Q2 prefill geometry");

static int slot(int expert) { return (53 * expert + CACHE - 1) % CACHE; }
static void *ptr(ds4_gpu_tensor *tensor) {
    return const_cast<void *>(ds4_gpu_tensor_ptr(tensor));
}

static ds4_gpu_tensor *alloc(uint64_t bytes, const void *data = nullptr) {
    auto *tensor = ds4_gpu_tensor_alloc(bytes);
    CHECK(tensor);
    if (data) { CHECK(ds4_gpu_tensor_write(tensor, 0, data, bytes)); }
    return tensor;
}

static ds4_gpu_tensor *reserve(uint64_t bytes) {
    auto *tensor = ds4_gpu_tensor_reserve(bytes);
    CHECK(tensor); /* Never fall back to a9.33GB physical allocation. */
    return tensor;
}

static void save(const std::string &path, const void *data, uint64_t bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    CHECK(file.good() && bool(file.write((const char *)data, bytes)));
}

static std::vector<float> read(ds4_gpu_tensor *tensor, uint64_t count) {
    std::vector<float> data(count + GUARD);
    CHECK(ds4_gpu_tensor_read(tensor, 0, data.data(), data.size() * sizeof(float)));
    for (uint64_t i = 0; i < count; ++i) { CHECK(std::isfinite(data[i]) && data[i] != SENTINEL); }
    for (uint64_t i = count; i < data.size(); ++i) { CHECK(data[i] == SENTINEL); }
    data.resize(count);
    return data;
}

static float half_float(ggml_half bits) {
    __half value;
    std::memcpy(&value, &bits, sizeof(value));
    return __half2float(value);
}
static float half_round(float value) { return __half2float(__float2half_rn(value)); }

struct Quant {
    std::array<int, K> codes;
    std::array<float, K / 64> scales;
    std::array<float, K / 16> sums;
};

/* Independent D2S6: one half scale/64, six original half sums/128.
 * Dyadic input keeps quantizer ties/reduction-order ambiguity out of this gate. */
static Quant quant(const float *x) {
    Quant q = {};
    for (unsigned start = 0; start < K; start += 64) {
        float maximum = 0;
        for (unsigned i = 0; i < 64; ++i) { maximum = std::max(maximum, std::fabs(x[start + i])); }
        CHECK(maximum > 0);
        const float inverse = 127.0f / maximum;
        q.scales[start / 64] = half_round(1.0f / inverse);
        for (unsigned i = 0; i < 64; ++i) { q.codes[start + i] = int(std::round(x[start + i] * inverse)); }
    }
    for (unsigned start = 0; start < K; start += 16) {
        float lanes[4];
        for (unsigned i = 0; i < 4; ++i) {
            const auto *p = x + start + i * 4;
            lanes[i] = ((p[0] + p[1]) + p[2]) + p[3];
        }
        q.sums[start / 16] = half_round((lanes[0] + lanes[2]) + (lanes[1] + lanes[3]));
    }
    return q;
}

struct Dot { double value, magnitude; };
static Dot dot(const unsigned char *raw, const Quant &q) {
    Dot result = {};
    for (unsigned block = 0; block < K / QK; ++block) {
        const auto *w = (const block_q2_K *)(raw + block * sizeof(block_q2_K));
        for (unsigned group = 0; group < QK / 16; ++group) {
            const unsigned start = block * QK + group * 16;
            const float d = half_round(half_float(w->d) * (w->scales[group] & 15u));
            const float m = half_round(half_float(w->dmin) * (w->scales[group] >> 4));
            int sum = 0, qs = 0;
            for (unsigned i = 0; i < 16; ++i) {
                const unsigned within = group * 16 + i;
                const unsigned code = (w->qs[(within / 128) * 32 + within % 32] >>
                                       (2 * ((within % 128) / 32))) & 3u;
                sum += int(code) * q.codes[start + i];
                qs += q.codes[start + i];
            }
            const double positive = double(d) * sum * q.scales[start / 64];
            const double negative = group % 8 < 6 ? double(m) * q.sums[start / 16]
                : double(m) * qs * q.scales[start / 64];
            result.value += positive - negative;
            result.magnitude += std::fabs(positive) + std::fabs(negative);
        }
    }
    return result;
}

static bool cpu_samples(const std::vector<unsigned char> &weights,
        const std::vector<float> &mid, const std::vector<int32_t> &ids,
        const std::vector<float> &down) {
    bool pass = true;
    double worst = 0;
    const double epsilon = std::numeric_limits<float>::epsilon();
    /*128 signed group contributions, including F32 scale products/accumulation.
     * This absolute forward-error allowance scales with cancellation, not logits. */
    const double gamma = 256 * epsilon / (1 - 256 * epsilon);
    /* Keep the 128-row boundary samples; smaller widths cover their own tail. */
    const unsigned samples[] = {0u, USED - 1u,
        ASSIGN > EXPERTS ? EXPERTS - 1u : ASSIGN / 2u - 1u,
        ASSIGN > EXPERTS ? EXPERTS : ASSIGN / 2u,
        ASSIGN > 512 ? 511u : ASSIGN - USED,
        ASSIGN - 1u};
    for (unsigned assignment : samples) {
        CHECK(assignment < ASSIGN);
        const auto q = quant(mid.data() + uint64_t(assignment) * K);
        for (unsigned row : {0u, 1u, 2u, 127u, 128u, unsigned(M - 1)}) {
            const auto reference = dot(weights.data() + uint64_t(ids[assignment]) * UNIT +
                                       uint64_t(row) * UNIT / M, q);
            const float actual = down[uint64_t(assignment) * M + row];
            const double delta = std::fabs(double(actual) - reference.value);
            const double bound = gamma * reference.magnitude + 1.0e-7;
            worst = std::max(worst, delta);
            std::printf("cpu assignment=%u expert=%d row=%u ref=%.12g got=%.12g abs=%.9g bound=%.9g\n",
                        assignment, ids[assignment], row, reference.value, actual, delta, bound);
            pass &= delta <= bound;
        }
    }
    std::printf("cpu sampled_dots=36 max_abs=%.9g pass=%d\n", worst, int(pass));
    return pass;
}

static __global__ void ordered_sum(float *out, const float *down) {
    const unsigned at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= ROWS * M) { return; }
    const unsigned row = at / M, col = at % M;
    float sum = 0;
    for (unsigned expert = 0; expert < USED; ++expert) { sum += down[(row * USED + expert) * M + col]; }
    out[at] = sum;
}

static std::vector<float> summed(ds4_gpu_tensor *out, ds4_gpu_tensor *down) {
    CHECK(ds4_gpu_tensor_fill_f32(out, SENTINEL, uint64_t(ROWS) * M + GUARD));
    ordered_sum<<<ROWS * M / 256, 256>>>((float *)ptr(out), (const float *)ptr(down));
    CUDA(cudaGetLastError());
    return read(out, uint64_t(ROWS) * M);
}

static void run(ds4_gpu_tensor *w, ds4_gpu_tensor *x, ds4_gpu_tensor *ids,
                ds4_gpu_tensor *out, int experts, uint64_t stride) {
    CHECK(ds4_mmq_glm_moe(10, ptr(w), (const float *)ptr(x), (const int32_t *)ptr(ids),
        (float *)ptr(out), M, K, ASSIGN, experts, 1, stride, (cudaStream_t)0) == 0);
    CUDA(cudaGetLastError());
}

int main(int argc, char **argv) {
    CHECK(argc == 4 || argc == 5);
    const std::string mode = argv[2], directory = argv[3];
    CHECK(mode == "numeric" || mode == "time" || mode == "ncu");
    const bool skew = argc == 5;
    CHECK(!skew || std::strcmp(argv[4], "skew8") == 0);
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    CHECK(file.good() && uint64_t(file.tellg()) == EXPERTS * UNIT);
    std::vector<unsigned char> weights(EXPERTS * UNIT);
    file.seekg(0);
    CHECK(bool(file.read((char *)weights.data(), weights.size())));
    std::vector<float> mid(uint64_t(ASSIGN) * K);
    std::vector<int32_t> ids(ASSIGN), remapped(ASSIGN), histogram(EXPERTS);
    const int skew_ids[USED] = {287, 256, 270, 3, 128, 255, 1, 281};
    for (unsigned assignment = 0; assignment < ASSIGN; ++assignment) {
        const unsigned row = assignment / USED, at = assignment % USED;
        ids[assignment] = skew ? skew_ids[(at + row) % USED] : assignment % EXPERTS;
        remapped[assignment] = slot(ids[assignment]);
        ++histogram[ids[assignment]];
        const float scale = std::ldexp(1.0f, int(assignment % 6) - 3);
        for (unsigned col = 0; col < K; ++col) {
            int value = int((col * 37 + assignment * 19 + col / 64 * 11) % 223) - 111;
            if (value == 64 || value == -64) { ++value; }
            mid[uint64_t(assignment) * K + col] = col % 64 == 0 ? 2 * scale : value * scale / 64;
        }
    }
    save(directory + "/mid.f32", mid.data(), mid.size() * sizeof(float));
    save(directory + "/ids.i32", ids.data(), ids.size() * sizeof(int32_t));
    save(directory + "/remapped.i32", remapped.data(), remapped.size() * sizeof(int32_t));
    save(directory + "/histogram.i32", histogram.data(), histogram.size() * sizeof(int32_t));
    CHECK(ds4_gpu_init() && ds4_gpu_vmm_demand_page());
    const uint64_t faults = ds4_gpu_mem_census_faults(), gov_faults = ds4_metrics_get()->memgov_faults;
    auto *cache = reserve(CACHE * STRIDE), *canonical = reserve(EXPERTS * UNIT);
    unsigned active = 0;
    int bucket_min = ASSIGN, bucket_max = 0;
    for (unsigned expert = 0; expert < EXPERTS; ++expert) {
        if (!histogram[expert]) { continue; }
        ++active;
        bucket_min = std::min(bucket_min, histogram[expert]);
        bucket_max = std::max(bucket_max, histogram[expert]);
        const uint64_t offset = uint64_t(slot(expert)) * STRIDE;
        CHECK(ds4_gpu_tensor_ensure(cache, offset, STRIDE));
        CHECK(ds4_gpu_tensor_ensure(canonical, expert * UNIT, UNIT));
        CHECK(ds4_gpu_tensor_write(cache, offset, weights.data() + expert * UNIT, UNIT));
        CHECK(ds4_gpu_tensor_write(canonical, expert * UNIT, weights.data() + expert * UNIT, UNIT));
        CUDA(cudaMemset((char *)ptr(cache) + offset + UNIT, PAD, STRIDE - UNIT));
    }
    /* The retained Spark rectangle uses 128 rows/columns per tile. */
    const uint64_t grid = uint64_t(M / 128) * ((ASSIGN + 127) / 128) * CACHE;
    std::printf("M=%d K=%d N=%d top=%d assignments=%d used=1 experts=%d stride=%llu expected_grid=%llu "
        "histogram=%s active=%u bucket_min=%d bucket_max=%d address_span=%llu populated_source=%llu padding=%llu "
        "cache_resident=%llu canonical_control_resident=%llu page=%llu\n",
        M, K, ROWS, USED, ASSIGN, CACHE, (unsigned long long)STRIDE, (unsigned long long)grid,
        skew ? "synthetic_skew8" : "synthetic_balanced288", active, bucket_min, bucket_max,
        (unsigned long long)(CACHE * STRIDE), (unsigned long long)(active * UNIT),
        (unsigned long long)(active * (STRIDE - UNIT)),
        (unsigned long long)ds4_gpu_tensor_resident(cache, 0, CACHE * STRIDE),
        (unsigned long long)ds4_gpu_tensor_resident(canonical, 0, EXPERTS * UNIT),
        (unsigned long long)ds4_gpu_vmm_demand_page());
    auto *x = alloc(mid.size() * sizeof(float), mid.data());
    auto *selected = alloc(ids.size() * sizeof(int32_t), ids.data());
    auto *mapped = alloc(remapped.size() * sizeof(int32_t), remapped.data());
    auto *out = alloc((uint64_t(ASSIGN) * M + GUARD) * sizeof(float));
    auto *sum = alloc((uint64_t(ROWS) * M + GUARD) * sizeof(float));
    CHECK(ds4_gpu_tensor_fill_f32(out, SENTINEL, uint64_t(ASSIGN) * M + GUARD));
    run(canonical, x, selected, out, EXPERTS, UNIT);
    const auto reference = read(out, uint64_t(ASSIGN) * M), ref_sum = summed(sum, out);
    save(directory + "/reference.f32", reference.data(), reference.size() * sizeof(float));
    save(directory + "/reference-sum.f32", ref_sum.data(), ref_sum.size() * sizeof(float));
    CHECK(ds4_gpu_tensor_fill_f32(out, SENTINEL, uint64_t(ASSIGN) * M + GUARD));
    if (mode == "ncu") { CUDA(cudaProfilerStart()); }
    run(cache, x, mapped, out, CACHE, STRIDE);
    CUDA(cudaDeviceSynchronize());
    if (mode == "ncu") { CUDA(cudaProfilerStop()); }
    const auto actual = read(out, uint64_t(ASSIGN) * M), actual_sum = summed(sum, out);
    save(directory + "/down.f32", actual.data(), actual.size() * sizeof(float));
    save(directory + "/sum.f32", actual_sum.data(), actual_sum.size() * sizeof(float));
    CHECK(std::memcmp(actual.data(), reference.data(), actual.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(actual_sum.data(), ref_sum.data(), actual_sum.size() * sizeof(float)) == 0);
    CHECK(cpu_samples(weights, mid, ids, actual));
    std::vector<unsigned char> padding(STRIDE - UNIT);
    for (unsigned expert = 0; expert < EXPERTS; ++expert) {
        if (!histogram[expert]) { continue; }
        CHECK(ds4_gpu_tensor_read(cache, uint64_t(slot(expert)) * STRIDE + UNIT, padding.data(), padding.size()));
        for (unsigned char byte : padding) { CHECK(byte == PAD); }
    }
    if (mode == "time") {
        cudaEvent_t start, end;
        CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
        for (unsigned iteration = 0; iteration < 5; ++iteration) {
            const auto wall_start = std::chrono::steady_clock::now();
            CUDA(cudaEventRecord(start));
            run(cache, x, mapped, out, CACHE, STRIDE);
            CUDA(cudaEventRecord(end)); CUDA(cudaEventSynchronize(end));
            const double wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - wall_start).count();
            float ms = 0;
            CUDA(cudaEventElapsedTime(&ms, start, end));
            std::printf("timing iteration=%u event_ms=%.9g wall_ms=%.9g scope=MMID+gatherQ8+MMQ+sanitize\n",
                        iteration, ms, wall_ms);
        }
        CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
        CHECK(std::memcmp(read(out, uint64_t(ASSIGN) * M).data(), actual.data(), actual.size() * sizeof(float)) == 0);
    }
    for (auto *tensor : {cache, canonical, x, selected, mapped, out, sum}) { ds4_gpu_tensor_free(tensor); }
    CUDA(cudaDeviceSynchronize());
    CHECK(ds4_gpu_mem_census_faults() == faults && ds4_metrics_get()->memgov_faults == gov_faults);
    std::printf("GLM Q2 PASS allslot+ordered_sum byte_exact, CPU sampled, padding/output guards, faults unchanged\n");
    return 0;
}
