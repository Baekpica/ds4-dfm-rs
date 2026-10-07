/* Real Q8 shared weights, synthetic activations; production CUDA wrappers. */
#include "../ds4_gpu.h"
extern "C" {
#include "../ds4.h"
}
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "shared FAIL %d: %s\n", __LINE__, #x); std::exit(1); \
} } while (0)
#define CUDA(x) CHECK((x) == cudaSuccess)
enum { K = 4096, M = 2048, QK = 32, Q8_BYTES = 34, GUARD = 16,
       WARM_CALLS = 3, TIME_CALLS = 12 };
enum Run { NUMERIC, TIMING, PROFILE };
enum Scale { HALF_SCALE, FULL_SCALE };
static constexpr float CLAMP = 10.0f;
static constexpr float SENTINEL = 12345.0f;
static constexpr float CLAMP_PROBE_SCALE = 128.0f;
static constexpr float INPUT_MAX = 2.0f;
static constexpr double CPU_ABS = 3.0e-4;
static constexpr double CPU_REL = 2.0e-5;
static constexpr uint64_t WEIGHT_BYTES = uint64_t(M) * K / QK * Q8_BYTES;

static ds4_gpu_tensor *upload(const void *src, uint64_t bytes) {
    auto *t = ds4_gpu_tensor_alloc(bytes);
    CHECK(t && ds4_gpu_tensor_write(t, 0, src, bytes));
    return t;
}

static float half_value(const unsigned char *p) {
    __half h;
    std::memcpy(&h, p, sizeof(h));
    return __half2float(h);
}

struct InputBlock { int8_t q[QK]; float d; };
static std::vector<InputBlock> quant_ref(const float *x, Scale scale) {
    std::vector<InputBlock> out(K / QK);
    for (unsigned b = 0; b < out.size(); b++) {
        float maximum = 0.0f;
        for (unsigned i = 0; i < QK; i++) {
            maximum = std::max(maximum, std::fabs(x[b * QK + i]));
        }
        const float inverse = maximum ? 127.0f / maximum : 0.0f;
        // MMVQ stores an FP16 scale; the wide MMQ D4 layout retains FP32.
        const float d = maximum / 127.0f;
        out[b].d = scale == HALF_SCALE ? __half2float(__float2half_rn(d)) : d;
        for (unsigned i = 0; i < QK; i++) {
            out[b].q[i] = int8_t(std::round(x[b * QK + i] * inverse));
        }
    }
    return out;
}

static double dot_ref(const unsigned char *w, const std::vector<InputBlock> &x) {
    double total = 0.0;
    for (unsigned b = 0; b < x.size(); b++) {
        const auto *p = w + b * Q8_BYTES;
        int dot = 0;
        for (unsigned i = 0; i < QK; i++) {
            dot += int(int8_t(p[2u + i])) * int(x[b].q[i]);
        }
        total += double(dot) * half_value(p) * x[b].d;
    }
    return total;
}

static void cpu_check(uint32_t rows, const unsigned char *weights,
        const std::vector<float> &x, const std::vector<float> &out) {
    const uint32_t tokens[] = {0u, rows / 3u, rows / 2u, rows - 1u};
    const uint32_t channels[] = {0u, 1u, 17u, 127u, 255u, 511u, 1023u, 1024u,
        1535u, 1791u, 2046u, 2047u};
    double worst = 0.0;
    uint32_t clipped = 0;
    for (uint32_t t : tokens) {
        const auto q = quant_ref(x.data() + uint64_t(t) * K,
            rows == 1u ? HALF_SCALE : FULL_SCALE);
        for (uint32_t c : channels) {
            double g = dot_ref(weights + uint64_t(c) * K / QK * Q8_BYTES, q);
            double u = dot_ref(weights + WEIGHT_BYTES + uint64_t(c) * K / QK * Q8_BYTES, q);
            CHECK(std::isfinite(g) && std::isfinite(u));
            clipped += g > CLAMP || std::fabs(u) > CLAMP;
            g = std::min(g, double(CLAMP));
            u = std::max(-double(CLAMP), std::min(u, double(CLAMP)));
            const double expected = g / (1.0 + std::exp(-g)) * u;
            const double error = std::fabs(out[uint64_t(t) * M + c] - expected);
            if (error > CPU_ABS + CPU_REL * std::fabs(expected)) {
                std::fprintf(stderr, "CPU mismatch row=%u channel=%u actual=%.9g expected=%.9g\n",
                    t, c, out[uint64_t(t) * M + c], expected);
            }
            CHECK(error <= CPU_ABS + CPU_REL * std::fabs(expected));
            worst = std::max(worst, error);
        }
    }
    std::printf("CPU samples=48 max_abs=%.9g clipped=%u PASS\n", worst, clipped);
}

static void launch(ds4_gpu_tensor *out, ds4_gpu_tensor *g, ds4_gpu_tensor *u,
        const ds4_gpu_tensor *x, const void *weights, uint32_t rows) {
    const char *value = std::getenv("DS4_GLM53_SHARED_Q8");
    const bool enabled = !value || value[0] != '0';
    if (rows == 1u) {
        const int result = ds4_gpu_glm53_shared_q8(out, weights, 2u * WEIGHT_BYTES,
            0u, WEIGHT_BYTES, K, M, x, CLAMP);
        CHECK(result == (enabled ? 1 : 0));
        if (result == 1) { return; }
    }
    CHECK(ds4_gpu_matmul_q8_0_tensor(g, weights, 2u * WEIGHT_BYTES, 0u, K, M, x, rows));
    CHECK(ds4_gpu_matmul_q8_0_tensor(u, weights, 2u * WEIGHT_BYTES, WEIGHT_BYTES, K, M, x, rows));
    CHECK(ds4_gpu_swiglu_tensor(out, g, u, rows * M, CLAMP, 1.0f));
}

static void refusal_check(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *weights) {
    auto *short_out = ds4_gpu_tensor_view(out, 0, (M - 1u) * sizeof(float));
    CHECK(short_out);
    CHECK(ds4_gpu_glm53_shared_q8(short_out, weights, 2u * WEIGHT_BYTES,
        0, WEIGHT_BYTES, K, M, x, CLAMP) == 0);
    CHECK(ds4_gpu_glm53_shared_q8(out, weights, 2u * WEIGHT_BYTES,
        0, WEIGHT_BYTES, K - QK, M, x, CLAMP) == 0);
    CHECK(ds4_gpu_glm53_shared_q8(out, weights, 2u * WEIGHT_BYTES,
        0, WEIGHT_BYTES, K, M, x, -CLAMP) == 0);
    CHECK(ds4_gpu_glm53_shared_q8(out, weights, 2u * WEIGHT_BYTES,
        UINT64_MAX, WEIGHT_BYTES, K, M, x, CLAMP) == 0);
    ds4_gpu_tensor_free(short_out);
}

int main(int argc, char **argv) {
    CHECK(argc == 5);
    const uint32_t rows = uint32_t(std::strtoul(argv[2], nullptr, 10));
    CHECK(rows == 1u || rows == 128u);
    Run mode = NUMERIC;
    if (!std::strcmp(argv[3], "time")) { mode = TIMING; }
    else if (!std::strcmp(argv[3], "ncu")) { mode = PROFILE; }
    else { CHECK(!std::strcmp(argv[3], "numeric") || !std::strcmp(argv[3], "clamp")); }
    const float input_scale = !std::strcmp(argv[3], "clamp") ? CLAMP_PROBE_SCALE : 1.0f;
    const int fd = open(argv[1], O_RDONLY);
    struct stat st;
    CHECK(fd >= 0 && !fstat(fd, &st) && uint64_t(st.st_size) == 2u * WEIGHT_BYTES);
    const auto *weights = static_cast<const unsigned char *>(
        mmap(nullptr, 2u * WEIGHT_BYTES, PROT_READ, MAP_PRIVATE, fd, 0));
    CHECK(weights != MAP_FAILED);
    // This bounded tensor copy isolates raw Q8 from huge-map/SSD behavior.
    CHECK(!setenv("DS4_CUDA_COPY_MODEL", "1", 1));
    CHECK(!setenv("DS4_CUDA_NO_Q8_ALIGNED", "1", 1));
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(weights, 2u * WEIGHT_BYTES));
    CHECK(!ds4_gpu_mem_census_faults());
    std::vector<float> input(uint64_t(rows) * K);
    uint32_t state = 0x74265813u;
    for (auto &v : input) {
        state ^= state << 13u; state ^= state >> 17u; state ^= state << 5u;
        v = input_scale * float(int(state % 1025u) - 512) / 256.0f;
    }
    // Exact power-of-two maxima avoid fast-division ambiguity at Q8 ties.
    for (uint64_t i = 0; i < input.size(); i += QK) {
        input[i] = std::copysign(INPUT_MAX * input_scale, input[i]);
    }
    std::vector<float> output(uint64_t(rows) * M + GUARD, SENTINEL);
    auto *x = upload(input.data(), input.size() * sizeof(float));
    auto *g = upload(output.data(), output.size() * sizeof(float));
    auto *u = upload(output.data(), output.size() * sizeof(float));
    auto *out = upload(output.data(), output.size() * sizeof(float));
    refusal_check(out, x, weights);
    std::printf("actual_Q8_weights=true synthetic_input=true rows=%u M=%u K=%u clamp=%.9g\n",
        rows, M, K, CLAMP);
    for (int i = 0; i < WARM_CALLS; i++) { launch(out, g, u, x, weights, rows); }
    CUDA(cudaDeviceSynchronize());
    if (mode == PROFILE) {
        CUDA(cudaProfilerStart());
        launch(out, g, u, x, weights, rows);
        CUDA(cudaDeviceSynchronize());
        CUDA(cudaProfilerStop());
    }
    if (mode == TIMING) {
        cudaEvent_t begin, end;
        CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end));
        for (int i = 0; i < TIME_CALLS; i++) {
            CUDA(cudaEventRecord(begin));
            launch(out, g, u, x, weights, rows);
            CUDA(cudaEventRecord(end)); CUDA(cudaEventSynchronize(end));
            float ms;
            CUDA(cudaEventElapsedTime(&ms, begin, end));
            std::printf("call=%d ms=%.9g\n", i, ms);
        }
        CUDA(cudaEventDestroy(begin)); CUDA(cudaEventDestroy(end));
    }
    CHECK(ds4_gpu_tensor_read(out, 0u, output.data(), output.size() * sizeof(float)));
    for (uint64_t i = 0; i < uint64_t(rows) * M; i++) {
        CHECK(std::isfinite(output[i]) && output[i] != SENTINEL && std::fabs(output[i]) <= CLAMP * CLAMP);
    }
    for (unsigned i = 0; i < GUARD; i++) { CHECK(output[uint64_t(rows) * M + i] == SENTINEL); }
    cpu_check(rows, weights, input, output);
    FILE *file = std::fopen(argv[4], "wb");
    CHECK(file && std::fwrite(output.data(), sizeof(float), uint64_t(rows) * M, file) == uint64_t(rows) * M);
    CHECK(!std::fclose(file) && !ds4_gpu_mem_census_faults());
    std::printf("all_checked=%llu guard=%u output=%s PASS\n",
        (unsigned long long)(uint64_t(rows) * M), GUARD, argv[4]);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(u); ds4_gpu_tensor_free(g); ds4_gpu_tensor_free(x);
    CHECK(!munmap((void *)weights, 2u * WEIGHT_BYTES) && !close(fd));
}
