/* Actual block0 dense tensors and exact production Q8 dispatcher, with only
 * source/artifact pointer resolution mocked. This is eager MMQ, no graph/model. */
#include "../ds4_gpu.h"
#include "../cuda/mmq/ds4_mmq.h"
extern "C" {
#include "../ds4.h"
}
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "GLM dense FAIL %d: %s\n", __LINE__, #x); std::exit(1); \
} } while (0)
#define CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "GLM dense CUDA FAIL %d: %s\n", __LINE__, \
                 cudaGetErrorString(e)); std::exit(2); \
} } while (0)
enum { MAX_ROWS = 128, GUARD = 16, CUDA_DERIVED_Q8_0_ALIGNED_DENSE = 5,
       DS4_Q8_DENSE_TAIL_MIN_TOKENS = 64 };
static constexpr float SENTINEL = 12345.0f;
static constexpr double DOT_BOUND = 1.0e-4;
enum class Arm { Raw, Aligned };
#include "glm53_dense_types.inc"

static const char *raw_ptr, *aligned_ptr;
static uint64_t raw_bytes;
static bool artifact;
static std::vector<void *> scratch;
static uint64_t scratch_bytes;
static cublasHandle_t g_cublas;
static int g_attention_output_b_n2_q8_override;
static int cuda_ok(cudaError_t error, const char *label) {
    if (error == cudaSuccess) { return 1; }
    std::fprintf(stderr, "%s: %s\n", label, cudaGetErrorString(error));
    return 0;
}
static cudaStream_t ds4_current_stream() { return 0; }
static cudaStream_t ds4_mmq_stream_for_call() { return 0; }
static int ds4_cuda_use_mmq() { return 1; }
static int ds4_cuda_use_cublas_q8() { return 0; }
static int ds4_capture_active() { return 0; }
static int ds4_cuda_inner_graphs_want(uint32_t) { return 0; }
static uint32_t cuda_mmvq_decode_max_tokens() { return 8; }
static uint32_t cuda_q8_aligned_enabled() { return 1; }
static int cuda_q8_use_dp4a() { return 1; }
static const char *cuda_model_range_ptr(const void *, uint64_t offset, uint64_t bytes, const char *) {
    CHECK(offset == 0 && bytes == raw_bytes);
    return raw_ptr;
}
static const char *cuda_derived_weight_ptr(const void *, uint64_t offset, uint64_t bytes,
        int kind, uint64_t, uint64_t, uint64_t, uint64_t, const char *) {
    CHECK(offset == 0 && bytes == raw_bytes && kind == CUDA_DERIVED_Q8_0_ALIGNED_DENSE);
    return artifact ? aligned_ptr : nullptr;
}
static void *cuda_norm_q8_lookup(const void *, uint64_t, uint64_t, size_t *) {
    return nullptr; /* GLM's plain RMSNorm entry has no producer Q8 emit. */
}
static void *cuda_tmp_alloc(uint64_t bytes, const char *) {
    void *ptr;
    CUDA(cudaMalloc(&ptr, bytes));
    scratch.push_back(ptr);
    scratch_bytes += bytes;
    return ptr;
}
/* Graph/cuBLAS fallbacks are outside this uncaptured default-MMQ fixture.
 * Any unexpected fallback is a failure, never a substitute arithmetic arm. */
static cudaStream_t ds4_cuda_moe_stream() { CHECK(false); return 0; }
static void ds4_cuda_moe_stream_sync_pre(cudaStream_t) {}
static void ds4_cuda_moe_stream_sync_post(cudaStream_t) {}
static void ds4_cuda_capture_warm_tmp_scratch() { CHECK(false); }
static dense_graph_entry *dense_graph_slot(const dense_graph_key *) { CHECK(false); return nullptr; }
static void cuda_graph_exec_destroy_noted(cudaGraphExec_t, uint64_t *) { CHECK(false); }
static cudaError_t cuda_graph_instantiate_noted(cudaGraphExec_t *, cudaGraph_t, uint64_t *) {
    CHECK(false); return cudaErrorNotSupported;
}
static int cuda_q8_label_is_attention_output_b(const char *) { return 0; }
static const float *cuda_q8_f32_ptr(const void *, uint64_t, uint64_t, uint64_t, uint64_t, const char *) {
    CHECK(false); return nullptr;
}
static const __half *cuda_q8_f16_ptr(const void *, uint64_t, uint64_t, uint64_t, uint64_t, const char *) {
    CHECK(false); return nullptr;
}
static void cuda_q8_f16_cache_disable_after_failure(const char *, uint64_t) { CHECK(false); }
static void cuda_cublas_ws_prep(cudaStream_t) { CHECK(false); }
static int cublas_ok(cublasStatus_t, const char *) { CHECK(false); return 0; }
#include "glm53_dense_prod.inc"

static ds4_gpu_tensor *alloc(uint64_t bytes, const void *values = nullptr) {
    auto *tensor = ds4_gpu_tensor_alloc(bytes);
    CHECK(tensor);
    if (values) { CHECK(ds4_gpu_tensor_write(tensor, 0, values, bytes)); }
    return tensor;
}
static float half_float(uint16_t bits) {
    __half value;
    std::memcpy(&value, &bits, 2);
    return __half2float(value);
}
static std::vector<float> read(ds4_gpu_tensor *tensor, uint64_t count) {
    std::vector<float> values(count + GUARD);
    CHECK(ds4_gpu_tensor_read(tensor, 0, values.data(), values.size() * sizeof(float)));
    for (uint64_t at = 0; at < count; ++at) { CHECK(std::isfinite(values[at])); }
    for (uint64_t at = count; at < values.size(); ++at) { CHECK(values[at] == SENTINEL); }
    values.resize(count);
    return values;
}
static void compare(const char *name, unsigned rows, const std::vector<float> &a,
                    const std::vector<float> &r) {
    double maximum = 0, peak = 0, signal = 0, error = 0;
    uint64_t changed = 0;
    for (size_t at = 0; at < a.size(); ++at) {
        const double delta = double(a[at]) - r[at];
        maximum = std::max(maximum, std::fabs(delta));
        peak = std::max(peak, std::fabs(double(r[at])));
        signal += double(r[at]) * r[at]; error += delta * delta;
        changed += std::memcmp(&a[at], &r[at], sizeof(float)) != 0;
    }
    std::printf("name=%s rows=%u aligned/raw max_abs=%.9g peak=%.9g rel_l2=%.9g changed=%llu/%llu\n",
        name, rows, maximum, peak, std::sqrt(error / std::max(signal, 1.0e-30)),
        (unsigned long long)changed, (unsigned long long)a.size());
}
/* Original source Q8 codes, independent CPU quantization and double dot.
 * Canonical MMVQ has half activation scales; MMQ/native use float scales.
 * Native K128 rounds to nearest-even, MMQ/canonical round away from zero. */
static double dot_ref(const unsigned char *w, const float *x, unsigned k,
                      unsigned rows, Arm arm) {
    double sum = 0;
    const bool native = k == 128 && (arm == Arm::Raw || rows == 1);
    for (unsigned b = 0; b < k / 32; ++b) {
        float maximum = 0;
        for (unsigned i = 0; i < 32; ++i) { maximum = std::max(maximum, std::fabs(x[b * 32 + i])); }
        const float divisor = maximum / 127.0f;
        const float inverse = native ? 1.0f / divisor : 127.0f / maximum;
        const float d = rows == 1 && k != 128
            ? __half2float(__float2half_rn(divisor)) : native ? divisor : 1.0f / inverse;
        uint16_t bits;
        std::memcpy(&bits, w + b * 34, 2);
        const auto *codes = (const int8_t *)(w + b * 34 + 2);
        int32_t integer = 0;
        for (unsigned i = 0; i < 32; ++i) {
            const float value = x[b * 32 + i];
            const float q = native ? std::nearbyint(value * inverse)
                : rows == 1 ? std::round(value / divisor) : std::round(value * inverse);
            integer += codes[i] * int(q);
        }
        sum += double(half_float(bits)) * d * integer;
    }
    return sum;
}
static bool cpu_samples(const char *name, const std::vector<unsigned char> &weights,
        const std::vector<float> &input, const std::vector<float> &raw,
        unsigned k, unsigned m, unsigned rows, Arm arm) {
    bool pass = true;
    for (unsigned token : {0u, rows - 1}) {
        for (unsigned row : {0u, 1u, m - 1}) {
            const double reference = dot_ref(weights.data() + uint64_t(row) * (k / 32) * 34,
                                             input.data() + uint64_t(token) * k, k, rows, arm);
            const double delta = std::fabs(raw[uint64_t(token) * m + row] - reference);
            std::printf("cpu name=%s arm=%s rows=%u token=%u out=%u reference=%.9g got=%.9g abs=%.9g\n",
                name, arm == Arm::Aligned ? "aligned" : "raw", rows, token, row,
                reference, raw[uint64_t(token) * m + row], delta);
            pass &= delta <= DOT_BOUND * std::max(1.0, std::fabs(reference));
        }
    }
    return pass;
}
template <class T> static T scalar(std::ifstream &file) {
    T value;
    CHECK(bool(file.read((char *)&value, sizeof(value))));
    return value;
}
int main(int argc, char **argv) {
    CHECK(argc == 2);
    std::ifstream file(argv[1], std::ios::binary);
    CHECK(file.good());
    char magic[8];
    CHECK(bool(file.read(magic, sizeof(magic))) && std::memcmp(magic, "GLMDQ8\0\1", 8) == 0);
    const unsigned count = scalar<uint32_t>(file);
    CHECK(count == 6);
    CHECK(ds4_gpu_init());
    const uint64_t census_before = ds4_gpu_mem_census_faults();
    const uint64_t gov_before = ds4_metrics_get()->memgov_faults;
    bool pass = true;
    for (unsigned tensor = 0; tensor < count; ++tensor) {
        const unsigned name_bytes = scalar<uint32_t>(file), k = scalar<uint32_t>(file);
        const unsigned m = scalar<uint32_t>(file), type = scalar<uint32_t>(file);
        raw_bytes = scalar<uint64_t>(file);
        CHECK(name_bytes < 128 && type == 8 && raw_bytes == uint64_t(k / 32) * m * 34);
        std::string name(name_bytes, '\0');
        CHECK(bool(file.read(name.data(), name.size())));
        CHECK((k == 128 && m == 8192) || (k == 8192 && m == 4096) ||
              (k == 4096 && m == 12288) || (k == 12288 && m == 4096));
        std::vector<unsigned char> weights(raw_bytes);
        CHECK(bool(file.read((char *)weights.data(), weights.size())));
        auto *raw_weight = alloc(raw_bytes, weights.data());
        const uint64_t blocks = raw_bytes / 34, scale_bytes = (blocks * 2 + 63) / 64 * 64;
        auto *aligned_weight = alloc(scale_bytes + blocks * 32);
        repack_q8_0_aligned_kernel<<<unsigned((blocks * 2 + 255) / 256), 256>>>(
            (__half *)aligned_weight->ptr, (unsigned char *)aligned_weight->ptr + scale_bytes,
            (const unsigned char *)raw_weight->ptr, blocks);
        CUDA(cudaGetLastError());
        std::vector<unsigned char> roundtrip(ds4_gpu_tensor_bytes(aligned_weight));
        CHECK(ds4_gpu_tensor_read(aligned_weight, 0, roundtrip.data(), roundtrip.size()));
        for (uint64_t block = 0; block < blocks; ++block) {
            CHECK(std::memcmp(roundtrip.data() + block * 2, weights.data() + block * 34, 2) == 0);
            CHECK(std::memcmp(roundtrip.data() + scale_bytes + block * 32,
                              weights.data() + block * 34 + 2, 32) == 0);
        }
        raw_ptr = (const char *)raw_weight->ptr; aligned_ptr = (const char *)aligned_weight->ptr;
        std::printf("name=%s type=8 K=%u M=%u raw_bytes=%llu aligned_bytes=%llu repack=EXACT\n",
            name.c_str(), k, m, (unsigned long long)raw_bytes,
            (unsigned long long)ds4_gpu_tensor_bytes(aligned_weight));
        std::vector<float> input(uint64_t(MAX_ROWS) * k);
        for (unsigned row = 0; row < MAX_ROWS; ++row) {
            for (unsigned col = 0; col < k; ++col) {
                input[uint64_t(row) * k + col] = 2 * std::sin(float(col) * 0.017f + row * 0.11f) +
                    0.31f * std::cos(float(col) * 0.071f - row * 0.13f);
            }
        }
        auto *x = alloc(input.size() * sizeof(float), input.data());
        auto *out = alloc((uint64_t(MAX_ROWS) * m + GUARD) * sizeof(float));
        for (unsigned rows : {1u, 17u, 128u}) {
            CHECK(ds4_gpu_tensor_fill_f32(out, SENTINEL, ds4_gpu_tensor_bytes(out) / sizeof(float)));
            artifact = false;
            CHECK(cuda_matmul_q8_0_tensor_labeled_impl(out, weights.data(), raw_bytes, 0, k, m, x, rows, name.c_str()));
            const auto raw = read(out, uint64_t(rows) * m);
            pass &= cpu_samples(name.c_str(), weights, input, raw, k, m, rows, Arm::Raw);
            CHECK(ds4_gpu_tensor_fill_f32(out, SENTINEL, ds4_gpu_tensor_bytes(out) / sizeof(float)));
            artifact = true;
            CHECK(cuda_matmul_q8_0_tensor_labeled_impl(out, weights.data(), raw_bytes, 0, k, m, x, rows, name.c_str()));
            const auto aligned = read(out, uint64_t(rows) * m);
            pass &= cpu_samples(name.c_str(), weights, input, aligned, k, m, rows, Arm::Aligned);
            compare(name.c_str(), rows, aligned, raw);
        }
        for (auto *p : scratch) { CUDA(cudaFree(p)); }
        scratch.clear();
        for (auto *p : {out, x, aligned_weight, raw_weight}) { ds4_gpu_tensor_free(p); }
        std::fflush(stdout);
    }
    CHECK(file.peek() == EOF);
    std::printf("faults census_before=%llu census_after=%llu memgov_before=%llu memgov_after=%llu "
                "private_scratch_total=%llu\n", (unsigned long long)census_before,
        (unsigned long long)ds4_gpu_mem_census_faults(), (unsigned long long)gov_before,
        (unsigned long long)ds4_metrics_get()->memgov_faults, (unsigned long long)scratch_bytes);
    CHECK(census_before == ds4_gpu_mem_census_faults() && gov_before == ds4_metrics_get()->memgov_faults);
    ds4_gpu_cleanup();
    std::puts(pass ? "GLM actual block0 dense: PASS" : "GLM actual block0 dense: FAIL");
    return pass ? 0 : 1;
}
