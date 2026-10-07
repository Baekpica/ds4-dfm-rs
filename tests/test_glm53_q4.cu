/* Same Q4_K MMQ backend and actual layer17 weights, with finite independent
 * inputs. Shape churn changes allocator history; no engine/model is opened. */
#include "../ds4_gpu.h"
#include "../cuda/mmq/ds4_mmq.h"
extern "C" {
#include "../ds4.h"
}
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
    std::fprintf(stderr, "GLM Q4 FAIL %d: %s\n", __LINE__, #x); std::exit(1); \
} } while (0)
#define CUDA(x) CHECK((x) == cudaSuccess)
enum { K = 4096, M = 8192, ROWS = 128, GUARD = 16, CYCLES = 64,
       HEADS = 64, DIM = 128, CONV = 4, Q4_BLOCK = 144,
       CHURN_M = 256, CHURN_K = 256, CHURN_EXPERTS = 16, USED = 8 };
static constexpr float SENTINEL = 12345.0f;
static constexpr double DOT_BOUND = 1.0e-4;
enum class Lane { ReadEach, Queued };

static void save_out(const char *prefix, const char *tag,
        const std::vector<float> &values) {
    if (!prefix) { return; }
    char path[4096];
    const int n = std::snprintf(path, sizeof(path), "%s.%s.f32", prefix, tag);
    CHECK(n > 0 && n < int(sizeof(path)));
    FILE *fp = std::fopen(path, "wb");
    CHECK(fp && std::fwrite(values.data(), sizeof(float), values.size(), fp) == values.size());
    CHECK(!std::fclose(fp));
}

static ds4_gpu_tensor *alloc(uint64_t bytes, const void *values = nullptr) {
    ds4_gpu_tensor *p = ds4_gpu_tensor_alloc(bytes);
    CHECK(p);
    if (values) { CHECK(ds4_gpu_tensor_write(p, 0, values, bytes)); }
    return p;
}

static float half_float(const unsigned char *p) {
    __half value;
    std::memcpy(&value, p, sizeof(value));
    return __half2float(value);
}

/* Mirror the independent 32-value quantization algebra. The float4 then
 * XOR reduction order matters before the Q4 offset correction's half sum. */
struct InputBlock { int8_t q[32]; float d, s; };
static std::vector<InputBlock> quant_ref(const float *x) {
    std::vector<InputBlock> blocks(K / 32);
    for (unsigned b = 0; b < blocks.size(); ++b) {
        float maximum = 0;
        float sums[8];
        for (unsigned i = 0; i < 32; ++i) {
            maximum = std::max(maximum, std::fabs(x[b * 32 + i]));
        }
        const float inverse = 127.0f / maximum;
        blocks[b].d = __half2float(__float2half_rn(1.0f / inverse));
        for (unsigned i = 0; i < 32; ++i) {
            blocks[b].q[i] = int8_t(std::round(x[b * 32 + i] * inverse));
        }
        for (unsigned i = 0; i < 8; ++i) {
            const float *v = x + b * 32 + i * 4;
            sums[i] = ((v[0] + v[1]) + v[2]) + v[3];
        }
        for (unsigned offset : {4u, 2u, 1u}) {
            float next[8];
            for (unsigned i = 0; i < 8; ++i) { next[i] = sums[i] + sums[i ^ offset]; }
            std::copy(next, next + 8, sums);
        }
        blocks[b].s = __half2float(__float2half_rn(sums[0]));
    }
    return blocks;
}

static double dot_ref(const unsigned char *w, const std::vector<InputBlock> &x) {
    double total = 0;
    for (unsigned b = 0; b < K / 256; ++b) {
        const unsigned char *p = w + b * Q4_BLOCK;
        const float d = half_float(p), dm = half_float(p + 2);
        const unsigned char *sc = p + 4, *qs = p + 16;
        for (unsigned g = 0; g < 8; ++g) {
            const unsigned scale = g < 4 ? sc[g] & 63u
                : (sc[g + 4] & 15u) | ((sc[g - 4] >> 6) << 4);
            const unsigned minimum = g < 4 ? sc[g + 4] & 63u
                : (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4);
            int sum = 0;
            for (unsigned i = 0; i < 32; ++i) {
                const unsigned code = (qs[(g / 2) * 32 + i] >> (g % 2 * 4)) & 15u;
                sum += int(code) * x[b * 8 + g].q[i];
            }
            /* The production MMA loader expands d*scale and dmin*minimum
             * into half2 before the dot, which differs from F32 dequant. */
            const float wd = __half2float(__float2half_rn(d * scale));
            const float wm = __half2float(__float2half_rn(dm * minimum));
            total += double(wd) * x[b * 8 + g].d * sum
                - double(wm) * x[b * 8 + g].s;
        }
    }
    return total;
}

static std::vector<float> read_out(ds4_gpu_tensor *out, unsigned rows) {
    std::vector<float> values(uint64_t(rows) * M + GUARD);
    CHECK(ds4_gpu_tensor_read(out, 0, values.data(), values.size() * sizeof(float)));
    for (uint64_t i = 0; i < uint64_t(rows) * M; ++i) { CHECK(std::isfinite(values[i])); }
    for (uint64_t i = uint64_t(rows) * M; i < values.size(); ++i) { CHECK(values[i] == SENTINEL); }
    values.resize(uint64_t(rows) * M);
    return values;
}

static void cpu_check(const unsigned char *weight, const std::vector<float> &input,
        const std::vector<float> &out, unsigned rows, double *peak) {
    for (unsigned token : {0u, rows - 1}) {
        const auto q = quant_ref(input.data() + uint64_t(token) * K);
        for (unsigned row : {0u, 1u, 2u, 127u, 128u, M - 1u}) {
            const double expected = dot_ref(weight + uint64_t(row) * (K / 256) * Q4_BLOCK, q);
            const double delta = std::fabs(out[uint64_t(token) * M + row] - expected);
            *peak = std::max(*peak, delta);
            if (delta > DOT_BOUND * std::max(1.0, std::fabs(expected))) {
                std::fprintf(stderr, "Q4 CPU mismatch rows=%u token=%u row=%u got=%.9g expected=%.9g abs=%.9g\n",
                    rows, token, row, out[uint64_t(token) * M + row], expected, delta);
                CHECK(false);
            }
        }
    }
}

static void same(const std::vector<float> &a, const std::vector<float> &b,
        unsigned cycle, const char *stage) {
    CHECK(a.size() == b.size());
    if (!std::memcmp(a.data(), b.data(), a.size() * sizeof(float))) { return; }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float))) {
            std::fprintf(stderr, "Q4 repeat RED cycle=%u stage=%s row=%zu baseline=%.9g got=%.9g\n",
                cycle, stage, i, a[i], b[i]);
            CHECK(false);
        }
    }
}

int main(int argc, char **argv) {
    CHECK(argc == 2 || argc == 4);
    const Lane lane = argc == 4 && !std::strcmp(argv[2], "queued")
        ? Lane::Queued : Lane::ReadEach;
    CHECK(argc == 2 || lane == Lane::Queued || !std::strcmp(argv[2], "sync"));
    const char *prefix = argc == 4 ? argv[3] : nullptr;
    const uint64_t weight_bytes = uint64_t(M) * (K / 256) * Q4_BLOCK;
    const int fd = open(argv[1], O_RDONLY);
    CHECK(fd >= 0);
    struct stat st;
    CHECK(!fstat(fd, &st) && uint64_t(st.st_size) == weight_bytes);
    const auto *weights = static_cast<const unsigned char *>(
        mmap(nullptr, weight_bytes, PROT_READ, MAP_PRIVATE, fd, 0));
    CHECK(weights != MAP_FAILED);
    /* This explicit small copy isolates the kernel from huge-map HMM and
     * expert streaming. It contains exactly the real query tensor. */
    CHECK(!setenv("DS4_CUDA_COPY_MODEL", "1", 1));
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(weights, weight_bytes));
    const uint64_t census = ds4_gpu_mem_census_faults();
    const uint64_t gov = ds4_metrics_get()->memgov_faults;
    std::vector<float> input(uint64_t(ROWS) * K + GUARD, SENTINEL);
    for (unsigned row = 0; row < ROWS; ++row) {
        for (unsigned col = 0; col < K; ++col) {
            input[uint64_t(row) * K + col] = 2.0f * std::sin(float(col) * 0.017f + row * 0.11f)
                + 0.31f * std::cos(float(col) * 0.071f - row * 0.13f);
        }
    }
    auto *x = alloc(input.size() * sizeof(float), input.data());
    auto *out = alloc((uint64_t(ROWS) * M + GUARD) * sizeof(float));
    /* One zero Q8 span can serve the real KDA f_a/f_b/g_a/g_b shapes. */
    std::vector<unsigned char> zero_q8(uint64_t(M) * K / 32 * 34, 0);
    std::vector<unsigned char> zero_iq(uint64_t(CHURN_EXPERTS) * CHURN_M * CHURN_K / 256 * 66, 0);
    CHECK(ds4_gpu_set_aux_model_map_range(zero_q8.data(), zero_q8.size(), 0, zero_q8.size()));
    auto *q8 = alloc(uint64_t(CHURN_M) * CHURN_K / 32 * 34, zero_q8.data());
    auto *iq = alloc(zero_iq.size(), zero_iq.data());
    auto *churn_out = alloc(uint64_t(USED) * CHURN_M * sizeof(float));
    auto *k_raw = alloc(uint64_t(M) * sizeof(float));
    auto *v_raw = alloc(uint64_t(M) * sizeof(float));
    auto *g_raw = alloc(uint64_t(M) * sizeof(float));
    auto *rank = alloc(128u * sizeof(float));
    const int32_t selected[USED] = {15, 0, 8, 1, 14, 3, 7, 9};
    auto *ids = alloc(sizeof(selected), selected);
    /* KDA carry isolates the newest raw Q write from projection arithmetic. */
    const uint64_t vector = uint64_t(HEADS) * DIM;
    const uint64_t carry = vector * CONV;
    auto *state = alloc(vector * DIM * sizeof(float));
    auto *qc = alloc((carry + GUARD) * sizeof(float));
    auto *kc = alloc(carry * sizeof(float));
    auto *vc = alloc(carry * sizeof(float));
    auto *zero = alloc(vector * sizeof(float));
    auto *cw = alloc(carry * sizeof(float));
    auto *beta = alloc(HEADS * sizeof(float));
    auto *decay = alloc(HEADS * sizeof(float));
    auto *carry_out = alloc(vector * sizeof(float));
    for (auto *p : {zero, cw, beta, decay}) {
        CHECK(ds4_gpu_tensor_fill_f32(p, 0.0f, ds4_gpu_tensor_bytes(p) / sizeof(float)));
    }
    CHECK(ds4_gpu_tensor_write(qc, carry * sizeof(float), input.data(), GUARD * sizeof(float)));
    double peak = 0;
    auto project = [&](unsigned rows) {
        CHECK(ds4_gpu_tensor_fill_f32(out, SENTINEL, ds4_gpu_tensor_bytes(out) / sizeof(float)));
        CHECK(ds4_gpu_matmul_q4_K_tensor(out, weights, weight_bytes, 0, K, M, x, rows));
        auto values = read_out(out, rows);
        cpu_check(weights, input, values, rows, &peak);
        return values;
    };
    const auto baseline = project(1);
    save_out(prefix, "n1", baseline);
    for (unsigned cycle = 0; cycle < CYCLES; ++cycle) {
        if (lane == Lane::ReadEach) {
            same(baseline, project(1), cycle, "initial1");
            const auto wide = project(ROWS);
            if (!cycle) { save_out(prefix, "n128", wide); }
            same(baseline, project(1), cycle, "after128");
        } else {
            CHECK(ds4_gpu_matmul_q4_K_tensor(out, weights, weight_bytes, 0, K, M, x, ROWS));
        }
        /* Reduced predictor-like Q8/IQ2 calls perturb the SAME MMQ pool;
         * their synthetic weights and reduced shapes are disclosed. */
        CHECK(ds4_mmq_q8_0_dense_vec(ds4_gpu_tensor_ptr(q8),
            static_cast<const float *>(ds4_gpu_tensor_ptr(x)),
            static_cast<float *>(const_cast<void *>(ds4_gpu_tensor_ptr(churn_out))), CHURN_M, 1, CHURN_K, 0) == 0);
        CHECK(ds4_mmq_iq2_xxs_moe_vec(ds4_gpu_tensor_ptr(iq),
            static_cast<const float *>(ds4_gpu_tensor_ptr(x)),
            static_cast<const int32_t *>(ds4_gpu_tensor_ptr(ids)),
            static_cast<float *>(const_cast<void *>(ds4_gpu_tensor_ptr(churn_out))),
            CHURN_M, CHURN_K, 1, CHURN_EXPERTS, USED, 0) == 0);
        for (auto *p : {state, kc, vc}) {
            CHECK(ds4_gpu_tensor_fill_f32(p, 0.0f, ds4_gpu_tensor_bytes(p) / sizeof(float)));
        }
        CHECK(ds4_gpu_tensor_fill_f32(qc, 0.0f, carry));
        std::vector<float> after;
        if (lane == Lane::ReadEach) {
            after = project(1);
            same(baseline, after, cycle, "after_churn");
        } else {
            /* Queue the raw-Q producer, other projections and carry consumer
             * without any intervening readback/copy/synchronize. */
            CHECK(ds4_gpu_tensor_fill_f32(out, SENTINEL, ds4_gpu_tensor_bytes(out) / sizeof(float)));
            CHECK(ds4_gpu_matmul_q4_K_tensor(out, weights, weight_bytes, 0, K, M, x, 1));
            CHECK(ds4_gpu_matmul_q4_K_tensor(k_raw, weights, weight_bytes, 0, K, M, x, 1));
            auto q8_project = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src,
                    unsigned in, unsigned rows) {
                CHECK(ds4_gpu_matmul_q8_0_tensor(dst, zero_q8.data(), zero_q8.size(),
                    0, in, rows, src, 1));
            };
            q8_project(v_raw, x, K, M);
            q8_project(rank, x, K, 128);
            q8_project(g_raw, rank, 128, M);
            q8_project(beta, x, K, HEADS);
            q8_project(rank, x, K, 128);
            q8_project(g_raw, rank, 128, M);
        }
        CHECK(ds4_gpu_glm53_kda_decode_tensor(carry_out, state, qc, kc, vc,
            out, lane == Lane::Queued ? k_raw : zero,
            lane == Lane::Queued ? v_raw : zero, lane == Lane::Queued ? g_raw : zero,
            beta, cw, cw, cw, decay, zero, HEADS, DIM, CONV, -5.0f));
        std::vector<float> got(carry + GUARD);
        CHECK(ds4_gpu_tensor_read(qc, 0, got.data(), got.size() * sizeof(float)));
        if (lane == Lane::Queued) {
            after = read_out(out, 1);
            same(baseline, after, cycle, "queued_end");
            cpu_check(weights, input, after, 1, &peak);
        }
        for (uint64_t channel = 0; channel < vector; ++channel) {
            CHECK(got[channel * CONV] == 0.0f && got[channel * CONV + 1] == 0.0f && got[channel * CONV + 2] == 0.0f);
            CHECK(!std::memcmp(&got[channel * CONV + 3], &after[channel], sizeof(float)));
        }
        CHECK(!std::memcmp(got.data() + carry, input.data(), GUARD * sizeof(float)));
        if (cycle % 8 == 0) {
            std::printf("cycle=%u q[2]=%.9g carry[11]=%.9g repeat=EXACT cpu_max_abs=%.9g\n",
                cycle, after[2], got[11], peak);
            std::fflush(stdout);
        }
    }
    std::vector<float> check_input(input.size());
    CHECK(ds4_gpu_tensor_read(x, 0, check_input.data(), check_input.size() * sizeof(float)));
    CHECK(!std::memcmp(input.data(), check_input.data(), input.size() * sizeof(float)));
    CHECK(census == ds4_gpu_mem_census_faults() && gov == ds4_metrics_get()->memgov_faults);
    for (auto *p : {carry_out, decay, beta, cw, zero, vc, kc, qc, state, rank, g_raw, v_raw, k_raw,
                   ids, churn_out, iq, q8, out, x}) { ds4_gpu_tensor_free(p); }
    ds4_gpu_unregister_model_map(zero_q8.data());
    ds4_gpu_unregister_model_map(weights);
    CHECK(!munmap(const_cast<unsigned char *>(weights), weight_bytes));
    CHECK(!close(fd));
    std::printf("GLM actual layer17 Q4: PASS lane=%s cycles=%u cpu_max_abs=%.9g faults=0 input=EXACT\n",
        lane == Lane::Queued ? "queued" : "read_each", CYCLES, peak);
    ds4_gpu_cleanup();
    return 0;
}
