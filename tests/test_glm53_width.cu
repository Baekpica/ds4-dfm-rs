/* Actual layer-0 weights and prompt rows. No engine or large model mapping.
 * Compare both width contracts to scalar references before judging drift. */
#include "../ds4_gpu.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "weights.h"
#include "../cuda/mmq/ggml.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <limits>

void quantize_mmq_q8_1_cuda(const float *, const int32_t *, void *, ggml_type,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t);
void quantize_row_q8_1_cuda(const float *, const int32_t *, void *, ggml_type,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t);

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "width FAIL %d: %s\n", __LINE__, #x); std::exit(1); \
} } while (0)
enum { K = 4096, M = 8192, HC = 4, MIX = 24, BASE_ROWS = 128, GUARD = 16 };
static constexpr float SENTINEL = 12345.0f;
static constexpr double REF_ABS = 3.0e-5;
static constexpr double REF_REL = 2.0e-5;
static constexpr float RMS_EPS = 1.0e-5f;
static constexpr float HC_EPS = 1.0e-6f;
/* Fast reciprocal/division may place a value on the opposite side of a
 * half-integer. Away from that local rounding boundary, codes are exact. */
static constexpr float ROUND_ULPS = 8.0f;
static constexpr float ROUND_BOUND = ROUND_ULPS * std::numeric_limits<float>::epsilon() * 127.0f;
static unsigned failures;
struct MmqBlock { unsigned char scales[16]; int8_t codes[128]; };
struct VecBlock { unsigned char scales[4]; int8_t codes[32]; };
enum class QuantPath { Mmq, Vec };

static std::vector<MmqBlock> read_quant(const ds4_gpu_tensor *x, unsigned rows,
        ggml_type type, QuantPath path) {
    const uint64_t count = uint64_t(rows) * K / 128;
    auto *t = ds4_gpu_tensor_alloc((count + 64) * sizeof(MmqBlock));
    CHECK(t);
    const auto launch = path == QuantPath::Mmq ? quantize_mmq_q8_1_cuda : quantize_row_q8_1_cuda;
    launch(static_cast<const float *>(ds4_gpu_tensor_ptr(x)), nullptr,
        const_cast<void *>(ds4_gpu_tensor_ptr(t)), type, K, K, 0, 0, K, rows, 1, 1, nullptr);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<MmqBlock> result(count);
    if (path == QuantPath::Mmq) {
        CHECK(ds4_gpu_tensor_read(t, 0, result.data(), count * sizeof(MmqBlock)));
    } else {
        std::vector<VecBlock> raw(count * 4);
        CHECK(ds4_gpu_tensor_read(t, 0, raw.data(), count * sizeof(MmqBlock)));
        for (uint64_t b = 0; b < raw.size(); ++b) {
            __half h;
            std::memcpy(&h, raw[b].scales, sizeof(h));
            const float d = __half2float(h);
            auto &block = result[(b / (K / 32)) + (b % (K / 32) / 4) * rows];
            const unsigned lane = b % 4;
            std::memcpy(block.scales + lane * 4, &d, sizeof(d));
            std::copy_n(raw[b].codes, 32, block.codes + lane * 32);
        }
    }
    ds4_gpu_tensor_free(t);
    return result;
}

static float half_value(const unsigned char *p) {
    __half h;
    std::memcpy(&h, p, sizeof(h));
    return __half2float(h);
}

struct QBlock { int8_t q[32]; float d, s; };
enum class Scale { Half, Full };

static std::vector<QBlock> quant(const float *x, Scale scale) {
    std::vector<QBlock> blocks(K / 32);
    for (unsigned b = 0; b < blocks.size(); ++b) {
        float maximum = 0;
        for (unsigned i = 0; i < 32; ++i) {
            maximum = std::max(maximum, std::fabs(x[b * 32 + i]));
        }
        const float inverse = maximum ? 127.0f / maximum : 0.0f;
        const float d = scale == Scale::Full ? (maximum ? 1.0f / inverse : 0.0f) : maximum / 127.0f;
        blocks[b].d = scale == Scale::Half ? __half2float(__float2half_rn(d)) : d;
        float sums[8];
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

static double q4_ref(const unsigned char *w, const std::vector<QBlock> &x) {
    double total = 0;
    for (unsigned b = 0; b < K / 256; ++b) {
        const unsigned char *p = w + b * 144;
        const float d = half_value(p), dm = half_value(p + 2);
        const unsigned char *sc = p + 4, *qs = p + 16;
        for (unsigned g = 0; g < 8; ++g) {
            const unsigned scale = g < 4 ? sc[g] & 63u
                : (sc[g + 4] & 15u) | ((sc[g - 4] >> 6) << 4);
            const unsigned minimum = g < 4 ? sc[g + 4] & 63u
                : (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4);
            int dot = 0;
            for (unsigned i = 0; i < 32; ++i) {
                const unsigned code = (qs[(g / 2) * 32 + i] >> (g % 2 * 4)) & 15u;
                dot += int(code) * x[b * 8 + g].q[i];
            }
            const float wd = __half2float(__float2half_rn(d * scale));
            const float wm = __half2float(__float2half_rn(dm * minimum));
            total += double(wd) * x[b * 8 + g].d * dot - double(wm) * x[b * 8 + g].s;
        }
    }
    return total;
}

static double q8_ref(const unsigned char *w, const std::vector<QBlock> &x) {
    double total = 0;
    for (unsigned b = 0; b < K / 32; ++b) {
        const unsigned char *p = w + b * 34;
        int dot = 0;
        for (unsigned i = 0; i < 32; ++i) {
            dot += int(int8_t(p[2 + i])) * int(x[b].q[i]);
        }
        total += double(dot) * half_value(p) * x[b].d;
    }
    return total;
}

static void bf16_check(const unsigned char *weight, const std::vector<float> &x,
        const std::vector<float> &y, unsigned rows) {
    double peak = 0;
    for (unsigned token : {0u, rows - 1u}) {
        for (unsigned row : {0u, 1u, 5u, MIX - 1u}) {
            double expected = 0;
            for (unsigned col = 0; col < K * HC; ++col) {
                __nv_bfloat16 w;
                std::memcpy(&w, weight + (uint64_t(row) * K * HC + col) * sizeof(w), sizeof(w));
                const auto value = __float2bfloat16_rn(x[uint64_t(token) * K * HC + col]);
                expected += double(__bfloat162float(w)) * __bfloat162float(value);
            }
            const double delta = std::fabs(y[uint64_t(token) * MIX + row] - expected);
            peak = std::max(peak, delta);
            CHECK(delta <= REF_ABS + REF_REL * std::fabs(expected));
        }
    }
    std::printf("reference bf16 rows=%u max_abs=%.9g\n", rows, peak);
}

static ds4_gpu_tensor *alloc(uint64_t n) {
    auto *t = ds4_gpu_tensor_alloc((n + GUARD) * sizeof(float));
    CHECK(t && ds4_gpu_tensor_fill_f32(t, SENTINEL, n + GUARD));
    return t;
}

static std::vector<float> read(ds4_gpu_tensor *t, uint64_t n) {
    std::vector<float> result(n + GUARD);
    CHECK(ds4_gpu_tensor_read(t, 0, result.data(), result.size() * sizeof(float)));
    for (uint64_t i = 0; i < n; ++i) { CHECK(std::isfinite(result[i])); }
    for (uint64_t i = n; i < n + GUARD; ++i) { CHECK(result[i] == SENTINEL); }
    result.resize(n);
    return result;
}

static void save(const std::string &prefix, const char *tag, const std::vector<float> &x) {
    const auto path = prefix + "." + tag + ".f32";
    FILE *file = std::fopen(path.c_str(), "wb");
    CHECK(file && std::fwrite(x.data(), sizeof(float), x.size(), file) == x.size());
    CHECK(!std::fclose(file));
}

static void ref_check(const char *tag, const unsigned char *weight,
        const std::vector<float> &x, const std::vector<float> &y, unsigned rows,
        ggml_type kind, const std::vector<MmqBlock> &stored) {
    double peak = 0;
    unsigned failed = 0;
    for (unsigned token : {0u, rows - 1u}) {
        const auto input = quant(x.data() + uint64_t(token) * K,
            kind == GGML_TYPE_Q4_K || rows == 1 ? Scale::Half : Scale::Full);
        auto gpu_input = input;
        unsigned code_changes = 0, scale_changes = 0, sum_changes = 0;
        if (!stored.empty()) {
            for (unsigned b = 0; b < K / 32; ++b) {
                const auto &block = stored[uint64_t(b / 4) * rows + token];
                if (kind == GGML_TYPE_Q4_K) {
                    gpu_input[b].d = half_value(block.scales + b % 4 * 4);
                    gpu_input[b].s = half_value(block.scales + b % 4 * 4 + 2);
                } else {
                    std::memcpy(&gpu_input[b].d, block.scales + b % 4 * 4, sizeof(float));
                }
                for (unsigned c = 0; c < 32; ++c) {
                    const int8_t code = block.codes[(b % 4) * 32 + c];
                    if (code != input[b].q[c]) {
                        CHECK(std::abs(int(code) - int(input[b].q[c])) == 1);
                        const float *start = x.data() + uint64_t(token) * K + b * 32;
                        float maximum = 0.0f;
                        for (unsigned j = 0; j < 32; ++j) {
                            maximum = std::max(maximum, std::fabs(start[j]));
                        }
                        const double scaled = double(start[c]) * 127.0 / maximum;
                        const double tie = 0.5 * (int(code) + int(input[b].q[c]));
                        CHECK(std::fabs(scaled - tie) <= ROUND_BOUND);
                    }
                    code_changes += code != input[b].q[c];
                    gpu_input[b].q[c] = code;
                }
                if (kind == GGML_TYPE_Q4_K) {
                    CHECK(gpu_input[b].d == input[b].d && gpu_input[b].s == input[b].s);
                } else {
                    CHECK(std::fabs(gpu_input[b].d - input[b].d) <=
                        ROUND_ULPS * std::numeric_limits<float>::epsilon() * std::fabs(input[b].d));
                }
                scale_changes += gpu_input[b].d != input[b].d;
                sum_changes += kind == GGML_TYPE_Q4_K && gpu_input[b].s != input[b].s;
            }
            std::printf("quant %s rows=%u token=%u code_changes=%u scale_changes=%u sum_changes=%u\n",
                tag, rows, token, code_changes, scale_changes, sum_changes);
        }
        for (unsigned row : {0u, 1u, 2u, 127u, 128u, M - 1u}) {
            const uint64_t row_bytes = kind == GGML_TYPE_Q4_K ? K / 256 * 144 : K / 32 * 34;
            const double expected = kind == GGML_TYPE_Q4_K ? q4_ref(weight + row * row_bytes, gpu_input)
                : q8_ref(weight + row * row_bytes, gpu_input);
            const double delta = std::fabs(y[uint64_t(token) * M + row] - expected);
            peak = std::max(peak, delta);
            if (delta > REF_ABS + REF_REL * std::fabs(expected)) {
                failed++;
                std::fprintf(stderr, "reference RED %s rows=%u token=%u row=%u got=%.9g reference=%.9g abs=%.9g\n",
                    tag, rows, token, row, y[uint64_t(token) * M + row], expected, delta);
            }
        }
    }
    std::printf("reference %s rows=%u max_abs=%.9g failed=%u\n", tag, rows, peak, failed);
    failures += failed;
}

int main(int argc, char **argv) {
    CHECK(argc == 5);
    const unsigned rows = unsigned(std::strtoul(argv[3], nullptr, 10));
    CHECK(rows == 1 || rows == 128 || rows == 256 || rows == 512 || rows == 1024 || rows == 2048);
    const int fd = open(argv[1], O_RDONLY);
    struct stat stat;
    CHECK(fd >= 0 && !fstat(fd, &stat));
    const uint64_t bytes = stat.st_size;
    CHECK(bytes == W_KDA_V + uint64_t(K / 32 * 34) * M);
    const auto *w = static_cast<const unsigned char *>(mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0));
    CHECK(w != MAP_FAILED);
    FILE *file = std::fopen(argv[2], "rb");
    std::vector<float> base(BASE_ROWS * K);
    CHECK(file && std::fread(base.data(), sizeof(float), base.size(), file) == base.size());
    CHECK(!std::fclose(file));
    std::vector<float> fixed_norm;
    if (const char *path = std::getenv("DS4_WIDTH_FIXED_NORM")) {
        file = std::fopen(path, "rb");
        fixed_norm.resize(BASE_ROWS * K);
        CHECK(file && std::fread(fixed_norm.data(), sizeof(float), fixed_norm.size(), file) == fixed_norm.size());
        CHECK(!std::fclose(file));
    }
    CHECK(!setenv("DS4_CUDA_COPY_MODEL", "1", 1));
    CHECK(!setenv("DS4_CUDA_NO_Q8_ALIGNED", "1", 1));
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(w, bytes));
    auto *embed = alloc(uint64_t(rows) * K);
    auto *hc_owner = alloc(uint64_t(rows) * K * HC);
    auto *flat_owner = alloc(uint64_t(rows) * K * HC);
    auto *mix_owner = alloc(uint64_t(rows) * MIX);
    auto *split_owner = alloc(uint64_t(rows) * MIX);
    auto *cur_owner = alloc(uint64_t(rows) * K);
    auto *norm_owner = alloc(uint64_t(rows) * K);
    auto *q = alloc(uint64_t(rows) * M);
    auto *k = alloc(uint64_t(rows) * M);
    auto *v = alloc(uint64_t(rows) * M);
    auto *hc = ds4_gpu_tensor_view(hc_owner, 0, uint64_t(rows) * K * HC * sizeof(float));
    auto *flat = ds4_gpu_tensor_view(flat_owner, 0, uint64_t(rows) * K * HC * sizeof(float));
    auto *mix = ds4_gpu_tensor_view(mix_owner, 0, uint64_t(rows) * MIX * sizeof(float));
    auto *split = ds4_gpu_tensor_view(split_owner, 0, uint64_t(rows) * MIX * sizeof(float));
    auto *cur = ds4_gpu_tensor_view(cur_owner, 0, uint64_t(rows) * K * sizeof(float));
    auto *norm = ds4_gpu_tensor_view(norm_owner, 0, uint64_t(rows) * K * sizeof(float));
    CHECK(hc && flat && mix && split && cur && norm);
    const unsigned calls = rows == 1 ? BASE_ROWS : 1;
    std::vector<float> all_norm, all_q, all_k, all_v, all_mix, all_flat, all_cur;
    for (unsigned call = 0; call < calls; ++call) {
        std::vector<float> input(uint64_t(rows) * K);
        for (unsigned row = 0; row < rows; ++row) {
            const unsigned source = rows == 1 ? call : row % BASE_ROWS;
            std::copy_n(base.data() + source * K, K, input.data() + uint64_t(row) * K);
        }
        CHECK(ds4_gpu_tensor_write(embed, 0, input.data(), input.size() * sizeof(float)));
        CHECK(ds4_gpu_repeat_hc_rows_tensor(hc, embed, K, HC, rows));
        CHECK(ds4_gpu_rms_norm_plain_rows_tensor(flat, hc, K * HC, rows, RMS_EPS));
        CHECK(ds4_gpu_matmul_bf16_tensor(mix, w, bytes, W_HC_ATTN_FN, K * HC, MIX, flat, rows));
        CHECK(ds4_gpu_hc_split_weighted_sum_tensor(cur, split, mix, hc, w, bytes,
            W_HC_ATTN_SCALE, W_HC_ATTN_BASE, K, HC, 20, HC_EPS));
        CHECK(ds4_gpu_rms_norm_weight_rows_tensor(norm, cur, w, bytes, W_ATTN_NORM, K, rows, RMS_EPS));
        if (!fixed_norm.empty()) {
            for (unsigned row = 0; row < rows; ++row) {
                const unsigned source = rows == 1 ? call : row % BASE_ROWS;
                std::copy_n(fixed_norm.data() + source * K, K, input.data() + uint64_t(row) * K);
            }
            CHECK(ds4_gpu_tensor_write(norm, 0, input.data(), input.size() * sizeof(float)));
        }
        CHECK(ds4_gpu_matmul_q4_K_tensor(q, w, bytes, W_KDA_Q, K, M, norm, rows));
        CHECK(ds4_gpu_matmul_q4_K_tensor(k, w, bytes, W_KDA_K, K, M, norm, rows));
        CHECK(ds4_gpu_matmul_q8_0_tensor(v, w, bytes, W_KDA_V, K, M, norm, rows));
        auto xn = read(norm_owner, uint64_t(rows) * K);
        auto yq = read(q, uint64_t(rows) * M);
        auto yk = read(k, uint64_t(rows) * M);
        auto yv = read(v, uint64_t(rows) * M);
        auto xf = read(flat_owner, uint64_t(rows) * K * HC);
        auto ym = read(mix_owner, uint64_t(rows) * MIX);
        if (!call || call == calls - 1) {
            bf16_check(w + W_HC_ATTN_FN, xf, ym, rows);
            const auto q4 = read_quant(norm, rows, GGML_TYPE_Q4_K, QuantPath::Mmq);
            const auto q8 = read_quant(norm, rows, GGML_TYPE_Q8_0,
                rows == 1 ? QuantPath::Vec : QuantPath::Mmq);
            ref_check("q", w + W_KDA_Q, xn, yq, rows, GGML_TYPE_Q4_K, q4);
            ref_check("k", w + W_KDA_K, xn, yk, rows, GGML_TYPE_Q4_K, q4);
            ref_check("v", w + W_KDA_V, xn, yv, rows, GGML_TYPE_Q8_0, q8);
        }
        auto append = [](std::vector<float> &dest, const std::vector<float> &src) {
            dest.insert(dest.end(), src.begin(), src.end());
        };
        append(all_norm, xn); append(all_q, yq); append(all_k, yk); append(all_v, yv);
        append(all_mix, ym);
        append(all_flat, xf);
        append(all_cur, read(cur_owner, uint64_t(rows) * K));
        read(embed, uint64_t(rows) * K);
        read(hc_owner, uint64_t(rows) * K * HC);
        read(split_owner, uint64_t(rows) * MIX);
    }
    const std::string prefix = argv[4];
    save(prefix, "flat", all_flat); save(prefix, "mix", all_mix);
    save(prefix, "cur", all_cur); save(prefix, "norm", all_norm);
    save(prefix, "q", all_q); save(prefix, "k", all_k); save(prefix, "v", all_v);
    std::printf("width rows=%u complete=1 weights=%llu\n", rows, (unsigned long long)bytes);
    for (auto *t : {hc, flat, mix, split, cur, norm}) { ds4_gpu_tensor_free(t); }
    for (auto *t : {embed, hc_owner, flat_owner, mix_owner, split_owner, cur_owner,
            norm_owner, q, k, v}) { ds4_gpu_tensor_free(t); }
    ds4_gpu_unregister_model_map(w);
    ds4_gpu_cleanup();
    CHECK(!munmap(const_cast<unsigned char *>(w), bytes) && !close(fd));
    return failures ? 1 : 0;
}
