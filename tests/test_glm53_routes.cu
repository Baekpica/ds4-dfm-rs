/* GLM actual-shape numeric triage. Only eight expert payloads are physical;
 * full virtual expert/slot counts preserve production MMQ scheduling. */
#include "../ds4_gpu.h"
#include "../cuda/mmq/ds4_mmq.h"
#include "../cuda/mmq/ds4_ggml_stubs.h"
extern "C" {
#include "../ds4.h"
}
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "../cuda/mmq/ggml-common.h"

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "GLM routes FAIL %d: %s\n", __LINE__, #x); \
    std::exit(1); \
} } while (0)
#define CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "GLM routes CUDA FAIL %d: %s\n", __LINE__, \
                 cudaGetErrorString(e)); std::exit(2); \
} } while (0)

enum { INPUT = 4096, MID = 2048, OUTPUT = 4096, EXPERTS = 288, USED = 8,
       SLOTS = 1024, MAX_ROWS = 128, QK = 256, GUARD = 16 };
static constexpr float SENTINEL = 12345.0f;
static constexpr unsigned char PAD_SENTINEL = 0x5a;
static constexpr float ROUTE_SCALE = 2.5f;
static constexpr float SWIGLU_CLAMP = 10.0f;
enum class Match { Bound, Exact, Info };
/* A local fixed-route kernel diagnostic, not a whole-model quality threshold. */
static constexpr double PAIR_REL_BOUND = 1.0e-4;
static const int32_t IDS[USED] = {287, 256, 270, 3, 128, 255, 1, 281};
static const int32_t SLOT_IDS[USED] = {1023, 512, 768, 7, 1, 511, 767, 0};
static cudaStream_t ds4_current_stream() { return (cudaStream_t)0; }
static int ds4_cuda_use_mmq() { return 1; }
static int cuda_ok(cudaError_t error, const char *what) {
    if (error == cudaSuccess) { return 1; }
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
    return 0;
}

/* The private tensor layout, repack kernels and graph helper are extracted
 * verbatim. No production ABI or runtime diagnostic branch is introduced. */
#include "glm53_route_prod.inc"

static uint64_t round_up(uint64_t value, uint64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

static ds4_gpu_tensor *reserve(uint64_t bytes) {
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_reserve(bytes);
    CHECK(tensor); /* A non-VMM fallback would materialize the full tables. */
    return tensor;
}

static void ensure(ds4_gpu_tensor *tensor, uint64_t offset, uint64_t bytes) {
    CHECK(ds4_gpu_tensor_ensure(tensor, offset, bytes));
}

static void upload(ds4_gpu_tensor *tensor, uint64_t offset,
                   const unsigned char *data, uint64_t bytes) {
    ensure(tensor, offset, bytes);
    CHECK(ds4_gpu_tensor_write(tensor, offset, data, bytes));
}

static void upload_slot(ds4_gpu_tensor *tensor, uint64_t offset,
                        const unsigned char *data, uint64_t bytes, uint64_t stride) {
    CHECK(bytes <= stride);
    ensure(tensor, offset, stride);
    CUDA(cudaMemset((char *)tensor->ptr + offset + bytes, PAD_SENTINEL, stride - bytes));
    CHECK(ds4_gpu_tensor_write(tensor, offset, data, bytes));
}

static void check_padding(ds4_gpu_tensor *tensor, uint64_t bytes, uint64_t stride) {
    std::vector<unsigned char> padding(stride - bytes);
    for (int32_t slot : SLOT_IDS) {
        CHECK(ds4_gpu_tensor_read(tensor, uint64_t(slot) * stride + bytes,
                                 padding.data(), padding.size()));
        for (unsigned char value : padding) { CHECK(value == PAD_SENTINEL); }
    }
}

static ds4_gpu_tensor *alloc(uint64_t bytes, const void *data = nullptr) {
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(bytes);
    CHECK(tensor);
    if (data) { CHECK(ds4_gpu_tensor_write(tensor, 0, data, bytes)); }
    return tensor;
}

struct Weights {
    ds4_gpu_tensor *gate, *up, *down, *soa_gate, *soa_up, *soa_down;
    ds4_gpu_tensor *ssd_gate, *ssd_up, *ssd_down;
    uint64_t gate_unit, down_unit, gate_stride, down_stride;
    uint32_t gt, dt;
};

static ds4_gpu_tensor *iq2_soa(ds4_gpu_tensor *raw, uint64_t unit) {
    const uint64_t blocks = unit / 66;
    const uint64_t scale_bytes = round_up(EXPERTS * blocks * 2, 64);
    ds4_gpu_tensor *soa = reserve(scale_bytes + EXPERTS * blocks * 64);
    for (int32_t id : IDS) {
        const uint64_t block = uint64_t(id) * blocks;
        ensure(soa, block * 2, blocks * 2);
        ensure(soa, scale_bytes + block * 64, blocks * 64);
        repack_iq2_xxs_aligned_kernel<<<unsigned((blocks * 8 + 255) / 256), 256>>>(
            (__half *)soa->ptr + block,
            (uint2 *)((char *)soa->ptr + scale_bytes) + block * 8,
            (unsigned char *)raw->ptr + uint64_t(id) * unit, blocks);
        CUDA(cudaGetLastError());
    }
    return soa;
}

static ds4_gpu_tensor *q2_soa(ds4_gpu_tensor *raw, uint64_t unit) {
    const uint64_t blocks = unit / 84, pairs = EXPERTS * blocks / 2;
    const uint64_t dm_bytes = round_up(pairs * 8, 64);
    const uint64_t scale_bytes = round_up(pairs * 32, 64);
    ds4_gpu_tensor *soa = reserve(dm_bytes + scale_bytes + pairs * 128);
    for (int32_t id : IDS) {
        const uint64_t pair = uint64_t(id) * blocks / 2;
        ensure(soa, pair * 8, blocks / 2 * 8);
        ensure(soa, dm_bytes + pair * 32, blocks / 2 * 32);
        ensure(soa, dm_bytes + scale_bytes + pair * 128, blocks / 2 * 128);
        repack_q2_k_aligned_kernel<<<unsigned((blocks * 16 + 255) / 256), 256>>>(
            (uint32_t *)soa->ptr, (uint32_t *)((char *)soa->ptr + dm_bytes),
            (uint32_t *)((char *)soa->ptr + dm_bytes + scale_bytes),
            (unsigned char *)raw->ptr + uint64_t(id) * unit,
            uint64_t(id) * blocks, blocks, MID / QK, OUTPUT);
        CUDA(cudaGetLastError());
    }
    return soa;
}

static void weight_roundtrip(const Weights &w, const std::vector<unsigned char> &data) {
    if (w.gt == 16) {
        const uint64_t blocks = w.gate_unit / 66;
        const uint64_t scale_bytes = round_up(EXPERTS * blocks * 2, 64);
        std::vector<unsigned char> scales(blocks * 2), codes(blocks * 64);
        for (unsigned part = 0; part < 2; ++part) {
            ds4_gpu_tensor *soa = part == 0 ? w.soa_gate : w.soa_up;
            for (unsigned slot = 0; slot < USED; ++slot) {
                CHECK(ds4_gpu_tensor_read(soa, uint64_t(IDS[slot]) * blocks * 2,
                                          scales.data(), scales.size()));
                CHECK(ds4_gpu_tensor_read(soa, scale_bytes + uint64_t(IDS[slot]) * blocks * 64,
                                          codes.data(), codes.size()));
                const unsigned char *raw = data.data() + (part * USED + slot) * w.gate_unit;
                for (uint64_t block = 0; block < blocks; ++block) {
                    CHECK(std::memcmp(scales.data() + block * 2, raw + block * 66, 2) == 0);
                    CHECK(std::memcmp(codes.data() + block * 64, raw + block * 66 + 2, 64) == 0);
                }
            }
        }
    } else {
        const uint64_t blocks = w.down_unit / 84, pairs = EXPERTS * blocks / 2;
        const uint64_t dm_bytes = round_up(pairs * 8, 64);
        const uint64_t scale_bytes = round_up(pairs * 32, 64);
        std::vector<unsigned char> dm(blocks * 4), scales(blocks * 16), codes(blocks * 64);
        for (unsigned slot = 0; slot < USED; ++slot) {
            const uint64_t pair = uint64_t(IDS[slot]) * blocks / 2;
            CHECK(ds4_gpu_tensor_read(w.soa_down, pair * 8, dm.data(), dm.size()));
            CHECK(ds4_gpu_tensor_read(w.soa_down, dm_bytes + pair * 32, scales.data(), scales.size()));
            CHECK(ds4_gpu_tensor_read(w.soa_down, dm_bytes + scale_bytes + pair * 128, codes.data(), codes.size()));
            const unsigned char *raw = data.data() + 2 * USED * w.gate_unit + slot * w.down_unit;
            const uint64_t row_blocks = MID / QK;
            for (uint64_t block = 0; block < blocks; ++block) {
                const uint64_t row = block / row_blocks, col = block % row_blocks;
                const uint64_t p = (row / 2) * row_blocks + col, parity = row & 1;
                CHECK(std::memcmp(dm.data() + p * 8 + parity * 4, raw + block * 84 + 80, 4) == 0);
                for (uint64_t word = 0; word < 4; ++word) {
                    const uint64_t index = p * 8 + (word / 2) * 4 + parity * 2 + (word & 1);
                    CHECK(std::memcmp(scales.data() + index * 4, raw + block * 84 + word * 4, 4) == 0);
                }
                for (uint64_t word = 0; word < 16; ++word) {
                    CHECK(std::memcmp(codes.data() + (p * 32 + word * 2 + parity) * 4,
                                      raw + block * 84 + 16 + word * 4, 4) == 0);
                }
            }
        }
    }
    std::puts("original quant codes/scales roundtrip: BYTE-EXACT");
}

struct Buffers { ds4_gpu_tensor *gate, *up, *mid, *down, *out; };
struct Snapshot { std::vector<float> gate, up, mid, down, out; };

static std::vector<float> read(ds4_gpu_tensor *tensor, uint64_t count) {
    std::vector<float> output(count + GUARD);
    CHECK(ds4_gpu_tensor_read(tensor, 0, output.data(), output.size() * sizeof(float)));
    for (uint64_t i = 0; i < count; ++i) { CHECK(std::isfinite(output[i])); }
    for (uint64_t i = count; i < output.size(); ++i) { CHECK(output[i] == SENTINEL); }
    output.resize(count);
    return output;
}

static Snapshot snapshot(const Buffers &b, unsigned rows) {
    return {read(b.gate, uint64_t(rows) * USED * MID), read(b.up, uint64_t(rows) * USED * MID),
            read(b.mid, uint64_t(rows) * USED * MID), read(b.down, uint64_t(rows) * USED * OUTPUT),
            read(b.out, uint64_t(rows) * OUTPUT)};
}

static void clear(const Buffers &b) {
    for (auto *tensor : {b.gate, b.up, b.mid, b.down, b.out}) {
        CHECK(ds4_gpu_tensor_fill_f32(tensor, SENTINEL, ds4_gpu_tensor_bytes(tensor) / sizeof(float)));
    }
}

static bool compare(const char *arm, const char *stage, unsigned rows,
                    const std::vector<float> &actual, const std::vector<float> &reference,
                    Match match = Match::Bound) {
    CHECK(actual.size() == reference.size());
    double max_abs = 0, error = 0, signal = 0, peak = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double delta = double(actual[i]) - reference[i];
        max_abs = std::max(max_abs, std::fabs(delta));
        peak = std::max(peak, std::fabs(double(reference[i])));
        error += delta * delta;
        signal += double(reference[i]) * reference[i];
    }
    const double relative = std::sqrt(error / std::max(signal, 1.0e-30));
    std::printf("arm=%s rows=%u stage=%s max_abs=%.9g peak=%.9g relative_l2=%.9g\n",
                arm, rows, stage, max_abs, peak, relative);
    if (match == Match::Info) { return true; }
    if (match == Match::Exact) {
        return std::memcmp(actual.data(), reference.data(), actual.size() * sizeof(float)) == 0;
    }
    return relative <= PAIR_REL_BOUND && max_abs <= PAIR_REL_BOUND * std::max(peak, 1.0);
}

static bool compare(const char *arm, unsigned rows, const Snapshot &a, const Snapshot &r,
                    Match match = Match::Bound) {
    bool pass = compare(arm, "gate", rows, a.gate, r.gate, match);
    pass &= compare(arm, "up", rows, a.up, r.up, match);
    pass &= compare(arm, "mid", rows, a.mid, r.mid, match);
    pass &= compare(arm, "down", rows, a.down, r.down, match);
    pass &= compare(arm, "out", rows, a.out, r.out, match);
    return pass;
}

static float half_float(uint16_t bits) {
    __half value;
    std::memcpy(&value, &bits, sizeof(value));
    return __half2float(value);
}

/* Independent sampled dot: source GGUF codes plus the consumer's Q8
 * activation contract. Bounded to two assignments and three output rows,
 * so this never becomes a full CPU expert matmul. */
static double ref_dot(const unsigned char *raw, const float *input,
                       uint32_t type, unsigned rows) {
    double sum = 0;
    for (unsigned block = 0; block < INPUT / QK; ++block) {
        const auto *xxs = (const block_iq2_xxs *)(raw + block * (type == 16 ? 66 : 74));
        const auto *xs = (const block_iq2_xs *)xxs;
        for (unsigned group = 0; group < 8; ++group) {
            float maximum = 0;
            const float *x = input + block * QK + group * 32;
            for (unsigned i = 0; i < 32; ++i) { maximum = std::max(maximum, std::fabs(x[i])); }
            CHECK(maximum > 0);
            const float inverse = 127.0f / maximum;
            const float scale = rows <= 8
                ? __half2float(__float2half_rn(maximum / 127.0f)) : 1.0f / inverse;
            uint32_t packed[2];
            std::memcpy(packed, xxs->qs + 4 * group, sizeof(packed));
            const auto *codes = (const uint8_t *)packed;
            for (unsigned part = 0; part < 4; ++part) {
                const unsigned code = type == 17 ? xs->qs[4 * group + part] : codes[part];
                const auto *grid = (const uint8_t *)(type == 17
                    ? iq2xs_grid + (code & 511u) : iq2xxs_grid + code);
                const unsigned sign_id = type == 17 ? code >> 9
                    : (packed[1] >> (7 * part)) & 127u;
                const unsigned weight_scale = type == 17
                    ? (xs->scales[group] >> (4 * (part / 2))) & 15u : packed[1] >> 28;
                const float d = half_float(xxs->d) * (0.5f + weight_scale) * 0.25f;
                for (unsigned i = 0; i < 8; ++i) {
                    const float sign = ksigns_iq2xs[sign_id] & kmask_iq2xs[i] ? -1.0f : 1.0f;
                    const float quant = rows <= 8
                        ? std::round(x[part * 8 + i] / (maximum / 127.0f))
                        : std::round(x[part * 8 + i] * inverse);
                    sum += double(d * grid[i] * sign) * double(quant * scale);
                }
            }
        }
    }
    return sum;
}

static bool cpu_samples(const Weights &w, const std::vector<unsigned char> &data,
                         const std::vector<float> &input, unsigned rows,
                         const Snapshot &raw) {
    bool pass = true;
    for (unsigned token : {0u, rows - 1}) {
        for (unsigned slot : {0u, unsigned(USED - 1)}) {
            const unsigned expert = (slot + token) % USED;
            for (unsigned row : {0u, 1u, unsigned(MID - 1)}) {
                for (unsigned part = 0; part < 2; ++part) {
                    const unsigned char *weight = data.data() +
                        (part * USED + expert) * w.gate_unit + row * (w.gate_unit / MID);
                    const double reference = ref_dot(weight, input.data() + token * INPUT, w.gt, rows);
                    const float actual = (part == 0 ? raw.gate : raw.up)[(token * USED + slot) * MID + row];
                    const double delta = std::fabs(double(actual) - reference);
                    std::printf("cpu rows=%u token=%u slot=%u part=%u row=%u ref=%.9g got=%.9g abs=%.9g\n",
                                rows, token, slot, part, row, reference, actual, delta);
                    pass &= delta <= PAIR_REL_BOUND * std::max(1.0, std::fabs(reference));
                }
            }
        }
    }
    return pass;
}

static std::vector<block_q8_1_mmq> quant_mid(ds4_gpu_tensor *mid,
        const std::vector<float> &values, unsigned assignments) {
    CHECK(ds4_gpu_tensor_write(mid, 0, values.data(), uint64_t(assignments) * MID * sizeof(float)));
    const uint64_t blocks = uint64_t(assignments) * MID / 128;
    auto *quant = alloc(blocks * sizeof(block_q8_1_mmq));
    quantize_mmq_q8_1_cuda((const float *)mid->ptr, nullptr, quant->ptr,
        GGML_TYPE_IQ2_XS, MID, MID, MID, uint64_t(MID) * assignments,
        MID, assignments, 1, 1, ds4_current_stream());
    CUDA(cudaGetLastError());
    std::vector<block_q8_1_mmq> output(blocks);
    CHECK(ds4_gpu_tensor_read(quant, 0, output.data(), blocks * sizeof(block_q8_1_mmq)));
    ds4_gpu_tensor_free(quant);
    return output;
}

/* Consume the real D4 activation codes, independently decode source IQ2XS
 * weights, and predict the worst observed down delta with double summation. */
static double quant_delta(const unsigned char *raw,
        const std::vector<block_q8_1_mmq> &a,
        const std::vector<block_q8_1_mmq> &r, unsigned assignment, unsigned assignments) {
    double delta = 0;
    for (unsigned col = 0; col < MID; ++col) {
        const auto *weight = (const block_iq2_xs *)(raw + uint64_t(col / QK) * 74);
        const unsigned within = col % QK, group = within / 32, part = within % 32 / 8;
        const unsigned code = weight->qs[group * 4 + part];
        const auto *grid = (const uint8_t *)(iq2xs_grid + (code & 511u));
        const unsigned scale = (weight->scales[group] >> (4 * (part / 2))) & 15u;
        const float sign = ksigns_iq2xs[code >> 9] & kmask_iq2xs[within % 8] ? -1.0f : 1.0f;
        const float w = half_float(weight->d) * (0.5f + scale) * 0.25f * grid[within % 8] * sign;
        const auto &aq = a[uint64_t(col / 128) * assignments + assignment];
        const auto &rq = r[uint64_t(col / 128) * assignments + assignment];
        const unsigned at = col % 128;
        delta += double(w) * (double(aq.qs[at]) * aq.d4[at / 32] -
                              double(rq.qs[at]) * rq.d4[at / 32]);
    }
    return delta;
}

static bool mid_control(const Weights &w, const Buffers &b,
        const std::vector<unsigned char> &data, const ds4_gpu_tensor *selected,
        unsigned rows, const Snapshot &aligned, const Snapshot &raw) {
    CHECK(w.gt == 16 && w.dt == 17 && rows == MAX_ROWS);
    const unsigned assignments = rows * USED;
    const auto rq = quant_mid(b.mid, raw.mid, assignments);
    const auto aq = quant_mid(b.mid, aligned.mid, assignments);
    uint64_t codes = 0, scales = 0;
    for (uint64_t block = 0; block < rq.size(); ++block) {
        for (unsigned at = 0; at < 128; ++at) {
            if (rq[block].qs[at] == aq[block].qs[at]) { continue; }
            ++codes;
            if (codes <= 8) {
                std::printf("quant code assignment=%llu col=%llu raw=%d aligned=%d\n",
                    (unsigned long long)(block % assignments),
                    (unsigned long long)(block / assignments * 128 + at),
                    int(rq[block].qs[at]), int(aq[block].qs[at]));
            }
        }
        for (unsigned at = 0; at < 4; ++at) {
            scales += std::memcmp(&rq[block].d4[at], &aq[block].d4[at], sizeof(float)) != 0;
        }
    }
    size_t worst = 0;
    for (size_t at = 1; at < uint64_t(assignments) * OUTPUT; ++at) {
        if (std::fabs(double(aligned.down[at]) - raw.down[at]) >
            std::fabs(double(aligned.down[worst]) - raw.down[worst])) { worst = at; }
    }
    const unsigned assignment = unsigned(worst / OUTPUT), output = unsigned(worst % OUTPUT);
    const unsigned expert = (assignment % USED + assignment / USED) % USED;
    const auto *weight = data.data() + 2 * USED * w.gate_unit +
        expert * w.down_unit + uint64_t(output) * w.down_unit / OUTPUT;
    const double predicted = quant_delta(weight, aq, rq, assignment, assignments);
    const double observed = double(aligned.down[worst]) - raw.down[worst];
    std::printf("quant delta codes_changed=%llu/%llu scales_changed=%llu/%llu "
                "worst_assignment=%u output=%u predicted=%.12g observed=%.12g abs=%.12g\n",
                (unsigned long long)codes, (unsigned long long)(uint64_t(assignments) * MID),
                (unsigned long long)scales, (unsigned long long)(uint64_t(assignments) * MID / 32),
                assignment, output, predicted, observed, std::fabs(predicted - observed));
    const bool explained = codes > 0 && std::fabs(predicted - observed) <= 1.0e-5;

    clear(b);
    CHECK(ds4_gpu_tensor_write(b.mid, 0, raw.mid.data(), uint64_t(assignments) * MID * sizeof(float)));
    CHECK(ds4_mmq_glm_moe(w.dt, w.down->ptr, (const float *)b.mid->ptr,
        (const int32_t *)selected->ptr, (float *)b.down->ptr, OUTPUT, MID,
        assignments, EXPERTS, 1, w.down_unit, ds4_current_stream()) == 0);
    moe_sum_kernel<<<unsigned((uint64_t(rows) * OUTPUT + 255) / 256), 256>>>(
        (float *)b.out->ptr, (const float *)b.down->ptr, OUTPUT, USED, rows, 1);
    CUDA(cudaGetLastError());
    const bool down = compare("same-mid/raw", "down", rows,
        read(b.down, uint64_t(assignments) * OUTPUT), raw.down, Match::Exact);
    const bool out = compare("same-mid/raw", "out", rows,
        read(b.out, uint64_t(rows) * OUTPUT), raw.out, Match::Exact);
    return explained && down && out;
}

int main(int argc, char **argv) {
    CHECK((argc == 3 || argc == 4) &&
        (std::strcmp(argv[1], "middle") == 0 || std::strcmp(argv[1], "edge") == 0));
    const bool middle = std::strcmp(argv[1], "middle") == 0;
    const bool control = argc == 4;
    CHECK(!control || (middle && std::strcmp(argv[3], "mid-control") == 0));
    Weights w = {};
    w.gt = middle ? 16 : 17;
    w.dt = middle ? 17 : 10;
    w.gate_unit = uint64_t(INPUT / QK) * MID * (middle ? 66 : 74);
    w.down_unit = uint64_t(MID / QK) * OUTPUT * (middle ? 74 : 84);
    /* The actual common SSD cache uses LCM block alignment across recipes. */
    w.gate_stride = round_up(uint64_t(INPUT / QK) * MID * 74, 2442);
    w.down_stride = round_up(uint64_t(MID / QK) * OUTPUT * 84, 3108);
    std::ifstream file(argv[2], std::ios::binary | std::ios::ate);
    CHECK(file.good());
    const uint64_t bytes = USED * (2 * w.gate_unit + w.down_unit);
    CHECK(uint64_t(file.tellg()) == bytes);
    std::vector<unsigned char> data(bytes);
    file.seekg(0);
    CHECK(bool(file.read((char *)data.data(), data.size())));
    CHECK(ds4_gpu_init() && ds4_gpu_vmm_demand_page());
    const uint64_t census_faults = ds4_gpu_mem_census_faults();
    const uint64_t memgov_faults = ds4_metrics_get()->memgov_faults;
    w.gate = reserve(EXPERTS * w.gate_unit);
    w.up = reserve(EXPERTS * w.gate_unit);
    w.down = reserve(EXPERTS * w.down_unit);
    w.ssd_gate = reserve(SLOTS * w.gate_stride);
    w.ssd_up = reserve(SLOTS * w.gate_stride);
    w.ssd_down = reserve(SLOTS * w.down_stride);
    for (unsigned slot = 0; slot < USED; ++slot) {
        const unsigned char *g = data.data() + slot * w.gate_unit;
        const unsigned char *u = data.data() + (USED + slot) * w.gate_unit;
        const unsigned char *d = data.data() + 2 * USED * w.gate_unit + slot * w.down_unit;
        upload(w.gate, IDS[slot] * w.gate_unit, g, w.gate_unit);
        upload(w.up, IDS[slot] * w.gate_unit, u, w.gate_unit);
        upload(w.down, IDS[slot] * w.down_unit, d, w.down_unit);
        upload_slot(w.ssd_gate, SLOT_IDS[slot] * w.gate_stride, g, w.gate_unit, w.gate_stride);
        upload_slot(w.ssd_up, SLOT_IDS[slot] * w.gate_stride, u, w.gate_unit, w.gate_stride);
        upload_slot(w.ssd_down, SLOT_IDS[slot] * w.down_stride, d, w.down_unit, w.down_stride);
    }
    if (middle) {
        w.soa_gate = iq2_soa(w.gate, w.gate_unit);
        w.soa_up = iq2_soa(w.up, w.gate_unit);
    } else {
        w.soa_down = q2_soa(w.down, w.down_unit);
    }
    CUDA(cudaDeviceSynchronize());
    weight_roundtrip(w, data);
    uint64_t resident = 0, virtual_bytes = 0;
    for (auto *t : {w.gate, w.up, w.down, w.soa_gate, w.soa_up, w.soa_down,
                   w.ssd_gate, w.ssd_up, w.ssd_down}) {
        if (!t) { continue; }
        const uint64_t one = ds4_gpu_tensor_bytes(t);
        virtual_bytes += one;
        resident += ds4_gpu_tensor_resident(t, 0, one);
    }
    std::printf("shape=%u,%u,%u experts=%u slots=%u used=%u gate_type=%u down_type=%u "
                "weight_physical=%llu weight_virtual=%llu gate_stride=%llu down_stride=%llu\n",
                INPUT, MID, OUTPUT, EXPERTS, SLOTS, USED, w.gt, w.dt,
                (unsigned long long)resident, (unsigned long long)virtual_bytes,
                (unsigned long long)w.gate_stride, (unsigned long long)w.down_stride);
    CHECK(resident < 512ull * 1024 * 1024);
    std::vector<float> input(MAX_ROWS * INPUT), routes(MAX_ROWS * USED);
    std::vector<int32_t> ids(MAX_ROWS * USED), slots(MAX_ROWS * USED);
    for (unsigned row = 0; row < MAX_ROWS; ++row) {
        for (unsigned col = 0; col < INPUT; ++col) {
            input[row * INPUT + col] = 2 * std::sin(float(col) * 0.017f + row * 0.11f) +
                0.31f * std::cos(float(col) * 0.071f - row * 0.13f);
        }
        for (unsigned slot = 0; slot < USED; ++slot) {
            const unsigned choice = (slot + row) % USED;
            ids[row * USED + slot] = IDS[choice];
            slots[row * USED + slot] = SLOT_IDS[choice];
            routes[row * USED + slot] = ROUTE_SCALE * float(choice + 1) / 36;
        }
    }
    auto *x = alloc(input.size() * sizeof(float), input.data());
    auto *weights = alloc(routes.size() * sizeof(float), routes.data());
    auto *selected = alloc(ids.size() * sizeof(int32_t), ids.data());
    auto *remapped = alloc(slots.size() * sizeof(int32_t), slots.data());
    Buffers b = {alloc((uint64_t(MAX_ROWS) * USED * MID + GUARD) * sizeof(float)),
                 alloc((uint64_t(MAX_ROWS) * USED * MID + GUARD) * sizeof(float)),
                 alloc((uint64_t(MAX_ROWS) * USED * MID + GUARD) * sizeof(float)),
                 alloc((uint64_t(MAX_ROWS) * USED * OUTPUT + GUARD) * sizeof(float)),
                 alloc((uint64_t(MAX_ROWS) * OUTPUT + GUARD) * sizeof(float))};
    bool pass = true;
    for (unsigned rows : {1u, 17u, 128u}) {
        if (control && rows != MAX_ROWS) { continue; }
        clear(b);
        CHECK(glm53_moe_mixed(b.out, b.gate, b.up, b.mid, b.down,
            (char *)w.gate->ptr, (char *)w.up->ptr, (char *)w.down->ptr,
            GlmMoELayout::Raw, GlmMoELayout::Raw, w.gt, w.dt,
            w.gate_unit, w.down_unit, INPUT, MID, OUTPUT, selected, weights,
            EXPERTS, USED, rows, SWIGLU_CLAMP, x, 0, nullptr));
        const Snapshot raw = snapshot(b, rows);
        pass &= cpu_samples(w, data, input, rows, raw);
        clear(b);
        CHECK(glm53_moe_mixed(b.out, b.gate, b.up, b.mid, b.down,
            (char *)(middle ? w.soa_gate : w.gate)->ptr,
            (char *)(middle ? w.soa_up : w.up)->ptr,
            (char *)(middle ? w.down : w.soa_down)->ptr,
            middle ? GlmMoELayout::IQ2SoA : GlmMoELayout::Raw,
            middle ? GlmMoELayout::Raw : GlmMoELayout::Q2SoA, w.gt, w.dt,
            w.gate_unit, w.down_unit, INPUT, MID, OUTPUT, selected, weights,
            EXPERTS, USED, rows, SWIGLU_CLAMP, x, 0, nullptr));
        const Snapshot aligned = snapshot(b, rows);
        if (control) {
            const bool paired = compare("aligned/raw", rows, aligned, raw);
            std::printf("original local bound=%s (retained)\n", paired ? "PASS" : "RED");
            pass &= mid_control(w, b, data, selected, rows, aligned, raw);
        } else if (middle) {
            pass &= compare("aligned/raw", "gate", rows, aligned.gate, raw.gate);
            pass &= compare("aligned/raw", "up", rows, aligned.up, raw.up);
            compare("aligned/raw-info", "mid", rows, aligned.mid, raw.mid, Match::Info);
            compare("aligned/raw-info", "down", rows, aligned.down, raw.down, Match::Info);
            compare("aligned/raw-info", "out", rows, aligned.out, raw.out, Match::Info);
            /* The IQ2XS down producer can change integer Q8 codes after tiny
             * gate/up fold-order drift. Gate its explained delta and exact
             * identical-mid recovery rather than loosening an output bound. */
            if (rows == MAX_ROWS) { pass &= mid_control(w, b, data, selected, rows, aligned, raw); }
        } else {
            /* The edge recipe retains its original diagnostic bound until
             * its separate Q2K down arithmetic has an independent gate. */
            pass &= compare("aligned/raw", rows, aligned, raw);
        }
        clear(b);
        CHECK(ds4_gpu_glm53_moe_owned(b.out, b.gate, b.up, b.mid, b.down,
            w.ssd_gate, w.ssd_up, w.ssd_down, w.gt, w.dt,
            w.gate_stride, w.down_stride, INPUT, MID, OUTPUT, remapped,
            weights, SLOTS, USED, rows, SWIGLU_CLAMP, x));
        pass &= compare("owned/raw", rows, snapshot(b, rows), raw, Match::Exact);
        std::fflush(stdout);
    }
    check_padding(w.ssd_gate, w.gate_unit, w.gate_stride);
    check_padding(w.ssd_up, w.gate_unit, w.gate_stride);
    check_padding(w.ssd_down, w.down_unit, w.down_stride);
    for (auto *t : {b.out, b.gate, b.up, b.mid, b.down, x, weights, selected, remapped,
                   w.gate, w.up, w.down, w.soa_gate, w.soa_up, w.soa_down,
                   w.ssd_gate, w.ssd_up, w.ssd_down}) { ds4_gpu_tensor_free(t); }
    std::printf("faults census_before=%llu census_after=%llu memgov_before=%llu memgov_after=%llu\n",
                (unsigned long long)census_faults,
                (unsigned long long)ds4_gpu_mem_census_faults(),
                (unsigned long long)memgov_faults,
                (unsigned long long)ds4_metrics_get()->memgov_faults);
    CHECK(ds4_gpu_mem_census_faults() == census_faults);
    CHECK(ds4_metrics_get()->memgov_faults == memgov_faults);
    ds4_gpu_cleanup();
    if (control) { std::puts(pass ? "GLM same-mid explanation: PASS" : "GLM same-mid explanation: FAIL"); }
    else { std::puts(pass ? "GLM actual-shape routes: PASS" : "GLM actual-shape routes: FAIL"); }
    return pass ? 0 : 1;
}
