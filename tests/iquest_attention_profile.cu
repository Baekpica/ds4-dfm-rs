// Synthetic resident operands, production baseline/candidate attention launches.
// Build with the production NVCCFLAGS; no model, owner, or GPU clock changes.
// NCU: ncu --clock-control none --kernel-name regex:iquest_attn_kernel \
//   --launch-count 1 ./tests/iquest_attention_profile --rows 128 \
//   --position 8191 --window swa --capacity 4223 --warmup 0 --repeat 1
#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>
#include "../cuda/iquest_primitives.cuh"
#include "../cuda/iquest_prefill.cuh"
#include "../cuda/iquest_decode.cuh"

#define CUDA_OK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        std::exit(2); \
    } \
} while (0)

enum class Fixture { Long, Reference13, F32Sink, Q8Stress };
enum class Implementation { Baseline, Reduced, Grouped, Tiled, Cache32, Cache64, Cache128, Async };
enum class Positions { Contiguous, Permuted };
constexpr unsigned kMaxCapacity = 8192;
constexpr unsigned kWideCapacity = 16384;
constexpr unsigned kSwaCapacity = IQ_WINDOW + IQ_PREFILL - 1;
constexpr unsigned kMaxLaunches = 16;
constexpr unsigned kGroupedBlocks = IQ_HEADS / IQ_ATTN_HEADS_PER_BLOCK;
constexpr unsigned kGroupedThreads = IQ_ATTN_HEADS_PER_BLOCK * IQ_WARP_WIDTH;
static_assert(IQ_HEADS % IQ_ATTN_HEADS_PER_BLOCK == 0 && IQ_HEAD == kGroupedThreads,
              "grouped launch requires four warps and complete head groups");
constexpr float kCancellationAtol = 2e-6f;
constexpr float kMinAgreement = 0.99f;

struct Options {
    Fixture fixture = Fixture::Long;
    Implementation implementation = Implementation::Baseline;
    Positions positions = Positions::Contiguous;
    bool compare = false;
    bool compare_all = false;
    const char *dump_prefix = nullptr;
    unsigned rows = 1;
    unsigned position = 2047;
    unsigned window = 0;
    unsigned capacity = kMaxCapacity;
    unsigned warmup = 1;
    unsigned repeat = 3;
    unsigned device = 0;
};

static void usage() {
    std::fprintf(stderr,
        "Usage: iquest_attention_profile [--case long|reference13|f32sink|q8stress]\n"
        "  [--implementation baseline|reduced|grouped|tiled|cache32|cache64|cache128|async] [--compare|--compare-all]\n"
        "  [--dump-prefix PATH]\n"
        "  [--rows 1|128] [--positions contiguous|permuted]\n"
        "  [--position 0|1|31|32|63|64|127|128|511|512|518|519|2047|4095|4096|4222|4223|8191|8445|8446]\n"
        "  [--window full|swa|mtp] [--capacity 519|4223|8192|16384]\n"
        "  [--warmup 0..16] [--repeat 1..16] [--device N]\n"
        "reference13 fixes rows=1, position=12, full attention, capacity=16.\n"
        "f32sink uses nonsymmetric non-BF16 F32 sink values.\n"
        "--compare checks baseline/reduced, or grouped against retained reduced.\n"
        "--compare-all requires byte-exact outputs from all three kernels.\n"
        "For tiled, --compare reports drift against retained grouped; --compare-all\n"
        "also checks grouped/baseline exactness. Tiled requires BF16 query inputs.\n"
        "Tiled runs sampled FP64 ordinary/LSE references and an exact independent\n"
        "sink-stage check outside timing; numerical adoption requires model review.\n"
        "q8stress directly writes signed Q8 endpoints and finite extreme scales.\n"
        "--dump-prefix saves each selected .IMPLEMENTATION.f32 and .positions.u32.\n"
        "Grouped production eligibility is rows128 and full/SWA only. Other\n"
        "grouped shapes are diagnostic math cases, including the MTP window.\n"
        "Long cases check finite BF16 output, not model numerical parity.\n"
        "Use --warmup 0 --repeat 1 for one attention launch under NCU.\n");
}

static void fail(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(2);
}

static unsigned number(const char *text) {
    char *end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (errno || !*text || *end || *text == '-' || parsed > UINT32_MAX) {
        fail("Expected an unsigned 32-bit integer");
    }
    return static_cast<unsigned>(parsed);
}

static Options options(int argc, char **argv) {
    Options result;
    bool shape_given = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!std::strcmp(arg, "--help")) { usage(); std::exit(0); }
        if (!std::strcmp(arg, "--compare")) { result.compare = true; continue; }
        if (!std::strcmp(arg, "--compare-all")) {
            result.compare = true; result.compare_all = true; continue;
        }
        if (++i == argc) { usage(); fail("Missing option value"); }
        const char *value = argv[i];
        if (!std::strcmp(arg, "--case")) {
            if (!std::strcmp(value, "long")) { result.fixture = Fixture::Long; }
            else if (!std::strcmp(value, "reference13")) { result.fixture = Fixture::Reference13; }
            else if (!std::strcmp(value, "f32sink")) { result.fixture = Fixture::F32Sink; }
            else if (!std::strcmp(value, "q8stress")) { result.fixture = Fixture::Q8Stress; }
            else { fail("Unknown fixture"); }
        } else if (!std::strcmp(arg, "--implementation")) {
            if (!std::strcmp(value, "baseline")) { result.implementation = Implementation::Baseline; }
            else if (!std::strcmp(value, "reduced")) { result.implementation = Implementation::Reduced; }
            else if (!std::strcmp(value, "grouped")) { result.implementation = Implementation::Grouped; }
            else if (!std::strcmp(value, "tiled")) { result.implementation = Implementation::Tiled; }
            else if (!std::strcmp(value, "cache32")) { result.implementation = Implementation::Cache32; }
            else if (!std::strcmp(value, "cache64")) { result.implementation = Implementation::Cache64; }
            else if (!std::strcmp(value, "cache128")) { result.implementation = Implementation::Cache128; }
            else if (!std::strcmp(value, "async")) { result.implementation = Implementation::Async; }
            else { fail("Unknown implementation"); }
        } else if (!std::strcmp(arg, "--positions")) {
            shape_given = true;
            if (!std::strcmp(value, "contiguous")) { result.positions = Positions::Contiguous; }
            else if (!std::strcmp(value, "permuted")) { result.positions = Positions::Permuted; }
            else { fail("Positions must be contiguous or permuted"); }
        } else if (!std::strcmp(arg, "--dump-prefix")) {
            result.dump_prefix = value;
        } else if (!std::strcmp(arg, "--window")) {
            shape_given = true;
            if (!std::strcmp(value, "full")) { result.window = 0; }
            else if (!std::strcmp(value, "swa")) { result.window = IQ_WINDOW; }
            else if (!std::strcmp(value, "mtp")) { result.window = IQ_DRAFT_WINDOW; }
            else { fail("Window must be full, swa, or mtp"); }
        } else if (!std::strcmp(arg, "--rows")) {
            shape_given = true; result.rows = number(value);
        } else if (!std::strcmp(arg, "--position")) {
            shape_given = true; result.position = number(value);
        } else if (!std::strcmp(arg, "--capacity")) {
            shape_given = true; result.capacity = number(value);
        } else if (!std::strcmp(arg, "--warmup")) {
            result.warmup = number(value);
        } else if (!std::strcmp(arg, "--repeat")) {
            result.repeat = number(value);
        } else if (!std::strcmp(arg, "--device")) {
            result.device = number(value);
        } else { usage(); fail("Unknown option"); }
    }
    if (!result.repeat || result.repeat > kMaxLaunches || result.warmup > kMaxLaunches) {
        fail("Warmup/repeat exceeds the bounded launch count");
    }
    if (result.device > INT32_MAX) { fail("Device ordinal is too large"); }
    if (result.fixture == Fixture::Reference13) {
        if (result.implementation == Implementation::Tiled) {
            fail("reference13 has non-BF16 queries outside the tiled input contract");
        }
        if (shape_given) { fail("reference13 cannot override shape options"); }
        result.rows = 1; result.position = 12; result.window = 0; result.capacity = 16;
        return result;
    }
    if (result.rows != 1 && result.rows != IQ_PREFILL) { fail("Rows must be 1 or 128"); }
    const unsigned positions[] = {0, 1, 31, 32, 63, 64, 127, 128, 511, 512, 518, 519, 2047, 4095, 4096,
                                  4222, 4223, 8191, 8445, 8446};
    if (std::find(std::begin(positions), std::end(positions), result.position) == std::end(positions)) {
        fail("Position must be a documented bounded window/ring case");
    }
    if (result.capacity != IQ_DRAFT_WINDOW + IQ_DRAFT_SLOTS &&
        result.capacity != kSwaCapacity && result.capacity != kMaxCapacity &&
        result.capacity != kWideCapacity) {
        fail("Capacity must be 519, 4223, 8192, or 16384");
    }
    if (result.position + 1 < result.rows) {
        fail("Query rows exceed the available logical positions");
    }
    const unsigned first = result.position + 1 - result.rows;
    const unsigned begin = result.window && first + 1 > result.window ? first + 1 - result.window : 0;
    // All rows share the cache after the whole chunk is appended. Retain the
    // earliest query's history as well as the later rows (4096 + 128 - 1).
    if (result.position + 1 - begin > result.capacity) {
        fail("Capacity would overwrite history still needed by this query chunk");
    }
    if (result.implementation == Implementation::Tiled) { result.compare = true; }
    return result;
}

static float ref_value(unsigned i) {
    return std::sin((i + 1) * 0.117f) + 0.3f * std::cos((i + 3) * 0.031f);
}

static float fixture_value(unsigned i, Fixture fixture) {
    if (fixture == Fixture::Reference13) { return ref_value(i); }
    // Integer mixing and power-of-two scaling make the long fixture portable;
    // round the F32 storage to the graph's BF16 attention-input boundary.
    i ^= i >> 16; i *= 0x7feb352du; i ^= i >> 15; i *= 0x846ca68bu; i ^= i >> 16;
    return iquest_bf16((static_cast<int>(i & 0xffffu) - 32768) / 32768.0f);
}

static float fixture_sink(unsigned i, Fixture fixture) {
    if (fixture != Fixture::F32Sink && fixture != Fixture::Q8Stress) {
        return fixture_value(i, fixture);
    }
    // Source sink weights are F32. Nonzero low mantissa bits exercise an
    // accidental BF16 cast, with distinct keys across the eight KV heads.
    i ^= UINT32_C(0x9e3779b9); i *= UINT32_C(0x846ca68b); i ^= i >> 13;
    const uint32_t bits = (i & UINT32_C(0x807ffffe)) | UINT32_C(0x3e800001);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static const char *impl_name(Implementation implementation) {
    if (implementation == Implementation::Async) { return "async"; }
    if (implementation == Implementation::Cache32) { return "cache32"; }
    if (implementation == Implementation::Cache64) { return "cache64"; }
    if (implementation == Implementation::Cache128) { return "cache128"; }
    if (implementation == Implementation::Baseline) { return "baseline"; }
    if (implementation == Implementation::Tiled) { return "tiled"; }
    return implementation == Implementation::Reduced ? "reduced" : "grouped";
}

static const char *kernel_name(Implementation implementation) {
    if (implementation >= Implementation::Cache32) { return "iq_decode::cached"; }
    if (implementation == Implementation::Baseline) { return "iquest_attn_kernel"; }
    if (implementation == Implementation::Tiled) { return "iq_prefill::prefill+sink"; }
    return implementation == Implementation::Reduced
        ? "iquest_attn_shuffle_kernel" : "iquest_attn_warp_kernel";
}

static unsigned grid_heads(Implementation implementation) {
    return implementation == Implementation::Grouped ? kGroupedBlocks : IQ_HEADS;
}

static void launch_attention(Implementation implementation, float *out,
        const float *query, const iquest_q8 *cache, const float *sink,
        const unsigned *positions, const Options &opt, float *lse = nullptr) {
    if (implementation == Implementation::Baseline) {
        iquest_attn_kernel<<<dim3(opt.rows, IQ_HEADS), IQ_HEAD>>>(
            out, query, cache, sink, positions, opt.capacity, opt.window);
    } else if (implementation == Implementation::Reduced) {
        iquest_attn_shuffle_kernel<<<dim3(opt.rows, IQ_HEADS), IQ_HEAD>>>(
            out, query, cache, sink, positions, opt.capacity, opt.window);
    } else if (implementation == Implementation::Grouped) {
        iquest_attn_warp_kernel<<<dim3(opt.rows, kGroupedBlocks), kGroupedThreads>>>(
            out, query, cache, sink, positions, opt.capacity, opt.window);
    } else if (implementation == Implementation::Cache32) {
        iq_decode::cached<32><<<dim3(opt.rows, IQ_HEADS), iq_decode::THREADS>>>(
            out, query, cache, sink, positions, opt.capacity, opt.window);
    } else if (implementation == Implementation::Cache64) {
        iq_decode::cached<64><<<dim3(opt.rows, IQ_HEADS), iq_decode::THREADS>>>(
            out, query, cache, sink, positions, opt.capacity, opt.window);
    } else if (implementation == Implementation::Cache128) {
        iq_decode::cached<128><<<dim3(opt.rows, IQ_HEADS), iq_decode::THREADS>>>(
            out, query, cache, sink, positions, opt.capacity, opt.window);
    } else if (implementation == Implementation::Async) {
        static const bool supported = iq_decode::async_supported();
        if (!supported) { fail("Async copy requires an SM80+ compiled target"); }
        iq_decode::cached<128, iq_decode::Transfer::Asynchronous>
            <<<dim3(opt.rows, IQ_HEADS), iq_decode::THREADS>>>(
                out, query, cache, sink, positions, opt.capacity, opt.window);
    } else {
        if (!lse) { fail("Tiled attention requires an LSE buffer"); }
        iq_prefill::prefill<<<dim3((opt.rows + iq_prefill::TQ - 1) / iq_prefill::TQ, IQ_HEADS),
                                 iq_prefill::THREADS>>>(
            out, lse, query, cache, positions, opt.rows, opt.capacity, opt.window);
        CUDA_OK(cudaGetLastError());
        iq_prefill::sink<<<dim3(opt.rows, kGroupedBlocks), kGroupedThreads>>>(
            out, lse, query, sink, opt.rows);
    }
    CUDA_OK(cudaGetLastError());
}

static void dump_bytes(const char *prefix, const char *suffix, const void *data, size_t bytes) {
    const std::string path = std::string(prefix) + suffix;
    FILE *file = std::fopen(path.c_str(), "wb");
    if (!file) { fail("Cannot open raw output path"); }
    const bool written = std::fwrite(data, 1, bytes, file) == bytes;
    const bool closed = std::fclose(file) == 0;
    if (!written || !closed) { fail("Cannot write complete raw output"); }
}

template<class T> static T *upload(const std::vector<T> &input) {
    T *device = nullptr;
    CUDA_OK(cudaMalloc(&device, input.size() * sizeof(T)));
    CUDA_OK(cudaMemcpy(device, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

static uint64_t hash_bytes(const void *data, size_t bytes) {
    const auto *input = static_cast<const unsigned char *>(data);
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < bytes; i++) { hash = (hash ^ input[i]) * UINT64_C(1099511628211); }
    return hash;
}

static float bf16_ulp(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x10000u;
    float next;
    std::memcpy(&next, &bits, sizeof(next));
    return std::fabs(next - value);
}

struct Comparison {
    Implementation implementation = Implementation::Baseline;
    std::vector<float> output;
    size_t differing_bits = 0, nonfinite = 0, non_bf16 = 0;
    double max_abs = 0;
    bool passed() const { return !differing_bits && !nonfinite && !non_bf16; }
};

static Comparison compare_output(Implementation implementation, const std::vector<float> &output,
        const float *query, const iquest_q8 *cache, const float *sink,
        const unsigned *positions, const Options &opt) {
    Comparison result;
    result.implementation = implementation;
    result.output.resize(output.size());
    float *other = nullptr;
    CUDA_OK(cudaMalloc(&other, output.size() * sizeof(float)));
    // A separate poisoned output catches missing writes in every row/head,
    // including four-head CTAs crossing the six-query-head GQA boundary.
    CUDA_OK(cudaMemset(other, 0xff, output.size() * sizeof(float)));
    launch_attention(implementation, other, query, cache, sink, positions, opt);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(result.output.data(), other, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(other));
    for (size_t i = 0; i < output.size(); i++) {
        uint32_t bits, other_bits;
        std::memcpy(&bits, &output[i], sizeof(bits));
        std::memcpy(&other_bits, &result.output[i], sizeof(other_bits));
        result.differing_bits += bits != other_bits;
        result.nonfinite += !std::isfinite(result.output[i]);
        result.non_bf16 += (other_bits & 0xffffu) != 0;
        if (std::isfinite(output[i]) && std::isfinite(result.output[i])) {
            result.max_abs = std::max(result.max_abs,
                std::fabs(static_cast<double>(output[i]) - result.output[i]));
        }
    }
    return result;
}

static void print_comparison(const char *name, const Comparison &comparison) {
    std::printf("\"%s\":{\"other_implementation\":\"%s\",\"untimed_launches\":1,\"elements\":%zu,"
        "\"byte_exact\":%s,\"different_elements\":%zu,\"max_abs\":%.9g,"
        "\"other_nonfinite\":%zu,\"other_non_bf16\":%zu,\"other_output_fnv1a64\":\"%016llx\"},",
        name, impl_name(comparison.implementation), comparison.output.size(),
        comparison.differing_bits ? "false" : "true", comparison.differing_bits,
        comparison.max_abs, comparison.nonfinite, comparison.non_bf16,
        static_cast<unsigned long long>(hash_bytes(comparison.output.data(), comparison.output.size() * sizeof(float))));
}

static void dump_output(const char *prefix, Implementation implementation,
                        const std::vector<float> &output) {
    const std::string suffix = std::string(".") + impl_name(implementation) + ".f32";
    dump_bytes(prefix, suffix.c_str(), output.data(), output.size() * sizeof(float));
}

struct Drift {
    size_t count = 0, different = 0, nonfinite = 0, worst = 0;
    double max_abs = 0, error_square = 0, reference_square = 0;
    size_t bf16_ulps[5] = {};

    void add(float actual, double expected, size_t index) {
        count++;
        if (!std::isfinite(actual) || !std::isfinite(expected)) { nonfinite++; return; }
        const double error = static_cast<double>(actual) - expected;
        different += error != 0;
        error_square += error * error;
        reference_square += expected * expected;
        if (std::fabs(error) > max_abs) { max_abs = std::fabs(error); worst = index; }
        const float rounded = iquest_bf16(static_cast<float>(expected));
        const double ulp = std::max(bf16_ulp(actual), bf16_ulp(rounded));
        const double distance = ulp ? std::fabs(error) / ulp : 0;
        const unsigned bucket = distance == 0 ? 0 : distance <= 1 ? 1 : distance <= 2 ? 2 : distance <= 8 ? 3 : 4;
        bf16_ulps[bucket]++;
    }
};

enum class UlpScale { Bf16, None };

static void print_drift(const char *name, const Drift &d, UlpScale scale = UlpScale::Bf16) {
    std::printf("\"%s\":{\"elements\":%zu,\"different\":%zu,\"nonfinite\":%zu,"
        "\"max_abs\":%.17g,\"rms\":%.17g,\"relative_l2\":",
        name, d.count, d.different, d.nonfinite, d.max_abs,
        d.count ? std::sqrt(d.error_square / d.count) : 0);
    if (d.reference_square) { std::printf("%.17g", std::sqrt(d.error_square / d.reference_square)); }
    else { std::printf("null"); }
    std::printf(",\"worst_flat_index\":%zu", d.worst);
    if (scale == UlpScale::Bf16) {
        std::printf(",\"worst_row_head_dim\":[%zu,%zu,%zu],\"bf16_ulp_buckets_0_le1_le2_le8_gt8\":[%zu,%zu,%zu,%zu,%zu]",
            d.worst / (IQ_HEADS * IQ_HEAD), d.worst / IQ_HEAD % IQ_HEADS, d.worst % IQ_HEAD,
            d.bf16_ulps[0], d.bf16_ulps[1], d.bf16_ulps[2], d.bf16_ulps[3], d.bf16_ulps[4]);
    }
    std::printf("},");
}

static void stress_cache(std::vector<iquest_q8> &cache, unsigned begin,
                         const Options &opt) {
    // Direct storage stress includes zero, half subnormals, normal scale
    // boundaries and the largest finite scale. Existing packer inputs stay fixed.
    constexpr uint16_t scales[] = {0x0000, 0x0001, 0x03ff, 0x0400,
                                   0x1401, 0x3bff, 0x6400, 0x7bff};
    constexpr int8_t codes[] = {-127, -64, -1, 0, 1, 63, 126, 127};
    for (unsigned token = begin; token <= opt.position; token++) {
        for (unsigned part = 0; part < IQ_Q8_ROW_BLOCKS; part++) {
            iquest_q8 &block = cache[(size_t)(token % opt.capacity) * IQ_Q8_ROW_BLOCKS + part];
            block.d = scales[(token * 3 + part * 5) % std::size(scales)];
            for (unsigned d = 0; d < IQ_Q8_BLOCK; d++) {
                block.qs[d] = codes[(token * 7 + part * 3 + d) % std::size(codes)];
            }
        }
    }
}

static double source_q8(const std::vector<iquest_q8> &cache, unsigned token,
                        unsigned head, unsigned d, unsigned capacity, unsigned offset) {
    const size_t part = offset + (head * IQ_HEAD + d) / IQ_Q8_BLOCK;
    const iquest_q8 &block = cache[(size_t)(token % capacity) * IQ_Q8_ROW_BLOCKS + part];
    return static_cast<double>(iquest_half(block.d)) * block.qs[d % IQ_Q8_BLOCK];
}

static float tf32_rna(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000)) {
        // TF32 cvt.rna rounds ties away from zero, unlike BF16's RN-even.
        bits = (bits + UINT32_C(0x1000)) & UINT32_C(0xffffe000);
    }
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

struct Components {
    size_t values = 0, pair_inexact = 0, nonfinite = 0, query_inexact = 0;
    double max_pair_error = 0;
};

static Components check_components(const std::vector<float> &query,
                                    const std::vector<iquest_q8> &cache) {
    Components result;
    for (float q : query) { result.query_inexact += tf32_rna(q) != q; }
    for (const auto &block : cache) {
        for (int8_t code : block.qs) {
            const float value = iquest_half(block.d) * code;
            const float high = tf32_rna(value);
            const float low = tf32_rna(value - high);
            const double error = static_cast<double>(high) + low - value;
            result.values++;
            result.nonfinite += !std::isfinite(high) || !std::isfinite(low);
            result.pair_inexact += error != 0;
            result.max_pair_error = std::max(result.max_pair_error, std::fabs(error));
        }
    }
    return result;
}

// Independent one-head CTA reproduces the retained sink's rounded leaf/tree
// order. It consumes candidate ordinary/LSE, isolating sink and BF16 boundaries.
__global__ static void sink_reference(float *out, const float *lse, const float *q,
                                      const float *sink) {
    __shared__ float scratch[IQ_HEAD];
    const unsigned row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const unsigned kh = head / (IQ_HEADS / IQ_KV_HEADS);
    const size_t index = ((size_t)row * IQ_HEADS + head) * IQ_HEAD + lane;
    scratch[lane] = __fmul_rn(q[index], sink[kh * IQ_HEAD + lane]);
    __syncthreads();
    for (unsigned stride = IQ_HEAD / 2; stride; stride >>= 1) {
        if (lane < stride) { scratch[lane] = __fadd_rn(scratch[lane], scratch[lane + stride]); }
        __syncthreads();
    }
    const float exponent = __fmaf_rn(scratch[0], rsqrtf((float)IQ_HEAD), -lse[row * IQ_HEADS + head]);
    const float factor = 1.0f / (1.0f + expf(exponent));
    const float ordinary = __bfloat162float(__float2bfloat16_rn(out[index]));
    out[index] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(ordinary, factor)));
}

struct TiledCheck {
    std::vector<float> ordinary, lse, sink_output;
    std::vector<unsigned> sample_rows;
    Drift full_drift, source_ordinary, source_lse, source_final, retained_final;
    Components components;
    size_t ordinary_nonfinite = 0, ordinary_non_bf16 = 0, lse_nonfinite = 0;
    size_t sink_different = 0;
    bool passed() const {
        return !ordinary_nonfinite && !ordinary_non_bf16 && !lse_nonfinite &&
            !sink_different && !components.nonfinite && !components.query_inexact &&
            !components.pair_inexact && !source_ordinary.nonfinite &&
            !source_lse.nonfinite && !source_final.nonfinite && !retained_final.nonfinite;
    }
};

static void source_samples(TiledCheck &check, const std::vector<float> &query,
        const std::vector<float> &sink, const std::vector<iquest_q8> &cache,
        const std::vector<unsigned> &positions, const std::vector<float> &output,
        const std::vector<float> &retained, const Options &opt) {
    check.sample_rows = {0, opt.rows / 2, opt.rows - 1};
    const auto bounds = std::minmax_element(positions.begin(), positions.end());
    check.sample_rows.push_back(static_cast<unsigned>(bounds.first - positions.begin()));
    check.sample_rows.push_back(static_cast<unsigned>(bounds.second - positions.begin()));
    std::sort(check.sample_rows.begin(), check.sample_rows.end());
    check.sample_rows.erase(std::unique(check.sample_rows.begin(), check.sample_rows.end()), check.sample_rows.end());
    const double scale = 1.0 / std::sqrt(static_cast<double>(IQ_HEAD));
    for (unsigned row : check.sample_rows) {
        const unsigned end = positions[row];
        const unsigned begin = opt.window && end + 1 > opt.window ? end + 1 - opt.window : 0;
        std::vector<double> scores(end + 1 - begin);
        for (unsigned head = 0; head < IQ_HEADS; head++) {
            const unsigned kh = head / (IQ_HEADS / IQ_KV_HEADS);
            const size_t base = ((size_t)row * IQ_HEADS + head) * IQ_HEAD;
            double maximum = -INFINITY, sink_dot = 0;
            for (unsigned d = 0; d < IQ_HEAD; d++) {
                sink_dot += static_cast<double>(query[base + d]) * sink[kh * IQ_HEAD + d];
            }
            for (unsigned token = begin; token <= end; token++) {
                double score = 0;
                for (unsigned d = 0; d < IQ_HEAD; d++) {
                    score += query[base + d] * source_q8(cache, token, kh, d, opt.capacity, 0);
                }
                scores[token - begin] = score * scale;
                maximum = std::max(maximum, scores[token - begin]);
            }
            double total = 0, values[IQ_HEAD] = {};
            for (unsigned token = begin; token <= end; token++) {
                const double weight = std::exp(scores[token - begin] - maximum);
                total += weight;
                for (unsigned d = 0; d < IQ_HEAD; d++) {
                    values[d] += weight * source_q8(cache, token, kh, d, opt.capacity, IQ_Q8_ROW_BLOCKS / 2);
                }
            }
            const double lse = maximum + std::log(total);
            const double difference = sink_dot * scale - lse;
            const double e = std::exp(-std::fabs(difference));
            const float factor = static_cast<float>(difference > 0 ? e / (1 + e) : 1 / (1 + e));
            check.source_lse.add(check.lse[row * IQ_HEADS + head], lse, row * IQ_HEADS + head);
            for (unsigned d = 0; d < IQ_HEAD; d++) {
                const float ordinary = iquest_bf16(static_cast<float>(values[d] / total));
                const float final = iquest_bf16(ordinary * factor);
                check.source_ordinary.add(check.ordinary[base + d], ordinary, base + d);
                check.source_final.add(output[base + d], final, base + d);
                check.retained_final.add(retained[base + d], final, base + d);
            }
        }
    }
}

static TiledCheck check_tiled(const std::vector<float> &query,
        const std::vector<float> &sink, const std::vector<iquest_q8> &cache,
        const std::vector<unsigned> &positions, const std::vector<float> &output,
        const std::vector<float> &retained, const float *dq, const float *ds,
        const iquest_q8 *dkv, const unsigned *dp, const Options &opt) {
    TiledCheck check;
    check.ordinary.resize(output.size()); check.sink_output.resize(output.size());
    check.lse.resize((size_t)opt.rows * IQ_HEADS);
    float *work = nullptr, *lse = nullptr;
    CUDA_OK(cudaMalloc(&work, output.size() * sizeof(float)));
    CUDA_OK(cudaMalloc(&lse, check.lse.size() * sizeof(float)));
    CUDA_OK(cudaMemset(work, 0xff, output.size() * sizeof(float)));
    CUDA_OK(cudaMemset(lse, 0xff, check.lse.size() * sizeof(float)));
    iq_prefill::prefill<<<dim3((opt.rows + iq_prefill::TQ - 1) / iq_prefill::TQ, IQ_HEADS), iq_prefill::THREADS>>>(
        work, lse, dq, dkv, dp, opt.rows, opt.capacity, opt.window);
    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(check.ordinary.data(), work, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(check.lse.data(), lse, check.lse.size() * sizeof(float), cudaMemcpyDeviceToHost));
    sink_reference<<<dim3(opt.rows, IQ_HEADS), IQ_HEAD>>>(work, lse, dq, ds);
    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(check.sink_output.data(), work, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(work)); CUDA_OK(cudaFree(lse));
    for (size_t i = 0; i < output.size(); i++) {
        uint32_t bits, expected_bits, output_bits;
        std::memcpy(&bits, &check.ordinary[i], sizeof(bits));
        std::memcpy(&expected_bits, &check.sink_output[i], sizeof(bits));
        std::memcpy(&output_bits, &output[i], sizeof(bits));
        check.ordinary_nonfinite += !std::isfinite(check.ordinary[i]);
        check.ordinary_non_bf16 += (bits & 0xffffu) != 0;
        check.sink_different += expected_bits != output_bits;
        check.full_drift.add(output[i], retained[i], i);
    }
    for (float value : check.lse) { check.lse_nonfinite += !std::isfinite(value); }
    check.components = check_components(query, cache);
    source_samples(check, query, sink, cache, positions, output, retained, opt);
    return check;
}

static void print_tiled(const TiledCheck &check, const Options &opt) {
    std::printf("\"tiled_numerics\":{\"gate\":\"finite_storage_components_and_exact_sink_stage_only\","
        "\"numerical_qualification\":\"requires_whole_model_review\",\"numerical_comparison_passed\":null,"
        "\"prefill_grid\":[%u,%u,1],\"prefill_block\":[%u,1,1],\"timed_kernels_per_iteration\":2,"
        "\"untimed_diagnostic_kernels\":2,\"ordinary_readback_elements\":%zu,\"lse_readback_elements\":%zu,"
        "\"ordinary_nonfinite\":%zu,\"ordinary_non_bf16\":%zu,\"lse_nonfinite\":%zu,"
        "\"sink_reference_different_bits\":%zu,\"ordinary_fnv1a64\":\"%016llx\",\"lse_fnv1a64\":\"%016llx\","
        "\"source_oracle\":\"FP64 stable ordinary softmax and LSE from serialized Q8; F32 sink; both BF16 boundaries\","
        "\"sample_rows\":[",
        (opt.rows + iq_prefill::TQ - 1) / iq_prefill::TQ, IQ_HEADS, iq_prefill::THREADS,
        check.ordinary.size(), check.lse.size(), check.ordinary_nonfinite, check.ordinary_non_bf16,
        check.lse_nonfinite, check.sink_different,
        (unsigned long long)hash_bytes(check.ordinary.data(), check.ordinary.size() * sizeof(float)),
        (unsigned long long)hash_bytes(check.lse.data(), check.lse.size() * sizeof(float)));
    for (size_t i = 0; i < check.sample_rows.size(); i++) {
        std::printf("%s%u", i ? "," : "", check.sample_rows[i]);
    }
    std::printf("],\"sample_heads_each\":%u,\"sample_dimensions_each\":%u,", IQ_HEADS, IQ_HEAD);
    print_drift("final_vs_retained_full", check.full_drift);
    print_drift("ordinary_vs_source_sampled", check.source_ordinary);
    print_drift("lse_vs_source_sampled", check.source_lse, UlpScale::None);
    print_drift("final_vs_source_sampled", check.source_final);
    print_drift("retained_vs_source_sampled", check.retained_final);
    std::printf("\"tf32_components\":{\"rounding\":\"RNA_hi_plus_RNA_F32_residual\","
        "\"q8_values\":%zu,\"pair_inexact\":%zu,\"max_pair_error\":%.17g,\"nonfinite\":%zu,"
        "\"query_inexact\":%zu,\"probability_pair_exactness_claimed\":false}},",
        check.components.values, check.components.pair_inexact, check.components.max_pair_error,
        check.components.nonfinite, check.components.query_inexact);
}

int main(int argc, char **argv) {
    const Options opt = options(argc, argv);
    const unsigned first = opt.position + 1 - opt.rows;
    const unsigned begin = opt.window && first + 1 > opt.window ? first + 1 - opt.window : 0;
    const size_t row_bytes = IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8);
    const size_t count = static_cast<size_t>(opt.rows) * IQ_HEADS * IQ_HEAD;
    std::vector<float> query(count), sink(IQ_KV_HEADS * IQ_HEAD);
    std::vector<float> key(sink.size()), value(sink.size()), output(count);
    std::vector<unsigned> positions(opt.rows);
    std::vector<iquest_q8> cache(static_cast<size_t>(opt.capacity) * IQ_Q8_ROW_BLOCKS);
    for (size_t i = 0; i < query.size(); i++) { query[i] = fixture_value(i + 71, opt.fixture); }
    if (opt.fixture == Fixture::Q8Stress) {
        // BF16 queries can be finite outside FP16's range. Powers of two
        // retain their BF16 mantissas while covering underflow and overflow.
        constexpr float query_scales[] = {0x1p-40f, 1.0f, 0x1p20f};
        for (size_t i = 0; i < query.size(); i++) {
            query[i] *= query_scales[(i / IQ_HEAD) % std::size(query_scales)];
        }
    }
    for (size_t i = 0; i < sink.size(); i++) { sink[i] = fixture_sink(i + 37, opt.fixture); }
    for (unsigned row = 0; row < opt.rows; row++) {
        // An odd multiplier permutes the 128 positions without changing
        // their union of required history in the production sliding ring.
        const unsigned index = opt.positions == Positions::Permuted ? (row * 73) % opt.rows : row;
        positions[row] = first + index;
    }
    for (unsigned token = begin; token <= opt.position; token++) {
        for (unsigned d = 0; d < key.size(); d++) {
            key[d] = fixture_value(d + token * 3, opt.fixture);
            value[d] = fixture_value(d + token * 7, opt.fixture);
        }
        iquest_store(cache.data(), key.data(), value.data(), token, opt.capacity);
    }
    if (opt.fixture == Fixture::Q8Stress) { stress_cache(cache, begin, opt); }

    CUDA_OK(cudaSetDevice(static_cast<int>(opt.device)));
    cudaDeviceProp properties{};
    CUDA_OK(cudaGetDeviceProperties(&properties, static_cast<int>(opt.device)));
    float *dq = upload(query), *ds = upload(sink), *out = nullptr, *dlse = nullptr;
    unsigned *dp = upload(positions);
    iquest_q8 *dkv = upload(cache);
    CUDA_OK(cudaMalloc(&out, count * sizeof(float)));
    const size_t lse_bytes = opt.implementation == Implementation::Tiled
        ? (size_t)opt.rows * IQ_HEADS * sizeof(float) : 0;
    if (lse_bytes) {
        CUDA_OK(cudaMalloc(&dlse, lse_bytes));
        CUDA_OK(cudaMemset(dlse, 0xff, lse_bytes));
    }
    // A missing write must remain nonfinite rather than pass as a zero result.
    CUDA_OK(cudaMemset(out, 0xff, count * sizeof(float)));
    const auto launch = [&]() {
        launch_attention(opt.implementation, out, dq, dkv, ds, dp, opt, dlse);
    };
    for (unsigned i = 0; i < opt.warmup; i++) { launch(); }
    CUDA_OK(cudaDeviceSynchronize());
    cudaEvent_t start, end;
    CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&end));
    CUDA_OK(cudaEventRecord(start));
    for (unsigned i = 0; i < opt.repeat; i++) { launch(); }
    CUDA_OK(cudaEventRecord(end)); CUDA_OK(cudaEventSynchronize(end));
    float elapsed_ms = 0;
    CUDA_OK(cudaEventElapsedTime(&elapsed_ms, start, end));
    CUDA_OK(cudaMemcpy(output.data(), out, count * sizeof(float), cudaMemcpyDeviceToHost));

    // Grouped P2 compares against retained P1. Existing baseline/reduced
    // pair semantics and all fixture inputs remain unchanged.
    const Implementation opposite = opt.implementation == Implementation::Async
        ? Implementation::Cache128 : opt.implementation == Implementation::Tiled
        ? Implementation::Grouped : opt.implementation != Implementation::Reduced
        ? Implementation::Reduced : Implementation::Baseline;
    Comparison other, additional;
    if (opt.compare) {
        other = compare_output(opposite, output, dq, dkv, ds, dp, opt);
    }
    if (opt.compare_all) {
        const Implementation third = opt.implementation >= Implementation::Grouped
            ? Implementation::Baseline : Implementation::Grouped;
        const auto &reference = opt.implementation == Implementation::Tiled ? other.output : output;
        additional = compare_output(third, reference, dq, dkv, ds, dp, opt);
    }
    TiledCheck tiled;
    if (opt.implementation == Implementation::Tiled) {
        tiled = check_tiled(query, sink, cache, positions, output, other.output, dq, ds, dkv, dp, opt);
    }
    if (opt.dump_prefix) {
        dump_output(opt.dump_prefix, opt.implementation, output);
        if (opt.compare) {
            dump_output(opt.dump_prefix, opposite, other.output);
        }
        if (opt.compare_all) {
            dump_output(opt.dump_prefix, additional.implementation, additional.output);
        }
        dump_bytes(opt.dump_prefix, ".positions.u32", positions.data(), positions.size() * sizeof(unsigned));
        if (opt.implementation == Implementation::Tiled) {
            dump_bytes(opt.dump_prefix, ".tiled-ordinary.f32", tiled.ordinary.data(), count * sizeof(float));
            dump_bytes(opt.dump_prefix, ".tiled-lse.f32", tiled.lse.data(), lse_bytes);
            dump_bytes(opt.dump_prefix, ".tiled-sink-reference.f32", tiled.sink_output.data(), count * sizeof(float));
        }
    }

    size_t nonfinite = 0, non_bf16 = 0, nonzero = 0, unequal = 0, excess_error = 0;
    float max_error = 0, max_ulps = 0;
    for (float actual : output) {
        uint32_t bits;
        std::memcpy(&bits, &actual, sizeof(bits));
        nonfinite += !std::isfinite(actual);
        non_bf16 += (bits & 0xffffu) != 0;
        nonzero += actual != 0;
    }
    float agreement = 0;
    if (opt.fixture == Fixture::Reference13) {
        std::vector<float> expected(count);
        iquest_attn(expected.data(), query.data(), cache.data(), sink.data(), opt.position, opt.capacity, opt.window);
        for (size_t i = 0; i < count; i++) {
            unequal += output[i] != expected[i];
            const float delta = std::fabs(output[i] - expected[i]);
            const float ulp = std::max(bf16_ulp(output[i]), bf16_ulp(expected[i]));
            max_error = std::max(max_error, delta);
            max_ulps = std::max(max_ulps, ulp ? delta / ulp : 0.0f);
            excess_error += delta > std::max(ulp, kCancellationAtol);
        }
        agreement = 1.0f - static_cast<float>(unequal) / count;
    }
    const bool comparison_passed = opt.implementation == Implementation::Tiled
        ? !other.nonfinite && !other.non_bf16 && tiled.passed() : other.passed();
    const bool passed = !nonfinite && !non_bf16 && nonzero && !excess_error &&
        comparison_passed && additional.passed() &&
        (opt.fixture != Fixture::Reference13 || agreement >= kMinAgreement);
    const size_t device_bytes = cache.size() * sizeof(iquest_q8) + (count * 2 + sink.size()) * sizeof(float) + positions.size() * sizeof(unsigned) + lse_bytes;
    const size_t peak_device_bytes = device_bytes + (opt.compare ? count * sizeof(float) : 0) + lse_bytes;
    std::printf("{\"kernel\":\"%s\",\"implementation\":\"%s\","
        "\"fixture\":\"%s\",\"scope\":\"synthetic_resident_attention_only\","
        "\"device\":%u,\"compute_capability\":\"%d.%d\",\"rows\":%u,"
        "\"q_heads\":%u,\"kv_heads\":%u,\"head_dim\":%u,\"first_position\":%u,"
        "\"last_position\":%u,\"window\":%u,\"capacity\":%u,\"cache_begin\":%u,"
        "\"cache_rows_populated\":%u,\"ring_wrapped\":%s,\"q8_row_bytes\":%zu,"
        "\"query_storage\":\"%s\",\"sink_storage\":\"%s\",\"production_sink_storage\":\"f32\","
        "\"sink\":\"synthetic_key_zero_value\",\"stream\":\"default\",\"surrounding_graph\":false,"
        "\"positions\":\"%s\",\"positions_fnv1a64\":\"%016llx\","
        "\"grid\":[%u,%u,1],\"block\":[%u,1,1],\"device_bytes\":%zu,\"peak_device_bytes\":%zu,"
        "\"warmup_launches\":%u,\"timed_launches\":%u,\"event_total_ms\":%.9g,\"event_mean_ms\":%.9g,"
        "\"query_fnv1a64\":\"%016llx\",\"sink_fnv1a64\":\"%016llx\","
        "\"cache_fnv1a64\":\"%016llx\",\"output_fnv1a64\":\"%016llx\","
        "\"readback_elements\":%zu,\"nonfinite\":%zu,\"non_bf16\":%zu,\"nonzero\":%zu,",
        kernel_name(opt.implementation),
        impl_name(opt.implementation),
        opt.fixture == Fixture::Reference13 ? "reference13" :
            opt.fixture == Fixture::F32Sink ? "f32sink_hash_v2" :
            opt.fixture == Fixture::Q8Stress ? "q8_extreme_f32sink_v1" : "long_bf16_hash_v1",
        opt.device, properties.major, properties.minor, opt.rows,
        IQ_HEADS, IQ_KV_HEADS, IQ_HEAD, first, opt.position, opt.window, opt.capacity, begin,
        opt.position + 1 - begin, opt.position >= opt.capacity ? "true" : "false", row_bytes,
        opt.fixture == Fixture::Reference13 ? "f32_existing_reference" : "bf16_in_f32",
        opt.fixture == Fixture::Long ? "bf16_in_f32" : "f32",
        opt.positions == Positions::Permuted ? "permuted" : "contiguous",
        static_cast<unsigned long long>(hash_bytes(positions.data(), positions.size() * sizeof(unsigned))),
        opt.implementation == Implementation::Tiled ? (opt.rows + iq_prefill::TQ - 1) / iq_prefill::TQ : opt.rows,
        grid_heads(opt.implementation), IQ_HEAD, device_bytes, peak_device_bytes, opt.warmup, opt.repeat, elapsed_ms, elapsed_ms / opt.repeat,
        static_cast<unsigned long long>(hash_bytes(query.data(), query.size() * sizeof(float))),
        static_cast<unsigned long long>(hash_bytes(sink.data(), sink.size() * sizeof(float))),
        static_cast<unsigned long long>(hash_bytes(cache.data(), cache.size() * sizeof(iquest_q8))),
        static_cast<unsigned long long>(hash_bytes(output.data(), output.size() * sizeof(float))),
        count, nonfinite, non_bf16, nonzero);
    const bool grouped_eligible = opt.rows == IQ_PREFILL && (opt.window == 0 || opt.window == IQ_WINDOW);
    std::printf("\"grouped_dispatch\":{\"shape_eligible\":%s,"
        "\"rule\":\"rows128_and_window_full_or_swa\",\"heads_per_block\":%u,"
        "\"query_heads_per_kv\":%u,\"all_query_heads_read_back\":%u,"
        "\"selected_outside_production_scope\":%s},",
        grouped_eligible ? "true" : "false", IQ_ATTN_HEADS_PER_BLOCK, IQ_HEADS / IQ_KV_HEADS, IQ_HEADS,
        opt.implementation == Implementation::Grouped && !grouped_eligible ? "true" : "false");
    if (opt.fixture == Fixture::Reference13) {
        // Same oracle as test_iquest_primitives: the CPU reduction is serial,
        // so BF16 ties and FP32 cancellation do not imply bitwise equivalence.
        std::printf("\"cpu_reference\":{\"source\":\"test_iquest_primitives:attn_test(12,16,0)\","
            "\"element_agreement\":%.9g,\"min_agreement\":%.9g,\"max_error\":%.9g,"
            "\"max_bf16_ulps\":%.9g,\"cancellation_atol\":%.9g,\"excess_error\":%zu},",
            agreement, kMinAgreement, max_error, max_ulps, kCancellationAtol, excess_error);
    } else { std::printf("\"cpu_reference\":null,"); }
    if (opt.compare) {
        print_comparison("comparison", other);
    } else { std::printf("\"comparison\":null,"); }
    if (opt.compare_all) { print_comparison("additional_comparison", additional); }
    else { std::printf("\"additional_comparison\":null,"); }
    if (opt.implementation == Implementation::Tiled) {
        std::printf("\"comparison_policy\":\"tiled_vs_grouped_drift; additional_grouped_vs_baseline_exact\",");
        print_tiled(tiled, opt);
    }
    std::printf("\"raw_output_saved\":%s,\"passed\":%s}\n", opt.dump_prefix ? "true" : "false", passed ? "true" : "false");
    CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(end));
    CUDA_OK(cudaFree(dq)); CUDA_OK(cudaFree(ds)); CUDA_OK(cudaFree(out));
    CUDA_OK(cudaFree(dp)); CUDA_OK(cudaFree(dkv));
    if (dlse) { CUDA_OK(cudaFree(dlse)); }
    return passed ? 0 : 1;
}
