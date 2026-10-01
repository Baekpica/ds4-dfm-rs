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

#define CUDA_OK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        std::exit(2); \
    } \
} while (0)

enum class Fixture { Long, Reference13, F32Sink };
enum class Implementation { Baseline, Reduced };
enum class Positions { Contiguous, Permuted };
constexpr unsigned kMaxCapacity = 8192;
constexpr unsigned kSwaCapacity = IQ_WINDOW + IQ_PREFILL - 1;
constexpr unsigned kMaxLaunches = 16;
constexpr float kCancellationAtol = 2e-6f;
constexpr float kMinAgreement = 0.99f;

struct Options {
    Fixture fixture = Fixture::Long;
    Implementation implementation = Implementation::Baseline;
    Positions positions = Positions::Contiguous;
    bool compare = false;
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
        "Usage: iquest_attention_profile [--case long|reference13|f32sink]\n"
        "  [--implementation baseline|reduced] [--compare] [--dump-prefix PATH]\n"
        "  [--rows 1|128] [--positions contiguous|permuted]\n"
        "  [--position 127|511|512|518|519|2047|4095|4096|4222|4223|8191|8445|8446]\n"
        "  [--window full|swa|mtp] [--capacity 519|4223|8192]\n"
        "  [--warmup 0..16] [--repeat 1..16] [--device N]\n"
        "reference13 fixes rows=1, position=12, full attention, capacity=16.\n"
        "f32sink uses nonsymmetric non-BF16 F32 sink values.\n"
        "--compare requires byte-exact baseline/reduced output across every row.\n"
        "--dump-prefix saves .baseline.f32/.reduced.f32 and .positions.u32.\n"
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
        if (++i == argc) { usage(); fail("Missing option value"); }
        const char *value = argv[i];
        if (!std::strcmp(arg, "--case")) {
            if (!std::strcmp(value, "long")) { result.fixture = Fixture::Long; }
            else if (!std::strcmp(value, "reference13")) { result.fixture = Fixture::Reference13; }
            else if (!std::strcmp(value, "f32sink")) { result.fixture = Fixture::F32Sink; }
            else { fail("Unknown fixture"); }
        } else if (!std::strcmp(arg, "--implementation")) {
            if (!std::strcmp(value, "baseline")) { result.implementation = Implementation::Baseline; }
            else if (!std::strcmp(value, "reduced")) { result.implementation = Implementation::Reduced; }
            else { fail("Implementation must be baseline or reduced"); }
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
        if (shape_given) { fail("reference13 cannot override shape options"); }
        result.rows = 1; result.position = 12; result.window = 0; result.capacity = 16;
        return result;
    }
    if (result.rows != 1 && result.rows != IQ_PREFILL) { fail("Rows must be 1 or 128"); }
    const unsigned positions[] = {127, 511, 512, 518, 519, 2047, 4095, 4096,
                                  4222, 4223, 8191, 8445, 8446};
    if (std::find(std::begin(positions), std::end(positions), result.position) == std::end(positions)) {
        fail("Position must be a documented bounded window/ring case");
    }
    if (result.capacity != IQ_DRAFT_WINDOW + IQ_DRAFT_SLOTS &&
        result.capacity != kSwaCapacity && result.capacity != kMaxCapacity) {
        fail("Capacity must be 519, 4223, or 8192");
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
    if (fixture != Fixture::F32Sink) { return fixture_value(i, fixture); }
    // Source sink weights are F32. Nonzero low mantissa bits exercise an
    // accidental BF16 cast, with distinct keys across the eight KV heads.
    i ^= UINT32_C(0x9e3779b9); i *= UINT32_C(0x846ca68b); i ^= i >> 13;
    const uint32_t bits = (i & UINT32_C(0x807ffffe)) | UINT32_C(0x3e800001);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static const char *impl_name(Implementation implementation) {
    return implementation == Implementation::Baseline ? "baseline" : "reduced";
}

static void launch_attention(Implementation implementation, float *out,
        const float *query, const iquest_q8 *cache, const float *sink,
        const unsigned *positions, const Options &opt) {
    if (implementation == Implementation::Baseline) {
        iquest_attn_kernel<<<dim3(opt.rows, IQ_HEADS), IQ_HEAD>>>(
            out, query, cache, sink, positions, opt.capacity, opt.window);
    } else {
        iquest_attn_shuffle_kernel<<<dim3(opt.rows, IQ_HEADS), IQ_HEAD>>>(
            out, query, cache, sink, positions, opt.capacity, opt.window);
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

    CUDA_OK(cudaSetDevice(static_cast<int>(opt.device)));
    cudaDeviceProp properties{};
    CUDA_OK(cudaGetDeviceProperties(&properties, static_cast<int>(opt.device)));
    float *dq = upload(query), *ds = upload(sink), *out = nullptr;
    unsigned *dp = upload(positions);
    iquest_q8 *dkv = upload(cache);
    CUDA_OK(cudaMalloc(&out, count * sizeof(float)));
    // A missing write must remain nonfinite rather than pass as a zero result.
    CUDA_OK(cudaMemset(out, 0xff, count * sizeof(float)));
    const auto launch = [&]() {
        launch_attention(opt.implementation, out, dq, dkv, ds, dp, opt);
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

    // Separate poisoned outputs prevent skipped candidate writes from
    // inheriting baseline values. The comparison launch is outside timing.
    std::vector<float> other;
    size_t differing_bits = 0, other_nonfinite = 0, other_non_bf16 = 0;
    double compare_max_abs = 0;
    const Implementation opposite = opt.implementation == Implementation::Baseline
        ? Implementation::Reduced : Implementation::Baseline;
    if (opt.compare) {
        float *other_out = nullptr;
        CUDA_OK(cudaMalloc(&other_out, count * sizeof(float)));
        CUDA_OK(cudaMemset(other_out, 0xff, count * sizeof(float)));
        launch_attention(opposite, other_out, dq, dkv, ds, dp, opt);
        CUDA_OK(cudaDeviceSynchronize());
        other.resize(count);
        CUDA_OK(cudaMemcpy(other.data(), other_out, count * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaFree(other_out));
        for (size_t i = 0; i < count; i++) {
            uint32_t bits, other_bits;
            std::memcpy(&bits, &output[i], sizeof(bits));
            std::memcpy(&other_bits, &other[i], sizeof(other_bits));
            differing_bits += bits != other_bits;
            other_nonfinite += !std::isfinite(other[i]);
            other_non_bf16 += (other_bits & 0xffffu) != 0;
            if (std::isfinite(output[i]) && std::isfinite(other[i])) {
                compare_max_abs = std::max(compare_max_abs,
                    std::fabs(static_cast<double>(output[i]) - other[i]));
            }
        }
    }
    if (opt.dump_prefix) {
        dump_bytes(opt.dump_prefix, opt.implementation == Implementation::Baseline
            ? ".baseline.f32" : ".reduced.f32", output.data(), count * sizeof(float));
        if (opt.compare) {
            dump_bytes(opt.dump_prefix, opposite == Implementation::Baseline
                ? ".baseline.f32" : ".reduced.f32", other.data(), count * sizeof(float));
        }
        dump_bytes(opt.dump_prefix, ".positions.u32", positions.data(), positions.size() * sizeof(unsigned));
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
    const bool passed = !nonfinite && !non_bf16 && nonzero && !excess_error &&
        !differing_bits && !other_nonfinite && !other_non_bf16 &&
        (opt.fixture != Fixture::Reference13 || agreement >= kMinAgreement);
    const size_t device_bytes = cache.size() * sizeof(iquest_q8) + (count * 2 + sink.size()) * sizeof(float) + positions.size() * sizeof(unsigned);
    const size_t peak_device_bytes = device_bytes + (opt.compare ? count * sizeof(float) : 0);
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
        opt.implementation == Implementation::Baseline ? "iquest_attn_kernel" : "iquest_attn_shuffle_kernel",
        impl_name(opt.implementation),
        opt.fixture == Fixture::Reference13 ? "reference13" :
            opt.fixture == Fixture::F32Sink ? "f32sink_hash_v2" : "long_bf16_hash_v1",
        opt.device, properties.major, properties.minor, opt.rows,
        IQ_HEADS, IQ_KV_HEADS, IQ_HEAD, first, opt.position, opt.window, opt.capacity, begin,
        opt.position + 1 - begin, opt.position >= opt.capacity ? "true" : "false", row_bytes,
        opt.fixture == Fixture::Reference13 ? "f32_existing_reference" : "bf16_in_f32",
        opt.fixture == Fixture::Long ? "bf16_in_f32" : "f32",
        opt.positions == Positions::Permuted ? "permuted" : "contiguous",
        static_cast<unsigned long long>(hash_bytes(positions.data(), positions.size() * sizeof(unsigned))),
        opt.rows, IQ_HEADS, IQ_HEAD, device_bytes, peak_device_bytes, opt.warmup, opt.repeat, elapsed_ms, elapsed_ms / opt.repeat,
        static_cast<unsigned long long>(hash_bytes(query.data(), query.size() * sizeof(float))),
        static_cast<unsigned long long>(hash_bytes(sink.data(), sink.size() * sizeof(float))),
        static_cast<unsigned long long>(hash_bytes(cache.data(), cache.size() * sizeof(iquest_q8))),
        static_cast<unsigned long long>(hash_bytes(output.data(), output.size() * sizeof(float))),
        count, nonfinite, non_bf16, nonzero);
    if (opt.fixture == Fixture::Reference13) {
        // Same oracle as test_iquest_primitives: the CPU reduction is serial,
        // so BF16 ties and FP32 cancellation do not imply bitwise equivalence.
        std::printf("\"cpu_reference\":{\"source\":\"test_iquest_primitives:attn_test(12,16,0)\","
            "\"element_agreement\":%.9g,\"min_agreement\":%.9g,\"max_error\":%.9g,"
            "\"max_bf16_ulps\":%.9g,\"cancellation_atol\":%.9g,\"excess_error\":%zu},",
            agreement, kMinAgreement, max_error, max_ulps, kCancellationAtol, excess_error);
    } else { std::printf("\"cpu_reference\":null,"); }
    if (opt.compare) {
        std::printf("\"comparison\":{\"other_implementation\":\"%s\",\"untimed_launches\":1,\"elements\":%zu,"
            "\"byte_exact\":%s,\"different_elements\":%zu,\"max_abs\":%.9g,"
            "\"other_nonfinite\":%zu,\"other_non_bf16\":%zu,\"other_output_fnv1a64\":\"%016llx\"},",
            impl_name(opposite), count, differing_bits ? "false" : "true", differing_bits,
            compare_max_abs, other_nonfinite, other_non_bf16,
            static_cast<unsigned long long>(hash_bytes(other.data(), other.size() * sizeof(float))));
    } else { std::printf("\"comparison\":null,"); }
    std::printf("\"raw_output_saved\":%s,\"passed\":%s}\n", opt.dump_prefix ? "true" : "false", passed ? "true" : "false");
    CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(end));
    CUDA_OK(cudaFree(dq)); CUDA_OK(cudaFree(ds)); CUDA_OK(cudaFree(out));
    CUDA_OK(cudaFree(dp)); CUDA_OK(cudaFree(dkv));
    return passed ? 0 : 1;
}
