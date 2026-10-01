// Synthetic resident logits; exact production router geometry and kernel.
// Build with production NVCCFLAGS, including --use_fast_math and normal FMA.
// NCU: ncu --clock-control none --kernel-name regex:iquest_router_kernel \
//   --launch-count 1 ./tests/iquest_router_profile --rows 1 --case finite \
//   --warmup 0 --repeat 1
#include <cuda_runtime.h>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "../cuda/iquest_primitives.cuh"
#include "../cuda/iquest_router.cuh"

#define CUDA_OK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        std::exit(2); \
    } \
} while (0)

enum class Fixture {
    Finite, Negative, Ties, NearTies, SignedZero,
    PositiveInf, NegativeInf, NanFirst, NanMiddle, NanLast, NanAll
};
struct NamedFixture { const char *name; Fixture fixture; };
constexpr NamedFixture kFixtures[] = {
    {"finite", Fixture::Finite}, {"negative", Fixture::Negative},
    {"ties", Fixture::Ties}, {"near_ties", Fixture::NearTies},
    {"signed_zero", Fixture::SignedZero}, {"positive_inf", Fixture::PositiveInf},
    {"negative_inf", Fixture::NegativeInf}, {"nan_first", Fixture::NanFirst},
    {"nan_middle", Fixture::NanMiddle}, {"nan_last", Fixture::NanLast},
    {"nan_all", Fixture::NanAll}
};
constexpr unsigned kMaxLaunches = 128;
constexpr uint32_t kFloatOneBits = 0x3f800000u;
constexpr uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);

enum class Implementation { Baseline, Warp };
struct Options {
    Implementation implementation = Implementation::Baseline;
    unsigned rows = 1;
    unsigned warmup = 1;
    unsigned repeat = 8;
    unsigned device = 0;
    const char *fixture = "finite";
};

static void fail(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(2);
}

static void usage() {
    std::fprintf(stderr,
        "Usage: iquest_router_profile [--rows 1|128] [--case NAME|all]\n"
        "  [--implementation baseline|warp] [--warmup 0..128] [--repeat 1..128] [--device N]\n"
        "Cases: finite negative ties near_ties signed_zero positive_inf\n"
        "       negative_inf nan_first nan_middle nan_last nan_all\n"
        "All IDs and weight bits must match; nonfinite cases record existing\n"
        "unsanitized behavior, not a valid-model-input guarantee.\n"
        "Use --warmup 0 --repeat 1 --case finite for one NCU launch.\n");
}

static unsigned number(const char *value) {
    char *end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno || !*value || *end || *value == '-' || parsed > UINT32_MAX) {
        fail("Expected an unsigned 32-bit integer");
    }
    return static_cast<unsigned>(parsed);
}

static Options options(int argc, char **argv) {
    Options result;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!std::strcmp(arg, "--help")) { usage(); std::exit(0); }
        if (++i == argc) { usage(); fail("Missing option value"); }
        const char *value = argv[i];
        if (!std::strcmp(arg, "--rows")) { result.rows = number(value); }
        else if (!std::strcmp(arg, "--warmup")) { result.warmup = number(value); }
        else if (!std::strcmp(arg, "--repeat")) { result.repeat = number(value); }
        else if (!std::strcmp(arg, "--device")) { result.device = number(value); }
        else if (!std::strcmp(arg, "--case")) { result.fixture = value; }
        else if (!std::strcmp(arg, "--implementation")) {
            if (!std::strcmp(value, "baseline")) { result.implementation = Implementation::Baseline; }
            else if (!std::strcmp(value, "warp")) { result.implementation = Implementation::Warp; }
            else { fail("Unknown implementation"); }
        }
        else { usage(); fail("Unknown option"); }
    }
    if (result.rows != 1 && result.rows != IQ_PREFILL) { fail("Rows must be 1 or 128"); }
    if (!result.repeat || result.repeat > kMaxLaunches || result.warmup > kMaxLaunches) {
        fail("Warmup/repeat exceeds the bounded launch count");
    }
    if (result.device > INT32_MAX) { fail("Device ordinal is too large"); }
    bool known = !std::strcmp(result.fixture, "all");
    for (const auto &entry : kFixtures) { known |= !std::strcmp(result.fixture, entry.name); }
    if (!known) { fail("Unknown fixture"); }
    return result;
}

static uint64_t hash_bytes(const void *data, size_t bytes) {
    const auto *input = static_cast<const unsigned char *>(data);
    uint64_t hash = kFnvOffset;
    for (size_t i = 0; i < bytes; i++) { hash = (hash ^ input[i]) * kFnvPrime; }
    return hash;
}

static float score(unsigned row, unsigned expert, Fixture fixture) {
    unsigned mixed = row * IQ_EXPERTS + expert;
    mixed ^= mixed >> 16; mixed *= 0x7feb352du;
    mixed ^= mixed >> 15; mixed *= 0x846ca68bu; mixed ^= mixed >> 16;
    const float finite = (static_cast<int>(mixed & 0x7ffu) - 1024) / 16.0f;
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    switch (fixture) {
    case Fixture::Finite: return finite;
    case Fixture::Negative: return -1.0f - static_cast<float>((expert + row) % IQ_EXPERTS);
    case Fixture::Ties: return -1.0f;
    case Fixture::NearTies: {
        // Adjacent representable scores distinguish exact selection from
        // a rounded/low-precision top-k implementation.
        const uint32_t bits = kFloatOneBits + ((expert + row) % (IQ_USED * 2));
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }
    case Fixture::SignedZero: return expert % 2 ? -0.0f : 0.0f;
    case Fixture::PositiveInf: return expert == row % IQ_EXPERTS ? infinity : finite;
    case Fixture::NegativeInf: return -infinity;
    case Fixture::NanFirst: return expert == 0 ? nan : finite;
    case Fixture::NanMiddle: return expert == IQ_EXPERTS / 2 ? nan : finite;
    case Fixture::NanLast: return expert == IQ_EXPERTS - 1 ? nan : finite;
    case Fixture::NanAll: return nan;
    }
    return nan;
}

template<class T> static T *upload(const std::vector<T> &input) {
    T *device = nullptr;
    CUDA_OK(cudaMalloc(&device, input.size() * sizeof(T)));
    CUDA_OK(cudaMemcpy(device, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

// IDs come from the independent native CPU oracle. Normalize them on the
// GPU to preserve production expf/division and the ascending eight-term sum;
// CPU libm is not an exact oracle for CUDA --use_fast_math weight bits.
__global__ static void router_weights_ref(float *weights, const unsigned *ids,
                                          const float *logits) {
    const unsigned row = blockIdx.x;
    float *weight = weights + static_cast<uint64_t>(row) * IQ_USED;
    const unsigned *selected = ids + static_cast<uint64_t>(row) * IQ_USED;
    const float *scores = logits + static_cast<uint64_t>(row) * IQ_EXPERTS;
    float sum = 0;
    for (unsigned k = 0; k < IQ_USED; k++) {
        weight[k] = expf(scores[selected[k]] - scores[selected[0]]);
        sum += weight[k];
    }
    for (unsigned k = 0; k < IQ_USED; k++) { weight[k] /= sum; }
}

static bool run_case(const Options &opt, const NamedFixture &test,
                     const cudaDeviceProp &properties) {
    const size_t selected_count = static_cast<size_t>(opt.rows) * IQ_USED;
    std::vector<float> logits(static_cast<size_t>(opt.rows) * IQ_EXPERTS);
    std::vector<float> expected_weights(selected_count), actual_weights(selected_count);
    std::vector<unsigned> expected_ids(selected_count), actual_ids(selected_count);
    for (unsigned row = 0; row < opt.rows; row++) {
        for (unsigned expert = 0; expert < IQ_EXPERTS; expert++) {
            logits[static_cast<size_t>(row) * IQ_EXPERTS + expert] = score(row, expert, test.fixture);
        }
        iquest_router(expected_ids.data() + static_cast<size_t>(row) * IQ_USED,
            expected_weights.data() + static_cast<size_t>(row) * IQ_USED,
            logits.data() + static_cast<size_t>(row) * IQ_EXPERTS);
    }

    float *input = upload(logits), *weights = nullptr, *reference = nullptr;
    unsigned *ids = nullptr, *reference_ids = upload(expected_ids);
    CUDA_OK(cudaMalloc(&weights, selected_count * sizeof(float)));
    CUDA_OK(cudaMalloc(&reference, selected_count * sizeof(float)));
    CUDA_OK(cudaMalloc(&ids, selected_count * sizeof(unsigned)));
    CUDA_OK(cudaMemset(weights, 0xff, selected_count * sizeof(float)));
    CUDA_OK(cudaMemset(ids, 0xff, selected_count * sizeof(unsigned)));
    const auto launch = [&]() {
        if (opt.implementation == Implementation::Warp) {
            iq_router::select<<<opt.rows, iq_router::WARP>>>(ids, weights, input);
        } else {
            iquest_router_kernel<<<opt.rows, 1>>>(ids, weights, input);
        }
        CUDA_OK(cudaGetLastError());
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
    CUDA_OK(cudaMemcpy(actual_ids.data(), ids, selected_count * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(actual_weights.data(), weights, selected_count * sizeof(float), cudaMemcpyDeviceToHost));

    router_weights_ref<<<opt.rows, 1>>>(reference, reference_ids, input);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(expected_weights.data(), reference, selected_count * sizeof(float), cudaMemcpyDeviceToHost));
    size_t id_mismatch = 0, weight_mismatch = 0, nonfinite_input = 0, nonfinite_weight = 0;
    size_t tie_contract_errors = 0;
    for (float value : logits) { nonfinite_input += !std::isfinite(value); }
    for (size_t i = 0; i < selected_count; i++) {
        id_mismatch += actual_ids[i] != expected_ids[i];
        weight_mismatch += std::memcmp(&actual_weights[i], &expected_weights[i], sizeof(float)) != 0;
        nonfinite_weight += !std::isfinite(actual_weights[i]);
        if (test.fixture == Fixture::Ties || test.fixture == Fixture::SignedZero) {
            // Equal scores select lower expert IDs; selected softmax is 1/8.
            const float uniform = 1.0f / IQ_USED;
            tie_contract_errors += actual_ids[i] != i % IQ_USED ||
                std::memcmp(&actual_weights[i], &uniform, sizeof(float)) != 0;
        }
    }
    const bool passed = !id_mismatch && !weight_mismatch && !tie_contract_errors &&
        (nonfinite_input || !nonfinite_weight);
    std::printf("{\"kernel\":\"%s\",\"implementation\":\"%s\","
        "\"fixture\":\"%s\",\"scope\":\"synthetic_resident_router_only\","
        "\"device\":%u,\"compute_capability\":\"%d.%d\",\"rows\":%u,\"experts\":%u,\"used\":%u,"
        "\"grid\":[%u,1,1],\"block\":[%u,1,1],\"input_storage\":\"f32\","
        "\"input_sanitized\":false,\"stream\":\"default\",\"surrounding_graph\":false,"
        "\"warmup_launches\":%u,\"timed_launches\":%u,\"event_total_ms\":%.9g,\"event_mean_ms\":%.9g,"
        "\"input_fnv1a64\":\"%016llx\",\"ids_fnv1a64\":\"%016llx\",\"weights_fnv1a64\":\"%016llx\","
        "\"reference_ids_fnv1a64\":\"%016llx\",\"reference_weights_fnv1a64\":\"%016llx\","
        "\"readback_elements_per_output\":%zu,\"id_mismatch\":%zu,\"weight_bit_mismatch\":%zu,"
        "\"nonfinite_input\":%zu,\"nonfinite_weight\":%zu,\"tie_contract_errors\":%zu,"
        "\"id_reference\":\"native_cpu_iquest_router\",\"weight_reference\":\"cuda_serial_selected_softmax\","
        "\"ids_exact\":%s,\"weights_bit_exact\":%s,\"passed\":%s}\n",
        opt.implementation == Implementation::Warp ? "iq_router::select" : "iquest_router_kernel",
        opt.implementation == Implementation::Warp ? "warp" : "baseline",
        test.name, opt.device, properties.major, properties.minor, opt.rows,
        static_cast<unsigned>(IQ_EXPERTS), static_cast<unsigned>(IQ_USED),
        opt.rows, opt.implementation == Implementation::Warp ? iq_router::WARP : 1, opt.warmup, opt.repeat, elapsed_ms, elapsed_ms / opt.repeat,
        static_cast<unsigned long long>(hash_bytes(logits.data(), logits.size() * sizeof(float))),
        static_cast<unsigned long long>(hash_bytes(actual_ids.data(), selected_count * sizeof(unsigned))),
        static_cast<unsigned long long>(hash_bytes(actual_weights.data(), selected_count * sizeof(float))),
        static_cast<unsigned long long>(hash_bytes(expected_ids.data(), selected_count * sizeof(unsigned))),
        static_cast<unsigned long long>(hash_bytes(expected_weights.data(), selected_count * sizeof(float))),
        selected_count, id_mismatch, weight_mismatch, nonfinite_input, nonfinite_weight, tie_contract_errors,
        id_mismatch ? "false" : "true", weight_mismatch ? "false" : "true", passed ? "true" : "false");

    CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(end));
    CUDA_OK(cudaFree(input)); CUDA_OK(cudaFree(weights)); CUDA_OK(cudaFree(reference));
    CUDA_OK(cudaFree(ids)); CUDA_OK(cudaFree(reference_ids));
    return passed;
}

int main(int argc, char **argv) {
    const Options opt = options(argc, argv);
    CUDA_OK(cudaSetDevice(static_cast<int>(opt.device)));
    cudaDeviceProp properties{};
    CUDA_OK(cudaGetDeviceProperties(&properties, static_cast<int>(opt.device)));
    bool passed = true;
    for (const auto &test : kFixtures) {
        if (std::strcmp(opt.fixture, "all") && std::strcmp(opt.fixture, test.name)) { continue; }
        passed = run_case(opt, test, properties) && passed;
    }
    return passed ? 0 : 1;
}
