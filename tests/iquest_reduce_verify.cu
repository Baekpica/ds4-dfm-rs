// Compile with production NVCCFLAGS, including fast math/FMA. This compares
// the actual reducers; run compute-sanitizer --tool racecheck on this binary.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../cuda/iquest_primitives.cuh"

#define CUDA_OK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        std::exit(2); \
    } \
} while (0)

constexpr unsigned kCases = 64;
constexpr unsigned kBlocks = 32;
constexpr unsigned kRounds = 256;
enum LeafCase { PositiveZero, NegativeZero, MixedZero, Ones, Cancellation,
                Tiny, NormalEdge, PositiveInf, NegativeInf, OppositeInf, QuietNaN };
enum class Reduction { Shared, Shuffle };

static float from_bits(uint32_t bits) {
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

static std::vector<float> leaves() {
    std::vector<float> input(kCases * IQ_HEAD);
    uint32_t seed = 0x51a8b47du;
    for (unsigned row = 0; row < kCases; row++) {
        for (unsigned lane = 0; lane < IQ_HEAD; lane++) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            const uint32_t sign = (seed & 1u) << 31;
            const uint32_t exponent = (64u + ((seed >> 1) % 127u)) << 23;
            float value = from_bits(sign | exponent | ((seed >> 8) & 0x7fffffu));
            switch (row) {
                case PositiveZero: value = 0.0f; break;
                case NegativeZero: value = -0.0f; break;
                case MixedZero: value = lane & 1u ? -0.0f : 0.0f; break;
                case Ones: value = 1.0f; break;
                case Cancellation: value = 0.0f; break;
                case Tiny: value = from_bits(sign | (lane + 1)); break;
                case NormalEdge: value = from_bits(sign | (0x00800000u + lane)); break;
                case PositiveInf: value = lane ? 1.0f : INFINITY; break;
                case NegativeInf: value = lane ? 1.0f : -INFINITY; break;
                case OppositeInf: value = lane ? -INFINITY : INFINITY; break;
                case QuietNaN: value = lane ? 1.0f : from_bits(0x7fc12345u); break;
                default: break;
            }
            input[row * IQ_HEAD + lane] = value;
        }
    }
    // The original (+64, +32) leaf tree yields 1, while pairing +32 first
    // yields 2. This guards rounding order before the shuffle stages.
    float *cancel = input.data() + Cancellation * IQ_HEAD;
    cancel[0] = 16777216.0f; cancel[64] = 1.0f;
    cancel[32] = -16777216.0f; cancel[96] = 1.0f;
    return input;
}

template<Reduction Kind>
__global__ static void reduce_probe(float *out, const float *input) {
    __shared__ float scratch[IQ_HEAD];
    const unsigned lane = threadIdx.x;
    for (unsigned round = 0; round < kRounds; round++) {
        const unsigned row = (round + blockIdx.x * 17u) % kCases;
        const float value = input[row * IQ_HEAD + lane];
        float result;
        if constexpr (Kind == Reduction::Shared) { result = iq_reduce(value, scratch); }
        else { result = iq_reduce128(value, scratch); }
        // Every lane's broadcast result is observed, including other warps.
        out[((blockIdx.x * kRounds + round) * IQ_HEAD) + lane] = result;
    }
}

int main() {
    const std::vector<float> input = leaves();
    const size_t count = static_cast<size_t>(kBlocks) * kRounds * IQ_HEAD;
    std::vector<float> before(count), after(count);
    float *di = nullptr, *db = nullptr, *da = nullptr;
    CUDA_OK(cudaMalloc(&di, input.size() * sizeof(float)));
    CUDA_OK(cudaMalloc(&db, count * sizeof(float)));
    CUDA_OK(cudaMalloc(&da, count * sizeof(float)));
    CUDA_OK(cudaMemcpy(di, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    // Distinct finite sentinels make unwritten output slots fail comparison.
    CUDA_OK(cudaMemset(db, 0x7f, count * sizeof(float)));
    CUDA_OK(cudaMemset(da, 0x3f, count * sizeof(float)));
    reduce_probe<Reduction::Shared><<<kBlocks, IQ_HEAD>>>(db, di);
    CUDA_OK(cudaGetLastError());
    reduce_probe<Reduction::Shuffle><<<kBlocks, IQ_HEAD>>>(da, di);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(before.data(), db, count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(after.data(), da, count * sizeof(float), cudaMemcpyDeviceToHost));
    size_t mismatches = 0, nan_pairs = 0, cancellation_errors = 0;
    for (size_t i = 0; i < count; i++) {
        if (std::isnan(before[i]) && std::isnan(after[i])) { nan_pairs++; }
        else if (bits(before[i]) != bits(after[i])) {
            if (mismatches < 4) {
                std::fprintf(stderr, "index=%zu old=%08x new=%08x\n", i, bits(before[i]), bits(after[i]));
            }
            mismatches++;
        }
        const unsigned block = i / (kRounds * IQ_HEAD);
        const unsigned round = (i / IQ_HEAD) % kRounds;
        if ((round + block * 17u) % kCases == Cancellation) {
            cancellation_errors += before[i] != 1.0f || after[i] != 1.0f;
        }
    }
    const bool passed = !mismatches && !cancellation_errors;
    std::printf("{\"test\":\"iquest_reduce128\",\"cases\":%u,\"blocks\":%u,\"rounds\":%u,"
        "\"readback_elements\":%zu,\"bit_mismatches\":%zu,\"nan_pairs\":%zu,"
        "\"nan_contract\":\"classification_only\",\"cancellation_errors\":%zu,\"passed\":%s}\n",
        kCases, kBlocks, kRounds, count, mismatches, nan_pairs, cancellation_errors, passed ? "true" : "false");
    CUDA_OK(cudaFree(di)); CUDA_OK(cudaFree(db)); CUDA_OK(cudaFree(da));
    return passed ? 0 : 1;
}
