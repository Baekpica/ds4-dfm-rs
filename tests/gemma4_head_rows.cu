/* The fused verify head against the one-row head it replaces: same weights,
 * same input rows, byte-identical output. `g4_head_gemv_rows_kernel<COLS,NT>`
 * must produce, for every input row t, exactly the bytes
 * `g4_head_gemv_kernel<COLS>` produces for that row alone: the staging is the
 * same g4_round value in bf16, the eight-FMA sequence per uint4 is the same,
 * the XOR reduction is the same, and the softcap epilogue is the same. A
 * single differing byte fails the test.
 *
 * Built with the production flags (the Makefile target does not pass
 * --fmad=false), because the claim is about the production arithmetic.
 *
 * usage: gemma4_head_rows [VOCAB_ROWS]
 */
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../cuda/gemma4_head.cuh"
#include "../ds4_gemma4_plan.h"

#define CUDA_OK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){ \
    std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e_)); std::exit(2); } } while(0)

static constexpr uint32_t kCols = G4_EMBED;
static constexpr uint32_t kRows = G4_HEAD_ROWS;
static constexpr uint32_t kBlocks = 8;

/* A deterministic spread with both signs and magnitudes near a bf16
 * rounding boundary, so a reordered FMA or a dropped rounding shows up. */
static float value(unsigned i) {
    return std::sin((i + 1) * 0.117f) + 0.3f * std::cos((i + 3) * 0.031f) +
        0.0001f * std::sin((i + 7) * 0.0007f);
}

static __nv_bfloat16 *bf16_device(const std::vector<float> &x) {
    std::vector<__nv_bfloat16> packed(x.size());
    for (unsigned i = 0; i < x.size(); i++) { packed[i] = __float2bfloat16_rn(x[i]); }
    __nv_bfloat16 *p = nullptr;
    CUDA_OK(cudaMalloc(&p, packed.size() * sizeof(packed[0])));
    CUDA_OK(cudaMemcpy(p, packed.data(), packed.size() * sizeof(packed[0]),
        cudaMemcpyHostToDevice));
    return p;
}

static float *float_device(const std::vector<float> &x) {
    float *p = nullptr;
    CUDA_OK(cudaMalloc(&p, x.size() * sizeof(float)));
    CUDA_OK(cudaMemcpy(p, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    return p;
}

/* One NT compile-time instantiation: the per-row head into `one` and the
 * fused head into `fused`, both over the same [kRows x kCols] table. */
template <uint32_t NT>
static void run_case(const __nv_bfloat16 *w, const float *in, float *one, float *fused,
        uint32_t rows, unsigned long *exact, unsigned long *count) {
    for (uint32_t t = 0; t < NT; t++) {
        g4_head_gemv_kernel<kCols, true><<<kBlocks, 256>>>(
            w, in + (uint64_t)t * kCols, one + (uint64_t)t * rows, rows, G4_SOFTCAP);
        CUDA_OK(cudaGetLastError());
    }
    g4_head_gemv_rows_kernel<kCols, NT, true><<<kBlocks, 256>>>(
        w, in, fused, rows, G4_SOFTCAP);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<float> a((size_t)NT * rows), b((size_t)NT * rows);
    CUDA_OK(cudaMemcpy(a.data(), one, a.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(b.data(), fused, b.size() * sizeof(float), cudaMemcpyDeviceToHost));
    *exact = 0;
    *count = a.size();
    for (size_t i = 0; i < a.size(); i++) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) == 0) { (*exact)++; }
    }
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
        for (size_t i = 0, shown = 0; i < a.size() && shown < 3; i++) {
            if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                std::printf("{\"test\":\"head_rows\",\"rows\":%u,\"miss\":%zu,"
                    "\"index\":%zu,\"one\":%.9g,\"fused\":%.9g}\n",
                    NT, shown++, i, a[i], b[i]);
            }
        }
        std::printf("{\"test\":\"head_rows\",\"rows\":%u,\"exact\":%lu,\"count\":%lu,"
            "\"equal\":false}\n", NT, *exact, *count);
        std::exit(1);
    }
    std::printf("{\"test\":\"head_rows\",\"rows\":%u,\"exact\":%lu,\"count\":%lu,"
        "\"equal\":true}\n", NT, *exact, *count);
}

int main(int argc, char **argv) {
    const uint32_t rows = argc > 1 ? (uint32_t)std::strtoul(argv[1], nullptr, 10) : 512u;
    if (!rows || rows > 8192u) {
        std::fprintf(stderr, "usage: gemma4_head_rows [VOCAB_ROWS 1..8192]\n");
        return 2;
    }
    std::vector<float> wf((size_t)rows * kCols), inf((size_t)kRows * kCols);
    for (size_t i = 0; i < wf.size(); i++) { wf[i] = value((unsigned)i); }
    for (size_t i = 0; i < inf.size(); i++) { inf[i] = value((unsigned)(i * 3u + 1u)); }
    __nv_bfloat16 *w = bf16_device(wf);
    float *in = float_device(inf);
    float *one = nullptr, *fused = nullptr;
    CUDA_OK(cudaMalloc(&one, (size_t)kRows * rows * sizeof(float)));
    CUDA_OK(cudaMalloc(&fused, (size_t)kRows * rows * sizeof(float)));

    unsigned long exact = 0, count = 0;
    run_case<1>(w, in, one, fused, rows, &exact, &count);
    run_case<2>(w, in, one, fused, rows, &exact, &count);
    run_case<3>(w, in, one, fused, rows, &exact, &count);
    run_case<4>(w, in, one, fused, rows, &exact, &count);
    run_case<5>(w, in, one, fused, rows, &exact, &count);
    run_case<6>(w, in, one, fused, rows, &exact, &count);
    run_case<7>(w, in, one, fused, rows, &exact, &count);
    run_case<8>(w, in, one, fused, rows, &exact, &count);

    CUDA_OK(cudaFree(one));
    CUDA_OK(cudaFree(fused));
    CUDA_OK(cudaFree(in));
    CUDA_OK(cudaFree(w));
    std::printf("{\"test\":\"head_rows\",\"verdict\":\"pass\",\"rows\":[1,2,3,4,5,6,7,8]}\n");
    return 0;
}
