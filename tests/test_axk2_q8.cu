// Q8 weight staging preserves the per-lane FMA and xor reduction.
#include "cuda/axk2_q8.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(cudaError_t err) {
    if (err != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(err)); std::exit(1); }
}

static bool run(uint32_t outs, uint32_t cols) {
    const uint32_t blocks = cols / AX_Q8_BLOCK, row_bytes = blocks * AX_Q8_BYTES;
    std::mt19937 rng(819);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::vector<uint8_t> packed((uint64_t)outs * row_bytes);
    std::vector<float> input(cols), expected(outs);
    for (float &v : input) { v = outs == 1 ? 0.0f : __bfloat162float(__float2bfloat16_rn(unit(rng))); }
    for (uint32_t o = 0; o < outs; o++) {
        for (uint32_t b = 0; b < blocks; b++) {
            uint8_t *p = packed.data() + (uint64_t)o * row_bytes + b * AX_Q8_BYTES;
            const __half scale = __float2half_rn(unit(rng) / 16.0f);
            memcpy(p, &scale, sizeof(scale));
            for (uint32_t lane = 0; lane < AX_Q8_BLOCK; lane++) { p[2 + lane] = (uint8_t)rng(); }
        }
        float lanes[32] = {};
        for (uint32_t b = 0; b < blocks; b++) {
            const uint8_t *p = packed.data() + (uint64_t)o * row_bytes + b * AX_Q8_BYTES;
            __half scale;
            memcpy(&scale, p, sizeof(scale));
            for (uint32_t lane = 0; lane < 32; lane++) {
                const float w = __half2float(scale) * (float)(int8_t)p[2 + lane];
                lanes[lane] = std::fma(w, input[b * AX_Q8_BLOCK + lane], lanes[lane]);
            }
        }
        for (uint32_t stride = 16; stride; stride >>= 1) {
            float next[32];
            for (uint32_t lane = 0; lane < 32; lane++) { next[lane] = lanes[lane] + lanes[lane ^ stride]; }
            memcpy(lanes, next, sizeof(lanes));
        }
        expected[o] = lanes[0];
    }

    uint8_t *weight = nullptr, *guarded = nullptr;
    float *x = nullptr;
    std::vector<uint8_t> result((uint64_t)outs * sizeof(float) + 32);
    check(cudaMalloc(&weight, packed.size()));
    check(cudaMalloc(&x, cols * sizeof(float)));
    check(cudaMalloc(&guarded, result.size()));
    check(cudaMemcpy(weight, packed.data(), packed.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(x, input.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    bool ok = true;
    for (uint32_t variant = 0; variant < 3; variant++) {
        const uint32_t warps = variant == 2 ? 1 : AX_Q8_WARPS, smem = warps * row_bytes;
        const auto kernel = variant == 0 ? ax_q8_gemv<ax_q8_path::reference> :
            variant == 1 ? ax_q8_gemv<ax_q8_path::stream> : ax_q8_gemv<ax_q8_path::stream, 1>;
        check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        check(cudaMemset(guarded, 0xa5, result.size()));
        kernel<<<(outs + warps - 1) / warps, warps * 32, smem>>>(
            (float *)(guarded + 16), weight, x, outs, cols);
        check(cudaMemcpy(result.data(), guarded, result.size(), cudaMemcpyDeviceToHost));
        ok = ok && !memcmp(result.data() + 16, expected.data(), outs * sizeof(float));
        for (uint32_t i = 0; i < 16; i++) {
            ok = ok && result[i] == 0xa5 && result[result.size() - 16 + i] == 0xa5;
        }
    }
    for (float v : expected) { ok = ok && std::isfinite(v); }
    printf("outs=%u cols=%u CPU parity/canaries %s\n", outs, cols, ok ? "PASS" : "FAIL");
    check(cudaFree(weight)); check(cudaFree(x)); check(cudaFree(guarded));
    return ok;
}

int main() {
    return run(1, 256) && run(7, 1536) && run(9, 7168) && run(128, 7168) && run(8192, 1536) ? 0 : 1;
}
