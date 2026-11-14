// Bias prefetch versus the original sliding attention at causal/ring edges.
#include "cuda/inkling2_attn.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); abort(); }
}

__global__ void inputs(float *q, float *gate, float *sink, uint16_t *rel, uint16_t *kv,
        size_t heads, size_t cap) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < heads * INK2A_DIM) {
        q[i] = ink2a_bf16(sinf(float(i % 137) * .07f));
        gate[i] = ink2a_bf16(cosf(float(i % 89) * .04f));
    }
    if (i < heads) { sink[i] = ink2a_bf16(float(i % 17) * .03f); }
    if (i < heads * INK2A_EXTENT) {
        rel[i] = __bfloat16_as_ushort(__float2bfloat16_rn(.3f * cosf(float(i % 71) * .03f)));
    }
    if (i < cap * INK2A_ROW) {
        kv[i] = __bfloat16_as_ushort(__float2bfloat16_rn(sinf(float(i % 113) * .09f)));
    }
}

static void run(uint32_t rows, uint32_t base) {
    const uint32_t cap = rows + INK2A_EXTENT + 7;
    const size_t heads = (size_t)rows * INK2A_HEADS, count = heads * INK2A_DIM;
    float *q, *gate, *sink, *ref, *got;
    uint16_t *rel, *kv;
    check(cudaMalloc(&q, count * 4)); check(cudaMalloc(&gate, count * 4));
    check(cudaMalloc(&sink, heads * 4)); check(cudaMalloc(&ref, count * 4)); check(cudaMalloc(&got, count * 4));
    check(cudaMalloc(&rel, heads * INK2A_EXTENT * 2)); check(cudaMalloc(&kv, (size_t)cap * INK2A_ROW * 2));
    const size_t n = std::max(heads * INK2A_EXTENT, (size_t)cap * INK2A_ROW);
    inputs<<<(n + 255) / 256, 256>>>(q, gate, sink, rel, kv, heads, cap);
    const dim3 grid((rows + INK2A_TQ - 1) / INK2A_TQ, INK2A_HEADS / INK2A_CTA_HEADS);
    ink2a_prefill<INK2A_FMT_BF16><<<grid, INK2A_THREADS, INK2A_SMEM>>>(ref, q, rel, sink, gate, kv,
        cap, INK2A_EXTENT, rows, base);
    ink2a_prefill<INK2A_FMT_BF16, INK2A_ONE, true><<<grid, INK2A_THREADS, INK2A_SMEM>>>(got, q, rel, sink, gate,
        kv, cap, INK2A_EXTENT, rows, base);
    std::vector<float> a(count), b(count);
    check(cudaMemcpy(a.data(), ref, count * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(b.data(), got, count * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; i++) { assert(std::isfinite(a[i]) && a[i] == b[i]); }
    for (void *p : {static_cast<void *>(q), static_cast<void *>(gate), static_cast<void *>(sink),
            static_cast<void *>(ref), static_cast<void *>(got), static_cast<void *>(rel), static_cast<void *>(kv)}) {
        check(cudaFree(p));
    }
    printf("sliding attention rows=%u base=%u cap=%u: equal\n", rows, base, cap);
}

int main() {
    check(cudaFuncSetAttribute(ink2a_prefill<INK2A_FMT_BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM));
    check(cudaFuncSetAttribute(ink2a_prefill<INK2A_FMT_BF16, INK2A_ONE, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM));
    for (uint32_t rows : {9u, 63u, 64u, 65u, 129u, 2048u}) {
        run(rows, 0);
        run(rows, 3 * (rows + INK2A_EXTENT + 7) + 31);
    }
}
