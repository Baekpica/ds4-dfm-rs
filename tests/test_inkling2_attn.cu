// Attention fast paths versus the original kernels at causal/ring/tile edges.
#include "cuda/inkling2_attn.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

enum class Cache { Sliding, Nvfp4 };
enum class Phase { Prefill, Decode };

static void check(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); abort(); }
}

__global__ void inputs(float *q, float *gate, float *sink, uint16_t *rel, uint16_t *kv,
        size_t heads, size_t cap, Cache format) {
    const bool fp4 = format == Cache::Nvfp4;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < heads * INK2A_DIM) {
        q[i] = ink2a_bf16(sinf(float(i % 137) * .07f));
        gate[i] = ink2a_bf16(cosf(float(i % 89) * .04f));
    }
    if (i < heads) { sink[i] = ink2a_bf16(float(i % 17) * .03f); }
    if (i < heads * INK2A_EXTENT) {
        rel[i] = __bfloat16_as_ushort(__float2bfloat16_rn(.3f * cosf(float(i % 71) * .03f)));
    }
    if (!fp4 && i < cap * INK2A_ROW) {
        kv[i] = __bfloat16_as_ushort(__float2bfloat16_rn(sinf(float(i % 113) * .09f)));
    }
    if (fp4 && i < cap / INK2A_KB * INK2A_BLOCK) {
        const size_t within = i % (INK2A_PAGE / 2);
        reinterpret_cast<uint8_t *>(kv)[i] = within < INK2A_KB * INK2A_FP4
            ? uint8_t((i * 13 + 7) % 256) : uint8_t(0x30 + i % 4);
    }
}

static void run(uint32_t rows, uint32_t base, Cache format = Cache::Sliding, Phase phase = Phase::Prefill) {
    const bool fp4 = format == Cache::Nvfp4;
    const uint32_t cap = phase == Phase::Decode ? 3071 : fp4 ? ((base + rows + INK2A_KB - 1) / INK2A_KB) * INK2A_KB : rows + INK2A_EXTENT + 7;
    const size_t kv_bytes = fp4 ? (size_t)cap / INK2A_KB * INK2A_BLOCK : (size_t)cap * INK2A_ROW * 2;
    const size_t heads = (size_t)rows * INK2A_HEADS, count = heads * INK2A_DIM;
    float *q, *gate, *sink, *ref, *got;
    uint16_t *rel, *kv;
    check(cudaMalloc(&q, count * 4)); check(cudaMalloc(&gate, count * 4));
    check(cudaMalloc(&sink, heads * 4)); check(cudaMalloc(&ref, count * 4)); check(cudaMalloc(&got, count * 4));
    check(cudaMalloc(&rel, heads * INK2A_EXTENT * 2)); check(cudaMalloc(&kv, kv_bytes));
    const size_t n = std::max(heads * INK2A_EXTENT, fp4 ? kv_bytes : kv_bytes / 2);
    inputs<<<(n + 255) / 256, 256>>>(q, gate, sink, rel, kv, heads, cap, format);
    const dim3 grid((rows + INK2A_TQ - 1) / INK2A_TQ, INK2A_HEADS / INK2A_CTA_HEADS);
    if (phase == Phase::Decode) {
        assert(!fp4);
        const uint32_t keys = std::min((uint32_t)INK2A_EXTENT, base + rows);
        const uint32_t splits = (keys + INK2A_SWA_SPLIT - 1) / INK2A_SWA_SPLIT;
        float *partials;
        check(cudaMalloc(&partials, heads * INK2A_SPLITS * INK2A_PARTIAL * sizeof(float)));
        const dim3 decode_grid(splits, INK2A_KV_HEADS, rows);
        ink2a_decode<INK2A_FMT_BF16><<<decode_grid, INK2A_DTHREADS>>>(partials, q, rel, kv,
            cap, INK2A_EXTENT, base, splits);
        ink2a_merge<<<heads, INK2A_DIM>>>(ref, partials, sink, gate, splits);
        ink2a_decode<INK2A_FMT_BF16, true><<<decode_grid, INK2A_DTHREADS>>>(partials, q, rel, kv,
            cap, INK2A_EXTENT, base, splits);
        ink2a_merge<<<heads, INK2A_DIM>>>(got, partials, sink, gate, splits);
        check(cudaFree(partials));
    } else if (fp4) {
        ink2a_prefill<INK2A_FMT_NVFP4><<<grid, INK2A_THREADS, INK2A_SMEM_FP4>>>(ref, q, rel, sink, gate, kv,
            cap, 0, rows, base);
        ink2a_prefill<INK2A_FMT_NVFP4, INK2A_ONE, false, true><<<grid, INK2A_THREADS, INK2A_SMEM_FP4>>>(
            got, q, rel, sink, gate, kv, cap, 0, rows, base);
    } else {
        ink2a_prefill<INK2A_FMT_BF16><<<grid, INK2A_THREADS, INK2A_SMEM>>>(ref, q, rel, sink, gate, kv,
            cap, INK2A_EXTENT, rows, base);
        ink2a_prefill<INK2A_FMT_BF16, INK2A_ONE, true><<<grid, INK2A_THREADS, INK2A_SMEM>>>(got, q, rel, sink, gate,
            kv, cap, INK2A_EXTENT, rows, base);
    }
    std::vector<float> a(count), b(count);
    check(cudaMemcpy(a.data(), ref, count * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(b.data(), got, count * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; i++) { assert(std::isfinite(a[i]) && a[i] == b[i]); }
    for (void *p : {static_cast<void *>(q), static_cast<void *>(gate), static_cast<void *>(sink),
            static_cast<void *>(ref), static_cast<void *>(got), static_cast<void *>(rel), static_cast<void *>(kv)}) {
        check(cudaFree(p));
    }
    printf("%s %s attention rows=%u base=%u cap=%u: equal\n", phase == Phase::Decode ? "decode" : "prefill", fp4 ? "full NVFP4" : "sliding", rows, base, cap);
}

int main() {
    check(cudaFuncSetAttribute(ink2a_prefill<INK2A_FMT_BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM));
    check(cudaFuncSetAttribute(ink2a_prefill<INK2A_FMT_BF16, INK2A_ONE, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM));
    check(cudaFuncSetAttribute(ink2a_prefill<INK2A_FMT_NVFP4>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM_FP4));
    check(cudaFuncSetAttribute(ink2a_prefill<INK2A_FMT_NVFP4, INK2A_ONE, false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM_FP4));
    for (uint32_t base : {0u, 7u, 30u, 31u, 32u, 62u, 63u, 64u, 126u, 127u, 128u,
            1022u, 1023u, 1024u, 3070u, 3071u, 9217u, 32768u}) {
        run(1, base, Cache::Sliding, Phase::Decode);
    }
    for (uint32_t base : {1023u, 3071u}) { run(8, base, Cache::Sliding, Phase::Decode); }
    for (uint32_t rows : {9u, 63u, 64u, 65u, 129u, 2048u}) {
        run(rows, 0);
        run(rows, 3 * (rows + INK2A_EXTENT + 7) + 31);
        for (uint32_t base : {0u, 1023u, 1087u, 1088u, 6144u, 32768u}) { run(rows, base, Cache::Nvfp4); }
    }
}
