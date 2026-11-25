// Attention fast paths versus the original kernels at causal/ring/tile edges.
// The ping-pong prefill tile, its NVFP4-score form, the rank-16 bias and the
// tensor-core decode split change rounding (an FMA in the exponent, 4-bit
// query components, a tensor-core sum, BF16 probabilities), so they compare
// within a bound; key segments (the SSD tier's shape) must reproduce one pass.
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
        // Mix high address bits so batches and pages carry distinct values.
        const uint64_t mixed = (uint64_t)i * UINT64_C(0x9e3779b97f4a7c15);
        reinterpret_cast<uint8_t *>(kv)[i] = within < INK2A_KB * INK2A_FP4
            ? uint8_t(mixed >> 32) : uint8_t(0x30 + ((mixed >> 40) & 7));
    }
}

static void run(uint32_t rows, uint32_t base, Cache format = Cache::Sliding, Phase phase = Phase::Prefill) {
    const bool fp4 = format == Cache::Nvfp4;
    const uint32_t cap = fp4 ? ((base + rows + INK2A_KB - 1) / INK2A_KB) * INK2A_KB
        : phase == Phase::Decode ? 3071 : rows + INK2A_EXTENT + 7;
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
        const uint32_t keys = fp4 ? base + rows : std::min((uint32_t)INK2A_EXTENT, base + rows);
        const uint32_t splits = fp4 ? ink2a_splits(keys) : (keys + INK2A_SWA_SPLIT - 1) / INK2A_SWA_SPLIT;
        float *partials;
        check(cudaMalloc(&partials, heads * INK2A_SPLITS * INK2A_PARTIAL * sizeof(float)));
        const dim3 decode_grid(splits, INK2A_KV_HEADS, rows);
        if (fp4) {
            ink2a_decode<INK2A_FMT_NVFP4><<<decode_grid, INK2A_DTHREADS>>>(partials, q, rel, kv,
                cap, 0, base, splits);
        } else {
            ink2a_decode<INK2A_FMT_BF16><<<decode_grid, INK2A_DTHREADS>>>(partials, q, rel, kv,
                cap, INK2A_EXTENT, base, splits);
        }
        ink2a_merge<<<heads, INK2A_DIM>>>(ref, partials, sink, gate, splits);
        if (fp4) {
            ink2a_decode<INK2A_FMT_NVFP4, true><<<decode_grid, INK2A_DTHREADS>>>(partials, q, rel, kv,
                cap, 0, base, splits);
        } else {
            ink2a_decode<INK2A_FMT_BF16, true><<<decode_grid, INK2A_DTHREADS>>>(partials, q, rel, kv,
                cap, INK2A_EXTENT, base, splits);
        }
        ink2a_merge<<<heads, INK2A_DIM>>>(got, partials, sink, gate, splits);
        check(cudaFree(partials));
    } else if (fp4) {
        ink2a_prefill<INK2A_FMT_NVFP4><<<grid, INK2A_THREADS, INK2A_SMEM_FP4>>>(ref, q, rel, sink, gate, kv,
            cap, 0, rows, base);
        ink2a_prefill<INK2A_FMT_NVFP4, INK2A_ONE, INK2A_BIAS_LOAD, true><<<grid, INK2A_THREADS, INK2A_SMEM_FP4>>>(
            got, q, rel, sink, gate, kv, cap, 0, rows, base);
    } else {
        ink2a_prefill<INK2A_FMT_BF16><<<grid, INK2A_THREADS, INK2A_SMEM>>>(ref, q, rel, sink, gate, kv,
            cap, INK2A_EXTENT, rows, base);
        ink2a_prefill<INK2A_FMT_BF16, INK2A_ONE, INK2A_BIAS_AHEAD><<<grid, INK2A_THREADS, INK2A_SMEM>>>(got, q, rel,
            sink, gate, kv, cap, INK2A_EXTENT, rows, base);
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

// A bias table from rank-16 factors (the q/r kernel's sum) and the factors
// in the attention's layout.
__global__ void rank_inputs(uint16_t *table, uint16_t *factors, size_t heads) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    auto r = [](size_t at) { return ink2a_bf16(.5f * sinf(float(at % 211) * .05f)); };
    auto proj = [](size_t k, size_t d) { return ink2a_bf16(.2f * cosf(float((k * INK2A_EXTENT + d) % 173) * .04f)); };
    if (i < heads * INK2A_EXTENT) {
        const size_t rh = i / INK2A_EXTENT, d = i % INK2A_EXTENT;
        float v = 0.0f;
        for (uint32_t k = 0; k < INK2A_RANK; k++) { v = fmaf(r(rh * INK2A_RANK + k), proj(k, d), v); }
        table[i] = __bfloat16_as_ushort(__float2bfloat16_rn(v));
    }
    if (i < INK2A_RANK_PROJ) {
        const size_t d = i / INK2A_RANK, quad = i / 4 % 4, e = i % 4;
        factors[i] = __bfloat16_as_ushort(__float2bfloat16_rn(proj((e / 2) * 8 + 2 * quad + e % 2, d)));
    }
    if (i < heads * INK2A_RANK) { factors[INK2A_RANK_PROJ + i] = __bfloat16_as_ushort(__float2bfloat16_rn(r(i))); }
}

/* Sliding prefill with the bias from its factors against the table: the
 * tensor core sums the 16 exact products in its own order, so a bias may
 * move one BF16 step. */
static void run_rank(uint32_t rows, uint32_t base) {
    enum { STEPS = 2, PER_MILLION = 500, FEW = 4 };
    constexpr float CANCEL = 1e-4f;
    const uint32_t cap = rows + INK2A_EXTENT + 7;
    const size_t heads = (size_t)rows * INK2A_HEADS, count = heads * INK2A_DIM;
    float *q, *gate, *sink, *ref, *got;
    uint16_t *rel, *factors, *kv;
    for (float **p : {&q, &gate, &ref, &got}) { check(cudaMalloc(p, count * 4)); }
    check(cudaMalloc(&sink, heads * 4));
    check(cudaMalloc(&rel, heads * INK2A_EXTENT * 2));
    check(cudaMalloc(&factors, heads * INK2A_EXTENT * 2));
    check(cudaMalloc(&kv, (size_t)cap * INK2A_ROW * 2));
    const size_t n = std::max(heads * INK2A_EXTENT, (size_t)cap * INK2A_ROW);
    inputs<<<(n + 255) / 256, 256>>>(q, gate, sink, rel, kv, heads, cap, Cache::Sliding);
    rank_inputs<<<(heads * INK2A_EXTENT + 255) / 256, 256>>>(rel, factors, heads);
    const dim3 grid((rows + INK2A_TQ - 1) / INK2A_TQ, INK2A_HEADS / INK2A_CTA_HEADS);
    ink2a_prefill<INK2A_FMT_BF16, INK2A_ONE, INK2A_BIAS_AHEAD><<<grid, INK2A_THREADS, INK2A_SMEM>>>(ref, q, rel, sink,
        gate, kv, cap, INK2A_EXTENT, rows, base);
    ink2a_prefill<INK2A_FMT_BF16, INK2A_ONE, INK2A_BIAS_RANK><<<grid, INK2A_THREADS, INK2A_SMEM + INK2A_RANK_SMEM>>>(
        got, q, factors, sink, gate, kv, cap, INK2A_EXTENT, rows, base);
    std::vector<float> a(count), b(count);
    check(cudaMemcpy(a.data(), ref, count * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(b.data(), got, count * 4, cudaMemcpyDeviceToHost));
    size_t moved = 0;
    float top = 0.0f;
    for (float v : a) { top = std::max(top, std::fabs(v)); }
    for (size_t i = 0; i < count; i++) {
        assert(std::isfinite(a[i]) && std::isfinite(b[i]));
        if (a[i] == b[i]) { continue; }
        moved++;
        const float mag = std::max(std::fabs(a[i]), std::fabs(b[i]));
        assert(std::fabs(a[i] - b[i]) <= std::max(STEPS * std::ldexp(1.0f, std::ilogb(mag) - 7), CANCEL * top));
    }
    assert(moved * 1000000 <= (size_t)PER_MILLION * count + (size_t)FEW * 1000000);
    for (void *p : {(void *)q, (void *)gate, (void *)sink, (void *)ref, (void *)got, (void *)rel, (void *)factors,
            (void *)kv}) { check(cudaFree(p)); }
    printf("prefill rank bias rows=%u base=%u: %zu of %zu moved\n", rows, base, moved, count);
}

// NVFP4 rows and their inputs on the device.
struct Full {
    float *q, *gate, *sink, *ref, *got, *seg, *state, *partials;
    uint16_t *rel, *kv;
    uint32_t rows, base;
    size_t heads, count;
};

static Full full_inputs(uint32_t rows, uint32_t base) {
    Full f = {};
    f.rows = rows;
    f.base = base;
    const uint32_t cap = ((base + rows + INK2A_KB - 1) / INK2A_KB) * INK2A_KB;
    const size_t kv_bytes = (size_t)cap / INK2A_KB * INK2A_BLOCK;
    f.heads = (size_t)rows * INK2A_HEADS;
    f.count = f.heads * INK2A_DIM;
    for (float **p : {&f.q, &f.gate, &f.ref, &f.got, &f.seg}) { check(cudaMalloc(p, f.count * 4)); }
    check(cudaMalloc(&f.sink, f.heads * 4));
    check(cudaMalloc(&f.state, f.heads * INK2A_STATE * 4));
    check(cudaMalloc(&f.partials, f.heads * INK2A_SPLITS * INK2A_PARTIAL * sizeof(float)));
    check(cudaMalloc(&f.rel, f.heads * INK2A_EXTENT * 2));
    check(cudaMalloc(&f.kv, kv_bytes));
    const size_t n = std::max(f.heads * INK2A_EXTENT, kv_bytes);
    inputs<<<(n + 255) / 256, 256>>>(f.q, f.gate, f.sink, f.rel, f.kv, f.heads, cap, Cache::Nvfp4);
    return f;
}

static void full_free(Full &f) {
    for (void *p : {(void *)f.q, (void *)f.gate, (void *)f.sink, (void *)f.ref, (void *)f.got, (void *)f.seg,
            (void *)f.state, (void *)f.partials, (void *)f.rel, (void *)f.kv}) { check(cudaFree(p)); }
}

static std::vector<float> rows_of(const Full &f, const float *device) {
    std::vector<float> host(f.count);
    check(cudaMemcpy(host.data(), device, f.count * 4, cudaMemcpyDeviceToHost));
    for (float v : host) { assert(std::isfinite(v)); }
    return host;
}

// Key segments of `span` positions: FIRST, NEXT..., LAST.
static int segment_mode(uint32_t lo, uint32_t hi, uint32_t end) {
    return lo == 0 ? INK2A_FIRST : hi == end ? INK2A_LAST : INK2A_NEXT;
}

/* Ping-pong prefill against the in-step tile. A rounding boundary may move
 * one BF16 step of the attention, two after the gate's product. */
static void run_pingpong(uint32_t rows, uint32_t base) {
    enum { SEGMENT = 4096, STEPS = 2, PER_MILLION = 500, FEW = 4 };
    constexpr float CANCEL = 1e-4f;
    Full f = full_inputs(rows, base);
    const dim3 grid((rows + INK2A_TQ - 1) / INK2A_TQ, INK2A_HEADS / INK2A_CTA_HEADS);
    const auto *pages = (const uint8_t *)f.kv;
    ink2a_prefill<INK2A_FMT_NVFP4><<<grid, INK2A_THREADS, INK2A_SMEM_FP4>>>(f.ref, f.q, f.rel, f.sink, f.gate, f.kv,
        0, 0, rows, base);
    ink2a_prefill_pp<INK2A_ONE><<<grid, INK2A_THREADS, INK2A_PP_SMEM>>>(f.got, f.q, f.rel, f.sink, f.gate, pages,
        rows, base);
    const uint32_t end = base + rows;
    for (uint32_t lo = 0; lo < end; lo += SEGMENT) {
        const uint32_t hi = std::min(lo + (uint32_t)SEGMENT, end);
        const int mode = segment_mode(lo, hi, end);
        if (end <= SEGMENT) {
            ink2a_prefill_pp<INK2A_ONE><<<grid, INK2A_THREADS, INK2A_PP_SMEM>>>(f.seg, f.q, f.rel, f.sink, f.gate,
                pages, rows, base);
        } else if (mode == INK2A_FIRST) {
            ink2a_prefill_pp<INK2A_FIRST><<<grid, INK2A_THREADS, INK2A_PP_SMEM>>>(f.seg, f.q, f.rel, f.sink, f.gate,
                pages, rows, base, f.state, lo, hi, 0);
        } else if (mode == INK2A_NEXT) {
            ink2a_prefill_pp<INK2A_NEXT><<<grid, INK2A_THREADS, INK2A_PP_SMEM>>>(f.seg, f.q, f.rel, f.sink, f.gate,
                pages, rows, base, f.state, lo, hi, 0);
        } else {
            ink2a_prefill_pp<INK2A_LAST><<<grid, INK2A_THREADS, INK2A_PP_SMEM>>>(f.seg, f.q, f.rel, f.sink, f.gate,
                pages, rows, base, f.state, lo, hi, 0);
        }
    }
    const std::vector<float> a = rows_of(f, f.ref), b = rows_of(f, f.got), c = rows_of(f, f.seg);
    size_t moved = 0;
    float top = 0.0f;
    for (float v : a) { top = std::max(top, std::fabs(v)); }
    for (size_t i = 0; i < f.count; i++) {
        assert(b[i] == c[i]);
        if (a[i] == b[i]) { continue; }
        moved++;
        // Sums that cancel to near zero keep F32's absolute error instead.
        const float mag = std::max(std::fabs(a[i]), std::fabs(b[i]));
        const float bound = std::max(STEPS * std::ldexp(1.0f, std::ilogb(mag) - 7), CANCEL * top);
        if (std::fabs(a[i] - b[i]) > bound) {
            printf("rows=%u base=%u at %zu: %.9g vs %.9g (top %.4g)\n", rows, base, i, a[i], b[i], top);
        }
        assert(std::fabs(a[i] - b[i]) <= bound);
    }
    printf("prefill ping-pong rows=%u base=%u: %zu of %zu within %d BF16 steps, segments equal\n", rows, base, moved,
        f.count, (int)STEPS);
    assert(moved * 1000000 <= (size_t)PER_MILLION * f.count + (size_t)FEW * 1000000);
    full_free(f);
}

#ifdef DS4_CUDA_HAVE_MXF4
/* NVFP4-score prefill against the in-step tile: the stored 4-bit keys times
 * three NVFP4 components of each query, far scores unrounded. */
static void run_fp4(uint32_t rows, uint32_t base) {
    enum { SEGMENT = 4096 };
    constexpr double BOUND = 2e-2;
    Full f = full_inputs(rows, base);
    const dim3 grid((rows + INK2A_TQ - 1) / INK2A_TQ, INK2A_HEADS / INK2A_CTA_HEADS);
    const auto *pages = (const uint8_t *)f.kv;
    ink2a_prefill<INK2A_FMT_NVFP4><<<grid, INK2A_THREADS, INK2A_SMEM_FP4>>>(f.ref, f.q, f.rel, f.sink, f.gate, f.kv,
        0, 0, rows, base);
    ink2a_prefill_q4<INK2A_ONE><<<grid, INK2A_THREADS, INK2A_Q4_SMEM>>>(f.got, f.q, f.rel, f.sink, f.gate, pages,
        rows, base);
    const uint32_t end = base + rows;
    for (uint32_t lo = 0; lo < end; lo += SEGMENT) {
        const uint32_t hi = std::min(lo + (uint32_t)SEGMENT, end);
        const int mode = segment_mode(lo, hi, end);
        if (end <= SEGMENT) {
            ink2a_prefill_q4<INK2A_ONE><<<grid, INK2A_THREADS, INK2A_Q4_SMEM>>>(f.seg, f.q, f.rel, f.sink, f.gate,
                pages, rows, base);
        } else if (mode == INK2A_FIRST) {
            ink2a_prefill_q4<INK2A_FIRST><<<grid, INK2A_THREADS, INK2A_Q4_SMEM>>>(f.seg, f.q, f.rel, f.sink, f.gate,
                pages, rows, base, f.state, lo, hi, 0);
        } else if (mode == INK2A_NEXT) {
            ink2a_prefill_q4<INK2A_NEXT><<<grid, INK2A_THREADS, INK2A_Q4_SMEM>>>(f.seg, f.q, f.rel, f.sink, f.gate,
                pages, rows, base, f.state, lo, hi, 0);
        } else {
            ink2a_prefill_q4<INK2A_LAST><<<grid, INK2A_THREADS, INK2A_Q4_SMEM>>>(f.seg, f.q, f.rel, f.sink, f.gate,
                pages, rows, base, f.state, lo, hi, 0);
        }
    }
    const std::vector<float> a = rows_of(f, f.ref), b = rows_of(f, f.got), c = rows_of(f, f.seg);
    double diff = 0.0, ref = 0.0;
    for (size_t i = 0; i < f.count; i++) {
        assert(b[i] == c[i]);
        diff += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
        ref += (double)a[i] * a[i];
    }
    const double rel = std::sqrt(diff / ref);
    printf("prefill NVFP4 scores rows=%u base=%u: relative RMS %.1e, segments equal\n", rows, base, rel);
    assert(rel <= BOUND);
    full_free(f);
}
#endif

/* One decode row on the tensor-core split against the SIMT walk: the split
 * rounds probabilities to BF16 (prefill's and verify's arithmetic). */
static void run_split(uint32_t rows, uint32_t base) {
    enum { SEGMENT = 4096 };
    constexpr double BOUND = 5e-3;
    Full f = full_inputs(rows, base);
    const auto *pages = (const uint8_t *)f.kv;
    const uint32_t end = base + rows, splits = ink2a_splits(end);
    ink2a_decode<INK2A_FMT_NVFP4><<<dim3(splits, INK2A_KV_HEADS, rows), INK2A_DTHREADS>>>(f.partials, f.q, f.rel,
        f.kv, 0, 0, base, splits);
    ink2a_merge<<<f.heads, INK2A_DIM>>>(f.ref, f.partials, f.sink, f.gate, splits);
    auto slices_of = [](uint32_t lo, uint32_t hi) {
        return std::min<uint32_t>((hi - 1) / INK2A_TK + 1 - lo / INK2A_TK, INK2A_SSLICES);
    };
    uint32_t slices = slices_of(0, end);
    ink2a_split<<<dim3(slices, INK2A_KV_HEADS, rows), INK2A_SWARPS * 32>>>(f.partials, f.q, f.rel, pages, base, slices);
    ink2a_merge<<<f.heads, INK2A_DIM>>>(f.got, f.partials, f.sink, f.gate, slices * INK2A_SWARPS);
    // Every row's keys end inside the last segment at these shapes.
    for (uint32_t lo = 0; lo < end; lo += SEGMENT) {
        const uint32_t hi = std::min(lo + (uint32_t)SEGMENT, end);
        slices = slices_of(lo, hi);
        ink2a_split<<<dim3(slices, INK2A_KV_HEADS, rows), INK2A_SWARPS * 32>>>(f.partials, f.q, f.rel, pages, base,
            slices, lo, hi, 0);
        ink2a_fold<<<f.heads, INK2A_DIM>>>(f.state, f.partials, f.sink, slices * INK2A_SWARPS, lo == 0);
    }
    ink2a_finish<<<f.heads, INK2A_DIM>>>(f.seg, f.state, f.gate);
    const std::vector<float> a = rows_of(f, f.ref), b = rows_of(f, f.got), c = rows_of(f, f.seg);
    double diff = 0.0, seg = 0.0, ref = 0.0;
    for (size_t i = 0; i < f.count; i++) {
        diff += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
        seg += ((double)a[i] - c[i]) * ((double)a[i] - c[i]);
        ref += (double)a[i] * a[i];
    }
    const double rel = std::sqrt(diff / ref), rel_seg = std::sqrt(seg / ref);
    assert(rel <= BOUND && rel_seg <= BOUND);
    full_free(f);
    printf("decode split rows=%u base=%u: relative RMS %.1e, segments %.1e\n", rows, base, rel, rel_seg);
}

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);   // a failed assertion keeps the lines before it
    check(cudaFuncSetAttribute(ink2a_prefill<INK2A_FMT_BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM));
    check(cudaFuncSetAttribute(ink2a_prefill<INK2A_FMT_BF16, INK2A_ONE, INK2A_BIAS_AHEAD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM));
    check(cudaFuncSetAttribute(ink2a_prefill<INK2A_FMT_BF16, INK2A_ONE, INK2A_BIAS_RANK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM + INK2A_RANK_SMEM));
    check(cudaFuncSetAttribute(ink2a_prefill<INK2A_FMT_NVFP4>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM_FP4));
    check(cudaFuncSetAttribute(ink2a_prefill<INK2A_FMT_NVFP4, INK2A_ONE, INK2A_BIAS_LOAD, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM_FP4));
    for (uint32_t base : {0u, 7u, 30u, 31u, 32u, 62u, 63u, 64u, 126u, 127u, 128u,
            1022u, 1023u, 1024u, 3070u, 3071u, 9217u, 32768u}) {
        run(1, base, Cache::Sliding, Phase::Decode);
        run(1, base, Cache::Nvfp4, Phase::Decode);
    }
    run(1, 65536, Cache::Nvfp4, Phase::Decode);
    for (uint32_t base : {1023u, 3071u}) {
        run(8, base, Cache::Sliding, Phase::Decode);
        run(8, base, Cache::Nvfp4, Phase::Decode);
    }
    for (uint32_t rows : {9u, 63u, 64u, 65u, 129u, 2048u}) {
        run(rows, 0);
        run(rows, 3 * (rows + INK2A_EXTENT + 7) + 31);
        for (uint32_t base : {0u, 1023u, 1087u, 1088u, 6144u, 32768u}) { run(rows, base, Cache::Nvfp4); }
    }
    for (uint32_t rows : {9u, 63u, 64u, 65u, 129u, 2048u}) {
        for (uint32_t base : {0u, 1u, 62u, 63u, 64u, 959u, 1023u, 1024u, 1087u, 6144u, 32769u}) { run_rank(rows, base); }
    }
    for (auto kernel : {ink2a_prefill_pp<INK2A_ONE>, ink2a_prefill_pp<INK2A_FIRST>, ink2a_prefill_pp<INK2A_NEXT>,
            ink2a_prefill_pp<INK2A_LAST>}) {
        check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_PP_SMEM));
    }
    for (uint32_t rows : {9u, 63u, 64u, 65u, 129u, 2048u}) {
        for (uint32_t base : {0u, 1023u, 1087u, 1088u, 6144u, 32768u}) { run_pingpong(rows, base); }
    }
#ifdef DS4_CUDA_HAVE_MXF4
    for (auto kernel : {ink2a_prefill_q4<INK2A_ONE>, ink2a_prefill_q4<INK2A_FIRST>, ink2a_prefill_q4<INK2A_NEXT>,
            ink2a_prefill_q4<INK2A_LAST>}) {
        check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_Q4_SMEM));
    }
    for (uint32_t rows : {9u, 63u, 64u, 65u, 129u, 2048u}) {
        for (uint32_t base : {0u, 1023u, 1087u, 1088u, 6144u, 32768u}) { run_fp4(rows, base); }
    }
#endif
    for (uint32_t base : {0u, 7u, 62u, 63u, 64u, 127u, 128u, 1022u, 1023u, 1024u, 1087u, 1088u, 3071u, 9217u,
            32768u, 65537u}) {
        run_split(1, base);
        run_split(3, base);
    }
}
