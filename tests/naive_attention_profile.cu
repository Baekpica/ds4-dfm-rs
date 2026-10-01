/* Bounded reproduction of the 8K attention walk: production geometry/layout,
 * synthetic BF16 operands and sorted scattered IDs, resident owner unused. */
#include "../cuda/naive_primitives.cuh"
#include "../cuda/naive_sparse_tile.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA(call) do { const cudaError_t rc = (call); if (rc != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(rc)); exit(1); } } while (0)

enum class AttentionPath : unsigned {
    Walk, Cache, Tile, DirectTile, DirectWalk, UnitWalk, UnitCache, RingCache, UnitRingCache
};

template<class T> static T *upload(const std::vector<T> &values) {
    T *out;
    CUDA(cudaMalloc(&out, values.size() * sizeof(T)));
    CUDA(cudaMemcpy(out, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return out;
}

template<unsigned WARPS> static void launch(
        float *out, const float *q, const __nv_bfloat16 *kv, const float *sinks,
        const unsigned *pos, const unsigned *ids, unsigned rows,
        unsigned heads, unsigned capacity, unsigned window, AttentionPath path) {
    const dim3 grid(N05_HEADS / WARPS, rows);
    if (path == AttentionPath::UnitRingCache && window) {
        naive_attention<WARPS, N05_WINDOW, NaiveCache::Ring, NaiveSoftmax::Unit, NaiveRing::Walk>
            <<<grid, WARPS * 32>>>(out, q, kv, sinks, pos, nullptr, heads, capacity, window);
    } else if (path == AttentionPath::RingCache && window) {
        naive_attention<WARPS, N05_WINDOW, NaiveCache::Ring, NaiveSoftmax::Walk, NaiveRing::Walk>
            <<<grid, WARPS * 32>>>(out, q, kv, sinks, pos, nullptr, heads, capacity, window);
    } else if (path == AttentionPath::UnitCache && window) {
        naive_attention<WARPS, N05_WINDOW, NaiveCache::Ring, NaiveSoftmax::Unit><<<grid, WARPS * 32>>>(
            out, q, kv, sinks, pos, nullptr, heads, capacity, window);
    } else if (path == AttentionPath::UnitWalk) {
        naive_attention<WARPS, 0, NaiveCache::Full, NaiveSoftmax::Unit><<<grid, WARPS * 32>>>(
            out, q, kv, nullptr, pos, ids, heads, capacity, 0);
    } else if (path == AttentionPath::DirectWalk) {
        naive_attention<WARPS, 0, NaiveCache::Full><<<grid, WARPS * 32>>>(out, q, kv, nullptr,
            pos, ids, heads, capacity, 0);
    } else if (path == AttentionPath::DirectTile) {
        naive_sparse_tile<NaiveCache::Full><<<dim3(N05_HEADS, rows), 128>>>(out, q, kv, pos, ids, capacity);
    } else if (path == AttentionPath::Tile) {
        naive_sparse_tile<<<dim3(N05_HEADS, rows), 128>>>(out, q, kv, pos, ids, capacity);
    } else if (path == AttentionPath::Cache && window) {
        naive_attention<WARPS, N05_WINDOW><<<grid, WARPS * 32>>>(out, q, kv, sinks,
            pos, nullptr, heads, capacity, window);
    } else if (path == AttentionPath::Cache) {
        naive_attention<WARPS, N05_TOP_K><<<grid, WARPS * 32>>>(out, q, kv, nullptr,
            pos, ids, heads, capacity, window);
    } else {
        naive_attention<WARPS><<<grid, WARPS * 32>>>(out, q, kv, sinks,
            pos, ids, heads, capacity, window);
    }
    CUDA(cudaGetLastError());
}

int main(int argc, char **argv) {
    enum { HISTORY = 8192, REPEATS = 20 };
    if (argc > 7) { return 2; }
    const unsigned rows = argc >= 2 ? (unsigned)atoi(argv[1]) : N05_QUERY_TILE;
    const unsigned warps = argc >= 3 ? (unsigned)atoi(argv[2]) : 4;
    const auto path = argc >= 4 ? static_cast<AttentionPath>((unsigned)atoi(argv[3])) : AttentionPath::Walk;
    const unsigned window = argc >= 5 ? (unsigned)atoi(argv[4]) : 0;
    const unsigned limit = window ? N05_PREFILL : N05_QUERY_TILE;
    const bool swa_path = path == AttentionPath::UnitCache || path == AttentionPath::RingCache ||
        path == AttentionPath::UnitRingCache;
    if (!rows || rows > limit || (warps != 1 && warps != 4) || path > AttentionPath::UnitRingCache ||
        (path >= AttentionPath::Tile && path <= AttentionPath::DirectWalk && (window || warps != 4)) ||
        (path == AttentionPath::UnitWalk && window) || (swa_path && !window) ||
        (window && window != N05_WINDOW)) { return 2; }
    const unsigned heads = window ? 8 : 4;
    const unsigned stride = heads * (N05_KEY + N05_VALUE);
    const unsigned full_capacity = argc >= 6 ? (unsigned)atoi(argv[5]) : HISTORY;
    const unsigned capacity = window && argc < 6 ? rows + N05_WINDOW - 1 : full_capacity;
    if (capacity > N05_CONTEXT || capacity < rows || (!window && capacity < HISTORY)) { return 2; }
    const unsigned last_pos = argc == 7 ? (unsigned)atoi(argv[6]) : HISTORY - 1;
    if ((argc == 7 && !window) || last_pos >= N05_CONTEXT || rows > last_pos + 1) { return 2; }
    std::vector<__nv_bfloat16> cache((size_t)capacity * stride);
    std::vector<float> sink(N05_HEADS);
    std::vector<float> query((size_t)rows * N05_HEADS * N05_KEY);
    std::vector<unsigned> positions(rows), selected((size_t)rows * N05_TOP_K);
    for (size_t i = 0; i < cache.size(); i++) {
        cache[i] = __float2bfloat16_rn(((int)((i * 13 + i / stride * 19) % 127) - 63) * .0078125f);
    }
    for (size_t i = 0; i < query.size(); i++) {
        query[i] = __bfloat162float(__float2bfloat16_rn(((int)((i * 7) % 113) - 56) * .015625f));
    }
    for (unsigned r = 0; r < rows; r++) {
        positions[r] = last_pos + 1 - rows + r;
        for (unsigned i = 0; i < N05_TOP_K; i++) { selected[r * N05_TOP_K + i] = i * (positions[r] + 1) / N05_TOP_K; }
    }
    auto *kv = upload(cache);
    float *sinks = window ? upload(sink) : nullptr;
    float *q = upload(query), *out, *reference;
    unsigned *pos = upload(positions), *ids = upload(selected);
    CUDA(cudaMalloc(&out, (size_t)rows * N05_HEADS * N05_VALUE * sizeof(float)));
    CUDA(cudaMalloc(&reference, (size_t)rows * N05_HEADS * N05_VALUE * sizeof(float)));
    launch<4>(reference, q, kv, sinks, pos, ids, rows, heads, capacity, window, AttentionPath::Walk);
    CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, end;
    CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
    for (unsigned repeat = 0; repeat <= REPEATS; repeat++) {
        if (repeat == 1) { CUDA(cudaEventRecord(start)); }
        if (warps == 1) {
            launch<1>(out, q, kv, sinks, pos, ids, rows, heads, capacity, window, path);
        } else {
            launch<4>(out, q, kv, sinks, pos, ids, rows, heads, capacity, window, path);
        }
        if (!repeat) { CUDA(cudaDeviceSynchronize()); }
    }
    CUDA(cudaEventRecord(end)); CUDA(cudaEventSynchronize(end));
    float elapsed;
    CUDA(cudaEventElapsedTime(&elapsed, start, end));
    std::vector<float> result((size_t)rows * N05_HEADS * N05_VALUE);
    std::vector<float> expected(result.size());
    CUDA(cudaMemcpy(result.data(), out, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(expected.data(), reference, expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (float value : result) { assert(std::isfinite(value)); }
    assert(!memcmp(result.data(), expected.data(), result.size() * sizeof(float)));
    printf("attention rows=%u warps=%u path=%u window=%u selected=%u history=%u capacity=%u last_pos=%u milliseconds=%.6f byte_exact=1\n",
        rows, warps, static_cast<unsigned>(path), window, window ? window : N05_TOP_K,
        last_pos + 1, capacity, last_pos, elapsed / REPEATS);
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
    CUDA(cudaFree(out)); CUDA(cudaFree(reference)); CUDA(cudaFree(q)); CUDA(cudaFree(kv));
    CUDA(cudaFree(sinks)); CUDA(cudaFree(pos)); CUDA(cudaFree(ids));
    return 0;
}
