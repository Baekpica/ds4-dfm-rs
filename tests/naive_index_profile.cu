/* Production score geometry and arithmetic; synthetic E4M3 history and
 * signed head weights. Cache residency and values differ from the model. */
#include "../cuda/naive_primitives.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA(call) do { const cudaError_t rc = (call); if (rc != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(rc)); exit(1); } } while (0)

template<class T> static T *upload(const std::vector<T> &values) {
    T *out;
    CUDA(cudaMalloc(&out, values.size() * sizeof(T)));
    CUDA(cudaMemcpy(out, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return out;
}

int main(int argc, char **argv) {
    enum { HISTORY = 32768, REPEATS = 20, WARPS = 4, THREADS = WARPS * 32 };
    const unsigned history = argc >= 2 ? (unsigned)atoi(argv[1]) : HISTORY;
    const unsigned rows = argc >= 3 ? (unsigned)atoi(argv[2]) : N05_QUERY_TILE;
    const auto layout = argc >= 4 ? static_cast<NaiveIndexLayout>((unsigned)atoi(argv[3])) : NaiveIndexLayout::Planar;
    const unsigned repeats = argc >= 5 ? (unsigned)atoi(argv[4]) : REPEATS;
    if (!rows || rows > N05_QUERY_TILE || history < rows || history > N05_CONTEXT ||
        layout > NaiveIndexLayout::Warp || !repeats || repeats > REPEATS) { return 2; }

    std::vector<uint8_t> codes((size_t)history * N05_INDEX_DIM);
    std::vector<float> scales(history), query((size_t)rows * N05_INDEX_HEADS * N05_INDEX_DIM);
    std::vector<float> weights((size_t)rows * N05_INDEX_HEADS);
    std::vector<unsigned> positions(rows);
    for (unsigned k = 0; k < history; k++) {
        scales[k] = .001234567f * (1 + k % 17);
        for (unsigned d = 0; d < N05_INDEX_DIM; d++) {
            codes[(size_t)k * N05_INDEX_DIM + d] = ((k * 13 + d * 19) % 127) | ((k + d) % 2 ? 128 : 0);
        }
    }
    for (size_t i = 0; i < query.size(); i++) { query[i] = ((int)((i * 7) % 113) - 56) * .0137f; }
    for (size_t i = 0; i < weights.size(); i++) { weights[i] = ((int)((i * 11) % 31) - 15) * .007123f; }
    for (unsigned r = 0; r < rows; r++) { positions[r] = history - rows + r; }

    auto *dc = upload(codes);
    auto *ds = upload(scales), *dq = upload(query), *dw = upload(weights), *packed = upload(query);
    auto *dp = upload(positions);
    float *out, *reference;
    CUDA(cudaMalloc(&out, (size_t)rows * history * sizeof(float)));
    CUDA(cudaMalloc(&reference, (size_t)rows * history * sizeof(float)));
    naive_fp8_query<<<rows * N05_INDEX_HEADS, N05_INDEX_DIM>>>(dq, rows * N05_INDEX_HEADS);
    naive_fp8_query<NaiveIndexLayout::Warp><<<rows * N05_INDEX_HEADS, N05_INDEX_DIM>>>(packed, rows * N05_INDEX_HEADS);
    naive_index_scores<<<dim3((history + WARPS - 1) / WARPS, rows), THREADS>>>(
        reference, dq, dc, ds, dw, dp, history);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, end;
    CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
    for (unsigned repeat = 0; repeat <= repeats; repeat++) {
        if (repeat == 1) { CUDA(cudaEventRecord(start)); }
        if (layout == NaiveIndexLayout::Warp) {
            naive_index_scores<NaiveIndexLayout::Warp><<<dim3((history + WARPS - 1) / WARPS, rows), THREADS>>>(
                out, packed, dc, ds, dw, dp, history);
        } else {
            naive_index_scores<<<dim3((history + WARPS - 1) / WARPS, rows), THREADS>>>(
                out, dq, dc, ds, dw, dp, history);
        }
        CUDA(cudaGetLastError());
        if (!repeat) { CUDA(cudaDeviceSynchronize()); }
    }
    CUDA(cudaEventRecord(end)); CUDA(cudaEventSynchronize(end));
    float elapsed;
    CUDA(cudaEventElapsedTime(&elapsed, start, end));
    std::vector<float> result((size_t)rows * history);
    std::vector<float> expected(result.size());
    CUDA(cudaMemcpy(result.data(), out, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(expected.data(), reference, expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    assert(!memcmp(result.data(), expected.data(), result.size() * sizeof(float)));
    for (unsigned r = 0; r < rows; r++) {
        for (unsigned k = 0; k < history; k++) {
            const float score = result[(size_t)r * history + k];
            assert(k <= positions[r] ? std::isfinite(score) : score == -INFINITY);
        }
    }
    printf("index history=%u rows=%u layout=%u heads=%u dim=%u milliseconds=%.6f byte_exact=1\n",
        history, rows, (unsigned)layout, N05_INDEX_HEADS, N05_INDEX_DIM, elapsed / repeats);
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
    CUDA(cudaFree(dc)); CUDA(cudaFree(ds)); CUDA(cudaFree(dq));
    CUDA(cudaFree(dw)); CUDA(cudaFree(dp)); CUDA(cudaFree(out));
    CUDA(cudaFree(packed)); CUDA(cudaFree(reference));
    return 0;
}
