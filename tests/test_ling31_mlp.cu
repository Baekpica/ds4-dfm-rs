#include "cuda/l31_swiglu.cuh"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { const cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s at %d\n", cudaGetErrorString(e), __LINE__); return false; } } while (0)

enum class Values { Normal, Hot };

static bool run(uint32_t rows, uint32_t cols, float limit, Values values) {
    const uint64_t count = (uint64_t)rows * cols, bytes = glmvq_q4p_bytes(rows, cols);
    std::mt19937 rng(31);
    std::normal_distribution<float> normal(0.0f, 2.0f);
    std::vector<float> gate(count), up(count);
    for (uint64_t i = 0; i < count; i++) { gate[i] = normal(rng); up[i] = normal(rng); }
    if (values == Values::Hot) { for (uint32_t i = 0; i < cols; i++) { up[(uint64_t)2 * cols + i] *= 1e6f; } }
    if (std::isfinite(limit)) { gate[0] = NAN; up[1] = INFINITY; up[2] = -INFINITY; }
    float *g, *u, *mid;
    void *old, *fused;
    CK(cudaMalloc(&g, count * 4)); CK(cudaMalloc(&u, count * 4)); CK(cudaMalloc(&mid, count * 4));
    CK(cudaMalloc(&old, bytes)); CK(cudaMalloc(&fused, bytes));
    CK(cudaMemcpy(g, gate.data(), count * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(u, up.data(), count * 4, cudaMemcpyHostToDevice));
    const auto a = glmvq_q4p_layout(old, rows, cols), b = glmvq_q4p_layout(fused, rows, cols);
    auto separate = [&]() {
        glmvq_swiglu<float, GLMVQ_CLAMP_SILU><<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS>>>(
            g, u, mid, count, limit);
        return glmvq_q4p_prepare(a, mid, rows, cols, 0);
    };
    auto joined = [&]() { l31_swiglu_prepare<<<rows, GLMVQ_Q4P_PREP>>>(g, u, b.xh, b.xscale, cols, limit); };
    CK(separate()); joined(); CK(cudaDeviceSynchronize());
    std::vector<uint8_t> ref(bytes), got(bytes);
    CK(cudaMemcpy(ref.data(), old, bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(got.data(), fused, bytes, cudaMemcpyDeviceToHost));
    const bool same = ref == got;
    printf("MLP prepare rows=%u cols=%u hot=%d %s\n", rows, cols, values == Values::Hot, same ? "identical" : "DIFFER");
    if (rows == 4096) {
        cudaEvent_t start, end;
        CK(cudaEventCreate(&start)); CK(cudaEventCreate(&end));
        for (int rep = 0; rep < 5; rep++) {
            for (bool join : {false, true}) {
                CK(cudaEventRecord(start));
                for (int i = 0; i < 20; i++) { if (join) { joined(); } else { CK(separate()); } }
                CK(cudaEventRecord(end)); CK(cudaEventSynchronize(end));
                float ms; CK(cudaEventElapsedTime(&ms, start, end));
                printf("  %s %.6f ms\n", join ? "fused" : "separate", ms / 20);
            }
        }
        CK(cudaEventDestroy(start)); CK(cudaEventDestroy(end));
    }
    cudaFree(g); cudaFree(u); cudaFree(mid); cudaFree(old); cudaFree(fused);
    return same;
}

int main() {
    return !(run(3, 256, INFINITY, Values::Normal) && run(257, 1280, 4.0f, Values::Normal) &&
        run(300, 1280, INFINITY, Values::Hot) && run(17, 10240, INFINITY, Values::Hot) &&
        run(4096, 1280, 4.0f, Values::Normal));
}
