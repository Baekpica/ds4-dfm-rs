// Random sinks and wrapped KV exercise the decode tile against FP64 softmax.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "../cuda/mimo2_primitives.cuh"

#define CK(call) do { if ((call) != cudaSuccess) { \
    fprintf(stderr, "CUDA failure at %d\n", __LINE__); return 1; } } while (0)

enum { HEADS = 64, KV_HEADS = 8, KEY = 192, VALUE = 128,
       WINDOW = 128, CAPACITY = 259, STRIDE = KV_HEADS * (KEY + VALUE) };

static std::vector<double> reference(const std::vector<float> &q,
        const std::vector<__half> &cache, const std::vector<float> &sinks,
        unsigned pos) {
    std::vector<double> out(HEADS * VALUE);
    const unsigned first = pos + 1 > WINDOW ? pos + 1 - WINDOW : 0;
    for (unsigned h = 0; h < HEADS; h++) {
        const unsigned kvh = h / (HEADS / KV_HEADS);
        std::vector<double> scores;
        double maximum = sinks[h];
        for (unsigned k = first; k <= pos; k++) {
            const size_t offset = (k % CAPACITY) * STRIDE + kvh * KEY;
            double dot = 0;
            for (unsigned d = 0; d < KEY; d++) {
                dot += double(q[h * KEY + d]) * __half2float(cache[offset + d]);
            }
            scores.push_back(dot / std::sqrt(double(KEY)));
            maximum = std::fmax(maximum, scores.back());
        }
        double denom = std::exp(double(sinks[h]) - maximum);
        for (double score : scores) { denom += std::exp(score - maximum); }
        for (unsigned k = first; k <= pos; k++) {
            const size_t offset = (k % CAPACITY) * STRIDE + KV_HEADS * KEY + kvh * VALUE;
            const double weight = std::exp(scores[k - first] - maximum) / denom;
            for (unsigned d = 0; d < VALUE; d++) {
                out[h * VALUE + d] += weight * __half2float(cache[offset + d]);
            }
        }
    }
    return out;
}

int main() {
    std::mt19937 rng(104729);
    std::normal_distribution<float> normal;
    std::vector<float> q(HEADS * KEY), sinks(HEADS), walk(HEADS * VALUE), got(walk.size());
    std::vector<__half> cache(CAPACITY * STRIDE);
    for (auto &v : q) { v = normal(rng); }
    for (auto &v : sinks) { v = normal(rng) * 2; }
    for (auto &v : cache) { v = __float2half_rn(normal(rng)); }
    float *dq, *ds, *out;
    __half *dc;
    unsigned *dp;
    CK(cudaMalloc(&dq, q.size() * sizeof(float)));
    CK(cudaMalloc(&ds, sinks.size() * sizeof(float)));
    CK(cudaMalloc(&out, got.size() * sizeof(float)));
    CK(cudaMalloc(&dc, cache.size() * sizeof(__half)));
    CK(cudaMalloc(&dp, sizeof(unsigned)));
    CK(cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds, sinks.data(), sinks.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dc, cache.data(), cache.size() * sizeof(__half), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CK(cudaStreamCreate(&stream));
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    mimo2_swa_decode<<<dim3(1, KV_HEADS), 256, 0, stream>>>(
        out, dq, dc, ds, dp, KV_HEADS, CAPACITY, WINDOW, 1);
    CK(cudaStreamEndCapture(stream, &graph));
    CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    for (unsigned pos : {0u, 31u, 127u, 128u, 258u, 259u, 8192u, 262144u}) {
        CK(cudaMemcpy(dp, &pos, sizeof(pos), cudaMemcpyHostToDevice));
        mimo2_attention<<<dim3(HEADS / 4, 1), 128>>>(
            out, dq, dc, ds, dp, KV_HEADS, CAPACITY, WINDOW);
        CK(cudaMemcpy(walk.data(), out, walk.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaGraphLaunch(exec, stream));
        CK(cudaStreamSynchronize(stream));
        CK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        auto ref = reference(q, cache, sinks, pos);
        double walk_err = 0, tile_err = 0, diff = 0;
        for (size_t i = 0; i < got.size(); i++) {
            if (!std::isfinite(got[i])) { return 1; }
            walk_err = std::fmax(walk_err, std::fabs(walk[i] - ref[i]));
            tile_err = std::fmax(tile_err, std::fabs(got[i] - ref[i]));
            diff = std::fmax(diff, std::fabs(double(got[i]) - walk[i]));
        }
        // Both paths use FP32 dot/softmax. Compare with the oracle's observed
        // rounding envelope; no token-level assumptions enter this test.
        const double limit = 4 * walk_err + 2e-6;
        printf("pos=%u walk_error=%.9g tile_error=%.9g diff=%.9g limit=%.9g\n",
               pos, walk_err, tile_err, diff, limit);
        if (tile_err > limit) { return 1; }
    }
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));
    CK(cudaStreamDestroy(stream));
    for (void *ptr : {(void *)dq, (void *)ds, (void *)out, (void *)dc, (void *)dp}) {
        CK(cudaFree(ptr));
    }
    return 0;
}
