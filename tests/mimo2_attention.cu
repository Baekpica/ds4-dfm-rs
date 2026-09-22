// Scalar softmax oracle includes causal/SWA masks, GQA head mapping and sinks.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../cuda/mimo2_primitives.cuh"

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(status));
        exit(1);
    }
}

int main() {
    enum { HEADS = 64, KEY = 192, VALUE = 128, ROWS = 3 };
    for (unsigned kv : {4u, 8u}) {
        for (unsigned start : {5u, 129u, 4095u, 262144u}) {
            for (unsigned use_sink : {0u, 1u}) {
                const unsigned window = kv == 8 ? 128 : 0;
                // Full attention's scalar oracle is linear in position.
                // The high index is the SWA-128 ring, which stays window-sized.
                if (window == 0 && start > 4095u) { continue; }
                const unsigned capacity = window ? std::min(256u, start + ROWS) : start + ROWS;
                const unsigned stride = kv * (KEY + VALUE);
                std::vector<__half> cache(capacity * stride);
                std::vector<float> q(ROWS * HEADS * KEY), out(ROWS * HEADS * VALUE);
                std::vector<float> sinks(HEADS);
                std::vector<unsigned> positions(ROWS);
                for (unsigned row = 0; row < ROWS; row++) { positions[row] = start + row; }
                for (unsigned i = 0; i < q.size(); i++) { q[i] = sinf(i * 0.013f); }
                for (unsigned h = 0; h < HEADS; h++) { sinks[h] = h % 3 == 0 ? 12 : h * 0.03f; }
                for (unsigned p = 0; p < start + ROWS; p++) {
                    for (unsigned d = 0; d < stride; d++) {
                        cache[(p % capacity) * stride + d] = __float2half_rn(sinf(p * 0.047f + d * 0.017f));
                    }
                }
                float *dq, *ds, *dout;
                __half *dc;
                unsigned *dp;
                check(cudaMalloc(&dq, q.size() * sizeof(float)));
                check(cudaMalloc(&ds, sinks.size() * sizeof(float)));
                check(cudaMalloc(&dout, out.size() * sizeof(float)));
                check(cudaMalloc(&dc, cache.size() * sizeof(__half)));
                check(cudaMalloc(&dp, positions.size() * sizeof(unsigned)));
                check(cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
                check(cudaMemcpy(ds, sinks.data(), sinks.size() * sizeof(float), cudaMemcpyHostToDevice));
                check(cudaMemcpy(dc, cache.data(), cache.size() * sizeof(__half), cudaMemcpyHostToDevice));
                check(cudaMemcpy(dp, positions.data(), positions.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
                mimo2_attention<<<dim3(HEADS / 4, ROWS), 128>>>(
                    dout, dq, dc, use_sink ? ds : nullptr, dp, kv, capacity, window);
                check(cudaGetLastError());
                check(cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
                double error = 0;
                for (unsigned row = 0; row < ROWS; row++) {
                    const unsigned pos = positions[row];
                    const unsigned first = window && pos + 1 > window ? pos + 1 - window : 0;
                    for (unsigned h = 0; h < HEADS; h++) {
                        const unsigned kh = h / (HEADS / kv);
                        std::vector<double> score(pos - first + 1);
                        double maximum = use_sink ? sinks[h] : -INFINITY;
                        for (unsigned p = first; p <= pos; p++) {
                            double dot = 0;
                            for (unsigned d = 0; d < KEY; d++) {
                                dot += (double)q[(row * HEADS + h) * KEY + d] *
                                    __half2float(cache[(p % capacity) * stride + kh * KEY + d]);
                            }
                            score[p - first] = dot / sqrt((double)KEY);
                            maximum = std::max(maximum, score[p - first]);
                        }
                        double denominator = use_sink ? exp(sinks[h] - maximum) : 0;
                        for (double &s : score) { s = exp(s - maximum); denominator += s; }
                        for (unsigned d = 0; d < VALUE; d++) {
                            double value = 0;
                            for (unsigned p = first; p <= pos; p++) {
                                value += score[p - first] * __half2float(
                                    cache[(p % capacity) * stride + kv * KEY + kh * VALUE + d]);
                            }
                            const float actual = out[(row * HEADS + h) * VALUE + d];
                            if (!std::isfinite(actual)) { return 2; }
                            error = std::max(error, fabs(actual - value / denominator));
                        }
                    }
                }
                printf("kv=%u start=%u sink=%u max_abs_error=%.9g\n", kv, start, use_sink, error);
                if (error > 2e-5) { return 3; }
                cudaStream_t stream;
                cudaGraph_t graph;
                cudaGraphExec_t replay;
                check(cudaStreamCreate(&stream));
                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                mimo2_attention<<<dim3(HEADS / 4, ROWS), 128, 0, stream>>>(
                    dout, dq, dc, use_sink ? ds : nullptr, dp, kv, capacity, window);
                check(cudaStreamEndCapture(stream, &graph));
                check(cudaGraphInstantiate(&replay, graph, nullptr, nullptr, 0));
                for (unsigned &p : positions) { p--; }
                check(cudaMemcpy(dp, positions.data(), positions.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
                mimo2_attention<<<dim3(HEADS / 4, ROWS), 128>>>(
                    dout, dq, dc, use_sink ? ds : nullptr, dp, kv, capacity, window);
                check(cudaGetLastError());
                check(cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
                std::vector<float> replayed(out.size());
                check(cudaGraphLaunch(replay, stream));
                check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(replayed.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
                if (out != replayed) { return 4; }
                check(cudaGraphExecDestroy(replay)); check(cudaGraphDestroy(graph));
                check(cudaStreamDestroy(stream));
                puts("changed_position_capture=eager_exact");
                check(cudaFree(dq)); check(cudaFree(ds)); check(cudaFree(dout));
                check(cudaFree(dc)); check(cudaFree(dp));
            }
        }
    }
    return 0;
}
