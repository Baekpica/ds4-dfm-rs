// Verify packed asymmetric KV, wrapped writes, and untouched resident rows.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../cuda/mimo2_primitives.cuh"

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(status));
        exit(1);
    }
}

int main() {
    for (unsigned heads : {4u, 8u}) {
        for (unsigned rows : {1u, 7u, 128u}) {
            for (unsigned position : {0u, 127u, 4095u, 262144u, 1048575u}) {
                const unsigned capacity = 256, kw = heads * 192, vw = heads * 128;
                const unsigned stride = kw + vw;
                std::vector<float> k(rows * kw), v(rows * vw);
                std::vector<__half> expected(capacity * stride, __float2half(-3.0f));
                std::vector<__half> result = expected;
                for (unsigned i = 0; i < k.size(); i++) { k[i] = sinf(i * 0.017f); }
                for (unsigned i = 0; i < v.size(); i++) { v[i] = cosf(i * 0.031f); }
                for (unsigned row = 0; row < rows; row++) {
                    const unsigned slot = (position + row) % capacity;
                    for (unsigned col = 0; col < kw; col++) {
                        expected[slot * stride + col] = __float2half_rn(k[row * kw + col]);
                    }
                    for (unsigned col = 0; col < vw; col++) {
                        expected[slot * stride + kw + col] = __float2half_rn(v[row * vw + col]);
                    }
                }
                std::vector<unsigned> positions(rows);
                for (unsigned row = 0; row < rows; row++) { positions[row] = position + row; }
                unsigned *dp;
                check(cudaMalloc(&dp, rows * sizeof(unsigned)));
                check(cudaMemcpy(dp, positions.data(), rows * sizeof(unsigned), cudaMemcpyHostToDevice));
                __half *cache;
                float *dk, *dv;
                check(cudaMalloc(&cache, result.size() * sizeof(__half)));
                check(cudaMalloc(&dk, k.size() * sizeof(float)));
                check(cudaMalloc(&dv, v.size() * sizeof(float)));
                check(cudaMemcpy(cache, result.data(), result.size() * sizeof(__half), cudaMemcpyHostToDevice));
                check(cudaMemcpy(dk, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice));
                check(cudaMemcpy(dv, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
                const unsigned count = rows * stride;
                mimo2_kv_store<<<(count + 255) / 256, 256>>>(cache, dk, dv, heads, rows, dp, capacity);
                check(cudaGetLastError());
                check(cudaMemcpy(result.data(), cache, result.size() * sizeof(__half), cudaMemcpyDeviceToHost));
                if (memcmp(result.data(), expected.data(), result.size() * sizeof(__half))) { return 2; }
                cudaStream_t stream;
                cudaGraph_t graph;
                cudaGraphExec_t replay;
                check(cudaStreamCreate(&stream));
                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                mimo2_kv_store<<<(count + 255) / 256, 256, 0, stream>>>(
                    cache, dk, dv, heads, rows, dp, capacity);
                check(cudaStreamEndCapture(stream, &graph));
                check(cudaGraphInstantiate(&replay, graph, nullptr, nullptr, 0));
                // Change positions after capture; replay must use the new ring slots.
                for (unsigned row = 0; row < rows; row++) {
                    positions[row]++;
                    const unsigned slot = positions[row] % capacity;
                    for (unsigned col = 0; col < kw; col++) {
                        expected[slot * stride + col] = __float2half_rn(k[row * kw + col]);
                    }
                    for (unsigned col = 0; col < vw; col++) {
                        expected[slot * stride + kw + col] = __float2half_rn(v[row * vw + col]);
                    }
                }
                check(cudaMemcpy(dp, positions.data(), rows * sizeof(unsigned), cudaMemcpyHostToDevice));
                check(cudaGraphLaunch(replay, stream));
                check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(result.data(), cache, result.size() * sizeof(__half), cudaMemcpyDeviceToHost));
                if (memcmp(result.data(), expected.data(), result.size() * sizeof(__half))) { return 3; }
                check(cudaGraphExecDestroy(replay)); check(cudaGraphDestroy(graph));
                check(cudaStreamDestroy(stream));
                printf("kv_heads=%u rows=%u pos=%u eager_and_replay=byte_exact\n", heads, rows, position);
                check(cudaFree(dp)); check(cudaFree(cache)); check(cudaFree(dk)); check(cudaFree(dv));
            }
        }
    }
    return 0;
}
