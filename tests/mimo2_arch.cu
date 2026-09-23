// Run native sm_121 and compute_75 PTX on GB10. Old PTX must use the
// walking/L2 fallback even though the physical GPU supports HMMA.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../cuda/mimo2_primitives.cuh"
#include "../cuda/mimo2_prefill.cuh"

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(status));
        exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s EXPECTED_PTX_VERSION\n", argv[0]);
        return 2;
    }
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, mimo2_hmma::prefill<mimo2_hmma::Scalar>));
    const int expected_ptx = atoi(argv[1]);
    const bool supported = mimo2_hmma::supported();
    printf("ptx=%d binary=%d hmma_supported=%d\n",
           attributes.ptxVersion, attributes.binaryVersion, (int)supported);
    if (attributes.ptxVersion != expected_ptx || supported != (expected_ptx >= 80)) {
        return 3;
    }

    enum { ROWS = 65, HEADS = 64, KV = 4, KEY = 192, VALUE = 128,
           START = 128, CAP = START + ROWS, STRIDE = KV * (KEY + VALUE) };
    std::vector<__half> cache(CAP * STRIDE, __float2half(0));
    std::vector<float> q(ROWS * HEADS * KEY, 0), result(ROWS * HEADS * VALUE);
    std::vector<unsigned> positions(ROWS);
    // QK is zero: each output is the independently computed prefix mean
    // of the exact stored FP16 V values. Every row/head/value is checked.
    std::vector<double> prefix((CAP + 1) * KV * VALUE, 0);
    for (unsigned p = 0; p < CAP; p++) {
        for (unsigned h = 0; h < KV; h++) {
            for (unsigned d = 0; d < VALUE; d++) {
                const __half value = __float2half_rn(p * 0.001f + h * 0.01f + d * 0.0005f);
                cache[p * STRIDE + KV * KEY + h * VALUE + d] = value;
                prefix[(p + 1) * KV * VALUE + h * VALUE + d] =
                    prefix[p * KV * VALUE + h * VALUE + d] + __half2float(value);
            }
        }
    }
    for (unsigned row = 0; row < ROWS; row++) { positions[row] = START + row; }
    float *dq, *out;
    __half *dc;
    unsigned *dp;
    check(cudaMalloc(&dq, q.size() * sizeof(float)));
    check(cudaMalloc(&out, result.size() * sizeof(float)));
    check(cudaMalloc(&dc, cache.size() * sizeof(__half)));
    check(cudaMalloc(&dp, positions.size() * sizeof(unsigned)));
    check(cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dc, cache.data(), cache.size() * sizeof(__half), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dp, positions.data(), positions.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
    for (int path = 0; path < 3; path++) {
        check(cudaMemset(out, 0xff, result.size() * sizeof(float)));
        if (path == 0) {
            mimo2_attention<<<dim3(HEADS / 4, ROWS), 128>>>(
                out, dq, dc, nullptr, dp, KV, CAP, 0);
        } else if (path == 1 || !supported) {
            mimo2_attn_l2<<<dim3(ROWS, KV), 512>>>(out, dq, dc, nullptr, dp, KV, CAP);
        } else {
            mimo2_hmma::prefill<mimo2_hmma::Async><<<dim3((ROWS + 63) / 64, HEADS), 128>>>(
                out, dq, dc, nullptr, dp, ROWS, KV, CAP);
        }
        check(cudaGetLastError());
        check(cudaMemcpy(result.data(), out, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double error = 0;
        for (unsigned row = 0; row < ROWS; row++) {
            const unsigned count = positions[row] + 1;
            for (unsigned head = 0; head < HEADS; head++) {
                const unsigned kv_head = head / (HEADS / KV);
                for (unsigned d = 0; d < VALUE; d++) {
                    const float actual = result[(row * HEADS + head) * VALUE + d];
                    const double expected = prefix[count * KV * VALUE + kv_head * VALUE + d] / count;
                    if (!std::isfinite(actual)) { return 4; }
                    error = std::max(error, fabs(actual - expected));
                }
            }
        }
        printf("path=%d prefix_mean_max_error=%.9g\n", path, error);
        if (error > 2e-5) { return 5; }
    }
    check(cudaFree(dq)); check(cudaFree(out)); check(cudaFree(dc)); check(cudaFree(dp));
    return 0;
}
