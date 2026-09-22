// Standalone primitive test; does not establish native graph support.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../cuda/mimo2_primitives.cuh"
#include "../cuda/step37_primitives.cuh"

static void check(cudaError_t code) {
    if (code != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(code));
        exit(1);
    }
}

int main() {
    const unsigned positions[] = {0, 1, 127, 128, 4095, 65535, 262144, 1048575};
    const unsigned rows = sizeof(positions) / sizeof(positions[0]);
    for (unsigned kv : {4u, 8u}) {
        const unsigned qw = 64 * 192, kw = kv * 192, vw = kv * 128;
        const unsigned stride = qw + kw + vw;
        const float theta = kv == 4 ? 10000000.0f : 10000.0f;
        std::vector<float> input(rows * stride), expected(rows * stride), result(rows * stride);
        std::vector<float> frequency(32);
        for (unsigned d = 0; d < 32; d++) { frequency[d] = powf(theta, -(float)d / 32); }
        for (unsigned i = 0; i < input.size(); i++) { input[i] = sinf((float)i * 0.017f); }
        // Independent scalar NeoX rotation, followed by contiguous Q/K/V packing.
        for (unsigned row = 0; row < rows; row++) {
            for (unsigned col = 0; col < stride; col++) {
                const unsigned d = col % 192;
                float val = input[row * stride + col];
                if (col < qw + kw && d < 64) {
                    const unsigned pair = d % 32;
                    const float phase = positions[row] * frequency[pair];
                    const unsigned base = row * stride + col - d;
                    const double a = input[base + pair], b = input[base + pair + 32];
                    val = d < 32 ? a * cos((double)phase) - b * sin((double)phase)
                                 : b * cos((double)phase) + a * sin((double)phase);
                }
                unsigned dst = col < qw ? row * qw + col :
                    col < qw + kw ? rows * qw + row * kw + col - qw :
                    rows * (qw + kw) + row * vw + col - qw - kw;
                expected[dst] = val;
            }
        }
        float *src, *out, *freq;
        unsigned *pos;
        float2 *table;
        check(cudaMalloc(&src, input.size() * sizeof(float)));
        check(cudaMalloc(&out, result.size() * sizeof(float)));
        check(cudaMalloc(&freq, frequency.size() * sizeof(float)));
        check(cudaMalloc(&pos, sizeof(positions)));
        check(cudaMalloc(&table, rows * 32 * sizeof(float2)));
        check(cudaMemcpy(src, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(freq, frequency.data(), frequency.size() * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(pos, positions, sizeof(positions), cudaMemcpyHostToDevice));
        step37_rope_table<<<(rows * 32 + 127) / 128, 128>>>(table, freq, pos, 32, rows);
        mimo2_split_rope<<<(input.size() + 255) / 256, 256>>>(
            out, out + rows * qw, out + rows * (qw + kw), src, table, kv, rows);
        check(cudaGetLastError());
        check(cudaMemcpy(result.data(), out, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (unsigned row = 0; row < rows; row++) {
            float row_error = 0;
            for (unsigned col = 0; col < qw; col++) {
                const unsigned i = row * qw + col;
                if (!std::isfinite(result[i])) { return 2; }
                row_error = fmaxf(row_error, fabsf(result[i] - expected[i]));
            }
            for (unsigned col = 0; col < kw; col++) {
                const unsigned i = rows * qw + row * kw + col;
                if (!std::isfinite(result[i])) { return 2; }
                row_error = fmaxf(row_error, fabsf(result[i] - expected[i]));
            }
            for (unsigned col = 0; col < vw; col++) {
                const unsigned i = rows * (qw + kw) + row * vw + col;
                if (!std::isfinite(result[i])) { return 2; }
                row_error = fmaxf(row_error, fabsf(result[i] - expected[i]));
            }
            printf("kv_heads=%u pos=%u max_abs_error=%.9g\n", kv, positions[row], row_error);
            if (row_error > 5e-7f) { return 3; }
        }
        check(cudaFree(src)); check(cudaFree(out)); check(cudaFree(freq));
        check(cudaFree(pos)); check(cudaFree(table));
    }
    return 0;
}
