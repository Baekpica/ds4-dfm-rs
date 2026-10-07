/* Exact production merger kernel; under 1 MiB of model-free device buffers. */
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "../cuda/glm53_vision_norm.cuh"

enum { WIDTH = 4096, ROWS = 13, THREADS = 256, REPEATS = 64 };
static const float EPS = 1.0e-5f;
static const double MAX_ERROR = 2.5e-5;

static void check(cudaError_t status, const char *op) {
    if (status == cudaSuccess) { return; }
    fprintf(stderr, "GLM norm %s: %s\n", op, cudaGetErrorString(status));
    exit(1);
}

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(bits >> 16u);
}

static double from_bf16(uint16_t value) {
    const uint32_t bits = (uint32_t)value << 16u;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

int main(void) {
    std::vector<float> input((size_t)WIDTH * ROWS), output(input.size());
    std::vector<double> expected(input.size());
    std::vector<uint16_t> weights(WIDTH), bias(WIDTH);
    for (unsigned d = 0u; d < WIDTH; d++) {
        weights[d] = to_bf16(0.5f + (float)(d % 31u) * 0.03125f);
        bias[d] = to_bf16((float)((int)(d % 19u) - 9) * 0.0625f);
    }
    for (unsigned row = 0u; row < ROWS; row++) {
        double sum = 0.0;
        for (unsigned d = 0u; d < WIDTH; d++) {
            const size_t at = (size_t)row * WIDTH + d;
            input[at] = (float)((int)row - 4) * 1.13f +
                (float)((int)((d * 37u + row * 17u) % 257u) - 128) * 0.125f +
                sinf((float)d * 0.013f) * 0.3f;
            sum += input[at];
        }
        const double mean = sum / WIDTH;
        double variance = 0.0;
        for (unsigned d = 0u; d < WIDTH; d++) {
            const double centered = input[(size_t)row * WIDTH + d] - mean;
            variance += centered * centered;
        }
        const double inv = 1.0 / sqrt(variance / WIDTH + EPS);
        for (unsigned d = 0u; d < WIDTH; d++) {
            const size_t at = (size_t)row * WIDTH + d;
            const double x = (input[at] - mean) * inv * from_bf16(weights[d]) + from_bf16(bias[d]);
            expected[at] = 0.5 * x * (1.0 + erf(x / sqrt(2.0)));
        }
    }
    float *device_in = NULL, *device_out = NULL;
    uint16_t *device_weight = NULL, *device_bias = NULL;
    const size_t bytes = input.size() * sizeof(float);
    check(cudaMalloc(&device_in, bytes), "input alloc");
    check(cudaMalloc(&device_out, bytes), "output alloc");
    check(cudaMalloc(&device_weight, WIDTH * sizeof(uint16_t)), "weight alloc");
    check(cudaMalloc(&device_bias, WIDTH * sizeof(uint16_t)), "bias alloc");
    check(cudaMemcpy(device_in, input.data(), bytes, cudaMemcpyHostToDevice), "input write");
    check(cudaMemcpy(device_weight, weights.data(), WIDTH * sizeof(uint16_t), cudaMemcpyHostToDevice), "weight write");
    check(cudaMemcpy(device_bias, bias.data(), WIDTH * sizeof(uint16_t), cudaMemcpyHostToDevice), "bias write");
    double worst = 0.0;
    for (unsigned repeat = 0u; repeat < REPEATS; repeat++) {
        glm53_vision_layernorm_gelu_kernel<<<ROWS, THREADS>>>(
            device_out, device_in, device_weight, device_bias, WIDTH, EPS);
        check(cudaGetLastError(), "kernel launch");
        check(cudaMemcpy(output.data(), device_out, bytes, cudaMemcpyDeviceToHost), "output read");
        for (size_t i = 0u; i < output.size(); i++) {
            const double error = fabs((double)output[i] - expected[i]);
            if (!isfinite(output[i]) || error > MAX_ERROR) {
                fprintf(stderr, "GLM norm FAIL repeat=%u row=%zu col=%zu got=%.9g expected=%.9g abs=%.9g\n",
                    repeat, i / WIDTH, i % WIDTH, output[i], expected[i], error);
                return 1;
            }
            if (error > worst) { worst = error; }
        }
    }
    check(cudaFree(device_bias), "bias free");
    check(cudaFree(device_weight), "weight free");
    check(cudaFree(device_out), "output free");
    check(cudaFree(device_in), "input free");
    printf("GLM norm: rows=%u width=%u repeats=%u max_abs=%.9g PASS\n", ROWS, WIDTH, REPEATS, worst);
    return 0;
}
