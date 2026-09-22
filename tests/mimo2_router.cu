// Tests route selection and unbiased normalization, independent of model loading.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>
#include "../cuda/mimo2_primitives.cuh"

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(status));
        exit(1);
    }
}

int main() {
    enum { EXPERTS = 256, USED = 8, ROWS = 17 };
    std::vector<float> x(ROWS * EXPERTS), bias(EXPERTS), out(ROWS * USED);
    std::vector<int> ids(ROWS * USED);
    for (unsigned e = 0; e < EXPERTS; e++) { bias[e] = 0.3f * cosf(e * 0.071f); }
    for (unsigned i = 0; i < x.size(); i++) { x[i] = 3 * sinf(i * 0.013f); }
    // Equal-score ties, saturated negative logits, and saturated positive logits.
    std::fill(x.begin(), x.begin() + EXPERTS, 0.0f);
    std::fill(x.begin() + EXPERTS, x.begin() + 2 * EXPERTS, -100.0f);
    std::fill(x.begin() + 2 * EXPERTS, x.begin() + 3 * EXPERTS, 100.0f);
    float *dx, *db, *dw;
    int *di;
    check(cudaMalloc(&dx, x.size() * sizeof(float)));
    check(cudaMalloc(&db, bias.size() * sizeof(float)));
    check(cudaMalloc(&dw, out.size() * sizeof(float)));
    check(cudaMalloc(&di, ids.size() * sizeof(int)));
    check(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(db, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    for (unsigned mode = 0; mode < 2; mode++) {
        mimo2_router<<<ROWS, 128>>>(di, dw, dx, mode ? db : nullptr);
        check(cudaGetLastError());
        check(cudaMemcpy(ids.data(), di, ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(out.data(), dw, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        float error = 0;
        for (unsigned row = 0; row < ROWS; row++) {
            std::vector<float> prob(EXPERTS), score(EXPERTS);
            std::vector<int> order(EXPERTS);
            std::iota(order.begin(), order.end(), 0);
            for (unsigned e = 0; e < EXPERTS; e++) {
                prob[e] = 1.0f / (1.0f + expf(-x[row * EXPERTS + e]));
                score[e] = prob[e] + (mode ? bias[e] : 0.0f);
            }
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                return score[a] > score[b];
            });
            double sum = 0;
            for (unsigned k = 0; k < USED; k++) { sum += prob[order[k]]; }
            for (unsigned k = 0; k < USED; k++) {
                const unsigned i = row * USED + k;
                if (ids[i] != order[k] || !std::isfinite(out[i])) { return 2; }
                const float expected = prob[order[k]] / std::max(sum, 0x1p-14);
                error = fmaxf(error, fabsf(out[i] - expected));
            }
        }
        printf("bias=%u rows=%u ids=exact max_abs_error=%.9g\n", mode, ROWS, error);
        if (error > 2e-7f) { return 3; }
    }
    // Invalid selection inputs must remain observable as non-finite weights.
    x[0] = NAN;
    check(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    mimo2_router<<<ROWS, 128>>>(di, dw, dx, db);
    check(cudaGetLastError());
    check(cudaMemcpy(out.data(), dw, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (unsigned k = 0; k < USED; k++) {
        if (!std::isnan(out[k])) { return 4; }
    }
    puts("nonfinite_input=propagated");
    check(cudaFree(dx)); check(cudaFree(db)); check(cudaFree(dw)); check(cudaFree(di));
    return 0;
}
