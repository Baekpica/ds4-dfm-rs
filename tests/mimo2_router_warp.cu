// Preserve expert ties, nonfinite rejection, and selected-weight arithmetic.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "../cuda/mimo2_primitives.cuh"

#define CK(call) do { if ((call) != cudaSuccess) { \
    fprintf(stderr, "CUDA failure at %d\n", __LINE__); return 1; } } while (0)

enum { EXPERTS = 256, USED = 8 };
enum class Input { Random, Ties, Floor, Nonfinite };

static int run_case(unsigned rows, Input input) {
    std::mt19937 rng(104729);
    std::uniform_real_distribution<float> random(-8, 8);
    std::vector<float> logits(rows * EXPERTS), bias(EXPERTS);
    for (auto &v : logits) { v = random(rng); }
    for (auto &v : bias) { v = random(rng) * 0.1f; }
    if (input == Input::Ties || input == Input::Floor) {
        for (auto &v : logits) { v = input == Input::Ties ? 0 : -1000; }
        for (auto &v : bias) { v = 0; }
    }
    if (input == Input::Nonfinite) { bias[77] = NAN; }
    float *dl, *db, *dw;
    int *di;
    CK(cudaMalloc(&dl, logits.size() * sizeof(float)));
    CK(cudaMalloc(&db, bias.size() * sizeof(float)));
    CK(cudaMalloc(&dw, rows * USED * sizeof(float)));
    CK(cudaMalloc(&di, rows * USED * sizeof(int)));
    CK(cudaMemcpy(dl, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<int> ref_ids(rows * USED), got_ids(ref_ids.size());
    std::vector<float> ref(rows * USED), got(ref.size());
    cudaEvent_t begin, end;
    CK(cudaEventCreate(&begin));
    CK(cudaEventCreate(&end));
    float times[2];
    const int repeats = 128;
    for (unsigned mode = 0; mode < 2; mode++) {
        CK(cudaEventRecord(begin));
        for (int i = 0; i < repeats; i++) {
            if (mode) { mimo2_router_warp<<<rows, 32>>>(di, dw, dl, db); }
            else { mimo2_router<<<rows, 128>>>(di, dw, dl, db); }
        }
        CK(cudaEventRecord(end));
        CK(cudaEventSynchronize(end));
        CK(cudaEventElapsedTime(&times[mode], begin, end));
        auto &ids = mode ? got_ids : ref_ids;
        auto &weights = mode ? got : ref;
        CK(cudaMemcpy(ids.data(), di, ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(weights.data(), dw, weights.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (ref_ids != got_ids) { fprintf(stderr, "expert mismatch\n"); return 1; }
    if (input == Input::Nonfinite) {
        for (float v : got) { if (!std::isnan(v)) { return 1; } }
    } else if (memcmp(ref.data(), got.data(), ref.size() * sizeof(float))) {
        fprintf(stderr, "weight mismatch\n"); return 1;
    }
    printf("rows=%u input=%d exact=1 serial_us=%.4f warp_us=%.4f\n",
           rows, (int)input, times[0] * 1000 / repeats, times[1] * 1000 / repeats);
    CK(cudaEventDestroy(begin));
    CK(cudaEventDestroy(end));
    for (void *ptr : {(void *)dl, (void *)db, (void *)dw, (void *)di}) { CK(cudaFree(ptr)); }
    return 0;
}

int main() {
    for (unsigned rows : {1u, 2u, 8u, 32u, 129u}) {
        for (auto input : {Input::Random, Input::Ties, Input::Floor, Input::Nonfinite}) {
            if (run_case(rows, input)) { return 1; }
        }
    }
    return 0;
}
