// axk2_topk_kernel against a stable CPU sort (score descending, index
// ascending): the same K indices for random rows, rows with ties and
// prefill rows whose keys past the position are -inf; then its latency for
// one decode row and a prefill slice.
#include "cuda/axk2_topk.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static constexpr uint32_t K = 2048;

enum class Fill { Random, Ties, Causal };

static bool check(const char *what, cudaError_t err) {
    if (err != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err)); }
    return err == cudaSuccess;
}

static std::vector<float> make(uint32_t rows, uint32_t n, Fill fill, std::mt19937 &rng) {
    std::vector<float> s((size_t)rows * n);
    std::normal_distribution<float> normal(0.0f, 3.0f);
    std::uniform_int_distribution<int> coarse(0, 40);
    for (uint32_t r = 0; r < rows; r++) {
        // Causal rows see keys [0, valid); the rest is -inf, as a prefill row.
        const uint32_t valid = fill == Fill::Causal ? std::min(n, K / 2 + r * (n / rows)) : n;
        for (uint32_t i = 0; i < n; i++) {
            float v = fill == Fill::Ties ? coarse(rng) * 0.25f : normal(rng);
            if (fill == Fill::Ties && i % 7 == 0) { v = 0.0f; }
            s[(size_t)r * n + i] = i < valid ? v : -INFINITY;
        }
    }
    return s;
}

static bool run(uint32_t rows, uint32_t n, Fill fill, std::mt19937 &rng) {
    const std::vector<float> s = make(rows, n, fill, rng);
    float *ds = nullptr;
    uint32_t *dsel = nullptr;
    if (!check("alloc", cudaMalloc(&ds, s.size() * 4)) || !check("alloc", cudaMalloc(&dsel, (size_t)rows * K * 4)) ||
        !check("copy", cudaMemcpy(ds, s.data(), s.size() * 4, cudaMemcpyHostToDevice))) { return false; }
    axk2_topk_kernel<K><<<rows, AXK2_TOPK_THREADS>>>(dsel, ds, n, n);
    std::vector<uint32_t> got((size_t)rows * K);
    const bool ok = check("kernel", cudaDeviceSynchronize()) &&
        check("copy", cudaMemcpy(got.data(), dsel, got.size() * 4, cudaMemcpyDeviceToHost));
    cudaFree(ds);
    cudaFree(dsel);
    if (!ok) { return false; }
    std::vector<uint32_t> idx(n);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t i = 0; i < n; i++) { idx[i] = i; }
        const float *row = s.data() + (size_t)r * n;
        std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return row[a] > row[b]; });
        std::vector<uint32_t> want(idx.begin(), idx.begin() + K), mine(got.begin() + (size_t)r * K,
            got.begin() + (size_t)(r + 1) * K);
        std::sort(want.begin(), want.end());
        std::sort(mine.begin(), mine.end());
        if (want != mine) {
            fprintf(stderr, "FAIL rows=%u n=%u fill=%d row=%u\n", rows, n, (int)fill, r);
            return false;
        }
    }
    return true;
}

static float time_ms(uint32_t rows, uint32_t n, std::mt19937 &rng) {
    const std::vector<float> s = make(rows, n, Fill::Random, rng);
    float *ds = nullptr;
    uint32_t *dsel = nullptr;
    cudaMalloc(&ds, s.size() * 4);
    cudaMalloc(&dsel, (size_t)rows * K * 4);
    cudaMemcpy(ds, s.data(), s.size() * 4, cudaMemcpyHostToDevice);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    axk2_topk_kernel<K><<<rows, AXK2_TOPK_THREADS>>>(dsel, ds, n, n);
    const int reps = 20;
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) { axk2_topk_kernel<K><<<rows, AXK2_TOPK_THREADS>>>(dsel, ds, n, n); }
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    cudaFree(ds);
    cudaFree(dsel);
    return ms / reps;
}

int main() {
    std::mt19937 rng(7);
    bool ok = true;
    for (uint32_t n : {2049u, 4096u, 8192u, 8193u, 20000u, 65536u, 262144u}) {
        for (Fill fill : {Fill::Random, Fill::Ties, Fill::Causal}) { ok = ok && run(n > 65536 ? 2 : 6, n, fill, rng); }
    }
    if (!ok) { return 1; }
    printf("axk2 topk: exact sets OK\n");
    for (uint32_t n : {4096u, 8192u, 65536u, 262144u}) { printf("  1 row x %6u keys: %7.1f us\n", n, 1000.0f * time_ms(1, n, rng)); }
    printf("  2048 rows x 8192 keys: %6.2f ms\n", time_ms(2048, 8192, rng));
    printf("  512 rows x 65536 keys: %6.2f ms\n", time_ms(512, 65536, rng));
    return 0;
}
