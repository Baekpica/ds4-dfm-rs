// Model-free prefill MoE reduce + residual: CPU sums and the separate path.
#include "cuda/minimax_rows.cuh"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

enum { H = DS4_MINIMAX_HIDDEN, U = DS4_MINIMAX_USED, E = DS4_MINIMAX_EXPERTS };
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "REDUCE FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)
#define CK(x) CHECK((x) == cudaSuccess)
static uint32_t state = 0x98f62a71;
static uint32_t rng() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
static float value(float x) { return x; }
static float value(__nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T> static T cast(float x) {
    if constexpr (std::is_same_v<T, float>) { return x; }
    else { return __float2bfloat16_rn(x); }
}
template <typename T> static T *device(const std::vector<T> &v) {
    T *p = nullptr; CK(cudaMalloc(&p, v.size() * sizeof(T)));
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <typename T> static std::vector<T> read(const T *p, size_t n) {
    std::vector<T> out(n); CK(cudaMemcpy(out.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return out;
}

// Retained mm_mma_reduce, before fusion. CPU arithmetic below is independent.
template <typename ST>
__global__ void reference_reduce(__nv_bfloat16 *out, const __nv_bfloat16 *down, const float *weights,
        const uint32_t *pairs, const uint32_t *order, const ST *shared, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint64_t token = i / H, col = i % H;
    float sum = 0.0f;
    for (uint32_t k = 0; k < U; k++) {
        const uint32_t row = order[token * U + k];
        sum = glmvq_add(sum, glmvq_mul(__bfloat162float(down[(uint64_t)row * H + col]), weights[pairs[row]]));
    }
    out[i] = __float2bfloat16_rn(glmvq_add(sum, glmvq_round(glmvq_value(shared[i]))));
}

template <typename CT, typename ST>
static void run(unsigned rows, bool next, bool prepared, bool timing) {
    const size_t n = (size_t)rows * H;
    std::vector<uint32_t> expert(rows * U), counts(E), pairs(rows * U), order(rows * U), cursor(E), filled(rows);
    std::vector<float> weights(rows * U), scales(rows), norm(n);
    for (unsigned r = 0; r < rows; r++) {
        std::array<unsigned, U> used = {E, E, E, E};
        for (unsigned k = 0; k < U; k++) {
            unsigned e;
            do { e = rng() % ((r % 4) ? 32 : E); } while (std::find(used.begin(), used.end(), e) != used.end());
            used[k] = e; expert[r * U + k] = e; counts[e]++; weights[r * U + k] = (1 + rng() % 1024) / 2048.f;
        }
    }
    for (unsigned e = 1; e < E; e++) { cursor[e] = cursor[e - 1] + counts[e - 1]; }
    for (unsigned p = 0; p < rows * U; p++) { pairs[cursor[expert[p]]++] = p; }
    for (unsigned p = 0; p < rows * U; p++) { order[(pairs[p] / U) * U + filled[pairs[p] / U]++] = p; }
    std::vector<__nv_bfloat16> down(n * U), branch(n), weight(H);
    std::vector<CT> cur(n);
    std::vector<ST> shared(n);
    std::vector<__half> half(n);
    for (auto &v : down) { v = __float2bfloat16_rn((int(rng() % 8193) - 4096) / 513.f); }
    for (auto &v : shared) { v = cast<ST>((int(rng() % 2049) - 1024) / 97.f); }
    for (auto &v : cur) { v = cast<CT>(value(__float2bfloat16_rn((int(rng() % 4097) - 2048) / 313.f))); }
    for (auto &v : weight) { v = __float2bfloat16_rn((int(rng() % 127) - 63) / 97.f); }
    auto *d = device(down), *b = device(branch), *w = device(weight);
    auto *s = device(shared);
    auto *original = device(cur), *a = device(cur), *f = device(cur);
    auto *rw = device(weights); auto *p = device(pairs), *o = device(order);
    auto *an = device(norm), *fn = device(norm), *as = device(scales), *fs = device(scales);
    auto *ah = device(half), *fh = device(half);
    CK(cudaMemset(an, 0xa5, n * sizeof(float))); CK(cudaMemset(fn, 0xa5, n * sizeof(float)));
    auto old = [&] {
        reference_reduce<<<(n + 255) / 256, 256>>>(b, d, rw, p, o, s, n);
        mm_residual_kernel<<<rows, MM_ROW_THREADS>>>(a, b, an, next ? w : nullptr,
            prepared ? ah : nullptr, prepared ? as : nullptr);
    };
    auto fused = [&] {
        mm_moe_residual_kernel<<<rows, MM_ROW_THREADS>>>(f, d, rw, p, o, s, fn, next ? w : nullptr,
            prepared ? fh : nullptr, prepared ? fs : nullptr);
    };
    old(); fused(); CK(cudaDeviceSynchronize());
    auto ac = read(a, n), fc = read(f, n);
    CHECK(!memcmp(ac.data(), fc.data(), n * sizeof(CT)));
    auto ab = read(b, n);
    for (size_t i = 0; i < n; i++) {
        const size_t row = i / H, col = i % H;
        float sum = 0;
        for (unsigned k = 0; k < U; k++) {
            const unsigned at = order[row * U + k];
            volatile float product = value(down[(size_t)at * H + col]) * weights[pairs[at]];
            sum += product;
        }
        const auto expected = __float2bfloat16_rn(sum + value(__float2bfloat16_rn(value(shared[i]))));
        CHECK(!memcmp(&expected, &ab[i], sizeof(expected)));
        const CT expected_cur = cast<CT>(value(__float2bfloat16_rn(value(cur[i]) + value(expected))));
        CHECK(!memcmp(&expected_cur, &fc[i], sizeof(CT)));
    }
    auto av = read(an, n), fv = read(fn, n);
    CHECK(!memcmp(av.data(), fv.data(), n * sizeof(float)));
    if (!next) {
        for (size_t i = 0; i < n * sizeof(float); i++) { CHECK(((const uint8_t *)fv.data())[i] == 0xa5); }
    }
    if (prepared) {
        auto ahv = read(ah, n), fhv = read(fh, n);
        auto asv = read(as, rows), fsv = read(fs, rows);
        CHECK(!memcmp(ahv.data(), fhv.data(), n * sizeof(__half)));
        CHECK(!memcmp(asv.data(), fsv.data(), rows * sizeof(float)));
    }
    if (timing) {
        cudaEvent_t start, end; CK(cudaEventCreate(&start)); CK(cudaEventCreate(&end));
        std::vector<float> old_ms, fused_ms;
        for (unsigned i = 0; i < 7; i++) {
            CK(cudaMemcpy(a, original, n * sizeof(CT), cudaMemcpyDeviceToDevice));
            CK(cudaEventRecord(start)); old(); CK(cudaEventRecord(end)); CK(cudaEventSynchronize(end));
            float ms; CK(cudaEventElapsedTime(&ms, start, end)); old_ms.push_back(ms);
            CK(cudaMemcpy(f, original, n * sizeof(CT), cudaMemcpyDeviceToDevice));
            CK(cudaEventRecord(start)); fused(); CK(cudaEventRecord(end)); CK(cudaEventSynchronize(end));
            CK(cudaEventElapsedTime(&ms, start, end)); fused_ms.push_back(ms);
        }
        std::sort(old_ms.begin(), old_ms.end()); std::sort(fused_ms.begin(), fused_ms.end());
        printf("REDUCE_TIME rows=%u old_ms=%.6f fused_ms=%.6f\n", rows, old_ms[3], fused_ms[3]);
        CK(cudaEventDestroy(end)); CK(cudaEventDestroy(start));
    }
    printf("REDUCE rows=%u cur_bf16=%d shared_bf16=%d next=%d q4p=%d PASS\n", rows,
        !std::is_same_v<CT, float>, !std::is_same_v<ST, float>, next, prepared);
    CK(cudaFree(fh)); CK(cudaFree(ah)); CK(cudaFree(fs)); CK(cudaFree(as)); CK(cudaFree(fn)); CK(cudaFree(an));
    CK(cudaFree(o)); CK(cudaFree(p)); CK(cudaFree(rw)); CK(cudaFree(f)); CK(cudaFree(a)); CK(cudaFree(original));
    CK(cudaFree(s)); CK(cudaFree(w)); CK(cudaFree(b)); CK(cudaFree(d));
}

int main(int argc, char **argv) {
    CHECK(argc <= 2);
    if (argc == 2) {
        const unsigned rows = (unsigned)strtoul(argv[1], nullptr, 10);
        CHECK(rows > DS4_MINIMAX_SPLIT_ROWS && rows <= 4096);
        run<__nv_bfloat16, __nv_bfloat16>(rows, true, true, true);
    } else {
        for (unsigned rows : {9u, 128u, 4096u}) {
            run<float, float>(rows, true, true, false);
            run<float, __nv_bfloat16>(rows, true, true, false);
            run<__nv_bfloat16, float>(rows, true, true, false);
            run<__nv_bfloat16, __nv_bfloat16>(rows, true, true, rows == 4096);
        }
        run<__nv_bfloat16, __nv_bfloat16>(9, false, false, false);
        run<__nv_bfloat16, __nv_bfloat16>(9, true, false, false);
    }
    puts("MINIMAX_REDUCE_PASS");
}
