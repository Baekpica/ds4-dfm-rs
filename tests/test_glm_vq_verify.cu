// MTP verify-width kernels vs their one-row decode counterparts.
// Q4_K: each token of a multi-token call must equal a one-token call bit for
// bit on staged shapes (same per-token FMA order), and stay within BF16
// rounding of it on scalar-baseline shapes. VQ experts: the expert-union
// kernels must reproduce the one-row decode kernels bit for bit per row.
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include "../cuda/glm_vq_linear.cuh"
#include "../cuda/glm_vq_decode.cuh"
#include <algorithm>
#define CK(...) do { cudaError_t e_ = (__VA_ARGS__); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)
static std::mt19937 rng(7);
template <typename T> static T *up(const std::vector<T> &v) { T *p; CK(cudaMalloc(&p, v.size() * sizeof(T))); CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; }
template <typename T> static std::vector<T> down(const T *p, size_t n) { std::vector<T> v(n); CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }

static int q4_case(uint32_t rows, uint32_t cols, uint32_t tokens, const glmvq_q4_config *config) {
    const uint32_t blocks = cols / GLMVQ_Q4_BLOCK;
    std::vector<uint8_t> w((size_t)rows * blocks * GLMVQ_Q4_BYTES);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto &b : w) { b = (uint8_t)byte(rng); }
    for (size_t b = 0; b < (size_t)rows * blocks; b++) {  // finite d / dmin
        const uint16_t d = __half_as_ushort(__float2half(0.01f)), m = __half_as_ushort(__float2half(0.002f));
        w[b * GLMVQ_Q4_BYTES] = d & 255; w[b * GLMVQ_Q4_BYTES + 1] = d >> 8;
        w[b * GLMVQ_Q4_BYTES + 2] = m & 255; w[b * GLMVQ_Q4_BYTES + 3] = m >> 8;
    }
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> x((size_t)tokens * cols);
    for (auto &v : x) { v = nd(rng); }
    uint8_t *dw = up(w); float *dx = up(x), *dy, *d1;
    CK(cudaMalloc(&dy, (size_t)tokens * rows * 4)); CK(cudaMalloc(&d1, (size_t)rows * 4));
    CK(glmvq_q4_decode(dw, dx, dy, rows, cols, tokens, 0, config));
    CK(cudaDeviceSynchronize());
    auto many = down(dy, (size_t)tokens * rows);
    const bool staged = (uint64_t)rows * blocks * GLMVQ_Q4_BYTES > (4ull << 20);
    uint32_t differ = 0; double worst = 0;
    for (uint32_t t = 0; t < tokens; t++) {
        CK(glmvq_q4_decode(dw, dx + (size_t)t * cols, d1, rows, cols, 1, 0, config));
        CK(cudaDeviceSynchronize());
        auto one = down(d1, rows);
        for (uint32_t r = 0; r < rows; r++) {
            const float a = many[(size_t)t * rows + r], b = one[r];
            differ += a != b;
            worst = fmax(worst, (double)fabsf(a - b) / fmaxf(1e-3f, fabsf(b)));
        }
    }
    const bool ok = staged ? differ == 0 : worst < 8e-3;
    printf("q4 %ux%u tokens=%u %s: %u differ, worst rel %.2g %s\n", rows, cols, tokens,
        staged ? "staged" : "scalar-one", differ, worst, ok ? "ok" : "FAIL");
    cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(d1);
    return !ok;
}

// Host twin of gvq_verify_route: union items in ascending expert order.
static void route(const std::vector<int> &sel, uint32_t rows, const std::vector<glmvq_matrix> &gate,
        const std::vector<glmvq_matrix> &upm, const std::vector<glmvq_matrix> &dn,
        std::vector<glmvq_verify_item> &items, glmvq_verify_plan &plan) {
    std::vector<int> experts(sel.begin(), sel.begin() + rows * 8);
    std::sort(experts.begin(), experts.end());
    experts.erase(std::unique(experts.begin(), experts.end()), experts.end());
    items.assign(experts.size(), glmvq_verify_item{});
    plan = {};
    for (size_t i = 0; i < experts.size(); i++) {
        const int e = experts[i];
        items[i].gate = gate[e]; items[i].up = upm[e]; items[i].down = dn[e];
        for (uint32_t t = 0; t < rows * 8; t++) { if (sel[t] == e) { items[i].pair[items[i].count++] = t; } }
    }
    for (uint32_t r = 0; r < rows; r++) {
        std::vector<uint32_t> k(8);
        for (uint32_t j = 0; j < 8; j++) { k[j] = j; }
        std::sort(k.begin(), k.end(), [&](uint32_t a, uint32_t b) { return sel[r * 8 + a] < sel[r * 8 + b]; });
        for (uint32_t j = 0; j < 8; j++) { plan.order[r * 8 + j] = k[j]; }
    }
    uint32_t used = GLMVQ_VER_GROUP;
    for (uint32_t i = 0; i < items.size(); i++) {
        if (used + items[i].count > GLMVQ_VER_GROUP) { plan.group[plan.groups++] = i; used = 0; }
        items[i].slot = used; used += items[i].count;
    }
    plan.group[plan.groups] = plan.items = (uint32_t)items.size();
}

static int vq_case(uint32_t blocks) {
    const uint32_t E = 12, H = GLMVQ_DEC_HIDDEN, F = GLMVQ_DEC_FFN;
    std::uniform_int_distribution<int> byte(0, 255);
    auto bytes = [&](size_t n) { std::vector<uint8_t> v(n); for (auto &b : v) { b = (uint8_t)byte(rng); } return up(v); };
    std::vector<uint8_t> book((size_t)8192 * 8);
    for (auto &b : book) { b = (uint8_t)byte(rng) & 0x77; }  // finite, modest E4M3
    uint8_t *dbook = up(book);
    std::vector<__half> gains(H);
    for (auto &g : gains) { g = __float2half(0.02f + 0.01f * (byte(rng) / 255.0f)); }
    __half *dgain = up(gains);
    std::vector<glmvq_matrix> gate(E), upm(E), dn(E);
    for (uint32_t e = 0; e < E; e++) {
        gate[e] = {bytes((size_t)F * H * 3 / 16), bytes((size_t)F * H / 64), dbook, dgain, F, H, GLMVQ_E4M3};
        upm[e] = {bytes((size_t)F * H * 3 / 16), bytes((size_t)F * H / 64), dbook, dgain, F, H, GLMVQ_E4M3};
        dn[e] = {bytes((size_t)H * F * 3 / 16), bytes((size_t)H * F / 64), dbook, dgain, H, F, GLMVQ_E4M3};
    }
    int failures = 0;
    std::normal_distribution<float> nd(0, 1);
    for (uint32_t rows = 2; rows <= GLMVQ_VER_ROWS; rows++) {
        std::vector<int> sel(rows * 8);
        for (uint32_t r = 0; r < rows; r++) {  // 8 distinct of 12 experts: heavy overlap
            std::vector<int> ids(E);
            for (uint32_t e = 0; e < E; e++) { ids[e] = (int)e; }
            std::shuffle(ids.begin(), ids.end(), rng);
            for (uint32_t k = 0; k < 8; k++) { sel[r * 8 + k] = ids[k]; }
        }
        std::vector<float> x((size_t)rows * H), w(rows * 8);
        for (auto &v : x) { v = nd(rng); }
        for (auto &v : w) { v = 0.1f + 0.2f * (byte(rng) / 255.0f); }
        float *dx = up(x), *dw = up(w), *ref, *got, *partial; __nv_bfloat16 *mid;
        CK(cudaMalloc(&ref, x.size() * 4)); CK(cudaMalloc(&got, x.size() * 4));
        CK(cudaMalloc(&partial, (size_t)GLMVQ_VER_PAIRS * H * 4)); CK(cudaMalloc(&mid, (size_t)GLMVQ_VER_PAIRS * F * 2));
        for (uint32_t r = 0; r < rows; r++) {
            std::vector<glmvq_decode_item> items(8);
            std::vector<uint32_t> k(8);
            for (uint32_t j = 0; j < 8; j++) { k[j] = j; }
            std::sort(k.begin(), k.end(), [&](uint32_t a, uint32_t b) { return sel[r * 8 + a] < sel[r * 8 + b]; });
            for (uint32_t j = 0; j < 8; j++) { const int e = sel[r * 8 + k[j]]; items[j] = {gate[e], upm[e], dn[e], k[j]}; }
            glmvq_decode_item *ditems = up(items);
            CK(glmvq_decode_launch<13, 32, GLMVQ_DEC_NORMAL>(ditems, dbook, dx + (size_t)r * H, mid, partial,
                ref + (size_t)r * H, dw + r * 8, 7.0f, blocks, 0));
            CK(cudaDeviceSynchronize());
            cudaFree(ditems);
        }
        std::vector<glmvq_verify_item> items; glmvq_verify_plan plan;
        route(sel, rows, gate, upm, dn, items, plan);
        glmvq_verify_item *ditems = up(items);
        glmvq_verify_plan *dplan = up(std::vector<glmvq_verify_plan>{plan});
        CK(glmvq_verify_launch<13, 32>(ditems, dplan, dbook, dx, rows, mid, partial, got, dw, 7.0f, blocks, 0));
        CK(cudaDeviceSynchronize());
        auto a = down(ref, x.size()), b = down(got, x.size());
        uint32_t differ = 0;
        for (size_t i = 0; i < a.size(); i++) { differ += a[i] != b[i]; }
        printf("vq union rows=%u experts=%zu groups=%u: %u of %zu differ %s\n", rows, items.size(), plan.groups,
            differ, a.size(), differ ? "FAIL" : "ok");
        failures += differ != 0;
        cudaFree(dx); cudaFree(dw); cudaFree(ref); cudaFree(got); cudaFree(partial); cudaFree(mid);
        cudaFree(ditems); cudaFree(dplan);
    }
    return failures;
}

int main() {
    uint32_t blocks = 0;
    CK(glmvq_decode_prepare<13, 32, GLMVQ_DEC_NORMAL>(&blocks));
    glmvq_q4_config config;
    CK(glmvq_q4_prepare(&config, GLMVQ_Q4_AUTO));
    int failures = 0;
    for (uint32_t tokens : {2u, 3u, 4u, 8u}) {
        failures += q4_case(4096, 4096, tokens, &config);   // KDA q/k/v
        failures += q4_case(4096, 8192, tokens, &config);   // KDA output
        failures += q4_case(512, 4096, tokens, &config);    // small: scalar one-token baseline
    }
    failures += vq_case(blocks);
    printf(failures ? "FAIL %d\n" : "GLMVQ_VERIFY_PASS\n", failures);
    return failures != 0;
}
