// glmvq_fa_kernel vs an FP64 softmax over the same selected latent rows.
// The bound is twice the BF16 output rounding (~0.0018 relative RMS).
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "../cuda/glm_vq_flash.cuh"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)
static std::mt19937 rng(3);
template <typename T> static T *up(const std::vector<T> &v) { T *p; CK(cudaMalloc(&p, v.size() * sizeof(T) + 64)); CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; }
template <typename T> static std::vector<T> down(const T *p, size_t n) { std::vector<T> v(n); CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
int main() {
    const uint32_t H = 64, D = 512, ctx = 8192, stride = 2051;
    const float scale = 1.0f / 16.0f;
    int failures = 0;
    std::normal_distribution<float> nd(0, 1);
    std::vector<__nv_bfloat16> cache((size_t)ctx * D);
    for (auto &v : cache) { v = __float2bfloat16_rn(nd(rng) * 0.5f); }
    __nv_bfloat16 *dcache = up(cache);
    CK(cudaFuncSetAttribute(glmvq_fa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_FA_SMEM));
    struct Case { uint32_t rows, first; bool sparse; uint32_t splits; };
    for (Case c : {Case{4, 100, false, 1}, Case{3, 2500, true, 1}, Case{1, 7000, true, 16}, Case{1, 1500, false, 8}, Case{2, 31, false, 1}}) {
        std::vector<__nv_bfloat16> q((size_t)c.rows * H * D);
        for (auto &v : q) { v = __float2bfloat16_rn(nd(rng) * 2.0f); }
        std::vector<uint32_t> pos(c.rows), sel((size_t)c.rows * stride, GLMVQ_FA_NONE);
        for (uint32_t r = 0; r < c.rows; r++) {
            pos[r] = c.first + r;
            if (c.sparse) {  // 512 random complete pools plus the tail, like pool_expand
                const uint32_t visible = pos[r] + 1, pools = visible / 4;
                std::vector<uint32_t> ids(pools); for (uint32_t i = 0; i < pools; i++) { ids[i] = i; }
                std::shuffle(ids.begin(), ids.end(), rng);
                for (uint32_t i = 0; i < std::min(pools, 512u); i++) { for (uint32_t k = 0; k < 4; k++) { sel[(size_t)r * stride + i * 4 + k] = ids[i] * 4 + k; } }
                for (uint32_t k = 0; k < visible % 4; k++) { sel[(size_t)r * stride + 2048 + k] = visible - visible % 4 + k; }
            }
        }
        __nv_bfloat16 *dq = up(q), *dlow; uint32_t *dpos = up(pos), *dsel = up(sel);
        CK(cudaMalloc(&dlow, q.size() * 2));
        float *part = nullptr, *stats = nullptr;
        const uint32_t slots = c.sparse ? stride : 0;
        uint32_t maxslots = 0; for (uint32_t r = 0; r < c.rows; r++) { maxslots = std::max(maxslots, c.sparse ? stride : pos[r] + 1); }
        const uint32_t tiles = (maxslots + 31) / 32, per = (tiles + c.splits - 1) / c.splits;
        if (c.splits > 1) {
            CK(cudaMalloc(&part, (size_t)c.rows * c.splits * H * D * 4)); CK(cudaMalloc(&stats, (size_t)c.rows * c.splits * H * 8));
        }
        glmvq_fa_kernel<<<dim3(c.rows, c.splits), GLMVQ_FA_THREADS, GLMVQ_FA_SMEM>>>(dlow, part, stats, dq, dcache, dpos,
            c.sparse ? dsel : nullptr, slots, c.splits > 1 ? per : 1u << 30, scale);
        if (c.splits > 1) { glmvq_fa_combine<<<dim3(c.rows, H), D / 2>>>(dlow, part, stats, c.splits); }
        CK(cudaDeviceSynchronize());
        auto got = down(dlow, q.size());
        double num = 0, den = 0, worst = 0;
        for (uint32_t r = 0; r < c.rows; r++) {
            std::vector<uint32_t> keys;
            if (c.sparse) { for (uint32_t i = 0; i < stride; i++) { if (sel[(size_t)r * stride + i] != GLMVQ_FA_NONE) { keys.push_back(sel[(size_t)r * stride + i]); } } }
            else { for (uint32_t k = 0; k <= pos[r]; k++) { keys.push_back(k); } }
            for (uint32_t h = 0; h < H; h += 7) {
                std::vector<double> sc(keys.size()); double mx = -1e300;
                for (size_t i = 0; i < keys.size(); i++) {
                    double d = 0; for (uint32_t x = 0; x < D; x++) { d += (double)__bfloat162float(q[((size_t)r * H + h) * D + x]) * __bfloat162float(cache[(size_t)keys[i] * D + x]); }
                    sc[i] = d * scale; mx = std::max(mx, sc[i]);
                }
                double z = 0; for (auto &v : sc) { v = exp(v - mx); z += v; }
                for (uint32_t x = 0; x < D; x++) {
                    double ref = 0; for (size_t i = 0; i < keys.size(); i++) { ref += sc[i] / z * __bfloat162float(cache[(size_t)keys[i] * D + x]); }
                    const double g = __bfloat162float(got[((size_t)r * H + h) * D + x]);
                    num += (g - ref) * (g - ref); den += ref * ref; worst = std::max(worst, fabs(g - ref));
                }
            }
        }
        printf("rows=%u first=%u sparse=%d splits=%u keys/query~%u: relRMS %.3g max|d| %.3g\n", c.rows, c.first, c.sparse, c.splits,
            c.sparse ? 2048u : c.first, sqrt(num / den), worst);
        failures += !(sqrt(num / den) < 0.004);
        cudaFree(dq); cudaFree(dlow); cudaFree(dpos); cudaFree(dsel); if (part) { cudaFree(part); cudaFree(stats); }
    }
    {   // decode absorption GEMVs vs FP64 on the same BF16-rounded inputs
        const uint32_t A = GLMVQ_ABS_DIM;
        std::vector<__nv_bfloat16> w((size_t)H * 2 * A * D), low((size_t)H * D);
        std::vector<float> q((size_t)H * A);
        for (auto &v : w) { v = __float2bfloat16_rn(nd(rng) * 0.05f); }
        for (auto &v : low) { v = __float2bfloat16_rn(nd(rng)); }
        for (auto &v : q) { v = nd(rng); }
        __nv_bfloat16 *dw = up(w), *dlow = up(low), *dql; float *dqv = up(q), *dout;
        CK(cudaMalloc(&dql, (size_t)H * D * 2)); CK(cudaMalloc(&dout, (size_t)H * A * 4));
        glmvq_absorb_k<<<H, D / 2>>>(dql, dqv, dw, 1u);
        glmvq_absorb_v<<<H * A / 8, 256>>>(dout, dlow, dw, 1u);
        CK(cudaDeviceSynchronize());
        auto ql = down(dql, (size_t)H * D); auto out = down(dout, (size_t)H * A);
        double nk = 0, dk = 0, nv = 0, dv = 0;
        for (uint32_t h = 0; h < H; h++) {
            for (uint32_t l = 0; l < D; l++) {
                double ref = 0;
                for (uint32_t d = 0; d < A; d++) { ref += (double)__bfloat162float(__float2bfloat16_rn(q[(size_t)h * A + d])) * __bfloat162float(w[((size_t)h * 2 * A + d) * D + l]); }
                const double g = __bfloat162float(ql[(size_t)h * D + l]); nk += (g - ref) * (g - ref); dk += ref * ref;
            }
            for (uint32_t c = 0; c < A; c++) {
                double ref = 0;
                for (uint32_t l = 0; l < D; l++) { ref += (double)__bfloat162float(low[(size_t)h * D + l]) * __bfloat162float(w[((size_t)h * 2 * A + A + c) * D + l]); }
                const double g = out[(size_t)h * A + c]; nv += (g - ref) * (g - ref); dv += ref * ref;
            }
        }
        printf("absorb k relRMS %.3g, v relRMS %.3g\n", sqrt(nk / dk), sqrt(nv / dv));
        failures += !(sqrt(nk / dk) < 0.004) + !(sqrt(nv / dv) < 0.004);
        cudaFree(dw); cudaFree(dlow); cudaFree(dql); cudaFree(dqv); cudaFree(dout);
    }
    printf(failures ? "FAIL %d\n" : "GLMVQ_FLASH_PASS\n", failures);
    return failures != 0;
}
