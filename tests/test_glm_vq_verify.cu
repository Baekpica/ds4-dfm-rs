// MTP verify-width kernels vs their one-row decode counterparts.
// Q4_K: each token of a multi-token call must equal a one-token call bit for
// bit on staged shapes (same per-token FMA order), and stay within BF16
// rounding of it on scalar-baseline shapes. VQ experts, BF16 dots and DSA
// absorption must reproduce their one-row kernels bit for bit per row.
// Tensor-core Q4_K verify rows: exact products of BF16 inputs, so each token
// stays within one BF16 ulp of an FP64 dot of the same weights.
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include "../cuda/glm_vq_linear.cuh"
#include "../cuda/glm_vq_q4m.cuh"
#include "../cuda/glm_vq_decode.cuh"
#include "../cuda/glm_vq_gemv.cuh"
#include "../cuda/glm_vq_flash.cuh"
#include "../cuda/glm_vq_fp8.cuh"
#include "../cuda/glm_vq_index.cuh"
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

static float q4_host_value(const uint8_t *block, uint32_t i) {
    const uint8_t *scales = block + 4;
    const uint32_t pair = i / 64, sub = i / 32, j = i % 32;
    const uint32_t q = sub & 1 ? block[16 + 32 * pair + j] >> 4 : block[16 + 32 * pair + j] & 15;
    const uint32_t sc = sub < 4 ? scales[sub] & 63 : (scales[sub + 4] & 15) | ((scales[sub - 4] >> 6) << 4);
    const uint32_t m = sub < 4 ? scales[sub + 4] & 63 : (scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4);
    const float d = __half2float(*(const __half *)block), dmin = __half2float(*(const __half *)(block + 2));
    return d * sc * q - dmin * m;
}

static int q4m_case(uint32_t rows, uint32_t cols, uint32_t tokens) {
    const uint32_t blocks = cols / GLMVQ_Q4_BLOCK;
    std::vector<uint8_t> w((size_t)rows * blocks * GLMVQ_Q4_BYTES);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto &b : w) { b = (uint8_t)byte(rng); }
    for (size_t b = 0; b < (size_t)rows * blocks; b++) {
        *(__half *)&w[b * GLMVQ_Q4_BYTES] = __float2half(scale(rng));
        *(__half *)&w[b * GLMVQ_Q4_BYTES + 2] = __float2half(scale(rng) * 0.2f);
    }
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> x((size_t)tokens * cols);
    for (auto &v : x) { v = __bfloat162float(__float2bfloat16_rn(nd(rng))); }   // graph inputs are BF16-rounded
    uint8_t *dw = up(w); float *dx = up(x), *dy;
    CK(cudaMalloc(&dy, (size_t)tokens * rows * 4));
    CK(glmvq_q4m_decode(dw, dx, dy, rows, cols, tokens, 0, GLMVQ_LAUNCH_PLAIN));
    CK(cudaDeviceSynchronize());
    auto got = down(dy, (size_t)tokens * rows);
    uint32_t differ = 0, beyond = 0;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t t = 0; t < tokens; t++) {
            double ref = 0;
            for (uint32_t k = 0; k < cols; k++) {
                const uint8_t *block = &w[((size_t)r * blocks + k / GLMVQ_Q4_BLOCK) * GLMVQ_Q4_BYTES];
                ref += (double)q4_host_value(block, k % GLMVQ_Q4_BLOCK) * x[(size_t)t * cols + k];
            }
            const float want = __bfloat162float(__float2bfloat16_rn((float)ref));
            const float have = got[(size_t)t * rows + r];
            const float ulp = fmaxf(fabsf(want), 1e-30f) * 0x1p-7f;
            differ += have != want;
            beyond += fabsf(have - want) > ulp;
        }
    }
    const bool ok = !beyond && differ * 50 <= rows * tokens;
    printf("q4 mma %ux%u tokens=%u: %u differ by one ulp, %u beyond %s\n", rows, cols, tokens, differ, beyond,
        ok ? "ok" : "FAIL");
    cudaFree(dw); cudaFree(dx); cudaFree(dy);
    return !ok;
}

// Fused gate/up/SwiGLU must equal the two plain MMA projections followed
// by the SwiGLU kernel bit for bit (each row's dot is unchanged).
static int q4m_swiglu_case(uint32_t rows, uint32_t cols, uint32_t tokens) {
    const uint32_t blocks = cols / GLMVQ_Q4_BLOCK;
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    std::vector<uint8_t> w[2];
    for (auto &m : w) {
        m.resize((size_t)rows * blocks * GLMVQ_Q4_BYTES);
        for (auto &b : m) { b = (uint8_t)byte(rng); }
        for (size_t b = 0; b < (size_t)rows * blocks; b++) {
            *(__half *)&m[b * GLMVQ_Q4_BYTES] = __float2half(scale(rng) * 40.0f);
            *(__half *)&m[b * GLMVQ_Q4_BYTES + 2] = __float2half(scale(rng) * 8.0f);
        }
    }
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> x((size_t)tokens * cols);
    for (auto &v : x) { v = __bfloat162float(__float2bfloat16_rn(nd(rng))); }
    uint8_t *dg = up(w[0]), *du = up(w[1]); float *dx = up(x), *gate, *upv, *ref, *got;
    const size_t n = (size_t)tokens * rows;
    CK(cudaMalloc(&gate, n * 4)); CK(cudaMalloc(&upv, n * 4)); CK(cudaMalloc(&ref, n * 4)); CK(cudaMalloc(&got, n * 4));
    CK(glmvq_q4m_decode(dg, dx, gate, rows, cols, tokens, 0, GLMVQ_LAUNCH_PLAIN));
    CK(glmvq_q4m_decode(du, dx, upv, rows, cols, tokens, 0, GLMVQ_LAUNCH_PLAIN));
    glmvq_swiglu<<<(uint32_t)((n + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS>>>(gate, upv, ref, n, 10.0f);
    CK(glmvq_q4m_swiglu(dg, du, dx, got, rows, cols, tokens, 10.0f, 0, GLMVQ_LAUNCH_PLAIN));
    CK(cudaDeviceSynchronize());
    auto a = down(ref, n), b = down(got, n);
    uint32_t differ = 0, saturated = 0;
    for (size_t i = 0; i < n; i++) { differ += a[i] != b[i]; saturated += fabsf(a[i]) > 50.0f; }
    printf("q4 mma swiglu %ux%u tokens=%u: %u of %zu differ (%u large) %s\n", rows, cols, tokens, differ, n,
        saturated, differ ? "FAIL" : "ok");
    cudaFree(dg); cudaFree(du); cudaFree(dx); cudaFree(gate); cudaFree(upv); cudaFree(ref); cudaFree(got);
    return differ != 0;
}

static int bf16_case(uint32_t rows, uint32_t cols, uint32_t tokens, glmvq_dot_math math) {
    std::normal_distribution<float> nd(0, 1);
    std::vector<__nv_bfloat16> w((size_t)rows * cols);
    for (auto &v : w) { v = __float2bfloat16_rn(nd(rng) * 0.05f); }
    std::vector<float> x((size_t)tokens * cols);
    for (auto &v : x) { v = nd(rng); }
    __nv_bfloat16 *dw = up(w); float *dx = up(x), *dy, *d1;
    CK(cudaMalloc(&dy, (size_t)tokens * rows * 4)); CK(cudaMalloc(&d1, (size_t)rows * 4));
    CK(glmvq_bf16_decode(dw, dx, dy, rows, cols, math, 0, tokens));
    CK(cudaDeviceSynchronize());
    auto many = down(dy, (size_t)tokens * rows);
    uint32_t differ = 0;
    for (uint32_t t = 0; t < tokens; t++) {
        CK(glmvq_bf16_decode(dw, dx + (size_t)t * cols, d1, rows, cols, math, 0, 1));
        CK(cudaDeviceSynchronize());
        auto one = down(d1, rows);
        for (uint32_t r = 0; r < rows; r++) { differ += many[(size_t)t * rows + r] != one[r]; }
    }
    printf("bf16 %ux%u tokens=%u %s: %u differ %s\n", rows, cols, tokens,
        math == GLMVQ_DOT_F32 ? "f32" : "bf16", differ, differ ? "FAIL" : "ok");
    cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(d1);
    return differ != 0;
}

static int fp8_case(uint32_t rows, uint32_t cols, uint32_t tokens) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> nd(0, 1);
    std::vector<uint8_t> w((size_t)rows * cols);
    for (auto &b : w) { b = (uint8_t)byte(rng) & 0x77; }
    std::vector<float> scale((size_t)rows / GLMVQ_SCALE_BLOCK * (cols / GLMVQ_SCALE_BLOCK)), x((size_t)tokens * cols);
    for (auto &v : scale) { v = 0.01f + 0.01f * (byte(rng) / 255.0f); }
    for (auto &v : x) { v = nd(rng); }
    uint8_t *dw = up(w); float *ds = up(scale), *dx = up(x), *dy, *d1;
    CK(cudaMalloc(&dy, (size_t)tokens * rows * 4)); CK(cudaMalloc(&d1, (size_t)rows * 4));
    const uint32_t split = glmvq_fp8_split(rows, cols);
    CK(glmvq_fp8_decode(dw, ds, dx, dy, rows, cols, split, GLMVQ_FP8_GLOBAL, 0, tokens));
    CK(cudaDeviceSynchronize());
    auto many = down(dy, (size_t)tokens * rows);
    uint32_t differ = 0;
    for (uint32_t t = 0; t < tokens; t++) {
        CK(glmvq_fp8_decode(dw, ds, dx + (size_t)t * cols, d1, rows, cols, split, GLMVQ_FP8_GLOBAL, 0, 1));
        CK(cudaDeviceSynchronize());
        auto one = down(d1, rows);
        for (uint32_t r = 0; r < rows; r++) { differ += many[(size_t)t * rows + r] != one[r]; }
    }
    printf("fp8 %ux%u tokens=%u: %u differ %s\n", rows, cols, tokens, differ, differ ? "FAIL" : "ok");
    cudaFree(dw); cudaFree(ds); cudaFree(dx); cudaFree(dy); cudaFree(d1);
    return differ != 0;
}

// Prefill mHC mix (inv * Σ W x, one pass) vs the decode kernel (Σ W (x * inv)).
static int mix_case(uint32_t rows) {
    const uint32_t width = 16384;
    std::normal_distribution<float> nd(0, 1);
    std::vector<__nv_bfloat16> w((size_t)GLMVQ_HC_MIX * width);
    std::vector<float> x((size_t)rows * width);
    for (auto &v : w) { v = __float2bfloat16_rn(nd(rng) * 0.02f); }
    for (auto &v : x) { v = nd(rng) * 3.0f; }
    __nv_bfloat16 *dw = up(w); float *dx = up(x), *a, *b;
    CK(cudaMalloc(&a, (size_t)rows * GLMVQ_HC_MIX * 4)); CK(cudaMalloc(&b, (size_t)rows * GLMVQ_HC_MIX * 4));
    glmvq_hc_mix_rows<<<(rows + GLMVQ_MIX_WARPS - 1) / GLMVQ_MIX_WARPS, GLMVQ_MIX_WARPS * 32>>>(dx, dw, a, rows, width, 1e-6f);
    glmvq_hc_mix<<<dim3(GLMVQ_HC_MIX, rows), GLMVQ_THREADS>>>(dx, dw, b, width, 1e-6f);
    CK(cudaDeviceSynchronize());
    auto ga = down(a, (size_t)rows * GLMVQ_HC_MIX), gb = down(b, (size_t)rows * GLMVQ_HC_MIX);
    double num = 0, den = 0;
    for (size_t i = 0; i < ga.size(); i++) { num += (ga[i] - gb[i]) * (ga[i] - gb[i]); den += (double)gb[i] * gb[i]; }
    const double rel = sqrt(num / den);
    printf("mHC mix rows=%u: relRMS %.2g %s\n", rows, rel, rel < 1e-5 ? "ok" : "FAIL");
    cudaFree(dw); cudaFree(dx); cudaFree(a); cudaFree(b);
    return !(rel < 1e-5);
}

// Fused prefill index scores vs cuBLAS dots + glmvq_index_reduce.
static int index_case(uint32_t rows, uint32_t first) {
    const uint32_t H = GLMVQ_INDEX_HEADS, D = DS4_GLM53_POOL_DIM, pools = (first + rows) / DS4_GLM53_POOL_SIZE;
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> q((size_t)rows * H * D), w((size_t)rows * H);
    std::vector<__nv_bfloat16> k((size_t)pools * D);
    for (auto &v : q) { v = __bfloat162float(__float2bfloat16_rn(nd(rng))); }
    for (auto &v : w) { v = __bfloat162float(__float2bfloat16_rn(nd(rng))); }
    for (auto &v : k) { v = __float2bfloat16_rn(nd(rng)); }
    float *dq = up(q), *dw = up(w), *a, *b; __nv_bfloat16 *dk = up(k); void *scratch;
    const uint64_t bytes = glmvq_index_bytes(rows, pools);
    CK(cudaMalloc(&a, (size_t)rows * pools * 4)); CK(cudaMalloc(&b, (size_t)rows * pools * 4)); CK(cudaMalloc(&scratch, bytes));
    cublasHandle_t handle; cublasCreate(&handle);
    CK(cudaFuncSetAttribute(glmvq_index_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_IDX_SMEM));
    const dim3 grid((pools + GLMVQ_IDX_POOLS - 1) / GLMVQ_IDX_POOLS, (rows + GLMVQ_IDX_ROWS - 1) / GLMVQ_IDX_ROWS);
    glmvq_index_fused<<<grid, GLMVQ_IDX_THREADS, GLMVQ_IDX_SMEM>>>(a, dq, dw, dk, pools, rows, first);
    if (glmvq_index_score(handle, b, dq, dw, dk, pools, rows, first, scratch, bytes, 0) != CUBLAS_STATUS_SUCCESS) { return 1; }
    CK(cudaDeviceSynchronize());
    auto ga = down(a, (size_t)rows * pools), gb = down(b, (size_t)rows * pools);
    double num = 0, den = 0; uint32_t masks = 0;
    for (size_t i = 0; i < ga.size(); i++) {
        if ((ga[i] == -FLT_MAX) != (gb[i] == -FLT_MAX)) { masks++; continue; }
        if (gb[i] == -FLT_MAX) { continue; }
        num += (ga[i] - gb[i]) * (ga[i] - gb[i]); den += (double)gb[i] * gb[i];
    }
    const double rel = sqrt(num / den);
    printf("index fused rows=%u first=%u pools=%u: relRMS %.2g mask mismatches %u %s\n", rows, first, pools, rel, masks,
        rel < 1e-5 && !masks ? "ok" : "FAIL");
    cublasDestroy(handle); cudaFree(dq); cudaFree(dw); cudaFree(dk); cudaFree(a); cudaFree(b); cudaFree(scratch);
    return !(rel < 1e-5 && !masks);
}

// Fused prefill expand + mix vs glmvq_hc_expand then glmvq_hc_mix_rows.
static int expand_mix_case(uint32_t rows) {
    const uint32_t width = 4096, flat = GLMVQ_HC * width;
    std::normal_distribution<float> nd(0, 1);
    std::vector<__nv_bfloat16> w((size_t)GLMVQ_HC_MIX * flat);
    std::vector<float> res((size_t)rows * flat), br((size_t)rows * width), split((size_t)rows * GLMVQ_HC_MIX);
    for (auto &v : w) { v = __float2bfloat16_rn(nd(rng) * 0.02f); }
    for (auto &v : res) { v = __bfloat162float(__float2bfloat16_rn(nd(rng))); }
    for (auto &v : br) { v = __bfloat162float(__float2bfloat16_rn(nd(rng))); }
    for (auto &v : split) { v = nd(rng) * 0.5f; }
    __nv_bfloat16 *dw = up(w); float *dres = up(res), *dbr = up(br), *dsp = up(split), *o1, *o2, *m1, *m2;
    CK(cudaMalloc(&o1, res.size() * 4)); CK(cudaMalloc(&o2, res.size() * 4));
    CK(cudaMalloc(&m1, (size_t)rows * GLMVQ_HC_MIX * 4)); CK(cudaMalloc(&m2, (size_t)rows * GLMVQ_HC_MIX * 4));
    glmvq_hc_expand<<<(res.size() + 255) / 256, 256>>>(dbr, dres, dsp, o1, width, res.size());
    glmvq_hc_mix_rows<<<(rows + GLMVQ_MIX_WARPS - 1) / GLMVQ_MIX_WARPS, GLMVQ_MIX_WARPS * 32>>>(o1, dw, m1, rows, flat, 1e-6f);
    CK(cudaFuncSetAttribute(glmvq_hc_expand_mix<float, float>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)glmvq_xmix_smem<float>()));
    glmvq_hc_expand_mix<<<(rows + GLMVQ_MIX_WARPS - 1) / GLMVQ_MIX_WARPS, GLMVQ_MIX_WARPS * 32, glmvq_xmix_smem<float>()>>>(
        dbr, dres, dsp, o2, dw, m2, rows, width, 1e-6f);
    CK(cudaDeviceSynchronize());
    auto a = down(o1, res.size()), b = down(o2, res.size());
    auto ma = down(m1, (size_t)rows * GLMVQ_HC_MIX), mb = down(m2, (size_t)rows * GLMVQ_HC_MIX);
    uint32_t differ = 0;
    for (size_t i = 0; i < a.size(); i++) { differ += a[i] != b[i]; }
    double num = 0, den = 0;
    for (size_t i = 0; i < ma.size(); i++) { num += (ma[i] - mb[i]) * (ma[i] - mb[i]); den += (double)ma[i] * ma[i]; }
    const double rel = sqrt(num / den);
    printf("mHC expand+mix rows=%u: expand %u differ, mix relRMS %.2g %s\n", rows, differ, rel, !differ && rel < 1e-5 ? "ok" : "FAIL");
    cudaFree(dw); cudaFree(dres); cudaFree(dbr); cudaFree(dsp); cudaFree(o1); cudaFree(o2); cudaFree(m1); cudaFree(m2);
    return !(!differ && rel < 1e-5);
}

// GLM-width mHC split with preloaded rows vs the generic split kernel.
template <typename R>
static int split_row_case(uint32_t rows) {
    const uint32_t width = GLMVQ_HC_WIDTH;
    std::normal_distribution<float> nd(0, 1);
    std::vector<R> res((size_t)rows * GLMVQ_HC * width);
    std::vector<float> mix((size_t)rows * GLMVQ_HC_MIX), scale = {0.5f, 0.7f, 0.9f}, base(GLMVQ_HC_MIX);
    std::vector<__nv_bfloat16> nw(width);
    for (auto &v : res) { v = (R)__bfloat162float(__float2bfloat16_rn(nd(rng))); }
    for (auto &v : mix) { v = nd(rng); }
    for (auto &v : base) { v = 0.1f * nd(rng); }
    for (auto &v : nw) { v = __float2bfloat16_rn(1.0f + 0.1f * nd(rng)); }
    R *dres = up(res); float *dmix = up(mix), *dscale = up(scale), *dbase = up(base); __nv_bfloat16 *dnw = up(nw);
    float *out[2], *split[2], *norm[2];
    for (int v = 0; v < 2; v++) {
        CK(cudaMalloc(&out[v], (size_t)rows * width * 4)); CK(cudaMalloc(&split[v], (size_t)rows * GLMVQ_HC_MIX * 4));
        CK(cudaMalloc(&norm[v], (size_t)rows * width * 4));
    }
    glmvq_hc_split<R><<<rows, GLMVQ_THREADS>>>(dmix, dres, dscale, dbase, out[0], split[0], width, 20u, 1e-6f, dnw, norm[0], 1e-5f);
    glmvq_hc_split_row<R><<<rows, GLMVQ_THREADS>>>(dmix, dres, dscale, dbase, out[1], split[1], 20u, 1e-6f, dnw, norm[1], 1e-5f);
    CK(cudaDeviceSynchronize());
    size_t differ = 0;
    for (auto *pair : {out, split, norm}) {
        const size_t n = pair == split ? (size_t)rows * GLMVQ_HC_MIX : (size_t)rows * width;
        auto a = down(pair[0], n), b = down(pair[1], n);
        for (size_t i = 0; i < n; i++) { differ += a[i] != b[i]; }
    }
    printf("mHC split row rows=%u %s: %zu differ %s\n", rows, sizeof(R) == 2 ? "bf16" : "f32", differ, differ ? "FAIL" : "ok");
    cudaFree(dres); cudaFree(dmix); cudaFree(dscale); cudaFree(dbase); cudaFree(dnw);
    for (int v = 0; v < 2; v++) { cudaFree(out[v]); cudaFree(split[v]); cudaFree(norm[v]); }
    return differ != 0;
}

static int absorb_case(uint32_t rows) {
    const uint32_t H = GLMVQ_FA_HEADS, A = GLMVQ_ABS_DIM, D = GLMVQ_FA_LATENT;
    std::normal_distribution<float> nd(0, 1);
    std::vector<__nv_bfloat16> w((size_t)H * 2 * A * D), low((size_t)rows * H * D);
    std::vector<float> q((size_t)rows * H * A);
    for (auto &v : w) { v = __float2bfloat16_rn(nd(rng) * 0.05f); }
    for (auto &v : low) { v = __float2bfloat16_rn(nd(rng)); }
    for (auto &v : q) { v = nd(rng); }
    __nv_bfloat16 *dw = up(w), *dlow = up(low), *dql; float *dq = up(q), *dout;
    CK(cudaMalloc(&dql, low.size() * 2)); CK(cudaMalloc(&dout, q.size() * 4));
    glmvq_absorb_k<<<H, D / 2>>>(dql, dq, dw, rows);
    glmvq_absorb_v<<<H * A / 8, 256>>>(dout, dlow, dw, rows);
    CK(cudaDeviceSynchronize());
    auto ql = down(dql, low.size()); auto out = down(dout, q.size());
    uint32_t differ = 0;
    for (uint32_t r = 0; r < rows; r++) {
        glmvq_absorb_k<<<H, D / 2>>>(dql, dq + (size_t)r * H * A, dw, 1u);
        glmvq_absorb_v<<<H * A / 8, 256>>>(dout, dlow + (size_t)r * H * D, dw, 1u);
        CK(cudaDeviceSynchronize());
        auto ql1 = down(dql, (size_t)H * D); auto out1 = down(dout, (size_t)H * A);
        for (size_t i = 0; i < ql1.size(); i++) {
            differ += __bfloat16_as_ushort(ql[(size_t)r * H * D + i]) != __bfloat16_as_ushort(ql1[i]);
        }
        for (size_t i = 0; i < out1.size(); i++) { differ += out[(size_t)r * H * A + i] != out1[i]; }
    }
    printf("absorb rows=%u: %u differ %s\n", rows, differ, differ ? "FAIL" : "ok");
    cudaFree(dw); cudaFree(dlow); cudaFree(dql); cudaFree(dq); cudaFree(dout);
    return differ != 0;
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

template <glmvq_decode_gain GAIN>
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
            CK(glmvq_decode_launch<13, 32, GLMVQ_DEC_NORMAL, GAIN>(ditems, dbook, dx + (size_t)r * H, mid, partial,
                ref + (size_t)r * H, dw + r * 8, 7.0f, blocks, 0));
            CK(cudaDeviceSynchronize());
            cudaFree(ditems);
        }
        std::vector<glmvq_verify_item> items; glmvq_verify_plan plan;
        route(sel, rows, gate, upm, dn, items, plan);
        glmvq_verify_item *ditems = up(items);
        glmvq_verify_plan *dplan = up(std::vector<glmvq_verify_plan>{plan});
        CK(glmvq_verify_launch<13, 32, GAIN>(ditems, dplan, dbook, dx, rows, mid, partial, got, dw, 7.0f, blocks, 0));
        CK(cudaDeviceSynchronize());
        auto a = down(ref, x.size()), b = down(got, x.size());
        uint32_t differ = 0;
        for (size_t i = 0; i < a.size(); i++) { differ += a[i] != b[i]; }
        printf("vq union %s rows=%u experts=%zu groups=%u: %u of %zu differ %s\n",
            GAIN == GLMVQ_GAIN_DOT ? "gain-dot" : "gain-weight", rows, items.size(), plan.groups,
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
    CK(glmvq_decode_prepare<13, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_DOT>(&blocks));
    glmvq_q4_config config;
    CK(glmvq_q4_prepare(&config, GLMVQ_Q4_AUTO));
    int failures = 0;
    for (uint32_t rows : {1u, 3u, 64u}) {
        failures += split_row_case<__nv_bfloat16>(rows);
        failures += split_row_case<float>(rows);
    }
    CK(glmvq_q4m_prepare());
    for (uint32_t tokens : {2u, 3u, 8u}) {
        failures += q4m_case(1024, 4096, tokens);   // KDA q/k/v rows
        failures += q4m_case(1000, 1536, tokens);   // q_b: partial last stage and row tile
    }
    for (uint32_t tokens : {1u, 3u, 8u}) {
        failures += q4m_swiglu_case(2048, 4096, tokens);   // shared expert
        failures += q4m_swiglu_case(1000, 1536, tokens);   // row and stage tails
    }
    for (uint32_t tokens : {2u, 3u, 4u, 8u}) {
        failures += q4_case(4096, 4096, tokens, &config);   // KDA q/k/v
        failures += q4_case(4096, 8192, tokens, &config);   // KDA output
        failures += q4_case(512, 4096, tokens, &config);    // small: scalar one-token baseline
    }
    for (uint32_t tokens : {2u, 4u}) {
        failures += bf16_case(4096, 4096, tokens, GLMVQ_DOT_BF16);   // KDA gate projections
        failures += bf16_case(64, 4096, tokens, GLMVQ_DOT_BF16);     // split rows
        failures += bf16_case(288, 4096, tokens, GLMVQ_DOT_F32);     // router
    }
    for (uint32_t rows = 2; rows <= GLMVQ_ABS_ROWS; rows++) { failures += absorb_case(rows); }
    for (uint32_t tokens : {2u, 4u}) {
        failures += fp8_case(4096, 8192, tokens);   // MTP eh_proj
        failures += fp8_case(512, 4096, tokens);    // kv_a
    }
    failures += mix_case(37);
    failures += expand_mix_case(37);
    failures += index_case(37, 3000);
    failures += index_case(256, 8000);
    failures += vq_case<GLMVQ_GAIN_WEIGHT>(blocks);
    failures += vq_case<GLMVQ_GAIN_DOT>(blocks);
    printf(failures ? "FAIL %d\n" : "GLMVQ_VERIFY_PASS\n", failures);
    return failures != 0;
}
