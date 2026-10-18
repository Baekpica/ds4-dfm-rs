// glmvq_q4p (FP16 prefill Q4_K GEMM) vs an FP64 dot of the dequantized
// Q4_K weights. FP16 activations and the folded weight roundings measure
// ~4e-4 relative RMS; the bound is 1e-3. Cases cover token tails, more
// tiles than SMs (persistent CTAs) and a token beyond the FP16 range;
// BF16 input rows must prepare the same bytes as their F32 values. The FP8
// variant (E4M3 rows, F32 128 x 128 scales, BF16 out) adds the BF16 output
// rounding to its bound.
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include "../cuda/glm_vq_q4p.cuh"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)
static std::mt19937 rng(11);

static float h2f(const uint8_t *p) { __half h; memcpy(&h, p, 2); return __half2float(h); }

static double weight(const uint8_t *block, uint32_t j) {
    const uint8_t *q = block + 4;
    const uint32_t p = j / 64, within = j % 64, sub = 2 * p + (within >= 32), l = within % 32;
    const uint8_t byte = block[16 + 32 * p + l];
    const int value = within < 32 ? byte & 15 : byte >> 4;
    const int sc = sub < 4 ? q[sub] & 63 : (q[sub + 4] & 15) | ((q[sub - 4] >> 6) << 4);
    const int m = sub < 4 ? q[sub + 4] & 63 : (q[sub + 4] >> 4) | ((q[sub] >> 6) << 4);
    return (double)h2f(block) * sc * value - (double)h2f(block + 2) * m;
}

static double e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const double v = e ? ldexp(1.0 + m / 8.0, e - 7) : ldexp((double)m, -9);
    return b & 0x80 ? -v : v;
}

static int fp8_case(uint32_t rows, uint32_t cols, uint32_t tokens, bool big) {
    std::vector<uint8_t> w((size_t)rows * cols);
    for (auto &b : w) { do { b = rng() & 0xff; } while ((b & 0x7f) == 0x7f); }
    std::vector<float> sc((size_t)rows / GLMVQ_SCALE_BLOCK * (cols / GLMVQ_SCALE_BLOCK));
    for (auto &v : sc) { v = 1e-4f * (1.0f + (rng() % 1000) * 1e-3f); }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t)tokens * cols);
    for (auto &v : x) { v = nd(rng); }
    const uint32_t hot = 3;
    if (big) { for (uint32_t k = 0; k < cols; k++) { x[(size_t)hot * cols + k] *= 1e5f; } }
    uint8_t *dw; float *ds, *dx; __nv_bfloat16 *dy; void *scratch; uint32_t *bad;
    CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&ds, sc.size() * 4)); CK(cudaMalloc(&dx, x.size() * 4));
    CK(cudaMalloc(&dy, (size_t)tokens * rows * 2)); CK(cudaMalloc(&scratch, glmvq_q4p_bytes(tokens, cols)));
    CK(cudaMalloc(&bad, 4));
    CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    const glmvq_q4p_input in = glmvq_q4p_layout(scratch, tokens, cols);
    CK(glmvq_q4p_prepare(in, dx, tokens, cols, 0));
    CK(glmvq_q4p_gemm_fp8(dw, ds, in, dy, rows, cols, tokens, 0));
    std::vector<__nv_bfloat16> y((size_t)tokens * rows);
    CK(cudaMemcpy(y.data(), dy, y.size() * 2, cudaMemcpyDeviceToHost));
    double err = 0, ref2 = 0;
    for (uint32_t i = 0; i < 2048 + (big ? rows : 0); i++) {
        const uint32_t t = i < 2048 ? rng() % tokens : hot, r = i < 2048 ? rng() % rows : i - 2048;
        double ref = 0;
        for (uint32_t k = 0; k < cols; k++) {
            ref += e4m3(w[(size_t)r * cols + k]) *
                sc[(size_t)(r / GLMVQ_SCALE_BLOCK) * (cols / GLMVQ_SCALE_BLOCK) + k / GLMVQ_SCALE_BLOCK] *
                x[(size_t)t * cols + k];
        }
        const double scale = t == hot && big ? 1e-5 : 1.0, got = __bfloat162float(y[(size_t)t * rows + r]);
        err += (got - ref) * (got - ref) * scale * scale;
        ref2 += ref * ref * scale * scale;
    }
    const double rel = sqrt(err / ref2);
    printf("fp8 rows=%u cols=%u tokens=%u big=%d: relRMS %.3g\n", rows, cols, tokens, (int)big, rel);
    int failures = !(rel < 4e-3);
    // Scales that would overflow the folded FP16 weight are rejected.
    const uint64_t blocks = sc.size();
    CK(cudaMemset(bad, 0, 4));
    glmvq_q4p_check_fp8<<<(unsigned)((blocks + 255) / 256), 256>>>(ds, blocks, bad);
    uint32_t flag = 1; CK(cudaMemcpy(&flag, bad, 4, cudaMemcpyDeviceToHost));
    const float huge = 3.0f;
    CK(cudaMemcpy(ds + blocks - 1, &huge, 4, cudaMemcpyHostToDevice)); CK(cudaMemset(bad, 0, 4));
    glmvq_q4p_check_fp8<<<(unsigned)((blocks + 255) / 256), 256>>>(ds, blocks, bad);
    uint32_t flagged = 0; CK(cudaMemcpy(&flagged, bad, 4, cudaMemcpyDeviceToHost));
    if (flag || !flagged) { printf("  fp8 check: in-range %u, out-of-range %u\n", flag, flagged); failures++; }
    CK(cudaFree(dw)); CK(cudaFree(ds)); CK(cudaFree(dx)); CK(cudaFree(dy)); CK(cudaFree(scratch)); CK(cudaFree(bad));
    return failures;
}

int main() {
    CK(glmvq_q4p_init());
    int failures = 0;
    failures += fp8_case(8192, 4096, 640, false);
    failures += fp8_case(1024, 512, 300, true);
    struct Case { uint32_t rows, cols, tokens; bool big; };
    for (Case c : {Case{2048, 1536, 640, false}, Case{512, 4096, 300, true}, Case{2048, 1536, 1000, false}}) {
        const uint32_t blocks = c.cols / GLMVQ_Q4_BLOCK;
        std::vector<uint8_t> w((size_t)c.rows * blocks * GLMVQ_Q4_BYTES);
        for (size_t b = 0; b < w.size() / GLMVQ_Q4_BYTES; b++) {
            uint8_t *block = w.data() + b * GLMVQ_Q4_BYTES;
            const __half d = __float2half(0.0005f + (rng() % 1000) * 1e-6f), dmin = __float2half(0.0002f + (rng() % 1000) * 1e-6f);
            memcpy(block, &d, 2); memcpy(block + 2, &dmin, 2);
            for (uint32_t i = 4; i < GLMVQ_Q4_BYTES; i++) { block[i] = rng() & 0xff; }
        }
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> x((size_t)c.tokens * c.cols);
        for (auto &v : x) { v = nd(rng); }
        const uint32_t big = 3;
        if (c.big) { for (uint32_t k = 0; k < c.cols; k++) { x[(size_t)big * c.cols + k] *= 1e5f; } }
        uint8_t *dw; float *dx, *dy; void *scratch;
        CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMalloc(&dy, (size_t)c.tokens * c.rows * 4)); CK(cudaMalloc(&scratch, glmvq_q4p_bytes(c.tokens, c.cols)));
        CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        const glmvq_q4p_input in = glmvq_q4p_layout(scratch, c.tokens, c.cols);
        CK(glmvq_q4p_prepare(in, dx, c.tokens, c.cols, 0));
        CK(glmvq_q4p_gemm(dw, in, dy, c.rows, c.cols, c.tokens, 0));
        std::vector<float> y((size_t)c.tokens * c.rows);
        CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
        // Sampled outputs plus every row of the big token.
        double err = 0, ref2 = 0;
        for (uint32_t i = 0; i < 2048 + (c.big ? c.rows : 0); i++) {
            const uint32_t t = i < 2048 ? rng() % c.tokens : big, r = i < 2048 ? rng() % c.rows : i - 2048;
            double ref = 0;
            for (uint32_t k = 0; k < c.cols; k++) {
                ref += weight(w.data() + ((size_t)r * blocks + k / GLMVQ_Q4_BLOCK) * GLMVQ_Q4_BYTES, k % GLMVQ_Q4_BLOCK) *
                    x[(size_t)t * c.cols + k];
            }
            const double scale = t == big && c.big ? 1e-5 : 1.0;
            err += (y[(size_t)t * c.rows + r] - ref) * (y[(size_t)t * c.rows + r] - ref) * scale * scale;
            ref2 += ref * ref * scale * scale;
        }
        const double rel = sqrt(err / ref2);
        printf("rows=%u cols=%u tokens=%u big=%d: relRMS %.3g\n", c.rows, c.cols, c.tokens, (int)c.big, rel);
        if (!(rel < 1e-3)) { failures++; }
        // One block beyond the folded range is rejected.
        uint32_t *bad; CK(cudaMalloc(&bad, 4)); CK(cudaMemset(bad, 0, 4));
        glmvq_q4p_check<<<(c.rows * blocks + 255) / 256, 256>>>(dw, (uint64_t)c.rows * blocks, bad);
        uint32_t flag = 1; CK(cudaMemcpy(&flag, bad, 4, cudaMemcpyDeviceToHost));
        const __half one = __float2half(1.0f);
        uint8_t edited[GLMVQ_Q4_BYTES];
        memcpy(edited, w.data(), GLMVQ_Q4_BYTES); memcpy(edited, &one, 2); edited[4] = 63;
        CK(cudaMemcpy(dw, edited, GLMVQ_Q4_BYTES, cudaMemcpyHostToDevice)); CK(cudaMemset(bad, 0, 4));
        glmvq_q4p_check<<<(c.rows * blocks + 255) / 256, 256>>>(dw, (uint64_t)c.rows * blocks, bad);
        uint32_t flagged = 0; CK(cudaMemcpy(&flagged, bad, 4, cudaMemcpyDeviceToHost));
        if (flag || !flagged) { printf("  check: in-range %u, out-of-range %u\n", flag, flagged); failures++; }
        // BF16 rows of BF16-rounded values prepare the same FP16 operand.
        std::vector<__nv_bfloat16> xb(x.size());
        for (size_t i = 0; i < x.size(); i++) { xb[i] = __float2bfloat16_rn(x[i]); x[i] = __bfloat162float(xb[i]); }
        __nv_bfloat16 *dxb; void *scratch_b;
        CK(cudaMalloc(&dxb, xb.size() * 2)); CK(cudaMalloc(&scratch_b, glmvq_q4p_bytes(c.tokens, c.cols)));
        CK(cudaMemcpy(dxb, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        const glmvq_q4p_input in_b = glmvq_q4p_layout(scratch_b, c.tokens, c.cols);
        CK(glmvq_q4p_prepare(in, dx, c.tokens, c.cols, 0));
        CK(glmvq_q4p_prepare(in_b, dxb, c.tokens, c.cols, 0));
        const size_t bytes = glmvq_q4p_bytes(c.tokens, c.cols);
        std::vector<uint8_t> pf(bytes), pb(bytes);
        CK(cudaMemcpy(pf.data(), scratch, bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pb.data(), scratch_b, bytes, cudaMemcpyDeviceToHost));
        if (pf != pb) { printf("  BF16 rows prepare a different operand\n"); failures++; }
        CK(cudaFree(dxb)); CK(cudaFree(scratch_b));
        CK(cudaFree(dw)); CK(cudaFree(dx)); CK(cudaFree(dy)); CK(cudaFree(scratch)); CK(cudaFree(bad));
    }
    puts(failures ? "GLMVQ_Q4P_FAIL" : "GLMVQ_Q4P_PASS");
    return failures != 0;
}
