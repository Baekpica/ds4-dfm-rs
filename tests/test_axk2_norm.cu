// Cooperative W_down loads preserve residual, RMS, warp-dot and SiLU arithmetic.
#include "cuda/axk2_norm.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(cudaError_t err) {
    if (err != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(err)); std::exit(1); }
}
static float rounded(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
static float warp_sum(float *lanes) {
    for (uint32_t stride = 16; stride; stride >>= 1) {
        float next[32];
        for (uint32_t lane = 0; lane < 32; lane++) { next[lane] = lanes[lane] + lanes[lane ^ stride]; }
        memcpy(lanes, next, sizeof(next));
    }
    return lanes[0];
}

enum class residual { absent, present, zero };
static bool run(uint32_t rows, residual mode) {
    constexpr uint32_t H = DS4_AXK2_HIDDEN, R = DS4_AXK2_GATE_RANK;
    const uint64_t count = (uint64_t)rows * H, outputs = (uint64_t)rows * AX_GNORM_MID;
    std::mt19937 rng(823 + rows);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::vector<float> cur(count), branch(count), expected(outputs), reference(outputs), result(outputs);
    std::vector<__nv_bfloat16> weight(H), down(R * H);
    for (uint64_t i = 0; i < count; i++) {
        cur[i] = mode == residual::zero ? 0.0f : rounded(unit(rng) * 4.0f);
        branch[i] = mode == residual::zero ? 0.0f : unit(rng);
    }
    for (auto &v : weight) { v = __float2bfloat16_rn(1.0f + unit(rng) / 8.0f); }
    for (auto &v : down) { v = __float2bfloat16_rn(unit(rng) / 32.0f); }
    for (uint32_t r = 0; r < rows; r++) {
        float partial[AX_ROW_THREADS] = {}, y[H];
        for (uint32_t t = 0; t < AX_ROW_THREADS; t++) {
            for (uint32_t col = t; col < H; col += AX_ROW_THREADS) {
                const uint64_t at = (uint64_t)r * H + col;
                const float v = mode == residual::absent ? cur[at] : rounded(cur[at] + rounded(branch[at]));
                y[col] = v;
                partial[t] = std::fma(v, v, partial[t]);
            }
        }
        float sum = 0.0f;
        for (uint32_t w = 0; w < AX_ROW_WARPS; w++) { sum += warp_sum(partial + w * 32); }
        const float variance = sum / (float)H + AX_EPS;
        const float inv = (float)(1.0 / std::sqrt((double)variance));
        expected[r * AX_GNORM_MID + R] = inv;
        for (uint32_t c = 0; c < H; c++) { y[c] = rounded(__bfloat162float(weight[c]) * rounded(y[c] * inv)); }
        for (uint32_t k = 0; k < R; k++) {
            float lanes[32] = {};
            for (uint32_t lane = 0; lane < 32; lane++) {
                for (uint32_t c = lane * 8; c < H; c += 256) {
                    for (uint32_t j = 0; j < 8; j++) {
                        lanes[lane] = std::fma(__bfloat162float(down[k * H + c + j]), y[c + j], lanes[lane]);
                    }
                }
            }
            const float g = rounded(warp_sum(lanes));
            expected[r * AX_GNORM_MID + k] = (float)((double)g / (1.0 + std::exp(-(double)g)));
        }
    }
    float *x = nullptr, *b = nullptr;
    __nv_bfloat16 *w = nullptr, *d = nullptr;
    uint8_t *guarded = nullptr;
    std::vector<uint8_t> bytes(outputs * sizeof(float) + 32);
    check(cudaMalloc(&x, count * 4)); check(cudaMalloc(&b, count * 4));
    check(cudaMalloc(&w, H * 2)); check(cudaMalloc(&d, R * H * 2)); check(cudaMalloc(&guarded, bytes.size()));
    check(cudaMemcpy(x, cur.data(), count * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(b, branch.data(), count * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(w, weight.data(), H * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d, down.data(), R * H * 2, cudaMemcpyHostToDevice));
    bool ok = true;
    for (uint32_t variant = 0; variant < 4; variant++) {
        const auto kernel = variant % 2 ? ax_gnorm_down_kernel<ax_norm_load::staged> :
            ax_gnorm_down_kernel<ax_norm_load::reference>;
        check(cudaMemset(guarded, 0xa5, bytes.size()));
        check(glmvq_launch_kernel(variant < 2 ? GLMVQ_LAUNCH_PLAIN : GLMVQ_LAUNCH_PDL, kernel,
            dim3(R, rows), AX_ROW_THREADS, 0, nullptr, (float *)(guarded + 16), (const float *)x,
            mode == residual::absent ? (const float *)nullptr : (const float *)b,
            (const __nv_bfloat16 *)w, (const __nv_bfloat16 *)d));
        check(cudaMemcpy(bytes.data(), guarded, bytes.size(), cudaMemcpyDeviceToHost));
        memcpy(result.data(), bytes.data() + 16, outputs * 4);
        if (!variant) { reference = result; }
        ok = ok && !memcmp(reference.data(), result.data(), outputs * 4);
        for (uint64_t i = 0; i < outputs; i++) {
            int exponent = 0;
            std::frexp(std::fabs(expected[i]), &exponent);
            const float tolerance = i % AX_GNORM_MID == R ? 0.0f :
                expected[i] == 0.0f ? 0.0f : std::ldexp(0.501f, exponent - 8);
            ok = ok && std::isfinite(result[i]) && std::fabs(result[i] - expected[i]) <= tolerance;
        }
        for (uint32_t i = 0; i < 16; i++) { ok = ok && bytes[i] == 0xa5 && bytes[bytes.size() - 16 + i] == 0xa5; }
    }
    std::vector<float> unchanged(count);
    check(cudaMemcpy(unchanged.data(), x, count * 4, cudaMemcpyDeviceToHost));
    ok = ok && !memcmp(cur.data(), unchanged.data(), count * 4);
    check(cudaMemcpy(unchanged.data(), b, count * 4, cudaMemcpyDeviceToHost));
    ok = ok && !memcmp(branch.data(), unchanged.data(), count * 4);
    printf("rows=%u residual=%u CPU/parity/PDL/canaries %s\n", rows, (unsigned)mode, ok ? "PASS" : "FAIL");
    check(cudaFree(x)); check(cudaFree(b)); check(cudaFree(w)); check(cudaFree(d)); check(cudaFree(guarded));
    return ok;
}

int main() {
    for (uint32_t rows : {1u, 2u, 4u}) {
        for (residual mode : {residual::absent, residual::present, residual::zero}) {
            if (!run(rows, mode)) { return 1; }
        }
    }
    return 0;
}
