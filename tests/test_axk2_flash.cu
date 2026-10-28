// Both BF16 schedules must equal CPU rounding of the F32 flash result.
#include "cuda/axk2_flash.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(cudaError_t err) {
    if (err != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(err)); std::exit(1); }
}

template <typename T> static T *upload(const std::vector<T> &v) {
    T *p = nullptr;
    check(cudaMalloc(&p, v.size() * sizeof(T)));
    check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

__global__ void pack_rows(__nv_bfloat16 *out, const float *in, uint32_t cols, uint32_t stride, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = __float2bfloat16_rn(in[i / cols * stride + i % cols]); }
}

enum class selection { dense, sparse };
enum class timing { off, pipeline };

static bool run(uint32_t rows, uint32_t pos, selection keys_mode, timing measure) {
    const bool sparse = keys_mode == selection::sparse;
    constexpr uint32_t H = DS4_AXK2_HEADS, L = DS4_AXK2_LATENT, K = DS4_AXK2_KEY;
    const uint32_t keys = pos + rows, cap = keys + 32;
    const uint64_t count = (uint64_t)rows * H * L;
    std::mt19937 rng(917);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::vector<float> q((uint64_t)rows * H * K), qa(count);
    for (auto &v : q) { v = unit(rng); }
    for (auto &v : qa) { v = unit(rng); }
    std::vector<uint8_t> latent((uint64_t)cap * L);
    for (auto &v : latent) { v = __nv_cvt_float_to_fp8(unit(rng), __NV_SATFINITE, __NV_E4M3); }
    std::vector<int8_t> exps(cap);
    for (uint32_t k = 0; k < keys; k++) { exps[k] = (int)(rng() % 9) - 4; }
    std::vector<__nv_bfloat16> pe((uint64_t)cap * DS4_AXK2_ROPE);
    for (auto &v : pe) { v = __float2bfloat16_rn(unit(rng)); }
    std::vector<uint32_t> selected((uint64_t)rows * DS4_AXK2_TOPK, UINT32_MAX);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t k = 0; k < DS4_AXK2_TOPK; k++) {
            // Short rows include masked fillers; long rows gather a sorted subset.
            const uint32_t key = keys > DS4_AXK2_TOPK ? k * keys / DS4_AXK2_TOPK : k;
            if (key < keys) { selected[(uint64_t)r * DS4_AXK2_TOPK + k] = key; }
        }
    }
    auto *dq = upload(q), *da = upload(qa);
    auto *dl = upload(latent);
    auto *de = upload(exps);
    auto *dp = upload(pe);
    auto *ds = upload(selected);
    float *out = nullptr;
    __nv_bfloat16 *packed = nullptr;
    uint8_t *guarded = nullptr;
    check(cudaMalloc(&out, count * sizeof(float)));
    check(cudaMalloc(&packed, count * sizeof(__nv_bfloat16)));
    check(cudaMalloc(&guarded, count * sizeof(__nv_bfloat16) + 32));
    check(cudaMemset(guarded, 0xa5, count * sizeof(__nv_bfloat16) + 32));
    auto *direct = (__nv_bfloat16 *)(guarded + 16);
    auto launch = [&](auto kernel, auto *dst) {
        kernel<<<rows, GLMVQ_FA_THREADS, L31_FA8_SMEM>>>(dst, nullptr, nullptr, dq, da,
            dl, de, dp, sparse ? ds : nullptr, pos, UINT32_MAX, 0.08f);
    };
    check(cudaFuncSetAttribute(ax_fa_kernel<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, L31_FA8_SMEM));
    check(cudaFuncSetAttribute(ax_fa_kernel<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, L31_FA8_SMEM));
    check(cudaFuncSetAttribute(ax_fa_kernel<__nv_bfloat16, ax_fa_keys::cooperative>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, L31_FA8_SMEM));
    launch(ax_fa_kernel<float>, out);
    launch(ax_fa_kernel<__nv_bfloat16>, packed);
    launch(ax_fa_kernel<__nv_bfloat16, ax_fa_keys::cooperative>, direct);
    check(cudaDeviceSynchronize());
    std::vector<float> ref(count);
    std::vector<__nv_bfloat16> serial(count);
    std::vector<uint8_t> got(count * sizeof(__nv_bfloat16) + 32);
    check(cudaMemcpy(ref.data(), out, count * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(serial.data(), packed, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(got.data(), guarded, got.size(), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (uint64_t i = 0; i < count; i++) {
        const auto expected = __float2bfloat16_rn(ref[i]);
        if (!std::isfinite(ref[i]) || memcmp(&expected, &serial[i], 2) ||
            memcmp(&expected, got.data() + 16 + i * 2, 2)) { ok = false; break; }
    }
    for (uint32_t i = 0; i < 16; i++) {
        ok = ok && got[i] == 0xa5 && got[got.size() - 16 + i] == 0xa5;
    }
    printf("rows=%u pos=%u sparse=%d BF16 output/canaries %s\n", rows, pos, sparse, ok ? "PASS" : "FAIL");
    if (ok && measure == timing::pipeline) {
        cudaEvent_t a, b;
        check(cudaEventCreate(&a)); check(cudaEventCreate(&b));
        for (int repeat = 0; repeat < 3; repeat++) {
            for (int order = 0; order < 3; order++) {
                const int mode = (repeat + order) % 3;
                check(cudaEventRecord(a));
                for (int i = 0; i < 20; i++) {
                    if (mode == 2) { launch(ax_fa_kernel<__nv_bfloat16, ax_fa_keys::cooperative>, direct); }
                    else if (mode == 1) { launch(ax_fa_kernel<__nv_bfloat16>, packed); }
                    else {
                        launch(ax_fa_kernel<float>, out);
                        pack_rows<<<(count + 255) / 256, 256>>>(packed, out, L, L, count);
                    }
                }
                check(cudaEventRecord(b)); check(cudaEventSynchronize(b));
                float ms; check(cudaEventElapsedTime(&ms, a, b));
                printf("mode=%d ms=%.6f\n", mode, ms / 20);
            }
        }
        check(cudaEventDestroy(a)); check(cudaEventDestroy(b));
    }
    for (void *p : { (void *)dq, (void *)da, (void *)dl, (void *)de, (void *)dp, (void *)ds,
            (void *)out, (void *)packed, (void *)guarded }) { check(cudaFree(p)); }
    return ok;
}

int main(int argc, char **) {
    bool ok = run(1, 0, selection::dense, timing::off) && run(48, 17, selection::dense, timing::off) &&
        run(49, 32, selection::sparse, timing::off) && run(64, 2033, selection::sparse, timing::off) &&
        run(255, 6145, selection::sparse, timing::off) &&
        run(256, 7936, selection::sparse, argc > 1 ? timing::pipeline : timing::off);
    return ok ? 0 : 1;
}
