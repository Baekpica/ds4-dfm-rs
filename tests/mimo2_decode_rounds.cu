// Drives ds4_gpu_mimo2_attention for the three decode-round switches.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool m2_hmma_available = false;
static int cuda_ok(cudaError_t err, const char *) { return err == cudaSuccess; }
static inline cudaStream_t ds4_current_stream(void) { return 0; }
static inline int ds4_capture_active(void) { return 0; }
static const char *cuda_model_range_ptr(const void *, uint64_t, uint64_t, const char *) { return nullptr; }
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; int owner; int memc; };
#include "../ds4_mimo2_gpu.cuh"

static int close_enough(const std::vector<float> &got, const std::vector<float> &ref) {
    for (size_t i = 0; i < got.size(); i++) {
        if (!std::isfinite(got[i]) || std::fabs(got[i] - ref[i]) > 2e-4f) { return 0; }
    }
    return 1;
}

static int run_case(unsigned window, unsigned kv_heads, unsigned cap, unsigned pos,
                    const char *swa, const char *vec, const char *split16, int want_path) {
    enum { HEADS = 64, KEY = 192, VALUE = 128 };
    const size_t stride = kv_heads * (KEY + VALUE);
    std::vector<__half> cache(cap * stride);
    std::vector<float> q(HEADS * KEY), host(HEADS * VALUE), walk(HEADS * VALUE);
    for (size_t i = 0; i < q.size(); i++) { q[i] = sinf((float)i * 0.017f); }
    for (size_t i = 0; i < cache.size(); i++) { cache[i] = __float2half_rn(sinf((float)i * 0.01f)); }
    float *dq = nullptr, *dout = nullptr;
    __half *dc = nullptr;
    unsigned *dp = nullptr;
    if (cudaMalloc(&dq, q.size() * sizeof(float)) || cudaMalloc(&dout, host.size() * sizeof(float)) ||
        cudaMalloc(&dc, cache.size() * sizeof(__half)) || cudaMalloc(&dp, sizeof(pos))) { return 0; }
    cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, cache.data(), cache.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dp, &pos, sizeof(pos), cudaMemcpyHostToDevice);
    mimo2_attention<<<dim3(HEADS / 4, 1), 128>>>(dout, dq, dc, nullptr, dp, kv_heads, cap, window);
    if (cudaDeviceSynchronize() != cudaSuccess) { return 0; }
    cudaMemcpy(walk.data(), dout, walk.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (swa) { setenv("DS4_MIMO2_SWA_DECODE", swa, 1); } else { unsetenv("DS4_MIMO2_SWA_DECODE"); }
    if (vec) { setenv("DS4_MIMO2_SPLIT_VEC", vec, 1); } else { unsetenv("DS4_MIMO2_SPLIT_VEC"); }
    if (split16) { setenv("DS4_MIMO2_SPLIT16", split16, 1); } else { unsetenv("DS4_MIMO2_SPLIT16"); }
    unsetenv("DS4_MIMO2_FATTN");
    unsetenv("DS4_MIMO2_ATTN_SPLIT");
    int zero = 0;
    cudaMemcpyToSymbol(m2_kernel_tag, &zero, sizeof(zero));
    ds4_gpu_tensor out{dout, host.size() * sizeof(float), 0, 0};
    ds4_gpu_tensor query{dq, q.size() * sizeof(float), 0, 0};
    ds4_gpu_tensor kv{dc, cache.size() * sizeof(__half), 0, 0};
    ds4_gpu_tensor positions{dp, sizeof(pos), 0, 0};
    const int rc = ds4_gpu_mimo2_attention(
        &out, &query, &kv, &positions, nullptr, 0, 0, kv_heads, cap, 1, window, pos);
    cudaMemcpy(host.data(), dout, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dq); cudaFree(dout); cudaFree(dc); cudaFree(dp);
    m2_split_release();
    int tag = 0;
    cudaMemcpyFromSymbol(&tag, m2_kernel_tag, sizeof(tag));
    if (tag != want_path || m2_attn_path != want_path) {
        fprintf(stderr, "kernel tag %d path %d want %d\n", tag, m2_attn_path, want_path);
        return 0;
    }
    return rc == 1 && close_enough(host, walk);
}

int main() {
    if (!run_case(128, 8, 256, 200, "1", nullptr, nullptr, 1)) {
        fprintf(stderr, "swa decode missed mimo2_swa_decode\n");
        return 1;
    }
    puts("swa_decode kernel=mimo2_swa_decode vec=0");
    setenv("DS4_MIMO2_SWA_VEC", "1", 1);
    if (!run_case(128, 8, 256, 200, "1", nullptr, nullptr, 2)) {
        fprintf(stderr, "swa vec missed mimo2_swa_decode\n");
        return 4;
    }
    unsetenv("DS4_MIMO2_SWA_VEC");
    puts("swa_vec kernel=mimo2_swa_decode vec=1");
    if (!run_case(0, 4, 256, 200, nullptr, "1", nullptr, 4)) {
        fprintf(stderr, "split vec missed mimo2_attn_split\n");
        return 2;
    }
    puts("split_vec kernel=mimo2_attn_split vec=1");
    if (!run_case(0, 4, 256, 200, nullptr, nullptr, "1", 3)) {
        fprintf(stderr, "split16 missed mimo2_attn_split\n");
        return 3;
    }
    puts("split16 kernel=mimo2_attn_split nsplit=16");
    return 0;
}
