// Drives ds4_gpu_mimo2_dflash_attn. The device path must match the host
// loops. DS4_MIMO2_DFLASH_CPU=1 must return those loops unchanged.
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
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

static void host_attn(std::vector<float> q, std::vector<float> k, std::vector<float> v,
                       const std::vector<float> &qw, const std::vector<float> &kw,
                       const std::vector<float> &sinks, unsigned q0, unsigned n, unsigned ctx,
                       std::vector<float> &out) {
    const unsigned kv_len = ctx + n;
    for (unsigned row = 0; row < n * DF_QH; row++) { df_rms(q.data() + row * DF_HD, qw.data(), DF_HD); }
    for (unsigned row = 0; row < kv_len * DF_KH; row++) { df_rms(k.data() + row * DF_HD, kw.data(), DF_HD); }
    for (unsigned i = 0; i < n; i++) { df_rope(q.data() + (size_t)i * DF_Q, DF_QH, q0 + i); }
    for (unsigned i = 0; i < ctx; i++) { df_rope(k.data() + (size_t)i * DF_KV, DF_KH, q0 - ctx + i); }
    for (unsigned i = 0; i < n; i++) { df_rope(k.data() + ((size_t)ctx + i) * DF_KV, DF_KH, q0 + i); }
    std::vector<unsigned> k_pos(kv_len);
    for (unsigned i = 0; i < ctx; i++) { k_pos[i] = q0 - ctx + i; }
    for (unsigned i = 0; i < n; i++) { k_pos[ctx + i] = q0 + i; }
    out.assign((size_t)n * DF_Q, 0.0f);
    df_attn(out.data(), q.data(), k.data(), v.data(), sinks.data(), k_pos.data(), q0, n, kv_len);
}

static int launch(const std::vector<float> &q, const std::vector<float> &k_ctx,
                  const std::vector<float> &k_noise, const std::vector<float> &v_ctx,
                  const std::vector<float> &v_noise, const std::vector<float> &qw,
                  const std::vector<float> &kw, const std::vector<float> &sinks,
                  unsigned q0, unsigned n, unsigned ctx, std::vector<float> &got) {
    float *dq, *dkc, *dkn, *dvc, *dvn, *dout;
    if (cudaMalloc(&dq, q.size() * sizeof(float)) || cudaMalloc(&dkc, k_ctx.size() * sizeof(float)) ||
        cudaMalloc(&dkn, k_noise.size() * sizeof(float)) || cudaMalloc(&dvc, v_ctx.size() * sizeof(float)) ||
        cudaMalloc(&dvn, v_noise.size() * sizeof(float)) || cudaMalloc(&dout, got.size() * sizeof(float))) {
        return 0;
    }
    cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dkc, k_ctx.data(), k_ctx.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dkn, k_noise.data(), k_noise.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc, v_ctx.data(), v_ctx.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvn, v_noise.data(), v_noise.size() * sizeof(float), cudaMemcpyHostToDevice);
    ds4_gpu_tensor attn{dout, got.size() * sizeof(float), 0, 0};
    ds4_gpu_tensor tq{dq, q.size() * sizeof(float), 0, 0};
    ds4_gpu_tensor tkc{dkc, k_ctx.size() * sizeof(float), 0, 0};
    ds4_gpu_tensor tkn{dkn, k_noise.size() * sizeof(float), 0, 0};
    ds4_gpu_tensor tvc{dvc, v_ctx.size() * sizeof(float), 0, 0};
    ds4_gpu_tensor tvn{dvn, v_noise.size() * sizeof(float), 0, 0};
    const int rc = ds4_gpu_mimo2_dflash_attn(
        &attn, &tq, &tkc, &tkn, &tvc, &tvn, qw.data(), kw.data(), sinks.data(), q0, n, ctx);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dout, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dq); cudaFree(dkc); cudaFree(dkn); cudaFree(dvc); cudaFree(dvn); cudaFree(dout);
    return rc;
}

static int run_case(unsigned n, unsigned ctx, unsigned q0, int plant_window) {
    std::vector<float> q((size_t)n * DF_Q), k_ctx((size_t)ctx * DF_KV), k_noise((size_t)n * DF_KV);
    std::vector<float> v_ctx((size_t)ctx * DF_KV), v_noise((size_t)n * DF_KV);
    std::vector<float> qw(DF_HD), kw(DF_HD), sinks(DF_QH);
    for (size_t i = 0; i < q.size(); i++) { q[i] = sinf((float)i * 0.02f); }
    for (size_t i = 0; i < k_ctx.size(); i++) { k_ctx[i] = sinf((float)i * 0.017f); }
    for (size_t i = 0; i < k_noise.size(); i++) { k_noise[i] = cosf((float)i * 0.013f); }
    for (size_t i = 0; i < v_ctx.size(); i++) { v_ctx[i] = sinf((float)i * 0.011f) * 0.1f; }
    for (size_t i = 0; i < v_noise.size(); i++) { v_noise[i] = cosf((float)i * 0.009f) * 0.1f; }
    for (unsigned i = 0; i < DF_HD; i++) { qw[i] = 0.8f + 0.01f * (float)(i % 7); }
    for (unsigned i = 0; i < DF_HD; i++) { kw[i] = 0.7f + 0.01f * (float)(i % 5); }
    for (unsigned i = 0; i < DF_QH; i++) { sinks[i] = (i % 4 == 0) ? 0.5f : -0.25f; }
    if (plant_window) {
        for (unsigned d = 0; d < DF_KV; d++) { k_ctx[d] = 40.0f; }
    }
    std::vector<float> k = k_ctx;
    k.insert(k.end(), k_noise.begin(), k_noise.end());
    std::vector<float> v = v_ctx;
    v.insert(v.end(), v_noise.begin(), v_noise.end());
    std::vector<float> ref, got((size_t)n * DF_Q);
    host_attn(q, k, v, qw, kw, sinks, q0, n, ctx, ref);
    unsetenv("DS4_MIMO2_DFLASH_CPU");
    const int gpu = launch(q, k_ctx, k_noise, v_ctx, v_noise, qw, kw, sinks, q0, n, ctx, got);
    double err = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) { return 0; }
        err = std::max(err, (double)std::fabs(got[i] - ref[i]));
    }
    printf("dflash_gpu n=%u ctx=%u q0=%u rc=%d err=%.9g\n", n, ctx, q0, gpu, err);
    if (gpu != 2 || err > 1e-4) { return 0; }
    setenv("DS4_MIMO2_DFLASH_CPU", "1", 1);
    const int cpu = launch(q, k_ctx, k_noise, v_ctx, v_noise, qw, kw, sinks, q0, n, ctx, got);
    unsetenv("DS4_MIMO2_DFLASH_CPU");
    for (size_t i = 0; i < ref.size(); i++) {
        if (got[i] != ref[i]) {
            printf("dflash_cpu mismatch at %zu\n", i);
            return 0;
        }
    }
    printf("dflash_cpu n=%u ctx=%u rc=%d exact=1\n", n, ctx, cpu);
    return cpu == 1;
}

int main() {
    if (!run_case(1, 1, 1, 0)) { return 4; }
    if (!run_case(2, 4, 4, 0)) { return 1; }
    if (!run_case(4, 17, 17, 0)) { return 2; }
    if (!run_case(2, 1024, 2000, 1)) { return 3; }
    if (!run_case(8, 1024, 262144, 1)) { return 5; }
    m2_split_release();
    puts("dflash_attn=host_match");
    return 0;
}
