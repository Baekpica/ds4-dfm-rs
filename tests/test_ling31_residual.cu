/* Ling-3.1 fused residual + RMS parity: cur and norm must equal ds4_cuda.cu's
 * add_kernel, glmvq_round_rows and glmvq_rms bit for bit, at prefill and
 * decode row counts; BF16 norm rows hold the same values. */
#include "cuda/l31_residual.cuh"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// ds4_cuda.cu add_kernel, compiled with the same flags.
__global__ static void reference_add(float *out, const float *a, const float *b, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = a[i] + b[i]; }
}

static bool check(cudaError_t err, const char *what) {
    if (err == cudaSuccess) { return true; }
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    return false;
}

static bool run(uint32_t rows, uint32_t width, std::mt19937 &rng) {
    const uint64_t n = (uint64_t)rows * width;
    std::normal_distribution<float> residual(0.0f, 4.0f), branch(0.0f, 0.5f);
    std::vector<float> cur(n), add(n);
    std::vector<__nv_bfloat16> weight(width);
    // BF16-rounded residuals, as the graph keeps them, plus a raw branch.
    for (uint64_t i = 0; i < n; i++) {
        cur[i] = __bfloat162float(__float2bfloat16_rn(residual(rng)));
        add[i] = branch(rng);
    }
    for (uint32_t i = 0; i < width; i++) { weight[i] = __float2bfloat16_rn(1.0f + 0.1f * branch(rng)); }
    float *d_ref = nullptr, *d_fused = nullptr, *d_add = nullptr, *d_norm_ref = nullptr, *d_norm = nullptr;
    float *d_twice = nullptr;
    __nv_bfloat16 *d_weight = nullptr, *d_norm16 = nullptr;
    const uint64_t bytes = n * sizeof(float);
    bool ok = check(cudaMalloc(&d_ref, bytes), "alloc") && check(cudaMalloc(&d_fused, bytes), "alloc") &&
        check(cudaMalloc(&d_add, bytes), "alloc") && check(cudaMalloc(&d_norm_ref, bytes), "alloc") &&
        check(cudaMalloc(&d_norm, bytes), "alloc") && check(cudaMalloc(&d_twice, bytes), "alloc") &&
        check(cudaMalloc(&d_norm16, n * sizeof(__nv_bfloat16)), "alloc") &&
        check(cudaMalloc(&d_weight, width * sizeof(__nv_bfloat16)), "alloc") &&
        check(cudaMemcpy(d_ref, cur.data(), bytes, cudaMemcpyHostToDevice), "copy") &&
        check(cudaMemcpy(d_fused, cur.data(), bytes, cudaMemcpyHostToDevice), "copy") &&
        check(cudaMemcpy(d_twice, cur.data(), bytes, cudaMemcpyHostToDevice), "copy") &&
        check(cudaMemcpy(d_add, add.data(), bytes, cudaMemcpyHostToDevice), "copy") &&
        check(cudaMemcpy(d_weight, weight.data(), width * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy");
    if (ok) {
        const float eps = 1.0e-6f;
        reference_add<<<(uint32_t)((n + 255) / 256), 256>>>(d_ref, d_ref, d_add, (uint32_t)n);
        glmvq_round_rows<<<(uint32_t)((n + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS>>>(d_ref, n);
        glmvq_rms<<<rows, GLMVQ_THREADS>>>(d_ref, d_weight, nullptr, d_norm_ref, width, eps, GLMVQ_RMS_WEIGHTED);
        l31_residual_rms<float><<<rows, GLMVQ_THREADS>>>(d_fused, d_add, d_weight, d_norm, width, eps);
        l31_residual_rms<__nv_bfloat16><<<rows, GLMVQ_THREADS>>>(d_twice, d_add, d_weight, d_norm16, width, eps);
        ok = check(cudaDeviceSynchronize(), "kernels");
    }
    std::vector<float> ref(n), fused(n), norm_ref(n), norm(n), twice(n);
    std::vector<__nv_bfloat16> norm16(n);
    ok = ok && check(cudaMemcpy(ref.data(), d_ref, bytes, cudaMemcpyDeviceToHost), "read") &&
        check(cudaMemcpy(twice.data(), d_twice, bytes, cudaMemcpyDeviceToHost), "read") &&
        check(cudaMemcpy(norm16.data(), d_norm16, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "read") &&
        check(cudaMemcpy(fused.data(), d_fused, bytes, cudaMemcpyDeviceToHost), "read") &&
        check(cudaMemcpy(norm_ref.data(), d_norm_ref, bytes, cudaMemcpyDeviceToHost), "read") &&
        check(cudaMemcpy(norm.data(), d_norm, bytes, cudaMemcpyDeviceToHost), "read");
    bool same = ok && !memcmp(ref.data(), fused.data(), bytes) && !memcmp(norm_ref.data(), norm.data(), bytes) &&
        !memcmp(ref.data(), twice.data(), bytes);
    for (uint64_t i = 0; same && i < n; i++) { same = __bfloat162float(norm16[i]) == norm[i]; }
    printf("rows=%u width=%u residual+norm (F32 and BF16 rows) %s\n", rows, width, same ? "bit-identical" : "DIFFER");
    cudaFree(d_ref); cudaFree(d_fused); cudaFree(d_add); cudaFree(d_norm_ref); cudaFree(d_norm); cudaFree(d_weight);
    cudaFree(d_twice); cudaFree(d_norm16);
    return same;
}

int main() {
    std::mt19937 rng(31);
    bool ok = true;
    for (uint32_t rows : {1u, 3u, 2048u}) { ok = run(rows, 4096u, rng) && ok; }
    return ok ? 0 : 1;
}
