/* Tensor-core compact attention (ds4_gpu_glm53_flash_low) against an FP64
 * oracle and the production F32 reference kernel, at the model geometry. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

struct ds4_gpu_tensor { void *ptr; uint64_t bytes; };
enum { GLM53_COMPACT_THREADS = 256, GLM53_COMPACT_MAX_DIM = 1024 };
static int capture;
static int ds4_capture_active(void) { return capture; }
static cudaStream_t ds4_current_stream(void) { return 0; }
static int cuda_ok(cudaError_t status, const char *label) {
    if (status == cudaSuccess) { return 1; }
    fprintf(stderr, "GLM flash %s: %s\n", label, cudaGetErrorString(status));
    return 0;
}
static bool glm53_compact_has(const ds4_gpu_tensor *t, uint64_t n, uint64_t elem) {
    return t && t->ptr && n <= UINT64_MAX / elem && t->bytes >= n * elem;
}
static bool glm53_compact_shape(uint32_t rows, uint32_t heads,
        uint32_t latent_dim, uint32_t head_dim) {
    return rows && rows <= 65535u && heads && heads <= 256u &&
        latent_dim && latent_dim <= GLM53_COMPACT_MAX_DIM && latent_dim % 32u == 0u &&
        head_dim && head_dim <= GLM53_COMPACT_MAX_DIM && head_dim % 32u == 0u;
}
#include "../cuda/glm53_low_attn.cuh"
#include "../cuda/glm53_flash.cuh"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM flash FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

enum { HEADS = 64, DIM = 512 };
// F16 query, P and output rounding against an F32 reference: |error| is a
// small fraction of the output scale (cache values are below 1.6).
static const double FLASH_TOL = 4.0e-3;

static ds4_gpu_tensor alloc(uint64_t bytes) {
    ds4_gpu_tensor out = {NULL, bytes};
    CHECK(cuda_ok(cudaMalloc(&out.ptr, (size_t)bytes), "alloc"));
    return out;
}

static void check_case(uint32_t pos0, uint32_t rows, bool sparse) {
    const uint32_t cap = pos0 + rows;
    const uint32_t stride = sparse ? DS4_GLM53_MAX_SELECTED : 0u;
    std::vector<__half> cache((uint64_t)cap * DIM);
    std::vector<float> query((uint64_t)rows * HEADS * DIM);
    std::vector<uint32_t> selected((uint64_t)rows * stride);
    for (uint32_t r = 0u; r < cap; r++) {
        for (uint32_t d = 0u; d < DIM; d++) {
            const float value = sinf((float)(r * 19u + d * 7u) * 0.017f) * 0.8f +
                (float)((int)((r + d) % 13u) - 6) * 0.03125f;
            cache[(uint64_t)r * DIM + d] = __float2half_rn(value);
        }
    }
    for (uint32_t t = 0u; t < rows; t++) {
        for (uint32_t h = 0u; h < HEADS; h++) {
            for (uint32_t d = 0u; d < DIM; d++) {
                query[((uint64_t)t * HEADS + h) * DIM + d] =
                    cosf((float)(d * 11u + h * 13u + t * 17u) * 0.023f) * (0.25f + 0.05f * (h % 7u));
            }
        }
        // Distinct keys up to the row, a future key (masked), then unused slots.
        for (uint32_t s = 0u; s < stride; s++) {
            const uint32_t visible = pos0 + t + 1u;
            const uint32_t used = stride - 3u;
            selected[(uint64_t)t * stride + s] = s >= used ? UINT32_MAX
                : s == used - 1u ? visible + 5u
                : s == used - 2u ? visible - 1u
                : (s * 397u + t * 23u) % visible;
        }
    }
    ds4_gpu_tensor dcache = alloc(cache.size() * sizeof(__half));
    ds4_gpu_tensor q = alloc(query.size() * sizeof(float)), out = alloc(q.bytes);
    ds4_gpu_tensor q2 = alloc(q.bytes), ref = alloc(q.bytes), picked = {NULL, 0u};
    CHECK(cuda_ok(cudaMemcpy(dcache.ptr, cache.data(), (size_t)dcache.bytes, cudaMemcpyHostToDevice), "cache"));
    CHECK(cuda_ok(cudaMemcpy(q.ptr, query.data(), (size_t)q.bytes, cudaMemcpyHostToDevice), "query"));
    CHECK(cuda_ok(cudaMemcpy(q2.ptr, query.data(), (size_t)q.bytes, cudaMemcpyHostToDevice), "query"));
    if (stride) {
        picked = alloc(selected.size() * sizeof(uint32_t));
        CHECK(cuda_ok(cudaMemcpy(picked.ptr, selected.data(), (size_t)picked.bytes, cudaMemcpyHostToDevice), "selection"));
    }
    CHECK(ds4_gpu_glm53_attn_low(&ref, &q, &dcache, stride ? &picked : NULL, stride, rows, pos0, cap,
        HEADS, DIM, 256u));
    CHECK(ds4_gpu_glm53_flash_low(&out, &q2, &dcache, stride ? &picked : NULL, stride, rows, pos0, cap,
        HEADS, DIM, 256u) == 1);
    std::vector<float> output(query.size()), expected_ref(query.size());
    CHECK(cuda_ok(cudaMemcpy(output.data(), out.ptr, (size_t)out.bytes, cudaMemcpyDeviceToHost), "out"));
    CHECK(cuda_ok(cudaMemcpy(expected_ref.data(), ref.ptr, (size_t)ref.bytes, cudaMemcpyDeviceToHost), "ref"));
    double worst_ref = 0.0;
    for (size_t i = 0; i < output.size(); i++) {
        CHECK(isfinite(output[i]));
        worst_ref = fmax(worst_ref, fabs((double)output[i] - (double)expected_ref[i]));
    }
    CHECK(worst_ref < FLASH_TOL);

    double worst = 0.0;
    const uint32_t sample[] = {0u, 1u, rows / 2u, rows - 1u};
    for (uint32_t t : sample) {
        const uint32_t visible = pos0 + t + 1u;
        const uint32_t count = stride ? stride : visible;
        for (uint32_t h = 0u; h < HEADS; h += 9u) {
            std::vector<double> scores(count);
            double maximum = -INFINITY;
            for (uint32_t s = 0u; s < count; s++) {
                const uint32_t r = stride ? selected[(uint64_t)t * stride + s] : s;
                double dot = 0.0;
                if (r < visible) {
                    for (uint32_t d = 0u; d < DIM; d++) {
                        dot += (double)query[((uint64_t)t * HEADS + h) * DIM + d] *
                            (double)__half2float(cache[(uint64_t)r * DIM + d]);
                    }
                }
                scores[s] = r < visible ? dot / 16.0 : -INFINITY;
                maximum = fmax(maximum, scores[s]);
            }
            double denominator = 0.0;
            for (uint32_t s = 0u; s < count; s++) {
                scores[s] = exp(scores[s] - maximum);
                denominator += scores[s];
            }
            for (uint32_t d = 0u; d < DIM; d++) {
                double expected = 0.0;
                for (uint32_t s = 0u; s < count; s++) {
                    const uint32_t r = stride ? selected[(uint64_t)t * stride + s] : s;
                    if (r < visible) { expected += scores[s] / denominator * (double)__half2float(cache[(uint64_t)r * DIM + d]); }
                }
                const double error = fabs((double)output[((uint64_t)t * HEADS + h) * DIM + d] - expected);
                if (error > FLASH_TOL) {
                    fprintf(stderr, "GLM flash mismatch pos0=%u row=%u head=%u dim=%u abs=%.9g\n", pos0, t, h, d, error);
                    exit(1);
                }
                worst = fmax(worst, error);
            }
        }
    }
    // Uncovered shapes fall back to the reference kernel.
    CHECK(ds4_gpu_glm53_flash_low(&out, &q2, &dcache, NULL, 0u, 1u, pos0, cap, HEADS, DIM, 256u) == 0);
    CHECK(ds4_gpu_glm53_flash_low(&out, &q2, &dcache, NULL, 0u, rows, pos0, cap, 4u, DIM, 256u) == 0);
    capture = 1;
    CHECK(ds4_gpu_glm53_flash_low(&out, &q2, &dcache, stride ? &picked : NULL, stride, rows, pos0, cap,
        HEADS, DIM, 256u) == 0);
    capture = 0;
    cudaFree(picked.ptr); cudaFree(ref.ptr); cudaFree(q2.ptr); cudaFree(out.ptr); cudaFree(q.ptr); cudaFree(dcache.ptr);
    printf("GLM flash: pos0=%u rows=%u selected=%u max_abs_oracle=%.3g max_abs_ref=%.3g PASS\n",
        pos0, rows, stride, worst, worst_ref);
}

int main(void) {
    check_case(0u, 300u, false);     // dense causal prefix from the first key
    check_case(1400u, 640u, false);  // dense prefix that ends at the selection edge
    check_case(2051u, 96u, true);    // selected keys with masked and unused slots
    check_case(6000u, 2u, true);     // two rows: smallest covered width
    return 0;
}
