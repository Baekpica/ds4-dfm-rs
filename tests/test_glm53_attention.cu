/* Production attention wrapper/kernels with an independent double CPU oracle. */
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
    fprintf(stderr, "GLM attention %s: %s\n", label, cudaGetErrorString(status));
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

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM attention FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static ds4_gpu_tensor alloc(uint64_t bytes) {
    ds4_gpu_tensor out = {NULL, bytes};
    CHECK(cuda_ok(cudaMalloc(&out.ptr, (size_t)bytes), "alloc"));
    return out;
}

enum { WIDE_ROWS = 2048u, WIDE_SELECTED = 16u };

static void check_case(uint32_t visible, uint32_t rows, uint32_t heads,
        uint32_t dim, uint32_t offset, int extreme, glm53_attn_mode mode) {
    const uint32_t cap = visible + rows - 1u;
    const uint32_t pos0 = visible - 1u;
    const uint32_t stride = mode == GLM53_ATTN_SELECTED ? DS4_GLM53_MAX_SELECTED : 0u;
    std::vector<__half> cache((uint64_t)(cap + offset) * dim);
    std::vector<float> query((uint64_t)rows * heads * dim), output(query.size());
    std::vector<uint32_t> selected((uint64_t)rows * stride);
    for (uint32_t r = 0u; r < cap + offset; r++) {
        for (uint32_t d = 0u; d < dim; d++) {
            const float value = extreme ? (float)((int)(r % 7u) - 3) * 2.0f + (float)(d % 5u) * 0.015625f
                : sinf((float)(r * 19u + d * 7u) * 0.017f) * 0.8f +
                  (float)((int)((r + d) % 13u) - 6) * 0.03125f;
            cache[(uint64_t)r * dim + d] = __float2half_rn(value);
        }
    }
    for (uint32_t t = 0u; t < rows; t++) {
        for (uint32_t h = 0u; h < heads; h++) {
            for (uint32_t d = 0u; d < dim; d++) {
                query[((uint64_t)t * heads + h) * dim + d] = extreme ? (h & 1u ? -100.0f : 100.0f)
                    : cosf((float)(d * 11u + h * 13u + t * 17u) * 0.023f) * 0.25f;
            }
        }
        for (uint32_t s = 0u; s < stride; s++) {
            const uint32_t used = rows >= WIDE_ROWS ? WIDE_SELECTED : stride - 1u;
            selected[(uint64_t)t * stride + s] = s >= used ? UINT32_MAX
                : rows >= WIDE_ROWS && s + 1u == used ? visible + t
                : (s * 397u + t * 23u) % (visible + t);
        }
    }
    ds4_gpu_tensor full = alloc(cache.size() * sizeof(__half));
    ds4_gpu_tensor device_cache = {(char *)full.ptr + (uint64_t)offset * dim * sizeof(__half),
                                   (uint64_t)cap * dim * sizeof(__half)};
    ds4_gpu_tensor q = alloc(query.size() * sizeof(float)), out = alloc(q.bytes);
    ds4_gpu_tensor picked = {NULL, 0u};
    CHECK(cuda_ok(cudaMemcpy(full.ptr, cache.data(), (size_t)full.bytes, cudaMemcpyHostToDevice), "cache write"));
    CHECK(cuda_ok(cudaMemcpy(q.ptr, query.data(), (size_t)q.bytes, cudaMemcpyHostToDevice), "query write"));
    if (stride) {
        picked = alloc(selected.size() * sizeof(uint32_t));
        CHECK(cuda_ok(cudaMemcpy(picked.ptr, selected.data(), (size_t)picked.bytes, cudaMemcpyHostToDevice), "selection write"));
    }
    CHECK(ds4_gpu_glm53_attn_low(&out, &q, &device_cache, stride ? &picked : NULL,
        stride, rows, pos0, cap, heads, dim, 256u));
    CHECK(cuda_ok(cudaMemcpy(output.data(), out.ptr, (size_t)out.bytes, cudaMemcpyDeviceToHost), "output read"));
    for (float value : output) { CHECK(isfinite(value)); }

    // Pairing changes load reuse, never the selected-order FMA transition.
    if (dim == 512u && (stride || cap <= DS4_GLM53_MAX_SELECTED)) {
        ds4_gpu_tensor ref = alloc(out.bytes);
        std::vector<float> expected(output.size());
        glm53_low_attn_kernel<GLM53_ATTN_REF><<<dim3(heads, rows), GLM53_COMPACT_THREADS>>>(
            (float *)ref.ptr, (const float *)q.ptr, (const __half *)device_cache.ptr,
            stride ? (const uint32_t *)picked.ptr : NULL, stride, pos0, heads, dim, 1.0f / 16.0f);
        CHECK(cuda_ok(cudaGetLastError(), "reference launch"));
        CHECK(cuda_ok(cudaMemcpy(expected.data(), ref.ptr, (size_t)ref.bytes, cudaMemcpyDeviceToHost), "reference read"));
        CHECK(memcmp(expected.data(), output.data(), (size_t)ref.bytes) == 0);
        glm53_low_attn_kernel<GLM53_ATTN_PAIR><<<dim3(heads, rows), GLM53_COMPACT_THREADS>>>(
            (float *)ref.ptr, (const float *)q.ptr, (const __half *)device_cache.ptr,
            stride ? (const uint32_t *)picked.ptr : NULL, stride, pos0, heads, dim, 1.0f / 16.0f);
        CHECK(cuda_ok(cudaGetLastError(), "pair launch"));
        std::vector<float> paired(output.size());
        CHECK(cuda_ok(cudaMemcpy(paired.data(), ref.ptr, (size_t)ref.bytes, cudaMemcpyDeviceToHost), "pair read"));
        CHECK(memcmp(expected.data(), paired.data(), (size_t)ref.bytes) == 0);
        CHECK(cuda_ok(cudaFree(ref.ptr), "reference free"));
    }

    double worst = 0.0;
    uint32_t cpu_rows = 0u;
    for (uint32_t t = 0u; t < rows; t++) {
        /* Wide coverage keeps the production grid and selection stride;
         * sample the scalar dot/softmax while checking every GPU row finite. */
        if (rows >= WIDE_ROWS && t != 0u && t != 1u && t != rows / 2u && t != rows - 1u) {
            continue;
        }
        cpu_rows++;
        const uint32_t count = stride ? stride : visible + t;
        for (uint32_t h = 0u; h < heads; h++) {
            std::vector<double> scores(count);
            double maximum = -INFINITY;
            for (uint32_t s = 0u; s < count; s++) {
                const uint32_t r = stride ? selected[(uint64_t)t * stride + s] : s;
                double dot = 0.0;
                if (r < visible + t) {
                    for (uint32_t d = 0u; d < dim; d++) {
                        dot += (double)query[((uint64_t)t * heads + h) * dim + d] *
                            (double)__half2float(cache[((uint64_t)offset + r) * dim + d]);
                    }
                }
                scores[s] = r < visible + t ? dot / 16.0 : -INFINITY;
                maximum = fmax(maximum, scores[s]);
            }
            double denominator = 0.0;
            for (uint32_t s = 0u; s < count; s++) {
                scores[s] = exp(scores[s] - maximum);
                denominator += scores[s];
            }
            for (uint32_t d = 0u; d < dim; d++) {
                double expected = 0.0;
                for (uint32_t s = 0u; s < count; s++) {
                    const uint32_t r = stride ? selected[(uint64_t)t * stride + s] : s;
                    if (r < visible + t) {
                        expected += scores[s] / denominator *
                            (double)__half2float(cache[((uint64_t)offset + r) * dim + d]);
                    }
                }
                const float actual = output[((uint64_t)t * heads + h) * dim + d];
                const double error = fabs((double)actual - expected);
                if (!isfinite(actual) || error > 3.0e-4) {
                    fprintf(stderr, "GLM attention mismatch visible=%u row=%u head=%u dim=%u actual=%.9g expected=%.9g abs=%.9g\n",
                        visible, t, h, d, actual, expected, error);
                    exit(1);
                }
                worst = fmax(worst, error);
            }
        }
    }
    CHECK(!ds4_gpu_glm53_attn_low(&out, &q, &device_cache, NULL, 0u, 0u,
        pos0, cap, heads, dim, 256u));
    CHECK(!ds4_gpu_glm53_attn_low(&out, &q, &device_cache, NULL, 0u, 2u,
        UINT32_MAX - 1u, UINT32_MAX, heads, dim, 256u));
    capture = 1;
    CHECK(!ds4_gpu_glm53_attn_low(&out, &q, &device_cache, stride ? &picked : NULL,
        stride, rows, pos0, cap, heads, dim, 256u));
    capture = 0;
    CHECK(cuda_ok(cudaFree(picked.ptr), "selection free"));
    CHECK(cuda_ok(cudaFree(out.ptr), "output free"));
    CHECK(cuda_ok(cudaFree(q.ptr), "query free"));
    CHECK(cuda_ok(cudaFree(full.ptr), "cache free"));
    printf("GLM attention: visible=%u rows=%u heads=%u dim=%u offset=%u extreme=%d selected=%u cpu_rows=%u max_abs=%.9g PASS\n",
        visible, rows, heads, dim, offset, extreme, stride, cpu_rows, worst);
}

int main(void) {
    check_case(2051u, 1u, 3u, 512u, 0u, 0, GLM53_ATTN_ALL);
    check_case(2052u, 1u, 3u, 512u, 5u, 0, GLM53_ATTN_ALL);
    check_case(4097u, 3u, 4u, 512u, 13u, 0, GLM53_ATTN_ALL);
    check_case(4097u, 3u, 4u, 1024u, 7u, 1, GLM53_ATTN_ALL);
    check_case(4097u, 3u, 4u, 512u, 3u, 0, GLM53_ATTN_SELECTED);
    check_case(2051u, WIDE_ROWS, 1u, 512u, 3u, 0, GLM53_ATTN_SELECTED);
    return 0;
}
