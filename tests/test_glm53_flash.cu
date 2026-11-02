/* Tensor-core compact attention (ds4_gpu_glm53_flash_low) against an FP64
 * oracle and the production F32 reference kernel, at the model geometry. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <limits.h>
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
enum { GLM53_COMPACT_Q8_BLOCK = 32, GLM53_COMPACT_Q8_BYTES = 34 };
static cublasHandle_t g_cublas;
static int g_cublas_ready;
static bool cublas_ok(cublasStatus_t status, const char *label) {
    if (status == CUBLAS_STATUS_SUCCESS) { return true; }
    fprintf(stderr, "GLM flash %s: cuBLAS %d\n", label, (int)status);
    return false;
}
static void cuda_cublas_ws_prep(cudaStream_t) {}
// The model map is a device buffer here: offsets address it directly.
static const char *glm53_compact_weight(const void *map, uint64_t size, uint64_t offset,
        uint64_t bytes, const ds4_gpu_tensor *, const char *) {
    return offset <= size && bytes <= size - offset ? (const char *)map + offset : NULL;
}
#include "../cuda/glm53_low_attn.cuh"
#include "../cuda/glm53_pool_score.cuh"
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

enum { HEAD_DIM = 256 };

// Q8_0 rows of `width` values with small scales, plus their exact F32 values.
static void make_q8(std::vector<uint8_t> &raw, std::vector<float> &values, uint64_t count, uint32_t seed) {
    raw.resize(count / 32u * 34u);
    values.resize(count);
    for (uint64_t b = 0; b < count / 32u; b++) {
        const __half d = __float2half_rn(0.0015f + 0.0005f * (float)((b * 7u + seed) % 5u));
        memcpy(&raw[b * 34u], &d, sizeof(d));
        for (uint32_t i = 0; i < 32u; i++) {
            const int8_t v = (int8_t)((int)((b * 131u + i * 29u + seed) % 255u) - 127);
            raw[b * 34u + 2u + i] = (uint8_t)v;
            values[b * 32u + i] = __half2float(d) * (float)v;
        }
    }
}

// q (F32) -> heads (F32) through F16 GEMMs and the flash tile vs an FP64 chain.
static void check_chain(uint32_t pos0, uint32_t rows, bool sparse) {
    const uint32_t cap = pos0 + rows, stride = sparse ? DS4_GLM53_MAX_SELECTED : 0u;
    const uint64_t wcount = (uint64_t)HEADS * DIM * HEAD_DIM;
    std::vector<uint8_t> kraw, vraw;
    std::vector<float> kw, vw;
    make_q8(kraw, kw, wcount, 3u);
    make_q8(vraw, vw, wcount, 11u);
    std::vector<__half> cache((uint64_t)cap * DIM);
    for (uint64_t i = 0; i < cache.size(); i++) {
        cache[i] = __float2half_rn(sinf((float)(i % 9973u) * 0.013f) * 0.7f);
    }
    std::vector<float> query((uint64_t)rows * HEADS * HEAD_DIM);
    for (uint64_t i = 0; i < query.size(); i++) { query[i] = cosf((float)(i % 7919u) * 0.031f) * 0.6f; }
    std::vector<uint32_t> selected((uint64_t)rows * stride);
    for (uint32_t t = 0u; t < rows; t++) {
        for (uint32_t s = 0u; s < stride; s++) {
            const uint32_t visible = pos0 + t + 1u;
            selected[(uint64_t)t * stride + s] = s >= stride - 2u ? UINT32_MAX : (s * 613u + t * 37u) % visible;
        }
    }
    std::vector<uint8_t> map(kraw.size() + vraw.size());
    memcpy(map.data(), kraw.data(), kraw.size());
    memcpy(map.data() + kraw.size(), vraw.data(), vraw.size());
    ds4_gpu_tensor dmap = alloc(map.size()), dcache = alloc(cache.size() * sizeof(__half));
    ds4_gpu_tensor q = alloc(query.size() * sizeof(float)), out = alloc(q.bytes);
    ds4_gpu_tensor low_q = alloc((uint64_t)rows * HEADS * DIM * sizeof(float)), low_out = alloc(low_q.bytes);
    ds4_gpu_tensor wscratch = alloc(2u * wcount * sizeof(__half)), picked = {NULL, 0u};
    CHECK(cuda_ok(cudaMemcpy(dmap.ptr, map.data(), map.size(), cudaMemcpyHostToDevice), "map"));
    CHECK(cuda_ok(cudaMemcpy(dcache.ptr, cache.data(), (size_t)dcache.bytes, cudaMemcpyHostToDevice), "cache"));
    CHECK(cuda_ok(cudaMemcpy(q.ptr, query.data(), (size_t)q.bytes, cudaMemcpyHostToDevice), "query"));
    if (stride) {
        picked = alloc(selected.size() * sizeof(uint32_t));
        CHECK(cuda_ok(cudaMemcpy(picked.ptr, selected.data(), (size_t)picked.bytes, cudaMemcpyHostToDevice), "sel"));
    }
    CHECK(ds4_gpu_glm53_flash(&out, &q, &low_q, &low_out, &wscratch, &dcache, stride ? &picked : NULL, stride,
        dmap.ptr, map.size(), 0u, kraw.size(), rows, pos0, cap, HEADS, DIM, HEAD_DIM) == 1);
    std::vector<float> output(query.size());
    CHECK(cuda_ok(cudaMemcpy(output.data(), out.ptr, (size_t)out.bytes, cudaMemcpyDeviceToHost), "out"));
    double worst = 0.0, scale = 0.0;
    uint64_t n = 0;
    const uint32_t sample[] = {0u, rows / 3u, rows - 1u};
    for (uint32_t t : sample) {
        const uint32_t visible = pos0 + t + 1u, count = stride ? stride : visible;
        for (uint32_t h = 0u; h < HEADS; h += 13u) {
            std::vector<double> qlow(DIM, 0.0), low(DIM, 0.0), scores(count);
            for (uint32_t j = 0u; j < DIM; j++) {
                for (uint32_t d = 0u; d < HEAD_DIM; d++) {
                    qlow[j] += (double)query[((uint64_t)t * HEADS + h) * HEAD_DIM + d] *
                        (double)kw[((uint64_t)h * DIM + j) * HEAD_DIM + d];
                }
            }
            double maximum = -INFINITY, denominator = 0.0;
            for (uint32_t s = 0u; s < count; s++) {
                const uint32_t r = stride ? selected[(uint64_t)t * stride + s] : s;
                double dot = 0.0;
                if (r < visible) {
                    for (uint32_t j = 0u; j < DIM; j++) { dot += qlow[j] * (double)__half2float(cache[(uint64_t)r * DIM + j]); }
                }
                scores[s] = r < visible ? dot / 16.0 : -INFINITY;
                maximum = fmax(maximum, scores[s]);
            }
            for (uint32_t s = 0u; s < count; s++) { scores[s] = exp(scores[s] - maximum); denominator += scores[s]; }
            for (uint32_t s = 0u; s < count; s++) {
                const uint32_t r = stride ? selected[(uint64_t)t * stride + s] : s;
                if (r >= visible) { continue; }
                for (uint32_t j = 0u; j < DIM; j++) { low[j] += scores[s] / denominator * (double)__half2float(cache[(uint64_t)r * DIM + j]); }
            }
            for (uint32_t d = 0u; d < HEAD_DIM; d++) {
                double expected = 0.0;
                for (uint32_t j = 0u; j < DIM; j++) { expected += low[j] * (double)vw[((uint64_t)h * HEAD_DIM + d) * DIM + j]; }
                const float actual = output[((uint64_t)t * HEADS + h) * HEAD_DIM + d];
                CHECK(isfinite(actual));
                worst = fmax(worst, fabs((double)actual - expected));
                scale += expected * expected;
                n++;
            }
        }
    }
    const double rms = sqrt(scale / (double)n);
    // F16 weights, query, absorbed query, P and latents: a few 1e-3 of the scale.
    CHECK(worst < 1.0e-2 * rms);
    CHECK(ds4_gpu_glm53_flash(&out, &q, &low_q, &low_out, &wscratch, &dcache, stride ? &picked : NULL, stride,
        dmap.ptr, map.size(), 0u, kraw.size(), GLM53_GEMM_MIN_ROWS - 1u, pos0, cap, HEADS, DIM, HEAD_DIM) == 0);
    cudaFree(picked.ptr); cudaFree(wscratch.ptr); cudaFree(low_out.ptr); cudaFree(low_q.ptr);
    cudaFree(out.ptr); cudaFree(q.ptr); cudaFree(dcache.ptr); cudaFree(dmap.ptr);
    printf("GLM absorbed chain: pos0=%u rows=%u selected=%u output_rms=%.4g max_abs=%.3g (%.2e of rms) PASS\n",
        pos0, rows, stride, rms, worst, worst / rms);
}

// Rank-128 Q8_0 projection through the F32-widened SGEMM vs an FP64 product.
static void check_small_k(uint32_t rows) {
    const uint32_t in = 128u, out_dim = 8192u;
    std::vector<uint8_t> raw;
    std::vector<float> w;
    make_q8(raw, w, (uint64_t)in * out_dim, 5u);
    std::vector<float> x((uint64_t)rows * in);
    for (uint64_t i = 0; i < x.size(); i++) { x[i] = sinf((float)(i % 6007u) * 0.021f) * 1.3f; }
    ds4_gpu_tensor dmap = alloc(raw.size()), dx = alloc(x.size() * sizeof(float));
    ds4_gpu_tensor dout = alloc((uint64_t)rows * out_dim * sizeof(float));
    ds4_gpu_tensor scratch = alloc((uint64_t)in * out_dim * sizeof(float));
    CHECK(cuda_ok(cudaMemcpy(dmap.ptr, raw.data(), raw.size(), cudaMemcpyHostToDevice), "map"));
    CHECK(cuda_ok(cudaMemcpy(dx.ptr, x.data(), (size_t)dx.bytes, cudaMemcpyHostToDevice), "x"));
    CHECK(ds4_gpu_glm53_q8_small_k(&dout, &dx, &scratch, dmap.ptr, raw.size(), 0u, in, out_dim, rows) == 1);
    std::vector<float> got((uint64_t)rows * out_dim);
    CHECK(cuda_ok(cudaMemcpy(got.data(), dout.ptr, (size_t)dout.bytes, cudaMemcpyDeviceToHost), "out"));
    double worst = 0.0, scale = 0.0;
    for (uint32_t r = 0u; r < rows; r++) {
        for (uint32_t o = 0u; o < out_dim; o++) {
            double expected = 0.0;
            for (uint32_t k = 0u; k < in; k++) { expected += (double)w[(uint64_t)o * in + k] * (double)x[(uint64_t)r * in + k]; }
            worst = fmax(worst, fabs((double)got[(uint64_t)r * out_dim + o] - expected));
            scale += expected * expected;
        }
    }
    const double rms = sqrt(scale / ((double)rows * out_dim));
    CHECK(worst < 5.0e-3 * rms);  // TF32 operand rounding
    CHECK(ds4_gpu_glm53_q8_small_k(&dout, &dx, &scratch, dmap.ptr, raw.size(), 0u, in, out_dim,
        GLM53_SMALL_K_MIN_ROWS - 1u) == 0);
    cudaFree(scratch.ptr); cudaFree(dout.ptr); cudaFree(dx.ptr); cudaFree(dmap.ptr);
    printf("GLM small-K: rows=%u output_rms=%.4g max_abs=%.3g (%.2e of rms) PASS\n", rows, rms, worst, worst / rms);
}

// Fused pool scores vs the production SIMT kernel, including invisible pools.
static void check_index(uint32_t pos0, uint32_t rows) {
    const uint32_t heads = 32u, pools = (pos0 + rows) / DS4_GLM53_POOL_SIZE;
    std::vector<float> q((uint64_t)rows * heads * DS4_GLM53_POOL_DIM), w((uint64_t)rows * heads);
    std::vector<__half> keys((uint64_t)pools * DS4_GLM53_POOL_DIM);
    for (uint64_t i = 0; i < q.size(); i++) { q[i] = sinf((float)(i % 8191u) * 0.037f) * 1.5f; }
    for (uint64_t i = 0; i < w.size(); i++) { w[i] = cosf((float)(i % 97u) * 0.41f); }
    for (uint64_t i = 0; i < keys.size(); i++) { keys[i] = __float2half_rn(sinf((float)(i % 4093u) * 0.071f)); }
    ds4_gpu_tensor dq = alloc(q.size() * 4), dw = alloc(w.size() * 4), dk = alloc(keys.size() * 2);
    ds4_gpu_tensor ref = alloc((uint64_t)rows * pools * 4), got = alloc(ref.bytes);
    CHECK(cuda_ok(cudaMemcpy(dq.ptr, q.data(), (size_t)dq.bytes, cudaMemcpyHostToDevice), "q"));
    CHECK(cuda_ok(cudaMemcpy(dw.ptr, w.data(), (size_t)dw.bytes, cudaMemcpyHostToDevice), "w"));
    CHECK(cuda_ok(cudaMemcpy(dk.ptr, keys.data(), (size_t)dk.bytes, cudaMemcpyHostToDevice), "k"));
    CHECK(ds4_gpu_glm53_pool_score(&ref, &dq, &dw, &dk, pools, rows, pos0, heads, 1.0f / 64.0f));
    CHECK(ds4_gpu_glm53_index_mma(&got, &dq, &dw, &dk, pools, rows, pos0, heads) == 1);
    std::vector<float> a((uint64_t)rows * pools), b(a.size());
    CHECK(cuda_ok(cudaMemcpy(a.data(), ref.ptr, (size_t)ref.bytes, cudaMemcpyDeviceToHost), "ref"));
    CHECK(cuda_ok(cudaMemcpy(b.data(), got.ptr, (size_t)got.bytes, cudaMemcpyDeviceToHost), "got"));
    double worst = 0.0, scale = 0.0;
    uint64_t finite = 0;
    for (uint64_t i = 0; i < a.size(); i++) {
        CHECK(isinf(a[i]) == isinf(b[i]) && (!isinf(a[i]) || a[i] == b[i]));
        if (isinf(a[i])) { continue; }
        worst = fmax(worst, fabs((double)a[i] - (double)b[i]));
        scale += (double)a[i] * a[i];
        finite++;
    }
    const double rms = sqrt(scale / (double)finite);
    CHECK(worst < 1.0e-2 * rms);  // F16 query rounding
    CHECK(ds4_gpu_glm53_index_mma(&got, &dq, &dw, &dk, pools, GLMVQ_IDX_ROWS - 1u, pos0, heads) == 0);
    cudaFree(got.ptr); cudaFree(ref.ptr); cudaFree(dk.ptr); cudaFree(dw.ptr); cudaFree(dq.ptr);
    printf("GLM index: pos0=%u rows=%u pools=%u score_rms=%.4g max_abs=%.3g (%.2e of rms) PASS\n",
        pos0, rows, pools, rms, worst, worst / rms);
}

int main(void) {
    check_index(2050u, 130u);
    check_index(8000u, 37u);
    CHECK(cublas_ok(cublasCreate(&g_cublas), "create"));
    CHECK(cublas_ok(cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH), "tf32"));
    g_cublas_ready = 1;
    check_small_k(64u);
    check_chain(0u, 300u, false);
    check_chain(2051u, 64u, true);
    check_case(0u, 300u, false);     // dense causal prefix from the first key
    check_case(1400u, 640u, false);  // dense prefix that ends at the selection edge
    check_case(2051u, 96u, true);    // selected keys with masked and unused slots
    check_case(6000u, 2u, true);     // two rows: smallest covered width
    return 0;
}
