/* Device DFlash attention. The host switch stays the numerical reference.
 * Q, Kctx and Knoise are RMSNorm'd and RoPE'd in place. V is not. */
#ifndef MIMO2_DFLASH_ATTN_CUH
#define MIMO2_DFLASH_ATTN_CUH
#include <cuda_runtime.h>
#include <stdlib.h>
#define MIMO2_DFLASH_HOST_FNS
#include "mimo2_dflash_host.h"

static float *m2df_w = nullptr;

static void m2df_release(void) {
    if (!m2df_w) { return; }
    (void)cudaFree(m2df_w);
    m2df_w = nullptr;
}

static int m2df_w_ready(void) {
    if (m2df_w) { return 1; }
    if (cudaMalloc(&m2df_w, (128 + 128 + 64) * sizeof(float)) != cudaSuccess) {
        m2df_w = nullptr;
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

__global__ static void m2df_rms_k(float *rows, const float *weight, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) { return; }
    float *x = rows + (size_t)row * DF_HD;
    double sum = 0.0;
    for (int i = 0; i < DF_HD; i++) {
        const double v = (double)x[i];
        sum += v * v;
    }
    const float scale = 1.0f / sqrtf((float)(sum / DF_HD) + 1e-6f);
    for (int i = 0; i < DF_HD; i++) { x[i] *= scale * weight[i]; }
}

__global__ static void m2df_rope_k(float *rows, int ntok, int heads, unsigned pos0) {
    const int tok = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= ntok) { return; }
    const unsigned pos = pos0 + (unsigned)tok;
    float *base = rows + (size_t)tok * (unsigned)heads * DF_HD;
    for (int h = 0; h < heads; h++) {
        float *x = base + (size_t)h * DF_HD;
        float rot[64];
        for (int i = 0; i < 32; i++) {
            const float freq = (float)pos / powf(10000.0f, (2.0f * (float)i) / 64.0f);
            const float c = cosf(freq), s = sinf(freq);
            const float a = x[i], b = x[32 + i];
            rot[i] = a * c - b * s;
            rot[32 + i] = b * c + a * s;
        }
        for (int i = 0; i < 64; i++) { x[i] = rot[i]; }
    }
}

__device__ static const float *m2df_row(const float *ctx, const float *noise,
                                         unsigned ki, unsigned nctx, unsigned kv_head) {
    if (ki < nctx) { return ctx + ((size_t)ki * DF_KH + kv_head) * DF_HD; }
    return noise + ((size_t)(ki - nctx) * DF_KH + kv_head) * DF_HD;
}

/* A KV head's eight queries share each K/V tile. Warp reductions replace
 * the serial three-pass dot products; the host path bounds rounding error.
 * DFlash attends to the context and all noise rows, including future rows. */
__global__ static void m2df_attn_tile_k(
        float *out, const float *q,
        const float *k_ctx, const float *k_noise,
        const float *v_ctx, const float *v_noise,
        const float *sinks, unsigned q0, unsigned n, unsigned nctx) {
    enum { WARP = 32, TILE = 32, GROUP = DF_QH / DF_KH };
    const unsigned lane = threadIdx.x % WARP;
    const unsigned head = blockIdx.y * GROUP + threadIdx.x / WARP;
    const unsigned qi = blockIdx.x;
    const unsigned q_pos = q0 + qi;
    const unsigned kv_len = nctx + n;
    const float *qq = q + ((size_t)qi * DF_QH + head) * DF_HD;
    float query[DF_HD / WARP], acc[DF_HD / WARP] = {};
    for (unsigned d = 0; d < DF_HD / WARP; d++) {
        query[d] = qq[lane + d * WARP];
    }
    float maximum = sinks[head], denominator = 1.0f;
    __shared__ float sk[TILE * DF_HD], sv[TILE * DF_HD];
    for (unsigned base = 0; base < kv_len; base += TILE) {
        const unsigned count = min((unsigned)TILE, kv_len - base);
        for (unsigned i = threadIdx.x; i < count * (DF_HD / 4); i += blockDim.x) {
            const unsigned row = i / (DF_HD / 4), col = i % (DF_HD / 4) * 4;
            const float *k = m2df_row(k_ctx, k_noise, base + row, nctx, blockIdx.y);
            const float *v = m2df_row(v_ctx, v_noise, base + row, nctx, blockIdx.y);
            *(float4 *)(sk + row * DF_HD + col) = *(const float4 *)(k + col);
            *(float4 *)(sv + row * DF_HD + col) = *(const float4 *)(v + col);
        }
        __syncthreads();
        for (unsigned row = 0; row < count; row++) {
            const unsigned ki = base + row;
            const unsigned kp = ki < nctx ? q0 - nctx + ki : q0 + ki - nctx;
            if (q_pos > kp && q_pos - kp >= DF_WIN) { continue; }
            if (kp > q_pos && kp - q_pos >= DF_WIN) { continue; }
            float dot = 0.0f;
            for (unsigned d = 0; d < DF_HD / WARP; d++) {
                dot += query[d] * sk[row * DF_HD + lane + d * WARP];
            }
            for (unsigned step = WARP / 2; step; step /= 2) {
                dot += __shfl_xor_sync(0xffffffffu, dot, step);
            }
            const float score = dot * rsqrtf((float)DF_HD);
            const float next = fmaxf(maximum, score);
            const float old = expf(maximum - next), weight = expf(score - next);
            denominator = denominator * old + weight;
            for (unsigned d = 0; d < DF_HD / WARP; d++) {
                acc[d] = acc[d] * old + weight * sv[row * DF_HD + lane + d * WARP];
            }
            maximum = next;
        }
        __syncthreads();
    }
    const float value_scale = 0.612f;
    float *dst = out + ((size_t)qi * DF_QH + head) * DF_HD;
    for (unsigned d = 0; d < DF_HD / WARP; d++) {
        dst[lane + d * WARP] = acc[d] / denominator * value_scale;
    }
}

static int m2df_cpu_on(void) {
    const char *env = getenv("DS4_MIMO2_DFLASH_CPU");
    return env && env[0] == '1' && env[1] == '\0';
}

/* 1: host loops. 2: device. 0: failure. */
static int m2df_attn_launch(
        float *attn, float *q, float *k_ctx, float *k_noise,
        float *v_ctx, float *v_noise,
        const float *q_weight, const float *k_weight, const float *sinks,
        unsigned q0, unsigned n, unsigned ctx, cudaStream_t stream) {
    const unsigned kv_len = ctx + n;
    if (m2df_cpu_on()) {
        float *hq = (float *)malloc((size_t)n * DF_Q * sizeof(float));
        float *hk = (float *)malloc((size_t)kv_len * DF_KV * sizeof(float));
        float *hv = (float *)malloc((size_t)kv_len * DF_KV * sizeof(float));
        float *ho = (float *)malloc((size_t)n * DF_Q * sizeof(float));
        unsigned *k_pos = (unsigned *)malloc((size_t)kv_len * sizeof(unsigned));
        int ok = hq && hk && hv && ho && k_pos;
        if (ok && cudaStreamSynchronize(stream) != cudaSuccess) { ok = 0; }
        if (ok && (cudaMemcpy(hq, q, (size_t)n * DF_Q * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(hk, k_ctx, (size_t)ctx * DF_KV * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(hk + (size_t)ctx * DF_KV, k_noise, (size_t)n * DF_KV * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(hv, v_ctx, (size_t)ctx * DF_KV * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(hv + (size_t)ctx * DF_KV, v_noise, (size_t)n * DF_KV * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)) {
            ok = 0;
        }
        if (ok) {
            unsigned row, i;
            for (row = 0; row < n * DF_QH; row++) { df_rms(hq + row * DF_HD, q_weight, DF_HD); }
            for (row = 0; row < kv_len * DF_KH; row++) { df_rms(hk + row * DF_HD, k_weight, DF_HD); }
            for (i = 0; i < n; i++) { df_rope(hq + (size_t)i * DF_Q, DF_QH, q0 + i); }
            for (i = 0; i < ctx; i++) { df_rope(hk + (size_t)i * DF_KV, DF_KH, q0 - ctx + i); }
            for (i = 0; i < n; i++) { df_rope(hk + ((size_t)ctx + i) * DF_KV, DF_KH, q0 + i); }
            for (i = 0; i < ctx; i++) { k_pos[i] = q0 - ctx + i; }
            for (i = 0; i < n; i++) { k_pos[ctx + i] = q0 + i; }
            df_attn(ho, hq, hk, hv, sinks, k_pos, q0, n, kv_len);
            if (cudaMemcpy(attn, ho, (size_t)n * DF_Q * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
                ok = 0;
            }
        }
        free(hq); free(hk); free(hv); free(ho); free(k_pos);
        return ok ? 1 : 0;
    }
    if (!m2df_w_ready()) { return 0; }
    if (cudaMemcpyAsync(m2df_w, q_weight, 128 * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(m2df_w + 128, k_weight, 128 * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(m2df_w + 256, sinks, 64 * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return 0;
    }
    m2df_rms_k<<<(n * DF_QH + 127) / 128, 128, 0, stream>>>(q, m2df_w, (int)(n * DF_QH));
    m2df_rms_k<<<(ctx * DF_KH + 127) / 128, 128, 0, stream>>>(k_ctx, m2df_w + 128, (int)(ctx * DF_KH));
    m2df_rms_k<<<(n * DF_KH + 127) / 128, 128, 0, stream>>>(k_noise, m2df_w + 128, (int)(n * DF_KH));
    m2df_rope_k<<<(n + 127) / 128, 128, 0, stream>>>(q, (int)n, DF_QH, q0);
    m2df_rope_k<<<(ctx + 127) / 128, 128, 0, stream>>>(k_ctx, (int)ctx, DF_KH, q0 - ctx);
    m2df_rope_k<<<(n + 127) / 128, 128, 0, stream>>>(k_noise, (int)n, DF_KH, q0);
    m2df_attn_tile_k<<<dim3(n, DF_KH), 256, 0, stream>>>(
        attn, q, k_ctx, k_noise, v_ctx, v_noise, m2df_w + 256, q0, n, ctx);
    return cudaGetLastError() == cudaSuccess ? 2 : 0;
}
#endif
