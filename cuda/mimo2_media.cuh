#pragma once
/* MiMo vision/audio kernels. Vision sinks are an extra zero-value key.
 * Window < 0 is full attention. A positive window is |q-k| > window.
 * causal keeps k <= q. group > 0 attends only inside that block. */
#include <cuda_runtime.h>

__device__ static float m2_block_sum(float value, float *shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int span = blockDim.x >> 1; span > 0; span >>= 1) {
        if (tid < span) { shared[tid] += shared[tid + span]; }
        __syncthreads();
    }
    const float sum = shared[0];
    // Every warp must consume the result before a later reduction reuses shared.
    __syncthreads();
    return sum;
}

__global__ static void mimo2_patch(
        float *out, const float *in, const float *w0, const float *w1,
        int n, int oc, int ic, int kt, int patch) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = ic * kt * patch * patch;
    if (i >= n * oc) { return; }
    const int row = i / oc, col = i % oc;
    const float *src = in + (int64_t)row * width;
    float acc = 0.f;
    for (int c = 0; c < ic; c++) {
        for (int t = 0; t < kt; t++) {
            const float *w = t == 0 ? w0 : w1;
            for (int kh = 0; kh < patch; kh++) {
                for (int kw = 0; kw < patch; kw++) {
                    const int inn = ((c * kt + t) * patch + kh) * patch + kw;
                    const int wn = ((col * ic + c) * patch + kh) * patch + kw;
                    acc += src[inn] * w[wn];
                }
            }
        }
    }
    out[i] = acc;
}

__global__ static void mimo2_rope(
        float *base, const float *cosv, const float *sinv,
        int n, int heads, int hd, int stride, int off) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * heads || hd > 128 || (hd & 1)) { return; }
    const int row = idx / heads, head = idx % heads;
    float *v = base + (int64_t)row * stride + off + head * hd;
    float tmp[128];
    for (int d = 0; d < hd; d++) { tmp[d] = v[d]; }
    const int half = hd / 2;
    const float *cos = cosv + (int64_t)row * hd;
    const float *sin = sinv + (int64_t)row * hd;
    for (int d = 0; d < hd; d++) {
        const float rot = d < half ? -tmp[d + half] : tmp[d - half];
        v[d] = tmp[d] * cos[d] + rot * sin[d];
    }
}

__device__ static bool m2_allow(int q, int k, int window, int causal, int group) {
    if (group > 0 && q / group != k / group) { return false; }
    if (causal && k > q) { return false; }
    if (window >= 0 && abs(q - k) > window) { return false; }
    return true;
}

__global__ static void mimo2_attn(
        float *out, const float *q, const float *k, const float *v, const float *sinks,
        int n, int q_heads, int kv_heads, int hd,
        int q_stride, int k_stride, int v_stride, int q_off, int k_off, int v_off,
        int window, int causal, int group) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * q_heads || q_heads % kv_heads) { return; }
    const int row = idx / q_heads, head = idx % q_heads;
    const int kv = head / (q_heads / kv_heads);
    const float *qq = q + (int64_t)row * q_stride + q_off + head * hd;
    const float scale = rsqrtf((float)hd);
    float maxv = -INFINITY;
    for (int key = 0; key < n; key++) {
        if (!m2_allow(row, key, window, causal, group)) { continue; }
        const float *kk = k + (int64_t)key * k_stride + k_off + kv * hd;
        float dot = 0.f;
        for (int d = 0; d < hd; d++) { dot += qq[d] * kk[d]; }
        maxv = fmaxf(maxv, dot * scale);
    }
    if (sinks) { maxv = fmaxf(maxv, sinks[head]); }
    float *dst = out + (int64_t)row * q_heads * hd + head * hd;
    if (!isfinite(maxv)) {
        for (int d = 0; d < hd; d++) { dst[d] = NAN; }
        return;
    }
    float sum = 0.f;
    for (int key = 0; key < n; key++) {
        if (!m2_allow(row, key, window, causal, group)) { continue; }
        const float *kk = k + (int64_t)key * k_stride + k_off + kv * hd;
        float dot = 0.f;
        for (int d = 0; d < hd; d++) { dot += qq[d] * kk[d]; }
        sum += expf(dot * scale - maxv);
    }
    if (sinks) { sum += expf(sinks[head] - maxv); }
    for (int d = 0; d < hd; d++) { dst[d] = 0.f; }
    if (!(sum > 0.f)) { return; }
    for (int key = 0; key < n; key++) {
        if (!m2_allow(row, key, window, causal, group)) { continue; }
        const float *kk = k + (int64_t)key * k_stride + k_off + kv * hd;
        const float *vv = v + (int64_t)key * v_stride + v_off + kv * hd;
        float dot = 0.f;
        for (int d = 0; d < hd; d++) { dot += qq[d] * kk[d]; }
        const float w = expf(dot * scale - maxv) / sum;
        for (int d = 0; d < hd; d++) { dst[d] += w * vv[d]; }
    }
}

enum {
    M2V_ATTN_HEADS = 32, M2V_ATTN_KV = 8, M2V_ATTN_HD = 64,
    M2V_ATTN_Q = M2V_ATTN_HEADS * M2V_ATTN_HD,
    M2V_ATTN_KVW = M2V_ATTN_KV * M2V_ATTN_HD,
    M2V_ATTN_QKV = M2V_ATTN_Q + 2 * M2V_ATTN_KVW,
    M2V_ATTN_MAX = 8192, M2V_ATTN_THREADS = 128, M2V_ATTN_SCALARS = 2
};

/* Full vision attention: cache each dot once and keep one output value per
 * thread. This removes repeated Q/K reads and output global read/modify/write
 * without changing the scalar dot, denominator, or output accumulation order. */
__global__ static void mimo2_vision_attn(float *out, const float *qkv, int n) {
    const int query = blockIdx.x;
    if (query >= n * M2V_ATTN_HEADS) { return; }
    const int tid = threadIdx.x;
    const int row = query / M2V_ATTN_HEADS, head = query % M2V_ATTN_HEADS;
    const int kv = head / (M2V_ATTN_HEADS / M2V_ATTN_KV);
    extern __shared__ float shared[];
    float *scores = shared;
    float *qq = scores + n;
    float *stats = qq + M2V_ATTN_HD;
    if (tid < M2V_ATTN_HD) {
        qq[tid] = qkv[(int64_t)row * M2V_ATTN_QKV + head * M2V_ATTN_HD + tid];
    }
    __syncthreads();

    const float scale = rsqrtf((float)M2V_ATTN_HD);
    for (int key = tid; key < n; key += blockDim.x) {
        const float *kk = qkv + (int64_t)key * M2V_ATTN_QKV + M2V_ATTN_Q + kv * M2V_ATTN_HD;
        float dot = 0.f;
        for (int d = 0; d < M2V_ATTN_HD; d++) { dot += qq[d] * kk[d]; }
        // Keep the unscaled dot so expf retains the original multiply/subtract.
        scores[key] = dot;
    }
    __syncthreads();
    if (tid == 0) {
        float maxv = -INFINITY;
        for (int key = 0; key < n; key++) { maxv = fmaxf(maxv, scores[key] * scale); }
        stats[0] = maxv;
    }
    __syncthreads();
    const float maxv = stats[0];
    if (!isfinite(maxv)) {
        if (tid < M2V_ATTN_HD) { out[(int64_t)query * M2V_ATTN_HD + tid] = NAN; }
        return;
    }

    for (int key = tid; key < n; key += blockDim.x) {
        scores[key] = expf(scores[key] * scale - maxv);
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.f;
        for (int key = 0; key < n; key++) { sum += scores[key]; }
        stats[1] = sum;
    }
    __syncthreads();
    if (tid >= M2V_ATTN_HD) { return; }

    float acc = 0.f;
    const float sum = stats[1];
    if (sum > 0.f) {
        for (int key = 0; key < n; key++) {
            const float w = scores[key] / sum;
            const float value = qkv[(int64_t)key * M2V_ATTN_QKV +
                                    M2V_ATTN_Q + M2V_ATTN_KVW + kv * M2V_ATTN_HD + tid];
            acc += w * value;
        }
    }
    out[(int64_t)query * M2V_ATTN_HD + tid] = acc;
}

enum { M2V_WINDOW = 64, M2V_WINDOW_KEYS = 2 * M2V_WINDOW + 1 };

/* The vision window is contiguous in the layer's current row/column order.
 * Cache only its valid keys, preserving scalar key order and sink-last math;
 * output dimensions cooperate on V loads and each write their accumulator once. */
__global__ static void mimo2_vision_window(
        float *out, const float *qkv, const float *sinks, int n) {
    const int query = blockIdx.x;
    if (query >= n * M2V_ATTN_HEADS) { return; }
    const int tid = threadIdx.x;
    const int row = query / M2V_ATTN_HEADS, head = query % M2V_ATTN_HEADS;
    const int kv = head / (M2V_ATTN_HEADS / M2V_ATTN_KV);
    const int first = max(0, row - M2V_WINDOW);
    const int keys = min(n, row + M2V_WINDOW + 1) - first;
    extern __shared__ float shared[];
    float *scores = shared;
    float *qq = scores + M2V_WINDOW_KEYS;
    float *stats = qq + M2V_ATTN_HD;
    if (tid < M2V_ATTN_HD) {
        qq[tid] = qkv[(int64_t)row * M2V_ATTN_QKV + head * M2V_ATTN_HD + tid];
    }
    __syncthreads();

    const float scale = rsqrtf((float)M2V_ATTN_HD);
    for (int i = tid; i < keys; i += blockDim.x) {
        const float *kk = qkv + (int64_t)(first + i) * M2V_ATTN_QKV +
                          M2V_ATTN_Q + kv * M2V_ATTN_HD;
        float dot = 0.f;
        for (int d = 0; d < M2V_ATTN_HD; d++) { dot += qq[d] * kk[d]; }
        scores[i] = dot;
    }
    __syncthreads();
    if (tid == 0) {
        float maxv = -INFINITY;
        for (int i = 0; i < keys; i++) { maxv = fmaxf(maxv, scores[i] * scale); }
        stats[0] = fmaxf(maxv, sinks[head]);
    }
    __syncthreads();
    const float maxv = stats[0];
    if (!isfinite(maxv)) {
        if (tid < M2V_ATTN_HD) { out[(int64_t)query * M2V_ATTN_HD + tid] = NAN; }
        return;
    }

    for (int i = tid; i < keys; i += blockDim.x) {
        scores[i] = expf(scores[i] * scale - maxv);
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.f;
        for (int i = 0; i < keys; i++) { sum += scores[i]; }
        stats[1] = sum + expf(sinks[head] - maxv);
    }
    __syncthreads();
    if (tid >= M2V_ATTN_HD) { return; }

    float acc = 0.f;
    const float sum = stats[1];
    if (sum > 0.f) {
        for (int i = 0; i < keys; i++) {
            const float w = scores[i] / sum;
            const float value = qkv[(int64_t)(first + i) * M2V_ATTN_QKV +
                                    M2V_ATTN_Q + M2V_ATTN_KVW + kv * M2V_ATTN_HD + tid];
            acc += w * value;
        }
    }
    out[(int64_t)query * M2V_ATTN_HD + tid] = acc;
}

enum {
    M2A_ATTN_HEADS = 16, M2A_ATTN_HD = 64,
    M2A_ATTN_WIDTH = M2A_ATTN_HEADS * M2A_ATTN_HD,
    M2A_ATTN_MAX = 8192, M2A_ATTN_THREADS = 128, M2A_ATTN_SCALARS = 2
};

/* Full causal codec attention has separate Q/K/V projections. One CTA per
 * query exposes enough parallelism and removes output global read/modify/write.
 * Cache raw dots while preserving scalar dimension and causal-prefix order. */
__global__ static void mimo2_audio_attn(
        float *out, const float *q, const float *k, const float *v, int n) {
    const int query = blockIdx.x;
    if (query >= n * M2A_ATTN_HEADS) { return; }
    const int tid = threadIdx.x;
    const int row = query / M2A_ATTN_HEADS, head = query % M2A_ATTN_HEADS;
    const int keys = row + 1;
    extern __shared__ float shared[];
    float *scores = shared;
    float *qq = scores + n;
    float *stats = qq + M2A_ATTN_HD;
    if (tid < M2A_ATTN_HD) {
        qq[tid] = q[(int64_t)row * M2A_ATTN_WIDTH + head * M2A_ATTN_HD + tid];
    }
    __syncthreads();

    const float scale = rsqrtf((float)M2A_ATTN_HD);
    for (int key = tid; key < keys; key += blockDim.x) {
        const float *kk = k + (int64_t)key * M2A_ATTN_WIDTH + head * M2A_ATTN_HD;
        float dot = 0.f;
        for (int d = 0; d < M2A_ATTN_HD; d++) { dot += qq[d] * kk[d]; }
        scores[key] = dot;
    }
    __syncthreads();
    if (tid == 0) {
        float maxv = -INFINITY;
        for (int key = 0; key < keys; key++) { maxv = fmaxf(maxv, scores[key] * scale); }
        stats[0] = maxv;
    }
    __syncthreads();
    const float maxv = stats[0];
    if (!isfinite(maxv)) {
        if (tid < M2A_ATTN_HD) { out[(int64_t)query * M2A_ATTN_HD + tid] = NAN; }
        return;
    }

    for (int key = tid; key < keys; key += blockDim.x) {
        scores[key] = expf(scores[key] * scale - maxv);
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.f;
        for (int key = 0; key < keys; key++) { sum += scores[key]; }
        stats[1] = sum;
    }
    __syncthreads();
    if (tid >= M2A_ATTN_HD) { return; }

    float acc = 0.f;
    const float sum = stats[1];
    if (sum > 0.f) {
        for (int key = 0; key < keys; key++) {
            const float w = scores[key] / sum;
            const float value = v[(int64_t)key * M2A_ATTN_WIDTH + head * M2A_ATTN_HD + tid];
            acc += w * value;
        }
    }
    out[(int64_t)query * M2A_ATTN_HD + tid] = acc;
}

__global__ static void mimo2_bias(float *x, const float *bias, int n, int dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * dim) { x[i] += bias[i % dim]; }
}

__global__ static void mimo2_swiglu(
        float *out, const float *gate, const float *up,
        const float *gate_b, const float *up_b, int n, int dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * dim) { return; }
    const int col = i % dim;
    const float g = gate[i] + (gate_b ? gate_b[col] : 0.f);
    const float u = up[i] + (up_b ? up_b[col] : 0.f);
    out[i] = (g / (1.f + expf(-g))) * u;
}

__global__ static void mimo2_gelu(float *x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { x[i] = 0.5f * x[i] * (1.f + erff(x[i] * 0.7071067811865476f)); }
}

__global__ static void mimo2_layernorm(
        float *out, const float *x, const float *weight, const float *bias,
        int dim, float eps) {
    extern __shared__ float shared[];
    const float *src = x + (int64_t)blockIdx.x * dim;
    float *dst = out + (int64_t)blockIdx.x * dim;
    float sum = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { sum += src[i]; }
    sum = m2_block_sum(sum, shared);
    const float mean = sum / (float)dim;
    float var = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float d = src[i] - mean;
        var += d * d;
    }
    var = m2_block_sum(var, shared) / (float)dim;
    const float inv = rsqrtf(var + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float y = (src[i] - mean) * inv;
        if (weight) { y *= weight[i]; }
        if (bias) { y += bias[i]; }
        dst[i] = y;
    }
}

__global__ static void mimo2_gather(
        float *out, const float *in, const int *index, int units, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= units * width) { return; }
    const int unit = i / width, col = i % width;
    const int src = index[unit];
    out[i] = (src >= 0 && src < units) ? in[(int64_t)src * width + col] : NAN;
}

__global__ static void mimo2_conv1d(
        float *out, const float *in, const float *weight, const float *bias,
        int n_in, int n_out, int cin, int cout, int k, int stride, int pad) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_out * cout) { return; }
    const int t = i / cout, oc = i % cout;
    float acc = bias ? bias[oc] : 0.f;
    for (int ic = 0; ic < cin; ic++) {
        for (int kk = 0; kk < k; kk++) {
            const int src = t * stride + kk - pad;
            if (src < 0 || src >= n_in) { continue; }
            acc += in[src + (int64_t)n_in * ic] * weight[((oc * cin + ic) * k) + kk];
        }
    }
    out[(int64_t)t * cout + oc] = acc;
}

__global__ static void mimo2_rvq(
        int *ids, float *residual, const float *book, int dim, int bins) {
    const int row = blockIdx.x;
    float best = -INFINITY;
    int best_i = 0;
    float *x = residual + (int64_t)row * dim;
    for (int b = threadIdx.x; b < bins; b += blockDim.x) {
        const float *code = book + (int64_t)b * dim;
        float dot = 0.f, norm = 0.f;
        for (int d = 0; d < dim; d++) {
            dot += x[d] * code[d];
            norm += code[d] * code[d];
        }
        const float score = 2.f * dot - norm;
        if (score > best || (score == best && b < best_i)) {
            best = score;
            best_i = b;
        }
    }
    __shared__ float scores[256];
    __shared__ int picks[256];
    scores[threadIdx.x] = best;
    picks[threadIdx.x] = best_i;
    __syncthreads();
    for (int span = blockDim.x >> 1; span > 0; span >>= 1) {
        if (threadIdx.x < span) {
            const float other = scores[threadIdx.x + span];
            const int other_i = picks[threadIdx.x + span];
            if (other > scores[threadIdx.x] ||
                (other == scores[threadIdx.x] && other_i < picks[threadIdx.x])) {
                scores[threadIdx.x] = other;
                picks[threadIdx.x] = other_i;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x) { return; }
    ids[row] = picks[0];
    const float *code = book + (int64_t)picks[0] * dim;
    for (int d = 0; d < dim; d++) { x[d] -= code[d]; }
}

/* [rows, cols] with cols fastest -> [cols, rows] with rows fastest. */
__global__ static void mimo2_to_ctime(float *out, const float *in, int rows, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) { return; }
    const int row = i / cols, col = i % cols;
    out[row + (int64_t)rows * col] = in[i];
}

__global__ static void mimo2_code_sum(
        float *out, const int *ids, const float *table,
        int n, int dim, int vocab, int channels) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * dim) { return; }
    const int row = i / dim, d = i % dim;
    float sum = 0.f;
    for (int ch = 0; ch < channels; ch++) {
        const int id = ids[(int64_t)row * channels + ch];
        if (id < 0 || id >= vocab) {
            out[i] = NAN;
            return;
        }
        sum += table[d + (int64_t)dim * (id + vocab * ch)];
    }
    out[i] = sum;
}
