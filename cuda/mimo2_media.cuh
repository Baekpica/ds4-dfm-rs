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
    return shared[0];
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
