/* Prism Bonsai (qwen35): the dense-attention and gated delta-net kernels the
 * family's CUDA graph needs.  Included by ds4_cuda.cu, after the qwen35 fold
 * header, so the residency, stream and temporary helpers have the same
 * lifetime as the other CUDA paths.
 *
 * Ported from the sibling tree's ds4_qwen4_cuda.cuh (/data/ds4), where one
 * kernel set serves both the qwen4 family and Bonsai.  This tree has no
 * qwen4 dense path: its qwen4exp attention is the indexer-bound QSA (its
 * entries require a selected-block list) and its gated delta-net is conv +
 * recurrent with no chunked scan, so neither can serve this family.  The
 * sibling's kernels are therefore carried here under their own namespace and
 * under the entry names the graph and the tests call.
 *
 * The two families differ in exactly two places, both parameters here:
 *   - the linear layer's output gate is silu for Bonsai, sigmoid for qwen4exp
 *     (the gate_silu selector of gdn_out);
 *   - Bonsai has no sparse indexer, so its attn_prep passes no indexer slot.
 *
 * Attention selection follows the sibling: a call with no partial buffer and
 * T >= 32 takes the token-tile MMA kernel (attention_group); everything else
 * takes the row-exact kernel, split-K when a partial buffer is supplied. */

#include <stdint.h>
#include <math.h>

namespace qwen35_attn {

__device__ __forceinline__ float sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
    return x;
}

__device__ __forceinline__ float sigmoid(float x) {
    const float e = expf(-fabsf(x));
    return x >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

__device__ __forceinline__ float silu(float x) { return x * sigmoid(x); }

__device__ __forceinline__ float softplus(float x) {
    return x > 20 ? x : x < -20 ? expf(x) : log1pf(expf(x));
}

/* Rotary table of the n_rot/2 rotated pairs.  Bonsai is trained at its
 * declared context with no rope scaling (config_validate_qwen35_model asserts
 * a plain table), so the frequencies are always the plain base^(-2i/n_rot)
 * with unit scale; the struct keeps the kernel signature the sibling's
 * shared with the qwen4 path. */
struct rope_args { float freq[32], scale; unsigned nrot; };

static rope_args rope(unsigned nrot, float base) {
    rope_args r = {};
    r.nrot = nrot;
    r.scale = 1;
    for (unsigned i = 0; i < nrot / 2; i++) {
        r.freq[i] = powf(base, -2.0f * i / nrot);
    }
    return r;
}

__device__ void apply_rope(float *row, const uint32_t *pos, rope_args r) {
    const unsigned i = threadIdx.x;
    if (i < r.nrot / 2) {
        const float theta = (float)pos[i % 3] * r.freq[i];
        const float c = cosf(theta) * r.scale, s = sinf(theta) * r.scale;
        const float a = row[i], b = row[i + r.nrot / 2];
        row[i] = a * c - b * s;
        row[i + r.nrot / 2] = a * s + b * c;
    }
    __syncwarp();
}

/* ldmatrix with the shared pointer passed as a generic address, the form the
 * sibling tree's build of this kernel uses, measured to be required: with this
 * tree's tt_* helpers instead (a uint32_t __cvta_generic_to_shared address and
 * an "r" constraint) the token-tile cases of tests/test_qwen35_cuda fault with
 * "an illegal memory access" and compute-sanitizer reports an invalid 16-byte
 * shared read at this kernel's kv load, while the generic form reproduces the
 * sibling's published numbers digit for digit.  Both forms denote the same
 * shared offset; only this one is usable here. */
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void *p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
#else
    (void)p;
    r[0] = r[1] = r[2] = r[3] = 0;
#endif
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void *p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "l"(p));
#else
    (void)p;
    r[0] = r[1] = 0;
#endif
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2], const void *p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "l"(p));
#else
    (void)p;
    r[0] = r[1] = 0;
#endif
}

/* Per-head q and k norm with the partial rope, and the v store.  One CUDA
 * block per (head slot, token); the indexer slots are absent for this family,
 * so isq/isk are the only two cases. */
__global__ void attn_prep(float *qout, float *gate, __half *kc, __half *vc,
        float *iqout, float *ikc, const float *qg, const float *kp, const float *vp,
        const float *iq, const float *ik, const uint32_t *pos3,
        const float *gq, const float *gk, const float *giq,
        unsigned H, unsigned Hkv, unsigned D, unsigned Hi, unsigned Di,
        unsigned pos0, float eps, rope_args rp) {
    const unsigned slot = blockIdx.x, t = blockIdx.y, pos = pos0 + t, lane = threadIdx.x;
    __shared__ float row[256];
    if (slot == H + Hkv + Hi) {
        for (unsigned i = lane; i < Di; i += 32) ikc[(uint64_t)pos * Di + i] = ik[(uint64_t)t * Di + i];
        return;
    }
    const bool isq = slot < H, isk = !isq && slot < H + Hkv;
    const unsigned h = isq ? slot : isk ? slot - H : slot - H - Hkv;
    const unsigned dim = isq || isk ? D : Di;
    const float *src = isq ? qg + ((uint64_t)t * H + h) * 2 * D :
                      isk ? kp + ((uint64_t)t * Hkv + h) * D : iq + ((uint64_t)t * Hi + h) * Di;
    const float *gamma = isq ? gq : isk ? gk : giq;
    const unsigned npt = dim / 32;
    float ss = 0;
    for (unsigned i = 0; i < npt; i++) { const float v = src[lane * npt + i]; ss += v * v; }
    const float inv = rsqrtf(sum(ss) / dim + eps);
    for (unsigned i = lane; i < dim; i += 32) row[i] = src[i] * inv * gamma[i];
    __syncwarp();
    apply_rope(row, pos3 + (uint64_t)pos * 4, rp);
    for (unsigned i = lane; i < dim; i += 32) {
        if (isq) {
            qout[((uint64_t)t * H + h) * D + i] = row[i];
            gate[((uint64_t)t * H + h) * D + i] = src[D + i];
        } else if (isk) {
            kc[((uint64_t)pos * Hkv + h) * D + i] = __float2half_rn(row[i]);
            vc[((uint64_t)pos * Hkv + h) * D + i] = __float2half_rn(vp[((uint64_t)t * Hkv + h) * D + i]);
        } else iqout[((uint64_t)t * Hi + h) * Di + i] = row[i];
    }
}

/* Row-exact attention: one warp owns a query head, one (split, key) range each.
 * With splits == 1 the warp writes the gated result; otherwise it writes the
 * online-softmax (m, denom, accumulator) triple that attn_merge reduces. */
template<unsigned D>
__global__ void attention(float *out, float *partial, const float *q, const float *gate,
        const __half *kc, const __half *vc, const int *sel, const unsigned *counts,
        unsigned H, unsigned Hkv, unsigned pos0, unsigned stride, bool sparse,
        unsigned splits, unsigned per, float scale) {
    const unsigned h = blockIdx.x * 4 + threadIdx.x / 32, t = blockIdx.y, split = blockIdx.z;
    if (h >= H) return;
    const unsigned lane = threadIdx.x & 31, kh = h / (H / Hkv), n = sparse ? counts[t] : pos0 + t + 1;
    float qv[D / 32], acc[D / 32] = {}, m = -3e38f, denom = 0;
    for (unsigned i = 0; i < D / 32; i++) qv[i] = q[((uint64_t)t * H + h) * D + lane + 32 * i] * scale;
    for (unsigned j = split * per; j < min(n, (split + 1) * per); j++) {
        const unsigned p = sparse ? (unsigned)sel[(uint64_t)t * stride + j] : j;
        if (p > pos0 + t) continue;
        float score = 0;
        for (unsigned i = 0; i < D / 32; i++) score += qv[i] * __half2float(kc[((uint64_t)p * Hkv + kh) * D + lane + 32 * i]);
        score = sum(score);
        const float nm = fmaxf(m, score), correction = expf(m - nm), w = expf(score - nm);
        denom = denom * correction + w;
        for (unsigned i = 0; i < D / 32; i++) acc[i] = acc[i] * correction + w * __half2float(vc[((uint64_t)p * Hkv + kh) * D + lane + 32 * i]);
        m = nm;
    }
    if (splits == 1) {
        for (unsigned i = 0; i < D / 32; i++) {
            const uint64_t p = ((uint64_t)t * H + h) * D + lane + 32 * i;
            out[p] = (denom > 0 ? acc[i] / denom : 0) * sigmoid(gate[p]);
        }
    } else {
        float *dst = partial + (((uint64_t)t * H + h) * splits + split) * (D + 2);
        if (!lane) { dst[0] = m; dst[1] = denom; }
        for (unsigned i = 0; i < D / 32; i++) dst[2 + lane + 32 * i] = acc[i];
    }
}

__global__ void attn_merge(float *out, const float *partial, const float *gate,
        unsigned H, unsigned D, unsigned splits) {
    const unsigned h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    if (d >= D) return;
    const float *p = partial + ((uint64_t)t * H + h) * splits * (D + 2);
    float m = -3e38f, denom = 0, acc = 0;
    for (unsigned s = 0; s < splits; s++) m = fmaxf(m, p[s * (D + 2)]);
    for (unsigned s = 0; s < splits; s++) {
        const float *row = p + s * (D + 2);
        const float w = row[1] > 0 ? expf(row[0] - m) : 0;
        denom += row[1] * w; acc += row[2 + d] * w;
    }
    const uint64_t i = ((uint64_t)t * H + h) * D + d;
    out[i] = (denom > 0 ? acc / denom : 0) * sigmoid(gate[i]);
}

/* The heads in a KV group share the same selected keys. Keep their output
 * accumulators in registers and use tensor cores for both products. Scaled
 * residual components retain the fine part of the FP32 queries/probabilities;
 * K and V are already half, so neither requires further rounding. */
__global__ void attention_group(float *out, const float *q, const float *gate,
        const __half *kc, const __half *vc, const int *sel, const unsigned *counts,
        unsigned H, unsigned Hkv, unsigned pos0, unsigned stride, bool sparse, float scale) {
#if __CUDA_ARCH__ >= 800
    const unsigned D = 256, tid = threadIdx.x, lane = tid&31, warp = tid/32;
    const unsigned t = blockIdx.y, kh = blockIdx.x, group = H/Hkv;
    const unsigned qr = tid/16, col = tid%16, h = kh*group+qr;
    const unsigned n = sparse ? counts[t] : pos0+t+1;
    __shared__ __align__(32) __half qh[16][264], ql[16][264], kv[32][264];
    union Scores { float part[2][16][32]; __half prob[2][16][40]; };
    __shared__ __align__(32) Scores scores;
    __shared__ float qs[16], max_score[16], denom[16], correction[16];
    __shared__ unsigned positions[32];
    float mx = 0;
    for (unsigned d = col; d < D; d += 16)
        if (qr < group) mx = fmaxf(mx,fabsf(q[((uint64_t)t*H+h)*D+d]*scale));
    for (unsigned off = 8; off; off /= 2) mx = fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,off,16));
    const int exp = mx > 0 ? max(-120,min(120,(int)((__float_as_uint(mx)>>23)&255)-127)) : 0;
    const float inv = ldexpf(1,-exp);
    if (!col) { qs[qr] = ldexpf(1,exp); max_score[qr] = -3e38f; denom[qr] = 0; }
    for (unsigned d = col; d < D; d += 16) {
        const float v = qr < group ? q[((uint64_t)t*H+h)*D+d]*scale*inv : 0;
        qh[qr][d] = __float2half_rn(v);
        ql[qr][d] = __float2half_rn((v-__half2float(qh[qr][d]))*4096);
    }
    float result[4][4] = {};
    __syncthreads();
    for (unsigned j0 = 0; j0 < n; j0 += 32) {
        if (tid < 32) {
            const unsigned j = j0+tid;
            const unsigned p = j < n ? (sparse ? (unsigned)sel[(uint64_t)t*stride+j] : j) : UINT_MAX;
            positions[tid] = p <= pos0+t ? p : UINT_MAX;
        }
        __syncthreads();
        for (unsigned i = tid*8; i < 32*D; i += 256*8) {
            const unsigned r = i/D, d = i%D, p = positions[r];
            tt_cp_async_16B(&kv[r][d],kc+((uint64_t)(p == UINT_MAX ? 0 : p)*Hkv+kh)*D+d,p != UINT_MAX);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
        __syncthreads();
        float hi[4] = {}, lo[4] = {};
        const unsigned split = warp/4, key0 = (warp%4)*8;
        for (unsigned k = split*128; k < (split+1)*128; k += 16) {
            uint32_t ah[4], al[4], b[2];
            ldmatrix_x4(ah,&qh[lane%16][k+(lane/16)*8]);
            ldmatrix_x4(al,&ql[lane%16][k+(lane/16)*8]);
            ldmatrix_x2(b,&kv[key0+lane%8][k+((lane%16)/8)*8]);
            tt_mma_m16n8k16_f16_f32(hi,ah,b);
            tt_mma_m16n8k16_f16_f32(lo,al,b);
        }
        #pragma unroll
        for (unsigned i = 0; i < 4; i++)
            scores.part[split][tt_mma_c_i(lane,i)][key0+tt_mma_c_j(lane,i)] = hi[i]+lo[i]*0x1p-12f;
        __syncthreads();
        float prob[2], peak = max_score[qr];
        #pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            const unsigned key = col+i*16;
            prob[i] = positions[key] != UINT_MAX ?
                (scores.part[0][qr][key]+scores.part[1][qr][key])*qs[qr] : -3e38f;
            peak = fmaxf(peak,prob[i]);
        }
        for (unsigned off = 8; off; off /= 2) peak = fmaxf(peak,__shfl_xor_sync(0xffffffff,peak,off,16));
        const float old = expf(max_score[qr]-peak);
        float total = 0;
        #pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            prob[i] = positions[col+i*16] != UINT_MAX ? expf(prob[i]-peak) : 0;
            total += prob[i];
        }
        for (unsigned off = 8; off; off /= 2) total += __shfl_xor_sync(0xffffffff,total,off,16);
        __syncthreads();
        if (!col) {
            correction[qr] = old;
            max_score[qr] = peak;
            denom[qr] = denom[qr]*old+total;
        }
        #pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            const __half p = __float2half_rn(prob[i]);
            scores.prob[0][qr][col+i*16] = p;
            scores.prob[1][qr][col+i*16] = __float2half_rn((prob[i]-__half2float(p))*4096);
        }
        for (unsigned i = tid*8; i < 32*D; i += 256*8) {
            const unsigned r = i/D, d = i%D, p = positions[r];
            tt_cp_async_16B(&kv[r][d],vc+((uint64_t)(p == UINT_MAX ? 0 : p)*Hkv+kh)*D+d,p != UINT_MAX);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
        __syncthreads();
        #pragma unroll
        for (unsigned tile = 0; tile < 4; tile++) {
            float hi[4] = {}, lo[4] = {};
            #pragma unroll
            for (unsigned k = 0; k < 32; k += 16) {
                uint32_t ah[4], al[4], b[2];
                ldmatrix_x4(ah,&scores.prob[0][lane%16][k+(lane/16)*8]);
                ldmatrix_x4(al,&scores.prob[1][lane%16][k+(lane/16)*8]);
                ldmatrix_x2_trans(b,&kv[k+lane%16][warp*32+tile*8]);
                tt_mma_m16n8k16_f16_f32(hi,ah,b);
                tt_mma_m16n8k16_f16_f32(lo,al,b);
            }
            #pragma unroll
            for (unsigned i = 0; i < 4; i++) result[tile][i] =
                result[tile][i]*correction[tt_mma_c_i(lane,i)]+(hi[i]+lo[i]*0x1p-12f);
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned tile = 0; tile < 4; tile++) {
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned r = tt_mma_c_i(lane,i), d = warp*32+tile*8+tt_mma_c_j(lane,i);
            if (r < group) {
                const uint64_t dst = ((uint64_t)t*H+kh*group+r)*D+d;
                out[dst] = (denom[r] > 0 ? result[tile][i]/denom[r] : 0)*sigmoid(gate[dst]);
            }
        }
    }
#endif
}

/* Depthwise causal convolution over T rows with the last K-1 columns of
 * history, optional silu activation, and optional history snapshots (used by
 * the checkpoint paths; the Bonsai graph passes neither). */
__global__ void conv(float *x, float *history, const float *w, unsigned T,
                     unsigned C, unsigned K, bool activate,
                     float *snap, unsigned snap_t, float *snap2, unsigned snap2_t) {
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float win[3], taps[4];
    for (unsigned i = 0; i < K - 1; i++) win[i] = history[(uint64_t)i * C + c];
    for (unsigned i = 0; i < K; i++) taps[i] = w[c * K + i];
    for (unsigned t = 0; t < T; t++) {
        const uint64_t pos = (uint64_t)t * C + c;
        const float raw = x[pos];
        float v = taps[K - 1] * raw;
        for (unsigned i = 0; i < K - 1; i++) v += taps[i] * win[i];
        for (unsigned i = 0; i + 2 < K; i++) win[i] = win[i + 1];
        win[K - 2] = raw;
        x[pos] = activate ? silu(v) : v;
        if (snap && t == snap_t) for (unsigned i = 0; i < K - 1; i++) snap[(uint64_t)i * C + c] = win[i];
        if (snap2 && t == snap2_t) for (unsigned i = 0; i < K - 1; i++) snap2[(uint64_t)i * C + c] = win[i];
    }
    for (unsigned i = 0; i < K - 1; i++) history[(uint64_t)i * C + c] = win[i];
}

/* Per-head L2 norm of q and k, and the per-head decay/beta of the gated
 * delta-net recurrence: a = exp(A * softplus(a_raw + bias)), b = sigmoid(b_raw). */
__global__ void gdn_prep(float *qkv, float *a, float *b, const float *A, const float *bias,
                         unsigned Hk, unsigned Hv, unsigned D) {
    const unsigned h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const unsigned C = (2 * Hk + Hv) * D, npt = D / 32;
    float *q = qkv + (uint64_t)t * C + h * D + lane * npt, *k = q + Hk * D;
    float qs = 0, ks = 0;
    for (unsigned i = 0; i < npt; i++) { qs += q[i] * q[i]; ks += k[i] * k[i]; }
    qs = rsqrtf(sum(qs) + 1e-6f) * rsqrtf((float)D);
    ks = rsqrtf(sum(ks) + 1e-6f);
    for (unsigned i = 0; i < npt; i++) { q[i] *= qs; k[i] *= ks; }
    if (!h) for (unsigned j = lane; j < Hv; j += 32) {
        const uint64_t p = (uint64_t)t * Hv + j;
        a[p] = expf(A[j] * softplus(a[p] + bias[j]));
        b[p] = sigmoid(b[p]);
    }
}

/* One warp owns a state row across the whole chunk. In particular, the
 * snapshot slots contain the state after the requested token, not the final
 * row. */
template<unsigned ROWS, unsigned D>
__global__ void gdn_scan(float *out, float *state, const float *qkv,
                         const float *a, const float *b, unsigned T, unsigned Hk,
                         unsigned Hv, float *snap, unsigned st,
                         float *snap2, unsigned st2) {
    const unsigned dv = (blockIdx.x * 4 + threadIdx.x / 32)*ROWS, h = blockIdx.y;
    if (dv >= D) return;
    const unsigned npt = D / 32, k0 = (threadIdx.x & 31) * npt, kh = h % Hk;
    const unsigned C = (2 * Hk + Hv) * D;
    const uint64_t idx = ((uint64_t)h * D + dv) * D + k0;
    float s[ROWS][4];
    #pragma unroll
    for (unsigned r = 0; r < ROWS; r++)
        #pragma unroll
        for (unsigned i = 0; i < npt; i++) s[r][i] = state[idx+(uint64_t)r*D+i];
    for (unsigned t = 0; t < T; t++) {
        const float *q = qkv + (uint64_t)t * C + kh * D + k0, *k = q + Hk * D;
        const float decay = a[(uint64_t)t * Hv + h], beta = b[(uint64_t)t * Hv + h];
        #pragma unroll
        for (unsigned r = 0; r < ROWS; r++) {
            const float v = qkv[(uint64_t)t*C+2*Hk*D+h*D+dv+r];
            float u = 0;
            #pragma unroll
            for (unsigned i = 0; i < npt; i++) { s[r][i] *= decay; u += s[r][i]*k[i]; }
            const float delta = (v-sum(u))*beta;
            float o = 0;
            #pragma unroll
            for (unsigned i = 0; i < npt; i++) { s[r][i] += k[i]*delta; o += s[r][i]*q[i]; }
            o = sum(o);
            if (!(threadIdx.x&31)) out[((uint64_t)t*Hv+h)*D+dv+r] = o;
            if (snap && t == st) for (unsigned i = 0; i < npt; i++) snap[idx+(uint64_t)r*D+i] = s[r][i];
            if (snap2 && t == st2) for (unsigned i = 0; i < npt; i++) snap2[idx+(uint64_t)r*D+i] = s[r][i];
        }
    }
    #pragma unroll
    for (unsigned r = 0; r < ROWS; r++)
        #pragma unroll
        for (unsigned i = 0; i < npt; i++) state[idx+(uint64_t)r*D+i] = s[r][i];
}

/* Gated RMS norm of the linear layer's output: rms(o) * w * gate(z), the gate
 * silu for Bonsai and sigmoid for qwen4exp. */
__global__ void gdn_out(float *o, const float *z, const float *w, unsigned H, unsigned D, float eps,
                        unsigned gate_silu) {
    const unsigned h = blockIdx.x, t = blockIdx.y, npt = D / 32, k0 = threadIdx.x * npt;
    const uint64_t idx = ((uint64_t)t * H + h) * D + k0;
    float ss = 0;
    for (unsigned i = 0; i < npt; i++) ss += o[idx + i] * o[idx + i];
    const float r = rsqrtf(sum(ss) / D + eps);
    for (unsigned i = 0; i < npt; i++) {
        /* qwen4exp gates this norm with sigmoid, the Bonsai trunk with silu. */
        const float gate = gate_silu ? silu(z[idx + i]) : sigmoid(z[idx + i]);
        o[idx + i] = o[idx + i] * r * w[k0 + i] * gate;
    }
}

} // namespace qwen35_attn

/* ---------------------------------------------------------------------------
 * Host entries.  tensor/weight/launched mirror the sibling's file-scope
 * helpers; the resolver and the stream come from ds4_cuda.cu. */

static bool qwen35_attn_tensor(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && bytes <= t->bytes;
}

static const char *qwen35_attn_weight(const void *map, uint64_t size, uint64_t off, uint64_t bytes) {
    if (!map || !bytes || off > size || bytes > size - off) return NULL;
    return cuda_resolve_weight_ptr(map, off, bytes, 0, "Bonsai weights");
}

static int qwen35_attn_launched(const char *what) {
    return cuda_ok(cudaGetLastError(), what);
}

extern "C" int ds4_gpu_qwen4_conv_stream_tensor(ds4_gpu_tensor *x, ds4_gpu_tensor *history,
        const void *map, uint64_t size, uint64_t offset, uint32_t T, uint32_t C, uint32_t K, bool activate) {
    using namespace qwen35_attn;
    if (!T || !C || K < 2 || K > 4 || !qwen35_attn_tensor(x, (uint64_t)T * C * 4) ||
        !qwen35_attn_tensor(history, (uint64_t)(K - 1) * C * 4)) return 0;
    const char *w = qwen35_attn_weight(map, size, offset, (uint64_t)C * K * 4);
    if (!w) return 0;
    conv<<<(C + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)x->ptr, (float *)history->ptr,
        (const float *)w, T, C, K, activate, NULL, UINT_MAX, NULL, UINT_MAX);
    return qwen35_attn_launched("Bonsai conv");
}

extern "C" int ds4_gpu_qwen4_gdn_prep_tensor(ds4_gpu_tensor *qkv, ds4_gpu_tensor *a, ds4_gpu_tensor *b,
        const void *map, uint64_t size, uint64_t ao, uint64_t bo, uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D) {
    using namespace qwen35_attn;
    if (!T || !Hk || !Hv || D < 32 || D > 128 || D % 32 ||
        !qwen35_attn_tensor(qkv, (uint64_t)T * (2 * Hk + Hv) * D * 4) ||
        !qwen35_attn_tensor(a, (uint64_t)T * Hv * 4) || !qwen35_attn_tensor(b, (uint64_t)T * Hv * 4)) return 0;
    const char *A = qwen35_attn_weight(map, size, ao, (uint64_t)Hv * 4);
    const char *bias = qwen35_attn_weight(map, size, bo, (uint64_t)Hv * 4);
    if (!A || !bias) return 0;
    gdn_prep<<<dim3(Hk, T), 32, 0, cuda_decode_stream()>>>((float *)qkv->ptr,
        (float *)a->ptr, (float *)b->ptr, (const float *)A, (const float *)bias, Hk, Hv, D);
    return qwen35_attn_launched("Bonsai gdn prep");
}

extern "C" int ds4_gpu_qwen4_gdn_scan_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *state,
        const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
        uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen35_attn;
    const uint64_t bytes = (uint64_t)Hv * D * D * 4;
    if (!T || !Hk || !Hv || Hv % Hk || D < 32 || D > 128 || D % 32 ||
        !qwen35_attn_tensor(out, (uint64_t)T * Hv * D * 4) || !qwen35_attn_tensor(state, bytes) ||
        !qwen35_attn_tensor(qkv, (uint64_t)T * (2 * Hk + Hv) * D * 4) ||
        !qwen35_attn_tensor(a, (uint64_t)T * Hv * 4) || !qwen35_attn_tensor(b, (uint64_t)T * Hv * 4) ||
        (snap && !qwen35_attn_tensor(snap, bytes)) || (snap2 && !qwen35_attn_tensor(snap2, bytes))) return 0;
#define QWEN_GDN(ROWS, DIM) gdn_scan<ROWS,DIM><<<dim3((DIM+4*ROWS-1)/(4*ROWS),Hv),128,0,cuda_decode_stream()>>>((float *)out->ptr, \
        (float *)state->ptr,(const float *)qkv->ptr,(const float *)a->ptr,(const float *)b->ptr, \
        T,Hk,Hv,snap ? (float *)snap->ptr : NULL,st,snap2 ? (float *)snap2->ptr : NULL,st2)
#define QWEN_GDN_DIM(DIM) case DIM: if (T > 8) { QWEN_GDN(4,DIM); } else { QWEN_GDN(1,DIM); } break
    switch (D) { QWEN_GDN_DIM(32); QWEN_GDN_DIM(64); QWEN_GDN_DIM(96); QWEN_GDN_DIM(128); }
#undef QWEN_GDN_DIM
#undef QWEN_GDN
    return qwen35_attn_launched("Bonsai gdn scan");
}

/* The qwen4exp gate: sigmoid.  Bonsai's silu variant is the qwen35 entry below,
 * the same kernel with the other selector. */
extern "C" int ds4_gpu_qwen4_gdn_out_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *z,
        const void *map, uint64_t size, uint64_t off, uint32_t T, uint32_t H, uint32_t D, float eps) {
    using namespace qwen35_attn;
    const uint64_t n = (uint64_t)T * H * D;
    if (!n || D < 32 || D > 128 || D % 32 || !qwen35_attn_tensor(out, n * 4) ||
        !qwen35_attn_tensor(z, n * 4)) return 0;
    const char *w = qwen35_attn_weight(map, size, off, (uint64_t)D * 4);
    if (!w) return 0;
    gdn_out<<<dim3(H, T), 32, 0, cuda_decode_stream()>>>((float *)out->ptr, (const float *)z->ptr,
        (const float *)w, H, D, eps, 0u);
    return qwen35_attn_launched("qwen4 gdn out");
}

/* Conditions under which the token-tile MMA attention kernel runs.  One place,
 * because two callers have to agree on them: this dispatcher, which takes the
 * kernel whenever it is handed no partial buffer, and the qwen35 graph, which
 * has to know whether handing over no partial buffer selects that kernel
 * before it decides to size a batch for it.  Pre-Ampere cards, quality mode and
 * other head dims answer no, and the caller then keeps the row-exact split-K
 * path with its per-row batch. */
static bool qwen35_attn_tokentile_ok(const ds4_gpu_tensor *kc, const ds4_gpu_tensor *vc,
                                     uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D) {
    return T >= 32u && D == 256u && Hkv && H % Hkv == 0u && H / Hkv <= 16u &&
           kc && vc && !((uintptr_t)kc->ptr & 15) && !((uintptr_t)vc->ptr & 15) &&
           ds4_cuda_attn_tokentile_arch_ok() && !g_quality_mode &&
           !getenv("DS4_QWEN4_NO_ATTN_MM");
}

extern "C" int ds4_gpu_qwen4_attn_tokentile_available(
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache,
        uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim) {
    return qwen35_attn_tokentile_ok(k_cache, v_cache, n_tokens, n_head, n_head_kv, head_dim) ? 1 : 0;
}

/* Dense grouped-query attention over a fp16 k/v cache.  `partial` selects the
 * split-K order (row-exact kernel plus attn_merge); without it a long enough
 * batch takes the token-tile kernel.  Bonsai always passes NULL, NULL, NULL for
 * sel/count/partial, so the sparse indexer slots stay unused. */
extern "C" int ds4_gpu_qwen4_attn_decode_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *gate, const ds4_gpu_tensor *kc, const ds4_gpu_tensor *vc,
        const ds4_gpu_tensor *sel, const ds4_gpu_tensor *count, ds4_gpu_tensor *partial,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D, uint32_t pos0, bool sparse, uint32_t stride, float scale) {
    using namespace qwen35_attn;
    const uint64_t n = (uint64_t)T * H * D * 4, cb = ((uint64_t)pos0 + T) * Hkv * D * 2;
    if (!T || !H || !Hkv || H % Hkv || (D != 32 && D != 128 && D != 256) ||
        !qwen35_attn_tensor(out, n) || !qwen35_attn_tensor(q, n) || !qwen35_attn_tensor(gate, n) ||
        !qwen35_attn_tensor(kc, cb) || !qwen35_attn_tensor(vc, cb) ||
        (sparse && (!stride || !qwen35_attn_tensor(sel, (uint64_t)T * stride * 4) ||
                    !qwen35_attn_tensor(count, (uint64_t)T * 4)))) return 0;
    const unsigned keys = sparse ? stride : pos0 + T;
    if (!partial && qwen35_attn_tokentile_ok(kc, vc, T, H, Hkv, D)) {
        attention_group<<<dim3(Hkv,T),256,0,cuda_decode_stream()>>>((float *)out->ptr,
            (const float *)q->ptr,(const float *)gate->ptr,(const __half *)kc->ptr,(const __half *)vc->ptr,
            sparse ? (const int *)sel->ptr : NULL,sparse ? (const unsigned *)count->ptr : NULL,
            H,Hkv,pos0,stride,sparse,scale);
        return qwen35_attn_launched("Bonsai attention token tile");
    }
    const unsigned splits = partial ? std::min(64u, (keys + 31) / 32) : 1;
    if (partial && !qwen35_attn_tensor(partial, (uint64_t)T * H * splits * (D + 2) * 4)) return 0;
    const dim3 grid((H + 3) / 4, T, splits);
#define QWEN_ATTN(DIM) attention<DIM><<<grid, 128, 0, cuda_decode_stream()>>>((float *)out->ptr, \
        partial ? (float *)partial->ptr : NULL, (const float *)q->ptr, (const float *)gate->ptr, \
        (const __half *)kc->ptr, (const __half *)vc->ptr, sparse ? (const int *)sel->ptr : NULL, \
        sparse ? (const unsigned *)count->ptr : NULL, H, Hkv, pos0, stride, sparse, splits, (keys + splits - 1) / splits, scale)
    if (D == 32) { QWEN_ATTN(32); }
    else if (D == 128) { QWEN_ATTN(128); }
    else { QWEN_ATTN(256); }
#undef QWEN_ATTN
    if (!qwen35_attn_launched("Bonsai attention")) return 0;
    if (splits > 1) attn_merge<<<dim3(H, T), D, 0, cuda_decode_stream()>>>((float *)out->ptr,
        (const float *)partial->ptr, (const float *)gate->ptr, H, D, splits);
    return qwen35_attn_launched("Bonsai attention merge");
}

/* Gated output norm of the Bonsai linear layer: the qwen4 kernel with the
 * silu gate this family uses (ds4_qwen35_ref_linear) instead of sigmoid. */
extern "C" int ds4_gpu_qwen35_gdn_out_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *z,
        const void *map, uint64_t size, uint64_t off, uint32_t T, uint32_t H,
        uint32_t D, float eps) {
    const uint64_t n = (uint64_t) T * H * D;
    if (!n || D < 32 || D > 128 || D % 32 || !qwen35_attn_tensor(out, n * 4) ||
        !qwen35_attn_tensor(z, n * 4)) {
        return 0;
    }
    const char *w = qwen35_attn_weight(map, size, off, (uint64_t) D * 4);
    if (!w) return 0;
    qwen35_attn::gdn_out<<<dim3(H, T), 32, 0, cuda_decode_stream()>>>(
        (float *) out->ptr, (const float *) z->ptr, (const float *) w, H, D,
        eps, /*gate_silu=*/1u);
    return qwen35_attn_launched("Bonsai gdn out");
}

/* Attention prep of the Bonsai full-attention layer: the qwen4 kernel with no
 * indexer slot (this model has no sparse indexer), so it produces the roped
 * q with its raw sigmoid gate, the roped k and the v store.  The q and k
 * per-head norms both carry a gamma of length D and the same eps. */
extern "C" int ds4_gpu_qwen35_attn_prep_tensor(
        ds4_gpu_tensor *q, ds4_gpu_tensor *gate, ds4_gpu_tensor *kc,
        ds4_gpu_tensor *vc, const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kp,
        const ds4_gpu_tensor *vp, const ds4_gpu_tensor *pos3,
        const void *map, uint64_t size, uint64_t qo, uint64_t ko, uint32_t T,
        uint32_t H, uint32_t Hkv, uint32_t D, uint32_t nrot, uint32_t pos0,
        uint32_t cap, float base, float eps) {
    using namespace qwen35_attn;
    const uint64_t qb = (uint64_t) T * H * D * 4, kb = (uint64_t) T * Hkv * D * 4;
    if (!T || !H || !Hkv || H % Hkv || D < 32 || D > 256 || D % 32 ||
        nrot > 64 || nrot > D || nrot % 2 || (uint64_t) pos0 + T > cap ||
        !qwen35_attn_tensor(q, qb) || !qwen35_attn_tensor(gate, qb) || !qwen35_attn_tensor(qg, qb * 2) ||
        !qwen35_attn_tensor(kp, kb) || !qwen35_attn_tensor(vp, kb) ||
        !qwen35_attn_tensor(kc, (uint64_t) cap * Hkv * D * 2) ||
        !qwen35_attn_tensor(vc, (uint64_t) cap * Hkv * D * 2) ||
        !qwen35_attn_tensor(pos3, (uint64_t) cap * 16)) {
        return 0;
    }
    const char *gq = qwen35_attn_weight(map, size, qo, (uint64_t) D * 4);
    const char *gk = qwen35_attn_weight(map, size, ko, (uint64_t) D * 4);
    if (!gq || !gk) return 0;
    attn_prep<<<dim3(H + Hkv, T), 32, 0, cuda_decode_stream()>>>(
        (float *) q->ptr, (float *) gate->ptr, (__half *) kc->ptr,
        (__half *) vc->ptr, /*iqout=*/NULL, /*ikc=*/NULL,
        (const float *) qg->ptr, (const float *) kp->ptr,
        (const float *) vp->ptr, /*iq=*/NULL, /*ik=*/NULL,
        (const uint32_t *) pos3->ptr, (const float *) gq, (const float *) gk,
        /*giq=*/NULL, H, Hkv, D, /*Hi=*/0, /*Di=*/0, pos0, eps,
        rope(nrot, base));
    return qwen35_attn_launched("Bonsai attn prep");
}
