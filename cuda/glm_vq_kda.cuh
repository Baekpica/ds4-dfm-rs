#ifndef DS4_GLM_VQ_KDA_CUH
#define DS4_GLM_VQ_KDA_CUH

#include "glm_vq_ops.cuh"

enum { GLMVQ_KDA_DIM = 128, GLMVQ_CONV = 4 };

/* State after each row but the last, for MTP verify rollback: row t goes
 * to slot t of `stride` floats, at the same offsets as the live state. */
struct glmvq_kda_snap { float *state, *q, *k, *v; uint64_t stride; };

/* Source baseline: modeling_glm5_next.py:340-369,409-449,453-514.
 * Conv and beta are BF16; the recurrent matrix is FP32. Sequence order is
 * explicit, so an append consumes exactly the prior state and raw conv tail.
 * This bounded fallback precedes profiling of the existing chunked KDA path.
 * HELD keeps column d of the head's state in registers across the rows:
 * one read and write per call instead of two of each per row (MTP verify
 * rows re-read it from L2), same operations and order. One decode row
 * keeps the memory loops (255 registers measured 1.5% slower there). */
template <bool HELD = false>
__global__ void glmvq_kda(float *output, float *state,
        float *q_tail, float *k_tail, float *v_tail,
        const float *q_raw, const float *k_raw, const float *v_raw,
        const float *forget, const float *beta_logits,
        const float *q_weight, const float *k_weight, const float *v_weight,
        const float *a_log, const float *dt_bias, uint32_t rows,
        uint32_t heads, float lower_bound, glmvq_kda_snap snap) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t head = blockIdx.x, d = threadIdx.x;
    const uint32_t dim = GLMVQ_KDA_DIM;
    const uint64_t channel = (uint64_t)head * dim + d, width = (uint64_t)heads * dim;
    const uint64_t state_base = (uint64_t)head * dim * dim;
    __shared__ float q[GLMVQ_KDA_DIM], k[GLMVQ_KDA_DIM], v[GLMVQ_KDA_DIM];
    __shared__ float qs[GLMVQ_KDA_DIM], ks[GLMVQ_KDA_DIM], decay[GLMVQ_KDA_DIM];
    const float scale = __fdiv_rn(1.0f, __fsqrt_rn((float)dim));
    const float a = expf(a_log[head]);
    float st[HELD ? GLMVQ_KDA_DIM : 1];
    if constexpr (HELD) {
        #pragma unroll
        for (uint32_t key = 0; key < GLMVQ_KDA_DIM; key++) { st[key] = state[state_base + (uint64_t)key * dim + d]; }
    }

    for (uint32_t t = 0; t < rows; t++) {
        const uint64_t input = (uint64_t)t * width + channel, tail = channel * GLMVQ_CONV;
        for (uint32_t tap = 0; tap + 1 < GLMVQ_CONV; tap++) {
            q_tail[tail + tap] = q_tail[tail + tap + 1];
            k_tail[tail + tap] = k_tail[tail + tap + 1];
            v_tail[tail + tap] = v_tail[tail + tap + 1];
        }
        q_tail[tail + GLMVQ_CONV - 1] = q_raw[input];
        k_tail[tail + GLMVQ_CONV - 1] = k_raw[input];
        v_tail[tail + GLMVQ_CONV - 1] = v_raw[input];
        const bool keep = snap.state && t + 1 < rows;
        if (keep) {
            for (uint32_t tap = 0; tap < GLMVQ_CONV; tap++) {
                snap.q[t * snap.stride + tail + tap] = q_tail[tail + tap];
                snap.k[t * snap.stride + tail + tap] = k_tail[tail + tap];
                snap.v[t * snap.stride + tail + tap] = v_tail[tail + tap];
            }
        }
        float qc = 0.0f, kc = 0.0f, vc = 0.0f;
        for (uint32_t tap = 0; tap < GLMVQ_CONV; tap++) {
            qc = glmvq_add(qc, glmvq_mul(q_tail[tail + tap], q_weight[tail + tap]));
            kc = glmvq_add(kc, glmvq_mul(k_tail[tail + tap], k_weight[tail + tap]));
            vc = glmvq_add(vc, glmvq_mul(v_tail[tail + tap], v_weight[tail + tap]));
        }
        qc = glmvq_round(qc); kc = glmvq_round(kc); vc = glmvq_round(vc);
        q[d] = glmvq_round(__fdiv_rn(qc, glmvq_add(1.0f, expf(-qc))));
        k[d] = glmvq_round(__fdiv_rn(kc, glmvq_add(1.0f, expf(-kc))));
        v[d] = glmvq_round(__fdiv_rn(vc, glmvq_add(1.0f, expf(-vc))));
        qs[d] = glmvq_mul(q[d], q[d]); ks[d] = glmvq_mul(k[d], k[d]);
        const float g = glmvq_add(forget[input], dt_bias[channel]);
        decay[d] = expf(glmvq_mul(lower_bound, glmvq_sigmoid(glmvq_mul(a, g))));
        __syncthreads();
        for (uint32_t stride = dim / 2; stride; stride /= 2) {
            if (d < stride) {
                qs[d] = glmvq_add(qs[d], qs[d + stride]);
                ks[d] = glmvq_add(ks[d], ks[d + stride]);
            }
            __syncthreads();
        }
        const float qnorm = __fsqrt_rn(glmvq_add(qs[0], 1.0e-6f));
        const float knorm = __fsqrt_rn(glmvq_add(ks[0], 1.0e-6f));
        q[d] = glmvq_mul(__fdiv_rn(q[d], qnorm), scale);
        k[d] = __fdiv_rn(k[d], knorm);
        __syncthreads();

        float memory = 0.0f;
        if constexpr (HELD) {
            #pragma unroll
            for (uint32_t key = 0; key < GLMVQ_KDA_DIM; key++) {
                st[key] = glmvq_mul(st[key], decay[key]);
                memory = glmvq_add(memory, glmvq_mul(st[key], k[key]));
            }
        } else {
            for (uint32_t key = 0; key < dim; key++) {
                const uint64_t at = state_base + (uint64_t)key * dim + d;
                const float decayed = glmvq_mul(state[at], decay[key]);
                state[at] = decayed;
                memory = glmvq_add(memory, glmvq_mul(decayed, k[key]));
            }
        }
        const float beta = glmvq_round(glmvq_sigmoid(beta_logits[(uint64_t)t * heads + head]));
        const float delta = glmvq_mul(glmvq_add(v[d], -memory), beta);
        float result = 0.0f;
        if constexpr (HELD) {
            #pragma unroll
            for (uint32_t key = 0; key < GLMVQ_KDA_DIM; key++) {
                st[key] = glmvq_add(st[key], glmvq_mul(k[key], delta));
                if (keep) { snap.state[t * snap.stride + state_base + (uint64_t)key * dim + d] = st[key]; }
                result = glmvq_add(result, glmvq_mul(st[key], q[key]));
            }
        } else {
            for (uint32_t key = 0; key < dim; key++) {
                const uint64_t at = state_base + (uint64_t)key * dim + d;
                const float updated = glmvq_add(state[at], glmvq_mul(k[key], delta));
                state[at] = updated;
                if (keep) { snap.state[t * snap.stride + at] = updated; }
                result = glmvq_add(result, glmvq_mul(updated, q[key]));
            }
        }
        output[input] = glmvq_round(result);
        __syncthreads();
    }
    if constexpr (HELD) {
        #pragma unroll
        for (uint32_t key = 0; key < GLMVQ_KDA_DIM; key++) { state[state_base + (uint64_t)key * dim + d] = st[key]; }
    }
}

/* Chunked-prefill inputs, token-parallel and bit-identical to glmvq_kda's
 * per-token q/k/v, beta and decay argument: planes are head-major
 * [head][token][dim] (the chunk factor/scan layout), the decay stays in log
 * space and beta is already rounded at its BF16 boundary. Tokens 0-2 read
 * the conv tail left by the previous call; the tail itself advances later. */
__global__ void glmvq_kda_prep(float *qn, float *kn, float *vv, float *gl, float *betas,
        const float *q_tail, const float *k_tail, const float *v_tail,
        const float *q_raw, const float *k_raw, const float *v_raw,
        const float *forget, const float *beta_logits,
        const float *q_weight, const float *k_weight, const float *v_weight,
        const float *a_log, const float *dt_bias, uint32_t rows,
        uint32_t heads, float lower_bound) {
    const uint32_t t = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const uint32_t dim = GLMVQ_KDA_DIM;
    const uint64_t channel = (uint64_t)head * dim + d, width = (uint64_t)heads * dim;
    __shared__ float qs[GLMVQ_KDA_DIM], ks[GLMVQ_KDA_DIM];
    const float scale = __fdiv_rn(1.0f, __fsqrt_rn((float)dim));
    float qc = 0.0f, kc = 0.0f, vc = 0.0f;
    for (uint32_t tap = 0; tap < GLMVQ_CONV; tap++) {
        const int32_t src = (int32_t)t - (int32_t)(GLMVQ_CONV - 1) + (int32_t)tap;
        const uint64_t at = (uint64_t)src * width + channel;
        const uint64_t slot = channel * GLMVQ_CONV + (uint32_t)((int32_t)GLMVQ_CONV + src);
        const uint64_t w = channel * GLMVQ_CONV + tap;
        qc = glmvq_add(qc, glmvq_mul(src >= 0 ? q_raw[at] : q_tail[slot], q_weight[w]));
        kc = glmvq_add(kc, glmvq_mul(src >= 0 ? k_raw[at] : k_tail[slot], k_weight[w]));
        vc = glmvq_add(vc, glmvq_mul(src >= 0 ? v_raw[at] : v_tail[slot], v_weight[w]));
    }
    qc = glmvq_round(qc); kc = glmvq_round(kc); vc = glmvq_round(vc);
    float q = glmvq_round(__fdiv_rn(qc, glmvq_add(1.0f, expf(-qc))));
    float k = glmvq_round(__fdiv_rn(kc, glmvq_add(1.0f, expf(-kc))));
    const float v = glmvq_round(__fdiv_rn(vc, glmvq_add(1.0f, expf(-vc))));
    qs[d] = glmvq_mul(q, q); ks[d] = glmvq_mul(k, k);
    __syncthreads();
    for (uint32_t stride = dim / 2; stride; stride /= 2) {
        if (d < stride) {
            qs[d] = glmvq_add(qs[d], qs[d + stride]);
            ks[d] = glmvq_add(ks[d], ks[d + stride]);
        }
        __syncthreads();
    }
    q = glmvq_mul(__fdiv_rn(q, __fsqrt_rn(glmvq_add(qs[0], 1.0e-6f))), scale);
    k = __fdiv_rn(k, __fsqrt_rn(glmvq_add(ks[0], 1.0e-6f)));
    const float g = glmvq_add(forget[(uint64_t)t * width + channel], dt_bias[channel]);
    const uint64_t row = ((uint64_t)head * rows + t) * dim + d;
    qn[row] = q; kn[row] = k; vv[row] = v;
    gl[row] = glmvq_mul(lower_bound, glmvq_sigmoid(glmvq_mul(expf(a_log[head]), g)));
    if (!d) {
        const uint64_t at = (uint64_t)t * heads + head;
        betas[at] = glmvq_round(glmvq_sigmoid(beta_logits[at]));
    }
}

/* Tensor-core chunked prefill. The Solar chunk kernels kept F32 [head][token]
 * planes and re-read them in four passes (12 of a KDA layer's 51 ms at 2,048
 * rows, ~2 GB of traffic). glmvq_kda_local reads the raw q/k/v/forget rows
 * once per (64-token chunk, head) and writes one record of F16 MMA operands;
 * glmvq_kda_scan walks one head's chunks with the state in F32 registers and
 * F16 tensor-core products, and applies the gated output norm.
 *
 *   chunk:  V' = U - W S          O = Q~ S + Mq V'       S <- diag(e^G_last) S + K^T V'
 *           U = T beta V, W = T beta K e^G, T = (I + beta K K^T e^(G_t - G_s))^-1 (strict lower)
 *
 * Arithmetic: q/k/v, beta and the log decay equal the recurrence's bit for
 * bit; k stays F32 (it feeds the state), q is kept as F16. Records are F16
 * except U; the scan rounds the state to F16 per chunk and splits V' into
 * two F16 halves. Every product accumulates in F32. */
enum { GLMVQ_KC = 64, GLMVQ_KINPUT = GLMVQ_KC / 4 * 32, GLMVQ_KLOCAL = GLMVQ_KINPUT + 32,
       GLMVQ_KSCAN = 256, GLMVQ_KAROW = GLMVQ_KC + 1,
       GLMVQ_KTILES = (GLMVQ_KC / 4) * (GLMVQ_KC / 4 + 1) / 2, GLMVQ_KSUB = 16 };

// One (chunk, head) record; padding tokens of a short last chunk are zero.
enum : uint32_t {
    GLMVQ_KR_QT = 0,                                           // F16 [t][k] q e^G
    GLMVQ_KR_KT = GLMVQ_KR_QT + GLMVQ_KC * GLMVQ_KDA_DIM * 2,  // F16 [k][t] k e^(G_last - G)
    GLMVQ_KR_WN = GLMVQ_KR_KT + GLMVQ_KC * GLMVQ_KDA_DIM * 2,  // F16 [t][k] -W
    GLMVQ_KR_MQ = GLMVQ_KR_WN + GLMVQ_KC * GLMVQ_KDA_DIM * 2,  // F16 [t][s] causal q.k e^(G_t - G_s)
    GLMVQ_KR_U = GLMVQ_KR_MQ + GLMVQ_KC * GLMVQ_KC * 2,        // F32 [t][v] U
    GLMVQ_KR_DL = GLMVQ_KR_U + GLMVQ_KC * GLMVQ_KDA_DIM * 4,   // F32 [k] e^G_last
    GLMVQ_KR_BYTES = GLMVQ_KR_DL + GLMVQ_KDA_DIM * 4 };

// q (F16; v as BF16 once q is spent), k and G (F32), A/T and beta: one CTA
// per SM, inside the 99 KiB opt-in.
static constexpr uint32_t GLMVQ_KLOCAL_SMEM = GLMVQ_KC * GLMVQ_KDA_DIM * 2 +
    2 * GLMVQ_KC * GLMVQ_KDA_DIM * 4 + GLMVQ_KC * GLMVQ_KAROW * 4 + GLMVQ_KC * 4;

__device__ __forceinline__ float glmvq_f4(const float4 &v, uint32_t i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ uint32_t glmvq_h2(float lo, float hi) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *(const uint32_t *)&v;
}

// T block (i, j) of a 64x64 unit lower triangle: diagonal blocks live in A's
// storage, the six below it in `tmp` until every A block has been read.
// Slots by distance: (1,0) (2,1) (3,2), then (2,0) (3,1), then (3,0).
__device__ __forceinline__ float *glmvq_kda_tblock(float *tmp, uint32_t i, uint32_t j) {
    const uint32_t dist = i - j, slot = dist == 1 ? j : dist == 2 ? 3 + j : 5;
    return tmp + slot * GLMVQ_KSUB * GLMVQ_KSUB;
}

/* One CTA per (chunk, head); 17 warps cover all 544 A/Mq tasks at once.
 * Produces everything independent of the incoming state. The conv window reads the previous
 * call's tail for its first three tokens; the tail advances later. */
__device__ __forceinline__ float4 glmvq_kda_load4(const float *p) { return *(const float4 *)p; }
__device__ __forceinline__ float4 glmvq_kda_load4(const __nv_bfloat16 *p) {
    const uint2 v = *(const uint2 *)p;
    const __nv_bfloat162 lo = *(const __nv_bfloat162 *)&v.x, hi = *(const __nv_bfloat162 *)&v.y;
    return make_float4(__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi));
}

// Q (raw q/k/v) and F (forget) are F32 or BF16 projection outputs read in place.
template <typename Q, typename F, uint32_t THREADS = GLMVQ_KLOCAL>
__global__ void __launch_bounds__(THREADS, 1) glmvq_kda_local(uint8_t *records,
        const float *q_tail, const float *k_tail, const float *v_tail,
        const Q *q_raw, const Q *k_raw, const Q *v_raw,
        const F *forget, const float *beta_logits,
        const float *q_weight, const float *k_weight, const float *v_weight,
        const float *a_log, const float *dt_bias, uint32_t rows,
        uint32_t heads, float lower_bound) {
    static_assert(THREADS == GLMVQ_KINPUT || THREADS == GLMVQ_KLOCAL, "16 or 17 local warps");
    constexpr uint32_t C = GLMVQ_KC, D = GLMVQ_KDA_DIM, AR = GLMVQ_KAROW, S = GLMVQ_KSUB;
    extern __shared__ __align__(16) uint8_t kda_shared[];
    __half *sq = (__half *)kda_shared;
    __nv_bfloat16 *sv = (__nv_bfloat16 *)kda_shared;  // replaces q after T
    float *sk = (float *)(sq + C * D), *sg = sk + C * D, *sa = sg + C * D, *sb = sa + C * AR;
    float vkeep[4][4];
    const uint32_t chunk = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    const uint32_t lane = tid & 31, warp = tid >> 5;
    const uint32_t t0 = chunk * C, n = min((uint32_t)C, rows - t0);
    const uint64_t width = (uint64_t)heads * D;
    uint8_t *rec = records + ((uint64_t)chunk * heads + head) * GLMVQ_KR_BYTES;

    // Inputs as the recurrence computes them. Warp w takes tokens 4w..4w+3 and
    // lane l dims 4l..4l+3: the conv window is 7 raw rows and the recurrence's
    // L2-norm tree (strides 64..1) becomes xor shuffles 16..1, then 2, 1 in-lane.
    if (tid < GLMVQ_KINPUT) {
        const uint32_t d0 = lane * 4;
        const uint64_t ch = (uint64_t)head * D + d0;
        const float scale = __fdiv_rn(1.0f, __fsqrt_rn((float)D));
        #pragma unroll
        for (uint32_t x = 0; x < 3; x++) {
            const Q *raw = x == 0 ? q_raw : x == 1 ? k_raw : v_raw;
            const float *tail = x == 0 ? q_tail : x == 1 ? k_tail : v_tail;
            const float *weight = x == 0 ? q_weight : x == 1 ? k_weight : v_weight;
            float4 win[GLMVQ_CONV + 3], tap[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++) { tap[i] = *(const float4 *)(weight + (ch + i) * GLMVQ_CONV); }
            #pragma unroll
            for (uint32_t r = 0; r < GLMVQ_CONV + 3; r++) {
                const int32_t src = (int32_t)(t0 + 4 * warp + r) - (int32_t)(GLMVQ_CONV - 1);
                if (src >= (int32_t)rows) { win[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f); }
                else if (src >= 0) { win[r] = glmvq_kda_load4(raw + (uint64_t)src * width + ch); }
                else {
                    const uint32_t slot = (uint32_t)((int32_t)GLMVQ_CONV + src);
                    win[r] = make_float4(tail[(ch + 0) * GLMVQ_CONV + slot], tail[(ch + 1) * GLMVQ_CONV + slot],
                        tail[(ch + 2) * GLMVQ_CONV + slot], tail[(ch + 3) * GLMVQ_CONV + slot]);
                }
            }
            #pragma unroll
            for (uint32_t j = 0; j < 4; j++) {
                const uint32_t t = 4 * warp + j;
                float y[4];
                #pragma unroll
                for (uint32_t i = 0; i < 4; i++) {
                    float c = 0.0f;
                    #pragma unroll
                    for (uint32_t k = 0; k < GLMVQ_CONV; k++) {
                        c = glmvq_add(c, glmvq_mul(glmvq_f4(win[j + k], i), glmvq_f4(tap[i], k)));
                    }
                    c = glmvq_round(c);
                    y[i] = glmvq_round(__fdiv_rn(c, glmvq_add(1.0f, expf(-c))));
                }
                if (x < 2) {
                    float s[4];
                    #pragma unroll
                    for (uint32_t i = 0; i < 4; i++) { s[i] = glmvq_mul(y[i], y[i]); }
                    #pragma unroll
                    for (uint32_t stride = 16; stride; stride >>= 1) {
                        #pragma unroll
                        for (uint32_t i = 0; i < 4; i++) { s[i] = glmvq_add(s[i], __shfl_xor_sync(0xffffffffu, s[i], stride)); }
                    }
                    const float sum = glmvq_add(glmvq_add(s[0], s[2]), glmvq_add(s[1], s[3]));
                    const float norm = __fsqrt_rn(glmvq_add(sum, 1.0e-6f));
                    #pragma unroll
                    for (uint32_t i = 0; i < 4; i++) {
                        y[i] = x == 0 ? glmvq_mul(__fdiv_rn(y[i], norm), scale) : __fdiv_rn(y[i], norm);
                    }
                }
                #pragma unroll
                for (uint32_t i = 0; i < 4; i++) {
                    const float value = t < n ? y[i] : 0.0f;
                    if (x == 0) { sq[t * D + d0 + i] = __float2half_rn(value); }
                    else if (x == 1) { sk[t * D + d0 + i] = value; }
                    else { vkeep[j][i] = value; }
                }
            }
        }
        const float a = expf(a_log[head]);
        const float4 bias = *(const float4 *)(dt_bias + ch);
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            const uint32_t t = 4 * warp + j;
            const float4 f = t < n ? glmvq_kda_load4(forget + (uint64_t)(t0 + t) * width + ch)
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++) {
                const float g = glmvq_add(glmvq_f4(f, i), glmvq_f4(bias, i));
                sg[t * D + d0 + i] = t < n ? glmvq_mul(lower_bound, glmvq_sigmoid(glmvq_mul(a, g))) : 0.0f;
            }
        }
        if (lane < 4) {
            const uint32_t t = 4 * warp + lane;
            sb[t] = t < n ? glmvq_round(glmvq_sigmoid(beta_logits[(uint64_t)(t0 + t) * heads + head])) : 0.0f;
        }
        for (uint32_t i = tid; i < C * AR; i += GLMVQ_KINPUT) { sa[i] = 0.0f; }
        ((uint4 *)(rec + GLMVQ_KR_MQ))[tid] = make_uint4(0, 0, 0, 0);
        static_assert(GLMVQ_KC * GLMVQ_KC * 2 == GLMVQ_KINPUT * 16, "one Mq clear per thread");
    }
    __syncthreads();

    // Inclusive log-decay prefix G per channel, in token order.
    if (tid < D) {
        float acc = 0.0f;
        for (uint32_t t = 0; t < n; t++) { acc += sg[t * D + tid]; sg[t * D + tid] = acc; }
    }
    __syncthreads();

    // Q~ = q e^G, K^ = k e^(G_last - G) (stored key-major) and e^G_last.
    {
        const float *glast = sg + (n - 1) * D;
        __half *qt = (__half *)(rec + GLMVQ_KR_QT), *kt = (__half *)(rec + GLMVQ_KR_KT);
        for (uint32_t i = tid; i < C * D / 2; i += THREADS) {
            const uint32_t t = i / (D / 2), d = i % (D / 2) * 2;
            float lo = 0.0f, hi = 0.0f;
            if (t < n) {
                lo = __half2float(sq[t * D + d]) * __expf(sg[t * D + d]);
                hi = __half2float(sq[t * D + d + 1]) * __expf(sg[t * D + d + 1]);
            }
            *(uint32_t *)(qt + t * D + d) = glmvq_h2(lo, hi);
        }
        for (uint32_t i = tid; i < C * D / 2; i += THREADS) {
            const uint32_t d = i % D, t = i / D * 2;
            const float lo = t < n ? sk[t * D + d] * __expf(glast[d] - sg[t * D + d]) : 0.0f;
            const float hi = t + 1 < n ? sk[(t + 1) * D + d] * __expf(glast[d] - sg[(t + 1) * D + d]) : 0.0f;
            *(uint32_t *)(kt + d * C + t) = glmvq_h2(lo, hi);
        }
        if (tid < D) { ((float *)(rec + GLMVQ_KR_DL))[tid] = __expf(glast[tid]); }
    }

    // A (beta k.k, strict lower) and Mq (q.k, causal) over 4x4 tiles, four
    // d-quarters per tile. Decays are relative to the row block's first token,
    // so the left factor never underflows and the right one only for terms
    // below e^-87. Each lane rotates its d order to spread the banks.
    {
        __half *mq = (__half *)(rec + GLMVQ_KR_MQ);
        for (uint32_t task = tid; task < GLMVQ_KTILES * 4; task += THREADS) {
            const uint32_t tile = task >> 2, part = task & 3;
            uint32_t bt = 0;
            while ((bt + 1) * (bt + 2) / 2 <= tile) { bt++; }
            const uint32_t bs = tile - bt * (bt + 1) / 2, ref = bt / 4 * S;
            float ak[4][4] = {}, am[4][4] = {};
            for (uint32_t step = 0; step < 32; step++) {
                const uint32_t d = part * 32 + ((step + lane) & 31);
                const float gr = sg[ref * D + d];
                float lk[4], lq[4], rk[4];
                #pragma unroll
                for (uint32_t i = 0; i < 4; i++) {
                    const uint32_t t = bt * 4 + i, s = bs * 4 + i;
                    const float et = t < n ? __expf(sg[t * D + d] - gr) : 0.0f;
                    lk[i] = sk[t * D + d] * et;
                    lq[i] = __half2float(sq[t * D + d]) * et;
                    rk[i] = s < n ? sk[s * D + d] * __expf(gr - sg[s * D + d]) : 0.0f;
                }
                #pragma unroll
                for (uint32_t i = 0; i < 4; i++) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; j++) { ak[i][j] += lk[i] * rk[j]; am[i][j] += lq[i] * rk[j]; }
                }
            }
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++) {
                #pragma unroll
                for (uint32_t j = 0; j < 4; j++) {
                    #pragma unroll
                    for (uint32_t m = 1; m < 4; m <<= 1) {
                        ak[i][j] += __shfl_xor_sync(0xffffffffu, ak[i][j], m);
                        am[i][j] += __shfl_xor_sync(0xffffffffu, am[i][j], m);
                    }
                }
            }
            if (part) { continue; }
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++) {
                const uint32_t t = bt * 4 + i;
                #pragma unroll
                for (uint32_t j = 0; j < 4; j++) {
                    const uint32_t s = bs * 4 + j;
                    if (t >= n || s > t) { continue; }
                    if (s < t) { sa[t * AR + s] = sb[t] * ak[i][j]; }
                    mq[t * C + s] = __float2half_rn(am[i][j]);
                }
            }
        }
    }
    __syncthreads();

    // T = (I + A)^-1 in 16-row blocks. Diagonal blocks: lane c of warps 0-3
    // owns column c (forward substitution needs only its own column).
    if (warp < 4 && lane < S) {
        const uint32_t base = warp * S, c = lane;
        float col[S];
        #pragma unroll
        for (uint32_t r = 0; r < S; r++) { col[r] = r == c ? 1.0f : 0.0f; }
        #pragma unroll
        for (uint32_t r = 1; r < S; r++) {
            float acc = 0.0f;
            #pragma unroll
            for (uint32_t k = 0; k < r; k++) { acc += sa[(base + r) * AR + base + k] * col[k]; }
            if (r > c) { col[r] = -acc; }
        }
        __syncwarp(0xffffu);
        #pragma unroll
        for (uint32_t r = 0; r < S; r++) { sa[(base + r) * AR + base + c] = col[r]; }
    }
    __syncthreads();
    // Below the diagonal, by distance: T_ij = -D_i sum_k A_ik T_kj (T_jj = D_j).
    {
        float *tmp = (float *)sq, *x = tmp + 6 * S * S;
        auto block = [&](uint32_t i, uint32_t j, uint32_t r, uint32_t c) -> float {
            return i == j ? sa[(i * S + r) * AR + j * S + c] : glmvq_kda_tblock(tmp, i, j)[r * S + c];
        };
        for (uint32_t dist = 1; dist < 4; dist++) {
            const uint32_t count = 4 - dist;
            for (uint32_t o = tid; o < count * S * S; o += THREADS) {
                const uint32_t j = o / (S * S), i = j + dist, r = o / S % S, c = o % S;
                float acc = 0.0f;
                for (uint32_t k = j; k < i; k++) {
                    #pragma unroll
                    for (uint32_t m = 0; m < S; m++) { acc += sa[(i * S + r) * AR + k * S + m] * block(k, j, m, c); }
                }
                x[o] = acc;
            }
            __syncthreads();
            for (uint32_t o = tid; o < count * S * S; o += THREADS) {
                const uint32_t j = o / (S * S), i = j + dist, r = o / S % S, c = o % S;
                float acc = 0.0f;
                #pragma unroll
                for (uint32_t m = 0; m < S; m++) { acc += sa[(i * S + r) * AR + i * S + m] * x[j * S * S + m * S + c]; }
                glmvq_kda_tblock(tmp, i, j)[r * S + c] = -acc;
            }
            __syncthreads();
        }
        for (uint32_t o = tid; o < 6 * S * S; o += THREADS) {
            const uint32_t b = o / (S * S), r = o / S % S, c = o % S;
            const uint32_t j = b < 3 ? b : b < 5 ? b - 3 : 0, i = j + (b < 3 ? 1 : b < 5 ? 2 : 3);
            sa[(i * S + r) * AR + j * S + c] = tmp[o];
        }
    }
    // K~ beta = beta k e^G replaces G, which nothing reads after this.
    for (uint32_t i = tid; i < C * D; i += THREADS) {
        const uint32_t t = i / D;
        sg[i] = t < n ? sb[t] * sk[i] * __expf(sg[i]) : 0.0f;
    }
    __syncthreads();
    if (tid < GLMVQ_KINPUT) {
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++) { sv[(4 * warp + j) * D + lane * 4 + i] = __float2bfloat16_rn(vkeep[j][i]); }
        }
    }
    __syncthreads();

    // W = T K~beta and U = T beta V: warp w owns rows 4w..4w+3, lane 4 dims.
    if (tid < GLMVQ_KINPUT) {
        const uint32_t tb = warp * 4, db = lane * 4, last = min(tb + 4, n);
        float aw[4][4] = {}, au[4][4] = {};
        for (uint32_t s = 0; s < last; s++) {
            const float4 kb = *(const float4 *)(sg + s * D + db);
            const float beta = sb[s];
            const __nv_bfloat162 v01 = *(const __nv_bfloat162 *)(sv + s * D + db);
            const __nv_bfloat162 v23 = *(const __nv_bfloat162 *)(sv + s * D + db + 2);
            const float vb[4] = {__low2float(v01) * beta, __high2float(v01) * beta,
                __low2float(v23) * beta, __high2float(v23) * beta};
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++) {
                const float c = sa[(tb + i) * AR + s];
                #pragma unroll
                for (uint32_t j = 0; j < 4; j++) { aw[i][j] += c * glmvq_f4(kb, j); au[i][j] += c * vb[j]; }
            }
        }
        __half *wn = (__half *)(rec + GLMVQ_KR_WN);
        float *u = (float *)(rec + GLMVQ_KR_U);
        #pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
            const uint32_t t = tb + i;
            const bool valid = t < n;
            *(uint2 *)(wn + t * D + db) = valid ? make_uint2(glmvq_h2(-aw[i][0], -aw[i][1]), glmvq_h2(-aw[i][2], -aw[i][3]))
                : make_uint2(0, 0);
            *(float4 *)(u + t * D + db) = valid ? make_float4(au[i][0], au[i][1], au[i][2], au[i][3])
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

__device__ __forceinline__ void glmvq_kda_mma(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The last GLMVQ_CONV raw rows become the F32 conv tails (BF16 rows widen
// exactly); fewer rows shift the older tail entries down, as the Solar
// chunk kernels do.
template <typename Q>
__global__ void glmvq_kda_tail(float *q_tail, float *k_tail, float *v_tail,
        const Q *q_raw, const Q *k_raw, const Q *v_raw, uint32_t rows, uint64_t width) {
    const uint64_t channel = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= width) { return; }
    for (uint32_t i = 0; i < GLMVQ_CONV; i++) {
        const int32_t src = (int32_t)rows - (int32_t)GLMVQ_CONV + (int32_t)i;
        const uint64_t slot = channel * GLMVQ_CONV + i;
        if (src >= 0) {
            const uint64_t at = (uint64_t)src * width + channel;
            q_tail[slot] = (float)q_raw[at];
            k_tail[slot] = (float)k_raw[at];
            v_tail[slot] = (float)v_raw[at];
        } else {
            const uint64_t from = channel * GLMVQ_CONV + (uint32_t)((int32_t)GLMVQ_CONV + src);
            q_tail[slot] = q_tail[from];
            k_tail[slot] = k_tail[from];
            v_tail[slot] = v_tail[from];
        }
    }
}

// Shared staging rows are padded by 16 bytes: an 8-row ldmatrix then
// touches 32 distinct banks (row stride 68 or 36 words).
enum { GLMVQ_KS_WIDE = GLMVQ_KDA_DIM + 8, GLMVQ_KS_NARROW = GLMVQ_KC + 8, GLMVQ_KS_UROW = GLMVQ_KDA_DIM + 4 };
enum : uint32_t {
    GLMVQ_KS_WN = 0,                                            // F16 [t][k]
    GLMVQ_KS_QT = GLMVQ_KS_WN + GLMVQ_KC * GLMVQ_KS_WIDE * 2,   // F16 [t][k]
    GLMVQ_KS_MQ = GLMVQ_KS_QT + GLMVQ_KC * GLMVQ_KS_WIDE * 2,   // F16 [t][s]; output BF16 [t][v] later
    GLMVQ_KS_KT = GLMVQ_KS_MQ + GLMVQ_KC * GLMVQ_KS_NARROW * 2, // F16 [k][t]
    GLMVQ_KS_DL = GLMVQ_KS_KT + GLMVQ_KDA_DIM * GLMVQ_KS_NARROW * 2,
    GLMVQ_KS_U = GLMVQ_KS_DL + GLMVQ_KDA_DIM * 4,                  // F32 [t][v], rows padded by 4
    GLMVQ_KSCAN_SMEM = GLMVQ_KS_U + GLMVQ_KC * GLMVQ_KS_UROW * 4 };
static_assert(GLMVQ_KC * GLMVQ_KS_WIDE * 2 <= GLMVQ_KS_DL - GLMVQ_KS_MQ, "output fits the Mq/K^ stage");

__device__ __forceinline__ void glmvq_kda_ldsm(uint32_t *r, const __half *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((uint32_t)__cvta_generic_to_shared(p)));
}

// Copies `rows` record rows of `cols` halves into padded shared rows.
__device__ __forceinline__ void glmvq_kda_stage(__half *dst, const __half *src, uint32_t rows,
        uint32_t cols, uint32_t pitch, uint32_t tid) {
    const uint32_t parts = cols / 8;
    for (uint32_t i = tid; i < rows * parts; i += GLMVQ_KSCAN) {
        const uint32_t r = i / parts, c = i % parts * 8;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
            "r"((uint32_t)__cvta_generic_to_shared(dst + r * pitch + c)), "l"(src + r * cols + c) : "memory");
    }
}

/* One CTA per head, 8 warps; warp w owns value rows 16w..16w+15 of S^T
 * (all 128 keys) as m16n8 accumulators. Everything runs value-major, so the
 * S^T and V'^T accumulators repack into F16 A fragments in registers and
 * every B operand is a record matrix stored [n][k], read by ldmatrix:
 *   V'^T = U^T + S^T (-W)^T,  O^T = S^T Q~^T + V'^T Mq^T,  S^T = S^T e^G_last + V'^T K^.
 * The next chunk's -W/Q~ stage lands while Mq/K^ are used and vice versa;
 * the 128 values of a token meet in shared memory for the gated norm. */
__device__ __forceinline__ uint32_t glmvq_h2_rest(float lo, float hi, uint32_t rounded) {
    const __half2 r = *(const __half2 *)&rounded;
    return glmvq_h2(lo - __low2float(r), hi - __high2float(r));
}

// G (gate) is F32 or the BF16 projection output read in place; O (out)
// is F32 or BF16 rows for a q4p projection.
template <typename G, typename O = float>
__global__ void __launch_bounds__(GLMVQ_KSCAN, 1) glmvq_kda_scan(O *out, float *state,
        const uint8_t *records, const G *gate, const __nv_bfloat16 *norm, uint32_t rows,
        uint32_t heads, float eps) {
    constexpr uint32_t C = GLMVQ_KC, D = GLMVQ_KDA_DIM, W = GLMVQ_KS_WIDE, N = GLMVQ_KS_NARROW;
    extern __shared__ __align__(16) uint8_t scan_shared[];
    __half *sw = (__half *)(scan_shared + GLMVQ_KS_WN), *sq = (__half *)(scan_shared + GLMVQ_KS_QT);
    __half *sm = (__half *)(scan_shared + GLMVQ_KS_MQ), *sk = (__half *)(scan_shared + GLMVQ_KS_KT);
    __nv_bfloat16 *so = (__nv_bfloat16 *)(scan_shared + GLMVQ_KS_MQ);
    float *sd = (float *)(scan_shared + GLMVQ_KS_DL), *su = (float *)(scan_shared + GLMVQ_KS_U);
    const uint32_t head = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t g = lane >> 2, q = lane & 3, v0 = warp * 16;
    // ldmatrix x4: lane l addresses row l%8 of matrix l/8, matrices 8 halves apart along k.
    const uint32_t lrow = lane & 7, lcol = (lane >> 3) * 8;
    const uint64_t width = (uint64_t)heads * D;
    const uint32_t chunks = (rows + C - 1) / C;
    auto record = [&](uint32_t chunk) { return records + ((uint64_t)chunk * heads + head) * GLMVQ_KR_BYTES; };
    float *sh = state + (uint64_t)head * D * D;
    float st[16][4];
    #pragma unroll
    for (uint32_t n8 = 0; n8 < 16; n8++) {
        const uint32_t k0 = n8 * 8 + 2 * q;
        st[n8][0] = sh[k0 * D + v0 + g]; st[n8][1] = sh[(k0 + 1) * D + v0 + g];
        st[n8][2] = sh[k0 * D + v0 + g + 8]; st[n8][3] = sh[(k0 + 1) * D + v0 + g + 8];
    }
    // U (64 x 128 F32) lands with -W and Q~: 4 x 16 bytes per thread.
    auto stage_a = [&](const uint8_t *rec) {
        glmvq_kda_stage(sw, (const __half *)(rec + GLMVQ_KR_WN), C, D, W, tid);
        glmvq_kda_stage(sq, (const __half *)(rec + GLMVQ_KR_QT), C, D, W, tid);
        for (uint32_t i = tid; i < C * D / 4; i += GLMVQ_KSCAN) {
            const uint32_t t = i / (D / 4), c = i % (D / 4) * 4;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                "r"((uint32_t)__cvta_generic_to_shared(su + t * GLMVQ_KS_UROW + c)),
                "l"((const float *)(rec + GLMVQ_KR_U) + t * D + c) : "memory");
        }
    };
    stage_a(record(0));
    asm volatile("cp.async.commit_group;" ::: "memory");
    for (uint32_t chunk = 0; chunk < chunks; chunk++) {
        const uint8_t *rec = record(chunk);
        glmvq_kda_stage(sm, (const __half *)(rec + GLMVQ_KR_MQ), C, C, N, tid);
        glmvq_kda_stage(sk, (const __half *)(rec + GLMVQ_KR_KT), D, C, N, tid);
        if (tid < D / 4) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                "r"((uint32_t)__cvta_generic_to_shared(sd + tid * 4)), "l"(rec + GLMVQ_KR_DL + tid * 16) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        // The gated norm reads 64 gate rows of this head at the chunk's end.
        constexpr uint32_t LINES = D * sizeof(G) / 128;
        if (tid < C * LINES) {
            const uint32_t row = min(chunk * C + tid / LINES, rows - 1);
            asm volatile("prefetch.global.L2 [%0];" :: "l"(gate + (uint64_t)row * width + head * D + tid % LINES * (128 / sizeof(G))));
        }
        uint32_t sa[8][4];
        #pragma unroll
        for (uint32_t kk = 0; kk < 8; kk++) {
            sa[kk][0] = glmvq_h2(st[2 * kk][0], st[2 * kk][1]);
            sa[kk][1] = glmvq_h2(st[2 * kk][2], st[2 * kk][3]);
            sa[kk][2] = glmvq_h2(st[2 * kk + 1][0], st[2 * kk + 1][1]);
            sa[kk][3] = glmvq_h2(st[2 * kk + 1][2], st[2 * kk + 1][3]);
        }
        asm volatile("cp.async.wait_group 1;" ::: "memory");
        __syncthreads();
        float vn[8][4], o[8][4];
        #pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            const float *u = su + (j * 8 + 2 * q) * GLMVQ_KS_UROW + v0 + g;
            vn[j][0] = u[0]; vn[j][1] = u[GLMVQ_KS_UROW]; vn[j][2] = u[8]; vn[j][3] = u[GLMVQ_KS_UROW + 8];
            o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            #pragma unroll
            for (uint32_t kk = 0; kk < 8; kk += 2) {
                uint32_t b[4];
                const uint32_t at = (j * 8 + lrow) * W + kk * 16 + lcol;
                glmvq_kda_ldsm(b, sw + at);
                glmvq_kda_mma(vn[j], sa[kk], b[0], b[1]);
                glmvq_kda_mma(vn[j], sa[kk + 1], b[2], b[3]);
                glmvq_kda_ldsm(b, sq + at);
                glmvq_kda_mma(o[j], sa[kk], b[0], b[1]);
                glmvq_kda_mma(o[j], sa[kk + 1], b[2], b[3]);
            }
        }
        __syncthreads();
        if (chunk + 1 < chunks) { stage_a(record(chunk + 1)); }
        asm volatile("cp.async.commit_group;" ::: "memory");
        // V' as F16 high and residual parts (the residual halves the state error).
        uint32_t va[4][4], vl[4][4];
        #pragma unroll
        for (uint32_t kt4 = 0; kt4 < 4; kt4++) {
            va[kt4][0] = glmvq_h2(vn[2 * kt4][0], vn[2 * kt4][1]);
            va[kt4][1] = glmvq_h2(vn[2 * kt4][2], vn[2 * kt4][3]);
            va[kt4][2] = glmvq_h2(vn[2 * kt4 + 1][0], vn[2 * kt4 + 1][1]);
            va[kt4][3] = glmvq_h2(vn[2 * kt4 + 1][2], vn[2 * kt4 + 1][3]);
            {
                vl[kt4][0] = glmvq_h2_rest(vn[2 * kt4][0], vn[2 * kt4][1], va[kt4][0]);
                vl[kt4][1] = glmvq_h2_rest(vn[2 * kt4][2], vn[2 * kt4][3], va[kt4][1]);
                vl[kt4][2] = glmvq_h2_rest(vn[2 * kt4 + 1][0], vn[2 * kt4 + 1][1], va[kt4][2]);
                vl[kt4][3] = glmvq_h2_rest(vn[2 * kt4 + 1][2], vn[2 * kt4 + 1][3], va[kt4][3]);
            }
        }
        asm volatile("cp.async.wait_group 1;" ::: "memory");
        __syncthreads();
        // Causal: token tile j needs s tiles up to its own.
        #pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            #pragma unroll
            for (uint32_t kt4 = 0; kt4 <= j / 2; kt4 += 2) {
                uint32_t b[4];
                glmvq_kda_ldsm(b, sm + (j * 8 + lrow) * N + kt4 * 16 + lcol);
                glmvq_kda_mma(o[j], va[kt4], b[0], b[1]);
                glmvq_kda_mma(o[j], vl[kt4], b[0], b[1]);
                if (kt4 + 1 <= j / 2) {
                    glmvq_kda_mma(o[j], va[kt4 + 1], b[2], b[3]);
                    glmvq_kda_mma(o[j], vl[kt4 + 1], b[2], b[3]);
                }
            }
        }
        #pragma unroll
        for (uint32_t n8 = 0; n8 < 16; n8++) {
            const float2 dl = *(const float2 *)(sd + n8 * 8 + 2 * q);
            st[n8][0] *= dl.x; st[n8][1] *= dl.y; st[n8][2] *= dl.x; st[n8][3] *= dl.y;
            #pragma unroll
            for (uint32_t kt4 = 0; kt4 < 4; kt4 += 2) {
                uint32_t b[4];
                glmvq_kda_ldsm(b, sk + (n8 * 8 + lrow) * N + kt4 * 16 + lcol);
                glmvq_kda_mma(st[n8], va[kt4], b[0], b[1]);
                glmvq_kda_mma(st[n8], va[kt4 + 1], b[2], b[3]);
                glmvq_kda_mma(st[n8], vl[kt4], b[0], b[1]);
                glmvq_kda_mma(st[n8], vl[kt4 + 1], b[2], b[3]);
            }
        }
        // This warp's 8 gate rows load while the outputs meet in shared memory.
        float gv[C / 8][D / 32];
        #pragma unroll
        for (uint32_t r = 0; r < C / 8; r++) {
            const uint32_t row = min(chunk * C + warp * (C / 8) + r, rows - 1);
            #pragma unroll
            for (uint32_t i = 0; i < D / 32; i++) { gv[r][i] = glmvq_value(gate[(uint64_t)row * width + head * D + lane + 32 * i]); }
        }
        __syncthreads();
        // The recurrence's BF16 output boundary, in the Mq/K^ stage.
        #pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            const uint32_t t = j * 8 + 2 * q;
            so[t * W + v0 + g] = __float2bfloat16_rn(o[j][0]); so[(t + 1) * W + v0 + g] = __float2bfloat16_rn(o[j][1]);
            so[t * W + v0 + g + 8] = __float2bfloat16_rn(o[j][2]); so[(t + 1) * W + v0 + g + 8] = __float2bfloat16_rn(o[j][3]);
        }
        __syncthreads();
        // glmvq_rms_warp's gated norm: lane holds columns lane + 32i.
        #pragma unroll
        for (uint32_t r = 0; r < C / 8; r++) {
            const uint32_t t = warp * (C / 8) + r, row = chunk * C + t;
            if (row >= rows) { break; }
            float x[D / 32], sum = 0.0f;
            #pragma unroll
            for (uint32_t i = 0; i < D / 32; i++) { x[i] = __bfloat162float(so[t * W + lane + 32 * i]); }
            #pragma unroll
            for (uint32_t i = 0; i < D / 32; i++) { sum = glmvq_add(sum, glmvq_mul(x[i], x[i])); }
            for (uint32_t stride = 16; stride; stride >>= 1) { sum = glmvq_add(sum, __shfl_xor_sync(0xffffffffu, sum, stride)); }
            const float inv = __fdiv_rn(1.0f, __fsqrt_rn(glmvq_add(__fdiv_rn(sum, (float)D), eps)));
            const uint64_t base = (uint64_t)row * width + (uint64_t)head * D;
            #pragma unroll
            for (uint32_t i = 0; i < D / 32; i++) {
                const uint32_t col = lane + 32 * i;
                const float value = glmvq_mul(glmvq_mul(glmvq_mul(x[i], inv), __bfloat162float(norm[col])),
                    glmvq_sigmoid(gv[r][i]));
                out[base + col] = glmvq_cast<O>(glmvq_round(value));
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (uint32_t n8 = 0; n8 < 16; n8++) {
        const uint32_t k0 = n8 * 8 + 2 * q;
        sh[k0 * D + v0 + g] = st[n8][0]; sh[(k0 + 1) * D + v0 + g] = st[n8][1];
        sh[k0 * D + v0 + g + 8] = st[n8][2]; sh[(k0 + 1) * D + v0 + g + 8] = st[n8][3];
    }
}

#endif
