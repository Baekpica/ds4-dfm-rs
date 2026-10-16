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
 * This bounded fallback precedes profiling of the existing chunked KDA path. */
__global__ void glmvq_kda(float *output, float *state,
        float *q_tail, float *k_tail, float *v_tail,
        const float *q_raw, const float *k_raw, const float *v_raw,
        const float *forget, const float *beta_logits,
        const float *q_weight, const float *k_weight, const float *v_weight,
        const float *a_log, const float *dt_bias, uint32_t rows,
        uint32_t heads, float lower_bound, glmvq_kda_snap snap) {
    const uint32_t head = blockIdx.x, d = threadIdx.x;
    const uint32_t dim = GLMVQ_KDA_DIM;
    const uint64_t channel = (uint64_t)head * dim + d, width = (uint64_t)heads * dim;
    const uint64_t state_base = (uint64_t)head * dim * dim;
    __shared__ float q[GLMVQ_KDA_DIM], k[GLMVQ_KDA_DIM], v[GLMVQ_KDA_DIM];
    __shared__ float qs[GLMVQ_KDA_DIM], ks[GLMVQ_KDA_DIM], decay[GLMVQ_KDA_DIM];
    const float scale = __fdiv_rn(1.0f, __fsqrt_rn((float)dim));
    const float a = expf(a_log[head]);

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
        for (uint32_t key = 0; key < dim; key++) {
            const uint64_t at = state_base + (uint64_t)key * dim + d;
            const float decayed = glmvq_mul(state[at], decay[key]);
            state[at] = decayed;
            memory = glmvq_add(memory, glmvq_mul(decayed, k[key]));
        }
        const float beta = glmvq_round(glmvq_sigmoid(beta_logits[(uint64_t)t * heads + head]));
        const float delta = glmvq_mul(glmvq_add(v[d], -memory), beta);
        float result = 0.0f;
        for (uint32_t key = 0; key < dim; key++) {
            const uint64_t at = state_base + (uint64_t)key * dim + d;
            const float updated = glmvq_add(state[at], glmvq_mul(k[key], delta));
            state[at] = updated;
            if (keep) { snap.state[t * snap.stride + at] = updated; }
            result = glmvq_add(result, glmvq_mul(updated, q[key]));
        }
        output[input] = glmvq_round(result);
        __syncthreads();
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

#endif
