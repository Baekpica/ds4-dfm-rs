#ifndef DS4_GLM_VQ_KDA_CUH
#define DS4_GLM_VQ_KDA_CUH

#include "glm_vq_ops.cuh"

enum { GLMVQ_KDA_DIM = 128, GLMVQ_CONV = 4 };

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
        uint32_t heads, float lower_bound) {
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
            result = glmvq_add(result, glmvq_mul(updated, q[key]));
        }
        output[input] = glmvq_round(result);
        __syncthreads();
    }
}

#endif
