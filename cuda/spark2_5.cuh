#ifndef DS4_SPARK2_5_CUH
#define DS4_SPARK2_5_CUH

/* Spark-X2.6's two graph-specific kernels. The rest of the graph reuses named
 * entries from ds4_spark2_5_graph.inc: the F32-weight RMS norm
 * (ds4_gpu_rms_norm_weight_rows_tensor), the VQ projections
 * (ds4_gpu_gvq_linear_many), the BF16 KV store and windowed GQA attention
 * (ds4_gpu_g4_kv_store / ds4_gpu_g4_attn), and the per-head sigmoid gate from
 * dots3, ds4_gpu_dots3_gate_mul_tensor (ds4_cuda.cu:42100), whose kernel keys
 * the gate as gate_logits[i / value_dim] and therefore takes Spark's 256-wide
 * head unchanged. step37_attn_gate is not that kernel: it hard-codes 128
 * (cuda/step37_primitives.cuh:20-25).
 *
 * Reference: modeling_spark.py in the Spark-X2.6-4B-Base checkpoint (<hf-dir>)
 *   compute_rope_cos_sin  :31-40
 *   apply_rotary_pos_emb  :43-56 (rotate_half :27-30)
 *   Spark2_5MLP.forward   :126-127 (ACT2FN["gelu"], the exact-erf GELU)
 */

#include <cuda_runtime.h>

/* Half-split rotary over the first 2*pairs dimensions of a head.
 *
 * The reference builds inv_freq over `rope_head_dim = int(head_dim *
 * partial_rotary_factor)` (modeling_spark.py:31-34), duplicates the angle
 * list and rotates with rotate_half, so dimension i of the head turns against
 * dimension i + rope_head_dim/2 (modeling_spark.py:43-56). `pairs` is
 * rope_head_dim/2 and the angle is position * inv[i], exponent 2i/rope_head_dim:
 * sliding layers pass pairs 128 of a 256-wide head, full layers pairs 32 of
 * the same head and leave dims 64..255 untouched. `dim` is the head width,
 * the input's row stride.
 *
 * One block per (row, head); every thread owns its own pair, so the in-place
 * rotation races nothing. */
__global__ static void spark_rope_kernel(float *x, const uint32_t *positions,
                                         const float *inv, uint32_t rows,
                                         uint32_t heads, uint32_t dim,
                                         uint32_t pairs) {
    const uint64_t job = (uint64_t)blockIdx.x;
    if (job >= (uint64_t)rows * heads || !pairs || 2u * pairs > dim) { return; }
    float *v = x + job * dim;
    const float pos = (float)positions[job / heads];
    for (uint32_t i = threadIdx.x; i < pairs; i += blockDim.x) {
        const float angle = pos * inv[i];
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float a = v[i];
        const float b = v[i + pairs];
        v[i] = a * c - b * s;
        v[i + pairs] = b * c + a * s;
    }
}

/* out = gelu(gate) * up, the exact erf GELU Transformers' ACT2FN["gelu"] is
 * (0.5 x (1 + erf(x/sqrt(2))); the tanh approximation belongs to other
 * families' kernels and is not this one). */
__global__ static void spark_gelu_mul_kernel(float *out, const float *gate,
                                             const float *up, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const float g = gate[i];
    out[i] = 0.5f * g * (1.0f + erff(g * 0.70710678118654752f)) * up[i];
}

/* The fused projection writes [rows][q | k | v] in that order
 * (modeling_spark.py:170-176 slices the one linear's output), while the
 * attention wants each of the three as its own [rows][heads*dim] buffer.
 * One thread per source element, so the split is a single pass. */
__global__ static void spark_qkv_split_kernel(float *q, float *k, float *v,
                                              const float *qkv, uint32_t rows,
                                              uint32_t q_dim, uint32_t kv_dim) {
    const uint64_t stride = (uint64_t)q_dim + 2u * kv_dim;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * stride) { return; }
    const uint32_t row = (uint32_t)(i / stride);
    const uint32_t col = (uint32_t)(i % stride);
    const float value = qkv[i];
    if (col < q_dim) {
        q[(uint64_t)row * q_dim + col] = value;
    } else if (col < q_dim + kv_dim) {
        k[(uint64_t)row * kv_dim + (col - q_dim)] = value;
    } else {
        v[(uint64_t)row * kv_dim + (col - q_dim - kv_dim)] = value;
    }
}

#endif
