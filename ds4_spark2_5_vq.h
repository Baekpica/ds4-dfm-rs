#ifndef DS4_SPARK2_5_VQ_H
#define DS4_SPARK2_5_VQ_H

#include <stdint.h>

/* Spark-X2.6-4B VQ (`spark2_5`): a dense, 36-layer GQA transformer over the
 * same R13+8 container Gemma-4 uses, with no routed experts and no drafter.
 * Topology and rope schedule: config.json in the Spark-X2.6-4B-Base checkpoint (<hf-dir>)
 * and modeling_spark.py; the artifact is documented in docs/spark-x2.6-vq.md.
 *
 * Two graph-specific kernels live in cuda/spark2_5.cuh; every other step is
 * an existing entry point (cuda/spark2_5.cuh names them). */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_gpu_tensor ds4_gpu_tensor;

enum {
    /* config.json: num_hidden_layers, hidden_size, vocab_size, head counts,
     * head_dim, intermediate_size, sliding_window, max_position_embeddings. */
    SPARK_LAYERS = 36,
    SPARK_EMBD = 2560,
    SPARK_VOCAB = 131072,
    SPARK_HEADS = 16,
    SPARK_KV_HEADS = 4,
    SPARK_HEAD_DIM = 256,
    /* The fused q_k_v_proj is [q 4096 | k 1024 | v 1024] in that order
     * (modeling_spark.py:152-176: qkv_out_dim = q_dim + 2 * kv_dim). */
    SPARK_Q_DIM = SPARK_HEADS * SPARK_HEAD_DIM,
    SPARK_KV_DIM = SPARK_KV_HEADS * SPARK_HEAD_DIM,
    SPARK_QKV_DIM = SPARK_Q_DIM + 2 * SPARK_KV_DIM,
    SPARK_FF = 10240,
    SPARK_WINDOW = 512,
    /* config.json rope_parameters: full layers 0.25 * 256 = 64 rotary dims at
     * 5e6, sliding layers all 256 at 1e4. */
    SPARK_ROT_FULL = 64,
    SPARK_ROT_SLIDING = 256,
    SPARK_CONTEXT = 1048576,
    /* Activation rows per prefill chunk. The sliding ring is window + chunk
     * rows (the Gemma-4 convention), so the cap is the only sizing knob. */
    SPARK_PREFILL_MAX = 1024,
    /* The R13+8 container: 13-bit indices into 8192 E4M3 words, plus the
     * 8-bit GVR8 residual indices into 256 words. */
    SPARK_VECTOR = 8,
    SPARK_WORDS = 8192,
    SPARK_RES_WORDS = 256,
};

#define SPARK_RMS_EPS 1e-6f
#define SPARK_THETA_FULL 5000000.0
#define SPARK_THETA_SLIDING 10000.0

/* 1/sqrt(256) = 2^-4, modeling_spark.py:146 and :84. Folded into q after the
 * rope; a power of two makes the fold exact in both BF16 and F32. */
#define SPARK_ATTN_SCALE 0.0625f

/* idx % 4 == 3: layer_types is three sliding layers then one full, nine
 * times (config.json layer_types; nine full layers 3, 7, ... 35). */
static inline int spark2_5_layer_full(unsigned il) {
    return il % 4u == 3u;
}

static inline uint32_t spark2_5_rot(uint32_t il) {
    return spark2_5_layer_full(il) ? SPARK_ROT_FULL : SPARK_ROT_SLIDING;
}

/* The three graph-specific entries (cuda/spark2_5.cuh). */
int ds4_gpu_spark_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *positions,
    const ds4_gpu_tensor *inv, uint32_t rows, uint32_t heads, uint32_t dim,
    uint32_t pairs);
int ds4_gpu_spark_gelu_mul(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
    const ds4_gpu_tensor *up, uint64_t count);
int ds4_gpu_spark_qkv_split(ds4_gpu_tensor *q, ds4_gpu_tensor *k,
    ds4_gpu_tensor *v, const ds4_gpu_tensor *qkv, uint32_t rows);

#ifdef __cplusplus
}
#endif

#endif
