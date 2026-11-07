#ifndef DS4_SPARK2_5_VQ_H
#define DS4_SPARK2_5_VQ_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
    /* GVQ1/GVBC/GVR8 headers are 128 bytes (FORMAT.md). */
    SPARK_HEADER = 128,
    SPARK_RES_WORDS = 256,
    /* Block width for FP8 E4M3 restored weights (one f16 scale per 128 inputs). */
    SPARK_FP8_BLOCK = 128,
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

/* All 36 layers (the 9 full layers and the 27 sliding layers) keep their KV
 * as E4M3 K and V with one f16 scale per KV head by default: at 128K context
 * the 9 full layers cost 2.27 GiB (vs 4.5 GiB BF16) and the 27 sliding rings
 * cost 54.4 MiB (vs 108.8 MiB BF16), 2.32 GiB total.
 * DS4_SPARK2_5_FULL_KV=bf16 (or 0) restores BF16 KV across all layers;
 * DS4_SPARK2_5_FULL_KV=full_only keeps BF16 on the 27 sliding layers. */
static inline int spark2_5_full_kv_fp8(void) {
    const char *v = getenv("DS4_SPARK2_5_FULL_KV");
    return !(v && (strcmp(v, "bf16") == 0 || strcmp(v, "0") == 0));
}

static inline int spark2_5_sliding_kv_fp8(void) {
    const char *v = getenv("DS4_SPARK2_5_FULL_KV");
    if (!spark2_5_full_kv_fp8()) { return 0; }
    return !(v && strcmp(v, "full_only") == 0);
}

static inline int spark2_5_layer_kv_fp8(unsigned il) {
    return spark2_5_layer_full(il)
        ? spark2_5_full_kv_fp8()
        : spark2_5_sliding_kv_fp8();
}

/* By default, restore the 180 VQ matrices once per engine into an FP8 E4M3
 * image with one f16 scale per 128 input columns (3.57 GiB vs 7.03 GiB BF16).
 * DS4_SPARK2_5_WEIGHTS=bf16 (or 0) keeps the BF16 restored image. */
static inline int spark2_5_weights_fp8(void) {
    const char *v = getenv("DS4_SPARK2_5_WEIGHTS");
    return !(v && (strcmp(v, "bf16") == 0 || strcmp(v, "0") == 0));
}

static inline uint64_t spark2_5_fp8_matrix_bytes(uint32_t in_dim,
                                                 uint32_t out_dim) {
    const uint64_t weights = (uint64_t)out_dim * in_dim;
    const uint64_t scales =
        (uint64_t)out_dim * (in_dim / SPARK_FP8_BLOCK) * sizeof(uint16_t);
    return weights + scales;
}

/* One combined FP8 KV row: E4M3 K | E4M3 V | one f16 scale per KV
 * head for each. 2,064 bytes against 4,096 BF16 for this geometry. */
static inline uint64_t spark2_5_fp8_row_bytes(void) {
    return 2ull * SPARK_KV_DIM + 2ull * SPARK_KV_HEADS * sizeof(uint16_t);
}

/* Spark-specific CUDA entries (cuda/spark2_5.cuh, ds4_spark2_5_gpu.cuh). */
int ds4_gpu_spark_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *positions,
    const ds4_gpu_tensor *inv, uint32_t rows, uint32_t heads, uint32_t dim,
    uint32_t pairs);
int ds4_gpu_spark_gelu_mul(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
    const ds4_gpu_tensor *up, uint64_t count);
int ds4_gpu_spark_qkv_split(ds4_gpu_tensor *q, ds4_gpu_tensor *k,
    ds4_gpu_tensor *v, const ds4_gpu_tensor *qkv, uint32_t rows);
/* Restore one GVQ1 matrix into BF16 rows; `vq2`/`book2` are the stage-2
 * stream and book of a container version 3 release, NULL for version 2. */
int ds4_gpu_spark_restore(ds4_gpu_tensor *out,
    const void *vq, uint64_t vq_bytes, const void *vq2, uint64_t vq2_bytes,
    const void *book, const void *book2, const void *res, uint64_t res_bytes,
    const void *res_book, uint32_t rows, uint32_t cols);
/* Restore one GVQ1 matrix into FP8 E4M3 rows + one f16 scale per 128 cols. */
int ds4_gpu_spark_restore_fp8(ds4_gpu_tensor *out,
    const void *vq, uint64_t vq_bytes, const void *vq2, uint64_t vq2_bytes,
    const void *book, const void *book2, const void *res, uint64_t res_bytes,
    const void *res_book, uint32_t rows, uint32_t cols);
/* Multiply x [n_tok][in_dim] F32 by an FP8 E4M3 + block-128 scale matrix
 * [out_dim][in_dim] -> out [n_tok][out_dim] F32. */
int ds4_gpu_spark_matmul_fp8(ds4_gpu_tensor *out, const ds4_gpu_tensor *w_fp8,
    uint32_t in_dim, uint32_t out_dim, const ds4_gpu_tensor *x, uint32_t n_tok);
/* Quantize a BF16 matrix [rows][cols] from host/mapped memory into an FP8 E4M3
 * + block-128 f16 scale tensor on the device. */
int ds4_gpu_spark_bf16_to_fp8(ds4_gpu_tensor *dst, const void *src_bf16,
    uint32_t rows, uint32_t cols);
/* Gather and dequantize `n_tokens` rows from an FP8 E4M3 + block-128 scale
 * embedding table [n_vocab][n_embd] into `out` [n_tokens][n_embd] F32. */
int ds4_gpu_spark_embed_fp8(ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens,
    const ds4_gpu_tensor *table_fp8, uint32_t n_vocab, uint32_t n_tokens,
    uint32_t n_embd);
/* Spark's own FP8 E4M3 KV store and causal GQA attention (sliding and full). */
int ds4_gpu_spark_kv_store_fp8(ds4_gpu_tensor *dst, const ds4_gpu_tensor *k,
    const ds4_gpu_tensor *v, const ds4_gpu_tensor *positions, uint32_t rows,
    uint32_t width, uint32_t kv_heads, uint32_t head_dim, uint32_t ring,
    uint32_t capacity);
int ds4_gpu_spark_attn_fp8(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
    const ds4_gpu_tensor *kv, const ds4_gpu_tensor *positions, uint32_t rows,
    uint32_t first, uint32_t heads, uint32_t kv_heads, uint32_t dim,
    uint32_t window, uint32_t ring, uint32_t keys, uint64_t row_bytes);
/* Decode one GVQ1 matrix inside the GEMM, so no BF16 copy of W exists
 * (cuda/spark2_5_vq_gemm.cuh). The stream pointers are resolved once at bind
 * from the artifact arena; `out` is [n_tok][rows] F32, `x` [n_tok][cols]. */
typedef struct {
    const void *gain, *main, *plane;             /* stage 1 */
    const void *gain2, *main2, *plane2;          /* stage 2, NULL on v2 */
    const void *book, *book2, *res, *res_book;
    uint32_t rows, cols;
} ds4_spark_vq_ref;

int ds4_gpu_spark_vq_resolve(ds4_spark_vq_ref *ref, const void *map,
    const void *arena, const void *vq, uint64_t vq_bytes, const void *vq2,
    uint64_t vq2_bytes, const void *book, const void *book2, const void *res,
    uint64_t res_bytes, const void *res_book, uint32_t rows, uint32_t cols);
int ds4_gpu_spark_vq_linear(ds4_gpu_tensor *out, const ds4_spark_vq_ref *w,
    const ds4_gpu_tensor *x, uint32_t n_tok);
/* The artifact arena the fused path decodes from, and its one-shot fill. */
int ds4_gpu_spark_codes_copy(ds4_gpu_tensor *dst, const void *src,
    uint64_t bytes);
const void *ds4_gpu_spark_codes_base(const ds4_gpu_tensor *codes);
/* Diagnostic: the fused path's decode into a BF16 matrix. */
int ds4_gpu_spark_vq_probe(ds4_gpu_tensor *out, const ds4_spark_vq_ref *w);

#ifdef __cplusplus
}
#endif

#endif
