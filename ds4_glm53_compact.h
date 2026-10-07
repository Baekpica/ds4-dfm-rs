#ifndef DS4_GLM53_COMPACT_H
#define DS4_GLM53_COMPACT_H

#include <stdint.h>

typedef struct ds4_gpu_tensor ds4_gpu_tensor;

enum {
    DS4_GLM53_POOL_DIM = 128,
    DS4_GLM53_POOL_SIZE = 4,
    DS4_GLM53_INDEX_TOPK = 2048,
    DS4_GLM53_MAX_SELECTED = DS4_GLM53_INDEX_TOPK + DS4_GLM53_POOL_SIZE - 1
};

#ifdef __cplusplus
extern "C" {
#endif

/* Cache rows are FP16; queries and temporary low-rank rows are F32.
 * K-b is [head, latent_dim, head_dim]; V-b is [head, head_dim, latent_dim]. */
int ds4_gpu_glm53_store_low(ds4_gpu_tensor *cache,
        const ds4_gpu_tensor *latent, uint32_t rows, uint32_t pos0,
        uint32_t cap, uint32_t latent_dim);
int ds4_gpu_glm53_absorb_q(ds4_gpu_tensor *low_q,
        const ds4_gpu_tensor *q, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t rows, uint32_t heads,
        uint32_t latent_dim, uint32_t head_dim);
int ds4_gpu_glm53_attn_low(ds4_gpu_tensor *low_out,
        const ds4_gpu_tensor *low_q, const ds4_gpu_tensor *cache,
        const ds4_gpu_tensor *selected, uint32_t sel_stride, uint32_t rows,
        uint32_t pos0, uint32_t cap, uint32_t heads, uint32_t latent_dim,
        uint32_t head_dim);
int ds4_gpu_glm53_proj_v(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *low_out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t rows, uint32_t heads,
        uint32_t latent_dim, uint32_t head_dim);

/* Each four-token pool applies affine LayerNorm to each raw key, then
 * weights it with per-dimension softmax(gate + BF16 APE). Tails are F32
 * [4,128], addressed by absolute position % 4. Pool rows are FP16. */
int ds4_gpu_glm53_pool_key(ds4_gpu_tensor *pool_cache,
        ds4_gpu_tensor *tail_k, ds4_gpu_tensor *tail_gate,
        const ds4_gpu_tensor *raw_k, const ds4_gpu_tensor *gate,
        const void *model_map, uint64_t model_size, uint64_t norm_offset,
        uint64_t bias_offset, uint64_t ape_offset, uint32_t pos0,
        uint32_t rows, uint32_t cap, float eps);
int ds4_gpu_glm53_pool_score(ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *pool_cache, uint32_t n_pools, uint32_t rows,
        uint32_t pos0, uint32_t heads, float scale);
int ds4_gpu_glm53_pool_expand(ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *pool_selected, uint32_t rows, uint32_t pos0,
        uint32_t selected_pools, uint32_t index_topk, uint32_t out_width);

#ifdef __cplusplus
}
#endif
#endif
