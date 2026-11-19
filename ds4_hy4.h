#ifndef DS4_HY4_H
#define DS4_HY4_H
#include <stdint.h>
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#ifdef __cplusplus
extern "C" {
#endif

/* Indices are logical token slots. Only sliding layers wrap their MLA cache.
 * The short ring holds unpooled IndexPool operands across appends and rewinds. */
typedef struct {
    ds4_gpu_tensor *latent, *rope, *groups, *recent;
    uint32_t cap, recent_cap;
    int fp8;
} ds4_hy4_cache;

/* Element type of a row buffer: the routed down rows the expert sum reads,
 * or a BF16 projection's input rows. */
typedef enum { DS4_HY4_ROWS_F32 = 1, DS4_HY4_ROWS_BF16 } ds4_hy4_rows;
ds4_gpu_tensor *ds4_gpu_hy4_draft_head(const void *map, uint64_t size, uint64_t offset);
int ds4_gpu_hy4_draft_logits(ds4_gpu_tensor *logits, const ds4_gpu_tensor *head, const ds4_gpu_tensor *x);
int ds4_gpu_hy4_rows_bf16(ds4_gpu_tensor *out, const void *map, uint64_t size, uint64_t offset, uint32_t in,
                          uint32_t outs, const ds4_gpu_tensor *x, ds4_hy4_rows input, uint32_t n);

/* Routed gate, up and down expert tensors: [512][1536][1536] Q4_K or Q5_K. */
typedef struct {
    uint64_t offset[3], bytes[3];
    uint32_t type[3];
} ds4_hy4_experts;

const float *ds4_gpu_hy4_weight(const void *map, uint64_t size, uint64_t offset, uint64_t bytes);
int ds4_gpu_hy4_experts(ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
                        const ds4_gpu_tensor *ids, const void *map, uint64_t size, const ds4_hy4_experts *w,
                        uint32_t tokens);
int ds4_gpu_hy4_add(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
                    const float *gate, uint32_t n);
int ds4_gpu_hy4_vision_pos(ds4_gpu_tensor *x, const ds4_gpu_tensor *table, uint32_t gh, uint32_t gw);
int ds4_gpu_hy4_embed(ds4_gpu_tensor *streams, const ds4_gpu_tensor *embed, uint32_t rows);
int ds4_gpu_hy4_hc(ds4_gpu_tensor *out, ds4_gpu_tensor *post, const ds4_gpu_tensor *streams,
                   const ds4_gpu_tensor *mix, const float *scale, const float *base, uint32_t rows);
int ds4_gpu_hy4_hc_step(ds4_gpu_tensor *streams, ds4_gpu_tensor *post, const ds4_gpu_tensor *branch,
                        const float *fn, const float *scale, const float *base, const float *norm_w,
                        ds4_gpu_tensor *norm, ds4_gpu_tensor *norm_bf16, uint32_t rows);
int ds4_gpu_hy4_gate(ds4_gpu_tensor *q, const ds4_gpu_tensor *gate, uint32_t rows, uint32_t width);
int ds4_gpu_hy4_residual(ds4_gpu_tensor *streams, const ds4_gpu_tensor *branch, const ds4_gpu_tensor *post,
                         uint32_t rows);
int ds4_gpu_hy4_unpack(ds4_gpu_tensor *out, const ds4_gpu_tensor *records, const float *norm,
                       uint32_t layer, uint32_t rows);
int ds4_gpu_hy4_engram(ds4_gpu_tensor *streams, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *ids,
                       const float *bias, const float *wk, const float *wq, const float *gamma,
                       const float *scales, uint32_t rows);
int ds4_gpu_hy4_down(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
                     const ds4_gpu_tensor *ids, const void *map, uint64_t size, uint64_t offset,
                     uint64_t bytes, uint32_t type, uint32_t tokens);
int ds4_gpu_hy4_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
                       uint32_t count, float limit);
int ds4_gpu_hy4_moe_sum(ds4_gpu_tensor *out, const ds4_gpu_tensor *down, const ds4_gpu_tensor *weights,
                        uint32_t rows, ds4_hy4_rows type);
int ds4_gpu_hy4_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *positions, uint32_t rows, uint32_t heads,
                     uint32_t stride, uint32_t tail, float theta);
/* Whether a head projection's BF16 operand still has to be packed from F32. */
typedef enum { DS4_HY4_HEAD_F32, DS4_HY4_HEAD_PACKED } ds4_hy4_head_input;
/* A head projection's result type: BF16 when its consumer rounds it anyway. */
typedef enum { DS4_HY4_HEAD_OUT_F32, DS4_HY4_HEAD_OUT_BF16 } ds4_hy4_head_output;
int ds4_gpu_hy4_head_mm(ds4_gpu_tensor *out, ds4_gpu_tensor *pack, const ds4_gpu_tensor *x, const void *map,
                        uint64_t size, uint64_t offset, uint32_t rows, uint32_t heads, uint32_t in,
                        uint32_t stride, uint32_t width, ds4_hy4_head_input input, ds4_hy4_head_output output);
/* Whether rows take the tiled MLA kernel, which reads q_abs as BF16. */
int ds4_gpu_hy4_flash(uint32_t rows, int draft);
int ds4_gpu_hy4_store(ds4_hy4_cache *cache, const ds4_gpu_tensor *raw, const float *norm, uint32_t pos,
                      uint32_t rows);
int ds4_gpu_hy4_pool(ds4_hy4_cache *cache, const ds4_gpu_tensor *values, const ds4_gpu_tensor *gates,
                     const ds4_gpu_tensor *positions, const float *ape, const float *norm, uint32_t pos,
                     uint32_t rows);
int ds4_gpu_hy4_index(ds4_gpu_tensor *selected, ds4_gpu_tensor *tiles, ds4_gpu_tensor *best,
                      const ds4_gpu_tensor *q, const ds4_gpu_tensor *w, const ds4_hy4_cache *cache,
                      uint32_t pos, uint32_t rows);
int ds4_gpu_hy4_attention(ds4_gpu_tensor *out, ds4_gpu_tensor *pack, const ds4_gpu_tensor *q_abs,
                          const ds4_gpu_tensor *q, const ds4_hy4_cache *cache, const ds4_gpu_tensor *selected,
                          const float *sinks, uint32_t pos, uint32_t rows, uint32_t heads, int draft);

#ifdef __cplusplus
}
#endif
#endif
