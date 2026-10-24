/*
 * h3t_* — the seven BF16 primitives the Qwen3-VL prompt encoder needs.
 *
 * Additive MiniMax-H3 family: new names, no existing family's kernel is
 * touched, and nothing here is reachable from another family's path. The
 * reference implementation is `/data/h3.c` at `eae5d54`, `h3_gpu_cuda.cu`;
 * each function below names the reference lines it mirrors.
 *
 * The host loop lives in Rust (`minimax_h3::text_forward`); this file only
 * owns device work. Every arithmetic order is the reference's: a different
 * reduction order would change the BF16 output and break the E4 hash gate.
 *
 * Deviation, named: the reference carries F32, BF16, U32 and patch types in
 * one `h3_gpu_tensor` with bound checking per call. This ABI keeps the same
 * checks but only the two dtypes the text path uses.
 */
#ifndef H3T_TEXT_GPU_H
#define H3T_TEXT_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum h3t_dtype { H3T_BF16 = 0, H3T_F32 = 1 };

typedef struct h3t_context h3t_context;
typedef struct h3t_tensor h3t_tensor;

h3t_context *h3t_create(char *error, size_t error_size);
void h3t_destroy(h3t_context *context);

h3t_tensor *h3t_alloc(h3t_context *context, int dtype, size_t elements,
                      char *error, size_t error_size);
void h3t_free(h3t_tensor *tensor);
int h3t_upload(h3t_context *context, h3t_tensor *tensor, const void *host,
               char *error, size_t error_size);
/* `h3_gpu_tensor_write_bf16_range` (`h3_gpu_cuda.cu:594-604`): copy `elements`
 * values from `host` into the tensor at `offset`. The reference copies with the
 * legacy blocking `cudaMemcpy`; here the same copy is stream-ordered, as
 * `h3t_upload` is, so it lands after the ops already queued. */
int h3t_write(h3t_context *context, h3t_tensor *tensor, size_t offset,
              const void *host, size_t elements, char *error,
              size_t error_size);
int h3t_read(h3t_context *context, void *host, const h3t_tensor *tensor,
             char *error, size_t error_size);

/* `h3_gpu_embedding_bf16` (`h3_gpu_cuda.cu:2822-2839`), ids from host. */
int h3t_embed(h3t_context *context, h3t_tensor *output,
              const h3t_tensor *weight, const uint32_t *ids, uint32_t tokens,
              uint32_t vocab, uint32_t width, char *error, size_t error_size);

/* `h3_gpu_rms_norm_bf16` (`:2512-2530`). */
int h3t_rms_norm(h3t_context *context, h3t_tensor *output,
                 const h3t_tensor *input, const h3t_tensor *weight,
                 uint32_t rows, uint32_t width, float epsilon, char *error,
                 size_t error_size);

/* `h3_gpu_linear_bf16` (`:852-870`) through `h3_linear_lt` (`:733-779`). */
int h3t_linear(h3t_context *context, h3t_tensor *output,
               const h3t_tensor *input, const h3t_tensor *weight,
               uint32_t rows, uint32_t input_dim, uint32_t output_dim,
               char *error, size_t error_size);

/* `h3_gpu_head_rms_norm_bf16` (`:3010-3031`), in place. */
int h3t_head_rms_norm(h3t_context *context, h3t_tensor *tensor,
                      const h3t_tensor *weight, uint32_t sequence,
                      uint32_t heads, uint32_t head_dim, float epsilon,
                      char *error, size_t error_size);

/* `h3_gpu_rope_text_bf16` (`:3055-3081`), in place; tables are F32. */
int h3t_rope_text(h3t_context *context, h3t_tensor *query, h3t_tensor *key,
                  const h3t_tensor *cosines, const h3t_tensor *sines,
                  uint32_t sequence, uint32_t query_heads, uint32_t kv_heads,
                  uint32_t head_dim, char *error, size_t error_size);

/* `h3_gpu_gqa_causal_bf16` (`:1683-1689`) with the row-major,
 * causal, non-head-major dispatch it selects (`:1648-1656`). */
int h3t_gqa_causal(h3t_context *context, h3t_tensor *output,
                   const h3t_tensor *query, const h3t_tensor *key,
                   const h3t_tensor *value, uint32_t sequence,
                   uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim,
                   float scale, char *error, size_t error_size);

/* `h3_gpu_add_bf16` (`:2373-2376`) and `h3_gpu_silu_mul_bf16`
 * (`:2383-2387`), both through `h3_binary_bf16_kernel` (`:2347-2357`). */
int h3t_add(h3t_context *context, h3t_tensor *output,
            const h3t_tensor *left, const h3t_tensor *right, size_t elements,
            char *error, size_t error_size);
int h3t_silu_mul(h3t_context *context, h3t_tensor *output,
                 const h3t_tensor *gate, const h3t_tensor *up,
                 size_t elements, char *error, size_t error_size);

/* Wait for every queued op; returns 0 and fills `error` on failure. */
int h3t_sync(h3t_context *context, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
