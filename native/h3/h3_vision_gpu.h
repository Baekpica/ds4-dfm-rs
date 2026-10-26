/*
 * h3v_* — the six BF16 primitives the Qwen3-VL vision tower needs.
 *
 * Additive MiniMax-H3 family: new names, its own translation unit, so neither
 * the prompt encoder's gated path (`h3_text_gpu.cu`) nor the DiT's is touched
 * or reachable from here. Source of truth `/data/h3.c` at `eae5d54`,
 * `h3_gpu_cuda.cu`; every function below names the reference lines it mirrors.
 *
 * The host loop lives in Rust (`minimax_h3::vision_forward`); this file only
 * owns device work. Every arithmetic order is the reference's: a different
 * reduction tree, a fused bias, or a different cuBLASLt descriptor set changes
 * the last bit of a BF16 value and fails the vision gate.
 *
 * Deviation, named: the reference carries F32, BF16, U32 and patch types in
 * one `h3_gpu_tensor`. The vision tower reads and writes BF16 only, so this
 * ABI keeps one dtype.
 */
#ifndef H3V_VISION_GPU_H
#define H3V_VISION_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct h3v_context h3v_context;
typedef struct h3v_tensor h3v_tensor;

h3v_context *h3v_create(char *error, size_t error_size);
void h3v_destroy(h3v_context *context);

h3v_tensor *h3v_alloc(h3v_context *context, size_t elements, char *error,
                      size_t error_size);
void h3v_free(h3v_tensor *tensor);
int h3v_upload(h3v_context *context, h3v_tensor *tensor, const void *host,
               char *error, size_t error_size);
int h3v_read(h3v_context *context, void *host, const h3v_tensor *tensor,
             char *error, size_t error_size);

/* `h3_gpu_linear_bf16` (`:852-870`) through `h3_linear_lt` (`:733-779`) and
 * the separate `h3_linear_bias` kernel (`:792-810`, kernel `:783-790`). The bias is a second
 * launch on purpose: fusing it into the epilogue would round once instead of
 * twice. */
int h3v_linear(h3v_context *context, h3v_tensor *output,
               const h3v_tensor *input, const h3v_tensor *weight,
               const h3v_tensor *bias, uint32_t rows, uint32_t input_dim,
               uint32_t output_dim, char *error, size_t error_size);

/* `h3_gpu_layer_norm_bf16` (`:2619-2637`) and its kernel (`:2563-2594`):
 * 256 threads, one block per row, weight and bias included. */
int h3v_layer_norm(h3v_context *context, h3v_tensor *output,
                   const h3v_tensor *input, const h3v_tensor *weight,
                   const h3v_tensor *bias, uint32_t rows, uint32_t width,
                   float epsilon, char *error, size_t error_size);

/* `h3_gpu_gelu_bf16` (`:2332-2343`) with the two branches of
 * `h3_unary_bf16_kernel` (`:2257-2278`): `approximate` for the blocks,
 * exact for the mergers (`h3_vision_encoder.c:326`, `:397`). */
int h3v_gelu(h3v_context *context, h3v_tensor *output, const h3v_tensor *input,
             uint32_t elements, int approximate, char *error,
             size_t error_size);

/* `h3_gpu_add_bf16` (`:2373-2376`) through `h3_binary_bf16_kernel`
 * (`:2347-2357`). */
int h3v_add(h3v_context *context, h3v_tensor *output, const h3v_tensor *left,
            const h3v_tensor *right, uint32_t elements, char *error,
            size_t error_size);

/* `h3_gpu_vision_qkv_rope_bf16` (`:3252-3267`) and `h3_qkv_rope_kernel`
 * (`:3078-3137`) with its vision branch set: no q/k normalization, no
 * per-dimension weights, head-major output. */
int h3v_qkv_rope(h3v_context *context, h3v_tensor *query, h3v_tensor *key,
                 h3v_tensor *value, const h3v_tensor *qkv,
                 const h3v_tensor *cosines, const h3v_tensor *sines,
                 uint32_t sequence, uint32_t heads, uint32_t head_dim,
                 uint32_t rope_half, char *error, size_t error_size);

/* `h3_gpu_sdpa_bf16` (`:1667-1672`): non-causal, batch 1, every head its own
 * KV head, which dispatches to `h3_attention_kernel` (`:1338-1440`). */
int h3v_sdpa(h3v_context *context, h3v_tensor *output, const h3v_tensor *query,
             const h3v_tensor *key, const h3v_tensor *value, uint32_t sequence,
             uint32_t heads, uint32_t head_dim, float scale, char *error,
             size_t error_size);

/* Wait for every queued op; returns 0 and fills `error` on failure. */
int h3v_sync(h3v_context *context, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
