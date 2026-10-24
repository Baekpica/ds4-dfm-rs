/*
 * h3d_* — the BF16 primitives the FL2VA DiT needs.
 *
 * Additive MiniMax-H3 family: new names, no other family's kernel touched, and
 * nothing here is reachable from another family's path. The reference is
 * `/data/h3.c` at `eae5d54`, `h3_gpu_cuda.cu` (`h3_gpu.h` for the signatures);
 * each function names the reference lines it mirrors.
 *
 * The host loop lives in Rust (`minimax_h3::dit`); this file owns device work
 * only. The block graph is:
 *
 *   hidden --adaln(norm1, slots 0/1)--> mod_attention
 *          --linear(qkv) + qk-norm + RoPE--> q,k,v  (head-major)
 *          --full attention--> heads --linear(out)--> attention_output
 *          --gate(residual hidden, slot 2) + adaln(norm2, slots 3/4)--> mod_mlp
 *          --linear(fc1) + swiglu + linear(fc2)--> mlp_output
 *          --gate(slot 5)--> hidden
 *
 * Arithmetic order is the reference's and is copied expression for expression,
 * including leaving the fused multiply-add decisions to the same compiler: the
 * E5 gate hashes the BF16 output, so a different reduction tree or an explicit
 * `fmaf` where the reference relies on contraction changes the last bit.
 *
 * Deviation, named: the reference carries F32, BF16, U32 and int8 tensors in
 * one `h3_gpu_tensor`; this ABI keeps the two dtypes the BF16 path uses and
 * refuses the int8/fused-NAX variants (CUDA returns 0 from
 * `h3_gpu_has_int8_mlp`/`h3_gpu_has_nax_mlp`, `h3_gpu_cuda.cu:332-334`, so the
 * reference's own CUDA path is the one implemented here).
 */
#ifndef H3D_DIT_GPU_H
#define H3D_DIT_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum h3d_dtype { H3D_BF16 = 0, H3D_F32 = 1, H3D_U32 = 2 };

typedef struct h3d_context h3d_context;
typedef struct h3d_tensor h3d_tensor;

h3d_context *h3d_create(char *error, size_t error_size);
void h3d_destroy(h3d_context *context);

h3d_tensor *h3d_alloc(h3d_context *context, int dtype, size_t elements,
                      char *error, size_t error_size);
void h3d_free(h3d_tensor *tensor);

/* Whole-tensor upload, then a range upload for the per-step latents. */
int h3d_upload(h3d_context *context, h3d_tensor *tensor, const void *host,
               char *error, size_t error_size);
int h3d_upload_at(h3d_context *context, h3d_tensor *tensor, size_t element,
                  const void *host, size_t elements, char *error,
                  size_t error_size);
int h3d_read(h3d_context *context, void *host, const h3d_tensor *tensor,
             char *error, size_t error_size);

/* `h3_gpu_linear_bf16`/`h3_gpu_linear_f32` (`:832-870`) through `h3_linear_lt`
 * (`:733-779`) and `h3_linear_bias` (`:783-804`). `bias` may be NULL; the
 * bias add happens after the matmul rounds to the output dtype. */
int h3d_linear(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *input, const h3d_tensor *weight,
               const h3d_tensor *bias, uint32_t rows, uint32_t input_dim,
               uint32_t output_dim, int dtype, char *error, size_t error_size);

/* `h3_gpu_patch_linear_bf16_offset` (`:872-906`) and `h3_patch_convert_kernel`
 * (`:808-829`): an F32 matmul with F32 weights, bias added in F32, one BF16
 * rounding at the end. */
int h3d_patch_linear(h3d_context *context, h3d_tensor *output,
                     size_t output_element, const h3d_tensor *input,
                     size_t input_element, const h3d_tensor *weight,
                     const h3d_tensor *bias, uint32_t rows,
                     uint32_t input_dim, uint32_t output_dim, char *error,
                     size_t error_size);

/* `h3_gpu_patch_linear_bf16_map` (`:932-970`) with `h3_patch_scatter_kernel`
 * (`:917-930`): the same projection scattered to the rows a multi-segment
 * layout assigns it, which is what a keyframe or reference layout needs. */
int h3d_patch_linear_map(h3d_context *context, h3d_tensor *output,
                         const h3d_tensor *input, const h3d_tensor *weight,
                         const h3d_tensor *bias, const h3d_tensor *row_map,
                         uint32_t output_rows, uint32_t rows,
                         uint32_t input_dim, uint32_t output_dim, char *error,
                         size_t error_size);

/* `h3_gpu_adaln_bf16_offset` (`:2737-2755`) with `h3_adaln_kernel`
 * (`:2656-2687`): one row per block, thread 0 owns the RMS sum. */
int h3d_adaln(h3d_context *context, h3d_tensor *output,
              const h3d_tensor *input, size_t input_element,
              const h3d_tensor *norm_weight, const h3d_tensor *modulation,
              const h3d_tensor *row_map, uint32_t rows, uint32_t width,
              uint32_t slots, uint32_t shift_slot, uint32_t scale_slot,
              float epsilon, char *error, size_t error_size);

/* `h3_gpu_gate_bf16` (`:2803-2809`) with `h3_gate_kernel` (`:2689-2702`). */
int h3d_gate(h3d_context *context, h3d_tensor *output,
             const h3d_tensor *residual, const h3d_tensor *branch,
             const h3d_tensor *modulation, const h3d_tensor *row_map,
             uint32_t rows, uint32_t width, uint32_t slots,
             uint32_t gate_slot, char *error, size_t error_size);

/* `h3_gpu_rms_norm_bf16` (`:2512-2530`) with `h3_rms_norm_bf16_kernel`
 * (`:2462-2489`). */
int h3d_rms_norm(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *input, const h3d_tensor *weight,
                 uint32_t rows, uint32_t width, float epsilon, char *error,
                 size_t error_size);

/* `h3_gpu_swiglu_bf16` (`:2879-2883`) with `h3_swiglu_kernel` (`:2841-2855`). */
int h3d_swiglu(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *fused, uint32_t rows, uint32_t width,
               char *error, size_t error_size);

/* `h3_gpu_grouped_qkv_rope_bf16` (`:3241-3249`) with `h3_qkv_rope_kernel`
 * (`:3079-3136`): the checkpoint's [head, q/k/v, dim] rows, Q/K head RMS norm,
 * then the pair rotation over the first `2 * rope_half` dimensions. */
int h3d_qkv_rope(h3d_context *context, h3d_tensor *query, h3d_tensor *key,
                 h3d_tensor *value, const h3d_tensor *qkv,
                 const h3d_tensor *q_norm, const h3d_tensor *k_norm,
                 const h3d_tensor *rope_cos, const h3d_tensor *rope_sin,
                 uint32_t sequence, uint32_t heads, uint32_t head_dim,
                 uint32_t rope_half, float epsilon, char *error,
                 size_t error_size);

/* `h3_gpu_sdpa_bf16` (`:1667-1672`) taking the CUDA tiled dispatch
 * (`:1620-1631`) with `h3_attention_tiled_bf16_kernel` (`:1418-1487`):
 * one block per (head, 8 rows), head-major input, row-major output. */
int h3d_sdpa(h3d_context *context, h3d_tensor *output,
             const h3d_tensor *query, const h3d_tensor *key,
             const h3d_tensor *value, uint32_t sequence, uint32_t heads,
             uint32_t head_dim, float scale, char *error, size_t error_size);

/* `h3_gpu_add_bf16` (`:2373-2376`) with `h3_binary_bf16_kernel`
 * (`:2347-2357`), `h3_gpu_cast_f32_to_bf16` (`:2287-2296`), and the SiLU
 * branches of `h3_gpu_silu_f32`/`h3_gpu_silu_bf16` (`:2276-2306`, `:2321-2329`)
 * the timestep embedding needs. */
int h3d_add(h3d_context *context, h3d_tensor *output,
            const h3d_tensor *left, const h3d_tensor *right, size_t elements,
            char *error, size_t error_size);
int h3d_cast_f32_to_bf16(h3d_context *context, h3d_tensor *output,
                         const h3d_tensor *input, size_t elements, char *error,
                         size_t error_size);
int h3d_silu_f32(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *input, size_t elements, char *error,
                 size_t error_size);
int h3d_silu_bf16(h3d_context *context, h3d_tensor *output,
                  const h3d_tensor *input, size_t elements, char *error,
                  size_t error_size);

/* `h3_gpu_copy_bf16` (`:690-696`) via `h3_copy` (`:672-688`). */
int h3d_copy(h3d_context *context, h3d_tensor *destination,
             size_t destination_element, const h3d_tensor *source,
             size_t source_element, size_t elements, char *error,
             size_t error_size);

/* Wait for every queued op; returns 0 and fills `error` on failure. */
int h3d_sync(h3d_context *context, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
