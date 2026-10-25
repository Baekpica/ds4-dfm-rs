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

/* `h3_gpu_token_pool_bf16` (`h3_gpu_cuda.cu:3297-3349`) with
 * `h3_token_pool_kernel` (`:3271-3295`): snapshot the source rows this pass
 * replaces, average each pair, and store the pooled video rows in the
 * baseline buffer the expansion later subtracts. A pair whose two rows are the
 * same (every non-video row, and an unpaired last column) passes through
 * unchanged. */
int h3d_token_pool(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *input, size_t input_element,
                   h3d_tensor *original, size_t original_element,
                   h3d_tensor *baseline, size_t baseline_element,
                   const h3d_tensor *baseline_indices,
                   const h3d_tensor *pairs, uint32_t input_rows,
                   uint32_t rows, uint32_t baseline_rows, uint32_t width,
                   char *error, size_t error_size);

/* `h3_gpu_token_expand_delta_bf16` (`:3353-3383`) with
 * `h3_token_expand_kernel` (`:3328-3351`): every row takes its parent's
 * reduced value as a delta over the pooled baseline, scaled and added to the
 * snapshot; rows before `prefix_rows` and rows with no baseline index take the
 * reduced value directly. */
int h3d_token_expand(h3d_context *context, h3d_tensor *output,
                     const h3d_tensor *original, size_t original_element,
                     const h3d_tensor *reduced, const h3d_tensor *baseline,
                     size_t baseline_element,
                     const h3d_tensor *baseline_indices,
                     const h3d_tensor *parents, uint32_t rows,
                     uint32_t reduced_rows, uint32_t baseline_rows,
                     uint32_t width, uint32_t prefix_rows, float scale,
                     char *error, size_t error_size);

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

/* ---- F32 primitives the AudioVAE decoder needs (`h3_audio_vae.c`) ---- */

/* `h3_gpu_copy_f32` (`:697-703`). */
int h3d_copy_f32(h3d_context *context, h3d_tensor *destination,
                 size_t destination_element, const h3d_tensor *source,
                 size_t source_element, size_t elements, char *error,
                 size_t error_size);

/* `h3_gpu_conv1d_f32`/`h3_gpu_conv1d_stride_f32` (`:1757-1794`) with
 * `h3_conv1d_kernel` (`:1681-1706`). Layout: [batch, time, channels]. */
int h3d_conv1d(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *input, const h3d_tensor *weight,
               const h3d_tensor *bias, uint32_t batch, uint32_t length,
               uint32_t input_channels, uint32_t output_channels,
               uint32_t kernel, uint32_t stride, uint32_t padding,
               uint32_t dilation, char *error, size_t error_size);

/* `h3_gpu_conv_transpose1d_f32` (`:1796-1840`) with
 * `h3_conv_transpose1d_kernel` (`:1714-1743`). */
int h3d_conv_transpose1d(h3d_context *context, h3d_tensor *output,
                         const h3d_tensor *input, const h3d_tensor *weight,
                         const h3d_tensor *bias, uint32_t batch,
                         uint32_t length, uint32_t input_channels,
                         uint32_t output_channels, uint32_t kernel,
                         uint32_t stride, uint32_t padding, char *error,
                         size_t error_size);

/* `h3_gpu_weight_norm_f32` (`:2976-2992`) with `h3_weight_norm_kernel`
 * (`:2958-2974`): one row per thread, the same fmaf square sum and rsqrtf. */
int h3d_weight_norm(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *vector, const h3d_tensor *magnitude,
                    uint32_t outer, uint32_t inner, char *error,
                    size_t error_size);

/* `h3_gpu_alias_free_snake_f32` (`:1943-1961`) with
 * `h3_alias_free_snake_kernel` (`:1907-1933`). */
int h3d_alias_free_snake(h3d_context *context, h3d_tensor *output,
                         const h3d_tensor *input,
                         const h3d_tensor *alpha_log,
                         const h3d_tensor *beta_log,
                         const h3d_tensor *upsample_filter,
                         const h3d_tensor *downsample_filter, uint32_t batch,
                         uint32_t length, uint32_t channels, char *error,
                         size_t error_size);

/* `h3_gpu_add_scaled_f32` (`:2397-2411`) with `h3_add_scaled_kernel`
 * (`:2385-2391`) and `h3_gpu_clip_f32` (`:2309-2317`), whose clip is the
 * `H3_CLIP` branch of `h3_unary_f32_kernel` (`:2246-2274`). */
int h3d_add_scaled(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *left, const h3d_tensor *right,
                   float left_scale, float right_scale, size_t elements,
                   char *error, size_t error_size);
int h3d_clip(h3d_context *context, h3d_tensor *output,
             const h3d_tensor *input, size_t elements, float minimum,
             float maximum, char *error, size_t error_size);

/* ---- F32 primitives the VideoVAE decoder needs (`h3_video_vae.c`) ---- */

/* `h3_gpu_sdpa_f32` (`:1651-1657`) through `h3_attention_dispatch`
 * (`:1562-1649`): one batch, non-causal, Q/K/V head-major, output row-major.
 * `head_dim == 64` takes `h3_attention_tiled_f32_kernel<64>` (`:1489-1547`),
 * the video VAE's shape; any other width takes the generic
 * `h3_attention_kernel<float>` (`:1337-1416`), as the reference dispatches
 * (`:1623-1632`). */
int h3d_sdpa_f32(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *query, const h3d_tensor *key,
                 const h3d_tensor *value, uint32_t sequence, uint32_t heads,
                 uint32_t head_dim, float scale, char *error,
                 size_t error_size);

/* `h3_gpu_rms_norm_f32` (`:2496-2510`) with `h3_rms_norm_f32_kernel`
 * (`:2438-2460`): 256 threads per row, fmaf square sum, rsqrtf. */
int h3d_rms_norm_f32(h3d_context *context, h3d_tensor *output,
                     const h3d_tensor *input, const h3d_tensor *weight,
                     uint32_t rows, uint32_t width, float epsilon, char *error,
                     size_t error_size);

/* `h3_gpu_layer_norm_f32` (`:2601-2617`) with `h3_layer_norm_f32_kernel`
 * (`:2529-2561`): the mean is a plain running sum, the variance an fmaf sum,
 * and the epilogue keeps the reference's multiply order. */
int h3d_layer_norm_f32(h3d_context *context, h3d_tensor *output,
                       const h3d_tensor *input, const h3d_tensor *weight,
                       const h3d_tensor *bias, uint32_t rows, uint32_t width,
                       float epsilon, char *error, size_t error_size);

/* `h3_gpu_swiglu_f32` (`:2874-2877`) with `h3_swiglu_kernel` (`:2841-2852`):
 * gate and up halves of each fused row, `gate / (1 + expf(-gate)) * up`. */
int h3d_swiglu_f32(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *fused, uint32_t rows, uint32_t width,
                   char *error, size_t error_size);

/* `h3_gpu_scale_add_f32` (`:2895-2910`) with `h3_scale_add_kernel`
 * (`:2885-2893`): `residual + branch * scale`, one multiply, no fma. */
int h3d_scale_add_f32(h3d_context *context, h3d_tensor *output,
                      const h3d_tensor *residual, const h3d_tensor *branch,
                      const h3d_tensor *scale, uint32_t rows, uint32_t width,
                      char *error, size_t error_size);

/* `h3_gpu_video_qkv_rope_f32` (`:3190-3205`) with `h3_qkv_rope_kernel`
 * (`:3078-3136`) at `grouped = 1`, `normalize = 1`, `weighted = 0`: the
 * checkpoint's [head][q,k,v][dim] rows, Q/K RMS over the head with no learned
 * weight, then the pair rotation. This is the video VAE variant, not the
 * DiT's. */
int h3d_video_qkv_rope_f32(h3d_context *context, h3d_tensor *query,
                           h3d_tensor *key, h3d_tensor *value,
                           const h3d_tensor *qkv,
                           const h3d_tensor *rope_cos,
                           const h3d_tensor *rope_sin, uint32_t sequence,
                           uint32_t heads, uint32_t head_dim,
                           uint32_t rope_half, float epsilon, char *error,
                           size_t error_size);

/* ---- F32 primitives the VideoVAE encoder needs (`h3_video_encoder.c`) ---- */

/* `h3_gpu_vae_encoder_pad_f32` (`h3_gpu_cuda.cu:2002-2043`) with
 * `h3_vae_pad_kernel` (`:1974-2000`): `depth_front` zeros in front, then a
 * single-reflection spatial pad (`h3_reflect`, `:1969-1973`). The temporal
 * axis is never reflected: the encoder pads time with zeros only. */
int h3d_vae_encoder_pad(h3d_context *context, h3d_tensor *output,
                        const h3d_tensor *input, uint32_t batch,
                        uint32_t depth, uint32_t height, uint32_t width,
                        uint32_t channels, uint32_t depth_front,
                        uint32_t height_before, uint32_t height_after,
                        uint32_t width_before, uint32_t width_after,
                        char *error, size_t error_size);

/* `h3_gpu_conv3d_f32` (`:2075-2127`) with `h3_conv3d_kernel` (`:2034-2073`):
 * one thread per output element, bias seeded before the first tap, then the
 * reference's fmaf order (input channel, kt, ky, kx). Weights are
 * `[output_channel, input_channel, kt, ky, kx]`. */
int h3d_conv3d(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *input, const h3d_tensor *weight,
               const h3d_tensor *bias, uint32_t batch, uint32_t depth,
               uint32_t height, uint32_t width, uint32_t input_channels,
               uint32_t output_channels, uint32_t kernel_depth,
               uint32_t kernel_height, uint32_t kernel_width,
               uint32_t stride_depth, uint32_t stride_height,
               uint32_t stride_width, char *error, size_t error_size);

/* `h3_gpu_vae_encoder_group_norm_silu_f32` (`:2153-2178`) with
 * `h3_group_norm_silu_kernel` (`:2114-2151`): one thread per
 * (temporal, group) row, spatial-major accumulation inside the group, a plain
 * running sum for the mean, an fmaf variance, then `value / (1 + expf(-value))`
 * with the per-channel weight and bias applied first. */
int h3d_group_norm_silu(h3d_context *context, h3d_tensor *output,
                        const h3d_tensor *input, const h3d_tensor *weight,
                        const h3d_tensor *bias, uint32_t batch,
                        uint32_t depth, uint32_t height, uint32_t width,
                        uint32_t channels, uint32_t groups, float epsilon,
                        char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
