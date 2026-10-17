#ifndef DS4_LING31_H
#define DS4_LING31_H

#include <stdint.h>

#include "ds4_glmvq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Ling-3.1-flash VQ device runtime. Weights stay in the GVQ pointer plan
 * (Rust validated every part); this adds what GLM's adapter does not cover:
 * 512 x (4096 -> 1280 -> 4096) routed experts on K4096 (12-bit) or K8192
 * (13-bit) layer books, Ling's post-SiLU SwiGLU clamp, and the MLA absorb
 * weights restored once from the FP8 kv_b. */
typedef struct ds4_l31_moe ds4_l31_moe;

enum { DS4_L31_EXPERTS = 512, DS4_L31_USED = 8, DS4_L31_HIDDEN = 4096, DS4_L31_FFN = 1280,
       DS4_L31_HEADS = 64, DS4_L31_LATENT = 512, DS4_L31_NOPE = 128, DS4_L31_ROPE = 64,
       DS4_L31_VALUE = 128, DS4_L31_MOE_LAYERS = 69 };

/* Routed-expert scratch for `rows` (prefill FP8 planes, decode/verify items
 * and the per-layer device routing tables). */
uint64_t ds4_gpu_l31_moe_bytes(uint32_t rows);
ds4_l31_moe *ds4_gpu_l31_moe_create(ds4_gvq_ctx *gvq, uint32_t rows);
void ds4_gpu_l31_moe_free(ds4_l31_moe *moe);
/* `matrices` holds the layer's gate[512], up[512], down[512] GVQ indices.
 * `shared` (optional) is the shared expert's output, added after the routed
 * F32 sum is rounded. `clamp` is +inf on unclamped layers. */
int ds4_gpu_l31_moe(ds4_l31_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
    const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
    const uint32_t *matrices, uint32_t rows, float clamp, const ds4_gpu_tensor *shared);

/* SwiGLU and fused Q4_K gate/up (decode/verify widths) in Ling's clamp form. */
int ds4_gpu_l31_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
    const ds4_gpu_tensor *up, uint32_t rows, uint32_t width, float limit, ds4_gvq_rows out_rows);
int ds4_gpu_l31_gate_up(ds4_gvq_ctx *ctx, ds4_gpu_tensor *mid, uint32_t gate, uint32_t up,
    const ds4_gpu_tensor *input, uint32_t rows, float limit);

/* BF16 k_b [heads][latent][nope] and v_b [heads][value][latent] from the FP8
 * kv_b [heads*(nope+value), latent] (FP8 x scale, then BF16 RNE). */
int ds4_gpu_l31_kv_b(ds4_gvq_ctx *ctx, ds4_gpu_tensor *k_b, ds4_gpu_tensor *v_b, uint32_t kv_b);
int ds4_gpu_l31_absorb(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
    const ds4_gpu_tensor *k_b, uint32_t rows);
int ds4_gpu_l31_value(ds4_gpu_tensor *out, const ds4_gpu_tensor *latent,
    const ds4_gpu_tensor *v_b, uint32_t rows);
/* Expanded-MLA prefill operand for latent rows [slot0, slot0 + rows). */
int ds4_gpu_l31_expand(ds4_gpu_tensor *k_full, ds4_gpu_tensor *value,
    const ds4_gpu_tensor *latent_cache, const ds4_gpu_tensor *k_pe_cache,
    const ds4_gpu_tensor *k_b, const ds4_gpu_tensor *v_b, uint32_t slot0, uint32_t rows);
/* RMS norm of each 576-wide kv_a row's first 512 values (F32 weight copy). */
int ds4_gpu_l31_latent_norm(ds4_gpu_tensor *out, const ds4_gpu_tensor *in,
    const ds4_gpu_tensor *weight, uint32_t rows, float eps);
/* cur = bf16(cur + branch) and norm = weighted RMS of cur (BF16 weight
 * `weight`), bit-identical to the separate add, rounding and RMS kernels. */
int ds4_gpu_l31_residual_rms(ds4_gvq_ctx *ctx, ds4_gpu_tensor *cur, const ds4_gpu_tensor *branch,
    ds4_gpu_tensor *norm, uint32_t weight, uint32_t rows, uint32_t width, float eps);

#ifdef __cplusplus
}
#endif
#endif
