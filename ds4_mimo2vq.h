#ifndef DS4_MIMO2VQ_H
#define DS4_MIMO2VQ_H

#include <stdint.h>

#include "ds4_glmvq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* MiMo-V2.6-Flash VQ routed experts. Weights stay in the GVQ pointer plan
 * (Rust validated every part). MiMo's experts have GLM-5.3's shape, 256 x
 * (4096 -> 2048 -> 4096) top-8, on 13-bit K8192 layer books with the GVR8
 * residual stage on 32 layers. Weight arguments are GVQ record indices.
 * Arithmetic contract: the handoff's tools/mimo_ref.py, expert =
 * down(silu(gate(x)) * up(x)) in BF16, routed outputs summed in F32 in
 * ascending expert order and rounded once. */
typedef struct ds4_mimo2vq_moe ds4_mimo2vq_moe;

enum { DS4_MIMO2VQ_EXPERTS = 256, DS4_MIMO2VQ_USED = 8, DS4_MIMO2VQ_MOE_LAYERS = 47 };

/* Routed-expert scratch for `rows` (prefill FP8 planes, decode/verify items
 * and the per-layer device routing tables). */
uint64_t ds4_gpu_mimo2vq_moe_bytes(uint32_t rows);
ds4_mimo2vq_moe *ds4_gpu_mimo2vq_moe_create(ds4_gvq_ctx *gvq, uint32_t rows);
void ds4_gpu_mimo2vq_moe_free(ds4_mimo2vq_moe *moe);
/* `matrices` holds the layer's gate[256], up[256], down[256] GVQ indices.
 * out = bf16(F32 sum of weight x expert row, ascending expert). */
int ds4_gpu_mimo2vq_moe(ds4_mimo2vq_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
    const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, const uint32_t *matrices, uint32_t rows);

#ifdef __cplusplus
}
#endif
#endif
