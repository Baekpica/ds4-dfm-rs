#ifndef DS4_GEMMA4_GPU_CUH
#define DS4_GEMMA4_GPU_CUH

/* Gemma-4 device entry points. The graph lives in ds4_gemma4_graph.inc
 * (host side, inside ds4.c); this file is the device side only: the KV store,
 * the routed experts, and tensor-shaped wrappers over the primitives.
 *
 * The shared GVQ runtime supplies the linear, the norm-free plumbing and the
 * allocator. Everything the runtime does not have lives here or in
 * cuda/gemma4_{primitives,attn,moe}.cuh, and every kernel cites
 * docs/gemma4-reference-semantics.md. */

#include "ds4_gemma4.h"
#include "cuda/gemma4_primitives.cuh"
#include "cuda/gemma4_head.cuh"
#include "cuda/gemma4_router.cuh"
#include "cuda/gemma4_attn.cuh"
#include "cuda/gemma4_attn_fp8.cuh"
#include "cuda/gemma4_attn_fast.cuh"
#include "cuda/gemma4_attn_tile.cuh"
#include "cuda/gemma_vq_decode.cuh"
#include "cuda/gemma_vq_prefill.cuh"
#include "cuda/gemma4_attn_mma.cuh"
#include "cuda/gemma4_attn_split.cuh"
#include "cuda/gemma4_moe_fused.cuh"
#include "cuda/gemma4_moe_persist.cuh"
#include "cuda/gemma4_q4_decode.cuh"

/* One routed-expert scratch set, sized for `chunk` tokens and reused across
 * layers: gather rows, the three projections of the route set, and the
 * device order table the host grouping writes each layer. The direct VQ
 * prefill pass adds the layer's work-item list and the uploaded matrix
 * table; both are built on the host per layer and are bounded by the chunk. */
struct ds4_g4_moe {
    ds4_gvq_ctx *gvq;
    uint32_t chunk;
    ds4_gpu_tensor *in, *out, *gate, *up, *mid, *order;
    int32_t *routes;
    uint32_t *grouped;
    ds4_gpu_tensor *tokpair;    /* per token, its G4_USED grouped rows in expert order */
    uint32_t *htokpair;         /* host staging for the same list */
    uint32_t htokfill[G4_PREFILL_MAX];
    ds4_gpu_tensor *items;      /* g4_vqp_item[items_cap] on the device */
    ds4_gpu_tensor *qmat;       /* glmvq_matrix[3 * G4_EXPERTS] on the device */
    g4_vqp_item *hitems;        /* host staging for the same list */
    glmvq_matrix *hqmat;        /* host staging for the matrix table */
    uint32_t items_cap;
    /* Persistent decode (cuda/gemma4_moe_persist.cuh): the gate/up output in
     * BF16, the per-layer matrix table uploaded on its first decode call,
     * and the host copy of each layer's indices that decides when to upload
     * again. The table is static while the experts stay resident; an SSD
     * rebind changes the descriptors and the copy check re-uploads. */
    ds4_gpu_tensor *pmid;       /* G4_USED x G4_FF BF16 */
    ds4_gpu_tensor *ptab;       /* G4_LAYERS x 3 x G4_EXPERTS glmvq_matrix */
    uint32_t *hbind;            /* G4_LAYERS x 3 x G4_EXPERTS indices */
    uint8_t hbind_ok[G4_LAYERS];
};

/* ---------------------------------------------------------------- KV store */

/* Write the projected, normed and roped K/V rows of `rows` tokens into their
 * bf16 cache slots. `ring` 0 means the slot is the absolute position. */
__global__ static void g4_kv_store_kernel(__nv_bfloat16 *dst, const float *src,
        const uint32_t *positions, uint32_t rows, uint32_t width, uint32_t ring) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * width) { return; }
    const uint32_t row = (uint32_t)(i / width);
    const uint32_t slot = ring ? (positions[row] % ring) : positions[row];
    dst[(uint64_t)slot * width + (uint32_t)(i % width)] = __float2bfloat16_rn(src[i]);
}

extern "C" int ds4_gpu_g4_kv_store(ds4_gpu_tensor *dst, const ds4_gpu_tensor *src,
        const ds4_gpu_tensor *positions, uint32_t rows, uint32_t width, uint32_t ring,
        uint32_t capacity) {
    const uint64_t count = (uint64_t)rows * width;
    if (!dst || !src || !positions || !count || !width || !capacity) { return 0; }
    if (dst->bytes < (uint64_t)capacity * width * sizeof(__nv_bfloat16) ||
        src->bytes < count * sizeof(float) || positions->bytes < (uint64_t)rows * sizeof(uint32_t)) {
        return 0;
    }
    const uint32_t blocks = (uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
    g4_kv_store_kernel<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
        (__nv_bfloat16 *)dst->ptr, (const float *)src->ptr, (const uint32_t *)positions->ptr,
        rows, width, ring);
    return cuda_ok(cudaGetLastError(), "Gemma-4 KV store");
}

/* --------------------------------------------------------- FP8 full-layer KV */

/* E4M3 FN value of a magnitude code (ds4_cuda.cu:11876-11881) and the exact
 * encode the Solar FP8 store uses (ds4_cuda.cu:44324-44343): a binary search
 * over the 127 magnitudes with round-to-even on a tie, then the sign bit. */
__device__ __forceinline__ static float g4_e4m3_value(int code) {
    const int exp = (code >> 3) & 15;
    const int mant = code & 7;
    if (exp == 0) { return (float)mant * 0.001953125f; }
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

__device__ __forceinline__ static uint8_t g4_e4m3_encode(float x) {
    if (!isfinite(x) || x == 0.0f) { return 0u; }
    const uint8_t sign = x < 0.0f ? 0x80u : 0u;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (g4_e4m3_value(mid) <= ax) { lo = mid; } else { hi = mid - 1; }
    }
    int best = lo;
    if (best < 126) {
        const float bd = fabsf(ax - g4_e4m3_value(best));
        const float nd = fabsf(ax - g4_e4m3_value(best + 1));
        if (nd < bd ||
            (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) {
            best++;
        }
    }
    return sign | (uint8_t)best;
}

/* f16(amax / quant_max), returned rounded so the codes divide by the value
 * the row stores (ds4_cuda.cu:44369-44383). */
__device__ __forceinline__ static float g4_scale_to_f16(
        float amax, float quant_max, __half *stored) {
    float scale = isfinite(amax) && amax > 0.0f ? amax / quant_max : 1.0f;
    if (scale > 65504.0f) { scale = 65504.0f; }
    __half half_scale = __float2half(scale);
    float rounded = __half2float(half_scale);
    if (!(rounded > 0.0f) || !isfinite(rounded)) {
        half_scale = __float2half(5.960464477539063e-8f);
        rounded = __half2float(half_scale);
    }
    *stored = half_scale;
    return rounded;
}

/* One block per (token, KV head): amax over the head's dims, one scale for K
 * and one for V, then the E4M3 codes. The row is K | V | scales
 * (ds4_cuda.cu:44407-44480). */
__global__ static void g4_kv_store_fp8_kernel(uint8_t *dst, const float *k,
        const float *v, const uint32_t *positions, uint32_t rows,
        uint32_t kv_heads, uint32_t dim, uint32_t ring) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (token >= rows || head >= kv_heads) { return; }
    const uint32_t tid = threadIdx.x;
    const uint64_t kv_dim = (uint64_t)kv_heads * dim;
    const uint64_t src = ((uint64_t)token * kv_heads + head) * dim;
    const uint32_t slot = ring ? positions[token] % ring : positions[token];
    const uint64_t row_bytes = 2u * kv_dim + 2u * (uint64_t)kv_heads * sizeof(uint16_t);
    uint8_t *row = dst + (uint64_t)slot * row_bytes;
    __shared__ float k_abs[256];
    __shared__ float v_abs[256];
    float kmax = 0.0f, vmax = 0.0f;
    for (uint32_t d = tid; d < dim; d += blockDim.x) {
        const float kx = k[src + d];
        const float vx = v[src + d];
        if (isfinite(kx)) { kmax = fmaxf(kmax, fabsf(kx)); }
        if (isfinite(vx)) { vmax = fmaxf(vmax, fabsf(vx)); }
    }
    k_abs[tid] = kmax;
    v_abs[tid] = vmax;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            k_abs[tid] = fmaxf(k_abs[tid], k_abs[tid + stride]);
            v_abs[tid] = fmaxf(v_abs[tid], v_abs[tid + stride]);
        }
        __syncthreads();
    }
    __shared__ float kscale, vscale;
    __half *scales = (__half *)(row + 2u * kv_dim);
    if (tid == 0u) {
        kscale = g4_scale_to_f16(k_abs[0], 448.0f, &scales[head]);
        vscale = g4_scale_to_f16(v_abs[0], 448.0f, &scales[kv_heads + head]);
    }
    __syncthreads();
    const uint64_t head0 = (uint64_t)head * dim;
    for (uint32_t d = tid; d < dim; d += blockDim.x) {
        row[head0 + d] = g4_e4m3_encode(k[src + d] / kscale);
        row[kv_dim + head0 + d] = g4_e4m3_encode(v[src + d] / vscale);
    }
}

extern "C" int ds4_gpu_g4_kv_store_fp8(ds4_gpu_tensor *dst, const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v, const ds4_gpu_tensor *positions, uint32_t rows,
        uint32_t width, uint32_t kv_heads, uint32_t head_dim, uint32_t ring,
        uint32_t capacity) {
    if (!dst || !k || !v || !positions || !rows || !width || !kv_heads || !head_dim ||
        !capacity || width != kv_heads * head_dim) { return 0; }
    const uint64_t row_bytes = 2ull * width + 2ull * kv_heads * sizeof(uint16_t);
    if (dst->bytes < (uint64_t)capacity * row_bytes ||
        k->bytes < (uint64_t)rows * width * sizeof(float) ||
        v->bytes < (uint64_t)rows * width * sizeof(float) ||
        positions->bytes < (uint64_t)rows * sizeof(uint32_t)) { return 0; }
    g4_kv_store_fp8_kernel<<<dim3(rows, kv_heads), 256, 0, ds4_current_stream()>>>(
        (uint8_t *)dst->ptr, (const float *)k->ptr, (const float *)v->ptr,
        (const uint32_t *)positions->ptr, rows, kv_heads, head_dim, ring);
    return cuda_ok(cudaGetLastError(), "Gemma-4 FP8 KV store");
}

/* -------------------------------------------------------------- primitives */

/* `weight` is a plan index (DS4_GVQ_NO_LINK for the weightless forms); the
 * pointer comes from the GVQ source table, still owned by the plan and stored
 * BF16, so every consumer widens per element. */
static const __nv_bfloat16 *g4_weight(const ds4_gvq_ctx *ctx, uint32_t index, uint64_t count) {
    if (!ctx || index == DS4_GVQ_NO_LINK) { return nullptr; }
    return (const __nv_bfloat16 *)gvq_weight(ctx, index, GVQ_TYPE_BF16, count);
}

extern "C" int ds4_gpu_g4_rms(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *in,
        uint32_t weight, uint32_t rows, uint32_t width, float eps) {
    const __nv_bfloat16 *w = g4_weight(ctx, weight, width);
    if (!out || !in || !rows || !width || (weight != DS4_GVQ_NO_LINK && !w)) { return 0; }
    if (out->bytes < (uint64_t)rows * width * sizeof(float) ||
        in->bytes < (uint64_t)rows * width * sizeof(float)) { return 0; }
    g4_rms_kernel<256><<<rows, 256, 0, ds4_current_stream()>>>((float *)out->ptr,
        (const float *)in->ptr, w, rows, width, eps);
    return cuda_ok(cudaGetLastError(), "Gemma-4 RMS norm");
}

extern "C" int ds4_gpu_g4_router_input(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out,
        const ds4_gpu_tensor *in, uint32_t scale, uint32_t rows, uint32_t width, float eps,
        float root) {
    const __nv_bfloat16 *s = g4_weight(ctx, scale, width);
    if (!out || !in || !s || !rows || !width) { return 0; }
    if (out->bytes < (uint64_t)rows * width * sizeof(float) ||
        in->bytes < (uint64_t)rows * width * sizeof(float)) { return 0; }
    g4_router_input_kernel<256><<<rows, 256, 0, ds4_current_stream()>>>((float *)out->ptr,
        (const float *)in->ptr, s, rows, width, eps, root);
    return cuda_ok(cudaGetLastError(), "Gemma-4 router input");
}

extern "C" int ds4_gpu_g4_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *inv, uint32_t rows, uint32_t heads, uint32_t dim, uint32_t pairs) {
    if (!x || !positions || !inv || !rows || !heads || !dim) { return 0; }
    if (x->bytes < (uint64_t)rows * heads * dim * sizeof(float) ||
        positions->bytes < (uint64_t)rows * sizeof(uint32_t) ||
        inv->bytes < (uint64_t)(dim / 2u) * sizeof(float)) { return 0; }
    const uint64_t jobs = (uint64_t)rows * heads;
    const uint32_t blocks = (uint32_t)((jobs + 127u) / 128u);
    g4_rope_kernel<<<blocks, 128, 0, ds4_current_stream()>>>((float *)x->ptr,
        (const uint32_t *)positions->ptr, (const float *)inv->ptr, rows, heads, dim, pairs);
    return cuda_ok(cudaGetLastError(), "Gemma-4 rope");
}

/* The norm and the rope in one block per (row, head): the row is staged once
 * and rotated in place, with g4_rms_kernel's and g4_rope_kernel's arithmetic
 * unchanged (cuda/gemma4_primitives.cuh). */
extern "C" int ds4_gpu_g4_rms_rope(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *in,
        uint32_t weight, const ds4_gpu_tensor *positions, const ds4_gpu_tensor *inv,
        uint32_t rows, uint32_t heads, uint32_t dim, uint32_t pairs, float eps) {
    const __nv_bfloat16 *w = g4_weight(ctx, weight, dim);
    if (!out || !in || !positions || !inv || !rows || !heads || !dim || !pairs) { return 0; }
    if (weight != DS4_GVQ_NO_LINK && !w) { return 0; }
    if (out->bytes < (uint64_t)rows * heads * dim * sizeof(float) ||
        in->bytes < (uint64_t)rows * heads * dim * sizeof(float) ||
        positions->bytes < (uint64_t)rows * sizeof(uint32_t) ||
        inv->bytes < (uint64_t)(dim / 2u) * sizeof(float)) { return 0; }
    const dim3 grid(heads, rows);
    g4_rms_rope_kernel<256><<<grid, 256, 0, ds4_current_stream()>>>((float *)out->ptr,
        (const float *)in->ptr, w, (const uint32_t *)positions->ptr,
        (const float *)inv->ptr, rows, heads, dim, pairs, eps);
    return cuda_ok(cudaGetLastError(), "Gemma-4 RMS norm + rope");
}

/* Parallel top-k in the router selection, off by default
 * (DS4_GEMMA4_ROUTER_PAR=1); the serial scan stays the reference. */
static bool g4_router_par(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_ROUTER_PAR");
        on = v && v[0] == '1' && !v[1];
    }
    return on != 0;
}

extern "C" int ds4_gpu_g4_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected,
        ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits, uint32_t per_expert,
        uint32_t rows) {
    const __nv_bfloat16 *scale = g4_weight(ctx, per_expert, G4_EXPERTS);
    if (!selected || !weights || !logits || !scale || !rows) { return 0; }
    if (selected->bytes < (uint64_t)rows * G4_USED * sizeof(uint32_t) ||
        weights->bytes < (uint64_t)rows * G4_USED * sizeof(float) ||
        logits->bytes < (uint64_t)rows * G4_EXPERTS * sizeof(float)) { return 0; }
    if (g4_router_par()) {
        g4_router_kernel<true><<<rows, G4_EXPERTS, 0, ds4_current_stream()>>>((uint32_t *)selected->ptr,
            (float *)weights->ptr, (const float *)logits->ptr, scale, rows, G4_EXPERTS, G4_USED);
    } else {
        g4_router_kernel<false><<<rows, G4_EXPERTS, 0, ds4_current_stream()>>>((uint32_t *)selected->ptr,
            (float *)weights->ptr, (const float *)logits->ptr, scale, rows, G4_EXPERTS, G4_USED);
    }
    return cuda_ok(cudaGetLastError(), "Gemma-4 router");
}

/* The router projection through the family's own BF16 dot
 * (cuda/gemma4_router.cuh) for one token; 0 for any other shape or record,
 * so the shared linear stays the fallback. */
extern "C" int ds4_gpu_g4_router_logits(ds4_gvq_ctx *ctx, ds4_gpu_tensor *logits, uint32_t index,
        const ds4_gpu_tensor *input, uint32_t rows) {
    if (!ctx || !logits || !input || rows != 1u || index >= ctx->model->source.size()) { return 0; }
    const glmvq_source &s = ctx->model->source[index];
    if (ctx->model->records[index].type != GVQ_TYPE_BF16 || s.storage != GLMVQ_WEIGHT_BF16 ||
        !s.data || s.rows != G4_EXPERTS || s.cols != G4_EMBED) {
        return 0;
    }
    if (input->bytes < (uint64_t)s.cols * sizeof(float) ||
        logits->bytes < (uint64_t)s.rows * sizeof(float)) {
        return 0;
    }
    g4_router_dot_kernel<<<s.rows, 256, 0, ds4_current_stream()>>>(
        (const __nv_bfloat16 *)s.data, (const float *)input->ptr, (float *)logits->ptr,
        s.rows, s.cols);
    return cuda_ok(cudaGetLastError(), "Gemma-4 router dot");
}

extern "C" int ds4_gpu_g4_gelu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up, uint64_t count) {
    if (!out || !gate || !up || !count) { return 0; }
    if (out->bytes < count * sizeof(float) || gate->bytes < count * sizeof(float) ||
        up->bytes < count * sizeof(float)) { return 0; }
    const uint32_t blocks = (uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
    g4_gelu_kernel<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>((float *)out->ptr,
        (const float *)gate->ptr, (const float *)up->ptr, count);
    return cuda_ok(cudaGetLastError(), "Gemma-4 gelu product");
}

extern "C" int ds4_gpu_g4_embed_scale(ds4_gpu_tensor *x, uint64_t count) {
    if (!x || !count || x->bytes < count * sizeof(float)) { return 0; }
    const uint32_t blocks = (uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
    g4_embed_kernel<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>((float *)x->ptr,
        (const float *)x->ptr, count);
    return cuda_ok(cudaGetLastError(), "Gemma-4 embedding scale");
}

extern "C" int ds4_gpu_g4_softcap(ds4_gpu_tensor *logits, uint64_t count) {
    if (!logits || !count || logits->bytes < count * sizeof(float)) { return 0; }
    const uint32_t blocks = (uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
    g4_softcap_kernel<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>((float *)logits->ptr,
        count, (float)G4_REF_SOFTCAP);
    return cuda_ok(cudaGetLastError(), "Gemma-4 logit softcap");
}

/* The vocabulary head through the family's own BF16 GEMV
 * (cuda/gemma4_head.cuh), with the softcap fused into the epilogue. Returns 0
 * when the record is not a BF16 table of the expected width, so the caller
 * falls back to the shared linear and its separate softcap. */
extern "C" int ds4_gpu_g4_head(ds4_gvq_ctx *ctx, ds4_gpu_tensor *logits, uint32_t index,
        const ds4_gpu_tensor *input, float cap) {
    if (!ctx || !logits || !input || index >= ctx->model->source.size()) { return 0; }
    const glmvq_source &s = ctx->model->source[index];
    if (ctx->model->records[index].type != GVQ_TYPE_BF16 || s.storage != GLMVQ_WEIGHT_BF16 ||
        !s.data || s.cols != G4_EMBED || !s.rows) {
        return 0;
    }
    if (input->bytes < (uint64_t)s.cols * sizeof(float) ||
        logits->bytes < (uint64_t)s.rows * sizeof(float)) {
        return 0;
    }
    static int sms = 0;
    if (sms <= 0) {
        int n = 0;
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess || n <= 0) {
            return 0;
        }
        sms = n;
    }
    uint32_t blocks = (uint32_t)((s.rows + 7u) / 8u);
    const uint32_t cap_blocks = (uint32_t)sms * 4u;
    if (blocks > cap_blocks) { blocks = cap_blocks; }
    g4_head_gemv_kernel<G4_EMBED, true><<<blocks, 256, 0, ds4_current_stream()>>>(
        (const __nv_bfloat16 *)s.data, (const float *)input->ptr, (float *)logits->ptr,
        s.rows, cap);
    return cuda_ok(cudaGetLastError(), "Gemma-4 vocabulary head");
}

/* The verify head: up to G4_HEAD_ROWS final-norm rows against the tied BF16
 * table in one pass (cuda/gemma4_head.cuh). Row t writes
 * logits[t * vocab + row], the layout of the drafter's verify tensor. Returns
 * 0 for any shape the fused kernel cannot take, so the caller keeps the
 * per-row shared linear. */
extern "C" int ds4_gpu_g4_head_rows(ds4_gvq_ctx *ctx, ds4_gpu_tensor *logits,
        uint32_t index, const ds4_gpu_tensor *input, uint32_t rows, float cap) {
    if (!ctx || !logits || !input || !rows || rows > (uint32_t)G4_HEAD_ROWS ||
        index >= ctx->model->source.size()) {
        return 0;
    }
    const glmvq_source &s = ctx->model->source[index];
    if (ctx->model->records[index].type != GVQ_TYPE_BF16 || s.storage != GLMVQ_WEIGHT_BF16 ||
        !s.data || s.cols != G4_EMBED || !s.rows) {
        return 0;
    }
    if (input->bytes < (uint64_t)rows * s.cols * sizeof(float) ||
        logits->bytes < (uint64_t)rows * s.rows * sizeof(float)) {
        return 0;
    }
    static int sms = 0;
    if (sms <= 0) {
        int n = 0;
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess || n <= 0) {
            return 0;
        }
        sms = n;
    }
    /* The block stages every input row once; the grid cap follows the shared
     * memory it needs, so the one-row shape keeps the four blocks per SM of
     * the single-row head and the eight-row shape gets two. */
    uint32_t per_sm = 98304u / (rows * (uint32_t)G4_EMBED * 2u);
    if (per_sm > 4u) { per_sm = 4u; }
    if (!per_sm) { per_sm = 1u; }
    uint32_t blocks = (uint32_t)((s.rows + 7u) / 8u);
    const uint32_t cap_blocks = (uint32_t)sms * per_sm;
    if (blocks > cap_blocks) { blocks = cap_blocks; }
#define G4_HEAD_ROWS_CASE(N) \
    case N: \
        g4_head_gemv_rows_kernel<G4_EMBED, N, true><<<blocks, 256, 0, ds4_current_stream()>>>( \
            (const __nv_bfloat16 *)s.data, (const float *)input->ptr, (float *)logits->ptr, \
            (uint32_t)s.rows, cap); \
        break
    switch (rows) {
    G4_HEAD_ROWS_CASE(1);
    G4_HEAD_ROWS_CASE(2);
    G4_HEAD_ROWS_CASE(3);
    G4_HEAD_ROWS_CASE(4);
    G4_HEAD_ROWS_CASE(5);
    G4_HEAD_ROWS_CASE(6);
    G4_HEAD_ROWS_CASE(7);
    G4_HEAD_ROWS_CASE(8);
    default: return 0;
    }
#undef G4_HEAD_ROWS_CASE
    return cuda_ok(cudaGetLastError(), "Gemma-4 verify head");
}

extern "C" int ds4_gpu_g4_layer_scale(ds4_gpu_tensor *x, uint64_t count, float scalar) {
    if (!x || !count || x->bytes < count * sizeof(float)) { return 0; }
    const uint32_t blocks = (uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
    g4_layer_scale_kernel<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>((float *)x->ptr,
        count, scalar);
    return cuda_ok(cudaGetLastError(), "Gemma-4 layer scalar");
}

extern "C" int ds4_gpu_g4_attn(ds4_gpu_tensor *out, const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *key, const ds4_gpu_tensor *value,
        const ds4_gpu_tensor *positions, uint32_t rows, uint32_t first,
        uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t window, uint32_t ring,
        uint32_t keys) {
    if (!out || !query || !key || !value || !positions || !rows || !heads || !kv_heads || !dim) {
        return 0;
    }
    const uint64_t kv_stride = (uint64_t)kv_heads * dim;
    const uint64_t capacity = ring ? ring : (uint64_t)first + rows;
    /* `keys` 0 is first + rows; anything else must fit the cache and cover at
     * least one key, or the softmax would divide by a zero total. */
    if (keys && (!keys || keys > first + rows)) { return 0; }
    if (out->bytes < (uint64_t)rows * heads * dim * sizeof(float) ||
        query->bytes < (uint64_t)rows * heads * dim * sizeof(float) ||
        key->bytes < capacity * kv_stride * sizeof(__nv_bfloat16) ||
        value->bytes < capacity * kv_stride * sizeof(__nv_bfloat16) ||
        positions->bytes < (uint64_t)rows * sizeof(uint32_t)) { return 0; }
    const uint32_t warps = 4u;
    const uint64_t jobs = (uint64_t)rows * heads;
    const uint32_t blocks = (uint32_t)((jobs + warps - 1u) / warps);
    g4_attn_kernel<<<blocks, warps * 32u, 0, ds4_current_stream()>>>((float *)out->ptr,
        (const float *)query->ptr, (const __nv_bfloat16 *)key->ptr,
        (const __nv_bfloat16 *)value->ptr, rows, first, heads, kv_heads, dim, window, ring,
        keys, kv_stride);
    return cuda_ok(cudaGetLastError(), "Gemma-4 attention");
}

extern "C" int ds4_gpu_g4_attn_fp8(ds4_gpu_tensor *out, const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *kv, const ds4_gpu_tensor *positions, uint32_t rows,
        uint32_t first, uint32_t heads, uint32_t kv_heads, uint32_t dim,
        uint32_t window, uint32_t ring, uint32_t keys, uint64_t row_bytes) {
    if (!out || !query || !kv || !positions || !rows || !heads || !kv_heads || !dim ||
        !row_bytes) { return 0; }
    const uint64_t kv_dim = (uint64_t)kv_heads * dim;
    const uint64_t capacity = ring ? ring : (uint64_t)first + rows;
    if (row_bytes < 2u * kv_dim + 2u * kv_heads * sizeof(uint16_t)) { return 0; }
    if (keys && (!keys || keys > first + rows)) { return 0; }
    if (out->bytes < (uint64_t)rows * heads * dim * sizeof(float) ||
        query->bytes < (uint64_t)rows * heads * dim * sizeof(float) ||
        kv->bytes < capacity * row_bytes ||
        positions->bytes < (uint64_t)rows * sizeof(uint32_t)) { return 0; }
    const uint32_t warps = 4u;
    const uint64_t jobs = (uint64_t)rows * heads;
    const uint32_t blocks = (uint32_t)((jobs + warps - 1u) / warps);
    g4_attn_fp8_kernel<<<blocks, warps * 32u, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)query->ptr, (const uint8_t *)kv->ptr,
        rows, first, heads, kv_heads, dim, window, ring, keys, row_bytes);
    return cuda_ok(cudaGetLastError(), "Gemma-4 FP8 attention");
}

extern "C" int ds4_gpu_g4_attn_fast(ds4_gpu_tensor *out, const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *key, const ds4_gpu_tensor *value,
        const ds4_gpu_tensor *positions, uint32_t rows, uint32_t first,
        uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t window,
        uint32_t ring, uint32_t keys) {
    /* `keys` is allowed here for the same reason the oracle allows it: the
     * MTP drafter queries position `start` while the trunk holds rows
     * 0..start-1, so the kernel must stop at keys-1. The kernel's own
     * `last = keys && keys <= position ? keys - 1 : position` already does
     * that; the old guard refused the drafter and forced it onto the oracle. */
    if (!out || !query || !key || !value || !positions || rows != 1u ||
        (keys && keys > first + rows) ||
        !heads || !kv_heads || !dim || dim > 512u || (heads % kv_heads)) { return 0; }
    const uint32_t group = heads / kv_heads;
    if (group == 0u || group > 16u || group * 32u > 512u) { return 0; }
    const uint64_t kv_stride = (uint64_t)kv_heads * dim;
    const uint64_t capacity = ring ? ring : (uint64_t)first + rows;
    if (out->bytes < (uint64_t)heads * dim * sizeof(float) ||
        query->bytes < (uint64_t)heads * dim * sizeof(float) ||
        key->bytes < capacity * kv_stride * sizeof(__nv_bfloat16) ||
        value->bytes < capacity * kv_stride * sizeof(__nv_bfloat16) ||
        positions->bytes < sizeof(uint32_t)) { return 0; }
    const uint32_t shared = 2u * 16u * dim * sizeof(__nv_bfloat16);
    g4_attn_fast_kernel<16><<<kv_heads, group * 32u, shared, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)query->ptr,
        (const __nv_bfloat16 *)key->ptr, (const __nv_bfloat16 *)value->ptr,
        first, heads, kv_heads, dim, window, ring, keys, kv_stride);
    return cuda_ok(cudaGetLastError(), "Gemma-4 fast attention");
}

/* Prefill fast attention: one CTA per (row tile, KV head), the tile staged
 * once for the block's warps (cuda/gemma4_attn_tile.cuh). Off by default;
 * DS4_GEMMA4_ATTN_PREFILL=1 selects it for rows > 1, and the oracle stays
 * the reference. Same arithmetic, so the exact-prefill gate still holds. */
extern "C" int ds4_gpu_g4_attn_tile(ds4_gpu_tensor *out, const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *key, const ds4_gpu_tensor *value,
        const ds4_gpu_tensor *positions, uint32_t rows, uint32_t first,
        uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t window,
        uint32_t ring, uint32_t keys) {
    if (!out || !query || !key || !value || !positions || rows < 2u || !heads ||
        !kv_heads || !dim || dim > 512u || (dim % 8u) || (heads % kv_heads)) { return 0; }
    const uint32_t group = heads / kv_heads;
    if (group == 0u || group > 16u) { return 0; }
    const uint64_t kv_stride = (uint64_t)kv_heads * dim;
    const uint64_t capacity = ring ? ring : (uint64_t)first + rows;
    if (keys && (!keys || keys > first + rows)) { return 0; }
    if (out->bytes < (uint64_t)rows * heads * dim * sizeof(float) ||
        query->bytes < (uint64_t)rows * heads * dim * sizeof(float) ||
        key->bytes < capacity * kv_stride * sizeof(__nv_bfloat16) ||
        value->bytes < capacity * kv_stride * sizeof(__nv_bfloat16) ||
        positions->bytes < (uint64_t)rows * sizeof(uint32_t)) { return 0; }
    /* 8 warps per block: 4 rows x 2 heads (sliding) or 1 row x 8 heads
     * (full). The tile is 16 KiB either way (16 keys at dim 256, 8 at 512). */
    const uint32_t rtile = group <= 2u ? 4u : (group <= 4u ? 2u : 1u);
    const uint32_t warps = rtile * group;
    const uint32_t blocks = (rows + rtile - 1u) / rtile;
    if (dim <= 256u) {
        const uint32_t shared = 2u * 16u * dim * sizeof(__nv_bfloat16);
        g4_attn_tile_kernel<16><<<dim3(kv_heads, blocks), warps * 32u, shared,
            ds4_current_stream()>>>((float *)out->ptr, (const float *)query->ptr,
            (const __nv_bfloat16 *)key->ptr, (const __nv_bfloat16 *)value->ptr,
            rows, first, heads, kv_heads, dim, window, ring, keys, kv_stride, rtile);
    } else {
        const uint32_t shared = 2u * 8u * dim * sizeof(__nv_bfloat16);
        g4_attn_tile_kernel<8><<<dim3(kv_heads, blocks), warps * 32u, shared,
            ds4_current_stream()>>>((float *)out->ptr, (const float *)query->ptr,
            (const __nv_bfloat16 *)key->ptr, (const __nv_bfloat16 *)value->ptr,
            rows, first, heads, kv_heads, dim, window, ring, keys, kv_stride, rtile);
    }
    return cuda_ok(cudaGetLastError(), "Gemma-4 tiled prefill attention");
}

/* Tensor-core prefill attention (Phase 8): the two specializations of
 * cuda/gemma4_attn_mma.cuh. Only the trunk's prefill chunk (keys = 0) and
 * rows >= 16; the MTP verify widths (2..8) stay on the bit-exact tile
 * kernel, and the parity gate is the model's greedy/frontier check. */
extern "C" int ds4_gpu_g4_attn_mma(ds4_gpu_tensor *out, const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *key, const ds4_gpu_tensor *value,
        const ds4_gpu_tensor *positions, uint32_t rows, uint32_t first,
        uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t window,
        uint32_t ring, uint32_t keys) {
    if (!out || !query || !key || !value || !positions || rows < 16u || keys ||
        heads != 16u || !kv_heads || (heads % kv_heads) || (dim != 256u && dim != 512u)) {
        return 0;
    }
    const uint64_t kv_stride = (uint64_t)kv_heads * dim;
    const uint64_t capacity = ring ? ring : (uint64_t)first + rows;
    if (out->bytes < (uint64_t)rows * heads * dim * sizeof(float) ||
        query->bytes < (uint64_t)rows * heads * dim * sizeof(float) ||
        key->bytes < capacity * kv_stride * sizeof(__nv_bfloat16) ||
        value->bytes < capacity * kv_stride * sizeof(__nv_bfloat16) ||
        positions->bytes < (uint64_t)rows * sizeof(uint32_t)) { return 0; }
    if (dim == 256u && kv_heads == 8u) {
        const uint32_t shared = 2u * G4_MMA_TK * dim * sizeof(__nv_bfloat16);
        const uint32_t rtile = 64u;
        g4_attn_mma_slide_kernel<<<dim3(kv_heads, (rows + rtile - 1u) / rtile),
            G4_MMA_THREADS, shared, ds4_current_stream()>>>((float *)out->ptr,
            (const float *)query->ptr, (const __nv_bfloat16 *)key->ptr,
            (const __nv_bfloat16 *)value->ptr, rows, first, heads, kv_heads,
            window, ring, keys, kv_stride);
    } else if (dim == 512u && kv_heads == 2u) {
        const uint32_t shared = G4_MMA_TK * dim * sizeof(__nv_bfloat16) +
            4u * 2u * 16u * 32u * sizeof(float);
        const uint32_t rtile = 8u;
        g4_attn_mma_full_kernel<<<dim3(kv_heads, (rows + rtile - 1u) / rtile),
            G4_MMA_THREADS, shared, ds4_current_stream()>>>((float *)out->ptr,
            (const float *)query->ptr, (const __nv_bfloat16 *)key->ptr,
            (const __nv_bfloat16 *)value->ptr, rows, first, heads, kv_heads,
            window, ring, keys, kv_stride);
    } else {
        return 0;
    }
    return cuda_ok(cudaGetLastError(), "Gemma-4 MMA attention");
}

/* Key span per split CTA (DS4_GEMMA4_ATTN_SPLIT_SPAN, default 128). The
 * split count is ceil(visible keys / span); 512 was the pre-knob value.
 * Measured on the GB10 (2026-11-04): 128 beats 512 at ctx 256 (35.8 vs
 * 29.1 tok/s), 1,024 (34.2 vs 24.7) and 13,447 (30.7 vs 23.0); 64, 32 and
 * 16 measured still better at 256 and 1,024 but are unmeasured at 13,447,
 * so 128 is the default until that point is taken. Clamped to [16, 4096]. */
static uint32_t g4_split_span(void) {
    static int span = -1;
    if (span < 0) {
        const char *v = getenv("DS4_GEMMA4_ATTN_SPLIT_SPAN");
        span = v ? atoi(v) : 0;
        if (span <= 0) { span = 128; }
        if (span < 16) { span = 16; }
        if (span > 4096) { span = 4096; }
    }
    return (uint32_t)span;
}

/* The combine kernel's weight array is dynamic shared memory; above the
 * 48 KiB default the launch needs the larger limit granted once. */
static bool g4_split_combine_ready(void) {
    static int ready = -1;
    if (ready < 0) {
        ready = cudaFuncSetAttribute(g4_attn_split_combine_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 64u * 1024u) == cudaSuccess;
    }
    return ready != 0;
}

/* Split-K decode attention (cuda/gemma4_attn_split.cuh): the fast kernel's
 * per-key arithmetic over `splits` key ranges, joined by a combine kernel.
 * One row only; the drafter's keys=start case is included. The scratch is
 * one process-wide buffer, grown to the largest geometry the session asks
 * for (the engine serializes calls). */
extern "C" int ds4_gpu_g4_attn_split(ds4_gpu_tensor *out, const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *key, const ds4_gpu_tensor *value,
        const ds4_gpu_tensor *positions, uint32_t rows, uint32_t first,
        uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t window,
        uint32_t ring, uint32_t keys) {
    if (!out || !query || !key || !value || !positions || rows != 1u ||
        !heads || !kv_heads || !dim || dim > 512u || (heads % kv_heads)) { return 0; }
    const uint32_t group = heads / kv_heads;
    if (group == 0u || group > 16u || group * 32u > 512u) { return 0; }
    const uint64_t kv_stride = (uint64_t)kv_heads * dim;
    const uint64_t capacity = ring ? ring : (uint64_t)first + rows;
    if (out->bytes < (uint64_t)heads * dim * sizeof(float) ||
        query->bytes < (uint64_t)heads * dim * sizeof(float) ||
        key->bytes < capacity * kv_stride * sizeof(__nv_bfloat16) ||
        value->bytes < capacity * kv_stride * sizeof(__nv_bfloat16) ||
        positions->bytes < sizeof(uint32_t)) { return 0; }
    const uint32_t begin = window && first + 1u > window ? first + 1u - window : 0u;
    const uint32_t last = keys && keys <= first ? keys - 1u : first;
    if (begin > last) { return 0; }
    const uint32_t span = g4_split_span();
    uint32_t splits = (last - begin + span) / span;
    if (splits < 1u) { splits = 1u; }
    const uint32_t combine_shared = splits * (uint32_t)sizeof(float);
    if (combine_shared > 48u * 1024u && !g4_split_combine_ready()) { return 0; }
    const uint64_t partial_count = (uint64_t)splits * heads * dim;
    const uint64_t stats_count = (uint64_t)splits * heads * 2u;
    static float *scratch = nullptr;
    static uint64_t scratch_floats = 0;
    if (partial_count + stats_count > scratch_floats) {
        if (scratch) { cudaFree(scratch); scratch = nullptr; }
        if (cudaMalloc((void **)&scratch,
                (partial_count + stats_count) * sizeof(float)) != cudaSuccess) {
            return 0;
        }
        scratch_floats = partial_count + stats_count;
    }
    float *partial = scratch;
    float *stats = scratch + partial_count;
    const uint32_t shared = 2u * G4_SPLIT_TILE * dim * sizeof(__nv_bfloat16);
    g4_attn_split_kernel<G4_SPLIT_TILE><<<dim3(kv_heads, splits), group * 32u,
        shared, ds4_current_stream()>>>(partial, stats, (const float *)query->ptr,
        (const __nv_bfloat16 *)key->ptr, (const __nv_bfloat16 *)value->ptr,
        first, heads, kv_heads, dim, window, ring, keys, kv_stride, splits);
    if (!cuda_ok(cudaGetLastError(), "Gemma-4 split attention")) { return 0; }
    g4_attn_split_combine_kernel<<<heads, 256u, combine_shared, ds4_current_stream()>>>(
        (float *)out->ptr, partial, stats, heads, dim, splits);
    return cuda_ok(cudaGetLastError(), "Gemma-4 split attention combine");
}

/* ------------------------------------------------------- routed experts */

uint64_t ds4_gpu_g4_moe_bytes(uint32_t chunk) {
    if (!chunk || chunk > G4_PREFILL_MAX) { return 0; }
    const uint64_t pairs = (uint64_t)chunk * G4_USED;
    return pairs * G4_EMBED * sizeof(float) * 2u + pairs * G4_FF * sizeof(float) * 3u +
        pairs * sizeof(uint32_t) + pairs * G4_USED * sizeof(uint32_t) +
        pairs * sizeof(int32_t) + pairs * sizeof(float) +
        (uint64_t)G4_LAYERS * 3u * G4_EXPERTS * sizeof(glmvq_matrix) +
        (uint64_t)G4_USED * G4_FF * sizeof(__nv_bfloat16);
}

ds4_g4_moe *ds4_gpu_g4_moe_create(ds4_gvq_ctx *gvq, uint32_t chunk) {
    if (!gvq || !chunk || chunk > G4_PREFILL_MAX) { return nullptr; }
    ds4_g4_moe *m = (ds4_g4_moe *)calloc(1, sizeof(*m));
    if (!m) { return nullptr; }
    m->gvq = gvq;
    m->chunk = chunk;
    const uint64_t pairs = (uint64_t)chunk * G4_USED;
    m->in = ds4_gpu_tensor_alloc(pairs * G4_EMBED * sizeof(float));
    m->out = ds4_gpu_tensor_alloc(pairs * G4_EMBED * sizeof(float));
    m->gate = ds4_gpu_tensor_alloc(pairs * G4_FF * sizeof(float));
    m->up = ds4_gpu_tensor_alloc(pairs * G4_FF * sizeof(float));
    m->mid = ds4_gpu_tensor_alloc(pairs * G4_FF * sizeof(float));
    m->order = ds4_gpu_tensor_alloc(pairs * sizeof(uint32_t));
    m->routes = (int32_t *)calloc((size_t)pairs, sizeof(int32_t));
    m->grouped = (uint32_t *)calloc((size_t)pairs, sizeof(uint32_t));
    m->tokpair = ds4_gpu_tensor_alloc(pairs * sizeof(uint32_t));
    m->htokpair = (uint32_t *)calloc((size_t)pairs, sizeof(uint32_t));
    /* Every expert's token tiles sum to at most pairs / BN + one per expert,
     * so the layer list is bounded by that many token tiles times the
     * matrix's row tiles (the down projection has the most). */
    const uint32_t token_tiles = (uint32_t)(pairs / G4_VQP_BN) + G4_EXPERTS;
    const uint32_t row_tiles = (G4_EMBED + G4_VQP_BM - 1u) / G4_VQP_BM;
    m->items_cap = token_tiles * row_tiles;
    m->items = ds4_gpu_tensor_alloc((uint64_t)m->items_cap * sizeof(g4_vqp_item));
    m->qmat = ds4_gpu_tensor_alloc(3u * G4_EXPERTS * sizeof(glmvq_matrix));
    m->hitems = (g4_vqp_item *)calloc(m->items_cap, sizeof(g4_vqp_item));
    m->hqmat = (glmvq_matrix *)calloc(3u * G4_EXPERTS, sizeof(glmvq_matrix));
    m->pmid = ds4_gpu_tensor_alloc((uint64_t)G4_USED * G4_FF * sizeof(__nv_bfloat16));
    m->ptab = ds4_gpu_tensor_alloc((uint64_t)G4_LAYERS * 3u * G4_EXPERTS *
        sizeof(glmvq_matrix));
    m->hbind = (uint32_t *)calloc((size_t)G4_LAYERS * 3u * G4_EXPERTS, sizeof(uint32_t));
    if (!m->in || !m->out || !m->gate || !m->up || !m->mid || !m->order ||
        !m->routes || !m->grouped || !m->tokpair || !m->htokpair || !m->items || !m->qmat ||
        !m->hitems || !m->hqmat || !m->pmid || !m->ptab || !m->hbind) {
        ds4_gpu_g4_moe_free(m);
        return nullptr;
    }
    return m;
}

void ds4_gpu_g4_moe_free(ds4_g4_moe *m) {
    if (!m) { return; }
    ds4_gpu_tensor_free(m->in);
    ds4_gpu_tensor_free(m->out);
    ds4_gpu_tensor_free(m->gate);
    ds4_gpu_tensor_free(m->up);
    ds4_gpu_tensor_free(m->mid);
    ds4_gpu_tensor_free(m->order);
    ds4_gpu_tensor_free(m->tokpair);
    ds4_gpu_tensor_free(m->items);
    ds4_gpu_tensor_free(m->qmat);
    ds4_gpu_tensor_free(m->pmid);
    ds4_gpu_tensor_free(m->ptab);
    free(m->routes);
    free(m->grouped);
    free(m->htokpair);
    free(m->hitems);
    free(m->hqmat);
    free(m->hbind);
    free(m);
}

/* --------------------------------------------------- expert gather/scatter */

__global__ static void g4_moe_gather_kernel(float *dst, const float *src, const uint32_t *pairs,
        uint32_t base, uint64_t elements, uint32_t width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) { return; }
    const uint64_t row = i / width;
    const uint32_t pair = pairs[base + row];
    dst[i] = src[(uint64_t)(pair / G4_USED) * width + (i % width)];
}

/* acc = bf16(acc + down * weight): the reference's fp32 product and its
 * rounded index_add_ into a bf16 accumulator (semantics 3.7). */
__global__ static void g4_moe_scatter_kernel(float *dst, const float *src, const float *weights,
        const uint32_t *pairs, uint32_t base, uint64_t elements, uint32_t width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) { return; }
    const uint64_t row = i / width;
    const uint32_t pair = pairs[base + row];
    const uint32_t column = (uint32_t)(i % width);
    const uint64_t target = (uint64_t)(pair / G4_USED) * width + column;
    dst[target] = g4_round(dst[target] + src[i] * weights[pair]);
}

/* The same accumulation over every expert of one layer in ONE launch, with
 * one thread per output element. The racy form above is only correct when a
 * launch carries a single expert's rows (each target is then touched once);
 * over all pairs at once, G4_USED threads would read-modify-write the same
 * output row, so the order of the eight contributions and lost updates would
 * vary run to run. That is not a theoretical concern: two identical
 * direct-prefill runs on the GB10 differed by relL2 0.085 to 0.125 on the
 * frontier logits while the per-expert path was bit-reproducible.
 *
 * `tokpair` lists, for each token, its G4_USED grouped rows in ASCENDING
 * EXPERT order (the host builds it by walking the grouped list, which is
 * expert-major). The reference loop scatters expert 0..127 in that same
 * order, so this kernel reproduces its sequence of rounded additions
 * exactly: acc starts at the caller's zero fill, then
 * acc = bf16(acc + down * weight) eight times, and the single store is the
 * value the eighth reference launch would have left. */
__global__ static void g4_moe_scatter_ordered_kernel(float *dst, const float *src,
        const float *weights, const uint32_t *order, const uint32_t *tokpair,
        uint64_t elements, uint32_t width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) { return; }
    const uint32_t token = (uint32_t)(i / width);
    const uint32_t column = (uint32_t)(i % width);
    float acc = 0.0f;
    for (uint32_t j = 0; j < G4_USED; j++) {
        const uint32_t row = tokpair[(uint64_t)token * G4_USED + j];
        acc = g4_round(acc + src[(uint64_t)row * width + column] * weights[order[row]]);
    }
    dst[(uint64_t)token * width + column] = acc;
}

/* Gemma-4 decode projection: one row reads the GVQ1 bytes directly
 * (cuda/gemma_vq_decode.cuh) instead of restoring the matrix to bf16 and
 * calling cuBLAS. The restore stays the fallback for every geometry the
 * kernel does not take, and DS4_GEMMA4_VQ_GEMV=0 disables the direct path.
 * Variant and width are the knobs a per-GPU profile will set; today they are
 * environment variables (variant 0 auto, 1 warp, 2 block; width 128/256/512). */
static bool g4_moe_linear_one(ds4_g4_moe *m, ds4_gpu_tensor *out, uint32_t index,
        const ds4_gpu_tensor *input) {
    if (!m || !m->gvq || !out || !input || index >= m->gvq->model->source.size()) { return false; }
    const char *off = getenv("DS4_GEMMA4_VQ_GEMV");
    if (off && off[0] == '0' && !off[1]) { return false; }
    const glmvq_source &s = m->gvq->model->source[index];
    if (s.storage != GLMVQ_WEIGHT_VQ || !s.vq.main || !s.vq.book || !s.vq.gain) { return false; }
    int variant = G4_VQ_AUTO, threads = 256;
    const char *v = getenv("DS4_GEMMA4_VQ_GEMV_VARIANT");
    if (v && *v) { variant = atoi(v); }
    const char *t = getenv("DS4_GEMMA4_VQ_GEMV_THREADS");
    if (t && *t) { threads = atoi(t); }
    return g4_vq_gemv(s.vq, (const float *)input->ptr, (float *)out->ptr,
        variant, threads, ds4_current_stream()) != 0;
}

/* Fused gate/up + GELU for one route. Off by default; DS4_GEMMA4_MOE_FUSED=1
 * turns it on, and the separate two-GEMV path stays the reference. */
static bool g4_moe_fused(void) {
    static int fused = -1;
    if (fused < 0) {
        const char *v = getenv("DS4_GEMMA4_MOE_FUSED");
        fused = v && v[0] == '1' && !v[1];
    }
    return fused != 0;
}

static bool g4_moe_linear_glu_one(ds4_g4_moe *m, ds4_gpu_tensor *out, uint32_t gate,
        uint32_t up, const ds4_gpu_tensor *input) {
    if (!m || !m->gvq || !out || !input ||
        gate >= m->gvq->model->source.size() || up >= m->gvq->model->source.size()) {
        return false;
    }
    const glmvq_source &gs = m->gvq->model->source[gate];
    const glmvq_source &us = m->gvq->model->source[up];
    if (gs.storage != GLMVQ_WEIGHT_VQ || us.storage != GLMVQ_WEIGHT_VQ ||
        !gs.vq.main || !gs.vq.book || !gs.vq.gain ||
        !us.vq.main || !us.vq.book || !us.vq.gain) {
        return false;
    }
    int variant = G4_VQ_AUTO, threads = 256;
    const char *v = getenv("DS4_GEMMA4_VQ_GEMV_VARIANT");
    if (v && *v) { variant = atoi(v); }
    const char *t = getenv("DS4_GEMMA4_VQ_GEMV_THREADS");
    if (t && *t) { threads = atoi(t); }
    return g4_vq_gemv_glu(gs.vq, us.vq, (const float *)input->ptr, (float *)out->ptr,
        variant, threads, ds4_current_stream()) != 0;
}

/* Direct VQ prefill GEMM for the grouped expert path (tokens > 1). Off by
 * default; DS4_GEMMA4_VQ_PREFILL=1 turns it on, and the restore + cuBLAS
 * path stays the reference. One tile decodes both gate and up records and
 * writes gelu(gate) * up, or one record for the down projection. */
static bool g4_vq_prefill(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_VQ_PREFILL");
        on = v && v[0] == '1' && !v[1];
    }
    return on != 0;
}

/* Debug: compare the direct prefill GEMM against the restore + cuBLAS path
 * on the live expert records. DS4_GEMMA4_VQ_VERIFY=1 prints the first
 * groups' relL2 and max abs difference for gate and down. */
static bool g4_moe_verify(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_VQ_VERIFY");
        on = v && v[0] == '1' && !v[1];
    }
    return on != 0;
}

static void g4_moe_verify_pair(const char *what, const ds4_gpu_tensor *got,
        const ds4_gpu_tensor *want, uint64_t count) {
    static float *a = nullptr, *b = nullptr;
    static uint64_t cap = 0;
    if (count > cap) {
        free(a); free(b);
        cap = count;
        a = (float *)malloc(cap * sizeof(float));
        b = (float *)malloc(cap * sizeof(float));
    }
    if (!a || !b || !ds4_gpu_tensor_read(got, 0, a, count * sizeof(float)) ||
        !ds4_gpu_tensor_read(want, 0, b, count * sizeof(float))) {
        fprintf(stderr, "ds4: VQ verify %s: read failed\n", what);
        return;
    }
    double num = 0.0, den = 0.0, mx = 0.0;
    for (uint64_t i = 0; i < count; i++) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * b[i];
        if (fabs(d) > mx) { mx = fabs(d); }
    }
    fprintf(stderr, "ds4: VQ verify %s: relL2 %.3e max %.3e first %.6f/%.6f\n",
        what, sqrt(num / (den > 0.0 ? den : 1.0)), mx, a[0], b[0]);
}

/* Direct VQ prefill for the whole layer: gather every route row once, run
 * the fused gate/up over all experts' work items in one launch, then the
 * down projection, then scatter every route once. Returns 0 without touching
 * the layer when any routed expert's matrices are outside the tile geometry,
 * so the caller falls back to the per-expert restore + cuBLAS loop.
 *
 * The gather and scatter keep the reference's grouped order (m->order is the
 * stable counting sort by expert), so each output row's index_add sees the
 * same sequence of expert contributions the per-expert loop would produce. */
static int g4_moe_run_prefill(ds4_g4_moe *m, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *weights, const uint32_t *matrices, uint32_t rows,
        const uint32_t *counts, const uint32_t *offsets) {
    if (!m || !m->items || !m->qmat || !m->hitems || !m->hqmat) { return 0; }
    const uint64_t pairs = (uint64_t)rows * G4_USED;
    for (uint32_t e = 0; e < G4_EXPERTS; e++) {
        if (!counts[e]) { continue; }
        for (uint32_t w = 0; w < 3u; w++) {
            const uint32_t index = matrices[w * G4_EXPERTS + e];
            if (index >= m->gvq->model->source.size()) { return 0; }
            const glmvq_source &s = m->gvq->model->source[index];
            if (s.storage != GLMVQ_WEIGHT_VQ || !g4_vq_prefill_ready(s.vq)) { return 0; }
            m->hqmat[w * G4_EXPERTS + e] = s.vq;
        }
        const glmvq_matrix &gate = m->hqmat[e];
        const glmvq_matrix &up = m->hqmat[G4_EXPERTS + e];
        const glmvq_matrix &down = m->hqmat[2u * G4_EXPERTS + e];
        if (gate.rows != G4_FF || gate.cols != G4_EMBED ||
            up.rows != G4_FF || up.cols != G4_EMBED ||
            down.rows != G4_EMBED || down.cols != G4_FF) {
            return 0;
        }
    }
    uint32_t count = 0;
    for (uint32_t e = 0; e < G4_EXPERTS; e++) {
        const uint32_t n = counts[e];
        if (!n) { continue; }
        for (uint32_t row0 = 0; row0 < G4_FF; row0 += G4_VQP_BM) {
            for (uint32_t t0 = 0; t0 < n; t0 += G4_VQP_BN) {
                if (count >= m->items_cap) { return 0; }
                m->hitems[count++] = {e, G4_EXPERTS + e, row0, offsets[e] + t0, offsets[e] + n};
            }
        }
    }
    if (!ds4_gpu_tensor_write(m->qmat, 0, m->hqmat, 3u * G4_EXPERTS * sizeof(glmvq_matrix)) ||
        !ds4_gpu_tensor_write(m->items, 0, m->hitems, (uint64_t)count * sizeof(g4_vqp_item))) {
        return 0;
    }
    const uint64_t elements = pairs * G4_EMBED;
    const uint32_t blocks = (uint32_t)((elements + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
    const uint32_t scatter_blocks = (uint32_t)(((uint64_t)rows * G4_EMBED + GLMVQ_THREADS - 1) /
        GLMVQ_THREADS);
    g4_moe_gather_kernel<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
        (float *)m->in->ptr, (const float *)input->ptr, (const uint32_t *)m->order->ptr,
        0u, elements, G4_EMBED);
    if (!cuda_ok(cudaGetLastError(), "Gemma-4 expert gather")) { return 0; }
    if (!g4_vq_gemm_items((const g4_vqp_item *)m->items->ptr, count,
            (const glmvq_matrix *)m->qmat->ptr, true, (const float *)m->in->ptr, G4_EMBED,
            (float *)m->mid->ptr, G4_FF, ds4_current_stream())) {
        return 0;
    }
    count = 0;
    for (uint32_t e = 0; e < G4_EXPERTS; e++) {
        const uint32_t n = counts[e];
        if (!n) { continue; }
        for (uint32_t row0 = 0; row0 < G4_EMBED; row0 += G4_VQP_BM) {
            for (uint32_t t0 = 0; t0 < n; t0 += G4_VQP_BN) {
                if (count >= m->items_cap) { return 0; }
                m->hitems[count++] = {2u * G4_EXPERTS + e, 0u, row0, offsets[e] + t0,
                    offsets[e] + n};
            }
        }
    }
    if (!ds4_gpu_tensor_write(m->items, 0, m->hitems, (uint64_t)count * sizeof(g4_vqp_item)) ||
        !g4_vq_gemm_items((const g4_vqp_item *)m->items->ptr, count,
            (const glmvq_matrix *)m->qmat->ptr, false, (const float *)m->mid->ptr, G4_FF,
            (float *)m->out->ptr, G4_EMBED, ds4_current_stream())) {
        return 0;
    }
    g4_moe_scatter_ordered_kernel<<<scatter_blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)m->out->ptr, (const float *)weights->ptr,
        (const uint32_t *)m->order->ptr, (const uint32_t *)m->tokpair->ptr,
        (uint64_t)rows * G4_EMBED, G4_EMBED);
    return cuda_ok(cudaGetLastError(), "Gemma-4 expert scatter") ? 1 : 0;
}

static int g4_moe_run_routes(ds4_g4_moe *m, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *weights, const uint32_t *matrices, const int32_t *routes,
        uint32_t rows) {
    if (!m || !out || !input || !weights || !matrices || !routes || !rows ||
        rows > m->chunk) { return 0; }
    const uint64_t pairs = (uint64_t)rows * G4_USED;
    if (routes != m->routes) {
        memcpy(m->routes, routes, (size_t)pairs * sizeof(int32_t));
    }
    uint32_t counts[G4_EXPERTS] = {}, offsets[G4_EXPERTS + 1] = {}, cursor[G4_EXPERTS];
    for (uint64_t i = 0; i < pairs; i++) {
        const int32_t expert = m->routes[i];
        if (expert < 0 || expert >= G4_EXPERTS) { return 0; }
        counts[expert]++;
    }
    for (uint32_t e = 0; e < G4_EXPERTS; e++) {
        offsets[e + 1] = offsets[e] + counts[e];
        cursor[e] = offsets[e];
    }
    for (uint64_t i = 0; i < pairs; i++) { m->grouped[cursor[m->routes[i]]++] = (uint32_t)i; }
    if (!ds4_gpu_tensor_write(m->order, 0, m->grouped, pairs * sizeof(uint32_t))) { return 0; }
    /* Per token, its grouped rows in the order the reference adds them: the
     * grouped list is expert-major, so walking it in index order appends
     * each token's experts in ascending expert id. Only the direct prefill
     * consumes it; the one-row path leaves it alone. */
    if (rows > 1u) {
        for (uint32_t t = 0; t < rows; t++) { m->htokfill[t] = 0; }
        for (uint64_t r = 0; r < pairs; r++) {
            const uint32_t token = m->grouped[r] / G4_USED;
            m->htokpair[(uint64_t)token * G4_USED + m->htokfill[token]++] = (uint32_t)r;
        }
        if (!ds4_gpu_tensor_write(m->tokpair, 0, m->htokpair, pairs * sizeof(uint32_t))) {
            return 0;
        }
    }
    if (!ds4_gpu_tensor_fill_f32(out, 0.0f, (uint64_t)rows * G4_EMBED)) { return 0; }
    if (rows > 1 && getenv("DS4_GEMMA4_EXACT_PREFILL")) {
        /* P6a's chunk-size exactness gate: run each route as its own one-row
         * projection, so the grouped prefill equals the one-row path bit for
         * bit. This is the fallback for the gate, not the default: it
         * restores every expert matrix once per route (measured at 72.7% of
         * the prefill's GPU time on the 4070 SUPER), while the grouped loop
         * below restores it once per expert per chunk, the Ling-3.1 /
         * A.X-K2 prefill shape. */
        const uint32_t blocks = (uint32_t)((G4_EMBED + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
        for (uint64_t i = 0; i < pairs; i++) {
            const uint32_t expert = (uint32_t)m->routes[m->grouped[i]];
            g4_moe_gather_kernel<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
                (float *)m->in->ptr, (const float *)input->ptr, (const uint32_t *)m->order->ptr,
                (uint32_t)i, G4_EMBED, G4_EMBED);
            if (!cuda_ok(cudaGetLastError(), "Gemma-4 expert gather") ||
                !ds4_gpu_gvq_linear(m->gvq, m->gate, matrices[expert], m->in, 1, DS4_GVQ_BF16) ||
                !ds4_gpu_gvq_linear(m->gvq, m->up, matrices[G4_EXPERTS + expert], m->in, 1,
                    DS4_GVQ_BF16) ||
                !ds4_gpu_g4_gelu(m->mid, m->gate, m->up, G4_FF) ||
                !ds4_gpu_gvq_linear(m->gvq, m->out, matrices[2u * G4_EXPERTS + expert], m->mid,
                    1, DS4_GVQ_BF16)) {
                return 0;
            }
            g4_moe_scatter_kernel<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
                (float *)out->ptr, (const float *)m->out->ptr, (const float *)weights->ptr,
                (const uint32_t *)m->order->ptr, (uint32_t)i, G4_EMBED, G4_EMBED);
            if (!cuda_ok(cudaGetLastError(), "Gemma-4 expert scatter")) { return 0; }
        }
        return 1;
    }
    if (rows > 1 && g4_vq_prefill()) {
        /* The whole layer in one pass: one gather, one gate/up launch over
         * every expert's work items, one down launch, one scatter. */
        if (g4_moe_run_prefill(m, out, input, weights, matrices, rows, counts, offsets)) {
            return 1;
        }
    }
    for (uint32_t e = 0; e < G4_EXPERTS; e++) {
        const uint32_t n = counts[e];
        if (!n) { continue; }
        const uint64_t elements = (uint64_t)n * G4_EMBED;
        const uint32_t blocks = (uint32_t)((elements + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
        g4_moe_gather_kernel<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
            (float *)m->in->ptr, (const float *)input->ptr, (const uint32_t *)m->order->ptr,
            offsets[e], elements, G4_EMBED);
        const bool one = n == 1;
        bool ok = cuda_ok(cudaGetLastError(), "Gemma-4 expert gather");
        static int reports = 0;
        if (ok && g4_moe_verify() && !one && reports < 6) {
            const glmvq_source &gs = m->gvq->model->source[matrices[e]];
            const glmvq_source &us = m->gvq->model->source[matrices[G4_EXPERTS + e]];
            reports++;
            ds4_gpu_gvq_linear(m->gvq, m->gate, matrices[e], m->in, n, DS4_GVQ_BF16);
            /* Reference against itself at half the token count before the
             * scratch is overwritten: cuBLAS picks its split-K by shape, so
             * this is the port's own chunk-width noise, the band the direct
             * kernel has to sit in. */
            if (n >= 4) {
                const uint32_t half = n / 2;
                ds4_gpu_gvq_linear(m->gvq, m->up, matrices[e], m->in, half, DS4_GVQ_BF16);
                g4_moe_verify_pair("ref gate n/2 vs n", m->up, m->gate, (uint64_t)half * G4_FF);
            }
            g4_vq_gemm(gs.vq, (const float *)m->in->ptr, gs.vq.cols,
                (float *)m->up->ptr, gs.vq.rows, n, ds4_current_stream());
            g4_moe_verify_pair("gate", m->up, m->gate, (uint64_t)n * G4_FF);
            ds4_gpu_gvq_linear(m->gvq, m->up, matrices[G4_EXPERTS + e], m->in, n, DS4_GVQ_BF16);
            if (n >= 4) {
                const uint32_t half = n / 2;
                ds4_gpu_gvq_linear(m->gvq, m->gate, matrices[G4_EXPERTS + e], m->in, half,
                    DS4_GVQ_BF16);
                g4_moe_verify_pair("ref up n/2 vs n", m->gate, m->up, (uint64_t)half * G4_FF);
            }
            g4_vq_gemm(us.vq, (const float *)m->in->ptr, us.vq.cols,
                (float *)m->gate->ptr, us.vq.rows, n, ds4_current_stream());
            g4_moe_verify_pair("up", m->gate, m->up, (uint64_t)n * G4_FF);
        }
        bool fused = false;
        if (ok && !fused && one && g4_moe_fused()) {
            /* One GEMV reads gate and up and writes gelu(gate) * up. */
            fused = g4_moe_linear_glu_one(m, m->mid, matrices[e],
                matrices[G4_EXPERTS + e], m->in);
        }
        if (ok && !fused) {
            ok = ((one && g4_moe_linear_one(m, m->gate, matrices[e], m->in)) ||
                  ds4_gpu_gvq_linear(m->gvq, m->gate, matrices[e], m->in, n, DS4_GVQ_BF16)) &&
                 ((one && g4_moe_linear_one(m, m->up, matrices[G4_EXPERTS + e], m->in)) ||
                  ds4_gpu_gvq_linear(m->gvq, m->up, matrices[G4_EXPERTS + e], m->in, n,
                    DS4_GVQ_BF16)) &&
                 ds4_gpu_g4_gelu(m->mid, m->gate, m->up, (uint64_t)n * G4_FF);
        }
        bool down = false;
        if (ok && g4_moe_verify() && !one && reports <= 6) {
            const glmvq_source &ds = m->gvq->model->source[matrices[2u * G4_EXPERTS + e]];
            ds4_gpu_gvq_linear(m->gvq, m->out, matrices[2u * G4_EXPERTS + e], m->mid, n,
                DS4_GVQ_BF16);
            g4_vq_gemm(ds.vq, (const float *)m->mid->ptr, ds.vq.cols,
                (float *)m->in->ptr, ds.vq.rows, n, ds4_current_stream());
            g4_moe_verify_pair("down", m->in, m->out, (uint64_t)n * G4_EMBED);
        }
        if (ok && !down) {
            down = (one && g4_moe_linear_one(m, m->out, matrices[2u * G4_EXPERTS + e], m->mid)) ||
                ds4_gpu_gvq_linear(m->gvq, m->out, matrices[2u * G4_EXPERTS + e], m->mid, n,
                    DS4_GVQ_BF16);
        }
        if (!ok || !down) {
            return 0;
        }
        g4_moe_scatter_kernel<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)m->out->ptr, (const float *)weights->ptr,
            (const uint32_t *)m->order->ptr, offsets[e], elements, G4_EMBED);
        if (!cuda_ok(cudaGetLastError(), "Gemma-4 expert scatter")) { return 0; }
    }
    return 1;
}

extern "C" int ds4_gpu_g4_moe_run(ds4_g4_moe *m, ds4_gpu_tensor *out,
        const ds4_gpu_tensor *input, const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights, const uint32_t *matrices, uint32_t rows) {
    if (!m || !selected || !rows || rows > m->chunk) { return 0; }
    const uint64_t pairs = (uint64_t)rows * G4_USED;
    if (!ds4_gpu_tensor_read(selected, 0, m->routes, pairs * sizeof(int32_t))) {
        return 0;
    }
    return g4_moe_run_routes(m, out, input, weights, matrices, m->routes, rows);
}

extern "C" int ds4_gpu_g4_moe_run_ids(ds4_g4_moe *m, ds4_gpu_tensor *out,
        const ds4_gpu_tensor *input, const ds4_gpu_tensor *weights,
        const uint32_t *matrices, const int32_t *routes, uint32_t rows) {
    return g4_moe_run_routes(m, out, input, weights, matrices, routes, rows);
}

/* -------------------------------------------------------- Q4_K decode (4af) */

/* Debug: DS4_GEMMA4_Q4_DECODE_VERIFY=1 runs the shared per-matrix path beside
 * the Gemma pass and prints the distance, one line per matrix. */
static bool g4_q4_decode_verify(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_Q4_DECODE_VERIFY");
        on = v && v[0] == '1' && !v[1];
    }
    return on != 0;
}

/* One Q4_K matrix of `rows` rows over `cols` inputs, or 0 for anything the
 * Gemma kernel does not take (the shared dispatcher then stays in charge).
 * The decode is one token, so the output must hold one column: `rows` here is
 * the matrix row count, not the token count. */
static int g4_q4_matrix(ds4_gvq_ctx *ctx, uint32_t index, uint32_t cols, uint32_t rows,
        const ds4_gpu_tensor *out) {
    if (!ctx || !out || !cols || cols % G4_Q4_BLOCK || !rows ||
        index >= ctx->model->source.size()) {
        return 0;
    }
    const glmvq_source &s = ctx->model->source[index];
    if (ctx->model->records[index].type != GVQ_TYPE_Q4 || !s.data ||
        (uintptr_t)s.data % 16 || s.cols != cols || s.rows != rows ||
        out->bytes < (uint64_t)s.rows * sizeof(float)) {
        return 0;
    }
    return 1;
}

/* One-token Q4_K decode over `count` matrices sharing one input. The kernel
 * stages eight rows of one matrix per block and walks them with one warp per
 * row; the matrices are laid out matrix by matrix in the same launch. */
extern "C" int ds4_gpu_g4_q4_many(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs,
        const uint32_t *weights, uint32_t count, const ds4_gpu_tensor *input, uint32_t rows) {
    if (!ctx || !outs || !weights || !input || !count || count > G4_Q4_MANY ||
        rows != 1u || (uintptr_t)input->ptr % 16 || weights[0] >= ctx->model->source.size() ||
        !outs[0]) {
        return 0;
    }
    const uint32_t cols = ctx->model->source[weights[0]].cols;
    if (!cols || cols % G4_Q4_BLOCK || input->bytes < (uint64_t)cols * sizeof(float)) {
        return 0;
    }
    g4_q4_many many = {};
    uint32_t groups = 0;
    for (uint32_t c = 0; c < count; c++) {
        if (weights[c] >= ctx->model->source.size() || !outs[c]) { return 0; }
        const glmvq_source &s = ctx->model->source[weights[c]];
        if (!g4_q4_matrix(ctx, weights[c], cols, s.rows, outs[c])) { return 0; }
        many.weight[c] = (const uint8_t *)s.data;
        many.out[c] = (float *)outs[c]->ptr;
        many.rows[c] = s.rows;
        many.first[c] = groups;
        groups += 1u + (s.rows - 1u) / G4_Q4_WARPS;
    }
    /* The kernel selects by compares, so every slot past the last matrix
     * must point at the end of the table, not at 0. */
    for (uint32_t c = count; c <= G4_Q4_MANY; c++) { many.first[c] = groups; }
    const uint64_t shared = (uint64_t)G4_Q4_WARPS * (cols / G4_Q4_BLOCK) * G4_Q4_BYTES;
    if (shared > 48u * 1024u) { return 0; }
    if (g4_q4_decode_verify()) {
        static ds4_gpu_tensor *ref = nullptr;
        static float *a = nullptr, *b = nullptr;
        static uint32_t reports = 0;
        if (!ref) {
            ref = ds4_gpu_tensor_alloc(4096u * sizeof(float));
            a = (float *)malloc(4096u * sizeof(float));
            b = (float *)malloc(4096u * sizeof(float));
        }
        if (!ref || !a || !b) { return 0; }
        for (uint32_t c = 0; c < count; c++) {
            const glmvq_source &s = ctx->model->source[weights[c]];
            if (s.rows > 4096u ||
                !ds4_gpu_gvq_linear(ctx, ref, weights[c], input, 1u, DS4_GVQ_BF16)) {
                return 0;
            }
            g4_q4_many one = {};
            one.weight[0] = many.weight[c];
            one.out[0] = many.out[c];
            one.rows[0] = many.rows[c];
            for (uint32_t k = 1; k <= G4_Q4_MANY; k++) { one.first[k] = many.first[c + 1] - many.first[c]; }
            g4_q4_many_kernel<<<one.first[G4_Q4_MANY], G4_Q4_WARPS * 32u, (size_t)shared,
                ds4_current_stream()>>>(one, (const float *)input->ptr, cols);
            if (!cuda_ok(cudaGetLastError(), "Gemma-4 Q4 decode verify") ||
                !ds4_gpu_tensor_read(ref, 0, a, (uint64_t)s.rows * sizeof(float)) ||
                !ds4_gpu_tensor_read(outs[c], 0, b, (uint64_t)s.rows * sizeof(float))) {
                return 0;
            }
            if (reports < 40u) {
                double num = 0.0, den = 0.0, mx = 0.0;
                uint32_t same = 0;
                for (uint32_t i = 0; i < s.rows; i++) {
                    const double d = (double)b[i] - (double)a[i];
                    num += d * d;
                    den += (double)a[i] * a[i];
                    if (fabs(d) > mx) { mx = fabs(d); }
                    same += a[i] == b[i];
                }
                reports++;
                fprintf(stderr, "ds4: G4 Q4 %ux%u relL2 %.3e max %.3e exact %u/%u\n",
                    s.rows, s.cols, sqrt(num / (den > 0.0 ? den : 1.0)), mx, same, s.rows);
            }
        }
        return 1;
    }
    g4_q4_many_kernel<<<groups, G4_Q4_WARPS * 32u, (size_t)shared, ds4_current_stream()>>>(
        many, (const float *)input->ptr, cols);
    return cuda_ok(cudaGetLastError(), "Gemma-4 Q4 decode") ? 1 : 0;
}

/* The dense gate/up pair in one launch: both matrices' eight-row groups are
 * staged beside each other and the epilogue is the gelu product the graph
 * applies today. */
extern "C" int ds4_gpu_g4_q4_glu(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t gate,
        uint32_t up, const ds4_gpu_tensor *input, uint32_t rows) {
    if (!ctx || !out || !input || rows != 1u || gate >= ctx->model->source.size() ||
        up >= ctx->model->source.size() || (uintptr_t)input->ptr % 16) {
        return 0;
    }
    const glmvq_source &g = ctx->model->source[gate], &u = ctx->model->source[up];
    if (ctx->model->records[gate].type != GVQ_TYPE_Q4 ||
        ctx->model->records[up].type != GVQ_TYPE_Q4 ||
        !g.data || !u.data || (uintptr_t)g.data % 16 || (uintptr_t)u.data % 16 ||
        g.cols != u.cols || g.rows != u.rows || !g.rows || g.cols % G4_Q4_BLOCK ||
        input->bytes < (uint64_t)g.cols * sizeof(float) ||
        out->bytes < (uint64_t)rows * g.rows * sizeof(float)) {
        return 0;
    }
    const uint64_t shared = 2ull * G4_Q4_WARPS * (g.cols / G4_Q4_BLOCK) * G4_Q4_BYTES;
    if (shared > 48u * 1024u) { return 0; }
    const uint32_t groups = 1u + (g.rows - 1u) / G4_Q4_WARPS;
    g4_q4_glu_kernel<<<groups, G4_Q4_WARPS * 32u, (size_t)shared, ds4_current_stream()>>>(
        (const uint8_t *)g.data, (const uint8_t *)u.data, (float *)out->ptr,
        (const float *)input->ptr, g.cols, g.rows);
    return cuda_ok(cudaGetLastError(), "Gemma-4 Q4 gate/up") ? 1 : 0;
}



/* Debug: DS4_GEMMA4_MOE_PERSIST_VERIFY=1 runs the per-expert loop beside the
 * persistent pass and prints the per-layer distance. Off by default; the
 * readback is a synchronization and belongs to the gate, not the server. */
static bool g4_moe_persist_verify(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_MOE_PERSIST_VERIFY");
        on = v && v[0] == '1' && !v[1];
    }
    return on != 0;
}

/* One CTA per SM: the staged book leaves room for exactly one. */
static uint32_t g4_moe_persist_ctas(void) {
    static uint32_t cached = 0;
    static int device = -1;
    int current = 0;
    if (cudaGetDevice(&current) != cudaSuccess) { return 48u; }
    if (device != current || !cached) {
        int sms = 0;
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, current) !=
                cudaSuccess || sms <= 0) {
            sms = 48;
        }
        cached = (uint32_t)sms;
        device = current;
    }
    return cached;
}

static int g4_moe_persist_shared(void) {
    static int ready = -1;
    if (ready < 0) {
        ready = cudaFuncSetAttribute(g4_moe_gu_kernel<G4_MOE_WARPS>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, G4_MOE_GU_STAGE) ==
                cudaSuccess &&
            cudaFuncSetAttribute(g4_moe_down_kernel<G4_MOE_WARPS>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, G4_MOE_DOWN_STAGE) ==
                cudaSuccess;
    }
    return ready;
}

/* Resolve one layer's 384 GVQ indices into the device table the persistent
 * kernels read, once per layer. The kernels stage one main book and one
 * residual book, so the layer must have exactly the artifact's layout: every
 * gate/up expert on the layer's gate/up book, every down expert on its down
 * book. Anything else returns 0 and the caller keeps the per-expert loop. */
static int g4_moe_persist_bind(ds4_g4_moe *m, const uint32_t *matrices, uint32_t layer) {
    if (!m || !m->gvq || !m->ptab || !m->hbind || !matrices || layer >= G4_LAYERS) { return 0; }
    const uint32_t count = 3u * G4_EXPERTS;
    uint32_t *bound = m->hbind + (size_t)layer * count;
    if (m->hbind_ok[layer] && !memcmp(bound, matrices, count * sizeof(uint32_t))) { return 1; }
    glmvq_matrix *table = m->hqmat;
    for (uint32_t p = 0; p < 3u; p++) {
        for (uint32_t e = 0; e < G4_EXPERTS; e++) {
            const uint32_t index = matrices[p * G4_EXPERTS + e];
            if (index >= m->gvq->model->source.size()) { return 0; }
            const glmvq_source &s = m->gvq->model->source[index];
            const glmvq_matrix &q = s.vq;
            if (s.storage != GLMVQ_WEIGHT_VQ || !q.main || !q.plane || !q.book || !q.gain ||
                !q.res || !q.res_book || q.precision != GLMVQ_E4M3) { return 0; }
            if (p < 2u) {
                if (q.rows != G4_FF || q.cols != G4_EMBED) { return 0; }
            } else if (q.rows != G4_EMBED || q.cols != G4_FF) {
                return 0;
            }
            table[p * G4_EXPERTS + e] = q;
        }
    }
    for (uint32_t e = 0; e < G4_EXPERTS; e++) {
        const glmvq_matrix &g = table[e], &u = table[G4_EXPERTS + e];
        const glmvq_matrix &d = table[2u * G4_EXPERTS + e];
        if (g.book != table[0].book || g.res_book != table[0].res_book ||
            u.book != table[0].book || u.res_book != table[0].res_book ||
            d.book != table[2u * G4_EXPERTS].book ||
            d.res_book != table[2u * G4_EXPERTS].res_book) {
            return 0;
        }
    }
    if (!ds4_gpu_tensor_write(m->ptab, (uint64_t)layer * count * sizeof(glmvq_matrix),
            table, (uint64_t)count * sizeof(glmvq_matrix))) {
        return 0;
    }
    memcpy(bound, matrices, count * sizeof(uint32_t));
    m->hbind_ok[layer] = 1;
    return 1;
}

/* The two launches of one layer. Returns 0 without launching anything the
 * caller cannot undo: both kernels write only their own outputs. */
static int g4_moe_persist_launch(ds4_g4_moe *m, ds4_gpu_tensor *out,
        const ds4_gpu_tensor *input, const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights, const uint32_t *matrices, uint32_t layer) {
    if (!m || !out || !input || !selected || !weights || !matrices || layer >= G4_LAYERS) {
        return 0;
    }
    if (out->bytes < (uint64_t)G4_EMBED * sizeof(float) ||
        input->bytes < (uint64_t)G4_EMBED * sizeof(float) ||
        selected->bytes < (uint64_t)G4_USED * sizeof(uint32_t) ||
        weights->bytes < (uint64_t)G4_USED * sizeof(float) ||
        !m->pmid || !g4_moe_persist_shared() || !g4_moe_persist_bind(m, matrices, layer)) {
        return 0;
    }
    const uint32_t ctas = g4_moe_persist_ctas();
    const glmvq_matrix *tab = (const glmvq_matrix *)m->ptab->ptr;
    const uint32_t *sel = (const uint32_t *)selected->ptr;
    g4_moe_gu_kernel<G4_MOE_WARPS><<<ctas, G4_MOE_THREADS, G4_MOE_GU_STAGE,
        ds4_current_stream()>>>(tab, layer, sel, (const float *)input->ptr,
        (__nv_bfloat16 *)m->pmid->ptr);
    if (!cuda_ok(cudaGetLastError(), "Gemma-4 persistent MoE gate/up")) { return 0; }
    g4_moe_down_kernel<G4_MOE_WARPS><<<ctas, G4_MOE_THREADS, G4_MOE_DOWN_STAGE,
        ds4_current_stream()>>>(tab, layer, sel, (const __nv_bfloat16 *)m->pmid->ptr,
        (const float *)weights->ptr, (float *)out->ptr);
    return cuda_ok(cudaGetLastError(), "Gemma-4 persistent MoE down") ? 1 : 0;
}

extern "C" int ds4_gpu_g4_moe_run_persist(ds4_g4_moe *m, ds4_gpu_tensor *out,
        const ds4_gpu_tensor *input, const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights, const uint32_t *matrices, uint32_t layer) {
    if (!g4_moe_persist_verify()) {
        return g4_moe_persist_launch(m, out, input, selected, weights, matrices, layer);
    }
    static ds4_gpu_tensor *ref = nullptr;
    static float *a = nullptr, *b = nullptr;
    if (!ref) {
        ref = ds4_gpu_tensor_alloc((uint64_t)G4_EMBED * sizeof(float));
        a = (float *)malloc((size_t)G4_EMBED * sizeof(float));
        b = (float *)malloc((size_t)G4_EMBED * sizeof(float));
    }
    if (!ref || !a || !b ||
        !ds4_gpu_g4_moe_run(m, ref, input, selected, weights, matrices, 1u) ||
        !g4_moe_persist_launch(m, out, input, selected, weights, matrices, layer) ||
        !ds4_gpu_tensor_read(ref, 0, a, (uint64_t)G4_EMBED * sizeof(float)) ||
        !ds4_gpu_tensor_read(out, 0, b, (uint64_t)G4_EMBED * sizeof(float))) {
        return 0;
    }
    static uint32_t reports = 0;
    if (reports < G4_LAYERS) {
        double num = 0.0, den = 0.0, mx = 0.0;
        uint32_t same = 0;
        for (uint32_t i = 0; i < G4_EMBED; i++) {
            const double d = (double)b[i] - (double)a[i];
            num += d * d;
            den += (double)a[i] * a[i];
            if (fabs(d) > mx) { mx = fabs(d); }
            same += a[i] == b[i];
        }
        reports++;
        fprintf(stderr, "ds4: G4 MoE persist L%u relL2 %.3e max %.3e exact %u/%u\n",
            layer, sqrt(num / (den > 0.0 ? den : 1.0)), mx, same, G4_EMBED);
    }
    return 1;
}

/* Rebase a streamed VQ1 record. This mirrors gvq_link's VQ branch (and
 * gvq_link_res for the residual), but reads the record's own header from the
 * host staging copy and points the source at the device slot. The book and
 * residual book are separate resident records, so their pointers stay as the
 * bind left them. */
extern "C" int ds4_gpu_g4_stream_rebind(ds4_gvq_model *model, uint32_t index,
        const ds4_gpu_tensor *slot, uint64_t offset, uint64_t bytes, const void *header,
        uint32_t res_index, const ds4_gpu_tensor *res_slot, uint64_t res_offset,
        uint64_t res_bytes, const void *res_header) {
    if (!model || index >= model->source.size() || !slot || !header || !bytes ||
        offset > slot->bytes || bytes > slot->bytes - offset || bytes < GVQ_HEADER) {
        return 0;
    }
    const ds4_gvq_record &r = model->records[index];
    if (r.type != GVQ_TYPE_VQ || r.bytes != bytes || r.book >= model->source.size() ||
        model->records[r.book].type != GVQ_TYPE_BOOK) { return 0; }
    const uint8_t *h = (const uint8_t *)header;
    const uint32_t precision = gvq_u32(h, 20), words = gvq_u32(h, 16);
    const uint64_t gain = gvq_u64(h, 32), main = gvq_u64(h, 40), plane = gvq_u64(h, 48);
    const uint64_t elements = (uint64_t)r.rows * r.cols;
    const bool v1 = h[4] == 1 && !h[5];
    const bool wide = v1 && h[6] == 13 && words == 8192 && plane;
    const bool narrow = v1 && h[6] == 12 && words == 4096 && !plane;
    const bool v2 = h[4] == 2 && !h[5] && plane && r.res == DS4_GVQ_NO_LINK &&
        ((h[6] == 9 && words == 512) || (h[6] == 10 && words == 1024));
    const uint64_t main_bytes = v2 ? elements / 8 : elements * 3 / 16;
    const uint64_t plane_bytes = v2 ? elements * (h[6] - 8u) / 64 : elements / 64;
    if (memcmp(h, "GVQ1", 4) || (!wide && !narrow && !v2) || h[7] != GLMVQ_VECTOR ||
        gvq_u32(h, 8) != r.rows || gvq_u32(h, 12) != r.cols ||
        precision != GLMVQ_E4M3 ||
        model->records[r.book].bytes != (uint64_t)words * GLMVQ_VECTOR ||
        gvq_u64(h, 56) != r.bytes ||
        !gvq_span(gain, (uint64_t)r.rows * sizeof(__half), r.bytes) ||
        !gvq_span(main, main_bytes, r.bytes) ||
        ((wide || v2) && !gvq_span(plane, plane_bytes, r.bytes))) { return 0; }

    const uint8_t *data = (const uint8_t *)slot->ptr + offset;
    glmvq_source &s = model->source[index];
    s.data = data;
    s.storage = GLMVQ_WEIGHT_VQ;
    s.rows = r.rows;
    s.cols = r.cols;
    s.vq = {data + main, (wide || v2) ? data + plane : nullptr,
        model->source[r.book].data, (const __half *)(data + gain),
        r.rows, r.cols, GLMVQ_E4M3, nullptr, nullptr};

    if (r.res == DS4_GVQ_NO_LINK) {
        return res_index == DS4_GVQ_NO_LINK;
    }
    if (res_index != r.res || res_index >= model->source.size() || !res_slot ||
        !res_header || res_bytes != model->records[res_index].bytes ||
        res_offset > res_slot->bytes || res_bytes > res_slot->bytes - res_offset ||
        res_bytes < GVQ_HEADER) { return 0; }
    const ds4_gvq_record &rr = model->records[res_index];
    const uint32_t res_book = rr.book;
    if (rr.type != GVQ_TYPE_RES || res_book >= model->source.size() ||
        model->records[res_book].type != GVQ_TYPE_BOOK ||
        model->records[res_book].bytes != (uint64_t)256u * GLMVQ_VECTOR) { return 0; }
    const uint8_t *rh = (const uint8_t *)res_header;
    if (memcmp(rh, "GVR1", 4) || rh[4] != 1 || rh[5] || rh[6] != 8 ||
        rh[7] != GLMVQ_VECTOR || gvq_u32(rh, 8) != rr.rows ||
        gvq_u32(rh, 12) != rr.cols || gvq_u32(rh, 16) != 256u ||
        gvq_u32(rh, 20) != GLMVQ_E4M3 || gvq_u32(rh, 28) ||
        gvq_u64(rh, 32) != GVQ_HEADER || gvq_u64(rh, 56) != rr.bytes ||
        !gvq_span(GVQ_HEADER, (uint64_t)rr.rows * rr.cols / 8u, rr.bytes)) { return 0; }
    s.vq.res = (const uint8_t *)res_slot->ptr + res_offset + GVQ_HEADER;
    s.vq.res_book = (const uint8_t *)model->source[res_book].data;
    return 1;
}

/* The expert FF 704 rows and the gathered rows hold one route each; the
 * transient gate/up/mid rows live in the same scratch as the gather. */
uint64_t ds4_gpu_g4_moe_workspace(uint32_t chunk) {
    return ds4_gpu_g4_moe_bytes(chunk);
}

#endif
