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
#include "cuda/gemma4_attn.cuh"
#include "cuda/gemma_vq_decode.cuh"

/* One routed-expert scratch set, sized for `chunk` tokens and reused across
 * layers: gather rows, the three projections of the route set, and the
 * device order table the host grouping writes each layer. */
struct ds4_g4_moe {
    ds4_gvq_ctx *gvq;
    uint32_t chunk;
    ds4_gpu_tensor *in, *out, *gate, *up, *mid, *order;
    int32_t *routes;
    uint32_t *grouped;
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

extern "C" int ds4_gpu_g4_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected,
        ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits, uint32_t per_expert,
        uint32_t rows) {
    const __nv_bfloat16 *scale = g4_weight(ctx, per_expert, G4_EXPERTS);
    if (!selected || !weights || !logits || !scale || !rows) { return 0; }
    if (selected->bytes < (uint64_t)rows * G4_USED * sizeof(uint32_t) ||
        weights->bytes < (uint64_t)rows * G4_USED * sizeof(float) ||
        logits->bytes < (uint64_t)rows * G4_EXPERTS * sizeof(float)) { return 0; }
    g4_router_kernel<<<rows, G4_EXPERTS, 0, ds4_current_stream()>>>((uint32_t *)selected->ptr,
        (float *)weights->ptr, (const float *)logits->ptr, scale, rows, G4_EXPERTS, G4_USED);
    return cuda_ok(cudaGetLastError(), "Gemma-4 router");
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

/* ------------------------------------------------------- routed experts */

uint64_t ds4_gpu_g4_moe_bytes(uint32_t chunk) {
    if (!chunk || chunk > G4_PREFILL_MAX) { return 0; }
    const uint64_t pairs = (uint64_t)chunk * G4_USED;
    return pairs * G4_EMBED * sizeof(float) * 2u + pairs * G4_FF * sizeof(float) * 3u +
        pairs * sizeof(uint32_t) + pairs * G4_USED * sizeof(uint32_t) +
        pairs * sizeof(int32_t) + pairs * sizeof(float);
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
    if (!m->in || !m->out || !m->gate || !m->up || !m->mid || !m->order ||
        !m->routes || !m->grouped) {
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
    free(m->routes);
    free(m->grouped);
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
    for (uint32_t e = 0; e < G4_EXPERTS; e++) {
        const uint32_t n = counts[e];
        if (!n) { continue; }
        const uint64_t elements = (uint64_t)n * G4_EMBED;
        const uint32_t blocks = (uint32_t)((elements + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
        g4_moe_gather_kernel<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
            (float *)m->in->ptr, (const float *)input->ptr, (const uint32_t *)m->order->ptr,
            offsets[e], elements, G4_EMBED);
        const bool one = n == 1;
        if (!cuda_ok(cudaGetLastError(), "Gemma-4 expert gather") ||
            !((one && g4_moe_linear_one(m, m->gate, matrices[e], m->in)) ||
              ds4_gpu_gvq_linear(m->gvq, m->gate, matrices[e], m->in, n, DS4_GVQ_BF16)) ||
            !((one && g4_moe_linear_one(m, m->up, matrices[G4_EXPERTS + e], m->in)) ||
              ds4_gpu_gvq_linear(m->gvq, m->up, matrices[G4_EXPERTS + e], m->in, n, DS4_GVQ_BF16)) ||
            !ds4_gpu_g4_gelu(m->mid, m->gate, m->up, (uint64_t)n * G4_FF) ||
            !((one && g4_moe_linear_one(m, m->out, matrices[2u * G4_EXPERTS + e], m->mid)) ||
              ds4_gpu_gvq_linear(m->gvq, m->out, matrices[2u * G4_EXPERTS + e], m->mid, n,
                DS4_GVQ_BF16))) {
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
