/* CUDA adapters for the pinned Naive graph. */
#include "cuda/naive_primitives.cuh"
#include "cuda/naive_draft.cuh"

static bool naive_buf(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && t->bytes >= bytes;
}

static const float *naive_control(const void *map, uint64_t size, uint64_t offset, uint32_t count) {
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    if (!map || offset > size || bytes > size - offset) { return nullptr; }
    return (const float *)cuda_model_range_ptr(map, offset, bytes, "Naive control");
}

extern "C" int ds4_gpu_naive_round(ds4_gpu_tensor *x, uint64_t count) {
    if (!count || count > INT_MAX || !naive_buf(x, count * sizeof(float))) { return 0; }
    naive_round<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>((float *)x->ptr, count);
    return cuda_ok(cudaGetLastError(), "Naive BF16 boundary");
}

extern "C" int ds4_gpu_naive_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *table,
        uint32_t heads, uint32_t width, uint32_t rows) {
    if (!heads || heads > N05_HEADS || !rows || rows > N05_PREFILL_MAX ||
        (width != N05_KEY && width != N05_INDEX_DIM) ||
        !naive_buf(x, (uint64_t)rows * heads * width * sizeof(float)) ||
        !naive_buf(table, (uint64_t)rows * N05_ROT * sizeof(float))) { return 0; }
    naive_rope<<<dim3(heads, rows), 256, 0, ds4_current_stream()>>>(
        (float *)x->ptr, (const float2 *)table->ptr, heads, width);
    return cuda_ok(cudaGetLastError(), "Naive partial RoPE");
}

extern "C" int ds4_gpu_naive_rms(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t offset, uint32_t rows) {
    const uint64_t bytes = (uint64_t)rows * N05_EMBED * sizeof(float);
    const float *weight = naive_control(map, size, offset, N05_EMBED);
    if (!weight || !rows || rows > N05_PREFILL_MAX || !naive_buf(x, bytes) || !naive_buf(out, bytes)) { return 0; }
    naive_rms<<<rows, 256, 0, ds4_current_stream()>>>((float *)out->ptr, (const float *)x->ptr, weight, N05_EMBED);
    return cuda_ok(cudaGetLastError(), "Naive RMSNorm");
}

extern "C" int ds4_gpu_naive_key_norm(ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t weight, uint64_t bias, uint32_t rows) {
    const float *w = naive_control(map, size, weight, N05_INDEX_DIM), *b = naive_control(map, size, bias, N05_INDEX_DIM);
    if (!w || !b || !rows || rows > N05_PREFILL_MAX ||
        !naive_buf(x, (uint64_t)rows * N05_INDEX_DIM * sizeof(float))) { return 0; }
    naive_key_norm<<<rows, N05_INDEX_DIM, 0, ds4_current_stream()>>>((float *)x->ptr, w, b);
    return cuda_ok(cudaGetLastError(), "Naive index key LayerNorm");
}

extern "C" int ds4_gpu_naive_kv_store(ds4_gpu_tensor *cache,
        const ds4_gpu_tensor *k, const ds4_gpu_tensor *v, const ds4_gpu_tensor *positions,
        uint32_t heads, uint32_t rows, uint32_t capacity) {
    const uint64_t kw = (uint64_t)heads * N05_KEY, vw = (uint64_t)heads * N05_VALUE;
    const uint64_t count = (uint64_t)rows * (kw + vw);
    if ((heads != 4 && heads != 8) || !rows || !capacity || rows > capacity || capacity > N05_CONTEXT ||
        !naive_buf(cache, (uint64_t)capacity * (kw + vw) * sizeof(uint16_t)) ||
        !naive_buf(k, (uint64_t)rows * kw * sizeof(float)) || !naive_buf(v, (uint64_t)rows * vw * sizeof(float)) ||
        !naive_buf(positions, (uint64_t)rows * sizeof(unsigned))) { return 0; }
    naive_kv_store<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (__nv_bfloat16 *)cache->ptr, (const float *)k->ptr, (const float *)v->ptr,
        (const unsigned *)positions->ptr, heads, rows, capacity);
    return cuda_ok(cudaGetLastError(), "Naive BF16 K/V store");
}

extern "C" int ds4_gpu_naive_index_store(ds4_gpu_tensor *codes, ds4_gpu_tensor *scales,
        ds4_gpu_tensor *query, const ds4_gpu_tensor *key,
        const ds4_gpu_tensor *positions, uint32_t rows, uint32_t capacity) {
    if (!rows || rows > capacity || capacity > N05_CONTEXT ||
        !naive_buf(codes, (uint64_t)capacity * N05_INDEX_DIM) ||
        !naive_buf(scales, (uint64_t)capacity * sizeof(float)) ||
        !naive_buf(query, (uint64_t)rows * N05_INDEX_HEADS * N05_INDEX_DIM * sizeof(float)) ||
        !naive_buf(key, (uint64_t)rows * N05_INDEX_DIM * sizeof(float)) ||
        !naive_buf(positions, (uint64_t)rows * sizeof(unsigned))) { return 0; }
    naive_fp8_query<<<rows * N05_INDEX_HEADS, N05_INDEX_DIM, 0, ds4_current_stream()>>>((float *)query->ptr, rows * N05_INDEX_HEADS);
    if (!cuda_ok(cudaGetLastError(), "Naive query E4M3")) { return 0; }
    naive_fp8_pack<<<rows, N05_INDEX_DIM, 0, ds4_current_stream()>>>(
        (uint8_t *)codes->ptr, (float *)scales->ptr, (const float *)key->ptr, (const unsigned *)positions->ptr, rows);
    return cuda_ok(cudaGetLastError(), "Naive index history E4M3");
}

extern "C" int ds4_gpu_naive_select(ds4_gpu_tensor *ids, ds4_gpu_tensor *scores,
        ds4_gpu_tensor *a, ds4_gpu_tensor *b, const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *codes, const ds4_gpu_tensor *scales,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *positions,
        uint32_t history, uint32_t rows) {
    if (!history || history > N05_CONTEXT || !rows || rows > N05_QUERY_TILE ||
        !naive_buf(ids, (uint64_t)rows * N05_TOP_K * sizeof(unsigned)) ||
        !naive_buf(positions, (uint64_t)rows * sizeof(unsigned))) { return 0; }
    if (history > N05_TOP_K) {
        const uint64_t tiles = ((uint64_t)history + N05_HISTORY_TILE - 1) / N05_HISTORY_TILE;
        const uint64_t lists = (uint64_t)rows * tiles * N05_TOP_K * sizeof(uint64_t);
        if (!naive_buf(scores, (uint64_t)rows * history * sizeof(float)) ||
            !naive_buf(a, lists) || !naive_buf(b, lists) ||
            !naive_buf(query, (uint64_t)rows * N05_INDEX_HEADS * N05_INDEX_DIM * sizeof(float)) ||
            !naive_buf(codes, (uint64_t)history * N05_INDEX_DIM) ||
            !naive_buf(scales, (uint64_t)history * sizeof(float)) ||
            !naive_buf(weights, (uint64_t)rows * N05_INDEX_HEADS * sizeof(float))) { return 0; }
        naive_index_scores<<<dim3((history + 3) / 4, rows), 128, 0, ds4_current_stream()>>>(
            (float *)scores->ptr, (const float *)query->ptr, (const uint8_t *)codes->ptr,
            (const float *)scales->ptr, (const float *)weights->ptr, (const unsigned *)positions->ptr, history);
        if (!cuda_ok(cudaGetLastError(), "Naive signed index scores")) { return 0; }
    }
    return cuda_ok(naive_topk_launch((unsigned *)ids->ptr, scores ? (const float *)scores->ptr : nullptr,
        a ? (uint64_t *)a->ptr : nullptr, b ? (uint64_t *)b->ptr : nullptr,
        (const unsigned *)positions->ptr, history, rows, ds4_current_stream()), "Naive stable top-k");
}

extern "C" int ds4_gpu_naive_attention(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *cache, const ds4_gpu_tensor *positions, const ds4_gpu_tensor *ids,
        const void *map, uint64_t size, uint64_t sink, uint32_t heads,
        uint32_t capacity, uint32_t rows, uint32_t window) {
    if (!rows || rows > N05_PREFILL_MAX || !capacity || capacity > N05_CONTEXT || rows > capacity ||
        !((heads == 4 && !window) || (heads == 8 && window == N05_WINDOW)) ||
        !naive_buf(q, (uint64_t)rows * N05_HEADS * N05_KEY * sizeof(float)) ||
        !naive_buf(out, (uint64_t)rows * N05_HEADS * N05_VALUE * sizeof(float)) ||
        !naive_buf(cache, (uint64_t)capacity * heads * (N05_KEY + N05_VALUE) * sizeof(uint16_t)) ||
        !naive_buf(positions, (uint64_t)rows * sizeof(unsigned)) ||
        (!window && !naive_buf(ids, (uint64_t)rows * N05_TOP_K * sizeof(unsigned)))) { return 0; }
    const float *sinks = window ? naive_control(map, size, sink, N05_HEADS) : nullptr;
    if (window && !sinks) { return 0; }
    const char *scores = getenv("DS4_NAIVE_DECODE_SCORES");
    // DSA score storage helps narrow decode but reduces wide-prefill occupancy.
    const bool cached = rows == 1 && (!scores || strcmp(scores, "0"));
    if (cached && window) {
        naive_attention<4, N05_WINDOW><<<dim3(N05_HEADS / 4, rows), 128, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)q->ptr, (const __nv_bfloat16 *)cache->ptr, sinks,
            (const unsigned *)positions->ptr, nullptr, heads, capacity, window);
    } else if (cached) {
        naive_attention<4, N05_TOP_K><<<dim3(N05_HEADS / 4, rows), 128, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)q->ptr, (const __nv_bfloat16 *)cache->ptr, nullptr,
            (const unsigned *)positions->ptr, (const unsigned *)ids->ptr, heads, capacity, window);
    } else {
        naive_attention<<<dim3(N05_HEADS / 4, rows), 128, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)q->ptr, (const __nv_bfloat16 *)cache->ptr, sinks,
            (const unsigned *)positions->ptr, ids ? (const unsigned *)ids->ptr : nullptr, heads, capacity, window);
    }
    return cuda_ok(cudaGetLastError(), "Naive GQA attention");
}

extern "C" int ds4_gpu_naive_router(ds4_gpu_tensor *ids, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, const void *map, uint64_t size, uint64_t offset, uint32_t rows) {
    const float *bias = naive_control(map, size, offset, N05_EXPERTS);
    if (!bias || !rows || rows > N05_PREFILL_MAX ||
        !naive_buf(ids, (uint64_t)rows * N05_USED * sizeof(int)) ||
        !naive_buf(weights, (uint64_t)rows * N05_USED * sizeof(float)) ||
        !naive_buf(logits, (uint64_t)rows * N05_EXPERTS * sizeof(float))) { return 0; }
    naive_router<<<rows, N05_EXPERTS, 0, ds4_current_stream()>>>(
        (int *)ids->ptr, (float *)weights->ptr, (const float *)logits->ptr, bias);
    return cuda_ok(cudaGetLastError(), "Naive unbiased router");
}

extern "C" int ds4_gpu_naive_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up, uint64_t count) {
    const uint64_t bytes = count * sizeof(float);
    if (!count || count > INT_MAX || !naive_buf(out, bytes) || !naive_buf(gate, bytes) || !naive_buf(up, bytes)) { return 0; }
    naive_swiglu<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, count);
    return cuda_ok(cudaGetLastError(), "Naive BF16 SwiGLU");
}

extern "C" int ds4_gpu_naive_sum(ds4_gpu_tensor *out, const ds4_gpu_tensor *down,
        const ds4_gpu_tensor *weights, uint32_t rows) {
    const uint64_t count = (uint64_t)rows * N05_EMBED, bytes = count * sizeof(float);
    if (!rows || rows > N05_PREFILL_MAX || !naive_buf(out, bytes) || !naive_buf(down, bytes * N05_USED) ||
        !naive_buf(weights, (uint64_t)rows * N05_USED * sizeof(float))) { return 0; }
    naive_sum<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)down->ptr, (const float *)weights->ptr, count);
    return cuda_ok(cudaGetLastError(), "Naive ordered expert sum");
}

extern "C" int ds4_gpu_naive_add(ds4_gpu_tensor *cur, const ds4_gpu_tensor *other, uint64_t count) {
    const uint64_t bytes = count * sizeof(float);
    if (!count || count > INT_MAX || !naive_buf(cur, bytes) || !naive_buf(other, bytes)) { return 0; }
    naive_add<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>((float *)cur->ptr, (const float *)other->ptr, count);
    return cuda_ok(cudaGetLastError(), "Naive BF16 residual");
}

extern "C" int ds4_gpu_naive_df_tap(ds4_gpu_tensor *out, const ds4_gpu_tensor *hidden,
        uint32_t first, uint32_t rows, uint32_t tap) {
    if (!rows || rows > N05_DF_WINDOW || first > N05_PREFILL_MAX || rows > N05_PREFILL_MAX - first || tap >= N05_DF_TAPS ||
        !naive_buf(out, (uint64_t)rows * N05_DF_SLOT * sizeof(float)) ||
        !naive_buf(hidden, (uint64_t)(first + rows) * N05_EMBED * sizeof(float))) { return 0; }
    naive_df_tap<<<((uint64_t)rows * N05_EMBED + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)hidden->ptr, first, rows, tap);
    return cuda_ok(cudaGetLastError(), "Naive DSpark target taps");
}

extern "C" int ds4_gpu_naive_df_mask(ds4_gpu_tensor *hidden, const void *map,
        uint64_t size, uint64_t offset, uint32_t rows) {
    const float *mask = naive_control(map, size, offset, N05_EMBED);
    if (!mask || !rows || rows > N05_DF_BLOCK ||
        !naive_buf(hidden, (uint64_t)rows * N05_EMBED * sizeof(float))) { return 0; }
    naive_df_mask<<<(rows * N05_EMBED + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)hidden->ptr, mask, rows);
    return cuda_ok(cudaGetLastError(), "Naive DSpark learned mask");
}

extern "C" int ds4_gpu_naive_df_norm(ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t offset, uint32_t heads, uint32_t rows) {
    const float *weight = naive_control(map, size, offset, N05_DF_DIM);
    if (!weight || !rows || rows > N05_DF_WINDOW ||
        (heads != N05_DF_HEADS && heads != N05_DF_KV) ||
        !naive_buf(x, (uint64_t)rows * heads * N05_DF_DIM * sizeof(float)) ||
        !ds4_gpu_naive_round(x, (uint64_t)rows * heads * N05_DF_DIM)) { return 0; }
    naive_rms<<<rows * heads, 256, 0, ds4_current_stream()>>>(
        (float *)x->ptr, (const float *)x->ptr, weight, N05_DF_DIM);
    return cuda_ok(cudaGetLastError(), "Naive DSpark per-head RMSNorm");
}

extern "C" int ds4_gpu_naive_df_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *table,
        uint32_t heads, uint32_t rows) {
    if (!rows || rows > N05_DF_WINDOW || (heads != N05_DF_HEADS && heads != N05_DF_KV) ||
        !naive_buf(x, (uint64_t)rows * heads * N05_DF_DIM * sizeof(float)) ||
        !naive_buf(table, (uint64_t)rows * N05_DF_DIM * sizeof(float))) { return 0; }
    naive_df_rope<<<dim3(heads, rows), N05_DF_DIM, 0, ds4_current_stream()>>>(
        (float *)x->ptr, (const float2 *)table->ptr, heads);
    return cuda_ok(cudaGetLastError(), "Naive DSpark full RoPE");
}

extern "C" int ds4_gpu_naive_df_store(ds4_gpu_tensor *cache, const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v, const ds4_gpu_tensor *positions, uint32_t rows) {
    const uint64_t width = N05_DF_KV * N05_DF_DIM;
    if (!rows || rows > N05_DF_WINDOW ||
        !naive_buf(cache, N05_DF_CAP * 2 * width * sizeof(uint16_t)) ||
        !naive_buf(k, (uint64_t)rows * width * sizeof(float)) ||
        !naive_buf(v, (uint64_t)rows * width * sizeof(float)) ||
        !naive_buf(positions, (uint64_t)rows * sizeof(unsigned))) { return 0; }
    naive_df_store<<<(rows * 2 * width + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (__nv_bfloat16 *)cache->ptr, (const float *)k->ptr, (const float *)v->ptr,
        (const unsigned *)positions->ptr, rows);
    return cuda_ok(cudaGetLastError(), "Naive DSpark projected context KV");
}

extern "C" int ds4_gpu_naive_df_attn(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *cache, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
        uint32_t first, uint32_t start, uint32_t rows) {
    const uint64_t qw = N05_DF_HEADS * N05_DF_DIM, kw = N05_DF_KV * N05_DF_DIM;
    if (!rows || rows > N05_DF_BLOCK || first > start || start > N05_CONTEXT || rows > N05_CONTEXT - start ||
        !naive_buf(out, (uint64_t)rows * qw * sizeof(float)) ||
        !naive_buf(q, (uint64_t)rows * qw * sizeof(float)) ||
        !naive_buf(cache, N05_DF_CAP * 2 * kw * sizeof(uint16_t)) ||
        !naive_buf(k, (uint64_t)rows * kw * sizeof(float)) ||
        !naive_buf(v, (uint64_t)rows * kw * sizeof(float))) { return 0; }
    naive_df_attn<<<dim3(N05_DF_HEADS / 4, rows), 128, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)q->ptr, (const __nv_bfloat16 *)cache->ptr,
        (const float *)k->ptr, (const float *)v->ptr, first, start, rows);
    return cuda_ok(cudaGetLastError(), "Naive DSpark noncausal local attention");
}

extern "C" int ds4_gpu_naive_df_conf(ds4_gpu_tensor *out, const ds4_gpu_tensor *hidden,
        const ds4_gpu_tensor *markov, const void *map, uint64_t size, uint64_t weight, uint64_t bias) {
    const float *w = naive_control(map, size, weight, N05_EMBED + N05_DF_RANK);
    const float *b = naive_control(map, size, bias, 1);
    if (!w || !b || !naive_buf(out, sizeof(float)) || !naive_buf(hidden, N05_EMBED * sizeof(float)) ||
        !naive_buf(markov, N05_DF_RANK * sizeof(float))) { return 0; }
    naive_df_conf<<<1, 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)hidden->ptr, (const float *)markov->ptr, w, b);
    return cuda_ok(cudaGetLastError(), "Naive DSpark raw confidence");
}

extern "C" int ds4_gpu_naive_df_top2(ds4_gpu_tensor *out, const ds4_gpu_tensor *logits) {
    if (!naive_buf(out, sizeof(ds4_gpu_top2_result)) || !naive_buf(logits, N05_VOCAB * sizeof(float))) { return 0; }
    naive_df_top2<<<1, 256, 0, ds4_current_stream()>>>((unsigned *)out->ptr, (const float *)logits->ptr);
    return cuda_ok(cudaGetLastError(), "Naive DSpark top-two margin");
}
