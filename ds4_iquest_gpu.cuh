/* Native tensor adapters. Model admission and ring lineage stay in the graph;
 * these kernels must receive contiguous, admitted live device positions. */
#include "cuda/iquest_primitives.cuh"
#include "cuda/iquest_prefill.cuh"

extern "C" int ds4_gpu_iquest_policy(void) {
    /* Only the active engine needs canonical MMQ weights. Cleanup releases
     * this overlay; the caller's environment and common cache stay intact. */
    g_weight_policy = CUDA_WEIGHT_CANONICAL;
    return 1;
}

static bool iq_tensor(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && t->bytes >= bytes;
}

static bool iq_attn_shuffle() {
    // Process-wide diagnostic fallback keeps graph selection stable.
    static const bool enabled = [] {
        const char *value = getenv("DS4_IQUEST_ATTN_SHUFFLE");
        return !value || strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool iq_attn_warp() {
    static const bool enabled = [] {
        const char *value = getenv("DS4_IQUEST_ATTN_WARP");
        return !value || strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool iq_attn_tiled() {
    // A process-wide fallback retains the scalar prefill arithmetic.
    static const bool enabled = [] {
        const char *value = getenv("DS4_IQUEST_ATTN_TILED");
        return (!value || strcmp(value, "0") != 0) && iq_prefill::supported();
    }();
    return enabled;
}

extern "C" int ds4_gpu_iquest_rms(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t offset, uint32_t width, uint32_t rows) {
    if (!rows || rows > IQ_PREFILL_MAX * IQ_HEADS || (width != IQ_HEAD && width != IQ_EMBED)) { return 0; }
    const uint64_t bytes = (uint64_t)width * rows * sizeof(float);
    const uint64_t weight_bytes = width * sizeof(float);
    if (!iq_tensor(out, bytes) || !iq_tensor(x, bytes) || offset > size || weight_bytes > size - offset) { return 0; }
    const float *weight = (const float *)cuda_model_range_ptr(map, offset, weight_bytes, "IQuest RMS");
    if (!weight) { return 0; }
    iquest_rms_kernel<<<rows, 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)x->ptr, weight, width);
    return cuda_ok(cudaGetLastError(), "IQuest RMS");
}

extern "C" int ds4_gpu_iquest_rope(ds4_gpu_tensor *x,
        const ds4_gpu_tensor *positions, uint32_t heads, uint32_t rows, float theta) {
    if (!rows || rows > 8192 || (heads != IQ_HEADS && heads != IQ_KV_HEADS) ||
        (theta != 1000000.0f && theta != 10000.0f) ||
        !iq_tensor(x, (uint64_t)rows * heads * IQ_HEAD * sizeof(float)) ||
        !iq_tensor(positions, (uint64_t)rows * sizeof(unsigned))) { return 0; }
    const uint64_t pairs = (uint64_t)rows * heads * IQ_ROT / 2;
    iquest_rope_kernel<<<(pairs + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)x->ptr, (const unsigned *)positions->ptr, heads, rows, theta);
    return cuda_ok(cudaGetLastError(), "IQuest partial NeoX RoPE");
}

extern "C" int ds4_gpu_iquest_router(ds4_gpu_tensor *ids, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, uint32_t rows) {
    if (!rows || rows > 8192 || !iq_tensor(ids, (uint64_t)rows * IQ_USED * sizeof(unsigned)) ||
        !iq_tensor(weights, (uint64_t)rows * IQ_USED * sizeof(float)) ||
        !iq_tensor(logits, (uint64_t)rows * IQ_EXPERTS * sizeof(float))) { return 0; }
    iquest_router_kernel<<<rows, 1, 0, ds4_current_stream()>>>(
        (unsigned *)ids->ptr, (float *)weights->ptr, (const float *)logits->ptr);
    return cuda_ok(cudaGetLastError(), "IQuest normalized softmax router");
}

extern "C" int ds4_gpu_iquest_kv(ds4_gpu_tensor *cache,
        const ds4_gpu_tensor *key, const ds4_gpu_tensor *value,
        const ds4_gpu_tensor *positions, uint32_t rows, uint32_t capacity) {
    const uint64_t bytes = (uint64_t)rows * IQ_KV_HEADS * IQ_HEAD * sizeof(float);
    if (!rows || rows > 8192 || rows > capacity || capacity > IQ_CONTEXT ||
        !iq_tensor(key, bytes) || !iq_tensor(value, bytes) ||
        !iq_tensor(cache, (uint64_t)capacity * IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8)) ||
        !iq_tensor(positions, (uint64_t)rows * sizeof(unsigned))) { return 0; }
    iquest_kv_kernel<<<rows * IQ_Q8_ROW_BLOCKS, IQ_Q8_BLOCK, 0, ds4_current_stream()>>>(
        (iquest_q8 *)cache->ptr, (const float *)key->ptr, (const float *)value->ptr,
        (const unsigned *)positions->ptr, rows, capacity);
    return cuda_ok(cudaGetLastError(), "IQuest Q8_0 KV store");
}

extern "C" int ds4_gpu_iquest_attn(ds4_gpu_tensor *out, const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *cache, const ds4_gpu_tensor *positions,
        const void *map, uint64_t size, uint64_t offset,
        uint32_t rows, uint32_t capacity, uint32_t window) {
    const uint64_t bytes = (uint64_t)rows * IQ_HEADS * IQ_HEAD * sizeof(float);
    const uint64_t sink_bytes = IQ_KV_HEADS * IQ_HEAD * sizeof(float);
    if (!rows || rows > 8192 || !capacity || capacity > IQ_CONTEXT || rows > capacity ||
        (window != 0 && window != IQ_WINDOW && window != IQ_DRAFT_WINDOW) ||
        !iq_tensor(out, bytes) || !iq_tensor(query, bytes) ||
        !iq_tensor(cache, (uint64_t)capacity * IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8)) ||
        !iq_tensor(positions, (uint64_t)rows * sizeof(unsigned)) ||
        offset > size || sink_bytes > size - offset) { return 0; }
    const float *sink = (const float *)cuda_model_range_ptr(map, offset, sink_bytes, "IQuest learned sink");
    if (!sink) { return 0; }
    // The four-head launch is measured at the default prefill width. Tails,
    // decode and the recursive draft window keep the retained reduction.
    if (iq_attn_shuffle() && rows == IQ_PREFILL &&
        (window == 0 || window == IQ_WINDOW) && iq_attn_warp()) {
        if (iq_attn_tiled()) {
            // Borrow LSE scratch across adjacent launches on the same stream.
            // The sink consumes it before subsequent projections reuse it.
            float *lse = (float *)cuda_tmp_alloc((uint64_t)rows * IQ_HEADS * sizeof(float),
                                                "IQuest tiled LSE");
            if (!lse) { return 0; }
            iq_prefill::prefill<<<dim3((rows + iq_prefill::TQ - 1) / iq_prefill::TQ, IQ_HEADS),
                iq_prefill::THREADS, 0, ds4_current_stream()>>>(
                (float *)out->ptr, lse, (const float *)query->ptr,
                (const iquest_q8 *)cache->ptr, (const unsigned *)positions->ptr,
                rows, capacity, window);
            if (!cuda_ok(cudaGetLastError(), "IQuest tiled ordinary attention")) { return 0; }
            iq_prefill::sink<<<dim3(rows, IQ_HEADS / IQ_ATTN_HEADS_PER_BLOCK),
                IQ_ATTN_HEADS_PER_BLOCK * IQ_WARP_WIDTH, 0, ds4_current_stream()>>>(
                (float *)out->ptr, lse, (const float *)query->ptr, sink, rows);
            return cuda_ok(cudaGetLastError(), "IQuest tiled learned sink");
        }
        iquest_attn_warp_kernel<<<dim3(rows, IQ_HEADS / IQ_ATTN_HEADS_PER_BLOCK),
            IQ_ATTN_HEADS_PER_BLOCK * IQ_WARP_WIDTH, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)query->ptr, (const iquest_q8 *)cache->ptr,
            sink, (const unsigned *)positions->ptr, capacity, window);
    } else if (iq_attn_shuffle()) {
        iquest_attn_shuffle_kernel<<<dim3(rows, IQ_HEADS), IQ_HEAD, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)query->ptr, (const iquest_q8 *)cache->ptr,
            sink, (const unsigned *)positions->ptr, capacity, window);
    } else {
        iquest_attn_kernel<<<dim3(rows, IQ_HEADS), IQ_HEAD, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)query->ptr, (const iquest_q8 *)cache->ptr,
            sink, (const unsigned *)positions->ptr, capacity, window);
    }
    return cuda_ok(cudaGetLastError(), "IQuest learned-key Q8_0 attention");
}

extern "C" int ds4_gpu_iquest_add(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual,
        const ds4_gpu_tensor *branch, uint32_t rows, float scale) {
    const uint64_t count = (uint64_t)rows * IQ_EMBED, bytes = count * sizeof(float);
    if (!rows || rows > 8192 || !isfinite(scale) || !iq_tensor(out, bytes) ||
        !iq_tensor(residual, bytes) || !iq_tensor(branch, bytes)) { return 0; }
    iquest_add_kernel<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)residual->ptr, (const float *)branch->ptr, count, scale);
    return cuda_ok(cudaGetLastError(), "IQuest scaled residual");
}

extern "C" int ds4_gpu_iquest_sum(ds4_gpu_tensor *out, const ds4_gpu_tensor *experts,
        const ds4_gpu_tensor *weights, uint32_t rows) {
    const uint64_t count = (uint64_t)rows * IQ_EMBED;
    if (!rows || rows > IQ_PREFILL_MAX || !iq_tensor(out, count * sizeof(float)) ||
        !iq_tensor(experts, count * IQ_USED * sizeof(float)) ||
        !iq_tensor(weights, (uint64_t)rows * IQ_USED * sizeof(float))) { return 0; }
    iquest_sum_kernel<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)experts->ptr, (const float *)weights->ptr, rows);
    return cuda_ok(cudaGetLastError(), "IQuest weighted expert sum");
}

extern "C" int ds4_gpu_iquest_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up, uint64_t count) {
    if (!count || count > (uint64_t)IQ_PREFILL_MAX * IQ_DENSE ||
        !iq_tensor(out, count * sizeof(float)) || !iq_tensor(gate, count * sizeof(float)) ||
        !iq_tensor(up, count * sizeof(float))) { return 0; }
    iquest_swiglu_kernel<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, count);
    return cuda_ok(cudaGetLastError(), "IQuest BF16 fused SwiGLU");
}

extern "C" int ds4_gpu_iquest_expert(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *ids, const void *map, uint64_t size, uint64_t offset,
        uint32_t type, uint32_t in, uint32_t width, uint32_t rows,
        uint32_t used, uint32_t input_used) {
    const uint64_t bytes = (uint64_t)IQ_EXPERTS * in * width * (type == 0 ? 4 : 2);
    if (!rows || rows > IQ_PREFILL_MAX || used != IQ_USED || (input_used != 1 && input_used != used) ||
        (type != 0 && type != 1 && type != 30) || offset > size || bytes > size - offset ||
        !iq_tensor(out, (uint64_t)rows * used * width * sizeof(float)) ||
        !iq_tensor(x, (uint64_t)rows * input_used * in * sizeof(float)) ||
        !iq_tensor(ids, (uint64_t)rows * used * sizeof(unsigned))) { return 0; }
    const void *weight = cuda_model_range_ptr(map, offset, bytes, "IQuest reference experts");
    if (!weight) { return 0; }
    iquest_expert_kernel<<<dim3(width, rows * used), 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)x->ptr, (const unsigned *)ids->ptr,
        weight, type, in, width, rows, used, input_used);
    return cuda_ok(cudaGetLastError(), "IQuest reference expert projection");
}
