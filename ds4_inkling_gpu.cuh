/* Inkling source-interleaved-v1 primitives, included by the CUDA backend. */

enum {
    INKLING_SCONV_TAPS = 4,
    INKLING_SCONV_HISTORY = INKLING_SCONV_TAPS - 1,
    INKLING_SCONV_MAX_CHANNELS = 4096,
    INKLING_THREADS = 256,
    INKLING_MAX_BLOCKS = 65535,
};

static __device__ __forceinline__ float inkling_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

static __global__ void inkling_sconv_kernel(
        float *out, const float *x, const float *history,
        const uint16_t *weight, uint32_t channels, uint64_t count) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (uint64_t)gridDim.x * blockDim.x) {
        const uint32_t c = i % channels;
        const int64_t t = i / channels;
        float sum = 0.0f;
        #pragma unroll
        for (int tap = 0; tap < INKLING_SCONV_TAPS; tap++) {
            const int64_t at = t - INKLING_SCONV_HISTORY + tap;
            const float value = at < 0
                ? history[(at + INKLING_SCONV_HISTORY) * channels + c]
                : x[at * channels + c];
            const float w = __uint_as_float((uint32_t)weight[c * INKLING_SCONV_TAPS + tap] << 16);
            sum = fmaf(inkling_bf16(value), w, sum);
        }
        // Match the source FP32 accumulation/residual, then its BF16 store.
        out[i] = inkling_bf16(__fadd_rn(sum, inkling_bf16(x[i])));
    }
}

static __global__ void inkling_history_kernel(
        float *next, const float *x, const float *history,
        uint32_t channels, uint32_t rows) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }
    // Load a channel's complete window before writing: next may equal history.
    float values[INKLING_SCONV_HISTORY];
    #pragma unroll
    for (int h = 0; h < INKLING_SCONV_HISTORY; h++) {
        const int64_t at = (int64_t)rows - INKLING_SCONV_HISTORY + h;
        values[h] = inkling_bf16(at < 0
            ? history[(at + INKLING_SCONV_HISTORY) * channels + c]
            : x[at * channels + c]);
    }
    #pragma unroll
    for (int h = 0; h < INKLING_SCONV_HISTORY; h++) {
        next[h * channels + c] = values[h];
    }
}

static bool inkling_overlap(const ds4_gpu_tensor *a, uint64_t a_bytes,
                            const ds4_gpu_tensor *b, uint64_t b_bytes) {
    const uintptr_t pa = (uintptr_t)a->ptr, pb = (uintptr_t)b->ptr;
    return pa <= pb ? pb - pa < a_bytes : pa - pb < b_bytes;
}

extern "C" int ds4_gpu_inkling_sconv(
        ds4_gpu_tensor *out, ds4_gpu_tensor *next,
        const ds4_gpu_tensor *x, const ds4_gpu_tensor *history,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t channels, uint32_t rows) {
    if (!out || !next || !x || !history || !model_map || !channels || !rows ||
        channels > INKLING_SCONV_MAX_CHANNELS) {
        return 0;
    }
    const uint64_t count = (uint64_t)channels * rows;
    const uint64_t bytes = count * sizeof(float);
    const uint64_t history_bytes = (uint64_t)channels * INKLING_SCONV_HISTORY * sizeof(float);
    const uint64_t weight_bytes = (uint64_t)channels * INKLING_SCONV_TAPS * sizeof(uint16_t);
    if (out->bytes < bytes || x->bytes < bytes || history->bytes < history_bytes ||
        next->bytes < history_bytes || weight_offset % sizeof(uint16_t) != 0 ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset ||
        inkling_overlap(out, bytes, x, bytes) ||
        inkling_overlap(out, bytes, history, history_bytes) ||
        inkling_overlap(out, bytes, next, history_bytes) ||
        inkling_overlap(next, history_bytes, x, bytes) ||
        (next->ptr != history->ptr && inkling_overlap(next, history_bytes, history, history_bytes))) {
        return 0;
    }
    const uint16_t *weight = (const uint16_t *)cuda_resolve_weight_ptr(
        model_map, weight_offset, weight_bytes, 0, "inkling sconv");
    if (!weight) {
        return 0;
    }
    const uint64_t grid = (count + INKLING_THREADS - 1) / INKLING_THREADS;
    const unsigned blocks = (unsigned)(grid < INKLING_MAX_BLOCKS ? grid : INKLING_MAX_BLOCKS);
    inkling_sconv_kernel<<<blocks, INKLING_THREADS, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)x->ptr, (const float *)history->ptr,
        weight, channels, count);
    if (!cuda_ok(cudaGetLastError(), "Inkling sconv launch")) {
        return 0;
    }
    // The ordered update cannot overwrite history while another CTA reads it.
    inkling_history_kernel<<<(channels + INKLING_THREADS - 1) / INKLING_THREADS,
                              INKLING_THREADS, 0, ds4_current_stream()>>>(
        (float *)next->ptr, (const float *)x->ptr, (const float *)history->ptr, channels, rows);
    return cuda_ok(cudaGetLastError(), "Inkling history launch");
}
