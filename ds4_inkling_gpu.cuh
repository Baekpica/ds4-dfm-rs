/* Inkling source-interleaved-v1 primitives, included by the CUDA backend. */

enum {
    INKLING_SCONV_TAPS = 4,
    INKLING_SCONV_HISTORY = INKLING_SCONV_TAPS - 1,
    INKLING_SCONV_MAX_CHANNELS = 4096,
    INKLING_THREADS = 256,
    INKLING_MAX_BLOCKS = 65535,
    INKLING_ROUTED = 256,
    INKLING_USED = 6,
    INKLING_SHARED = 2,
    INKLING_ACTIVE = INKLING_USED + INKLING_SHARED,
    INKLING_LOGITS = INKLING_ROUTED + INKLING_SHARED,
    INKLING_WARP = 32,
    INKLING_ROUTE_WARPS = 4,
    INKLING_KEY_SHIFT = 16,
    INKLING_INDEX_MASK = (1u << INKLING_KEY_SHIFT) - 1,
    INKLING_FF_MAX = 16384,
};
static constexpr uint32_t INKLING_FLOAT_SIGN = UINT32_C(1) << 31;

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

static __device__ __forceinline__ float inkling_sigmoid(float x) {
    const float e = __expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

static __global__ void inkling_route_kernel(
        int32_t *ids, float *routed, float *shared, const float *logits,
        const float *bias, const float *scale, uint32_t rows, uint32_t stride) {
    const uint32_t row = blockIdx.x * INKLING_ROUTE_WARPS + threadIdx.x / INKLING_WARP;
    const uint32_t lane = threadIdx.x % INKLING_WARP;
    if (row >= rows) { return; }
    const float *x = logits + (uint64_t)row * stride;
    unsigned long long keys[INKLING_ROUTED / INKLING_WARP];
    #pragma unroll
    for (int j = 0; j < INKLING_ROUTED / INKLING_WARP; j++) {
        const uint32_t expert = lane + j * INKLING_WARP;
        const float score = __fadd_rn(inkling_sigmoid(x[expert]), bias[expert]);
        const uint32_t bits = __float_as_uint(score);
        const uint32_t ordered = bits ^ ((bits & INKLING_FLOAT_SIGN) ? UINT32_MAX : INKLING_FLOAT_SIGN);
        // Source sort key: descending float score, then ascending expert ID.
        keys[j] = ((unsigned long long)ordered << INKLING_KEY_SHIFT) | (INKLING_ROUTED - expert);
    }
    int chosen[INKLING_USED];
    #pragma unroll
    for (int k = 0; k < INKLING_USED; k++) {
        unsigned long long best = 0;
        #pragma unroll
        for (int j = 0; j < INKLING_ROUTED / INKLING_WARP; j++) {
            best = best > keys[j] ? best : keys[j];
        }
        #pragma unroll
        for (int delta = INKLING_WARP / 2; delta > 0; delta /= 2) {
            const unsigned long long other = __shfl_xor_sync(UINT32_MAX, best, delta);
            best = best > other ? best : other;
        }
        chosen[k] = INKLING_ROUTED - (best & INKLING_INDEX_MASK);
        #pragma unroll
        for (int j = 0; j < INKLING_ROUTED / INKLING_WARP; j++) {
            if (keys[j] == best) { keys[j] = 0; }
        }
    }
    float lp = -INFINITY;
    if (lane < INKLING_ACTIVE) {
        const float value = x[lane < INKLING_USED ? chosen[lane] : INKLING_ROUTED + lane - INKLING_USED];
        // Normalize raw logits, not selection scores. Log space also handles
        // all-negative rows whose sigmoid probabilities underflow to zero.
        lp = fminf(value, 0.0f) - log1pf(__expf(-fabsf(value)));
    }
    float maximum = lp;
    #pragma unroll
    for (int delta = INKLING_WARP / 2; delta > 0; delta /= 2) {
        maximum = fmaxf(maximum, __shfl_xor_sync(UINT32_MAX, maximum, delta));
    }
    float sum = lane < INKLING_ACTIVE ? __expf(lp - maximum) : 0.0f;
    #pragma unroll
    for (int delta = INKLING_WARP / 2; delta > 0; delta /= 2) {
        sum = __fadd_rn(sum, __shfl_xor_sync(UINT32_MAX, sum, delta));
    }
    if (lane >= INKLING_ACTIVE) { return; }
    const float lse = __fadd_rn(maximum, logf(sum));
    const float weight = __fmul_rn(__fmul_rn(__expf(lp - lse), (float)INKLING_ACTIVE), *scale);
    if (lane < INKLING_USED) {
        ids[(uint64_t)row * INKLING_USED + lane] = chosen[lane];
        routed[(uint64_t)row * INKLING_USED + lane] = weight;
    } else {
        shared[(uint64_t)row * INKLING_SHARED + lane - INKLING_USED] = weight;
    }
}

extern "C" int ds4_gpu_inkling_route(
        ds4_gpu_tensor *ids, ds4_gpu_tensor *routed, ds4_gpu_tensor *shared,
        const ds4_gpu_tensor *logits, const void *model_map, uint64_t model_size,
        uint64_t bias_offset, uint64_t scale_offset, uint32_t rows, uint32_t stride) {
    if (!ids || !routed || !shared || !logits || !model_map || !rows ||
        stride < INKLING_LOGITS || (uint64_t)rows > UINT64_MAX / sizeof(float) / stride ||
        bias_offset % sizeof(float) || scale_offset % sizeof(float) ||
        bias_offset > model_size || INKLING_ROUTED * sizeof(float) > model_size - bias_offset ||
        scale_offset > model_size || sizeof(float) > model_size - scale_offset) { return 0; }
    const uint64_t xbytes = (uint64_t)rows * stride * sizeof(float);
    if (logits->bytes < xbytes) { return 0; }
    const ds4_gpu_tensor *outputs[] = {ids, routed, shared};
    const uint64_t bytes[] = {(uint64_t)rows * INKLING_USED * sizeof(int32_t),
        (uint64_t)rows * INKLING_USED * sizeof(float), (uint64_t)rows * INKLING_SHARED * sizeof(float)};
    for (unsigned i = 0; i < sizeof(outputs) / sizeof(outputs[0]); i++) {
        if (outputs[i]->bytes < bytes[i] || inkling_overlap(outputs[i], bytes[i], logits, xbytes)) { return 0; }
        for (unsigned j = 0; j < i; j++) {
            if (inkling_overlap(outputs[i], bytes[i], outputs[j], bytes[j])) { return 0; }
        }
    }
    const float *bias = (const float *)cuda_resolve_weight_ptr(model_map, bias_offset,
        INKLING_ROUTED * sizeof(float), 0, "inkling gate bias");
    const float *scale = (const float *)cuda_resolve_weight_ptr(model_map, scale_offset,
        sizeof(float), 0, "inkling gate scale");
    if (!bias || !scale) { return 0; }
    const unsigned blocks = ((uint64_t)rows + INKLING_ROUTE_WARPS - 1) / INKLING_ROUTE_WARPS;
    inkling_route_kernel<<<blocks, INKLING_WARP * INKLING_ROUTE_WARPS, 0, ds4_current_stream()>>>(
        (int32_t *)ids->ptr, (float *)routed->ptr, (float *)shared->ptr,
        (const float *)logits->ptr, bias, scale, rows, stride);
    return cuda_ok(cudaGetLastError(), "Inkling route launch");
}

static __global__ void inkling_swiglu_kernel(
        float *out, const float *pairs, const float *gamma, uint32_t width, uint64_t count) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (uint64_t)gridDim.x * blockDim.x) {
        const float gate = inkling_bf16(pairs[2 * i]), up = inkling_bf16(pairs[2 * i + 1]);
        float value = __fmul_rn(__fmul_rn(gate, inkling_sigmoid(gate)), up);
        if (gamma) { value = __fmul_rn(value, gamma[i / width]); }
        out[i] = inkling_bf16(value);
    }
}

extern "C" int ds4_gpu_inkling_swiglu(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *pairs, const ds4_gpu_tensor *gamma,
        uint32_t width, uint32_t rows) {
    if (!out || !pairs || !width || !rows || width > INKLING_FF_MAX) { return 0; }
    const uint64_t count = (uint64_t)width * rows, bytes = count * sizeof(float);
    const uint64_t gbytes = (uint64_t)rows * sizeof(float);
    if (out->bytes < bytes || pairs->bytes < 2 * bytes || inkling_overlap(out, bytes, pairs, 2 * bytes) ||
        (gamma && (gamma->bytes < gbytes || inkling_overlap(out, bytes, gamma, gbytes)))) { return 0; }
    const uint64_t grid = (count + INKLING_THREADS - 1) / INKLING_THREADS;
    const unsigned blocks = (unsigned)(grid < INKLING_MAX_BLOCKS ? grid : INKLING_MAX_BLOCKS);
    inkling_swiglu_kernel<<<blocks, INKLING_THREADS, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)pairs->ptr, gamma ? (const float *)gamma->ptr : NULL, width, count);
    return cuda_ok(cudaGetLastError(), "Inkling SwiGLU launch");
}

static __global__ void inkling_combine_kernel(
        float *out, const float *routed, const float *shared, const float *weights,
        uint32_t width, uint64_t count) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t row = i / width, c = i % width;
        float rsum = 0.0f, ssum = 0.0f;
        #pragma unroll
        for (int k = 0; k < INKLING_USED; k++) {
            const float value = inkling_bf16(routed[(row * INKLING_USED + k) * width + c]);
            rsum = __fadd_rn(rsum, __fmul_rn(value, weights[row * INKLING_USED + k]));
        }
        #pragma unroll
        for (int k = 0; k < INKLING_SHARED; k++) {
            ssum = __fadd_rn(ssum, inkling_bf16(shared[(row * INKLING_SHARED + k) * width + c]));
        }
        // Routed gamma belongs after down projection. Shared gamma was already
        // applied inside SwiGLU, before its down projection; do not apply it twice.
        out[i] = inkling_bf16(__fadd_rn(inkling_bf16(rsum), inkling_bf16(ssum)));
    }
}

extern "C" int ds4_gpu_inkling_combine(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *routed, const ds4_gpu_tensor *shared,
        const ds4_gpu_tensor *weights, uint32_t width, uint32_t rows) {
    if (!out || !routed || !shared || !weights || !width || !rows || width > INKLING_SCONV_MAX_CHANNELS) { return 0; }
    const uint64_t count = (uint64_t)width * rows, bytes = count * sizeof(float);
    const uint64_t wbytes = (uint64_t)rows * INKLING_USED * sizeof(float);
    if (out->bytes < bytes || routed->bytes < INKLING_USED * bytes || shared->bytes < INKLING_SHARED * bytes ||
        weights->bytes < wbytes || inkling_overlap(out, bytes, routed, INKLING_USED * bytes) ||
        inkling_overlap(out, bytes, shared, INKLING_SHARED * bytes) || inkling_overlap(out, bytes, weights, wbytes)) { return 0; }
    const uint64_t grid = (count + INKLING_THREADS - 1) / INKLING_THREADS;
    const unsigned blocks = (unsigned)(grid < INKLING_MAX_BLOCKS ? grid : INKLING_MAX_BLOCKS);
    inkling_combine_kernel<<<blocks, INKLING_THREADS, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)routed->ptr, (const float *)shared->ptr,
        (const float *)weights->ptr, width, count);
    return cuda_ok(cudaGetLastError(), "Inkling MoE combine launch");
}
