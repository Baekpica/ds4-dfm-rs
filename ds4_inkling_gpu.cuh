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
    INKLING_LINEAR_TILE = 8,
    INKLING_LINEAR_WARPS = 8,
    INKLING_LT_TOKENS = 16,
    INKLING_LT_ROWS = 4,
    INKLING_LT_WARPS = 4,
    INKLING_LT_SLAB = 4,
    INKLING_LT_MIN_BLOCKS = 2,
    INKLING_LT_PANEL_TOKENS = 512,
    INKLING_LT_PANEL_MIN_ROWS = 4097,
    INKLING_LINEAR_DECODE_WARPS = 4,
    INKLING_ROUTE_WARPS = 4,
    INKLING_KEY_SHIFT = 16,
    INKLING_INDEX_MASK = (1u << INKLING_KEY_SHIFT) - 1,
    INKLING_FF_MAX = 16384,
    INKLING_HEADS = 32,
    INKLING_HEAD_DIM = 128,
    INKLING_REL_DIM = 16,
    INKLING_LOCAL_EXTENT = 512,
    INKLING_GLOBAL_EXTENT = 1024,
    INKLING_TAU_FLOOR = 128000,
    INKLING_KV_HEADS = 8,
    INKLING_KV_WIDTH = INKLING_KV_HEADS * INKLING_HEAD_DIM,
    INKLING_KV_ROW = 2 * INKLING_KV_WIDTH,
    INKLING_ATTN_WARPS = 4,
    INKLING_ATTN_DPL = INKLING_HEAD_DIM / INKLING_WARP,
    INKLING_ATTN_GROUP = INKLING_HEADS / INKLING_KV_HEADS,
    INKLING_ATTN_GROUP_MIN_ROWS = 16,
    INKLING_ATTN_GROUP_MIN_BLOCKS = 6,
    INKLING_ATTN_HEAD_LANES = INKLING_WARP / INKLING_ATTN_GROUP,
    INKLING_NORM_MAX = 16384,
    INKLING_AUDIO_BINS = 80,
    INKLING_AUDIO_LEVELS = 16,
    INKLING_MEDIA_WIDTH = 4096,
    INKLING_MOE_MIDDLE = 2048,
    INKLING_PREFILL_MAX = 8192,
    INKLING_Q8_BLOCK = 32,
    INKLING_Q8_COLUMNS = 8,
    INKLING_IQ2_XXS = 16, /* DS4_TENSOR_IQ2_XXS / GGML_TYPE_IQ2_XXS */
    INKLING_IQ2_XS = 17,  /* DS4_TENSOR_IQ2_XS / GGML_TYPE_IQ2_XS */
};
static constexpr uint32_t INKLING_FLOAT_SIGN = UINT32_C(1) << 31;
static constexpr float INKLING_TAU_ALPHA = 0.1f;
static constexpr float INKLING_RMS_EPS = 1e-6f;
static constexpr float INKLING_GELU_SCALE = 0.7071067811865475244f;

static __device__ __forceinline__ float inkling_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

template<unsigned TOKENS>
static __global__ void inkling_linear_kernel(
        float *out, const __nv_bfloat16 *w, const __nv_bfloat16 *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    const unsigned lane = threadIdx.x % INKLING_WARP;
    const uint64_t groups = TOKENS == 1 ? 1 : ((uint64_t)rows + TOKENS - 1) / TOKENS;
    const uint64_t total = (uint64_t)out_dim * groups * TOKENS;
    const uint64_t step = (uint64_t)gridDim.x * blockDim.x / INKLING_WARP;
    for (uint64_t warp = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) / INKLING_WARP;
         warp < total; warp += step) {
        // Keep one accumulator per output; adjacent warps reuse a weight row.
        const uint32_t row = TOKENS == 1 ? warp : (warp / TOKENS) / groups;
        const uint32_t tok = TOKENS == 1 ? 0 : ((warp / TOKENS) % groups) * TOKENS + warp % TOKENS;
        if (tok >= rows) { continue; }
        const uint4 *wr = (const uint4 *)(w + (uint64_t)row * in_dim);
        const uint4 *xr = (const uint4 *)(x + (uint64_t)tok * in_dim);
        float sum = 0.0f;
        for (uint32_t i = lane; i < in_dim / 8u; i += INKLING_WARP) {
            sum += bf16x8_dot(wr[i], xr[i]);
        }
        #pragma unroll
        for (unsigned off = INKLING_WARP / 2; off; off /= 2) {
            sum += __shfl_xor_sync(0xffffffffu, sum, off);
        }
        if (lane == 0) {
            // Match add_scale(..., 1), including its FP32 multiply boundary.
            out[(uint64_t)tok * out_dim + row] =
                inkling_bf16(__fmul_rn(inkling_bf16(sum), 1.0f));
        }
    }
}

/* Prefill tile: one CTA owns WARPS*ROWS weight rows and TOKENS tokens. The
 * token slab lives in shared memory, so every weight vector is read once per
 * token group and every activation vector once per row slab. Each output
 * keeps the lane K stripe, eight-FMA chains, FP32 adds and XOR tree of
 * inkling_linear_kernel, so results are byte-identical.
 *
 *   xs[t][s][lane]   uint4 (8 BF16) of token t at K step s0+s, lane stripe
 *   acc[r][t]        one accumulator per (row, token), same order as before
 */
template<unsigned TOKENS, unsigned ROWS, unsigned WARPS, unsigned SLAB, unsigned PANEL_GROUPS = 0>
__launch_bounds__(WARPS * INKLING_WARP, INKLING_LT_MIN_BLOCKS)
static __global__ void inkling_linear_tile_kernel(
        float *out, const __nv_bfloat16 *w, const __nv_bfloat16 *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    __shared__ uint4 xs[TOKENS][SLAB][INKLING_WARP];
    const unsigned warp = threadIdx.x / INKLING_WARP, lane = threadIdx.x % INKLING_WARP;
    const uint32_t groups = (rows + TOKENS - 1) / TOKENS;
    const uint32_t steps = in_dim / (8 * INKLING_WARP);
    const uint64_t jobs = (uint64_t)groups * (out_dim / (WARPS * ROWS));
    for (uint64_t job = blockIdx.x; job < jobs; job += gridDim.x) {
        uint32_t tok0, row0;
        if constexpr (PANEL_GROUPS == 0) {
            // Preserve the existing schedule for short inputs and rollback.
            tok0 = (job % groups) * TOKENS;
            row0 = (job / groups) * (WARPS * ROWS) + warp * ROWS;
        } else {
            // Sweep weight rows within a bounded token panel to reuse inputs
            // in L2. The last panel uses its actual width, without padding jobs.
            const uint64_t panel_jobs = (uint64_t)PANEL_GROUPS * (out_dim / (WARPS * ROWS));
            const uint32_t base = (job / panel_jobs) * PANEL_GROUPS;
            const uint32_t width = min(PANEL_GROUPS, groups - base);
            const uint64_t local = job % panel_jobs;
            tok0 = (base + local % width) * TOKENS;
            row0 = (local / width) * (WARPS * ROWS) + warp * ROWS;
        }
        const uint4 *wr[ROWS];
        #pragma unroll
        for (unsigned r = 0; r < ROWS; r++) {
            wr[r] = (const uint4 *)(w + (uint64_t)(row0 + r) * in_dim);
        }
        float acc[ROWS][TOKENS] = {};
        for (uint32_t s0 = 0; s0 < steps; s0 += SLAB) {
            for (unsigned i = threadIdx.x; i < TOKENS * SLAB * INKLING_WARP; i += WARPS * INKLING_WARP) {
                const unsigned t = i / (SLAB * INKLING_WARP), s = i / INKLING_WARP % SLAB, l = i % INKLING_WARP;
                const uint32_t tok = tok0 + t;
                xs[t][s][l] = tok < rows && s0 + s < steps
                    ? ((const uint4 *)(x + (uint64_t)tok * in_dim))[(s0 + s) * INKLING_WARP + l]
                    : make_uint4(0u, 0u, 0u, 0u);
            }
            __syncthreads();
            #pragma unroll
            for (unsigned s = 0; s < SLAB; s++) {
                if (s0 + s >= steps) { break; }
                uint4 wv[ROWS];
                #pragma unroll
                for (unsigned r = 0; r < ROWS; r++) { wv[r] = wr[r][(s0 + s) * INKLING_WARP + lane]; }
                #pragma unroll
                for (unsigned t = 0; t < TOKENS; t++) {
                    const uint4 xv = xs[t][s][lane];
                    #pragma unroll
                    for (unsigned r = 0; r < ROWS; r++) { acc[r][t] += bf16x8_dot(wv[r], xv); }
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (unsigned r = 0; r < ROWS; r++) {
            #pragma unroll
            for (unsigned t = 0; t < TOKENS; t++) {
                float sum = acc[r][t];
                #pragma unroll
                for (unsigned off = INKLING_WARP / 2; off; off /= 2) {
                    sum += __shfl_xor_sync(0xffffffffu, sum, off);
                }
                if (lane == 0 && tok0 + t < rows) {
                    out[(uint64_t)(tok0 + t) * out_dim + row0 + r] =
                        inkling_bf16(__fmul_rn(inkling_bf16(sum), 1.0f));
                }
            }
        }
    }
}

extern "C" int ds4_gpu_inkling_linear(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    if (!out || !x || !model_map || !in_dim || !out_dim || !rows ||
        out_dim > UINT64_MAX / sizeof(uint16_t) / in_dim ||
        x->bytes / sizeof(float) / in_dim < rows ||
        out->bytes / sizeof(float) / out_dim < rows ||
        weight_offset > model_size ||
        ds4_tensor_device_idx(out) != ds4_tensor_device_idx(x)) {
        return -1;
    }
    const uint64_t weight_bytes = (uint64_t)out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset) { return -1; }
    if ((in_dim & 7u) || getenv("DS4_INKLING_NO_LINEAR") ||
        getenv("DS4_CUDA_NO_BF16_ROWS_WARP") ||
        (uint64_t)rows * in_dim > (uint64_t)INT_MAX * INKLING_THREADS) {
        return 0;
    }
    const int tier = ds4_tensor_device_idx(out);
    const __nv_bfloat16 *w = (const __nv_bfloat16 *)cuda_resolve_weight_ptr(
        model_map, weight_offset, weight_bytes, tier, "inkling BF16 linear");
    if (!w) { return -1; }
    if ((uintptr_t)w & 15u) { return 0; }
    __nv_bfloat16 *xb = (__nv_bfloat16 *)cuda_tmp_alloc_on(
        tier, (uint64_t)rows * in_dim * sizeof(__nv_bfloat16), "inkling BF16 input");
    if (!xb) { return -1; }
    const uint64_t count = (uint64_t)rows * in_dim;
    const cudaStream_t stream = cuda_decode_stream();
    f32_to_bf16_kernel<<<(count + INKLING_THREADS - 1) / INKLING_THREADS,
                         INKLING_THREADS, 0, stream>>>(xb, (const float *)x->ptr, count);
    if (!cuda_ok(cudaGetLastError(), "Inkling BF16 input launch")) { return -1; }
    cuda_norm_q8_invalidate(out->ptr);
    if (rows >= INKLING_LT_TOKENS && in_dim % (8 * INKLING_WARP) == 0 &&
        out_dim % (INKLING_LT_WARPS * INKLING_LT_ROWS) == 0 &&
        !getenv("DS4_INKLING_NO_LINEAR_TILE")) {
        const uint64_t jobs = (((uint64_t)rows + INKLING_LT_TOKENS - 1) / INKLING_LT_TOKENS) *
            (out_dim / (INKLING_LT_WARPS * INKLING_LT_ROWS));
        const unsigned blocks = jobs < INKLING_MAX_BLOCKS ? jobs : INKLING_MAX_BLOCKS;
        // Restrict the panel schedule to measured wide q/k/v/r/o shapes.
        if (rows >= INKLING_LT_PANEL_MIN_ROWS && in_dim == INKLING_MEDIA_WIDTH &&
            (out_dim == INKLING_MEDIA_WIDTH || out_dim == INKLING_KV_WIDTH ||
             out_dim == INKLING_HEADS * INKLING_REL_DIM) &&
            !getenv("DS4_INKLING_NO_LINEAR_PANEL")) {
            inkling_linear_tile_kernel<INKLING_LT_TOKENS, INKLING_LT_ROWS, INKLING_LT_WARPS,
                INKLING_LT_SLAB, INKLING_LT_PANEL_TOKENS / INKLING_LT_TOKENS>
                <<<blocks, INKLING_LT_WARPS * INKLING_WARP, 0, stream>>>(
                (float *)out->ptr, w, xb, in_dim, out_dim, rows);
        } else {
            inkling_linear_tile_kernel<INKLING_LT_TOKENS, INKLING_LT_ROWS, INKLING_LT_WARPS, INKLING_LT_SLAB>
                <<<blocks, INKLING_LT_WARPS * INKLING_WARP, 0, stream>>>(
                (float *)out->ptr, w, xb, in_dim, out_dim, rows);
        }
        return cuda_ok(cudaGetLastError(), "Inkling BF16 tile launch") ? 1 : -1;
    }
    unsigned tile = INKLING_LINEAR_TILE;
    while (tile > rows) { tile /= 2; }
    const unsigned warps = rows == 1 ? INKLING_LINEAR_DECODE_WARPS : INKLING_LINEAR_WARPS;
    const uint64_t outputs = (uint64_t)out_dim * (((uint64_t)rows + tile - 1) / tile) * tile;
    const uint64_t grid = (outputs + warps - 1) / warps;
    const unsigned blocks = grid < INKLING_MAX_BLOCKS ? grid : INKLING_MAX_BLOCKS;
    #define IK_LINEAR_LAUNCH(T) inkling_linear_kernel<T><<<blocks, warps * INKLING_WARP, 0, stream>>>( \
        (float *)out->ptr, w, xb, in_dim, out_dim, rows)
    switch (tile) {
    case 8: IK_LINEAR_LAUNCH(8); break;
    case 4: IK_LINEAR_LAUNCH(4); break;
    case 2: IK_LINEAR_LAUNCH(2); break;
    default: IK_LINEAR_LAUNCH(1); break;
    }
    #undef IK_LINEAR_LAUNCH
    return cuda_ok(cudaGetLastError(), "Inkling BF16 projection launch") ? 1 : -1;
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

extern "C" int ds4_gpu_inkling_q8(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t weight_bytes, uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    if (!out || !x || !model_map || !rows || weight_offset > model_size ||
        weight_bytes > model_size - weight_offset) { return -1; }
    if (rows <= 1 || rows > INKLING_PREFILL_MAX ||
        !((in_dim == INKLING_MEDIA_WIDTH && out_dim == 2 * INKLING_FF_MAX) ||
          (in_dim == INKLING_FF_MAX && out_dim == INKLING_MEDIA_WIDTH))) { return 0; }
    const uint64_t required = (uint64_t)in_dim * out_dim / INKLING_Q8_BLOCK *
        sizeof(cuda_block_q8_0);
    const uint64_t xbytes = (uint64_t)rows * in_dim * sizeof(float);
    const uint64_t obytes = (uint64_t)rows * out_dim * sizeof(float);
    if (weight_bytes < required || x->bytes < xbytes || out->bytes < obytes ||
        inkling_overlap(out, obytes, x, xbytes) ||
        ds4_tensor_device_idx(out) != ds4_tensor_device_idx(x)) { return -1; }
    if (getenv("DS4_INKLING_NO_Q8_BATCH") || getenv("DS4_CUDA_NO_Q8_ALIGNED_NC") ||
        !cuda_q8_aligned_enabled() || !ds4_cuda_use_mmq()) { return 0; }
    const uint64_t aligned_bytes = ds4_mmq_q8_0_aligned_bytes(out_dim, in_dim);
    const void *weights = cuda_derived_weight_ptr(model_map, weight_offset, weight_bytes,
        CUDA_DERIVED_Q8_0_ALIGNED_DENSE, in_dim, out_dim, 1, aligned_bytes,
        "Inkling dense Q8 prefill");
    if (!weights) { return 0; }
    cuda_norm_q8_invalidate(out->ptr);
    // Prefill tiles read each aligned weight row once per call; other shapes
    // and the rollback keep the eight-column loop below.
    if (!getenv("DS4_INKLING_NO_Q8_TILE")) {
        const int rc = ds4_mmq_q8_0_aligned_dense_batch(weights, (const float *)x->ptr,
            (float *)out->ptr, out_dim, rows, in_dim, ds4_current_stream());
        if (rc < 0) { return -1; }
        if (rc == 0) { return 1; }
    }
    // Tile the width without entering the raw MMVQ/MMQ numerical paths.
    for (uint32_t row = 0; row < rows; row += INKLING_Q8_COLUMNS) {
        const uint32_t cols = std::min<uint32_t>(rows - row, INKLING_Q8_COLUMNS);
        if (ds4_mmq_q8_0_aligned_dense_vec(weights,
                (const float *)x->ptr + (uint64_t)row * in_dim,
                (float *)out->ptr + (uint64_t)row * out_dim,
                out_dim, cols, in_dim, ds4_current_stream()) != 0) { return -1; }
    }
    return 1;
}

extern "C" int ds4_gpu_inkling_routed(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *ids,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t weight_bytes, uint32_t type, uint32_t in_dim, uint32_t out_dim,
        uint32_t experts, uint32_t rows, uint32_t used) {
    if (!out || !x || !ids || !model_map || !rows || !used ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        return -1;
    }
    const unsigned active = experts == INKLING_SHARED ? INKLING_SHARED : INKLING_USED;
    const unsigned group = used > 1 ? 1 : active;
    const unsigned tokens = group ? rows / group : 0;
    if ((experts != INKLING_ROUTED && experts != INKLING_SHARED) ||
        (used != 1 && used != active) || out_dim != INKLING_MEDIA_WIDTH ||
        in_dim != (used > 1 ? INKLING_MEDIA_WIDTH : INKLING_MOE_MIDDLE) ||
        rows % group || tokens == 0 || tokens > INKLING_PREFILL_MAX) {
        return 0;
    }
    /* Decode (one source token) stays on the vec fallback except IQ2 SoA,
     * which must reuse the prefill MMVQ tile so full vs incremental match. */
    const int decode_one = tokens <= 1;
    if (decode_one) {
        const int xxs_ok = type == INKLING_IQ2_XXS &&
            getenv("DS4_INKLING_NO_IQ2_ALIGNED") == NULL;
        const int xs_ok = type == INKLING_IQ2_XS &&
            getenv("DS4_INKLING_NO_IQ2_XS_ALIGNED") == NULL;
        if (experts == INKLING_SHARED || (!xxs_ok && !xs_ok)) {
            return 0;
        }
    }
    const uint64_t required = ds4_mmq_inkling_wbytes(type, out_dim, in_dim, experts);
    if (!required) { return 0; }
    const uint64_t xbytes = (uint64_t)rows * in_dim * sizeof(float);
    const uint64_t obytes = (uint64_t)rows * used * out_dim * sizeof(float);
    const uint64_t ibytes = (uint64_t)rows * used * sizeof(int32_t);
    if (weight_bytes < required || x->bytes < xbytes || out->bytes < obytes ||
        ids->bytes < ibytes || inkling_overlap(out, obytes, x, xbytes) ||
        inkling_overlap(out, obytes, ids, ibytes) ||
        inkling_overlap(x, xbytes, ids, ibytes) ||
        ds4_tensor_device_idx(out) != ds4_tensor_device_idx(x) ||
        ds4_tensor_device_idx(out) != ds4_tensor_device_idx(ids)) { return -1; }
    if (getenv("DS4_INKLING_NO_MOE_BATCH") || !ds4_cuda_use_mmq()) { return 0; }
    // Fused IQ2_XXS w13: SoA tile when the owner artifact is present. The
    // kill switch derepacks into a raw scratch; range_ptr would hit the
    // excluded VMM hole.
    const void *weights = nullptr;
    if (type == INKLING_IQ2_XXS && experts != INKLING_SHARED) {
        const uint64_t aligned_bytes = ds4_mmq_iq2_xxs_aligned_bytes(out_dim, in_dim, experts);
        const void *aligned = aligned_bytes
            ? cuda_derived_weight_ptr(model_map, weight_offset, required,
                CUDA_DERIVED_IQ2_XXS_ALIGNED_MOE, in_dim, out_dim, experts,
                aligned_bytes, "Inkling routed IQ2 aligned")
            : nullptr;
        if (aligned && !getenv("DS4_INKLING_NO_IQ2_ALIGNED")) {
            cuda_norm_q8_invalidate(out->ptr);
            const int rc = ds4_mmq_inkling_moe_iq2_aligned(aligned, (const float *)x->ptr,
                (const int32_t *)ids->ptr, (float *)out->ptr,
                out_dim, in_dim, rows, experts, used, ds4_current_stream());
            return rc == 0 ? 1 : -1;
        }
        if (decode_one) { return 0; }
        if (aligned) {
            weights = cuda_moe_iq2_derepack_scratch(
                0, (const char *)aligned, weight_offset, required,
                out_dim, in_dim, experts, ds4_current_stream());
        }
    } else if (type == INKLING_IQ2_XS && experts != INKLING_SHARED) {
        const uint64_t aligned_bytes = ds4_mmq_iq2_xs_aligned_bytes(out_dim, in_dim, experts);
        const void *aligned = aligned_bytes
            ? cuda_derived_weight_ptr(model_map, weight_offset, required,
                CUDA_DERIVED_IQ2_XS_ALIGNED_MOE, in_dim, out_dim, experts,
                aligned_bytes, "Inkling routed IQ2_XS aligned")
            : nullptr;
        if (aligned && !getenv("DS4_INKLING_NO_IQ2_XS_ALIGNED")) {
            cuda_norm_q8_invalidate(out->ptr);
            const int rc = ds4_mmq_inkling_moe_iq2_xs_aligned(aligned, (const float *)x->ptr,
                (const int32_t *)ids->ptr, (float *)out->ptr,
                out_dim, in_dim, rows, experts, used, ds4_current_stream());
            return rc == 0 ? 1 : -1;
        }
        if (decode_one) { return 0; }
        if (aligned) {
            weights = cuda_moe_iq2_xs_derepack_scratch(
                (const char *)aligned, weight_offset, required,
                out_dim, in_dim, experts, ds4_current_stream());
        }
    }
    if (!weights) {
        weights = cuda_model_range_ptr(
            model_map, weight_offset, required, "Inkling batched experts");
    }
    if (!weights) { return -1; }
    cuda_norm_q8_invalidate(out->ptr);
    const int rc = ds4_mmq_inkling_moe(weights, type, (const float *)x->ptr,
        (const int32_t *)ids->ptr, (float *)out->ptr,
        out_dim, in_dim, rows, experts, used, ds4_current_stream());
    return rc == 0 ? 1 : -1;
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

static __global__ void inkling_attn_prep_kernel(
        float *q_out, float *rel_out, const float *q, const float *r,
        const uint32_t *positions, const uint16_t *proj, uint64_t head_rows, uint32_t extent) {
    for (uint64_t head = blockIdx.x; head < head_rows; head += gridDim.x) {
        float tau = 1.0f;
        if (extent == INKLING_GLOBAL_EXTENT) {
            const float n = (float)((uint64_t)positions[head / INKLING_HEADS] + 1);
            const float ratio = fmaxf(__fdiv_rn(n, (float)INKLING_TAU_FLOOR), 1.0f);
            tau = __fadd_rn(1.0f, __fmul_rn(INKLING_TAU_ALPHA, logf(ratio)));
        }
        if (threadIdx.x < INKLING_HEAD_DIM) {
            const uint64_t i = head * INKLING_HEAD_DIM + threadIdx.x;
            q_out[i] = inkling_bf16(__fmul_rn(inkling_bf16(q[i]), tau));
        }
        for (uint32_t e = threadIdx.x; e < extent; e += blockDim.x) {
            float sum = 0.0f;
            #pragma unroll
            for (int d = 0; d < INKLING_REL_DIM; d++) {
                const float w = __uint_as_float((uint32_t)proj[d * extent + e] << 16);
                sum = fmaf(inkling_bf16(r[head * INKLING_REL_DIM + d]), w, sum);
            }
            // Preserve the published post-projection tau contract. Folding tau
            // into R moves a BF16 rounding boundary and changes the model math.
            rel_out[head * extent + e] = inkling_bf16(__fmul_rn(inkling_bf16(sum), tau));
        }
    }
}

extern "C" int ds4_gpu_inkling_attn_prep(
        ds4_gpu_tensor *q_out, ds4_gpu_tensor *rel_out,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *r, const ds4_gpu_tensor *positions,
        const void *model_map, uint64_t model_size, uint64_t proj_offset,
        uint32_t rows, uint32_t extent) {
    if (!q_out || !rel_out || !q || !r || !positions || !model_map || !rows ||
        (extent != INKLING_LOCAL_EXTENT && extent != INKLING_GLOBAL_EXTENT) ||
        proj_offset % sizeof(uint16_t) || proj_offset > model_size) { return 0; }
    const uint64_t weight_bytes = (uint64_t)INKLING_REL_DIM * extent * sizeof(uint16_t);
    const uint64_t head_rows = (uint64_t)rows * INKLING_HEADS;
    const uint64_t qbytes = head_rows * INKLING_HEAD_DIM * sizeof(float);
    const uint64_t rbytes = head_rows * INKLING_REL_DIM * sizeof(float);
    const uint64_t obytes = head_rows * extent * sizeof(float);
    const uint64_t pbytes = (uint64_t)rows * sizeof(uint32_t);
    if (weight_bytes > model_size - proj_offset || q_out->bytes < qbytes || rel_out->bytes < obytes ||
        q->bytes < qbytes || r->bytes < rbytes || positions->bytes < pbytes ||
        inkling_overlap(q_out, qbytes, rel_out, obytes) ||
        (q_out->ptr != q->ptr && inkling_overlap(q_out, qbytes, q, qbytes)) ||
        inkling_overlap(q_out, qbytes, r, rbytes) || inkling_overlap(q_out, qbytes, positions, pbytes) ||
        inkling_overlap(rel_out, obytes, q, qbytes) || inkling_overlap(rel_out, obytes, r, rbytes) ||
        inkling_overlap(rel_out, obytes, positions, pbytes)) { return 0; }
    const uint16_t *proj = (const uint16_t *)cuda_resolve_weight_ptr(
        model_map, proj_offset, weight_bytes, 0, "inkling relative projection");
    if (!proj) { return 0; }
    const unsigned blocks = (unsigned)(head_rows < INKLING_MAX_BLOCKS ? head_rows : INKLING_MAX_BLOCKS);
    inkling_attn_prep_kernel<<<blocks, INKLING_THREADS, 0, ds4_current_stream()>>>(
        (float *)q_out->ptr, (float *)rel_out->ptr, (const float *)q->ptr, (const float *)r->ptr,
        (const uint32_t *)positions->ptr, proj, head_rows, extent);
    return cuda_ok(cudaGetLastError(), "Inkling attention preparation launch");
}

static __device__ __forceinline__ float inkling_kv_value(
        const uint16_t *cache, const float *current, uint64_t key, uint32_t base,
        uint32_t capacity, uint32_t channel, uint32_t cache_offset) {
    if (key >= base) {
        return inkling_bf16(current[(key - base) * INKLING_KV_WIDTH + channel]);
    }
    const uint64_t index = (key % capacity) * INKLING_KV_ROW + cache_offset + channel;
    return __uint_as_float((uint32_t)cache[index] << 16);
}

static __global__ void inkling_attention_kernel(
        float *out, const float *q, const float *relative, const float *k, const float *v,
        const uint16_t *cache, const uint32_t *position, uint32_t rows, uint32_t capacity,
        uint32_t extent) {
    const unsigned warp = threadIdx.x / INKLING_WARP, lane = threadIdx.x % INKLING_WARP;
    __shared__ float maxima[INKLING_ATTN_WARPS], sums[INKLING_ATTN_WARPS];
    __shared__ float partial[INKLING_ATTN_WARPS * INKLING_HEAD_DIM];
    const uint32_t base = *position;
    const uint64_t end = (uint64_t)base + rows;
    const bool valid = end - 1 <= UINT32_MAX && (extent == INKLING_LOCAL_EXTENT || end <= capacity);
    const uint64_t head_rows = (uint64_t)rows * INKLING_HEADS;
    for (uint64_t head_row = blockIdx.x; head_row < head_rows; head_row += gridDim.x) {
        if (!valid) {
            if (threadIdx.x < INKLING_HEAD_DIM) { out[head_row * INKLING_HEAD_DIM + threadIdx.x] = NAN; }
            continue;
        }
        const uint64_t query = base + head_row / INKLING_HEADS;
        const unsigned kv_head = (head_row % INKLING_HEADS) / (INKLING_HEADS / INKLING_KV_HEADS);
        const uint64_t first = extent == INKLING_LOCAL_EXTENT && query + 1 > INKLING_LOCAL_EXTENT
            ? query + 1 - INKLING_LOCAL_EXTENT : 0;
        float qv[INKLING_ATTN_DPL], acc[INKLING_ATTN_DPL] = {0.0f};
        #pragma unroll
        for (int d = 0; d < INKLING_ATTN_DPL; d++) {
            qv[d] = inkling_bf16(q[head_row * INKLING_HEAD_DIM + lane + d * INKLING_WARP]);
        }
        float maximum = -INFINITY, sum = 0.0f;
        // Independent warp scans avoid a block barrier for every key. Each
        // warp carries an FP32 online-softmax state; merge once after the scan.
        for (uint64_t key = first + warp; key <= query; key += INKLING_ATTN_WARPS) {
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < INKLING_ATTN_DPL; d++) {
                const unsigned c = kv_head * INKLING_HEAD_DIM + lane + d * INKLING_WARP;
                dot = fmaf(qv[d], inkling_kv_value(cache, k, key, base, capacity, c, 0), dot);
            }
            #pragma unroll
            for (int delta = INKLING_WARP / 2; delta > 0; delta /= 2) {
                dot = __fadd_rn(dot, __shfl_xor_sync(UINT32_MAX, dot, delta));
            }
            const uint64_t distance = query - key;
            const float bias = distance < extent ? inkling_bf16(relative[head_row * extent + distance]) : 0.0f;
            const float score = __fadd_rn(dot / (float)INKLING_HEAD_DIM, bias);
            const float next_max = fmaxf(maximum, score);
            const float alpha = __expf(maximum - next_max), beta = __expf(score - next_max);
            sum = fmaf(sum, alpha, beta);
            maximum = next_max;
            #pragma unroll
            for (int d = 0; d < INKLING_ATTN_DPL; d++) {
                const unsigned c = kv_head * INKLING_HEAD_DIM + lane + d * INKLING_WARP;
                const float value = inkling_kv_value(cache, v, key, base, capacity, c, INKLING_KV_WIDTH);
                acc[d] = fmaf(beta, value, __fmul_rn(acc[d], alpha));
            }
        }
        if (lane == 0) { maxima[warp] = maximum; sums[warp] = sum; }
        #pragma unroll
        for (int d = 0; d < INKLING_ATTN_DPL; d++) {
            partial[warp * INKLING_HEAD_DIM + lane + d * INKLING_WARP] = acc[d];
        }
        __syncthreads();
        float max_all = -INFINITY;
        #pragma unroll
        for (int w = 0; w < INKLING_ATTN_WARPS; w++) { max_all = fmaxf(max_all, maxima[w]); }
        float total = 0.0f, value = 0.0f;
        #pragma unroll
        for (int w = 0; w < INKLING_ATTN_WARPS; w++) {
            const float scale = __expf(maxima[w] - max_all);
            total = fmaf(sums[w], scale, total);
            value = fmaf(partial[w * INKLING_HEAD_DIM + threadIdx.x], scale, value);
        }
        out[head_row * INKLING_HEAD_DIM + threadIdx.x] = inkling_bf16(__fdiv_rn(value, total));
        __syncthreads();
    }
}


/* Prefill attention grouped by KV head. One CTA owns one (query, KV head)
 * pair; its four warps keep the release key phases, but each warp scores the
 * four query heads sharing that KV head, so every K/V element is read once
 * per four heads. Keys use 32-bit arithmetic, the ring slot advances without
 * a modulo, and the next key's K/V/bias are prefetched while the current key
 * is scored. Per (head, key) the FMA chain, XOR tree, score, online-softmax
 * update and four-warp merge are the release sequence, so outputs are
 * byte-identical. The block bound keeps six CTAs per SM without spills.
 *
 *   release CTA: head row h,  warp w -> keys first+w, first+w+4, ...
 *   this CTA:    (query, kv), warp w -> same keys, heads 4kv..4kv+3 together
 *
 * TRANSPOSED reduces the four head dots with a transposed butterfly: after
 * the offset-16 and offset-8 steps each 8-lane group carries one head, so
 * steps 4/2/1, the score and the online-softmax scalars run once per head
 * instead of in all 32 lanes, and alpha/beta are broadcast for the V update.
 * Every head's XOR tree pairs the same lanes in the same order, so the sums
 * are the release values.
 *
 *   step 16: lanes 0-15 keep heads 0,1 (lanes 16-31: heads 2,3)
 *   step 8:  lanes 0-7 keep head 0, 8-15 head 1, 16-23 head 2, 24-31 head 3
 */
template<bool TRANSPOSED>
__launch_bounds__(INKLING_WARP * INKLING_ATTN_WARPS, INKLING_ATTN_GROUP_MIN_BLOCKS)
static __global__ void inkling_attention_group_kernel(
        float *out, const float *q, const float *relative, const float *k, const float *v,
        const uint16_t *cache, const uint32_t *position, uint32_t rows, uint32_t capacity,
        uint32_t extent) {
    const unsigned warp = threadIdx.x / INKLING_WARP, lane = threadIdx.x % INKLING_WARP;
    __shared__ float maxima[INKLING_ATTN_GROUP][INKLING_ATTN_WARPS];
    __shared__ float sums[INKLING_ATTN_GROUP][INKLING_ATTN_WARPS];
    __shared__ float partial[INKLING_ATTN_GROUP][INKLING_ATTN_WARPS * INKLING_HEAD_DIM];
    const uint32_t base = *position;
    const uint64_t end = (uint64_t)base + rows;
    const bool valid = end - 1 <= UINT32_MAX && (extent == INKLING_LOCAL_EXTENT || end <= capacity);
    const uint64_t groups = (uint64_t)rows * INKLING_KV_HEADS;
    const unsigned head_lane = lane / INKLING_ATTN_HEAD_LANES;
    for (uint64_t group = blockIdx.x; group < groups; group += gridDim.x) {
        const unsigned kv_head = group % INKLING_KV_HEADS;
        const uint64_t head_row0 = (group / INKLING_KV_HEADS) * INKLING_HEADS +
            kv_head * INKLING_ATTN_GROUP;
        if (!valid) {
            if (threadIdx.x < INKLING_HEAD_DIM) {
                #pragma unroll
                for (int h = 0; h < INKLING_ATTN_GROUP; h++) {
                    out[(head_row0 + h) * INKLING_HEAD_DIM + threadIdx.x] = NAN;
                }
            }
            continue;
        }
        // valid only bounds the last query, so query and first stay 64-bit.
        // Every per-key quantity below is a 32-bit distance from first:
        // query - first <= query fits, and the step guard never wraps.
        const uint64_t query = base + group / INKLING_KV_HEADS;
        const uint64_t first = extent == INKLING_LOCAL_EXTENT && query + 1 > INKLING_LOCAL_EXTENT
            ? query + 1 - INKLING_LOCAL_EXTENT : 0;
        const uint32_t span = (uint32_t)(query - first);
        // Index i is in the current rows once first + i >= base; the row is
        // then i - cur0 + lead (only one of the two is nonzero, both < rows).
        const uint32_t cur0 = base > first ? (uint32_t)(base - first) : 0;
        const uint32_t lead = first > base ? (uint32_t)(first - base) : 0;
        const unsigned channel = kv_head * INKLING_HEAD_DIM + lane;
        const float *rel[INKLING_ATTN_GROUP];
        float qv[INKLING_ATTN_GROUP][INKLING_ATTN_DPL];
        float acc[INKLING_ATTN_GROUP][INKLING_ATTN_DPL] = {};
        float maximum[INKLING_ATTN_GROUP], sum[INKLING_ATTN_GROUP];
        #pragma unroll
        for (int h = 0; h < INKLING_ATTN_GROUP; h++) {
            maximum[h] = -INFINITY;
            sum[h] = 0.0f;
            rel[h] = relative + (head_row0 + h) * extent;
            #pragma unroll
            for (int d = 0; d < INKLING_ATTN_DPL; d++) {
                qv[h][d] = inkling_bf16(q[(head_row0 + h) * INKLING_HEAD_DIM + lane + d * INKLING_WARP]);
            }
        }
        // Fetch key first+i: lane channels of K and V plus the four head biases.
        auto fetch = [&](uint32_t i, uint32_t slot, float *kk, float *vv, float *bb) {
            if (i >= cur0) {
                const uint32_t at = (i - cur0 + lead) * INKLING_KV_WIDTH + channel;
                #pragma unroll
                for (int d = 0; d < INKLING_ATTN_DPL; d++) {
                    kk[d] = inkling_bf16(k[at + d * INKLING_WARP]);
                    vv[d] = inkling_bf16(v[at + d * INKLING_WARP]);
                }
            } else {
                const uint16_t *row = cache + (uint64_t)slot * INKLING_KV_ROW + channel;
                #pragma unroll
                for (int d = 0; d < INKLING_ATTN_DPL; d++) {
                    kk[d] = __uint_as_float((uint32_t)row[d * INKLING_WARP] << 16);
                    vv[d] = __uint_as_float((uint32_t)row[INKLING_KV_WIDTH + d * INKLING_WARP] << 16);
                }
            }
            const uint32_t distance = span - i;
            if constexpr (TRANSPOSED) {
                // Only the bias of the head this lane group scores.
                bb[0] = distance < extent ? inkling_bf16(rel[head_lane][distance]) : 0.0f;
            } else {
                #pragma unroll
                for (int h = 0; h < INKLING_ATTN_GROUP; h++) {
                    bb[h] = distance < extent ? inkling_bf16(rel[h][distance]) : 0.0f;
                }
            }
        };
        uint32_t i = warp;
        uint32_t slot = (uint32_t)((first + warp) % capacity);
        float kk[INKLING_ATTN_DPL] = {}, vv[INKLING_ATTN_DPL] = {}, bb[INKLING_ATTN_GROUP] = {};
        if (i <= span) { fetch(i, slot, kk, vv, bb); }
        while (i <= span) {
            // Capacity is at least the warp count, so one wrap suffices.
            uint32_t next_slot = slot + INKLING_ATTN_WARPS;
            if (next_slot >= capacity) { next_slot -= capacity; }
            const bool more = span - i >= INKLING_ATTN_WARPS;
            float kn[INKLING_ATTN_DPL] = {}, vn[INKLING_ATTN_DPL] = {}, bn[INKLING_ATTN_GROUP] = {};
            if (more) { fetch(i + INKLING_ATTN_WARPS, next_slot, kn, vn, bn); }
            if constexpr (TRANSPOSED) {
                float dot[INKLING_ATTN_GROUP];
                #pragma unroll
                for (int h = 0; h < INKLING_ATTN_GROUP; h++) {
                    dot[h] = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < INKLING_ATTN_DPL; d++) { dot[h] = fmaf(qv[h][d], kk[d], dot[h]); }
                }
                // Each lane sends the values its partner keeps and adds the
                // partner's copy of its own, the release pairing per head.
                const bool upper = lane & (INKLING_WARP / 2);
                const float r0 = __shfl_xor_sync(UINT32_MAX, upper ? dot[0] : dot[2], INKLING_WARP / 2);
                const float r1 = __shfl_xor_sync(UINT32_MAX, upper ? dot[1] : dot[3], INKLING_WARP / 2);
                const float a0 = __fadd_rn(upper ? dot[2] : dot[0], r0);
                const float a1 = __fadd_rn(upper ? dot[3] : dot[1], r1);
                const bool odd = lane & INKLING_ATTN_HEAD_LANES;
                const float r2 = __shfl_xor_sync(UINT32_MAX, odd ? a0 : a1, INKLING_ATTN_HEAD_LANES);
                float red = __fadd_rn(odd ? a1 : a0, r2);
                #pragma unroll
                for (int delta = INKLING_ATTN_HEAD_LANES / 2; delta > 0; delta /= 2) {
                    red = __fadd_rn(red, __shfl_xor_sync(UINT32_MAX, red, delta));
                }
                const float score = __fadd_rn(red / (float)INKLING_HEAD_DIM, bb[0]);
                const float next_max = fmaxf(maximum[0], score);
                const float alpha = __expf(maximum[0] - next_max), beta = __expf(score - next_max);
                sum[0] = fmaf(sum[0], alpha, beta);
                maximum[0] = next_max;
                #pragma unroll
                for (int h = 0; h < INKLING_ATTN_GROUP; h++) {
                    const float ah = __shfl_sync(UINT32_MAX, alpha, h * INKLING_ATTN_HEAD_LANES);
                    const float bh = __shfl_sync(UINT32_MAX, beta, h * INKLING_ATTN_HEAD_LANES);
                    #pragma unroll
                    for (int d = 0; d < INKLING_ATTN_DPL; d++) {
                        acc[h][d] = fmaf(bh, vv[d], __fmul_rn(acc[h][d], ah));
                    }
                }
            } else {
            #pragma unroll
            for (int h = 0; h < INKLING_ATTN_GROUP; h++) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < INKLING_ATTN_DPL; d++) { dot = fmaf(qv[h][d], kk[d], dot); }
                #pragma unroll
                for (int delta = INKLING_WARP / 2; delta > 0; delta /= 2) {
                    dot = __fadd_rn(dot, __shfl_xor_sync(UINT32_MAX, dot, delta));
                }
                const float score = __fadd_rn(dot / (float)INKLING_HEAD_DIM, bb[h]);
                const float next_max = fmaxf(maximum[h], score);
                const float alpha = __expf(maximum[h] - next_max), beta = __expf(score - next_max);
                sum[h] = fmaf(sum[h], alpha, beta);
                maximum[h] = next_max;
                #pragma unroll
                for (int d = 0; d < INKLING_ATTN_DPL; d++) {
                    acc[h][d] = fmaf(beta, vv[d], __fmul_rn(acc[h][d], alpha));
                }
            }
            }
            #pragma unroll
            for (int d = 0; d < INKLING_ATTN_DPL; d++) { kk[d] = kn[d]; vv[d] = vn[d]; }
            #pragma unroll
            for (int h = 0; h < INKLING_ATTN_GROUP; h++) { bb[h] = bn[h]; }
            slot = next_slot;
            if (!more) { break; }
            i += INKLING_ATTN_WARPS;
        }
        if constexpr (TRANSPOSED) {
            if (lane % INKLING_ATTN_HEAD_LANES == 0) {
                maxima[head_lane][warp] = maximum[0];
                sums[head_lane][warp] = sum[0];
            }
        }
        #pragma unroll
        for (int h = 0; h < INKLING_ATTN_GROUP; h++) {
            if (!TRANSPOSED && lane == 0) { maxima[h][warp] = maximum[h]; sums[h][warp] = sum[h]; }
            #pragma unroll
            for (int d = 0; d < INKLING_ATTN_DPL; d++) {
                partial[h][warp * INKLING_HEAD_DIM + lane + d * INKLING_WARP] = acc[h][d];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int h = 0; h < INKLING_ATTN_GROUP; h++) {
            float max_all = -INFINITY;
            #pragma unroll
            for (int w = 0; w < INKLING_ATTN_WARPS; w++) { max_all = fmaxf(max_all, maxima[h][w]); }
            float total = 0.0f, value = 0.0f;
            #pragma unroll
            for (int w = 0; w < INKLING_ATTN_WARPS; w++) {
                const float scale = __expf(maxima[h][w] - max_all);
                total = fmaf(sums[h][w], scale, total);
                value = fmaf(partial[h][w * INKLING_HEAD_DIM + threadIdx.x], scale, value);
            }
            out[(head_row0 + h) * INKLING_HEAD_DIM + threadIdx.x] = inkling_bf16(__fdiv_rn(value, total));
        }
        __syncthreads();
    }
}

extern "C" int ds4_gpu_inkling_attention(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *relative,
        const ds4_gpu_tensor *k, const ds4_gpu_tensor *v, const ds4_gpu_tensor *cache,
        const ds4_gpu_tensor *position, uint32_t rows, uint32_t capacity, uint32_t extent) {
    if (!out || !q || !relative || !k || !v || !cache || !position || !rows || !capacity ||
        (extent != INKLING_LOCAL_EXTENT && extent != INKLING_GLOBAL_EXTENT) ||
        (extent == INKLING_LOCAL_EXTENT && capacity < INKLING_LOCAL_EXTENT)) { return 0; }
    const uint64_t head_rows = (uint64_t)rows * INKLING_HEADS;
    const uint64_t qbytes = head_rows * INKLING_HEAD_DIM * sizeof(float);
    const uint64_t rbytes = head_rows * extent * sizeof(float);
    const uint64_t kvbytes = (uint64_t)rows * INKLING_KV_WIDTH * sizeof(float);
    const uint64_t cbytes = (uint64_t)capacity * INKLING_KV_ROW * sizeof(uint16_t);
    const ds4_gpu_tensor *inputs[] = {q, relative, k, v, cache, position};
    const uint64_t sizes[] = {qbytes, rbytes, kvbytes, kvbytes, cbytes, sizeof(uint32_t)};
    if (out->bytes < qbytes) { return 0; }
    for (unsigned i = 0; i < sizeof(inputs) / sizeof(inputs[0]); i++) {
        if (inputs[i]->bytes < sizes[i] || inkling_overlap(out, qbytes, inputs[i], sizes[i])) { return 0; }
    }
    // Prefill widths share K/V across each KV head's query heads; decode and
    // MTP verify widths keep the per-head kernel.
    if (rows >= INKLING_ATTN_GROUP_MIN_ROWS && !getenv("DS4_INKLING_NO_ATTN_GROUP")) {
        const uint64_t groups = head_rows / INKLING_ATTN_GROUP;
        const unsigned blocks = (unsigned)(groups < INKLING_MAX_BLOCKS ? groups : INKLING_MAX_BLOCKS);
        // The switch restores the all-lane reduction for A/B controls.
        #define IK_ATTN_GROUP(T) inkling_attention_group_kernel<T> \
            <<<blocks, INKLING_WARP * INKLING_ATTN_WARPS, 0, ds4_current_stream()>>>( \
            (float *)out->ptr, (const float *)q->ptr, (const float *)relative->ptr, \
            (const float *)k->ptr, (const float *)v->ptr, (const uint16_t *)cache->ptr, \
            (const uint32_t *)position->ptr, rows, capacity, extent)
        if (getenv("DS4_INKLING_NO_ATTN_TRANSPOSE")) { IK_ATTN_GROUP(false); } else { IK_ATTN_GROUP(true); }
        #undef IK_ATTN_GROUP
        return cuda_ok(cudaGetLastError(), "Inkling grouped attention launch");
    }
    const unsigned blocks = (unsigned)(head_rows < INKLING_MAX_BLOCKS ? head_rows : INKLING_MAX_BLOCKS);
    inkling_attention_kernel<<<blocks, INKLING_WARP * INKLING_ATTN_WARPS, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)q->ptr, (const float *)relative->ptr,
        (const float *)k->ptr, (const float *)v->ptr, (const uint16_t *)cache->ptr,
        (const uint32_t *)position->ptr, rows, capacity, extent);
    return cuda_ok(cudaGetLastError(), "Inkling attention launch");
}

static __global__ void inkling_kv_store_kernel(
        uint16_t *cache, const float *k, const float *v, const uint32_t *position,
        uint32_t rows, uint32_t capacity, uint32_t start, uint64_t count) {
    const uint32_t base = *position;
    if ((uint64_t)base + rows - 1 > UINT32_MAX) { return; }
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t row = start + i / INKLING_KV_WIDTH;
        const unsigned c = i % INKLING_KV_WIDTH;
        const uint64_t src = row * INKLING_KV_WIDTH + c;
        const uint64_t dst = ((base + row) % capacity) * INKLING_KV_ROW + c;
        cache[dst] = __bfloat16_as_ushort(__float2bfloat16_rn(k[src]));
        cache[dst + INKLING_KV_WIDTH] = __bfloat16_as_ushort(__float2bfloat16_rn(v[src]));
    }
}

extern "C" int ds4_gpu_inkling_kv_store(
        ds4_gpu_tensor *cache, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *position, uint32_t rows, uint32_t capacity) {
    if (!cache || !k || !v || !position || !capacity) { return 0; }
    const uint64_t kvbytes = (uint64_t)rows * INKLING_KV_WIDTH * sizeof(float);
    const uint64_t cbytes = (uint64_t)capacity * INKLING_KV_ROW * sizeof(uint16_t);
    if (cache->bytes < cbytes || k->bytes < kvbytes || v->bytes < kvbytes || position->bytes < sizeof(uint32_t) ||
        inkling_overlap(cache, cbytes, k, kvbytes) || inkling_overlap(cache, cbytes, v, kvbytes) ||
        inkling_overlap(cache, cbytes, position, sizeof(uint32_t))) { return 0; }
    if (!rows) { return 1; }
    const uint32_t start = rows > capacity ? rows - capacity : 0;
    const uint64_t count = (uint64_t)(rows - start) * INKLING_KV_WIDTH;
    const uint64_t grid = (count + INKLING_THREADS - 1) / INKLING_THREADS;
    const unsigned blocks = (unsigned)(grid < INKLING_MAX_BLOCKS ? grid : INKLING_MAX_BLOCKS);
    inkling_kv_store_kernel<<<blocks, INKLING_THREADS, 0, ds4_current_stream()>>>(
        (uint16_t *)cache->ptr, (const float *)k->ptr, (const float *)v->ptr,
        (const uint32_t *)position->ptr, rows, capacity, start, count);
    return cuda_ok(cudaGetLastError(), "Inkling KV store launch");
}

static __global__ void inkling_norm_kernel(
        float *out, const float *x, const uint16_t *weight, uint32_t width, uint32_t rows) {
    __shared__ float partial[INKLING_THREADS];
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const float *xr = x + row * width;
        float sum = 0.0f;
        for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
            const float value = inkling_bf16(xr[i]);
            sum = __fadd_rn(sum, __fmul_rn(value, value));
        }
        partial[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned stride = INKLING_THREADS / 2; stride; stride /= 2) {
            if (threadIdx.x < stride) { partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + stride]); }
            __syncthreads();
        }
        const float variance = __fadd_rn(__fdiv_rn(partial[0], (float)width), INKLING_RMS_EPS);
        const float scale = rsqrtf(variance);
        for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
            const float w = __uint_as_float((uint32_t)weight[i] << 16);
            // The CUDA source applies the BF16 weight in FP32, before its
            // only output cast; do not round the normalized value first.
            out[row * width + i] = inkling_bf16(__fmul_rn(__fmul_rn(inkling_bf16(xr[i]), scale), w));
        }
        __syncthreads();
    }
}

extern "C" int ds4_gpu_inkling_norm(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map,
        uint64_t model_size, uint64_t weight_offset, uint32_t width, uint32_t rows) {
    if (!out || !x || !model_map || !width || !rows || width > INKLING_NORM_MAX ||
        weight_offset % sizeof(uint16_t) || weight_offset > model_size ||
        (uint64_t)width * sizeof(uint16_t) > model_size - weight_offset) { return 0; }
    const uint64_t bytes = (uint64_t)rows * width * sizeof(float);
    if (out->bytes < bytes || x->bytes < bytes ||
        (out->ptr != x->ptr && inkling_overlap(out, bytes, x, bytes))) { return 0; }
    const uint16_t *weight = (const uint16_t *)cuda_resolve_weight_ptr(
        model_map, weight_offset, (uint64_t)width * sizeof(uint16_t), 0, "inkling RMS weight");
    if (!weight) { return 0; }
    cuda_norm_q8_invalidate(out->ptr);
    const unsigned blocks = rows < INKLING_MAX_BLOCKS ? rows : INKLING_MAX_BLOCKS;
    inkling_norm_kernel<<<blocks, INKLING_THREADS, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)x->ptr, weight, width, rows);
    return cuda_ok(cudaGetLastError(), "Inkling RMSNorm launch");
}

static __global__ void inkling_add_scale_kernel(
        float *out, const float *a, const float *b, float scale, uint64_t count) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (uint64_t)gridDim.x * blockDim.x) {
        float value = __fmul_rn(inkling_bf16(a[i]), scale);
        if (b) { value = __fadd_rn(value, inkling_bf16(b[i])); }
        out[i] = inkling_bf16(value);
    }
}

extern "C" int ds4_gpu_inkling_add_scale(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
        float scale, uint64_t count) {
    if (!out || !a || !count || count > UINT64_MAX / sizeof(float) || !isfinite(scale)) { return 0; }
    const uint64_t bytes = count * sizeof(float);
    if (out->bytes < bytes || a->bytes < bytes ||
        (out->ptr != a->ptr && inkling_overlap(out, bytes, a, bytes)) ||
        (b && (b->bytes < bytes || (out->ptr != b->ptr && inkling_overlap(out, bytes, b, bytes))))) { return 0; }
    cuda_norm_q8_invalidate(out->ptr);
    const uint64_t grid = (count + INKLING_THREADS - 1) / INKLING_THREADS;
    const unsigned blocks = (unsigned)(grid < INKLING_MAX_BLOCKS ? grid : INKLING_MAX_BLOCKS);
    inkling_add_scale_kernel<<<blocks, INKLING_THREADS, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)a->ptr, b ? (const float *)b->ptr : NULL, scale, count);
    return cuda_ok(cudaGetLastError(), "Inkling BF16 scale/residual launch");
}

struct inkling_fold_shape {
    uint32_t time, spatial, channels, time_fold, spatial_fold;
};
static constexpr inkling_fold_shape inkling_fold_shapes[] = {
    {2, 40, 3, 1, 5}, {2, 8, 128, 1, 2},
    {2, 4, 320, 1, 4}, {2, 1, 4800, 2, 1},
};

static __global__ void inkling_fold_kernel(
        float *out, const float *x, inkling_fold_shape s, uint64_t count) {
    const uint32_t new_t = s.time / s.time_fold;
    const uint32_t new_hw = s.spatial / s.spatial_fold;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (uint64_t)gridDim.x * blockDim.x) {
        // Invert source reshape/permute: B,T',H',W',tf,hf,wf,C.
        uint64_t at = i;
        const uint32_t c = at % s.channels; at /= s.channels;
        const uint32_t fw = at % s.spatial_fold; at /= s.spatial_fold;
        const uint32_t fh = at % s.spatial_fold; at /= s.spatial_fold;
        const uint32_t ft = at % s.time_fold; at /= s.time_fold;
        const uint32_t nw = at % new_hw; at /= new_hw;
        const uint32_t nh = at % new_hw; at /= new_hw;
        const uint32_t nt = at % new_t; at /= new_t;
        const uint64_t src = ((((at * s.time + nt * s.time_fold + ft) * s.spatial +
            nh * s.spatial_fold + fh) * s.spatial + nw * s.spatial_fold + fw) * s.channels + c);
        out[i] = inkling_bf16(x[src]);
    }
}

extern "C" int ds4_gpu_inkling_fold(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t patches, uint32_t stage) {
    if (!out || !x || !patches || stage >= sizeof(inkling_fold_shapes) / sizeof(inkling_fold_shapes[0])) {
        return 0;
    }
    const inkling_fold_shape shape = inkling_fold_shapes[stage];
    const uint64_t count = (uint64_t)patches * shape.time * shape.spatial * shape.spatial * shape.channels;
    const uint64_t bytes = count * sizeof(float);
    if (out->bytes < bytes || x->bytes < bytes || inkling_overlap(out, bytes, x, bytes)) {
        return 0;
    }
    cuda_norm_q8_invalidate(out->ptr);
    const uint64_t grid = (count + INKLING_THREADS - 1) / INKLING_THREADS;
    const unsigned blocks = (unsigned)(grid < INKLING_MAX_BLOCKS ? grid : INKLING_MAX_BLOCKS);
    inkling_fold_kernel<<<blocks, INKLING_THREADS, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)x->ptr, shape, count);
    return cuda_ok(cudaGetLastError(), "Inkling HMLP fold launch");
}

static __global__ void inkling_gelu_kernel(float *out, const float *x, uint64_t count) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (uint64_t)gridDim.x * blockDim.x) {
        const float value = inkling_bf16(x[i]);
        const float cdf = __fadd_rn(1.0f, erff(__fmul_rn(value, INKLING_GELU_SCALE)));
        out[i] = inkling_bf16(__fmul_rn(__fmul_rn(0.5f, value), cdf));
    }
}

extern "C" int ds4_gpu_inkling_gelu(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint64_t count) {
    if (!out || !x || !count || count > UINT64_MAX / sizeof(float)) { return 0; }
    const uint64_t bytes = count * sizeof(float);
    if (out->bytes < bytes || x->bytes < bytes ||
        (out->ptr != x->ptr && inkling_overlap(out, bytes, x, bytes))) { return 0; }
    cuda_norm_q8_invalidate(out->ptr);
    const uint64_t grid = (count + INKLING_THREADS - 1) / INKLING_THREADS;
    const unsigned blocks = (unsigned)(grid < INKLING_MAX_BLOCKS ? grid : INKLING_MAX_BLOCKS);
    inkling_gelu_kernel<<<blocks, INKLING_THREADS, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)x->ptr, count);
    return cuda_ok(cudaGetLastError(), "Inkling GELU launch");
}

static __global__ void inkling_audio_kernel(
        float *out, const int32_t *ids, const uint16_t *weight, uint64_t count) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (uint64_t)gridDim.x * blockDim.x) {
        const uint32_t col = i % INKLING_MEDIA_WIDTH;
        const uint64_t row = i / INKLING_MEDIA_WIDTH;
        float sum = 0.0f;
        for (unsigned bin = 0; bin < INKLING_AUDIO_BINS; bin++) {
            const int32_t code = ids[row * INKLING_AUDIO_BINS + bin];
            if (code < 0 || code >= INKLING_AUDIO_LEVELS) {
                sum = NAN;
                break;
            }
            const uint64_t index = (bin * INKLING_AUDIO_LEVELS + (unsigned)code) *
                                   (uint64_t)INKLING_MEDIA_WIDTH + col;
            sum = __fadd_rn(sum, __uint_as_float((uint32_t)weight[index] << 16));
        }
        out[i] = inkling_bf16(sum);
    }
}

extern "C" int ds4_gpu_inkling_audio(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *ids, const void *model_map,
        uint64_t model_size, uint64_t weight_offset, uint32_t rows) {
    const uint64_t weight_bytes = (uint64_t)INKLING_AUDIO_BINS * INKLING_AUDIO_LEVELS *
                                  INKLING_MEDIA_WIDTH * sizeof(uint16_t);
    if (!out || !ids || !model_map || !rows || weight_offset % sizeof(uint16_t) ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) { return 0; }
    const uint64_t count = (uint64_t)rows * INKLING_MEDIA_WIDTH;
    const uint64_t bytes = count * sizeof(float), ibytes = (uint64_t)rows * INKLING_AUDIO_BINS * sizeof(int32_t);
    if (out->bytes < bytes || ids->bytes < ibytes || inkling_overlap(out, bytes, ids, ibytes)) { return 0; }
    const uint16_t *weight = (const uint16_t *)cuda_resolve_weight_ptr(
        model_map, weight_offset, weight_bytes, 0, "inkling audio embeddings");
    if (!weight) { return 0; }
    cuda_norm_q8_invalidate(out->ptr);
    const uint64_t grid = (count + INKLING_THREADS - 1) / INKLING_THREADS;
    const unsigned blocks = (unsigned)(grid < INKLING_MAX_BLOCKS ? grid : INKLING_MAX_BLOCKS);
    inkling_audio_kernel<<<blocks, INKLING_THREADS, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const int32_t *)ids->ptr, weight, count);
    return cuda_ok(cudaGetLastError(), "Inkling audio embedding launch");
}
