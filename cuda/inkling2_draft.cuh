/* Inkling-2 DSpark drafter kernels (TokenSpeed inkling2_mtp.py, c485a46a):
 * target taps, their per-stage mixtures, the draft layers' pre-norm block
 * over BF16 context rings, and the rnn_hidden head. Included by
 * ds4_inkling2_gpu.cuh after its helpers. Norms are PrecisionRMSNorm: F32,
 * the mean square by a multiply with 1 / width, one BF16 rounding. */

enum { INK2D_THREADS = 256, INK2D_PER = INK2_HIDDEN / INK2D_THREADS, INK2D_DTK = 32, INK2D_PARTIAL = 130 };

static_assert(INK2_HIDDEN % INK2D_THREADS == 0, "a CTA covers a hidden row");

__device__ __forceinline__ uint16_t ink2d_bits(float x) { return (uint16_t)(__float_as_uint(glmvq_round(x)) >> 16); }

// Tap `tap` of rows 0..rows-1: the F32 rows hold BF16 values.
__global__ void ink2d_tap_kernel(uint16_t *taps, const float *x, uint32_t tap) {
    const uint32_t row = blockIdx.x;
    for (uint32_t i = threadIdx.x; i < INK2_HIDDEN; i += blockDim.x) {
        taps[((uint64_t)row * DS4_INK2_DRAFT_TAPS + tap) * INK2_HIDDEN + i] =
            (uint16_t)(__float_as_uint(x[(uint64_t)row * INK2_HIDDEN + i]) >> 16);
    }
}

extern "C" int ds4_gpu_ink2_draft_tap(ds4_gpu_tensor *taps, const ds4_gpu_tensor *x, uint32_t tap, uint32_t rows) {
    if (!rows || tap >= DS4_INK2_DRAFT_TAPS || !gvq_floats(x, (uint64_t)rows * INK2_HIDDEN) ||
        taps->bytes < (uint64_t)rows * DS4_INK2_DRAFT_TAPS * INK2_HIDDEN * sizeof(uint16_t)) { return 0; }
    ink2d_tap_kernel<<<rows, INK2D_THREADS, 0, ds4_current_stream()>>>((uint16_t *)taps->ptr, (const float *)x->ptr,
        tap);
    return cuda_ok(cudaGetLastError(), "Inkling-2 draft tap");
}

/* One row's norm of the F32 values v (INK2D_PER per thread): BF16(v r w). */
__device__ __forceinline__ void ink2d_norm_store(float *out, const float (&v)[INK2D_PER], const uint16_t *w,
        float *scratch) {
    float sq[1] = {0.0f};
    #pragma unroll
    for (uint32_t k = 0; k < INK2D_PER; k++) { sq[0] = fmaf(v[k], v[k], sq[0]); }
    ink2_block_sum<1>(sq, 1, scratch);
    const float r = rsqrtf(glmvq_add(glmvq_mul(sq[0], 1.0f / (float)INK2_HIDDEN), 1e-6f));
    #pragma unroll
    for (uint32_t k = 0; k < INK2D_PER; k++) {
        const uint32_t i = threadIdx.x + k * INK2D_THREADS;
        out[i] = glmvq_round(glmvq_mul(glmvq_mul(v[k], r), ink2_bf(w[i])));
    }
}

/* Stage mixture: BF16(sum_j alpha_j tap_j) in tap order, then the stage's
 * dflare_norm. */
__global__ void __launch_bounds__(INK2D_THREADS) ink2d_mix_kernel(float *out, const uint16_t *taps,
        const float *alpha, const uint16_t *w) {
    __shared__ float scratch[INK2D_THREADS / 32];
    const uint32_t row = blockIdx.x;
    float v[INK2D_PER];
    #pragma unroll
    for (uint32_t k = 0; k < INK2D_PER; k++) {
        const uint32_t i = threadIdx.x + k * INK2D_THREADS;
        float s = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < DS4_INK2_DRAFT_TAPS; j++) {
            s = fmaf(alpha[j], ink2_bf(taps[((uint64_t)row * DS4_INK2_DRAFT_TAPS + j) * INK2_HIDDEN + i]), s);
        }
        v[k] = glmvq_round(s);
    }
    ink2d_norm_store(out + (uint64_t)row * INK2_HIDDEN, v, w, scratch);
}

extern "C" int ds4_gpu_ink2_draft_mix(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *taps,
        const ds4_gpu_tensor *alpha, uint32_t stage, uint32_t norm, uint32_t first, uint32_t rows) {
    const uint16_t *w = (const uint16_t *)gvq_weight(ctx, norm, GVQ_TYPE_BF16, INK2_HIDDEN);
    const uint64_t tap_row = (uint64_t)DS4_INK2_DRAFT_TAPS * INK2_HIDDEN;
    if (!w || !rows || stage > DS4_INK2_DRAFT_LAYERS || !gvq_floats(out, (uint64_t)rows * INK2_HIDDEN) ||
        !gvq_floats(alpha, (uint64_t)(DS4_INK2_DRAFT_LAYERS + 1) * DS4_INK2_DRAFT_TAPS) ||
        taps->bytes < ((uint64_t)first + rows) * tap_row * sizeof(uint16_t)) { return 0; }
    ink2d_mix_kernel<<<rows, INK2D_THREADS, 0, ds4_current_stream()>>>((float *)out->ptr,
        (const uint16_t *)taps->ptr + (uint64_t)first * tap_row, (const float *)alpha->ptr + stage * DS4_INK2_DRAFT_TAPS,
        w);
    return cuda_ok(cudaGetLastError(), "Inkling-2 draft mixture");
}

/* norm_out = norm(h + residual) from the F32 sum, residual = BF16(h +
 * residual); residual NULL: norm_out = norm(h), residual_out = h. */
__global__ void __launch_bounds__(INK2D_THREADS) ink2d_add_norm_kernel(float *norm_out, float *residual,
        const float *h, const uint16_t *w) {
    __shared__ float scratch[INK2D_THREADS / 32];
    const uint64_t at = (uint64_t)blockIdx.x * INK2_HIDDEN;
    float v[INK2D_PER];
    #pragma unroll
    for (uint32_t k = 0; k < INK2D_PER; k++) {
        const uint32_t i = threadIdx.x + k * INK2D_THREADS;
        v[k] = residual ? glmvq_add(h[at + i], residual[at + i]) : h[at + i];
    }
    __syncthreads();
    if (residual) {
        #pragma unroll
        for (uint32_t k = 0; k < INK2D_PER; k++) { residual[at + threadIdx.x + k * INK2D_THREADS] = glmvq_round(v[k]); }
    }
    ink2d_norm_store(norm_out + at, v, w, scratch);
}

extern "C" int ds4_gpu_ink2_draft_add_norm(ds4_gvq_ctx *ctx, ds4_gpu_tensor *norm_out, ds4_gpu_tensor *residual,
        const ds4_gpu_tensor *h, uint32_t norm, uint32_t rows) {
    const uint16_t *w = (const uint16_t *)gvq_weight(ctx, norm, GVQ_TYPE_BF16, INK2_HIDDEN);
    const uint64_t count = (uint64_t)rows * INK2_HIDDEN;
    if (!w || !rows || !gvq_floats(norm_out, count) || !gvq_floats(h, count) || (residual && !gvq_floats(residual, count))) {
        return 0;
    }
    ink2d_add_norm_kernel<<<rows, INK2D_THREADS, 0, ds4_current_stream()>>>((float *)norm_out->ptr,
        residual ? (float *)residual->ptr : nullptr, (const float *)h->ptr, w);
    return cuda_ok(cudaGetLastError(), "Inkling-2 draft norm");
}

/* Per (row, head): BF16 RMS norm over 128 values with weight w (rounded
 * before RoPE), then the NeoX rotation at position pos0 + row (theta 1e6,
 * the F32 angle table of TokenSpeed's RotaryEmbedding), rounded once. */
__global__ void __launch_bounds__(INK2A_DIM) ink2d_qk_kernel(float *x, const uint16_t *w, uint32_t heads,
        uint32_t pos0) {
    __shared__ float scratch[INK2A_DIM / 32];
    __shared__ float y[INK2A_DIM];
    const uint32_t row = blockIdx.x / heads, t = threadIdx.x;
    float *v = x + (uint64_t)blockIdx.x * INK2A_DIM;
    const float a = v[t];
    float sq[1] = {glmvq_mul(a, a)};
    ink2_block_sum<1>(sq, 1, scratch);
    y[t] = glmvq_round(glmvq_mul(glmvq_mul(a, rsqrtf(glmvq_add(glmvq_mul(sq[0], 1.0f / (float)INK2A_DIM), 1e-6f))),
        ink2_bf(w[t])));
    __syncthreads();
    const uint32_t i = t % (INK2A_DIM / 2);
    const float inv = __fdiv_rn(1.0f, powf(DS4_INK2_DRAFT_THETA, __fdiv_rn((float)(2u * i), (float)INK2A_DIM)));
    float s, c;
    sincosf(glmvq_mul((float)(pos0 + row), inv), &s, &c);
    const float lo = y[i], hi = y[i + INK2A_DIM / 2];
    v[t] = glmvq_round(t < INK2A_DIM / 2 ? glmvq_add(glmvq_mul(lo, c), -glmvq_mul(hi, s))
                                         : glmvq_add(glmvq_mul(hi, c), glmvq_mul(lo, s)));
}

extern "C" int ds4_gpu_ink2_draft_qk(ds4_gvq_ctx *ctx, ds4_gpu_tensor *x, uint32_t norm, uint32_t heads,
        uint32_t rows, uint32_t pos0) {
    const uint16_t *w = (const uint16_t *)gvq_weight(ctx, norm, GVQ_TYPE_BF16, INK2A_DIM);
    if (!w || !rows || !heads || !gvq_floats(x, (uint64_t)rows * heads * INK2A_DIM)) { return 0; }
    ink2d_qk_kernel<<<rows * heads, INK2A_DIM, 0, ds4_current_stream()>>>((float *)x->ptr, w, heads, pos0);
    return cuda_ok(cudaGetLastError(), "Inkling-2 draft q/k");
}

// Rows pos.. of K and V (BF16 values) into ring slots (pos + r) % ring;
// rows below the last `ring` are displaced by later ones and skipped.
__global__ void ink2d_ring_kernel(uint16_t *ring_k, uint16_t *ring_v, const float *k, const float *v, uint32_t rows,
        uint32_t pos, uint32_t ring) {
    const uint32_t row = blockIdx.x;
    if (rows > ring && row < rows - ring) { return; }
    const uint64_t slot = (uint64_t)((pos + row) % ring) * INK2A_KV;
    for (uint32_t i = threadIdx.x; i < INK2A_KV; i += blockDim.x) {
        ring_k[slot + i] = ink2d_bits(k[(uint64_t)row * INK2A_KV + i]);
        ring_v[slot + i] = ink2d_bits(v[(uint64_t)row * INK2A_KV + i]);
    }
}

extern "C" int ds4_gpu_ink2_draft_ring(ds4_gpu_tensor *ring_k, ds4_gpu_tensor *ring_v, const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v, uint32_t rows, uint32_t pos, uint32_t ring) {
    const uint64_t bytes = (uint64_t)ring * INK2A_KV * sizeof(uint16_t);
    if (!rows || !ring || ring_k->bytes < bytes || ring_v->bytes < bytes || !gvq_floats(k, (uint64_t)rows * INK2A_KV) ||
        !gvq_floats(v, (uint64_t)rows * INK2A_KV)) { return 0; }
    ink2d_ring_kernel<<<rows, 128, 0, ds4_current_stream()>>>((uint16_t *)ring_k->ptr, (uint16_t *)ring_v->ptr,
        (const float *)k->ptr, (const float *)v->ptr, rows, pos, ring);
    return cuda_ok(cudaGetLastError(), "Inkling-2 draft ring");
}

/* Block attention, split over keys: CTA (split, KV head) for every row,
 * warp w = query head 6 kv + w, a lane per key of each 32-key tile staged
 * once for all rows. Row r attends the ring positions [lo_r, end): every
 * block key (non-causal), context keys from max(first, end - ring) and, with
 * a window, from pos0 + r - (window - 1). */
__global__ __launch_bounds__(INK2A_DTHREADS)
void ink2d_attn_kernel(float *partials, const float *q, const uint16_t *ring_k, const uint16_t *ring_v,
        uint32_t ring, uint32_t pos0, uint32_t end, uint32_t first, uint32_t window, uint32_t splits, uint32_t rows) {
    __shared__ __align__(16) uint32_t sk[INK2D_DTK][INK2A_DIM / 2], sv[INK2D_DTK][INK2A_DIM / 2];
    __shared__ float sq[INK2A_SPLIT_ROWS][INK2A_GROUP][INK2A_DIM];
    const uint32_t split = blockIdx.x, kv = blockIdx.y;
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32, h = kv * INK2A_GROUP + warp;
    const uint32_t base = max(first, end > ring ? end - ring : 0u);
    // The common range starts at the first row's bound.
    uint32_t lo = base;
    if (window && pos0 >= window - 1u) { lo = max(lo, pos0 - (window - 1u)); }
    const uint32_t keys = end > lo ? end - lo : 0u;
    const uint32_t span = ((keys + splits - 1) / splits + INK2D_DTK - 1) / INK2D_DTK * INK2D_DTK;
    const uint32_t k0 = lo + split * span, k1 = min(end, k0 + span);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t i = lane; i < INK2A_DIM; i += 32) {
            sq[r][warp][i] = glmvq_round(q[((uint64_t)r * INK2A_HEADS + h) * INK2A_DIM + i]);
        }
    }
    float m[INK2A_SPLIT_ROWS], l[INK2A_SPLIT_ROWS], acc[INK2A_SPLIT_ROWS][4];
    #pragma unroll
    for (uint32_t r = 0; r < INK2A_SPLIT_ROWS; r++) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
        #pragma unroll
        for (int d = 0; d < 4; d++) { acc[r][d] = 0.0f; }
    }
    for (uint32_t t0 = k0; t0 < k1; t0 += INK2D_DTK) {
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < INK2D_DTK * INK2A_DIM / 4; i += INK2A_DTHREADS) {
            const uint32_t j = i / (INK2A_DIM / 4), c = i % (INK2A_DIM / 4);
            const uint32_t p = min(t0 + j, k1 - 1);
            const uint16_t *src = ring_k + (uint64_t)(p % ring) * INK2A_KV + kv * INK2A_DIM;
            const uint16_t *vsrc = ring_v + (uint64_t)(p % ring) * INK2A_KV + kv * INK2A_DIM;
            const uint2 kk = *reinterpret_cast<const uint2 *>(src + 4 * c);
            const uint2 vv = *reinterpret_cast<const uint2 *>(vsrc + 4 * c);
            sk[j][2 * c] = kk.x; sk[j][2 * c + 1] = kk.y;
            sv[j][2 * c] = vv.x; sv[j][2 * c + 1] = vv.y;
        }
        __syncthreads();
        const uint32_t p = t0 + lane;
        #pragma unroll
        for (uint32_t r = 0; r < INK2A_SPLIT_ROWS; r++) {
            if (r >= rows) { break; }
            uint32_t row_lo = base;
            if (window && pos0 + r >= window - 1u) { row_lo = max(row_lo, pos0 + r - (window - 1u)); }
            float score = -INFINITY;
            if (p < k1 && p >= row_lo) {
                float dot = 0.0f;
                #pragma unroll 8
                for (uint32_t i = 0; i < INK2A_DIM / 2; i++) {
                    const uint32_t w = (i + lane) & (INK2A_DIM / 2 - 1);
                    const uint32_t bits = sk[lane][w];
                    dot = fmaf(__uint_as_float(bits << 16), sq[r][warp][2 * w], dot);
                    dot = fmaf(__uint_as_float(bits & 0xffff0000u), sq[r][warp][2 * w + 1], dot);
                }
                score = glmvq_mul(dot, 0.08838834764831845f);
            }
            float tmax = score;
            for (uint32_t o = 16; o; o >>= 1) { tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o)); }
            const float next = fmaxf(m[r], tmax);
            const float scale = m[r] == -INFINITY ? 0.0f : __expf(m[r] - next);
            const float wgt = score == -INFINITY ? 0.0f : __expf(score - next);
            float sum = wgt;
            for (uint32_t o = 16; o; o >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, o); }
            l[r] = l[r] * scale + sum;
            m[r] = next;
            #pragma unroll
            for (int d = 0; d < 4; d++) { acc[r][d] *= scale; }
            for (uint32_t j = 0; j < INK2D_DTK; j++) {
                const float pj = __shfl_sync(0xffffffffu, wgt, j);
                if (pj == 0.0f) { continue; }
                const uint2 vv = *reinterpret_cast<const uint2 *>(&sv[j][2 * lane]);
                acc[r][0] = fmaf(pj, __uint_as_float(vv.x << 16), acc[r][0]);
                acc[r][1] = fmaf(pj, __uint_as_float(vv.x & 0xffff0000u), acc[r][1]);
                acc[r][2] = fmaf(pj, __uint_as_float(vv.y << 16), acc[r][2]);
                acc[r][3] = fmaf(pj, __uint_as_float(vv.y & 0xffff0000u), acc[r][3]);
            }
        }
    }
    #pragma unroll
    for (uint32_t r = 0; r < INK2A_SPLIT_ROWS; r++) {
        if (r >= rows) { break; }
        float *dst = partials + (((uint64_t)r * INK2A_HEADS + h) * INK2A_SPLITS + split) * INK2D_PARTIAL;
        if (lane == 0) { dst[0] = m[r]; dst[1] = l[r]; }
        #pragma unroll
        for (int d = 0; d < 4; d++) { dst[2 + 4 * lane + d] = acc[r][d]; }
    }
}

// out = BF16(sum of split partials / sum of their l).
__global__ void ink2d_merge_kernel(float *out, const float *partials, uint32_t splits) {
    const uint32_t rh = blockIdx.x, d = threadIdx.x;
    const float *src = partials + (uint64_t)rh * INK2A_SPLITS * INK2D_PARTIAL;
    float top = -INFINITY;
    for (uint32_t s = 0; s < splits; s++) { top = fmaxf(top, src[s * INK2D_PARTIAL]); }
    float total = 0.0f, value = 0.0f;
    for (uint32_t s = 0; s < splits; s++) {
        const float ms = src[s * INK2D_PARTIAL];
        if (ms == -INFINITY) { continue; }
        const float e = __expf(ms - top);
        total = fmaf(src[s * INK2D_PARTIAL + 1], e, total);
        value = fmaf(src[s * INK2D_PARTIAL + 2 + d], e, value);
    }
    out[(uint64_t)rh * INK2A_DIM + d] = glmvq_round(__fdiv_rn(value, total));
}

extern "C" int ds4_gpu_ink2_draft_attn(ds4_gpu_tensor *out, ds4_gpu_tensor *partials, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *ring_k, const ds4_gpu_tensor *ring_v, uint32_t ring, uint32_t rows, uint32_t pos0,
        uint32_t first, uint32_t window) {
    const uint64_t heads = (uint64_t)rows * INK2A_HEADS;
    const uint32_t end = pos0 + rows;
    if (!rows || rows > INK2A_SPLIT_ROWS || !ring || first >= end || !gvq_floats(out, heads * INK2A_DIM) ||
        !gvq_floats(q, heads * INK2A_DIM) || !partials || partials->bytes < ds4_gpu_ink2_attn_bytes() ||
        ring_k->bytes < (uint64_t)ring * INK2A_KV * sizeof(uint16_t) ||
        ring_v->bytes < (uint64_t)ring * INK2A_KV * sizeof(uint16_t)) { return 0; }
    const uint32_t lo = max(first, end > ring ? end - ring : 0u);
    const uint32_t splits = ink2a_splits(end - lo);
    const cudaStream_t stream = ds4_current_stream();
    ink2d_attn_kernel<<<dim3(splits, INK2A_KV_HEADS), INK2A_DTHREADS, 0, stream>>>((float *)partials->ptr,
        (const float *)q->ptr, (const uint16_t *)ring_k->ptr, (const uint16_t *)ring_v->ptr, ring, pos0, end, first,
        window, splits, rows);
    ink2d_merge_kernel<<<(uint32_t)heads, INK2A_DIM, 0, stream>>>((float *)out->ptr, (const float *)partials->ptr,
        splits);
    return cuda_ok(cudaGetLastError(), "Inkling-2 draft attention");
}

// out = BF16(silu(g) u) over BF16 rows (SiluAndMul without a limit).
__global__ void ink2d_silu_kernel(float *out, const float *g, const float *u, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const float a = glmvq_round(g[i]);
    out[i] = glmvq_round(glmvq_mul(__fdiv_rn(a, glmvq_add(1.0f, __expf(-a))), glmvq_round(u[i])));
}

extern "C" int ds4_gpu_ink2_draft_silu(ds4_gpu_tensor *out, const ds4_gpu_tensor *g, const ds4_gpu_tensor *u,
        uint64_t count) {
    if (!count || !gvq_floats(out, count) || !gvq_floats(g, count) || !gvq_floats(u, count)) { return 0; }
    ink2d_silu_kernel<<<(uint32_t)((count + 255) / 256), 256, 0, ds4_current_stream()>>>((float *)out->ptr,
        (const float *)g->ptr, (const float *)u->ptr, count);
    return cuda_ok(cudaGetLastError(), "Inkling-2 draft SiLU");
}

/* Block rows: row 0 already holds the anchor's normalized embedding; rows
 * 1.. take mask_embed. */
__global__ void ink2d_mask_kernel(float *h, const uint16_t *mask, uint32_t rows) {
    const uint32_t row = 1u + blockIdx.x;
    if (row >= rows) { return; }
    for (uint32_t i = threadIdx.x; i < INK2_HIDDEN; i += blockDim.x) { h[(uint64_t)row * INK2_HIDDEN + i] = ink2_bf(mask[i]); }
}

extern "C" int ds4_gpu_ink2_draft_mask(ds4_gvq_ctx *ctx, ds4_gpu_tensor *h, uint32_t mask, uint32_t rows) {
    const uint16_t *w = (const uint16_t *)gvq_weight(ctx, mask, GVQ_TYPE_BF16, INK2_HIDDEN);
    if (!w || rows < 2 || !gvq_floats(h, (uint64_t)rows * INK2_HIDDEN)) { return 0; }
    ink2d_mask_kernel<<<rows - 1u, INK2D_THREADS, 0, ds4_current_stream()>>>((float *)h->ptr, w, rows);
    return cuda_ok(cudaGetLastError(), "Inkling-2 draft mask rows");
}

/* rnn_proj's step-invariant input [0 (state), 0 (previous), hidden_k, f_rnn]
 * for proposal rows k = 0..steps-1 (block rows 1..steps). */
__global__ void ink2d_markov_in_kernel(float *z, const float *hidden, const float *f_rnn) {
    const uint32_t k = blockIdx.x;
    float *dst = z + (uint64_t)k * DS4_INK2_RNN_IN;
    for (uint32_t i = threadIdx.x; i < DS4_INK2_RNN_IN; i += blockDim.x) {
        const uint32_t at = i - 2u * DS4_INK2_RANK;
        dst[i] = i < 2u * DS4_INK2_RANK ? 0.0f
            : at < INK2_HIDDEN ? hidden[(uint64_t)(k + 1u) * INK2_HIDDEN + at] : f_rnn[at - INK2_HIDDEN];
    }
}

extern "C" int ds4_gpu_ink2_draft_markov_in(ds4_gpu_tensor *z, const ds4_gpu_tensor *hidden,
        const ds4_gpu_tensor *f_rnn, uint32_t steps) {
    if (!steps || !gvq_floats(z, (uint64_t)steps * DS4_INK2_RNN_IN) ||
        !gvq_floats(hidden, (uint64_t)(steps + 1u) * INK2_HIDDEN) || !gvq_floats(f_rnn, INK2_HIDDEN)) { return 0; }
    ink2d_markov_in_kernel<<<steps, INK2D_THREADS, 0, ds4_current_stream()>>>((float *)z->ptr,
        (const float *)hidden->ptr, (const float *)f_rnn->ptr);
    return cuda_ok(cudaGetLastError(), "Inkling-2 draft Markov input");
}

/* One recurrence step: z = BF16(invariant + bias + W[:, :256] state +
 * W[:, 256:512] w1[prev]) one warp per output, then (one CTA) the gates:
 * state = BF16(sigmoid(z0) state + (1 - sigmoid(z0)) tanh(z1)), latent =
 * BF16(tanh(z2)). prev = ids[step], the anchor or the last proposal. */
__global__ void ink2d_markov_z_kernel(float *z, const float *invariant, const uint16_t *w, const uint16_t *bias,
        const float *state, const uint16_t *w1, const int32_t *ids, uint32_t step) {
    const uint32_t o = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (o >= 3u * DS4_INK2_RANK) { return; }
    const uint16_t *row = w + (uint64_t)o * DS4_INK2_RNN_IN;
    const uint16_t *prev = w1 + (uint64_t)ids[step] * DS4_INK2_RANK;
    float s = 0.0f;
    for (uint32_t c = lane; c < DS4_INK2_RANK; c += 32) {
        s = fmaf(ink2_bf(row[c]), state[c], s);
        s = fmaf(ink2_bf(row[DS4_INK2_RANK + c]), ink2_bf(prev[c]), s);
    }
    for (uint32_t d = 16; d; d >>= 1) { s += __shfl_xor_sync(0xffffffffu, s, d); }
    if (!lane) {
        z[o] = glmvq_round(glmvq_add(glmvq_add(invariant[(uint64_t)step * 3u * DS4_INK2_RANK + o], ink2_bf(bias[o])), s));
    }
}

__global__ void ink2d_markov_gate_kernel(float *state, float *latent, const float *z) {
    const uint32_t i = threadIdx.x;
    const float g = __fdiv_rn(1.0f, glmvq_add(1.0f, __expf(-z[i])));
    const float c = tanhf(z[DS4_INK2_RANK + i]);
    state[i] = glmvq_round(glmvq_add(glmvq_mul(g, state[i]), glmvq_mul(glmvq_add(1.0f, -g), c)));
    latent[i] = glmvq_round(tanhf(z[2u * DS4_INK2_RANK + i]));
}

/* logits = 30 tanh((BF16(base) + BF16(fold_b + fold_w . latent)) / 30) over
 * the tokenizer rows, -inf past them: one warp per vocabulary row. */
__global__ void ink2d_fold_kernel(float *logits, const float *base, const uint16_t *fold_w, const uint16_t *fold_b,
        const float *latent) {
    const uint32_t v = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (v >= DS4_INK2_VOCAB) { return; }
    const uint16_t *row = fold_w + (uint64_t)v * DS4_INK2_RANK;
    float s = 0.0f;
    for (uint32_t c = lane; c < DS4_INK2_RANK; c += 32) { s = fmaf(ink2_bf(row[c]), latent[c], s); }
    for (uint32_t d = 16; d; d >>= 1) { s += __shfl_xor_sync(0xffffffffu, s, d); }
    if (lane) { return; }
    if (v >= DS4_INK2_TOKENS) { logits[v] = -INFINITY; return; }
    const float x = glmvq_add(glmvq_round(base[v]), glmvq_round(glmvq_add(ink2_bf(fold_b[v]), s)));
    logits[v] = glmvq_mul(30.0f, tanhf(__fdiv_rn(x, 30.0f)));
}

extern "C" int ds4_gpu_ink2_draft_markov(ds4_gvq_ctx *ctx, ds4_gpu_tensor *logits, ds4_gpu_tensor *state,
        ds4_gpu_tensor *latent, ds4_gpu_tensor *z, const ds4_gpu_tensor *invariant, const ds4_gpu_tensor *base,
        const ds4_gpu_tensor *fold_w, const ds4_gpu_tensor *fold_b, const ds4_gpu_tensor *ids, uint32_t proj,
        uint32_t bias, uint32_t w1, uint32_t step) {
    const uint16_t *pw = (const uint16_t *)gvq_weight(ctx, proj, GVQ_TYPE_BF16, 3ull * DS4_INK2_RANK * DS4_INK2_RNN_IN);
    const uint16_t *pb = (const uint16_t *)gvq_weight(ctx, bias, GVQ_TYPE_BF16, 3ull * DS4_INK2_RANK);
    const uint16_t *emb = (const uint16_t *)gvq_weight(ctx, w1, GVQ_TYPE_BF16, (uint64_t)DS4_INK2_VOCAB * DS4_INK2_RANK);
    if (!pw || !pb || !emb || step + 1u >= DS4_INK2_DRAFT_BLOCK || !gvq_floats(logits, DS4_INK2_VOCAB) ||
        !gvq_floats(state, DS4_INK2_RANK) || !gvq_floats(latent, DS4_INK2_RANK) ||
        !gvq_floats(z, 3u * DS4_INK2_RANK) || !gvq_floats(invariant, (uint64_t)(step + 1u) * 3u * DS4_INK2_RANK) ||
        !gvq_floats(base, (uint64_t)(step + 1u) * DS4_INK2_VOCAB) ||
        fold_w->bytes < (uint64_t)DS4_INK2_VOCAB * DS4_INK2_RANK * sizeof(uint16_t) ||
        fold_b->bytes < (uint64_t)DS4_INK2_VOCAB * sizeof(uint16_t) || ids->bytes < (step + 1u) * sizeof(int32_t)) {
        return 0;
    }
    const cudaStream_t stream = ds4_current_stream();
    ink2d_markov_z_kernel<<<3u * DS4_INK2_RANK / 8u, 256, 0, stream>>>((float *)z->ptr, (const float *)invariant->ptr,
        pw, pb, (const float *)state->ptr, emb, (const int32_t *)ids->ptr, step);
    ink2d_markov_gate_kernel<<<1, DS4_INK2_RANK, 0, stream>>>((float *)state->ptr, (float *)latent->ptr,
        (const float *)z->ptr);
    ink2d_fold_kernel<<<(DS4_INK2_VOCAB + 7u) / 8u, 256, 0, stream>>>((float *)logits->ptr,
        (const float *)base->ptr + (uint64_t)step * DS4_INK2_VOCAB, (const uint16_t *)fold_w->ptr,
        (const uint16_t *)fold_b->ptr, (const float *)latent->ptr);
    return cuda_ok(cudaGetLastError(), "Inkling-2 draft Markov step");
}

/* Fold inputs: rows c < 256 are hidden_out's column c ([4608][256] BF16),
 * row 256 its bias; the target head then maps each row to the vocabulary. */
__global__ void ink2d_fold_x_kernel(float *x, const uint16_t *w, const uint16_t *b) {
    const uint32_t c = blockIdx.x;
    for (uint32_t h = threadIdx.x; h < INK2_HIDDEN; h += blockDim.x) {
        x[(uint64_t)c * INK2_HIDDEN + h] = c < DS4_INK2_RANK ? ink2_bf(w[(uint64_t)h * DS4_INK2_RANK + c]) : ink2_bf(b[h]);
    }
}

__global__ void ink2d_fold_pack_kernel(uint16_t *fold_w, uint16_t *fold_b, const float *head_rows) {
    const uint64_t v = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= DS4_INK2_VOCAB) { return; }
    for (uint32_t c = 0; c < DS4_INK2_RANK; c++) {
        fold_w[v * DS4_INK2_RANK + c] = ink2d_bits(head_rows[(uint64_t)c * DS4_INK2_VOCAB + v]);
    }
    fold_b[v] = ink2d_bits(head_rows[(uint64_t)DS4_INK2_RANK * DS4_INK2_VOCAB + v]);
}

extern "C" int ds4_gpu_ink2_draft_fold_x(ds4_gvq_ctx *ctx, ds4_gpu_tensor *x, uint32_t weight, uint32_t bias) {
    const uint16_t *w = (const uint16_t *)gvq_weight(ctx, weight, GVQ_TYPE_BF16, (uint64_t)INK2_HIDDEN * DS4_INK2_RANK);
    const uint16_t *b = (const uint16_t *)gvq_weight(ctx, bias, GVQ_TYPE_BF16, INK2_HIDDEN);
    if (!w || !b || !gvq_floats(x, (uint64_t)(DS4_INK2_RANK + 1u) * INK2_HIDDEN)) { return 0; }
    ink2d_fold_x_kernel<<<DS4_INK2_RANK + 1u, INK2D_THREADS, 0, ds4_current_stream()>>>((float *)x->ptr, w, b);
    return cuda_ok(cudaGetLastError(), "Inkling-2 draft fold input");
}

extern "C" int ds4_gpu_ink2_draft_fold_pack(ds4_gpu_tensor *fold_w, ds4_gpu_tensor *fold_b,
        const ds4_gpu_tensor *head_rows) {
    if (!gvq_floats(head_rows, (uint64_t)(DS4_INK2_RANK + 1u) * DS4_INK2_VOCAB) ||
        fold_w->bytes < (uint64_t)DS4_INK2_VOCAB * DS4_INK2_RANK * sizeof(uint16_t) ||
        fold_b->bytes < (uint64_t)DS4_INK2_VOCAB * sizeof(uint16_t)) { return 0; }
    ink2d_fold_pack_kernel<<<(DS4_INK2_VOCAB + 255u) / 256u, 256, 0, ds4_current_stream()>>>(
        (uint16_t *)fold_w->ptr, (uint16_t *)fold_b->ptr, (const float *)head_rows->ptr);
    return cuda_ok(cudaGetLastError(), "Inkling-2 draft fold pack");
}

// Copy of F32 rows (a kernel, so it orders with the stream's launches).
__global__ void ink2d_copy_kernel(float *dst, const float *src, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { dst[i] = src[i]; }
}

extern "C" int ds4_gpu_ink2_draft_copy(ds4_gpu_tensor *dst, uint64_t dst_off, const ds4_gpu_tensor *src,
        uint64_t count) {
    if (!count || !gvq_floats(src, count) || dst->bytes < (dst_off + count) * sizeof(float)) { return 0; }
    ink2d_copy_kernel<<<(uint32_t)((count + 255) / 256), 256, 0, ds4_current_stream()>>>((float *)dst->ptr + dst_off,
        (const float *)src->ptr, count);
    return cuda_ok(cudaGetLastError(), "Inkling-2 draft copy");
}

/* A trial's rollback: the history of `width`-wide rows becomes the last
 * three rows of [hist; x[0..rows)] (the kept rows of a verify forward). */
extern "C" int ds4_gpu_ink2_history(ds4_gpu_tensor *hist, const ds4_gpu_tensor *x, uint32_t width, uint32_t rows) {
    if (!rows || !width || !gvq_floats(hist, (uint64_t)DS4_INK2_HISTORY * width) ||
        !gvq_floats(x, (uint64_t)rows * width)) { return 0; }
    ink2_history_kernel<<<(width + 127u) / 128u, 128, 0, ds4_current_stream()>>>((float *)hist->ptr,
        (const float *)x->ptr, width, rows);
    return cuda_ok(cudaGetLastError(), "Inkling-2 history rollback");
}
