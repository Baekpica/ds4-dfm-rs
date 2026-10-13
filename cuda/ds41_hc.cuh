/* ds41_hc.cuh — DeepSeek V4.1 (ds41) hyper-connection (mHC) family, ported from
 * the C engine at /data/YoungAi (commit 3946dbc), file
 * src/cuda/cuda_v41_hc.inc.cu (mix -> sinkhorn split -> hc_pre -> hc_post).
 *
 * The official model.py hc_mixes / hc_split_sinkhorn / hc_pre / hc_post are
 * the ground truth; f32 compute with explicit bf16 rounding at the official
 * bf16 module boundaries.  The fused variants (mix's three launches folded
 * into the GEMV, split+pre+norm folded into one) are the engine's measured
 * mega-mHC first cut — their accumulation shapes are copied verbatim from the
 * unfused kernels, which is why the engine gates them by NLL/top-1 where a
 * thread count changed and by byte-compare where it did not.
 *
 * Port notes: launches on ds4_current_stream() (the engine's g_cur_stream);
 * the fallback of the mix entry needs the dense f32 GEMV (ds41_dense.cuh),
 * which is included before this file. */
#pragma once

/* ---- hc_mix: rsqrt(mean(flat^2)+eps) per row, multiplied onto the GEMM
 * result (cuda_v41_hc.inc.cu:12-22) ---- */
__global__ static void v41_row_rsqrt_kernel(float *inv, const float *x, uint32_t dim, float eps) {
    const uint32_t r = blockIdx.x; const float *xr = x + (uint64_t)r * dim;
    float s = 0.f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i] * xr[i];
    __shared__ float sh[256]; sh[threadIdx.x] = s; __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    if (threadIdx.x == 0) inv[r] = rsqrtf(sh[0] / (float)dim + eps);
}
__global__ static void v41_scale_rows_kernel(float *y, const float *inv, uint32_t cols, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= inv[i / cols];
}

/* ---- the fused mix (cuda_v41_hc.inc.cu:38-148) ----
 * Three launches folded into one: the row rsqrt (its grid was n_tok = one
 * block at decode, 1.1 ms/step of pure latency), the f32 GEMV (24 rows) and
 * the scale_rows pass.  V41_HC_MIX_SPLIT blocks per row share the 32 K
 * segments; the last-arriving block sums part[] in segment order (atomic
 * counter + threadfence) and folds the scale multiply in.  The rms preamble
 * and the segment dot keep the unfused kernels' accumulation shapes exactly,
 * so the output is bit-identical to the three-launch form. */
#define V41_HC_MIX_SPLIT 2u
__global__ static void v41_hc_mix_fused_kernel(float *mix, float *part, uint32_t *ctr, const float *w, const float *hc,
                                               uint32_t dim, uint32_t out_dim, float eps, uint32_t ksplit, uint32_t nsplit, uint32_t n_tok) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t r = blockIdx.x / nsplit, sp = blockIdx.x % nsplit, kper = ksplit / nsplit;
    __shared__ float sh[1024];
    __shared__ uint32_t s_last;
    /* Weights first, rms preamble second (2026-09-23): W has no dependency on
     * the preamble, and issuing it up front hides DRAM latency behind the
     * tree reduction.  Bit-identical: every item is the same float4 dot in
     * ascending c, just fetched earlier. */
    constexpr uint32_t PF = 5u;
    const uint32_t kpart = sp * kper + warp;
    const bool act = r < out_dim && warp < kper;
    const float *wr = w + (uint64_t)r * dim;
    const uint32_t c0 = kpart * 128u + lane * 4u, cst = ksplit * 128u;
    float4 wpre[PF];
    #pragma unroll
    for (uint32_t i = 0; i < PF; i++) {
        wpre[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (act && c0 + i * cst < dim) wpre[i] = *(const float4 *)(wr + c0 + i * cst);
    }
    v41_pdl_wait();   /* PDL: W (constants) already in flight; wait for the upstream hc here */
    for (uint32_t t = 0; t < n_tok; t++) {   /* tokens in ascending order; the body below is the single-token form */
    const float *x = hc + (uint64_t)t * dim;
    /* (1) the rms preamble, shape verbatim from v41_row_rsqrt_kernel; reads
     * batched 10 at a time (same accumulation order, bit-identical). */
    {
        float s = 0.f;
        for (uint32_t i0 = threadIdx.x; i0 < dim; i0 += 10u * blockDim.x) {
            float xv[10];
            #pragma unroll
            for (uint32_t j = 0; j < 10u; j++) { const uint32_t i = i0 + j * blockDim.x; xv[j] = i < dim ? x[i] : 0.f; }
            #pragma unroll
            for (uint32_t j = 0; j < 10u; j++) if (i0 + j * blockDim.x < dim) s += xv[j] * xv[j];
        }
        sh[threadIdx.x] = s; __syncthreads();
        /* The tree tail moves into warp 0 (2026-10-07): the last six levels
         * keep adding sh[i] + sh[i+k] in the same left-right order, without
         * the block-wide barrier per level (11 -> 6 barriers/token). */
        for (uint32_t k = blockDim.x / 2; k > 32u; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
        if (threadIdx.x < 32u) {
            float v = sh[threadIdx.x] + sh[threadIdx.x + 32u];              /* k=32 */
            #pragma unroll
            for (int k = 16; k > 0; k >>= 1) v += __shfl_down_sync(0xffffffffu, v, k);   /* k=16..1 */
            if (threadIdx.x == 0) sh[0] = v;
        }
        __syncthreads();
    }
    const float inv = rsqrtf(sh[0] / (float)dim + eps);
    /* (2) the GEMV ksplit branch, this block's kpart range only */
    float acc = 0.f;
    if (act) {
        #pragma unroll
        for (uint32_t i = 0; i < PF; i++) {
            const uint32_t c = c0 + i * cst;
            if (c < dim) {
                const float4 wv = wpre[i];
                const float4 xv = *(const float4 *)(x + c);
                acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
            }
        }
        for (uint32_t c = c0 + PF * cst; c < dim; c += cst) {
            const float4 wv = *(const float4 *)(wr + c);
            const float4 xv = *(const float4 *)(x + c);
            acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) part[((uint64_t)t * out_dim + r) * ksplit + kpart] = acc;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s_last = (atomicAdd(&ctr[(uint64_t)t * out_dim + r], 1u) == nsplit - 1u) ? 1u : 0u;
    __syncthreads();
    if (s_last && threadIdx.x == 0 && r < out_dim) {   /* last block: sum 32 segments in order, fold the scale, reset */
        __threadfence();
        const float *pr = part + ((uint64_t)t * out_dim + r) * ksplit;
        float v = 0.f;
        for (uint32_t k = 0; k < ksplit; k++) v += __ldcg(pr + k);   /* bypass L1 for other blocks' segments */
        mix[(uint64_t)t * out_dim + r] = v * inv;   /* (3) the retired scale_rows pass is this multiply */
        ctr[(uint64_t)t * out_dim + r] = 0u;
    }
    }
}
static v41_scratch g_v41_hc_part, g_v41_hc_ctr;

extern "C" int ds4_gpu_v41_hc_mix_tensor(ds4_gpu_tensor *mix, const ds4_gpu_tensor *hc, const void *model_map, uint64_t model_size,
                                         uint64_t fn_offset, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps) {
    const uint32_t dim = n_embd * n_hc, mix_hc = 2u * n_hc + n_hc * n_hc;
    if (!mix || !hc || hc->bytes < (uint64_t)n_tok * dim * 4 || mix->bytes < (uint64_t)n_tok * mix_hc * 4) return 0;
    /* The fused kernel only takes the shape whose GEMV would pick ksplit=8,
     * one row per block — the live 24 x 20480.  Any other shape (other hc
     * widths, prefill batches) walks the original three-launch path so an
     * accumulation order cannot silently change. */
    const uint64_t wbytes = (uint64_t)dim * mix_hc * 4;
    const uint32_t nseg = dim / 128u;
    if (n_tok <= V41_GEMV_MAX_TOK && (dim % 128u) == 0u && mix_hc * 8u < 32768u && nseg >= 32u &&
        fn_offset <= model_size && wbytes <= model_size - fn_offset) {
        const float *W = (const float *)cuda_model_range_ptr(model_map, fn_offset, wbytes, "v41 hc fn");
        if (W) {
            /* 1024 threads = 32 warps (the ncu account in the kernel header):
             * ksplit must equal the warp count, and the K segment unit is 128
             * elements, so dim >= 32*128 = 4096 (20480 live). */
            const uint64_t npart = (uint64_t)n_tok * mix_hc;
            const int fresh = g_v41_hc_ctr.cap < npart * 4;
            float *part = (float *)v41_grow(&g_v41_hc_part, npart * 32u * 4, "v41 hc mix part");
            uint32_t *ctr = (uint32_t *)v41_grow(&g_v41_hc_ctr, npart * 4, "v41 hc mix ctr");
            if (!part || !ctr) return 0;
            if (fresh && !cuda_ok(cudaMemsetAsync(ctr, 0, (size_t)npart * 4, ds4_current_stream()), "v41 hc mix ctr zero")) return 0;
            v41_pdl_register((const void *)v41_hc_mix_fused_kernel);   /* waits before touching hc/part/ctr */
            v41_hc_mix_fused_kernel<<<mix_hc * V41_HC_MIX_SPLIT, 1024, 0, ds4_current_stream()>>>(
                (float *)mix->ptr, part, ctr, W, (const float *)hc->ptr, dim, mix_hc, eps, 32u, V41_HC_MIX_SPLIT, n_tok);
            return cuda_ok(cudaGetLastError(), "v41 hc mix fused");
        }
    }
    static v41_scratch g_v41_hc_inv;
    float *inv = (float *)v41_grow(&g_v41_hc_inv, (uint64_t)n_tok * 4, "v41 hc inv");
    if (!inv) return 0;
    v41_row_rsqrt_kernel<<<n_tok, 256, 0, ds4_current_stream()>>>(inv, (const float *)hc->ptr, dim, eps);
    if (!cuda_ok(cudaGetLastError(), "v41 hc rsqrt")) return 0;
    if (!ds4_gpu_v41_matmul_f32_tensor(mix, model_map, model_size, fn_offset, dim, mix_hc, hc, n_tok)) return 0;
    const uint64_t n = (uint64_t)n_tok * mix_hc;
    v41_scale_rows_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ds4_current_stream()>>>((float *)mix->ptr, inv, mix_hc, n);
    return cuda_ok(cudaGetLastError(), "v41 hc mix scale");
}

/* ---- sinkhorn split (cuda_v41_hc.inc.cu:188-230): one block per row ---- */
__global__ static void v41_hc_split_kernel(float *pre, float *post, float *comb, const float *mix, const float *scale,
                                           const float *base, uint32_t hc, uint32_t iters, float eps) {
    const uint32_t n = blockIdx.x, mix_hc = 2u * hc + hc * hc;
    const float *m = mix + (uint64_t)n * mix_hc;
    __shared__ float c[64];
    if (threadIdx.x < hc) {
        pre[n * hc + threadIdx.x] = 1.f / (1.f + expf(-(m[threadIdx.x] * scale[0] + base[threadIdx.x]))) + eps;
        post[n * hc + threadIdx.x] = 2.f / (1.f + expf(-(m[hc + threadIdx.x] * scale[1] + base[hc + threadIdx.x])));
    }
    if (threadIdx.x < hc * hc) c[threadIdx.x] = m[2u * hc + threadIdx.x] * scale[2] + base[2u * hc + threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) {   /* the 4x4 sinkhorn runs serially in the reference's order */
        for (uint32_t j = 0; j < hc; j++) {            /* softmax(-1) + eps */
            float mx = -INFINITY; for (uint32_t k = 0; k < hc; k++) mx = fmaxf(mx, c[j * hc + k]);
            float s = 0.f; for (uint32_t k = 0; k < hc; k++) { c[j * hc + k] = expf(c[j * hc + k] - mx); s += c[j * hc + k]; }
            for (uint32_t k = 0; k < hc; k++) c[j * hc + k] = c[j * hc + k] / s + eps;
        }
        for (uint32_t k = 0; k < hc; k++) {            /* / (sum(-2) + eps) */
            float s = 0.f; for (uint32_t j = 0; j < hc; j++) s += c[j * hc + k];
            for (uint32_t j = 0; j < hc; j++) c[j * hc + k] /= (s + eps);
        }
        for (uint32_t it = 1; it < iters; it++) {
            for (uint32_t j = 0; j < hc; j++) { float s = 0.f; for (uint32_t k = 0; k < hc; k++) s += c[j * hc + k]; for (uint32_t k = 0; k < hc; k++) c[j * hc + k] /= (s + eps); }
            for (uint32_t k = 0; k < hc; k++) { float s = 0.f; for (uint32_t j = 0; j < hc; j++) s += c[j * hc + k]; for (uint32_t j = 0; j < hc; j++) c[j * hc + k] /= (s + eps); }
        }
    }
    __syncthreads();
    if (threadIdx.x < hc * hc) comb[(uint64_t)n * hc * hc + threadIdx.x] = c[threadIdx.x];
}
extern "C" int ds4_gpu_v41_hc_split_tensor(ds4_gpu_tensor *pre, ds4_gpu_tensor *post, ds4_gpu_tensor *comb,
                                           const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size,
                                           uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t iters,
                                           float eps, uint32_t n_tok) {
    if (!pre || !post || !comb || !mix || n_hc > 8u) return 0;
    const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    const float *sc = (const float *)cuda_model_range_ptr(model_map, scale_offset, 12, "v41 hc scale");
    const float *bs = (const float *)cuda_model_range_ptr(model_map, base_offset, (uint64_t)mix_hc * 4, "v41 hc base");
    if (!sc || !bs) return 0;
    v41_hc_split_kernel<<<n_tok, 64, 0, ds4_current_stream()>>>((float *)pre->ptr, (float *)post->ptr, (float *)comb->ptr,
                                                                 (const float *)mix->ptr, sc, bs, n_hc, iters, eps);
    return cuda_ok(cudaGetLastError(), "v41 hc split");
}

/* ---- hc_pre (cuda_v41_hc.inc.cu:233-245) ---- */
__global__ static void v41_hc_pre_kernel(float *out, const float *hc, const float *pre, uint32_t n_embd, uint32_t n_hc) {
    const uint32_t n = blockIdx.y;
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    float s = 0.f;
    for (uint32_t c = 0; c < n_hc; c++) s += pre[n * n_hc + c] * hc[((uint64_t)n * n_hc + c) * n_embd + d];
    out[(uint64_t)n * n_embd + d] = v41_bf16r(s);
}
extern "C" int ds4_gpu_v41_hc_pre_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *hc, const ds4_gpu_tensor *pre,
                                         uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!out || !hc || !pre) return 0;
    v41_hc_pre_kernel<<<dim3((n_embd + 255) / 256, n_tok), 256, 0, ds4_current_stream()>>>((float *)out->ptr, (const float *)hc->ptr, (const float *)pre->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "v41 hc pre");
}

/* ---- the fused split + hc_pre + rms_norm (cuda_v41_hc.inc.cu:246-404) ----
 * ★pre_in and pre are two different things★: the split inside produces THIS
 * layer's pre (for the lower half), while hc_pre must use the pre_in handed
 * down from the previous layer (the official Block.forward pre_mix).  Using
 * the same slot here does not error — every layer's residual mixing
 * coefficients shift one layer and the output still looks plausible. */
__global__ static void v41_hc_fused_kernel(float *pre, float *post, float *comb, float *x, float *xn,
                                           const float *mix, const float *hc, const float *pre_in,
                                           const float *scale, const float *base,
                                           const float *nw, uint32_t n_embd, uint32_t n_hc, uint32_t iters,
                                           float hc_eps, float norm_eps) {
    v41_pdl_wait();
    const uint32_t n = blockIdx.x, mix_hc = 2u * n_hc + n_hc * n_hc;
    const float *m = mix + (uint64_t)n * mix_hc;
    __shared__ float c[64], sh[1024];   /* sh slots = blockDim */
    /* The whole sinkhorn lives in warp 0 (2026-09-18): a 4x4 matrix does not
     * need 1024 threads through 80 block-wide barriers; the per-element
     * accumulation order is unchanged (row/column sums run in ascending
     * j/k inside one thread) -> bit-identical.  n_hc <= 5 keeps n_hc^2 <= 32. */
    const uint32_t warp = threadIdx.x >> 5;
    if (warp == 0) {
        if (threadIdx.x < n_hc) {
            pre[n * n_hc + threadIdx.x] = 1.f / (1.f + expf(-(m[threadIdx.x] * scale[0] + base[threadIdx.x]))) + hc_eps;
            post[n * n_hc + threadIdx.x] = 2.f / (1.f + expf(-(m[n_hc + threadIdx.x] * scale[1] + base[n_hc + threadIdx.x])));
        }
        if (threadIdx.x < n_hc * n_hc) c[threadIdx.x] = m[2u * n_hc + threadIdx.x] * scale[2] + base[2u * n_hc + threadIdx.x];
        __syncwarp();
        const uint32_t tj = threadIdx.x / n_hc, tk = threadIdx.x % n_hc;
        const bool act = threadIdx.x < n_hc * n_hc;
        float nv = 0.f;
        if (act) {   /* (1) row softmax + eps */
            float mx = -INFINITY;
            for (uint32_t k = 0; k < n_hc; k++) mx = fmaxf(mx, c[tj * n_hc + k]);
            float sum = 0.f;
            for (uint32_t k = 0; k < n_hc; k++) sum += expf(c[tj * n_hc + k] - mx);
            nv = expf(c[tj * n_hc + tk] - mx) / sum + hc_eps;
        }
        __syncwarp();
        if (act) c[tj * n_hc + tk] = nv;
        __syncwarp();
        if (act) {   /* (2) column normalization */
            float cs = 0.f;
            for (uint32_t j = 0; j < n_hc; j++) cs += c[j * n_hc + tk];
            nv = c[tj * n_hc + tk] / (cs + hc_eps);
        }
        __syncwarp();
        if (act) c[tj * n_hc + tk] = nv;
        __syncwarp();
        for (uint32_t it = 1; it < iters; it++) {
            if (act) {   /* row normalization */
                float rs = 0.f;
                for (uint32_t k = 0; k < n_hc; k++) rs += c[tj * n_hc + k];
                nv = c[tj * n_hc + tk] / (rs + hc_eps);
            }
            __syncwarp();
            if (act) c[tj * n_hc + tk] = nv;
            __syncwarp();
            if (act) {   /* column normalization */
                float cs = 0.f;
                for (uint32_t j = 0; j < n_hc; j++) cs += c[j * n_hc + tk];
                nv = c[tj * n_hc + tk] / (cs + hc_eps);
            }
            __syncwarp();
            if (act) c[tj * n_hc + tk] = nv;
            __syncwarp();
        }
        if (act) comb[(uint64_t)n * n_hc * n_hc + threadIdx.x] = c[threadIdx.x];
    }
    /* hc_pre does not wait on the sinkhorn (2026-09-23): it consumes pre_in
     * (the previous layer's), independent of this layer's warp-0 work. */
    float acc = 0.f;
    const float *hcn = hc + (uint64_t)n * n_hc * n_embd;
    const float *pin = pre_in + (uint64_t)n * n_hc;
    float *xr = x + (uint64_t)n * n_embd;
    /* The n_hc == 4, n_embd <= 5*blockDim specialization reads this thread's 5
     * d's x 4 paths in one batch (same values and order as the loop below). */
    constexpr uint32_t DP = 5u;
    float *on = xn + (uint64_t)n * n_embd;
    if (n_hc == 4u && n_embd <= DP * blockDim.x) {
        float hv[DP][4], vv[DP], wv[DP];
        #pragma unroll
        for (uint32_t j = 0; j < DP; j++) {
            const uint32_t d = threadIdx.x + j * blockDim.x;
            #pragma unroll
            for (uint32_t k = 0; k < 4u; k++) hv[j][k] = d < n_embd ? hcn[(uint64_t)k * n_embd + d] : 0.f;
            wv[j] = d < n_embd ? nw[d] : 0.f;
        }
        const float p0 = pin[0], p1 = pin[1], p2 = pin[2], p3 = pin[3];
        #pragma unroll
        for (uint32_t j = 0; j < DP; j++) {
            const uint32_t d = threadIdx.x + j * blockDim.x;
            if (d < n_embd) {
                float v = 0.f;
                v += p0 * hv[j][0]; v += p1 * hv[j][1]; v += p2 * hv[j][2]; v += p3 * hv[j][3];
                v = v41_bf16r(v);
                xr[d] = v; vv[j] = v;
                acc += v * v;
            }
        }
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
        const float inv = rsqrtf(sh[0] / (float)n_embd + norm_eps);
        #pragma unroll
        for (uint32_t j = 0; j < DP; j++) { const uint32_t d = threadIdx.x + j * blockDim.x; if (d < n_embd) on[d] = v41_bf16r(wv[j] * (vv[j] * inv)); }
        return;
    }
    for (uint32_t d = threadIdx.x; d < n_embd; d += blockDim.x) {
        float v = 0.f;
        for (uint32_t k = 0; k < n_hc; k++) v += pin[k] * hcn[(uint64_t)k * n_embd + d];
        v = v41_bf16r(v);
        xr[d] = v;
        acc += v * v;
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    const float inv = rsqrtf(sh[0] / (float)n_embd + norm_eps);
    for (uint32_t d = threadIdx.x; d < n_embd; d += blockDim.x) on[d] = v41_bf16r(nw[d] * (xr[d] * inv));
}
extern "C" int ds4_gpu_v41_hc_fused_tensor(ds4_gpu_tensor *pre, ds4_gpu_tensor *post, ds4_gpu_tensor *comb,
                                           ds4_gpu_tensor *x, ds4_gpu_tensor *xn, const ds4_gpu_tensor *mix,
                                           const ds4_gpu_tensor *hc, const ds4_gpu_tensor *pre_in,
                                           const void *model_map, uint64_t model_size,
                                           uint64_t scale_offset, uint64_t base_offset, uint64_t norm_offset,
                                           uint32_t n_embd, uint32_t n_hc, uint32_t iters, float hc_eps, float norm_eps,
                                           uint32_t n_tok) {
    if (!pre || !post || !comb || !x || !xn || !mix || !hc || !pre_in || n_hc > 5u) return 0;   /* n_hc^2 in one warp */
    const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    const float *sc = (const float *)cuda_model_range_ptr(model_map, scale_offset, 12, "v41 hc scale");
    const float *bs = (const float *)cuda_model_range_ptr(model_map, base_offset, (uint64_t)mix_hc * 4, "v41 hc base");
    const float *nw = (const float *)cuda_model_range_ptr(model_map, norm_offset, (uint64_t)n_embd * 4, "v41 norm w");
    if (!sc || !bs || !nw) return 0;
    /* 1024 threads (2026-09-17, ncu): the grid is n_tok — one block at decode
     * — and the rms tree changes slots with the thread count, so this cut's
     * gate is the quality scale, not cmp. */
    v41_hc_fused_kernel<<<n_tok, 1024, 0, ds4_current_stream()>>>((float *)pre->ptr, (float *)post->ptr, (float *)comb->ptr,
        (float *)x->ptr, (float *)xn->ptr, (const float *)mix->ptr, (const float *)hc->ptr,
        (const float *)pre_in->ptr, sc, bs, nw,
        n_embd, n_hc, iters, hc_eps, norm_eps);
    return cuda_ok(cudaGetLastError(), "v41 hc fused");
}

/* ---- hc_post (cuda_v41_hc.inc.cu:409-430): out[k][d] = post[k]*y[d] + sum_j comb[j][k]*res[j][d] -> bf16 ---- */
__global__ static void v41_hc_post_kernel(float *out, const float *y, const float *res, const float *post, const float *comb,
                                          uint32_t n_embd, uint32_t n_hc) {
    v41_pdl_wait();
    const uint32_t n = blockIdx.y;
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    const float yv = y[(uint64_t)n * n_embd + d];
    for (uint32_t k = 0; k < n_hc; k++) {
        float s = post[n * n_hc + k] * yv;
        for (uint32_t j = 0; j < n_hc; j++) s += comb[(uint64_t)n * n_hc * n_hc + j * n_hc + k] * res[((uint64_t)n * n_hc + j) * n_embd + d];
        out[((uint64_t)n * n_hc + k) * n_embd + d] = v41_bf16r(s);
    }
}
extern "C" int ds4_gpu_v41_hc_post_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *y, const ds4_gpu_tensor *res,
                                          const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
                                          uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!out_hc || !y || !res || !post || !comb || out_hc->ptr == res->ptr) return 0;   /* no in-place: every k reads all j */
    v41_hc_post_kernel<<<dim3((n_embd + 255) / 256, n_tok), 256, 0, ds4_current_stream()>>>((float *)out_hc->ptr, (const float *)y->ptr, (const float *)res->ptr,
                                                                                             (const float *)post->ptr, (const float *)comb->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "v41 hc post");
}
