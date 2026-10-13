/* ds41_router.cuh — DeepSeek V4.1 (ds41) router and SwiGLU, ported from the C
 * engine at /data/YoungAi (commit 3946dbc), file `src/cuda/cuda_v41_3.inc.cu`:
 *   - the router kernel and entry (:8-91, official Gate.forward:
 *     scores = linear(x, w); probs = sqrt(softplus(scores));
 *     indices = topk(probs + bias); weights = probs[indices]/(sum + 1e-20)
 *     * route_scale.  One warp per token, each lane holding 12 experts
 *     (384/32); one warp argmax per round, ties to the smaller expert id);
 *   - the SwiGLU kernel and entry (:93-107, official Expert: up clamped to
 *     +-limit, gate clamped to max=limit, h = silu(gate)*up rounded to bf16).
 *
 * The route-bias override store (ds4_gpu_v41_set_rb_override, :48-77) is the
 * zchain sidecar's rb half: it keeps a per-layer device copy of (on-disk
 * exp_probs_b + delta) and the router entry consults it by bias_offset.  No
 * mounted table, no delta — the router then reads the on-disk bias unchanged.
 */
#pragma once

/* 12 experts per lane covers 384 experts (384/32); the entry refuses above
 * that.  softplus uses torch's threshold 20 (z > 20 => z, else log1pf(expf)). */
#define V41_ROUTER_PER_LANE 12u

__global__ static void v41_router_kernel(int32_t *sel, float *wts, const float *logits, const float *bias,
                                         uint32_t n_tok, uint32_t n_expert, uint32_t topk, float route_scale) {
    v41_pdl_wait();   /* PDL: first instruction waits for the upstream; no-op when not launched through PDL */
    const uint32_t t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31u;
    if (t >= n_tok) return;
    const float *lg = logits + (uint64_t)t * n_expert;
    float pr[V41_ROUTER_PER_LANE], sc[V41_ROUTER_PER_LANE];
    #pragma unroll
    for (uint32_t k = 0; k < V41_ROUTER_PER_LANE; k++) {
        const uint32_t e = lane + 32u * k;
        if (e < n_expert) {
            const float z = lg[e];
            const float sp = z > 20.0f ? z : log1pf(expf(z));   /* softplus (beta=1, threshold 20 as in torch) */
            pr[k] = sqrtf(sp); sc[k] = pr[k] + bias[e];
        } else { pr[k] = 0.f; sc[k] = -INFINITY; }
    }
    uint32_t used = 0; float wsum = 0.f;
    for (uint32_t r = 0; r < topk; r++) {
        float bv = -INFINITY; int bk = -1;
        #pragma unroll
        for (uint32_t k = 0; k < V41_ROUTER_PER_LANE; k++) if (!((used >> k) & 1u) && sc[k] > bv) { bv = sc[k]; bk = (int)k; }
        int be = bk >= 0 ? (int)(lane + 32u * (uint32_t)bk) : -1;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off); const int oe = __shfl_xor_sync(0xffffffffu, be, off);
            if (oe >= 0 && (be < 0 || ov > bv || (ov == bv && oe < be))) { bv = ov; be = oe; }
        }
        float myp = 0.f;
        if (be >= 0 && (uint32_t)(be & 31) == lane) { const uint32_t k = (uint32_t)be >> 5; used |= 1u << k; myp = pr[k]; }
        for (int off = 16; off > 0; off >>= 1) myp += __shfl_xor_sync(0xffffffffu, myp, off);   /* only one lane is nonzero */
        /* Round results land immediately (2026-09-18): the old chosen[16]/
         * chosen_p[16] arrays used runtime (topk) indices and fell into local
         * memory.  The unnormalized myp is stored and divided by wsum at the
         * end: the formula myp/(wsum+1e-20)*route_scale is unchanged. */
        if (lane == 0) { sel[(uint64_t)t * topk + r] = be; wts[(uint64_t)t * topk + r] = myp; }
        wsum += myp;
    }
    if (lane == 0)
        for (uint32_t r = 0; r < topk; r++) wts[(uint64_t)t * topk + r] = wts[(uint64_t)t * topk + r] / (wsum + 1e-20f) * route_scale;
}

/* Route-bias override store (cuda_v41_3.inc.cu:48-77): per layer (keyed by
 * the exp_probs_b file offset — the 40 main layers and each tower have
 * distinct offsets, so the router signature does not change) a device copy of
 * (on-disk bias + delta).  The delta moves the SELECTION score only; the
 * weight score (the sqrtsoftplus probability) is untouched.  host_delta NULL
 * unloads: bias_offset 0 unloads all, else just that layer.  The table is
 * process-level, so a fresh mount unloads first (the engine's loader does).
 * Mounted means 100% applied; the on-disk file is never written. */
static struct { uint64_t off; float *dev; uint32_t n; } g_v41_rb[64];
static uint32_t g_v41_rb_n = 0;

extern "C" int ds4_gpu_v41_set_rb_override(const void *model_map, uint64_t model_size, uint64_t bias_offset, const float *host_delta, uint32_t n_expert) {
    if (!host_delta) {   /* unload: offset 0 = all layers */
        for (uint32_t i = 0; i < g_v41_rb_n;) {
            if (bias_offset == 0 || g_v41_rb[i].off == bias_offset) { (void)cudaFree(g_v41_rb[i].dev); g_v41_rb[i] = g_v41_rb[--g_v41_rb_n]; }
            else i++;
        }
        return 1;
    }
    if (!model_map || !n_expert || bias_offset > model_size || (uint64_t)n_expert * 4 > model_size - bias_offset) return 0;
    const float *base = (const float *)cuda_model_range_ptr(model_map, bias_offset, (uint64_t)n_expert * 4, "v41 rb base");
    if (!base) return 0;
    float *h = (float *)malloc((size_t)n_expert * 4);
    if (!h) return 0;
    /* the on-disk bias is a device copy or the host mapping: cudaMemcpyDefault
     * reads by UVA pointer (the engine's comment, cuda_v41_3.inc.cu:64) */
    if (cudaMemcpy(h, base, (size_t)n_expert * 4, cudaMemcpyDefault) != cudaSuccess) { (void)cudaGetLastError(); free(h); return 0; }
    for (uint32_t e = 0; e < n_expert; e++) h[e] += host_delta[e];
    uint32_t i = 0;
    for (; i < g_v41_rb_n; i++) if (g_v41_rb[i].off == bias_offset) break;
    if (i == g_v41_rb_n) {
        if (g_v41_rb_n >= 64u) { free(h); return 0; }
        if (cudaMalloc((void **)&g_v41_rb[i].dev, (size_t)n_expert * 4) != cudaSuccess) { (void)cudaGetLastError(); free(h); return 0; }
        g_v41_rb[i].off = bias_offset; g_v41_rb[i].n = n_expert; g_v41_rb_n++;
    } else if (g_v41_rb[i].n != n_expert) { free(h); return 0; }
    const int ok = cudaMemcpy(g_v41_rb[i].dev, h, (size_t)n_expert * 4, cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) (void)cudaGetLastError();
    free(h);
    return ok;
}

extern "C" int ds4_gpu_v41_router_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits,
                                         const void *model_map, uint64_t model_size, uint64_t bias_offset,
                                         uint32_t n_tok, uint32_t n_expert, uint32_t topk, float route_scale) {
    if (!selected || !weights || !logits || topk > 16u || n_expert > 32u * V41_ROUTER_PER_LANE) return 0;
    const float *bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, (uint64_t)n_expert * 4, "v41 gate bias");
    if (!bias) return 0;
    for (uint32_t i = 0; i < g_v41_rb_n; i++) if (g_v41_rb[i].off == bias_offset && g_v41_rb[i].n == n_expert) { bias = g_v41_rb[i].dev; break; }
    v41_router_kernel<<<(n_tok + 7u) / 8u, 256, 0, ds4_current_stream()>>>((int32_t *)selected->ptr, (float *)weights->ptr,
        (const float *)logits->ptr, bias, n_tok, n_expert, topk, route_scale);
    return cuda_ok(cudaGetLastError(), "v41 router");
}

/* Official Expert.forward: gate=w1(x).float(); up=w3(x).float();
 * up=clamp(+-limit); gate=clamp(max=limit); h = silu(gate)*up ->
 * (the routing weight multiply happens outside) -> .to(bf16).  gate/up here
 * are the bf16 outputs of the fp4 linear (the caller rounded them already). */
__global__ static void v41_swiglu_kernel(float *h, const float *g, const float *u, uint64_t n, float limit) {
    v41_pdl_wait();   /* PDL: first instruction waits for the upstream; no-op when not launched through PDL */
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    if (limit > 0.0f) { uv = fminf(fmaxf(uv, -limit), limit); gv = fminf(gv, limit); }
    h[i] = v41_bf16r((gv / (1.0f + expf(-gv))) * uv);
}

extern "C" int ds4_gpu_v41_swiglu_tensor(ds4_gpu_tensor *h, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
                                         uint32_t n_tok, uint32_t mid, float limit) {
    if (!h || !gate || !up) return 0;
    const uint64_t n = (uint64_t)n_tok * mid;
    v41_swiglu_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ds4_current_stream()>>>((float *)h->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, limit);
    return cuda_ok(cudaGetLastError(), "v41 swiglu");
}
