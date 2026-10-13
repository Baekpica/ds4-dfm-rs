/* ds41_vq_prefill_fused.cuh — DeepSeek V4.1 (ds41) VQ expert prefill, the
 * v2 (f16-codebook) arm: decode is the multiply, no f16 staging, ported from
 * the C engine at /data/YoungAi (commit 3946dbc), file
 * `src/cuda/cuda_vq_prefill_fused.inc.cu` (whole file, the negative-result
 * archives at its tail included as comments).
 *
 * Why this shape (the engine's 2026-09-15 measurement, kept): a 512-token
 * block routes ~8 tokens per expert on average, while the expert holds 35.4M
 * weights — dequantizing the whole expert to f16 for 8 tokens is arithmetic
 * intensity 8 (225 ms/layer of traffic for 2.7 ms of math).  Fused, one
 * layer reads the blob once: ~11 ms.  So a work item is (expert, token
 * segment, count): a block loads one codebook and each of its weight rows is
 * decoded once and dotted against the <= 8 activations the expert owns.
 *
 * The K-order numerics (kept from the engine): activation/row sums run f32
 * with the official bf16 boundary rounding — the same contract as the decode
 * fused kernels — and the dot's K sum order is "32-lane groups + shuffle
 * tree", which is not bit-equal to any GEMM by construction.  The gate is the
 * NLL/PPL scale (the engine's cuda_v41_hc.inc.cu:36 doctrine), not cmp.
 *
 * The v2 arm is what a DQVL v2 blob (V4 legacy) routes to; the artifact's
 * real blob is DQVL v3 13-bit (measured, plan §6.7) and routes to the
 * tensor-core run in ds41_vq_prefill_mma.cuh.  Both arms are ported; only the
 * v3 arm is gated (the v2 one has no artifact to gate against — named).
 *
 * Must follow ds41_vq_prefill.cuh (V41_VQN_PAD, g_v41_gr, g_vqp) and
 * ds41_vq_decode.cuh (v41_vq_cb_to_shared, v41_vq_swiglu); the mma file
 * follows this one (vqp_item, vqm_run's forward declaration).
 */
#pragma once

#define V41_VQP_NT     8u          /* tokens per work item (acc lives in local memory, see the NT archive below) */
#define V41_VQP_ITERS  8u          /* rows a warp loops: the codebook is copied once per 32*8 = 256 rows */
#define V41_VQP_ROWS   (32u * V41_VQP_ITERS)

/* Sorted activations, f32 (the old path used f16; the fused kernel reads f32
 * directly — one less precision round trip). */
__global__ static void vqp_gather32_kernel(float *xs, const float *x, const int32_t *perm, uint32_t K, uint32_t IN) {
    const uint32_t i = blockIdx.x;
    const uint32_t t = (uint32_t)perm[i] / K;
    const float *src = x + (uint64_t)t * IN;
    float *dst = xs + (uint64_t)i * IN;
    for (uint32_t d = threadIdx.x; d < IN; d += blockDim.x) dst[d] = src[d];
}

/* One weight row x nt activation vectors.  The decode part is
 * v41_vq_row_dot's expression verbatim (bitstream -> 16 B codebook entry ->
 * 8 halves); the one difference is that the weight row is decoded once and
 * reused for nt activations — the whole reason this file exists.  acc comes
 * out warp-reduced and multiplied by the row gain (gov override included);
 * only lane 0's value is valid. */
template <int V3, int EXT>
__device__ __forceinline__ static void v41_vq_row_dotN(const v41_vq_mat &m, uint32_t r, const float *xs,
                                                       uint32_t xstride, uint32_t nt, const uint8_t *cbs, float *acc) {
    const uint64_t i0 = (uint64_t)r * m.nidx_row;
    const uint32_t MB = V3 ? 12u : m.nbit;                       /* main-stream bit width */
    const uint8_t *ex = EXT ? m.ex + (size_t)r * ((m.nidx_row + 7u) >> 3) : NULL;
    for (uint32_t t = 0; t < V41_VQP_NT; t++) acc[t] = 0.f;
    for (uint32_t j = threadIdx.x & 31u; j < m.nidx_row; j += 32u) {
        const uint64_t bit = (V3 ? (uint64_t)j : (i0 + j)) * MB + (V3 ? (uint64_t)r * ((m.nidx_row * MB + 7u) / 8u) * 8u : 0u), by = bit >> 3;
        uint32_t wv; memcpy(&wv, m.ix + by, 4);
        uint32_t v = (wv >> (bit & 7)) & (EXT ? 0xFFFu : m.imsk);
        if (EXT) v |= (uint32_t)((ex[j >> 3] >> (j & 7)) & 1u) << 12;
        float f0x, f0y, f1x, f1y, f2x, f2y, f3x, f3y;
        if (V3) {   /* E4M3 via the hardware instruction, never a scalar table (the engine's v41_e4m3x2_to_half2 lesson) */
            const uint2 w = *(const uint2 *)(cbs + (size_t)v * 8u);
            const float2 e0 = __half22float2(v41_e4m3x2_to_half2(w.x)), e1 = __half22float2(v41_e4m3x2_to_half2(w.x >> 16)),
                         e2 = __half22float2(v41_e4m3x2_to_half2(w.y)), e3 = __half22float2(v41_e4m3x2_to_half2(w.y >> 16));
            f0x = e0.x; f0y = e0.y; f1x = e1.x; f1y = e1.y; f2x = e2.x; f2y = e2.y; f3x = e3.x; f3y = e3.y;
        } else {
            const uint4 c = *(const uint4 *)(cbs + (size_t)v * 16u);
            __half2 h0, h1, h2, h3; memcpy(&h0, &c.x, 4); memcpy(&h1, &c.y, 4); memcpy(&h2, &c.z, 4); memcpy(&h3, &c.w, 4);
            const float2 g0 = __half22float2(h0), g1 = __half22float2(h1), g2 = __half22float2(h2), g3 = __half22float2(h3);
            f0x = g0.x; f0y = g0.y; f1x = g1.x; f1y = g1.y; f2x = g2.x; f2y = g2.y; f3x = g3.x; f3y = g3.y;
        }
        const float2 f0 = make_float2(f0x, f0y), f1 = make_float2(f1x, f1y), f2 = make_float2(f2x, f2y), f3 = make_float2(f3x, f3y);
        for (uint32_t t = 0; t < nt; t++) {
            const float *xt = xs + (uint64_t)t * xstride + (size_t)j * 8u;
            const float4 xa = *(const float4 *)xt, xb = *(const float4 *)(xt + 4u);
            acc[t] += f0.x * xa.x + f0.y * xa.y + f1.x * xa.z + f1.y * xa.w +
                      f2.x * xb.x + f2.y * xb.y + f3.x * xb.z + f3.y * xb.w;
        }
    }
    __half gh; memcpy(&gh, m.gr + (size_t)r * 2u, 2);
    const float g = __half2float(gh) * (m.gov ? m.gov[r] : 1.0f);
    for (uint32_t t = 0; t < nt; t++) {
        for (int o = 16; o > 0; o >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], o);
        acc[t] *= g;
    }
}

/* A work item: the [t0, t0+nt) tokens of expert e in sorted order (= off[e]+t0 onwards). */
typedef struct { int32_t e, t0, nt; } vqp_item;

/* gate and up are two launches, not one (2026-09-15 measurement): keeping
 * g[ITERS][NT] and u[ITERS][NT] live together is 128 f32 accumulators per
 * thread, far over the register budget — everything spills to local memory
 * and the fusion buys +5% only.  Split, each thread holds acc[NT] = 8; the
 * codebook is loaded twice (64 KB/block, L2-resident, negligible).
 * which=0: gate -> g32; which=1: up, reads g32 back, clamp + SwiGLU -> h32
 * (the official Expert's order and rounding point). */
template <int V3, int EXT>
__global__ static void vqp_fused_gu_kernel(float *dst, const float *g32, const uint8_t *blob, const vqp_item *items,
                                           const float *xs, const uint32_t *off, uint32_t IN, uint32_t MID,
                                           float clamp, uint32_t cb_bytes, int which) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const vqp_item it = items[blockIdx.y];
    const uint32_t nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
    const v41_vq_mat m = v41_vq_open<V3>(blob, it.e, which, MID, IN, NULL);
    if (!m.ok || m.nc * (V3 ? 8u : 16u) != cb_bytes) return;
    const uint32_t r0 = blockIdx.x * V41_VQP_ROWS + (threadIdx.x >> 5);
    const float *xb = xs + (uint64_t)base * IN;
    v41_vq_cb_to_shared(vqsh, m.cb, cb_bytes);
    __syncthreads();
    for (uint32_t i = 0; i < V41_VQP_ITERS; i++) {
        const uint32_t r = r0 + i * 32u;
        if (r >= MID) break;
        float acc[V41_VQP_NT];
        v41_vq_row_dotN<V3, EXT>(m, r, xb, IN, nt, vqsh, acc);
        if ((threadIdx.x & 31u) != 0) continue;
        for (uint32_t t = 0; t < nt; t++) {
            const uint64_t o = (uint64_t)(base + t) * MID + r;
            if (!which) { dst[o] = v41_bf16r(acc[t]); continue; }
            float gi = g32[o], ui = v41_bf16r(acc[t]);
            if (clamp > 0.f) { if (gi > clamp) gi = clamp; if (ui > clamp) ui = clamp; if (ui < -clamp) ui = -clamp; }
            const float sg = gi / (1.0f + expf(-gi));
            dst[o] = v41_bf16r(sg * ui);
        }
    }
}

/* down: ys[sorted pos][OUT] = bf16(W2*h) — the same buffer the old path
 * wrote, so the downstream reduce is untouched. */
template <int V3, int EXT>
__global__ static void vqp_fused_down_kernel(float *ys, const uint8_t *blob, const vqp_item *items, const float *h,
                                             const uint32_t *off, uint32_t MID, uint32_t OUT, uint32_t cb_bytes, const float *gr) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const vqp_item it = items[blockIdx.y];
    const uint32_t nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
    const v41_vq_mat md = v41_vq_open<V3>(blob, it.e, 2, OUT, MID, gr ? gr + (size_t)it.e * OUT : NULL);
    const uint32_t r0 = blockIdx.x * V41_VQP_ROWS + (threadIdx.x >> 5);
    if (!md.ok || md.nc * (V3 ? 8u : 16u) != cb_bytes) {   /* bad payload: zero the rows this block owns, no dirty values for the reduce */
        if ((threadIdx.x & 31u) == 0)
            for (uint32_t i = 0; i < V41_VQP_ITERS; i++) {
                const uint32_t r = r0 + i * 32u;
                if (r < OUT) for (uint32_t t = 0; t < nt; t++) ys[(uint64_t)(base + t) * OUT + r] = 0.f;
            }
        return;
    }
    v41_vq_cb_to_shared(vqsh, md.cb, cb_bytes);
    __syncthreads();
    const float *hb = h + (uint64_t)base * MID;
    for (uint32_t i = 0; i < V41_VQP_ITERS; i++) {
        const uint32_t r = r0 + i * 32u;
        if (r >= OUT) break;
        float acc[V41_VQP_NT];
        v41_vq_row_dotN<V3, EXT>(md, r, hb, MID, nt, vqsh, acc);
        if ((threadIdx.x & 31u) == 0)
            for (uint32_t t = 0; t < nt; t++) ys[(uint64_t)(base + t) * OUT + r] = v41_bf16r(acc[t]);
    }
}

static struct { v41_scratch d, doff, h32, g32, xs32; } g_vqpf;
static int g_vqpf_sh = 0;   /* 0 undecided / 1 codebook in shared / -1 does not fit (then this path is unavailable, hard failure) */

/* The tensor-core run for DQVL v3 (ds41_vq_prefill_mma.cuh). */
static int vqm_run(const uint8_t *blob, const uint32_t *cnt, const uint32_t *off_h, uint32_t n_total_expert, uint32_t nvalid,
                   uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t nc, float clamp, const float *x, const int32_t *perm,
                   uint32_t n_expert, uint32_t layer_index, float *ys, const float *gr);

/* Returns 0 = this path is unavailable (the caller hard-fails; a silent
 * fallback to another path would hide which path ran, cuda_vq_prefill_fused.inc.cu:162). */
static int vqp_fused_run(const uint8_t *blob, const uint32_t *cnt, const uint32_t *off_h, uint32_t n_total_expert,
                         uint32_t nvalid, uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t nc, float clamp,
                         const float *x, const int32_t *perm, uint32_t n_expert, uint32_t layer_index, uint32_t ver) {
    /* v3 (E4M3 codebook) takes the bf16 tensor cores (2026-09-24): same
     * products, same rounding point, different K accumulation order, several
     * times faster per expert block.  v2's f16 codebook would lose 3 mantissa
     * bits on the way to bf16, so v2 stays on the fused kernels below. */
    if (ver == 3u)
        return vqm_run(blob, cnt, off_h, n_total_expert, nvalid, IN, MID, OUT, nc, clamp, x, perm, n_expert, layer_index,
                       (float *)g_vqp.ys.p, g_v41_gr[layer_index < 64u ? layer_index : 0]);
    uint32_t nbit = 0; while ((1u << nbit) < nc) nbit++;
    /* v3 codebooks are 8 E4M3 per word (nc8192 = 64 KB, same as today's
     * nc4096 f16); v2 carries 8 f16 per word. */
    const uint32_t cbb = nc * (ver == 3u ? 8u : 16u);
    if (g_vqpf_sh == 0) {
        /* The opt-in shared size must be set on the instance pair that will
         * actually launch; all four pairs are set here (two kernels each) so
         * whichever fires later is already armed.  One failure marks this
         * path unavailable — never a silent fallback. */
        bool og = true, od = true;
#define VQP_ATTR(V3, EXT) do { \
            og = og && cudaFuncSetAttribute(vqp_fused_gu_kernel<V3, EXT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess; \
            od = od && cudaFuncSetAttribute(vqp_fused_down_kernel<V3, EXT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess; \
        } while (0)
        if (ver == 3u) { if (nbit == 13u) VQP_ATTR(1, 1); else VQP_ATTR(1, 0); }
        else VQP_ATTR(0, 0);
#undef VQP_ATTR
        (void)cudaGetLastError();
        g_vqpf_sh = (og && od) ? 1 : -1;
        fprintf(stderr, "ds4: [ds41] vq-prefill fused codebook %u KB/copy (DQVL v%u, %u bit) -> %s\n", cbb >> 10, ver, nbit,
                g_vqpf_sh == 1 ? "in shared" : "does not fit, path unavailable");
    }
    if (g_vqpf_sh != 1) return 0;
    /* Work items: each expert split into NT-token segments. */
    uint32_t nit = 0;
    for (uint32_t e = 0; e < n_total_expert; e++) nit += (cnt[e] + V41_VQP_NT - 1u) / V41_VQP_NT;
    if (!nit) return 0;
    vqp_item *ih = (vqp_item *)malloc((size_t)nit * sizeof(vqp_item));
    if (!ih) return 0;
    uint32_t k = 0;
    for (uint32_t e = 0; e < n_total_expert; e++)
        for (uint32_t t0 = 0; t0 < cnt[e]; t0 += V41_VQP_NT) {
            const uint32_t nt = cnt[e] - t0 < V41_VQP_NT ? cnt[e] - t0 : V41_VQP_NT;
            ih[k].e = (int32_t)e; ih[k].t0 = (int32_t)t0; ih[k].nt = (int32_t)nt; k++;
        }
    int ok = v41_grow(&g_vqpf.d, (uint64_t)nit * sizeof(vqp_item), "vq prefill fused items") &&
             v41_grow(&g_vqpf.doff, ((uint64_t)n_total_expert + 1u) * sizeof(uint32_t), "vq prefill fused off") &&
             v41_grow(&g_vqpf.xs32, ((uint64_t)nvalid + V41_VQN_PAD) * IN * sizeof(float), "vq prefill fused xs32") &&
             v41_grow(&g_vqpf.h32, ((uint64_t)nvalid + V41_VQN_PAD) * MID * sizeof(float), "vq prefill fused h32") &&
             v41_grow(&g_vqpf.g32, ((uint64_t)nvalid + V41_VQN_PAD) * MID * sizeof(float), "vq prefill fused g32");
    if (ok) ok = cudaMemcpyAsync(g_vqpf.d.p, ih, (size_t)nit * sizeof(vqp_item), cudaMemcpyHostToDevice, ds4_current_stream()) == cudaSuccess &&
                 cudaMemcpyAsync(g_vqpf.doff.p, off_h, (size_t)(n_total_expert + 1u) * sizeof(uint32_t), cudaMemcpyHostToDevice, ds4_current_stream()) == cudaSuccess;
    free(ih);
    if (!ok) { (void)cudaGetLastError(); fprintf(stderr, "ds4: [ds41] vq-prefill L%u fused scratch/copy failed\n", layer_index); return 0; }
    vqp_gather32_kernel<<<nvalid, 256, 0, ds4_current_stream()>>>((float *)g_vqpf.xs32.p, x, perm, n_expert, IN);
    if (!cuda_ok(cudaGetLastError(), "vq prefill gather32")) return 0;
    /* The NVFP4 tensor-core route was retired here on 2026-09-20 (the engine's
     * own decision, measured): it requantized weights/activations/h to E2M1
     * three times per layer and cost 0.013 Σmin — more than the 12->11 bit
     * width cost of the file itself.  Not ported; the fused kernels below are
     * the live v2 path. */
    const dim3 gm((MID + V41_VQP_ROWS - 1u) / V41_VQP_ROWS, nit);
    const dim3 gd((OUT + V41_VQP_ROWS - 1u) / V41_VQP_ROWS, nit);
    /* Version/width select the instance (2026-09-21): v2 and v3 layouts
     * differ, a wrong pick decodes fake weights silently, so ver comes from
     * the blob header (the caller) and the width from the codebook word count.
     * One macro launches all three kernels with the same template pair — three
     * separate blocks would eventually miss one. */
#define VQP_LAUNCH(V3, EXT) do { \
        vqp_fused_gu_kernel<V3, EXT><<<gm, 32u * 32u, cbb, ds4_current_stream()>>>((float *)g_vqpf.g32.p, NULL, blob, (const vqp_item *)g_vqpf.d.p, (const float *)g_vqpf.xs32.p, (const uint32_t *)g_vqpf.doff.p, IN, MID, clamp, cbb, 0); \
        if (!cuda_ok(cudaGetLastError(), "vq prefill fused gate")) return 0; \
        vqp_fused_gu_kernel<V3, EXT><<<gm, 32u * 32u, cbb, ds4_current_stream()>>>((float *)g_vqpf.h32.p, (const float *)g_vqpf.g32.p, blob, (const vqp_item *)g_vqpf.d.p, (const float *)g_vqpf.xs32.p, (const uint32_t *)g_vqpf.doff.p, IN, MID, clamp, cbb, 1); \
        if (!cuda_ok(cudaGetLastError(), "vq prefill fused up")) return 0; \
        vqp_fused_down_kernel<V3, EXT><<<gd, 32u * 32u, cbb, ds4_current_stream()>>>( \
            (float *)g_vqp.ys.p, blob, (const vqp_item *)g_vqpf.d.p, (const float *)g_vqpf.h32.p, (const uint32_t *)g_vqpf.doff.p, MID, OUT, cbb, g_v41_gr[layer_index < 64u ? layer_index : 0]); \
        return cuda_ok(cudaGetLastError(), "vq prefill fused down"); \
    } while (0)
    if (ver == 3u) {
        if (nbit == 12u) VQP_LAUNCH(1, 0);
        if (nbit == 13u) VQP_LAUNCH(1, 1);
        fprintf(stderr, "ds4: [ds41] vq-prefill L%u DQVL v3 codebook %u words has no instance (12/13 only)\n", layer_index, nc);
        return 0;
    }
    VQP_LAUNCH(0, 0);
#undef VQP_LAUNCH
}

/* Negative-result archive kept from the engine (2026-09-15), so nobody walks
 * it again: "one warp owns R rows, activations read once and shared" lost in
 * all three configurations.  The account was right (each row reads nt x 20 KB
 * of activations while its weights are 960 B, 170:1) but it hit two walls:
 *   1. a 64 KB codebook fills shared => one block per SM => block size IS the
 *      occupancy.  The R-row form's register need (acc[R][NT] + xv[NT][8])
 *      forces 256-thread blocks, 8 warps/SM: 34.3 -> 38.1 s.
 *   2. holding 1024 threads (<= 64 registers/thread, NT=2/R=16 = 48) still
 *      spills to local memory: 34.3 -> 78.1 s.
 *   Also all three R-row configs landed on PPL 15.53 vs this version's 15.17 —
 *   an unresolved numeric difference; restarting that road requires finding
 *   it first, never covering it with "faster". */
