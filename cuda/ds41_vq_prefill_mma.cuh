/* ds41_vq_prefill_mma.cuh — DeepSeek V4.1 (ds41) VQ expert prefill on bf16
 * tensor cores, the DQVL v3 arm, ported from the C engine at /data/YoungAi
 * (commit 3946dbc), files `src/cuda/cuda_vq_prefill_mma.inc.cu` (vqm, the
 * tile form) and `src/cuda/cuda_vq_reg_mma.inc.cu` (vqs, the register-direct
 * form; bit-identical to vqm, the engine's own microbench).
 *
 * Why (the engine's 2026-09-24 profile): the fused scalar kernels were 76% of
 * a 12k-prompt run's GPU time — 2.6 TFLOPS against a ~11 ms/layer read floor.
 * A work item could only carry 8 tokens (64 KB codebook fills shared, 64
 * registers/thread), so the same expert's bitstream was re-read for hot
 * experts.  Tensor cores take the math out: one mma.m16n8k16 is 4096 MACs.
 *
 * Why bf16, and the same products as the fused path (the 2026-09-19 lesson:
 * the NVFP4 route pressed activations to 4 bits and gave up 0.013 Σmin):
 *   - weights: the v3 codebook is E4M3 (3 mantissa bits); bf16 has 7 — E4M3 ->
 *     bf16 is bit-exact.
 *   - activations: rms_norm / swiglu exits are already rounded to bf16.
 *   - a bf16 x bf16 product is exact in f32; the tensor core accumulates in
 *     f32; the row gain (override included) multiplies after the sum, the
 *     same point as the fused path.
 *   => the only difference from the fused path is the K accumulation order
 *   (which is a 32-lane split + shuffle tree there and matches no GEMM
 *   bit-for-bit anyway).  The criterion is PPL / the five metrics, not bytes.
 *   v2's f16 codebook would lose 3 mantissa bits to bf16, so v2 stays on the
 *   fused kernels.
 *
 * Shapes (2026-09-29, shaped by the engine's microbench on the real payload
 * and real routing): vqm = one block is 16 warps / 512 threads = 128 weight
 * rows x one work item (<= 128 tokens of one expert), stepping 64 columns of
 * K per round; vqs = one block is (work item, 256 rows) x all of K with
 * cp.async segments in flight and the decode landing straight in mma
 * fragments.  The engine's negative results are archived in the comments.
 *
 * Bit-exactness (kept from the engine): products, the per-k16 zero-start +
 * FADD-back accumulation, and the exit rounding point are the same as the
 * 09-24 shape — the engine's microbench compared g32/h16/H_u/ys byte for byte
 * against vqm_kernel, 12/13-bit x both routings.  vqm/vqs vs the fused path
 * is the K-order difference above.
 *
 * Portability deviation, named: vqs's L2 bulk prefetch (vqs_pf_l2) is sm_90+;
 * this TU also compiles for sm_89 (the local dev/test device).  Below sm_90
 * the prefetch asm is compiled out — a pure performance hint, the segment
 * cp.async loads are the real fetches — so those devices run vqs correctly
 * and numerically identically; sm_121 gets the engine's verbatim code.
 *
 * Must follow ds41_vq_prefill_fused.cuh (vqp_item, g_vqp.ys, g_v41_gr).
 */
#pragma once
#include <type_traits>   /* std::integral_constant for the vqs slice-count switch */

#define VQM_BM      128u   /* weight rows per block: 8 tiles of 16 */
#define VQM_BN      128u   /* tokens per work item: 16 n8 slices */
#define VQM_BK      64u    /* K columns per round: 8 code words per row */
#define VQM_TW      2u     /* token halves across warp groups => 16 warps, 512 threads */
#define VQM_THREADS (256u * VQM_TW)
#define VQM_TILE_BYTES ((VQM_BM + VQM_BN) * VQM_BK * 2u)   /* A 16 KB + B 16 KB, after the codebook */

/* Two E4M3 -> two bf16 (packed).  Hardware cvt to f16 first (E4M3 is a
 * subset of f16 normals, subnormals included), then shift the exponent bias:
 * a f16 normal shifted right 3 lands its exponent in the bf16 exponent bits
 * and its top 7 mantissa bits in bf16's (E4M3 only has 3 nonzero, the lost
 * ones are all 0), then add (127-15)<<7 = 0x3800 to rebias.  Zero must be
 * preserved separately (else it becomes 2^-15). */
__device__ __forceinline__ static uint32_t vqm_e4m3x2_to_bf16x2(uint32_t two) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((unsigned short)(two & 0xffffu)));
    const uint32_t mag = h & 0x7fff7fffu, nz = __vcmpne2(mag, 0u);
    return (h & 0x80008000u) | ((((mag >> 3) & 0x0fff0fffu) + 0x38003800u) & nz);
}
__device__ __forceinline__ static void vqm_ldsm4(uint32_t *r, const uint8_t *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ static void vqm_mma(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
/* A tile row is 64 bf16 = 8 chunks of 16 B; chunk ^ (row & 7) puts the 8 rows
 * ldmatrix reads at once into 8 different bank groups. */
__device__ __forceinline__ static uint32_t vqm_swz(uint32_t row, uint32_t chunk) { return row * 128u + ((chunk ^ (row & 7u)) << 4); }
/* One code word -> 8 bf16 (16 B): CB=0 converts the E4M3 codebook on the fly;
 * CB=1's codebook was converted to bf16 when it entered shared. */
template <int CB>
__device__ __forceinline__ static uint4 vqm_cw(const uint8_t *cbs, uint32_t v) {
    if (CB == 1) return *(const uint4 *)(cbs + (size_t)v * 16u);
    const uint2 cw = *(const uint2 *)(cbs + (size_t)v * 8u);
    uint4 o;
    o.x = vqm_e4m3x2_to_bf16x2(cw.x); o.y = vqm_e4m3x2_to_bf16x2(cw.x >> 16);
    o.z = vqm_e4m3x2_to_bf16x2(cw.y); o.w = vqm_e4m3x2_to_bf16x2(cw.y >> 16);
    return o;
}

/* Sorted activations, bf16.  The values already sit on the bf16 grid; bf16r
 * (round) not truncation, so a missed upstream round costs one correct
 * rounding, not a bias. */
__global__ static void vqm_gather16_kernel(uint16_t *xs, const float *x, const int32_t *perm, uint32_t K, uint32_t IN) {
    const uint32_t i = blockIdx.x, t = (uint32_t)perm[i] / K;
    const float *src = x + (uint64_t)t * IN;
    uint16_t *dst = xs + (uint64_t)i * IN;
    for (uint32_t d = threadIdx.x; d < IN; d += blockDim.x) dst[d] = (uint16_t)(__float_as_uint(v41_bf16r(src[d])) >> 16);
}

/* MODE 0 = gate: g32 = bf16(W1*x*g) | 1 = up: read g32, clamp + SwiGLU -> h16
 * | 2 = down: ys = bf16(W2*h*g).  MODE 1 with ys non-NULL also writes H_u
 * (the up exit before SwiGLU) — the backward pass needs it; the inference
 * path passes NULL.  The exit rounding point is vqp_fused_gu/down's
 * expression: sum -> row gain -> bf16r. */
template <int EXT, int MODE, int CB>
__global__ __launch_bounds__(VQM_THREADS, 1) static void vqm_kernel(
        float *g32, uint16_t *h16, float *ys, const uint8_t *blob, const vqp_item *items, uint32_t nitems,
        const uint16_t *act, const uint32_t *off, uint32_t M, uint32_t K, float clamp, uint32_t cb_bytes, const float *gr) {
    constexpr uint32_t CPR = VQM_BK / 8u, TPR = VQM_THREADS / VQM_BM, CPT = CPR / TPR;   /* 8 words per row-round, 4 threads x 2 */
    constexpr uint32_t NF = VQM_BN / 8u, NFW = NF / VQM_TW, CH = (VQM_BN * CPR) / VQM_THREADS;   /* 16 n8 slices, 8 per warp; 2 x 16 B of B per thread */
    extern __shared__ __align__(16) uint8_t vqmsh[];
    uint8_t *cbs = vqmsh, *As = vqmsh + cb_bytes, *Bs = As + VQM_BM * VQM_BK * 2u;
    const int which = MODE;   /* payload slot: 0 w1(gate) / 1 w3(up) / 2 w2(down) */
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, wr = warp & 7u, th = warp >> 3;
    const uint32_t ntile = (M + VQM_BM - 1u) / VQM_BM, nwork = nitems * ntile, nit = K / VQM_BK;
    {   /* the v3 codebook is one per layer (three matrices, all experts) => the first work item's is this layer's */
        const v41_vq_mat m0 = v41_vq_open<1>(blob, items[0].e, which, M, K, NULL);
        if (!m0.ok || m0.nc * (CB ? 16u : 8u) != cb_bytes) return;   /* one decision for the whole block, nobody stuck at a later barrier */
        if (CB == 0) v41_vq_cb_to_shared(cbs, m0.cb, cb_bytes);
        else for (uint32_t v = tid; v < m0.nc; v += VQM_THREADS) *(uint4 *)(cbs + (size_t)v * 16u) = vqm_cw<0>(m0.cb, v);
    }
    __syncthreads();
    const uint32_t ar = tid / TPR, sub = tid % TPR;   /* decode split: the thread owns row ar's code words sub*CPT.. */
    for (uint32_t w = blockIdx.x; w < nwork; w += gridDim.x) {
        const vqp_item it = items[w / ntile];
        const uint32_t r0 = (w % ntile) * VQM_BM, nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
        const v41_vq_mat m = v41_vq_open<1>(blob, it.e, which, M, K, (MODE == 2 && gr) ? gr + (size_t)it.e * M : NULL);
        if (!m.ok) {   /* the host walked the slot table already; reaching here means a bad payload; down writes 0, no dirty reduce input (same as the fused path) */
            if (MODE == 2)
                for (uint32_t i = tid; i < VQM_BM * nt; i += VQM_THREADS)
                    if (r0 + i % VQM_BM < M) ys[(uint64_t)(base + i / VQM_BM) * M + r0 + i % VQM_BM] = 0.f;
            continue;
        }
        const uint32_t grow = r0 + ar, mrow = m.nidx_row * 12u / 8u, erow = (m.nidx_row + 7u) >> 3;
        const bool rv = grow < M;
        const uint8_t *rowp = m.ix + (size_t)(rv ? grow : 0u) * mrow;
        const uint8_t *ep = EXT ? m.ex + (size_t)(rv ? grow : 0u) * erow : NULL;
        float acc[NFW][4];
        #pragma unroll
        for (uint32_t f = 0; f < NFW; f++) { acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.f; }
        uint32_t w0 = 0, w1 = 0, eb = 0;
        uint4 bx[CH];
        #pragma unroll
        for (uint32_t c = 0; c < CH; c++) bx[c] = make_uint4(0, 0, 0, 0);
        /* This thread's code words this round sit at bit offset round*96 + sub*24: two aligned words are read (the bit offset mod 32 is in {0,24,16,8}, all 24 bits inside the 64-bit window) and shifted. */
        auto load_round = [&](uint32_t i) {
            const uint32_t bitoff = i * 12u * CPR + sub * 12u * CPT, a = (bitoff >> 5) << 2;
            if (rv) { w0 = __ldg((const unsigned int *)(rowp + a)); w1 = __ldg((const unsigned int *)(rowp + a + 4u));
                      if (EXT) eb = __ldg(ep + i); }   /* the bit plane: 8 code words per round = 1 byte */
            #pragma unroll
            for (uint32_t c = 0; c < CH; c++) {
                const uint32_t ch = tid + c * VQM_THREADS, bt = ch / CPR, bc = ch % CPR;
                if (bt < nt) bx[c] = __ldg((const uint4 *)(act + (uint64_t)(base + bt) * K + (uint64_t)i * VQM_BK + bc * 8u));
            }
        };
        load_round(0);
        for (uint32_t i = 0; i < nit; i++) {
            const uint32_t sh = (i * 12u * CPR + sub * 12u * CPT) & 31u;
            const uint64_t u = (((uint64_t)w1 << 32) | w0) >> sh;
            #pragma unroll
            for (uint32_t q = 0; q < CPT; q++) {
                uint32_t v = (uint32_t)(u >> (12u * q)) & 0xFFFu;
                if (EXT) v |= ((eb >> (sub * CPT + q)) & 1u) << 12;   /* bit 13: byte i of the plane, bit (j&7) */
                *(uint4 *)(As + vqm_swz(ar, sub * CPT + q)) = vqm_cw<CB>(cbs, v);
            }
            #pragma unroll
            for (uint32_t c = 0; c < CH; c++) { const uint32_t ch = tid + c * VQM_THREADS; *(uint4 *)(Bs + vqm_swz(ch / CPR, ch % CPR)) = bx[c]; }
            __syncthreads();
            if (i + 1u < nit) load_round(i + 1u);   /* next round's bitstream/activations issue now, in flight during the mma */
            /* the warp's 16 rows x this half's n8 slices */
            #pragma unroll
            for (uint32_t kk = 0; kk < VQM_BK / 16u; kk++) {
                uint32_t a[4];
                {   const uint32_t mt = lane >> 3, row = wr * 16u + (lane & 7u) + (mt & 1u) * 8u;
                    vqm_ldsm4(a, As + vqm_swz(row, kk * 2u + (mt >> 1))); }
                #pragma unroll
                for (uint32_t np = 0; np < NFW / 2u; np++) {
                    const uint32_t f0 = th * NFW + 2u * np;   /* this warp's 2np-th slice among all */
                    if (f0 * 8u >= nt) break;
                    uint32_t b[4];
                    const uint32_t mt = lane >> 3, tok = (f0 + (mt >> 1)) * 8u + (lane & 7u);
                    vqm_ldsm4(b, Bs + vqm_swz(tok, kk * 2u + (mt & 1u)));
                    /* Every k16 starts from zero and adds back with a plain FADD (2026-09-24 measurement): accumulating K straight in the
                     * accumulator lets the tensor core's internal f32 addition drop low bits (truncation on alignment, not round-to-nearest) —
                     * 3.5% of outputs off by one bf16 ulp vs the fused path, relative L2 5e-4, and 2048-scale PPL 14.096 -> 14.192 (+0.68%,
                     * one-sided).  Hoisting the add back cuts it to 0.5% / 1.7e-4.  Same idea as the official FP8 GEMM's per-128 hoist. */
                    float t0[4] = {0.f, 0.f, 0.f, 0.f}, t1[4] = {0.f, 0.f, 0.f, 0.f};
                    vqm_mma(t0, a, b[0], b[1]);
                    #pragma unroll
                    for (int c = 0; c < 4; c++) acc[2 * np][c] += t0[c];
                    if ((f0 + 1u) * 8u < nt) { vqm_mma(t1, a, b[2], b[3]);
                        #pragma unroll
                        for (int c = 0; c < 4; c++) acc[2 * np + 1][c] += t1[c]; }
                }
            }
            __syncthreads();
        }
        /* exit: c0,c1 = row lane/4, token (lane&3)*2+{0,1}; c2,c3 = row +8 */
        const uint32_t rr[2] = { r0 + wr * 16u + (lane >> 2), r0 + wr * 16u + (lane >> 2) + 8u };
        #pragma unroll
        for (uint32_t h = 0; h < 2u; h++) {
            const uint32_t r = rr[h];
            if (r >= M) continue;
            __half gh; memcpy(&gh, m.gr + (size_t)r * 2u, 2);
            const float g = __half2float(gh) * (m.gov ? m.gov[r] : 1.0f);
            #pragma unroll
            for (uint32_t f = 0; f < NFW; f++) {
                const uint32_t gf = th * NFW + f;
                if (gf * 8u >= nt) break;
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const uint32_t t = gf * 8u + (lane & 3u) * 2u + e;
                    if (t >= nt) continue;
                    const uint64_t o = (uint64_t)(base + t) * M + r;
                    const float val = v41_bf16r(acc[f][h * 2u + e] * g);
                    if (MODE == 0) g32[o] = val;
                    else if (MODE == 1) { h16[o] = v41_vq_swiglu(g32[o], val, clamp); if (ys) ys[o] = val; }
                    else ys[o] = val;
                }
            }
        }
    }
}

static struct { v41_scratch d, doff, xs16, h16, g32; int ready, nsm; int occ[2][3]; } g_vqm;

/* Per-instance grid: SM count x the occupancy API's blocks/SM (never a fixed
 * tier; 96 KB shared gives 1 today). */
template <int EXT, int MODE, int CB>
static int vqm_grid_of(size_t shb, uint32_t nwork) {
    int *occ = &g_vqm.occ[EXT][MODE];
    if (*occ == 0) {
        if (cudaFuncSetAttribute(vqm_kernel<EXT, MODE, CB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(occ, vqm_kernel<EXT, MODE, CB>, (int)VQM_THREADS, shb) != cudaSuccess || *occ <= 0) {
            (void)cudaGetLastError(); *occ = -1;
        }
    }
    if (*occ < 0) return 0;
    const uint32_t g = (uint32_t)g_vqm.nsm * (uint32_t)*occ;
    return (int)(g < nwork ? g : nwork);
}

/* ---- vqs, the register-direct form (cuda_vq_reg_mma.inc.cu, whole file) ----
 * One block (16 warps) owns (work item, 256 rows) x all of K:
 *   1. bitstream and activations move in RS-round segments via cp.async, two
 *      segments in flight, one barrier per segment; each row's bitstream is
 *      additionally bulk-prefetched into L2 in 384 B (3 whole lines) chunks
 *      two chunks ahead — the segment cp.async then reads only L2 and DRAM
 *      sees whole-line sequential requests (measured 2026-09-18: a load that
 *      cannot fill a 128 B line gets 142 GB/s; cold experts sat at ~130 GB/s,
 *      this cut 8.6 -> 6.2 ms);
 *   2. lane (g = lane/4, q = lane%4) decodes rows 16w+g and 16w+8+g of the
 *      unit, and the codebook lookup (E4M3, 8 B/word) converts straight to
 *      bf16 — that IS the mma A fragment: no A/B tile writes, no ldmatrix.
 *      k16 logical k = 2q+{0,1} maps to physical column 16kk+4q+{0,1}, and
 *      2q+8+{0,1} to 16kk+4q+{2,3} => this lane needs code word 2kk+q/2's
 *      elements 4(q&1)..+3 (half a word, one LDS.32); the activation columns
 *      are one 8 B read.
 * Bit-exactness: the mma k16 groups are the same 16 columns, only who takes
 * which column inside the group changes; the per-k16 zero-start + FADD-back,
 * the row gain and the bf16r exit are unchanged — the engine's microbench
 * compared g32/h16/H_u/ys byte for byte with vqm_kernel, 12/13-bit x both
 * routings (a tensor core's intra-group sum is order-independent).
 * Wrong writes here do not fault: the mma eats other columns and PPL explodes
 * into the tens of thousands; the gate is the engine's warm-0 output cmp. */
#define VQS_THREADS 512u
#define VQS_BM      256u   /* rows per work unit: 16 warps x 16 rows (M must be a multiple: 2304 = 9*256, 5120 = 20*256) */
#define VQS_NTM     32u    /* tokens per work item (4 n8 slices); hot experts split by it, 64/128 measured no faster */

/* The kk-th k16 of a row's round needs code word c = 2kk + h (h = q/2), bit offset 12c; this round's 96 bits live in w0..w2. */
__device__ __forceinline__ static uint32_t vqs_code(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t kk, uint32_t h) {
    if (kk == 0u) return (w0 >> (12u * h)) & 0xFFFu;
    if (kk == 1u) return (h ? (w1 >> 4) : __funnelshift_r(w0, w1, 24u)) & 0xFFFu;
    if (kk == 2u) return __funnelshift_r(w1, w2, 16u + 12u * h) & 0xFFFu;
    return (w2 >> (8u + 12u * h)) & 0xFFFu;
}
template <uint32_t N>
__device__ __forceinline__ static void vqs_cp(uint8_t *dst, const void *src) {   /* N = 16 takes .cg (L2 only), 8/4 take .ca */
    if (N == 16u) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
    else asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "n"(N) : "memory");
}
__device__ __forceinline__ static void vqs_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ static void vqs_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
/* Portability deviation, named: the bulk prefetch is sm_90+ (ptxas rejects
 * it below); compiled out there.  It is a hint — the segment cp.async loads
 * are the real fetches — so the kernel stays correct and numerically
 * identical; sm_121 compiles the engine's asm verbatim. */
__device__ __forceinline__ static void vqs_pf_l2(const void *p, uint32_t bytes) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
#endif
}
__device__ __forceinline__ static uint32_t vqs_e4m3x2_bf16x2(uint32_t two) {
    uint32_t hh, o;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(hh) : "h"((unsigned short)(two & 0xffffu)));
    const float lo = __half2float(__ushort_as_half((unsigned short)(hh & 0xffffu))), hi = __half2float(__ushort_as_half((unsigned short)(hh >> 16)));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(o) : "f"(hi), "f"(lo));
    return o;
}
/* D = A*B (C is always +0, the output gets its own register group).  Not
 * vqm_mma(t, ...) with t = {0}: that is a "+f" read-write constraint, ptxas
 * schedules D into A's registers, and a reused A fragment costs 4 MOVs before
 * every HMMA (10-03 SASS: 32 HMMAs per segment paired with 128 MOVs).  Same
 * numerics (C = +0). */
__device__ __forceinline__ static void vqs_mma0(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f));
}
/* shared bytes: codebook (E4M3 nc*8) + S segments x (bitstream 256 rows x 12*RS B + plane 256 x 4 B + activations NTM x 128*RS B) */
static constexpr uint32_t vqs_smem(uint32_t cbb, int ext, uint32_t rs, uint32_t s) {
    return cbb + s * (VQS_BM * 12u * rs + (ext ? VQS_BM * 4u : 0u) + VQS_NTM * 128u * rs);
}

/* MODE 0 = gate: g32 = bf16(W1*x*g) | 1 = up: read g32, clamp + SwiGLU -> h16 (ys non-NULL also writes H_u) | 2 = down: ys = bf16(W2*h*g).
 * Parameters identical to vqm_kernel (same work items / offsets / scratch from vqm_run_impl); only the work-item cut (VQS_NTM) differs. */
template <int EXT, int MODE, uint32_t RS, uint32_t S>
__global__ __launch_bounds__(VQS_THREADS, 1) static void vqs_kernel(
        float *g32, uint16_t *h16, float *ys, const uint8_t *blob, const vqp_item *items, uint32_t nitems,
        const uint16_t *act, const uint32_t *off, uint32_t M, uint32_t K, float clamp, uint32_t cb_bytes, const float *gr) {
    constexpr uint32_t NTN = VQS_NTM / 8u, RB = 12u * RS, CP = RS == 4u ? 16u : 8u;   /* per row per segment, moved in 3 chunks of CP bytes */
    constexpr uint32_t BSB = VQS_BM * RB, EXB = EXT ? VQS_BM * 4u : 0u, ATB = VQS_NTM * 128u * RS, STB = BSB + EXB + ATB;
    constexpr uint32_t PCH = 384u, PD = 2u, SPC = PCH / RB;   /* L2 prefetch: 384 B per row chunk, 2 chunks ahead; SPC = chunks per segment */
    static_assert(RS == 2u || RS == 4u, "a segment is 2 or 4 rounds");
    extern __shared__ __align__(16) uint8_t vqssh[];
    uint8_t *cbs = vqssh, *ring = vqssh + cb_bytes;
    const int which = MODE;   /* payload slot: 0 w1(gate) / 1 w3(up) / 2 w2(down) */
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, q = lane & 3u, h = q >> 1, hf = q & 1u;
    {   /* the v3 codebook is one per layer (three matrices, all experts); enters shared as raw E4M3 */
        const v41_vq_mat m0 = v41_vq_open<1>(blob, items[0].e, which, M, K, NULL);
        if (!m0.ok || m0.nc * 8u != cb_bytes) return;   /* one decision for the whole block, nobody stuck at a later barrier */
        v41_vq_cb_to_shared(cbs, m0.cb, cb_bytes);
    }
    __syncthreads();
    const uint32_t ntile = M / VQS_BM, nwork = nitems * ntile, nst = K / (64u * RS);
    const uint32_t la = warp * 16u + g, lb = la + 8u;
    /* This thread moves the bitstream's two chunks (k = tid, tid+512; only k < 768 exist): chunk k%3 of row k/3 */
    const uint32_t k1 = tid + VQS_THREADS, rw0 = tid / 3u, pt0 = tid - rw0 * 3u, rw1 = k1 / 3u, pt1 = k1 - rw1 * 3u;
    const uint32_t cofs[4] = { ((0u + h) ^ (2u * (g & 3u))) << 4, ((2u + h) ^ (2u * (g & 3u))) << 4,
                               ((4u + h) ^ (2u * (g & 3u))) << 4, ((6u + h) ^ (2u * (g & 3u))) << 4 };
    for (uint32_t w = blockIdx.x; w < nwork; w += gridDim.x) {
        const vqp_item it = items[w / ntile];
        const uint32_t r0 = (w % ntile) * VQS_BM, nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
        const v41_vq_mat m = v41_vq_open<1>(blob, it.e, which, M, K, (MODE == 2 && gr) ? gr + (size_t)it.e * M : NULL);
        if (!m.ok) {   /* host walked the slot table; a bad payload zeroes the rows (same as vqm_kernel) */
            if (MODE == 2)
                for (uint32_t i = tid; i < VQS_BM * nt; i += VQS_THREADS) ys[(uint64_t)(base + i / VQS_BM) * M + r0 + i % VQS_BM] = 0.f;
            continue;
        }
        const uint32_t mrow = m.nidx_row * 12u / 8u, erow = (m.nidx_row + 7u) >> 3, nach = nt * 8u * RS;
        const uint8_t *prow = m.ix + (size_t)(r0 + (tid & 255u)) * mrow;
        auto pf_chunk = [&](uint32_t c) {
            if (tid < VQS_BM && c * PCH < mrow) vqs_pf_l2(prow + c * PCH, mrow - c * PCH < PCH ? mrow - c * PCH : PCH);
        };
        #pragma unroll
        for (uint32_t c = 0; c < PD; c++) pf_chunk(c);
        if (EXT && tid == 0) vqs_pf_l2(m.ex + (size_t)r0 * erow, (VQS_BM * erow) & ~15u);   /* the whole plane is 9-20 KB, one call */
        const uint8_t *bsrc0 = m.ix + (size_t)(r0 + rw0) * mrow + CP * pt0, *bsrc1 = m.ix + (size_t)(r0 + (k1 < 768u ? rw1 : 0u)) * mrow + CP * pt1;
        const uint8_t *esrc = EXT ? m.ex + (size_t)(r0 + (tid & 255u)) * erow : NULL;
        const uint16_t *xa = act + (uint64_t)base * K;
        auto issue = [&](uint32_t s) {   /* segment s (rounds RS*s ..) -> slot s % S */
            uint8_t *st = ring + (s % S) * STB;
            vqs_cp<CP>(st + rw0 * RB + CP * pt0, bsrc0 + RB * s);
            if (k1 < 768u) vqs_cp<CP>(st + rw1 * RB + CP * pt1, bsrc1 + RB * s);
            if (EXT && tid < VQS_BM) vqs_cp<4>(st + BSB + tid * 4u, esrc + 4u * ((RS * s) >> 2));   /* plane: one word per 4 rounds, RS=2 shares it between two segments */
            uint8_t *ad = st + BSB + EXB;
            for (uint32_t k = tid; k < nach; k += VQS_THREADS) {
                const uint32_t t = k / (8u * RS), rem = k - t * 8u * RS, r = rem >> 3, c = rem & 7u;
                vqs_cp<16>(ad + (r * VQS_NTM + t) * 128u + ((c ^ (2u * (t & 3u))) << 4), xa + (uint64_t)t * K + 64u * (RS * s + r) + 8u * c);
            }
        };
        /* Prologue: one group per segment 0..S-2.  Then one group per segment, so segment s sits in group s and wait_group(S-2) reaches it. */
        #pragma unroll
        for (uint32_t s = 0; s + 1u < S; s++) { if (s < nst) issue(s); vqs_commit(); }
        const uint32_t nnt = (nt + 7u) >> 3;
        float acc[NTN][4];
        #pragma unroll
        for (uint32_t j = 0; j < NTN; j++) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }
        /* The segment loop is instantiated per n8-slice count (a compile-time constant): no per-slice branch in the slice loop */
        auto run = [&](auto nntc) {
            constexpr uint32_t NNT = decltype(nntc)::value;
            for (uint32_t s = 0; s < nst; s++) {
                vqs_wait<(int)S - 2>();
                __syncthreads();   /* segment s is fully in; also proves everyone finished segment s-1's slot => segment s+S-1 can move in below */
                if (s + S - 1u < nst) issue(s + S - 1u);
                vqs_commit();
                if (s % SPC == 0u) pf_chunk(s / SPC + PD);
                const uint8_t *st = ring + (s % S) * STB;
                uint32_t wa[3 * RS], wb[3 * RS];
                if (RS == 4u) {
                    #pragma unroll
                    for (uint32_t k = 0; k < 3u; k++) {
                        const uint4 u = *(const uint4 *)(st + la * RB + 16u * k), v = *(const uint4 *)(st + lb * RB + 16u * k);
                        wa[4 * k] = u.x; wa[4 * k + 1] = u.y; wa[4 * k + 2] = u.z; wa[4 * k + 3] = u.w;
                        wb[4 * k] = v.x; wb[4 * k + 1] = v.y; wb[4 * k + 2] = v.z; wb[4 * k + 3] = v.w;
                    }
                } else {
                    #pragma unroll
                    for (uint32_t k = 0; k < 3u; k++) {
                        const uint2 u = *(const uint2 *)(st + la * RB + 8u * k), v = *(const uint2 *)(st + lb * RB + 8u * k);
                        wa[2 * k] = u.x; wa[2 * k + 1] = u.y; wb[2 * k] = v.x; wb[2 * k + 1] = v.y;
                    }
                }
                uint32_t xe = 0u, ye = 0u;
                if (EXT) { xe = *(const uint32_t *)(st + BSB + la * 4u); ye = *(const uint32_t *)(st + BSB + lb * 4u); }
                const uint8_t *as = st + BSB + EXB + 8u * hf + g * 128u;
                #pragma unroll
                for (uint32_t r = 0; r < RS; r++) {
                    const uint32_t eb = ((RS * s + r) & 3u) * 8u;   /* this round's byte inside the plane word */
                    #pragma unroll
                    for (uint32_t kk = 0; kk < 4u; kk++) {
                        uint32_t va = vqs_code(wa[3 * r], wa[3 * r + 1], wa[3 * r + 2], kk, h), vb = vqs_code(wb[3 * r], wb[3 * r + 1], wb[3 * r + 2], kk, h);
                        if (EXT) { const uint32_t c = eb + 2u * kk + h; va |= ((xe >> c) & 1u) << 12; vb |= ((ye >> c) & 1u) << 12; }
                        const uint32_t ea = *(const uint32_t *)(cbs + (size_t)va * 8u + hf * 4u), ebb = *(const uint32_t *)(cbs + (size_t)vb * 8u + hf * 4u);
                        const uint32_t a[4] = { vqs_e4m3x2_bf16x2(ea), vqs_e4m3x2_bf16x2(ebb), vqs_e4m3x2_bf16x2(ea >> 16), vqs_e4m3x2_bf16x2(ebb >> 16) };
                        const uint8_t *ap = as + r * VQS_NTM * 128u + cofs[kk];
                        #pragma unroll
                        for (uint32_t j = 0; j < NNT; j++) {
                            const uint2 x = *(const uint2 *)(ap + j * 8u * 128u);
                            /* Every k16 starts from zero and adds back with a plain FADD (same as vqm_kernel: a long in-core K accumulation truncates on alignment, a one-sided low-bit loss) */
                            float t[4];
                            vqs_mma0(t, a, x.x, x.y);
                            #pragma unroll
                            for (int c = 0; c < 4; c++) acc[j][c] += t[c];
                        }
                    }
                }
            }
        };
#define VQS_NNT(n) std::integral_constant<uint32_t, ((n) < NTN ? (n) : NTN)>{}
        switch (nnt) {
        case 1: run(VQS_NNT(1)); break;
        case 2: run(VQS_NNT(2)); break;
        case 3: run(VQS_NNT(3)); break;
        default: run(VQS_NNT(4)); break;
        }
#undef VQS_NNT
        __syncthreads();   /* the next unit's prologue overwrites these slots */
        /* exit: c0,c1 = row 16w+g, token 8j+2q+{0,1}; c2,c3 = row +8.  Rounding point same as vqm_kernel (sum -> row gain -> bf16r) */
        const uint32_t ra = r0 + la, rb = r0 + lb;
        float gna, gnb;
        { __half gh; memcpy(&gh, m.gr + (size_t)ra * 2u, 2); gna = __half2float(gh) * (m.gov ? m.gov[ra] : 1.0f);
          memcpy(&gh, m.gr + (size_t)rb * 2u, 2); gnb = __half2float(gh) * (m.gov ? m.gov[rb] : 1.0f); }
        #pragma unroll
        for (uint32_t j = 0; j < NTN; j++) {
            if (j >= nnt) break;
            #pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                const uint32_t t = 8u * j + 2u * q + e;
                if (t >= nt) continue;
                const uint64_t o = (uint64_t)(base + t) * M;
                const float va = v41_bf16r(acc[j][e] * gna), vb = v41_bf16r(acc[j][2u + e] * gnb);
                if (MODE == 0) { g32[o + ra] = va; g32[o + rb] = vb; }
                else if (MODE == 1) { h16[o + ra] = v41_vq_swiglu(g32[o + ra], va, clamp); h16[o + rb] = v41_vq_swiglu(g32[o + rb], vb, clamp);
                                      if (ys) { ys[o + ra] = va; ys[o + rb] = vb; } }
                else { ys[o + ra] = va; ys[o + rb] = vb; }
            }
        }
    }
}

static uint32_t vqs_item_tokens(void) { return VQS_NTM; }
/* vqs/vqst preconditions: bitstream 16 B aligned + device memory (cp.async's
 * 16 B copies and the L2 bulk prefetch both need it).  Only the boot-cached
 * device copy (the vq-align relocation) satisfies them; a flat copy of the
 * raw GGUF bytes is 8 B aligned.  Checked once per (layer, blob pointer). */
static struct { const uint8_t *blob; int ok; } g_vqs_fit[64];
static int vqs_blob_fits(uint32_t layer, const uint8_t *blob, uint32_t n_total, uint32_t IN, uint32_t MID, uint32_t OUT) {
    if (layer >= 64u) return 0;
    if (g_vqs_fit[layer].blob == blob) return g_vqs_fit[layer].ok;
    if (!g_vqp_hdr[layer] && !vqp_hdr_build(layer, blob, n_total, IN, MID, OUT, 3u)) return 0;
    int ok = ((uintptr_t)blob & 15u) == 0u;
    cudaPointerAttributes at;
    if (ok && (cudaPointerGetAttributes(&at, blob) != cudaSuccess || at.type != cudaMemoryTypeDevice)) { (void)cudaGetLastError(); ok = 0; }
    for (uint32_t k = 0; ok && k < n_total * 3u; k++) {   /* bitstream start = payload + 32 + row gains rows*2 */
        const uint64_t off = g_vqp_hdr[layer][k].off;
        if (off && ((off + 32u + (uint64_t)((k % 3u) == 2u ? OUT : MID) * 2u) & 15u)) ok = 0;
    }
    if (!ok) { static int said; if (!said++) fprintf(stderr, "ds4: [ds41] vq-prefill L%u expert blob is not an aligned device copy (mapping/flat copy), those layers take the tile kernel\n", layer); }
    g_vqs_fit[layer].blob = blob; g_vqs_fit[layer].ok = ok;
    return ok;
}
/* What vqs accepts: M (MID for gate/up, OUT for down) a multiple of 256, K a multiple of one segment (64*RS columns) */
static bool vqs_shape_ok(uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t nbit) {
    const uint32_t seg = 64u * (nbit == 13u ? 2u : 4u);
    return (nbit == 12u || nbit == 13u) && MID % VQS_BM == 0u && OUT % VQS_BM == 0u && IN % seg == 0u && MID % seg == 0u;
}
/* Three launches (gate -> up -> down).  13-bit layers: 64 KB codebook, shared fits one 2-round segment; 12-bit layers: 32 KB codebook, 4-round segments (half the barriers, 7% faster in the microbench). */
static int vqs_launch3(float *g32, uint16_t *h16, float *hu, float *ys, const uint8_t *blob, const vqp_item *items, uint32_t nitems,
                       const uint16_t *xs, const uint32_t *doff, uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t nc, uint32_t nbit,
                       float clamp, const float *gr, uint32_t layer_index) {
    static int occ[2][3];
    const uint32_t cbb = nc * 8u;
#define VQS_GO(E, RS_) do { \
        const uint32_t shb = vqs_smem(cbb, E, RS_, 2u); \
        int *oc = occ[E]; \
        if (!oc[0]) { \
            const bool ok = cudaFuncSetAttribute(vqs_kernel<E, 0, RS_, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb) == cudaSuccess && \
                            cudaFuncSetAttribute(vqs_kernel<E, 1, RS_, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb) == cudaSuccess && \
                            cudaFuncSetAttribute(vqs_kernel<E, 2, RS_, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb) == cudaSuccess && \
                            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&oc[0], vqs_kernel<E, 0, RS_, 2>, (int)VQS_THREADS, shb) == cudaSuccess && oc[0] > 0; \
            if (!ok) { (void)cudaGetLastError(); oc[0] = -1; } \
        } \
        if (oc[0] < 0) { fprintf(stderr, "ds4: [ds41] vq-prefill L%u register-direct instance cannot open %u KB shared\n", layer_index, shb >> 10); return 0; } \
        const uint32_t gcap = (uint32_t)g_vqm.nsm * (uint32_t)oc[0], ng = nitems * (MID / VQS_BM), nd = nitems * (OUT / VQS_BM); \
        vqs_kernel<E, 0, RS_, 2><<<ng < gcap ? ng : gcap, VQS_THREADS, shb, ds4_current_stream()>>>(g32, NULL, NULL, blob, items, nitems, xs, doff, MID, IN, clamp, cbb, NULL); \
        if (!cuda_ok(cudaGetLastError(), "vq prefill reg gate")) return 0; \
        vqs_kernel<E, 1, RS_, 2><<<ng < gcap ? ng : gcap, VQS_THREADS, shb, ds4_current_stream()>>>(g32, h16, hu, blob, items, nitems, xs, doff, MID, IN, clamp, cbb, NULL); \
        if (!cuda_ok(cudaGetLastError(), "vq prefill reg up")) return 0; \
        vqs_kernel<E, 2, RS_, 2><<<nd < gcap ? nd : gcap, VQS_THREADS, shb, ds4_current_stream()>>>(NULL, NULL, ys, blob, items, nitems, h16, doff, OUT, MID, clamp, cbb, gr); \
        return cuda_ok(cudaGetLastError(), "vq prefill reg down"); \
    } while (0)
    if (nbit == 13u) VQS_GO(1, 2u);
    VQS_GO(0, 4u);
#undef VQS_GO
}

/* ---- the run (cuda_vq_prefill_mma.inc.cu:233-302) ----
 * Returns 0 = failure (the caller hard-fails).  ys = sorted down output
 * [nvalid][OUT], the same buffer the fused path writes, so the reduce is
 * untouched.  hg/ha/hu non-NULL = per-pair intermediates into the caller's
 * buffers (backward capture); NULL = this file's scratch, ys only (the
 * inference path). */
static int vqm_run_impl(const uint8_t *blob, const uint32_t *cnt, const uint32_t *off_h, uint32_t n_total_expert, uint32_t nvalid,
                        uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t nc, float clamp, const float *x, const int32_t *perm,
                        uint32_t n_expert, uint32_t layer_index, float *ys, const float *gr, float *hg, uint16_t *ha, float *hu) {
    uint32_t nbit = 0; while ((1u << nbit) < nc) nbit++;
    if ((nbit != 12u && nbit != 13u) || IN % VQM_BK || MID % VQM_BK) {
        fprintf(stderr, "ds4: [ds41] vq-prefill L%u tensor-core path does not take this shape (codebook %u words, IN %u MID %u)\n", layer_index, nc, IN, MID);
        return 0;
    }
    /* 12-bit layers convert the codebook to bf16 on the way into shared
     * (nc*16 B = 64 KB); 13-bit layers' 8192 words would need 128 KB, so they
     * keep E4M3 (64 KB) and convert on the fly. */
    const int cb = nbit == 12u ? 1 : 0;
    const uint32_t cbb = nc * (cb ? 16u : 8u), shb = cbb + VQM_TILE_BYTES;
    if (!g_vqm.ready) {
        int nsm = 0;
        const bool ok = cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0) == cudaSuccess && nsm > 0;
        (void)cudaGetLastError();
        g_vqm.ready = ok ? 1 : -1; g_vqm.nsm = nsm;
        fprintf(stderr, "ds4: [ds41] vq-prefill tensor-core path (bf16 mma, <=%u tokens/item, %u threads) %s, %d SMs\n",
                VQM_BN, VQM_THREADS, ok ? "ready" : "unavailable", nsm);
    }
    if (g_vqm.ready != 1) return 0;
    const bool rs = vqs_shape_ok(IN, MID, OUT, nbit) && vqs_blob_fits(layer_index, blob, n_total_expert, IN, MID, OUT);
    const uint32_t bn = rs ? vqs_item_tokens() : VQM_BN;   /* tokens per work item: the two kernels cut differently */
    uint32_t nit = 0;
    for (uint32_t e = 0; e < n_total_expert; e++) nit += (cnt[e] + bn - 1u) / bn;
    if (!nit) return 0;
    vqp_item *ih = (vqp_item *)malloc((size_t)nit * sizeof(vqp_item));
    if (!ih) return 0;
    uint32_t k = 0;
    for (uint32_t e = 0; e < n_total_expert; e++)
        for (uint32_t t0 = 0; t0 < cnt[e]; t0 += bn) {
            ih[k].e = (int32_t)e; ih[k].t0 = (int32_t)t0; ih[k].nt = (int32_t)(cnt[e] - t0 < bn ? cnt[e] - t0 : bn); k++;
        }
    int ok = v41_grow(&g_vqm.d, (uint64_t)nit * sizeof(vqp_item), "vq prefill mma items") &&
             v41_grow(&g_vqm.doff, ((uint64_t)n_total_expert + 1u) * sizeof(uint32_t), "vq prefill mma off") &&
             v41_grow(&g_vqm.xs16, (uint64_t)nvalid * IN * sizeof(uint16_t), "vq prefill mma xs16") &&
             (ha || v41_grow(&g_vqm.h16, (uint64_t)nvalid * MID * sizeof(uint16_t), "vq prefill mma h16")) &&
             (hg || v41_grow(&g_vqm.g32, (uint64_t)nvalid * MID * sizeof(float), "vq prefill mma g32"));
    float *g32 = hg ? hg : (float *)g_vqm.g32.p;
    uint16_t *h16 = ha ? ha : (uint16_t *)g_vqm.h16.p;
    if (ok) ok = cudaMemcpyAsync(g_vqm.d.p, ih, (size_t)nit * sizeof(vqp_item), cudaMemcpyHostToDevice, ds4_current_stream()) == cudaSuccess &&
                 cudaMemcpyAsync(g_vqm.doff.p, off_h, (size_t)(n_total_expert + 1u) * sizeof(uint32_t), cudaMemcpyHostToDevice, ds4_current_stream()) == cudaSuccess;
    /* ih is the source of an async H2D from pageable memory => the runtime
     * staged it before returning, freeing here is safe (same as the fused path) */
    free(ih);
    if (!ok) { (void)cudaGetLastError(); fprintf(stderr, "ds4: [ds41] vq-prefill L%u tensor-core scratch/copy failed\n", layer_index); return 0; }
    vqm_gather16_kernel<<<nvalid, 256, 0, ds4_current_stream()>>>((uint16_t *)g_vqm.xs16.p, x, perm, n_expert, IN);
    if (!cuda_ok(cudaGetLastError(), "vq prefill mma gather16")) return 0;
    if (rs) return vqs_launch3(g32, h16, hu, ys, blob, (const vqp_item *)g_vqm.d.p, nit, (const uint16_t *)g_vqm.xs16.p, (const uint32_t *)g_vqm.doff.p, IN, MID, OUT, nc, nbit, clamp, gr, layer_index);
    const uint32_t tg = (MID + VQM_BM - 1u) / VQM_BM, td = (OUT + VQM_BM - 1u) / VQM_BM;
#define VQM_LAUNCH(E, CB) do { \
        const int bg = vqm_grid_of<E, 0, CB>(shb, nit * tg), bu = vqm_grid_of<E, 1, CB>(shb, nit * tg), bd = vqm_grid_of<E, 2, CB>(shb, nit * td); \
        if (bg <= 0 || bu <= 0 || bd <= 0) { fprintf(stderr, "ds4: [ds41] vq-prefill L%u tensor-core instance cannot open %u KB shared\n", layer_index, shb >> 10); return 0; } \
        vqm_kernel<E, 0, CB><<<bg, VQM_THREADS, shb, ds4_current_stream()>>>(g32, NULL, NULL, blob, (const vqp_item *)g_vqm.d.p, nit, (const uint16_t *)g_vqm.xs16.p, (const uint32_t *)g_vqm.doff.p, MID, IN, clamp, cbb, NULL); \
        if (!cuda_ok(cudaGetLastError(), "vq prefill mma gate")) return 0; \
        vqm_kernel<E, 1, CB><<<bu, VQM_THREADS, shb, ds4_current_stream()>>>(g32, h16, hu, blob, (const vqp_item *)g_vqm.d.p, nit, (const uint16_t *)g_vqm.xs16.p, (const uint32_t *)g_vqm.doff.p, MID, IN, clamp, cbb, NULL); \
        if (!cuda_ok(cudaGetLastError(), "vq prefill mma up")) return 0; \
        vqm_kernel<E, 2, CB><<<bd, VQM_THREADS, shb, ds4_current_stream()>>>(NULL, NULL, ys, blob, (const vqp_item *)g_vqm.d.p, nit, h16, (const uint32_t *)g_vqm.doff.p, OUT, MID, clamp, cbb, gr); \
        return cuda_ok(cudaGetLastError(), "vq prefill mma down"); \
    } while (0)
    if (nbit == 13u) VQM_LAUNCH(1, 0);
    VQM_LAUNCH(0, 1);
#undef VQM_LAUNCH
}

/* Inference entry (prefill / scoring): ys only.  The engine's backward
 * capture (g_vqm_cap / vqm_run_parts / vqm_run's H_u hoist) is training-only
 * and not ported — the port has no backward pass. */
static int vqm_run(const uint8_t *blob, const uint32_t *cnt, const uint32_t *off_h, uint32_t n_total_expert, uint32_t nvalid,
                   uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t nc, float clamp, const float *x, const int32_t *perm,
                   uint32_t n_expert, uint32_t layer_index, float *ys, const float *gr) {
    return vqm_run_impl(blob, cnt, off_h, n_total_expert, nvalid, IN, MID, OUT, nc, clamp, x, perm, n_expert, layer_index, ys, gr, NULL, NULL, NULL);
}
