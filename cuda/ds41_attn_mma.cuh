/* ds41_attn_mma.cuh — DeepSeek V4.1 (ds41) sparse-attention tensor-core
 * family, ported from the C engine at /data/YoungAi (commit 3946dbc):
 *   - cuda_sparse_attn_mma.inc.cu: the gather/scores/stats building blocks
 *     (:42-120) and the single-pass prefill mma kernel (:126-318, the
 *     2026-09-29 form with the 520-element key stride);
 *   - cuda_v41_attn_split.inc.cu: the seg-length formulas (:146-171), the
 *     merge kernel (:184-199) and the scalar split-K decode (:50-118, :203-230);
 *   - cuda_v41_attn_mma_decode.inc.cu: the two-pass decode seg kernel
 *     (:37-132) and the v41_attn_mma_decode host dispatcher (:150-210);
 *   - the entry ds4_gpu_v41_sparse_attn_tensor (cuda_v41_2.inc.cu:229-303),
 *     now here so the dispatch can mirror the engine's try order.
 *
 * Numerics (the engine's own header, cuda_v41_attn_mma_decode.inc.cu:18):
 * this family is NOT bit-equal to the scalar kernel — tensor-core
 * accumulation order and the online-softmax grouping both differ; the gate
 * is the quality scale (NLL/PPL), not cmp.  It IS the engine's live decode
 * path at every key count (the nkeys<64 split fallback was removed
 * 2026-09-16, M1' in the file), so the port's scalar-only deviation (P4-4)
 * ends here for n<=8 and n>=64; the scalar kernel stays as the engine's own
 * fallback for the 9..63-token band and for shapes the family refuses.
 *
 * Try order mirrored from cuda_v41_2.inc.cu:229-303: mma decode first for
 * !full_block (any n<=8), then the draft-block arm (full_block, linear ring,
 * no compressed keys), then the scalar split-K (n==1), then the prefill mma
 * (n>=64), then the scalar kernel.
 *
 * Invalid topk slots (idx<0) get BOTH a zeroed key row and a -inf score —
 * the shared gather does both (cuda_sparse_attn_mma.inc.cu:25-27: 0*NaN is
 * NaN, and residue read as float can be NaN/Inf).  Do not write a second
 * gather here; a second copy is how the two paths drift apart. */
#pragma once

/* ---- constants (cuda_sparse_attn_mma.inc.cu:29-32, cuda_v41_attn_split.inc.cu:31-47) ---- */
#define DS4_ATTN_MMA_HEADS 16u   /* heads per block = one wmma M tile */
#define DS4_ATTN_MMA_KT    16u   /* keys per shared tile = one wmma N tile */
#define DS4_ATTN_MMA_WARPS 8u    /* warps per block: pass 1/2 split K, the O step splits the output dim */
#define DS4_ATTN_MMA_HD    512u  /* head dim, bound to the v41 shape (the entry checks it) */
#define DS4_ATTN_FA_LD 520u      /* prefill kernel key-tile stride in bf16 elems (1024 B stride = 8-way bank conflict, cuda_sparse_attn_mma.inc.cu:129) */
#define V41_ATTN_MMA_DEC_TARGET_SEG 24u   /* decode target segment count (cuda_v41_attn_split.inc.cu:141) */
#define V41_ATTN_SPLIT_MIN_KEYS 16u       /* below this the split is pure merge overhead (cuda_v41_attn_split.inc.cu:31) */
#define V41_ATTN_SPLIT_SEG_KEYS 64u       /* scalar split max keys per segment */
#define V41_ATTN_SPLIT_SEG_MIN  8u        /* scalar split min = KTILE */
#define V41_ATTN_SPLIT_TARGET_SEG 6u      /* scalar split target segments (36 was measured NEGATIVE, do not raise) */
#define V41_ATTN_SPLIT_MAX_SEG  64u       /* segment cap: scratch = 64 * 64 heads * 512 * 4 B = 8 MB */

/* ---- building block 1: gather 16 key rows into shared, bf16
 * (cuda_sparse_attn_mma.inc.cu:42-73).  Window rows first in ascending
 * order, then the topk compressed rows — the scalar kernel's key order
 * exactly.  The compressed unpack decodes 32 scales one per lane and the 16
 * fp4 values one per lane, then resolves each element with two shuffles
 * (the 2026-09-29 form; per-element v41_ckv_get was 2.4 -> 1.4 ms/step at
 * 12k).  Same scale, same nibble, same multiply -> bit-equal to v41_ckv_get. */
__device__ __forceinline__ static void ds4_attn_mma_gather_keys(
        __nv_bfloat16 *ks, int *valid, const float *kvw, const uint8_t *kvc, const int32_t *idx,
        uint32_t i, uint32_t base, uint32_t nt, uint32_t nwin, uint32_t lo, uint32_t pos0,
        uint32_t window, uint32_t ng, uint32_t topk, uint32_t ring) {
    const uint32_t lane = threadIdx.x & 31u;
    const float tv = ds4_fp4_nibble_to_f32((uint8_t)(lane & 15u));   /* fp4 value table: lane l holds entry l&15 */
    for (uint32_t t = threadIdx.x / 32u; t < DS4_ATTN_MMA_KT; t += blockDim.x / 32u) {
        const uint32_t kk = base + t;
        const float *krow = NULL; const uint8_t *cpk = NULL;   /* window row f32 / compressed row packed FP4 */
        if (t < nt) {
            /* ring: main-path history is the ring (1); the DSpark tower's window is a linear segment (0, cuda_v41_attn_mma_decode.inc.cu:57) */
            if (kk < nwin) krow = kvw + v41_win_row((int64_t)lo + kk, pos0, window, ring) * DS4_ATTN_MMA_HD;
            else if (kvc && idx) { const int32_t g = idx[(uint64_t)i * topk + (kk - nwin)];
                                   if (g >= 0 && (uint32_t)g < ng) cpk = kvc + (uint64_t)g * DS4_V41_CKV_BYTES; }
        }
        if (lane == 0) valid[t] = (krow || cpk) ? 1 : 0;
        __nv_bfloat16 *kt = ks + (size_t)t * DS4_ATTN_MMA_HD;
        if (krow) { for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u) kt[d] = __float2bfloat16(krow[d]); }
        else if (cpk) {
            const float sc = ds4_e4m3fn_to_f32(cpk[DS4_V41_CKV_NIB + lane]);   /* 32 scales for 512 elems: one per lane */
            #pragma unroll
            for (uint32_t j = 0; j < DS4_ATTN_MMA_HD / 32u; j++) {
                const uint32_t d = lane + 32u * j;
                const uint8_t by = cpk[d >> 1];
                const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);   /* same nibble take as v41_ckv_get */
                const float s = __shfl_sync(0xffffffffu, sc, (int)(d >> 4));
                kt[d] = __float2bfloat16(__shfl_sync(0xffffffffu, tv, (int)nib) * s);
            }
        } else { for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u) kt[d] = (__nv_bfloat16)0.0f; }
    }
}

/* ---- building block 2: the S tile [16 heads][16 keys] (cuda_sparse_attn_mma.inc.cu:76-101).
 * 8 warps each take 1/8 of the K dim (4 wmma k-steps); partials land in
 * shared and are added in FIXED order — same input, two runs, same result
 * (the 2026-09-15 conviction).  Invalid slots read -1e30 so exp is 0. ---- */
__device__ __forceinline__ static void ds4_attn_mma_scores(
        float *stile, float *spart, const __nv_bfloat16 *qs, const __nv_bfloat16 *ks,
        const int *valid, uint32_t nt, float scale) {
    namespace wmma = nvcuda::wmma;
    const uint32_t warp = threadIdx.x >> 5;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;   /* B[d][key] = ks[key][d] => col-major, ldm=512 */
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    const uint32_t ksteps = DS4_ATTN_MMA_HD / 16u / DS4_ATTN_MMA_WARPS;   /* 512/16/8 = 4 */
    for (uint32_t s = 0; s < ksteps; s++) {
        const uint32_t d0 = (warp * ksteps + s) * 16u;
        wmma::load_matrix_sync(a, qs + d0, DS4_ATTN_MMA_HD);
        wmma::load_matrix_sync(b, ks + d0, DS4_ATTN_MMA_HD);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(spart + (size_t)warp * 256u, c, 16, wmma::mem_row_major);
    __syncthreads();
    for (uint32_t e = threadIdx.x; e < 256u; e += blockDim.x) {
        float v = 0.f;
        for (uint32_t w = 0; w < DS4_ATTN_MMA_WARPS; w++) v += spart[(size_t)w * 256u + e];
        const uint32_t k = e & 15u;   /* stile is row-major [head][key] */
        stile[e] = (k < nt && valid[k]) ? v * scale : -1e30f;
    }
    __syncthreads();
}

/* ---- building block 3: the online max/sum over one tile (cuda_sparse_attn_mma.inc.cu:105-120).
 * Parallel form (2026-09-29): warp w owns heads 2w (lanes 0..15) and 2w+1
 * (lanes 16..31), one key per lane.  Bit-equal to the serial form: max is
 * exact (order-free); the sum is rsum*expf(m-tm) then expf(s_k-tm) added in
 * k = 0..15 order — same order, same value.  ALL threads must call it (the
 * warp shuffles are collective), never wrap it in threadIdx.x < 16. ---- */
__device__ __forceinline__ static void ds4_attn_mma_stats(const float *stile, float *rmax, float *rsum) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, k = lane & 15u;
    const uint32_t h = warp * 2u + (lane >> 4);
    const float s = stile[h * 16u + k], m = rmax[h];
    float tm = fmaxf(m, s);
    tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 8)); tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 4));
    tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 2)); tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 1));
    const float e = expf(s - tm);
    float sm = rsum[h] * expf(m - tm);
    const int b = (int)(lane & 16u);
    #pragma unroll
    for (int kk = 0; kk < 16; kk++) sm += __shfl_sync(0xffffffffu, e, b + kk);
    if (k == 0u) { rmax[h] = tm; rsum[h] = sm; }
}

/* ---- the single-pass prefill mma kernel (cuda_sparse_attn_mma.inc.cu:126-282).
 * Four changes vs the two-pass form: q stays in registers (A fragments),
 * S via mma.m16n8k16 (not wmma), single pass (O scaled by alpha and P*V
 * accumulated in the same walk; the accumulator layout is documented, so
 * per-head scaling is reachable), key stride 520 bf16 (bank conflicts).
 * Not bit-equal to the two-pass form; the gate is five-metric/PPL. ---- */
__device__ __forceinline__ static void ds4_fa_ldsm4(uint32_t *r, const void *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ static void ds4_fa_ldsm4t(uint32_t *r, const void *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ static void ds4_fa_mma(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ static uint32_t ds4_fa_bf16x2(float lo, float hi) {
    return (__float_as_uint(v41_bf16r(lo)) >> 16) | (__float_as_uint(v41_bf16r(hi)) & 0xffff0000u);
}
/* gather: same decode as ds4_attn_mma_gather_keys (bit-equal values), only the row stride is DS4_ATTN_FA_LD */
__device__ __forceinline__ static void ds4_fa_gather(__nv_bfloat16 *ks, int *valid, const float *kvw, const uint8_t *kvc, const int32_t *idx,
                                                     uint32_t i, uint32_t base, uint32_t nt, uint32_t nwin, uint32_t lo, uint32_t pos0,
                                                     uint32_t window, uint32_t ng, uint32_t topk) {
    const uint32_t lane = threadIdx.x & 31u;
    const float tv = ds4_fp4_nibble_to_f32((uint8_t)(lane & 15u));
    for (uint32_t t = threadIdx.x / 32u; t < DS4_ATTN_MMA_KT; t += blockDim.x / 32u) {
        const uint32_t kk = base + t;
        const float *krow = NULL; const uint8_t *cpk = NULL;
        if (t < nt) {
            if (kk < nwin) krow = kvw + v41_win_row((int64_t)lo + kk, pos0, window, 1u) * DS4_ATTN_MMA_HD;
            else if (kvc && idx) { const int32_t g = idx[(uint64_t)i * topk + (kk - nwin)];
                                   if (g >= 0 && (uint32_t)g < ng) cpk = kvc + (uint64_t)g * DS4_V41_CKV_BYTES; }
        }
        if (lane == 0) valid[t] = (krow || cpk) ? 1 : 0;
        __nv_bfloat16 *kt = ks + (size_t)t * DS4_ATTN_FA_LD;
        if (krow) { for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u) kt[d] = __float2bfloat16(krow[d]); }
        else if (cpk) {
            const float sc = ds4_e4m3fn_to_f32(cpk[DS4_V41_CKV_NIB + lane]);
            #pragma unroll
            for (uint32_t j = 0; j < DS4_ATTN_MMA_HD / 32u; j++) {
                const uint32_t d = lane + 32u * j;
                const uint8_t by = cpk[d >> 1];
                const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
                const float s = __shfl_sync(0xffffffffu, sc, (int)(d >> 4));
                kt[d] = __float2bfloat16(__shfl_sync(0xffffffffu, tv, (int)nib) * s);
            }
        } else { for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u) kt[d] = (__nv_bfloat16)0.0f; }
    }
}
/* shared: ks 16x520x2 + spart 8x16x16x4 + stile 16x16x4 + ptile 16x16x2 + rmax/rsum/alpha 3x16x4 + valid 16x4 ~= 26 KB => 3 blocks per SM */
__global__ __launch_bounds__(256) static void ds4_sparse_attn_mma_kernel(float *o, const float *q, const float *kvw, const uint8_t *kvc,
                                                  const int32_t *idx, const float *sink, uint32_t pos0, uint32_t window,
                                                  uint32_t ng, uint32_t topk, uint32_t n_head, float scale, uint32_t win_lo) {
    constexpr uint32_t KT = DS4_ATTN_MMA_KT, HB = DS4_ATTN_MMA_HEADS;
    extern __shared__ __align__(16) char ds4_attn_fa_smem[];
    __nv_bfloat16 *ks = (__nv_bfloat16 *)ds4_attn_fa_smem;                        /* [16][520] */
    float *spart = (float *)(ks + (size_t)KT * DS4_ATTN_FA_LD);                     /* [8 warp][16][16] */
    float *stile = spart + 8u * HB * KT;                                            /* [16][16] */
    __nv_bfloat16 *ptile = (__nv_bfloat16 *)(stile + HB * KT);                      /* [16][16] */
    float *rmax = (float *)(ptile + HB * KT), *rsum = rmax + HB, *alpha = rsum + HB;
    int *valid = (int *)(alpha + HB);
    const uint32_t i = blockIdx.x, h0 = blockIdx.y * HB, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t p = pos0 + i;
    uint32_t lo = p + 1u > window ? p + 1u - window : 0u;
    if (lo < win_lo) lo = win_lo;   /* ring slots before win_lo were never written (CED), do not read (official -1 mask) */
    const uint32_t nwin = p - lo + 1u, nkeys = nwin + topk;
    /* q's A fragments live in registers: warp w covers dims [64w, 64w+64) as 4 k16 steps (a0: row r col c..c+1; a1: row r+8; a2: col +8; a3: row +8 col +8) */
    uint32_t qa[4][4];
    {
        const uint32_t r = lane >> 2, c = (lane & 3u) * 2u;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; s++) {
            const float *q0 = q + ((uint64_t)i * n_head + h0 + r) * DS4_ATTN_MMA_HD + warp * 64u + s * 16u + c;
            const float *q1 = q0 + 8u * DS4_ATTN_MMA_HD;
            qa[s][0] = ds4_fa_bf16x2(q0[0], q0[1]); qa[s][1] = ds4_fa_bf16x2(q1[0], q1[1]);
            qa[s][2] = ds4_fa_bf16x2(q0[8], q0[9]); qa[s][3] = ds4_fa_bf16x2(q1[8], q1[9]);
        }
    }
    if (threadIdx.x < HB) { rmax[threadIdx.x] = -1e30f; rsum[threadIdx.x] = 0.f; }
    float oacc[8][4];   /* O[16 heads][this warp's 64 dims] = 8 n8 slices */
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) { oacc[j][0] = oacc[j][1] = oacc[j][2] = oacc[j][3] = 0.f; }
    __syncthreads();
    for (uint32_t base = 0; base < nkeys; base += KT) {
        const uint32_t nt = (nkeys - base) < KT ? (nkeys - base) : KT;
        __syncthreads();   /* previous tile's ks/ptile fully consumed */
        ds4_fa_gather(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
        __syncthreads();
        {   /* (1) S partials: this warp's 64 dims x 16 heads x 16 keys (two n8 slices) */
            float sp[2][4] = { {0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f} };
            #pragma unroll
            for (uint32_t s = 0; s < 4u; s++) {
                uint32_t b[4];
                const uint32_t mt = lane >> 3, key = (mt >> 1) * 8u + (lane & 7u), col = warp * 64u + s * 16u + (mt & 1u) * 8u;
                ds4_fa_ldsm4(b, ks + (size_t)key * DS4_ATTN_FA_LD + col);
                ds4_fa_mma(sp[0], qa[s], b[0], b[1]); ds4_fa_mma(sp[1], qa[s], b[2], b[3]);
            }
            #pragma unroll
            for (uint32_t nb = 0; nb < 2u; nb++) {   /* c0,c1 = row lane/4, cols (lane&3)*2+{0,1}; c2,c3 = row +8 */
                float *sw = spart + (size_t)warp * HB * KT + (lane >> 2) * KT + nb * 8u + (lane & 3u) * 2u;
                sw[0] = sp[nb][0]; sw[1] = sp[nb][1]; sw[8u * KT] = sp[nb][2]; sw[8u * KT + 1u] = sp[nb][3];
            }
        }
        __syncthreads();
        /* (2) fixed-order sum + scale + mask (invalid slots -1e30 => exp 0) */
        {
            const uint32_t e = threadIdx.x, k = e % KT;
            float v = 0.f;
            #pragma unroll
            for (uint32_t w = 0; w < 8u; w++) v += spart[(size_t)w * HB * KT + e];
            stile[e] = (k < nt && valid[k]) ? v * scale : -1e30f;
        }
        __syncthreads();
        /* (3) online max/sum + P tile (bf16, official acc_s_cast) + alpha: one thread per head, key order serial (the two-pass serial order) */
        if (threadIdx.x < HB) {
            const uint32_t h = threadIdx.x; const float *sr = stile + h * KT; const float m = rmax[h];
            float tm = m;
            for (uint32_t k = 0; k < KT; k++) tm = fmaxf(tm, sr[k]);
            const float al = expf(m - tm);
            float sm = rsum[h] * al;
            for (uint32_t k = 0; k < KT; k++) { const float pv = expf(sr[k] - tm); sm += pv; ptile[h * KT + k] = __float2bfloat16(pv); }
            rmax[h] = tm; rsum[h] = sm; alpha[h] = al;
        }
        __syncthreads();
        /* (4) O = O*alpha + P*V: A = ptile [16 heads][16 keys], B = ks [keys][dim] via .trans; this warp covers dims [64w, 64w+64) as 8 n8 slices */
        {
            const float a0 = alpha[lane >> 2], a1 = alpha[(lane >> 2) + 8u];
            #pragma unroll
            for (uint32_t j = 0; j < 8u; j++) { oacc[j][0] *= a0; oacc[j][1] *= a0; oacc[j][2] *= a1; oacc[j][3] *= a1; }
            uint32_t pa[4];
            {   const uint32_t mt = lane >> 3, row = (lane & 7u) + (mt & 1u) * 8u, col = (mt >> 1) * 8u;
                ds4_fa_ldsm4(pa, ptile + (size_t)row * KT + col); }
            #pragma unroll
            for (uint32_t j = 0; j < 8u; j += 2u) {
                uint32_t b[4];
                const uint32_t mt = lane >> 3, key = (mt & 1u) * 8u + (lane & 7u), col = warp * 64u + j * 8u + (mt >> 1) * 8u;
                ds4_fa_ldsm4t(b, ks + (size_t)key * DS4_ATTN_FA_LD + col);
                ds4_fa_mma(oacc[j], pa, b[0], b[1]); ds4_fa_mma(oacc[j + 1], pa, b[2], b[3]);
            }
        }
    }
    __syncthreads();
    /* exit: divide by (sum + exp(sink - max)) (sink enters the denominator only), round bf16, write */
    {
        const uint32_t hA = lane >> 2, hB = hA + 8u;
        const float dA = rsum[hA] + expf(sink[h0 + hA] - rmax[hA]), dB = rsum[hB] + expf(sink[h0 + hB] - rmax[hB]);
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t d = warp * 64u + j * 8u + (lane & 3u) * 2u;
            float *oA = o + ((uint64_t)i * n_head + h0 + hA) * DS4_ATTN_MMA_HD + d, *oB = o + ((uint64_t)i * n_head + h0 + hB) * DS4_ATTN_MMA_HD + d;
            oA[0] = v41_bf16r(oacc[j][0] / dA); oA[1] = v41_bf16r(oacc[j][1] / dA);
            oB[0] = v41_bf16r(oacc[j][2] / dB); oB[1] = v41_bf16r(oacc[j][3] / dB);
        }
    }
}

/* prefill kernel shared size (~26 KB); the decode seg kernel keeps its own layout below */
static size_t ds4_attn_mma_smem_bytes(void) {
    return (size_t)DS4_ATTN_MMA_KT * DS4_ATTN_FA_LD * sizeof(__nv_bfloat16)
         + (size_t)DS4_ATTN_MMA_WARPS * DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_KT * sizeof(float)
         + (size_t)DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_KT * (sizeof(float) + sizeof(__nv_bfloat16))
         + 3u * DS4_ATTN_MMA_HEADS * sizeof(float) + DS4_ATTN_MMA_KT * sizeof(int);
}
static size_t ds4_attn_mma_seg_smem_bytes(void) {
    return (size_t)(DS4_ATTN_MMA_HEADS + DS4_ATTN_MMA_KT) * DS4_ATTN_MMA_HD * sizeof(__nv_bfloat16)
         + (size_t)DS4_ATTN_MMA_WARPS * 256u * sizeof(float) + 256u * sizeof(float)
         + 256u * sizeof(__nv_bfloat16) + 2u * DS4_ATTN_MMA_HEADS * sizeof(float)
         + DS4_ATTN_MMA_KT * sizeof(int);
}

/* Usable? Shape (64 heads x 512, head count divisible by 16) + big enough
 * (at small n_tok the grid is n_tok x 4 blocks, 48 SMs stay idle and the
 * scalar path wins) + shared can be raised.  0 = caller falls back. */
static int ds4_sparse_attn_mma_launch(float *o, const float *q, const float *kvw, const uint8_t *kvc,
                                      const int32_t *idx, const float *sink, uint32_t n_tok, uint32_t pos0,
                                      uint32_t window, uint32_t ng, uint32_t topk, uint32_t n_head,
                                      uint32_t head_dim, float scale, uint32_t win_lo) {
    if (head_dim != DS4_ATTN_MMA_HD || (n_head % DS4_ATTN_MMA_HEADS) || n_tok < 64u) return 0;
    static int s_ok = 0;   /* 0 untried / 1 usable / -1 shared cannot be raised */
    const size_t smem = ds4_attn_mma_smem_bytes();
    if (s_ok == 0) {
        (void)cudaGetLastError();   /* the opt-in must not see a previously latched error (see ds41_vq_decode.cuh) */
        s_ok = cudaFuncSetAttribute(ds4_sparse_attn_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem) == cudaSuccess ? 1 : -1;
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [attn] tensor-core prefill shared %zu KB %s\n", smem >> 10, s_ok == 1 ? "enabled" : "*cannot be raised, back to scalar*");
    }
    if (s_ok != 1) return 0;
    ds4_sparse_attn_mma_kernel<<<dim3(n_tok, n_head / DS4_ATTN_MMA_HEADS), DS4_ATTN_MMA_WARPS * 32u, smem, ds4_current_stream()>>>(
        o, q, kvw, kvc, idx, sink, pos0, window, ng, topk, n_head, scale, win_lo);
    return cuda_ok(cudaGetLastError(), "sparse attn mma");
}

/* ---- seg-length formulas (cuda_v41_attn_split.inc.cu:121-171).
 * The segment length may only depend on a batch-size-independent reference
 * key count (2026-09-16 M1'): the same position in pure decode (n=1) and in
 * a spec-verify batch (n=4) must get the same grouping, or the two paths
 * diverge at temperature 0.  Feeding the formula each query's OWN absolute
 * position is the fix; "first query of the batch" was measured WRONG
 * (2K, byte 270 diverged). ---- */
__host__ __device__ __forceinline__ static uint32_t v41_attn_seg_keys(uint32_t p, uint32_t window,
                                                                     uint32_t ratio, uint32_t topk) {
    const uint32_t nwin = p + 1u > window ? window : p + 1u;
    uint32_t tref = 0;
    if (ratio) { const uint32_t vis = (p + 1u) / ratio; tref = topk < vis ? topk : vis; }
    uint32_t seg = (nwin + tref + V41_ATTN_MMA_DEC_TARGET_SEG - 1u) / V41_ATTN_MMA_DEC_TARGET_SEG;
    seg = ((seg + DS4_ATTN_MMA_KT - 1u) / DS4_ATTN_MMA_KT) * DS4_ATTN_MMA_KT;
    if (seg < DS4_ATTN_MMA_KT) seg = DS4_ATTN_MMA_KT;
    return seg > 64u ? 64u : seg;
}
/* full-visible block (DSpark draft tower, 2026-09-29): every in-block
 * position sees one key set (whole window + all n in-block positions, no
 * compressed keys); the length depends on the key count only.  Drafts only
 * propose, verification decides, so this path is bound by determinism, not
 * by the decode-==-verify byte gate. */
__host__ __device__ __forceinline__ static uint32_t v41_attn_fb_seg_keys(uint32_t nkeys) {
    uint32_t seg = (nkeys + V41_ATTN_MMA_DEC_TARGET_SEG - 1u) / V41_ATTN_MMA_DEC_TARGET_SEG;
    seg = ((seg + DS4_ATTN_MMA_KT - 1u) / DS4_ATTN_MMA_KT) * DS4_ATTN_MMA_KT;
    if (seg < DS4_ATTN_MMA_KT) seg = DS4_ATTN_MMA_KT;
    return seg > 64u ? 64u : seg;
}
/* segments at position p (= the eager path's grid; the graph path's merge
 * recomputes the same number from the device position) */
__host__ __device__ __forceinline__ static uint32_t v41_attn_nseg_at(uint32_t p, uint32_t window, uint32_t ratio, uint32_t topk) {
    const uint32_t nw = p + 1u > window ? window : p + 1u;
    uint32_t tk = topk;
    if (ratio) { const uint32_t vis = (p + 1u) / ratio; if (vis < tk) tk = vis; }
    const uint32_t sk = v41_attn_seg_keys(p, window, ratio, topk);
    return (nw + tk + sk - 1u) / sk;
}

/* ---- the merge (cuda_v41_attn_split.inc.cu:184-199): one block per head,
 * segments merged in FIXED segment order (atomic add would make two runs of
 * one input differ), sink into the denominator, divide, round bf16.
 * Local tiles are laid out [(token*nseg + seg)*head + h]; n_tok=1 is the
 * original [seg*head + h] untouched.  The posd arm re-derives the true
 * segment count from the device position and reads only real segments —
 * bit-equal to the eager path (extra segments never existed there). ---- */
__global__ static void v41_sparse_attn_merge_kernel(float *o, const float *pacc, const float *pmax, const float *psum,
                                                    const float *sink, uint32_t nseg, uint32_t n_head, uint32_t hd,
                                                    const int32_t *posd, uint32_t window, uint32_t ratio, uint32_t topk) {
    v41_pdl_wait();   /* PDL: wait for the upstream as the first instruction */
    const uint32_t h = blockIdx.x, i = blockIdx.y;
    const uint64_t b0 = (uint64_t)i * nseg * n_head;   /* tile row stride follows the grid nseg (the cap) */
    if (posd) nseg = v41_attn_nseg_at((uint32_t)posd[0] + i, window, ratio, topk);   /* row i's position = pos0 + i */
    float m = -1e30f;
    for (uint32_t s = 0; s < nseg; s++) m = fmaxf(m, pmax[b0 + (uint64_t)s * n_head + h]);
    float den = 0.f;
    for (uint32_t s = 0; s < nseg; s++) den += psum[b0 + (uint64_t)s * n_head + h] * expf(pmax[b0 + (uint64_t)s * n_head + h] - m);
    den += expf(sink[h] - m);
    for (uint32_t d = threadIdx.x; d < hd; d += blockDim.x) {
        float v = 0.f;
        for (uint32_t s = 0; s < nseg; s++)
            v += pacc[(b0 + (uint64_t)s * n_head + h) * hd + d] * expf(pmax[b0 + (uint64_t)s * n_head + h] - m);
        o[((uint64_t)i * n_head + h) * hd + d] = v41_bf16r(v / den);
    }
}

/* ---- the scalar split-K decode (cuda_v41_attn_split.inc.cu:50-118): one
 * block per (segment, 8-head group), 128 threads = 4 warps x 2 heads; the
 * local exit keeps the denominator out (the merge owns it).  Numerics:
 * segmenting regroups the online softmax -> not bit-equal, quality gate. ---- */
__global__ static void v41_sparse_attn_split_kernel(float *pacc, float *pmax, float *psum,
                                                    const float *q, const float *kvw, const uint8_t *kvc, const int32_t *idx,
                                                    uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk,
                                                    uint32_t n_head, uint32_t hd, float scale, uint32_t seg_keys) {
    const uint32_t seg = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t per = hd / 32u;
    __shared__ float ks[V41_ATTN_KTILE][512];
    __shared__ int   kok[V41_ATTN_KTILE];
    __shared__ float ksc[V41_ATTN_KTILE][32];   /* the compressed row's 32 scales, decoded once per row */
    float qa[2][16], acc[2][16], mx[2], sum[2];
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * V41_ATTN_HEADS_PER_BLOCK + warp * 2u + hh;
        for (uint32_t e = 0; e < per; e++) { qa[hh][e] = q[(uint64_t)h * hd + lane * per + e]; acc[hh][e] = 0.f; }
        mx[hh] = -1e30f; sum[hh] = 0.f;
    }
    const uint32_t p = pos0;                       /* decode: one query, its absolute position is pos0 */
    const uint32_t lo = p + 1u > window ? p + 1u - window : 0u;
    const uint32_t nwin = p - lo + 1u;
    const uint32_t nkeys = nwin + topk;
    const uint32_t k0 = seg * seg_keys, k1 = (k0 + seg_keys) < nkeys ? (k0 + seg_keys) : nkeys;
    for (uint32_t base = k0; base < k1; base += V41_ATTN_KTILE) {
        const uint32_t nt = (k1 - base) < V41_ATTN_KTILE ? (k1 - base) : V41_ATTN_KTILE;
        __syncthreads();
        for (uint32_t t = threadIdx.x / 32u; t < nt; t += blockDim.x / 32u) {
            const uint32_t kk = base + t;
            const float *krow = NULL; const uint8_t *cpk = NULL;   /* window row f32 / compressed row packed FP4 */
            /* ring=1 holds here: this path only runs on the main route (!full_block), whose history is the ring */
            if (kk < nwin) krow = kvw + v41_win_row((int64_t)lo + kk, pos0, window, 1u) * hd;
            else { const int32_t g = idx[kk - nwin];
                   if (g >= 0 && (uint32_t)g < ng) cpk = kvc + (uint64_t)g * DS4_V41_CKV_BYTES; }
            if (lane == 0) kok[t] = (krow || cpk) ? 1 : 0;
            if (cpk) ksc[t][lane] = ds4_e4m3fn_to_f32(cpk[DS4_V41_CKV_NIB + lane]);
            __syncwarp();                              /* this warp writes and reads its own rows */
            for (uint32_t d = lane; d < hd; d += 32u)   /* invalid slots must be ZEROED, see the file header */
                ks[t][d] = krow ? krow[d] : (cpk ? v41_ckv_get_s(cpk, d, ksc[t]) : 0.f);
        }
        __syncthreads();
        for (int hh = 0; hh < 2; hh++) {
            float s[V41_ATTN_KTILE];
            float tm = -1e30f;
            for (uint32_t t = 0; t < nt; t++) {
                float d = 0.f;
                for (uint32_t e = 0; e < per; e++) d += qa[hh][e] * ks[t][lane * per + e];
                for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
                s[t] = kok[t] ? d * scale : -1e30f;
                tm = fmaxf(tm, s[t]);
            }
            const float nm = fmaxf(mx[hh], tm);
            const float rs = expf(mx[hh] - nm);
            float acc_add[16];
            for (uint32_t e = 0; e < per; e++) acc_add[e] = 0.f;
            float ps = 0.f;
            for (uint32_t t = 0; t < nt; t++) {
                const float pv = expf(s[t] - nm);
                ps += pv;
                const float pb = v41_bf16r(pv);
                for (uint32_t e = 0; e < per; e++) acc_add[e] += pb * ks[t][lane * per + e];
            }
            sum[hh] = sum[hh] * rs + ps;
            for (uint32_t e = 0; e < per; e++) acc[hh][e] = acc[hh][e] * rs + acc_add[e];
            mx[hh] = nm;
        }
    }
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * V41_ATTN_HEADS_PER_BLOCK + warp * 2u + hh;
        float *pa = pacc + ((uint64_t)seg * n_head + h) * hd;
        for (uint32_t e = 0; e < per; e++) pa[lane * per + e] = acc[hh][e];
        if (lane == 0) { pmax[(uint64_t)seg * n_head + h] = mx[hh]; psum[(uint64_t)seg * n_head + h] = sum[hh]; }
    }
}

/* decode-family scratch (cuda_v41_attn_split.inc.cu:201): the engine keeps
 * one triple per concurrent lane; the port is single-lane (ds41_indexer.cuh
 * precedent). */
static v41_scratch g_v41_attn_pacc, g_v41_attn_pmax, g_v41_attn_psum;

/* 1 = the scalar split-K took this launch; 0 = shape unfit, caller falls through. */
static int v41_sparse_attn_split(float *o, const float *q, const float *kvw, const uint8_t *kvc, const int32_t *idx,
                                 const float *sink, uint32_t n_tok, uint32_t pos0, uint32_t window, uint32_t ng,
                                 uint32_t topk, uint32_t n_head, uint32_t hd, float scale) {
    if (n_tok != 1u || hd != 512u || (n_head % V41_ATTN_HEADS_PER_BLOCK)) return 0;
    const uint32_t nwin = pos0 + 1u > window ? window : pos0 + 1u;
    const uint32_t nkeys = nwin + topk;
    if (nkeys < V41_ATTN_SPLIT_MIN_KEYS) return 0;
    /* seg length = min(64, keys/TARGET_SEG), multiple of 8, at least KTILE.
     * Keys >= 384 give 64, the behavior validated on 09-15. */
    uint32_t seg_keys = (nkeys + V41_ATTN_SPLIT_TARGET_SEG - 1u) / V41_ATTN_SPLIT_TARGET_SEG;
    seg_keys = ((seg_keys + V41_ATTN_SPLIT_SEG_MIN - 1u) / V41_ATTN_SPLIT_SEG_MIN) * V41_ATTN_SPLIT_SEG_MIN;
    if (seg_keys < V41_ATTN_SPLIT_SEG_MIN) seg_keys = V41_ATTN_SPLIT_SEG_MIN;
    if (seg_keys > V41_ATTN_SPLIT_SEG_KEYS) seg_keys = V41_ATTN_SPLIT_SEG_KEYS;
    uint32_t nseg = (nkeys + seg_keys - 1u) / seg_keys;
    while (nseg > V41_ATTN_SPLIT_MAX_SEG) { seg_keys *= 2u; nseg = (nkeys + seg_keys - 1u) / seg_keys; }
    const uint64_t na = (uint64_t)nseg * n_head;
    float *pacc = (float *)v41_grow(&g_v41_attn_pacc, na * hd * 4, "v41 attn split acc");
    float *pmax = (float *)v41_grow(&g_v41_attn_pmax, na * 4, "v41 attn split max");
    float *psum = (float *)v41_grow(&g_v41_attn_psum, na * 4, "v41 attn split sum");
    if (!pacc || !pmax || !psum) return 0;
    v41_sparse_attn_split_kernel<<<dim3(nseg, n_head / V41_ATTN_HEADS_PER_BLOCK), V41_ATTN_HEADS_PER_BLOCK * 16u, 0, ds4_current_stream()>>>(
        pacc, pmax, psum, q, kvw, kvc, idx, pos0, window, ng, topk, n_head, hd, scale, seg_keys);
    if (!cuda_ok(cudaGetLastError(), "v41 sparse attn split")) return 0;
    v41_pdl_register((const void *)v41_sparse_attn_merge_kernel);   /* the kernel v41_pdl_wait before reading the tiles */
    v41_sparse_attn_merge_kernel<<<dim3(n_head, 1), 256, 0, ds4_current_stream()>>>(o, pacc, pmax, psum, sink, nseg, n_head, hd,
                                                                            NULL, window, 0u, topk);
    return cuda_ok(cudaGetLastError(), "v41 sparse attn merge");
}

/* ---- the decode seg kernel (cuda_v41_attn_mma_decode.inc.cu:30-132): one
 * segment of keys per block, full two-pass online softmax locally, local
 * (max, sum, acc) out, merged in fixed segment order by the kernel above.
 * For one query, S = Q[64x512] . K^T[512xnkeys] is a real GEMM; the scalar
 * kernel pays 5 shuffles per 16 FMAs (69 GFLOP/s = 0.4% of the board). ---- */
/* posd (graph route, 2026-09-18): position from the device slot, ng/topk
 * derived from it; nseg is the bucket cap and segments past the true count
 * return without writing — the merge reads only real segments. */
/* Negative archive (2026-10-07): fusing the merge into this kernel via
 * threadfence (each (row, head-group)'s last block merges in fixed order)
 * was byte-equal but 46.33 -> 45.66 t/s — the merge tail serializes over
 * n_tok x 4 = 16 blocks instead of running as its own n_head x n_tok = 256
 * block launch.  Reverted to two launches; do not re-fuse. */
__global__ static void v41_attn_mma_seg_kernel(float *pacc, float *pmax, float *psum,
                                               const float *q, const float *kvw, const uint8_t *kvc,
                                               const int32_t *idx, uint32_t pos0, uint32_t window,
                                               uint32_t ng, uint32_t topk, uint32_t n_head,
                                               float scale, uint32_t ratio, uint32_t nseg, const int32_t *posd,
                                               uint32_t full_block, uint32_t ring) {
    v41_pdl_wait();   /* PDL: wait for the upstream as the first instruction */
    /* graph: position in the device slot; at n rows (gridDim.z, spec verify batch) source groups = (pos0 + n)/ratio, the eager path's ng_src formula */
    if (posd) { pos0 = (uint32_t)posd[0]; ng = ratio ? (pos0 + gridDim.z) / ratio : 0u; if (ng < topk) topk = ng; }
    namespace wmma = nvcuda::wmma;
    extern __shared__ char ds4_attn_mma_smem[];
    __nv_bfloat16 *qs = (__nv_bfloat16 *)ds4_attn_mma_smem;
    __nv_bfloat16 *ks = qs + DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD;
    float *spart = (float *)(ks + DS4_ATTN_MMA_KT * DS4_ATTN_MMA_HD);
    float *stile = spart + DS4_ATTN_MMA_WARPS * 256u;
    __nv_bfloat16 *ptile = (__nv_bfloat16 *)(stile + 256u);
    float *rmax = (float *)(ptile + 256u), *rsum = rmax + DS4_ATTN_MMA_HEADS;
    int *valid = (int *)(rsum + DS4_ATTN_MMA_HEADS);
    /* i = this batch's query index (decode: always 0; spec verify: 0..k).  Each query sees its own key range:
     * nseg is sized for the LAST query (most keys), so earlier queries have whole segments outside their
     * visible range — those must still write max=-inf / sum=0 / acc=0 or the merge reads last round's residue. */
    const uint32_t seg = blockIdx.x, h0 = blockIdx.y * DS4_ATTN_MMA_HEADS, i = blockIdx.z;
    const uint64_t pbase = ((uint64_t)i * nseg + seg) * n_head + h0;
    const uint32_t p = pos0 + i;
    /* full_block (DSpark draft block, also here since 2026-09-29): every position of the chunk sees one key
     * set = the whole window before it + all n in-block positions, no causal cut (official get_dspark_topk_idxs);
     * the main route (0) is causal: position i sees [p+1-window, p].  Window layout follows ring (towers are linear). */
    const uint32_t last = full_block ? pos0 + full_block - 1u : p;
    const uint32_t lo = full_block ? (pos0 > window ? pos0 - window : 0u) : (p + 1u > window ? p + 1u - window : 0u);
    const uint32_t nwin = last - lo + 1u, nkeys = nwin + topk;
    /* seg length by this query's OWN position (see the formulas): earlier queries may have a shorter
     * length and thus fewer segments than the grid's nseg — the extra segments take the empty branch,
     * which is exactly neutral to the merge. */
    const uint32_t seg_keys = full_block ? v41_attn_fb_seg_keys(nkeys) : v41_attn_seg_keys(p, window, ratio, topk);
    const uint32_t k0 = seg * seg_keys;
    if (k0 >= nkeys) {   /* empty segment: write neutral values and leave (exp(-1e30 - m) = 0 contributes nothing) */
        if (posd) return;   /* graph route: the merge reads only real segments, skip the 32 KB/segment white write */
        for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD; e += blockDim.x)
            pacc[(pbase + e / DS4_ATTN_MMA_HD) * DS4_ATTN_MMA_HD + e % DS4_ATTN_MMA_HD] = 0.f;
        if (threadIdx.x < DS4_ATTN_MMA_HEADS) { pmax[pbase + threadIdx.x] = -1e30f; psum[pbase + threadIdx.x] = 0.f; }
        return;
    }
    const uint32_t k1 = (k0 + seg_keys) < nkeys ? (k0 + seg_keys) : nkeys;
    for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD; e += blockDim.x)
        qs[e] = __float2bfloat16(q[((uint64_t)i * n_head + h0 + e / DS4_ATTN_MMA_HD) * DS4_ATTN_MMA_HD + e % DS4_ATTN_MMA_HD]);
    if (threadIdx.x < DS4_ATTN_MMA_HEADS) { rmax[threadIdx.x] = -1e30f; rsum[threadIdx.x] = 0.f; }
    __syncthreads();

    for (uint32_t base = k0; base < k1; base += DS4_ATTN_MMA_KT) {   /* pass 1: this segment's max and exp sum */
        const uint32_t nt = (k1 - base) < DS4_ATTN_MMA_KT ? (k1 - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        ds4_attn_mma_gather_keys(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk, ring);
        __syncthreads();
        ds4_attn_mma_scores(stile, spart, qs, ks, valid, nt, scale);
        ds4_attn_mma_stats(stile, rmax, rsum);   /* online max/sum (parallel form, bit-equal to serial) */
    }
    __syncthreads();

    const uint32_t warp = threadIdx.x >> 5;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[4];
    for (int j = 0; j < 4; j++) wmma::fill_fragment(oacc[j], 0.0f);
    for (uint32_t base = k0; base < k1; base += DS4_ATTN_MMA_KT) {   /* pass 2: recompute S -> P(bf16) -> O */
        const uint32_t nt = (k1 - base) < DS4_ATTN_MMA_KT ? (k1 - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        ds4_attn_mma_gather_keys(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk, ring);
        __syncthreads();
        ds4_attn_mma_scores(stile, spart, qs, ks, valid, nt, scale);
        for (uint32_t e = threadIdx.x; e < 256u; e += blockDim.x)
            ptile[e] = __float2bfloat16(expf(stile[e] - rmax[e >> 4]));
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vb;
        wmma::load_matrix_sync(pa, ptile, 16);
        for (int j = 0; j < 4; j++) {
            wmma::load_matrix_sync(vb, ks + warp * 64u + (uint32_t)j * 16u, DS4_ATTN_MMA_HD);
            wmma::mma_sync(oacc[j], pa, vb, oacc[j]);
        }
    }

    /* exit: local acc/max/sum only.  Two slices per round (2026-09-29, micro-bench V7; the old per-j
     * rounds borrowed 8 KB of spart): 8 warps x 4 slices = 32 KB does not fit shared, but after the last
     * P*V the 16 KB of ks is free — exactly 8 warps x 2 slices.  Each slice keeps its warp offset
     * (a shared slab without offsets is the 09-15 bug that made PPL 280k). */
    float *otile = (float *)ks;
    for (int r = 0; r < 2; r++) {
        __syncthreads();
        wmma::store_matrix_sync(otile + (size_t)warp * 512u, oacc[2 * r], 16, wmma::mem_row_major);
        wmma::store_matrix_sync(otile + (size_t)warp * 512u + 256u, oacc[2 * r + 1], 16, wmma::mem_row_major);
        __syncthreads();
        for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_WARPS * 512u; e += blockDim.x) {
            const uint32_t w = e >> 9, rr = e & 511u, jj = rr >> 8, t = rr & 255u, h = t >> 4, d16 = t & 15u;
            pacc[(pbase + h) * DS4_ATTN_MMA_HD + w * 64u + (uint32_t)(2 * r) * 16u + jj * 16u + d16] = otile[e];
        }
    }
    __syncthreads();
    if (threadIdx.x < DS4_ATTN_MMA_HEADS) {
        pmax[pbase + threadIdx.x] = rmax[threadIdx.x];
        psum[pbase + threadIdx.x] = rsum[threadIdx.x];
    }
}

/* Scratch pre-grow for graph capture (cuda_v41_attn_mma_decode.inc.cu:141-146):
 * v41_grow inside a capture does cudaMalloc + a device sync, both of which
 * void the capture — so grow the local tiles to cap-segment x graph-rows
 * BEFORE capturing, grow-only.  n_tok must be the graph's row count (1 for
 * pure decode, 1+k for the spec batch); one row short and the verify batch
 * re-grows inside the capture and voids the graph.  n_tok <= 8 => <= 64 MB. */
extern "C" int ds4_gpu_v41_attn_scratch_prepare(uint32_t n_tok, uint32_t n_head, uint32_t head_dim) {
    const uint64_t na = (uint64_t)V41_ATTN_SPLIT_MAX_SEG * n_tok * n_head;
    return v41_grow(&g_v41_attn_pacc, na * head_dim * 4, "v41 attn mma acc") &&
           v41_grow(&g_v41_attn_pmax, na * 4, "v41 attn mma max") &&
           v41_grow(&g_v41_attn_psum, na * 4, "v41 attn mma sum") ? 1 : 0;
}

/* posd / pos_cap: graph route (cuda_v41_attn_mma_decode.inc.cu:148-150):
 * pos0..pos_cap is the graph's valid position range; the grid's segment
 * count is the max over that range (the count is not monotonic in position —
 * it drops when the seg-length steps — so scan every position, not the ends). */
static int v41_attn_mma_decode(float *o, const float *q, const float *kvw, const uint8_t *kvc, const int32_t *idx,
                               const float *sink, uint32_t n_tok, uint32_t pos0, uint32_t window, uint32_t ng,
                               uint32_t topk, uint32_t ratio, uint32_t n_head, uint32_t hd, float scale,
                               const int32_t *posd, uint32_t pos_cap, uint32_t full_block, uint32_t ring) {
    /* n_tok 1 = pure decode; 2..8 = spec verify batch — both MUST run this
     * kernel family (mtp.md M1): at temperature 0 a speculated token has to
     * be byte-equal to pure decode, and two kernels with different
     * accumulation orders can never be. */
    if (n_tok == 0u || n_tok > 8u || hd != DS4_ATTN_MMA_HD || (n_head % DS4_ATTN_MMA_HEADS)) return 0;
    const uint32_t plast = pos0 + n_tok - 1u;        /* segment count by the query with the most keys (earlier queries' extras are empty) */
    const uint32_t nwin = plast + 1u > window ? window : plast + 1u;
    const uint32_t nkeys = nwin + topk;
    /* Few keys still come here, no scalar fallback (2026-09-16 M1'): the old
     * nkeys < 64 -> split hand-off only accepted n_tok == 1, so the first
     * steps ran split while the verify batch ran the prefill kernel — not
     * even the same kernel, same-track impossible.  Few keys = one segment,
     * the cost is one q load; no second route. */
    static int s_ok = 0;                             /* 0 untried / 1 usable / -1 cannot be raised */
    const size_t smem = ds4_attn_mma_seg_smem_bytes();   /* the seg kernel keeps the old qs+ks two-slab layout; the prefill kernel's shared account differs */
    if (s_ok == 0) {
        (void)cudaGetLastError();   /* the opt-in must not see a previously latched error (see ds41_vq_decode.cuh) */
        s_ok = cudaFuncSetAttribute(v41_attn_mma_seg_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem) == cudaSuccess ? 1 : -1;
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [attn] decode tensor-core shared %zu KB %s\n", smem >> 10, s_ok == 1 ? "enabled" : "*cannot be raised, back to scalar*");
    }
    if (s_ok != 1) return 0;
    /* grid segments = the max of the per-query segment counts (their lengths may differ); the kernel
     * recomputes each length from its own position — the host only sizes the grid. */
    uint32_t nseg = 1u;
    if (full_block) {   /* draft block: n positions share one key set (window + the n in-block positions), length from the count only */
        const uint32_t lo = pos0 > window ? pos0 - window : 0u, nk = pos0 + n_tok - lo;
        nseg = (nk + v41_attn_fb_seg_keys(nk) - 1u) / v41_attn_fb_seg_keys(nk);
    } else if (posd) {   /* graph: max over each position in the bucket (the kernel computes the true count; extras run empty); with n rows the last position is pos_cap + n - 1 */
        if (pos_cap < pos0) return 0;
        for (uint32_t p = pos0; p <= pos_cap + n_tok - 1u; p++) {
            const uint32_t ns = v41_attn_nseg_at(p, window, ratio, topk);
            if (ns > nseg) nseg = ns;
        }
    } else
    for (uint32_t i = 0; i < n_tok; i++) {
        const uint32_t p = pos0 + i;
        const uint32_t nw = p + 1u > window ? window : p + 1u;
        const uint32_t nk = nw + topk;
        const uint32_t sk = v41_attn_seg_keys(p, window, ratio, topk);
        const uint32_t ns = (nk + sk - 1u) / sk;
        if (ns > nseg) nseg = ns;
    }
    if (nseg > V41_ATTN_SPLIT_MAX_SEG) {   /* key cap = window + indexer top-k cap, unreachable */
        fprintf(stderr, "ds4: *[attn] %u keys need %u segments, over the cap %u — another kernel ran, same-track no longer holds*\n",
                nkeys, nseg, (unsigned)V41_ATTN_SPLIT_MAX_SEG);
        return 0;
    }
    const uint64_t na = (uint64_t)nseg * n_tok * n_head;
    float *pacc = (float *)v41_grow(&g_v41_attn_pacc, na * hd * 4, "v41 attn mma acc");
    float *pmax = (float *)v41_grow(&g_v41_attn_pmax, na * 4, "v41 attn mma max");
    float *psum = (float *)v41_grow(&g_v41_attn_psum, na * 4, "v41 attn mma sum");
    if (!pacc || !pmax || !psum) return 0;
    v41_pdl_register((const void *)v41_attn_mma_seg_kernel);   /* the kernel v41_pdl_wait before reading upstream output */
    v41_attn_mma_seg_kernel<<<dim3(nseg, n_head / DS4_ATTN_MMA_HEADS, n_tok), DS4_ATTN_MMA_WARPS * 32u, smem, ds4_current_stream()>>>(
        pacc, pmax, psum, q, kvw, kvc, idx, pos0, window, ng, topk, n_head, scale, ratio, nseg, posd, full_block, ring);
    if (!cuda_ok(cudaGetLastError(), "v41 attn mma seg")) return 0;
    /* the merge reuses the scalar split's launch (fixed segment order + sink in the denominator + divide + bf16 round), identical semantics */
    v41_pdl_register((const void *)v41_sparse_attn_merge_kernel);
    v41_sparse_attn_merge_kernel<<<dim3(n_head, n_tok), 256, 0, ds4_current_stream()>>>(o, pacc, pmax, psum, sink, nseg, n_head, hd,
                                                                                posd, window, ratio, topk);
    return cuda_ok(cudaGetLastError(), "v41 attn mma merge");
}

/* ---- the entry (cuda_v41_2.inc.cu:229-303): the engine's try order. ----
 * The clamp only exists in the prefill kernel; at decode (n<=8) it must be a
 * no-op — refuse rather than silently read dirty ring slots.  The generation
 * route leaves the last window positions to the last block (core_v41_api.c),
 * so at decode p+1-window >= win_lo always holds. */
extern "C" int ds4_gpu_v41_sparse_attn_tensor(ds4_gpu_tensor *o, const ds4_gpu_tensor *q, const ds4_gpu_tensor *kv_win,
                                              const ds4_gpu_tensor *kv_comp, const ds4_gpu_tensor *idx,
                                              const void *model_map, uint64_t model_size, uint64_t sink_offset,
                                              uint32_t n_tok, uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk,
                                              uint32_t ratio,
                                              uint32_t n_head, uint32_t head_dim, float scale, int full_block, int ring,
                                              uint32_t win_lo, const ds4_gpu_tensor *posd, uint32_t pos_cap) {
    if (!o || !q || !kv_win || n_head != 64u || head_dim != 512u) { fprintf(stderr, "ds4: [v41] sparse attn only implements 64 heads x 512\n"); return 0; }
    const uint32_t lo_first = pos0 + 1u > window ? pos0 + 1u - window : 0u;   /* this batch's first query's natural lower bound (later ones only grow) */
    const int clamp_matters = win_lo > lo_first;
    if (clamp_matters && n_tok <= 8u && !full_block) {
        fprintf(stderr, "ds4: *[v41] sparse attn: the decode kernels have no window clamp, but pos0 %u's window lower bound %u < win_lo %u — did the last block not keep a full window?*\n",
                pos0, lo_first, win_lo);
        return 0;
    }
    /* The graph route (posd) only accepts the decode tensor-core path; the
     * scalar split sizes its segments on the host and cannot enter one
     * captured graph.  The tensor-core path failing to raise shared fails
     * HERE, loudly — never a silent switch to another kernel: a different
     * kernel is a different accumulation order and the byte gate would fork
     * (engine cuda_v41_2.inc.cu:246-260). */
    if (posd) {
        if (full_block || !ring || n_tok > 8u) { fprintf(stderr, "ds4: [ds41] sparse attn: the graph route takes only the main path's n<=8 shape\n"); return 0; }
        const float *sink = (const float *)cuda_model_range_ptr(model_map, sink_offset, (uint64_t)n_head * 4, "v41 sink");
        if (!sink) return 0;
        const int hasc = kv_comp && idx;   /* pure-window layers (ratio 0) come here too: the window range moves with the position as well */
        if (!v41_attn_mma_decode((float *)o->ptr, (const float *)q->ptr, (const float *)kv_win->ptr,
                                 hasc ? (const uint8_t *)kv_comp->ptr : NULL, hasc ? (const int32_t *)idx->ptr : NULL,
                                 sink, n_tok, pos0, window, hasc ? ng : 0u, hasc ? topk : 0u, hasc ? ratio : 0u, n_head, head_dim, scale,
                                 (const int32_t *)posd->ptr, pos_cap, 0u, 1u)) {
            fprintf(stderr, "ds4: [ds41] sparse attn: the decode tensor-core attention the graph route needs is unavailable\n"); return 0;
        }
        return 1;
    }
    if (kv_win->bytes < (uint64_t)(window + n_tok) * head_dim * 4) { fprintf(stderr, "ds4: [v41] window buffer short %u+%u rows\n", window, n_tok); return 0; }
    const float *sink = (const float *)cuda_model_range_ptr(model_map, sink_offset, (uint64_t)n_head * 4, "v41 sink");
    if (!sink) return 0;
    /* The split-K and mma fast routes only serve the main route (history is
     * the ring): their call condition is !full_block, and full_block only
     * ever comes from the DSpark draft tower — so ring is 1 whenever they
     * run, and the kernels compute ring directly.  A non-draft full_block or
     * a non-ring main route trips this check instead of silently reading
     * wrong rows. */
    if (!full_block && !ring) { fprintf(stderr, "ds4: [v41] sparse attn: the main path's window must be a ring\n"); return 0; }
    /* Decode first tries the tensor-core family (decode.md D2): for one
     * query S = Q[64x512] . K^T[512xnkeys] is a real GEMM; the scalar kernel
     * pays 5 shuffles per 16 FMAs (69 GFLOP/s = 0.4% of the board's peak).
     * Few keys / unfit shapes return 0 here by themselves. */
    if (!full_block &&
        v41_attn_mma_decode((float *)o->ptr, (const float *)q->ptr, (const float *)kv_win->ptr,
                            kv_comp ? (const uint8_t *)kv_comp->ptr : NULL, idx ? (const int32_t *)idx->ptr : NULL,
                            sink, n_tok, pos0, window, ng, (kv_comp && idx) ? topk : 0u,
                            (kv_comp && idx) ? ratio : 0u, n_head, head_dim, scale, NULL, 0u, 0u, 1u))
        return 1;
    /* The DSpark draft block runs the tensor-core family too (2026-09-29):
     * the tower's attention has window keys only, in-block full visibility,
     * and a linear window segment (ring=0).  It used to fall to the scalar
     * kernel: 1.6 ms per 3-tower round (0.5 ms per launch for 5 positions x
     * 133 keys), the draft step's third-largest item.  Drafts only propose,
     * verification decides the output, so the gate is the acceptance
     * histogram, not the text.  Few keys / unfit shapes return 0 and fall
     * through to the scalar kernel. */
    if (full_block && !ring && !kv_comp && n_tok <= 8u &&
        v41_attn_mma_decode((float *)o->ptr, (const float *)q->ptr, (const float *)kv_win->ptr, NULL, NULL,
                            sink, n_tok, pos0, window, 0u, 0u, 0u, n_head, head_dim, scale, NULL, 0u, n_tok, 0u))
        return 1;
    if (!full_block &&
        v41_sparse_attn_split((float *)o->ptr, (const float *)q->ptr, (const float *)kv_win->ptr,
                              kv_comp ? (const uint8_t *)kv_comp->ptr : NULL, idx ? (const int32_t *)idx->ptr : NULL,
                              sink, n_tok, pos0, window, ng, (kv_comp && idx) ? topk : 0u, n_head, head_dim, scale))
        return 1;
    /* Prefill (n >= 64) takes the single-pass mma kernel; it checks shape,
     * size and shared itself and returns 0 to fall through. */
    if (!full_block && ds4_sparse_attn_mma_launch((float *)o->ptr, (const float *)q->ptr, (const float *)kv_win->ptr,
                                   kv_comp ? (const uint8_t *)kv_comp->ptr : NULL, idx ? (const int32_t *)idx->ptr : NULL,
                                   sink, n_tok, pos0, window, ng, (kv_comp && idx) ? topk : 0u, n_head, head_dim, scale, win_lo))
        return 1;
    v41_sparse_attn_kernel<<<dim3(n_tok, n_head / V41_ATTN_HEADS_PER_BLOCK), V41_ATTN_HEADS_PER_BLOCK * 16u, 0, ds4_current_stream()>>>(
        (float *)o->ptr, (const float *)q->ptr, (const float *)kv_win->ptr,
        kv_comp ? (const uint8_t *)kv_comp->ptr : NULL, idx ? (const int32_t *)idx->ptr : NULL, sink, pos0, window, ng,
        (kv_comp && idx) ? topk : 0u, n_head, head_dim, scale, full_block ? n_tok : 0u, ring ? 1u : 0u, win_lo);
    return cuda_ok(cudaGetLastError(), "v41 sparse attn");
}
