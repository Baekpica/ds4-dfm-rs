/* ds41_indexer_mma.cuh — the V4.1 indexer score kernel's tensor-core arms.
 *
 * Why: the score kernel is the only long-context term of prefill — every
 * prompt row scores every earlier source group (32 heads x 128 dims per
 * (row, group)). The scalar kernel in ds41_indexer.cuh is issue-bound at long
 * context: its time sits on the instruction-issue floor, so more threads or
 * occupancy cannot move it, and only the tensor cores can. On the tensor
 * cores the 4096 multiply-adds of one (row, group) take tens of instructions
 * instead of hundreds. The arms, their repeats and their conditions:
 * docs/ds41-indexer-2026-11-08.md.
 *
 * Two arms share this file; DS4_DS41_IDX_MMA picks one (2 = native MXFP4,
 * the default on a Blackwell build; 1 = s8).
 *
 * Numerics (s8 arm): the index q is e2m1 x ue8m0 per 32 dims (act_quant
 * mode 1) and the index key is MXFP4 e2m1 x E8M0 per 32 dims. Both mantissas
 * times 2 become integers of magnitude <= 12, so an m16n8k32 int8 MMA (one
 * instruction per 32-dim block) accumulates exactly in int32 (|sum| <= 32 *
 * 144), then multiplies the block scale 2^(eq+ek-2), itself exact. Each
 * block's dot product T_b therefore equals the elementwise f32 sum. The four
 * block sums (T0+T1)+(T2+T3) are the single rounding point: the scalar kernel
 * folds lane-in 4 dims then an xor 16/8/4/2/1 butterfly (across blocks before
 * within), so the two differ only when a block-scale spread > 2^9 lands
 * exactly on a bf16 boundary.
 *
 * Shapes: the head count is a template (32 or 64, the metadata's value); the
 * head dim must be 128. The dispatch in ds41_indexer.cuh sends only those
 * shapes here, so any other metadata shape keeps the scalar kernel.
 *
 * Named port deviation: one static q scratch slot, not a per-lane array (the
 * port has no decode-lane concept yet; ds41_indexer.cuh's header records the
 * same deviation for the candidate scratch).
 */
#pragma once

/* Per-launch q scratch: [rows][NH][128] s8 mantissas + [rows][NH][4] s8 block
 * exponents. */
static v41_scratch g_v41_idxq;
static inline uint64_t v41_idxq_row_bytes(uint32_t n_head) { return (uint64_t)n_head * 128u + (uint64_t)n_head * 4u; }

/* SM count, queried once. The launch can run at capture time; a
 * cudaDeviceGetAttribute there is legal but has no business on the recorded
 * path, so it is resolved here (one lazy query, never at capture time). */
static int v41_idx_sm(void) {
    static int s_sm = 0;
    if (!s_sm && cudaDeviceGetAttribute(&s_sm, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    return s_sm;
}

/* Grow-only q scratch. cudaMalloc and cudaDeviceSynchronize are illegal under
 * stream capture, so the growth happens eagerly: the port calls
 * ds4_gpu_v41_indexer_scratch_prepare from v41_index_scratch_prepare
 * (ds4_ds41_forward.inc) before a capture, and a short scratch met under
 * capture fails loud instead of voiding it (ds4_ds41_graph.inc:206-209 is
 * that failure: the step silently ran direct all night). */
static int v41_idx_grow(uint64_t need) {
    if (need <= g_v41_idxq.cap) { return 1; }
    if (ds4_capture_active()) {
        fprintf(stderr, "ds4: [ds41] indexer q scratch growth asked under capture (%.2f MB > %.2f MB)\n",
                (double)need / 1048576.0, (double)g_v41_idxq.cap / 1048576.0);
        return 0;
    }
    return v41_grow(&g_v41_idxq, need, "v41 indexer q s8") ? 1 : 0;
}

extern "C" int ds4_gpu_v41_indexer_scratch_prepare(uint32_t n_tok, uint32_t n_head) {
    if (!v41_idx_mma_mode() || n_tok == 0u) { return 1; }
    if (!v41_idx_sm()) { return 0; }
    return v41_idx_grow((uint64_t)n_tok * v41_idxq_row_bytes(n_head));
}

/* q (f32, value = e2m1 x 2^s) -> per (head, 32-dim block) one unit exponent e
 * and the integer mantissa m2 = q * 2^(1-e) with |m2| <= 12.
 * One warp per head, one lane per 4 dims = the MMA A fragment's k order.
 * amax = mmax * 2^s with mmax in {0.5..6}; frexp gives amax = f * 2^ex, so
 * e = ex - 3 and m2 lands in {2m, 4m, 8m, 16m}, the one <= 12 being an
 * integer by construction. */
__global__ static void v41_idxq_prep_kernel(int8_t *qi, int8_t *qe, const float *q, uint32_t n_rows, uint32_t n_head) {
    v41_pdl_wait();
    const uint32_t lane = threadIdx.x & 31u, wid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);   /* wid = row * heads + head */
    if (wid >= n_rows * n_head) return;
    const float *qh = q + (uint64_t)wid * 128u + lane * 4u;
    float v[4];
    #pragma unroll
    for (uint32_t e = 0; e < 4u; e++) v[e] = qh[e];
    float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    for (int o = 1; o < 8; o <<= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));   /* 8 lanes = one 32-dim block */
    /* v41_pow2i(e-2) is an exact normal power of two only for e >= -124, i.e.
     * amax >= 2^-122 (frexp gives ex = floor(log2 a) + 1). Beneath that the
     * dequantised block's own scale is not representable: n = -127 yields
     * 0.0f and n <= -128 yields -inf (v41_pow2i's bit pattern). Such a block
     * is zeroed here, so every block whose scale is representable keeps its
     * values unchanged; the zero is a named deviation for the rest. */
    int ex = 0; (void)frexpf(a, &ex);
    const bool tiny = !(a >= 0x1p-122f);
    const int e = tiny ? 0 : ex - 3;
    uint32_t packed = 0;
    #pragma unroll
    for (uint32_t kk = 0; kk < 4u; kk++) {
        const int m2 = tiny ? 0 : __float2int_rn(ldexpf(v[kk], 1 - e));   /* exact integer */
        packed |= ((uint32_t)(uint8_t)(int8_t)m2) << (8u * kk);
    }
    *(uint32_t *)(qi + (uint64_t)wid * 128u + lane * 4u) = packed;
    if ((lane & 7u) == 0u) qe[(uint64_t)wid * 4u + (lane >> 3)] = (int8_t)e;
}

/* Compact item c -> real group g and liveness, the same rule as the scalar
 * kernel: off-list, g >= ng and g >= vis are dead groups. */
__device__ __forceinline__ static bool v41_cand_live(const int32_t *cl, uint32_t nc, uint32_t c, uint32_t ns, uint32_t bs, uint32_t ng, uint32_t vis, uint32_t *g) {
    uint32_t gg = c;
    if (cl) { const uint32_t b = c / bs; gg = b < nc ? (uint32_t)cl[1u + b] * bs + c % bs : ng; }
    *g = gg;
    return c < ns && gg < ng && gg < vis;
}

/* m16n8k32 s8 -> s32, operands "row.col" (A is the q fragment, B the key). */
__device__ __forceinline__ static void v41_mma_s8(int32_t *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    const int32_t z = 0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(z), "r"(z), "r"(z), "r"(z));
}

/* 2^n as an f32 bit pattern (block scales are exact powers of two). Requires
 * n >= -126: the prep zeroes blocks below 2^-122, so e-2 never underflows and
 * a block whose dequantised max sits at 2^-126 cannot reach this function. */
__device__ __forceinline__ static float v41_pow2i(int n) { uint32_t u = (uint32_t)(n + 127) << 23; float f; memcpy(&f, &u, 4); return f; }

#define V41_IMMA_QS_STRIDE 144u   /* shared q row pitch (128 + 16 pad): an A fragment reads 8 rows x 4 words on 32 banks */

/* NH = head count (template, multiple of 16): MT = NH/16 m-tiles; one block is
 * 8 warps and each warp advances 8 groups per tile. */
template <uint32_t NH>
__global__ static void __launch_bounds__(256) v41_indexer_score_mma_kernel(float *score, const int8_t *qi, const int8_t *qe, const uint8_t *k,
                                                                          const float *w, const int32_t *cand, uint32_t pos0, uint32_t ng,
                                                                          uint32_t ratio, const int32_t *posd, uint32_t cand_bs, uint32_t cand_cap,
                                                                          uint32_t tiles) {
    constexpr uint32_t MT = NH / 16u, XS = NH + 1u;   /* XS: shared transpose pitch (heads + 1) so 4 groups never share a bank */
    v41_pdl_wait();
    const uint32_t i = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    if (posd) { pos0 = (uint32_t)posd[0]; ng = (pos0 + gridDim.x) / ratio; }
    const uint32_t vis = (pos0 + i + 1u) / ratio;
    const int32_t *cl = cand ? cand + (uint64_t)i * (1u + cand_cap) : NULL;
    const uint32_t nc = cl ? (uint32_t)cl[0] : 0u;
    const uint32_t ns = cl ? v41_cand_ns(ng, cand_bs, cand_cap) : ng;
    __shared__ uint16_t lut[256];                             /* byte (two nibbles) -> two s8 mantissas x2 (low nibble in the low byte) */
    __shared__ __align__(16) int8_t qs[NH * V41_IMMA_QS_STRIDE];   /* this row's NH heads of q mantissas */
    __shared__ float xs[8][8][XS];                            /* [warp][group][head] relu(dot)*w on bf16, transposed for the head-order sum */
    {
        const uint32_t v = threadIdx.x, lo = v & 15u, hi = v >> 4;
        const int ml = ((lo >> 1) & 3u) ? ((2 + (int)(lo & 1u)) << (((lo >> 1) & 3u) - 1u)) : (int)(lo & 1u);
        const int mh = ((hi >> 1) & 3u) ? ((2 + (int)(hi & 1u)) << (((hi >> 1) & 3u) - 1u)) : (int)(hi & 1u);
        const int sl = (lo & 8u) ? -ml : ml, sh = (hi & 8u) ? -mh : mh;
        lut[v] = (uint16_t)((uint8_t)(int8_t)sl | ((uint16_t)(uint8_t)(int8_t)sh << 8));
        const uint32_t *src = (const uint32_t *)(qi + (uint64_t)i * NH * 128u);
        #pragma unroll
        for (uint32_t t = 0; t < NH / 8u; t++) {   /* NH * 128 B = NH*32 words, NH/8 per thread: word x -> head x/32, dim (x%32)*4 */
            const uint32_t x = threadIdx.x + t * 256u;
            *(uint32_t *)(qs + (x >> 5) * V41_IMMA_QS_STRIDE + (x & 31u) * 4u) = src[x];
        }
    }
    __syncthreads();
    const uint32_t g4 = lane >> 2, t4 = lane & 3u;
    /* This lane's C-fragment heads: mt*16 + g4 (+8); their packed block exponents and weights */
    uint32_t eqp[MT][2]; float wh[MT][2];
    #pragma unroll
    for (uint32_t mt = 0; mt < MT; mt++) {
        #pragma unroll
        for (uint32_t hh = 0; hh < 2u; hh++) {
            const uint32_t h = mt * 16u + g4 + hh * 8u;
            wh[mt][hh] = w[(uint64_t)i * NH + h];
            eqp[mt][hh] = *(const uint32_t *)(qe + ((uint64_t)i * NH + h) * 4u);
        }
    }
    const uint32_t c_base = (blockIdx.y * 8u + warp) * tiles * 8u;
    for (uint32_t t = 0; t < tiles; t++) {
        const uint32_t c0 = c_base + t * 8u;
        if (c0 >= ns) break;
        /* B fragment: this lane's group n = lane/4, block b's dims (lane%4)*4 (+16) = two nibble bytes */
        uint32_t gB; const bool liveB = v41_cand_live(cl, nc, c0 + g4, ns, cand_bs, ng, vis, &gB);
        const uint8_t *kr = k + (uint64_t)(liveB ? gB : 0u) * DS4_V41_IDXK_BYTES;
        /* The C fragment's two groups (lane%4)*2 + {0,1}: their E8M0 block scales (4 bytes at the row tail, one u32) */
        uint32_t gC[2]; bool liveC[2]; uint32_t ek[2];
        #pragma unroll
        for (uint32_t j = 0; j < 2u; j++) {
            liveC[j] = v41_cand_live(cl, nc, c0 + t4 * 2u + j, ns, cand_bs, ng, vis, &gC[j]);
            ek[j] = *(const uint32_t *)(k + (uint64_t)(liveC[j] ? gC[j] : 0u) * DS4_V41_IDXK_BYTES + DS4_V41_IDXK_NIB);
        }
        float s01[MT][4], S[MT][4];
        #pragma unroll
        for (uint32_t b = 0; b < 4u; b++) {
            const uint32_t x0 = *(const uint16_t *)(kr + b * 16u + t4 * 2u), x1 = *(const uint16_t *)(kr + b * 16u + 8u + t4 * 2u);
            const uint32_t b0 = liveB ? ((uint32_t)lut[x0 & 0xFFu] | ((uint32_t)lut[x0 >> 8] << 16)) : 0u;
            const uint32_t b1 = liveB ? ((uint32_t)lut[x1 & 0xFFu] | ((uint32_t)lut[x1 >> 8] << 16)) : 0u;
            const float fk0 = ds4_e8m0_to_f32((uint8_t)(ek[0] >> (8u * b))), fk1 = ds4_e8m0_to_f32((uint8_t)(ek[1] >> (8u * b)));
            #pragma unroll
            for (uint32_t mt = 0; mt < MT; mt++) {
                const int8_t *q0 = qs + (mt * 16u + g4) * V41_IMMA_QS_STRIDE + b * 32u + t4 * 4u, *q1 = q0 + 8u * V41_IMMA_QS_STRIDE;
                int32_t d[4];
                v41_mma_s8(d, *(const uint32_t *)q0, *(const uint32_t *)q1, *(const uint32_t *)(q0 + 16u), *(const uint32_t *)(q1 + 16u), b0, b1);
                /* c0,c1: head g4, groups t4*2+{0,1}; c2,c3: head g4+8. Scale = 2^(eq-2) * 2^ek (both multiplies exact). */
                const float fq0 = v41_pow2i((int)(int8_t)(eqp[mt][0] >> (8u * b)) - 2), fq1 = v41_pow2i((int)(int8_t)(eqp[mt][1] >> (8u * b)) - 2);
                float T[4];
                T[0] = ((float)d[0] * fq0) * fk0; T[1] = ((float)d[1] * fq0) * fk1;
                T[2] = ((float)d[2] * fq1) * fk0; T[3] = ((float)d[3] * fq1) * fk1;
                #pragma unroll
                for (uint32_t c = 0; c < 4u; c++) {   /* S = (T0 + T1) + (T2 + T3) */
                    if (b == 0u) s01[mt][c] = T[c];
                    else if (b == 1u) s01[mt][c] = s01[mt][c] + T[c];
                    else if (b == 2u) S[mt][c] = T[c];
                    else S[mt][c] = s01[mt][c] + (S[mt][c] + T[c]);
                }
            }
        }
        /* Tail, official expression order: bf16(dot) -> relu -> bf16(* w_h) -> transpose to shared, one lane per group sums by head order -> bf16 */
        #pragma unroll
        for (uint32_t mt = 0; mt < MT; mt++) {
            #pragma unroll
            for (uint32_t c = 0; c < 4u; c++) {
                const uint32_t hh = c >> 1, h = mt * 16u + g4 + hh * 8u, gl = t4 * 2u + (c & 1u);
                const float dd = fmaxf(v41_bf16r(S[mt][c]), 0.0f);
                xs[warp][gl][h] = v41_bf16r(dd * wh[mt][hh]);
            }
        }
        __syncwarp();
        if (lane < 8u) {
            uint32_t gW; const bool liveW = v41_cand_live(cl, nc, c0 + lane, ns, cand_bs, ng, vis, &gW);
            float acc = 0.f;
            #pragma unroll 8
            for (uint32_t h = 0; h < NH; h++) acc += xs[warp][lane][h];
            if (c0 + lane < ns) score[(uint64_t)i * ns + c0 + lane] = liveW ? v41_bf16r(acc) : -INFINITY;
        }
        __syncwarp();
    }
}

/* Grid shape for a compact width ns: (warps of 8 groups, group blocks). Each
 * warp runs `tiles` slices of 8 groups, tiles sized from ns and the SM count
 * so the first wave covers at least two blocks per SM. The entry consults
 * this before dispatching so the CUDA grid.y cap falls back to the scalar
 * kernel (which wraps by grid stride) instead of failing the forward. */
static void v41_idx_mma_grid(uint32_t ns, uint32_t *tiles_out, uint32_t *gblocks_out) {
    const uint32_t warps = (ns + 7u) / 8u;
    const uint32_t sm = (uint32_t)v41_idx_sm();
    uint32_t tiles = sm ? (warps + 16u * sm - 1u) / (16u * sm) : 1u;   /* two blocks of 8 warps per SM is one wave */
    if (tiles < 1u) tiles = 1u;
    if (tiles > 32u) tiles = 32u;
    *tiles_out = tiles;
    *gblocks_out = (warps + 8u * tiles - 1u) / (8u * tiles);
}

/* Tensor-core entry, same contract as ds4_gpu_v41_indexer_score_tensor (which
 * dispatches here while the flag is on). */
static int v41_indexer_score_mma_launch(ds4_gpu_tensor *score, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *weights,
                                        const ds4_gpu_tensor *cand_list, uint32_t cand_bs, uint32_t cand_cap, uint32_t n_tok, uint32_t pos0,
                                        uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio, const ds4_gpu_tensor *posd) {
    if ((n_head != 32u && n_head != 64u) || dk != 128u) {
        fprintf(stderr, "ds4: [ds41] indexer tensor-core score is compiled for 32/64 heads x 128 dims (metadata %u x %u)\n", n_head, dk); return 0;
    }
    if (!v41_idx_grow((uint64_t)n_tok * v41_idxq_row_bytes(n_head))) return 0;
    int8_t *qi = (int8_t *)g_v41_idxq.p;
    int8_t *qe = qi + (uint64_t)n_tok * n_head * 128u;
    v41_idxq_prep_kernel<<<(unsigned)((n_tok * n_head + 7u) / 8u), 256, 0, ds4_current_stream()>>>(qi, qe, (const float *)q->ptr, n_tok, n_head);
    if (!cuda_ok(cudaGetLastError(), "v41 indexer q prep")) return 0;
    const uint32_t ns = cand_list ? v41_cand_ns(ng, cand_bs, cand_cap) : ng;
    uint32_t tiles = 1, gblocks = 1;
    v41_idx_mma_grid(ns, &tiles, &gblocks);
    static int s_once = 0;
    if (!s_once) { s_once = 1; fprintf(stderr, "ds4: [ds41] indexer score on tensor cores: first n_tok %u ns %u heads %u x %u dims, tiles %u, grid (%u, %u), SM %d\n", n_tok, ns, n_head, dk, tiles, n_tok, gblocks, v41_idx_sm()); }
    if (gblocks > 65535u) { fprintf(stderr, "ds4: [ds41] indexer tensor-core grid.y %u over CUDA's cap (ns %u)\n", gblocks, ns); return 0; }   /* 1M: tiles=32 gives 512 blocks */
    const dim3 grid(n_tok, gblocks);
    float *sp = (float *)score->ptr; const uint8_t *kp = (const uint8_t *)k->ptr; const float *wp = (const float *)weights->ptr;
    const int32_t *clp = cand_list ? (const int32_t *)cand_list->ptr : NULL, *pp = posd ? (const int32_t *)posd->ptr : NULL;
    if (n_head == 32u) v41_indexer_score_mma_kernel<32u><<<grid, 256, 0, ds4_current_stream()>>>(sp, qi, qe, kp, wp, clp, pos0, ng, ratio, pp, cand_bs, cand_cap, tiles);
    else               v41_indexer_score_mma_kernel<64u><<<grid, 256, 0, ds4_current_stream()>>>(sp, qi, qe, kp, wp, clp, pos0, ng, ratio, pp, cand_bs, cand_cap, tiles);
    return cuda_ok(cudaGetLastError(), "v41 indexer score mma");
}

/* ---- the native MXFP4 arm (kind::mxf4 block-scale, sm_120a+) -------------
 * The indexer's operands are already MXFP4: q is act_quant mode 1 output
 * (e2m1 x ue8m0 per 32 dims) and the index K cache stores e2m1 nibbles plus
 * E8M0 block scales (ds41_kvfmt.h). The int8 arm above turns both into s8
 * mantissas and applies the block scales itself; this arm hands the raw
 * nibbles and the scale bytes to the hardware, which applies one scale per
 * 32-dim chunk (PTX ISA 9.5 9.7.16.3 and Table 44: scale_A is M x 2 and
 * scale_B 2 x N for .scale_vec::2X, exactly the two 32-dim blocks of one
 * k=64 MMA). One instruction covers 64 dims against the int8 arm's 32, so
 * 8192 MACs per instruction, and the nibble LUT and the explicit scale
 * multiplies both vanish.
 *
 * Numerics: the hardware multiplies each chunk by its own ue8m0 scale and
 * accumulates in f32 in an implementation-chosen order, so a per-head dot is
 * not bit-equal to the int8 arm's exact int32 sum. bf16 rounding, relu, the
 * weight multiply and the head-order sum are unchanged code below, so the
 * gate is final output against the measured noise floor, never bit identity.
 *
 * Only a build defining DS4_CUDA_HAVE_MXF4 compiles it (the Makefile ties
 * that define to -gencode arch=compute_121a; ds4_cuda.cu's RR note records
 * why -arch=sm_121a alone is not enough). */
#if defined(DS4_CUDA_HAVE_MXF4)
#define V41_IDX_MXF4 1
#else
#define V41_IDX_MXF4 0
#endif

#if V41_IDX_MXF4

/* q (f32, the act_quant output) -> per (head, 32-dim block) the original
 * e2m1 nibbles plus the ue8m0 scale byte, the two halves the block-scaled
 * MMA consumes. The scale is recovered exactly as act_quant mode 1 chose it
 * (ds41_attn.cuh:184-186), so the nibbles represent the same 32 values the
 * KV pack stored for k (a block whose largest nibble is 3 may re-derive half
 * the scale and double every nibble; the represented values and the dot are
 * unchanged). One warp per (row, head), lane = 4 dims, 8 lanes = one block. */
__global__ static void v41_idxq_mxf4_prep_kernel(int8_t *qn, int8_t *qe, const float *q, uint32_t n_rows, uint32_t n_head) {
    v41_pdl_wait();
    const uint32_t lane = threadIdx.x & 31u, wid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (wid >= n_rows * n_head) return;
    const float *qh = q + (uint64_t)wid * 128u + lane * 4u;
    float v[4];
    #pragma unroll
    for (uint32_t e = 0; e < 4u; e++) v[e] = qh[e];
    float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    for (int o = 1; o < 8; o <<= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));   /* 8 lanes = one 32-dim block */
    const float s = v41_pow2_ceil_log2(fmaxf(a, 6.0f * ldexpf(1.0f, -126)) / 6.0f);
    const uint16_t packed = (uint16_t)(ds4_fp4_f32_to_nibble(v[0] / s) | (ds4_fp4_f32_to_nibble(v[1] / s) << 4) |
                                       (ds4_fp4_f32_to_nibble(v[2] / s) << 8) | (ds4_fp4_f32_to_nibble(v[3] / s) << 12));
    const uint32_t row = wid / n_head, head = wid % n_head;
    *(uint16_t *)(qn + ((uint64_t)row * n_head + head) * 64u + lane * 2u) = packed;
    if ((lane & 7u) == 0u) qe[((uint64_t)row * n_head + head) * 4u + (lane >> 3)] = (int8_t)ds4_e8m0_f32_to_byte(s);
}

#define V41_MXF4_QS_STRIDE 144u   /* shared pitch per head: 64 B of nibbles + pad, 36 words so an A fragment lands on 4*g4+t4 (32 banks) */

/* NH = head count (32/64): MT = NH/16 m-tiles, 2 k halves of 64 dims each.
 * Same block/grid shape and epilogue as the int8 arm; the difference is four
 * MMAs per 8-group tile instead of eight, no LUT, and no scale multiplies. */
template <uint32_t NH>
__global__ static void __launch_bounds__(256) v41_indexer_score_mxf4_kernel(float *score, const int8_t *qn, const int8_t *qe, const uint8_t *k,
                                                                           const float *w, const int32_t *cand, uint32_t pos0, uint32_t ng,
                                                                           uint32_t ratio, const int32_t *posd, uint32_t cand_bs, uint32_t cand_cap,
                                                                           uint32_t tiles) {
    constexpr uint32_t MT = NH / 16u, XS = NH + 1u;
    v41_pdl_wait();
    const uint32_t i = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    if (posd) { pos0 = (uint32_t)posd[0]; ng = (pos0 + gridDim.x) / ratio; }
    const uint32_t vis = (pos0 + i + 1u) / ratio;
    const int32_t *cl = cand ? cand + (uint64_t)i * (1u + cand_cap) : NULL;
    const uint32_t nc = cl ? (uint32_t)cl[0] : 0u;
    const uint32_t ns = cl ? v41_cand_ns(ng, cand_bs, cand_cap) : ng;
    __shared__ __align__(16) int8_t qs[NH * V41_MXF4_QS_STRIDE];
    __shared__ int8_t qe_s[NH * 4u];
    __shared__ float xs[8][8][XS];
    {
        const uint32_t *src = (const uint32_t *)(qn + (uint64_t)i * NH * 64u);
        #pragma unroll
        for (uint32_t t = 0; t < NH / 16u; t++) {   /* NH*16 words; head = x/16, word = x%16 */
            const uint32_t x = threadIdx.x + t * 256u;
            *(uint32_t *)(qs + (x >> 4) * V41_MXF4_QS_STRIDE + (x & 15u) * 4u) = src[x];
        }
        for (uint32_t x = threadIdx.x; x < NH * 4u; x += blockDim.x) qe_s[x] = qe[(uint64_t)i * NH * 4u + x];
    }
    __syncthreads();
    const uint32_t g4 = lane >> 2, t4 = lane & 3u;
    /* A fragments and their scale pairs, loaded once per block and reused by
     * every 8-group tile: reg0 head g dims t*8.., reg1 head g+8, reg2/3 the
     * same heads at dims 32+t*8 (PTX 9.7.16.5.11). */
    uint32_t A[MT][2][4]; uint32_t aS[MT][2];
    float wh[MT][2];
    #pragma unroll
    for (uint32_t mt = 0; mt < MT; mt++) {
        #pragma unroll
        for (uint32_t hh = 0; hh < 2u; hh++) { wh[mt][hh] = w[(uint64_t)i * NH + mt * 16u + g4 + hh * 8u]; }
        #pragma unroll
        for (uint32_t kh = 0; kh < 2u; kh++) {
            const int8_t *h0 = qs + (mt * 16u + g4) * V41_MXF4_QS_STRIDE + kh * 32u + t4 * 4u;
            const int8_t *h1 = h0 + 8u * V41_MXF4_QS_STRIDE;
            A[mt][kh][0] = *(const uint32_t *)h0;
            A[mt][kh][1] = *(const uint32_t *)h1;
            A[mt][kh][2] = *(const uint32_t *)(h0 + 16u);
            A[mt][kh][3] = *(const uint32_t *)(h1 + 16u);
            /* thread-id-a = 0: lanes %4 == 0 supply row g's two chunk scales,
             * lanes %4 == 1 supply row g+8's (byte-id-a = 0 selects the low
             * byte pair, PTX Table 46 and the byte-id figure). */
            const uint32_t hrow = mt * 16u + g4 + ((lane & 3u) == 1u ? 8u : 0u);
            const int8_t *sb = qe_s + hrow * 4u + 2u * kh;
            aS[mt][kh] = (uint32_t)(uint8_t)sb[0] | ((uint32_t)(uint8_t)sb[1] << 8);
        }
    }
    const uint32_t c_base = (blockIdx.y * 8u + warp) * tiles * 8u;
    for (uint32_t t = 0; t < tiles; t++) {
        const uint32_t c0 = c_base + t * 8u;
        if (c0 >= ns) break;
        /* B fragment: this lane's group g4, rows t*8.. and 32+t*8..; its two
         * chunk scales ride in the low byte pair of the selected lane's word. */
        uint32_t gB; const bool liveB = v41_cand_live(cl, nc, c0 + g4, ns, cand_bs, ng, vis, &gB);
        const uint8_t *kr = k + (uint64_t)(liveB ? gB : 0u) * DS4_V41_IDXK_BYTES;
        const uint32_t ekB = *(const uint32_t *)(kr + DS4_V41_IDXK_NIB);
        float C[MT][4];
        #pragma unroll
        for (uint32_t mt = 0; mt < MT; mt++) { C[mt][0] = 0.f; C[mt][1] = 0.f; C[mt][2] = 0.f; C[mt][3] = 0.f; }
        #pragma unroll
        for (uint32_t kh = 0; kh < 2u; kh++) {
            const uint32_t b0 = liveB ? *(const uint32_t *)(kr + kh * 32u + t4 * 4u) : 0u;
            const uint32_t b1 = liveB ? *(const uint32_t *)(kr + kh * 32u + 16u + t4 * 4u) : 0u;
            const uint32_t bS = liveB ? ((ekB >> (16u * kh)) & 0xFFFFu) : 0u;
            #pragma unroll
            for (uint32_t mt = 0; mt < MT; mt++) {
                asm volatile("mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
                             : "+f"(C[mt][0]), "+f"(C[mt][1]), "+f"(C[mt][2]), "+f"(C[mt][3])
                             : "r"(A[mt][kh][0]), "r"(A[mt][kh][1]), "r"(A[mt][kh][2]), "r"(A[mt][kh][3]),
                               "r"(b0), "r"(b1), "r"(aS[mt][kh]), "r"(bS));
            }
        }
        /* Tail, the int8 arm's code: bf16(dot) -> relu -> bf16(* w_h) -> shared
         * transpose, one lane per group sums in head order -> bf16. */
        #pragma unroll
        for (uint32_t mt = 0; mt < MT; mt++) {
            #pragma unroll
            for (uint32_t c = 0; c < 4u; c++) {
                const uint32_t hh = c >> 1, h = mt * 16u + g4 + hh * 8u, gl = t4 * 2u + (c & 1u);
                const float dd = fmaxf(v41_bf16r(C[mt][c]), 0.0f);
                xs[warp][gl][h] = v41_bf16r(dd * wh[mt][hh]);
            }
        }
        __syncwarp();
        if (lane < 8u) {
            uint32_t gW; const bool liveW = v41_cand_live(cl, nc, c0 + lane, ns, cand_bs, ng, vis, &gW);
            float acc = 0.f;
            #pragma unroll 8
            for (uint32_t h = 0; h < NH; h++) acc += xs[warp][lane][h];
            if (c0 + lane < ns) score[(uint64_t)i * ns + c0 + lane] = liveW ? v41_bf16r(acc) : -INFINITY;
        }
        __syncwarp();
    }
}

static int v41_indexer_score_mxf4_launch(ds4_gpu_tensor *score, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *weights,
                                         const ds4_gpu_tensor *cand_list, uint32_t cand_bs, uint32_t cand_cap, uint32_t n_tok, uint32_t pos0,
                                         uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio, const ds4_gpu_tensor *posd) {
    if ((n_head != 32u && n_head != 64u) || dk != 128u) {
        fprintf(stderr, "ds4: [ds41] indexer MXFP4 score is compiled for 32/64 heads x 128 dims (metadata %u x %u)\n", n_head, dk); return 0;
    }
    if (!v41_idx_grow((uint64_t)n_tok * v41_idxq_row_bytes(n_head))) return 0;
    int8_t *qn = (int8_t *)g_v41_idxq.p;
    int8_t *qe = qn + (uint64_t)n_tok * n_head * 64u;
    v41_idxq_mxf4_prep_kernel<<<(unsigned)((n_tok * n_head + 7u) / 8u), 256, 0, ds4_current_stream()>>>(qn, qe, (const float *)q->ptr, n_tok, n_head);
    if (!cuda_ok(cudaGetLastError(), "v41 indexer q mxf4 prep")) return 0;
    const uint32_t ns = cand_list ? v41_cand_ns(ng, cand_bs, cand_cap) : ng;
    uint32_t tiles = 1, gblocks = 1;
    v41_idx_mma_grid(ns, &tiles, &gblocks);
    static int s_once = 0;
    if (!s_once) { s_once = 1; fprintf(stderr, "ds4: [ds41] indexer score on MXFP4 tensor cores: first n_tok %u ns %u heads %u x %u dims, tiles %u, grid (%u, %u), SM %d\n", n_tok, ns, n_head, dk, tiles, n_tok, gblocks, v41_idx_sm()); }
    if (gblocks > 65535u) { fprintf(stderr, "ds4: [ds41] indexer MXFP4 grid.y %u over CUDA's cap (ns %u)\n", gblocks, ns); return 0; }
    const dim3 grid(n_tok, gblocks);
    float *sp = (float *)score->ptr; const uint8_t *kp = (const uint8_t *)k->ptr; const float *wp = (const float *)weights->ptr;
    const int32_t *clp = cand_list ? (const int32_t *)cand_list->ptr : NULL, *pp = posd ? (const int32_t *)posd->ptr : NULL;
    if (n_head == 32u) v41_indexer_score_mxf4_kernel<32u><<<grid, 256, 0, ds4_current_stream()>>>(sp, qn, qe, kp, wp, clp, pos0, ng, ratio, pp, cand_bs, cand_cap, tiles);
    else               v41_indexer_score_mxf4_kernel<64u><<<grid, 256, 0, ds4_current_stream()>>>(sp, qn, qe, kp, wp, clp, pos0, ng, ratio, pp, cand_bs, cand_cap, tiles);
    return cuda_ok(cudaGetLastError(), "v41 indexer score mxf4");
}

#else  /* !V41_IDX_MXF4 */

/* The arm exists only for a Blackwell build; asking for it elsewhere falls
 * back to the int8 arm with a loud line rather than failing the forward. */
static int v41_indexer_score_mxf4_launch(ds4_gpu_tensor *score, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *weights,
                                         const ds4_gpu_tensor *cand_list, uint32_t cand_bs, uint32_t cand_cap, uint32_t n_tok, uint32_t pos0,
                                         uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio, const ds4_gpu_tensor *posd) {
    static int s_once = 0;
    if (!s_once) { s_once = 1; fprintf(stderr, "ds4: [ds41] DS4_DS41_IDX_MMA=2 needs a Blackwell build (DS4_CUDA_HAVE_MXF4 unset); using the int8 arm\n"); }
    return v41_indexer_score_mma_launch(score, q, k, weights, cand_list, cand_bs, cand_cap, n_tok, pos0, ng, n_head, dk, ratio, posd);
}

#endif /* V41_IDX_MXF4 */
