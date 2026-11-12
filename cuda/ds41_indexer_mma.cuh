/* ds41_indexer_mma.cuh — DeepSeek V4.1 (ds41) indexer score on tensor cores,
 * ported from the C engine at /data/YoungAi (commit 7fb26cc), file
 * `src/cuda/cuda_v41_indexer_mma.inc.cu` (whole file, 2026-09-30).
 *
 * Why (measured on the port, 2026-11-12, 64K prefill): a prefill block scores
 * 2,048 rows against every visible group of each index source, so the scalar
 * kernel's cost grows with the context: 71 ms per 2,048-row block at 8K,
 * 827 ms at 60K (35% of the block), 4.8 ns per (row, group).  The engine
 * measured the same shape at 106K: prefill 133.5 -> 96.3 s with this kernel,
 * byte-identical output (fable5.md, 2026-09-30).
 *
 * Numerics (the engine's header, :9-14): index q and k are both an E2M1
 * mantissa times a power-of-two block scale (q: act_quant mode 1, 32 dims per
 * block; k: the packed MXFP4 cache, 32 dims per E8M0 scale).  Doubled, the
 * mantissas are integers |m| <= 12, so one s8 `mma.m16n8k32` is one 32-dim
 * block and its s32 sum is exact (|sum| <= 32 * 144).  The block scale
 * 2^(eq + ek - 2) is exact as well, so each block dot T_b equals the scalar
 * kernel's element-wise f32 sum.  The four blocks add as (T0 + T1) + (T2 + T3):
 * the only rounding point, and the only difference from the scalar kernel,
 * whose xor tree adds across blocks first.  The two disagree on a bf16 dot
 * only when block scales differ by more than 2^9 and the sum sits on a bf16
 * rounding boundary.  relu, the bf16 weight product and the f32 sum over
 * heads in head order are unchanged (transposed through shared memory, one
 * lane per group adds serially).
 *
 * Shape: the head count is a template (32 for V4.1 Flash; 64 compiled too),
 * the key width is 128; other shapes are refused, never a fallback.
 *
 * Port deviations, named:
 *   - the engine keeps one q scratch per decode lane; the port has no lanes
 *     (one static slot, like the candidate scratch in ds41_indexer.cuh);
 *   - dispatch: the engine routes every width here behind --idx-mma (default
 *     off).  The port takes this kernel for eager prefill blocks by default
 *     and keeps the scalar kernel for decode and verify widths; see the
 *     entry in ds41_indexer.cuh.
 *
 * Must follow ds41_indexer.cuh (v41_cand_ns) and ds41_fp8blk.cuh
 * (ds4_e8m0_to_f32).
 */
#pragma once

/* q as integer mantissas: [rows][NH][128] s8, then exponents [rows][NH][4] s8. */
static v41_scratch g_v41_idxq;
static uint64_t v41_idxq_row_bytes(uint32_t n_head) { return (uint64_t)n_head * 128u + (uint64_t)n_head * 4u; }

/* q (f32, value = e2m1 * 2^s) -> per (head, 32-dim block) a unit exponent e
 * and integer mantissas m2 = q * 2^(1 - e), |m2| <= 12 (:26-45).  One warp
 * per head, four dims per lane (the mma A fragment's k layout).  The block
 * peak amax = mmax * 2^s (mmax in 0.5..6) gives frexp's exponent ex and
 * e = ex - 3, the step that keeps every m2 an integer <= 12. */
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
    int ex = 0; (void)frexpf(a, &ex);
    const int e = a > 0.f ? ex - 3 : 0;
    uint32_t packed = 0;
    #pragma unroll
    for (uint32_t kk = 0; kk < 4u; kk++) {
        const int m2 = __float2int_rn(ldexpf(v[kk], 1 - e));   /* an exact integer */
        packed |= ((uint32_t)(uint8_t)(int8_t)m2) << (8u * kk);
    }
    *(uint32_t *)(qi + (uint64_t)wid * 128u + lane * 4u) = packed;
    if ((lane & 7u) == 0u) qe[(uint64_t)wid * 4u + (lane >> 3)] = (int8_t)e;
}

/* Compact item c -> its group g and liveness, the scalar kernel's rule
 * (:48-53): off-list, g >= ng and g >= vis are dead groups. */
__device__ __forceinline__ static bool v41_cand_live(const int32_t *cl, uint32_t nc, uint32_t c, uint32_t ns, uint32_t bs, uint32_t ng, uint32_t vis, uint32_t *g) {
    uint32_t gg = c;
    if (cl) { const uint32_t b = c / bs; gg = b < nc ? (uint32_t)cl[1u + b] * bs + c % bs : ng; }
    *g = gg;
    return c < ns && gg < ng && gg < vis;
}
__device__ __forceinline__ static void v41_mma_s8(int32_t *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    const int32_t z = 0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(z), "r"(z), "r"(z), "r"(z));
}
/* 2^n as f32 bits (a block exponent comes from a normal amax, so it stays in the f32 normal range). */
__device__ __forceinline__ static float v41_pow2i(int n) { uint32_t u = (uint32_t)(n + 127) << 23; float f; memcpy(&f, &u, 4); return f; }

/* Head row pitch of the q mantissas in shared (128 + 16 pad): an A fragment
 * reads 8 rows x 4 words over all 32 banks. */
#define V41_IMMA_QS_STRIDE 144u
/* NH = head count (a multiple of 16): MT = NH/16 m16 tiles.  One block is one
 * query row and 8 warps; a warp scores 8 groups per tile. */
template <uint32_t NH>
__global__ static void __launch_bounds__(256) v41_indexer_score_mma_kernel(float *score, const int8_t *qi, const int8_t *qe, const uint8_t *k,
                                                                          const float *w, const int32_t *cand, uint32_t pos0, uint32_t ng,
                                                                          uint32_t ratio, const int32_t *posd, uint32_t cand_bs, uint32_t cand_cap,
                                                                          uint32_t tiles) {
    constexpr uint32_t MT = NH / 16u, XS = NH + 1u;   /* XS: transposed row pitch (heads + 1: four groups never share a bank) */
    v41_pdl_wait();
    const uint32_t i = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    if (posd) { pos0 = (uint32_t)posd[0]; ng = (pos0 + gridDim.x) / ratio; }
    const uint32_t vis = (pos0 + i + 1u) / ratio;
    const int32_t *cl = cand ? cand + (uint64_t)i * (1u + cand_cap) : NULL;
    const uint32_t nc = cl ? (uint32_t)cl[0] : 0u;
    const uint32_t ns = cl ? v41_cand_ns(ng, cand_bs, cand_cap) : ng;
    __shared__ uint16_t lut[256];                                  /* byte (two nibbles) -> two s8 doubled mantissas, low nibble in the low byte */
    __shared__ __align__(16) int8_t qs[NH * V41_IMMA_QS_STRIDE];   /* this row's NH heads of q mantissas */
    __shared__ float xs[8][8][XS];                                 /* [warp][group][head] bf16(relu(dot) * w), transposed for the head-order sum */
    {
        const uint32_t v = threadIdx.x, lo = v & 15u, hi = v >> 4;
        const int ml = ((lo >> 1) & 3u) ? ((2 + (int)(lo & 1u)) << (((lo >> 1) & 3u) - 1u)) : (int)(lo & 1u);
        const int mh = ((hi >> 1) & 3u) ? ((2 + (int)(hi & 1u)) << (((hi >> 1) & 3u) - 1u)) : (int)(hi & 1u);
        const int sl = (lo & 8u) ? -ml : ml, sh = (hi & 8u) ? -mh : mh;
        lut[v] = (uint16_t)((uint8_t)(int8_t)sl | ((uint16_t)(uint8_t)(int8_t)sh << 8));
        const uint32_t *src = (const uint32_t *)(qi + (uint64_t)i * NH * 128u);
        #pragma unroll
        for (uint32_t t = 0; t < NH / 8u; t++) {   /* NH x 128 B = NH*32 words, NH/8 per thread: word x -> head x/32, dim (x%32)*4 */
            const uint32_t x = threadIdx.x + t * 256u;
            *(uint32_t *)(qs + (x >> 5) * V41_IMMA_QS_STRIDE + (x & 31u) * 4u) = src[x];
        }
    }
    __syncthreads();
    const uint32_t g4 = lane >> 2, t4 = lane & 3u;
    /* The heads this lane owns in the C fragment, mt*16 + g4 (+8): their block exponents (four per u32) and weights. */
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
        /* B fragment: this lane holds group lane/4, dims b*32 + (lane%4)*4 (+16): two bytes of nibbles each. */
        uint32_t gB; const bool liveB = v41_cand_live(cl, nc, c0 + g4, ns, cand_bs, ng, vis, &gB);
        const uint8_t *kr = k + (uint64_t)(liveB ? gB : 0u) * DS4_V41_IDXK_BYTES;
        /* The C fragment's two groups (lane%4)*2 + {0,1}: their E8M0 block scales (the row's last 4 bytes; rows are 8-aligned, one u32). */
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
                /* c0,c1: head g4, groups t4*2+{0,1}; c2,c3: head g4+8.  Scale = 2^(eq-2) * 2^ek, both products exact. */
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
        /* Tail, the official expression: bf16(dot) -> relu -> bf16(* w_h) -> transposed into shared, one lane per group adds in head order -> bf16. */
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

/* The tensor-core score entry, ds4_gpu_v41_indexer_score_tensor's contract
 * (:169-200).  grid = (rows, group blocks); each warp runs `tiles` tiles of 8
 * groups (the row's q mantissas enter shared once per block); tiles is sized
 * from ns and the SM count to cover at least two waves, never hard-coded. */
static int v41_indexer_score_mma_launch(ds4_gpu_tensor *score, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *weights,
                                        const ds4_gpu_tensor *cand_list, uint32_t cand_bs, uint32_t cand_cap, uint32_t n_tok, uint32_t pos0,
                                        uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio, const ds4_gpu_tensor *posd) {
    if ((n_head != 32u && n_head != 64u) || dk != 128u) {
        fprintf(stderr, "ds4: [ds41] the tensor-core indexer score is built for 32/64 heads x 128 dims (metadata %u x %u)\n", n_head, dk);
        return 0;
    }
    int8_t *qi = (int8_t *)v41_grow(&g_v41_idxq, (uint64_t)n_tok * v41_idxq_row_bytes(n_head), "v41 indexer q s8");
    if (!qi) return 0;
    int8_t *qe = qi + (uint64_t)n_tok * n_head * 128u;
    v41_idxq_prep_kernel<<<(unsigned)((n_tok * n_head + 7u) / 8u), 256, 0, ds4_current_stream()>>>(qi, qe, (const float *)q->ptr, n_tok, n_head);
    if (!cuda_ok(cudaGetLastError(), "v41 indexer q prep")) return 0;
    static int s_sm = 0;
    if (!s_sm && cudaDeviceGetAttribute(&s_sm, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) return cuda_ok(cudaGetLastError(), "v41 indexer mma");
    const uint32_t ns = cand_list ? v41_cand_ns(ng, cand_bs, cand_cap) : ng;
    const uint32_t warps = (ns + 7u) / 8u;
    uint32_t tiles = (warps + 16u * (uint32_t)s_sm - 1u) / (16u * (uint32_t)s_sm);   /* one wave = 2 blocks x 8 warps per SM */
    if (tiles < 1u) tiles = 1u;
    if (tiles > 32u) tiles = 32u;
    const uint32_t gblocks = (warps + 8u * tiles - 1u) / (8u * tiles);
    static int s_once = 0;
    if (!s_once) {
        s_once = 1;
        fprintf(stderr, "ds4: [ds41] indexer score on tensor cores: first launch n_tok %u ns %u, %u heads x %u dims, tiles %u, grid (%u, %u), %d SMs\n",
                n_tok, ns, n_head, dk, tiles, n_tok, gblocks, s_sm);
    }
    if (gblocks > 65535u) { fprintf(stderr, "ds4: [ds41] tensor-core indexer grid.y %u over the CUDA limit (ns %u)\n", gblocks, ns); return 0; }
    const dim3 grid(n_tok, gblocks);
    float *sp = (float *)score->ptr; const uint8_t *kp = (const uint8_t *)k->ptr; const float *wp = (const float *)weights->ptr;
    const int32_t *clp = cand_list ? (const int32_t *)cand_list->ptr : NULL, *pp = posd ? (const int32_t *)posd->ptr : NULL;
    if (n_head == 32u) v41_indexer_score_mma_kernel<32u><<<grid, 256, 0, ds4_current_stream()>>>(sp, qi, qe, kp, wp, clp, pos0, ng, ratio, pp, cand_bs, cand_cap, tiles);
    else               v41_indexer_score_mma_kernel<64u><<<grid, 256, 0, ds4_current_stream()>>>(sp, qi, qe, kp, wp, clp, pos0, ng, ratio, pp, cand_bs, cand_cap, tiles);
    return cuda_ok(cudaGetLastError(), "v41 indexer score mma");
}
