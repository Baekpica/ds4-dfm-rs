#ifndef DS4_BEAM_MMA8W_CUH
#define DS4_BEAM_MMA8W_CUH

#include "glm_vq_mma8.cuh"

/* Warp-specialized FP8 prefill experts for Beam's 12-bit (K4096) layers,
 * computed as tokens x weights^T. A book vector holds k {4q..4q+3,
 * 16+4q..16+4q+3} of one weight row: exactly the B fragment of m16n8k32
 * for column g, so one LDS.64 gather feeds the MMAs with no register
 * shuffles; token (pair) rows are the A operand, one ldmatrix.x4 per 16
 * rows. glmvq_mma8_kernel used the weights as A, and ptxas rebuilt the
 * interleaved A registers with MOVs before its MMAs.
 *
 * A work unit is 128 rows of one 128-token item. Warp w owns rows
 * 16w..16w+15 (two n8 tiles) and all eight m16 token tiles.
 *
 *   warps 0-7   consumers  book gathers + ldmatrix + MMAs, then the epilogue
 *   warps 8-11  producers  cp.async index rows and token rows into a
 *                          three-stage ring (full/empty mbarriers), the item
 *                          record into stage 0, and the next unit's index
 *                          block into L2, one piece per stage
 *
 * Producers load the next unit's item and token rows one unit ahead; the
 * consumers read the item record from the ring and load gains and pair
 * scales before the K loop, so no role waits on a dependent global load at
 * a unit boundary (those waits were the largest stall: -21% on gate).
 * Gate/up activations carry one E4M3 scale per token row and accumulate
 * across all of K; down keeps glmvq_mma8_kernel's per-128-k promotion and
 * produces its output bit for bit. Real-route replay of a 2,048-row chunk
 * (glmvq_mma8_kernel -> these): gate 5.04 -> 3.98 ms, up 5.69 -> 4.41 ms,
 * down 6.06 -> 4.40 ms. */
enum { BEAM_W_BM = 128, BEAM_W_BT = 128, BEAM_W_STAGES = 3, BEAM_W_CONSUMERS = 8, BEAM_W_PRODUCERS = 4,
       BEAM_W_THREADS = (BEAM_W_CONSUMERS + BEAM_W_PRODUCERS) * 32, BEAM_W_PF_FIRST = 4,
       BEAM_W_BOOK = 4096 * GLMVQ_VECTOR, BEAM_W_IX = BEAM_W_BM * GLMVQ_MMA8_INDEX,
       BEAM_W_STAGE = BEAM_W_IX + BEAM_W_BT * GLMVQ_MMA8_ROW + 16,
       BEAM_W_BYTES = BEAM_W_BOOK + BEAM_W_STAGES * BEAM_W_STAGE + BEAM_W_BT * 4 + 2 * BEAM_W_STAGES * 8 };

// One expert matrix for a run of at most BEAM_W_BT grouped pairs; the first
// 16 bytes travel to the consumers in the ring.
struct __align__(16) beam_w_item { uint32_t offset, tokens; const __half *gain; const uint8_t *main; uint64_t pad; };

__device__ __forceinline__ uint32_t beam_w_smem(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }

// The suspend-time hint parks a waiting warp instead of spinning on try_wait.
__device__ __forceinline__ void beam_w_wait(const uint64_t *bar, uint32_t parity) {
    asm volatile("{\n .reg .pred done;\n BEAM_W_WAIT%=:\n mbarrier.try_wait.parity.shared.b64 done, [%0], %1, %2;\n"
        " @!done bra BEAM_W_WAIT%=;\n}" :: "r"(beam_w_smem(bar)), "r"(parity), "r"(0x989680u) : "memory");
}

__device__ __forceinline__ void beam_w_arrive(uint64_t *bar) {
    asm volatile("{\n .reg .b64 state;\n mbarrier.arrive.shared.b64 state, [%0];\n}" :: "r"(beam_w_smem(bar)) : "memory");
}

__device__ __forceinline__ void beam_w_consumers_sync() {
    asm volatile("bar.sync 1, %0;" :: "n"(BEAM_W_CONSUMERS * 32) : "memory");
}

// Token row t's 16-byte chunk c sits at chunk c ^ (t & 7): the eight rows of
// an ldmatrix 8x16B matrix hit distinct banks.
__device__ __forceinline__ uint32_t beam_w_swz(uint32_t t, uint32_t c) { return t * GLMVQ_MMA8_ROW + ((c ^ (t & 7u)) << 4); }

// Byte position of column k inside its row in the ldmatrix k order (see
// beam_w_quant): 4-byte group m of each 32-byte block sits at (m & 1) * 16 + (m >> 1) * 4.
__host__ __device__ __forceinline__ uint32_t beam_w_perm(uint32_t k) {
    const uint32_t m = (k % 32) / 4;
    return k / 32 * 32 + (m & 1u) * 16 + (m >> 1) * 4 + k % 4;
}

__device__ __forceinline__ void beam_w_ldsm4(uint32_t (&a)[4], const void *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(beam_w_smem(p)));
}

__device__ __forceinline__ void beam_w_mma(float (&d)[4], const uint32_t (&a)[4], uint2 b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y));
}

__device__ __forceinline__ float4 beam_w_load4(const float *p) { return *(const float4 *)p; }
__device__ __forceinline__ float4 beam_w_load4(const __nv_bfloat16 *p) {
    const uint2 v = *(const uint2 *)p;
    const __nv_bfloat162 lo = *(const __nv_bfloat162 *)&v.x, hi = *(const __nv_bfloat162 *)&v.y;
    return make_float4(__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi));
}

// x = e4m3 * scale[row], one scale per token row (amax / 448), one block per
// row. Within each 32-byte k block, 4-byte group m lands at byte
// (m & 1) * 16 + (m >> 1) * 4: ldmatrix then hands lane q the bytes 8q..8q+7
// of the block as its A registers 0 and 2, the k order of a book vector.
// Input rows are F32 or BF16 (Ling-3.1's prefill FFN norm).
template <typename T>
__global__ void __launch_bounds__(GLMVQ_THREADS) beam_w_quant(uint8_t *out, float *scales, const T *input,
        uint32_t cols) {
    __shared__ float peak[GLMVQ_THREADS / 32];
    const T *x = input + (uint64_t)blockIdx.x * cols;
    float amax = 0.0f;
    for (uint32_t i = threadIdx.x * 4; i < cols; i += GLMVQ_THREADS * 4) {
        const float4 v = beam_w_load4(x + i);
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    for (uint32_t stride = 16; stride; stride >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, stride)); }
    if ((threadIdx.x & 31) == 0) { peak[threadIdx.x / 32] = amax; }
    __syncthreads();
    amax = 0.0f;
    for (uint32_t w = 0; w < GLMVQ_THREADS / 32; w++) { amax = fmaxf(amax, peak[w]); }
    const float scale = amax / GLMVQ_E4M3_MAX;
    for (uint32_t i = threadIdx.x * 4; i < cols; i += GLMVQ_THREADS * 4) {
        const float4 v = beam_w_load4(x + i);
        const float a = amax > 0.0f ? __fdiv_rn(v.x, scale) : 0.0f, b = amax > 0.0f ? __fdiv_rn(v.y, scale) : 0.0f;
        const float c = amax > 0.0f ? __fdiv_rn(v.z, scale) : 0.0f, d = amax > 0.0f ? __fdiv_rn(v.w, scale) : 0.0f;
        const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
        const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
        const uint32_t m = (i % 32) / 4, at = i / 32 * 32 + (m & 1u) * 16 + (m >> 1) * 4;
        *(uint32_t *)(out + (uint64_t)blockIdx.x * cols + at) = lo | (hi << 16);
    }
    if (threadIdx.x == 0) { scales[blockIdx.x] = scale; }
}

// Per grouped pair: its token row and that row's activation scale.
__global__ void beam_w_pairs(uint32_t *token_of, float *pair_scale, const float *scale, const uint32_t *map,
        uint32_t used, uint32_t pairs) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < pairs) {
        const uint32_t t = map[p] / used;
        token_of[p] = t;
        pair_scale[p] = scale[t];
    }
}

template <glmvq_mma_stage MODE>
__global__ void __launch_bounds__(BEAM_W_THREADS, 1) beam_w_kernel(const beam_w_item *items, uint32_t nitems,
        const uint8_t *book, const uint8_t *act, const uint32_t *token_of, const float *pair_scale,
        __nv_bfloat16 *gate, uint32_t rows, uint32_t cols, uint8_t *mid8, float *mid_scale, float clamp) {
    static_assert(MODE == GLMVQ_MMA_GATE || MODE == GLMVQ_MMA_UP, "gate/up input rows are the token rows");
    constexpr uint32_t BM = BEAM_W_BM, BT = BEAM_W_BT, S = BEAM_W_STAGES, C = BEAM_W_CONSUMERS;
    constexpr uint32_t PT = BEAM_W_PRODUCERS * 32, MT = BT / 16, STB = BEAM_W_STAGE, TQ = PT / 8, TJ = BT / TQ;
    extern __shared__ __align__(16) uint8_t memory[];
    uint8_t *ring = memory + BEAM_W_BOOK;
    uint32_t *amax = (uint32_t *)(ring + S * STB);
    uint64_t *full = (uint64_t *)(amax + BT), *empty = full + S;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    for (uint32_t i = tid * 16; i < BEAM_W_BOOK; i += BEAM_W_THREADS * 16) { glmvq_mma_copy<16>(memory + i, book + i); }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    if (tid == 0) {
        for (uint32_t s = 0; s < S; s++) {
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(beam_w_smem(full + s)), "r"(PT) : "memory");
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(beam_w_smem(empty + s)), "r"(C) : "memory");
        }
    }
    __syncthreads();
    const uint32_t tiles = rows / BM, stages = cols / GLMVQ_MMA8_K, units = nitems * tiles;
    const uint32_t stride = cols * 3 / 16, block = BM * stride;
    if (warp >= C) {
        const uint32_t pt = tid - C * 32, chunk = pt % 8, tq = pt / 8;
        // The next unit's index block streams into L2 in sequential pieces.
        auto prefetch = [&](const uint8_t *p, uint32_t piece, uint32_t pieces) {
            const uint32_t step = (block / pieces + 15) / 16 * 16, off = piece * step;
            if (pt == 0 && p && off < block) { glmvq_mma_prefetch(p + off, min(step, block - off)); }
        };
        // cur: this unit's item index, index block and this thread's token rows.
        uint32_t ci = min(blockIdx.x, units - 1) / tiles;
        beam_w_item cur = items[ci];
        const uint8_t *cm = cur.main + (uint64_t)(blockIdx.x % tiles) * block;
        uint32_t src[TJ];
        #pragma unroll
        for (uint32_t j = 0; j < TJ; j++) {
            const uint32_t t = tq + TQ * j;
            src[j] = token_of[cur.offset + (t < cur.tokens ? t : 0)];
        }
        for (uint32_t piece = 0; piece < BEAM_W_PF_FIRST; piece++) {
            prefetch(blockIdx.x < units ? cm : nullptr, piece, BEAM_W_PF_FIRST);
        }
        uint32_t ringc = 0;
        for (uint32_t work = blockIdx.x; work < units; work += gridDim.x) {
            const uint32_t next = work + gridDim.x, nt = cur.tokens, padded = (nt + 15) / 16 * 16;
            const uint32_t ni = min(next, units - 1) / tiles;
            const beam_w_item nxt = items[ni];
            const uint8_t *nm = nullptr;
            uint32_t nsrc[TJ];
            for (uint32_t kt = 0; kt < stages; kt++, ringc++) {
                const uint32_t s = ringc % S, use = ringc / S;
                if (use) { beam_w_wait(empty + s, (use - 1) & 1); }
                if (kt == 1) {
                    // nxt arrived during stage 0: its block and token rows load now.
                    nm = next < units ? nxt.main + (uint64_t)(next % tiles) * block : nullptr;
                    #pragma unroll
                    for (uint32_t j = 0; j < TJ; j++) {
                        const uint32_t t = tq + TQ * j;
                        nsrc[j] = token_of[nxt.offset + (t < nxt.tokens ? t : 0)];
                    }
                }
                if (kt >= 1) { prefetch(nm, kt - 1, stages - 1); }
                uint8_t *dst = ring + s * STB;
                const uint8_t *from = cm + kt * GLMVQ_MMA8_INDEX;
                for (uint32_t i = pt; i < BM * 3; i += PT) {
                    const uint32_t r = i / 3, part = i % 3;
                    glmvq_mma_copy<8>(dst + r * GLMVQ_MMA8_INDEX + part * 8, from + (uint64_t)r * stride + part * 8);
                }
                // Zero rows pad the last m16 tile.
                #pragma unroll
                for (uint32_t j = 0; j < TJ; j++) {
                    const uint32_t t = tq + TQ * j;
                    if (t < padded) {
                        glmvq_mma_pad(dst + BEAM_W_IX + beam_w_swz(t, chunk),
                            act + (uint64_t)src[j] * cols + kt * GLMVQ_MMA8_K + chunk * 16, t < nt ? 16 : 0);
                    }
                }
                // The item record (offset, tokens, gain) rides in stage 0's slot.
                if (kt == 0 && pt == 0) { glmvq_mma_copy<16>(dst + STB - 16, items + ci); }
                asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(beam_w_smem(full + s)) : "memory");
            }
            cur = nxt; ci = ni; cm = nm;
            #pragma unroll
            for (uint32_t j = 0; j < TJ; j++) { src[j] = nsrc[j]; }
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
        return;
    }
    // MMA fragment coordinates: D rows (tokens) g and g + 8, columns (weight rows) 2q, 2q + 1.
    const uint32_t g = lane >> 2, q = lane & 3u, r0 = warp * 16 + g;
    // ldmatrix.x4 row address of this lane: matrix lane / 8 = (token half, k half).
    const uint32_t lt = (lane & 7u) + ((lane >> 3) & 1u) * 8, lk = lane >> 4;
    uint32_t ringc = 0;
    for (uint32_t work = blockIdx.x; work < units; work += gridDim.x) {
        const uint32_t row = (work % tiles) * BM;
        // The item record arrives with stage 0.
        beam_w_wait(full + ringc % S, (ringc / S) & 1);
        const uint4 rec = *(const uint4 *)(ring + (ringc % S) * STB + STB - 16);
        const uint32_t offset = rec.x, nt = rec.y, nmt = (nt + 15) / 16;
        const __half *gain = (const __half *)((uint64_t)rec.z | (uint64_t)rec.w << 32);
        // Gains (this lane's columns) and pair scales (its D rows) load now
        // and are first used after the K loop.
        const uint32_t c0 = row + warp * 16 + 2 * q;
        const __half2 hg0 = *(const __half2 *)(gain + c0), hg1 = *(const __half2 *)(gain + c0 + 8);
        float sv[MT][2];
        #pragma unroll
        for (uint32_t m = 0; m < MT; m++) {
            #pragma unroll
            for (uint32_t h = 0; h < 2; h++) {
                const uint32_t t = m * 16 + h * 8 + g;
                sv[m][h] = pair_scale[offset + (t < nt ? t : 0)];
            }
        }
        // acc[m][n]: token tile m, weight rows 16w + 8n + (0..7).
        float acc[MT][2][4] = {};
        auto run = [&](auto count) {
            constexpr uint32_t NM = decltype(count)::value;
            for (uint32_t kt = 0; kt < stages; kt++, ringc++) {
                const uint32_t s = ringc % S;
                beam_w_wait(full + s, (ringc / S) & 1);
                const uint8_t *stage = ring + s * STB;
                uint32_t w[2][6];
                #pragma unroll
                for (uint32_t n = 0; n < 2; n++) {
                    #pragma unroll
                    for (uint32_t k = 0; k < 3; k++) {
                        const uint2 v = *(const uint2 *)(stage + (r0 + 8 * n) * GLMVQ_MMA8_INDEX + k * 8);
                        w[n][2 * k] = v.x; w[n][2 * k + 1] = v.y;
                    }
                }
                const uint8_t *tokens = stage + BEAM_W_IX;
                auto step = [&](auto kk) {
                    constexpr uint32_t K = decltype(kk)::value;
                    const uint2 b0 = *(const uint2 *)(memory + glmvq_mma8_code<K>(w[0], q) * 8);
                    const uint2 b1 = *(const uint2 *)(memory + glmvq_mma8_code<K>(w[1], q) * 8);
                    #pragma unroll
                    for (uint32_t m = 0; m < NM; m++) {
                        uint32_t a[4];
                        beam_w_ldsm4(a, tokens + beam_w_swz(m * 16 + lt, 2 * K + lk));
                        beam_w_mma(acc[m][0], a, b0);
                        beam_w_mma(acc[m][1], a, b1);
                    }
                };
                step(std::integral_constant<uint32_t, 0>{});
                step(std::integral_constant<uint32_t, 1>{});
                step(std::integral_constant<uint32_t, 2>{});
                step(std::integral_constant<uint32_t, 3>{});
                // The stage's operands are in registers: free the slot.
                __syncwarp();
                if (lane == 0) { beam_w_arrive(empty + s); }
            }
        };
        // Exact m16 tile counts (padding MMAs are not free).
        auto pick = [&](auto self, auto n) {
            constexpr uint32_t N = decltype(n)::value;
            if constexpr (N >= MT) { run(std::integral_constant<uint32_t, MT>{}); }
            else if (nmt == N) { run(n); }
            else { self(self, std::integral_constant<uint32_t, N + 1>{}); }
        };
        pick(pick, std::integral_constant<uint32_t, 1>{});
        const float2 f0 = __half22float2(hg0), f1 = __half22float2(hg1);
        const float gn[2][2] = {{f0.x, f0.y}, {f1.x, f1.y}};
        if constexpr (MODE == GLMVQ_MMA_GATE) {
            #pragma unroll
            for (uint32_t m = 0; m < MT; m++) {
                if (m >= nmt) { break; }
                #pragma unroll
                for (uint32_t h = 0; h < 2; h++) {
                    const uint32_t t = m * 16 + h * 8 + g;
                    if (t >= nt) { continue; }
                    const uint64_t base = (uint64_t)(offset + t) * rows + c0;
                    #pragma unroll
                    for (uint32_t n = 0; n < 2; n++) {
                        *(__nv_bfloat162 *)(gate + base + 8 * n) = __floats2bfloat162_rn(
                            glmvq_round(glmvq_mul(acc[m][n][2 * h] * sv[m][h], gn[n][0])),
                            glmvq_round(glmvq_mul(acc[m][n][2 * h + 1] * sv[m][h], gn[n][1])));
                    }
                }
            }
        } else {
            // Up: SwiGLU on the BF16 gate, then E4M3 rows for down with one
            // scale per (token, 128 rows). Gate pairs of four token tiles
            // load before any use.
            for (uint32_t i = tid; i < BT; i += C * 32) { amax[i] = 0; }
            beam_w_consumers_sync();
            #pragma unroll
            for (uint32_t half = 0; half < MT; half += MT / 2) {
                __nv_bfloat162 gv[MT / 2][2][2];
                #pragma unroll
                for (uint32_t mm = 0; mm < MT / 2; mm++) {
                    #pragma unroll
                    for (uint32_t h = 0; h < 2; h++) {
                        const uint32_t t = (half + mm) * 16 + h * 8 + g, tt = t < nt ? t : 0;
                        const uint64_t base = (uint64_t)(offset + tt) * rows + c0;
                        #pragma unroll
                        for (uint32_t n = 0; n < 2; n++) { gv[mm][h][n] = *(const __nv_bfloat162 *)(gate + base + 8 * n); }
                    }
                }
                #pragma unroll
                for (uint32_t mm = 0; mm < MT / 2; mm++) {
                    const uint32_t m = half + mm;
                    if (m >= nmt) { break; }
                    #pragma unroll
                    for (uint32_t h = 0; h < 2; h++) {
                        float peak = 0.0f;
                        #pragma unroll
                        for (uint32_t n = 0; n < 2; n++) {
                            const float2 x = __bfloat1622float2(gv[mm][h][n]);
                            const float y0 = glmvq_round(glmvq_mul(acc[m][n][2 * h] * sv[m][h], gn[n][0]));
                            const float y1 = glmvq_round(glmvq_mul(acc[m][n][2 * h + 1] * sv[m][h], gn[n][1]));
                            const float v0 = glmvq_swiglu_value<GLMVQ_CLAMP_SILU>(x.x, y0, clamp);
                            const float v1 = glmvq_swiglu_value<GLMVQ_CLAMP_SILU>(x.y, y1, clamp);
                            acc[m][n][2 * h] = v0;
                            acc[m][n][2 * h + 1] = v1;
                            peak = fmaxf(peak, fmaxf(fabsf(v0), fabsf(v1)));
                        }
                        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 1));
                        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 2));
                        const uint32_t t = m * 16 + h * 8 + g;
                        if (q == 0 && t < nt) { atomicMax(amax + t, __float_as_uint(peak)); }
                    }
                }
            }
            beam_w_consumers_sync();
            #pragma unroll
            for (uint32_t m = 0; m < MT; m++) {
                if (m >= nmt) { break; }
                #pragma unroll
                for (uint32_t h = 0; h < 2; h++) {
                    const uint32_t t = m * 16 + h * 8 + g;
                    if (t >= nt) { continue; }
                    const float peak = __uint_as_float(amax[t]), scale = peak / GLMVQ_E4M3_MAX;
                    #pragma unroll
                    for (uint32_t n = 0; n < 2; n++) {
                        const float a = peak > 0.0f ? __fdiv_rn(acc[m][n][2 * h], scale) : 0.0f;
                        const float b = peak > 0.0f ? __fdiv_rn(acc[m][n][2 * h + 1], scale) : 0.0f;
                        // Down reads these rows with ldmatrix: store them in its k order.
                        *(uint16_t *)(mid8 + (uint64_t)(offset + t) * rows + beam_w_perm(c0 + 8 * n)) =
                            (uint16_t)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
                    }
                    if (q == 0 && warp == 0) {
                        mid_scale[(uint64_t)(offset + t) * (rows / GLMVQ_MMA8_K) + row / GLMVQ_MMA8_K] = scale;
                    }
                }
            }
            // amax is rewritten by the next unit only after every consumer read it.
            beam_w_consumers_sync();
        }
    }
}

/* Down: pairs x weights^T over the up epilogue's E4M3 rows (ldmatrix k
 * order, one scale per (pair, 128 columns)). Each 128-k stage sums in a
 * zeroed MMA accumulator, then joins the FP32 total through one FMA with the
 * pair's scale, as glmvq_mma8_kernel does; the epilogue writes BF16
 * round(total * gain) per pair for the reduce. Same roles and ring as the
 * gate/up kernel; the stage also carries the pairs' scales. */
enum { BEAM_W_DOWN_STAGE = BEAM_W_IX + BEAM_W_BT * GLMVQ_MMA8_ROW + BEAM_W_BT * 4 + 16,
       BEAM_W_DOWN_BYTES = BEAM_W_BOOK + BEAM_W_STAGES * BEAM_W_DOWN_STAGE + 2 * BEAM_W_STAGES * 8 };

__global__ void __launch_bounds__(BEAM_W_THREADS, 1) beam_w_down_kernel(const beam_w_item *items, uint32_t nitems,
        const uint8_t *book, const uint8_t *mid8, const float *mid_scale, __nv_bfloat16 *output,
        uint32_t rows, uint32_t cols) {
    constexpr uint32_t BM = BEAM_W_BM, BT = BEAM_W_BT, S = BEAM_W_STAGES, C = BEAM_W_CONSUMERS;
    constexpr uint32_t PT = BEAM_W_PRODUCERS * 32, MT = BT / 16, STB = BEAM_W_DOWN_STAGE, TQ = PT / 8, TJ = BT / TQ;
    constexpr uint32_t ATB = BT * GLMVQ_MMA8_ROW;
    extern __shared__ __align__(16) uint8_t memory[];
    uint8_t *ring = memory + BEAM_W_BOOK;
    uint64_t *full = (uint64_t *)(ring + S * STB), *empty = full + S;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    for (uint32_t i = tid * 16; i < BEAM_W_BOOK; i += BEAM_W_THREADS * 16) { glmvq_mma_copy<16>(memory + i, book + i); }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    if (tid == 0) {
        for (uint32_t s = 0; s < S; s++) {
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(beam_w_smem(full + s)), "r"(PT) : "memory");
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(beam_w_smem(empty + s)), "r"(C) : "memory");
        }
    }
    __syncthreads();
    const uint32_t tiles = rows / BM, stages = cols / GLMVQ_MMA8_K, units = nitems * tiles;
    const uint32_t stride = cols * 3 / 16, block = BM * stride;
    if (warp >= C) {
        const uint32_t pt = tid - C * 32, chunk = pt % 8, tq = pt / 8;
        auto prefetch = [&](const uint8_t *p, uint32_t piece, uint32_t pieces) {
            const uint32_t step = (block / pieces + 15) / 16 * 16, off = piece * step;
            if (pt == 0 && p && off < block) { glmvq_mma_prefetch(p + off, min(step, block - off)); }
        };
        uint32_t ci = min(blockIdx.x, units - 1) / tiles;
        beam_w_item cur = items[ci];
        const uint8_t *cm = cur.main + (uint64_t)(blockIdx.x % tiles) * block;
        for (uint32_t piece = 0; piece < BEAM_W_PF_FIRST; piece++) {
            prefetch(blockIdx.x < units ? cm : nullptr, piece, BEAM_W_PF_FIRST);
        }
        uint32_t ringc = 0;
        for (uint32_t work = blockIdx.x; work < units; work += gridDim.x) {
            const uint32_t next = work + gridDim.x, nt = cur.tokens, padded = (nt + 15) / 16 * 16;
            const uint32_t ni = min(next, units - 1) / tiles;
            const beam_w_item nxt = items[ni];
            const uint8_t *nm = nullptr;
            for (uint32_t kt = 0; kt < stages; kt++, ringc++) {
                const uint32_t s = ringc % S, use = ringc / S;
                if (use) { beam_w_wait(empty + s, (use - 1) & 1); }
                if (kt == 1) { nm = next < units ? nxt.main + (uint64_t)(next % tiles) * block : nullptr; }
                if (kt >= 1) { prefetch(nm, kt - 1, stages - 1); }
                uint8_t *dst = ring + s * STB;
                const uint8_t *from = cm + kt * GLMVQ_MMA8_INDEX;
                for (uint32_t i = pt; i < BM * 3; i += PT) {
                    const uint32_t r = i / 3, part = i % 3;
                    glmvq_mma_copy<8>(dst + r * GLMVQ_MMA8_INDEX + part * 8, from + (uint64_t)r * stride + part * 8);
                }
                // Pair rows (zero rows pad the last m16 tile) and their stage scales.
                #pragma unroll
                for (uint32_t j = 0; j < TJ; j++) {
                    const uint32_t t = tq + TQ * j, pair = cur.offset + (t < nt ? t : 0);
                    if (t < padded) {
                        glmvq_mma_pad(dst + BEAM_W_IX + beam_w_swz(t, chunk),
                            mid8 + (uint64_t)pair * cols + kt * GLMVQ_MMA8_K + chunk * 16, t < nt ? 16 : 0);
                    }
                }
                for (uint32_t t = pt; t < padded; t += PT) {
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::
                        "r"(beam_w_smem(dst + BEAM_W_IX + ATB + t * 4)),
                        "l"(mid_scale + (uint64_t)(cur.offset + (t < nt ? t : 0)) * stages + kt), "r"(t < nt ? 4 : 0)
                        : "memory");
                }
                if (kt == 0 && pt == 0) { glmvq_mma_copy<16>(dst + STB - 16, items + ci); }
                asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(beam_w_smem(full + s)) : "memory");
            }
            cur = nxt; ci = ni; cm = nm;
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
        return;
    }
    const uint32_t g = lane >> 2, q = lane & 3u, r0 = warp * 16 + g;
    const uint32_t lt = (lane & 7u) + ((lane >> 3) & 1u) * 8, lk = lane >> 4;
    uint32_t ringc = 0;
    for (uint32_t work = blockIdx.x; work < units; work += gridDim.x) {
        const uint32_t row = (work % tiles) * BM;
        beam_w_wait(full + ringc % S, (ringc / S) & 1);
        const uint4 rec = *(const uint4 *)(ring + (ringc % S) * STB + STB - 16);
        const uint32_t offset = rec.x, nt = rec.y, nmt = (nt + 15) / 16;
        const __half *gain = (const __half *)((uint64_t)rec.z | (uint64_t)rec.w << 32);
        const uint32_t c0 = row + warp * 16 + 2 * q;
        const __half2 hg0 = *(const __half2 *)(gain + c0), hg1 = *(const __half2 *)(gain + c0 + 8);
        float acc[MT][2][4] = {};
        auto run = [&](auto count) {
            constexpr uint32_t NM = decltype(count)::value;
            for (uint32_t kt = 0; kt < stages; kt++, ringc++) {
                const uint32_t s = ringc % S;
                beam_w_wait(full + s, (ringc / S) & 1);
                const uint8_t *stage = ring + s * STB;
                uint32_t w[2][6];
                #pragma unroll
                for (uint32_t n = 0; n < 2; n++) {
                    #pragma unroll
                    for (uint32_t k = 0; k < 3; k++) {
                        const uint2 v = *(const uint2 *)(stage + (r0 + 8 * n) * GLMVQ_MMA8_INDEX + k * 8);
                        w[n][2 * k] = v.x; w[n][2 * k + 1] = v.y;
                    }
                }
                uint2 b[4][2];
                b[0][0] = *(const uint2 *)(memory + glmvq_mma8_code<0>(w[0], q) * 8);
                b[0][1] = *(const uint2 *)(memory + glmvq_mma8_code<0>(w[1], q) * 8);
                b[1][0] = *(const uint2 *)(memory + glmvq_mma8_code<1>(w[0], q) * 8);
                b[1][1] = *(const uint2 *)(memory + glmvq_mma8_code<1>(w[1], q) * 8);
                b[2][0] = *(const uint2 *)(memory + glmvq_mma8_code<2>(w[0], q) * 8);
                b[2][1] = *(const uint2 *)(memory + glmvq_mma8_code<2>(w[1], q) * 8);
                b[3][0] = *(const uint2 *)(memory + glmvq_mma8_code<3>(w[0], q) * 8);
                b[3][1] = *(const uint2 *)(memory + glmvq_mma8_code<3>(w[1], q) * 8);
                const uint8_t *pairs = stage + BEAM_W_IX;
                const float *scale = (const float *)(stage + BEAM_W_IX + ATB);
                #pragma unroll
                for (uint32_t m = 0; m < NM; m++) {
                    float part[2][4] = {};
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; k++) {
                        uint32_t a[4];
                        beam_w_ldsm4(a, pairs + beam_w_swz(m * 16 + lt, 2 * k + lk));
                        beam_w_mma(part[0], a, b[k][0]);
                        beam_w_mma(part[1], a, b[k][1]);
                    }
                    const float s0 = scale[m * 16 + g], s1 = scale[m * 16 + 8 + g];
                    #pragma unroll
                    for (uint32_t n = 0; n < 2; n++) {
                        acc[m][n][0] = fmaf(part[n][0], s0, acc[m][n][0]);
                        acc[m][n][1] = fmaf(part[n][1], s0, acc[m][n][1]);
                        acc[m][n][2] = fmaf(part[n][2], s1, acc[m][n][2]);
                        acc[m][n][3] = fmaf(part[n][3], s1, acc[m][n][3]);
                    }
                }
                // The stage's operands are in registers: free the slot.
                __syncwarp();
                if (lane == 0) { beam_w_arrive(empty + s); }
            }
        };
        auto pick = [&](auto self, auto n) {
            constexpr uint32_t N = decltype(n)::value;
            if constexpr (N >= MT) { run(std::integral_constant<uint32_t, MT>{}); }
            else if (nmt == N) { run(n); }
            else { self(self, std::integral_constant<uint32_t, N + 1>{}); }
        };
        pick(pick, std::integral_constant<uint32_t, 1>{});
        const float2 f0 = __half22float2(hg0), f1 = __half22float2(hg1);
        const float gn[2][2] = {{f0.x, f0.y}, {f1.x, f1.y}};
        #pragma unroll
        for (uint32_t m = 0; m < MT; m++) {
            if (m >= nmt) { break; }
            #pragma unroll
            for (uint32_t h = 0; h < 2; h++) {
                const uint32_t t = m * 16 + h * 8 + g;
                if (t >= nt) { continue; }
                const uint64_t base = (uint64_t)(offset + t) * rows + c0;
                #pragma unroll
                for (uint32_t n = 0; n < 2; n++) {
                    *(__nv_bfloat162 *)(output + base + 8 * n) = __floats2bfloat162_rn(
                        glmvq_round(glmvq_mul(acc[m][n][2 * h], gn[n][0])),
                        glmvq_round(glmvq_mul(acc[m][n][2 * h + 1], gn[n][1])));
                }
            }
        }
    }
}

static cudaError_t beam_w_down_launch(const beam_w_item *items, uint32_t nitems, const uint8_t *book, const uint8_t *mid8,
        const float *mid_scale, __nv_bfloat16 *output, uint32_t rows, uint32_t cols, uint32_t sms, cudaStream_t stream) {
    static cudaError_t ready = cudaFuncSetAttribute(beam_w_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        BEAM_W_DOWN_BYTES);
    if (ready != cudaSuccess) { return ready; }
    const uint32_t units = nitems * (rows / BEAM_W_BM);
    beam_w_down_kernel<<<min(units, sms), BEAM_W_THREADS, BEAM_W_DOWN_BYTES, stream>>>(items, nitems, book, mid8, mid_scale,
        output, rows, cols);
    return cudaGetLastError();
}

template <glmvq_mma_stage MODE>
static cudaError_t beam_w_launch(const beam_w_item *items, uint32_t nitems, const uint8_t *book, const uint8_t *act,
        const uint32_t *token_of, const float *pair_scale, __nv_bfloat16 *gate, uint32_t rows, uint32_t cols,
        uint8_t *mid8, float *mid_scale, float clamp, uint32_t sms, cudaStream_t stream) {
    static cudaError_t ready = cudaFuncSetAttribute(beam_w_kernel<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        BEAM_W_BYTES);
    if (ready != cudaSuccess) { return ready; }
    const uint32_t units = nitems * (rows / BEAM_W_BM);
    beam_w_kernel<MODE><<<min(units, sms), BEAM_W_THREADS, BEAM_W_BYTES, stream>>>(items, nitems, book, act, token_of,
        pair_scale, gate, rows, cols, mid8, mid_scale, clamp);
    return cudaGetLastError();
}

#endif
