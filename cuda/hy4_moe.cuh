#ifndef DS4_HY4_MOE_CUH
#define DS4_HY4_MOE_CUH

#include "glm_vq_q4p.cuh"

/* Hy4 routed Q4_K / Q5_K experts on FP16 tensor cores (prefill and verify).
 *
 * The compact MMQ worklist scales every k32 partial by its weight and
 * activation scales; at Hy4's ~64 routed rows per expert in a 2,048-row
 * chunk it streamed the expert weights at under half the DRAM rate. As in
 * glmvq_q4p, producer warps dequantize each weight once into FP16 A
 * fragments and consumers run a plain FP16 GEMM, here over the gathered
 * rows routed to one expert:
 *
 *   y[r] = 2^(e-6) * sum_k fp16(fp16(q_k S_s) - M_s) x'_k
 *
 * q has 4 bits (Q4_K) or 5 (Q5_K: the qh bit ORed into the 1024+q magic).
 * Tile: 128 weight rows x 64 routed rows of one expert; the work list keeps
 * an expert's column tiles adjacent so concurrent CTAs share its weight
 * tiles in L2. Each routed row's dot product is independent of the tile it
 * lands in, so the (atomic) routing order does not change any value. */
enum { HY4M_BM = 128, HY4M_BN = 64, HY4M_BK = 64, HY4M_STAGES = 4, HY4M_NT = 4,
       HY4M_CONSUMERS = 256, HY4M_PRODUCERS = 256, HY4M_THREADS = HY4M_CONSUMERS + HY4M_PRODUCERS,
       HY4M_Q4 = 144, HY4M_Q5 = 176 };
static constexpr uint32_t HY4M_A = HY4M_BM * HY4M_BK * sizeof(__half);
static constexpr uint32_t HY4M_B = HY4M_BN * HY4M_BK * sizeof(__half);
static constexpr uint32_t HY4M_SMEM = HY4M_STAGES * (HY4M_A + HY4M_B) + 2 * HY4M_STAGES * sizeof(uint64_t);

// Routed rows grouped by expert: bounds[e]..bounds[e + 1] index `rows`.
__global__ void hy4m_count(uint32_t *count, const int32_t *ids, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { atomicAdd(count + ids[i], 1u); }
}
__global__ void hy4m_scan(uint32_t *bounds, uint32_t *cursor, const uint32_t *count, uint32_t experts) {
    __shared__ uint32_t part[1024];
    const uint32_t t = threadIdx.x;
    uint32_t v = t < experts ? count[t] : 0;
    part[t] = v;
    __syncthreads();
    for (uint32_t d = 1; d < blockDim.x; d <<= 1) {
        const uint32_t add = t >= d ? part[t - d] : 0;
        __syncthreads();
        part[t] += add;
        __syncthreads();
    }
    if (t < experts) {
        bounds[t + 1] = part[t];
        cursor[t] = part[t] - v;
    }
    if (!t) { bounds[0] = 0; }
}
__global__ void hy4m_scatter(uint32_t *rows, uint32_t *cursor, const int32_t *ids, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { rows[atomicAdd(cursor + ids[i], 1u)] = i; }
}
// One item per (expert, 128-row weight tile, 64-row routed tile):
// {expert, weight tile, first sorted row, routed rows in the tile}.
__global__ void hy4m_items(uint4 *items, uint32_t *count, const uint32_t *bounds, uint32_t m_tiles) {
    const uint32_t e = blockIdx.x, first = bounds[e], rows = bounds[e + 1] - first;
    const uint32_t cols = (rows + HY4M_BN - 1) / HY4M_BN, n = cols * m_tiles;
    if (!n) { return; }
    __shared__ uint32_t base;
    if (!threadIdx.x) { base = atomicAdd(count, n); }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const uint32_t j0 = i % cols * HY4M_BN;
        items[base + i] = make_uint4(e, i / cols, first + j0, min((uint32_t)HY4M_BN, rows - j0));
    }
}
// Per sorted row: its prepped source row and output scale, so no tile
// boundary waits on a routing -> scale chain.
__global__ void hy4m_rows(uint32_t *src, float *scale, const uint32_t *sorted, const float *xscale,
        uint32_t n, uint32_t shift) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const uint32_t r = sorted[i] / shift;
        src[i] = r;
        scale[i] = xscale[r];
    }
}

// Q5_K's fifth bit for the nibbles at bits 0-3 and 16-19, as the 16s of 1024+q.
__device__ __forceinline__ uint32_t hy4m_weight(uint32_t v, uint32_t high, const glmvq_q4p_scale &k) {
    uint32_t magic;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(magic) : "r"(v), "r"(0x000F000Fu), "r"(0x64006400u));
    magic |= (high & 0x00010001u) << 4;
    const __half2 w = __hsub2(__hfma2(*(const __half2 *)&magic, k.s, k.bias), k.m);
    return *(const uint32_t *)&w;
}

// try_wait with CUTLASS's suspend hint parks the warp instead of spinning.
__device__ __forceinline__ void hy4m_wait(const uint64_t *bar, uint32_t parity) {
    asm volatile("{\n .reg .pred done;\n HY4M_WAIT%=:\n"
        " mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
        " @!done bra HY4M_WAIT%=;\n}" :: "r"(glmvq_q4p_smem(bar)), "r"(parity), "r"(0x989680u) : "memory");
}

/* out[r][M] for every routed row r: weights [experts][M][K] in `BYTES`-byte
 * super-blocks, xh the prepped source rows (src[s] for sorted row s) and
 * scale[s] its 2^(e-6). The stage loop is unrolled per super-block so every
 * sub-block index is a constant. */
template <uint32_t BYTES>
__global__ void __launch_bounds__(HY4M_THREADS, 1) hy4m_gemm(const uint8_t *weight, const __half *xh,
        const uint32_t *src, const float *scale, float *out, const uint32_t *sorted,
        const uint4 *items, const uint32_t *item_count, uint32_t M, uint32_t K) {
    constexpr uint32_t STAGES = HY4M_STAGES, NT = HY4M_NT, CONSUMERS = HY4M_CONSUMERS, PRODUCERS = HY4M_PRODUCERS;
    constexpr bool Q5 = BYTES == HY4M_Q5;
    extern __shared__ __align__(16) uint8_t hy4m_shared[];
    uint8_t *sa = hy4m_shared, *sb = sa + STAGES * HY4M_A;
    uint64_t *full = (uint64_t *)(sb + STAGES * HY4M_B), *empty = full + STAGES;
    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t n_items = *item_count, blocks = K / 256, steps = K / HY4M_BK;
    const uint64_t row_bytes = (uint64_t)blocks * BYTES, expert_bytes = row_bytes * M;

    if (!tid) {
        for (uint32_t s = 0; s < STAGES; s++) {
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(glmvq_q4p_smem(full + s)), "r"(2u * PRODUCERS) : "memory");
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(glmvq_q4p_smem(empty + s)), "r"(CONSUMERS / 32) : "memory");
        }
    }
    __syncthreads();

    if (tid >= CONSUMERS) {
        // Producer lane pl: fragment item pl (rows 16mt + g and + 8, qs bytes
        // 8c..8c+7 of each stage) and token chunk pl & 7 of routed rows
        // pl / 8 + 32i. Eight producer warps: a 64-column tile gives one
        // stage too little MMA work to hide four warps' dequantization.
        constexpr uint32_t ITEMS = HY4M_BM * 2 / PRODUCERS, ROWS = HY4M_BN * 8 / PRODUCERS;
        const uint32_t pl = tid - CONSUMERS, c8 = 8 * (pl & 3), qs = (Q5 ? 48 : 16) + c8;
        uint32_t slot[ITEMS];
        #pragma unroll
        for (uint32_t i = 0; i < ITEMS; i++) { slot[i] = (pl + PRODUCERS * i) / 32 * 4 * 512 + (pl & 31) * 16; }
        const uint32_t chunk = pl & 7, t0 = pl >> 3;
        const uint32_t b_dst = glmvq_q4p_smem(sb + t0 * 128 + ((chunk ^ glmvq_q4p_swz(t0)) << 4));
        // The next item's record and source rows load one item ahead.
        auto rows_of = [&](const uint4 &item, uint32_t r[ROWS]) {
            #pragma unroll
            for (uint32_t i = 0; i < ROWS; i++) {
                const uint32_t t = t0 + PRODUCERS / 8 * i;
                r[i] = t < item.w ? __ldg(src + item.z + t) : UINT32_MAX;
            }
        };
        uint4 cur = blockIdx.x < n_items ? items[blockIdx.x] : make_uint4(0, 0, 0, 0);
        uint32_t rows_cur[ROWS];
        rows_of(cur, rows_cur);
        uint32_t ring = 0;
        for (uint32_t it = blockIdx.x; it < n_items; it += gridDim.x) {
            const uint32_t next = it + gridDim.x;
            const uint4 nxt = next < n_items ? items[next] : make_uint4(0, 0, 0, 0);
            const uint32_t e = cur.x, m0 = cur.y * HY4M_BM;
            const uint8_t *rowp[ITEMS][2];
            #pragma unroll
            for (uint32_t i = 0; i < ITEMS; i++) {
                const uint32_t f = pl + PRODUCERS * i, mt = f >> 5, g = (f >> 2) & 7;
                rowp[i][0] = weight + e * expert_bytes + (uint64_t)(m0 + 16 * mt + g) * row_bytes;
                rowp[i][1] = rowp[i][0] + 8 * row_bytes;
            }
            const __half *b_src[ROWS];
            #pragma unroll
            for (uint32_t i = 0; i < ROWS; i++) {
                b_src[i] = rows_cur[i] != UINT32_MAX ? xh + (uint64_t)rows_cur[i] * K + chunk * 8 : nullptr;
            }
            auto raw = [&](uint32_t kt, uint2 w[ITEMS][2]) {
                const uint32_t off = (kt >> 2) * BYTES + qs + 32 * (kt & 3);
                #pragma unroll
                for (uint32_t i = 0; i < ITEMS; i++) {
                    w[i][0] = __ldg((const uint2 *)(rowp[i][0] + off));
                    w[i][1] = __ldg((const uint2 *)(rowp[i][1] + off));
                }
            };
            // Super-block header (d, dmin, scales) and, for Q5_K, qh bytes 8c..8c+7.
            auto header = [&](uint32_t block, uint4 h[ITEMS][2], uint2 q[ITEMS][2]) {
                #pragma unroll
                for (uint32_t i = 0; i < ITEMS; i++) {
                    #pragma unroll
                    for (uint32_t r = 0; r < 2; r++) {
                        h[i][r] = __ldg((const uint4 *)(rowp[i][r] + block * BYTES));
                        if constexpr (Q5) { q[i][r] = __ldg((const uint2 *)(rowp[i][r] + block * BYTES + 16 + c8)); }
                    }
                }
            };
            // Whole super-blocks of this lane's rows into L2 two blocks ahead:
            // the stages read 32-byte pieces of rows 1 KB apart.
            // The next item's first two blocks follow during this item's last.
            auto prefetch_rows = [&](const uint4 &item, uint32_t block) {
                if ((pl & 3) || block >= blocks || !item.w) { return; }
                #pragma unroll
                for (uint32_t i = 0; i < ITEMS; i++) {
                    const uint32_t f = pl + PRODUCERS * i, mt = f >> 5, g = (f >> 2) & 7;
                    const uint8_t *row = weight + item.x * expert_bytes +
                                         (uint64_t)(item.y * HY4M_BM + 16 * mt + g) * row_bytes + block * BYTES;
                    #pragma unroll
                    for (uint32_t r = 0; r < 2; r++) {
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::
                            "l"(row + r * 8 * row_bytes), "r"(BYTES) : "memory");
                    }
                }
            };
            auto prefetch = [&](uint32_t block) { prefetch_rows(cur, block); };
            if (it == blockIdx.x) {
                prefetch(0);
                prefetch(1);
            }
            // Raw bytes two stages ahead in four sets indexed by the (constant)
            // stage, so no rotation move waits on a load just issued; one
            // stage of a 64-column tile is too little MMA work to cover a DRAM
            // round trip. Headers load one super-block ahead.
            uint2 w[4][ITEMS][2], q[ITEMS][2] = {}, qn[ITEMS][2] = {};
            uint4 h[ITEMS][2], hn[ITEMS][2];
            raw(0, w[0]);
            raw(1, w[1]);
            header(0, h, q);
            for (uint32_t kb = 0; kb < blocks; kb++) {
                prefetch(kb + 2);
                if (kb + 2 >= blocks) { prefetch_rows(nxt, kb + 2 - blocks); }
                if (kb + 1 < blocks) { header(kb + 1, hn, qn); }
                #pragma unroll
                for (uint32_t p = 0; p < 4; p++, ring++) {
                    const uint32_t kt = 4 * kb + p, s = ring % STAGES, use = ring / STAGES;
                    if (kt + 2 < steps) { raw(kt + 2, w[(p + 2) & 3]); }
                    if (kt == 3) { rows_of(nxt, rows_cur); }
                    if (use) { hy4m_wait(empty + s, (use - 1) & 1); }
                    #pragma unroll
                    for (uint32_t i = 0; i < ROWS; i++) {
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                            "r"(b_dst + s * HY4M_B + i * PRODUCERS / 8 * 128),
                            "l"(b_src[i] ? b_src[i] + kt * HY4M_BK : xh), "r"(b_src[i] ? 16 : 0) : "memory");
                    }
                    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(glmvq_q4p_smem(full + s)) : "memory");
                    // Sub-block 2p + n sits in the low (n = 0) or high (n = 1)
                    // nibbles; its fifth bits are bit 2p + n of the qh bytes.
                    #pragma unroll
                    for (uint32_t i = 0; i < ITEMS; i++) {
                        #pragma unroll
                        for (uint32_t n = 0; n < 2; n++) {
                            const glmvq_q4p_scale k0 = glmvq_q4p_scales(h[i][0], 2 * p + n);
                            const glmvq_q4p_scale k1 = glmvq_q4p_scales(h[i][1], 2 * p + n);
                            #pragma unroll
                            for (uint32_t kh = 0; kh < 2; kh++) {
                                const uint2 r0 = w[p][i][0], r1 = w[p][i][1];
                                const uint32_t u0 = kh ? r0.y : r0.x, u1 = kh ? r1.y : r1.x;
                                const uint32_t h0 = Q5 ? (kh ? q[i][0].y : q[i][0].x) >> (2 * p + n) : 0;
                                const uint32_t h1 = Q5 ? (kh ? q[i][1].y : q[i][1].x) >> (2 * p + n) : 0;
                                uint4 a;
                                a.x = hy4m_weight(u0 >> (4 * n), h0, k0);
                                a.z = hy4m_weight(u0 >> (8 + 4 * n), h0 >> 8, k0);
                                a.y = hy4m_weight(u1 >> (4 * n), h1, k1);
                                a.w = hy4m_weight(u1 >> (8 + 4 * n), h1 >> 8, k1);
                                *(uint4 *)(sa + s * HY4M_A + slot[i] + (n * 2 + kh) * 512) = a;
                            }
                        }
                    }
                    glmvq_q4p_arrive(full + s);
                }
                #pragma unroll
                for (uint32_t i = 0; i < ITEMS; i++) {
                    h[i][0] = hn[i][0];
                    h[i][1] = hn[i][1];
                    q[i][0] = qn[i][0];
                    q[i][1] = qn[i][1];
                }
            }
            cur = nxt;
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
        return;
    }

    const uint32_t g = lane >> 2, c = lane & 3, wm = warp & 3, wn = warp >> 2;
    uint32_t ring = 0;
    for (uint32_t it = blockIdx.x; it < n_items; it += gridDim.x) {
        const uint4 item = items[it];
        const uint32_t m0 = item.y * HY4M_BM;
        // Output rows and scales of this lane's columns, loaded during the MMAs.
        uint32_t dst_row[NT][2];
        float dst_scale[NT][2];
        #pragma unroll
        for (uint32_t j = 0; j < NT; j++) {
            #pragma unroll
            for (uint32_t x = 0; x < 2; x++) {
                const uint32_t t = wn * NT * 8 + 8 * j + 2 * c + x;
                dst_row[j][x] = t < item.w ? __ldg(sorted + item.z + t) : UINT32_MAX;
                dst_scale[j][x] = t < item.w ? __ldg(scale + item.z + t) : 0.0f;
            }
        }
        float acc[2][NT][4] = {};
        for (uint32_t kt = 0; kt < steps; kt++, ring++) {
            const uint32_t stage = ring % STAGES;
            hy4m_wait(full + stage, (ring / STAGES) & 1);
            const uint8_t *as = sa + stage * HY4M_A + 2 * wm * 4 * 512 + lane * 16;
            #pragma unroll
            for (uint32_t n = 0; n < 2; n++) {
                uint4 a[2][2];
                #pragma unroll
                for (uint32_t mt = 0; mt < 2; mt++) {
                    #pragma unroll
                    for (uint32_t kh = 0; kh < 2; kh++) { a[mt][kh] = *(const uint4 *)(as + (mt * 4 + n * 2 + kh) * 512); }
                }
                uint32_t b[NT][4];
                #pragma unroll
                for (uint32_t j = 0; j < NT; j++) {
                    const uint32_t t = wn * NT * 8 + 8 * j + g;
                    *(uint4 *)b[j] = *(const uint4 *)(sb + stage * HY4M_B + t * 128 + (((4 * n + c) ^ glmvq_q4p_swz(t)) << 4));
                }
                #pragma unroll
                for (uint32_t kh = 0; kh < 2; kh++) {
                    #pragma unroll
                    for (uint32_t j = 0; j < NT; j++) {
                        #pragma unroll
                        for (uint32_t mt = 0; mt < 2; mt++) { glmvq_q4p_mma(acc[mt][j], a[mt][kh], b[j][2 * kh], b[j][2 * kh + 1]); }
                    }
                }
            }
            __syncwarp();
            if (!lane) { glmvq_q4p_arrive(empty + stage); }
        }
        #pragma unroll
        for (uint32_t j = 0; j < NT; j++) {
            #pragma unroll
            for (uint32_t x = 0; x < 2; x++) {
                if (dst_row[j][x] == UINT32_MAX) { continue; }
                #pragma unroll
                for (uint32_t mt = 0; mt < 2; mt++) {
                    float *dst = out + (uint64_t)dst_row[j][x] * M + m0 + wm * 32 + 16 * mt + g;
                    dst[0] = acc[mt][j][x] * dst_scale[j][x];
                    dst[8] = acc[mt][j][x + 2] * dst_scale[j][x];
                }
            }
        }
    }
}

/* Hy4's SwiGLU (hy4::swiglu: limit 10, BF16 boundaries) straight into the
 * down's FP16 operand in the q4p element order. |SwiGLU| < 2^7, so every
 * row keeps the 2^-6 output scale. */
__global__ void __launch_bounds__(GLMVQ_Q4P_PREP) hy4m_swiglu_prep(const float *gate, const float *up,
        __half *xh, float *xscale, uint32_t cols) {
    constexpr float limit = 10.0f;
    const uint32_t row = blockIdx.x;
    const float *gr = gate + (uint64_t)row * cols, *ur = up + (uint64_t)row * cols;
    __half *dst = xh + (uint64_t)row * cols;
    for (uint32_t i = threadIdx.x; i < cols / 4; i += GLMVQ_Q4P_PREP) {
        const float4 g4 = ((const float4 *)gr)[i], u4 = ((const float4 *)ur)[i];
        const float gv[4] = {g4.x, g4.y, g4.z, g4.w}, uv[4] = {u4.x, u4.y, u4.z, u4.w};
        __half h[4];
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            const float g = fminf(__bfloat162float(__float2bfloat16(gv[j])), limit);
            const float u = fmaxf(-limit, fminf(__bfloat162float(__float2bfloat16(uv[j])), limit));
            const float silu = __bfloat162float(__float2bfloat16(g * (1.f / (1.f + expf(-g)))));
            h[j] = __float2half_rn(__bfloat162float(__float2bfloat16(silu * u)));
        }
        *(uint2 *)(dst + 4 * i) = make_uint2(glmvq_q4p_h2(h[0], h[2]), glmvq_q4p_h2(h[1], h[3]));
    }
    if (!threadIdx.x) { xscale[row] = ldexpf(1.0f, -GLMVQ_Q4P_SHIFT); }
}

// Sets *bad when a Q4_K / Q5_K super-block's scales leave the folded FP16 range.
template <uint32_t BYTES>
__global__ void hy4m_check(const uint8_t *weight, uint64_t blocks, uint32_t *bad) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) { return; }
    const uint4 h = *(const uint4 *)(weight + b * BYTES);
    const float d = __half2float(__ushort_as_half(h.x & 65535u)), dmin = __half2float(__ushort_as_half(h.x >> 16));
    float worst = 0.0f, minimum = 0.0f;
    for (uint32_t s = 0; s < 8; s++) {
        float sc, m;
        glmvq_q4_scales(h, s, &sc, &m);
        worst = fmaxf(worst, fabsf(d) * sc);
        minimum = fmaxf(minimum, fabsf(dmin) * m);
    }
    if (!(worst <= GLMVQ_Q4P_SCALE_MAX) || !(minimum <= GLMVQ_Q4P_MIN_MAX)) { atomicOr(bad, 1u); }
}

#endif
