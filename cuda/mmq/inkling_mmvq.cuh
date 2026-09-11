#include <cstdlib>
/* Inkling prefill: share expert weights without changing MMVQ reductions.
 * Included by mmvq.cu after its type traits and warp reduction helpers. */
enum {
    IK_MMVQ_COLUMNS = 4,
    IK_MMVQ_ROWS = 2,
    IK_MMVQ_THREADS = 256,
    IK_MMVQ_RESIDENT_THREADS = 1024,
    IK_MMVQ_EXPERTS = 256,
    IK_MMVQ_ASSIGNMENTS = 8192 * 6,
    IK_MMVQ_HIDDEN = 4096,
    IK_MMVQ_MIDDLE = 2048,
};

static __global__ void inkling_bucket_kernel(
        int32_t *counts, int32_t *buckets, const int32_t *ids,
        uint32_t assignments, uint32_t experts) {
    const uint32_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= assignments) { return; }
    const int32_t expert = ids[a];
    // Invalid routes retain the zero written before bucketing.
    if (expert < 0 || (uint32_t)expert >= experts) { return; }
    const uint32_t slot = atomicAdd(counts + expert, 1);
    buckets[(uint64_t)expert * assignments + slot] = a;
}

template<unsigned COLUMNS>
static __global__ void inkling_tiles_kernel(
        int32_t *counts, int32_t *tile_experts, int32_t *tile_starts,
        uint32_t experts) {
    __shared__ uint32_t offsets[IK_MMVQ_EXPERTS];
    if (threadIdx.x == 0) {
        uint32_t total = 0;
        for (uint32_t e = 0; e < experts; e++) {
            offsets[e] = total;
            total += (counts[e] + COLUMNS - 1) / COLUMNS;
        }
        counts[experts] = total;
    }
    __syncthreads();
    const uint32_t e = threadIdx.x;
    if (e >= experts) { return; }
    for (int32_t at = 0; at < counts[e]; at += COLUMNS) {
        const uint32_t tile = offsets[e] + at / COLUMNS;
        tile_experts[tile] = e;
        tile_starts[tile] = at;
    }
}

template<ggml_type TYPE, unsigned WARPS>
__launch_bounds__(WARPS * 32, 1)
static __global__ void inkling_mmvq_kernel(
        const void *weights, const block_q8_1 *x, float *out,
        const int32_t *counts, const int32_t *buckets,
        const int32_t *tile_experts, const int32_t *tile_starts,
        uint32_t m, uint32_t k, uint32_t assignments, uint32_t experts,
        uint32_t used) {
    constexpr unsigned WARP = 32;
    constexpr int QK = ggml_cuda_type_traits<TYPE>::qk;
    constexpr int QI = ggml_cuda_type_traits<TYPE>::qi;
    constexpr int VDR = get_vdr_mmvq(TYPE);
    constexpr auto dot = get_vec_dot_q_cuda(TYPE);
    constexpr unsigned K_STEP = VDR * WARPS * WARP / QI;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid % WARP, warp = tid / WARP;
    const uint32_t row_tiles = m / IK_MMVQ_ROWS;
    const uint64_t jobs = (uint64_t)counts[experts] * row_tiles;
    const uint32_t blocks_per_row = k / QK, x_blocks = k / QK8_1;
    __shared__ float partial[WARPS > 1 ? WARPS - 1 : 1]
                           [IK_MMVQ_COLUMNS][IK_MMVQ_ROWS][WARP];

    for (uint64_t job = blockIdx.x; job < jobs; job += gridDim.x) {
        const uint32_t tile = job / row_tiles;
        const uint32_t row = (job % row_tiles) * IK_MMVQ_ROWS;
        const uint32_t expert = tile_experts[tile], start = tile_starts[tile];
        const uint32_t weight_base = (expert * m + row) * blocks_per_row;
        int32_t selected[IK_MMVQ_COLUMNS];
        #pragma unroll
        for (unsigned c = 0; c < IK_MMVQ_COLUMNS; c++) {
            selected[c] = start + c < (uint32_t)counts[expert]
                ? buckets[(uint64_t)expert * assignments + start + c] : -1;
        }
        float sum[IK_MMVQ_COLUMNS][IK_MMVQ_ROWS] = {};
        // Up retains the original four-warp K partition; down retains one.
        // Token grouping only shares decoding of the same weight payload.
        for (uint32_t bx = tid / (QI / VDR); bx < blocks_per_row; bx += K_STEP) {
            const uint32_t by = bx * (QK / QK8_1), qs = VDR * (tid % (QI / VDR));
            #pragma unroll
            for (unsigned c = 0; c < IK_MMVQ_COLUMNS; c++) {
                if (selected[c] < 0) { continue; }
                const block_q8_1 *xr = x + (uint64_t)(selected[c] / used) * x_blocks + by;
                #pragma unroll
                for (unsigned r = 0; r < IK_MMVQ_ROWS; r++) {
                    sum[c][r] += dot(weights, xr, weight_base + r * blocks_per_row + bx, qs);
                }
            }
        }
        if constexpr (WARPS > 1) {
            if (warp > 0) {
                #pragma unroll
                for (unsigned c = 0; c < IK_MMVQ_COLUMNS; c++) {
                    #pragma unroll
                    for (unsigned r = 0; r < IK_MMVQ_ROWS; r++) {
                        partial[warp - 1][c][r][lane] = sum[c][r];
                    }
                }
            }
            __syncthreads();
        }
        if (warp == 0) {
            #pragma unroll
            for (unsigned c = 0; c < IK_MMVQ_COLUMNS; c++) {
                #pragma unroll
                for (unsigned r = 0; r < IK_MMVQ_ROWS; r++) {
                    #pragma unroll
                    for (unsigned w = 0; w + 1 < WARPS; w++) {
                        sum[c][r] += partial[w][c][r][lane];
                    }
                    sum[c][r] = warp_reduce_sum<WARP>(sum[c][r]);
                }
                if (lane < IK_MMVQ_ROWS && selected[c] >= 0) {
                    const float value = sum[c][lane];
                    out[(uint64_t)selected[c] * m + row + lane] = isfinite(value) ? value : 0.0f;
                }
            }
        }
        if constexpr (WARPS > 1) { __syncthreads(); }
    }
}

/* Warp-owned output tiles: R rows x C columns per warp, decoded once per
 * weight fragment. The per-output float reduction is byte-identical to the
 * MMVQ kernels above: every original lane product, warp partial and XOR
 * merge is recomputed in the same order by one warp.
 *
 *   original up (4 warps)         this kernel (1 warp, steps q = 0..3)
 *   warp q lane l: p_q            step q lane l: p_q  (same fragment)
 *   warp 0: ((p0 + p1) + p2) + p3 acc = p0; acc += p1; acc += p2; acc += p3
 *   XOR butterfly over lanes      XOR butterfly over lanes
 */
enum {
    IK_TILE_COLUMNS = 8,
    IK_TILE_WARPS = 4,
    IK_TILE_ALIGN = 16,
};

/* Activation SoA: qs[rows][k] int8 rows (16-byte aligned) and ds[rows][k/32]
 * half2 scales. The canonical 36-byte Q8_1 blocks are only 4-byte aligned. */
static __global__ void inkling_relayout_kernel(
        int8_t *qs, half2 *ds, const block_q8_1 *x, uint64_t groups) {
    const uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) { return; }
    const block_q8_1 *b = x + g;
    ds[g] = b->ds;
    int w[QK8_1 / 4];
    #pragma unroll
    for (int i = 0; i < QK8_1 / 4; i++) { w[i] = get_int_b4(b->qs, i); }
    int4 *dst = (int4 *)(qs + g * QK8_1);
    dst[0] = make_int4(w[0], w[1], w[2], w[3]);
    dst[1] = make_int4(w[4], w[5], w[6], w[7]);
}

template<ggml_type TYPE> struct ik_tile_traits;

/* IQ2_XXS fragment: 32 values = 4 grid bytes + 4x7 sign bits + 4-bit scale.
 * Decoding matches vec_dot_iq2_xxs_q8_1 exactly; the integer dot is exact. */
template<> struct ik_tile_traits<GGML_TYPE_IQ2_XXS> {
    static constexpr unsigned QK = QK_K, TPB = QI2_XXS / VDR_IQ2_XXS_Q8_1_MMVQ;
    static constexpr unsigned VDR = VDR_IQ2_XXS_Q8_1_MMVQ, X_WORDS = 8;
    struct Raw { uint32_t q2, aux; };
    struct Frag { int v[8]; int ls; float wd; };
    static __device__ __forceinline__ Raw load(const void *w, uint64_t block, unsigned iqs) {
        const block_iq2_xxs *b = (const block_iq2_xxs *)w + block;
        return { (uint32_t)get_int_b2(b->qs, iqs), (uint32_t)get_int_b2(b->qs, iqs + 1) };
    }
    static __device__ __forceinline__ float delta(const void *w, uint64_t block) {
        return __half2float(((const block_iq2_xxs *)w)[block].d);
    }
    /* Aligned SoA: [__half dq[nblk]][pad 64B][uint2 qs[nblk*8]]. VDR=2 so
     * iqs is even and one uint2 is the same 8 bytes as the two get_int_b2
     * reads. Decode and the integer dot stay the MMVQ tile formula. */
    static __device__ __forceinline__ Raw load_soa(const void *w, uint64_t nblk,
                                                  uint64_t block, unsigned iqs) {
        const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
        const uint2 *qs = (const uint2 *)((const char *)w + dq_bytes);
        const uint2 cw = qs[block * 8ull + (uint64_t)(iqs / 2u)];
        return { cw.x, cw.y };
    }
    static __device__ __forceinline__ float delta_soa(const void *w, uint64_t block) {
        return __half2float(((const __half *)w)[block]);
    }
    static __device__ __forceinline__ void decode(Frag &f, const Raw &r) {
        #pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            const uint2 grid = ((const uint2 *)iq2xxs_grid)[(r.q2 >> (8 * j)) & 0xFF];
            const uint32_t signs = unpack_ksigns((uint8_t)(r.aux >> (7 * j)));
            const int s0 = __vcmpne4(signs & 0x08040201, 0);
            const int s1 = __vcmpne4(signs & 0x80402010, 0);
            f.v[2 * j] = __vsub4(grid.x ^ s0, s0);
            f.v[2 * j + 1] = __vsub4(grid.y ^ s1, s1);
        }
        f.ls = r.aux >> 27 | 1;
    }
    static __device__ __forceinline__ int dot(const Frag &f, const int *u) {
        int sumi = 0;
        #pragma unroll
        for (unsigned i = 0; i < 8; i++) { sumi = ggml_cuda_dp4a(f.v[i], u[i], sumi); }
        return sumi * f.ls / 8;
    }
    static __device__ __forceinline__ uint32_t group(uint32_t bx, unsigned iqs) { return bx * (QK_K / QK8_1) + iqs / 2; }
    static __device__ __forceinline__ unsigned offset(unsigned) { return 0; }
};

/* IQ2_XS fragment: 4x9-bit grid indices with 7 sign bits each, two 4-bit
 * scales; two 16-value integer dots feed the original rounding formula. */
template<> struct ik_tile_traits<GGML_TYPE_IQ2_XS> {
    static constexpr unsigned QK = QK_K, TPB = QI2_XS / VDR_IQ2_XS_Q8_1_MMVQ;
    static constexpr unsigned VDR = VDR_IQ2_XS_Q8_1_MMVQ, X_WORDS = 8;
    struct Raw { uint32_t lo, hi; uint8_t scale; };
    struct Frag { int v[8]; int ls0, ls1; float wd; };
    static __device__ __forceinline__ Raw load(const void *w, uint64_t block, unsigned iqs) {
        const block_iq2_xs *b = (const block_iq2_xs *)w + block;
        return { (uint32_t)get_int_b2(b->qs, iqs), (uint32_t)get_int_b2(b->qs, iqs + 1), b->scales[iqs / 2] };
    }
    static __device__ __forceinline__ float delta(const void *w, uint64_t block) {
        return __half2float(((const block_iq2_xs *)w)[block].d);
    }
    /* SoA: [__half d[nblk]][pad 64][uint8 sc[nblk*8]][pad 64][uint2 qs[nblk*8]].
     * 74-byte AoS is d + 64B qs + 8B scales; VDR=2 keeps iqs even. */
    static __device__ __forceinline__ Raw load_soa(const void *w, uint64_t nblk,
                                                  uint64_t block, unsigned iqs) {
        const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
        const uint64_t sc_bytes = (nblk * 8u + 63u) & ~63ull;
        const uint8_t *sc = (const uint8_t *)w + dq_bytes;
        const uint2 *qs = (const uint2 *)((const char *)w + dq_bytes + sc_bytes);
        const uint2 cw = qs[block * 8ull + (uint64_t)(iqs / 2u)];
        return { cw.x, cw.y, sc[block * 8ull + (uint64_t)(iqs / 2u)] };
    }
    static __device__ __forceinline__ float delta_soa(const void *w, uint64_t block) {
        return __half2float(((const __half *)w)[block]);
    }
    static __device__ __forceinline__ void decode(Frag &f, const Raw &r) {
        #pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            const uint16_t q = (uint16_t)((j < 2 ? r.lo : r.hi) >> (16 * (j % 2)));
            const uint2 grid = ((const uint2 *)iq2xs_grid)[q & 0x1FF];
            const uint32_t signs = unpack_ksigns((uint8_t)(q >> 9));
            const int s0 = __vcmpne4(signs & 0x08040201, 0);
            const int s1 = __vcmpne4(signs & 0x80402010, 0);
            f.v[2 * j] = __vsub4(grid.x ^ s0, s0);
            f.v[2 * j + 1] = __vsub4(grid.y ^ s1, s1);
        }
        f.ls0 = r.scale & 0x0F;
        f.ls1 = r.scale >> 4;
    }
    static __device__ __forceinline__ int dot(const Frag &f, const int *u) {
        int sumi0 = 0, sumi1 = 0;
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) { sumi0 = ggml_cuda_dp4a(f.v[i], u[i], sumi0); }
        #pragma unroll
        for (unsigned i = 4; i < 8; i++) { sumi1 = ggml_cuda_dp4a(f.v[i], u[i], sumi1); }
        return (sumi0 * f.ls0 + sumi1 * f.ls1 + (sumi0 + sumi1) / 2) / 4;
    }
    static __device__ __forceinline__ uint32_t group(uint32_t bx, unsigned iqs) { return bx * (QK_K / QK8_1) + iqs / 2; }
    static __device__ __forceinline__ unsigned offset(unsigned) { return 0; }
};

/* Q8_0 fragment: eight int8 weights at a 4*iqs byte offset of a 34-byte block. */
template<> struct ik_tile_traits<GGML_TYPE_Q8_0> {
    static constexpr unsigned QK = QK8_0, TPB = QI8_0 / VDR_Q8_0_Q8_1_MMVQ;
    static constexpr unsigned VDR = VDR_Q8_0_Q8_1_MMVQ, X_WORDS = 2;
    struct Raw { int v0, v1; };
    struct Frag { int v[2]; float wd; };
    static __device__ __forceinline__ Raw load(const void *w, uint64_t block, unsigned iqs) {
        const block_q8_0 *b = (const block_q8_0 *)w + block;
        return { get_int_b2(b->qs, iqs), get_int_b2(b->qs, iqs + 1) };
    }
    static __device__ __forceinline__ float delta(const void *w, uint64_t block) {
        return __half2float(((const block_q8_0 *)w)[block].d);
    }
    static __device__ __forceinline__ void decode(Frag &f, const Raw &r) { f.v[0] = r.v0; f.v[1] = r.v1; }
    static __device__ __forceinline__ int dot(const Frag &f, const int *u) {
        int sumi = ggml_cuda_dp4a(f.v[0], u[0], 0);
        return ggml_cuda_dp4a(f.v[1], u[1], sumi);
    }
    static __device__ __forceinline__ uint32_t group(uint32_t bx, unsigned) { return bx; }
    static __device__ __forceinline__ unsigned offset(unsigned iqs) { return 4 * iqs; }
};

template<unsigned WORDS>
static __device__ __forceinline__ void inkling_tile_load_x(int *u, const int8_t *p) {
    if constexpr (WORDS == 8) {
        const int4 a = ((const int4 *)p)[0], b = ((const int4 *)p)[1];
        u[0] = a.x; u[1] = a.y; u[2] = a.z; u[3] = a.w;
        u[4] = b.x; u[5] = b.y; u[6] = b.z; u[7] = b.w;
    } else {
        const int2 a = *(const int2 *)p;
        u[0] = a.x; u[1] = a.y;
    }
}

template<ggml_type TYPE, unsigned R, unsigned C, unsigned WARPS, unsigned ITERS, bool ALIGNED>
__launch_bounds__(IK_TILE_WARPS * 32)
static __global__ void inkling_tile_kernel(
        const void *weights, const int8_t *xq, const half2 *xd, float *out,
        const int32_t *counts, const int32_t *buckets,
        const int32_t *tile_experts, const int32_t *tile_starts,
        uint32_t m, uint32_t k, uint32_t assignments, uint32_t experts,
        uint32_t used) {
    using T = ik_tile_traits<TYPE>;
    constexpr unsigned WARP = 32, K_STEP = WARP * WARPS / T::TPB;
    const unsigned lane = threadIdx.x % WARP;
    const uint32_t blocks_per_row = k / T::QK, groups_per_row = k / QK8_1;
    const uint32_t row_groups = m / R;
    const uint64_t jobs = (uint64_t)counts[experts] * row_groups;
    const uint64_t stride = (uint64_t)gridDim.x * (blockDim.x / WARP);
    uint64_t nblk = 0;
    if constexpr (ALIGNED) {
        nblk = (uint64_t)experts * m * blocks_per_row;
    }

    for (uint64_t job = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) / WARP;
         job < jobs; job += stride) {
        const uint32_t tile = job / row_groups, row0 = (job % row_groups) * R;
        const uint32_t expert = tile_experts[tile], start = tile_starts[tile];
        const uint64_t weight_base = ((uint64_t)expert * m + row0) * blocks_per_row;
        int32_t source[C];
        #pragma unroll
        for (unsigned c = 0; c < C; c++) {
            source[c] = start + c < (uint32_t)counts[expert]
                ? buckets[(uint64_t)expert * assignments + start + c] : -1;
        }
        float acc[R][C] = {};

        // Step q replays original warp q; its lanes keep their fragments.
        #pragma unroll
        for (unsigned q = 0; q < WARPS; q++) {
            const unsigned tid = q * WARP + lane;
            const uint32_t bx0 = tid / T::TPB, iqs = T::VDR * (tid % T::TPB);
            float pq[R][C] = {};
            #pragma unroll
            for (unsigned i = 0; i < ITERS; i++) {
                const uint32_t bx = bx0 + i * K_STEP;
                typename T::Frag frag[R];
                #pragma unroll
                for (unsigned r = 0; r < R; r++) {
                    const uint64_t block = weight_base + r * blocks_per_row + bx;
                    if constexpr (ALIGNED) {
                        T::decode(frag[r], T::load_soa(weights, nblk, block, iqs));
                        frag[r].wd = T::delta_soa(weights, block);
                    } else {
                        T::decode(frag[r], T::load(weights, block, iqs));
                        frag[r].wd = T::delta(weights, block);
                    }
                }
                const uint32_t group = T::group(bx, iqs);
                #pragma unroll
                for (unsigned c = 0; c < C; c++) {
                    if (source[c] < 0) { continue; }
                    const uint32_t xrow = source[c] / used;
                    int u[T::X_WORDS];
                    inkling_tile_load_x<T::X_WORDS>(u, xq + (uint64_t)xrow * k + group * QK8_1 + T::offset(iqs));
                    const float xs = __low2float(xd[(uint64_t)xrow * groups_per_row + group]);
                    #pragma unroll
                    for (unsigned r = 0; r < R; r++) {
                        const int sumi = T::dot(frag[r], u);
                        const float d = __fmul_rn(frag[r].wd, xs);
                        if constexpr (WARPS == 1) {
                            acc[r][c] = fmaf(d, (float)sumi, acc[r][c]);
                        } else if constexpr (ITERS == 1) {
                            const float p = fmaf(d, (float)sumi, 0.0f);
                            acc[r][c] = q == 0 ? p : __fadd_rn(acc[r][c], p);
                        } else {
                            pq[r][c] = fmaf(d, (float)sumi, pq[r][c]);
                        }
                    }
                }
            }
            if constexpr (WARPS > 1 && ITERS > 1) {
                #pragma unroll
                for (unsigned r = 0; r < R; r++) {
                    #pragma unroll
                    for (unsigned c = 0; c < C; c++) {
                        acc[r][c] = q == 0 ? pq[r][c] : __fadd_rn(acc[r][c], pq[r][c]);
                    }
                }
            }
        }

        #pragma unroll
        for (unsigned r = 0; r < R; r++) {
            #pragma unroll
            for (unsigned c = 0; c < C; c++) {
                const float value = warp_reduce_sum<WARP>(acc[r][c]);
                if (lane == 0 && source[c] >= 0) {
                    out[(uint64_t)source[c] * m + row0 + r] = isfinite(value) ? value : 0.0f;
                }
            }
        }
    }
}

template<ggml_type TYPE, unsigned R, unsigned C, unsigned WARPS, unsigned ITERS, bool ALIGNED>
static int inkling_tile_launch(
        const void *weights, const int8_t *xq, const half2 *xd, float *out,
        const int32_t *counts, const int32_t *buckets, const int32_t *tile_experts,
        const int32_t *tile_starts, uint32_t m, uint32_t k, uint32_t assignments,
        uint32_t experts, uint32_t used, uint32_t sms, cudaStream_t stream) {
    using T = ik_tile_traits<TYPE>;
    constexpr unsigned K_STEP = 32 * WARPS / T::TPB;
    if (m % R || k / T::QK != ITERS * K_STEP) { return 1; }
    static int resident = 0;
    if (!resident && (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
            inkling_tile_kernel<TYPE, R, C, WARPS, ITERS, ALIGNED>, IK_TILE_WARPS * 32, 0) != cudaSuccess ||
            resident <= 0)) { return -2; }
    inkling_tile_kernel<TYPE, R, C, WARPS, ITERS, ALIGNED><<<sms * resident, IK_TILE_WARPS * 32, 0, stream>>>(
        weights, xq, xd, out, counts, buckets, tile_experts, tile_starts,
        m, k, assignments, experts, used);
    return 0;
}

// Returns 0 when launched, 1 when the shape has no tile instantiation.
static int inkling_tile_dispatch(
        const void *weights, ggml_type type, const int8_t *xq, const half2 *xd,
        float *out, const int32_t *counts, const int32_t *buckets,
        const int32_t *tile_experts, const int32_t *tile_starts, uint32_t m,
        uint32_t k, uint32_t assignments, uint32_t experts, uint32_t used,
        uint32_t sms, cudaStream_t stream, int aligned) {
    #define IK_TILE(T, R, W, I, A) inkling_tile_launch<T, R, IK_TILE_COLUMNS, W, I, A>( \
        weights, xq, xd, out, counts, buckets, tile_experts, tile_starts, \
        m, k, assignments, experts, used, sms, stream)
    // Four-warp up keeps one fragment per lane; Q8 up carries four per warp.
    switch (type) {
    case GGML_TYPE_IQ2_XXS:
        if (used <= 1) { return 1; }
        return aligned ? IK_TILE(GGML_TYPE_IQ2_XXS, 4, 4, 1, true)
                       : IK_TILE(GGML_TYPE_IQ2_XXS, 4, 4, 1, false);
    case GGML_TYPE_IQ2_XS:
        if (used > 1) { return 1; }
        return aligned ? IK_TILE(GGML_TYPE_IQ2_XS, 4, 1, 2, true)
                       : IK_TILE(GGML_TYPE_IQ2_XS, 4, 1, 2, false);
    case GGML_TYPE_Q8_0: return used > 1 ? IK_TILE(GGML_TYPE_Q8_0, 2, 4, 4, false)
                                         : IK_TILE(GGML_TYPE_Q8_0, 4, 1, 8, false);
    default: return 1;
    }
    #undef IK_TILE
}

// Shared Q8 tiles reuse each decoded payload across columns while retaining
// the MMVQ K partitions, FMA chains and ordered inter-warp merge. Prefill only.
enum { IK_SHARED_EXPERTS = 2, IK_SHARED_Q8_COLS = 8, IK_SHARED_Q8_MIN = 16, IK_SHARED_WARP = 32 };
template<unsigned C, unsigned W>
__launch_bounds__(W * IK_SHARED_WARP, 1)
static __global__ void inkling_shared_q8_kernel(const block_q8_0 *weights,
        const block_q8_1 *x, float *out, const int32_t *counts,
        const int32_t *buckets, const int32_t *experts, const int32_t *starts,
        unsigned m, unsigned k, unsigned assignments, unsigned used) {
    const unsigned tid = threadIdx.x, lane = tid % IK_SHARED_WARP;
    const unsigned warp = tid / IK_SHARED_WARP, blocks = k / QK8_1;
    const unsigned row_tiles = m / IK_MMVQ_ROWS;
    const uint64_t jobs = (uint64_t)counts[IK_SHARED_EXPERTS] * row_tiles;
    __shared__ float partial[W > 1 ? W - 1 : 1][C][IK_MMVQ_ROWS][IK_SHARED_WARP];
    for (uint64_t job = blockIdx.x; job < jobs; job += gridDim.x) {
        const unsigned tile = job / row_tiles, row = job % row_tiles * IK_MMVQ_ROWS;
        const unsigned expert = experts[tile], begin = starts[tile];
        int32_t selected[C];
        #pragma unroll
        for (unsigned c = 0; c < C; c++) {
            selected[c] = begin + c < (unsigned)counts[expert]
                ? buckets[(uint64_t)expert * assignments + begin + c] : -1;
        }
        float sum[C][IK_MMVQ_ROWS] = {};
        // Q8 MMVQ assigns four eight-value fragments per block. Keep the
        // same K iterations and FMA/warp merge; hoist only repeated loads.
        const unsigned qs = 2 * (tid % 4);
        for (unsigned bx = tid / 4; bx < blocks; bx += W * IK_SHARED_WARP / 4) {
            int payload[IK_MMVQ_ROWS][2];
            float delta[IK_MMVQ_ROWS];
            #pragma unroll
            for (unsigned r = 0; r < IK_MMVQ_ROWS; r++) {
                const auto *w = weights + ((uint64_t)expert * m + row + r) * blocks + bx;
                payload[r][0] = get_int_b2(w->qs, qs);
                payload[r][1] = get_int_b2(w->qs, qs + 1);
                delta[r] = __half2float(w->d);
            }
            #pragma unroll
            for (unsigned c = 0; c < C; c++) {
                if (selected[c] < 0) { continue; }
                const auto *q = x + (uint64_t)(selected[c] / used) * blocks + bx;
                const int u0 = get_int_b4(q->qs, qs), u1 = get_int_b4(q->qs, qs + 1);
                const float xd = __low2float(q->ds);
                #pragma unroll
                for (unsigned r = 0; r < IK_MMVQ_ROWS; r++) {
                    int dot = ggml_cuda_dp4a(payload[r][0], u0, 0);
                    dot = ggml_cuda_dp4a(payload[r][1], u1, dot);
                    const float scale = __fmul_rn(delta[r], xd);
                    sum[c][r] = fmaf(scale, (float)dot, sum[c][r]);
                }
            }
        }
        if constexpr (W > 1) {
            if (warp > 0) {
                #pragma unroll
                for (unsigned c = 0; c < C; c++) {
                    #pragma unroll
                    for (unsigned r = 0; r < IK_MMVQ_ROWS; r++) {
                        partial[warp - 1][c][r][lane] = sum[c][r];
                    }
                }
            }
            __syncthreads();
        }
        if (warp == 0) {
            #pragma unroll
            for (unsigned c = 0; c < C; c++) {
                #pragma unroll
                for (unsigned r = 0; r < IK_MMVQ_ROWS; r++) {
                    #pragma unroll
                    for (unsigned w = 0; w + 1 < W; w++) {
                        sum[c][r] = __fadd_rn(sum[c][r], partial[w][c][r][lane]);
                    }
                    sum[c][r] = warp_reduce_sum<IK_SHARED_WARP>(sum[c][r]);
                }
                if (lane < IK_MMVQ_ROWS && selected[c] >= 0) {
                    const float value = sum[c][lane];
                    out[(uint64_t)selected[c] * m + row + lane] = isfinite(value) ? value : 0.0f;
                }
            }
        }
        if constexpr (W > 1) { __syncthreads(); }
    }
}

template<unsigned W>
static int inkling_shared_q8_launch(
        const void *weights, const block_q8_1 *x, float *out, const int32_t *counts,
        const int32_t *buckets, const int32_t *experts, const int32_t *starts,
        uint32_t m, uint32_t k, uint32_t assignments, uint32_t used,
        uint32_t sms, cudaStream_t stream) {
    static int active = 0;
    if (!active && (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
            inkling_shared_q8_kernel<IK_SHARED_Q8_COLS, W>, W * IK_SHARED_WARP, 0) != cudaSuccess ||
            active <= 0)) { return -2; }
    inkling_shared_q8_kernel<IK_SHARED_Q8_COLS, W><<<sms * active, W * IK_SHARED_WARP, 0, stream>>>(
        (const block_q8_0 *)weights, x, out, counts, buckets, experts, starts,
        m, k, assignments, used);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

#include "inkling_shared_tile.cuh"
#include "inkling_q4.cuh"

// Routing tables, then the 16-byte aligned activation SoA for the tile path.
static uint64_t inkling_route_bytes(uint64_t assignments, int experts) {
    return ((experts + 1) + assignments * (experts + 2)) * sizeof(int32_t);
}

uint64_t ds4_mmvq_inkling_bytes(int rows, int experts, int used) {
    if (rows <= 0 || experts <= 0 || experts > IK_MMVQ_EXPERTS || used <= 0 ||
        used > experts || rows > IK_MMVQ_ASSIGNMENTS / used) { return 0; }
    const uint64_t assignments = (uint64_t)rows * used;
    const uint64_t k = used > 1 ? IK_MMVQ_HIDDEN : IK_MMVQ_MIDDLE;
    const uint64_t soa = (uint64_t)rows * k + (uint64_t)rows * (k / QK8_1) * sizeof(half2);
    return inkling_route_bytes(assignments, experts) + IK_TILE_ALIGN + soa;
}

template<ggml_type TYPE>
static void inkling_mmvq_launch(
        const void *weights, const block_q8_1 *x, float *out, int32_t *workspace,
        uint32_t m, uint32_t k, uint32_t rows, uint32_t experts, uint32_t used,
        uint32_t sms, cudaStream_t stream) {
    const uint32_t assignments = rows * used;
    int32_t *buckets = workspace + experts + 1;
    int32_t *tile_experts = buckets + (uint64_t)assignments * experts;
    int32_t *tile_starts = tile_experts + assignments;
    const uint32_t warps = used > 1 ? 4 : 1;
    const uint32_t blocks = sms * IK_MMVQ_RESIDENT_THREADS / (warps * 32);
    #define IK_MMVQ_LAUNCH(W) inkling_mmvq_kernel<TYPE, W><<<blocks, W * 32, 0, stream>>>( \
        weights, x, out, workspace, buckets, tile_experts, tile_starts, m, k, assignments, experts, used)
    if (used > 1) { IK_MMVQ_LAUNCH(4); }
    else { IK_MMVQ_LAUNCH(1); }
    #undef IK_MMVQ_LAUNCH
}

int ds4_mmvq_inkling(
        const void *weights, ggml_type type, const void *x, const int32_t *ids,
        float *out, void *workspace, uint64_t workspace_bytes, int m, int k,
        int rows, int experts, int used, cudaStream_t stream) {
    const uint64_t required = ds4_mmvq_inkling_bytes(rows, experts, used);
    if (!weights || !x || !ids || !out || !workspace || m <= 0 || m > IK_MMVQ_HIDDEN ||
        m % IK_MMVQ_ROWS || (used > 1 ? k != IK_MMVQ_HIDDEN : k != IK_MMVQ_MIDDLE) ||
        !required || workspace_bytes < required) { return -1; }
    if (type != GGML_TYPE_Q8_0 && type != GGML_TYPE_Q3_K && type != GGML_TYPE_Q4_K &&
        type != GGML_TYPE_IQ2_XXS && type != GGML_TYPE_IQ2_XS) { return -1; }
    const auto &device = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if (device.warp_size != 32 || device.nsm <= 0) { return -1; }
    const uint32_t assignments = rows * used;
    int32_t *counts = (int32_t *)workspace, *buckets = counts + experts + 1;
    int32_t *tile_experts = buckets + (uint64_t)assignments * experts;
    int32_t *tile_starts = tile_experts + assignments;
    if (cudaMemsetAsync(counts, 0, (experts + 1) * sizeof(int32_t), stream) != cudaSuccess ||
        cudaMemsetAsync(out, 0, (uint64_t)assignments * m * sizeof(float), stream) != cudaSuccess) { return -2; }
    inkling_bucket_kernel<<<(assignments + IK_MMVQ_THREADS - 1) / IK_MMVQ_THREADS,
                            IK_MMVQ_THREADS, 0, stream>>>(counts, buckets, ids, assignments, experts);
    // Shared up keeps its four-partition sum; wide prefill retains weights
    // across input groups, while narrow rows use the cooperating-warp kernel.
    if (type == GGML_TYPE_Q8_0 && experts == IK_SHARED_EXPERTS &&
        used == IK_SHARED_EXPERTS && rows >= IK_SHARED_Q8_MIN &&
        !getenv("DS4_INKLING_NO_SHARED_Q8") && !getenv("DS4_INKLING_NO_MOE_TILE")) {
        // Keep narrow verification on the existing kernel. Wider prefill can
        // retain every weight fragment while its routed input groups stream.
        if (rows >= IK_ST_MIN && !getenv("DS4_INKLING_NO_SHARED_TILE")) {
            const int rc = inkling_shared_tile_launch(weights, (const block_q8_1 *)x,
                out, counts, buckets, m, k, assignments, used, experts, stream);
            if (rc <= 0) { return rc; }
        }
        inkling_tiles_kernel<IK_SHARED_Q8_COLS><<<1, IK_MMVQ_THREADS, 0, stream>>>(
            counts, tile_experts, tile_starts, experts);
        return inkling_shared_q8_launch<4>(weights, (const block_q8_1 *)x, out,
            counts, buckets, tile_experts, tile_starts,
            m, k, assignments, used, device.nsm, stream);
    }
    // Shared down retains its one-warp, eight-step MMVQ chain. Its rows
    // contain two assignments per prompt token, unlike shared up.
    if (type == GGML_TYPE_Q8_0 && experts == IK_SHARED_EXPERTS && used == 1 &&
        rows >= IK_ST_MIN * IK_SHARED_EXPERTS &&
        !getenv("DS4_INKLING_NO_SHARED_DOWN_TILE") && !getenv("DS4_INKLING_NO_MOE_TILE")) {
        const int rc = inkling_shared_tile_launch(weights, (const block_q8_1 *)x,
            out, counts, buckets, m, k, assignments, used, experts, stream);
        if (rc <= 0) { return rc; }
    }
    // Routed Q8 otherwise reloads weights for every eight-column tile after
    // an activation relayout. Wide prefill keeps the shared-tile schedule.
    if (type == GGML_TYPE_Q8_0 && experts != IK_SHARED_EXPERTS &&
        rows >= IK_ST_MIN && !getenv("DS4_INKLING_NO_Q8_ROUTED_TILE") &&
        !getenv("DS4_INKLING_NO_MOE_TILE")) {
        const int rc = inkling_shared_tile_launch(weights, (const block_q8_1 *)x,
            out, counts, buckets, m, k, assignments, used, experts, stream);
        if (rc <= 0) { return rc; }
    }
    // Q4_K otherwise repeats scale/payload decoding per column and builds
    // an unused activation SoA before falling back to the four-column path.
    if (type == GGML_TYPE_Q4_K && assignments >= IK_Q4_MIN_ASSIGNMENTS &&
        !getenv("DS4_INKLING_NO_Q4_TILE") && !getenv("DS4_INKLING_NO_MOE_TILE")) {
        inkling_tiles_kernel<IK_Q4_COLS><<<1, IK_MMVQ_THREADS, 0, stream>>>(
            counts, tile_experts, tile_starts, experts);
        if (used > 1) {
            return inkling_q4_launch<2, IK_Q4_COLS, 4, 2>(weights, (const block_q8_1 *)x,
                out, counts, buckets, tile_experts, tile_starts,
                m, k, assignments, experts, used, device.nsm, stream);
        }
        return inkling_q4_launch<2, IK_Q4_COLS, 1, 4>(weights, (const block_q8_1 *)x,
            out, counts, buckets, tile_experts, tile_starts,
            m, k, assignments, experts, used, device.nsm, stream);
    }
    // Warp tiles are the release path; the switch restores the four-warp
    // column kernel for A/B controls. Both keep the same routing tables.
    if (!getenv("DS4_INKLING_NO_MOE_TILE")) {
        const uint64_t groups = (uint64_t)rows * k / QK8_1;
        const uintptr_t soa = ((uintptr_t)workspace + inkling_route_bytes(assignments, experts) +
                               IK_TILE_ALIGN - 1) & ~(uintptr_t)(IK_TILE_ALIGN - 1);
        int8_t *xq = (int8_t *)soa;
        half2 *xd = (half2 *)(xq + (uint64_t)rows * k);
        inkling_tiles_kernel<IK_TILE_COLUMNS><<<1, IK_MMVQ_THREADS, 0, stream>>>(
            counts, tile_experts, tile_starts, experts);
        inkling_relayout_kernel<<<(groups + IK_MMVQ_THREADS - 1) / IK_MMVQ_THREADS,
                                  IK_MMVQ_THREADS, 0, stream>>>(xq, xd, (const block_q8_1 *)x, groups);
        const int rc = inkling_tile_dispatch(weights, type, xq, xd, out, counts, buckets,
            tile_experts, tile_starts, m, k, assignments, experts, used, device.nsm, stream, 0);
        if (rc <= 0) { return rc == 0 && cudaGetLastError() == cudaSuccess ? 0 : -2; }
        // Unsupported tile shape: rebuild the four-column tables below.
        if (cudaMemsetAsync(counts + experts, 0, sizeof(int32_t), stream) != cudaSuccess) { return -2; }
    }
    inkling_tiles_kernel<IK_MMVQ_COLUMNS><<<1, IK_MMVQ_THREADS, 0, stream>>>(counts, tile_experts, tile_starts, experts);
    #define IK_MMVQ_CASE(T) case T: inkling_mmvq_launch<T>(weights, (const block_q8_1 *)x, \
        out, counts, m, k, rows, experts, used, device.nsm, stream); break
    switch (type) {
    IK_MMVQ_CASE(GGML_TYPE_Q8_0);
    IK_MMVQ_CASE(GGML_TYPE_Q3_K);
    IK_MMVQ_CASE(GGML_TYPE_Q4_K);
    IK_MMVQ_CASE(GGML_TYPE_IQ2_XXS);
    IK_MMVQ_CASE(GGML_TYPE_IQ2_XS);
    default: return -1;
    }
    #undef IK_MMVQ_CASE
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int ds4_mmvq_inkling_iq2_aligned(
        const void *weights, const void *x, const int32_t *ids,
        float *out, void *workspace, uint64_t workspace_bytes, int m, int k,
        int rows, int experts, int used, cudaStream_t stream) {
    const uint64_t required = ds4_mmvq_inkling_bytes(rows, experts, used);
    if (!weights || !x || !ids || !out || !workspace || m <= 0 || m > IK_MMVQ_HIDDEN ||
        m % IK_MMVQ_ROWS || k != IK_MMVQ_HIDDEN || used <= 1 ||
        !required || workspace_bytes < required) { return -1; }
    const auto &device = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if (device.warp_size != 32 || device.nsm <= 0) { return -1; }
    const uint32_t assignments = rows * used;
    int32_t *counts = (int32_t *)workspace, *buckets = counts + experts + 1;
    int32_t *tile_experts = buckets + (uint64_t)assignments * experts;
    int32_t *tile_starts = tile_experts + assignments;
    if (cudaMemsetAsync(counts, 0, (experts + 1) * sizeof(int32_t), stream) != cudaSuccess ||
        cudaMemsetAsync(out, 0, (uint64_t)assignments * m * sizeof(float), stream) != cudaSuccess) { return -2; }
    inkling_bucket_kernel<<<(assignments + IK_MMVQ_THREADS - 1) / IK_MMVQ_THREADS,
                            IK_MMVQ_THREADS, 0, stream>>>(counts, buckets, ids, assignments, experts);
    const uint64_t groups = (uint64_t)rows * k / QK8_1;
    const uintptr_t soa = ((uintptr_t)workspace + inkling_route_bytes(assignments, experts) +
                           IK_TILE_ALIGN - 1) & ~(uintptr_t)(IK_TILE_ALIGN - 1);
    int8_t *xq = (int8_t *)soa;
    half2 *xd = (half2 *)(xq + (uint64_t)rows * k);
    inkling_tiles_kernel<IK_TILE_COLUMNS><<<1, IK_MMVQ_THREADS, 0, stream>>>(
        counts, tile_experts, tile_starts, experts);
    inkling_relayout_kernel<<<(groups + IK_MMVQ_THREADS - 1) / IK_MMVQ_THREADS,
                              IK_MMVQ_THREADS, 0, stream>>>(xq, xd, (const block_q8_1 *)x, groups);
    const int rc = inkling_tile_dispatch(weights, GGML_TYPE_IQ2_XXS, xq, xd, out, counts, buckets,
        tile_experts, tile_starts, m, k, assignments, experts, used, device.nsm, stream, 1);
    if (rc <= 0) { return rc == 0 && cudaGetLastError() == cudaSuccess ? 0 : -2; }
    return -1;
}

int ds4_mmvq_inkling_iq2_xs_aligned(
        const void *weights, const void *x, const int32_t *ids,
        float *out, void *workspace, uint64_t workspace_bytes, int m, int k,
        int rows, int experts, int used, cudaStream_t stream) {
    const uint64_t required = ds4_mmvq_inkling_bytes(rows, experts, used);
    if (!weights || !x || !ids || !out || !workspace || m <= 0 || m > IK_MMVQ_HIDDEN ||
        m % IK_MMVQ_ROWS || k != IK_MMVQ_MIDDLE || used != 1 ||
        !required || workspace_bytes < required) { return -1; }
    const auto &device = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if (device.warp_size != 32 || device.nsm <= 0) { return -1; }
    const uint32_t assignments = rows * used;
    int32_t *counts = (int32_t *)workspace, *buckets = counts + experts + 1;
    int32_t *tile_experts = buckets + (uint64_t)assignments * experts;
    int32_t *tile_starts = tile_experts + assignments;
    if (cudaMemsetAsync(counts, 0, (experts + 1) * sizeof(int32_t), stream) != cudaSuccess ||
        cudaMemsetAsync(out, 0, (uint64_t)assignments * m * sizeof(float), stream) != cudaSuccess) { return -2; }
    inkling_bucket_kernel<<<(assignments + IK_MMVQ_THREADS - 1) / IK_MMVQ_THREADS,
                            IK_MMVQ_THREADS, 0, stream>>>(counts, buckets, ids, assignments, experts);
    const uint64_t groups = (uint64_t)rows * k / QK8_1;
    const uintptr_t soa = ((uintptr_t)workspace + inkling_route_bytes(assignments, experts) +
                           IK_TILE_ALIGN - 1) & ~(uintptr_t)(IK_TILE_ALIGN - 1);
    int8_t *xq = (int8_t *)soa;
    half2 *xd = (half2 *)(xq + (uint64_t)rows * k);
    inkling_tiles_kernel<IK_TILE_COLUMNS><<<1, IK_MMVQ_THREADS, 0, stream>>>(
        counts, tile_experts, tile_starts, experts);
    inkling_relayout_kernel<<<(groups + IK_MMVQ_THREADS - 1) / IK_MMVQ_THREADS,
                              IK_MMVQ_THREADS, 0, stream>>>(xq, xd, (const block_q8_1 *)x, groups);
    const int rc = inkling_tile_dispatch(weights, GGML_TYPE_IQ2_XS, xq, xd, out, counts, buckets,
        tile_experts, tile_starts, m, k, assignments, experts, used, device.nsm, stream, 1);
    if (rc <= 0) { return rc == 0 && cudaGetLastError() == cudaSuccess ? 0 : -2; }
    return -1;
}
