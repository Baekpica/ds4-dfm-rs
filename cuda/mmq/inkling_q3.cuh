/* Q3_K expert prefill: decode each row fragment once for eight routed
 * columns. The four-column kernel re-unpacks the 3-bit values and scales
 * inside vec_dot_q3_K_q8_1 for every column. Per (row, column) the lane
 * products, the four-term FMA chain, the per-fragment d3 FMA, the ascending
 * warp merge and the XOR tree are the inkling_mmvq_kernel<Q3_K, 4> sequence,
 * so outputs are byte-identical. Padded columns read row 0; their stores
 * stay guarded. Four rows per warp need 255 registers (two CTAs per SM),
 * which still beat every smaller tile in the 1024-token probe. */
enum { IK_Q3_WARP = 32, IK_Q3_WARPS = 4, IK_Q3_COLS = 8, IK_Q3_ROWS = 4,
       IK_Q3_TPB = QI3_K / VDR_Q3_K_Q8_1_MMVQ, IK_Q3_GROUPS = QK_K / QK8_1,
       IK_Q3_MIN_ASSIGNMENTS = 256 * 6 };

struct InklingQ3Frag { int vi[QR3_K]; int sc[QR3_K]; float d; };

// iqs in [0, 16): 16 low-bit pairs, their high bits and the four scales of
// the Q8 groups bq8_offset..bq8_offset+3, exactly as vec_dot_q3_K_q8_1.
static __device__ __forceinline__ InklingQ3Frag inkling_q3_load(
        const block_q3_K *w, unsigned iqs) {
    const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
    const int vl = get_int_b2(w->qs, iqs);
    const int vh = ~get_int_b2(w->hmask, iqs % (QI3_K / 2)) >> bq8_offset;
    InklingQ3Frag f;
    f.d = w->d;
    #pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % (QK_K / 32);
        const int sc_shift_low = 4 * (isc / (QK_K / 32));
        const int sc_low = (w->scales[isc_low] >> sc_shift_low) & 0xF;
        const int isc_high = isc % (QK_K / 64);
        const int sc_shift_high = 2 * (isc / (QK_K / 64));
        const int sc_high = ((w->scales[(QK_K / 32) + isc_high] >> sc_shift_high) & 3) << 4;
        f.sc[i] = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        f.vi[i] = __vsubss4(vil, vih);
    }
    return f;
}

template<unsigned R, unsigned C, unsigned WARPS, unsigned ITERS>
__launch_bounds__(IK_Q3_WARPS * IK_Q3_WARP)
static __global__ void inkling_q3_kernel(
        const block_q3_K *weights, const int8_t *xq, const half2 *xd, float *out,
        const int32_t *counts, const int32_t *buckets,
        const int32_t *tile_experts, const int32_t *tile_starts,
        uint32_t m, uint32_t k, uint32_t assignments, uint32_t experts,
        uint32_t used) {
    constexpr unsigned K_STEP = IK_Q3_WARP * WARPS / IK_Q3_TPB;
    const unsigned lane = threadIdx.x % IK_Q3_WARP;
    const uint32_t blocks_per_row = k / QK_K, groups_per_row = k / QK8_1;
    const uint32_t row_groups = m / R;
    const uint64_t jobs = (uint64_t)counts[experts] * row_groups;
    const uint64_t stride = (uint64_t)gridDim.x * (blockDim.x / IK_Q3_WARP);

    for (uint64_t job = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) / IK_Q3_WARP;
         job < jobs; job += stride) {
        const uint32_t tile = job / row_groups, row0 = (job % row_groups) * R;
        const uint32_t expert = tile_experts[tile], start = tile_starts[tile];
        const block_q3_K *wrow = weights + ((uint64_t)expert * m + row0) * blocks_per_row;
        int32_t source[C];
        uint32_t xrow[C];
        #pragma unroll
        for (unsigned c = 0; c < C; c++) {
            source[c] = start + c < (uint32_t)counts[expert]
                ? buckets[(uint64_t)expert * assignments + start + c] : -1;
            xrow[c] = source[c] >= 0 ? (uint32_t)source[c] / used : 0u;
        }
        float acc[R][C] = {};

        // Step q replays original warp q; pq is its K-partition chain.
        #pragma unroll
        for (unsigned q = 0; q < WARPS; q++) {
            const unsigned tid = q * IK_Q3_WARP + lane;
            const uint32_t bx0 = tid / IK_Q3_TPB, iqs = tid % IK_Q3_TPB;
            const unsigned bq8_offset = QR3_K * (iqs / (QI3_K / 2)), word = iqs % QI8_1;
            float pq[R][C] = {};
            #pragma unroll
            for (unsigned i = 0; i < ITERS; i++) {
                const uint32_t bx = bx0 + i * K_STEP;
                InklingQ3Frag frag[R];
                #pragma unroll
                for (unsigned r = 0; r < R; r++) {
                    frag[r] = inkling_q3_load(wrow + r * blocks_per_row + bx, iqs);
                }
                const uint32_t group = bx * IK_Q3_GROUPS + bq8_offset;
                #pragma unroll
                for (unsigned c = 0; c < C; c++) {
                    // Four Q8 groups: one word each at a 32-byte stride, and
                    // their four scales as one 16-byte load.
                    const int8_t *xb = xq + (uint64_t)xrow[c] * k + group * QK8_1 + 4 * word;
                    int u[QR3_K];
                    #pragma unroll
                    for (int j = 0; j < QR3_K; j++) { u[j] = *(const int *)(xb + j * QK8_1); }
                    const int4 dv = *(const int4 *)(xd + (uint64_t)xrow[c] * groups_per_row + group);
                    const float d8[QR3_K] = {
                        __low2float(*(const half2 *)&dv.x), __low2float(*(const half2 *)&dv.y),
                        __low2float(*(const half2 *)&dv.z), __low2float(*(const half2 *)&dv.w)};
                    #pragma unroll
                    for (unsigned r = 0; r < R; r++) {
                        float sumf = 0.0f;
                        #pragma unroll
                        for (int j = 0; j < QR3_K; j++) {
                            const int t = ggml_cuda_dp4a(frag[r].vi[j], u[j], 0) * frag[r].sc[j];
                            sumf = fmaf(d8[j], (float)t, sumf);
                        }
                        pq[r][c] = fmaf(frag[r].d, sumf, pq[r][c]);
                    }
                }
            }
            #pragma unroll
            for (unsigned r = 0; r < R; r++) {
                #pragma unroll
                for (unsigned c = 0; c < C; c++) {
                    acc[r][c] = q == 0 ? pq[r][c] : __fadd_rn(acc[r][c], pq[r][c]);
                }
            }
        }

        #pragma unroll
        for (unsigned r = 0; r < R; r++) {
            #pragma unroll
            for (unsigned c = 0; c < C; c++) {
                const float value = warp_reduce_sum<IK_Q3_WARP>(acc[r][c]);
                if (lane == 0 && source[c] >= 0) {
                    out[(uint64_t)source[c] * m + row0 + r] = isfinite(value) ? value : 0.0f;
                }
            }
        }
    }
}

// Returns 1 when the shape has no tile instantiation (rows or K mismatch).
template<unsigned R, unsigned WARPS, unsigned ITERS>
static int inkling_q3_launch(const void *weights, const int8_t *xq, const half2 *xd,
        float *out, const int32_t *counts, const int32_t *buckets,
        const int32_t *tile_experts, const int32_t *tile_starts, uint32_t m,
        uint32_t k, uint32_t assignments, uint32_t experts, uint32_t used,
        uint32_t sms, cudaStream_t stream) {
    constexpr unsigned K_STEP = IK_Q3_WARP * WARPS / IK_Q3_TPB;
    if (m % R || k / QK_K != ITERS * K_STEP) { return 1; }
    static int resident = 0;
    if (!resident && (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
            inkling_q3_kernel<R, IK_Q3_COLS, WARPS, ITERS>, IK_Q3_WARPS * IK_Q3_WARP, 0) != cudaSuccess ||
            resident <= 0)) { return -2; }
    inkling_q3_kernel<R, IK_Q3_COLS, WARPS, ITERS><<<sms * resident, IK_Q3_WARPS * IK_Q3_WARP, 0, stream>>>(
        (const block_q3_K *)weights, xq, xd, out, counts, buckets, tile_experts, tile_starts,
        m, k, assignments, experts, used);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
