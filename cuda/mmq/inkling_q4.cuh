/* Q4_K prefill coverage: decode each row fragment once for eight routed
 * columns, preserving the original MMVQ lane products and ordered sums. */
enum { IK_Q4_WARP = 32, IK_Q4_WARPS = 4, IK_Q4_COLS = 8,
       IK_Q4_MIN_ASSIGNMENTS = 512 * 6 };

struct InklingQ4Frag {
    int v[2];
    uint16_t aux[2];
    half2 dm;
};

static __device__ __forceinline__ InklingQ4Frag inkling_q4_load(
        const block_q4_K *w, unsigned iqs) {
    InklingQ4Frag f;
    const unsigned offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int *q = (const int *)(w->qs + 16 * offset + 4 * ((iqs / 2) % 4));
    f.v[0] = q[0]; f.v[1] = q[4]; f.dm = w->dm;
    const uint16_t *scales = (const uint16_t *)w->scales;
    const unsigned j = offset / 2;
    if (j < 2) {
        f.aux[0] = scales[j] & 0x3f3f;
        f.aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        f.aux[0] = (scales[j + 2] & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        f.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j] & 0xc0c0) >> 2);
    }
    return f;
}

template<unsigned R, unsigned C, unsigned W, unsigned ITERS>
__launch_bounds__(IK_Q4_WARPS * IK_Q4_WARP)
static __global__ void inkling_q4_kernel(const block_q4_K *weights,
        const block_q8_1 *x, float *out, const int32_t *counts,
        const int32_t *buckets, const int32_t *tile_experts,
        const int32_t *starts, unsigned m, unsigned k, unsigned assignments,
        unsigned ne, unsigned used) {
    constexpr unsigned TPB = QI4_K / VDR_Q4_K_Q8_1_MMVQ;
    constexpr unsigned STEP = IK_Q4_WARP * W / TPB;
    const unsigned lane = threadIdx.x % IK_Q4_WARP, row_groups = m / R;
    const unsigned blocks = k / QK_K, x_blocks = k / QK8_1;
    const uint64_t jobs = (uint64_t)counts[ne] * row_groups;
    const uint64_t stride = (uint64_t)gridDim.x * IK_Q4_WARPS;
    for (uint64_t job = (uint64_t)blockIdx.x * IK_Q4_WARPS + threadIdx.x / IK_Q4_WARP;
         job < jobs; job += stride) {
        const unsigned tile = job / row_groups, row = job % row_groups * R;
        const unsigned expert = tile_experts[tile], begin = starts[tile];
        int selected[C];
        #pragma unroll
        for (unsigned c = 0; c < C; c++) {
            selected[c] = begin + c < (unsigned)counts[expert]
                ? buckets[(uint64_t)expert * assignments + begin + c] : -1;
        }
        float acc[R][C] = {};
        #pragma unroll
        for (unsigned warp = 0; warp < W; warp++) {
            const unsigned tid = warp * IK_Q4_WARP + lane;
            const unsigned iqs = VDR_Q4_K_Q8_1_MMVQ * (tid % TPB);
            const unsigned offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
            float partial[R][C] = {};
            #pragma unroll
            for (unsigned it = 0; it < ITERS; it++) {
                const unsigned bx = tid / TPB + it * STEP;
                InklingQ4Frag frag[R];
                #pragma unroll
                for (unsigned r = 0; r < R; r++) {
                    frag[r] = inkling_q4_load(weights + ((uint64_t)expert * m + row + r) * blocks + bx, iqs);
                }
                #pragma unroll
                for (unsigned c = 0; c < C; c++) {
                    if (selected[c] < 0) { continue; }
                    const auto *xr = x + (uint64_t)(selected[c] / used) * x_blocks + bx * (QK_K / QK8_1) + offset;
                    int u[2 * QR4_K];
                    float d8[QR4_K];
                    #pragma unroll
                    for (unsigned i = 0; i < QR4_K; i++) {
                        const int *q = (const int *)xr[i].qs + ((iqs / 2) % 4);
                        u[2 * i] = q[0]; u[2 * i + 1] = q[4];
                        d8[i] = __low2float(xr[i].ds);
                    }
                    #pragma unroll
                    for (unsigned r = 0; r < R; r++) {
                        const uint8_t *sc = (const uint8_t *)frag[r].aux;
                        const float v = vec_dot_q4_K_q8_1_impl_vmmq(frag[r].v, u, sc, sc + 2, frag[r].dm, d8);
                        partial[r][c] = __fadd_rn(partial[r][c], v);
                    }
                }
            }
            #pragma unroll
            for (unsigned r = 0; r < R; r++) {
                #pragma unroll
                for (unsigned c = 0; c < C; c++) {
                    acc[r][c] = warp == 0 ? partial[r][c] : __fadd_rn(acc[r][c], partial[r][c]);
                }
            }
        }
        #pragma unroll
        for (unsigned r = 0; r < R; r++) {
            #pragma unroll
            for (unsigned c = 0; c < C; c++) {
                const float v = warp_reduce_sum<IK_Q4_WARP>(acc[r][c]);
                if (lane == 0 && selected[c] >= 0) {
                    out[(uint64_t)selected[c] * m + row + r] = isfinite(v) ? v : 0.0f;
                }
            }
        }
    }
}

template<unsigned R, unsigned C, unsigned W, unsigned I>
static int inkling_q4_launch(const void *w, const block_q8_1 *x, float *out,
        const int32_t *counts, const int32_t *buckets, const int32_t *experts,
        const int32_t *starts, unsigned m, unsigned k, unsigned assignments,
        unsigned ne, unsigned used, unsigned sms, cudaStream_t stream) {
    if (m % R) { return -1; }
    static int active = 0;
    if (!active && (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
            inkling_q4_kernel<R, C, W, I>, IK_Q4_WARPS * IK_Q4_WARP, 0) != cudaSuccess || active <= 0)) { return -2; }
    inkling_q4_kernel<R, C, W, I><<<sms * active, IK_Q4_WARPS * IK_Q4_WARP, 0, stream>>>(
        (const block_q4_K *)w, x, out, counts, buckets, experts, starts, m, k, assignments, ne, used);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
