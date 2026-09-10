/* Inkling prefill: share expert weights without changing MMVQ reductions.
 * Included by mmvq.cu after its type traits and warp reduction helpers. */
enum {
    IK_MMVQ_COLUMNS = 4,
    IK_MMVQ_ROWS = 2,
    IK_MMVQ_THREADS = 256,
    IK_MMVQ_RESIDENT_THREADS = 1024,
    IK_MMVQ_EXPERTS = 256,
    IK_MMVQ_ASSIGNMENTS = 2048 * 6,
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

static __global__ void inkling_tiles_kernel(
        int32_t *counts, int32_t *tile_experts, int32_t *tile_starts,
        uint32_t experts) {
    __shared__ uint32_t offsets[IK_MMVQ_EXPERTS];
    if (threadIdx.x == 0) {
        uint32_t total = 0;
        for (uint32_t e = 0; e < experts; e++) {
            offsets[e] = total;
            total += (counts[e] + IK_MMVQ_COLUMNS - 1) / IK_MMVQ_COLUMNS;
        }
        counts[experts] = total;
    }
    __syncthreads();
    const uint32_t e = threadIdx.x;
    if (e >= experts) { return; }
    for (int32_t at = 0; at < counts[e]; at += IK_MMVQ_COLUMNS) {
        const uint32_t tile = offsets[e] + at / IK_MMVQ_COLUMNS;
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

uint64_t ds4_mmvq_inkling_bytes(int rows, int experts, int used) {
    if (rows <= 0 || experts <= 0 || experts > IK_MMVQ_EXPERTS || used <= 0 ||
        used > experts || rows > IK_MMVQ_ASSIGNMENTS / used) { return 0; }
    const uint64_t assignments = (uint64_t)rows * used;
    return ((experts + 1) + assignments * (experts + 2)) * sizeof(int32_t);
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
    inkling_tiles_kernel<<<1, IK_MMVQ_THREADS, 0, stream>>>(counts, tile_experts, tile_starts, experts);
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
