/* Shared expert prefill: keep weights in registers across all routed columns.
 * Each warp replays the MMVQ K partitions, then merges them in their original
 * order. A CTA stages the same eight inputs for every output row it owns. */
enum { IK_ST_WARP = 32, IK_ST_COLS = 8, IK_ST_WORDS = sizeof(block_q8_1) / sizeof(int), IK_ST_MIN = 64 };

template<unsigned R, unsigned WARPS, unsigned PARTS, unsigned STEPS>
__launch_bounds__(WARPS * IK_ST_WARP, PARTS == 1 ? 2 : 1)
static __global__ void inkling_shared_tile_kernel(
        const block_q8_0 *weights, const block_q8_1 *x, float *out,
        const int32_t *counts, const int32_t *buckets,
        unsigned m, unsigned k, unsigned assignments, unsigned used) {
    const unsigned expert = blockIdx.y, lane = threadIdx.x % IK_ST_WARP;
    const unsigned row0 = (blockIdx.x * WARPS + threadIdx.x / IK_ST_WARP) * R;
    const unsigned blocks = k / QK8_1, n = counts[expert];
    const unsigned row_words = blocks * IK_ST_WORDS, row_vectors = row_words / 4;
    extern __shared__ int4 slab4[];
    int *slab = (int *)slab4;
    __shared__ int32_t selected[IK_ST_COLS];
    int2 payload[R][PARTS][STEPS];
    float delta[R][PARTS][STEPS];

    // The last CTA may own padded rows; clamp loads and guard their stores.
    #pragma unroll
    for (unsigned r = 0; r < R; r++) {
        #pragma unroll
        for (unsigned q = 0; q < PARTS; q++) {
            #pragma unroll
            for (unsigned i = 0; i < STEPS; i++) {
                const unsigned bx = (q * IK_ST_WARP + lane) / 4 + i * PARTS * IK_ST_WARP / 4;
                const auto *b = weights + ((uint64_t)expert * m + min(row0 + r, m - 1)) * blocks + bx;
                payload[r][q][i] = make_int2(get_int_b2(b->qs, 2 * (lane % 4)),
                                           get_int_b2(b->qs, 2 * (lane % 4) + 1));
                delta[r][q][i] = __half2float(b->d);
            }
        }
    }

    for (unsigned begin = 0; begin < n; begin += IK_ST_COLS) {
        __syncthreads();
        if (threadIdx.x < IK_ST_COLS) {
            selected[threadIdx.x] = begin + threadIdx.x < n
                ? buckets[(uint64_t)expert * assignments + begin + threadIdx.x] : -1;
        }
        __syncthreads();
        // Canonical Q8_1 rows at K=2048/4096 have a 16-byte aligned row stride.
        for (unsigned at = threadIdx.x; at < IK_ST_COLS * row_vectors; at += WARPS * IK_ST_WARP) {
            const unsigned c = at / row_vectors, j = at % row_vectors;
            slab4[at] = selected[c] >= 0
                ? ((const int4 *)(x + (uint64_t)(selected[c] / used) * blocks))[j]
                : make_int4(0, 0, 0, 0);
        }
        __syncthreads();

        float acc[R][IK_ST_COLS] = {};
        #pragma unroll
        for (unsigned q = 0; q < PARTS; q++) {
            float partial[R][IK_ST_COLS] = {};
            #pragma unroll
            for (unsigned i = 0; i < STEPS; i++) {
                const unsigned bx = (q * IK_ST_WARP + lane) / 4 + i * PARTS * IK_ST_WARP / 4;
                #pragma unroll
                for (unsigned c = 0; c < IK_ST_COLS; c++) {
                    const int *b = slab + c * row_words + bx * IK_ST_WORDS;
                    const float xd = __low2float(*(const half2 *)b);
                    const int u0 = b[1 + 2 * (lane % 4)], u1 = b[2 + 2 * (lane % 4)];
                    #pragma unroll
                    for (unsigned r = 0; r < R; r++) {
                        int dot = ggml_cuda_dp4a(payload[r][q][i].x, u0, 0);
                        dot = ggml_cuda_dp4a(payload[r][q][i].y, u1, dot);
                        const float scale = __fmul_rn(delta[r][q][i], xd);
                        partial[r][c] = fmaf(scale, (float)dot, partial[r][c]);
                    }
                }
            }
            #pragma unroll
            for (unsigned r = 0; r < R; r++) {
                #pragma unroll
                for (unsigned c = 0; c < IK_ST_COLS; c++) {
                    acc[r][c] = q == 0 ? partial[r][c] : __fadd_rn(acc[r][c], partial[r][c]);
                }
            }
        }

        #pragma unroll
        for (unsigned r = 0; r < R; r++) {
            #pragma unroll
            for (unsigned c = 0; c < IK_ST_COLS; c++) {
                const float value = warp_reduce_sum<IK_ST_WARP>(acc[r][c]);
                if (lane == 0 && selected[c] >= 0 && row0 + r < m) {
                    out[(uint64_t)selected[c] * m + row0 + r] = isfinite(value) ? value : 0.0f;
                }
            }
        }
    }
}

// Return 1 before launch when shared memory cannot hold the activation slab.
static int inkling_shared_tile_launch(
        const void *weights, const block_q8_1 *x, float *out,
        const int32_t *counts, const int32_t *buckets, unsigned m,
        unsigned k, unsigned assignments, unsigned used, cudaStream_t stream) {
    constexpr unsigned ROWS = 2, WARPS = 8;
    const size_t slab_bytes = IK_ST_COLS * (k / QK8_1) * sizeof(block_q8_1);
    const auto &device = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if ((uintptr_t)x % alignof(int4) ||
        device.smpb < slab_bytes + IK_ST_COLS * sizeof(int32_t)) { return 1; }
    const dim3 grid((m + ROWS * WARPS - 1) / (ROWS * WARPS), IK_SHARED_EXPERTS);
    if (used > 1) {
        inkling_shared_tile_kernel<ROWS, WARPS, 4, 4><<<grid, WARPS * IK_ST_WARP, slab_bytes, stream>>>(
            (const block_q8_0 *)weights, x, out, counts, buckets, m, k, assignments, used);
    } else {
        inkling_shared_tile_kernel<ROWS, WARPS, 1, 8><<<grid, WARPS * IK_ST_WARP, slab_bytes, stream>>>(
            (const block_q8_0 *)weights, x, out, counts, buckets, m, k, assignments, used);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
