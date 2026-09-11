/* Shared expert prefill: keep weights in registers across all routed columns.
 * Each warp replays the MMVQ K partitions, then merges them in their original
 * order. A CTA stages the same eight inputs for every output row it owns.
 *
 * Slab layouts (same byte count, chosen at launch):
 *   IK_ST_ROWS  canonical 36-byte Q8_1 blocks copied verbatim (retained control)
 *   IK_ST_SOA   qs[c][k] int8 then ds[c][k/32] half2, split while staging:
 *               one 8-byte LDS feeds both dp4a and a K step is conflict-free
 * IK_ST_SOA also keeps block deltas as half2 until each (row, block) use;
 * the conversion equals the load-time float, so outputs stay byte-identical
 * while the smaller register footprint doubles resident CTAs. */
enum { IK_ST_WARP = 32, IK_ST_COLS = 8, IK_ST_WORDS = sizeof(block_q8_1) / sizeof(int), IK_ST_MIN = 64 };
enum IkStSlab { IK_ST_ROWS, IK_ST_SOA };

template<unsigned R, unsigned WARPS, unsigned PARTS, unsigned STEPS, IkStSlab SLAB>
__launch_bounds__(WARPS * IK_ST_WARP, (SLAB == IK_ST_SOA ? 2 : 1) * (PARTS == 1 ? 2 : 1))
static __global__ void inkling_shared_tile_kernel(
        const block_q8_0 *weights, const block_q8_1 *x, float *out,
        const int32_t *counts, const int32_t *buckets,
        unsigned m, unsigned k, unsigned assignments, unsigned used) {
    const unsigned expert = blockIdx.y, lane = threadIdx.x % IK_ST_WARP;
    const unsigned row0 = (blockIdx.x * WARPS + threadIdx.x / IK_ST_WARP) * R;
    const unsigned blocks = k / QK8_1, n = counts[expert];
    const unsigned row_words = blocks * IK_ST_WORDS, row_vectors = row_words / 4;
    const unsigned qs_words = blocks * (QK8_1 / 4), block_shift = __ffs(blocks) - 1;
    extern __shared__ int4 slab4[];
    int *slab = (int *)slab4;
    half2 *slab_ds = (half2 *)(slab + IK_ST_COLS * qs_words);
    __shared__ int32_t selected[IK_ST_COLS];
    int2 payload[R][PARTS][STEPS];
    float delta[SLAB == IK_ST_SOA ? 1 : R][PARTS][STEPS];
    half2 delta2[SLAB == IK_ST_SOA ? R : 1][PARTS][STEPS / 2];
    // Routed Q8 has 256 experts; most CTAs own an empty bucket and must not
    // load another expert's weights. Shared experts keep n > 0.
    if (n == 0) { return; }

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
                if constexpr (SLAB == IK_ST_SOA) {
                    if (i % 2 == 0) { delta2[r][q][i / 2].x = b->d; }
                    else { delta2[r][q][i / 2].y = b->d; }
                } else {
                    delta[r][q][i] = __half2float(b->d);
                }
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
        if constexpr (SLAB == IK_ST_SOA) {
            // One 36-byte block per pass: nine word loads, then eight qs words
            // as int2 pairs and the half2 delta. Nine words in flight keep the
            // 128-register budget; blocks is a power of two (launch check).
            for (unsigned at = threadIdx.x; at < IK_ST_COLS * blocks; at += WARPS * IK_ST_WARP) {
                const unsigned c = at >> block_shift, j = at & (blocks - 1);
                int w[IK_ST_WORDS] = {};
                if (selected[c] >= 0) {
                    const int *src = (const int *)(x + (uint64_t)(selected[c] / used) * blocks + j);
                    #pragma unroll
                    for (unsigned t = 0; t < IK_ST_WORDS; t++) { w[t] = src[t]; }
                }
                int2 *qs = (int2 *)(slab + c * qs_words + j * (QK8_1 / 4));
                #pragma unroll
                for (unsigned t = 0; t < QK8_1 / 8; t++) { qs[t] = make_int2(w[1 + 2 * t], w[2 + 2 * t]); }
                slab_ds[c * blocks + j] = *(const half2 *)&w[0];
            }
        } else {
            // Canonical Q8_1 rows at K=2048/4096 have a 16-byte aligned row stride.
            for (unsigned at = threadIdx.x; at < IK_ST_COLS * row_vectors; at += WARPS * IK_ST_WARP) {
                const unsigned c = at / row_vectors, j = at % row_vectors;
                slab4[at] = selected[c] >= 0
                    ? ((const int4 *)(x + (uint64_t)(selected[c] / used) * blocks))[j]
                    : make_int4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        float acc[R][IK_ST_COLS] = {};
        #pragma unroll
        for (unsigned q = 0; q < PARTS; q++) {
            float partial[R][IK_ST_COLS] = {};
            #pragma unroll
            for (unsigned i = 0; i < STEPS; i++) {
                const unsigned bx = (q * IK_ST_WARP + lane) / 4 + i * PARTS * IK_ST_WARP / 4;
                float dw[R];
                #pragma unroll
                for (unsigned r = 0; r < R; r++) {
                    if constexpr (SLAB == IK_ST_SOA) {
                        dw[r] = i % 2 == 0 ? __low2float(delta2[r][q][i / 2])
                                           : __high2float(delta2[r][q][i / 2]);
                    } else {
                        dw[r] = delta[r][q][i];
                    }
                }
                #pragma unroll
                for (unsigned c = 0; c < IK_ST_COLS; c++) {
                    float xs;
                    int u0, u1;
                    if constexpr (SLAB == IK_ST_SOA) {
                        const int2 u = *(const int2 *)(slab + c * qs_words + bx * (QK8_1 / 4) + 2 * (lane % 4));
                        u0 = u.x; u1 = u.y;
                        xs = __low2float(slab_ds[c * blocks + bx]);
                    } else {
                        const int *b = slab + c * row_words + bx * IK_ST_WORDS;
                        xs = __low2float(*(const half2 *)b);
                        u0 = b[1 + 2 * (lane % 4)]; u1 = b[2 + 2 * (lane % 4)];
                    }
                    #pragma unroll
                    for (unsigned r = 0; r < R; r++) {
                        int dot = ggml_cuda_dp4a(payload[r][q][i].x, u0, 0);
                        dot = ggml_cuda_dp4a(payload[r][q][i].y, u1, dot);
                        const float scale = __fmul_rn(dw[r], xs);
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
        unsigned k, unsigned assignments, unsigned used, unsigned experts,
        cudaStream_t stream) {
    constexpr unsigned ROWS = 2, WARPS = 8;
    const size_t slab_bytes = IK_ST_COLS * (k / QK8_1) * sizeof(block_q8_1);
    const auto &device = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if (!experts || experts > IK_MMVQ_EXPERTS || (uintptr_t)x % alignof(int4) ||
        device.smpb < slab_bytes + IK_ST_COLS * sizeof(int32_t)) { return 1; }
    const dim3 grid((m + ROWS * WARPS - 1) / (ROWS * WARPS), experts);
    // The SoA slab indexes blocks by shift; the switch restores the
    // canonical-row slab and float deltas for A/B controls.
    const unsigned blocks = k / QK8_1;
    const bool soa = (blocks & (blocks - 1)) == 0 && !getenv("DS4_INKLING_NO_SHARED_SOA");
    #define IK_ST_LAUNCH(P, S, L) inkling_shared_tile_kernel<ROWS, WARPS, P, S, L> \
        <<<grid, WARPS * IK_ST_WARP, slab_bytes, stream>>>( \
        (const block_q8_0 *)weights, x, out, counts, buckets, m, k, assignments, used)
    if (used > 1) {
        if (soa) { IK_ST_LAUNCH(4, 4, IK_ST_SOA); } else { IK_ST_LAUNCH(4, 4, IK_ST_ROWS); }
    } else {
        if (soa) { IK_ST_LAUNCH(1, 8, IK_ST_SOA); } else { IK_ST_LAUNCH(1, 8, IK_ST_ROWS); }
    }
    #undef IK_ST_LAUNCH
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
