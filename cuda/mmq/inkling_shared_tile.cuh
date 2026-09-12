/* Shared expert prefill: keep weights in registers across all routed columns.
 * Each warp replays the MMVQ K partitions, then merges them in their original
 * order. A CTA stages the same eight inputs for every output row it owns.
 *
 * Slab layouts (same byte count, chosen at launch):
 *   IK_ST_ROWS    canonical 36-byte Q8_1 blocks copied verbatim (retained control)
 *   IK_ST_SOA     qs[c][k] int8 then ds[c][k/32] half2, split while staging:
 *                 one 8-byte LDS feeds both dp4a and a K step is conflict-free
 *   IK_ST_COLUMN  the SoA slab with float scales converted once while staging,
 *                 and the K loop run once per column so only one column's
 *                 partial and merged sums stay live (no register spills)
 * IK_ST_SOA and IK_ST_COLUMN keep block deltas as half2 until each
 * (row, block) use; the conversion equals the load-time float, so outputs
 * stay byte-identical while the smaller register footprint doubles resident
 * CTAs. IK_ST_SOA at 128 registers still spilled 144 bytes; per-column passes
 * cut the up tile 6.4 -> 5.7 ms and down 3.4 -> 2.9 ms at 1024 tokens. */
enum { IK_ST_WARP = 32, IK_ST_COLS = 8, IK_ST_WORDS = sizeof(block_q8_1) / sizeof(int), IK_ST_MIN = 64 };
enum IkStSlab { IK_ST_ROWS, IK_ST_SOA, IK_ST_COLUMN };

template<unsigned R, unsigned WARPS, unsigned PARTS, unsigned STEPS, IkStSlab SLAB>
__launch_bounds__(WARPS * IK_ST_WARP, (SLAB == IK_ST_ROWS ? 1 : 2) * (PARTS == 1 ? 2 : 1))
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
    constexpr bool PACKED = SLAB != IK_ST_ROWS;
    constexpr unsigned PASSES = SLAB == IK_ST_COLUMN ? IK_ST_COLS : 1, CP = IK_ST_COLS / PASSES;
    int2 payload[R][PARTS][STEPS];
    float delta[PACKED ? 1 : R][PARTS][STEPS];
    half2 delta2[PACKED ? R : 1][PARTS][STEPS / 2];
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
                if constexpr (PACKED) {
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
        if constexpr (PACKED) {
            // One 36-byte block per pass: nine word loads, then eight qs words
            // as int2 pairs and the delta (half2, or its float once for the
            // column layout). Nine words in flight keep the 128-register
            // budget; blocks is a power of two (launch check).
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
                if constexpr (SLAB == IK_ST_COLUMN) {
                    ((float *)slab_ds)[c * blocks + j] = __low2float(*(const half2 *)&w[0]);
                } else {
                    slab_ds[c * blocks + j] = *(const half2 *)&w[0];
                }
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

        // The column layout runs the K loop once per column; the rolled loop
        // keeps one column's partial and merged sums live.
        #pragma unroll 1
        for (unsigned pass = 0; pass < PASSES; pass++) {
        float acc[R][CP] = {};
        #pragma unroll
        for (unsigned q = 0; q < PARTS; q++) {
            float partial[R][CP] = {};
            #pragma unroll
            for (unsigned i = 0; i < STEPS; i++) {
                const unsigned bx = (q * IK_ST_WARP + lane) / 4 + i * PARTS * IK_ST_WARP / 4;
                float dw[R];
                #pragma unroll
                for (unsigned r = 0; r < R; r++) {
                    if constexpr (PACKED) {
                        dw[r] = i % 2 == 0 ? __low2float(delta2[r][q][i / 2])
                                           : __high2float(delta2[r][q][i / 2]);
                    } else {
                        dw[r] = delta[r][q][i];
                    }
                }
                #pragma unroll
                for (unsigned cc = 0; cc < CP; cc++) {
                    const unsigned c = pass * CP + cc;
                    float xs;
                    int u0, u1;
                    if constexpr (PACKED) {
                        const int2 u = *(const int2 *)(slab + c * qs_words + bx * (QK8_1 / 4) + 2 * (lane % 4));
                        u0 = u.x; u1 = u.y;
                        xs = SLAB == IK_ST_COLUMN ? ((const float *)slab_ds)[c * blocks + bx]
                                                  : __low2float(slab_ds[c * blocks + bx]);
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
                        partial[r][cc] = fmaf(scale, (float)dot, partial[r][cc]);
                    }
                }
            }
            #pragma unroll
            for (unsigned r = 0; r < R; r++) {
                #pragma unroll
                for (unsigned cc = 0; cc < CP; cc++) {
                    acc[r][cc] = q == 0 ? partial[r][cc] : __fadd_rn(acc[r][cc], partial[r][cc]);
                }
            }
        }

        #pragma unroll
        for (unsigned r = 0; r < R; r++) {
            #pragma unroll
            for (unsigned cc = 0; cc < CP; cc++) {
                const unsigned c = pass * CP + cc;
                const float value = warp_reduce_sum<IK_ST_WARP>(acc[r][cc]);
                if (lane == 0 && selected[c] >= 0 && row0 + r < m) {
                    out[(uint64_t)selected[c] * m + row0 + r] = isfinite(value) ? value : 0.0f;
                }
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
    // The SoA slabs index blocks by shift. DS4_INKLING_NO_SHARED_SOA restores
    // the canonical-row slab and float deltas; DS4_INKLING_NO_SHARED_COLUMN
    // keeps the SoA slab but sums all eight columns in one K loop.
    const unsigned blocks = k / QK8_1;
    const IkStSlab slab = (blocks & (blocks - 1)) || getenv("DS4_INKLING_NO_SHARED_SOA") ? IK_ST_ROWS
        : getenv("DS4_INKLING_NO_SHARED_COLUMN") ? IK_ST_SOA : IK_ST_COLUMN;
    #define IK_ST_LAUNCH(P, S, L) inkling_shared_tile_kernel<ROWS, WARPS, P, S, L> \
        <<<grid, WARPS * IK_ST_WARP, slab_bytes, stream>>>( \
        (const block_q8_0 *)weights, x, out, counts, buckets, m, k, assignments, used)
    #define IK_ST_SELECT(P, S) do { \
        if (slab == IK_ST_COLUMN) { IK_ST_LAUNCH(P, S, IK_ST_COLUMN); } \
        else if (slab == IK_ST_SOA) { IK_ST_LAUNCH(P, S, IK_ST_SOA); } \
        else { IK_ST_LAUNCH(P, S, IK_ST_ROWS); } } while (0)
    if (used > 1) { IK_ST_SELECT(4, 4); } else { IK_ST_SELECT(1, 8); }
    #undef IK_ST_SELECT
    #undef IK_ST_LAUNCH
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
