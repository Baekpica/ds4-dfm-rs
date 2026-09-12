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
#include <cuda_pipeline.h>

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

/* Column pipeline (release path): the same resident weights and per-output
 * arithmetic as IK_ST_COLUMN, but the activations come from the float-scale
 * SoA and stream through a ring of STAGES column buffers that cp.async fills
 * AHEAD columns early, so staging overlaps compute and needs no registers.
 * Buffer (j + AHEAD) % STAGES was last read for column j + AHEAD - STAGES,
 * which every warp finished before the previous iteration's barrier.
 *
 *   iteration j: issue j+AHEAD -> wait column j -> barrier -> K loop over j
 *
 * Up keeps two rows per warp at two CTAs per SM (4.96 vs 5.95 ms); down
 * takes four rows per four-warp CTA at four CTAs (2.44 vs 3.09 ms), halving
 * shared-memory bytes per output where the LDS pipe was saturated. */
enum { IK_SP_STAGES = 4, IK_SP_AHEAD = 2, IK_SP_UP_ROWS = 2, IK_SP_UP_WARPS = 8, IK_SP_UP_BLOCKS = 2,
       IK_SP_DOWN_ROWS = 4, IK_SP_DOWN_WARPS = 4, IK_SP_DOWN_BLOCKS = 4 };

template<unsigned R, unsigned WARPS, unsigned PARTS, unsigned STEPS, unsigned MINB>
__launch_bounds__(WARPS * IK_ST_WARP, MINB)
static __global__ void inkling_shared_pipe_kernel(
        const block_q8_0 *weights, const int8_t *xq, const float *xd, float *out,
        const int32_t *counts, const int32_t *buckets,
        unsigned m, unsigned k, unsigned assignments, unsigned used) {
    const unsigned expert = blockIdx.y, lane = threadIdx.x % IK_ST_WARP;
    const unsigned row0 = (blockIdx.x * WARPS + threadIdx.x / IK_ST_WARP) * R;
    const unsigned blocks = k / QK8_1, n = counts[expert];
    const unsigned col_words = blocks * (QK8_1 / 4) + blocks;
    extern __shared__ int4 ring4[];
    int *ring = (int *)ring4;
    int2 payload[R][PARTS][STEPS];
    half2 delta2[R][PARTS][STEPS / 2];
    if (n == 0) { return; }

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
                if (i % 2 == 0) { delta2[r][q][i / 2].x = b->d; }
                else { delta2[r][q][i / 2].y = b->d; }
            }
        }
    }

    // Column j of this expert: its activation row's qs then float scales.
    auto issue = [&](unsigned j) {
        if (j >= n) { return; }
        const uint64_t row = (uint64_t)(buckets[(uint64_t)expert * assignments + j] / used);
        int4 *dst = (int4 *)(ring + (j % IK_SP_STAGES) * col_words);
        const int4 *qs = (const int4 *)(xq + row * k);
        const int4 *ds = (const int4 *)(xd + row * blocks);
        const unsigned qs_vectors = blocks * (QK8_1 / sizeof(int4)), ds_vectors = blocks / 4;
        for (unsigned v = threadIdx.x; v < qs_vectors + ds_vectors; v += WARPS * IK_ST_WARP) {
            __pipeline_memcpy_async(dst + v, v < qs_vectors ? qs + v : ds + (v - qs_vectors), sizeof(int4));
        }
    };
    #pragma unroll
    for (unsigned a = 0; a < IK_SP_AHEAD; a++) { issue(a); __pipeline_commit(); }

    for (unsigned j = 0; j < n; j++) {
        issue(j + IK_SP_AHEAD);
        __pipeline_commit();
        __pipeline_wait_prior(IK_SP_AHEAD);
        __syncthreads();
        const int *col = ring + (j % IK_SP_STAGES) * col_words;
        const float *scales = (const float *)(col + blocks * (QK8_1 / 4));
        float acc[R] = {};
        #pragma unroll
        for (unsigned q = 0; q < PARTS; q++) {
            float partial[R] = {};
            #pragma unroll
            for (unsigned i = 0; i < STEPS; i++) {
                const unsigned bx = (q * IK_ST_WARP + lane) / 4 + i * PARTS * IK_ST_WARP / 4;
                const int2 u = *(const int2 *)(col + bx * (QK8_1 / 4) + 2 * (lane % 4));
                const float xs = scales[bx];
                #pragma unroll
                for (unsigned r = 0; r < R; r++) {
                    const float dw = i % 2 == 0 ? __low2float(delta2[r][q][i / 2])
                                                : __high2float(delta2[r][q][i / 2]);
                    int dot = ggml_cuda_dp4a(payload[r][q][i].x, u.x, 0);
                    dot = ggml_cuda_dp4a(payload[r][q][i].y, u.y, dot);
                    const float scale = __fmul_rn(dw, xs);
                    partial[r] = fmaf(scale, (float)dot, partial[r]);
                }
            }
            #pragma unroll
            for (unsigned r = 0; r < R; r++) { acc[r] = q == 0 ? partial[r] : __fadd_rn(acc[r], partial[r]); }
        }
        const int32_t selected = buckets[(uint64_t)expert * assignments + j];
        #pragma unroll
        for (unsigned r = 0; r < R; r++) {
            const float value = warp_reduce_sum<IK_ST_WARP>(acc[r]);
            if (lane == 0 && row0 + r < m) {
                out[(uint64_t)selected * m + row0 + r] = isfinite(value) ? value : 0.0f;
            }
        }
    }
}

// Return 1 when the ring does not fit or the SoA rows are not 16-byte aligned.
static int inkling_shared_pipe_launch(
        const void *weights, const int8_t *xq, const float *xd, float *out,
        const int32_t *counts, const int32_t *buckets, unsigned m, unsigned k,
        unsigned assignments, unsigned used, unsigned experts, cudaStream_t stream) {
    const unsigned blocks = k / QK8_1;
    const size_t ring_bytes = (size_t)IK_SP_STAGES * (blocks * QK8_1 + blocks * sizeof(float));
    const auto &device = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if (!experts || experts > IK_MMVQ_EXPERTS || k % (4 * QK8_1) ||
        (uintptr_t)xq % alignof(int4) || (uintptr_t)xd % alignof(int4) ||
        device.smpb < ring_bytes) { return 1; }
    #define IK_SP_LAUNCH(R, W, P, S, B) inkling_shared_pipe_kernel<R, W, P, S, B> \
        <<<dim3((m + R * W - 1) / (R * W), experts), W * IK_ST_WARP, ring_bytes, stream>>>( \
        (const block_q8_0 *)weights, xq, xd, out, counts, buckets, m, k, assignments, used)
    if (used > 1) { IK_SP_LAUNCH(IK_SP_UP_ROWS, IK_SP_UP_WARPS, 4, 4, IK_SP_UP_BLOCKS); }
    else { IK_SP_LAUNCH(IK_SP_DOWN_ROWS, IK_SP_DOWN_WARPS, 1, 8, IK_SP_DOWN_BLOCKS); }
    #undef IK_SP_LAUNCH
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
