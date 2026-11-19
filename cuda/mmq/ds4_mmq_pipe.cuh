// ds4: software-pipelined K loop for the compact routed worklist kernel
// (ds4_moe_worklist_mmq_kernel in ds4_mmq.cu).  Included after mmq.cuh.
//
// ncu on the K2-Horizon 8K prefill (1024-token chunks, IQ1_S gate/up and
// IQ2_XXS down through the worklist kernel): one 256-thread block per SM
// (255 registers, 58 KB of shared memory against a 100 KB SM), 61-74 % of
// the issue slots idle, and the dominant stall is the long scoreboard, i.e.
// warps waiting for the global loads that mul_mat_q_process_tile issues at
// the top of every K iteration and consumes right away.  Memory throughput
// sits at 31-37 % and the tensor pipe near 35 %: nothing is saturated, the
// loop simply serialises "load weights -> dequantize -> load activations ->
// MMA" and exposes one DRAM round trip (plus two L2 round trips for the
// activation halves) per iteration, four barriers apart.
//
// This loop keeps the upstream dequantization and MMA dots and only changes
// when the bytes arrive:
//   * the raw block bytes each thread needs for the *next* K iteration are
//     fetched into registers right after the current tile has been expanded
//     into shared memory, so the DRAM latency overlaps the MMA phase;
//   * both 128-value halves of the activation tile are copied with cp.async
//     into a two-stage shared buffer one iteration ahead, so the L2 latency
//     overlaps too and the barrier between the halves goes away
//     (4 -> 2 barriers per iteration).
// The expanded tile (tile_x) and the dots are the upstream ones, so the
// output is bit-identical to the non-pipelined loop; DS4_MMQ_PIPE=0 is the
// kill switch and tests/test_exaone_kernels test_moe_pipe checks the bytes.
//
// Only the block types the K2 recipe routes through the worklist have a
// fetch/expand split (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS; all qk = 256, one block
// per K iteration), plus Hy4's Q4_K / Q5_K in the 64-wide worklist.  Tile widths above 64 keep the upstream loop: the
// two-stage activation buffer of a 128-wide tile would push the block past
// the 99 KB shared-memory limit.
#pragma once

static constexpr int DS4_MMQ_PIPE_MAX_X = 64;   // widest pipelined tile
static constexpr int DS4_MMQ_PIPE_Y_STAGES = 2;

static constexpr __host__ __device__ bool ds4_mmq_pipe_kquant(ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K;
}

// Q4_K / Q5_K pipeline only in launches that ask for the 64-wide tile
// (Hy4's 1536 x 1536 experts); other K-quant callers keep the upstream loop.
static constexpr __host__ __device__ bool ds4_mmq_pipe_supported(ggml_type type) {
    return type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ||
           type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS ||
           ds4_mmq_pipe_kquant(type);
}

// Shared bytes of one pipelined tile: ids, the two-stage activation buffer
// (both halves of a K iteration per stage) and the upstream x tile.
static __host__ __device__ size_t ds4_mmq_pipe_nbytes_shared(
        ggml_type type, int mmq_x, int mmq_y) {
    const size_t nbs_ids = (size_t)mmq_x * sizeof(int);
    const size_t nbs_y = (size_t)DS4_MMQ_PIPE_Y_STAGES * 2 * mmq_x *
                         MMQ_TILE_Y_K * sizeof(int);
    const size_t nbs_x = (size_t)mmq_y * mmq_get_mma_tile_x_k(type) * sizeof(int);
    return nbs_ids + nbs_y + nbs_x;
}

#if defined(TURING_MMA_AVAILABLE)

static __device__ __forceinline__ void ds4_pipe_cp_async_16(
        void *smem_dst, const void *gmem_src) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(s), "l"(gmem_src));
}

static __device__ __forceinline__ void ds4_pipe_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
static __device__ __forceinline__ void ds4_pipe_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// Raw block bytes one thread owns for one K iteration, laid out exactly as
// the upstream loader reads them (same access widths, same lanes).
template <ggml_type type, int mmq_y, int nwarps, int warp_size>
struct ds4_pipe_raw;

template <int mmq_y, int nwarps, int warp_size>
struct ds4_pipe_raw<GGML_TYPE_IQ1_S, mmq_y, nwarps, warp_size> {
    static constexpr int threads_per_row = MMQ_ITER_K / (4 * QR1_S);
    static constexpr int nrows = warp_size / threads_per_row;
    static constexpr int nit = mmq_y / (nwarps * nrows);
    int  qs[nit];
    int  qh[nit];
    half d[nit];
};

template <int mmq_y, int nwarps, int warp_size>
struct ds4_pipe_raw<GGML_TYPE_IQ1_M, mmq_y, nwarps, warp_size> {
    static constexpr int threads_per_row = MMQ_ITER_K / (4 * QR1_M);
    static constexpr int nrows = warp_size / threads_per_row;
    static constexpr int nit = mmq_y / (nwarps * nrows);
    int      qs[nit];
    int      qh[nit];      // qh[2*kqsx] | qh[2*kqsx+1] << 8
    uint16_t sc[nit][4];
};

template <int mmq_y, int nwarps, int warp_size>
struct ds4_pipe_raw<GGML_TYPE_IQ2_XXS, mmq_y, nwarps, warp_size> {
    static constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XXS)) / 2;
    static constexpr int nrows = warp_size / threads_per_row;
    static constexpr int nit = mmq_y / (nwarps * nrows);
    int      q2[nit];
    uint32_t aux32[nit];
    half     d[nit];
};

template <int mmq_y, int nwarps, int warp_size>
struct ds4_pipe_raw<GGML_TYPE_IQ2_XS, mmq_y, nwarps, warp_size> {
    static constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XS)) / 2;
    static constexpr int nrows = warp_size / threads_per_row;
    static constexpr int nit = mmq_y / (nwarps * nrows);
    int2 q2[nit];
    int  ls[nit];
    half d[nit];
};

// K-quants load in two passes, as upstream: one quant int per lane and row,
// then two lanes per row for the 12 scale bytes and dm.
template <int mmq_y, int nwarps, int warp_size>
struct ds4_pipe_raw<GGML_TYPE_Q4_K, mmq_y, nwarps, warp_size> {
    static constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);
    static constexpr int nrows = warp_size / threads_per_row;
    static constexpr int nit = mmq_y / (nwarps * nrows);
    static constexpr int nit_dm = mmq_y / (nwarps * (warp_size / 2));
    int   qs[nit];
    int   sc[nit_dm][3];
    half2 dm[nit_dm];
};

template <int mmq_y, int nwarps, int warp_size>
struct ds4_pipe_raw<GGML_TYPE_Q5_K, mmq_y, nwarps, warp_size> {
    static constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_K);
    static constexpr int nrows = warp_size / threads_per_row;
    static constexpr int nit = mmq_y / (nwarps * nrows);
    static constexpr int nit_dm = mmq_y / (nwarps * (warp_size / 2));
    int   qs[nit];
    int   qh[nit];
    int   sc[nit_dm][3];
    half2 dm[nit_dm];
};

// Scale rows of the K-quant loaders: (i0 + warp row + lane/2) % mmq_y.
template <int mmq_y, bool need_check>
static __device__ __forceinline__ int ds4_pipe_dm_row(const int it, const int i_max) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int rows_per_warp = ggml_cuda_get_physical_warp_size() / 2;
    int i = (it*nwarps*rows_per_warp + threadIdx.y*rows_per_warp + threadIdx.x/2) % mmq_y;
    if (need_check) {
        i = min(i, i_max);
    }
    return i;
}

// Global -> registers.  Row selection and clamping are the upstream
// loader's; the block index kbx0 already includes the row-tile offset.
template <ggml_type type, int mmq_y, bool need_check, typename raw_t>
static __device__ __forceinline__ void ds4_pipe_fetch(
        raw_t &r, const char * __restrict__ x, const int kbx0,
        const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int threads_per_row = raw_t::threads_per_row;
    constexpr int nrows = raw_t::nrows;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int it = 0; it < raw_t::nit; ++it) {
        int i = it*(nwarps*nrows) + threadIdx.y*nrows + threadIdx.x/threads_per_row;
        if (need_check) {
            i = min(i, i_max);
        }
        if constexpr (type == GGML_TYPE_IQ1_S) {
            const block_iq1_s * bxi = (const block_iq1_s *) x + kbx0 + i*stride;
            r.qs[it] = get_int_b2(bxi->qs, kqsx);
            r.qh[it] = bxi->qh[kqsx];
            r.d[it]  = bxi->d;
        } else if constexpr (type == GGML_TYPE_IQ1_M) {
            const block_iq1_m * bxi = (const block_iq1_m *) x + kbx0 + i*stride;
            r.qs[it] = get_int_b4(bxi->qs, kqsx);
            r.qh[it] = (int)bxi->qh[2*kqsx + 0] | ((int)bxi->qh[2*kqsx + 1] << 8);
            const uint16_t * sc = (const uint16_t *) bxi->scales;
            r.sc[it][0] = sc[0];
            r.sc[it][1] = sc[1];
            r.sc[it][2] = sc[2];
            r.sc[it][3] = sc[3];
        } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
            const block_iq2_xxs * bxi = (const block_iq2_xxs *) x + kbx0 + i*stride;
            r.q2[it]    = get_int_b2(bxi->qs, 2*kqsx+0);
            r.aux32[it] = get_int_b2(bxi->qs, 2*kqsx+1);
            r.d[it]     = bxi->d;
        } else if constexpr (type == GGML_TYPE_Q4_K) {
            const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;
            r.qs[it] = get_int_b4(bxi->qs, kqsx);
        } else if constexpr (type == GGML_TYPE_Q5_K) {
            const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;
            r.qs[it] = get_int_b4(bxi->qs, kqsx);
            r.qh[it] = get_int_b4(bxi->qh, kqsx % (QI5_K/4));
        } else {
            static_assert(type == GGML_TYPE_IQ2_XS, "unsupported pipelined type");
            const block_iq2_xs * bxi = (const block_iq2_xs *) x + kbx0 + i*stride;
            r.q2[it] = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
            r.ls[it] = bxi->scales[kqsx];
            r.d[it]  = bxi->d;
        }
    }

    if constexpr (ds4_mmq_pipe_kquant(type)) {
        using block_t = std::conditional_t<type == GGML_TYPE_Q4_K, block_q4_K, block_q5_K>;
#pragma unroll
        for (int it = 0; it < raw_t::nit_dm; ++it) {
            const int i = ds4_pipe_dm_row<mmq_y, need_check>(it, i_max);
            const block_t * bxi = (const block_t *) x + kbx0 + i*stride;
            const int * scales = (const int *) bxi->scales;
            r.sc[it][0] = scales[0];
            r.sc[it][1] = scales[1];
            r.sc[it][2] = scales[2];
            r.dm[it]    = bxi->dm;
        }
    }
}

// Registers -> tile_x, the upstream load_tiles_* arithmetic verbatim (MMA
// layouts only; this path is compiled for Turing+ MMA).
template <ggml_type type, int mmq_y, bool need_check, typename raw_t>
static __device__ __forceinline__ void ds4_pipe_expand(
        const raw_t &r, int * __restrict__ x_tile, const int i_max) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int threads_per_row = raw_t::threads_per_row;
    constexpr int nrows = raw_t::nrows;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int it = 0; it < raw_t::nit; ++it) {
        int i = it*(nwarps*nrows) + threadIdx.y*nrows + threadIdx.x/threads_per_row;
        if (need_check) {
            i = min(i, i_max);
        }
        if constexpr (type == GGML_TYPE_IQ1_S) {
            int   * x_qs = (int   *)  x_tile;
            half2 * x_ds = (half2 *) (x_qs + MMQ_TILE_NE_K*2);

            const int       qs_packed = r.qs[it];
            const uint8_t * qs        = (const uint8_t *) &qs_packed;
            const int       qh        = r.qh[it];

#pragma unroll
            for (int l = 0; l < QR1_S/2; ++l) {
                const int grid = iq1s_grid_gpu[qs[l] | (((qh >> (3*l)) & 0x07) << 8)];

                const int grid0 = (grid >> 0) & 0x0F0F0F0F;
                const int grid1 = (grid >> 4) & 0x0F0F0F0F;

                x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 8*kqsx + (2*l+0)] = grid0;
                x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 8*kqsx + (2*l+1)] = grid1;
            }

            const float  d1q   = __half2float(r.d[it]) * (((qh >> 11) & 0x0E) + 1);
            const float  delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);

            x_ds[i*MMQ_MMA_TILE_X_K_Q8_1 + kqsx] = make_half2(d1q, d1q*delta);
        } else if constexpr (type == GGML_TYPE_IQ1_M) {
            int   * x_qs = (int   *)  x_tile;
            float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

            const int       qs_packed = r.qs[it];
            const uint8_t * qs        = (const uint8_t *) &qs_packed;

#pragma unroll
            for (int l = 0; l < QR1_M/2; ++l) {
                const int qhl  = ((r.qh[it] >> (8*(l/2))) & 0xFF) >> (4*(l % 2));
                const int grid = iq1s_grid_gpu[qs[l] | ((qhl & 0x07) << 8)];

                const int bias  = (qhl & 0x08) ? 0x09090909 : 0x07070707;
                const int grid0 = __vsub4(((grid >> 0) & 0x0F0F0F0F) << 3, bias);
                const int grid1 = __vsub4(((grid >> 4) & 0x0F0F0F0F) << 3, bias);

                x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 0)] = grid0;
                x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 1)] = grid1;
            }

            const uint16_t * sc = r.sc[it];
            iq1m_scale_t scale;
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
            const float d8  = __half2float(scale.f16) * (1.0f/8);
            const int   tmp = sc[kqsx/2] >> (6*(kqsx%2));
            const float d0  = d8 * (2*((tmp >> 0) & 0x07) + 1);
            const float d1  = d8 * (2*((tmp >> 3) & 0x07) + 1);

            x_df[i*MMQ_MMA_TILE_X_K_Q3_K + 2*kqsx+0] = d0;
            x_df[i*MMQ_MMA_TILE_X_K_Q3_K + 2*kqsx+1] = d1;
        } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
            int   * x_qs = (int   *)  x_tile;
            float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

            const int       q2    = r.q2[it];
            const uint8_t * aux8  = (const uint8_t *) &q2;
            const uint32_t  aux32 = r.aux32[it];

#pragma unroll
            for (int l = 0; l < QR2_XXS; ++l) {
                const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[l]];
                const uint32_t signs = unpack_ksigns(aux32 >> (7 * l));

                const int signs0 = __vcmpne4(signs & 0x08040201, 0);
                const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);

                const int signs1 = __vcmpne4(signs & 0x80402010, 0);
                const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);

                x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 0)] = grid0;
                x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 1)] = grid1;
            }

            const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
            const float d = r.d[it];
            x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = d * ls / 8; // (d * scale + d / 2) / 4
        } else if constexpr (type == GGML_TYPE_Q4_K) {
            int * x_qs = (int *) x_tile;
            const int qs0 = r.qs[it];
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(kqsx/8) + kqsx % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(kqsx/8) + kqsx % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
        } else if constexpr (type == GGML_TYPE_Q5_K) {
            int * x_qs = (int *) x_tile;
            const int ky = QR5_K*kqsx;
            const int ql = r.qs[it];
            const int ql0 = (ql >> 0) & 0x0F0F0F0F;
            const int ql1 = (ql >> 4) & 0x0F0F0F0F;
            const int qh = r.qh[it];
            const int qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
            const int qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;
            const int kq0 = ky - ky % (QI5_K/2) + kqsx % (QI5_K/4) + 0;
            const int kq1 = ky - ky % (QI5_K/2) + kqsx % (QI5_K/4) + QI5_K/4;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kq0] = ql0 | qh0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kq1] = ql1 | qh1;
        } else {
            static_assert(type == GGML_TYPE_IQ2_XS, "unsupported pipelined type");
            int   * x_qs = (int   *)  x_tile;
            float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

            const int2       q2_packed = r.q2[it];
            const uint16_t * q2        = (const uint16_t *) &q2_packed;

#pragma unroll
            for (int l = 0; l < QR2_XS; ++l) {
                const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l] & 0x1FF];
                const uint32_t signs = unpack_ksigns(q2[l] >> 9);

                const int signs0 = __vcmpne4(signs & 0x08040201, 0);
                const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

                const int signs1 = __vcmpne4(signs & 0x80402010, 0);
                const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

                x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 0)] = grid_l;
                x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 1)] = grid_h;
            }

            const int ls = r.ls[it];
            const float d = r.d[it];
            x_df[i*MMQ_MMA_TILE_X_K_Q3_K + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
            x_df[i*MMQ_MMA_TILE_X_K_Q3_K + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
        }
    }

    if constexpr (ds4_mmq_pipe_kquant(type)) {
        half2 * x_dm = (half2 *) ((int *) x_tile + 2*MMQ_TILE_NE_K);
        const int ksc = threadIdx.x % 2;
#pragma unroll
        for (int it = 0; it < raw_t::nit_dm; ++it) {
            const int i = ds4_pipe_dm_row<mmq_y, need_check>(it, i_max);
            const int s0 = r.sc[it][0], s1 = r.sc[it][1], s2 = r.sc[it][2];
            // unpack_scales_q45_K(scales, ksc) and (scales, ksc + 2) for
            // ksc in {0, 1}, selected without a runtime register index.
            const int sc32 = ksc ? (s2 & 0x0F0F0F0F) | ((s0 >> 2) & 0x30303030) : s0 & 0x3F3F3F3F;
            const int m32 = ksc ? ((s2 >> 4) & 0x0F0F0F0F) | ((s1 >> 2) & 0x30303030) : s1 & 0x3F3F3F3F;
            const uint8_t * sc8 = (const uint8_t *) &sc32;
            const uint8_t *  m8 = (const uint8_t *)  &m32;
            const half2 dm = r.dm[it] * make_half2(1.0f, -1.0f);
#pragma unroll
            for (int l = 0; l < int(sizeof(int)); ++l) {
                x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
            }
        }
    }
}

// Both 128-value halves of K iteration kb (block_q8_1_mmq rows 2kb and
// 2kb+1 of the mmq_x tile columns) -> one activation stage, 16 bytes per
// cp.async.  y already points at the tile's first column; every offset is a
// multiple of sizeof(block_q8_1_mmq) = 144 bytes, so the 16-byte alignment
// holds whenever the caller's y buffer is 16-byte aligned (it is a pool
// allocation).
template <int mmq_x, int nthreads>
static __device__ __forceinline__ void ds4_pipe_y_copy(
        int * __restrict__ stage, const int * __restrict__ y,
        const int ncols_y, const int kb, const int tid) {
    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);
    constexpr int half_ints = mmq_x * MMQ_TILE_Y_K;
    static_assert(MMQ_TILE_Y_K == sz, "activation tile row is one block_q8_1_mmq");
    static_assert(half_ints % 4 == 0, "16-byte cp.async chunks");
    constexpr int chunks = half_ints / 4;
    const int * by0 = y + ncols_y * (2*kb) * sz;
    const int * by1 = by0 + ncols_y * sz;
#pragma unroll
    for (int c0 = 0; c0 < chunks; c0 += nthreads) {
        const int c = c0 + tid;
        if (chunks % nthreads == 0 || c < chunks) {
            ds4_pipe_cp_async_16(stage + 4*c, by0 + 4*c);
            ds4_pipe_cp_async_16(stage + half_ints + 4*c, by1 + 4*c);
        }
    }
}

template <ggml_type type, int mmq_x, bool need_check>
static __device__ __forceinline__ void ds4_mul_mat_q_process_tile_pipe(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_stop) {
    static_assert(ds4_mmq_pipe_supported(type), "no fetch/expand split for this type");
    static_assert(mmq_x <= DS4_MMQ_PIPE_MAX_X, "two-stage activation buffer would exceed shared memory");
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int nthreads  = nwarps * warp_size;
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int mmq_y     = get_mmq_y_device();
    static_assert(MMQ_ITER_K / qk == 1, "one block per K iteration");

    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, need_check, type>::vec_dot_mma;
    constexpr mmq_write_back_t write_back = mmq_write_back_mma<type, mmq_x, mmq_y, need_check>;

    constexpr int y_half = mmq_x * MMQ_TILE_Y_K;
    constexpr int y_iter = 2 * y_half;

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + mmq_x;
    int * tile_x = tile_y + DS4_MMQ_PIPE_Y_STAGES * y_iter;

    float sum[mmq_x*mmq_y / nthreads] = {0.0f};

    const int tid = (int)threadIdx.y*warp_size + (int)threadIdx.x;

    ds4_pipe_raw<type, mmq_y, nwarps, warp_size> raw;
    ds4_pipe_fetch<type, mmq_y, need_check>(raw, x, offset_x, tile_x_max_i, stride_row_x);
    ds4_pipe_y_copy<mmq_x, nthreads>(tile_y, y, ncols_y, 0, tid);
    ds4_pipe_cp_async_commit();

    int stage = 0;
    for (int kb0 = 0; kb0 < kb0_stop; ++kb0, stage ^= 1) {
        // tile_x was released by the trailing barrier of the previous
        // iteration; the other activation stage was last read there too.
        ds4_pipe_expand<type, mmq_y, need_check>(raw, tile_x, tile_x_max_i);
        const int kb1 = kb0 + 1;
        if (kb1 < kb0_stop) {
            ds4_pipe_fetch<type, mmq_y, need_check>(raw, x, offset_x + kb1, tile_x_max_i, stride_row_x);
            ds4_pipe_y_copy<mmq_x, nthreads>(tile_y + (stage ^ 1)*y_iter, y, ncols_y, kb1, tid);
        }
        ds4_pipe_cp_async_commit();
        // Groups retire in order: leaving one pending means this
        // iteration's activation stage has landed.
        ds4_pipe_cp_async_wait<1>();
        __syncthreads();

        const int * ys = tile_y + stage*y_iter;
        vec_dot(tile_x, ys,          sum, 0);
        vec_dot(tile_x, ys + y_half, sum, MMQ_TILE_NE_K);

        __syncthreads();
    }

    write_back(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
}

#endif // defined(TURING_MMA_AVAILABLE)
