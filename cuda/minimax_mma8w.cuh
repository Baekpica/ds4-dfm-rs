#ifndef DS4_MINIMAX_MMA8W_CUH
#define DS4_MINIMAX_MMA8W_CUH

#include "beam_mma8w.cuh"

/* Warp-specialized FP8 prefill experts for 13-bit (K8192) GVQ1 layers with
 * the optional GVR8 residual stage (MiniMax's R13+8 layers): beam_w's
 * tokens x weights^T scheme. A book vector is the m16n8k32 B fragment of its
 * weight row, so one LDS.64 per (row, k32) feeds the MMAs; token rows are the
 * A operand through ldmatrix (beam_w_quant's k order); producer warps fill a
 * mbarrier ring with cp.async while the consumers compute.
 *
 *   warps 0-3   consumers  rows 32w..32w+31 of a 128-row tile (four n8
 *                          tiles), all four m16 tiles of a 64-pair item: each
 *                          A fragment feeds four MMAs (eight consumers of 16
 *                          rows re-read the pair rows twice as often, and
 *                          their ldmatrix and book traffic filled the MIO
 *                          queue)
 *   warps 4-7   producers  per 128-k stage: 128 rows' 12-bit indices (24 B),
 *                          their plane word (the 13th bits of two stages),
 *                          the residual indices (16 B, R13+8), 64 input rows
 *                          and their stage scales; the next unit's index
 *                          blocks stream into L2 meanwhile
 *
 * The 64 KiB book (+2 KiB residual book) leaves room for two stages of 64
 * rows. A residual codeword is a second MMA into the same accumulator
 * (W = gain * (C1 + C2)). Every projection keeps glmvq_mma8_kernel's
 * activation contract: E4M3 rows with one scale per (row, 128 k) and an F32
 * promotion per stage, so a token's large channels do not flatten the rest
 * of its row (beam_w's gate/up use one scale per row). Gate, up and down
 * write BF16 rows of round(gain * dot) per pair; mw_swiglu_quant turns gate
 * and up into down's E4M3 rows and scales. Results equal glmvq_mma8_kernel's
 * bit for bit (tests/test_minimax_mma8w): the k32 reduction is symmetric in
 * A and B. In the up kernel's own epilogue, SwiGLU, F32 divisions and E4M3
 * stores held the MMA warps (up ran 12-25% behind gate, and its R13+8 form
 * spilled at 255 registers); the separate pass reads 2 x 2 bytes per value.
 *
 * Measured limits (tests/test_minimax_mma8w, 2,048 rows, random routes):
 * gate 4.8 ms with weights streamed, 3.6 ms with every expert in L2, 2.3 ms
 * with the producers' copies removed (134 TFLOPS, 53% of the E4M3 peak); the
 * copies' L1 wavefronts and the consumers' ldmatrix plus book gathers share
 * the shared-memory pipe. TMA bulk copies of the 16-128-byte pieces were 3x
 * slower. At the 4,096-row production chunk (R13 / R13+8): gate 7.1 / 9.9
 * ms, up with mw_swiglu_quant 8.1 / 10.6, down 7.3 / 11.2 against MMA
 * floors of 2.6 / 5.2 ms per projection; with every expert in L2, gate
 * 6.6 / 9.1 (two slots). ncu (two slots): tensor pipe 38-44%, shared-memory
 * pipe 48% (42% of its load wavefronts are book-gather bank conflicts); the
 * consumers waited on `full` (mw_stages). Rejected, slower or even: 128-pair
 * items over 64-k stages (half the gathers per MMA; the two-stage ring then
 * covers half the time, +16-19%), separate index and input barriers per
 * slot (index filled a stage ahead, +16% on R13), accumulators rescaled
 * per stage in place of the per-stage sum (even), a k-outer loop with one
 * activation scale per row and no per-stage promotion (180 registers,
 * even), and the next stage's codewords loaded between a stage's MMA
 * halves (consumers hold two of three slots and starve the producers: +11%
 * on R13; R13+8 spills). */
enum { MW_BM = 128, MW_BT = 64, MW_MT = MW_BT / 16, MW_CONSUMERS = 4, MW_PRODUCERS = 4,
       MW_NT = MW_BM / (8 * MW_CONSUMERS),
       MW_THREADS = (MW_CONSUMERS + MW_PRODUCERS) * 32, MW_PF_FIRST = 4,
       MW_BOOK = 8192 * GLMVQ_VECTOR, MW_RBOOK = GLMVQ_RES_WORDS * GLMVQ_VECTOR,
       MW_PL = MW_BM * GLMVQ_MMA8_PLANE, MW_RS = MW_BM * GLMVQ_MMA8_RES };

/* Ring slots. A copy (L2 to shared) takes longer than the consumers' stage
 * (~0.9 us at 2,048 rows): with two slots they waited on `full` ~40% of
 * their time and the producers on `empty`. R13 fits three slots beside the
 * 64 KiB book with a stage's 24 index bytes of a row packed (three 8-byte
 * copies) and the plane words (two stages' 13th bits each) in a two-entry
 * ring of their own. R13+8 (2 KiB residual book, 2 KiB residual indices
 * per slot) keeps two slots, 16-byte aligned 32-byte index windows (two
 * 16-byte copies; packing cost it 1-5%) and the plane words in the slot. */
template <bool RES>
__host__ __device__ constexpr uint32_t mw_stages() { return RES ? 2u : 3u; }

template <bool RES>
__host__ __device__ constexpr uint32_t mw_ixw() { return RES ? 32u : GLMVQ_MMA8_INDEX; }

template <bool RES>
__host__ __device__ constexpr uint32_t mw_ix() { return MW_BM * mw_ixw<RES>(); }

// One expert matrix for a run of at most MW_BT grouped pairs; the first 16
// bytes travel to the consumers in the ring.
struct __align__(16) mw_item {
    uint32_t offset, tokens;
    const __half *gain;
    const uint8_t *main, *plane, *res;
    uint64_t pad;
};

template <bool RES>
__host__ __device__ constexpr uint32_t mw_index_bytes() { return mw_ix<RES>() + (RES ? MW_PL + MW_RS : 0u); }

template <bool RES>
__host__ __device__ constexpr uint32_t mw_stage_bytes() {
    return mw_index_bytes<RES>() + MW_BT * GLMVQ_MMA8_ROW + MW_BT * sizeof(float) + 16;
}

template <bool RES>
__host__ __device__ constexpr uint32_t mw_bytes() {
    return MW_BOOK + (RES ? MW_RBOOK : 0u) + mw_stages<RES>() * mw_stage_bytes<RES>() + (RES ? 0u : 2 * MW_PL) +
        2 * mw_stages<RES>() * sizeof(uint64_t);
}
static_assert(mw_bytes<true>() <= 101376 && mw_bytes<false>() <= 101376,
    "the rings fit GB10's 99 KiB of shared memory per block");

// Activations: [row][K] E4M3 in beam_w_quant's ldmatrix k order with one
// scale per (row, 128 k) (glmvq_mma8_quant's values, permuted). One warp per
// 128 block: lane l's four values are 4-byte group l % 8 of 32-byte block l / 8.
__global__ void mw_quant(uint8_t *out, float *scales, const float *input, uint64_t blocks) {
    const uint64_t block = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint32_t lane = threadIdx.x & 31;
    if (block >= blocks) { return; }
    const float4 x = *(const float4 *)(input + block * GLMVQ_MMA8_K + lane * 4);
    float amax = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
    for (uint32_t stride = 16; stride; stride >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, stride)); }
    const float scale = amax / GLMVQ_E4M3_MAX;
    const float a = amax > 0.0f ? __fdiv_rn(x.x, scale) : 0.0f, b = amax > 0.0f ? __fdiv_rn(x.y, scale) : 0.0f;
    const float c = amax > 0.0f ? __fdiv_rn(x.z, scale) : 0.0f, d = amax > 0.0f ? __fdiv_rn(x.w, scale) : 0.0f;
    const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
    const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
    *(uint32_t *)(out + block * GLMVQ_MMA8_K + beam_w_perm(lane * 4)) = lo | (hi << 16);
    if (!lane) { scales[block] = scale; }
}

/* MODE: GATE (gate and up) reads token rows (token_of[pair]) of `act` with
 * their stage scales `act_scale[token][kt]` and writes BF16 rows into
 * `gate`; DOWN reads pair rows of mw_swiglu_quant's `act` (mid8) with
 * `act_scale[pair][kt]` and writes BF16 rows into `out`. */
template <glmvq_mma_stage MODE, bool RES>
__global__ void __launch_bounds__(MW_THREADS, 1) mw_kernel(const mw_item *items, uint32_t nitems,
        const uint8_t *book, const uint8_t *res_book, const uint8_t *act, const float *act_scale,
        const uint32_t *token_of, __nv_bfloat16 *gate, __nv_bfloat16 *out, uint32_t rows, uint32_t cols) {
    static_assert(MODE == GLMVQ_MMA_GATE || MODE == GLMVQ_MMA_DOWN, "up runs as GATE, then mw_swiglu_quant");
    constexpr uint32_t BM = MW_BM, BT = MW_BT, S = mw_stages<RES>(), C = MW_CONSUMERS, MT = MW_MT;
    constexpr uint32_t PT = MW_PRODUCERS * 32, IXB = mw_index_bytes<RES>(), ATB = BT * GLMVQ_MMA8_ROW;
    constexpr uint32_t STB = mw_stage_bytes<RES>();
    static_assert(PT == BM && PT == 2 * BT, "producer lanes: one per weight row, eight per input row");
    constexpr uint32_t BOOKS = MW_BOOK + (RES ? MW_RBOOK : 0u);
    extern __shared__ __align__(16) uint8_t memory[];
    uint8_t *ring = memory + BOOKS, *planes = ring + S * STB;
    uint64_t *full = (uint64_t *)(planes + (RES ? 0u : 2 * MW_PL)), *empty = full + S;
    constexpr uint32_t IXW = mw_ixw<RES>(), IX = mw_ix<RES>();
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    for (uint32_t i = tid * 16; i < MW_BOOK; i += MW_THREADS * 16) { glmvq_mma_copy<16>(memory + i, book + i); }
    if constexpr (RES) {
        for (uint32_t i = tid * 16; i < MW_RBOOK; i += MW_THREADS * 16) {
            glmvq_mma_copy<16>(memory + MW_BOOK + i, res_book + i);
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    if (tid == 0) {
        for (uint32_t s = 0; s < S; s++) {
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(beam_w_smem(full + s)), "r"(PT) : "memory");
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(beam_w_smem(empty + s)), "r"(C) : "memory");
        }
    }
    __syncthreads();
    const uint32_t tiles = rows / BM, stages = cols / GLMVQ_MMA8_K, units = nitems * tiles;
    const uint32_t main_stride = cols * 3 / 16, plane_stride = cols / 64, res_stride = cols / 8;
    if (warp >= C) {
        // Producer copies, in units the L1 pipe serves cheaply: consecutive
        // lanes take consecutive bytes of one row, so a warp's cp.async
        // touches 16 rows of index windows (2 x 16 B each), 32 rows of plane
        // words (even stages only: a word holds two stages' bits, the
        // consumers keep the odd half) or residuals, and 4 input rows (8 x
        // 16 B). One lane per row put 32 rows' lines behind every 8-byte
        // copy (a third of the shared pipe's wavefronts). Each unit's
        // sources are resolved once; stages only advance them.
        const uint32_t pt = tid - C * 32, chunk = pt % 8, tq = pt / 8;
        constexpr uint32_t TQ = PT / 8, TJ = BT / TQ;
        // The next unit's index blocks stream into L2 in sequential pieces.
        auto prefetch = [&](const uint8_t *p, uint32_t block, uint32_t piece, uint32_t pieces) {
            const uint32_t step = (block / pieces + 15) / 16 * 16, off = piece * step;
            if (pt == 0 && p && off < block) { glmvq_mma_prefetch(p + off, min(step, block - off)); }
        };
        auto source_row = [&](const mw_item &it, uint32_t t) {
            const uint32_t pair = it.offset + (t < it.tokens ? t : 0);
            return MODE == GLMVQ_MMA_DOWN ? pair : token_of[pair];
        };
        uint32_t ci = min(blockIdx.x, units - 1) / tiles;
        mw_item cur = items[ci];
        uint32_t crow = (min(blockIdx.x, units - 1) % tiles) * BM;
        uint32_t src[TJ];
        #pragma unroll
        for (uint32_t j = 0; j < TJ; j++) { src[j] = source_row(cur, tq + TQ * j); }
        for (uint32_t piece = 0; piece < MW_PF_FIRST; piece++) {
            prefetch(blockIdx.x < units ? cur.main + (uint64_t)crow * main_stride : nullptr, BM * main_stride, piece,
                MW_PF_FIRST);
            if (RES) {
                prefetch(blockIdx.x < units ? cur.res + (uint64_t)crow * res_stride : nullptr, BM * res_stride, piece,
                    MW_PF_FIRST);
            }
        }
        // R13: lane pt copies row pt's index run and plane word. R13+8: index
        // window halves of rows pt / 2 and 64 + pt / 2 (half pt % 2), the plane
        // word of row pt in the slot.
        const uint32_t ir0 = RES ? pt / 2 : pt, ir1 = BM / 2 + pt / 2, ih = RES ? (pt % 2) * 16 : 0u;
        const uint32_t o_main0 = ir0 * IXW + ih, o_main1 = ir1 * IXW + ih;
        const uint32_t o_plane = (RES ? IX : 0u) + pt * GLMVQ_MMA8_PLANE;
        [[maybe_unused]] const uint32_t o_res = IX + MW_PL + pt * GLMVQ_MMA8_RES;
        uint32_t o_tok[TJ], o_scale[TJ];
        #pragma unroll
        for (uint32_t j = 0; j < TJ; j++) {
            o_tok[j] = IXB + beam_w_swz(tq + TQ * j, chunk);
            o_scale[j] = IXB + ATB + (tq + TQ * j) * 4;
        }
        const uint32_t ring_s = beam_w_smem(ring), planes_s = beam_w_smem(planes);
        uint32_t ringc = 0;
        for (uint32_t work = blockIdx.x; work < units; work += gridDim.x) {
            const uint32_t next = work + gridDim.x, nt = cur.tokens, padded = (nt + 15) / 16 * 16;
            const uint32_t ni = min(next, units - 1) / tiles, nrow = (min(next, units - 1) % tiles) * BM;
            const mw_item nxt = items[ni];
            uint32_t nsrc[TJ];
            const uint8_t *g_main0 = cur.main + (uint64_t)(crow + ir0) * main_stride + ih;
            const uint8_t *g_main1 = cur.main + (uint64_t)(crow + ir1) * main_stride + ih;
            const uint8_t *g_plane = cur.plane + (uint64_t)(crow + pt) * plane_stride;
            [[maybe_unused]] const uint8_t *g_res = RES ? cur.res + (uint64_t)(crow + pt) * res_stride : nullptr;
            const uint8_t *g_tok[TJ];
            const float *g_scale[TJ];
            uint32_t bytes[TJ];
            bool live[TJ];
            #pragma unroll
            for (uint32_t j = 0; j < TJ; j++) {
                const uint32_t t = tq + TQ * j;
                g_tok[j] = act + (uint64_t)src[j] * cols + chunk * 16;
                g_scale[j] = act_scale + (uint64_t)src[j] * stages;
                bytes[j] = t < nt ? 16u : 0u;   // zero rows pad the last m16 tile, scales too
                live[j] = t < padded;
            }
            for (uint32_t kt = 0; kt < stages; kt++, ringc++) {
                const uint32_t s = ringc % S, use = ringc / S;
                if (use) { beam_w_wait(empty + s, (use - 1) & 1); }
                if (kt == 1) {
                    // nxt arrived during stage 0: its input rows resolve now.
                    #pragma unroll
                    for (uint32_t j = 0; j < TJ; j++) { nsrc[j] = source_row(nxt, tq + TQ * j); }
                }
                if (kt >= 1 && next < units) {
                    prefetch(nxt.main + (uint64_t)nrow * main_stride, BM * main_stride, kt - 1, stages - 1);
                    if (RES) { prefetch(nxt.res + (uint64_t)nrow * res_stride, BM * res_stride, kt - 1, stages - 1); }
                }
                const uint32_t slot = ring_s + s * STB;
                if constexpr (RES) {
                    const uint32_t win = (kt * GLMVQ_MMA8_INDEX) & ~15u;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(slot + o_main0),
                        "l"(g_main0 + win) : "memory");
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(slot + o_main1),
                        "l"(g_main1 + win) : "memory");
                } else {
                    const uint8_t *g_ix = g_main0 + kt * GLMVQ_MMA8_INDEX;   // 8-byte aligned
                    // Rows lie 1,152 B (9 x 128 B) apart, so one piece of every
                    // lane's row maps to one L1 bank pair (32 wavefronts per
                    // copy); rotating the pieces spreads a copy over three.
                    #pragma unroll
                    for (uint32_t k = 0; k < 3; k++) {
                        const uint32_t p = (k + pt) % 3;
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(slot + o_main0 + 8 * p),
                            "l"(g_ix + 8 * p) : "memory");
                    }
                }
                if (!(kt & 1u)) {
                    // R13: entry (kt / 2) % 2 is next rewritten four stages on,
                    // after the consumers released a later slot.
                    const uint32_t dst = RES ? slot + o_plane : planes_s + ((kt / 2) & 1u) * MW_PL + o_plane;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst), "l"(g_plane + 4 * (kt / 2))
                        : "memory");
                }
                if constexpr (RES) {
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(slot + o_res),
                        "l"(g_res + kt * GLMVQ_MMA8_RES) : "memory");
                }
                #pragma unroll
                for (uint32_t j = 0; j < TJ; j++) {
                    if (!live[j]) { continue; }
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(slot + o_tok[j]),
                        "l"(g_tok[j] + kt * GLMVQ_MMA8_K), "r"(bytes[j]) : "memory");
                    if (chunk == 0) {
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(slot + o_scale[j]),
                            "l"(g_scale[j] + kt), "r"(bytes[j] / 4) : "memory");
                    }
                }
                // The item record (offset, tokens, gain) rides in stage 0's slot.
                if (kt == 0 && pt == 0) { glmvq_mma_copy<16>(ring + s * STB + STB - 16, items + ci); }
                asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(beam_w_smem(full + s)) : "memory");
            }
            cur = nxt; ci = ni; crow = nrow;
            #pragma unroll
            for (uint32_t j = 0; j < TJ; j++) { src[j] = nsrc[j]; }
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
        return;
    }
    // MMA fragment coordinates: D rows (pairs) g and g + 8, columns (weight
    // rows) 2q, 2q + 1 of each n8 tile; the B fragment is row g's codeword.
    constexpr uint32_t NT = MW_NT, WR = 8 * NT;
    const uint32_t g = lane >> 2, q = lane & 3u;
    // ldmatrix.x4 row address of this lane: matrix lane / 8 = (row half, k half).
    const uint32_t lt = (lane & 7u) + ((lane >> 3) & 1u) * 8, lk = lane >> 4;
    uint32_t ringc = 0;
    for (uint32_t work = blockIdx.x; work < units; work += gridDim.x) {
        const uint32_t row = (work % tiles) * BM;
        // The item record arrives with stage 0.
        beam_w_wait(full + ringc % S, (ringc / S) & 1);
        const uint4 rec = *(const uint4 *)(ring + (ringc % S) * STB + STB - 16);
        const uint32_t offset = rec.x, nt = rec.y, nmt = (nt + 15) / 16;
        const __half *gain = (const __half *)((uint64_t)rec.z | (uint64_t)rec.w << 32);
        // Gains (this lane's columns) load now and are first used after the K loop.
        const uint32_t c0 = row + warp * WR + 2 * q;
        __half2 hg[NT];
        #pragma unroll
        for (uint32_t n = 0; n < NT; n++) { hg[n] = *(const __half2 *)(gain + c0 + 8 * n); }
        // acc[m][n]: pair tile m, weight rows WR w + 8n + (0..7); e_odd[n]:
        // the plane bits of odd stages (only even slots carry the word).
        float acc[MT][NT][4] = {};
        uint32_t e_odd[NT] = {};
        auto run = [&](auto count) {
            constexpr uint32_t NM = decltype(count)::value;
            for (uint32_t kt = 0; kt < stages; kt++, ringc++) {
                const uint32_t s = ringc % S;
                beam_w_wait(full + s, (ringc / S) & 1);
                const uint8_t *stage = ring + s * STB;
                // This lane's codewords 4K + q of rows WR w + 8n + g (and their residuals);
                // R13+8's index bytes start 0 or 8 bytes into the row's window.
                const uint32_t ix = RES ? (kt * GLMVQ_MMA8_INDEX) & 15u : 0u;
                uint2 b[4][NT];
                [[maybe_unused]] uint2 rb[4][NT];
                #pragma unroll
                for (uint32_t n = 0; n < NT; n++) {
                    const uint32_t r = warp * WR + 8 * n + g;
                    uint32_t w[6];
                    #pragma unroll
                    for (uint32_t k = 0; k < 3; k++) {
                        const uint2 v = *(const uint2 *)(stage + r * IXW + ix + k * 8);
                        w[2 * k] = v.x; w[2 * k + 1] = v.y;
                    }
                    uint32_t e = e_odd[n];
                    if (!(kt & 1u)) {
                        e = *(const uint32_t *)((RES ? stage + IX : planes + ((kt / 2) & 1u) * MW_PL) +
                            r * GLMVQ_MMA8_PLANE);
                        e_odd[n] = e >> 16;
                    }
                    b[0][n] = *(const uint2 *)(memory + (glmvq_mma8_code<0>(w, q) | ((e >> q) & 1u) << 12) * 8);
                    b[1][n] = *(const uint2 *)(memory + (glmvq_mma8_code<1>(w, q) | ((e >> (4 + q)) & 1u) << 12) * 8);
                    b[2][n] = *(const uint2 *)(memory + (glmvq_mma8_code<2>(w, q) | ((e >> (8 + q)) & 1u) << 12) * 8);
                    b[3][n] = *(const uint2 *)(memory + (glmvq_mma8_code<3>(w, q) | ((e >> (12 + q)) & 1u) << 12) * 8);
                    if constexpr (RES) {
                        // Word K of the row's 16 residual bytes holds codes 4K..4K+3.
                        const uint4 v = *(const uint4 *)(stage + IX + MW_PL + r * GLMVQ_MMA8_RES);
                        const uint32_t words[4] = {v.x, v.y, v.z, v.w};
                        #pragma unroll
                        for (uint32_t k = 0; k < 4; k++) {
                            rb[k][n] = *(const uint2 *)(memory + MW_BOOK + ((words[k] >> (8 * q)) & 255u) * 8);
                        }
                    }
                }
                const uint8_t *tokens = stage + IXB;
                const float *scale = (const float *)(stage + IXB + ATB);
                #pragma unroll
                for (uint32_t m = 0; m < NM; m++) {
                    float part[NT][4] = {};
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; k++) {
                        uint32_t a[4];
                        beam_w_ldsm4(a, tokens + beam_w_swz(m * 16 + lt, 2 * k + lk));
                        #pragma unroll
                        for (uint32_t n = 0; n < NT; n++) {
                            beam_w_mma(part[n], a, b[k][n]);
                            if constexpr (RES) { beam_w_mma(part[n], a, rb[k][n]); }
                        }
                    }
                    const float s0 = scale[m * 16 + g], s1 = scale[m * 16 + 8 + g];
                    #pragma unroll
                    for (uint32_t n = 0; n < NT; n++) {
                        acc[m][n][0] = fmaf(part[n][0], s0, acc[m][n][0]);
                        acc[m][n][1] = fmaf(part[n][1], s0, acc[m][n][1]);
                        acc[m][n][2] = fmaf(part[n][2], s1, acc[m][n][2]);
                        acc[m][n][3] = fmaf(part[n][3], s1, acc[m][n][3]);
                    }
                }
                // The stage's operands are in registers: free the slot.
                __syncwarp();
                if (lane == 0) { beam_w_arrive(empty + s); }
            }
        };
        // Exact m16 tile counts (padding MMAs are not free).
        if (nmt <= 1) { run(std::integral_constant<uint32_t, 1>{}); }
        else if (nmt == 2) { run(std::integral_constant<uint32_t, 2>{}); }
        else if (nmt == 3) { run(std::integral_constant<uint32_t, 3>{}); }
        else { run(std::integral_constant<uint32_t, MT>{}); }
        float gn[NT][2];
        #pragma unroll
        for (uint32_t n = 0; n < NT; n++) {
            const float2 f = __half22float2(hg[n]);
            gn[n][0] = f.x; gn[n][1] = f.y;
        }
        __nv_bfloat16 *dst = MODE == GLMVQ_MMA_GATE ? gate : out;
        #pragma unroll
        for (uint32_t m = 0; m < MT; m++) {
            if (m >= nmt) { break; }
            #pragma unroll
            for (uint32_t h = 0; h < 2; h++) {
                const uint32_t t = m * 16 + h * 8 + g;
                if (t >= nt) { continue; }
                const uint64_t base = (uint64_t)(offset + t) * rows + c0;
                #pragma unroll
                for (uint32_t n = 0; n < NT; n++) {
                    *(__nv_bfloat162 *)(dst + base + 8 * n) = __floats2bfloat162_rn(
                        glmvq_round(glmvq_mul(acc[m][n][2 * h], gn[n][0])),
                        glmvq_round(glmvq_mul(acc[m][n][2 * h + 1], gn[n][1])));
                }
            }
        }
    }
}

template <glmvq_mma_stage MODE, bool RES>
static cudaError_t mw_launch(const mw_item *items, uint32_t nitems, const uint8_t *book, const uint8_t *res_book,
        const uint8_t *act, const float *act_scale, const uint32_t *token_of, __nv_bfloat16 *gate, __nv_bfloat16 *out,
        uint32_t rows, uint32_t cols, uint32_t sms, cudaStream_t stream) {
    static cudaError_t ready = cudaFuncSetAttribute(mw_kernel<MODE, RES>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        mw_bytes<RES>());
    if (ready != cudaSuccess) { return ready; }
    const uint32_t units = nitems * (rows / MW_BM);
    if (!units) { return cudaSuccess; }
    mw_kernel<MODE, RES><<<min(units, sms), MW_THREADS, mw_bytes<RES>(), stream>>>(items, nitems, book, res_book, act,
        act_scale, token_of, gate, out, rows, cols);
    return cudaGetLastError();
}

/* SwiGLU-OAI of the BF16 gate and up rows, then down's input: E4M3 rows in
 * its k order with one scale per (pair, 128 rows), a warp per block (four
 * values per lane; the block's peak by shuffles). */
__global__ void mw_swiglu_quant(uint8_t *mid8, float *mid_scale, const __nv_bfloat16 *gate,
        const __nv_bfloat16 *up, uint32_t rows, uint32_t pairs, float limit) {
    const uint64_t w = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint32_t lane = threadIdx.x & 31, blocks = rows / GLMVQ_MMA8_K;
    if (w >= (uint64_t)pairs * blocks) { return; }
    const uint64_t p = w / blocks;
    const uint32_t blk = (uint32_t)(w % blocks), c = blk * GLMVQ_MMA8_K + lane * 4;
    const uint2 gr = *(const uint2 *)(gate + p * rows + c), ur = *(const uint2 *)(up + p * rows + c);
    const float2 g0 = __bfloat1622float2(*(const __nv_bfloat162 *)&gr.x);
    const float2 g1 = __bfloat1622float2(*(const __nv_bfloat162 *)&gr.y);
    const float2 u0 = __bfloat1622float2(*(const __nv_bfloat162 *)&ur.x);
    const float2 u1 = __bfloat1622float2(*(const __nv_bfloat162 *)&ur.y);
    const float v[4] = {glmvq_swiglu_value<GLMVQ_CLAMP_OAI>(g0.x, u0.x, limit),
        glmvq_swiglu_value<GLMVQ_CLAMP_OAI>(g0.y, u0.y, limit), glmvq_swiglu_value<GLMVQ_CLAMP_OAI>(g1.x, u1.x, limit),
        glmvq_swiglu_value<GLMVQ_CLAMP_OAI>(g1.y, u1.y, limit)};
    float peak = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    for (uint32_t s = 16; s; s >>= 1) { peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, s)); }
    const float scale = peak / GLMVQ_E4M3_MAX;
    #pragma unroll
    for (uint32_t i = 0; i < 4; i += 2) {
        const float a = peak > 0.0f ? __fdiv_rn(v[i], scale) : 0.0f;
        const float b = peak > 0.0f ? __fdiv_rn(v[i + 1], scale) : 0.0f;
        // Down reads these rows with ldmatrix: store them in its k order.
        *(uint16_t *)(mid8 + p * rows + beam_w_perm(c + i)) =
            (uint16_t)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
    }
    if (lane == 0) { mid_scale[p * blocks + blk] = scale; }
}

#endif
