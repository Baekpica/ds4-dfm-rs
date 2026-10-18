#ifndef DS4_GLM_VQ_MMA8_CUH
#define DS4_GLM_VQ_MMA8_CUH

#include "glm_vq_mma.cuh"

/* FP8 tensor-core prefill MoE. The source FP8 recipe quantizes activations
 * dynamically per token and 128 channels; GVQ1 codewords are already E4M3,
 * so they feed mma.m16n8k32.e4m3 unconverted and the F16 row gain scales the
 * FP32 dot. The BF16 register-MMA path rebuilt BF16 weights per fragment
 * (23 issued instructions per MMA, 28% tensor-pipe use, NCU). Arithmetic
 * change: activations carry E4M3 rounding (1x128 amax scale) and weights skip
 * the BF16 gain*codeword rounding; outputs keep their BF16 boundaries. */
enum { GLMVQ_MMA8_K = 128, GLMVQ_MMA8_ROW = 128, GLMVQ_MMA8_STAGES = 3,
       GLMVQ_MMA8_INDEX = 24, GLMVQ_MMA8_PLANE = 4, GLMVQ_MMA8_RES = 16 };
static constexpr float GLMVQ_E4M3_MAX = 448.0f;

// Activations: [row][K] E4M3 in source order plus [row][K/128] F32 scales
// (x = e4m3 * scale). One warp per 128 block.
template <typename T>
__global__ void glmvq_mma8_quant(uint8_t *out, float *scales, const T *input, uint64_t blocks) {
    const uint64_t block = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint32_t lane = threadIdx.x & 31;
    if (block >= blocks) { return; }
    const T *x = input + block * GLMVQ_MMA8_K + lane * 4;
    float v[4], amax = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) { v[i] = (float)x[i]; amax = fmaxf(amax, fabsf(v[i])); }
    for (uint32_t stride = 16; stride; stride >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, stride)); }
    const float scale = amax / GLMVQ_E4M3_MAX;
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) { v[i] = amax > 0.0f ? __fdiv_rn(v[i], scale) : 0.0f; }
    const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v[0], v[1]), __NV_SATFINITE, __NV_E4M3);
    const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v[2], v[3]), __NV_SATFINITE, __NV_E4M3);
    *(uint32_t *)(out + block * GLMVQ_MMA8_K + lane * 4) = lo | (hi << 16);
    if (!lane) { scales[block] = scale; }
}

// 64-token items fit beside the 64 KiB book only with double buffering;
// a 12-bit (K4096) layer's 32 KiB book leaves room for three stages. R13+8
// layers add a 2 KiB book. 32-token tiles stage 16 residual bytes per row
// (98,560 B); 64-token tiles cannot (107,008 of GB10's 101,376 B), so each
// thread loads its rows' residual bytes into registers one stage ahead
// (98,816 B): half the codebook gathers and B loads per token.
template <uint32_t BT, uint32_t BITS = 13, bool RES = false>
__host__ __device__ constexpr uint32_t glmvq_mma8_stages() {
    return (BT == 64 && BITS == 13) || RES ? 2 : GLMVQ_MMA8_STAGES;
}

template <uint32_t BT, bool RES>
__host__ __device__ constexpr bool glmvq_mma8_res_regs() { return RES && BT == 64; }

template <uint32_t BITS, bool RES = false>
__host__ __device__ constexpr uint32_t glmvq_mma8_book() {
    return (1u << BITS) * GLMVQ_VECTOR + (RES ? GLMVQ_RES_WORDS * GLMVQ_VECTOR : 0u);
}

template <uint32_t BM, uint32_t BT, uint32_t BITS = 13, bool RES = false>
static constexpr uint32_t glmvq_mma8_bytes() {
    return glmvq_mma8_book<BITS, RES>() + glmvq_mma8_stages<BT, BITS, RES>() *
        (BM * (GLMVQ_MMA8_INDEX + (BITS == 13 ? GLMVQ_MMA8_PLANE : 0u) +
               (RES && !glmvq_mma8_res_regs<BT, RES>() ? GLMVQ_MMA8_RES : 0u)) +
         BT * (GLMVQ_MMA8_ROW + sizeof(float)));
}

// Staged token rows keep 128 bytes; their 32-byte k32 groups rotate by
// (token & 3), so the 8 tokens x 4 lanes of a B fragment load hit distinct banks.
__device__ __forceinline__ uint32_t glmvq_mma8_swz(uint32_t token, uint32_t group) {
    return token * GLMVQ_MMA8_ROW + ((group ^ (token & 3u)) << 5);
}

__device__ __forceinline__ void glmvq_mma8_acc(float *d, const uint32_t *a, uint2 b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y));
}

// Index j of a stage occupies bits 12j..12j+11 of its 24 staged bytes; lane
// q of k32 step K needs j = 4K + q, inside the 64-bit window at word 48K/32.
template <uint32_t K>
__device__ __forceinline__ uint32_t glmvq_mma8_code(const uint32_t (&w)[6], uint32_t q) {
    constexpr uint32_t P = 48 * K;
    const uint64_t window = ((uint64_t)w[P / 32 + 1] << 32) | w[P / 32];
    return (uint32_t)(window >> (P % 32 + 12 * q)) & 4095u;
}

// Each persistent CTA keeps the layer book; a warp owns MT m16 row tiles of
// a BM tile and all n8 token tiles of one expert item, so every B fragment it
// loads feeds MT MMAs. `map` turns a grouped pair into its token (pair / used)
// for gate/up; down reads the per-pair intermediate rows. Gate and down
// outputs are BF16-rounded, so they are stored as BF16 (half the traffic of
// the up epilogue's and the reduce's reads). With `mid8`, the up epilogue
// also quantizes the SwiGLU rows for down as glmvq_mma8_quant would (each
// 128-row block of a tile is one activation scale block), so the BF16
// intermediate is never written or reread.
// The k32 reduction order is free as long as A and B agree: lane q takes
// whole codeword 4K+q (k 4q.. <- bytes 0-3, k 16+4q.. <- bytes 4-7), so one
// 8-byte book read fills two A registers and B is 8 source-order bytes.
template <uint32_t BM, uint32_t MT>
__host__ __device__ constexpr uint32_t glmvq_mma8_threads() { return BM / (16 * MT) * 32; }

template <uint32_t BM, uint32_t BT, uint32_t MT, glmvq_mma_stage MODE, uint32_t BITS = 13,
          glmvq_clamp CLAMP = GLMVQ_CLAMP_GATE, bool RES = false>
__global__ __launch_bounds__(glmvq_mma8_threads<BM, MT>(), 1) void glmvq_mma8_kernel(
        const glmvq_source *sources, const glmvq_mma_item *items, uint32_t nitems,
        const uint8_t *book, const uint8_t *act, const float *act_scale, const uint32_t *map,
        uint32_t used, __nv_bfloat16 *gate, __nv_bfloat16 *mid, __nv_bfloat16 *output,
        uint32_t rows, uint32_t cols, float limit, uint8_t *mid8 = nullptr, float *mid_scale = nullptr,
        const uint8_t *book2 = nullptr) {
    static_assert((BM == 128 || BM == 256) && (BT == 16 || BT == 32 || BT == 64) && (MT == 1 || MT == 2),
        "bounded GLM tiles");
    constexpr uint32_t THREADS = glmvq_mma8_threads<BM, MT>(), NT = BT / 8, PREFETCH_AHEAD = 2;
    constexpr uint32_t STAGES = glmvq_mma8_stages<BT, BITS, RES>(), BOOK = glmvq_mma8_book<BITS, RES>();
    constexpr uint32_t BOOK1 = glmvq_mma8_book<BITS>();
    constexpr uint32_t BSB = BM * GLMVQ_MMA8_INDEX, EXB = BITS == 13 ? BM * GLMVQ_MMA8_PLANE : 0u;
    // Staged residual bytes follow the plane; activations follow both.
    constexpr bool RREG = glmvq_mma8_res_regs<BT, RES>();
    constexpr uint32_t RSB = RES && !RREG ? BM * GLMVQ_MMA8_RES : 0u, IXB = BSB + EXB + RSB;
    constexpr uint32_t ATB = BT * GLMVQ_MMA8_ROW, STB = IXB + ATB + BT * sizeof(float);
    static_assert(BITS == 12 || BITS == 13, "GVQ1 stores 12- or 13-bit indices");
    static_assert(THREADS >= BT * 8 && THREADS >= BM, "one activation chunk and row per thread");
    extern __shared__ __align__(16) uint8_t memory[];
    uint8_t *ring = memory + BOOK;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, q = lane & 3u;
    for (uint32_t i = tid * 16; i < BOOK1; i += THREADS * 16) {
        glmvq_mma_copy<16>(memory + i, book + i);
    }
    if constexpr (RES) {
        for (uint32_t i = tid * 16; i < BOOK - BOOK1; i += THREADS * 16) {
            glmvq_mma_copy<16>(memory + BOOK1 + i, book2 + i);
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    const uint32_t tiles = rows / BM, stages = cols / GLMVQ_MMA8_K, blocks = cols / GLMVQ_MMA8_K;
    uint32_t lr[MT][2];
    #pragma unroll
    for (uint32_t m = 0; m < MT; m++) { lr[m][0] = (warp * MT + m) * 16 + g; lr[m][1] = lr[m][0] + 8; }
    for (uint32_t work = blockIdx.x; work < nitems * tiles; work += gridDim.x) {
        const glmvq_mma_item item = items[work / tiles];
        const glmvq_matrix mat = sources[item.matrix[MODE]].vq;
        const uint32_t row = (work % tiles) * BM, nt = item.tokens;
        const uint32_t main_stride = cols * 3 / 16, plane_stride = cols / 64, res_stride = cols / 8;
        float gain[MT][2];
        #pragma unroll
        for (uint32_t m = 0; m < MT; m++) {
            gain[m][0] = __half2float(mat.gain[row + lr[m][0]]);
            gain[m][1] = __half2float(mat.gain[row + lr[m][1]]);
        }
        auto source_row = [&](uint32_t t) {
            const uint32_t pair = item.offset + (t < nt ? t : 0);
            return MODE == GLMVQ_MMA_DOWN ? pair : map[pair] / used;
        };
        auto prefetch = [&](uint32_t chunk) {
            const uint32_t offset = chunk * GLMVQ_PREFETCH_BYTES;
            if (tid < BM && offset < main_stride) {
                glmvq_mma_prefetch(mat.main + (uint64_t)(row + tid) * main_stride + offset,
                    min((uint32_t)GLMVQ_PREFETCH_BYTES, main_stride - offset));
            }
        };
        for (uint32_t chunk = 0; chunk < PREFETCH_AHEAD; chunk++) { prefetch(chunk); }
        if (BITS == 13 && tid == 0) { glmvq_mma_prefetch(mat.plane + (uint64_t)row * plane_stride, BM * plane_stride); }

        const uint32_t padded = (nt + 7) / 8 * 8;
        // Each thread copies the same token chunk (and scale) at every stage.
        const uint32_t copy_t = tid / 8, copy_c = tid % 8;
        const uint8_t *copy_row = act + (uint64_t)source_row(copy_t) * cols + copy_c * 16;
        const float *copy_scale = act_scale + (uint64_t)source_row(tid) * blocks;
        const uint32_t copy_dst = glmvq_mma8_swz(copy_t, copy_c / 2) + (copy_c % 2) * 16;
        auto issue = [&](uint32_t stage) {
            uint8_t *dst = ring + (stage % STAGES) * STB;
            for (uint32_t i = tid; i < BM * 3; i += THREADS) {
                const uint32_t r = i / 3, part = i % 3;
                glmvq_mma_copy<8>(dst + r * GLMVQ_MMA8_INDEX + part * 8,
                    mat.main + (uint64_t)(row + r) * main_stride + stage * GLMVQ_MMA8_INDEX + part * 8);
            }
            if (BITS == 13 && tid < BM) {
                glmvq_mma_copy<4>(dst + BSB + tid * 4,
                    mat.plane + (uint64_t)(row + tid) * plane_stride + 4 * (stage / 2));
            }
            if (RES && !RREG && tid < BM) {
                glmvq_mma_copy<16>(dst + BSB + EXB + tid * GLMVQ_MMA8_RES,
                    mat.res + (uint64_t)(row + tid) * res_stride + stage * GLMVQ_MMA8_RES);
            }
            // Zero rows pad the last n8 tile; their scale is zero as well.
            if (copy_t < padded) {
                glmvq_mma_pad(dst + IXB + copy_dst, copy_row + stage * GLMVQ_MMA8_K, copy_t < nt ? 16 : 0);
            }
            if (tid < padded) {
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::
                    "r"((uint32_t)__cvta_generic_to_shared(dst + IXB + ATB + tid * 4)),
                    "l"(copy_scale + stage), "r"(tid < nt ? 4 : 0) : "memory");
            }
        };
        for (uint32_t s0 = 0; s0 + 1 < STAGES; s0++) {
            if (s0 < stages) { issue(s0); }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        const uint32_t nnt = padded / 8;
        float acc[MT][NT][4] = {};
        // RREG: a row's 16 residual bytes of one stage; word K holds codes
        // 4K..4K+3. Lane q loads word q; the quad transposes so lane q packs
        // byte q of each word (code 4K+q at bits 8K).
        uint32_t rnext[MT][2];
        auto res_load = [&](uint32_t stage) {
            #pragma unroll
            for (uint32_t m = 0; m < MT; m++) {
                #pragma unroll
                for (uint32_t side = 0; side < 2; side++) {
                    rnext[m][side] = __ldg((const uint32_t *)(mat.res + (uint64_t)(row + lr[m][side]) * res_stride +
                        stage * GLMVQ_MMA8_RES) + q);
                }
            }
        };
        if constexpr (RREG) { res_load(0); }
        auto run = [&](auto count) {
            constexpr uint32_t NNT = decltype(count)::value;
            for (uint32_t stage = 0; stage < stages; stage++) {
                uint32_t rcode[MT][2];
                if constexpr (RREG) {
                    #pragma unroll
                    for (uint32_t m = 0; m < MT; m++) {
                        #pragma unroll
                        for (uint32_t side = 0; side < 2; side++) {
                            uint32_t v[4];
                            #pragma unroll
                            for (uint32_t k = 0; k < 4; k++) {
                                v[k] = __shfl_sync(0xffffffffu, rnext[m][side], (lane & ~3u) | k);
                            }
                            const uint32_t pick = q | (4u + q) << 4;
                            rcode[m][side] = __byte_perm(__byte_perm(v[0], v[1], pick), __byte_perm(v[2], v[3], pick), 0x5410);
                        }
                    }
                    if (stage + 1 < stages) { res_load(stage + 1); }
                }
                // Oldest group lands; the slot being refilled was read last stage.
                asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2) : "memory");
                __syncthreads();
                if (stage + STAGES - 1 < stages) { issue(stage + STAGES - 1); }
                asm volatile("cp.async.commit_group;" ::: "memory");
                if (stage % (GLMVQ_PREFETCH_BYTES / GLMVQ_MMA8_INDEX) == 0) {
                    prefetch(stage / (GLMVQ_PREFETCH_BYTES / GLMVQ_MMA8_INDEX) + PREFETCH_AHEAD);
                }
                const uint8_t *src = ring + (stage % STAGES) * STB;
                const uint32_t bit = (stage & 1u) * 16;
                uint32_t w[MT][2][6], e[MT][2];
                #pragma unroll
                for (uint32_t m = 0; m < MT; m++) {
                    #pragma unroll
                    for (uint32_t side = 0; side < 2; side++) {
                        #pragma unroll
                        for (uint32_t k = 0; k < 3; k++) {
                            const uint2 v = *(const uint2 *)(src + lr[m][side] * GLMVQ_MMA8_INDEX + k * 8);
                            w[m][side][2 * k] = v.x; w[m][side][2 * k + 1] = v.y;
                        }
                        e[m][side] = BITS == 13 ? *(const uint32_t *)(src + BSB + lr[m][side] * 4) >> bit : 0u;
                    }
                }
                const uint8_t *activation = src + IXB + q * 8;
                const uint8_t *residual = src + BSB + EXB;
                float part[MT][NNT][4] = {};
                auto step = [&](auto kk) {
                    constexpr uint32_t K = decltype(kk)::value;
                    uint32_t a[MT][4], r[MT][4];
                    #pragma unroll
                    for (uint32_t m = 0; m < MT; m++) {
                        const uint32_t ia = glmvq_mma8_code<K>(w[m][0], q) | ((e[m][0] >> (4 * K + q)) & 1u) << 12;
                        const uint32_t ib = glmvq_mma8_code<K>(w[m][1], q) | ((e[m][1] >> (4 * K + q)) & 1u) << 12;
                        const uint2 ca = *(const uint2 *)(memory + ia * 8), cb = *(const uint2 *)(memory + ib * 8);
                        a[m][0] = ca.x; a[m][1] = cb.x; a[m][2] = ca.y; a[m][3] = cb.y;
                        if constexpr (RES) {
                            // Lane q's residual codeword 4K+q of both rows, the same
                            // k order as the layer codeword: a second MMA adds it.
                            const uint32_t ra = RREG ? (rcode[m][0] >> (8 * K)) & 255u
                                : residual[lr[m][0] * GLMVQ_MMA8_RES + 4 * K + q];
                            const uint32_t rb = RREG ? (rcode[m][1] >> (8 * K)) & 255u
                                : residual[lr[m][1] * GLMVQ_MMA8_RES + 4 * K + q];
                            const uint2 da = *(const uint2 *)(memory + BOOK1 + ra * 8);
                            const uint2 db = *(const uint2 *)(memory + BOOK1 + rb * 8);
                            r[m][0] = da.x; r[m][1] = db.x; r[m][2] = da.y; r[m][3] = db.y;
                        }
                    }
                    #pragma unroll
                    for (uint32_t j = 0; j < NNT; j++) {
                        const uint2 b = *(const uint2 *)(activation + glmvq_mma8_swz(j * 8 + g, K));
                        #pragma unroll
                        for (uint32_t m = 0; m < MT; m++) {
                            glmvq_mma8_acc(part[m][j], a[m], b);
                            if constexpr (RES) { glmvq_mma8_acc(part[m][j], r[m], b); }
                        }
                    }
                };
                step(std::integral_constant<uint32_t, 0>{});
                step(std::integral_constant<uint32_t, 1>{});
                step(std::integral_constant<uint32_t, 2>{});
                step(std::integral_constant<uint32_t, 3>{});
                const float *scale = (const float *)(src + IXB + ATB);
                #pragma unroll
                for (uint32_t j = 0; j < NNT; j++) {
                    const float2 sv = *(const float2 *)(scale + j * 8 + 2 * q);
                    #pragma unroll
                    for (uint32_t m = 0; m < MT; m++) {
                        acc[m][j][0] = fmaf(part[m][j][0], sv.x, acc[m][j][0]);
                        acc[m][j][1] = fmaf(part[m][j][1], sv.y, acc[m][j][1]);
                        acc[m][j][2] = fmaf(part[m][j][2], sv.x, acc[m][j][2]);
                        acc[m][j][3] = fmaf(part[m][j][3], sv.y, acc[m][j][3]);
                    }
                }
            }
        };
        if (nnt == 1) { run(std::integral_constant<uint32_t, 1>{}); }
        else if (nnt == 2) { run(std::integral_constant<uint32_t, 2>{}); }
        else if (nnt == 3) { run(std::integral_constant<uint32_t, (NT < 3 ? NT : 3)>{}); }
        else if (nnt == 4 || NT == 4) { run(std::integral_constant<uint32_t, (NT < 4 ? NT : 4)>{}); }
        else if (nnt <= 6) { run(std::integral_constant<uint32_t, (NT < 6 ? NT : 6)>{}); }
        else { run(std::integral_constant<uint32_t, NT>{}); }
        __syncthreads();

        if (MODE == GLMVQ_MMA_UP && mid8) {
            // amax per (128-row block, token) in the ring's first slot, as float bits.
            constexpr uint32_t BLOCKS = BM / GLMVQ_MMA8_K;
            uint32_t *amax = (uint32_t *)ring;
            for (uint32_t i = tid; i < BLOCKS * BT; i += THREADS) { amax[i] = 0; }
            __syncthreads();
            const uint32_t block = warp * MT * 16 / GLMVQ_MMA8_K;
            // All gate values load before any use: one latency per item
            // instead of one per element (17% of this kernel, NCU).
            __nv_bfloat16 gv[NT][2][MT][2];
            #pragma unroll
            for (uint32_t j = 0; j < NT; j++) {
                #pragma unroll
                for (uint32_t e = 0; e < 2; e++) {
                    const uint32_t t = j * 8 + q * 2 + e;
                    const uint64_t base = (uint64_t)(item.offset + (t < nt ? t : 0)) * rows + row;
                    #pragma unroll
                    for (uint32_t m = 0; m < MT; m++) {
                        #pragma unroll
                        for (uint32_t side = 0; side < 2; side++) {
                            gv[j][e][m][side] = j < nnt ? gate[base + lr[m][side]] : __nv_bfloat16();
                        }
                    }
                }
            }
            #pragma unroll
            for (uint32_t j = 0; j < NT; j++) {
                if (j >= nnt) { break; }
                #pragma unroll
                for (uint32_t e = 0; e < 2; e++) {
                    const uint32_t t = j * 8 + q * 2 + e;
                    float peak = 0.0f;
                    #pragma unroll
                    for (uint32_t m = 0; m < MT; m++) {
                        #pragma unroll
                        for (uint32_t side = 0; side < 2; side++) {
                            const float value = glmvq_round(glmvq_mul(acc[m][j][side * 2 + e], gain[m][side]));
                            float v;
                            if constexpr (CLAMP == GLMVQ_CLAMP_GATE) {
                                const float g0 = fminf(__bfloat162float(gv[j][e][m][side]), limit);
                                const float u0 = fminf(fmaxf(value, -limit), limit);
                                const float silu = glmvq_round(__fdiv_rn(g0, glmvq_add(1.0f, expf(-g0))));
                                v = glmvq_round(glmvq_mul(silu, u0));
                            } else {
                                v = glmvq_swiglu_value<CLAMP>(__bfloat162float(gv[j][e][m][side]), value, limit);
                            }
                            acc[m][j][side * 2 + e] = v;
                            peak = fmaxf(peak, fabsf(v));
                        }
                    }
                    for (uint32_t stride = 4; stride < 32; stride <<= 1) {
                        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, stride));
                    }
                    if (g == 0 && t < nt) { atomicMax(amax + block * BT + t, __float_as_uint(peak)); }
                }
            }
            __syncthreads();
            #pragma unroll
            for (uint32_t j = 0; j < NT; j++) {
                if (j >= nnt) { break; }
                #pragma unroll
                for (uint32_t e = 0; e < 2; e++) {
                    const uint32_t t = j * 8 + q * 2 + e;
                    if (t >= nt) { continue; }
                    const float peak = __uint_as_float(amax[block * BT + t]);
                    const float scale = peak / GLMVQ_E4M3_MAX;
                    const uint64_t base = (uint64_t)(item.offset + t) * rows + row;
                    #pragma unroll
                    for (uint32_t m = 0; m < MT; m++) {
                        #pragma unroll
                        for (uint32_t side = 0; side < 2; side++) {
                            const float v = peak > 0.0f ? __fdiv_rn(acc[m][j][side * 2 + e], scale) : 0.0f;
                            const __nv_fp8x2_storage_t pair = __nv_cvt_float2_to_fp8x2(make_float2(v, 0.0f), __NV_SATFINITE, __NV_E4M3);
                            mid8[base + lr[m][side]] = (uint8_t)(pair & 0xffu);
                        }
                    }
                    if (g == 0 && warp % (GLMVQ_MMA8_K / (MT * 16)) == 0) {
                        mid_scale[(uint64_t)(item.offset + t) * (rows / GLMVQ_MMA8_K) + row / GLMVQ_MMA8_K + block] = scale;
                    }
                }
            }
            __syncthreads();
            continue;
        }

        #pragma unroll
        for (uint32_t j = 0; j < NT; j++) {
            if (j >= nnt) { break; }
            #pragma unroll
            for (uint32_t e = 0; e < 2; e++) {
                const uint32_t t = j * 8 + q * 2 + e;
                if (t >= nt) { continue; }
                const uint64_t base = (uint64_t)(item.offset + t) * rows + row;
                #pragma unroll
                for (uint32_t m = 0; m < MT; m++) {
                    #pragma unroll
                    for (uint32_t side = 0; side < 2; side++) {
                        const uint64_t index = base + lr[m][side];
                        const float value = glmvq_round(glmvq_mul(acc[m][j][side * 2 + e], gain[m][side]));
                        if (MODE == GLMVQ_MMA_GATE) { gate[index] = __float2bfloat16_rn(value); }
                        else if (MODE == GLMVQ_MMA_DOWN) { output[index] = __float2bfloat16_rn(value); }
                        else if constexpr (CLAMP == GLMVQ_CLAMP_GATE) {
                            const float g0 = fminf(__bfloat162float(gate[index]), limit);
                            const float u0 = fminf(fmaxf(value, -limit), limit);
                            const float silu = glmvq_round(__fdiv_rn(g0, glmvq_add(1.0f, expf(-g0))));
                            mid[index] = __float2bfloat16_rn(glmvq_mul(silu, u0));
                        } else {
                            mid[index] = __float2bfloat16_rn(glmvq_swiglu_value<CLAMP>(__bfloat162float(gate[index]), value, limit));
                        }
                    }
                }
            }
        }
    }
}

template <uint32_t BM, uint32_t BT, uint32_t MT, glmvq_mma_stage MODE, uint32_t BITS = 13,
          glmvq_clamp CLAMP = GLMVQ_CLAMP_GATE, bool RES = false>
static cudaError_t glmvq_mma8_prepare(int *blocks) {
    int device = 0, limit = 0, sms = 0, active = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) { return error; }
    error = cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (error != cudaSuccess) { return error; }
    const uint32_t bytes = glmvq_mma8_bytes<BM, BT, BITS, RES>();
    if (bytes > (uint32_t)limit) { return cudaErrorNotSupported; }
    error = cudaFuncSetAttribute(glmvq_mma8_kernel<BM, BT, MT, MODE, BITS, CLAMP, RES>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (error != cudaSuccess) { return error; }
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, glmvq_mma8_kernel<BM, BT, MT, MODE, BITS, CLAMP, RES>,
        glmvq_mma8_threads<BM, MT>(), bytes);
    if (error != cudaSuccess || !active) { return error != cudaSuccess ? error : cudaErrorNotSupported; }
    error = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (error == cudaSuccess) { *blocks = sms * active; }
    return error;
}

#endif
