#ifndef DS4_GLM_VQ_Q4M_CUH
#define DS4_GLM_VQ_Q4M_CUH

#include "glm_vq_q4.cuh"
#include "glm_vq_ops.cuh"
#include <cuda_bf16.h>

// Verify-width Q4_K GEMV on BF16 tensor cores (2-8 tokens; the fused
// gate/up also at one token).
//
// The SIMT dot reloads every token's input per weight block and stalls on
// those loads (126-224 GB/s at three tokens). One m16n8k16 MMA covers 16
// rows x 8 tokens instead: A holds the exact 4-bit integers q, B the inputs,
// which the graph rounds at BF16 boundaries, so every product is exact and
// only the F32 summation order differs from the SIMT dot.
//
//   w = d*sc_s*q - dmin*m_s
//   dot = sum_s d*sc_s * (q . x)_s  -  dmin * sum_s m_s * (sum x)_s
//
// Each 32-element sub-block keeps its own MMA partial, scaled by d*sc in
// F32; a second MMA with A = m accumulates the min term over the block.
// Lane (g, c) owns elements 8c..8c+7 of every sub-block for rows g and g+8:
// one 8-byte load per row and sub-block pair feeds four MMAs, and B uses the
// same element order (the k order inside an MMA is free when A and B agree).
//
//   CTA: TILES x 16 rows, WARPS warps; a stage holds CHUNK blocks of those
//   rows and warp w takes blocks w, w + WARPS, ... of it; cp.async keeps
//   STAGES - 1 stages ahead. Long row segments keep DRAM pages open: the
//   4-block (576-byte) stages ran 6% below the SIMT stream on the head.
enum { GLMVQ_Q4M_TOKENS = 8 };
// SWIGLU: a 16-row tile holds 8 gate rows over the same 8 up rows, so each
// lane ends with gate and up of one output row and writes the clamped
// SwiGLU of their BF16 roundings, as the separate kernels do.
enum glmvq_q4m_epilogue { GLMVQ_Q4M_PLAIN, GLMVQ_Q4M_SWIGLU };

template <uint32_t TILES, uint32_t WARPS, uint32_t CHUNK, uint32_t STAGES>
struct glmvq_q4m_shape {
    static constexpr uint32_t rows = 16 * TILES, threads = 32 * WARPS;
    static constexpr uint32_t pieces = CHUNK * GLMVQ_Q4_BYTES / 16;
    // Rows 32 B apart (mod 128) in the banks: the four rows of a half-warp
    // fragment load hit distinct banks, so each 8-byte load is conflict-free.
    static constexpr uint32_t stride = CHUNK * GLMVQ_Q4_BYTES + (160 - CHUNK * GLMVQ_Q4_BYTES % 128) % 128;
    static constexpr uint32_t stage = rows * stride, smem = STAGES * stage;
};

__device__ __forceinline__ uint32_t glmvq_q4m_u32(__nv_bfloat162 v) { return *(uint32_t *)&v; }
__device__ __forceinline__ __nv_bfloat162 glmvq_q4m_bf(uint32_t v) { return *(__nv_bfloat162 *)&v; }

// (v & mask) | 0x4300 per half: 128 + nibble, exact in BF16.
__device__ __forceinline__ uint32_t glmvq_q4m_or(uint32_t v, uint32_t mask) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(v), "r"(mask), "r"(0x43004300u));
    return r;
}

// Bytes b0..b3 of w: lo = {(b0, b2), (b1, b3)} low nibbles, hi likewise
// for the high nibbles, as exact BF16 pairs.
__device__ __forceinline__ void glmvq_q4m_unpack(uint32_t w, uint32_t lo[2], uint32_t hi[2]) {
    const __nv_bfloat162 base = __floats2bfloat162_rn(128.0f, 128.0f);
    #pragma unroll
    for (uint32_t i = 0; i < 2; i++) {
        lo[i] = glmvq_q4m_u32(__hsub2(glmvq_q4m_bf(glmvq_q4m_or(w >> (8 * i), 0x000F000Fu)), base));
        hi[i] = glmvq_q4m_u32(__hsub2(glmvq_q4m_bf(glmvq_q4m_or(w >> (8 * i + 4), 0x000F000Fu)), base));
    }
}

__device__ __forceinline__ void glmvq_q4m_mma(float c[4], uint32_t a0, uint32_t a1, uint32_t a2,
        uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One 256-element block of the staged rows. sum[t] holds rows (g, g+8) of
// tile t for tokens (2c, 2c+1), the MMA accumulator layout.
template <uint32_t TILES, uint32_t STRIDE>
__device__ __forceinline__ void glmvq_q4m_block(const uint8_t *rows, const float *input,
        uint32_t cols, uint32_t block, uint32_t tokens, float sum[TILES][4]) {
    const uint32_t lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    uint32_t b[8][4];
    const float *x = input + (uint64_t)g * cols + block * GLMVQ_Q4_BLOCK + 8 * c;
    #pragma unroll
    for (uint32_t s = 0; s < 8; s++) {
        float4 u = {}, v = {};
        if (g < tokens) { u = *(const float4 *)(x + 32 * s); v = *(const float4 *)(x + 32 * s + 4); }
        b[s][0] = glmvq_q4m_u32(__floats2bfloat162_rn(u.x, u.z));
        b[s][1] = glmvq_q4m_u32(__floats2bfloat162_rn(u.y, u.w));
        b[s][2] = glmvq_q4m_u32(__floats2bfloat162_rn(v.x, v.z));
        b[s][3] = glmvq_q4m_u32(__floats2bfloat162_rn(v.y, v.w));
    }
    #pragma unroll
    for (uint32_t t = 0; t < TILES; t++) {
        const uint8_t *r0 = rows + (16 * t + g) * STRIDE, *r1 = r0 + 8 * STRIDE;
        const uint4 h0 = *(const uint4 *)r0, h1 = *(const uint4 *)r1;
        const float d0 = __half2float(__ushort_as_half(h0.x & 65535u));
        const float d1 = __half2float(__ushort_as_half(h1.x & 65535u));
        float minimum[4] = {};
        #pragma unroll
        for (uint32_t p = 0; p < 4; p++) {
            const uint2 w0 = *(const uint2 *)(r0 + 16 + 32 * p + 8 * c);
            const uint2 w1 = *(const uint2 *)(r1 + 16 + 32 * p + 8 * c);
            // q[row][word][nibble][pair]: words x/y hold elements 8c..8c+3 / 8c+4..8c+7.
            uint32_t q[2][2][2][2];
            glmvq_q4m_unpack(w0.x, q[0][0][0], q[0][0][1]);
            glmvq_q4m_unpack(w0.y, q[0][1][0], q[0][1][1]);
            glmvq_q4m_unpack(w1.x, q[1][0][0], q[1][0][1]);
            glmvq_q4m_unpack(w1.y, q[1][1][0], q[1][1][1]);
            #pragma unroll
            for (uint32_t n = 0; n < 2; n++) {
                const uint32_t s = 2 * p + n;
                float sc0, m0, sc1, m1;
                glmvq_q4_scales(h0, s, &sc0, &m0);
                glmvq_q4_scales(h1, s, &sc1, &m1);
                float part[4] = {};
                glmvq_q4m_mma(part, q[0][0][n][0], q[1][0][n][0], q[0][0][n][1], q[1][0][n][1], b[s][0], b[s][1]);
                glmvq_q4m_mma(part, q[0][1][n][0], q[1][1][n][0], q[0][1][n][1], q[1][1][n][1], b[s][2], b[s][3]);
                const uint32_t a0 = glmvq_q4m_u32(__float2bfloat162_rn(m0));
                const uint32_t a1 = glmvq_q4m_u32(__float2bfloat162_rn(m1));
                glmvq_q4m_mma(minimum, a0, a1, a0, a1, b[s][0], b[s][1]);
                glmvq_q4m_mma(minimum, a0, a1, a0, a1, b[s][2], b[s][3]);
                const float s0 = d0 * sc0, s1 = d1 * sc1;
                sum[t][0] = fmaf(s0, part[0], sum[t][0]);
                sum[t][1] = fmaf(s0, part[1], sum[t][1]);
                sum[t][2] = fmaf(s1, part[2], sum[t][2]);
                sum[t][3] = fmaf(s1, part[3], sum[t][3]);
            }
        }
        const float n0 = -__half2float(__ushort_as_half(h0.x >> 16));
        const float n1 = -__half2float(__ushort_as_half(h1.x >> 16));
        sum[t][0] = fmaf(n0, minimum[0], sum[t][0]);
        sum[t][1] = fmaf(n0, minimum[1], sum[t][1]);
        sum[t][2] = fmaf(n1, minimum[2], sum[t][2]);
        sum[t][3] = fmaf(n1, minimum[3], sum[t][3]);
    }
}

template <uint32_t TILES, uint32_t WARPS, uint32_t CHUNK, uint32_t STAGES,
    glmvq_q4m_epilogue EPI = GLMVQ_Q4M_PLAIN>
__global__ void __launch_bounds__(WARPS * 32) glmvq_q4m(const uint8_t *weight, const uint8_t *up,
        const float *input, float *out, uint32_t cols, uint32_t rows, uint32_t tokens, float limit) {
    using shape = glmvq_q4m_shape<TILES, WARPS, CHUNK, STAGES>;
    static_assert(EPI == GLMVQ_Q4M_PLAIN || TILES == 1, "SwiGLU pairs one tile");
    // Output rows per CTA: SWIGLU stages 8 gate and the same 8 up rows.
    constexpr uint32_t outs = EPI == GLMVQ_Q4M_SWIGLU ? 8 : shape::rows;
    extern __shared__ uint4 glmvq_q4m_smem[];
    __shared__ float red[WARPS][shape::rows][GLMVQ_Q4M_TOKENS];
    uint8_t *staged = (uint8_t *)glmvq_q4m_smem;
    glmvq_pdl_open();
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, c = lane & 3;
    const uint32_t blocks = cols / GLMVQ_Q4_BLOCK, chunks = (blocks + CHUNK - 1) / CHUNK;
    const uint32_t row0 = blockIdx.x * outs;
    const uint64_t row_bytes = (uint64_t)blocks * GLMVQ_Q4_BYTES;

    // Stage j: blocks CHUNK*j.. of the CTA's rows; constant weights need no
    // predecessor, so the first stages stream while it drains.
    auto load = [&](uint32_t j) {
        if (j < chunks) {
            const uint32_t first = j * CHUNK, pieces = min(CHUNK, blocks - first) * (GLMVQ_Q4_BYTES / 16);
            uint8_t *dst = staged + j % STAGES * shape::stage;
            const uint64_t offset = (uint64_t)row0 * row_bytes + first * GLMVQ_Q4_BYTES;
            for (uint32_t i = threadIdx.x; i < shape::rows * shape::pieces; i += shape::threads) {
                const uint32_t r = i / shape::pieces, piece = i % shape::pieces;
                // SWIGLU: staged rows 0-7 are gate rows row0.., rows 8-15 the same up rows.
                const uint32_t local = EPI == GLMVQ_Q4M_SWIGLU ? r % outs : r;
                const uint8_t *src = (EPI == GLMVQ_Q4M_SWIGLU && r >= outs ? up : weight) + offset;
                if (piece < pieces && row0 + local < rows) {
                    __pipeline_memcpy_async(dst + r * shape::stride + piece * 16, src + local * row_bytes + piece * 16, 16);
                }
            }
        }
        __pipeline_commit();
    };
    #pragma unroll
    for (uint32_t j = 0; j + 1 < STAGES; j++) { load(j); }
    glmvq_pdl_wait();

    float sum[TILES][4] = {};
    for (uint32_t j = 0; j < chunks; j++) {
        __pipeline_wait_prior(STAGES - 2);
        // Also retires every warp's reads of the stage reloaded below.
        __syncthreads();
        load(j + STAGES - 1);
        const uint8_t *stage = staged + j % STAGES * shape::stage;
        for (uint32_t k = warp; k < CHUNK && j * CHUNK + k < blocks; k += WARPS) {
            glmvq_q4m_block<TILES, shape::stride>(stage + k * GLMVQ_Q4_BYTES, input, cols, j * CHUNK + k, tokens, sum);
        }
    }

    // Warps hold disjoint blocks; add them in warp order.
    #pragma unroll
    for (uint32_t t = 0; t < TILES; t++) {
        red[warp][16 * t + g][2 * c] = sum[t][0];
        red[warp][16 * t + g][2 * c + 1] = sum[t][1];
        red[warp][16 * t + g + 8][2 * c] = sum[t][2];
        red[warp][16 * t + g + 8][2 * c + 1] = sum[t][3];
    }
    __syncthreads();
    auto total = [&](uint32_t r, uint32_t n) {
        float value = red[0][r][n];
        #pragma unroll
        for (uint32_t w = 1; w < WARPS; w++) { value += red[w][r][n]; }
        return __bfloat162float(__float2bfloat16_rn(value));
    };
    for (uint32_t i = threadIdx.x; i < outs * tokens; i += shape::threads) {
        const uint32_t r = i % outs, n = i / outs;
        if (row0 + r >= rows) { continue; }
        out[(uint64_t)n * rows + row0 + r] = EPI == GLMVQ_Q4M_SWIGLU
            ? glmvq_swiglu_value(total(r, n), total(r + outs, n), limit) : total(r, n);
    }
}

// Production shape, the fastest of eleven on the GLM projections at two to
// four tokens: one 16-row tile, 16 warps, 16-block (2,304-byte, whole-line)
// row segments and two stages (75 KiB, one CTA per SM).
enum { GLMVQ_Q4M_TILES = 1, GLMVQ_Q4M_WARPS = 16, GLMVQ_Q4M_CHUNK = 16, GLMVQ_Q4M_STAGES = 2 };
using glmvq_q4m_prod = glmvq_q4m_shape<GLMVQ_Q4M_TILES, GLMVQ_Q4M_WARPS, GLMVQ_Q4M_CHUNK, GLMVQ_Q4M_STAGES>;

template <glmvq_q4m_epilogue EPI>
static constexpr auto glmvq_q4m_kernel =
    glmvq_q4m<GLMVQ_Q4M_TILES, GLMVQ_Q4M_WARPS, GLMVQ_Q4M_CHUNK, GLMVQ_Q4M_STAGES, EPI>;

static cudaError_t glmvq_q4m_prepare() {
    cudaError_t err = cudaFuncSetAttribute(glmvq_q4m_kernel<GLMVQ_Q4M_PLAIN>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, glmvq_q4m_prod::smem);
    if (err == cudaSuccess) {
        err = cudaFuncSetAttribute(glmvq_q4m_kernel<GLMVQ_Q4M_SWIGLU>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, glmvq_q4m_prod::smem);
    }
    return err;
}

static bool glmvq_q4m_fits(const void *weight, const float *input, uint32_t cols, uint32_t tokens) {
    return tokens && tokens <= GLMVQ_Q4M_TOKENS && cols && cols % GLMVQ_Q4_BLOCK == 0 &&
        (uintptr_t)weight % 16 == 0 && (uintptr_t)input % 16 == 0;
}

static cudaError_t glmvq_q4m_decode(const void *weight, const float *input, float *out,
        uint32_t rows, uint32_t cols, uint32_t tokens, cudaStream_t stream, glmvq_launch launch) {
    if (!out || !rows || !glmvq_q4m_fits(weight, input, cols, tokens)) { return cudaErrorInvalidValue; }
    return glmvq_launch_kernel(launch, glmvq_q4m_kernel<GLMVQ_Q4M_PLAIN>, (rows + glmvq_q4m_prod::rows - 1) / glmvq_q4m_prod::rows,
        glmvq_q4m_prod::threads, glmvq_q4m_prod::smem, stream, (const uint8_t *)weight, (const uint8_t *)nullptr,
        input, out, cols, rows, tokens, 0.0f);
}

// Gate and up of one input with the clamped SwiGLU: `rows` mid rows.
static cudaError_t glmvq_q4m_swiglu(const void *gate, const void *up, const float *input, float *mid,
        uint32_t rows, uint32_t cols, uint32_t tokens, float limit, cudaStream_t stream, glmvq_launch launch) {
    if (!mid || !rows || !glmvq_q4m_fits(gate, input, cols, tokens) || (uintptr_t)up % 16) {
        return cudaErrorInvalidValue;
    }
    constexpr uint32_t outs = glmvq_q4m_prod::rows / 2;
    return glmvq_launch_kernel(launch, glmvq_q4m_kernel<GLMVQ_Q4M_SWIGLU>, (rows + outs - 1) / outs,
        glmvq_q4m_prod::threads, glmvq_q4m_prod::smem, stream, (const uint8_t *)gate, (const uint8_t *)up,
        input, mid, cols, rows, tokens, limit);
}
#endif
