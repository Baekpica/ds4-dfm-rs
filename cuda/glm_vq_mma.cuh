#ifndef DS4_GLM_VQ_MMA_CUH
#define DS4_GLM_VQ_MMA_CUH

#include "glm_vq_linear.cuh"
#include <type_traits>

/* Packed staging and MMA layout adapted from YoungAi 7fb26cc332d740416792204a29c7d13d2986caf5,
 * src/cuda/cuda_vq_reg_mma.inc.cu:29-208 (MIT; notices in ../LICENSE).
 * GLM differs: gain is applied before BF16 MMA, and SwiGLU keeps GLM rounding. */
enum { GLMVQ_BOOK_BYTES = 8192 * GLMVQ_VECTOR, GLMVQ_MMA_RS = 2,
       GLMVQ_MMA_STAGES = 2, GLMVQ_PREFETCH_BYTES = 384 };
enum glmvq_mma_stage { GLMVQ_MMA_GATE, GLMVQ_MMA_UP, GLMVQ_MMA_DOWN };

struct glmvq_mma_item {
    uint32_t matrix[3];
    uint32_t offset, tokens;
};

static bool glmvq_mma_fits(const glmvq_source &s, uint32_t rows,
        uint32_t cols, const void *book) {
    const glmvq_matrix &m = s.vq;
    return s.storage == GLMVQ_WEIGHT_VQ && s.rows == rows && s.cols == cols &&
        m.rows == rows && m.cols == cols && m.precision == GLMVQ_E4M3 &&
        m.main && m.plane && m.gain && book && m.book == book &&
        ((uintptr_t)m.main & 127u) == 0 && ((uintptr_t)m.plane & 127u) == 0 &&
        ((uintptr_t)m.gain & 1u) == 0 && ((uintptr_t)book & 15u) == 0;
}

__global__ void glmvq_mma_gather(__nv_bfloat16 *output, const float *input,
        const uint32_t *pairs, uint32_t used, uint32_t width, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint32_t token = pairs[i / width] / used;
    output[i] = __float2bfloat16_rn(input[(uint64_t)token * width + i % width]);
}

// `shared` (optional): the shared expert's output, added at the same BF16
// boundary the separate add + round passes applied.
__global__ void glmvq_mma_reduce(float *output, const float *input,
        const float *weights, const uint32_t *pairs, const uint32_t *order,
        uint32_t used, uint32_t width, uint64_t count, const float *shared = nullptr) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint32_t token = i / width, col = i % width;
    float sum = 0.0f;
    // The worklist stores ascending expert order, preserving GLM's BF16
    // multiply/add boundary without atomics or cross-token reductions.
    for (uint32_t k = 0; k < used; k++) {
        const uint32_t row = order[(uint64_t)token * used + k];
        const float value = glmvq_round(glmvq_mul(input[(uint64_t)row * width + col], weights[pairs[row]]));
        sum = glmvq_round(glmvq_add(sum, value));
    }
    output[i] = shared ? glmvq_round(glmvq_add(sum, shared[i])) : sum;
}

__device__ __forceinline__ uint32_t glmvq_mma_code(uint32_t a, uint32_t b,
        uint32_t c, uint32_t k, uint32_t half) {
    if (k == 0) { return (a >> (12 * half)) & 4095u; }
    if (k == 1) { return (half ? b >> 4 : __funnelshift_r(a, b, 24)) & 4095u; }
    if (k == 2) { return __funnelshift_r(b, c, 16 + 12 * half) & 4095u; }
    return (c >> (8 + 12 * half)) & 4095u;
}

template<uint32_t BYTES>
__device__ __forceinline__ void glmvq_mma_copy(void *dst, const void *src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::
        "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "n"(BYTES) : "memory");
}

__device__ __forceinline__ void glmvq_mma_pad(void *dst, const void *src, uint32_t bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
        "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void glmvq_mma_prefetch(const void *src, uint32_t bytes) {
#if __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src), "r"(bytes) : "memory");
#else
    asm volatile("prefetch.global.L2 [%0];" :: "l"(src) : "memory");
#endif
}

__device__ __forceinline__ void glmvq_mma_zero(float *d, const uint32_t *a, uint2 b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y), "f"(0.0f));
}

template<uint32_t BM, uint32_t BT>
static constexpr uint32_t glmvq_mma_bytes() {
    return GLMVQ_BOOK_BYTES + GLMVQ_MMA_STAGES *
        (BM * (12 * GLMVQ_MMA_RS + 4) + BT * 128 * GLMVQ_MMA_RS);
}

template<uint32_t BM, uint32_t BT, glmvq_mma_stage MODE>
__global__ __launch_bounds__(BM * 2, 1) void glmvq_mma_kernel(
        const glmvq_source *sources, const glmvq_mma_item *items, uint32_t nitems,
        const uint8_t *book, const __nv_bfloat16 *input, float *gate,
        __nv_bfloat16 *mid, float *output, uint32_t rows, uint32_t cols, float limit) {
    static_assert((BM == 128 || BM == 256) && (BT == 16 || BT == 32), "bounded GLM tiles");
    constexpr uint32_t THREADS = BM * 2, RS = GLMVQ_MMA_RS, RB = 12 * RS;
    constexpr uint32_t BSB = BM * RB, EXB = BM * 4, ATB = BT * 128 * RS;
    constexpr uint32_t STB = BSB + EXB + ATB, NT = BT / 8, PREFETCH_AHEAD = 2;
    extern __shared__ __align__(16) uint8_t memory[];
    uint8_t *ring = memory + GLMVQ_BOOK_BYTES;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, q = lane & 3u, h = q >> 1, half = q & 1u;
    for (uint32_t i = tid * 16; i < GLMVQ_BOOK_BYTES; i += THREADS * 16) {
        glmvq_mma_copy<16>(memory + i, book + i);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    const uint32_t tiles = rows / BM, stages = cols / (64 * RS);
    const uint32_t la = warp * 16 + g, lb = la + 8;
    const uint32_t cofs[4] = {((0 + h) ^ (2 * (g & 3u))) * 16,
        ((2 + h) ^ (2 * (g & 3u))) * 16, ((4 + h) ^ (2 * (g & 3u))) * 16,
        ((6 + h) ^ (2 * (g & 3u))) * 16};
    for (uint32_t work = blockIdx.x; work < nitems * tiles; work += gridDim.x) {
        const glmvq_mma_item item = items[work / tiles];
        const glmvq_matrix m = sources[item.matrix[MODE]].vq;
        const uint32_t row = (work % tiles) * BM, nt = item.tokens;
        const uint32_t main_stride = cols * 3 / 16, plane_stride = cols / 64;
        const float ga = __half2float(m.gain[row + la]), gb = __half2float(m.gain[row + lb]);
        const __nv_bfloat16 *x = input + (uint64_t)item.offset * cols;
        auto prefetch = [&](uint32_t chunk) {
            const uint32_t offset = chunk * GLMVQ_PREFETCH_BYTES;
            if (tid < BM && offset < main_stride) {
                glmvq_mma_prefetch(m.main + (uint64_t)(row + tid) * main_stride + offset,
                    min((uint32_t)GLMVQ_PREFETCH_BYTES, main_stride - offset));
            }
        };
        for (uint32_t chunk = 0; chunk < PREFETCH_AHEAD; chunk++) { prefetch(chunk); }
        if (tid == 0) { glmvq_mma_prefetch(m.plane + (uint64_t)row * plane_stride, BM * plane_stride); }

        auto issue = [&](uint32_t stage) {
            uint8_t *dst = ring + (stage % GLMVQ_MMA_STAGES) * STB;
            for (uint32_t i = tid; i < BM * 3; i += THREADS) {
                const uint32_t r = i / 3, part = i % 3;
                glmvq_mma_copy<8>(dst + r * RB + part * 8,
                    m.main + (uint64_t)(row + r) * main_stride + stage * RB + part * 8);
            }
            if (tid < BM) {
                glmvq_mma_copy<4>(dst + BSB + tid * 4,
                    m.plane + (uint64_t)(row + tid) * plane_stride + 4 * ((stage * RS) / 4));
            }
            // Fill the last n8 tile with zeros; valid token columns stay independent.
            const uint32_t copies = ((nt + 7) / 8 * 8) * 8 * RS;
            for (uint32_t i = tid; i < copies; i += THREADS) {
                const uint32_t t = i / (8 * RS), rem = i % (8 * RS), r = rem / 8, c = rem % 8;
                glmvq_mma_pad(dst + BSB + EXB + (r * BT + t) * 128 + ((c ^ (2 * (t & 3u))) * 16),
                    x + (uint64_t)(t < nt ? t : 0) * cols + (RS * stage + r) * 64 + c * 8,
                    t < nt ? 16 : 0);
            }
        };
        issue(0);
        asm volatile("cp.async.commit_group;" ::: "memory");
        const uint32_t nnt = (nt + 7) / 8;
        float acc[NT][4] = {};
        auto run = [&](auto count) {
            constexpr uint32_t NNT = decltype(count)::value;
            for (uint32_t stage = 0; stage < stages; stage++) {
                asm volatile("cp.async.wait_group 0;" ::: "memory");
                __syncthreads();
                if (stage + 1 < stages) { issue(stage + 1); }
                asm volatile("cp.async.commit_group;" ::: "memory");
                if (stage % (GLMVQ_PREFETCH_BYTES / RB) == 0) {
                    prefetch(stage / (GLMVQ_PREFETCH_BYTES / RB) + PREFETCH_AHEAD);
                }
                const uint8_t *src = ring + (stage % GLMVQ_MMA_STAGES) * STB;
                uint32_t wa[3 * RS], wb[3 * RS];
                #pragma unroll
                for (uint32_t k = 0; k < 3; k++) {
                    const uint2 a = *(const uint2 *)(src + la * RB + k * 8);
                    const uint2 b = *(const uint2 *)(src + lb * RB + k * 8);
                    wa[2 * k] = a.x; wa[2 * k + 1] = a.y;
                    wb[2 * k] = b.x; wb[2 * k + 1] = b.y;
                }
                const uint32_t ea = *(const uint32_t *)(src + BSB + la * 4);
                const uint32_t eb = *(const uint32_t *)(src + BSB + lb * 4);
                const uint8_t *activation = src + BSB + EXB + half * 8 + g * 128;
                #pragma unroll
                for (uint32_t r = 0; r < RS; r++) {
                    const uint32_t bit = ((RS * stage + r) & 3u) * 8;
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; k++) {
                        uint32_t ia = glmvq_mma_code(wa[3 * r], wa[3 * r + 1], wa[3 * r + 2], k, h);
                        uint32_t ib = glmvq_mma_code(wb[3 * r], wb[3 * r + 1], wb[3 * r + 2], k, h);
                        ia |= ((ea >> (bit + 2 * k + h)) & 1u) << 12;
                        ib |= ((eb >> (bit + 2 * k + h)) & 1u) << 12;
                        const uint32_t ca = *(const uint32_t *)(memory + ia * 8 + half * 4);
                        const uint32_t cb = *(const uint32_t *)(memory + ib * 8 + half * 4);
                        const uint32_t a[4] = {glmvq_scaled_pair(ca, ga), glmvq_scaled_pair(cb, gb),
                            glmvq_scaled_pair(ca >> 16, ga), glmvq_scaled_pair(cb >> 16, gb)};
                        const uint8_t *ap = activation + r * BT * 128 + cofs[k];
                        #pragma unroll
                        for (uint32_t j = 0; j < NNT; j++) {
                            float partial[4];
                            glmvq_mma_zero(partial, a, *(const uint2 *)(ap + j * 8 * 128));
                            #pragma unroll
                            for (uint32_t c = 0; c < 4; c++) { acc[j][c] = glmvq_add(acc[j][c], partial[c]); }
                        }
                    }
                }
            }
        };
        if (nnt == 1) { run(std::integral_constant<uint32_t, 1>{}); }
        else if (nnt == 2) { run(std::integral_constant<uint32_t, 2>{}); }
        else if (nnt == 3) { run(std::integral_constant<uint32_t, (NT < 3 ? NT : 3)>{}); }
        else { run(std::integral_constant<uint32_t, NT>{}); }
        __syncthreads();

        #pragma unroll
        for (uint32_t j = 0; j < NT; j++) {
            if (j >= nnt) { break; }
            #pragma unroll
            for (uint32_t e = 0; e < 2; e++) {
                const uint32_t t = j * 8 + q * 2 + e;
                if (t >= nt) { continue; }
                const uint64_t base = (uint64_t)(item.offset + t) * rows + row;
                #pragma unroll
                for (uint32_t side = 0; side < 2; side++) {
                    const uint64_t index = base + (side ? lb : la);
                    const float value = glmvq_round(acc[j][side * 2 + e]);
                    if (MODE == GLMVQ_MMA_GATE) { gate[index] = value; }
                    else if (MODE == GLMVQ_MMA_DOWN) { output[index] = value; }
                    else {
                        const float g0 = fminf(gate[index], limit);
                        const float u0 = fminf(fmaxf(value, -limit), limit);
                        const float silu = glmvq_round(__fdiv_rn(g0, glmvq_add(1.0f, expf(-g0))));
                        mid[index] = __float2bfloat16_rn(glmvq_mul(silu, u0));
                    }
                }
            }
        }
    }
}

template<uint32_t BM, uint32_t BT, glmvq_mma_stage MODE>
static cudaError_t glmvq_mma_prepare(int *blocks) {
    int device = 0, limit = 0, sms = 0, active = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) { return error; }
    error = cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (error != cudaSuccess) { return error; }
    const uint32_t bytes = glmvq_mma_bytes<BM, BT>();
    if (bytes > (uint32_t)limit) { return cudaErrorNotSupported; }
    error = cudaFuncSetAttribute(glmvq_mma_kernel<BM, BT, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (error != cudaSuccess) { return error; }
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, glmvq_mma_kernel<BM, BT, MODE>, BM * 2, bytes);
    if (error != cudaSuccess || !active) { return error != cudaSuccess ? error : cudaErrorNotSupported; }
    error = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (error == cudaSuccess) { *blocks = sms * active; }
    return error;
}

#endif
