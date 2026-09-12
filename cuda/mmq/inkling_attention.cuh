/* Inkling prefill attention on the tensor cores (round 27).
 *
 * One CTA owns a 64-query tile of two query heads sharing a KV head. The
 * current chunk's K/V rows are first rounded to bf16 into a staging copy
 * (the value the KV store commits), so every key is a bf16 row read with
 * cp.async into a two-stage shared-memory ring of 64-key tiles. Scores are
 * bf16 m16n8k16 MMAs on the bf16-rounded Q against the tile (products are
 * exact, fp32 accumulation); the relative bias is added in fp32 and the
 * softmax runs online per tile. Probabilities are carried as a bf16 hi/lo
 * pair so the PV MMAs see ~16-bit weights; the fp32 output is normalized
 * and rounded to bf16 like the exact kernels.
 *
 * Contract: the arithmetic differs from the per-head and grouped kernels
 * only in summation order (MMA accumulation, per-tile softmax) and the
 * bf16-pair probability rounding, so outputs are not byte-identical; the
 * fixture bounds them to the exact kernels' error against an FP64 softmax
 * (tests/test_inkling_attention). Key tiles start at multiples of 64 in
 * absolute key position and every masked key contributes an exact zero,
 * so a query's result does not depend on the prefill chunk boundary. */

#include <cuda_pipeline.h>

namespace {

enum {
    IA_HEADS = 32,
    IA_HEAD_DIM = 128,
    IA_KV_HEADS = 8,
    IA_GROUP = IA_HEADS / IA_KV_HEADS,
    IA_KV_WIDTH = IA_KV_HEADS * IA_HEAD_DIM,
    IA_KV_ROW = 2 * IA_KV_WIDTH,
    IA_LOCAL_EXTENT = 512,
    IA_GLOBAL_EXTENT = 1024,
    IA_WQ = 16,
    IA_WARPS = 4,
    IA_TQ = IA_WQ * IA_WARPS,
    IA_CTA_HEADS = 2,
    IA_THREADS = IA_CTA_HEADS * IA_WARPS * 32,
    IA_TK = 64,
    IA_NB = IA_TK / 8,
    IA_KS = IA_TK / 16,
    IA_PAD = 8,
    IA_ROW = IA_HEAD_DIM + IA_PAD,
    IA_CHUNK = 8,
    IA_ROW_CHUNKS = 2 * IA_HEAD_DIM / IA_CHUNK,
    IA_TILE_CHUNKS = IA_TK * IA_ROW_CHUNKS,
    IA_CPT = IA_TILE_CHUNKS / IA_THREADS,
    IA_STAGES = 2,
    IA_STAGE_ELEMS = 2 * IA_TK * IA_ROW,
    IA_SMEM = IA_STAGES * IA_STAGE_ELEMS * (int)sizeof(uint16_t),
    IA_STAGE_THREADS = 256,
    IA_MAX_BLOCKS = 65535,
};
static_assert(IA_CPT * IA_THREADS == IA_TILE_CHUNKS, "tile copy splits evenly over the block");
static_assert((IA_ROW * sizeof(uint16_t)) % 16 == 0, "ldmatrix rows stay 16-byte aligned");

typedef tile<16, 8, nv_bfloat162> ia_tile_a;
typedef tile<8, 8, nv_bfloat162> ia_tile_b;
typedef tile<16, 8, float> ia_tile_c;

__device__ __forceinline__ float ia_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ uint32_t ia_pack(float a, float b) {
    const nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t *>(&h);
}

__device__ __forceinline__ void ia_ldsm_x4(
        uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const uint16_t *p) {
    const uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ia_ldsm_x4_trans(
        uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const uint16_t *p) {
    const uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

/* bf16 rows [rows][IA_KV_ROW] of the current chunk in the cache layout. */
__global__ void ia_stage_kernel(uint16_t *stage, const float *k, const float *v, uint32_t rows) {
    const uint64_t units = (uint64_t)rows * (IA_KV_WIDTH / 4);
    for (uint64_t u = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; u < units;
         u += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t row = u / (IA_KV_WIDTH / 4), c = (u % (IA_KV_WIDTH / 4)) * 4;
        const float4 kf = *reinterpret_cast<const float4 *>(k + row * IA_KV_WIDTH + c);
        const float4 vf = *reinterpret_cast<const float4 *>(v + row * IA_KV_WIDTH + c);
        *reinterpret_cast<uint2 *>(stage + row * IA_KV_ROW + c) =
            make_uint2(ia_pack(kf.x, kf.y), ia_pack(kf.z, kf.w));
        *reinterpret_cast<uint2 *>(stage + row * IA_KV_ROW + IA_KV_WIDTH + c) =
            make_uint2(ia_pack(vf.x, vf.y), ia_pack(vf.z, vf.w));
    }
}

__global__ __launch_bounds__(IA_THREADS, 1)
void ia_hmma_kernel(
        float *out, const float *q, const float *relative, const uint16_t *stage_rows,
        const uint16_t *cache, const uint32_t *position, uint32_t rows, uint32_t capacity,
        uint32_t extent) {
    extern __shared__ __align__(16) uint16_t ia_smem[];
    auto s_k = [&](int stage) {
        return reinterpret_cast<uint16_t (*)[IA_ROW]>(ia_smem + (size_t)stage * IA_STAGE_ELEMS);
    };
    auto s_v = [&](int stage) { return s_k(stage) + IA_TK; };
    const uint32_t tq0 = blockIdx.x * IA_TQ, h0 = blockIdx.y * IA_CTA_HEADS, kv_head = h0 / IA_GROUP;
    const uint32_t h = h0 + threadIdx.x / (IA_WARPS * 32);
    const uint32_t local = threadIdx.x % (IA_WARPS * 32), warp = local / 32, lane = local % 32;
    const uint32_t base = *position;
    const uint64_t end = (uint64_t)base + rows;
    const bool valid = end - 1 <= UINT32_MAX && (extent == IA_LOCAL_EXTENT || end <= capacity);
    if (tq0 >= rows) { return; }
    if (!valid) {
        for (uint32_t i = threadIdx.x; i < IA_TQ * IA_CTA_HEADS * IA_HEAD_DIM; i += IA_THREADS) {
            const uint32_t row = i / (IA_CTA_HEADS * IA_HEAD_DIM), rest = i % (IA_CTA_HEADS * IA_HEAD_DIM);
            if (tq0 + row < rows) {
                out[((uint64_t)(tq0 + row) * IA_HEADS + h0 + rest / IA_HEAD_DIM) * IA_HEAD_DIM +
                    rest % IA_HEAD_DIM] = NAN;
            }
        }
        return;
    }

    // C-fragment rows lane/4 and lane/4 + 8 of this warp's 16-query tile.
    uint32_t qrow[2], qpos[2], qfirst[2];
    bool alive[2];
    const float *rel[2];
    float row_m[2], row_l[2];
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        qrow[r] = warp * IA_WQ + lane / 4 + r * 8;
        alive[r] = tq0 + qrow[r] < rows;
        const uint32_t t = alive[r] ? tq0 + qrow[r] : 0;
        qpos[r] = base + t;
        qfirst[r] = extent == IA_LOCAL_EXTENT && qpos[r] >= IA_LOCAL_EXTENT
            ? qpos[r] - (IA_LOCAL_EXTENT - 1) : 0;
        rel[r] = relative + ((size_t)t * IA_HEADS + h) * extent;
        row_m[r] = -INFINITY;
        row_l[r] = 0.0f;
    }
    ia_tile_a qa[IA_HEAD_DIM / 16];
    #pragma unroll
    for (int kc = 0; kc < IA_HEAD_DIM / 16; kc++) {
        #pragma unroll
        for (int l = 0; l < ia_tile_a::ne; l++) {
            const int r = l % 2, j = (l / 2) * 4 + (int)(lane % 4);
            const uint32_t t = alive[r] ? tq0 + qrow[r] : 0;
            const float2 xy = *reinterpret_cast<const float2 *>(
                q + ((size_t)t * IA_HEADS + h) * IA_HEAD_DIM + kc * 16 + 2 * j);
            reinterpret_cast<uint32_t *>(qa[kc].x)[l] = ia_pack(xy.x, xy.y);
        }
    }

    // Keys any query of the CTA attends: [key_lo, key_hi]. Tiles start at a
    // multiple of IA_TK in absolute position (chunk invariance) and iterate
    // by count, so a range ending at UINT32_MAX cannot wrap.
    const uint32_t q_lo = base + tq0;
    const uint32_t q_hi = base + (tq0 + IA_TQ - 1 < rows ? tq0 + IA_TQ - 1 : rows - 1);
    const uint32_t key_lo = extent == IA_LOCAL_EXTENT && q_lo >= IA_LOCAL_EXTENT
        ? q_lo - (IA_LOCAL_EXTENT - 1) : 0;
    const uint32_t key_hi = q_hi;
    const uint32_t kt_start = key_lo & ~(uint32_t)(IA_TK - 1);
    const uint32_t tiles = (key_hi - kt_start) / IA_TK + 1;
    ia_tile_c output[IA_HEAD_DIM / 8];

    // 16-byte chunks of tile kt0: row r, chunk j (0-15 K dims, 16-31 V dims).
    // Rows outside [key_lo, key_hi] duplicate key_lo so masked keys stay finite.
    auto issue_tile = [&](int stage, uint32_t kt0) {
        #pragma unroll
        for (uint32_t p = 0; p < IA_CPT; p++) {
            const uint32_t idx = threadIdx.x + p * IA_THREADS;
            const uint32_t r = idx / IA_ROW_CHUNKS, j = idx % IA_ROW_CHUNKS;
            uint32_t key = kt0 + r;
            if (key < key_lo || key > key_hi) { key = key_lo; }
            const uint16_t *row = key >= base ? stage_rows + (size_t)(key - base) * IA_KV_ROW
                                              : cache + (size_t)(key % capacity) * IA_KV_ROW;
            const bool value = j >= IA_ROW_CHUNKS / 2;
            const uint32_t c = (j % (IA_ROW_CHUNKS / 2)) * IA_CHUNK;
            const uint16_t *src = row + kv_head * IA_HEAD_DIM + (value ? IA_KV_WIDTH : 0) + c;
            uint16_t *dst = (value ? s_v(stage)[r] : s_k(stage)[r]) + c;
            __pipeline_memcpy_async(dst, src, IA_CHUNK * sizeof(uint16_t));
        }
    };
    // Biases of tile kt0 as packed bf16 pairs: bpk[nb][r] holds columns 2r, 2r+1.
    auto load_bias = [&](uint32_t kt0, uint32_t (*bpk)[2]) {
        #pragma unroll
        for (int nb = 0; nb < IA_NB; nb++) {
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                uint32_t bits[2];
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    const uint32_t p = kt0 + nb * 8 + (lane % 4) * 2 + e;
                    const uint32_t distance = qpos[r] - p;
                    const float b = alive[r] && distance < extent ? ia_bf16(rel[r][distance]) : 0.0f;
                    bits[e] = __float_as_uint(b) >> 16;
                }
                bpk[nb][r] = bits[0] | (bits[1] << 16);
            }
        }
    };
    auto consume = [&](int stage, uint32_t kt0, const uint32_t (*bpk)[2]) {
        const uint16_t (*sk)[IA_ROW] = s_k(stage);
        const uint16_t (*sv)[IA_ROW] = s_v(stage);
        ia_tile_c s[IA_NB];
        const uint32_t k_row = lane & 7u, k_col = (lane >> 3) * 8u;
        #pragma unroll
        for (int nb = 0; nb < IA_NB; nb++) {
            #pragma unroll
            for (int kc = 0; kc < IA_HEAD_DIM / 16; kc += 2) {
                ia_tile_b keys0, keys1;
                uint32_t *k0 = reinterpret_cast<uint32_t *>(keys0.x);
                uint32_t *k1 = reinterpret_cast<uint32_t *>(keys1.x);
                ia_ldsm_x4(k0[0], k0[1], k1[0], k1[1], &sk[nb * 8 + k_row][kc * 16 + k_col]);
                mma(s[nb], qa[kc], keys0);
                mma(s[nb], qa[kc + 1], keys1);
            }
        }

        // score = dot / head_dim + bias; keys outside [qfirst, qpos] and
        // dead rows are masked before the row maximum.
        float tile_max[2] = {-INFINITY, -INFINITY};
        #pragma unroll
        for (int nb = 0; nb < IA_NB; nb++) {
            #pragma unroll
            for (int l = 0; l < ia_tile_c::ne; l++) {
                const int r = l / 2;
                const uint32_t p = kt0 + nb * 8 + (lane % 4) * 2 + (l % 2);
                const float bias = l % 2 ? __uint_as_float(bpk[nb][r] & 0xffff0000u)
                                         : __uint_as_float(bpk[nb][r] << 16);
                float x = fmaf(s[nb].x[l], 1.0f / IA_HEAD_DIM, bias);
                if (!alive[r] || p > qpos[r] || p < qfirst[r]) { x = -INFINITY; }
                s[nb].x[l] = x;
                tile_max[r] = fmaxf(tile_max[r], x);
            }
        }
        float rescale[2], tile_sum[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
            const float next_max = fmaxf(row_m[r], tile_max[r]);
            rescale[r] = row_m[r] == -INFINITY ? 0.0f : __expf(row_m[r] - next_max);
            row_m[r] = next_max;
        }
        #pragma unroll
        for (int nb = 0; nb < IA_NB; nb++) {
            #pragma unroll
            for (int l = 0; l < ia_tile_c::ne; l++) {
                const int r = l / 2;
                const float w = s[nb].x[l] == -INFINITY || row_m[r] == -INFINITY
                    ? 0.0f : __expf(s[nb].x[l] - row_m[r]);
                s[nb].x[l] = w;
                tile_sum[r] += w;
            }
        }
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            tile_sum[r] += __shfl_xor_sync(0xffffffffu, tile_sum[r], 1);
            tile_sum[r] += __shfl_xor_sync(0xffffffffu, tile_sum[r], 2);
            row_l[r] = row_l[r] * rescale[r] + tile_sum[r];
        }

        // Probabilities as a bf16 pair: hi = bf16(w), lo = bf16(w - hi).
        ia_tile_a phi[IA_KS], plo[IA_KS];
        #pragma unroll
        for (int ks = 0; ks < IA_KS; ks++) {
            #pragma unroll
            for (int l = 0; l < ia_tile_a::ne; l++) {
                const float w0 = s[2 * ks + l / 2].x[(l % 2) * 2], w1 = s[2 * ks + l / 2].x[(l % 2) * 2 + 1];
                phi[ks].x[l] = __floats2bfloat162_rn(w0, w1);
                const float2 back = __bfloat1622float2(phi[ks].x[l]);
                plo[ks].x[l] = __floats2bfloat162_rn(w0 - back.x, w1 - back.y);
            }
        }
        #pragma unroll
        for (int cb = 0; cb < IA_HEAD_DIM / 8; cb++) {
            #pragma unroll
            for (int l = 0; l < ia_tile_c::ne; l++) { output[cb].x[l] *= rescale[l / 2]; }
        }
        const uint32_t v_row = ((lane >> 3) & 1u) * 8u + (lane & 7u), v_col = (lane >> 4) * 8u;
        #pragma unroll
        for (int ks = 0; ks < IA_KS; ks++) {
            #pragma unroll
            for (int cb = 0; cb < IA_HEAD_DIM / 8; cb += 2) {
                ia_tile_b values0, values1;
                uint32_t *v0 = reinterpret_cast<uint32_t *>(values0.x);
                uint32_t *v1 = reinterpret_cast<uint32_t *>(values1.x);
                ia_ldsm_x4_trans(v0[0], v0[1], v1[0], v1[1], &sv[ks * 16 + v_row][cb * 8 + v_col]);
                mma(output[cb], phi[ks], values0);
                mma(output[cb], plo[ks], values0);
                mma(output[cb + 1], phi[ks], values1);
                mma(output[cb + 1], plo[ks], values1);
            }
        }
    };

    uint32_t bpk[IA_NB][2];
    issue_tile(0, kt_start);
    __pipeline_commit();
    int stage = 0;
    #pragma unroll 1
    for (uint32_t t = 0; t < tiles; t++) {
        const uint32_t kt0 = kt_start + t * IA_TK;
        if (t + 1 < tiles) {
            issue_tile(stage ^ 1, kt0 + IA_TK);
            __pipeline_commit();
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
        load_bias(kt0, bpk);
        consume(stage, kt0, bpk);
        __syncthreads();
        stage ^= 1;
    }

    #pragma unroll
    for (int cb = 0; cb < IA_HEAD_DIM / 8; cb++) {
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (!alive[r]) { continue; }
            const float2 o = make_float2(ia_bf16(__fdiv_rn(output[cb].x[2 * r], row_l[r])),
                                         ia_bf16(__fdiv_rn(output[cb].x[2 * r + 1], row_l[r])));
            *reinterpret_cast<float2 *>(
                out + ((size_t)(tq0 + qrow[r]) * IA_HEADS + h) * IA_HEAD_DIM + cb * 8 + (lane % 4) * 2) = o;
        }
    }
}

}  // namespace

extern "C" int ds4_mmq_inkling_prefill_attn_hmma(
        float *out, const float *q, const float *relative, const float *k, const float *v,
        uint16_t *stage, const uint16_t *cache, const uint32_t *position,
        uint32_t rows, uint32_t capacity, uint32_t extent, cudaStream_t stream) {
    if (!out || !q || !relative || !k || !v || !stage || !cache || !position || !rows || !capacity ||
        (extent != IA_LOCAL_EXTENT && extent != IA_GLOBAL_EXTENT)) { return -1; }
    const auto &device = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if (device.cc < GGML_CUDA_CC_AMPERE || device.smpbo < (size_t)IA_SMEM) { return -1; }
    if (cudaFuncSetAttribute(ia_hmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, IA_SMEM) !=
        cudaSuccess) {
        (void)cudaGetLastError();
        return -1;
    }
    const uint64_t units = (uint64_t)rows * (IA_KV_WIDTH / 4);
    const uint64_t wanted = (units + IA_STAGE_THREADS - 1) / IA_STAGE_THREADS;
    const unsigned blocks = (unsigned)(wanted < IA_MAX_BLOCKS ? wanted : IA_MAX_BLOCKS);
    ia_stage_kernel<<<blocks, IA_STAGE_THREADS, 0, stream>>>(stage, k, v, rows);
    const dim3 grid((rows + IA_TQ - 1) / IA_TQ, IA_HEADS / IA_CTA_HEADS);
    ia_hmma_kernel<<<grid, IA_THREADS, IA_SMEM, stream>>>(
        out, q, relative, stage, cache, position, rows, capacity, extent);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
