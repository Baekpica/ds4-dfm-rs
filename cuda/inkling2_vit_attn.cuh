/* Inkling-2 ViT dense attention, flash style: 16 heads of 64 over one image's
 * rows (packed BF16 [q | k | v], 3,072 per row), scale 1/8, no mask.
 *
 *   CTA = 64 query rows x 1 head, 4 warps of 16 rows
 *   for each block of 64 keys (K and V double-buffered through cp.async):
 *       S = Q K^T            m16n8k16 BF16 MMA, F32
 *       online softmax       exp2 of S * log2(e) / 8, F32 row max and sum
 *       O += P V             P rounded to BF16 (unnormalized, in [0, 1])
 *   O / l, rounded to BF16
 *
 * V4.1's tiled path materializes S (F32) and P (BF16) for every query and key:
 * ~460 GB of traffic per layer at 10,609 tokens (108 s per image); here S
 * and P stay in registers. Shared rows are 128 bytes (64 BF16): their 16-byte
 * chunks are XOR-swizzled by row so ldmatrix reads eight rows conflict-free.
 * Standalone: tests/test_inkling2_vit_attn.cu includes it. */
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

enum {
    INK2_FA_HEAD = 64, INK2_FA_HEADS = 16, INK2_FA_ROW = 3 * INK2_FA_HEADS * INK2_FA_HEAD,
    INK2_FA_BQ = 64, INK2_FA_BK = 64, INK2_FA_THREADS = 128,
};

// Byte offset of (row, 16-byte chunk) in a swizzled 64 x 64 BF16 tile.
__device__ __forceinline__ uint32_t ink2_fa_at(uint32_t row, uint32_t chunk) {
    return row * 128u + ((chunk ^ (row & 7u)) << 4);
}

// A 64-row tile of one head's q, k or v columns (rows past n read as zero).
__device__ __forceinline__ void ink2_fa_load(uint32_t tile, const __nv_bfloat16 *qkv, uint32_t row0, uint32_t n,
        uint32_t col) {
    for (uint32_t i = threadIdx.x; i < 64u * 8u; i += INK2_FA_THREADS) {
        const uint32_t r = i / 8u, ch = i % 8u, row = row0 + r;
        const __nv_bfloat16 *src = qkv + (uint64_t)(row < n ? row : 0u) * INK2_FA_ROW + col + ch * 8u;
        const uint32_t bytes = row < n ? 16u : 0u;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(tile + ink2_fa_at(r, ch)), "l"(src), "r"(bytes));
    }
}

__device__ __forceinline__ void ink2_fa_ldsm4(uint32_t addr, uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ink2_fa_ldsm4t(uint32_t addr, uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ink2_fa_mma(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t ink2_fa_pack(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&v);
}

/* o[r - q0][h * 64 + d] for query rows q0 <= r < q0 + m (grid: m / 64 up, 16 heads). */
__global__ void __launch_bounds__(INK2_FA_THREADS) ink2_vit_flash_kernel(float *o, const __nv_bfloat16 *qkv,
        uint32_t n, uint32_t q0, uint32_t m) {
    __shared__ __align__(128) __nv_bfloat16 smem[5 * INK2_FA_BQ * INK2_FA_HEAD];
    // Q, then K and V of key blocks b at tiles 1 + b % 2 and 3 + b % 2 (no
    // runtime-indexed arrays: those land on the stack).
    const uint32_t sq = (uint32_t)__cvta_generic_to_shared(smem), tile = INK2_FA_BQ * INK2_FA_HEAD * 2u;
    const uint32_t h = blockIdx.y, w = threadIdx.x / 32u, lane = threadIdx.x % 32u, g = lane / 4u, t = lane % 4u;
    const uint32_t row0 = q0 + blockIdx.x * INK2_FA_BQ, end = q0 + m;
    const uint32_t qc = h * INK2_FA_HEAD, kc = INK2_FA_HEADS * INK2_FA_HEAD + qc, vc = 2u * INK2_FA_HEADS * INK2_FA_HEAD + qc;
    const uint32_t blocks = (n + INK2_FA_BK - 1u) / INK2_FA_BK;

    ink2_fa_load(sq, qkv, row0, end, qc);
    ink2_fa_load(sq + tile, qkv, 0u, n, kc);
    ink2_fa_load(sq + 3u * tile, qkv, 0u, n, vc);
    asm volatile("cp.async.commit_group;\n");

    uint32_t qa[4][4];
    float acc[8][4] = {}, mx[2] = {-INFINITY, -INFINITY}, sum[2] = {0.0f, 0.0f};
    const float scale = 0.125f * 1.4426950408889634f;
    for (uint32_t b = 0; b < blocks; b++) {
        if (b + 1u < blocks) {
            const uint32_t next = ((b + 1u) & 1u) * tile;
            ink2_fa_load(sq + tile + next, qkv, (b + 1u) * INK2_FA_BK, n, kc);
            ink2_fa_load(sq + 3u * tile + next, qkv, (b + 1u) * INK2_FA_BK, n, vc);
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();
        if (b == 0u) {
            // A fragments of this warp's 16 query rows, four k steps of 16 dims.
            for (uint32_t k = 0; k < 4u; k++) {
                const uint32_t r = w * 16u + lane % 8u + (lane / 8u) % 2u * 8u, ch = 2u * k + lane / 16u;
                ink2_fa_ldsm4(sq + ink2_fa_at(r, ch), qa[k][0], qa[k][1], qa[k][2], qa[k][3]);
            }
        }
        const uint32_t kt = sq + (1u + (b & 1u)) * tile, vt = sq + (3u + (b & 1u)) * tile, key0 = b * INK2_FA_BK;

        // S = Q K^T: eight n tiles of 8 keys; ldmatrix x4 covers two of them per k step.
        float s[8][4] = {};
        for (uint32_t j = 0; j < 8u; j += 2u) {
            for (uint32_t k = 0; k < 4u; k++) {
                const uint32_t mtx = lane / 8u, r = j * 8u + lane % 8u + mtx / 2u * 8u, ch = 2u * k + mtx % 2u;
                uint32_t b0, b1, b2, b3;
                ink2_fa_ldsm4(kt + ink2_fa_at(r, ch), b0, b1, b2, b3);
                ink2_fa_mma(s[j], qa[k], b0, b1);
                ink2_fa_mma(s[j + 1u], qa[k], b2, b3);
            }
        }

        // Online softmax over this block (rows g and g + 8 of the warp).
        float top[2] = {-INFINITY, -INFINITY};
        for (uint32_t j = 0; j < 8u; j++) {
            for (uint32_t e = 0; e < 4u; e++) {
                const uint32_t key = key0 + j * 8u + 2u * t + e % 2u;
                s[j][e] = key < n ? s[j][e] * scale : -INFINITY;
                top[e / 2u] = fmaxf(top[e / 2u], s[j][e]);
            }
        }
        float alpha[2];
        for (uint32_t i = 0; i < 2u; i++) {
            top[i] = fmaxf(top[i], __shfl_xor_sync(0xffffffffu, top[i], 1));
            top[i] = fmaxf(top[i], __shfl_xor_sync(0xffffffffu, top[i], 2));
            const float next = fmaxf(mx[i], top[i]);
            alpha[i] = exp2f(mx[i] - next);
            mx[i] = next;
            sum[i] *= alpha[i];
        }
        for (uint32_t j = 0; j < 8u; j++) {
            for (uint32_t e = 0; e < 4u; e++) {
                s[j][e] = exp2f(s[j][e] - mx[e / 2u]);
                sum[e / 2u] += s[j][e];
                acc[j][e] *= alpha[e / 2u];
            }
        }

        // O += P V: P's C fragments of key tiles 2k and 2k + 1 are the A
        // fragment of k step k; V through ldmatrix.trans, two dim tiles each.
        for (uint32_t k = 0; k < 4u; k++) {
            const uint32_t pa[4] = {ink2_fa_pack(s[2u * k][0], s[2u * k][1]), ink2_fa_pack(s[2u * k][2], s[2u * k][3]),
                                    ink2_fa_pack(s[2u * k + 1u][0], s[2u * k + 1u][1]),
                                    ink2_fa_pack(s[2u * k + 1u][2], s[2u * k + 1u][3])};
            for (uint32_t j = 0; j < 8u; j += 2u) {
                const uint32_t mtx = lane / 8u, r = 16u * k + lane % 8u + mtx % 2u * 8u, ch = j + mtx / 2u;
                uint32_t b0, b1, b2, b3;
                ink2_fa_ldsm4t(vt + ink2_fa_at(r, ch), b0, b1, b2, b3);
                ink2_fa_mma(acc[j], pa, b0, b1);
                ink2_fa_mma(acc[j + 1u], pa, b2, b3);
            }
        }
        __syncthreads();
    }

    for (uint32_t i = 0; i < 2u; i++) {
        sum[i] += __shfl_xor_sync(0xffffffffu, sum[i], 1);
        sum[i] += __shfl_xor_sync(0xffffffffu, sum[i], 2);
    }
    for (uint32_t i = 0; i < 2u; i++) {
        const uint32_t row = row0 + w * 16u + g + 8u * i;
        if (row >= end) { continue; }
        float *out = o + (uint64_t)(row - q0) * (INK2_FA_HEADS * INK2_FA_HEAD) + qc + 2u * t;
        const float inv = 1.0f / sum[i];
        for (uint32_t j = 0; j < 8u; j++) {
            out[j * 8u] = __bfloat162float(__float2bfloat16_rn(acc[j][2u * i] * inv));
            out[j * 8u + 1u] = __bfloat162float(__float2bfloat16_rn(acc[j][2u * i + 1u] * inv));
        }
    }
}

static inline cudaError_t ink2_vit_flash(float *o, const __nv_bfloat16 *qkv, uint32_t n, uint32_t q0, uint32_t m,
        cudaStream_t stream) {
    const dim3 grid((m + INK2_FA_BQ - 1u) / INK2_FA_BQ, INK2_FA_HEADS);
    ink2_vit_flash_kernel<<<grid, INK2_FA_THREADS, 0, stream>>>(o, qkv, n, q0, m);
    return cudaGetLastError();
}
