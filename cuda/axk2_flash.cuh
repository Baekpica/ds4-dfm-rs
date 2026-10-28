#ifndef DS4_AXK2_FLASH_CUH
#define DS4_AXK2_FLASH_CUH

#include "../ds4_axk2.h"
#include "l31_flash.cuh"

/* Ling-3.1's FP8 flash tile (l31_fa_kernel<true>) over either the dense
 * causal keys 0..position or the row's DSA selection: slot i of a tile is
 * selected[row][i], skipped when it lies above the row's position (a row
 * that sees fewer than 2,048 keys selects every one of them plus masked
 * fillers). Sparse tiles gather each selected row's exponent. */
enum class ax_fa_keys { serial, cooperative };

template <typename T, ax_fa_keys KEYS = ax_fa_keys::serial>
__global__ void __launch_bounds__(GLMVQ_FA_THREADS, 1) ax_fa_kernel(
        T *out, float *partial, float *stats, const float *q, const float *q_abs,
        const void *latent, const int8_t *exps, const __nv_bfloat16 *k_pe, const uint32_t *selected,
        uint32_t pos0, uint32_t tiles_per_split, float scale) {
    extern __shared__ __align__(16) uint8_t fa_shared[];
    glmvq_pdl_open();
    glmvq_pdl_wait();
    uint8_t *tiles = fa_shared;
    uint8_t *stages = fa_shared + GLMVQ_FA_TILE;
    uint8_t *pe_tiles = fa_shared + 2 * GLMVQ_FA_TILE;
    float *exchange = (float *)(pe_tiles + 2 * L31_FA_PE_TILE);
    uint8_t *qpe = (uint8_t *)exchange + L31_FA_EXCHANGE;
    uint32_t *kidx = (uint32_t *)(qpe + L31_FA_QPE);
    int8_t *kexp = (int8_t *)(kidx + 2 * GLMVQ_FA_KEYS);
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, q4 = lane & 3u, group = warp & 3u, half = warp >> 2;
    const uint32_t row = blockIdx.x, split = blockIdx.y;
    const uint32_t position = pos0 + row, slots = selected ? (uint32_t)DS4_AXK2_TOPK : position + 1;
    const uint32_t total = (slots + GLMVQ_FA_KEYS - 1) / GLMVQ_FA_KEYS;
    const uint32_t first = split * tiles_per_split;
    const uint32_t last = min(total, first + tiles_per_split);
    const uint32_t head0 = group * 16;
    const uint32_t *sel = selected ? selected + (uint64_t)row * DS4_AXK2_TOPK : nullptr;

    for (uint32_t i = tid; i < GLMVQ_FA_HEADS * (L31_FA_ROPE / 8); i += GLMVQ_FA_THREADS) {
        const uint32_t head = i / (L31_FA_ROPE / 8), chunk = i % (L31_FA_ROPE / 8);
        const float *src = q + ((uint64_t)row * GLMVQ_FA_HEADS + head) * L31_FA_KEY + L31_FA_NOPE + chunk * 8;
        const uint4 v = make_uint4(l31_fa_pack2(src), l31_fa_pack2(src + 2), l31_fa_pack2(src + 4), l31_fa_pack2(src + 6));
        *(uint4 *)(qpe + l31_fa_pe_swz(head, chunk)) = v;
    }
    uint32_t qa[16][4];
    const float *qrow = q_abs + ((uint64_t)row * GLMVQ_FA_HEADS + head0) * GLMVQ_FA_LATENT + half * GLMVQ_FA_HALF;
    #pragma unroll
    for (uint32_t k = 0; k < 16; k++) {
        const uint32_t col = k * 16 + 2 * q4;
        qa[k][0] = l31_fa_pack2(qrow + (uint64_t)g * GLMVQ_FA_LATENT + col);
        qa[k][1] = l31_fa_pack2(qrow + (uint64_t)(g + 8) * GLMVQ_FA_LATENT + col);
        qa[k][2] = l31_fa_pack2(qrow + (uint64_t)g * GLMVQ_FA_LATENT + col + 8);
        qa[k][3] = l31_fa_pack2(qrow + (uint64_t)(g + 8) * GLMVQ_FA_LATENT + col + 8);
    }

    // Warp w copies keys 4w..4w+3: an FP8 latent row as one 512-byte warp
    // copy into its stage, rope rows by lanes 0-7, the exponent by lane 0.
    auto issue = [&](uint32_t tile, uint32_t buf) {
        uint8_t *dst = stages + buf * L31_FA8_STAGE;
        uint8_t *pe = pe_tiles + buf * L31_FA_PE_TILE;
        constexpr uint32_t per_warp = GLMVQ_FA_KEYS / GLMVQ_FA_WARPS;
        uint32_t my_key = GLMVQ_FA_NONE;
        if constexpr (KEYS == ax_fa_keys::cooperative) {
            // Four lanes gather independent metadata before the four row copies.
            const uint32_t key_slot = warp * per_warp + lane;
            const uint32_t slot = tile * GLMVQ_FA_KEYS + key_slot;
            if (lane < per_warp) {
                if (slot < slots) { my_key = sel ? sel[slot] : slot; }
                if (my_key > position) { my_key = GLMVQ_FA_NONE; }
                kidx[buf * GLMVQ_FA_KEYS + key_slot] = my_key;
                if (sel) { kexp[buf * GLMVQ_FA_KEYS + key_slot] = my_key == GLMVQ_FA_NONE ? 0 : exps[my_key]; }
            }
        }
        #pragma unroll
        for (uint32_t i = 0; i < per_warp; i++) {
            const uint32_t key_slot = warp * per_warp + i;
            const uint32_t slot = tile * GLMVQ_FA_KEYS + key_slot;
            uint32_t key = GLMVQ_FA_NONE;
            if constexpr (KEYS == ax_fa_keys::cooperative) {
                key = __shfl_sync(0xffffffffu, my_key, i);
            } else {
                if (slot < slots) { key = sel ? sel[slot] : slot; }
                if (key > position) { key = GLMVQ_FA_NONE; }
                if (!lane) {
                    kidx[buf * GLMVQ_FA_KEYS + key_slot] = key;
                    if (sel) { kexp[buf * GLMVQ_FA_KEYS + key_slot] = key == GLMVQ_FA_NONE ? 0 : exps[key]; }
                }
            }
            const uint32_t bytes = key == GLMVQ_FA_NONE ? 0 : 16;
            const uint64_t at = key == GLMVQ_FA_NONE ? 0 : key;
            glmvq_fa_copy(dst + key_slot * GLMVQ_FA_LATENT + lane * 16,
                (const uint8_t *)latent + at * GLMVQ_FA_LATENT + lane * 16, bytes);
            if (lane < L31_FA_PE_ROW / 16) {
                glmvq_fa_copy(pe + l31_fa_pe_swz(key_slot, lane), (const uint8_t *)(k_pe + at * L31_FA_ROPE) + lane * 16,
                    bytes);
            }
        }
        // Dense tiles copy their 32 row exponents at once (the cache starts
        // zeroed, so rows past the live keys hold finite exponents).
        if (!sel && tid < GLMVQ_FA_KEYS / 16) {
            glmvq_fa_copy(kexp + buf * GLMVQ_FA_KEYS + tid * 16, exps + (uint64_t)tile * GLMVQ_FA_KEYS + tid * 16, 16);
        }
    };

    float o[32][4] = {};
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    const float log2_scale = scale * 1.4426950408889634f;
    if (first < last) { issue(first, 0); }
    asm volatile("cp.async.commit_group;" ::: "memory");
    for (uint32_t tile = first; tile < last; tile++) {
        const uint32_t buf = (tile - first) & 1u;
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (tile + 1 < last) { issue(tile + 1, buf ^ 1u); }
        asm volatile("cp.async.commit_group;" ::: "memory");
        const uint8_t *kt = tiles;
        const uint8_t *pt = pe_tiles + buf * L31_FA_PE_TILE;
        {
            const uint8_t *stage = stages + buf * L31_FA8_STAGE;
            #pragma unroll
            for (uint32_t i = 0; i < GLMVQ_FA_KEYS * GLMVQ_FA_LATENT / 16 / GLMVQ_FA_THREADS; i++) {
                const uint32_t id = i * GLMVQ_FA_THREADS + tid, key = id / 32, c = id % 32;
                uint4 lo, hi;
                l31_latent_bf16x16(*(const uint4 *)(stage + key * GLMVQ_FA_LATENT + c * 16),
                    l31_pow2(kexp[buf * GLMVQ_FA_KEYS + key]), &lo, &hi);
                *(uint4 *)(tiles + glmvq_fa_swz(key, 2 * c)) = lo;
                *(uint4 *)(tiles + glmvq_fa_swz(key, 2 * c + 1)) = hi;
            }
            __syncthreads();
        }

        float s[4][4] = {};
        #pragma unroll
        for (uint32_t part = 0; part < 2; part++) {
            const uint32_t quarter = 2 * half + part;
            #pragma unroll
            for (uint32_t k = 0; k < 8; k++) {
                #pragma unroll
                for (uint32_t j = 0; j < 4; j += 2) {
                    uint32_t b[4];
                    const uint32_t mat = lane >> 3;
                    const uint32_t key = (j + (mat >> 1)) * 8 + (lane & 7u);
                    const uint32_t chunk = quarter * 16 + k * 2 + (mat & 1u);
                    glmvq_fa_ldsm4(b, kt + glmvq_fa_swz(key, chunk), false);
                    glmvq_fa_mma(s[j], qa[part * 8 + k], b[0], b[1]);
                    glmvq_fa_mma(s[j + 1], qa[part * 8 + k], b[2], b[3]);
                }
            }
        }
        if (!half) {
            #pragma unroll
            for (uint32_t k = 0; k < L31_FA_ROPE / 16; k++) {
                uint32_t a[4];
                const uint32_t mat = lane >> 3;
                glmvq_fa_ldsm4(a, qpe + l31_fa_pe_swz(head0 + (mat & 1u) * 8 + (lane & 7u), k * 2 + (mat >> 1)), false);
                #pragma unroll
                for (uint32_t j = 0; j < 4; j += 2) {
                    uint32_t b[4];
                    const uint32_t key = (j + (mat >> 1)) * 8 + (lane & 7u);
                    glmvq_fa_ldsm4(b, pt + l31_fa_pe_swz(key, k * 2 + (mat & 1u)), false);
                    glmvq_fa_mma(s[j], a, b[0], b[1]);
                    glmvq_fa_mma(s[j + 1], a, b[2], b[3]);
                }
            }
        }
        float *mine = exchange + (group + 4 * half) * 16 * GLMVQ_FA_KEYS;
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            *(float2 *)(mine + glmvq_fa_slot(g, 4 * j + q4)) = make_float2(s[j][0], s[j][1]);
            *(float2 *)(mine + glmvq_fa_slot(g + 8, 4 * j + q4)) = make_float2(s[j][2], s[j][3]);
        }
        asm volatile("bar.sync %0, %1;" :: "r"(1 + group), "r"(64) : "memory");
        const float *other = exchange + (group + 4 * (half ^ 1u)) * 16 * GLMVQ_FA_KEYS;
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            const float2 a = *(const float2 *)(other + glmvq_fa_slot(g, 4 * j + q4));
            const float2 c = *(const float2 *)(other + glmvq_fa_slot(g + 8, 4 * j + q4));
            s[j][0] += a.x; s[j][1] += a.y; s[j][2] += c.x; s[j][3] += c.y;
        }
        const uint32_t *ids = kidx + buf * GLMVQ_FA_KEYS;
        float peak[2] = {m[0], m[1]};
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            #pragma unroll
            for (uint32_t e = 0; e < 2; e++) {
                const bool valid = ids[j * 8 + 2 * q4 + e] != GLMVQ_FA_NONE;
                s[j][e] = valid ? s[j][e] * log2_scale : -INFINITY;
                s[j][e + 2] = valid ? s[j][e + 2] * log2_scale : -INFINITY;
                peak[0] = fmaxf(peak[0], s[j][e]);
                peak[1] = fmaxf(peak[1], s[j][e + 2]);
            }
        }
        #pragma unroll
        for (uint32_t h = 0; h < 2; h++) {
            peak[h] = fmaxf(peak[h], __shfl_xor_sync(0xffffffffu, peak[h], 1));
            peak[h] = fmaxf(peak[h], __shfl_xor_sync(0xffffffffu, peak[h], 2));
        }
        float alpha[2], sum[2] = {0.0f, 0.0f};
        #pragma unroll
        for (uint32_t h = 0; h < 2; h++) {
            const float ref = peak[h] == -INFINITY ? 0.0f : peak[h];
            alpha[h] = exp2f(m[h] - ref);
            m[h] = peak[h];
            #pragma unroll
            for (uint32_t j = 0; j < 4; j++) {
                s[j][2 * h] = exp2f(s[j][2 * h] - ref);
                s[j][2 * h + 1] = exp2f(s[j][2 * h + 1] - ref);
                sum[h] += s[j][2 * h] + s[j][2 * h + 1];
            }
        }
        #pragma unroll
        for (uint32_t h = 0; h < 2; h++) {
            sum[h] += __shfl_xor_sync(0xffffffffu, sum[h], 1);
            sum[h] += __shfl_xor_sync(0xffffffffu, sum[h], 2);
            l[h] = l[h] * alpha[h] + sum[h];
        }
        #pragma unroll
        for (uint32_t t = 0; t < 32; t++) {
            o[t][0] *= alpha[0]; o[t][1] *= alpha[0]; o[t][2] *= alpha[1]; o[t][3] *= alpha[1];
        }
        #pragma unroll
        for (uint32_t kk = 0; kk < 2; kk++) {
            const uint32_t pa[4] = {glmvq_fa_pack(s[2 * kk][0], s[2 * kk][1]),
                glmvq_fa_pack(s[2 * kk][2], s[2 * kk][3]),
                glmvq_fa_pack(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                glmvq_fa_pack(s[2 * kk + 1][2], s[2 * kk + 1][3])};
            #pragma unroll
            for (uint32_t t = 0; t < 32; t += 2) {
                uint32_t b[4];
                const uint32_t mat = lane >> 3;
                const uint32_t key = kk * 16 + (mat & 1u) * 8 + (lane & 7u);
                const uint32_t chunk = half * 32 + t + (mat >> 1);
                glmvq_fa_ldsm4(b, kt + glmvq_fa_swz(key, chunk), true);
                glmvq_fa_mma(o[t], pa, b[0], b[1]);
                glmvq_fa_mma(o[t + 1], pa, b[2], b[3]);
            }
        }
    }

    const uint32_t heads[2] = {head0 + g, head0 + g + 8};
    if (partial) {
        const uint64_t base = ((uint64_t)row * gridDim.y + split) * GLMVQ_FA_HEADS;
        #pragma unroll
        for (uint32_t h = 0; h < 2; h++) {
            float *dst = partial + (base + heads[h]) * GLMVQ_FA_LATENT + half * GLMVQ_FA_HALF + 2 * q4;
            #pragma unroll
            for (uint32_t t = 0; t < 32; t++) { *(float2 *)(dst + t * 8) = make_float2(o[t][2 * h], o[t][2 * h + 1]); }
            if (!half && !q4) { stats[(base + heads[h]) * 2] = m[h]; stats[(base + heads[h]) * 2 + 1] = l[h]; }
        }
        return;
    }
    #pragma unroll
    for (uint32_t h = 0; h < 2; h++) {
        const float inv = l[h] > 0.0f ? 1.0f / l[h] : 0.0f;
        T *dst = out + ((uint64_t)row * GLMVQ_FA_HEADS + heads[h]) * GLMVQ_FA_LATENT + half * GLMVQ_FA_HALF + 2 * q4;
        #pragma unroll
        for (uint32_t t = 0; t < 32; t++) {
            if constexpr (sizeof(T) == sizeof(float)) {
                *(float2 *)(dst + t * 8) = make_float2(o[t][2 * h] * inv, o[t][2 * h + 1] * inv);
            } else {
                *(__nv_bfloat162 *)(dst + t * 8) = __floats2bfloat162_rn(o[t][2 * h] * inv, o[t][2 * h + 1] * inv);
            }
        }
    }
}

#endif
