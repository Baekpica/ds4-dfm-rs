#ifndef DS4_L31_FLASH_CUH
#define DS4_L31_FLASH_CUH

#include "glm_vq_flash.cuh"
#include "l31_latent.cuh"

/* Ling-3.1 absorbed MLA attention for decode and MTP verify rows on tensor
 * cores: GLM VQ's flash tile (8 warps = 4 groups of 16 heads x 2 latent
 * halves, 32-key BF16 tiles, online softmax, P·V over the 512 latent
 * columns) plus the 64-dim rope keys, s = (q_abs·c + q_pe·k_pe) / sqrt(192).
 * A warp sums its two latent quarters (half 0 also the rope part) before
 * the exchange, which is then half of GLM's and leaves shared memory for the
 * rope tiles. Keys are the dense causal range 0..pos0 + row; a CTA (row,
 * split) covers `per` tiles and leaves unnormalized O plus per-head max and
 * sum for l31_fa_combine. Arithmetic change against Motif's FP32 kernel: Q
 * and P round to BF16 for the MMAs; scores, softmax and O stay FP32. */
enum { L31_FA_ROPE = 64, L31_FA_KEY = 192, L31_FA_NOPE = L31_FA_KEY - L31_FA_ROPE,
       L31_FA_PE_ROW = L31_FA_ROPE * 2 };
static constexpr uint32_t L31_FA_PE_TILE = GLMVQ_FA_KEYS * L31_FA_PE_ROW;
static constexpr uint32_t L31_FA_EXCHANGE = 8 * 16 * GLMVQ_FA_KEYS * sizeof(float);
static constexpr uint32_t L31_FA_QPE = GLMVQ_FA_HEADS * L31_FA_PE_ROW;
static constexpr uint32_t L31_FA_SMEM = 2 * GLMVQ_FA_TILE + 2 * L31_FA_PE_TILE + L31_FA_EXCHANGE +
    L31_FA_QPE + 2 * GLMVQ_FA_KEYS * sizeof(uint32_t);
static_assert(L31_FA_SMEM <= 101376, "one CTA per SM");
/* FP8 latents land in the second BF16 tile's slot as two 16 KiB stages with
 * their row exponents after the key ids; each tile is converted into the
 * first slot's BF16 tile, which the BF16 kernel's MMAs then read unchanged. */
static constexpr uint32_t L31_FA8_STAGE = GLMVQ_FA_KEYS * GLMVQ_FA_LATENT;
static constexpr uint32_t L31_FA8_SMEM = L31_FA_SMEM + 2 * GLMVQ_FA_KEYS;
static_assert(2 * L31_FA8_STAGE == GLMVQ_FA_TILE && L31_FA8_SMEM <= 101376, "FP8 stages in one tile");

// Rope rows are eight 16-byte chunks; chunk ^ (row & 7) spreads an ldmatrix.
__device__ __forceinline__ uint32_t l31_fa_pe_swz(uint32_t row, uint32_t chunk) {
    return row * L31_FA_PE_ROW + ((chunk ^ (row & 7u)) << 4);
}

__device__ __forceinline__ uint32_t l31_fa_pack2(const float *x) { return glmvq_fa_pack(x[0], x[1]); }

template <bool FP8>
__global__ void __launch_bounds__(GLMVQ_FA_THREADS, 1) l31_fa_kernel(
        float *out, float *partial, float *stats, const float *q, const float *q_abs,
        const void *latent, const int8_t *exps, const __nv_bfloat16 *k_pe, uint32_t pos0,
        uint32_t tiles_per_split, float scale) {
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
    const uint32_t position = pos0 + row, slots = position + 1;
    const uint32_t total = (slots + GLMVQ_FA_KEYS - 1) / GLMVQ_FA_KEYS;
    const uint32_t first = split * tiles_per_split;
    const uint32_t last = min(total, first + tiles_per_split);
    const uint32_t head0 = group * 16;

    // q_pe of all 64 heads as BF16 rows for the rope A fragments.
    for (uint32_t i = tid; i < GLMVQ_FA_HEADS * (L31_FA_ROPE / 8); i += GLMVQ_FA_THREADS) {
        const uint32_t head = i / (L31_FA_ROPE / 8), chunk = i % (L31_FA_ROPE / 8);
        const float *src = q + ((uint64_t)row * GLMVQ_FA_HEADS + head) * L31_FA_KEY + L31_FA_NOPE + chunk * 8;
        const uint4 v = make_uint4(l31_fa_pack2(src), l31_fa_pack2(src + 2), l31_fa_pack2(src + 4), l31_fa_pack2(src + 6));
        *(uint4 *)(qpe + l31_fa_pe_swz(head, chunk)) = v;
    }
    // q_abs fragments for this warp's 16 heads and latent half, in registers.
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

    // Warp w copies keys 4w..4w+3: BF16 latent rows as two 512-byte warp
    // copies (an FP8 row as one, unswizzled into its stage), rope rows by
    // lanes 0-7 (eight 16-byte chunks each).
    auto issue = [&](uint32_t tile, uint32_t buf) {
        uint8_t *dst = FP8 ? stages + buf * L31_FA8_STAGE : tiles + buf * GLMVQ_FA_TILE;
        uint8_t *pe = pe_tiles + buf * L31_FA_PE_TILE;
        #pragma unroll
        for (uint32_t i = 0; i < GLMVQ_FA_KEYS / GLMVQ_FA_WARPS; i++) {
            const uint32_t key_slot = warp * (GLMVQ_FA_KEYS / GLMVQ_FA_WARPS) + i;
            const uint32_t slot = tile * GLMVQ_FA_KEYS + key_slot;
            const uint32_t key = slot < slots ? slot : GLMVQ_FA_NONE;
            if (!lane) { kidx[buf * GLMVQ_FA_KEYS + key_slot] = key; }
            const uint32_t bytes = key == GLMVQ_FA_NONE ? 0 : 16;
            const uint64_t at = key == GLMVQ_FA_NONE ? 0 : key;
            if (FP8) {
                glmvq_fa_copy(dst + key_slot * GLMVQ_FA_LATENT + lane * 16,
                    (const uint8_t *)latent + at * GLMVQ_FA_LATENT + lane * 16, bytes);
            } else {
                const uint8_t *src = (const uint8_t *)latent + at * GLMVQ_FA_ROW;
                #pragma unroll
                for (uint32_t c = 0; c < GLMVQ_FA_ROW / 16; c += 32) {
                    glmvq_fa_copy(dst + glmvq_fa_swz(key_slot, c + lane), src + (c + lane) * 16, bytes);
                }
            }
            if (lane < L31_FA_PE_ROW / 16) {
                glmvq_fa_copy(pe + l31_fa_pe_swz(key_slot, lane), (const uint8_t *)(k_pe + at * L31_FA_ROPE) + lane * 16, bytes);
            }
        }
        // The tile's row exponents. Masked rows are zero-filled and every
        // stored exponent is finite (the cache starts zeroed), so they stay 0.
        if (FP8 && tid < GLMVQ_FA_KEYS / 16) {
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
        const uint8_t *kt = FP8 ? tiles : tiles + buf * GLMVQ_FA_TILE;
        const uint8_t *pt = pe_tiles + buf * L31_FA_PE_TILE;
        if (FP8) {
            // The stage's 32 rows x 32 chunks of 16 E4M3 bytes into the BF16
            // tile; every warp finished the previous tile at the barrier above.
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

        // This warp's partial scores (16 heads x 32 keys): its two latent
        // quarters, and for half 0 the rope dims.
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
                // A: heads 0-7 / 8-15 x dims 0-7 / 8-15 of this k step.
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
        // The two half warps of a head group meet on their own barrier; two
        // partials add commutatively, so both hold the same scores.
        asm volatile("bar.sync %0, %1;" :: "r"(1 + group), "r"(64) : "memory");
        const float *other = exchange + (group + 4 * (half ^ 1u)) * 16 * GLMVQ_FA_KEYS;
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            const float2 a = *(const float2 *)(other + glmvq_fa_slot(g, 4 * j + q4));
            const float2 c = *(const float2 *)(other + glmvq_fa_slot(g + 8, 4 * j + q4));
            s[j][0] += a.x; s[j][1] += a.y; s[j][2] += c.x; s[j][3] += c.y;
        }
        // Online softmax over valid keys; rows g and g+8 of the head group.
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
        // P (16 heads x 32 keys) as two k16 A fragments times V's 256 columns.
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
        float *dst = out + ((uint64_t)row * GLMVQ_FA_HEADS + heads[h]) * GLMVQ_FA_LATENT + half * GLMVQ_FA_HALF + 2 * q4;
        #pragma unroll
        for (uint32_t t = 0; t < 32; t++) { *(float2 *)(dst + t * 8) = make_float2(o[t][2 * h] * inv, o[t][2 * h + 1] * inv); }
    }
}

// glmvq_fa_combine with F32 latent rows for Ling's value projection.
__global__ void l31_fa_combine(float *out, const float *partial, const float *stats, uint32_t splits) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t row = blockIdx.x, head = blockIdx.y, col = threadIdx.x * 2;
    float peak = -INFINITY;
    for (uint32_t s = 0; s < splits; s++) {
        peak = fmaxf(peak, stats[(((uint64_t)row * splits + s) * GLMVQ_FA_HEADS + head) * 2]);
    }
    const float ref = peak == -INFINITY ? 0.0f : peak;
    float sum = 0.0f, a = 0.0f, b = 0.0f;
    for (uint32_t s = 0; s < splits; s++) {
        const uint64_t at = ((uint64_t)row * splits + s) * GLMVQ_FA_HEADS + head;
        const float w = exp2f(stats[at * 2] - ref);
        sum += w * stats[at * 2 + 1];
        const float2 v = *(const float2 *)(partial + at * GLMVQ_FA_LATENT + col);
        a += w * v.x; b += w * v.y;
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    *(float2 *)(out + ((uint64_t)row * GLMVQ_FA_HEADS + head) * GLMVQ_FA_LATENT + col) = make_float2(a * inv, b * inv);
}

/* Decode and verify rows (1-4) split their keys over GLM's 48 CTAs. Scratch
 * holds every split's unnormalized O and its per-head max and sum. FP8
 * latents have no Motif fallback: short appends below the expanded path's
 * 64 rows take one CTA per row. */
enum { L31_FA_ROWS = 4, L31_FA8_ROWS = 63, L31_FA_SPLITS = 48 };

static uint64_t l31_fa_scratch_bytes(void) {
    return (uint64_t)L31_FA_SPLITS * GLMVQ_FA_HEADS * (GLMVQ_FA_LATENT + 2) * sizeof(float);
}

// `exps` (the FP8 cache's row exponents) selects the FP8 kernel.
static cudaError_t l31_fa_launch(float *out, void *scratch, const float *q, const float *q_abs,
        const void *latent, const int8_t *exps, const __nv_bfloat16 *k_pe, uint32_t rows, uint32_t pos0,
        float scale, glmvq_launch mode, cudaStream_t stream) {
    if (!rows || rows > (exps ? L31_FA8_ROWS : L31_FA_ROWS)) { return cudaErrorInvalidValue; }
    const uint32_t tiles = (pos0 + rows + GLMVQ_FA_KEYS - 1) / GLMVQ_FA_KEYS;
    const uint32_t splits = max(1u, min(tiles, (uint32_t)L31_FA_SPLITS / rows));
    const uint32_t per = (tiles + splits - 1) / splits;
    float *partial = (float *)scratch;
    float *stats = partial + (uint64_t)rows * splits * GLMVQ_FA_HEADS * GLMVQ_FA_LATENT;
    cudaError_t err = exps
        ? glmvq_launch_kernel(mode, l31_fa_kernel<true>, dim3(rows, splits), GLMVQ_FA_THREADS, L31_FA8_SMEM,
            stream, out, splits > 1 ? partial : nullptr, stats, q, q_abs, latent, exps, k_pe, pos0,
            splits > 1 ? per : UINT32_MAX, scale)
        : glmvq_launch_kernel(mode, l31_fa_kernel<false>, dim3(rows, splits), GLMVQ_FA_THREADS, L31_FA_SMEM,
            stream, out, splits > 1 ? partial : nullptr, stats, q, q_abs, latent, exps, k_pe, pos0,
            splits > 1 ? per : UINT32_MAX, scale);
    if (err == cudaSuccess && splits > 1) {
        err = glmvq_launch_kernel(mode, l31_fa_combine, dim3(rows, GLMVQ_FA_HEADS), GLMVQ_FA_LATENT / 2, 0,
            stream, out, (const float *)partial, (const float *)stats, splits);
    }
    return err;
}

#endif
