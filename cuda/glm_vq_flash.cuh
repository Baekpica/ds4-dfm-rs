#ifndef DS4_GLM_VQ_FLASH_CUH
#define DS4_GLM_VQ_FLASH_CUH

#include "glm_vq_ops.cuh"

/* Single-pass absorbed (latent) DSA attention. Every head scores the same
 * BF16 latent cache rows C (512 wide): s = (q W_k)·c / 16, o = Σ p c, then
 * the caller applies W_v. The expanded path built per-chunk K/V for all
 * prefix keys and a [heads, rows, keys] BF16 score tensor (≈1 GB at 8K) for a
 * separate softmax; here scores stay FP32 in registers, the softmax is online
 * and each selected key row is gathered once per query.
 *
 * One CTA per query (and key split): 8 warps = 4 head groups x 2 latent
 * halves. A warp's QK covers its half of the 512 latent dims as two quarter
 * partials; the four quarter partials meet in shared memory and add in
 * quarter order, so each warp holds the full scores of its 16 heads and
 * accumulates P·V for its 256 output columns. (Sixteen warps of 128
 * registers spilled O and Q: 34 GB of local traffic per 3,840-row call.)
 * Arithmetic change against the expanded path: scores are not rounded to BF16
 * before softmax, P rounds to BF16 for the PV MMA, O accumulates in FP32. */
enum { GLMVQ_FA_HEADS = 64, GLMVQ_FA_LATENT = 512, GLMVQ_FA_KEYS = 32,
       GLMVQ_FA_WARPS = 8, GLMVQ_FA_THREADS = GLMVQ_FA_WARPS * 32,
       GLMVQ_FA_QUARTER = GLMVQ_FA_LATENT / 4, GLMVQ_FA_HALF = GLMVQ_FA_LATENT / 2,
       GLMVQ_FA_ROW = GLMVQ_FA_LATENT * 2, GLMVQ_FA_PARTIALS = 16 };
static constexpr uint32_t GLMVQ_FA_TILE = GLMVQ_FA_KEYS * GLMVQ_FA_ROW;
// One 16 x 32 score partial per (head group, latent quarter).
static constexpr uint32_t GLMVQ_FA_EXCHANGE = GLMVQ_FA_PARTIALS * 16 * GLMVQ_FA_KEYS * sizeof(float);
static constexpr uint32_t GLMVQ_FA_SMEM = 2 * GLMVQ_FA_TILE + GLMVQ_FA_EXCHANGE +
    2 * GLMVQ_FA_KEYS * sizeof(uint32_t);
static constexpr uint32_t GLMVQ_FA_NONE = 0xffffffffu;

__device__ __forceinline__ void glmvq_fa_copy(void *dst, const void *src, uint32_t bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
        "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(bytes) : "memory");
}

// Latent rows are 64 chunks of 16 bytes; chunk ^ (key & 7) spreads the eight
// rows one ldmatrix reads over eight bank groups.
__device__ __forceinline__ uint32_t glmvq_fa_swz(uint32_t key, uint32_t chunk) {
    return key * GLMVQ_FA_ROW + ((chunk ^ (key & 7u)) << 4);
}

__device__ __forceinline__ void glmvq_fa_ldsm4(uint32_t *r, const void *p, bool trans) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    if (trans) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    }
}

__device__ __forceinline__ void glmvq_fa_mma(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t glmvq_fa_pack(float lo, float hi) {
    const __nv_bfloat162_raw v = __floats2bfloat162_rn(lo, hi);
    return (uint32_t)v.x | ((uint32_t)v.y << 16);
}

// Score slot of (row, key pair) inside a warp's 16 x 32 exchange tile; the
// XOR keeps a half-warp's eight-byte stores on sixteen distinct bank pairs.
__device__ __forceinline__ uint32_t glmvq_fa_slot(uint32_t row, uint32_t pair) {
    return row * GLMVQ_FA_KEYS + ((pair ^ ((row & 3u) << 2)) << 1);
}

/* `selected` = [rows, stride] key ids (GLMVQ_FA_NONE skips a slot) or null
 * for dense causal keys 0..position. With `partial`, CTA (row, split) covers
 * tiles split*per..: it writes unnormalized O plus per-head max and sum for
 * glmvq_fa_combine; otherwise it writes normalized BF16 rows to `low`. */
__global__ void __launch_bounds__(GLMVQ_FA_THREADS, 1) glmvq_fa_kernel(
        __nv_bfloat16 *low, float *partial, float *stats,
        const __nv_bfloat16 *qlow, const __nv_bfloat16 *cache,
        const uint32_t *positions, const uint32_t *selected, uint32_t stride,
        uint32_t tiles_per_split, float scale) {
    extern __shared__ __align__(16) uint8_t fa_shared[];
    glmvq_pdl_open();
    glmvq_pdl_wait();
    uint8_t *tiles = fa_shared;
    float *exchange = (float *)(fa_shared + 2 * GLMVQ_FA_TILE);
    uint32_t *kidx = (uint32_t *)(fa_shared + 2 * GLMVQ_FA_TILE + GLMVQ_FA_EXCHANGE);
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, q = lane & 3u, group = warp & 3u, half = warp >> 2;
    const uint32_t row = blockIdx.x, split = blockIdx.y;
    const uint32_t position = positions[row];
    const uint32_t slots = selected ? stride : position + 1;
    const uint32_t total = (slots + GLMVQ_FA_KEYS - 1) / GLMVQ_FA_KEYS;
    const uint32_t first = split * tiles_per_split;
    const uint32_t last = min(total, first + tiles_per_split);
    const uint32_t head0 = group * 16;

    // Q fragments for this warp's 16 heads and latent half, kept in registers.
    uint32_t qa[16][4];
    const __nv_bfloat16 *qrow = qlow + ((uint64_t)row * GLMVQ_FA_HEADS + head0) * GLMVQ_FA_LATENT +
        half * GLMVQ_FA_HALF;
    #pragma unroll
    for (uint32_t k = 0; k < 16; k++) {
        const uint32_t col = k * 16 + 2 * q;
        qa[k][0] = *(const uint32_t *)(qrow + (uint64_t)g * GLMVQ_FA_LATENT + col);
        qa[k][1] = *(const uint32_t *)(qrow + (uint64_t)(g + 8) * GLMVQ_FA_LATENT + col);
        qa[k][2] = *(const uint32_t *)(qrow + (uint64_t)g * GLMVQ_FA_LATENT + col + 8);
        qa[k][3] = *(const uint32_t *)(qrow + (uint64_t)(g + 8) * GLMVQ_FA_LATENT + col + 8);
    }

    // Warp w copies keys 4w..4w+3, each 1 KB row as two 512-byte warp copies:
    // every 32-byte sector is requested once (16-byte copies 64 bytes apart
    // fetched each sector twice).
    auto issue = [&](uint32_t tile, uint32_t buf) {
        uint8_t *dst = tiles + buf * GLMVQ_FA_TILE;
        #pragma unroll
        for (uint32_t i = 0; i < GLMVQ_FA_KEYS / GLMVQ_FA_WARPS; i++) {
            const uint32_t key_slot = warp * (GLMVQ_FA_KEYS / GLMVQ_FA_WARPS) + i;
            const uint32_t slot = tile * GLMVQ_FA_KEYS + key_slot;
            uint32_t key = GLMVQ_FA_NONE;
            if (slot < slots) { key = selected ? selected[(uint64_t)row * stride + slot] : slot; }
            if (key != GLMVQ_FA_NONE && key > position) { key = GLMVQ_FA_NONE; }
            if (!lane) { kidx[buf * GLMVQ_FA_KEYS + key_slot] = key; }
            const uint8_t *src = (const uint8_t *)(cache + (uint64_t)(key == GLMVQ_FA_NONE ? 0 : key) * GLMVQ_FA_LATENT);
            #pragma unroll
            for (uint32_t c = 0; c < GLMVQ_FA_ROW / 16; c += 32) {
                glmvq_fa_copy(dst + glmvq_fa_swz(key_slot, c + lane), src + (c + lane) * 16,
                    key == GLMVQ_FA_NONE ? 0 : 16);
            }
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
        const uint8_t *kt = tiles + buf * GLMVQ_FA_TILE;

        // Partial scores over this warp's two latent quarters: 16 heads x 32 keys each.
        #pragma unroll
        for (uint32_t part = 0; part < 2; part++) {
            const uint32_t quarter = 2 * half + part;
            float s[4][4] = {};
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
            float *mine = exchange + (group + 4 * quarter) * 16 * GLMVQ_FA_KEYS;
            #pragma unroll
            for (uint32_t j = 0; j < 4; j++) {
                *(float2 *)(mine + glmvq_fa_slot(g, 4 * j + q)) = make_float2(s[j][0], s[j][1]);
                *(float2 *)(mine + glmvq_fa_slot(g + 8, 4 * j + q)) = make_float2(s[j][2], s[j][3]);
            }
        }
        // The two half warps of a head group meet on their own barrier.
        asm volatile("bar.sync %0, %1;" :: "r"(1 + group), "r"(64) : "memory");
        float s[4][4] = {};
        #pragma unroll
        for (uint32_t part = 0; part < 4; part++) {
            const float *other = exchange + (group + 4 * part) * 16 * GLMVQ_FA_KEYS;
            #pragma unroll
            for (uint32_t j = 0; j < 4; j++) {
                const float2 a = *(const float2 *)(other + glmvq_fa_slot(g, 4 * j + q));
                const float2 c = *(const float2 *)(other + glmvq_fa_slot(g + 8, 4 * j + q));
                s[j][0] += a.x; s[j][1] += a.y; s[j][2] += c.x; s[j][3] += c.y;
            }
        }
        // Online softmax over valid keys; rows g and g+8 of the head group.
        const uint32_t *ids = kidx + buf * GLMVQ_FA_KEYS;
        float peak[2] = {m[0], m[1]};
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            #pragma unroll
            for (uint32_t e = 0; e < 2; e++) {
                const bool valid = ids[j * 8 + 2 * q + e] != GLMVQ_FA_NONE;
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
            float *dst = partial + (base + heads[h]) * GLMVQ_FA_LATENT + half * GLMVQ_FA_HALF + 2 * q;
            #pragma unroll
            for (uint32_t t = 0; t < 32; t++) { *(float2 *)(dst + t * 8) = make_float2(o[t][2 * h], o[t][2 * h + 1]); }
            if (!half && !q) { stats[(base + heads[h]) * 2] = m[h]; stats[(base + heads[h]) * 2 + 1] = l[h]; }
        }
        return;
    }
    #pragma unroll
    for (uint32_t h = 0; h < 2; h++) {
        const float inv = l[h] > 0.0f ? 1.0f / l[h] : 0.0f;
        __nv_bfloat16 *dst = low + ((uint64_t)row * GLMVQ_FA_HEADS + heads[h]) * GLMVQ_FA_LATENT +
            half * GLMVQ_FA_HALF + 2 * q;
        #pragma unroll
        for (uint32_t t = 0; t < 32; t++) {
            *(uint32_t *)(dst + t * 8) = glmvq_fa_pack(o[t][2 * h] * inv, o[t][2 * h + 1] * inv);
        }
    }
}

// Joins key splits: block (row, head), one thread per latent column pair.
__global__ void glmvq_fa_combine(__nv_bfloat16 *low, const float *partial, const float *stats,
        uint32_t splits) {
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
    *(uint32_t *)(low + ((uint64_t)row * GLMVQ_FA_HEADS + head) * GLMVQ_FA_LATENT + col) = glmvq_fa_pack(a * inv, b * inv);
}


/* Absorption GEMVs for 1-4 rows (decode, MTP verify). cuBLAS ran these
 * per-head products as strided-batched GEMMs at 90-215 GB/s (2.8 ms per
 * decoded token over 11 layers); each reads 16.8 MB of BF16 KV_B once for
 * all rows. FP32 accumulation, BF16 outputs as before; only the reduction
 * order differs, and it is the same for every row count. */
enum { GLMVQ_ABS_DIM = 256, GLMVQ_ABS_UNROLL = 8, GLMVQ_ABS_ROWS = 4 };

// qlow[r][h][l] = Σ_d q[r][h][d] W_k[h][d][l]: a thread owns a column pair
// and walks d, so a warp reads 128 contiguous bytes of each W_k row.
__global__ void __launch_bounds__(256) glmvq_absorb_k(__nv_bfloat16 *qlow, const float *q,
        const __nv_bfloat16 *weight, uint32_t rows) {
    const uint32_t head = blockIdx.x, col = threadIdx.x * 2;
    glmvq_pdl_open();
    // The head's W_k (256 KB) heads for L2 in four pieces while the predecessor drains.
    if (threadIdx.x < 4) {
        constexpr uint32_t piece = GLMVQ_ABS_DIM * GLMVQ_FA_LATENT * sizeof(__nv_bfloat16) / 4;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::
            "l"((const uint8_t *)(weight + (uint64_t)head * 2 * GLMVQ_ABS_DIM * GLMVQ_FA_LATENT) + threadIdx.x * piece),
            "r"(piece) : "memory");
    }
    glmvq_pdl_wait();
    const __nv_bfloat16 *w = weight + (uint64_t)head * 2 * GLMVQ_ABS_DIM * GLMVQ_FA_LATENT + col;
    float a[GLMVQ_ABS_ROWS] = {}, b[GLMVQ_ABS_ROWS] = {};
    #pragma unroll GLMVQ_ABS_UNROLL
    for (uint32_t d = 0; d < GLMVQ_ABS_DIM; d++) {
        const uint32_t pair = *(const uint32_t *)(w + (uint64_t)d * GLMVQ_FA_LATENT);
        #pragma unroll
        for (uint32_t r = 0; r < GLMVQ_ABS_ROWS; r++) {
            if (r >= rows) { continue; }
            const float v = glmvq_round(q[((uint64_t)r * GLMVQ_FA_HEADS + head) * GLMVQ_ABS_DIM + d]);
            a[r] = fmaf(__uint_as_float(pair << 16), v, a[r]);
            b[r] = fmaf(__uint_as_float(pair & 0xffff0000u), v, b[r]);
        }
    }
    for (uint32_t r = 0; r < rows; r++) {
        *(uint32_t *)(qlow + ((uint64_t)r * GLMVQ_FA_HEADS + head) * GLMVQ_FA_LATENT + col) = glmvq_fa_pack(a[r], b[r]);
    }
}

// out[r][h][c] = Σ_l low[r][h][l] W_v[h][c][l]: one warp per output column.
__global__ void __launch_bounds__(256) glmvq_absorb_v(float *out, const __nv_bfloat16 *low,
        const __nv_bfloat16 *weight, uint32_t rows) {
    const uint32_t lane = threadIdx.x & 31, row = blockIdx.x * 8 + threadIdx.x / 32;
    const uint32_t head = row / GLMVQ_ABS_DIM, c = row % GLMVQ_ABS_DIM;
    const __nv_bfloat16 *w = weight + ((uint64_t)head * 2 * GLMVQ_ABS_DIM + GLMVQ_ABS_DIM + c) * GLMVQ_FA_LATENT;
    glmvq_pdl_open();
    if (!lane) {
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::
            "l"(w), "r"((uint32_t)(GLMVQ_FA_LATENT * sizeof(__nv_bfloat16))) : "memory");
    }
    glmvq_pdl_wait();
    float sum[GLMVQ_ABS_ROWS] = {};
    #pragma unroll
    for (uint32_t i = 0; i < GLMVQ_FA_LATENT / (32 * 8); i++) {
        const uint32_t at = (i * 32 + lane) * 8;
        const uint4 wv = *(const uint4 *)(w + at);
        const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w};
        #pragma unroll
        for (uint32_t r = 0; r < GLMVQ_ABS_ROWS; r++) {
            if (r >= rows) { continue; }
            const uint4 xv = *(const uint4 *)(low + ((uint64_t)r * GLMVQ_FA_HEADS + head) * GLMVQ_FA_LATENT + at);
            const uint32_t xs[4] = {xv.x, xv.y, xv.z, xv.w};
            #pragma unroll
            for (uint32_t k = 0; k < 4; k++) {
                sum[r] = fmaf(__uint_as_float(ws[k] << 16), __uint_as_float(xs[k] << 16), sum[r]);
                sum[r] = fmaf(__uint_as_float(ws[k] & 0xffff0000u), __uint_as_float(xs[k] & 0xffff0000u), sum[r]);
            }
        }
    }
    #pragma unroll
    for (uint32_t r = 0; r < GLMVQ_ABS_ROWS; r++) {
        for (uint32_t stride = 16; stride; stride >>= 1) { sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], stride); }
        if (!lane && r < rows) { out[(uint64_t)r * GLMVQ_FA_HEADS * GLMVQ_ABS_DIM + row] = glmvq_round(sum[r]); }
    }
}

#endif
