/* ds41_attn.cuh — DeepSeek V4.1 (ds41) attention primitives, ported from the C
 * engine at /data/YoungAi (commit 3946dbc):
 *   - RoPE: src/cuda/cuda_v41_2.inc.cu:8-43 (official precompute_freqs_cis +
 *     apply_rotary_emb; adjacent element pairs are one complex number, YaRN
 *     ramp for the compressor layers);
 *   - the compressor pool (:49-70, official Compressor.forward ratio>1);
 *   - the scalar sparse attention (:132-303, official sparse_attn_kernel
 *     semantics: window rows in ascending order then the topk compressed
 *     rows, online softmax per 8-key tile, p rounded to bf16 before the v
 *     multiply, sink in the denominator only);
 *   - act_quant / the KV pack family and the pack constants:
 *     src/cuda/cuda_kv_pack.inc.cu (official act_quant_kernel inplace branch;
 *     the 288 B / 72 B group formats);
 *   - the SWA window ring commit: src/cuda/cuda_kv_ring.inc.cu:20-56.
 *   - v41_pow2_ceil_log2 and v41_win_row: src/cuda/cuda_v41_1.inc.cu:130-151.
 *
 * The sparse-attention tensor-core family (decode seg, prefill, split-K,
 * merge) and the ds4_gpu_v41_sparse_attn_tensor entry live in
 * cuda/ds41_attn_mma.cuh (P4-5); this file keeps the scalar kernel that the
 * entry's dispatch falls back to (the engine's own fallback for the 9..63
 * token band and refused shapes). */
#pragma once

/* Packed-KV group geometry (ds4_gpu_v41.h:229-234): the main KV stores 512
 * dims as 256 B nibbles + 32 B E4M3 scales = 288 B/group; the index key
 * stores 128 dims as 64 B nibbles + 4 B E8M0 scales = 68 B padded to 72 (8
 * aligned: 68 makes a warp's 64 B read straddle two 32 B sectors).  The
 * constants live in ds41_kvfmt.h so the host allocator uses the same copy. */
#include "../ds41_kvfmt.h"

/* fast_round_scale's exponent: 2^ceil(log2 v) (cuda_v41_1.inc.cu:130-133).
 * act_quant and the KV pack share this one copy — a drift here would make the
 * packed value differ from the f32 it replaces. */
__device__ __forceinline__ static float v41_pow2_ceil_log2(float v) {
    int e; const float m = frexpf(v, &e);      /* v = m*2^e, m in [0.5,1) => log2 v in (e-1, e] */
    return ldexpf(1.0f, (m == 0.5f) ? e - 1 : e);
}

/* Which row of the SWA window buffer absolute position a lives in
 * (cuda_v41_1.inc.cu:148-151): [0, window) is the history segment, [window,
 * window+n) this batch's rows.  ring=1 (main path) keeps a % window; ring=0
 * (the DSpark tower) is the linear segment push_main shifts. */
__device__ __forceinline__ static uint64_t v41_win_row(int64_t a, uint32_t pos0, uint32_t window, uint32_t ring) {
    if (a >= (int64_t)pos0) return (uint64_t)((int64_t)window + a - (int64_t)pos0);
    return ring ? (uint64_t)(a % (int64_t)window)
                : (uint64_t)(a - ((int64_t)pos0 - (int64_t)window));
}

/* ---- RoPE (cuda_v41_2.inc.cu:8-43) ---- */
__device__ __forceinline__ static float v41_rope_freq(uint32_t i, uint32_t dim, float theta, uint32_t osl,
                                                      float factor, float beta_fast, float beta_slow) {
    float f = 1.0f / powf(theta, (float)(2u * i) / (float)dim);
    if (osl > 0u) {   /* YaRN ramp: corrected_dim(rot) = dim*ln(osl/(rot*2pi)) / (2 ln theta) */
        const float lt = 2.0f * logf(theta);
        float low = floorf((float)dim * logf((float)osl / (beta_fast * 2.0f * (float)M_PI)) / lt);
        float high = ceilf((float)dim * logf((float)osl / (beta_slow * 2.0f * (float)M_PI)) / lt);
        low = fmaxf(low, 0.0f); high = fminf(high, (float)(dim - 1u));
        float ramp = ((float)i - low) / fmaxf(high - low, 1e-3f);
        ramp = fminf(fmaxf(ramp, 0.0f), 1.0f);
        const float smooth = 1.0f - ramp;
        f = f / factor * (1.0f - smooth) + f * smooth;
    }
    return f;
}
__global__ static void v41_rope_kernel(float *x, const int32_t *pos, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
                                       float theta, uint32_t osl, float factor, float bf, float bs, int inverse) {
    v41_pdl_wait();
    const uint32_t t = blockIdx.y, h = blockIdx.x;
    const uint32_t i = threadIdx.x;                 /* complex index i < n_rot/2 */
    if (i >= n_rot / 2u) return;
    float *xr = x + ((uint64_t)t * n_head + h) * head_dim + (head_dim - n_rot) + 2u * i;
    const float ang = (float)pos[t] * v41_rope_freq(i, n_rot, theta, osl, factor, bf, bs);
    float c = cosf(ang), s = sinf(ang);
    if (inverse) s = -s;
    const float a = xr[0], b = xr[1];
    xr[0] = v41_bf16r(a * c - b * s);   /* official: x.float() rotated, copy_ back into the bf16 tensor */
    xr[1] = v41_bf16r(a * s + b * c);
}
extern "C" int ds4_gpu_v41_rope_tensor(ds4_gpu_tensor *x, const ds4_gpu_tensor *pos, uint32_t n_tok, uint32_t n_head,
                                       uint32_t head_dim, uint32_t n_rot, float theta, uint32_t original_seq_len,
                                       float factor, float beta_fast, float beta_slow, bool inverse) {
    if (!x || !pos || (n_rot & 1u) || n_rot > 128u) return 0;
    v41_rope_kernel<<<dim3(n_head, n_tok), 64, 0, ds4_current_stream()>>>((float *)x->ptr, (const int32_t *)pos->ptr, n_head, head_dim, n_rot,
                                                                           theta, original_seq_len, factor, beta_fast, beta_slow, inverse ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "v41 rope");
}

/* ---- compressor pool (cuda_v41_2.inc.cu:49-70): per-dimension softmax
 * weighting inside the group -> bf16 ---- */
__global__ static void v41_compress_pool_kernel(float *out, const float *kv, const float *sc, uint32_t ratio, uint32_t dim) {
    const uint32_t g = blockIdx.x;
    for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
        float mx = -INFINITY;
        for (uint32_t t = 0; t < ratio; t++) mx = fmaxf(mx, sc[((uint64_t)g * ratio + t) * dim + d]);
        float den = 0.f, acc = 0.f;
        for (uint32_t t = 0; t < ratio; t++) {
            const float e = expf(sc[((uint64_t)g * ratio + t) * dim + d] - mx);
            den += e; acc += e * kv[((uint64_t)g * ratio + t) * dim + d];
        }
        out[(uint64_t)g * dim + d] = v41_bf16r(acc / den);
    }
}
extern "C" int ds4_gpu_v41_compress_pool_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *score,
                                                uint32_t n_tok, uint32_t ratio, uint32_t dim) {
    if (!out || !kv || !score || ratio == 0) return 0;
    const uint32_t ng = n_tok / ratio;
    if (ng == 0) return 1;
    v41_compress_pool_kernel<<<ng, 256, 0, ds4_current_stream()>>>((float *)out->ptr, (const float *)kv->ptr, (const float *)score->ptr, ratio, dim);
    return cuda_ok(cudaGetLastError(), "v41 compress pool");
}

/* ---- compressor source, one decode step, n rows (cuda_v41_2.inc.cu:74-128) ----
 * The graph route's replacement for the direct path's three host-shaped steps
 * (host-offset memcpy of the batch rows after the pending rows, a host
 * decision to launch the pool, host-offset memcpy of the remainder to the
 * head): one launch whose shape does not depend on the position, so a single
 * captured graph replays every step.  The position comes from the device slot
 * (posd), the pending slot is pos0 % ratio, and every completed group pools
 * with the pool kernel's exact arithmetic (same t order, same mx/den/acc,
 * same bf16 round) -- the direct and graph routes are bit-identical.
 * snap_kv/snap_sc non-NULL store [old pending rows | this batch's n rows]
 * linearly (the direct path's snap_cpre layout, so v41_spec_rollback reads it
 * unchanged); a pure-decode n=1 capture passes NULL. */
__global__ static void v41_compress_step_n_kernel(float *pooled, int32_t *posg, float *ckv_c, float *csc_c, float *snap_kv, float *snap_sc,
                                                  const float *ckv, const float *csc, const int32_t *posd, uint32_t ratio, uint32_t dim, uint32_t n) {
    v41_pdl_wait();
    const uint32_t pos0 = (uint32_t)posd[0], pend = pos0 % ratio;
    if (snap_kv)
        for (uint32_t r = 0; r < pend; r++)
            for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) { snap_kv[(uint64_t)r * dim + d] = ckv_c[(uint64_t)r * dim + d]; snap_sc[(uint64_t)r * dim + d] = csc_c[(uint64_t)r * dim + d]; }
    uint32_t j = 0;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t slot = (pend + i) % ratio;
        __syncthreads();   /* the previous group's pool read slots 0..ratio-1 before the next row may overwrite slot 0 */
        for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
            const float a = ckv[(uint64_t)i * dim + d], b = csc[(uint64_t)i * dim + d];
            ckv_c[(uint64_t)slot * dim + d] = a; csc_c[(uint64_t)slot * dim + d] = b;
            if (snap_kv) { snap_kv[(uint64_t)(pend + i) * dim + d] = a; snap_sc[(uint64_t)(pend + i) * dim + d] = b; }
        }
        if (slot + 1u != ratio) continue;   /* not a complete group yet: append only (slot is block-uniform, no divergence) */
        __syncthreads();
        if (threadIdx.x == 0) posg[j] = (int32_t)(pos0 + i + 1u - ratio);
        for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
            float mx = -INFINITY;
            for (uint32_t t = 0; t < ratio; t++) mx = fmaxf(mx, csc_c[(uint64_t)t * dim + d]);
            float den = 0.f, acc = 0.f;
            for (uint32_t t = 0; t < ratio; t++) {
                const float e = expf(csc_c[(uint64_t)t * dim + d] - mx);
                den += e; acc += e * ckv_c[(uint64_t)t * dim + d];
            }
            pooled[(uint64_t)j * dim + d] = v41_bf16r(acc / den);
        }
        j++;
    }
}
extern "C" int ds4_gpu_v41_compress_step_n_tensor(ds4_gpu_tensor *pooled, ds4_gpu_tensor *posg, ds4_gpu_tensor *cpre_kv, ds4_gpu_tensor *cpre_sc,
                                                  ds4_gpu_tensor *snap_kv, ds4_gpu_tensor *snap_sc, const ds4_gpu_tensor *ckv, const ds4_gpu_tensor *csc,
                                                  const ds4_gpu_tensor *posd, uint32_t ratio, uint32_t dim, uint32_t n) {
    if (!pooled || !posg || !cpre_kv || !cpre_sc || !ckv || !csc || !posd || ratio < 2u || n == 0u || n > 8u) return 0;
    const uint32_t ngmax = (ratio - 1u + n) / ratio;
    if (cpre_kv->bytes < (uint64_t)ratio * dim * 4 || cpre_sc->bytes < (uint64_t)ratio * dim * 4) return 0;
    if (pooled->bytes < (uint64_t)ngmax * dim * 4 || posg->bytes < (uint64_t)ngmax * 4 || ckv->bytes < (uint64_t)n * dim * 4) return 0;
    if ((snap_kv != NULL) != (snap_sc != NULL)) return 0;
    if (snap_kv && (snap_kv->bytes < (uint64_t)(ratio - 1u + n) * dim * 4 || snap_sc->bytes < (uint64_t)(ratio - 1u + n) * dim * 4)) return 0;
    v41_compress_step_n_kernel<<<1, 256, 0, ds4_current_stream()>>>((float *)pooled->ptr, (int32_t *)posg->ptr, (float *)cpre_kv->ptr, (float *)cpre_sc->ptr,
        snap_kv ? (float *)snap_kv->ptr : NULL, snap_sc ? (float *)snap_sc->ptr : NULL,
        (const float *)ckv->ptr, (const float *)csc->ptr, (const int32_t *)posd->ptr, ratio, dim, n);
    return cuda_ok(cudaGetLastError(), "v41 compress step (n rows)");
}

/* ---- act_quant inplace (cuda_kv_pack.inc.cu:107-141) ----
 * fast_round_scale(amax, inv) = 2^ceil(log2(amax*inv)); one warp per block
 * (<= 32 elements), one element per thread.  mode 0: fp8 e4m3 + ue8m0 scale
 * (max 448); 1: fp4 + ue8m0 (max 6); 2: fp4 + e4m3 scale (max 6). */
__global__ static void v41_act_quant_kernel(float *x, uint32_t dim, uint32_t block, int mode, uint32_t nb_row) {
    v41_pdl_wait();
    const uint32_t row = blockIdx.x / nb_row, b = blockIdx.x % nb_row;
    const uint32_t lane = threadIdx.x;
    float *p = x + (uint64_t)row * dim + (uint64_t)b * block;
    float v = lane < block ? p[lane] : 0.0f;
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float s;
    if (mode == 0) { a = fmaxf(a, 1e-4f); s = v41_pow2_ceil_log2(a / 448.0f); }
    else if (mode == 1) { a = fmaxf(a, 6.0f * ldexpf(1.0f, -126)); s = v41_pow2_ceil_log2(a / 6.0f); }
    else { a = fmaxf(a, 6.0f * ldexpf(1.0f, -9)); s = ds4_e4m3fn_round(a / 6.0f); }
    if (lane < block) {
        float q = v / s;
        if (mode == 0) { q = fminf(fmaxf(q, -448.0f), 448.0f); q = ds4_e4m3fn_round(q); }
        else { q = fminf(fmaxf(q, -6.0f), 6.0f); q = ds4_e2m1fn_round(q); }
        p[lane] = v41_bf16r(q * s);
    }
}
extern "C" int ds4_gpu_v41_act_quant_fp8_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t dim, uint32_t block) {
    if (!x || block == 0 || block > 32u || (dim % block)) return 0;
    v41_act_quant_kernel<<<(unsigned)((uint64_t)n_rows * (dim / block)), 32, 0, ds4_current_stream()>>>((float *)x->ptr, dim, block, 0, dim / block);
    return cuda_ok(cudaGetLastError(), "v41 act quant fp8");
}
extern "C" int ds4_gpu_v41_act_quant_fp4_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t dim, uint32_t block, bool e4m3_scale) {
    if (!x || block == 0 || block > 32u || (dim % block)) return 0;
    v41_act_quant_kernel<<<(unsigned)((uint64_t)n_rows * (dim / block)), 32, 0, ds4_current_stream()>>>((float *)x->ptr, dim, block, e4m3_scale ? 2 : 1, dim / block);
    return cuda_ok(cudaGetLastError(), "v41 act quant fp4");
}

/* ---- KV pack (cuda_kv_pack.inc.cu:20-102) ----
 * The packed bytes ARE the already-quantized values at their native width:
 * pack stores (nibble, scale); reading back computes bf16(nibble*scale) —
 * the same expression act_quant's last line evaluates.  One warp per quant
 * block, the max/shuffle/scale steps verbatim from act_quant (a drift there
 * would break the "packed == the f32 it replaces" identity). */
__global__ static void v41_kv_pack_kernel(uint8_t *cache, const float *rows, uint32_t g0,
                                          uint32_t dim, uint32_t blk, uint32_t row_bytes,
                                          uint32_t nib_bytes, uint32_t nb_row, int mode,
                                          const int32_t *posd, uint32_t ratio, uint32_t g_trash, uint32_t nbatch) {
    v41_pdl_wait();
    const uint32_t r = blockIdx.x / nb_row, b = blockIdx.x % nb_row, lane = threadIdx.x;
    /* graph path: the group number comes from the device position; rows past
     * the completed groups write the trash slot (allocated past the group
     * cap, never read).  nbatch=1 reduces to the original expression. */
    uint32_t grp = g0 + r;
    if (posd) {
        const uint32_t pos = (uint32_t)posd[0], ng_new = (pos % ratio + nbatch) / ratio;
        grp = r < ng_new ? pos / ratio + r : g_trash;
    }
    const float *p = rows + (uint64_t)r * dim + (uint64_t)b * blk;
    const float v = lane < blk ? p[lane] : 0.0f;
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float s;
    if (mode == 0) { a = fmaxf(a, 6.0f * ldexpf(1.0f, -9));   s = ds4_e4m3fn_round(a / 6.0f); }
    else           { a = fmaxf(a, 6.0f * ldexpf(1.0f, -126)); s = v41_pow2_ceil_log2(a / 6.0f); }
    uint8_t *row = cache + (uint64_t)grp * row_bytes;
    if (lane == 0) row[nib_bytes + b] = mode == 0 ? ds4_e4m3fn_f32_to_byte(s) : ds4_e8m0_f32_to_byte(s);
    const uint8_t nib = lane < blk ? ds4_fp4_f32_to_nibble(v / s) : 0u;
    const uint8_t odd = (uint8_t)__shfl_down_sync(0xffffffffu, (uint32_t)nib, 1);
    if (lane < blk && (lane & 1u) == 0u) row[(b * blk + lane) >> 1] = (uint8_t)(nib | (odd << 4));
}
/* Unpack: cached dim d = bf16(nibble * scale) — the pack side's inverse. */
__device__ __forceinline__ static float v41_ckv_get(const uint8_t *row, uint32_t d) {
    const uint8_t by = row[d >> 1];
    const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
    return v41_bf16r(ds4_fp4_nibble_to_f32(nib) * ds4_e4m3fn_to_f32(row[DS4_V41_CKV_NIB + (d >> 4)]));
}
/* Scales decoded once into shared first (a 512-dim row has 32 scales but 512
 * elements; same value, same multiply -> bit-identical, 480 decodes saved). */
__device__ __forceinline__ static float v41_ckv_get_s(const uint8_t *row, uint32_t d, const float *scales) {
    const uint8_t by = row[d >> 1];
    const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
    return v41_bf16r(ds4_fp4_nibble_to_f32(nib) * scales[d >> 4]);
}
__device__ __forceinline__ static float v41_idxk_get(const uint8_t *row, uint32_t d) {
    const uint8_t by = row[d >> 1];
    const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
    return v41_bf16r(ds4_fp4_nibble_to_f32(nib) * ds4_e8m0_to_f32(row[DS4_V41_IDXK_NIB + (d >> 5)]));
}
static int v41_kv_pack_check(const ds4_gpu_tensor *cache, uint32_t g0, uint32_t n_rows, const ds4_gpu_tensor *posd,
                             uint32_t ratio, uint32_t g_trash, uint32_t row_bytes, uint32_t nbatch) {
    if (!posd) return cache->bytes >= (uint64_t)(g0 + n_rows) * row_bytes;
    if (ratio == 0u || nbatch == 0u || n_rows > (ratio - 1u + nbatch) / ratio) return 0;
    return cache->bytes >= ((uint64_t)g_trash + 1u) * row_bytes;
}
extern "C" int ds4_gpu_v41_ckv_pack_tensor(ds4_gpu_tensor *cache, uint32_t g0, const ds4_gpu_tensor *rows, uint32_t n_rows,
                                           const ds4_gpu_tensor *posd, uint32_t ratio, uint32_t g_trash, uint32_t nbatch) {
    if (!cache || !rows || !n_rows) return 0;
    if (!v41_kv_pack_check(cache, g0, n_rows, posd, ratio, g_trash, DS4_V41_CKV_BYTES, nbatch)) return 0;
    const uint32_t dim = DS4_V41_CKV_NIB * 2u, nb = dim / DS4_V41_CKV_BLK;
    v41_kv_pack_kernel<<<(unsigned)((uint64_t)n_rows * nb), 32, 0, ds4_current_stream()>>>(
        (uint8_t *)cache->ptr, (const float *)rows->ptr, g0, dim, DS4_V41_CKV_BLK,
        DS4_V41_CKV_BYTES, DS4_V41_CKV_NIB, nb, 0, posd ? (const int32_t *)posd->ptr : NULL, ratio, g_trash, nbatch);
    return cuda_ok(cudaGetLastError(), "v41 ckv pack");
}
extern "C" int ds4_gpu_v41_idxk_pack_tensor(ds4_gpu_tensor *cache, uint32_t g0, const ds4_gpu_tensor *rows, uint32_t n_rows,
                                            const ds4_gpu_tensor *posd, uint32_t ratio, uint32_t g_trash, uint32_t nbatch) {
    if (!cache || !rows || !n_rows) return 0;
    if (!v41_kv_pack_check(cache, g0, n_rows, posd, ratio, g_trash, DS4_V41_IDXK_BYTES, nbatch)) return 0;
    const uint32_t dim = DS4_V41_IDXK_NIB * 2u, nb = dim / DS4_V41_IDXK_BLK;
    v41_kv_pack_kernel<<<(unsigned)((uint64_t)n_rows * nb), 32, 0, ds4_current_stream()>>>(
        (uint8_t *)cache->ptr, (const float *)rows->ptr, g0, dim, DS4_V41_IDXK_BLK,
        DS4_V41_IDXK_BYTES, DS4_V41_IDXK_NIB, nb, 1, posd ? (const int32_t *)posd->ptr : NULL, ratio, g_trash, nbatch);
    return cuda_ok(cudaGetLastError(), "v41 idxk pack");
}

/* ---- scalar sparse attention (cuda_v41_2.inc.cu:132-303) ----
 * One block per (query, 8-head group), 128 threads = 4 warps x 2 heads; key
 * order = window rows ascending then the topk compressed rows; 8-key tiles
 * through shared (the engine's flash-style online softmax; its grouping
 * differs from the official 64-key tiles because GB10's opt-in shared cap is
 * 99 KB, not 128).  Invalid topk slots get a -inf score AND a zeroed key row:
 * 0 x NaN is NaN, and the shared residue depends on block scheduling (the
 * engine's 2026-09-15 conviction: same input, two runs, different results). */
#define V41_ATTN_HEADS_PER_BLOCK 8u
#define V41_ATTN_KTILE 8u
__global__ static void v41_sparse_attn_kernel(float *o, const float *q, const float *kvw, const uint8_t *kvc, const int32_t *idx,
                                              const float *sink, uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk,
                                              uint32_t n_head, uint32_t hd, float scale, uint32_t full_block, uint32_t ring,
                                              uint32_t win_lo) {
    const uint32_t i = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t per = hd / 32u;                 /* 512/32 = 16 dims per lane */
    __shared__ float ks[V41_ATTN_KTILE][512];
    __shared__ int   kok[V41_ATTN_KTILE];
    __shared__ float ksc[V41_ATTN_KTILE][32];      /* the compressed row's 32 scales, decoded once per row */
    float qa[2][16], acc[2][16], mx[2], sum[2];
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * V41_ATTN_HEADS_PER_BLOCK + warp * 2u + hh;
        for (uint32_t e = 0; e < per; e++) { qa[hh][e] = q[((uint64_t)i * n_head + h) * hd + lane * per + e]; acc[hh][e] = 0.f; }
        mx[hh] = -1e30f; sum[hh] = 0.f;
    }
    const uint32_t p = pos0 + i;                   /* absolute position */
    /* full_block (the DSpark draft block): every position of this chunk sees
     * one key set = the whole window + all n in-block positions (official
     * get_dspark_topk_idxs, no causal cut).  Main path (full_block=0) is
     * causal: position i sees [p+1-window, p]. */
    const uint32_t last = full_block ? pos0 + full_block - 1u : p;
    uint32_t lo = full_block ? (pos0 > window ? pos0 - window : 0u)
                             : (p + 1u > window ? p + 1u - window : 0u);
    if (lo < win_lo) lo = win_lo;                  /* ring slots before win_lo were never written (CED) */
    const uint32_t nwin = last - lo + 1u;
    const uint32_t nkeys = nwin + topk;
    for (uint32_t base = 0; base < nkeys; base += V41_ATTN_KTILE) {
        const uint32_t nt = (nkeys - base) < V41_ATTN_KTILE ? (nkeys - base) : V41_ATTN_KTILE;
        __syncthreads();                            /* previous tile read before overwrite */
        for (uint32_t t = threadIdx.x / 32u; t < nt; t += blockDim.x / 32u) {
            const uint32_t kk = base + t;
            /* two key sources, two storages: window rows stay f32 (2.6 MB,
             * not worth packing), compressed rows are packed FP4 */
            const float *krow = NULL; const uint8_t *cpk = NULL;
            if (kk < nwin) krow = kvw + v41_win_row((int64_t)lo + kk, pos0, window, ring) * hd;
            else { const int32_t g = idx[(uint64_t)i * topk + (kk - nwin)];
                   if (g >= 0 && (uint32_t)g < ng) cpk = kvc + (uint64_t)g * DS4_V41_CKV_BYTES; }
            if (lane == 0) kok[t] = (krow || cpk) ? 1 : 0;
            if (cpk) ksc[t][lane] = ds4_e4m3fn_to_f32(cpk[DS4_V41_CKV_NIB + lane]);
            __syncwarp();
            /* invalid slots must be ZEROED, not left with residue: 0 x residue
             * is only 0 when the residue is finite, and shared reinterpreted
             * as float can be NaN/Inf (see the file header). */
            for (uint32_t d = lane; d < hd; d += 32u)
                ks[t][d] = krow ? krow[d] : (cpk ? v41_ckv_get_s(cpk, d, ksc[t]) : 0.f);
        }
        __syncthreads();
        for (int hh = 0; hh < 2; hh++) {
            float s[V41_ATTN_KTILE];
            float tm = -1e30f;
            for (uint32_t t = 0; t < nt; t++) {
                float d = 0.f;
                for (uint32_t e = 0; e < per; e++) d += qa[hh][e] * ks[t][lane * per + e];
                for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
                s[t] = kok[t] ? d * scale : -1e30f;
                tm = fmaxf(tm, s[t]);
            }
            const float nm = fmaxf(mx[hh], tm);
            const float rs = expf(mx[hh] - nm);     /* rescale once per tile */
            float acc_add[16];
            for (uint32_t e = 0; e < per; e++) acc_add[e] = 0.f;
            float ps = 0.f;
            for (uint32_t t = 0; t < nt; t++) {
                const float pv = expf(s[t] - nm);
                ps += pv;
                const float pb = v41_bf16r(pv);     /* official acc_s_cast */
                for (uint32_t e = 0; e < per; e++) acc_add[e] += pb * ks[t][lane * per + e];
            }
            sum[hh] = sum[hh] * rs + ps;
            for (uint32_t e = 0; e < per; e++) acc[hh][e] = acc[hh][e] * rs + acc_add[e];
            mx[hh] = nm;
        }
    }
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * V41_ATTN_HEADS_PER_BLOCK + warp * 2u + hh;
        const float den = sum[hh] + expf(sink[h] - mx[hh]);   /* the sink enters the denominator only */
        for (uint32_t e = 0; e < per; e++) o[((uint64_t)i * n_head + h) * hd + lane * per + e] = v41_bf16r(acc[hh][e] / den);
    }
}
/* The ds4_gpu_v41_sparse_attn_tensor entry (cuda_v41_2.inc.cu:229-303) moved
 * to cuda/ds41_attn_mma.cuh (P4-5) so its dispatch can mirror the engine's
 * try order: mma decode -> draft-block mma -> scalar split-K -> prefill mma
 * -> this scalar kernel. */

/* ---- SWA window ring commit (cuda_kv_ring.inc.cu:20-56) ----
 * The batch's rows sit in [window, window+n) until committed; committing i
 * rows moves them to their ring slots.  Only the last window rows are
 * committed when n > window: ring slots alias at +window, so the ascending
 * write's net result is exactly "the last window rows in place". */
__global__ static void v41_win_commit_kernel(float *win, uint32_t pos0, uint32_t i0, uint32_t window, uint32_t hd,
                                             const int32_t *posd) {
    v41_pdl_wait();
    const uint32_t j = blockIdx.x, i = i0 + j;
    if (posd) pos0 = (uint32_t)posd[0];
    const float *src = win + (uint64_t)(window + i) * hd;
    float *dst = win + (uint64_t)((pos0 + i) % window) * hd;
    for (uint32_t d = threadIdx.x; d < hd; d += blockDim.x) dst[d] = src[d];
}
extern "C" int ds4_gpu_v41_win_commit_tensor(ds4_gpu_tensor *win, uint32_t pos0, uint32_t n, uint32_t window, uint32_t head_dim,
                                             const ds4_gpu_tensor *posd) {
    if (!win || !window || !n) return 0;
    if (win->bytes < (uint64_t)(window + n) * head_dim * 4) return 0;
    /* graph route: pure decode 1 row or a verify batch <= 8 rows; the kernel
     * lands row i at ring cell (posd[0] + i) % window (cuda_kv_ring.inc.cu:49). */
    if (posd && n > 8u) return 0;
    const uint32_t i0 = n > window ? n - window : 0u, rows = n - i0;
    v41_win_commit_kernel<<<rows, 256, 0, ds4_current_stream()>>>((float *)win->ptr, pos0, i0, window, head_dim,
                                                                  posd ? (const int32_t *)posd->ptr : NULL);
    return cuda_ok(cudaGetLastError(), "v41 win commit");
}

/* Spec decode: save/restore the ring cells a verify batch's commit will
 * overwrite (cuda_kv_ring.inc.cu:34-68).  save (back=0) copies ring cell
 * (pos0+i)%window into snap[i] (indexed by batch row, so the rollback can take
 * the [keep, n) interval); back=1 writes them back.  The block rows of a
 * verify batch sit in the window buffer's block region until commit, so only
 * the cells about to be overwritten need saving: 40 layers x n(<=6) rows x
 * 512 x 4 B ~ 0.5 MB, against 10.5 MB for the whole 128-row window. */
__global__ static void v41_win_ring_snap_kernel(float *win, float *snap, uint32_t pos0, uint32_t i0,
                                                uint32_t window, uint32_t hd, uint32_t back, const int32_t *posd) {
    const uint32_t j = blockIdx.x, i = i0 + j;
    if (posd) pos0 = (uint32_t)posd[0];
    float *ring = win + (uint64_t)((pos0 + i) % window) * hd;
    float *sp = snap + (uint64_t)i * hd;
    if (back) { for (uint32_t d = threadIdx.x; d < hd; d += blockDim.x) ring[d] = sp[d]; }
    else      { for (uint32_t d = threadIdx.x; d < hd; d += blockDim.x) sp[d] = ring[d]; }
}
extern "C" int ds4_gpu_v41_win_ring_snap_tensor(ds4_gpu_tensor *win, ds4_gpu_tensor *snap, uint32_t pos0,
                                                uint32_t i0, uint32_t n, uint32_t window, uint32_t head_dim, int back,
                                                const ds4_gpu_tensor *posd) {
    if (!win || !snap || !window || n <= i0) return 1;   /* no rows to touch = success */
    if (snap->bytes < (uint64_t)n * head_dim * 4) return 0;
    v41_win_ring_snap_kernel<<<n - i0, 256, 0, ds4_current_stream()>>>((float *)win->ptr, (float *)snap->ptr,
                                                                       pos0, i0, window, head_dim, back ? 1u : 0u,
                                                                       posd ? (const int32_t *)posd->ptr : NULL);
    return cuda_ok(cudaGetLastError(), back ? "v41 win ring restore" : "v41 win ring save");
}
