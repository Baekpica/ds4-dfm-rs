/* cuda/ds41_draft.cuh — DeepSeek V4.1 (ds41) DSpark draft-tower device
 * pieces, ported from the C engine at /data/YoungAi (commit 3946dbc), file
 * `src/cuda/cuda_v41_draft.inc.cu` (every entry cites its engine line).
 *
 * The draft towers differ from a normal layer in exactly two places; the rest
 * (hc / norm / attention / shared experts / head) reuses the main path's
 * kernels:
 *   1. Their experts are either one VQ blob per tower (this artifact; the
 *      main fused decode kernels take it with layer = DS4_N_LAYER + tower) or
 *      per-expert fp4x32 tensors, which need the dense MoE below (the weight
 *      base comes from the routed expert id, so it cannot be compile-time).
 *   2. The main model's attention input (the hc four-route mean) is captured
 *      inside the main forward into an absolute-position ring -> one mean
 *      kernel (v41_hc_mean_kernel).
 * The markov head also gathers a bf16/f32 table row by a DEVICE token id
 * (a host round-trip per position would cost the whole speculation gain).
 *
 * Failure modes (the engine's warnings, kept): a wrong main_hidden capture
 * point does not error, it just drops acceptance to about 1; a dense-MoE sel
 * out of range reads another expert's bytes silently, so the kernels range-
 * check e.
 *
 * Not ported here (named): nothing — the amp/distillation tables
 * (core_v41_draft_amp.c) are a separate sidecar unit.
 */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

/* One warp computes the dot of an fp4x32 weight row with the activation x
 * (f32 accumulate, activations read on the bf16 grid; engine
 * cuda_v41_draft.inc.cu:16-31).  Layout as the skeleton GEMV: 4 lanes share a
 * 32-element block, each lane takes 4 consecutive bytes = 8 elements (one
 * scale decode amortized over 8).  Returns the full warp sum (lanes 0..31
 * reduced). */
__device__ __forceinline__ static float v41_fp4_row_dot(const uint8_t *wr, const float *x, uint32_t nblk) {
    const uint32_t lane = threadIdx.x & 31u, q = lane & 3u, sub = lane >> 2;
    float acc = 0.f;
    const uint32_t ngrp = (nblk + 7u) / 8u;
    for (uint32_t gi = 0; gi < ngrp; gi++) {
        const uint32_t b = gi * 8u + sub;
        if (b >= nblk) continue;   /* tail group holes: other lanes still have work, no break */
        const uint8_t *p = wr + (uint64_t)b * 17u;
        const float sc = ds4_e8m0_to_f32(p[16]);
        const float *xb = x + b * 32u + q * 8u;
        #pragma unroll
        for (uint32_t j = 0; j < 4u; j++) {
            const uint8_t by = p[q * 4u + j];
            acc += ds4_fp4_nibble_to_f32(by & 0x0Fu) * sc * v41_bf16r(xb[2u * j]);
            acc += ds4_fp4_nibble_to_f32(by >> 4)    * sc * v41_bf16r(xb[2u * j + 1u]);
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}

/* gate/up share one kernel (official Expert: w1/w3 -> bf16 -> swiglu -> bf16).
 * One warp per row, 8 rows per block.  grid (ceil(MID/8), n_tok*K): blockIdx.y
 * is one (token, k-th selected expert) pair.  Engine :38-52. */
__global__ static void v41_mtp_gateup_kernel(float *h, const uint8_t *const *wg, const uint8_t *const *wu,
                                             const int32_t *sel, const float *x, uint32_t IN, uint32_t MID,
                                             uint32_t K, uint32_t n_expert, float clamp) {
    const uint32_t pair = blockIdx.y, t = pair / K, warp = threadIdx.x >> 5;
    const int32_t e = sel[pair];
    if (e < 0 || (uint32_t)e >= n_expert) return;
    const uint32_t r = blockIdx.x * 8u + warp;
    if (r >= MID) return;
    const uint32_t nblk = IN / 32u;
    const float *xs = x + (uint64_t)t * IN;
    float gv = v41_bf16r(v41_fp4_row_dot(wg[e] + (uint64_t)r * nblk * 17u, xs, nblk));
    float uv = v41_bf16r(v41_fp4_row_dot(wu[e] + (uint64_t)r * nblk * 17u, xs, nblk));
    if (clamp > 0.f) { if (gv > clamp) gv = clamp; if (uv > clamp) uv = clamp; if (uv < -clamp) uv = -clamp; }
    if ((threadIdx.x & 31u) == 0) h[(uint64_t)pair * MID + r] = v41_bf16r((gv / (1.0f + expf(-gv))) * uv);
}

/* down: partial[pair][OUT] = bf16(W2·h).  Engine :54-67. */
__global__ static void v41_mtp_down_kernel(float *partial, const uint8_t *const *wd, const int32_t *sel,
                                           const float *h, uint32_t MID, uint32_t OUT, uint32_t K, uint32_t n_expert) {
    const uint32_t pair = blockIdx.y, warp = threadIdx.x >> 5;
    const int32_t e = sel[pair];
    const uint32_t r = blockIdx.x * 8u + warp;
    if (r >= OUT) return;
    if (e < 0 || (uint32_t)e >= n_expert) {   /* unselected slots must write 0: the reduce reads the whole row */
        if ((threadIdx.x & 31u) == 0) partial[(uint64_t)pair * OUT + r] = 0.f;
        return;
    }
    const uint32_t nblk = MID / 32u;
    const float y = v41_fp4_row_dot(wd[e] + (uint64_t)r * nblk * 17u, h + (uint64_t)pair * MID, nblk);
    if ((threadIdx.x & 31u) == 0) partial[(uint64_t)pair * OUT + r] = v41_bf16r(y);
}

/* Per-tower expert device pointer table: n_expert x 3 matrices, resolved once.
 * Why a table: the kernels pick the base by sel, and every expert is its own
 * GGUF tensor (discontinuous offsets) — resolving 384 ranges per step is pure
 * host waste.  Engine :75-99. */
typedef struct { const uint8_t **dev_g, **dev_u, **dev_d; uint32_t n; } v41_mtp_ptrs;
static v41_mtp_ptrs g_mtp_ptrs[8];   /* tower-count slots (official 3); values from the caller */
static v41_scratch g_mtp_h, g_mtp_part;

static int v41_mtp_bind(uint32_t tower, const void *model_map, const uint64_t *off, uint32_t n_expert) {
    if (tower >= 8u) return 0;
    v41_mtp_ptrs *P = &g_mtp_ptrs[tower];
    if (P->n == n_expert && P->dev_g) return 1;
    const uint8_t **host = (const uint8_t **)malloc(sizeof(void *) * 3u * n_expert);
    if (!host) return 0;
    for (uint32_t i = 0; i < 3u * n_expert; i++) {
        /* bytes 1: only the base address is wanted; bounds are the kernels' row
         * math against the GGUF registration (rows x blocks x 17 B). */
        const char *p = cuda_model_range_ptr(model_map, off[i], 1, "mtp expert");
        if (!p) { free((void *)host); return 0; }
        host[i] = (const uint8_t *)p;
    }
    void *dev = NULL;
    if (cudaMalloc(&dev, sizeof(void *) * 3u * n_expert) != cudaSuccess) { (void)cudaGetLastError(); free((void *)host); return 0; }
    const cudaError_t st = cudaMemcpy(dev, host, sizeof(void *) * 3u * n_expert, cudaMemcpyHostToDevice);
    free((void *)host);
    if (st != cudaSuccess) { (void)cudaGetLastError(); (void)cudaFree(dev); return 0; }
    P->dev_g = (const uint8_t **)dev;
    P->dev_u = P->dev_g + n_expert;
    P->dev_d = P->dev_g + 2u * n_expert;
    P->n = n_expert;
    return 1;
}

/* The dense per-expert MoE (engine :101-141).  The engine's union-by-expert
 * negative result (2026-09-17) is archived there and NOT re-tried: the kernel
 * is wavefront-bound (activations are 4 of every 5 wavefronts), not weight-
 * byte-bound. */
extern "C" int ds4_gpu_v41_mtp_moe_tensor(ds4_gpu_tensor *out, const void *model_map, uint32_t tower, const uint64_t *exp_off,
                               uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
                               const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
                               uint32_t n_expert, uint32_t topk, float clamp,
                               const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !selected || !weights || !x || !exp_off || !n_tok || !topk) return 0;
    if ((in_dim % 32u) || (mid_dim % 32u)) return 0;
    if (!v41_mtp_bind(tower, model_map, exp_off, n_expert)) return 0;
    const v41_mtp_ptrs *P = &g_mtp_ptrs[tower];
    const uint64_t np = (uint64_t)n_tok * topk;
    float *h = (float *)v41_grow(&g_mtp_h, np * mid_dim * 4, "mtp h");
    float *part = (float *)v41_grow(&g_mtp_part, np * out_dim * 4, "mtp partial");
    if (!h || !part) return 0;
    v41_mtp_gateup_kernel<<<dim3((mid_dim + 7u) / 8u, (unsigned)np), 256, 0, ds4_current_stream()>>>(
        h, P->dev_g, P->dev_u, (const int32_t *)selected->ptr, (const float *)x->ptr, in_dim, mid_dim, topk, n_expert, clamp);
    if (!cuda_ok(cudaGetLastError(), "v41 mtp gateup")) return 0;
    v41_mtp_down_kernel<<<dim3((out_dim + 7u) / 8u, (unsigned)np), 256, 0, ds4_current_stream()>>>(
        part, P->dev_d, (const int32_t *)selected->ptr, h, mid_dim, out_dim, topk, n_expert);
    if (!cuda_ok(cudaGetLastError(), "v41 mtp down")) return 0;
    v41_vq_reduce_kernel<<<dim3((out_dim + 255u) / 256u, n_tok), 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, part, (const float *)weights->ptr, topk, out_dim);
    return cuda_ok(cudaGetLastError(), "v41 mtp reduce");
}

/* hc four-route mean -> main_hidden slot (official Transformer.forward:
 * main_hiddens.append(h.mean(dim=2))).  Captured at the engram-to-layer edge
 * (core_v41_forward.c:200), NOT at the layer output — the wrong point does
 * not error, acceptance just collapses to about 1.
 *
 * Only the last n_rows rows move (a prefill block has thousands of tokens and
 * the drafter only eats the last few confirmed positions), and the landing
 * slot is an absolute-position ring: row t -> cell ((pos + t) % cap)
 * (engine :143-166).  The graph path passes posd (the device position slot);
 * the eager path passes the position directly. */
__global__ static void v41_hc_mean_kernel(float *out, const float *hc, uint32_t E, uint32_t n_hc, uint32_t n_rows,
                                          uint32_t src_row0, uint32_t slot, uint32_t n_slot,
                                          uint32_t dst_pos0, uint32_t cap, const int32_t *posd) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)n_rows * E) return;
    const uint32_t t = (uint32_t)(i / E), d = (uint32_t)(i % E);
    const float *h = hc + (uint64_t)(src_row0 + t) * n_hc * E + d;
    float s = 0.f;
    for (uint32_t c = 0; c < n_hc; c++) s += h[(uint64_t)c * E];
    const uint32_t pos = posd ? (uint32_t)posd[0] + src_row0 : dst_pos0;
    out[(uint64_t)((pos + t) % cap) * n_slot * E + (uint64_t)slot * E + d] = s / (float)n_hc;
}
extern "C" int ds4_gpu_v41_hc_mean_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *hc, uint32_t n_embd, uint32_t n_hc,
                               uint32_t n_rows, uint32_t src_row0, uint32_t slot, uint32_t n_slot,
                               uint32_t dst_pos0, uint32_t cap, const ds4_gpu_tensor *posd) {
    if (!out || !hc || !n_rows || !cap || n_rows > cap) return 0;
    const uint64_t n = (uint64_t)n_rows * n_embd;
    if (out->bytes < (uint64_t)cap * n_slot * n_embd * 4) return 0;
    v41_hc_mean_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)hc->ptr, n_embd, n_hc, n_rows, src_row0, slot, n_slot,
        dst_pos0, cap, posd ? (const int32_t *)posd->ptr : NULL);
    return cuda_ok(cudaGetLastError(), "v41 hc mean");
}

/* Ring -> contiguous: dst[t] = ring[(first_row + t) % cap].  One block per
 * row, floats in steps (row_floats = n_target * E = 15360).  firstd (non-NULL
 * = the graph path) reads the start row from a device int: the fill start
 * moves with the previous round's accepted count, so a baked value is wrong
 * (silently wrong rows -> dropping acceptance).  Engine :169-189. */
__global__ static void v41_ring_rows_kernel(float *dst, const float *ring, uint32_t row_floats, uint32_t cap,
                                            uint32_t first_row, uint32_t count, const int32_t *firstd) {
    const uint32_t t = blockIdx.x;
    if (t >= count) return;
    if (firstd) first_row = (uint32_t)firstd[0];
    const float *src = ring + (uint64_t)((first_row + t) % cap) * row_floats;
    float *d = dst + (uint64_t)t * row_floats;
    for (uint32_t j = threadIdx.x; j < row_floats; j += blockDim.x) d[j] = src[j];
}
extern "C" int ds4_gpu_v41_ring_rows_tensor(ds4_gpu_tensor *dst, const ds4_gpu_tensor *ring, uint32_t row_floats, uint32_t cap,
                                 uint32_t first_row, uint32_t count, const ds4_gpu_tensor *firstd) {
    if (!dst || !ring || !count || !cap || count > cap) return 0;
    if (dst->bytes < (uint64_t)count * row_floats * 4 || ring->bytes < (uint64_t)cap * row_floats * 4) return 0;
    v41_ring_rows_kernel<<<count, 256, 0, ds4_current_stream()>>>((float *)dst->ptr, (const float *)ring->ptr, row_floats, cap, first_row, count,
                                                                  firstd ? (const int32_t *)firstd->ptr : NULL);
    return cuda_ok(cudaGetLastError(), "v41 ring rows");
}

/* markov embed: gather a table row by a DEVICE token id -> f32 (bf16 grid).
 * No host readback: the block's positions are sequentially dependent (the
 * bias of position i needs the token sampled at i), so a D2H per position
 * would cost the whole gain.  Engine :191-215. */
__global__ static void v41_row_gather_kernel(float *out, const uint8_t *tab, const int32_t *ids,
                                             uint32_t which, uint32_t dim, uint32_t out_row, uint32_t is_f32) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    const int32_t id = ids[which];
    float v = 0.f;
    if (id >= 0) {
        if (is_f32) memcpy(&v, tab + ((uint64_t)id * dim + d) * 4u, 4);
        else { __nv_bfloat16 h; memcpy(&h, tab + ((uint64_t)id * dim + d) * 2u, 2); v = __bfloat162float(h); }
    }
    out[(uint64_t)out_row * dim + d] = v;
}
/* elem_bytes comes from the GGUF-registered tensor type (4 = f32, 2 = bf16) —
 * the engine never assumes it, and guessing does not error, it reads garbage
 * (the 2026-09-15 crash: a bf16 read of the f32 markov table, acceptance
 * 0.14/5). */
extern "C" int ds4_gpu_v41_row_gather_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t tab_offset, uint64_t n_rows, uint32_t dim, uint32_t elem_bytes,
                                  const ds4_gpu_tensor *ids, uint32_t which, uint32_t out_row) {
    if (!out || !ids || (elem_bytes != 2u && elem_bytes != 4u)) return 0;
    const uint64_t bytes = n_rows * dim * elem_bytes;
    if (tab_offset > model_size || bytes > model_size - tab_offset) return 0;
    const uint8_t *tab = (const uint8_t *)cuda_model_range_ptr(model_map, tab_offset, bytes, "markov embed");
    if (!tab) return 0;
    v41_row_gather_kernel<<<(dim + 255u) / 256u, 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, tab, (const int32_t *)ids->ptr, which, dim, out_row, elem_bytes == 4u);
    return cuda_ok(cudaGetLastError(), "v41 row gather");
}

/* logits row += bias row (markov bias; official logits[:, i].add_(logits_bias)).
 * Engine :218-229. */
__global__ static void v41_row_add_kernel(float *dst, const float *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}
extern "C" int ds4_gpu_v41_row_add_tensor(ds4_gpu_tensor *dst, uint64_t dst_row, const ds4_gpu_tensor *src, uint64_t n) {
    if (!dst || !src) return 0;
    v41_row_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)dst->ptr + dst_row * n, (const float *)src->ptr, n);
    return cuda_ok(cudaGetLastError(), "v41 row add");
}

/* ---- markov bias cache (2026-10-07; interface semantics in the engine's
 * ds4_gpu_v41.h).  Only the original bf16 markov_head path uses it; the
 * distilled bias table (mkH_dev) does not. ----
 * lookup: one block, slot = thread; one id lives in at most one slot (the
 * claim records the id, a hit never claims), so at most one thread hits.
 * Miss = round-robin evict: the old slot's id becomes the new id on the spot,
 * and the bias lands in add — nobody reads that slot in between (same-stream
 * order: lookup -> gather -> GEMV -> add -> next position's lookup).
 * Engine :232-253. */
__global__ static void v41_mkcache_lookup_kernel(int32_t *hit, int32_t *cache_ids, uint32_t *next, const int32_t *ids, uint32_t which, uint32_t n_slots) {
    __shared__ int found;
    if (threadIdx.x == 0) found = -1;
    __syncthreads();
    const int32_t id = ids[which];
    if (threadIdx.x < n_slots && cache_ids[threadIdx.x] == id) found = (int)threadIdx.x;
    __syncthreads();
    if (threadIdx.x == 0) {
        if (found >= 0) { hit[0] = found; return; }
        const uint32_t s = *next % n_slots;
        *next = s + 1u;
        cache_ids[s] = id;
        hit[0] = -(int32_t)s - 2;
    }
}
extern "C" int ds4_gpu_v41_mkcache_lookup_tensor(ds4_gpu_tensor *hit, ds4_gpu_tensor *cache_ids, ds4_gpu_tensor *next, const ds4_gpu_tensor *ids,
                                      uint32_t which, uint32_t n_slots) {
    if (!hit || !cache_ids || !next || !ids || n_slots == 0u || n_slots > 1024u) return 0;
    v41_mkcache_lookup_kernel<<<1, (unsigned)n_slots, 0, ds4_current_stream()>>>(
        (int32_t *)hit->ptr, (int32_t *)cache_ids->ptr, (uint32_t *)next->ptr, (const int32_t *)ids->ptr, which, n_slots);
    return cuda_ok(cudaGetLastError(), "v41 mkcache lookup");
}
/* add: hit => += cache[slot]; miss => += bias and store it.  Same addition as
 * v41_row_add_kernel (dst += the same f32) => bit-identical.  Engine :255-267. */
__global__ static void v41_mkcache_add_kernel(float *dst, const float *bias, float *cache, const int32_t *hit, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t h = hit[0];
    if (h >= 0) dst[i] += cache[(uint64_t)h * n + i];
    else { const float b = bias[i]; dst[i] += b; cache[(uint64_t)(-h - 2) * n + i] = b; }
}
extern "C" int ds4_gpu_v41_mkcache_add_tensor(ds4_gpu_tensor *logits, uint64_t row, const ds4_gpu_tensor *bias, ds4_gpu_tensor *cache,
                                   const ds4_gpu_tensor *hit, uint64_t n) {
    if (!logits || !bias || !cache || !hit) return 0;
    v41_mkcache_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)logits->ptr + row * n, (const float *)bias->ptr, (float *)cache->ptr, (const int32_t *)hit->ptr, n);
    return cuda_ok(cudaGetLastError(), "v41 mkcache add");
}
