/* ds41_vq_prefill.cuh — DeepSeek V4.1 (ds41) routed-MoE prefill scheduler,
 * ported from the C engine at /data/YoungAi (commit 3946dbc), file
 * `src/cuda/cuda_vq_prefill.inc.cu` (the whole file minus the backward
 * capture).
 *
 * This file does no multiply; it is what surrounds the multiply:
 *   1. a per-layer slot-header cache (vqp_hdr_build, :79-136): copy the blob's
 *      n_expert*3 offset table and each slot's 16 B payload header to the host
 *      once per layer, validate magic/dims, cache dim/nc/nbit.  The blob's
 *      host pages are dropped after the device copy, so every host touch of a
 *      header is a page fault back into the SSD (the engine's profile: 768
 *      faults/layer, 8-35 ms of GPU idle between experts, pf1);
 *   2. a stable host-side counting sort of the (token, pick) pairs by expert
 *      (:159-184) — one expert's weights are touched once, and the same input
 *      reproduces bit for bit (an atomicAdd reduce was deleted 2026-08-22 for
 *      exactly this: two runs of one prompt disagreed);
 *   3. the fixed-pick-order weighted reduce (vqp_reduce_kernel, :60-73).
 * The multiply itself is the run behind vqp_fused_run (declared here, defined
 * in ds41_vq_prefill_fused.cuh -> ds41_vq_prefill_mma.cuh), which writes the
 * sorted ys this file reduces.
 *
 * Scope: the inference scheduler.  The engine's backward capture
 * (g_vqp_last_*, ds4_gpu_v41_vq_capture_expert_out, :222-243) is not ported
 * (the port has no backward pass).  The gr-override store
 * (ds4_gpu_v41_set_gr_override, :246-258) is ported below; g_v41_gr stays
 * all-NULL until the zchain sidecar is applied, which is the state the BARE
 * golden (NO_ZCHAIN=1) was captured in.
 */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds41_primitives.cuh"
#include "ds41_vq_row.cuh"

/* zchain per-expert gain override (cuda_vq_prefill.inc.cu:19-21): [layer] ->
 * device s[n_expert][OUT] row gains for the down matrix, NULL = none for the
 * layer.  Only the down output consumes it (m.gov in ds41_vq_row.cuh); the
 * store below mounts a table, the zchain sidecar's gr half. */
static float *g_v41_gr[64];

/* Mount/unmount one layer's gain table (cuda_vq_prefill.inc.cu:246-258).
 * host NULL unloads; n_expert x out_dim must match the layer's real shape
 * (the caller checks, as the engine's loader does).  A failed upload leaves
 * the slot NULL and returns 0 — never a half-mounted table. */
extern "C" int ds4_gpu_v41_set_gr_override(uint32_t layer, const float *host, uint32_t n_expert, uint32_t out_dim) {
    if (layer >= 64u) return 0;
    if (g_v41_gr[layer]) { (void)cudaFree(g_v41_gr[layer]); g_v41_gr[layer] = NULL; }
    if (!host) return 1;
    const size_t nb = (size_t)n_expert * out_dim * 4;
    if (cudaMalloc((void **)&g_v41_gr[layer], nb) != cudaSuccess) { (void)cudaGetLastError(); g_v41_gr[layer] = NULL; return 0; }
    if (cudaMemcpy(g_v41_gr[layer], host, nb, cudaMemcpyHostToDevice) != cudaSuccess) {
        (void)cudaGetLastError(); (void)cudaFree(g_v41_gr[layer]); g_v41_gr[layer] = NULL; return 0;
    }
    return 1;
}

/* The NVFP4 prefill path's n-alignment pad (cuda_vq_prefill_nvfp4.inc.cu:24).
 * That path is retired in the engine itself (2026-09-20: measured 0.013 Σmin
 * quality loss, see the fused file's account) and is not ported; the +PAD
 * allocation stays because the engine's buffers still carry it, and the pad
 * rows are never written by any live path. */
#define V41_VQN_PAD 2048u

/* ---- per-layer slot header cache (cuda_vq_prefill.inc.cu:74-136) ----
 * off=0 -> slot absent.  nbit = ceil(log2(nc)), floor 1. */
typedef struct { uint64_t off; uint32_t dim, nc, nbit; } vqp_slot_hdr;
static vqp_slot_hdr *g_vqp_hdr[64];   /* [layer] -> [e*3+which] */
static v41_scratch g_vqp_hdr_dev;     /* [n_total*3][6] u32: off(2) magic dim|nc rows cols */

__global__ static void vqp_copy_hdr_kernel(uint32_t *dst, const uint8_t *blob, uint32_t n_total) {
    const uint32_t e = blockIdx.x, w = threadIdx.x;
    if (e >= n_total || w >= 3u) return;
    uint64_t off; memcpy(&off, blob + 16 + ((size_t)e * 3 + w) * 8, 8);
    uint32_t *d = dst + ((size_t)e * 3 + w) * 6;
    memcpy(d, &off, 8);
    if (off) memcpy(d + 2, blob + off, 16);
    else d[2] = d[3] = d[4] = d[5] = 0;
}

static int vqp_hdr_build(uint32_t layer, const uint8_t *blob, uint32_t n_total,
                         uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t ver) {
    if (layer >= 64u) return 0;
    if (g_vqp_hdr[layer]) return 1;
    const uint64_t n = (uint64_t)n_total * 3u;
    uint32_t *dst = (uint32_t *)v41_grow(&g_vqp_hdr_dev, n * 6u * sizeof(uint32_t), "vq prefill hdr");
    if (!dst) return 0;
    vqp_copy_hdr_kernel<<<n_total, 4, 0, ds4_current_stream()>>>(dst, blob, n_total);
    if (!cuda_ok(cudaGetLastError(), "vq prefill hdr copy launch")) return 0;
    uint32_t *raw = (uint32_t *)malloc((size_t)n * 6u * sizeof(uint32_t));
    vqp_slot_hdr *tab = (vqp_slot_hdr *)calloc((size_t)n, sizeof(vqp_slot_hdr));
    if (!raw || !tab ||
        cudaMemcpy(raw, dst, (size_t)n * 6u * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void)cudaGetLastError(); free(raw); free(tab); return 0;
    }
    int bad = 0;
    for (uint64_t k = 0; k < n; k++) {
        const uint32_t *d = raw + k * 6u;
        uint64_t off; memcpy(&off, d, 8);
        tab[k].off = off;
        if (!off) continue;
        const uint32_t which = (uint32_t)(k % 3u), e = (uint32_t)(k / 3u);
        const uint32_t exp_rows = which == 2u ? OUT : MID, exp_cols = which == 2u ? MID : IN;
        const uint32_t d16 = d[3] & 0xFFFFu, n16 = d[3] >> 16;
        /* The expected payload magic follows the on-disk version (:118-121):
         * v3 payloads are 'DQV3' with a different layout.  A wrong magic is a
         * hard failure, never a decode under the wrong layout. */
        if (d[2] != (ver == 3u ? DS4VQ_MAT3_MAGIC : DS4VQ_MAT_MAGIC) || d[4] != exp_rows || d[5] != exp_cols) {
            fprintf(stderr, "ds4: [ds41] vq-prefill L%u e=%u which=%u header bad (magic %08x %ux%u, want %ux%u) -- aborting\n",
                    layer, e, which, d[2], d[4], d[5], exp_rows, exp_cols);
            bad = 1; break;
        }
        /* 48 KB is fused2(dim4)'s codebook-into-shared cap; only dim==4 is
         * still held to it (the v3 dim8 codebook is 64 KB by design). */
        if (d16 == 4u && (size_t)n16 * d16 * 2u > 48u * 1024u) {
            fprintf(stderr, "ds4: [ds41] vq-prefill L%u e=%u codebook %ux%u over the 48 KB shared cap -- aborting\n", layer, e, n16, d16);
            bad = 1; break;
        }
        uint32_t nb = 0; while ((1u << nb) < n16) nb++; if (nb < 1u) nb = 1u;
        tab[k].dim = d16; tab[k].nc = n16; tab[k].nbit = nb;
    }
    free(raw);
    if (bad) { free(tab); return 0; }
    g_vqp_hdr[layer] = tab;
    return 1;
}

/* ---- sorted-run scratch + the reduce (:29-38, :60-73) ---- */
static struct { v41_scratch ys, perm, inv; } g_vqp;

/* out[t][o] = sum_pk w[t][pk] * ys[inv[t*n_expert+pk]][o], pick order fixed
 * -> reproducible.  Never an atomicAdd (see the file header). */
__global__ static void vqp_reduce_kernel(float *out, const float *ys, const int32_t *inv, const float *rw,
                                         uint32_t n_expert, uint32_t OUT) {
    const uint32_t t = blockIdx.y;
    const uint32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= OUT) return;
    float s = 0.0f;
    for (uint32_t pk = 0; pk < n_expert; pk++) {
        const int32_t i = inv[(uint64_t)t * n_expert + pk];
        if (i < 0) continue;
        s += rw[(uint64_t)t * n_expert + pk] * ys[(uint64_t)i * OUT + o];
    }
    out[(uint64_t)t * OUT + o] = s;
}

/* The multiply run (cuda_vq_prefill_fused.inc.cu:168-256).  Returns 0 = the
 * path is unavailable; the caller hard-fails rather than silently falling
 * back (a fallback would hide which path ran). */
static int vqp_fused_run(const uint8_t *blob, const uint32_t *cnt, const uint32_t *off_h, uint32_t n_total_expert,
                         uint32_t nvalid, uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t nc, float clamp,
                         const float *x, const int32_t *perm, uint32_t n_expert, uint32_t layer_index, uint32_t ver);

/* ---- the prefill scheduler (cuda_vq_prefill.inc.cu:139-218) ----
 * Host-side counting sort, then the fused/tensor-core run, then the reduce.
 * The sort's D2H of `selected` is per layer and the path is never captured,
 * so plain synchronous copies are the engine's shape. */
static int cuda_vq_moe_prefill_gemm(
        ds4_gpu_tensor *out, const uint8_t *blob,
        const void *model_map, uint64_t down_offset, uint64_t down_expert_bytes,
        uint32_t IN, uint32_t MID, uint32_t OUT,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, uint32_t ver) {
    if (!g_cublas_ready) { fprintf(stderr, "ds4: [ds41] vq-prefill cuBLAS not ready (L%u)\n", layer_index); return 0; }
    if (!vqp_hdr_build(layer_index, blob, n_total_expert, IN, MID, OUT, ver)) return 0;
    const vqp_slot_hdr *tab = g_vqp_hdr[layer_index];
    const uint64_t npair = (uint64_t)n_tokens * n_expert;
    int32_t *sel_h = (int32_t *)malloc(npair * sizeof(int32_t));
    int32_t *perm_h = (int32_t *)malloc(npair * sizeof(int32_t));
    int32_t *inv_h = (int32_t *)malloc(npair * sizeof(int32_t));
    uint32_t *cnt = (uint32_t *)calloc((size_t)n_total_expert, sizeof(uint32_t));
    uint32_t *off = (uint32_t *)malloc(((size_t)n_total_expert + 1u) * sizeof(uint32_t));
    uint32_t *cur = (uint32_t *)malloc((size_t)n_total_expert * sizeof(uint32_t));
    int ok = 0;
    do {
        if (!sel_h || !perm_h || !inv_h || !cnt || !off || !cur) break;
        if (cudaMemcpy(sel_h, selected->ptr, npair * sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            (void)cudaGetLastError(); break;
        }
        for (uint64_t k = 0; k < npair; k++) {
            const int32_t e = sel_h[k];
            if (e >= 0 && (uint32_t)e < n_total_expert) cnt[e]++;
        }
        off[0] = 0;
        uint32_t ne_max = 0;
        for (uint32_t e = 0; e < n_total_expert; e++) {
            off[e + 1] = off[e] + cnt[e];
            if (cnt[e] > ne_max) ne_max = cnt[e];
        }
        const uint32_t nvalid = off[n_total_expert];
        if (nvalid == 0) {   /* every pick empty: the routed output is 0 */
            ok = (cudaMemsetAsync(out->ptr, 0, (size_t)n_tokens * OUT * sizeof(float), ds4_current_stream()) == cudaSuccess);
            break;
        }
        memcpy(cur, off, (size_t)n_total_expert * sizeof(uint32_t));
        for (uint64_t k = 0; k < npair; k++) {   /* stable: same-expert pairs keep token order */
            const int32_t e = sel_h[k];
            if (e < 0 || (uint32_t)e >= n_total_expert) { inv_h[k] = -1; continue; }
            const uint32_t pos = cur[e]++;
            perm_h[pos] = (int32_t)k;
            inv_h[k] = (int32_t)pos;
        }
        if (!v41_grow(&g_vqp.ys, ((uint64_t)nvalid + V41_VQN_PAD) * OUT * sizeof(float), "vq prefill ys") ||
            !v41_grow(&g_vqp.perm, (uint64_t)nvalid * sizeof(int32_t), "vq prefill perm") ||
            !v41_grow(&g_vqp.inv, npair * sizeof(int32_t), "vq prefill inv")) break;
        (void)ne_max;   /* the fused run accumulates per work item, not per max expert width */
        if (cudaMemcpy(g_vqp.perm.p, perm_h, (size_t)nvalid * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(g_vqp.inv.p, inv_h, (size_t)npair * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            (void)cudaGetLastError(); break;
        }
        /* Walk the slot table on the host first: an expert with tokens but a
         * missing w1/w3/w2 slot is a hard failure — never a kernel that
         * writes zeros into the quality silently (:196-208). */
        int bad = 0;
        for (uint32_t e = 0; e < n_total_expert && !bad; e++) {
            if (!cnt[e]) continue;
            const vqp_slot_hdr *h1 = &tab[(size_t)e * 3u];
            if (!h1->off || !(h1 + 1)->off || !(h1 + 2)->off) {
                fprintf(stderr, "ds4: [ds41] vq-prefill L%u e=%u slot missing (w1/w3/w2) -- aborting (no silent quality downgrade)\n",
                        layer_index, e);
                bad = 1;
            }
        }
        if (bad) break;
        (void)model_map; (void)down_offset; (void)down_expert_bytes;
        if (!vqp_fused_run(blob, cnt, off, n_total_expert, nvalid, IN, MID, OUT,
                           tab[0].nc, clamp, (const float *)x->ptr, (const int32_t *)g_vqp.perm.p, n_expert, layer_index, ver)) break;
        vqp_reduce_kernel<<<dim3((OUT + 255u) / 256u, n_tokens, 1), 256, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)g_vqp.ys.p, (const int32_t *)g_vqp.inv.p, (const float *)weights->ptr, n_expert, OUT);
        ok = cuda_ok(cudaGetLastError(), "vq prefill reduce launch");
    } while (0);
    free(sel_h); free(perm_h); free(inv_h); free(cnt); free(off); free(cur);
    return ok;
}
