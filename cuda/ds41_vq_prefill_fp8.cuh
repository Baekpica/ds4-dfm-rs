/* ds41_vq_prefill_fp8.cuh — DeepSeek V4.1 (ds41) routed experts on FP8 tensor
 * cores for prefill blocks: the port's own arm (no engine counterpart).
 *
 * Why (measured on the hVQ artifact, 2026-11-12, 64K prefill): the bf16
 * register-direct kernel (vqs, ds41_vq_prefill_mma.cuh) is 27% of prefill,
 * 9.1 ms per launch at 2,048 rows.  It rebuilds bf16 weights from the E4M3
 * book in every fragment (a cvt chain per operand register).  A codeword is
 * already an `mma.m16n8k32.e4m3` operand, and MiniMax's warp-specialized
 * kernels (cuda/minimax_mma8w.cuh) feed it unconverted.  They read the row
 * layout a DQVL v3 payload has: F16 row gains, 12-bit indices, a 13th-bit
 * plane.  In the model they take 5.1 ms per launch, three per layer.
 *
 * Numerics, a named deviation from the engine: the engine multiplies bf16
 * activations.  Here they are E4M3 with one amax scale per (row, 128 dims),
 * the way the official model quantizes every GEMM input to FP8 (act_quant;
 * its scales are powers of two per 32 dims).  Products accumulate on the
 * tensor cores for 128 dims, then join the f32 sum with the scale.  The row
 * gain multiplies the f32 sum and outputs round to bf16 as before; SwiGLU is
 * the official form (glmvq_swiglu_value<GLMVQ_CLAMP_BOTH>); the reduce keeps
 * vqp_reduce_kernel's pick order.  The gate is teacher-forced NLL and KL
 * against a chunk-size change (docs/deepseek41-optimization-2026-11-12.md).
 *
 * A 12-bit layer rides the same 13-bit kernels: its 4,096-word book is
 * zero-padded to 8,192 and its plane is zeros.
 *
 * Must follow ds41_vq_prefill_mma.cuh (vqs_blob_fits) and the GLM VQ
 * family (ds4_glmvq_gpu.cuh precedes this file in ds4_cuda.cu).
 */
#pragma once
#include "minimax_mma8w.cuh"

/* DQVL v3 payload geometry (v41_vq_open<1>, ds41_vq_row.cuh): a header, one
 * F16 gain per row, then per row one 12-bit index per 8-dim codeword and,
 * on 13-bit books, one plane bit per codeword after all main rows. */
enum { VQF_HDR_BYTES = 32, VQF_GAIN_BYTES = 2, VQF_MAIN_COLS = 16, VQF_MAIN_BYTES = 3, VQF_PLANE_COLS = 64 };

static struct {
    v41_scratch x8, xs, gate, mid8, ms, ys, items, token, zero;
    v41_scratch book[64];
    const uint8_t *book_src[64];
    /* Per layer: 0 unchecked, 1 the blob fits this arm, -1 it does not. */
    struct { const uint8_t *blob; int fits; } layer[64];
    int sms;
} g_vqf;

/* DS4_DS41_VQ_FP8=0 keeps every layer on the bf16 kernels (vqs / vqm). */
static bool vqf_enabled(void) {
    static int off = -1;
    if (off < 0) {
        const char *env = getenv("DS4_DS41_VQ_FP8");
        off = env && !strcmp(env, "0");
    }
    return !off;
}

/* The layer's first present slot: every slot shares its book and width. */
static const vqp_slot_hdr *vqf_first(const vqp_slot_hdr *tab, uint32_t n_total) {
    for (uint32_t k = 0; k < n_total * 3u; k++) {
        if (tab[k].off) { return &tab[k]; }
    }
    return NULL;
}

/* What this arm takes: v3 payloads in the aligned device copy (vqs's
 * precondition: the kernels copy and prefetch 16-byte pieces), one book for
 * the layer, 12- or 13-bit, dims in whole 128-blocks, and no mounted gain
 * override (the kernels read the payload's F16 gains alone). */
static bool vqf_takes(uint32_t layer, const uint8_t *blob, const vqp_slot_hdr *tab, uint32_t n_total,
                      uint32_t IN, uint32_t MID, uint32_t OUT, const float *gr) {
    if (!vqf_enabled() || gr || layer >= 64u) { return false; }
    if (g_vqf.layer[layer].blob == blob && g_vqf.layer[layer].fits) { return g_vqf.layer[layer].fits > 0; }

    const vqp_slot_hdr *first = vqf_first(tab, n_total);
    const uint32_t nbit = first ? first->nbit : 0u;
    bool fits = (nbit == 12u || nbit == 13u) && !(IN % GLMVQ_MMA8_K) && !(MID % GLMVQ_MMA8_K) && !(OUT % GLMVQ_MMA8_K) &&
        vqs_blob_fits(layer, blob, n_total, IN, MID, OUT);
    for (uint32_t k = 0; fits && k < n_total * 3u; k++) {
        if (!tab[k].off) { continue; }
        fits = tab[k].v3 && tab[k].nbit == nbit && tab[k].cb_off == first->cb_off;
    }
    g_vqf.layer[layer].blob = blob;
    g_vqf.layer[layer].fits = fits ? 1 : -1;
    return fits;
}

/* The layer's book as the kernels stage it: MW_BOOK bytes, 16-byte aligned. */
static const uint8_t *vqf_book(uint32_t layer, const uint8_t *src, uint32_t words) {
    if (g_vqf.book_src[layer] == src) { return (const uint8_t *)g_vqf.book[layer].p; }
    uint8_t *book = (uint8_t *)v41_grow(&g_vqf.book[layer], MW_BOOK, "vq fp8 book");
    if (!book) { return NULL; }

    const cudaStream_t stream = ds4_current_stream();
    const size_t bytes = (size_t)words * GLMVQ_VECTOR;
    if (cudaMemsetAsync(book, 0, MW_BOOK, stream) != cudaSuccess ||
        cudaMemcpyAsync(book, src, bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        (void)cudaGetLastError();
        return NULL;
    }
    g_vqf.book_src[layer] = src;
    return book;
}

/* Plane words of a 12-bit layer: every 13th bit is zero. */
static const uint8_t *vqf_zero_plane(uint64_t bytes) {
    if (g_vqf.zero.cap >= bytes) { return (const uint8_t *)g_vqf.zero.p; }
    uint8_t *zero = (uint8_t *)v41_grow(&g_vqf.zero, bytes, "vq fp8 zero plane");
    if (!zero || cudaMemsetAsync(zero, 0, bytes, ds4_current_stream()) != cudaSuccess) {
        (void)cudaGetLastError();
        return NULL;
    }
    return zero;
}

/* Sorted pair -> its token row (perm holds token * K + pick). */
__global__ static void vqf_token_kernel(uint32_t *token, const int32_t *perm, uint32_t K, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { token[i] = (uint32_t)perm[i] / K; }
}

/* vqp_reduce_kernel over bf16 rows: the kernels store their bf16-rounded
 * outputs as bf16 (half the reduce's read). */
__global__ static void vqf_reduce_kernel(float *out, const __nv_bfloat16 *ys, const int32_t *inv, const float *rw,
                                         uint32_t n_expert, uint32_t OUT) {
    const uint32_t t = blockIdx.y;
    const uint32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= OUT) { return; }
    float s = 0.0f;
    for (uint32_t pk = 0; pk < n_expert; pk++) {
        const int32_t i = inv[(uint64_t)t * n_expert + pk];
        if (i < 0) { continue; }
        s += rw[(uint64_t)t * n_expert + pk] * __bfloat162float(ys[(uint64_t)i * OUT + o]);
    }
    out[(uint64_t)t * OUT + o] = s;
}

/* One layer: quantize the block's rows once, gate and up as two launches of
 * the gate kernel, SwiGLU into down's E4M3 rows, down, reduce.
 *
 *   x [n][IN] f32 -> x8 E4M3 + xs [n][IN/128]
 *        gate items -> gate [pairs][MID] bf16 --+
 *        up items   -> up   [pairs][MID] bf16 --+-> mid8 E4M3 + ms [pairs][MID/128]
 *        down items -> ys   [pairs][OUT] bf16 -> out [n][OUT] f32
 *
 * Returns 0 = failure (the caller hard-fails). */
static int vqf_run(float *out, const uint8_t *blob, const vqp_slot_hdr *tab, const uint32_t *cnt, const uint32_t *off_h,
                   uint32_t n_total, uint32_t nvalid, uint32_t IN, uint32_t MID, uint32_t OUT, float clamp,
                   const float *x, const int32_t *perm, const int32_t *inv, const float *rw, uint32_t n_expert,
                   uint32_t n_tokens, uint32_t layer_index) {
    if (!g_vqf.sms) {
        if (cudaDeviceGetAttribute(&g_vqf.sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess || g_vqf.sms <= 0) {
            (void)cudaGetLastError();
            g_vqf.sms = 0;
            return 0;
        }
        fprintf(stderr, "ds4: [ds41] vq-prefill FP8 tensor-core path (e4m3 mma, <=%u pairs/item), %d SMs\n", (unsigned)MW_BT, g_vqf.sms);
    }
    const vqp_slot_hdr *first = vqf_first(tab, n_total);
    const uint32_t nbit = first->nbit;
    const uint8_t *book = vqf_book(layer_index, blob + first->cb_off, first->nc);
    const uint64_t plane_max = (uint64_t)(MID > OUT ? MID : OUT) * (IN > MID ? IN : MID) / VQF_PLANE_COLS;
    const uint8_t *zero = nbit == 13u ? NULL : vqf_zero_plane(plane_max);
    if (!book || (nbit != 13u && !zero)) { return 0; }

    uint32_t nit = 0;
    for (uint32_t e = 0; e < n_total; e++) { nit += (cnt[e] + MW_BT - 1u) / MW_BT; }
    const uint64_t xb = (uint64_t)n_tokens * IN / GLMVQ_MMA8_K;
    uint8_t *x8 = (uint8_t *)v41_grow(&g_vqf.x8, (uint64_t)n_tokens * IN, "vq fp8 x8");
    float *xs = (float *)v41_grow(&g_vqf.xs, xb * sizeof(float), "vq fp8 x scales");
    __nv_bfloat16 *gate = (__nv_bfloat16 *)v41_grow(&g_vqf.gate, (uint64_t)nvalid * MID * 2u, "vq fp8 gate");
    uint8_t *mid8 = (uint8_t *)v41_grow(&g_vqf.mid8, (uint64_t)nvalid * MID, "vq fp8 mid8");
    float *ms = (float *)v41_grow(&g_vqf.ms, (uint64_t)nvalid * (MID / GLMVQ_MMA8_K) * sizeof(float), "vq fp8 mid scales");
    /* ys also holds up's bf16 rows until SwiGLU has read them (OUT >= MID rows fit). */
    const uint64_t ys_cols = OUT > MID ? OUT : MID;
    __nv_bfloat16 *ys = (__nv_bfloat16 *)v41_grow(&g_vqf.ys, (uint64_t)nvalid * ys_cols * 2u, "vq fp8 ys");
    mw_item *items = (mw_item *)v41_grow(&g_vqf.items, (uint64_t)nit * 3u * sizeof(mw_item), "vq fp8 items");
    uint32_t *token = (uint32_t *)v41_grow(&g_vqf.token, (uint64_t)nvalid * sizeof(uint32_t), "vq fp8 tokens");
    mw_item *ih = (mw_item *)malloc((size_t)nit * 3u * sizeof(mw_item));
    if (!x8 || !xs || !gate || !mid8 || !ms || !ys || !items || !token || !ih) { free(ih); return 0; }

    /* Items: gate, up, down runs of at most MW_BT sorted pairs per expert. */
    uint32_t k = 0;
    for (uint32_t m = 0; m < 3u; m++) {
        const uint32_t rows = m == 2u ? OUT : MID, cols = m == 2u ? MID : IN;
        const uint32_t main_stride = cols * VQF_MAIN_BYTES / VQF_MAIN_COLS;
        for (uint32_t e = 0; e < n_total; e++) {
            if (!cnt[e]) { continue; }
            const uint8_t *gain = blob + tab[(size_t)e * 3u + m].off + VQF_HDR_BYTES;
            const uint8_t *main = gain + (size_t)rows * VQF_GAIN_BYTES;
            const uint8_t *plane = nbit == 13u ? main + (size_t)rows * main_stride : zero;
            for (uint32_t t0 = 0; t0 < cnt[e]; t0 += MW_BT) {
                const uint32_t nt = cnt[e] - t0 < MW_BT ? cnt[e] - t0 : MW_BT;
                ih[k++] = {off_h[e] + t0, nt, (const __half *)gain, main, plane, NULL, 0};
            }
        }
    }
    const cudaStream_t stream = ds4_current_stream();
    /* ih is pageable: the runtime stages it before returning (as the bf16 arm's items). */
    const bool copied = cudaMemcpyAsync(items, ih, (size_t)nit * 3u * sizeof(mw_item), cudaMemcpyHostToDevice, stream) == cudaSuccess;
    free(ih);
    if (!copied) { (void)cudaGetLastError(); fprintf(stderr, "ds4: [ds41] vq-prefill L%u FP8 item copy failed\n", layer_index); return 0; }

    mw_quant<<<(uint32_t)((xb * 32u + GLMVQ_THREADS - 1u) / GLMVQ_THREADS), GLMVQ_THREADS, 0, stream>>>(x8, xs, x, xb);
    vqf_token_kernel<<<(nvalid + 255u) / 256u, 256, 0, stream>>>(token, perm, n_expert, nvalid);
    if (!cuda_ok(cudaGetLastError(), "vq prefill fp8 quant")) { return 0; }

    const uint32_t sms = (uint32_t)g_vqf.sms;
    __nv_bfloat16 *up = ys;
    cudaError_t err = mw_launch<GLMVQ_MMA_GATE, false>(items, nit, book, NULL, x8, xs, token, gate, NULL, MID, IN, sms, stream);
    if (err == cudaSuccess) {
        err = mw_launch<GLMVQ_MMA_GATE, false>(items + nit, nit, book, NULL, x8, xs, token, up, NULL, MID, IN, sms, stream);
    }
    if (err == cudaSuccess) {
        const uint64_t warps = (uint64_t)nvalid * (MID / GLMVQ_MMA8_K);
        mw_swiglu_quant<GLMVQ_CLAMP_BOTH><<<(uint32_t)((warps * 32u + GLMVQ_THREADS - 1u) / GLMVQ_THREADS), GLMVQ_THREADS, 0, stream>>>(
            mid8, ms, gate, up, MID, nvalid, clamp > 0.0f ? clamp : INFINITY);
        err = mw_launch<GLMVQ_MMA_DOWN, false>(items + 2u * nit, nit, book, NULL, mid8, ms, token, NULL, ys, OUT, MID, sms, stream);
    }
    if (!cuda_ok(err, "vq prefill fp8 experts")) { return 0; }

    vqf_reduce_kernel<<<dim3((OUT + 255u) / 256u, n_tokens, 1), 256, 0, stream>>>(out, ys, inv, rw, n_expert, OUT);
    return cuda_ok(cudaGetLastError(), "vq prefill fp8 reduce");
}
