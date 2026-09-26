/* Included by the CUDA backend. MiMo's model geometry is explicit here. */
#include "cuda/mimo2_primitives.cuh"
#include "cuda/mimo2_media.cuh"
#include "cuda/mimo2_prefill.cuh"
#include "cuda/mimo2_dflash_attn.cuh"

static void m2_hmma_init(void) {
    m2_hmma_available = mimo2_hmma::supported();
}

extern "C" int ds4_gpu_mimo2_sum_add(
        ds4_gpu_tensor *cur, const ds4_gpu_tensor *down,
        const ds4_gpu_tensor *weights, uint32_t rows) {
    enum { WIDTH = 4096, USED = 8, MIN_ROWS = 32, MAX_ROWS = 8192, THREADS = 256 };
    if (rows < MIN_ROWS || rows > MAX_ROWS) { return -1; }
    const char *env = getenv("DS4_MIMO2_SUM_RESIDUAL");
    if (env && strcmp(env, "1") != 0) { return -1; }
    const uint64_t bytes = (uint64_t)rows * WIDTH * sizeof(float);
    if (!cur || !down || !weights || !cur->ptr || !down->ptr || !weights->ptr ||
        cur->bytes < bytes || down->bytes < bytes * USED ||
        weights->bytes < (uint64_t)rows * USED * sizeof(float)) { return 0; }
    if ((uintptr_t)cur->ptr % alignof(float) ||
        (uintptr_t)down->ptr % alignof(float) ||
        (uintptr_t)weights->ptr % alignof(float)) { return -1; }

    const uint64_t count = bytes / sizeof(float);
    mimo2_sum_residual<<<(count + THREADS - 1) / THREADS, THREADS, 0, ds4_current_stream()>>>(
        (float *)cur->ptr, (const float *)down->ptr, (const float *)weights->ptr, count);
    return cuda_ok(cudaGetLastError(), "MiMo sum residual");
}

extern "C" int ds4_gpu_mimo2_qkv(
        ds4_gpu_tensor *q, ds4_gpu_tensor *k, ds4_gpu_tensor *v,
        const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *table,
        uint32_t kv_heads, uint32_t rows) {
    enum { Q_WIDTH = 64 * 192, KEY = 192, VALUE = 128, ROTARY = 64 };
    const uint64_t kw = (uint64_t)kv_heads * KEY, vw = (uint64_t)kv_heads * VALUE;
    const uint64_t count = (uint64_t)rows * (Q_WIDTH + kw + vw);
    if (!q || !k || !v || !qkv || !table || !rows || count > INT_MAX ||
        (kv_heads != 4 && kv_heads != 8) ||
        q->bytes < (uint64_t)rows * Q_WIDTH * sizeof(float) ||
        k->bytes < (uint64_t)rows * kw * sizeof(float) ||
        v->bytes < (uint64_t)rows * vw * sizeof(float) ||
        qkv->bytes < count * sizeof(float) ||
        table->bytes < (uint64_t)rows * ROTARY * sizeof(float)) { return 0; }
    mimo2_split_rope<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)q->ptr, (float *)k->ptr, (float *)v->ptr,
        (const float *)qkv->ptr, (const float2 *)table->ptr, kv_heads, rows);
    return cuda_ok(cudaGetLastError(), "MiMo QKV/RoPE");
}

extern "C" int ds4_gpu_mimo2_router(
        ds4_gpu_tensor *ids, ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits,
        const void *map, uint64_t size, uint64_t offset, uint32_t rows) {
    enum { EXPERTS = 256, USED = 8 };
    const uint64_t bias_bytes = EXPERTS * sizeof(float);
    if (!ids || !weights || !logits || !map || !rows || rows > INT_MAX ||
        offset > size || bias_bytes > size - offset ||
        ids->bytes < (uint64_t)rows * USED * sizeof(int) ||
        weights->bytes < (uint64_t)rows * USED * sizeof(float) ||
        logits->bytes < (uint64_t)rows * EXPERTS * sizeof(float)) { return 0; }
    const float *bias = (const float *)cuda_model_range_ptr(map, offset, bias_bytes, "MiMo bias");
    if (!bias) { return 0; }
    const char *warp = getenv("DS4_MIMO2_ROUTER_WARP");
    if (rows <= 8 && !(warp && warp[0] == '0' && warp[1] == '\0')) {
        mimo2_router_warp<<<rows, 32, 0, ds4_current_stream()>>>(
            (int *)ids->ptr, (float *)weights->ptr, (const float *)logits->ptr, bias);
        return cuda_ok(cudaGetLastError(), "mimo2 router warp");
    }
    mimo2_router<<<rows, 128, 0, ds4_current_stream()>>>(
        (int *)ids->ptr, (float *)weights->ptr, (const float *)logits->ptr, bias);
    return cuda_ok(cudaGetLastError(), "MiMo router");
}

extern "C" int ds4_gpu_mimo2_kv_store(
        ds4_gpu_tensor *cache, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *positions, uint32_t kv_heads, uint32_t rows, uint32_t capacity) {
    const uint64_t kw = (uint64_t)kv_heads * 192, vw = (uint64_t)kv_heads * 128;
    const uint64_t count = (uint64_t)rows * (kw + vw);
    if (!cache || !k || !v || !positions || (kv_heads != 4 && kv_heads != 8) ||
        !rows || !capacity || rows > capacity ||
        positions->bytes < (uint64_t)rows * sizeof(uint32_t) ||
        count > INT_MAX || cache->bytes < (uint64_t)capacity * (kw + vw) * sizeof(__half) ||
        k->bytes < (uint64_t)rows * kw * sizeof(float) ||
        v->bytes < (uint64_t)rows * vw * sizeof(float)) { return 0; }
    mimo2_kv_store<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (__half *)cache->ptr, (const float *)k->ptr, (const float *)v->ptr,
        kv_heads, rows, (const unsigned *)positions->ptr, capacity);
    return cuda_ok(cudaGetLastError(), "MiMo KV store");
}

static float *m2_split_buf = nullptr;
/* Which attention kernel the last call launched. The decode-round test
 * reads this so a matching walk result cannot hide a missed dispatch. */
static int m2_attn_path = 0;
/* 5: tensor-core full-attention prefill. 0 is the walk or the shared tile. */
enum { M2_PATH_HMMA = 5, M2_PATH_SWA_HMMA = 6 };

// ds4_gpu_cleanup calls this. A second init must allocate again, not leak.
static void m2_split_release(void) {
    m2df_release();
    if (!m2_split_buf) { return; }
    (void)cudaFree(m2_split_buf);
    m2_split_buf = nullptr;
}

static int m2_split_ready(void) {
    if (m2_split_buf) { return 1; }
    if (ds4_capture_active()) { return 0; }
    const size_t n = (size_t)M2_DECODE_SPLITS * 64;
    const size_t bytes = n * (2 + 128) * sizeof(float);
    if (cudaMalloc(&m2_split_buf, bytes) != cudaSuccess) {
        m2_split_buf = nullptr;
        // The failure is handled by the walking fallback. Leave no sticky error.
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_mimo2_dflash_attn(
        ds4_gpu_tensor *attn, ds4_gpu_tensor *q,
        ds4_gpu_tensor *k_ctx, ds4_gpu_tensor *k_noise,
        ds4_gpu_tensor *v_ctx, ds4_gpu_tensor *v_noise,
        const float *q_weight, const float *k_weight, const float *sinks,
        uint32_t q0, uint32_t n, uint32_t ctx) {
    const uint64_t q_bytes = (uint64_t)n * DF_Q * sizeof(float);
    const uint64_t ctx_bytes = (uint64_t)ctx * DF_KV * sizeof(float);
    const uint64_t noise_bytes = (uint64_t)n * DF_KV * sizeof(float);
    if (!attn || !q || !k_ctx || !k_noise || !v_ctx || !v_noise ||
        !q_weight || !k_weight || !sinks || n < 1 || n > 8 ||
        ctx < 1 || ctx > DF_WIN || q0 < ctx ||
        attn->bytes < q_bytes || q->bytes < q_bytes ||
        k_ctx->bytes < ctx_bytes || v_ctx->bytes < ctx_bytes ||
        k_noise->bytes < noise_bytes || v_noise->bytes < noise_bytes) { return 0; }
    return m2df_attn_launch(
        (float *)attn->ptr, (float *)q->ptr, (float *)k_ctx->ptr, (float *)k_noise->ptr,
        (float *)v_ctx->ptr, (float *)v_noise->ptr, q_weight, k_weight, sinks,
        q0, n, ctx, ds4_current_stream());
}

extern "C" int ds4_gpu_mimo2_attention(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *cache,
        const ds4_gpu_tensor *positions, const void *map, uint64_t size,
        uint64_t sink_offset, uint32_t kv_heads, uint32_t capacity,
        uint32_t rows, uint32_t window, uint32_t pos0) {
    enum { HEADS = 64, KEY = 192, VALUE = 128, CONTEXT = 1048576, MAX_ROWS = 65535 };
    const uint64_t cache_bytes = (uint64_t)capacity * kv_heads * (KEY + VALUE) * sizeof(__half);
    if (!out || !q || !cache || !positions || !rows || rows > MAX_ROWS ||
        !capacity || capacity > CONTEXT || rows > capacity ||
        !((kv_heads == 4 && window == 0) || (kv_heads == 8 && window == 128)) ||
        out->bytes < (uint64_t)rows * HEADS * VALUE * sizeof(float) ||
        q->bytes < (uint64_t)rows * HEADS * KEY * sizeof(float) ||
        positions->bytes < (uint64_t)rows * sizeof(uint32_t) || cache->bytes < cache_bytes) { return 0; }
    const float *sinks = nullptr;
    if (map) {
        const uint64_t bytes = HEADS * sizeof(float);
        if (sink_offset > size || bytes > size - sink_offset) { return 0; }
        sinks = (const float *)cuda_model_range_ptr(map, sink_offset, bytes, "MiMo sinks");
        if (!sinks) { return 0; }
    }
    /* Session admission owns position bounds, contiguous rows and ring retention.
     * No scalar position is baked into capture; all queries read live state.
     * m2_use_tile is the measured crossover. The L2 load keeps the KV head
     * resident across query rows. DS4_MIMO2_FATTN_L2=0 keeps the scalar tile.
     * DS4_MIMO2_FATTN=0 keeps the walk. One full-attention row is split
     * across key slices; DS4_MIMO2_ATTN_SPLIT=0 keeps that walk. SWA stays.
     * Windowless prefill of 32 or more rows uses tensor cores. The walk
     * reloads one KV head per query head; HMMA scores a 64-row tile once.
     * Summation order changes. DS4_MIMO2_NO_PREFILL_HMMA=1 restores the walk.
     * The tensor-core path stages KV with cp.async. That copy is byte-identical
     * to the scalar loads. DS4_MIMO2_NO_PREFILL_ASYNC=1 keeps the scalar loads. */
    const char *fattn = getenv("DS4_MIMO2_FATTN");
    const char *l2 = getenv("DS4_MIMO2_FATTN_L2");
    const int fattn_off = fattn && fattn[0] == '0' && fattn[1] == '\0';
    const char *no_hmma = getenv("DS4_MIMO2_NO_PREFILL_HMMA");
    const int hmma_off = no_hmma && no_hmma[0] == '1' && no_hmma[1] == '\0';
    /* FATTN=0 is the older full-attention kill switch. It still skips HMMA
     * and the shared tile, so diagnostic runs keep the walk. */
    const int hmma = m2_hmma_available && window == 0 && kv_heads == 4 && rows >= 32 &&
        !hmma_off && !fattn_off;
    m2_attn_path = 0;
    const char *no_async = getenv("DS4_MIMO2_NO_PREFILL_ASYNC");
    const int async_off = no_async && no_async[0] == '1' && no_async[1] == '\0';
    const int async_copy = !async_off;
    const char *no_swa = getenv("DS4_MIMO2_NO_SWA_HMMA");
    const int swa_off = no_swa && no_swa[0] == '1' && no_swa[1] == '\0';
    /* Window-128 prefill reloads each key once per query head. The tensor-core
     * tile shares that key. Summation order changes.
     * DS4_MIMO2_NO_SWA_HMMA=1 restores the walk. Decode stays at one row. */
    const int swa_hmma = m2_hmma_available && window == 128 && kv_heads == 8 && rows >= 32 && !swa_off;
    const char *split_env = getenv("DS4_MIMO2_ATTN_SPLIT");
    const char *swa_decode_env = getenv("DS4_MIMO2_SWA_DECODE");
    const char *swa_vec_env = getenv("DS4_MIMO2_SWA_VEC");
    const char *split_vec_env = getenv("DS4_MIMO2_SPLIT_VEC");
    const char *split16_env = getenv("DS4_MIMO2_SPLIT16");
    // Share the window across eight query heads. FP32 reduction order may
    // change; =0 retains the walk for numerical comparisons.
    const int swa_decode = rows == 1 && window == 128 && kv_heads == 8 &&
        !(swa_decode_env && swa_decode_env[0] == '0' && swa_decode_env[1] == '\0');
    const int split_vec = split_vec_env && split_vec_env[0] == '1' && split_vec_env[1] == '\0';
    const int nsplit = (split16_env && split16_env[0] == '1' && split16_env[1] == '\0')
        ? 16 : M2_DECODE_SPLITS;
    const int split_off = split_env && split_env[0] == '0' && split_env[1] == '\0';
    const int tile = !fattn_off && m2_use_tile(window, kv_heads, rows, pos0);
    const int hinted = tile && !(l2 && l2[0] == '0' && l2[1] == '\0');
    // Either kill switch skips the scratch alloc and keeps the walk.
    // Rows 2..8 are the speculative verify width. The split kernel is one
    // query, so each row reuses the same scratch on this stream.
    const int split = rows >= 1 && rows <= 8 && window == 0 && kv_heads == 4 &&
        !fattn_off && !split_off && m2_split_ready();
    if (swa_decode) {
        const int swa_vec = !(swa_vec_env && swa_vec_env[0] == '0' && swa_vec_env[1] == '\0');
        m2_attn_path = swa_vec ? 2 : 1;
        mimo2_swa_decode<<<dim3(1, kv_heads), 256, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)q->ptr, (const __half *)cache->ptr,
            sinks, (const unsigned *)positions->ptr, kv_heads, capacity, window, swa_vec);
    } else if (swa_hmma) {
        m2_attn_path = M2_PATH_SWA_HMMA;
        mimo2_hmma::prefill<mimo2_hmma::Async, 128><<<
            dim3((rows + mimo2_hmma::TQ - 1) / mimo2_hmma::TQ, HEADS),
            32 * mimo2_hmma::WARPS, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)q->ptr, (const __half *)cache->ptr,
            sinks, (const unsigned *)positions->ptr, rows, kv_heads, capacity);
    } else if (hmma && async_copy) {
        m2_attn_path = M2_PATH_HMMA;
        mimo2_hmma::prefill<mimo2_hmma::Async><<<
            dim3((rows + mimo2_hmma::TQ - 1) / mimo2_hmma::TQ, HEADS),
            32 * mimo2_hmma::WARPS, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)q->ptr, (const __half *)cache->ptr,
            sinks, (const unsigned *)positions->ptr, rows, kv_heads, capacity);
    } else if (hmma) {
        m2_attn_path = M2_PATH_HMMA;
        mimo2_hmma::prefill<<<dim3((rows + mimo2_hmma::TQ - 1) / mimo2_hmma::TQ, HEADS),
            32 * mimo2_hmma::WARPS, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)q->ptr, (const __half *)cache->ptr,
            sinks, (const unsigned *)positions->ptr, rows, kv_heads, capacity);
    } else if (split) {
        const size_t n = (size_t)M2_DECODE_SPLITS * 64;
        float *pmax = m2_split_buf;
        float *pden = pmax + n;
        float *pacc = pden + n;
        const float *qbase = (const float *)q->ptr;
        float *obase = (float *)out->ptr;
        const unsigned *pbase = (const unsigned *)positions->ptr;
        m2_attn_path = split_vec ? 4 : 3;
        for (uint32_t r = 0; r < rows; r++) {
            mimo2_attn_split<<<dim3(nsplit, kv_heads), 512, 0, ds4_current_stream()>>>(
                pmax, pden, pacc, qbase + (uint64_t)r * HEADS * KEY,
                (const __half *)cache->ptr, sinks, pbase + r, kv_heads, capacity,
                nsplit, split_vec);
            mimo2_attn_merge<<<64, 32, 0, ds4_current_stream()>>>(
                obase + (uint64_t)r * HEADS * VALUE, pmax, pden, pacc, nsplit);
        }
    } else if (hinted) {
        mimo2_attn_l2<<<dim3(rows, kv_heads), 512, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)q->ptr, (const __half *)cache->ptr,
            sinks, (const unsigned *)positions->ptr, kv_heads, capacity);
    } else if (tile) {
        mimo2_attn_tile<<<dim3(rows, kv_heads), 512, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)q->ptr, (const __half *)cache->ptr,
            sinks, (const unsigned *)positions->ptr, kv_heads, capacity);
    } else {
        mimo2_attention<<<dim3(HEADS / 4, rows), 128, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)q->ptr, (const __half *)cache->ptr,
            sinks, (const unsigned *)positions->ptr, kv_heads, capacity, window);
    }
    return cuda_ok(cudaGetLastError(), "MiMo asymmetric attention");
}

static const float *m2_f32(const void *map, uint64_t size, uint64_t offset, uint64_t n, const char *what) {
    const uint64_t bytes = n * sizeof(float);
    if (!map || !n || offset > size || bytes > size - offset) { return nullptr; }
    return (const float *)cuda_model_range_ptr(map, offset, bytes, what);
}

extern "C" int ds4_gpu_mimo2_patch(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *in, const void *map, uint64_t size,
        uint64_t w0, uint64_t w1, uint32_t n, uint32_t oc, uint32_t ic, uint32_t kt, uint32_t patch) {
    const uint64_t width = (uint64_t)ic * kt * patch * patch;
    const uint64_t count = (uint64_t)n * oc;
    if (!out || !in || !n || !oc || !ic || !kt || !patch || count > INT_MAX ||
        in->bytes < (uint64_t)n * width * sizeof(float) ||
        out->bytes < count * sizeof(float)) { return 0; }
    const float *a = m2_f32(map, size, w0, (uint64_t)oc * ic * patch * patch, "MiMo patch0");
    const float *b = m2_f32(map, size, w1, (uint64_t)oc * ic * patch * patch, "MiMo patch1");
    if (!a || !b) { return 0; }
    mimo2_patch<<<(unsigned)((count + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)in->ptr, a, b, (int)n, (int)oc, (int)ic, (int)kt, (int)patch);
    return cuda_ok(cudaGetLastError(), "MiMo patch");
}

extern "C" int ds4_gpu_mimo2_rope(
        ds4_gpu_tensor *base, const ds4_gpu_tensor *cos, const ds4_gpu_tensor *sin,
        uint32_t n, uint32_t heads, uint32_t hd, uint32_t stride, uint32_t off) {
    const uint64_t count = (uint64_t)n * heads;
    if (!base || !cos || !sin || !n || !heads || !hd || hd > 128 || (hd & 1) ||
        count > INT_MAX || stride < off + heads * hd ||
        base->bytes < (uint64_t)n * stride * sizeof(float) ||
        cos->bytes < (uint64_t)n * hd * sizeof(float) ||
        sin->bytes < (uint64_t)n * hd * sizeof(float)) { return 0; }
    mimo2_rope<<<(unsigned)((count + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)base->ptr, (const float *)cos->ptr, (const float *)sin->ptr,
        (int)n, (int)heads, (int)hd, (int)stride, (int)off);
    return cuda_ok(cudaGetLastError(), "MiMo media rope");
}

extern "C" int ds4_gpu_mimo2_attn(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
        const void *map, uint64_t size, uint64_t sink_off, int have_sink,
        uint32_t n, uint32_t q_heads, uint32_t kv_heads, uint32_t hd,
        uint32_t q_stride, uint32_t k_stride, uint32_t v_stride,
        uint32_t q_off, uint32_t k_off, uint32_t v_off,
        int window, int causal, int group) {
    const uint64_t count = (uint64_t)n * q_heads;
    if (!out || !q || !k || !v || !n || !q_heads || !kv_heads || !hd ||
        q_heads % kv_heads || count > INT_MAX ||
        q_stride < q_off + q_heads * hd || k_stride < k_off + kv_heads * hd ||
        v_stride < v_off + kv_heads * hd ||
        q->bytes < (uint64_t)n * q_stride * sizeof(float) ||
        k->bytes < (uint64_t)n * k_stride * sizeof(float) ||
        v->bytes < (uint64_t)n * v_stride * sizeof(float) ||
        out->bytes < count * hd * sizeof(float)) { return 0; }
    const float *sinks = nullptr;
    if (have_sink) {
        sinks = m2_f32(map, size, sink_off, q_heads, "MiMo vision sinks");
        if (!sinks) { return 0; }
    }
    mimo2_attn<<<(unsigned)((count + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)q->ptr, (const float *)k->ptr, (const float *)v->ptr, sinks,
        (int)n, (int)q_heads, (int)kv_heads, (int)hd,
        (int)q_stride, (int)k_stride, (int)v_stride, (int)q_off, (int)k_off, (int)v_off,
        window, causal, group);
    return cuda_ok(cudaGetLastError(), "MiMo media attention");
}

extern "C" int ds4_gpu_mimo2_bias(
        ds4_gpu_tensor *x, const void *map, uint64_t size, uint64_t offset,
        uint32_t n, uint32_t dim) {
    const uint64_t count = (uint64_t)n * dim;
    if (!x || !n || !dim || count > INT_MAX || x->bytes < count * sizeof(float)) { return 0; }
    const float *bias = m2_f32(map, size, offset, dim, "MiMo bias");
    if (!bias) { return 0; }
    mimo2_bias<<<(unsigned)((count + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)x->ptr, bias, (int)n, (int)dim);
    return cuda_ok(cudaGetLastError(), "MiMo bias");
}

extern "C" int ds4_gpu_mimo2_swiglu(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
        const void *map, uint64_t size, uint64_t gate_b, uint64_t up_b, int have_bias,
        uint32_t n, uint32_t dim) {
    const uint64_t count = (uint64_t)n * dim;
    if (!out || !gate || !up || !n || !dim || count > INT_MAX ||
        out->bytes < count * sizeof(float) || gate->bytes < count * sizeof(float) ||
        up->bytes < count * sizeof(float)) { return 0; }
    const float *gb = nullptr, *ub = nullptr;
    if (have_bias) {
        gb = m2_f32(map, size, gate_b, dim, "MiMo gate bias");
        ub = m2_f32(map, size, up_b, dim, "MiMo up bias");
        if (!gb || !ub) { return 0; }
    }
    mimo2_swiglu<<<(unsigned)((count + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, gb, ub, (int)n, (int)dim);
    return cuda_ok(cudaGetLastError(), "MiMo swiglu");
}

extern "C" int ds4_gpu_mimo2_gelu(ds4_gpu_tensor *x, uint32_t n) {
    if (!x || !n || (uint64_t)n > INT_MAX || x->bytes < (uint64_t)n * sizeof(float)) { return 0; }
    mimo2_gelu<<<(n + 255) / 256, 256, 0, ds4_current_stream()>>>((float *)x->ptr, (int)n);
    return cuda_ok(cudaGetLastError(), "MiMo gelu");
}

extern "C" int ds4_gpu_mimo2_ln(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t weight, uint64_t bias, int have_bias, uint32_t rows, uint32_t dim, float eps) {
    const uint64_t count = (uint64_t)rows * dim;
    if (!out || !x || !rows || !dim || count > INT_MAX || !(eps > 0.f) ||
        out->bytes < count * sizeof(float) || x->bytes < count * sizeof(float)) { return 0; }
    const float *w = m2_f32(map, size, weight, dim, "MiMo layernorm");
    const float *b = nullptr;
    if (!w) { return 0; }
    if (have_bias) {
        b = m2_f32(map, size, bias, dim, "MiMo layernorm bias");
        if (!b) { return 0; }
    }
    mimo2_layernorm<<<rows, 256, 256 * sizeof(float), ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)x->ptr, w, b, (int)dim, eps);
    return cuda_ok(cudaGetLastError(), "MiMo layernorm");
}

extern "C" int ds4_gpu_mimo2_gather(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *in, const ds4_gpu_tensor *index,
        uint32_t units, uint32_t width) {
    const uint64_t count = (uint64_t)units * width;
    if (!out || !in || !index || !units || !width || count > INT_MAX ||
        out->bytes < count * sizeof(float) || in->bytes < count * sizeof(float) ||
        index->bytes < (uint64_t)units * sizeof(int)) { return 0; }
    mimo2_gather<<<(unsigned)((count + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)in->ptr, (const int *)index->ptr, (int)units, (int)width);
    return cuda_ok(cudaGetLastError(), "MiMo gather");
}

extern "C" int ds4_gpu_mimo2_conv1d(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *in, const void *map, uint64_t size,
        uint64_t weight, uint64_t bias, int have_bias,
        uint32_t n_in, uint32_t n_out, uint32_t cin, uint32_t cout,
        uint32_t k, uint32_t stride, uint32_t pad) {
    const uint64_t count = (uint64_t)n_out * cout;
    if (!out || !in || !n_in || !n_out || !cin || !cout || !k || !stride || count > INT_MAX ||
        in->bytes < (uint64_t)cin * n_in * sizeof(float) ||
        out->bytes < count * sizeof(float)) { return 0; }
    const float *w = m2_f32(map, size, weight, (uint64_t)cout * cin * k, "MiMo conv");
    const float *b = nullptr;
    if (!w) { return 0; }
    if (have_bias) {
        b = m2_f32(map, size, bias, cout, "MiMo conv bias");
        if (!b) { return 0; }
    }
    mimo2_conv1d<<<(unsigned)((count + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)in->ptr, w, b,
        (int)n_in, (int)n_out, (int)cin, (int)cout, (int)k, (int)stride, (int)pad);
    return cuda_ok(cudaGetLastError(), "MiMo conv1d");
}

extern "C" int ds4_gpu_mimo2_rvq(
        ds4_gpu_tensor *ids, ds4_gpu_tensor *residual, const void *map, uint64_t size,
        uint64_t offset, uint32_t n, uint32_t dim, uint32_t bins) {
    if (!ids || !residual || !n || !dim || !bins || n > INT_MAX ||
        ids->bytes < (uint64_t)n * sizeof(int) ||
        residual->bytes < (uint64_t)n * dim * sizeof(float)) { return 0; }
    const float *book = m2_f32(map, size, offset, (uint64_t)bins * dim, "MiMo rvq");
    if (!book) { return 0; }
    mimo2_rvq<<<n, 256, 0, ds4_current_stream()>>>(
        (int *)ids->ptr, (float *)residual->ptr, book, (int)dim, (int)bins);
    return cuda_ok(cudaGetLastError(), "MiMo rvq");
}

extern "C" int ds4_gpu_mimo2_to_ctime(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *in, uint32_t rows, uint32_t cols) {
    const uint64_t count = (uint64_t)rows * cols;
    if (!out || !in || !rows || !cols || count > INT_MAX ||
        out->bytes < count * sizeof(float) || in->bytes < count * sizeof(float)) { return 0; }
    mimo2_to_ctime<<<(unsigned)((count + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)in->ptr, (int)rows, (int)cols);
    return cuda_ok(cudaGetLastError(), "MiMo transpose");
}

extern "C" int ds4_gpu_mimo2_code_sum(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *ids, const void *map, uint64_t size,
        uint64_t offset, uint32_t n, uint32_t dim, uint32_t vocab, uint32_t channels) {
    const uint64_t count = (uint64_t)n * dim;
    if (!out || !ids || !n || !dim || !vocab || !channels || count > INT_MAX ||
        out->bytes < count * sizeof(float) ||
        ids->bytes < (uint64_t)n * channels * sizeof(int)) { return 0; }
    const float *table = m2_f32(map, size, offset, (uint64_t)dim * vocab * channels, "MiMo codes");
    if (!table) { return 0; }
    mimo2_code_sum<<<(unsigned)((count + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const int *)ids->ptr, table, (int)n, (int)dim, (int)vocab, (int)channels);
    return cuda_ok(cudaGetLastError(), "MiMo code sum");
}
