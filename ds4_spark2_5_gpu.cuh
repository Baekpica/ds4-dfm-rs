/* Spark-X2.6 CUDA entries. Included by ds4_cuda.cu beside the other family
 * graphs; the two kernels live in cuda/spark2_5.cuh. */

#include "ds4_spark2_5_vq.h"
#include "cuda/spark2_5.cuh"
#include "cuda/spark2_5_vq_gemm.cuh"

extern "C" int ds4_gpu_spark_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *inv, uint32_t rows, uint32_t heads, uint32_t dim,
        uint32_t pairs) {
    if (!x || !positions || !inv || !rows || !heads || !dim || !pairs ||
        2u * pairs > dim) {
        return 0;
    }
    if (x->bytes < (uint64_t)rows * heads * dim * sizeof(float) ||
        positions->bytes < (uint64_t)rows * sizeof(uint32_t) ||
        inv->bytes < (uint64_t)pairs * sizeof(float)) {
        return 0;
    }
    const uint64_t jobs = (uint64_t)rows * heads;
    spark_rope_kernel<<<(uint32_t)jobs, 128, 0, ds4_current_stream()>>>(
        (float *)x->ptr, (const uint32_t *)positions->ptr, (const float *)inv->ptr,
        rows, heads, dim, pairs);
    return cuda_ok(cudaGetLastError(), "Spark-X2.6 rotary");
}

extern "C" int ds4_gpu_spark_gelu_mul(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up, uint64_t count) {
    if (!out || !gate || !up || !count ||
        out->bytes < count * sizeof(float) ||
        gate->bytes < count * sizeof(float) ||
        up->bytes < count * sizeof(float)) {
        return 0;
    }
    spark_gelu_mul_kernel<<<(uint32_t)((count + 255u) / 256u), 256, 0,
        ds4_current_stream()>>>((float *)out->ptr, (const float *)gate->ptr,
        (const float *)up->ptr, count);
    return cuda_ok(cudaGetLastError(), "Spark-X2.6 GELU product");
}

extern "C" int ds4_gpu_spark_qkv_split(ds4_gpu_tensor *q, ds4_gpu_tensor *k,
        ds4_gpu_tensor *v, const ds4_gpu_tensor *qkv, uint32_t rows) {
    const uint64_t stride = (uint64_t)SPARK_Q_DIM + 2u * SPARK_KV_DIM;
    if (!q || !k || !v || !qkv || !rows ||
        q->bytes < (uint64_t)rows * SPARK_Q_DIM * sizeof(float) ||
        k->bytes < (uint64_t)rows * SPARK_KV_DIM * sizeof(float) ||
        v->bytes < (uint64_t)rows * SPARK_KV_DIM * sizeof(float) ||
        qkv->bytes < (uint64_t)rows * stride * sizeof(float)) {
        return 0;
    }
    const uint64_t count = (uint64_t)rows * stride;
    spark_qkv_split_kernel<<<(uint32_t)((count + 255u) / 256u), 256, 0,
        ds4_current_stream()>>>((float *)q->ptr, (float *)k->ptr, (float *)v->ptr,
        (const float *)qkv->ptr, rows, SPARK_Q_DIM, SPARK_KV_DIM);
    return cuda_ok(cudaGetLastError(), "Spark-X2.6 q/k/v split");
}

/* ---- VQ restore (container v2 and v3) ---- */

static uint64_t spark2_5_le64(const uint8_t *p) {
    uint64_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static uint32_t spark2_5_le32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

/* A GVQ1 stream's payload pointers. FORMAT.md: 128-byte header, row gains at
 * 32 (f16 each), 12-bit pairs at 40 (3 bytes per two vectors), the 13th-bit
 * plane at 48 (one bit per vector), total size at 56. */
struct spark2_5_stream {
    const uint8_t *gain, *main, *plane;
};

static bool spark2_5_stream_open(const void *blob, uint64_t bytes, uint32_t rows,
        uint32_t cols, struct spark2_5_stream *s) {
    const uint8_t *h = (const uint8_t *)blob;
    if (!h || bytes < SPARK_HEADER) { return false; }
    const uint64_t elements = (uint64_t)rows * cols;
    const uint64_t gain = spark2_5_le64(h + 32);
    const uint64_t main = spark2_5_le64(h + 40);
    const uint64_t plane = spark2_5_le64(h + 48);
    if (memcmp(h, "GVQ1", 4) || h[4] != 1 || h[5] || h[6] != 13 ||
        h[7] != SPARK_VECTOR || spark2_5_le32(h + 8) != rows ||
        spark2_5_le32(h + 12) != cols || spark2_5_le32(h + 16) != SPARK_WORDS ||
        spark2_5_le32(h + 20) != 1 || spark2_5_le32(h + 28) ||
        spark2_5_le64(h + 56) != bytes ||
        gain + 2ull * rows > bytes || main + elements * 3 / 16 > bytes ||
        plane + elements / 64 > bytes) {
        return false;
    }
    s->gain = h + gain;
    s->main = h + main;
    s->plane = h + plane;
    return true;
}

/* GVR1 residual plane: 128-byte header (rows at 8, cols at 12, book at 24,
 * 256-word E4M3 book, index offset at 32, size at 56, payload digest at
 * 96..128 — FORMAT.md "GVR1 residual"), one index byte per 8-weight vector. */
static bool spark2_5_res_open(const void *blob, uint64_t bytes, uint32_t rows,
        uint32_t cols, const uint8_t **index) {
    const uint8_t *h = (const uint8_t *)blob;
    if (!h || bytes < SPARK_HEADER) { return false; }
    const uint64_t start = spark2_5_le64(h + 32);
    const uint64_t vectors = (uint64_t)rows * cols / SPARK_VECTOR;
    if (memcmp(h, "GVR1", 4) || h[4] != 1 || h[5] || h[6] != 8 ||
        h[7] != SPARK_VECTOR || spark2_5_le32(h + 8) != rows ||
        spark2_5_le32(h + 12) != cols || spark2_5_le32(h + 16) != 256 ||
        spark2_5_le32(h + 20) != 1 || spark2_5_le32(h + 28) ||
        spark2_5_le64(h + 56) != bytes || start + vectors > bytes) {
        return false;
    }
    *index = h + start;
    return true;
}

/* Restore one matrix into `out` (rows x cols BF16). `vq2`/`book2` are NULL for
 * a version-2 release; both stages share the geometry. */
extern "C" int ds4_gpu_spark_restore(ds4_gpu_tensor *out,
        const void *vq, uint64_t vq_bytes, const void *vq2, uint64_t vq2_bytes,
        const void *book, const void *book2, const void *res, uint64_t res_bytes,
        const void *res_book, uint32_t rows, uint32_t cols) {
    struct spark2_5_stream a, b;
    const uint8_t *r8 = NULL;
    memset(&b, 0, sizeof(b));
    if (!out || !book || !rows || !cols || cols % SPARK_VECTOR ||
        out->bytes < (uint64_t)rows * cols * sizeof(__nv_bfloat16) ||
        !spark2_5_stream_open(vq, vq_bytes, rows, cols, &a) ||
        !spark2_5_res_open(res, res_bytes, rows, cols, &r8)) {
        return 0;
    }
    const bool staged = vq2 != NULL;
    if (staged && !spark2_5_stream_open(vq2, vq2_bytes, rows, cols, &b)) {
        return 0;
    }
    const uint64_t vectors = (uint64_t)rows * cols / SPARK_VECTOR;
    spark_restore_kernel<<<(uint32_t)((vectors + 255u) / 256u), 256, 0,
        ds4_current_stream()>>>(a.main, a.plane, (const __half *)a.gain,
        (const uint8_t *)book, r8, (const uint8_t *)res_book,
        b.main, b.plane, (const __half *)b.gain, (const uint8_t *)book2,
        rows, cols, (__nv_bfloat16 *)out->ptr);
    return cuda_ok(cudaGetLastError(), "Spark-X2.6 VQ restore");
}

/* Restore one matrix into `out` as FP8 E4M3 codes [rows][cols] followed by
 * block-128 __half scales [rows][cols / 128]. */
extern "C" int ds4_gpu_spark_restore_fp8(ds4_gpu_tensor *out,
        const void *vq, uint64_t vq_bytes, const void *vq2, uint64_t vq2_bytes,
        const void *book, const void *book2, const void *res, uint64_t res_bytes,
        const void *res_book, uint32_t rows, uint32_t cols) {
    struct spark2_5_stream a, b;
    const uint8_t *r8 = NULL;
    memset(&b, 0, sizeof(b));
    if (!out || !book || !rows || !cols || cols % SPARK_FP8_BLOCK ||
        out->bytes < spark2_5_fp8_matrix_bytes(cols, rows) ||
        !spark2_5_stream_open(vq, vq_bytes, rows, cols, &a) ||
        !spark2_5_res_open(res, res_bytes, rows, cols, &r8)) {
        return 0;
    }
    const bool staged = vq2 != NULL;
    if (staged && !spark2_5_stream_open(vq2, vq2_bytes, rows, cols, &b)) {
        return 0;
    }
    const uint64_t vectors = (uint64_t)rows * cols / SPARK_VECTOR;
    uint8_t *codes = (uint8_t *)out->ptr;
    __half *scales = (__half *)(codes + (uint64_t)rows * cols);
    spark_restore_fp8_kernel<<<(uint32_t)((vectors + 255u) / 256u), 256, 0,
        ds4_current_stream()>>>(a.main, a.plane, (const __half *)a.gain,
        (const uint8_t *)book, r8, (const uint8_t *)res_book,
        b.main, b.plane, (const __half *)b.gain, (const uint8_t *)book2,
        rows, cols, codes, scales);
    return cuda_ok(cudaGetLastError(), "Spark-X2.6 FP8 VQ restore");
}

extern "C" int ds4_gpu_v41_matmul_bf16_dev_tensor(ds4_gpu_tensor *out,
    const ds4_gpu_tensor *W, uint64_t in_dim, uint64_t out_dim,
    const ds4_gpu_tensor *x, uint32_t n_tok);

static ds4_gpu_tensor *g_spark_dequant_bf16 = NULL;

extern "C" int ds4_gpu_spark_bf16_to_fp8(ds4_gpu_tensor *dst,
        const void *src_bf16, uint32_t rows, uint32_t cols) {
    if (!dst || !src_bf16 || !rows || !cols || (cols % SPARK_FP8_BLOCK) != 0u ||
        dst->bytes < spark2_5_fp8_matrix_bytes(cols, rows)) {
        return 0;
    }
    const uint32_t n_blocks = cols / SPARK_FP8_BLOCK;
    uint8_t *codes = (uint8_t *)dst->ptr;
    __half *scales = (__half *)(codes + (uint64_t)rows * cols);

    /* Stream in 4096-row chunks through device memory so large host-mapped
     * tables (e.g. 131072 x 2560 token_embd) copy via DMA and quantize at
     * device HBM speed. */
    const uint32_t chunk_rows = rows < 4096u ? rows : 4096u;
    const uint64_t chunk_bytes = (uint64_t)chunk_rows * cols * sizeof(uint16_t);
    ds4_gpu_tensor *stage = ds4_gpu_tensor_alloc(chunk_bytes);
    if (!stage) { return 0; }

    for (uint32_t r0 = 0; r0 < rows; r0 += chunk_rows) {
        const uint32_t nr = (rows - r0) < chunk_rows ? (rows - r0) : chunk_rows;
        const uint64_t bbytes = (uint64_t)nr * cols * sizeof(uint16_t);
        const uint8_t *src_ptr = (const uint8_t *)src_bf16 +
            (uint64_t)r0 * cols * sizeof(uint16_t);
        if (cudaMemcpyAsync(stage->ptr, src_ptr, (size_t)bbytes,
                cudaMemcpyHostToDevice, ds4_current_stream()) != cudaSuccess) {
            ds4_gpu_tensor_free(stage);
            return 0;
        }
        const uint64_t vectors = (uint64_t)nr * cols / 8u;
        spark_bf16_to_fp8_kernel<<<(uint32_t)((vectors + 255u) / 256u), 256u, 0,
            ds4_current_stream()>>>(
            (const __nv_bfloat16 *)stage->ptr, nr, cols,
            codes + (uint64_t)r0 * cols,
            scales + (uint64_t)r0 * n_blocks);
        if (!cuda_ok(cudaGetLastError(), "Spark-X2.6 BF16->FP8 quantize")) {
            ds4_gpu_tensor_free(stage);
            return 0;
        }
    }
    (void)cudaStreamSynchronize(ds4_current_stream());
    ds4_gpu_tensor_free(stage);
    return 1;
}

extern "C" int ds4_gpu_spark_embed_fp8(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *tokens, const ds4_gpu_tensor *table_fp8,
        uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd) {
    if (!out || !tokens || !table_fp8 || !n_vocab || !n_tokens || !n_embd ||
        (n_embd % SPARK_FP8_BLOCK) != 0u ||
        table_fp8->bytes < spark2_5_fp8_matrix_bytes(n_embd, n_vocab) ||
        out->bytes < (uint64_t)n_tokens * n_embd * sizeof(float) ||
        tokens->bytes < (uint64_t)n_tokens * sizeof(int32_t)) {
        return 0;
    }
    const uint8_t *codes = (const uint8_t *)table_fp8->ptr;
    const __half *scales = (const __half *)(codes + (uint64_t)n_vocab * n_embd);
    spark_embed_fp8_kernel<<<n_tokens, 256u, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const int32_t *)tokens->ptr, codes, scales,
        n_vocab, n_embd);
    return cuda_ok(cudaGetLastError(), "Spark-X2.6 FP8 embedding gather");
}

extern "C" int ds4_gpu_spark_matmul_fp8(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *w_fp8, uint32_t in_dim, uint32_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !w_fp8 || !x || !in_dim || !out_dim || !n_tok ||
        in_dim % SPARK_FP8_BLOCK ||
        w_fp8->bytes < spark2_5_fp8_matrix_bytes(in_dim, out_dim) ||
        out->bytes < (uint64_t)n_tok * out_dim * sizeof(float) ||
        x->bytes < (uint64_t)n_tok * in_dim * sizeof(float)) {
        return 0;
    }
    const uint8_t *codes = (const uint8_t *)w_fp8->ptr;
    const __half *scales = (const __half *)(codes + (uint64_t)out_dim * in_dim);
    if (n_tok == 1u && in_dim <= 10240u) {
        const uint32_t blocks = (out_dim + 7u) / 8u;
        const uint32_t smem = in_dim * (uint32_t)sizeof(float);
        spark_fp8_gemv_1_kernel<<<blocks, 256u, smem, ds4_current_stream()>>>(
            codes, scales, (const float *)x->ptr, (float *)out->ptr,
            in_dim, out_dim);
        return cuda_ok(cudaGetLastError(), "Spark-X2.6 FP8 GEMV-1");
    }
    if (n_tok <= 4u) {
        const uint32_t blocks = (out_dim + 7u) / 8u;
        spark_fp8_gemv_kernel<<<blocks, 256u, 0, ds4_current_stream()>>>(
            codes, scales, (const float *)x->ptr, (float *)out->ptr,
            n_tok, in_dim, out_dim);
        return cuda_ok(cudaGetLastError(), "Spark-X2.6 FP8 GEMV");
    }
    /* Prefill (n_tok > 4): vectorized FP8 -> BF16 unpack into the shared
     * single-matrix staging buffer + cuBLAS Tensor Core GEMM. */
    const uint64_t need_bytes = (uint64_t)out_dim * in_dim * sizeof(uint16_t);
    if (need_bytes <= (uint64_t)SPARK_FF * SPARK_EMBD * sizeof(uint16_t)) {
        if (!g_spark_dequant_bf16 || g_spark_dequant_bf16->bytes < need_bytes) {
            ds4_gpu_tensor_free(g_spark_dequant_bf16);
            g_spark_dequant_bf16 = ds4_gpu_tensor_alloc(
                (uint64_t)SPARK_FF * SPARK_EMBD * sizeof(uint16_t));
        }
        if (g_spark_dequant_bf16) {
            const uint64_t vectors = (uint64_t)out_dim * in_dim / 8u;
            spark_fp8_dequant_bf16_kernel<<<(uint32_t)((vectors + 255u) / 256u),
                256u, 0, ds4_current_stream()>>>(
                (__nv_bfloat16 *)g_spark_dequant_bf16->ptr, codes, scales, vectors);
            if (!cuda_ok(cudaGetLastError(), "Spark-X2.6 FP8->BF16 dequant")) {
                return 0;
            }
            return ds4_gpu_v41_matmul_bf16_dev_tensor(
                out, g_spark_dequant_bf16, in_dim, out_dim, x, n_tok);
        }
    }
    const dim3 grid((out_dim + 63u) / 64u, (n_tok + 63u) / 64u);
    spark_fp8_gemm_kernel<<<grid, 256u, 0, ds4_current_stream()>>>(
        codes, scales, (const float *)x->ptr, (float *)out->ptr,
        n_tok, in_dim, out_dim);
    return cuda_ok(cudaGetLastError(), "Spark-X2.6 FP8 GEMM");
}

extern "C" int ds4_gpu_spark_kv_store_fp8(ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *positions, uint32_t rows, uint32_t width,
        uint32_t kv_heads, uint32_t head_dim, uint32_t ring,
        uint32_t capacity) {
    if (!dst || !k || !v || !positions || !rows || !width || !kv_heads ||
        !head_dim || !capacity || width != kv_heads * head_dim) {
        return 0;
    }
    const uint64_t row_bytes = 2ull * width + 2ull * kv_heads * sizeof(uint16_t);
    if (dst->bytes < (uint64_t)capacity * row_bytes ||
        k->bytes < (uint64_t)rows * width * sizeof(float) ||
        v->bytes < (uint64_t)rows * width * sizeof(float) ||
        positions->bytes < (uint64_t)rows * sizeof(uint32_t)) {
        return 0;
    }
    spark_kv_store_fp8_kernel<<<dim3(rows, kv_heads), 256u, 0,
        ds4_current_stream()>>>((uint8_t *)dst->ptr, (const float *)k->ptr,
        (const float *)v->ptr, (const uint32_t *)positions->ptr, rows,
        kv_heads, head_dim, ring);
    return cuda_ok(cudaGetLastError(), "Spark-X2.6 FP8 KV store");
}

extern "C" int ds4_gpu_spark_attn_fp8(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *positions, uint32_t rows, uint32_t first,
        uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t window,
        uint32_t ring, uint32_t keys, uint64_t row_bytes) {
    if (!out || !q || !kv || !positions || !rows || !heads || !kv_heads ||
        !dim || heads % kv_heads) {
        return 0;
    }
    const uint64_t jobs = (uint64_t)rows * heads;
    if (out->bytes < jobs * dim * sizeof(float) ||
        q->bytes < jobs * dim * sizeof(float) ||
        positions->bytes < (uint64_t)rows * sizeof(uint32_t)) {
        return 0;
    }
    static int scalar_only = -1;
    if (scalar_only < 0) {
        const char *env = getenv("DS4_SPARK2_5_ATTN_SCALAR");
        scalar_only = (env && env[0] == '1') ? 1 : 0;
    }
    if (!scalar_only && dim == 256u && heads == 16u && kv_heads == 4u) {
        if (rows == 1u) {
            spark_attn_decode_fp8_kernel<<<heads, 256u, 0, ds4_current_stream()>>>(
                (float *)out->ptr, (const float *)q->ptr,
                (const uint8_t *)kv->ptr, first, heads, kv_heads,
                window, ring, keys, row_bytes);
            return cuda_ok(cudaGetLastError(), "Spark-X2.6 FP8 decode attention");
        }
        /* Prefill / multi-row verify: Tensor-Core m16n8k16 MMA attention. */
        const dim3 grid(kv_heads, (rows + 31u) / 32u);
        const uint32_t smem = 2u * 32u * 256u * (uint32_t)sizeof(__half);
        spark_attn_mma_fp8_kernel<<<grid, 256u, smem, ds4_current_stream()>>>(
            (float *)out->ptr, (const float *)q->ptr,
            (const uint8_t *)kv->ptr, rows, first, heads, kv_heads,
            window, ring, keys, row_bytes);
        return cuda_ok(cudaGetLastError(), "Spark-X2.6 FP8 Tensor-Core MMA attention");
    }
    spark_attn_fp8_kernel<<<(uint32_t)((jobs + 3u) / 4u), 128u, 0,
        ds4_current_stream()>>>((float *)out->ptr, (const float *)q->ptr,
        (const uint8_t *)kv->ptr, rows, first, heads, kv_heads, dim,
        window, ring, keys, row_bytes);
    return cuda_ok(cudaGetLastError(), "Spark-X2.6 FP8 attention");
}

/* Host-side debug decode of the first vectors of one matrix, printed beside
 * the restore's dump (DS4_SPARK2_5_VQ_DEBUG=1). Reads the host map, the same
 * bytes the restore kernel sees. */
static void spark2_5_vq_debug_host(const struct spark2_5_stream &a,
        const struct spark2_5_stream &b, const uint8_t *book,
        const uint8_t *book2, const uint8_t *res, const uint8_t *res_book,
        uint32_t rows, uint32_t cols) {
    struct spark_vq_stream s1 = {a.gain, a.main, a.plane};
    struct spark_vq_stream s2 = {b.gain, b.main, b.plane};
    fprintf(stderr, "ds4: [spark vq debug] rows=%u cols=%u res=%d stage2=%d\n",
            rows, cols, res ? 1 : 0, b.main ? 1 : 0);
    for (uint32_t v = 0; v < 8u; v++) {
        float w[8];
        spark_vq_decode(book, book2, res_book, s1, s2, res, v, 0, w);
        fprintf(stderr, "ds4: [spark vq debug] v=%u %f %f %f %f %f %f %f %f\n",
                v, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
    }
    /* Which terms produce the image's value: the four combinations for v=0. */
    struct spark_vq_stream no_s2 = {NULL, NULL, NULL};
    float w[8];
    spark_vq_decode(book, book2, res_book, s1, no_s2, NULL, 0, 0, w);
    fprintf(stderr, "ds4: [spark vq debug] v0 stage1-only        %f\n", w[0]);
    spark_vq_decode(book, book2, res_book, s1, no_s2, res, 0, 0, w);
    fprintf(stderr, "ds4: [spark vq debug] v0 stage1+residual    %f\n", w[0]);
    spark_vq_decode(book, book2, res_book, s1, s2, NULL, 0, 0, w);
    fprintf(stderr, "ds4: [spark vq debug] v0 stage1+stage2      %f\n", w[0]);
    spark_vq_decode(book, book2, res_book, s1, s2, res, 0, 0, w);
    fprintf(stderr, "ds4: [spark vq debug] v0 all three          %f\n", w[0]);
    /* Intermediates for v=0, so a mismatch can be traced to one read. */
    uint32_t i1 = (uint32_t)s1.main[0] | ((uint32_t)(s1.main[1] & 15u) << 8);
    i1 |= ((uint32_t)(s1.plane[0] & 1u)) << 12;
    uint32_t i2 = (uint32_t)s2.main[0] | ((uint32_t)(s2.main[1] & 15u) << 8);
    i2 |= ((uint32_t)(s2.plane[0] & 1u)) << 12;
    __nv_fp8_e4m3 c1, c2, r;
    c1.__x = book[i1 * 8u];
    c2.__x = book2[i2 * 8u];
    r.__x = res_book[(uint32_t)res[0] * 8u];
    const float g1 = __half2float(((const __half *)s1.gain)[0]);
    const float g2 = __half2float(((const __half *)s2.gain)[0]);
    fprintf(stderr,
        "ds4: [spark vq debug] v0 g1=%f g2=%f i1=%u i2=%u r8=%u c1=%f c2=%f r=%f\n",
        g1, g2, i1, i2, (uint32_t)res[0], (float)c1, (float)c2, (float)r);
    /* The stage-2 term alone, 8 vectors: a factor or a permutation against
     * the tool's stage decode says which read is wrong. */
    for (uint32_t v = 0; v < 8u; v++) {
        float only[8];
        spark_vq_decode(book, book2, res_book, s1, s2, NULL, v, 0, only);
        float base[8];
        spark_vq_decode(book, book2, res_book, s1, no_s2, NULL, v, 0, base);
        fprintf(stderr,
            "ds4: [spark vq debug] stage2 v=%u %f %f %f %f %f %f %f %f | gain2=%f main2[0..2]=%02x %02x %02x plane2[0]=%02x\n",
            v, only[0] - base[0], only[1] - base[1], only[2] - base[2],
            only[3] - base[3], only[4] - base[4], only[5] - base[5],
            only[6] - base[6], only[7] - base[7],
            __half2float(((const __half *)s2.gain)[v / (10240u / 8u)]),
            s2.main[(v / 2u) * 3u], s2.main[(v / 2u) * 3u + 1u],
            s2.main[(v / 2u) * 3u + 2u], s2.plane[v / 8u]);
    }
}

/* Resolve one matrix's streams into the artifact arena. The headers are read
 * from the host map (the arena is a byte copy, so every offset is the map's),
 * and the pointers handed to the kernel are device addresses. */
extern "C" int ds4_gpu_spark_vq_resolve(ds4_spark_vq_ref *ref,
        const void *map, const void *arena, const void *vq, uint64_t vq_bytes,
        const void *vq2, uint64_t vq2_bytes, const void *book, const void *book2,
        const void *res, uint64_t res_bytes, const void *res_book,
        uint32_t rows, uint32_t cols) {
    struct spark2_5_stream a, b;
    const uint8_t *r8 = NULL;
    memset(&b, 0, sizeof(b));
    if (!ref || !map || !arena || !book || !rows || !cols ||
        cols % SPARK_VECTOR || !spark2_5_stream_open(vq, vq_bytes, rows, cols, &a) ||
        !spark2_5_res_open(res, res_bytes, rows, cols, &r8)) {
        return 0;
    }
    if (vq2 && !spark2_5_stream_open(vq2, vq2_bytes, rows, cols, &b)) {
        return 0;
    }
    const char *dbgenv = getenv("DS4_SPARK2_5_VQ_DEBUG");
    static int hdr_printed;
    if (dbgenv && dbgenv[0] == '1' && !hdr_printed) {
        hdr_printed = 1;
        const uint8_t *h1 = (const uint8_t *)vq, *h2 = (const uint8_t *)vq2;
        fprintf(stderr, "ds4: [spark vq debug] stage1 header:");
        for (uint32_t i = 0; i < 64u; i++) { fprintf(stderr, " %02x", h1[i]); }
        fprintf(stderr, "\nds4: [spark vq debug] stage2 header:");
        for (uint32_t i = 0; h2 && i < 64u; i++) { fprintf(stderr, " %02x", h2[i]); }
        fprintf(stderr, "\n");
    }
    /* Integer arithmetic: the two allocations are unrelated objects, so a
     * pointer difference between them would be undefined. */
    const uintptr_t delta = (uintptr_t)arena - (uintptr_t)map;
#define SPARK_VQ_AT(p) ((const void *)((uintptr_t)(p) + delta))
    ref->gain = SPARK_VQ_AT(a.gain);
    ref->main = SPARK_VQ_AT(a.main);
    ref->plane = SPARK_VQ_AT(a.plane);
    ref->gain2 = vq2 ? SPARK_VQ_AT(b.gain) : NULL;
    ref->main2 = vq2 ? SPARK_VQ_AT(b.main) : NULL;
    ref->plane2 = vq2 ? SPARK_VQ_AT(b.plane) : NULL;
    ref->book = SPARK_VQ_AT(book);
    ref->book2 = book2 ? SPARK_VQ_AT(book2) : NULL;
    /* r8 is the residual index plane: a GVR1 plane's indices begin at
     * header + 32, not at the blob start (spark2_5_res_open). */
    ref->res = SPARK_VQ_AT(r8);
    ref->res_book = SPARK_VQ_AT(res_book);
#undef SPARK_VQ_AT
    ref->rows = rows;
    ref->cols = cols;
    const char *dbg = getenv("DS4_SPARK2_5_VQ_DEBUG");
    static int printed;
    if (dbg && dbg[0] == '1' && rows == 2560u && cols == 10240u && !printed) {
        printed = 1;
        spark2_5_vq_debug_host(a, b, (const uint8_t *)book,
            (const uint8_t *)book2, r8, (const uint8_t *)res_book, rows, cols);
    }
    return 1;
}

/* One VQ projection decoded and multiplied without a BF16 copy of W
 * (cuda/spark2_5_vq_gemm.cuh). */
extern "C" int ds4_gpu_spark_vq_linear(ds4_gpu_tensor *out,
        const ds4_spark_vq_ref *w, const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !w || !x || !n_tok || !w->rows || !w->cols ||
        out->bytes < (uint64_t)n_tok * w->rows * sizeof(float) ||
        x->bytes < (uint64_t)n_tok * w->cols * sizeof(float)) {
        return 0;
    }
    const struct spark_vq_stream main_stream = {(const uint8_t *)w->gain,
        (const uint8_t *)w->main, (const uint8_t *)w->plane};
    const struct spark_vq_stream stage2_stream = {(const uint8_t *)w->gain2,
        (const uint8_t *)w->main2, (const uint8_t *)w->plane2};
    const uint8_t *res = (const uint8_t *)w->res;
    return spark_vq_gemm_launch((const uint8_t *)w->book,
        (const uint8_t *)w->book2, (const uint8_t *)w->res_book, main_stream,
        stage2_stream, res, n_tok, w->cols, w->rows, (const float *)x->ptr,
        (float *)out->ptr);
}

/* One-shot device copy of the mapped artifact for the fused path. */
extern "C" int ds4_gpu_spark_codes_copy(ds4_gpu_tensor *dst, const void *src,
        uint64_t bytes) {
    if (!dst || !src || !bytes || dst->bytes < bytes) { return 0; }
    if (!cuda_ok(cudaMemcpyAsync(dst->ptr, src, (size_t)bytes,
                                 cudaMemcpyHostToDevice, ds4_current_stream()),
                 "Spark-X2.6 artifact copy")) {
        return 0;
    }
    return cuda_ok(cudaStreamSynchronize(ds4_current_stream()),
                   "Spark-X2.6 artifact copy sync");
}

/* Device bytes the fused path holds: the artifact's mapped span. */
extern "C" uint64_t ds4_gpu_spark_codes_bytes(uint64_t model_bytes) {
    return model_bytes;
}

extern "C" const void *ds4_gpu_spark_codes_base(const ds4_gpu_tensor *codes) {
    return codes ? codes->ptr : NULL;
}

/* Diagnostic: decode a whole matrix with the fused path's own arithmetic, so
 * a parity run can diff it against the restore (DS4_SPARK2_5_DUMP_WEIGHT). */
__global__ void spark_vq_probe_kernel(const uint8_t *book, const uint8_t *book2,
        const uint8_t *res_book, const spark_vq_stream s1,
        const spark_vq_stream s2, const uint8_t *res, uint32_t rows,
        uint32_t cols, __nv_bfloat16 *out) {
    const uint64_t v = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= (uint64_t)rows * cols / SPARK_VQG_VECTOR) { return; }
    const uint32_t o = (uint32_t)(v / (cols / SPARK_VQG_VECTOR));
    float w[8];
    spark_vq_decode(book, book2, res_book, s1, s2, res, (uint32_t)v, o, w);
    for (uint32_t d = 0; d < 8u; d++) {
        out[v * 8u + d] = __float2bfloat16_rn(w[d]);
    }
}

extern "C" int ds4_gpu_spark_vq_probe(ds4_gpu_tensor *out,
        const ds4_spark_vq_ref *w) {
    if (!out || !w ||
        out->bytes < (uint64_t)w->rows * w->cols * sizeof(uint16_t)) {
        return 0;
    }
    const struct spark_vq_stream s1 = {(const uint8_t *)w->gain,
        (const uint8_t *)w->main, (const uint8_t *)w->plane};
    const struct spark_vq_stream s2 = {(const uint8_t *)w->gain2,
        (const uint8_t *)w->main2, (const uint8_t *)w->plane2};
    const uint64_t vectors = (uint64_t)w->rows * w->cols / SPARK_VQG_VECTOR;
    spark_vq_probe_kernel<<<(uint32_t)((vectors + 255u) / 256u), 256, 0,
        ds4_current_stream()>>>((const uint8_t *)w->book,
        (const uint8_t *)w->book2, (const uint8_t *)w->res_book, s1, s2,
        (const uint8_t *)w->res, w->rows, w->cols, (__nv_bfloat16 *)out->ptr);
    return cuda_ok(cudaGetLastError(), "Spark-X2.6 fused decode probe");
}
