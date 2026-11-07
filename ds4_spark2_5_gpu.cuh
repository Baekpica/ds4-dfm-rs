/* Spark-X2.6 CUDA entries. Included by ds4_cuda.cu beside the other family
 * graphs; the two kernels live in cuda/spark2_5.cuh. */

#include "ds4_spark2_5_vq.h"
#include "cuda/spark2_5.cuh"

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
