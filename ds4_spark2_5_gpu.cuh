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
