/* Included by the native CUDA backend; no host or Rust ABI state escapes. */
#include "cuda/step37_primitives.cuh"

/* Keep the original norm reduction. Q/K/V and the FFN consumers reuse one
 * exact D4 quantization; scalar/verify widths retain their existing path. */
extern "C" int ds4_gpu_step37_norm(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t offset,
        uint32_t width, uint32_t rows, float eps) {
    if (!ds4_gpu_exaone_rms_norm_tensor(out, x, map, size, offset, width, rows, eps)) { return 0; }
    if (width == 4096 && rows >= 64 && !getenv("DS4_STEP37_NO_Q8_REUSE")) {
        cuda_norm_emit_q8(out, rows, width);
    }
    return 1;
}

extern "C" int ds4_gpu_step37_sum(ds4_gpu_tensor *out, const ds4_gpu_tensor *down,
        const ds4_gpu_tensor *weights, uint32_t width, uint32_t rows) {
    const uint64_t count = (uint64_t)width * rows;
    if (!out || !down || !weights || !width || !rows || count > INT_MAX ||
        out->bytes < count * sizeof(float) || down->bytes < count * 8 * sizeof(float) ||
        weights->bytes < (uint64_t)rows * 8 * sizeof(float)) { return 0; }
    step37_expert_sum<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)down->ptr, (const float *)weights->ptr, width, count);
    return cuda_ok(cudaGetLastError(), "Step37 expert sum");
}

extern "C" int ds4_gpu_step37_swiglu(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
        const ds4_gpu_tensor *weights, uint32_t width, uint32_t rows, float limit) {
    const uint64_t count = (uint64_t)width * rows;
    const uint64_t bytes = count * sizeof(float);
    if (!out || !gate || !up || !width || !rows || count > INT_MAX ||
        out->bytes < bytes || gate->bytes < bytes || up->bytes < bytes ||
        (weights && weights->bytes < (uint64_t)rows * sizeof(float)) ||
        !isfinite(limit) || limit < 0) { return 0; }
    step37_swiglu<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr,
        weights ? (const float *)weights->ptr : nullptr, width, count, limit);
    return cuda_ok(cudaGetLastError(), "Step37 SwiGLU");
}

extern "C" int ds4_gpu_step37_router(
        ds4_gpu_tensor *ids, ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits,
        const void *map, uint64_t size, uint64_t offset, uint32_t rows) {
    enum { EXPERTS = 288, USED = 8 };
    const uint64_t bias_bytes = EXPERTS * sizeof(float);
    if (!ids || !weights || !logits || !map || !rows || rows > INT_MAX ||
        offset > size || bias_bytes > size - offset ||
        ids->bytes < (uint64_t)rows * USED * sizeof(int) ||
        weights->bytes < (uint64_t)rows * USED * sizeof(float) ||
        logits->bytes < (uint64_t)rows * EXPERTS * sizeof(float)) { return 0; }
    const float *bias = (const float *)cuda_model_range_ptr(map, offset, bias_bytes, "step37 bias");
    if (!bias) { return 0; }
    step37_router<<<rows, 128, 2 * EXPERTS * sizeof(float), ds4_current_stream()>>>(
        (int *)ids->ptr, (float *)weights->ptr, (const float *)logits->ptr, bias);
    return cuda_ok(cudaGetLastError(), "Step37 router");
}

extern "C" int ds4_gpu_step37_rope(
        ds4_gpu_tensor *table, const ds4_gpu_tensor *frequency,
        const ds4_gpu_tensor *positions, uint32_t rotary, uint32_t rows) {
    const uint64_t count = (uint64_t)rows * (rotary / 2);
    if (!table || !frequency || !positions || !rows || count > INT_MAX ||
        (rotary != 64 && rotary != 128) || table->bytes < count * sizeof(float2) ||
        frequency->bytes < rotary / 2 * sizeof(float) ||
        positions->bytes < (uint64_t)rows * sizeof(uint32_t)) { return 0; }
    step37_rope_table<<<(count + 127) / 128, 128, 0, ds4_current_stream()>>>(
        (float2 *)table->ptr, (const float *)frequency->ptr,
        (const unsigned *)positions->ptr, rotary / 2, rows);
    return cuda_ok(cudaGetLastError(), "Step37 RoPE table");
}

extern "C" int ds4_gpu_step37_qk(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t offset,
        const ds4_gpu_tensor *table, uint32_t heads, uint32_t rotary, uint32_t rows) {
    const uint64_t count = (uint64_t)rows * heads * 128;
    const uint64_t norm_bytes = 128 * sizeof(float);
    if (!out || !x || !table || !map || !rows || count > INT_MAX ||
        (heads != 8 && heads != 64 && heads != 96) ||
        (rotary != 64 && rotary != 128) || offset > size || norm_bytes > size - offset ||
        out->bytes < count * sizeof(float) || x->bytes < count * sizeof(float) ||
        table->bytes < (uint64_t)rows * rotary * sizeof(float)) { return 0; }
    const float *norm = (const float *)cuda_model_range_ptr(map, offset, norm_bytes, "step37 QK norm");
    if (!norm) { return 0; }
    step37_qk_rope<<<rows * heads, 128, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)x->ptr, norm, (const float2 *)table->ptr, heads, rotary);
    return cuda_ok(cudaGetLastError(), "Step37 QK norm/RoPE");
}

extern "C" int ds4_gpu_step37_gate(
        ds4_gpu_tensor *values, const ds4_gpu_tensor *gate, uint32_t heads, uint32_t rows) {
    const uint64_t count = (uint64_t)rows * heads * 128;
    if (!values || !gate || !rows || count > INT_MAX || (heads != 64 && heads != 96) ||
        values->bytes < count * sizeof(float) ||
        gate->bytes < (uint64_t)rows * heads * sizeof(float)) { return 0; }
    step37_attn_gate<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)values->ptr, (const float *)gate->ptr, count);
    return cuda_ok(cudaGetLastError(), "Step37 attention gate");
}
