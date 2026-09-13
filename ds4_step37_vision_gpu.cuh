/* Bounded native entries for the pinned Step projector only. */
#include "cuda/step37_vision.cuh"

static const float *s37v_weights(const void *map, uint64_t size, uint64_t offset, uint64_t count) {
    if (!map || count > UINT64_MAX / sizeof(float) || offset > size ||
        count * sizeof(float) > size - offset) { return nullptr; }
    return (const float *)cuda_model_range_ptr(map, offset, count * sizeof(float), "Step vision F32");
}

extern "C" int ds4_gpu_step37_columns(ds4_gpu_tensor *out, const ds4_gpu_tensor *in,
        uint32_t edge, uint32_t channels) {
    const bool patch = channels == 3 && (edge == 504 || edge == 728);
    const bool down1 = channels == 1536 && (edge == 36 || edge == 52);
    const bool down2 = channels == 3072 && (edge == 18 || edge == 26);
    if (!out || !in || (!patch && !down1 && !down2)) { return 0; }
    const unsigned kernel = patch ? 14 : 3, output = patch ? edge / 14 : edge / 2;
    const uint64_t count = (uint64_t)output * output * channels * kernel * kernel;
    if (in->bytes < (uint64_t)edge * edge * channels * sizeof(float) ||
        out->bytes < count * sizeof(float)) { return 0; }
    s37v_im2col<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)in->ptr, edge, channels, kernel);
    return cuda_ok(cudaGetLastError(), "Step vision columns");
}

extern "C" int ds4_gpu_step37_position(ds4_gpu_tensor *hidden,
        const void *map, uint64_t size, uint64_t offset, uint32_t edge) {
    const uint64_t count = (uint64_t)edge * edge * 1536;
    if (!hidden || (edge != 36 && edge != 52) || hidden->bytes < count * sizeof(float)) { return 0; }
    const float *p = s37v_weights(map, size, offset, 52 * 52 * 1536);
    if (!p) { return 0; }
    s37v_position<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>((float *)hidden->ptr, p, edge);
    return cuda_ok(cudaGetLastError(), "Step vision position");
}

extern "C" int ds4_gpu_step37_vqkv(ds4_gpu_tensor *qkv,
        const void *map, uint64_t size, uint64_t offset, uint32_t edge) {
    const uint64_t count = (uint64_t)edge * edge * 3 * 1536;
    if (!qkv || (edge != 36 && edge != 52) || qkv->bytes < count * sizeof(float)) { return 0; }
    const float *p = s37v_weights(map, size, offset, 3 * 1536);
    if (!p) { return 0; }
    s37v_qkv<<<(count / 2 + 255) / 256, 256, 0, ds4_current_stream()>>>((float *)qkv->ptr, p, edge);
    return cuda_ok(cudaGetLastError(), "Step vision QKV");
}

extern "C" int ds4_gpu_step37_vgelu(ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t offset, uint32_t rows) {
    const uint64_t count = (uint64_t)rows * 8960;
    if (!x || (rows != 36 * 36 && rows != 52 * 52) || x->bytes < count * sizeof(float)) { return 0; }
    const float *p = s37v_weights(map, size, offset, 8960);
    if (!p) { return 0; }
    s37v_quick_gelu<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>((float *)x->ptr, p, count, 8960);
    return cuda_ok(cudaGetLastError(), "Step vision QuickGELU");
}

extern "C" int ds4_gpu_step37_vresidual(ds4_gpu_tensor *residual, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t bias, uint64_t scale, uint32_t rows) {
    const uint64_t count = (uint64_t)rows * 1536;
    if (!residual || !x || (rows != 36 * 36 && rows != 52 * 52) ||
        residual->bytes < count * sizeof(float) || x->bytes < count * sizeof(float)) { return 0; }
    const float *b = s37v_weights(map, size, bias, 1536), *s = s37v_weights(map, size, scale, 1536);
    if (!b || !s) { return 0; }
    s37v_residual<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)residual->ptr, (const float *)x->ptr, b, s, count);
    return cuda_ok(cudaGetLastError(), "Step vision scaled residual");
}
