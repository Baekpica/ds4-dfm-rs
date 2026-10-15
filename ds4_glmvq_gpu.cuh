/* GVQ1 execution adapter; included once by ds4_cuda.cu. Weight pointers
 * resolve before capture, and each session owns bounded GEMM scratch. */
#include "ds4_glmvq.h"
#include "cuda/glm_vq_linear.cuh"
#include "cuda/glm_vq_kda.cuh"
#include "cuda/glm_vq_attention.cuh"

enum { GVQ_TYPE_F32 = 0, GVQ_TYPE_BF16 = 30,
       GVQ_TYPE_FP8 = 0x47514638, GVQ_TYPE_BOOK = 0x47564342, GVQ_TYPE_VQ = 0x47565138,
       GVQ_EXPERTS = 288, GVQ_USED = 8, GVQ_HIDDEN = 4096, GVQ_FFN = 2048, GVQ_WORDS = 8192 };
static constexpr uint64_t GVQ_CUBLAS_WORKSPACE = 32ull * 1024 * 1024;

struct ds4_gvq_model {
    std::vector<glmvq_source> source;
    std::vector<ds4_gvq_record> records;
    ds4_gpu_tensor *device = nullptr;
};

struct ds4_gvq_ctx {
    const ds4_gvq_model *model;
    ds4_gpu_tensor *scratch, *workspace, *pairs;
    cublasHandle_t handle;
    cudaStream_t stream;
    std::vector<int32_t> routes;
    std::vector<uint32_t> grouped;
};

static bool gvq_span(uint64_t off, uint64_t bytes, uint64_t size) {
    return off <= size && bytes <= size - off;
}

static uint64_t gvq_u64(const uint8_t *data, size_t offset) {
    uint64_t value;
    memcpy(&value, data + offset, sizeof(value));
    return value;
}

static uint32_t gvq_u32(const uint8_t *data, size_t offset) {
    uint32_t value;
    memcpy(&value, data + offset, sizeof(value));
    return value;
}

static bool gvq_link(ds4_gvq_model *plan, const void *map, uint32_t index) {
    const ds4_gvq_record &r = plan->records[index];
    glmvq_source &s = plan->source[index];
    if (r.type == GVQ_TYPE_FP8) {
        if (r.scale >= plan->source.size()) { return false; }
        const ds4_gvq_record &scale = plan->records[r.scale];
        if (r.bytes != (uint64_t)r.rows * r.cols || scale.type != GVQ_TYPE_F32 ||
            r.cols % GLMVQ_SCALE_BLOCK || r.rows % GLMVQ_SCALE_BLOCK ||
            scale.rows != r.rows / GLMVQ_SCALE_BLOCK || scale.cols != r.cols / GLMVQ_SCALE_BLOCK ||
            scale.bytes != (uint64_t)scale.rows * scale.cols * sizeof(float)) { return false; }
        s.storage = GLMVQ_WEIGHT_FP8;
        s.scale = (const float *)plan->source[r.scale].data;
        return true;
    }
    if (r.type != GVQ_TYPE_VQ) { return true; }
    if (r.book >= plan->source.size() || r.bytes < 128) { return false; }
    const uint8_t *h = (const uint8_t *)map + r.offset;
    const ds4_gvq_record &book = plan->records[r.book];
    const uint32_t precision = gvq_u32(h, 20), words = gvq_u32(h, 16);
    const uint64_t gain = gvq_u64(h, 32), main = gvq_u64(h, 40), plane = gvq_u64(h, 48);
    const uint64_t elements = (uint64_t)r.rows * r.cols;
    if (memcmp(h, "GVQ1", 4) || h[4] != 1 || h[5] || h[6] != 13 || h[7] != GLMVQ_VECTOR ||
        gvq_u32(h, 8) != r.rows || gvq_u32(h, 12) != r.cols ||
        precision != GLMVQ_E4M3 || words != GVQ_WORDS || book.type != GVQ_TYPE_BOOK ||
        book.bytes != (uint64_t)words * GLMVQ_VECTOR || gvq_u64(h, 56) != r.bytes ||
        !gvq_span(gain, (uint64_t)r.rows * sizeof(__half), r.bytes) ||
        !gvq_span(main, elements * 3 / 16, r.bytes) || !plane ||
        !gvq_span(plane, elements / 64, r.bytes)) { return false; }
    // Only the published uniform8192 E4M3 model contract reaches this adapter.
    const uint8_t *data = (const uint8_t *)s.data;
    s.storage = GLMVQ_WEIGHT_VQ;
    s.vq = {data + main, data + plane, plan->source[r.book].data,
        (const __half *)(data + gain), r.rows, r.cols, GLMVQ_E4M3};
    return true;
}

extern "C" void ds4_gpu_gvq_unbind(ds4_gvq_model *model) {
    if (!model) { return; }
    ds4_gpu_tensor_free(model->device);
    delete model;
}

extern "C" ds4_gvq_model *ds4_gpu_gvq_bind(const void *map, uint64_t size,
        const ds4_gvq_record *records, uint32_t count) {
    if (!map || !records || !count) { return nullptr; }
    ds4_gvq_model *plan = new (std::nothrow) ds4_gvq_model;
    if (!plan) { return nullptr; }
    plan->records.assign(records, records + count);
    plan->source.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        const ds4_gvq_record &r = records[i];
        if (!r.bytes || !r.rows || !r.cols || !gvq_span(r.offset, r.bytes, size)) {
            ds4_gpu_gvq_unbind(plan);
            return nullptr;
        }
        const void *data = cuda_resolve_weight_ptr(map, r.offset, r.bytes, 0, "GLM VQ source");
        if (!data) { ds4_gpu_gvq_unbind(plan); return nullptr; }
        plan->source[i] = {data, nullptr, {}, r.rows, r.cols, GLMVQ_WEIGHT_BF16};
    }
    for (uint32_t i = 0; i < count; i++) {
        if (!gvq_link(plan, map, i)) { ds4_gpu_gvq_unbind(plan); return nullptr; }
    }
    const uint64_t bytes = (uint64_t)count * sizeof(glmvq_source);
    plan->device = ds4_gpu_tensor_alloc(bytes);
    if (!plan->device || !ds4_gpu_tensor_write(plan->device, 0, plan->source.data(), bytes)) {
        ds4_gpu_gvq_unbind(plan);
        return nullptr;
    }
    return plan;
}

extern "C" uint64_t ds4_gpu_gvq_workspace(uint32_t rows) {
    return GVQ_CUBLAS_WORKSPACE + (uint64_t)rows * GVQ_USED * sizeof(uint32_t);
}

extern "C" uint64_t ds4_gpu_gvq_linear_bytes(uint32_t rows, uint32_t cols,
        uint32_t tokens, uint32_t type, ds4_gvq_math math) {
    glmvq_source source = {};
    source.rows = rows; source.cols = cols;
    source.storage = type == GVQ_TYPE_BF16 ? GLMVQ_WEIGHT_BF16 : GLMVQ_WEIGHT_FP8;
    return glmvq_linear_bytes(source, tokens, (glmvq_compute)math);
}

extern "C" void ds4_gpu_gvq_free(ds4_gvq_ctx *ctx) {
    if (!ctx) { return; }
    if (ctx->handle) { (void)cublasDestroy(ctx->handle); }
    ds4_gpu_tensor_free(ctx->pairs);
    ds4_gpu_tensor_free(ctx->workspace);
    ds4_gpu_tensor_free(ctx->scratch);
    delete ctx;
}

extern "C" ds4_gvq_ctx *ds4_gpu_gvq_create(const ds4_gvq_model *model, uint64_t bytes, uint32_t rows) {
    if (!model || !bytes || !rows) { return nullptr; }
    ds4_gvq_ctx *ctx = new (std::nothrow) ds4_gvq_ctx{};
    if (!ctx) { return nullptr; }
    ctx->model = model;
    ctx->scratch = ds4_gpu_tensor_alloc(bytes);
    ctx->workspace = ds4_gpu_tensor_alloc(GVQ_CUBLAS_WORKSPACE);
    ctx->pairs = ds4_gpu_tensor_alloc((uint64_t)rows * GVQ_USED * sizeof(uint32_t));
    ctx->routes.resize((uint64_t)rows * GVQ_USED);
    ctx->grouped.resize((uint64_t)rows * GVQ_USED);
    if (!ctx->scratch || !ctx->workspace || !ctx->pairs || !cublas_ok(cublasCreate(&ctx->handle), "GLM VQ cuBLAS create") ||
        !cublas_ok(cublasSetMathMode(ctx->handle, CUBLAS_DEFAULT_MATH), "GLM VQ cuBLAS math") ||
        !cublas_ok(cublasSetWorkspace(ctx->handle, ctx->workspace->ptr, ctx->workspace->bytes), "GLM VQ cuBLAS workspace")) {
        ds4_gpu_gvq_free(ctx);
        return nullptr;
    }
    return ctx;
}

static bool gvq_stream(ds4_gvq_ctx *ctx) {
    const cudaStream_t stream = ds4_current_stream();
    if (ctx->stream == stream) { return true; }
    if (!cublas_ok(cublasSetStream(ctx->handle, stream), "GLM VQ cuBLAS stream") ||
        !cublas_ok(cublasSetWorkspace(ctx->handle, ctx->workspace->ptr, ctx->workspace->bytes), "GLM VQ cuBLAS workspace")) {
        return false;
    }
    ctx->stream = stream;
    return true;
}

static bool gvq_floats(const ds4_gpu_tensor *t, uint64_t count) {
    return t && count <= t->bytes / sizeof(float);
}

static const void *gvq_weight(const ds4_gvq_ctx *ctx, uint32_t index,
        uint32_t type, uint64_t count) {
    if (!ctx || index >= ctx->model->source.size()) { return nullptr; }
    const ds4_gvq_record &r = ctx->model->records[index];
    const uint32_t size = type == GVQ_TYPE_F32 ? sizeof(float) : sizeof(__nv_bfloat16);
    return r.type == type && count <= r.bytes / size ? ctx->model->source[index].data : nullptr;
}

extern "C" int ds4_gpu_gvq_linear(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t index,
        const ds4_gpu_tensor *input, uint32_t rows, ds4_gvq_math math) {
    if (!ctx || !rows || index >= ctx->model->source.size()) { return 0; }
    const glmvq_source &s = ctx->model->source[index];
    const uint32_t type = ctx->model->records[index].type;
    if ((type != GVQ_TYPE_BF16 && type != GVQ_TYPE_FP8 && type != GVQ_TYPE_VQ) ||
        !gvq_floats(input, (uint64_t)rows * s.cols) || !gvq_floats(out, (uint64_t)rows * s.rows) ||
        (math != DS4_GVQ_BF16 && math != DS4_GVQ_F32) || !gvq_stream(ctx)) { return 0; }
    return cublas_ok(glmvq_gemm(ctx->handle, s, (const float *)input->ptr, (float *)out->ptr,
        rows, (glmvq_compute)math, ctx->scratch->ptr, ctx->scratch->bytes, ctx->stream), "GLM VQ linear");
}

extern "C" int ds4_gpu_gvq_rms(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out,
        const ds4_gpu_tensor *input, const ds4_gpu_tensor *gate, uint32_t weight,
        uint32_t rows, uint32_t width, float eps, ds4_gvq_norm kind) {
    const uint64_t count = (uint64_t)rows * width;
    if (!rows || !width || !gvq_floats(out, count) || !gvq_floats(input, count) ||
        (kind == DS4_GVQ_GATED_RMS && !gvq_floats(gate, count))) { return 0; }
    const __nv_bfloat16 *w = kind == DS4_GVQ_PLAIN_RMS ? nullptr :
        (const __nv_bfloat16 *)gvq_weight(ctx, weight, GVQ_TYPE_BF16, width);
    if ((kind != DS4_GVQ_PLAIN_RMS && !w) || kind > DS4_GVQ_PLAIN_RMS) { return 0; }
    glmvq_rms<<<rows, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
        (const float *)input->ptr, w, gate ? (const float *)gate->ptr : nullptr,
        (float *)out->ptr, width, eps, (glmvq_norm)kind);
    return cuda_ok(cudaGetLastError(), "GLM VQ RMS");
}

extern "C" int ds4_gpu_gvq_split(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gpu_tensor *split,
        const ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual,
        uint32_t scale, uint32_t base, uint32_t rows, uint32_t width, uint32_t iterations, float eps) {
    const uint64_t count = (uint64_t)rows * width, coeff = (uint64_t)rows * GLMVQ_HC_MIX;
    const float *s = (const float *)gvq_weight(ctx, scale, GVQ_TYPE_F32, 3);
    const float *b = (const float *)gvq_weight(ctx, base, GVQ_TYPE_F32, GLMVQ_HC_MIX);
    if (!s || !b || !rows || !width || !iterations || !gvq_floats(out, count) ||
        !gvq_floats(residual, count * GLMVQ_HC) || !gvq_floats(split, coeff) || !gvq_floats(mix, coeff)) { return 0; }
    glmvq_hc_split<<<rows, GLMVQ_THREADS, 0, ds4_current_stream()>>>((const float *)mix->ptr,
        (const float *)residual->ptr, s, b, (float *)out->ptr, (float *)split->ptr, width, iterations, eps);
    return cuda_ok(cudaGetLastError(), "GLM VQ mHC split");
}

extern "C" int ds4_gpu_gvq_expand(ds4_gpu_tensor *out, const ds4_gpu_tensor *branch,
        const ds4_gpu_tensor *residual, const ds4_gpu_tensor *split, uint32_t rows, uint32_t width) {
    const uint64_t count = (uint64_t)rows * width * GLMVQ_HC;
    if (!rows || !width || !gvq_floats(out, count) || !gvq_floats(residual, count) ||
        !gvq_floats(branch, count / GLMVQ_HC) || !gvq_floats(split, (uint64_t)rows * GLMVQ_HC_MIX)) { return 0; }
    glmvq_hc_expand<<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
        (const float *)branch->ptr, (const float *)residual->ptr, (const float *)split->ptr,
        (float *)out->ptr, width, count);
    return cuda_ok(cudaGetLastError(), "GLM VQ mHC expand");
}

extern "C" int ds4_gpu_gvq_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up, uint64_t count, float limit) {
    if (!count || !gvq_floats(out, count) || !gvq_floats(gate, count) || !gvq_floats(up, count)) { return 0; }
    glmvq_swiglu<<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
        (const float *)gate->ptr, (const float *)up->ptr, (float *)out->ptr, count, limit);
    return cuda_ok(cudaGetLastError(), "GLM VQ SwiGLU");
}

extern "C" int ds4_gpu_gvq_round(ds4_gpu_tensor *values, uint64_t count) {
    if (!count || !gvq_floats(values, count)) { return 0; }
    glmvq_round_rows<<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
        (float *)values->ptr, count);
    return cuda_ok(cudaGetLastError(), "GLM VQ BF16 boundary");
}

extern "C" int ds4_gpu_gvq_control(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out,
        uint32_t weight, uint64_t count) {
    if (!ctx || weight >= ctx->model->source.size() || !count || !gvq_floats(out, count)) { return 0; }
    const uint32_t type = ctx->model->records[weight].type;
    if (type != GVQ_TYPE_BF16 && type != GVQ_TYPE_F32) { return 0; }
    const void *source = gvq_weight(ctx, weight, type, count);
    if (!source) { return 0; }
    const uint32_t blocks = (uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
    if (type == GVQ_TYPE_F32) {
        glmvq_convert<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>((const float *)source, (float *)out->ptr, count);
    } else {
        glmvq_convert<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>((const __nv_bfloat16 *)source, (float *)out->ptr, count);
    }
    return cuda_ok(cudaGetLastError(), "GLM VQ control copy");
}

extern "C" int ds4_gpu_gvq_unpack(ds4_gpu_tensor *out, const ds4_gpu_tensor *cache, uint64_t count) {
    if (!count || !gvq_floats(out, count) || !cache || count > cache->bytes / sizeof(__nv_bfloat16)) { return 0; }
    glmvq_convert<<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
        (const __nv_bfloat16 *)cache->ptr, (float *)out->ptr, count);
    return cuda_ok(cudaGetLastError(), "GLM VQ latent unpack");
}

extern "C" int ds4_gpu_gvq_store(ds4_gpu_tensor *cache, const ds4_gpu_tensor *input,
        uint32_t rows, uint32_t first, uint32_t capacity, uint32_t width) {
    const uint64_t count = (uint64_t)rows * width;
    if (ds4_capture_active() || !rows || !width || first > capacity || rows > capacity - first ||
        !cache || (uint64_t)capacity * width > cache->bytes / sizeof(__nv_bfloat16) ||
        !gvq_floats(input, count)) { return 0; }
    // Original latent KV retains BF16, unlike the MXQ compact FP16 cache.
    glmvq_convert<<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
        (const float *)input->ptr, (__nv_bfloat16 *)cache->ptr + (uint64_t)first * width, count);
    return cuda_ok(cudaGetLastError(), "GLM VQ BF16 latent store");
}

__global__ void gvq_positions(uint32_t *out, uint32_t first, uint32_t rows) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) { out[row] = first + row; }
}

extern "C" int ds4_gpu_gvq_positions(ds4_gpu_tensor *out, uint32_t first, uint32_t rows) {
    if (!out || !rows || rows > out->bytes / sizeof(uint32_t) || first > UINT32_MAX - rows) { return 0; }
    gvq_positions<<<(rows + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
        (uint32_t *)out->ptr, first, rows);
    return cuda_ok(cudaGetLastError(), "GLM VQ positions");
}

__global__ void gvq_router(int32_t *selected, float *weights, const float *logits,
        const float *bias, float scale) {
    __shared__ float prob[GVQ_EXPERTS], score[GVQ_EXPERTS];
    const uint64_t row = (uint64_t)blockIdx.x * GVQ_EXPERTS;
    for (uint32_t e = threadIdx.x; e < GVQ_EXPERTS; e += blockDim.x) {
        prob[e] = glmvq_sigmoid(logits[row + e]);
        score[e] = glmvq_add(prob[e], bias[e]);
    }
    __syncthreads();
    if (threadIdx.x) { return; }
    const uint64_t offset = (uint64_t)blockIdx.x * GVQ_USED;
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < GVQ_USED; slot++) {
        uint32_t best = 0;
        for (uint32_t e = 1; e < GVQ_EXPERTS; e++) {
            if (score[e] > score[best]) { best = e; }
        }
        selected[offset + slot] = (int32_t)best;
        weights[offset + slot] = prob[best];
        sum = glmvq_add(sum, prob[best]);
        score[best] = -INFINITY;
    }
    // Source router: normalize uncorrected sigmoid scores with +1e-20,
    // then apply routed_scaling_factor (modeling_glm5_next.py:158-184).
    sum = glmvq_add(sum, 1.0e-20f);
    for (uint32_t slot = 0; slot < GVQ_USED; slot++) {
        weights[offset + slot] = glmvq_mul(__fdiv_rn(weights[offset + slot], sum), scale);
    }
}

extern "C" int ds4_gpu_gvq_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected,
        ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits, uint32_t bias,
        uint32_t rows, float scale) {
    const float *b = (const float *)gvq_weight(ctx, bias, GVQ_TYPE_F32, GVQ_EXPERTS);
    if (!b || !rows || !selected || (uint64_t)rows * GVQ_USED > selected->bytes / sizeof(int32_t) ||
        !gvq_floats(weights, (uint64_t)rows * GVQ_USED) || !gvq_floats(logits, (uint64_t)rows * GVQ_EXPERTS)) { return 0; }
    gvq_router<<<rows, GLMVQ_THREADS, 0, ds4_current_stream()>>>((int32_t *)selected->ptr,
        (float *)weights->ptr, (const float *)logits->ptr, b, scale);
    return cuda_ok(cudaGetLastError(), "GLM VQ router");
}

extern "C" uint64_t ds4_gpu_gvq_attn_bytes(uint32_t rows, uint32_t keys, uint32_t heads, uint32_t dim) {
    return glmvq_attn_bytes(rows, keys, heads, dim).bytes;
}

extern "C" int ds4_gpu_gvq_attn(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out,
        const ds4_gpu_tensor *query, const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *positions, const ds4_gpu_tensor *selected,
        uint32_t stride, uint32_t rows, uint32_t keys, uint32_t heads, uint32_t dim) {
    const uint64_t nq = (uint64_t)rows * heads * dim, nkv = (uint64_t)keys * heads * 2 * dim;
    if (!ctx || !gvq_floats(out, nq) || !gvq_floats(query, nq) || !gvq_floats(kv, nkv) ||
        !positions || rows > positions->bytes / sizeof(uint32_t) ||
        (selected && (!stride || (uint64_t)rows * stride > selected->bytes / sizeof(int32_t))) ||
        !gvq_stream(ctx)) { return 0; }
    return cublas_ok(glmvq_attention(ctx->handle, (const float *)query->ptr, (const float *)kv->ptr,
        (float *)out->ptr, (const uint32_t *)positions->ptr, selected ? (const int32_t *)selected->ptr : nullptr,
        stride, rows, keys, heads, dim, ctx->scratch->ptr, ctx->scratch->bytes, ctx->stream), "GLM VQ attention");
}

extern "C" int ds4_gpu_gvq_kda(ds4_gpu_tensor *out, ds4_gpu_tensor *state,
        ds4_gpu_tensor *q_tail, ds4_gpu_tensor *k_tail, ds4_gpu_tensor *v_tail,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *forget, const ds4_gpu_tensor *beta,
        const ds4_gpu_tensor *q_weight, const ds4_gpu_tensor *k_weight,
        const ds4_gpu_tensor *v_weight, const ds4_gpu_tensor *a_log,
        const ds4_gpu_tensor *dt_bias, uint32_t rows, uint32_t heads, float lower_bound) {
    const uint64_t width = (uint64_t)heads * GLMVQ_KDA_DIM, count = (uint64_t)rows * width;
    const uint64_t tail = width * GLMVQ_CONV;
    if (!rows || !heads || !gvq_floats(out, count) || !gvq_floats(q, count) ||
        !gvq_floats(k, count) || !gvq_floats(v, count) || !gvq_floats(forget, count) ||
        !gvq_floats(beta, (uint64_t)rows * heads) || !gvq_floats(state, width * GLMVQ_KDA_DIM) ||
        !gvq_floats(q_tail, tail) || !gvq_floats(k_tail, tail) || !gvq_floats(v_tail, tail) ||
        !gvq_floats(q_weight, tail) || !gvq_floats(k_weight, tail) || !gvq_floats(v_weight, tail) ||
        !gvq_floats(a_log, heads) || !gvq_floats(dt_bias, width)) { return 0; }
    glmvq_kda<<<heads, GLMVQ_KDA_DIM, 0, ds4_current_stream()>>>((float *)out->ptr,
        (float *)state->ptr, (float *)q_tail->ptr, (float *)k_tail->ptr, (float *)v_tail->ptr,
        (const float *)q->ptr, (const float *)k->ptr, (const float *)v->ptr,
        (const float *)forget->ptr, (const float *)beta->ptr,
        (const float *)q_weight->ptr, (const float *)k_weight->ptr, (const float *)v_weight->ptr,
        (const float *)a_log->ptr, (const float *)dt_bias->ptr, rows, heads, lower_bound);
    return cuda_ok(cudaGetLastError(), "GLM VQ KDA");
}

__global__ void gvq_gather(float *output, const float *input, const uint32_t *pairs,
        uint32_t offset, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint32_t token = pairs[offset + i / GVQ_HIDDEN] / GVQ_USED;
    output[i] = input[(uint64_t)token * GVQ_HIDDEN + i % GVQ_HIDDEN];
}

__global__ void gvq_scatter(float *output, const float *input, const float *weights,
        const uint32_t *pairs, uint32_t offset, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint32_t pair = pairs[offset + i / GVQ_HIDDEN];
    const uint64_t target = (uint64_t)(pair / GVQ_USED) * GVQ_HIDDEN + i % GVQ_HIDDEN;
    const float value = glmvq_round(glmvq_mul(input[i], weights[pair]));
    // One expert has at most one contribution per token; launches are ordered.
    output[target] = glmvq_round(glmvq_add(output[target], value));
}

extern "C" int ds4_gpu_gvq_moe(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out,
        ds4_gpu_tensor *gathered, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        const uint32_t *matrices, uint32_t rows, float clamp) {
    const uint64_t pairs = (uint64_t)rows * GVQ_USED, count = (uint64_t)rows * GVQ_HIDDEN;
    if (!ctx || !rows || !matrices || pairs > ctx->routes.size() || !selected ||
        pairs > selected->bytes / sizeof(int32_t) || !gvq_floats(weights, pairs) ||
        !gvq_floats(out, count) || !gvq_floats(input, count) || !gvq_floats(gathered, count) ||
        !gvq_floats(down, count) || !gvq_floats(gate, (uint64_t)rows * GVQ_FFN) ||
        !gvq_floats(up, (uint64_t)rows * GVQ_FFN) || !gvq_floats(mid, (uint64_t)rows * GVQ_FFN)) { return 0; }
    // The numerical baseline groups actual routes on the host. Captured
    // decode needs the later GPU-selected dispatch, never a captured readback.
    if (ds4_capture_active()) { return 0; }
    if (!ds4_gpu_tensor_read(selected, 0, ctx->routes.data(), pairs * sizeof(int32_t))) { return 0; }
    uint32_t counts[GVQ_EXPERTS] = {}, offsets[GVQ_EXPERTS + 1] = {}, cursor[GVQ_EXPERTS];
    for (uint64_t i = 0; i < pairs; i++) {
        const int32_t expert = ctx->routes[i];
        if (expert < 0 || expert >= GVQ_EXPERTS) { return 0; }
        for (uint64_t j = i - i % GVQ_USED; j < i; j++) {
            if (ctx->routes[j] == expert) { return 0; }
        }
        counts[expert]++;
    }
    for (uint32_t e = 0; e < GVQ_EXPERTS; e++) {
        offsets[e + 1] = offsets[e] + counts[e];
        cursor[e] = offsets[e];
    }
    for (uint64_t i = 0; i < pairs; i++) { ctx->grouped[cursor[ctx->routes[i]]++] = (uint32_t)i; }
    if (!ds4_gpu_tensor_write(ctx->pairs, 0, ctx->grouped.data(), pairs * sizeof(uint32_t)) ||
        !ds4_gpu_tensor_fill_f32(out, 0.0f, count)) { return 0; }
    for (uint32_t e = 0; e < GVQ_EXPERTS; e++) {
        const uint32_t n = counts[e];
        if (!n) { continue; }
        const uint64_t elements = (uint64_t)n * GVQ_HIDDEN;
        const uint32_t blocks = (uint32_t)((elements + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
        gvq_gather<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>((float *)gathered->ptr,
            (const float *)input->ptr, (const uint32_t *)ctx->pairs->ptr, offsets[e], elements);
        if (!cuda_ok(cudaGetLastError(), "GLM VQ expert gather") ||
            !ds4_gpu_gvq_linear(ctx, gate, matrices[e], gathered, n, DS4_GVQ_BF16) ||
            !ds4_gpu_gvq_linear(ctx, up, matrices[GVQ_EXPERTS + e], gathered, n, DS4_GVQ_BF16) ||
            !ds4_gpu_gvq_swiglu(mid, gate, up, (uint64_t)n * GVQ_FFN, clamp) ||
            !ds4_gpu_gvq_linear(ctx, down, matrices[2 * GVQ_EXPERTS + e], mid, n, DS4_GVQ_BF16)) { return 0; }
        gvq_scatter<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>((float *)out->ptr,
            (const float *)down->ptr, (const float *)weights->ptr, (const uint32_t *)ctx->pairs->ptr,
            offsets[e], elements);
        if (!cuda_ok(cudaGetLastError(), "GLM VQ expert scatter")) { return 0; }
    }
    return 1;
}
