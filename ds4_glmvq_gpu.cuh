/* GVQ1 execution adapter; included once by ds4_cuda.cu. Weight pointers
 * resolve before capture, and each session owns bounded GEMM scratch. */
#include "ds4_glmvq.h"
#include <type_traits>
#include "cuda/glm_vq_linear.cuh"
#include "cuda/glm_vq_q4m.cuh"
#include "cuda/glm_vq_q4p.cuh"
#include "cuda/glm_vq_kda.cuh"
#include "cuda/glm_vq_attention.cuh"
#include "cuda/glm_vq_index.cuh"
#include "cuda/glm_vq_mma.cuh"
#include "cuda/glm_vq_mma8.cuh"
#include "cuda/glm_vq_flash.cuh"
#include "cuda/glm_vq_decode.cuh"
#include "cuda/glm_vq_fp8.cuh"
#include "cuda/glm_vq_gemv.cuh"

enum { GVQ_TYPE_F32 = 0, GVQ_TYPE_Q4 = 12, GVQ_TYPE_BF16 = 30,
       GVQ_TYPE_FP8 = 0x47514638, GVQ_TYPE_BOOK = 0x47564342, GVQ_TYPE_VQ = 0x47565138,
       GVQ_TYPE_RES = 0x47565238,
       GVQ_EXPERTS = 288, GVQ_USED = 8, GVQ_HIDDEN = 4096, GVQ_FFN = 2048, GVQ_WORDS = 8192 };
static constexpr uint64_t GVQ_CUBLAS_WORKSPACE = 32ull * 1024 * 1024;
enum { GVQ_MMA_ROWS = 256, GVQ_MMA_TOKENS = 32, GVQ_MMA8_TOKENS = 64 };
// GVQ1 header bytes in front of every VQ matrix.
enum { GVQ_HEADER = 128 };
// Widest MTP verify forward (first token + three drafts).
enum { GVQ_VERIFY_ROWS = 4 };
static_assert(GVQ_VERIFY_ROWS == GLMVQ_VER_ROWS, "verify kernels cover every MTP verify width");
enum { GVQ_LATENT_MIN_KEYS = 512, GVQ_LATENT_MAX_KEYS = DS4_GVQ_CTX_MAX,
       GVQ_LATENT_HEADS = 64, GVQ_LATENT_DIM = 256, GVQ_LATENT_WIDTH = 512 };

struct ds4_gvq_model {
    std::vector<glmvq_source> source;
    std::vector<ds4_gvq_record> records;
    ds4_gpu_tensor *device = nullptr;
};

// Decode and verify widths chain their launches with programmatic dependent
// launch (DS4_GLMVQ_PDL=0 turns it off); prefill grids launch normally.
static glmvq_launch g_gvq_pdl = GLMVQ_LAUNCH_PLAIN;
static glmvq_launch gvq_mode(uint32_t rows) { return rows <= GVQ_VERIFY_ROWS ? g_gvq_pdl : GLMVQ_LAUNCH_PLAIN; }

struct ds4_gvq_ctx {
    const ds4_gvq_model *model;
    ds4_gpu_tensor *scratch, *workspace, *pairs;
    cublasHandle_t handle;
    cudaStream_t stream;
    glmvq_restore_mode restore;
    bool direct_q4;
    bool q4_mma;
    bool gate_up;
    bool split_row;
    bool gain_dot;
    bool q4_round;
    bool q4_prefill;
    bool direct_fp8;
    bool direct_kv;
    bool latent_attn;
    bool decode_route;
    bool direct_bf16;
    bool expand_mix;
    bool kda_chunk;
    bool kda_mma;
    bool flash;
    glmvq_launch launch;
    bool index_fused;
    uint64_t flash_calls = 0;
    uint64_t kda_chunks = 0;
    uint64_t kda_mma_calls = 0;
    uint64_t bf16_calls = 0;
    uint64_t latent_calls = 0;
    uint64_t kv_calls = 0;
    uint64_t fp8_calls = 0;
    glmvq_q4_config q4;
    std::vector<uint8_t> q4p_fit, fp8p_fit;
    std::vector<int32_t> routes;
    std::vector<uint32_t> grouped;
    ds4_gpu_tensor *mma = nullptr;
    int mma_blocks[3] = {};
    bool mma8 = false;
    int mma8_blocks[3] = {};
    // R13+8 layers: FP8 MMA with the residual stage (mma8_tokens items).
    bool mma8_res = false;
    int mma8_res_blocks[3] = {};
    // 2,048-token chunks route ~57 tokens per expert: 64-token items decode
    // each weight tile about once instead of twice (DS4_GLMVQ_MMA8_T64=0: 32).
    uint32_t mma8_tokens = GVQ_MMA_TOKENS;
    std::vector<glmvq_mma_item> items;
    std::vector<uint32_t> order;
    std::vector<uint8_t> token_used;
    uint64_t mma_calls = 0, mma_items = 0;
    ds4_gpu_tensor *decode = nullptr;
    uint32_t decode_blocks = 0;
    bool decode_res = false;
    glmvq_decode_item decode_items[GVQ_USED] = {};
    uint64_t decode_calls = 0;
    // Per-layer expert->matrix tables on device, keyed by expert 0's gate
    // index. A layer whose 864 matrices all fit routes decode on the GPU.
    std::vector<uint32_t> decode_keys;
    std::vector<uint8_t> decode_fit;
};

enum { GVQ_DECODE_LAYERS = 46, GVQ_DECODE_TABLE = 3 * GVQ_EXPERTS };
static constexpr uint64_t GVQ_DECODE_WORK =
    (sizeof(glmvq_decode_item) * GVQ_USED + 15u) / 16u * 16u;
static constexpr uint64_t GVQ_DECODE_ROWS =
    GVQ_USED * (GVQ_FFN * sizeof(__nv_bfloat16) + GVQ_HIDDEN * sizeof(float));

// Verify-width scratch after the tables: items, plan, mid and partial rows.
static constexpr uint64_t GVQ_VERIFY_AT = (GVQ_DECODE_WORK + GVQ_DECODE_ROWS +
    (uint64_t)GVQ_DECODE_LAYERS * GVQ_DECODE_TABLE * sizeof(uint32_t) + 255u) / 256u * 256u;
static constexpr uint64_t GVQ_VERIFY_ITEMS = (sizeof(glmvq_verify_item) * GLMVQ_VER_PAIRS + 255u) / 256u * 256u;
static constexpr uint64_t GVQ_VERIFY_PLAN = (sizeof(glmvq_verify_plan) + 255u) / 256u * 256u;
static constexpr uint64_t GVQ_VERIFY_MID = GLMVQ_VER_PAIRS * GVQ_FFN * sizeof(__nv_bfloat16);

static uint64_t gvq_decode_bytes() {
    const char *enabled = getenv("DS4_GLMVQ_DECODE");
    if (enabled && !strcmp(enabled, "0")) { return 0; }
    return GVQ_VERIFY_AT + GVQ_VERIFY_ITEMS + GVQ_VERIFY_PLAN + GVQ_VERIFY_MID +
        GLMVQ_VER_PAIRS * GVQ_HIDDEN * sizeof(float);
}

static uint64_t gvq_mma_bytes(uint32_t rows) {
    const char *enabled = getenv("DS4_GLMVQ_MMA");
    if ((enabled && !strcmp(enabled, "0")) || rows < GVQ_MMA_TOKENS) { return 0; }
    // Grouped x/mid are BF16, gate/down F32; no expanded expert weights.
    const uint64_t per_pair = (GVQ_HIDDEN + GVQ_FFN) *
        (sizeof(float) + sizeof(__nv_bfloat16)) + sizeof(glmvq_mma_item) + sizeof(uint32_t);
    return (uint64_t)rows * GVQ_USED * per_pair;
}

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

// GVR8 residual of a 13-bit GVQ1 matrix: 128-byte GVR1 header `h`, then one
// index byte per vector into a K=256 E4M3 book. Rust validated hashes and
// links; this re-checks the geometry the kernels assume.
static bool gvq_link_res(ds4_gvq_model *plan, glmvq_source &s, const ds4_gvq_record &r, const uint8_t *h) {
    enum { RES_BITS = 8, RES_WORDS = 256 };
    if (r.res >= plan->source.size() || !h || !s.vq.plane) { return false; }
    const ds4_gvq_record &res = plan->records[r.res];
    if (res.type != GVQ_TYPE_RES || res.book >= plan->source.size()) { return false; }
    const ds4_gvq_record &book = plan->records[res.book];
    const uint64_t vectors = (uint64_t)r.rows * r.cols / GLMVQ_VECTOR;
    const uint64_t index = gvq_u64(h, 32), size = gvq_u64(h, 56);
    if (memcmp(h, "GVR1", 4) || h[4] != 1 || h[5] || h[6] != RES_BITS || h[7] != GLMVQ_VECTOR ||
        gvq_u32(h, 8) != r.rows || gvq_u32(h, 12) != r.cols || gvq_u32(h, 16) != RES_WORDS ||
        gvq_u32(h, 20) != GLMVQ_E4M3 || gvq_u32(h, 28) || index != GVQ_HEADER || size != res.bytes ||
        !gvq_span(index, vectors, res.bytes) || book.type != GVQ_TYPE_BOOK ||
        book.bytes != (uint64_t)RES_WORDS * GLMVQ_VECTOR) { return false; }
    s.vq.res = (const uint8_t *)plan->source[r.res].data + index;
    s.vq.res_book = (const uint8_t *)plan->source[res.book].data;
    return true;
}

// `hres`: the linked residual's header (nullptr without one).
static bool gvq_link(ds4_gvq_model *plan, const uint8_t *h, uint32_t index, const uint8_t *hres) {
    const ds4_gvq_record &r = plan->records[index];
    glmvq_source &s = plan->source[index];
    if (r.type == GVQ_TYPE_Q4) {
        if (r.cols % GLMVQ_Q4_BLOCK || r.bytes != (uint64_t)r.rows * r.cols /
            GLMVQ_Q4_BLOCK * GLMVQ_Q4_BYTES || (uintptr_t)s.data % 16) { return false; }
        s.storage = GLMVQ_WEIGHT_Q4;
        return true;
    }
    if (r.type == GVQ_TYPE_FP8) {
        if (r.scale >= plan->source.size()) { return false; }
        const ds4_gvq_record &scale = plan->records[r.scale];
        // A last row block may be partial (Ling MTP kv_a [576,4096]: 5 x 32).
        if (r.bytes != (uint64_t)r.rows * r.cols || scale.type != GVQ_TYPE_F32 ||
            r.cols % GLMVQ_SCALE_BLOCK ||
            scale.rows != (r.rows + GLMVQ_SCALE_BLOCK - 1) / GLMVQ_SCALE_BLOCK ||
            scale.cols != r.cols / GLMVQ_SCALE_BLOCK ||
            scale.bytes != (uint64_t)scale.rows * scale.cols * sizeof(float)) { return false; }
        s.storage = GLMVQ_WEIGHT_FP8;
        s.scale = (const float *)plan->source[r.scale].data;
        return true;
    }
    if (r.type != GVQ_TYPE_VQ) { return true; }
    if (!h || r.book >= plan->source.size() || r.bytes < GVQ_HEADER) { return false; }
    const ds4_gvq_record &book = plan->records[r.book];
    const uint32_t precision = gvq_u32(h, 20), words = gvq_u32(h, 16);
    const uint64_t gain = gvq_u64(h, 32), main = gvq_u64(h, 40), plane = gvq_u64(h, 48);
    const uint64_t elements = (uint64_t)r.rows * r.cols;
    // E4M3 books only: K8192 rows carry the 13th-bit plane, K4096 rows none.
    const bool wide = h[6] == 13 && words == 8192 && plane;
    const bool narrow = h[6] == 12 && words == 4096 && !plane;
    if (memcmp(h, "GVQ1", 4) || h[4] != 1 || h[5] || (!wide && !narrow) || h[7] != GLMVQ_VECTOR ||
        gvq_u32(h, 8) != r.rows || gvq_u32(h, 12) != r.cols ||
        precision != GLMVQ_E4M3 || book.type != GVQ_TYPE_BOOK ||
        book.bytes != (uint64_t)words * GLMVQ_VECTOR || gvq_u64(h, 56) != r.bytes ||
        !gvq_span(gain, (uint64_t)r.rows * sizeof(__half), r.bytes) ||
        !gvq_span(main, elements * 3 / 16, r.bytes) ||
        (wide && !gvq_span(plane, elements / 64, r.bytes))) { return false; }
    const uint8_t *data = (const uint8_t *)s.data;
    s.storage = GLMVQ_WEIGHT_VQ;
    s.vq = {data + main, wide ? data + plane : nullptr, plan->source[r.book].data,
        (const __half *)(data + gain), r.rows, r.cols, GLMVQ_E4M3, nullptr, nullptr};
    return r.res == DS4_GVQ_NO_LINK || gvq_link_res(plan, s, r, hres);
}

// One warp per VQ matrix copies its GVQ1 header out of the device copy, so
// linking never faults the host map: the loader returned those pages, and
// ~10^5 header faults would read gigabytes back through readahead.
static __global__ void gvq_header_gather(uint32_t *out, const uint32_t *const *src, uint32_t count) {
    if (blockIdx.x < count) {
        out[(uint64_t)blockIdx.x * (GVQ_HEADER / 4u) + threadIdx.x] = src[blockIdx.x][threadIdx.x];
    }
}

// `slot[i]` indexes record i's header in `out` (UINT32_MAX: not VQ).
static bool gvq_headers(const ds4_gvq_model *plan, std::vector<uint32_t> &slot,
        std::vector<uint8_t> &out) {
    std::vector<const uint32_t *> src;
    slot.assign(plan->records.size(), UINT32_MAX);
    for (size_t i = 0; i < plan->records.size(); i++) {
        const uint32_t type = plan->records[i].type;
        if ((type != GVQ_TYPE_VQ && type != GVQ_TYPE_RES) || plan->records[i].bytes < GVQ_HEADER) { continue; }
        slot[i] = (uint32_t)src.size();
        src.push_back((const uint32_t *)plan->source[i].data);
    }
    out.resize(src.size() * GVQ_HEADER);
    if (src.empty()) { return true; }
    const cudaStream_t stream = ds4_current_stream();
    ds4_gpu_tensor *ptrs = ds4_gpu_tensor_alloc(src.size() * sizeof(src[0]));
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(out.size());
    bool ok = ptrs && heads && ds4_gpu_tensor_write(ptrs, 0, src.data(), src.size() * sizeof(src[0]));
    if (ok) {
        gvq_header_gather<<<(unsigned)src.size(), GVQ_HEADER / 4u, 0, stream>>>(
            (uint32_t *)heads->ptr, (const uint32_t *const *)ptrs->ptr, (uint32_t)src.size());
        ok = cuda_ok(cudaGetLastError(), "GLM VQ header gather") &&
            cuda_ok(cudaStreamSynchronize(stream), "GLM VQ header gather") &&
            ds4_gpu_tensor_read(heads, 0, out.data(), out.size());
    }
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(ptrs);
    return ok;
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
    std::vector<uint32_t> slot;
    std::vector<uint8_t> headers;
    if (!gvq_headers(plan, slot, headers)) { ds4_gpu_gvq_unbind(plan); return nullptr; }
    for (uint32_t i = 0; i < count; i++) {
        const uint8_t *h = slot[i] == UINT32_MAX ? nullptr : headers.data() + (uint64_t)slot[i] * GVQ_HEADER;
        const uint32_t res = records[i].type == GVQ_TYPE_VQ ? records[i].res : DS4_GVQ_NO_LINK;
        const uint8_t *hres = res < count && slot[res] != UINT32_MAX
            ? headers.data() + (uint64_t)slot[res] * GVQ_HEADER : nullptr;
        if (!gvq_link(plan, h, i, hres)) { ds4_gpu_gvq_unbind(plan); return nullptr; }
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
    return GVQ_CUBLAS_WORKSPACE + (uint64_t)rows * GVQ_USED * sizeof(uint32_t) +
        gvq_mma_bytes(rows) + gvq_decode_bytes();
}

// A context for another family's linear/KDA/norm work: no GLM expert scratch.
extern "C" uint64_t ds4_gpu_gvq_workspace_linear(uint32_t rows) {
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
    if (ctx->latent_attn) {
        fprintf(stderr, "ds4: GLM VQ latent attention calls=%llu\n", (unsigned long long)ctx->latent_calls);
    }
    if (ctx->direct_kv) {
        fprintf(stderr, "ds4: GLM VQ BF16 KV calls=%llu\n", (unsigned long long)ctx->kv_calls);
    }
    if (ctx->direct_fp8) {
        fprintf(stderr, "ds4: GLM VQ direct FP8 calls=%llu\n", (unsigned long long)ctx->fp8_calls);
    }
    if (ctx->direct_bf16) {
        fprintf(stderr, "ds4: GLM VQ direct BF16 calls=%llu\n", (unsigned long long)ctx->bf16_calls);
    }
    if (ctx->kda_chunk) {
        fprintf(stderr, "ds4: GLM VQ chunked KDA calls=%llu\n", (unsigned long long)ctx->kda_chunks);
    }
    if (ctx->kda_mma) {
        fprintf(stderr, "ds4: GLM VQ tensor-core KDA calls=%llu\n", (unsigned long long)ctx->kda_mma_calls);
    }
    if (ctx->flash) {
        fprintf(stderr, "ds4: GLM VQ flash attention calls=%llu\n", (unsigned long long)ctx->flash_calls);
    }
    if (ctx->mma) {
        fprintf(stderr, "ds4: GLM VQ MMA calls=%llu items=%llu workspace=%llu fp8=%d\n",
            (unsigned long long)ctx->mma_calls, (unsigned long long)ctx->mma_items,
            (unsigned long long)ctx->mma->bytes, ctx->mma8 ? 1 : 0);
    }
    if (ctx->decode) {
        fprintf(stderr, "ds4: GLM VQ direct decode calls=%llu workspace=%llu\n",
            (unsigned long long)ctx->decode_calls, (unsigned long long)ctx->decode->bytes);
    }
    if (ctx->handle) { (void)cublasDestroy(ctx->handle); }
    ds4_gpu_tensor_free(ctx->mma);
    ds4_gpu_tensor_free(ctx->decode);
    ds4_gpu_tensor_free(ctx->pairs);
    ds4_gpu_tensor_free(ctx->workspace);
    ds4_gpu_tensor_free(ctx->scratch);
    delete ctx;
}

// Q4_K matrices whose shapes and scales fit the q4p FP16 folding: one
// header pass per matrix at creation, then one readback.
static bool gvq_q4p_scan(ds4_gvq_ctx *ctx) {
    const size_t count = ctx->model->source.size();
    ctx->q4p_fit.assign(count, 0);
    ctx->fp8p_fit.assign(count, 0);
    ds4_gpu_tensor *flags = count ? ds4_gpu_tensor_alloc(count * sizeof(uint32_t)) : nullptr;
    if (!flags) { return false; }
    const cudaStream_t stream = ds4_current_stream();
    bool ok = cuda_ok(cudaMemsetAsync(flags->ptr, 0, count * sizeof(uint32_t), stream), "GLM VQ Q4 prefill scan");
    for (size_t i = 0; ok && i < count; i++) {
        const glmvq_source &s = ctx->model->source[i];
        // FP8 matrices with 128 x 128 scales fold into the same FP16 operand.
        if (ctx->model->records[i].type == GVQ_TYPE_FP8 && s.storage == GLMVQ_WEIGHT_FP8 && s.data && s.scale &&
            glmvq_q4p_fits(s.rows, s.cols)) {
            const uint64_t blocks = (uint64_t)s.rows / GLMVQ_SCALE_BLOCK * (s.cols / GLMVQ_SCALE_BLOCK);
            glmvq_q4p_check_fp8<<<(unsigned)((blocks + 255) / 256), 256, 0, stream>>>(
                s.scale, blocks, (uint32_t *)flags->ptr + i);
            ok = cuda_ok(cudaGetLastError(), "GLM VQ FP8 prefill scan");
            ctx->fp8p_fit[i] = 1;
            continue;
        }
        if (ctx->model->records[i].type != GVQ_TYPE_Q4 || !s.data || !glmvq_q4p_fits(s.rows, s.cols)) { continue; }
        const uint64_t blocks = (uint64_t)s.rows * s.cols / GLMVQ_Q4_BLOCK;
        glmvq_q4p_check<<<(unsigned)((blocks + 255) / 256), 256, 0, stream>>>(
            (const uint8_t *)s.data, blocks, (uint32_t *)flags->ptr + i);
        ok = cuda_ok(cudaGetLastError(), "GLM VQ Q4 prefill scan");
        ctx->q4p_fit[i] = 1;
    }
    std::vector<uint32_t> bad(count);
    ok = ok && cuda_ok(cudaStreamSynchronize(stream), "GLM VQ Q4 prefill scan") &&
        ds4_gpu_tensor_read(flags, 0, bad.data(), count * sizeof(uint32_t));
    uint32_t rejected = 0;
    for (size_t i = 0; i < count; i++) {
        if (ctx->q4p_fit[i] && (!ok || bad[i])) { ctx->q4p_fit[i] = 0; rejected++; }
        if (ctx->fp8p_fit[i] && (!ok || bad[i])) { ctx->fp8p_fit[i] = 0; rejected++; }
    }
    if (rejected) { fprintf(stderr, "ds4: GLM VQ Q4 prefill keeps MMQ for %u matrices (scale range)\n", rejected); }
    ds4_gpu_tensor_free(flags);
    return ok;
}

// `experts` funds GLM's routed-expert MMA and decode tables; another family
// with its own routed runtime (Ling-3.1) borrows only the linear kernels.
static ds4_gvq_ctx *gvq_create(const ds4_gvq_model *model, uint64_t bytes, uint32_t rows, bool experts) {
    if (!model || !bytes || !rows) { return nullptr; }
    const char *restore = getenv("DS4_GLMVQ_RESTORE_VEC");
    if (restore && strcmp(restore, "0") && strcmp(restore, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_RESTORE_VEC accepts 0 or 1\n");
        return nullptr;
    }
    const char *mma = getenv("DS4_GLMVQ_MMA");
    if (mma && strcmp(mma, "0") && strcmp(mma, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_MMA accepts 0 or 1\n");
        return nullptr;
    }
    const char *decode = getenv("DS4_GLMVQ_DECODE");
    if (decode && strcmp(decode, "0") && strcmp(decode, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_DECODE accepts 0 or 1\n");
        return nullptr;
    }
    const char *fp8 = getenv("DS4_GLMVQ_FP8_DIRECT");
    if (fp8 && strcmp(fp8, "0") && strcmp(fp8, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_FP8_DIRECT accepts 0 or 1\n");
        return nullptr;
    }
    const char *kv = getenv("DS4_GLMVQ_KV_BF16");
    if (kv && strcmp(kv, "0") && strcmp(kv, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_KV_BF16 accepts 0 or 1\n");
        return nullptr;
    }
    const char *latent_attn = getenv("DS4_GLMVQ_LATENT_ATTN");
    if (latent_attn && strcmp(latent_attn, "0") && strcmp(latent_attn, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_LATENT_ATTN accepts 0 or 1\n");
        return nullptr;
    }
    const char *bf16 = getenv("DS4_GLMVQ_BF16_DIRECT");
    if (bf16 && strcmp(bf16, "0") && strcmp(bf16, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_BF16_DIRECT accepts 0 or 1\n");
        return nullptr;
    }
    const char *mma8 = getenv("DS4_GLMVQ_MMA8");
    if (mma8 && strcmp(mma8, "0") && strcmp(mma8, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_MMA8 accepts 0 or 1\n");
        return nullptr;
    }
    const char *flash = getenv("DS4_GLMVQ_FLASH");
    if (flash && strcmp(flash, "0") && strcmp(flash, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_FLASH accepts 0 or 1\n");
        return nullptr;
    }
    const char *q4_round = getenv("DS4_GLMVQ_Q4_ROUND");
    if (q4_round && strcmp(q4_round, "0") && strcmp(q4_round, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_Q4_ROUND accepts 0 or 1\n");
        return nullptr;
    }
    const char *kda = getenv("DS4_GLMVQ_KDA_CHUNK");
    if (kda && strcmp(kda, "0") && strcmp(kda, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_KDA_CHUNK accepts 0 or 1\n");
        return nullptr;
    }
    const char *pdl = getenv("DS4_GLMVQ_PDL");
    if (pdl && strcmp(pdl, "0") && strcmp(pdl, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_PDL accepts 0 or 1\n");
        return nullptr;
    }
    const char *kda_mma = getenv("DS4_GLMVQ_KDA_MMA");
    if (kda_mma && strcmp(kda_mma, "0") && strcmp(kda_mma, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_KDA_MMA accepts 0 or 1\n");
        return nullptr;
    }
    const char *route = getenv("DS4_GLMVQ_DECODE_ROUTE");
    if (route && strcmp(route, "0") && strcmp(route, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_DECODE_ROUTE accepts 0 or 1\n");
        return nullptr;
    }
    ds4_gvq_ctx *ctx = new (std::nothrow) ds4_gvq_ctx{};
    if (!ctx) { return nullptr; }
    ctx->direct_fp8 = !fp8 || strcmp(fp8, "0");
    ctx->direct_kv = !kv || strcmp(kv, "0");
    ctx->latent_attn = !latent_attn || strcmp(latent_attn, "0");
    ctx->decode_route = !route || strcmp(route, "0");
    ctx->direct_bf16 = !bf16 || strcmp(bf16, "0");
    ctx->expand_mix = cuda_ok(cudaFuncSetAttribute(glmvq_hc_expand_mix<float, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)glmvq_xmix_smem<float>()), "GLM VQ mHC expand mix shared memory") &&
        cuda_ok(cudaFuncSetAttribute(glmvq_hc_expand_mix<__nv_bfloat16, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)glmvq_xmix_smem<__nv_bfloat16>()), "GLM VQ mHC expand mix shared memory");
    ctx->kda_chunk = !kda || strcmp(kda, "0");
    // Decode-width kernels launch with programmatic dependent launch.
    ctx->launch = !pdl || strcmp(pdl, "0") ? GLMVQ_LAUNCH_PDL : GLMVQ_LAUNCH_PLAIN;
    g_gvq_pdl = ctx->launch;
    ctx->kda_mma = (!kda_mma || strcmp(kda_mma, "0")) &&
        cuda_ok(cudaFuncSetAttribute(glmvq_kda_local<float, float>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)GLMVQ_KLOCAL_SMEM), "GLM VQ KDA local shared memory") &&
        cuda_ok(cudaFuncSetAttribute(glmvq_kda_local<float, __nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)GLMVQ_KLOCAL_SMEM), "GLM VQ KDA local shared memory") &&
        cuda_ok(cudaFuncSetAttribute(glmvq_kda_local<__nv_bfloat16, float>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)GLMVQ_KLOCAL_SMEM), "GLM VQ KDA local shared memory") &&
        cuda_ok(cudaFuncSetAttribute(glmvq_kda_local<__nv_bfloat16, __nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)GLMVQ_KLOCAL_SMEM), "GLM VQ KDA local shared memory") &&
        cuda_ok(cudaFuncSetAttribute(glmvq_kda_scan<float>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)GLMVQ_KSCAN_SMEM), "GLM VQ KDA scan shared memory") &&
        cuda_ok(cudaFuncSetAttribute(glmvq_kda_scan<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)GLMVQ_KSCAN_SMEM), "GLM VQ KDA scan shared memory") &&
        cuda_ok(cudaFuncSetAttribute(glmvq_kda_scan<float, __nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)GLMVQ_KSCAN_SMEM), "GLM VQ KDA scan shared memory") &&
        cuda_ok(cudaFuncSetAttribute(glmvq_kda_scan<__nv_bfloat16, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_KSCAN_SMEM), "GLM VQ KDA scan shared memory");
    ctx->q4_round = q4_round && !strcmp(q4_round, "1");
    ctx->flash = (!flash || strcmp(flash, "0")) && cuda_ok(cudaFuncSetAttribute(glmvq_fa_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_FA_SMEM), "GLM VQ flash attention shared memory");
    const char *fused = getenv("DS4_GLMVQ_INDEX_FUSED");
    ctx->index_fused = (!fused || strcmp(fused, "0")) && cuda_ok(cudaFuncSetAttribute(glmvq_index_fused,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_IDX_SMEM), "GLM VQ fused index shared memory");
    // Scratch omits the cuBLAS dot buffer for prefill rows while fused is on.
    if ((!fused || strcmp(fused, "0")) && !ctx->index_fused) {
        fprintf(stderr, "ds4: GLM VQ fused index unavailable; set DS4_GLMVQ_INDEX_FUSED=0\n");
        delete ctx;
        return nullptr;
    }
    // Graphs size no expanded-attention fallback while flash is requested.
    if ((!flash || strcmp(flash, "0")) && !ctx->flash) {
        fprintf(stderr, "ds4: GLM VQ flash attention unavailable; set DS4_GLMVQ_FLASH=0\n");
        delete ctx;
        return nullptr;
    }
    ctx->restore = restore && strcmp(restore, "0") == 0
        ? GLMVQ_RESTORE_SCALAR : GLMVQ_RESTORE_VECTOR;
    const char *q4 = getenv("DS4_GLMVQ_Q4_DIRECT");
    if (q4 && strcmp(q4, "0") && strcmp(q4, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_Q4_DIRECT accepts 0 or 1\n");
        delete ctx;
        return nullptr;
    }
    ctx->direct_q4 = !q4 || strcmp(q4, "0");
    const char *gain = getenv("DS4_GLMVQ_GAIN_DOT");
    if (gain && strcmp(gain, "0") && strcmp(gain, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_GAIN_DOT accepts 0 or 1\n");
        delete ctx;
        return nullptr;
    }
    ctx->gain_dot = !gain || strcmp(gain, "0");
    const char *q4_mma = getenv("DS4_GLMVQ_Q4_MMA");
    if (q4_mma && strcmp(q4_mma, "0") && strcmp(q4_mma, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_Q4_MMA accepts 0 or 1\n");
        delete ctx;
        return nullptr;
    }
    ctx->q4_mma = (!q4_mma || strcmp(q4_mma, "0")) &&
        cuda_ok(glmvq_q4m_prepare(), "GLM VQ Q4 MMA shared memory");
    const char *gate_up = getenv("DS4_GLMVQ_GATE_UP");
    if (gate_up && strcmp(gate_up, "0") && strcmp(gate_up, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_GATE_UP accepts 0 or 1\n");
        delete ctx;
        return nullptr;
    }
    ctx->gate_up = !gate_up || strcmp(gate_up, "0");
    const char *split_row = getenv("DS4_GLMVQ_SPLIT_PRELOAD");
    if (split_row && strcmp(split_row, "0") && strcmp(split_row, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_SPLIT_PRELOAD accepts 0 or 1\n");
        delete ctx;
        return nullptr;
    }
    ctx->split_row = !split_row || strcmp(split_row, "0");
    const char *q4_prefill = getenv("DS4_GLMVQ_Q4_PREFILL");
    if (q4_prefill && strcmp(q4_prefill, "0") && strcmp(q4_prefill, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_Q4_PREFILL accepts 0 or 1\n");
        delete ctx;
        return nullptr;
    }
    ctx->q4_prefill = (!q4_prefill || strcmp(q4_prefill, "0")) &&
        cuda_ok(glmvq_q4p_init(), "GLM VQ Q4 prefill shared memory");
    const char *q4_stream = getenv("DS4_GLMVQ_Q4_STREAM");
    if (q4_stream && strcmp(q4_stream, "0") && strcmp(q4_stream, "1")) {
        fprintf(stderr, "ds4: DS4_GLMVQ_Q4_STREAM accepts 0 or 1\n");
        delete ctx;
        return nullptr;
    }
    const glmvq_q4_load q4_mode = q4_stream && !strcmp(q4_stream, "0") ? GLMVQ_Q4_SCALAR : GLMVQ_Q4_AUTO;
    if (!cuda_ok(glmvq_q4_prepare(&ctx->q4, q4_mode), "GLM VQ Q4 resources")) {
        delete ctx;
        return nullptr;
    }
    ctx->q4.launch = ctx->launch;
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
    const uint64_t mma_bytes = experts ? gvq_mma_bytes(rows) : 0;
    if (mma_bytes) {
        // Reserve before execution/capture; shape/resource failures keep GEMM.
        const cudaError_t status[] = {
            glmvq_mma_prepare<GVQ_MMA_ROWS, GVQ_MMA_TOKENS, GLMVQ_MMA_GATE>(&ctx->mma_blocks[0]),
            glmvq_mma_prepare<GVQ_MMA_ROWS, GVQ_MMA_TOKENS, GLMVQ_MMA_UP>(&ctx->mma_blocks[1]),
            glmvq_mma_prepare<GVQ_MMA_ROWS, GVQ_MMA_TOKENS, GLMVQ_MMA_DOWN>(&ctx->mma_blocks[2])
        };
        if (status[0] == cudaSuccess && status[1] == cudaSuccess && status[2] == cudaSuccess) {
            if (!mma8 || strcmp(mma8, "0")) {
                const char *t64 = getenv("DS4_GLMVQ_MMA8_T64");
                const bool wide = !t64 || strcmp(t64, "0");
                const cudaError_t fp8[] = {
                    wide ? glmvq_mma8_prepare<GVQ_MMA_ROWS, GVQ_MMA8_TOKENS, 1, GLMVQ_MMA_GATE>(&ctx->mma8_blocks[0])
                         : glmvq_mma8_prepare<GVQ_MMA_ROWS, GVQ_MMA_TOKENS, 1, GLMVQ_MMA_GATE>(&ctx->mma8_blocks[0]),
                    wide ? glmvq_mma8_prepare<GVQ_MMA_ROWS, GVQ_MMA8_TOKENS, 1, GLMVQ_MMA_UP>(&ctx->mma8_blocks[1])
                         : glmvq_mma8_prepare<GVQ_MMA_ROWS, GVQ_MMA_TOKENS, 1, GLMVQ_MMA_UP>(&ctx->mma8_blocks[1]),
                    wide ? glmvq_mma8_prepare<GVQ_MMA_ROWS, GVQ_MMA8_TOKENS, 1, GLMVQ_MMA_DOWN>(&ctx->mma8_blocks[2])
                         : glmvq_mma8_prepare<GVQ_MMA_ROWS, GVQ_MMA_TOKENS, 1, GLMVQ_MMA_DOWN>(&ctx->mma8_blocks[2])
                };
                ctx->mma8 = fp8[0] == cudaSuccess && fp8[1] == cudaSuccess && fp8[2] == cudaSuccess;
                ctx->mma8_tokens = wide ? GVQ_MMA8_TOKENS : GVQ_MMA_TOKENS;
                // R13+8 layers take the same item width (64: residual codes in
                // registers; 32: staged in shared memory).
                auto res8 = [&](auto tokens) {
                    constexpr uint32_t BT = decltype(tokens)::value;
                    return glmvq_mma8_prepare<GVQ_MMA_ROWS, BT, 1, GLMVQ_MMA_GATE, 13, GLMVQ_CLAMP_GATE, true>(
                            &ctx->mma8_res_blocks[0]) == cudaSuccess &&
                        glmvq_mma8_prepare<GVQ_MMA_ROWS, BT, 1, GLMVQ_MMA_UP, 13, GLMVQ_CLAMP_GATE, true>(
                            &ctx->mma8_res_blocks[1]) == cudaSuccess &&
                        glmvq_mma8_prepare<GVQ_MMA_ROWS, BT, 1, GLMVQ_MMA_DOWN, 13, GLMVQ_CLAMP_GATE, true>(
                            &ctx->mma8_res_blocks[2]) == cudaSuccess;
                };
                const bool res = wide ? res8(std::integral_constant<uint32_t, GVQ_MMA8_TOKENS>{})
                                      : res8(std::integral_constant<uint32_t, GVQ_MMA_TOKENS>{});
                ctx->mma8_res = ctx->mma8 && res;
                if (ctx->mma8 && !ctx->mma8_res) { (void)cudaGetLastError(); }
                if (!ctx->mma8) {
                    (void)cudaGetLastError();
                    fprintf(stderr, "ds4: GLM VQ FP8 MMA unavailable; using BF16 register MMA\n");
                }
            }
            ctx->mma = ds4_gpu_tensor_alloc(mma_bytes);
            if (!ctx->mma) { ds4_gpu_gvq_free(ctx); return nullptr; }
            ctx->items.reserve((uint64_t)rows * GVQ_USED);
            ctx->order.resize((uint64_t)rows * GVQ_USED);
            ctx->token_used.resize(rows);
        } else {
            (void)cudaGetLastError();
            fprintf(stderr, "ds4: GLM VQ MMA resources unavailable; using restore+GEMM\n");
        }
    }
    const uint64_t decode_bytes = experts ? gvq_decode_bytes() : 0;
    if (decode_bytes) {
        // The published artifact uses a shared 8192-word E4M3 layer book.
        // Reserve descriptors and intermediate rows before any execution.
        cudaError_t status = glmvq_decode_prepare<13, 32, GLMVQ_DEC_NORMAL>(&ctx->decode_blocks);
        if (status == cudaSuccess) {
            status = glmvq_decode_prepare<13, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_DOT>(&ctx->decode_blocks);
        }
        if (status == cudaSuccess) {
            // R13+8 layers; without them those layers keep restore+GEMM.
            ctx->decode_res = glmvq_decode_prepare<13, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_WEIGHT, glmvq_glm_form, true>(
                &ctx->decode_blocks) == cudaSuccess &&
                glmvq_decode_prepare<13, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_DOT, glmvq_glm_form, true>(
                &ctx->decode_blocks) == cudaSuccess;
            if (!ctx->decode_res) { (void)cudaGetLastError(); }
        }
        if (status == cudaSuccess) {
            ctx->decode = ds4_gpu_tensor_alloc(decode_bytes);
            if (!ctx->decode) { ds4_gpu_gvq_free(ctx); return nullptr; }
        } else {
            (void)cudaGetLastError();
            fprintf(stderr, "ds4: GLM VQ direct decode resources unavailable; using restore+GEMM\n");
        }
    }
    if (ctx->q4_prefill && !gvq_q4p_scan(ctx)) { ctx->q4_prefill = false; }
    return ctx;
}

extern "C" ds4_gvq_ctx *ds4_gpu_gvq_create(const ds4_gvq_model *model, uint64_t bytes, uint32_t rows) {
    return gvq_create(model, bytes, rows, true);
}

extern "C" ds4_gvq_ctx *ds4_gpu_gvq_create_linear(const ds4_gvq_model *model, uint64_t bytes, uint32_t rows) {
    return gvq_create(model, bytes, rows, false);
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

/* Prefill Q4_K on FP16 tensor cores (cuda/glm_vq_q4p.cuh) from 256 tokens:
 * below that MMQ's narrow tiles stream the weights faster (8192 x 4096 at
 * 64 tokens: 0.064 vs 0.147 ms; at 256: 0.343 vs 0.288). The prepared input
 * lives in scratch, so an input or output inside it keeps MMQ.
 * DS4_GLMVQ_Q4_PREFILL=0 keeps MMQ everywhere. */
enum { GVQ_Q4P_MIN_TOKENS = 256 };

static bool gvq_q4p_ready(const ds4_gvq_ctx *ctx, const uint32_t *weights, uint32_t count,
        const ds4_gpu_tensor *input, ds4_gpu_tensor *const *outs, uint32_t rows) {
    if (!ctx->q4_prefill || ctx->q4_round || rows < GVQ_Q4P_MIN_TOKENS || !count) { return false; }
    const uint8_t *lo = (const uint8_t *)ctx->scratch->ptr, *hi = lo + ctx->scratch->bytes;
    auto inside = [&](const ds4_gpu_tensor *t) {
        const uint8_t *p = (const uint8_t *)t->ptr;
        return p < hi && p + t->bytes > lo;
    };
    if (inside(input) ||
        glmvq_q4p_bytes(rows, ctx->model->source[weights[0]].cols) > ctx->scratch->bytes) { return false; }
    for (uint32_t c = 0; c < count; c++) {
        if (weights[c] >= ctx->q4p_fit.size() || !ctx->q4p_fit[weights[c]] || inside(outs[c])) { return false; }
    }
    return true;
}

template <typename OUT = float, typename IN = float>
static int gvq_q4p_run(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs, const uint32_t *weights,
        uint32_t count, const ds4_gpu_tensor *input, uint32_t rows) {
    const uint32_t cols = ctx->model->source[weights[0]].cols;
    const glmvq_q4p_input in = glmvq_q4p_layout(ctx->scratch->ptr, rows, cols);
    if (!cuda_ok(glmvq_q4p_prepare(in, (const IN *)input->ptr, rows, cols, ctx->stream), "GLM VQ Q4 prefill input")) {
        return 0;
    }
    for (uint32_t c = 0; c < count; c++) {
        const glmvq_source &s = ctx->model->source[weights[c]];
        if (!cuda_ok(glmvq_q4p_gemm((const uint8_t *)s.data, in, (OUT *)outs[c]->ptr, s.rows, s.cols, rows,
                ctx->stream), "GLM VQ Q4 prefill")) { return 0; }
    }
    return 1;
}

extern "C" int ds4_gpu_gvq_linear(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t index,
        const ds4_gpu_tensor *input, uint32_t rows, ds4_gvq_math math) {
    if (!ctx || !rows || index >= ctx->model->source.size()) { return 0; }
    const glmvq_source &s = ctx->model->source[index];
    const uint32_t type = ctx->model->records[index].type;
    if ((type != GVQ_TYPE_BF16 && type != GVQ_TYPE_FP8 && type != GVQ_TYPE_VQ && type != GVQ_TYPE_Q4) ||
        !gvq_floats(input, (uint64_t)rows * s.cols) || !gvq_floats(out, (uint64_t)rows * s.rows) ||
        (math != DS4_GVQ_BF16 && math != DS4_GVQ_F32) || !gvq_stream(ctx)) { return 0; }
    if (type == GVQ_TYPE_FP8 && ctx->direct_fp8 && rows <= GLMVQ_FP8_TOKENS && math == DS4_GVQ_BF16) {
        const uint32_t split = glmvq_fp8_split(s.rows, s.cols);
        if (split && glmvq_fp8_fits(s.data, s.scale, input->ptr, out->ptr, s.rows, s.cols)) {
            // Preserve source BF16 boundaries without writing expanded weights.
            ctx->fp8_calls++;
            return cuda_ok(glmvq_fp8_decode(s.data, s.scale, (const float *)input->ptr,
                (float *)out->ptr, s.rows, s.cols, split, GLMVQ_FP8_GLOBAL, ctx->stream, rows),
                "GLM VQ FP8 decode");
        }
    }
    if (type == GVQ_TYPE_BF16 && ctx->direct_bf16 && rows <= GLMVQ_DOT_TOKENS &&
        glmvq_bf16_fits(s.data, input->ptr, out->ptr, s.cols)) {
        ctx->bf16_calls++;
        return cuda_ok(glmvq_bf16_decode(s.data, (const float *)input->ptr, (float *)out->ptr,
            s.rows, s.cols, math == DS4_GVQ_F32 ? GLMVQ_DOT_F32 : GLMVQ_DOT_BF16, ctx->stream, rows, ctx->launch),
            "GLM VQ BF16 decode");
    }
    if (type == GVQ_TYPE_Q4 && ctx->direct_q4) {
        if (math != DS4_GVQ_BF16) { return 0; }
        // Verify widths read each weight once on tensor cores; one row keeps
        // the SIMT stream. DS4_GLMVQ_Q4_MMA=0 keeps SIMT for every width.
        if (rows > 1 && ctx->q4_mma && glmvq_q4m_fits(s.data, (const float *)input->ptr, s.cols, rows)) {
            return cuda_ok(glmvq_q4m_decode(s.data, (const float *)input->ptr, (float *)out->ptr,
                s.rows, s.cols, rows, ctx->stream, gvq_mode(rows)), "GLM VQ Q4 MMA");
        }
        if (rows <= GLMVQ_Q4_WARPS) {
            return cuda_ok(glmvq_q4_decode(s.data, (const float *)input->ptr, (float *)out->ptr,
                s.rows, s.cols, rows, ctx->stream, &ctx->q4), "GLM VQ Q4 decode");
        }
        if (gvq_q4p_ready(ctx, &index, 1, input, &out, rows)) { return gvq_q4p_run(ctx, &out, &index, 1, input, rows); }
        // Wide projections reuse the established Q4_K x Q8_1 MMQ primitive.
        // Prefill keeps its outputs in F32: the BF16 boundary pass cost 4.6%
        // of an 8K prefill and moved frontier logits within the chunk-size
        // noise band. DS4_GLMVQ_Q4_ROUND=1 restores the boundary.
        const int status = ctx->q4_round
            ? ds4_mmq_q4_K_dense_bf16(s.data, (const float *)input->ptr, (float *)out->ptr, s.rows, rows, s.cols, ctx->stream)
            : ds4_mmq_q4_K_dense_raw(s.data, (const float *)input->ptr, (float *)out->ptr, s.rows, rows, s.cols, ctx->stream);
        return status == 0 && cuda_ok(cudaGetLastError(), "GLM VQ Q4 MMQ");
    }
    return cublas_ok(glmvq_gemm(ctx->handle, s, (const float *)input->ptr, (float *)out->ptr,
        rows, (glmvq_compute)math, ctx->scratch->ptr, ctx->scratch->bytes, ctx->stream,
        ctx->restore), "GLM VQ linear");
}

// One token: the SIMT stream beats the MMA kernel on the 12,288-row dense
// FFN (231 vs 249 us per gate/up pair); the 2,048-row shared expert gains
// 57 -> 49 us. Verify widths always fuse (shared pair 58 -> 53 us at three
// tokens). DS4_GLMVQ_GATE_UP=0 restores the separate projections.
enum { GVQ_GATE_UP_ONE_ROW_MAX = 4096 };

extern "C" int ds4_gpu_gvq_gate_up(ds4_gvq_ctx *ctx, ds4_gpu_tensor *mid, uint32_t gate, uint32_t up,
        const ds4_gpu_tensor *input, uint32_t rows, float limit) {
    if (!ctx || !ctx->gate_up || !rows || rows > GLMVQ_Q4M_TOKENS || !ctx->direct_q4 || !ctx->q4_mma ||
        gate >= ctx->model->source.size() || up >= ctx->model->source.size()) { return 0; }
    const glmvq_source &g = ctx->model->source[gate], &u = ctx->model->source[up];
    if (ctx->model->records[gate].type != GVQ_TYPE_Q4 || ctx->model->records[up].type != GVQ_TYPE_Q4 ||
        g.rows != u.rows || g.cols != u.cols || (rows == 1 && g.rows > GVQ_GATE_UP_ONE_ROW_MAX) ||
        !gvq_floats(input, (uint64_t)rows * g.cols) || !gvq_floats(mid, (uint64_t)rows * g.rows)) { return 0; }
    return cuda_ok(glmvq_q4m_swiglu(g.data, u.data, (const float *)input->ptr, (float *)mid->ptr,
        g.rows, g.cols, rows, limit, ctx->stream, gvq_mode(rows)), "GLM VQ Q4 gate/up");
}

extern "C" int ds4_gpu_gvq_linear_bf16(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t index,
        const ds4_gpu_tensor *input, uint32_t rows) {
    if (!ctx || rows <= GLMVQ_DOT_TOKENS || index >= ctx->model->source.size() || !gvq_stream(ctx)) { return 0; }
    const glmvq_source &s = ctx->model->source[index];
    if (ctx->model->records[index].type != GVQ_TYPE_BF16 || s.storage != GLMVQ_WEIGHT_BF16 || !s.data ||
        !gvq_floats(input, (uint64_t)rows * s.cols) || !gvq_floats(out, (uint64_t)rows * s.rows / 2)) { return 0; }
    return cublas_ok(glmvq_gemm(ctx->handle, s, (const float *)input->ptr, (float *)out->ptr,
        rows, GLMVQ_COMPUTE_BF16, ctx->scratch->ptr, ctx->scratch->bytes, ctx->stream,
        ctx->restore, GLMVQ_OUT_BF16), "GLM VQ linear BF16 rows");
}

enum { GVQ_LINEAR_MANY = 4 };

/* Matrices sharing one input. Prefill Q4_K quantizes it to Q8_1 once and
 * BF16 converts it once (each output otherwise as ds4_gpu_gvq_linear, so
 * results are unchanged); other shapes and widths run one call each. */
extern "C" int ds4_gpu_gvq_linear_many(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs,
        const uint32_t *weights, uint32_t count, const ds4_gpu_tensor *input, uint32_t rows, ds4_gvq_math math) {
    if (!ctx || !outs || !weights || !count || count > GVQ_LINEAR_MANY || !rows || !gvq_stream(ctx)) { return 0; }
    bool q4 = ctx->direct_q4 && !ctx->q4_round && rows > GLMVQ_Q4_WARPS && math == DS4_GVQ_BF16;
    bool bf16 = rows > GLMVQ_DOT_TOKENS && math == DS4_GVQ_BF16;
    uint32_t cols = 0, widest = 0;
    for (uint32_t c = 0; c < count; c++) {
        if (weights[c] >= ctx->model->source.size()) { return 0; }
        const glmvq_source &s = ctx->model->source[weights[c]];
        const uint32_t type = ctx->model->records[weights[c]].type;
        if ((c && s.cols != cols) || !gvq_floats(outs[c], (uint64_t)rows * s.rows)) { q4 = bf16 = false; }
        cols = s.cols;
        widest = max(widest, s.rows);
        q4 = q4 && type == GVQ_TYPE_Q4;
        bf16 = bf16 && type == GVQ_TYPE_BF16 && s.storage == GLMVQ_WEIGHT_BF16 && s.data;
    }
    if (q4 && gvq_floats(input, (uint64_t)rows * cols) && gvq_q4p_ready(ctx, weights, count, input, outs, rows)) {
        return gvq_q4p_run(ctx, outs, weights, count, input, rows);
    }
    if (q4 && gvq_floats(input, (uint64_t)rows * cols)) {
        const void *w[GVQ_LINEAR_MANY];
        float *y[GVQ_LINEAR_MANY];
        int m[GVQ_LINEAR_MANY];
        for (uint32_t c = 0; c < count; c++) {
            const glmvq_source &s = ctx->model->source[weights[c]];
            w[c] = s.data; y[c] = (float *)outs[c]->ptr; m[c] = (int)s.rows;
        }
        return ds4_mmq_q4_K_dense_raw_many(w, y, m, (int)count, (const float *)input->ptr, (int)rows, (int)cols,
            ctx->stream) == 0 && cuda_ok(cudaGetLastError(), "GLM VQ Q4 MMQ many");
    }
    const uint64_t nx = (uint64_t)rows * cols;
    if (bf16 && gvq_floats(input, nx) && ctx->scratch->bytes >= (nx + (uint64_t)rows * widest) * sizeof(__nv_bfloat16)) {
        auto *x = (__nv_bfloat16 *)ctx->scratch->ptr, *y = x + nx;
        glmvq_convert<<<(nx + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, ctx->stream>>>(
            (const float *)input->ptr, x, nx);
        const float alpha = 1.0f, beta = 0.0f;
        for (uint32_t c = 0; c < count; c++) {
            const glmvq_source &s = ctx->model->source[weights[c]];
            const uint64_t ny = (uint64_t)rows * s.rows;
            if (!cuda_ok(cudaGetLastError(), "GLM VQ BF16 many input") ||
                !cublas_ok(cublasGemmEx(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N, s.rows, rows, s.cols,
                    &alpha, s.data, CUDA_R_16BF, s.cols, x, CUDA_R_16BF, s.cols, &beta, y, CUDA_R_16BF, s.rows,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "GLM VQ BF16 many")) { return 0; }
            glmvq_convert<<<(ny + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, ctx->stream>>>(
                y, (float *)outs[c]->ptr, ny);
        }
        return cuda_ok(cudaGetLastError(), "GLM VQ BF16 many output");
    }
    for (uint32_t c = 0; c < count; c++) {
        if (!ds4_gpu_gvq_linear(ctx, outs[c], weights[c], input, rows, math)) { return 0; }
    }
    return 1;
}

extern "C" int ds4_gpu_gvq_q4_decode_many(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs,
        const uint32_t *weights, uint32_t count, const ds4_gpu_tensor *input, uint32_t tokens) {
    if (!ctx || !outs || !weights || !count || count > GLMVQ_Q4_MANY || !tokens || tokens > GLMVQ_Q4_WARPS ||
        !ctx->direct_q4 || !gvq_stream(ctx)) {
        return 0;
    }
    const void *w[GLMVQ_Q4_MANY];
    float *y[GLMVQ_Q4_MANY];
    uint32_t rows[GLMVQ_Q4_MANY], cols = 0;
    for (uint32_t c = 0; c < count; c++) {
        if (weights[c] >= ctx->model->source.size()) { return 0; }
        const glmvq_source &s = ctx->model->source[weights[c]];
        if (ctx->model->records[weights[c]].type != GVQ_TYPE_Q4 || (c && s.cols != cols) ||
            !gvq_floats(outs[c], (uint64_t)tokens * s.rows)) { return 0; }
        cols = s.cols;
        w[c] = s.data; y[c] = (float *)outs[c]->ptr; rows[c] = s.rows;
    }
    if (!gvq_floats(input, (uint64_t)tokens * cols)) { return 0; }
    return cuda_ok(glmvq_q4_decode_many(w, y, rows, count, cols, (const float *)input->ptr, tokens, ctx->stream,
        &ctx->q4), "GLM VQ Q4 decode many");
}

// Up to eight tokens of one Q4_K matrix on the SIMT decode kernels, whatever
// the MMA setting: each token equals a one-token ds4_gpu_gvq_linear call bit
// for bit (staged shapes). 0 for other types and widths.
extern "C" int ds4_gpu_gvq_q4_decode_rows(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t index,
        const ds4_gpu_tensor *input, uint32_t rows) {
    if (!ctx || !rows || rows > GLMVQ_Q4_WARPS || !ctx->direct_q4 || index >= ctx->model->source.size() ||
        ctx->model->records[index].type != GVQ_TYPE_Q4 || !gvq_stream(ctx)) { return 0; }
    const glmvq_source &s = ctx->model->source[index];
    if (!gvq_floats(input, (uint64_t)rows * s.cols) || !gvq_floats(out, (uint64_t)rows * s.rows)) { return 0; }
    return cuda_ok(glmvq_q4_decode(s.data, (const float *)input->ptr, (float *)out->ptr, s.rows, s.cols, rows,
        ctx->stream, &ctx->q4), "GLM VQ Q4 decode rows");
}

// Whether q4p runs these Q4_K matrices at `rows` (a caller that prepares the
// input itself then calls ds4_gpu_gvq_q4p_prepared).
extern "C" int ds4_gpu_gvq_q4p_takes(ds4_gvq_ctx *ctx, const uint32_t *weights, uint32_t count, uint32_t rows) {
    if (!ctx || !weights || !count || count > GVQ_LINEAR_MANY || !ctx->direct_q4 || !ctx->q4_prefill ||
        ctx->q4_round || rows < GVQ_Q4P_MIN_TOKENS) { return 0; }
    for (uint32_t c = 0; c < count; c++) {
        if (weights[c] >= ctx->q4p_fit.size() || !ctx->q4p_fit[weights[c]] ||
            ctx->model->source[weights[c]].cols != ctx->model->source[weights[0]].cols) { return 0; }
    }
    return 1;
}

// q4p over an input prepared as glmvq_q4p_prep writes it: FP16 rows in its
// (0, 2, 1, 3) quad order and one 2^(e-6) scale per row. Outputs are F32 as
// ds4_gpu_gvq_linear_many's q4p path leaves them; 0 when q4p does not take
// the weights (outputs untouched).
extern "C" int ds4_gpu_gvq_q4p_prepared(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs, const uint32_t *weights,
        uint32_t count, const ds4_gpu_tensor *xh, const ds4_gpu_tensor *xscale, uint32_t rows) {
    if (!outs || !xh || !xscale || !ds4_gpu_gvq_q4p_takes(ctx, weights, count, rows) || !gvq_stream(ctx)) { return 0; }
    const uint32_t cols = ctx->model->source[weights[0]].cols;
    if (xh->bytes < (uint64_t)rows * cols * sizeof(__half) || !gvq_floats(xscale, rows)) { return 0; }
    for (uint32_t c = 0; c < count; c++) {
        if (!gvq_floats(outs[c], (uint64_t)rows * ctx->model->source[weights[c]].rows)) { return 0; }
    }
    const glmvq_q4p_input in = {(__half *)xh->ptr, (float *)xscale->ptr};
    for (uint32_t c = 0; c < count; c++) {
        const glmvq_source &s = ctx->model->source[weights[c]];
        if (!cuda_ok(glmvq_q4p_gemm((const uint8_t *)s.data, in, (float *)outs[c]->ptr, s.rows, s.cols, rows,
                ctx->stream), "GLM VQ Q4 prefill (prepared input)")) { return 0; }
    }
    return 1;
}

// ds4_gpu_gvq_q4p_prepared of one matrix into BF16 rows: q4p's epilogue
// rounds the same F32 product to nearest, as a later BF16 rounding of the
// F32 rows would.
extern "C" int ds4_gpu_gvq_q4p_prepared_bf16(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t weight,
        const ds4_gpu_tensor *xh, const ds4_gpu_tensor *xscale, uint32_t rows) {
    if (!out || !xh || !xscale || !ds4_gpu_gvq_q4p_takes(ctx, &weight, 1, rows) || !gvq_stream(ctx)) { return 0; }
    const glmvq_source &s = ctx->model->source[weight];
    if (xh->bytes < (uint64_t)rows * s.cols * sizeof(__half) || !gvq_floats(xscale, rows) ||
        !gvq_floats(out, (uint64_t)rows * s.rows / 2)) { return 0; }
    const glmvq_q4p_input in = {(__half *)xh->ptr, (float *)xscale->ptr};
    return cuda_ok(glmvq_q4p_gemm((const uint8_t *)s.data, in, (__nv_bfloat16 *)out->ptr, s.rows, s.cols, rows,
        ctx->stream), "GLM VQ Q4 prefill (prepared input, BF16 rows)");
}

// Q4_K matrices sharing one input, stored as BF16 rows (the rounding decode
// and verify keep) when q4p takes them; 0 leaves the outputs untouched.
extern "C" int ds4_gpu_gvq_linear_many_bf16(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs,
        const uint32_t *weights, uint32_t count, const ds4_gpu_tensor *input, uint32_t rows) {
    if (!ctx || !outs || !weights || !count || count > GVQ_LINEAR_MANY || !rows || !gvq_stream(ctx)) { return 0; }
    uint32_t cols = 0;
    for (uint32_t c = 0; c < count; c++) {
        if (weights[c] >= ctx->model->source.size()) { return 0; }
        const glmvq_source &s = ctx->model->source[weights[c]];
        if ((c && s.cols != cols) || ctx->model->records[weights[c]].type != GVQ_TYPE_Q4 ||
            !gvq_floats(outs[c], (uint64_t)rows * s.rows / 2)) { return 0; }
        cols = s.cols;
    }
    if (!ctx->direct_q4 || !gvq_floats(input, (uint64_t)rows * cols) ||
        !gvq_q4p_ready(ctx, weights, count, input, outs, rows)) { return 0; }
    return gvq_q4p_run<__nv_bfloat16>(ctx, outs, weights, count, input, rows);
}

/* FP8 prefill projections of one input as BF16 rows: one BF16 copy of the
 * input, then the restore + cuBLAS of ds4_gpu_gvq_linear per matrix with
 * its BF16 result left in place (the F32 path only widens these values). */
extern "C" int ds4_gpu_gvq_fp8_many_bf16(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs,
        const uint32_t *weights, uint32_t count, const ds4_gpu_tensor *input, uint32_t rows) {
    if (!ctx || !outs || !weights || !count || count > GVQ_LINEAR_MANY || rows <= GLMVQ_FP8_TOKENS ||
        !gvq_stream(ctx)) { return 0; }
    uint32_t cols = 0;
    uint64_t widest = 0;
    for (uint32_t c = 0; c < count; c++) {
        if (weights[c] >= ctx->model->source.size()) { return 0; }
        const glmvq_source &s = ctx->model->source[weights[c]];
        if ((c && s.cols != cols) || ctx->model->records[weights[c]].type != GVQ_TYPE_FP8 ||
            s.storage != GLMVQ_WEIGHT_FP8 || !s.data || !s.scale || s.cols % GLMVQ_SCALE_BLOCK ||
            !gvq_floats(outs[c], (uint64_t)rows * s.rows / 2)) { return 0; }
        cols = s.cols;
        widest = max(widest, (uint64_t)s.rows * s.cols);
    }
    const uint64_t nx = (uint64_t)rows * cols;
    if (!gvq_floats(input, nx)) { return 0; }
    // q4p's FP16 operands from the FP8 bytes directly: no restored weight and
    // no BF16 input copy (DS4_LING31_FP8_Q4P=0 keeps restore + cuBLAS).
    static const int fp8p = [] { const char *e = getenv("DS4_LING31_FP8_Q4P"); return !(e && !strcmp(e, "0")); }();
    bool direct = fp8p && ctx->q4_prefill && rows >= GVQ_Q4P_MIN_TOKENS &&
        glmvq_q4p_bytes(rows, cols) <= ctx->scratch->bytes;
    for (uint32_t c = 0; direct && c < count; c++) {
        direct = weights[c] < ctx->fp8p_fit.size() && ctx->fp8p_fit[weights[c]];
    }
    if (direct) {
        const glmvq_q4p_input in = glmvq_q4p_layout(ctx->scratch->ptr, rows, cols);
        if (!cuda_ok(glmvq_q4p_prepare(in, (const float *)input->ptr, rows, cols, ctx->stream), "GLM VQ FP8 prefill input")) {
            return 0;
        }
        for (uint32_t c = 0; c < count; c++) {
            const glmvq_source &s = ctx->model->source[weights[c]];
            if (!cuda_ok(glmvq_q4p_gemm_fp8((const uint8_t *)s.data, s.scale, in, (__nv_bfloat16 *)outs[c]->ptr,
                    s.rows, s.cols, rows, ctx->stream), "GLM VQ FP8 prefill")) { return 0; }
        }
        return 1;
    }
    if (ctx->scratch->bytes < (widest + nx) * sizeof(__nv_bfloat16)) { return 0; }
    auto *weight = (__nv_bfloat16 *)ctx->scratch->ptr, *x = weight + widest;
    glmvq_convert<<<(nx + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, ctx->stream>>>(
        (const float *)input->ptr, x, nx);
    const float alpha = 1.0f, beta = 0.0f;
    for (uint32_t c = 0; c < count; c++) {
        const glmvq_source &s = ctx->model->source[weights[c]];
        const uint64_t n = (uint64_t)s.rows * s.cols;
        glmvq_fp8_restore<<<(unsigned)((n + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0, ctx->stream>>>(
            (const uint8_t *)s.data, s.scale, s.rows, s.cols, weight);
        if (!cuda_ok(cudaGetLastError(), "GLM VQ FP8 BF16 restore") ||
            !cublas_ok(cublasGemmEx(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N, s.rows, rows, s.cols,
                &alpha, weight, CUDA_R_16BF, s.cols, x, CUDA_R_16BF, s.cols, &beta, outs[c]->ptr, CUDA_R_16BF, s.rows,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "GLM VQ FP8 BF16 rows")) { return 0; }
    }
    return 1;
}

extern "C" int ds4_gpu_gvq_bf16_input(ds4_gvq_ctx *ctx, uint32_t weight, uint32_t rows) {
    return ctx && ctx->direct_q4 && ctx->q4_prefill && !ctx->q4_round && rows >= GVQ_Q4P_MIN_TOKENS &&
        weight < ctx->q4p_fit.size() && ctx->q4p_fit[weight] &&
        glmvq_q4p_bytes(rows, ctx->model->source[weight].cols) <= ctx->scratch->bytes;
}

// BF16 input rows go straight to the q4p FP16 preparation (the values are
// BF16-rounded either way, so the FP16 operand is unchanged).
extern "C" int ds4_gpu_gvq_linear_rows(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t weight,
        const ds4_gpu_tensor *input, ds4_gvq_rows in_rows, uint32_t rows) {
    if (in_rows == DS4_GVQ_ROWS_F32) { return ds4_gpu_gvq_linear(ctx, out, weight, input, rows, DS4_GVQ_BF16); }
    if (!ds4_gpu_gvq_bf16_input(ctx, weight, rows) || !gvq_stream(ctx)) { return 0; }
    const glmvq_source &s = ctx->model->source[weight];
    if (!gvq_floats(input, (uint64_t)rows * s.cols / 2) || !gvq_floats(out, (uint64_t)rows * s.rows) ||
        !gvq_q4p_ready(ctx, &weight, 1, input, &out, rows)) { return 0; }
    return gvq_q4p_run<float, __nv_bfloat16>(ctx, &out, &weight, 1, input, rows);
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
    if (width <= GLMVQ_RMS_NARROW && width % 32 == 0) {
        glmvq_launch_kernel(gvq_mode(rows), glmvq_rms_warp, (rows + GLMVQ_RMS_ROWS - 1) / GLMVQ_RMS_ROWS,
            GLMVQ_RMS_ROWS * 32, 0, ds4_current_stream(), (const float *)input->ptr, w,
            gate ? (const float *)gate->ptr : nullptr, (float *)out->ptr, rows, width, eps, (glmvq_norm)kind);
        return cuda_ok(cudaGetLastError(), "GLM VQ RMS");
    }
    glmvq_launch_kernel(gvq_mode(rows), glmvq_rms, rows, GLMVQ_THREADS, 0, ds4_current_stream(),
        (const float *)input->ptr, w, gate ? (const float *)gate->ptr : nullptr,
        (float *)out->ptr, width, eps, (glmvq_norm)kind);
    return cuda_ok(cudaGetLastError(), "GLM VQ RMS");
}

extern "C" int ds4_gpu_gvq_split(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gpu_tensor *split,
        const ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual, ds4_gvq_rows format,
        uint32_t scale, uint32_t base, uint32_t rows, uint32_t width, uint32_t iterations, float eps,
        ds4_gpu_tensor *norm, uint32_t norm_weight, float norm_eps) {
    const uint64_t count = (uint64_t)rows * width, coeff = (uint64_t)rows * GLMVQ_HC_MIX;
    const float *s = (const float *)gvq_weight(ctx, scale, GVQ_TYPE_F32, 3);
    const float *b = (const float *)gvq_weight(ctx, base, GVQ_TYPE_F32, GLMVQ_HC_MIX);
    const __nv_bfloat16 *w = norm ? (const __nv_bfloat16 *)gvq_weight(ctx, norm_weight, GVQ_TYPE_BF16, width) : nullptr;
    if (!s || !b || !rows || !width || !iterations || !gvq_floats(out, count) ||
        !gvq_floats(residual, count * GLMVQ_HC) || !gvq_floats(split, coeff) || !gvq_floats(mix, coeff) ||
        (norm && (!w || !gvq_floats(norm, count)))) { return 0; }
    auto launch = [&](auto *r) {
        using R = std::remove_const_t<std::remove_pointer_t<decltype(r)>>;
        float *n = norm ? (float *)norm->ptr : nullptr;
        // GLM rows preload their streams under the Sinkhorn (bit-identical);
        // DS4_GLMVQ_SPLIT_PRELOAD=0 keeps the generic loop.
        if (ctx->split_row && width == GLMVQ_HC_WIDTH) {
            glmvq_launch_kernel(gvq_mode(rows), glmvq_hc_split_row<R>, rows, GLMVQ_THREADS, 0, ds4_current_stream(),
                (const float *)mix->ptr, r, s, b, (float *)out->ptr, (float *)split->ptr, iterations, eps, w, n, norm_eps);
            return;
        }
        glmvq_launch_kernel(gvq_mode(rows), glmvq_hc_split<R>,
            rows, GLMVQ_THREADS, 0, ds4_current_stream(), (const float *)mix->ptr,
            r, s, b, (float *)out->ptr, (float *)split->ptr, width, iterations, eps, w, n, norm_eps);
    };
    if (format == DS4_GVQ_ROWS_BF16) { launch((const __nv_bfloat16 *)residual->ptr); }
    else { launch((const float *)residual->ptr); }
    return cuda_ok(cudaGetLastError(), "GLM VQ mHC split");
}

extern "C" int ds4_gpu_gvq_hc_mix(ds4_gvq_ctx *ctx, ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual, uint32_t weight, uint32_t width, float eps, uint32_t rows) {
    const __nv_bfloat16 *w = (const __nv_bfloat16 *)gvq_weight(ctx, weight, GVQ_TYPE_BF16,
        (uint64_t)GLMVQ_HC_MIX * width);
    if (!ctx->direct_bf16 || !w || !width || width % (GLMVQ_THREADS * GLMVQ_DOT_VECTOR) || !rows ||
        ctx->model->records[weight].rows != GLMVQ_HC_MIX || ctx->model->records[weight].cols != width ||
        !gvq_floats(mix, (uint64_t)rows * GLMVQ_HC_MIX) || !gvq_floats(residual, (uint64_t)rows * width) ||
        (uintptr_t)w % 16 || (uintptr_t)residual->ptr % 16) { return 0; }
    // Decode and verify rows keep the one-row order; prefill rows share W tiles.
    if (rows > GVQ_VERIFY_ROWS) {
        glmvq_hc_mix_rows<<<(rows + GLMVQ_MIX_WARPS - 1) / GLMVQ_MIX_WARPS, GLMVQ_MIX_WARPS * 32, 0,
            ds4_current_stream()>>>((const float *)residual->ptr, w, (float *)mix->ptr, rows, width, eps);
        return cuda_ok(cudaGetLastError(), "GLM VQ mHC mix rows");
    }
    glmvq_launch_kernel(gvq_mode(rows), glmvq_hc_mix, dim3(GLMVQ_HC_MIX, rows), GLMVQ_THREADS, 0,
        ds4_current_stream(), (const float *)residual->ptr, w, (float *)mix->ptr, width, eps);
    return cuda_ok(cudaGetLastError(), "GLM VQ mHC mix");
}

extern "C" int ds4_gpu_gvq_expand(ds4_gpu_tensor *out, const ds4_gpu_tensor *branch,
        const ds4_gpu_tensor *residual, ds4_gvq_rows format, const ds4_gpu_tensor *split,
        uint32_t rows, uint32_t width) {
    const uint64_t count = (uint64_t)rows * width * GLMVQ_HC;
    if (!rows || !width || !gvq_floats(out, count) || !gvq_floats(residual, count) ||
        !gvq_floats(branch, count / GLMVQ_HC) || !gvq_floats(split, (uint64_t)rows * GLMVQ_HC_MIX)) { return 0; }
    auto launch = [&](auto *r) {
        glmvq_launch_kernel(gvq_mode(rows), glmvq_hc_expand<std::remove_const_t<std::remove_pointer_t<decltype(r)>>>,
            (uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0, ds4_current_stream(),
            (const float *)branch->ptr, r, (const float *)split->ptr, (float *)out->ptr, width, count);
    };
    if (format == DS4_GVQ_ROWS_BF16) { launch((const __nv_bfloat16 *)residual->ptr); }
    else { launch((const float *)residual->ptr); }
    return cuda_ok(cudaGetLastError(), "GLM VQ mHC expand");
}

// Prefill rows only; 0 asks the caller for separate expand and mix.
extern "C" int ds4_gpu_gvq_expand_mix(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *branch, const ds4_gpu_tensor *residual, ds4_gvq_rows format,
        const ds4_gpu_tensor *split, uint32_t weight, uint32_t rows, uint32_t width, float eps) {
    const uint64_t count = (uint64_t)rows * width * GLMVQ_HC;
    const __nv_bfloat16 *w = (const __nv_bfloat16 *)gvq_weight(ctx, weight, GVQ_TYPE_BF16,
        (uint64_t)GLMVQ_HC_MIX * GLMVQ_HC * width);
    if (!ctx->direct_bf16 || !ctx->expand_mix || !w || rows <= GVQ_VERIFY_ROWS || !width || width % GLMVQ_XMIX_COLS ||
        ctx->model->records[weight].rows != GLMVQ_HC_MIX || ctx->model->records[weight].cols != GLMVQ_HC * width ||
        !gvq_floats(out, count) || !gvq_floats(residual, count) || !gvq_floats(branch, count / GLMVQ_HC) ||
        !gvq_floats(split, (uint64_t)rows * GLMVQ_HC_MIX) || !gvq_floats(mix, (uint64_t)rows * GLMVQ_HC_MIX) ||
        (uintptr_t)w % 16 || ((uintptr_t)residual->ptr | (uintptr_t)branch->ptr | (uintptr_t)out->ptr) % 16) { return 0; }
    auto launch = [&](auto *r) {
        using R = std::remove_cv_t<std::remove_pointer_t<decltype(r)>>;
        glmvq_hc_expand_mix<<<(rows + GLMVQ_MIX_WARPS - 1) / GLMVQ_MIX_WARPS, GLMVQ_MIX_WARPS * 32,
            glmvq_xmix_smem<R>(), ds4_current_stream()>>>(
            (const float *)branch->ptr, r, (const float *)split->ptr, (__nv_bfloat16 *)out->ptr,
            w, (float *)mix->ptr, rows, width, eps);
    };
    if (format == DS4_GVQ_ROWS_BF16) { launch((const __nv_bfloat16 *)residual->ptr); }
    else { launch((const float *)residual->ptr); }
    return cuda_ok(cudaGetLastError(), "GLM VQ mHC expand mix");
}

extern "C" int ds4_gpu_gvq_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up, uint32_t rows, uint32_t width, float limit, ds4_gvq_rows out_rows) {
    const uint64_t count = (uint64_t)rows * width;
    if (!count || !gvq_floats(out, count) || !gvq_floats(gate, count) || !gvq_floats(up, count)) { return 0; }
    const uint32_t blocks = (uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
    if (out_rows == DS4_GVQ_ROWS_BF16) {
        glmvq_launch_kernel(gvq_mode(rows), glmvq_swiglu<__nv_bfloat16>, blocks, GLMVQ_THREADS, 0, ds4_current_stream(),
            (const float *)gate->ptr, (const float *)up->ptr, (__nv_bfloat16 *)out->ptr, count, limit);
    } else {
        glmvq_launch_kernel(gvq_mode(rows), glmvq_swiglu<float>, blocks, GLMVQ_THREADS, 0, ds4_current_stream(),
            (const float *)gate->ptr, (const float *)up->ptr, (float *)out->ptr, count, limit);
    }
    return cuda_ok(cudaGetLastError(), "GLM VQ SwiGLU");
}

// MTP eh_proj input rows: [embedding norm | hidden norm] (SGLang nextn order).
__global__ void gvq_concat(float *out, const float *left, const float *right,
        uint32_t width, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint64_t row = i / (2u * width), col = i % (2u * width);
    out[i] = col < width ? left[row * width + col] : right[row * width + col - width];
}

extern "C" int ds4_gpu_gvq_concat(ds4_gpu_tensor *out, const ds4_gpu_tensor *left,
        const ds4_gpu_tensor *right, uint32_t rows, uint32_t width) {
    const uint64_t count = 2u * (uint64_t)rows * width;
    if (!rows || !width || !gvq_floats(out, count) || !gvq_floats(left, count / 2u) ||
        !gvq_floats(right, count / 2u)) { return 0; }
    gvq_concat<<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)left->ptr, (const float *)right->ptr, width, count);
    return cuda_ok(cudaGetLastError(), "GLM VQ MTP concat");
}

extern "C" int ds4_gpu_gvq_copy(ds4_gpu_tensor *dst, uint64_t dst_offset, const ds4_gpu_tensor *src,
        uint64_t src_offset, uint64_t bytes, uint32_t rows) {
    if (!dst || !src || bytes % sizeof(uint32_t) || dst_offset % sizeof(uint32_t) || src_offset % sizeof(uint32_t) ||
        dst_offset > dst->bytes || bytes > dst->bytes - dst_offset ||
        src_offset > src->bytes || bytes > src->bytes - src_offset) { return 0; }
    if (!bytes) { return 1; }
    const uint64_t count = bytes / sizeof(uint32_t);
    glmvq_launch_kernel(gvq_mode(rows), glmvq_copy, (uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS),
        GLMVQ_THREADS, 0, ds4_current_stream(), (uint32_t *)((char *)dst->ptr + dst_offset),
        (const uint32_t *)((const char *)src->ptr + src_offset), count);
    return cuda_ok(cudaGetLastError(), "GLM VQ copy");
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
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) { out[row] = first + row; }
}

extern "C" int ds4_gpu_gvq_positions(ds4_gpu_tensor *out, uint32_t first, uint32_t rows) {
    if (!out || !rows || rows > out->bytes / sizeof(uint32_t) || first > UINT32_MAX - rows) { return 0; }
    glmvq_launch_kernel(gvq_mode(rows), gvq_positions, (rows + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0,
        ds4_current_stream(),
        (uint32_t *)out->ptr, first, rows);
    return cuda_ok(cudaGetLastError(), "GLM VQ positions");
}

enum { GVQ_ROUTER_LANES = 32, GVQ_ROUTER_PER_LANE = GVQ_EXPERTS / GVQ_ROUTER_LANES };
static_assert(GVQ_EXPERTS % GVQ_ROUTER_LANES == 0, "router lanes own whole expert columns");

// One warp per token; lane l owns experts l, l+32, ... A serial scan took
// ~56 us per decode layer. The (score desc, index asc) warp argmax keeps
// the scan's lowest-index tie rule; probabilities and the slot-ordered sum
// are unchanged, so selections and weights stay bit-identical.
__global__ void gvq_router(int32_t *selected, float *weights, const float *logits,
        const float *bias, float scale) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t lane = threadIdx.x;
    const uint64_t row = (uint64_t)blockIdx.x * GVQ_EXPERTS;
    float prob[GVQ_ROUTER_PER_LANE], score[GVQ_ROUTER_PER_LANE];
    #pragma unroll
    for (uint32_t i = 0; i < GVQ_ROUTER_PER_LANE; i++) {
        const uint32_t e = lane + i * GVQ_ROUTER_LANES;
        prob[i] = glmvq_sigmoid(logits[row + e]);
        score[i] = glmvq_add(prob[i], bias[e]);
    }
    const uint64_t offset = (uint64_t)blockIdx.x * GVQ_USED;
    float sum = 0.0f, mine = 0.0f;
    for (uint32_t slot = 0; slot < GVQ_USED; slot++) {
        float value = score[0];
        uint32_t best = lane;
        #pragma unroll
        for (uint32_t i = 1; i < GVQ_ROUTER_PER_LANE; i++) {
            if (score[i] > value) { value = score[i]; best = lane + i * GVQ_ROUTER_LANES; }
        }
        for (uint32_t stride = GVQ_ROUTER_LANES / 2; stride; stride >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, value, stride);
            const uint32_t at = __shfl_xor_sync(0xffffffffu, best, stride);
            if (other > value || (other == value && at < best)) { value = other; best = at; }
        }
        float p = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < GVQ_ROUTER_PER_LANE; i++) {
            if (lane + i * GVQ_ROUTER_LANES == best) { p = prob[i]; score[i] = -INFINITY; }
        }
        p = __shfl_sync(0xffffffffu, p, best % GVQ_ROUTER_LANES);
        if (lane == slot) { mine = p; selected[offset + slot] = (int32_t)best; }
        sum = glmvq_add(sum, p);
    }
    // Source router: normalize uncorrected sigmoid scores with +1e-20,
    // then apply routed_scaling_factor (modeling_glm5_next.py:158-184).
    sum = glmvq_add(sum, 1.0e-20f);
    if (lane < GVQ_USED) { weights[offset + lane] = glmvq_mul(__fdiv_rn(mine, sum), scale); }
}

extern "C" int ds4_gpu_gvq_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected,
        ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits, uint32_t bias,
        uint32_t rows, float scale) {
    const float *b = (const float *)gvq_weight(ctx, bias, GVQ_TYPE_F32, GVQ_EXPERTS);
    if (!b || !rows || !selected || (uint64_t)rows * GVQ_USED > selected->bytes / sizeof(int32_t) ||
        !gvq_floats(weights, (uint64_t)rows * GVQ_USED) || !gvq_floats(logits, (uint64_t)rows * GVQ_EXPERTS)) { return 0; }
    glmvq_launch_kernel(gvq_mode(rows), gvq_router, rows, GVQ_ROUTER_LANES, 0, ds4_current_stream(),
        (int32_t *)selected->ptr, (float *)weights->ptr, (const float *)logits->ptr, b, scale);
    return cuda_ok(cudaGetLastError(), "GLM VQ router");
}

extern "C" uint64_t ds4_gpu_gvq_attn_bytes(uint32_t rows, uint32_t keys, uint32_t heads, uint32_t dim) {
    return glmvq_attn_bytes(rows, keys, heads, dim).bytes;
}


// The fused prefill kernel needs no dot buffer; cuBLAS serves fewer rows.
extern "C" uint64_t ds4_gpu_gvq_index_bytes(uint32_t rows, uint32_t pools) {
    const char *fused = getenv("DS4_GLMVQ_INDEX_FUSED");
    const bool small = !fused || strcmp(fused, "0");
    return glmvq_index_bytes(small && rows >= GLMVQ_IDX_ROWS ? GLMVQ_IDX_ROWS - 1 : rows, pools);
}

extern "C" int ds4_gpu_gvq_pool(ds4_gvq_ctx *ctx, ds4_gpu_tensor *cache,
        ds4_gpu_tensor *tail_k, ds4_gpu_tensor *tail_gate,
        const ds4_gpu_tensor *raw, const ds4_gpu_tensor *gate,
        uint32_t norm, uint32_t bias, uint32_t ape,
        uint32_t first, uint32_t rows, uint32_t capacity) {
    constexpr uint32_t width = DS4_GLM53_POOL_DIM, ratio = DS4_GLM53_POOL_SIZE;
    if (!ctx || ds4_capture_active() || !rows || capacity > DS4_GVQ_CTX_MAX ||
        first > capacity || rows > capacity - first ||
        !cache || (uint64_t)((capacity + ratio - 1) / ratio) * width > cache->bytes / sizeof(__nv_bfloat16) ||
        !gvq_floats(tail_k, ratio * width) || !gvq_floats(tail_gate, ratio * width) ||
        !gvq_floats(raw, (uint64_t)rows * width) || !gvq_floats(gate, (uint64_t)rows * width) ||
        !gvq_stream(ctx)) { return 0; }
    const auto *n = (const __nv_bfloat16 *)gvq_weight(ctx, norm, GVQ_TYPE_BF16, width);
    const auto *b = (const __nv_bfloat16 *)gvq_weight(ctx, bias, GVQ_TYPE_BF16, width);
    const auto *a = (const __nv_bfloat16 *)gvq_weight(ctx, ape, GVQ_TYPE_BF16, ratio * width);
    if (!n || !b || !a) { return 0; }
    return cuda_ok(glmvq_pool_update((__nv_bfloat16 *)cache->ptr,
        (float *)tail_k->ptr, (float *)tail_gate->ptr, (const float *)raw->ptr,
        (const float *)gate->ptr, n, b, a, first, rows, ctx->stream), "GLM VQ key pools");
}

extern "C" int ds4_gpu_gvq_index(ds4_gvq_ctx *ctx, ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *query, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *cache, uint32_t pools, uint32_t rows, uint32_t first) {
    if (!ctx || ds4_capture_active() || !rows || !pools || first > DS4_GVQ_CTX_MAX ||
        rows > DS4_GVQ_CTX_MAX - first || pools != (first + rows) / DS4_GLM53_POOL_SIZE ||
        !gvq_floats(scores, (uint64_t)rows * pools) ||
        !gvq_floats(query, (uint64_t)rows * GLMVQ_INDEX_HEADS * DS4_GLM53_POOL_DIM) ||
        !gvq_floats(weights, (uint64_t)rows * GLMVQ_INDEX_HEADS) ||
        !cache || (uint64_t)pools * DS4_GLM53_POOL_DIM > cache->bytes / sizeof(__nv_bfloat16) ||
        !gvq_stream(ctx)) { return 0; }
    // Prefill rows score pools in registers; decode/verify rows keep cuBLAS.
    if (ctx->index_fused && rows >= GLMVQ_IDX_ROWS) {
        const dim3 grid((pools + GLMVQ_IDX_POOLS - 1) / GLMVQ_IDX_POOLS, (rows + GLMVQ_IDX_ROWS - 1) / GLMVQ_IDX_ROWS);
        glmvq_index_fused<<<grid, GLMVQ_IDX_THREADS, GLMVQ_IDX_SMEM, ctx->stream>>>((float *)scores->ptr,
            (const float *)query->ptr, (const float *)weights->ptr, (const __nv_bfloat16 *)cache->ptr, pools, rows, first);
        return cuda_ok(cudaGetLastError(), "GLM VQ fused index scores");
    }
    return cublas_ok(glmvq_index_score(ctx->handle, (float *)scores->ptr,
        (const float *)query->ptr, (const float *)weights->ptr,
        (const __nv_bfloat16 *)cache->ptr, pools, rows, first,
        ctx->scratch->ptr, ctx->scratch->bytes, ctx->stream), "GLM VQ index scores");
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

enum { GVQ_FLASH_SPLITS = 48, GVQ_FLASH_ALIGN = 256 };

// Rows up to an MTP verify width split keys across CTAs like decode.
static uint32_t gvq_flash_splits(uint32_t rows) {
    return rows <= GVQ_VERIFY_ROWS ? GVQ_FLASH_SPLITS / rows : 1u;
}

struct gvq_flash_layout { uint64_t q, qlow, low, y, partial, stats, bytes; };

// One query row splits its keys across CTAs; prefill rows each own a CTA.
static gvq_flash_layout gvq_flash_plan(uint32_t rows) {
    uint64_t cursor = 0;
    auto take = [&](uint64_t bytes) {
        const uint64_t at = cursor;
        cursor += (bytes + GVQ_FLASH_ALIGN - 1) / GVQ_FLASH_ALIGN * GVQ_FLASH_ALIGN;
        return at;
    };
    const uint64_t qbytes = (uint64_t)rows * GLMVQ_FA_HEADS * GVQ_LATENT_DIM * sizeof(__nv_bfloat16);
    const uint64_t lbytes = (uint64_t)rows * GLMVQ_FA_HEADS * GLMVQ_FA_LATENT * sizeof(__nv_bfloat16);
    gvq_flash_layout l;
    l.q = take(qbytes); l.qlow = take(lbytes); l.low = take(lbytes); l.y = take(qbytes);
    const uint64_t splits = rows <= GVQ_VERIFY_ROWS ? (uint64_t)rows * gvq_flash_splits(rows) : 0;
    l.partial = take(splits * GLMVQ_FA_HEADS * GLMVQ_FA_LATENT * sizeof(float));
    l.stats = take(splits * GLMVQ_FA_HEADS * 2 * sizeof(float));
    l.bytes = cursor;
    return l;
}

extern "C" uint64_t ds4_gpu_gvq_flash_bytes(uint32_t rows) {
    return gvq_flash_plan(rows).bytes;
}

// Absorbed attention: q·W_k (cuBLAS), the flash kernel over latent rows,
// then low·W_v (cuBLAS). Rows stay [row][head][...] so no transposes occur.
static int gvq_flash(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *query,
        const __nv_bfloat16 *weight, const ds4_gpu_tensor *cache, const uint32_t *positions,
        const ds4_gpu_tensor *selected, uint32_t stride, uint32_t rows, uint32_t keys,
        ds4_gvq_rows query_rows, ds4_gvq_rows out_rows) {
    const gvq_flash_layout l = gvq_flash_plan(rows);
    uint8_t *base = (uint8_t *)ctx->scratch->ptr;
    auto *q = (__nv_bfloat16 *)(base + l.q), *qlow = (__nv_bfloat16 *)(base + l.qlow);
    auto *low = (__nv_bfloat16 *)(base + l.low), *y = (__nv_bfloat16 *)(base + l.y);
    const uint32_t heads = GLMVQ_FA_HEADS, dim = GVQ_LATENT_DIM, latent = GLMVQ_FA_LATENT;
    const uint64_t nq = (uint64_t)rows * heads * dim;
    const long long wstride = (long long)2 * dim * latent;
    const float alpha = 1.0f, beta = 0.0f;
    // Decode and MTP verify rows: direct GEMVs read F32 rows in place
    // (cuBLAS at N=1 ran at 90-215 GB/s). Prefill keeps tensor-core GEMMs.
    const bool gemv = rows <= GLMVQ_ABS_ROWS && ctx->direct_bf16;
    // BF16 rows come only from prefill projections, which take the GEMMs.
    if (gemv && (query_rows == DS4_GVQ_ROWS_BF16 || out_rows == DS4_GVQ_ROWS_BF16)) { return 0; }
    if (gemv) {
        glmvq_launch_kernel(gvq_mode(rows), glmvq_absorb_k, heads, latent / 2, 0, ctx->stream,
            qlow, (const float *)query->ptr, weight, rows);
    } else if (query_rows == DS4_GVQ_ROWS_BF16) {
        q = (__nv_bfloat16 *)query->ptr;
    } else {
        glmvq_convert<<<(nq + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, ctx->stream>>>(
            (const float *)query->ptr, q, nq);
    }
    if (!cuda_ok(cudaGetLastError(), "GLM VQ flash query") ||
        (!gemv && !cublas_ok(cublasGemmStridedBatchedEx(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N,
            latent, rows, dim, &alpha, weight, CUDA_R_16BF, latent, wstride,
            q, CUDA_R_16BF, heads * dim, dim, &beta, qlow, CUDA_R_16BF, heads * latent, latent,
            heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "GLM VQ flash q absorb"))) { return 0; }
    const uint32_t slots = selected ? stride : keys;
    const uint32_t tiles = (slots + GLMVQ_FA_KEYS - 1) / GLMVQ_FA_KEYS;
    const uint32_t splits = min(tiles, gvq_flash_splits(rows));
    const uint32_t per = (tiles + splits - 1) / splits;
    float *partial = splits > 1 ? (float *)(base + l.partial) : nullptr;
    float *stats = splits > 1 ? (float *)(base + l.stats) : nullptr;
    glmvq_launch_kernel(gvq_mode(rows), glmvq_fa_kernel, dim3(rows, splits), GLMVQ_FA_THREADS, GLMVQ_FA_SMEM, ctx->stream,
        low, partial, stats, (const __nv_bfloat16 *)qlow, (const __nv_bfloat16 *)cache->ptr, positions,
        selected ? (const uint32_t *)selected->ptr : nullptr, selected ? stride : 0u,
        splits > 1 ? per : UINT32_MAX, 1.0f / sqrtf((float)dim));
    if (splits > 1) {
        glmvq_launch_kernel(gvq_mode(rows), glmvq_fa_combine, dim3(rows, heads), latent / 2, 0, ctx->stream,
            low, (const float *)partial, (const float *)stats, splits);
    }
    if (gemv) {
        glmvq_launch_kernel(gvq_mode(rows), glmvq_absorb_v, heads * dim / 8, 256, 0, ctx->stream,
            (float *)out->ptr, (const __nv_bfloat16 *)low, weight, rows);
        ctx->flash_calls++;
        return cuda_ok(cudaGetLastError(), "GLM VQ flash v absorb");
    }
    // BF16 output rows take the GEMM's BF16 rounding without an F32 copy.
    if (out_rows == DS4_GVQ_ROWS_BF16) { y = (__nv_bfloat16 *)out->ptr; }
    if (!cuda_ok(cudaGetLastError(), "GLM VQ flash attention") ||
        !cublas_ok(cublasGemmStridedBatchedEx(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N,
            dim, rows, latent, &alpha, weight + (uint64_t)dim * latent, CUDA_R_16BF, latent, wstride,
            low, CUDA_R_16BF, heads * latent, latent, &beta, y, CUDA_R_16BF, heads * dim, dim,
            heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "GLM VQ flash v absorb")) { return 0; }
    if (out_rows == DS4_GVQ_ROWS_F32) {
        glmvq_convert<<<(nq + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, ctx->stream>>>(
            y, (float *)out->ptr, nq);
    }
    ctx->flash_calls++;
    return cuda_ok(cudaGetLastError(), "GLM VQ flash output");
}

extern "C" int ds4_gpu_gvq_dsa_rows(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gvq_rows out_rows,
        const ds4_gpu_tensor *query, ds4_gvq_rows query_rows, const ds4_gpu_tensor *cache, uint32_t weight,
        ds4_gpu_tensor *latent, ds4_gpu_tensor *kv, ds4_gpu_tensor *positions,
        uint32_t first, uint32_t rows, uint32_t heads, uint32_t dim,
        const ds4_gpu_tensor *selected, uint32_t stride) {
    if (!ctx || ds4_capture_active() || !rows || !heads || !dim ||
        first > INT_MAX || rows > (uint32_t)INT_MAX - first ||
        weight >= ctx->model->source.size() || (uint64_t)heads * dim > INT_MAX / 2u) { return 0; }
    const glmvq_source &s = ctx->model->source[weight];
    const uint32_t keys = first + rows;
    const uint64_t nq = (uint64_t)rows * heads * dim;
    const uint64_t nx = (uint64_t)keys * s.cols, ny = (uint64_t)keys * s.rows;
    if (!s.cols || s.cols > INT_MAX || s.rows != (uint64_t)heads * 2 * dim ||
        !cache || nx > cache->bytes / sizeof(__nv_bfloat16) ||
        !gvq_floats(query, nq) || !gvq_floats(out, nq) ||
        (selected && (!stride || (uint64_t)rows * stride > selected->bytes / sizeof(int32_t))) ||
        !positions || rows > positions->bytes / sizeof(uint32_t) ||
        !gvq_stream(ctx)) { return 0; }

    // Flash reads BF16 KV_B and the latent cache directly; it needs no
    // expanded latent/KV buffers (allocated only with DS4_GLMVQ_FLASH=0).
    const bool bf16_kv = ctx->direct_kv && ctx->model->records[weight].type == GVQ_TYPE_BF16 &&
        s.storage == GLMVQ_WEIGHT_BF16 && s.rows % 8 == 0 && s.cols % 8 == 0 &&
        ((uintptr_t)s.data | (uintptr_t)cache->ptr) % 16 == 0;
    if (ctx->flash && bf16_kv && heads == GLMVQ_FA_HEADS && dim == GVQ_LATENT_DIM && s.cols == GLMVQ_FA_LATENT &&
        keys <= GVQ_LATENT_MAX_KEYS && ctx->scratch->bytes >= gvq_flash_plan(rows).bytes &&
        (uintptr_t)ctx->scratch->ptr % 16 == 0) {
        return ds4_gpu_gvq_positions(positions, first, rows) &&
            gvq_flash(ctx, out, query, (const __nv_bfloat16 *)s.data, cache,
                (const uint32_t *)positions->ptr, selected, stride, rows, keys, query_rows, out_rows);
    }
    // Only flash reads or writes BF16 rows; the caller retries with F32.
    if (query_rows == DS4_GVQ_ROWS_BF16 || out_rows == DS4_GVQ_ROWS_BF16) { return 0; }
    if (!gvq_floats(latent, nx) || !gvq_floats(kv, ny) ||
        ctx->scratch->bytes < glmvq_attn_bytes(rows, keys, heads, dim).bytes) { return 0; }
    if (!bf16_kv || (uintptr_t)kv->ptr % 16) {
        return ds4_gpu_gvq_unpack(latent, cache, nx) &&
            ds4_gpu_gvq_linear(ctx, kv, weight, latent, keys, DS4_GVQ_BF16) &&
            ds4_gpu_gvq_positions(positions, first, rows) &&
            ds4_gpu_gvq_attn(ctx, out, query, kv, positions, selected, stride, rows, keys, heads, dim);
    }
    // Full-prefix expansion dominates longer N=1 decode. Other widths keep
    // the expanded path. Sparse queries use the same causal selection mask.
    if (ctx->latent_attn && rows == 1 && heads == GVQ_LATENT_HEADS &&
        dim == GVQ_LATENT_DIM && s.cols == GVQ_LATENT_WIDTH &&
        keys >= GVQ_LATENT_MIN_KEYS && keys <= GVQ_LATENT_MAX_KEYS) {
        if (!ds4_gpu_gvq_positions(positions, first, rows)) { return 0; }
        ctx->latent_calls++;
        return cublas_ok(glmvq_latent_attn(ctx->handle,
            (const __nv_bfloat16 *)s.data, (const __nv_bfloat16 *)cache->ptr,
            (const float *)query->ptr, (float *)out->ptr, (const uint32_t *)positions->ptr,
            rows, keys, heads, dim, s.cols, ctx->scratch->ptr, ctx->scratch->bytes,
            ctx->stream, selected ? (const int32_t *)selected->ptr : nullptr, stride),
            "GLM VQ latent attention");
    }

    // Both latent cache and KV_B output are BF16 in the source model. Keep
    // the same GEMM and attention rounding without two F32 round trips.
    const float alpha = 1.0f, beta = 0.0f;
    if (!cublas_ok(cublasGemmEx(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N,
        s.rows, keys, s.cols, &alpha, s.data, CUDA_R_16BF, s.cols,
        cache->ptr, CUDA_R_16BF, s.cols, &beta, kv->ptr, CUDA_R_16BF, s.rows,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "GLM VQ BF16 KV projection") ||
        !ds4_gpu_gvq_positions(positions, first, rows)) { return 0; }
    ctx->kv_calls++;
    return cublas_ok(glmvq_attention(ctx->handle, (const float *)query->ptr,
        (const __nv_bfloat16 *)kv->ptr, (float *)out->ptr, (const uint32_t *)positions->ptr,
        selected ? (const int32_t *)selected->ptr : nullptr, stride,
        rows, keys, heads, dim, ctx->scratch->ptr, ctx->scratch->bytes,
        ctx->stream), "GLM VQ BF16 KV attention");
}

extern "C" int ds4_gpu_gvq_dsa(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out,
        const ds4_gpu_tensor *query, const ds4_gpu_tensor *cache, uint32_t weight,
        ds4_gpu_tensor *latent, ds4_gpu_tensor *kv, ds4_gpu_tensor *positions,
        uint32_t first, uint32_t rows, uint32_t heads, uint32_t dim,
        const ds4_gpu_tensor *selected, uint32_t stride) {
    return ds4_gpu_gvq_dsa_rows(ctx, out, DS4_GVQ_ROWS_F32, query, DS4_GVQ_ROWS_F32, cache, weight,
        latent, kv, positions, first, rows, heads, dim, selected, stride);
}

extern "C" uint64_t ds4_gpu_gvq_kda_bytes(uint32_t rows, uint32_t heads) {
    uint64_t planes = 0;
    if (!solar_kda_chunk_layout(rows, heads, GLMVQ_KDA_DIM, nullptr, nullptr, &planes)) { return 0; }
    const uint64_t records = (uint64_t)((rows + GLMVQ_KC - 1) / GLMVQ_KC) * heads * GLMVQ_KR_BYTES;
    return std::max(planes + (uint64_t)rows * heads * sizeof(float), records);
}

extern "C" int ds4_gpu_gvq_kda_fused(ds4_gvq_ctx *ctx, uint32_t rows) {
    return ctx && ctx->kda_mma && rows >= GLMVQ_KC;
}

extern "C" int ds4_gpu_gvq_kda_gated(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gpu_tensor *state,
        ds4_gpu_tensor *q_tail, ds4_gpu_tensor *k_tail, ds4_gpu_tensor *v_tail,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v, ds4_gvq_rows qkv_rows,
        const ds4_gpu_tensor *forget, ds4_gvq_rows forget_rows, const ds4_gpu_tensor *beta,
        const ds4_gpu_tensor *q_weight, const ds4_gpu_tensor *k_weight,
        const ds4_gpu_tensor *v_weight, const ds4_gpu_tensor *a_log,
        const ds4_gpu_tensor *dt_bias, const ds4_gpu_tensor *gate, ds4_gvq_rows gate_rows, uint32_t norm,
        uint32_t rows, uint32_t heads, float lower_bound, float eps, ds4_gvq_rows out_rows) {
    if (!ds4_gpu_gvq_kda_fused(ctx, rows)) { return -1; }
    const uint64_t width = (uint64_t)heads * GLMVQ_KDA_DIM, count = (uint64_t)rows * width;
    const uint64_t tail = width * GLMVQ_CONV, chunks = (rows + GLMVQ_KC - 1) / GLMVQ_KC;
    const __nv_bfloat16 *w = (const __nv_bfloat16 *)gvq_weight(ctx, norm, GVQ_TYPE_BF16, GLMVQ_KDA_DIM);
    if (!w || !gvq_floats(out, count) || !gvq_floats(q, count) || !gvq_floats(k, count) ||
        !gvq_floats(v, count) || !gvq_floats(forget, count) || !gvq_floats(gate, count) ||
        !gvq_floats(beta, (uint64_t)rows * heads) || !gvq_floats(state, width * GLMVQ_KDA_DIM) ||
        !gvq_floats(q_tail, tail) || !gvq_floats(k_tail, tail) || !gvq_floats(v_tail, tail) ||
        !gvq_floats(q_weight, tail) || !gvq_floats(k_weight, tail) || !gvq_floats(v_weight, tail) ||
        !gvq_floats(a_log, heads) || !gvq_floats(dt_bias, width) ||
        ctx->scratch->bytes < chunks * heads * GLMVQ_KR_BYTES) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    auto *records = (uint8_t *)ctx->scratch->ptr;
    auto local = [&](auto *qkv, auto *f) {
        using Q = std::remove_cv_t<std::remove_pointer_t<decltype(qkv)>>;
        glmvq_kda_local<<<dim3((uint32_t)chunks, heads), GLMVQ_KLOCAL, GLMVQ_KLOCAL_SMEM, stream>>>(records,
            (const float *)q_tail->ptr, (const float *)k_tail->ptr, (const float *)v_tail->ptr,
            qkv, (const Q *)k->ptr, (const Q *)v->ptr, f, (const float *)beta->ptr,
            (const float *)q_weight->ptr, (const float *)k_weight->ptr, (const float *)v_weight->ptr,
            (const float *)a_log->ptr, (const float *)dt_bias->ptr, rows, heads, lower_bound);
        glmvq_kda_tail<<<(uint32_t)((width + 255) / 256), 256, 0, stream>>>(
            (float *)q_tail->ptr, (float *)k_tail->ptr, (float *)v_tail->ptr,
            qkv, (const Q *)k->ptr, (const Q *)v->ptr, rows, width);
    };
    auto forget_as = [&](auto *qkv) {
        if (forget_rows == DS4_GVQ_ROWS_BF16) { local(qkv, (const __nv_bfloat16 *)forget->ptr); }
        else { local(qkv, (const float *)forget->ptr); }
    };
    if (qkv_rows == DS4_GVQ_ROWS_BF16) { forget_as((const __nv_bfloat16 *)q->ptr); }
    else { forget_as((const float *)q->ptr); }
    auto launch = [&](auto *g, auto *o) {
        glmvq_kda_scan<<<heads, GLMVQ_KSCAN, GLMVQ_KSCAN_SMEM, stream>>>(o,
            (float *)state->ptr, records, g, w, rows, heads, eps);
    };
    auto scan = [&](auto *g) {
        if (out_rows == DS4_GVQ_ROWS_BF16) { launch(g, (__nv_bfloat16 *)out->ptr); }
        else { launch(g, (float *)out->ptr); }
    };
    if (gate_rows == DS4_GVQ_ROWS_BF16) { scan((const __nv_bfloat16 *)gate->ptr); }
    else { scan((const float *)gate->ptr); }
    ctx->kda_mma_calls++;
    return cuda_ok(cudaGetLastError(), "GLM VQ tensor-core KDA");
}

// Chunked prefill (Solar delta-rule UT kernels) on GLM's own token inputs.
// FP32 accumulation order moves; q/k/v/beta/decay and the BF16 output
// boundary match the recurrence. 1024 rows: 16.3 -> 6.1 ms in isolation.
static int gvq_kda_chunk(ds4_gvq_ctx *ctx, float *out, float *state,
        float *q_tail, float *k_tail, float *v_tail,
        const float *q, const float *k, const float *v, const float *forget, const float *beta,
        const float *q_weight, const float *k_weight, const float *v_weight,
        const float *a_log, const float *dt_bias, uint32_t rows, uint32_t heads, float lower_bound) {
    uint64_t plane = 0, mq_bytes = 0, total = 0;
    if (!solar_kda_chunk_layout(rows, heads, GLMVQ_KDA_DIM, &plane, &mq_bytes, &total) ||
        ctx->scratch->bytes < ds4_gpu_gvq_kda_bytes(rows, heads) || !solar_kda_chunk_shmem_prepare()) {
        return -1;
    }
    auto *base = (uint8_t *)ctx->scratch->ptr;
    float *qn = (float *)base, *kn = (float *)(base + plane), *vv = (float *)(base + 2 * plane);
    float *gl = (float *)(base + 3 * plane), *ul = (float *)(base + 4 * plane), *ww = (float *)(base + 5 * plane);
    float *mq = (float *)(base + 6 * plane), *betas = (float *)(base + total);
    const uint64_t width = (uint64_t)heads * GLMVQ_KDA_DIM, count = (uint64_t)rows * width;
    const uint32_t chunks = (rows + SOLAR_KDA_CHUNK - 1) / SOLAR_KDA_CHUNK;
    const cudaStream_t stream = ds4_current_stream();
    glmvq_kda_prep<<<dim3(rows, heads), GLMVQ_KDA_DIM, 0, stream>>>(qn, kn, vv, gl, betas,
        q_tail, k_tail, v_tail, q, k, v, forget, beta, q_weight, k_weight, v_weight,
        a_log, dt_bias, rows, heads, lower_bound);
    solar_kda_chunk_conv_tail_kernel<<<(uint32_t)((width + 255) / 256), 256, 0, stream>>>(
        q_tail, k_tail, v_tail, q, k, v, rows, width, GLMVQ_CONV);
    solar_kda_chunk_factor_kernel<<<dim3(chunks, heads), 256, SOLAR_KDA_FACTOR_SHARED, stream>>>(
        gl, ul, ww, mq, qn, kn, vv, betas, rows, heads, DS4_KDA_GLMVQ);
    solar_kda_chunk_scan_kernel<<<dim3(heads, 2), 256, SOLAR_KDA_SCAN_SHARED, stream>>>(
        out, state, ul, ww, gl, qn, kn, rows, heads);
    solar_kda_chunk_intra_kernel<<<dim3(chunks, heads), 256, SOLAR_KDA_INTRA_SHARED, stream>>>(
        out, mq, ul, rows, heads);
    glmvq_round_rows<<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream>>>(out, count);
    ctx->kda_chunks++;
    return cuda_ok(cudaGetLastError(), "GLM VQ chunked KDA");
}

extern "C" int ds4_gpu_gvq_kda(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gpu_tensor *state,
        ds4_gpu_tensor *q_tail, ds4_gpu_tensor *k_tail, ds4_gpu_tensor *v_tail,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *forget, const ds4_gpu_tensor *beta,
        const ds4_gpu_tensor *q_weight, const ds4_gpu_tensor *k_weight,
        const ds4_gpu_tensor *v_weight, const ds4_gpu_tensor *a_log,
        const ds4_gpu_tensor *dt_bias, uint32_t rows, uint32_t heads, float lower_bound,
        ds4_gpu_tensor *snap, uint64_t stride) {
    const uint64_t width = (uint64_t)heads * GLMVQ_KDA_DIM, count = (uint64_t)rows * width;
    const uint64_t tail = width * GLMVQ_CONV;
    if (!ctx || !rows || !heads || !gvq_floats(out, count) || !gvq_floats(q, count) ||
        !gvq_floats(k, count) || !gvq_floats(v, count) || !gvq_floats(forget, count) ||
        !gvq_floats(beta, (uint64_t)rows * heads) || !gvq_floats(state, width * GLMVQ_KDA_DIM) ||
        !gvq_floats(q_tail, tail) || !gvq_floats(k_tail, tail) || !gvq_floats(v_tail, tail) ||
        !gvq_floats(q_weight, tail) || !gvq_floats(k_weight, tail) || !gvq_floats(v_weight, tail) ||
        !gvq_floats(a_log, heads) || !gvq_floats(dt_bias, width)) { return 0; }
    // Snapshot slots mirror the state pool: tails sit after the matrix.
    glmvq_kda_snap shot = {};
    if (snap && rows > 1) {
        const auto *origin = (const float *)state->ptr;
        const int64_t qo = (const float *)q_tail->ptr - origin, ko = (const float *)k_tail->ptr - origin;
        const int64_t vo = (const float *)v_tail->ptr - origin;
        const uint64_t floats = stride / sizeof(float), last = (uint64_t)(rows - 2) * floats;
        if (stride % sizeof(float) || qo < (int64_t)(width * GLMVQ_KDA_DIM) || ko < qo || vo < ko ||
            !gvq_floats(snap, last + (uint64_t)vo + tail)) { return 0; }
        auto *base = (float *)snap->ptr;
        shot = {base, base + qo, base + ko, base + vo, floats};
    }
    if (ctx->kda_chunk && rows >= SOLAR_KDA_CHUNK && !shot.state) {
        const int chunked = gvq_kda_chunk(ctx, (float *)out->ptr, (float *)state->ptr,
            (float *)q_tail->ptr, (float *)k_tail->ptr, (float *)v_tail->ptr,
            (const float *)q->ptr, (const float *)k->ptr, (const float *)v->ptr,
            (const float *)forget->ptr, (const float *)beta->ptr,
            (const float *)q_weight->ptr, (const float *)k_weight->ptr, (const float *)v_weight->ptr,
            (const float *)a_log->ptr, (const float *)dt_bias->ptr, rows, heads, lower_bound);
        if (chunked >= 0) { return chunked; }
    }
    glmvq_launch_kernel(gvq_mode(rows), glmvq_kda, heads, GLMVQ_KDA_DIM, 0, ds4_current_stream(), (float *)out->ptr,
        (float *)state->ptr, (float *)q_tail->ptr, (float *)k_tail->ptr, (float *)v_tail->ptr,
        (const float *)q->ptr, (const float *)k->ptr, (const float *)v->ptr,
        (const float *)forget->ptr, (const float *)beta->ptr,
        (const float *)q_weight->ptr, (const float *)k_weight->ptr, (const float *)v_weight->ptr,
        (const float *)a_log->ptr, (const float *)dt_bias->ptr, rows, heads, lower_bound, shot);
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

// A layer's residual book, or nullptr for a plain GVQ1 layer.
static const uint8_t *gvq_res_book(const ds4_gvq_ctx *ctx, const uint32_t *matrices) {
    const auto &sources = ctx->model->source;
    return matrices[0] < sources.size() ? sources[matrices[0]].vq.res_book : nullptr;
}

// Every matrix of a layer carries the same residual book, or none does.
static bool gvq_res_matches(const glmvq_source &s, const uint8_t *res_book) {
    if (!res_book) { return !s.vq.res; }
    return s.vq.res && s.vq.res_book == res_book && ((uintptr_t)s.vq.res & 127u) == 0 &&
        ((uintptr_t)res_book & 15u) == 0;
}

static bool gvq_mma_fits(const ds4_gvq_ctx *ctx, const uint32_t *matrices) {
    const auto &sources = ctx->model->source;
    if (matrices[0] >= sources.size()) { return false; }
    const void *book = sources[matrices[0]].vq.book;
    const uint8_t *res_book = gvq_res_book(ctx, matrices);
    // The BF16 register MMA has no residual stage.
    if (res_book && !ctx->mma8_res) { return false; }
    for (uint32_t i = 0; i < 3 * GVQ_EXPERTS; i++) {
        const uint32_t index = matrices[i];
        const uint32_t rows = i < 2 * GVQ_EXPERTS ? GVQ_FFN : GVQ_HIDDEN;
        const uint32_t cols = i < 2 * GVQ_EXPERTS ? GVQ_HIDDEN : GVQ_FFN;
        if (index >= sources.size() || !glmvq_mma_fits(sources[index], rows, cols, book) ||
            !gvq_res_matches(sources[index], res_book)) { return false; }
    }
    return true;
}

static int gvq_mma_moe(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out,
        const ds4_gpu_tensor *input, const ds4_gpu_tensor *weights,
        const uint32_t *matrices, const uint32_t *offsets, uint32_t rows, float clamp, const float *addend) {
    const uint64_t pairs = (uint64_t)rows * GVQ_USED;
    uint8_t *cursor = (uint8_t *)ctx->mma->ptr;
    auto *x = (__nv_bfloat16 *)cursor; cursor += pairs * GVQ_HIDDEN * sizeof(*x);
    auto *gate = (float *)cursor; cursor += pairs * GVQ_FFN * sizeof(*gate);
    auto *mid = (__nv_bfloat16 *)cursor; cursor += pairs * GVQ_FFN * sizeof(*mid);
    auto *down = (float *)cursor; cursor += pairs * GVQ_HIDDEN * sizeof(*down);
    auto *work = (glmvq_mma_item *)cursor; cursor += pairs * sizeof(*work);
    auto *order = (uint32_t *)cursor;
    ctx->items.clear();
    const uint8_t *res_book = gvq_res_book(ctx, matrices);
    const uint32_t per_item = ctx->mma8 ? ctx->mma8_tokens : (uint32_t)GVQ_MMA_TOKENS;
    for (uint32_t e = 0; e < GVQ_EXPERTS; e++) {
        for (uint32_t start = offsets[e]; start < offsets[e + 1]; start += per_item) {
            ctx->items.push_back({{matrices[e], matrices[GVQ_EXPERTS + e], matrices[2 * GVQ_EXPERTS + e]},
                start, min(per_item, offsets[e + 1] - start)});
        }
    }
    std::fill(ctx->token_used.begin(), ctx->token_used.begin() + rows, 0);
    for (uint32_t i = 0; i < pairs; i++) {
        const uint32_t token = ctx->grouped[i] / GVQ_USED;
        ctx->order[token * GVQ_USED + ctx->token_used[token]++] = i;
    }
    const cudaStream_t stream = ds4_current_stream();
    if (!cuda_ok(cudaMemcpyAsync(work, ctx->items.data(), ctx->items.size() * sizeof(*work),
            cudaMemcpyHostToDevice, stream), "GLM VQ MMA work") ||
        !cuda_ok(cudaMemcpyAsync(order, ctx->order.data(), pairs * sizeof(*order),
            cudaMemcpyHostToDevice, stream), "GLM VQ MMA order")) { return 0; }
    const auto *source = (const glmvq_source *)ctx->model->device->ptr;
    const auto *book = (const uint8_t *)ctx->model->source[matrices[0]].vq.book;
    const auto *grouped = (const uint32_t *)ctx->pairs->ptr;
    const uint32_t nitems = ctx->items.size();
    ctx->mma_calls++;
    ctx->mma_items += nitems;
    const uint64_t count = (uint64_t)rows * GVQ_HIDDEN;
    if (ctx->mma8) {
        // Tokens quantize once for their eight experts; the BF16 gathered
        // copy is never written. FP8 planes reuse that region.
        auto *x8 = (uint8_t *)x;
        auto *xs = (float *)(x8 + count);
        auto *mid8 = (uint8_t *)(xs + count / GLMVQ_MMA8_K);
        auto *ms = (float *)(mid8 + pairs * GVQ_FFN);
        const uint64_t xb = count / GLMVQ_MMA8_K;
        glmvq_mma8_quant<float><<<(uint32_t)((xb * 32 + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0, stream>>>(
            x8, xs, (const float *)input->ptr, xb);
        auto launch = [&](auto tokens, auto mode, const uint8_t *in, const float *scale, uint32_t rows_out, uint32_t cols_in,
                uint8_t *q8 = nullptr, float *q8_scale = nullptr) {
            constexpr uint32_t BT = decltype(tokens)::value;
            constexpr glmvq_mma_stage MODE = decltype(mode)::value;
            constexpr uint32_t threads8 = glmvq_mma8_threads<GVQ_MMA_ROWS, 1>();
            if (res_book) {
                constexpr uint32_t shared8 = glmvq_mma8_bytes<GVQ_MMA_ROWS, BT, 13, true>();
                glmvq_mma8_kernel<GVQ_MMA_ROWS, BT, 1, MODE, 13, GLMVQ_CLAMP_GATE, true><<<
                    min(nitems * (rows_out / GVQ_MMA_ROWS), (uint32_t)ctx->mma8_res_blocks[MODE]), threads8, shared8,
                    stream>>>(source, work, nitems, book, in, scale, grouped, GVQ_USED, (__nv_bfloat16 *)gate, mid,
                    (__nv_bfloat16 *)down, rows_out, cols_in, clamp, q8, q8_scale, res_book);
                return;
            }
            constexpr uint32_t shared8 = glmvq_mma8_bytes<GVQ_MMA_ROWS, BT>();
            glmvq_mma8_kernel<GVQ_MMA_ROWS, BT, 1, MODE><<<
                min(nitems * (rows_out / GVQ_MMA_ROWS), (uint32_t)ctx->mma8_blocks[MODE]), threads8, shared8, stream>>>(
                source, work, nitems, book, in, scale, grouped, GVQ_USED, (__nv_bfloat16 *)gate, mid,
                (__nv_bfloat16 *)down, rows_out, cols_in, clamp, q8, q8_scale);
        };
        auto stages = [&](auto tokens) {
            launch(tokens, std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_GATE>{}, x8, xs, GVQ_FFN, GVQ_HIDDEN);
            // The up epilogue writes down's E4M3 rows and scales directly.
            launch(tokens, std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_UP>{}, x8, xs, GVQ_FFN, GVQ_HIDDEN, mid8, ms);
            launch(tokens, std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_DOWN>{}, mid8, ms, GVQ_HIDDEN, GVQ_FFN);
        };
        if (ctx->mma8_tokens == GVQ_MMA8_TOKENS) { stages(std::integral_constant<uint32_t, GVQ_MMA8_TOKENS>{}); }
        else { stages(std::integral_constant<uint32_t, GVQ_MMA_TOKENS>{}); }
        glmvq_mma_reduce<<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream>>>(
            (float *)out->ptr, (const __nv_bfloat16 *)down, (const float *)weights->ptr, grouped, order,
            GVQ_USED, GVQ_HIDDEN, count, addend);
        return cuda_ok(cudaGetLastError(), "GLM VQ FP8 grouped MMA");
    }
    const uint64_t elements = pairs * GVQ_HIDDEN;
    glmvq_mma_gather<<<(elements + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream>>>(
        x, (const float *)input->ptr, grouped, GVQ_USED, GVQ_HIDDEN, elements);
    constexpr uint32_t shared = glmvq_mma_bytes<GVQ_MMA_ROWS, GVQ_MMA_TOKENS>();
    const uint32_t gate_blocks = min(nitems * (GVQ_FFN / GVQ_MMA_ROWS), (uint32_t)ctx->mma_blocks[0]);
    const uint32_t up_blocks = min(nitems * (GVQ_FFN / GVQ_MMA_ROWS), (uint32_t)ctx->mma_blocks[1]);
    const uint32_t down_blocks = min(nitems * (GVQ_HIDDEN / GVQ_MMA_ROWS), (uint32_t)ctx->mma_blocks[2]);
    glmvq_mma_kernel<GVQ_MMA_ROWS, GVQ_MMA_TOKENS, GLMVQ_MMA_GATE><<<gate_blocks, GVQ_MMA_ROWS * 2, shared, stream>>>(
        source, work, nitems, book, x, gate, mid, down, GVQ_FFN, GVQ_HIDDEN, clamp);
    glmvq_mma_kernel<GVQ_MMA_ROWS, GVQ_MMA_TOKENS, GLMVQ_MMA_UP><<<up_blocks, GVQ_MMA_ROWS * 2, shared, stream>>>(
        source, work, nitems, book, x, gate, mid, down, GVQ_FFN, GVQ_HIDDEN, clamp);
    glmvq_mma_kernel<GVQ_MMA_ROWS, GVQ_MMA_TOKENS, GLMVQ_MMA_DOWN><<<down_blocks, GVQ_MMA_ROWS * 2, shared, stream>>>(
        source, work, nitems, book, mid, gate, mid, down, GVQ_HIDDEN, GVQ_FFN, clamp);
    glmvq_mma_reduce<<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream>>>(
        (float *)out->ptr, down, (const float *)weights->ptr, grouped, order, GVQ_USED, GVQ_HIDDEN, count, addend);
    return cuda_ok(cudaGetLastError(), "GLM VQ grouped MMA");
}

static bool gvq_decode_matrix(const glmvq_source &s, uint32_t projection, const void *book,
        const uint8_t *res_book) {
    const uint32_t rows = projection == 2 ? GVQ_HIDDEN : GVQ_FFN;
    const uint32_t cols = projection == 2 ? GVQ_FFN : GVQ_HIDDEN;
    return s.storage == GLMVQ_WEIGHT_VQ && s.rows == rows && s.cols == cols &&
        s.vq.rows == rows && s.vq.cols == cols && s.vq.precision == GLMVQ_E4M3 &&
        s.vq.main && s.vq.plane && s.vq.gain && book && s.vq.book == book &&
        (uintptr_t)s.vq.main % 128 == 0 && (uintptr_t)s.vq.plane % 128 == 0 &&
        (uintptr_t)book % 16 == 0 && (uintptr_t)s.vq.gain % sizeof(__half) == 0 &&
        gvq_res_matches(s, res_book);
}

static bool gvq_decode_fits(ds4_gvq_ctx *ctx, const uint32_t *matrices) {
    const auto &sources = ctx->model->source;
    const void *book = nullptr;
    const uint8_t *res_book = gvq_res_book(ctx, matrices);
    if (res_book && !ctx->decode_res) { return false; }
    for (uint32_t slot = 0; slot < GVQ_USED; slot++) {
        const uint32_t pair = ctx->grouped[slot], expert = ctx->routes[pair];
        auto &item = ctx->decode_items[slot];
        glmvq_matrix *outputs[] = {&item.gate, &item.up, &item.down};
        for (uint32_t projection = 0; projection < 3; projection++) {
            const uint32_t index = matrices[projection * GVQ_EXPERTS + expert];
            if (index >= sources.size()) { return false; }
            const auto &s = sources[index];
            if (!book) { book = s.vq.book; }
            if (!gvq_decode_matrix(s, projection, book, res_book)) { return false; }
            *outputs[projection] = s.vq;
        }
        item.pair = pair;
    }
    return true;
}

// N=1 routing on the GPU: thread t places expert selected[t] at its rank.
// Ascending expert order is the host grouping's order, so the reduction is
// unchanged; the two per-layer host syncs (route readback, item upload) go.
__global__ void gvq_decode_route(glmvq_decode_item *items, const int32_t *selected,
        const uint32_t *table, const glmvq_source *sources) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x;
    const uint32_t e = min((uint32_t)selected[t], (uint32_t)GVQ_EXPERTS - 1u);
    uint32_t rank = 0;
    for (uint32_t j = 0; j < GVQ_USED; j++) { rank += (uint32_t)selected[j] < (uint32_t)selected[t]; }
    items[rank] = {sources[table[e]].vq, sources[table[GVQ_EXPERTS + e]].vq,
        sources[table[2 * GVQ_EXPERTS + e]].vq, t};
}

// Union of 2-4 rows' experts in ascending expert order (see glm_vq_decode.cuh).
__global__ void gvq_verify_route(glmvq_verify_item *items, glmvq_verify_plan *plan,
        const int32_t *selected, uint32_t rows, const uint32_t *table, const glmvq_source *sources) {
    __shared__ uint32_t expert[GLMVQ_VER_PAIRS], lead[GLMVQ_VER_PAIRS], width[GLMVQ_VER_PAIRS];
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x, pairs = rows * GVQ_USED;
    const bool live = t < pairs;
    if (live) { expert[t] = min((uint32_t)selected[t], (uint32_t)GVQ_EXPERTS - 1u); }
    __syncthreads();
    uint32_t e = 0, before = 0, same = 0;
    if (live) {
        e = expert[t];
        for (uint32_t j = 0; j < pairs; j++) {
            same += expert[j] == e;
            before += j < t && expert[j] == e;
        }
        lead[t] = before == 0;
    }
    __syncthreads();
    if (live) {
        uint32_t rank = 0, below = 0;
        for (uint32_t j = 0; j < pairs; j++) { rank += lead[j] && expert[j] < e; }
        if (!before) {
            items[rank].gate = sources[table[e]].vq;
            items[rank].up = sources[table[GVQ_EXPERTS + e]].vq;
            items[rank].down = sources[table[2 * GVQ_EXPERTS + e]].vq;
            items[rank].count = same;
            width[rank] = same;
        }
        items[rank].pair[before] = t;
        const uint32_t row = t / GVQ_USED;
        for (uint32_t k = 0; k < GVQ_USED; k++) { below += expert[row * GVQ_USED + k] < e; }
        plan->order[row * GVQ_USED + below] = t % GVQ_USED;
    }
    __syncthreads();
    if (t) { return; }
    uint32_t n = 0;
    for (uint32_t j = 0; j < pairs; j++) { n += lead[j]; }
    // Greedy groups at item boundaries: an item (<= 4 pairs) never splits.
    uint32_t groups = 0, used = GLMVQ_VER_GROUP;
    for (uint32_t i = 0; i < n; i++) {
        if (used + width[i] > GLMVQ_VER_GROUP) { plan->group[groups++] = i; used = 0; }
        items[i].slot = used;
        used += width[i];
    }
    plan->group[groups] = n;
    plan->items = n;
    plan->groups = groups;
}

// The layer's device table, uploaded once; nullptr keeps the host-routed
// fallback when any of its 864 matrices misses the direct layout.
static const uint32_t *gvq_decode_table(ds4_gvq_ctx *ctx, const uint32_t *matrices) {
    const auto &sources = ctx->model->source;
    const uint64_t first = GVQ_DECODE_WORK + GVQ_DECODE_ROWS;
    const auto *tables = (const uint32_t *)((const uint8_t *)ctx->decode->ptr + first);
    for (size_t i = 0; i < ctx->decode_keys.size(); i++) {
        if (ctx->decode_keys[i] == matrices[0]) {
            return ctx->decode_fit[i] ? tables + i * GVQ_DECODE_TABLE : nullptr;
        }
    }
    const size_t slot = ctx->decode_keys.size();
    if (slot == GVQ_DECODE_LAYERS || matrices[0] >= sources.size()) { return nullptr; }
    const void *book = sources[matrices[0]].vq.book;
    const uint8_t *res_book = gvq_res_book(ctx, matrices);
    bool fit = !res_book || ctx->decode_res;
    for (uint32_t i = 0; i < GVQ_DECODE_TABLE && fit; i++) {
        fit = matrices[i] < sources.size() && gvq_decode_matrix(sources[matrices[i]], i / GVQ_EXPERTS, book, res_book);
    }
    if (fit && !ds4_gpu_tensor_write(ctx->decode, first + slot * GVQ_DECODE_TABLE * sizeof(uint32_t),
            matrices, GVQ_DECODE_TABLE * sizeof(uint32_t))) { return nullptr; }
    ctx->decode_keys.push_back(matrices[0]);
    ctx->decode_fit.push_back(fit);
    return fit ? tables + slot * GVQ_DECODE_TABLE : nullptr;
}

// Expert kernels for the context's gain arithmetic (glmvq_decode_gain).
template <typename F>
static cudaError_t gvq_gain(const ds4_gvq_ctx *ctx, F &&launch) {
    return ctx->gain_dot ? launch(std::integral_constant<glmvq_decode_gain, GLMVQ_GAIN_DOT>{})
        : launch(std::integral_constant<glmvq_decode_gain, GLMVQ_GAIN_WEIGHT>{});
}

// Expert kernels for the layer's storage: R13 or R13+8 (residual book).
template <typename F>
static cudaError_t gvq_res(const uint8_t *res_book, F &&launch) {
    return res_book ? launch(std::true_type{}) : launch(std::false_type{});
}

static int gvq_decode_device(ds4_gvq_ctx *ctx, float *out, const float *input,
        const int32_t *selected, const float *weights, const uint32_t *table,
        const void *book, float clamp, const float *shared, const uint8_t *res_book) {
    uint8_t *base = (uint8_t *)ctx->decode->ptr;
    auto *items = (glmvq_decode_item *)base;
    auto *mid = (__nv_bfloat16 *)(base + GVQ_DECODE_WORK);
    auto *partial = (float *)(mid + GVQ_USED * GVQ_FFN);
    const cudaStream_t stream = ds4_current_stream();
    glmvq_launch_kernel(ctx->launch, gvq_decode_route, 1, GVQ_USED, 0, stream, items, selected,
        table, (const glmvq_source *)ctx->model->device->ptr);
    ctx->decode_calls++;
    return cuda_ok(gvq_gain(ctx, [&](auto gain) {
        return gvq_res(res_book, [&](auto res) {
            return glmvq_decode_launch<13, 32, GLMVQ_DEC_NORMAL, decltype(gain)::value, glmvq_glm_form,
                decltype(res)::value>(items, book, input, mid, partial, out, weights, clamp, ctx->decode_blocks,
                stream, nullptr, nullptr, shared, ctx->launch, res_book);
        });
    }), "GLM VQ routed decode");
}

// MTP verify widths: one pass over the rows' expert union.
static int gvq_verify_device(ds4_gvq_ctx *ctx, float *out, const float *input,
        const int32_t *selected, const float *weights, const uint32_t *table,
        const void *book, uint32_t rows, float clamp, const float *shared, const uint8_t *res_book) {
    uint8_t *base = (uint8_t *)ctx->decode->ptr + GVQ_VERIFY_AT;
    auto *items = (glmvq_verify_item *)base;
    auto *plan = (glmvq_verify_plan *)(base + GVQ_VERIFY_ITEMS);
    auto *mid = (__nv_bfloat16 *)(base + GVQ_VERIFY_ITEMS + GVQ_VERIFY_PLAN);
    auto *partial = (float *)((uint8_t *)mid + GVQ_VERIFY_MID);
    const cudaStream_t stream = ds4_current_stream();
    glmvq_launch_kernel(ctx->launch, gvq_verify_route, 1, GLMVQ_VER_PAIRS, 0, stream, items, plan, selected, rows,
        table, (const glmvq_source *)ctx->model->device->ptr);
    ctx->decode_calls++;
    return cuda_ok(gvq_gain(ctx, [&](auto gain) {
        return gvq_res(res_book, [&](auto res) {
            return glmvq_verify_launch<13, 32, decltype(gain)::value, glmvq_glm_form, decltype(res)::value>(items,
                plan, book, input, rows, mid, partial, out, weights, clamp, ctx->decode_blocks, stream, shared,
                ctx->launch, res_book);
        });
    }), "GLM VQ verify decode");
}

static int gvq_decode_moe(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out,
        const ds4_gpu_tensor *input, const ds4_gpu_tensor *weights, float clamp, const float *shared) {
    if (!ds4_gpu_tensor_write(ctx->decode, 0, ctx->decode_items, sizeof(ctx->decode_items))) { return 0; }
    uint8_t *base = (uint8_t *)ctx->decode->ptr;
    auto *items = (glmvq_decode_item *)base;
    auto *mid = (__nv_bfloat16 *)(base + GVQ_DECODE_WORK);
    auto *partial = (float *)(mid + GVQ_USED * GVQ_FFN);
    ctx->decode_calls++;
    const uint8_t *res_book = ctx->decode_items[0].gate.res_book;
    return cuda_ok(gvq_gain(ctx, [&](auto gain) {
        return gvq_res(res_book, [&](auto res) {
            return glmvq_decode_launch<13, 32, GLMVQ_DEC_NORMAL, decltype(gain)::value, glmvq_glm_form,
                decltype(res)::value>(items, ctx->decode_items[0].gate.book, (const float *)input->ptr, mid, partial,
                (float *)out->ptr, (const float *)weights->ptr, clamp, ctx->decode_blocks,
                ds4_current_stream(), nullptr, nullptr, shared, GLMVQ_LAUNCH_PLAIN, res_book);
        });
    }), "GLM VQ direct decode");
}

extern "C" int ds4_gpu_gvq_moe(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out,
        ds4_gpu_tensor *gathered, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        const uint32_t *matrices, uint32_t rows, float clamp, const ds4_gpu_tensor *shared) {
    const uint64_t pairs = (uint64_t)rows * GVQ_USED, count = (uint64_t)rows * GVQ_HIDDEN;
    if (!ctx || !rows || !matrices || pairs > ctx->routes.size() || !selected ||
        pairs > selected->bytes / sizeof(int32_t) || !gvq_floats(weights, pairs) ||
        !gvq_floats(out, count) || !gvq_floats(input, count) || !gvq_floats(gathered, count) ||
        !gvq_floats(down, count) || !gvq_floats(gate, (uint64_t)rows * GVQ_FFN) ||
        !gvq_floats(up, (uint64_t)rows * GVQ_FFN) || !gvq_floats(mid, (uint64_t)rows * GVQ_FFN) ||
        (shared && !gvq_floats(shared, count))) { return 0; }
    const float *add = shared ? (const float *)shared->ptr : nullptr;
    // The numerical baseline groups actual routes on the host. Captured
    // decode needs the later GPU-selected dispatch, never a captured readback.
    if (ds4_capture_active()) { return 0; }
    if (ctx->decode && ctx->decode_route && rows <= GVQ_VERIFY_ROWS) {
        const uint32_t *table = gvq_decode_table(ctx, matrices);
        const void *book = ctx->model->source[matrices[0]].vq.book;
        const uint8_t *res_book = gvq_res_book(ctx, matrices);
        if (table && rows == 1) {
            return gvq_decode_device(ctx, (float *)out->ptr, (const float *)input->ptr,
                (const int32_t *)selected->ptr, (const float *)weights->ptr, table, book, clamp, add, res_book);
        }
        if (table) {
            return gvq_verify_device(ctx, (float *)out->ptr, (const float *)input->ptr,
                (const int32_t *)selected->ptr, (const float *)weights->ptr, table, book, rows, clamp, add, res_book);
        }
    }
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
    // Verification widths keep their separate fallback. This path changes
    // transport/reduction, never expert weight precision or routing order.
    if (ctx->decode && rows == 1 && gvq_decode_fits(ctx, matrices)) {
        return gvq_decode_moe(ctx, out, input, weights, clamp, add);
    }
    if (!ds4_gpu_tensor_write(ctx->pairs, 0, ctx->grouped.data(), pairs * sizeof(uint32_t))) { return 0; }
    // Keep single-token decode and small verification widths on their own path.
    if (ctx->mma && rows >= GVQ_MMA_TOKENS && gvq_mma_fits(ctx, matrices)) {
        return gvq_mma_moe(ctx, out, input, weights, matrices, offsets, rows, clamp, add);
    }
    if (!ds4_gpu_tensor_fill_f32(out, 0.0f, count)) { return 0; }
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
            !ds4_gpu_gvq_swiglu(mid, gate, up, n, GVQ_FFN, clamp, DS4_GVQ_ROWS_F32) ||
            !ds4_gpu_gvq_linear(ctx, down, matrices[2 * GVQ_EXPERTS + e], mid, n, DS4_GVQ_BF16)) { return 0; }
        gvq_scatter<<<blocks, GLMVQ_THREADS, 0, ds4_current_stream()>>>((float *)out->ptr,
            (const float *)down->ptr, (const float *)weights->ptr, (const uint32_t *)ctx->pairs->ptr,
            offsets[e], elements);
        if (!cuda_ok(cudaGetLastError(), "GLM VQ expert scatter")) { return 0; }
    }
    return !add || (ds4_gpu_add_tensor(out, out, shared, count) && ds4_gpu_gvq_round(out, count));
}
