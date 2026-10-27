/* MiniMax-M3.1-Flash VQ device runtime; included once by ds4_cuda.cu after
 * the GVQ adapter (whose linear kernels, context and records it uses) and
 * V4.1's FP4 helpers.
 *
 * Routed experts run GLM's GVQ1 decode helpers and FP8 tensor-core prefill
 * kernel in MiniMax's form: 128 x (6144 -> 3072 -> 6144) rows, top-4 of
 * them per token, every layer on a K8192 (13-bit) book, twenty layers with
 * the 8-bit residual stage. The activation is the clamped SwiGLU-OAI and the
 * routed outputs sum in F32 in ascending expert order with the shared
 * expert's row before one rounding (the reference's F32 index_add):
 *
 *   rows <= 8    decode and DSpark verify: device routing, persistent CTAs
 *                keep the book, one CTA group per row in one launch (an
 *                expert two rows share meets in L2); every width runs the
 *                one-row arithmetic
 *   rows >= 9    prefill: host grouping, FP8 MMA on the E4M3 codewords
 *
 * The rest is MiniMax's arithmetic (tools/minimax_ref.py): Gemma RMSNorm
 * (x * rsqrt(mean x^2 + eps) * (1 + w), rounded once), BF16 residual adds,
 * per-head q/k/index norms followed by NeoX RoPE on 64 of 128 dims, and MSA:
 * index scores pick 16 blocks of 128 tokens per KV head (the own block
 * always), attention reads NVFP4 K/V pages of those blocks only. F32 rows
 * hold BF16-rounded values; every kernel rounds its inputs where the source
 * has a BF16 tensor, since prefill projections leave F32 sums. */
#include "ds4_minimax.h"
#include "cuda/minimax_decode.cuh"
#include "cuda/minimax_q8.cuh"
#include "cuda/minimax_mma8w.cuh"

enum { MM_EXPERTS = DS4_MINIMAX_EXPERTS, MM_TABLE = 3 * MM_EXPERTS, MM_LAYERS = DS4_MINIMAX_MOE_LAYERS,
       MM_DECODE_ROWS = DS4_MINIMAX_SPLIT_ROWS, MM_MMA_ROWS = 256, MM_MMA_TOKENS = 64,
       MM_ALIGN = 256, MM_HEAD = DS4_MINIMAX_HEAD, MM_HEADS = DS4_MINIMAX_HEADS,
       MM_KV_HEADS = DS4_MINIMAX_KV_HEADS, MM_GROUP = MM_HEADS / MM_KV_HEADS, MM_ROT = DS4_MINIMAX_ROT,
       MM_BLOCK = DS4_MINIMAX_BLOCK, MM_TOPK = DS4_MINIMAX_TOPK, MM_FP4_DATA = DS4_MINIMAX_FP4_DATA,
       MM_FP4_ROW = DS4_MINIMAX_FP4_ROW, MM_PAGE = DS4_MINIMAX_KV_PAGE, MM_SPLITS = DS4_MINIMAX_SPLITS };
static_assert(MM_TOPK * MM_BLOCK == 2048, "MSA attends 16 blocks of 128 tokens");
static const float MM_EPS = 1e-6f;

static uint64_t mm_align(uint64_t bytes) { return (bytes + MM_ALIGN - 1) / MM_ALIGN * MM_ALIGN; }

/* ------------------------------------------------------------------ experts */

// Routing of up to MM_DECODE_ROWS rows on the GPU: thread t (row t / 4)
// places expert selected[t] at its rank in the row, so each row's items run
// in ascending expert order with no host sync.
__global__ void mm_decode_route(glmvq_decode_item *items, const int32_t *selected, const uint32_t *table,
        const glmvq_source *sources) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x, base = t - t % MM_USED;
    const uint32_t e = min((uint32_t)selected[t], (uint32_t)MM_EXPERTS - 1u);
    uint32_t rank = 0;
    for (uint32_t j = 0; j < MM_USED; j++) { rank += (uint32_t)selected[base + j] < (uint32_t)selected[t]; }
    items[base + rank] = {sources[table[e]].vq, sources[table[MM_EXPERTS + e]].vq,
        sources[table[2 * MM_EXPERTS + e]].vq, t};
}

// Per row (blockIdx.y): the F32 sum of weight * BF16 expert row in ascending
// expert order, plus the shared expert's row, rounded once.
__global__ void mm_decode_reduce(float *out, const float *partial, const float *weights,
        const glmvq_decode_item *items, const float *shared) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (col >= MM_HIDDEN) { return; }
    partial += (uint64_t)r * MM_USED * MM_HIDDEN;
    items += r * MM_USED;
    float sum = 0.0f;
    for (uint32_t e = 0; e < MM_USED; e++) {
        sum = glmvq_add(sum, glmvq_mul(partial[e * MM_HIDDEN + col], weights[items[e].pair]));
    }
    const uint64_t at = (uint64_t)r * MM_HIDDEN + col;
    out[at] = glmvq_round(glmvq_add(sum, glmvq_round(shared[at])));
}

// Prefill: pair rows of `down` (grouped order) per token in ascending expert
// order (`order`), as mm_decode_reduce.
template <typename ST, typename OT>
__global__ void mm_mma_reduce(OT *out, const __nv_bfloat16 *down, const float *weights, const uint32_t *pairs,
        const uint32_t *order, const ST *shared, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint64_t token = i / MM_HIDDEN, col = i % MM_HIDDEN;
    float sum = 0.0f;
    for (uint32_t k = 0; k < MM_USED; k++) {
        const uint32_t row = order[token * MM_USED + k];
        sum = glmvq_add(sum, glmvq_mul(__bfloat162float(down[(uint64_t)row * MM_HIDDEN + col]), weights[pairs[row]]));
    }
    out[i] = glmvq_cast<OT>(glmvq_round(glmvq_add(sum, glmvq_round(glmvq_value(shared[i])))));
}

// mm_mma_reduce over F32 or BF16 shared rows into F32 or BF16 rows (both
// BF16-rounded where they are read).
static void mm_reduce_launch(void *out, const __nv_bfloat16 *down, const float *weights, const uint32_t *pairs,
        const uint32_t *order, const void *shared, uint64_t count, uint32_t bf16, cudaStream_t stream) {
    const uint32_t grid = (uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
    const bool sb = bf16 & DS4_MINIMAX_BF16_SHARED, ob = bf16 & DS4_MINIMAX_BF16_OUT;
    if (sb && ob) {
        mm_mma_reduce<<<grid, GLMVQ_THREADS, 0, stream>>>((__nv_bfloat16 *)out, down, weights, pairs, order,
            (const __nv_bfloat16 *)shared, count);
    } else if (sb) {
        mm_mma_reduce<<<grid, GLMVQ_THREADS, 0, stream>>>((float *)out, down, weights, pairs, order,
            (const __nv_bfloat16 *)shared, count);
    } else if (ob) {
        mm_mma_reduce<<<grid, GLMVQ_THREADS, 0, stream>>>((__nv_bfloat16 *)out, down, weights, pairs, order,
            (const float *)shared, count);
    } else {
        mm_mma_reduce<<<grid, GLMVQ_THREADS, 0, stream>>>((float *)out, down, weights, pairs, order,
            (const float *)shared, count);
    }
}

static bool mm_residual_valid(ds4_gvq_ctx *ctx, const ds4_minimax_residual_out *out, uint32_t rows);
static cudaError_t mm_reduce_finish(ds4_gvq_ctx *ctx, void *out, const __nv_bfloat16 *down, const float *weights,
    const uint32_t *pairs, const uint32_t *order, const void *shared, uint32_t rows, uint32_t bf16,
    const ds4_minimax_residual_out *residual);

// Decode scratch, then the device tables (one 3 x 128 row per layer).
struct mm_decode_layout { uint64_t items, mid, partial, tables, bytes; };

static mm_decode_layout mm_decode_plan() {
    mm_decode_layout l = {};
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += mm_align(bytes); return here; };
    l.items = take(sizeof(glmvq_decode_item) * MM_DECODE_ROWS * MM_USED);
    l.mid = take((uint64_t)MM_DECODE_ROWS * MM_USED * MM_FFN * sizeof(__nv_bfloat16));
    l.partial = take((uint64_t)MM_DECODE_ROWS * MM_USED * MM_HIDDEN * sizeof(float));
    l.tables = take((uint64_t)MM_LAYERS * MM_TABLE * sizeof(uint32_t));
    l.bytes = at;
    return l;
}

// Prefill: E4M3 input rows and 1x128 scales, gate rows (BF16), the up
// epilogue's E4M3 SwiGLU rows and scales, down rows (BF16), worklists (the
// warp-specialized kernels' gate, up and down items and each pair's token).
struct mm_mma_layout { uint64_t x8, xs, gate, mid8, ms, down, work, order, witems, token, bytes; };

static mm_mma_layout mm_mma_plan(uint32_t rows) {
    mm_mma_layout l = {};
    const uint64_t pairs = (uint64_t)rows * MM_USED;
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += mm_align(bytes); return here; };
    l.x8 = take((uint64_t)rows * MM_HIDDEN);
    l.xs = take((uint64_t)rows * (MM_HIDDEN / GLMVQ_MMA8_K) * sizeof(float));
    l.gate = take(pairs * MM_FFN * sizeof(__nv_bfloat16));
    l.mid8 = take(pairs * MM_FFN);
    l.ms = take(pairs * (MM_FFN / GLMVQ_MMA8_K) * sizeof(float));
    l.down = take(pairs * MM_HIDDEN * sizeof(__nv_bfloat16));
    l.work = take(pairs * sizeof(glmvq_mma_item));
    l.order = take(pairs * sizeof(uint32_t));
    l.witems = take(3 * (MM_EXPERTS + pairs / MW_BT) * sizeof(mw_item));
    l.token = take(pairs * sizeof(uint32_t));
    l.bytes = at;
    return l;
}

// Each grouped pair's token row (pair index / top-k).
__global__ void mm_pair_tokens(uint32_t *token_of, const uint32_t *map, uint32_t pairs) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < pairs) { token_of[p] = map[p] / MM_USED; }
}

struct ds4_minimax_moe {
    ds4_gvq_ctx *gvq = nullptr;     // borrowed: stream, routes, grouped, pairs
    ds4_gpu_tensor *decode = nullptr, *mma = nullptr;
    uint32_t rows = 0, decode_blocks = 0;
    int mma8_blocks[2][3] = {};     // [R13, R13+8][gate, up, down]
    // DS4_MINIMAX_MMA_WS=0: prefill experts on glmvq_mma8_kernel (A/B).
    bool mmaw = true;
    bool gu16 = true;
    std::vector<uint32_t> keys;     // table slot per layer, keyed by expert 0's w1
    std::vector<glmvq_mma_item> items;
    std::vector<mw_item> witems;    // gate, up, down 64-pair items (mw_kernel)
    std::vector<uint32_t> order;
    std::vector<uint8_t> token_used;
};

extern "C" uint64_t ds4_gpu_minimax_moe_bytes(uint32_t rows) {
    return mm_decode_plan().bytes + (rows > MM_DECODE_ROWS ? mm_mma_plan(rows).bytes : 0u);
}

extern "C" void ds4_gpu_minimax_moe_free(ds4_minimax_moe *moe) {
    if (!moe) { return; }
    ds4_gpu_tensor_free(moe->decode);
    ds4_gpu_tensor_free(moe->mma);
    delete moe;
}

template <bool RES>
static cudaError_t mm_prepare(ds4_minimax_moe *moe) {
    constexpr uint32_t books = glmvq_decode_books<13, RES>();
    cudaError_t err = cudaFuncSetAttribute(mm_decode_gu<RES>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        books + MM_HIDDEN * sizeof(__nv_bfloat16));
    if constexpr (RES) {
        if (err == cudaSuccess) {
            err = cudaFuncSetAttribute(mm_decode_gu<RES, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                books + MM_HIDDEN * sizeof(__nv_bfloat16));
        }
    }
    if (err == cudaSuccess) {
        err = cudaFuncSetAttribute(mm_decode_down<RES>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            books + MM_USED * MM_FFN * sizeof(__nv_bfloat16));
    }
    int *mma = moe->mma8_blocks[RES];
    if (err == cudaSuccess) {
        err = glmvq_mma8_prepare<MM_MMA_ROWS, MM_MMA_TOKENS, 1, GLMVQ_MMA_GATE, 13, GLMVQ_CLAMP_OAI, RES>(&mma[0]);
    }
    if (err == cudaSuccess) {
        err = glmvq_mma8_prepare<MM_MMA_ROWS, MM_MMA_TOKENS, 1, GLMVQ_MMA_UP, 13, GLMVQ_CLAMP_OAI, RES>(&mma[1]);
    }
    if (err == cudaSuccess) {
        err = glmvq_mma8_prepare<MM_MMA_ROWS, MM_MMA_TOKENS, 1, GLMVQ_MMA_DOWN, 13, GLMVQ_CLAMP_OAI, RES>(&mma[2]);
    }
    return err;
}

extern "C" ds4_minimax_moe *ds4_gpu_minimax_moe_create(ds4_gvq_ctx *gvq, uint32_t rows) {
    if (!gvq || !rows) { return nullptr; }
    ds4_minimax_moe *moe = new (std::nothrow) ds4_minimax_moe;
    if (!moe) { return nullptr; }
    const char *gu16 = getenv("DS4_MINIMAX_GU16");
    if (gu16 && strcmp(gu16, "0") && strcmp(gu16, "1")) {
        fprintf(stderr, "ds4: DS4_MINIMAX_GU16 accepts 0 or 1\n");
        delete moe;
        return nullptr;
    }
    moe->gu16 = !gu16 || strcmp(gu16, "0");
    moe->gvq = gvq;
    moe->rows = rows;
    int device = 0, sms = 0;
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess) {
        moe->decode_blocks = (uint32_t)sms;
    }
    const cudaError_t err = moe->decode_blocks ? mm_prepare<false>(moe) : cudaErrorNotSupported;
    const cudaError_t err_res = err == cudaSuccess ? mm_prepare<true>(moe) : err;
    if (err_res != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: MiniMax expert kernels unavailable (%s)\n", cudaGetErrorString(err_res));
        ds4_gpu_minimax_moe_free(moe);
        return nullptr;
    }
    moe->decode = ds4_gpu_tensor_alloc(mm_decode_plan().bytes);
    moe->mma = rows > MM_DECODE_ROWS ? ds4_gpu_tensor_alloc(mm_mma_plan(rows).bytes) : nullptr;
    if (!moe->decode || (rows > MM_DECODE_ROWS && !moe->mma)) {
        ds4_gpu_minimax_moe_free(moe);
        return nullptr;
    }
    const char *ws = getenv("DS4_MINIMAX_MMA_WS");
    moe->mmaw = !(ws && !strcmp(ws, "0"));
    moe->items.reserve((uint64_t)rows * MM_USED);
    moe->order.resize((uint64_t)rows * MM_USED);
    moe->token_used.resize(rows);
    return moe;
}

// Shape, books and alignment the kernels read: every one of a layer's 384
// matrices shares its 13-bit book and, on R13+8 layers, its residual book.
static bool mm_matrix_fits(const glmvq_source &s, uint32_t projection, const void *book, const void *res_book) {
    const uint32_t rows = projection == 2 ? MM_HIDDEN : MM_FFN;
    const uint32_t cols = projection == 2 ? MM_FFN : MM_HIDDEN;
    const glmvq_matrix &m = s.vq;
    return s.storage == GLMVQ_WEIGHT_VQ && s.rows == rows && s.cols == cols && m.rows == rows && m.cols == cols &&
        m.precision == GLMVQ_E4M3 && m.main && m.plane && m.gain && book && m.book == book &&
        m.res_book == res_book && (res_book != nullptr) == (m.res != nullptr) &&
        (uintptr_t)m.main % 128 == 0 && (uintptr_t)m.plane % 128 == 0 && (!m.res || (uintptr_t)m.res % 128 == 0) &&
        (uintptr_t)book % 16 == 0 && (!res_book || (uintptr_t)res_book % 16 == 0) &&
        (uintptr_t)m.gain % sizeof(__half) == 0;
}

// The layer's device table, validated and uploaded on first use.
static const uint32_t *mm_table(ds4_minimax_moe *moe, const uint32_t *matrices) {
    const auto &sources = moe->gvq->model->source;
    const mm_decode_layout l = mm_decode_plan();
    const auto *tables = (const uint32_t *)((const uint8_t *)moe->decode->ptr + l.tables);
    for (size_t i = 0; i < moe->keys.size(); i++) {
        if (moe->keys[i] == matrices[0]) { return tables + i * MM_TABLE; }
    }
    if (moe->keys.size() == MM_LAYERS || matrices[0] >= sources.size()) { return nullptr; }
    const void *book = sources[matrices[0]].vq.book;
    const void *res_book = sources[matrices[0]].vq.res_book;
    for (uint32_t i = 0; i < MM_TABLE; i++) {
        if (matrices[i] >= sources.size() || !mm_matrix_fits(sources[matrices[i]], i / MM_EXPERTS, book, res_book)) {
            fprintf(stderr, "ds4: MiniMax expert matrix %u misses the GVQ1 layout\n", i);
            return nullptr;
        }
    }
    const size_t slot = moe->keys.size();
    if (!ds4_gpu_tensor_write(moe->decode, l.tables + slot * MM_TABLE * sizeof(uint32_t), matrices,
            MM_TABLE * sizeof(uint32_t))) { return nullptr; }
    moe->keys.push_back(matrices[0]);
    return tables + slot * MM_TABLE;
}

template <bool RES>
static cudaError_t mm_decode_run(ds4_minimax_moe *moe, float *out, const float *input, const int32_t *selected,
        const float *weights, const float *shared, const uint32_t *table, const void *book, const void *res_book,
        uint32_t rows) {
    const mm_decode_layout l = mm_decode_plan();
    uint8_t *base = (uint8_t *)moe->decode->ptr;
    auto *items = (glmvq_decode_item *)(base + l.items);
    auto *mid = (__nv_bfloat16 *)(base + l.mid);
    auto *partial = (float *)(base + l.partial);
    const cudaStream_t stream = ds4_current_stream();
    const glmvq_launch launch = moe->gvq->launch;
    constexpr uint32_t books = glmvq_decode_books<13, RES>();
    // One CTA group per row (mm_decode_gu).
    const uint32_t grid = rows * (moe->decode_blocks / rows);
    glmvq_launch_kernel(launch, mm_decode_route, 1, rows * MM_USED, 0, stream, items, selected, table,
        (const glmvq_source *)moe->gvq->model->device->ptr);
    if (RES && rows == 1 && moe->gu16) {
        glmvq_launch_kernel(launch, mm_decode_gu<RES, 16>, grid, 16 * 32,
            books + MM_HIDDEN * sizeof(__nv_bfloat16), stream, (const glmvq_decode_item *)items, (const uint4 *)book,
            (const uint4 *)res_book, input, mid, rows);
    } else {
        glmvq_launch_kernel(launch, mm_decode_gu<RES>, grid, MM_WARPS * 32,
            books + MM_HIDDEN * sizeof(__nv_bfloat16), stream, (const glmvq_decode_item *)items, (const uint4 *)book,
            (const uint4 *)res_book, input, mid, rows);
    }
    glmvq_launch_kernel(launch, mm_decode_down<RES>, grid, MM_WARPS * 32,
        books + MM_USED * MM_FFN * sizeof(__nv_bfloat16), stream, (const glmvq_decode_item *)items,
        (const uint4 *)book, (const uint4 *)res_book, (const __nv_bfloat16 *)mid, partial, rows);
    glmvq_launch_kernel(launch, mm_decode_reduce, dim3(MM_HIDDEN / GLMVQ_THREADS, rows), GLMVQ_THREADS, 0, stream,
        out, (const float *)partial, weights, (const glmvq_decode_item *)items, shared);
    return cudaGetLastError();
}

// Host grouping by expert (ascending), as the GLM prefill path builds it.
static bool mm_group(ds4_minimax_moe *moe, const ds4_gpu_tensor *selected, uint32_t rows, uint32_t *offsets) {
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * MM_USED;
    if (pairs > ctx->routes.size() ||
        !ds4_gpu_tensor_read(selected, 0, ctx->routes.data(), pairs * sizeof(int32_t))) { return false; }
    std::vector<uint32_t> counts(MM_EXPERTS, 0), cursor(MM_EXPERTS);
    for (uint64_t i = 0; i < pairs; i++) {
        const int32_t expert = ctx->routes[i];
        if (expert < 0 || expert >= MM_EXPERTS) { return false; }
        for (uint64_t j = i - i % MM_USED; j < i; j++) {
            if (ctx->routes[j] == expert) { return false; }
        }
        counts[expert]++;
    }
    offsets[0] = 0;
    for (uint32_t e = 0; e < MM_EXPERTS; e++) {
        offsets[e + 1] = offsets[e] + counts[e];
        cursor[e] = offsets[e];
    }
    for (uint64_t i = 0; i < pairs; i++) { ctx->grouped[cursor[ctx->routes[i]]++] = (uint32_t)i; }
    return ds4_gpu_tensor_write(ctx->pairs, 0, ctx->grouped.data(), pairs * sizeof(uint32_t)) != 0;
}

// Prefill experts on the warp-specialized kernels: 64-pair items per
// projection, tokens quantized once (E4M3, 1x128 scales, ldmatrix k order).
template <bool RES>
static int mm_mmaw_run(ds4_minimax_moe *moe, void *out, const float *input, const float *weights,
        const void *shared, const uint32_t *matrices, const uint32_t *offsets, uint32_t rows, uint32_t bf16,
        const ds4_minimax_residual_out *residual) {
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * MM_USED, count = (uint64_t)rows * MM_HIDDEN;
    const mm_mma_layout l = mm_mma_plan(rows);
    uint8_t *base = (uint8_t *)moe->mma->ptr;
    auto *x8 = base + l.x8, *mid8 = base + l.mid8;
    auto *xs = (float *)(base + l.xs), *ms = (float *)(base + l.ms);
    auto *gate = (__nv_bfloat16 *)(base + l.gate), *down = (__nv_bfloat16 *)(base + l.down);
    auto *order = (uint32_t *)(base + l.order);
    auto *witems = (mw_item *)(base + l.witems);
    auto *token = (uint32_t *)(base + l.token);
    moe->witems.clear();
    for (uint32_t m = 0; m < 3; m++) {
        for (uint32_t e = 0; e < MM_EXPERTS; e++) {
            const glmvq_matrix &vq = ctx->model->source[matrices[m * MM_EXPERTS + e]].vq;
            for (uint32_t start = offsets[e]; start < offsets[e + 1]; start += MW_BT) {
                moe->witems.push_back({start, min((uint32_t)MW_BT, offsets[e + 1] - start), vq.gain, vq.main, vq.plane,
                    vq.res, 0});
            }
        }
    }
    std::fill(moe->token_used.begin(), moe->token_used.begin() + rows, 0);
    for (uint32_t i = 0; i < pairs; i++) {
        const uint32_t t = ctx->grouped[i] / MM_USED;
        moe->order[t * MM_USED + moe->token_used[t]++] = i;
    }
    const cudaStream_t stream = ds4_current_stream();
    if (!cuda_ok(cudaMemcpyAsync(witems, moe->witems.data(), moe->witems.size() * sizeof(mw_item),
            cudaMemcpyHostToDevice, stream), "MiniMax warp-specialized items") ||
        !cuda_ok(cudaMemcpyAsync(order, moe->order.data(), pairs * sizeof(*order), cudaMemcpyHostToDevice, stream),
            "MiniMax MMA order")) { return 0; }
    const auto *book = (const uint8_t *)ctx->model->source[matrices[0]].vq.book;
    const auto *res_book = (const uint8_t *)ctx->model->source[matrices[0]].vq.res_book;
    const auto *grouped = (const uint32_t *)ctx->pairs->ptr;
    const uint32_t n = (uint32_t)moe->witems.size() / 3, sms = moe->decode_blocks;
    const uint64_t xb = count / GLMVQ_MMA8_K;
    mw_quant<<<(uint32_t)((xb * 32 + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0, stream>>>(x8, xs, input, xb);
    mm_pair_tokens<<<(uint32_t)((pairs + 255) / 256), 256, 0, stream>>>(token, grouped, (uint32_t)pairs);
    // Up's BF16 rows borrow down's buffer (twice their size) until
    // mw_swiglu_quant has read them.
    cudaError_t err = mw_launch<GLMVQ_MMA_GATE, RES>(witems, n, book, res_book, x8, xs, token, gate, nullptr,
        MM_FFN, MM_HIDDEN, sms, stream);
    if (err == cudaSuccess) {
        err = mw_launch<GLMVQ_MMA_GATE, RES>(witems + n, n, book, res_book, x8, xs, token, down, nullptr,
            MM_FFN, MM_HIDDEN, sms, stream);
    }
    if (err == cudaSuccess) {
        const uint64_t warps = pairs * (MM_FFN / GLMVQ_MMA8_K);
        mw_swiglu_quant<<<(uint32_t)((warps * 32 + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0, stream>>>(
            mid8, ms, gate, down, MM_FFN, (uint32_t)pairs, MM_LIMIT);
        err = mw_launch<GLMVQ_MMA_DOWN, RES>(witems + 2 * n, n, book, res_book, mid8, ms, token, nullptr, down,
            MM_HIDDEN, MM_FFN, sms, stream);
    }
    if (!cuda_ok(err, "MiniMax warp-specialized experts")) { return 0; }
    return cuda_ok(mm_reduce_finish(ctx, out, down, weights, grouped, order, shared, rows, bf16, residual),
        "MiniMax FP8 warp-specialized MMA");
}

template <bool RES>
static int mm_mma_run(ds4_minimax_moe *moe, void *out, const float *input, const float *weights,
        const void *shared, const uint32_t *matrices, const uint32_t *offsets, uint32_t rows, uint32_t bf16,
        const ds4_minimax_residual_out *residual) {
    if (moe->mmaw) { return mm_mmaw_run<RES>(moe, out, input, weights, shared, matrices, offsets, rows, bf16, residual); }
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * MM_USED, count = (uint64_t)rows * MM_HIDDEN;
    const mm_mma_layout l = mm_mma_plan(rows);
    uint8_t *base = (uint8_t *)moe->mma->ptr;
    auto *x8 = base + l.x8, *mid8 = base + l.mid8;
    auto *xs = (float *)(base + l.xs), *ms = (float *)(base + l.ms);
    auto *gate = (__nv_bfloat16 *)(base + l.gate), *down = (__nv_bfloat16 *)(base + l.down);
    auto *work = (glmvq_mma_item *)(base + l.work);
    auto *order = (uint32_t *)(base + l.order);
    moe->items.clear();
    for (uint32_t e = 0; e < MM_EXPERTS; e++) {
        for (uint32_t start = offsets[e]; start < offsets[e + 1]; start += MM_MMA_TOKENS) {
            moe->items.push_back({{matrices[e], matrices[MM_EXPERTS + e], matrices[2 * MM_EXPERTS + e]},
                start, min((uint32_t)MM_MMA_TOKENS, offsets[e + 1] - start)});
        }
    }
    std::fill(moe->token_used.begin(), moe->token_used.begin() + rows, 0);
    for (uint32_t i = 0; i < pairs; i++) {
        const uint32_t token = ctx->grouped[i] / MM_USED;
        moe->order[token * MM_USED + moe->token_used[token]++] = i;
    }
    const cudaStream_t stream = ds4_current_stream();
    if (!cuda_ok(cudaMemcpyAsync(work, moe->items.data(), moe->items.size() * sizeof(*work),
            cudaMemcpyHostToDevice, stream), "MiniMax MMA work") ||
        !cuda_ok(cudaMemcpyAsync(order, moe->order.data(), pairs * sizeof(*order), cudaMemcpyHostToDevice, stream),
            "MiniMax MMA order")) { return 0; }
    const auto *source = (const glmvq_source *)ctx->model->device->ptr;
    const auto *book = (const uint8_t *)ctx->model->source[matrices[0]].vq.book;
    const auto *res_book = (const uint8_t *)ctx->model->source[matrices[0]].vq.res_book;
    const auto *grouped = (const uint32_t *)ctx->pairs->ptr;
    const uint32_t nitems = (uint32_t)moe->items.size();
    const int *blocks = moe->mma8_blocks[RES];
    // Tokens quantize once (E4M3, 1x128 scales) for their four experts.
    const uint64_t xb = count / GLMVQ_MMA8_K;
    glmvq_mma8_quant<float><<<(uint32_t)((xb * 32 + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0,
        stream>>>(x8, xs, input, xb);
    auto launch = [&](auto mode, const uint8_t *in, const float *scale, uint32_t rows_out, uint32_t cols_in,
            uint8_t *q8, float *q8_scale) {
        constexpr glmvq_mma_stage MODE = decltype(mode)::value;
        constexpr uint32_t shared8 = glmvq_mma8_bytes<MM_MMA_ROWS, MM_MMA_TOKENS, 13, RES>();
        constexpr uint32_t threads8 = glmvq_mma8_threads<MM_MMA_ROWS, 1>();
        glmvq_mma8_kernel<MM_MMA_ROWS, MM_MMA_TOKENS, 1, MODE, 13, GLMVQ_CLAMP_OAI, RES><<<
            min(nitems * (rows_out / MM_MMA_ROWS), (uint32_t)blocks[MODE]), threads8, shared8, stream>>>(
            source, work, nitems, book, in, scale, grouped, MM_USED, gate, nullptr, down, rows_out, cols_in,
            MM_LIMIT, q8, q8_scale, res_book);
    };
    launch(std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_GATE>{}, x8, xs, MM_FFN, MM_HIDDEN, nullptr, nullptr);
    // The up epilogue writes down's E4M3 rows and scales directly.
    launch(std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_UP>{}, x8, xs, MM_FFN, MM_HIDDEN, mid8, ms);
    launch(std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_DOWN>{}, mid8, ms, MM_HIDDEN, MM_FFN, nullptr, nullptr);
    return cuda_ok(mm_reduce_finish(ctx, out, down, weights, grouped, order, shared, rows, bf16, residual),
        "MiniMax FP8 grouped MMA");
}

extern "C" int ds4_gpu_minimax_moe(ds4_minimax_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *shared,
        const uint32_t *matrices, uint32_t rows, uint32_t bf16, const ds4_minimax_residual_out *residual) {
    if (!moe || !rows || rows > moe->rows || !matrices || !selected || !gvq_stream(moe->gvq) ||
        (bf16 && rows <= MM_DECODE_ROWS) ||
        (residual && (!(bf16 & DS4_MINIMAX_BF16_OUT) || !mm_residual_valid(moe->gvq, residual, rows)))) { return 0; }
    const uint64_t pairs = (uint64_t)rows * MM_USED, count = (uint64_t)rows * MM_HIDDEN;
    if (pairs > selected->bytes / sizeof(int32_t) || !gvq_floats(weights, pairs) || !gvq_floats(out, count) ||
        !gvq_floats(input, count) || !gvq_floats(shared, count) || ds4_capture_active()) { return 0; }
    const uint32_t *table = mm_table(moe, matrices);
    if (!table) { return 0; }
    const glmvq_matrix &first = moe->gvq->model->source[matrices[0]].vq;
    const bool res = first.res_book != nullptr;
    if (rows <= MM_DECODE_ROWS) {
        // Every row runs the one-row arithmetic in one launch per stage:
        // verify rows equal decode rows.
        const auto *sel = (const int32_t *)selected->ptr;
        const auto *w = (const float *)weights->ptr, *sh = (const float *)shared->ptr;
        const auto *x = (const float *)input->ptr;
        auto *y = (float *)out->ptr;
        const cudaError_t err = res
            ? mm_decode_run<true>(moe, y, x, sel, w, sh, table, first.book, first.res_book, rows)
            : mm_decode_run<false>(moe, y, x, sel, w, sh, table, first.book, nullptr, rows);
        return cuda_ok(err, "MiniMax routed decode");
    }
    std::vector<uint32_t> offsets(MM_EXPERTS + 1);
    if (!moe->mma || !mm_group(moe, selected, rows, offsets.data())) { return 0; }
    const auto *x = (const float *)input->ptr, *w = (const float *)weights->ptr;
    return res ? mm_mma_run<true>(moe, out->ptr, x, w, shared->ptr, matrices, offsets.data(), rows, bf16, residual)
               : mm_mma_run<false>(moe, out->ptr, x, w, shared->ptr, matrices, offsets.data(), rows, bf16, residual);
}

/* ------------------------------------------------------------------- router */

/* Sigmoid router without groups (models/minimax_m3): one warp per row, lane
 * l holds experts 4l..4l+3. s = sigmoid(logit) in double, rounded once;
 * choice c = s + bias; the top-4 by c, ties to the lower index as torch.topk
 * does on these sizes; weights are the unbiased s over their F32 sum, times
 * routed_scaling_factor 2.0 (exact). */
// Row `row`'s pick by one warp (lane l holds experts 4l..4l+3); logits load
// through L2 because the fused caller reads other CTAs' stores.
__device__ __forceinline__ void mm_route_row(int32_t *ids, float *weights, const float *logits, const float *bias,
        uint32_t row, uint32_t lane) {
    float s[4], c[4];
    for (uint32_t j = 0; j < 4; j++) {
        const uint32_t e = lane * 4 + j;
        s[j] = (float)(1.0 / (1.0 + exp(-(double)__ldcg(logits + (uint64_t)row * MM_EXPERTS + e))));
        c[j] = glmvq_add(s[j], bias[e]);
    }
    float sum = 0.0f;
    uint32_t chosen[MM_USED];
    float chosen_s[MM_USED];
    for (uint32_t k = 0; k < MM_USED; k++) {
        float best = -INFINITY;
        uint32_t index = UINT32_MAX;
        for (uint32_t j = 0; j < 4; j++) {
            if (c[j] > best || (index == UINT32_MAX && c[j] == best)) { best = c[j]; index = lane * 4 + j; }
        }
        for (uint32_t stride = 16; stride; stride >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, stride);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, index, stride);
            if (ob > best || (ob == best && oi < index)) { best = ob; index = oi; }
        }
        const uint32_t owner = index / 4, slot = index % 4;
        const float picked = __shfl_sync(0xffffffffu, s[slot], owner);
        if (lane == owner) { c[slot] = -INFINITY; }
        chosen[k] = index;
        chosen_s[k] = picked;
        sum = glmvq_add(sum, picked);
    }
    if (lane) { return; }
    for (uint32_t k = 0; k < MM_USED; k++) {
        ids[(uint64_t)row * MM_USED + k] = (int32_t)chosen[k];
        weights[(uint64_t)row * MM_USED + k] = glmvq_mul(__fdiv_rn(chosen_s[k], sum), 2.0f);
    }
}

__global__ void mm_router_kernel(int32_t *ids, float *weights, const float *logits, const float *bias,
        uint32_t rows) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) { return; }
    mm_route_row(ids, weights, logits, bias, row, lane);
}

/* Router logits of up to MM_ROUTER_ROWS rows in F32 (decode and verify):
 * warp e streams expert e's 6,144 weights once and keeps one dot per row,
 * lane l summing float4 steps l, l + 32, ... in order, then a fixed xor
 * tree. A row's logits do not depend on how many rows share the launch,
 * so verify rows route exactly as decode rows. cuBLAS took 62-116 us per
 * layer at two or three rows (gemmSN) for a 3 MiB matrix. */
enum { MM_ROUTER_ROWS = DS4_MINIMAX_SPLIT_ROWS, MM_ROUTER_WARPS = 2 };

/* Decode and verify kernels take their row count (1..MAX) as a template
 * argument: with a runtime count their unrolled per-row loops issued every
 * predicated row, eight rows' instructions for a decode row. `launch` gets
 * std::integral_constant<uint32_t, R> for the R == rows instance. */
/* Decode and verify widths launch MiniMax's kernels with programmatic
 * dependent launch (glmvq_pdl_open first, glmvq_pdl_wait before reading a
 * predecessor's output or writing): a chain of plain launches between the
 * Q4_K and expert kernels kept those from staging their weights while the
 * short kernels ran, and each boundary waited for a full drain.
 * DS4_GLMVQ_PDL=0 turns it off with the GVQ kernels'. */
static glmvq_launch mm_mode(uint32_t rows) { return rows <= DS4_MINIMAX_SPLIT_ROWS ? g_gvq_pdl : GLMVQ_LAUNCH_PLAIN; }

template <uint32_t MAX, typename F>
static void mm_rows(uint32_t rows, F launch) {
    if constexpr (MAX > 1) {
        if (rows < MAX) { mm_rows<MAX - 1>(rows, launch); return; }
    }
    launch(std::integral_constant<uint32_t, MAX>{});
}

template <uint32_t ROWS>
__global__ void __launch_bounds__(MM_ROUTER_WARPS * 32) mm_router_gemv(float *logits, const float *w,
        const float *x) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t lane = threadIdx.x & 31, e = blockIdx.x * MM_ROUTER_WARPS + threadIdx.x / 32;
    const float4 *we = (const float4 *)(w + (uint64_t)e * MM_HIDDEN);
    float acc[ROWS];
    #pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) { acc[r] = 0.0f; }
    // Eight steps' loads in flight per lane (the sums keep their order): one
    // load at a time held the 3 MiB matrix at ~120 GB/s. Two warps per CTA
    // spread the 128 experts over every SM (four left 16 SMs idle).
    #pragma unroll 8
    for (uint32_t i = lane; i < MM_HIDDEN / 4; i += 32) {
        const float4 a = __ldg(we + i);
        #pragma unroll
        for (uint32_t r = 0; r < ROWS; r++) {
            const float4 v = ((const float4 *)(x + (uint64_t)r * MM_HIDDEN))[i];
            acc[r] = fmaf(a.x, v.x, acc[r]);
            acc[r] = fmaf(a.y, v.y, acc[r]);
            acc[r] = fmaf(a.z, v.z, acc[r]);
            acc[r] = fmaf(a.w, v.w, acc[r]);
        }
    }
    #pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) {
        float v = acc[r];
        for (uint32_t stride = 16; stride; stride >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, stride); }
        if (!lane) { logits[(uint64_t)r * MM_EXPERTS + e] = v; }
    }
}

/* The indexer's two Q8_0 projections (index_q, index_k) of up to
 * MM_ROUTER_ROWS rows in one launch, CTAs [0, first1) on the first matrix:
 * warp o streams output o's 34-byte blocks once, lane l takes element l of
 * every block (d * q is exact in F32) against the F32 input, in block
 * order, then a fixed xor tree. A row's result does not depend on the
 * launch's other rows; the mmvq path picked its column layout by width and
 * quantized the input to Q8_1. The CTA's two output rows (contiguous) first
 * land in shared memory by 16-byte loads: a warp reading one block per step
 * with byte loads held this at ~17% of the DRAM rate, 6 ms of a decode
 * token. */
enum { MM_Q8_WARPS = 2, MM_Q8_BLOCK = 32, MM_Q8_BYTES = 34, MM_TYPE_Q8_0 = 8 };   // ggml's Q8_0 id

struct mm_q8_two {
    const uint8_t *w0, *w1;
    float *out0, *out1;
    uint32_t outs0, outs1, first1;
};

template <uint32_t ROWS>
__global__ void __launch_bounds__(MM_Q8_WARPS * 32) mm_q8_gemv(mm_q8_two m, const float *x, uint32_t cols,
        bool staged) {
    extern __shared__ __align__(16) uint8_t mm_q8_rows[];
    // Selected by compares: a dynamic index would copy the parameters to local memory.
    const bool second = blockIdx.x >= m.first1;
    const uint8_t *weight = second ? m.w1 : m.w0;
    float *out = second ? m.out1 : m.out0;
    const uint32_t outs = second ? m.outs1 : m.outs0;
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const uint32_t o0 = (second ? blockIdx.x - m.first1 : blockIdx.x) * MM_Q8_WARPS;
    const uint32_t blocks = cols / MM_Q8_BLOCK, row_bytes = blocks * MM_Q8_BYTES;
    glmvq_pdl_open();
    // The weights are constant: they stage while the predecessor drains.
    if (staged) {
        const uint32_t pieces = min((uint32_t)MM_Q8_WARPS, outs - o0) * row_bytes / 16;
        const uint4 *src = (const uint4 *)(weight + (uint64_t)o0 * row_bytes);
        for (uint32_t i = threadIdx.x; i < pieces; i += MM_Q8_WARPS * 32) { ((uint4 *)mm_q8_rows)[i] = src[i]; }
        __syncthreads();
    }
    glmvq_pdl_wait();
    const uint32_t o = o0 + warp;
    if (o >= outs) { return; }
    const uint8_t *w = staged ? mm_q8_rows + warp * row_bytes : weight + (uint64_t)o * row_bytes;
    float acc[ROWS];
    #pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) { acc[r] = 0.0f; }
    #pragma unroll 4
    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *block = w + b * MM_Q8_BYTES;
        const float value = __half2float(__ushort_as_half((uint16_t)(block[0] | block[1] << 8))) *
            (float)(int8_t)block[2 + lane];
        #pragma unroll
        for (uint32_t r = 0; r < ROWS; r++) { acc[r] = fmaf(value, x[(uint64_t)r * cols + b * MM_Q8_BLOCK + lane], acc[r]); }
    }
    #pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) {
        float v = acc[r];
        for (uint32_t stride = 16; stride; stride >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, stride); }
        if (!lane) { out[(uint64_t)r * outs + o] = v; }
    }
}

/* The indexer's two Q8_0 projections of one input: mm_q8_gemv at decode and
 * verify widths, MMQ for prefill rows (the input quantized to Q8_1 once, each
 * output as ds4_mmq_q8_0_dense computes it). */
extern "C" int ds4_gpu_minimax_q8_pair(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out0, ds4_gpu_tensor *out1, uint32_t w0,
        uint32_t w1, const ds4_gpu_tensor *input, uint32_t cols, uint32_t outs0, uint32_t outs1, uint32_t rows,
        ds4_minimax_norm prepared) {
    const auto *a = (const uint8_t *)gvq_weight(ctx, w0, MM_TYPE_Q8_0, (uint64_t)outs0 * (cols / MM_Q8_BLOCK) * MM_Q8_BYTES / 2);
    const auto *b = (const uint8_t *)gvq_weight(ctx, w1, MM_TYPE_Q8_0, (uint64_t)outs1 * (cols / MM_Q8_BLOCK) * MM_Q8_BYTES / 2);
    if ((prepared != DS4_MINIMAX_NORM_F32 && prepared != DS4_MINIMAX_NORM_Q8) ||
        !a || !b || !rows || !cols || cols % MM_Q8_BLOCK || !outs0 || !outs1 ||
        (prepared == DS4_MINIMAX_NORM_F32 && !gvq_floats(input, (uint64_t)rows * cols)) ||
        !gvq_floats(out0, (uint64_t)rows * outs0) ||
        !gvq_floats(out1, (uint64_t)rows * outs1) || !gvq_stream(ctx)) {
        return 0;
    }
    if (prepared == DS4_MINIMAX_NORM_Q8) {
        if (rows <= MM_ROUTER_ROWS || cols != MM_HIDDEN || ctx->scratch->bytes < mm_norm_q8_bytes(rows)) { return 0; }
        const cudaStream_t stream = ds4_mmq_stream_for_call();
        return ds4_mmq_q8_0_dense_preq(a, ctx->scratch->ptr, ctx->scratch->bytes, (float *)out0->ptr,
                (int)outs0, (int)rows, (int)cols, stream) == 0 &&
            ds4_mmq_q8_0_dense_preq(b, ctx->scratch->ptr, ctx->scratch->bytes, (float *)out1->ptr,
                (int)outs1, (int)rows, (int)cols, stream) == 0 &&
            cuda_ok(cudaGetLastError(), "MiniMax prepared indexer projections");
    }
    if (rows > MM_ROUTER_ROWS) {
        return ds4_mmq_q8_0_dense_pair(a, b, (const float *)input->ptr, (float *)out0->ptr, (float *)out1->ptr,
            (int)outs0, (int)outs1, (int)rows, (int)cols, ds4_mmq_stream_for_call()) == 0 &&
            cuda_ok(cudaGetLastError(), "MiniMax indexer projections");
    }
    // The staged copy takes whole 16-byte pieces (every MiniMax indexer row: 6,528 bytes).
    const uint32_t row_bytes = cols / MM_Q8_BLOCK * MM_Q8_BYTES;
    const bool staged = row_bytes % 16 == 0 && (uintptr_t)a % 16 == 0 && (uintptr_t)b % 16 == 0 &&
        MM_Q8_WARPS * row_bytes <= 48 * 1024;
    const uint32_t first1 = (outs0 + MM_Q8_WARPS - 1) / MM_Q8_WARPS, grid = first1 + (outs1 + MM_Q8_WARPS - 1) / MM_Q8_WARPS;
    const mm_q8_two m = {a, b, (float *)out0->ptr, (float *)out1->ptr, outs0, outs1, first1};
    mm_rows<MM_ROUTER_ROWS>(rows, [&](auto r) {
        glmvq_launch_kernel(mm_mode(rows), mm_q8_gemv<decltype(r)::value>, grid, MM_Q8_WARPS * 32,
            staged ? MM_Q8_WARPS * row_bytes : 0, ds4_current_stream(), m, (const float *)input->ptr, cols, staged);
    });
    return cuda_ok(cudaGetLastError(), "MiniMax indexer projections");
}

/* One CTA of glmvq_q4_stream_many<NT> (split 2, four rows per CTA) over
 * `many`'s groups, bit for bit: the joint decode launches below put other
 * work in the CTAs after its groups. */
template <uint32_t NT>
__device__ __forceinline__ void mm_q4_group(const glmvq_q4_many &many, const float *input, uint32_t cols,
        uint32_t tokens, uint4 *staged, float *partial, uint32_t block) {
    // Selected by compares, not an index: a dynamic index would copy the
    // parameter block to local memory in every thread.
    const uint8_t *weight = many.weight[0];
    float *out = many.out[0];
    uint32_t rows = many.rows[0], first = 0;
    #pragma unroll
    for (uint32_t c = 1; c < GLMVQ_Q4_MANY; c++) {
        if (block >= many.first[c]) { weight = many.weight[c]; out = many.out[c]; rows = many.rows[c]; first = many.first[c]; }
    }
    constexpr uint32_t split = 2, rpb = GLMVQ_Q4_WARPS / split;
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5, local = warp / split, part = warp % split;
    const uint32_t blocks = cols / GLMVQ_Q4_BLOCK, group = block - first;
    const uint32_t row16 = blocks * (GLMVQ_Q4_BYTES / sizeof(uint4));
    glmvq_q4_copy<GLMVQ_Q4_STAGE>(staged, (const uint4 *)weight + (uint64_t)group * rpb * row16,
        min(rpb, rows - group * rpb) * row16);
    glmvq_pdl_wait();
    __syncthreads();
    const uint32_t row = group * rpb + local;
    float sum[NT] = {};
    if (row < rows) {
        const uint8_t *w = (const uint8_t *)(staged + local * row16);
        for (uint32_t b = part; b < blocks; b += split) {
            glmvq_q4_acc<NT>(w + b * GLMVQ_Q4_BYTES, input, cols, b * GLMVQ_Q4_BLOCK, tokens, sum);
        }
    }
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        for (int stride = 16; stride; stride >>= 1) { sum[t] += __shfl_xor_sync(0xffffffffu, sum[t], stride); }
        if (!lane) { partial[warp * NT + t] = sum[t]; }
    }
    __syncthreads();
    if (row < rows && !lane && !part) {
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            if (t >= tokens) { continue; }
            float value = 0;
            for (uint32_t p = 0; p < split; p++) { value += partial[(warp + p) * NT + t]; }
            out[(uint64_t)t * rows + row] = __bfloat162float(__float2bfloat16_rn(value));
        }
    }
}

// The groups of the Q4_K matrices `weights` (one column count) in `many`;
// 0 when a record is not Q4_K or an output is short.
static uint32_t mm_q4_many(ds4_gvq_ctx *ctx, glmvq_q4_many *many, ds4_gpu_tensor *const *outs,
        const uint32_t *weights, uint32_t count, uint32_t rows, uint32_t *cols) {
    uint32_t groups = 0;
    *many = {};
    for (uint32_t c = 0; c < GLMVQ_Q4_MANY; c++) {
        many->first[c] = groups;
        if (c >= count) { continue; }
        if (weights[c] >= ctx->model->source.size()) { return 0; }
        const glmvq_source &s = ctx->model->source[weights[c]];
        if (ctx->model->records[weights[c]].type != GVQ_TYPE_Q4 || (c && s.cols != *cols) ||
            (uintptr_t)s.data % 16 || s.cols % GLMVQ_Q4_BLOCK || !gvq_floats(outs[c], (uint64_t)rows * s.rows)) {
            return 0;
        }
        *cols = s.cols;
        many->weight[c] = (const uint8_t *)s.data;
        many->out[c] = (float *)outs[c]->ptr;
        many->rows[c] = s.rows;
        groups += (s.rows + 3) / 4;
    }
    many->first[GLMVQ_Q4_MANY] = groups;
    return groups;
}

/* Decode and verify widths: q/k/v (Q4_K) and the indexer's index_q/index_k
 * (Q8_0) of one input in one launch, CTAs [0, q4_groups) on q/k/v as
 * glmvq_q4_stream_many<NT> (split 2) runs them, the rest as mm_q8_gemv<ROWS>
 * (two outputs per CTA, its other warps only stage). Every output equals
 * the two launches' bit for bit. Apart, the indexer's CTAs waited for every
 * q/k/v CTA before computing while holding their slots, which kept the next
 * weights from streaming across the boundary (~14 us per layer). */
template <uint32_t NT, uint32_t ROWS>
__global__ void __launch_bounds__(GLMVQ_Q4_WARPS * 32, 4) mm_proj_kernel(glmvq_q4_many many, mm_q8_two m,
        uint32_t q4_groups, const float *input, uint32_t cols, uint32_t tokens, bool staged) {
    extern __shared__ uint4 mm_proj_smem[];
    __shared__ float partial[GLMVQ_Q4_WARPS * NT];
    glmvq_pdl_open();
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (blockIdx.x < q4_groups) {
        mm_q4_group<NT>(many, input, cols, tokens, mm_proj_smem, partial, blockIdx.x);
        return;
    }
    const uint32_t bx = blockIdx.x - q4_groups;
    const bool second = bx >= m.first1;
    const uint8_t *weight = second ? m.w1 : m.w0;
    float *out = second ? m.out1 : m.out0;
    const uint32_t outs = second ? m.outs1 : m.outs0;
    const uint32_t o0 = (second ? bx - m.first1 : bx) * MM_Q8_WARPS;
    const uint32_t blocks = cols / MM_Q8_BLOCK, row_bytes = blocks * MM_Q8_BYTES;
    if (staged) {
        const uint32_t pieces = min((uint32_t)MM_Q8_WARPS, outs - o0) * row_bytes / 16;
        const uint4 *src = (const uint4 *)(weight + (uint64_t)o0 * row_bytes);
        for (uint32_t i = threadIdx.x; i < pieces; i += blockDim.x) { mm_proj_smem[i] = src[i]; }
        __syncthreads();
    }
    glmvq_pdl_wait();
    const uint32_t o = o0 + warp;
    if (warp >= MM_Q8_WARPS || o >= outs) { return; }
    const uint8_t *w = staged ? (const uint8_t *)mm_proj_smem + warp * row_bytes : weight + (uint64_t)o * row_bytes;
    float acc[ROWS];
    #pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) { acc[r] = 0.0f; }
    #pragma unroll 4
    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *block = w + b * MM_Q8_BYTES;
        const float value = __half2float(__ushort_as_half((uint16_t)(block[0] | block[1] << 8))) *
            (float)(int8_t)block[2 + lane];
        #pragma unroll
        for (uint32_t r = 0; r < ROWS; r++) { acc[r] = fmaf(value, input[(uint64_t)r * cols + b * MM_Q8_BLOCK + lane], acc[r]); }
    }
    #pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) {
        float v = acc[r];
        for (uint32_t stride = 16; stride; stride >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, stride); }
        if (!lane) { out[(uint64_t)r * outs + o] = v; }
    }
}

/* q/k/v (`weights`, Q4_K) and index_q/index_k (Q8_0) of `input` at decode
 * and verify widths in one launch (mm_proj_kernel); 0 when a matrix is not
 * of those types, for the caller's separate launches. */
extern "C" int ds4_gpu_minimax_proj(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs, const uint32_t *weights,
        ds4_gpu_tensor *iq, ds4_gpu_tensor *ik, uint32_t wiq, uint32_t wik, uint32_t outs_q, uint32_t outs_k,
        const ds4_gpu_tensor *input, uint32_t rows) {
    if (!ctx || !rows || rows > MM_ROUTER_ROWS || !ctx->direct_q4 || !input || (uintptr_t)input->ptr % 16 ||
        !gvq_stream(ctx)) { return 0; }
    glmvq_q4_many many;
    uint32_t cols = 0;
    const uint32_t groups = mm_q4_many(ctx, &many, outs, weights, 3u, rows, &cols);
    if (!groups) { return 0; }
    const uint32_t row_bytes = cols / MM_Q8_BLOCK * MM_Q8_BYTES;
    const auto *a = (const uint8_t *)gvq_weight(ctx, wiq, MM_TYPE_Q8_0, (uint64_t)outs_q * row_bytes / 2);
    const auto *b = (const uint8_t *)gvq_weight(ctx, wik, MM_TYPE_Q8_0, (uint64_t)outs_k * row_bytes / 2);
    if (!a || !b || cols % MM_Q8_BLOCK || !outs_q || !outs_k ||
        !gvq_floats(input, (uint64_t)rows * cols) || !gvq_floats(iq, (uint64_t)rows * outs_q) ||
        !gvq_floats(ik, (uint64_t)rows * outs_k)) { return 0; }
    const bool staged = row_bytes % 16 == 0 && (uintptr_t)a % 16 == 0 && (uintptr_t)b % 16 == 0;
    const uint32_t first1 = (outs_q + MM_Q8_WARPS - 1) / MM_Q8_WARPS;
    const uint32_t grid = groups + first1 + (outs_k + MM_Q8_WARPS - 1) / MM_Q8_WARPS;
    const uint64_t q4_bytes = 4ull * cols / GLMVQ_Q4_BLOCK * GLMVQ_Q4_BYTES;
    const uint64_t bytes = staged && MM_Q8_WARPS * row_bytes > q4_bytes ? MM_Q8_WARPS * row_bytes : q4_bytes;
    if (bytes + GLMVQ_Q4_WARPS * 8 * sizeof(float) > ctx->q4.shared) { return 0; }
    const mm_q8_two m = {a, b, (float *)iq->ptr, (float *)ik->ptr, outs_q, outs_k, first1};
    mm_rows<MM_ROUTER_ROWS>(rows, [&](auto r) {
        constexpr uint32_t R = decltype(r)::value, NT = R == 1 ? 1 : R <= 4 ? 4 : 8;
        // Half the unified memory as shared, as glmvq_q4_stream_many: the
        // input row is read through L1.
        static const cudaError_t carve =
            cudaFuncSetAttribute(mm_proj_kernel<NT, R>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
        (void)carve;
        glmvq_launch_kernel(mm_mode(rows), mm_proj_kernel<NT, R>, grid, GLMVQ_Q4_WARPS * 32, bytes,
            ds4_current_stream(), many, m, groups, (const float *)input->ptr, cols, rows, staged);
    });
    return cuda_ok(cudaGetLastError(), "MiniMax q/k/v and indexer projections");
}

/* Decode and verify widths: the shared expert's gate/up (Q4_K) and the F32
 * router logits of one input in one launch, CTAs [0, q4_groups) on gate/up
 * (mm_q4_group), the rest one expert per warp as mm_router_gemv<ROWS>
 * computes it. Apart, the router's CTAs waited for the whole down
 * projection before streaming their 3 MiB. */
// Router CTAs of the joint launch (one warp per expert), first in its grid,
// and their count of finished CTAs. A launch passes griddepcontrol.wait only
// after every earlier kernel completed, so one ticket serves every layer.
enum { MM_SR_ROUTER = MM_EXPERTS / GLMVQ_Q4_WARPS };
__device__ uint32_t mm_sr_ticket;

template <uint32_t NT, uint32_t ROWS>
__global__ void __launch_bounds__(GLMVQ_Q4_WARPS * 32, 4) mm_shared_router_kernel(glmvq_q4_many many,
        float *logits, const float *router, const float *input, uint32_t tokens, int32_t *ids, float *weights,
        const float *bias) {
    extern __shared__ uint4 mm_sr_smem[];
    __shared__ float partial[GLMVQ_Q4_WARPS * NT];
    __shared__ bool last;
    glmvq_pdl_open();
    if (blockIdx.x >= MM_SR_ROUTER) {
        mm_q4_group<NT>(many, input, MM_HIDDEN, tokens, mm_sr_smem, partial, blockIdx.x - MM_SR_ROUTER);
        return;
    }
    glmvq_pdl_wait();
    const uint32_t lane = threadIdx.x & 31, e = blockIdx.x * GLMVQ_Q4_WARPS + threadIdx.x / 32;
    const float4 *we = (const float4 *)(router + (uint64_t)e * MM_HIDDEN);
    float acc[ROWS];
    #pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) { acc[r] = 0.0f; }
    #pragma unroll 8
    for (uint32_t i = lane; i < MM_HIDDEN / 4; i += 32) {
        const float4 a = __ldg(we + i);
        #pragma unroll
        for (uint32_t r = 0; r < ROWS; r++) {
            const float4 v = ((const float4 *)(input + (uint64_t)r * MM_HIDDEN))[i];
            acc[r] = fmaf(a.x, v.x, acc[r]);
            acc[r] = fmaf(a.y, v.y, acc[r]);
            acc[r] = fmaf(a.z, v.z, acc[r]);
            acc[r] = fmaf(a.w, v.w, acc[r]);
        }
    }
    #pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) {
        float v = acc[r];
        for (uint32_t stride = 16; stride; stride >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, stride); }
        if (!lane) { logits[(uint64_t)r * MM_EXPERTS + e] = v; }
    }
    // The last router CTA picks every row's experts while the shared
    // expert's CTAs stream (mm_router_kernel's arithmetic, off the chain).
    __threadfence();
    __syncthreads();
    if (!threadIdx.x) { last = atomicAdd(&mm_sr_ticket, 1u) == MM_SR_ROUTER - 1; }
    __syncthreads();
    if (!last) { return; }
    __threadfence();
    if (threadIdx.x / 32 < ROWS) { mm_route_row(ids, weights, logits, bias, threadIdx.x / 32, lane); }
    // Every router CTA has counted: the next launch starts from zero.
    if (!threadIdx.x) { mm_sr_ticket = 0; }
}

/* The shared expert's gate/up (`weights`, Q4_K, into outs), the router
 * logits (record `router`) of `input` and the routes (selected, route
 * weights; bias record `bias`) at decode and verify widths in one launch.
 * 0 when a record is not of those types. */
extern "C" int ds4_gpu_minimax_shared_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs, const uint32_t *weights,
        ds4_gpu_tensor *logits, uint32_t router, const ds4_gpu_tensor *input, uint32_t rows, ds4_gpu_tensor *selected,
        ds4_gpu_tensor *route_weights, uint32_t bias) {
    static_assert(MM_EXPERTS % GLMVQ_Q4_WARPS == 0, "whole router CTAs");
    if (!ctx || !rows || rows > MM_ROUTER_ROWS || !ctx->direct_q4 || !input || (uintptr_t)input->ptr % 16 ||
        !gvq_stream(ctx)) { return 0; }
    const auto *w = (const float *)gvq_weight(ctx, router, GVQ_TYPE_F32, (uint64_t)MM_EXPERTS * MM_HIDDEN);
    const auto *b = (const float *)gvq_weight(ctx, bias, GVQ_TYPE_F32, MM_EXPERTS);
    if (!b || !selected || selected->bytes < (uint64_t)rows * MM_USED * sizeof(int32_t) ||
        !gvq_floats(route_weights, (uint64_t)rows * MM_USED)) { return 0; }
    glmvq_q4_many many;
    uint32_t cols = 0;
    const uint32_t groups = mm_q4_many(ctx, &many, outs, weights, 2u, rows, &cols);
    if (!groups || !w || cols != MM_HIDDEN || !gvq_floats(input, (uint64_t)rows * MM_HIDDEN) ||
        !gvq_floats(logits, (uint64_t)rows * MM_EXPERTS)) { return 0; }
    const uint64_t bytes = 4ull * cols / GLMVQ_Q4_BLOCK * GLMVQ_Q4_BYTES;
    if (bytes + GLMVQ_Q4_WARPS * 8 * sizeof(float) > ctx->q4.shared) { return 0; }
    mm_rows<MM_ROUTER_ROWS>(rows, [&](auto r) {
        constexpr uint32_t R = decltype(r)::value, NT = R == 1 ? 1 : R <= 4 ? 4 : 8;
        static const cudaError_t carve =
            cudaFuncSetAttribute(mm_shared_router_kernel<NT, R>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
        (void)carve;
        glmvq_launch_kernel(mm_mode(rows), mm_shared_router_kernel<NT, R>, MM_SR_ROUTER + groups,
            GLMVQ_Q4_WARPS * 32, bytes, ds4_current_stream(), many, (float *)logits->ptr, w,
            (const float *)input->ptr, rows, (int32_t *)selected->ptr, (float *)route_weights->ptr, b);
    });
    return cuda_ok(cudaGetLastError(), "MiniMax shared expert and router logits");
}

extern "C" int ds4_gpu_minimax_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        ds4_gpu_tensor *logits, const ds4_gpu_tensor *input, uint32_t router, uint32_t bias, uint32_t rows) {
    const auto *w = (const float *)gvq_weight(ctx, router, GVQ_TYPE_F32, (uint64_t)MM_EXPERTS * MM_HIDDEN);
    const auto *b = (const float *)gvq_weight(ctx, bias, GVQ_TYPE_F32, MM_EXPERTS);
    if (!w || !b || !rows || !gvq_floats(logits, (uint64_t)rows * MM_EXPERTS) ||
        !gvq_floats(input, (uint64_t)rows * MM_HIDDEN) || !gvq_floats(weights, (uint64_t)rows * MM_USED) ||
        selected->bytes < (uint64_t)rows * MM_USED * sizeof(int32_t) || !gvq_stream(ctx)) { return 0; }
    // F.linear(x.float(), router): the handle's TF32 mode would move near-tied routes.
    const float alpha = 1.0f, beta = 0.0f;
    if (rows <= MM_ROUTER_ROWS) {
        mm_rows<MM_ROUTER_ROWS>(rows, [&](auto r) {
            glmvq_launch_kernel(mm_mode(rows), mm_router_gemv<decltype(r)::value>, MM_EXPERTS / MM_ROUTER_WARPS,
                MM_ROUTER_WARPS * 32, 0, ds4_current_stream(), (float *)logits->ptr, w, (const float *)input->ptr);
        });
    } else if (!cublas_ok(cublasGemmEx(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N, MM_EXPERTS, (int)rows, MM_HIDDEN,
            &alpha, w, CUDA_R_32F, MM_HIDDEN, input->ptr, CUDA_R_32F, MM_HIDDEN, &beta, logits->ptr, CUDA_R_32F,
            MM_EXPERTS, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT), "MiniMax router logits")) {
        return 0;
    }
    return ds4_gpu_minimax_route(ctx, selected, weights, logits, bias, rows);
}

extern "C" int ds4_gpu_minimax_route(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, uint32_t bias, uint32_t rows) {
    const auto *b = (const float *)gvq_weight(ctx, bias, GVQ_TYPE_F32, MM_EXPERTS);
    if (!b || !rows || !gvq_floats(logits, (uint64_t)rows * MM_EXPERTS) ||
        !gvq_floats(weights, (uint64_t)rows * MM_USED) || selected->bytes < (uint64_t)rows * MM_USED * sizeof(int32_t) ||
        !gvq_stream(ctx)) { return 0; }
    enum { WARPS = 8 };
    glmvq_launch_kernel(mm_mode(rows), mm_router_kernel, (rows + WARPS - 1) / WARPS, WARPS * 32, 0,
        ds4_current_stream(), (int32_t *)selected->ptr, (float *)weights->ptr, (const float *)logits->ptr, b, rows);
    return cuda_ok(cudaGetLastError(), "MiniMax router");
}

extern "C" int ds4_gpu_minimax_prepare_decode(void) {
    const char *v = getenv("DS4_MINIMAX_PRELOAD_DECODE");
    if (v && !strcmp(v, "0")) { return 1; }
    // Keep decode code loading independent of the selected prefill path,
    // without eagerly loading every unrelated CUDA/cuBLAS kernel.
    cudaFuncAttributes attributes;
    const void *const q4[] = {
        (const void *)glmvq_q4_stream<1, GLMVQ_Q4_STAGE>,
        (const void *)glmvq_q4_stream<4, GLMVQ_Q4_STAGE>,
        (const void *)glmvq_q4_stream<8, GLMVQ_Q4_STAGE>,
        (const void *)glmvq_q4_stream_many<1>,
        (const void *)glmvq_q4_stream_many<4>,
        (const void *)glmvq_q4_stream_many<8>
    };
    for (const void *kernel : q4) {
        if (!cuda_ok(cudaFuncGetAttributes(&attributes, kernel), "MiniMax decode preload")) { return 0; }
    }
    cudaError_t err = cudaSuccess;
    for (uint32_t rows = 1; rows <= MM_DECODE_ROWS && err == cudaSuccess; rows++) {
        mm_rows<MM_DECODE_ROWS>(rows, [&](auto r) {
            constexpr uint32_t R = decltype(r)::value, NT = R == 1 ? 1 : R <= 4 ? 4 : 8;
            err = cudaFuncGetAttributes(&attributes, mm_proj_kernel<NT, R>);
            if (err == cudaSuccess) { err = cudaFuncGetAttributes(&attributes, mm_shared_router_kernel<NT, R>); }
        });
    }
    return cuda_ok(err, "MiniMax joint decode preload");
}

/* -------------------------------------------------------------- row kernels */

#include "cuda/minimax_rows.cuh"

static bool mm_residual_valid(ds4_gvq_ctx *ctx, const ds4_minimax_residual_out *out, uint32_t rows) {
    if (!ctx || rows <= MM_DECODE_ROWS || !out ||
        (out->format != DS4_GVQ_ROWS_F32 && out->format != DS4_GVQ_ROWS_BF16) ||
        (out->norm_format != DS4_MINIMAX_NORM_F32 && out->norm_format != DS4_MINIMAX_NORM_Q8)) { return false; }
    const bool next = out->next != DS4_GVQ_NO_LINK;
    const bool index_q8 = out->norm_format == DS4_MINIMAX_NORM_Q8;
    if (index_q8 && (!next || !out->q4h || !ctx->scratch ||
            ctx->scratch->bytes < mm_norm_q8_bytes(rows))) { return false; }
    const uint64_t count = (uint64_t)rows * MM_HIDDEN;
    return gvq_floats(out->cur, out->format == DS4_GVQ_ROWS_BF16 ? count / 2 : count) &&
        (!next || (gvq_weight(ctx, out->next, GVQ_TYPE_BF16, MM_HIDDEN) &&
            (index_q8 || gvq_floats(out->norm, count)))) &&
        (!out->q4h || (next && out->q4h->bytes >= count * sizeof(__half) && gvq_floats(out->q4s, rows)));
}

static cudaError_t mm_reduce_finish(ds4_gvq_ctx *ctx, void *out, const __nv_bfloat16 *down, const float *weights,
        const uint32_t *pairs, const uint32_t *order, const void *shared, uint32_t rows, uint32_t bf16,
        const ds4_minimax_residual_out *residual) {
    const cudaStream_t stream = ds4_current_stream();
    if (!residual) {
        mm_reduce_launch(out, down, weights, pairs, order, shared, (uint64_t)rows * MM_HIDDEN, bf16, stream);
        return cudaGetLastError();
    }
    const auto *next = residual->next == DS4_GVQ_NO_LINK ? nullptr
        : (const __nv_bfloat16 *)gvq_weight(ctx, residual->next, GVQ_TYPE_BF16, MM_HIDDEN);
    const bool index_q8 = residual->norm_format == DS4_MINIMAX_NORM_Q8;
    float *target = next ? (float *)(index_q8 ? ctx->scratch->ptr : residual->norm->ptr) : nullptr;
    auto launch = [&](auto *cur, auto *sh, auto index) {
        using CT = std::remove_pointer_t<decltype(cur)>;
        using ST = std::remove_const_t<std::remove_pointer_t<decltype(sh)>>;
        constexpr bool INDEX = decltype(index)::value;
        mm_moe_residual_kernel<CT, ST, INDEX><<<rows, MM_ROW_THREADS, 0, stream>>>(cur, down, weights, pairs, order, sh,
            target, next,
            residual->q4h ? (__half *)residual->q4h->ptr : nullptr,
            residual->q4h ? (float *)residual->q4s->ptr : nullptr);
    };
    auto select = [&](auto *cur, auto index) {
        if (bf16 & DS4_MINIMAX_BF16_SHARED) { launch(cur, (const __nv_bfloat16 *)shared, index); }
        else { launch(cur, (const float *)shared, index); }
    };
    if (residual->format == DS4_GVQ_ROWS_BF16) {
        if (index_q8) { select((__nv_bfloat16 *)residual->cur->ptr, std::true_type{}); }
        else { select((__nv_bfloat16 *)residual->cur->ptr, std::false_type{}); }
    } else {
        if (index_q8) { select((float *)residual->cur->ptr, std::true_type{}); }
        else { select((float *)residual->cur->ptr, std::false_type{}); }
    }
    return cudaGetLastError();
}

template <typename T>
__global__ void mm_rms_kernel(float *out, const T *input, const __nv_bfloat16 *weight) {
    __shared__ float scratch[MM_ROW_THREADS / 32];
    glmvq_pdl_open();
    float w[MM_ROW_VALUES];
    mm_row_weights(weight, w);
    glmvq_pdl_wait();
    const uint64_t base = (uint64_t)blockIdx.x * MM_HIDDEN;
    float x[MM_ROW_VALUES], y[MM_ROW_VALUES];
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) {
        x[j] = glmvq_round(glmvq_value(input[base + threadIdx.x + j * MM_ROW_THREADS]));
    }
    mm_row_rms(x, w, y, scratch);
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) { out[base + threadIdx.x + j * MM_ROW_THREADS] = y[j]; }
}

extern "C" int ds4_gpu_minimax_rms(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t weight,
        uint32_t rows, ds4_gvq_rows format) {
    const auto *w = (const __nv_bfloat16 *)gvq_weight(ctx, weight, GVQ_TYPE_BF16, MM_HIDDEN);
    const uint64_t count = (uint64_t)rows * MM_HIDDEN;
    const bool compact = format == DS4_GVQ_ROWS_BF16;
    if (!w || !rows || !gvq_floats(out, count) || !gvq_floats(x, compact ? count / 2 : count) ||
        (compact && out->ptr == x->ptr)) { return 0; }
    if (compact) {
        glmvq_launch_kernel(mm_mode(rows), mm_rms_kernel<__nv_bfloat16>, rows, MM_ROW_THREADS, 0,
            ds4_current_stream(), (float *)out->ptr, (const __nv_bfloat16 *)x->ptr, w);
    } else {
        glmvq_launch_kernel(mm_mode(rows), mm_rms_kernel<float>, rows, MM_ROW_THREADS, 0,
            ds4_current_stream(), (float *)out->ptr, (const float *)x->ptr, w);
    }
    return cuda_ok(cudaGetLastError(), "MiniMax RMS");
}

extern "C" int ds4_gpu_minimax_residual(ds4_gvq_ctx *ctx, ds4_gpu_tensor *cur, const ds4_gpu_tensor *branch,
        ds4_gpu_tensor *norm, uint32_t next, uint32_t rows, ds4_gpu_tensor *q4h, ds4_gpu_tensor *q4s, int branch_bf16,
        ds4_gvq_rows format, ds4_minimax_norm norm_output) {
    const auto *n = next == DS4_GVQ_NO_LINK ? nullptr
        : (const __nv_bfloat16 *)gvq_weight(ctx, next, GVQ_TYPE_BF16, MM_HIDDEN);
    const uint64_t count = (uint64_t)rows * MM_HIDDEN;
    const bool compact = format == DS4_GVQ_ROWS_BF16;
    const bool index_q8 = norm_output == DS4_MINIMAX_NORM_Q8;
    if (norm_output != DS4_MINIMAX_NORM_F32 && norm_output != DS4_MINIMAX_NORM_Q8) { return 0; }
    if (index_q8 && (!n || !q4h || rows <= MM_DECODE_ROWS || !ctx ||
            ctx->scratch->bytes < mm_norm_q8_bytes(rows))) { return 0; }
    if ((next != DS4_GVQ_NO_LINK && !n) || !rows || !gvq_floats(cur, compact ? count / 2 : count) ||
        !gvq_floats(branch, count) ||
        (n && !index_q8 && !gvq_floats(norm, count)) || (q4h && (!n || q4h->bytes < count * sizeof(__half) ||
            !gvq_floats(q4s, rows)))) { return 0; }
    float *target = n ? (float *)(index_q8 ? ctx->scratch->ptr : norm->ptr) : nullptr;
    auto launch = [&](auto *values, auto index) {
        using CT = std::remove_pointer_t<decltype(values)>;
        constexpr bool INDEX = decltype(index)::value;
        if (branch_bf16) {
            glmvq_launch_kernel(mm_mode(rows), mm_residual_kernel<CT, __nv_bfloat16, INDEX>, rows, MM_ROW_THREADS, 0,
                ds4_current_stream(), values, (const __nv_bfloat16 *)branch->ptr, target,
                n, q4h ? (__half *)q4h->ptr : nullptr, q4h ? (float *)q4s->ptr : nullptr);
        } else {
            glmvq_launch_kernel(mm_mode(rows), mm_residual_kernel<CT, float, INDEX>, rows, MM_ROW_THREADS, 0,
                ds4_current_stream(), values, (const float *)branch->ptr, target, n,
                q4h ? (__half *)q4h->ptr : nullptr, q4h ? (float *)q4s->ptr : nullptr);
        }
    };
    if (compact && index_q8) { launch((__nv_bfloat16 *)cur->ptr, std::true_type{}); }
    else if (compact) { launch((__nv_bfloat16 *)cur->ptr, std::false_type{}); }
    else if (index_q8) { launch((float *)cur->ptr, std::true_type{}); }
    else { launch((float *)cur->ptr, std::false_type{}); }
    return cuda_ok(cudaGetLastError(), "MiniMax residual");
}

__global__ void mm_swiglu_kernel(float *out, const float *gate, const float *up, uint64_t count) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = glmvq_swiglu_value<GLMVQ_CLAMP_OAI>(glmvq_round(gate[i]), glmvq_round(up[i]), MM_LIMIT);
}

// SwiGLU-OAI rows straight into q4p's prepared input for down (nothing else
// reads the F32 rows at prefill widths).
template <uint32_t WIDTH, typename GT>
__global__ void __launch_bounds__(MM_ROW_THREADS) mm_swiglu_q4p_kernel(__half *q4h, float *q4s, const GT *gate,
        const GT *up) {
    __shared__ float scratch[MM_ROW_THREADS / 32];
    constexpr uint32_t VALUES = WIDTH / MM_ROW_THREADS;
    const uint64_t base = (uint64_t)blockIdx.x * WIDTH;
    float v[VALUES];
    #pragma unroll
    for (uint32_t j = 0; j < VALUES; j++) {
        const uint64_t at = base + threadIdx.x + j * MM_ROW_THREADS;
        v[j] = glmvq_swiglu_value<GLMVQ_CLAMP_OAI>(glmvq_round(glmvq_value(gate[at])), glmvq_round(glmvq_value(up[at])),
            MM_LIMIT);
    }
    mm_row_q4p<VALUES>(v, WIDTH, q4h, q4s, scratch);
}

extern "C" int ds4_gpu_minimax_swiglu_q4p(ds4_gpu_tensor *q4h, ds4_gpu_tensor *q4s, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up, uint32_t rows, uint32_t width, int bf16) {
    const uint64_t count = (uint64_t)rows * width;
    if (!rows || (width != MM_FFN && width != DS4_MINIMAX_DENSE_FF) || !q4h || q4h->bytes < count * sizeof(__half) ||
        !gvq_floats(q4s, rows) || !gvq_floats(gate, count) || !gvq_floats(up, count)) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    const auto launch = [&](auto width_c, const auto *g, const auto *u) {
        mm_swiglu_q4p_kernel<decltype(width_c)::value><<<rows, MM_ROW_THREADS, 0, stream>>>((__half *)q4h->ptr,
            (float *)q4s->ptr, g, u);
    };
    using ffn = std::integral_constant<uint32_t, MM_FFN>;
    using dense = std::integral_constant<uint32_t, DS4_MINIMAX_DENSE_FF>;
    const auto *gb = (const __nv_bfloat16 *)gate->ptr, *ub = (const __nv_bfloat16 *)up->ptr;
    const auto *gf = (const float *)gate->ptr, *uf = (const float *)up->ptr;
    if (width == MM_FFN) {
        if (bf16) { launch(ffn{}, gb, ub); } else { launch(ffn{}, gf, uf); }
    } else {
        if (bf16) { launch(dense{}, gb, ub); } else { launch(dense{}, gf, uf); }
    }
    return cuda_ok(cudaGetLastError(), "MiniMax SwiGLU-OAI q4p rows");
}

extern "C" int ds4_gpu_minimax_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
        uint32_t rows, uint32_t width) {
    const uint64_t count = (uint64_t)rows * width;
    if (!count || !gvq_floats(out, count) || !gvq_floats(gate, count) || !gvq_floats(up, count)) { return 0; }
    glmvq_launch_kernel(mm_mode(rows), mm_swiglu_kernel, (uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS),
        GLMVQ_THREADS, 0, ds4_current_stream(), (float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr,
        count);
    return cuda_ok(cudaGetLastError(), "MiniMax SwiGLU-OAI");
}

/* RoPE rows: the reference evaluates inv_freq = 1 / 1e7^(2j/64) and the
 * angles pos * inv_freq in float64, then cos/sin in float64 rounded to F32
 * once. GB10 runs FP64 at 1/64 rate, so every q/k kernel of the forward
 * reads these rows instead. */
__global__ void mm_rope_kernel(float *rope, uint32_t pos0) {
    const uint32_t j = threadIdx.x, pos = pos0 + blockIdx.x;
    const double inv = 1.0 / pow(1.0e7, (double)(2u * j) / (double)MM_ROT);
    const double angle = (double)pos * inv;
    rope[(uint64_t)blockIdx.x * MM_ROT + j] = (float)cos(angle);
    rope[(uint64_t)blockIdx.x * MM_ROT + MM_ROT / 2 + j] = (float)sin(angle);
}

extern "C" int ds4_gpu_minimax_rope(ds4_gpu_tensor *rope, uint32_t rows, uint32_t pos0) {
    if (!rows || !gvq_floats(rope, (uint64_t)rows * MM_ROT)) { return 0; }
    mm_rope_kernel<<<rows, MM_ROT / 2, 0, ds4_current_stream()>>>((float *)rope->ptr, pos0);
    return cuda_ok(cudaGetLastError(), "MiniMax RoPE rows");
}

/* Per-head Gemma RMSNorm of head row p (thread i: dim i, w its norm
 * weight), rounded once; then the NeoX rotation of dims [0, 64) as torch's
 * separate F32 ops (x1 c - x2 s, x2 c + x1 s) on rope row t, rounded once;
 * dims [64, 128) pass. */
template <typename T>
__device__ __forceinline__ void mm_qk_head(T *p, float w, const float *t, float *scratch, float *y) {
    const uint32_t i = threadIdx.x;
    const float v = glmvq_round(glmvq_value(p[i]));
    float sum = v * v;
    for (uint32_t stride = 16; stride; stride >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, stride); }
    if ((i & 31) == 0) { scratch[i / 32] = sum; }
    __syncthreads();
    sum = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    const float inv = __frsqrt_rn(glmvq_add(__fdiv_rn(sum, (float)MM_HEAD), MM_EPS));
    y[i] = glmvq_round(glmvq_mul(glmvq_add(1.0f, w), glmvq_mul(v, inv)));
    __syncthreads();
    if (i >= MM_ROT) {
        p[i] = glmvq_cast<T>(y[i]);
        return;
    }
    const uint32_t j = i % (MM_ROT / 2);
    const float c = t[j], s = t[MM_ROT / 2 + j];
    const float a = y[j], b = y[j + MM_ROT / 2];
    p[i] = glmvq_cast<T>(i < MM_ROT / 2 ? glmvq_round(glmvq_add(glmvq_mul(a, c), -glmvq_mul(b, s)))
                                        : glmvq_round(glmvq_add(glmvq_mul(b, c), glmvq_mul(a, s))));
}

// One block per (head, row).
__global__ void mm_qk_kernel(float *x, const __nv_bfloat16 *weight, uint32_t heads, const float *rope) {
    __shared__ float scratch[MM_HEAD / 32];
    __shared__ float y[MM_HEAD];
    glmvq_pdl_open();
    const float w = __bfloat162float(weight[threadIdx.x]);
    glmvq_pdl_wait();
    const uint32_t h = blockIdx.x, r = blockIdx.y;
    mm_qk_head(x + ((uint64_t)r * heads + h) * MM_HEAD, w, rope + (uint64_t)r * MM_ROT, scratch, y);
}

extern "C" int ds4_gpu_minimax_qk(ds4_gvq_ctx *ctx, ds4_gpu_tensor *x, uint32_t weight, uint32_t heads,
        uint32_t rows, const ds4_gpu_tensor *rope) {
    const auto *w = (const __nv_bfloat16 *)gvq_weight(ctx, weight, GVQ_TYPE_BF16, MM_HEAD);
    if (!w || !rows || !heads || !gvq_floats(x, (uint64_t)rows * heads * MM_HEAD) ||
        !gvq_floats(rope, (uint64_t)rows * MM_ROT)) { return 0; }
    glmvq_launch_kernel(mm_mode(rows), mm_qk_kernel, dim3(heads, rows), MM_HEAD, 0, ds4_current_stream(),
        (float *)x->ptr, w, heads, (const float *)rope->ptr);
    return cuda_ok(cudaGetLastError(), "MiniMax q/k norm");
}

/* -------------------------------------------------------------------- NVFP4 */

// One 128-wide vector, a warp: lane l holds values 4l..4l+3 of scale group
// l / 4. The vLLM encoder: scale = E4M3(max(amax / 6, 2^-9)) (global 1),
// codes = E2M1(x / scale) rounded to nearest even, saturated at 6.
__device__ __forceinline__ void mm_fp4_encode(const float *src, uint8_t *data, uint8_t *scales) {
    const uint32_t lane = threadIdx.x & 31;
    float v[4], amax = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        v[i] = glmvq_round(src[lane * 4 + i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float wanted = fmaxf(__fdiv_rn(amax, 6.0f), 0.001953125f);
    const __nv_fp8_storage_t code = __nv_cvt_float_to_fp8(wanted, __NV_SATFINITE, __NV_E4M3);
    const float scale = glmvq_e4m3((uint8_t)code);
    uint8_t n[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) { n[i] = ds4_fp4_f32_to_nibble(__fdiv_rn(v[i], scale)); }
    data[lane * 2] = (uint8_t)(n[0] | n[1] << 4);
    data[lane * 2 + 1] = (uint8_t)(n[2] | n[3] << 4);
    if ((lane & 3) == 0) { scales[lane / 4] = (uint8_t)code; }
}

// The pages of blocks [0, ceil(pos / 128)) are addressable: directly, or
// through a table of that many slots.
static bool mm_pages_ok(const ds4_gpu_tensor *pages, const ds4_gpu_tensor *table, uint32_t pos) {
    const uint64_t blocks = (pos + MM_BLOCK - 1) / MM_BLOCK;
    return pages && (table ? table->bytes >= blocks * sizeof(int32_t)
                           : pages->bytes >= blocks * MM_KV_HEADS * MM_PAGE);
}

/* The projections' tails, one block per (b, row): b < 64 normalizes and
 * rotates q head b; 64..67 k head b - 64, then stores it in the row's K
 * page; 68..71 rounds v head b - 68 (prefill projections leave F32 sums)
 * and stores it in the V page; 72..75 index_q head b - 72; 76 the index key,
 * stored as the token's 72-byte row (data, then scales). A table maps
 * blocks to pool slots (SSD tier). One launch for what took seven. */
enum { MM_TAIL_K = MM_HEADS, MM_TAIL_V = MM_TAIL_K + MM_KV_HEADS, MM_TAIL_IQ = MM_TAIL_V + MM_KV_HEADS,
       MM_TAIL_IK = MM_TAIL_IQ + MM_KV_HEADS, MM_TAIL_BLOCKS = MM_TAIL_IK + 1 };
struct mm_tail_args {
    void *q;   // QT rows (mm_tail_kernel<QT>)
    float *k, *v, *iq, *ik;
    const __nv_bfloat16 *qn, *kn, *iqn, *ikn;
    uint8_t *pages, *index;
    const int32_t *table;
    const float *rope;
    uint32_t pos;
};

template <typename QT>
__global__ void __launch_bounds__(MM_HEAD) mm_tail_kernel(mm_tail_args a) {
    __shared__ float scratch[MM_HEAD / 32];
    __shared__ float y[MM_HEAD];
    glmvq_pdl_open();
    const uint32_t b = blockIdx.x, r = blockIdx.y;
    if (b < MM_TAIL_K) {
        // q heads: QT rows (BF16 at prefill when q4p wrote them so).
        const float w = __bfloat162float(a.qn[threadIdx.x]);
        glmvq_pdl_wait();
        mm_qk_head((QT *)a.q + ((uint64_t)r * MM_HEADS + b) * MM_HEAD, w, a.rope + (uint64_t)r * MM_ROT, scratch, y);
        return;
    }
    // Selected by compares: a dynamic index would copy the parameters to local memory.
    float *x = a.k;
    const __nv_bfloat16 *norm = a.kn;
    uint32_t heads = MM_KV_HEADS, h = b - MM_TAIL_K;
    if (b >= MM_TAIL_IK) { x = a.ik; norm = a.ikn; heads = 1; h = 0; }
    else if (b >= MM_TAIL_IQ) { x = a.iq; norm = a.iqn; heads = MM_KV_HEADS; h = b - MM_TAIL_IQ; }
    else if (b >= MM_TAIL_V) { x = a.v; norm = nullptr; heads = MM_KV_HEADS; h = b - MM_TAIL_V; }
    const float w = norm ? __bfloat162float(norm[threadIdx.x]) : 0.0f;
    glmvq_pdl_wait();
    float *p = x + ((uint64_t)r * heads + h) * MM_HEAD;
    if (norm) { mm_qk_head(p, w, a.rope + (uint64_t)r * MM_ROT, scratch, y); }
    else { p[threadIdx.x] = glmvq_round(p[threadIdx.x]); }
    if (b >= MM_TAIL_IQ && b < MM_TAIL_IK) { return; }
    __syncthreads();
    if (threadIdx.x >= 32) { return; }
    const uint32_t t = a.pos + r;
    if (b == MM_TAIL_IK) {
        uint8_t *row = a.index + (uint64_t)t * MM_FP4_ROW;
        mm_fp4_encode(p, row, row + MM_FP4_DATA);
        return;
    }
    const uint32_t block = t / MM_BLOCK, slot = t % MM_BLOCK, side = b >= MM_TAIL_V;
    const uint64_t page_id = a.table ? (uint64_t)a.table[block] : block;
    uint8_t *page = a.pages + (page_id * MM_KV_HEADS + h) * MM_PAGE + side * (MM_PAGE / 2);
    mm_fp4_encode(p, page + slot * MM_FP4_DATA, page + MM_BLOCK * MM_FP4_DATA + slot * (MM_FP4_ROW - MM_FP4_DATA));
}

extern "C" int ds4_gpu_minimax_tail(ds4_gvq_ctx *ctx, ds4_gpu_tensor *q, ds4_gpu_tensor *k, ds4_gpu_tensor *v,
        ds4_gpu_tensor *iq, ds4_gpu_tensor *ik, uint32_t q_norm, uint32_t k_norm, uint32_t iq_norm, uint32_t ik_norm,
        ds4_gpu_tensor *pages, const ds4_gpu_tensor *table, ds4_gpu_tensor *index, const ds4_gpu_tensor *rope,
        uint32_t rows, uint32_t pos, int q_bf16) {
    const auto norm = [&](uint32_t weight) {
        return (const __nv_bfloat16 *)gvq_weight(ctx, weight, GVQ_TYPE_BF16, MM_HEAD);
    };
    const mm_tail_args a = {q ? q->ptr : nullptr, k ? (float *)k->ptr : nullptr,
        v ? (float *)v->ptr : nullptr, iq ? (float *)iq->ptr : nullptr, ik ? (float *)ik->ptr : nullptr,
        norm(q_norm), norm(k_norm), norm(iq_norm), norm(ik_norm), pages ? (uint8_t *)pages->ptr : nullptr,
        index ? (uint8_t *)index->ptr : nullptr, table ? (const int32_t *)table->ptr : nullptr,
        rope ? (const float *)rope->ptr : nullptr, pos};
    const uint64_t kv = (uint64_t)rows * MM_KV_HEADS * MM_HEAD;
    if (!rows || !a.qn || !a.kn || !a.iqn || !a.ikn || !mm_pages_ok(pages, table, pos + rows) ||
        !gvq_floats(q, (uint64_t)rows * MM_HEADS * MM_HEAD) || !gvq_floats(k, kv) || !gvq_floats(v, kv) ||
        !gvq_floats(iq, kv) || !gvq_floats(ik, (uint64_t)rows * MM_HEAD) || !gvq_floats(rope, (uint64_t)rows * MM_ROT) ||
        !index || index->bytes < (uint64_t)(pos + rows) * MM_FP4_ROW) { return 0; }
    if (q_bf16) {
        glmvq_launch_kernel(mm_mode(rows), mm_tail_kernel<__nv_bfloat16>, dim3(MM_TAIL_BLOCKS, rows), MM_HEAD, 0,
            ds4_current_stream(), a);
    } else {
        glmvq_launch_kernel(mm_mode(rows), mm_tail_kernel<float>, dim3(MM_TAIL_BLOCKS, rows), MM_HEAD, 0,
            ds4_current_stream(), a);
    }
    return cuda_ok(cudaGetLastError(), "MiniMax q/k/v tails");
}

/* E2M1 to F16 in hardware (cvt.rn.f16x2.e2m1x2, exact): the constant table
 * cost a dependent load per value, 256 per thread of a decode attention CTA
 * (8.8 -> 6.6 us in isolation, the same bits). */
__device__ __forceinline__ float2 mm_e2m1x2(uint32_t byte) {
    uint32_t h;
    asm("{ .reg .b8 lo, hi;\n mov.b16 {lo, hi}, %1;\n cvt.rn.f16x2.e2m1x2 %0, lo; }"
        : "=r"(h) : "h"((unsigned short)byte));
    return __half22float2(*(const __half2 *)&h);
}

// The eight E2M1 values of a word (byte b: values 2b, 2b + 1, low nibble first) as four F16 pairs.
__device__ __forceinline__ void mm_e2m1x8(uint32_t word, uint32_t *h) {
    asm("{ .reg .b8 b0, b1, b2, b3;\n mov.b32 {b0, b1, b2, b3}, %4;\n"
        " cvt.rn.f16x2.e2m1x2 %0, b0;\n cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        " cvt.rn.f16x2.e2m1x2 %2, b2;\n cvt.rn.f16x2.e2m1x2 %3, b3; }"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]) : "r"(word));
}

// The E2M1 values of a byte's two nibbles times their group scale: exact in F32.
__device__ __forceinline__ float2 mm_fp4_pair(uint8_t byte, float scale) {
    const float2 v = mm_e2m1x2(byte);
    return make_float2(v.x * scale, v.y * scale);
}

/* ------------------------------------------------------------- MSA indexer */

enum { MM_SCORE_ROWS = 16, MM_SCORE_THREADS = 256 };

/* Block scores of `count` rows starting at row `first` (positions pos +
 * first...) against one key block per CTA: the max over the block's visible
 * tokens (key <= query position) of index_q[h] . index_key. Keys decode to
 * F32 in shared memory once for MM_SCORE_ROWS rows. The reference's F32
 * matmul: dot order differs, values agree to accumulation order. */
__global__ void mm_score_kernel(float *scores, const float *query, const uint8_t *pool, uint32_t pos,
        uint32_t first, uint32_t count, uint32_t stride) {
    extern __shared__ __align__(16) float mm_score_smem[];
    auto keys = (float (*)[MM_HEAD + 1])mm_score_smem;
    auto q = (float (*)[MM_HEAD])(mm_score_smem + MM_BLOCK * (MM_HEAD + 1));
    const uint32_t block = blockIdx.x, tile = blockIdx.y * MM_SCORE_ROWS;
    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const uint32_t rows = min((uint32_t)MM_SCORE_ROWS, count - tile);
    // The tile's last row sees the most keys; later tokens are never read,
    // and a block past it scores nothing the selection reads.
    const uint32_t last = pos + first + tile + rows - 1;
    const uint32_t t0 = block * MM_BLOCK;
    if (t0 > last) { return; }
    const uint32_t visible = min((uint32_t)MM_BLOCK, last + 1 - t0);
    for (uint32_t t = warp; t < visible; t += MM_SCORE_THREADS / 32) {
        const uint8_t *row = pool + (uint64_t)(t0 + t) * MM_FP4_ROW;
        const float scale = glmvq_e4m3(row[MM_FP4_DATA + lane / 4]);
        const float2 a = mm_fp4_pair(row[lane * 2], scale), b = mm_fp4_pair(row[lane * 2 + 1], scale);
        keys[t][lane * 4] = a.x;
        keys[t][lane * 4 + 1] = a.y;
        keys[t][lane * 4 + 2] = b.x;
        keys[t][lane * 4 + 3] = b.y;
    }
    for (uint32_t i = tid; i < rows * MM_KV_HEADS * MM_HEAD; i += MM_SCORE_THREADS) {
        q[i / MM_HEAD][i % MM_HEAD] = glmvq_round(query[(uint64_t)(first + tile) * MM_KV_HEADS * MM_HEAD + i]);
    }
    __syncthreads();
    // Warp w scores query vectors w, w + 8, ...; lane l tokens l, l + 32, ...
    for (uint32_t v = warp; v < rows * MM_KV_HEADS; v += MM_SCORE_THREADS / 32) {
        const uint32_t row = v / MM_KV_HEADS, head = v % MM_KV_HEADS;
        const uint32_t p = pos + first + tile + row;
        float best = -INFINITY;
        for (uint32_t t = lane; t < MM_BLOCK; t += 32) {
            if (t0 + t > p) { break; }
            float dot = 0.0f;
            #pragma unroll 8
            for (uint32_t d = 0; d < MM_HEAD; d++) { dot = fmaf(q[v][d], keys[t][d], dot); }
            best = fmaxf(best, dot);
        }
        for (uint32_t s = 16; s; s >>= 1) { best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, s)); }
        if (!lane) { scores[((uint64_t)(tile + row) * MM_KV_HEADS + head) * stride + block] = best; }
    }
}

/* Block scores of COUNT <= MM_SCORE_FEW rows (decode, verify) against one
 * key block per CTA: the block's raw keys land in shared memory with 16-byte
 * loads, thread t decodes key t and dots it with every row and head in the
 * tiled kernel's d order, then a block max. Same scores bit for bit; a
 * decode row reads the pool once at stream rate instead of decoding every
 * block into F32 tiles. COUNT is a template argument (mm_rows): with a
 * runtime count the unrolled 32-vector loop issued every predicated vector,
 * 8x a decode row's work (124K decode: 167 us per layer, 53 GB/s of keys). */
enum { MM_SCORE_FEW = 8 };

template <uint32_t COUNT>
__global__ void __launch_bounds__(MM_BLOCK) mm_score_few_kernel(float *scores, const float *query, const uint8_t *pool,
        uint32_t pos, uint32_t first, uint32_t stride) {
    constexpr uint32_t V = COUNT * MM_KV_HEADS, WORDS = MM_BLOCK * MM_FP4_ROW / sizeof(uint4);
    __shared__ __align__(16) uint8_t raw[MM_BLOCK * MM_FP4_ROW];
    __shared__ __align__(16) float q[V][MM_HEAD];
    __shared__ float part[MM_BLOCK / 32][V];
    glmvq_pdl_open();
    const uint32_t block = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const uint32_t t0 = block * MM_BLOCK;
    // The grid stops below the last row's own block (no row reads that
    // score), so every block here is full. Keys of earlier forwards load
    // while the predecessor drains; keys this forward stores (verify rows
    // across a block boundary) wait for the tail.
    const uint4 *src = (const uint4 *)(pool + (uint64_t)t0 * MM_FP4_ROW);
    const bool stored = t0 + MM_BLOCK <= pos + first;
    if (stored) {
        for (uint32_t i = tid; i < WORDS; i += MM_BLOCK) { ((uint4 *)raw)[i] = src[i]; }
    }
    glmvq_pdl_wait();
    if (!stored) {
        for (uint32_t i = tid; i < WORDS; i += MM_BLOCK) { ((uint4 *)raw)[i] = src[i]; }
    }
    for (uint32_t i = tid; i < V * MM_HEAD; i += MM_BLOCK) {
        q[i / MM_HEAD][i % MM_HEAD] = glmvq_round(query[(uint64_t)first * MM_KV_HEADS * MM_HEAD + i]);
    }
    __syncthreads();
    float dot[V];
    #pragma unroll
    for (uint32_t v = 0; v < V; v++) { dot[v] = 0.0f; }
    {
        const uint8_t *key = raw + tid * MM_FP4_ROW;
        #pragma unroll 1
        for (uint32_t g = 0; g < MM_HEAD / 16; g++) {
            const float scale = glmvq_e4m3(key[MM_FP4_DATA + g]);
            #pragma unroll
            for (uint32_t b = 0; b < 8; b++) {
                const float2 kv = mm_fp4_pair(key[g * 8 + b], scale);
                const uint32_t d = g * 16 + b * 2;
                #pragma unroll
                for (uint32_t v = 0; v < V; v++) {
                    dot[v] = fmaf(q[v][d], kv.x, dot[v]);
                    dot[v] = fmaf(q[v][d + 1], kv.y, dot[v]);
                }
            }
        }
    }
    // Row r sees keys up to its position; past it -inf.
    #pragma unroll
    for (uint32_t v = 0; v < V; v++) {
        const uint32_t p = pos + first + v / MM_KV_HEADS;
        float best = t0 + tid <= p ? dot[v] : -INFINITY;
        for (uint32_t s = 16; s; s >>= 1) { best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, s)); }
        if (!lane) { part[warp][v] = best; }
    }
    __syncthreads();
    if (tid < V) {
        float best = part[0][tid];
        for (uint32_t w = 1; w < MM_BLOCK / 32; w++) { best = fmaxf(best, part[w][tid]); }
        scores[(uint64_t)tid * stride + block] = best;
    }
}


// Whether score v of block i ranks before score w of block j: higher
// first, ties to the lower block; NaN never ranks (an empty entry is
// (-inf, UINT32_MAX), so a -inf block still ranks before it).
__device__ __forceinline__ bool mm_topk_before(float v, uint32_t i, float w, uint32_t j) {
    return v > w || (v == w && i < j);
}

/* Top-16 blocks of one (row, KV head) per warp: the own block first, then
 * the fifteen earlier blocks that rank first, the list sorted ascending; -1
 * fills the tail. One pass over the scores: lane l keeps the best fifteen
 * of blocks l, l + 32, ... in registers, then fifteen warp argmax rounds
 * over the lanes' heads merge them. (Fifteen argmax passes over every
 * score took 12 us per layer at 124K and grow with the context.) Rows with
 * at most 16 visible blocks take them all and read no scores. */
__global__ void mm_topk_kernel(int32_t *blocks, const float *scores, uint32_t pos, uint32_t first, uint32_t count,
        uint32_t stride) {
    constexpr uint32_t KEEP = MM_TOPK - 1;
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t v = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (v >= count * MM_KV_HEADS) { return; }
    const uint32_t row = v / MM_KV_HEADS, head = v % MM_KV_HEADS;
    const uint32_t own = (pos + first + row) / MM_BLOCK;
    int32_t *out = blocks + ((uint64_t)(first + row) * MM_KV_HEADS + head) * MM_TOPK;
    if (own < MM_TOPK) {
        if (lane < MM_TOPK) { out[lane] = lane <= own ? (int32_t)lane : -1; }
        return;
    }
    const float *s = scores + ((uint64_t)row * MM_KV_HEADS + head) * stride;
    float kv[KEEP];
    uint32_t ki[KEEP];
    #pragma unroll
    for (uint32_t j = 0; j < KEEP; j++) { kv[j] = -INFINITY; ki[j] = UINT32_MAX; }
    #pragma unroll 4
    for (uint32_t b = lane; b < own; b += 32) {
        float cv = s[b];
        uint32_t ci = b;
        if (!mm_topk_before(cv, ci, kv[KEEP - 1], ki[KEEP - 1])) { continue; }
        // Insert: the carried entry swaps down past every entry it ranks before.
        #pragma unroll
        for (uint32_t j = 0; j < KEEP; j++) {
            if (mm_topk_before(cv, ci, kv[j], ki[j])) {
                const float tv = kv[j];
                const uint32_t ti = ki[j];
                kv[j] = cv; ki[j] = ci; cv = tv; ci = ti;
            }
        }
    }
    int32_t pick = lane == 0 ? (int32_t)own : -1;
    for (uint32_t k = 1; k < MM_TOPK; k++) {
        float best = kv[0];
        uint32_t index = ki[0];
        for (uint32_t stride2 = 16; stride2; stride2 >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, stride2);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, index, stride2);
            if (mm_topk_before(ob, oi, best, index)) { best = ob; index = oi; }
        }
        if (lane == k) { pick = index == UINT32_MAX ? -1 : (int32_t)index; }
        // Blocks are lane-disjoint: the lane holding the winner pops it.
        if (index != UINT32_MAX && ki[0] == index) {
            #pragma unroll
            for (uint32_t j = 0; j + 1 < KEEP; j++) { kv[j] = kv[j + 1]; ki[j] = ki[j + 1]; }
            kv[KEEP - 1] = -INFINITY;
            ki[KEEP - 1] = UINT32_MAX;
        }
    }
    // Sort the 16 picks ascending (lanes 0..15), -1 last: rank by the picks
    // below (unsigned) and equal ones on lower lanes.
    uint32_t rank = 0;
    for (uint32_t j = 0; j < MM_TOPK; j++) {
        const uint32_t other = (uint32_t)__shfl_sync(0xffffffffu, pick, j);
        rank += other < (uint32_t)pick || (other == (uint32_t)pick && j < lane);
    }
    if (lane < MM_TOPK) { out[rank] = pick; }
}

// DS4_MINIMAX_SCORE_TILED=1: decode rows take the tiled scorer (A/B).
// DS4_MINIMAX_SCORE_TILED=1: the tiled SIMT scorer at every width (A/B).
static bool mm_score_tiled(void) {
    static int tiled = -1;
    if (tiled < 0) { tiled = getenv("DS4_MINIMAX_SCORE_TILED") != nullptr; }
    return tiled != 0;
}

// Prefill widths score on tensor cores (defined with the attention MMA helpers).
enum { MM_SC_ROWS = 32, MM_SC_GROUP = 32, MM_SC_LD = DS4_MINIMAX_HEAD + 8,
       MM_SC_SMEM_BYTES = DS4_MINIMAX_BLOCK * MM_SC_LD * 2 };   // one block's keys as BF16
__global__ void mm_score_mma_kernel(float *scores, const float *query, const uint8_t *pool, uint32_t pos,
        uint32_t first, uint32_t count, uint32_t stride);
static bool mm_score_mma_ready(void);

// Decoded keys of one block plus the query vectors of MM_SCORE_ROWS rows.
static const uint32_t MM_SCORE_SMEM = (MM_BLOCK * (MM_HEAD + 1) + MM_SCORE_ROWS * MM_KV_HEADS * MM_HEAD) * sizeof(float);

extern "C" int ds4_gpu_minimax_select(ds4_gpu_tensor *blocks, ds4_gpu_tensor *scores, uint32_t score_rows,
        const ds4_gpu_tensor *query, const ds4_gpu_tensor *pool, uint32_t rows, uint32_t pos) {
    static bool ready = false;
    if (!ready) {
        if (!cuda_ok(cudaFuncSetAttribute(mm_score_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                MM_SCORE_SMEM), "MiniMax score shared memory")) { return 0; }
        ready = true;
    }
    const uint32_t nblocks = (pos + rows + MM_BLOCK - 1) / MM_BLOCK;
    if (!rows || !score_rows || !blocks || !scores || !pool ||
        blocks->bytes < (uint64_t)rows * MM_KV_HEADS * MM_TOPK * sizeof(int32_t) ||
        !gvq_floats(query, (uint64_t)rows * MM_KV_HEADS * MM_HEAD) ||
        pool->bytes < (uint64_t)(pos + rows) * MM_FP4_ROW) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    enum { WARPS = 8 };
    for (uint32_t first = 0; first < rows; first += score_rows) {
        const uint32_t count = min(score_rows, rows - first);
        // Rows whose own block is below 16 read every visible block.
        const uint32_t own_last = (pos + first + count - 1) / MM_BLOCK;
        if (own_last >= MM_TOPK) {
            const uint32_t stride = (pos + first + count - 1) / MM_BLOCK + 1;
            if (!gvq_floats(scores, (uint64_t)count * MM_KV_HEADS * stride) || stride > nblocks) { return 0; }
            if (count <= MM_SCORE_FEW && !mm_score_tiled()) {
                // The rows' top-k reads only blocks below each own block.
                mm_rows<MM_SCORE_FEW>(count, [&](auto r) {
                    glmvq_launch_kernel(mm_mode(count), mm_score_few_kernel<decltype(r)::value>, stride - 1, MM_BLOCK, 0,
                        stream, (float *)scores->ptr, (const float *)query->ptr, (const uint8_t *)pool->ptr, pos,
                        first, stride);
                });
            } else if (!mm_score_tiled()) {
                if (!mm_score_mma_ready()) { return 0; }
                mm_score_mma_kernel<<<dim3((count + MM_SC_ROWS - 1) / MM_SC_ROWS, (stride + MM_SC_GROUP - 1) / MM_SC_GROUP),
                    256, MM_SC_SMEM_BYTES, stream>>>((float *)scores->ptr, (const float *)query->ptr,
                    (const uint8_t *)pool->ptr, pos, first, count, stride);
            } else {
                mm_score_kernel<<<dim3(stride, (count + MM_SCORE_ROWS - 1) / MM_SCORE_ROWS), MM_SCORE_THREADS,
                    MM_SCORE_SMEM, stream>>>((float *)scores->ptr, (const float *)query->ptr, (const uint8_t *)pool->ptr,
                    pos, first, count, stride);
            }
            glmvq_launch_kernel(mm_mode(count), mm_topk_kernel, (count * MM_KV_HEADS + WARPS - 1) / WARPS, WARPS * 32,
                0, stream, (int32_t *)blocks->ptr, (const float *)scores->ptr, pos, first, count, stride);
        } else {
            glmvq_launch_kernel(mm_mode(count), mm_topk_kernel, (count * MM_KV_HEADS + WARPS - 1) / WARPS, WARPS * 32,
                0, stream, (int32_t *)blocks->ptr, (const float *)scores->ptr, pos, first, count, 1u);
        }
    }
    return cuda_ok(cudaGetLastError(), "MiniMax block selection");
}

/* -------------------------------------------------------------- attention */

enum { MM_ATTN_THREADS = 128, MM_ATTN_TILE = 32 };

/* Sparse attention of one row and KV head over `nb` of its selected blocks
 * starting at list entry `j0`: the 16 query heads of the group share the
 * K/V. Tiles of 32 keys decode to F32 in shared memory; thread i scores
 * head i / 8 against 4 keys and owns 16 output dims of that head; online
 * softmax in F32. FINAL rounds O / l into `heads`; otherwise one partial
 * (max, sum, O) per head goes to `partial`. */
template <bool FINAL>
__global__ void __launch_bounds__(MM_ATTN_THREADS) mm_attn_kernel(float *heads, float *partial, const float *q,
        const uint8_t *pages, const int32_t *table, const int32_t *blocks, uint32_t pos, uint32_t per) {
    __shared__ float qs[MM_GROUP][MM_HEAD];
    __shared__ float ks[MM_ATTN_TILE][MM_HEAD + 1];
    __shared__ float vs[MM_ATTN_TILE][MM_HEAD];
    __shared__ float ps[MM_GROUP][MM_ATTN_TILE];
    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const uint32_t row = FINAL ? blockIdx.x : 0u, kv = FINAL ? blockIdx.y : blockIdx.x / MM_TOPK;
    const uint32_t j0 = FINAL ? 0u : blockIdx.x % MM_TOPK * per;
    const uint32_t p = pos + row;
    const int32_t *list = blocks + ((uint64_t)row * MM_KV_HEADS + kv) * MM_TOPK;
    for (uint32_t i = tid; i < MM_GROUP * MM_HEAD; i += MM_ATTN_THREADS) {
        qs[i / MM_HEAD][i % MM_HEAD] = glmvq_round(q[((uint64_t)row * MM_HEADS + kv * MM_GROUP) * MM_HEAD + i]);
    }
    const uint32_t h = tid / 8, part = tid % 8;
    const float scale = 0.08838834764831845f;   // 128^-0.5
    float m = -INFINITY, l = 0.0f, o[16] = {};
    for (uint32_t j = j0; j < min(j0 + per, (uint32_t)MM_TOPK); j++) {
        const int32_t b = list[j];
        if (b < 0) { break; }
        const uint64_t page_id = table ? (uint64_t)table[b] : (uint64_t)b;
        const uint8_t *page = pages + (page_id * MM_KV_HEADS + kv) * MM_PAGE;
        const uint32_t t0 = (uint32_t)b * MM_BLOCK;
        const uint32_t keys = min((uint32_t)MM_BLOCK, p + 1 - t0);
        for (uint32_t base = 0; base < keys; base += MM_ATTN_TILE) {
            const uint32_t n = min((uint32_t)MM_ATTN_TILE, keys - base);
            __syncthreads();
            // Warp w decodes keys/values w, w + 4, ... of the tile.
            for (uint32_t t = warp; t < n; t += MM_ATTN_THREADS / 32) {
                const uint32_t slot = base + t;
                const uint8_t *kd = page + slot * MM_FP4_DATA, *ksc = page + MM_BLOCK * MM_FP4_DATA + slot * 8;
                const uint8_t *vd = page + MM_PAGE / 2 + slot * MM_FP4_DATA;
                const uint8_t *vsc = page + MM_PAGE / 2 + MM_BLOCK * MM_FP4_DATA + slot * 8;
                const float kscale = glmvq_e4m3(ksc[lane / 4]), vscale = glmvq_e4m3(vsc[lane / 4]);
                const float2 k0 = mm_fp4_pair(kd[lane * 2], kscale), k1 = mm_fp4_pair(kd[lane * 2 + 1], kscale);
                const float2 v0 = mm_fp4_pair(vd[lane * 2], vscale), v1 = mm_fp4_pair(vd[lane * 2 + 1], vscale);
                ks[t][lane * 4] = k0.x; ks[t][lane * 4 + 1] = k0.y; ks[t][lane * 4 + 2] = k1.x; ks[t][lane * 4 + 3] = k1.y;
                vs[t][lane * 4] = v0.x; vs[t][lane * 4 + 1] = v0.y; vs[t][lane * 4 + 2] = v1.x; vs[t][lane * 4 + 3] = v1.y;
            }
            __syncthreads();
            // Scores of head h against keys part * 4 .. part * 4 + 3.
            float s[4], tile_max = -INFINITY;
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++) {
                const uint32_t t = part * 4 + i;
                float dot = 0.0f;
                if (t < n) {
                    #pragma unroll 8
                    for (uint32_t d = 0; d < MM_HEAD; d++) { dot = fmaf(qs[h][d], ks[t][d], dot); }
                }
                s[i] = t < n ? glmvq_mul(dot, scale) : -INFINITY;
                tile_max = fmaxf(tile_max, s[i]);
            }
            for (uint32_t stride = 1; stride < 8; stride <<= 1) {
                tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, stride));
            }
            const float next = fmaxf(m, tile_max);
            const float correction = m == -INFINITY ? 0.0f : expf(m - next);
            float tile_sum = 0.0f;
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++) {
                const float e = s[i] == -INFINITY ? 0.0f : expf(s[i] - next);
                ps[h][part * 4 + i] = e;
                tile_sum += e;
            }
            for (uint32_t stride = 1; stride < 8; stride <<= 1) {
                tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, stride);
            }
            l = l * correction + tile_sum;
            m = next;
            __syncwarp();
            #pragma unroll
            for (uint32_t d = 0; d < 16; d++) { o[d] *= correction; }
            for (uint32_t t = 0; t < n; t++) {
                const float w = ps[h][t];
                #pragma unroll
                for (uint32_t d = 0; d < 16; d++) { o[d] = fmaf(w, vs[t][part * 16 + d], o[d]); }
            }
        }
    }
    const uint32_t head = kv * MM_GROUP + h;
    if (FINAL) {
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        #pragma unroll
        for (uint32_t d = 0; d < 16; d++) {
            heads[((uint64_t)row * MM_HEADS + head) * MM_HEAD + part * 16 + d] = glmvq_round(o[d] * inv);
        }
        return;
    }
    float *out = partial + ((uint64_t)blockIdx.x * MM_GROUP + h) * (MM_HEAD + 2);
    #pragma unroll
    for (uint32_t d = 0; d < 16; d++) { out[part * 16 + d] = o[d]; }
    if (part == 0) {
        out[MM_HEAD] = m;
        out[MM_HEAD + 1] = l;
    }
}

/* Prefill attention on BF16 tensor cores (more than DS4_MINIMAX_SPLIT_ROWS
 * rows; fewer split by block, mm_attn_split_mma_kernel). A CTA takes
 * MM_AT_ROWS consecutive rows and one KV head; warp w is row w's 16 query
 * heads, the m16n8k16 M dimension. The CTA walks the union of its rows'
 * selected blocks in ascending order: each block's NVFP4 K/V page is decoded
 * once into shared memory as BF16 (exact: E2M1 x E4M3), and every warp whose
 * row selected it runs S = Q K^T in F32, the causal mask in its own block,
 * an F32 online softmax and O += P V with P in BF16 (FlashAttention-2).
 * Not bit-equal to the SIMT kernel (MMA accumulation order, BF16 P); the
 * quality gate judges it. DS4_MINIMAX_ATTN_SIMT=1 keeps the SIMT kernel. */
enum { MM_AT_ROWS = 8, MM_AT_LD = MM_HEAD + 8 };   // padded row stride: conflict-free ldmatrix
static const uint32_t MM_AT_SMEM = 2u * MM_BLOCK * MM_AT_LD * sizeof(__nv_bfloat16);

__device__ __forceinline__ void mm_ldsm4(uint32_t *r, const void *p, bool trans) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    if (trans) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    }
}

__device__ __forceinline__ void mm_mma_bf16(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t mm_bf16x2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *(const uint32_t *)&v;
}

// 64 NVFP4 values (eight data words; word w scaled by byte w / 2 of sc) as
// 32 BF16 pairs, exact: an E2M1 value times its E4M3 scale has at most six
// significant bits.
__device__ __forceinline__ void mm_fp4_bf16(const uint32_t *words, uint32_t sc, uint32_t *packed) {
    #pragma unroll
    for (uint32_t w = 0; w < 8; w++) {
        const float s = glmvq_e4m3((uint8_t)(sc >> (w / 2 * 8)));
        uint32_t h[4];
        mm_e2m1x8(words[w], h);
        #pragma unroll
        for (uint32_t k = 0; k < 4; k++) {
            const float2 v = __half22float2(*(const __half2 *)&h[k]);
            packed[w * 4 + k] = mm_bf16x2(v.x * s, v.y * s);
        }
    }
}

/* Block scores of prefill rows on BF16 tensor cores. A CTA takes 32 rows
 * (128 query vectors: row x index head, warp w the vectors 16w..16w+15 as
 * the m16n8k16 M dimension) and up to MM_SC_GROUP key blocks; each block's
 * NVFP4 index keys decode once into shared memory as BF16 (exact: E2M1 x
 * E4M3), S = Q K^T accumulates in F32 and each vector keeps the max over
 * its visible keys (key <= its row's position). The SIMT scorer re-decoded
 * a block for every 16 rows and ran at ~2.8 TFLOPS, 26% of a 32K prefill.
 * Values agree with it to accumulation order; the top-16 can move at near
 * ties, so prefill rows are judged by the quality gate (decode and verify
 * widths keep mm_score_few_kernel). */
__global__ void __launch_bounds__(256) mm_score_mma_kernel(float *scores, const float *query, const uint8_t *pool,
        uint32_t pos, uint32_t first, uint32_t count, uint32_t stride) {
    extern __shared__ __align__(16) uint8_t mm_sc_smem[];
    __nv_bfloat16 *ks = (__nv_bfloat16 *)mm_sc_smem;
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32, g = lane >> 2, t = lane & 3;
    const uint32_t tile = blockIdx.x * MM_SC_ROWS, rows = min((uint32_t)MM_SC_ROWS, count - tile);
    const uint32_t last = pos + first + tile + rows - 1;
    const uint32_t b0 = blockIdx.y * MM_SC_GROUP, b1 = min(b0 + MM_SC_GROUP, last / MM_BLOCK + 1);
    if (b0 >= b1) { return; }
    // Vectors va = 16w + g and vb = va + 8 of this thread's C fragment rows.
    const uint32_t va = warp * 16 + g, vb = va + 8;
    const bool live_a = va / MM_KV_HEADS < rows, live_b = vb / MM_KV_HEADS < rows;
    const uint32_t pa = pos + first + tile + va / MM_KV_HEADS, pb = pos + first + tile + vb / MM_KV_HEADS;
    const float *q = query + (uint64_t)(first + tile) * MM_KV_HEADS * MM_HEAD;
    uint32_t qa[8][4];
    #pragma unroll
    for (uint32_t j = 0; j < 8; j++) {
        const uint32_t c = 16 * j + 2 * t;
        const float *ra = q + (uint64_t)va * MM_HEAD, *rb = q + (uint64_t)vb * MM_HEAD;
        qa[j][0] = live_a ? mm_bf16x2(ra[c], ra[c + 1]) : 0u;
        qa[j][1] = live_b ? mm_bf16x2(rb[c], rb[c + 1]) : 0u;
        qa[j][2] = live_a ? mm_bf16x2(ra[c + 8], ra[c + 9]) : 0u;
        qa[j][3] = live_b ? mm_bf16x2(rb[c + 8], rb[c + 9]) : 0u;
    }
    for (uint32_t b = b0; b < b1; b++) {
        const uint32_t t0 = b * MM_BLOCK, visible = min((uint32_t)MM_BLOCK, last + 1 - t0);
        __syncthreads();   // the previous block's fragments are read
        {
            // Thread i: key i / 2, values 64 (i % 2).. of its 128 (32 data bytes, 4 scales).
            const uint32_t key = threadIdx.x / 2, half = threadIdx.x % 2;
            uint32_t packed[32];
            if (key < visible) {
                const uint8_t *row = pool + (uint64_t)(t0 + key) * MM_FP4_ROW;
                const uint2 *d = (const uint2 *)(row + half * 32);
                const uint2 d0 = d[0], d1 = d[1], d2 = d[2], d3 = d[3];
                const uint32_t sc = *(const uint32_t *)(row + MM_FP4_DATA + half * 4);
                const uint32_t words[8] = {d0.x, d0.y, d1.x, d1.y, d2.x, d2.y, d3.x, d3.y};
                mm_fp4_bf16(words, sc, packed);
            } else {
                #pragma unroll
                for (uint32_t k = 0; k < 32; k++) { packed[k] = 0u; }
            }
            uint4 *dst = (uint4 *)(ks + key * MM_SC_LD + half * 64);
            #pragma unroll
            for (uint32_t k = 0; k < 8; k++) {
                dst[k] = make_uint4(packed[k * 4], packed[k * 4 + 1], packed[k * 4 + 2], packed[k * 4 + 3]);
            }
        }
        __syncthreads();
        float best_a = -INFINITY, best_b = -INFINITY;
        #pragma unroll 1
        for (uint32_t half = 0; half < 2; half++) {
            const uint32_t k0 = half * 64;
            float s[8][4];
            #pragma unroll
            for (uint32_t n = 0; n < 8; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f; }
            #pragma unroll
            for (uint32_t j = 0; j < 8; j++) {
                #pragma unroll
                for (uint32_t n = 0; n < 8; n += 2) {
                    const uint32_t mi = lane / 8;
                    const uint32_t key = k0 + (n + (mi >> 1)) * 8 + lane % 8, dim = 16 * j + (mi & 1) * 8;
                    uint32_t b4[4];
                    mm_ldsm4(b4, ks + key * MM_SC_LD + dim, false);
                    mm_mma_bf16(s[n], qa[j], b4[0], b4[1]);
                    mm_mma_bf16(s[n + 1], qa[j], b4[2], b4[3]);
                }
            }
            #pragma unroll
            for (uint32_t n = 0; n < 8; n++) {
                const uint32_t key = t0 + k0 + n * 8 + 2 * t;
                best_a = fmaxf(best_a, key <= pa ? s[n][0] : -INFINITY);
                best_a = fmaxf(best_a, key + 1 <= pa ? s[n][1] : -INFINITY);
                best_b = fmaxf(best_b, key <= pb ? s[n][2] : -INFINITY);
                best_b = fmaxf(best_b, key + 1 <= pb ? s[n][3] : -INFINITY);
            }
        }
        best_a = fmaxf(best_a, __shfl_xor_sync(0xffffffffu, best_a, 1));
        best_a = fmaxf(best_a, __shfl_xor_sync(0xffffffffu, best_a, 2));
        best_b = fmaxf(best_b, __shfl_xor_sync(0xffffffffu, best_b, 1));
        best_b = fmaxf(best_b, __shfl_xor_sync(0xffffffffu, best_b, 2));
        if (!t) {
            if (live_a) { scores[((uint64_t)tile * MM_KV_HEADS + va) * stride + b] = best_a; }
            if (live_b) { scores[((uint64_t)tile * MM_KV_HEADS + vb) * stride + b] = best_b; }
        }
    }
}

static bool mm_score_mma_ready(void) {
    static bool ready = false;
    if (!ready) {
        if (!cuda_ok(cudaFuncSetAttribute(mm_score_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                MM_SC_SMEM_BYTES), "MiniMax MMA score shared memory")) { return false; }
        ready = true;
    }
    return true;
}

// Raw page words of block b for this thread's two (side, key, half) units of
// mm_attn_mma_kernel's decode.
__device__ __forceinline__ void mm_at_raw(const uint8_t *pages, const int32_t *table, uint32_t kv, uint32_t b,
        uint4 *d0, uint4 *d1, uint32_t *sc) {
    const uint64_t page_id = table ? (uint64_t)table[b] : (uint64_t)b;
    const uint8_t *page = pages + (page_id * MM_KV_HEADS + kv) * MM_PAGE;
    #pragma unroll
    for (uint32_t u = 0; u < 2; u++) {
        const uint32_t i = threadIdx.x + u * MM_AT_ROWS * 32;
        const uint32_t side = i / (2u * MM_BLOCK), key = i / 2 % MM_BLOCK, half = i % 2;
        const uint8_t *base = page + side * (MM_PAGE / 2);
        d0[u] = *(const uint4 *)(base + key * MM_FP4_DATA + half * 32);
        d1[u] = *(const uint4 *)(base + key * MM_FP4_DATA + half * 32 + 16);
        sc[u] = *(const uint32_t *)(base + MM_BLOCK * MM_FP4_DATA + key * 8 + half * 4);
    }
}
static_assert(4u * MM_BLOCK == 2u * MM_AT_ROWS * 32, "two decode units per thread");

// Heads as F32 rows or, when q4p prepares o_proj's input from them, BF16
// rows (the values are BF16-rounded either way).
__device__ __forceinline__ void mm_at_store(float *p, float v) { *p = v; }
__device__ __forceinline__ void mm_at_store(__nv_bfloat16 *p, float v) { *p = __float2bfloat16_rn(v); }

template <typename OUT, typename QT>
__global__ void __launch_bounds__(MM_AT_ROWS * 32, 1) mm_attn_mma_kernel(OUT *heads, const QT *q,
        const uint8_t *pages, const int32_t *table, const int32_t *blocks, uint32_t pos, uint32_t rows) {
    extern __shared__ __align__(16) uint8_t mm_at_smem[];
    __nv_bfloat16 *ks = (__nv_bfloat16 *)mm_at_smem, *vs = ks + MM_BLOCK * MM_AT_LD;
    __shared__ uint32_t next[2][MM_AT_ROWS];
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32, g = lane >> 2, t = lane & 3;
    const uint32_t kv = blockIdx.y, row = blockIdx.x * MM_AT_ROWS + warp;
    const bool live = row < rows;
    const uint32_t r = live ? row : rows - 1, p = pos + r;
    const int32_t *list = blocks + ((uint64_t)r * MM_KV_HEADS + kv) * MM_TOPK;
    // Q as A fragments (16 heads x 128 dims, eight k-steps), rounded once.
    uint32_t qa[8][4];
    const QT *qr = q + ((uint64_t)r * MM_HEADS + kv * MM_GROUP) * MM_HEAD;
    #pragma unroll
    for (uint32_t j = 0; j < 8; j++) {
        const uint32_t c = 16 * j + 2 * t;
        qa[j][0] = mm_bf16x2(glmvq_value(qr[g * MM_HEAD + c]), glmvq_value(qr[g * MM_HEAD + c + 1]));
        qa[j][1] = mm_bf16x2(glmvq_value(qr[(g + 8) * MM_HEAD + c]), glmvq_value(qr[(g + 8) * MM_HEAD + c + 1]));
        qa[j][2] = mm_bf16x2(glmvq_value(qr[g * MM_HEAD + c + 8]), glmvq_value(qr[g * MM_HEAD + c + 9]));
        qa[j][3] = mm_bf16x2(glmvq_value(qr[(g + 8) * MM_HEAD + c + 8]), glmvq_value(qr[(g + 8) * MM_HEAD + c + 9]));
    }
    float o[16][4];
    #pragma unroll
    for (uint32_t n = 0; n < 16; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f; }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float scale = 0.08838834764831845f;   // 128^-0.5
    // Lane l < 16 holds entry l of the row's ascending list.
    const int32_t entries = live && lane < MM_TOPK ? list[lane] : -1;
    uint32_t cursor = 0, it = 0;
    int32_t mine = __shfl_sync(0xffffffffu, entries, 0);
    if (!lane) { next[0][warp] = mine < 0 ? UINT32_MAX : (uint32_t)mine; }
    __syncthreads();
    uint32_t b = UINT32_MAX;
    #pragma unroll
    for (uint32_t w = 0; w < MM_AT_ROWS; w++) { b = min(b, next[0][w]); }
    // A block's raw page words load one block ahead: the smallest pending
    // entry is known before the current block's MMAs, which hide the loads.
    uint4 d0[2], d1[2];
    uint32_t sc[2];
    if (b != UINT32_MAX) { mm_at_raw(pages, table, kv, b, d0, d1, sc); }
    while (b != UINT32_MAX) {
        // Block b's K and V pages of this head into shared memory as BF16:
        // 512 units of (side, key, 64 values), 32 data bytes and 4 scales each.
        #pragma unroll
        for (uint32_t u = 0; u < 2; u++) {
            const uint32_t i = threadIdx.x + u * MM_AT_ROWS * 32;
            const uint32_t side = i / (2u * MM_BLOCK), key = i / 2 % MM_BLOCK, half = i % 2;
            const uint32_t words[8] = {d0[u].x, d0[u].y, d0[u].z, d0[u].w, d1[u].x, d1[u].y, d1[u].z, d1[u].w};
            uint32_t packed[32];
            mm_fp4_bf16(words, sc[u], packed);
            uint4 *dst = (uint4 *)((side ? vs : ks) + key * MM_AT_LD + half * 64);
            #pragma unroll
            for (uint32_t k = 0; k < 8; k++) {
                dst[k] = make_uint4(packed[k * 4], packed[k * 4 + 1], packed[k * 4 + 2], packed[k * 4 + 3]);
            }
        }
        const bool take = mine == (int32_t)b;
        const int32_t after = __shfl_sync(0xffffffffu, entries, (cursor + 1) & 31u);
        const int32_t later = !take ? mine : cursor + 1 < MM_TOPK ? after : -1;
        // Double-buffered: a warp may write the next entry while another still
        // reads this one's.
        if (!lane) { next[(it + 1) & 1][warp] = later < 0 ? UINT32_MAX : (uint32_t)later; }
        __syncthreads();
        uint32_t nb = UINT32_MAX;
        #pragma unroll
        for (uint32_t w = 0; w < MM_AT_ROWS; w++) { nb = min(nb, next[(it + 1) & 1][w]); }
        if (nb != UINT32_MAX) { mm_at_raw(pages, table, kv, nb, d0, d1, sc); }
        if (take) {
            const uint32_t t0 = b * MM_BLOCK;
            #pragma unroll 1
            for (uint32_t half = 0; half < 2; half++) {
                const uint32_t k0 = half * 64;
                float s[8][4];
                #pragma unroll
                for (uint32_t n = 0; n < 8; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f; }
                #pragma unroll
                for (uint32_t j = 0; j < 8; j++) {
                    #pragma unroll
                    for (uint32_t n = 0; n < 8; n += 2) {
                        // Keys k0 + n*8.. (n-tiles n, n+1), dims 16j..16j+15.
                        const uint32_t mi = lane / 8;
                        const uint32_t key = k0 + (n + (mi >> 1)) * 8 + lane % 8, dim = 16 * j + (mi & 1) * 8;
                        uint32_t b4[4];
                        mm_ldsm4(b4, ks + key * MM_AT_LD + dim, false);
                        mm_mma_bf16(s[n], qa[j], b4[0], b4[1]);
                        mm_mma_bf16(s[n + 1], qa[j], b4[2], b4[3]);
                    }
                }
                float mx0 = -INFINITY, mx1 = -INFINITY;
                #pragma unroll
                for (uint32_t n = 0; n < 8; n++) {
                    const uint32_t key = t0 + k0 + n * 8 + 2 * t;
                    s[n][0] = key <= p ? s[n][0] * scale : -INFINITY;
                    s[n][1] = key + 1 <= p ? s[n][1] * scale : -INFINITY;
                    s[n][2] = key <= p ? s[n][2] * scale : -INFINITY;
                    s[n][3] = key + 1 <= p ? s[n][3] * scale : -INFINITY;
                    mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
                    mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
                }
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
                const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
                const float c0 = n0 == -INFINITY || m0 == -INFINITY ? (n0 == -INFINITY ? 1.0f : 0.0f) : expf(m0 - n0);
                const float c1 = n1 == -INFINITY || m1 == -INFINITY ? (n1 == -INFINITY ? 1.0f : 0.0f) : expf(m1 - n1);
                float sum0 = 0.0f, sum1 = 0.0f;
                #pragma unroll
                for (uint32_t n = 0; n < 8; n++) {
                    s[n][0] = s[n][0] == -INFINITY ? 0.0f : expf(s[n][0] - n0);
                    s[n][1] = s[n][1] == -INFINITY ? 0.0f : expf(s[n][1] - n0);
                    s[n][2] = s[n][2] == -INFINITY ? 0.0f : expf(s[n][2] - n1);
                    s[n][3] = s[n][3] == -INFINITY ? 0.0f : expf(s[n][3] - n1);
                    sum0 += s[n][0] + s[n][1];
                    sum1 += s[n][2] + s[n][3];
                }
                l0 = fmaf(l0, c0, sum0);
                l1 = fmaf(l1, c1, sum1);
                m0 = n0;
                m1 = n1;
                #pragma unroll
                for (uint32_t n = 0; n < 16; n++) {
                    o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1;
                }
                #pragma unroll
                for (uint32_t kk = 0; kk < 4; kk++) {
                    const uint32_t pa[4] = {mm_bf16x2(s[2 * kk][0], s[2 * kk][1]),
                        mm_bf16x2(s[2 * kk][2], s[2 * kk][3]), mm_bf16x2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                        mm_bf16x2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
                    #pragma unroll
                    for (uint32_t n = 0; n < 16; n += 2) {
                        // Keys k0 + 16kk.. (two 8-key halves), dims n*8.. (n-tiles n, n+1).
                        const uint32_t mi = lane / 8;
                        const uint32_t key = k0 + 16 * kk + (mi & 1) * 8 + lane % 8, dim = (n + (mi >> 1)) * 8;
                        uint32_t b4[4];
                        mm_ldsm4(b4, vs + key * MM_AT_LD + dim, true);
                        mm_mma_bf16(o[n], pa, b4[0], b4[1]);
                        mm_mma_bf16(o[n + 1], pa, b4[2], b4[3]);
                    }
                }
            }
            cursor++;
        }
        mine = later;
        b = nb;
        it++;
        __syncthreads();
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    if (!live) { return; }
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    OUT *out = heads + ((uint64_t)row * MM_HEADS + kv * MM_GROUP) * MM_HEAD;
    #pragma unroll
    for (uint32_t n = 0; n < 16; n++) {
        const uint32_t d = n * 8 + 2 * t;
        mm_at_store(out + g * MM_HEAD + d, glmvq_round(o[n][0] * i0));
        mm_at_store(out + g * MM_HEAD + d + 1, glmvq_round(o[n][1] * i0));
        mm_at_store(out + (g + 8) * MM_HEAD + d, glmvq_round(o[n][2] * i1));
        mm_at_store(out + (g + 8) * MM_HEAD + d + 1, glmvq_round(o[n][3] * i1));
    }
}

// Half a K or V row (64 values) from NVFP4 into tile row `key` as BF16.
__device__ __forceinline__ void mm_at_unpack(__nv_bfloat16 *tile, uint32_t key, uint32_t half, uint4 d0, uint4 d1,
        uint32_t sc) {
    const uint32_t words[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
    uint32_t packed[32];
    mm_fp4_bf16(words, sc, packed);
    uint4 *dst = (uint4 *)(tile + key * MM_AT_LD + half * 64);
    #pragma unroll
    for (uint32_t k = 0; k < 8; k++) {
        dst[k] = make_uint4(packed[k * 4], packed[k * 4 + 1], packed[k * 4 + 2], packed[k * 4 + 3]);
    }
}

/* Decode and verify attention on BF16 tensor cores, split by selected
 * block: CTA (KV head, list entry j, row) decodes the row's block j keys
 * into a shared tile, warps 0-3 score their 32 keys each for the 16 heads
 * (S = Q K^T, the causal mask in the own block, F32 softmax), the values
 * replace the keys in the tile (O = P V with P in BF16), and the four
 * warps' (O, max, sum) merge into the partial that mm_attn_combine reads.
 * All eight warps unpack and merge. One 34 KiB tile keeps two CTAs per SM:
 * the 64 CTAs of a decode row run in one wave (K and V side by side took
 * two). Every global read is issued up front. A row's arithmetic does not
 * depend on the other rows: verify rows equal decode rows bit for bit. */
static const uint32_t MM_AT_SPLIT_SMEM = MM_BLOCK * MM_AT_LD * sizeof(__nv_bfloat16);
static_assert(4u * MM_GROUP * MM_HEAD * sizeof(float) <= MM_AT_SPLIT_SMEM, "the warps' O rows reuse the tile");
enum { MM_AT_SPLIT_THREADS = 256 };

__global__ void __launch_bounds__(MM_AT_SPLIT_THREADS) mm_attn_split_mma_kernel(float *partial, const float *q,
        const uint8_t *pages, const int32_t *table, const int32_t *blocks, uint32_t pos) {
    extern __shared__ __align__(16) uint8_t mm_at_smem[];
    __nv_bfloat16 *tile = (__nv_bfloat16 *)mm_at_smem;
    __shared__ float wm[4][MM_GROUP], wl[4][MM_GROUP];
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32, g = lane >> 2, t = lane & 3;
    const bool mma = warp < 4;   // warps 0-3 score and attend; all eight unpack and merge
    const uint32_t kv = blockIdx.x / MM_TOPK, j = blockIdx.x % MM_TOPK, row = blockIdx.y, p = pos + row;
    float *out = partial + ((uint64_t)row * MM_SPLITS + blockIdx.x) * MM_GROUP * (MM_HEAD + 2);
    const int32_t b = blocks[((uint64_t)row * MM_KV_HEADS + kv) * MM_TOPK + j];
    // The query heads load while the block id (and its pool slot) resolves.
    float qf[8][8];
    const float *qr = q + ((uint64_t)row * MM_HEADS + kv * MM_GROUP) * MM_HEAD;
    if (mma) {
        #pragma unroll
        for (uint32_t s = 0; s < 8; s++) {
            const uint32_t c = 16 * s + 2 * t;
            const float2 a0 = *(const float2 *)(qr + g * MM_HEAD + c), a1 = *(const float2 *)(qr + (g + 8) * MM_HEAD + c);
            const float2 a2 = *(const float2 *)(qr + g * MM_HEAD + c + 8);
            const float2 a3 = *(const float2 *)(qr + (g + 8) * MM_HEAD + c + 8);
            qf[s][0] = a0.x; qf[s][1] = a0.y; qf[s][2] = a1.x; qf[s][3] = a1.y;
            qf[s][4] = a2.x; qf[s][5] = a2.y; qf[s][6] = a3.x; qf[s][7] = a3.y;
        }
    }
    if (b < 0) {
        if (threadIdx.x < MM_GROUP) {
            out[threadIdx.x * (MM_HEAD + 2) + MM_HEAD] = -INFINITY;
            out[threadIdx.x * (MM_HEAD + 2) + MM_HEAD + 1] = 0.0f;
        }
        return;
    }
    const uint64_t page_id = table ? (uint64_t)table[b] : (uint64_t)b;
    const uint8_t *page = pages + (page_id * MM_KV_HEADS + kv) * MM_PAGE;
    // Thread i: half i % 2 of key i / 2, its K (n = 0) and V (n = 1).
    const uint32_t key = threadIdx.x / 2, half = threadIdx.x % 2;
    uint4 d0[2], d1[2];
    uint32_t sc[2];
    #pragma unroll
    for (uint32_t n = 0; n < 2; n++) {
        const uint8_t *base = page + n * (MM_PAGE / 2);
        d0[n] = *(const uint4 *)(base + key * MM_FP4_DATA + half * 32);
        d1[n] = *(const uint4 *)(base + key * MM_FP4_DATA + half * 32 + 16);
        sc[n] = *(const uint32_t *)(base + MM_BLOCK * MM_FP4_DATA + key * 8 + half * 4);
    }
    mm_at_unpack(tile, key, half, d0[0], d1[0], sc[0]);
    uint32_t qa[8][4];
    if (mma) {
        #pragma unroll
        for (uint32_t s = 0; s < 8; s++) {
            qa[s][0] = mm_bf16x2(qf[s][0], qf[s][1]);
            qa[s][1] = mm_bf16x2(qf[s][2], qf[s][3]);
            qa[s][2] = mm_bf16x2(qf[s][4], qf[s][5]);
            qa[s][3] = mm_bf16x2(qf[s][6], qf[s][7]);
        }
    }
    __syncthreads();
    // Warp w < 4: keys 32w..32w+31 (four 8-key n-tiles).
    const uint32_t k0 = warp * 32, t0 = (uint32_t)b * MM_BLOCK;
    float s[4][4];
    #pragma unroll
    for (uint32_t n = 0; n < 4; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f; }
    if (mma) {
        #pragma unroll
        for (uint32_t st = 0; st < 8; st++) {
            #pragma unroll
            for (uint32_t n = 0; n < 4; n += 2) {
                const uint32_t mi = lane / 8;
                const uint32_t kk = k0 + (n + (mi >> 1)) * 8 + lane % 8, dim = 16 * st + (mi & 1) * 8;
                uint32_t b4[4];
                mm_ldsm4(b4, tile + kk * MM_AT_LD + dim, false);
                mm_mma_bf16(s[n], qa[st], b4[0], b4[1]);
                mm_mma_bf16(s[n + 1], qa[st], b4[2], b4[3]);
            }
        }
    }
    // Every warp has read its keys: the values take the tile.
    __syncthreads();
    mm_at_unpack(tile, key, half, d0[1], d1[1], sc[1]);
    const float scale = 0.08838834764831845f;
    float m0 = -INFINITY, m1 = -INFINITY;
    #pragma unroll
    for (uint32_t n = 0; n < 4; n++) {
        const uint32_t kk = t0 + k0 + n * 8 + 2 * t;
        s[n][0] = kk <= p ? s[n][0] * scale : -INFINITY;
        s[n][1] = kk + 1 <= p ? s[n][1] * scale : -INFINITY;
        s[n][2] = kk <= p ? s[n][2] * scale : -INFINITY;
        s[n][3] = kk + 1 <= p ? s[n][3] * scale : -INFINITY;
        m0 = fmaxf(m0, fmaxf(s[n][0], s[n][1]));
        m1 = fmaxf(m1, fmaxf(s[n][2], s[n][3]));
    }
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    float l0 = 0.0f, l1 = 0.0f;
    #pragma unroll
    for (uint32_t n = 0; n < 4; n++) {
        s[n][0] = s[n][0] == -INFINITY ? 0.0f : expf(s[n][0] - m0);
        s[n][1] = s[n][1] == -INFINITY ? 0.0f : expf(s[n][1] - m0);
        s[n][2] = s[n][2] == -INFINITY ? 0.0f : expf(s[n][2] - m1);
        s[n][3] = s[n][3] == -INFINITY ? 0.0f : expf(s[n][3] - m1);
        l0 += s[n][0] + s[n][1];
        l1 += s[n][2] + s[n][3];
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    __syncthreads();
    float o[16][4];
    #pragma unroll
    for (uint32_t n = 0; n < 16; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f; }
    if (mma) {
        #pragma unroll
        for (uint32_t kk = 0; kk < 2; kk++) {
            const uint32_t pa[4] = {mm_bf16x2(s[2 * kk][0], s[2 * kk][1]), mm_bf16x2(s[2 * kk][2], s[2 * kk][3]),
                mm_bf16x2(s[2 * kk + 1][0], s[2 * kk + 1][1]), mm_bf16x2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
            #pragma unroll
            for (uint32_t n = 0; n < 16; n += 2) {
                const uint32_t mi = lane / 8;
                const uint32_t vk = k0 + 16 * kk + (mi & 1) * 8 + lane % 8, dim = (n + (mi >> 1)) * 8;
                uint32_t b4[4];
                mm_ldsm4(b4, tile + vk * MM_AT_LD + dim, true);
                mm_mma_bf16(o[n], pa, b4[0], b4[1]);
                mm_mma_bf16(o[n + 1], pa, b4[2], b4[3]);
            }
        }
    }
    // Merge the four warps: per-warp (max, sum) and O rows into the tile
    // (the values are no longer read), then thread i sums dim i % 128 of
    // heads i / 128, i / 128 + 2, ...
    __syncthreads();
    float *wo = (float *)mm_at_smem;   // [4 warps][16 heads][128]
    if (mma) {
        if (t == 0) {
            wm[warp][g] = m0; wm[warp][g + 8] = m1;
            wl[warp][g] = l0; wl[warp][g + 8] = l1;
        }
        #pragma unroll
        for (uint32_t n = 0; n < 16; n++) {
            const uint32_t d = n * 8 + 2 * t;
            wo[(warp * MM_GROUP + g) * MM_HEAD + d] = o[n][0];
            wo[(warp * MM_GROUP + g) * MM_HEAD + d + 1] = o[n][1];
            wo[(warp * MM_GROUP + g + 8) * MM_HEAD + d] = o[n][2];
            wo[(warp * MM_GROUP + g + 8) * MM_HEAD + d + 1] = o[n][3];
        }
    }
    __syncthreads();
    const uint32_t d = threadIdx.x % MM_HEAD;
    #pragma unroll 1
    for (uint32_t h = threadIdx.x / MM_HEAD; h < MM_GROUP; h += MM_AT_SPLIT_THREADS / MM_HEAD) {
        float best = -INFINITY;
        for (uint32_t w = 0; w < 4; w++) { best = fmaxf(best, wm[w][h]); }
        float sum = 0.0f, acc = 0.0f;
        for (uint32_t w = 0; w < 4; w++) {
            if (wm[w][h] == -INFINITY) { continue; }
            const float c = expf(wm[w][h] - best);
            sum = fmaf(wl[w][h], c, sum);
            acc = fmaf(wo[(w * MM_GROUP + h) * MM_HEAD + d], c, acc);
        }
        out[h * (MM_HEAD + 2) + d] = acc;
        if (!d) {
            out[h * (MM_HEAD + 2) + MM_HEAD] = best;
            out[h * (MM_HEAD + 2) + MM_HEAD + 1] = sum;
        }
    }
}

// DS4_MINIMAX_ATTN_SIMT=1: every width takes the SIMT kernel (A/B).
static bool mm_attn_simt(void) {
    static int simt = -1;
    if (simt < 0) { simt = getenv("DS4_MINIMAX_ATTN_SIMT") != nullptr; }
    return simt != 0;
}

// Row blockIdx.y's partials: per head the max of the splits' maxima, then
// the rescaled sums and outputs, rounded once.
__global__ void mm_attn_combine(float *heads, const float *partial) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t head = blockIdx.x, d = threadIdx.x, kv = head / MM_GROUP, h = head % MM_GROUP;
    partial += (uint64_t)blockIdx.y * MM_SPLITS * MM_GROUP * (MM_HEAD + 2);
    heads += (uint64_t)blockIdx.y * MM_HEADS * MM_HEAD;
    float m = -INFINITY;
    for (uint32_t j = 0; j < MM_TOPK; j++) {
        m = fmaxf(m, partial[((uint64_t)(kv * MM_TOPK + j) * MM_GROUP + h) * (MM_HEAD + 2) + MM_HEAD]);
    }
    float l = 0.0f, o = 0.0f;
    for (uint32_t j = 0; j < MM_TOPK; j++) {
        const float *pj = partial + ((uint64_t)(kv * MM_TOPK + j) * MM_GROUP + h) * (MM_HEAD + 2);
        if (pj[MM_HEAD] == -INFINITY) { continue; }
        const float c = expf(pj[MM_HEAD] - m);
        l = fmaf(pj[MM_HEAD + 1], c, l);
        o = fmaf(pj[d], c, o);
    }
    heads[(uint64_t)head * MM_HEAD + d] = glmvq_round(l > 0.0f ? o / l : 0.0f);
}

extern "C" int ds4_gpu_minimax_attn_bf16(uint32_t rows) { return rows > DS4_MINIMAX_SPLIT_ROWS && !mm_attn_simt(); }

extern "C" int ds4_gpu_minimax_attn(ds4_gpu_tensor *heads, const ds4_gpu_tensor *q, const ds4_gpu_tensor *pages,
        const ds4_gpu_tensor *table, const ds4_gpu_tensor *blocks, ds4_gpu_tensor *partials, uint32_t rows,
        uint32_t pos, uint32_t bf16) {
    const uint64_t count = (uint64_t)rows * MM_HEADS * MM_HEAD;
    if (!rows || !mm_pages_ok(pages, table, pos + rows) || !gvq_floats(heads, count) || !gvq_floats(q, count) ||
        !blocks || blocks->bytes < (uint64_t)rows * MM_KV_HEADS * MM_TOPK * sizeof(int32_t)) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    const int32_t *slots = table ? (const int32_t *)table->ptr : nullptr;
    if (bf16 && !ds4_gpu_minimax_attn_bf16(rows)) { return 0; }
    if (rows > DS4_MINIMAX_SPLIT_ROWS && !mm_attn_simt()) {
        const dim3 grid((rows + MM_AT_ROWS - 1) / MM_AT_ROWS, MM_KV_HEADS);
        const auto launch = [&](auto *out, const auto *qrows) {
            using OUT = std::remove_pointer_t<decltype(out)>;
            using QT = std::remove_const_t<std::remove_pointer_t<decltype(qrows)>>;
            static const cudaError_t ready = cudaFuncSetAttribute(mm_attn_mma_kernel<OUT, QT>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, MM_AT_SMEM);
            if (!cuda_ok(ready, "MiniMax MMA attention shared memory")) { return; }
            mm_attn_mma_kernel<<<grid, MM_AT_ROWS * 32, MM_AT_SMEM, stream>>>(out, qrows, (const uint8_t *)pages->ptr,
                slots, (const int32_t *)blocks->ptr, pos, rows);
        };
        const bool hb = bf16 & DS4_MINIMAX_BF16_HEADS, qb = bf16 & DS4_MINIMAX_BF16_Q;
        if (hb && qb) { launch((__nv_bfloat16 *)heads->ptr, (const __nv_bfloat16 *)q->ptr); }
        else if (hb) { launch((__nv_bfloat16 *)heads->ptr, (const float *)q->ptr); }
        else if (qb) { launch((float *)heads->ptr, (const __nv_bfloat16 *)q->ptr); }
        else { launch((float *)heads->ptr, (const float *)q->ptr); }
        return cuda_ok(cudaGetLastError(), "MiniMax MMA attention");
    }
    if (rows > DS4_MINIMAX_SPLIT_ROWS || (rows > 1 && mm_attn_simt())) {
        mm_attn_kernel<true><<<dim3(rows, MM_KV_HEADS), MM_ATTN_THREADS, 0, stream>>>((float *)heads->ptr, nullptr,
            (const float *)q->ptr, (const uint8_t *)pages->ptr, slots, (const int32_t *)blocks->ptr, pos, MM_TOPK);
        return cuda_ok(cudaGetLastError(), "MiniMax attention");
    }
    if (!gvq_floats(partials, (uint64_t)rows * MM_SPLITS * MM_GROUP * (MM_HEAD + 2))) { return 0; }
    if (!mm_attn_simt()) {
        static bool ready = false;
        if (!ready) {
            // Two 34 KiB tiles per SM need the large shared split.
            if (!cuda_ok(cudaFuncSetAttribute(mm_attn_split_mma_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                    100), "MiniMax split MMA attention shared memory")) { return 0; }
            ready = true;
        }
        glmvq_launch_kernel(mm_mode(rows), mm_attn_split_mma_kernel, dim3(MM_SPLITS, rows), MM_AT_SPLIT_THREADS,
            MM_AT_SPLIT_SMEM, stream,
            (float *)partials->ptr, (const float *)q->ptr, (const uint8_t *)pages->ptr, slots,
            (const int32_t *)blocks->ptr, pos);
    } else {
        mm_attn_kernel<false><<<MM_SPLITS, MM_ATTN_THREADS, 0, stream>>>(nullptr, (float *)partials->ptr,
            (const float *)q->ptr, (const uint8_t *)pages->ptr, slots, (const int32_t *)blocks->ptr, pos, 1u);
    }
    glmvq_launch_kernel(mm_mode(rows), mm_attn_combine, dim3(MM_HEADS, rows), MM_HEAD, 0, stream, (float *)heads->ptr,
        (const float *)partials->ptr);
    return cuda_ok(cudaGetLastError(), "MiniMax decode attention");
}

/* ------------------------------------------------------------------ DSpark */

enum { MM_DRAFT_TAPS = 5, MM_DRAFT_KV = MM_KV_HEADS * MM_HEAD, MM_DRAFT_WARPS = 8, MM_MARKOV_RANK = 256 };

// taps[r][k] = the residual entering tap layer k, stored as BF16 (exact: the
// residual stream holds BF16-rounded values).
template <typename T>
__global__ void mm_draft_tap_kernel(__nv_bfloat16 *taps, const T *cur, uint32_t k, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    taps[(i / MM_HIDDEN * MM_DRAFT_TAPS + k) * MM_HIDDEN + i % MM_HIDDEN] =
        __float2bfloat16_rn(glmvq_value(cur[i]));
}

extern "C" int ds4_gpu_minimax_draft_tap(ds4_gpu_tensor *taps, const ds4_gpu_tensor *cur, uint32_t k, uint32_t rows,
        ds4_gvq_rows format) {
    const uint64_t count = (uint64_t)rows * MM_HIDDEN;
    if (!taps || k >= MM_DRAFT_TAPS || taps->bytes < count * MM_DRAFT_TAPS * sizeof(__nv_bfloat16) ||
        !gvq_floats(cur, format == DS4_GVQ_ROWS_BF16 ? count / 2 : count)) { return 0; }
    auto launch = [&](auto *values) {
        mm_draft_tap_kernel<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>(
            (__nv_bfloat16 *)taps->ptr, values, k, count);
    };
    if (format == DS4_GVQ_ROWS_BF16) { launch((const __nv_bfloat16 *)cur->ptr); }
    else { launch((const float *)cur->ptr); }
    return cuda_ok(cudaGetLastError(), "MiniMax DSpark tap");
}

// out = F32 rows r0.. of the taps (the fc input, 30,720 per row).
__global__ void mm_draft_widen_kernel(float *out, const __nv_bfloat16 *taps, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = __bfloat162float(taps[i]); }
}

extern "C" int ds4_gpu_minimax_draft_widen(ds4_gpu_tensor *out, const ds4_gpu_tensor *taps, uint32_t r0, uint32_t rows) {
    const uint64_t row = (uint64_t)MM_DRAFT_TAPS * MM_HIDDEN, count = (uint64_t)rows * row;
    if (!taps || taps->bytes < ((uint64_t)r0 + rows) * row * sizeof(__nv_bfloat16) || !gvq_floats(out, count)) { return 0; }
    mm_draft_widen_kernel<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>((float *)out->ptr,
        (const __nv_bfloat16 *)taps->ptr + (uint64_t)r0 * row, count);
    return cuda_ok(cudaGetLastError(), "MiniMax DSpark widen");
}

// K and V rows (F32, BF16-rounded) into the ring at (pos + r) % cap.
__global__ void mm_draft_ring_kernel(__nv_bfloat16 *ring_k, __nv_bfloat16 *ring_v, const float *k, const float *v,
        uint32_t pos, uint32_t cap) {
    const uint32_t r = blockIdx.x;
    const uint64_t slot = (uint64_t)((pos + r) % cap) * MM_DRAFT_KV;
    for (uint32_t c = threadIdx.x; c < MM_DRAFT_KV; c += blockDim.x) {
        ring_k[slot + c] = __float2bfloat16_rn(k[(uint64_t)r * MM_DRAFT_KV + c]);
        ring_v[slot + c] = __float2bfloat16_rn(v[(uint64_t)r * MM_DRAFT_KV + c]);
    }
}

extern "C" int ds4_gpu_minimax_draft_ring(ds4_gpu_tensor *ring_k, ds4_gpu_tensor *ring_v, const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v, uint32_t rows, uint32_t pos, uint32_t cap) {
    const uint64_t count = (uint64_t)rows * MM_DRAFT_KV, ring = (uint64_t)cap * MM_DRAFT_KV * sizeof(__nv_bfloat16);
    if (!rows || !cap || !ring_k || !ring_v || ring_k->bytes < ring || ring_v->bytes < ring || !gvq_floats(k, count) ||
        !gvq_floats(v, count)) { return 0; }
    mm_draft_ring_kernel<<<rows, 256, 0, ds4_current_stream()>>>((__nv_bfloat16 *)ring_k->ptr,
        (__nv_bfloat16 *)ring_v->ptr, (const float *)k->ptr, (const float *)v->ptr, pos, cap);
    return cuda_ok(cudaGetLastError(), "MiniMax DSpark ring");
}

/* Draft attention of block row i and head h over ring positions [lo, hi):
 * the context window and every block row (block rows see each other).
 * Eight warps stride the keys, lane l owns dims 4l..4l+3; F32 online
 * softmax with scale 128^-1/2, the warps merged in shared memory, output
 * rounded once. */
/* DSpark attention on BF16 tensor cores: CTA (KV head, 128-key chunk c of
 * [lo, hi)) copies the chunk's K and V rows from the ring into shared
 * memory once; warp r scores block row r's 16 query heads against them
 * (S = Q K^T in F32, keys at or past hi masked, F32 online softmax,
 * O += P V with P in BF16) and leaves an (O, max, sum) partial per (row,
 * head, chunk) for mm_draft_attn_combine. The block attends to itself and
 * to its context window, so nothing else is masked. A CTA per (row, head)
 * re-read the window's K/V for every one of 448 query vectors (480 us). */
enum { MM_DRAFT_CHUNK = MM_BLOCK, MM_DRAFT_ROWS = 8 };

__global__ void __launch_bounds__(MM_DRAFT_ROWS * 32, 1) mm_draft_attn_kernel(float *partial, const float *q,
        const __nv_bfloat16 *ring_k, const __nv_bfloat16 *ring_v, uint32_t rows, uint32_t lo, uint32_t hi,
        uint32_t cap, uint32_t chunks) {
    extern __shared__ __align__(16) uint8_t mm_at_smem[];
    __nv_bfloat16 *ks = (__nv_bfloat16 *)mm_at_smem, *vs = ks + MM_DRAFT_CHUNK * MM_AT_LD;
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32, g = lane >> 2, t = lane & 3;
    const uint32_t kv = blockIdx.x / chunks, c = blockIdx.x % chunks, first = lo + c * MM_DRAFT_CHUNK;
    enum { UNITS = MM_HEAD / 8 };   // 16-byte units per key row
    for (uint32_t i = threadIdx.x; i < 2u * MM_DRAFT_CHUNK * UNITS; i += blockDim.x) {
        const uint32_t side = i / (MM_DRAFT_CHUNK * UNITS), key = i / UNITS % MM_DRAFT_CHUNK, u = i % UNITS;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (first + key < hi) {
            v = *(const uint4 *)((side ? ring_v : ring_k) +
                ((uint64_t)((first + key) % cap) * MM_KV_HEADS + kv) * MM_HEAD + u * 8);
        }
        *(uint4 *)((side ? vs : ks) + key * MM_AT_LD + u * 8) = v;
    }
    __syncthreads();
    if (warp >= rows) { return; }
    uint32_t qa[8][4];
    const float *qr = q + ((uint64_t)warp * MM_HEADS + kv * MM_GROUP) * MM_HEAD;
    #pragma unroll
    for (uint32_t j = 0; j < 8; j++) {
        const uint32_t col = 16 * j + 2 * t;
        qa[j][0] = mm_bf16x2(qr[g * MM_HEAD + col], qr[g * MM_HEAD + col + 1]);
        qa[j][1] = mm_bf16x2(qr[(g + 8) * MM_HEAD + col], qr[(g + 8) * MM_HEAD + col + 1]);
        qa[j][2] = mm_bf16x2(qr[g * MM_HEAD + col + 8], qr[g * MM_HEAD + col + 9]);
        qa[j][3] = mm_bf16x2(qr[(g + 8) * MM_HEAD + col + 8], qr[(g + 8) * MM_HEAD + col + 9]);
    }
    float o[16][4];
    #pragma unroll
    for (uint32_t n = 0; n < 16; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f; }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float scale = 0.08838834764831845f;   // 128^-0.5
    #pragma unroll 1
    for (uint32_t half = 0; half < 2; half++) {
        const uint32_t k0 = half * 64;
        float s[8][4];
        #pragma unroll
        for (uint32_t n = 0; n < 8; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f; }
        #pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            #pragma unroll
            for (uint32_t n = 0; n < 8; n += 2) {
                const uint32_t mi = lane / 8;
                const uint32_t key = k0 + (n + (mi >> 1)) * 8 + lane % 8, dim = 16 * j + (mi & 1) * 8;
                uint32_t b4[4];
                mm_ldsm4(b4, ks + key * MM_AT_LD + dim, false);
                mm_mma_bf16(s[n], qa[j], b4[0], b4[1]);
                mm_mma_bf16(s[n + 1], qa[j], b4[2], b4[3]);
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
        #pragma unroll
        for (uint32_t n = 0; n < 8; n++) {
            const uint32_t key = first + k0 + n * 8 + 2 * t;
            s[n][0] = key < hi ? s[n][0] * scale : -INFINITY;
            s[n][1] = key + 1 < hi ? s[n][1] * scale : -INFINITY;
            s[n][2] = key < hi ? s[n][2] * scale : -INFINITY;
            s[n][3] = key + 1 < hi ? s[n][3] * scale : -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
            mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float c0 = n0 == -INFINITY || m0 == -INFINITY ? (n0 == -INFINITY ? 1.0f : 0.0f) : expf(m0 - n0);
        const float c1 = n1 == -INFINITY || m1 == -INFINITY ? (n1 == -INFINITY ? 1.0f : 0.0f) : expf(m1 - n1);
        float sum0 = 0.0f, sum1 = 0.0f;
        #pragma unroll
        for (uint32_t n = 0; n < 8; n++) {
            s[n][0] = s[n][0] == -INFINITY ? 0.0f : expf(s[n][0] - n0);
            s[n][1] = s[n][1] == -INFINITY ? 0.0f : expf(s[n][1] - n0);
            s[n][2] = s[n][2] == -INFINITY ? 0.0f : expf(s[n][2] - n1);
            s[n][3] = s[n][3] == -INFINITY ? 0.0f : expf(s[n][3] - n1);
            sum0 += s[n][0] + s[n][1];
            sum1 += s[n][2] + s[n][3];
        }
        l0 = fmaf(l0, c0, sum0);
        l1 = fmaf(l1, c1, sum1);
        m0 = n0;
        m1 = n1;
        #pragma unroll
        for (uint32_t n = 0; n < 16; n++) {
            o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1;
        }
        #pragma unroll
        for (uint32_t kk = 0; kk < 4; kk++) {
            const uint32_t pa[4] = {mm_bf16x2(s[2 * kk][0], s[2 * kk][1]),
                mm_bf16x2(s[2 * kk][2], s[2 * kk][3]), mm_bf16x2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                mm_bf16x2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
            #pragma unroll
            for (uint32_t n = 0; n < 16; n += 2) {
                const uint32_t mi = lane / 8;
                const uint32_t key = k0 + 16 * kk + (mi & 1) * 8 + lane % 8, dim = (n + (mi >> 1)) * 8;
                uint32_t b4[4];
                mm_ldsm4(b4, vs + key * MM_AT_LD + dim, true);
                mm_mma_bf16(o[n], pa, b4[0], b4[1]);
                mm_mma_bf16(o[n + 1], pa, b4[2], b4[3]);
            }
        }
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const uint64_t head0 = (uint64_t)warp * MM_HEADS + kv * MM_GROUP + g;
    float *out0 = partial + (head0 * chunks + c) * (MM_HEAD + 2);
    float *out1 = partial + ((head0 + 8) * chunks + c) * (MM_HEAD + 2);
    #pragma unroll
    for (uint32_t n = 0; n < 16; n++) {
        const uint32_t d = n * 8 + 2 * t;
        out0[d] = o[n][0]; out0[d + 1] = o[n][1];
        out1[d] = o[n][2]; out1[d + 1] = o[n][3];
    }
    if (!t) {
        out0[MM_HEAD] = m0; out0[MM_HEAD + 1] = l0;
        out1[MM_HEAD] = m1; out1[MM_HEAD + 1] = l1;
    }
}

// Block (row, head): the max of its chunks' maxima, then the rescaled sums
// and outputs, rounded once.
__global__ void mm_draft_attn_combine(float *heads, const float *partial, uint32_t chunks) {
    const float *pp = partial + (uint64_t)blockIdx.x * chunks * (MM_HEAD + 2);
    float m = -INFINITY;
    for (uint32_t c = 0; c < chunks; c++) { m = fmaxf(m, pp[c * (MM_HEAD + 2) + MM_HEAD]); }
    float l = 0.0f, o = 0.0f;
    for (uint32_t c = 0; c < chunks; c++) {
        const float *pc = pp + c * (MM_HEAD + 2);
        if (pc[MM_HEAD] == -INFINITY) { continue; }
        const float e = expf(pc[MM_HEAD] - m);
        l = fmaf(pc[MM_HEAD + 1], e, l);
        o = fmaf(pc[threadIdx.x], e, o);
    }
    heads[(uint64_t)blockIdx.x * MM_HEAD + threadIdx.x] = glmvq_round(l > 0.0f ? o / l : 0.0f);
}

extern "C" int ds4_gpu_minimax_draft_attn(ds4_gpu_tensor *heads, ds4_gpu_tensor *partials, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *ring_k, const ds4_gpu_tensor *ring_v, uint32_t rows, uint32_t lo, uint32_t hi,
        uint32_t cap) {
    const uint64_t count = (uint64_t)rows * MM_HEADS * MM_HEAD, ring = (uint64_t)cap * MM_DRAFT_KV * sizeof(__nv_bfloat16);
    const uint32_t chunks = (hi - lo + MM_DRAFT_CHUNK - 1) / MM_DRAFT_CHUNK;
    if (!rows || rows > MM_DRAFT_ROWS || lo >= hi || hi - lo > cap || !ring_k || !ring_v || ring_k->bytes < ring ||
        ring_v->bytes < ring || !gvq_floats(heads, count) || !gvq_floats(q, count) ||
        !gvq_floats(partials, (uint64_t)rows * MM_HEADS * chunks * (MM_HEAD + 2))) { return 0; }
    static bool ready = false;
    if (!ready) {
        if (!cuda_ok(cudaFuncSetAttribute(mm_draft_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                MM_AT_SMEM), "MiniMax DSpark attention shared memory")) { return 0; }
        ready = true;
    }
    const cudaStream_t stream = ds4_current_stream();
    mm_draft_attn_kernel<<<MM_KV_HEADS * chunks, MM_DRAFT_ROWS * 32, MM_AT_SMEM, stream>>>(
        (float *)partials->ptr, (const float *)q->ptr, (const __nv_bfloat16 *)ring_k->ptr,
        (const __nv_bfloat16 *)ring_v->ptr, rows, lo, hi, cap, chunks);
    mm_draft_attn_combine<<<rows * MM_HEADS, MM_HEAD, 0, stream>>>((float *)heads->ptr,
        (const float *)partials->ptr, chunks);
    return cuda_ok(cudaGetLastError(), "MiniMax DSpark attention");
}

/* Markov head of one draft row: logits[v] += BF16(markov_w2[v] . markov_w1[prev])
 * over the vocabulary (rank 256; the source runs these auxiliary GEMMs in
 * BF16). `prev` is read on the device: the previous row's argmax, or the
 * anchor. One warp per output row, the w1 row in shared memory. */
__global__ void mm_draft_markov_kernel(float *logits, const __nv_bfloat16 *w1, const __nv_bfloat16 *w2,
        const int32_t *prev, uint32_t vocab) {
    __shared__ float row[MM_MARKOV_RANK];
    const int32_t token = *prev;
    for (uint32_t r = threadIdx.x; r < MM_MARKOV_RANK; r += blockDim.x) {
        row[r] = __bfloat162float(w1[(uint64_t)token * MM_MARKOV_RANK + r]);
    }
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31, warps = blockDim.x / 32;
    for (uint32_t v = blockIdx.x * warps + threadIdx.x / 32; v < vocab; v += gridDim.x * warps) {
        const uint4 packed = *(const uint4 *)(w2 + (uint64_t)v * MM_MARKOV_RANK + lane * 8);
        const __nv_bfloat162 *pair = (const __nv_bfloat162 *)&packed;
        float dot = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
            const float2 w = __bfloat1622float2(pair[i]);
            dot = fmaf(w.x, row[lane * 8 + i * 2], dot);
            dot = fmaf(w.y, row[lane * 8 + i * 2 + 1], dot);
        }
        for (uint32_t s = 16; s; s >>= 1) { dot += __shfl_xor_sync(0xffffffffu, dot, s); }
        if (!lane) { logits[v] += glmvq_round(dot); }
    }
}

extern "C" int ds4_gpu_minimax_draft_markov(ds4_gvq_ctx *ctx, ds4_gpu_tensor *logits, uint32_t w1, uint32_t w2,
        const ds4_gpu_tensor *prev, uint32_t vocab) {
    const auto *a = (const __nv_bfloat16 *)gvq_weight(ctx, w1, GVQ_TYPE_BF16, (uint64_t)vocab * MM_MARKOV_RANK);
    const auto *b = (const __nv_bfloat16 *)gvq_weight(ctx, w2, GVQ_TYPE_BF16, (uint64_t)vocab * MM_MARKOV_RANK);
    if (!a || !b || !prev || prev->bytes < sizeof(int32_t) || !gvq_floats(logits, vocab)) { return 0; }
    mm_draft_markov_kernel<<<512, 256, 0, ds4_current_stream()>>>((float *)logits->ptr, a, b,
        (const int32_t *)prev->ptr, vocab);
    return cuda_ok(cudaGetLastError(), "MiniMax DSpark Markov head");
}

/* ------------------------------------------------------------- pinned pool */

// Pinned host memory the GPU reads and writes in place (mapped
// cudaHostAlloc): the SSD tier's pool and slot table, which O_DIRECT reads
// fill and the host rewrites without a copy. The tensor wraps the device
// alias; ds4_gpu_minimax_host_free releases both.
extern "C" void *ds4_gpu_minimax_host_alloc(uint64_t bytes, ds4_gpu_tensor **device) {
    void *host = nullptr, *alias = nullptr;
    *device = nullptr;
    if (!bytes || !cuda_ok(cudaHostAlloc(&host, (size_t)bytes, cudaHostAllocMapped | cudaHostAllocPortable),
            "MiniMax pinned pool")) { return nullptr; }
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t || !cuda_ok(cudaHostGetDevicePointer(&alias, host, 0), "MiniMax pinned alias")) {
        free(t);
        cudaFreeHost(host);
        return nullptr;
    }
    t->ptr = alias;
    t->bytes = bytes;
    t->memc = cuda_mem_scope_class();
    cuda_mem_note_alloc(t->memc, DS4_MEMD_PINNED_HOST, bytes, bytes);
    *device = t;
    return host;
}

// CTA (b, part): part of block b's 73,728 bytes.
__global__ void mm_kv_stage_kernel(uint4 *stage, const uint4 *pool, const int32_t *table) {
    constexpr uint32_t WORDS = DS4_MINIMAX_KV_BLOCK / sizeof(uint4);
    const int32_t slot = table[blockIdx.x];
    if (slot < 0) { return; }
    const uint4 *src = pool + (uint64_t)slot * WORDS;
    uint4 *dst = stage + (uint64_t)blockIdx.x * WORDS;
    for (uint32_t i = blockIdx.y * blockDim.x + threadIdx.x; i < WORDS; i += gridDim.y * blockDim.x) { dst[i] = src[i]; }
}

extern "C" int ds4_gpu_minimax_kv_stage(ds4_gpu_tensor *stage, const ds4_gpu_tensor *pool, const ds4_gpu_tensor *table,
        uint32_t blocks) {
    if (!stage || !pool || !table || !blocks || stage->bytes < (uint64_t)blocks * DS4_MINIMAX_KV_BLOCK ||
        table->bytes < (uint64_t)blocks * sizeof(int32_t)) { return 0; }
    mm_kv_stage_kernel<<<dim3(blocks, 4), 256, 0, ds4_current_stream()>>>((uint4 *)stage->ptr, (const uint4 *)pool->ptr,
        (const int32_t *)table->ptr);
    return cuda_ok(cudaGetLastError(), "MiniMax K/V stage");
}

extern "C" void ds4_gpu_minimax_host_free(void *host, ds4_gpu_tensor *device) {
    if (device) {
        cuda_mem_note_free(device->memc, DS4_MEMD_PINNED_HOST, device->bytes, device->bytes);
        free(device);
    }
    if (host) { cudaFreeHost(host); }
}

// One event after each attention launch: the SSD tier's streamed reads
// reuse the previous layer's pool slots only once its attention has run.
static cudaEvent_t g_mm_mark;

extern "C" int ds4_gpu_minimax_mark(void) {
    if (!g_mm_mark && !cuda_ok(cudaEventCreateWithFlags(&g_mm_mark, cudaEventDisableTiming | cudaEventBlockingSync),
            "MiniMax mark")) { return 0; }
    return cuda_ok(cudaEventRecord(g_mm_mark, ds4_current_stream()), "MiniMax mark");
}

extern "C" int ds4_gpu_minimax_mark_wait(void) {
    return !g_mm_mark || cuda_ok(cudaEventSynchronize(g_mm_mark), "MiniMax mark wait");
}

/* ----------------------------------------------------------- vision tower */

extern "C" int ds4_gpu_minimax_vit_clear(void) {
    if (ds4_capture_active() || !cuda_ok(cudaDeviceSynchronize(), "MiniMax vision completion")) { return 0; }
    v41_scratch *slots[] = {&g_vis_qkv16, &g_vis_s, &g_vis_p, &g_v41_dense_xbf};
    bool ok = true, released = false;
    for (v41_scratch *s : slots) {
        if (!s->p) { s->cap = 0; continue; }
        if (!cuda_ok(cudaFree(s->p), "MiniMax vision scratch free")) { ok = false; continue; }
        *s = {};
        released = true;
    }
    // Captured ds41 graphs also consume the shared BF16 slot; force recapture.
    if (released) { g_v41_scratch_gen++; }
    return ok;
}

/* The CLIP-style tower of the vision part (ds4_minimax_vision.inc): F32
 * compute, values on the BF16 grid at the module boundaries the reference
 * rounds at (torch BF16 modules). Linears, GELU, attention and row
 * placement reuse DeepSeek V4.1's vision primitives; these three are
 * MiniMax's own. */
enum { MM_VIT_DIM = DS4_MINIMAX_VIT_DIM, MM_VIT_HEADS = DS4_MINIMAX_VIT_HEADS, MM_VIT_HEAD = DS4_MINIMAX_VIT_HEAD,
       MM_VIT_ROT = 78, MM_VIT_AXIS = 13, MM_VIT_PATCH = DS4_MINIMAX_VIT_PATCH };

// LayerNorm over rows of `dim` (eps 1e-5, F32 weight and bias): two-pass
// mean and variance in F32, one BF16 rounding of the output.
__global__ void mm_vit_ln_kernel(float *out, const float *x, const float *w, const float *b, uint32_t dim, float eps) {
    __shared__ float red[32];
    const float *row = x + (uint64_t)blockIdx.x * dim;
    auto block_sum = [&](float v) {
        for (uint32_t s = 16; s; s >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, s); }
        if ((threadIdx.x & 31) == 0) { red[threadIdx.x / 32] = v; }
        __syncthreads();
        v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.0f;
        if (threadIdx.x < 32) {
            for (uint32_t s = 16; s; s >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, s); }
            if (threadIdx.x == 0) { red[0] = v; }
        }
        __syncthreads();
        const float total = red[0];
        __syncthreads();
        return total;
    };
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) { sum += row[i]; }
    const float mean = block_sum(sum) / (float)dim;
    float sq = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) { const float d = row[i] - mean; sq = fmaf(d, d, sq); }
    const float inv = rsqrtf(block_sum(sq) / (float)dim + eps);
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        out[(uint64_t)blockIdx.x * dim + i] = glmvq_round(fmaf((row[i] - mean) * inv, w[i], b[i]));
    }
}

extern "C" int ds4_gpu_minimax_vit_ln(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *w,
        const ds4_gpu_tensor *b, uint32_t rows, uint32_t dim, float eps) {
    if (!rows || !dim || !gvq_floats(out, (uint64_t)rows * dim) || !gvq_floats(x, (uint64_t)rows * dim) ||
        !gvq_floats(w, dim) || !gvq_floats(b, dim)) { return 0; }
    mm_vit_ln_kernel<<<rows, 256, 0, ds4_current_stream()>>>((float *)out->ptr, (const float *)x->ptr,
        (const float *)w->ptr, (const float *)b->ptr, dim, eps);
    return cuda_ok(cudaGetLastError(), "MiniMax vision layer norm");
}

/* 3-D RoPE of q and k in place, rows of packed [q | k | v] (3 x 16 heads x
 * 80): a patch's (t, h, w) give 13 frequencies each, inv_freq_j =
 * 10000^(-2j/26), duplicated to 78 dims with NeoX rotate_half (pairs i, i +
 * 39); dims 78-79 pass. Patches run in the processor's order: frame-major,
 * then 2x2 blocks row-major, each block's four patches row-major. `t0` is
 * the frame index where the row's segment starts (t restarts per segment).
 * cos/sin in F32 as vLLM computes them; the result is rounded to BF16. */
__global__ void mm_vit_rope_kernel(float *qkv, uint32_t rows, uint32_t gh, uint32_t gw, uint32_t seg_frames) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t half = MM_VIT_ROT / 2;   // 39
    if (i >= (uint64_t)rows * 2 * MM_VIT_HEADS * half) { return; }
    const uint32_t j = (uint32_t)(i % half), head = (uint32_t)(i / half % (2 * MM_VIT_HEADS));
    const uint32_t r = (uint32_t)(i / ((uint64_t)half * 2 * MM_VIT_HEADS));
    const uint32_t frame = r / (gh * gw), cell = r % (gh * gw), block = cell / 4, in = cell % 4;
    const uint32_t bw = gw / 2, hh = block / bw * 2 + in / 2, ww = block % bw * 2 + in % 2;
    const uint32_t axis = j / MM_VIT_AXIS, f = j % MM_VIT_AXIS;
    const float coord = axis == 0 ? (float)(frame % seg_frames) : axis == 1 ? (float)hh : (float)ww;
    // F32 inv_freq and angle as torch forms them; sin/cos in double (fast-math
    // sincosf loses accuracy past pi, and angles reach ~50 rad).
    const float inv = __frcp_rn(powf(10000.0f, (float)(2 * f) / (float)(2 * MM_VIT_AXIS)));
    double sd, cd;
    sincos((double)(coord * inv), &sd, &cd);
    const float s = (float)sd, c = (float)cd;
    // head < 16: q heads; 16..31: k heads (k starts 1280 floats into the row).
    float *x = qkv + (uint64_t)r * 3 * MM_VIT_DIM + (uint64_t)head * MM_VIT_HEAD;
    const float a = x[j], bv = x[j + half];
    x[j] = glmvq_round(a * c - bv * s);
    x[j + half] = glmvq_round(bv * c + a * s);
}

extern "C" int ds4_gpu_minimax_vit_rope(ds4_gpu_tensor *qkv, uint32_t rows, uint32_t gh, uint32_t gw,
        uint32_t seg_frames) {
    if (!rows || !gh || !gw || gh % 2 || gw % 2 || !seg_frames || !gvq_floats(qkv, (uint64_t)rows * 3 * MM_VIT_DIM)) {
        return 0;
    }
    const uint64_t n = (uint64_t)rows * 2 * MM_VIT_HEADS * (MM_VIT_ROT / 2);
    mm_vit_rope_kernel<<<(uint32_t)((n + 255) / 256), 256, 0, ds4_current_stream()>>>((float *)qkv->ptr, rows, gh,
        gw, seg_frames);
    return cuda_ok(cudaGetLastError(), "MiniMax vision rope");
}

/* Patch embedding: Conv3d with kernel = stride (2, 14, 14) and no bias is a
 * F32 matmul of the flattened (C, T, P, P) patches with the F32 weight
 * [1280][1176], then one BF16 rounding (the reference casts after). */
extern "C" int ds4_gpu_minimax_vit_patch(ds4_gpu_tensor *out, const ds4_gpu_tensor *w, const ds4_gpu_tensor *pixels,
        uint32_t rows) {
    if (!rows || !g_cublas_ready || !gvq_floats(out, (uint64_t)rows * MM_VIT_DIM) ||
        !gvq_floats(w, (uint64_t)MM_VIT_DIM * MM_VIT_PATCH) || !gvq_floats(pixels, (uint64_t)rows * MM_VIT_PATCH)) {
        return 0;
    }
    (void)cublasSetStream(g_cublas, ds4_current_stream());
    cuda_cublas_ws_prep(ds4_current_stream());
    const float alpha = 1.0f, beta = 0.0f;
    if (!cublas_ok(cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, MM_VIT_DIM, (int)rows, MM_VIT_PATCH, &alpha,
            (const float *)w->ptr, MM_VIT_PATCH, (const float *)pixels->ptr, MM_VIT_PATCH, &beta, (float *)out->ptr,
            MM_VIT_DIM), "MiniMax vision patch embedding")) { return 0; }
    return ds4_gpu_v41_round_bf16_tensor(out, (uint64_t)rows * MM_VIT_DIM);
}

/* ------------------------------------------------------------- head, embed */

__global__ void mm_logits_kernel(float *logits, uint32_t tokens) {
    const uint32_t i = tokens + blockIdx.y * blockDim.x + threadIdx.x;
    if (i < DS4_MINIMAX_VOCAB) { logits[(uint64_t)blockIdx.x * DS4_MINIMAX_VOCAB + i] = -1e30f; }
}

extern "C" int ds4_gpu_minimax_logits(ds4_gpu_tensor *logits, uint32_t rows, uint32_t tokens) {
    if (!rows || tokens > DS4_MINIMAX_VOCAB || !gvq_floats(logits, (uint64_t)rows * DS4_MINIMAX_VOCAB)) { return 0; }
    if (tokens == DS4_MINIMAX_VOCAB) { return 1; }
    const uint32_t pad = DS4_MINIMAX_VOCAB - tokens;
    mm_logits_kernel<<<dim3(rows, (pad + 255) / 256), 256, 0, ds4_current_stream()>>>((float *)logits->ptr, tokens);
    return cuda_ok(cudaGetLastError(), "MiniMax logits");
}

template <typename T>
__global__ void mm_embed_kernel(T *cur, const __nv_bfloat16 *rows, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { cur[i] = glmvq_cast<T>(__bfloat162float(rows[i])); }
}

extern "C" int ds4_gpu_minimax_embed(ds4_gpu_tensor *cur, const ds4_gpu_tensor *rows_bf16, uint32_t rows,
        ds4_gvq_rows format) {
    const uint64_t count = (uint64_t)rows * MM_HIDDEN;
    if (!rows || !gvq_floats(cur, format == DS4_GVQ_ROWS_BF16 ? count / 2 : count) || !rows_bf16 ||
        rows_bf16->bytes < count * sizeof(__nv_bfloat16)) {
        return 0;
    }
    auto launch = [&](auto *values) {
        mm_embed_kernel<<<(uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0,
            ds4_current_stream()>>>(values, (const __nv_bfloat16 *)rows_bf16->ptr, count);
    };
    if (format == DS4_GVQ_ROWS_BF16) { launch((__nv_bfloat16 *)cur->ptr); }
    else { launch((float *)cur->ptr); }
    return cuda_ok(cudaGetLastError(), "MiniMax embedding");
}
