/* A.X-K2 VQ device runtime; included once by ds4_cuda.cu after the GVQ
 * adapter, the Ling-3.1 runtime (whose FP8 latent rows and flash tile it
 * extends with selected keys) and the Beam runtime (whose router kernel it
 * reuses).
 *
 * Routed experts run GLM's GVQ kernels on GVQ1 version 2 (9/10-bit,
 * byte-major) in A.X-K2's form: 7168 -> 2048 -> 7168 rows, gate/up/down
 * books of their own, silu(g) and silu(g) * u each rounded to BF16
 * (transformers' BF16 ops), the weighted outputs summed in F32 and rounded
 * once before the shared expert's BF16 add (the reference's FP32 sum):
 *
 *   rows == 1     decode: device routing, persistent CTAs keep the books
 *   rows >= 2     prefill: host grouping, FP8 MMA on the E4M3 codewords
 *
 * The attention is MLA over FP8 latent rows (Ling-3.1's 641-byte rows) with
 * the DSA indexer's top-2,048 keys per query once a row sees more than
 * 2,048 keys; kv_b stays Q4_K, so the absorptions read it directly at
 * decode widths and from a per-call BF16 restore at prefill widths. */
#include "ds4_axk2.h"
#include "cuda/axk2_topk.cuh"
#include "cuda/axk2_flash.cuh"
#include "cuda/axk2_q8.cuh"

enum { AX_EXPERTS = DS4_AXK2_EXPERTS, AX_USED = DS4_AXK2_USED, AX_HIDDEN = DS4_AXK2_HIDDEN, AX_FFN = DS4_AXK2_FFN,
       AX_TABLE = 3 * AX_EXPERTS, AX_LAYERS = DS4_AXK2_MOE_LAYERS, AX_MMA_ROWS = 256, AX_MMA_TOKENS = 64,
       AX_ALIGN = 256, AX_HEADS = DS4_AXK2_HEADS, AX_LATENT = DS4_AXK2_LATENT, AX_NOPE = DS4_AXK2_NOPE,
       AX_ROPE = DS4_AXK2_ROPE, AX_KEY = DS4_AXK2_KEY, AX_VALUE = DS4_AXK2_VALUE, AX_KV_ROW = DS4_AXK2_KV_ROW,
       AX_IDX_HEADS = DS4_AXK2_INDEX_HEADS, AX_IDX_DIM = DS4_AXK2_INDEX_DIM, AX_TOPK = DS4_AXK2_TOPK,
       AX_RANK = DS4_AXK2_GATE_RANK, AX_FREQS = AX_ROPE / 2, AX_KV_B_ROWS = AX_HEADS * (AX_NOPE + AX_VALUE) };
static_assert((uint32_t)AX_USED == (uint32_t)GLMVQ_DEC_USED, "GVQ decode geometry");
static_assert((uint32_t)AX_HEADS == (uint32_t)GLMVQ_FA_HEADS && (uint32_t)AX_LATENT == (uint32_t)GLMVQ_FA_LATENT &&
    (uint32_t)AX_KEY == (uint32_t)L31_FA_KEY, "Ling-3.1's flash tile geometry");
static const float AX_EPS = 1e-6f;
using ax_form = glmvq_moe_form<AX_FFN, GLMVQ_CLAMP_GATE, GLMVQ_REDUCE_F32, AX_HIDDEN>;

static uint64_t ax_align(uint64_t bytes) { return (bytes + AX_ALIGN - 1) / AX_ALIGN * AX_ALIGN; }

// Decode scratch, then the device tables (one 3 x 256 row per layer).
struct ax_decode_layout { uint64_t items, mid, partial, tables, bytes; };

static ax_decode_layout ax_decode_plan() {
    ax_decode_layout l = {};
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += ax_align(bytes); return here; };
    l.items = take(sizeof(glmvq_decode_item) * AX_USED);
    l.mid = take((uint64_t)AX_USED * AX_FFN * sizeof(__nv_bfloat16));
    l.partial = take((uint64_t)AX_USED * AX_HIDDEN * sizeof(float));
    l.tables = take((uint64_t)AX_LAYERS * AX_TABLE * sizeof(uint32_t));
    l.bytes = at;
    return l;
}

// Prefill: E4M3 input rows and 1x128 scales, gate rows (BF16), the up
// epilogue's E4M3 SwiGLU rows and scales, down rows (BF16), worklists.
struct ax_mma_layout { uint64_t x8, xs, gate, mid8, ms, down, work, order, bytes; };

static ax_mma_layout ax_mma_plan(uint32_t rows) {
    ax_mma_layout l = {};
    const uint64_t pairs = (uint64_t)rows * AX_USED;
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += ax_align(bytes); return here; };
    l.x8 = take((uint64_t)rows * AX_HIDDEN);
    l.xs = take((uint64_t)rows * (AX_HIDDEN / GLMVQ_MMA8_K) * sizeof(float));
    l.gate = take(pairs * AX_FFN * sizeof(__nv_bfloat16));
    l.mid8 = take(pairs * AX_FFN);
    l.ms = take(pairs * (AX_FFN / GLMVQ_MMA8_K) * sizeof(float));
    l.down = take(pairs * AX_HIDDEN * sizeof(__nv_bfloat16));
    l.work = take(pairs * sizeof(glmvq_mma_item));
    l.order = take(pairs * sizeof(uint32_t));
    l.bytes = at;
    return l;
}

struct ds4_axk2_moe {
    ds4_gvq_ctx *gvq = nullptr;     // borrowed: stream, routes, grouped, pairs
    ds4_gpu_tensor *decode = nullptr, *mma = nullptr;
    uint32_t rows = 0, decode_blocks = 0;
    int mma8_blocks[2][3] = {};     // [9-bit, 10-bit][gate, up, down]
    // DS4_AXK2_GAIN_DOT=0: gain x codeword rounded per weight (the restore's
    // arithmetic) instead of one multiply after the dot.
    bool gain_dot = true;
    std::vector<uint32_t> keys;     // table slot per layer, keyed by expert 0's gate
    std::vector<glmvq_mma_item> items;
    std::vector<uint32_t> order;
    std::vector<uint8_t> token_used;
};

static bool ax_env_off(const char *name) {
    const char *value = getenv(name);
    return value && !strcmp(value, "0");
}

extern "C" uint64_t ds4_gpu_axk2_moe_bytes(uint32_t rows) {
    return ax_decode_plan().bytes + (rows > 1 ? ax_mma_plan(rows).bytes : 0u);
}

extern "C" void ds4_gpu_axk2_moe_free(ds4_axk2_moe *moe) {
    if (!moe) { return; }
    ds4_gpu_tensor_free(moe->decode);
    ds4_gpu_tensor_free(moe->mma);
    delete moe;
}

// Shared memory of the v2 decode kernels: gate/up stage two books and the
// BF16 input row, down one book and the eight experts' mid rows.
template <uint32_t BITS>
static constexpr uint32_t ax_gu_bytes() { return 2 * glmvq_decode_books<BITS, false>() + AX_HIDDEN * 2; }
template <uint32_t BITS>
static constexpr uint32_t ax_down_bytes() { return glmvq_decode_books<BITS, false>() + AX_USED * AX_FFN * 2; }

template <uint32_t BITS, glmvq_decode_gain GAIN>
static auto ax_gu_kernel() {
    static const bool compact = !ax_env_off("DS4_AXK2_GU_COMPACT");
    return compact
        ? glmvq_decode_gu<BITS, 32, GLMVQ_DEC_NORMAL, GAIN, AX_FFN, GLMVQ_CLAMP_GATE, false, AX_HIDDEN, GLMVQ_DEC_COMPACT>
        : glmvq_decode_gu<BITS, 32, GLMVQ_DEC_NORMAL, GAIN, AX_FFN, GLMVQ_CLAMP_GATE, false, AX_HIDDEN>;
}

template <uint32_t BITS, glmvq_decode_gain GAIN>
static cudaError_t ax_decode_attrs() {
    cudaError_t err = cudaFuncSetAttribute(ax_gu_kernel<BITS, GAIN>(),
        cudaFuncAttributeMaxDynamicSharedMemorySize, ax_gu_bytes<BITS>());
    if (err == cudaSuccess) {
        err = cudaFuncSetAttribute(glmvq_decode_down<BITS, 32, GAIN, AX_FFN, false, AX_HIDDEN, AX_USED>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, ax_down_bytes<BITS>());
    }
    return err;
}

template <uint32_t BITS>
static cudaError_t ax_prepare(ds4_axk2_moe *moe) {
    cudaError_t err = ax_decode_attrs<BITS, GLMVQ_GAIN_DOT>();
    if (err == cudaSuccess) { err = ax_decode_attrs<BITS, GLMVQ_GAIN_WEIGHT>(); }
    if (err != cudaSuccess) { return err; }
    int *mma = moe->mma8_blocks[BITS - 9];
    err = glmvq_mma8_prepare<AX_MMA_ROWS, AX_MMA_TOKENS, 1, GLMVQ_MMA_GATE, BITS, GLMVQ_CLAMP_GATE>(&mma[0]);
    if (err == cudaSuccess) {
        err = glmvq_mma8_prepare<AX_MMA_ROWS, AX_MMA_TOKENS, 1, GLMVQ_MMA_UP, BITS, GLMVQ_CLAMP_GATE>(&mma[1]);
    }
    if (err == cudaSuccess) {
        err = glmvq_mma8_prepare<AX_MMA_ROWS, AX_MMA_TOKENS, 1, GLMVQ_MMA_DOWN, BITS, GLMVQ_CLAMP_GATE>(&mma[2]);
    }
    return err;
}

extern "C" ds4_axk2_moe *ds4_gpu_axk2_moe_create(ds4_gvq_ctx *gvq, uint32_t rows) {
    if (!gvq || !rows) { return nullptr; }
    ds4_axk2_moe *moe = new (std::nothrow) ds4_axk2_moe;
    if (!moe) { return nullptr; }
    moe->gvq = gvq;
    moe->rows = rows;
    moe->gain_dot = !ax_env_off("DS4_AXK2_GAIN_DOT");
    int device = 0, sms = 0;
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess) {
        moe->decode_blocks = (uint32_t)sms;
    }
    const cudaError_t err9 = ax_prepare<9>(moe), err10 = ax_prepare<10>(moe);
    if (err9 != cudaSuccess || err10 != cudaSuccess || !moe->decode_blocks) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: A.X-K2 expert kernels unavailable (%s)\n",
            cudaGetErrorString(err9 != cudaSuccess ? err9 : err10));
        ds4_gpu_axk2_moe_free(moe);
        return nullptr;
    }
    moe->decode = ds4_gpu_tensor_alloc(ax_decode_plan().bytes);
    moe->mma = rows > 1 ? ds4_gpu_tensor_alloc(ax_mma_plan(rows).bytes) : nullptr;
    if (!moe->decode || (rows > 1 && !moe->mma)) {
        ds4_gpu_axk2_moe_free(moe);
        return nullptr;
    }
    moe->items.reserve((uint64_t)rows * AX_USED);
    moe->order.resize((uint64_t)rows * AX_USED);
    moe->token_used.resize(rows);
    return moe;
}

// Shape, version, book and alignment the direct kernels read: a layer's 256
// gate matrices share one book, its up and down matrices one each. Rows are
// 896 + 112/224 (gate/up) or 256 + 32/64 (down) bytes.
static bool ax_matrix_fits(const glmvq_source &s, uint32_t projection, const void *book, uint32_t bits) {
    const uint32_t rows = projection == 2 ? AX_HIDDEN : AX_FFN;
    const uint32_t cols = projection == 2 ? AX_FFN : AX_HIDDEN;
    const glmvq_matrix &m = s.vq;
    return s.storage == GLMVQ_WEIGHT_VQ && s.rows == rows && s.cols == cols &&
        m.rows == rows && m.cols == cols && m.precision == GLMVQ_E4M3 &&
        m.main && m.plane && m.gain && book && m.book == book && !m.res &&
        (uintptr_t)m.main % 128 == 0 && (uintptr_t)m.plane % 128 == 0 &&
        (uintptr_t)book % 16 == 0 && (uintptr_t)m.gain % sizeof(__half) == 0 && bits >= 9;
}

// A layer's index width from its gate book: v2 books are 4 KiB (K512,
// 9-bit) or 8 KiB (K1024, 10-bit); v1 books (32/64 KiB) never reach here
// (Rust admits only GVQ1 version 2 matrices in A.X-K2 parts).
static uint32_t ax_bits(const ds4_axk2_moe *moe, const uint32_t *matrices) {
    const auto &records = moe->gvq->model->records;
    const uint64_t bytes = records[records[matrices[0]].book].bytes;
    return bytes == 1024u * GLMVQ_VECTOR ? 10u : bytes == 512u * GLMVQ_VECTOR ? 9u : 0u;
}

static const void *ax_book(const ds4_axk2_moe *moe, const uint32_t *matrices, uint32_t projection) {
    return moe->gvq->model->source[matrices[projection * AX_EXPERTS]].vq.book;
}

// The layer's device table, validated and uploaded on first use.
static const uint32_t *ax_table(ds4_axk2_moe *moe, const uint32_t *matrices, uint32_t bits) {
    const auto &sources = moe->gvq->model->source;
    const ax_decode_layout l = ax_decode_plan();
    const auto *tables = (const uint32_t *)((const uint8_t *)moe->decode->ptr + l.tables);
    for (size_t i = 0; i < moe->keys.size(); i++) {
        if (moe->keys[i] == matrices[0]) { return tables + i * AX_TABLE; }
    }
    if (moe->keys.size() == AX_LAYERS || !bits) { return nullptr; }
    for (uint32_t i = 0; i < AX_TABLE; i++) {
        const uint32_t projection = i / AX_EXPERTS;
        if (matrices[i] >= sources.size() ||
            !ax_matrix_fits(sources[matrices[i]], projection, ax_book(moe, matrices, projection), bits)) {
            fprintf(stderr, "ds4: A.X-K2 expert matrix %u misses the GVQ1 v2 layout\n", i);
            return nullptr;
        }
    }
    const size_t slot = moe->keys.size();
    if (!ds4_gpu_tensor_write(moe->decode, l.tables + slot * AX_TABLE * sizeof(uint32_t),
            matrices, AX_TABLE * sizeof(uint32_t))) { return nullptr; }
    moe->keys.push_back(matrices[0]);
    return tables + slot * AX_TABLE;
}

// N=1 routing on the GPU: thread t places expert selected[t] at its rank, so
// items run in ascending expert order with no host sync.
__global__ void ax_decode_route(glmvq_decode_item *items, const int32_t *selected,
        const uint32_t *table, const glmvq_source *sources) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x;
    const uint32_t e = min((uint32_t)selected[t], (uint32_t)AX_EXPERTS - 1u);
    uint32_t rank = 0;
    for (uint32_t j = 0; j < AX_USED; j++) { rank += (uint32_t)selected[j] < (uint32_t)selected[t]; }
    items[rank] = {sources[table[e]].vq, sources[table[AX_EXPERTS + e]].vq,
        sources[table[2 * AX_EXPERTS + e]].vq, t};
}

template <uint32_t BITS, glmvq_decode_gain GAIN>
static cudaError_t ax_decode_launch(ds4_axk2_moe *moe, float *out, const float *input, const float *weights,
        const glmvq_decode_item *items, __nv_bfloat16 *mid, float *partial, const uint32_t *matrices,
        const float *shared) {
    const cudaStream_t stream = ds4_current_stream();
    const glmvq_launch launch = moe->gvq->launch;
    glmvq_launch_kernel(launch, ax_gu_kernel<BITS, GAIN>(),
        moe->decode_blocks, 1024, ax_gu_bytes<BITS>(), stream, items, (const uint4 *)ax_book(moe, matrices, 0),
        input, mid, INFINITY, (float *)nullptr, (float *)nullptr, (const uint4 *)ax_book(moe, matrices, 1));
    glmvq_launch_kernel(launch, glmvq_decode_down<BITS, 32, GAIN, AX_FFN, false, AX_HIDDEN, AX_USED>,
        moe->decode_blocks, 1024, ax_down_bytes<BITS>(), stream, items, (const uint4 *)ax_book(moe, matrices, 2),
        (const __nv_bfloat16 *)mid, partial, (const uint4 *)nullptr);
    glmvq_launch_kernel(launch, glmvq_decode_reduce<GLMVQ_REDUCE_F32, AX_HIDDEN>, AX_HIDDEN / GLMVQ_THREADS,
        GLMVQ_THREADS, 0, stream, out, (const float *)partial, weights, items, shared);
    return cudaGetLastError();
}

template <uint32_t BITS>
static cudaError_t ax_decode_run(ds4_axk2_moe *moe, float *out, const float *input, const int32_t *selected,
        const float *weights, const uint32_t *table, const uint32_t *matrices, const float *shared) {
    const ax_decode_layout l = ax_decode_plan();
    uint8_t *base = (uint8_t *)moe->decode->ptr;
    auto *items = (glmvq_decode_item *)(base + l.items);
    auto *mid = (__nv_bfloat16 *)(base + l.mid);
    auto *partial = (float *)(base + l.partial);
    glmvq_launch_kernel(moe->gvq->launch, ax_decode_route, 1, AX_USED, 0, ds4_current_stream(), items, selected,
        table, (const glmvq_source *)moe->gvq->model->device->ptr);
    return moe->gain_dot
        ? ax_decode_launch<BITS, GLMVQ_GAIN_DOT>(moe, out, input, weights, items, mid, partial, matrices, shared)
        : ax_decode_launch<BITS, GLMVQ_GAIN_WEIGHT>(moe, out, input, weights, items, mid, partial, matrices, shared);
}

// Host grouping by expert (ascending), as the GLM prefill path builds it.
static bool ax_group(ds4_axk2_moe *moe, const ds4_gpu_tensor *selected, uint32_t rows, uint32_t *offsets) {
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * AX_USED;
    if (pairs > ctx->routes.size() ||
        !ds4_gpu_tensor_read(selected, 0, ctx->routes.data(), pairs * sizeof(int32_t))) { return false; }
    std::vector<uint32_t> counts(AX_EXPERTS, 0), cursor(AX_EXPERTS);
    for (uint64_t i = 0; i < pairs; i++) {
        const int32_t expert = ctx->routes[i];
        if (expert < 0 || expert >= AX_EXPERTS) { return false; }
        for (uint64_t j = i - i % AX_USED; j < i; j++) {
            if (ctx->routes[j] == expert) { return false; }
        }
        counts[expert]++;
    }
    offsets[0] = 0;
    for (uint32_t e = 0; e < AX_EXPERTS; e++) {
        offsets[e + 1] = offsets[e] + counts[e];
        cursor[e] = offsets[e];
    }
    for (uint64_t i = 0; i < pairs; i++) { ctx->grouped[cursor[ctx->routes[i]]++] = (uint32_t)i; }
    return ds4_gpu_tensor_write(ctx->pairs, 0, ctx->grouped.data(), pairs * sizeof(uint32_t)) != 0;
}

template <uint32_t BITS>
static int ax_mma_run(ds4_axk2_moe *moe, float *out, const float *input, const float *weights,
        const uint32_t *matrices, const uint32_t *offsets, uint32_t rows, const float *shared) {
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * AX_USED, count = (uint64_t)rows * AX_HIDDEN;
    const ax_mma_layout l = ax_mma_plan(rows);
    uint8_t *base = (uint8_t *)moe->mma->ptr;
    auto *x8 = base + l.x8, *mid8 = base + l.mid8;
    auto *xs = (float *)(base + l.xs), *ms = (float *)(base + l.ms);
    auto *gate = (__nv_bfloat16 *)(base + l.gate), *down = (__nv_bfloat16 *)(base + l.down);
    auto *work = (glmvq_mma_item *)(base + l.work);
    auto *order = (uint32_t *)(base + l.order);
    moe->items.clear();
    for (uint32_t e = 0; e < AX_EXPERTS; e++) {
        for (uint32_t start = offsets[e]; start < offsets[e + 1]; start += AX_MMA_TOKENS) {
            moe->items.push_back({{matrices[e], matrices[AX_EXPERTS + e], matrices[2 * AX_EXPERTS + e]},
                start, min((uint32_t)AX_MMA_TOKENS, offsets[e + 1] - start)});
        }
    }
    std::fill(moe->token_used.begin(), moe->token_used.begin() + rows, 0);
    for (uint32_t i = 0; i < pairs; i++) {
        const uint32_t token = ctx->grouped[i] / AX_USED;
        moe->order[token * AX_USED + moe->token_used[token]++] = i;
    }
    const cudaStream_t stream = ds4_current_stream();
    if (!cuda_ok(cudaMemcpyAsync(work, moe->items.data(), moe->items.size() * sizeof(*work),
            cudaMemcpyHostToDevice, stream), "A.X-K2 MMA work") ||
        !cuda_ok(cudaMemcpyAsync(order, moe->order.data(), pairs * sizeof(*order),
            cudaMemcpyHostToDevice, stream), "A.X-K2 MMA order")) { return 0; }
    const auto *source = (const glmvq_source *)ctx->model->device->ptr;
    const auto *grouped = (const uint32_t *)ctx->pairs->ptr;
    const uint32_t nitems = (uint32_t)moe->items.size();
    const int *blocks = moe->mma8_blocks[BITS - 9];
    // Tokens quantize once (E4M3, 1x128 scales) for their eight experts.
    const uint64_t xb = count / GLMVQ_MMA8_K;
    glmvq_mma8_quant<float><<<(uint32_t)((xb * 32 + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0, stream>>>(
        x8, xs, input, xb);
    // Each projection stages its own book.
    auto launch = [&](auto mode, const uint8_t *in, const float *scale, uint32_t rows_out, uint32_t cols_in,
            uint8_t *q8, float *q8_scale) {
        constexpr glmvq_mma_stage MODE = decltype(mode)::value;
        constexpr uint32_t shared8 = glmvq_mma8_bytes<AX_MMA_ROWS, AX_MMA_TOKENS, BITS>();
        constexpr uint32_t threads8 = glmvq_mma8_threads<AX_MMA_ROWS, 1>();
        glmvq_mma8_kernel<AX_MMA_ROWS, AX_MMA_TOKENS, 1, MODE, BITS, GLMVQ_CLAMP_GATE><<<
            min(nitems * (rows_out / AX_MMA_ROWS), (uint32_t)blocks[MODE]), threads8, shared8, stream>>>(
            source, work, nitems, (const uint8_t *)ax_book(moe, matrices, (uint32_t)MODE), in, scale, grouped,
            AX_USED, gate, nullptr, down, rows_out, cols_in, INFINITY, q8, q8_scale);
    };
    launch(std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_GATE>{}, x8, xs, AX_FFN, AX_HIDDEN, nullptr, nullptr);
    // The up epilogue writes down's E4M3 rows and scales directly.
    launch(std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_UP>{}, x8, xs, AX_FFN, AX_HIDDEN, mid8, ms);
    launch(std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_DOWN>{}, mid8, ms, AX_HIDDEN, AX_FFN, nullptr, nullptr);
    glmvq_mma_reduce<__nv_bfloat16, GLMVQ_REDUCE_F32><<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS,
        GLMVQ_THREADS, 0, stream>>>(out, down, weights, grouped, order, AX_USED, AX_HIDDEN, count, shared);
    return cuda_ok(cudaGetLastError(), "A.X-K2 FP8 grouped MMA");
}

extern "C" int ds4_gpu_axk2_moe(ds4_axk2_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, const uint32_t *matrices, uint32_t rows,
        const ds4_gpu_tensor *shared) {
    if (!moe || !rows || rows > moe->rows || !matrices || !selected || !gvq_stream(moe->gvq)) { return 0; }
    const uint64_t pairs = (uint64_t)rows * AX_USED, count = (uint64_t)rows * AX_HIDDEN;
    if (pairs > selected->bytes / sizeof(int32_t) || !gvq_floats(weights, pairs) ||
        !gvq_floats(out, count) || !gvq_floats(input, count) || (shared && !gvq_floats(shared, count)) ||
        matrices[0] >= moe->gvq->model->source.size() || ds4_capture_active()) { return 0; }
    const uint32_t bits = ax_bits(moe, matrices);
    const uint32_t *table = ax_table(moe, matrices, bits);
    if (!table) { return 0; }
    const float *sh = shared ? (const float *)shared->ptr : nullptr;
    if (rows == 1) {
        const auto *sel = (const int32_t *)selected->ptr;
        const auto *w = (const float *)weights->ptr;
        const cudaError_t err = bits == 10
            ? ax_decode_run<10>(moe, (float *)out->ptr, (const float *)input->ptr, sel, w, table, matrices, sh)
            : ax_decode_run<9>(moe, (float *)out->ptr, (const float *)input->ptr, sel, w, table, matrices, sh);
        return cuda_ok(err, "A.X-K2 routed decode");
    }
    std::vector<uint32_t> offsets(AX_EXPERTS + 1);
    if (!moe->mma || !ax_group(moe, selected, rows, offsets.data())) { return 0; }
    return bits == 10
        ? ax_mma_run<10>(moe, (float *)out->ptr, (const float *)input->ptr, (const float *)weights->ptr,
            matrices, offsets.data(), rows, sh)
        : ax_mma_run<9>(moe, (float *)out->ptr, (const float *)input->ptr, (const float *)weights->ptr,
            matrices, offsets.data(), rows, sh);
}

/* Router: Beam's group-limited sigmoid kernel (256 experts, 8 groups, keep
 * 4, -inf outside, top-8, s / (sum s + 1e-20)), then the routed scale the
 * reference applies to the weights (wt * routed_scaling_factor). */
__global__ void ax_route_scale(float *weights, uint64_t count, float scale) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { weights[i] = glmvq_mul(weights[i], scale); }
}

extern "C" int ds4_gpu_axk2_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, uint32_t bias, uint32_t rows) {
    const auto *b = (const float *)gvq_weight(ctx, bias, GVQ_TYPE_F32, AX_EXPERTS);
    const uint64_t pairs = (uint64_t)rows * AX_USED;
    if (!b || !rows || !gvq_floats(logits, (uint64_t)rows * AX_EXPERTS) || !gvq_floats(weights, pairs) ||
        selected->bytes < pairs * sizeof(int32_t)) { return 0; }
    static_assert((uint32_t)AX_EXPERTS == (uint32_t)BEAM_GPU_EXPERTS && (uint32_t)AX_USED == (uint32_t)BEAM_GPU_USED,
        "Beam's router geometry");
    enum { WARPS = 8 };
    const cudaStream_t stream = ds4_current_stream();
    beam_router_kernel<<<(rows + WARPS - 1) / WARPS, WARPS * 32, 0, stream>>>(
        (int32_t *)selected->ptr, (float *)weights->ptr, (const float *)logits->ptr, b, rows);
    ax_route_scale<<<(uint32_t)((pairs + 255) / 256), 256, 0, stream>>>((float *)weights->ptr, pairs, 2.5f);
    return cuda_ok(cudaGetLastError(), "A.X-K2 router");
}

/* ---------------------------------------------------------------- norms */

enum { AX_ROW_THREADS = 256, AX_ROW_VALUES = AX_HIDDEN / AX_ROW_THREADS, AX_ROW_WARPS = AX_ROW_THREADS / 32 };
static_assert(AX_HIDDEN % (AX_ROW_THREADS * 8) == 0 || AX_HIDDEN % AX_ROW_THREADS == 0, "whole rows per block");

// Sum of `value` over the 256-thread block; every thread gets it.
__device__ __forceinline__ float ax_block_sum(float value, float *scratch) {
    for (uint32_t stride = 16; stride; stride >>= 1) { value += __shfl_xor_sync(0xffffffffu, value, stride); }
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x / 32] = value; }
    __syncthreads();
    float sum = 0.0f;
    for (uint32_t w = 0; w < AX_ROW_WARPS; w++) { sum += scratch[w]; }
    __syncthreads();
    return sum;
}

/* One block per row (axk2_ref.rms_norm / gated_norm):
 *   cur = bf16(cur + bf16(branch))                       when branch
 *   y = bf16(w * bf16(x * rsqrt(mean(x^2) + 1e-6)))
 *   out = y, or bf16(y * sigmoid(bf16(W_up silu_bf16(bf16(W_down y)))))
 * W_down rows are dotted by warps (two of the 16 each, 16-byte loads);
 * each thread then takes its columns' 16-wide W_up rows. */
__global__ void __launch_bounds__(AX_ROW_THREADS) ax_norm_kernel(float *out, float *cur, const float *branch,
        const __nv_bfloat16 *weight, const __nv_bfloat16 *down, const __nv_bfloat16 *up) {
    __shared__ float scratch[AX_ROW_WARPS];
    __shared__ __align__(16) __nv_bfloat16 y_row[AX_HIDDEN];
    __shared__ float mid[AX_RANK];
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const uint64_t base = (uint64_t)blockIdx.x * AX_HIDDEN;
    float x[AX_ROW_VALUES];
    float sum = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < AX_ROW_VALUES; j++) {
        const uint32_t col = t + j * AX_ROW_THREADS;
        float v = glmvq_round(cur[base + col]);
        if (branch) {
            v = glmvq_round(glmvq_add(v, glmvq_round(branch[base + col])));
            cur[base + col] = v;
        }
        x[j] = v;
        sum = fmaf(v, v, sum);
    }
    const float inv = __frsqrt_rn(glmvq_add(__fdiv_rn(ax_block_sum(sum, scratch), (float)AX_HIDDEN), AX_EPS));
    #pragma unroll
    for (uint32_t j = 0; j < AX_ROW_VALUES; j++) {
        const uint32_t col = t + j * AX_ROW_THREADS;
        x[j] = glmvq_round(glmvq_mul(__bfloat162float(weight[col]), glmvq_round(glmvq_mul(x[j], inv))));
        if (!down) { out[base + col] = x[j]; } else { y_row[col] = __float2bfloat16_rn(x[j]); }
    }
    if (!down) { return; }
    __syncthreads();
    for (uint32_t k = warp; k < AX_RANK; k += AX_ROW_WARPS) {
        const uint4 *w = (const uint4 *)(down + (uint64_t)k * AX_HIDDEN);
        const uint4 *y = (const uint4 *)y_row;
        float acc = 0.0f;
        #pragma unroll 4
        for (uint32_t c = lane; c < AX_HIDDEN / 8; c += 32) {
            const uint4 wv = w[c], yv = y[c];
            const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w}, ys[4] = {yv.x, yv.y, yv.z, yv.w};
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++) {
                acc = fmaf(__uint_as_float(ws[i] << 16), __uint_as_float(ys[i] << 16), acc);
                acc = fmaf(__uint_as_float(ws[i] & 0xffff0000u), __uint_as_float(ys[i] & 0xffff0000u), acc);
            }
        }
        for (uint32_t stride = 16; stride; stride >>= 1) { acc += __shfl_xor_sync(0xffffffffu, acc, stride); }
        // F.silu of the BF16 projection, rounded to BF16.
        if (!lane) {
            const float g = glmvq_round(acc);
            mid[k] = glmvq_round(__fdiv_rn(g, glmvq_add(1.0f, expf(-g))));
        }
    }
    __syncthreads();
    #pragma unroll
    for (uint32_t j = 0; j < AX_ROW_VALUES; j++) {
        const uint32_t col = t + j * AX_ROW_THREADS;
        const uint4 *w = (const uint4 *)(up + (uint64_t)col * AX_RANK);
        const uint4 lo = w[0], hi = w[1];
        const uint32_t ws[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        float g = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < 8; i++) {
            g = fmaf(__uint_as_float(ws[i] << 16), mid[2 * i], g);
            g = fmaf(__uint_as_float(ws[i] & 0xffff0000u), mid[2 * i + 1], g);
        }
        out[base + col] = glmvq_round(glmvq_mul(x[j], glmvq_sigmoid(glmvq_round(g))));
    }
}

/* Decode widths split the gated norm over CTAs: one block per row took
 * ~42 us streaming both 229 KB gate matrices. CTA (k, r) of the first
 * kernel recomputes row r's norm in ax_norm_kernel's order and writes
 * mid[r][k] (the same warp dot of W_down row k), CTA (0, r) also 1/rms in
 * mid[r][16]; the second kernel's CTAs take 256 columns each: the residual
 * add, y and the W_up gate. Same arithmetic and order: bit-identical. */
enum { AX_GNORM_MID = AX_RANK + 1, AX_GNORM_ROWS = 4 };

__global__ void __launch_bounds__(AX_ROW_THREADS) ax_gnorm_down_kernel(float *mid, const float *cur,
        const float *branch, const __nv_bfloat16 *weight, const __nv_bfloat16 *down) {
    __shared__ float scratch[AX_ROW_WARPS];
    __shared__ __align__(16) __nv_bfloat16 y_row[AX_HIDDEN];
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x, lane = t & 31, k = blockIdx.x, r = blockIdx.y;
    const uint64_t base = (uint64_t)r * AX_HIDDEN;
    float x[AX_ROW_VALUES];
    float sum = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < AX_ROW_VALUES; j++) {
        const uint32_t col = t + j * AX_ROW_THREADS;
        float v = glmvq_round(cur[base + col]);
        if (branch) { v = glmvq_round(glmvq_add(v, glmvq_round(branch[base + col]))); }
        x[j] = v;
        sum = fmaf(v, v, sum);
    }
    const float inv = __frsqrt_rn(glmvq_add(__fdiv_rn(ax_block_sum(sum, scratch), (float)AX_HIDDEN), AX_EPS));
    #pragma unroll
    for (uint32_t j = 0; j < AX_ROW_VALUES; j++) {
        const uint32_t col = t + j * AX_ROW_THREADS;
        y_row[col] = __float2bfloat16_rn(glmvq_round(glmvq_mul(__bfloat162float(weight[col]),
            glmvq_round(glmvq_mul(x[j], inv)))));
    }
    __syncthreads();
    if (t >= 32) { return; }
    const uint4 *w = (const uint4 *)(down + (uint64_t)k * AX_HIDDEN);
    const uint4 *y = (const uint4 *)y_row;
    float acc = 0.0f;
    #pragma unroll 4
    for (uint32_t c = lane; c < AX_HIDDEN / 8; c += 32) {
        const uint4 wv = w[c], yv = y[c];
        const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w}, ys[4] = {yv.x, yv.y, yv.z, yv.w};
        #pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
            acc = fmaf(__uint_as_float(ws[i] << 16), __uint_as_float(ys[i] << 16), acc);
            acc = fmaf(__uint_as_float(ws[i] & 0xffff0000u), __uint_as_float(ys[i] & 0xffff0000u), acc);
        }
    }
    for (uint32_t stride = 16; stride; stride >>= 1) { acc += __shfl_xor_sync(0xffffffffu, acc, stride); }
    if (lane) { return; }
    const float g = glmvq_round(acc);
    mid[r * AX_GNORM_MID + k] = glmvq_round(__fdiv_rn(g, glmvq_add(1.0f, expf(-g))));
    if (!k) { mid[r * AX_GNORM_MID + AX_RANK] = inv; }
}

__global__ void __launch_bounds__(AX_ROW_THREADS) ax_gnorm_up_kernel(float *out, float *cur, const float *branch,
        const __nv_bfloat16 *weight, const __nv_bfloat16 *up, const float *mid) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t col = blockIdx.x * AX_ROW_THREADS + threadIdx.x, r = blockIdx.y;
    const uint64_t at = (uint64_t)r * AX_HIDDEN + col;
    float v = glmvq_round(cur[at]);
    if (branch) {
        v = glmvq_round(glmvq_add(v, glmvq_round(branch[at])));
        cur[at] = v;
    }
    const float *m = mid + r * AX_GNORM_MID;
    const float x = glmvq_round(glmvq_mul(__bfloat162float(weight[col]), glmvq_round(glmvq_mul(v, m[AX_RANK]))));
    const uint4 *w = (const uint4 *)(up + (uint64_t)col * AX_RANK);
    const uint4 lo = w[0], hi = w[1];
    const uint32_t ws[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    float g = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        g = fmaf(__uint_as_float(ws[i] << 16), m[2 * i], g);
        g = fmaf(__uint_as_float(ws[i] & 0xffff0000u), m[2 * i + 1], g);
    }
    out[at] = glmvq_round(glmvq_mul(x, glmvq_sigmoid(glmvq_round(g))));
}

extern "C" int ds4_gpu_axk2_norm(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gpu_tensor *cur,
        const ds4_gpu_tensor *branch, uint32_t norm, uint32_t down, uint32_t up, ds4_gpu_tensor *mid,
        uint32_t rows) {
    const auto *w = (const __nv_bfloat16 *)gvq_weight(ctx, norm, GVQ_TYPE_BF16, AX_HIDDEN);
    const bool gated = down != DS4_GVQ_NO_LINK;
    const auto *wd = gated ? (const __nv_bfloat16 *)gvq_weight(ctx, down, GVQ_TYPE_BF16, AX_RANK * AX_HIDDEN) : nullptr;
    const auto *wu = gated ? (const __nv_bfloat16 *)gvq_weight(ctx, up, GVQ_TYPE_BF16, AX_RANK * AX_HIDDEN) : nullptr;
    const uint64_t count = (uint64_t)rows * AX_HIDDEN;
    if (!w || (gated && (!wd || !wu || (uintptr_t)wd % 16 || (uintptr_t)wu % 16)) || !rows ||
        !gvq_floats(out, count) || !gvq_floats(cur, count) || (branch && !gvq_floats(branch, count)) ||
        !gvq_floats(mid, (uint64_t)rows * AX_GNORM_MID)) { return 0; }
    const float *pending = branch ? (const float *)branch->ptr : (const float *)nullptr;
    if (gated && rows <= AX_GNORM_ROWS) {
        glmvq_launch_kernel(gvq_mode(rows), ax_gnorm_down_kernel, dim3(AX_RANK, rows), AX_ROW_THREADS, 0,
            ds4_current_stream(), (float *)mid->ptr, (const float *)cur->ptr, pending, w, wd);
        glmvq_launch_kernel(gvq_mode(rows), ax_gnorm_up_kernel, dim3(AX_HIDDEN / AX_ROW_THREADS, rows),
            AX_ROW_THREADS, 0, ds4_current_stream(), (float *)out->ptr, (float *)cur->ptr, pending, w, wu,
            (const float *)mid->ptr);
        return cuda_ok(cudaGetLastError(), "A.X-K2 gated norm");
    }
    glmvq_launch_kernel(gvq_mode(rows), ax_norm_kernel, rows, AX_ROW_THREADS, 0, ds4_current_stream(),
        (float *)out->ptr, (float *)cur->ptr, branch ? (const float *)branch->ptr : (const float *)nullptr, w, wd, wu);
    return cuda_ok(cudaGetLastError(), "A.X-K2 norm");
}

/* ----------------------------------------------------------------- RoPE */

/* (cos, sin) of the F32 angle pos * inv_freq (one product, as the
 * reference's F32 matmul), in double (fast-math cosf is unusable at 262K
 * positions), rounded to F32 and then BF16 (the source's .to(x.dtype)).
 * Every RoPE kernel of the forward reads this row. */
__global__ void ax_rope_kernel(float2 *rope, const float *inv, uint32_t pos0) {
    const uint32_t i = threadIdx.x, r = blockIdx.x;
    const float angle = glmvq_mul((float)(pos0 + r), inv[i]);
    rope[(uint64_t)r * AX_FREQS + i] = make_float2(glmvq_round((float)cos((double)angle)),
        glmvq_round((float)sin((double)angle)));
}

extern "C" int ds4_gpu_axk2_rope(ds4_gpu_tensor *rope, const ds4_gpu_tensor *inv_freq, uint32_t rows, uint32_t pos0) {
    if (!rows || !gvq_floats(rope, (uint64_t)rows * AX_ROPE) || !gvq_floats(inv_freq, AX_FREQS)) { return 0; }
    ax_rope_kernel<<<rows, AX_FREQS, 0, ds4_current_stream()>>>((float2 *)rope->ptr, (const float *)inv_freq->ptr,
        pos0);
    return cuda_ok(cudaGetLastError(), "A.X-K2 RoPE rows");
}

// Interleaved pair i of a 64-dim rope slice (x[2i], x[2i+1]) -> outputs i
// and 32 + i (de-interleaved), each product and sum at the BF16 boundary.
__device__ __forceinline__ float2 ax_rope_pair(float a, float b, float2 cs) {
    return make_float2(glmvq_round(glmvq_add(glmvq_round(glmvq_mul(a, cs.x)), -glmvq_round(glmvq_mul(b, cs.y)))),
        glmvq_round(glmvq_add(glmvq_round(glmvq_mul(b, cs.x)), glmvq_round(glmvq_mul(a, cs.y)))));
}

/* q_b row -> q (nope copied, rope slice rotated) and gate, one block of
 * 128 threads per (head, row). */
__global__ void ax_q_split_kernel(float *q, float *gate, const float *qb, const float2 *rope) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t h = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    const float *src = qb + ((uint64_t)r * AX_HEADS + h) * (AX_KEY + AX_VALUE);
    float *dq = q + ((uint64_t)r * AX_HEADS + h) * AX_KEY;
    dq[t] = glmvq_round(src[t]);
    gate[((uint64_t)r * AX_HEADS + h) * AX_VALUE + t] = glmvq_round(src[AX_KEY + t]);
    if (t < AX_FREQS) {
        const float2 v = ax_rope_pair(glmvq_round(src[AX_NOPE + 2 * t]), glmvq_round(src[AX_NOPE + 2 * t + 1]),
            rope[(uint64_t)r * AX_FREQS + t]);
        dq[AX_NOPE + t] = v.x;
        dq[AX_NOPE + AX_FREQS + t] = v.y;
    }
}

extern "C" int ds4_gpu_axk2_q_split(ds4_gpu_tensor *q, ds4_gpu_tensor *gate, const ds4_gpu_tensor *qb,
        const ds4_gpu_tensor *rope, uint32_t rows) {
    if (!rows || !gvq_floats(q, (uint64_t)rows * AX_HEADS * AX_KEY) ||
        !gvq_floats(gate, (uint64_t)rows * AX_HEADS * AX_VALUE) || !gvq_floats(qb, (uint64_t)rows * DS4_AXK2_QB) ||
        !gvq_floats(rope, (uint64_t)rows * AX_ROPE)) { return 0; }
    glmvq_launch_kernel(gvq_mode(rows), ax_q_split_kernel, dim3(AX_HEADS, rows), AX_VALUE, 0, ds4_current_stream(),
        (float *)q->ptr, (float *)gate->ptr, (const float *)qb->ptr, (const float2 *)rope->ptr);
    return cuda_ok(cudaGetLastError(), "A.X-K2 q split");
}

/* One 128-thread block per kv_a row: kv_a_layernorm over the 512 latent
 * values (four per thread), stored as Ling-3.1's FP8 latent row (E4M3 under
 * a power-of-two exponent: value x scale is an exact BF16), and the 64 rope
 * values rotated into k_pe. */
__global__ void ax_kv_store_kernel(uint8_t *latent, __nv_bfloat16 *k_pe, const float *kv, const __nv_bfloat16 *norm,
        const float2 *rope, uint32_t pos0, uint32_t cap) {
    __shared__ float scratch[4];
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t r = blockIdx.x, t = threadIdx.x, slot = pos0 + r;
    const float *row = kv + (uint64_t)r * AX_KV_ROW;
    float x[4], sum = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) { x[i] = glmvq_round(row[4 * t + i]); sum = fmaf(x[i], x[i], sum); }
    for (uint32_t stride = 16; stride; stride >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, stride); }
    if (!(t & 31u)) { scratch[t >> 5] = sum; }
    __syncthreads();
    const float total = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    __syncthreads();
    const float inv = __frsqrt_rn(glmvq_add(__fdiv_rn(total, (float)AX_LATENT), AX_EPS));
    float amax = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        x[i] = glmvq_round(glmvq_mul(__bfloat162float(norm[4 * t + i]), glmvq_round(glmvq_mul(x[i], inv))));
        amax = fmaxf(amax, fabsf(x[i]));
    }
    for (uint32_t stride = 16; stride; stride >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, stride)); }
    if (!(t & 31u)) { scratch[t >> 5] = amax; }
    __syncthreads();
    amax = fmaxf(fmaxf(scratch[0], scratch[1]), fmaxf(scratch[2], scratch[3]));
    const int e = l31_latent_exp(amax);
    const float descale = l31_pow2(-e);
    __nv_fp8_storage_t code[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) { code[i] = __nv_cvt_float_to_fp8(x[i] * descale, __NV_SATFINITE, __NV_E4M3); }
    *(uint32_t *)(latent + (uint64_t)slot * AX_LATENT + 4 * t) =
        (uint32_t)code[0] | (uint32_t)code[1] << 8 | (uint32_t)code[2] << 16 | (uint32_t)code[3] << 24;
    if (!t) { ((int8_t *)(latent + (uint64_t)cap * AX_LATENT))[slot] = (int8_t)e; }
    if (t < AX_FREQS) {
        const float2 v = ax_rope_pair(glmvq_round(row[AX_LATENT + 2 * t]), glmvq_round(row[AX_LATENT + 2 * t + 1]),
            rope[(uint64_t)r * AX_FREQS + t]);
        k_pe[(uint64_t)slot * AX_ROPE + t] = __float2bfloat16_rn(v.x);
        k_pe[(uint64_t)slot * AX_ROPE + AX_FREQS + t] = __float2bfloat16_rn(v.y);
    }
}

extern "C" int ds4_gpu_axk2_kv_store(ds4_gvq_ctx *ctx, ds4_gpu_tensor *latent, ds4_gpu_tensor *k_pe,
        const ds4_gpu_tensor *kv, uint32_t norm, const ds4_gpu_tensor *rope, uint32_t rows, uint32_t pos0,
        uint32_t cap) {
    const auto *w = (const __nv_bfloat16 *)gvq_weight(ctx, norm, GVQ_TYPE_BF16, AX_LATENT);
    if (!w || !rows || pos0 > cap || rows > cap - pos0 || !gvq_floats(kv, (uint64_t)rows * AX_KV_ROW) ||
        !gvq_floats(rope, (uint64_t)rows * AX_ROPE) || ds4_gpu_tensor_bytes(latent) < ds4_axk2_latent_bytes(cap) ||
        ds4_gpu_tensor_bytes(k_pe) < (uint64_t)cap * AX_ROPE * 2) { return 0; }
    glmvq_launch_kernel(gvq_mode(rows), ax_kv_store_kernel, rows, AX_LATENT / 4, 0, ds4_current_stream(),
        (uint8_t *)latent->ptr, (__nv_bfloat16 *)k_pe->ptr, (const float *)kv->ptr, w, (const float2 *)rope->ptr,
        pos0, cap);
    return cuda_ok(cudaGetLastError(), "A.X-K2 latent store");
}

/* Split-half RoPE of a 64-dim slice (rotate_half): element i < 32 pairs
 * with 32 + i, BF16 products and sums. Lane l holds dims l and 32 + l. */
__device__ __forceinline__ float2 ax_rope_half(float lo, float hi, float2 cs) {
    return make_float2(glmvq_round(glmvq_add(glmvq_round(glmvq_mul(lo, cs.x)), glmvq_round(glmvq_mul(-hi, cs.y)))),
        glmvq_round(glmvq_add(glmvq_round(glmvq_mul(hi, cs.x)), glmvq_round(glmvq_mul(lo, cs.y)))));
}

/* One warp per indexer key row (lane l: dims l, 32 + l, 64 + l, 96 + l):
 * LayerNorm with bias in F32, one BF16 rounding (torch's layer_norm), RoPE
 * on dims 0..63, then E4M3 under a row power-of-two (or BF16). */
__global__ void ax_index_store_kernel(uint8_t *cache, const float *k, const __nv_bfloat16 *w,
        const __nv_bfloat16 *b, const float2 *rope, uint32_t rows, uint32_t pos0, uint32_t cap, int bf16) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t lane = threadIdx.x & 31, r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (r >= rows) { return; }
    const float *row = k + (uint64_t)r * AX_IDX_DIM;
    float x[4], sum = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) { x[i] = glmvq_round(row[lane + 32 * i]); sum = glmvq_add(sum, x[i]); }
    for (uint32_t stride = 16; stride; stride >>= 1) { sum = glmvq_add(sum, __shfl_xor_sync(0xffffffffu, sum, stride)); }
    const float mean = __fdiv_rn(sum, (float)AX_IDX_DIM);
    float var = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) { const float d = glmvq_add(x[i], -mean); var = fmaf(d, d, var); }
    for (uint32_t stride = 16; stride; stride >>= 1) { var = glmvq_add(var, __shfl_xor_sync(0xffffffffu, var, stride)); }
    const float rstd = __frsqrt_rn(glmvq_add(__fdiv_rn(var, (float)AX_IDX_DIM), AX_EPS));
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        const uint32_t d = lane + 32 * i;
        x[i] = glmvq_round(glmvq_add(glmvq_mul(glmvq_mul(glmvq_add(x[i], -mean), rstd), __bfloat162float(w[d])),
            __bfloat162float(b[d])));
    }
    const float2 v = ax_rope_half(x[0], x[1], rope[(uint64_t)r * AX_FREQS + lane]);
    x[0] = v.x;
    x[1] = v.y;
    const uint32_t slot = pos0 + r;
    if (bf16) {
        __nv_bfloat16 *dst = (__nv_bfloat16 *)cache + (uint64_t)slot * AX_IDX_DIM;
        #pragma unroll
        for (uint32_t i = 0; i < 4; i++) { dst[lane + 32 * i] = __float2bfloat16_rn(x[i]); }
        return;
    }
    float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
    for (uint32_t stride = 16; stride; stride >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, stride)); }
    const int e = l31_latent_exp(amax);
    const float descale = l31_pow2(-e);
    uint8_t *dst = cache + (uint64_t)slot * AX_IDX_DIM;
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        dst[lane + 32 * i] = (uint8_t)__nv_cvt_float_to_fp8(x[i] * descale, __NV_SATFINITE, __NV_E4M3);
    }
    if (!lane) { ((int8_t *)(cache + (uint64_t)cap * AX_IDX_DIM))[slot] = (int8_t)e; }
}

extern "C" int ds4_gpu_axk2_index_store(ds4_gvq_ctx *ctx, ds4_gpu_tensor *keys, const ds4_gpu_tensor *k,
        uint32_t norm, uint32_t bias, const ds4_gpu_tensor *rope, uint32_t rows, uint32_t pos0, uint32_t cap,
        ds4_axk2_keys format) {
    const auto *w = (const __nv_bfloat16 *)gvq_weight(ctx, norm, GVQ_TYPE_BF16, AX_IDX_DIM);
    const auto *b = (const __nv_bfloat16 *)gvq_weight(ctx, bias, GVQ_TYPE_BF16, AX_IDX_DIM);
    if (!w || !b || !rows || pos0 > cap || rows > cap - pos0 || !gvq_floats(k, (uint64_t)rows * AX_IDX_DIM) ||
        !gvq_floats(rope, (uint64_t)rows * AX_ROPE) ||
        ds4_gpu_tensor_bytes(keys) < ds4_axk2_index_bytes(cap, format)) { return 0; }
    enum { WARPS = 8 };
    glmvq_launch_kernel(gvq_mode(rows), ax_index_store_kernel, (rows + WARPS - 1) / WARPS, WARPS * 32, 0,
        ds4_current_stream(), (uint8_t *)keys->ptr, (const float *)k->ptr, w, b, (const float2 *)rope->ptr, rows,
        pos0, cap, format == DS4_AXK2_KEYS_BF16 ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "A.X-K2 index key store");
}

// One warp per (head, row) of the indexer queries, in place.
__global__ void ax_index_q_kernel(float *q, const float2 *rope, uint32_t count) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t lane = threadIdx.x & 31, i = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (i >= count) { return; }
    float *row = q + (uint64_t)i * AX_IDX_DIM;
    const float2 v = ax_rope_half(glmvq_round(row[lane]), glmvq_round(row[32 + lane]),
        rope[(uint64_t)(i / AX_IDX_HEADS) * AX_FREQS + lane]);
    row[lane] = v.x;
    row[32 + lane] = v.y;
    row[64 + lane] = glmvq_round(row[64 + lane]);
    row[96 + lane] = glmvq_round(row[96 + lane]);
}

extern "C" int ds4_gpu_axk2_index_q(ds4_gpu_tensor *q, const ds4_gpu_tensor *rope, uint32_t rows) {
    const uint32_t count = rows * AX_IDX_HEADS;
    if (!rows || !gvq_floats(q, (uint64_t)count * AX_IDX_DIM) || !gvq_floats(rope, (uint64_t)rows * AX_ROPE)) {
        return 0;
    }
    enum { WARPS = 8 };
    glmvq_launch_kernel(gvq_mode(rows), ax_index_q_kernel, (count + WARPS - 1) / WARPS, WARPS * 32, 0,
        ds4_current_stream(), (float *)q->ptr, (const float2 *)rope->ptr, count);
    return cuda_ok(cudaGetLastError(), "A.X-K2 index queries");
}

/* --------------------------------------------------------- index scores */

/* Indexer scores on BF16 tensor cores. A CTA of 4 warps keeps ROWS query
 * rows (64 heads x 128 dims, BF16) in shared memory and walks 64-key tiles;
 * warp w converts keys 16w..16w+15 of a tile into its own BF16 A tile (E4M3
 * x 2^e is an exact BF16) and runs C[16 keys][8 heads] for the 8 head
 * groups of each row: s_h = relu(C x 128^-0.5), score = sum_h w_h s_h. The
 * products are exact (BF16 operands), only the F32 accumulation order
 * differs from the reference's einsum. */
enum { AX_SC_WARPS = 4, AX_SC_KEYS = 16 * AX_SC_WARPS, AX_SC_QLD = AX_IDX_DIM + 8, AX_SC_KLD = AX_IDX_DIM + 8 };

template <uint32_t ROWS>
static constexpr uint32_t ax_score_smem() {
    return ROWS * AX_IDX_HEADS * AX_SC_QLD * 2 + ROWS * AX_IDX_HEADS * 4 + AX_SC_WARPS * 16 * AX_SC_KLD * 2;
}

template <uint32_t ROWS, bool BF16_KEYS>
__global__ void __launch_bounds__(AX_SC_WARPS * 32) ax_score_kernel(float *scores, const float *q, const float *w,
        const uint8_t *cache, uint32_t rows, uint32_t pos0, uint32_t keys, uint32_t stride, uint32_t cap) {
    extern __shared__ __align__(16) uint8_t sc_shared[];
    auto *qs = (__nv_bfloat16 *)sc_shared;
    auto *ws = (float *)(sc_shared + ROWS * AX_IDX_HEADS * AX_SC_QLD * 2);
    auto *kt = (__nv_bfloat16 *)(sc_shared + ROWS * AX_IDX_HEADS * (AX_SC_QLD * 2 + 4));
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, qd = lane & 3;
    const uint32_t r0 = blockIdx.y * ROWS, nrows = min(ROWS, rows - r0);
    for (uint32_t i = tid; i < ROWS * AX_IDX_HEADS * AX_IDX_DIM; i += AX_SC_WARPS * 32) {
        const uint32_t r = i / (AX_IDX_HEADS * AX_IDX_DIM), hd = i % (AX_IDX_HEADS * AX_IDX_DIM);
        const float v = r < nrows ? q[(uint64_t)(r0 + r) * AX_IDX_HEADS * AX_IDX_DIM + hd] : 0.0f;
        qs[(r * AX_IDX_HEADS + hd / AX_IDX_DIM) * AX_SC_QLD + hd % AX_IDX_DIM] = __float2bfloat16_rn(v);
    }
    for (uint32_t i = tid; i < ROWS * AX_IDX_HEADS; i += AX_SC_WARPS * 32) {
        ws[i] = i / AX_IDX_HEADS < nrows ? w[(uint64_t)r0 * AX_IDX_HEADS + i] : 0.0f;
    }
    __syncthreads();
    const float scale = 0.08838834764831845f;  // 128 ** -0.5
    __nv_bfloat16 *mine = kt + warp * 16 * AX_SC_KLD;
    const uint32_t last = pos0 + r0 + nrows - 1;   // the CTA's highest causal key
    const uint32_t tiles = (keys + AX_SC_KEYS - 1) / AX_SC_KEYS;
    for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const uint32_t key0 = tile * AX_SC_KEYS + warp * 16;
        if (key0 >= keys) { continue; }
        if (key0 > last) {
            // Every row of the CTA is below these keys.
            for (uint32_t i = lane; i < 16 * nrows; i += 32) {
                const uint32_t key = key0 + i % 16;
                if (key < keys) { scores[(uint64_t)(r0 + i / 16) * stride + key] = -INFINITY; }
            }
            continue;
        }
        // The warp's 16 keys as BF16 rows (lane: half a key row).
        {
            const uint32_t k = lane >> 1, part = lane & 1, key = min(key0 + k, keys - 1);
            if constexpr (BF16_KEYS) {
                const uint4 *src = (const uint4 *)(cache + (uint64_t)key * AX_IDX_DIM * 2) + part * 8;
                #pragma unroll
                for (uint32_t c = 0; c < 8; c++) { *(uint4 *)(mine + k * AX_SC_KLD + part * 64 + c * 8) = src[c]; }
            } else {
                const uint4 *src = (const uint4 *)(cache + (uint64_t)key * AX_IDX_DIM) + part * 4;
                const float s = l31_pow2(((const int8_t *)(cache + (uint64_t)cap * AX_IDX_DIM))[key]);
                #pragma unroll
                for (uint32_t c = 0; c < 4; c++) {
                    uint4 lo, hi;
                    l31_latent_bf16x16(src[c], s, &lo, &hi);
                    *(uint4 *)(mine + k * AX_SC_KLD + part * 64 + c * 16) = lo;
                    *(uint4 *)(mine + k * AX_SC_KLD + part * 64 + c * 16 + 8) = hi;
                }
            }
        }
        __syncwarp();
        uint32_t a[8][4];
        #pragma unroll
        for (uint32_t kk = 0; kk < 8; kk++) {
            const uint32_t col = kk * 16 + 2 * qd;
            a[kk][0] = *(const uint32_t *)(mine + g * AX_SC_KLD + col);
            a[kk][1] = *(const uint32_t *)(mine + (g + 8) * AX_SC_KLD + col);
            a[kk][2] = *(const uint32_t *)(mine + g * AX_SC_KLD + col + 8);
            a[kk][3] = *(const uint32_t *)(mine + (g + 8) * AX_SC_KLD + col + 8);
        }
        __syncwarp();
        for (uint32_t r = 0; r < nrows; r++) {
            float lo = 0.0f, hi = 0.0f;     // keys g and g + 8
            #pragma unroll
            for (uint32_t nt = 0; nt < AX_IDX_HEADS / 8; nt++) {
                float c[4] = {};
                const __nv_bfloat16 *qrow = qs + (r * AX_IDX_HEADS + nt * 8 + g) * AX_SC_QLD;
                #pragma unroll
                for (uint32_t kk = 0; kk < 8; kk++) {
                    const uint32_t col = kk * 16 + 2 * qd;
                    glmvq_fa_mma(c, a[kk], *(const uint32_t *)(qrow + col), *(const uint32_t *)(qrow + col + 8));
                }
                const float w0 = ws[r * AX_IDX_HEADS + nt * 8 + 2 * qd], w1 = ws[r * AX_IDX_HEADS + nt * 8 + 2 * qd + 1];
                lo = fmaf(w0, fmaxf(glmvq_mul(c[0], scale), 0.0f), lo);
                lo = fmaf(w1, fmaxf(glmvq_mul(c[1], scale), 0.0f), lo);
                hi = fmaf(w0, fmaxf(glmvq_mul(c[2], scale), 0.0f), hi);
                hi = fmaf(w1, fmaxf(glmvq_mul(c[3], scale), 0.0f), hi);
            }
            for (uint32_t s = 1; s < 4; s <<= 1) {
                lo += __shfl_xor_sync(0xffffffffu, lo, s);
                hi += __shfl_xor_sync(0xffffffffu, hi, s);
            }
            if (qd == 0) {
                const uint32_t position = pos0 + r0 + r;
                float *dst = scores + (uint64_t)(r0 + r) * stride;
                if (key0 + g < keys) { dst[key0 + g] = key0 + g <= position ? lo : -INFINITY; }
                if (key0 + g + 8 < keys) { dst[key0 + g + 8] = key0 + g + 8 <= position ? hi : -INFINITY; }
            }
        }
    }
}

template <uint32_t ROWS>
static cudaError_t ax_score_launch(float *scores, const float *q, const float *w, const uint8_t *cache, uint32_t rows,
        uint32_t pos0, uint32_t keys, uint32_t stride, uint32_t cap, bool bf16, uint32_t sms, cudaStream_t stream) {
    constexpr uint32_t smem = ax_score_smem<ROWS>();
    auto kernel = bf16 ? ax_score_kernel<ROWS, true> : ax_score_kernel<ROWS, false>;
    cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) { return err; }
    // Enough CTAs to fill the GPU; each walks tiles, so its query rows stage once.
    const uint32_t groups = (rows + ROWS - 1) / ROWS, tiles = (keys + AX_SC_KEYS - 1) / AX_SC_KEYS;
    const uint32_t across = max(1u, min(tiles, (8u * sms + groups - 1) / groups));
    kernel<<<dim3(across, groups), AX_SC_WARPS * 32, smem, stream>>>(scores, q, w, cache, rows, pos0, keys, stride,
        cap);
    return cudaGetLastError();
}

extern "C" int ds4_gpu_axk2_index_scores(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q, const ds4_gpu_tensor *w,
        const ds4_gpu_tensor *cache, uint32_t rows, uint32_t pos0, uint32_t keys, uint32_t stride, uint32_t cap,
        ds4_axk2_keys format) {
    if (!rows || !keys || keys > cap || stride < keys || !gvq_floats(scores, (uint64_t)rows * stride) ||
        !gvq_floats(q, (uint64_t)rows * AX_IDX_HEADS * AX_IDX_DIM) || !gvq_floats(w, (uint64_t)rows * AX_IDX_HEADS) ||
        ds4_gpu_tensor_bytes(cache) < ds4_axk2_index_bytes(cap, format)) { return 0; }
    static uint32_t sms = 0;
    if (!sms) {
        int device = 0, count = 0;
        sms = cudaGetDevice(&device) == cudaSuccess &&
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) == cudaSuccess ? (uint32_t)count : 48u;
    }
    const bool bf16 = format == DS4_AXK2_KEYS_BF16;
    const cudaStream_t stream = ds4_current_stream();
    const cudaError_t err = rows == 1
        ? ax_score_launch<1>((float *)scores->ptr, (const float *)q->ptr, (const float *)w->ptr,
            (const uint8_t *)cache->ptr, rows, pos0, keys, stride, cap, bf16, sms, stream)
        : ax_score_launch<4>((float *)scores->ptr, (const float *)q->ptr, (const float *)w->ptr,
            (const uint8_t *)cache->ptr, rows, pos0, keys, stride, cap, bf16, sms, stream);
    return cuda_ok(err, "A.X-K2 index scores");
}

// DS4_AXK2_TOPK=0 selects on DeepSeek's indexer kernels instead (the
// chunked sort and merge tree past 8,192 keys), a diagnostic.
extern "C" int ds4_gpu_axk2_topk(ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores, uint32_t keys,
        uint32_t rows) {
    if (!rows || keys < AX_TOPK || !gvq_floats(scores, (uint64_t)rows * keys) ||
        ds4_gpu_tensor_bytes(selected) < (uint64_t)rows * AX_TOPK * sizeof(uint32_t)) { return 0; }
    static const bool legacy = ax_env_off("DS4_AXK2_TOPK");
    if (legacy) { return ds4_gpu_indexer_topk_tensor(selected, scores, keys, rows, AX_TOPK, 0u, UINT32_MAX); }
    axk2_topk_kernel<AX_TOPK><<<rows, AXK2_TOPK_THREADS, 0, ds4_current_stream()>>>((uint32_t *)selected->ptr,
        (const float *)scores->ptr, keys, keys);
    return cuda_ok(cudaGetLastError(), "A.X-K2 top-k");
}

/* ------------------------------------------------- absorbed attention */

enum { AX_DIRECT_ROWS = 4, AX_Q4_ROW = AX_LATENT / GLMVQ_Q4_BLOCK * GLMVQ_Q4_BYTES };

enum { AX_ABSORB_WARPS = 8, AX_ABSORB_ROWS = AX_NOPE / AX_ABSORB_WARPS };

/* q_abs[r][h][c] = sum_d q[r][h][d] W[h * 256 + d][c] over the head's 128
 * k rows. One block per head: warp w dequantizes rows 16w..16w+15, a whole
 * 256-block per step in glmvq_q4_acc's lanes (bytes 4l..4l+3: columns
 * 64g + o.. and 64g + 32 + o..), and the warps' partial rows add in warp
 * order. The weights are the restore's values; only the F32 sum order
 * differs from a column walk (one value per thread and row ran ~25 GB/s). */
template <uint32_t NT>
__global__ void __launch_bounds__(256) ax_absorb_kernel(float *q_abs, const float *q, const uint8_t *kv_b,
        uint32_t rows) {
    __shared__ float qs[NT][AX_NOPE];
    __shared__ __align__(16) float part[AX_ABSORB_WARPS][AX_LATENT];
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t h = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const uint32_t group = lane >> 3, offset = (lane & 7) * 4;
    for (uint32_t i = t; i < NT * AX_NOPE; i += 256) {
        const uint32_t r = i / AX_NOPE;
        qs[r][i % AX_NOPE] = r < rows ? q[((uint64_t)r * AX_HEADS + h) * AX_KEY + i % AX_NOPE] : 0.0f;
    }
    __syncthreads();
    float acc[NT][2][8] = {};
    #pragma unroll 4
    for (uint32_t k = 0; k < AX_ABSORB_ROWS; k++) {
        const uint32_t d = warp * AX_ABSORB_ROWS + k;
        const uint8_t *row = kv_b + (uint64_t)(h * (AX_NOPE + AX_VALUE) + d) * AX_Q4_ROW;
        #pragma unroll
        for (uint32_t b = 0; b < 2; b++) {
            const uint8_t *block = row + b * GLMVQ_Q4_BYTES;
            const uint4 hd = *(const uint4 *)block;
            const uint32_t packed = *(const uint32_t *)(block + 16 + lane * 4);
            const float dd = __half2float(__ushort_as_half(hd.x & 65535u));
            const float dm = __half2float(__ushort_as_half(hd.x >> 16));
            float sl, ml, sh, mh;
            glmvq_q4_scales(hd, group * 2, &sl, &ml);
            glmvq_q4_scales(hd, group * 2 + 1, &sh, &mh);
            sl *= dd; sh *= dd; ml *= dm; mh *= dm;
            #pragma unroll
            for (uint32_t j = 0; j < 4; j++) {
                const uint32_t byte = packed >> (j * 8) & 255u;
                const float lo = sl * (byte & 15u) - ml, hi = sh * (byte >> 4) - mh;
                #pragma unroll
                for (uint32_t r = 0; r < NT; r++) {
                    acc[r][b][j] = fmaf(qs[r][d], lo, acc[r][b][j]);
                    acc[r][b][4 + j] = fmaf(qs[r][d], hi, acc[r][b][4 + j]);
                }
            }
        }
    }
    #pragma unroll
    for (uint32_t r = 0; r < NT; r++) {
        if (r >= rows) { break; }
        #pragma unroll
        for (uint32_t b = 0; b < 2; b++) {
            float *dst = part[warp] + b * GLMVQ_Q4_BLOCK + group * 64 + offset;
            *(float4 *)dst = make_float4(acc[r][b][0], acc[r][b][1], acc[r][b][2], acc[r][b][3]);
            *(float4 *)(dst + 32) = make_float4(acc[r][b][4], acc[r][b][5], acc[r][b][6], acc[r][b][7]);
        }
        __syncthreads();
        float s0 = 0.0f, s1 = 0.0f;
        #pragma unroll
        for (uint32_t w = 0; w < AX_ABSORB_WARPS; w++) {
            s0 += part[w][2 * t];
            s1 += part[w][2 * t + 1];
        }
        *(float2 *)(q_abs + ((uint64_t)r * AX_HEADS + h) * AX_LATENT + 2 * t) = make_float2(s0, s1);
        __syncthreads();
    }
}

/* heads[r][h][v] = bf16(bf16(sum_c lat[r][h][c] W[h * 256 + 128 + v][c])
 * x sigmoid(gate)); a warp per output row (glmvq_q4_acc's lane order). */
template <uint32_t NT>
__global__ void __launch_bounds__(256) ax_value_kernel(float *heads, const float *lat, const uint8_t *kv_b,
        const float *gate, uint32_t rows) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t lane = threadIdx.x & 31, o = blockIdx.x * 8 + threadIdx.x / 32;
    const uint32_t h = o / AX_VALUE, v = o % AX_VALUE;
    const uint8_t *row = kv_b + (uint64_t)(h * (AX_NOPE + AX_VALUE) + AX_NOPE + v) * AX_Q4_ROW;
    float sum[NT] = {};
    #pragma unroll
    for (uint32_t b = 0; b < AX_LATENT / GLMVQ_Q4_BLOCK; b++) {
        glmvq_q4_acc<NT>(row + b * GLMVQ_Q4_BYTES, lat + (uint64_t)h * AX_LATENT, AX_HEADS * AX_LATENT,
            b * GLMVQ_Q4_BLOCK, rows, sum);
    }
    #pragma unroll
    for (uint32_t r = 0; r < NT; r++) {
        for (uint32_t s = 16; s; s >>= 1) { sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], s); }
        if (lane || r >= rows) { continue; }
        const uint64_t at = ((uint64_t)r * AX_HEADS + h) * AX_VALUE + v;
        heads[at] = glmvq_round(glmvq_mul(glmvq_round(sum[r]), glmvq_sigmoid(glmvq_round(gate[at]))));
    }
}

// Prefill widths: kv_b restored to BF16, then one strided-batched cuBLAS
// GEMM per absorption (64 heads), BF16 operands and F32 accumulation.
struct ax_absorb_layout { uint64_t kv_b, q, lat, out, bytes; };

static ax_absorb_layout ax_absorb_plan(uint32_t rows) {
    ax_absorb_layout l = {};
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += ax_align(bytes); return here; };
    l.kv_b = take((uint64_t)AX_KV_B_ROWS * AX_LATENT * 2);
    l.q = take((uint64_t)rows * AX_HEADS * AX_NOPE * 2);
    l.lat = take((uint64_t)rows * AX_HEADS * AX_LATENT * 2);
    l.out = take((uint64_t)rows * AX_HEADS * AX_VALUE * 4);
    l.bytes = at;
    return l;
}

extern "C" uint64_t ds4_gpu_axk2_absorb_bytes(uint32_t rows) { return rows > AX_DIRECT_ROWS ? ax_absorb_plan(rows).bytes : 0; }

// Single-split prefill writes the value GEMM's BF16 operand directly.
static bool ax_latent_bf16(uint32_t rows) {
    static const bool enabled = !ax_env_off("DS4_AXK2_FLASH_BF16");
    return enabled && rows >= L31_FA_SPLITS;
}

// BF16 copies of strided F32 rows: `cols` of each `stride`-float row.
__global__ void ax_bf16_rows(__nv_bfloat16 *out, const float *in, uint32_t cols, uint32_t stride, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = __float2bfloat16_rn(in[i / cols * stride + i % cols]); }
}

__global__ void ax_gate_kernel(float *heads, const float *raw, const float *gate, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { heads[i] = glmvq_round(glmvq_mul(glmvq_round(raw[i]), glmvq_sigmoid(glmvq_round(gate[i])))); }
}

static const uint8_t *ax_kv_b(ds4_gvq_ctx *ctx, uint32_t kv_b) {
    return (const uint8_t *)gvq_weight(ctx, kv_b, GVQ_TYPE_Q4, (uint64_t)AX_KV_B_ROWS * AX_Q4_ROW / 2);
}

extern "C" int ds4_gpu_axk2_absorb(ds4_gvq_ctx *ctx, ds4_gpu_tensor *q_abs, const ds4_gpu_tensor *q, uint32_t kv_b,
        uint32_t rows, ds4_gpu_tensor *scratch, ds4_axk2_kv_b restore) {
    const uint8_t *w = ax_kv_b(ctx, kv_b);
    if (!w || !rows || !gvq_floats(q_abs, (uint64_t)rows * AX_HEADS * AX_LATENT) ||
        !gvq_floats(q, (uint64_t)rows * AX_HEADS * AX_KEY) || !gvq_stream(ctx)) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    if (rows <= AX_DIRECT_ROWS) {
        auto run = [&](auto nt) {
            glmvq_launch_kernel(gvq_mode(rows), ax_absorb_kernel<decltype(nt)::value>, AX_HEADS, 256, 0, stream,
                (float *)q_abs->ptr, (const float *)q->ptr, w, rows);
        };
        if (rows == 1) { run(std::integral_constant<uint32_t, 1>{}); }
        else { run(std::integral_constant<uint32_t, AX_DIRECT_ROWS>{}); }
        return cuda_ok(cudaGetLastError(), "A.X-K2 absorb");
    }
    const ax_absorb_layout l = ax_absorb_plan(rows);
    if (!scratch || ds4_gpu_tensor_bytes(scratch) < l.bytes) { return 0; }
    uint8_t *base = (uint8_t *)scratch->ptr;
    auto *wb = (__nv_bfloat16 *)(base + l.kv_b), *qb = (__nv_bfloat16 *)(base + l.q);
    const uint64_t weights = (uint64_t)AX_KV_B_ROWS * AX_LATENT, nq = (uint64_t)rows * AX_HEADS * AX_NOPE;
    // The restore sits at the plan's offset 0 whatever the slice height.
    if (restore == DS4_AXK2_KV_B_RESTORE) {
        glmvq_q4_restore<__nv_bfloat16><<<(uint32_t)((weights + 255) / 256), 256, 0, stream>>>(w, wb, weights);
    }
    ax_bf16_rows<<<(uint32_t)((nq + 255) / 256), 256, 0, stream>>>(qb, (const float *)q->ptr, AX_NOPE, AX_KEY, nq);
    // Head h: q_abs_h^T [512 x rows] = W_UK_h^T [512 x 128] x q_h^T [128 x rows].
    const float alpha = 1.0f, beta = 0.0f;
    return cuda_ok(cudaGetLastError(), "A.X-K2 absorb operands") &&
        cublas_ok(cublasGemmStridedBatchedEx(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N, AX_LATENT, (int)rows, AX_NOPE,
            &alpha, wb, CUDA_R_16BF, AX_LATENT, (long long)(AX_NOPE + AX_VALUE) * AX_LATENT,
            qb, CUDA_R_16BF, AX_HEADS * AX_NOPE, AX_NOPE, &beta,
            q_abs->ptr, CUDA_R_32F, AX_HEADS * AX_LATENT, AX_LATENT, AX_HEADS,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "A.X-K2 absorb GEMM");
}

extern "C" int ds4_gpu_axk2_value(ds4_gvq_ctx *ctx, ds4_gpu_tensor *heads, const ds4_gpu_tensor *latent_out,
        uint32_t kv_b, const ds4_gpu_tensor *gate, uint32_t rows, ds4_gpu_tensor *scratch) {
    const uint8_t *w = ax_kv_b(ctx, kv_b);
    const uint64_t count = (uint64_t)rows * AX_HEADS * AX_VALUE;
    if (!w || !rows || !gvq_floats(heads, count) || !gvq_floats(gate, count) ||
        !gvq_floats(latent_out, (uint64_t)rows * AX_HEADS * AX_LATENT) || !gvq_stream(ctx)) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    if (rows <= AX_DIRECT_ROWS) {
        auto run = [&](auto nt) {
            glmvq_launch_kernel(gvq_mode(rows), ax_value_kernel<decltype(nt)::value>, AX_HEADS * AX_VALUE / 8, 256, 0,
                stream, (float *)heads->ptr, (const float *)latent_out->ptr, w, (const float *)gate->ptr, rows);
        };
        if (rows == 1) { run(std::integral_constant<uint32_t, 1>{}); }
        else { run(std::integral_constant<uint32_t, AX_DIRECT_ROWS>{}); }
        return cuda_ok(cudaGetLastError(), "A.X-K2 value");
    }
    // kv_b's BF16 restore is the one ds4_gpu_axk2_absorb left in scratch.
    const ax_absorb_layout l = ax_absorb_plan(rows);
    if (!scratch || ds4_gpu_tensor_bytes(scratch) < l.bytes) { return 0; }
    uint8_t *base = (uint8_t *)scratch->ptr;
    auto *wb = (__nv_bfloat16 *)(base + l.kv_b), *lb = (__nv_bfloat16 *)(base + l.lat);
    auto *raw = (float *)(base + l.out);
    const uint64_t nl = (uint64_t)rows * AX_HEADS * AX_LATENT;
    if (ax_latent_bf16(rows)) {
        lb = (__nv_bfloat16 *)latent_out->ptr;
    } else {
        ax_bf16_rows<<<(uint32_t)((nl + 255) / 256), 256, 0, stream>>>(lb, (const float *)latent_out->ptr, AX_LATENT,
            AX_LATENT, nl);
    }
    // Head h: o_h^T [128 x rows] = W_UV_h [128 x 512] x lat_h^T [512 x rows].
    const float alpha = 1.0f, beta = 0.0f;
    if (!cuda_ok(cudaGetLastError(), "A.X-K2 value operands") ||
        !cublas_ok(cublasGemmStridedBatchedEx(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N, AX_VALUE, (int)rows, AX_LATENT,
            &alpha, wb + (uint64_t)AX_NOPE * AX_LATENT, CUDA_R_16BF, AX_LATENT,
            (long long)(AX_NOPE + AX_VALUE) * AX_LATENT, lb, CUDA_R_16BF, AX_HEADS * AX_LATENT, AX_LATENT, &beta,
            raw, CUDA_R_32F, AX_HEADS * AX_VALUE, AX_VALUE, AX_HEADS, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP), "A.X-K2 value GEMM")) { return 0; }
    ax_gate_kernel<<<(uint32_t)((count + 255) / 256), 256, 0, stream>>>((float *)heads->ptr, raw,
        (const float *)gate->ptr, count);
    return cuda_ok(cudaGetLastError(), "A.X-K2 output gate");
}

// Decode rows split their keys over Ling's 48 CTAs; wider rows take one CTA each.
extern "C" uint64_t ds4_gpu_axk2_flash_bytes(void) { return l31_fa_scratch_bytes(); }

extern "C" int ds4_gpu_axk2_flash(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gpu_tensor *partials,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *q_abs, const ds4_gpu_tensor *latent, const ds4_gpu_tensor *k_pe,
        const ds4_gpu_tensor *selected, uint32_t rows, uint32_t pos0, float scale, uint32_t cap) {
    (void)ctx;
    if (!rows || pos0 + rows > cap || !gvq_floats(out, (uint64_t)rows * AX_HEADS * AX_LATENT) ||
        !gvq_floats(q, (uint64_t)rows * AX_HEADS * AX_KEY) || !gvq_floats(q_abs, (uint64_t)rows * AX_HEADS * AX_LATENT) ||
        ds4_gpu_tensor_bytes(latent) < ds4_axk2_latent_bytes(cap) ||
        ds4_gpu_tensor_bytes(k_pe) < (uint64_t)cap * AX_ROPE * 2 ||
        (selected && ds4_gpu_tensor_bytes(selected) < (uint64_t)rows * AX_TOPK * sizeof(uint32_t)) ||
        !partials || ds4_gpu_tensor_bytes(partials) < l31_fa_scratch_bytes()) { return 0; }
    static bool ready = false;
    if (!ready) {
        if (!cuda_ok(cudaFuncSetAttribute(ax_fa_kernel<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, L31_FA8_SMEM),
                "A.X-K2 flash attributes") ||
            !cuda_ok(cudaFuncSetAttribute(ax_fa_kernel<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, L31_FA8_SMEM),
                "A.X-K2 flash attributes") ||
            !cuda_ok(cudaFuncSetAttribute(ax_fa_kernel<__nv_bfloat16, ax_fa_keys::cooperative>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, L31_FA8_SMEM),
                "A.X-K2 flash attributes") ||
            !cuda_ok(cudaFuncSetAttribute(ax_fa_kernel<__nv_bfloat16, ax_fa_keys::staged>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, L31_FA8_SMEM),
                "A.X-K2 flash attributes")) { return 0; }
        ready = true;
    }
    const uint32_t keys = selected ? (uint32_t)AX_TOPK : pos0 + rows;
    const uint32_t tiles = (keys + GLMVQ_FA_KEYS - 1) / GLMVQ_FA_KEYS;
    const uint32_t splits = max(1u, min(tiles, (uint32_t)L31_FA_SPLITS / rows));
    const uint32_t per = (tiles + splits - 1) / splits;
    float *partial = (float *)partials->ptr;
    float *stats = partial + (uint64_t)rows * splits * GLMVQ_FA_HEADS * GLMVQ_FA_LATENT;
    const glmvq_launch mode = gvq_mode(rows);
    const cudaStream_t stream = ds4_current_stream();
    const uint8_t *cache = (const uint8_t *)latent->ptr;
    const int8_t *exps = (const int8_t *)(cache + (uint64_t)cap * AX_LATENT);
    const void *key_data = selected ? selected->ptr : nullptr;
    auto launch = [&](auto kernel, auto *dst) {
        return glmvq_launch_kernel(mode, kernel, dim3(rows, splits), GLMVQ_FA_THREADS, L31_FA8_SMEM, stream,
            dst, splits > 1 ? partial : (float *)nullptr, stats, (const float *)q->ptr,
            (const float *)q_abs->ptr, (const void *)cache, exps,
            (const __nv_bfloat16 *)k_pe->ptr, key_data,
            pos0, splits > 1 ? per : UINT32_MAX, scale);
    };
    static const bool coop_keys = !ax_env_off("DS4_AXK2_FLASH_KEYS");
    static const bool stage_keys = !ax_env_off("DS4_AXK2_FLASH_META");
    cudaError_t err;
    if (!ax_latent_bf16(rows)) {
        err = launch(ax_fa_kernel<float>, (float *)out->ptr);
    } else if (coop_keys && stage_keys && selected) {
        // BF16 output leaves half of this F32-capacity tensor free for metadata.
        auto *meta = (ax_fa_key *)((__nv_bfloat16 *)out->ptr + (uint64_t)rows * AX_HEADS * AX_LATENT);
        ax_fa_pack_keys<<<(rows * AX_TOPK + 255) / 256, 256, 0, stream>>>(meta,
            (const uint32_t *)selected->ptr, exps, rows, pos0);
        if (!cuda_ok(cudaGetLastError(), "A.X-K2 flash key staging")) { return 0; }
        key_data = meta;
        err = launch(ax_fa_kernel<__nv_bfloat16, ax_fa_keys::staged>, (__nv_bfloat16 *)out->ptr);
    } else if (coop_keys) {
        err = launch(ax_fa_kernel<__nv_bfloat16, ax_fa_keys::cooperative>, (__nv_bfloat16 *)out->ptr);
    } else {
        err = launch(ax_fa_kernel<__nv_bfloat16>, (__nv_bfloat16 *)out->ptr);
    }
    if (err == cudaSuccess && splits > 1) {
        err = glmvq_launch_kernel(mode, l31_fa_combine, dim3(rows, GLMVQ_FA_HEADS), GLMVQ_FA_LATENT / 2, 0, stream,
            (float *)out->ptr, (const float *)partial, (const float *)stats, splits);
    }
    return cuda_ok(err, "A.X-K2 flash attention");
}

/* ------------------------------------------------------- projections */

extern "C" int ds4_gpu_axk2_q8(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t weight, const ds4_gpu_tensor *input,
        uint32_t rows) {
    if (!ctx || weight >= ctx->model->source.size() || ctx->model->records[weight].type != AX_TYPE_Q8_0) { return 0; }
    const glmvq_source &s = ctx->model->source[weight];
    if (!rows || s.cols % 256 || !gvq_floats(input, (uint64_t)rows * s.cols) ||
        !gvq_floats(out, (uint64_t)rows * s.rows) || !gvq_stream(ctx)) { return 0; }
    if (rows > 1) {
        // Prefill: MMQ over the Q8_1-quantized input (its rows are F32 per token).
        return ds4_mmq_q8_0_dense(s.data, (const float *)input->ptr, (float *)out->ptr, (int)s.rows, (int)rows,
            (int)s.cols, ds4_mmq_stream_for_call()) == 0 && cuda_ok(cudaGetLastError(), "A.X-K2 Q8_0 MMQ");
    }
    static const bool stream = !ax_env_off("DS4_AXK2_Q8_STREAM");
    const uint32_t warps = stream && s.rows == 8192 && s.cols == 1536 ? 1 : AX_Q8_WARPS;
    const auto kernel = !stream ? ax_q8_gemv<ax_q8_path::reference> :
        warps == 1 ? ax_q8_gemv<ax_q8_path::stream, 1> : ax_q8_gemv<ax_q8_path::stream>;
    // Rows are whole 16-byte pieces (wq_b 1,632 bytes, wk 7,616).
    const uint32_t row_bytes = s.cols / AX_Q8_BLOCK * AX_Q8_BYTES, smem = warps * row_bytes;
    if (row_bytes % 16 || (uintptr_t)s.data % 16 ||
        (smem > 48 * 1024 && !cuda_ok(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem), "A.X-K2 Q8_0 GEMV attributes"))) { return 0; }
    glmvq_launch_kernel(gvq_mode(rows), kernel, (s.rows + warps - 1) / warps, warps * 32, smem,
        ds4_current_stream(), (float *)out->ptr, (const uint8_t *)s.data, (const float *)input->ptr, s.rows, s.cols);
    return cuda_ok(cudaGetLastError(), "A.X-K2 Q8_0 GEMV");
}

__global__ void ax_embed_kernel(float *cur, const __nv_bfloat16 *rows, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { cur[i] = __bfloat162float(rows[i]); }
}

extern "C" int ds4_gpu_axk2_embed(ds4_gpu_tensor *cur, const ds4_gpu_tensor *rows_bf16, uint32_t rows) {
    const uint64_t count = (uint64_t)rows * AX_HIDDEN;
    if (!rows || !gvq_floats(cur, count) || ds4_gpu_tensor_bytes(rows_bf16) < count * 2) { return 0; }
    ax_embed_kernel<<<(uint32_t)((count + 255) / 256), 256, 0, ds4_current_stream()>>>((float *)cur->ptr,
        (const __nv_bfloat16 *)rows_bf16->ptr, count);
    return cuda_ok(cudaGetLastError(), "A.X-K2 embedding rows");
}
