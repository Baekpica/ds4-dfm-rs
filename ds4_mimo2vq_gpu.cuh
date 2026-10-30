/* MiMo-V2.6-Flash VQ routed experts; included once by ds4_cuda.cu after the
 * GVQ adapter (whose context, records and kernels it uses).
 *
 * MiMo's experts are GLM-5.3's shape, so GLM's GVQ1 kernels run in GLM's
 * form except the reduction: routed outputs sum in F32 and round once (the
 * reference's index_add_ into an F32 accumulator), not GLM's per-add BF16.
 * SiLU(gate) and its product with up round to BF16 as the reference's BF16
 * ops do (GLM's gate form with no clamp). Every layer has a K8192 book;
 * R13+8 layers add the 2 KiB GVR8 book, so decode down and the 4-row verify
 * stage 100,352 of GB10's 101,376 B. Each layer's storage decides:
 *
 *   rows == 1     decode: device routing, persistent CTAs keep the book
 *   rows 2..4     verify: one pass over the rows' expert union
 *   rows >= 5     prefill: host grouping, FP8 MMA on the E4M3 codewords */
#include "ds4_mimo2vq.h"
#include "cuda/minimax_mma8w.cuh"

enum { M2VQ_EXPERTS = DS4_MIMO2VQ_EXPERTS, M2VQ_USED = DS4_MIMO2VQ_USED, M2VQ_HIDDEN = GLMVQ_DEC_HIDDEN,
       M2VQ_FFN = GLMVQ_DEC_FFN, M2VQ_TABLE = 3 * M2VQ_EXPERTS, M2VQ_LAYERS = DS4_MIMO2VQ_MOE_LAYERS,
       M2VQ_VERIFY_ROWS = GLMVQ_VER_ROWS, M2VQ_MMA_ROWS = 256, M2VQ_MMA_TOKENS = 64, M2VQ_ALIGN = 256 };
static_assert((uint32_t)M2VQ_USED == (uint32_t)GLMVQ_DEC_USED, "GVQ decode geometry");

using m2vq_form = glmvq_moe_form<M2VQ_FFN, GLMVQ_CLAMP_GATE, GLMVQ_REDUCE_F32, M2VQ_HIDDEN>;

static uint64_t m2vq_align(uint64_t bytes) { return (bytes + M2VQ_ALIGN - 1) / M2VQ_ALIGN * M2VQ_ALIGN; }

// Decode/verify scratch, then the device tables (one 3 x 256 row per layer).
struct m2vq_decode_layout { uint64_t items, mid, partial, vitems, plan, vmid, vpartial, tables, bytes; };

static m2vq_decode_layout m2vq_decode_plan() {
    m2vq_decode_layout l = {};
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += m2vq_align(bytes); return here; };
    l.items = take(sizeof(glmvq_decode_item) * M2VQ_USED);
    l.mid = take((uint64_t)M2VQ_USED * M2VQ_FFN * sizeof(__nv_bfloat16));
    l.partial = take((uint64_t)M2VQ_USED * M2VQ_HIDDEN * sizeof(float));
    l.vitems = take(sizeof(glmvq_verify_item) * GLMVQ_VER_PAIRS);
    l.plan = take(sizeof(glmvq_verify_plan));
    l.vmid = take((uint64_t)GLMVQ_VER_PAIRS * M2VQ_FFN * sizeof(__nv_bfloat16));
    l.vpartial = take((uint64_t)GLMVQ_VER_PAIRS * M2VQ_HIDDEN * sizeof(float));
    l.tables = take((uint64_t)M2VQ_LAYERS * M2VQ_TABLE * sizeof(uint32_t));
    l.bytes = at;
    return l;
}

// Prefill: E4M3 input rows and 1x128 scales, gate rows (BF16), the up
// epilogue's E4M3 SwiGLU rows and scales, down rows (BF16), worklists.
struct m2vq_mma_layout { uint64_t x8, xs, gate, mid8, ms, down, work, order, witems, token, bytes; };

static m2vq_mma_layout m2vq_mma_plan(uint32_t rows) {
    m2vq_mma_layout l = {};
    const uint64_t pairs = (uint64_t)rows * M2VQ_USED;
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += m2vq_align(bytes); return here; };
    l.x8 = take((uint64_t)rows * M2VQ_HIDDEN);
    l.xs = take((uint64_t)rows * (M2VQ_HIDDEN / GLMVQ_MMA8_K) * sizeof(float));
    l.gate = take(pairs * M2VQ_FFN * sizeof(__nv_bfloat16));
    l.mid8 = take(pairs * M2VQ_FFN);
    l.ms = take(pairs * (M2VQ_FFN / GLMVQ_MMA8_K) * sizeof(float));
    l.down = take(pairs * M2VQ_HIDDEN * sizeof(__nv_bfloat16));
    l.work = take(pairs * sizeof(glmvq_mma_item));
    l.order = take(pairs * sizeof(uint32_t));
    // Three projections of at most pairs / MW_BT + one partial item per expert.
    l.witems = take(3 * (pairs / MW_BT + M2VQ_EXPERTS) * sizeof(mw_item));
    l.token = take(pairs * sizeof(uint32_t));
    l.bytes = at;
    return l;
}

struct ds4_mimo2vq_moe {
    ds4_gvq_ctx *gvq = nullptr;     // borrowed: stream, routes, grouped, pairs
    ds4_gpu_tensor *decode = nullptr, *mma = nullptr;
    uint32_t rows = 0, decode_blocks = 0;
    int mma8_blocks[2][3] = {};     // [R13, R13+8][gate, up, down]
    bool gain_dot = true, mma8 = false, mmaw = true;
    std::vector<uint32_t> keys;     // table slot per layer, keyed by expert 0's gate
    std::vector<glmvq_mma_item> items;
    std::vector<mw_item> witems;
    std::vector<uint32_t> order;
    std::vector<uint8_t> token_used;
};

extern "C" uint64_t ds4_gpu_mimo2vq_moe_bytes(uint32_t rows) {
    return m2vq_decode_plan().bytes + (rows > M2VQ_VERIFY_ROWS ? m2vq_mma_plan(rows).bytes : 0u);
}

extern "C" void ds4_gpu_mimo2vq_moe_free(ds4_mimo2vq_moe *moe) {
    if (!moe) { return; }
    ds4_gpu_tensor_free(moe->decode);
    ds4_gpu_tensor_free(moe->mma);
    delete moe;
}

template <bool RES>
static cudaError_t m2vq_prepare(ds4_mimo2vq_moe *moe) {
    uint32_t blocks = 0;
    cudaError_t err = glmvq_decode_prepare<13, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_DOT, m2vq_form, RES>(&blocks);
    if (err == cudaSuccess) {
        err = glmvq_decode_prepare<13, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_WEIGHT, m2vq_form, RES>(&blocks);
    }
    if (err != cudaSuccess) { return err; }
    moe->decode_blocks = blocks;
    int *mma = moe->mma8_blocks[RES];
    err = glmvq_mma8_prepare<M2VQ_MMA_ROWS, M2VQ_MMA_TOKENS, 1, GLMVQ_MMA_GATE, 13, GLMVQ_CLAMP_GATE, RES>(&mma[0]);
    if (err == cudaSuccess) {
        err = glmvq_mma8_prepare<M2VQ_MMA_ROWS, M2VQ_MMA_TOKENS, 1, GLMVQ_MMA_UP, 13, GLMVQ_CLAMP_GATE, RES>(&mma[1]);
    }
    if (err == cudaSuccess) {
        err = glmvq_mma8_prepare<M2VQ_MMA_ROWS, M2VQ_MMA_TOKENS, 1, GLMVQ_MMA_DOWN, 13, GLMVQ_CLAMP_GATE, RES>(&mma[2]);
    }
    return err;
}

extern "C" ds4_mimo2vq_moe *ds4_gpu_mimo2vq_moe_create(ds4_gvq_ctx *gvq, uint32_t rows) {
    if (!gvq || !rows) { return nullptr; }
    ds4_mimo2vq_moe *moe = new (std::nothrow) ds4_mimo2vq_moe;
    if (!moe) { return nullptr; }
    moe->gvq = gvq;
    moe->rows = rows;
    const char *gain = getenv("DS4_MIMO2VQ_GAIN_DOT");
    moe->gain_dot = !gain || strcmp(gain, "0");
    // DS4_MIMO2VQ_MMAW=0 keeps glmvq_mma8_kernel (bit-identical rows).
    const char *mmaw = getenv("DS4_MIMO2VQ_MMAW");
    moe->mmaw = !mmaw || strcmp(mmaw, "0");
    const cudaError_t r13 = m2vq_prepare<false>(moe), r13p8 = m2vq_prepare<true>(moe);
    if (r13 != cudaSuccess || r13p8 != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: MiMo VQ expert kernels unavailable (%s)\n",
            cudaGetErrorString(r13 != cudaSuccess ? r13 : r13p8));
        ds4_gpu_mimo2vq_moe_free(moe);
        return nullptr;
    }
    moe->mma8 = rows > M2VQ_VERIFY_ROWS;
    moe->decode = ds4_gpu_tensor_alloc(m2vq_decode_plan().bytes);
    moe->mma = moe->mma8 ? ds4_gpu_tensor_alloc(m2vq_mma_plan(rows).bytes) : nullptr;
    if (!moe->decode || (moe->mma8 && !moe->mma)) {
        ds4_gpu_mimo2vq_moe_free(moe);
        return nullptr;
    }
    moe->items.reserve((uint64_t)rows * M2VQ_USED);
    moe->order.resize((uint64_t)rows * M2VQ_USED);
    moe->token_used.resize(rows);
    return moe;
}

// A layer's residual book, or nullptr for an R13 layer.
static const uint8_t *m2vq_res_book(const ds4_mimo2vq_moe *moe, const uint32_t *matrices) {
    return moe->gvq->model->source[matrices[0]].vq.res_book;
}

// Shape, books and alignment the direct kernels read; every one of a layer's
// 768 matrices shares its books (GLM's gvq_decode_matrix).
static bool m2vq_matrix_fits(const glmvq_source &s, uint32_t projection, const void *book,
        const uint8_t *res_book) {
    const uint32_t rows = projection == 2 ? M2VQ_HIDDEN : M2VQ_FFN;
    const uint32_t cols = projection == 2 ? M2VQ_FFN : M2VQ_HIDDEN;
    const glmvq_matrix &m = s.vq;
    const bool res = res_book ? m.res && m.res_book == res_book && (uintptr_t)m.res % 128 == 0 &&
        (uintptr_t)res_book % 16 == 0 : !m.res;
    return s.storage == GLMVQ_WEIGHT_VQ && s.rows == rows && s.cols == cols &&
        m.rows == rows && m.cols == cols && m.precision == GLMVQ_E4M3 &&
        m.main && m.plane && m.gain && book && m.book == book &&
        (uintptr_t)m.main % 128 == 0 && (uintptr_t)m.plane % 128 == 0 &&
        (uintptr_t)book % 16 == 0 && (uintptr_t)m.gain % sizeof(__half) == 0 && res;
}

// The layer's device table, validated and uploaded on first use.
static const uint32_t *m2vq_table(ds4_mimo2vq_moe *moe, const uint32_t *matrices, const uint8_t *res_book) {
    const auto &sources = moe->gvq->model->source;
    const m2vq_decode_layout l = m2vq_decode_plan();
    const auto *tables = (const uint32_t *)((const uint8_t *)moe->decode->ptr + l.tables);
    for (size_t i = 0; i < moe->keys.size(); i++) {
        if (moe->keys[i] == matrices[0]) { return tables + i * M2VQ_TABLE; }
    }
    if (moe->keys.size() == M2VQ_LAYERS) { return nullptr; }
    const void *book = sources[matrices[0]].vq.book;
    for (uint32_t i = 0; i < M2VQ_TABLE; i++) {
        if (matrices[i] >= sources.size() ||
            !m2vq_matrix_fits(sources[matrices[i]], i / M2VQ_EXPERTS, book, res_book)) {
            fprintf(stderr, "ds4: MiMo VQ expert matrix %u misses the GVQ1 layout\n", i);
            return nullptr;
        }
    }
    const size_t slot = moe->keys.size();
    if (!ds4_gpu_tensor_write(moe->decode, l.tables + slot * M2VQ_TABLE * sizeof(uint32_t),
            matrices, M2VQ_TABLE * sizeof(uint32_t))) { return nullptr; }
    moe->keys.push_back(matrices[0]);
    return tables + slot * M2VQ_TABLE;
}

// N=1 routing on the GPU: thread t places expert selected[t] at its rank, so
// items run in ascending expert order with no host sync.
__global__ void m2vq_decode_route(glmvq_decode_item *items, const int32_t *selected,
        const uint32_t *table, const glmvq_source *sources) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x;
    const uint32_t e = min((uint32_t)selected[t], (uint32_t)M2VQ_EXPERTS - 1u);
    uint32_t rank = 0;
    for (uint32_t j = 0; j < M2VQ_USED; j++) { rank += (uint32_t)selected[j] < (uint32_t)selected[t]; }
    items[rank] = {sources[table[e]].vq, sources[table[M2VQ_EXPERTS + e]].vq,
        sources[table[2 * M2VQ_EXPERTS + e]].vq, t};
}

// Union of 2-4 rows' experts in ascending expert order (gvq_verify_route);
// down groups hold at most `capacity` pairs' mid rows.
__global__ void m2vq_verify_route(glmvq_verify_item *items, glmvq_verify_plan *plan,
        const int32_t *selected, uint32_t rows, const uint32_t *table, const glmvq_source *sources,
        uint32_t capacity) {
    __shared__ uint32_t expert[GLMVQ_VER_PAIRS], lead[GLMVQ_VER_PAIRS], width[GLMVQ_VER_PAIRS];
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x, pairs = rows * M2VQ_USED;
    const bool live = t < pairs;
    if (live) { expert[t] = min((uint32_t)selected[t], (uint32_t)M2VQ_EXPERTS - 1u); }
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
            items[rank].up = sources[table[M2VQ_EXPERTS + e]].vq;
            items[rank].down = sources[table[2 * M2VQ_EXPERTS + e]].vq;
            items[rank].count = same;
            width[rank] = same;
        }
        items[rank].pair[before] = t;
        const uint32_t row = t / M2VQ_USED;
        for (uint32_t k = 0; k < M2VQ_USED; k++) { below += expert[row * M2VQ_USED + k] < e; }
        plan->order[row * M2VQ_USED + below] = t % M2VQ_USED;
    }
    __syncthreads();
    if (t) { return; }
    uint32_t n = 0;
    for (uint32_t j = 0; j < pairs; j++) { n += lead[j]; }
    // Greedy groups at item boundaries: an item (<= rows pairs) never splits.
    uint32_t groups = 0, used = capacity;
    for (uint32_t i = 0; i < n; i++) {
        if (used + width[i] > capacity) { plan->group[groups++] = i; used = 0; }
        items[i].slot = used;
        used += width[i];
    }
    plan->group[groups] = n;
    plan->items = n;
    plan->groups = groups;
}

template <bool RES>
static cudaError_t m2vq_decode_run(ds4_mimo2vq_moe *moe, float *out, const float *input, const int32_t *selected,
        const float *weights, const uint32_t *table, const void *book, const uint8_t *res_book) {
    const m2vq_decode_layout l = m2vq_decode_plan();
    uint8_t *base = (uint8_t *)moe->decode->ptr;
    auto *items = (glmvq_decode_item *)(base + l.items);
    auto *mid = (__nv_bfloat16 *)(base + l.mid);
    auto *partial = (float *)(base + l.partial);
    const cudaStream_t stream = ds4_current_stream();
    const glmvq_launch launch = moe->gvq->launch;
    glmvq_launch_kernel(launch, m2vq_decode_route, 1, M2VQ_USED, 0, stream, items, selected, table,
        (const glmvq_source *)moe->gvq->model->device->ptr);
    return moe->gain_dot
        ? glmvq_decode_launch<13, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_DOT, m2vq_form, RES>(items, book, input, mid,
            partial, out, weights, INFINITY, moe->decode_blocks, stream, nullptr, nullptr, nullptr, launch, res_book)
        : glmvq_decode_launch<13, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_WEIGHT, m2vq_form, RES>(items, book, input, mid,
            partial, out, weights, INFINITY, moe->decode_blocks, stream, nullptr, nullptr, nullptr, launch, res_book);
}

// Each verify row's arithmetic is the one-row kernel's, so the union pass
// changes no value.
template <bool RES>
static cudaError_t m2vq_verify_run(ds4_mimo2vq_moe *moe, float *out, const float *input, const int32_t *selected,
        const float *weights, const uint32_t *table, const void *book, const uint8_t *res_book, uint32_t rows) {
    const m2vq_decode_layout l = m2vq_decode_plan();
    uint8_t *base = (uint8_t *)moe->decode->ptr;
    auto *items = (glmvq_verify_item *)(base + l.vitems);
    auto *plan = (glmvq_verify_plan *)(base + l.plan);
    auto *mid = (__nv_bfloat16 *)(base + l.vmid);
    auto *partial = (float *)(base + l.vpartial);
    const cudaStream_t stream = ds4_current_stream();
    const glmvq_launch launch = moe->gvq->launch;
    glmvq_launch_kernel(launch, m2vq_verify_route, 1, GLMVQ_VER_PAIRS, 0, stream, items, plan, selected, rows,
        table, (const glmvq_source *)moe->gvq->model->device->ptr, (uint32_t)m2vq_form::VER_GROUP);
    return moe->gain_dot
        ? glmvq_verify_launch<13, 32, GLMVQ_GAIN_DOT, m2vq_form, RES>(items, plan, book, input, rows, mid, partial,
            out, weights, INFINITY, moe->decode_blocks, stream, nullptr, launch, res_book)
        : glmvq_verify_launch<13, 32, GLMVQ_GAIN_WEIGHT, m2vq_form, RES>(items, plan, book, input, rows, mid,
            partial, out, weights, INFINITY, moe->decode_blocks, stream, nullptr, launch, res_book);
}

// Host grouping by expert (ascending), as the GLM prefill path builds it.
static bool m2vq_group(ds4_mimo2vq_moe *moe, const ds4_gpu_tensor *selected, uint32_t rows, uint32_t *offsets) {
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * M2VQ_USED;
    if (pairs > ctx->routes.size() ||
        !ds4_gpu_tensor_read(selected, 0, ctx->routes.data(), pairs * sizeof(int32_t))) { return false; }
    std::vector<uint32_t> counts(M2VQ_EXPERTS, 0), cursor(M2VQ_EXPERTS);
    for (uint64_t i = 0; i < pairs; i++) {
        const int32_t expert = ctx->routes[i];
        if (expert < 0 || expert >= M2VQ_EXPERTS) { return false; }
        for (uint64_t j = i - i % M2VQ_USED; j < i; j++) {
            if (ctx->routes[j] == expert) { return false; }
        }
        counts[expert]++;
    }
    offsets[0] = 0;
    for (uint32_t e = 0; e < M2VQ_EXPERTS; e++) {
        offsets[e + 1] = offsets[e] + counts[e];
        cursor[e] = offsets[e];
    }
    for (uint64_t i = 0; i < pairs; i++) { ctx->grouped[cursor[ctx->routes[i]]++] = (uint32_t)i; }
    return ds4_gpu_tensor_write(ctx->pairs, 0, ctx->grouped.data(), pairs * sizeof(uint32_t)) != 0;
}

// Each grouped pair's token row (pair index / top-k).
__global__ static void m2vq_pair_tokens(uint32_t *token_of, const uint32_t *map, uint32_t pairs) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < pairs) { token_of[p] = map[p] / M2VQ_USED; }
}

// Prefill experts on MiniMax's warp-specialized kernels: 64-pair items per
// projection, tokens quantized once (E4M3, 1x128 scales, ldmatrix k order).
// Gate, up and down rows equal glmvq_mma8_kernel's bit for bit.
template <bool RES>
static int m2vq_mmaw_run(ds4_mimo2vq_moe *moe, float *out, const float *input, const float *weights,
        const uint32_t *matrices, const uint32_t *offsets, uint32_t rows, const uint8_t *res_book) {
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * M2VQ_USED, count = (uint64_t)rows * M2VQ_HIDDEN;
    const m2vq_mma_layout l = m2vq_mma_plan(rows);
    uint8_t *base = (uint8_t *)moe->mma->ptr;
    auto *x8 = base + l.x8, *mid8 = base + l.mid8;
    auto *xs = (float *)(base + l.xs), *ms = (float *)(base + l.ms);
    auto *gate = (__nv_bfloat16 *)(base + l.gate), *down = (__nv_bfloat16 *)(base + l.down);
    auto *order = (uint32_t *)(base + l.order), *token = (uint32_t *)(base + l.token);
    auto *witems = (mw_item *)(base + l.witems);
    moe->witems.clear();
    for (uint32_t m = 0; m < 3; m++) {
        for (uint32_t e = 0; e < M2VQ_EXPERTS; e++) {
            const glmvq_matrix &vq = ctx->model->source[matrices[m * M2VQ_EXPERTS + e]].vq;
            for (uint32_t start = offsets[e]; start < offsets[e + 1]; start += MW_BT) {
                moe->witems.push_back({start, min((uint32_t)MW_BT, offsets[e + 1] - start), vq.gain, vq.main,
                    vq.plane, vq.res, 0});
            }
        }
    }
    std::fill(moe->token_used.begin(), moe->token_used.begin() + rows, 0);
    for (uint32_t i = 0; i < pairs; i++) {
        const uint32_t t = ctx->grouped[i] / M2VQ_USED;
        moe->order[t * M2VQ_USED + moe->token_used[t]++] = i;
    }
    const cudaStream_t stream = ds4_current_stream();
    if (!cuda_ok(cudaMemcpyAsync(witems, moe->witems.data(), moe->witems.size() * sizeof(mw_item),
            cudaMemcpyHostToDevice, stream), "MiMo VQ warp-specialized items") ||
        !cuda_ok(cudaMemcpyAsync(order, moe->order.data(), pairs * sizeof(*order),
            cudaMemcpyHostToDevice, stream), "MiMo VQ MMA order")) { return 0; }
    const auto *book = (const uint8_t *)ctx->model->source[matrices[0]].vq.book;
    const auto *grouped = (const uint32_t *)ctx->pairs->ptr;
    const uint32_t n = (uint32_t)moe->witems.size() / 3, sms = moe->decode_blocks;
    const uint64_t xb = count / GLMVQ_MMA8_K;
    mw_quant<<<(uint32_t)((xb * 32 + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0, stream>>>(
        x8, xs, input, xb);
    m2vq_pair_tokens<<<(uint32_t)((pairs + 255) / 256), 256, 0, stream>>>(token, grouped, (uint32_t)pairs);
    // Up's BF16 rows borrow down's buffer (twice their size) until
    // mw_swiglu_quant has read them.
    cudaError_t err = mw_launch<GLMVQ_MMA_GATE, RES>(witems, n, book, res_book, x8, xs, token, gate, nullptr,
        M2VQ_FFN, M2VQ_HIDDEN, sms, stream);
    if (err == cudaSuccess) {
        err = mw_launch<GLMVQ_MMA_GATE, RES>(witems + n, n, book, res_book, x8, xs, token, down, nullptr,
            M2VQ_FFN, M2VQ_HIDDEN, sms, stream);
    }
    if (err == cudaSuccess) {
        const uint64_t warps = pairs * (M2VQ_FFN / GLMVQ_MMA8_K);
        mw_swiglu_quant<GLMVQ_CLAMP_GATE><<<(uint32_t)((warps * 32 + GLMVQ_THREADS - 1) / GLMVQ_THREADS),
            GLMVQ_THREADS, 0, stream>>>(mid8, ms, gate, down, M2VQ_FFN, (uint32_t)pairs, INFINITY);
        err = mw_launch<GLMVQ_MMA_DOWN, RES>(witems + 2 * n, n, book, res_book, mid8, ms, token, nullptr, down,
            M2VQ_HIDDEN, M2VQ_FFN, sms, stream);
    }
    if (!cuda_ok(err, "MiMo VQ warp-specialized experts")) { return 0; }
    glmvq_mma_reduce<__nv_bfloat16, GLMVQ_REDUCE_F32><<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS,
        GLMVQ_THREADS, 0, stream>>>(out, down, weights, grouped, order, M2VQ_USED, M2VQ_HIDDEN, count);
    return cuda_ok(cudaGetLastError(), "MiMo VQ FP8 warp-specialized MMA");
}

template <bool RES>
static int m2vq_mma_run(ds4_mimo2vq_moe *moe, float *out, const float *input, const float *weights,
        const uint32_t *matrices, const uint32_t *offsets, uint32_t rows, const uint8_t *res_book) {
    if (moe->mmaw) { return m2vq_mmaw_run<RES>(moe, out, input, weights, matrices, offsets, rows, res_book); }
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * M2VQ_USED, count = (uint64_t)rows * M2VQ_HIDDEN;
    const m2vq_mma_layout l = m2vq_mma_plan(rows);
    uint8_t *base = (uint8_t *)moe->mma->ptr;
    auto *x8 = base + l.x8, *mid8 = base + l.mid8;
    auto *xs = (float *)(base + l.xs), *ms = (float *)(base + l.ms);
    auto *gate = (__nv_bfloat16 *)(base + l.gate), *down = (__nv_bfloat16 *)(base + l.down);
    auto *work = (glmvq_mma_item *)(base + l.work);
    auto *order = (uint32_t *)(base + l.order);
    moe->items.clear();
    for (uint32_t e = 0; e < M2VQ_EXPERTS; e++) {
        for (uint32_t start = offsets[e]; start < offsets[e + 1]; start += M2VQ_MMA_TOKENS) {
            moe->items.push_back({{matrices[e], matrices[M2VQ_EXPERTS + e], matrices[2 * M2VQ_EXPERTS + e]},
                start, min((uint32_t)M2VQ_MMA_TOKENS, offsets[e + 1] - start)});
        }
    }
    std::fill(moe->token_used.begin(), moe->token_used.begin() + rows, 0);
    for (uint32_t i = 0; i < pairs; i++) {
        const uint32_t token = ctx->grouped[i] / M2VQ_USED;
        moe->order[token * M2VQ_USED + moe->token_used[token]++] = i;
    }
    const cudaStream_t stream = ds4_current_stream();
    if (!cuda_ok(cudaMemcpyAsync(work, moe->items.data(), moe->items.size() * sizeof(*work),
            cudaMemcpyHostToDevice, stream), "MiMo VQ MMA work") ||
        !cuda_ok(cudaMemcpyAsync(order, moe->order.data(), pairs * sizeof(*order),
            cudaMemcpyHostToDevice, stream), "MiMo VQ MMA order")) { return 0; }
    const auto *source = (const glmvq_source *)ctx->model->device->ptr;
    const auto *book = (const uint8_t *)ctx->model->source[matrices[0]].vq.book;
    const auto *grouped = (const uint32_t *)ctx->pairs->ptr;
    const uint32_t nitems = (uint32_t)moe->items.size();
    const int *blocks = moe->mma8_blocks[RES];
    // Tokens quantize once (E4M3, 1x128 scales) for their eight experts.
    const uint64_t xb = count / GLMVQ_MMA8_K;
    glmvq_mma8_quant<float><<<(uint32_t)((xb * 32 + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0,
        stream>>>(x8, xs, input, xb);
    auto launch = [&](auto mode, const uint8_t *in, const float *scale, uint32_t rows_out, uint32_t cols_in,
            uint8_t *q8, float *q8_scale) {
        constexpr glmvq_mma_stage MODE = decltype(mode)::value;
        constexpr uint32_t shared8 = glmvq_mma8_bytes<M2VQ_MMA_ROWS, M2VQ_MMA_TOKENS, 13, RES>();
        constexpr uint32_t threads8 = glmvq_mma8_threads<M2VQ_MMA_ROWS, 1>();
        glmvq_mma8_kernel<M2VQ_MMA_ROWS, M2VQ_MMA_TOKENS, 1, MODE, 13, GLMVQ_CLAMP_GATE, RES><<<
            min(nitems * (rows_out / M2VQ_MMA_ROWS), (uint32_t)blocks[MODE]), threads8, shared8, stream>>>(
            source, work, nitems, book, in, scale, grouped, M2VQ_USED, gate, nullptr, down,
            rows_out, cols_in, INFINITY, q8, q8_scale, res_book);
    };
    launch(std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_GATE>{}, x8, xs, M2VQ_FFN, M2VQ_HIDDEN,
        nullptr, nullptr);
    // The up epilogue writes down's E4M3 rows and scales directly.
    launch(std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_UP>{}, x8, xs, M2VQ_FFN, M2VQ_HIDDEN, mid8, ms);
    launch(std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_DOWN>{}, mid8, ms, M2VQ_HIDDEN, M2VQ_FFN,
        nullptr, nullptr);
    glmvq_mma_reduce<__nv_bfloat16, GLMVQ_REDUCE_F32><<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS,
        GLMVQ_THREADS, 0, stream>>>(out, down, weights, grouped, order, M2VQ_USED, M2VQ_HIDDEN, count);
    return cuda_ok(cudaGetLastError(), "MiMo VQ FP8 grouped MMA");
}

extern "C" int ds4_gpu_mimo2vq_moe(ds4_mimo2vq_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, const uint32_t *matrices, uint32_t rows) {
    if (!moe || !rows || rows > moe->rows || !matrices || !selected || !gvq_stream(moe->gvq)) { return 0; }
    const uint64_t pairs = (uint64_t)rows * M2VQ_USED, count = (uint64_t)rows * M2VQ_HIDDEN;
    if (pairs > selected->bytes / sizeof(int32_t) || !gvq_floats(weights, pairs) ||
        !gvq_floats(out, count) || !gvq_floats(input, count) ||
        matrices[0] >= moe->gvq->model->source.size() || ds4_capture_active()) { return 0; }
    const uint8_t *res_book = m2vq_res_book(moe, matrices);
    const uint32_t *table = m2vq_table(moe, matrices, res_book);
    if (!table) { return 0; }
    const void *book = moe->gvq->model->source[matrices[0]].vq.book;
    float *y = (float *)out->ptr;
    const float *x = (const float *)input->ptr;
    const auto *w = (const float *)weights->ptr;
    if (rows <= M2VQ_VERIFY_ROWS) {
        const auto *sel = (const int32_t *)selected->ptr;
        const cudaError_t err = rows == 1
            ? (res_book ? m2vq_decode_run<true>(moe, y, x, sel, w, table, book, res_book)
                        : m2vq_decode_run<false>(moe, y, x, sel, w, table, book, res_book))
            : (res_book ? m2vq_verify_run<true>(moe, y, x, sel, w, table, book, res_book, rows)
                        : m2vq_verify_run<false>(moe, y, x, sel, w, table, book, res_book, rows));
        return cuda_ok(err, "MiMo VQ routed decode");
    }
    std::vector<uint32_t> offsets(M2VQ_EXPERTS + 1);
    if (!moe->mma8 || !moe->mma || !m2vq_group(moe, selected, rows, offsets.data())) { return 0; }
    return res_book ? m2vq_mma_run<true>(moe, y, x, w, matrices, offsets.data(), rows, res_book)
                    : m2vq_mma_run<false>(moe, y, x, w, matrices, offsets.data(), rows, res_book);
}
