/* K2-Horizon-375B-A23B hVQ device runtime; included once by ds4_cuda.cu after
 * the GVQ adapter (context, records, MMQ), MiniMax's runtime (NVFP4 encoder,
 * warp-specialized 13-bit prefill experts) and A.X-K2's.
 *
 * Routed experts run GLM's GVQ1 kernels in K2's form: 6144 -> 1792 -> 6144
 * rows (three whole 2048-input blocks; 1792 is seven k-steps), SiLU SwiGLU
 * with each step rounded to BF16, and the reference's BF16 index_add_: a BF16
 * running sum in ascending expert order, then the shared expert:
 *
 *   rows == 1     decode: device routing, persistent CTAs keep the book
 *   rows 2..4     verify kernels: one pass over the rows' expert union
 *   rows >= 5     prefill: host grouping, MiniMax's warp-specialized FP8
 *                 MMA on the E4M3 codewords (64-pair items)
 *
 * The rest follows tools/vq_calib/k2h_ref.py: Q8_0 projections, RMSNorm in
 * F32 returned as BF16, partial RoPE in BF16 tensor arithmetic, NVFP4 K/V
 * and the SDPA attention of cuda/k2h_attn.cuh. */
#include "ds4_k2h.h"
#include "cuda/k2h_attn.cuh"

enum { K2H_EXPERTS = DS4_K2H_EXPERTS, K2H_USED = DS4_K2H_USED, K2H_FFN = DS4_K2H_FFN,
       K2H_HIDDEN = DS4_K2H_HIDDEN, K2H_TABLE = 3 * K2H_EXPERTS, K2H_MOE_LAYERS = 58,
       K2H_VERIFY_ROWS = GLMVQ_VER_ROWS, K2H_ALIGN = 256, K2H_THREADS = 256,
       K2H_Q8_BLOCK = DS4_K2H_Q8_BLOCK, K2H_Q8_BYTES = DS4_K2H_Q8_BYTES, K2H_TYPE_Q8_0 = 8 };
static_assert((uint32_t)K2H_USED == (uint32_t)GLMVQ_DEC_USED, "GVQ decode geometry");
static_assert(K2H_HIDDEN % K2H_THREADS == 0, "a CTA covers a hidden row");
static_assert((uint32_t)DS4_K2H_SPLITS == (uint32_t)INK2A_SPLITS, "decode partial geometry");
static_assert(K2H_FFN % MW_BM == 0 && K2H_HIDDEN % MW_BM == 0, "warp-specialized row tiles");

// Verify gate/up stages each row's 6,144-wide input beside the 64 KiB book:
// two rows fit GB10's 99 KiB (88 KiB), four would not (112 KiB).
enum { K2H_VER_ROWS = 2 };
using k2h_form = glmvq_moe_form<K2H_FFN, GLMVQ_CLAMP_GATE, GLMVQ_REDUCE_BF16, K2H_HIDDEN, GLMVQ_DEC_USED,
    K2H_VER_ROWS>;

static uint64_t k2h_align(uint64_t bytes) { return (bytes + K2H_ALIGN - 1) / K2H_ALIGN * K2H_ALIGN; }

static bool k2h_env_off(const char *name) {
    const char *value = getenv(name);
    return value && !strcmp(value, "0");
}

/* ------------------------------------------------------------------ experts */

// Decode/verify scratch, then the device tables (one 3 x 192 row per layer).
struct k2h_decode_layout { uint64_t items, mid, partial, vitems, plan, vmid, vpartial, tables, bytes; };

static k2h_decode_layout k2h_decode_plan() {
    k2h_decode_layout l = {};
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += k2h_align(bytes); return here; };
    l.items = take(sizeof(glmvq_decode_item) * K2H_USED);
    l.mid = take((uint64_t)K2H_USED * K2H_FFN * sizeof(__nv_bfloat16));
    l.partial = take((uint64_t)K2H_USED * K2H_HIDDEN * sizeof(float));
    l.vitems = take(sizeof(glmvq_verify_item) * GLMVQ_VER_PAIRS);
    l.plan = take(sizeof(glmvq_verify_plan));
    l.vmid = take((uint64_t)GLMVQ_VER_PAIRS * K2H_FFN * sizeof(__nv_bfloat16));
    l.vpartial = take((uint64_t)GLMVQ_VER_PAIRS * K2H_HIDDEN * sizeof(float));
    l.tables = take((uint64_t)K2H_MOE_LAYERS * K2H_TABLE * sizeof(uint32_t));
    l.bytes = at;
    return l;
}

// Prefill: E4M3 input rows and their 1x128 scales, gate rows (BF16), up rows
// (BF16, in down's buffer until the SwiGLU pass reads them), down's E4M3
// input and scales, down rows (BF16), 64-pair items, each pair's token and
// the per-token order of its pairs.
struct k2h_mma_layout { uint64_t x8, xs, gate, mid8, ms, down, order, witems, token, bytes; };

static k2h_mma_layout k2h_mma_plan(uint32_t rows) {
    k2h_mma_layout l = {};
    const uint64_t pairs = (uint64_t)rows * K2H_USED;
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += k2h_align(bytes); return here; };
    l.x8 = take((uint64_t)rows * K2H_HIDDEN);
    l.xs = take((uint64_t)rows * (K2H_HIDDEN / GLMVQ_MMA8_K) * sizeof(float));
    l.gate = take(pairs * K2H_FFN * sizeof(__nv_bfloat16));
    l.mid8 = take(pairs * K2H_FFN);
    l.ms = take(pairs * (K2H_FFN / GLMVQ_MMA8_K) * sizeof(float));
    l.down = take(pairs * K2H_HIDDEN * sizeof(__nv_bfloat16));
    l.order = take(pairs * sizeof(uint32_t));
    l.witems = take(3 * (K2H_EXPERTS + pairs / MW_BT) * sizeof(mw_item));
    l.token = take(pairs * sizeof(uint32_t));
    l.bytes = at;
    return l;
}

struct ds4_k2h_moe {
    ds4_gvq_ctx *gvq = nullptr;     // borrowed: stream, routes, grouped, pairs
    ds4_gpu_tensor *decode = nullptr, *mma = nullptr;
    uint32_t rows = 0, decode_blocks = 0, sms = 0;
    bool gain_dot = true;
    std::vector<uint32_t> keys;     // table slot per layer, keyed by expert 0's gate
    std::vector<mw_item> witems;
    std::vector<uint32_t> order;
    std::vector<uint8_t> token_used;
};

extern "C" uint64_t ds4_gpu_k2h_moe_bytes(uint32_t rows) {
    return k2h_decode_plan().bytes + (rows > K2H_VERIFY_ROWS ? k2h_mma_plan(rows).bytes : 0u);
}

extern "C" void ds4_gpu_k2h_moe_free(ds4_k2h_moe *moe) {
    if (!moe) { return; }
    ds4_gpu_tensor_free(moe->decode);
    ds4_gpu_tensor_free(moe->mma);
    delete moe;
}

extern "C" ds4_k2h_moe *ds4_gpu_k2h_moe_create(ds4_gvq_ctx *gvq, uint32_t rows) {
    if (!gvq || !rows) { return nullptr; }
    ds4_k2h_moe *moe = new (std::nothrow) ds4_k2h_moe;
    if (!moe) { return nullptr; }
    moe->gvq = gvq;
    moe->rows = rows;
    // DS4_K2H_GAIN_DOT=0: decode rounds gain x codeword to BF16 per weight
    // (the reference's restored weights) instead of one multiply per dot.
    moe->gain_dot = !k2h_env_off("DS4_K2H_GAIN_DOT");
    int device = 0, sms = 0;
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess) {
        moe->sms = (uint32_t)sms;
    }
    uint32_t blocks = 0;
    cudaError_t err = glmvq_decode_prepare<13, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_DOT, k2h_form>(&blocks);
    if (err == cudaSuccess) {
        err = glmvq_decode_prepare<13, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_WEIGHT, k2h_form>(&blocks);
    }
    if (err != cudaSuccess || !moe->sms) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: K2-Horizon expert kernels unavailable (%s)\n", cudaGetErrorString(err));
        ds4_gpu_k2h_moe_free(moe);
        return nullptr;
    }
    moe->decode_blocks = blocks;
    moe->decode = ds4_gpu_tensor_alloc(k2h_decode_plan().bytes);
    moe->mma = rows > K2H_VERIFY_ROWS ? ds4_gpu_tensor_alloc(k2h_mma_plan(rows).bytes) : nullptr;
    if (!moe->decode || (rows > K2H_VERIFY_ROWS && !moe->mma)) {
        ds4_gpu_k2h_moe_free(moe);
        return nullptr;
    }
    moe->order.resize((uint64_t)rows * K2H_USED);
    moe->token_used.resize(rows);
    return moe;
}

// Shape, book and alignment the direct kernels read: every one of a layer's
// 576 matrices shares its 13-bit book and has a thirteenth-bit plane.
static bool k2h_matrix_fits(const glmvq_source &s, uint32_t projection, const void *book) {
    const uint32_t rows = projection == 2 ? K2H_HIDDEN : K2H_FFN;
    const uint32_t cols = projection == 2 ? K2H_FFN : K2H_HIDDEN;
    const glmvq_matrix &m = s.vq;
    return s.storage == GLMVQ_WEIGHT_VQ && s.rows == rows && s.cols == cols &&
        m.rows == rows && m.cols == cols && m.precision == GLMVQ_E4M3 &&
        m.main && m.gain && m.plane && book && m.book == book && !m.res &&
        (uintptr_t)m.main % 128 == 0 && (uintptr_t)m.plane % 16 == 0 && (uintptr_t)book % 16 == 0 &&
        (uintptr_t)m.gain % sizeof(__half) == 0;
}

// The layer's device table, validated and uploaded on first use.
static const uint32_t *k2h_table(ds4_k2h_moe *moe, const uint32_t *matrices) {
    const auto &sources = moe->gvq->model->source;
    const k2h_decode_layout l = k2h_decode_plan();
    const auto *tables = (const uint32_t *)((const uint8_t *)moe->decode->ptr + l.tables);
    for (size_t i = 0; i < moe->keys.size(); i++) {
        if (moe->keys[i] == matrices[0]) { return tables + i * K2H_TABLE; }
    }
    if (moe->keys.size() == K2H_MOE_LAYERS) { return nullptr; }
    const void *book = sources[matrices[0]].vq.book;
    for (uint32_t i = 0; i < K2H_TABLE; i++) {
        if (matrices[i] >= sources.size() || !k2h_matrix_fits(sources[matrices[i]], i / K2H_EXPERTS, book)) {
            fprintf(stderr, "ds4: K2-Horizon expert matrix %u misses the GVQ1 layout\n", i);
            return nullptr;
        }
    }
    const size_t slot = moe->keys.size();
    if (!ds4_gpu_tensor_write(moe->decode, l.tables + slot * K2H_TABLE * sizeof(uint32_t),
            matrices, K2H_TABLE * sizeof(uint32_t))) { return nullptr; }
    moe->keys.push_back(matrices[0]);
    return tables + slot * K2H_TABLE;
}

// N=1 routing on the GPU: thread t places expert selected[t] at its rank, so
// items run in ascending expert order with no host sync.
__global__ void k2h_decode_route(glmvq_decode_item *items, const int32_t *selected,
        const uint32_t *table, const glmvq_source *sources) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x;
    const uint32_t e = min((uint32_t)selected[t], (uint32_t)K2H_EXPERTS - 1u);
    uint32_t rank = 0;
    for (uint32_t j = 0; j < K2H_USED; j++) { rank += (uint32_t)selected[j] < (uint32_t)selected[t]; }
    items[rank] = {sources[table[e]].vq, sources[table[K2H_EXPERTS + e]].vq,
        sources[table[2 * K2H_EXPERTS + e]].vq, t};
}

// Union of 2-4 rows' experts in ascending expert order (gvq_verify_route);
// down groups hold at most `capacity` pairs' mid rows.
__global__ void k2h_verify_route(glmvq_verify_item *items, glmvq_verify_plan *plan,
        const int32_t *selected, uint32_t rows, const uint32_t *table, const glmvq_source *sources,
        uint32_t capacity) {
    __shared__ uint32_t expert[GLMVQ_VER_PAIRS], lead[GLMVQ_VER_PAIRS], width[GLMVQ_VER_PAIRS];
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x, pairs = rows * K2H_USED;
    const bool live = t < pairs;
    if (live) { expert[t] = min((uint32_t)selected[t], (uint32_t)K2H_EXPERTS - 1u); }
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
            items[rank].up = sources[table[K2H_EXPERTS + e]].vq;
            items[rank].down = sources[table[2 * K2H_EXPERTS + e]].vq;
            items[rank].count = same;
            width[rank] = same;
        }
        items[rank].pair[before] = t;
        const uint32_t row = t / K2H_USED;
        for (uint32_t k = 0; k < K2H_USED; k++) { below += expert[row * K2H_USED + k] < e; }
        plan->order[row * K2H_USED + below] = t % K2H_USED;
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

static cudaError_t k2h_decode_run(ds4_k2h_moe *moe, float *out, const float *input, const int32_t *selected,
        const float *weights, const float *shared, const uint32_t *table, const void *book) {
    const k2h_decode_layout l = k2h_decode_plan();
    uint8_t *base = (uint8_t *)moe->decode->ptr;
    auto *items = (glmvq_decode_item *)(base + l.items);
    auto *mid = (__nv_bfloat16 *)(base + l.mid);
    auto *partial = (float *)(base + l.partial);
    const cudaStream_t stream = ds4_current_stream();
    const glmvq_launch launch = moe->gvq->launch;
    glmvq_launch_kernel(launch, k2h_decode_route, 1, K2H_USED, 0, stream, items, selected, table,
        (const glmvq_source *)moe->gvq->model->device->ptr);
    return moe->gain_dot
        ? glmvq_decode_launch<13, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_DOT, k2h_form>(items, book, input, mid,
            partial, out, weights, INFINITY, moe->decode_blocks, stream, nullptr, nullptr, shared, launch)
        : glmvq_decode_launch<13, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_WEIGHT, k2h_form>(items, book, input,
            mid, partial, out, weights, INFINITY, moe->decode_blocks, stream, nullptr, nullptr, shared, launch);
}

// Rows above the form's verify width run as consecutive slices (each row's
// arithmetic is the one-row kernel's, so slicing changes no value).
static cudaError_t k2h_verify_run(ds4_k2h_moe *moe, float *out, const float *input, const int32_t *selected,
        const float *weights, const float *shared, const uint32_t *table, const void *book, uint32_t rows) {
    const k2h_decode_layout l = k2h_decode_plan();
    uint8_t *base = (uint8_t *)moe->decode->ptr;
    auto *items = (glmvq_verify_item *)(base + l.vitems);
    auto *plan = (glmvq_verify_plan *)(base + l.plan);
    auto *mid = (__nv_bfloat16 *)(base + l.vmid);
    auto *partial = (float *)(base + l.vpartial);
    const cudaStream_t stream = ds4_current_stream();
    const glmvq_launch launch = moe->gvq->launch;
    constexpr uint32_t slice = k2h_form::VER_ROWS;
    cudaError_t err = cudaSuccess;
    for (uint32_t r = 0; err == cudaSuccess && r < rows; r += slice) {
        const uint32_t n = min(slice, rows - r);
        const uint64_t at = (uint64_t)r * K2H_USED, row = (uint64_t)r * K2H_HIDDEN;
        if (n == 1) {
            err = k2h_decode_run(moe, out + row, input + row, selected + at, weights + at, shared + row, table, book);
            continue;
        }
        glmvq_launch_kernel(launch, k2h_verify_route, 1, GLMVQ_VER_PAIRS, 0, stream, items, plan, selected + at, n,
            table, (const glmvq_source *)moe->gvq->model->device->ptr, k2h_form::VER_GROUP);
        err = moe->gain_dot
            ? glmvq_verify_launch<13, 32, GLMVQ_GAIN_DOT, k2h_form>(items, plan, book, input + row, n, mid, partial,
                out + row, weights + at, INFINITY, moe->decode_blocks, stream, shared + row, launch)
            : glmvq_verify_launch<13, 32, GLMVQ_GAIN_WEIGHT, k2h_form>(items, plan, book, input + row, n, mid,
                partial, out + row, weights + at, INFINITY, moe->decode_blocks, stream, shared + row, launch);
    }
    return err;
}

// Host grouping by expert (ascending), as the GLM prefill path builds it.
static bool k2h_group(ds4_k2h_moe *moe, const ds4_gpu_tensor *selected, uint32_t rows, uint32_t *offsets) {
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * K2H_USED;
    if (pairs > ctx->routes.size() ||
        !ds4_gpu_tensor_read(selected, 0, ctx->routes.data(), pairs * sizeof(int32_t))) { return false; }
    std::vector<uint32_t> counts(K2H_EXPERTS, 0), cursor(K2H_EXPERTS);
    for (uint64_t i = 0; i < pairs; i++) {
        const int32_t expert = ctx->routes[i];
        if (expert < 0 || expert >= K2H_EXPERTS) { return false; }
        for (uint64_t j = i - i % K2H_USED; j < i; j++) {
            if (ctx->routes[j] == expert) { return false; }
        }
        counts[expert]++;
    }
    offsets[0] = 0;
    for (uint32_t e = 0; e < K2H_EXPERTS; e++) {
        offsets[e + 1] = offsets[e] + counts[e];
        cursor[e] = offsets[e];
    }
    for (uint64_t i = 0; i < pairs; i++) { ctx->grouped[cursor[ctx->routes[i]]++] = (uint32_t)i; }
    return ds4_gpu_tensor_write(ctx->pairs, 0, ctx->grouped.data(), pairs * sizeof(uint32_t)) != 0;
}

// Each grouped pair's token row (pair index / top-k).
__global__ void k2h_pair_tokens(uint32_t *token_of, const uint32_t *map, uint32_t pairs) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < pairs) { token_of[p] = map[p] / K2H_USED; }
}

// Prefill experts on the warp-specialized 13-bit kernels: 64-pair items per
// projection, tokens quantized once (E4M3, 1x128 scales, ldmatrix k order).
static int k2h_mma_run(ds4_k2h_moe *moe, float *out, const float *input, const float *weights, const float *shared,
        const uint32_t *matrices, const uint32_t *offsets, uint32_t rows) {
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * K2H_USED, count = (uint64_t)rows * K2H_HIDDEN;
    const k2h_mma_layout l = k2h_mma_plan(rows);
    uint8_t *base = (uint8_t *)moe->mma->ptr;
    auto *x8 = base + l.x8, *mid8 = base + l.mid8;
    auto *xs = (float *)(base + l.xs), *ms = (float *)(base + l.ms);
    auto *gate = (__nv_bfloat16 *)(base + l.gate), *down = (__nv_bfloat16 *)(base + l.down);
    auto *order = (uint32_t *)(base + l.order);
    auto *witems = (mw_item *)(base + l.witems);
    auto *token = (uint32_t *)(base + l.token);
    moe->witems.clear();
    for (uint32_t m = 0; m < 3; m++) {
        for (uint32_t e = 0; e < K2H_EXPERTS; e++) {
            const glmvq_matrix &vq = ctx->model->source[matrices[m * K2H_EXPERTS + e]].vq;
            for (uint32_t start = offsets[e]; start < offsets[e + 1]; start += MW_BT) {
                moe->witems.push_back({start, min((uint32_t)MW_BT, offsets[e + 1] - start), vq.gain, vq.main, vq.plane,
                    nullptr, 0});
            }
        }
    }
    std::fill(moe->token_used.begin(), moe->token_used.begin() + rows, 0);
    for (uint32_t i = 0; i < pairs; i++) {
        const uint32_t t = ctx->grouped[i] / K2H_USED;
        moe->order[t * K2H_USED + moe->token_used[t]++] = i;
    }
    const cudaStream_t stream = ds4_current_stream();
    if (!cuda_ok(cudaMemcpyAsync(witems, moe->witems.data(), moe->witems.size() * sizeof(mw_item),
            cudaMemcpyHostToDevice, stream), "K2-Horizon expert items") ||
        !cuda_ok(cudaMemcpyAsync(order, moe->order.data(), pairs * sizeof(*order), cudaMemcpyHostToDevice, stream),
            "K2-Horizon expert order")) { return 0; }
    const auto *book = (const uint8_t *)ctx->model->source[matrices[0]].vq.book;
    const auto *grouped = (const uint32_t *)ctx->pairs->ptr;
    const uint32_t n = (uint32_t)moe->witems.size() / 3;
    const uint64_t xb = count / GLMVQ_MMA8_K;
    mw_quant<<<(uint32_t)((xb * 32 + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0, stream>>>(x8, xs, input, xb);
    k2h_pair_tokens<<<(uint32_t)((pairs + 255) / 256), 256, 0, stream>>>(token, grouped, (uint32_t)pairs);
    // Up's BF16 rows borrow down's buffer (larger) until the SwiGLU pass reads them.
    cudaError_t err = mw_launch<GLMVQ_MMA_GATE, false>(witems, n, book, nullptr, x8, xs, token, gate, nullptr,
        K2H_FFN, K2H_HIDDEN, moe->sms, stream);
    if (err == cudaSuccess) {
        err = mw_launch<GLMVQ_MMA_GATE, false>(witems + n, n, book, nullptr, x8, xs, token, down, nullptr,
            K2H_FFN, K2H_HIDDEN, moe->sms, stream);
    }
    if (err == cudaSuccess) {
        const uint64_t warps = pairs * (K2H_FFN / GLMVQ_MMA8_K);
        mw_swiglu_quant<GLMVQ_CLAMP_GATE><<<(uint32_t)((warps * 32 + GLMVQ_THREADS - 1) / GLMVQ_THREADS),
            GLMVQ_THREADS, 0, stream>>>(mid8, ms, gate, down, K2H_FFN, (uint32_t)pairs, INFINITY);
        err = mw_launch<GLMVQ_MMA_DOWN, false>(witems + 2 * n, n, book, nullptr, mid8, ms, token, nullptr, down,
            K2H_HIDDEN, K2H_FFN, moe->sms, stream);
    }
    if (!cuda_ok(err, "K2-Horizon warp-specialized experts")) { return 0; }
    glmvq_mma_reduce<__nv_bfloat16, GLMVQ_REDUCE_BF16><<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS,
        GLMVQ_THREADS, 0, stream>>>(out, down, weights, grouped, order, K2H_USED, K2H_HIDDEN, count, shared);
    return cuda_ok(cudaGetLastError(), "K2-Horizon FP8 grouped MMA");
}

extern "C" int ds4_gpu_k2h_moe(ds4_k2h_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *shared,
        const uint32_t *matrices, uint32_t rows) {
    if (!moe || !rows || rows > moe->rows || !matrices || !selected || !gvq_stream(moe->gvq)) { return 0; }
    const uint64_t pairs = (uint64_t)rows * K2H_USED, count = (uint64_t)rows * K2H_HIDDEN;
    if (pairs > selected->bytes / sizeof(int32_t) || !gvq_floats(weights, pairs) || !gvq_floats(out, count) ||
        !gvq_floats(input, count) || !gvq_floats(shared, count) || matrices[0] >= moe->gvq->model->source.size() ||
        ds4_capture_active()) { return 0; }
    const uint32_t *table = k2h_table(moe, matrices);
    if (!table) { return 0; }
    const void *book = moe->gvq->model->source[matrices[0]].vq.book;
    const auto *sel = (const int32_t *)selected->ptr;
    const auto *w = (const float *)weights->ptr;
    const auto *sh = (const float *)shared->ptr;
    float *y = (float *)out->ptr;
    const float *x = (const float *)input->ptr;
    if (rows <= K2H_VERIFY_ROWS) {
        const cudaError_t err = rows == 1 ? k2h_decode_run(moe, y, x, sel, w, sh, table, book)
                                          : k2h_verify_run(moe, y, x, sel, w, sh, table, book, rows);
        return cuda_ok(err, "K2-Horizon routed decode");
    }
    std::vector<uint32_t> offsets(K2H_EXPERTS + 1);
    if (!moe->mma || !k2h_group(moe, selected, rows, offsets.data())) { return 0; }
    return k2h_mma_run(moe, y, x, w, sh, matrices, offsets.data(), rows);
}

/* ------------------------------------------------------------------- router */

/* F32 router logits of up to K2H_ROUTER_ROWS rows (decode widths): warp e
 * streams expert e's 6,144 weights once and keeps one dot per row, lane l
 * summing float4 steps l, l + 32, ... in order, then a fixed xor tree, so a
 * row's logits do not depend on its launch's other rows (MiniMax's
 * mm_router_gemv). */
enum { K2H_ROUTER_ROWS = K2H_VERIFY_ROWS, K2H_ROUTER_WARPS = 2 };

template <uint32_t ROWS>
__global__ void __launch_bounds__(K2H_ROUTER_WARPS * 32) k2h_router_gemv(float *logits, const float *w,
        const float *x) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t lane = threadIdx.x & 31, e = blockIdx.x * K2H_ROUTER_WARPS + threadIdx.x / 32;
    const float4 *we = (const float4 *)(w + (uint64_t)e * K2H_HIDDEN);
    float acc[ROWS];
    #pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) { acc[r] = 0.0f; }
    #pragma unroll 8
    for (uint32_t i = lane; i < K2H_HIDDEN / 4; i += 32) {
        const float4 a = __ldg(we + i);
        #pragma unroll
        for (uint32_t r = 0; r < ROWS; r++) {
            const float4 v = ((const float4 *)(x + (uint64_t)r * K2H_HIDDEN))[i];
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
        if (!lane) { logits[(uint64_t)r * K2H_EXPERTS + e] = v; }
    }
}

/* Top-8 of a row, a warp per row: lane l holds experts l, l + 32, ...
 * (six), selection score sigmoid(BF16(logit)) + bias, eight warp argmax
 * rounds (ties to the lower expert); weights = BF16(s / sum * 2.5) of the
 * unbiased sigmoids. */
enum { K2H_SLOTS = K2H_EXPERTS / 32 };

__global__ void k2h_route_kernel(int32_t *ids, float *weights, const float *logits, const float *bias,
        uint32_t rows) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) { return; }
    float s[K2H_SLOTS], c[K2H_SLOTS];
    #pragma unroll
    for (uint32_t i = 0; i < K2H_SLOTS; i++) {
        const uint32_t e = lane + 32 * i;
        s[i] = glmvq_sigmoid(glmvq_round(logits[(uint64_t)row * K2H_EXPERTS + e]));
        c[i] = glmvq_add(s[i], bias[e]);
    }
    uint32_t chosen[K2H_USED];
    float chosen_s[K2H_USED], sum = 0.0f;
    for (uint32_t k = 0; k < K2H_USED; k++) {
        float best = -INFINITY;
        uint32_t index = UINT32_MAX;
        #pragma unroll
        for (uint32_t i = 0; i < K2H_SLOTS; i++) {
            if (c[i] > best) { best = c[i]; index = lane + 32 * i; }
        }
        for (uint32_t stride = 16; stride; stride >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, stride);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, index, stride);
            if (ob > best || (ob == best && oi < index)) { best = ob; index = oi; }
        }
        const uint32_t owner = index % 32, slot = index / 32;
        float picked = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < K2H_SLOTS; i++) {
            const float v = __shfl_sync(0xffffffffu, s[i], owner);
            if (i == slot) { picked = v; }
        }
        if (lane == owner) {
            #pragma unroll
            for (uint32_t i = 0; i < K2H_SLOTS; i++) {
                if (i == slot) { c[i] = -INFINITY; }
            }
        }
        chosen[k] = index;
        chosen_s[k] = picked;
        sum = glmvq_add(sum, picked);
    }
    if (lane) { return; }
    for (uint32_t k = 0; k < K2H_USED; k++) {
        ids[(uint64_t)row * K2H_USED + k] = (int32_t)chosen[k];
        weights[(uint64_t)row * K2H_USED + k] = glmvq_round(glmvq_mul(__fdiv_rn(chosen_s[k], sum), 2.5f));
    }
}

template <uint32_t MAX, typename F>
static void k2h_rows(uint32_t rows, F launch) {
    if constexpr (MAX > 1) {
        if (rows < MAX) { k2h_rows<MAX - 1>(rows, launch); return; }
    }
    launch(std::integral_constant<uint32_t, MAX>{});
}

extern "C" int ds4_gpu_k2h_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        ds4_gpu_tensor *logits, const ds4_gpu_tensor *input, uint32_t router, uint32_t bias, uint32_t rows) {
    const auto *w = (const float *)gvq_weight(ctx, router, GVQ_TYPE_F32, (uint64_t)K2H_EXPERTS * K2H_HIDDEN);
    const auto *b = (const float *)gvq_weight(ctx, bias, GVQ_TYPE_F32, K2H_EXPERTS);
    if (!w || !b || !rows || !gvq_floats(logits, (uint64_t)rows * K2H_EXPERTS) ||
        !gvq_floats(input, (uint64_t)rows * K2H_HIDDEN) || !gvq_floats(weights, (uint64_t)rows * K2H_USED) ||
        selected->bytes < (uint64_t)rows * K2H_USED * sizeof(int32_t) || !gvq_stream(ctx)) { return 0; }
    const glmvq_launch mode = gvq_mode(rows);
    if (rows <= K2H_ROUTER_ROWS) {
        k2h_rows<K2H_ROUTER_ROWS>(rows, [&](auto r) {
            glmvq_launch_kernel(mode, k2h_router_gemv<decltype(r)::value>, K2H_EXPERTS / K2H_ROUTER_WARPS,
                K2H_ROUTER_WARPS * 32, 0, ds4_current_stream(), (float *)logits->ptr, w, (const float *)input->ptr);
        });
    } else {
        // F32 products of BF16 values: pedantic F32 keeps TF32 off the routes.
        const float alpha = 1.0f, beta = 0.0f;
        if (!cublas_ok(cublasGemmEx(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N, K2H_EXPERTS, (int)rows, K2H_HIDDEN,
                &alpha, w, CUDA_R_32F, K2H_HIDDEN, input->ptr, CUDA_R_32F, K2H_HIDDEN, &beta, logits->ptr, CUDA_R_32F,
                K2H_EXPERTS, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT), "K2-Horizon router logits")) {
            return 0;
        }
    }
    enum { WARPS = 8 };
    glmvq_launch_kernel(mode, k2h_route_kernel, (rows + WARPS - 1) / WARPS, WARPS * 32, 0, ds4_current_stream(),
        (int32_t *)selected->ptr, (float *)weights->ptr, (const float *)logits->ptr, b, rows);
    return cuda_ok(cudaGetLastError(), "K2-Horizon router");
}

/* -------------------------------------------------------------------- Q8_0 */

/* Q8_0 decode GEMV (A.X-K2's ax_q8_gemv order): warp o stages output o's
 * row in shared memory by 16-byte copies while the predecessor drains, then
 * lane l takes element l of every 34-byte block (d * q exact in F32)
 * against the F32 input, and a fixed xor tree sums the lanes. */
template <uint32_t WARPS, ds4_k2h_q8_out MODE>
__global__ void __launch_bounds__(WARPS * 32) k2h_q8_gemv(float *out, const uint8_t *weight, const float *x,
        uint32_t outs, uint32_t cols) {
    extern __shared__ __align__(16) uint8_t k2h_q8_rows[];
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32, o = blockIdx.x * WARPS + warp;
    const uint32_t blocks = cols / K2H_Q8_BLOCK, row_bytes = blocks * K2H_Q8_BYTES;
    uint8_t *w = k2h_q8_rows + warp * row_bytes;
    glmvq_pdl_open();
    if (o < outs) {
        const uint4 *src = (const uint4 *)(weight + (uint64_t)o * row_bytes);
        for (uint32_t i = lane; i < row_bytes / 16; i += 32) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                "r"((uint32_t)__cvta_generic_to_shared((uint4 *)w + i)), "l"(src + i) : "memory");
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    glmvq_pdl_wait();
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncwarp();
    if (o >= outs) { return; }
    float acc = 0.0f;
    #pragma unroll 16
    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *block = w + b * K2H_Q8_BYTES;
        const float value = __half2float(__ushort_as_half((uint16_t)(block[0] | block[1] << 8))) *
            (float)(int8_t)block[2 + lane];
        acc = fmaf(value, x[b * K2H_Q8_BLOCK + lane], acc);
    }
    for (uint32_t stride = 16; stride; stride >>= 1) { acc += __shfl_xor_sync(0xffffffffu, acc, stride); }
    if (!lane) { out[o] = MODE == DS4_K2H_Q8_ROUND ? glmvq_round(acc) : acc; }
}

// Warps per CTA so a CTA's staged rows fit beside a second CTA's: 6,144-wide
// rows (6,528 B) take 8, the dense down's 16,384 (17,408 B) 2.
static uint32_t k2h_q8_warps(uint32_t row_bytes) {
    return row_bytes * 8u <= 48u * 1024u ? 8u : row_bytes * 4u <= 48u * 1024u ? 4u : 2u;
}

template <uint32_t WARPS, ds4_k2h_q8_out MODE>
static bool k2h_q8_launch(const glmvq_source &s, float *out, const float *x, uint32_t rows) {
    const auto kernel = k2h_q8_gemv<WARPS, MODE>;
    const uint32_t row_bytes = s.cols / K2H_Q8_BLOCK * K2H_Q8_BYTES, smem = WARPS * row_bytes;
    if (smem > 48u * 1024u && !cuda_ok(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem), "K2-Horizon Q8_0 GEMV attributes")) { return false; }
    glmvq_launch_kernel(gvq_mode(rows), kernel, (s.rows + WARPS - 1) / WARPS, WARPS * 32, smem,
        ds4_current_stream(), out, (const uint8_t *)s.data, x, s.rows, s.cols);
    return cuda_ok(cudaGetLastError(), "K2-Horizon Q8_0 GEMV");
}

// One Q8_0 record over `rows` F32 inputs of its width, or 0.
static const glmvq_source *k2h_q8_source(ds4_gvq_ctx *ctx, uint32_t weight, const ds4_gpu_tensor *out,
        const ds4_gpu_tensor *input, uint32_t rows) {
    if (weight >= ctx->model->source.size() || ctx->model->records[weight].type != K2H_TYPE_Q8_0) { return nullptr; }
    const glmvq_source &s = ctx->model->source[weight];
    return s.cols % 256 || !gvq_floats(input, (uint64_t)rows * s.cols) || !gvq_floats(out, (uint64_t)rows * s.rows)
        ? nullptr : &s;
}

// One row: the GEMV per matrix.
static int k2h_q8_row(const glmvq_source &s, float *y, const float *x, ds4_k2h_q8_out mode) {
    const uint32_t row_bytes = s.cols / K2H_Q8_BLOCK * K2H_Q8_BYTES;
    if (row_bytes % 16 || (uintptr_t)s.data % 16) { return 0; }
    const uint32_t warps = k2h_q8_warps(row_bytes);
    const bool round = mode == DS4_K2H_Q8_ROUND;
    if (warps == 8) {
        return round ? k2h_q8_launch<8, DS4_K2H_Q8_ROUND>(s, y, x, 1) : k2h_q8_launch<8, DS4_K2H_Q8_F32>(s, y, x, 1);
    }
    if (warps == 4) {
        return round ? k2h_q8_launch<4, DS4_K2H_Q8_ROUND>(s, y, x, 1) : k2h_q8_launch<4, DS4_K2H_Q8_F32>(s, y, x, 1);
    }
    return round ? k2h_q8_launch<2, DS4_K2H_Q8_ROUND>(s, y, x, 1) : k2h_q8_launch<2, DS4_K2H_Q8_F32>(s, y, x, 1);
}

/* Prefill quantizes the shared input to Q8_1 once for every matrix (MMQ);
 * the outputs skip MMQ's non-finite scrub (finite F16 scales, which the host
 * checked, and finite activations give finite sums) or, rounded, take the
 * BF16 boundary in that pass's place. */
extern "C" int ds4_gpu_k2h_q8_many(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs, const uint32_t *weights,
        uint32_t count, const ds4_gpu_tensor *input, uint32_t rows, ds4_k2h_q8_out mode) {
    if (!ctx || !outs || !weights || !count || count > DS4_K2H_Q8_MANY || !rows || !gvq_stream(ctx)) { return 0; }
    const void *w[DS4_K2H_Q8_MANY];
    float *y[DS4_K2H_Q8_MANY];
    int m[DS4_K2H_Q8_MANY];
    uint32_t cols = 0;
    for (uint32_t c = 0; c < count; c++) {
        const glmvq_source *s = k2h_q8_source(ctx, weights[c], outs[c], input, rows);
        if (!s || (c && s->cols != cols)) { return 0; }
        cols = s->cols;
        w[c] = s->data;
        y[c] = (float *)outs[c]->ptr;
        m[c] = (int)s->rows;
    }
    const float *x = (const float *)input->ptr;
    if (rows == 1) {
        for (uint32_t c = 0; c < count; c++) {
            if (!k2h_q8_row(ctx->model->source[weights[c]], y[c], x, mode)) { return 0; }
        }
        return 1;
    }
    const cudaStream_t stream = ds4_mmq_stream_for_call();
    const int rc = mode == DS4_K2H_Q8_ROUND
        ? ds4_mmq_q8_0_dense_bf16_many(w, y, m, (int)count, x, (int)rows, (int)cols, stream)
        : ds4_mmq_q8_0_dense_raw_many(w, y, m, (int)count, x, (int)rows, (int)cols, stream);
    return rc == 0 && cuda_ok(cudaGetLastError(), "K2-Horizon Q8_0 MMQ");
}

extern "C" int ds4_gpu_k2h_q8(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t weight, const ds4_gpu_tensor *input,
        uint32_t rows, ds4_k2h_q8_out mode) {
    return ds4_gpu_k2h_q8_many(ctx, &out, &weight, 1u, input, rows, mode);
}

/* -------------------------------------------------------------------- norms */

// Block sum of one value per thread (K2H_THREADS), fixed order.
__device__ __forceinline__ float k2h_block_sum(float v, float *scratch) {
    for (uint32_t stride = 16; stride; stride >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, stride); }
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    if (!lane) { scratch[warp] = v; }
    __syncthreads();
    float total = 0.0f;
    #pragma unroll
    for (uint32_t w = 0; w < K2H_THREADS / 32; w++) { total = glmvq_add(total, scratch[w]); }
    return total;
}

/* CTA per row: cur = BF16(cur + BF16(branch)) when there is a branch, then
 * norm = BF16((cur * rsqrt(mean cur^2 + 1e-6)) * w), F32 throughout. Every
 * load is issued before the first store: interleaved with the cur stores
 * they ran one round trip each (24 us a decode row, now ~4). */
__global__ void __launch_bounds__(K2H_THREADS) k2h_norm_kernel(float *norm, float *cur, const float *branch,
        const float *weight) {
    __shared__ float scratch[K2H_THREADS / 32];
    constexpr uint32_t PER = K2H_HIDDEN / K2H_THREADS;
    glmvq_pdl_open();
    float w[PER];
    #pragma unroll
    for (uint32_t i = 0; i < PER; i++) { w[i] = weight[threadIdx.x + i * K2H_THREADS]; }
    glmvq_pdl_wait();
    const uint64_t row = (uint64_t)blockIdx.x * K2H_HIDDEN;
    float x[PER], b[PER], sq = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < PER; i++) {
        x[i] = cur[row + threadIdx.x + i * K2H_THREADS];
        b[i] = branch ? branch[row + threadIdx.x + i * K2H_THREADS] : 0.0f;
    }
    #pragma unroll
    for (uint32_t i = 0; i < PER; i++) {
        if (branch) {
            x[i] = glmvq_round(glmvq_add(x[i], glmvq_round(b[i])));
            cur[row + threadIdx.x + i * K2H_THREADS] = x[i];
        }
        sq = fmaf(x[i], x[i], sq);
    }
    const float total = k2h_block_sum(sq, scratch);
    const float inv = rsqrtf(glmvq_add(__fdiv_rn(total, (float)K2H_HIDDEN), 1e-6f));
    #pragma unroll
    for (uint32_t i = 0; i < PER; i++) {
        norm[row + threadIdx.x + i * K2H_THREADS] = glmvq_round(glmvq_mul(glmvq_mul(x[i], inv), w[i]));
    }
}

extern "C" int ds4_gpu_k2h_norm(ds4_gvq_ctx *ctx, ds4_gpu_tensor *norm, ds4_gpu_tensor *cur,
        const ds4_gpu_tensor *branch, uint32_t weight, uint32_t rows) {
    const auto *w = (const float *)gvq_weight(ctx, weight, GVQ_TYPE_F32, K2H_HIDDEN);
    const uint64_t count = (uint64_t)rows * K2H_HIDDEN;
    if (!w || !rows || !gvq_floats(norm, count) || !gvq_floats(cur, count) || (branch && !gvq_floats(branch, count)) ||
        !gvq_stream(ctx)) { return 0; }
    glmvq_launch_kernel(gvq_mode(rows), k2h_norm_kernel, rows, K2H_THREADS, 0, ds4_current_stream(),
        (float *)norm->ptr, (float *)cur->ptr, branch ? (const float *)branch->ptr : nullptr, w);
    return cuda_ok(cudaGetLastError(), "K2-Horizon RMSNorm");
}

/* -------------------------------------------------------------- RoPE and K/V */

// inv_freq_j = 1 / 1e7^(j / 32) as torch computes it in F32 (powf, then a
// rounded division), uploaded once.
__constant__ float k2h_inv_freq[DS4_K2H_ROPE_PAIRS];
static bool k2h_inv_freq_ready = false;

static bool k2h_inv_freq_upload(void) {
    if (k2h_inv_freq_ready) { return true; }
    float inv[DS4_K2H_ROPE_PAIRS];
    for (uint32_t j = 0; j < DS4_K2H_ROPE_PAIRS; j++) {
        const float exponent = (float)(2u * j) / 64.0f;
        inv[j] = 1.0f / powf(10000000.0f, exponent);
    }
    k2h_inv_freq_ready = cuda_ok(cudaMemcpyToSymbol(k2h_inv_freq, inv, sizeof(inv)), "K2-Horizon RoPE table");
    return k2h_inv_freq_ready;
}

__global__ void k2h_rope_kernel(float *rope, uint32_t rows, uint32_t pos0) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * DS4_K2H_ROPE_PAIRS) { return; }
    const uint32_t row = i / DS4_K2H_ROPE_PAIRS, j = i % DS4_K2H_ROPE_PAIRS;
    // The reference's F32 angle; cos/sin in double (--use_fast_math maps the
    // single-precision ones to __sinf/__cosf, wrong far from zero).
    const double angle = (double)__fmul_rn((float)(pos0 + row), k2h_inv_freq[j]);
    rope[(uint64_t)row * 2 * DS4_K2H_ROPE_PAIRS + j] = glmvq_round((float)cos(angle));
    rope[(uint64_t)row * 2 * DS4_K2H_ROPE_PAIRS + DS4_K2H_ROPE_PAIRS + j] = glmvq_round((float)sin(angle));
}

extern "C" int ds4_gpu_k2h_rope(ds4_gpu_tensor *rope, uint32_t rows, uint32_t pos0) {
    if (!rows || !gvq_floats(rope, (uint64_t)rows * 2 * DS4_K2H_ROPE_PAIRS) || !k2h_inv_freq_upload()) { return 0; }
    const uint32_t n = rows * DS4_K2H_ROPE_PAIRS;
    k2h_rope_kernel<<<(n + 255) / 256, 256, 0, ds4_current_stream()>>>((float *)rope->ptr, rows, pos0);
    return cuda_ok(cudaGetLastError(), "K2-Horizon RoPE rows");
}

/* CTA (row, head): heads 0..47 are q, 48..55 k, 56..63 v; a thread per
 * dimension. q and k round to BF16 and rotate pairs (j, j + 64), j < 32:
 * x_j' = BF16(BF16(x_j c) + BF16(-x_{j+64} s)), x_{j+64}' = BF16(BF16(x_{j+64}
 * c) + BF16(x_j s)) (rotate_half on the split layout); q stays in place, K and
 * V go to their NVFP4 page (one warp encodes the vector). */
__global__ void __launch_bounds__(DS4_K2H_HEAD) k2h_qkv_kernel(float *q, const float *k, const float *v,
        uint8_t *pages, const float *rope, uint32_t pos) {
    __shared__ float vec[DS4_K2H_HEAD];
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t row = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const bool is_q = head < DS4_K2H_HEADS, is_v = head >= DS4_K2H_HEADS + DS4_K2H_KV_HEADS;
    const uint32_t kv = is_q ? 0u : (head - DS4_K2H_HEADS) % DS4_K2H_KV_HEADS;
    const float *src = is_q ? q + ((uint64_t)row * DS4_K2H_HEADS + head) * DS4_K2H_HEAD
                     : (is_v ? v : k) + ((uint64_t)row * DS4_K2H_KV_HEADS + kv) * DS4_K2H_HEAD;
    vec[d] = glmvq_round(src[d]);
    __syncthreads();
    float out = vec[d];
    if (!is_v && (d < DS4_K2H_ROPE_PAIRS || (d >= 64 && d < 64 + DS4_K2H_ROPE_PAIRS))) {
        const uint32_t j = d % 64;
        const float c = rope[(uint64_t)row * 2 * DS4_K2H_ROPE_PAIRS + j];
        const float s = rope[(uint64_t)row * 2 * DS4_K2H_ROPE_PAIRS + DS4_K2H_ROPE_PAIRS + j];
        const float other = d < 64 ? -vec[d + 64] : vec[d - 64];
        out = glmvq_round(glmvq_add(glmvq_round(glmvq_mul(vec[d], c)), glmvq_round(glmvq_mul(other, s))));
    }
    if (is_q) {
        q[((uint64_t)row * DS4_K2H_HEADS + head) * DS4_K2H_HEAD + d] = out;
        return;
    }
    __syncthreads();
    vec[d] = out;
    __syncthreads();
    if (d >= 32) { return; }
    const uint32_t p = pos + row;
    uint8_t *half = (uint8_t *)ink2a_page(pages, p, kv) + (is_v ? DS4_K2H_KV_BLOCK * DS4_K2H_FP4_ROW : 0u);
    const uint32_t slot = p % DS4_K2H_KV_BLOCK;
    mm_fp4_encode(vec, half + slot * DS4_K2H_FP4_DATA, half + DS4_K2H_KV_BLOCK * DS4_K2H_FP4_DATA + slot * 8);
}

extern "C" int ds4_gpu_k2h_qkv(ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
        ds4_gpu_tensor *pages, const ds4_gpu_tensor *rope, uint32_t rows, uint32_t pos) {
    const uint64_t blocks = ((uint64_t)pos + rows + DS4_K2H_KV_BLOCK - 1) / DS4_K2H_KV_BLOCK;
    if (!rows || !gvq_floats(q, (uint64_t)rows * DS4_K2H_Q) || !gvq_floats(k, (uint64_t)rows * DS4_K2H_KV) ||
        !gvq_floats(v, (uint64_t)rows * DS4_K2H_KV) || !gvq_floats(rope, (uint64_t)rows * 2 * DS4_K2H_ROPE_PAIRS) ||
        !pages || pages->bytes < blocks * DS4_K2H_BLOCK_BYTES) { return 0; }
    const uint32_t heads = DS4_K2H_HEADS + 2 * DS4_K2H_KV_HEADS;
    glmvq_launch_kernel(gvq_mode(rows), k2h_qkv_kernel, dim3(rows, heads), DS4_K2H_HEAD, 0, ds4_current_stream(),
        (float *)q->ptr, (const float *)k->ptr, (const float *)v->ptr, (uint8_t *)pages->ptr,
        (const float *)rope->ptr, pos);
    return cuda_ok(cudaGetLastError(), "K2-Horizon q/k/v tails");
}

/* ---------------------------------------------------------------- attention */

// Rows up to this many split their keys (decode); wider forwards tile.
enum { K2H_SPLIT_ROWS = K2H_VERIFY_ROWS };

extern "C" uint64_t ds4_gpu_k2h_attn_bytes(void) {
    return (uint64_t)K2H_SPLIT_ROWS * DS4_K2H_HEADS * INK2A_SPLITS * K2A_PARTIAL * sizeof(float);
}

extern "C" int ds4_gpu_k2h_attn(ds4_gpu_tensor *heads, const ds4_gpu_tensor *q, const ds4_gpu_tensor *pages,
        ds4_gpu_tensor *partials, uint32_t rows, uint32_t pos) {
    const uint64_t count = (uint64_t)rows * DS4_K2H_Q;
    const uint64_t blocks = ((uint64_t)pos + rows + DS4_K2H_KV_BLOCK - 1) / DS4_K2H_KV_BLOCK;
    if (!rows || !gvq_floats(heads, count) || !gvq_floats(q, count) || !pages ||
        pages->bytes < blocks * DS4_K2H_BLOCK_BYTES) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    if (rows <= K2H_SPLIT_ROWS) {
        if (!partials || partials->bytes < ds4_gpu_k2h_attn_bytes()) { return 0; }
        // Tensor-core split (DS4_K2H_ATTN_MMA=0: the k2a_decode lane-per-key walk).
        static const bool mma = !k2h_env_off("DS4_K2H_ATTN_MMA");
        const glmvq_launch mode = gvq_mode(rows);
        uint32_t splits = ink2a_splits(pos + rows);
        if (mma) {
            const uint32_t tiles = (pos + rows - 1) / INK2A_TK + 1, slices = tiles < K2A_DSLICES ? tiles : K2A_DSLICES;
            splits = slices * K2A_DWARPS;
            glmvq_launch_kernel(mode, k2a_split, dim3(slices, DS4_K2H_KV_HEADS, rows), K2A_DWARPS * 32, 0, stream,
                (float *)partials->ptr, (const float *)q->ptr, (const uint8_t *)pages->ptr, pos, slices);
        } else {
            k2a_decode<<<dim3(splits, DS4_K2H_KV_HEADS, rows), INK2A_DTHREADS, 0, stream>>>((float *)partials->ptr,
                (const float *)q->ptr, (const uint8_t *)pages->ptr, pos, splits);
        }
        glmvq_launch_kernel(mode, k2a_merge, rows * DS4_K2H_HEADS, INK2A_DIM, 0, stream, (float *)heads->ptr,
            (const float *)partials->ptr, splits);
        return cuda_ok(cudaGetLastError(), "K2-Horizon decode attention");
    }
    // Ping-pong tiles (DS4_K2H_ATTN_PP=0: k2a_prefill's warps in step).
    static const bool pp = !k2h_env_off("DS4_K2H_ATTN_PP");
    static const cudaError_t ready = pp
        ? cudaFuncSetAttribute(k2a_prefill_pp, cudaFuncAttributeMaxDynamicSharedMemorySize, K2A_PP_SMEM)
        : cudaFuncSetAttribute(k2a_prefill, cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM_FP4);
    if (!cuda_ok(ready, "K2-Horizon prefill attention attributes")) { return 0; }
    const dim3 grid((rows + INK2A_TQ - 1) / INK2A_TQ, DS4_K2H_HEADS / INK2A_CTA_HEADS);
    if (pp) {
        k2a_prefill_pp<<<grid, INK2A_THREADS, K2A_PP_SMEM, stream>>>((float *)heads->ptr, (const float *)q->ptr,
            (const uint8_t *)pages->ptr, rows, pos);
    } else {
        k2a_prefill<<<grid, INK2A_THREADS, INK2A_SMEM_FP4, stream>>>((float *)heads->ptr, (const float *)q->ptr,
            (const uint8_t *)pages->ptr, rows, pos);
    }
    return cuda_ok(cudaGetLastError(), "K2-Horizon prefill attention");
}

/* ---------------------------------------------------------- SwiGLU, embedding */

__global__ void k2h_swiglu_kernel(float *out, const float *gate, const float *up, uint64_t count) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const float g = glmvq_round(gate[i]);
    const float silu = glmvq_round(__fdiv_rn(g, glmvq_add(1.0f, expf(-g))));
    out[i] = glmvq_round(glmvq_mul(silu, glmvq_round(up[i])));
}

extern "C" int ds4_gpu_k2h_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
        uint32_t rows, uint32_t width) {
    const uint64_t count = (uint64_t)rows * width;
    if (!count || !gvq_floats(out, count) || !gvq_floats(gate, count) || !gvq_floats(up, count)) { return 0; }
    glmvq_launch_kernel(gvq_mode(rows), k2h_swiglu_kernel, (uint32_t)((count + K2H_THREADS - 1) / K2H_THREADS),
        K2H_THREADS, 0, ds4_current_stream(), (float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr,
        count);
    return cuda_ok(cudaGetLastError(), "K2-Horizon SwiGLU");
}

// A thread per value of a 34-byte block row: BF16(F32(d) * q).
__global__ void k2h_embed_kernel(float *cur, const uint8_t *rows_q8, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint64_t row = i / K2H_HIDDEN, col = i % K2H_HIDDEN;
    const uint8_t *block = rows_q8 + row * DS4_K2H_EMBED_ROW + col / K2H_Q8_BLOCK * K2H_Q8_BYTES;
    const float d = __half2float(__ushort_as_half((uint16_t)(block[0] | block[1] << 8)));
    cur[i] = glmvq_round(d * (float)(int8_t)block[2 + col % K2H_Q8_BLOCK]);
}

extern "C" int ds4_gpu_k2h_embed(ds4_gpu_tensor *cur, const ds4_gpu_tensor *rows_q8, uint32_t rows) {
    const uint64_t count = (uint64_t)rows * K2H_HIDDEN;
    if (!rows || !gvq_floats(cur, count) || !rows_q8 || rows_q8->bytes < (uint64_t)rows * DS4_K2H_EMBED_ROW) {
        return 0;
    }
    k2h_embed_kernel<<<(uint32_t)((count + K2H_THREADS - 1) / K2H_THREADS), K2H_THREADS, 0, ds4_current_stream()>>>(
        (float *)cur->ptr, (const uint8_t *)rows_q8->ptr, count);
    return cuda_ok(cudaGetLastError(), "K2-Horizon embedding");
}
