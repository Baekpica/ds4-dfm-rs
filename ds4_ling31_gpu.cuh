/* Ling-3.1-flash VQ device runtime; included once by ds4_cuda.cu after the
 * GVQ adapter and the Ling-3.0 primitives it reuses.
 *
 * Routed experts reuse GLM's GVQ1 kernels with Ling's form: FFN 1280 (down
 * rows are one partial 160-index block), the post-SiLU clamp and an F32
 * routed sum. Each layer's book decides the index width at run time:
 *
 *   rows == 1     decode: device routing, persistent CTAs keep the book
 *   rows 2..4     MTP verify: one pass over the rows' expert union
 *   rows >= 5     prefill: host grouping, FP8 MMA on the E4M3 codewords;
 *                 12-bit layers run Beam's warp-specialized kernels on
 *                 128-token items (DS4_LING31_MMA_WS=0: glmvq_mma8_kernel)
 *   DS4_LING31_MOE_RESTORE=1: BF16 restore + cuBLAS per expert (reference) */
#include "ds4_ling31.h"
#include "cuda/beam_mma8w.cuh"
#include "cuda/l31_residual.cuh"
#include "cuda/l31_swiglu.cuh"
#include "cuda/l31_latent.cuh"
#include "cuda/l31_flash.cuh"

enum { L31_EXPERTS = DS4_L31_EXPERTS, L31_USED = DS4_L31_USED, L31_HIDDEN = DS4_L31_HIDDEN,
       L31_FFN = DS4_L31_FFN, L31_TABLE = 3 * L31_EXPERTS, L31_LAYERS = DS4_L31_MOE_LAYERS,
       L31_VERIFY_ROWS = GLMVQ_VER_ROWS, L31_MMA_ROWS = 256, L31_MMA_TOKENS = 64, L31_ALIGN = 256 };
static_assert((uint32_t)L31_HIDDEN == (uint32_t)GLMVQ_DEC_HIDDEN && (uint32_t)L31_USED == (uint32_t)GLMVQ_DEC_USED,
    "GVQ decode geometry");
using l31_form = glmvq_moe_form<L31_FFN, GLMVQ_CLAMP_SILU, GLMVQ_REDUCE_F32>;

static uint64_t l31_align(uint64_t bytes) { return (bytes + L31_ALIGN - 1) / L31_ALIGN * L31_ALIGN; }

// Decode/verify scratch, then the device tables (one 3 x 512 row per layer).
struct l31_decode_layout { uint64_t items, mid, partial, vitems, plan, vmid, vpartial, tables, bytes; };

static l31_decode_layout l31_decode_plan() {
    l31_decode_layout l = {};
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += l31_align(bytes); return here; };
    l.items = take(sizeof(glmvq_decode_item) * L31_USED);
    l.mid = take((uint64_t)L31_USED * L31_FFN * sizeof(__nv_bfloat16));
    l.partial = take((uint64_t)L31_USED * L31_HIDDEN * sizeof(float));
    l.vitems = take(sizeof(glmvq_verify_item) * GLMVQ_VER_PAIRS);
    l.plan = take(sizeof(glmvq_verify_plan));
    l.vmid = take((uint64_t)GLMVQ_VER_PAIRS * L31_FFN * sizeof(__nv_bfloat16));
    l.vpartial = take((uint64_t)GLMVQ_VER_PAIRS * L31_HIDDEN * sizeof(float));
    l.tables = take((uint64_t)L31_LAYERS * L31_TABLE * sizeof(uint32_t));
    l.bytes = at;
    return l;
}

// Prefill: E4M3 input rows and 1x128 scales, gate rows (BF16), the up
// epilogue's E4M3 SwiGLU rows and scales, down rows (BF16), worklists; the
// warp-specialized path adds its items and each pair's token row and scale.
struct l31_mma_layout { uint64_t x8, xs, gate, mid8, ms, down, work, order, witems, token, pscale, bytes; };

static l31_mma_layout l31_mma_plan(uint32_t rows) {
    l31_mma_layout l = {};
    const uint64_t pairs = (uint64_t)rows * L31_USED;
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += l31_align(bytes); return here; };
    l.x8 = take((uint64_t)rows * L31_HIDDEN);
    l.xs = take((uint64_t)rows * (L31_HIDDEN / GLMVQ_MMA8_K) * sizeof(float));
    l.gate = take(pairs * L31_FFN * sizeof(__nv_bfloat16));
    l.mid8 = take(pairs * L31_FFN);
    l.ms = take(pairs * (L31_FFN / GLMVQ_MMA8_K) * sizeof(float));
    l.down = take(pairs * L31_HIDDEN * sizeof(__nv_bfloat16));
    l.work = take(pairs * sizeof(glmvq_mma_item));
    l.order = take(pairs * sizeof(uint32_t));
    l.witems = take(3 * (L31_EXPERTS + pairs / BEAM_W_BT) * sizeof(beam_w_item));
    l.token = take(pairs * sizeof(uint32_t));
    l.pscale = take(pairs * sizeof(float));
    l.bytes = at;
    return l;
}

struct ds4_l31_moe {
    ds4_gvq_ctx *gvq = nullptr;     // borrowed: stream, cuBLAS, routes, grouped, pairs
    ds4_gpu_tensor *decode = nullptr, *mma = nullptr;
    uint32_t rows = 0, decode_blocks = 0, sms = 0;
    int mma8_blocks[2][3] = {};     // [12-bit, 13-bit][gate, up, down]
    bool gain_dot = true, route = true, restore = false, mma8 = false, mma8w = true;
    std::vector<uint32_t> keys;     // table slot per layer, keyed by expert 0's gate
    std::vector<glmvq_mma_item> items;
    std::vector<beam_w_item> witems; // gate, up, down 128-token items (12-bit layers)
    std::vector<uint32_t> order;
    std::vector<uint8_t> token_used;
    uint64_t decode_calls = 0, verify_calls = 0, mma_calls = 0, restore_calls = 0;
};

static bool l31_env_off(const char *name) {
    const char *value = getenv(name);
    return value && !strcmp(value, "0");
}

extern "C" uint64_t ds4_gpu_l31_moe_bytes(uint32_t rows) {
    return l31_decode_plan().bytes + (rows > L31_VERIFY_ROWS ? l31_mma_plan(rows).bytes : 0u);
}

extern "C" void ds4_gpu_l31_moe_free(ds4_l31_moe *moe) {
    if (!moe) { return; }
    if (getenv("DS4_LING31_MOE_STATS")) {
        fprintf(stderr, "ds4: Ling-3.1 MoE decode=%llu verify=%llu mma=%llu restore=%llu\n",
            (unsigned long long)moe->decode_calls, (unsigned long long)moe->verify_calls,
            (unsigned long long)moe->mma_calls, (unsigned long long)moe->restore_calls);
    }
    ds4_gpu_tensor_free(moe->decode);
    ds4_gpu_tensor_free(moe->mma);
    delete moe;
}

template <uint32_t BITS>
static cudaError_t l31_prepare(ds4_l31_moe *moe) {
    uint32_t blocks = 0;
    cudaError_t err = glmvq_decode_prepare<BITS, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_DOT, l31_form>(&blocks);
    if (err == cudaSuccess) {
        err = glmvq_decode_prepare<BITS, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_WEIGHT, l31_form>(&blocks);
    }
    if (err != cudaSuccess) { return err; }
    moe->decode_blocks = blocks;
    int *mma = moe->mma8_blocks[BITS - 12];
    err = glmvq_mma8_prepare<L31_MMA_ROWS, L31_MMA_TOKENS, 1, GLMVQ_MMA_GATE, BITS, GLMVQ_CLAMP_SILU>(&mma[0]);
    if (err == cudaSuccess) {
        err = glmvq_mma8_prepare<L31_MMA_ROWS, L31_MMA_TOKENS, 1, GLMVQ_MMA_UP, BITS, GLMVQ_CLAMP_SILU>(&mma[1]);
    }
    if (err == cudaSuccess) {
        err = glmvq_mma8_prepare<L31_MMA_ROWS, L31_MMA_TOKENS, 1, GLMVQ_MMA_DOWN, BITS, GLMVQ_CLAMP_SILU>(&mma[2]);
    }
    return err;
}

extern "C" ds4_l31_moe *ds4_gpu_l31_moe_create(ds4_gvq_ctx *gvq, uint32_t rows) {
    if (!gvq || !rows) { return nullptr; }
    ds4_l31_moe *moe = new (std::nothrow) ds4_l31_moe;
    if (!moe) { return nullptr; }
    moe->gvq = gvq;
    gvq->kda_reload = !l31_env_off("DS4_LING31_KDA_RELOAD");
    moe->rows = rows;
    moe->gain_dot = !l31_env_off("DS4_LING31_GAIN_DOT");
    moe->route = !l31_env_off("DS4_LING31_DECODE_ROUTE");
    moe->mma8w = !l31_env_off("DS4_LING31_MMA_WS");
    int device = 0, sms = 0;
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess) {
        moe->sms = (uint32_t)sms;
    }
    const char *restore = getenv("DS4_LING31_MOE_RESTORE");
    moe->restore = restore && !strcmp(restore, "1");
    const cudaError_t err12 = l31_prepare<12>(moe), err13 = l31_prepare<13>(moe);
    if (err12 != cudaSuccess || err13 != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: Ling-3.1 expert kernels unavailable (%s)\n",
            cudaGetErrorString(err12 != cudaSuccess ? err12 : err13));
        ds4_gpu_l31_moe_free(moe);
        return nullptr;
    }
    moe->mma8 = rows > L31_VERIFY_ROWS;
    moe->decode = ds4_gpu_tensor_alloc(l31_decode_plan().bytes);
    moe->mma = moe->mma8 ? ds4_gpu_tensor_alloc(l31_mma_plan(rows).bytes) : nullptr;
    if (!moe->decode || (moe->mma8 && !moe->mma)) {
        ds4_gpu_l31_moe_free(moe);
        return nullptr;
    }
    moe->items.reserve((uint64_t)rows * L31_USED);
    moe->order.resize((uint64_t)rows * L31_USED);
    moe->token_used.resize(rows);
    return moe;
}

// Shape, book and alignment the direct kernels read; every one of a layer's
// 1,536 matrices shares its book (12-bit: no plane, 13-bit: plane).
static bool l31_matrix_fits(const glmvq_source &s, uint32_t projection, const void *book, uint32_t bits) {
    const uint32_t rows = projection == 2 ? L31_HIDDEN : L31_FFN;
    const uint32_t cols = projection == 2 ? L31_FFN : L31_HIDDEN;
    const glmvq_matrix &m = s.vq;
    return s.storage == GLMVQ_WEIGHT_VQ && s.rows == rows && s.cols == cols &&
        m.rows == rows && m.cols == cols && m.precision == GLMVQ_E4M3 &&
        m.main && m.gain && book && m.book == book && (bits == 13) == (m.plane != nullptr) &&
        (uintptr_t)m.main % 128 == 0 && (!m.plane || (uintptr_t)m.plane % 128 == 0) &&
        (uintptr_t)book % 16 == 0 && (uintptr_t)m.gain % sizeof(__half) == 0;
}

static uint32_t l31_bits(const ds4_l31_moe *moe, const uint32_t *matrices) {
    const auto &records = moe->gvq->model->records;
    const auto &book = records[records[matrices[0]].book];
    return book.bytes == 8192u * GLMVQ_VECTOR ? 13u : 12u;
}

// The layer's device table, validated and uploaded on first use.
static const uint32_t *l31_table(ds4_l31_moe *moe, const uint32_t *matrices, uint32_t bits) {
    const auto &sources = moe->gvq->model->source;
    const l31_decode_layout l = l31_decode_plan();
    const auto *tables = (const uint32_t *)((const uint8_t *)moe->decode->ptr + l.tables);
    for (size_t i = 0; i < moe->keys.size(); i++) {
        if (moe->keys[i] == matrices[0]) { return tables + i * L31_TABLE; }
    }
    if (moe->keys.size() == L31_LAYERS) { return nullptr; }
    const void *book = sources[matrices[0]].vq.book;
    for (uint32_t i = 0; i < L31_TABLE; i++) {
        if (matrices[i] >= sources.size() ||
            !l31_matrix_fits(sources[matrices[i]], i / L31_EXPERTS, book, bits)) {
            fprintf(stderr, "ds4: Ling-3.1 expert matrix %u misses the GVQ1 layout\n", i);
            return nullptr;
        }
    }
    const size_t slot = moe->keys.size();
    if (!ds4_gpu_tensor_write(moe->decode, l.tables + slot * L31_TABLE * sizeof(uint32_t),
            matrices, L31_TABLE * sizeof(uint32_t))) { return nullptr; }
    moe->keys.push_back(matrices[0]);
    return tables + slot * L31_TABLE;
}

// N=1 routing on the GPU: thread t places expert selected[t] at its rank, so
// items run in ascending expert order with no host sync.
__global__ void l31_decode_route(glmvq_decode_item *items, const int32_t *selected,
        const uint32_t *table, const glmvq_source *sources) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x;
    const uint32_t e = min((uint32_t)selected[t], (uint32_t)L31_EXPERTS - 1u);
    uint32_t rank = 0;
    for (uint32_t j = 0; j < L31_USED; j++) { rank += (uint32_t)selected[j] < (uint32_t)selected[t]; }
    items[rank] = {sources[table[e]].vq, sources[table[L31_EXPERTS + e]].vq,
        sources[table[2 * L31_EXPERTS + e]].vq, t};
}

// Union of 2-4 rows' experts in ascending expert order (gvq_verify_route).
__global__ void l31_verify_route(glmvq_verify_item *items, glmvq_verify_plan *plan,
        const int32_t *selected, uint32_t rows, const uint32_t *table, const glmvq_source *sources) {
    __shared__ uint32_t expert[GLMVQ_VER_PAIRS], lead[GLMVQ_VER_PAIRS], width[GLMVQ_VER_PAIRS];
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x, pairs = rows * L31_USED;
    const bool live = t < pairs;
    if (live) { expert[t] = min((uint32_t)selected[t], (uint32_t)L31_EXPERTS - 1u); }
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
            items[rank].up = sources[table[L31_EXPERTS + e]].vq;
            items[rank].down = sources[table[2 * L31_EXPERTS + e]].vq;
            items[rank].count = same;
            width[rank] = same;
        }
        items[rank].pair[before] = t;
        const uint32_t row = t / L31_USED;
        for (uint32_t k = 0; k < L31_USED; k++) { below += expert[row * L31_USED + k] < e; }
        plan->order[row * L31_USED + below] = t % L31_USED;
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

template <uint32_t BITS>
static cudaError_t l31_decode_run(ds4_l31_moe *moe, float *out, const float *input, const int32_t *selected,
        const float *weights, const uint32_t *table, const void *book, float clamp, const float *shared) {
    const l31_decode_layout l = l31_decode_plan();
    uint8_t *base = (uint8_t *)moe->decode->ptr;
    auto *items = (glmvq_decode_item *)(base + l.items);
    auto *mid = (__nv_bfloat16 *)(base + l.mid);
    auto *partial = (float *)(base + l.partial);
    const cudaStream_t stream = ds4_current_stream();
    const glmvq_launch launch = moe->gvq->launch;
    glmvq_launch_kernel(launch, l31_decode_route, 1, L31_USED, 0, stream, items, selected, table,
        (const glmvq_source *)moe->gvq->model->device->ptr);
    return moe->gain_dot
        ? glmvq_decode_launch<BITS, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_DOT, l31_form>(items, book, input, mid,
            partial, out, weights, clamp, moe->decode_blocks, stream, nullptr, nullptr, shared, launch)
        : glmvq_decode_launch<BITS, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_WEIGHT, l31_form>(items, book, input, mid,
            partial, out, weights, clamp, moe->decode_blocks, stream, nullptr, nullptr, shared, launch);
}

template <uint32_t BITS>
static cudaError_t l31_verify_run(ds4_l31_moe *moe, float *out, const float *input, const int32_t *selected,
        const float *weights, const uint32_t *table, const void *book, uint32_t rows, float clamp,
        const float *shared) {
    const l31_decode_layout l = l31_decode_plan();
    uint8_t *base = (uint8_t *)moe->decode->ptr;
    auto *items = (glmvq_verify_item *)(base + l.vitems);
    auto *plan = (glmvq_verify_plan *)(base + l.plan);
    auto *mid = (__nv_bfloat16 *)(base + l.vmid);
    auto *partial = (float *)(base + l.vpartial);
    const cudaStream_t stream = ds4_current_stream();
    const glmvq_launch launch = moe->gvq->launch;
    glmvq_launch_kernel(launch, l31_verify_route, 1, GLMVQ_VER_PAIRS, 0, stream, items, plan, selected, rows,
        table, (const glmvq_source *)moe->gvq->model->device->ptr);
    return moe->gain_dot
        ? glmvq_verify_launch<BITS, 32, GLMVQ_GAIN_DOT, l31_form>(items, plan, book, input, rows, mid, partial,
            out, weights, clamp, moe->decode_blocks, stream, shared, launch)
        : glmvq_verify_launch<BITS, 32, GLMVQ_GAIN_WEIGHT, l31_form>(items, plan, book, input, rows, mid, partial,
            out, weights, clamp, moe->decode_blocks, stream, shared, launch);
}

// Host grouping by expert (ascending), as the GLM prefill path builds it.
static bool l31_group(ds4_l31_moe *moe, const ds4_gpu_tensor *selected, uint32_t rows,
        uint32_t *offsets) {
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * L31_USED;
    if (pairs > ctx->routes.size() ||
        !ds4_gpu_tensor_read(selected, 0, ctx->routes.data(), pairs * sizeof(int32_t))) { return false; }
    std::vector<uint32_t> counts(L31_EXPERTS, 0), cursor(L31_EXPERTS);
    for (uint64_t i = 0; i < pairs; i++) {
        const int32_t expert = ctx->routes[i];
        if (expert < 0 || expert >= L31_EXPERTS) { return false; }
        for (uint64_t j = i - i % L31_USED; j < i; j++) {
            if (ctx->routes[j] == expert) { return false; }
        }
        counts[expert]++;
    }
    offsets[0] = 0;
    for (uint32_t e = 0; e < L31_EXPERTS; e++) {
        offsets[e + 1] = offsets[e] + counts[e];
        cursor[e] = offsets[e];
    }
    for (uint64_t i = 0; i < pairs; i++) { ctx->grouped[cursor[ctx->routes[i]]++] = (uint32_t)i; }
    return ds4_gpu_tensor_write(ctx->pairs, 0, ctx->grouped.data(), pairs * sizeof(uint32_t)) != 0;
}

template <uint32_t BITS, typename T>
static int l31_mma_run(ds4_l31_moe *moe, float *out, const T *input, const float *weights,
        const uint32_t *matrices, const uint32_t *offsets, uint32_t rows, float clamp, const float *shared) {
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * L31_USED, count = (uint64_t)rows * L31_HIDDEN;
    const l31_mma_layout l = l31_mma_plan(rows);
    uint8_t *base = (uint8_t *)moe->mma->ptr;
    auto *x8 = base + l.x8, *mid8 = base + l.mid8;
    auto *xs = (float *)(base + l.xs), *ms = (float *)(base + l.ms);
    auto *gate = (__nv_bfloat16 *)(base + l.gate), *down = (__nv_bfloat16 *)(base + l.down);
    auto *work = (glmvq_mma_item *)(base + l.work);
    auto *order = (uint32_t *)(base + l.order);
    auto *witems = (beam_w_item *)(base + l.witems);
    auto *token = (uint32_t *)(base + l.token);
    auto *pscale = (float *)(base + l.pscale);
    // 12-bit layers run the warp-specialized kernels on 128-token items;
    // 13-bit layers (and DS4_LING31_MMA_WS=0) glmvq_mma8_kernel on 64-token items.
    const bool wide = BITS == 12 && moe->mma8w && moe->sms;
    moe->items.clear();
    moe->witems.clear();
    if (wide) {
        for (uint32_t m = 0; m < 3; m++) {
            for (uint32_t e = 0; e < L31_EXPERTS; e++) {
                const glmvq_matrix &vq = ctx->model->source[matrices[m * L31_EXPERTS + e]].vq;
                for (uint32_t start = offsets[e]; start < offsets[e + 1]; start += BEAM_W_BT) {
                    moe->witems.push_back({start, min((uint32_t)BEAM_W_BT, offsets[e + 1] - start), vq.gain, vq.main, 0});
                }
            }
        }
    } else {
        for (uint32_t e = 0; e < L31_EXPERTS; e++) {
            for (uint32_t start = offsets[e]; start < offsets[e + 1]; start += L31_MMA_TOKENS) {
                moe->items.push_back({{matrices[e], matrices[L31_EXPERTS + e], matrices[2 * L31_EXPERTS + e]},
                    start, min((uint32_t)L31_MMA_TOKENS, offsets[e + 1] - start)});
            }
        }
    }
    std::fill(moe->token_used.begin(), moe->token_used.begin() + rows, 0);
    for (uint32_t i = 0; i < pairs; i++) {
        const uint32_t token_row = ctx->grouped[i] / L31_USED;
        moe->order[token_row * L31_USED + moe->token_used[token_row]++] = i;
    }
    const cudaStream_t stream = ds4_current_stream();
    const bool copied = wide
        ? cuda_ok(cudaMemcpyAsync(witems, moe->witems.data(), moe->witems.size() * sizeof(beam_w_item),
            cudaMemcpyHostToDevice, stream), "Ling-3.1 warp-specialized items")
        : cuda_ok(cudaMemcpyAsync(work, moe->items.data(), moe->items.size() * sizeof(*work),
            cudaMemcpyHostToDevice, stream), "Ling-3.1 MMA work");
    if (!copied || !cuda_ok(cudaMemcpyAsync(order, moe->order.data(), pairs * sizeof(*order),
            cudaMemcpyHostToDevice, stream), "Ling-3.1 MMA order")) { return 0; }
    const auto *source = (const glmvq_source *)ctx->model->device->ptr;
    const auto *book = (const uint8_t *)ctx->model->source[matrices[0]].vq.book;
    const auto *grouped = (const uint32_t *)ctx->pairs->ptr;
    if (wide) {
        // Tokens quantize once (E4M3, one scale per row, ldmatrix k order) for
        // their eight experts; each pair gets its token row and scale.
        const uint32_t nwide = (uint32_t)moe->witems.size() / 3;
        beam_w_quant<<<rows, GLMVQ_THREADS, 0, stream>>>(x8, xs, input, L31_HIDDEN);
        beam_w_pairs<<<(uint32_t)((pairs + 255) / 256), 256, 0, stream>>>(token, pscale, xs, grouped, L31_USED,
            (uint32_t)pairs);
        cudaError_t err = beam_w_launch<GLMVQ_MMA_GATE>(witems, nwide, book, x8, token, pscale, gate,
            L31_FFN, L31_HIDDEN, nullptr, nullptr, clamp, moe->sms, stream);
        if (err == cudaSuccess) {
            err = beam_w_launch<GLMVQ_MMA_UP>(witems + nwide, nwide, book, x8, token, pscale, gate,
                L31_FFN, L31_HIDDEN, mid8, ms, clamp, moe->sms, stream);
        }
        if (err == cudaSuccess) {
            err = beam_w_down_launch(witems + 2 * nwide, nwide, book, mid8, ms, down, L31_HIDDEN, L31_FFN,
                moe->sms, stream);
        }
        if (!cuda_ok(err, "Ling-3.1 warp-specialized experts")) { return 0; }
    } else {
        const uint32_t nitems = (uint32_t)moe->items.size();
        const int *blocks = moe->mma8_blocks[BITS - 12];
        // Tokens quantize once (E4M3, 1x128 scales) for their eight experts.
        const uint64_t xb = count / GLMVQ_MMA8_K;
        glmvq_mma8_quant<T><<<(uint32_t)((xb * 32 + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0,
            stream>>>(x8, xs, input, xb);
        auto launch = [&](auto mode, const uint8_t *in, const float *scale, uint32_t rows_out, uint32_t cols_in,
                uint8_t *q8, float *q8_scale) {
            constexpr glmvq_mma_stage MODE = decltype(mode)::value;
            constexpr uint32_t shared8 = glmvq_mma8_bytes<L31_MMA_ROWS, L31_MMA_TOKENS, BITS>();
            constexpr uint32_t threads8 = glmvq_mma8_threads<L31_MMA_ROWS, 1>();
            glmvq_mma8_kernel<L31_MMA_ROWS, L31_MMA_TOKENS, 1, MODE, BITS, GLMVQ_CLAMP_SILU><<<
                min(nitems * (rows_out / L31_MMA_ROWS), (uint32_t)blocks[MODE]), threads8, shared8, stream>>>(
                source, work, nitems, book, in, scale, grouped, L31_USED, gate, nullptr, down,
                rows_out, cols_in, clamp, q8, q8_scale);
        };
        launch(std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_GATE>{}, x8, xs, L31_FFN, L31_HIDDEN, nullptr, nullptr);
        // The up epilogue writes down's E4M3 rows and scales directly.
        launch(std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_UP>{}, x8, xs, L31_FFN, L31_HIDDEN, mid8, ms);
        launch(std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_DOWN>{}, mid8, ms, L31_HIDDEN, L31_FFN, nullptr, nullptr);
    }
    glmvq_mma_reduce<__nv_bfloat16, GLMVQ_REDUCE_F32><<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS,
        GLMVQ_THREADS, 0, stream>>>(out, down, weights, grouped, order, L31_USED, L31_HIDDEN, count, shared);
    moe->mma_calls++;
    return cuda_ok(cudaGetLastError(), "Ling-3.1 FP8 grouped MMA");
}

template <typename T>
__global__ void l31_gather(float *output, const T *input, const uint32_t *pairs,
        uint32_t offset, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint32_t token = pairs[offset + i / L31_HIDDEN] / L31_USED;
    output[i] = (float)input[(uint64_t)token * L31_HIDDEN + i % L31_HIDDEN];
}

// Weighted F32 accumulation into `sum` (one expert per token per launch).
__global__ void l31_scatter(float *sum, const float *input, const float *weights,
        const uint32_t *pairs, uint32_t offset, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint32_t pair = pairs[offset + i / L31_HIDDEN];
    const uint64_t target = (uint64_t)(pair / L31_USED) * L31_HIDDEN + i % L31_HIDDEN;
    sum[target] = fmaf(input[i], weights[pair], sum[target]);
}

__global__ void l31_reduce_end(float *out, const float *shared, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = glmvq_reduce_end<GLMVQ_REDUCE_F32>(out[i], shared, i); }
}

// Reference: each expert's BF16 weights restored, cuBLAS BF16 GEMMs. Scratch
// lives in the borrowed context (DS4_LING31_MOE_RESTORE=1 sizes it).
template <typename T>
static int l31_restore_run(ds4_l31_moe *moe, float *out, const T *input, const float *weights,
        const uint32_t *matrices, const uint32_t *offsets, uint32_t rows, float clamp, const float *shared) {
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t count = (uint64_t)rows * L31_HIDDEN;
    const cudaStream_t stream = ds4_current_stream();
    ds4_gpu_tensor *rows_in = ds4_gpu_tensor_alloc((uint64_t)rows * L31_HIDDEN * sizeof(float));
    ds4_gpu_tensor *g = ds4_gpu_tensor_alloc((uint64_t)rows * L31_FFN * sizeof(float));
    ds4_gpu_tensor *u = ds4_gpu_tensor_alloc((uint64_t)rows * L31_FFN * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)rows * L31_FFN * sizeof(float));
    ds4_gpu_tensor *dn = ds4_gpu_tensor_alloc((uint64_t)rows * L31_HIDDEN * sizeof(float));
    bool ok = rows_in && g && u && mid && dn &&
        cuda_ok(cudaMemsetAsync(out, 0, count * sizeof(float), stream), "Ling-3.1 restore clear");
    for (uint32_t e = 0; ok && e < L31_EXPERTS; e++) {
        const uint32_t n = offsets[e + 1] - offsets[e];
        if (!n) { continue; }
        const uint64_t elements = (uint64_t)n * L31_HIDDEN;
        const uint32_t blocks = (uint32_t)((elements + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
        l31_gather<<<blocks, GLMVQ_THREADS, 0, stream>>>((float *)rows_in->ptr, input,
            (const uint32_t *)ctx->pairs->ptr, offsets[e], elements);
        ok = cuda_ok(cudaGetLastError(), "Ling-3.1 expert gather") &&
            ds4_gpu_gvq_linear(ctx, g, matrices[e], rows_in, n, DS4_GVQ_BF16) &&
            ds4_gpu_gvq_linear(ctx, u, matrices[L31_EXPERTS + e], rows_in, n, DS4_GVQ_BF16) &&
            ds4_gpu_l31_swiglu(mid, g, u, n, L31_FFN, clamp, DS4_GVQ_ROWS_F32) &&
            ds4_gpu_gvq_linear(ctx, dn, matrices[2 * L31_EXPERTS + e], mid, n, DS4_GVQ_BF16);
        if (!ok) { break; }
        l31_scatter<<<blocks, GLMVQ_THREADS, 0, stream>>>(out, (const float *)dn->ptr, weights,
            (const uint32_t *)ctx->pairs->ptr, offsets[e], elements);
        ok = cuda_ok(cudaGetLastError(), "Ling-3.1 expert scatter");
    }
    if (ok) {
        l31_reduce_end<<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream>>>(
            out, shared, count);
        ok = cuda_ok(cudaGetLastError(), "Ling-3.1 expert reduce");
    }
    // The temporaries are freed after the stream drains.
    ok = ok && cuda_ok(cudaStreamSynchronize(stream), "Ling-3.1 restore sync");
    ds4_gpu_tensor *all[] = {rows_in, g, u, mid, dn};
    for (auto *t : all) { ds4_gpu_tensor_free(t); }
    moe->restore_calls++;
    return ok;
}

extern "C" int ds4_gpu_l31_moe(ds4_l31_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
        ds4_gvq_rows in_rows, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        const uint32_t *matrices, uint32_t rows, float clamp, const ds4_gpu_tensor *shared) {
    if (!moe || !rows || rows > moe->rows || !matrices || !selected || !gvq_stream(moe->gvq)) { return 0; }
    const uint64_t pairs = (uint64_t)rows * L31_USED, count = (uint64_t)rows * L31_HIDDEN;
    const bool bf16 = in_rows == DS4_GVQ_ROWS_BF16;
    if (pairs > selected->bytes / sizeof(int32_t) || !gvq_floats(weights, pairs) ||
        !gvq_floats(out, count) || !gvq_floats(input, bf16 ? count / 2 : count) ||
        (shared && !gvq_floats(shared, count)) ||
        matrices[0] >= moe->gvq->model->source.size() || ds4_capture_active()) { return 0; }
    const float *add = shared ? (const float *)shared->ptr : nullptr;
    const uint32_t bits = l31_bits(moe, matrices);
    const uint32_t *table = l31_table(moe, matrices, bits);
    if (!table) { return 0; }
    const void *book = moe->gvq->model->source[matrices[0]].vq.book;
    if (!moe->restore && moe->route && rows <= L31_VERIFY_ROWS) {
        if (bf16) { return 0; }   // decode and verify kernels read F32 rows
        const auto *sel = (const int32_t *)selected->ptr;
        const auto *w = (const float *)weights->ptr;
        float *y = (float *)out->ptr;
        const float *x = (const float *)input->ptr;
        cudaError_t err;
        if (rows == 1) {
            err = bits == 13 ? l31_decode_run<13>(moe, y, x, sel, w, table, book, clamp, add)
                             : l31_decode_run<12>(moe, y, x, sel, w, table, book, clamp, add);
            moe->decode_calls++;
        } else {
            err = bits == 13 ? l31_verify_run<13>(moe, y, x, sel, w, table, book, rows, clamp, add)
                             : l31_verify_run<12>(moe, y, x, sel, w, table, book, rows, clamp, add);
            moe->verify_calls++;
        }
        return cuda_ok(err, "Ling-3.1 routed decode");
    }
    std::vector<uint32_t> offsets(L31_EXPERTS + 1);
    if (!l31_group(moe, selected, rows, offsets.data())) { return 0; }
    auto grouped = [&](auto *x) {
        if (!moe->restore && moe->mma8 && moe->mma) {
            return bits == 13
                ? l31_mma_run<13>(moe, (float *)out->ptr, x, (const float *)weights->ptr,
                    matrices, offsets.data(), rows, clamp, add)
                : l31_mma_run<12>(moe, (float *)out->ptr, x, (const float *)weights->ptr,
                    matrices, offsets.data(), rows, clamp, add);
        }
        return l31_restore_run(moe, (float *)out->ptr, x, (const float *)weights->ptr,
            matrices, offsets.data(), rows, clamp, add);
    };
    return bf16 ? grouped((const __nv_bfloat16 *)input->ptr) : grouped((const float *)input->ptr);
}

extern "C" int ds4_gpu_l31_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up, uint32_t rows, uint32_t width, float limit, ds4_gvq_rows out_rows) {
    const uint64_t count = (uint64_t)rows * width;
    if (!count || !gvq_floats(out, count) || !gvq_floats(gate, count) || !gvq_floats(up, count)) { return 0; }
    const uint32_t blocks = (uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
    if (out_rows == DS4_GVQ_ROWS_BF16) {
        glmvq_launch_kernel(gvq_mode(rows), glmvq_swiglu<__nv_bfloat16, GLMVQ_CLAMP_SILU>, blocks, GLMVQ_THREADS,
            0, ds4_current_stream(), (const float *)gate->ptr, (const float *)up->ptr,
            (__nv_bfloat16 *)out->ptr, count, limit);
    } else {
        glmvq_launch_kernel(gvq_mode(rows), glmvq_swiglu<float, GLMVQ_CLAMP_SILU>, blocks, GLMVQ_THREADS,
            0, ds4_current_stream(), (const float *)gate->ptr, (const float *)up->ptr,
            (float *)out->ptr, count, limit);
    }
    return cuda_ok(cudaGetLastError(), "Ling-3.1 SwiGLU");
}

// Prepare the down projection without a materialized SwiGLU row.
extern "C" int ds4_gpu_l31_mlp(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t down,
        const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t rows, float limit) {
    static const bool enabled = !l31_env_off("DS4_LING31_MLP_Q4P");
    if (!enabled || !ctx || !out || !gate || !up || !ctx->direct_q4 || down >= ctx->model->source.size() ||
        !gvq_q4p_ready(ctx, &down, 1, gate, &out, rows) ||
        !gvq_q4p_ready(ctx, &down, 1, up, &out, rows)) { return 0; }
    const glmvq_source &s = ctx->model->source[down];
    if (!gvq_floats(gate, (uint64_t)rows * s.cols) || !gvq_floats(up, (uint64_t)rows * s.cols) ||
        !gvq_floats(out, (uint64_t)rows * s.rows) ||
        (uintptr_t)gate->ptr % alignof(float4) || (uintptr_t)up->ptr % alignof(float4) || !gvq_stream(ctx)) {
        return 0;
    }
    const auto in = glmvq_q4p_layout(ctx->scratch->ptr, rows, s.cols);
    l31_swiglu_prepare<<<rows, GLMVQ_Q4P_PREP, 0, ctx->stream>>>((const float *)gate->ptr,
        (const float *)up->ptr, in.xh, in.xscale, s.cols, limit);
    return cuda_ok(cudaGetLastError(), "Ling-3.1 SwiGLU input") &&
        cuda_ok(glmvq_q4p_gemm((const uint8_t *)s.data, in, (float *)out->ptr, s.rows, s.cols, rows,
            ctx->stream), "Ling-3.1 SwiGLU down");
}

// One token keeps the SIMT pair below GVQ's row limit (see ds4_gpu_gvq_gate_up).
extern "C" int ds4_gpu_l31_gate_up(ds4_gvq_ctx *ctx, ds4_gpu_tensor *mid, uint32_t gate, uint32_t up,
        const ds4_gpu_tensor *input, uint32_t rows, float limit) {
    if (!ctx || !ctx->gate_up || !rows || rows > GLMVQ_Q4M_TOKENS || !ctx->direct_q4 || !ctx->q4_mma ||
        gate >= ctx->model->source.size() || up >= ctx->model->source.size()) { return 0; }
    const glmvq_source &g = ctx->model->source[gate], &u = ctx->model->source[up];
    if (ctx->model->records[gate].type != GVQ_TYPE_Q4 || ctx->model->records[up].type != GVQ_TYPE_Q4 ||
        g.rows != u.rows || g.cols != u.cols || (rows == 1 && g.rows > GVQ_GATE_UP_ONE_ROW_MAX) ||
        !gvq_floats(input, (uint64_t)rows * g.cols) || !gvq_floats(mid, (uint64_t)rows * g.rows)) { return 0; }
    return cuda_ok(glmvq_q4m_swiglu<GLMVQ_Q4M_SWIGLU_SILU>(g.data, u.data, (const float *)input->ptr,
        (float *)mid->ptr, g.rows, g.cols, rows, limit, ctx->stream, gvq_mode(rows)), "Ling-3.1 Q4 gate/up");
}

// kv_b row h*256+d (d < 128) is head h's K-nope row, h*256+128+e its V row.
__global__ void l31_kv_b_restore(__nv_bfloat16 *k_b, __nv_bfloat16 *v_b,
        const uint8_t *weight, const float *scales) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    constexpr uint32_t per_head = (DS4_L31_NOPE + DS4_L31_VALUE) * DS4_L31_LATENT;
    if (i >= (uint64_t)DS4_L31_HEADS * per_head) { return; }
    const uint32_t row = (uint32_t)(i / DS4_L31_LATENT), col = (uint32_t)(i % DS4_L31_LATENT);
    const uint32_t head = row / (DS4_L31_NOPE + DS4_L31_VALUE), r = row % (DS4_L31_NOPE + DS4_L31_VALUE);
    const float scale = scales[(row / GLMVQ_SCALE_BLOCK) * (DS4_L31_LATENT / GLMVQ_SCALE_BLOCK) +
        col / GLMVQ_SCALE_BLOCK];
    const __nv_bfloat16 value = __float2bfloat16_rn(glmvq_mul(glmvq_e4m3(weight[i]), scale));
    if (r < DS4_L31_NOPE) {
        k_b[((uint64_t)head * DS4_L31_LATENT + col) * DS4_L31_NOPE + r] = value;
    } else {
        v_b[((uint64_t)head * DS4_L31_VALUE + (r - DS4_L31_NOPE)) * DS4_L31_LATENT + col] = value;
    }
}

extern "C" int ds4_gpu_l31_kv_b(ds4_gvq_ctx *ctx, ds4_gpu_tensor *k_b, ds4_gpu_tensor *v_b, uint32_t kv_b) {
    constexpr uint64_t half = (uint64_t)DS4_L31_HEADS * DS4_L31_LATENT * DS4_L31_NOPE;
    static_assert(DS4_L31_NOPE == DS4_L31_VALUE, "k_b and v_b hold the same count");
    if (!ctx || kv_b >= ctx->model->source.size() || !k_b || !v_b ||
        k_b->bytes < half * sizeof(__nv_bfloat16) || v_b->bytes < half * sizeof(__nv_bfloat16)) { return 0; }
    const glmvq_source &s = ctx->model->source[kv_b];
    if (s.storage != GLMVQ_WEIGHT_FP8 || !s.scale ||
        s.rows != DS4_L31_HEADS * (DS4_L31_NOPE + DS4_L31_VALUE) || s.cols != DS4_L31_LATENT) { return 0; }
    const uint64_t count = 2 * half;
    l31_kv_b_restore<<<(uint32_t)((count + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (__nv_bfloat16 *)k_b->ptr, (__nv_bfloat16 *)v_b->ptr, (const uint8_t *)s.data, s.scale);
    return cuda_ok(cudaGetLastError(), "Ling-3.1 kv_b restore");
}

extern "C" int ds4_gpu_l31_absorb(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k_b, uint32_t rows) {
    constexpr uint32_t key = DS4_L31_NOPE + DS4_L31_ROPE;
    if (!out || !q || !k_b || !rows ||
        !gvq_floats(out, (uint64_t)rows * DS4_L31_HEADS * DS4_L31_LATENT) ||
        !gvq_floats(q, (uint64_t)rows * DS4_L31_HEADS * key)) { return 0; }
    const dim3 grid(DS4_L31_HEADS, rows);
    ling3vl_qk_absorb_bf16<<<grid, 256, DS4_L31_NOPE * sizeof(float), ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)q->ptr, (const __nv_bfloat16 *)k_b->ptr, rows,
        DS4_L31_HEADS, key, DS4_L31_NOPE, DS4_L31_LATENT);
    return cuda_ok(cudaGetLastError(), "Ling-3.1 MLA absorb");
}

// Tensor-core absorbed attention for decode and verify rows (cuda/l31_flash.cuh),
// its split partials in the GVQ scratch; 0 leaves the caller's Motif path.
extern "C" int ds4_gpu_l31_flash(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *q_abs, const ds4_gpu_tensor *latent, const ds4_gpu_tensor *k_pe,
        uint32_t rows, uint32_t pos0, float scale, uint32_t cap, ds4_l31_latent mode) {
    static int shared = -1;
    if (shared < 0) {
        shared = cuda_ok(cudaFuncSetAttribute(l31_fa_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)L31_FA_SMEM), "Ling-3.1 flash shared memory") &&
            cuda_ok(cudaFuncSetAttribute(l31_fa_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)L31_FA8_SMEM), "Ling-3.1 FP8 flash shared memory");
    }
    const bool fp8 = mode == DS4_L31_LATENT_FP8;
    const uint64_t keys = (uint64_t)pos0 + rows;
    if (!shared || !ctx || !out || !q || !q_abs || !latent || !k_pe || !rows || keys > cap ||
        rows > (fp8 ? L31_FA8_ROWS : L31_FA_ROWS) ||
        !gvq_stream(ctx) || ctx->scratch->bytes < l31_fa_scratch_bytes() ||
        !gvq_floats(out, (uint64_t)rows * DS4_L31_HEADS * DS4_L31_LATENT) ||
        !gvq_floats(q_abs, (uint64_t)rows * DS4_L31_HEADS * DS4_L31_LATENT) ||
        !gvq_floats(q, (uint64_t)rows * DS4_L31_HEADS * (DS4_L31_NOPE + DS4_L31_ROPE)) ||
        latent->bytes < ds4_l31_latent_bytes(cap, mode) ||
        k_pe->bytes < keys * DS4_L31_ROPE * sizeof(__nv_bfloat16)) { return 0; }
    return cuda_ok(l31_fa_launch((float *)out->ptr, ctx->scratch->ptr, (const float *)q->ptr,
        (const float *)q_abs->ptr, latent->ptr, fp8 ? (const int8_t *)latent->ptr + (uint64_t)cap * DS4_L31_LATENT : nullptr,
        (const __nv_bfloat16 *)k_pe->ptr, rows, pos0, scale, gvq_mode(rows), ctx->stream), "Ling-3.1 flash attention");
}

extern "C" int ds4_gpu_l31_expanded_attn(ds4_gpu_tensor *out, ds4_gpu_tensor *lse, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k_full, const ds4_gpu_tensor *value, uint32_t n_query,
        uint32_t query_pos0, uint32_t n_kv, uint32_t kv_pos0, float scale, ds4_l31_segment segment) {
    const uint64_t key = DS4_L31_NOPE + DS4_L31_ROPE;
    const uint64_t bf16 = sizeof(__nv_bfloat16);
    if (!out || !lse || !q || !k_full || !value || !n_query || !n_kv || kv_pos0 > query_pos0 ||
        !gvq_floats(out, (uint64_t)n_query * DS4_L31_HEADS * DS4_L31_VALUE) ||
        !gvq_floats(lse, (uint64_t)n_query * DS4_L31_HEADS) ||
        !gvq_floats(q, (uint64_t)n_query * DS4_L31_HEADS * key) ||
        k_full->bytes < (uint64_t)n_kv * DS4_L31_HEADS * key * bf16 ||
        value->bytes < (uint64_t)n_kv * DS4_L31_HEADS * DS4_L31_VALUE * bf16) { return 0; }
    return ds4_mmq_ling31_prefill_attn_pipe((float *)out->ptr, (float *)lse->ptr, (const float *)q->ptr,
        k_full->ptr, value->ptr, (int)n_query, (int)query_pos0, (int)n_kv, (int)kv_pos0,
        (int)DS4_L31_HEADS, (int)key, (int)DS4_L31_VALUE, scale, segment == DS4_L31_SEGMENT_MERGE,
        ds4_current_stream()) == 0;
}

extern "C" int ds4_gpu_l31_value(ds4_gpu_tensor *out, const ds4_gpu_tensor *latent,
        const ds4_gpu_tensor *v_b, uint32_t rows) {
    if (!out || !latent || !v_b || !rows ||
        !gvq_floats(out, (uint64_t)rows * DS4_L31_HEADS * DS4_L31_VALUE) ||
        !gvq_floats(latent, (uint64_t)rows * DS4_L31_HEADS * DS4_L31_LATENT)) { return 0; }
    const dim3 grid(DS4_L31_HEADS, rows);
    ling3vl_value_project_bf16<<<grid, 256, DS4_L31_LATENT * sizeof(float), ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)latent->ptr, (const __nv_bfloat16 *)v_b->ptr, rows,
        DS4_L31_HEADS, DS4_L31_LATENT, DS4_L31_VALUE);
    return cuda_ok(cudaGetLastError(), "Ling-3.1 MLA value projection");
}

/* Per-head K = [latent . k_b[h] | k_pe] and V = latent . v_b[h]^T as BF16
 * rows [rows][heads][dim] (ds4_gpu_ling3vl_expand_kv with device weights). */
extern "C" int ds4_gpu_l31_expand(ds4_gpu_tensor *k_full, ds4_gpu_tensor *value,
        const ds4_gpu_tensor *latent_cache, const ds4_gpu_tensor *k_pe_cache,
        const ds4_gpu_tensor *k_b, const ds4_gpu_tensor *v_b, uint32_t slot0, uint32_t rows,
        uint32_t cap, ds4_l31_latent mode, ds4_gpu_tensor *scratch) {
    constexpr uint32_t heads = DS4_L31_HEADS, key = DS4_L31_NOPE + DS4_L31_ROPE;
    const uint64_t end = (uint64_t)slot0 + rows, elem = sizeof(__nv_bfloat16);
    const bool fp8 = mode == DS4_L31_LATENT_FP8;
    if (!k_full || !value || !latent_cache || !k_pe_cache || !k_b || !v_b || !rows || !g_cublas_ready ||
        end > cap || k_full->bytes < (uint64_t)rows * heads * key * elem ||
        value->bytes < (uint64_t)rows * heads * DS4_L31_VALUE * elem ||
        latent_cache->bytes < ds4_l31_latent_bytes(cap, mode) || k_pe_cache->bytes < end * DS4_L31_ROPE * elem ||
        (fp8 && (!scratch || scratch->bytes < (uint64_t)rows * DS4_L31_LATENT * elem))) {
        return 0;
    }
    const auto *latent = (const __nv_bfloat16 *)latent_cache->ptr + (uint64_t)slot0 * DS4_L31_LATENT;
    if (fp8) {
        l31_latent_dequant<<<rows, DS4_L31_LATENT / 16, 0, ds4_current_stream()>>>(
            (__nv_bfloat16 *)scratch->ptr, (const uint8_t *)latent_cache->ptr, slot0, cap);
        if (!cuda_ok(cudaGetLastError(), "Ling-3.1 FP8 latent rows")) { return 0; }
        latent = (const __nv_bfloat16 *)scratch->ptr;
    }
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasGemmStridedBatchedEx(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        (int)DS4_L31_NOPE, (int)rows, (int)DS4_L31_LATENT, &alpha,
        k_b->ptr, CUDA_R_16BF, (int)DS4_L31_NOPE, (long long)DS4_L31_LATENT * DS4_L31_NOPE,
        latent, CUDA_R_16BF, (int)DS4_L31_LATENT, 0ll,
        &beta, k_full->ptr, CUDA_R_16BF, (int)(heads * key), (long long)key, (int)heads,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (!cublas_ok(st, "Ling-3.1 MLA expand K")) { return 0; }
    st = cublasGemmStridedBatchedEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)DS4_L31_VALUE, (int)rows, (int)DS4_L31_LATENT, &alpha,
        v_b->ptr, CUDA_R_16BF, (int)DS4_L31_LATENT, (long long)DS4_L31_VALUE * DS4_L31_LATENT,
        latent, CUDA_R_16BF, (int)DS4_L31_LATENT, 0ll,
        &beta, value->ptr, CUDA_R_16BF, (int)(heads * DS4_L31_VALUE), (long long)DS4_L31_VALUE, (int)heads,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (!cublas_ok(st, "Ling-3.1 MLA expand V")) { return 0; }
    const uint64_t quads = (uint64_t)rows * heads * (DS4_L31_ROPE / 4u);
    ling3vl_expand_k_pe<__nv_bfloat16><<<(unsigned)((quads + 255u) / 256u), 256, 0, ds4_current_stream()>>>(
        (__nv_bfloat16 *)k_full->ptr, (const __nv_bfloat16 *)k_pe_cache->ptr, slot0, rows, heads, key,
        DS4_L31_NOPE, DS4_L31_ROPE);
    return cuda_ok(cudaGetLastError(), "Ling-3.1 MLA expand k_pe");
}

extern "C" int ds4_gpu_l31_latent_norm(ds4_gpu_tensor *out, const ds4_gpu_tensor *in,
        const ds4_gpu_tensor *weight, uint32_t rows, float eps) {
    constexpr uint32_t row = DS4_L31_LATENT + DS4_L31_ROPE;
    if (!out || !in || !weight || !rows || !gvq_floats(weight, DS4_L31_LATENT) ||
        !gvq_floats(out, (uint64_t)rows * DS4_L31_LATENT) || !gvq_floats(in, (uint64_t)rows * row)) { return 0; }
    ling3vl_rms_norm<<<rows, 256, 0, ds4_current_stream()>>>((float *)out->ptr, (const float *)in->ptr,
        (const float *)weight->ptr, DS4_L31_LATENT, row, rows, eps);
    return cuda_ok(cudaGetLastError(), "Ling-3.1 latent RMSNorm");
}

extern "C" int ds4_gpu_l31_store_latent(ds4_gpu_tensor *latent, ds4_gpu_tensor *k_pe,
        const ds4_gpu_tensor *kv_norm, const ds4_gpu_tensor *kv_raw, uint32_t rows, uint32_t pos0,
        uint32_t cap, ds4_l31_latent mode) {
    if (mode == DS4_L31_LATENT_BF16) {
        return ds4_gpu_ling3vl_store_latent(latent, k_pe, kv_norm, kv_raw, rows, pos0, cap,
            DS4_L31_LATENT + DS4_L31_ROPE, DS4_L31_LATENT, DS4_L31_ROPE);
    }
    if (!latent || !k_pe || !rows || pos0 > cap || rows > cap - pos0 ||
        latent->bytes < ds4_l31_latent_bytes(cap, mode) ||
        k_pe->bytes < (uint64_t)cap * DS4_L31_ROPE * sizeof(__nv_bfloat16) ||
        !gvq_floats(kv_norm, (uint64_t)rows * DS4_L31_LATENT) ||
        !gvq_floats(kv_raw, (uint64_t)rows * (DS4_L31_LATENT + DS4_L31_ROPE))) { return 0; }
    auto kernel = mode == DS4_L31_LATENT_FP8 ? l31_store_latent<DS4_L31_LATENT_FP8> : l31_store_latent<DS4_L31_LATENT_SIM>;
    kernel<<<rows, L31_LATENT_THREADS, 0, ds4_current_stream()>>>(latent->ptr, (__nv_bfloat16 *)k_pe->ptr,
        (const float *)kv_norm->ptr, (const float *)kv_raw->ptr, pos0, cap);
    return cuda_ok(cudaGetLastError(), "Ling-3.1 FP8 latent store");
}

extern "C" int ds4_gpu_l31_residual_rms(ds4_gvq_ctx *ctx, ds4_gpu_tensor *cur, const ds4_gpu_tensor *branch,
        ds4_gpu_tensor *norm, ds4_gvq_rows norm_rows, ds4_l31_residual mode,
        uint32_t weight, uint32_t rows, uint32_t width, float eps) {
    const uint64_t count = (uint64_t)rows * width;
    const bool bf16 = norm_rows == DS4_GVQ_ROWS_BF16;
    const __nv_bfloat16 *w = (const __nv_bfloat16 *)gvq_weight(ctx, weight, GVQ_TYPE_BF16, width);
    // The block RMS kernel's widths only: one register batch per lane.
    if (!w || !rows || mode < DS4_L31_RES_F32 || mode > DS4_L31_RES_BF16 || width <= GLMVQ_RMS_NARROW || width > GLMVQ_RMS_BATCH * GLMVQ_THREADS ||
        !gvq_floats(cur, count) || !gvq_floats(branch, count) || !gvq_floats(norm, bf16 ? count / 2 : count)) {
        return 0;
    }
    auto launch = [&](auto *out) {
        using N = std::remove_pointer_t<decltype(out)>;
        auto kernel = mode == DS4_L31_RES_PACK ? l31_residual_rms<N, float, __nv_bfloat16>
            : mode == DS4_L31_RES_BF16 ? l31_residual_rms<N, __nv_bfloat16> : l31_residual_rms<N>;
        glmvq_launch_kernel(gvq_mode(rows), kernel, rows, GLMVQ_THREADS, 0, ds4_current_stream(),
            (float *)cur->ptr, (const float *)branch->ptr, w, out, width, eps);
    };
    if (bf16) { launch((__nv_bfloat16 *)norm->ptr); }
    else { launch((float *)norm->ptr); }
    return cuda_ok(cudaGetLastError(), "Ling-3.1 residual RMS");
}

extern "C" int ds4_gpu_l31_residual_unpack(ds4_gpu_tensor *cur, const ds4_gpu_tensor *branch,
        uint32_t rows, uint32_t width) {
    const uint64_t count = (uint64_t)rows * width;
    if (!rows || width <= GLMVQ_RMS_NARROW || width > GLMVQ_RMS_BATCH * GLMVQ_THREADS ||
        !gvq_floats(cur, count) || !gvq_floats(branch, count)) { return 0; }
    l31_residual_unpack<<<rows, GLMVQ_THREADS, 0, ds4_current_stream()>>>(
        (float *)cur->ptr, (const float *)branch->ptr, width);
    return cuda_ok(cudaGetLastError(), "Ling-3.1 residual unpack");
}
