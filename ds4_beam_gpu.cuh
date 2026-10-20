/* Beam-1.0 VQ device runtime; included once by ds4_cuda.cu after the GVQ
 * adapter (whose linear kernels, context and records it uses).
 *
 * Routed experts run GLM's GVQ1 kernels in Beam's form: 5120 -> 2560 ->
 * 5120 rows (whole 2048-input blocks plus a partial one), SiLU times up
 * rounded once (no clamp) and a BF16 routed sum in ascending expert order,
 * as the source's index_add_. K8192 (13-bit) layers stage four experts' mid
 * rows per down pass and verify at most three rows per pass: their 64 KiB
 * book leaves 35 KiB of GB10's 99 KiB block. Each layer's book decides:
 *
 *   rows == 1     decode: device routing, persistent CTAs keep the book
 *   rows 2..4     verify: one pass over the rows' expert union
 *   rows >= 5     prefill: host grouping, FP8 MMA on the E4M3 codewords
 *
 * The rest is Beam's arithmetic (transformers models/beam): RMSNorm as
 * weight x F32-normalized row rounded once, depth-scaled sandwich residuals,
 * q/k norm with RoPE or the NoPE query scale, the gated output and the
 * soft-capped head. F32 rows hold BF16-rounded values; every kernel rounds
 * its inputs where the source has a BF16 tensor, since prefill projections
 * leave F32 sums. */
#include "ds4_beam.h"
#include "cuda/beam_mma8w.cuh"

enum { BEAM_GPU_EXPERTS = DS4_BEAM_EXPERTS, BEAM_GPU_USED = DS4_BEAM_USED, BEAM_GPU_HIDDEN = DS4_BEAM_HIDDEN,
       BEAM_GPU_FFN = DS4_BEAM_FFN, BEAM_GPU_TABLE = 3 * BEAM_GPU_EXPERTS, BEAM_GPU_LAYERS = DS4_BEAM_MOE_LAYERS,
       BEAM_GPU_VERIFY_ROWS = GLMVQ_VER_ROWS, BEAM_GPU_MMA_ROWS = 256, BEAM_GPU_MMA_TOKENS = 64,
       BEAM_GPU_ALIGN = 256, BEAM_GPU_HEAD = DS4_BEAM_HEAD, BEAM_GPU_HEADS = DS4_BEAM_HEADS,
       BEAM_GPU_GROUPS = 8, BEAM_GPU_GROUPS_USED = 4, BEAM_GPU_GROUP = BEAM_GPU_EXPERTS / BEAM_GPU_GROUPS };
static_assert((uint32_t)BEAM_GPU_USED == (uint32_t)GLMVQ_DEC_USED, "GVQ decode geometry");

// 13-bit books stage four experts (down) or three rows (verify gate/up) at a time.
template <uint32_t BITS>
using beam_form = glmvq_moe_form<BEAM_GPU_FFN, GLMVQ_CLAMP_SILU, GLMVQ_REDUCE_BF16, BEAM_GPU_HIDDEN,
    BITS == 13 ? 4u : 8u, BITS == 13 ? 3u : 4u, BITS == 13 ? 4u : 8u>;

static uint64_t beam_align(uint64_t bytes) { return (bytes + BEAM_GPU_ALIGN - 1) / BEAM_GPU_ALIGN * BEAM_GPU_ALIGN; }

// Decode/verify scratch, then the device tables (one 3 x 256 row per layer).
struct beam_decode_layout { uint64_t items, mid, partial, vitems, plan, vmid, vpartial, tables, bytes; };

static beam_decode_layout beam_decode_plan() {
    beam_decode_layout l = {};
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += beam_align(bytes); return here; };
    l.items = take(sizeof(glmvq_decode_item) * BEAM_GPU_USED);
    l.mid = take((uint64_t)BEAM_GPU_USED * BEAM_GPU_FFN * sizeof(__nv_bfloat16));
    l.partial = take((uint64_t)BEAM_GPU_USED * BEAM_GPU_HIDDEN * sizeof(float));
    l.vitems = take(sizeof(glmvq_verify_item) * GLMVQ_VER_PAIRS);
    l.plan = take(sizeof(glmvq_verify_plan));
    l.vmid = take((uint64_t)GLMVQ_VER_PAIRS * BEAM_GPU_FFN * sizeof(__nv_bfloat16));
    l.vpartial = take((uint64_t)GLMVQ_VER_PAIRS * BEAM_GPU_HIDDEN * sizeof(float));
    l.tables = take((uint64_t)BEAM_GPU_LAYERS * BEAM_GPU_TABLE * sizeof(uint32_t));
    l.bytes = at;
    return l;
}

// Prefill: E4M3 input rows and 1x128 scales, gate rows (BF16), the up
// epilogue's E4M3 SwiGLU rows and scales, down rows (BF16), worklists.
struct beam_mma_layout { uint64_t x8, xs, gate, mid8, ms, down, work, order, witems, token, pscale, bytes; };

static beam_mma_layout beam_mma_plan(uint32_t rows) {
    beam_mma_layout l = {};
    const uint64_t pairs = (uint64_t)rows * BEAM_GPU_USED;
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += beam_align(bytes); return here; };
    l.x8 = take((uint64_t)rows * BEAM_GPU_HIDDEN);
    l.xs = take((uint64_t)rows * (BEAM_GPU_HIDDEN / GLMVQ_MMA8_K) * sizeof(float));
    l.gate = take(pairs * BEAM_GPU_FFN * sizeof(__nv_bfloat16));
    l.mid8 = take(pairs * BEAM_GPU_FFN);
    l.ms = take(pairs * (BEAM_GPU_FFN / GLMVQ_MMA8_K) * sizeof(float));
    l.down = take(pairs * BEAM_GPU_HIDDEN * sizeof(__nv_bfloat16));
    l.work = take(pairs * sizeof(glmvq_mma_item));
    l.order = take(pairs * sizeof(uint32_t));
    // Warp-specialized 12-bit path: gate, up and down items (each expert's
    // runs of BEAM_W_BT pairs), and each pair's token row and scale.
    l.witems = take(3 * (BEAM_GPU_EXPERTS + pairs / BEAM_W_BT) * sizeof(beam_w_item));
    l.token = take(pairs * sizeof(uint32_t));
    l.pscale = take(pairs * sizeof(float));
    l.bytes = at;
    return l;
}

struct ds4_beam_moe {
    ds4_gvq_ctx *gvq = nullptr;     // borrowed: stream, routes, grouped, pairs
    ds4_gpu_tensor *decode = nullptr, *mma = nullptr;
    uint32_t rows = 0, decode_blocks = 0;
    int mma8_blocks[2][3] = {};     // [12-bit, 13-bit][gate, up, down]
    uint32_t sms = 0;
    // DS4_BEAM_MMA_WS=0: 12-bit experts on glmvq_mma8_kernel (A/B diagnostic).
    bool gain_dot = true, mma8 = false, mma8w = true;
    std::vector<uint32_t> keys;     // table slot per layer, keyed by expert 0's gate
    std::vector<glmvq_mma_item> items;   // 64-token items (glmvq_mma8_kernel)
    std::vector<beam_w_item> witems;     // gate, up, down 128-token items (beam_w kernels)
    std::vector<uint32_t> order;
    std::vector<uint8_t> token_used;
};

static bool beam_env_off(const char *name) {
    const char *value = getenv(name);
    return value && !strcmp(value, "0");
}

extern "C" uint64_t ds4_gpu_beam_moe_bytes(uint32_t rows) {
    return beam_decode_plan().bytes + (rows > BEAM_GPU_VERIFY_ROWS ? beam_mma_plan(rows).bytes : 0u);
}

extern "C" void ds4_gpu_beam_moe_free(ds4_beam_moe *moe) {
    if (!moe) { return; }
    ds4_gpu_tensor_free(moe->decode);
    ds4_gpu_tensor_free(moe->mma);
    delete moe;
}

template <uint32_t BITS>
static cudaError_t beam_prepare(ds4_beam_moe *moe) {
    uint32_t blocks = 0;
    cudaError_t err = glmvq_decode_prepare<BITS, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_DOT, beam_form<BITS>>(&blocks);
    if (err == cudaSuccess) {
        err = glmvq_decode_prepare<BITS, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_WEIGHT, beam_form<BITS>>(&blocks);
    }
    if (err != cudaSuccess) { return err; }
    moe->decode_blocks = blocks;
    int *mma = moe->mma8_blocks[BITS - 12];
    err = glmvq_mma8_prepare<BEAM_GPU_MMA_ROWS, BEAM_GPU_MMA_TOKENS, 1, GLMVQ_MMA_GATE, BITS, GLMVQ_CLAMP_SILU>(&mma[0]);
    if (err == cudaSuccess) {
        err = glmvq_mma8_prepare<BEAM_GPU_MMA_ROWS, BEAM_GPU_MMA_TOKENS, 1, GLMVQ_MMA_UP, BITS, GLMVQ_CLAMP_SILU>(&mma[1]);
    }
    if (err == cudaSuccess) {
        err = glmvq_mma8_prepare<BEAM_GPU_MMA_ROWS, BEAM_GPU_MMA_TOKENS, 1, GLMVQ_MMA_DOWN, BITS, GLMVQ_CLAMP_SILU>(&mma[2]);
    }
    return err;
}

extern "C" ds4_beam_moe *ds4_gpu_beam_moe_create(ds4_gvq_ctx *gvq, uint32_t rows) {
    if (!gvq || !rows) { return nullptr; }
    ds4_beam_moe *moe = new (std::nothrow) ds4_beam_moe;
    if (!moe) { return nullptr; }
    moe->gvq = gvq;
    moe->rows = rows;
    moe->gain_dot = !beam_env_off("DS4_BEAM_GAIN_DOT");
    moe->mma8w = !beam_env_off("DS4_BEAM_MMA_WS");
    int device = 0, sms = 0;
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess) {
        moe->sms = (uint32_t)sms;
    }
    const cudaError_t err12 = beam_prepare<12>(moe), err13 = beam_prepare<13>(moe);
    if (err12 != cudaSuccess || err13 != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: Beam expert kernels unavailable (%s)\n",
            cudaGetErrorString(err12 != cudaSuccess ? err12 : err13));
        ds4_gpu_beam_moe_free(moe);
        return nullptr;
    }
    moe->mma8 = rows > BEAM_GPU_VERIFY_ROWS;
    moe->decode = ds4_gpu_tensor_alloc(beam_decode_plan().bytes);
    moe->mma = moe->mma8 ? ds4_gpu_tensor_alloc(beam_mma_plan(rows).bytes) : nullptr;
    if (!moe->decode || (moe->mma8 && !moe->mma)) {
        ds4_gpu_beam_moe_free(moe);
        return nullptr;
    }
    moe->items.reserve((uint64_t)rows * BEAM_GPU_USED);
    moe->order.resize((uint64_t)rows * BEAM_GPU_USED);
    moe->token_used.resize(rows);
    return moe;
}

// Shape, book and alignment the direct kernels read; every one of a layer's
// 768 matrices shares its book (12-bit: no plane, 13-bit: plane). Rows are
// 960 (gate/up) or 480 (down) bytes, so only the region starts are aligned.
static bool beam_matrix_fits(const glmvq_source &s, uint32_t projection, const void *book, uint32_t bits) {
    const uint32_t rows = projection == 2 ? BEAM_GPU_HIDDEN : BEAM_GPU_FFN;
    const uint32_t cols = projection == 2 ? BEAM_GPU_FFN : BEAM_GPU_HIDDEN;
    const glmvq_matrix &m = s.vq;
    return s.storage == GLMVQ_WEIGHT_VQ && s.rows == rows && s.cols == cols &&
        m.rows == rows && m.cols == cols && m.precision == GLMVQ_E4M3 &&
        m.main && m.gain && book && m.book == book && (bits == 13) == (m.plane != nullptr) &&
        (uintptr_t)m.main % 128 == 0 && (!m.plane || (uintptr_t)m.plane % 128 == 0) &&
        (uintptr_t)book % 16 == 0 && (uintptr_t)m.gain % sizeof(__half) == 0;
}

static uint32_t beam_bits(const ds4_beam_moe *moe, const uint32_t *matrices) {
    const auto &records = moe->gvq->model->records;
    const auto &book = records[records[matrices[0]].book];
    return book.bytes == 8192u * GLMVQ_VECTOR ? 13u : 12u;
}

// The layer's device table, validated and uploaded on first use.
static const uint32_t *beam_table(ds4_beam_moe *moe, const uint32_t *matrices, uint32_t bits) {
    const auto &sources = moe->gvq->model->source;
    const beam_decode_layout l = beam_decode_plan();
    const auto *tables = (const uint32_t *)((const uint8_t *)moe->decode->ptr + l.tables);
    for (size_t i = 0; i < moe->keys.size(); i++) {
        if (moe->keys[i] == matrices[0]) { return tables + i * BEAM_GPU_TABLE; }
    }
    if (moe->keys.size() == BEAM_GPU_LAYERS) { return nullptr; }
    const void *book = sources[matrices[0]].vq.book;
    for (uint32_t i = 0; i < BEAM_GPU_TABLE; i++) {
        if (matrices[i] >= sources.size() ||
            !beam_matrix_fits(sources[matrices[i]], i / BEAM_GPU_EXPERTS, book, bits)) {
            fprintf(stderr, "ds4: Beam expert matrix %u misses the GVQ1 layout\n", i);
            return nullptr;
        }
    }
    const size_t slot = moe->keys.size();
    if (!ds4_gpu_tensor_write(moe->decode, l.tables + slot * BEAM_GPU_TABLE * sizeof(uint32_t),
            matrices, BEAM_GPU_TABLE * sizeof(uint32_t))) { return nullptr; }
    moe->keys.push_back(matrices[0]);
    return tables + slot * BEAM_GPU_TABLE;
}

// N=1 routing on the GPU: thread t places expert selected[t] at its rank, so
// items run in ascending expert order with no host sync.
__global__ void beam_decode_route(glmvq_decode_item *items, const int32_t *selected,
        const uint32_t *table, const glmvq_source *sources) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x;
    const uint32_t e = min((uint32_t)selected[t], (uint32_t)BEAM_GPU_EXPERTS - 1u);
    uint32_t rank = 0;
    for (uint32_t j = 0; j < BEAM_GPU_USED; j++) { rank += (uint32_t)selected[j] < (uint32_t)selected[t]; }
    items[rank] = {sources[table[e]].vq, sources[table[BEAM_GPU_EXPERTS + e]].vq,
        sources[table[2 * BEAM_GPU_EXPERTS + e]].vq, t};
}

// Union of 2-4 rows' experts in ascending expert order (gvq_verify_route);
// down groups hold at most `capacity` pairs' mid rows.
__global__ void beam_verify_route(glmvq_verify_item *items, glmvq_verify_plan *plan,
        const int32_t *selected, uint32_t rows, const uint32_t *table, const glmvq_source *sources,
        uint32_t capacity) {
    __shared__ uint32_t expert[GLMVQ_VER_PAIRS], lead[GLMVQ_VER_PAIRS], width[GLMVQ_VER_PAIRS];
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x, pairs = rows * BEAM_GPU_USED;
    const bool live = t < pairs;
    if (live) { expert[t] = min((uint32_t)selected[t], (uint32_t)BEAM_GPU_EXPERTS - 1u); }
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
            items[rank].up = sources[table[BEAM_GPU_EXPERTS + e]].vq;
            items[rank].down = sources[table[2 * BEAM_GPU_EXPERTS + e]].vq;
            items[rank].count = same;
            width[rank] = same;
        }
        items[rank].pair[before] = t;
        const uint32_t row = t / BEAM_GPU_USED;
        for (uint32_t k = 0; k < BEAM_GPU_USED; k++) { below += expert[row * BEAM_GPU_USED + k] < e; }
        plan->order[row * BEAM_GPU_USED + below] = t % BEAM_GPU_USED;
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

template <uint32_t BITS>
static cudaError_t beam_decode_run(ds4_beam_moe *moe, float *out, const float *input, const int32_t *selected,
        const float *weights, const uint32_t *table, const void *book) {
    const beam_decode_layout l = beam_decode_plan();
    uint8_t *base = (uint8_t *)moe->decode->ptr;
    auto *items = (glmvq_decode_item *)(base + l.items);
    auto *mid = (__nv_bfloat16 *)(base + l.mid);
    auto *partial = (float *)(base + l.partial);
    const cudaStream_t stream = ds4_current_stream();
    const glmvq_launch launch = moe->gvq->launch;
    glmvq_launch_kernel(launch, beam_decode_route, 1, BEAM_GPU_USED, 0, stream, items, selected, table,
        (const glmvq_source *)moe->gvq->model->device->ptr);
    return moe->gain_dot
        ? glmvq_decode_launch<BITS, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_DOT, beam_form<BITS>>(items, book, input, mid,
            partial, out, weights, INFINITY, moe->decode_blocks, stream, nullptr, nullptr, nullptr, launch)
        : glmvq_decode_launch<BITS, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_WEIGHT, beam_form<BITS>>(items, book, input,
            mid, partial, out, weights, INFINITY, moe->decode_blocks, stream, nullptr, nullptr, nullptr, launch);
}

// Rows above the form's verify width run as consecutive slices (each row's
// arithmetic is the one-row kernel's, so slicing changes no value).
template <uint32_t BITS>
static cudaError_t beam_verify_run(ds4_beam_moe *moe, float *out, const float *input, const int32_t *selected,
        const float *weights, const uint32_t *table, const void *book, uint32_t rows) {
    const beam_decode_layout l = beam_decode_plan();
    uint8_t *base = (uint8_t *)moe->decode->ptr;
    auto *items = (glmvq_verify_item *)(base + l.vitems);
    auto *plan = (glmvq_verify_plan *)(base + l.plan);
    auto *mid = (__nv_bfloat16 *)(base + l.vmid);
    auto *partial = (float *)(base + l.vpartial);
    const cudaStream_t stream = ds4_current_stream();
    const glmvq_launch launch = moe->gvq->launch;
    constexpr uint32_t slice = beam_form<BITS>::VER_ROWS;
    cudaError_t err = cudaSuccess;
    for (uint32_t r = 0; err == cudaSuccess && r < rows; r += slice) {
        const uint32_t n = min(slice, rows - r);
        const uint64_t at = (uint64_t)r * BEAM_GPU_USED;
        if (n == 1) {
            err = beam_decode_run<BITS>(moe, out + (uint64_t)r * BEAM_GPU_HIDDEN, input + (uint64_t)r * BEAM_GPU_HIDDEN,
                selected + at, weights + at, table, book);
            continue;
        }
        glmvq_launch_kernel(launch, beam_verify_route, 1, GLMVQ_VER_PAIRS, 0, stream, items, plan, selected + at, n,
            table, (const glmvq_source *)moe->gvq->model->device->ptr, beam_form<BITS>::VER_GROUP);
        err = moe->gain_dot
            ? glmvq_verify_launch<BITS, 32, GLMVQ_GAIN_DOT, beam_form<BITS>>(items, plan, book,
                input + (uint64_t)r * BEAM_GPU_HIDDEN, n, mid, partial, out + (uint64_t)r * BEAM_GPU_HIDDEN,
                weights + at, INFINITY, moe->decode_blocks, stream, nullptr, launch)
            : glmvq_verify_launch<BITS, 32, GLMVQ_GAIN_WEIGHT, beam_form<BITS>>(items, plan, book,
                input + (uint64_t)r * BEAM_GPU_HIDDEN, n, mid, partial, out + (uint64_t)r * BEAM_GPU_HIDDEN,
                weights + at, INFINITY, moe->decode_blocks, stream, nullptr, launch);
    }
    return err;
}

// Host grouping by expert (ascending), as the GLM prefill path builds it.
static bool beam_group(ds4_beam_moe *moe, const ds4_gpu_tensor *selected, uint32_t rows, uint32_t *offsets) {
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * BEAM_GPU_USED;
    if (pairs > ctx->routes.size() ||
        !ds4_gpu_tensor_read(selected, 0, ctx->routes.data(), pairs * sizeof(int32_t))) { return false; }
    std::vector<uint32_t> counts(BEAM_GPU_EXPERTS, 0), cursor(BEAM_GPU_EXPERTS);
    for (uint64_t i = 0; i < pairs; i++) {
        const int32_t expert = ctx->routes[i];
        if (expert < 0 || expert >= BEAM_GPU_EXPERTS) { return false; }
        for (uint64_t j = i - i % BEAM_GPU_USED; j < i; j++) {
            if (ctx->routes[j] == expert) { return false; }
        }
        counts[expert]++;
    }
    offsets[0] = 0;
    for (uint32_t e = 0; e < BEAM_GPU_EXPERTS; e++) {
        offsets[e + 1] = offsets[e] + counts[e];
        cursor[e] = offsets[e];
    }
    for (uint64_t i = 0; i < pairs; i++) { ctx->grouped[cursor[ctx->routes[i]]++] = (uint32_t)i; }
    return ds4_gpu_tensor_write(ctx->pairs, 0, ctx->grouped.data(), pairs * sizeof(uint32_t)) != 0;
}

template <uint32_t BITS>
static int beam_mma_run(ds4_beam_moe *moe, float *out, const float *input, const float *weights,
        const uint32_t *matrices, const uint32_t *offsets, uint32_t rows) {
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * BEAM_GPU_USED, count = (uint64_t)rows * BEAM_GPU_HIDDEN;
    const beam_mma_layout l = beam_mma_plan(rows);
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
    // 13-bit layers (and DS4_BEAM_MMA_WS=0) glmvq_mma8_kernel on 64-token items.
    const bool wide = BITS == 12 && moe->mma8w && moe->sms;
    moe->items.clear();
    moe->witems.clear();
    if (wide) {
        for (uint32_t m = 0; m < 3; m++) {
            for (uint32_t e = 0; e < BEAM_GPU_EXPERTS; e++) {
                const glmvq_matrix &vq = ctx->model->source[matrices[m * BEAM_GPU_EXPERTS + e]].vq;
                for (uint32_t start = offsets[e]; start < offsets[e + 1]; start += BEAM_W_BT) {
                    moe->witems.push_back({start, min((uint32_t)BEAM_W_BT, offsets[e + 1] - start), vq.gain, vq.main, 0});
                }
            }
        }
    } else {
        for (uint32_t e = 0; e < BEAM_GPU_EXPERTS; e++) {
            for (uint32_t start = offsets[e]; start < offsets[e + 1]; start += BEAM_GPU_MMA_TOKENS) {
                moe->items.push_back({{matrices[e], matrices[BEAM_GPU_EXPERTS + e], matrices[2 * BEAM_GPU_EXPERTS + e]},
                    start, min((uint32_t)BEAM_GPU_MMA_TOKENS, offsets[e + 1] - start)});
            }
        }
    }
    std::fill(moe->token_used.begin(), moe->token_used.begin() + rows, 0);
    for (uint32_t i = 0; i < pairs; i++) {
        const uint32_t token_row = ctx->grouped[i] / BEAM_GPU_USED;
        moe->order[token_row * BEAM_GPU_USED + moe->token_used[token_row]++] = i;
    }
    const cudaStream_t stream = ds4_current_stream();
    const bool copied = wide
        ? cuda_ok(cudaMemcpyAsync(witems, moe->witems.data(), moe->witems.size() * sizeof(beam_w_item),
            cudaMemcpyHostToDevice, stream), "Beam warp-specialized items")
        : cuda_ok(cudaMemcpyAsync(work, moe->items.data(), moe->items.size() * sizeof(*work),
            cudaMemcpyHostToDevice, stream), "Beam MMA work");
    if (!copied || !cuda_ok(cudaMemcpyAsync(order, moe->order.data(), pairs * sizeof(*order),
            cudaMemcpyHostToDevice, stream), "Beam MMA order")) { return 0; }
    const auto *source = (const glmvq_source *)ctx->model->device->ptr;
    const auto *book = (const uint8_t *)ctx->model->source[matrices[0]].vq.book;
    const auto *grouped = (const uint32_t *)ctx->pairs->ptr;
    const uint32_t nitems = (uint32_t)moe->items.size(), nwide = (uint32_t)moe->witems.size() / 3;
    const int *blocks = moe->mma8_blocks[BITS - 12];
    if (wide) {
        // Tokens quantize once (E4M3, one scale per row, ldmatrix k order) for
        // their eight experts; each pair gets its token row and scale.
        beam_w_quant<<<rows, GLMVQ_THREADS, 0, stream>>>(x8, xs, input, BEAM_GPU_HIDDEN);
        beam_w_pairs<<<(uint32_t)((pairs + 255) / 256), 256, 0, stream>>>(token, pscale, xs, grouped, BEAM_GPU_USED,
            (uint32_t)pairs);
        cudaError_t err = beam_w_launch<GLMVQ_MMA_GATE>(witems, nwide, book, x8, token, pscale, gate,
            BEAM_GPU_FFN, BEAM_GPU_HIDDEN, nullptr, nullptr, moe->sms, stream);
        if (err == cudaSuccess) {
            err = beam_w_launch<GLMVQ_MMA_UP>(witems + nwide, nwide, book, x8, token, pscale, gate,
                BEAM_GPU_FFN, BEAM_GPU_HIDDEN, mid8, ms, moe->sms, stream);
        }
        if (err == cudaSuccess) {
            err = beam_w_down_launch(witems + 2 * nwide, nwide, book, mid8, ms, down, BEAM_GPU_HIDDEN, BEAM_GPU_FFN,
                moe->sms, stream);
        }
        if (!cuda_ok(err, "Beam warp-specialized experts")) { return 0; }
    } else {
        // Tokens quantize once (E4M3, 1x128 scales) for their eight experts.
        const uint64_t xb = count / GLMVQ_MMA8_K;
        glmvq_mma8_quant<float><<<(uint32_t)((xb * 32 + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0,
            stream>>>(x8, xs, input, xb);
    }
    auto launch = [&](auto mode, const uint8_t *in, const float *scale, uint32_t rows_out, uint32_t cols_in,
            uint8_t *q8, float *q8_scale) {
        constexpr glmvq_mma_stage MODE = decltype(mode)::value;
        constexpr uint32_t shared8 = glmvq_mma8_bytes<BEAM_GPU_MMA_ROWS, BEAM_GPU_MMA_TOKENS, BITS>();
        constexpr uint32_t threads8 = glmvq_mma8_threads<BEAM_GPU_MMA_ROWS, 1>();
        glmvq_mma8_kernel<BEAM_GPU_MMA_ROWS, BEAM_GPU_MMA_TOKENS, 1, MODE, BITS, GLMVQ_CLAMP_SILU><<<
            min(nitems * (rows_out / BEAM_GPU_MMA_ROWS), (uint32_t)blocks[MODE]), threads8, shared8, stream>>>(
            source, work, nitems, book, in, scale, grouped, BEAM_GPU_USED, gate, nullptr, down,
            rows_out, cols_in, INFINITY, q8, q8_scale);
    };
    if (!wide) {
        launch(std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_GATE>{}, x8, xs, BEAM_GPU_FFN, BEAM_GPU_HIDDEN,
            nullptr, nullptr);
        // The up epilogue writes down's E4M3 rows and scales directly.
        launch(std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_UP>{}, x8, xs, BEAM_GPU_FFN, BEAM_GPU_HIDDEN,
            mid8, ms);
        launch(std::integral_constant<glmvq_mma_stage, GLMVQ_MMA_DOWN>{}, mid8, ms, BEAM_GPU_HIDDEN, BEAM_GPU_FFN,
            nullptr, nullptr);
    }
    glmvq_mma_reduce<__nv_bfloat16, GLMVQ_REDUCE_BF16><<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS,
        GLMVQ_THREADS, 0, stream>>>(out, down, weights, grouped, order, BEAM_GPU_USED, BEAM_GPU_HIDDEN, count);
    return cuda_ok(cudaGetLastError(), "Beam FP8 grouped MMA");
}

extern "C" int ds4_gpu_beam_moe(ds4_beam_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, const uint32_t *matrices, uint32_t rows) {
    if (!moe || !rows || rows > moe->rows || !matrices || !selected || !gvq_stream(moe->gvq)) { return 0; }
    const uint64_t pairs = (uint64_t)rows * BEAM_GPU_USED, count = (uint64_t)rows * BEAM_GPU_HIDDEN;
    if (pairs > selected->bytes / sizeof(int32_t) || !gvq_floats(weights, pairs) ||
        !gvq_floats(out, count) || !gvq_floats(input, count) ||
        matrices[0] >= moe->gvq->model->source.size() || ds4_capture_active()) { return 0; }
    const uint32_t bits = beam_bits(moe, matrices);
    const uint32_t *table = beam_table(moe, matrices, bits);
    if (!table) { return 0; }
    const void *book = moe->gvq->model->source[matrices[0]].vq.book;
    if (rows <= BEAM_GPU_VERIFY_ROWS) {
        const auto *sel = (const int32_t *)selected->ptr;
        const auto *w = (const float *)weights->ptr;
        float *y = (float *)out->ptr;
        const float *x = (const float *)input->ptr;
        const cudaError_t err = rows == 1
            ? (bits == 13 ? beam_decode_run<13>(moe, y, x, sel, w, table, book)
                          : beam_decode_run<12>(moe, y, x, sel, w, table, book))
            : (bits == 13 ? beam_verify_run<13>(moe, y, x, sel, w, table, book, rows)
                          : beam_verify_run<12>(moe, y, x, sel, w, table, book, rows));
        return cuda_ok(err, "Beam routed decode");
    }
    std::vector<uint32_t> offsets(BEAM_GPU_EXPERTS + 1);
    if (!moe->mma8 || !moe->mma || !beam_group(moe, selected, rows, offsets.data())) { return 0; }
    return bits == 13
        ? beam_mma_run<13>(moe, (float *)out->ptr, (const float *)input->ptr, (const float *)weights->ptr,
            matrices, offsets.data(), rows)
        : beam_mma_run<12>(moe, (float *)out->ptr, (const float *)input->ptr, (const float *)weights->ptr,
            matrices, offsets.data(), rows);
}

/* Group-limited sigmoid router (DeepSeek V3's noaux_tc, models/beam): one
 * warp per row, lane l holds experts 8l..8l+7, so group g (32 experts) is
 * lanes 4g..4g+3. s = sigmoid(logit) in double (fast-math expf would move
 * near-tied choices); choice c = s + bias; a group scores the sum of its two
 * best c; the best four groups admit their experts to the top-8 by c; the
 * weights are the unbiased s over their sum + 1e-20. Ties take the lower
 * index, as torch.topk does on these sizes. */
__global__ void beam_router_kernel(int32_t *ids, float *weights, const float *logits, const float *bias,
        uint32_t rows) {
    const uint32_t lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) { return; }
    float s[8], c[8];
    for (uint32_t j = 0; j < 8; j++) {
        const uint32_t e = lane * 8 + j;
        s[j] = (float)(1.0 / (1.0 + exp(-(double)logits[(uint64_t)row * BEAM_GPU_EXPERTS + e])));
        c[j] = glmvq_add(s[j], bias[e]);
    }
    // This lane's best two choices, then the group's over its four lanes.
    float a = -INFINITY, b = -INFINITY;
    for (uint32_t j = 0; j < 8; j++) {
        if (c[j] > a) { b = a; a = c[j]; } else if (c[j] > b) { b = c[j]; }
    }
    for (uint32_t stride = 1; stride < 4; stride <<= 1) {
        const float oa = __shfl_xor_sync(0xffffffffu, a, stride), ob = __shfl_xor_sync(0xffffffffu, b, stride);
        const float hi = fmaxf(a, oa), lo = fmaxf(fminf(a, oa), fmaxf(b, ob));
        a = hi;
        b = lo;
    }
    const float score = glmvq_add(a, b);
    // Four groups by score, lowest group index first on ties.
    bool open = false;
    uint32_t taken = 0;
    for (uint32_t pick = 0; pick < BEAM_GPU_GROUPS_USED; pick++) {
        float best = -INFINITY;
        uint32_t best_group = BEAM_GPU_GROUPS;
        for (uint32_t gi = 0; gi < BEAM_GPU_GROUPS; gi++) {
            const float v = __shfl_sync(0xffffffffu, score, gi * 4);
            if (!(taken >> gi & 1u) && (v > best || best_group == BEAM_GPU_GROUPS)) { best = v; best_group = gi; }
        }
        taken |= 1u << best_group;
        open = open || lane / 4 == best_group;
    }
    if (!open) {
        for (uint32_t j = 0; j < 8; j++) { c[j] = -INFINITY; }
    }
    // Top-8 by choice over the open groups.
    float sum = 0.0f;
    uint32_t chosen[BEAM_GPU_USED];
    float chosen_s[BEAM_GPU_USED];
    for (uint32_t k = 0; k < BEAM_GPU_USED; k++) {
        float best = -INFINITY;
        uint32_t index = UINT32_MAX;
        for (uint32_t j = 0; j < 8; j++) {
            if (c[j] > best || (index == UINT32_MAX && c[j] == best)) { best = c[j]; index = lane * 8 + j; }
        }
        for (uint32_t stride = 16; stride; stride >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, stride);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, index, stride);
            if (ob > best || (ob == best && oi < index)) { best = ob; index = oi; }
        }
        const uint32_t owner = index / 8, slot = index % 8;
        const float picked = __shfl_sync(0xffffffffu, s[slot], owner);
        if (lane == owner) { c[slot] = -INFINITY; }
        chosen[k] = index;
        chosen_s[k] = picked;
        sum = glmvq_add(sum, picked);
    }
    if (lane) { return; }
    const float denom = glmvq_add(sum, 1e-20f);
    for (uint32_t k = 0; k < BEAM_GPU_USED; k++) {
        ids[(uint64_t)row * BEAM_GPU_USED + k] = (int32_t)chosen[k];
        weights[(uint64_t)row * BEAM_GPU_USED + k] = __fdiv_rn(chosen_s[k], denom);
    }
}

extern "C" int ds4_gpu_beam_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        ds4_gpu_tensor *logits, const ds4_gpu_tensor *input, uint32_t router, uint32_t bias, uint32_t rows) {
    const auto *w = (const float *)gvq_weight(ctx, router, GVQ_TYPE_F32, (uint64_t)BEAM_GPU_EXPERTS * BEAM_GPU_HIDDEN);
    const auto *b = (const float *)gvq_weight(ctx, bias, GVQ_TYPE_F32, BEAM_GPU_EXPERTS);
    if (!w || !b || !rows || !gvq_floats(logits, (uint64_t)rows * BEAM_GPU_EXPERTS) ||
        !gvq_floats(input, (uint64_t)rows * BEAM_GPU_HIDDEN) || !gvq_floats(weights, (uint64_t)rows * BEAM_GPU_USED) ||
        selected->bytes < (uint64_t)rows * BEAM_GPU_USED * sizeof(int32_t) || !gvq_stream(ctx)) { return 0; }
    // F.linear(x.float(), router): the handle's TF32 mode would move near-tied routes.
    const float alpha = 1.0f, beta = 0.0f;
    if (!cublas_ok(cublasGemmEx(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N, BEAM_GPU_EXPERTS, (int)rows, BEAM_GPU_HIDDEN,
            &alpha, w, CUDA_R_32F, BEAM_GPU_HIDDEN, input->ptr, CUDA_R_32F, BEAM_GPU_HIDDEN, &beta, logits->ptr,
            CUDA_R_32F, BEAM_GPU_EXPERTS, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT), "Beam router logits")) {
        return 0;
    }
    enum { WARPS = 8 };
    beam_router_kernel<<<(rows + WARPS - 1) / WARPS, WARPS * 32, 0, ds4_current_stream()>>>(
        (int32_t *)selected->ptr, (float *)weights->ptr, (const float *)logits->ptr, b, rows);
    return cuda_ok(cudaGetLastError(), "Beam router");
}

enum { BEAM_ROW_THREADS = 256, BEAM_ROW_VALUES = BEAM_GPU_HIDDEN / BEAM_ROW_THREADS };
static const float BEAM_RMS_EPS = 1e-6f;

// Sum of `value` over the block (256 threads); every thread gets it.
__device__ __forceinline__ float beam_block_sum(float value, float *scratch) {
    for (uint32_t stride = 16; stride; stride >>= 1) { value += __shfl_xor_sync(0xffffffffu, value, stride); }
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x / 32] = value; }
    __syncthreads();
    float sum = 0.0f;
    for (uint32_t w = 0; w < BEAM_ROW_THREADS / 32; w++) { sum += scratch[w]; }
    __syncthreads();
    return sum;
}

// rms(x) of one 5120-wide row held as 20 values per thread: torch's
// x * rsqrt(mean(x^2) + eps), times the BF16 weight in F32, rounded once.
__device__ __forceinline__ void beam_row_rms(const float *x, const __nv_bfloat16 *weight, float *out,
        float *scratch) {
    float sum = 0.0f;
    for (uint32_t j = 0; j < BEAM_ROW_VALUES; j++) { sum = fmaf(x[j], x[j], sum); }
    const float inv = __frsqrt_rn(glmvq_add(__fdiv_rn(beam_block_sum(sum, scratch), (float)BEAM_GPU_HIDDEN),
        BEAM_RMS_EPS));
    for (uint32_t j = 0; j < BEAM_ROW_VALUES; j++) {
        const uint32_t col = threadIdx.x + j * BEAM_ROW_THREADS;
        out[j] = glmvq_round(glmvq_mul(__bfloat162float(weight[col]), glmvq_mul(x[j], inv)));
    }
}

__global__ void beam_rms_kernel(float *out, const float *input, const __nv_bfloat16 *weight) {
    __shared__ float scratch[BEAM_ROW_THREADS / 32];
    const uint64_t base = (uint64_t)blockIdx.x * BEAM_GPU_HIDDEN;
    float x[BEAM_ROW_VALUES], y[BEAM_ROW_VALUES];
    for (uint32_t j = 0; j < BEAM_ROW_VALUES; j++) {
        x[j] = glmvq_round(input[base + threadIdx.x + j * BEAM_ROW_THREADS]);
    }
    beam_row_rms(x, weight, y, scratch);
    for (uint32_t j = 0; j < BEAM_ROW_VALUES; j++) { out[base + threadIdx.x + j * BEAM_ROW_THREADS] = y[j]; }
}

extern "C" int ds4_gpu_beam_rms(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        uint32_t weight, uint32_t rows) {
    const auto *w = (const __nv_bfloat16 *)gvq_weight(ctx, weight, GVQ_TYPE_BF16, BEAM_GPU_HIDDEN);
    const uint64_t count = (uint64_t)rows * BEAM_GPU_HIDDEN;
    if (!w || !rows || !gvq_floats(out, count) || !gvq_floats(x, count)) { return 0; }
    beam_rms_kernel<<<rows, BEAM_ROW_THREADS, 0, ds4_current_stream()>>>((float *)out->ptr,
        (const float *)x->ptr, w);
    return cuda_ok(cudaGetLastError(), "Beam RMS");
}

/* Sandwich residual of one row: t = rms(scale * branch) * post, then
 * cur = t * c1 + cur * c2 with each product and the sum at the BF16
 * boundary (the source's BF16 tensor ops), then the next pre-norm. */
__global__ void beam_residual_kernel(float *cur, const float *branch, const __nv_bfloat16 *post, float *norm,
        const __nv_bfloat16 *next, float c1, float c2, float scale) {
    __shared__ float scratch[BEAM_ROW_THREADS / 32];
    const uint64_t base = (uint64_t)blockIdx.x * BEAM_GPU_HIDDEN;
    float b[BEAM_ROW_VALUES], t[BEAM_ROW_VALUES];
    for (uint32_t j = 0; j < BEAM_ROW_VALUES; j++) {
        b[j] = glmvq_round(branch[base + threadIdx.x + j * BEAM_ROW_THREADS]);
        if (scale != 1.0f) { b[j] = glmvq_round(glmvq_mul(b[j], scale)); }
    }
    beam_row_rms(b, post, t, scratch);
    for (uint32_t j = 0; j < BEAM_ROW_VALUES; j++) {
        const uint64_t at = base + threadIdx.x + j * BEAM_ROW_THREADS;
        b[j] = glmvq_round(glmvq_add(glmvq_round(glmvq_mul(t[j], c1)), glmvq_round(glmvq_mul(cur[at], c2))));
        cur[at] = b[j];
    }
    if (!next) { return; }
    beam_row_rms(b, next, t, scratch);
    for (uint32_t j = 0; j < BEAM_ROW_VALUES; j++) { norm[base + threadIdx.x + j * BEAM_ROW_THREADS] = t[j]; }
}

extern "C" int ds4_gpu_beam_residual(ds4_gvq_ctx *ctx, ds4_gpu_tensor *cur, const ds4_gpu_tensor *branch,
        uint32_t post, ds4_gpu_tensor *norm, uint32_t next, uint32_t rows, uint32_t index, float scale) {
    const auto *p = (const __nv_bfloat16 *)gvq_weight(ctx, post, GVQ_TYPE_BF16, BEAM_GPU_HIDDEN);
    const auto *n = next == DS4_GVQ_NO_LINK ? nullptr
        : (const __nv_bfloat16 *)gvq_weight(ctx, next, GVQ_TYPE_BF16, BEAM_GPU_HIDDEN);
    const uint64_t count = (uint64_t)rows * BEAM_GPU_HIDDEN;
    if (!p || (next != DS4_GVQ_NO_LINK && !n) || !rows || index < 2 || !gvq_floats(cur, count) ||
        !gvq_floats(branch, count) || (n && !gvq_floats(norm, count))) { return 0; }
    // Python floats: 1/sqrt(i) and sqrt((i-1)/i) in double, F32 in the op.
    const float c1 = (float)(1.0 / sqrt((double)index)), c2 = (float)sqrt((double)(index - 1) / index);
    beam_residual_kernel<<<rows, BEAM_ROW_THREADS, 0, ds4_current_stream()>>>((float *)cur->ptr,
        (const float *)branch->ptr, p, n ? (float *)norm->ptr : nullptr, n, c1, c2, scale);
    return cuda_ok(cudaGetLastError(), "Beam residual");
}

/* The F32 cos/sin of the F32 angle pos * inv_freq (Olmo3RotaryEmbedding;
 * sin/cos in double, fast-math sinf is unusable at 1M positions) and the
 * query scale, one block per position. They depend on the position alone:
 * GB10 runs FP64 at 1/64 rate, so every layer's q/k kernel reads this row
 * instead of evaluating them per element. */
__global__ void beam_rope_kernel(float *rope, uint32_t pos0) {
    const uint32_t j = threadIdx.x, pos = pos0 + blockIdx.x;
    float *t = rope + (uint64_t)blockIdx.x * (BEAM_GPU_HEAD + 1u);
    // inv_freq = 1 / theta ** (2j / 128) and the angle as torch's F32 ops
    // (pow rounded to F32, then the division).
    const float freq = __fdiv_rn(1.0f, (float)pow(10000.0, (double)(2u * j) / (double)BEAM_GPU_HEAD));
    const float angle = glmvq_mul((float)pos, freq);
    t[j] = (float)cos((double)angle);
    t[BEAM_GPU_HEAD / 2 + j] = (float)sin((double)angle);
    if (j == 0) {
        // exp(0.1 * log1p((pos + 1) / 2048)), the scalable-softmax query
        // scale, as torch's three F32 ops.
        const float s = glmvq_mul(0.1f, (float)log1p((double)(pos + 1u) / 2048.0));
        t[BEAM_GPU_HEAD] = (float)exp((double)s);
    }
}

extern "C" int ds4_gpu_beam_rope(ds4_gpu_tensor *rope, uint32_t rows, uint32_t pos0) {
    if (!rows || !gvq_floats(rope, (uint64_t)rows * (BEAM_GPU_HEAD + 1u))) { return 0; }
    beam_rope_kernel<<<rows, BEAM_GPU_HEAD / 2, 0, ds4_current_stream()>>>((float *)rope->ptr, pos0);
    return cuda_ok(cudaGetLastError(), "Beam RoPE rows");
}

/* q/k RMSNorm per 128-wide head, one block per (head, row): rounded once
 * after the weight, then rotate_half RoPE with the row's cos/sin, the full
 * layers' query scale, or nothing. Products and sums round as the source's
 * separate F32 ops. */
__global__ void beam_qk_kernel(float *x, const __nv_bfloat16 *weight, uint32_t heads, const float *rope,
        ds4_beam_qk mode) {
    __shared__ float scratch[BEAM_GPU_HEAD / 32];
    __shared__ float y[BEAM_GPU_HEAD];
    const uint32_t h = blockIdx.x, r = blockIdx.y, i = threadIdx.x;
    float *p = x + ((uint64_t)r * heads + h) * BEAM_GPU_HEAD;
    const float v = glmvq_round(p[i]);
    float sum = v * v;
    for (uint32_t stride = 16; stride; stride >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, stride); }
    if ((i & 31) == 0) { scratch[i / 32] = sum; }
    __syncthreads();
    sum = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    const float inv = __frsqrt_rn(glmvq_add(__fdiv_rn(sum, (float)BEAM_GPU_HEAD), BEAM_RMS_EPS));
    y[i] = glmvq_round(glmvq_mul(__bfloat162float(weight[i]), glmvq_mul(v, inv)));
    __syncthreads();
    const float *t = rope + (uint64_t)r * (BEAM_GPU_HEAD + 1u);
    if (mode == DS4_BEAM_QK_ROPE) {
        const uint32_t j = i % (BEAM_GPU_HEAD / 2);
        const float c = t[j], s = t[BEAM_GPU_HEAD / 2 + j];
        const float a = y[j], b = y[j + BEAM_GPU_HEAD / 2];
        p[i] = i < BEAM_GPU_HEAD / 2 ? glmvq_round(glmvq_add(glmvq_mul(a, c), glmvq_mul(-b, s)))
                                     : glmvq_round(glmvq_add(glmvq_mul(b, c), glmvq_mul(a, s)));
    } else if (mode == DS4_BEAM_QK_SCALE) {
        p[i] = glmvq_round(glmvq_mul(y[i], t[BEAM_GPU_HEAD]));
    } else {
        p[i] = y[i];
    }
}

extern "C" int ds4_gpu_beam_qk(ds4_gvq_ctx *ctx, ds4_gpu_tensor *x, uint32_t weight, uint32_t heads,
        uint32_t rows, const ds4_gpu_tensor *rope, ds4_beam_qk mode) {
    const auto *w = (const __nv_bfloat16 *)gvq_weight(ctx, weight, GVQ_TYPE_BF16, BEAM_GPU_HEAD);
    if (!w || !rows || !heads || !gvq_floats(x, (uint64_t)rows * heads * BEAM_GPU_HEAD) ||
        !gvq_floats(rope, (uint64_t)rows * (BEAM_GPU_HEAD + 1u))) { return 0; }
    beam_qk_kernel<<<dim3(heads, rows), BEAM_GPU_HEAD, 0, ds4_current_stream()>>>((float *)x->ptr, w, heads,
        (const float *)rope->ptr, mode);
    return cuda_ok(cudaGetLastError(), "Beam q/k norm");
}

// heads = attn * sigmoid(gate + bias[head]): BF16 add, sigmoid and product.
__global__ void beam_gate_kernel(float *heads, const float *gate, const __nv_bfloat16 *bias, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint32_t head = (uint32_t)(i / BEAM_GPU_HEAD % BEAM_GPU_HEADS);
    const float g = glmvq_round(glmvq_add(glmvq_round(gate[i]), __bfloat162float(bias[head])));
    heads[i] = glmvq_round(glmvq_mul(glmvq_round(heads[i]), glmvq_round(glmvq_sigmoid(g))));
}

extern "C" int ds4_gpu_beam_gate(ds4_gvq_ctx *ctx, ds4_gpu_tensor *heads, const ds4_gpu_tensor *gate,
        uint32_t bias, uint32_t rows) {
    const auto *b = (const __nv_bfloat16 *)gvq_weight(ctx, bias, GVQ_TYPE_BF16, BEAM_GPU_HEADS);
    const uint64_t count = (uint64_t)rows * BEAM_GPU_HEADS * BEAM_GPU_HEAD;
    if (!b || !rows || !gvq_floats(heads, count) || !gvq_floats(gate, count)) { return 0; }
    beam_gate_kernel<<<(uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0,
        ds4_current_stream()>>>((float *)heads->ptr, (const float *)gate->ptr, b, count);
    return cuda_ok(cudaGetLastError(), "Beam gate");
}

/* Decode attention of one query row over keys [first, last]: a block per
 * (key split, KV head) serves the KV head's six query heads. Each warp walks
 * 32-key tiles: lane = key for the scores (its K segment read whole, F32
 * dots against the shared query rows, the row's K scale applied to the
 * sum), then lane = four dimensions for P.V with the V scale folded into
 * the weights. Warps keep their own online softmax; the block merges them
 * into (max, sum, output) per head and split, or the normalized output when
 * one split covers every key. Rows are Solar's FP8 (E4M3 K | E4M3 V | f16 K
 * and V scales per KV head) or f16 [K | V], at slot key % kv_cap. The
 * per-key warp walks this replaces (Solar's grouped split kernel, EXAONE's
 * pair kernel) read the rows at ~25 GB/s. */
enum { BEAM_ATTN_THREADS = 128, BEAM_ATTN_WARPS = BEAM_ATTN_THREADS / 32, BEAM_ATTN_TILE = 32,
       BEAM_ATTN_GROUP = DS4_BEAM_HEADS / DS4_BEAM_KV_HEADS, BEAM_ATTN_KV = DS4_BEAM_KV_HEADS * BEAM_GPU_HEAD,
       BEAM_ATTN_SPLITS = 48 };
static_assert((uint32_t)BEAM_GPU_HEAD == (uint32_t)BEAM_ATTN_THREADS && BEAM_GPU_HEAD == 4 * 32,
    "one thread per head dimension");

// Four E4M3 codes (the bytes of `w`, lowest first) as floats; exact.
static __device__ __forceinline__ void beam_e4m3x4(uint32_t w, float f[4]) {
    const float2 a = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(w & 0xffffu), __NV_E4M3)));
    const float2 b = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(w >> 16), __NV_E4M3)));
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}

// Four f16 values (two words) as floats.
static __device__ __forceinline__ void beam_f16x4(uint2 w, float f[4]) {
    const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&w.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&w.y));
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}

template <bool FP8>
__global__ void __launch_bounds__(BEAM_ATTN_THREADS) beam_attn_decode_kernel(float *heads, float *partials,
        const float *q, const uint8_t *kv, uint32_t kv_cap, uint32_t first, uint32_t last, uint32_t chunk,
        uint32_t splits, float scale) {
    constexpr uint32_t G = BEAM_ATTN_GROUP, D = BEAM_GPU_HEAD, W = BEAM_ATTN_WARPS, T = BEAM_ATTN_TILE;
    constexpr uint32_t CODE = FP8 ? 1u : 2u;  // bytes per K or V value
    constexpr uint64_t ROW = 2ull * BEAM_ATTN_KV * CODE + (FP8 ? 4u * DS4_BEAM_KV_HEADS : 0u);
    __shared__ __align__(16) float qs[G][D];
    __shared__ __align__(16) float ps[W][T][8];
    __shared__ float ms[W][G], ls[W][G];
    __shared__ __align__(16) float os[W][G][D];
    const uint32_t split = blockIdx.x, kvh = blockIdx.y, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t c0 = first + split * chunk, c1 = min(last, c0 + chunk - 1u);
    for (uint32_t i = threadIdx.x; i < G * D; i += BEAM_ATTN_THREADS) {
        qs[i / D][i % D] = q[(uint64_t)kvh * G * D + i];
    }
    __syncthreads();
    const uint64_t koff = (uint64_t)kvh * D * CODE, voff = (uint64_t)BEAM_ATTN_KV * CODE + koff;
    float m[G], l[G], acc[G][4];
#pragma unroll
    for (uint32_t h = 0; h < G; h++) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) { acc[h][i] = 0.0f; }
    }
    for (uint32_t t0 = c0 + warp * T; t0 <= c1; t0 += W * T) {
        const uint32_t key = t0 + lane;
        const bool valid = key <= c1;
        const uint8_t *row = kv + (uint64_t)((valid ? key : t0) % kv_cap) * ROW;
        float s[G];
#pragma unroll
        for (uint32_t h = 0; h < G; h++) { s[h] = 0.0f; }
#pragma unroll 2
        for (uint32_t c = 0; c < D; c += 16u / CODE) {
            const uint4 w = *reinterpret_cast<const uint4 *>(row + koff + c * CODE);
            float f[16 / CODE];
            if constexpr (FP8) {
                beam_e4m3x4(w.x, f); beam_e4m3x4(w.y, f + 4); beam_e4m3x4(w.z, f + 8); beam_e4m3x4(w.w, f + 12);
            } else {
                beam_f16x4(make_uint2(w.x, w.y), f); beam_f16x4(make_uint2(w.z, w.w), f + 4);
            }
#pragma unroll
            for (uint32_t d = 0; d < 16u / CODE; d += 4) {
#pragma unroll
                for (uint32_t h = 0; h < G; h++) {
                    const float4 qq = *reinterpret_cast<const float4 *>(&qs[h][c + d]);
                    s[h] = fmaf(f[d], qq.x, s[h]);
                    s[h] = fmaf(f[d + 1], qq.y, s[h]);
                    s[h] = fmaf(f[d + 2], qq.z, s[h]);
                    s[h] = fmaf(f[d + 3], qq.w, s[h]);
                }
            }
        }
        float kscale = scale, vscale = 1.0f;
        if constexpr (FP8) {
            const __half *sc = reinterpret_cast<const __half *>(row + 2ull * BEAM_ATTN_KV);
            kscale *= __half2float(sc[kvh]);
            vscale = __half2float(sc[DS4_BEAM_KV_HEADS + kvh]);
        }
#pragma unroll
        for (uint32_t h = 0; h < G; h++) {
            const float sc = valid ? s[h] * kscale : -INFINITY;
            float top = sc;
#pragma unroll
            for (uint32_t o = 16; o; o >>= 1) { top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, o)); }
            const float mn = fmaxf(m[h], top), r = __expf(m[h] - mn), p = valid ? __expf(sc - mn) : 0.0f;
            l[h] = l[h] * r + p;
#pragma unroll
            for (uint32_t i = 0; i < 4; i++) { acc[h][i] *= r; }
            m[h] = mn;
            ps[warp][lane][h] = p * vscale;
        }
        __syncwarp();
        const uint32_t n = min(T, c1 - t0 + 1u);
#pragma unroll 4
        for (uint32_t j = 0; j < n; j++) {
            const uint8_t *vrow = kv + (uint64_t)((t0 + j) % kv_cap) * ROW + voff;
            float f[4];
            if constexpr (FP8) {
                beam_e4m3x4(*reinterpret_cast<const uint32_t *>(vrow + 4u * lane), f);
            } else {
                beam_f16x4(*reinterpret_cast<const uint2 *>(vrow + 8u * lane), f);
            }
            const float4 pa = *reinterpret_cast<const float4 *>(&ps[warp][j][0]);
            const float2 pb = *reinterpret_cast<const float2 *>(&ps[warp][j][4]);
            const float p[G] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y};
#pragma unroll
            for (uint32_t h = 0; h < G; h++) {
#pragma unroll
                for (uint32_t i = 0; i < 4; i++) { acc[h][i] = fmaf(p[h], f[i], acc[h][i]); }
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (uint32_t h = 0; h < G; h++) {
#pragma unroll
        for (uint32_t o = 16; o; o >>= 1) { l[h] += __shfl_xor_sync(0xffffffffu, l[h], o); }
        if (lane == 0) { ms[warp][h] = m[h]; ls[warp][h] = l[h]; }
        *reinterpret_cast<float4 *>(&os[warp][h][4u * lane]) = make_float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
    }
    __syncthreads();
    const uint32_t d = threadIdx.x;
#pragma unroll
    for (uint32_t h = 0; h < G; h++) {
        float top = -INFINITY, sum = 0.0f, o = 0.0f;
#pragma unroll
        for (uint32_t w = 0; w < W; w++) { top = fmaxf(top, ms[w][h]); }
#pragma unroll
        for (uint32_t w = 0; w < W; w++) {
            const float e = ms[w][h] == -INFINITY ? 0.0f : __expf(ms[w][h] - top);
            sum += ls[w][h] * e;
            o += os[w][h][d] * e;
        }
        const uint32_t head = kvh * G + h;
        if (splits == 1u) {
            heads[(uint64_t)head * D + d] = o / sum;
        } else {
            float *part = partials + ((uint64_t)head * splits + split) * (D + 2u);
            if (d == 0) { part[0] = top; part[1] = sum; }
            part[2u + d] = o;
        }
    }
}

/* FP8 rows on sm_90+: a block per key split serves all eight KV heads (warp
 * = KV head). Whole 2,080-byte rows arrive by one TMA bulk copy per key into
 * a ring of three 16-key stages (99,840 B), so each block keeps two stages
 * in flight; QK and PV run on mma.m16n8k16 (f16 in, F32 accumulate):
 *
 *   S[16 x 16]   = Q[16 x 128] K^T       A = Q (6 of 16 rows), B = K by ldmatrix
 *   O^T[128 x 8] += V^T[128 x 16] P^T    A = V by ldmatrix.trans + byte permute,
 *                                        B = P straight from S's accumulators
 *
 * E4M3 codes and the BF16-valued queries are exact in f16; P (the softmax
 * weights times the V scale) rounds to f16, finer than the reference's BF16
 * attention. Each thread's four consecutive codes feed k slots 2c, 2c+1,
 * 2c+8, 2c+9 of a k-step, the same permutation for Q and K. The per-key
 * cp.async and lane-per-key loads of the SIMT kernel reached 130-180 GB/s:
 * their 16-byte requests reread every 32-byte sector and filled the LSU
 * queue; this path streams the rows at about 230 GB/s. */
enum { BEAM_ATTN8_THREADS = 256, BEAM_ATTN8_TILE = 16, BEAM_ATTN8_STAGES = 3,
       BEAM_ATTN8_ROW = 2 * BEAM_ATTN_KV + 4 * DS4_BEAM_KV_HEADS,
       BEAM_ATTN8_SMEM = BEAM_ATTN8_STAGES * BEAM_ATTN8_TILE * BEAM_ATTN8_ROW };
static_assert(BEAM_ATTN8_THREADS / 32 == DS4_BEAM_KV_HEADS && BEAM_ATTN8_ROW % 16 == 0, "warp per KV head, bulk rows");

static __device__ __forceinline__ uint32_t beam_smem(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
static __device__ __forceinline__ uint32_t beam_e4m3x2_h2(uint32_t two) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((unsigned short)(two & 0xffffu)));
    return h;
}
static __device__ __forceinline__ uint32_t beam_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&h);
}
static __device__ __forceinline__ void beam_mma_f16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(BEAM_ATTN8_THREADS, 1) beam_attn_fp8_kernel(float *heads, float *partials,
        const float *q, const uint8_t *kv, uint32_t kv_cap, uint32_t first, uint32_t last, uint32_t chunk,
        uint32_t splits, float scale) {
#if __CUDA_ARCH__ >= 900
    constexpr uint32_t G = BEAM_ATTN_GROUP, D = BEAM_GPU_HEAD, T = BEAM_ATTN8_TILE, S = BEAM_ATTN8_STAGES;
    constexpr uint32_t ROW = BEAM_ATTN8_ROW, STAGE = T * ROW;
    extern __shared__ __align__(128) uint8_t rows_smem[];      // [S][T][ROW]
    __shared__ __align__(8) uint64_t bars[S];
    const uint32_t split = blockIdx.x, lane = threadIdx.x & 31u, kvh = threadIdx.x >> 5;
    const uint32_t g = lane >> 2, c = lane & 3u;
    const uint32_t c0 = first + split * chunk, c1 = min(last, c0 + chunk - 1u);
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < S; i++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(beam_smem(&bars[i])));
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    // Q as the A operand: row g = head g (rows 6-15 zero); k-step s holds
    // dims 16s + 4c .. +3 in slots 2c, 2c+1 (a0) and 2c+8, 2c+9 (a2).
    uint32_t qa[8][2];
    {
        const float *qh = q + ((uint64_t)kvh * G + (g < G ? g : 0u)) * D;
#pragma unroll
        for (uint32_t s = 0; s < 8; s++) {
            const float4 v = *reinterpret_cast<const float4 *>(qh + 16u * s + 4u * c);
            qa[s][0] = g < G ? beam_h2(v.x, v.y) : 0u;
            qa[s][1] = g < G ? beam_h2(v.z, v.w) : 0u;
        }
    }
    __syncthreads();
    // Tile t0 into stage st: thread j < T copies key j's row; keys past the
    // chunk repeat the first so every stage carries STAGE bytes.
    auto issue = [&](uint32_t t0, uint32_t st) {
        if (threadIdx.x < T) {
            const uint32_t n = min(T, c1 - t0 + 1u), key = threadIdx.x;
            uint32_t slot = t0 % kv_cap + (key < n ? key : 0u);
            if (slot >= kv_cap) { slot -= kv_cap; }
            if (key == 0) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                    :: "r"(beam_smem(&bars[st])), "r"(STAGE) : "memory");
            }
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :: "r"(beam_smem(rows_smem + (uint64_t)st * STAGE + key * ROW)), "l"(kv + (uint64_t)slot * ROW),
                   "r"(ROW), "r"(beam_smem(&bars[st])) : "memory");
        }
    };
    float m = -INFINITY, l = 0.0f, o[8][4];
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) { o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f; }
#pragma unroll
    for (uint32_t i = 0; i + 1 < S; i++) {
        if (c0 + i * T <= c1) { issue(c0 + i * T, i); }
    }
    const uint32_t koff = kvh * D, voff = BEAM_ATTN_KV + kvh * D, soff = 2u * BEAM_ATTN_KV + 2u * kvh;
    uint32_t it = 0;
    for (uint32_t t0 = c0; t0 <= c1; t0 += T, it++) {
        const uint32_t st = it % S;
        // Every warp has left the stage the next copy refills.
        __syncthreads();
        if (t0 + (S - 1u) * T <= c1) { issue(t0 + (S - 1u) * T, (it + S - 1u) % S); }
        asm volatile("{\n .reg .pred p;\n BEAM_WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
            " @!p bra BEAM_WAIT_%=;\n}\n" :: "r"(beam_smem(&bars[st])), "r"((it / S) & 1u) : "memory");
        const uint8_t *rows = rows_smem + (uint64_t)st * STAGE;
        const uint32_t n = min(T, c1 - t0 + 1u);
        // Scores of two 8-key n-tiles: ldmatrix gives key nt*8 + g's codes
        // 4c .. 4c+3 of four k-steps per x4.
        float s[2][4];
#pragma unroll
        for (uint32_t nt = 0; nt < 2; nt++) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
            for (uint32_t half = 0; half < 2; half++) {
                uint32_t r[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                    : "r"(beam_smem(rows + (nt * 8u + (lane & 7u)) * ROW + koff + 64u * half + 16u * (lane >> 3))));
#pragma unroll
                for (uint32_t j = 0; j < 4; j++) {
                    const uint32_t a[4] = {qa[4u * half + j][0], 0u, qa[4u * half + j][1], 0u};
                    beam_mma_f16(s[nt], a, beam_e4m3x2_h2(r[j]), beam_e4m3x2_h2(r[j] >> 16));
                }
            }
        }
        // Head g's online softmax: this thread holds keys nt*8 + 2c + e.
        float sc[2][2], top = -INFINITY;
#pragma unroll
        for (uint32_t nt = 0; nt < 2; nt++) {
#pragma unroll
            for (uint32_t e = 0; e < 2; e++) {
                const uint32_t key = nt * 8u + 2u * c + e;
                const float ks = __half2float(*reinterpret_cast<const __half *>(rows + key * ROW + soff));
                sc[nt][e] = key < n ? s[nt][e] * scale * ks : -INFINITY;
                top = fmaxf(top, sc[nt][e]);
            }
        }
        top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
        top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
        const float mn = fmaxf(m, top), r = __expf(m - mn);
        m = mn;
        // P^T (B of the PV MMA): keys 2c, 2c+1 of n-tile 0 are b0, of n-tile 1 b1.
        uint32_t pb[2];
        float psum = 0.0f;
#pragma unroll
        for (uint32_t nt = 0; nt < 2; nt++) {
            float p[2];
#pragma unroll
            for (uint32_t e = 0; e < 2; e++) {
                const uint32_t key = nt * 8u + 2u * c + e;
                const float vs = __half2float(*reinterpret_cast<const __half *>(rows + key * ROW + soff + 16u));
                p[e] = key < n ? __expf(sc[nt][e] - mn) : 0.0f;
                psum += p[e];
                p[e] *= vs;
            }
            pb[nt] = beam_h2(p[0], p[1]);
        }
        l = l * r + psum;
        // O^T's columns are heads 2c, 2c+1: their rescale comes from rows g = 2c, 2c+1.
        const float r0 = __shfl_sync(0xffffffffu, r, 8u * c), r1 = __shfl_sync(0xffffffffu, r, 8u * c + 4u);
#pragma unroll
        for (uint32_t mt = 0; mt < 8; mt++) {
            o[mt][0] *= r0; o[mt][1] *= r1; o[mt][2] *= r0; o[mt][3] *= r1;
        }
        // m-tile mt: rows g / g + 8 are dims 16mt + 2g / 16mt + 2g + 1. The
        // transposed 8x8 b16 load gives keys 2c, 2c+1 of codes 16mt + 2g, +1;
        // byte permutes split them by dimension. One x4 serves two m-tiles.
#pragma unroll
        for (uint32_t mp = 0; mp < 8; mp += 2) {
            uint32_t t[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(t[0]), "=r"(t[1]), "=r"(t[2]), "=r"(t[3])
                : "r"(beam_smem(rows + (8u * ((lane >> 3) & 1u) + (lane & 7u)) * ROW + voff + 16u * (mp + (lane >> 4)))));
#pragma unroll
            for (uint32_t h = 0; h < 2; h++) {
                const uint32_t a[4] = {beam_e4m3x2_h2(__byte_perm(t[2u * h], 0, 0x0020)),
                                       beam_e4m3x2_h2(__byte_perm(t[2u * h], 0, 0x0031)),
                                       beam_e4m3x2_h2(__byte_perm(t[2u * h + 1u], 0, 0x0020)),
                                       beam_e4m3x2_h2(__byte_perm(t[2u * h + 1u], 0, 0x0031))};
                beam_mma_f16(o[mp + h], a, pb[0], pb[1]);
            }
        }
    }
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    const float m0 = __shfl_sync(0xffffffffu, m, 8u * c), m1 = __shfl_sync(0xffffffffu, m, 8u * c + 4u);
    const float l0 = __shfl_sync(0xffffffffu, l, 8u * c), l1 = __shfl_sync(0xffffffffu, l, 8u * c + 4u);
    if (c < 3u) {  // columns 6, 7 are padding heads
#pragma unroll
        for (uint32_t e = 0; e < 2; e++) {
            const uint32_t head = kvh * G + 2u * c + e;
            const float hm = e ? m1 : m0, hl = e ? l1 : l0;
            if (splits == 1u) {
                float *out = heads + (uint64_t)head * D;
#pragma unroll
                for (uint32_t mt = 0; mt < 8; mt++) {
                    out[16u * mt + 2u * g] = o[mt][e] / hl;
                    out[16u * mt + 2u * g + 1u] = o[mt][2u + e] / hl;
                }
            } else {
                float *part = partials + ((uint64_t)head * splits + split) * (D + 2u);
                if (g == 0) { part[0] = hm; part[1] = hl; }
#pragma unroll
                for (uint32_t mt = 0; mt < 8; mt++) {
                    part[2u + 16u * mt + 2u * g] = o[mt][e];
                    part[2u + 16u * mt + 2u * g + 1u] = o[mt][2u + e];
                }
            }
        }
    }
#else
    __trap();
#endif
}

// Splits of one head merged: thread = dimension.
__global__ void __launch_bounds__(BEAM_ATTN_THREADS) beam_attn_combine_kernel(float *heads, const float *partials,
        uint32_t splits) {
    constexpr uint32_t D = BEAM_GPU_HEAD;
    const uint32_t head = blockIdx.x, d = threadIdx.x;
    const float *part = partials + (uint64_t)head * splits * (D + 2u);
    float top = -INFINITY, sum = 0.0f, o = 0.0f;
    for (uint32_t s = 0; s < splits; s++) { top = fmaxf(top, part[(uint64_t)s * (D + 2u)]); }
    for (uint32_t s = 0; s < splits; s++) {
        const float *p = part + (uint64_t)s * (D + 2u);
        const float e = __expf(p[0] - top);
        sum += p[1] * e;
        o += p[2u + d] * e;
    }
    heads[(uint64_t)head * D + d] = o / sum;
}

extern "C" int ds4_gpu_beam_attn_decode(ds4_gpu_tensor *heads, const ds4_gpu_tensor *q, const ds4_gpu_tensor *kv,
        ds4_gpu_tensor *partials, uint32_t max_splits, uint32_t kv_cap, uint32_t pos, uint32_t window, int fp8) {
    static int tma = -1;  // compute capability 9+, dynamic shared memory opted in
    if (tma < 0) {
        int device = 0, major = 0;
        tma = cudaGetDevice(&device) == cudaSuccess &&
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess && major >= 9 &&
            cudaFuncSetAttribute(beam_attn_fp8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                BEAM_ATTN8_SMEM) == cudaSuccess;
    }
    const bool bulk = fp8 && tma;
    uint32_t first = window && pos + 1u > window ? pos + 1u - window : 0u;
    if (pos + 1u > kv_cap && first < pos + 1u - kv_cap) { first = pos + 1u - kv_cap; }
    const uint32_t keys = pos - first + 1u;
    // About BEAM_ATTN_SPLITS splits of whole tiles: one block per SM on the
    // bulk path, a wave per KV head otherwise (at least four tiles each).
    const uint32_t tile = bulk ? (uint32_t)BEAM_ATTN8_TILE : (uint32_t)BEAM_ATTN_TILE, least = bulk ? tile : 4u * tile;
    uint32_t chunk = (keys + BEAM_ATTN_SPLITS - 1u) / BEAM_ATTN_SPLITS;
    chunk = chunk < least ? least : (chunk + tile - 1u) / tile * tile;
    if ((keys + chunk - 1u) / chunk > max_splits) {
        chunk = ((keys + max_splits - 1u) / max_splits + tile - 1u) / tile * tile;
    }
    const uint32_t splits = (keys + chunk - 1u) / chunk;
    const uint64_t row = fp8 ? 2ull * BEAM_ATTN_KV + 4u * DS4_BEAM_KV_HEADS : 4ull * BEAM_ATTN_KV;
    if (!kv_cap || !gvq_floats(heads, (uint64_t)BEAM_GPU_HEADS * BEAM_GPU_HEAD) ||
        !gvq_floats(q, (uint64_t)BEAM_GPU_HEADS * BEAM_GPU_HEAD) || !kv || kv->bytes < (uint64_t)kv_cap * row ||
        (splits > 1u && !gvq_floats(partials, (uint64_t)BEAM_GPU_HEADS * splits * (BEAM_GPU_HEAD + 2u)))) {
        return 0;
    }
    float *part = partials ? (float *)partials->ptr : nullptr;
    const auto *k = (const uint8_t *)kv->ptr;
    const float scale = rsqrtf((float)BEAM_GPU_HEAD);
    const dim3 grid(splits, DS4_BEAM_KV_HEADS);
    const cudaStream_t stream = ds4_current_stream();
    if (bulk) {
        beam_attn_fp8_kernel<<<splits, BEAM_ATTN8_THREADS, BEAM_ATTN8_SMEM, stream>>>((float *)heads->ptr, part,
            (const float *)q->ptr, k, kv_cap, first, pos, chunk, splits, scale);
    } else if (fp8) {
        beam_attn_decode_kernel<true><<<grid, BEAM_ATTN_THREADS, 0, stream>>>((float *)heads->ptr, part,
            (const float *)q->ptr, k, kv_cap, first, pos, chunk, splits, scale);
    } else {
        beam_attn_decode_kernel<false><<<grid, BEAM_ATTN_THREADS, 0, stream>>>((float *)heads->ptr, part,
            (const float *)q->ptr, k, kv_cap, first, pos, chunk, splits, scale);
    }
    if (splits > 1u) {
        beam_attn_combine_kernel<<<BEAM_GPU_HEADS, BEAM_ATTN_THREADS, 0, stream>>>((float *)heads->ptr, part, splits);
    }
    return cuda_ok(cudaGetLastError(), "Beam decode attention");
}

// The head's BF16 logits (F32 MMQ sums rounded), capped as 30 tanh(x / 30)
// in F32; padded rows past the tokenizer can never be sampled.
__global__ void beam_logits_kernel(float *logits, uint32_t vocab, uint32_t tokens, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    if (i % vocab >= tokens) {
        logits[i] = -1e30f;
        return;
    }
    const float x = glmvq_round(logits[i]);
    logits[i] = glmvq_mul((float)tanh((double)__fdiv_rn(x, 30.0f)), 30.0f);
}

extern "C" int ds4_gpu_beam_logits(ds4_gpu_tensor *logits, uint32_t rows, uint32_t tokens) {
    const uint32_t vocab = DS4_BEAM_VOCAB;
    const uint64_t count = (uint64_t)rows * vocab;
    if (!rows || !tokens || tokens > vocab || !gvq_floats(logits, count)) { return 0; }
    beam_logits_kernel<<<(uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0,
        ds4_current_stream()>>>((float *)logits->ptr, vocab, tokens, count);
    return cuda_ok(cudaGetLastError(), "Beam logits");
}

__global__ void beam_embed_kernel(float *cur, const __nv_bfloat16 *rows, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { cur[i] = __bfloat162float(rows[i]); }
}

extern "C" int ds4_gpu_beam_embed(ds4_gpu_tensor *cur, const ds4_gpu_tensor *rows_bf16, uint32_t rows) {
    const uint64_t count = (uint64_t)rows * BEAM_GPU_HIDDEN;
    if (!rows || !gvq_floats(cur, count) || rows_bf16->bytes < count * sizeof(__nv_bfloat16)) { return 0; }
    beam_embed_kernel<<<(uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0,
        ds4_current_stream()>>>((float *)cur->ptr, (const __nv_bfloat16 *)rows_bf16->ptr, count);
    return cuda_ok(cudaGetLastError(), "Beam embed");
}

/* DFlash2 drafter kernels. Draft rows only choose proposals that the target
 * verifies, so they stay F32 without the source's BF16 rounding. */

// One stage of GroupedDynamicCausalConv: per channel, the base kernel plus
// the row's dynamic coefficient of the channel's 16-wide group, two taps.
__global__ void beam_df_conv_kernel(float *out, const float *x, const float *dyn, const float *base,
        uint32_t stage) {
    constexpr uint32_t groups = BEAM_GPU_HIDDEN / DS4_BEAM_DF_GROUP;
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x, l = blockIdx.y;
    if (c >= BEAM_GPU_HIDDEN) { return; }
    const float *d = dyn + (uint64_t)l * DS4_BEAM_DF_DYN + stage * 2u * groups;
    const float *b = base + (uint64_t)stage * 2u * BEAM_GPU_HIDDEN;
    const uint32_t g = c / DS4_BEAM_DF_GROUP;
    float sum = (b[c] + d[g]) * x[(uint64_t)l * BEAM_GPU_HIDDEN + c];
    if (l) { sum = fmaf(b[BEAM_GPU_HIDDEN + c] + d[groups + g], x[(uint64_t)(l - 1u) * BEAM_GPU_HIDDEN + c], sum); }
    out[(uint64_t)l * BEAM_GPU_HIDDEN + c] = sum;
}

extern "C" int ds4_gpu_beam_df_conv(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *dyn,
        const void *map, uint64_t size, uint64_t base, uint32_t stage, uint32_t rows) {
    const uint64_t kernel = 4u * BEAM_GPU_HIDDEN * sizeof(float), count = (uint64_t)rows * BEAM_GPU_HIDDEN;
    if (!map || base > size || kernel > size - base || !rows || rows > DS4_BEAM_DF_BLOCK || stage > 1u ||
        !gvq_floats(out, count) || !gvq_floats(x, count) || out->ptr == x->ptr ||
        !gvq_floats(dyn, (uint64_t)rows * DS4_BEAM_DF_DYN)) { return 0; }
    const float *b = (const float *)cuda_model_range_ptr(map, base, kernel, "Beam DFlash2 conv kernel");
    if (!b) { return 0; }
    beam_df_conv_kernel<<<dim3((BEAM_GPU_HIDDEN + 255u) / 256u, rows), 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)x->ptr, (const float *)dyn->ptr, b, stage);
    return cuda_ok(cudaGetLastError(), "Beam DFlash2 conv");
}

// One (KV head, block row) per CTA: the group's five query heads over the
// ring keys [lo, start + rows). Each warp takes every fourth key with an
// online softmax per head; shared memory merges the four warps.
__global__ void beam_df_attn_kernel(float *out, const float *q, const __half *kv, uint32_t first,
        uint32_t start, uint32_t rows) {
    constexpr uint32_t G = DS4_BEAM_DF_HEADS / DS4_BEAM_KV_HEADS, D = BEAM_GPU_HEAD, WARPS = D / 32u, L = D / 32u;
    constexpr uint32_t KV = DS4_BEAM_KV_HEADS * D;
    __shared__ float sm_m[WARPS][G], sm_s[WARPS][G], sm_acc[WARPS][G][D];
    const uint32_t h = blockIdx.x, r = blockIdx.y, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t pos = start + r, hi = start + rows;
    uint32_t lo = pos >= DS4_BEAM_DF_WINDOW - 1u ? pos - (DS4_BEAM_DF_WINDOW - 1u) : 0u;
    if (lo < first) { lo = first; }
    const float scale = rsqrtf((float)D);
    float qv[G][L], acc[G][L], m[G], s[G];
    for (uint32_t g = 0; g < G; g++) {
        const float *qp = q + ((uint64_t)r * DS4_BEAM_DF_HEADS + h * G + g) * D + lane * L;
        for (uint32_t j = 0; j < L; j++) { qv[g][j] = qp[j] * scale; acc[g][j] = 0.0f; }
        m[g] = -INFINITY;
        s[g] = 0.0f;
    }
    for (uint32_t p = lo + warp; p < hi; p += WARPS) {
        const __half *row = kv + (uint64_t)(p % DS4_BEAM_DF_CAP) * 2u * KV + h * D + lane * L;
        float kf[L], vf[L];
        for (uint32_t j = 0; j < L; j++) { kf[j] = __half2float(row[j]); vf[j] = __half2float(row[KV + j]); }
        for (uint32_t g = 0; g < G; g++) {
            float dot = 0.0f;
            for (uint32_t j = 0; j < L; j++) { dot = fmaf(qv[g][j], kf[j], dot); }
            for (uint32_t o = 16; o; o >>= 1) { dot += __shfl_xor_sync(0xffffffffu, dot, o); }
            const float top = fmaxf(m[g], dot), corr = __expf(m[g] - top), e = __expf(dot - top);
            s[g] = fmaf(s[g], corr, e);
            for (uint32_t j = 0; j < L; j++) { acc[g][j] = fmaf(acc[g][j], corr, e * vf[j]); }
            m[g] = top;
        }
    }
    for (uint32_t g = 0; g < G; g++) {
        if (!lane) { sm_m[warp][g] = m[g]; sm_s[warp][g] = s[g]; }
        for (uint32_t j = 0; j < L; j++) { sm_acc[warp][g][lane * L + j] = acc[g][j]; }
    }
    __syncthreads();
    const uint32_t d = threadIdx.x;
    for (uint32_t g = 0; g < G; g++) {
        float top = -INFINITY, sum = 0.0f, value = 0.0f;
        for (uint32_t w = 0; w < WARPS; w++) { top = fmaxf(top, sm_m[w][g]); }
        for (uint32_t w = 0; w < WARPS; w++) {
            const float e = sm_s[w][g] > 0.0f ? __expf(sm_m[w][g] - top) : 0.0f;
            sum = fmaf(sm_s[w][g], e, sum);
            value = fmaf(sm_acc[w][g][d], e, value);
        }
        out[((uint64_t)r * DS4_BEAM_DF_HEADS + h * G + g) * D + d] = value / sum;
    }
}

extern "C" int ds4_gpu_beam_df_attn(ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *kv,
        uint32_t first, uint32_t start, uint32_t rows) {
    const uint64_t width = (uint64_t)rows * DS4_BEAM_DF_HEADS * BEAM_GPU_HEAD;
    const uint64_t ring = (uint64_t)DS4_BEAM_DF_CAP * 2u * DS4_BEAM_KV_HEADS * BEAM_GPU_HEAD * sizeof(__half);
    if (!rows || rows > DS4_BEAM_DF_BLOCK || first > start || !gvq_floats(out, width) || !gvq_floats(q, width) ||
        !kv || kv->bytes < ring) { return 0; }
    beam_df_attn_kernel<<<dim3(DS4_BEAM_KV_HEADS, rows), BEAM_GPU_HEAD, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)q->ptr, (const __half *)kv->ptr, first, start, rows);
    return cuda_ok(cudaGetLastError(), "Beam DFlash2 attention");
}
