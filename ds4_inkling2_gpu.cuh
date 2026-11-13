/* Inkling-2-Small VQ device runtime; included once by ds4_cuda.cu after the
 * GVQ adapter (whose linear kernels, context and records it uses) and
 * Beam's warp-specialized 12-bit expert kernels.
 *
 * Routed experts run GLM's GVQ1 kernels in Inkling 2's form: 4608 -> 2304 ->
 * 4608 rows (two whole 2048-input blocks plus a two-step partial one; 2304
 * is one block plus one step), the softcapped SwiGLU 4 tanh(g / 4) sigmoid(g)
 * 25 tanh(u / 25) rounded once, and an F32 routed sum in ascending expert
 * order rounded once, as the source's F32 index_add_:
 *
 *   rows == 1     decode: device routing, persistent CTAs keep the book
 *   rows 2..4     verify: one pass over the rows' expert union
 *   rows >= 5     prefill: host grouping, Beam's warp-specialized FP8 MMA on
 *                 the E4M3 codewords (128-token items)
 *
 * The rest is Inkling 2's own arithmetic (tools/vq_calib/inkling2_ref.py):
 * block attention residuals, 4-tap causal convolutions (static or dynamic
 * taps), L2 clips, relative-position profiles and the attention of
 * cuda/inkling2_attn.cuh. */
#include "ds4_inkling2.h"
#include "cuda/inkling2_attn.cuh"

enum { INK2_EXPERTS = DS4_INK2_EXPERTS, INK2_USED = DS4_INK2_USED, INK2_HIDDEN = DS4_INK2_HIDDEN,
       INK2_FFN = DS4_INK2_FFN, INK2_TABLE = 3 * INK2_EXPERTS, INK2_LAYERS = DS4_INK2_LAYERS,
       INK2_VERIFY_ROWS = GLMVQ_VER_ROWS, INK2_ALIGN = 256, INK2_THREADS = 256,
       INK2_PER_THREAD = INK2_HIDDEN / INK2_THREADS, INK2_MAX_STACK = DS4_INK2_BLOCKS + 1,
       INK2_MAX_QUERIES = 3 };
static_assert((uint32_t)INK2_USED == (uint32_t)GLMVQ_DEC_USED, "GVQ decode geometry");
static_assert(INK2_HIDDEN % INK2_THREADS == 0, "a CTA covers a hidden row");
static_assert((uint32_t)DS4_INK2_SPLITS == (uint32_t)INK2A_SPLITS, "decode partial geometry");

using ink2_form = glmvq_moe_form<INK2_FFN, GLMVQ_CLAMP_SOFTCAP, GLMVQ_REDUCE_F32, INK2_HIDDEN>;

static uint64_t ink2_align(uint64_t bytes) { return (bytes + INK2_ALIGN - 1) / INK2_ALIGN * INK2_ALIGN; }

// Decode/verify scratch, then the device tables (one 3 x 384 row per layer).
struct ink2_decode_layout { uint64_t items, mid, partial, vitems, plan, vmid, vpartial, tables, bytes; };

static ink2_decode_layout ink2_decode_plan() {
    ink2_decode_layout l = {};
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += ink2_align(bytes); return here; };
    l.items = take(sizeof(glmvq_decode_item) * INK2_USED);
    l.mid = take((uint64_t)INK2_USED * INK2_FFN * sizeof(__nv_bfloat16));
    l.partial = take((uint64_t)INK2_USED * INK2_HIDDEN * sizeof(float));
    l.vitems = take(sizeof(glmvq_verify_item) * GLMVQ_VER_PAIRS);
    l.plan = take(sizeof(glmvq_verify_plan));
    l.vmid = take((uint64_t)GLMVQ_VER_PAIRS * INK2_FFN * sizeof(__nv_bfloat16));
    l.vpartial = take((uint64_t)GLMVQ_VER_PAIRS * INK2_HIDDEN * sizeof(float));
    l.tables = take((uint64_t)INK2_LAYERS * INK2_TABLE * sizeof(uint32_t));
    l.bytes = at;
    return l;
}

// Prefill: E4M3 input rows and their scales, gate rows (BF16), the up
// epilogue's E4M3 SwiGLU rows and 1x128 scales, down rows (BF16), the
// gate/up/down items (each expert's runs of BEAM_W_BT pairs), each pair's
// token row and scale, and the per-token order of its pairs.
struct ink2_mma_layout { uint64_t x8, xs, gate, mid8, ms, down, order, witems, token, pscale, bytes; };

static ink2_mma_layout ink2_mma_plan(uint32_t rows) {
    ink2_mma_layout l = {};
    const uint64_t pairs = (uint64_t)rows * INK2_USED;
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += ink2_align(bytes); return here; };
    l.x8 = take((uint64_t)rows * INK2_HIDDEN);
    l.xs = take((uint64_t)rows * sizeof(float));
    l.gate = take(pairs * INK2_FFN * sizeof(__nv_bfloat16));
    l.mid8 = take(pairs * INK2_FFN);
    l.ms = take(pairs * (INK2_FFN / GLMVQ_MMA8_K) * sizeof(float));
    l.down = take(pairs * INK2_HIDDEN * sizeof(__nv_bfloat16));
    l.order = take(pairs * sizeof(uint32_t));
    l.witems = take(3 * (INK2_EXPERTS + pairs / BEAM_W_BT) * sizeof(beam_w_item));
    l.token = take(pairs * sizeof(uint32_t));
    l.pscale = take(pairs * sizeof(float));
    l.bytes = at;
    return l;
}

struct ds4_ink2_moe {
    ds4_gvq_ctx *gvq = nullptr;     // borrowed: stream, routes, grouped, pairs
    ds4_gpu_tensor *decode = nullptr, *mma = nullptr;
    uint32_t rows = 0, decode_blocks = 0, sms = 0;
    bool gain_dot = true;
    std::vector<uint32_t> keys;     // table slot per layer, keyed by expert 0's gate
    std::vector<beam_w_item> witems;
    std::vector<uint32_t> order;
    std::vector<uint8_t> token_used;
};

static bool ink2_env_off(const char *name) {
    const char *value = getenv(name);
    return value && !strcmp(value, "0");
}

extern "C" uint64_t ds4_gpu_ink2_moe_bytes(uint32_t rows) {
    return ink2_decode_plan().bytes + (rows > INK2_VERIFY_ROWS ? ink2_mma_plan(rows).bytes : 0u);
}

extern "C" void ds4_gpu_ink2_moe_free(ds4_ink2_moe *moe) {
    if (!moe) { return; }
    ds4_gpu_tensor_free(moe->decode);
    ds4_gpu_tensor_free(moe->mma);
    delete moe;
}

extern "C" ds4_ink2_moe *ds4_gpu_ink2_moe_create(ds4_gvq_ctx *gvq, uint32_t rows) {
    if (!gvq || !rows) { return nullptr; }
    ds4_ink2_moe *moe = new (std::nothrow) ds4_ink2_moe;
    if (!moe) { return nullptr; }
    moe->gvq = gvq;
    moe->rows = rows;
    moe->gain_dot = !ink2_env_off("DS4_INKLING2_GAIN_DOT");
    int device = 0, sms = 0;
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess) {
        moe->sms = (uint32_t)sms;
    }
    uint32_t blocks = 0;
    cudaError_t err = glmvq_decode_prepare<12, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_DOT, ink2_form>(&blocks);
    if (err == cudaSuccess) {
        err = glmvq_decode_prepare<12, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_WEIGHT, ink2_form>(&blocks);
    }
    if (err != cudaSuccess || !moe->sms) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: Inkling-2 expert kernels unavailable (%s)\n", cudaGetErrorString(err));
        ds4_gpu_ink2_moe_free(moe);
        return nullptr;
    }
    moe->decode_blocks = blocks;
    moe->decode = ds4_gpu_tensor_alloc(ink2_decode_plan().bytes);
    moe->mma = rows > INK2_VERIFY_ROWS ? ds4_gpu_tensor_alloc(ink2_mma_plan(rows).bytes) : nullptr;
    if (!moe->decode || (rows > INK2_VERIFY_ROWS && !moe->mma)) {
        ds4_gpu_ink2_moe_free(moe);
        return nullptr;
    }
    moe->order.resize((uint64_t)rows * INK2_USED);
    moe->token_used.resize(rows);
    return moe;
}

// Shape, book and alignment the direct kernels read; every one of a layer's
// 1,152 matrices shares its 12-bit book (no plane). Rows are 864 (gate/up)
// or 432 (down) bytes, so only the region starts are aligned.
static bool ink2_matrix_fits(const glmvq_source &s, uint32_t projection, const void *book) {
    const uint32_t rows = projection == 2 ? INK2_HIDDEN : INK2_FFN;
    const uint32_t cols = projection == 2 ? INK2_FFN : INK2_HIDDEN;
    const glmvq_matrix &m = s.vq;
    return s.storage == GLMVQ_WEIGHT_VQ && s.rows == rows && s.cols == cols &&
        m.rows == rows && m.cols == cols && m.precision == GLMVQ_E4M3 &&
        m.main && m.gain && book && m.book == book && !m.plane &&
        (uintptr_t)m.main % 128 == 0 && (uintptr_t)book % 16 == 0 && (uintptr_t)m.gain % sizeof(__half) == 0;
}

// The layer's device table, validated and uploaded on first use.
static const uint32_t *ink2_table(ds4_ink2_moe *moe, const uint32_t *matrices) {
    const auto &sources = moe->gvq->model->source;
    const ink2_decode_layout l = ink2_decode_plan();
    const auto *tables = (const uint32_t *)((const uint8_t *)moe->decode->ptr + l.tables);
    for (size_t i = 0; i < moe->keys.size(); i++) {
        if (moe->keys[i] == matrices[0]) { return tables + i * INK2_TABLE; }
    }
    if (moe->keys.size() == INK2_LAYERS) { return nullptr; }
    const void *book = sources[matrices[0]].vq.book;
    for (uint32_t i = 0; i < INK2_TABLE; i++) {
        if (matrices[i] >= sources.size() || !ink2_matrix_fits(sources[matrices[i]], i / INK2_EXPERTS, book)) {
            fprintf(stderr, "ds4: Inkling-2 expert matrix %u misses the GVQ1 layout\n", i);
            return nullptr;
        }
    }
    const size_t slot = moe->keys.size();
    if (!ds4_gpu_tensor_write(moe->decode, l.tables + slot * INK2_TABLE * sizeof(uint32_t),
            matrices, INK2_TABLE * sizeof(uint32_t))) { return nullptr; }
    moe->keys.push_back(matrices[0]);
    return tables + slot * INK2_TABLE;
}

// N=1 routing on the GPU: thread t places expert selected[t] at its rank, so
// items run in ascending expert order with no host sync.
__global__ void ink2_decode_route(glmvq_decode_item *items, const int32_t *selected,
        const uint32_t *table, const glmvq_source *sources) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x;
    const uint32_t e = min((uint32_t)selected[t], (uint32_t)INK2_EXPERTS - 1u);
    uint32_t rank = 0;
    for (uint32_t j = 0; j < INK2_USED; j++) { rank += (uint32_t)selected[j] < (uint32_t)selected[t]; }
    items[rank] = {sources[table[e]].vq, sources[table[INK2_EXPERTS + e]].vq,
        sources[table[2 * INK2_EXPERTS + e]].vq, t};
}

// Union of 2-4 rows' experts in ascending expert order (gvq_verify_route);
// down groups hold at most `capacity` pairs' mid rows.
__global__ void ink2_verify_route(glmvq_verify_item *items, glmvq_verify_plan *plan,
        const int32_t *selected, uint32_t rows, const uint32_t *table, const glmvq_source *sources,
        uint32_t capacity) {
    __shared__ uint32_t expert[GLMVQ_VER_PAIRS], lead[GLMVQ_VER_PAIRS], width[GLMVQ_VER_PAIRS];
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x, pairs = rows * INK2_USED;
    const bool live = t < pairs;
    if (live) { expert[t] = min((uint32_t)selected[t], (uint32_t)INK2_EXPERTS - 1u); }
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
            items[rank].up = sources[table[INK2_EXPERTS + e]].vq;
            items[rank].down = sources[table[2 * INK2_EXPERTS + e]].vq;
            items[rank].count = same;
            width[rank] = same;
        }
        items[rank].pair[before] = t;
        const uint32_t row = t / INK2_USED;
        for (uint32_t k = 0; k < INK2_USED; k++) { below += expert[row * INK2_USED + k] < e; }
        plan->order[row * INK2_USED + below] = t % INK2_USED;
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

static cudaError_t ink2_decode_run(ds4_ink2_moe *moe, float *out, const float *input, const int32_t *selected,
        const float *weights, const uint32_t *table, const void *book) {
    const ink2_decode_layout l = ink2_decode_plan();
    uint8_t *base = (uint8_t *)moe->decode->ptr;
    auto *items = (glmvq_decode_item *)(base + l.items);
    auto *mid = (__nv_bfloat16 *)(base + l.mid);
    auto *partial = (float *)(base + l.partial);
    const cudaStream_t stream = ds4_current_stream();
    const glmvq_launch launch = moe->gvq->launch;
    glmvq_launch_kernel(launch, ink2_decode_route, 1, INK2_USED, 0, stream, items, selected, table,
        (const glmvq_source *)moe->gvq->model->device->ptr);
    return moe->gain_dot
        ? glmvq_decode_launch<12, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_DOT, ink2_form>(items, book, input, mid,
            partial, out, weights, INFINITY, moe->decode_blocks, stream, nullptr, nullptr, nullptr, launch)
        : glmvq_decode_launch<12, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_WEIGHT, ink2_form>(items, book, input,
            mid, partial, out, weights, INFINITY, moe->decode_blocks, stream, nullptr, nullptr, nullptr, launch);
}

// Rows above the form's verify width run as consecutive slices (each row's
// arithmetic is the one-row kernel's, so slicing changes no value).
static cudaError_t ink2_verify_run(ds4_ink2_moe *moe, float *out, const float *input, const int32_t *selected,
        const float *weights, const uint32_t *table, const void *book, uint32_t rows) {
    const ink2_decode_layout l = ink2_decode_plan();
    uint8_t *base = (uint8_t *)moe->decode->ptr;
    auto *items = (glmvq_verify_item *)(base + l.vitems);
    auto *plan = (glmvq_verify_plan *)(base + l.plan);
    auto *mid = (__nv_bfloat16 *)(base + l.vmid);
    auto *partial = (float *)(base + l.vpartial);
    const cudaStream_t stream = ds4_current_stream();
    const glmvq_launch launch = moe->gvq->launch;
    constexpr uint32_t slice = ink2_form::VER_ROWS;
    cudaError_t err = cudaSuccess;
    for (uint32_t r = 0; err == cudaSuccess && r < rows; r += slice) {
        const uint32_t n = min(slice, rows - r);
        const uint64_t at = (uint64_t)r * INK2_USED;
        if (n == 1) {
            err = ink2_decode_run(moe, out + (uint64_t)r * INK2_HIDDEN, input + (uint64_t)r * INK2_HIDDEN,
                selected + at, weights + at, table, book);
            continue;
        }
        glmvq_launch_kernel(launch, ink2_verify_route, 1, GLMVQ_VER_PAIRS, 0, stream, items, plan, selected + at, n,
            table, (const glmvq_source *)moe->gvq->model->device->ptr, ink2_form::VER_GROUP);
        err = moe->gain_dot
            ? glmvq_verify_launch<12, 32, GLMVQ_GAIN_DOT, ink2_form>(items, plan, book,
                input + (uint64_t)r * INK2_HIDDEN, n, mid, partial, out + (uint64_t)r * INK2_HIDDEN,
                weights + at, INFINITY, moe->decode_blocks, stream, nullptr, launch)
            : glmvq_verify_launch<12, 32, GLMVQ_GAIN_WEIGHT, ink2_form>(items, plan, book,
                input + (uint64_t)r * INK2_HIDDEN, n, mid, partial, out + (uint64_t)r * INK2_HIDDEN,
                weights + at, INFINITY, moe->decode_blocks, stream, nullptr, launch);
    }
    return err;
}

// Host grouping by expert (ascending), as the GLM prefill path builds it.
static bool ink2_group(ds4_ink2_moe *moe, const ds4_gpu_tensor *selected, uint32_t rows, uint32_t *offsets) {
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * INK2_USED;
    if (pairs > ctx->routes.size() ||
        !ds4_gpu_tensor_read(selected, 0, ctx->routes.data(), pairs * sizeof(int32_t))) { return false; }
    std::vector<uint32_t> counts(INK2_EXPERTS, 0), cursor(INK2_EXPERTS);
    for (uint64_t i = 0; i < pairs; i++) {
        const int32_t expert = ctx->routes[i];
        if (expert < 0 || expert >= INK2_EXPERTS) { return false; }
        for (uint64_t j = i - i % INK2_USED; j < i; j++) {
            if (ctx->routes[j] == expert) { return false; }
        }
        counts[expert]++;
    }
    offsets[0] = 0;
    for (uint32_t e = 0; e < INK2_EXPERTS; e++) {
        offsets[e + 1] = offsets[e] + counts[e];
        cursor[e] = offsets[e];
    }
    for (uint64_t i = 0; i < pairs; i++) { ctx->grouped[cursor[ctx->routes[i]]++] = (uint32_t)i; }
    return ds4_gpu_tensor_write(ctx->pairs, 0, ctx->grouped.data(), pairs * sizeof(uint32_t)) != 0;
}

static int ink2_mma_run(ds4_ink2_moe *moe, float *out, const float *input, const float *weights,
        const uint32_t *matrices, const uint32_t *offsets, uint32_t rows) {
    ds4_gvq_ctx *ctx = moe->gvq;
    const uint64_t pairs = (uint64_t)rows * INK2_USED, count = (uint64_t)rows * INK2_HIDDEN;
    const ink2_mma_layout l = ink2_mma_plan(rows);
    uint8_t *base = (uint8_t *)moe->mma->ptr;
    auto *x8 = base + l.x8, *mid8 = base + l.mid8;
    auto *xs = (float *)(base + l.xs), *ms = (float *)(base + l.ms);
    auto *gate = (__nv_bfloat16 *)(base + l.gate), *down = (__nv_bfloat16 *)(base + l.down);
    auto *order = (uint32_t *)(base + l.order);
    auto *witems = (beam_w_item *)(base + l.witems);
    auto *token = (uint32_t *)(base + l.token);
    auto *pscale = (float *)(base + l.pscale);
    moe->witems.clear();
    for (uint32_t m = 0; m < 3; m++) {
        for (uint32_t e = 0; e < INK2_EXPERTS; e++) {
            const glmvq_matrix &vq = ctx->model->source[matrices[m * INK2_EXPERTS + e]].vq;
            for (uint32_t start = offsets[e]; start < offsets[e + 1]; start += BEAM_W_BT) {
                moe->witems.push_back({start, min((uint32_t)BEAM_W_BT, offsets[e + 1] - start), vq.gain, vq.main, 0});
            }
        }
    }
    std::fill(moe->token_used.begin(), moe->token_used.begin() + rows, 0);
    for (uint32_t i = 0; i < pairs; i++) {
        const uint32_t token_row = ctx->grouped[i] / INK2_USED;
        moe->order[token_row * INK2_USED + moe->token_used[token_row]++] = i;
    }
    const cudaStream_t stream = ds4_current_stream();
    if (!cuda_ok(cudaMemcpyAsync(witems, moe->witems.data(), moe->witems.size() * sizeof(beam_w_item),
            cudaMemcpyHostToDevice, stream), "Inkling-2 expert items") ||
        !cuda_ok(cudaMemcpyAsync(order, moe->order.data(), pairs * sizeof(*order),
            cudaMemcpyHostToDevice, stream), "Inkling-2 expert order")) { return 0; }
    const auto *book = (const uint8_t *)ctx->model->source[matrices[0]].vq.book;
    const auto *grouped = (const uint32_t *)ctx->pairs->ptr;
    const uint32_t nwide = (uint32_t)moe->witems.size() / 3;
    // Tokens quantize once (E4M3, one scale per row, ldmatrix k order) for
    // their eight experts; each pair gets its token row and scale.
    beam_w_quant<<<rows, GLMVQ_THREADS, 0, stream>>>(x8, xs, input, INK2_HIDDEN);
    beam_w_pairs<<<(uint32_t)((pairs + 255) / 256), 256, 0, stream>>>(token, pscale, xs, grouped, INK2_USED,
        (uint32_t)pairs);
    cudaError_t err = beam_w_launch<GLMVQ_MMA_GATE, GLMVQ_CLAMP_SOFTCAP>(witems, nwide, book, x8, token, pscale,
        gate, INK2_FFN, INK2_HIDDEN, nullptr, nullptr, INFINITY, moe->sms, stream);
    if (err == cudaSuccess) {
        err = beam_w_launch<GLMVQ_MMA_UP, GLMVQ_CLAMP_SOFTCAP>(witems + nwide, nwide, book, x8, token, pscale,
            gate, INK2_FFN, INK2_HIDDEN, mid8, ms, INFINITY, moe->sms, stream);
    }
    if (err == cudaSuccess) {
        err = beam_w_down_launch(witems + 2 * nwide, nwide, book, mid8, ms, down, INK2_HIDDEN, INK2_FFN,
            moe->sms, stream);
    }
    if (!cuda_ok(err, "Inkling-2 warp-specialized experts")) { return 0; }
    glmvq_mma_reduce<__nv_bfloat16, GLMVQ_REDUCE_F32><<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS,
        GLMVQ_THREADS, 0, stream>>>(out, down, weights, grouped, order, INK2_USED, INK2_HIDDEN, count);
    return cuda_ok(cudaGetLastError(), "Inkling-2 FP8 grouped MMA");
}

extern "C" int ds4_gpu_ink2_moe(ds4_ink2_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, const uint32_t *matrices, uint32_t rows) {
    if (!moe || !rows || rows > moe->rows || !matrices || !selected || !gvq_stream(moe->gvq)) { return 0; }
    const uint64_t pairs = (uint64_t)rows * INK2_USED, count = (uint64_t)rows * INK2_HIDDEN;
    if (pairs > selected->bytes / sizeof(int32_t) || !gvq_floats(weights, pairs) ||
        !gvq_floats(out, count) || !gvq_floats(input, count) ||
        matrices[0] >= moe->gvq->model->source.size() || ds4_capture_active()) { return 0; }
    const uint32_t *table = ink2_table(moe, matrices);
    if (!table) { return 0; }
    const void *book = moe->gvq->model->source[matrices[0]].vq.book;
    if (rows <= INK2_VERIFY_ROWS) {
        const auto *sel = (const int32_t *)selected->ptr;
        const auto *w = (const float *)weights->ptr;
        float *y = (float *)out->ptr;
        const float *x = (const float *)input->ptr;
        const cudaError_t err = rows == 1 ? ink2_decode_run(moe, y, x, sel, w, table, book)
                                          : ink2_verify_run(moe, y, x, sel, w, table, book, rows);
        return cuda_ok(err, "Inkling-2 routed decode");
    }
    std::vector<uint32_t> offsets(INK2_EXPERTS + 1);
    if (!moe->mma || !ink2_group(moe, selected, rows, offsets.data())) { return 0; }
    return ink2_mma_run(moe, (float *)out->ptr, (const float *)input->ptr, (const float *)weights->ptr,
        matrices, offsets.data(), rows);
}

/* Sigmoid router: one warp per row, lane l holds experts l, l + 32, ...;
 * s = sigmoid(logit) in F32 (torch's), choice c = s + bias; eight rounds of
 * warp argmax over c (ties to the lower expert); weights = the chosen s over
 * their sum + 1e-20, times 8. */
__global__ void ink2_route_kernel(int32_t *ids, float *weights, const float *logits, const float *bias,
        uint32_t rows) {
    constexpr uint32_t PER = INK2_EXPERTS / 32;
    const uint32_t lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) { return; }
    float s[PER], c[PER];
    #pragma unroll
    for (uint32_t j = 0; j < PER; j++) {
        const uint32_t e = lane + 32 * j;
        s[j] = __fdiv_rn(1.0f, glmvq_add(1.0f, expf(-logits[(uint64_t)row * INK2_EXPERTS + e])));
        c[j] = glmvq_add(s[j], bias[e]);
    }
    float sum = 0.0f, chosen_s[INK2_USED];
    uint32_t chosen[INK2_USED];
    for (uint32_t k = 0; k < INK2_USED; k++) {
        float best = -INFINITY;
        uint32_t index = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < PER; j++) {
            const uint32_t e = lane + 32 * j;
            if (c[j] > best || (c[j] == best && e < index)) { best = c[j]; index = e; }
        }
        for (uint32_t stride = 16; stride; stride >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, stride);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, index, stride);
            if (ob > best || (ob == best && oi < index)) { best = ob; index = oi; }
        }
        const uint32_t owner = index % 32, slot = index / 32;
        float picked = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < PER; j++) {
            if (j == slot) { picked = s[j]; }
        }
        picked = __shfl_sync(0xffffffffu, picked, owner);
        if (lane == owner) {
            #pragma unroll
            for (uint32_t j = 0; j < PER; j++) {
                if (j == slot) { c[j] = -INFINITY; }
            }
        }
        chosen[k] = index;
        chosen_s[k] = picked;
        sum = glmvq_add(sum, picked);
    }
    if (lane) { return; }
    const float denom = glmvq_add(sum, 1e-20f);
    for (uint32_t k = 0; k < INK2_USED; k++) {
        ids[(uint64_t)row * INK2_USED + k] = (int32_t)chosen[k];
        weights[(uint64_t)row * INK2_USED + k] = glmvq_mul(__fdiv_rn(chosen_s[k], denom), 8.0f);
    }
}

extern "C" int ds4_gpu_ink2_route(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, uint32_t bias, uint32_t rows) {
    const float *b = (const float *)gvq_weight(ctx, bias, GVQ_TYPE_F32, INK2_EXPERTS);
    if (!b || !rows || !gvq_floats(logits, (uint64_t)rows * INK2_EXPERTS) ||
        !gvq_floats(weights, (uint64_t)rows * INK2_USED) ||
        selected->bytes < (uint64_t)rows * INK2_USED * sizeof(int32_t)) { return 0; }
    constexpr uint32_t warps = 4;
    ink2_route_kernel<<<(rows + warps - 1) / warps, warps * 32, 0, ds4_current_stream()>>>(
        (int32_t *)selected->ptr, (float *)weights->ptr, (const float *)logits->ptr, b, rows);
    return cuda_ok(cudaGetLastError(), "Inkling-2 router");
}

// Block-wide sums of `n` per-thread values (n <= 40), in place.
template <uint32_t N>
__device__ __forceinline__ void ink2_block_sum(float (&v)[N], uint32_t n, float *scratch) {
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    for (uint32_t i = 0; i < n; i++) {
        float x = v[i];
        for (uint32_t o = 16; o; o >>= 1) { x += __shfl_xor_sync(0xffffffffu, x, o); }
        if (!lane) { scratch[warp * N + i] = x; }
    }
    __syncthreads();
    for (uint32_t i = 0; i < n; i++) {
        float x = 0.0f;
        for (uint32_t w = 0; w < warps; w++) { x += scratch[w * N + i]; }
        v[i] = x;
    }
    __syncthreads();
}

__device__ __forceinline__ float ink2_bf(uint16_t bits) { return __uint_as_float((uint32_t)bits << 16); }

/* One CTA per token. Pass 1: each stacked state's sum of squares and its dot
 * with every query; pass 2: the attention-weighted sum of the normalized
 * states; then the weighted RMS norm of each output. */
__global__ void __launch_bounds__(INK2_THREADS) ink2_attn_res_kernel(float *out0, float *out1, float *out2,
        const uint16_t *blocks, uint32_t nblocks, uint32_t cap, const float *partial, const float *query,
        uint32_t nq, const uint16_t *norm0, const uint16_t *norm1, const uint16_t *norm2) {
    __shared__ float scratch[(INK2_THREADS / 32) * (INK2_MAX_STACK * (INK2_MAX_QUERIES + 1))];
    constexpr uint32_t STATS = INK2_MAX_STACK * (INK2_MAX_QUERIES + 1);
    const uint32_t row = blockIdx.x, n = nblocks + 1;
    const float *x = partial + (uint64_t)row * INK2_HIDDEN;
    auto state = [&](uint32_t j, uint32_t i) {
        return j < nblocks ? ink2_bf(blocks[((uint64_t)j * cap + row) * INK2_HIDDEN + i]) : x[i];
    };
    float *outs[INK2_MAX_QUERIES] = {out0, out1, out2};
    const uint16_t *norms[INK2_MAX_QUERIES] = {norm0, norm1, norm2};
    float y[INK2_MAX_QUERIES][INK2_PER_THREAD];
    uint32_t outputs = nq;
    if (!query) {
        // No attention: the partial state itself is normalized.
        outputs = 1;
        #pragma unroll
        for (uint32_t k = 0; k < INK2_PER_THREAD; k++) { y[0][k] = x[threadIdx.x + k * INK2_THREADS]; }
    } else {
        float stats[STATS];
        for (uint32_t i = 0; i < STATS; i++) { stats[i] = 0.0f; }
        for (uint32_t j = 0; j < n; j++) {
            #pragma unroll 2
            for (uint32_t k = 0; k < INK2_PER_THREAD; k++) {
                const uint32_t i = threadIdx.x + k * INK2_THREADS;
                const float v = state(j, i);
                stats[j] = fmaf(v, v, stats[j]);
                for (uint32_t qi = 0; qi < nq; qi++) {
                    stats[INK2_MAX_STACK + qi * INK2_MAX_STACK + j] =
                        fmaf(query[(uint64_t)qi * INK2_HIDDEN + i], v,
                            stats[INK2_MAX_STACK + qi * INK2_MAX_STACK + j]);
                }
            }
        }
        ink2_block_sum<STATS>(stats, STATS, scratch);
        float inv[INK2_MAX_STACK], p[INK2_MAX_QUERIES][INK2_MAX_STACK];
        for (uint32_t j = 0; j < n; j++) {
            inv[j] = rsqrtf(__fdiv_rn(stats[j], (float)INK2_HIDDEN) + 1e-6f);
        }
        const float scale = rsqrtf((float)INK2_HIDDEN);
        for (uint32_t qi = 0; qi < nq; qi++) {
            float top = -INFINITY, total = 0.0f;
            for (uint32_t j = 0; j < n; j++) {
                p[qi][j] = stats[INK2_MAX_STACK + qi * INK2_MAX_STACK + j] * inv[j] * scale;
                top = fmaxf(top, p[qi][j]);
            }
            for (uint32_t j = 0; j < n; j++) {
                p[qi][j] = expf(p[qi][j] - top);
                total += p[qi][j];
            }
            for (uint32_t j = 0; j < n; j++) { p[qi][j] = __fdiv_rn(p[qi][j], total); }
        }
        for (uint32_t qi = 0; qi < nq; qi++) {
            #pragma unroll
            for (uint32_t k = 0; k < INK2_PER_THREAD; k++) { y[qi][k] = 0.0f; }
        }
        for (uint32_t j = 0; j < n; j++) {
            #pragma unroll 2
            for (uint32_t k = 0; k < INK2_PER_THREAD; k++) {
                const float v = state(j, threadIdx.x + k * INK2_THREADS) * inv[j];
                for (uint32_t qi = 0; qi < nq; qi++) { y[qi][k] = fmaf(p[qi][j], v, y[qi][k]); }
            }
        }
    }
    float sq[INK2_MAX_QUERIES] = {0.0f, 0.0f, 0.0f};
    for (uint32_t qi = 0; qi < outputs; qi++) {
        #pragma unroll
        for (uint32_t k = 0; k < INK2_PER_THREAD; k++) { sq[qi] = fmaf(y[qi][k], y[qi][k], sq[qi]); }
    }
    ink2_block_sum<INK2_MAX_QUERIES>(sq, outputs, scratch);
    for (uint32_t qi = 0; qi < outputs; qi++) {
        const float r = rsqrtf(__fdiv_rn(sq[qi], (float)INK2_HIDDEN) + 1e-6f);
        float *o = outs[qi] + (uint64_t)row * INK2_HIDDEN;
        #pragma unroll
        for (uint32_t k = 0; k < INK2_PER_THREAD; k++) {
            const uint32_t i = threadIdx.x + k * INK2_THREADS;
            o[i] = glmvq_round(glmvq_mul(ink2_bf(norms[qi][i]), glmvq_mul(y[qi][k], r)));
        }
    }
}

extern "C" int ds4_gpu_ink2_attn_res(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *out, const ds4_gpu_tensor *blocks,
        uint32_t nblocks, uint32_t cap, const ds4_gpu_tensor *partial, uint32_t query, uint32_t nq,
        const uint32_t *norms, uint32_t rows) {
    const uint64_t count = (uint64_t)rows * INK2_HIDDEN;
    const bool attend = query != DS4_GVQ_NO_LINK;
    const uint32_t outputs = attend ? nq : 1u;
    if (!rows || !partial || !gvq_floats(partial, count) || nblocks > DS4_INK2_BLOCKS || rows > cap ||
        outputs < 1 || outputs > INK2_MAX_QUERIES || (nblocks && (!blocks ||
            blocks->bytes < (uint64_t)nblocks * cap * INK2_HIDDEN * sizeof(uint16_t)))) { return 0; }
    const float *q = attend ? (const float *)gvq_weight(ctx, query, GVQ_TYPE_F32, (uint64_t)nq * INK2_HIDDEN)
                            : nullptr;
    const uint16_t *w[INK2_MAX_QUERIES] = {nullptr, nullptr, nullptr};
    float *o[INK2_MAX_QUERIES] = {nullptr, nullptr, nullptr};
    for (uint32_t i = 0; i < outputs; i++) {
        w[i] = (const uint16_t *)gvq_weight(ctx, norms[i], GVQ_TYPE_BF16, INK2_HIDDEN);
        if (!w[i] || !gvq_floats(out[i], count)) { return 0; }
        o[i] = (float *)out[i]->ptr;
    }
    if (attend && !q) { return 0; }
    ink2_attn_res_kernel<<<rows, INK2_THREADS, 0, ds4_current_stream()>>>(o[0], o[1], o[2],
        nblocks ? (const uint16_t *)blocks->ptr : nullptr, nblocks, cap, (const float *)partial->ptr, q,
        attend ? nq : 0u, w[0], w[1], w[2]);
    return cuda_ok(cudaGetLastError(), "Inkling-2 attention residual");
}

__global__ void __launch_bounds__(INK2_THREADS) ink2_embed_kernel(float *out, const uint16_t *rows_bf16,
        const uint16_t *norm) {
    __shared__ float scratch[INK2_THREADS / 32];
    const uint32_t row = blockIdx.x;
    float v[INK2_PER_THREAD], sq[1] = {0.0f};
    #pragma unroll
    for (uint32_t k = 0; k < INK2_PER_THREAD; k++) {
        v[k] = ink2_bf(rows_bf16[(uint64_t)row * INK2_HIDDEN + threadIdx.x + k * INK2_THREADS]);
        sq[0] = fmaf(v[k], v[k], sq[0]);
    }
    ink2_block_sum<1>(sq, 1, scratch);
    const float r = rsqrtf(__fdiv_rn(sq[0], (float)INK2_HIDDEN) + 1e-6f);
    #pragma unroll
    for (uint32_t k = 0; k < INK2_PER_THREAD; k++) {
        const uint32_t i = threadIdx.x + k * INK2_THREADS;
        out[(uint64_t)row * INK2_HIDDEN + i] = glmvq_round(glmvq_mul(ink2_bf(norm[i]), glmvq_mul(v[k], r)));
    }
}

extern "C" int ds4_gpu_ink2_embed(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *rows_bf16,
        uint32_t norm, uint32_t rows) {
    const uint16_t *w = (const uint16_t *)gvq_weight(ctx, norm, GVQ_TYPE_BF16, INK2_HIDDEN);
    if (!w || !rows || !gvq_floats(out, (uint64_t)rows * INK2_HIDDEN) ||
        rows_bf16->bytes < (uint64_t)rows * INK2_HIDDEN * sizeof(uint16_t)) { return 0; }
    ink2_embed_kernel<<<rows, INK2_THREADS, 0, ds4_current_stream()>>>((float *)out->ptr,
        (const uint16_t *)rows_bf16->ptr, w);
    return cuda_ok(cudaGetLastError(), "Inkling-2 embedding norm");
}

__global__ void ink2_push_kernel(uint16_t *dst, const float *x, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { dst[i] = __bfloat16_as_ushort(__float2bfloat16_rn(x[i])); }
}

extern "C" int ds4_gpu_ink2_push(ds4_gpu_tensor *blocks, uint32_t slot, uint32_t cap, const ds4_gpu_tensor *x,
        uint32_t rows) {
    const uint64_t count = (uint64_t)rows * INK2_HIDDEN;
    if (!rows || rows > cap || slot >= DS4_INK2_BLOCKS || !gvq_floats(x, count) ||
        blocks->bytes < (uint64_t)(slot + 1) * cap * INK2_HIDDEN * sizeof(uint16_t)) { return 0; }
    ink2_push_kernel<<<(uint32_t)((count + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (uint16_t *)blocks->ptr + (uint64_t)slot * cap * INK2_HIDDEN, (const float *)x->ptr, count);
    return cuda_ok(cudaGetLastError(), "Inkling-2 block state");
}

// History rows (oldest first) of a convolution stream: input row r - j for
// r >= j, else history row 3 - (j - r).
__device__ __forceinline__ float ink2_tap(const float *x, const float *hist, int64_t r, uint32_t j,
        uint32_t width, uint32_t c) {
    const int64_t at = r - (int64_t)j;
    return glmvq_round(at >= 0 ? x[at * width + c] : hist[(at + DS4_INK2_HISTORY) * width + c]);
}

// The next history: the last three rows of [history; x].
__global__ void ink2_history_kernel(float *hist, const float *x, uint32_t width, uint32_t rows) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= width) { return; }
    float v[DS4_INK2_HISTORY];
    #pragma unroll
    for (uint32_t h = 0; h < DS4_INK2_HISTORY; h++) {
        const int64_t at = (int64_t)rows - DS4_INK2_HISTORY + h;
        v[h] = glmvq_round(at >= 0 ? x[at * width + c] : hist[(at + DS4_INK2_HISTORY) * width + c]);
    }
    #pragma unroll
    for (uint32_t h = 0; h < DS4_INK2_HISTORY; h++) { hist[h * width + c] = v[h]; }
}

/* K/V tails: CTA (row, k|v, KV head), a thread per channel. BF16 rows go to
 * the cache row; NVFP4 vectors (full layers) to their page through the
 * MiniMax VQ release's encoder (mm_fp4_encode, one warp per vector). */
__global__ void __launch_bounds__(INK2A_DIM) ink2_kv_kernel(uint16_t *cache, uint32_t format, uint32_t cap,
        uint8_t *tier_window, uint32_t tier_resident, uint32_t tier_block0, uint32_t window,
        const float *kin, const float *vin, const float *hist_k, const float *hist_v, const uint16_t *tap_k,
        const uint16_t *tap_v, const float *dyn_k, const float *dyn_v, uint32_t pos) {
    __shared__ float scratch[INK2A_DIM / 32];
    const uint32_t row = blockIdx.x, value = blockIdx.y, head = blockIdx.z;
    const uint32_t c = head * INK2A_DIM + threadIdx.x;
    const float *x = value ? vin : kin, *hist = value ? hist_v : hist_k;
    const uint16_t *taps = value ? tap_v : tap_k;
    const float *dyn = value ? dyn_v : dyn_k;
    const float now = ink2_tap(x, hist, row, 0, INK2A_KV, c);
    float y = 0.0f, out;
    if (!dyn) {
        // Conv1d weight [c][0][i] multiplies x[t - 3 + i].
        #pragma unroll
        for (uint32_t i = 0; i < DS4_INK2_TAPS; i++) {
            y = fmaf(ink2_bf(taps[c * DS4_INK2_TAPS + i]), ink2_tap(x, hist, row, DS4_INK2_HISTORY - i, INK2A_KV, c), y);
        }
        out = glmvq_round(__fdiv_rn(glmvq_add(y, now), 2.2360680103302f));
    } else {
        // Tap j is the backward distance: static_w[j][c] + the row's dynamic
        // tap of this KV head.
        #pragma unroll
        for (uint32_t j = 0; j < DS4_INK2_TAPS; j++) {
            const float w = glmvq_add(ink2_bf(taps[j * INK2A_KV + c]),
                glmvq_round(dyn[(uint64_t)row * INK2A_KV_HEADS * DS4_INK2_TAPS + head * DS4_INK2_TAPS + j]));
            y = fmaf(ink2_tap(x, hist, row, j, INK2A_KV, c), w, y);
        }
        out = glmvq_round(__fdiv_rn(glmvq_add(y, now), 3.0f));
    }
    float sq[1] = {glmvq_mul(out, out)};
    ink2_block_sum<1>(sq, 1, scratch);
    float stored;
    if (!value) {
        const float n = sqrtf(sq[0]);
        stored = glmvq_round(glmvq_mul(out, fminf(__fdiv_rn(22.627416997969522f, glmvq_add(n, 1e-6f)), 1.0f)));
    } else {
        stored = glmvq_round(glmvq_mul(out, rsqrtf(glmvq_add(__fdiv_rn(sq[0], (float)INK2A_DIM), 1e-6f))));
    }
    const uint32_t p = pos + row;
    if (format == DS4_INK2_KV_NVFP4) {
        __shared__ float vec[INK2A_DIM];
        vec[threadIdx.x] = stored;
        __syncthreads();
        if (threadIdx.x >= 32) { return; }
        // Blocks past the resident ones are written into the pinned window.
        const bool cold = tier_window && p / INK2A_KB >= tier_resident;
        uint8_t *half = (uint8_t *)(cold ? ink2a_page(tier_window, p, head, tier_block0)
                                         : ink2a_page((const uint8_t *)cache, p, head)) +
            value * INK2A_KB * INK2A_FP4_ROW;
        const uint32_t slot = p % INK2A_KB;
        mm_fp4_encode(vec, half + slot * INK2A_FP4, half + INK2A_KB * INK2A_FP4 + slot * 8);
        return;
    }
    uint16_t *dst = cache + (uint64_t)(window ? p % cap : p) * INK2A_ROW + (value ? INK2A_KV : 0u) + c;
    *dst = __bfloat16_as_ushort(__float2bfloat16_rn(stored));
}

// Bytes a K/V store of `cap` positions needs in `format`.
static uint64_t ink2_kv_bytes(uint32_t format, uint32_t cap) {
    return format == DS4_INK2_KV_NVFP4 ? (uint64_t)((cap + INK2A_KB - 1) / INK2A_KB) * INK2A_BLOCK
                                        : (uint64_t)cap * INK2A_ROW * sizeof(uint16_t);
}

extern "C" int ds4_gpu_ink2_kv(ds4_gvq_ctx *ctx, ds4_gpu_tensor *cache, uint32_t cap, uint32_t format,
        const ds4_gpu_tensor *kin, const ds4_gpu_tensor *vin, ds4_gpu_tensor *hist_k, ds4_gpu_tensor *hist_v,
        const uint32_t *taps, const ds4_gpu_tensor *dyn_k, const ds4_gpu_tensor *dyn_v, uint32_t rows,
        uint32_t pos, const ds4_ink2_tier *tier) {
    const uint64_t count = (uint64_t)rows * INK2A_KV;
    const bool full = dyn_k != nullptr;
    if ((format != DS4_INK2_KV_BF16 && (format != DS4_INK2_KV_NVFP4 || !dyn_k)) || !rows ||
        !gvq_floats(kin, count) || !gvq_floats(vin, count) ||
        !gvq_floats(hist_k, (uint64_t)DS4_INK2_HISTORY * INK2A_KV) ||
        !gvq_floats(hist_v, (uint64_t)DS4_INK2_HISTORY * INK2A_KV) || (full && (!dyn_v ||
            !gvq_floats(dyn_k, (uint64_t)rows * INK2A_KV_HEADS * DS4_INK2_TAPS) ||
            !gvq_floats(dyn_v, (uint64_t)rows * INK2A_KV_HEADS * DS4_INK2_TAPS)))) { return 0; }
    const uint64_t tap_count = (uint64_t)INK2A_KV * DS4_INK2_TAPS;
    const uint16_t *tk = (const uint16_t *)gvq_weight(ctx, taps[0], GVQ_TYPE_BF16, tap_count);
    const uint16_t *tv = (const uint16_t *)gvq_weight(ctx, taps[1], GVQ_TYPE_BF16, tap_count);
    // Sliding layers keep a ring that holds the window behind a whole chunk.
    const uint32_t window = full ? 0u : DS4_INK2_WINDOW;
    const uint64_t last = (uint64_t)pos + rows;
    // Under the tier the store holds the resident blocks and the window the
    // frontier blocks of this forward.
    const bool tiered = tier && tier->window;
    const uint32_t first_block = pos / INK2A_KB, last_block = (uint32_t)((last - 1) / INK2A_KB);
    if (!tk || !tv || (window ? cap < rows + window - 1 : last > cap) ||
        (tiered ? format != DS4_INK2_KV_NVFP4 ||
            cache->bytes < (uint64_t)tier->resident * INK2A_BLOCK ||
            (last_block >= tier->resident && (max(first_block, tier->resident) < tier->window_block0 ||
                last_block >= tier->window_block0 + tier->window_blocks ||
                tier->window->bytes < (uint64_t)tier->window_blocks * INK2A_BLOCK))
                : cache->bytes < ink2_kv_bytes(format, cap))) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    ink2_kv_kernel<<<dim3(rows, 2, INK2A_KV_HEADS), INK2A_DIM, 0, stream>>>((uint16_t *)cache->ptr, format, cap,
        tiered ? (uint8_t *)tier->window->ptr : nullptr, tiered ? tier->resident : 0u,
        tiered ? tier->window_block0 : 0u, window,
        (const float *)kin->ptr, (const float *)vin->ptr, (const float *)hist_k->ptr, (const float *)hist_v->ptr,
        tk, tv, full ? (const float *)dyn_k->ptr : nullptr, full ? (const float *)dyn_v->ptr : nullptr, pos);
    // The ordered update cannot overwrite a history another CTA still reads.
    ink2_history_kernel<<<INK2A_KV / 128, 128, 0, stream>>>((float *)hist_k->ptr, (const float *)kin->ptr,
        INK2A_KV, rows);
    ink2_history_kernel<<<INK2A_KV / 128, 128, 0, stream>>>((float *)hist_v->ptr, (const float *)vin->ptr,
        INK2A_KV, rows);
    return cuda_ok(cudaGetLastError(), "Inkling-2 K/V");
}

/* q and r tails, then the relative profiles: CTA (row, head), 128 threads. */
__global__ void __launch_bounds__(INK2A_DIM) ink2_qr_kernel(float *q, float *r, uint16_t *rel,
        const uint16_t *q_scale, const uint16_t *r_scale, const uint16_t *proj, uint32_t pos, int global) {
    __shared__ float scratch[INK2A_DIM / 32];
    __shared__ float rr[DS4_INK2_REL];
    const uint32_t row = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    float tau = 1.0f;
    if (global) {
        const float n = __fdiv_rn((float)(pos + row + 1u), 128000.0f);
        tau = glmvq_add(1.0f, glmvq_mul(0.1f, logf(fmaxf(n, 1.0f))));
    }
    // q: scaled L2 clip at 22.627.
    float *qv = q + ((uint64_t)row * INK2A_HEADS + h) * INK2A_DIM;
    const float xq = glmvq_mul(glmvq_round(qv[t]), ink2_bf(q_scale[h * INK2A_DIM + t]));
    float sq[1] = {glmvq_mul(xq, xq)};
    ink2_block_sum<1>(sq, 1, scratch);
    float outq = glmvq_round(glmvq_mul(xq, fminf(__fdiv_rn(22.627416997969522f, glmvq_add(sqrtf(sq[0]), 1e-6f)),
        1.0f)));
    if (global) { outq = glmvq_round(glmvq_mul(outq, tau)); }
    qv[t] = outq;
    // r: scaled L2 clip at 8 over 16 values (the first warp).
    float *rv = r + ((uint64_t)row * INK2A_HEADS + h) * DS4_INK2_REL;
    if (t < 32) {
        const float xr = t < DS4_INK2_REL ? glmvq_mul(glmvq_round(rv[t]), ink2_bf(r_scale[h * DS4_INK2_REL + t]))
                                          : 0.0f;
        float s = glmvq_mul(xr, xr);
        for (uint32_t o = 16; o; o >>= 1) { s += __shfl_xor_sync(0xffffffffu, s, o); }
        float outr = glmvq_round(glmvq_mul(xr, fminf(__fdiv_rn(8.0f, glmvq_add(sqrtf(s), 1e-6f)), 1.0f)));
        if (global) { outr = glmvq_round(glmvq_mul(outr, tau)); }
        if (t < DS4_INK2_REL) {
            rv[t] = outr;
            rr[t] = outr;
        }
    }
    __syncthreads();
    // rel[d] = BF16(r . proj[:, d]) (a BF16 matmul with F32 accumulation).
    uint16_t *dst = rel + ((uint64_t)row * INK2A_HEADS + h) * INK2A_EXTENT;
    for (uint32_t d = t; d < INK2A_EXTENT; d += INK2A_DIM) {
        float v = 0.0f;
        #pragma unroll
        for (uint32_t k = 0; k < DS4_INK2_REL; k++) { v = fmaf(rr[k], ink2_bf(proj[k * INK2A_EXTENT + d]), v); }
        dst[d] = __bfloat16_as_ushort(__float2bfloat16_rn(v));
    }
}

extern "C" int ds4_gpu_ink2_qr(ds4_gvq_ctx *ctx, ds4_gpu_tensor *q, ds4_gpu_tensor *r, ds4_gpu_tensor *rel,
        uint32_t q_scale, uint32_t r_scale, uint32_t proj, uint32_t rows, uint32_t pos, int global) {
    const uint16_t *qs = (const uint16_t *)gvq_weight(ctx, q_scale, GVQ_TYPE_BF16, (uint64_t)INK2A_HEADS * INK2A_DIM);
    const uint16_t *rs = (const uint16_t *)gvq_weight(ctx, r_scale, GVQ_TYPE_BF16,
        (uint64_t)INK2A_HEADS * DS4_INK2_REL);
    const uint16_t *pj = (const uint16_t *)gvq_weight(ctx, proj, GVQ_TYPE_BF16,
        (uint64_t)DS4_INK2_REL * INK2A_EXTENT);
    if (!qs || !rs || !pj || !rows || !gvq_floats(q, (uint64_t)rows * INK2A_HEADS * INK2A_DIM) ||
        !gvq_floats(r, (uint64_t)rows * INK2A_HEADS * DS4_INK2_REL) ||
        rel->bytes < (uint64_t)rows * INK2A_HEADS * INK2A_EXTENT * sizeof(uint16_t)) { return 0; }
    ink2_qr_kernel<<<dim3(rows, INK2A_HEADS), INK2A_DIM, 0, ds4_current_stream()>>>((float *)q->ptr,
        (float *)r->ptr, (uint16_t *)rel->ptr, qs, rs, pj, pos, global);
    return cuda_ok(cudaGetLastError(), "Inkling-2 q/r tails");
}

extern "C" uint64_t ds4_gpu_ink2_attn_bytes(void) {
    return (uint64_t)INK2A_SPLIT_ROWS * INK2A_HEADS * INK2A_SPLITS * INK2A_PARTIAL * sizeof(float);
}

extern "C" int ds4_gpu_ink2_attn(ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *rel,
        const ds4_gpu_tensor *sinks, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *cache, uint32_t cap,
        uint32_t window, uint32_t format, ds4_gpu_tensor *partials, uint32_t rows, uint32_t pos) {
    const uint64_t heads = (uint64_t)rows * INK2A_HEADS;
    const bool fp4 = format == DS4_INK2_KV_NVFP4;
    if ((format != DS4_INK2_KV_BF16 && (!fp4 || window)) || !rows || !gvq_floats(out, heads * INK2A_DIM) ||
        !gvq_floats(q, heads * INK2A_DIM) || !gvq_floats(gate, heads * INK2A_DIM) || !gvq_floats(sinks, heads) ||
        rel->bytes < heads * INK2A_EXTENT * sizeof(uint16_t) || cache->bytes < ink2_kv_bytes(format, cap) ||
        (window ? cap < rows + window - 1 : (uint64_t)pos + rows > cap)) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    if (rows <= INK2A_SPLIT_ROWS) {
        if (!partials || partials->bytes < ds4_gpu_ink2_attn_bytes()) { return 0; }
        const uint32_t last = pos + rows - 1;
        const uint32_t keys = window && last >= window ? window : last + 1;
        const uint32_t splits = ink2a_splits(keys);
        const dim3 grid(splits, INK2A_KV_HEADS, rows);
        if (fp4) {
            ink2a_decode<INK2A_FMT_NVFP4><<<grid, INK2A_DTHREADS, 0, stream>>>((float *)partials->ptr,
                (const float *)q->ptr, (const uint16_t *)rel->ptr, cache->ptr, cap, window, pos, splits);
        } else {
            ink2a_decode<INK2A_FMT_BF16><<<grid, INK2A_DTHREADS, 0, stream>>>((float *)partials->ptr,
                (const float *)q->ptr, (const uint16_t *)rel->ptr, cache->ptr, cap, window, pos, splits);
        }
        ink2a_merge<<<(uint32_t)heads, INK2A_DIM, 0, stream>>>((float *)out->ptr, (const float *)partials->ptr,
            (const float *)sinks->ptr, (const float *)gate->ptr, splits);
        return cuda_ok(cudaGetLastError(), "Inkling-2 decode attention");
    }
    static cudaError_t ready = cudaFuncSetAttribute(ink2a_prefill<INK2A_FMT_BF16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM);
    static cudaError_t ready4 = cudaFuncSetAttribute(ink2a_prefill<INK2A_FMT_NVFP4>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM_FP4);
    if (!cuda_ok(fp4 ? ready4 : ready, "Inkling-2 attention shared memory")) { return 0; }
    const dim3 grid((rows + INK2A_TQ - 1) / INK2A_TQ, INK2A_HEADS / INK2A_CTA_HEADS);
    if (fp4) {
        ink2a_prefill<INK2A_FMT_NVFP4><<<grid, INK2A_THREADS, INK2A_SMEM_FP4, stream>>>((float *)out->ptr,
            (const float *)q->ptr, (const uint16_t *)rel->ptr, (const float *)sinks->ptr, (const float *)gate->ptr,
            cache->ptr, cap, window, rows, pos);
    } else {
        ink2a_prefill<INK2A_FMT_BF16><<<grid, INK2A_THREADS, INK2A_SMEM, stream>>>((float *)out->ptr,
            (const float *)q->ptr, (const uint16_t *)rel->ptr, (const float *)sinks->ptr, (const float *)gate->ptr,
            cache->ptr, cap, window, rows, pos);
    }
    return cuda_ok(cudaGetLastError(), "Inkling-2 prefill attention");
}

extern "C" uint64_t ds4_gpu_ink2_state_bytes(uint32_t rows) {
    return (uint64_t)rows * INK2A_HEADS * INK2A_STATE * sizeof(float);
}

extern "C" int ds4_gpu_ink2_attn_seg(ds4_gpu_tensor *out, ds4_gpu_tensor *state, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *rel, const ds4_gpu_tensor *sinks, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *store, uint32_t seg_lo, uint32_t seg_hi, uint32_t block0, ds4_gpu_tensor *partials,
        uint32_t rows, uint32_t pos, int mode) {
    const uint64_t heads = (uint64_t)rows * INK2A_HEADS;
    const uint32_t blocks = (uint32_t)(((uint64_t)min(seg_hi, pos + rows) + INK2A_KB - 1) / INK2A_KB);
    if (!rows || mode < DS4_INK2_SEG_FIRST || mode > DS4_INK2_SEG_LAST || seg_lo >= seg_hi ||
        seg_lo % INK2A_KB || seg_lo / INK2A_KB < block0 || !state || state->bytes < ds4_gpu_ink2_state_bytes(rows) ||
        !gvq_floats(out, heads * INK2A_DIM) || !gvq_floats(q, heads * INK2A_DIM) ||
        !gvq_floats(gate, heads * INK2A_DIM) || !gvq_floats(sinks, heads) ||
        rel->bytes < heads * INK2A_EXTENT * sizeof(uint16_t) ||
        store->bytes < (uint64_t)(blocks - block0) * INK2A_BLOCK) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    if (rows <= INK2A_SPLIT_ROWS) {
        if (!partials || partials->bytes < ds4_gpu_ink2_attn_bytes()) { return 0; }
        const uint32_t keys = min(seg_hi, pos + rows) - seg_lo;
        const uint32_t splits = ink2a_splits(keys);
        ink2a_decode<INK2A_FMT_NVFP4><<<dim3(splits, INK2A_KV_HEADS, rows), INK2A_DTHREADS, 0, stream>>>(
            (float *)partials->ptr, (const float *)q->ptr, (const uint16_t *)rel->ptr, store->ptr, 0, 0, pos,
            splits, seg_lo, seg_hi, block0);
        ink2a_fold<<<(uint32_t)heads, INK2A_DIM, 0, stream>>>((float *)state->ptr, (const float *)partials->ptr,
            (const float *)sinks->ptr, splits, mode == DS4_INK2_SEG_FIRST);
        if (mode == DS4_INK2_SEG_LAST) {
            ink2a_finish<<<(uint32_t)heads, INK2A_DIM, 0, stream>>>((float *)out->ptr, (const float *)state->ptr,
                (const float *)gate->ptr);
        }
        return cuda_ok(cudaGetLastError(), "Inkling-2 segment decode attention");
    }
    const dim3 grid((rows + INK2A_TQ - 1) / INK2A_TQ, INK2A_HEADS / INK2A_CTA_HEADS);
    auto launch = [&](auto kernel) {
        cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM_FP4);
        if (err == cudaSuccess) {
            kernel<<<grid, INK2A_THREADS, INK2A_SMEM_FP4, stream>>>((float *)out->ptr, (const float *)q->ptr,
                (const uint16_t *)rel->ptr, (const float *)sinks->ptr, (const float *)gate->ptr, store->ptr, 0u, 0u,
                rows, pos, (float *)state->ptr, seg_lo, seg_hi, block0);
            err = cudaGetLastError();
        }
        return cuda_ok(err, "Inkling-2 segment prefill attention");
    };
    return mode == DS4_INK2_SEG_FIRST ? launch(ink2a_prefill<INK2A_FMT_NVFP4, INK2A_FIRST>)
        : mode == DS4_INK2_SEG_NEXT ? launch(ink2a_prefill<INK2A_FMT_NVFP4, INK2A_NEXT>)
        : launch(ink2a_prefill<INK2A_FMT_NVFP4, INK2A_LAST>);
}

static cudaEvent_t g_ink2_mark[2];

extern "C" int ds4_gpu_ink2_mark(uint32_t slot) {
    if (slot > 1) { return 0; }
    if (!g_ink2_mark[slot] && !cuda_ok(cudaEventCreateWithFlags(&g_ink2_mark[slot],
            cudaEventDisableTiming | cudaEventBlockingSync), "Inkling-2 mark")) { return 0; }
    return cuda_ok(cudaEventRecord(g_ink2_mark[slot], ds4_current_stream()), "Inkling-2 mark");
}

extern "C" int ds4_gpu_ink2_mark_wait(uint32_t slot) {
    return slot <= 1 && (!g_ink2_mark[slot] || cuda_ok(cudaEventSynchronize(g_ink2_mark[slot]), "Inkling-2 mark wait"));
}

/* Residual convolution of 4,608-wide rows: y = BF16((conv4(x) + x) /
 * sqrt(5)), out = y + residual (F32) or BF16(y + residual). */
__global__ void ink2_conv_kernel(float *out, const float *x, const float *hist, const uint16_t *taps,
        const float *residual, uint32_t rows, int round_out) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * INK2_HIDDEN) { return; }
    const uint32_t c = i % INK2_HIDDEN;
    const int64_t r = i / INK2_HIDDEN;
    float y = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < DS4_INK2_TAPS; k++) {
        y = fmaf(ink2_bf(taps[c * DS4_INK2_TAPS + k]), ink2_tap(x, hist, r, DS4_INK2_HISTORY - k, INK2_HIDDEN, c), y);
    }
    const float now = ink2_tap(x, hist, r, 0, INK2_HIDDEN, c);
    float v = glmvq_round(__fdiv_rn(glmvq_add(y, now), 2.2360680103302f));
    if (residual) { v = glmvq_add(residual[i], v); }
    out[i] = round_out ? glmvq_round(v) : v;
}

extern "C" int ds4_gpu_ink2_conv(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        ds4_gpu_tensor *hist, uint32_t taps, const ds4_gpu_tensor *residual, uint32_t rows, int round_out) {
    const uint64_t count = (uint64_t)rows * INK2_HIDDEN;
    const uint16_t *w = (const uint16_t *)gvq_weight(ctx, taps, GVQ_TYPE_BF16, (uint64_t)INK2_HIDDEN * DS4_INK2_TAPS);
    if (!w || !rows || !gvq_floats(out, count) || !gvq_floats(x, count) ||
        !gvq_floats(hist, (uint64_t)DS4_INK2_HISTORY * INK2_HIDDEN) || (residual && !gvq_floats(residual, count)) ||
        out->ptr == x->ptr) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    ink2_conv_kernel<<<(uint32_t)((count + 255) / 256), 256, 0, stream>>>((float *)out->ptr, (const float *)x->ptr,
        (const float *)hist->ptr, w, residual ? (const float *)residual->ptr : nullptr, rows, round_out);
    ink2_history_kernel<<<(INK2_HIDDEN + 255) / 256, 256, 0, stream>>>((float *)hist->ptr, (const float *)x->ptr,
        INK2_HIDDEN, rows);
    return cuda_ok(cudaGetLastError(), "Inkling-2 residual convolution");
}

__global__ void ink2_logits_kernel(float *logits, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint32_t v = i % DS4_INK2_VOCAB;
    logits[i] = v < DS4_INK2_TOKENS ? glmvq_mul(tanhf(__fdiv_rn(logits[i], 30.0f)), 30.0f) : -INFINITY;
}

extern "C" int ds4_gpu_ink2_logits(ds4_gpu_tensor *logits, uint32_t rows) {
    const uint64_t count = (uint64_t)rows * DS4_INK2_VOCAB;
    if (!rows || !gvq_floats(logits, count)) { return 0; }
    ink2_logits_kernel<<<(uint32_t)((count + 255) / 256), 256, 0, ds4_current_stream()>>>((float *)logits->ptr,
        count);
    return cuda_ok(cudaGetLastError(), "Inkling-2 logits");
}

#include "cuda/inkling2_draft.cuh"
