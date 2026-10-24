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
 *                keep the book; one row at a time, so every width runs the
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
#include "cuda/minimax_mma8w.cuh"

enum { MM_EXPERTS = DS4_MINIMAX_EXPERTS, MM_USED = DS4_MINIMAX_USED, MM_HIDDEN = DS4_MINIMAX_HIDDEN,
       MM_FFN = DS4_MINIMAX_FFN, MM_TABLE = 3 * MM_EXPERTS, MM_LAYERS = DS4_MINIMAX_MOE_LAYERS,
       MM_WORDS = 8192, MM_WARPS = 32, MM_DECODE_ROWS = DS4_MINIMAX_SPLIT_ROWS, MM_MMA_ROWS = 256, MM_MMA_TOKENS = 64,
       MM_ALIGN = 256, MM_HEAD = DS4_MINIMAX_HEAD, MM_HEADS = DS4_MINIMAX_HEADS,
       MM_KV_HEADS = DS4_MINIMAX_KV_HEADS, MM_GROUP = MM_HEADS / MM_KV_HEADS, MM_ROT = DS4_MINIMAX_ROT,
       MM_BLOCK = DS4_MINIMAX_BLOCK, MM_TOPK = DS4_MINIMAX_TOPK, MM_FP4_DATA = DS4_MINIMAX_FP4_DATA,
       MM_FP4_ROW = DS4_MINIMAX_FP4_ROW, MM_PAGE = DS4_MINIMAX_KV_PAGE, MM_SPLITS = DS4_MINIMAX_SPLITS };
static_assert(MM_TOPK * MM_BLOCK == 2048, "MSA attends 16 blocks of 128 tokens");
static const float MM_LIMIT = 7.0f;
static const float MM_EPS = 1e-6f;

static uint64_t mm_align(uint64_t bytes) { return (bytes + MM_ALIGN - 1) / MM_ALIGN * MM_ALIGN; }

/* ------------------------------------------------------------------ experts */

// One 3072-wide down row: a whole 2048-input block, then a 1024-input tail
// of four k-steps. R13+8 rows carry 64 residual words for the block and 32
// for the tail (one byte per vector, row-major), which GLM's whole-block
// helper does not cover. Same lanes, FMA order and gain step as
// glmvq_decode_dot.
template <bool RES>
__device__ __forceinline__ float mm_down_dot(const glmvq_matrix &m, uint32_t row, const __nv_bfloat16 *x,
        const uint2 *book, const uint2 *book2) {
    const uint32_t lane = threadIdx.x & 31, offset = lane * 12 / 32, shift = lane * 12 % 32;
    const float gain = __half2float(m.gain[row]);
    const auto *main = (const uint32_t *)m.main + (uint64_t)row * MM_FFN * 3 / 64;
    const auto *plane = (const uint32_t *)m.plane + (uint64_t)row * MM_FFN / 256;
    const auto *res = RES ? (const uint32_t *)m.res + (uint64_t)row * MM_FFN / 32 : nullptr;
    const glmvq_decode_block head = {main[lane], main[lane + 32], main[lane + 64], lane < 8 ? plane[lane] : 0u,
        RES ? res[lane] : 0u, RES ? res[lane + 32] : 0u};
    const glmvq_decode_block tail = {main[96 + lane], lane < 16 ? main[128 + lane] : 0u, 0u,
        lane < 4 ? plane[8 + lane] : 0u, RES ? res[64 + lane] : 0u, 0u};
    float sum = 0.0f;
    #pragma unroll
    for (uint32_t part = 0; part < 2; part++) {
        const glmvq_decode_block &cur = part ? tail : head;
        const uint32_t steps = part ? 4u : 8u;
        #pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
            if (k >= steps) { break; }
            const uint32_t first = k * 12;
            const uint32_t lr = (first & 31u) > 20 ? (first + 31 - lane) / 32 : first / 32;
            const uint32_t hr = ((first + 1) & 31u) > 20 ? (first + 32 - lane) / 32 : (first + 1) / 32;
            const uint32_t lo = __shfl_sync(0xffffffffu, glmvq_decode_word(cur, lr), first + offset);
            const uint32_t hi = __shfl_sync(0xffffffffu, glmvq_decode_word(cur, hr), first + offset + 1);
            uint32_t index = __funnelshift_r(lo, hi, shift) & 4095u;
            index |= ((__shfl_sync(0xffffffffu, cur.plane, k) >> lane) & 1u) << 12;
            const uint4 values = *(const uint4 *)(x + part * 2048 + k * 256 + lane * GLMVQ_VECTOR);
            if constexpr (RES) {
                sum = glmvq_decode_word8_res<GLMVQ_GAIN_DOT>(book[index], book2[glmvq_decode_res(cur, k)], values,
                    gain, sum);
            } else {
                sum = glmvq_decode_word8<GLMVQ_GAIN_DOT>(book[index], values, gain, sum);
            }
        }
    }
    for (uint32_t stride = 16; stride; stride >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, stride); }
    return glmvq_round(glmvq_mul(sum, gain));
}

// N=1 routing on the GPU: thread t places expert selected[t] at its rank, so
// items run in ascending expert order with no host sync.
__global__ void mm_decode_route(glmvq_decode_item *items, const int32_t *selected, const uint32_t *table,
        const glmvq_source *sources) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t t = threadIdx.x;
    const uint32_t e = min((uint32_t)selected[t], (uint32_t)MM_EXPERTS - 1u);
    uint32_t rank = 0;
    for (uint32_t j = 0; j < MM_USED; j++) { rank += (uint32_t)selected[j] < (uint32_t)selected[t]; }
    items[rank] = {sources[table[e]].vq, sources[table[MM_EXPERTS + e]].vq, sources[table[2 * MM_EXPERTS + e]].vq, t};
}

// gate/up of the four experts: one warp per (expert, row); the layer book
// (and residual book) stays in shared memory of a persistent CTA.
template <bool RES>
__global__ void __launch_bounds__(MM_WARPS * 32, 1) mm_decode_gu(const glmvq_decode_item *items,
        const uint4 *book, const uint4 *book2, const float *input, __nv_bfloat16 *mid) {
    extern __shared__ __align__(16) uint8_t mm_stage[];
    constexpr uint32_t books = glmvq_decode_books<13, RES>();
    auto *x = (__nv_bfloat16 *)(mm_stage + books);
    const auto *res_book = (const uint2 *)(mm_stage + MM_WORDS * GLMVQ_VECTOR);
    glmvq_pdl_open();
    glmvq_decode_book<13, RES>((uint4 *)mm_stage, book, book2);
    glmvq_pdl_wait();
    for (uint32_t i = threadIdx.x; i < MM_HIDDEN; i += blockDim.x) { x[i] = __float2bfloat16_rn(input[i]); }
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31, warp = blockIdx.x * MM_WARPS + threadIdx.x / 32;
    for (uint32_t task = warp; task < MM_USED * MM_FFN; task += gridDim.x * MM_WARPS) {
        const uint32_t e = task / MM_FFN, row = task % MM_FFN;
        const glmvq_decode_item item = items[e];
        const auto *gm = (const uint32_t *)item.gate.main + (uint64_t)row * MM_HIDDEN * 3 / 64;
        const auto *gp = (const uint32_t *)item.gate.plane + (uint64_t)row * MM_HIDDEN / 256;
        const auto *um = (const uint32_t *)item.up.main + (uint64_t)row * MM_HIDDEN * 3 / 64;
        const auto *up = (const uint32_t *)item.up.plane + (uint64_t)row * MM_HIDDEN / 256;
        const auto *gr = RES ? (const uint32_t *)item.gate.res + (uint64_t)row * MM_HIDDEN / 32 : nullptr;
        const auto *ur = RES ? (const uint32_t *)item.up.res + (uint64_t)row * MM_HIDDEN / 32 : nullptr;
        auto carry = glmvq_decode_load<13, 8, RES>(gm, gp, gr);
        const float g = glmvq_decode_dot<MM_HIDDEN, 13, GLMVQ_GAIN_DOT, RES>(item.gate, row, x,
            (const uint2 *)mm_stage, &carry, um, up, res_book, ur);
        const float u = glmvq_decode_dot<MM_HIDDEN, 13, GLMVQ_GAIN_DOT, RES>(item.up, row, x,
            (const uint2 *)mm_stage, &carry, nullptr, nullptr, res_book, nullptr);
        if (!lane) { mid[task] = __float2bfloat16_rn(glmvq_swiglu_value<GLMVQ_CLAMP_OAI>(g, u, MM_LIMIT)); }
    }
}

// down of the four experts over their staged mid rows (4 x 3072 BF16 fit
// beside the 64 KiB book: 90,112 / 92,160 of 101,376 B).
template <bool RES>
__global__ void __launch_bounds__(MM_WARPS * 32, 1) mm_decode_down(const glmvq_decode_item *items,
        const uint4 *book, const uint4 *book2, const __nv_bfloat16 *mid, float *partial) {
    extern __shared__ __align__(16) uint8_t mm_stage[];
    constexpr uint32_t books = glmvq_decode_books<13, RES>();
    auto *x = (__nv_bfloat16 *)(mm_stage + books);
    const auto *res_book = (const uint2 *)(mm_stage + MM_WORDS * GLMVQ_VECTOR);
    glmvq_pdl_open();
    glmvq_decode_book<13, RES>((uint4 *)mm_stage, book, book2);
    glmvq_pdl_wait();
    for (uint32_t i = threadIdx.x; i < MM_USED * MM_FFN / 8; i += blockDim.x) {
        ((uint4 *)x)[i] = ((const uint4 *)mid)[i];
    }
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31, warp = blockIdx.x * MM_WARPS + threadIdx.x / 32;
    for (uint32_t task = warp; task < MM_USED * MM_HIDDEN; task += gridDim.x * MM_WARPS) {
        const uint32_t e = task / MM_HIDDEN, row = task % MM_HIDDEN;
        const float value = mm_down_dot<RES>(items[e].down, row, x + e * MM_FFN, (const uint2 *)mm_stage, res_book);
        if (!lane) { partial[task] = value; }
    }
}

// The F32 sum of weight * BF16 expert row in ascending expert order, plus
// the shared expert's row, rounded once.
__global__ void mm_decode_reduce(float *out, const float *partial, const float *weights,
        const glmvq_decode_item *items, const float *shared) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= MM_HIDDEN) { return; }
    float sum = 0.0f;
    for (uint32_t e = 0; e < MM_USED; e++) {
        sum = glmvq_add(sum, glmvq_mul(partial[e * MM_HIDDEN + col], weights[items[e].pair]));
    }
    out[col] = glmvq_round(glmvq_add(sum, glmvq_round(shared[col])));
}

// Prefill: pair rows of `down` (grouped order) per token in ascending expert
// order (`order`), as mm_decode_reduce.
__global__ void mm_mma_reduce(float *out, const __nv_bfloat16 *down, const float *weights, const uint32_t *pairs,
        const uint32_t *order, const float *shared, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint64_t token = i / MM_HIDDEN, col = i % MM_HIDDEN;
    float sum = 0.0f;
    for (uint32_t k = 0; k < MM_USED; k++) {
        const uint32_t row = order[token * MM_USED + k];
        sum = glmvq_add(sum, glmvq_mul(__bfloat162float(down[(uint64_t)row * MM_HIDDEN + col]), weights[pairs[row]]));
    }
    out[i] = glmvq_round(glmvq_add(sum, glmvq_round(shared[i])));
}

// Decode scratch, then the device tables (one 3 x 128 row per layer).
struct mm_decode_layout { uint64_t items, mid, partial, tables, bytes; };

static mm_decode_layout mm_decode_plan() {
    mm_decode_layout l = {};
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t here = at; at += mm_align(bytes); return here; };
    l.items = take(sizeof(glmvq_decode_item) * MM_USED);
    l.mid = take((uint64_t)MM_USED * MM_FFN * sizeof(__nv_bfloat16));
    l.partial = take((uint64_t)MM_USED * MM_HIDDEN * sizeof(float));
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
        const float *weights, const float *shared, const uint32_t *table, const void *book, const void *res_book) {
    const mm_decode_layout l = mm_decode_plan();
    uint8_t *base = (uint8_t *)moe->decode->ptr;
    auto *items = (glmvq_decode_item *)(base + l.items);
    auto *mid = (__nv_bfloat16 *)(base + l.mid);
    auto *partial = (float *)(base + l.partial);
    const cudaStream_t stream = ds4_current_stream();
    const glmvq_launch launch = moe->gvq->launch;
    constexpr uint32_t books = glmvq_decode_books<13, RES>();
    glmvq_launch_kernel(launch, mm_decode_route, 1, MM_USED, 0, stream, items, selected, table,
        (const glmvq_source *)moe->gvq->model->device->ptr);
    glmvq_launch_kernel(launch, mm_decode_gu<RES>, moe->decode_blocks, MM_WARPS * 32,
        books + MM_HIDDEN * sizeof(__nv_bfloat16), stream, (const glmvq_decode_item *)items, (const uint4 *)book,
        (const uint4 *)res_book, input, mid);
    glmvq_launch_kernel(launch, mm_decode_down<RES>, moe->decode_blocks, MM_WARPS * 32,
        books + MM_USED * MM_FFN * sizeof(__nv_bfloat16), stream, (const glmvq_decode_item *)items,
        (const uint4 *)book, (const uint4 *)res_book, (const __nv_bfloat16 *)mid, partial);
    glmvq_launch_kernel(launch, mm_decode_reduce, MM_HIDDEN / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream, out,
        (const float *)partial, weights, (const glmvq_decode_item *)items, shared);
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
static int mm_mmaw_run(ds4_minimax_moe *moe, float *out, const float *input, const float *weights,
        const float *shared, const uint32_t *matrices, const uint32_t *offsets, uint32_t rows) {
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
    cudaError_t err = mw_launch<GLMVQ_MMA_GATE, RES>(witems, n, book, res_book, x8, xs, token, gate, nullptr,
        MM_FFN, MM_HIDDEN, MM_LIMIT, nullptr, nullptr, sms, stream);
    if (err == cudaSuccess) {
        err = mw_launch<GLMVQ_MMA_UP, RES>(witems + n, n, book, res_book, x8, xs, token, gate, nullptr,
            MM_FFN, MM_HIDDEN, MM_LIMIT, mid8, ms, sms, stream);
    }
    if (err == cudaSuccess) {
        err = mw_launch<GLMVQ_MMA_DOWN, RES>(witems + 2 * n, n, book, res_book, mid8, ms, token, nullptr, down,
            MM_HIDDEN, MM_FFN, MM_LIMIT, nullptr, nullptr, sms, stream);
    }
    if (!cuda_ok(err, "MiniMax warp-specialized experts")) { return 0; }
    mm_mma_reduce<<<(uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0, stream>>>(out,
        down, weights, grouped, order, shared, count);
    return cuda_ok(cudaGetLastError(), "MiniMax FP8 warp-specialized MMA");
}

template <bool RES>
static int mm_mma_run(ds4_minimax_moe *moe, float *out, const float *input, const float *weights,
        const float *shared, const uint32_t *matrices, const uint32_t *offsets, uint32_t rows) {
    if (moe->mmaw) { return mm_mmaw_run<RES>(moe, out, input, weights, shared, matrices, offsets, rows); }
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
    mm_mma_reduce<<<(uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0, stream>>>(out,
        down, weights, grouped, order, shared, count);
    return cuda_ok(cudaGetLastError(), "MiniMax FP8 grouped MMA");
}

extern "C" int ds4_gpu_minimax_moe(ds4_minimax_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *shared,
        const uint32_t *matrices, uint32_t rows) {
    if (!moe || !rows || rows > moe->rows || !matrices || !selected || !gvq_stream(moe->gvq)) { return 0; }
    const uint64_t pairs = (uint64_t)rows * MM_USED, count = (uint64_t)rows * MM_HIDDEN;
    if (pairs > selected->bytes / sizeof(int32_t) || !gvq_floats(weights, pairs) || !gvq_floats(out, count) ||
        !gvq_floats(input, count) || !gvq_floats(shared, count) || ds4_capture_active()) { return 0; }
    const uint32_t *table = mm_table(moe, matrices);
    if (!table) { return 0; }
    const glmvq_matrix &first = moe->gvq->model->source[matrices[0]].vq;
    const bool res = first.res_book != nullptr;
    if (rows <= MM_DECODE_ROWS) {
        // Every row takes the one-row kernels: verify rows equal decode rows.
        const auto *sel = (const int32_t *)selected->ptr;
        const auto *w = (const float *)weights->ptr, *sh = (const float *)shared->ptr;
        const auto *x = (const float *)input->ptr;
        auto *y = (float *)out->ptr;
        cudaError_t err = cudaSuccess;
        for (uint32_t r = 0; err == cudaSuccess && r < rows; r++) {
            const uint64_t at = (uint64_t)r * MM_USED, row = (uint64_t)r * MM_HIDDEN;
            err = res ? mm_decode_run<true>(moe, y + row, x + row, sel + at, w + at, sh + row, table, first.book,
                            first.res_book)
                      : mm_decode_run<false>(moe, y + row, x + row, sel + at, w + at, sh + row, table, first.book,
                            nullptr);
        }
        return cuda_ok(err, "MiniMax routed decode");
    }
    std::vector<uint32_t> offsets(MM_EXPERTS + 1);
    if (!moe->mma || !mm_group(moe, selected, rows, offsets.data())) { return 0; }
    const auto *x = (const float *)input->ptr, *w = (const float *)weights->ptr, *sh = (const float *)shared->ptr;
    return res ? mm_mma_run<true>(moe, (float *)out->ptr, x, w, sh, matrices, offsets.data(), rows)
               : mm_mma_run<false>(moe, (float *)out->ptr, x, w, sh, matrices, offsets.data(), rows);
}

/* ------------------------------------------------------------------- router */

/* Sigmoid router without groups (models/minimax_m3): one warp per row, lane
 * l holds experts 4l..4l+3. s = sigmoid(logit) in double, rounded once;
 * choice c = s + bias; the top-4 by c, ties to the lower index as torch.topk
 * does on these sizes; weights are the unbiased s over their F32 sum, times
 * routed_scaling_factor 2.0 (exact). */
__global__ void mm_router_kernel(int32_t *ids, float *weights, const float *logits, const float *bias,
        uint32_t rows) {
    const uint32_t lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) { return; }
    float s[4], c[4];
    for (uint32_t j = 0; j < 4; j++) {
        const uint32_t e = lane * 4 + j;
        s[j] = (float)(1.0 / (1.0 + exp(-(double)logits[(uint64_t)row * MM_EXPERTS + e])));
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

/* Router logits of up to MM_ROUTER_ROWS rows in F32 (decode and verify):
 * warp e streams expert e's 6,144 weights once and keeps one dot per row,
 * lane l summing float4 steps l, l + 32, ... in order, then a fixed xor
 * tree. A row's logits do not depend on how many rows share the launch,
 * so verify rows route exactly as decode rows. cuBLAS took 62-116 us per
 * layer at two or three rows (gemmSN) for a 3 MiB matrix. */
enum { MM_ROUTER_ROWS = DS4_MINIMAX_SPLIT_ROWS, MM_ROUTER_WARPS = 4 };

__global__ void __launch_bounds__(MM_ROUTER_WARPS * 32) mm_router_gemv(float *logits, const float *w,
        const float *x, uint32_t rows) {
    const uint32_t lane = threadIdx.x & 31, e = blockIdx.x * MM_ROUTER_WARPS + threadIdx.x / 32;
    const float4 *we = (const float4 *)(w + (uint64_t)e * MM_HIDDEN);
    float acc[MM_ROUTER_ROWS];
    #pragma unroll
    for (uint32_t r = 0; r < MM_ROUTER_ROWS; r++) { acc[r] = 0.0f; }
    for (uint32_t i = lane; i < MM_HIDDEN / 4; i += 32) {
        const float4 a = __ldg(we + i);
        #pragma unroll
        for (uint32_t r = 0; r < MM_ROUTER_ROWS; r++) {
            if (r < rows) {
                const float4 v = ((const float4 *)(x + (uint64_t)r * MM_HIDDEN))[i];
                acc[r] = fmaf(a.x, v.x, acc[r]);
                acc[r] = fmaf(a.y, v.y, acc[r]);
                acc[r] = fmaf(a.z, v.z, acc[r]);
                acc[r] = fmaf(a.w, v.w, acc[r]);
            }
        }
    }
    #pragma unroll
    for (uint32_t r = 0; r < MM_ROUTER_ROWS; r++) {
        float v = acc[r];
        for (uint32_t stride = 16; stride; stride >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, stride); }
        if (!lane && r < rows) { logits[(uint64_t)r * MM_EXPERTS + e] = v; }
    }
}

/* Q8_0 projections of up to MM_ROUTER_ROWS rows (the indexer's index_q and
 * index_k at decode and verify widths): warp o streams output o's 34-byte
 * blocks once, lane l takes element l of every block (d * q is exact in
 * F32) against the F32 input, in block order, then a fixed xor tree. A
 * row's result does not depend on the launch's other rows; the mmvq path
 * picked its column layout by width and quantized the input to Q8_1. */
enum { MM_Q8_WARPS = 4, MM_Q8_BLOCK = 32, MM_Q8_BYTES = 34, MM_TYPE_Q8_0 = 8 };   // ggml's Q8_0 id

__global__ void __launch_bounds__(MM_Q8_WARPS * 32) mm_q8_gemv(float *out, const uint8_t *weight, const float *x,
        uint32_t cols, uint32_t outs, uint32_t rows) {
    const uint32_t lane = threadIdx.x & 31, o = blockIdx.x * MM_Q8_WARPS + threadIdx.x / 32;
    if (o >= outs) { return; }
    const uint32_t blocks = cols / MM_Q8_BLOCK;
    const uint8_t *w = weight + (uint64_t)o * blocks * MM_Q8_BYTES;
    float acc[MM_ROUTER_ROWS];
    #pragma unroll
    for (uint32_t r = 0; r < MM_ROUTER_ROWS; r++) { acc[r] = 0.0f; }
    #pragma unroll 4
    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *block = w + (uint64_t)b * MM_Q8_BYTES;
        const float value = __half2float(__ushort_as_half((uint16_t)(block[0] | block[1] << 8))) *
            (float)(int8_t)block[2 + lane];
        #pragma unroll
        for (uint32_t r = 0; r < MM_ROUTER_ROWS; r++) {
            if (r < rows) { acc[r] = fmaf(value, x[(uint64_t)r * cols + b * MM_Q8_BLOCK + lane], acc[r]); }
        }
    }
    #pragma unroll
    for (uint32_t r = 0; r < MM_ROUTER_ROWS; r++) {
        float v = acc[r];
        for (uint32_t stride = 16; stride; stride >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, stride); }
        if (!lane && r < rows) { out[(uint64_t)r * outs + o] = v; }
    }
}

extern "C" int ds4_gpu_minimax_q8(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t weight, const ds4_gpu_tensor *input,
        uint32_t cols, uint32_t outs, uint32_t rows) {
    const uint64_t bytes = (uint64_t)outs * (cols / MM_Q8_BLOCK) * MM_Q8_BYTES;
    const auto *w = (const uint8_t *)gvq_weight(ctx, weight, MM_TYPE_Q8_0, bytes / 2);
    if (!w || !rows || rows > MM_ROUTER_ROWS || !cols || cols % MM_Q8_BLOCK || !outs ||
        !gvq_floats(input, (uint64_t)rows * cols) || !gvq_floats(out, (uint64_t)rows * outs) || !gvq_stream(ctx)) {
        return 0;
    }
    mm_q8_gemv<<<(outs + MM_Q8_WARPS - 1) / MM_Q8_WARPS, MM_Q8_WARPS * 32, 0, ds4_current_stream()>>>(
        (float *)out->ptr, w, (const float *)input->ptr, cols, outs, rows);
    return cuda_ok(cudaGetLastError(), "MiniMax Q8_0 rows");
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
        mm_router_gemv<<<MM_EXPERTS / MM_ROUTER_WARPS, MM_ROUTER_WARPS * 32, 0, ds4_current_stream()>>>(
            (float *)logits->ptr, w, (const float *)input->ptr, rows);
    } else if (!cublas_ok(cublasGemmEx(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N, MM_EXPERTS, (int)rows, MM_HIDDEN,
            &alpha, w, CUDA_R_32F, MM_HIDDEN, input->ptr, CUDA_R_32F, MM_HIDDEN, &beta, logits->ptr, CUDA_R_32F,
            MM_EXPERTS, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT), "MiniMax router logits")) {
        return 0;
    }
    enum { WARPS = 8 };
    mm_router_kernel<<<(rows + WARPS - 1) / WARPS, WARPS * 32, 0, ds4_current_stream()>>>(
        (int32_t *)selected->ptr, (float *)weights->ptr, (const float *)logits->ptr, b, rows);
    return cuda_ok(cudaGetLastError(), "MiniMax router");
}

/* -------------------------------------------------------------- row kernels */

enum { MM_ROW_THREADS = 256, MM_ROW_VALUES = MM_HIDDEN / MM_ROW_THREADS };

// Sum of `value` over the block (256 threads); every thread gets it.
__device__ __forceinline__ float mm_block_sum(float value, float *scratch) {
    for (uint32_t stride = 16; stride; stride >>= 1) { value += __shfl_xor_sync(0xffffffffu, value, stride); }
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x / 32] = value; }
    __syncthreads();
    float sum = 0.0f;
    for (uint32_t w = 0; w < MM_ROW_THREADS / 32; w++) { sum += scratch[w]; }
    __syncthreads();
    return sum;
}

// Gemma RMSNorm of one 6144-wide row held as 24 values per thread:
// x * rsqrt(mean(x^2) + eps), times (1 + w) in F32, rounded once.
__device__ __forceinline__ void mm_row_rms(const float *x, const __nv_bfloat16 *weight, float *out,
        float *scratch) {
    float sum = 0.0f;
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) { sum = fmaf(x[j], x[j], sum); }
    const float inv = __frsqrt_rn(glmvq_add(__fdiv_rn(mm_block_sum(sum, scratch), (float)MM_HIDDEN), MM_EPS));
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) {
        const uint32_t col = threadIdx.x + j * MM_ROW_THREADS;
        out[j] = glmvq_round(glmvq_mul(glmvq_add(1.0f, __bfloat162float(weight[col])), glmvq_mul(x[j], inv)));
    }
}

__global__ void mm_rms_kernel(float *out, const float *input, const __nv_bfloat16 *weight) {
    __shared__ float scratch[MM_ROW_THREADS / 32];
    const uint64_t base = (uint64_t)blockIdx.x * MM_HIDDEN;
    float x[MM_ROW_VALUES], y[MM_ROW_VALUES];
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) { x[j] = glmvq_round(input[base + threadIdx.x + j * MM_ROW_THREADS]); }
    mm_row_rms(x, weight, y, scratch);
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) { out[base + threadIdx.x + j * MM_ROW_THREADS] = y[j]; }
}

extern "C" int ds4_gpu_minimax_rms(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t weight,
        uint32_t rows) {
    const auto *w = (const __nv_bfloat16 *)gvq_weight(ctx, weight, GVQ_TYPE_BF16, MM_HIDDEN);
    const uint64_t count = (uint64_t)rows * MM_HIDDEN;
    if (!w || !rows || !gvq_floats(out, count) || !gvq_floats(x, count)) { return 0; }
    mm_rms_kernel<<<rows, MM_ROW_THREADS, 0, ds4_current_stream()>>>((float *)out->ptr, (const float *)x->ptr, w);
    return cuda_ok(cudaGetLastError(), "MiniMax RMS");
}

// cur = round(cur + branch), then the next norm when `next` is set.
__global__ void mm_residual_kernel(float *cur, const float *branch, float *norm, const __nv_bfloat16 *next) {
    __shared__ float scratch[MM_ROW_THREADS / 32];
    const uint64_t base = (uint64_t)blockIdx.x * MM_HIDDEN;
    float x[MM_ROW_VALUES], y[MM_ROW_VALUES];
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) {
        const uint64_t at = base + threadIdx.x + j * MM_ROW_THREADS;
        x[j] = glmvq_round(glmvq_add(cur[at], glmvq_round(branch[at])));
        cur[at] = x[j];
    }
    if (!next) { return; }
    mm_row_rms(x, next, y, scratch);
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) { norm[base + threadIdx.x + j * MM_ROW_THREADS] = y[j]; }
}

extern "C" int ds4_gpu_minimax_residual(ds4_gvq_ctx *ctx, ds4_gpu_tensor *cur, const ds4_gpu_tensor *branch,
        ds4_gpu_tensor *norm, uint32_t next, uint32_t rows) {
    const auto *n = next == DS4_GVQ_NO_LINK ? nullptr
        : (const __nv_bfloat16 *)gvq_weight(ctx, next, GVQ_TYPE_BF16, MM_HIDDEN);
    const uint64_t count = (uint64_t)rows * MM_HIDDEN;
    if ((next != DS4_GVQ_NO_LINK && !n) || !rows || !gvq_floats(cur, count) || !gvq_floats(branch, count) ||
        (n && !gvq_floats(norm, count))) { return 0; }
    mm_residual_kernel<<<rows, MM_ROW_THREADS, 0, ds4_current_stream()>>>((float *)cur->ptr,
        (const float *)branch->ptr, n ? (float *)norm->ptr : nullptr, n);
    return cuda_ok(cudaGetLastError(), "MiniMax residual");
}

__global__ void mm_swiglu_kernel(float *out, const float *gate, const float *up, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = glmvq_swiglu_value<GLMVQ_CLAMP_OAI>(glmvq_round(gate[i]), glmvq_round(up[i]), MM_LIMIT);
}

extern "C" int ds4_gpu_minimax_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
        uint32_t rows, uint32_t width) {
    const uint64_t count = (uint64_t)rows * width;
    if (!count || !gvq_floats(out, count) || !gvq_floats(gate, count) || !gvq_floats(up, count)) { return 0; }
    mm_swiglu_kernel<<<(uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0,
        ds4_current_stream()>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, count);
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

/* Per-head Gemma RMSNorm, one block per (head, row), rounded once; then the
 * NeoX rotation of dims [0, 64) as torch's separate F32 ops
 * (x1 c - x2 s, x2 c + x1 s), rounded once; dims [64, 128) pass. */
__global__ void mm_qk_kernel(float *x, const __nv_bfloat16 *weight, uint32_t heads, const float *rope) {
    __shared__ float scratch[MM_HEAD / 32];
    __shared__ float y[MM_HEAD];
    const uint32_t h = blockIdx.x, r = blockIdx.y, i = threadIdx.x;
    float *p = x + ((uint64_t)r * heads + h) * MM_HEAD;
    const float v = glmvq_round(p[i]);
    float sum = v * v;
    for (uint32_t stride = 16; stride; stride >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, stride); }
    if ((i & 31) == 0) { scratch[i / 32] = sum; }
    __syncthreads();
    sum = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    const float inv = __frsqrt_rn(glmvq_add(__fdiv_rn(sum, (float)MM_HEAD), MM_EPS));
    y[i] = glmvq_round(glmvq_mul(glmvq_add(1.0f, __bfloat162float(weight[i])), glmvq_mul(v, inv)));
    __syncthreads();
    if (i >= MM_ROT) {
        p[i] = y[i];
        return;
    }
    const float *t = rope + (uint64_t)r * MM_ROT;
    const uint32_t j = i % (MM_ROT / 2);
    const float c = t[j], s = t[MM_ROT / 2 + j];
    const float a = y[j], b = y[j + MM_ROT / 2];
    p[i] = i < MM_ROT / 2 ? glmvq_round(glmvq_add(glmvq_mul(a, c), -glmvq_mul(b, s)))
                          : glmvq_round(glmvq_add(glmvq_mul(b, c), glmvq_mul(a, s)));
}

extern "C" int ds4_gpu_minimax_qk(ds4_gvq_ctx *ctx, ds4_gpu_tensor *x, uint32_t weight, uint32_t heads,
        uint32_t rows, const ds4_gpu_tensor *rope) {
    const auto *w = (const __nv_bfloat16 *)gvq_weight(ctx, weight, GVQ_TYPE_BF16, MM_HEAD);
    if (!w || !rows || !heads || !gvq_floats(x, (uint64_t)rows * heads * MM_HEAD) ||
        !gvq_floats(rope, (uint64_t)rows * MM_ROT)) { return 0; }
    mm_qk_kernel<<<dim3(heads, rows), MM_HEAD, 0, ds4_current_stream()>>>((float *)x->ptr, w, heads,
        (const float *)rope->ptr);
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

// K and V rows of `rows` tokens into their pages: warp w of block (row,
// side) encodes KV head w. A table maps blocks to pool slots (SSD tier).
__global__ void mm_kv_store_kernel(uint8_t *pages, const int32_t *table, const float *k, const float *v,
        uint32_t pos) {
    const uint32_t r = blockIdx.x, side = blockIdx.y, head = threadIdx.x / 32;
    const uint32_t t = pos + r, block = t / MM_BLOCK, slot = t % MM_BLOCK;
    const uint64_t page_id = table ? (uint64_t)table[block] : block;
    uint8_t *page = pages + (page_id * MM_KV_HEADS + head) * MM_PAGE + side * (MM_PAGE / 2);
    const float *src = (side ? v : k) + ((uint64_t)r * MM_KV_HEADS + head) * MM_HEAD;
    mm_fp4_encode(src, page + slot * MM_FP4_DATA, page + MM_BLOCK * MM_FP4_DATA + slot * (MM_FP4_ROW - MM_FP4_DATA));
}

// The pages of blocks [0, ceil(pos / 128)) are addressable: directly, or
// through a table of that many slots.
static bool mm_pages_ok(const ds4_gpu_tensor *pages, const ds4_gpu_tensor *table, uint32_t pos) {
    const uint64_t blocks = (pos + MM_BLOCK - 1) / MM_BLOCK;
    return pages && (table ? table->bytes >= blocks * sizeof(int32_t)
                           : pages->bytes >= blocks * MM_KV_HEADS * MM_PAGE);
}

extern "C" int ds4_gpu_minimax_kv_store(ds4_gpu_tensor *pages, const ds4_gpu_tensor *table,
        const ds4_gpu_tensor *k, const ds4_gpu_tensor *v, uint32_t rows, uint32_t pos) {
    const uint64_t count = (uint64_t)rows * MM_KV_HEADS * MM_HEAD;
    if (!rows || !mm_pages_ok(pages, table, pos + rows) || !gvq_floats(k, count) || !gvq_floats(v, count)) { return 0; }
    mm_kv_store_kernel<<<dim3(rows, 2), MM_KV_HEADS * 32, 0, ds4_current_stream()>>>((uint8_t *)pages->ptr,
        table ? (const int32_t *)table->ptr : nullptr, (const float *)k->ptr, (const float *)v->ptr, pos);
    return cuda_ok(cudaGetLastError(), "MiniMax K/V store");
}

// Index keys: one 72-byte row per token, data then scales.
__global__ void mm_index_store_kernel(uint8_t *pool, const float *key, uint32_t pos, uint32_t rows) {
    const uint32_t r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (r >= rows) { return; }
    uint8_t *row = pool + (uint64_t)(pos + r) * MM_FP4_ROW;
    mm_fp4_encode(key + (uint64_t)r * MM_HEAD, row, row + MM_FP4_DATA);
}

extern "C" int ds4_gpu_minimax_index_store(ds4_gpu_tensor *pool, const ds4_gpu_tensor *key, uint32_t rows,
        uint32_t pos) {
    if (!rows || !pool || pool->bytes < (uint64_t)(pos + rows) * MM_FP4_ROW ||
        !gvq_floats(key, (uint64_t)rows * MM_HEAD)) { return 0; }
    enum { WARPS = 4 };
    mm_index_store_kernel<<<(rows + WARPS - 1) / WARPS, WARPS * 32, 0, ds4_current_stream()>>>(
        (uint8_t *)pool->ptr, (const float *)key->ptr, pos, rows);
    return cuda_ok(cudaGetLastError(), "MiniMax index store");
}

// The E2M1 values of a byte's two nibbles times their group scale: exact in F32.
__device__ __forceinline__ float2 mm_fp4_pair(uint8_t byte, float scale) {
    return make_float2(ds4_fp4_nibble_to_f32(byte & 15u) * scale, ds4_fp4_nibble_to_f32(byte >> 4) * scale);
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

/* Block scores of up to MM_SCORE_FEW rows (decode, verify) against one key
 * block per CTA: the block's raw keys land in shared memory with 8-byte
 * loads, thread t decodes key t and dots it with every row and head in the
 * tiled kernel's d order, then a block max. Same scores bit for bit; a
 * decode row reads the pool once at stream rate instead of decoding every
 * block into F32 tiles. */
enum { MM_SCORE_FEW = 8 };

__global__ void __launch_bounds__(MM_BLOCK) mm_score_few_kernel(float *scores, const float *query, const uint8_t *pool,
        uint32_t pos, uint32_t first, uint32_t count, uint32_t stride) {
    __shared__ __align__(16) uint8_t raw[MM_BLOCK * MM_FP4_ROW];
    __shared__ __align__(16) float q[MM_SCORE_FEW * MM_KV_HEADS][MM_HEAD];
    __shared__ float part[MM_BLOCK / 32][MM_SCORE_FEW * MM_KV_HEADS];
    const uint32_t block = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const uint32_t last = pos + first + count - 1, t0 = block * MM_BLOCK;
    if (t0 > last) { return; }
    const uint32_t visible = min((uint32_t)MM_BLOCK, last + 1 - t0);
    const uint2 *src = (const uint2 *)(pool + (uint64_t)t0 * MM_FP4_ROW);
    for (uint32_t i = tid; i < visible * (MM_FP4_ROW / 8); i += MM_BLOCK) { ((uint2 *)raw)[i] = src[i]; }
    for (uint32_t i = tid; i < count * MM_KV_HEADS * MM_HEAD; i += MM_BLOCK) {
        q[i / MM_HEAD][i % MM_HEAD] = glmvq_round(query[(uint64_t)first * MM_KV_HEADS * MM_HEAD + i]);
    }
    __syncthreads();
    float dot[MM_SCORE_FEW * MM_KV_HEADS];
    #pragma unroll
    for (uint32_t v = 0; v < MM_SCORE_FEW * MM_KV_HEADS; v++) { dot[v] = 0.0f; }
    if (tid < visible) {
        const uint8_t *key = raw + tid * MM_FP4_ROW;
        #pragma unroll 1
        for (uint32_t g = 0; g < MM_HEAD / 16; g++) {
            const float scale = glmvq_e4m3(key[MM_FP4_DATA + g]);
            #pragma unroll
            for (uint32_t b = 0; b < 8; b++) {
                const float2 kv = mm_fp4_pair(key[g * 8 + b], scale);
                const uint32_t d = g * 16 + b * 2;
                #pragma unroll
                for (uint32_t v = 0; v < MM_SCORE_FEW * MM_KV_HEADS; v++) {
                    if (v < count * MM_KV_HEADS) {
                        dot[v] = fmaf(q[v][d], kv.x, dot[v]);
                        dot[v] = fmaf(q[v][d + 1], kv.y, dot[v]);
                    }
                }
            }
        }
    }
    // Row r sees keys up to its position; past it (or past `visible`) -inf.
    #pragma unroll
    for (uint32_t v = 0; v < MM_SCORE_FEW * MM_KV_HEADS; v++) {
        if (v >= count * MM_KV_HEADS) { continue; }
        const uint32_t p = pos + first + v / MM_KV_HEADS;
        float best = tid < visible && t0 + tid <= p ? dot[v] : -INFINITY;
        for (uint32_t s = 16; s; s >>= 1) { best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, s)); }
        if (!lane) { part[warp][v] = best; }
    }
    __syncthreads();
    if (tid < count * MM_KV_HEADS) {
        float best = part[0][tid];
        for (uint32_t w = 1; w < MM_BLOCK / 32; w++) { best = fmaxf(best, part[w][tid]); }
        scores[(uint64_t)tid * stride + block] = best;
    }
}

/* Top-16 blocks of one (row, KV head) per warp: the own block first, then
 * fifteen rounds of warp argmax over the earlier blocks' scores (ties to the
 * lower block), then the list sorted ascending; -1 fills the tail. Rows
 * with at most 16 visible blocks take them all and read no scores. */
__global__ void mm_topk_kernel(int32_t *blocks, float *scores, uint32_t pos, uint32_t first, uint32_t count,
        uint32_t stride) {
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
    float *s = scores + ((uint64_t)row * MM_KV_HEADS + head) * stride;
    int32_t pick = lane == 0 ? (int32_t)own : -1;
    for (uint32_t k = 1; k < MM_TOPK; k++) {
        float best = -INFINITY;
        uint32_t index = UINT32_MAX;
        for (uint32_t b = lane; b < own; b += 32) {
            const float value = s[b];
            if (value > best || (index == UINT32_MAX && value == best)) { best = value; index = b; }
        }
        for (uint32_t stride2 = 16; stride2; stride2 >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, stride2);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, index, stride2);
            if (ob > best || (ob == best && oi < index)) { best = ob; index = oi; }
        }
        if (lane == k) { pick = index == UINT32_MAX ? -1 : (int32_t)index; }
        if (lane == 0 && index != UINT32_MAX) { s[index] = -INFINITY; }
        __syncwarp();
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
                mm_score_few_kernel<<<stride, MM_BLOCK, 0, stream>>>((float *)scores->ptr, (const float *)query->ptr,
                    (const uint8_t *)pool->ptr, pos, first, count, stride);
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
            mm_topk_kernel<<<(count * MM_KV_HEADS + WARPS - 1) / WARPS, WARPS * 32, 0, stream>>>(
                (int32_t *)blocks->ptr, (float *)scores->ptr, pos, first, count, stride);
        } else {
            mm_topk_kernel<<<(count * MM_KV_HEADS + WARPS - 1) / WARPS, WARPS * 32, 0, stream>>>(
                (int32_t *)blocks->ptr, (float *)scores->ptr, pos, first, count, 1u);
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
                #pragma unroll
                for (uint32_t w = 0; w < 8; w++) {
                    const float s = glmvq_e4m3((uint8_t)(sc >> (w / 2 * 8)));
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; k++) {
                        const uint8_t byte = (uint8_t)(words[w] >> (k * 8));
                        packed[w * 4 + k] = mm_bf16x2(ds4_fp4_nibble_to_f32(byte & 15u) * s,
                            ds4_fp4_nibble_to_f32(byte >> 4) * s);
                    }
                }
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

__global__ void __launch_bounds__(MM_AT_ROWS * 32, 1) mm_attn_mma_kernel(float *heads, const float *q,
        const uint8_t *pages, const int32_t *table, const int32_t *blocks, uint32_t pos, uint32_t rows) {
    extern __shared__ __align__(16) uint8_t mm_at_smem[];
    __nv_bfloat16 *ks = (__nv_bfloat16 *)mm_at_smem, *vs = ks + MM_BLOCK * MM_AT_LD;
    __shared__ uint32_t next[MM_AT_ROWS];
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32, g = lane >> 2, t = lane & 3;
    const uint32_t kv = blockIdx.y, row = blockIdx.x * MM_AT_ROWS + warp;
    const bool live = row < rows;
    const uint32_t r = live ? row : rows - 1, p = pos + r;
    const int32_t *list = blocks + ((uint64_t)r * MM_KV_HEADS + kv) * MM_TOPK;
    // Q as A fragments (16 heads x 128 dims, eight k-steps), rounded once.
    uint32_t qa[8][4];
    const float *qr = q + ((uint64_t)r * MM_HEADS + kv * MM_GROUP) * MM_HEAD;
    #pragma unroll
    for (uint32_t j = 0; j < 8; j++) {
        const uint32_t c = 16 * j + 2 * t;
        qa[j][0] = mm_bf16x2(qr[g * MM_HEAD + c], qr[g * MM_HEAD + c + 1]);
        qa[j][1] = mm_bf16x2(qr[(g + 8) * MM_HEAD + c], qr[(g + 8) * MM_HEAD + c + 1]);
        qa[j][2] = mm_bf16x2(qr[g * MM_HEAD + c + 8], qr[g * MM_HEAD + c + 9]);
        qa[j][3] = mm_bf16x2(qr[(g + 8) * MM_HEAD + c + 8], qr[(g + 8) * MM_HEAD + c + 9]);
    }
    float o[16][4];
    #pragma unroll
    for (uint32_t n = 0; n < 16; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f; }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float scale = 0.08838834764831845f;   // 128^-0.5
    uint32_t cursor = 0;
    for (;;) {
        const int32_t mine = live && cursor < MM_TOPK ? list[cursor] : -1;
        if (!lane) { next[warp] = mine < 0 ? UINT32_MAX : (uint32_t)mine; }
        __syncthreads();
        uint32_t b = UINT32_MAX;
        #pragma unroll
        for (uint32_t w = 0; w < MM_AT_ROWS; w++) { b = min(b, next[w]); }
        if (b == UINT32_MAX) { break; }
        // Block b's K and V pages of this head into shared memory as BF16:
        // 512 units of (side, key, 64 values), 32 data bytes and 4 scales each.
        const uint64_t page_id = table ? (uint64_t)table[b] : (uint64_t)b;
        const uint8_t *page = pages + (page_id * MM_KV_HEADS + kv) * MM_PAGE;
        for (uint32_t i = threadIdx.x; i < 4u * MM_BLOCK; i += blockDim.x) {
            const uint32_t side = i / (2u * MM_BLOCK), key = i / 2 % MM_BLOCK, half = i % 2;
            const uint8_t *base = page + side * (MM_PAGE / 2);
            const uint4 d0 = *(const uint4 *)(base + key * MM_FP4_DATA + half * 32);
            const uint4 d1 = *(const uint4 *)(base + key * MM_FP4_DATA + half * 32 + 16);
            const uint32_t sc = *(const uint32_t *)(base + MM_BLOCK * MM_FP4_DATA + key * 8 + half * 4);
            const uint32_t words[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
            uint32_t packed[32];
            #pragma unroll
            for (uint32_t w = 0; w < 8; w++) {
                const float s = glmvq_e4m3((uint8_t)(sc >> (w / 2 * 8)));
                #pragma unroll
                for (uint32_t k = 0; k < 4; k++) {
                    const uint8_t byte = (uint8_t)(words[w] >> (k * 8));
                    packed[w * 4 + k] = mm_bf16x2(ds4_fp4_nibble_to_f32(byte & 15u) * s,
                        ds4_fp4_nibble_to_f32(byte >> 4) * s);
                }
            }
            uint4 *dst = (uint4 *)((side ? vs : ks) + key * MM_AT_LD + half * 64);
            #pragma unroll
            for (uint32_t k = 0; k < 8; k++) {
                dst[k] = make_uint4(packed[k * 4], packed[k * 4 + 1], packed[k * 4 + 2], packed[k * 4 + 3]);
            }
        }
        __syncthreads();
        if (mine == (int32_t)b) {
            cursor++;
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
        }
        __syncthreads();
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    if (!live) { return; }
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    float *out = heads + ((uint64_t)row * MM_HEADS + kv * MM_GROUP) * MM_HEAD;
    #pragma unroll
    for (uint32_t n = 0; n < 16; n++) {
        const uint32_t d = n * 8 + 2 * t;
        out[g * MM_HEAD + d] = glmvq_round(o[n][0] * i0);
        out[g * MM_HEAD + d + 1] = glmvq_round(o[n][1] * i0);
        out[(g + 8) * MM_HEAD + d] = glmvq_round(o[n][2] * i1);
        out[(g + 8) * MM_HEAD + d + 1] = glmvq_round(o[n][3] * i1);
    }
}

/* Decode and verify attention on BF16 tensor cores, split by selected
 * block: CTA (KV head, list entry j, row) decodes the row's block j K/V once
 * into shared memory (as mm_attn_mma_kernel), warp w scores its 32 keys for
 * the 16 heads (S = Q K^T, the causal mask in the own block, F32 softmax,
 * O = P V with P in BF16), and the four warps merge into the (O, max, sum)
 * partial that mm_attn_combine reads. A row's arithmetic does not depend on
 * the other rows: verify rows equal decode rows bit for bit. */
__global__ void __launch_bounds__(128) mm_attn_split_mma_kernel(float *partial, const float *q,
        const uint8_t *pages, const int32_t *table, const int32_t *blocks, uint32_t pos) {
    extern __shared__ __align__(16) uint8_t mm_at_smem[];
    __nv_bfloat16 *ks = (__nv_bfloat16 *)mm_at_smem, *vs = ks + MM_BLOCK * MM_AT_LD;
    __shared__ float wm[4][MM_GROUP], wl[4][MM_GROUP];
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32, g = lane >> 2, t = lane & 3;
    const uint32_t kv = blockIdx.x / MM_TOPK, j = blockIdx.x % MM_TOPK, row = blockIdx.y, p = pos + row;
    float *out = partial + ((uint64_t)row * MM_SPLITS + blockIdx.x) * MM_GROUP * (MM_HEAD + 2);
    const int32_t b = blocks[((uint64_t)row * MM_KV_HEADS + kv) * MM_TOPK + j];
    if (b < 0) {
        if (threadIdx.x < MM_GROUP) {
            out[threadIdx.x * (MM_HEAD + 2) + MM_HEAD] = -INFINITY;
            out[threadIdx.x * (MM_HEAD + 2) + MM_HEAD + 1] = 0.0f;
        }
        return;
    }
    const uint64_t page_id = table ? (uint64_t)table[b] : (uint64_t)b;
    const uint8_t *page = pages + (page_id * MM_KV_HEADS + kv) * MM_PAGE;
    for (uint32_t i = threadIdx.x; i < 4u * MM_BLOCK; i += blockDim.x) {
        const uint32_t side = i / (2u * MM_BLOCK), key = i / 2 % MM_BLOCK, half = i % 2;
        const uint8_t *base = page + side * (MM_PAGE / 2);
        const uint4 d0 = *(const uint4 *)(base + key * MM_FP4_DATA + half * 32);
        const uint4 d1 = *(const uint4 *)(base + key * MM_FP4_DATA + half * 32 + 16);
        const uint32_t sc = *(const uint32_t *)(base + MM_BLOCK * MM_FP4_DATA + key * 8 + half * 4);
        const uint32_t words[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
        uint32_t packed[32];
        #pragma unroll
        for (uint32_t w = 0; w < 8; w++) {
            const float s = glmvq_e4m3((uint8_t)(sc >> (w / 2 * 8)));
            #pragma unroll
            for (uint32_t k = 0; k < 4; k++) {
                const uint8_t byte = (uint8_t)(words[w] >> (k * 8));
                packed[w * 4 + k] = mm_bf16x2(ds4_fp4_nibble_to_f32(byte & 15u) * s,
                    ds4_fp4_nibble_to_f32(byte >> 4) * s);
            }
        }
        uint4 *dst = (uint4 *)((side ? vs : ks) + key * MM_AT_LD + half * 64);
        #pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
            dst[k] = make_uint4(packed[k * 4], packed[k * 4 + 1], packed[k * 4 + 2], packed[k * 4 + 3]);
        }
    }
    uint32_t qa[8][4];
    const float *qr = q + ((uint64_t)row * MM_HEADS + kv * MM_GROUP) * MM_HEAD;
    #pragma unroll
    for (uint32_t s = 0; s < 8; s++) {
        const uint32_t c = 16 * s + 2 * t;
        qa[s][0] = mm_bf16x2(qr[g * MM_HEAD + c], qr[g * MM_HEAD + c + 1]);
        qa[s][1] = mm_bf16x2(qr[(g + 8) * MM_HEAD + c], qr[(g + 8) * MM_HEAD + c + 1]);
        qa[s][2] = mm_bf16x2(qr[g * MM_HEAD + c + 8], qr[g * MM_HEAD + c + 9]);
        qa[s][3] = mm_bf16x2(qr[(g + 8) * MM_HEAD + c + 8], qr[(g + 8) * MM_HEAD + c + 9]);
    }
    __syncthreads();
    // Warp w: keys 32w..32w+31 (four 8-key n-tiles).
    const uint32_t k0 = warp * 32, t0 = (uint32_t)b * MM_BLOCK;
    float s[4][4];
    #pragma unroll
    for (uint32_t n = 0; n < 4; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f; }
    #pragma unroll
    for (uint32_t st = 0; st < 8; st++) {
        #pragma unroll
        for (uint32_t n = 0; n < 4; n += 2) {
            const uint32_t mi = lane / 8;
            const uint32_t key = k0 + (n + (mi >> 1)) * 8 + lane % 8, dim = 16 * st + (mi & 1) * 8;
            uint32_t b4[4];
            mm_ldsm4(b4, ks + key * MM_AT_LD + dim, false);
            mm_mma_bf16(s[n], qa[st], b4[0], b4[1]);
            mm_mma_bf16(s[n + 1], qa[st], b4[2], b4[3]);
        }
    }
    const float scale = 0.08838834764831845f;
    float m0 = -INFINITY, m1 = -INFINITY;
    #pragma unroll
    for (uint32_t n = 0; n < 4; n++) {
        const uint32_t key = t0 + k0 + n * 8 + 2 * t;
        s[n][0] = key <= p ? s[n][0] * scale : -INFINITY;
        s[n][1] = key + 1 <= p ? s[n][1] * scale : -INFINITY;
        s[n][2] = key <= p ? s[n][2] * scale : -INFINITY;
        s[n][3] = key + 1 <= p ? s[n][3] * scale : -INFINITY;
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
    float o[16][4];
    #pragma unroll
    for (uint32_t n = 0; n < 16; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f; }
    #pragma unroll
    for (uint32_t kk = 0; kk < 2; kk++) {
        const uint32_t pa[4] = {mm_bf16x2(s[2 * kk][0], s[2 * kk][1]), mm_bf16x2(s[2 * kk][2], s[2 * kk][3]),
            mm_bf16x2(s[2 * kk + 1][0], s[2 * kk + 1][1]), mm_bf16x2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
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
    // Merge the four warps: per-warp (max, sum) and O rows into shared
    // memory (the K tile is no longer read), then thread d sums dim d.
    __syncthreads();
    float *wo = (float *)mm_at_smem;   // [4 warps][16 heads][128]
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
    __syncthreads();
    const uint32_t d = threadIdx.x;
    #pragma unroll 1
    for (uint32_t h = 0; h < MM_GROUP; h++) {
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

extern "C" int ds4_gpu_minimax_attn(ds4_gpu_tensor *heads, const ds4_gpu_tensor *q, const ds4_gpu_tensor *pages,
        const ds4_gpu_tensor *table, const ds4_gpu_tensor *blocks, ds4_gpu_tensor *partials, uint32_t rows,
        uint32_t pos) {
    const uint64_t count = (uint64_t)rows * MM_HEADS * MM_HEAD;
    if (!rows || !mm_pages_ok(pages, table, pos + rows) || !gvq_floats(heads, count) || !gvq_floats(q, count) ||
        !blocks || blocks->bytes < (uint64_t)rows * MM_KV_HEADS * MM_TOPK * sizeof(int32_t)) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    const int32_t *slots = table ? (const int32_t *)table->ptr : nullptr;
    if (rows > DS4_MINIMAX_SPLIT_ROWS && !mm_attn_simt()) {
        static bool ready = false;
        if (!ready) {
            if (!cuda_ok(cudaFuncSetAttribute(mm_attn_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    MM_AT_SMEM), "MiniMax MMA attention shared memory")) { return 0; }
            ready = true;
        }
        mm_attn_mma_kernel<<<dim3((rows + MM_AT_ROWS - 1) / MM_AT_ROWS, MM_KV_HEADS), MM_AT_ROWS * 32, MM_AT_SMEM,
            stream>>>((float *)heads->ptr, (const float *)q->ptr, (const uint8_t *)pages->ptr, slots,
            (const int32_t *)blocks->ptr, pos, rows);
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
            if (!cuda_ok(cudaFuncSetAttribute(mm_attn_split_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    MM_AT_SMEM), "MiniMax split MMA attention shared memory")) { return 0; }
            ready = true;
        }
        mm_attn_split_mma_kernel<<<dim3(MM_SPLITS, rows), 128, MM_AT_SMEM, stream>>>((float *)partials->ptr,
            (const float *)q->ptr, (const uint8_t *)pages->ptr, slots, (const int32_t *)blocks->ptr, pos);
    } else {
        mm_attn_kernel<false><<<MM_SPLITS, MM_ATTN_THREADS, 0, stream>>>(nullptr, (float *)partials->ptr,
            (const float *)q->ptr, (const uint8_t *)pages->ptr, slots, (const int32_t *)blocks->ptr, pos, 1u);
    }
    mm_attn_combine<<<dim3(MM_HEADS, rows), MM_HEAD, 0, stream>>>((float *)heads->ptr,
        (const float *)partials->ptr);
    return cuda_ok(cudaGetLastError(), "MiniMax decode attention");
}

/* ------------------------------------------------------------------ DSpark */

enum { MM_DRAFT_TAPS = 5, MM_DRAFT_KV = MM_KV_HEADS * MM_HEAD, MM_DRAFT_WARPS = 8, MM_MARKOV_RANK = 256 };

// taps[r][k] = the residual entering tap layer k, stored as BF16 (exact: the
// residual stream holds BF16-rounded values).
__global__ void mm_draft_tap_kernel(__nv_bfloat16 *taps, const float *cur, uint32_t k, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    taps[(i / MM_HIDDEN * MM_DRAFT_TAPS + k) * MM_HIDDEN + i % MM_HIDDEN] = __float2bfloat16_rn(cur[i]);
}

extern "C" int ds4_gpu_minimax_draft_tap(ds4_gpu_tensor *taps, const ds4_gpu_tensor *cur, uint32_t k, uint32_t rows) {
    const uint64_t count = (uint64_t)rows * MM_HIDDEN;
    if (!taps || k >= MM_DRAFT_TAPS || taps->bytes < count * MM_DRAFT_TAPS * sizeof(__nv_bfloat16) ||
        !gvq_floats(cur, count)) { return 0; }
    mm_draft_tap_kernel<<<(count + 255) / 256, 256, 0, ds4_current_stream()>>>((__nv_bfloat16 *)taps->ptr,
        (const float *)cur->ptr, k, count);
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

extern "C" void ds4_gpu_minimax_host_free(void *host, ds4_gpu_tensor *device) {
    if (device) {
        cuda_mem_note_free(device->memc, DS4_MEMD_PINNED_HOST, device->bytes, device->bytes);
        free(device);
    }
    if (host) { cudaFreeHost(host); }
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

__global__ void mm_embed_kernel(float *cur, const __nv_bfloat16 *rows, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { cur[i] = __bfloat162float(rows[i]); }
}

extern "C" int ds4_gpu_minimax_embed(ds4_gpu_tensor *cur, const ds4_gpu_tensor *rows_bf16, uint32_t rows) {
    const uint64_t count = (uint64_t)rows * MM_HIDDEN;
    if (!rows || !gvq_floats(cur, count) || !rows_bf16 || rows_bf16->bytes < count * sizeof(__nv_bfloat16)) {
        return 0;
    }
    mm_embed_kernel<<<(uint32_t)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS), GLMVQ_THREADS, 0,
        ds4_current_stream()>>>((float *)cur->ptr, (const __nv_bfloat16 *)rows_bf16->ptr, count);
    return cuda_ok(cudaGetLastError(), "MiniMax embedding");
}
