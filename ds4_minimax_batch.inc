/* MiniMax banks. The scheduler serializes bank forwards under the engine
 * generation lock, so graph 0's scratch serves every bank; each bank owns
 * its index keys and K/V pages. Both are position-indexed and every layer
 * keeps every position, so any prefix of a bank is exact on its own: a fork
 * copies the prefix (or rewinds in place) and partial reuse needs no
 * checkpoint slots. */
enum { MINIMAX_BANK_MAX = DS4_MULTISEQ_MAX_SEQ };
typedef enum { MINIMAX_MORE_ROWS, MINIMAX_FINAL_ROWS } minimax_prefill_mode;
typedef enum { MINIMAX_STRICT_BANKS, MINIMAX_FIT_BANKS } minimax_fit_mode;

typedef struct {
    uint32_t max_seq, ctx_size, prefill_cap;
    ds4_minimax_graph *graph;
    float *bank_logits;
    uint8_t *bank_logits_valid;
} ds4_minimax_batch_runtime;

// What one more bank adds beyond graph 0: its index keys and K/V pages.
static uint64_t minimax_bank_bytes(uint32_t ctx) { return minimax_kv_bytes(ctx); }

// Index rows [0, pos) and the pages of blocks [0, ceil(pos / 128)) of
// every layer of `src` into `dst`.
static bool minimax_rows_copy(ds4_minimax_graph *dst, const ds4_minimax_graph *src, uint32_t pos) {
    for (uint32_t il = 0; pos && il < MINIMAX_LAYERS; il++) {
        if (!ds4_gpu_tensor_copy(dst->index[il], 0, src->index[il], 0, minimax_index_bytes(pos)) ||
            !ds4_gpu_tensor_copy(dst->pages[il], 0, src->pages[il], 0, minimax_page_bytes(pos))) { return false; }
    }
    return true;
}

static void minimax_batch_free(ds4_minimax_batch_runtime *rt) {
    if (!rt) { return; }
    // Banks borrow graph 0's buffers: free them first.
    for (uint32_t b = rt->max_seq; rt->graph && b > 0; b--) { minimax_graph_free(&rt->graph[b - 1u]); }
    free(rt->graph);
    free(rt->bank_logits);
    free(rt->bank_logits_valid);
    free(rt);
}

static ds4_minimax_batch_runtime *minimax_batch_create(ds4_engine *e, uint32_t ctx, uint32_t max_seq,
        uint32_t cap) {
    if (!e || !ctx || !max_seq || max_seq > MINIMAX_BANK_MAX || !cap || cap > ctx) { return NULL; }
    ds4_minimax_batch_runtime *rt = xcalloc(1, sizeof(*rt));
    rt->ctx_size = ctx;
    rt->max_seq = max_seq;
    rt->prefill_cap = cap;
    rt->graph = xcalloc(max_seq, sizeof(*rt->graph));
    if (!minimax_graph_alloc(&rt->graph[0], &e->model, &e->weights, ctx, cap)) {
        minimax_batch_free(rt);
        return NULL;
    }
    for (uint32_t b = 1; b < max_seq; b++) {
        if (!minimax_graph_bank(&rt->graph[b], &rt->graph[0])) {
            minimax_batch_free(rt);
            return NULL;
        }
    }
    rt->bank_logits = xcalloc((size_t)max_seq * DS4_N_VOCAB, sizeof(float));
    rt->bank_logits_valid = xcalloc(max_seq, sizeof(uint8_t));
    return rt;
}

static bool minimax_bank_read_logits(ds4_minimax_batch_runtime *rt, uint32_t bank) {
    float *dst = rt->bank_logits + (size_t)bank * DS4_N_VOCAB;
    bool ok = ds4_gpu_tensor_read(rt->graph[bank].logits, 0, dst, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    for (uint32_t i = 0; ok && i < DS4_N_VOCAB; i++) { ok = isfinite(dst[i]); }
    rt->bank_logits_valid[bank] = ok;
    if (!ok) { rt->graph[bank].failed = true; }
    return ok;
}

static void minimax_bank_reset(ds4_minimax_batch_runtime *rt, uint32_t bank) {
    rt->bank_logits_valid[bank] = 0;
    (void)minimax_graph_reset(&rt->graph[bank]);
}

// Exact fork at the source's frontier: its rows and its logits.
static bool minimax_bank_copy(ds4_minimax_batch_runtime *rt, uint32_t src, uint32_t dst, uint32_t tokens) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || src == dst) { return false; }
    ds4_minimax_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (sg->failed || sg->position != tokens || tokens > dg->ctx_cap) { return false; }
    dg->failed = true;
    if (!minimax_rows_copy(dg, sg, tokens) || !ds4_gpu_synchronize()) { return false; }
    dg->position = tokens;
    dg->failed = false;
    memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB, rt->bank_logits + (size_t)src * DS4_N_VOCAB,
           DS4_N_VOCAB * sizeof(float));
    rt->bank_logits_valid[dst] = rt->bank_logits_valid[src];
    return true;
}

// Partial fork below the source's frontier: the prefix [0, pos) (in place
// when src == dst). The rows past it are rewritten before any read; the
// logits at pos are unknown, so the caller prefills at least one row.
static bool minimax_bank_fork(ds4_minimax_batch_runtime *rt, uint32_t src, uint32_t dst, uint32_t pos) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || !pos) { return false; }
    ds4_minimax_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (sg->failed || pos > sg->position || pos > dg->ctx_cap) { return false; }
    dg->failed = true;
    rt->bank_logits_valid[dst] = 0;
    if (src != dst && (!minimax_rows_copy(dg, sg, pos) || !ds4_gpu_synchronize())) { return false; }
    dg->position = pos;
    dg->failed = false;
    return true;
}

static bool minimax_bank_prefill(ds4_minimax_batch_runtime *rt, ds4_engine *e, uint32_t bank, const int *tokens,
        uint32_t rows, uint32_t pos, minimax_prefill_mode mode) {
    if (!rt || !e || bank >= rt->max_seq) { return false; }
    rt->bank_logits_valid[bank] = 0;
    const minimax_head_mode head = mode == MINIMAX_FINAL_ROWS ? MINIMAX_HEAD_LAST : MINIMAX_HEAD_SKIP;
    if (!minimax_forward(&rt->graph[bank], &e->model, &e->weights, tokens, rows, pos, head)) { return false; }
    return mode == MINIMAX_MORE_ROWS || minimax_bank_read_logits(rt, bank);
}

/* Each bank attends over its own K/V, so decode rows of different banks
 * share no forward: run them in bank order. */
static bool minimax_bank_decode(ds4_minimax_batch_runtime *rt, ds4_engine *e, const uint32_t *banks,
        const int *tokens, const uint32_t *positions, uint32_t rows) {
    if (!rt || !e || !banks || !tokens || !positions || !rows) { return false; }
    for (uint32_t i = 0; i < rows; i++) {
        if (banks[i] >= rt->max_seq ||
            !minimax_bank_prefill(rt, e, banks[i], &tokens[i], 1u, positions[i], MINIMAX_FINAL_ROWS)) {
            return false;
        }
    }
    return true;
}
