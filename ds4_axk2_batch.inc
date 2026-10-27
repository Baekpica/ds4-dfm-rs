/* A.X-K2 banks. The scheduler serializes bank forwards under the engine
 * generation lock, so graph 0's scratch serves every bank; each bank owns
 * its latent rows, rope keys and indexer keys. All three are
 * position-indexed and every layer keeps every position, so any prefix of a
 * bank is exact on its own: a fork copies the prefix (or rewinds in place)
 * and partial reuse needs no checkpoint slots (MiniMax's rule). */
enum { AXK2_BANK_MAX = DS4_MULTISEQ_MAX_SEQ };
typedef enum { AXK2_MORE_ROWS, AXK2_FINAL_ROWS } axk2_prefill_mode;
typedef enum { AXK2_STRICT_BANKS, AXK2_FIT_BANKS } axk2_fit_mode;

typedef struct {
    uint32_t max_seq, ctx_size, prefill_cap;
    ds4_axk2_graph *graph;
    float *bank_logits;
    uint8_t *bank_logits_valid;
} ds4_axk2_batch_runtime;

// What one more bank adds beyond graph 0: its cache rows.
static uint64_t axk2_bank_bytes(uint32_t ctx) { return axk2_kv_bytes(ctx); }

// Rows [0, pos) of every layer's caches of `src` into `dst`; exponents too.
static bool axk2_rows_copy(ds4_axk2_graph *dst, const ds4_axk2_graph *src, uint32_t pos) {
    const ds4_axk2_keys format = axk2_key_format();
    const uint64_t key = format == DS4_AXK2_KEYS_BF16 ? (uint64_t)AXK2_IDX_DIM * 2u : AXK2_IDX_DIM;
    for (uint32_t il = 0; pos && il < AXK2_LAYERS; il++) {
        const uint64_t lat = (uint64_t)src->ctx_cap * AXK2_KV_LORA, idx = (uint64_t)src->ctx_cap * AXK2_IDX_DIM;
        if (!ds4_gpu_tensor_copy(dst->latent[il], 0, src->latent[il], 0, (uint64_t)pos * AXK2_KV_LORA) ||
            !ds4_gpu_tensor_copy(dst->latent[il], lat, src->latent[il], lat, pos) ||
            !ds4_gpu_tensor_copy(dst->k_pe[il], 0, src->k_pe[il], 0, (uint64_t)pos * AXK2_ROPE * 2u) ||
            !ds4_gpu_tensor_copy(dst->keys[il], 0, src->keys[il], 0, (uint64_t)pos * key) ||
            (format == DS4_AXK2_KEYS_FP8 && !ds4_gpu_tensor_copy(dst->keys[il], idx, src->keys[il], idx, pos))) {
            return false;
        }
    }
    return true;
}

static void axk2_batch_free(ds4_axk2_batch_runtime *rt) {
    if (!rt) { return; }
    // Banks borrow graph 0's buffers: free them first.
    for (uint32_t b = rt->max_seq; rt->graph && b > 0; b--) { axk2_graph_free(&rt->graph[b - 1u]); }
    free(rt->graph);
    free(rt->bank_logits);
    free(rt->bank_logits_valid);
    free(rt);
}

static ds4_axk2_batch_runtime *axk2_batch_create(ds4_engine *e, uint32_t ctx, uint32_t max_seq, uint32_t cap) {
    if (!e || !ctx || !max_seq || max_seq > AXK2_BANK_MAX || !cap || cap > ctx) { return NULL; }
    ds4_axk2_batch_runtime *rt = xcalloc(1, sizeof(*rt));
    rt->ctx_size = ctx;
    rt->max_seq = max_seq;
    rt->prefill_cap = cap;
    rt->graph = xcalloc(max_seq, sizeof(*rt->graph));
    if (!axk2_graph_alloc(&rt->graph[0], &e->model, &e->weights, ctx, cap)) {
        axk2_batch_free(rt);
        return NULL;
    }
    for (uint32_t b = 1; b < max_seq; b++) {
        if (!axk2_graph_bank(&rt->graph[b], &rt->graph[0])) {
            axk2_batch_free(rt);
            return NULL;
        }
    }
    rt->bank_logits = xcalloc((size_t)max_seq * DS4_N_VOCAB, sizeof(float));
    rt->bank_logits_valid = xcalloc(max_seq, sizeof(uint8_t));
    return rt;
}

static bool axk2_bank_read_logits(ds4_axk2_batch_runtime *rt, uint32_t bank) {
    float *dst = rt->bank_logits + (size_t)bank * DS4_N_VOCAB;
    bool ok = ds4_gpu_tensor_read(rt->graph[bank].logits, 0, dst, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    for (uint32_t i = 0; ok && i < DS4_N_VOCAB; i++) { ok = isfinite(dst[i]); }
    rt->bank_logits_valid[bank] = ok;
    if (!ok) { rt->graph[bank].failed = true; }
    return ok;
}

static void axk2_bank_reset(ds4_axk2_batch_runtime *rt, uint32_t bank) {
    rt->bank_logits_valid[bank] = 0;
    (void)axk2_graph_reset(&rt->graph[bank]);
}

// Exact fork at the source's frontier: its rows and its logits.
static bool axk2_bank_copy(ds4_axk2_batch_runtime *rt, uint32_t src, uint32_t dst, uint32_t tokens) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || src == dst) { return false; }
    ds4_axk2_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (sg->failed || sg->position != tokens || tokens > dg->ctx_cap) { return false; }
    dg->failed = true;
    if (!axk2_rows_copy(dg, sg, tokens) || !ds4_gpu_synchronize()) { return false; }
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
static bool axk2_bank_fork(ds4_axk2_batch_runtime *rt, uint32_t src, uint32_t dst, uint32_t pos) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || !pos) { return false; }
    ds4_axk2_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (sg->failed || pos > sg->position || pos > dg->ctx_cap) { return false; }
    dg->failed = true;
    rt->bank_logits_valid[dst] = 0;
    if (src != dst && (!axk2_rows_copy(dg, sg, pos) || !ds4_gpu_synchronize())) { return false; }
    dg->position = pos;
    dg->failed = false;
    return true;
}

static bool axk2_bank_prefill(ds4_axk2_batch_runtime *rt, ds4_engine *e, uint32_t bank, const int *tokens,
        uint32_t rows, uint32_t pos, axk2_prefill_mode mode) {
    if (!rt || !e || bank >= rt->max_seq) { return false; }
    rt->bank_logits_valid[bank] = 0;
    const axk2_head_mode head = mode == AXK2_FINAL_ROWS ? AXK2_HEAD_LAST : AXK2_HEAD_SKIP;
    if (!axk2_forward(&rt->graph[bank], &e->model, &e->weights, tokens, rows, pos, head)) { return false; }
    return mode == AXK2_MORE_ROWS || axk2_bank_read_logits(rt, bank);
}

/* Each bank attends over its own caches, so decode rows of different banks
 * share no forward: run them in bank order. */
static bool axk2_bank_decode(ds4_axk2_batch_runtime *rt, ds4_engine *e, const uint32_t *banks,
        const int *tokens, const uint32_t *positions, uint32_t rows) {
    if (!rt || !e || !banks || !tokens || !positions || !rows) { return false; }
    for (uint32_t i = 0; i < rows; i++) {
        if (banks[i] >= rt->max_seq ||
            !axk2_bank_prefill(rt, e, banks[i], &tokens[i], 1u, positions[i], AXK2_FINAL_ROWS)) {
            return false;
        }
    }
    return true;
}
