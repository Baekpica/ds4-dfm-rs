/* Gemma-4 banks. The scheduler serializes bank forwards under the engine
 * generation lock, so graph 0's GVQ workspace, expert scratch and working
 * rows serve every bank; each bank owns its KV caches (both layouts: the
 * sliding ring and the full context) and its host logits.
 *
 * Every cache row is indexed by position -- a sliding layer writes slot
 * position % rows, a full layer slot position -- and the forward writes a
 * row before any query at or below it reads it. Copying the first
 * min(pos, rows) rows of each cache copies the slots the positions [0, pos)
 * occupy, but a ring slot keeps the newest position written to it: a fork
 * more than rows - G4_WINDOW below the source's high-water mark would read
 * rows a later branch overwrote. A fork (or an in-place rewind) is taken
 * only where the ring still holds the window (gemma4_bank_keeps); deeper
 * reuse prefills from zero, so this file needs no checkpoint slab. */

enum { G4_BANK_MAX = DS4_MULTISEQ_MAX_SEQ };
typedef enum { G4_MORE_ROWS, G4_FINAL_ROWS } gemma4_prefill_mode;
typedef enum { G4_STRICT_BANKS, G4_FIT_BANKS } gemma4_fit_mode;

typedef struct {
    uint32_t max_seq, ctx_size, prefill_cap;
    ds4_gemma4_graph *graph;
    float *bank_logits;
    uint8_t *bank_logits_valid;
} ds4_gemma4_batch_runtime;

/* What one more bank adds beyond graph 0: its KV caches. */
static uint64_t gemma4_bank_bytes(uint32_t ctx, uint32_t cap) {
    return ds4_gemma4_kv_bytes(ctx, cap);
}

static uint64_t gemma4_bank_row_bytes(unsigned il) {
    return (uint64_t)gemma4_kv_heads(il) * gemma4_head_dim(il) * sizeof(uint16_t);
}

/* Slots [0, min(pos, rows)) of every layer of `src` into `dst`. */
static bool gemma4_rows_copy(ds4_gemma4_graph *dst, const ds4_gemma4_graph *src,
        uint32_t pos) {
    for (uint32_t il = 0; pos && il < G4_LAYERS; il++) {
        const uint32_t rows = ds4_gemma4_kv_rows(il, src->ctx, src->cap);
        const uint32_t n = pos < rows ? pos : rows;
        const uint64_t bytes = (uint64_t)n * gemma4_bank_row_bytes(il);
        if (!n) { continue; }
        if (!ds4_gpu_tensor_copy(dst->kcache[il], 0, src->kcache[il], 0, bytes) ||
            !ds4_gpu_tensor_copy(dst->vcache[il], 0, src->vcache[il], 0, bytes)) {
            return false;
        }
    }
    return true;
}

static void gemma4_batch_free(ds4_gemma4_batch_runtime *rt) {
    if (!rt) { return; }
    /* Banks borrow graph 0's buffers: clear them first. */
    for (uint32_t b = rt->max_seq; rt->graph && b > 0; b--) {
        gemma4_graph_clear(&rt->graph[b - 1u]);
    }
    free(rt->graph);
    free(rt->bank_logits);
    free(rt->bank_logits_valid);
    free(rt);
}

static ds4_gemma4_batch_runtime *gemma4_batch_create(ds4_engine *e, uint32_t ctx,
        uint32_t max_seq, uint32_t cap) {
    if (!e || !ctx || !max_seq || max_seq > G4_BANK_MAX || !cap || cap > ctx) { return NULL; }
    ds4_gemma4_batch_runtime *rt = xcalloc(1, sizeof(*rt));
    rt->ctx_size = ctx;
    rt->max_seq = max_seq;
    rt->prefill_cap = cap;
    rt->graph = xcalloc(max_seq, sizeof(*rt->graph));
    ds4_gemma4_graph *owner = gemma4_graph_create(&e->model, &e->weights, ctx, cap);
    if (!owner) {
        gemma4_batch_free(rt);
        return NULL;
    }
    rt->graph[0] = *owner;
    free(owner);
    for (uint32_t b = 1; b < max_seq; b++) {
        if (!gemma4_graph_bank(&rt->graph[b], &rt->graph[0])) {
            gemma4_batch_free(rt);
            return NULL;
        }
    }
    rt->bank_logits = xcalloc((size_t)max_seq * DS4_N_VOCAB, sizeof(float));
    rt->bank_logits_valid = xcalloc(max_seq, sizeof(uint8_t));
    return rt;
}

static bool gemma4_bank_read_logits(ds4_gemma4_batch_runtime *rt, uint32_t bank) {
    float *dst = rt->bank_logits + (size_t)bank * DS4_N_VOCAB;
    bool ok = ds4_gpu_tensor_read(rt->graph[bank].logits, 0, dst,
        (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    for (uint32_t i = 0; ok && i < DS4_N_VOCAB; i++) { ok = isfinite(dst[i]); }
    rt->bank_logits_valid[bank] = ok;
    if (!ok) { rt->graph[bank].failed = true; }
    return ok;
}

static void gemma4_bank_reset(ds4_gemma4_batch_runtime *rt, uint32_t bank) {
    rt->bank_logits_valid[bank] = 0;
    (void)gemma4_graph_reset(&rt->graph[bank]);
}

/* Exact fork at the source's frontier: its rows and its logits. */
static bool gemma4_bank_copy(ds4_gemma4_batch_runtime *rt, uint32_t src,
        uint32_t dst, uint32_t tokens) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || src == dst) { return false; }
    ds4_gemma4_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (sg->failed || sg->position != tokens || tokens > dg->ctx) { return false; }
    dg->failed = true;
    if (!gemma4_rows_copy(dg, sg, tokens) || !ds4_gpu_synchronize()) { return false; }
    dg->position = tokens;
    dg->high = sg->high;
    dg->restored = sg->restored;
    dg->failed = false;
    memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB,
           rt->bank_logits + (size_t)src * DS4_N_VOCAB, DS4_N_VOCAB * sizeof(float));
    rt->bank_logits_valid[dst] = rt->bank_logits_valid[src];
    return true;
}

/* Whether bank `src` can fork at `pos` (gemma4_graph_keeps). */
static bool gemma4_bank_keeps(const ds4_gemma4_batch_runtime *rt, uint32_t src,
        uint32_t pos) {
    return rt && src < rt->max_seq && gemma4_graph_keeps(&rt->graph[src], pos);
}

/* Partial fork below the source's frontier: the prefix [0, pos) (in place
 * when src == dst). The rows past it are rewritten before any read; the
 * logits at pos are unknown, so the caller prefills at least one row. */
static bool gemma4_bank_fork(ds4_gemma4_batch_runtime *rt, uint32_t src,
        uint32_t dst, uint32_t pos) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || !pos) { return false; }
    ds4_gemma4_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (sg->failed || pos > sg->position || pos > dg->ctx || !gemma4_graph_keeps(sg, pos)) {
        return false;
    }
    dg->failed = true;
    rt->bank_logits_valid[dst] = 0;
    if (src != dst && (!gemma4_rows_copy(dg, sg, pos) || !ds4_gpu_synchronize())) { return false; }
    /* The copied ring carries the source's rows, so it keeps its marks. */
    dg->high = sg->high;
    dg->restored = sg->restored;
    dg->position = pos;
    dg->failed = false;
    return true;
}

static bool gemma4_bank_prefill(ds4_gemma4_batch_runtime *rt, ds4_engine *e,
        uint32_t bank, const int *tokens, uint32_t rows, uint32_t pos,
        gemma4_prefill_mode mode) {
    if (!rt || !e || bank >= rt->max_seq) { return false; }
    rt->bank_logits_valid[bank] = 0;
    if (!gemma4_forward(&rt->graph[bank], &e->model, &e->weights, (const uint32_t *)tokens,
            rows, pos, mode == G4_FINAL_ROWS)) {
        rt->graph[bank].failed = true;
        return false;
    }
    return mode == G4_MORE_ROWS || gemma4_bank_read_logits(rt, bank);
}

/* Each bank attends over its own caches, so decode rows of different banks
 * share no forward: run them in bank order. */
static bool gemma4_bank_decode(ds4_gemma4_batch_runtime *rt, ds4_engine *e,
        const uint32_t *banks, const int *tokens, const uint32_t *positions,
        uint32_t rows) {
    if (!rt || !e || !banks || !tokens || !positions || !rows) { return false; }
    for (uint32_t i = 0; i < rows; i++) {
        if (banks[i] >= rt->max_seq ||
            !gemma4_bank_prefill(rt, e, banks[i], &tokens[i], 1u, positions[i],
                G4_FINAL_ROWS)) {
            return false;
        }
    }
    return true;
}
