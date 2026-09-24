/* MiMo banks share one scratch graph. Their target KV and frontiers stay
 * separate; the scheduler forwards one bank at a time. Media uses the serial
 * lane until feature spans have a bank identity. */
enum { M2_BANK_MAX = 2, M2_BANK_CHECKPOINTS = 8, M2_CKPT_ALIGN = 4096 };
static const uint64_t M2_MEDIA_WORK_BYTES = 1ull << 30;
typedef enum { M2_ROWS_SAVE, M2_ROWS_RESTORE } mimo2_rows_mode;
typedef enum { M2_NO_LOGITS, M2_HAS_LOGITS } mimo2_logits_mode;
typedef enum { M2_MORE_ROWS, M2_FINAL_ROWS } mimo2_prefill_mode;
typedef enum { M2_STRICT_BANKS, M2_FIT_BANKS } mimo2_fit_mode;

typedef struct {
    uint32_t max_seq, ctx_size, prefill_cap, checkpoint_stride;
    ds4_mimo2_graph *graph;
    float *bank_logits, *checkpoint_logits;
    uint8_t *bank_logits_valid;
    ds4_gpu_tensor *checkpoint_slab;
    ds4_partial_checkpoint checkpoint[M2_BANK_CHECKPOINTS];
    uint64_t checkpoint_slot_bytes, checkpoint_clock;
} ds4_mimo2_batch_runtime;

static uint64_t mimo2_swa_row_bytes(void) {
    return 8ull * (192 + 128) * sizeof(uint16_t);
}

static uint64_t mimo2_swa_slot_bytes(void) {
    uint64_t bytes = 0;
    for (unsigned il = 0; il < MIMO2_LAYERS; il++) {
        if (!mimo2_is_full(il)) { bytes += M2_WINDOW * mimo2_swa_row_bytes(); }
    }
    return bytes;
}

static void mimo2_ckpt_drop(ds4_mimo2_batch_runtime *rt, uint32_t bank) {
    for (unsigned i = 0; i < M2_BANK_CHECKPOINTS; i++) {
        if (partial_checkpoint_ref(&rt->checkpoint[i], bank)) {
            partial_checkpoint_clear_ref(&rt->checkpoint[i], bank);
        }
    }
}

static void mimo2_ckpt_inherit(ds4_mimo2_batch_runtime *rt, uint32_t src,
                               uint32_t dst, uint32_t cut) {
    if (src != dst) { mimo2_ckpt_drop(rt, dst); }
    for (unsigned i = 0; i < M2_BANK_CHECKPOINTS; i++) {
        ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (!partial_checkpoint_ref(cp, src)) { continue; }
        if (cp->pos <= cut) { partial_checkpoint_set_ref(cp, dst); }
        else if (src == dst) { partial_checkpoint_clear_ref(cp, dst); }
    }
}

static int mimo2_ckpt_find(ds4_mimo2_batch_runtime *rt, uint32_t bank,
                           uint32_t cut, uint32_t request_len) {
    int best = -1;
    for (unsigned i = 0; i < M2_BANK_CHECKPOINTS; i++) {
        const ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (!cp->pos || cp->pos > cut || !partial_checkpoint_ref(cp, bank) ||
            (cp->pos == request_len && !cp->logits_valid)) { continue; }
        if (best < 0 || cp->pos > rt->checkpoint[best].pos) { best = (int)i; }
    }
    return best;
}

/* Copy a logical SWA window to/from packed checkpoint rows. A wrapped ring
 * needs two spans; older ring positions are irrelevant to the next token. */
static bool mimo2_ckpt_rows(ds4_mimo2_graph *g, unsigned il,
                            ds4_gpu_tensor *slab, uint64_t base,
                            uint32_t pos, mimo2_rows_mode mode) {
    const uint32_t first = pos > M2_WINDOW ? pos - M2_WINDOW : 0u;
    const uint64_t row = mimo2_swa_row_bytes();
    for (uint32_t p = first; p < pos;) {
        const uint32_t slot = p % g->kv_cap[il];
        uint32_t rows = g->kv_cap[il] - slot;
        if (rows > pos - p) { rows = pos - p; }
        const uint64_t packed = base + (uint64_t)(p - first) * row;
        const uint64_t ring = (uint64_t)slot * row;
        const bool ok = mode == M2_ROWS_RESTORE
            ? ds4_gpu_tensor_copy(g->kv[il], ring, slab, packed,
                                  (uint64_t)rows * row)
            : ds4_gpu_tensor_copy(slab, packed, g->kv[il], ring,
                                  (uint64_t)rows * row);
        if (!ok) { return false; }
        p += rows;
    }
    return true;
}

static bool mimo2_ckpt_capture(ds4_mimo2_batch_runtime *rt, uint32_t bank,
                               uint32_t pos, mimo2_logits_mode logits, uint64_t reserve) {
    if (!rt || !rt->checkpoint_slab || bank >= rt->max_seq || !pos ||
        rt->graph[bank].failed || rt->graph[bank].position != pos) { return false; }
    uint32_t slot = 0;
    for (uint32_t i = 0; i < M2_BANK_CHECKPOINTS; i++) {
        ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (cp->pos == pos && partial_checkpoint_ref(cp, bank)) {
            if (logits == M2_HAS_LOGITS && rt->bank_logits_valid[bank]) {
                memcpy(rt->checkpoint_logits + (size_t)i * DS4_N_VOCAB,
                       rt->bank_logits + (size_t)bank * DS4_N_VOCAB,
                       DS4_N_VOCAB * sizeof(float));
                cp->logits_valid = 1;
            }
            cp->last_use = ++rt->checkpoint_clock;
            return true;
        }
        if (!cp->pos || (rt->checkpoint[slot].pos &&
                         cp->last_use < rt->checkpoint[slot].last_use)) { slot = i; }
    }
    const uint64_t base = (uint64_t)slot * rt->checkpoint_slot_bytes;
    const uint64_t need = batch_span_need(rt->checkpoint_slab, base,
                                          rt->checkpoint_slot_bytes);
    if (need > ds4_mem_usable_beyond(reserve) ||
        !ds4_gpu_tensor_ensure(rt->checkpoint_slab, base,
                               rt->checkpoint_slot_bytes)) { return false; }
    ds4_partial_checkpoint *cp = &rt->checkpoint[slot];
    memset(cp, 0, sizeof(*cp));
    uint64_t offset = base;
    for (unsigned il = 0; il < MIMO2_LAYERS; il++) {
        if (mimo2_is_full(il)) { continue; }
        if (!mimo2_ckpt_rows(&rt->graph[bank], il, rt->checkpoint_slab,
                             offset, pos, M2_ROWS_SAVE)) { return false; }
        offset += M2_WINDOW * mimo2_swa_row_bytes();
    }
    if (!ds4_gpu_synchronize()) { return false; }
    cp->pos = pos;
    cp->last_use = ++rt->checkpoint_clock;
    partial_checkpoint_set_ref(cp, bank);
    if (logits == M2_HAS_LOGITS && rt->bank_logits_valid[bank]) {
        memcpy(rt->checkpoint_logits + (size_t)slot * DS4_N_VOCAB,
               rt->bank_logits + (size_t)bank * DS4_N_VOCAB,
               DS4_N_VOCAB * sizeof(float));
        cp->logits_valid = 1;
    }
    return true;
}

static bool mimo2_ckpt_restore(ds4_mimo2_batch_runtime *rt, uint32_t src,
                               uint32_t dst, uint32_t slot, uint32_t cut,
                               uint32_t *out) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq ||
        slot >= M2_BANK_CHECKPOINTS) { return false; }
    ds4_partial_checkpoint *cp = &rt->checkpoint[slot];
    const uint32_t pos = cp->pos;
    ds4_mimo2_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (!pos || pos > cut || pos > sg->position || sg->failed ||
        !partial_checkpoint_ref(cp, src)) { return false; }
    dg->failed = true;
    rt->bank_logits_valid[dst] = 0;
    uint64_t offset = (uint64_t)slot * rt->checkpoint_slot_bytes;
    for (unsigned il = 0; il < MIMO2_LAYERS; il++) {
        if (mimo2_is_full(il)) {
            const uint64_t row = 4ull * (192 + 128) * sizeof(uint16_t);
            if (src != dst && !ds4_gpu_tensor_copy(dg->kv[il], 0,
                    sg->kv[il], 0, (uint64_t)pos * row)) { return false; }
            continue;
        }
        if (!mimo2_ckpt_rows(dg, il, rt->checkpoint_slab,
                             offset, pos, M2_ROWS_RESTORE)) { return false; }
        offset += M2_WINDOW * mimo2_swa_row_bytes();
    }
    if (!ds4_gpu_synchronize()) { return false; }
    dg->position = pos;
    dg->failed = false;
    dg->hidden_valid = false;
    dg->span_n = 0;
    dg->media_tag = 0;
    dg->checkpoint_tag = 0;
    if (cp->logits_valid) {
        memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB,
               rt->checkpoint_logits + (size_t)slot * DS4_N_VOCAB,
               DS4_N_VOCAB * sizeof(float));
        rt->bank_logits_valid[dst] = 1;
    }
    cp->last_use = ++rt->checkpoint_clock;
    mimo2_ckpt_inherit(rt, src, dst, pos);
    if (out) { *out = pos; }
    return true;
}

static uint64_t mimo2_ckpt_trim(ds4_mimo2_batch_runtime *rt, uint64_t want) {
    uint64_t freed = 0;
    if (!rt || !rt->checkpoint_slab || !want || !ds4_gpu_synchronize()) { return 0; }
    for (uint32_t i = 0; i < M2_BANK_CHECKPOINTS && freed < want;) {
        if (rt->checkpoint[i].pos) { i++; continue; }
        const uint64_t start = (uint64_t)i * rt->checkpoint_slot_bytes;
        do { i++; } while (i < M2_BANK_CHECKPOINTS && !rt->checkpoint[i].pos);
        const uint64_t end = i == M2_BANK_CHECKPOINTS
            ? ds4_gpu_tensor_bytes(rt->checkpoint_slab)
            : (uint64_t)i * rt->checkpoint_slot_bytes;
        freed += ds4_gpu_tensor_trim(rt->checkpoint_slab, start, end - start);
    }
    return freed;
}

static void mimo2_batch_free(ds4_mimo2_batch_runtime *rt) {
    if (!rt) { return; }
    if (rt->graph) {
        for (uint32_t b = rt->max_seq; b > 0; b--) {
            mimo2_graph_free(&rt->graph[b - 1]);
        }
    }
    free(rt->graph);
    free(rt->bank_logits);
    free(rt->bank_logits_valid);
    ds4_gpu_tensor_free(rt->checkpoint_slab);
    free(rt->checkpoint_logits);
    free(rt);
}

static ds4_mimo2_batch_runtime *mimo2_batch_create(
        uint32_t ctx, uint32_t max_seq, uint32_t cap) {
    if (!ctx || max_seq < 1 || max_seq > M2_BANK_MAX ||
        !cap || cap > ctx) { return NULL; }
    ds4_mimo2_batch_runtime *rt = xcalloc(1, sizeof(*rt));
    rt->ctx_size = ctx;
    rt->max_seq = max_seq;
    rt->prefill_cap = cap;
    rt->graph = xcalloc(max_seq, sizeof(*rt->graph));
    if (!mimo2_graph_alloc(&rt->graph[0], ctx, cap, false)) {
        mimo2_batch_free(rt);
        return NULL;
    }
    for (uint32_t b = 1; b < max_seq; b++) {
        if (!mimo2_graph_clone_bank(&rt->graph[b], &rt->graph[0])) {
            mimo2_batch_free(rt);
            return NULL;
        }
    }
    rt->bank_logits = xcalloc((size_t)max_seq * DS4_N_VOCAB, sizeof(float));
    rt->bank_logits_valid = xcalloc(max_seq, sizeof(uint8_t));
    if (getenv("DS4_SERVER_FORK_PARTIAL") &&
        strcmp(getenv("DS4_SERVER_FORK_PARTIAL"), "0") == 0) { return rt; }
    rt->checkpoint_slot_bytes = mimo2_swa_slot_bytes();
    uint64_t bytes = rt->checkpoint_slot_bytes * M2_BANK_CHECKPOINTS;
    const uint64_t page = ds4_gpu_vmm_demand_page();
    if (page) { bytes = (bytes + page - 1u) / page * page; }
    rt->checkpoint_slab = ds4_gpu_tensor_reserve(bytes);
    if (rt->checkpoint_slab) {
        rt->checkpoint_logits = xcalloc(
            (size_t)M2_BANK_CHECKPOINTS * DS4_N_VOCAB, sizeof(float));
        const uint32_t stride = (ctx + DS4_PARTIAL_PERIODIC_TARGET - 1u) /
                                DS4_PARTIAL_PERIODIC_TARGET;
        rt->checkpoint_stride =
            ((stride + M2_CKPT_ALIGN - 1u) / M2_CKPT_ALIGN) * M2_CKPT_ALIGN;
    }
    return rt;
}

static bool mimo2_bank_read_logits(ds4_mimo2_batch_runtime *rt, uint32_t bank) {
    if (!rt || bank >= rt->max_seq || !ds4_gpu_synchronize()) { return false; }
    float *dst = rt->bank_logits + (size_t)bank * DS4_N_VOCAB;
    const bool ok = ds4_gpu_tensor_read(rt->graph[bank].logits, 0, dst,
                                        (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    rt->bank_logits_valid[bank] = ok ? 1u : 0u;
    return ok;
}

static void mimo2_bank_reset(ds4_mimo2_batch_runtime *rt, uint32_t bank) {
    mimo2_ckpt_drop(rt, bank);
    ds4_mimo2_graph *g = &rt->graph[bank];
    g->position = 0;
    g->failed = false;
    g->hidden_valid = false;
    g->trial_n = 0;
    g->span_n = 0;
    g->media_tag = 0;
    g->checkpoint_tag = 0;
    rt->bank_logits_valid[bank] = 0;
}

static bool mimo2_bank_copy(ds4_mimo2_batch_runtime *rt, uint32_t src,
                            uint32_t dst, uint32_t tokens) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || src == dst) { return false; }
    ds4_mimo2_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (sg->failed || sg->position != tokens || tokens > dg->context) { return false; }
    dg->failed = true;
    for (unsigned il = 0; il < MIMO2_LAYERS; il++) {
        const uint64_t row = (mimo2_is_full(il) ? 4ull : 8ull) *
                             (192 + 128) * sizeof(uint16_t);
        if (mimo2_is_full(il)) {
            if (!ds4_gpu_tensor_copy(dg->kv[il], 0, sg->kv[il], 0,
                                     (uint64_t)tokens * row)) { return false; }
            continue;
        }
        const uint32_t first = tokens > M2_WINDOW ? tokens - M2_WINDOW : 0u;
        for (uint32_t p = first; p < tokens;) {
            const uint32_t slot = p % sg->kv_cap[il];
            uint32_t rows = sg->kv_cap[il] - slot;
            if (rows > tokens - p) { rows = tokens - p; }
            if (!ds4_gpu_tensor_copy(dg->kv[il], (uint64_t)slot * row,
                                     sg->kv[il], (uint64_t)slot * row,
                                     (uint64_t)rows * row)) { return false; }
            p += rows;
        }
    }
    if (!ds4_gpu_synchronize()) { return false; }
    dg->position = tokens;
    dg->failed = false;
    dg->hidden_valid = false;
    dg->span_n = 0;
    dg->media_tag = 0;
    dg->checkpoint_tag = 0;
    memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB,
           rt->bank_logits + (size_t)src * DS4_N_VOCAB,
           DS4_N_VOCAB * sizeof(float));
    rt->bank_logits_valid[dst] = rt->bank_logits_valid[src];
    mimo2_ckpt_inherit(rt, src, dst, tokens);
    return true;
}

static bool mimo2_bank_prefill(ds4_mimo2_batch_runtime *rt, ds4_engine *e,
                               uint32_t bank, const int *tokens, uint32_t rows,
                               uint32_t pos, mimo2_prefill_mode mode) {
    if (!rt || !e || bank >= rt->max_seq) { return false; }
    rt->bank_logits_valid[bank] = 0;
    if (!mimo2_forward(&rt->graph[bank], &e->model, &e->weights,
                       tokens, rows, pos)) { return false; }
    return mode == M2_MORE_ROWS || mimo2_bank_read_logits(rt, bank);
}

static bool mimo2_bank_decode(ds4_mimo2_batch_runtime *rt, ds4_engine *e,
                              const uint32_t *banks, const int *tokens,
                              const uint32_t *positions, uint32_t rows) {
    if (!rt || !e || !banks || !tokens || !positions || !rows) { return false; }
    for (uint32_t i = 0; i < rows; i++) {
        const uint32_t bank = banks[i];
        if (bank >= rt->max_seq || !mimo2_bank_prefill(rt, e, bank,
                &tokens[i], 1u, positions[i], M2_FINAL_ROWS)) { return false; }
    }
    return true;
}
