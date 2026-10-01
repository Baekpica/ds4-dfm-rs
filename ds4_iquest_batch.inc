/* Banks share projection scratch and own independent Q8 target/MTP rings.
 * Partial checkpoints retain displaced SWA rows and the target hidden carry. */
enum { IQ_BANK_MAX = DS4_MULTISEQ_MAX_SEQ, IQ_BANK_CHECKPOINTS = 8, IQ_CKPT_ALIGN = 4096 };
typedef enum { IQ_ROWS_SAVE, IQ_ROWS_RESTORE } iquest_rows_mode;
typedef enum { IQ_MORE_ROWS, IQ_FINAL_ROWS } iquest_prefill_mode;
typedef enum { IQ_STRICT_BANKS, IQ_FIT_BANKS } iquest_fit_mode;
typedef enum { IQ_REUSE_FULL, IQ_REUSE_PARTIAL } iquest_reuse_mode;

typedef struct {
    uint32_t max_seq, ctx_size, prefill_cap, checkpoint_stride;
    ds4_iquest_graph *graph;
    float *bank_logits, *checkpoint_logits;
    uint8_t *bank_logits_valid;
    ds4_gpu_tensor *checkpoint_slab;
    ds4_partial_checkpoint checkpoint[IQ_BANK_CHECKPOINTS];
    uint64_t checkpoint_slot_bytes, checkpoint_clock;
} ds4_iquest_batch_runtime;

/* Emit only after synchronized reuse succeeds. Frontiers distinguish a
 * source-preserving fork from an in-place partial rewind in live gates. */
static void iquest_reuse_trace(const ds4_iquest_batch_runtime *rt, uint32_t src,
                               uint32_t dst, uint32_t cached, uint32_t before,
                               iquest_reuse_mode mode) {
    const char *enabled = getenv("DS4_IQUEST_BATCH_TRACE");
    if (!enabled || strcmp(enabled, "1") != 0) { return; }
    fprintf(stderr,
            "ds4: IQuest-Q1 bank reuse source=%u target=%u cached=%u "
            "partial=%u source_before=%u source_after=%u target_after=%u\n",
            src, dst, cached, mode == IQ_REUSE_PARTIAL ? 1u : 0u, before,
            rt->graph[src].position, rt->graph[dst].position);
}

static uint64_t iquest_swa_row_bytes(void) {
    return IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8);
}

static uint64_t iquest_swa_slot_bytes(void) {
    uint64_t bytes = 0;
    for (unsigned il = 0; il < IQ_LAYERS; il++) {
        if (!iquest_full(il)) { bytes += IQ_WINDOW * iquest_swa_row_bytes(); }
    }
    return bytes;
}

static uint64_t iquest_mtp_slot_bytes(void) {
    return (uint64_t)IQ_DRAFT_WINDOW * iquest_swa_row_bytes() + IQ_EMBED * sizeof(float);
}

static bool iquest_mtp_ckpt(ds4_iquest_graph *g, ds4_gpu_tensor *slab,
        uint64_t base, uint32_t pos, iquest_rows_mode mode) {
    if (!pos) { return false; }
    const uint64_t hidden_bytes = IQ_EMBED * sizeof(float);
    bool ok = mode == IQ_ROWS_SAVE
        ? ds4_gpu_tensor_copy(slab, base, g->carry, 0, hidden_bytes)
        : ds4_gpu_tensor_copy(g->carry, 0, slab, base, hidden_bytes);
    if (g->mtp_enabled) {
        const uint32_t end = pos - 1, first = end > IQ_DRAFT_WINDOW ? end - IQ_DRAFT_WINDOW : 0;
        const uint64_t row = iquest_swa_row_bytes();
        const unsigned cap = IQ_DRAFT_WINDOW + IQ_DRAFT_SLOTS;
        for (unsigned p = first; ok && p < end;) {
            const unsigned slot = p % cap;
            unsigned rows = cap - slot;
            if (rows > end - p) { rows = end - p; }
            const uint64_t packed = base + hidden_bytes + (uint64_t)(p - first) * row;
            ok = mode == IQ_ROWS_SAVE
                ? ds4_gpu_tensor_copy(slab, packed, g->mtp_kv, (uint64_t)slot * row, rows * row)
                : ds4_gpu_tensor_copy(g->mtp_kv, (uint64_t)slot * row, slab, packed, rows * row);
            p += rows;
        }
    }
    if (mode == IQ_ROWS_RESTORE) { g->mtp_position = g->mtp_enabled ? pos - 1 : 0; g->trial_n = 0; }
    return ok;
}

static void iquest_ckpt_drop(ds4_iquest_batch_runtime *rt, uint32_t bank) {
    for (unsigned i = 0; i < IQ_BANK_CHECKPOINTS; i++) {
        if (partial_checkpoint_ref(&rt->checkpoint[i], bank)) {
            partial_checkpoint_clear_ref(&rt->checkpoint[i], bank);
        }
    }
}

static void iquest_ckpt_inherit(ds4_iquest_batch_runtime *rt, uint32_t src,
                               uint32_t dst, uint32_t cut) {
    if (src != dst) { iquest_ckpt_drop(rt, dst); }
    for (unsigned i = 0; i < IQ_BANK_CHECKPOINTS; i++) {
        ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (!partial_checkpoint_ref(cp, src)) { continue; }
        if (cp->pos <= cut) { partial_checkpoint_set_ref(cp, dst); }
        else if (src == dst) { partial_checkpoint_clear_ref(cp, dst); }
    }
}

static int iquest_ckpt_find(ds4_iquest_batch_runtime *rt, uint32_t bank,
                           uint32_t cut, uint32_t request_len) {
    int best = -1;
    for (unsigned i = 0; i < IQ_BANK_CHECKPOINTS; i++) {
        const ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (!cp->pos || cp->pos > cut || !partial_checkpoint_ref(cp, bank) ||
            (cp->pos == request_len && !cp->logits_valid)) { continue; }
        if (best < 0 || cp->pos > rt->checkpoint[best].pos) { best = (int)i; }
    }
    return best;
}

/* Copy a logical SWA window to/from packed checkpoint rows. A wrapped ring
 * needs two spans; older ring positions are irrelevant to the next token. */
static bool iquest_ckpt_rows(ds4_iquest_graph *g, unsigned il,
                            ds4_gpu_tensor *slab, uint64_t base,
                            uint32_t pos, iquest_rows_mode mode) {
    const uint32_t first = pos > IQ_WINDOW ? pos - IQ_WINDOW : 0u;
    const uint64_t row = iquest_swa_row_bytes();
    for (uint32_t p = first; p < pos;) {
        const uint32_t slot = p % g->kv_cap[il];
        uint32_t rows = g->kv_cap[il] - slot;
        if (rows > pos - p) { rows = pos - p; }
        const uint64_t packed = base + (uint64_t)(p - first) * row;
        const uint64_t ring = (uint64_t)slot * row;
        const bool ok = mode == IQ_ROWS_RESTORE
            ? ds4_gpu_tensor_copy(g->kv[il], ring, slab, packed,
                                  (uint64_t)rows * row)
            : ds4_gpu_tensor_copy(slab, packed, g->kv[il], ring,
                                  (uint64_t)rows * row);
        if (!ok) { return false; }
        p += rows;
    }
    return true;
}

static bool iquest_ckpt_capture(ds4_iquest_batch_runtime *rt, uint32_t bank,
                               uint32_t pos, iquest_logits_mode logits, uint64_t reserve) {
    if (!rt || !rt->checkpoint_slab || bank >= rt->max_seq || !pos ||
        rt->graph[bank].failed || rt->graph[bank].position != pos ||
        rt->graph[bank].trial_n) { return false; }
    uint32_t slot = 0;
    for (uint32_t i = 0; i < IQ_BANK_CHECKPOINTS; i++) {
        ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (cp->pos == pos && partial_checkpoint_ref(cp, bank)) {
            if (logits == IQ_HAS_LOGITS && rt->bank_logits_valid[bank]) {
                memcpy(rt->checkpoint_logits + (size_t)i * IQ_VOCAB,
                       rt->bank_logits + (size_t)bank * IQ_VOCAB,
                       IQ_VOCAB * sizeof(float));
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
    for (unsigned il = 0; il < IQ_LAYERS; il++) {
        if (iquest_full(il)) { continue; }
        if (!iquest_ckpt_rows(&rt->graph[bank], il, rt->checkpoint_slab,
                             offset, pos, IQ_ROWS_SAVE)) { return false; }
        offset += IQ_WINDOW * iquest_swa_row_bytes();
    }
    if (!iquest_mtp_ckpt(&rt->graph[bank], rt->checkpoint_slab, offset, pos, IQ_ROWS_SAVE)) { return false; }
    if (!ds4_gpu_synchronize()) { return false; }
    cp->pos = pos;
    cp->last_use = ++rt->checkpoint_clock;
    partial_checkpoint_set_ref(cp, bank);
    if (logits == IQ_HAS_LOGITS && rt->bank_logits_valid[bank]) {
        memcpy(rt->checkpoint_logits + (size_t)slot * IQ_VOCAB,
               rt->bank_logits + (size_t)bank * IQ_VOCAB,
               IQ_VOCAB * sizeof(float));
        cp->logits_valid = 1;
    }
    return true;
}

/* Full-attention histories move as one immutable prefix. */
static bool iquest_history_copy(ds4_iquest_graph *dst, const ds4_iquest_graph *src,
                                 unsigned il, unsigned n) {
    if (dst == src) { return true; }
    if (!iquest_full(il) || n > src->context || n > dst->context) { return false; }
    return ds4_gpu_tensor_copy(dst->kv[il], 0, src->kv[il], 0, (uint64_t)n * iquest_swa_row_bytes());
}

static bool iquest_ckpt_restore(ds4_iquest_batch_runtime *rt, uint32_t src,
                               uint32_t dst, uint32_t slot, uint32_t cut,
                               uint32_t *out) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq ||
        slot >= IQ_BANK_CHECKPOINTS) { return false; }
    ds4_partial_checkpoint *cp = &rt->checkpoint[slot];
    const uint32_t pos = cp->pos;
    ds4_iquest_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (!pos || pos > cut || pos > sg->position || sg->failed ||
        !partial_checkpoint_ref(cp, src)) { return false; }
    const uint32_t source_before = sg->position;
    dg->failed = true;
    rt->bank_logits_valid[dst] = 0;
    uint64_t offset = (uint64_t)slot * rt->checkpoint_slot_bytes;
    for (unsigned il = 0; il < IQ_LAYERS; il++) {
        if (iquest_full(il)) {
            if (!iquest_history_copy(dg, sg, il, pos)) { return false; }
            continue;
        }
        if (!iquest_ckpt_rows(dg, il, rt->checkpoint_slab,
                             offset, pos, IQ_ROWS_RESTORE)) { return false; }
        offset += IQ_WINDOW * iquest_swa_row_bytes();
    }
    if (!iquest_mtp_ckpt(dg, rt->checkpoint_slab, offset, pos, IQ_ROWS_RESTORE)) { return false; }
    if (!ds4_gpu_synchronize()) { return false; }
    dg->position = pos;
    dg->failed = false;
    if (cp->logits_valid) {
        memcpy(rt->bank_logits + (size_t)dst * IQ_VOCAB,
               rt->checkpoint_logits + (size_t)slot * IQ_VOCAB,
               IQ_VOCAB * sizeof(float));
        rt->bank_logits_valid[dst] = 1;
    }
    cp->last_use = ++rt->checkpoint_clock;
    iquest_ckpt_inherit(rt, src, dst, pos);
    if (out) { *out = pos; }
    iquest_reuse_trace(rt, src, dst, pos, source_before, IQ_REUSE_PARTIAL);
    return true;
}

static uint64_t iquest_ckpt_trim(ds4_iquest_batch_runtime *rt, uint64_t want) {
    uint64_t freed = 0;
    if (!rt || !rt->checkpoint_slab || !want || !ds4_gpu_synchronize()) { return 0; }
    for (uint32_t i = 0; i < IQ_BANK_CHECKPOINTS && freed < want;) {
        if (rt->checkpoint[i].pos) { i++; continue; }
        const uint64_t start = (uint64_t)i * rt->checkpoint_slot_bytes;
        do { i++; } while (i < IQ_BANK_CHECKPOINTS && !rt->checkpoint[i].pos);
        const uint64_t end = i == IQ_BANK_CHECKPOINTS
            ? ds4_gpu_tensor_bytes(rt->checkpoint_slab)
            : (uint64_t)i * rt->checkpoint_slot_bytes;
        freed += ds4_gpu_tensor_trim(rt->checkpoint_slab, start, end - start);
    }
    return freed;
}

static void iquest_batch_free(ds4_iquest_batch_runtime *rt) {
    if (!rt) { return; }
    if (rt->graph) {
        for (uint32_t b = rt->max_seq; b > 0; b--) {
            iquest_graph_free(&rt->graph[b - 1]);
        }
    }
    free(rt->graph);
    free(rt->bank_logits);
    free(rt->bank_logits_valid);
    ds4_gpu_tensor_free(rt->checkpoint_slab);
    free(rt->checkpoint_logits);
    free(rt);
}

static ds4_iquest_batch_runtime *iquest_batch_create(
        uint32_t ctx, uint32_t max_seq, uint32_t cap, iquest_mtp_mode mtp) {
    if (!ctx || max_seq < 1 || max_seq > IQ_BANK_MAX ||
        !cap || cap > ctx) { return NULL; }
    ds4_iquest_batch_runtime *rt = xcalloc(1, sizeof(*rt));
    rt->ctx_size = ctx;
    rt->max_seq = max_seq;
    rt->prefill_cap = cap;
    rt->graph = xcalloc(max_seq, sizeof(*rt->graph));
    if (!iquest_graph_alloc(&rt->graph[0], ctx, cap, mtp)) {
        iquest_batch_free(rt);
        return NULL;
    }
    for (uint32_t b = 1; b < max_seq; b++) {
        if (!iquest_graph_clone_bank(&rt->graph[b], &rt->graph[0])) {
            iquest_batch_free(rt);
            return NULL;
        }
    }
    rt->bank_logits = xcalloc((size_t)max_seq * IQ_VOCAB, sizeof(float));
    rt->bank_logits_valid = xcalloc(max_seq, sizeof(uint8_t));
    if (getenv("DS4_SERVER_FORK_PARTIAL") &&
        strcmp(getenv("DS4_SERVER_FORK_PARTIAL"), "0") == 0) { return rt; }
    rt->checkpoint_slot_bytes = iquest_swa_slot_bytes() + iquest_mtp_slot_bytes();
    uint64_t bytes = rt->checkpoint_slot_bytes * IQ_BANK_CHECKPOINTS;
    const uint64_t page = ds4_gpu_vmm_demand_page();
    if (page) { bytes = (bytes + page - 1u) / page * page; }
    rt->checkpoint_slab = ds4_gpu_tensor_reserve(bytes);
    if (rt->checkpoint_slab) {
        rt->checkpoint_logits = xcalloc(
            (size_t)IQ_BANK_CHECKPOINTS * IQ_VOCAB, sizeof(float));
        const uint32_t stride = (ctx + DS4_PARTIAL_PERIODIC_TARGET - 1u) /
                                DS4_PARTIAL_PERIODIC_TARGET;
        rt->checkpoint_stride =
            ((stride + IQ_CKPT_ALIGN - 1u) / IQ_CKPT_ALIGN) * IQ_CKPT_ALIGN;
    }
    return rt;
}

static bool iquest_bank_read_logits(ds4_iquest_batch_runtime *rt, uint32_t bank) {
    if (!rt || bank >= rt->max_seq || !ds4_gpu_synchronize()) { return false; }
    float *dst = rt->bank_logits + (size_t)bank * IQ_VOCAB;
    bool ok = ds4_gpu_tensor_read(rt->graph[bank].logits, 0, dst,
                                  (uint64_t)IQ_VOCAB * sizeof(float)) != 0;
    for (unsigned i = 0; ok && i < IQ_VOCAB; i++) { ok = isfinite(dst[i]); }
    rt->bank_logits_valid[bank] = ok;
    if (!ok) { rt->graph[bank].failed = true; }
    return ok;
}

static void iquest_bank_reset(ds4_iquest_batch_runtime *rt, uint32_t bank) {
    iquest_ckpt_drop(rt, bank);
    ds4_iquest_graph *g = &rt->graph[bank];
    if (g->failed && !ds4_gpu_synchronize()) { return; }
    g->position = 0;
    g->failed = false;
    g->mtp_position = 0; g->trial_n = 0;
    rt->bank_logits_valid[bank] = 0;
}

static bool iquest_bank_copy(ds4_iquest_batch_runtime *rt, uint32_t src,
                            uint32_t dst, uint32_t tokens) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || src == dst) { return false; }
    ds4_iquest_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (sg->failed || sg->position != tokens || tokens > dg->context || sg->trial_n) { return false; }
    const uint32_t source_before = sg->position;
    dg->failed = true;
    for (unsigned il = 0; il < IQ_LAYERS; il++) {
        const uint64_t row = iquest_swa_row_bytes();
        if (iquest_full(il)) {
            if (!iquest_history_copy(dg, sg, il, tokens)) { return false; }
            continue;
        }
        const uint32_t first = tokens > IQ_WINDOW ? tokens - IQ_WINDOW : 0u;
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
    if (!ds4_gpu_tensor_copy(dg->mtp_kv, 0, sg->mtp_kv, 0, ds4_gpu_tensor_bytes(sg->mtp_kv)) ||
        !ds4_gpu_tensor_copy(dg->carry, 0, sg->carry, 0, IQ_EMBED * sizeof(float))) { return false; }
    dg->mtp_position = sg->mtp_position; dg->trial_n = 0;
    if (!ds4_gpu_synchronize()) { return false; }
    dg->position = tokens;
    dg->failed = false;
    memcpy(rt->bank_logits + (size_t)dst * IQ_VOCAB,
           rt->bank_logits + (size_t)src * IQ_VOCAB,
           IQ_VOCAB * sizeof(float));
    rt->bank_logits_valid[dst] = rt->bank_logits_valid[src];
    iquest_ckpt_inherit(rt, src, dst, tokens);
    iquest_reuse_trace(rt, src, dst, tokens, source_before, IQ_REUSE_FULL);
    return true;
}

static bool iquest_bank_prefill(ds4_iquest_batch_runtime *rt, ds4_engine *e,
                               uint32_t bank, const int *tokens, uint32_t rows,
                               uint32_t pos, iquest_prefill_mode mode) {
    if (!rt || !e || bank >= rt->max_seq) { return false; }
    rt->bank_logits_valid[bank] = 0;
    if (!iquest_forward(&rt->graph[bank], &e->model, &e->weights,
                       tokens, rows, pos, ds4_engine_has_mtp(e) ? IQ_MTP_ON : IQ_MTP_OFF)) { return false; }
    return mode == IQ_MORE_ROWS || iquest_bank_read_logits(rt, bank);
}

static bool iquest_bank_decode(ds4_iquest_batch_runtime *rt, ds4_engine *e,
                              const uint32_t *banks, const int *tokens,
                              const uint32_t *positions, uint32_t rows) {
    if (!rt || !e || !banks || !tokens || !positions || !rows) { return false; }
    for (uint32_t i = 0; i < rows; i++) {
        const uint32_t bank = banks[i];
        if (bank >= rt->max_seq || !iquest_bank_prefill(rt, e, bank,
                &tokens[i], 1u, positions[i], IQ_FINAL_ROWS)) { return false; }
    }
    return true;
}
