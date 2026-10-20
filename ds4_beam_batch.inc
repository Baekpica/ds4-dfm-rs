/* Beam banks. The scheduler serializes bank forwards under the engine
 * generation lock, so graph 0's scratch serves every bank; each bank owns its
 * KV rows (the full layers' FP8 rows and the sliding rings). Full-layer rows
 * are append-only and fork straight from the source bank; a ring keeps only
 * the last window plus one chunk, so partial reuse snapshots the rings' last
 * BEAM_WINDOW rows into virtual checkpoint slots funded at capture, at most as
 * many as the memory the fit left (slot_cap). */
enum { BEAM_BANK_MAX = DS4_MULTISEQ_MAX_SEQ, BEAM_BANK_CHECKPOINTS = 8, BEAM_CKPT_ALIGN = 4096 };
typedef enum { BEAM_MORE_ROWS, BEAM_FINAL_ROWS } beam_prefill_mode;
typedef enum { BEAM_STRICT_BANKS, BEAM_FIT_BANKS } beam_fit_mode;

typedef struct {
    uint32_t max_seq, ctx_size, prefill_cap, checkpoint_stride, slot_cap;
    ds4_beam_graph *graph;
    float *bank_logits, *checkpoint_logits;
    uint8_t *bank_logits_valid;
    ds4_gpu_tensor *checkpoint_slab;
    ds4_partial_checkpoint checkpoint[BEAM_BANK_CHECKPOINTS];
    uint64_t checkpoint_slot_bytes, checkpoint_clock;
} ds4_beam_batch_runtime;

// What one more bank adds beyond graph 0: its KV rows.
static uint64_t beam_bank_bytes(uint32_t ctx, uint32_t cap) { return beam_kv_bytes(ctx, cap); }

// One checkpoint: every ring's last BEAM_WINDOW rows in position order.
static uint64_t beam_ckpt_bytes(void) {
    return (uint64_t)(BEAM_LAYERS - BEAM_FULL_LAYERS) * BEAM_WINDOW * BEAM_SWA_ROW;
}

/* Rows [first, first + keep) of a ring occupy slots p % cap: at most two
 * spans, written to (or read from) `out` in position order. */
typedef struct { uint64_t slot, rows; } beam_span;

static uint32_t beam_ring_spans(uint32_t cap, uint32_t first, uint32_t keep, beam_span span[2]) {
    const uint32_t slot = first % cap, head = keep < cap - slot ? keep : cap - slot;
    span[0] = (beam_span){slot, head};
    span[1] = (beam_span){0u, keep - head};
    return span[1].rows ? 2u : 1u;
}

// Ring rows [pos - keep, pos) of every sliding layer of `src` into `dst`
// (same ring geometry), and full-layer rows [0, pos) unless `full` is false.
static bool beam_rows_copy(ds4_beam_graph *dst, const ds4_beam_graph *src, uint32_t pos, bool full) {
    const uint32_t keep = beam_payload_window(pos);
    for (uint32_t il = 0; il < BEAM_LAYERS; il++) {
        if (ds4_beam_layer_is_full(il)) {
            if (full && pos && !ds4_gpu_tensor_copy(dst->kv[il], 0, src->kv[il], 0,
                    (uint64_t)pos * beam_full_row())) { return false; }
            continue;
        }
        beam_span span[2];
        const uint32_t n = beam_ring_spans(src->swa_cap, pos - keep, keep, span);
        for (uint32_t s = 0; s < n; s++) {
            const uint64_t at = span[s].slot * BEAM_SWA_ROW;
            if (!ds4_gpu_tensor_copy(dst->kv[il], at, src->kv[il], at, span[s].rows * BEAM_SWA_ROW)) {
                return false;
            }
        }
    }
    return true;
}

// The rings' window rows of `g` at `pos` to (SAVE) or from (LOAD) a slot.
static bool beam_ckpt_rows(ds4_beam_batch_runtime *rt, ds4_beam_graph *g, uint32_t slot, uint32_t pos,
        beam_payload_dir dir) {
    const uint32_t keep = beam_payload_window(pos);
    uint64_t at = (uint64_t)slot * rt->checkpoint_slot_bytes;
    for (uint32_t il = 0; il < BEAM_LAYERS; il++) {
        if (ds4_beam_layer_is_full(il)) { continue; }
        beam_span span[2];
        const uint32_t n = beam_ring_spans(g->swa_cap, pos - keep, keep, span);
        uint64_t row = at;
        for (uint32_t s = 0; s < n; s++) {
            const uint64_t ring = span[s].slot * BEAM_SWA_ROW, bytes = span[s].rows * BEAM_SWA_ROW;
            const bool ok = dir == BEAM_PAYLOAD_SAVE
                ? ds4_gpu_tensor_copy(rt->checkpoint_slab, row, g->kv[il], ring, bytes)
                : ds4_gpu_tensor_copy(g->kv[il], ring, rt->checkpoint_slab, row, bytes);
            if (!ok) { return false; }
            row += bytes;
        }
        at += (uint64_t)BEAM_WINDOW * BEAM_SWA_ROW;
    }
    return true;
}

static void beam_ckpt_drop(ds4_beam_batch_runtime *rt, uint32_t bank) {
    for (uint32_t i = 0; i < BEAM_BANK_CHECKPOINTS; i++) {
        if (partial_checkpoint_ref(&rt->checkpoint[i], bank)) { partial_checkpoint_clear_ref(&rt->checkpoint[i], bank); }
    }
}

static void beam_ckpt_inherit(ds4_beam_batch_runtime *rt, uint32_t src, uint32_t dst, uint32_t cut) {
    if (src != dst) { beam_ckpt_drop(rt, dst); }
    for (uint32_t i = 0; i < BEAM_BANK_CHECKPOINTS; i++) {
        ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (!partial_checkpoint_ref(cp, src)) { continue; }
        if (cp->pos <= cut) { partial_checkpoint_set_ref(cp, dst); }
        else if (src == dst) { partial_checkpoint_clear_ref(cp, dst); }
    }
}

static int beam_ckpt_find(ds4_beam_batch_runtime *rt, uint32_t bank, uint32_t cut, uint32_t request_len) {
    int best = -1;
    for (uint32_t i = 0; i < BEAM_BANK_CHECKPOINTS; i++) {
        const ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (!cp->pos || cp->pos > cut || !partial_checkpoint_ref(cp, bank) ||
            (cp->pos == request_len && !cp->logits_valid)) { continue; }
        if (best < 0 || cp->pos > rt->checkpoint[best].pos) { best = (int)i; }
    }
    return best;
}

static bool beam_ckpt_capture(ds4_beam_batch_runtime *rt, uint32_t bank, uint32_t pos, bool logits,
        uint64_t reserve) {
    if (!rt || !rt->checkpoint_slab || bank >= rt->max_seq || !pos || rt->graph[bank].failed ||
        rt->graph[bank].position != pos) { return false; }
    float *bank_logits = rt->bank_logits + (size_t)bank * DS4_N_VOCAB;
    uint32_t slot = 0;
    for (uint32_t i = 0; i < BEAM_BANK_CHECKPOINTS; i++) {
        ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (cp->pos == pos && partial_checkpoint_ref(cp, bank)) {
            if (logits && rt->bank_logits_valid[bank]) {
                memcpy(rt->checkpoint_logits + (size_t)i * DS4_N_VOCAB, bank_logits, DS4_N_VOCAB * sizeof(float));
                cp->logits_valid = 1;
            }
            cp->last_use = ++rt->checkpoint_clock;
            return true;
        }
        if (!cp->pos || (rt->checkpoint[slot].pos && cp->last_use < rt->checkpoint[slot].last_use)) { slot = i; }
    }
    const uint64_t bytes = rt->checkpoint_slot_bytes;
    uint32_t mapped = 0;
    int oldest = -1;
    for (uint32_t i = 0; i < BEAM_BANK_CHECKPOINTS; i++) {
        if (batch_span_need(rt->checkpoint_slab, (uint64_t)i * bytes, bytes)) { continue; }
        mapped++;
        if (oldest < 0 || rt->checkpoint[i].last_use < rt->checkpoint[oldest].last_use) { oldest = (int)i; }
    }
    const uint64_t need = batch_span_need(rt->checkpoint_slab, (uint64_t)slot * bytes, bytes);
    if (need && (mapped >= rt->slot_cap || need > ds4_mem_usable_beyond(reserve))) {
        // Budget spent or memory short: recycle the oldest mapped slot.
        if (oldest < 0) { return false; }
        slot = (uint32_t)oldest;
    }
    if (!ds4_gpu_tensor_ensure(rt->checkpoint_slab, (uint64_t)slot * bytes, bytes)) { return false; }
    // Invalidate first: a failed copy must not leave lineage on a torn slot.
    ds4_partial_checkpoint *cp = &rt->checkpoint[slot];
    memset(cp, 0, sizeof(*cp));
    if (!beam_ckpt_rows(rt, &rt->graph[bank], slot, pos, BEAM_PAYLOAD_SAVE) || !ds4_gpu_synchronize()) {
        return false;
    }
    cp->pos = pos;
    cp->last_use = ++rt->checkpoint_clock;
    partial_checkpoint_set_ref(cp, bank);
    if (logits && rt->bank_logits_valid[bank]) {
        memcpy(rt->checkpoint_logits + (size_t)slot * DS4_N_VOCAB, bank_logits, DS4_N_VOCAB * sizeof(float));
        cp->logits_valid = 1;
    }
    return true;
}

static bool beam_ckpt_restore(ds4_beam_batch_runtime *rt, uint32_t src, uint32_t dst, uint32_t slot,
        uint32_t cut, uint32_t *out) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || slot >= BEAM_BANK_CHECKPOINTS) { return false; }
    ds4_partial_checkpoint *cp = &rt->checkpoint[slot];
    const uint32_t pos = cp->pos;
    ds4_beam_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (!pos || pos > cut || pos > sg->position || sg->failed || !partial_checkpoint_ref(cp, src)) { return false; }
    dg->failed = true;
    rt->bank_logits_valid[dst] = 0;
    if ((src != dst && !beam_rows_copy(dg, sg, pos, true)) ||
        !beam_ckpt_rows(rt, dg, slot, pos, BEAM_PAYLOAD_LOAD) || !ds4_gpu_synchronize()) { return false; }
    dg->position = pos;
    dg->failed = false;
    if (cp->logits_valid) {
        memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB, rt->checkpoint_logits + (size_t)slot * DS4_N_VOCAB,
               DS4_N_VOCAB * sizeof(float));
        rt->bank_logits_valid[dst] = 1;
    }
    cp->last_use = ++rt->checkpoint_clock;
    beam_ckpt_inherit(rt, src, dst, pos);
    if (out) { *out = pos; }
    return true;
}

static uint64_t beam_ckpt_trim(ds4_beam_batch_runtime *rt, uint64_t want) {
    uint64_t freed = 0;
    if (!rt || !rt->checkpoint_slab || !want || !ds4_gpu_synchronize()) { return 0; }
    for (uint32_t i = 0; i < BEAM_BANK_CHECKPOINTS && freed < want;) {
        if (rt->checkpoint[i].pos) { i++; continue; }
        const uint64_t start = (uint64_t)i * rt->checkpoint_slot_bytes;
        do { i++; } while (i < BEAM_BANK_CHECKPOINTS && !rt->checkpoint[i].pos);
        const uint64_t end = i == BEAM_BANK_CHECKPOINTS
            ? ds4_gpu_tensor_bytes(rt->checkpoint_slab) : (uint64_t)i * rt->checkpoint_slot_bytes;
        freed += ds4_gpu_tensor_trim(rt->checkpoint_slab, start, end - start);
    }
    return freed;
}

static void beam_batch_free(ds4_beam_batch_runtime *rt) {
    if (!rt) { return; }
    // Banks borrow graph 0's buffers: free them first.
    for (uint32_t b = rt->max_seq; rt->graph && b > 0; b--) { beam_graph_free(&rt->graph[b - 1u]); }
    free(rt->graph);
    free(rt->bank_logits);
    free(rt->bank_logits_valid);
    ds4_gpu_tensor_free(rt->checkpoint_slab);
    free(rt->checkpoint_logits);
    free(rt);
}

static void beam_ckpt_init(ds4_beam_batch_runtime *rt, uint32_t slots) {
    const char *off = getenv("DS4_SERVER_FORK_PARTIAL");
    if ((off && strcmp(off, "0") == 0) || !slots) { return; }
    rt->slot_cap = slots < BEAM_BANK_CHECKPOINTS ? slots : BEAM_BANK_CHECKPOINTS;
    rt->checkpoint_slot_bytes = beam_ckpt_bytes();
    uint64_t bytes = rt->checkpoint_slot_bytes * BEAM_BANK_CHECKPOINTS;
    const uint64_t page = ds4_gpu_vmm_demand_page();
    if (page) { bytes = (bytes + page - 1u) / page * page; }
    rt->checkpoint_slab = ds4_gpu_tensor_reserve(bytes);
    if (!rt->checkpoint_slab) { return; }
    rt->checkpoint_logits = xcalloc((size_t)BEAM_BANK_CHECKPOINTS * DS4_N_VOCAB, sizeof(float));
    const uint32_t stride = (rt->ctx_size + DS4_PARTIAL_PERIODIC_TARGET - 1u) / DS4_PARTIAL_PERIODIC_TARGET;
    rt->checkpoint_stride = (stride + BEAM_CKPT_ALIGN - 1u) / BEAM_CKPT_ALIGN * BEAM_CKPT_ALIGN;
}

static ds4_beam_batch_runtime *beam_batch_create(ds4_engine *e, uint32_t ctx, uint32_t max_seq, uint32_t cap,
        uint32_t slots) {
    if (!e || !ctx || !max_seq || max_seq > BEAM_BANK_MAX || !cap || cap > ctx) { return NULL; }
    ds4_beam_batch_runtime *rt = xcalloc(1, sizeof(*rt));
    rt->ctx_size = ctx;
    rt->max_seq = max_seq;
    rt->prefill_cap = cap;
    rt->graph = xcalloc(max_seq, sizeof(*rt->graph));
    if (!beam_graph_alloc(&rt->graph[0], &e->model, &e->weights, ctx, cap)) {
        beam_batch_free(rt);
        return NULL;
    }
    for (uint32_t b = 1; b < max_seq; b++) {
        if (!beam_graph_bank(&rt->graph[b], &rt->graph[0])) {
            beam_batch_free(rt);
            return NULL;
        }
    }
    rt->bank_logits = xcalloc((size_t)max_seq * DS4_N_VOCAB, sizeof(float));
    rt->bank_logits_valid = xcalloc(max_seq, sizeof(uint8_t));
    beam_ckpt_init(rt, slots);
    return rt;
}

static bool beam_bank_read_logits(ds4_beam_batch_runtime *rt, uint32_t bank) {
    float *dst = rt->bank_logits + (size_t)bank * DS4_N_VOCAB;
    bool ok = ds4_gpu_tensor_read(rt->graph[bank].logits, 0, dst, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    for (uint32_t i = 0; ok && i < DS4_N_VOCAB; i++) { ok = isfinite(dst[i]); }
    rt->bank_logits_valid[bank] = ok;
    if (!ok) { rt->graph[bank].failed = true; }
    return ok;
}

static void beam_bank_reset(ds4_beam_batch_runtime *rt, uint32_t bank) {
    beam_ckpt_drop(rt, bank);
    rt->bank_logits_valid[bank] = 0;
    (void)beam_graph_reset(&rt->graph[bank]);
}

// Exact fork: every full-layer row of the prefix and the rings' window.
static bool beam_bank_copy(ds4_beam_batch_runtime *rt, uint32_t src, uint32_t dst, uint32_t tokens) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || src == dst) { return false; }
    ds4_beam_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (sg->failed || sg->position != tokens || tokens > dg->ctx_cap) { return false; }
    dg->failed = true;
    if (!beam_rows_copy(dg, sg, tokens, true) || !ds4_gpu_synchronize()) { return false; }
    dg->position = tokens;
    dg->failed = false;
    memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB, rt->bank_logits + (size_t)src * DS4_N_VOCAB,
           DS4_N_VOCAB * sizeof(float));
    rt->bank_logits_valid[dst] = rt->bank_logits_valid[src];
    beam_ckpt_inherit(rt, src, dst, tokens);
    return true;
}

static bool beam_bank_prefill(ds4_beam_batch_runtime *rt, ds4_engine *e, uint32_t bank, const int *tokens,
        uint32_t rows, uint32_t pos, beam_prefill_mode mode) {
    if (!rt || !e || bank >= rt->max_seq) { return false; }
    rt->bank_logits_valid[bank] = 0;
    const beam_head_mode head = mode == BEAM_FINAL_ROWS ? BEAM_HEAD_LAST : BEAM_HEAD_SKIP;
    if (!beam_forward(&rt->graph[bank], &e->model, &e->weights, tokens, rows, pos, head)) { return false; }
    return mode == BEAM_MORE_ROWS || beam_bank_read_logits(rt, bank);
}

/* Each bank attends over its own KV rows, so decode rows of different banks
 * share no forward: run them in bank order. */
static bool beam_bank_decode(ds4_beam_batch_runtime *rt, ds4_engine *e, const uint32_t *banks, const int *tokens,
        const uint32_t *positions, uint32_t rows) {
    if (!rt || !e || !banks || !tokens || !positions || !rows) { return false; }
    for (uint32_t i = 0; i < rows; i++) {
        if (banks[i] >= rt->max_seq ||
            !beam_bank_prefill(rt, e, banks[i], &tokens[i], 1u, positions[i], BEAM_FINAL_ROWS)) { return false; }
    }
    return true;
}
