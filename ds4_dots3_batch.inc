/* Persistent dots3 text banks. Complete graphs own their state and scratch;
 * scheduling shares only resident weights and runs the existing scalar body. */
typedef struct ds4_dots3_batch_runtime {
    uint32_t max_seq, ctx_size, prefill_cap;
    ds4_dots3_gpu_graph *graph;
    float *bank_logits;
    uint8_t *bank_logits_valid;
    ds4_gpu_tensor *checkpoint_slab;
    ds4_partial_checkpoint checkpoint[DS4_PARTIAL_CHECKPOINT_SLOTS];
    float *checkpoint_logits;
    uint64_t checkpoint_slot_bytes, checkpoint_clock;
    uint32_t checkpoint_stride;
} ds4_dots3_batch_runtime;

/* Local MLA rings overwrite latent and rotated key rows. Full layers keep
 * those prefixes and F32 DSA keys in the source bank. The bound-only MTP
 * block owns no state here. Keep the serial forward arithmetic unchanged. */
static void dots3_ckpt_init(ds4_dots3_batch_runtime *rt) {
    const char *off = getenv("DS4_SERVER_FORK_PARTIAL");
    if ((off && strcmp(off, "0") == 0) ||
        DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_DOTS3_NOTE) { return; }
    const ds4_dots3_gpu_graph *g = &rt->graph[0];
    const uint32_t window = DS4_N_SWA < rt->ctx_size ? DS4_N_SWA : rt->ctx_size;
    for (uint32_t il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        if (g->layer_kv_latent[il] && !ds4_dots3_layer_is_full_attention(il)) {
            rt->checkpoint_slot_bytes += (uint64_t)window * (ds4_dots3_layer_kv_lora(il) + DS4_N_ROT) * sizeof(uint16_t);
        }
    }
    if (!rt->checkpoint_slot_bytes) { return; }
    uint64_t bytes = rt->checkpoint_slot_bytes * DS4_PARTIAL_CHECKPOINT_SLOTS;
    const uint64_t page = ds4_gpu_vmm_demand_page();
    if (page) { bytes = (bytes + page - 1u) / page * page; }
    rt->checkpoint_slab = ds4_gpu_tensor_reserve(bytes);
    if (!rt->checkpoint_slab) { return; }
    rt->checkpoint_logits = xcalloc(
        (size_t)DS4_PARTIAL_CHECKPOINT_SLOTS * DS4_N_VOCAB, sizeof(float));
    enum { CHECKPOINT_ALIGN = 4096 };
    const uint32_t stride = (rt->ctx_size + DS4_PARTIAL_PERIODIC_TARGET - 1u) / DS4_PARTIAL_PERIODIC_TARGET;
    rt->checkpoint_stride = ((stride + CHECKPOINT_ALIGN - 1u) / CHECKPOINT_ALIGN) * CHECKPOINT_ALIGN;
}

static void dots3_ckpt_drop(ds4_dots3_batch_runtime *rt, uint32_t bank) {
    for (unsigned i = 0; i < DS4_PARTIAL_CHECKPOINT_SLOTS; i++) {
        if (partial_checkpoint_ref(&rt->checkpoint[i], bank)) {
            partial_checkpoint_clear_ref(&rt->checkpoint[i], bank);
        }
    }
}

static void dots3_ckpt_inherit(ds4_dots3_batch_runtime *rt, uint32_t src,
                                 uint32_t dst, uint32_t cut) {
    if (src != dst) { dots3_ckpt_drop(rt, dst); }
    for (unsigned i = 0; i < DS4_PARTIAL_CHECKPOINT_SLOTS; i++) {
        ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (!partial_checkpoint_ref(cp, src)) { continue; }
        if (cp->pos <= cut) { partial_checkpoint_set_ref(cp, dst); }
        else if (src == dst) { partial_checkpoint_clear_ref(cp, dst); }
    }
}

static int dots3_ckpt_find(ds4_dots3_batch_runtime *rt, uint32_t bank,
                             uint32_t cut, uint32_t request_len) {
    int best = -1;
    for (unsigned i = 0; i < DS4_PARTIAL_CHECKPOINT_SLOTS; i++) {
        const ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (!cp->pos || cp->pos > cut || !partial_checkpoint_ref(cp, bank) ||
            (cp->pos == request_len && !cp->logits_valid)) { continue; }
        if (best < 0 || cp->pos > rt->checkpoint[best].pos) { best = (int)i; }
    }
    return best;
}

typedef enum { DOTS3_CKPT_SAVE, DOTS3_CKPT_LOAD } dots3_ckpt_dir;

static bool dots3_ckpt_window(ds4_dots3_batch_runtime *rt,
                               ds4_dots3_gpu_graph *g, uint32_t il,
                               uint64_t off, uint32_t pos, dots3_ckpt_dir dir) {
    const uint32_t count = pos < DS4_N_SWA ? pos : DS4_N_SWA;
    const uint32_t cap = g->layer_cache_cap[il];
    const uint32_t window = DS4_N_SWA < rt->ctx_size ? DS4_N_SWA : rt->ctx_size;
    const uint32_t widths[] = {ds4_dots3_layer_kv_lora(il), DS4_N_ROT};
    ds4_gpu_tensor *parts[] = {g->layer_kv_latent[il], g->layer_k_pe[il]};
    if (!cap || count > cap) { return false; }
    for (uint32_t part = 0; part < 2; part++) {
        const uint64_t row = (uint64_t)widths[part] * sizeof(uint16_t);
        for (uint32_t done = 0; done < count;) {
            const uint32_t slot = (pos - count + done) % cap;
            uint32_t rows = count - done;
            if (rows > cap - slot) { rows = cap - slot; }
            const bool ok = dir == DOTS3_CKPT_SAVE
                ? ds4_gpu_tensor_copy(rt->checkpoint_slab, off + done * row, parts[part], slot * row, rows * row)
                : ds4_gpu_tensor_copy(parts[part], slot * row, rt->checkpoint_slab, off + done * row, rows * row);
            if (!ok) { return false; }
            done += rows;
        }
        off += window * row;
    }
    return true;
}

typedef enum { DOTS3_COPY_ALL, DOTS3_COPY_FULL } dots3_copy_kind;

static bool dots3_copy_prefix(ds4_dots3_batch_runtime *rt, uint32_t src,
                               uint32_t dst, uint32_t pos, dots3_copy_kind kind) {
    if (src == dst) { return true; }
    ds4_dots3_gpu_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    for (uint32_t il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        const bool full = ds4_dots3_layer_is_full_attention(il);
        if (kind == DOTS3_COPY_FULL && !full) { continue; }
        const uint32_t rows = pos < sg->layer_cache_cap[il] ? pos : sg->layer_cache_cap[il];
        if (!ds4_gpu_tensor_copy(dg->layer_kv_latent[il], 0, sg->layer_kv_latent[il], 0,
                (uint64_t)rows * ds4_dots3_layer_kv_lora(il) * sizeof(uint16_t)) ||
            !ds4_gpu_tensor_copy(dg->layer_k_pe[il], 0, sg->layer_k_pe[il], 0,
                (uint64_t)rows * DS4_N_ROT * sizeof(uint16_t)) ||
            (full && !ds4_gpu_tensor_copy(dg->layer_idx_k[il], 0, sg->layer_idx_k[il], 0,
                (uint64_t)rows * DS4_N_INDEXER_HEAD_DIM * sizeof(float)))) { return false; }
    }
    return true;
}

static bool dots3_ckpt_capture(ds4_dots3_batch_runtime *rt, uint32_t bank,
                                 uint32_t pos, bool logits_valid, uint64_t reserve) {
    if (!rt || !rt->checkpoint_slab || bank >= rt->max_seq || !pos ||
        pos != rt->graph[bank].cache_len) { return false; }
    uint32_t slot = 0;
    for (uint32_t i = 0; i < DS4_PARTIAL_CHECKPOINT_SLOTS; i++) {
        ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (cp->pos == pos && partial_checkpoint_ref(cp, bank)) {
            if (logits_valid && rt->bank_logits_valid[bank]) {
                memcpy(rt->checkpoint_logits + (size_t)i * DS4_N_VOCAB,
                       rt->bank_logits + (size_t)bank * DS4_N_VOCAB, DS4_N_VOCAB * sizeof(float));
                cp->logits_valid = 1;
            }
            cp->last_use = ++rt->checkpoint_clock;
            return true;
        }
        if (!cp->pos || (rt->checkpoint[slot].pos && cp->last_use < rt->checkpoint[slot].last_use)) {
            slot = i;
        }
    }
    const uint64_t base = (uint64_t)slot * rt->checkpoint_slot_bytes;
    const uint64_t need = batch_span_need(rt->checkpoint_slab, base, rt->checkpoint_slot_bytes);
    if (need > ds4_mem_usable_beyond(reserve) ||
        !ds4_gpu_tensor_ensure(rt->checkpoint_slab, base, rt->checkpoint_slot_bytes)) { return false; }
    /* Invalidate before copying so a failed overwrite cannot retain lineage. */
    ds4_partial_checkpoint *cp = &rt->checkpoint[slot];
    memset(cp, 0, sizeof(*cp));
    ds4_dots3_gpu_graph *g = &rt->graph[bank];
    const uint32_t window = DS4_N_SWA < rt->ctx_size ? DS4_N_SWA : rt->ctx_size;
    uint64_t off = base;
    for (uint32_t il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        if (!g->layer_kv_latent[il] || ds4_dots3_layer_is_full_attention(il)) { continue; }
        if (!dots3_ckpt_window(rt, g, il, off, pos, DOTS3_CKPT_SAVE)) { return false; }
        off += (uint64_t)window * (ds4_dots3_layer_kv_lora(il) + DS4_N_ROT) * sizeof(uint16_t);
    }
    if (!ds4_gpu_synchronize()) { return false; }
    cp->pos = pos;
    cp->last_use = ++rt->checkpoint_clock;
    partial_checkpoint_set_ref(cp, bank);
    if (logits_valid && rt->bank_logits_valid[bank]) {
        memcpy(rt->checkpoint_logits + (size_t)slot * DS4_N_VOCAB,
               rt->bank_logits + (size_t)bank * DS4_N_VOCAB, DS4_N_VOCAB * sizeof(float));
        cp->logits_valid = 1;
    }
    return true;
}

static bool dots3_ckpt_restore(ds4_dots3_batch_runtime *rt, uint32_t src,
                                 uint32_t dst, uint32_t slot, uint32_t cut, uint32_t *out) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || slot >= DS4_PARTIAL_CHECKPOINT_SLOTS) { return false; }
    ds4_partial_checkpoint *cp = &rt->checkpoint[slot];
    const uint32_t pos = cp->pos;
    if (!pos || pos > cut || pos > rt->graph[src].cache_len || !partial_checkpoint_ref(cp, src)) { return false; }
    ds4_dots3_gpu_graph *g = &rt->graph[dst];
    rt->graph[dst].cache_len = 0;
    rt->bank_logits_valid[dst] = 0;
    const uint32_t window = DS4_N_SWA < rt->ctx_size ? DS4_N_SWA : rt->ctx_size;
    uint64_t off = (uint64_t)slot * rt->checkpoint_slot_bytes;
    for (uint32_t il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        if (ds4_dots3_layer_is_full_attention(il)) { continue; }
        if (!dots3_ckpt_window(rt, g, il, off, pos, DOTS3_CKPT_LOAD)) { return false; }
        off += (uint64_t)window * (ds4_dots3_layer_kv_lora(il) + DS4_N_ROT) * sizeof(uint16_t);
    }
    if (!dots3_copy_prefix(rt, src, dst, pos, DOTS3_COPY_FULL) || !ds4_gpu_synchronize()) { return false; }
    rt->graph[dst].cache_len = pos;
    if (cp->logits_valid) {
        memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB,
               rt->checkpoint_logits + (size_t)slot * DS4_N_VOCAB, DS4_N_VOCAB * sizeof(float));
        rt->bank_logits_valid[dst] = 1;
    }
    cp->last_use = ++rt->checkpoint_clock;
    dots3_ckpt_inherit(rt, src, dst, pos);
    if (out) { *out = pos; }
    return true;
}

static uint64_t dots3_ckpt_trim(ds4_dots3_batch_runtime *rt, uint64_t want) {
    if (!rt || !rt->checkpoint_slab || !want || !ds4_gpu_synchronize()) { return 0; }
    uint64_t freed = 0;
    /* Coalesce unreferenced slots so a shared VMM page is unmapped only when
     * all adjacent slots using it are dead. The caller holds the bank lock. */
    for (uint32_t i = 0; i < DS4_PARTIAL_CHECKPOINT_SLOTS && freed < want;) {
        if (rt->checkpoint[i].pos) { i++; continue; }
        const uint64_t start = (uint64_t)i * rt->checkpoint_slot_bytes;
        do { i++; } while (i < DS4_PARTIAL_CHECKPOINT_SLOTS && !rt->checkpoint[i].pos);
        const uint64_t end = i == DS4_PARTIAL_CHECKPOINT_SLOTS
            ? ds4_gpu_tensor_bytes(rt->checkpoint_slab)
            : (uint64_t)i * rt->checkpoint_slot_bytes;
        freed += ds4_gpu_tensor_trim(rt->checkpoint_slab, start, end - start);
    }
    return freed;
}

static void dots3_bank_reset(ds4_dots3_batch_runtime *rt, uint32_t bank) {
    dots3_ckpt_drop(rt, bank);
    rt->graph[bank].cache_len = 0;
    rt->bank_logits_valid[bank] = 0;
}

static void dots3_batch_free(ds4_dots3_batch_runtime *rt) {
    if (!rt) { return; }
    if (rt->graph) {
        for (uint32_t b = 0; b < rt->max_seq; b++) { dots3_graph_free(&rt->graph[b]); }
    }
    free(rt->graph);
    free(rt->bank_logits);
    free(rt->bank_logits_valid);
    ds4_gpu_tensor_free(rt->checkpoint_slab);
    free(rt->checkpoint_logits);
    free(rt);
}

static ds4_dots3_batch_runtime *dots3_batch_create(ds4_engine *e, uint32_t ctx,
                                                  uint32_t banks, uint32_t cap) {
    ds4_dots3_batch_runtime *rt = xcalloc(1, sizeof(*rt));
    rt->max_seq = banks;
    rt->ctx_size = ctx;
    rt->prefill_cap = cap;
    rt->graph = xcalloc(banks, sizeof(*rt->graph));
    rt->bank_logits = xcalloc((size_t)banks * DS4_N_VOCAB, sizeof(float));
    rt->bank_logits_valid = xcalloc(banks, 1);
    for (uint32_t b = 0; b < banks; b++) {
        if (!dots3_graph_alloc(&rt->graph[b], cap) ||
            !dots3_graph_alloc_cache(&rt->graph[b], &e->model, &e->weights, ctx)) {
            dots3_batch_free(rt);
            return NULL;
        }
    }
    dots3_ckpt_init(rt);
    return rt;
}

static bool dots3_bank_copy(ds4_dots3_batch_runtime *rt, uint32_t src,
                             uint32_t dst, uint32_t pos) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq ||
        pos != rt->graph[src].cache_len) { return false; }
    if (src == dst) { return true; }
    rt->graph[dst].cache_len = 0;
    rt->bank_logits_valid[dst] = 0;
    if (!dots3_copy_prefix(rt, src, dst, pos, DOTS3_COPY_ALL) || !ds4_gpu_synchronize()) { return false; }
    rt->graph[dst].cache_len = pos;
    if (rt->bank_logits_valid[src]) {
        memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB,
               rt->bank_logits + (size_t)src * DS4_N_VOCAB, DS4_N_VOCAB * sizeof(float));
        rt->bank_logits_valid[dst] = 1;
    }
    dots3_ckpt_inherit(rt, src, dst, pos);
    return true;
}

static bool dots3_bank_forward(ds4_dots3_batch_runtime *rt, ds4_engine *e,
                                uint32_t bank, const int *tokens, uint32_t rows,
                                uint32_t pos, bool final) {
    if (!rt || bank >= rt->max_seq) { return false; }
    rt->bank_logits_valid[bank] = 0;
    const bool ok = dots3_graph_forward_chunk(&rt->graph[bank], &e->model, &e->weights,
        tokens, rows, pos, final ? rt->bank_logits + (size_t)bank * DS4_N_VOCAB : NULL);
    if (!ok) { dots3_bank_reset(rt, bank); }
    else if (final) { rt->bank_logits_valid[bank] = 1; }
    return ok;
}
