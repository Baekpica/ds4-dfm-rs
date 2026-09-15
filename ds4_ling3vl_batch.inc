/* Ling continuous lane: N self-contained per-bank graphs sharing only the
 * resident weights. Each bank owns its 42-layer state -- seven linear latent
 * caches and 35 recurrent tiles -- plus its prefill scratch, so admission,
 * prefill and decode reuse the single-graph forward byte for byte. Images
 * stay on the serial lane.
 *
 * Partial reuse costs one constant snapshot rather than a per-token window.
 * The latent caches are append-only, so a fork copies rows [0, pos) straight
 * from the source bank; only the recurrent state has no rewind and needs the
 * checkpoint slab. */
enum { DS4_LING3VL_BANK_MAX = DS4_MULTISEQ_MAX_SEQ };

typedef struct ds4_ling3vl_batch_runtime {
    uint32_t max_seq;
    uint32_t ctx_size;
    uint32_t prefill_cap;
    ds4_ling3vl_graph *graph;
    float *bank_logits;
    uint8_t *bank_logits_valid;
    ds4_gpu_tensor *checkpoint_slab;
    ds4_partial_checkpoint checkpoint[DS4_PARTIAL_CHECKPOINT_SLOTS];
    float *checkpoint_logits;
    uint64_t checkpoint_slot_bytes, checkpoint_clock;
    uint32_t checkpoint_stride;
} ds4_ling3vl_batch_runtime;

static void ling3vl_ckpt_init(ds4_ling3vl_batch_runtime *rt) {
    const char *off = getenv("DS4_SERVER_FORK_PARTIAL");
    if (off && strcmp(off, "0") == 0) { return; }
    rt->checkpoint_slot_bytes = rt->graph[0].state_bytes;
    if (!rt->checkpoint_slot_bytes ||
        rt->checkpoint_slot_bytes > UINT64_MAX / DS4_PARTIAL_CHECKPOINT_SLOTS) {
        return;
    }
    uint64_t bytes = rt->checkpoint_slot_bytes * DS4_PARTIAL_CHECKPOINT_SLOTS;
    const uint64_t page = ds4_gpu_vmm_demand_page();
    if (page) { bytes = (bytes + page - 1u) / page * page; }
    rt->checkpoint_slab = ds4_gpu_tensor_reserve(bytes);
    if (!rt->checkpoint_slab) { return; }
    rt->checkpoint_logits = xcalloc(
        (size_t)DS4_PARTIAL_CHECKPOINT_SLOTS * DS4_N_VOCAB, sizeof(float));
    enum { CHECKPOINT_ALIGN = 4096 };
    uint32_t stride = (rt->ctx_size + DS4_PARTIAL_PERIODIC_TARGET - 1u) /
                      DS4_PARTIAL_PERIODIC_TARGET;
    rt->checkpoint_stride =
        ((stride + CHECKPOINT_ALIGN - 1u) / CHECKPOINT_ALIGN) * CHECKPOINT_ALIGN;
}

static void ling3vl_ckpt_drop(ds4_ling3vl_batch_runtime *rt, uint32_t bank) {
    for (unsigned i = 0; i < DS4_PARTIAL_CHECKPOINT_SLOTS; i++) {
        if (partial_checkpoint_ref(&rt->checkpoint[i], bank)) {
            partial_checkpoint_clear_ref(&rt->checkpoint[i], bank);
        }
    }
}

static uint64_t ling3vl_ckpt_trim(ds4_ling3vl_batch_runtime *rt, uint64_t want) {
    uint64_t freed = 0;
    if (!rt || !rt->checkpoint_slab || !want || !ds4_gpu_synchronize()) { return 0; }
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

static void ling3vl_ckpt_inherit(ds4_ling3vl_batch_runtime *rt, uint32_t src,
                                 uint32_t dst, uint32_t cut) {
    if (src != dst) { ling3vl_ckpt_drop(rt, dst); }
    for (unsigned i = 0; i < DS4_PARTIAL_CHECKPOINT_SLOTS; i++) {
        ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (!partial_checkpoint_ref(cp, src)) { continue; }
        if (cp->pos <= cut) { partial_checkpoint_set_ref(cp, dst); }
        else if (src == dst) { partial_checkpoint_clear_ref(cp, dst); }
    }
}

static int ling3vl_ckpt_find(ds4_ling3vl_batch_runtime *rt, uint32_t bank,
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

static bool ling3vl_ckpt_capture(ds4_ling3vl_batch_runtime *rt, uint32_t bank,
                                 uint32_t pos, bool logits_valid, uint64_t reserve) {
    if (!rt || !rt->checkpoint_slab || bank >= rt->max_seq || !pos ||
        rt->graph[bank].failed || pos != rt->graph[bank].position) { return false; }
    uint32_t slot = 0;
    for (uint32_t i = 0; i < DS4_PARTIAL_CHECKPOINT_SLOTS; i++) {
        ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (cp->pos == pos && partial_checkpoint_ref(cp, bank)) {
            if (logits_valid && rt->bank_logits_valid[bank]) {
                memcpy(rt->checkpoint_logits + (size_t)i * DS4_N_VOCAB,
                       rt->bank_logits + (size_t)bank * DS4_N_VOCAB,
                       DS4_N_VOCAB * sizeof(float));
                cp->logits_valid = 1;
            }
            cp->last_use = ++rt->checkpoint_clock;
            return true;
        }
        if (!cp->pos ||
            (rt->checkpoint[slot].pos && cp->last_use < rt->checkpoint[slot].last_use)) {
            slot = i;
        }
    }
    const uint64_t base = (uint64_t)slot * rt->checkpoint_slot_bytes;
    const uint64_t need = batch_span_need(rt->checkpoint_slab, base,
                                          rt->checkpoint_slot_bytes);
    if (need > ds4_mem_usable_beyond(reserve) ||
        !ds4_gpu_tensor_ensure(rt->checkpoint_slab, base,
                               rt->checkpoint_slot_bytes)) { return false; }
    /* Invalidate before overwriting: a failed copy must never leave old
     * lineage pointing at a partially replaced slot. */
    ds4_partial_checkpoint *cp = &rt->checkpoint[slot];
    memset(cp, 0, sizeof(*cp));
    if (!ds4_gpu_tensor_copy(rt->checkpoint_slab, base,
                             rt->graph[bank].state_pool, 0,
                             rt->checkpoint_slot_bytes) ||
        !ds4_gpu_synchronize()) { return false; }
    cp->pos = pos;
    cp->last_use = ++rt->checkpoint_clock;
    partial_checkpoint_set_ref(cp, bank);
    if (logits_valid && rt->bank_logits_valid[bank]) {
        memcpy(rt->checkpoint_logits + (size_t)slot * DS4_N_VOCAB,
               rt->bank_logits + (size_t)bank * DS4_N_VOCAB,
               DS4_N_VOCAB * sizeof(float));
        cp->logits_valid = 1;
    }
    return true;
}

static bool ling3vl_ckpt_restore(ds4_ling3vl_batch_runtime *rt, uint32_t src,
                                 uint32_t dst, uint32_t slot, uint32_t cut,
                                 uint32_t *out) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq ||
        slot >= DS4_PARTIAL_CHECKPOINT_SLOTS) { return false; }
    ds4_partial_checkpoint *cp = &rt->checkpoint[slot];
    const uint32_t pos = cp->pos;
    if (!pos || pos > cut || pos > rt->graph[src].position ||
        rt->graph[src].failed || !partial_checkpoint_ref(cp, src)) { return false; }
    ds4_ling3vl_graph *g = &rt->graph[dst];
    g->failed = true;
    rt->bank_logits_valid[dst] = 0;
    if (!ds4_gpu_tensor_copy(g->state_pool, 0, rt->checkpoint_slab,
                             (uint64_t)slot * rt->checkpoint_slot_bytes,
                             rt->checkpoint_slot_bytes)) { return false; }
    /* The latent caches are append-only, so the source still holds the exact
     * rows this prefix needs; a same-bank restore already has them. */
    for (uint32_t il = 0; src != dst && il < LING3VL_LAYERS; il++) {
        if (ds4_ling3vl_layer_is_kda(il)) { continue; }
        if (!ds4_gpu_tensor_copy(g->latent_cache[il], 0,
                                 rt->graph[src].latent_cache[il], 0,
                                 (uint64_t)pos * L3V_KV_LORA * sizeof(uint16_t)) ||
            !ds4_gpu_tensor_copy(g->k_pe_cache[il], 0,
                                 rt->graph[src].k_pe_cache[il], 0,
                                 (uint64_t)pos * L3V_ROPE * sizeof(uint16_t))) {
            return false;
        }
    }
    if (!ds4_gpu_synchronize()) { return false; }
    g->position = pos;
    g->high_water = pos;
    g->failed = false;
    if (cp->logits_valid) {
        memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB,
               rt->checkpoint_logits + (size_t)slot * DS4_N_VOCAB,
               DS4_N_VOCAB * sizeof(float));
        rt->bank_logits_valid[dst] = 1;
    }
    cp->last_use = ++rt->checkpoint_clock;
    ling3vl_ckpt_inherit(rt, src, dst, pos);
    if (out) { *out = pos; }
    return true;
}

static void ling3vl_batch_runtime_free(ds4_ling3vl_batch_runtime *rt) {
    if (!rt) { return; }
    if (rt->graph) {
        for (uint32_t b = 0; b < rt->max_seq; b++) { ling3vl_graph_free(&rt->graph[b]); }
    }
    free(rt->graph);
    free(rt->bank_logits);
    free(rt->bank_logits_valid);
    ds4_gpu_tensor_free(rt->checkpoint_slab);
    free(rt->checkpoint_logits);
    free(rt);
}

static ds4_ling3vl_batch_runtime *ling3vl_batch_runtime_create(
        ds4_engine *e, uint32_t ctx_size, uint32_t max_seq, uint32_t prefill_cap) {
    if (!e || max_seq == 0u || max_seq > DS4_LING3VL_BANK_MAX ||
        prefill_cap == 0u || prefill_cap > ctx_size) { return NULL; }
    ds4_ling3vl_batch_runtime *rt = xcalloc(1, sizeof(*rt));
    rt->max_seq = max_seq;
    rt->ctx_size = ctx_size;
    rt->prefill_cap = prefill_cap;
    rt->graph = xcalloc(max_seq, sizeof(*rt->graph));
    for (uint32_t b = 0; b < max_seq; b++) {
        if (!ling3vl_graph_alloc(&rt->graph[b], &e->model, &e->weights, ctx_size,
                                 prefill_cap)) {
            ling3vl_batch_runtime_free(rt);
            return NULL;
        }
    }
    rt->bank_logits = xcalloc((size_t)max_seq * DS4_N_VOCAB, sizeof(*rt->bank_logits));
    rt->bank_logits_valid = xcalloc(max_seq, sizeof(*rt->bank_logits_valid));
    ling3vl_ckpt_init(rt);
    return rt;
}

static bool ling3vl_batch_runtime_read_logits(ds4_ling3vl_batch_runtime *rt,
                                              uint32_t bank) {
    if (!rt || bank >= rt->max_seq || ds4_gpu_synchronize() == 0) { return false; }
    float *dst = rt->bank_logits + (size_t)bank * DS4_N_VOCAB;
    const bool ok = ds4_gpu_tensor_read(rt->graph[bank].logits, 0, dst,
                                        (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    rt->bank_logits_valid[bank] = ok ? 1u : 0u;
    return ok;
}

static bool ling3vl_batch_runtime_reset_bank(ds4_ling3vl_batch_runtime *rt,
                                             uint32_t bank) {
    if (!rt || bank >= rt->max_seq) { return false; }
    ling3vl_ckpt_drop(rt, bank);
    rt->bank_logits_valid[bank] = 0u;
    return ling3vl_graph_reset(&rt->graph[bank]);
}

/* Exact fork: the destination inherits the whole committed prefix, which for
 * this family is the live recurrent state plus every latent row. */
static bool ling3vl_batch_runtime_copy_bank(ds4_ling3vl_batch_runtime *rt,
                                            uint32_t src, uint32_t dst,
                                            uint32_t tokens) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || src == dst) { return false; }
    ds4_ling3vl_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (sg->failed || sg->position != tokens || tokens > dg->context) { return false; }
    dg->failed = true;
    if (!ds4_gpu_tensor_copy(dg->state_pool, 0, sg->state_pool, 0, sg->state_bytes)) {
        return false;
    }
    for (uint32_t il = 0; il < LING3VL_LAYERS; il++) {
        if (ds4_ling3vl_layer_is_kda(il)) { continue; }
        if (!ds4_gpu_tensor_copy(dg->latent_cache[il], 0, sg->latent_cache[il], 0,
                                 (uint64_t)tokens * L3V_KV_LORA * sizeof(uint16_t)) ||
            !ds4_gpu_tensor_copy(dg->k_pe_cache[il], 0, sg->k_pe_cache[il], 0,
                                 (uint64_t)tokens * L3V_ROPE * sizeof(uint16_t))) {
            return false;
        }
    }
    if (ds4_gpu_synchronize() == 0) { return false; }
    dg->position = tokens;
    dg->high_water = tokens;
    dg->failed = false;
    if (rt->bank_logits_valid[src]) {
        memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB,
               rt->bank_logits + (size_t)src * DS4_N_VOCAB,
               DS4_N_VOCAB * sizeof(float));
    }
    rt->bank_logits_valid[dst] = rt->bank_logits_valid[src];
    return true;
}

static bool ling3vl_batch_runtime_prefill(ds4_ling3vl_batch_runtime *rt,
                                          ds4_engine *e, uint32_t bank,
                                          const int *tokens, uint32_t rows,
                                          uint32_t pos, bool final) {
    if (!rt || !e || bank >= rt->max_seq) { return false; }
    rt->bank_logits_valid[bank] = 0u;
    if (!ling3vl_forward(&rt->graph[bank], &e->model, &e->weights, tokens, rows, pos)) {
        return false;
    }
    return !final || ling3vl_batch_runtime_read_logits(rt, bank);
}

static bool ling3vl_batch_runtime_decode(ds4_ling3vl_batch_runtime *rt,
                                         ds4_engine *e, const uint32_t *banks,
                                         const int *tokens, const uint32_t *positions,
                                         uint32_t n_rows) {
    /* Each bank carries its own recurrent state, so a cross-bank decode has
     * no shared row to batch: run them in bank order. */
    if (!rt || !e || !banks || !tokens || !positions || !n_rows) { return false; }
    for (uint32_t i = 0; i < n_rows; i++) {
        const uint32_t bank = banks[i];
        if (bank >= rt->max_seq) { return false; }
        rt->bank_logits_valid[bank] = 0u;
        if (!ling3vl_forward(&rt->graph[bank], &e->model, &e->weights, &tokens[i],
                             1u, positions[i]) ||
            !ling3vl_batch_runtime_read_logits(rt, bank)) {
            return false;
        }
    }
    return true;
}
