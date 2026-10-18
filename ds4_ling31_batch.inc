/* Ling-3.1 banks. The scheduler serializes bank forwards under the engine
 * generation lock, so graph 0's scratch, controls and MTP verify buffers
 * serve every bank; each bank owns its state pool (KDA tiles, convolution
 * tails, last hidden) and latent rows. Partial reuse snapshots the state
 * pool into virtual checkpoint slots funded at capture, at most as many as
 * the memory the fit left (slot_cap); latent rows are append-only and fork
 * straight from the source bank. */
enum { L31_BANK_MAX = DS4_MULTISEQ_MAX_SEQ, L31_BANK_CHECKPOINTS = 8, L31_CKPT_ALIGN = 4096 };
typedef enum { L31_MORE_ROWS, L31_FINAL_ROWS } ling31_prefill_mode;
typedef enum { L31_STRICT_BANKS, L31_FIT_BANKS } ling31_fit_mode;

typedef struct {
    uint32_t max_seq, ctx_size, prefill_cap, checkpoint_stride, slot_cap;
    ds4_ling31_graph *graph;
    float *bank_logits, *checkpoint_logits;
    uint8_t *bank_logits_valid;
    ds4_gpu_tensor *checkpoint_slab;
    ds4_partial_checkpoint checkpoint[L31_BANK_CHECKPOINTS];
    uint64_t checkpoint_slot_bytes, checkpoint_clock;
} ds4_ling31_batch_runtime;

// What one more bank adds beyond graph 0: its state pool and latent rows.
static uint64_t ling31_bank_bytes(uint32_t ctx, uint32_t drafts) {
    uint32_t kda, mla;
    ling31_layers(&kda, &mla);
    return ling31_state_bytes() + (uint64_t)(mla + (drafts ? 1u : 0u)) * ling31_latent_bytes(ctx);
}

static void ling31_ckpt_drop(ds4_ling31_batch_runtime *rt, uint32_t bank) {
    for (uint32_t i = 0; i < L31_BANK_CHECKPOINTS; i++) {
        if (partial_checkpoint_ref(&rt->checkpoint[i], bank)) {
            partial_checkpoint_clear_ref(&rt->checkpoint[i], bank);
        }
    }
}

static void ling31_ckpt_inherit(ds4_ling31_batch_runtime *rt, uint32_t src, uint32_t dst, uint32_t cut) {
    if (src != dst) { ling31_ckpt_drop(rt, dst); }
    for (uint32_t i = 0; i < L31_BANK_CHECKPOINTS; i++) {
        ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (!partial_checkpoint_ref(cp, src)) { continue; }
        if (cp->pos <= cut) { partial_checkpoint_set_ref(cp, dst); }
        else if (src == dst) { partial_checkpoint_clear_ref(cp, dst); }
    }
}

static int ling31_ckpt_find(ds4_ling31_batch_runtime *rt, uint32_t bank, uint32_t cut, uint32_t request_len) {
    int best = -1;
    for (uint32_t i = 0; i < L31_BANK_CHECKPOINTS; i++) {
        const ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (!cp->pos || cp->pos > cut || !partial_checkpoint_ref(cp, bank) ||
            (cp->pos == request_len && !cp->logits_valid)) { continue; }
        if (best < 0 || cp->pos > rt->checkpoint[best].pos) { best = (int)i; }
    }
    return best;
}

// Rows [0, n) of one layer's latent cache (FP8: bytes and row exponents)
// and rope keys; every graph shares ctx_cap and the latent layout.
static bool ling31_latent_rows(ds4_ling31_graph *dst, uint32_t dl, const ds4_ling31_graph *src,
        uint32_t sl, uint32_t n) {
    const bool fp8 = dst->latent_mode == DS4_L31_LATENT_FP8;
    const uint64_t row = fp8 ? L31_KV_LORA : L31_KV_LORA * sizeof(uint16_t);
    const uint64_t exps = (uint64_t)dst->ctx_cap * L31_KV_LORA;
    return ds4_gpu_tensor_copy(dst->latent[dl], 0, src->latent[sl], 0, (uint64_t)n * row) &&
        (!fp8 || ds4_gpu_tensor_copy(dst->latent[dl], exps, src->latent[sl], exps, n)) &&
        ds4_gpu_tensor_copy(dst->k_pe[dl], 0, src->k_pe[sl], 0, (uint64_t)n * L31_ROPE * sizeof(uint16_t));
}

// Latent rows [0, rows) of every MLA layer (the draft layer's [0, rows-1)).
static bool ling31_latent_copy(ds4_ling31_graph *dst, const ds4_ling31_graph *src, uint32_t rows) {
    const uint32_t layers = dst->mtp_drafts ? LING31_LAYERS + 1u : LING31_LAYERS;
    for (uint32_t il = 0; il < layers; il++) {
        if (!ds4_ling31_layer_is_mla(il)) { continue; }
        const uint32_t n = il == LING31_MTP ? (rows ? rows - 1u : 0u) : rows;
        if (n && !ling31_latent_rows(dst, il, src, il, n)) { return false; }
    }
    return true;
}

static void ling31_frontier(ds4_ling31_graph *g, uint32_t pos) {
    g->position = pos;
    g->mtp_len = g->mtp_drafts && pos ? pos - 1u : 0u;
    g->mtp_n = 0u;
    g->failed = false;
}

static bool ling31_ckpt_capture(ds4_ling31_batch_runtime *rt, uint32_t bank, uint32_t pos,
        bool logits, uint64_t reserve) {
    if (!rt || !rt->checkpoint_slab || bank >= rt->max_seq || !pos ||
        rt->graph[bank].failed || rt->graph[bank].mtp_n || rt->graph[bank].position != pos) { return false; }
    float *bank_logits = rt->bank_logits + (size_t)bank * DS4_N_VOCAB;
    uint32_t slot = 0;
    for (uint32_t i = 0; i < L31_BANK_CHECKPOINTS; i++) {
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
    for (uint32_t i = 0; i < L31_BANK_CHECKPOINTS; i++) {
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
    const uint64_t base = (uint64_t)slot * bytes;
    if (!ds4_gpu_tensor_ensure(rt->checkpoint_slab, base, bytes)) { return false; }
    // Invalidate first: a failed copy must not leave lineage on a torn slot.
    ds4_partial_checkpoint *cp = &rt->checkpoint[slot];
    memset(cp, 0, sizeof(*cp));
    if (!ds4_gpu_tensor_copy(rt->checkpoint_slab, base, rt->graph[bank].state_pool, 0,
            rt->checkpoint_slot_bytes) || !ds4_gpu_synchronize()) { return false; }
    cp->pos = pos;
    cp->last_use = ++rt->checkpoint_clock;
    partial_checkpoint_set_ref(cp, bank);
    if (logits && rt->bank_logits_valid[bank]) {
        memcpy(rt->checkpoint_logits + (size_t)slot * DS4_N_VOCAB, bank_logits, DS4_N_VOCAB * sizeof(float));
        cp->logits_valid = 1;
    }
    return true;
}

static bool ling31_ckpt_restore(ds4_ling31_batch_runtime *rt, uint32_t src, uint32_t dst,
        uint32_t slot, uint32_t cut, uint32_t *out) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || slot >= L31_BANK_CHECKPOINTS) { return false; }
    ds4_partial_checkpoint *cp = &rt->checkpoint[slot];
    const uint32_t pos = cp->pos;
    ds4_ling31_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (!pos || pos > cut || pos > sg->position || sg->failed || sg->mtp_n ||
        !partial_checkpoint_ref(cp, src)) { return false; }
    dg->failed = true;
    rt->bank_logits_valid[dst] = 0;
    if (!ds4_gpu_tensor_copy(dg->state_pool, 0, rt->checkpoint_slab,
            (uint64_t)slot * rt->checkpoint_slot_bytes, rt->checkpoint_slot_bytes) ||
        (src != dst && !ling31_latent_copy(dg, sg, pos)) || !ds4_gpu_synchronize()) { return false; }
    ling31_frontier(dg, pos);
    if (cp->logits_valid) {
        memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB,
               rt->checkpoint_logits + (size_t)slot * DS4_N_VOCAB, DS4_N_VOCAB * sizeof(float));
        rt->bank_logits_valid[dst] = 1;
    }
    cp->last_use = ++rt->checkpoint_clock;
    ling31_ckpt_inherit(rt, src, dst, pos);
    if (out) { *out = pos; }
    return true;
}

static uint64_t ling31_ckpt_trim(ds4_ling31_batch_runtime *rt, uint64_t want) {
    uint64_t freed = 0;
    if (!rt || !rt->checkpoint_slab || !want || !ds4_gpu_synchronize()) { return 0; }
    for (uint32_t i = 0; i < L31_BANK_CHECKPOINTS && freed < want;) {
        if (rt->checkpoint[i].pos) { i++; continue; }
        const uint64_t start = (uint64_t)i * rt->checkpoint_slot_bytes;
        do { i++; } while (i < L31_BANK_CHECKPOINTS && !rt->checkpoint[i].pos);
        const uint64_t end = i == L31_BANK_CHECKPOINTS
            ? ds4_gpu_tensor_bytes(rt->checkpoint_slab) : (uint64_t)i * rt->checkpoint_slot_bytes;
        freed += ds4_gpu_tensor_trim(rt->checkpoint_slab, start, end - start);
    }
    return freed;
}

static void ling31_batch_free(ds4_ling31_batch_runtime *rt) {
    if (!rt) { return; }
    // Banks borrow graph 0's buffers: free them first.
    for (uint32_t b = rt->max_seq; rt->graph && b > 0; b--) { ling31_graph_free(&rt->graph[b - 1u]); }
    free(rt->graph);
    free(rt->bank_logits);
    free(rt->bank_logits_valid);
    ds4_gpu_tensor_free(rt->checkpoint_slab);
    free(rt->checkpoint_logits);
    free(rt);
}

static void ling31_ckpt_init(ds4_ling31_batch_runtime *rt, uint32_t slots) {
    const char *off = getenv("DS4_SERVER_FORK_PARTIAL");
    if ((off && strcmp(off, "0") == 0) || !slots) { return; }
    rt->slot_cap = slots < L31_BANK_CHECKPOINTS ? slots : L31_BANK_CHECKPOINTS;
    rt->checkpoint_slot_bytes = rt->graph[0].state_bytes;
    uint64_t bytes = rt->checkpoint_slot_bytes * L31_BANK_CHECKPOINTS;
    const uint64_t page = ds4_gpu_vmm_demand_page();
    if (page) { bytes = (bytes + page - 1u) / page * page; }
    rt->checkpoint_slab = ds4_gpu_tensor_reserve(bytes);
    if (!rt->checkpoint_slab) { return; }
    rt->checkpoint_logits = xcalloc((size_t)L31_BANK_CHECKPOINTS * DS4_N_VOCAB, sizeof(float));
    const uint32_t stride = (rt->ctx_size + DS4_PARTIAL_PERIODIC_TARGET - 1u) / DS4_PARTIAL_PERIODIC_TARGET;
    rt->checkpoint_stride = (stride + L31_CKPT_ALIGN - 1u) / L31_CKPT_ALIGN * L31_CKPT_ALIGN;
}

static ds4_ling31_batch_runtime *ling31_batch_create(ds4_engine *e, uint32_t ctx, uint32_t max_seq,
        uint32_t cap, uint32_t drafts, uint32_t slots) {
    if (!e || !ctx || !max_seq || max_seq > L31_BANK_MAX || !cap || cap > ctx) { return NULL; }
    ds4_ling31_batch_runtime *rt = xcalloc(1, sizeof(*rt));
    rt->ctx_size = ctx;
    rt->max_seq = max_seq;
    rt->prefill_cap = cap;
    rt->graph = xcalloc(max_seq, sizeof(*rt->graph));
    if (!ling31_graph_alloc(&rt->graph[0], &e->model, &e->weights, ctx, cap, drafts)) {
        ling31_batch_free(rt);
        return NULL;
    }
    for (uint32_t b = 1; b < max_seq; b++) {
        if (!ling31_graph_bank(&rt->graph[b], &rt->graph[0])) {
            ling31_batch_free(rt);
            return NULL;
        }
    }
    rt->bank_logits = xcalloc((size_t)max_seq * DS4_N_VOCAB, sizeof(float));
    rt->bank_logits_valid = xcalloc(max_seq, sizeof(uint8_t));
    ling31_ckpt_init(rt, slots);
    return rt;
}

static bool ling31_bank_read_logits(ds4_ling31_batch_runtime *rt, uint32_t bank) {
    float *dst = rt->bank_logits + (size_t)bank * DS4_N_VOCAB;
    bool ok = ds4_gpu_tensor_read(rt->graph[bank].logits, 0, dst, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    for (uint32_t i = 0; ok && i < DS4_N_VOCAB; i++) { ok = isfinite(dst[i]); }
    rt->bank_logits_valid[bank] = ok;
    if (!ok) { rt->graph[bank].failed = true; }
    return ok;
}

static void ling31_bank_reset(ds4_ling31_batch_runtime *rt, uint32_t bank) {
    ling31_ckpt_drop(rt, bank);
    rt->bank_logits_valid[bank] = 0;
    (void)ling31_graph_reset(&rt->graph[bank]);
}

// Exact fork: the live state pool plus every latent row of the prefix.
static bool ling31_bank_copy(ds4_ling31_batch_runtime *rt, uint32_t src, uint32_t dst, uint32_t tokens) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || src == dst) { return false; }
    ds4_ling31_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (sg->failed || sg->mtp_n || sg->position != tokens || tokens > dg->ctx_cap) { return false; }
    dg->failed = true;
    if (!ds4_gpu_tensor_copy(dg->state_pool, 0, sg->state_pool, 0, sg->state_bytes) ||
        !ling31_latent_copy(dg, sg, tokens) || !ds4_gpu_synchronize()) { return false; }
    ling31_frontier(dg, tokens);
    memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB, rt->bank_logits + (size_t)src * DS4_N_VOCAB,
           DS4_N_VOCAB * sizeof(float));
    rt->bank_logits_valid[dst] = rt->bank_logits_valid[src];
    ling31_ckpt_inherit(rt, src, dst, tokens);
    return true;
}

static bool ling31_bank_prefill(ds4_ling31_batch_runtime *rt, ds4_engine *e, uint32_t bank,
        const int *tokens, uint32_t rows, uint32_t pos, ling31_prefill_mode mode) {
    if (!rt || !e || bank >= rt->max_seq) { return false; }
    rt->bank_logits_valid[bank] = 0;
    const ling31_head_mode head = mode == L31_FINAL_ROWS ? LING31_HEAD_LAST : LING31_HEAD_SKIP;
    if (!ling31_forward(&rt->graph[bank], &e->model, &e->weights, tokens, rows, pos, head)) { return false; }
    return mode == L31_MORE_ROWS || ling31_bank_read_logits(rt, bank);
}

/* Each bank carries its own recurrent state, so decode rows of different
 * banks share no forward: run them in bank order. */
static bool ling31_bank_decode(ds4_ling31_batch_runtime *rt, ds4_engine *e, const uint32_t *banks,
        const int *tokens, const uint32_t *positions, uint32_t rows) {
    if (!rt || !e || !banks || !tokens || !positions || !rows) { return false; }
    for (uint32_t i = 0; i < rows; i++) {
        if (banks[i] >= rt->max_seq ||
            !ling31_bank_prefill(rt, e, banks[i], &tokens[i], 1u, positions[i], L31_FINAL_ROWS)) { return false; }
    }
    return true;
}
