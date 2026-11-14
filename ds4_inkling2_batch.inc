/* Inkling-2 banks. The scheduler serializes bank forwards under the engine
 * generation lock, so graph 0's scratch serves every bank; each bank owns
 * its full-layer NVFP4 blocks (with its own SSD tier past the resident
 * context), sliding rings and convolution histories. Full-layer blocks are
 * positional and fork straight from the source bank; a ring keeps only the
 * window behind one chunk and the histories only the frontier, so a fork
 * below the frontier restores a checkpoint: the histories and every ring's
 * 1,023 rows below its position (the serial resume copy), in virtual slots
 * funded at capture, at most as many as the memory the fit left (slot_cap). */
enum { INKLING2_BANK_MAX = DS4_MULTISEQ_MAX_SEQ, INKLING2_BANK_CHECKPOINTS = 8, INKLING2_CKPT_ALIGN = 4096 };
typedef enum { INKLING2_MORE_ROWS, INKLING2_FINAL_ROWS } inkling2_prefill_mode;
typedef enum { INKLING2_STRICT_BANKS, INKLING2_FIT_BANKS } inkling2_fit_mode;

typedef struct {
    uint32_t max_seq, ctx_size, prefill_cap, checkpoint_stride, slot_cap;
    ds4_inkling2_graph *graph;
    float *bank_logits, *checkpoint_logits;
    uint8_t *bank_logits_valid;
    ds4_gpu_tensor *checkpoint_slab;
    ds4_partial_checkpoint checkpoint[INKLING2_BANK_CHECKPOINTS];
    uint64_t checkpoint_slot_bytes, checkpoint_clock;
} ds4_inkling2_batch_runtime;

// What one more bank adds beyond graph 0: its K/V, tier and histories.
static uint64_t inkling2_bank_bytes(uint32_t ctx, uint32_t cap) { return inkling2_kv_bytes(ctx, cap); }

// One checkpoint: every layer's histories, then every ring's window rows.
static uint64_t inkling2_ckpt_bytes(void) {
    return (uint64_t)INKLING2_LAYERS * INKLING2_HIST * sizeof(float) +
        (uint64_t)(INKLING2_LAYERS - INKLING2_FULL_LAYERS) * (DS4_INK2_WINDOW - 1u) * INKLING2_KV_ROW;
}

#ifndef __APPLE__
/* Ring slots of the 1,023 positions below `pos` (slots below position 0
 * hold nothing a read uses): at most two spans in position order. */
typedef struct { uint64_t slot, rows; } inkling2_span;

static uint32_t inkling2_ring_spans(uint32_t cap, uint32_t pos, inkling2_span span[2]) {
    const uint32_t rows = DS4_INK2_WINDOW - 1u, first = (pos % cap + cap - rows) % cap;
    const uint32_t head = rows < cap - first ? rows : cap - first;
    span[0] = (inkling2_span){first, head};
    span[1] = (inkling2_span){0u, rows - head};
    return span[1].rows ? 2u : 1u;
}

static bool inkling2_ring_copy(ds4_inkling2_graph *dst, const ds4_inkling2_graph *src, uint32_t il, uint32_t pos) {
    inkling2_span span[2];
    const uint32_t n = inkling2_ring_spans(src->ring_cap, pos, span);
    for (uint32_t s = 0; s < n; s++) {
        const uint64_t at = span[s].slot * INKLING2_KV_ROW;
        if (!ds4_gpu_tensor_copy(dst->kv[il], at, src->kv[il], at, span[s].rows * INKLING2_KV_ROW)) { return false; }
    }
    return true;
}

/* Full-layer blocks [0, ceil(pos / 128)) of `src` into `dst` (one tier
 * geometry): device copies below the resident count, then a host buffer
 * from src's file or window to dst's file or window, dst's window restarting
 * at the block holding pos as a rewind's does. */
static bool inkling2_blocks_copy(ds4_inkling2_graph *dst, ds4_inkling2_graph *src, uint32_t pos) {
    inkling2_kv *sk = &src->tier, *dk = &dst->tier;
    const uint64_t size = DS4_INK2_BLOCK_BYTES;
    const uint32_t blocks = (pos + DS4_INK2_KV_BLOCK - 1u) / DS4_INK2_KV_BLOCK;
    const uint32_t resident = sk->spill && sk->resident < blocks ? sk->resident : blocks;
    for (uint32_t il = DS4_INK2_BLOCK - 1u; resident && il < INKLING2_LAYERS; il += DS4_INK2_BLOCK) {
        if (!ds4_gpu_tensor_copy(dst->kv[il], 0, src->kv[il], 0, (uint64_t)resident * size)) { return false; }
    }
    const uint32_t front = pos / DS4_INK2_KV_BLOCK, per = DS4_SESSION_IO_CHUNK / DS4_INK2_BLOCK_BYTES;
    if (!dk->spill) { return true; }
    inkling2_io_idle(dk);
    dk->window_block0 = front > dk->resident ? front : dk->resident;
    if (resident == blocks) { return true; }
    inkling2_io_idle(sk);
    if (!ds4_gpu_synchronize()) { return false; }
    uint8_t *buf = aligned_alloc(4096, DS4_SESSION_IO_CHUNK);
    bool ok = buf != NULL;
    for (uint32_t l = 0; ok && l < INKLING2_FULL_LAYERS; l++) {
        for (uint32_t b = resident, count = 0; ok && b < blocks; b += count) {
            // A run inside one location on both sides.
            count = blocks - b < per ? blocks - b : per;
            if (b < sk->window_block0 && count > sk->window_block0 - b) { count = sk->window_block0 - b; }
            if (b < dk->window_block0 && count > dk->window_block0 - b) { count = dk->window_block0 - b; }
            const uint64_t bytes = (uint64_t)count * size;
            if (b < sk->window_block0) {
                ok = inkling2_kv_io(sk, false, buf, bytes, inkling2_file_offset(sk, l, b));
            } else {
                ok = b + count - sk->window_block0 <= sk->window_blocks;
                if (ok) { memcpy(buf, sk->window + ((uint64_t)l * sk->window_blocks + b - sk->window_block0) * size, bytes); }
            }
            if (ok && b < dk->window_block0) {
                ok = inkling2_kv_io(dk, true, buf, bytes, inkling2_file_offset(dk, l, b));
            } else if (ok) {
                ok = b + count - dk->window_block0 <= dk->window_blocks;
                if (ok) { memcpy(dk->window + ((uint64_t)l * dk->window_blocks + b - dk->window_block0) * size, buf, bytes); }
            }
        }
    }
    free(buf);
    return ok;
}

// The histories and the rings' window rows of `g` at `pos` to (SAVE) or
// from (LOAD) a checkpoint slot.
static bool inkling2_ckpt_rows(ds4_inkling2_batch_runtime *rt, ds4_inkling2_graph *g, uint32_t slot, uint32_t pos,
        inkling2_payload_dir dir) {
    const uint64_t hist = (uint64_t)INKLING2_HIST * sizeof(float);
    uint64_t at = (uint64_t)slot * rt->checkpoint_slot_bytes;
    for (uint32_t il = 0; il < INKLING2_LAYERS; il++, at += hist) {
        const bool ok = dir == INKLING2_PAYLOAD_SAVE
            ? ds4_gpu_tensor_copy(rt->checkpoint_slab, at, g->hist[il], 0, hist)
            : ds4_gpu_tensor_copy(g->hist[il], 0, rt->checkpoint_slab, at, hist);
        if (!ok) { return false; }
    }
    for (uint32_t il = 0; il < INKLING2_LAYERS; il++) {
        if (inkling2_global(il)) { continue; }
        inkling2_span span[2];
        const uint32_t n = inkling2_ring_spans(g->ring_cap, pos, span);
        for (uint32_t s = 0; s < n; s++) {
            const uint64_t ring = span[s].slot * INKLING2_KV_ROW, bytes = span[s].rows * INKLING2_KV_ROW;
            const bool ok = dir == INKLING2_PAYLOAD_SAVE
                ? ds4_gpu_tensor_copy(rt->checkpoint_slab, at, g->kv[il], ring, bytes)
                : ds4_gpu_tensor_copy(g->kv[il], ring, rt->checkpoint_slab, at, bytes);
            if (!ok) { return false; }
            at += bytes;
        }
    }
    return true;
}

static void inkling2_ckpt_drop(ds4_inkling2_batch_runtime *rt, uint32_t bank) {
    for (uint32_t i = 0; i < INKLING2_BANK_CHECKPOINTS; i++) {
        if (partial_checkpoint_ref(&rt->checkpoint[i], bank)) { partial_checkpoint_clear_ref(&rt->checkpoint[i], bank); }
    }
}

static void inkling2_ckpt_inherit(ds4_inkling2_batch_runtime *rt, uint32_t src, uint32_t dst, uint32_t cut) {
    if (src != dst) { inkling2_ckpt_drop(rt, dst); }
    for (uint32_t i = 0; i < INKLING2_BANK_CHECKPOINTS; i++) {
        ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (!partial_checkpoint_ref(cp, src)) { continue; }
        if (cp->pos <= cut) { partial_checkpoint_set_ref(cp, dst); }
        else if (src == dst) { partial_checkpoint_clear_ref(cp, dst); }
    }
}

static int inkling2_ckpt_find(ds4_inkling2_batch_runtime *rt, uint32_t bank, uint32_t cut, uint32_t request_len) {
    int best = -1;
    for (uint32_t i = 0; i < INKLING2_BANK_CHECKPOINTS; i++) {
        const ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (!cp->pos || cp->pos > cut || !partial_checkpoint_ref(cp, bank) ||
            (cp->pos == request_len && !cp->logits_valid)) { continue; }
        if (best < 0 || cp->pos > rt->checkpoint[best].pos) { best = (int)i; }
    }
    return best;
}

static bool inkling2_ckpt_capture(ds4_inkling2_batch_runtime *rt, uint32_t bank, uint32_t pos, bool logits,
        uint64_t reserve) {
    if (!rt || !rt->checkpoint_slab || bank >= rt->max_seq || !pos || rt->graph[bank].failed ||
        rt->graph[bank].position != pos) { return false; }
    float *bank_logits = rt->bank_logits + (size_t)bank * DS4_N_VOCAB;
    uint32_t slot = 0;
    for (uint32_t i = 0; i < INKLING2_BANK_CHECKPOINTS; i++) {
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
    for (uint32_t i = 0; i < INKLING2_BANK_CHECKPOINTS; i++) {
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
    if (!inkling2_ckpt_rows(rt, &rt->graph[bank], slot, pos, INKLING2_PAYLOAD_SAVE) || !ds4_gpu_synchronize()) {
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

/* Resume bank dst at checkpoint `slot` of src: the full-layer blocks below
 * its position (src's, or dst's own window rewound in place), then the
 * slot's histories and ring rows. */
static bool inkling2_ckpt_restore(ds4_inkling2_batch_runtime *rt, uint32_t src, uint32_t dst, uint32_t slot,
        uint32_t cut, uint32_t *out) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || slot >= INKLING2_BANK_CHECKPOINTS) { return false; }
    ds4_partial_checkpoint *cp = &rt->checkpoint[slot];
    const uint32_t pos = cp->pos;
    ds4_inkling2_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (!pos || pos > cut || pos > sg->position || sg->failed || !partial_checkpoint_ref(cp, src)) { return false; }
    dg->failed = true;
    rt->bank_logits_valid[dst] = 0;
    const bool rows = src == dst ? inkling2_rewind(dg, pos) : inkling2_blocks_copy(dg, sg, pos);
    if (!rows || !inkling2_ckpt_rows(rt, dg, slot, pos, INKLING2_PAYLOAD_LOAD) || !ds4_gpu_synchronize()) {
        return false;
    }
    dg->position = pos;
    dg->high = pos;
    dg->failed = false;
    if (cp->logits_valid) {
        memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB, rt->checkpoint_logits + (size_t)slot * DS4_N_VOCAB,
               DS4_N_VOCAB * sizeof(float));
        rt->bank_logits_valid[dst] = 1;
    }
    cp->last_use = ++rt->checkpoint_clock;
    inkling2_ckpt_inherit(rt, src, dst, pos);
    if (out) { *out = pos; }
    return true;
}

static uint64_t inkling2_ckpt_trim(ds4_inkling2_batch_runtime *rt, uint64_t want) {
    uint64_t freed = 0;
    if (!rt || !rt->checkpoint_slab || !want || !ds4_gpu_synchronize()) { return 0; }
    for (uint32_t i = 0; i < INKLING2_BANK_CHECKPOINTS && freed < want;) {
        if (rt->checkpoint[i].pos) { i++; continue; }
        const uint64_t start = (uint64_t)i * rt->checkpoint_slot_bytes;
        do { i++; } while (i < INKLING2_BANK_CHECKPOINTS && !rt->checkpoint[i].pos);
        const uint64_t end = i == INKLING2_BANK_CHECKPOINTS
            ? ds4_gpu_tensor_bytes(rt->checkpoint_slab) : (uint64_t)i * rt->checkpoint_slot_bytes;
        freed += ds4_gpu_tensor_trim(rt->checkpoint_slab, start, end - start);
    }
    return freed;
}

static void inkling2_batch_free(ds4_inkling2_batch_runtime *rt) {
    if (!rt) { return; }
    // Banks borrow graph 0's buffers: free them first.
    for (uint32_t b = rt->max_seq; rt->graph && b > 0; b--) { inkling2_graph_free(&rt->graph[b - 1u]); }
    free(rt->graph);
    free(rt->bank_logits);
    free(rt->bank_logits_valid);
    ds4_gpu_tensor_free(rt->checkpoint_slab);
    free(rt->checkpoint_logits);
    free(rt);
}

static void inkling2_ckpt_init(ds4_inkling2_batch_runtime *rt, uint32_t slots) {
    const char *off = getenv("DS4_SERVER_FORK_PARTIAL");
    if ((off && strcmp(off, "0") == 0) || !slots) { return; }
    rt->slot_cap = slots < INKLING2_BANK_CHECKPOINTS ? slots : INKLING2_BANK_CHECKPOINTS;
    rt->checkpoint_slot_bytes = inkling2_ckpt_bytes();
    uint64_t bytes = rt->checkpoint_slot_bytes * INKLING2_BANK_CHECKPOINTS;
    const uint64_t page = ds4_gpu_vmm_demand_page();
    if (page) { bytes = (bytes + page - 1u) / page * page; }
    rt->checkpoint_slab = ds4_gpu_tensor_reserve(bytes);
    if (!rt->checkpoint_slab) { return; }
    rt->checkpoint_logits = xcalloc((size_t)INKLING2_BANK_CHECKPOINTS * DS4_N_VOCAB, sizeof(float));
    const uint32_t stride = (rt->ctx_size + DS4_PARTIAL_PERIODIC_TARGET - 1u) / DS4_PARTIAL_PERIODIC_TARGET;
    rt->checkpoint_stride = (stride + INKLING2_CKPT_ALIGN - 1u) / INKLING2_CKPT_ALIGN * INKLING2_CKPT_ALIGN;
}

static ds4_inkling2_batch_runtime *inkling2_batch_create(ds4_engine *e, uint32_t ctx, uint32_t max_seq,
        uint32_t cap, uint32_t slots) {
    if (!e || !ctx || !max_seq || max_seq > INKLING2_BANK_MAX || !cap || cap > ctx) { return NULL; }
    ds4_inkling2_batch_runtime *rt = xcalloc(1, sizeof(*rt));
    rt->ctx_size = ctx;
    rt->max_seq = max_seq;
    rt->prefill_cap = cap;
    rt->graph = xcalloc(max_seq, sizeof(*rt->graph));
    if (!inkling2_graph_alloc(&rt->graph[0], &e->model, &e->weights, ctx, cap)) {
        inkling2_batch_free(rt);
        return NULL;
    }
    for (uint32_t b = 1; b < max_seq; b++) {
        if (!inkling2_graph_bank(&rt->graph[b], &rt->graph[0])) {
            inkling2_batch_free(rt);
            return NULL;
        }
    }
    rt->bank_logits = xcalloc((size_t)max_seq * DS4_N_VOCAB, sizeof(float));
    rt->bank_logits_valid = xcalloc(max_seq, sizeof(uint8_t));
    inkling2_ckpt_init(rt, slots);
    return rt;
}

// Logits past the tokenizer's rows are padding (masked by the head).
static bool inkling2_bank_read_logits(ds4_inkling2_batch_runtime *rt, uint32_t bank) {
    float *dst = rt->bank_logits + (size_t)bank * DS4_N_VOCAB;
    bool ok = ds4_gpu_tensor_read(rt->graph[bank].logits, 0, dst, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    for (uint32_t i = 0; ok && i < DS4_INK2_TOKENS; i++) { ok = isfinite(dst[i]); }
    rt->bank_logits_valid[bank] = ok;
    if (!ok) { rt->graph[bank].failed = true; }
    return ok;
}

static void inkling2_bank_reset(ds4_inkling2_batch_runtime *rt, uint32_t bank) {
    inkling2_ckpt_drop(rt, bank);
    rt->bank_logits_valid[bank] = 0;
    (void)inkling2_graph_reset(&rt->graph[bank]);
}

// Exact fork at the source's frontier: its blocks, ring windows, histories
// and logits.
static bool inkling2_bank_copy(ds4_inkling2_batch_runtime *rt, uint32_t src, uint32_t dst, uint32_t tokens) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || src == dst) { return false; }
    ds4_inkling2_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (sg->failed || sg->position != tokens || tokens > dg->ctx_cap) { return false; }
    dg->failed = true;
    bool ok = inkling2_blocks_copy(dg, sg, tokens);
    for (uint32_t il = 0; ok && il < INKLING2_LAYERS; il++) {
        ok = ds4_gpu_tensor_copy(dg->hist[il], 0, sg->hist[il], 0, (uint64_t)INKLING2_HIST * sizeof(float)) &&
            (inkling2_global(il) || inkling2_ring_copy(dg, sg, il, tokens));
    }
    if (!ok || !ds4_gpu_synchronize()) { return false; }
    dg->position = tokens;
    dg->high = tokens;
    dg->failed = false;
    memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB, rt->bank_logits + (size_t)src * DS4_N_VOCAB,
           DS4_N_VOCAB * sizeof(float));
    rt->bank_logits_valid[dst] = rt->bank_logits_valid[src];
    inkling2_ckpt_inherit(rt, src, dst, tokens);
    return true;
}

static bool inkling2_bank_prefill(ds4_inkling2_batch_runtime *rt, ds4_engine *e, uint32_t bank, const int *tokens,
        uint32_t rows, uint32_t pos, inkling2_prefill_mode mode) {
    if (!rt || !e || bank >= rt->max_seq) { return false; }
    rt->bank_logits_valid[bank] = 0;
    const inkling2_head_rows head = mode == INKLING2_FINAL_ROWS ? INKLING2_HEAD_LAST : INKLING2_HEAD_NONE;
    if (!inkling2_forward(&rt->graph[bank], &e->model, &e->weights, tokens, rows, pos, head)) { return false; }
    return mode == INKLING2_MORE_ROWS || inkling2_bank_read_logits(rt, bank);
}

/* Each bank attends over its own K/V, so decode rows of different banks
 * share no forward: run them in bank order. */
static bool inkling2_bank_decode(ds4_inkling2_batch_runtime *rt, ds4_engine *e, const uint32_t *banks,
        const int *tokens, const uint32_t *positions, uint32_t rows) {
    if (!rt || !e || !banks || !tokens || !positions || !rows) { return false; }
    for (uint32_t i = 0; i < rows; i++) {
        if (banks[i] >= rt->max_seq ||
            !inkling2_bank_prefill(rt, e, banks[i], &tokens[i], 1u, positions[i], INKLING2_FINAL_ROWS)) {
            return false;
        }
    }
    return true;
}
#else
// Inkling 2 is CUDA-only; Metal keeps linkage and creates no banks.
static ds4_inkling2_batch_runtime *inkling2_batch_create(ds4_engine *e, uint32_t ctx, uint32_t max_seq,
        uint32_t cap, uint32_t slots) {
    (void)e; (void)ctx; (void)max_seq; (void)cap; (void)slots;
    return NULL;
}
static void inkling2_batch_free(ds4_inkling2_batch_runtime *rt) { free(rt); }
static void inkling2_ckpt_drop(ds4_inkling2_batch_runtime *rt, uint32_t bank) { (void)rt; (void)bank; }
static int inkling2_ckpt_find(ds4_inkling2_batch_runtime *rt, uint32_t bank, uint32_t cut, uint32_t request_len) {
    (void)rt; (void)bank; (void)cut; (void)request_len;
    return -1;
}
static bool inkling2_ckpt_capture(ds4_inkling2_batch_runtime *rt, uint32_t bank, uint32_t pos, bool logits,
        uint64_t reserve) {
    (void)rt; (void)bank; (void)pos; (void)logits; (void)reserve;
    return false;
}
static bool inkling2_ckpt_restore(ds4_inkling2_batch_runtime *rt, uint32_t src, uint32_t dst, uint32_t slot,
        uint32_t cut, uint32_t *out) {
    (void)rt; (void)src; (void)dst; (void)slot; (void)cut; (void)out;
    return false;
}
static uint64_t inkling2_ckpt_trim(ds4_inkling2_batch_runtime *rt, uint64_t want) { (void)rt; (void)want; return 0; }
static void inkling2_bank_reset(ds4_inkling2_batch_runtime *rt, uint32_t bank) { (void)rt; (void)bank; }
static bool inkling2_bank_copy(ds4_inkling2_batch_runtime *rt, uint32_t src, uint32_t dst, uint32_t tokens) {
    (void)rt; (void)src; (void)dst; (void)tokens;
    return false;
}
static bool inkling2_bank_prefill(ds4_inkling2_batch_runtime *rt, ds4_engine *e, uint32_t bank, const int *tokens,
        uint32_t rows, uint32_t pos, inkling2_prefill_mode mode) {
    (void)rt; (void)e; (void)bank; (void)tokens; (void)rows; (void)pos; (void)mode;
    return false;
}
static bool inkling2_bank_decode(ds4_inkling2_batch_runtime *rt, ds4_engine *e, const uint32_t *banks,
        const int *tokens, const uint32_t *positions, uint32_t rows) {
    (void)rt; (void)e; (void)banks; (void)tokens; (void)positions; (void)rows;
    return false;
}
#endif
