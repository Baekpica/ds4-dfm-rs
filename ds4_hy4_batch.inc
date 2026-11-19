/* The common continuous scheduler serializes the forwards on one stream.
 * Every bank owns its position-indexed state and its host frontier logits. */
typedef struct {
    uint32_t max_seq, prefill_cap;
    ds4_hy4_graph *graph;
    float *bank_logits;
    uint8_t *bank_logits_valid;
} ds4_hy4_batch_runtime;

static void hy4_batch_free(ds4_hy4_batch_runtime *rt) {
    if (!rt) {
        return;
    }
    for (uint32_t i = rt->max_seq; rt->graph && i; i--) {
        hy4_graph_free(&rt->graph[i - 1]);
    }
    free(rt->graph);
    free(rt->bank_logits);
    free(rt->bank_logits_valid);
    free(rt);
}
static ds4_hy4_batch_runtime *hy4_batch_create(uint32_t ctx, uint32_t banks, uint32_t cap) {
    if (!banks || banks > DS4_MULTISEQ_MAX_SEQ) {
        return NULL;
    }
    ds4_hy4_batch_runtime *rt = xcalloc(1, sizeof(*rt));
    rt->max_seq = banks;
    rt->prefill_cap = cap;
    rt->graph = xcalloc(banks, sizeof(*rt->graph));
    for (uint32_t i = 0; i < banks; i++) {
        if (!hy4_graph_alloc(&rt->graph[i], ctx, cap, i ? &rt->graph[0] : NULL)) {
            hy4_batch_free(rt);
            return NULL;
        }
    }
    rt->bank_logits = xcalloc((size_t)banks * 120832, sizeof(float));
    rt->bank_logits_valid = xcalloc(banks, sizeof(*rt->bank_logits_valid));
    return rt;
}
static void hy4_bank_reset(ds4_hy4_batch_runtime *rt, uint32_t b) {
    rt->bank_logits_valid[b] = 0;
    hy4_reset(&rt->graph[b]);
}
static bool hy4_bank_keeps(const ds4_hy4_batch_runtime *rt, uint32_t b, uint32_t pos) {
    const ds4_hy4_graph *g = &rt->graph[b];
    const uint32_t ring = hy4_ring(g->context, g->cap);
    return g->ready && !g->failed && pos && pos <= g->position && pos >= g->restored &&
           (g->high <= ring || g->high - pos <= ring - HY4_WINDOW);
}

static bool hy4_bank_transfer(ds4_hy4_batch_runtime *rt, uint32_t src, uint32_t dst, uint32_t pos,
                              bool exact) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || !hy4_bank_keeps(rt, src, pos)) {
        return false;
    }
    ds4_hy4_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (exact && (sg->position != pos || !rt->bank_logits_valid[src])) {
        return false;
    }
    if (src != dst) {
        dg->failed = true;
        rt->bank_logits_valid[dst] = 0;
        hy4_state_io io = {.op = HY4_COPY};
        if (hy4_state_transfer(&io, sg, dg, pos) || !ds4_gpu_synchronize()) {
            return false;
        }
        dg->high = dg->restored = pos;
        dg->failed = false;
        if (exact) {
            memcpy(rt->bank_logits + (size_t)dst * 120832, rt->bank_logits + (size_t)src * 120832,
                   120832 * sizeof(float));
        }
    }
    dg->position = pos;
    rt->bank_logits_valid[dst] = exact;
    return true;
}

static bool hy4_bank_forward(ds4_hy4_batch_runtime *rt, ds4_engine *e, uint32_t b, const int *prefix,
                             const int *tokens, uint32_t n, uint32_t pos, bool final) {
    if (!rt || b >= rt->max_seq) {
        return false;
    }
    ds4_hy4_graph *g = &rt->graph[b];
    const uint32_t prev = pos < 2 ? pos : 2;
    rt->bank_logits_valid[b] = 0;
    if (!hy4_forward(g, e, prefix + pos - prev, prev, tokens, n, pos, final)) {
        g->failed = true;
        return false;
    }
    if (!final) {
        return true;
    }
    float *logits = rt->bank_logits + (size_t)b * 120832;
    if (!ds4_gpu_tensor_read(g->logits, 0, logits, 120832 * sizeof(float))) {
        g->failed = true;
        return false;
    }
    for (uint32_t i = 0; i < 120832; i++) {
        if (!isfinite(logits[i])) {
            g->failed = true;
            return false;
        }
    }
    rt->bank_logits_valid[b] = 1;
    return true;
}
