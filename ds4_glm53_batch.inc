#if !defined(GLM53_BATCH_API) && !defined(GLM53_BATCH_CREATE)
/* The scheduler serializes bank forwards under the engine generation lock.
 * Projection scratch and the streamed expert cache are shared; every mutable
 * frontier (KDA, pool tails, latent history and predictor carry) is bank-owned. */
enum { GLM53_BANK_CHECKPOINTS = 8, GLM53_CKPT_ALIGN = 4096 };
typedef enum { GLM53_MORE_ROWS, GLM53_FINAL_ROWS, GLM53_DECODE_ROW } glm53_prefill_mode;
typedef enum { GLM53_NO_LOGITS, GLM53_HAS_LOGITS } glm53_logits_mode;
typedef enum { GLM53_STRICT_BANKS, GLM53_FIT_BANKS } glm53_fit_mode;
typedef enum { GLM53_BANK_MTP_OFF, GLM53_BANK_MTP_ON } glm53_bank_mtp;

typedef struct {
    uint32_t max_seq, ctx_size, prefill_cap, checkpoint_stride;
    ds4_glm53_gpu_graph *graph;
    float *bank_logits, *checkpoint_logits;
    uint8_t *bank_logits_valid, *failed;
    ds4_gpu_tensor *checkpoint_slab;
    ds4_partial_checkpoint checkpoint[GLM53_BANK_CHECKPOINTS];
    uint64_t checkpoint_slot_bytes, checkpoint_clock;
} ds4_glm53_batch_runtime;

/* One workspace/control set, independent mutable state and history per bank.
 * Checkpoint slots are virtual reservations and are funded at capture time. */
static uint64_t glm53_bank_bytes_for(uint32_t ctx, uint32_t slots,
        uint32_t banks, glm53_bank_mtp mtp) {
    const uint64_t base = glm53_graph_bytes_for(ctx, slots);
    if (!base || !banks || banks > DS4_MULTISEQ_MAX_SEQ) { return 0u; }
    uint32_t kda, dsa;
    glm53_graph_layers(&kda, &dsa);
    const uint64_t dim = (uint64_t)DS4_N_HEAD * DS4_N_KDA_HEAD_DIM;
    const uint64_t state = (dim * DS4_N_KDA_HEAD_DIM +
        3u * dim * DS4_N_SSM_CONV) * sizeof(float) * kda +
        2u * (uint64_t)DS4_GLM53_POOL_SIZE * DS4_N_INDEXER_HEAD_DIM * sizeof(float) * dsa +
        2u * (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint32_t pools = (ctx + DS4_GLM53_POOL_SIZE - 1u) / DS4_GLM53_POOL_SIZE;
    const uint64_t cache = glm53_graph_diag()
        ? (uint64_t)ctx * DS4_N_HEAD * DS4_N_KEY_MLA * 2u * sizeof(uint16_t) * dsa
        : ((uint64_t)ctx * DS4_N_KV_LORA +
           (uint64_t)pools * DS4_N_INDEXER_HEAD_DIM) * sizeof(uint16_t) * dsa;
    uint64_t bytes = base + (uint64_t)(banks - 1u) *
        (state + cache + (uint64_t)DS4_N_VOCAB * sizeof(float));
    if (mtp == GLM53_BANK_MTP_ON) {
        bytes += (uint64_t)banks * ctx * DS4_N_KV_LORA * sizeof(uint16_t) +
                 2u * (uint64_t)DS4_N_EMBD * sizeof(float) + GLM53_MTP_SAVES * state;
    }
    return bytes;
}

static uint64_t glm53_bank_row_bytes(const ds4_glm53_gpu_graph *g) {
    return (g->expanded_diag ? (uint64_t)DS4_N_HEAD * DS4_N_KEY_MLA * 2u
                            : DS4_N_KV_LORA) * sizeof(uint16_t);
}

static void glm53_ckpt_drop(ds4_glm53_batch_runtime *rt, uint32_t bank) {
    for (uint32_t i = 0; i < GLM53_BANK_CHECKPOINTS; i++) {
        partial_checkpoint_clear_ref(&rt->checkpoint[i], bank);
    }
}

static void glm53_ckpt_inherit(ds4_glm53_batch_runtime *rt, uint32_t src,
                               uint32_t dst, uint32_t cut) {
    if (src != dst) { glm53_ckpt_drop(rt, dst); }
    for (uint32_t i = 0; i < GLM53_BANK_CHECKPOINTS; i++) {
        ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (!partial_checkpoint_ref(cp, src)) { continue; }
        if (cp->pos <= cut) { partial_checkpoint_set_ref(cp, dst); }
        else if (src == dst) { partial_checkpoint_clear_ref(cp, dst); }
    }
}

static int glm53_ckpt_find(ds4_glm53_batch_runtime *rt, uint32_t bank,
                           uint32_t cut, uint32_t request_len) {
    int best = -1;
    for (uint32_t i = 0; i < GLM53_BANK_CHECKPOINTS; i++) {
        const ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (!cp->pos || cp->pos > cut || !partial_checkpoint_ref(cp, bank) ||
            (cp->pos == request_len && !cp->logits_valid)) { continue; }
        if (best < 0 || cp->pos > rt->checkpoint[best].pos) { best = (int)i; }
    }
    return best;
}

static bool glm53_history_copy(ds4_glm53_gpu_graph *dst,
        const ds4_glm53_gpu_graph *src, uint32_t n) {
    if (!dst || !src || n > src->cache_len || n > dst->ctx_cap ||
        dst->expanded_diag != src->expanded_diag) { return false; }
    if (dst == src) { return true; }
    const uint64_t latent = (uint64_t)n * glm53_bank_row_bytes(src);
    const uint64_t pool = (uint64_t)(n / DS4_GLM53_POOL_SIZE) *
                          DS4_N_INDEXER_HEAD_DIM * sizeof(uint16_t);
    for (uint32_t il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        if (ds4_glm53_layer_is_kda(il)) { continue; }
        if (!ds4_gpu_tensor_copy(dst->layer_kv[il], 0u, src->layer_kv[il], 0u, latent) ||
            (!src->expanded_diag && pool && !ds4_gpu_tensor_copy(
                dst->layer_pool[il], 0u, src->layer_pool[il], 0u, pool))) { return false; }
    }
    return true;
}

static bool glm53_ckpt_capture(ds4_glm53_batch_runtime *rt, uint32_t bank,
        uint32_t pos, glm53_logits_mode mode, uint64_t reserve) {
    if (!rt || !rt->checkpoint_slab || bank >= rt->max_seq || !pos ||
        rt->failed[bank] || rt->graph[bank].cache_len != pos) { return false; }
    uint32_t slot = 0u;
    for (uint32_t i = 0; i < GLM53_BANK_CHECKPOINTS; i++) {
        ds4_partial_checkpoint *cp = &rt->checkpoint[i];
        if (cp->pos == pos && partial_checkpoint_ref(cp, bank)) {
            if (mode == GLM53_HAS_LOGITS && rt->bank_logits_valid[bank]) {
                memcpy(rt->checkpoint_logits + (size_t)i * DS4_N_VOCAB,
                       rt->bank_logits + (size_t)bank * DS4_N_VOCAB,
                       (size_t)DS4_N_VOCAB * sizeof(float));
                cp->logits_valid = 1u;
            }
            cp->last_use = ++rt->checkpoint_clock;
            return true;
        }
        if (!cp->pos || (rt->checkpoint[slot].pos &&
            cp->last_use < rt->checkpoint[slot].last_use)) { slot = i; }
    }
    const uint64_t base = (uint64_t)slot * rt->checkpoint_slot_bytes;
    const uint64_t need = batch_span_need(rt->checkpoint_slab, base, rt->checkpoint_slot_bytes);
    if (need > ds4_mem_usable_beyond(reserve) || !ds4_gpu_tensor_ensure(
            rt->checkpoint_slab, base, rt->checkpoint_slot_bytes)) { return false; }
    ds4_partial_checkpoint *cp = &rt->checkpoint[slot];
    memset(cp, 0, sizeof(*cp));
    const ds4_glm53_gpu_graph *g = &rt->graph[bank];
    if (!ds4_gpu_tensor_copy(rt->checkpoint_slab, base, g->state_pool, 0u, g->state_bytes) ||
        !glm53_mtp_ckpt(&rt->graph[bank], rt->checkpoint_slab,
                         base + g->state_bytes, GLM53_MTP_SAVE) ||
        !ds4_gpu_synchronize()) { return false; }
    cp->pos = pos;
    cp->last_use = ++rt->checkpoint_clock;
    partial_checkpoint_set_ref(cp, bank);
    if (mode == GLM53_HAS_LOGITS && rt->bank_logits_valid[bank]) {
        memcpy(rt->checkpoint_logits + (size_t)slot * DS4_N_VOCAB,
               rt->bank_logits + (size_t)bank * DS4_N_VOCAB,
               (size_t)DS4_N_VOCAB * sizeof(float));
        cp->logits_valid = 1u;
    }
    return true;
}

/* Latent/pool rows below a checkpoint remain immutable. Recurrent state and
 * the incomplete group of four cannot be recovered by merely truncating KV. */
static bool glm53_ckpt_mtp(ds4_glm53_gpu_graph *dst,
        const ds4_glm53_gpu_graph *src, ds4_gpu_tensor *slab,
        uint64_t base, uint32_t pos) {
    if (dst->mtp_ready != src->mtp_ready ||
        !glm53_mtp_ckpt(dst, slab, base, GLM53_MTP_RESTORE) || dst->mtp_len >= pos) { return false; }
    if (!dst->mtp_ready || dst == src || dst->mtp_len == dst->mtp_min) { return true; }
    const uint64_t row = (uint64_t)DS4_N_KV_LORA * sizeof(uint16_t);
    const uint64_t end = (uint64_t)dst->mtp_len * row;
    if (end > ds4_gpu_tensor_bytes(dst->mtp_kv) || end > ds4_gpu_tensor_bytes(src->mtp_kv)) { return false; }
    const uint64_t start = (uint64_t)dst->mtp_min * row;
    /* A prefix append may reset the source's current predictor window. The
     * old checkpoint's physical rows remain, so copy its saved range. */
    return ds4_gpu_tensor_copy(dst->mtp_kv, start, src->mtp_kv, start, end - start) != 0;
}

static bool glm53_ckpt_restore(ds4_glm53_batch_runtime *rt, uint32_t src,
        uint32_t dst, uint32_t slot, uint32_t cut, uint32_t *out) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq ||
        slot >= GLM53_BANK_CHECKPOINTS) { return false; }
    ds4_partial_checkpoint *cp = &rt->checkpoint[slot];
    const uint32_t pos = cp->pos;
    ds4_glm53_gpu_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    if (!pos || pos > cut || pos > sg->cache_len || rt->failed[src] ||
        !partial_checkpoint_ref(cp, src)) { return false; }
    rt->failed[dst] = 1u;
    rt->bank_logits_valid[dst] = 0u;
    const uint64_t base = (uint64_t)slot * rt->checkpoint_slot_bytes;
    if (!glm53_history_copy(dg, sg, pos) ||
        !ds4_gpu_tensor_copy(dg->state_pool, 0u, rt->checkpoint_slab, base, dg->state_bytes) ||
        !glm53_ckpt_mtp(dg, sg, rt->checkpoint_slab, base + dg->state_bytes, pos) ||
        !ds4_gpu_synchronize()) { return false; }
    dg->cache_len = pos;
    dg->last_rows = 0u;
    rt->failed[dst] = 0u;
    if (cp->logits_valid) {
        memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB,
               rt->checkpoint_logits + (size_t)slot * DS4_N_VOCAB,
               (size_t)DS4_N_VOCAB * sizeof(float));
        rt->bank_logits_valid[dst] = 1u;
    }
    cp->last_use = ++rt->checkpoint_clock;
    glm53_ckpt_inherit(rt, src, dst, pos);
    if (out) { *out = pos; }
    return true;
}

static uint64_t glm53_ckpt_trim(ds4_glm53_batch_runtime *rt, uint64_t want) {
    uint64_t freed = 0u;
    if (!rt || !rt->checkpoint_slab || !want || !ds4_gpu_synchronize()) { return 0u; }
    for (uint32_t i = 0; i < GLM53_BANK_CHECKPOINTS && freed < want;) {
        if (rt->checkpoint[i].pos) { i++; continue; }
        const uint64_t start = (uint64_t)i * rt->checkpoint_slot_bytes;
        do { i++; } while (i < GLM53_BANK_CHECKPOINTS && !rt->checkpoint[i].pos);
        const uint64_t end = i == GLM53_BANK_CHECKPOINTS
            ? ds4_gpu_tensor_bytes(rt->checkpoint_slab) : (uint64_t)i * rt->checkpoint_slot_bytes;
        freed += ds4_gpu_tensor_trim(rt->checkpoint_slab, start, end - start);
    }
    return freed;
}

static ds4_gpu_tensor *glm53_bank_view(ds4_gpu_tensor *pool,
        ds4_gpu_tensor *source_pool, ds4_gpu_tensor *source) {
    if (!source) { return NULL; }
    const uint64_t offset = (uintptr_t)ds4_gpu_tensor_ptr(source) -
                            (uintptr_t)ds4_gpu_tensor_ptr(source_pool);
    return ds4_gpu_tensor_view(pool, offset, ds4_gpu_tensor_bytes(source));
}

static bool glm53_bank_clone(ds4_glm53_gpu_graph *g,
                             ds4_glm53_gpu_graph *source) {
    if (!glm53_graph_bind(g, source, source->row_cap)) {
        memset(g, 0, sizeof(*g));
        return false;
    }
    g->state_pool = NULL; g->control_pool = NULL; g->logits = NULL; g->kda_scratch = NULL;
    g->last_hidden = NULL; g->mtp_hidden = NULL;
    g->mtp_kv = NULL; g->mtp_concat = NULL; g->mtp_journal = NULL;
    g->mtp_logit_rows = NULL;
    g->mtp_epoch = NULL;
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        g->recurrent[il] = NULL; g->q_conv_state[il] = NULL;
        g->k_conv_state[il] = NULL; g->v_conv_state[il] = NULL;
        g->q_conv_weight[il] = NULL; g->k_conv_weight[il] = NULL; g->v_conv_weight[il] = NULL;
        g->decay_scale[il] = NULL; g->dt_bias[il] = NULL; g->o_norm[il] = NULL;
        g->layer_tail_k[il] = NULL; g->layer_tail_gate[il] = NULL;
        g->layer_kv[il] = NULL; g->layer_pool[il] = NULL;
    }
    g->state_pool = ds4_gpu_tensor_alloc(g->state_bytes);
    g->control_pool = ds4_gpu_tensor_view(source->control_pool, 0u, source->control_bytes);
    g->logits = ds4_gpu_tensor_alloc((uint64_t)DS4_N_VOCAB * sizeof(float));
    if (source->kda_scratch) {
        g->kda_scratch = ds4_gpu_tensor_view(source->kda_scratch, 0u,
                                            ds4_gpu_tensor_bytes(source->kda_scratch));
    }
    if (!g->state_pool || !g->control_pool || !g->logits ||
        (source->kda_scratch && !g->kda_scratch)) { return false; }
    for (uint32_t il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
#define GLM53_MUTABLE(field) do { \
        if (source->field[il]) { \
            g->field[il] = glm53_bank_view(g->state_pool, source->state_pool, source->field[il]); \
            if (!g->field[il]) { return false; } \
        } \
    } while (0)
        GLM53_MUTABLE(recurrent); GLM53_MUTABLE(q_conv_state);
        GLM53_MUTABLE(k_conv_state); GLM53_MUTABLE(v_conv_state);
        GLM53_MUTABLE(layer_tail_k); GLM53_MUTABLE(layer_tail_gate);
#undef GLM53_MUTABLE
#define GLM53_CONTROL(field) do { \
        if (source->field[il]) { \
            g->field[il] = ds4_gpu_tensor_view(source->field[il], 0u, ds4_gpu_tensor_bytes(source->field[il])); \
            if (!g->field[il]) { return false; } \
        } \
    } while (0)
        GLM53_CONTROL(q_conv_weight); GLM53_CONTROL(k_conv_weight); GLM53_CONTROL(v_conv_weight);
        GLM53_CONTROL(decay_scale); GLM53_CONTROL(dt_bias); GLM53_CONTROL(o_norm);
#undef GLM53_CONTROL
        if (source->layer_kv[il]) {
            g->layer_kv[il] = ds4_gpu_tensor_alloc(ds4_gpu_tensor_bytes(source->layer_kv[il]));
            if (!g->layer_kv[il]) { return false; }
        }
        if (source->layer_pool[il]) {
            g->layer_pool[il] = ds4_gpu_tensor_alloc(ds4_gpu_tensor_bytes(source->layer_pool[il]));
            if (!g->layer_pool[il]) { return false; }
        }
    }
    g->last_hidden = glm53_bank_view(g->state_pool, source->state_pool, source->last_hidden);
    g->mtp_hidden = glm53_bank_view(g->state_pool, source->state_pool, source->mtp_hidden);
    return g->last_hidden && g->mtp_hidden && glm53_mtp_clone(g, source) && glm53_graph_reset(g);
}

static void glm53_batch_free(ds4_glm53_batch_runtime *rt) {
    if (!rt) { return; }
    for (uint32_t b = rt->max_seq; b > 0; b--) {
        if (rt->graph) { glm53_graph_free(&rt->graph[b - 1u]); }
    }
    free(rt->graph); free(rt->bank_logits);
    free(rt->bank_logits_valid); free(rt->failed);
    ds4_gpu_tensor_free(rt->checkpoint_slab);
    free(rt->checkpoint_logits); free(rt);
}

static ds4_glm53_batch_runtime *glm53_batch_create(ds4_engine *e,
        uint32_t ctx, uint32_t max_seq) {
    if (!e || !ctx || !max_seq || max_seq > DS4_MULTISEQ_MAX_SEQ) { return NULL; }
    ds4_glm53_batch_runtime *rt = xcalloc(1u, sizeof(*rt));
    rt->ctx_size = ctx; rt->max_seq = max_seq;
    rt->graph = xcalloc(max_seq, sizeof(*rt->graph));
    ds4_glm53_stream *stream = e->glm53_stream.count ? &e->glm53_stream : NULL;
    if (!glm53_graph_alloc(&rt->graph[0], &e->model, &e->weights, ctx, stream)) {
        glm53_batch_free(rt); return NULL;
    }
    if (ds4_engine_has_mtp(e) && !glm53_mtp_enable(&rt->graph[0])) {
        glm53_batch_free(rt); return NULL;
    }
    rt->prefill_cap = rt->graph[0].row_cap;
    for (uint32_t b = 0; b < max_seq; b++) {
        if (b && !glm53_bank_clone(&rt->graph[b], &rt->graph[0])) {
            glm53_batch_free(rt); return NULL;
        }
    }
    rt->bank_logits = xcalloc((size_t)max_seq * DS4_N_VOCAB, sizeof(float));
    rt->bank_logits_valid = xcalloc(max_seq, sizeof(uint8_t));
    rt->failed = xcalloc(max_seq, sizeof(uint8_t));
    const char *partial = getenv("DS4_SERVER_FORK_PARTIAL");
    if (partial && strcmp(partial, "0") == 0) { return rt; }
    rt->checkpoint_slot_bytes = rt->graph[0].state_bytes + glm53_mtp_ckpt_bytes(&rt->graph[0]);
    uint64_t bytes = rt->checkpoint_slot_bytes * GLM53_BANK_CHECKPOINTS;
    const uint64_t page = ds4_gpu_vmm_demand_page();
    if (page) { bytes = (bytes + page - 1u) / page * page; }
    rt->checkpoint_slab = ds4_gpu_tensor_reserve(bytes);
    if (rt->checkpoint_slab) {
        rt->checkpoint_logits = xcalloc((size_t)GLM53_BANK_CHECKPOINTS * DS4_N_VOCAB, sizeof(float));
        const uint32_t stride = (ctx + DS4_PARTIAL_PERIODIC_TARGET - 1u) / DS4_PARTIAL_PERIODIC_TARGET;
        rt->checkpoint_stride = ((stride + GLM53_CKPT_ALIGN - 1u) / GLM53_CKPT_ALIGN) * GLM53_CKPT_ALIGN;
    }
    return rt;
}

static void glm53_bank_reset(ds4_glm53_batch_runtime *rt, uint32_t bank) {
    if (!rt || bank >= rt->max_seq) { return; }
    glm53_ckpt_drop(rt, bank);
    rt->bank_logits_valid[bank] = 0u;
    rt->failed[bank] = !glm53_graph_reset(&rt->graph[bank]);
}

static bool glm53_bank_copy(ds4_glm53_batch_runtime *rt, uint32_t src,
                            uint32_t dst, uint32_t tokens) {
    if (!rt || src >= rt->max_seq || dst >= rt->max_seq || src == dst ||
        rt->failed[src] || rt->graph[src].cache_len != tokens) { return false; }
    ds4_glm53_gpu_graph *sg = &rt->graph[src], *dg = &rt->graph[dst];
    rt->failed[dst] = 1u; rt->bank_logits_valid[dst] = 0u;
    if (!glm53_history_copy(dg, sg, tokens) ||
        !ds4_gpu_tensor_copy(dg->state_pool, 0u, sg->state_pool, 0u, sg->state_bytes) ||
        !glm53_mtp_copy(dg, sg, tokens) ||
        !ds4_gpu_synchronize()) { return false; }
    dg->cache_len = tokens; dg->last_rows = 0u;
    rt->failed[dst] = 0u;
    memcpy(rt->bank_logits + (size_t)dst * DS4_N_VOCAB,
           rt->bank_logits + (size_t)src * DS4_N_VOCAB, (size_t)DS4_N_VOCAB * sizeof(float));
    rt->bank_logits_valid[dst] = rt->bank_logits_valid[src];
    glm53_ckpt_inherit(rt, src, dst, tokens);
    return true;
}

static bool glm53_bank_prefill(ds4_glm53_batch_runtime *rt, ds4_engine *e,
        uint32_t bank, const int *tokens, uint32_t rows, uint32_t pos,
        glm53_prefill_mode mode) {
    if (!rt || !e || bank >= rt->max_seq || rt->failed[bank]) { return false; }
    ds4_glm53_gpu_graph *g = &rt->graph[bank];
    rt->bank_logits_valid[bank] = 0u;
    float *logits = mode != GLM53_MORE_ROWS ? rt->bank_logits + (size_t)bank * DS4_N_VOCAB : NULL;
    /* A one-row prefill remains prefill: only decode warms the predictor. */
    bool ok = mode == GLM53_DECODE_ROW && rows == 1u && g->mtp_ready
        ? glm53_graph_forward_token(g, &e->model, &e->weights, tokens[0], pos, NULL, logits)
        : glm53_graph_prefill(g, &e->model, &e->weights, tokens, rows, pos, NULL, logits);
    for (uint32_t i = 0; ok && logits && i < DS4_N_VOCAB; i++) { ok = isfinite(logits[i]); }
    rt->failed[bank] = !ok;
    rt->bank_logits_valid[bank] = ok && logits;
    return ok;
}

static bool glm53_bank_decode(ds4_glm53_batch_runtime *rt, ds4_engine *e,
        const uint32_t *banks, const int *tokens, const uint32_t *positions, uint32_t rows) {
    if (!rt || !e || !banks || !tokens || !positions || !rows) { return false; }
    for (uint32_t i = 0; i < rows; i++) {
        if (!glm53_bank_prefill(rt, e, banks[i], tokens + i, 1u, positions[i], GLM53_DECODE_ROW)) {
            return false;
        }
    }
    return true;
}

#elif defined(GLM53_BATCH_API)
static uint64_t glm53_bank_payload_bytes(ds4_batch_ctx *ctx, uint32_t bank) {
    if (!ctx || !ctx->glm53 || bank >= ctx->max_seq ||
        !ctx->bank_hist_valid[bank] || !ctx->bank_hist_len[bank] ||
        ctx->glm53->failed[bank] || !ctx->glm53->bank_logits_valid[bank]) { return 0u; }
    return glm53_payload_bytes(&ctx->glm53->graph[bank], ctx->bank_hist_len[bank]);
}

static int glm53_bank_save_payload(ds4_batch_ctx *ctx, uint32_t bank,
        FILE *fp, char *err, size_t errlen) {
    if (!glm53_bank_payload_bytes(ctx, bank)) {
        payload_set_err(err, errlen, "GLM-5.3 bank has no complete snapshot frontier");
        return 1;
    }
    return glm53_payload_save(&ctx->glm53->graph[bank],
        ctx->bank_hist + (size_t)bank * ctx->seq_cap, ctx->bank_hist_len[bank],
        ctx->glm53->bank_logits + (size_t)bank * DS4_N_VOCAB, fp, err, errlen);
}

static int glm53_bank_load_payload(ds4_batch_ctx *ctx, uint32_t bank,
        FILE *fp, uint64_t bytes, char *err, size_t errlen) {
    ds4_glm53_batch_runtime *rt = ctx->glm53;
    ctx->bank_gen[bank]++;
    ctx->bank_hist_valid[bank] = 0u; ctx->bank_hist_len[bank] = 0u;
    glm53_bank_reset(rt, bank);
    rt->failed[bank] = 1u;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &bytes, err, errlen)) { return 1; }
    }
    if (!h[7] || h[7] > ctx->seq_cap) {
        payload_set_err(err, errlen, "GLM-5.3 bank payload exceeds the token bound");
        return 1;
    }
    int *tokens = NULL;
    float *logits = rt->bank_logits + (size_t)bank * DS4_N_VOCAB;
    if (glm53_payload_restore(&rt->graph[bank], fp, &bytes, h,
                              &tokens, logits, err, errlen)) { return 1; }
    bool valid = true;
    for (uint32_t i = 0; valid && i < DS4_N_VOCAB; i++) { valid = isfinite(logits[i]); }
    if (!valid) {
        free(tokens);
        (void)glm53_graph_reset(&rt->graph[bank]);
        payload_set_err(err, errlen, "GLM-5.3 bank payload has nonfinite logits");
        return 1;
    }
    memcpy(ctx->bank_hist + (size_t)bank * ctx->seq_cap, tokens, (size_t)h[7] * sizeof(int));
    free(tokens);
    ctx->bank_hist_len[bank] = h[7]; ctx->bank_hist_valid[bank] = 1u;
    rt->bank_logits_valid[bank] = 1u; rt->failed[bank] = 0u;
    return 0;
}

#elif defined(GLM53_BATCH_CREATE)
static int glm53_batch_ctx_create(ds4_engine *e, int ctx_size, int max_seq,
        glm53_fit_mode mode, ds4_batch_ctx **out, char *err, size_t errlen) {
#define GLM53_BANK_ERR(...) do { if (err && errlen) { snprintf(err, errlen, __VA_ARGS__); } } while (0)
    const uint32_t slots = e->glm53_stream.count;
    const uint32_t cap = glm53_graph_row_cap((uint32_t)ctx_size, slots);
    const glm53_bank_mtp mtp = ds4_engine_has_mtp(e) ? GLM53_BANK_MTP_ON : GLM53_BANK_MTP_OFF;
    const uint64_t serial = e->vision_ready
        ? glm53_bank_bytes_for((uint32_t)ctx_size, slots, 1u, mtp) : 0u;
    uint32_t chosen = (uint32_t)max_seq;
    if (!cap || !glm53_bank_bytes_for((uint32_t)ctx_size, slots, chosen, mtp)) {
        GLM53_BANK_ERR("batch_ctx_create: invalid GLM-5.3 context/prefill/bank plan");
        return 1;
    }
    uint64_t free_b = 0u, total_b = 0u;
    if (ds4_gpu_mem_info(&free_b, &total_b) == 0) {
        const uint64_t outstanding = ds4_gpu_substrate_outstanding();
        free_b = free_b > outstanding ? free_b - outstanding : 0u;
        const uint64_t headroom = ds4_batch_fit_headroom_bytes(ctx_size);
        /* Reserve a serial image lane alongside text banks when vision exists. */
        while (chosen && glm53_bank_bytes_for((uint32_t)ctx_size, slots, chosen, mtp) +
                serial + headroom > free_b) { chosen--; }
        if (!chosen || (mode == GLM53_STRICT_BANKS && chosen < (uint32_t)max_seq)) {
            GLM53_BANK_ERR("batch_ctx_create: GLM-5.3 banks/serial reserve/headroom exceed free memory "
                          "(ctx=%d max_seq=%d free=%.2f GiB)", ctx_size, max_seq,
                          (double)free_b / 1073741824.0);
            return 1;
        }
    }
    ds4_batch_ctx *ctx = xcalloc(1u, sizeof(*ctx));
    ctx->e = e; ctx->ctx_size = (uint32_t)ctx_size; ctx->prefill_cap = cap;
    ctx->raw_cap = (uint32_t)ctx_size; ctx->seq_cap = (uint32_t)ctx_size; ctx->max_seq = chosen;
    ctx->serial_reserve = serial;
    for (;;) {
        ctx->glm53 = glm53_batch_create(e, ctx->ctx_size, ctx->max_seq);
        if (ctx->glm53) { break; }
        if (mode == GLM53_STRICT_BANKS || ctx->max_seq <= 1u) {
            GLM53_BANK_ERR("batch_ctx_create: GLM-5.3 bank allocation failed (ctx=%d max_seq=%u)",
                          ctx_size, ctx->max_seq);
            free(ctx); return 1;
        }
        ctx->max_seq--;
    }
    if ((uint64_t)ctx->max_seq * ctx->seq_cap > SIZE_MAX / sizeof(int)) {
        GLM53_BANK_ERR("batch_ctx_create: GLM-5.3 bank history size overflow");
        glm53_batch_free(ctx->glm53); free(ctx); return 1;
    }
    ctx->bank_hist = xmalloc((size_t)ctx->max_seq * ctx->seq_cap * sizeof(int));
    ctx->bank_hist_len = xcalloc(ctx->max_seq, sizeof(*ctx->bank_hist_len));
    ctx->bank_hist_valid = xcalloc(ctx->max_seq, sizeof(*ctx->bank_hist_valid));
    ctx->bank_gen = xmalloc((size_t)ctx->max_seq * sizeof(*ctx->bank_gen));
    for (uint32_t b = 0; b < ctx->max_seq; b++) { ctx->bank_gen[b] = 1u; }
    ctx->bank_last_use = xcalloc(ctx->max_seq, sizeof(*ctx->bank_last_use));
    ctx->supports_partial_reuse = ctx->glm53->checkpoint_slab != NULL;
    ctx->mtp = mtp == GLM53_BANK_MTP_ON;
    ds4_metric_set(&ds4_metrics_get()->banks_total, ctx->max_seq);
    *out = ctx;
    return 0;
#undef GLM53_BANK_ERR
}
#endif
