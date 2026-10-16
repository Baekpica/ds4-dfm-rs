/* GLM predictor algebra follows pinned antirez/ds4 0aaea5a238fb41a35106a551e73c8409dfb751ac,
 * glm_graph_mtp_step: next-token embedding + mean trunk HC, full causal MLA,
 * ordinary residual MoE, nextn head norm and the shared output head. */
#ifdef GLM53_MTP_STATE
typedef enum { GLM53_MTP_SAVE, GLM53_MTP_RESTORE } glm53_mtp_move;

/* Bank forwards are serialized. A new trial replaces the shared journal,
 * so older banks must lose permission to commit or rewind those rows. */
struct glm53_mtp_epoch { uint64_t seq; uint32_t refs; };

static uint64_t glm53_mtp_kv_bytes(uint32_t ctx) {
    return (uint64_t)ctx * DS4_N_KV_LORA * sizeof(uint16_t);
}

#ifdef GLM53_MTP_VERIFY
/* One stash slot per layer: the pre-trial index tails, then the verified
 * rows' raw index keys and gates (GLM VQ tail replay on rejection). */
static uint64_t glm53_mtp_stash_slot(void) {
    return 2u * (DS4_GLM53_POOL_SIZE + GLM53_MTP_ROWS) * (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
}

static uint64_t glm53_mtp_verify_bytes(void) {
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM53_VQ) { return 0u; }
    return DS4_N_LAYER * glm53_mtp_stash_slot() + (uint64_t)DS4_N_EMBD * sizeof(float) +
        GLM53_MTP_ROWS * (uint64_t)(3u * DS4_N_EMBD + DS4_N_VOCAB) * sizeof(float);
}

static void glm53_mtp_verify_clear(ds4_glm53_gpu_graph *g) {
    g->mtp_stash = NULL; g->mtp_rows = NULL; g->mtp_logits = NULL;
}

static void glm53_mtp_verify_free(ds4_glm53_gpu_graph *g) {
    ds4_gpu_tensor_free(g->mtp_stash);
    ds4_gpu_tensor_free(g->mtp_rows);
    ds4_gpu_tensor_free(g->mtp_logits);
    glm53_mtp_verify_clear(g);
}

static bool glm53_mtp_verify_alloc(ds4_glm53_gpu_graph *g) {
    if (!glm53_mtp_verify_bytes()) { return true; }
    g->mtp_stash = ds4_gpu_tensor_alloc(DS4_N_LAYER * glm53_mtp_stash_slot());
    // Row 0 is the pre-trial hidden; row r+1 is verified row r.
    g->mtp_rows = ds4_gpu_tensor_alloc((GLM53_MTP_ROWS + 1u) * (uint64_t)DS4_N_EMBD * sizeof(float));
    g->mtp_logits = ds4_gpu_tensor_alloc(GLM53_MTP_ROWS * (uint64_t)DS4_N_VOCAB * sizeof(float));
    return g->mtp_stash && g->mtp_rows && g->mtp_logits;
}

// Verify buffers follow the shared journal's single-live-trial epoch.
static bool glm53_mtp_verify_share(ds4_glm53_gpu_graph *g, const ds4_glm53_gpu_graph *src) {
    if (!src->mtp_stash) { return true; }
    g->mtp_stash = ds4_gpu_tensor_view(src->mtp_stash, 0u, ds4_gpu_tensor_bytes(src->mtp_stash));
    g->mtp_rows = ds4_gpu_tensor_view(src->mtp_rows, 0u, ds4_gpu_tensor_bytes(src->mtp_rows));
    g->mtp_logits = ds4_gpu_tensor_view(src->mtp_logits, 0u, ds4_gpu_tensor_bytes(src->mtp_logits));
    return g->mtp_stash && g->mtp_rows && g->mtp_logits;
}
#else
static uint64_t glm53_mtp_verify_bytes(void) { return 0u; }
static void glm53_mtp_verify_clear(ds4_glm53_gpu_graph *g) { (void)g; }
static void glm53_mtp_verify_free(ds4_glm53_gpu_graph *g) { (void)g; }
static bool glm53_mtp_verify_alloc(ds4_glm53_gpu_graph *g) { (void)g; return true; }
static bool glm53_mtp_verify_share(ds4_glm53_gpu_graph *g, const ds4_glm53_gpu_graph *src) {
    (void)g; (void)src;
    return true;
}
#endif

static uint64_t glm53_mtp_bytes(uint32_t ctx) {
    return glm53_mtp_kv_bytes(ctx) + 2u * (uint64_t)DS4_N_EMBD * sizeof(float) + glm53_mtp_verify_bytes();
}

static void glm53_mtp_free(ds4_glm53_gpu_graph *g) {
    ds4_gpu_tensor_free(g->mtp_kv);
    ds4_gpu_tensor_free(g->mtp_concat);
    ds4_gpu_tensor_free(g->mtp_journal);
    glm53_mtp_verify_free(g);
    free(g->mtp_logit_rows);
    if (g->mtp_epoch && --g->mtp_epoch->refs == 0u) { free(g->mtp_epoch); }
    g->mtp_kv = NULL; g->mtp_concat = NULL; g->mtp_journal = NULL;
    g->mtp_logit_rows = NULL;
    g->mtp_epoch = NULL;
    g->mtp_rewind_epoch = 0u;
    g->mtp_ready = false;
    g->mtp_trial_n = 0u;
    g->mtp_rewind_n = 0u;
}

static bool glm53_mtp_enable(ds4_glm53_gpu_graph *g) {
    if (!g || !g->ready || !g->ctx_cap || !g->state_bytes) { return false; }
    if (g->mtp_ready) { return true; }
    g->mtp_kv = ds4_gpu_tensor_alloc(glm53_mtp_kv_bytes(g->ctx_cap));
    const bool verify = glm53_mtp_verify_bytes() != 0u;
    // Batched verify rebuilds draft-layer KV for every verified row at once.
    g->mtp_concat = ds4_gpu_tensor_alloc((verify ? GLM53_MTP_ROWS : 1u) * 2u * (uint64_t)DS4_N_EMBD * sizeof(float));
    g->mtp_journal = ds4_gpu_tensor_alloc(GLM53_MTP_SAVES * g->state_bytes);
    const bool funded = glm53_mtp_verify_alloc(g);
    g->mtp_logit_rows = calloc((size_t)GLM53_MTP_SAVES * DS4_N_VOCAB, sizeof(float));
    g->mtp_epoch = calloc(1u, sizeof(*g->mtp_epoch));
    if (g->mtp_epoch) { g->mtp_epoch->refs = 1u; }
    if (!g->mtp_kv || !g->mtp_concat || !g->mtp_journal || !g->mtp_logit_rows || !g->mtp_epoch || !funded) {
        glm53_mtp_free(g);
        return false;
    }
    g->mtp_ready = true;
    g->mtp_len = g->mtp_min = 0u;
    return true;
}

static bool glm53_mtp_clone(ds4_glm53_gpu_graph *g,
                            const ds4_glm53_gpu_graph *src) {
    g->mtp_kv = NULL; g->mtp_concat = NULL; g->mtp_journal = NULL;
    glm53_mtp_verify_clear(g);
    g->mtp_logit_rows = NULL;
    g->mtp_epoch = NULL;
    g->mtp_rewind_n = 0u;
    g->mtp_rewind_epoch = 0u;
    g->mtp_ready = false;
    g->mtp_len = g->mtp_min = g->mtp_trial_n = 0u;
    if (!src->mtp_ready) { return true; }
    if (!src->mtp_epoch || src->mtp_epoch->refs == UINT32_MAX) { return false; }
    g->mtp_epoch = src->mtp_epoch;
    g->mtp_epoch->refs++;
    g->mtp_kv = ds4_gpu_tensor_alloc(glm53_mtp_kv_bytes(g->ctx_cap));
    g->mtp_concat = ds4_gpu_tensor_view(src->mtp_concat, 0u, ds4_gpu_tensor_bytes(src->mtp_concat));
    g->mtp_journal = ds4_gpu_tensor_view(src->mtp_journal, 0u, ds4_gpu_tensor_bytes(src->mtp_journal));
    const bool shared = glm53_mtp_verify_share(g, src);
    g->mtp_logit_rows = calloc((size_t)GLM53_MTP_SAVES * DS4_N_VOCAB, sizeof(float));
    if (!g->mtp_kv || !g->mtp_concat || !g->mtp_journal || !g->mtp_logit_rows || !shared) {
        glm53_mtp_free(g);
        return false;
    }
    g->mtp_ready = true;
    return true;
}

static bool glm53_mtp_copy(ds4_glm53_gpu_graph *dst,
        const ds4_glm53_gpu_graph *src, uint32_t n) {
    if (!dst || !src || dst->mtp_ready != src->mtp_ready ||
        n > src->cache_len || n > dst->ctx_cap || src->mtp_trial_n) { return false; }
    dst->mtp_rewind_n = 0u;
    dst->mtp_rewind_epoch = 0u;
    if (!src->mtp_ready || dst == src) { return true; }
    const uint32_t limit = n ? n - 1u : 0u;
    const uint32_t len = src->mtp_len < limit ? src->mtp_len : limit;
    const uint32_t min = src->mtp_min < len ? src->mtp_min : len;
    const uint64_t row = (uint64_t)DS4_N_KV_LORA * sizeof(uint16_t);
    if (len > min && !ds4_gpu_tensor_copy(dst->mtp_kv, (uint64_t)min * row,
            src->mtp_kv, (uint64_t)min * row, (uint64_t)(len - min) * row)) { return false; }
    dst->mtp_min = min;
    dst->mtp_len = len;
    dst->mtp_trial_n = 0u;
    return true;
}

static uint64_t glm53_mtp_ckpt_bytes(const ds4_glm53_gpu_graph *g) {
    (void)g;
    return 2u * sizeof(uint32_t);
}

static bool glm53_mtp_ckpt(ds4_glm53_gpu_graph *g, ds4_gpu_tensor *slab,
                            uint64_t base, glm53_mtp_move dir) {
    if (!g || !slab || g->mtp_trial_n) { return false; }
    uint32_t cursors[2] = {g->mtp_len, g->mtp_min};
    if (dir == GLM53_MTP_SAVE) {
        return ds4_gpu_tensor_write(slab, base, cursors, sizeof(cursors)) != 0;
    }
    if (!ds4_gpu_tensor_read(slab, base, cursors, sizeof(cursors)) ||
        cursors[1] > cursors[0] || cursors[0] > g->ctx_cap ||
        (!g->mtp_ready && cursors[0])) { return false; }
    g->mtp_len = cursors[0];
    g->mtp_min = cursors[1];
    g->mtp_rewind_n = 0u;
    g->mtp_rewind_epoch = 0u;
    return true;
}

static bool glm53_mtp_journal(ds4_glm53_gpu_graph *g, uint32_t row,
                              glm53_mtp_move dir) {
    if (!g || !g->mtp_ready || row >= GLM53_MTP_SAVES) { return false; }
    const uint64_t base = (uint64_t)row * g->state_bytes;
    if (dir == GLM53_MTP_SAVE) {
        if (!ds4_gpu_tensor_copy(g->mtp_journal, base, g->state_pool, 0u, g->state_bytes)) {
            return false;
        }
        g->mtp_saved_len[row] = g->mtp_len;
        g->mtp_saved_min[row] = g->mtp_min;
        return true;
    }
    if (!ds4_gpu_tensor_copy(g->state_pool, 0u, g->mtp_journal, base, g->state_bytes)) {
        return false;
    }
    g->mtp_len = g->mtp_saved_len[row];
    g->mtp_min = g->mtp_saved_min[row];
    return true;
}
#endif

#ifdef GLM53_MTP_STEP
static bool glm53_mtp_step(ds4_glm53_gpu_graph *owner, const ds4_model *m,
        const ds4_weights *w, const ds4_gpu_tensor *hidden, int token,
        uint32_t pos, int *next) {
    if (!owner || !owner->mtp_ready || !hidden || !next || !owner->cache_len ||
        pos >= owner->ctx_cap || token < 0 || (uint32_t)token >= DS4_N_VOCAB) { return false; }
    if (!owner->mtp_len) { owner->mtp_min = pos; }
    if (owner->mtp_len && pos != owner->mtp_len) { return false; }
    ds4_glm53_gpu_graph active;
    if (!glm53_graph_bind(&active, owner, 1u)) { return false; }
    ds4_glm53_gpu_graph *g = &active;
#ifndef __APPLE__
    if (g->gvq) {
        const bool ok = gvq_mtp_block(g, m, w, hidden, token, pos, owner->mtp_min) &&
            ds4_gpu_argmax_tensor(g->token, g->logits, DS4_N_VOCAB) &&
            ds4_gpu_tensor_read(g->token, 0u, next, sizeof(*next));
        glm53_graph_unbind(g);
        if (ok) { owner->mtp_len = pos + 1u; }
        return ok && *next >= 0 && (uint32_t)*next < DS4_N_VOCAB;
    }
#endif
    const uint32_t il = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    const ds4_layer_weights *l = &w->layer[il];
    const uint64_t bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint32_t qdim = DS4_N_HEAD * DS4_N_KEY_MLA;
    const uint32_t min = owner->mtp_min;
    const uint64_t row = (uint64_t)DS4_N_KV_LORA * sizeof(uint16_t);
    ds4_gpu_tensor *left = ds4_gpu_tensor_view(g->mtp_concat, 0u, bytes);
    ds4_gpu_tensor *right = ds4_gpu_tensor_view(g->mtp_concat, bytes, bytes);
    ds4_gpu_tensor *cache = ds4_gpu_tensor_view(g->mtp_kv, (uint64_t)min * row,
                                              (uint64_t)(g->ctx_cap - min) * row);
    const bool ok = left && right && cache &&
        ds4_gpu_embed_token_quant_tensor(g->cur, m->map, m->size,
            w->token_embd->abs_offset, w->token_embd->type, DS4_N_VOCAB, (uint32_t)token, DS4_N_EMBD) &&
        ds4_gpu_rms_norm_weight_tensor(left, g->cur, m->map, m->size,
            l->nextn_enorm->abs_offset, DS4_N_EMBD, DS4_RMS_EPS) &&
        ds4_gpu_rms_norm_weight_tensor(right, hidden, m->map, m->size,
            l->nextn_hnorm->abs_offset, DS4_N_EMBD, DS4_RMS_EPS) &&
        plain_graph_matmul_tensor(g->cur, m, l->nextn_eh_proj,
            2u * DS4_N_EMBD, DS4_N_EMBD, g->mtp_concat, 1u) &&
        ds4_gpu_rms_norm_weight_tensor(g->norm, g->cur, m->map, m->size,
            l->attn_norm->abs_offset, DS4_N_EMBD, DS4_RMS_EPS) &&
        plain_graph_matmul_tensor(g->q_rank, m, l->attn_q_a, DS4_N_EMBD, DS4_N_LORA_Q, g->norm, 1u) &&
        ds4_gpu_rms_norm_weight_tensor(g->q_rank_norm, g->q_rank, m->map, m->size,
            l->attn_q_a_norm->abs_offset, DS4_N_LORA_Q, DS4_RMS_EPS) &&
        plain_graph_matmul_tensor(g->q, m, l->attn_q_b, DS4_N_LORA_Q, qdim, g->q_rank_norm, 1u) &&
        plain_graph_matmul_tensor(g->kv_rank, m, l->attn_kv_a_mqa,
            DS4_N_EMBD, DS4_N_KV_LORA, g->norm, 1u) &&
        ds4_gpu_rms_norm_weight_tensor(g->kv_rank_norm, g->kv_rank, m->map, m->size,
            l->attn_kv_a_norm->abs_offset, DS4_N_KV_LORA, DS4_RMS_EPS) &&
        ds4_gpu_glm53_store_low(g->mtp_kv, g->kv_rank_norm, 1u, pos, g->ctx_cap, DS4_N_KV_LORA) &&
        ds4_gpu_glm53_absorb_q(g->low_q, g->q, m->map, m->size,
            l->attn_k_b->abs_offset, 1u, DS4_N_HEAD, DS4_N_KV_LORA, DS4_N_KEY_MLA) &&
        ds4_gpu_glm53_attn_low(g->low_out, g->low_q, cache, NULL, 0u,
            1u, pos - min, g->ctx_cap - min, DS4_N_HEAD, DS4_N_KV_LORA, DS4_N_KEY_MLA) &&
        ds4_gpu_glm53_proj_v(g->heads, g->low_out, m->map, m->size,
            l->attn_v_b->abs_offset, 1u, DS4_N_HEAD, DS4_N_KV_LORA, DS4_N_VALUE_MLA) &&
        plain_graph_matmul_tensor(g->attn_out, m, l->attn_output, qdim, DS4_N_EMBD, g->heads, 1u) &&
        ds4_gpu_add_tensor(g->mtp_hidden, g->cur, g->attn_out, DS4_N_EMBD) &&
        ds4_gpu_rms_norm_weight_tensor(g->norm, g->mtp_hidden, m->map, m->size,
            l->ffn_norm->abs_offset, DS4_N_EMBD, DS4_RMS_EPS) &&
        glm53_graph_ffn(g, m, l, il, 1u) &&
        ds4_gpu_add_tensor(g->mtp_hidden, g->mtp_hidden, g->ffn_out, DS4_N_EMBD) &&
        ds4_gpu_rms_norm_weight_tensor(g->norm, g->mtp_hidden, m->map, m->size,
            l->nextn_shared_head_norm->abs_offset, DS4_N_EMBD, DS4_RMS_EPS) &&
        plain_graph_matmul_tensor(g->logits, m, w->output, DS4_N_EMBD, DS4_N_VOCAB, g->norm, 1u) &&
        ds4_gpu_argmax_tensor(g->token, g->logits, DS4_N_VOCAB) &&
        ds4_gpu_tensor_read(g->token, 0u, next, sizeof(*next));
    ds4_gpu_tensor_free(cache); ds4_gpu_tensor_free(right); ds4_gpu_tensor_free(left);
    glm53_graph_unbind(g);
    if (ok) { owner->mtp_len = pos + 1u; }
    return ok && *next >= 0 && (uint32_t)*next < DS4_N_VOCAB;
}
#endif

#ifdef GLM53_MTP_TRIAL
static int glm53_mtp_argmax(const float *logits) {
    uint32_t best = 0u;
    for (uint32_t i = 0u; i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { return -1; }
        if (logits[i] > logits[best]) { best = i; }
    }
    return (int)best;
}

#ifdef GLM53_MTP_VERIFY
/* GLM VQ batched verify. The n candidates run as one n-row forward, so each
 * weight is read once rather than n times. A rejection restores KDA from the
 * per-row journal snapshots and replays the index tail from stashed rows;
 * latent, pool and draft-layer rows past the frontier are rewritten before
 * any read. Rewinding to the trial start is unsupported (no slot-0 copy). */
static bool gvq_mtp_verify(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_weights *w, uint32_t n, uint32_t pos, int *target) {
    const uint64_t hidden = (uint64_t)DS4_N_EMBD * sizeof(float);
    ds4_glm53_gpu_graph a;
    if (!glm53_graph_bind(&a, g, n)) { return false; }
    a.mtp_capture = true;
    bool ok = ds4_gpu_gvq_copy(g->mtp_rows, 0u, g->last_hidden, 0u, hidden, n) &&
        glm53_graph_embed(&a, m, w, g->mtp_trial, n, NULL);
    const uint32_t layers = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    for (uint32_t il = 0; ok && il < layers; il++) {
        ok = gvq_layer(&a, m, w, il, n, pos);
        ds4_gpu_tensor *swap = a.hc_cur; a.hc_cur = a.hc_next; a.hc_next = swap;
    }
    // Every row's final-norm hidden (draft input) and logits.
    ds4_gpu_tensor *rows = ok ? ds4_gpu_tensor_view(g->mtp_rows, hidden, n * hidden) : NULL;
    ds4_gpu_tensor *logits = ok ? ds4_gpu_tensor_view(g->mtp_logits, 0u,
        (uint64_t)n * DS4_N_VOCAB * sizeof(float)) : NULL;
    ok = ok && rows && logits &&
        ds4_gpu_hc_weighted_sum_tensor(a.cur, a.hc_cur, a.hc_mean, DS4_N_EMBD, DS4_N_HC) &&
        ds4_gpu_gvq_round(a.cur, (uint64_t)n * DS4_N_EMBD) &&
        gvq_norm(&a, m, rows, a.cur, w->output_norm, n, DS4_N_EMBD) &&
        gvq_mm(&a, m, logits, w->output, rows, n, DS4_GVQ_BF16) &&
        ds4_gpu_argmax_rows_tensor(a.token, logits, DS4_N_VOCAB, n) &&
        ds4_gpu_tensor_read(a.token, 0u, target, (uint64_t)n * sizeof(*target));
    if (ok && ds4_gpu_tensor_ptr(a.hc_cur) != ds4_gpu_tensor_ptr(g->hc_cur)) {
        ds4_gpu_tensor *swap = g->hc_cur; g->hc_cur = g->hc_next; g->hc_next = swap;
    }
    ds4_gpu_tensor_free(logits); ds4_gpu_tensor_free(rows);
    glm53_graph_unbind(&a);
    for (uint32_t i = 0; ok && i < n; i++) { ok = target[i] >= 0 && (uint32_t)target[i] < DS4_N_VOCAB; }
    return ok;
}

// Draft-layer KV for verify rows pos..pos+n-1 (mtp_rows hold their hidden).
static bool gvq_mtp_rows(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_weights *w, uint32_t n, uint32_t pos) {
    const uint64_t bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    ds4_glm53_gpu_graph a;
    if (!glm53_graph_bind(&a, g, n)) { return false; }
    ds4_gpu_tensor *hidden = ds4_gpu_tensor_view(g->mtp_rows, 0u, n * bytes);
    const bool ok = hidden && gvq_mtp_kv(&a, m, w, hidden, g->mtp_trial, n, pos, g->mtp_concat);
    ds4_gpu_tensor_free(hidden);
    glm53_graph_unbind(&a);
    return ok;
}

// State after `keep` verified rows (1 <= keep < verified rows).
static bool gvq_mtp_restore(ds4_glm53_gpu_graph *g, uint32_t keep) {
    const ds4_model *m = g->mtp_model;
    const uint8_t *pool = (const uint8_t *)ds4_gpu_tensor_ptr(g->state_pool);
    const uint64_t row = DS4_N_INDEXER_HEAD_DIM * sizeof(float), tail = DS4_GLM53_POOL_SIZE * row;
    const uint32_t start = g->mtp_trial_start, end = start + keep;
    const uint32_t base = end - end % DS4_GLM53_POOL_SIZE, first = base > start ? base : start;
    const bool tails = g->ctx_cap > DS4_GLM53_INDEX_TOPK && end != base;
    const uint32_t layers = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    for (uint32_t il = 0; il < layers; il++) {
        if (ds4_glm53_layer_is_kda(il)) {
            const uint8_t *at = (const uint8_t *)ds4_gpu_tensor_ptr(g->recurrent[il]);
            const uint8_t *end_v = (const uint8_t *)ds4_gpu_tensor_ptr(g->v_conv_state[il]) +
                ds4_gpu_tensor_bytes(g->v_conv_state[il]);
            const uint64_t offset = (uint64_t)(at - pool);
            if (!ds4_gpu_gvq_copy(g->state_pool, offset, g->mtp_journal,
                    keep * g->state_bytes + offset, (uint64_t)(end_v - at), 1u)) { return false; }
            continue;
        }
        if (!tails) { continue; }
        // Pre-trial tail, then the accepted rows of the open pool.
        const uint64_t slot = il * glm53_mtp_stash_slot(), keys = slot + 2u * tail + (first - start) * row;
        ds4_gpu_tensor *k = ds4_gpu_tensor_view(g->mtp_stash, keys, (end - first) * row);
        ds4_gpu_tensor *gate = ds4_gpu_tensor_view(g->mtp_stash, keys + GLM53_MTP_ROWS * row, (end - first) * row);
        const ds4_layer_weights *l = &g->mtp_weights->layer[il];
        const bool ok = k && gate &&
            ds4_gpu_gvq_copy(g->layer_tail_k[il], 0u, g->mtp_stash, slot, tail, 1u) &&
            ds4_gpu_gvq_copy(g->layer_tail_gate[il], 0u, g->mtp_stash, slot + tail, tail, 1u) &&
            ds4_gpu_gvq_pool(g->gvq, g->layer_pool[il], g->layer_tail_k[il], g->layer_tail_gate[il],
                k, gate, glmvq_index(m, l->indexer_k_norm), glmvq_index(m, l->indexer_k_norm_b),
                glmvq_index(m, l->indexer_compressor_ape), first, end - first, g->ctx_cap);
        ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(k);
        if (!ok) { return false; }
    }
    return true;
}

// Frontier after `keep` rows: state, last hidden, logits and cursors.
static bool gvq_mtp_land(ds4_glm53_gpu_graph *g, uint32_t keep, uint32_t live, float *logits) {
    const uint64_t vocab = (uint64_t)DS4_N_VOCAB * sizeof(float), hidden = (uint64_t)DS4_N_EMBD * sizeof(float);
    if (!keep || (keep < live && !gvq_mtp_restore(g, keep)) ||
        !ds4_gpu_gvq_copy(g->last_hidden, 0u, g->mtp_rows, keep * hidden, hidden, 1u) ||
        !ds4_gpu_gvq_copy(g->logits, 0u, g->mtp_logits, (keep - 1u) * vocab, vocab, 1u) ||
        !ds4_gpu_tensor_read(g->mtp_logits, (keep - 1u) * vocab, logits, vocab)) { return false; }
    g->cache_len = g->mtp_trial_start + keep;
    g->mtp_len = g->cache_len - 1u;
    g->last_rows = 1u;
    return true;
}
#endif

static bool glm53_mtp_trial(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_weights *w, int first, uint32_t n, const float *logits,
        int *tokens, int *target) {
    if (!g || !g->mtp_ready || !g->mtp_epoch || g->mtp_epoch->seq == UINT64_MAX ||
        g->mtp_trial_n || !n || n > GLM53_MTP_ROWS ||
        !g->cache_len || g->cache_len >= g->ctx_cap ||
        n > g->ctx_cap - g->cache_len || !logits || !tokens || !target) {
        return false;
    }
    const uint32_t start = g->cache_len;
    g->mtp_rewind_epoch = ++g->mtp_epoch->seq;
    g->mtp_rewind_n = 0u;
    const uint64_t bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    memcpy(g->mtp_logit_rows, logits, bytes);
#ifdef GLM53_MTP_VERIFY
    // Drafting leaves target state alone; batched verify needs no slot 0.
    const bool batched = g->gvq && g->mtp_stash;
#else
    const bool batched = false;
#endif
    if (!batched && !glm53_mtp_journal(g, 0u, GLM53_MTP_SAVE)) { return false; }
    if (g->mtp_len && g->mtp_len != start - 1u) { g->mtp_len = g->mtp_min = 0u; }
    g->mtp_trial[0] = first;
    for (uint32_t i = 0u; i + 1u < n; i++) {
        const ds4_gpu_tensor *hidden = i ? g->mtp_hidden : g->last_hidden;
        if (!glm53_mtp_step(g, m, w, hidden, g->mtp_trial[i], start - 1u + i,
                            &g->mtp_trial[i + 1u])) { return false; }
    }
#ifdef GLM53_MTP_VERIFY
    if (batched) {
        // Draft rows from start on used recursive hidden: rebuild them.
        if (!g->mtp_len) { g->mtp_min = start - 1u; }
        g->mtp_model = m;
        g->mtp_weights = w;
        if (!gvq_mtp_verify(g, m, w, n, start, target) ||
            !gvq_mtp_rows(g, m, w, n, start - 1u)) { return false; }
        g->cache_len = start + n;
        g->last_rows = 1u;
        g->mtp_len = start + n - 1u;
        g->mtp_trial_start = start;
        g->mtp_trial_n = n;
        memcpy(tokens, g->mtp_trial, (size_t)n * sizeof(*tokens));
        return true;
    }
#endif
    /* Proposals use approximate recursive hidden. Restore, then build each
     * accepted candidate's MTP row from the true preceding target hidden. */
    if (!glm53_mtp_journal(g, 0u, GLM53_MTP_RESTORE)) { return false; }
    for (uint32_t i = 0u; i < n; i++) {
        float *row = g->mtp_logit_rows + (size_t)(i + 1u) * DS4_N_VOCAB;
        if (!glm53_graph_forward_token(g, m, w, g->mtp_trial[i], start + i, NULL, row) ||
            (target[i] = glm53_mtp_argmax(row)) < 0 ||
            !glm53_mtp_journal(g, i + 1u, GLM53_MTP_SAVE)) { return false; }
    }
    g->mtp_trial_start = start;
    g->mtp_trial_n = n;
    memcpy(tokens, g->mtp_trial, (size_t)n * sizeof(*tokens));
    return true;
}

static bool glm53_mtp_accept(ds4_glm53_gpu_graph *g, uint32_t keep, float *logits) {
    if (!g || !g->mtp_trial_n || keep > g->mtp_trial_n || !logits ||
        !g->mtp_epoch || g->mtp_rewind_epoch != g->mtp_epoch->seq ||
        g->cache_len != g->mtp_trial_start + g->mtp_trial_n) { return false; }
#ifdef GLM53_MTP_VERIFY
    if (g->gvq && g->mtp_stash) {
        if (!gvq_mtp_land(g, keep, g->mtp_trial_n, logits)) { return false; }
        g->mtp_trial_n = 0u;
        g->mtp_rewind_n = keep;
        return true;
    }
#endif
    const uint64_t bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    const float *row = g->mtp_logit_rows + (size_t)keep * DS4_N_VOCAB;
    if (!glm53_mtp_journal(g, keep, GLM53_MTP_RESTORE) ||
        !ds4_gpu_tensor_write(g->logits, 0u, row, bytes)) { return false; }
    memcpy(logits, row, bytes);
    g->cache_len = g->mtp_trial_start + keep;
    g->last_rows = keep ? 1u : 0u;
    g->mtp_trial_n = 0u;
    g->mtp_rewind_n = keep;
    return true;
}

static bool glm53_mtp_rewind(ds4_glm53_gpu_graph *g, uint32_t pos, float *logits) {
    if (!g || !g->mtp_ready || g->mtp_trial_n || !g->mtp_rewind_n ||
        !g->mtp_epoch || g->mtp_rewind_epoch != g->mtp_epoch->seq ||
        g->cache_len != g->mtp_trial_start + g->mtp_rewind_n ||
        pos < g->mtp_trial_start || pos >= g->cache_len || !logits) { return false; }
    const uint32_t keep = pos - g->mtp_trial_start;
#ifdef GLM53_MTP_VERIFY
    if (g->gvq && g->mtp_stash) {
        if (!gvq_mtp_land(g, keep, g->mtp_rewind_n, logits)) { return false; }
        g->mtp_rewind_n = keep;
        return true;
    }
#endif
    const uint64_t bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    const float *row = g->mtp_logit_rows + (size_t)keep * DS4_N_VOCAB;
    if (!glm53_mtp_journal(g, keep, GLM53_MTP_RESTORE) ||
        !ds4_gpu_tensor_write(g->logits, 0u, row, bytes)) { return false; }
    memcpy(logits, row, bytes);
    g->cache_len = pos;
    g->last_rows = keep ? 1u : 0u;
    g->mtp_rewind_n = keep;
    return true;
}
#endif

#ifdef GLM53_MTP_SESSION
static int glm53_mtp_fail(ds4_session *s, char *err, size_t len) {
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
    s->generation++;
    (void)glm53_graph_reset(&s->glm53_graph);
    payload_set_err(err, len, "GLM-5.3 MTP device operation failed");
    return -1;
}

int ds4_session_glm53_trial(ds4_session *s, int first, int max_tokens,
        int *tokens, int *target, int cap, char *err, size_t len) {
    if (!ds4_session_is_glm53(s) || !tokens || !target || cap <= 0 ||
        first < 0 || (uint32_t)first >= DS4_N_VOCAB) {
        payload_set_err(err, len, "invalid GLM-5.3 MTP trial");
        return -1;
    }
    if (max_tokens <= 0 || !ds4_engine_has_mtp(s->engine)) { return 0; }
    if (ds4_session_ensure_graph(s, err, len)) { return -1; }
    ds4_glm53_gpu_graph *g = &s->glm53_graph;
    if (!s->checkpoint_valid || !g->ready || !g->mtp_ready || g->mtp_trial_n ||
        g->cache_len != (uint32_t)s->checkpoint.len || !g->cache_len ||
        g->cache_len >= g->ctx_cap) {
        payload_set_err(err, len, "GLM-5.3 MTP needs a committed checkpoint");
        return -1;
    }
    uint32_t n = (uint32_t)max_tokens;
    if (n > (uint32_t)cap) { n = (uint32_t)cap; }
    if (n > GLM53_MTP_ROWS) { n = GLM53_MTP_ROWS; }
    if (n > (uint32_t)s->engine->mtp_draft_tokens + 1u) {
        n = (uint32_t)s->engine->mtp_draft_tokens + 1u;
    }
    if (n > g->ctx_cap - g->cache_len) { n = g->ctx_cap - g->cache_len; }
#ifdef GLM53_MTP_VERIFY
    // Diagnostic: drafts per GLM VQ verify. Consecutive tokens share few
    // experts, so each verified row costs nearly a full expert pass.
    static int draft_cap = -1;
    if (draft_cap < 0) {
        const char *value = getenv("DS4_GLMVQ_MTP_DRAFT");
        draft_cap = value ? atoi(value) : 0;
    }
    if (g->gvq && draft_cap > 0 && n > (uint32_t)draft_cap + 1u) { n = (uint32_t)draft_cap + 1u; }
#endif
    if (!glm53_mtp_trial(g, &s->engine->model, &s->engine->weights,
            first, n, s->logits, tokens, target)) { return glm53_mtp_fail(s, err, len); }
    ds4_metric_add(&ds4_metrics_get()->spec_drafts, n - 1u);
    return (int)n;
}

int ds4_session_glm53_commit(ds4_session *s, int keep, char *err, size_t len) {
    if (!ds4_session_is_glm53(s) || !s->checkpoint_valid || keep < 0 ||
        !s->glm53_graph.mtp_trial_n || (uint32_t)keep > s->glm53_graph.mtp_trial_n) {
        payload_set_err(err, len, "invalid GLM-5.3 MTP accepted prefix");
        return 1;
    }
    ds4_glm53_gpu_graph *g = &s->glm53_graph;
    if (!glm53_mtp_accept(g, (uint32_t)keep, s->logits)) {
        (void)glm53_mtp_fail(s, err, len);
        return 1;
    }
    for (int i = 0; i < keep; i++) { token_vec_push(&s->checkpoint, g->mtp_trial[i]); }
    if (keep > 1) { ds4_metric_add(&ds4_metrics_get()->spec_hits, (uint32_t)keep - 1u); }
    s->mtp_draft_valid = false;
    return 0;
}
#endif
