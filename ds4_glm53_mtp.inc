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

static uint64_t glm53_mtp_bytes(uint32_t ctx) {
    return glm53_mtp_kv_bytes(ctx) + 2u * (uint64_t)DS4_N_EMBD * sizeof(float);
}

static void glm53_mtp_free(ds4_glm53_gpu_graph *g) {
    ds4_gpu_tensor_free(g->mtp_kv);
    ds4_gpu_tensor_free(g->mtp_concat);
    ds4_gpu_tensor_free(g->mtp_journal);
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
    g->mtp_concat = ds4_gpu_tensor_alloc(2u * (uint64_t)DS4_N_EMBD * sizeof(float));
    g->mtp_journal = ds4_gpu_tensor_alloc(GLM53_MTP_SAVES * g->state_bytes);
    g->mtp_logit_rows = calloc((size_t)GLM53_MTP_SAVES * DS4_N_VOCAB, sizeof(float));
    g->mtp_epoch = calloc(1u, sizeof(*g->mtp_epoch));
    if (g->mtp_epoch) { g->mtp_epoch->refs = 1u; }
    if (!g->mtp_kv || !g->mtp_concat || !g->mtp_journal || !g->mtp_logit_rows || !g->mtp_epoch) {
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
    g->mtp_logit_rows = calloc((size_t)GLM53_MTP_SAVES * DS4_N_VOCAB, sizeof(float));
    if (!g->mtp_kv || !g->mtp_concat || !g->mtp_journal || !g->mtp_logit_rows) {
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
    if (!glm53_mtp_journal(g, 0u, GLM53_MTP_SAVE)) { return false; }
    if (g->mtp_len && g->mtp_len != start - 1u) { g->mtp_len = g->mtp_min = 0u; }
    g->mtp_trial[0] = first;
    for (uint32_t i = 0u; i + 1u < n; i++) {
        const ds4_gpu_tensor *hidden = i ? g->mtp_hidden : g->last_hidden;
        if (!glm53_mtp_step(g, m, w, hidden, g->mtp_trial[i], start - 1u + i,
                            &g->mtp_trial[i + 1u])) { return false; }
    }
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
