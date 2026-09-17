/* dots3 has one recursively shared SWA/dense predictor with its own embedding
 * and the target output head. Teacher rows pair token[i+1] with POST-norm
 * target hidden[i], at position i (SGLang dots3_common/nextn.py and
 * speculative/eagle_worker_v2.py). Native owns state; Rust owns acceptance. */
enum { DOTS3_TRIAL_MAX = 4 };
struct ds4_dots3_spec {
    ds4_dots3_gpu_graph draft;
    ds4_gpu_tensor *joined, *carry, *hidden, *journal;
    float *verify_logits, *draft_logits;
    int trial[DOTS3_TRIAL_MAX];
    unsigned trial_n, target_pos, hidden_cap;
};

static uint64_t dots3_journal_bytes(void) {
    uint64_t bytes = 0;
    for (unsigned il = 0; il < DS4_N_LAYER; il++) {
        bytes += (uint64_t)DOTS3_TRIAL_MAX *
            ((ds4_dots3_layer_kv_lora(il) + DS4_N_ROT) * sizeof(uint16_t) +
             (ds4_dots3_layer_is_full_attention(il) ? DS4_N_INDEXER_HEAD_DIM * sizeof(float) : 0));
    }
    return bytes;
}

static uint64_t dots3_spec_bytes(unsigned ctx, unsigned cap) {
    if (!ctx || ctx > 524288 || !cap || cap > 8192) { return 0; }
    uint64_t norms = 0;
    for (unsigned il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        norms += DS4_N_LORA_Q + ds4_dots3_layer_kv_lora(il) + DS4_N_ROT;
        if (ds4_dots3_layer_is_full_attention(il)) { norms += 2 * DS4_N_INDEXER_HEAD_DIM; }
    }
    /* Reuse the existing scalar scratch allocator but only allocate cache and
     * norms for predictor layer46; it has no DSA score scratch. */
    const uint64_t scratch = dots3_graph_bytes(ctx, 1) - dots3_graph_cache_bytes(ctx, 1)
        - (norms + ctx) * sizeof(float);
    const unsigned ring = ctx < DS4_N_SWA + 1 ? ctx : DS4_N_SWA + 1;
    const unsigned hidden = cap < DOTS3_TRIAL_MAX ? DOTS3_TRIAL_MAX : cap;
    return scratch + (uint64_t)ring * (DS4_N_SWA_KV_LORA + DS4_N_ROT) * sizeof(uint16_t)
        + (uint64_t)(DS4_N_LORA_Q + DS4_N_SWA_KV_LORA + DS4_N_ROT) * sizeof(float)
        + (uint64_t)(hidden + 3) * DS4_N_EMBD * sizeof(float) + dots3_journal_bytes()
        + (uint64_t)(DOTS3_TRIAL_MAX + 1) * DS4_N_VOCAB * sizeof(float);
}

static void dots3_spec_free(ds4_dots3_spec *sp) {
    if (!sp) { return; }
    dots3_graph_free(&sp->draft);
    ds4_gpu_tensor_free(sp->joined); ds4_gpu_tensor_free(sp->carry);
    ds4_gpu_tensor_free(sp->hidden); ds4_gpu_tensor_free(sp->journal);
    free(sp->verify_logits); free(sp->draft_logits); free(sp);
}

static ds4_dots3_spec *dots3_spec_alloc(ds4_engine *e, unsigned ctx, unsigned cap) {
    ds4_dots3_spec *sp = xcalloc(1, sizeof(*sp));
    ds4_dots3_gpu_graph *g = &sp->draft;
    const unsigned il = DS4_N_LAYER - 1;
    const ds4_layer_weights *l = &e->weights.layer[il];
    const unsigned ring = ctx < DS4_N_SWA + 1 ? ctx : DS4_N_SWA + 1;
    if (!dots3_graph_alloc(g, 1)) { dots3_spec_free(sp); return NULL; }
    g->w_q_a_norm[il] = dots3_graph_upload_norm(&e->model, l->attn_q_a_norm, DS4_N_LORA_Q,
                                               sqrtf((float)DS4_N_EMBD / DS4_N_LORA_Q));
    g->w_kv_a_norm[il] = dots3_graph_upload_norm(&e->model, l->attn_kv_a_norm, DS4_N_SWA_KV_LORA,
                                                sqrtf((float)DS4_N_EMBD / DS4_N_SWA_KV_LORA));
    g->w_k_rope_norm[il] = dots3_graph_upload_norm(&e->model, l->attn_k_rope_norm, DS4_N_ROT, 1);
    g->layer_kv_latent[il] = ds4_gpu_tensor_alloc((uint64_t)ring * DS4_N_SWA_KV_LORA * sizeof(uint16_t));
    g->layer_k_pe[il] = ds4_gpu_tensor_alloc((uint64_t)ring * DS4_N_ROT * sizeof(uint16_t));
    g->layer_cache_cap[il] = ring;
    g->ctx_cap = ctx;
    g->cache_ready = true;
    sp->hidden_cap = cap < DOTS3_TRIAL_MAX ? DOTS3_TRIAL_MAX : cap;
    sp->joined = ds4_gpu_tensor_alloc(2ull * DS4_N_EMBD * sizeof(float));
    sp->carry = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    sp->hidden = ds4_gpu_tensor_alloc((uint64_t)sp->hidden_cap * DS4_N_EMBD * sizeof(float));
    sp->journal = ds4_gpu_tensor_alloc(dots3_journal_bytes());
    sp->verify_logits = xcalloc((size_t)DOTS3_TRIAL_MAX * DS4_N_VOCAB, sizeof(float));
    sp->draft_logits = xcalloc(DS4_N_VOCAB, sizeof(float));
    if (!g->w_q_a_norm[il] || !g->w_kv_a_norm[il] || !g->w_k_rope_norm[il] ||
        !g->layer_kv_latent[il] || !g->layer_k_pe[il] || !sp->joined || !sp->carry || !sp->hidden || !sp->journal) {
        dots3_spec_free(sp); return NULL;
    }
    return sp;
}

static void dots3_spec_reset(ds4_dots3_spec *sp) {
    if (!sp) { return; }
    sp->draft.cache_len = 0;
    sp->target_pos = sp->trial_n = 0;
}

/* The draft never borrows a target cache or scratch pointer. The recursive
 * input is its normalized output; teacher extension passes target hidden. */
static bool dots3_draft_row(ds4_dots3_spec *sp, ds4_engine *e,
                             int token, ds4_gpu_tensor *previous, float *logits) {
    ds4_dots3_gpu_graph *g = &sp->draft;
    const unsigned il = DS4_N_LAYER - 1;
    const ds4_layer_weights *l = &e->weights.layer[il];
    const int32_t pos = (int32_t)g->cache_len, tok = token;
    if (!previous || token < 0 || (unsigned)token >= DS4_N_VOCAB || g->cache_len >= g->ctx_cap) { return false; }
    ds4_gpu_tensor *left = ds4_gpu_tensor_view(sp->joined, 0, (uint64_t)DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *right = ds4_gpu_tensor_view(sp->joined, (uint64_t)DS4_N_EMBD * sizeof(float),
                                                (uint64_t)DS4_N_EMBD * sizeof(float));
    bool ok = left && right && ds4_gpu_tensor_write(g->tokens, 0, &tok, sizeof(tok)) &&
        ds4_gpu_tensor_write(g->positions, 0, &pos, sizeof(pos)) &&
        ds4_gpu_embed_tokens_q8_0_tensor(g->x, g->tokens, e->model.map, e->model.size,
            e->weights.dots3_token_embd_mtp->abs_offset, DS4_N_VOCAB, 1, DS4_N_EMBD) &&
        ds4_gpu_rms_norm_weight_rows_tensor(left, g->x, e->model.map, e->model.size,
            l->nextn_enorm->abs_offset, DS4_N_EMBD, 1, DS4_RMS_EPS) &&
        ds4_gpu_rms_norm_weight_rows_tensor(right, previous, e->model.map, e->model.size,
            l->nextn_hnorm->abs_offset, DS4_N_EMBD, 1, DS4_RMS_EPS);
    ds4_gpu_tensor_free(left); ds4_gpu_tensor_free(right);
    if (!ok) { return false; }
    ok = ds4_gpu_matmul_q8_0_tensor(g->x, e->model.map, e->model.size, l->nextn_eh_proj->abs_offset,
            2ull * DS4_N_EMBD, DS4_N_EMBD, sp->joined, 1) &&
        ds4_gpu_rms_norm_weight_rows_tensor(g->norm, g->x, e->model.map, e->model.size,
            l->attn_norm->abs_offset, DS4_N_EMBD, 1, DS4_RMS_EPS) &&
        dots3_graph_attention(g, &e->model, l, 1, (unsigned)pos, il) &&
        ds4_gpu_exaone_add_tensor(g->x, g->block_out, DS4_N_EMBD) &&
        ds4_gpu_rms_norm_weight_rows_tensor(g->norm, g->x, e->model.map, e->model.size,
            l->ffn_norm->abs_offset, DS4_N_EMBD, 1, DS4_RMS_EPS) &&
        dots3_graph_ffn(g, &e->model, l, 1, il) &&
        ds4_gpu_exaone_add_tensor(g->x, g->ffn_out, DS4_N_EMBD) &&
        ds4_gpu_rms_norm_weight_rows_tensor(g->final_norm, g->x, e->model.map, e->model.size,
            l->nextn_shared_head_norm->abs_offset, DS4_N_EMBD, 1, DS4_RMS_EPS);
    if (ok && logits) {
        ok = ds4_gpu_matmul_q8_0_tensor(g->logits, e->model.map, e->model.size,
                e->weights.output->abs_offset, DS4_N_EMBD, DS4_N_VOCAB, g->final_norm, 1) &&
            ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float));
    }
    if (ok) { g->cache_len++; }
    return ok;
}

static bool dots3_hidden_rows(ds4_dots3_spec *sp, ds4_engine *e,
                               ds4_dots3_gpu_graph *target, unsigned row, unsigned rows) {
    if (row + rows > sp->hidden_cap) { return false; }
    ds4_gpu_tensor *out = ds4_gpu_tensor_view(sp->hidden, (uint64_t)row * DS4_N_EMBD * sizeof(float),
                                              (uint64_t)rows * DS4_N_EMBD * sizeof(float));
    const bool ok = out && ds4_gpu_rms_norm_weight_rows_tensor(out, target->x,
        e->model.map, e->model.size, e->weights.output_norm->abs_offset, DS4_N_EMBD, rows, DS4_RMS_EPS);
    ds4_gpu_tensor_free(out);
    return ok;
}

/* Stable draft frontier is one token behind the target: the final hidden
 * needs the next real token before it can form a teacher row. */
static bool dots3_spec_extend(ds4_dots3_spec *sp, ds4_engine *e,
                               const int *tokens, unsigned rows) {
    if (!sp || !rows || rows > sp->hidden_cap) { return false; }
    const uint64_t bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    for (unsigned i = 0; i < rows; i++) {
        if (sp->target_pos && !dots3_draft_row(sp, e, tokens[i], sp->carry, NULL)) { return false; }
        if (!ds4_gpu_tensor_copy(sp->carry, 0, sp->hidden, (uint64_t)i * bytes, bytes)) { return false; }
        sp->target_pos++;
    }
    return true;
}

typedef enum { DOTS3_UNDO_SAVE, DOTS3_UNDO_LOAD } dots3_undo_dir;
static bool dots3_spec_undo(ds4_dots3_spec *sp, ds4_dots3_gpu_graph *target,
                             unsigned start, unsigned n, unsigned keep, dots3_undo_dir dir) {
    uint64_t off = 0;
    for (unsigned il = 0; il < DS4_N_LAYER; il++) {
        const bool draft = il + 1 == DS4_N_LAYER;
        ds4_dots3_gpu_graph *g = draft ? &sp->draft : target;
        const unsigned pos = draft ? start - 1 : start;
        const unsigned cap = g->layer_cache_cap[il];
        ds4_gpu_tensor *parts[] = {g->layer_kv_latent[il], g->layer_k_pe[il], g->layer_idx_k[il]};
        const uint64_t widths[] = {ds4_dots3_layer_kv_lora(il) * sizeof(uint16_t),
            DS4_N_ROT * sizeof(uint16_t), DS4_N_INDEXER_HEAD_DIM * sizeof(float)};
        for (unsigned part = 0; part < 3; part++) {
            if (!parts[part]) { continue; }
            for (unsigned i = dir == DOTS3_UNDO_SAVE || draft ? 0 : keep; i < n; i++) {
                const uint64_t slot = (pos + i) % cap;
                const bool ok = dir == DOTS3_UNDO_SAVE
                    ? ds4_gpu_tensor_copy(sp->journal, off + i * widths[part], parts[part], slot * widths[part], widths[part])
                    : ds4_gpu_tensor_copy(parts[part], slot * widths[part], sp->journal, off + i * widths[part], widths[part]);
                if (!ok) { return false; }
            }
            off += DOTS3_TRIAL_MAX * widths[part];
        }
    }
    return true;
}

static int dots3_spec_fail(ds4_session *s, char *err, size_t errlen) {
    ds4_session_invalidate(s);
    if (err && errlen) { snprintf(err, errlen, "dots3 MTP device state failed"); }
    return -1;
}

int ds4_session_dots3_trial(ds4_session *s, int first, int max_tokens,
                             int *tokens, int *target, int cap, char *err, size_t errlen) {
    if (!s || !tokens || !target || cap <= 0 || DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_DOTS3_NOTE) { return -1; }
    ds4_dots3_spec *sp = s->dots3_spec;
    if (max_tokens <= 0 || !sp) { return 0; }
    ds4_dots3_gpu_graph *g = &s->dots3_graph;
    const unsigned base = (unsigned)s->checkpoint.len;
    if (!s->checkpoint_valid || sp->trial_n || !base || g->cache_len != base ||
        sp->target_pos != base || sp->draft.cache_len != base - 1 || first < 0 ||
        (unsigned)first >= DS4_N_VOCAB || base >= g->ctx_cap) {
        if (err && errlen) { snprintf(err, errlen, "dots3 trial needs a committed target and draft frontier"); }
        return -1;
    }
    unsigned n = (unsigned)max_tokens;
    if (n > (unsigned)cap) { n = (unsigned)cap; }
    if (n > DOTS3_TRIAL_MAX) { n = DOTS3_TRIAL_MAX; }
    if (n > (unsigned)s->engine->mtp_draft_tokens + 1) { n = (unsigned)s->engine->mtp_draft_tokens + 1; }
    if (n > g->ctx_cap - base) { n = g->ctx_cap - base; }
    if (!dots3_spec_undo(sp, g, base, n, 0, DOTS3_UNDO_SAVE)) { return dots3_spec_fail(s, err, errlen); }
    sp->trial[0] = first;
    for (unsigned i = 1; i < n; i++) {
        if (!dots3_draft_row(sp, s->engine, sp->trial[i - 1],
                              i == 1 ? sp->carry : sp->draft.final_norm, sp->draft_logits)) {
            return dots3_spec_fail(s, err, errlen);
        }
        sp->trial[i] = argmax_f32(sp->draft_logits, DS4_N_VOCAB);
    }
    /* Width-one target verification is the initial numerical contract. It
     * executes the predictor but makes no speculation throughput claim. */
    for (unsigned i = 0; i < n; i++) {
        float *row = sp->verify_logits + (size_t)i * DS4_N_VOCAB;
        if (!dots3_graph_forward_chunk(g, &s->engine->model, &s->engine->weights,
                                        &sp->trial[i], 1, base + i, row) ||
            !dots3_hidden_rows(sp, s->engine, g, i, 1)) { return dots3_spec_fail(s, err, errlen); }
        target[i] = argmax_f32(row, DS4_N_VOCAB);
        tokens[i] = sp->trial[i];
    }
    sp->trial_n = n;
    ds4_metric_add(&ds4_metrics_get()->spec_drafts, n - 1);
    return (int)n;
}

int ds4_session_dots3_commit(ds4_session *s, int keep, char *err, size_t errlen) {
    if (!s || !s->dots3_spec || !s->checkpoint_valid || keep <= 0 ||
        (unsigned)keep > s->dots3_spec->trial_n) { return 1; }
    ds4_dots3_spec *sp = s->dots3_spec;
    ds4_dots3_gpu_graph *g = &s->dots3_graph;
    const unsigned base = (unsigned)s->checkpoint.len;
    if (g->cache_len != base + sp->trial_n ||
        !dots3_spec_undo(sp, g, base, sp->trial_n, (unsigned)keep, DOTS3_UNDO_LOAD)) {
        (void)dots3_spec_fail(s, err, errlen); return 1;
    }
    g->cache_len = base + (unsigned)keep;
    sp->draft.cache_len = base - 1;
    if (!dots3_spec_extend(sp, s->engine, sp->trial, (unsigned)keep)) {
        (void)dots3_spec_fail(s, err, errlen); return 1;
    }
    const float *logits = sp->verify_logits + (size_t)(keep - 1) * DS4_N_VOCAB;
    if (!ds4_gpu_tensor_write(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) ||
        !ds4_gpu_synchronize()) { (void)dots3_spec_fail(s, err, errlen); return 1; }
    memcpy(s->logits, logits, (size_t)DS4_N_VOCAB * sizeof(float));
    for (int i = 0; i < keep; i++) { token_vec_push(&s->checkpoint, sp->trial[i]); }
    ds4_metric_add(&ds4_metrics_get()->spec_hits, (unsigned)keep - 1);
    sp->trial_n = 0;
    return 0;
}
