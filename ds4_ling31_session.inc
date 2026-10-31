/* Serial session: one graph owns every KDA tile, MLA latent row and the
 * carried hidden row. A prompt that extends the checkpoint continues in
 * place; any other prompt resets the recurrent state and prefills from 0
 * (a KDA tile cannot rewind). */
static int ling31_session_fail(ds4_session *s, char *err, size_t errlen) {
    s->ling31_graph.failed = true;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
    s->generation++;
    payload_set_err(err, errlen, "Ling-3.1 forward or logits readback failed");
    return 1;
}

static bool ling31_read_logits(ds4_session *s) {
    if (!ds4_gpu_tensor_read(s->ling31_graph.logits, 0, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float))) {
        return false;
    }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(s->logits[i])) { return false; }
    }
    return true;
}

static int ling31_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    for (int i = 0; i < prompt->len; i++) {
        if (prompt->v[i] < 0 || (uint32_t)prompt->v[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "Ling-3.1 prompt has an invalid token");
            return 1;
        }
    }
    if (ds4_session_ensure_graph(s, err, errlen)) { return 1; }
    if (!s->ling31_graph_ready) { payload_set_err(err, errlen, "Ling-3.1 graph is not initialized"); return 1; }
    ds4_ling31_graph *g = &s->ling31_graph;
    if (g->mtp_n) { payload_set_err(err, errlen, "Ling-3.1 MTP trial needs commit or invalidation"); return 1; }
    uint32_t start = 0;
    if (s->checkpoint_valid && !g->failed && g->position == (uint32_t)s->checkpoint.len &&
        prompt->len >= s->checkpoint.len && ds4_tokens_starts_with(prompt, &s->checkpoint)) {
        start = (uint32_t)s->checkpoint.len;
        if (start == (uint32_t)prompt->len) { return 0; }
    } else {
        ds4_session_invalidate(s);
        if (g->failed) { payload_set_err(err, errlen, "Ling-3.1 graph reset failed"); return 1; }
    }
    const uint32_t end = (uint32_t)prompt->len;
    while (start < end) {
        const uint32_t n = end - start < g->row_cap ? end - start : g->row_cap;
        // Only the final chunk's last row needs the vocabulary head.
        const ling31_head_mode head = start + n == end ? LING31_HEAD_LAST : LING31_HEAD_SKIP;
        if (!ling31_forward(g, &s->engine->model, &s->engine->weights, prompt->v + start, n, start, head)) {
            return ling31_session_fail(s, err, errlen);
        }
        start += n;
        if (s->progress) { s->progress(s->progress_ud, "prefill_chunk", (int)start, prompt->len); }
    }
    if (!ling31_read_logits(s)) { return ling31_session_fail(s, err, errlen); }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    return 0;
}

static int ling31_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    ds4_ling31_graph *g = &s->ling31_graph;
    if (token < 0 || (uint32_t)token >= DS4_N_VOCAB || !s->ling31_graph_ready || !s->checkpoint_valid ||
        g->failed || g->mtp_n || g->position >= g->ctx_cap || g->position != (uint32_t)s->checkpoint.len) {
        payload_set_err(err, errlen, "Ling-3.1 decode needs a valid token and checkpoint");
        return 1;
    }
    if (!ling31_forward(g, &s->engine->model, &s->engine->weights, &token, 1u, g->position, LING31_HEAD_LAST) ||
        !ling31_read_logits(s)) { return ling31_session_fail(s, err, errlen); }
    token_vec_push(&s->checkpoint, token);
    s->mtp_draft_valid = false;
    return 0;
}

#ifndef __APPLE__
/* MTP trial: recursive drafts from the last hidden row, then one batched
 * verify of first + drafts. The caller commits the matching prefix. */
int ds4_session_ling31_trial(ds4_session *s, int first, int max_tokens,
        int *tokens, int *target, int capacity, char *err, size_t errlen) {
    if (!ds4_session_is_ling31(s) || !tokens || !target || capacity < 1 ||
        first < 0 || (uint32_t)first >= DS4_N_VOCAB) {
        payload_set_err(err, errlen, "invalid Ling-3.1 MTP trial");
        return -1;
    }
    ds4_ling31_graph *g = &s->ling31_graph;
    if (max_tokens < 2 || !s->ling31_graph_ready || !g->mtp_drafts) { return 0; }
    if (!s->checkpoint_valid || g->failed || g->mtp_n || !g->position ||
        g->position != (uint32_t)s->checkpoint.len || g->mtp_len != g->position - 1u) {
        payload_set_err(err, errlen, "Ling-3.1 MTP needs a committed checkpoint");
        return -1;
    }
    uint32_t n = (uint32_t)max_tokens;
    if (n > (uint32_t)capacity) { n = (uint32_t)capacity; }
    if (n > g->mtp_drafts + 1u) { n = g->mtp_drafts + 1u; }
    if (n > g->ctx_cap - g->position) { n = g->ctx_cap - g->position; }
    if (n < 2u) { return 0; }
    const ds4_model *m = &s->engine->model;
    const ds4_weights *w = &s->engine->weights;
    const uint32_t start = g->position;
    g->mtp_trial[0] = first;
    for (uint32_t i = 0; i + 1u < n; i++) {
        const ds4_gpu_tensor *hidden = i ? g->mtp_hidden : g->last_hidden;
        if (!l31_mtp_step(g, m, w, hidden, g->mtp_trial[i], start - 1u + i, &g->mtp_trial[i + 1u])) {
            ling31_session_fail(s, err, errlen);
            return -1;
        }
    }
    if (!l31_mtp_verify(g, m, w, n, start, target)) {
        ling31_session_fail(s, err, errlen);
        return -1;
    }
    if (spec_sampling(&s->spec)) {
        const float *rows = spec_host_rows(g->mtp_logits, DS4_N_VOCAB, n);
        if (!rows) {
            ling31_session_fail(s, err, errlen);
            return -1;
        }
        spec_sample_rows(&s->spec, rows, DS4_N_VOCAB, DS4_N_VOCAB, g->mtp_trial, n, target);
    }
    g->position = start + n;
    g->mtp_len = start + n - 1u;
    g->mtp_start = start;
    g->mtp_n = n;
    memcpy(tokens, g->mtp_trial, (size_t)n * sizeof(*tokens));
    ds4_metric_add(&ds4_metrics_get()->spec_drafts, n - 1u);
    return (int)n;
}

int ds4_session_ling31_commit(ds4_session *s, int keep, char *err, size_t errlen) {
    ds4_ling31_graph *g = &s->ling31_graph;
    if (!ds4_session_is_ling31(s) || !s->checkpoint_valid || !g->mtp_n || keep < 1 ||
        (uint32_t)keep > g->mtp_n || (uint32_t)s->checkpoint.len != g->mtp_start) {
        payload_set_err(err, errlen, "invalid Ling-3.1 MTP accepted prefix");
        return 1;
    }
    if (!l31_mtp_land(g, (uint32_t)keep, s->logits)) { return ling31_session_fail(s, err, errlen); }
    for (int i = 0; i < keep; i++) { token_vec_push(&s->checkpoint, g->mtp_trial[i]); }
    if (keep > 1) { ds4_metric_add(&ds4_metrics_get()->spec_hits, (uint64_t)keep - 1u); }
    s->mtp_draft_valid = false;
    if (getenv("DS4_MTP_SPEC_LOG")) {
        fprintf(stderr, "ds4: Ling-3.1 MTP keep=%d pos=%u\n", keep, g->position);
    }
    return 0;
}
#else
int ds4_session_ling31_trial(ds4_session *s, int first, int max_tokens,
        int *tokens, int *target, int capacity, char *err, size_t errlen) {
    (void)s; (void)first; (void)max_tokens; (void)tokens; (void)target; (void)capacity;
    payload_set_err(err, errlen, "Ling-3.1 MTP requires CUDA");
    return -1;
}
int ds4_session_ling31_commit(ds4_session *s, int keep, char *err, size_t errlen) {
    (void)s; (void)keep;
    payload_set_err(err, errlen, "Ling-3.1 MTP requires CUDA");
    return 1;
}
#endif
