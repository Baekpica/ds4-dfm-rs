/* Serial session frontiers include every main KV and indexer row together. */
static bool iquest_reset(ds4_iquest_graph *g) {
    if (g->failed && !ds4_gpu_synchronize()) { return false; }
    g->position = 0; g->failed = false;
    g->mtp_position = 0; g->trial_n = 0;
    return true;
}

static int iquest_session_fail(ds4_session *s, char *err, size_t errlen) {
    s->iquest_graph.failed = true;
    /* Discard the failed journal so sync can reach the synchronized reset. */
    s->iquest_graph.trial_n = 0;
    s->checkpoint_valid = false; s->checkpoint.len = 0;
    s->mtp_draft_valid = false; s->generation++;
    payload_set_err(err, errlen, "IQuest-Q1 forward or logits readback failed");
    return 1;
}

static bool iquest_read_logits(ds4_session *s) {
    if (!ds4_gpu_tensor_read(s->iquest_graph.logits, 0, s->logits, IQ_VOCAB * sizeof(float))) { return false; }
    for (unsigned i = 0; i < IQ_VOCAB; i++) { if (!isfinite(s->logits[i])) { return false; } }
    return true;
}

static int iquest_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    if (!prompt->v) { payload_set_err(err, errlen, "IQuest-Q1 sync needs tokens"); return 1; }
    for (int i = 0; i < prompt->len; i++) {
        if (prompt->v[i] < 0 || (unsigned)prompt->v[i] >= IQ_VOCAB) {
            payload_set_err(err, errlen, "IQuest-Q1 prompt has an invalid token"); return 1;
        }
    }
    if (ds4_session_ensure_graph(s, err, errlen)) { return 1; }
    ds4_iquest_graph *g = &s->iquest_graph;
    if (g->trial_n) { payload_set_err(err, errlen, "IQuest-Q1 trial needs commit or invalidation"); return 1; }
    if (!s->iquest_graph_ready) { payload_set_err(err, errlen, "IQuest-Q1 graph is not initialized"); return 1; }
    unsigned start = 0;
    if (s->checkpoint_valid && !g->failed && g->position == (unsigned)s->checkpoint.len &&
        prompt->len >= s->checkpoint.len && ds4_tokens_starts_with(prompt, &s->checkpoint)) {
        start = (unsigned)s->checkpoint.len;
        if (start == (unsigned)prompt->len) { return 0; }
    } else {
        ds4_session_invalidate(s);
        if (g->failed) { payload_set_err(err, errlen, "IQuest-Q1 graph reset failed"); return 1; }
    }
    while (start < (unsigned)prompt->len) {
        const unsigned remaining = (unsigned)prompt->len - start;
        const unsigned n = remaining < g->cap ? remaining : g->cap;
        if (!iquest_forward(g, &s->engine->model, &s->engine->weights, prompt->v + start, n, start,
                ds4_engine_has_mtp(s->engine) ? IQ_MTP_ON : IQ_MTP_OFF)) {
            return iquest_session_fail(s, err, errlen);
        }
        start += n;
        if (s->progress) { s->progress(s->progress_ud, "prefill_chunk", (int)start, prompt->len); }
    }
    if (!iquest_read_logits(s)) { return iquest_session_fail(s, err, errlen); }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true; s->mtp_draft_valid = false;
    return 0;
}

static int iquest_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    ds4_iquest_graph *g = &s->iquest_graph;
    if (token < 0 || (unsigned)token >= IQ_VOCAB || !s->iquest_graph_ready ||
        !s->checkpoint_valid || g->failed || g->trial_n || g->position >= g->context ||
        g->position != (unsigned)s->checkpoint.len) {
        payload_set_err(err, errlen, "IQuest-Q1 decode needs a valid token and checkpoint"); return 1;
    }
    if (!iquest_forward(g, &s->engine->model, &s->engine->weights, &token, 1, g->position,
            ds4_engine_has_mtp(s->engine) ? IQ_MTP_ON : IQ_MTP_OFF) ||
        !iquest_read_logits(s)) { return iquest_session_fail(s, err, errlen); }
    token_vec_push(&s->checkpoint, token); s->mtp_draft_valid = false;
    return 0;
}

/* Recursive MTP candidates are teacher-verified with the target's ordinary
 * one-row arithmetic. Journals retain each displaced Q8 row; partial accept
 * restores every rejected target and MTP write before publishing a frontier. */
int ds4_session_iquest_trial(ds4_session *s, int first, int max_tokens,
        int *tokens, int *target, int capacity, char *err, size_t errlen) {
    if (!s || !ds4_session_is_iquest(s) || !tokens || !target || capacity < 1 ||
        first < 0 || first >= IQ_VOCAB || !s->checkpoint_valid || !s->iquest_graph_ready) {
        payload_set_err(err, errlen, "invalid IQuest-Q1 MTP trial"); return -1;
    }
    ds4_iquest_graph *g = &s->iquest_graph;
    if (!ds4_engine_has_mtp(s->engine) || max_tokens < 2) { return 0; }
    if (g->failed || g->trial_n || !g->position || g->position != (unsigned)s->checkpoint.len ||
        g->mtp_position != g->position - 1) {
        payload_set_err(err, errlen, "IQuest-Q1 MTP needs a committed target and draft frontier"); return -1;
    }
    unsigned n = (unsigned)max_tokens;
    if (n > (unsigned)capacity) { n = (unsigned)capacity; }
    if (n > (unsigned)s->engine->mtp_draft_tokens + 1) { n = (unsigned)s->engine->mtp_draft_tokens + 1; }
    if (n > IQ_DRAFT_SLOTS + 1) { n = IQ_DRAFT_SLOTS + 1; }
    if (n > g->context - g->position) { n = g->context - g->position; }
    if (n < 2) { return 0; }
    const uint64_t row = IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8);
    const unsigned base = g->position, mtp_base = g->mtp_position;
    bool ok = ds4_gpu_tensor_copy(g->mtp_backup, 0, g->mtp_kv, 0,
        ds4_gpu_tensor_bytes(g->mtp_kv));
    tokens[0] = first;
    for (unsigned k = 0; ok && k + 1 < n; k++) {
        const ds4_gpu_tensor *prior = k ? g->draft_carry : g->carry;
        ok = iquest_mtp_forward(g, &s->engine->model, &s->engine->weights,
            &tokens[k], prior, 1, mtp_base + k, IQ_HAS_LOGITS) &&
            ds4_gpu_tensor_read(g->mtp_logits, 0, g->draft_logits, IQ_VOCAB * sizeof(float));
        for (unsigned j = 0; ok && j < IQ_VOCAB; j++) { ok = isfinite(g->draft_logits[j]); }
        if (ok) {
            int candidate = -1;
            float best = 0.0f, runner = 0.0f;
            logits_top2(g->draft_logits, IQ_VOCAB, &candidate, &best, NULL, &runner);
            /* Retain only the contiguous confident recursive prefix. Margin
             * zero disables this cutoff, including a tied top logit. */
            if (best - runner < s->engine->mtp_margin) { n = k + 1; break; }
            tokens[k + 1] = candidate;
        }
    }
    ok = ok && ds4_gpu_tensor_copy(g->mtp_kv, 0, g->mtp_backup, 0,
        ds4_gpu_tensor_bytes(g->mtp_kv));
    g->mtp_position = mtp_base;
    if (!ok) { iquest_session_fail(s, err, errlen); return -1; }
    if (n < 2) { return 0; }  /* caller performs ordinary target continuation */
    for (unsigned k = 0; ok && k < n; k++) {
        const unsigned p = base + k;
        for (unsigned il = 0; ok && il < IQ_LAYERS; il++) {
            ok = ds4_gpu_tensor_copy(g->trial_kv, ((uint64_t)k * IQ_LAYERS + il) * row,
                g->kv[il], (uint64_t)(p % g->kv_cap[il]) * row, row);
        }
        ok = ok && ds4_gpu_tensor_copy(g->trial_mtp, (uint64_t)k * row,
            g->mtp_kv, (uint64_t)((p - 1) % (IQ_DRAFT_WINDOW + IQ_DRAFT_SLOTS)) * row, row) &&
            iquest_forward(g, &s->engine->model, &s->engine->weights, &tokens[k], 1, p, IQ_MTP_ON) &&
            ds4_gpu_tensor_copy(g->trial_hidden, (uint64_t)k * IQ_EMBED * sizeof(float),
                g->carry, 0, IQ_EMBED * sizeof(float));
        float *logits = g->trial_logits + (size_t)k * IQ_VOCAB;
        ok = ok && ds4_gpu_tensor_read(g->logits, 0, logits, IQ_VOCAB * sizeof(float));
        for (unsigned j = 0; ok && j < IQ_VOCAB; j++) { ok = isfinite(logits[j]); }
        if (ok) { target[k] = sample_argmax(logits, IQ_VOCAB); }
    }
    if (!ok) { iquest_session_fail(s, err, errlen); return -1; }
    memcpy(g->trial_tokens, tokens, n * sizeof(int));
    g->trial_base = base; g->trial_n = n;
    ds4_metric_add(&ds4_metrics_get()->spec_drafts, n - 1);
    return (int)n;
}

int ds4_session_iquest_commit(ds4_session *s, int keep, char *err, size_t errlen) {
    if (!s || !ds4_session_is_iquest(s) || !s->iquest_graph_ready) { return 1; }
    ds4_iquest_graph *g = &s->iquest_graph;
    if (g->failed || !g->trial_n || keep < 1 || (unsigned)keep > g->trial_n ||
        !s->checkpoint_valid || (unsigned)s->checkpoint.len != g->trial_base) {
        payload_set_err(err, errlen, "invalid IQuest-Q1 MTP commit"); return 1;
    }
    const uint64_t row = IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8);
    for (unsigned k = (unsigned)keep; k < g->trial_n; k++) {
        const unsigned p = g->trial_base + k;
        for (unsigned il = 0; il < IQ_LAYERS; il++) {
            if (!ds4_gpu_tensor_copy(g->kv[il], (uint64_t)(p % g->kv_cap[il]) * row,
                g->trial_kv, ((uint64_t)k * IQ_LAYERS + il) * row, row)) { return iquest_session_fail(s, err, errlen); }
        }
        if (!ds4_gpu_tensor_copy(g->mtp_kv,
                (uint64_t)((p - 1) % (IQ_DRAFT_WINDOW + IQ_DRAFT_SLOTS)) * row,
                g->trial_mtp, (uint64_t)k * row, row)) { return iquest_session_fail(s, err, errlen); }
    }
    if (!ds4_gpu_tensor_copy(g->carry, 0, g->trial_hidden,
        (uint64_t)(keep - 1) * IQ_EMBED * sizeof(float), IQ_EMBED * sizeof(float)) ||
        !ds4_gpu_tensor_write(g->logits, 0,
            g->trial_logits + (size_t)(keep - 1) * IQ_VOCAB, IQ_VOCAB * sizeof(float)) ||
        !ds4_gpu_synchronize()) { return iquest_session_fail(s, err, errlen); }
    memcpy(s->logits, g->trial_logits + (size_t)(keep - 1) * IQ_VOCAB, IQ_VOCAB * sizeof(float));
    for (int k = 0; k < keep; k++) { token_vec_push(&s->checkpoint, g->trial_tokens[k]); }
    g->position = g->trial_base + (unsigned)keep;
    const unsigned trial = g->trial_n;
    g->mtp_position = g->position - 1; g->trial_n = 0;
    ds4_metric_add(&ds4_metrics_get()->spec_hits, (uint64_t)keep - 1);
    if (getenv("DS4_MTP_SPEC_LOG")) {
        fprintf(stderr, "ds4: IQuest-Q1 MTP trial=%u keep=%d pos=%u margin=%g\n",
                trial, keep, g->position, s->engine->mtp_margin);
    }
    return 0;
}
