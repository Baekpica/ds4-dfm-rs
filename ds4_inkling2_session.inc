/* Serial session: one graph owns every layer's K/V (full layers by
 * position, sliding layers in a ring) and the convolution histories, which
 * describe only the live frontier: a prompt that extends the checkpoint
 * continues it, any other prompt rebuilds. */
static int inkling2_session_fail(ds4_session *s, char *err, size_t errlen) {
    s->inkling2_graph.failed = true;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
    s->generation++;
    payload_set_err(err, errlen, "Inkling-2 forward or logits readback failed");
    return 1;
}

static bool inkling2_read_logits(ds4_session *s) {
    if (!ds4_gpu_tensor_read(s->inkling2_graph.logits, 0, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float))) {
        return false;
    }
    for (uint32_t i = 0; i < DS4_INK2_TOKENS; i++) {
        if (!isfinite(s->logits[i])) { return false; }
    }
    return true;
}

/* Rows the live state keeps for `prompt`; 0 rebuilds. */
static uint32_t inkling2_sync_keep(ds4_session *s, const ds4_tokens *prompt) {
    const ds4_inkling2_graph *g = &s->inkling2_graph;
    if (!s->checkpoint_valid || g->failed || g->position != (uint32_t)s->checkpoint.len) { return 0; }
    const uint32_t keep = session_keep_prefix(s, prompt);
    return keep == g->position ? keep : 0u;
}

static int inkling2_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    for (int i = 0; i < prompt->len; i++) {
        if (prompt->v[i] < 0 || (uint32_t)prompt->v[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "Inkling-2 prompt has an invalid token");
            return 1;
        }
    }
    if (ds4_session_ensure_graph(s, err, errlen)) { return 1; }
    if (!s->inkling2_graph_ready) { payload_set_err(err, errlen, "Inkling-2 graph is not initialized"); return 1; }
    ds4_inkling2_graph *g = &s->inkling2_graph;
    uint32_t start = inkling2_sync_keep(s, prompt);
    if (start && start == (uint32_t)prompt->len) { return 0; }
    if (!start) {
        ds4_session_invalidate(s);
        if (g->failed) { payload_set_err(err, errlen, "Inkling-2 graph reset failed"); return 1; }
    }
    const uint32_t end = (uint32_t)prompt->len;
    while (start < end) {
        const uint32_t n = end - start < g->row_cap ? end - start : g->row_cap;
        // The next chunk's embedding rows load while this one computes.
        if (start + n < end) {
            const uint32_t next = end - start - n < g->row_cap ? end - start - n : g->row_cap;
            inkling2_embed_prefetch(&s->engine->model, &s->engine->weights, prompt->v + start + n, next);
        }
        // Only the final chunk's last row needs the vocabulary head.
        if (!inkling2_forward(g, &s->engine->model, &s->engine->weights, prompt->v + start, n, start,
                start + n == end ? INKLING2_HEAD_LAST : INKLING2_HEAD_NONE)) {
            return inkling2_session_fail(s, err, errlen);
        }
        start += n;
        if (s->progress) { s->progress(s->progress_ud, "prefill_chunk", (int)start, prompt->len); }
    }
    if (!inkling2_read_logits(s)) { return inkling2_session_fail(s, err, errlen); }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    return 0;
}

static int inkling2_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    ds4_inkling2_graph *g = &s->inkling2_graph;
    if (token < 0 || (uint32_t)token >= DS4_N_VOCAB || !s->inkling2_graph_ready || !s->checkpoint_valid ||
        g->failed || g->position >= g->ctx_cap || g->position != (uint32_t)s->checkpoint.len) {
        payload_set_err(err, errlen, "Inkling-2 decode needs a valid token and checkpoint");
        return 1;
    }
    if (!inkling2_forward(g, &s->engine->model, &s->engine->weights, &token, 1u, g->position, INKLING2_HEAD_LAST) ||
        !inkling2_read_logits(s)) { return inkling2_session_fail(s, err, errlen); }
    token_vec_push(&s->checkpoint, token);
    s->mtp_draft_valid = false;
    return 0;
}

