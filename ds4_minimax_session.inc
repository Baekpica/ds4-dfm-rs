/* Serial session: one graph owns every layer's index keys and K/V pages. A
 * prompt that extends the checkpoint continues in place; any other prompt
 * prefills from 0. */
static int minimax_session_fail(ds4_session *s, char *err, size_t errlen) {
    s->minimax_graph.failed = true;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
    s->generation++;
    payload_set_err(err, errlen, "MiniMax forward or logits readback failed");
    return 1;
}

static bool minimax_read_logits(ds4_session *s) {
    if (!ds4_gpu_tensor_read(s->minimax_graph.logits, 0, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float))) {
        return false;
    }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(s->logits[i])) { return false; }
    }
    return true;
}

static int minimax_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    for (int i = 0; i < prompt->len; i++) {
        if (prompt->v[i] < 0 || (uint32_t)prompt->v[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "MiniMax prompt has an invalid token");
            return 1;
        }
    }
    if (ds4_session_ensure_graph(s, err, errlen)) { return 1; }
    if (!s->minimax_graph_ready) { payload_set_err(err, errlen, "MiniMax graph is not initialized"); return 1; }
    ds4_minimax_graph *g = &s->minimax_graph;
    uint32_t start = 0;
    if (s->checkpoint_valid && !g->failed && g->position == (uint32_t)s->checkpoint.len &&
        prompt->len >= s->checkpoint.len && ds4_tokens_starts_with(prompt, &s->checkpoint)) {
        start = (uint32_t)s->checkpoint.len;
        if (start == (uint32_t)prompt->len) { return 0; }
    } else {
        ds4_session_invalidate(s);
        if (g->failed) { payload_set_err(err, errlen, "MiniMax graph reset failed"); return 1; }
    }
    const uint32_t end = (uint32_t)prompt->len;
    while (start < end) {
        const uint32_t n = end - start < g->row_cap ? end - start : g->row_cap;
        // The next chunk's embedding rows load while this one computes.
        if (start + n < end) {
            const uint32_t next = end - start - n < g->row_cap ? end - start - n : g->row_cap;
            minimax_embed_prefetch(&s->engine->model, &s->engine->weights, prompt->v + start + n, next);
        }
        // Only the final chunk's last row needs the vocabulary head.
        const minimax_head_mode head = start + n == end ? MINIMAX_HEAD_LAST : MINIMAX_HEAD_SKIP;
        if (!minimax_forward(g, &s->engine->model, &s->engine->weights, prompt->v + start, n, start, head)) {
            return minimax_session_fail(s, err, errlen);
        }
        start += n;
        if (s->progress) { s->progress(s->progress_ud, "prefill_chunk", (int)start, prompt->len); }
    }
    if (!minimax_read_logits(s)) { return minimax_session_fail(s, err, errlen); }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    return 0;
}

static int minimax_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    ds4_minimax_graph *g = &s->minimax_graph;
    if (token < 0 || (uint32_t)token >= DS4_N_VOCAB || !s->minimax_graph_ready || !s->checkpoint_valid ||
        g->failed || g->position >= g->ctx_cap || g->position != (uint32_t)s->checkpoint.len) {
        payload_set_err(err, errlen, "MiniMax decode needs a valid token and checkpoint");
        return 1;
    }
    if (!minimax_forward(g, &s->engine->model, &s->engine->weights, &token, 1u, g->position, MINIMAX_HEAD_LAST) ||
        !minimax_read_logits(s)) { return minimax_session_fail(s, err, errlen); }
    token_vec_push(&s->checkpoint, token);
    s->mtp_draft_valid = false;
    return 0;
}
