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
        g->failed || g->position >= g->ctx_cap || g->position != (uint32_t)s->checkpoint.len) {
        payload_set_err(err, errlen, "Ling-3.1 decode needs a valid token and checkpoint");
        return 1;
    }
    if (!ling31_forward(g, &s->engine->model, &s->engine->weights, &token, 1u, g->position, LING31_HEAD_LAST) ||
        !ling31_read_logits(s)) { return ling31_session_fail(s, err, errlen); }
    token_vec_push(&s->checkpoint, token);
    s->mtp_draft_valid = false;
    return 0;
}
