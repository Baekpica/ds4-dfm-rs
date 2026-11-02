/* Serial session: one graph owns every SWA ring and full-layer FP8 row. A
 * prompt resumes at its common prefix while the rings (the window and one
 * chunk) still hold the window under it. */
static int beam_session_fail(ds4_session *s, char *err, size_t errlen) {
    s->beam_graph.failed = true;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
    s->generation++;
    payload_set_err(err, errlen, "Beam forward or logits readback failed");
    return 1;
}

static bool beam_read_logits(ds4_session *s) {
    if (!ds4_gpu_tensor_read(s->beam_graph.logits, 0, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float))) {
        return false;
    }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(s->logits[i])) { return false; }
    }
    return true;
}

/* Rows the live state keeps for `prompt`; 0 rebuilds. */
static uint32_t beam_sync_keep(ds4_session *s, const ds4_tokens *prompt) {
    const ds4_beam_graph *g = &s->beam_graph;
    if (!s->checkpoint_valid || g->failed || g->position != (uint32_t)s->checkpoint.len) { return 0; }
    const uint32_t keep = session_keep_prefix(s, prompt);
    return ring_holds(g->swa_cap, BEAM_WINDOW, g->high, g->restored, keep) ? keep : 0;
}

static int beam_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    for (int i = 0; i < prompt->len; i++) {
        if (prompt->v[i] < 0 || (uint32_t)prompt->v[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "Beam prompt has an invalid token");
            return 1;
        }
    }
    if (ds4_session_ensure_graph(s, err, errlen)) { return 1; }
    if (!s->beam_graph_ready) { payload_set_err(err, errlen, "Beam graph is not initialized"); return 1; }
    ds4_beam_graph *g = &s->beam_graph;
    uint32_t start = beam_sync_keep(s, prompt);
    const uint32_t live = (uint32_t)s->checkpoint.len;
    if (!start) {
        ds4_session_invalidate(s);
        if (g->failed) { payload_set_err(err, errlen, "Beam graph reset failed"); return 1; }
    } else if (start == (uint32_t)prompt->len) {
        return 0;
    } else if (start < live) {
        /* The old branch's verify rows (up to live + a block) displaced the
         * DFlash2 rings' rows below live - window: the draft context restarts
         * above them, or at the resume row. */
        const uint32_t lost = live > DS4_BEAM_DF_WINDOW ? live - DS4_BEAM_DF_WINDOW : 0u;
        if (g->draft_first < lost) { g->draft_first = lost < start ? lost : start; }
        g->position = start;
        session_rewound(s, start);
    }
    const uint32_t end = (uint32_t)prompt->len;
    while (start < end) {
        const uint32_t n = end - start < g->row_cap ? end - start : g->row_cap;
        // The next chunk's embedding rows load while this one computes.
        if (start + n < end) {
            const uint32_t next = end - start - n < g->row_cap ? end - start - n : g->row_cap;
            beam_embed_prefetch(&s->engine->model, &s->engine->weights, prompt->v + start + n, next);
        }
        // Only the final chunk's last row needs the vocabulary head.
        const beam_head_mode head = start + n == end ? BEAM_HEAD_LAST : BEAM_HEAD_SKIP;
        if (!beam_forward(g, &s->engine->model, &s->engine->weights, prompt->v + start, n, start, head)) {
            return beam_session_fail(s, err, errlen);
        }
        start += n;
        if (s->progress) { s->progress(s->progress_ud, "prefill_chunk", (int)start, prompt->len); }
    }
    if (!beam_read_logits(s)) { return beam_session_fail(s, err, errlen); }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    return 0;
}

static int beam_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    ds4_beam_graph *g = &s->beam_graph;
    if (token < 0 || (uint32_t)token >= DS4_N_VOCAB || !s->beam_graph_ready || !s->checkpoint_valid ||
        g->failed || g->position >= g->ctx_cap || g->position != (uint32_t)s->checkpoint.len) {
        payload_set_err(err, errlen, "Beam decode needs a valid token and checkpoint");
        return 1;
    }
    if (!beam_forward(g, &s->engine->model, &s->engine->weights, &token, 1u, g->position, BEAM_HEAD_LAST) ||
        !beam_read_logits(s)) { return beam_session_fail(s, err, errlen); }
    token_vec_push(&s->checkpoint, token);
    s->mtp_draft_valid = false;
    return 0;
}
