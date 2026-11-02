/* Serial session: one graph owns every layer's index keys and K/V pages, all
 * indexed by position, so a prompt resumes at any common prefix. */
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

enum { MINIMAX_IMAGE_PAD = 200025, MINIMAX_VIDEO_PAD = 200026 };

static bool minimax_media_pad(int token) { return token == MINIMAX_IMAGE_PAD || token == MINIMAX_VIDEO_PAD; }

/* Rows the live state keeps for `prompt`; 0 rebuilds. */
static uint32_t minimax_sync_keep(ds4_session *s, const ds4_tokens *prompt) {
    const ds4_minimax_graph *g = &s->minimax_graph;
    if (!s->checkpoint_valid || g->failed || g->position != (uint32_t)s->checkpoint.len) { return 0; }
    return session_keep_prefix(s, prompt);
}

/* `tag` != 0: a media sync whose spans the graph holds (g->span_*). A prompt
 * extends the checkpoint only under the same media tag; identical pads over
 * other pixels prefill again. A pad no span covers is refused: its
 * embedding row is only a placeholder. */
static int minimax_session_prefill(ds4_session *s, const ds4_tokens *prompt, uint64_t tag, char *err,
        size_t errlen) {
    for (int i = 0; i < prompt->len; i++) {
        if (prompt->v[i] < 0 || (uint32_t)prompt->v[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "MiniMax prompt has an invalid token");
            return 1;
        }
    }
    if (ds4_session_ensure_graph(s, err, errlen)) { return 1; }
    if (!s->minimax_graph_ready) { payload_set_err(err, errlen, "MiniMax graph is not initialized"); return 1; }
    ds4_minimax_graph *g = &s->minimax_graph;
    uint32_t start = !tag || tag == g->checkpoint_tag ? minimax_sync_keep(s, prompt) : 0;
    const uint32_t live = (uint32_t)s->checkpoint.len;
    if (!start) {
        ds4_session_invalidate(s);
        if (g->failed) { payload_set_err(err, errlen, "MiniMax graph reset failed"); return 1; }
        g->checkpoint_tag = 0;
    } else if (start == (uint32_t)prompt->len) {
        return 0;
    } else if (start < live) {
        /* The old branch's verify rows (up to live + a block) displaced the
         * DSpark ring's rows below live - window: its context restarts above
         * them, or at the resume row. */
        const uint32_t lost = live > MINIMAX_DRAFT_WINDOW ? live - MINIMAX_DRAFT_WINDOW : 0u;
        if (g->draft_first < lost) { g->draft_first = lost < start ? lost : start; }
        g->position = start;
        session_rewound(s, start);
    }
    for (uint32_t i = start; i < (uint32_t)prompt->len; i++) {
        if (!minimax_media_pad(prompt->v[i])) { continue; }
        bool covered = false;
        for (uint32_t k = 0; !covered && k < g->span_n; k++) {
            covered = i >= g->span_start[k] && i - g->span_start[k] < g->span_count[k];
        }
        if (!covered) {
            payload_set_err(err, errlen, "MiniMax media pad has no vision rows");
            return 1;
        }
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
    if (tag) { g->checkpoint_tag = tag; }
    return 0;
}

static int minimax_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    return minimax_session_prefill(s, prompt, 0, err, errlen);
}

/* Image and video rows (host F32 [count][6144] per span) over the prompt's
 * pad runs, then the usual prefill. */
int ds4_session_minimax_sync_media(ds4_session *s, const int *tokens, int n_tokens, const unsigned *starts,
        const unsigned *counts, const float *const *rows, unsigned nspans, uint64_t tag, char *err, size_t errlen) {
    if (!ds4_session_is_minimax(s) || !tokens || n_tokens <= 0 || !tag ||
        (nspans && (!starts || !counts || !rows))) {
        payload_set_err(err, errlen, "MiniMax media sync needs tokens, spans and a tag");
        return 1;
    }
    for (unsigned i = 0; i < nspans; i++) {
        if (!counts[i] || !rows[i] || starts[i] > (unsigned)n_tokens || counts[i] > (unsigned)n_tokens - starts[i]) {
            payload_set_err(err, errlen, "MiniMax media span is empty or past the prompt");
            return 1;
        }
    }
    if (ds4_session_ensure_graph(s, err, errlen)) { return 1; }
    ds4_minimax_graph *g = &s->minimax_graph;
    g->span_start = starts;
    g->span_count = counts;
    g->span_rows = rows;
    g->span_n = nspans;
    const ds4_tokens prompt = {.v = (int *)tokens, .len = n_tokens, .cap = n_tokens};
    const int rc = minimax_session_prefill(s, &prompt, tag, err, errlen);
    g->span_n = 0;
    g->span_start = g->span_count = NULL;
    g->span_rows = NULL;
    return rc;
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
