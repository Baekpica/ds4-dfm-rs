/* Serial session frontiers include every main KV and indexer row together. */
static bool naive_reset(ds4_naive_graph *g) {
    if (g->failed && !ds4_gpu_synchronize()) { return false; }
    g->position = 0; g->failed = false;
    g->high = g->restored = 0;
    g->draft.position = 0; g->draft.first = 0; g->draft.trial_n = 0;
    return true;
}

static int naive_session_fail(ds4_session *s, char *err, size_t errlen) {
    s->naive_graph.failed = true;
    s->checkpoint_valid = false; s->checkpoint.len = 0;
    s->mtp_draft_valid = false; s->generation++;
    payload_set_err(err, errlen, "Naive forward or logits readback failed");
    return 1;
}

static bool naive_read_logits(ds4_session *s) {
    if (!ds4_gpu_tensor_read(s->naive_graph.logits, 0, s->logits, N05_VOCAB * sizeof(float))) { return false; }
    for (unsigned i = 0; i < N05_VOCAB; i++) { if (!isfinite(s->logits[i])) { return false; } }
    return true;
}

/* Rows the live state keeps for `prompt`; 0 rebuilds. The SWA rings must
 * still hold the window under the resume row. A loaded DSpark drafter
 * resumes one draft window lower: the old branch overwrote part of its ring
 * (six rows of slack), and a trial needs the whole window, which the
 * re-prefill then rebuilds. */
static unsigned naive_sync_keep(ds4_session *s, const ds4_tokens *prompt) {
    const ds4_naive_graph *g = &s->naive_graph;
    if (!s->checkpoint_valid || g->failed || g->draft.trial_n ||
        g->position != (unsigned)s->checkpoint.len) { return 0; }
    unsigned keep = session_keep_prefix(s, prompt);
    if (keep == g->position) { return keep; }
    if (g->draft.ws) { keep = keep > N05_DF_WINDOW ? keep - N05_DF_WINDOW : 0; }
    unsigned ring = g->context;
    for (unsigned il = 0; il < N05_LAYERS; il++) {
        if (g->kv_cap[il] < ring) { ring = g->kv_cap[il]; }
    }
    return ring_holds(ring, N05_WINDOW, g->high, g->restored, keep) ? keep : 0;
}

static int naive_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    if (!prompt->v) { payload_set_err(err, errlen, "Naive sync needs tokens"); return 1; }
    for (int i = 0; i < prompt->len; i++) {
        if (prompt->v[i] < 0 || (unsigned)prompt->v[i] >= N05_VOCAB) {
            payload_set_err(err, errlen, "Naive prompt has an invalid token"); return 1;
        }
    }
    if (ds4_session_ensure_graph(s, err, errlen)) { return 1; }
    ds4_naive_graph *g = &s->naive_graph;
    if (g->draft.trial_n) { payload_set_err(err, errlen, "Naive trial needs commit or invalidation"); return 1; }
    if (!s->naive_graph_ready) { payload_set_err(err, errlen, "Naive graph is not initialized"); return 1; }
    unsigned start = naive_sync_keep(s, prompt);
    if (!start) {
        ds4_session_invalidate(s);
        if (g->failed) { payload_set_err(err, errlen, "Naive graph reset failed"); return 1; }
    } else if (start == (unsigned)prompt->len) {
        return 0;
    } else if (start < (unsigned)s->checkpoint.len) {
        g->position = start;
        g->draft.position = g->draft.first = start;
        session_rewound(s, start);
    }
    while (start < (unsigned)prompt->len) {
        const unsigned remaining = (unsigned)prompt->len - start;
        const unsigned n = remaining < g->cap ? remaining : g->cap;
        if (!naive_forward(g, &s->engine->model, &s->engine->weights, prompt->v + start, n, start)) {
            return naive_session_fail(s, err, errlen);
        }
        start += n;
        if (s->progress) { s->progress(s->progress_ud, "prefill_chunk", (int)start, prompt->len); }
    }
    if (!naive_read_logits(s)) { return naive_session_fail(s, err, errlen); }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true; s->mtp_draft_valid = false;
    return 0;
}

static int naive_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    ds4_naive_graph *g = &s->naive_graph;
    if (token < 0 || (unsigned)token >= N05_VOCAB || !s->naive_graph_ready ||
        !s->checkpoint_valid || g->failed || g->position >= g->context ||
        g->position != (unsigned)s->checkpoint.len) {
        payload_set_err(err, errlen, "Naive decode needs a valid token and checkpoint"); return 1;
    }
    if (!naive_forward(g, &s->engine->model, &s->engine->weights, &token, 1, g->position) ||
        !naive_read_logits(s)) { return naive_session_fail(s, err, errlen); }
    token_vec_push(&s->checkpoint, token); s->mtp_draft_valid = false;
    return 0;
}
