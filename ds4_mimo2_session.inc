/* Serial session glue. Execution remains guarded until real-model validation
 * and media/MTP integration are complete. No DeepSeek cache layout is reused. */
static bool mimo2_reset(ds4_mimo2_graph *g) {
    if (g->failed && !ds4_gpu_synchronize()) { return false; }
    g->position = 0;
    g->high = 0;
    g->restored = 0;
    g->failed = false;
    return true;
}

static int mimo2_session_fail(ds4_session *s, char *err, size_t errlen) {
    s->mimo2_graph.failed = true;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
    s->generation++;
    payload_set_err(err, errlen, "MiMo forward or logits readback failed");
    return 1;
}

static bool mimo2_read_logits(ds4_session *s) {
    if (!ds4_gpu_tensor_read(s->mimo2_graph.logits, 0, s->logits,
            DS4_N_VOCAB * sizeof(float))) { return false; }
    for (unsigned i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(s->logits[i])) { return false; }
    }
    return true;
}

static void mimo2_draft_reset(ds4_mimo2_graph *g);

/* Re-run this many rows below a rewound prefix: the DFlash tap window and
 * the MTP ring then hold only this prefix's rows (an abandoned branch
 * overwrote some of their slots). */
enum { M2_REDRAFT = 1024 };

/* Whether the SWA ring still holds the window before q: C ring rows written
 * below `high` keep every row at or after high - C. */
static bool mimo2_ring_holds(const ds4_mimo2_graph *g, unsigned live, unsigned q) {
    const long ring = (long)M2_WINDOW + g->cap - 1;
    const long high = g->high > live ? g->high : live;
    if (q < g->restored) { return false; }
    return ring >= (long)g->context || high - q <= ring - M2_WINDOW + 1;
}

/* Rows the live state keeps for `prompt`; 0 rebuilds. Full layers index KV
 * by position, so any common prefix holds there; SWA layers need their
 * ring. A prompt that ends inside the state re-runs its last token. */
static unsigned mimo2_sync_keep(const ds4_session *s, const ds4_mimo2_graph *g,
                                const ds4_tokens *prompt) {
    if (!s->checkpoint_valid || g->failed || g->position != (unsigned)s->checkpoint.len) { return 0; }
    unsigned common = 0;
    const unsigned live = (unsigned)s->checkpoint.len;
    while (common < live && common < (unsigned)prompt->len &&
           s->checkpoint.v[common] == prompt->v[common]) {
        common++;
    }
    if (common == live) { return live; }
    if (common && common == (unsigned)prompt->len) { common--; }
    const unsigned redraft = common > M2_REDRAFT ? common - M2_REDRAFT : 0;
    if (mimo2_ring_holds(g, live, redraft)) { return redraft; }
    return mimo2_ring_holds(g, live, common) ? common : 0;
}

static int mimo2_session_sync(ds4_session *s, const ds4_tokens *prompt,
                               char *err, size_t errlen) {
    enum { IMAGE_PAD = 151655, VIDEO_PAD = 151656, AUDIO_PAD = 151669 };
    if (!prompt->v) { payload_set_err(err, errlen, "MiMo sync needs tokens"); return 1; }
    for (int i = 0; i < prompt->len; i++) {
        const int token = prompt->v[i];
        if (token < 0 || (unsigned)token >= DS4_N_VOCAB || token == IMAGE_PAD ||
            token == VIDEO_PAD || token == AUDIO_PAD) {
            payload_set_err(err, errlen, "MiMo prompt has an invalid or unbound media token");
            return 1;
        }
    }
    if (ds4_session_ensure_graph(s, err, errlen)) { return 1; }
    ds4_mimo2_graph *g = &s->mimo2_graph;
    if (!s->mimo2_graph_ready) { payload_set_err(err, errlen, "MiMo graph is not initialized"); return 1; }
    unsigned start = 0;
    const unsigned keep = mimo2_sync_keep(s, g, prompt);
    if (keep && keep == (unsigned)s->checkpoint.len) {
        start = keep;
        if (start == (unsigned)prompt->len) { return 0; }
    } else if (keep) {
        // Agent clients re-render the last answer: resume below the common prefix.
        g->position = keep;
        session_rewound(s, keep);
        mimo2_draft_reset(g);
        start = keep;
    } else {
        ds4_session_invalidate(s);
        if (g->failed) { payload_set_err(err, errlen, "MiMo graph reset failed"); return 1; }
    }
    while (start < (unsigned)prompt->len) {
        unsigned n = (unsigned)prompt->len - start;
        if (n > g->cap) { n = g->cap; }
        if (!mimo2_forward(g, &s->engine->model, &s->engine->weights, prompt->v + start, n, start)) {
            return mimo2_session_fail(s, err, errlen);
        }
        start += n;
        if (s->progress) { s->progress(s->progress_ud, "prefill_chunk", (int)start, prompt->len); }
    }
    if (!mimo2_read_logits(s)) { return mimo2_session_fail(s, err, errlen); }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    return 0;
}

static int mimo2_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    ds4_mimo2_graph *g = &s->mimo2_graph;
    if (token < 0 || (unsigned)token >= DS4_N_VOCAB || !s->mimo2_graph_ready ||
        !s->checkpoint_valid || g->failed || g->position >= g->context ||
        g->position != (unsigned)s->checkpoint.len) {
        payload_set_err(err, errlen, "MiMo decode needs a valid token and checkpoint");
        return 1;
    }
    if (!mimo2_forward(g, &s->engine->model, &s->engine->weights, &token, 1, g->position) ||
        !mimo2_read_logits(s)) { return mimo2_session_fail(s, err, errlen); }
    token_vec_push(&s->checkpoint, token);
    s->mtp_draft_valid = false;
    return 0;
}
