/* Serial session: one graph owns every layer's K/V (full layers by
 * position, sliding layers in a ring) and the convolution histories. The
 * histories describe only the live frontier: a sync copies them, with the
 * sliding rings' 1,023 rows below, where its final chunk starts (a resent
 * turn resumes there) and at its end (a re-rendered continuation resumes
 * there, however long the answer that displaced the rings). A prompt resumes
 * where it extends the checkpoint or at a copy at or below the common
 * prefix, and rebuilds otherwise. */
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

// A media sync's placeholder past the kept rows has no image rows.
enum { INKLING2_NEEDS_ROWS = 2 };

/* `keep` cut at the first image the live state and the request (g->span_*)
 * do not share: the same placeholder run over the same pixels. Placeholder
 * tokens do not name pixels, and rows below a run never depend on it. */
static uint32_t inkling2_image_cut(const ds4_inkling2_graph *g, uint32_t keep) {
    for (uint32_t i = 0;; i++) {
        const uint32_t held = i < g->live_n ? g->live[i].start : UINT32_MAX;
        const uint32_t asked = i < g->span_n ? g->span_start[i] : UINT32_MAX;
        const uint32_t first = held < asked ? held : asked;
        if (first >= keep) { return keep; }
        if (held != asked || g->live[i].count != g->span_count[i] || g->live[i].tag != g->span_tag[i]) {
            return first;
        }
    }
}

/* Rows the live state keeps for `prompt`; 0 rebuilds. */
static uint32_t inkling2_sync_keep(ds4_session *s, const ds4_tokens *prompt) {
    const ds4_inkling2_graph *g = &s->inkling2_graph;
    if (!s->checkpoint_valid || g->failed || g->position != (uint32_t)s->checkpoint.len) { return 0; }
    const uint32_t keep = inkling2_image_cut(g, session_keep_prefix(s, prompt));
    if (keep == g->position) { return keep; }
    const int slot = serial_ckpt_find(s, keep);
    return slot < 0 ? 0u : s->rewind.pos[slot];
}

enum { INKLING2_RESUME_PARTS = INKLING2_LAYERS + 2 * (INKLING2_LAYERS - INKLING2_FULL_LAYERS) };

/* What a resume at `at` restores: the convolution histories, then each
 * sliding ring's 1,023 slots below `at` as two spans in ring order (slots
 * below position 0 hold nothing a read uses). parts[INKLING2_LAYERS..) are
 * views the caller frees. */
static bool inkling2_resume_parts(ds4_inkling2_graph *g, uint32_t at, serial_part *parts) {
    for (uint32_t il = 0; il < INKLING2_LAYERS; il++) {
        parts[il] = (serial_part){g->hist[il], (uint64_t)INKLING2_HIST * sizeof(float)};
    }
    const uint32_t rows = DS4_INK2_WINDOW - 1u, ring = g->ring_cap;
    const uint32_t s0 = (at % ring + ring - rows) % ring, head = s0 + rows <= ring ? 1u : ring - s0;
    bool ok = true;
    uint32_t n = INKLING2_LAYERS;
    for (uint32_t il = 0; il < INKLING2_LAYERS; il++) {
        if (inkling2_global(il)) { continue; }
        parts[n] = (serial_part){ds4_gpu_tensor_view(g->kv[il], (uint64_t)s0 * INKLING2_KV_ROW,
            (uint64_t)head * INKLING2_KV_ROW), (uint64_t)head * INKLING2_KV_ROW};
        parts[n + 1] = (serial_part){ds4_gpu_tensor_view(g->kv[il], (uint64_t)((s0 + head) % ring) * INKLING2_KV_ROW,
            (uint64_t)(rows - head) * INKLING2_KV_ROW), (uint64_t)(rows - head) * INKLING2_KV_ROW};
        ok = ok && parts[n].tensor && parts[n + 1].tensor;
        n += 2;
    }
    return ok;
}

static void inkling2_resume_free(serial_part *parts) {
    for (uint32_t i = INKLING2_LAYERS; i < INKLING2_RESUME_PARTS; i++) { ds4_gpu_tensor_free(parts[i].tensor); }
}

// A serial copy of the state after `at` rows.
static void inkling2_resume_capture(ds4_session *s, uint32_t at) {
    serial_part parts[INKLING2_RESUME_PARTS] = {{0}};
    if (inkling2_resume_parts(&s->inkling2_graph, at, parts)) {
        (void)serial_ckpt_capture(s, parts, INKLING2_RESUME_PARTS, at);
    }
    inkling2_resume_free(parts);
}

static bool inkling2_resume_restore(ds4_session *s, int slot, uint32_t at) {
    serial_part parts[INKLING2_RESUME_PARTS] = {{0}};
    const bool ok = inkling2_resume_parts(&s->inkling2_graph, at, parts) &&
        serial_ckpt_restore(s, slot, parts, INKLING2_RESUME_PARTS);
    inkling2_resume_free(parts);
    return ok;
}

/* The request's images (g->span_*, none for text) become the live state's. */
static bool inkling2_images_hold(ds4_inkling2_graph *g) {
    if (g->span_n > g->live_cap) {
        inkling2_image *grown = realloc(g->live, (size_t)g->span_n * sizeof(*grown));
        if (!grown) { return false; }
        g->live = grown;
        g->live_cap = g->span_n;
    }
    for (uint32_t i = 0; i < g->span_n; i++) {
        g->live[i] = (inkling2_image){g->span_start[i], g->span_count[i], g->span_tag[i]};
    }
    g->live_n = g->span_n;
    return true;
}

/* A media sync names its images (g->span_*): the live state keeps rows below
 * the first image the prompt does not share, and every placeholder the sync
 * prefills takes its image's rows; without them it returns
 * INKLING2_NEEDS_ROWS (an embedding row is not an image). */
static int inkling2_session_prefill(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
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
    const int slot = start && start < (uint32_t)s->checkpoint.len ? serial_ckpt_find(s, start) : -1;
    if (start && start >= (uint32_t)s->checkpoint.len) {
        if (start == (uint32_t)prompt->len) { return 0; }
    } else if (slot >= 0 && inkling2_resume_restore(s, slot, start) && inkling2_rewind(g, start)) {
        session_rewound(s, start);
    } else {
        start = 0;
        ds4_session_invalidate(s);
        if (g->failed) { payload_set_err(err, errlen, "Inkling-2 graph reset failed"); return 1; }
        g->live_n = 0u;
    }
    for (uint32_t i = start; i < (uint32_t)prompt->len; i++) {
        if (prompt->v[i] != INKLING2_IMAGE_PAD) { continue; }
        bool covered = false;
        for (uint32_t k = 0; !covered && k < g->span_n; k++) {
            covered = g->span_rows[k] && i >= g->span_start[k] && i - g->span_start[k] < g->span_count[k];
        }
        if (!covered) {
            payload_set_err(err, errlen, "Inkling-2 image placeholder has no vision rows");
            return INKLING2_NEEDS_ROWS;
        }
    }
    const uint32_t resume = start, end = (uint32_t)prompt->len;
    while (start < end) {
        const uint32_t n = end - start < g->row_cap ? end - start : g->row_cap;
        // Where the final chunk starts: a resent turn resumes here.
        if (start + n == end && start > resume) { inkling2_resume_capture(s, start); }
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
    // The prompt's end: a re-rendered continuation resumes here.
    inkling2_resume_capture(s, end);
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    if (!inkling2_images_hold(g)) { return inkling2_session_fail(s, err, errlen); }
    return 0;
}

static int inkling2_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    return inkling2_session_prefill(s, prompt, err, errlen);
}

// Image spans in prompt order, nonempty, inside the prompt, nonzero tags.
static bool inkling2_spans_valid(int n_tokens, const unsigned *starts, const unsigned *counts,
        const uint64_t *tags, unsigned nspans) {
    if (n_tokens <= 0 || (nspans && (!starts || !counts || !tags))) { return false; }
    uint64_t end = 0;
    for (unsigned i = 0; i < nspans; i++) {
        if (!counts[i] || !tags[i] || starts[i] < end || (uint64_t)starts[i] + counts[i] > (uint64_t)n_tokens) {
            return false;
        }
        end = (uint64_t)starts[i] + counts[i];
    }
    return true;
}

static void inkling2_spans_set(ds4_inkling2_graph *g, const unsigned *starts, const unsigned *counts,
        const uint64_t *tags, const float *const *rows, unsigned nspans) {
    g->span_start = starts;
    g->span_count = counts;
    g->span_tag = tags;
    g->span_rows = rows;
    g->span_n = nspans;
}

int ds4_session_inkling2_media_keep(ds4_session *s, const int *tokens, int n_tokens, const unsigned *starts,
        const unsigned *counts, const uint64_t *tags, unsigned nspans) {
    if (!ds4_session_is_inkling2(s) || !tokens || !inkling2_spans_valid(n_tokens, starts, counts, tags, nspans)) {
        return -1;
    }
    if (!s->inkling2_graph_ready) { return 0; }
    ds4_inkling2_graph *g = &s->inkling2_graph;
    inkling2_spans_set(g, starts, counts, tags, NULL, nspans);
    const ds4_tokens prompt = {.v = (int *)tokens, .len = n_tokens, .cap = n_tokens};
    const uint32_t keep = inkling2_sync_keep(s, &prompt);
    inkling2_spans_set(g, NULL, NULL, NULL, NULL, 0);
    return (int)keep;
}

int ds4_session_inkling2_sync_media(ds4_session *s, const int *tokens, int n_tokens, const unsigned *starts,
        const unsigned *counts, const uint64_t *tags, const float *const *rows, unsigned nspans, char *err,
        size_t errlen) {
    if (!ds4_session_is_inkling2(s) || !tokens || !inkling2_spans_valid(n_tokens, starts, counts, tags, nspans) ||
        (nspans && !rows)) {
        payload_set_err(err, errlen, "Inkling-2 media spans are empty, unordered or past the prompt");
        return 1;
    }
    if (ds4_session_ensure_graph(s, err, errlen)) { return 1; }
    ds4_inkling2_graph *g = &s->inkling2_graph;
    inkling2_spans_set(g, starts, counts, tags, rows, nspans);
    const ds4_tokens prompt = {.v = (int *)tokens, .len = n_tokens, .cap = n_tokens};
    const int rc = inkling2_session_prefill(s, &prompt, err, errlen);
    inkling2_spans_set(g, NULL, NULL, NULL, NULL, 0);
    return rc;
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

/* DSpark trial: the drafter proposes tokens[1..n) after `first`; the target
 * verifies the anchor and the proposals in one forward, target[i] its draw
 * after row i (argmax, or the request's sampler). Returns n, 0 to decode
 * plainly or -1. Convolution histories are saved before the verify forward
 * and its rows' inputs kept, so commit can roll back a rejected tail. */
int ds4_session_inkling2_trial(ds4_session *s, int first, int max_tokens, int *tokens, int *target, int cap,
        char *err, size_t errlen) {
    if (!s || !ds4_session_is_inkling2(s) || !tokens || !target || cap < 1) {
        payload_set_err(err, errlen, "invalid Inkling-2 trial output");
        return -1;
    }
#ifdef __APPLE__
    (void)first; (void)max_tokens;
    return 0;
#else
    ds4_inkling2_graph *g = &s->inkling2_graph;
    inkling2_draft_rt *d = g->draft;
    if (max_tokens <= 0 || !d || getenv("DS4_MTP_SPEC_DISABLE")) { return 0; }
    if (!s->inkling2_graph_ready || !s->checkpoint_valid || g->failed || d->trial_n ||
        g->position != (uint32_t)s->checkpoint.len || first < 0 || (uint32_t)first >= DS4_INK2_TOKENS) {
        payload_set_err(err, errlen, "Inkling-2 trial needs a committed checkpoint");
        return -1;
    }
    uint32_t n = (uint32_t)max_tokens;
    if (n > (uint32_t)cap) { n = (uint32_t)cap; }
    if (n > (uint32_t)s->engine->mtp_draft_tokens + 1u) { n = (uint32_t)s->engine->mtp_draft_tokens + 1u; }
    if (n > DS4_INK2_DRAFT_BLOCK) { n = DS4_INK2_DRAFT_BLOCK; }
    // The block's rows must fit, and the context rings hold the prompt.
    if (n < 2u || !g->position || g->ctx_cap - g->position < DS4_INK2_DRAFT_BLOCK || n > g->row_cap) { return 0; }
    const ds4_model *m = &s->engine->model;
    const ds4_weights *w = &s->engine->weights;
    const uint32_t start = g->position;
    tokens[0] = first;
    const double t0 = now_sec();
    if (!inkling2_draft_propose(g, m, w, n, tokens)) {
        inkling2_session_fail(s, err, errlen);
        return -1;
    }
    const double t1 = now_sec();
    d->draft_ms = (t1 - t0) * 1000.0;
    bool ok = true;
    for (uint32_t il = 0; ok && il < INKLING2_LAYERS; il++) {
        ok = ds4_gpu_ink2_draft_copy(d->hist_save, (uint64_t)il * INKLING2_HIST, g->hist[il], INKLING2_HIST);
    }
    d->capture = true;
    ok = ok && inkling2_forward(g, m, w, tokens, n, start, INKLING2_HEAD_ALL);
    d->capture = false;
    if (!ok || !ds4_gpu_argmax_rows_tensor(d->vids, d->vlogits, DS4_INK2_VOCAB, n) ||
        !ds4_gpu_tensor_read(d->vids, 0, target, (uint64_t)n * sizeof(int))) {
        inkling2_session_fail(s, err, errlen);
        return -1;
    }
    if (spec_sampling(&s->spec)) {
        const float *rows = spec_host_rows(d->vlogits, DS4_N_VOCAB, n);
        if (!rows) {
            inkling2_session_fail(s, err, errlen);
            return -1;
        }
        spec_sample_rows(&s->spec, rows, DS4_N_VOCAB, DS4_N_VOCAB, tokens, n, target);
    }
    for (uint32_t i = 0; i < n; i++) {
        if (target[i] < 0 || (uint32_t)target[i] >= DS4_INK2_TOKENS) {
            inkling2_session_fail(s, err, errlen);
            return -1;
        }
    }
    d->verify_ms = (now_sec() - t1) * 1000.0;
    if (getenv("DS4_MTP_SPEC_LOG")) {
        // Per slot: does proposal i equal the target's draw after row i - 1?
        char hits[DS4_INK2_DRAFT_BLOCK + 1] = {0};
        for (uint32_t i = 1; i < n; i++) { hits[i - 1] = tokens[i] == target[i - 1] ? '1' : '0'; }
        fprintf(stderr, "ds4: Inkling-2 DSpark slots=%s\n", hits);
    }
    d->trial_n = n;
    d->trial_pos = start;
    memcpy(d->trial, tokens, n * sizeof(*tokens));
    ds4_metric_add(&ds4_metrics_get()->spec_drafts, n - 1u);
    return (int)n;
#endif
}

/* Keep the trial's first `keep` rows: the logits after the last kept row,
 * the histories rolled back to it, and its Markov row as the next anchor's
 * context. */
int ds4_session_inkling2_commit(ds4_session *s, int keep, char *err, size_t errlen) {
#ifdef __APPLE__
    (void)s; (void)keep;
    payload_set_err(err, errlen, "Inkling-2 accepted prefix has no matching trial");
    return 1;
#else
    ds4_inkling2_graph *g = s && ds4_session_is_inkling2(s) ? &s->inkling2_graph : NULL;
    inkling2_draft_rt *d = g ? g->draft : NULL;
    if (!d || !s->checkpoint_valid || keep < 1 || (uint32_t)keep > d->trial_n || g->failed ||
        d->trial_pos != (uint32_t)s->checkpoint.len || g->position != d->trial_pos + d->trial_n) {
        payload_set_err(err, errlen, "Inkling-2 accepted prefix has no matching trial");
        return 1;
    }
    const uint32_t trial = d->trial_n;
    d->trial_n = 0;
    if (!ds4_gpu_tensor_read(d->vlogits, (uint64_t)(keep - 1) * DS4_N_VOCAB * sizeof(float), s->logits,
            (uint64_t)DS4_N_VOCAB * sizeof(float))) { return inkling2_session_fail(s, err, errlen); }
    for (uint32_t i = 0; i < DS4_INK2_TOKENS; i++) {
        if (!isfinite(s->logits[i])) { return inkling2_session_fail(s, err, errlen); }
    }
    if (((uint32_t)keep < trial && !inkling2_draft_rollback(g, (uint32_t)keep)) ||
        !inkling2_draft_anchor(d, (uint32_t)keep - 1u)) { return inkling2_session_fail(s, err, errlen); }
    g->position = d->trial_pos + (uint32_t)keep;
    for (int i = 0; i < keep; i++) { token_vec_push(&s->checkpoint, d->trial[i]); }
    s->mtp_draft_valid = false;
    ds4_metric_add(&ds4_metrics_get()->spec_hits, (uint64_t)keep - 1u);
    if (getenv("DS4_MTP_SPEC_LOG")) {
        fprintf(stderr, "ds4: Inkling-2 DSpark trial=%u keep=%d pos=%u draft=%.1fms verify=%.1fms\n", trial, keep,
                g->position, d->draft_ms, d->verify_ms);
    }
    return 0;
#endif
}
