/* Serial session: one graph owns every layer's KV. A sliding layer keeps a
 * ring of window + chunk rows, a full layer the whole context, and both are
 * indexed by position, so a prompt resumes at its common prefix: the rows it
 * keeps stay in place and the rows it rewrites land on their own slots.
 *
 * The ring is the reason the row count is window + chunk and not the window:
 * a prefill chunk writes rows ahead of the rows it reads, and a ring of
 * exactly window rows would let a later row of the same chunk overwrite a
 * slot an earlier row still needs (semantics 3.4). */

static bool gemma4_read_logits(ds4_session *s) {
    if (!ds4_gpu_tensor_read(s->gemma4_graph->logits, 0, s->logits,
            (uint64_t)DS4_N_VOCAB * sizeof(float))) {
        return false;
    }
    uint32_t bad = 0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(s->logits[i])) { bad++; }
    }
    if (bad) {
        fprintf(stderr, "ds4: Gemma-4 logits hold %u non-finite values of %u\n",
            bad, (uint32_t)DS4_N_VOCAB);
        return false;
    }
    return true;
}

static int gemma4_session_fail(ds4_session *s, char *err, size_t errlen) {
    if (s->gemma4_graph) { s->gemma4_graph->failed = true; }
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
    s->generation++;
    payload_set_err(err, errlen, "Gemma-4 forward or logits readback failed");
    return 1;
}

/* Rows the live state keeps for `prompt`; 0 rebuilds. */
static uint32_t gemma4_sync_keep(ds4_session *s, const ds4_tokens *prompt) {
    const ds4_gemma4_graph *g = s->gemma4_graph;
    if (!g || !s->checkpoint_valid || g->failed ||
        g->position != (uint32_t)s->checkpoint.len) {
        return 0;
    }
    const uint32_t keep = (uint32_t)session_keep_prefix(s, prompt);
    if (keep == g->position) { return keep; }
    return gemma4_graph_keeps(g, keep) ? keep : 0;
}

static int gemma4_session_sync(ds4_session *s, const ds4_tokens *prompt,
                               char *err, size_t errlen) {
    for (int i = 0; i < prompt->len; i++) {
        if (prompt->v[i] < 0 || (uint32_t)prompt->v[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "Gemma-4 prompt has an invalid token");
            return 1;
        }
    }
    if (ds4_session_ensure_graph(s, err, errlen)) { return 1; }
    if (!s->gemma4_graph_ready || !s->gemma4_graph) {
        payload_set_err(err, errlen, "Gemma-4 graph is not initialized");
        return 1;
    }
    ds4_gemma4_graph *g = s->gemma4_graph;
    uint32_t start = gemma4_sync_keep(s, prompt);
    if (!start) {
        ds4_session_invalidate(s);
        if (g->failed) {
            payload_set_err(err, errlen, "Gemma-4 graph reset failed");
            return 1;
        }
    } else if (start == (uint32_t)prompt->len) {
        return 0;
    } else if (start < (uint32_t)s->checkpoint.len) {
        /* Rows at or beyond `start` are stale but harmless: attention only
         * reads slots at or below the query's own position, every slot the
         * new rows use is overwritten before it is read, and
         * gemma4_sync_keep resumed only where the ring holds the window. */
        g->position = start;
        session_rewound(s, start);
    }
    const uint32_t end = (uint32_t)prompt->len;
    while (start < end) {
        const uint32_t n = end - start < g->cap ? end - start : g->cap;
        /* Only the final chunk's last row needs the tied-table head. */
        const bool head = start + n == end;
        if (!gemma4_forward(g, &s->engine->model, &s->engine->weights,
                prompt->v + start, n, start, head)) {
            return gemma4_session_fail(s, err, errlen);
        }
        start += n;
        if (s->progress) {
            s->progress(s->progress_ud, "prefill_chunk", (int)start, prompt->len);
        }
    }
    if (!gemma4_read_logits(s)) { return gemma4_session_fail(s, err, errlen); }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    return 0;
}

static int gemma4_session_eval(ds4_session *s, int token, char *err,
                               size_t errlen) {
    ds4_gemma4_graph *g = s->gemma4_graph;
    if (token < 0 || (uint32_t)token >= DS4_N_VOCAB || !s->gemma4_graph_ready ||
        !g || !s->checkpoint_valid || g->failed || g->position >= g->ctx ||
        g->position != (uint32_t)s->checkpoint.len) {
        payload_set_err(err, errlen,
                        "Gemma-4 decode needs a valid token and checkpoint");
        return 1;
    }
    if (!gemma4_forward(g, &s->engine->model, &s->engine->weights, &token, 1u,
            g->position, true) ||
        !gemma4_read_logits(s)) {
        return gemma4_session_fail(s, err, errlen);
    }
    token_vec_push(&s->checkpoint, token);
    s->mtp_draft_valid = false;
    return 0;
}
