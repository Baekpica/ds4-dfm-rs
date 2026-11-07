/* Spark-X2.6 serial session: one graph owns every layer's KV. A sliding
 * layer keeps window + chunk rows indexed by position, a full layer the whole
 * context; both are written before any query at or below them reads, so a
 * prompt that extends the checkpoint resumes where it stopped, and a prompt
 * that rewinds resumes at the common prefix while the ring still holds the
 * window (spark2_5_graph_keeps), the serial rule the other families use. */

#ifndef DS4_NO_GPU

static bool spark2_5_read_logits(ds4_session *s) {
    if (!ds4_gpu_tensor_read(s->spark_graph->logits, 0, s->logits,
            (uint64_t)SPARK_VOCAB * sizeof(float))) {
        return false;
    }
    uint32_t bad = 0;
    for (uint32_t i = 0; i < SPARK_VOCAB; i++) {
        if (!isfinite(s->logits[i])) { bad++; }
    }
    if (bad) {
        fprintf(stderr, "ds4: Spark-X2.6 logits hold %u non-finite values of %u\n",
            bad, (uint32_t)SPARK_VOCAB);
        return false;
    }
    return true;
}

static int spark2_5_session_fail(ds4_session *s, char *err, size_t errlen) {
    if (s->spark_graph) { s->spark_graph->failed = true; }
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
    s->generation++;
    payload_set_err(err, errlen, "Spark-X2.6 forward or logits readback failed");
    return 1;
}

static uint32_t spark2_5_sync_keep(ds4_session *s, const ds4_tokens *prompt) {
    const ds4_spark2_5_graph *g = s->spark_graph;
    if (!g || !s->checkpoint_valid || g->failed ||
        g->position != (uint32_t)s->checkpoint.len) {
        return 0;
    }
    const uint32_t keep = (uint32_t)session_keep_prefix(s, prompt);
    if (keep == g->position) { return keep; }
    return spark2_5_graph_keeps(g, keep) ? keep : 0;
}

static int spark2_5_session_sync(ds4_session *s, const ds4_tokens *prompt,
                                 char *err, size_t errlen) {
    for (int i = 0; i < prompt->len; i++) {
        if (prompt->v[i] < 0 || prompt->v[i] >= (int)SPARK_VOCAB) {
            payload_set_err(err, errlen, "Spark-X2.6 prompt has an invalid token");
            return 1;
        }
    }
    if (ds4_session_ensure_graph(s, err, errlen)) { return 1; }
    if (!s->spark_graph_ready || !s->spark_graph) {
        payload_set_err(err, errlen, "Spark-X2.6 graph is not initialized");
        return 1;
    }
    ds4_spark2_5_graph *g = s->spark_graph;
    uint32_t start = spark2_5_sync_keep(s, prompt);
    if (!start) {
        ds4_session_invalidate(s);
        if (g->failed && !spark2_5_graph_reset(g)) {
            payload_set_err(err, errlen, "Spark-X2.6 graph reset failed");
            return 1;
        }
    } else if (start == (uint32_t)prompt->len) {
        return 0;
    } else if (start < (uint32_t)s->checkpoint.len) {
        /* Rows at or beyond `start` are stale but harmless: attention only
         * reads slots at or below the query's own position, every slot the
         * new rows use is overwritten before it is read, and
         * spark2_5_sync_keep resumed only where the ring holds the window. */
        g->position = start;
        session_rewound(s, start);
    }
    const uint32_t end = (uint32_t)prompt->len;
    while (start < end) {
        const uint32_t n = end - start < g->cap ? end - start : g->cap;
        /* Only the final chunk's last row needs the tied-table head. */
        const bool head = start + n == end;
        if (!spark2_5_forward(g, &s->engine->model, &s->engine->weights,
                (const uint32_t *)prompt->v + start, n, start, head)) {
            return spark2_5_session_fail(s, err, errlen);
        }
        start += n;
        if (s->progress) {
            s->progress(s->progress_ud, "prefill_chunk", (int)start,
                        prompt->len);
        }
    }
    if (!spark2_5_read_logits(s)) { return spark2_5_session_fail(s, err, errlen); }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    return 0;
}

static int spark2_5_session_eval(ds4_session *s, int token, char *err,
                                 size_t errlen) {
    ds4_spark2_5_graph *g = s->spark_graph;
    if (token < 0 || token >= (int)SPARK_VOCAB || !s->spark_graph_ready ||
        !g || !s->checkpoint_valid || g->failed || g->position >= g->ctx ||
        g->position != (uint32_t)s->checkpoint.len) {
        payload_set_err(err, errlen,
                        "Spark-X2.6 decode needs a valid token and checkpoint");
        return 1;
    }
    if (!spark2_5_forward(g, &s->engine->model, &s->engine->weights,
            (const uint32_t *)&token, 1u, g->position, true) ||
        !spark2_5_read_logits(s)) {
        return spark2_5_session_fail(s, err, errlen);
    }
    token_vec_push(&s->checkpoint, token);
    s->mtp_draft_valid = false;
    return 0;
}

#endif
