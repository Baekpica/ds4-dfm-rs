static int hy4_session_fail(ds4_session *s, char *err, size_t len) {
    s->hy4_graph.failed = true;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
    s->generation++;
    payload_set_err(err, len, "Hy4 forward or logits readback failed");
    return 1;
}

static uint32_t hy4_sync_keep(ds4_session *s, const ds4_tokens *prompt) {
    const ds4_hy4_graph *g = &s->hy4_graph;
    if (!g->ready || g->failed || !s->checkpoint_valid || g->position != (uint32_t)s->checkpoint.len) {
        return 0;
    }
    const uint32_t common = session_keep_prefix(s, prompt);
    if (common == g->position) {
        return common;
    }
    const uint32_t ring = hy4_ring(g->context, g->cap);
    return ring_holds(ring, HY4_WINDOW, g->high, g->restored, common) ? common : 0;
}

static bool hy4_logits(ds4_session *s) {
    if (!ds4_gpu_tensor_read(s->hy4_graph.logits, 0, s->logits, 120832 * sizeof(float))) {
        return false;
    }
    for (unsigned i = 0; i < 120832; i++) {
        if (!isfinite(s->logits[i])) {
            return false;
        }
    }
    return true;
}
static int hy4_sync_input(ds4_session *s, const ds4_tokens *prompt, const ds4_hy4_span *spans,
                          uint32_t count, uint64_t tag, char *err, size_t len) {
    if (!prompt || prompt->len <= 0 || prompt->len > s->ctx_size || !prompt->v) {
        payload_set_err(err, len, "Hy4 prompt length is outside context");
        return 1;
    }
    uint32_t last = 0;
    for (uint32_t i = 0; i < count; i++) {
        const ds4_hy4_span *v = &spans[i];
        if (!v->features || !v->height || !v->width || v->height > 8192 || v->width > 8192 || !v->frames ||
            v->frames > 2048 || (v->token != 120120 && v->token != 120124) ||
            (v->token == 120120 && v->frames != 1) || v->start < last ||
            v->start >= (uint32_t)prompt->len) {
            payload_set_err(err, len, "invalid Hy4 media geometry");
            return 1;
        }
        uint64_t rows = (uint64_t)v->frames * v->height * (v->width + 1);
        if (rows > (uint32_t)prompt->len - v->start) {
            payload_set_err(err, len, "Hy4 media span exceeds prompt");
            return 1;
        }
        last = v->start + (uint32_t)rows;
    }
    uint32_t span = 0;
    for (int i = 0; i < prompt->len; i++) {
        const int token = prompt->v[i];
        bool visual = span < count && (uint32_t)i >= spans[span].start;
        if (token < 0 || token >= 120832 ||
            (visual ? token != spans[span].token : token == 120120 || token == 120124)) {
            payload_set_err(err, len, "Hy4 prompt contains an invalid or unbound media token");
            return 1;
        }
        if (visual && (uint32_t)i + 1 == spans[span].start + spans[span].frames * spans[span].height *
                                                                 (spans[span].width + 1)) {
            span++;
        }
    }
    if (ds4_session_ensure_graph(s, err, len)) {
        return 1;
    }
    ds4_hy4_graph *g = &s->hy4_graph;
    uint32_t pos = tag == g->media_tag ? hy4_sync_keep(s, prompt) : 0, end = (uint32_t)prompt->len;
    g->media_tag = tag;
    g->media_count = count;
    if (count) {
        memcpy(g->media, spans, sizeof(*spans) * count);
    }
    if (!pos) {
        ds4_session_invalidate(s);
        if (g->failed) {
            return hy4_session_fail(s, err, len);
        }
    } else if (pos == end) {
        return 0;
    } else if (pos < g->position) {
        g->position = pos;
        session_rewound(s, pos);
    }
    while (pos < end) {
        const uint32_t rows = g->cap < end - pos ? g->cap : end - pos, prev = pos < 2 ? pos : 2;
        if (!hy4_forward(g, s->engine, prompt->v + pos - prev, prev, prompt->v + pos, rows, pos,
                         pos + rows == end)) {
            return hy4_session_fail(s, err, len);
        }
        pos += rows;
        if (s->progress) {
            s->progress(s->progress_ud, "prefill_chunk", (int)pos, prompt->len);
        }
    }
    if (!hy4_logits(s)) {
        return hy4_session_fail(s, err, len);
    }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    return 0;
}
static int hy4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t len) {
    return hy4_sync_input(s, prompt, NULL, 0, 0, err, len);
}
int ds4_session_hy4_sync_media(ds4_session *s, const int32_t *tokens, int32_t n, const ds4_hy4_span *spans,
                               uint32_t count, uint64_t tag, char *err, size_t len) {
    if (!ds4_session_is_hy4(s) || !spans || !count || count > 4 || !tag) {
        payload_set_err(err, len, "invalid Hy4 media request");
        return 1;
    }
    ds4_tokens prompt = {.v = (int *)tokens, .len = n, .cap = n};
    const int rc = hy4_sync_input(s, &prompt, spans, count, tag, err, len);
    for (uint32_t i = 0; i < s->hy4_graph.media_count; i++) {
        s->hy4_graph.media[i].features = NULL;
    }
    return rc;
}
static int hy4_session_eval(ds4_session *s, int token, char *err, size_t len) {
    ds4_hy4_graph *g = &s->hy4_graph;
    if (token < 0 || token >= 120832 || !s->checkpoint_valid ||
        g->position != (uint32_t)s->checkpoint.len || g->position >= g->context || !g->ready || g->failed) {
        payload_set_err(err, len, "Hy4 decode needs a valid token and checkpoint");
        return 1;
    }
    const unsigned prev = g->position < 2 ? g->position : 2;
    if (!hy4_forward(g, s->engine, s->checkpoint.v + g->position - prev, prev, &token, 1, g->position,
                     true) ||
        !hy4_logits(s)) {
        return hy4_session_fail(s, err, len);
    }
    token_vec_push(&s->checkpoint, token);
    s->mtp_draft_valid = false;
    return 0;
}
