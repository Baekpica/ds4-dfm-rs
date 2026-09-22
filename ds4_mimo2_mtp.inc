static void mimo2_draft_reset(ds4_mimo2_graph *g) {
    g->hidden_valid = false;
    g->trial_n = 0;
    g->verify_rows = 0;
    g->span_n = 0;
    g->media_tag = 0;
    g->checkpoint_tag = 0;
    for (unsigned depth = 0; depth < MIMO2_DRAFT_LAYERS; depth++) {
        g->draft_pos[depth] = 0;
        g->draft_saved[depth] = 0;
    }
}

/* Embedded MTP is the three dense blocks at layers 48..50. DFlash is a
 * different artifact and is not loaded here. A trial may write rejected
 * draft slots; commit restores those slots and leaves the trunk cursor on
 * the accepted prefix. Draft RoPE uses each block's own cursor, which is
 * rewound with the slot, so a rejected row is overwritten before it is read. */

static int mimo2_argmax(const float *logits) {
    unsigned best = 0;
    for (unsigned i = 1; i < DS4_N_VOCAB; i++) {
        if (logits[i] > logits[best]) { best = i; }
    }
    if (!isfinite(logits[best])) { return -1; }
    return (int)best;
}

static bool mimo2_journal_slot(ds4_mimo2_graph *g, unsigned depth, unsigned pos, bool save) {
    const unsigned layer = MIMO2_LAYERS + depth;
    const uint64_t row = 8ull * (192 + 128) * sizeof(uint16_t);
    if (depth >= MIMO2_DRAFT_LAYERS || !g->kv[layer] || !g->kv_cap[layer] || !g->journal) {
        return false;
    }
    const uint64_t slot = ((uint64_t)pos % g->kv_cap[layer]) * row;
    const uint64_t saved = (uint64_t)depth * row;
    if (save) {
        return ds4_gpu_tensor_copy(g->journal, saved, g->kv[layer], slot, row);
    }
    return ds4_gpu_tensor_copy(g->kv[layer], slot, g->journal, saved, row);
}

static bool mimo2_draft_head(ds4_mimo2_graph *g, const ds4_model *m, const ds4_weights *w,
                             unsigned il) {
    const ds4_layer_weights *l = &w->layer[il];
    return ds4_gpu_exaone_rms_norm_tensor(g->ws.b_norm, g->ws.b_cur, m->map, m->size,
            l->layer_output_norm->abs_offset, M2_EMBED, 1, DS4_RMS_EPS) &&
        plain_graph_matmul_tensor(g->logits, m, w->output, M2_EMBED, DS4_N_VOCAB, g->ws.b_norm, 1);
}

static int mimo2_draft_step(ds4_mimo2_graph *g, const ds4_model *m, const ds4_weights *w,
                            unsigned depth, int token, float *logits) {
    if (depth >= MIMO2_DRAFT_LAYERS || !g->hidden_valid || token < 0 ||
        (unsigned)token >= DS4_N_VOCAB) {
        return -1;
    }
    const unsigned pos = g->draft_pos[depth];
    const unsigned il = MIMO2_LAYERS + depth;
    const ds4_layer_weights *l = &w->layer[il];
    if (pos >= g->context) { return -1; }
    if (!mimo2_journal_slot(g, depth, pos, true)) { return -1; }
    g->draft_saved[depth] = 1;
    g->host_positions[0] = pos;
    ds4_gpu_tensor *left = ds4_gpu_tensor_view(g->joined, 0, (uint64_t)M2_EMBED * sizeof(float));
    ds4_gpu_tensor *right = ds4_gpu_tensor_view(
        g->joined, (uint64_t)M2_EMBED * sizeof(float), (uint64_t)M2_EMBED * sizeof(float));
    bool ok = left && right &&
        ds4_gpu_tensor_write(g->tokens, 0, &token, sizeof(token)) &&
        ds4_gpu_tensor_write(g->positions, 0, &pos, sizeof(pos)) &&
        ds4_gpu_embed_tokens_quant_tensor(g->ws.b_attn_out, g->tokens, m->map, m->size,
            w->token_embd->abs_offset, w->token_embd->type, DS4_N_VOCAB, 1, M2_EMBED) &&
        ds4_gpu_exaone_rms_norm_tensor(left, g->ws.b_attn_out, m->map, m->size,
            l->nextn_enorm->abs_offset, M2_EMBED, 1, DS4_RMS_EPS) &&
        ds4_gpu_exaone_rms_norm_tensor(right, g->hidden, m->map, m->size,
            l->nextn_hnorm->abs_offset, M2_EMBED, 1, DS4_RMS_EPS) &&
        plain_graph_matmul_tensor(g->ws.b_cur, m, l->nextn_eh_proj,
            2 * M2_EMBED, M2_EMBED, g->joined, 1) &&
        ds4_gpu_step37_rope(g->rope_swa, g->freq_swa, g->positions, M2_ROT, 1) &&
        mimo2_layer(g, m, w, il, 1) &&
        mimo2_draft_head(g, m, w, il) &&
        ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) &&
        ds4_gpu_tensor_copy(g->hidden, 0, g->ws.b_cur, 0, (uint64_t)M2_EMBED * sizeof(float));
    ds4_gpu_tensor_free(left);
    ds4_gpu_tensor_free(right);
    if (!ok) { return -1; }
    const int next = mimo2_argmax(logits);
    if (next < 0) { return -1; }
    g->draft_pos[depth] = pos + 1;
    return next;
}

static bool mimo2_pad_covered(const ds4_mimo2_graph *g, unsigned index) {
    for (unsigned s = 0; s < g->span_n; s++) {
        const mimo2_media_span *span = &g->spans[s];
        if (index >= span->start && index < span->start + span->count) { return span->rows != NULL; }
    }
    return false;
}

int ds4_session_mimo2_encode_vision(ds4_session *s, const float *patches,
                                    unsigned n_patches, unsigned grid_h, unsigned grid_w,
                                    float *out, char *err, size_t errlen) {
#ifndef DS4_NO_GPU
    if (s && s->engine && s->engine->vision_ready && s->engine->mimo2_media.bound) {
        return mimo2_vision_rows(&s->engine->mimo2_media, &s->engine->vision_model,
                                 patches, n_patches, grid_h, grid_w, out);
    }
#else
    (void)s; (void)patches; (void)n_patches; (void)grid_h; (void)grid_w; (void)out;
#endif
    payload_set_err(err, errlen, "MiMo image features require the BF16 projector");
    return 1;
}

int ds4_session_mimo2_encode_audio(ds4_session *s, const float *mel, unsigned frames,
                                   float *out, unsigned cap, unsigned *rows,
                                   char *err, size_t errlen) {
#ifndef DS4_NO_GPU
    if (s && s->engine && s->engine->vision_ready && s->engine->mimo2_media.bound) {
        return mimo2_audio_rows(&s->engine->mimo2_media, &s->engine->vision_model,
                                mel, frames, out, cap, rows);
    }
#else
    (void)s; (void)mel; (void)frames; (void)out; (void)cap; (void)rows;
#endif
    payload_set_err(err, errlen, "MiMo audio features require the BF16 projector");
    return 1;
}

int ds4_session_mimo2_sync_media(ds4_session *s, const int *tokens, int n_tokens,
                                 const unsigned *starts, const unsigned *counts,
                                 const float *const *rows, unsigned nspans, uint64_t tag,
                                 char *err, size_t errlen) {
    enum { IMAGE_PAD = 151655, VIDEO_PAD = 151656, AUDIO_PAD = 151669 };
    if (!ds4_session_is_mimo2(s) || !tokens || n_tokens <= 0 ||
        (nspans && (!starts || !counts || !rows))) {
        payload_set_err(err, errlen, "MiMo media sync needs tokens and spans");
        return 1;
    }
    if (nspans > M2_SPAN_MAX) {
        payload_set_err(err, errlen, "MiMo media exceeds the span budget");
        return 1;
    }
    if (ds4_session_ensure_graph(s, err, errlen)) { return 1; }
    ds4_mimo2_graph *g = &s->mimo2_graph;
    if (!s->mimo2_graph_ready) {
        payload_set_err(err, errlen, "MiMo graph is not initialized");
        return 1;
    }
    if (g->span_cap < nspans) {
        mimo2_media_span *next = realloc(g->spans, (size_t)nspans * sizeof(*next));
        if (!next) {
            payload_set_err(err, errlen, "MiMo media span allocation failed");
            return 1;
        }
        g->spans = next;
        g->span_cap = nspans;
    }
    g->span_n = nspans;
    for (unsigned sidx = 0; sidx < nspans; sidx++) {
        if (!counts[sidx] || !rows[sidx]) {
            payload_set_err(err, errlen, "MiMo media span is empty");
            return 1;
        }
        g->spans[sidx].start = starts[sidx];
        g->spans[sidx].count = counts[sidx];
        g->spans[sidx].rows = rows[sidx];
    }
    for (int i = 0; i < n_tokens; i++) {
        const int token = tokens[i];
        if (token < 0 || (unsigned)token >= DS4_N_VOCAB) {
            g->span_n = 0;
            payload_set_err(err, errlen, "MiMo media token is out of range");
            return 1;
        }
        if ((token == IMAGE_PAD || token == VIDEO_PAD || token == AUDIO_PAD) &&
            !mimo2_pad_covered(g, (unsigned)i)) {
            g->span_n = 0;
            payload_set_err(err, errlen, "MiMo media pad has no feature rows");
            return 1;
        }
    }
    ds4_tokens prompt = {.v = (int *)tokens, .len = n_tokens, .cap = n_tokens};
    unsigned start = 0;
    const bool extend = s->checkpoint_valid && !g->failed &&
        g->position == (unsigned)s->checkpoint.len &&
        n_tokens >= s->checkpoint.len && ds4_tokens_starts_with(&prompt, &s->checkpoint) &&
        tag == g->checkpoint_tag;
    if (extend) {
        start = (unsigned)s->checkpoint.len;
        if (start == (unsigned)n_tokens) {
            g->span_n = 0;
            return 0;
        }
    } else {
        /* Invalidate resets draft bookkeeping, including span_n. The row
         * records above are still the ones just checked. Restoring the count
         * is what lets this prefill overwrite pad embeddings with pixels. */
        const unsigned spans = g->span_n;
        ds4_session_invalidate(s);
        g->span_n = spans;
        if (g->failed) {
            g->span_n = 0;
            payload_set_err(err, errlen, "MiMo graph reset failed");
            return 1;
        }
    }
    while (start < (unsigned)n_tokens) {
        unsigned n = (unsigned)n_tokens - start;
        if (n > g->cap) { n = g->cap; }
        if (!mimo2_forward(g, &s->engine->model, &s->engine->weights, tokens + start, n, start)) {
            g->span_n = 0;
            return mimo2_session_fail(s, err, errlen);
        }
        start += n;
    }
    g->span_n = 0;
    if (!mimo2_read_logits(s)) { return mimo2_session_fail(s, err, errlen); }
    ds4_tokens_copy(&s->checkpoint, &prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    g->checkpoint_tag = tag;
    g->media_tag = tag;
    return 0;
}

int ds4_session_mimo2_trial(ds4_session *s, int first, int max_tokens,
                            int *tokens, int *target, int cap,
                            char *err, size_t errlen) {
    if (!ds4_session_is_mimo2(s) || !tokens || !target || cap <= 0) {
        payload_set_err(err, errlen, "invalid MiMo trial output");
        return -1;
    }
    if (max_tokens <= 0 || !s->engine || s->engine->mtp_draft_tokens <= 1) { return 0; }
    ds4_mimo2_graph *g = &s->mimo2_graph;
    if (!s->mimo2_graph_ready || !s->checkpoint_valid || g->trial_n || g->failed ||
        g->position != (unsigned)s->checkpoint.len ||
        g->position >= g->context || first < 0 || (unsigned)first >= DS4_N_VOCAB) {
        payload_set_err(err, errlen, "MiMo trial needs a committed checkpoint");
        return -1;
    }
    /* Snapshot restore keeps KV and logits, not the MTP hidden row.
     * Zero selects one ordinary step; that forward rebuilds the row.
     * The DFlash ring is also absent after a disk restore. Ordinary
     * steps keep writing the new position until that window is whole. */
    if (!g->hidden_valid) { return 0; }
    if (s->engine->mimo2_dflash && !mimo2_dflash_window_ready(g, g->position)) { return 0; }
    unsigned n = (unsigned)max_tokens;
    if (n > (unsigned)cap) { n = (unsigned)cap; }
    if (n > M2_TRIAL) { n = M2_TRIAL; }
    if (n > (unsigned)s->engine->mtp_draft_tokens + 1) {
        n = (unsigned)s->engine->mtp_draft_tokens + 1;
    }
    if (n > g->context - g->position) { n = g->context - g->position; }
    if (!n) { return 0; }
    g->trial[0] = first;
    if (s->engine->mimo2_dflash) {
        /* One leftover token has no mask block. Fill rejects n < 2. */
        if (n >= 2 && !mimo2_dflash_fill(s, first, n)) {
            return mimo2_session_fail(s, err, errlen) ? -1 : -1;
        }
    } else for (unsigned depth = 0; depth + 1 < n; depth++) {
        const int next = mimo2_draft_step(g, &s->engine->model, &s->engine->weights,
                                           depth, g->trial[depth], s->logits);
        if (next < 0) { return mimo2_session_fail(s, err, errlen) ? -1 : -1; }
        g->trial[depth + 1] = next;
    }
    const unsigned base = g->position;
    if (!mimo2_dflash_save_ring(g, base, n) ||
        !mimo2_forward(g, &s->engine->model, &s->engine->weights, g->trial, n, base)) {
        return mimo2_session_fail(s, err, errlen) ? -1 : -1;
    }
    for (unsigned row = 0; row < n; row++) {
        float *logits = g->verify_logits + (size_t)row * DS4_N_VOCAB;
        if (!mimo2_head(g, &s->engine->model, &s->engine->weights, row) ||
            !ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float))) {
            return mimo2_session_fail(s, err, errlen) ? -1 : -1;
        }
        const int id = mimo2_argmax(logits);
        if (id < 0) { return mimo2_session_fail(s, err, errlen) ? -1 : -1; }
        target[row] = id;
    }
    g->verify_rows = n;
    g->trial_n = n;
    memcpy(tokens, g->trial, n * sizeof(*tokens));
    return (int)n;
}

int ds4_session_mimo2_commit(ds4_session *s, int keep, char *err, size_t errlen) {
    if (!ds4_session_is_mimo2(s) || !s->checkpoint_valid || keep <= 0 ||
        (unsigned)keep > s->mimo2_graph.trial_n) {
        payload_set_err(err, errlen, "invalid MiMo accepted prefix");
        return 1;
    }
    ds4_mimo2_graph *g = &s->mimo2_graph;
    const unsigned base = (unsigned)s->checkpoint.len;
    const unsigned end = base + (unsigned)keep;
    if (g->position != base + g->trial_n) { return mimo2_session_fail(s, err, errlen); }
    if (!mimo2_dflash_restore(g, base, (unsigned)keep)) {
        return mimo2_session_fail(s, err, errlen);
    }
    g->position = end;
    for (unsigned depth = 0; depth < MIMO2_DRAFT_LAYERS; depth++) {
        if (g->draft_saved[depth] && depth >= (unsigned)keep) {
            if (!g->draft_pos[depth] ||
                !mimo2_journal_slot(g, depth, g->draft_pos[depth] - 1, false)) {
                return mimo2_session_fail(s, err, errlen);
            }
            g->draft_pos[depth]--;
        }
        g->draft_saved[depth] = 0;
    }
    if (g->verify_rows != g->trial_n) { return mimo2_session_fail(s, err, errlen); }
    const float *logits = g->verify_logits + (size_t)(keep - 1) * DS4_N_VOCAB;
    if (!ds4_gpu_tensor_write(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) ||
        !ds4_gpu_tensor_copy(g->hidden, 0, g->ws.b_cur,
            (uint64_t)(keep - 1) * M2_EMBED * sizeof(float),
            (uint64_t)M2_EMBED * sizeof(float))) {
        return mimo2_session_fail(s, err, errlen);
    }
    memcpy(s->logits, logits, (size_t)DS4_N_VOCAB * sizeof(float));
    g->hidden_valid = true;
    for (int i = 0; i < keep; i++) { token_vec_push(&s->checkpoint, g->trial[i]); }
    g->trial_n = 0;
    g->verify_rows = 0;
    s->mtp_draft_valid = false;
    return 0;
}
