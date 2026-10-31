static void mimo2_draft_reset(ds4_mimo2_graph *g) {
    g->hidden_valid = false;
    g->trial_n = 0;
    g->verify_rows = 0;
    g->span_n = 0;
    g->media_tag = 0;
    g->checkpoint_tag = 0;
    mimo2_mtp_rebase(g, g->position);
}

/* Embedded MTP is the three dense blocks at layers 48..50, run as SGLang's
 * multi-layer EAGLE runs MiMo-V2: block d's row i reads the target's row i
 * and token i+d+1 at position i, and every depth keeps the target rows (no
 * hidden chain). A draft step first writes the block's rows
 * that are still missing, from the ring. Rows past the committed tokens use
 * this trial's drafts; the next step rewrites them, so rejects need no undo.
 * DFlash is a different artifact and is not loaded here. */

static int mimo2_argmax(const float *logits) {
    unsigned best = 0;
    for (unsigned i = 1; i < DS4_N_VOCAB; i++) {
        if (logits[i] > logits[best]) { best = i; }
    }
    if (!isfinite(logits[best])) { return -1; }
    return (int)best;
}

/* MTP block rows lo..lo+n-1 into b_cur: eh_proj of [enorm(embedding of each
 * row's token) | hnorm(target row)]. The target row is the one before the
 * final norm: SGLang's normed hidden accepted 0.5-2% fewer drafts on four
 * workloads. The joined rows borrow the qkv scratch; the block's projection
 * overwrites it only after eh_proj has read them. */
static bool mimo2_mtp_input(ds4_mimo2_graph *g, const ds4_model *m, const ds4_weights *w,
                            unsigned il, const int *tokens, unsigned lo, unsigned n) {
    ds4_exaone_batch_ws *b = &g->ws;
    const ds4_layer_weights *l = &w->layer[il];
    return mimo2_ring_copy(g, b->b_ffn_out, lo, n, M2_RING_LOAD) &&
        mimo2_norm(b->b_dense_gate, b->b_ffn_out, m, l->nextn_hnorm, n) &&
        mimo2_embed(g, m, w, tokens, n, b->b_attn_out) &&
        mimo2_norm(b->b_dense_up, b->b_attn_out, m, l->nextn_enorm, n) &&
        ds4_gpu_concat_rows_tensor(g->qkv, b->b_dense_up, b->b_dense_gate, n, M2_EMBED) &&
        plain_graph_matmul_tensor(b->b_cur, m, l->nextn_eh_proj, 2 * M2_EMBED, M2_EMBED, g->qkv, n);
}

/* One draft from block `depth`: g->trial[0] is the target's token at the
 * frontier p, g->trial[1..depth] the earlier drafts. Rows lo..p-1 run through
 * the block, in segments of the prefill width; the last row predicts. */
static int mimo2_draft_step(ds4_mimo2_graph *g, const ds4_model *m, const ds4_weights *w,
                            unsigned depth, float *logits) {
    const unsigned p = g->position, il = MIMO2_LAYERS + depth;
    if (depth >= MIMO2_DRAFT_LAYERS || !g->mtp_hidden || !g->hidden_valid ||
        g->mtp_fill[depth] >= p) {
        return -1;
    }
    unsigned lo = g->mtp_fill[depth];
    if (p - lo > M2_WINDOW) {
        /* Older rows are outside the query's window. Restart the block's
         * positions here so its attention never reads the skipped slots. */
        lo = p - M2_WINDOW;
        g->mtp_base[depth] = lo;
    }
    int tokens[M2_WINDOW];
    unsigned n = 0;
    for (unsigned at = lo; at < p; at += n) {
        n = p - at < g->cap ? p - at : g->cap;
        for (unsigned j = 0; j < n; j++) {
            const unsigned t = at + j + depth + 1;
            tokens[j] = t < p ? g->mtp_tok[t % M2_MTP_RING] : g->trial[t - p];
            g->host_positions[j] = at + j - g->mtp_base[depth];
        }
        if (!ds4_gpu_tensor_write(g->tokens, 0, tokens, n * sizeof(int)) ||
            !ds4_gpu_tensor_write(g->positions, 0, g->host_positions, n * sizeof(unsigned)) ||
            !ds4_gpu_step37_rope(g->rope_swa, g->freq_swa, g->positions, M2_ROT, n) ||
            !mimo2_mtp_input(g, m, w, il, tokens, at, n) ||
            !mimo2_layer(g, m, w, il, n)) {
            return -1;
        }
    }
    const uint64_t row = (uint64_t)M2_EMBED * sizeof(float);
    ds4_gpu_tensor *last = ds4_gpu_tensor_view(g->ws.b_cur, (n - 1) * row, row);
    const bool ok = last &&
        mimo2_norm(g->ws.b_norm, last, m, w->layer[il].layer_output_norm, 1) &&
        plain_graph_matmul_tensor(g->logits, m, w->output, M2_EMBED, DS4_N_VOCAB, g->ws.b_norm, 1) &&
        ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float));
    ds4_gpu_tensor_free(last);
    if (!ok) { return -1; }
    // Rows from p - depth on read this trial's drafts (every row when p <= depth).
    const unsigned drafted = p > depth ? p - depth : 0;
    g->mtp_fill[depth] = drafted > lo ? drafted : lo;
    return mimo2_argmax(logits);
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
        // The VQ release keeps its audio encoder in a separate part.
        const ds4_model *audio = s->engine->mimo2_media.audio;
        return mimo2_audio_rows(&s->engine->mimo2_media, audio ? audio : &s->engine->vision_model,
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
    // The VQ release binds its MTP blocks only when their part is mapped.
    if (!s->engine->mimo2_dflash && !s->engine->weights.layer[MIMO2_LAYERS].attn_qkv) { return 0; }
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
                                           depth, s->logits);
        if (next < 0) { return mimo2_session_fail(s, err, errlen) ? -1 : -1; }
        g->trial[depth + 1] = next;
    }
    const unsigned base = g->position;
    if (!mimo2_dflash_save_ring(g, base, n) ||
        !mimo2_forward(g, &s->engine->model, &s->engine->weights, g->trial, n, base)) {
        return mimo2_session_fail(s, err, errlen) ? -1 : -1;
    }
    if (!mimo2_head_rows(g, &s->engine->model, &s->engine->weights, n, g->verify_logits)) {
        return mimo2_session_fail(s, err, errlen) ? -1 : -1;
    }
    if (spec_sampling(&s->spec)) {
        spec_sample_rows(&s->spec, g->verify_logits, DS4_N_VOCAB, DS4_N_VOCAB, g->trial, n, target);
    } else for (unsigned row = 0; row < n; row++) {
        const int id = mimo2_argmax(g->verify_logits + (size_t)row * DS4_N_VOCAB);
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
    if (g->verify_rows != g->trial_n) { return mimo2_session_fail(s, err, errlen); }
    // The verify forward already left the accepted rows in the MTP ring.
    const float *logits = g->verify_logits + (size_t)(keep - 1) * DS4_N_VOCAB;
    if (!ds4_gpu_tensor_write(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float))) {
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
