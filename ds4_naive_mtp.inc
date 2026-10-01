/* The target verifies causal rows once. Rejection crops both frontiers;
 * 1030 draft slots and cap+127 target SWA slots retain the accepted window. */
#ifndef DS4_NO_GPU
static bool naive_verify_rows(ds4_naive_graph *g, const ds4_model *m,
                               const ds4_weights *w, unsigned n, int *target) {
    ds4_naive_df_scratch *s = g->draft.ws;
    ds4_gpu_top2_result result[N05_DF_BLOCK];
    if (!s || !n || n > N05_DF_BLOCK ||
        !ds4_gpu_naive_rms(g->ws.b_norm, g->ws.b_cur, m->map, m->size, w->output_norm->abs_offset, n) ||
        !naive_linear(s->verify, m, w->output, N05_EMBED, N05_VOCAB, g->ws.b_norm, n) ||
        !ds4_gpu_naive_round(s->verify, (uint64_t)n * N05_VOCAB)) { return false; }
    for (unsigned i = 0; i < n; i++) {
        ds4_gpu_tensor *row = naive_view(s->verify, i, N05_VOCAB, 1);
        ds4_gpu_tensor *pick = ds4_gpu_tensor_view(s->pick,
            i * sizeof(*result), sizeof(*result));
        const bool ok = row && pick && ds4_gpu_naive_df_top2(pick, row);
        ds4_gpu_tensor_free(row); ds4_gpu_tensor_free(pick);
        if (!ok) { return false; }
    }
    if (!ds4_gpu_tensor_read(s->pick, 0, result, n * sizeof(*result))) { return false; }
    for (unsigned i = 0; i < n; i++) {
        if (result[i].id0 >= N05_VOCAB || !isfinite(result[i].value0) || !isfinite(result[i].value1)) { return false; }
        target[i] = (int)result[i].id0;
    }
    return true;
}

static int naive_trial_fail(ds4_session *s, char *err, size_t errlen) {
    s->naive_graph.draft.trial_n = 0;
    naive_session_fail(s, err, errlen);
    return -1;
}
#endif

int ds4_session_naive_trial(ds4_session *s, int first, int max_tokens,
                             int *tokens, int *target, int cap, char *err, size_t errlen) {
#ifdef DS4_NO_GPU
    (void)s; (void)first; (void)max_tokens; (void)tokens; (void)target; (void)cap;
    payload_set_err(err, errlen, "Naive DSpark requires CUDA"); return -1;
#else
    if (!s || !ds4_session_is_naive(s) || !tokens || !target || cap < 1) {
        payload_set_err(err, errlen, "invalid Naive trial output"); return -1;
    }
    ds4_engine *e = s->engine;
    if (max_tokens <= 0 || !e->dspark_ready || getenv("DS4_MTP_SPEC_DISABLE")) { return 0; }
    ds4_naive_graph *g = &s->naive_graph;
    ds4_naive_df_runtime *d = &g->draft;
    if (!s->naive_graph_ready || !s->checkpoint_valid || g->failed || !d->ws || d->trial_n ||
        g->position > g->context || g->position != (unsigned)s->checkpoint.len || d->position != g->position ||
        d->first > (g->position > N05_DF_WINDOW ? g->position - N05_DF_WINDOW : 0) ||
        first < 0 || (unsigned)first >= N05_VOCAB) {
        payload_set_err(err, errlen, "Naive trial needs a committed target/draft frontier"); return -1;
    }
    if (g->context - g->position < N05_DF_BLOCK || max_tokens < 2 || cap < 2) { return 0; }
    unsigned n = (unsigned)max_tokens;
    if (n > (unsigned)cap) { n = (unsigned)cap; }
    if (n > (unsigned)e->mtp_draft_tokens + 1u) { n = (unsigned)e->mtp_draft_tokens + 1u; }
    if (n > N05_DF_BLOCK) { n = N05_DF_BLOCK; }
    if (n > g->cap) { n = g->cap; }
    if (n < 2) { return 0; }
    /* Noise attention is noncausal: shortening its seven-row block changes
     * proposals. Budget/depth/margin shorten only the target verification. */
    if (!naive_draft_block(d, &e->model, &e->weights, first, e->mtp_margin, N05_DF_BLOCK, d->trial)) {
        return naive_trial_fail(s, err, errlen);
    }
    if (n > d->usable) { n = d->usable; }
    if (n < 2) { return 0; }
    d->trial_pos = g->position;
    if (!naive_forward(g, &e->model, &e->weights, d->trial, n, g->position) ||
        !naive_verify_rows(g, &e->model, &e->weights, n, target)) { return naive_trial_fail(s, err, errlen); }
    d->trial_n = n;
    memcpy(tokens, d->trial, n * sizeof(*tokens));
    ds4_metric_add(&ds4_metrics_get()->spec_drafts, n - 1);
    return (int)n;
#endif
}

int ds4_session_naive_commit(ds4_session *s, int keep, char *err, size_t errlen) {
#ifdef DS4_NO_GPU
    (void)s; (void)keep;
    payload_set_err(err, errlen, "Naive DSpark requires CUDA"); return 1;
#else
    if (!s || !ds4_session_is_naive(s) || !s->checkpoint_valid || keep < 1) {
        payload_set_err(err, errlen, "invalid Naive accepted prefix"); return 1;
    }
    ds4_naive_graph *g = &s->naive_graph;
    ds4_naive_df_runtime *d = &g->draft;
    if (!d->ws || (unsigned)keep > d->trial_n || g->failed ||
        d->trial_pos != (unsigned)s->checkpoint.len ||
        g->position != d->trial_pos + d->trial_n || d->position != g->position) {
        payload_set_err(err, errlen, "Naive accepted prefix has no matching trial"); return 1;
    }
    const uint64_t offset = (uint64_t)(keep - 1) * N05_VOCAB * sizeof(float);
    if (!ds4_gpu_tensor_copy(g->logits, 0, d->ws->verify, offset, N05_VOCAB * sizeof(float)) ||
        !naive_read_logits(s)) { return naive_session_fail(s, err, errlen); }
    const unsigned trial = d->trial_n, end = d->trial_pos + (unsigned)keep;
    g->position = end; d->position = end; d->trial_n = 0;
    for (int i = 0; i < keep; i++) { token_vec_push(&s->checkpoint, d->trial[i]); }
    ds4_metric_add(&ds4_metrics_get()->spec_hits, (uint64_t)keep - 1);
    if (getenv("DS4_MTP_SPEC_LOG")) {
        fprintf(stderr, "ds4: Naive DSpark trial=%u keep=%d pos=%u confidence=%g\n",
                trial, keep, end, d->confidence[1]);
    }
    return 0;
#endif
}
