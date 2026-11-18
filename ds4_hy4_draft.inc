// DFly2 uses the five target KV caches and the last committed target selection.
// New draft slots are overwritten by the target verification before publication.
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
static bool hy4_propose(ds4_hy4_graph *g, const ds4_engine *e, int first, uint32_t n) {
    const ds4_model *m = &e->mtp_model;
    const ds4_hy4_draft *d = &e->hy4_draft;
    const uint32_t pos = g->position;
    const uint8_t *emb = e->model.map + e->weights.token_embd->abs_offset;
    for (uint32_t i = 0; i < n; i++) {
        if (!hy4_input_row(g, i ? 120009 : first, pos + i, emb, g->host_embed + (size_t)i * 3072,
                           g->host_positions + 3 * i)) {
            return false;
        }
    }
    const uint64_t rb = 3072 * sizeof(float);
    ds4_gpu_tensor *left = ds4_gpu_tensor_view(g->streams, 0, rb),
                   *right = ds4_gpu_tensor_view(g->streams, rb, rb);
    ds4_gpu_tensor *seed = ds4_gpu_tensor_view(g->seed, 0, rb);
    bool ok = left && right && seed &&
              ds4_gpu_tensor_write(g->positions, 0, g->host_positions, n * 3 * 4) &&
              ds4_gpu_tensor_write(g->embed, 0, g->host_embed, (uint64_t)n * 6144) &&
              ds4_gpu_qwen4exp_bf16_to_f32_tensor(g->input, g->embed, n * 3072) &&
              hy4_norm(left, m, d->enorm, g->input, 1) && hy4_norm(right, m, d->hnorm, seed, 1) &&
              hy4_mm(g->branch, m, d->fusion, g->streams, 1) &&
              ds4_gpu_hy4_add(g->input, g->input, g->branch, hy4_weight(m, d->gate), 3072);
    for (uint32_t i = 0; ok && i < HY4_TAPS; i++) {
        const ds4_layer_weights *l = &d->layer[i];
        ds4_gpu_tensor *index = ds4_gpu_tensor_view(g->seed, rb + i * 511 * 4, 511 * 4);
        ok = index && hy4_norm(g->norm, m, l->attn_norm, g->input, n) &&
             hy4_attention(g, m, l, 25 + 4 * i, n, pos, index) &&
             ds4_gpu_hy4_add(g->input, g->input, g->branch, NULL, n * 3072) &&
             hy4_norm(g->norm, m, l->ffn_norm, g->input, n) &&
             hy4_dense(g, m, g->branch, l->ffn_gate, l->ffn_up, l->ffn_down, n) &&
             ds4_gpu_hy4_add(g->input, g->input, g->branch, NULL, n * 3072);
        ds4_gpu_tensor_free(index);
    }
    ok = ok && hy4_norm(g->shared, m, d->norm, g->input, n);
    int previous = first;
    for (uint32_t i = 1; ok && i < n; i++) {
        ds4_gpu_tensor *row = ds4_gpu_tensor_view(g->shared, i * rb, rb);
        ok = row && ds4_gpu_tensor_write(g->embed, 0, emb + (uint64_t)previous * 6144, 6144) &&
             ds4_gpu_qwen4exp_bf16_to_f32_tensor(g->input, g->embed, 3072) &&
             hy4_norm(left, m, d->hidden_norm, row, 1) && hy4_norm(right, m, d->embed_norm, g->input, 1) &&
             hy4_mm(g->ff_gate, m, d->correct_gate, g->streams, 1) &&
             hy4_mm(g->ff_up, m, d->correct_up, g->streams, 1) &&
             ds4_gpu_hy4_swiglu(g->ff_mid, g->ff_gate, g->ff_up, 3072, INFINITY) &&
             hy4_mm(g->branch, m, d->correct_down, g->ff_mid, 1) &&
             ds4_gpu_hy4_add(g->branch, row, g->branch, NULL, 3072) &&
             hy4_mm(g->logits, &e->model, e->weights.output, g->branch, 1) &&
             ds4_gpu_argmax_tensor(g->ids, g->logits, 120832) &&
             ds4_gpu_tensor_read(g->ids, 0, &previous, sizeof(int)) && previous >= 0 && previous < 120832;
        ds4_gpu_tensor_free(row);
        if (ok) {
            g->trial[i] = previous;
        }
    }
    ds4_gpu_tensor_free(seed);
    ds4_gpu_tensor_free(left);
    ds4_gpu_tensor_free(right);
    return ok;
}
#endif
int ds4_session_hy4_trial(ds4_session *s, int first, int max_tokens, int *tokens, int *target, int cap,
                          char *err, size_t len) {
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
    if (!ds4_session_is_hy4(s) || !tokens || !target || cap < 1 || (unsigned)first >= 120832) {
        payload_set_err(err, len, "invalid Hy4 DFly2 trial");
        return -1;
    }
    if (!s->engine->mtp_ready || !s->engine->mtp_draft_tokens || max_tokens <= 0) {
        return 0;
    }
    ds4_hy4_graph *g = &s->hy4_graph;
    const uint32_t pos = g->position;
    if (!g->ready || g->failed || g->trial_n || !s->checkpoint_valid || !pos ||
        pos != (uint32_t)s->checkpoint.len || pos >= g->context) {
        payload_set_err(err, len, "Hy4 DFly2 needs a committed frontier");
        return -1;
    }
    uint32_t n = max_tokens < cap ? (uint32_t)max_tokens : (uint32_t)cap;
    if (n > HY4_DRAFT_BLOCK) {
        n = HY4_DRAFT_BLOCK;
    }
    if (n > (uint32_t)s->engine->mtp_draft_tokens + 1) {
        n = (uint32_t)s->engine->mtp_draft_tokens + 1;
    }
    if (n > g->cap) {
        n = g->cap;
    }
    if (n > g->context - pos) {
        n = g->context - pos;
    }
    if (!g->verify_logits) {
        g->verify_logits = malloc((size_t)HY4_DRAFT_BLOCK * 120832 * 4);
    }
    if (!g->verify_logits) {
        payload_set_err(err, len, "Hy4 verifier logits allocation failed");
        return -1;
    }
    g->trial[0] = first;
    const uint32_t np = pos < 2 ? pos : 2;
    bool ok = n == 1 || hy4_propose(g, s->engine, first, n);
    g->verifying = true;
    ok = ok && hy4_forward(g, s->engine, s->checkpoint.v + pos - np, np, g->trial, n, pos, true);
    g->verifying = false;
    for (uint32_t i = 0; ok && i < n; i++) {
        const float *logits = g->verify_logits + (size_t)i * 120832;
        for (uint32_t j = 0; j < 120832; j++) {
            if (!isfinite(logits[j])) {
                ok = false;
                break;
            }
        }
        target[i] = sample_argmax(logits, 120832);
    }
    if (!ok) {
        hy4_session_fail(s, err, len);
        return -1;
    }
    memcpy(tokens, g->trial, n * sizeof(int));
    g->trial_n = n;
    if (spec_sampling(&s->spec)) {
        spec_sample_rows(&s->spec, g->verify_logits, 120832, 120832, tokens, n, target);
    }
    ds4_metric_add(&ds4_metrics_get()->spec_drafts, n - 1);
    return (int)n;
#else
    (void)s;
    (void)first;
    (void)max_tokens;
    (void)tokens;
    (void)target;
    (void)cap;
    payload_set_err(err, len, "Hy4 DFly2 requires CUDA");
    return -1;
#endif
}
int ds4_session_hy4_commit(ds4_session *s, int keep, char *err, size_t len) {
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
    if (!ds4_session_is_hy4(s)) {
        payload_set_err(err, len, "invalid Hy4 DFly2 session");
        return 1;
    }
    ds4_hy4_graph *g = &s->hy4_graph;
    if (keep < 1 || (uint32_t)keep > g->trial_n || !g->verify_logits ||
        g->position != (uint32_t)s->checkpoint.len + g->trial_n) {
        return hy4_session_fail(s, err, len);
    }
    if (!ds4_gpu_tensor_copy(g->seed, 0, g->seed, keep * HY4_SEED_BYTES, HY4_SEED_BYTES)) {
        return hy4_session_fail(s, err, len);
    }
    memcpy(s->logits, g->verify_logits + (size_t)(keep - 1) * 120832, 120832 * 4);
    g->position = (uint32_t)s->checkpoint.len + (uint32_t)keep;
    for (int i = 0; i < keep; i++) {
        token_vec_push(&s->checkpoint, g->trial[i]);
    }
    ds4_metric_add(&ds4_metrics_get()->spec_hits, (uint64_t)keep - 1);
    g->trial_n = 0;
    return 0;
#else
    (void)s;
    (void)keep;
    payload_set_err(err, len, "Hy4 DFly2 requires CUDA");
    return 1;
#endif
}
