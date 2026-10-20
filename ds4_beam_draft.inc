/* DFlash2 drafting in the serial session (z-lab dflash model.py,
 * dflash_generate at temperature 0):
 *
 *   target forward ─▶ rows after layers 4/16/28/40/46 ─▶ sum fc_k ─▶ hidden_norm
 *                  ─▶ per draft layer: k_norm + RoPE K, V ─▶ that layer's ring
 *   trial  embed [anchor, mask x (n - 1)] ─▶ 5 x [ rms ─▶ conv ─▶ q/k/v ─▶ ring + block ─▶ o ─▶ conv ─▶ +h ;
 *                                                  rms ─▶ conv ─▶ SwiGLU ─▶ conv ─▶ +h ]
 *          ─▶ norm ─▶ Beam's capped head + candidate selector ─▶ n - 1 proposals
 *   verify the target forwards the n rows once; commit keeps the accepted prefix.
 *
 * Nothing rolls back. Beam's rows above the committed position are rewritten
 * before any read (SWA rings keep 512 + one chunk, full layers index by
 * position), and draft reads stop below the block, whose own rows each
 * trial writes first. */

static const float BEAM_DF_EPS = 1e-6f, BEAM_DF_THETA = 10000.0f;

static void beam_draft_free(beam_draft_rt *d) {
    if (!d) { return; }
    for (uint32_t il = 0; il < BEAM_DF_LAYERS; il++) { ds4_gpu_tensor_free(d->kv[il]); }
    ds4_gpu_tensor *all[] = {d->acc, d->tmp, d->ck, d->cv, d->h, d->x, d->xc, d->dyn, d->q, d->k, d->v,
        d->heads, d->o, d->gate, d->up, d->logits, d->proj, d->ids};
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) { ds4_gpu_tensor_free(all[i]); }
    free(d->host_logits);
    free(d->host_proj);
    free(d);
}

// Rings, the tap rows of one prefill chunk, then the block rows.
static uint64_t beam_draft_bytes(uint32_t rows) {
    return (uint64_t)BEAM_DF_LAYERS * BEAM_DF_CAP * 2u * BEAM_DF_KV * sizeof(uint16_t) +
        (uint64_t)rows * (2u * BEAM_EMBD + 2u * BEAM_DF_KV) * sizeof(float) +
        (uint64_t)BEAM_DF_BLOCK * (6u * BEAM_EMBD + BEAM_DF_DYN + 2u * BEAM_DF_KV + 2u * BEAM_DF_FF +
            DS4_N_VOCAB + BEAM_DF_RANK + 1u) * sizeof(float);
}

#ifndef __APPLE__
static beam_draft_rt *beam_draft_alloc(const ds4_beam_draft *w, uint32_t rows) {
    if (!w || !w->model || !rows) { return NULL; }
    beam_draft_rt *d = xcalloc(1, sizeof(*d));
    d->w = w;
    d->rows = rows;
    bool ok = true;
#define BEAM_DF_ALLOC(field, size) do { \
        d->field = ds4_gpu_tensor_alloc(size); \
        ok = ok && d->field; \
        d->bytes += (size); \
    } while (0)
    for (uint32_t il = 0; il < BEAM_DF_LAYERS; il++) {
        BEAM_DF_ALLOC(kv[il], (uint64_t)BEAM_DF_CAP * 2u * BEAM_DF_KV * sizeof(uint16_t));
    }
    const uint64_t f = sizeof(float), b = BEAM_DF_BLOCK;
    BEAM_DF_ALLOC(acc, rows * BEAM_EMBD * f);
    BEAM_DF_ALLOC(tmp, rows * BEAM_EMBD * f);
    BEAM_DF_ALLOC(ck, rows * BEAM_DF_KV * f);
    BEAM_DF_ALLOC(cv, rows * BEAM_DF_KV * f);
    BEAM_DF_ALLOC(h, b * BEAM_EMBD * f);
    BEAM_DF_ALLOC(x, b * BEAM_EMBD * f);
    BEAM_DF_ALLOC(xc, b * BEAM_EMBD * f);
    BEAM_DF_ALLOC(q, b * BEAM_EMBD * f);
    BEAM_DF_ALLOC(heads, b * BEAM_EMBD * f);
    BEAM_DF_ALLOC(o, b * BEAM_EMBD * f);
    BEAM_DF_ALLOC(dyn, b * BEAM_DF_DYN * f);
    BEAM_DF_ALLOC(k, b * BEAM_DF_KV * f);
    BEAM_DF_ALLOC(v, b * BEAM_DF_KV * f);
    BEAM_DF_ALLOC(gate, b * BEAM_DF_FF * f);
    BEAM_DF_ALLOC(up, b * BEAM_DF_FF * f);
    BEAM_DF_ALLOC(logits, b * DS4_N_VOCAB * f);
    BEAM_DF_ALLOC(proj, b * BEAM_DF_RANK * f);
    BEAM_DF_ALLOC(ids, b * f);
#undef BEAM_DF_ALLOC
    d->host_logits = xmalloc((size_t)BEAM_DF_BLOCK * DS4_N_VOCAB * sizeof(float));
    d->host_proj = xmalloc((size_t)BEAM_DF_BLOCK * BEAM_DF_RANK * sizeof(float));
    if (!ok || d->bytes != beam_draft_bytes(rows)) {
        beam_draft_free(d);
        return NULL;
    }
    return d;
}

/* After target layer il: fc_k of the rows into acc; after the last tap the
 * context rows (hidden_norm of the sum) go into every layer's ring as
 * k_norm + RoPE keys and plain values at their positions. */
static bool beam_draft_tap(ds4_beam_graph *g, uint32_t il, uint32_t rows, uint32_t pos) {
    beam_draft_rt *d = g->draft;
    uint32_t k = 0;
    while (k < BEAM_DF_TAPS && beam_df_taps[k] != il) { k++; }
    if (!d || k == BEAM_DF_TAPS) { return true; }
    const ds4_beam_draft *w = d->w;
    const ds4_model *m = w->model;
    if (!plain_graph_matmul_tensor(k ? d->tmp : d->acc, m, w->fc[k], BEAM_EMBD, BEAM_EMBD, g->cur, rows) ||
        (k && !ds4_gpu_exaone_add_tensor(d->acc, d->tmp, (uint64_t)rows * BEAM_EMBD))) { return false; }
    if (k + 1u < BEAM_DF_TAPS) { return true; }
    // A forward below the frontier (reset or rewind) restarts the window there.
    if (pos < g->draft_first) { g->draft_first = pos; }
    if (!ds4_gpu_exaone_rms_norm_tensor(d->acc, d->acc, m->map, m->size, w->hidden_norm->abs_offset, BEAM_EMBD,
            rows, BEAM_DF_EPS)) { return false; }
    for (uint32_t l = 0; l < BEAM_DF_LAYERS; l++) {
        const ds4_beam_draft_layer *L = &w->layer[l];
        if (!plain_graph_matmul_tensor(d->ck, m, L->k, BEAM_EMBD, BEAM_DF_KV, d->acc, rows) ||
            !plain_graph_matmul_tensor(d->cv, m, L->v, BEAM_EMBD, BEAM_DF_KV, d->acc, rows) ||
            !ds4_gpu_exaone_qk_norm_rope_tensor(d->ck, m->map, m->size, L->k_norm->abs_offset, BEAM_KV_HEADS,
                BEAM_HEAD, BEAM_HEAD, pos, rows, BEAM_DF_THETA, BEAM_DF_EPS, 1) ||
            !ds4_gpu_exaone_kv_store_tensor(d->kv[l], d->ck, d->cv, BEAM_DF_KV, rows, pos, BEAM_DF_CAP)) {
            return false;
        }
    }
    return true;
}

// The five layers over the block rows at start..start+n-1; x ends as norm(h).
static bool beam_draft_layers(ds4_beam_graph *g, uint32_t start, uint32_t n) {
    beam_draft_rt *d = g->draft;
    const ds4_beam_draft *w = d->w;
    const ds4_model *m = w->model;
    const uint64_t hidden = (uint64_t)n * BEAM_EMBD;
    for (uint32_t il = 0; il < BEAM_DF_LAYERS; il++) {
        const ds4_beam_draft_layer *l = &w->layer[il];
        // The kernel_projection of a sublayer's input feeds both its conv stages.
        if (!ds4_gpu_exaone_rms_norm_tensor(d->x, d->h, m->map, m->size, l->attn_norm->abs_offset, BEAM_EMBD, n,
                BEAM_DF_EPS) ||
            !plain_graph_matmul_tensor(d->dyn, m, l->attn_kp, BEAM_EMBD, BEAM_DF_DYN, d->x, n) ||
            !ds4_gpu_beam_df_conv(d->xc, d->x, d->dyn, m->map, m->size, l->attn_base->abs_offset, 0u, n) ||
            !plain_graph_matmul_tensor(d->q, m, l->q, BEAM_EMBD, BEAM_EMBD, d->xc, n) ||
            !plain_graph_matmul_tensor(d->k, m, l->k, BEAM_EMBD, BEAM_DF_KV, d->xc, n) ||
            !plain_graph_matmul_tensor(d->v, m, l->v, BEAM_EMBD, BEAM_DF_KV, d->xc, n) ||
            !ds4_gpu_exaone_qk_norm_rope_tensor(d->q, m->map, m->size, l->q_norm->abs_offset, BEAM_DF_HEADS,
                BEAM_HEAD, BEAM_HEAD, start, n, BEAM_DF_THETA, BEAM_DF_EPS, 1) ||
            !ds4_gpu_exaone_qk_norm_rope_tensor(d->k, m->map, m->size, l->k_norm->abs_offset, BEAM_KV_HEADS,
                BEAM_HEAD, BEAM_HEAD, start, n, BEAM_DF_THETA, BEAM_DF_EPS, 1) ||
            !ds4_gpu_exaone_kv_store_tensor(d->kv[il], d->k, d->v, BEAM_DF_KV, n, start, BEAM_DF_CAP) ||
            !ds4_gpu_beam_df_attn(d->heads, d->q, d->kv[il], g->draft_first, start, n) ||
            !plain_graph_matmul_tensor(d->o, m, l->o, BEAM_EMBD, BEAM_EMBD, d->heads, n) ||
            !ds4_gpu_beam_df_conv(d->xc, d->o, d->dyn, m->map, m->size, l->attn_base->abs_offset, 1u, n) ||
            !ds4_gpu_exaone_add_tensor(d->h, d->xc, hidden) ||
            !ds4_gpu_exaone_rms_norm_tensor(d->x, d->h, m->map, m->size, l->ffn_norm->abs_offset, BEAM_EMBD, n,
                BEAM_DF_EPS) ||
            !plain_graph_matmul_tensor(d->dyn, m, l->ffn_kp, BEAM_EMBD, BEAM_DF_DYN, d->x, n) ||
            !ds4_gpu_beam_df_conv(d->xc, d->x, d->dyn, m->map, m->size, l->ffn_base->abs_offset, 0u, n) ||
            !plain_graph_matmul_tensor(d->gate, m, l->gate, BEAM_EMBD, BEAM_DF_FF, d->xc, n) ||
            !plain_graph_matmul_tensor(d->up, m, l->up, BEAM_EMBD, BEAM_DF_FF, d->xc, n) ||
            !ds4_gpu_step37_swiglu(d->gate, d->gate, d->up, NULL, BEAM_DF_FF, n, 0.0f) ||
            !plain_graph_matmul_tensor(d->o, m, l->down, BEAM_DF_FF, BEAM_EMBD, d->gate, n) ||
            !ds4_gpu_beam_df_conv(d->xc, d->o, d->dyn, m->map, m->size, l->ffn_base->abs_offset, 1u, n) ||
            !ds4_gpu_exaone_add_tensor(d->h, d->xc, hidden)) {
            fprintf(stderr, "ds4: Beam DFlash2 layer %u failed (rows=%u pos=%u)\n", il, n, start);
            return false;
        }
    }
    return ds4_gpu_exaone_rms_norm_tensor(d->x, d->h, m->map, m->size, w->norm->abs_offset, BEAM_EMBD, n,
        BEAM_DF_EPS);
}

/* CandidateSelector.select at temperature 0: the row's top-16 capped logits
 * (unary), then score = unary + (pred[prev] * proj) . succ[candidate]; the
 * margin is the best score minus the second.
 * ponytail: host top-16 over 151,681 logits per row; a device top-k when
 * the draft step is profiled. */
static int beam_draft_pick(const ds4_beam_draft *w, const float *logits, const float *proj, int prev,
        float *margin) {
    uint32_t id[BEAM_DF_TOP];
    float unary[BEAM_DF_TOP];
    uint32_t n = 0;
    for (uint32_t i = 0; i < BEAM_TOKENS; i++) {
        const float x = logits[i];
        if (!isfinite(x)) { return -1; }
        if (n == BEAM_DF_TOP && x <= unary[n - 1u]) { continue; }
        uint32_t at = n < BEAM_DF_TOP ? n++ : n - 1u;
        for (; at && unary[at - 1u] < x; at--) { unary[at] = unary[at - 1u]; id[at] = id[at - 1u]; }
        unary[at] = x;
        id[at] = i;
    }
    float pred[BEAM_DF_RANK], succ[BEAM_DF_RANK];
    ds4_ref_row(w->model, w->pred, (uint64_t)prev, pred);
    for (uint32_t r = 0; r < BEAM_DF_RANK; r++) { pred[r] *= proj[r]; }
    int best = -1;
    float top = -INFINITY, second = -INFINITY;
    for (uint32_t k = 0; k < n; k++) {
        ds4_ref_row(w->model, w->succ, id[k], succ);
        float score = unary[k];
        for (uint32_t r = 0; r < BEAM_DF_RANK; r++) { score = fmaf(pred[r], succ[r], score); }
        if (score > top) {
            second = top;
            top = score;
            best = (int)id[k];
        } else if (score > second) {
            second = score;
        }
    }
    *margin = top - second;
    return best;
}

/* Proposals for tokens[1..n): the block's rows 1..n-1 through Beam's head
 * and the selector walk from the anchor tokens[0]. */
static bool beam_draft_propose(ds4_beam_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t n, int *tokens) {
    beam_draft_rt *d = g->draft;
    const uint32_t start = g->position, rows = n - 1u;
    int ids[BEAM_DF_BLOCK];
    ids[0] = tokens[0];
    for (uint32_t i = 1; i < n; i++) { ids[i] = BEAM_DF_MASK; }
    const uint64_t row = (uint64_t)BEAM_EMBD * sizeof(float);
    ds4_gpu_tensor *tail = ds4_gpu_tensor_view(d->x, row, (uint64_t)rows * row);
    const bool ok = tail && beam_embed(g, m, w, ids, n, d->h) && beam_draft_layers(g, start, n) &&
        beam_mm(g, m, d->logits, w->output, tail, rows) && ds4_gpu_beam_logits(d->logits, rows, BEAM_TOKENS) &&
        plain_graph_matmul_tensor(d->proj, d->w->model, d->w->project, BEAM_EMBD, BEAM_DF_RANK, tail, rows) &&
        ds4_gpu_tensor_read(d->logits, 0, d->host_logits, (uint64_t)rows * DS4_N_VOCAB * sizeof(float)) &&
        ds4_gpu_tensor_read(d->proj, 0, d->host_proj, (uint64_t)rows * BEAM_DF_RANK * sizeof(float));
    ds4_gpu_tensor_free(tail);
    if (!ok) { return false; }
    for (uint32_t i = 1; i < n; i++) {
        tokens[i] = beam_draft_pick(d->w, d->host_logits + (size_t)(i - 1u) * DS4_N_VOCAB,
            d->host_proj + (size_t)(i - 1u) * BEAM_DF_RANK, tokens[i - 1u], &d->margin[i]);
        if (tokens[i] < 0) { return false; }
    }
    return true;
}
#else
static beam_draft_rt *beam_draft_alloc(const ds4_beam_draft *w, uint32_t rows) {
    (void)w; (void)rows;
    return NULL;
}

static bool beam_draft_tap(ds4_beam_graph *g, uint32_t il, uint32_t rows, uint32_t pos) {
    (void)g; (void)il; (void)rows; (void)pos;
    return true;
}
#endif

int ds4_session_beam_trial(ds4_session *s, int first, int max_tokens, int *tokens, int *target, int cap,
        char *err, size_t errlen) {
    if (!s || !ds4_session_is_beam(s) || !tokens || !target || cap < 1) {
        payload_set_err(err, errlen, "invalid Beam trial output");
        return -1;
    }
    ds4_beam_graph *g = &s->beam_graph;
    beam_draft_rt *d = g->draft;
    if (max_tokens <= 0 || !d || getenv("DS4_MTP_SPEC_DISABLE")) { return 0; }
    if (!s->beam_graph_ready || !s->checkpoint_valid || g->failed || d->trial_n ||
        g->position != (uint32_t)s->checkpoint.len || first < 0 || (uint32_t)first >= DS4_N_VOCAB) {
        payload_set_err(err, errlen, "Beam trial needs a committed checkpoint");
        return -1;
    }
#ifdef __APPLE__
    return 0;
#else
    uint32_t n = (uint32_t)max_tokens;
    if (n > (uint32_t)cap) { n = (uint32_t)cap; }
    if (n > (uint32_t)s->engine->mtp_draft_tokens + 1u) { n = (uint32_t)s->engine->mtp_draft_tokens + 1u; }
    if (n > BEAM_DF_BLOCK) { n = BEAM_DF_BLOCK; }
    if (n > g->ctx_cap - g->position) { n = g->ctx_cap - g->position; }
    // A disk or snapshot restore brings no context rows: ordinary steps
    // refill the rings until the block has one.
    if (n < 2u || g->position <= g->draft_first) { return 0; }
    const uint32_t start = g->position;
    tokens[0] = first;
    // Both phases end in a host read, so the clock needs no extra sync.
    const double t0 = now_sec();
    if (!beam_draft_propose(g, &s->engine->model, &s->engine->weights, n, tokens)) {
        beam_session_fail(s, err, errlen);
        return -1;
    }
    /* The block is drafted whole (its attention is not causal); verification
     * stops before the first proposal whose selector margin is below
     * --mtp-margin, and a doubtful first proposal skips it for an ordinary step. */
    uint32_t usable = 1;
    while (usable < n && d->margin[usable] >= s->engine->mtp_margin) { usable++; }
    d->drafted = n;
    if (usable < 2u) {
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr, "ds4: Beam DFlash2 skip pos=%u margin=%.2f\n", start, d->margin[1]);
        }
        return 0;
    }
    n = usable;
    const double t1 = now_sec();
    d->draft_ms = (t1 - t0) * 1000.0;
    if (!beam_forward(g, &s->engine->model, &s->engine->weights, tokens, n, start, BEAM_HEAD_ALL) ||
        !ds4_gpu_argmax_rows_tensor(d->ids, d->logits, DS4_N_VOCAB, n) ||
        !ds4_gpu_tensor_read(d->ids, 0, target, (uint64_t)n * sizeof(int))) {
        beam_session_fail(s, err, errlen);
        return -1;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (target[i] < 0 || (uint32_t)target[i] >= BEAM_TOKENS) {
            beam_session_fail(s, err, errlen);
            return -1;
        }
    }
    d->verify_ms = (now_sec() - t1) * 1000.0;
    d->trial_n = n;
    d->trial_pos = start;
    memcpy(d->trial, tokens, n * sizeof(*tokens));
    ds4_metric_add(&ds4_metrics_get()->spec_drafts, n - 1u);
    return (int)n;
#endif
}

int ds4_session_beam_commit(ds4_session *s, int keep, char *err, size_t errlen) {
    ds4_beam_graph *g = s && ds4_session_is_beam(s) ? &s->beam_graph : NULL;
    beam_draft_rt *d = g ? g->draft : NULL;
    if (!d || !s->checkpoint_valid || keep < 1 || (uint32_t)keep > d->trial_n || g->failed ||
        d->trial_pos != (uint32_t)s->checkpoint.len || g->position != d->trial_pos + d->trial_n) {
        payload_set_err(err, errlen, "Beam accepted prefix has no matching trial");
        return 1;
    }
    const uint32_t trial = d->trial_n;
    d->trial_n = 0;
    if (!ds4_gpu_tensor_read(d->logits, (uint64_t)(keep - 1) * DS4_N_VOCAB * sizeof(float), s->logits,
            (uint64_t)DS4_N_VOCAB * sizeof(float))) { return beam_session_fail(s, err, errlen); }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(s->logits[i])) { return beam_session_fail(s, err, errlen); }
    }
    g->position = d->trial_pos + (uint32_t)keep;
    for (int i = 0; i < keep; i++) { token_vec_push(&s->checkpoint, d->trial[i]); }
    s->mtp_draft_valid = false;
    ds4_metric_add(&ds4_metrics_get()->spec_hits, (uint64_t)keep - 1u);
    if (getenv("DS4_MTP_SPEC_LOG")) {
        char margins[BEAM_DF_BLOCK * 12] = {0};
        for (uint32_t i = 1, at = 0; i < d->drafted && at < sizeof(margins) - 12u; i++) {
            at += (uint32_t)snprintf(margins + at, sizeof(margins) - at, i > 1u ? ",%.2f" : "%.2f", d->margin[i]);
        }
        fprintf(stderr, "ds4: Beam DFlash2 trial=%u keep=%d pos=%u draft=%.1fms verify=%.1fms margins=%s\n",
                trial, keep, g->position, d->draft_ms, d->verify_ms, margins);
    }
    return 0;
}
