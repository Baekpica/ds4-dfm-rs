/* DSpark speculative decoding in the serial session (the release's
 * `drafter` part; TokenSpeed minimax_m3_dspark, DFlash generation at
 * temperature 0):
 *
 *   target forward ─▶ residual after layers 3/17/31/45/59 (BF16 taps)
 *        ─▶ fc(concat) ─▶ hidden_norm ─▶ per draft layer k_norm + RoPE K, V
 *        ─▶ that layer's ring (the last 4,096 positions and one block)
 *   trial  embed [anchor, mask x 6] at p.. ─▶ 5 x [ grms ─▶ q/k/v ─▶ norms + RoPE
 *          ─▶ ring ─▶ attention over [max(first, p - 4096), p + 7) ─▶ o ─▶ +h ;
 *          grms ─▶ SwiGLU-OAI 12,288 ─▶ +h ] ─▶ final_norm ─▶ target head
 *          ─▶ row i + markov_w2 · markov_w1[token i] ─▶ argmax = token i + 1, on the device
 *   verify the target forwards the anchor and the proposals once; commit keeps
 *          the accepted prefix.
 *
 * DSpark samples from the anchor: block row i proposes position p + i + 1
 * (llama.cpp's sample_from_anchor default for DFlash-lineage exports, and the
 * DS41 port), so d proposals read rows 0..d-1 of the seven-row block.
 *
 * Nothing rolls back: target rows above the committed position are
 * rewritten before any read, ring rows of rejected positions are rewritten
 * by the next trial's block before attention reads them, and the context
 * window starts at `draft_first`, the lowest position this lineage has
 * written (a legacy restore or a cross-bank copy brings no context rows). */

enum { MINIMAX_DRAFT_RING = MINIMAX_DRAFT_WINDOW + MINIMAX_DRAFT_BLOCK + 1, MINIMAX_DRAFT_FC_ROWS = 256,
       MINIMAX_DRAFT_CHUNKS = (MINIMAX_DRAFT_RING + MINIMAX_BLOCK - 1) / MINIMAX_BLOCK };
static const uint32_t minimax_draft_taps[MINIMAX_DRAFT_TAPS] = {3, 17, 31, 45, 59};

struct minimax_draft_rt {
    const ds4_minimax_draft *w;
    uint32_t rows;
    uint64_t bytes;
    ds4_gpu_tensor *taps, *fcin, *ctx, *ck, *cv;
    ds4_gpu_tensor *ring_k[MINIMAX_DRAFT_LAYERS], *ring_v[MINIMAX_DRAFT_LAYERS];
    ds4_gpu_tensor *h, *x, *q, *k, *v, *heads, *part, *o, *gate, *up, *rope, *logits, *ids, *vlogits, *vids;
    uint32_t trial_n, trial_pos;
    int trial[MINIMAX_DRAFT_BLOCK + 1];
    double draft_ms, verify_ms;
};

static bool minimax_draft_payload_ready(const minimax_draft_rt *d) {
    if (!d) { return false; }
    const uint64_t bytes = (uint64_t)MINIMAX_DRAFT_RING * MINIMAX_KV * sizeof(uint16_t);
    for (uint32_t l = 0; l < MINIMAX_DRAFT_LAYERS; l++) {
        if (!d->ring_k[l] || !d->ring_v[l] || ds4_gpu_tensor_bytes(d->ring_k[l]) != bytes ||
            ds4_gpu_tensor_bytes(d->ring_v[l]) != bytes) {
            return false;
        }
    }
    return true;
}

static bool minimax_draft_committed(const minimax_draft_rt *d) {
    return d && !d->trial_n && minimax_draft_payload_ready(d);
}

static void minimax_draft_clear_trial(minimax_draft_rt *d) {
    if (!d) { return; }
    d->trial_n = d->trial_pos = 0;
    d->draft_ms = d->verify_ms = 0.0;
    memset(d->trial, 0, sizeof(d->trial));
}

/* One chronological span has at most two physical segments. Serialized
 * rows are BF16 bytes: this changes neither values nor arithmetic. */
static int minimax_draft_payload_ring(minimax_payload_dir dir, FILE *fp, ds4_gpu_tensor *ring,
        uint32_t first, uint32_t rows, uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    const uint64_t row = (uint64_t)MINIMAX_KV * sizeof(uint16_t);
    uint32_t slot = first % MINIMAX_DRAFT_RING;
    while (rows) {
        const uint32_t count = rows < MINIMAX_DRAFT_RING - slot ? rows : MINIMAX_DRAFT_RING - slot;
        const int rc = dir == MINIMAX_PAYLOAD_LOAD
            ? payload_read_tensor_span(fp, ring, slot * row, count * row, buf, DS4_SESSION_IO_CHUNK, left, err, errlen)
            : payload_write_tensor_span(fp, ring, slot * row, count * row, buf, DS4_SESSION_IO_CHUNK, err, errlen);
        if (rc) { return rc; }
        rows -= count;
        slot = 0;
    }
    return 0;
}

static int minimax_draft_payload(minimax_payload_dir dir, FILE *fp, minimax_draft_rt *d,
        uint32_t n, uint32_t rows, uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    if (rows > n || rows > MINIMAX_DRAFT_WINDOW || (rows && d && !minimax_draft_payload_ready(d)) ||
        (rows && !d && dir == MINIMAX_PAYLOAD_SAVE)) {
        payload_set_err(err, errlen, "invalid MiniMax draft snapshot rows or rings");
        return 1;
    }
    if (!rows) { return 0; }
    const off_t start = ftello(fp);
    if (!d) {
        // A target-only session can restore MMX2 without allocating a drafter.
        uint64_t bytes = minimax_payload_draft_bytes(rows);
        while (bytes) {
            const uint64_t chunk = bytes < DS4_SESSION_IO_CHUNK ? bytes : DS4_SESSION_IO_CHUNK;
            if (payload_read_bytes(fp, buf, chunk, left, err, errlen)) { return 1; }
            bytes -= chunk;
        }
    } else {
        for (uint32_t l = 0; l < MINIMAX_DRAFT_LAYERS; l++) {
            if (minimax_draft_payload_ring(dir, fp, d->ring_k[l], n - rows, rows, buf, left, err, errlen) ||
                minimax_draft_payload_ring(dir, fp, d->ring_v[l], n - rows, rows, buf, left, err, errlen)) { return 1; }
        }
    }
    if (start < 0) { return 0; } // Advisory trim requires a seekable stream.
    return snapshot_io_trim(fp, (uint64_t)start,
        dir == MINIMAX_PAYLOAD_LOAD ? SNAPSHOT_IO_READ : SNAPSHOT_IO_WRITE, err, errlen);
}

static ds4_gpu_tensor *minimax_draft_vlogits(minimax_draft_rt *d) { return d->vlogits; }

static void minimax_draft_free(minimax_draft_rt *d) {
    if (!d) { return; }
    for (uint32_t l = 0; l < MINIMAX_DRAFT_LAYERS; l++) {
        ds4_gpu_tensor_free(d->ring_k[l]);
        ds4_gpu_tensor_free(d->ring_v[l]);
    }
    ds4_gpu_tensor *all[] = {d->taps, d->fcin, d->ctx, d->ck, d->cv, d->h, d->x, d->q, d->k, d->v, d->heads, d->part, d->o,
        d->gate, d->up, d->rope, d->logits, d->ids, d->vlogits, d->vids};
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) { ds4_gpu_tensor_free(all[i]); }
    free(d);
}

// Tap rows of one prefill chunk, the context rows, the rings, the block rows,
// then the verify rows (the anchor and a block's proposals).
static uint64_t minimax_draft_bytes(uint32_t rows) {
    const uint64_t f = sizeof(float), b = MINIMAX_DRAFT_BLOCK;
    return (uint64_t)rows * MINIMAX_DRAFT_TAPS * MINIMAX_EMBD * sizeof(uint16_t) +
        (uint64_t)MINIMAX_DRAFT_FC_ROWS * MINIMAX_DRAFT_TAPS * MINIMAX_EMBD * f +
        (uint64_t)rows * (MINIMAX_EMBD + 2u * MINIMAX_KV) * f +
        (uint64_t)MINIMAX_DRAFT_LAYERS * 2u * MINIMAX_DRAFT_RING * MINIMAX_KV * sizeof(uint16_t) +
        b * (3u * MINIMAX_EMBD + 2u * MINIMAX_Q + 2u * MINIMAX_KV + 2u * MINIMAX_DENSE_FF + MINIMAX_ROT +
            DS4_N_VOCAB + (uint64_t)MINIMAX_HEADS * MINIMAX_DRAFT_CHUNKS * (MINIMAX_HEAD + 2u)) * f +
        (b + 1u) * (DS4_N_VOCAB + 2u) * f;
}

#ifndef __APPLE__
static minimax_draft_rt *minimax_draft_alloc(const ds4_minimax_draft *w, uint32_t rows) {
    if (!w || !rows) { return NULL; }
    minimax_draft_rt *d = xcalloc(1, sizeof(*d));
    d->w = w;
    d->rows = rows;
    bool ok = true;
#define MINIMAX_DRAFT_ALLOC(field, size) do { \
        d->field = ds4_gpu_tensor_alloc(size); \
        ok = ok && d->field; \
        d->bytes += (size); \
    } while (0)
    const uint64_t f = sizeof(float), b = MINIMAX_DRAFT_BLOCK;
    MINIMAX_DRAFT_ALLOC(taps, (uint64_t)rows * MINIMAX_DRAFT_TAPS * MINIMAX_EMBD * sizeof(uint16_t));
    MINIMAX_DRAFT_ALLOC(fcin, (uint64_t)MINIMAX_DRAFT_FC_ROWS * MINIMAX_DRAFT_TAPS * MINIMAX_EMBD * f);
    MINIMAX_DRAFT_ALLOC(ctx, (uint64_t)rows * MINIMAX_EMBD * f);
    MINIMAX_DRAFT_ALLOC(ck, (uint64_t)rows * MINIMAX_KV * f);
    MINIMAX_DRAFT_ALLOC(cv, (uint64_t)rows * MINIMAX_KV * f);
    for (uint32_t l = 0; l < MINIMAX_DRAFT_LAYERS; l++) {
        MINIMAX_DRAFT_ALLOC(ring_k[l], (uint64_t)MINIMAX_DRAFT_RING * MINIMAX_KV * sizeof(uint16_t));
        MINIMAX_DRAFT_ALLOC(ring_v[l], (uint64_t)MINIMAX_DRAFT_RING * MINIMAX_KV * sizeof(uint16_t));
    }
    MINIMAX_DRAFT_ALLOC(h, b * MINIMAX_EMBD * f);
    MINIMAX_DRAFT_ALLOC(x, b * MINIMAX_EMBD * f);
    MINIMAX_DRAFT_ALLOC(o, b * MINIMAX_EMBD * f);
    MINIMAX_DRAFT_ALLOC(q, b * MINIMAX_Q * f);
    MINIMAX_DRAFT_ALLOC(heads, b * MINIMAX_Q * f);
    MINIMAX_DRAFT_ALLOC(part, b * MINIMAX_HEADS * MINIMAX_DRAFT_CHUNKS * (MINIMAX_HEAD + 2u) * f);
    MINIMAX_DRAFT_ALLOC(k, b * MINIMAX_KV * f);
    MINIMAX_DRAFT_ALLOC(v, b * MINIMAX_KV * f);
    MINIMAX_DRAFT_ALLOC(gate, b * MINIMAX_DENSE_FF * f);
    MINIMAX_DRAFT_ALLOC(up, b * MINIMAX_DENSE_FF * f);
    MINIMAX_DRAFT_ALLOC(rope, b * MINIMAX_ROT * f);
    MINIMAX_DRAFT_ALLOC(logits, b * DS4_N_VOCAB * f);
    MINIMAX_DRAFT_ALLOC(vlogits, (b + 1u) * DS4_N_VOCAB * f);
    MINIMAX_DRAFT_ALLOC(ids, (b + 1u) * f);
    MINIMAX_DRAFT_ALLOC(vids, (b + 1u) * f);
#undef MINIMAX_DRAFT_ALLOC
    if (!ok || d->bytes != minimax_draft_bytes(rows)) {
        minimax_draft_free(d);
        return NULL;
    }
    return d;
}

// The residual at target layer il (see minimax_forward), when il is a tap.
static bool minimax_draft_tap(ds4_minimax_graph *g, uint32_t il, uint32_t rows) {
    minimax_draft_rt *d = g->draft;
    for (uint32_t k = 0; d && k < MINIMAX_DRAFT_TAPS; k++) {
        if (minimax_draft_taps[k] == il) {
            return ds4_gpu_minimax_draft_tap(d->taps, g->cur, k, rows, g->cur_rows) != 0;
        }
    }
    return true;
}

/* After the target layers: the rows' context, fc over the taps'
 * concatenation (fc input widened MINIMAX_DRAFT_FC_ROWS rows at a time),
 * hidden_norm, then every draft layer's k_norm + RoPE keys and plain values
 * into its ring. The context skips the draft layers' input norm. */
static bool minimax_draft_context(ds4_minimax_graph *g, const ds4_model *m, uint32_t rows, uint32_t pos) {
    minimax_draft_rt *d = g->draft;
    if (!d) { return true; }
    const ds4_minimax_draft *w = d->w;
    // A forward below the frontier (reset or rewind) restarts the window there.
    if (pos < g->draft_first) { g->draft_first = pos; }
    const uint64_t row = (uint64_t)MINIMAX_EMBD * sizeof(float);
    for (uint32_t r0 = 0; r0 < rows; r0 += MINIMAX_DRAFT_FC_ROWS) {
        const uint32_t n = rows - r0 < MINIMAX_DRAFT_FC_ROWS ? rows - r0 : MINIMAX_DRAFT_FC_ROWS;
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(d->ctx, (uint64_t)r0 * row, (uint64_t)n * row);
        const bool ok = out && ds4_gpu_minimax_draft_widen(d->fcin, d->taps, r0, n) &&
            plain_graph_matmul_tensor(out, m, w->fc, MINIMAX_DRAFT_TAPS * MINIMAX_EMBD, MINIMAX_EMBD, d->fcin, n);
        ds4_gpu_tensor_free(out);
        if (!ok) { return false; }
    }
    if (!ds4_gpu_minimax_rms(g->gvq, d->ctx, d->ctx, minimax_index(m, w->hidden_norm), rows, DS4_GVQ_ROWS_F32)) {
        return false;
    }
    for (uint32_t l = 0; l < MINIMAX_DRAFT_LAYERS; l++) {
        const ds4_minimax_draft_layer *L = &w->layer[l];
        if (!plain_graph_matmul_tensor(d->ck, m, L->k, MINIMAX_EMBD, MINIMAX_KV, d->ctx, rows) ||
            !plain_graph_matmul_tensor(d->cv, m, L->v, MINIMAX_EMBD, MINIMAX_KV, d->ctx, rows) ||
            !ds4_gpu_minimax_qk(g->gvq, d->ck, minimax_index(m, L->k_norm), MINIMAX_KV_HEADS, rows, g->rope) ||
            !ds4_gpu_minimax_draft_ring(d->ring_k[l], d->ring_v[l], d->ck, d->cv, rows, pos, MINIMAX_DRAFT_RING)) {
            fprintf(stderr, "ds4: MiniMax DSpark context of layer %u failed (rows=%u pos=%u)\n", l, rows, pos);
            return false;
        }
    }
    return true;
}

// The five layers over the block rows at p..p+6; x ends as final_norm(h).
static bool minimax_draft_layers(ds4_minimax_graph *g, const ds4_model *m, uint32_t p) {
    minimax_draft_rt *d = g->draft;
    const ds4_minimax_draft *w = d->w;
    const uint32_t n = MINIMAX_DRAFT_BLOCK;
    uint32_t lo = p - g->draft_first > MINIMAX_DRAFT_WINDOW ? p - MINIMAX_DRAFT_WINDOW : g->draft_first;
    // DS4_MINIMAX_DSPARK_NOCTX=1 (diagnostic): the block attends to itself only.
    if (getenv("DS4_MINIMAX_DSPARK_NOCTX")) { lo = p; }
    if (!ds4_gpu_minimax_rope(d->rope, n, p) ||
        !ds4_gpu_minimax_rms(g->gvq, d->x, d->h, minimax_index(m, w->layer[0].attn_norm), n, DS4_GVQ_ROWS_F32)) {
        return false;
    }
    for (uint32_t l = 0; l < MINIMAX_DRAFT_LAYERS; l++) {
        const ds4_minimax_draft_layer *L = &w->layer[l];
        const ds4_tensor *next = l + 1u < MINIMAX_DRAFT_LAYERS ? w->layer[l + 1u].attn_norm : w->final_norm;
        if (!plain_graph_matmul_tensor(d->q, m, L->q, MINIMAX_EMBD, MINIMAX_Q, d->x, n) ||
            !plain_graph_matmul_tensor(d->k, m, L->k, MINIMAX_EMBD, MINIMAX_KV, d->x, n) ||
            !plain_graph_matmul_tensor(d->v, m, L->v, MINIMAX_EMBD, MINIMAX_KV, d->x, n) ||
            !ds4_gpu_minimax_qk(g->gvq, d->q, minimax_index(m, L->q_norm), MINIMAX_HEADS, n, d->rope) ||
            !ds4_gpu_minimax_qk(g->gvq, d->k, minimax_index(m, L->k_norm), MINIMAX_KV_HEADS, n, d->rope) ||
            !ds4_gpu_minimax_draft_ring(d->ring_k[l], d->ring_v[l], d->k, d->v, n, p, MINIMAX_DRAFT_RING) ||
            !ds4_gpu_minimax_draft_attn(d->heads, d->part, d->q, d->ring_k[l], d->ring_v[l], n, lo, p + n,
                MINIMAX_DRAFT_RING) ||
            !plain_graph_matmul_tensor(d->o, m, L->o, MINIMAX_Q, MINIMAX_EMBD, d->heads, n) ||
            !ds4_gpu_minimax_residual(g->gvq, d->h, d->o, d->x, minimax_index(m, L->ffn_norm), n, NULL, NULL, 0,
                DS4_GVQ_ROWS_F32, DS4_MINIMAX_NORM_F32) ||
            !plain_graph_matmul_tensor(d->gate, m, L->gate, MINIMAX_EMBD, MINIMAX_DENSE_FF, d->x, n) ||
            !plain_graph_matmul_tensor(d->up, m, L->up, MINIMAX_EMBD, MINIMAX_DENSE_FF, d->x, n) ||
            !ds4_gpu_minimax_swiglu(d->gate, d->gate, d->up, n, MINIMAX_DENSE_FF) ||
            !plain_graph_matmul_tensor(d->o, m, L->down, MINIMAX_DENSE_FF, MINIMAX_EMBD, d->gate, n) ||
            !ds4_gpu_minimax_residual(g->gvq, d->h, d->o, d->x, minimax_index(m, next), n, NULL, NULL, 0,
                DS4_GVQ_ROWS_F32, DS4_MINIMAX_NORM_F32)) {
            fprintf(stderr, "ds4: MiniMax DSpark layer %u failed (pos=%u)\n", l, p);
            return false;
        }
    }
    return true;
}

/* Proposals for tokens[1..n): block rows 0..n-2 through the target head,
 * row i plus the Markov bias of token i, argmax into token i + 1 in order on
 * the device (the anchor is tokens[0]). */
static bool minimax_draft_propose(ds4_minimax_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t n,
        int *tokens) {
    minimax_draft_rt *d = g->draft;
    const uint32_t p = g->position, rows = n - 1u;
    int ids[MINIMAX_DRAFT_BLOCK + 1];
    ids[0] = tokens[0];
    for (uint32_t i = 1; i < MINIMAX_DRAFT_BLOCK; i++) { ids[i] = MINIMAX_DRAFT_MASK; }
    const uint64_t vrow = (uint64_t)DS4_N_VOCAB * sizeof(float);
    bool ok = minimax_embed(g, m, w, ids, MINIMAX_DRAFT_BLOCK, d->h) && minimax_draft_layers(g, m, p) &&
        minimax_mm(g, m, d->logits, w->output, d->x, rows) && ds4_gpu_minimax_logits(d->logits, rows, MINIMAX_TOKENS) &&
        ds4_gpu_tensor_write(d->ids, 0, ids, sizeof(int32_t));
    for (uint32_t i = 0; ok && i < rows; i++) {
        ds4_gpu_tensor *logits = ds4_gpu_tensor_view(d->logits, (uint64_t)i * vrow, vrow);
        ds4_gpu_tensor *prev = ds4_gpu_tensor_view(d->ids, (uint64_t)i * sizeof(int32_t), sizeof(int32_t));
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(d->ids, (uint64_t)(i + 1u) * sizeof(int32_t), sizeof(int32_t));
        // DS4_MINIMAX_DSPARK_MARKOV=0 (diagnostic): the head's argmax alone.
        static int markov = -1;
        if (markov < 0) {
            const char *v = getenv("DS4_MINIMAX_DSPARK_MARKOV");
            markov = !(v && !strcmp(v, "0"));
        }
        ok = logits && prev && out &&
            (!markov || ds4_gpu_minimax_draft_markov(g->gvq, logits, minimax_index(m, d->w->markov_w1),
                minimax_index(m, d->w->markov_w2), prev, DS4_N_VOCAB)) &&
            ds4_gpu_argmax_rows_tensor(out, logits, DS4_N_VOCAB, 1u);
        ds4_gpu_tensor_free(logits);
        ds4_gpu_tensor_free(prev);
        ds4_gpu_tensor_free(out);
    }
    if (!ok || !ds4_gpu_tensor_read(d->ids, 0, ids, (uint64_t)n * sizeof(int32_t))) { return false; }
    for (uint32_t i = 1; i < n; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= MINIMAX_TOKENS) { return false; }
        tokens[i] = ids[i];
    }
    return true;
}
#else
static minimax_draft_rt *minimax_draft_alloc(const ds4_minimax_draft *w, uint32_t rows) {
    (void)w; (void)rows;
    return NULL;
}

static bool minimax_draft_tap(ds4_minimax_graph *g, uint32_t il, uint32_t rows) {
    (void)g; (void)il; (void)rows;
    return true;
}

static bool minimax_draft_context(ds4_minimax_graph *g, const ds4_model *m, uint32_t rows, uint32_t pos) {
    (void)g; (void)m; (void)rows; (void)pos;
    return true;
}
#endif

static bool minimax_drafting(const ds4_engine *e) {
    return e->weights.minimax_draft && e->mtp_draft_tokens > 1;
}

#ifndef __APPLE__
/* DS4_MINIMAX_VERIFY_CHECK=1 (diagnostic): re-run the n verified rows as
 * single-row decodes at the same positions and report, per row, how many
 * logits moved, the largest move, both argmaxes and the decode top-2 gap.
 * The decodes rewrite the rows' K/V; the verify logits stay for commit. */
static bool minimax_verify_check(ds4_session *s, const int *tokens, uint32_t n, uint32_t start) {
    ds4_minimax_graph *g = &s->minimax_graph;
    const uint64_t vrow = DS4_N_VOCAB;
    float *verify = xmalloc((size_t)n * vrow * sizeof(float)), *one = xmalloc((size_t)vrow * sizeof(float));
    bool ok = ds4_gpu_tensor_read(g->draft->vlogits, 0, verify, (uint64_t)n * vrow * sizeof(float));
    g->position = start;
    for (uint32_t i = 0; ok && i < n; i++) {
        ok = minimax_forward(g, &s->engine->model, &s->engine->weights, tokens + i, 1u, start + i,
                MINIMAX_HEAD_LAST) && ds4_gpu_tensor_read(g->logits, 0, one, vrow * sizeof(float));
        if (!ok) { break; }
        const float *v = verify + (uint64_t)i * vrow;
        uint32_t differ = 0, av = 0, ad = 0, second = UINT32_MAX;
        float worst = 0.0f;
        for (uint32_t t = 0; t < MINIMAX_TOKENS; t++) {
            if (v[t] != one[t]) { differ++; worst = fmaxf(worst, fabsf(v[t] - one[t])); }
            if (v[t] > v[av]) { av = t; }
            if (one[t] > one[ad]) { ad = t; }
        }
        for (uint32_t t = 0; t < MINIMAX_TOKENS; t++) {
            if (t != ad && (second == UINT32_MAX || one[t] > one[second])) { second = t; }
        }
        fprintf(stderr, "ds4: MiniMax verify check pos=%u row=%u: %u logits differ, max |d| %.4g, argmax verify %u "
                "decode %u, decode top-2 gap %.4g\n", start + i, i, differ, worst, av, ad, one[ad] - one[second]);
    }
    free(one);
    free(verify);
    return ok;
}
#endif

int ds4_session_minimax_trial(ds4_session *s, int first, int max_tokens, int *tokens, int *target, int cap,
        char *err, size_t errlen) {
    if (!s || !ds4_session_is_minimax(s) || !tokens || !target || cap < 1) {
        payload_set_err(err, errlen, "invalid MiniMax trial output");
        return -1;
    }
    ds4_minimax_graph *g = &s->minimax_graph;
    minimax_draft_rt *d = g->draft;
    if (max_tokens <= 0 || !d || getenv("DS4_MTP_SPEC_DISABLE")) { return 0; }
    if (!s->minimax_graph_ready || !s->checkpoint_valid || g->failed || d->trial_n ||
        g->position != (uint32_t)s->checkpoint.len || first < 0 || (uint32_t)first >= MINIMAX_TOKENS) {
        payload_set_err(err, errlen, "MiniMax trial needs a committed checkpoint");
        return -1;
    }
#ifdef __APPLE__
    return 0;
#else
    uint32_t n = (uint32_t)max_tokens;
    if (n > (uint32_t)cap) { n = (uint32_t)cap; }
    if (n > (uint32_t)s->engine->mtp_draft_tokens + 1u) { n = (uint32_t)s->engine->mtp_draft_tokens + 1u; }
    if (n > MINIMAX_DRAFT_BLOCK + 1u) { n = MINIMAX_DRAFT_BLOCK + 1u; }
    // The block's seven rows and the verify rows must fit. A legacy restore
    // needs an ordinary step to refill context before the block can draft.
    if (n < 2u || g->ctx_cap - g->position < MINIMAX_DRAFT_BLOCK + 1u || g->position <= g->draft_first) { return 0; }
    const uint32_t start = g->position;
    tokens[0] = first;
    const double t0 = now_sec();
    if (!minimax_draft_propose(g, &s->engine->model, &s->engine->weights, n, tokens)) {
        minimax_session_fail(s, err, errlen);
        return -1;
    }
    const double t1 = now_sec();
    d->draft_ms = (t1 - t0) * 1000.0;
    if (!minimax_forward(g, &s->engine->model, &s->engine->weights, tokens, n, start, MINIMAX_HEAD_ALL) ||
        !ds4_gpu_argmax_rows_tensor(d->vids, d->vlogits, DS4_N_VOCAB, n) ||
        !ds4_gpu_tensor_read(d->vids, 0, target, (uint64_t)n * sizeof(int))) {
        minimax_session_fail(s, err, errlen);
        return -1;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (target[i] < 0 || (uint32_t)target[i] >= MINIMAX_TOKENS) {
            minimax_session_fail(s, err, errlen);
            return -1;
        }
    }
    d->verify_ms = (now_sec() - t1) * 1000.0;
    if (getenv("DS4_MINIMAX_VERIFY_CHECK") && !minimax_verify_check(s, tokens, n, start)) {
        minimax_session_fail(s, err, errlen);
        return -1;
    }
    if (getenv("DS4_MTP_SPEC_LOG")) {
        // Per slot: does proposal i equal the target's argmax after row i - 1?
        char hits[MINIMAX_DRAFT_BLOCK + 1] = {0};
        for (uint32_t i = 1; i < n; i++) { hits[i - 1] = tokens[i] == target[i - 1] ? '1' : '0'; }
        fprintf(stderr, "ds4: MiniMax DSpark slots=%s\n", hits);
    }
    d->trial_n = n;
    d->trial_pos = start;
    memcpy(d->trial, tokens, n * sizeof(*tokens));
    ds4_metric_add(&ds4_metrics_get()->spec_drafts, n - 1u);
    return (int)n;
#endif
}

int ds4_session_minimax_commit(ds4_session *s, int keep, char *err, size_t errlen) {
    ds4_minimax_graph *g = s && ds4_session_is_minimax(s) ? &s->minimax_graph : NULL;
    minimax_draft_rt *d = g ? g->draft : NULL;
    if (!d || !s->checkpoint_valid || keep < 1 || (uint32_t)keep > d->trial_n || g->failed ||
        d->trial_pos != (uint32_t)s->checkpoint.len || g->position != d->trial_pos + d->trial_n) {
        payload_set_err(err, errlen, "MiniMax accepted prefix has no matching trial");
        return 1;
    }
    const uint32_t trial = d->trial_n;
    d->trial_n = 0;
    if (!ds4_gpu_tensor_read(d->vlogits, (uint64_t)(keep - 1) * DS4_N_VOCAB * sizeof(float), s->logits,
            (uint64_t)DS4_N_VOCAB * sizeof(float))) { return minimax_session_fail(s, err, errlen); }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(s->logits[i])) { return minimax_session_fail(s, err, errlen); }
    }
    g->position = d->trial_pos + (uint32_t)keep;
    for (int i = 0; i < keep; i++) { token_vec_push(&s->checkpoint, d->trial[i]); }
    s->mtp_draft_valid = false;
    ds4_metric_add(&ds4_metrics_get()->spec_hits, (uint64_t)keep - 1u);
    if (getenv("DS4_MTP_SPEC_LOG")) {
        fprintf(stderr, "ds4: MiniMax DSpark trial=%u keep=%d pos=%u draft=%.1fms verify=%.1fms\n", trial, keep,
                g->position, d->draft_ms, d->verify_ms);
    }
    return 0;
}
