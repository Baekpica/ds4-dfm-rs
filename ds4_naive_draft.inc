/* DSpark context is projected once from new target taps. Only this bounded
 * KV survives between rounds; tap/forward scratch is shared by all banks. */
static bool naive_linear(ds4_gpu_tensor *out, const ds4_model *m,
                           const ds4_tensor *w, unsigned in, unsigned width,
                           const ds4_gpu_tensor *x, unsigned n) {
    if (n <= 1 || n > N05_DF_BLOCK) {
        return plain_graph_matmul_tensor(out, m, w, in, width, x, n);
    }
    /* Verification must use the same reduction/activation quantization as
     * decode. Width-dependent MMVQ rounding can otherwise flip routed IDs. */
    for (unsigned row = 0; row < n; row++) {
        ds4_gpu_tensor *src = ds4_gpu_tensor_view(x, (uint64_t)row * in * sizeof(float), in * sizeof(float));
        ds4_gpu_tensor *dst = ds4_gpu_tensor_view(out, (uint64_t)row * width * sizeof(float), width * sizeof(float));
        const bool ok = src && dst && plain_graph_matmul_tensor(dst, m, w, in, width, src, 1);
        ds4_gpu_tensor_free(src); ds4_gpu_tensor_free(dst);
        if (!ok) { return false; }
    }
    return true;
}

typedef struct {
    ds4_gpu_tensor *tap, *feature, *ctx_k, *ctx_v, *positions, *rope, *freq;
    ds4_gpu_tensor *cur, *norm, *q, *k, *v, *heads, *attn, *gate, *up, *mid, *ffn, *hidden;
    ds4_gpu_tensor *base, *verify, *markov, *bias, *token, *pick, *confidence;
    uint64_t bytes;
} ds4_naive_df_scratch;

typedef struct {
    const ds4_naive_draft *weights;
    ds4_naive_df_scratch *ws;
    ds4_gpu_tensor *kv[N05_DF_LAYERS];
    unsigned position, first, trial_n, trial_pos, usable;
    int trial[N05_DF_BLOCK];
    float confidence[N05_DF_BLOCK];
    bool owns_scratch;
} ds4_naive_df_runtime;

static uint64_t naive_draft_kv_bytes(void) {
    return (uint64_t)N05_DF_LAYERS * N05_DF_CAP * 2 * N05_DF_KV * N05_DF_DIM * sizeof(uint16_t);
}

static uint64_t naive_draft_ws_bytes(void) {
    const uint64_t ctx = (uint64_t)N05_DF_WINDOW *
        (N05_DF_SLOT + N05_EMBED + 2 * N05_DF_KV * N05_DF_DIM + 1 + N05_DF_DIM);
    const uint64_t block = (uint64_t)N05_DF_BLOCK *
        (10 * N05_EMBED + 2 * N05_DF_KV * N05_DF_DIM + 2 * N05_VOCAB + 1);
    return (ctx + block + N05_DF_DIM / 2 + N05_DF_RANK + N05_VOCAB + 2) * sizeof(float)
        + N05_DF_BLOCK * (sizeof(ds4_gpu_top2_result) - sizeof(float));
}

static void naive_draft_free(ds4_naive_df_runtime *d) {
    if (!d) { return; }
    for (unsigned il = 0; il < N05_DF_LAYERS; il++) { ds4_gpu_tensor_free(d->kv[il]); }
    if (d->owns_scratch && d->ws) {
        ds4_naive_df_scratch *s = d->ws;
        ds4_gpu_tensor *all[] = {s->tap, s->feature, s->ctx_k, s->ctx_v, s->positions, s->rope, s->freq,
            s->cur, s->norm, s->q, s->k, s->v, s->heads, s->attn, s->gate, s->up, s->mid, s->ffn, s->hidden,
            s->base, s->verify, s->markov, s->bias, s->token, s->pick, s->confidence};
        for (unsigned i = 0; i < sizeof(all) / sizeof(all[0]); i++) { ds4_gpu_tensor_free(all[i]); }
        free(s);
    }
    memset(d, 0, sizeof(*d));
}

static bool naive_draft_cache(ds4_naive_df_runtime *d) {
    const uint64_t row = 2 * N05_DF_KV * N05_DF_DIM * sizeof(uint16_t);
    for (unsigned il = 0; il < N05_DF_LAYERS; il++) {
        d->kv[il] = ds4_gpu_tensor_alloc(N05_DF_CAP * row);
        if (!d->kv[il]) { return false; }
    }
    return true;
}

static bool naive_draft_alloc(ds4_naive_df_runtime *d, const ds4_naive_draft *w) {
    if (!d || !w || !w->model) { return false; }
    naive_draft_free(d);
    d->weights = w; d->owns_scratch = true;
    d->ws = xcalloc(1, sizeof(*d->ws));
    ds4_naive_df_scratch *s = d->ws;
#define N05_DF_ALLOC(field, count) do { \
    const uint64_t bytes = (uint64_t)(count) * sizeof(float); \
    s->field = ds4_gpu_tensor_alloc(bytes); \
    if (!s->field) { naive_draft_free(d); return false; } \
    s->bytes += bytes; \
} while (0)
    N05_DF_ALLOC(tap, N05_DF_WINDOW * N05_DF_SLOT);
    N05_DF_ALLOC(feature, N05_DF_WINDOW * N05_EMBED);
    N05_DF_ALLOC(ctx_k, N05_DF_WINDOW * N05_DF_KV * N05_DF_DIM);
    N05_DF_ALLOC(ctx_v, N05_DF_WINDOW * N05_DF_KV * N05_DF_DIM);
    N05_DF_ALLOC(positions, N05_DF_WINDOW);
    N05_DF_ALLOC(rope, N05_DF_WINDOW * N05_DF_DIM);
    N05_DF_ALLOC(freq, N05_DF_DIM / 2);
    N05_DF_ALLOC(cur, N05_DF_BLOCK * N05_EMBED);
    N05_DF_ALLOC(norm, N05_DF_BLOCK * N05_EMBED);
    N05_DF_ALLOC(q, N05_DF_BLOCK * N05_DF_HEADS * N05_DF_DIM);
    N05_DF_ALLOC(k, N05_DF_BLOCK * N05_DF_KV * N05_DF_DIM);
    N05_DF_ALLOC(v, N05_DF_BLOCK * N05_DF_KV * N05_DF_DIM);
    N05_DF_ALLOC(heads, N05_DF_BLOCK * N05_EMBED);
    N05_DF_ALLOC(attn, N05_DF_BLOCK * N05_EMBED);
    N05_DF_ALLOC(gate, N05_DF_BLOCK * N05_EMBED);
    N05_DF_ALLOC(up, N05_DF_BLOCK * N05_EMBED);
    N05_DF_ALLOC(mid, N05_DF_BLOCK * N05_EMBED);
    N05_DF_ALLOC(ffn, N05_DF_BLOCK * N05_EMBED);
    N05_DF_ALLOC(hidden, N05_DF_BLOCK * N05_EMBED);
    N05_DF_ALLOC(base, N05_DF_BLOCK * N05_VOCAB);
    N05_DF_ALLOC(verify, N05_DF_BLOCK * N05_VOCAB);
    N05_DF_ALLOC(markov, N05_DF_RANK);
    N05_DF_ALLOC(bias, N05_VOCAB);
    N05_DF_ALLOC(token, 1);
    N05_DF_ALLOC(pick, N05_DF_BLOCK * sizeof(ds4_gpu_top2_result) / sizeof(float));
    N05_DF_ALLOC(confidence, 1);
#undef N05_DF_ALLOC
    float freq[N05_DF_DIM / 2];
    for (unsigned i = 0; i < N05_DF_DIM / 2; i++) { freq[i] = (float)pow(10000.0, -2.0 * i / N05_DF_DIM); }
    if (s->bytes != naive_draft_ws_bytes() || !naive_draft_cache(d) ||
        !ds4_gpu_tensor_write(s->freq, 0, freq, sizeof(freq))) { naive_draft_free(d); return false; }
    return true;
}

static bool naive_draft_clone(ds4_naive_df_runtime *d, const ds4_naive_df_runtime *src) {
    if (!d || !src || !src->ws || !src->owns_scratch) { return false; }
    memset(d, 0, sizeof(*d));
    d->ws = src->ws; d->weights = src->weights;
    if (!naive_draft_cache(d)) { naive_draft_free(d); return false; }
    return true;
}

static bool naive_df_positions(ds4_naive_df_scratch *s, unsigned start, unsigned rows) {
    unsigned positions[N05_DF_WINDOW];
    if (!rows || rows > N05_DF_WINDOW || start > N05_CONTEXT || rows > N05_CONTEXT - start) { return false; }
    for (unsigned i = 0; i < rows; i++) { positions[i] = start + i; }
    return ds4_gpu_tensor_write(s->positions, 0, positions, rows * sizeof(unsigned)) &&
        ds4_gpu_step37_rope(s->rope, s->freq, s->positions, N05_DF_DIM, rows);
}

static bool naive_draft_prepare(ds4_naive_df_runtime *d, unsigned pos, unsigned n) {
    if (!d) { return false; }
    if (!d->ws) { return true; }
    if (!n || d->position != pos) { return false; }
    ds4_naive_df_scratch *s = d->ws;
    const ds4_naive_draft *w = d->weights;
    const ds4_model *m = w->model;
    const unsigned rows = n < N05_DF_WINDOW ? n : N05_DF_WINDOW, start = pos + n - rows;
    if (!naive_df_positions(s, start, rows) ||
        !naive_linear(s->feature, m, w->fc, N05_DF_SLOT, N05_EMBED, s->tap, rows) ||
        !ds4_gpu_naive_round(s->feature, (uint64_t)rows * N05_EMBED) ||
        !ds4_gpu_naive_rms(s->feature, s->feature, m->map, m->size, w->hidden_norm->abs_offset, rows)) { return false; }
    for (unsigned il = 0; il < N05_DF_LAYERS; il++) {
        if (!naive_linear(s->ctx_k, m, w->k[il], N05_EMBED, N05_DF_KV * N05_DF_DIM, s->feature, rows) ||
            !naive_linear(s->ctx_v, m, w->v[il], N05_EMBED, N05_DF_KV * N05_DF_DIM, s->feature, rows) ||
            !ds4_gpu_naive_df_norm(s->ctx_k, m->map, m->size, w->k_norm[il]->abs_offset, N05_DF_KV, rows) ||
            !ds4_gpu_naive_df_rope(s->ctx_k, s->rope, N05_DF_KV, rows) ||
            !ds4_gpu_naive_df_store(d->kv[il], s->ctx_k, s->ctx_v, s->positions, rows)) { return false; }
    }
    d->position = pos + n;
    if (n >= N05_DF_WINDOW) { d->first = start; }
    if (d->position > N05_DF_CAP && d->first < d->position - N05_DF_CAP) { d->first = d->position - N05_DF_CAP; }
    return true;
}

static bool naive_draft_layers(ds4_naive_df_runtime *d, unsigned rows) {
    if (!d || !d->ws || !d->weights) { return false; }
    ds4_naive_df_scratch *s = d->ws;
    const ds4_naive_draft *w = d->weights;
    const ds4_model *m = w->model;
    if (!naive_df_positions(s, d->position, rows) ||
        !ds4_gpu_naive_df_mask(s->cur, m->map, m->size, w->mask->abs_offset, rows)) { return false; }
    for (unsigned il = 0; il < N05_DF_LAYERS; il++) {
        if (!ds4_gpu_naive_rms(s->norm, s->cur, m->map, m->size, w->attn_norm[il]->abs_offset, rows) ||
            !plain_graph_matmul_tensor(s->q, m, w->q[il], N05_EMBED, N05_EMBED, s->norm, rows) ||
            !plain_graph_matmul_tensor(s->k, m, w->k[il], N05_EMBED, N05_DF_KV * N05_DF_DIM, s->norm, rows) ||
            !plain_graph_matmul_tensor(s->v, m, w->v[il], N05_EMBED, N05_DF_KV * N05_DF_DIM, s->norm, rows) ||
            !ds4_gpu_naive_df_norm(s->q, m->map, m->size, w->q_norm[il]->abs_offset, N05_DF_HEADS, rows) ||
            !ds4_gpu_naive_df_norm(s->k, m->map, m->size, w->k_norm[il]->abs_offset, N05_DF_KV, rows) ||
            !ds4_gpu_naive_df_rope(s->q, s->rope, N05_DF_HEADS, rows) ||
            !ds4_gpu_naive_df_rope(s->k, s->rope, N05_DF_KV, rows) ||
            !ds4_gpu_naive_df_attn(s->heads, s->q, d->kv[il], s->k, s->v, d->first, d->position, rows) ||
            !plain_graph_matmul_tensor(s->attn, m, w->o[il], N05_EMBED, N05_EMBED, s->heads, rows) ||
            !ds4_gpu_naive_add(s->cur, s->attn, (uint64_t)rows * N05_EMBED) ||
            !ds4_gpu_naive_rms(s->norm, s->cur, m->map, m->size, w->ffn_norm[il]->abs_offset, rows) ||
            !plain_graph_matmul_tensor(s->gate, m, w->gate[il], N05_EMBED, N05_EMBED, s->norm, rows) ||
            !plain_graph_matmul_tensor(s->up, m, w->up[il], N05_EMBED, N05_EMBED, s->norm, rows) ||
            !ds4_gpu_naive_swiglu(s->mid, s->gate, s->up, (uint64_t)rows * N05_EMBED) ||
            !plain_graph_matmul_tensor(s->ffn, m, w->down[il], N05_EMBED, N05_EMBED, s->mid, rows) ||
            !ds4_gpu_naive_add(s->cur, s->ffn, (uint64_t)rows * N05_EMBED)) { return false; }
    }
    return ds4_gpu_naive_rms(s->hidden, s->cur, m->map, m->size, w->output_norm->abs_offset, rows);
}

static bool naive_draft_block(ds4_naive_df_runtime *d, const ds4_model *target,
                               const ds4_weights *w, int anchor, float margin, unsigned rows, int *tokens) {
    if (!d || !d->ws || !d->weights || !target || !w) { return false; }
    ds4_naive_df_scratch *s = d->ws;
    const ds4_naive_draft *dw = d->weights;
    const ds4_model *m = dw->model;
    if (!s || !tokens || rows < 2 || rows > N05_DF_BLOCK || d->trial_n || anchor < 0 || (unsigned)anchor >= N05_VOCAB ||
        !ds4_gpu_tensor_write(s->token, 0, &anchor, sizeof(anchor)) ||
        !ds4_gpu_embed_tokens_quant_tensor(s->cur, s->token, target->map, target->size,
            w->token_embd->abs_offset, w->token_embd->type, N05_VOCAB, 1, N05_EMBED) ||
        !naive_draft_layers(d, rows) ||
        !plain_graph_matmul_tensor(s->base, target, w->output, N05_EMBED, N05_VOCAB, s->hidden, rows) ||
        !ds4_gpu_naive_round(s->base, (uint64_t)rows * N05_VOCAB)) { return false; }
    tokens[0] = anchor;
    d->usable = 1;
    for (unsigned i = 1; i < rows; i++) {
        ds4_gpu_top2_result top = {0};
        ds4_gpu_tensor *logits = ds4_gpu_tensor_view(s->base, (uint64_t)i * N05_VOCAB * sizeof(float), N05_VOCAB * sizeof(float));
        ds4_gpu_tensor *hidden = ds4_gpu_tensor_view(s->hidden, (uint64_t)i * N05_EMBED * sizeof(float), N05_EMBED * sizeof(float));
        const bool ok = logits && hidden && ds4_gpu_tensor_write(s->token, 0, &tokens[i - 1], sizeof(int)) &&
            ds4_gpu_embed_tokens_quant_tensor(s->markov, s->token, m->map, m->size,
                dw->markov1->abs_offset, dw->markov1->type, N05_VOCAB, 1, N05_DF_RANK) &&
            ds4_gpu_naive_round(s->markov, N05_DF_RANK) &&
            plain_graph_matmul_tensor(s->bias, m, dw->markov2, N05_DF_RANK, N05_VOCAB, s->markov, 1) &&
            ds4_gpu_naive_add(logits, s->bias, N05_VOCAB) &&
            ds4_gpu_naive_df_conf(s->confidence, hidden, s->markov, m->map, m->size, dw->conf->abs_offset, dw->bias->abs_offset) &&
            ds4_gpu_naive_df_top2(s->pick, logits) &&
            ds4_gpu_tensor_read(s->pick, 0, &top, sizeof(top)) &&
            ds4_gpu_tensor_read(s->confidence, 0, &d->confidence[i], sizeof(float));
        ds4_gpu_tensor_free(logits); ds4_gpu_tensor_free(hidden);
        if (!ok || top.id0 >= N05_VOCAB || !isfinite(top.value0) || !isfinite(top.value1) || !isfinite(d->confidence[i])) { return false; }
        tokens[i] = (int)top.id0;
        if (d->usable == i && top.value0 - top.value1 >= margin) { d->usable++; }
    }
    return true;
}
