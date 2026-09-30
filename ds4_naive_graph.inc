/* Naive owns full DSA histories and bounded SWA rings. All matmuls retain
 * the common mmap/MMQ boundary; no target state uses MLA/compressed KV. */
typedef struct {
    ds4_exaone_batch_ws ws;
    ds4_gpu_tensor *positions, *tokens, *freq_full, *freq_swa;
    ds4_gpu_tensor *rope_full, *rope_swa, *logits;
    ds4_gpu_tensor *index_q, *index_k, *index_weights, *selected;
    ds4_gpu_tensor *scores, *top_a, *top_b;
    ds4_gpu_tensor *kv[N05_LAYERS], *codes[N05_LAYERS], *scales[N05_LAYERS];
    unsigned kv_cap[N05_LAYERS], context, cap, position;
    unsigned *host_positions;
    uint64_t bytes;
    ds4_naive_df_runtime draft;
    bool failed, owns_scratch;
} ds4_naive_graph;

static void naive_graph_free(ds4_naive_graph *g) {
    if (!g) { return; }
    naive_draft_free(&g->draft);
    if (g->owns_scratch) {
        exaone_batch_ws_free(&g->ws);
        ds4_gpu_tensor *all[] = {g->positions, g->tokens, g->freq_full, g->freq_swa,
            g->rope_full, g->rope_swa, g->logits, g->index_q, g->index_k,
            g->index_weights, g->selected, g->scores, g->top_a, g->top_b};
        for (unsigned i = 0; i < sizeof(all) / sizeof(all[0]); i++) { ds4_gpu_tensor_free(all[i]); }
        free(g->host_positions);
    }
    for (unsigned il = 0; il < N05_LAYERS; il++) {
        ds4_gpu_tensor_free(g->kv[il]);
        ds4_gpu_tensor_free(g->codes[il]);
        ds4_gpu_tensor_free(g->scales[il]);
    }
    memset(g, 0, sizeof(*g));
}

static bool naive_graph_alloc(ds4_naive_graph *g, unsigned ctx, unsigned cap) {
    const ds4_context_memory plan = naive_context_memory(ctx, cap);
    if (!g || DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_NAIVE || !plan.total_bytes) { return false; }
    naive_graph_free(g);
    g->owns_scratch = true;
    g->context = ctx; g->cap = cap; g->ws.prefill_cap = cap;
#define N05_ALLOC(field, size) do { \
    const uint64_t bytes = (uint64_t)(size); \
    g->field = ds4_gpu_tensor_alloc(bytes); \
    if (!g->field) { naive_graph_free(g); return false; } \
    g->bytes += bytes; \
} while (0)
#define N05_FLOAT(field, count) N05_ALLOC(field, (uint64_t)(count) * sizeof(float))
    N05_FLOAT(ws.b_cur, (uint64_t)cap * N05_EMBED);
    N05_FLOAT(ws.b_norm, (uint64_t)cap * N05_EMBED);
    N05_FLOAT(ws.b_attn_out, (uint64_t)cap * N05_EMBED);
    N05_FLOAT(ws.b_ffn_out, (uint64_t)cap * N05_EMBED);
    N05_FLOAT(ws.b_q, (uint64_t)cap * N05_HEADS * N05_KEY);
    N05_FLOAT(ws.b_k, (uint64_t)cap * 8 * N05_KEY);
    N05_FLOAT(ws.b_v, (uint64_t)cap * 8 * N05_VALUE);
    N05_FLOAT(ws.b_heads, (uint64_t)cap * N05_HEADS * N05_VALUE);
    N05_FLOAT(ws.b_dense_gate, (uint64_t)cap * N05_DENSE);
    N05_FLOAT(ws.b_dense_up, (uint64_t)cap * N05_DENSE);
    N05_FLOAT(ws.b_dense_mid, (uint64_t)cap * N05_DENSE);
    N05_FLOAT(ws.b_router_logits, (uint64_t)cap * N05_EXPERTS);
    N05_FLOAT(ws.b_router_selected, (uint64_t)cap * N05_USED);
    N05_FLOAT(ws.b_router_weights, (uint64_t)cap * N05_USED);
    N05_FLOAT(ws.b_routed_gate, (uint64_t)cap * N05_USED * N05_FF);
    N05_FLOAT(ws.b_routed_up, (uint64_t)cap * N05_USED * N05_FF);
    N05_FLOAT(ws.b_routed_mid, (uint64_t)cap * N05_USED * N05_FF);
    N05_FLOAT(ws.b_routed_down, (uint64_t)cap * N05_USED * N05_EMBED);
    N05_FLOAT(positions, cap);
    N05_FLOAT(tokens, cap);
    N05_FLOAT(freq_full, N05_ROT / 2);
    N05_FLOAT(freq_swa, N05_ROT / 2);
    N05_FLOAT(rope_full, (uint64_t)cap * N05_ROT);
    N05_FLOAT(rope_swa, (uint64_t)cap * N05_ROT);
    N05_FLOAT(logits, N05_VOCAB);
    N05_FLOAT(index_q, (uint64_t)cap * N05_INDEX_HEADS * N05_INDEX_DIM);
    N05_FLOAT(index_k, (uint64_t)cap * N05_INDEX_DIM);
    N05_FLOAT(index_weights, (uint64_t)cap * N05_INDEX_HEADS);
    N05_FLOAT(selected, (uint64_t)cap * N05_TOP_K);
    const uint64_t queries = cap < N05_QUERY_TILE ? cap : N05_QUERY_TILE;
    const uint64_t tiles = ((uint64_t)ctx + N05_HISTORY_TILE - 1) / N05_HISTORY_TILE;
    const uint64_t keep = ctx < N05_TOP_K ? ctx : N05_TOP_K;
    N05_FLOAT(scores, queries * ctx);
    N05_ALLOC(top_a, queries * tiles * keep * sizeof(uint64_t));
    N05_ALLOC(top_b, queries * tiles * keep * sizeof(uint64_t));
    for (unsigned il = 0; il < N05_LAYERS; il++) {
        const unsigned heads = naive_kv_heads(il);
        g->kv_cap[il] = naive_kv_capacity(il, ctx, cap);
        N05_ALLOC(kv[il], (uint64_t)g->kv_cap[il] * heads * (N05_KEY + N05_VALUE) * sizeof(uint16_t));
        if (!naive_is_dsa(il)) { continue; }
        N05_ALLOC(codes[il], (uint64_t)ctx * N05_INDEX_DIM);
        N05_FLOAT(scales[il], ctx);
    }
#undef N05_FLOAT
#undef N05_ALLOC
    if (g->bytes != plan.total_bytes) { naive_graph_free(g); return false; }
    g->host_positions = xmalloc(cap * sizeof(unsigned));
    float full[N05_ROT / 2], swa[N05_ROT / 2];
    for (unsigned d = 0; d < N05_ROT / 2; d++) {
        full[d] = (float)pow(10000000.0, -2.0 * d / N05_ROT);
        swa[d] = (float)pow(10000.0, -2.0 * d / N05_ROT);
    }
    if (!ds4_gpu_tensor_write(g->freq_full, 0, full, sizeof(full)) ||
        !ds4_gpu_tensor_write(g->freq_swa, 0, swa, sizeof(swa))) { naive_graph_free(g); return false; }
    return true;
}

/* Banks serialize forwards and share scratch. Every KV and indexer buffer
 * is distinct, so one bank's append cannot mutate another bank's prefix. */
static bool naive_graph_clone_bank(ds4_naive_graph *g, const ds4_naive_graph *shared) {
    if (!g || !shared || !shared->owns_scratch || shared->failed) { return false; }
    *g = *shared;
    g->owns_scratch = false; g->position = 0; g->bytes = 0;
    memset(&g->draft, 0, sizeof(g->draft));
    memset(g->kv, 0, sizeof(g->kv));
    memset(g->codes, 0, sizeof(g->codes));
    memset(g->scales, 0, sizeof(g->scales));
    for (unsigned il = 0; il < N05_LAYERS; il++) {
        const uint64_t bytes = (uint64_t)g->kv_cap[il] * naive_kv_heads(il) * (N05_KEY + N05_VALUE) * sizeof(uint16_t);
        g->kv[il] = ds4_gpu_tensor_alloc(bytes);
        if (!g->kv[il]) { naive_graph_free(g); return false; }
        g->bytes += bytes;
        if (!naive_is_dsa(il)) { continue; }
        g->codes[il] = ds4_gpu_tensor_alloc((uint64_t)g->context * N05_INDEX_DIM);
        g->scales[il] = ds4_gpu_tensor_alloc((uint64_t)g->context * sizeof(float));
        if (!g->codes[il] || !g->scales[il]) { naive_graph_free(g); return false; }
        g->bytes += (uint64_t)g->context * (N05_INDEX_DIM + sizeof(float));
    }
    if (shared->draft.ws && !naive_draft_clone(&g->draft, &shared->draft)) { naive_graph_free(g); return false; }
    return true;
}

static ds4_gpu_tensor *naive_view(ds4_gpu_tensor *t, unsigned row, unsigned width, unsigned rows) {
    return ds4_gpu_tensor_view(t, (uint64_t)row * width * sizeof(float), (uint64_t)rows * width * sizeof(float));
}

static bool naive_sparse(ds4_naive_graph *g, const ds4_model *m,
                          const ds4_layer_weights *l, unsigned il, unsigned n) {
    ds4_exaone_batch_ws *b = &g->ws;
    if (!plain_graph_matmul_tensor(g->index_q, m, l->indexer_attn_q_b, N05_EMBED,
            N05_INDEX_HEADS * N05_INDEX_DIM, b->b_norm, n) ||
        !plain_graph_matmul_tensor(g->index_k, m, l->indexer_attn_k, N05_EMBED, N05_INDEX_DIM, b->b_norm, n) ||
        !plain_graph_matmul_tensor(g->index_weights, m, l->indexer_proj, N05_EMBED, N05_INDEX_HEADS, b->b_norm, n) ||
        !ds4_gpu_naive_key_norm(g->index_k, m->map, m->size,
            l->indexer_k_norm->abs_offset, l->indexer_k_norm_b->abs_offset, n) ||
        !ds4_gpu_naive_rope(g->index_q, g->rope_full, N05_INDEX_HEADS, N05_INDEX_DIM, n) ||
        !ds4_gpu_naive_rope(g->index_k, g->rope_full, 1, N05_INDEX_DIM, n) ||
        !ds4_gpu_naive_round(g->index_weights, (uint64_t)n * N05_INDEX_HEADS) ||
        !ds4_gpu_dots3_scale_tensor(g->index_weights, .25f, (uint64_t)n * N05_INDEX_HEADS) ||
        !ds4_gpu_naive_round(g->index_weights, (uint64_t)n * N05_INDEX_HEADS) ||
        !ds4_gpu_naive_index_store(g->codes[il], g->scales[il], g->index_q, g->index_k,
            g->positions, n, g->context)) { return false; }
    for (unsigned row = 0; row < n; row += N05_QUERY_TILE) {
        const unsigned count = n - row < N05_QUERY_TILE ? n - row : N05_QUERY_TILE;
        ds4_gpu_tensor *v[] = {
            naive_view(g->index_q, row, N05_INDEX_HEADS * N05_INDEX_DIM, count),
            naive_view(g->index_weights, row, N05_INDEX_HEADS, count),
            naive_view(g->positions, row, 1, count),
            naive_view(g->selected, row, N05_TOP_K, count),
            naive_view(b->b_q, row, N05_HEADS * N05_KEY, count),
            naive_view(b->b_heads, row, N05_HEADS * N05_VALUE, count),
        };
        bool ok = true;
        for (unsigned i = 0; i < sizeof(v) / sizeof(v[0]); i++) { ok = ok && v[i]; }
        ok = ok && ds4_gpu_naive_select(v[3], g->scores, g->top_a, g->top_b, v[0],
            g->codes[il], g->scales[il], v[1], v[2], g->position + row + count, count) &&
            ds4_gpu_naive_attention(v[5], v[4], g->kv[il], v[2], v[3], NULL, 0, 0, 4, g->kv_cap[il], count, 0);
        for (unsigned i = 0; i < sizeof(v) / sizeof(v[0]); i++) { ds4_gpu_tensor_free(v[i]); }
        if (!ok) { return false; }
    }
    return true;
}

static bool naive_layer(ds4_naive_graph *g, const ds4_model *m,
                         const ds4_weights *w, unsigned il, unsigned n) {
    ds4_exaone_batch_ws *b = &g->ws;
    const ds4_layer_weights *l = &w->layer[il];
    const unsigned heads = naive_kv_heads(il);
    const ds4_gpu_tensor *rope = naive_is_dsa(il) ? g->rope_full : g->rope_swa;
    if (!ds4_gpu_naive_rms(b->b_norm, b->b_cur, m->map, m->size, l->attn_norm->abs_offset, n) ||
        !plain_graph_matmul_tensor(b->b_q, m, l->attn_q, N05_EMBED, N05_HEADS * N05_KEY, b->b_norm, n) ||
        !plain_graph_matmul_tensor(b->b_k, m, l->attn_k, N05_EMBED, heads * N05_KEY, b->b_norm, n) ||
        !plain_graph_matmul_tensor(b->b_v, m, l->attn_v, N05_EMBED, heads * N05_VALUE, b->b_norm, n) ||
        !ds4_gpu_naive_rope(b->b_q, rope, N05_HEADS, N05_KEY, n) ||
        !ds4_gpu_naive_rope(b->b_k, rope, heads, N05_KEY, n) ||
        !ds4_gpu_naive_kv_store(g->kv[il], b->b_k, b->b_v, g->positions, heads, n, g->kv_cap[il])) { return false; }
    if (naive_is_dsa(il)) {
        if (!naive_sparse(g, m, l, il, n)) { return false; }
    } else if (!ds4_gpu_naive_attention(b->b_heads, b->b_q, g->kv[il], g->positions, NULL,
        m->map, m->size, l->attn_sinks->abs_offset, heads, g->kv_cap[il], n, N05_WINDOW)) { return false; }
    if (!plain_graph_matmul_tensor(b->b_attn_out, m, l->attn_output, N05_HEADS * N05_VALUE, N05_EMBED, b->b_heads, n) ||
        !ds4_gpu_naive_add(b->b_cur, b->b_attn_out, (uint64_t)n * N05_EMBED) ||
        !ds4_gpu_naive_rms(b->b_norm, b->b_cur, m->map, m->size, l->ffn_norm->abs_offset, n)) { return false; }
    if (il == 0) {
        if (!plain_graph_matmul_tensor(b->b_dense_gate, m, l->ffn_gate, N05_EMBED, N05_DENSE, b->b_norm, n) ||
            !plain_graph_matmul_tensor(b->b_dense_up, m, l->ffn_up, N05_EMBED, N05_DENSE, b->b_norm, n) ||
            !ds4_gpu_naive_swiglu(b->b_dense_mid, b->b_dense_gate, b->b_dense_up, (uint64_t)n * N05_DENSE) ||
            !plain_graph_matmul_tensor(b->b_ffn_out, m, l->ffn_down, N05_DENSE, N05_EMBED, b->b_dense_mid, n)) { return false; }
    } else {
        if (!plain_graph_matmul_tensor(b->b_router_logits, m, l->ffn_gate_inp, N05_EMBED, N05_EXPERTS, b->b_norm, n) ||
            !ds4_gpu_naive_router(b->b_router_selected, b->b_router_weights, b->b_router_logits,
                m->map, m->size, l->ffn_exp_probs_b->abs_offset, n) ||
            !ds4_gpu_routed_gate_up_tensor(b->b_routed_gate, b->b_routed_up, b->b_norm, b->b_router_selected,
                m->map, m->size, l->ffn_gate_exps->abs_offset, l->ffn_gate_exps->bytes,
                l->ffn_up_exps->abs_offset, l->ffn_up_exps->bytes, l->ffn_gate_exps->type,
                N05_EMBED, N05_FF, N05_EXPERTS, n, N05_USED) ||
            !ds4_gpu_naive_swiglu(b->b_routed_mid, b->b_routed_gate, b->b_routed_up, (uint64_t)n * N05_USED * N05_FF) ||
            !ds4_gpu_routed_matmul_guarded_tensor(b->b_routed_down, b->b_routed_mid, b->b_router_selected,
                m->map, m->size, l->ffn_down_exps->abs_offset, l->ffn_down_exps->bytes,
                l->ffn_down_exps->type, N05_FF, N05_EMBED, N05_EXPERTS, n * N05_USED, 1, n) ||
            !ds4_gpu_naive_sum(b->b_ffn_out, b->b_routed_down, b->b_router_weights, n)) { return false; }
    }
    return ds4_gpu_naive_add(b->b_cur, b->b_ffn_out, (uint64_t)n * N05_EMBED);
}

static bool naive_head(ds4_naive_graph *g, const ds4_model *m, const ds4_weights *w, unsigned row) {
    ds4_gpu_tensor *last = naive_view(g->ws.b_cur, row, N05_EMBED, 1);
    const bool ok = last && ds4_gpu_naive_rms(g->ws.b_norm, last, m->map, m->size, w->output_norm->abs_offset, 1) &&
        plain_graph_matmul_tensor(g->logits, m, w->output, N05_EMBED, N05_VOCAB, g->ws.b_norm, 1) &&
        ds4_gpu_naive_round(g->logits, N05_VOCAB);
    ds4_gpu_tensor_free(last);
    return ok;
}

static bool naive_forward(ds4_naive_graph *g, const ds4_model *m, const ds4_weights *w,
                           const int *tokens, unsigned n, unsigned pos) {
    if (!g || !tokens || g->failed || !n || n > g->cap || pos != g->position ||
        pos > g->context || n > g->context - pos) { return false; }
    for (unsigned i = 0; i < n; i++) {
        if (tokens[i] < 0 || (unsigned)tokens[i] >= N05_VOCAB) { return false; }
        g->host_positions[i] = pos + i;
    }
    // Any partial failure invalidates the entire KV/indexer frontier.
    g->failed = true;
    if (!ds4_gpu_tensor_write(g->tokens, 0, tokens, n * sizeof(int)) ||
        !ds4_gpu_tensor_write(g->positions, 0, g->host_positions, n * sizeof(unsigned)) ||
        !ds4_gpu_step37_rope(g->rope_full, g->freq_full, g->positions, N05_ROT, n) ||
        !ds4_gpu_step37_rope(g->rope_swa, g->freq_swa, g->positions, N05_ROT, n) ||
        !ds4_gpu_embed_tokens_quant_tensor(g->ws.b_cur, g->tokens, m->map, m->size,
            w->token_embd->abs_offset, w->token_embd->type, N05_VOCAB, n, N05_EMBED) ||
        !ds4_gpu_naive_round(g->ws.b_cur, (uint64_t)n * N05_EMBED)) { return false; }
    for (unsigned il = 0; il < N05_LAYERS; il++) {
        if (!naive_layer(g, m, w, il, n)) { return false; }
        const int tap = naive_tap(il);
        const unsigned keep = n < N05_DF_WINDOW ? n : N05_DF_WINDOW;
        if (g->draft.ws && tap >= 0 && !ds4_gpu_naive_df_tap(g->draft.ws->tap, g->ws.b_cur,
                n - keep, keep, (unsigned)tap)) { return false; }
    }
    if (!naive_head(g, m, w, n - 1) || !naive_draft_prepare(&g->draft, pos, n)) { return false; }
    g->position = pos + n; g->failed = false;
    return true;
}
