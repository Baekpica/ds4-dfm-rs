/* Eager IQuest-Q1 graph. All GGUF projections use the existing mmap/MMQ
 * boundary. Target KV is Q8_0; SWA capacities include the current chunk
 * so appending a chunk cannot overwrite history needed by its first row. */
typedef struct {
    ds4_exaone_batch_ws ws;
    ds4_gpu_tensor *positions, *tokens, *logits, *mtp_logits;
    ds4_gpu_tensor *residual, *hidden, *saved_hidden, *draft_input;
    ds4_gpu_tensor *carry, *draft_carry, *previous;
    ds4_gpu_tensor *kv[IQ_LAYERS], *mtp_kv;
    ds4_gpu_tensor *trial_kv, *trial_mtp, *mtp_backup, *trial_hidden;
    float *trial_logits, *draft_logits;
    int trial_tokens[IQ_DRAFT_SLOTS + 1];
    unsigned trial_n, trial_base;
    unsigned kv_cap[IQ_LAYERS], context, cap, position, mtp_position;
    unsigned *host_positions;
    uint64_t bytes;
    bool failed, owns_scratch, mtp_enabled;
} ds4_iquest_graph;

static void iquest_graph_free(ds4_iquest_graph *g) {
    if (!g) { return; }
    if (g->owns_scratch) {
        exaone_batch_ws_free(&g->ws);
        ds4_gpu_tensor *all[] = {g->positions, g->tokens, g->logits, g->mtp_logits,
            g->residual, g->hidden, g->saved_hidden, g->draft_input,
            g->draft_carry, g->previous, g->trial_kv,
            g->trial_mtp, g->mtp_backup, g->trial_hidden};
        for (unsigned i = 0; i < sizeof(all) / sizeof(all[0]); i++) { ds4_gpu_tensor_free(all[i]); }
        free(g->host_positions);
        free(g->trial_logits); free(g->draft_logits);
    }
    for (unsigned il = 0; il < IQ_LAYERS; il++) { ds4_gpu_tensor_free(g->kv[il]); }
    ds4_gpu_tensor_free(g->mtp_kv);
    ds4_gpu_tensor_free(g->carry);
    memset(g, 0, sizeof(*g));
}

static bool iquest_graph_alloc(ds4_iquest_graph *g, unsigned ctx, unsigned cap,
                               iquest_mtp_mode mtp) {
    const ds4_context_memory plan = iquest_memory(ctx, cap);
    if (!g || DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_IQUEST || !plan.total_bytes) { return false; }
    iquest_graph_free(g);
    g->owns_scratch = true; g->mtp_enabled = mtp == IQ_MTP_ON;
    g->context = ctx; g->cap = cap; g->ws.prefill_cap = cap;
#define IQ_ALLOC(field, size) do { \
    const uint64_t bytes = (uint64_t)(size); \
    g->field = ds4_gpu_tensor_alloc(bytes); \
    if (!g->field) { iquest_graph_free(g); return false; } \
    g->bytes += bytes; \
} while (0)
#define IQ_FLOAT(field, count) IQ_ALLOC(field, (uint64_t)(count) * sizeof(float))
    IQ_FLOAT(ws.b_cur, (uint64_t)cap * IQ_EMBED);
    IQ_FLOAT(ws.b_norm, (uint64_t)cap * IQ_EMBED);
    IQ_FLOAT(ws.b_attn_out, (uint64_t)cap * IQ_EMBED);
    IQ_FLOAT(ws.b_ffn_out, (uint64_t)cap * IQ_EMBED);
    IQ_FLOAT(residual, (uint64_t)cap * IQ_EMBED);
    IQ_FLOAT(ws.b_q, (uint64_t)cap * IQ_HEADS * IQ_HEAD);
    IQ_FLOAT(ws.b_k, (uint64_t)cap * IQ_KV_HEADS * IQ_HEAD);
    IQ_FLOAT(ws.b_v, (uint64_t)cap * IQ_KV_HEADS * IQ_HEAD);
    IQ_FLOAT(ws.b_heads, (uint64_t)cap * IQ_HEADS * IQ_HEAD);
    IQ_FLOAT(ws.b_dense_gate, (uint64_t)cap * IQ_DENSE);
    IQ_FLOAT(ws.b_dense_up, (uint64_t)cap * IQ_DENSE);
    IQ_FLOAT(ws.b_dense_mid, (uint64_t)cap * IQ_DENSE);
    IQ_FLOAT(ws.b_router_logits, (uint64_t)cap * IQ_EXPERTS);
    IQ_FLOAT(ws.b_router_selected, (uint64_t)cap * IQ_USED);
    IQ_FLOAT(ws.b_router_weights, (uint64_t)cap * IQ_USED);
    IQ_FLOAT(ws.b_routed_gate, (uint64_t)cap * IQ_USED * IQ_FF);
    IQ_FLOAT(ws.b_routed_up, (uint64_t)cap * IQ_USED * IQ_FF);
    IQ_FLOAT(ws.b_routed_mid, (uint64_t)cap * IQ_USED * IQ_FF);
    IQ_FLOAT(ws.b_routed_down, (uint64_t)cap * IQ_USED * IQ_EMBED);
    IQ_FLOAT(positions, cap); IQ_FLOAT(tokens, cap);
    IQ_FLOAT(hidden, (uint64_t)cap * IQ_EMBED);
    IQ_FLOAT(saved_hidden, (uint64_t)cap * IQ_EMBED);
    IQ_FLOAT(draft_input, (uint64_t)cap * 2 * IQ_EMBED);
    IQ_FLOAT(carry, IQ_EMBED); IQ_FLOAT(draft_carry, IQ_EMBED); IQ_FLOAT(previous, IQ_EMBED);
    IQ_FLOAT(logits, IQ_VOCAB); IQ_FLOAT(mtp_logits, IQ_VOCAB);
    for (unsigned il = 0; il < IQ_LAYERS; il++) {
        g->kv_cap[il] = iquest_kv_cap(il, ctx, cap);
        IQ_ALLOC(kv[il], (uint64_t)g->kv_cap[il] * IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8));
    }
    IQ_ALLOC(mtp_kv, (uint64_t)(IQ_DRAFT_WINDOW + IQ_DRAFT_SLOTS) * IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8));
    IQ_ALLOC(mtp_backup, (uint64_t)(IQ_DRAFT_WINDOW + IQ_DRAFT_SLOTS) * IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8));
    IQ_ALLOC(trial_kv, (uint64_t)(IQ_DRAFT_SLOTS + 1) * IQ_LAYERS * IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8));
    IQ_ALLOC(trial_mtp, (uint64_t)(IQ_DRAFT_SLOTS + 1) * IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8));
    IQ_FLOAT(trial_hidden, (uint64_t)(IQ_DRAFT_SLOTS + 1) * IQ_EMBED);
#undef IQ_FLOAT
#undef IQ_ALLOC
    if (g->bytes != plan.total_bytes) { iquest_graph_free(g); return false; }
    g->host_positions = xmalloc(cap * sizeof(unsigned));
    g->trial_logits = xmalloc((size_t)(IQ_DRAFT_SLOTS + 1) * IQ_VOCAB * sizeof(float));
    g->draft_logits = xmalloc(IQ_VOCAB * sizeof(float));
    return true;
}

/* Banks serialize on shared projection scratch. Their target histories,
 * MTP rings and final target hidden rows have independent ownership. */
static bool iquest_graph_clone_bank(ds4_iquest_graph *g, const ds4_iquest_graph *shared) {
    if (!g || !shared || !shared->owns_scratch || shared->failed) { return false; }
    *g = *shared; g->owns_scratch = false; g->position = 0; g->mtp_position = 0;
    g->trial_n = 0; g->bytes = 0;
    memset(g->kv, 0, sizeof(g->kv)); g->mtp_kv = NULL; g->carry = NULL;
    for (unsigned il = 0; il < IQ_LAYERS; il++) {
        const uint64_t bytes = (uint64_t)g->kv_cap[il] * IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8);
        g->kv[il] = ds4_gpu_tensor_alloc(bytes);
        if (!g->kv[il]) { iquest_graph_free(g); return false; }
        g->bytes += bytes;
    }
    const uint64_t mtp = (uint64_t)(IQ_DRAFT_WINDOW + IQ_DRAFT_SLOTS) * IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8);
    g->mtp_kv = ds4_gpu_tensor_alloc(mtp);
    g->carry = ds4_gpu_tensor_alloc(IQ_EMBED * sizeof(float));
    if (!g->mtp_kv || !g->carry) { iquest_graph_free(g); return false; }
    g->bytes += mtp + IQ_EMBED * sizeof(float);
    return true;
}

static ds4_gpu_tensor *iquest_view(ds4_gpu_tensor *t, unsigned row, unsigned width, unsigned n) {
    return ds4_gpu_tensor_view(t, (uint64_t)row * width * sizeof(float), (uint64_t)n * width * sizeof(float));
}

static bool iquest_round(ds4_gpu_tensor *t, uint64_t count) {
    return ds4_gpu_motif3_round_bf16_tensor(t, t, count) != 0;
}

static bool iquest_linear(ds4_gpu_tensor *out, const ds4_model *m,
        const ds4_tensor *w, unsigned in, unsigned width,
        const ds4_gpu_tensor *x, ds4_gpu_tensor *ids, unsigned n, iquest_round_mode rounded) {
    if (!w) { return false; }
    bool ok;
    if (w->type == DS4_TENSOR_Q5_K || w->type == DS4_TENSOR_Q6_K ||
        w->type == DS4_TENSOR_IQ2_XXS || w->type == DS4_TENSOR_IQ2_XS) {
        ok = ds4_gpu_tensor_fill_f32(ids, 0, n) &&
            ds4_gpu_routed_matmul_guarded_tensor(out, x, ids, m->map, m->size,
                w->abs_offset, w->bytes, w->type, in, width, 1, n, 1, n);
    } else { ok = plain_graph_matmul_tensor(out, m, w, in, width, x, n); }
    return ok && (rounded == IQ_KEEP_F32 || iquest_round(out, (uint64_t)n * width));
}

static bool iquest_norm(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_model *m, const ds4_tensor *w, unsigned width, unsigned rows,
        iquest_round_mode rounded) {
    return w && ds4_gpu_iquest_rms(out, x, m->map, m->size, w->abs_offset, width, rows) &&
        (rounded == IQ_KEEP_F32 || iquest_round(out, (uint64_t)width * rows));
}

static bool iquest_expert_proj(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *ids, const ds4_model *m, const ds4_tensor *w,
        unsigned in, unsigned width, unsigned n, unsigned input_used) {
    if (!w) { return false; }
    if (w->type == DS4_TENSOR_F32 || w->type == DS4_TENSOR_F16 || w->type == DS4_TENSOR_BF16) {
        return ds4_gpu_iquest_expert(out, x, ids, m->map, m->size, w->abs_offset,
            w->type, in, width, n, IQ_USED, input_used) != 0;
    }
    return ds4_gpu_routed_matmul_guarded_tensor(out, x, ids, m->map, m->size,
        w->abs_offset, w->bytes, w->type, in, width, IQ_EXPERTS,
        n * input_used, input_used == IQ_USED ? 1 : IQ_USED, n) != 0;
}

static bool iquest_experts(ds4_exaone_batch_ws *b, const ds4_model *m,
                            const ds4_layer_weights *l, unsigned n) {
    if (!l->ffn_gate_exps || !l->ffn_up_exps || !l->ffn_down_exps) { return false; }
    const unsigned type = l->ffn_gate_exps->type;
    bool ok;
    if (type == l->ffn_up_exps->type && type != DS4_TENSOR_F32 &&
        type != DS4_TENSOR_F16 && type != DS4_TENSOR_BF16) {
        ok = ds4_gpu_routed_gate_up_tensor(b->b_routed_gate, b->b_routed_up, b->b_norm,
            b->b_router_selected, m->map, m->size,
            l->ffn_gate_exps->abs_offset, l->ffn_gate_exps->bytes,
            l->ffn_up_exps->abs_offset, l->ffn_up_exps->bytes, type,
            IQ_EMBED, IQ_FF, IQ_EXPERTS, n, IQ_USED);
    } else {
        ok = iquest_expert_proj(b->b_routed_gate, b->b_norm, b->b_router_selected,
                m, l->ffn_gate_exps, IQ_EMBED, IQ_FF, n, 1) &&
            iquest_expert_proj(b->b_routed_up, b->b_norm, b->b_router_selected,
                m, l->ffn_up_exps, IQ_EMBED, IQ_FF, n, 1);
    }
    ok = ok && ds4_gpu_iquest_swiglu(b->b_routed_mid, b->b_routed_gate, b->b_routed_up,
        (uint64_t)n * IQ_USED * IQ_FF);
    ok = ok && iquest_expert_proj(b->b_routed_down, b->b_routed_mid, b->b_router_selected,
        m, l->ffn_down_exps, IQ_FF, IQ_EMBED, n, IQ_USED);
    /* FusedMoE accumulates weighted expert outputs in FP32 and casts the
     * combined branch back to BF16 before the output norm/residual. */
    return ok && ds4_gpu_iquest_sum(b->b_ffn_out, b->b_routed_down, b->b_router_weights, n) &&
        iquest_round(b->b_ffn_out, (uint64_t)n * IQ_EMBED);
}

static bool iquest_layer(ds4_iquest_graph *g, const ds4_model *m,
        const ds4_layer_weights *l, ds4_gpu_tensor *cache, unsigned capacity,
        unsigned n, unsigned window, iquest_layer_kind kind) {
    const bool first = kind != IQ_MOE_LAYER;
    const bool dense = kind == IQ_DENSE_LAYER;
    const bool fp32_residual = kind == IQ_DRAFT_LAYER;
    ds4_exaone_batch_ws *b = &g->ws;
    const uint64_t hidden = (uint64_t)n * IQ_EMBED;
    if (!iquest_norm(b->b_norm, b->b_cur, m, l->attn_norm, IQ_EMBED, n, IQ_ROUND_BF16) ||
        !ds4_gpu_tensor_copy(g->residual, 0, first ? b->b_cur : b->b_norm, 0, hidden * sizeof(float)) ||
        !iquest_linear(b->b_q, m, l->attn_q, IQ_EMBED, IQ_HEADS * IQ_HEAD,
            b->b_norm, b->b_router_selected, n, IQ_ROUND_BF16) ||
        !iquest_linear(b->b_k, m, l->attn_k, IQ_EMBED, IQ_KV_HEADS * IQ_HEAD,
            b->b_norm, b->b_router_selected, n, IQ_ROUND_BF16) ||
        !iquest_linear(b->b_v, m, l->attn_v, IQ_EMBED, IQ_KV_HEADS * IQ_HEAD,
            b->b_norm, b->b_router_selected, n, IQ_ROUND_BF16) ||
        !iquest_norm(b->b_q, b->b_q, m, l->attn_q_norm, IQ_HEAD, n * IQ_HEADS, IQ_ROUND_BF16) ||
        !iquest_norm(b->b_k, b->b_k, m, l->attn_k_norm, IQ_HEAD, n * IQ_KV_HEADS, IQ_ROUND_BF16) ||
        !ds4_gpu_iquest_rope(b->b_q, g->positions, IQ_HEADS, n, window ? 10000.0f : 1000000.0f) ||
        !ds4_gpu_iquest_rope(b->b_k, g->positions, IQ_KV_HEADS, n, window ? 10000.0f : 1000000.0f) ||
        !iquest_round(b->b_q, (uint64_t)n * IQ_HEADS * IQ_HEAD) ||
        !iquest_round(b->b_k, (uint64_t)n * IQ_KV_HEADS * IQ_HEAD) ||
        !ds4_gpu_iquest_kv(cache, b->b_k, b->b_v, g->positions, n, capacity) ||
        !ds4_gpu_iquest_attn(b->b_heads, b->b_q, cache, g->positions,
            m->map, m->size, l->attn_sink_k->abs_offset, n, capacity, window) ||
        !iquest_round(b->b_heads, (uint64_t)n * IQ_HEADS * IQ_HEAD) ||
        !iquest_linear(b->b_attn_out, m, l->attn_output, IQ_HEADS * IQ_HEAD, IQ_EMBED,
            b->b_heads, b->b_router_selected, n, IQ_ROUND_BF16) ||
        !iquest_norm(b->b_attn_out, b->b_attn_out, m, l->attn_output_norm, IQ_EMBED, n, IQ_ROUND_BF16) ||
        !ds4_gpu_iquest_add(b->b_cur, g->residual, b->b_attn_out, n, 1.0f) ||
        (!fp32_residual && !iquest_round(b->b_cur, hidden)) ||
        !iquest_norm(b->b_norm, b->b_cur, m, l->ffn_norm, IQ_EMBED, n, IQ_ROUND_BF16)) { return false; }
    if (dense) {
        if (!iquest_linear(b->b_dense_gate, m, l->ffn_gate, IQ_EMBED, IQ_DENSE,
                b->b_norm, b->b_router_selected, n, IQ_ROUND_BF16) ||
            !iquest_linear(b->b_dense_up, m, l->ffn_up, IQ_EMBED, IQ_DENSE,
                b->b_norm, b->b_router_selected, n, IQ_ROUND_BF16) ||
            !ds4_gpu_iquest_swiglu(b->b_dense_mid, b->b_dense_gate, b->b_dense_up, (uint64_t)n * IQ_DENSE) ||
            !iquest_linear(b->b_ffn_out, m, l->ffn_down, IQ_DENSE, IQ_EMBED,
                b->b_dense_mid, b->b_router_selected, n, IQ_ROUND_BF16)) { return false; }
    } else {
        if (!iquest_linear(b->b_router_logits, m, l->ffn_gate_inp, IQ_EMBED, IQ_EXPERTS,
                b->b_norm, b->b_router_selected, n, IQ_KEEP_F32) ||
            !ds4_gpu_iquest_router(b->b_router_selected, b->b_router_weights, b->b_router_logits, n) ||
            !iquest_experts(b, m, l, n)) { return false; }
    }
    if (first) {
        if (!iquest_norm(b->b_ffn_out, b->b_ffn_out, m, l->ffn_output_norm, IQ_EMBED, n,
                         IQ_ROUND_BF16)) { return false; }
    } else if (!ds4_gpu_dots3_scale_tensor(b->b_ffn_out, 0.5388159060803247f, hidden) ||
               !iquest_round(b->b_ffn_out, hidden)) { return false; }
    return ds4_gpu_iquest_add(b->b_cur, b->b_cur, b->b_ffn_out, n, 1.0f) &&
        (fp32_residual || iquest_round(b->b_cur, hidden));
}

static bool iquest_mtp_forward(ds4_iquest_graph *g, const ds4_model *m,
        const ds4_weights *w, const int *tokens, const ds4_gpu_tensor *hidden,
        unsigned n, unsigned pos, iquest_logits_mode logits) {
    if (!n || n > g->cap || n > IQ_DRAFT_SLOTS || pos != g->mtp_position) { return false; }
    for (unsigned row = 0; row < n; row++) { g->host_positions[row] = pos + row; }
    ds4_exaone_batch_ws *b = &g->ws;
    if (!ds4_gpu_tensor_write(g->tokens, 0, tokens, n * sizeof(int)) ||
        !ds4_gpu_tensor_write(g->positions, 0, g->host_positions, n * sizeof(unsigned)) ||
        !ds4_gpu_embed_tokens_quant_tensor(b->b_cur, g->tokens, m->map, m->size,
            w->token_embd->abs_offset, w->token_embd->type, IQ_VOCAB, n, IQ_EMBED) ||
        !iquest_round(b->b_cur, (uint64_t)n * IQ_EMBED) ||
        !iquest_norm(b->b_norm, b->b_cur, m, w->iquest_enorm, IQ_EMBED, n, IQ_ROUND_BF16) ||
        !iquest_norm(b->b_attn_out, hidden, m, w->iquest_hnorm, IQ_EMBED, n, IQ_ROUND_BF16) ||
        !ds4_gpu_concat_rows_tensor(g->draft_input, b->b_norm, b->b_attn_out, n, IQ_EMBED) ||
        !iquest_linear(b->b_cur, m, w->iquest_eh, 2 * IQ_EMBED, IQ_EMBED,
            g->draft_input, b->b_router_selected, n, IQ_ROUND_BF16) ||
        !iquest_layer(g, m, &w->iquest_mtp, g->mtp_kv, IQ_DRAFT_WINDOW + IQ_DRAFT_SLOTS,
            n, IQ_DRAFT_WINDOW, IQ_DRAFT_LAYER) ||
        !iquest_norm(g->saved_hidden, b->b_cur, m, w->iquest_mtp_norm, IQ_EMBED, n, IQ_ROUND_BF16) ||
        !ds4_gpu_tensor_copy(g->draft_carry, 0, g->saved_hidden,
            (uint64_t)(n - 1) * IQ_EMBED * sizeof(float), IQ_EMBED * sizeof(float))) { return false; }
    if (logits == IQ_HAS_LOGITS) {
        ds4_gpu_tensor *last = iquest_view(g->saved_hidden, n - 1, IQ_EMBED, 1);
        const bool ok = last && iquest_linear(g->mtp_logits, m, w->output,
            IQ_EMBED, IQ_VOCAB, last, b->b_router_selected, 1, IQ_ROUND_BF16);
        ds4_gpu_tensor_free(last);
        if (!ok) { return false; }
    }
    g->mtp_position = pos + n;
    return true;
}

static bool iquest_forward(ds4_iquest_graph *g, const ds4_model *m,
        const ds4_weights *w, const int *tokens, unsigned n, unsigned pos,
        iquest_mtp_mode warm_mtp) {
    if (!g || !tokens || g->failed || !n || n > g->cap || pos != g->position ||
        pos > g->context || n > g->context - pos) { return false; }
    for (unsigned row = 0; row < n; row++) {
        if (tokens[row] < 0 || (unsigned)tokens[row] >= IQ_VOCAB) { return false; }
        g->host_positions[row] = pos + row;
    }
    g->failed = true;
    if (!ds4_gpu_tensor_write(g->tokens, 0, tokens, n * sizeof(int)) ||
        !ds4_gpu_tensor_write(g->positions, 0, g->host_positions, n * sizeof(unsigned)) ||
        !ds4_gpu_embed_tokens_quant_tensor(g->ws.b_cur, g->tokens, m->map, m->size,
            w->token_embd->abs_offset, w->token_embd->type, IQ_VOCAB, n, IQ_EMBED) ||
        !iquest_round(g->ws.b_cur, (uint64_t)n * IQ_EMBED)) { return false; }
    for (unsigned il = 0; il < IQ_LAYERS; il++) {
        if (!iquest_layer(g, m, &w->layer[il], g->kv[il], g->kv_cap[il], n,
                iquest_full(il) ? 0 : IQ_WINDOW,
                il == 0 ? IQ_DENSE_LAYER : IQ_MOE_LAYER)) { return false; }
    }
    if (!iquest_norm(g->hidden, g->ws.b_cur, m, w->output_norm, IQ_EMBED, n, IQ_ROUND_BF16)) { return false; }
    ds4_gpu_tensor *last = iquest_view(g->hidden, n - 1, IQ_EMBED, 1);
    bool ok = last && iquest_linear(g->logits, m, w->output, IQ_EMBED, IQ_VOCAB,
        last, g->ws.b_router_selected, 1, IQ_ROUND_BF16);
    ds4_gpu_tensor_free(last);
    if (!ok) { return false; }
    if (warm_mtp == IQ_MTP_ON) {
        /* Teacher pairs are embed(token[i+1]), target_hidden[i], position i.
         * Preserve only one carry across chunks; never retain whole history. */
        unsigned begin = 0;
        if (pos) {
            if (g->mtp_position != pos - 1 || !iquest_mtp_forward(g, m, w, tokens,
                    g->carry, 1, pos - 1, IQ_NO_LOGITS)) { return false; }
        }
        while (begin + 1 < n) {
            const unsigned remain = n - begin - 1;
            unsigned count = remain < IQ_DRAFT_SLOTS ? remain : IQ_DRAFT_SLOTS;
            if (count > g->cap) { count = g->cap; }
            ds4_gpu_tensor *teacher = iquest_view(g->hidden, begin, IQ_EMBED, count);
            ok = teacher && iquest_mtp_forward(g, m, w, tokens + begin + 1,
                teacher, count, pos + begin, IQ_NO_LOGITS);
            ds4_gpu_tensor_free(teacher);
            if (!ok) { return false; }
            begin += count;
        }
    }
    if (!ds4_gpu_tensor_copy(g->carry, 0, g->hidden,
            (uint64_t)(n - 1) * IQ_EMBED * sizeof(float), IQ_EMBED * sizeof(float))) { return false; }
    g->position = pos + n; g->failed = false;
    return true;
}
