/* MiMo owns asymmetric KV and layer scheduling. Reuse the existing mmap
 * matmul/MMQ paths, without borrowing another family's attention semantics. */
/* Span ceiling matches the 64 MiB raw decode cap. The smallest legal
 * frame is 32x256. Each pair can carry a video span and an audio span,
 * and a request may hold four clips. 8 spans was about 8 seconds. */
enum {
    M2_TRIAL = 8,
    M2_VIDEO_RAW_CAP = 64 * 1024 * 1024,
    M2_MIN_FRAME_BYTES = 32 * 256 * 3,
    M2_MEDIA_INPUTS = 4,
    M2_SPAN_FRAMES = M2_VIDEO_RAW_CAP / M2_MIN_FRAME_BYTES,
    M2_SPAN_PAIRS = (M2_SPAN_FRAMES + 1) / 2,
    M2_SPAN_MAX = M2_SPAN_PAIRS * 2 * M2_MEDIA_INPUTS
};

typedef struct {
    unsigned start, count;
    const float *rows;
} mimo2_media_span;

typedef struct {
    ds4_exaone_batch_ws ws;
    ds4_gpu_tensor *qkv, *positions, *tokens, *freq_full, *freq_swa;
    ds4_gpu_tensor *rope_full, *rope_swa, *logits;
    ds4_gpu_tensor *hidden, *joined, *journal;
    ds4_gpu_tensor *kv[MIMO2_LAYERS + MIMO2_DRAFT_LAYERS];
    unsigned kv_cap[MIMO2_LAYERS + MIMO2_DRAFT_LAYERS];
    unsigned draft_pos[MIMO2_DRAFT_LAYERS];
    unsigned draft_saved[MIMO2_DRAFT_LAYERS];
    unsigned context, cap, position;
    unsigned *host_positions;
    float *verify_logits;
    int trial[M2_TRIAL];
    unsigned trial_n, verify_rows;
    uint64_t bytes, media_tag, checkpoint_tag;
    bool failed, hidden_valid;
    /* Grown up to M2_SPAN_MAX on sync. Not part of the KV snapshot. */
    mimo2_media_span *spans;
    unsigned span_cap, span_n;
    /* Last 1024 positions, five target layers, when DFlash is loaded. */
    ds4_gpu_tensor *dflash_tap;
    unsigned *dflash_pos;
    /* Trial rows about to be overwritten. Rejected slots are copied back. */
    float *dflash_saved;
    unsigned dflash_saved_pos[M2_TRIAL];
    unsigned dflash_saved_n;
    bool dflash_on;
} ds4_mimo2_graph;

static void mimo2_graph_free(ds4_mimo2_graph *g) {
    exaone_batch_ws_free(&g->ws);
    ds4_gpu_tensor *all[] = {g->qkv, g->positions, g->tokens, g->freq_full,
                            g->freq_swa, g->rope_full, g->rope_swa, g->logits,
                            g->hidden, g->joined, g->journal};
    for (unsigned i = 0; i < sizeof(all) / sizeof(all[0]); i++) { ds4_gpu_tensor_free(all[i]); }
    for (unsigned il = 0; il < MIMO2_LAYERS + MIMO2_DRAFT_LAYERS; il++) { ds4_gpu_tensor_free(g->kv[il]); }
    free(g->host_positions);
    free(g->verify_logits);
    free(g->spans);
    free(g->dflash_pos);
    free(g->dflash_saved);
    ds4_gpu_tensor_free(g->dflash_tap);
    memset(g, 0, sizeof(*g));
}

static bool mimo2_graph_alloc(ds4_mimo2_graph *g, unsigned ctx, unsigned cap, bool dflash) {
    free(g->spans);
    free(g->dflash_pos);
    free(g->dflash_saved);
    ds4_gpu_tensor_free(g->dflash_tap);
    memset(g, 0, sizeof(*g));
    g->dflash_on = dflash;
    const ds4_context_memory estimate = mimo2_memory(ctx, cap);
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_MIMO2 || !estimate.total_bytes) { return false; }
    g->context = ctx;
    g->cap = cap;
    g->ws.prefill_cap = cap;
    /* No shared-expert scratch: MiMo has no shared experts. K/V and attention
     * outputs have different widths, unlike the common GQA allocator. */
#define M2_ALLOC(field, elems) do { \
    const uint64_t size = (uint64_t)(elems) * sizeof(float); \
    g->field = ds4_gpu_tensor_alloc(size); \
    if (!g->field) { mimo2_graph_free(g); return false; } \
    g->bytes += size; \
} while (0)
    M2_ALLOC(ws.b_cur, (uint64_t)cap * M2_EMBED);
    M2_ALLOC(ws.b_norm, (uint64_t)cap * M2_EMBED);
    M2_ALLOC(ws.b_attn_out, (uint64_t)cap * M2_EMBED);
    M2_ALLOC(ws.b_ffn_out, (uint64_t)cap * M2_EMBED);
    M2_ALLOC(ws.b_q, (uint64_t)cap * M2_Q);
    M2_ALLOC(ws.b_k, (uint64_t)cap * M2_K);
    M2_ALLOC(ws.b_v, (uint64_t)cap * M2_V);
    M2_ALLOC(ws.b_heads, (uint64_t)cap * M2_HEADS);
    M2_ALLOC(ws.b_dense_gate, (uint64_t)cap * M2_DENSE);
    M2_ALLOC(ws.b_dense_up, (uint64_t)cap * M2_DENSE);
    M2_ALLOC(ws.b_dense_mid, (uint64_t)cap * M2_DENSE);
    M2_ALLOC(ws.b_router_logits, (uint64_t)cap * M2_EXPERTS);
    M2_ALLOC(ws.b_router_selected, (uint64_t)cap * M2_USED);
    M2_ALLOC(ws.b_router_weights, (uint64_t)cap * M2_USED);
    M2_ALLOC(ws.b_routed_gate, (uint64_t)cap * M2_USED * M2_FF);
    M2_ALLOC(ws.b_routed_up, (uint64_t)cap * M2_USED * M2_FF);
    M2_ALLOC(ws.b_routed_mid, (uint64_t)cap * M2_USED * M2_FF);
    M2_ALLOC(ws.b_routed_down, (uint64_t)cap * M2_USED * M2_EMBED);
    M2_ALLOC(qkv, (uint64_t)cap * M2_QKV);
    M2_ALLOC(positions, cap);
    M2_ALLOC(tokens, cap);
    M2_ALLOC(freq_full, M2_ROT / 2);
    M2_ALLOC(freq_swa, M2_ROT / 2);
    M2_ALLOC(rope_full, (uint64_t)cap * M2_ROT);
    M2_ALLOC(rope_swa, (uint64_t)cap * M2_ROT);
    M2_ALLOC(logits, DS4_N_VOCAB);
#undef M2_ALLOC
    g->host_positions = xmalloc(cap * sizeof(unsigned));
    float full[M2_ROT / 2], swa[M2_ROT / 2];
    for (unsigned d = 0; d < M2_ROT / 2; d++) {
        full[d] = (float)pow(10000000.0, -2.0 * d / M2_ROT);
        swa[d] = (float)pow(10000.0, -2.0 * d / M2_ROT);
    }
    if (!ds4_gpu_tensor_write(g->freq_full, 0, full, sizeof(full)) ||
        !ds4_gpu_tensor_write(g->freq_swa, 0, swa, sizeof(swa))) { mimo2_graph_free(g); return false; }
    for (unsigned il = 0; il < MIMO2_LAYERS; il++) {
        const unsigned heads = mimo2_is_full(il) ? 4 : 8;
        g->kv_cap[il] = mimo2_kv_capacity(il, ctx, cap);
        const uint64_t bytes = (uint64_t)g->kv_cap[il] * heads * (192 + 128) * sizeof(uint16_t);
        g->kv[il] = ds4_gpu_tensor_alloc(bytes);
        if (!g->kv[il]) { mimo2_graph_free(g); return false; }
        g->bytes += bytes;
    }
    if (g->bytes != estimate.total_bytes) { mimo2_graph_free(g); return false; }
    /* Draft KV is outside the trunk estimate. One journal slot per block
     * covers the single step that block takes in a trial. */
    g->hidden = ds4_gpu_tensor_alloc((uint64_t)M2_EMBED * sizeof(float));
    g->joined = ds4_gpu_tensor_alloc(2ull * M2_EMBED * sizeof(float));
    const uint64_t draft_row = 8ull * (192 + 128) * sizeof(uint16_t);
    g->journal = ds4_gpu_tensor_alloc(MIMO2_DRAFT_LAYERS * draft_row);
    g->verify_logits = xcalloc((size_t)M2_TRIAL * DS4_N_VOCAB, sizeof(float));
    for (unsigned il = 0; il < MIMO2_DRAFT_LAYERS; il++) {
        const unsigned layer = MIMO2_LAYERS + il;
        g->kv_cap[layer] = mimo2_kv_capacity(layer, ctx, cap);
        g->kv[layer] = ds4_gpu_tensor_alloc((uint64_t)g->kv_cap[layer] * draft_row);
        if (!g->kv[layer]) { mimo2_graph_free(g); return false; }
    }
    if (!g->hidden || !g->joined || !g->journal || !g->verify_logits) {
        mimo2_graph_free(g);
        return false;
    }
    if (dflash) {
        g->dflash_tap = ds4_gpu_tensor_alloc(1024ull * 5 * 4096 * sizeof(float));
        g->dflash_pos = xmalloc(1024 * sizeof(unsigned));
        g->dflash_saved = xmalloc((size_t)M2_TRIAL * 5 * 4096 * sizeof(float));
        if (!g->dflash_tap || !g->dflash_pos || !g->dflash_saved) {
            mimo2_graph_free(g);
            return false;
        }
        for (unsigned i = 0; i < 1024; i++) { g->dflash_pos[i] = UINT_MAX; }
    }
    return true;
}

static bool mimo2_apply_spans(ds4_mimo2_graph *g, unsigned n, unsigned pos) {
    for (unsigned s = 0; s < g->span_n; s++) {
        const mimo2_media_span *span = &g->spans[s];
        const unsigned stop = span->start + span->count;
        const unsigned from = span->start > pos ? span->start : pos;
        const unsigned to = stop < pos + n ? stop : pos + n;
        if (!span->rows || from >= to) { continue; }
        const uint64_t bytes = (uint64_t)(to - from) * M2_EMBED * sizeof(float);
        ds4_gpu_tensor *view = ds4_gpu_tensor_view(
            g->ws.b_cur, (uint64_t)(from - pos) * M2_EMBED * sizeof(float), bytes);
        const float *src = span->rows + (uint64_t)(from - span->start) * M2_EMBED;
        const bool ok = view && ds4_gpu_tensor_write(view, 0, src, bytes);
        ds4_gpu_tensor_free(view);
        if (getenv("DS4_MIMO2_VISION_TRACE")) {
            fprintf(stderr, "mimo-span from=%u to=%u n=%u first=%.8g\n",
                    from, to, n, src[0]);
            fflush(stderr);
        }
        if (!ok) { return false; }
    }
    return true;
}

static bool mimo2_layer(ds4_mimo2_graph *g, const ds4_model *m,
                         const ds4_weights *w, unsigned il, unsigned n) {
    ds4_exaone_batch_ws *b = &g->ws;
    const ds4_layer_weights *l = &w->layer[il];
    const bool full = mimo2_is_full(il);
    const unsigned heads = full ? 4 : 8, qkv_rows = M2_Q + heads * (192 + 128);
    if (!ds4_gpu_exaone_rms_norm_tensor(b->b_norm, b->b_cur, m->map, m->size,
            l->attn_norm->abs_offset, M2_EMBED, n, DS4_RMS_EPS) ||
        !plain_graph_matmul_tensor(g->qkv, m, l->attn_qkv, M2_EMBED, qkv_rows, b->b_norm, n) ||
        !ds4_gpu_mimo2_qkv(b->b_q, b->b_k, b->b_v, g->qkv,
            full ? g->rope_full : g->rope_swa, heads, n) ||
        !ds4_gpu_mimo2_kv_store(g->kv[il], b->b_k, b->b_v, g->positions, heads, n, g->kv_cap[il]) ||
        !ds4_gpu_mimo2_attention(b->b_heads, b->b_q, g->kv[il], g->positions,
            l->attn_sinks ? m->map : NULL, m->size, l->attn_sinks ? l->attn_sinks->abs_offset : 0,
            heads, g->kv_cap[il], n, full ? 0 : M2_WINDOW) ||
        !plain_graph_matmul_tensor(b->b_attn_out, m, l->attn_output, M2_HEADS, M2_EMBED, b->b_heads, n) ||
        !ds4_gpu_dots3_scale_tensor(b->b_attn_out, 0.707f, (uint64_t)n * M2_EMBED) ||
        !ds4_gpu_exaone_add_tensor(b->b_cur, b->b_attn_out, (uint64_t)n * M2_EMBED) ||
        !ds4_gpu_exaone_rms_norm_tensor(b->b_norm, b->b_cur, m->map, m->size,
            l->ffn_norm->abs_offset, M2_EMBED, n, DS4_RMS_EPS)) { return false; }
    if (!l->ffn_gate_exps) {
        if (!plain_graph_matmul_tensor(b->b_dense_gate, m, l->ffn_gate, M2_EMBED, M2_DENSE, b->b_norm, n) ||
            !plain_graph_matmul_tensor(b->b_dense_up, m, l->ffn_up, M2_EMBED, M2_DENSE, b->b_norm, n) ||
            !ds4_gpu_step37_swiglu(b->b_dense_mid, b->b_dense_gate, b->b_dense_up, NULL, M2_DENSE, n, 0) ||
            !plain_graph_matmul_tensor(b->b_ffn_out, m, l->ffn_down, M2_DENSE, M2_EMBED, b->b_dense_mid, n)) { return false; }
    } else {
        if (!plain_graph_matmul_tensor(b->b_router_logits, m, l->ffn_gate_inp, M2_EMBED, M2_EXPERTS, b->b_norm, n) ||
            !ds4_gpu_mimo2_router(b->b_router_selected, b->b_router_weights, b->b_router_logits,
                m->map, m->size, l->ffn_exp_probs_b->abs_offset, n) ||
            !ds4_gpu_routed_gate_up_tensor(b->b_routed_gate, b->b_routed_up, b->b_norm, b->b_router_selected,
                m->map, m->size, l->ffn_gate_exps->abs_offset, l->ffn_gate_exps->bytes,
                l->ffn_up_exps->abs_offset, l->ffn_up_exps->bytes, l->ffn_gate_exps->type,
                M2_EMBED, M2_FF, M2_EXPERTS, n, M2_USED) ||
            !ds4_gpu_step37_swiglu(b->b_routed_mid, b->b_routed_gate, b->b_routed_up, NULL, M2_FF, n * M2_USED, 0) ||
            !ds4_gpu_routed_matmul_guarded_tensor(b->b_routed_down, b->b_routed_mid, b->b_router_selected,
                m->map, m->size, l->ffn_down_exps->abs_offset, l->ffn_down_exps->bytes,
                l->ffn_down_exps->type, M2_FF, M2_EMBED, M2_EXPERTS, n * M2_USED, 1, n) ||
            !ds4_gpu_step37_sum(b->b_ffn_out, b->b_routed_down, b->b_router_weights, M2_EMBED, n)) { return false; }
    }
    return ds4_gpu_exaone_add_tensor(b->b_cur, b->b_ffn_out, (uint64_t)n * M2_EMBED);
}

static bool mimo2_head(ds4_mimo2_graph *g, const ds4_model *m, const ds4_weights *w, unsigned row) {
    ds4_gpu_tensor *last = ds4_gpu_tensor_view(g->ws.b_cur,
        (uint64_t)row * M2_EMBED * sizeof(float), M2_EMBED * sizeof(float));
    const bool ok = last && ds4_gpu_exaone_rms_norm_tensor(g->ws.b_norm, last, m->map, m->size,
        w->output_norm->abs_offset, M2_EMBED, 1, DS4_RMS_EPS) &&
        plain_graph_matmul_tensor(g->logits, m, w->output, M2_EMBED, DS4_N_VOCAB, g->ws.b_norm, 1);
    ds4_gpu_tensor_free(last);
    return ok;
}

static bool mimo2_forward(ds4_mimo2_graph *g, const ds4_model *m, const ds4_weights *w,
                           const int *tokens, unsigned n, unsigned pos) {
    if (!g || !tokens || g->failed || !n || n > g->cap || pos != g->position ||
        pos > g->context || n > g->context - pos) { return false; }
    for (unsigned i = 0; i < n; i++) {
        if (tokens[i] < 0 || (unsigned)tokens[i] >= DS4_N_VOCAB) { return false; }
        g->host_positions[i] = pos + i;
    }
    // A partial device failure invalidates KV; the owner must discard this state.
    g->failed = true;
    if (!ds4_gpu_tensor_write(g->tokens, 0, tokens, n * sizeof(int)) ||
        !ds4_gpu_tensor_write(g->positions, 0, g->host_positions, n * sizeof(unsigned)) ||
        !ds4_gpu_step37_rope(g->rope_full, g->freq_full, g->positions, M2_ROT, n) ||
        !ds4_gpu_step37_rope(g->rope_swa, g->freq_swa, g->positions, M2_ROT, n) ||
        !ds4_gpu_embed_tokens_quant_tensor(g->ws.b_cur, g->tokens, m->map, m->size,
            w->token_embd->abs_offset, w->token_embd->type, DS4_N_VOCAB, n, M2_EMBED) ||
        !mimo2_apply_spans(g, n, pos)) { return false; }
    for (unsigned il = 0; il < MIMO2_LAYERS; il++) {
        if (!mimo2_layer(g, m, w, il, n)) { return false; }
        if (g->dflash_tap && (il == 0 || il == 11 || il == 23 || il == 35 || il == 47)) {
            const unsigned which = il == 0 ? 0 : il == 11 ? 1 : il == 23 ? 2 : il == 35 ? 3 : 4;
            for (unsigned i = 0; i < n; i++) {
                const unsigned p = pos + i;
                const unsigned slot = p % 1024u;
                const uint64_t dst = ((uint64_t)slot * 5u + which) * 4096u * sizeof(float);
                const uint64_t src = (uint64_t)i * 4096u * sizeof(float);
                if (!ds4_gpu_tensor_copy(g->dflash_tap, dst, g->ws.b_cur, src,
                        4096u * sizeof(float))) { return false; }
                g->dflash_pos[slot] = p;
            }
        }
    }
    if (!mimo2_head(g, m, w, n - 1)) { return false; }
    if (!ds4_gpu_tensor_copy(g->hidden, 0, g->ws.b_cur,
            (uint64_t)(n - 1) * M2_EMBED * sizeof(float), (uint64_t)M2_EMBED * sizeof(float))) {
        return false;
    }
    g->hidden_valid = true;
    g->position = pos + n;
    g->failed = false;
    return true;
}
