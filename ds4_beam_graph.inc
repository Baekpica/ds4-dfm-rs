/* Native GPU state only. Rust owns artifact validation and input processing.
 *
 * Beam-1.0 runs 52 decoder layers over a BF16 residual with sandwich norms
 * and depth-scaled residual adds (sublayer i: h = y / sqrt(i) + h sqrt((i-1)/i),
 * attention i = 2(L+1), FFN i = 2(L+1)+1):
 *
 *   embed ─▶ [ rms ─▶ attn ─▶ rms ─▶ +h ─▶ rms ─▶ dense | MoE ─▶ rms ─▶ +h ] x52 ─▶ rms ─▶ head ─▶ 30 tanh(·/30)
 *
 * 43 sliding layers (window 512, full-head RoPE) keep a ring of 512 + one
 * chunk of f16 [K | V] rows; the nine NoPE full layers keep every position as
 * Solar's FP8 rows (2,080 bytes per token and layer, 18.3 GiB at 1M). The
 * embedding is read from the mapped SSD part row by row; Q4_K projections run
 * on GLM VQ's kernels and the routed experts on Beam's runtime
 * (ds4_beam_gpu.cuh). */

typedef struct {
    uint32_t ctx_cap, row_cap, swa_cap, position;
    bool ready, failed, borrowed;
    uint64_t workspace_bytes, kv_bytes;
#define BEAM_FIELD(field, cols, type) ds4_gpu_tensor *field;
    BEAM_ROW_BUFFERS(BEAM_FIELD)
#undef BEAM_FIELD
    ds4_gpu_tensor *logits, *partials;
    ds4_gpu_tensor *kv[DS4_MAX_LAYER];
    uint16_t *host_embed;
    uint32_t matrices[BEAM_LAYERS][3u * BEAM_EXPERTS];
#ifndef __APPLE__
    ds4_gvq_ctx *gvq;
    ds4_beam_moe *moe;
#endif
} ds4_beam_graph;

/* Every graph owns its KV rows. A bank graph borrows everything else from
 * graph 0: bank forwards are serialized, so scratch is shared. */
static void beam_graph_free(ds4_beam_graph *g) {
    if (!g) { return; }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) { ds4_gpu_tensor_free(g->kv[il]); }
    if (g->borrowed) {
        memset(g, 0, sizeof(*g));
        return;
    }
#ifndef __APPLE__
    ds4_gpu_beam_moe_free(g->moe);
    ds4_gpu_gvq_free(g->gvq);
#endif
#define BEAM_FREE(field, cols, type) ds4_gpu_tensor_free(g->field);
    BEAM_ROW_BUFFERS(BEAM_FREE)
#undef BEAM_FREE
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->partials);
    free(g->host_embed);
    memset(g, 0, sizeof(*g));
}

/* Attention reads only rows below the position (full layers) or inside the
 * window (rings), and every row is written before it is read: a reset only
 * rewinds the position. */
static bool beam_graph_reset(ds4_beam_graph *g) {
    if (!g || !g->ready) { return false; }
    if (g->failed && !ds4_gpu_synchronize()) { return false; }
    g->position = 0u;
    g->failed = false;
    return true;
}

typedef enum { BEAM_HEAD_LAST, BEAM_HEAD_SKIP } beam_head_mode;

#ifndef __APPLE__
static bool beam_graph_kv(ds4_beam_graph *g) {
    g->kv_bytes = 0;
    for (uint32_t il = 0; il < BEAM_LAYERS; il++) {
        const uint64_t bytes = ds4_beam_layer_is_full(il) ? (uint64_t)g->ctx_cap * beam_full_row()
                                                          : (uint64_t)g->swa_cap * BEAM_SWA_ROW;
        g->kv[il] = ds4_gpu_tensor_alloc(bytes);
        if (!g->kv[il]) { return false; }
        g->kv_bytes += bytes;
    }
    return g->kv_bytes == beam_kv_bytes(g->ctx_cap, g->row_cap);
}

static bool beam_graph_alloc(ds4_beam_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t ctx, uint32_t cap) {
    memset(g, 0, sizeof(*g));
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_BEAM_VQ || !w->glmvq_gpu || !ctx || !cap ||
        cap > ctx || ctx > BEAM_CONTEXT) { return false; }
    g->ctx_cap = ctx;
    g->row_cap = cap;
    g->swa_cap = beam_swa_cap(ctx, cap);
    g->gvq = ds4_gpu_gvq_create_linear(w->glmvq_gpu, beam_gvq_scratch(cap), cap);
    g->moe = g->gvq ? ds4_gpu_beam_moe_create(g->gvq, cap) : NULL;
    if (!g->gvq || !g->moe) { beam_graph_free(g); return false; }
#define BEAM_ALLOC(field, cols, type) do { \
        g->field = ds4_gpu_tensor_alloc((uint64_t)cap * (cols) * sizeof(type)); \
        if (!g->field) { \
            fprintf(stderr, "ds4: Beam graph allocation failed: %s\n", #field); \
            beam_graph_free(g); \
            return false; \
        } \
    } while (0);
    BEAM_ROW_BUFFERS(BEAM_ALLOC)
#undef BEAM_ALLOC
    g->logits = ds4_gpu_tensor_alloc((uint64_t)DS4_N_VOCAB * sizeof(float));
    g->partials = ds4_gpu_tensor_alloc((uint64_t)BEAM_HEADS * BEAM_DECODE_SPLITS * (BEAM_HEAD + 2u) * sizeof(float));
    g->host_embed = xmalloc((size_t)cap * BEAM_EMBD * sizeof(uint16_t));
    if (!g->logits || !g->partials || !beam_graph_kv(g)) {
        fprintf(stderr, "ds4: Beam graph allocation failed\n");
        beam_graph_free(g);
        return false;
    }
    for (uint32_t il = BEAM_DENSE; il < BEAM_LAYERS; il++) {
        const ds4_beam_layer *e = &w->beam[il];
        for (uint32_t x = 0; x < BEAM_EXPERTS; x++) {
            g->matrices[il][x] = beam_index(m, e->gate[x]);
            g->matrices[il][BEAM_EXPERTS + x] = beam_index(m, e->up[x]);
            g->matrices[il][2u * BEAM_EXPERTS + x] = beam_index(m, e->down[x]);
        }
    }
    g->workspace_bytes = beam_workspace_bytes(cap);
    g->ready = true;
    if (!beam_graph_reset(g)) { beam_graph_free(g); return false; }
    ds4_log(stderr, DS4_LOG_TIMING,
        "ds4: Beam graph ctx=%u chunk=%u KV=%.2f GiB workspace=%.2f GiB\n", ctx, cap,
        (double)g->kv_bytes / 1073741824.0, (double)g->workspace_bytes / 1073741824.0);
    return true;
}

/* DS4_BEAM_TRACE=DIR: the forward starting at DS4_BEAM_TRACE_POS (default 0)
 * writes each traced tensor's rows as stored (F32, ids I32) to
 * DIR/layer_LL_NAME.bin. */
static bool beam_trace(uint32_t il, uint32_t pos, const char *name, const ds4_gpu_tensor *t, uint64_t bytes) {
    const char *root = getenv("DS4_BEAM_TRACE"), *at = getenv("DS4_BEAM_TRACE_POS");
    if (!root || !root[0] || pos != (at ? (uint32_t)strtoul(at, NULL, 10) : 0u)) { return true; }
    char path[PATH_MAX];
    const int n = snprintf(path, sizeof(path), "%s/layer_%02u_%s.bin", root, il, name);
    if (n < 0 || (size_t)n >= sizeof(path)) { return false; }
    void *buffer = xmalloc((size_t)bytes);
    bool ok = ds4_gpu_tensor_read(t, 0, buffer, bytes) != 0;
    FILE *fp = ok ? fopen(path, "wb") : NULL;
    ok = fp && fwrite(buffer, 1u, (size_t)bytes, fp) == bytes;
    if (fp && fclose(fp)) { ok = false; }
    free(buffer);
    return ok;
}

// A bank graph: its own KV rows, everything else borrowed from graph 0.
static bool beam_graph_bank(ds4_beam_graph *g, const ds4_beam_graph *owner) {
    *g = *owner;
    g->borrowed = true;
    g->ready = false;
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) { g->kv[il] = NULL; }
    if (!beam_graph_kv(g)) {
        beam_graph_free(g);
        return false;
    }
    g->ready = true;
    if (!beam_graph_reset(g)) { beam_graph_free(g); return false; }
    return true;
}

static bool beam_mm(ds4_beam_graph *g, const ds4_model *m, ds4_gpu_tensor *out,
        const ds4_tensor *w, const ds4_gpu_tensor *x, uint32_t rows) {
    const bool ok = ds4_gpu_gvq_linear(g->gvq, out, beam_index(m, w), x, rows, DS4_GVQ_BF16) != 0;
    if (!ok && w) {
        fprintf(stderr, "ds4: Beam linear %.*s failed (rows=%u)\n", (int)w->name.len, w->name.ptr, rows);
    }
    return ok;
}

// Projections of one input: prefill prepares it once (Q8_1 / FP16).
static bool beam_mm_many(ds4_beam_graph *g, const ds4_model *m, ds4_gpu_tensor *const *outs,
        const ds4_tensor *const *ws, uint32_t count, const ds4_gpu_tensor *x, uint32_t rows) {
    uint32_t index[4];
    for (uint32_t c = 0; c < count; c++) { index[c] = beam_index(m, ws[c]); }
    const bool ok = count <= 4 && ds4_gpu_gvq_linear_many(g->gvq, outs, index, count, x, rows, DS4_GVQ_BF16);
    if (!ok) { fprintf(stderr, "ds4: Beam linear group of %u failed (rows=%u)\n", count, rows); }
    return ok;
}

/* Gated GQA: q/k RMSNorm per head, RoPE on the sliding layers (the full
 * layers' queries take the position scale instead), K/V into the layer's
 * ring or FP8 rows, attention, the per-element sigmoid gate and o_proj. */
static bool beam_attention(ds4_beam_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t il, uint32_t rows, uint32_t pos) {
    const ds4_layer_weights *l = &w->layer[il];
    const bool full = ds4_beam_layer_is_full(il);
    ds4_gpu_tensor *const outs[] = {g->q, g->k, g->v, g->gate};
    const ds4_tensor *const ws[] = {l->attn_q, l->attn_k, l->attn_v, l->attn_gate};
    // v_proj's BF16 output: prefill projections leave F32 sums.
    if (!beam_mm_many(g, m, outs, ws, 4u, g->norm, rows) ||
        !ds4_gpu_gvq_round(g->v, (uint64_t)rows * BEAM_KV) ||
        !ds4_gpu_beam_qk(g->gvq, g->q, beam_index(m, l->attn_q_norm), BEAM_HEADS, rows, pos,
            full ? DS4_BEAM_QK_SCALE : DS4_BEAM_QK_ROPE) ||
        !ds4_gpu_beam_qk(g->gvq, g->k, beam_index(m, l->attn_k_norm), BEAM_KV_HEADS, rows, pos,
            full ? DS4_BEAM_QK_NORM : DS4_BEAM_QK_ROPE) ||
        !beam_trace(il, pos, "q", g->q, (uint64_t)rows * BEAM_Q * sizeof(float)) ||
        !beam_trace(il, pos, "k", g->k, (uint64_t)rows * BEAM_KV * sizeof(float)) ||
        !beam_trace(il, pos, "v", g->v, (uint64_t)rows * BEAM_KV * sizeof(float))) { return false; }
    bool ok;
    if (full) {
        ok = ds4_gpu_solar_kv_store_tensor(g->kv[il], g->k, g->v, BEAM_KV_HEADS, BEAM_HEAD, rows, pos,
                g->ctx_cap, beam_full_kv());
        if (ok && rows == 1u) {
            // About BEAM_DECODE_SPLITS / 2 chunks of at least 256 keys.
            const uint32_t keys = pos + 1u, chunk = (keys + BEAM_DECODE_SPLITS / 2u - 1u) / (BEAM_DECODE_SPLITS / 2u);
            ok = ds4_gpu_solar_attention_decode_split_tensor(g->heads, g->q, g->kv[il], g->partials,
                BEAM_HEADS, BEAM_KV_HEADS, BEAM_HEAD, g->ctx_cap, pos, 0u, chunk < 256u ? 256u : chunk,
                beam_full_kv());
        } else if (ok) {
            ok = ds4_gpu_solar_attention_prefill_tensor(g->heads, g->q, g->kv[il], rows, pos, BEAM_HEADS,
                BEAM_KV_HEADS, BEAM_HEAD, g->ctx_cap, 0u, beam_full_kv());
        }
    } else {
        ok = ds4_gpu_exaone_kv_store_tensor(g->kv[il], g->k, g->v, BEAM_KV, rows, pos, g->swa_cap) &&
            (rows == 1u
                ? ds4_gpu_exaone_attention_decode_tensor(g->heads, g->q, g->kv[il], BEAM_HEADS,
                      BEAM_KV_HEADS, BEAM_HEAD, g->swa_cap, pos, BEAM_WINDOW)
                : ds4_gpu_exaone_attention_prefill_tensor(g->heads, g->q, g->kv[il], rows, pos,
                      BEAM_HEADS, BEAM_KV_HEADS, BEAM_HEAD, g->swa_cap, BEAM_WINDOW));
    }
    const uint64_t heads = (uint64_t)rows * BEAM_Q * sizeof(float);
    return ok && beam_trace(il, pos, "heads", g->heads, heads) &&
        ds4_gpu_beam_gate(g->gvq, g->heads, g->gate, beam_index(m, w->beam[il].gate_bias), rows) &&
        beam_trace(il, pos, "gated", g->heads, heads) &&
        beam_mm(g, m, g->attn_out, l->attn_output, g->heads, rows) &&
        beam_trace(il, pos, "attn_out", g->attn_out, (uint64_t)rows * BEAM_EMBD * sizeof(float));
}

// SwiGLU MLP of the dense layers: fused Q4_K gate/up at decode and verify
// widths, separate projections otherwise; mid stays in ff_gate.
static bool beam_dense(ds4_beam_graph *g, const ds4_model *m, const ds4_layer_weights *l, uint32_t rows) {
    if (ds4_gpu_l31_gate_up(g->gvq, g->ff_gate, beam_index(m, l->ffn_gate), beam_index(m, l->ffn_up),
            g->norm, rows, INFINITY)) {
        return beam_mm(g, m, g->ffn_out, l->ffn_down, g->ff_gate, rows);
    }
    ds4_gpu_tensor *const outs[] = {g->ff_gate, g->ff_up};
    const ds4_tensor *const ws[] = {l->ffn_gate, l->ffn_up};
    return beam_mm_many(g, m, outs, ws, 2u, g->norm, rows) &&
        ds4_gpu_l31_swiglu(g->ff_gate, g->ff_gate, g->ff_up, rows, BEAM_DENSE_FF, INFINITY, DS4_GVQ_ROWS_F32) &&
        beam_mm(g, m, g->ffn_out, l->ffn_down, g->ff_gate, rows);
}

// F32 router logits (the source keeps F32 weights), grouped top-8, VQ experts.
static bool beam_moe(ds4_beam_graph *g, const ds4_model *m, const ds4_layer_weights *l, uint32_t il,
        uint32_t rows) {
    return ds4_gpu_beam_router(g->gvq, g->selected, g->route_weights, g->router, g->norm,
            beam_index(m, l->ffn_gate_inp), beam_index(m, l->ffn_exp_probs_b), rows) &&
        ds4_gpu_beam_moe(g->moe, g->ffn_out, g->norm, g->selected, g->route_weights, g->matrices[il], rows);
}

/* Layer il over norm = input_layernorm(cur). Each residual also writes the
 * RMS norm its successor reads: pre_feedforward, then `next` (the next
 * layer's input_layernorm; NULL leaves the residual for the head). */
static bool beam_layer(ds4_beam_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t il, uint32_t rows, uint32_t pos, const ds4_tensor *next) {
    const ds4_layer_weights *l = &w->layer[il];
    const uint32_t sublayer = 2u * (il + 1u);
    const bool dense = il < BEAM_DENSE;
    const uint64_t hidden = (uint64_t)rows * BEAM_EMBD * sizeof(float), route = (uint64_t)rows * BEAM_USED * 4u;
    const bool ok = beam_attention(g, m, w, il, rows, pos) &&
        ds4_gpu_beam_residual(g->gvq, g->cur, g->attn_out, beam_index(m, l->attn_output_norm), g->norm,
            beam_index(m, l->ffn_norm), rows, sublayer, 1.0f) &&
        beam_trace(il, pos, "attn", g->cur, hidden) &&
        (dense ? beam_dense(g, m, l, rows)
               : beam_moe(g, m, l, il, rows) && beam_trace(il, pos, "routes", g->selected, route) &&
                     beam_trace(il, pos, "weights", g->route_weights, route)) &&
        beam_trace(il, pos, "ffn_out", g->ffn_out, hidden) &&
        ds4_gpu_beam_residual(g->gvq, g->cur, g->ffn_out, beam_index(m, l->ffn_output_norm), g->norm,
            next ? beam_index(m, next) : DS4_GVQ_NO_LINK, rows, sublayer + 1u,
            dense ? 1.0f : DS4_EXPERT_WEIGHT_SCALE) &&
        beam_trace(il, pos, "ffn", g->cur, hidden);
    if (!ok) { fprintf(stderr, "ds4: Beam layer %u failed (rows=%u pos=%u)\n", il, rows, pos); }
    return ok;
}

// The host gathers BF16 rows from the mapped embedding part (SSD-backed
// page cache); the device widens them into cur.
static bool beam_embed(ds4_beam_graph *g, const ds4_model *m, const ds4_weights *w,
        const int *tokens, uint32_t rows) {
    const uint64_t row = (uint64_t)BEAM_EMBD * sizeof(uint16_t);
    const uint8_t *table = m->map + w->token_embd->abs_offset;
    for (uint32_t r = 0; r < rows; r++) {
        memcpy((uint8_t *)g->host_embed + r * row, table + (uint64_t)tokens[r] * row, row);
    }
    return ds4_gpu_tensor_write(g->embed, 0u, g->host_embed, (uint64_t)rows * row) &&
        ds4_gpu_beam_embed(g->cur, g->embed, rows);
}

// Final norm, head and the soft cap of `row`.
static bool beam_head(ds4_beam_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t row) {
    const uint64_t bytes = (uint64_t)BEAM_EMBD * sizeof(float);
    ds4_gpu_tensor *cur = ds4_gpu_tensor_view(g->cur, (uint64_t)row * bytes, bytes);
    ds4_gpu_tensor *norm = ds4_gpu_tensor_view(g->norm, 0u, bytes);
    const bool ok = cur && norm && ds4_gpu_beam_rms(g->gvq, norm, cur, beam_index(m, w->output_norm), 1u) &&
        beam_mm(g, m, g->logits, w->output, norm, 1u) &&
        ds4_gpu_beam_logits(g->logits, 1u, BEAM_TOKENS);
    ds4_gpu_tensor_free(norm);
    ds4_gpu_tensor_free(cur);
    return ok;
}

static bool beam_forward(ds4_beam_graph *g, const ds4_model *m, const ds4_weights *w,
        const int *tokens, uint32_t rows, uint32_t pos, beam_head_mode head) {
    if (g->failed || !g->ready || !rows || rows > g->row_cap || pos != g->position ||
        pos > g->ctx_cap || rows > g->ctx_cap - pos) { return false; }
    for (uint32_t i = 0; i < rows; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) { return false; }
    }
    /* A failed enqueue leaves partially written KV rows; the session must
     * reset before another forward. */
    g->failed = true;
    if (!beam_embed(g, m, w, tokens, rows) ||
        !beam_trace(0u, pos, "embed", g->cur, (uint64_t)rows * BEAM_EMBD * sizeof(float)) ||
        !ds4_gpu_beam_rms(g->gvq, g->norm, g->cur, beam_index(m, w->layer[0].attn_norm), rows)) { return false; }
    for (uint32_t il = 0; il < BEAM_LAYERS; il++) {
        const ds4_tensor *next = il + 1u < BEAM_LAYERS ? w->layer[il + 1u].attn_norm : NULL;
        if (!beam_layer(g, m, w, il, rows, pos, next)) { return false; }
    }
    if (head == BEAM_HEAD_LAST && (!beam_head(g, m, w, rows - 1u) ||
            !beam_trace(BEAM_LAYERS, pos, "logits", g->logits, (uint64_t)DS4_N_VOCAB * sizeof(float)))) {
        return false;
    }
    g->position = pos + rows;
    g->failed = false;
    return true;
}
#else
// Beam is CUDA-only; Metal keeps linkage and refuses the graph.
static bool beam_graph_alloc(ds4_beam_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t ctx, uint32_t cap) {
    (void)g; (void)m; (void)w; (void)ctx; (void)cap;
    return false;
}

static bool beam_forward(ds4_beam_graph *g, const ds4_model *m, const ds4_weights *w,
        const int *tokens, uint32_t rows, uint32_t pos, beam_head_mode head) {
    (void)g; (void)m; (void)w; (void)tokens; (void)rows; (void)pos; (void)head;
    return false;
}

static bool beam_graph_bank(ds4_beam_graph *g, const ds4_beam_graph *owner) {
    (void)g; (void)owner;
    return false;
}
#endif
