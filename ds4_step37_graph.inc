/* Native GPU state only. Rust owns artifact validation and input processing.
 * The common GQA scratch/matmul executor supplies the existing quant paths;
 * Step retains its own layer schedule, RoPE, routing and activation ordering. */
enum { S37_HIDDEN = 4096, S37_MAX_HEADS = 96, S37_KV = 1024, S37_HEAD = 128,
       S37_DENSE = 11264, S37_EXPERT_FF = 1280, S37_EXPERTS = 288, S37_USED = 8,
       S37_WINDOW = 512, S37_CONTEXT = 262144 };

typedef struct {
    ds4_exaone_batch_ws ws;
    ds4_gpu_tensor *positions, *tokens, *freq_full, *freq_swa;
    ds4_gpu_tensor *rope_full, *rope_swa, *gate, *logits;
    ds4_gpu_tensor *kv[STEP37_LAYERS + STEP37_DRAFT_LAYERS];
    unsigned kv_cap[STEP37_LAYERS + STEP37_DRAFT_LAYERS];
    unsigned context, cap, position, high_water;
    uint64_t bytes;
    unsigned *host_positions;
    bool failed;
} ds4_step37_graph;

static unsigned step37_prefill_cap(unsigned ctx) {
    enum { DEFAULT_CHUNK = 512, MAX_CHUNK = 4096 };
    unsigned cap = DEFAULT_CHUNK;
    const char *value = getenv("DS4_STEP37_PREFILL_CHUNK");
    if (value && *value) {
        char *end = NULL;
        unsigned long parsed = strtoul(value, &end, 10);
        if (end != value && !*end && parsed > 0 && parsed <= MAX_CHUNK) {
            cap = (unsigned)parsed;
        }
    }
    return cap < ctx ? cap : ctx;
}

static bool step37_sliding(unsigned il) {
    return il >= STEP37_LAYERS || il % STEP37_FULL_PERIOD != 0;
}

static ds4_context_memory step37_memory(unsigned ctx, unsigned cap) {
    ds4_context_memory m = {0};
    if (!ctx || !cap || cap > ctx || ctx > S37_CONTEXT) { return m; }
    m.raw_cap = ctx;
    m.prefill_cap = cap;
    for (unsigned il = 0; il < STEP37_LAYERS; il++) {
        unsigned rows = step37_sliding(il) ? S37_WINDOW + cap : ctx;
        if (rows > ctx) { rows = ctx; }
        m.raw_bytes += (uint64_t)rows * 2 * S37_KV * sizeof(uint16_t);
    }
    /* Shared GQA scratch plus Step's positions, token IDs, gates and two
     * RoPE tables. Decode uses the first row of the same workspace. */
    const uint64_t common = 5 * S37_HIDDEN + 2 * S37_MAX_HEADS * S37_HEAD + 2 * S37_KV +
        3 * S37_DENSE + 3 * S37_EXPERT_FF + S37_EXPERTS + 2 * S37_USED +
        3 * S37_USED * S37_EXPERT_FF + S37_USED * S37_HIDDEN;
    const uint64_t controls = 2 + S37_MAX_HEADS + S37_HEAD + S37_HEAD / 2;
    m.scratch_bytes = ((uint64_t)cap * (common + controls) +
                      S37_HEAD * 3 / 4 + DS4_N_VOCAB) * sizeof(float);
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}

static void step37_graph_free(ds4_step37_graph *g) {
    exaone_batch_ws_free(&g->ws);
    ds4_gpu_tensor *all[] = {g->positions, g->tokens, g->freq_full, g->freq_swa,
                            g->rope_full, g->rope_swa, g->gate, g->logits};
    for (unsigned i = 0; i < sizeof(all) / sizeof(all[0]); i++) {
        ds4_gpu_tensor_free(all[i]);
    }
    for (unsigned i = 0; i < STEP37_LAYERS + STEP37_DRAFT_LAYERS; i++) {
        ds4_gpu_tensor_free(g->kv[i]);
    }
    free(g->host_positions);
    memset(g, 0, sizeof(*g));
}

static bool step37_scratch_alloc(ds4_step37_graph *g, const ds4_model *m,
                                const ds4_weights *w, unsigned ctx, unsigned cap) {
    memset(g, 0, sizeof(*g));
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_STEP37 || !ctx || !cap || cap > ctx ||
        ctx > S37_CONTEXT || !w->step37_rope_freqs) { return false; }
    g->context = ctx;
    g->cap = cap;
    if (!exaone_batch_ws_alloc(&g->ws, cap, S37_HIDDEN, S37_MAX_HEADS * S37_HEAD, S37_KV)) {
        return false;
    }
    g->bytes = g->ws.bytes;
#define S37_ALLOC(field, elems) do { \
    const uint64_t nbytes = (uint64_t)(elems) * sizeof(float); \
    g->field = ds4_gpu_tensor_alloc(nbytes); \
    if (!g->field) { step37_graph_free(g); return false; } \
    g->bytes += nbytes; \
} while (0)
    S37_ALLOC(positions, cap);
    S37_ALLOC(tokens, cap);
    S37_ALLOC(freq_full, S37_HEAD / 4);
    S37_ALLOC(freq_swa, S37_HEAD / 2);
    S37_ALLOC(rope_full, (uint64_t)cap * S37_HEAD / 2);
    S37_ALLOC(rope_swa, (uint64_t)cap * S37_HEAD);
    S37_ALLOC(gate, (uint64_t)cap * S37_MAX_HEADS);
    S37_ALLOC(logits, DS4_N_VOCAB);
#undef S37_ALLOC
    g->host_positions = xmalloc(cap * sizeof(*g->host_positions));
    const float *factor = tensor_data(m, w->step37_rope_freqs);
    float full[S37_HEAD / 4], swa[S37_HEAD / 2];
    for (unsigned i = 0; i < S37_HEAD / 2; i++) {
        swa[i] = (float)pow(10000.0, -2.0 * i / S37_HEAD);
        if (i < S37_HEAD / 4) {
            if (!isfinite(factor[i]) || factor[i] <= 0) { step37_graph_free(g); return false; }
            full[i] = (float)pow(5000000.0, -4.0 * i / S37_HEAD) / factor[i];
        }
    }
    if (!ds4_gpu_tensor_write(g->freq_full, 0, full, sizeof(full)) ||
        !ds4_gpu_tensor_write(g->freq_swa, 0, swa, sizeof(swa))) {
        step37_graph_free(g); return false;
    }
    return true;
}

static bool step37_graph_alloc(ds4_step37_graph *g, const ds4_model *m,
                                const ds4_weights *w, unsigned ctx, unsigned cap) {
    if (!step37_scratch_alloc(g, m, w, ctx, cap)) { return false; }
    for (unsigned i = 0; i < STEP37_LAYERS; i++) {
        /* Store the whole chunk before attention: keep its preceding window. */
        unsigned rows = step37_sliding(i) ? S37_WINDOW + cap : ctx;
        if (rows > ctx) { rows = ctx; }
        const uint64_t bytes = (uint64_t)rows * 2 * S37_KV * sizeof(uint16_t);
        g->kv[i] = ds4_gpu_tensor_alloc(bytes);
        if (!g->kv[i]) { step37_graph_free(g); return false; }
        g->kv_cap[i] = rows;
        g->bytes += bytes;
    }
    fprintf(stderr, "ds4: Step graph ctx=%u chunk=%u memory=%.3f GiB\n",
            ctx, cap, (double)g->bytes / 1073741824.0);
    if (g->bytes != step37_memory(ctx, cap).total_bytes) {
        fprintf(stderr, "ds4: Step graph allocation differs from admission estimate\n");
        step37_graph_free(g);
        return false;
    }
    return true;
}

#ifndef STEP37_TRACE
#define STEP37_TRACE(name, tensor, width, rows, layer, pos) ((void)0)
#endif

static bool step37_layer(ds4_step37_graph *g, const ds4_model *m,
                          const ds4_weights *w, unsigned il, unsigned n, unsigned pos) {
    ds4_exaone_batch_ws *b = &g->ws;
    const ds4_layer_weights *l = &w->layer[il];
    const bool sliding = step37_sliding(il);
    const unsigned heads = sliding ? S37_MAX_HEADS : 64;
    const unsigned rotary = sliding ? S37_HEAD : S37_HEAD / 2;
    const unsigned q_dim = heads * S37_HEAD;
    const unsigned window = sliding ? S37_WINDOW : 0;
    ds4_gpu_tensor *rope = sliding ? g->rope_swa : g->rope_full;
    STEP37_TRACE("attn_norm_in", b->b_cur, S37_HIDDEN, n, il, pos);
    if (!ds4_gpu_exaone_rms_norm_tensor(b->b_norm, b->b_cur, m->map, m->size,
            l->attn_norm->abs_offset, S37_HIDDEN, n, DS4_RMS_EPS) ||
        !plain_graph_matmul_tensor(b->b_q, m, l->attn_q, S37_HIDDEN, q_dim, b->b_norm, n) ||
        !plain_graph_matmul_tensor(b->b_k, m, l->attn_k, S37_HIDDEN, S37_KV, b->b_norm, n) ||
        !plain_graph_matmul_tensor(b->b_v, m, l->attn_v, S37_HIDDEN, S37_KV, b->b_norm, n) ||
        !plain_graph_matmul_tensor(g->gate, m, l->attn_gate, S37_HIDDEN, heads, b->b_norm, n) ||
        !ds4_gpu_step37_qk(b->b_q, b->b_q, m->map, m->size, l->attn_q_norm->abs_offset,
                           rope, heads, rotary, n) ||
        !ds4_gpu_step37_qk(b->b_k, b->b_k, m->map, m->size, l->attn_k_norm->abs_offset,
                           rope, DS4_N_HEAD_KV, rotary, n) ||
        !ds4_gpu_exaone_kv_store_tensor(g->kv[il], b->b_k, b->b_v, S37_KV, n, pos, g->kv_cap[il])) {
        return false;
    }
    STEP37_TRACE("attn_norm", b->b_norm, S37_HIDDEN, n, il, pos);
    STEP37_TRACE("Qcur_pos", b->b_q, q_dim, n, il, pos);
    STEP37_TRACE("Kcur_pos", b->b_k, S37_KV, n, il, pos);
    STEP37_TRACE("Vcur", b->b_v, S37_KV, n, il, pos);
    const int attn_ok = n > 1
        ? ds4_gpu_exaone_attention_prefill_tensor(b->b_heads, b->b_q, g->kv[il], n, pos,
                heads, DS4_N_HEAD_KV, S37_HEAD, g->kv_cap[il], window)
        : ds4_gpu_exaone_attention_decode_tensor(b->b_heads, b->b_q, g->kv[il],
                heads, DS4_N_HEAD_KV, S37_HEAD, g->kv_cap[il], pos, window);
    STEP37_TRACE("attn_out", b->b_heads, q_dim, n, il, pos);
    if (!attn_ok || !ds4_gpu_step37_gate(b->b_heads, g->gate, heads, n) ||
        !plain_graph_matmul_tensor(b->b_attn_out, m, l->attn_output, q_dim, S37_HIDDEN, b->b_heads, n) ||
        !ds4_gpu_exaone_add_tensor(b->b_cur, b->b_attn_out, (uint64_t)n * S37_HIDDEN) ||
        !ds4_gpu_exaone_rms_norm_tensor(b->b_norm, b->b_cur, m->map, m->size,
                l->ffn_norm->abs_offset, S37_HIDDEN, n, DS4_RMS_EPS)) { return false; }
    STEP37_TRACE("attn_gated", b->b_heads, q_dim, n, il, pos);
    STEP37_TRACE("attn_proj", b->b_attn_out, S37_HIDDEN, n, il, pos);
    STEP37_TRACE("ffn_inp", b->b_cur, S37_HIDDEN, n, il, pos);
    STEP37_TRACE("ffn_norm", b->b_norm, S37_HIDDEN, n, il, pos);
    if (!l->ffn_gate_exps) {
        if (!plain_graph_matmul_tensor(b->b_dense_gate, m, l->ffn_gate, S37_HIDDEN, S37_DENSE, b->b_norm, n) ||
            !plain_graph_matmul_tensor(b->b_dense_up, m, l->ffn_up, S37_HIDDEN, S37_DENSE, b->b_norm, n) ||
            !ds4_gpu_step37_swiglu(b->b_dense_mid, b->b_dense_gate, b->b_dense_up, NULL, S37_DENSE, n, 0) ||
            !plain_graph_matmul_tensor(b->b_ffn_out, m, l->ffn_down, S37_DENSE, S37_HIDDEN, b->b_dense_mid, n)) {
            return false;
        }
        STEP37_TRACE("ffn_out", b->b_ffn_out, S37_HIDDEN, n, il, pos);
        return ds4_gpu_exaone_add_tensor(b->b_cur, b->b_ffn_out, (uint64_t)n * S37_HIDDEN);
    }
    const float clamp = il >= 43 && il < STEP37_LAYERS ? 7.0f : 0.0f;
    const float shared_clamp = il >= 43 && il < STEP37_LAYERS ? 16.0f : 0.0f;
    if (!plain_graph_matmul_tensor(b->b_router_logits, m, l->ffn_gate_inp,
                S37_HIDDEN, S37_EXPERTS, b->b_norm, n) ||
        !ds4_gpu_step37_router(b->b_router_selected, b->b_router_weights, b->b_router_logits,
                m->map, m->size, l->ffn_exp_probs_b->abs_offset, n) ||
        !ds4_gpu_routed_gate_up_tensor(b->b_routed_gate, b->b_routed_up, b->b_norm, b->b_router_selected,
                m->map, m->size, l->ffn_gate_exps->abs_offset, l->ffn_gate_exps->bytes,
                l->ffn_up_exps->abs_offset, l->ffn_up_exps->bytes, l->ffn_gate_exps->type,
                S37_HIDDEN, S37_EXPERT_FF, S37_EXPERTS, n, S37_USED) ||
        /* No route weights before down: its activation quantizer is nonlinear. */
        !ds4_gpu_step37_swiglu(b->b_routed_mid, b->b_routed_gate, b->b_routed_up,
                NULL, S37_EXPERT_FF, n * S37_USED, clamp) ||
        !ds4_gpu_routed_matmul_guarded_tensor(b->b_routed_down, b->b_routed_mid, b->b_router_selected,
                m->map, m->size, l->ffn_down_exps->abs_offset, l->ffn_down_exps->bytes,
                l->ffn_down_exps->type, S37_EXPERT_FF, S37_HIDDEN, S37_EXPERTS, n * S37_USED, 1, n) ||
        !ds4_gpu_step37_sum(b->b_ffn_out, b->b_routed_down, b->b_router_weights, S37_HIDDEN, n) ||
        !plain_graph_matmul_tensor(b->b_shared_gate, m, l->ffn_gate_shexp,
                S37_HIDDEN, S37_EXPERT_FF, b->b_norm, n) ||
        !plain_graph_matmul_tensor(b->b_shared_up, m, l->ffn_up_shexp,
                S37_HIDDEN, S37_EXPERT_FF, b->b_norm, n) ||
        !ds4_gpu_step37_swiglu(b->b_shared_mid, b->b_shared_gate, b->b_shared_up,
                NULL, S37_EXPERT_FF, n, shared_clamp) ||
        !plain_graph_matmul_tensor(b->b_shared_out, m, l->ffn_down_shexp,
                S37_EXPERT_FF, S37_HIDDEN, b->b_shared_mid, n) ||
        !ds4_gpu_exaone_add_tensor(b->b_ffn_out, b->b_shared_out, (uint64_t)n * S37_HIDDEN)) {
        return false;
    }
    STEP37_TRACE("ffn_moe_topk", b->b_router_selected, S37_USED, n, il, pos);
    STEP37_TRACE("ffn_moe_weights_scaled", b->b_router_weights, S37_USED, n, il, pos);
    STEP37_TRACE("ffn_shared_out", b->b_shared_out, S37_HIDDEN, n, il, pos);
    STEP37_TRACE("ffn_out", b->b_ffn_out, S37_HIDDEN, n, il, pos);
    return ds4_gpu_exaone_add_tensor(b->b_cur, b->b_ffn_out, (uint64_t)n * S37_HIDDEN);
}

static bool step37_forward(ds4_step37_graph *g, const ds4_model *m, const ds4_weights *w,
                            const int *tokens, unsigned n, unsigned pos) {
    if (g->failed || !n || n > g->cap || pos != g->position || pos > g->context ||
        n > g->context - pos) { return false; }
    for (unsigned i = 0; i < n; i++) {
        if (tokens[i] < 0 || (unsigned)tokens[i] >= DS4_N_VOCAB) { return false; }
        g->host_positions[i] = pos + i;
    }
    /* A failed enqueue leaves partially written KV and cannot be resumed. */
    g->failed = true;
    if (!ds4_gpu_tensor_write(g->tokens, 0, tokens, n * sizeof(int)) ||
        !ds4_gpu_tensor_write(g->positions, 0, g->host_positions, n * sizeof(unsigned)) ||
        !ds4_gpu_step37_rope(g->rope_full, g->freq_full, g->positions, S37_HEAD / 2, n) ||
        !ds4_gpu_step37_rope(g->rope_swa, g->freq_swa, g->positions, S37_HEAD, n) ||
        !ds4_gpu_embed_tokens_quant_tensor(g->ws.b_cur, g->tokens, m->map, m->size,
                w->token_embd->abs_offset, w->token_embd->type, DS4_N_VOCAB, n, S37_HIDDEN)) {
        return false;
    }
    for (unsigned il = 0; il < STEP37_LAYERS; il++) {
        if (!step37_layer(g, m, w, il, n, pos)) {
            fprintf(stderr, "ds4: Step forward failed at layer %u\n", il);
            return false;
        }
    }
    ds4_gpu_tensor *last = ds4_gpu_tensor_view(g->ws.b_cur,
            (uint64_t)(n - 1) * S37_HIDDEN * sizeof(float), S37_HIDDEN * sizeof(float));
    const bool ok = last && ds4_gpu_exaone_rms_norm_tensor(g->ws.b_norm, last, m->map, m->size,
            w->output_norm->abs_offset, S37_HIDDEN, 1, DS4_RMS_EPS) &&
        plain_graph_matmul_tensor(g->logits, m, w->output, S37_HIDDEN, DS4_N_VOCAB, g->ws.b_norm, 1);
    ds4_gpu_tensor_free(last);
    if (ok) {
        g->position = pos + n;
        if (g->high_water < g->position) { g->high_water = g->position; }
        g->failed = false;
    }
    return ok;
}

/* The widened SWA ring retains one extra chunk for a tail rewrite. Reset to
 * zero is also safe: causal attention cannot read the old future rows. */
static bool step37_rewind(ds4_step37_graph *g, unsigned pos) {
    if (!g || !g->cap || g->failed || pos > g->position) { return false; }
    /* Repeated rewinds must not move the physical retention boundary. */
    unsigned newest = g->high_water > g->position ? g->high_water : g->position;
    if (pos && newest - pos > g->cap) { return false; }
    g->high_water = pos ? newest : 0;
    g->position = pos;
    return true;
}

static bool step37_reset(ds4_step37_graph *g) {
    if (g->failed && !ds4_gpu_synchronize()) { return false; }
    g->position = 0;
    g->high_water = 0;
    g->failed = false;
    return true;
}

/* External predictors have separate sliding KV and output heads. Their
 * unnormalized hidden output feeds the next depth; the head norm is separate. */
typedef struct {
    ds4_step37_graph graph;
    ds4_gpu_tensor *combined;
    unsigned position[STEP37_DRAFT_LAYERS], high_water[STEP37_DRAFT_LAYERS];
    unsigned hidden_depth, hidden_rows;
    bool hidden_valid;
} ds4_step37_draft;

static uint64_t step37_draft_bytes(unsigned ctx, unsigned cap) {
    ds4_context_memory main = step37_memory(ctx, cap);
    if (!main.total_bytes) { return 0; }
    const unsigned rows = ctx < S37_WINDOW + cap ? ctx : S37_WINDOW + cap;
    return main.scratch_bytes +
        (uint64_t)STEP37_DRAFT_LAYERS * rows * 2 * S37_KV * sizeof(uint16_t) +
        (uint64_t)cap * 2 * S37_HIDDEN * sizeof(float);
}

static void step37_draft_free(ds4_step37_draft *d) {
    step37_graph_free(&d->graph);
    ds4_gpu_tensor_free(d->combined);
    memset(d, 0, sizeof(*d));
}

static bool step37_draft_alloc(ds4_step37_draft *d, const ds4_model *m,
                                const ds4_weights *w, unsigned ctx, unsigned cap) {
    memset(d, 0, sizeof(*d));
    if (!step37_scratch_alloc(&d->graph, m, w, ctx, cap)) { return false; }
    const unsigned rows = ctx < S37_WINDOW + cap ? ctx : S37_WINDOW + cap;
    const uint64_t kv_bytes = (uint64_t)rows * 2 * S37_KV * sizeof(uint16_t);
    for (unsigned il = STEP37_LAYERS; il < STEP37_LAYERS + STEP37_DRAFT_LAYERS; il++) {
        if (!w->layer[il].nextn_shared_head_head) { step37_draft_free(d); return false; }
        d->graph.kv[il] = ds4_gpu_tensor_alloc(kv_bytes);
        if (!d->graph.kv[il]) { step37_draft_free(d); return false; }
        d->graph.kv_cap[il] = rows;
        d->graph.bytes += kv_bytes;
    }
    const uint64_t combined_bytes = (uint64_t)cap * 2 * S37_HIDDEN * sizeof(float);
    d->combined = ds4_gpu_tensor_alloc(combined_bytes);
    if (!d->combined) { step37_draft_free(d); return false; }
    d->graph.bytes += combined_bytes;
    if (d->graph.bytes != step37_draft_bytes(ctx, cap)) {
        step37_draft_free(d); return false;
    }
    return true;
}

static bool step37_draft_forward(ds4_step37_draft *d, const ds4_model *m,
                                  const ds4_weights *w, unsigned depth,
                                  ds4_gpu_tensor *hidden, ds4_gpu_tensor *embedding,
                                  unsigned n) {
    ds4_step37_graph *g = &d->graph;
    if (g->failed || depth >= STEP37_DRAFT_LAYERS || !n || n > g->cap ||
        !hidden || !embedding || d->position[depth] > g->context ||
        n > g->context - d->position[depth]) { return false; }
    const unsigned pos = d->position[depth];
    const unsigned il = STEP37_LAYERS + depth;
    const ds4_layer_weights *l = &w->layer[il];
    for (unsigned i = 0; i < n; i++) { g->host_positions[i] = pos + i; }
    d->hidden_valid = false;
    g->failed = true;
    if (!ds4_gpu_tensor_write(g->positions, 0, g->host_positions, n * sizeof(unsigned)) ||
        !ds4_gpu_step37_rope(g->rope_swa, g->freq_swa, g->positions, S37_HEAD, n) ||
        !ds4_gpu_exaone_rms_norm_tensor(g->ws.b_attn_out, embedding, m->map, m->size,
                l->nextn_enorm->abs_offset, S37_HIDDEN, n, DS4_RMS_EPS) ||
        !ds4_gpu_exaone_rms_norm_tensor(g->ws.b_norm, hidden, m->map, m->size,
                l->nextn_hnorm->abs_offset, S37_HIDDEN, n, DS4_RMS_EPS) ||
        !ds4_gpu_concat_rows_tensor(d->combined, g->ws.b_attn_out, g->ws.b_norm, n, S37_HIDDEN) ||
        !plain_graph_matmul_tensor(g->ws.b_cur, m, l->nextn_eh_proj,
                2 * S37_HIDDEN, S37_HIDDEN, d->combined, n) ||
        !step37_layer(g, m, w, il, n, pos)) { return false; }
    d->position[depth] = pos + n;
    if (d->high_water[depth] < pos + n) { d->high_water[depth] = pos + n; }
    d->hidden_depth = depth;
    d->hidden_rows = n;
    d->hidden_valid = true;
    g->failed = false;
    return true;
}

static bool step37_draft_logits(ds4_step37_draft *d, const ds4_model *m,
                                 const ds4_weights *w, unsigned depth, unsigned n) {
    ds4_step37_graph *g = &d->graph;
    if (g->failed || !d->hidden_valid || depth != d->hidden_depth || n != d->hidden_rows) {
        return false;
    }
    const ds4_layer_weights *l = &w->layer[STEP37_LAYERS + depth];
    ds4_gpu_tensor *last = ds4_gpu_tensor_view(g->ws.b_cur,
            (uint64_t)(n - 1) * S37_HIDDEN * sizeof(float), S37_HIDDEN * sizeof(float));
    g->failed = true;
    const bool ok = last && ds4_gpu_exaone_rms_norm_tensor(g->ws.b_norm, last, m->map, m->size,
            l->nextn_shared_head_norm->abs_offset, S37_HIDDEN, 1, DS4_RMS_EPS) &&
        plain_graph_matmul_tensor(g->logits, m, l->nextn_shared_head_head,
                S37_HIDDEN, DS4_N_VOCAB, g->ws.b_norm, 1);
    ds4_gpu_tensor_free(last);
    g->failed = !ok;
    return ok;
}

static bool step37_draft_rewind(ds4_step37_draft *d, unsigned depth, unsigned pos) {
    if (d->graph.failed || depth >= STEP37_DRAFT_LAYERS || pos > d->position[depth] ||
        (pos && d->high_water[depth] - pos > d->graph.cap)) { return false; }
    d->position[depth] = pos;
    if (!pos) { d->high_water[depth] = 0; }
    d->hidden_valid = false;
    return true;
}
