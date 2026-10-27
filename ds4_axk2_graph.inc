/* Native GPU state only. Rust owns artifact validation and input processing.
 *
 * A.X-K2 runs 61 pre-norm decoder layers over a BF16 residual (layer 0 a
 * dense SwiGLU, 1-60 MoE), every norm before attention and before the MoE a
 * gated RMSNorm (rank-16 sigmoid gate):
 *
 *   embed ─▶ [ gnorm ─▶ MLA(+DSA) ─▶ +h ─▶ gnorm ─▶ shared + top-8 of 256 ─▶ +h ] x61 ─▶ rms ─▶ head
 *
 * Every layer keeps every position: an FP8 latent row (512 E4M3 + one
 * exponent) with its BF16 rope key, and an FP8 indexer key (128 E4M3 + one
 * exponent), 770 bytes per token and layer (11.5 GiB at 262,144). A row
 * that sees more than 2,048 keys attends the indexer's top 2,048; MLA runs
 * absorbed through the Q4_K kv_b. The embedding is read from the mapped SSD
 * part row by row; Q4_K projections run on GLM VQ's kernels, the experts
 * and the attention on A.X-K2's runtime (ds4_axk2_gpu.cuh). */

typedef struct {
    uint32_t ctx_cap, row_cap, position, attn_rows, score_rows;
    bool ready, failed, borrowed;
    uint64_t workspace_bytes, kv_bytes;
#define AXK2_FIELD(field, cols, type) ds4_gpu_tensor *field;
    AXK2_ROW_BUFFERS(AXK2_FIELD)
#undef AXK2_FIELD
    ds4_gpu_tensor *logits, *inv_freq, *q_abs, *latent_out, *absorb, *partials, *scores;
    ds4_gpu_tensor *latent[DS4_MAX_LAYER], *k_pe[DS4_MAX_LAYER], *keys[DS4_MAX_LAYER];
    uint16_t *host_embed;
    uint32_t matrices[AXK2_LAYERS][3u * AXK2_EXPERTS];
#ifndef __APPLE__
    ds4_gvq_ctx *gvq;
    ds4_axk2_moe *moe;
#endif
} ds4_axk2_graph;

/* Every graph owns its KV rows. A bank graph borrows everything else from
 * graph 0: bank forwards are serialized, so scratch is shared. */
static void axk2_graph_free(ds4_axk2_graph *g) {
    if (!g) { return; }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor_free(g->latent[il]);
        ds4_gpu_tensor_free(g->k_pe[il]);
        ds4_gpu_tensor_free(g->keys[il]);
    }
    if (g->borrowed) {
        memset(g, 0, sizeof(*g));
        return;
    }
#ifndef __APPLE__
    ds4_gpu_axk2_moe_free(g->moe);
    ds4_gpu_gvq_free(g->gvq);
#endif
#define AXK2_FREE(field, cols, type) ds4_gpu_tensor_free(g->field);
    AXK2_ROW_BUFFERS(AXK2_FREE)
#undef AXK2_FREE
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->inv_freq);
    ds4_gpu_tensor_free(g->q_abs);
    ds4_gpu_tensor_free(g->latent_out);
    ds4_gpu_tensor_free(g->absorb);
    ds4_gpu_tensor_free(g->partials);
    ds4_gpu_tensor_free(g->scores);
    free(g->host_embed);
    memset(g, 0, sizeof(*g));
}

/* Attention reads only rows at or below a row's position, and every row is
 * written before it is read: a reset only rewinds the position. */
static bool axk2_graph_reset(ds4_axk2_graph *g) {
    if (!g || !g->ready) { return false; }
    if (g->failed && !ds4_gpu_synchronize()) { return false; }
    g->position = 0u;
    g->failed = false;
    return true;
}

typedef enum { AXK2_HEAD_LAST, AXK2_HEAD_SKIP } axk2_head_mode;

#ifndef __APPLE__
/* FP8 caches start with zero exponents: dense flash tiles read the
 * exponents of rows past the live keys (their bytes zero-fill). */
static bool axk2_zero_exps(ds4_gpu_tensor *cache, uint64_t at) {
    ds4_gpu_tensor *exps = ds4_gpu_tensor_view(cache, at, ds4_gpu_tensor_bytes(cache) - at);
    const bool ok = exps && ds4_gpu_tensor_fill_f32(exps, 0.0f, ds4_gpu_tensor_bytes(exps) / sizeof(float));
    ds4_gpu_tensor_free(exps);
    return ok;
}

static bool axk2_graph_kv(ds4_axk2_graph *g) {
    const ds4_axk2_keys format = axk2_key_format();
    g->kv_bytes = 0;
    for (uint32_t il = 0; il < AXK2_LAYERS; il++) {
        g->latent[il] = ds4_gpu_tensor_alloc(ds4_axk2_latent_bytes(g->ctx_cap));
        g->k_pe[il] = ds4_gpu_tensor_alloc((uint64_t)g->ctx_cap * AXK2_ROPE * sizeof(uint16_t));
        g->keys[il] = ds4_gpu_tensor_alloc(ds4_axk2_index_bytes(g->ctx_cap, format));
        if (!g->latent[il] || !g->k_pe[il] || !g->keys[il] ||
            !axk2_zero_exps(g->latent[il], (uint64_t)g->ctx_cap * AXK2_KV_LORA) ||
            (format == DS4_AXK2_KEYS_FP8 && !axk2_zero_exps(g->keys[il], (uint64_t)g->ctx_cap * AXK2_IDX_DIM))) {
            return false;
        }
        g->kv_bytes += axk2_layer_kv_bytes(g->ctx_cap);
    }
    return g->kv_bytes == axk2_kv_bytes(g->ctx_cap);
}

/* transformers _compute_yarn_parameters for A.X-K2's rope_parameters (theta
 * 1e6, 64 rope dims, factor 2 over 131,072, beta 32/1, truncated correction
 * range 15..24) in the source's float32 ops; each step rounds once in double
 * so host fast-math cannot contract or reciprocate it. */
static bool axk2_rope_table(ds4_axk2_graph *g) {
    const double base = 1e6, factor = 2.0, orig = 131072.0, dim = AXK2_ROPE;
    const double pi2 = 2.0 * 3.14159265358979323846;
    const double low = floor(dim * log(orig / (32.0 * pi2)) / (2.0 * log(base)));
    const double high = ceil(dim * log(orig / (1.0 * pi2)) / (2.0 * log(base)));
    float inv[AXK2_ROPE / 2u];
    for (uint32_t j = 0; j < AXK2_ROPE / 2u; j++) {
        const float pos = (float)pow(base, (double)(float)(2.0 * j / dim));
        const float extrapolation = (float)(1.0 / (double)pos);
        const float interpolation = (float)(1.0 / (double)(float)(factor * (double)pos));
        float ramp = (float)(((double)(float)j - low) / (high - low));
        ramp = ramp < 0.0f ? 0.0f : ramp > 1.0f ? 1.0f : ramp;
        const float keep = (float)(1.0 - (double)ramp);   // inv_freq_extrapolation_factor
        const float a = (float)((double)interpolation * (double)(float)(1.0 - (double)keep));
        const float b = (float)((double)extrapolation * (double)keep);
        inv[j] = (float)((double)a + (double)b);
    }
    return ds4_gpu_tensor_write(g->inv_freq, 0, inv, sizeof(inv)) != 0;
}

static bool axk2_graph_alloc(ds4_axk2_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t ctx, uint32_t cap) {
    memset(g, 0, sizeof(*g));
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_AXK2_VQ || !w->glmvq_gpu || !ctx || !cap ||
        cap > ctx || ctx > AXK2_CONTEXT) { return false; }
    g->ctx_cap = ctx;
    g->row_cap = cap;
    g->attn_rows = axk2_attn_rows(cap);
    g->score_rows = axk2_score_rows(ctx, cap);
    g->gvq = ds4_gpu_gvq_create_linear(w->glmvq_gpu, axk2_gvq_scratch(cap), cap);
    g->moe = g->gvq ? ds4_gpu_axk2_moe_create(g->gvq, cap) : NULL;
    if (!g->gvq || !g->moe) { axk2_graph_free(g); return false; }
#define AXK2_ALLOC(field, cols, type) do { \
        g->field = ds4_gpu_tensor_alloc((uint64_t)cap * (cols) * sizeof(type)); \
        if (!g->field) { \
            fprintf(stderr, "ds4: A.X-K2 graph allocation failed: %s\n", #field); \
            axk2_graph_free(g); \
            return false; \
        } \
    } while (0);
    AXK2_ROW_BUFFERS(AXK2_ALLOC)
#undef AXK2_ALLOC
    const uint64_t slice = (uint64_t)g->attn_rows * AXK2_HEADS * AXK2_KV_LORA * sizeof(float);
    const uint64_t absorb = ds4_gpu_axk2_absorb_bytes(g->attn_rows);
    g->logits = ds4_gpu_tensor_alloc((uint64_t)DS4_N_VOCAB * sizeof(float));
    g->inv_freq = ds4_gpu_tensor_alloc(AXK2_ROPE / 2u * sizeof(float));
    g->q_abs = ds4_gpu_tensor_alloc(slice);
    g->latent_out = ds4_gpu_tensor_alloc(slice);
    g->absorb = absorb ? ds4_gpu_tensor_alloc(absorb) : NULL;
    g->partials = ds4_gpu_tensor_alloc(ds4_gpu_axk2_flash_bytes());
    g->scores = ds4_gpu_tensor_alloc((uint64_t)g->score_rows * ctx * sizeof(float));
    g->host_embed = xmalloc((size_t)cap * AXK2_EMBD * sizeof(uint16_t));
    if (!g->logits || !g->inv_freq || !g->q_abs || !g->latent_out || (absorb && !g->absorb) || !g->partials ||
        !g->scores || !axk2_rope_table(g) || !axk2_graph_kv(g)) {
        fprintf(stderr, "ds4: A.X-K2 graph allocation failed\n");
        axk2_graph_free(g);
        return false;
    }
    for (uint32_t il = AXK2_DENSE; il < AXK2_LAYERS; il++) {
        const ds4_axk2_layer *e = &w->axk2[il];
        for (uint32_t x = 0; x < AXK2_EXPERTS; x++) {
            g->matrices[il][x] = axk2_index(m, e->gate[x]);
            g->matrices[il][AXK2_EXPERTS + x] = axk2_index(m, e->up[x]);
            g->matrices[il][2u * AXK2_EXPERTS + x] = axk2_index(m, e->down[x]);
        }
    }
    g->workspace_bytes = axk2_workspace_bytes(ctx, cap);
    g->ready = true;
    if (!axk2_graph_reset(g)) { axk2_graph_free(g); return false; }
    ds4_log(stderr, DS4_LOG_TIMING,
        "ds4: A.X-K2 graph ctx=%u chunk=%u KV=%.2f GiB workspace=%.2f GiB%s\n", ctx, cap,
        (double)g->kv_bytes / 1073741824.0, (double)g->workspace_bytes / 1073741824.0,
        axk2_key_format() == DS4_AXK2_KEYS_BF16 ? " (BF16 index keys)" : "");
    return true;
}

/* DS4_AXK2_TRACE=DIR: the forward starting at DS4_AXK2_TRACE_POS (default 0)
 * writes each traced tensor's rows as stored (F32, ids I32) to
 * DIR/layer_LL_NAME.bin. */
static bool axk2_trace(uint32_t il, uint32_t pos, const char *name, const ds4_gpu_tensor *t, uint64_t bytes) {
    const char *root = getenv("DS4_AXK2_TRACE"), *at = getenv("DS4_AXK2_TRACE_POS");
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
static bool axk2_graph_bank(ds4_axk2_graph *g, const ds4_axk2_graph *owner) {
    *g = *owner;
    g->borrowed = true;
    g->ready = false;
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) { g->latent[il] = g->k_pe[il] = g->keys[il] = NULL; }
    if (!axk2_graph_kv(g)) {
        axk2_graph_free(g);
        return false;
    }
    g->ready = true;
    if (!axk2_graph_reset(g)) { axk2_graph_free(g); return false; }
    return true;
}

static bool axk2_mm(ds4_axk2_graph *g, const ds4_model *m, ds4_gpu_tensor *out,
        const ds4_tensor *w, const ds4_gpu_tensor *x, uint32_t rows, ds4_gvq_math math) {
    const bool ok = ds4_gpu_gvq_linear(g->gvq, out, axk2_index(m, w), x, rows, math) != 0;
    if (!ok && w) {
        fprintf(stderr, "ds4: A.X-K2 linear %.*s failed (rows=%u)\n", (int)w->name.len, w->name.ptr, rows);
    }
    return ok;
}

// Q4_K projections of one input: prefill prepares it once (Q8_1 / FP16).
// One token streams each matrix on its own: the shared launch splits a row
// over two warps (q_a and kv_a ran at ~114 GB/s), alone they split by height.
static bool axk2_mm_many(ds4_axk2_graph *g, const ds4_model *m, ds4_gpu_tensor *const *outs,
        const ds4_tensor *const *ws, uint32_t count, const ds4_gpu_tensor *x, uint32_t rows) {
    if (rows == 1u) {
        for (uint32_t c = 0; c < count; c++) {
            if (!axk2_mm(g, m, outs[c], ws[c], x, rows, DS4_GVQ_BF16)) { return false; }
        }
        return true;
    }
    uint32_t index[4];
    for (uint32_t c = 0; c < count; c++) { index[c] = axk2_index(m, ws[c]); }
    const bool ok = count <= 4 && ds4_gpu_gvq_linear_many(g->gvq, outs, index, count, x, rows, DS4_GVQ_BF16);
    if (!ok) { fprintf(stderr, "ds4: A.X-K2 linear group of %u failed (rows=%u)\n", count, rows); }
    return ok;
}

static bool axk2_q8(ds4_axk2_graph *g, const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *w,
        const ds4_gpu_tensor *x, uint32_t rows) {
    const bool ok = ds4_gpu_axk2_q8(g->gvq, out, axk2_index(m, w), x, rows) != 0;
    if (!ok) { fprintf(stderr, "ds4: A.X-K2 Q8_0 %.*s failed (rows=%u)\n", (int)w->name.len, w->name.ptr, rows); }
    return ok;
}

// A view of rows [first, first + count) of a buffer `cols` elements wide.
static ds4_gpu_tensor *axk2_rows(ds4_gpu_tensor *t, uint32_t first, uint32_t count, uint64_t cols, uint64_t size) {
    return ds4_gpu_tensor_view(t, (uint64_t)first * cols * size, (uint64_t)count * cols * size);
}

/* The DSA indexer over every causal key: scores of `score_rows` rows at a
 * time, then each row's top 2,048 into `selected`. Keys are always stored
 * (a later sparse continuation must see them); queries only score once the
 * chunk's last row sees more than 2,048 keys. */
static bool axk2_select(ds4_axk2_graph *g, const ds4_model *m, const ds4_layer_weights *l, uint32_t il,
        uint32_t rows, uint32_t pos) {
    const uint32_t keys = pos + rows;
    if (!axk2_q8(g, m, g->idx_q, l->attn_idx_q_b, g->q_r, rows) || !ds4_gpu_axk2_index_q(g->idx_q, g->rope, rows) ||
        !axk2_mm(g, m, g->idx_w, l->attn_idx_w, g->norm, rows, DS4_GVQ_F32)) { return false; }
    bool ok = true;
    for (uint32_t r0 = 0; ok && r0 < rows; r0 += g->score_rows) {
        const uint32_t n = rows - r0 < g->score_rows ? rows - r0 : g->score_rows;
        ds4_gpu_tensor *q = axk2_rows(g->idx_q, r0, n, AXK2_IDX_HEADS * AXK2_IDX_DIM, sizeof(float));
        ds4_gpu_tensor *w = axk2_rows(g->idx_w, r0, n, AXK2_IDX_HEADS, sizeof(float));
        ds4_gpu_tensor *sel = axk2_rows(g->selected, r0, n, AXK2_TOPK, sizeof(uint32_t));
        ds4_gpu_tensor *scores = ds4_gpu_tensor_view(g->scores, 0, (uint64_t)n * keys * sizeof(float));
        ok = q && w && sel && scores &&
            ds4_gpu_axk2_index_scores(scores, q, w, g->keys[il], n, pos + r0, keys, keys, g->ctx_cap,
                axk2_key_format()) &&
            ds4_gpu_axk2_topk(sel, scores, keys, n);
        ds4_gpu_tensor_free(scores);
        ds4_gpu_tensor_free(sel);
        ds4_gpu_tensor_free(w);
        ds4_gpu_tensor_free(q);
    }
    return ok;
}

/* MLA with the fused query/output gate: q_a -> q_a_layernorm -> q_b over
 * [q_r | q_a] (192 query + 128 gate values per head), kv_a -> latent row and
 * rope key, the indexer's keys (and selection), then absorbed attention in
 * row slices and o_proj over the gated heads. */
static bool axk2_attention(ds4_axk2_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t il,
        uint32_t rows, uint32_t pos) {
    const ds4_layer_weights *l = &w->layer[il];
    ds4_gpu_tensor *const lows[] = {g->q_a, g->kv};
    const ds4_tensor *const lows_w[] = {l->attn_q_a, l->attn_kv_a_mqa};
    if (!axk2_mm_many(g, m, lows, lows_w, 2u, g->norm, rows) ||
        !ds4_gpu_gvq_rms(g->gvq, g->q_r, g->q_a, NULL, axk2_index(m, l->attn_q_a_norm), rows, AXK2_Q_RANK,
            DS4_RMS_EPS, DS4_GVQ_RMS) ||
        !ds4_gpu_gvq_concat(g->qin, g->q_r, g->q_a, rows, AXK2_Q_RANK) ||
        !axk2_mm(g, m, g->qb, l->attn_q_b, g->qin, rows, DS4_GVQ_BF16) ||
        !ds4_gpu_axk2_q_split(g->q, g->gate, g->qb, g->rope, rows) ||
        !ds4_gpu_axk2_kv_store(g->gvq, g->latent[il], g->k_pe[il], g->kv, axk2_index(m, l->attn_kv_a_norm), g->rope,
            rows, pos, g->ctx_cap) ||
        !axk2_q8(g, m, g->idx_k, l->attn_idx_k, g->norm, rows) ||
        !ds4_gpu_axk2_index_store(g->gvq, g->keys[il], g->idx_k, axk2_index(m, l->attn_idx_k_norm),
            axk2_index(m, l->attn_idx_k_norm_bias), g->rope, rows, pos, g->ctx_cap, axk2_key_format()) ||
        !axk2_trace(il, pos, "q", g->q, (uint64_t)rows * AXK2_HEADS * AXK2_KEY * sizeof(float))) { return false; }
    const bool sparse = pos + rows > AXK2_TOPK;
    if (sparse && (!axk2_select(g, m, l, il, rows, pos) ||
            !axk2_trace(il, pos, "selected", g->selected, (uint64_t)rows * AXK2_TOPK * sizeof(uint32_t)))) {
        return false;
    }
    const uint32_t kv_b = axk2_index(m, l->attn_kv_b);
    bool ok = true;
    for (uint32_t s0 = 0; ok && s0 < rows; s0 += g->attn_rows) {
        const uint32_t n = rows - s0 < g->attn_rows ? rows - s0 : g->attn_rows;
        ds4_gpu_tensor *q = axk2_rows(g->q, s0, n, AXK2_HEADS * AXK2_KEY, sizeof(float));
        ds4_gpu_tensor *gate = axk2_rows(g->gate, s0, n, AXK2_HEADS * AXK2_VALUE, sizeof(float));
        ds4_gpu_tensor *heads = axk2_rows(g->heads, s0, n, AXK2_HEADS * AXK2_VALUE, sizeof(float));
        ds4_gpu_tensor *sel = sparse ? axk2_rows(g->selected, s0, n, AXK2_TOPK, sizeof(uint32_t)) : NULL;
        ok = q && gate && heads && (!sparse || sel) &&
            ds4_gpu_axk2_absorb(g->gvq, g->q_abs, q, kv_b, n, g->absorb,
                s0 ? DS4_AXK2_KV_B_REUSE : DS4_AXK2_KV_B_RESTORE) &&
            ds4_gpu_axk2_flash(g->gvq, g->latent_out, g->partials, q, g->q_abs, g->latent[il], g->k_pe[il], sel, n,
                pos + s0, AXK2_SCALE, g->ctx_cap) &&
            ds4_gpu_axk2_value(g->gvq, heads, g->latent_out, kv_b, gate, n, g->absorb);
        ds4_gpu_tensor_free(sel);
        ds4_gpu_tensor_free(heads);
        ds4_gpu_tensor_free(gate);
        ds4_gpu_tensor_free(q);
    }
    return ok && axk2_trace(il, pos, "heads", g->heads, (uint64_t)rows * AXK2_HEADS * AXK2_VALUE * sizeof(float)) &&
        axk2_mm(g, m, g->attn_out, l->attn_output, g->heads, rows, DS4_GVQ_BF16);
}

// SwiGLU MLP (dense FFN or shared expert); the mid rows land in qb. The
// fused Q4_K gate/up serves 2..GLMVQ_Q4M_TOKENS rows; one row streams gate
// and up apart (the fused pass ran at ~147 GB/s there).
static bool axk2_mlp(ds4_axk2_graph *g, const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *gate,
        const ds4_tensor *up, const ds4_tensor *down, uint32_t rows) {
    if (rows > 1u &&
        ds4_gpu_gvq_gate_up(g->gvq, g->qb, axk2_index(m, gate), axk2_index(m, up), g->norm, rows, INFINITY)) {
        return axk2_mm(g, m, out, down, g->qb, rows, DS4_GVQ_BF16);
    }
    ds4_gpu_tensor *const outs[] = {g->qb, g->ff_up};
    const ds4_tensor *const ws[] = {gate, up};
    return axk2_mm_many(g, m, outs, ws, 2u, g->norm, rows) &&
        ds4_gpu_gvq_swiglu(g->qb, g->qb, g->ff_up, rows, (uint32_t)gate->dim[1], INFINITY, DS4_GVQ_ROWS_F32) &&
        axk2_mm(g, m, out, down, g->qb, rows, DS4_GVQ_BF16);
}

// F32 router logits (BF16 weights widened), grouped top-8, VQ experts plus
// the shared expert in the routed reduce.
static bool axk2_ffn(ds4_axk2_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t il, uint32_t rows,
        uint32_t pos) {
    const ds4_layer_weights *l = &w->layer[il];
    if (il < AXK2_DENSE) { return axk2_mlp(g, m, g->ffn_out, l->ffn_gate, l->ffn_up, l->ffn_down, rows); }
    const uint64_t route = (uint64_t)rows * AXK2_USED * 4u;
    return axk2_mm(g, m, g->router, l->ffn_gate_inp, g->norm, rows, DS4_GVQ_F32) &&
        ds4_gpu_axk2_router(g->gvq, g->route_ids, g->route_weights, g->router, axk2_index(m, l->ffn_exp_probs_b),
            rows) &&
        axk2_trace(il, pos, "routes", g->route_ids, route) && axk2_trace(il, pos, "weights", g->route_weights, route) &&
        axk2_mlp(g, m, g->shared, l->ffn_gate_shexp, l->ffn_up_shexp, l->ffn_down_shexp, rows) &&
        ds4_gpu_axk2_moe(g->moe, g->ffn_out, g->norm, g->route_ids, g->route_weights, g->matrices[il], rows, g->shared);
}

/* Layer il over norm = input_layernorm(cur). Each residual add also writes
 * the norm its successor reads: post_attention_layernorm, then the next
 * layer's input_layernorm (or the final norm after the last layer). */
static bool axk2_layer(ds4_axk2_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t il, uint32_t rows,
        uint32_t pos) {
    const ds4_layer_weights *l = &w->layer[il];
    const ds4_axk2_layer *a = &w->axk2[il];
    const bool dense = il < AXK2_DENSE, last = il + 1u == AXK2_LAYERS;
    const ds4_tensor *next = last ? w->output_norm : w->layer[il + 1u].attn_norm;
    const uint64_t hidden = (uint64_t)rows * AXK2_EMBD * sizeof(float);
    const bool ok = axk2_attention(g, m, w, il, rows, pos) &&
        ds4_gpu_axk2_norm(g->gvq, g->norm, g->cur, g->attn_out, axk2_index(m, l->ffn_norm),
            dense ? DS4_GVQ_NO_LINK : axk2_index(m, a->post_down), dense ? DS4_GVQ_NO_LINK : axk2_index(m, a->post_up),
            g->gmid, rows) &&
        axk2_trace(il, pos, "attn", g->cur, hidden) &&
        axk2_ffn(g, m, w, il, rows, pos) &&
        axk2_trace(il, pos, "ffn_out", g->ffn_out, hidden) &&
        ds4_gpu_axk2_norm(g->gvq, g->norm, g->cur, g->ffn_out, axk2_index(m, next),
            last ? DS4_GVQ_NO_LINK : axk2_index(m, w->axk2[il + 1u].in_down),
            last ? DS4_GVQ_NO_LINK : axk2_index(m, w->axk2[il + 1u].in_up), g->gmid, rows) &&
        axk2_trace(il, pos, "hidden", g->cur, hidden);
    if (!ok) { fprintf(stderr, "ds4: A.X-K2 layer %u failed (rows=%u pos=%u)\n", il, rows, pos); }
    return ok;
}

/* Ask the kernel to read `rows` tokens' embedding rows ahead (Beam's
 * prefetch: a cold 2,048-row gather faulted one page at a time). */
static void axk2_embed_prefetch(const ds4_model *m, const ds4_weights *w, const int *tokens, uint32_t rows) {
    const uint64_t row = (uint64_t)AXK2_EMBD * sizeof(uint16_t);
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint8_t *table = m->map + w->token_embd->abs_offset;
    for (uint32_t r = 0; r < rows; r++) {
        if (tokens[r] < 0 || (uint32_t)tokens[r] >= DS4_N_VOCAB) { continue; }
        const uintptr_t at = (uintptr_t)(table + (uint64_t)tokens[r] * row);
        const uintptr_t first = at / page * page, last = (at + row + page - 1) / page * page;
        (void)madvise((void *)first, last - first, MADV_WILLNEED);
    }
}

// The host gathers BF16 rows from the mapped embedding part (SSD-backed
// page cache); the device widens them into cur.
static bool axk2_embed(ds4_axk2_graph *g, const ds4_model *m, const ds4_weights *w, const int *tokens,
        uint32_t rows) {
    const uint64_t row = (uint64_t)AXK2_EMBD * sizeof(uint16_t);
    const uint8_t *table = m->map + w->token_embd->abs_offset;
    if (rows > 1u) { axk2_embed_prefetch(m, w, tokens, rows); }
    for (uint32_t r = 0; r < rows; r++) {
        memcpy((uint8_t *)g->host_embed + r * row, table + (uint64_t)tokens[r] * row, row);
    }
    return ds4_gpu_tensor_write(g->embed, 0u, g->host_embed, (uint64_t)rows * row) &&
        ds4_gpu_axk2_embed(g->cur, g->embed, rows);
}

// The head over the final-normed row `row` (no soft cap).
static bool axk2_head(ds4_axk2_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t row) {
    const uint64_t bytes = (uint64_t)AXK2_EMBD * sizeof(float);
    ds4_gpu_tensor *norm = ds4_gpu_tensor_view(g->norm, (uint64_t)row * bytes, bytes);
    const bool ok = norm && axk2_mm(g, m, g->logits, w->output, norm, 1u, DS4_GVQ_BF16);
    ds4_gpu_tensor_free(norm);
    return ok;
}

static bool axk2_forward(ds4_axk2_graph *g, const ds4_model *m, const ds4_weights *w,
        const int *tokens, uint32_t rows, uint32_t pos, axk2_head_mode head) {
    if (g->failed || !g->ready || !rows || rows > g->row_cap || pos != g->position ||
        pos > g->ctx_cap || rows > g->ctx_cap - pos) { return false; }
    for (uint32_t i = 0; i < rows; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) { return false; }
    }
    /* A failed enqueue leaves partially written KV rows; the session must
     * reset before another forward. */
    g->failed = true;
    const ds4_axk2_layer *a0 = &w->axk2[0];
    if (!axk2_embed(g, m, w, tokens, rows) || !ds4_gpu_axk2_rope(g->rope, g->inv_freq, rows, pos) ||
        !axk2_trace(0u, pos, "embed", g->cur, (uint64_t)rows * AXK2_EMBD * sizeof(float)) ||
        !ds4_gpu_axk2_norm(g->gvq, g->norm, g->cur, NULL, axk2_index(m, w->layer[0].attn_norm),
            axk2_index(m, a0->in_down), axk2_index(m, a0->in_up), g->gmid, rows)) { return false; }
    for (uint32_t il = 0; il < AXK2_LAYERS; il++) {
        if (!axk2_layer(g, m, w, il, rows, pos)) { return false; }
    }
    if (head == AXK2_HEAD_LAST && (!axk2_head(g, m, w, rows - 1u) ||
            !axk2_trace(AXK2_LAYERS, pos, "logits", g->logits, (uint64_t)DS4_N_VOCAB * sizeof(float)))) {
        return false;
    }
    g->position = pos + rows;
    g->failed = false;
    return true;
}
#else
// A.X-K2 is CUDA-only; Metal keeps linkage and refuses the graph.
static bool axk2_graph_alloc(ds4_axk2_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t ctx, uint32_t cap) {
    (void)g; (void)m; (void)w; (void)ctx; (void)cap;
    return false;
}

static bool axk2_forward(ds4_axk2_graph *g, const ds4_model *m, const ds4_weights *w,
        const int *tokens, uint32_t rows, uint32_t pos, axk2_head_mode head) {
    (void)g; (void)m; (void)w; (void)tokens; (void)rows; (void)pos; (void)head;
    return false;
}

static bool axk2_graph_bank(ds4_axk2_graph *g, const ds4_axk2_graph *owner) {
    (void)g; (void)owner;
    return false;
}

static void axk2_embed_prefetch(const ds4_model *m, const ds4_weights *w, const int *tokens, uint32_t rows) {
    (void)m; (void)w; (void)tokens; (void)rows;
}
#endif
