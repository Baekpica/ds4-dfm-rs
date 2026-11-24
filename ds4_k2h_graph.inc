/* Native GPU state only. Rust owns artifact validation and input processing.
 *
 * K2-Horizon runs 61 pre-norm decoder layers over a BF16 residual (layers
 * 0-2 a dense SwiGLU, 3-60 MoE), full causal GQA attention in every layer:
 *
 *   embed ─▶ [ rms ─▶ GQA(NVFP4 K/V) ─▶ +h ─▶ rms ─▶ shared + top-8 of 192 ─▶ +h ] x61 ─▶ rms ─▶ head
 *
 * Every layer keeps every position as NVFP4 pages (1,152 bytes per token and
 * layer, 34.31 GiB at 524,288). The embedding is read from the mapped SSD
 * part row by row; Q8_0 projections, the experts and the attention run on
 * K2's runtime (ds4_k2h_gpu.cuh). */

typedef struct {
    uint32_t ctx_cap, row_cap, position;
    bool ready, failed, borrowed;
    uint64_t workspace_bytes, kv_bytes;
#define K2H_FIELD(field, cols, type) ds4_gpu_tensor *field;
    K2H_ROW_BUFFERS(K2H_FIELD)
#undef K2H_FIELD
    ds4_gpu_tensor *ff_gate, *ff_up, *logits, *partials;
    ds4_gpu_tensor *pages[DS4_MAX_LAYER];
    uint8_t *host_embed;
    uint32_t matrices[K2H_LAYERS][3u * K2H_EXPERTS];
#ifndef __APPLE__
    ds4_gvq_ctx *gvq;
    ds4_k2h_moe *moe;
#endif
} ds4_k2h_graph;

/* Every graph owns its K/V pages. A bank graph borrows everything else from
 * graph 0: bank forwards are serialized, so scratch is shared. */
static void k2h_graph_free(ds4_k2h_graph *g) {
    if (!g) { return; }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) { ds4_gpu_tensor_free(g->pages[il]); }
    if (g->borrowed) {
        memset(g, 0, sizeof(*g));
        return;
    }
#ifndef __APPLE__
    ds4_gpu_k2h_moe_free(g->moe);
    ds4_gpu_gvq_free(g->gvq);
#endif
#define K2H_FREE(field, cols, type) ds4_gpu_tensor_free(g->field);
    K2H_ROW_BUFFERS(K2H_FREE)
#undef K2H_FREE
    ds4_gpu_tensor_free(g->ff_gate);
    ds4_gpu_tensor_free(g->ff_up);
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->partials);
    free(g->host_embed);
    memset(g, 0, sizeof(*g));
}

/* Attention reads only keys at or below a row's position, and every row is
 * written before it is read: a reset only rewinds the position. */
static bool k2h_graph_reset(ds4_k2h_graph *g) {
    if (!g || !g->ready) { return false; }
    if (g->failed && !ds4_gpu_synchronize()) { return false; }
    g->position = 0u;
    g->failed = false;
    return true;
}

typedef enum { K2H_HEAD_LAST, K2H_HEAD_SKIP } k2h_head_mode;

#ifndef __APPLE__
/* Pages start zeroed: a prefill tile reads the raw bytes of keys past the
 * live ones (decoded as zeros, but never NaN scales). */
static bool k2h_graph_kv(ds4_k2h_graph *g) {
    g->kv_bytes = 0;
    const uint64_t bytes = k2h_layer_kv_bytes(g->ctx_cap);
    for (uint32_t il = 0; il < K2H_LAYERS; il++) {
        g->pages[il] = ds4_gpu_tensor_alloc(bytes);
        if (!g->pages[il] || !ds4_gpu_tensor_fill_f32(g->pages[il], 0.0f, bytes / sizeof(float))) { return false; }
        g->kv_bytes += bytes;
    }
    return g->kv_bytes == k2h_kv_bytes(g->ctx_cap);
}

static bool k2h_graph_alloc(ds4_k2h_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t ctx, uint32_t cap) {
    memset(g, 0, sizeof(*g));
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_K2H_VQ || !w->glmvq_gpu || !ctx || !cap || cap > ctx ||
        ctx > K2H_CONTEXT) { return false; }
    g->ctx_cap = ctx;
    g->row_cap = cap;
    g->gvq = ds4_gpu_gvq_create_linear(w->glmvq_gpu, k2h_gvq_scratch(cap), cap);
    g->moe = g->gvq ? ds4_gpu_k2h_moe_create(g->gvq, cap) : NULL;
    if (!g->gvq || !g->moe) { k2h_graph_free(g); return false; }
#define K2H_ALLOC(field, cols, type) do { \
        g->field = ds4_gpu_tensor_alloc((uint64_t)cap * (cols) * sizeof(type)); \
        if (!g->field) { \
            fprintf(stderr, "ds4: K2-Horizon graph allocation failed: %s\n", #field); \
            k2h_graph_free(g); \
            return false; \
        } \
    } while (0);
    K2H_ROW_BUFFERS(K2H_ALLOC)
#undef K2H_ALLOC
    g->ff_gate = ds4_gpu_tensor_alloc(k2h_ff_floats(cap) * sizeof(float));
    g->ff_up = ds4_gpu_tensor_alloc(k2h_ff_floats(cap) * sizeof(float));
    g->logits = ds4_gpu_tensor_alloc((uint64_t)DS4_N_VOCAB * sizeof(float));
    g->partials = ds4_gpu_tensor_alloc(ds4_gpu_k2h_attn_bytes());
    g->host_embed = xmalloc((size_t)cap * DS4_K2H_EMBED_ROW);
    if (!g->ff_gate || !g->ff_up || !g->logits || !g->partials || !k2h_graph_kv(g)) {
        fprintf(stderr, "ds4: K2-Horizon graph allocation failed\n");
        k2h_graph_free(g);
        return false;
    }
    for (uint32_t il = K2H_DENSE; il < K2H_LAYERS; il++) {
        const ds4_k2h_layer *e = &w->k2h[il];
        for (uint32_t x = 0; x < K2H_EXPERTS; x++) {
            g->matrices[il][x] = k2h_index(m, e->gate[x]);
            g->matrices[il][K2H_EXPERTS + x] = k2h_index(m, e->up[x]);
            g->matrices[il][2u * K2H_EXPERTS + x] = k2h_index(m, e->down[x]);
        }
    }
    g->workspace_bytes = k2h_workspace_bytes(cap);
    g->ready = true;
    if (!k2h_graph_reset(g)) { k2h_graph_free(g); return false; }
    ds4_log(stderr, DS4_LOG_TIMING, "ds4: K2-Horizon graph ctx=%u chunk=%u KV=%.2f GiB workspace=%.2f GiB (%llu B)\n",
        ctx, cap, (double)g->kv_bytes / 1073741824.0, (double)g->workspace_bytes / 1073741824.0,
        (unsigned long long)g->workspace_bytes);
    return true;
}

/* DS4_K2H_TRACE=DIR: the forward starting at DS4_K2H_TRACE_POS (default 0)
 * writes each traced tensor's rows as stored (F32, ids I32) to
 * DIR/layer_LL_NAME.bin; DS4_K2H_TRACE_ONLY=LL_NAME keeps one of them. */
static bool k2h_trace(uint32_t il, uint32_t pos, const char *name, const ds4_gpu_tensor *t, uint64_t bytes) {
    const char *root = getenv("DS4_K2H_TRACE"), *at = getenv("DS4_K2H_TRACE_POS");
    if (!root || !root[0] || pos != (at ? (uint32_t)strtoul(at, NULL, 10) : 0u)) { return true; }
    const char *only = getenv("DS4_K2H_TRACE_ONLY");
    char key[64];
    snprintf(key, sizeof(key), "%02u_%s", il, name);
    if (only && only[0] && strcmp(only, key)) { return true; }
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

// A bank graph: its own K/V pages, everything else borrowed from graph 0.
static bool k2h_graph_bank(ds4_k2h_graph *g, const ds4_k2h_graph *owner) {
    *g = *owner;
    g->borrowed = true;
    g->ready = false;
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) { g->pages[il] = NULL; }
    if (!k2h_graph_kv(g)) {
        k2h_graph_free(g);
        return false;
    }
    g->ready = true;
    if (!k2h_graph_reset(g)) { k2h_graph_free(g); return false; }
    return true;
}

static bool k2h_q8(ds4_k2h_graph *g, const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *w,
        const ds4_gpu_tensor *x, uint32_t rows, ds4_k2h_q8_out mode) {
    const bool ok = ds4_gpu_k2h_q8(g->gvq, out, k2h_index(m, w), x, rows, mode) != 0;
    if (!ok) { fprintf(stderr, "ds4: K2-Horizon Q8_0 %.*s failed (rows=%u)\n", (int)w->name.len, w->name.ptr, rows); }
    return ok;
}

// A view of rows [first, first + count) of a buffer `cols` elements wide.
static ds4_gpu_tensor *k2h_rows(ds4_gpu_tensor *t, uint32_t first, uint32_t count, uint64_t cols, uint64_t size) {
    return ds4_gpu_tensor_view(t, (uint64_t)first * cols * size, (uint64_t)count * cols * size);
}

/* q/k/v of norm, their tails (BF16, RoPE, NVFP4 K/V), the attention over
 * every key from 0 and o_proj over the heads. */
static bool k2h_attention(ds4_k2h_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t il, uint32_t rows,
        uint32_t pos) {
    const ds4_layer_weights *l = &w->layer[il];
    return k2h_q8(g, m, g->q, l->attn_q, g->norm, rows, DS4_K2H_Q8_F32) &&
        k2h_q8(g, m, g->k, l->attn_k, g->norm, rows, DS4_K2H_Q8_F32) &&
        k2h_q8(g, m, g->v, l->attn_v, g->norm, rows, DS4_K2H_Q8_F32) &&
        ds4_gpu_k2h_qkv(g->q, g->k, g->v, g->pages[il], g->rope, rows, pos) &&
        k2h_trace(il, pos, "q", g->q, (uint64_t)rows * DS4_K2H_Q * sizeof(float)) &&
        ds4_gpu_k2h_attn(g->heads, g->q, g->pages[il], g->partials, rows, pos) &&
        k2h_trace(il, pos, "heads", g->heads, (uint64_t)rows * DS4_K2H_Q * sizeof(float)) &&
        k2h_q8(g, m, g->attn_out, l->attn_output, g->heads, rows, DS4_K2H_Q8_F32);
}

// SwiGLU MLP of norm rows [first, first + rows): gate and up into the FFN
// buffers, the SwiGLU in place over gate, down into out's rows.
static bool k2h_mlp_rows(ds4_k2h_graph *g, const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *gate,
        const ds4_tensor *up, const ds4_tensor *down, uint32_t first, uint32_t rows, uint32_t width) {
    ds4_gpu_tensor *x = k2h_rows(g->norm, first, rows, DS4_K2H_HIDDEN, sizeof(float));
    ds4_gpu_tensor *y = k2h_rows(out, first, rows, DS4_K2H_HIDDEN, sizeof(float));
    ds4_gpu_tensor *a = ds4_gpu_tensor_view(g->ff_gate, 0, (uint64_t)rows * width * sizeof(float));
    ds4_gpu_tensor *b = ds4_gpu_tensor_view(g->ff_up, 0, (uint64_t)rows * width * sizeof(float));
    const bool ok = x && y && a && b &&
        k2h_q8(g, m, a, gate, x, rows, DS4_K2H_Q8_F32) && k2h_q8(g, m, b, up, x, rows, DS4_K2H_Q8_F32) &&
        ds4_gpu_k2h_swiglu(a, a, b, rows, width) &&
        k2h_q8(g, m, y, down, a, rows, DS4_K2H_Q8_ROUND);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
    ds4_gpu_tensor_free(y);
    ds4_gpu_tensor_free(x);
    return ok;
}

// Dense layers' 16,384-wide rows run in slices that fit the FFN buffers.
static bool k2h_mlp(ds4_k2h_graph *g, const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *gate,
        const ds4_tensor *up, const ds4_tensor *down, uint32_t rows, uint32_t width) {
    const uint32_t slice = width == DS4_K2H_DENSE_FF ? k2h_dense_rows(g->row_cap) : rows;
    for (uint32_t r0 = 0; r0 < rows; r0 += slice) {
        const uint32_t n = rows - r0 < slice ? rows - r0 : slice;
        if (!k2h_mlp_rows(g, m, out, gate, up, down, r0, n, width)) { return false; }
    }
    return true;
}

// Router logits and top-8, the shared expert, then the VQ experts with the
// shared rows in the routed reduce.
static bool k2h_ffn(ds4_k2h_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t il, uint32_t rows,
        uint32_t pos) {
    const ds4_layer_weights *l = &w->layer[il];
    if (il < K2H_DENSE) {
        return k2h_mlp(g, m, g->ffn_out, l->ffn_gate, l->ffn_up, l->ffn_down, rows, DS4_K2H_DENSE_FF);
    }
    const uint64_t route = (uint64_t)rows * DS4_K2H_USED * 4u;
    return ds4_gpu_k2h_router(g->gvq, g->route_ids, g->route_weights, g->router, g->norm,
            k2h_index(m, l->ffn_gate_inp), k2h_index(m, l->ffn_exp_probs_b), rows) &&
        k2h_trace(il, pos, "routes", g->route_ids, route) && k2h_trace(il, pos, "weights", g->route_weights, route) &&
        k2h_mlp(g, m, g->shared, l->ffn_gate_shexp, l->ffn_up_shexp, l->ffn_down_shexp, rows, DS4_K2H_FFN) &&
        ds4_gpu_k2h_moe(g->moe, g->ffn_out, g->norm, g->route_ids, g->route_weights, g->shared, g->matrices[il], rows);
}

/* Layer il over norm = input_layernorm(cur). Each residual add also writes
 * the norm its successor reads: post_attention_layernorm, then the next
 * layer's input_layernorm (or the final norm after the last layer). */
static bool k2h_layer(ds4_k2h_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t il, uint32_t rows,
        uint32_t pos) {
    const ds4_layer_weights *l = &w->layer[il];
    const ds4_tensor *next = il + 1u == K2H_LAYERS ? w->output_norm : w->layer[il + 1u].attn_norm;
    const uint64_t hidden = (uint64_t)rows * DS4_K2H_HIDDEN * sizeof(float);
    const bool ok = k2h_attention(g, m, w, il, rows, pos) &&
        ds4_gpu_k2h_norm(g->gvq, g->norm, g->cur, g->attn_out, k2h_index(m, l->ffn_norm), rows) &&
        k2h_trace(il, pos, "attn", g->cur, hidden) &&
        k2h_ffn(g, m, w, il, rows, pos) &&
        k2h_trace(il, pos, "ffn_out", g->ffn_out, hidden) &&
        ds4_gpu_k2h_norm(g->gvq, g->norm, g->cur, g->ffn_out, k2h_index(m, next), rows) &&
        k2h_trace(il, pos, "hidden", g->cur, hidden);
    if (!ok) { fprintf(stderr, "ds4: K2-Horizon layer %u failed (rows=%u pos=%u)\n", il, rows, pos); }
    return ok;
}

/* Ask the kernel to read `rows` tokens' embedding rows ahead (Beam's
 * prefetch: a cold 2,048-row gather faulted one page at a time). */
static void k2h_embed_prefetch(const ds4_model *m, const ds4_weights *w, const int *tokens, uint32_t rows) {
    const uint64_t row = DS4_K2H_EMBED_ROW, page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint8_t *table = m->map + w->token_embd->abs_offset;
    for (uint32_t r = 0; r < rows; r++) {
        if (tokens[r] < 0 || (uint32_t)tokens[r] >= DS4_N_VOCAB) { continue; }
        const uintptr_t at = (uintptr_t)(table + (uint64_t)tokens[r] * row);
        const uintptr_t first = at / page * page, last = (at + row + page - 1) / page * page;
        (void)madvise((void *)first, last - first, MADV_WILLNEED);
    }
}

// The host gathers Q8_0 rows from the mapped embedding part (SSD-backed
// page cache); the device dequantizes them into cur.
static bool k2h_embed(ds4_k2h_graph *g, const ds4_model *m, const ds4_weights *w, const int *tokens, uint32_t rows) {
    const uint64_t row = DS4_K2H_EMBED_ROW;
    const uint8_t *table = m->map + w->token_embd->abs_offset;
    if (rows > 1u) { k2h_embed_prefetch(m, w, tokens, rows); }
    for (uint32_t r = 0; r < rows; r++) { memcpy(g->host_embed + r * row, table + (uint64_t)tokens[r] * row, row); }
    return ds4_gpu_tensor_write(g->embed, 0u, g->host_embed, (uint64_t)rows * row) &&
        ds4_gpu_k2h_embed(g->cur, g->embed, rows);
}

// The head over the final-normed row `row`.
static bool k2h_head(ds4_k2h_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t row) {
    const uint64_t bytes = (uint64_t)DS4_K2H_HIDDEN * sizeof(float);
    ds4_gpu_tensor *norm = ds4_gpu_tensor_view(g->norm, (uint64_t)row * bytes, bytes);
    const bool ok = norm && k2h_q8(g, m, g->logits, w->output, norm, 1u, DS4_K2H_Q8_F32);
    ds4_gpu_tensor_free(norm);
    return ok;
}

static bool k2h_forward(ds4_k2h_graph *g, const ds4_model *m, const ds4_weights *w, const int *tokens,
        uint32_t rows, uint32_t pos, k2h_head_mode head) {
    if (g->failed || !g->ready || !rows || rows > g->row_cap || pos != g->position || pos > g->ctx_cap ||
        rows > g->ctx_cap - pos) { return false; }
    for (uint32_t i = 0; i < rows; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) { return false; }
    }
    /* A failed enqueue leaves partially written K/V rows; the session must
     * reset before another forward. */
    g->failed = true;
    if (!k2h_embed(g, m, w, tokens, rows) || !ds4_gpu_k2h_rope(g->rope, rows, pos) ||
        !k2h_trace(0u, pos, "embed", g->cur, (uint64_t)rows * DS4_K2H_HIDDEN * sizeof(float)) ||
        !ds4_gpu_k2h_norm(g->gvq, g->norm, g->cur, NULL, k2h_index(m, w->layer[0].attn_norm), rows)) { return false; }
    for (uint32_t il = 0; il < K2H_LAYERS; il++) {
        if (!k2h_layer(g, m, w, il, rows, pos)) { return false; }
    }
    if (head == K2H_HEAD_LAST && (!k2h_head(g, m, w, rows - 1u) ||
            !k2h_trace(K2H_LAYERS, pos, "logits", g->logits, (uint64_t)DS4_N_VOCAB * sizeof(float)))) {
        return false;
    }
    g->position = pos + rows;
    g->failed = false;
    return true;
}
#else
// K2-Horizon hVQ is CUDA-only; Metal keeps linkage and refuses the graph.
static bool k2h_graph_alloc(ds4_k2h_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t ctx, uint32_t cap) {
    (void)g; (void)m; (void)w; (void)ctx; (void)cap;
    return false;
}

static bool k2h_forward(ds4_k2h_graph *g, const ds4_model *m, const ds4_weights *w, const int *tokens,
        uint32_t rows, uint32_t pos, k2h_head_mode head) {
    (void)g; (void)m; (void)w; (void)tokens; (void)rows; (void)pos; (void)head;
    return false;
}

static bool k2h_graph_bank(ds4_k2h_graph *g, const ds4_k2h_graph *owner) {
    (void)g; (void)owner;
    return false;
}

static void k2h_embed_prefetch(const ds4_model *m, const ds4_weights *w, const int *tokens, uint32_t rows) {
    (void)m; (void)w; (void)tokens; (void)rows;
}
#endif
