/* Native GPU state only. Rust owns artifact validation and input processing.
 *
 * MiniMax-M3.1 runs 60 decoder layers over a BF16 residual (Gemma norms,
 * MSA attention on every layer, dense FFN on 0-2, shared + top-4 routed
 * experts on 3-59):
 *
 *   embed ─▶ [ grms ─▶ MSA ─▶ +h ─▶ grms ─▶ dense | shared + MoE ─▶ +h ] x60 ─▶ grms ─▶ head
 *
 *   MSA:  q/k/v (Q4_K), index_q/index_k (Q8_0) ─▶ per-head norm + RoPE
 *         ─▶ K/V and index key into NVFP4 pages ─▶ block scores (max over a
 *         block's tokens of index_q . index_key) ─▶ 16 blocks per KV head
 *         ─▶ attention over those blocks ─▶ o_proj
 *
 * Every layer keeps every position: 72 bytes of index key and 576 bytes of
 * K/V per token, the K/V pages on SSD past the resident context
 * (ds4_minimax_kv.inc). The embedding is read from the mapped SSD part row
 * by row; Q4_K projections run on GLM VQ's kernels and the routed experts
 * on MiniMax's runtime (ds4_minimax_gpu.cuh). */

// MINIMAX_HEAD_ALL (DSpark verify) writes every row's logits to draft->vlogits.
typedef enum { MINIMAX_HEAD_LAST, MINIMAX_HEAD_SKIP, MINIMAX_HEAD_ALL } minimax_head_mode;
typedef struct minimax_draft_rt minimax_draft_rt;

typedef struct {
    uint32_t ctx_cap, row_cap, position;
    // Lowest position whose DSpark context rows are in the draft rings.
    uint32_t draft_first;
    bool ready, failed, borrowed;
    // q4h/q4s hold q4p's prepared copy of norm (the next layer's input norm).
    bool norm_q4p;
    uint64_t workspace_bytes, kv_bytes, score_bytes;
#define MINIMAX_FIELD(field, cols, type) ds4_gpu_tensor *field;
    MINIMAX_ROW_BUFFERS(MINIMAX_FIELD)
#undef MINIMAX_FIELD
    ds4_gpu_tensor *logits, *partials, *scores;
    ds4_gpu_tensor *index[DS4_MAX_LAYER], *pages[DS4_MAX_LAYER];
    uint16_t *host_embed;
    int32_t *host_blocks;   // a layer's selections, read back by the SSD tier
    uint32_t matrices[MINIMAX_LAYERS][3u * MINIMAX_EXPERTS];
#ifndef __APPLE__
    ds4_gvq_ctx *gvq;
    ds4_minimax_moe *moe;
#endif
    minimax_kv kv;
    minimax_draft_rt *draft;   // serial session with the drafter part only
    // A media sync's pad runs (prompt start, count, host rows [count][6144])
    // for the forwards it runs; checkpoint_tag names the media its
    // checkpoint's rows came from (0: none).
    const unsigned *span_start, *span_count;
    const float *const *span_rows;
    uint32_t span_n;
    uint64_t checkpoint_tag;
} ds4_minimax_graph;

// DSpark runtime (ds4_minimax_draft.inc).
static void minimax_draft_free(minimax_draft_rt *d);
static uint64_t minimax_draft_bytes(uint32_t rows);
static minimax_draft_rt *minimax_draft_alloc(const ds4_minimax_draft *w, uint32_t rows);
static bool minimax_drafting(const ds4_engine *e);
static ds4_gpu_tensor *minimax_draft_vlogits(minimax_draft_rt *d);
static bool minimax_draft_tap(ds4_minimax_graph *g, uint32_t il, uint32_t rows);
static bool minimax_draft_context(ds4_minimax_graph *g, const ds4_model *m, uint32_t rows, uint32_t pos);

/* Every graph owns its index keys and K/V pages (or SSD tier). A bank graph
 * borrows everything else from graph 0: bank forwards are serialized, so
 * scratch is shared. */
static void minimax_graph_free(ds4_minimax_graph *g) {
    if (!g) { return; }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor_free(g->index[il]);
        ds4_gpu_tensor_free(g->pages[il]);
    }
#ifndef __APPLE__
    minimax_kv_close(&g->kv);
#endif
    if (g->borrowed) {
        memset(g, 0, sizeof(*g));
        return;
    }
#ifndef __APPLE__
    ds4_gpu_minimax_moe_free(g->moe);
    ds4_gpu_gvq_free(g->gvq);
#endif
#define MINIMAX_FREE(field, cols, type) ds4_gpu_tensor_free(g->field);
    MINIMAX_ROW_BUFFERS(MINIMAX_FREE)
#undef MINIMAX_FREE
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->partials);
    ds4_gpu_tensor_free(g->scores);
    minimax_draft_free(g->draft);
    free(g->host_embed);
    free(g->host_blocks);
    memset(g, 0, sizeof(*g));
}

/* Attention and the indexer read only positions below the row, and every
 * position is written before it is read: a reset only rewinds the position
 * (and drops a failed forward's SSD-tier pins). */
static bool minimax_graph_reset(ds4_minimax_graph *g) {
    if (!g || !g->ready) { return false; }
    if (g->failed && !ds4_gpu_synchronize()) { return false; }
#ifndef __APPLE__
    minimax_kv_end(&g->kv);
    minimax_kv_restart(&g->kv);
#endif
    g->position = 0u;
    g->draft_first = 0u;
    g->failed = false;
    return true;
}

#ifndef __APPLE__
static bool minimax_graph_kv(ds4_minimax_graph *g) {
    g->kv_bytes = minimax_kv_bytes(g->ctx_cap);
    if (!minimax_kv_open(&g->kv, g->ctx_cap)) { return false; }
    for (uint32_t il = 0; il < MINIMAX_LAYERS; il++) {
        g->index[il] = ds4_gpu_tensor_alloc(minimax_index_bytes(g->ctx_cap));
        g->pages[il] = g->kv.spill ? NULL : ds4_gpu_tensor_alloc(minimax_page_bytes(g->ctx_cap));
        if (!g->index[il] || (!g->kv.spill && !g->pages[il])) { return false; }
    }
    return true;
}

static bool minimax_graph_alloc(ds4_minimax_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t ctx, uint32_t cap) {
    memset(g, 0, sizeof(*g));
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_MINIMAX_VQ || !w->glmvq_gpu || !ctx || !cap || cap > ctx ||
        ctx > MINIMAX_CONTEXT) { return false; }
    g->ctx_cap = ctx;
    g->row_cap = cap;
    g->gvq = ds4_gpu_gvq_create_linear(w->glmvq_gpu, minimax_gvq_scratch(cap), cap);
    g->moe = g->gvq ? ds4_gpu_minimax_moe_create(g->gvq, cap) : NULL;
    if (!g->gvq || !g->moe) { minimax_graph_free(g); return false; }
#define MINIMAX_ALLOC(field, cols, type) do { \
        g->field = ds4_gpu_tensor_alloc((uint64_t)cap * (cols) * sizeof(type)); \
        if (!g->field) { \
            fprintf(stderr, "ds4: MiniMax graph allocation failed: %s\n", #field); \
            minimax_graph_free(g); \
            return false; \
        } \
    } while (0);
    MINIMAX_ROW_BUFFERS(MINIMAX_ALLOC)
#undef MINIMAX_ALLOC
    g->score_bytes = minimax_score_bytes(ctx, cap);
    g->logits = ds4_gpu_tensor_alloc((uint64_t)DS4_N_VOCAB * sizeof(float));
    g->partials = ds4_gpu_tensor_alloc((uint64_t)DS4_MINIMAX_SPLIT_ROWS * MINIMAX_SPLITS *
        (MINIMAX_HEADS / MINIMAX_KV_HEADS) * (MINIMAX_HEAD + 2u) * sizeof(float));
    g->scores = ds4_gpu_tensor_alloc(g->score_bytes);
    g->host_embed = xmalloc((size_t)cap * MINIMAX_EMBD * sizeof(uint16_t));
    g->host_blocks = xmalloc((size_t)cap * MINIMAX_KV_HEADS * MINIMAX_TOPK * sizeof(int32_t));
    if (!g->logits || !g->partials || !g->scores || !minimax_graph_kv(g)) {
        fprintf(stderr, "ds4: MiniMax graph allocation failed\n");
        minimax_graph_free(g);
        return false;
    }
    for (uint32_t il = MINIMAX_DENSE; il < MINIMAX_LAYERS; il++) {
        const ds4_minimax_layer *e = &w->minimax[il];
        for (uint32_t x = 0; x < MINIMAX_EXPERTS; x++) {
            g->matrices[il][x] = minimax_index(m, e->w1[x]);
            g->matrices[il][MINIMAX_EXPERTS + x] = minimax_index(m, e->w3[x]);
            g->matrices[il][2u * MINIMAX_EXPERTS + x] = minimax_index(m, e->w2[x]);
        }
    }
    g->workspace_bytes = minimax_workspace_bytes(ctx, cap);
    g->ready = true;
    if (!minimax_graph_reset(g)) { minimax_graph_free(g); return false; }
    ds4_log(stderr, DS4_LOG_TIMING,
        "ds4: MiniMax graph ctx=%u chunk=%u KV=%.2f GiB (%s) workspace=%.2f GiB\n", ctx, cap,
        (double)g->kv_bytes / 1073741824.0, g->kv.spill ? "index keys + SSD-tier pool" : "resident",
        (double)g->workspace_bytes / 1073741824.0);
    return true;
}

/* DS4_MINIMAX_TRACE=DIR: the forward starting at DS4_MINIMAX_TRACE_POS
 * (default 0) writes each traced tensor's rows as stored (F32, ids I32) to
 * DIR/layer_LL_NAME.bin. */
static bool minimax_trace(uint32_t il, uint32_t pos, const char *name, const ds4_gpu_tensor *t, uint64_t bytes) {
    const char *root = getenv("DS4_MINIMAX_TRACE"), *at = getenv("DS4_MINIMAX_TRACE_POS");
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

// A bank graph: its own index keys and K/V pages (or SSD tier), everything
// else borrowed from graph 0.
static bool minimax_graph_bank(ds4_minimax_graph *g, const ds4_minimax_graph *owner) {
    *g = *owner;
    g->borrowed = true;
    g->ready = false;
    g->draft = NULL;   // banks never draft
    memset(&g->kv, 0, sizeof(g->kv));
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        g->index[il] = NULL;
        g->pages[il] = NULL;
    }
    if (!minimax_graph_kv(g)) {
        minimax_graph_free(g);
        return false;
    }
    g->ready = true;
    if (!minimax_graph_reset(g)) { minimax_graph_free(g); return false; }
    return true;
}

/* Decode and verify widths (up to DS4_MINIMAX_SPLIT_ROWS rows) take the
 * one-token kernels for every row: Q4_K here and in minimax_mm_many, the
 * Q8_0 indexer, the router, routed experts and split attention, so a DSpark
 * verify row computes exactly what a decode of that token would. */
static bool minimax_mm(ds4_minimax_graph *g, const ds4_model *m, ds4_gpu_tensor *out,
        const ds4_tensor *w, const ds4_gpu_tensor *x, uint32_t rows) {
    const uint32_t index = minimax_index(m, w);
    const bool ok = (rows <= DS4_MINIMAX_SPLIT_ROWS && ds4_gpu_gvq_q4_decode_rows(g->gvq, out, index, x, rows)) ||
        ds4_gpu_gvq_linear(g->gvq, out, index, x, rows, DS4_GVQ_BF16) != 0;
    if (!ok && w) {
        fprintf(stderr, "ds4: MiniMax linear %.*s failed (rows=%u)\n", (int)w->name.len, w->name.ptr, rows);
    }
    return ok;
}

// Whether q4p runs these matrices at `rows`: then the kernel that writes
// their input also writes q4p's prepared copy into q4h/q4s.
static bool minimax_q4p(ds4_minimax_graph *g, const ds4_model *m, const ds4_tensor *const *ws, uint32_t count,
        uint32_t rows) {
    uint32_t index[4];
    for (uint32_t c = 0; c < count && c < 4u; c++) { index[c] = minimax_index(m, ws[c]); }
    return count <= 4u && ds4_gpu_gvq_q4p_takes(g->gvq, index, count, rows) != 0;
}

// Projections of one input: prefill prepares it once (Q8_1 / FP16), or the
// producer already did (`prepared`: q4h/q4s hold it).
static bool minimax_mm_many(ds4_minimax_graph *g, const ds4_model *m, ds4_gpu_tensor *const *outs,
        const ds4_tensor *const *ws, uint32_t count, const ds4_gpu_tensor *x, uint32_t rows, bool prepared) {
    uint32_t index[4];
    for (uint32_t c = 0; c < count && c < 4u; c++) { index[c] = minimax_index(m, ws[c]); }
    // Decode and verify widths: a single launch streams every matrix (the k/v
    // rows too), each token as a one-token call.
    const bool ok = count <= 4 && ((prepared && ds4_gpu_gvq_q4p_prepared(g->gvq, outs, index, count, g->q4h, g->q4s,
            rows)) || (rows <= DS4_MINIMAX_SPLIT_ROWS &&
            ds4_gpu_gvq_q4_decode_many(g->gvq, outs, index, count, x, rows)) ||
        ds4_gpu_gvq_linear_many(g->gvq, outs, index, count, x, rows, DS4_GVQ_BF16));
    if (!ok) { fprintf(stderr, "ds4: MiniMax linear group of %u failed (rows=%u)\n", count, rows); }
    return ok;
}

// The Q8_0 indexer projections (index_q [512], index_k [128]) of one input:
// one row-stable GEMV launch at decode and verify widths, one MMQ pair (the
// input quantized to Q8_1 once) for prefill rows.
static bool minimax_indexer(ds4_minimax_graph *g, const ds4_model *m, const ds4_layer_weights *l, uint32_t rows) {
    const uint32_t q = minimax_index(m, l->attn_idx_q_b), k = minimax_index(m, l->attn_idx_k);
    const bool ok =
        ds4_gpu_minimax_q8_pair(g->gvq, g->iq, g->ik, q, k, g->norm, MINIMAX_EMBD, MINIMAX_IQ, MINIMAX_INDEX, rows) != 0;
    if (!ok) { fprintf(stderr, "ds4: MiniMax indexer projection failed (rows=%u)\n", rows); }
    return ok;
}

// SSD tier: the layer's selected blocks (or, streaming, every block) into
// the pool before attention.
static bool minimax_resolve(ds4_minimax_graph *g, uint32_t il, uint32_t rows, uint32_t pos) {
    if (!minimax_kv_needs_resolve(&g->kv)) { return true; }
    const uint32_t blocks = (uint32_t)minimax_blocks(pos + rows);
    if (minimax_kv_streams(&g->kv, rows, blocks)) {
        return ds4_gpu_minimax_mark_wait() && minimax_kv_resolve(&g->kv, il, g->kv.all, blocks);
    }
    const uint32_t count = rows * MINIMAX_KV_HEADS * MINIMAX_TOPK;
    return ds4_gpu_tensor_read(g->blocks, 0, g->host_blocks, (uint64_t)count * sizeof(int32_t)) &&
        minimax_kv_resolve(&g->kv, il, g->host_blocks, count);
}

/* MSA: projections, per-head norms with partial RoPE, NVFP4 K/V and index
 * key stores, block selection per KV head, attention over the selected
 * blocks and o_proj. The chunk stores every row's keys before any row reads
 * them: a row only reads keys at or below its own position. */
static bool minimax_attention(ds4_minimax_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t il, uint32_t rows, uint32_t pos) {
    const ds4_layer_weights *l = &w->layer[il];
    const ds4_minimax_layer *x = &w->minimax[il];
    ds4_gpu_tensor *pages = g->kv.spill ? g->kv.pool_gpu : g->pages[il];
    const ds4_gpu_tensor *table = g->kv.spill ? g->kv.rows[il] : NULL;
    ds4_gpu_tensor *const outs[] = {g->q, g->k, g->v};
    const ds4_tensor *const ws[] = {l->attn_q, l->attn_k, l->attn_v};
    const uint64_t kv_row = (uint64_t)MINIMAX_KV_HEADS * MINIMAX_TOPK * sizeof(float);
    const uint32_t blocks_now = (uint32_t)minimax_blocks(pos + rows);
    const uint64_t score_row = (uint64_t)MINIMAX_KV_HEADS * blocks_now * sizeof(float);
    const uint32_t score_rows = (uint32_t)(g->score_bytes / score_row < rows ? g->score_bytes / score_row : rows);
    if (!minimax_mm_many(g, m, outs, ws, 3u, g->norm, rows, g->norm_q4p) ||
        !minimax_indexer(g, m, l, rows) ||
        // v_proj's BF16 output: prefill projections leave F32 sums.
        !ds4_gpu_gvq_round(g->v, (uint64_t)rows * MINIMAX_KV) ||
        !ds4_gpu_minimax_qk(g->gvq, g->q, minimax_index(m, l->attn_q_norm), MINIMAX_HEADS, rows, g->rope) ||
        !ds4_gpu_minimax_qk(g->gvq, g->k, minimax_index(m, l->attn_k_norm), MINIMAX_KV_HEADS, rows, g->rope) ||
        !ds4_gpu_minimax_qk(g->gvq, g->iq, minimax_index(m, x->index_q_norm), MINIMAX_KV_HEADS, rows, g->rope) ||
        !ds4_gpu_minimax_qk(g->gvq, g->ik, minimax_index(m, x->index_k_norm), 1u, rows, g->rope) ||
        !minimax_trace(il, pos, "q", g->q, (uint64_t)rows * MINIMAX_Q * sizeof(float)) ||
        !minimax_trace(il, pos, "k", g->k, (uint64_t)rows * MINIMAX_KV * sizeof(float)) ||
        !minimax_trace(il, pos, "v", g->v, (uint64_t)rows * MINIMAX_KV * sizeof(float)) ||
        !minimax_trace(il, pos, "iq", g->iq, (uint64_t)rows * MINIMAX_IQ * sizeof(float)) ||
        !minimax_trace(il, pos, "ik", g->ik, (uint64_t)rows * MINIMAX_INDEX * sizeof(float)) ||
        !ds4_gpu_minimax_kv_store(pages, table, g->k, g->v, rows, pos) ||
        !ds4_gpu_minimax_index_store(g->index[il], g->ik, rows, pos) ||
        !ds4_gpu_minimax_select(g->blocks, g->scores, score_rows ? score_rows : 1u, g->iq, g->index[il], rows,
            pos) ||
        !minimax_trace(il, pos, "blocks", g->blocks, (uint64_t)rows * kv_row) ||
        !minimax_resolve(g, il, rows, pos) ||
        !ds4_gpu_minimax_attn(g->heads, g->q, pages, table, g->blocks, g->partials, rows, pos) ||
        (g->kv.spill && !ds4_gpu_minimax_mark()) ||
        !minimax_trace(il, pos, "heads", g->heads, (uint64_t)rows * MINIMAX_Q * sizeof(float))) {
        return false;
    }
    return minimax_mm(g, m, g->attn_out, l->attn_output, g->heads, rows) &&
        minimax_trace(il, pos, "attn_out", g->attn_out, (uint64_t)rows * MINIMAX_EMBD * sizeof(float));
}

// SwiGLU-OAI MLP of `width` (dense 12,288 or shared 3,072): gate/up (from
// q4h when `prepared`), the clamped activation into ff_gate or, when q4p
// takes down, straight into q4h, then down.
static bool minimax_mlp(ds4_minimax_graph *g, const ds4_model *m, const ds4_tensor *gate, const ds4_tensor *up,
        const ds4_tensor *down, ds4_gpu_tensor *out, uint32_t width, uint32_t rows, bool prepared) {
    ds4_gpu_tensor *const outs[] = {g->ff_gate, g->ff_up};
    const ds4_tensor *const ws[] = {gate, up};
    if (!minimax_mm_many(g, m, outs, ws, 2u, g->norm, rows, prepared)) { return false; }
    if (minimax_q4p(g, m, &down, 1u, rows)) {
        return ds4_gpu_minimax_swiglu_q4p(g->q4h, g->q4s, g->ff_gate, g->ff_up, rows, width) &&
            minimax_mm_many(g, m, &out, &down, 1u, g->ff_gate, rows, true);
    }
    return ds4_gpu_minimax_swiglu(g->ff_gate, g->ff_gate, g->ff_up, rows, width) &&
        minimax_mm(g, m, out, down, g->ff_gate, rows);
}

// F32 router logits (the source keeps F32 weights), top-4, VQ experts with
// the shared expert added in the routed sum.
static bool minimax_moe(ds4_minimax_graph *g, const ds4_model *m, const ds4_layer_weights *l, uint32_t il,
        uint32_t rows, bool prepared) {
    return minimax_mlp(g, m, l->ffn_gate_shexp, l->ffn_up_shexp, l->ffn_down_shexp, g->shared, MINIMAX_FF, rows,
            prepared) &&
        ds4_gpu_minimax_router(g->gvq, g->selected, g->route_weights, g->router, g->norm,
            minimax_index(m, l->ffn_gate_inp), minimax_index(m, l->ffn_exp_probs_b), rows) &&
        ds4_gpu_minimax_moe(g->moe, g->ffn_out, g->norm, g->selected, g->route_weights, g->shared, g->matrices[il],
            rows);
}

/* Layer il over norm = input_layernorm(cur). Each residual also writes the
 * norm its successor reads: post_attention_layernorm, then `next` (the next
 * layer's input_layernorm; NULL leaves the residual for the head). */
static bool minimax_layer(ds4_minimax_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t il, uint32_t rows, uint32_t pos, const ds4_tensor *next) {
    const ds4_layer_weights *l = &w->layer[il];
    const bool dense = il < MINIMAX_DENSE;
    const uint64_t hidden = (uint64_t)rows * MINIMAX_EMBD * sizeof(float), route = (uint64_t)rows * MINIMAX_USED * 4u;
    // Each norm's q4p consumers (gate/up here, the next layer's q/k/v) read
    // the copy its residual writes when q4p takes them.
    const ds4_tensor *const gu[] = {dense ? l->ffn_gate : l->ffn_gate_shexp, dense ? l->ffn_up : l->ffn_up_shexp};
    const bool gu_q4p = minimax_q4p(g, m, gu, 2u, rows);
    const ds4_layer_weights *n = il + 1u < MINIMAX_LAYERS ? &w->layer[il + 1u] : NULL;
    const ds4_tensor *const qkv[] = {n ? n->attn_q : NULL, n ? n->attn_k : NULL, n ? n->attn_v : NULL};
    const bool qkv_q4p = next && n && minimax_q4p(g, m, qkv, 3u, rows);
    const bool ok = minimax_attention(g, m, w, il, rows, pos) &&
        ds4_gpu_minimax_residual(g->gvq, g->cur, g->attn_out, g->norm, minimax_index(m, l->ffn_norm), rows,
            gu_q4p ? g->q4h : NULL, g->q4s) &&
        minimax_trace(il, pos, "attn", g->cur, hidden) &&
        (dense ? minimax_mlp(g, m, l->ffn_gate, l->ffn_up, l->ffn_down, g->ffn_out, MINIMAX_DENSE_FF, rows, gu_q4p)
               : minimax_moe(g, m, l, il, rows, gu_q4p) && minimax_trace(il, pos, "routes", g->selected, route) &&
                     minimax_trace(il, pos, "weights", g->route_weights, route)) &&
        minimax_trace(il, pos, "ffn_out", g->ffn_out, hidden) &&
        ds4_gpu_minimax_residual(g->gvq, g->cur, g->ffn_out, g->norm, next ? minimax_index(m, next) : DS4_GVQ_NO_LINK,
            rows, qkv_q4p ? g->q4h : NULL, g->q4s) &&
        minimax_trace(il, pos, "ffn", g->cur, hidden);
    g->norm_q4p = ok && qkv_q4p;
    if (!ok) { fprintf(stderr, "ds4: MiniMax layer %u failed (rows=%u pos=%u)\n", il, rows, pos); }
    return ok;
}

/* Media rows over the embedded rows pos..pos+rows-1: each pad of a span
 * takes its row from the vision tower's output. */
static bool minimax_apply_spans(ds4_minimax_graph *g, uint32_t rows, uint32_t pos) {
    const uint64_t row = (uint64_t)MINIMAX_EMBD * sizeof(float);
    for (uint32_t i = 0; i < g->span_n; i++) {
        const uint32_t start = g->span_start[i], stop = start + g->span_count[i];
        const uint32_t from = start > pos ? start : pos, to = stop < pos + rows ? stop : pos + rows;
        if (from >= to) { continue; }
        ds4_gpu_tensor *view = ds4_gpu_tensor_view(g->cur, (uint64_t)(from - pos) * row, (uint64_t)(to - from) * row);
        const bool ok = view && ds4_gpu_tensor_write(view, 0, g->span_rows[i] + (uint64_t)(from - start) * MINIMAX_EMBD,
            (uint64_t)(to - from) * row);
        ds4_gpu_tensor_free(view);
        if (!ok) { return false; }
    }
    return true;
}

/* Ask the kernel to read `rows` tokens' embedding rows ahead (Beam's R5):
 * the prefill loop issues the next chunk's rows before it runs the current
 * one, and a gather first queues all of its own rows. Advisory only. */
static void minimax_embed_prefetch(const ds4_model *m, const ds4_weights *w, const int *tokens, uint32_t rows) {
    const uint64_t row = (uint64_t)MINIMAX_EMBD * sizeof(uint16_t);
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
// page cache); the device widens them into `out`.
static bool minimax_embed(ds4_minimax_graph *g, const ds4_model *m, const ds4_weights *w,
        const int *tokens, uint32_t rows, ds4_gpu_tensor *out) {
    const uint64_t row = (uint64_t)MINIMAX_EMBD * sizeof(uint16_t);
    const uint8_t *table = m->map + w->token_embd->abs_offset;
    if (rows > 1u) { minimax_embed_prefetch(m, w, tokens, rows); }
    for (uint32_t r = 0; r < rows; r++) {
        memcpy((uint8_t *)g->host_embed + r * row, table + (uint64_t)tokens[r] * row, row);
    }
    return ds4_gpu_tensor_write(g->embed, 0u, g->host_embed, (uint64_t)rows * row) &&
        ds4_gpu_minimax_embed(out, g->embed, rows);
}

// Final norm and head of `row`.
static bool minimax_head(ds4_minimax_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t row) {
    const uint64_t bytes = (uint64_t)MINIMAX_EMBD * sizeof(float);
    ds4_gpu_tensor *cur = ds4_gpu_tensor_view(g->cur, (uint64_t)row * bytes, bytes);
    ds4_gpu_tensor *norm = ds4_gpu_tensor_view(g->norm, 0u, bytes);
    const bool ok = cur && norm && ds4_gpu_minimax_rms(g->gvq, norm, cur, minimax_index(m, w->output_norm), 1u) &&
        minimax_mm(g, m, g->logits, w->output, norm, 1u) &&
        ds4_gpu_minimax_logits(g->logits, 1u, MINIMAX_TOKENS);
    ds4_gpu_tensor_free(norm);
    ds4_gpu_tensor_free(cur);
    return ok;
}

static bool minimax_forward(ds4_minimax_graph *g, const ds4_model *m, const ds4_weights *w,
        const int *tokens, uint32_t rows, uint32_t pos, minimax_head_mode head) {
    if (g->failed || !g->ready || !rows || rows > g->row_cap || pos != g->position ||
        pos > g->ctx_cap || rows > g->ctx_cap - pos) { return false; }
    for (uint32_t i = 0; i < rows; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) { return false; }
    }
    /* A failed enqueue leaves partially written K/V; the session must reset
     * before another forward. */
    g->failed = true;
    if (!minimax_kv_begin(&g->kv, rows, pos) || !minimax_embed(g, m, w, tokens, rows, g->cur) ||
        !minimax_apply_spans(g, rows, pos) || !ds4_gpu_minimax_rope(g->rope, rows, pos) ||
        !minimax_trace(0u, pos, "embed", g->cur, (uint64_t)rows * MINIMAX_EMBD * sizeof(float)) ||
        !ds4_gpu_minimax_rms(g->gvq, g->norm, g->cur, minimax_index(m, w->layer[0].attn_norm), rows)) {
        return false;
    }
    g->norm_q4p = false;   // layer 0's q/k/v prepare their input themselves
    // DS4_MINIMAX_DSPARK_TAP=entry taps the residual entering a target layer
    // (TokenSpeed's reading); the default takes it after the layer (DFlash's
    // hidden_states[layer_id + 1], vLLM's boundary idx + 1, llama.cpp's
    // target_layers = id + 1 layer inputs); =norm also
    // final-norms the last layer's tap (HF hidden_states[-1]).
    static int tap_mode = -1;
    if (tap_mode < 0) {
        const char *v = getenv("DS4_MINIMAX_DSPARK_TAP");
        tap_mode = v && !strcmp(v, "entry") ? 1 : v && !strcmp(v, "norm") ? 2 : 0;
    }
    for (uint32_t il = 0; il < MINIMAX_LAYERS; il++) {
        const ds4_tensor *next = il + 1u < MINIMAX_LAYERS ? w->layer[il + 1u].attn_norm : NULL;
        if ((tap_mode == 1 && !minimax_draft_tap(g, il, rows)) || !minimax_layer(g, m, w, il, rows, pos, next)) {
            return false;
        }
        if (tap_mode != 1 && il + 1u == MINIMAX_LAYERS && tap_mode == 2 && g->draft) {
            // The last tap after the final norm: norm(cur) through the tap path.
            if (!ds4_gpu_minimax_rms(g->gvq, g->norm, g->cur, minimax_index(m, w->output_norm), rows)) { return false; }
            ds4_gpu_tensor *keep = g->cur;
            g->cur = g->norm;
            const bool ok = minimax_draft_tap(g, il, rows);
            g->cur = keep;
            if (!ok) { return false; }
        } else if (tap_mode != 1 && !minimax_draft_tap(g, il, rows)) {
            return false;
        }
    }
    if (!minimax_draft_context(g, m, rows, pos)) { return false; }
    if (head == MINIMAX_HEAD_LAST && (!minimax_head(g, m, w, rows - 1u) ||
            !minimax_trace(MINIMAX_LAYERS, pos, "logits", g->logits, (uint64_t)DS4_N_VOCAB * sizeof(float)))) {
        return false;
    }
    if (head == MINIMAX_HEAD_ALL && (!g->draft || rows > MINIMAX_DRAFT_BLOCK + 1u ||
            !ds4_gpu_minimax_rms(g->gvq, g->norm, g->cur, minimax_index(m, w->output_norm), rows) ||
            !minimax_mm(g, m, minimax_draft_vlogits(g->draft), w->output, g->norm, rows) ||
            !ds4_gpu_minimax_logits(minimax_draft_vlogits(g->draft), rows, MINIMAX_TOKENS))) {
        return false;
    }
    minimax_kv_end(&g->kv);
    g->position = pos + rows;
    g->failed = false;
    return true;
}
#else
// MiniMax is CUDA-only; Metal keeps linkage and refuses the graph.
static bool minimax_graph_alloc(ds4_minimax_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t ctx, uint32_t cap) {
    (void)g; (void)m; (void)w; (void)ctx; (void)cap;
    return false;
}

static bool minimax_forward(ds4_minimax_graph *g, const ds4_model *m, const ds4_weights *w,
        const int *tokens, uint32_t rows, uint32_t pos, minimax_head_mode head) {
    (void)g; (void)m; (void)w; (void)tokens; (void)rows; (void)pos; (void)head;
    return false;
}

static bool minimax_graph_bank(ds4_minimax_graph *g, const ds4_minimax_graph *owner) {
    (void)g; (void)owner;
    return false;
}

static void minimax_embed_prefetch(const ds4_model *m, const ds4_weights *w, const int *tokens, uint32_t rows) {
    (void)m; (void)w; (void)tokens; (void)rows;
}
#endif
