/* Native GPU state only. Rust owns artifact validation and input processing.
 *
 * Inkling-2-Small runs 48 decoder layers over a BF16 residual; layer l reads
 * the inputs of the block-boundary layers before it (0, 6, ..., 42) through
 * learned queries (block attention residuals):
 *
 *   embed ─▶ [ attn-res ─▶ q k v r gate sink ─▶ conv ─▶ attention ─▶ o ─▶ conv
 *              ─▶ (+h | restart at a boundary) ─▶ attn-res ─▶ router ─▶ experts
 *              ─▶ conv ─▶ +h ] x48 ─▶ attn-res ─▶ norm ─▶ head
 *
 * Full-attention layers (5, 11, ..., 47) keep every K/V row and take q, k
 * and v inputs from three queries; sliding layers keep a ring that holds the
 * 1,024-key window behind a whole chunk. Each layer also carries four 4-tap
 * convolution histories (K and V inputs, attention and MLP outputs). The
 * embedding is read from the mapped SSD part row by row; Q4_K projections
 * run on GLM VQ's kernels and the routed experts on Inkling 2's runtime
 * (ds4_inkling2_gpu.cuh). */

typedef struct inkling2_draft_rt inkling2_draft_rt;

// The open drafts with the release's drafter part (--mtp-draft above 1).
static bool inkling2_drafting(const ds4_engine *e) {
    return e->weights.inkling2_draft && e->mtp_draft_tokens > 1 && !getenv("DS4_MTP_SPEC_DISABLE");
}
#ifndef __APPLE__
static void inkling2_draft_free(inkling2_draft_rt *d);
static void inkling2_draft_reset(inkling2_draft_rt *d);
#endif

// Rows a forward sends through the head: none, the last (logits), every row
// (a trial's verify logits, into the drafter's rows).
typedef enum { INKLING2_HEAD_NONE, INKLING2_HEAD_LAST, INKLING2_HEAD_ALL } inkling2_head_rows;

typedef struct {
    // high: one past the highest row any forward of this lineage wrote (the
    // draft rings' displacement bound for a rewind).
    uint32_t ctx_cap, row_cap, ring_cap, position, high;
    bool ready, failed;
    uint64_t workspace_bytes, kv_bytes;
#define INKLING2_FIELD(field, cols, type) ds4_gpu_tensor *field;
    INKLING2_ROW_BUFFERS(INKLING2_FIELD)
#undef INKLING2_FIELD
    ds4_gpu_tensor *logits, *partials;
    ds4_gpu_tensor *kv[INKLING2_LAYERS], *hist[INKLING2_LAYERS];
    // SSD tier of the full layers past the resident context, and the
    // attention state its key segments carry (NULL without a tier).
    inkling2_kv tier;
    ds4_gpu_tensor *state;
    // DSpark drafter runtime (ds4_inkling2_draft.inc), when the open drafts.
    inkling2_draft_rt *draft;
    uint16_t *host_embed;
    uint32_t matrices[INKLING2_LAYERS][3u * DS4_INK2_EXPERTS];
#ifndef __APPLE__
    ds4_gvq_ctx *gvq;
    ds4_ink2_moe *moe;
#endif
} ds4_inkling2_graph;

static void inkling2_graph_free(ds4_inkling2_graph *g) {
    if (!g) { return; }
    for (uint32_t il = 0; il < INKLING2_LAYERS; il++) {
        ds4_gpu_tensor_free(g->kv[il]);
        ds4_gpu_tensor_free(g->hist[il]);
    }
#ifndef __APPLE__
    ds4_gpu_ink2_moe_free(g->moe);
    ds4_gpu_gvq_free(g->gvq);
#endif
#define INKLING2_FREE(field, cols, type) ds4_gpu_tensor_free(g->field);
    INKLING2_ROW_BUFFERS(INKLING2_FREE)
#undef INKLING2_FREE
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->partials);
    ds4_gpu_tensor_free(g->state);
    inkling2_kv_close(&g->tier);
#ifndef __APPLE__
    inkling2_draft_free(g->draft);
#endif
    free(g->host_embed);
    memset(g, 0, sizeof(*g));
}

/* Attention reads only rows at or below the query and every row is written
 * before it is read; the convolution histories restart empty. */
static bool inkling2_graph_reset(ds4_inkling2_graph *g) {
    if (!g || !g->ready) { return false; }
    if (g->failed && !ds4_gpu_synchronize()) { return false; }
    for (uint32_t il = 0; il < INKLING2_LAYERS; il++) {
        if (!ds4_gpu_tensor_fill_f32(g->hist[il], 0.0f, INKLING2_HIST)) { return false; }
    }
    inkling2_kv_restart(&g->tier);
#ifndef __APPLE__
    inkling2_draft_reset(g->draft);
#endif
    g->position = 0u;
    g->high = 0u;
    g->failed = false;
    return true;
}

#ifndef __APPLE__
// One layer's convolution history of stream s (0: K input, 1: V input,
// 2: attention output, 3: MLP output).
static ds4_gpu_tensor *inkling2_hist(ds4_inkling2_graph *g, uint32_t il, uint32_t stream) {
    static const uint64_t offset[4] = {0, DS4_INK2_KV, 2u * DS4_INK2_KV, 2u * DS4_INK2_KV + DS4_INK2_HIDDEN};
    static const uint64_t width[4] = {DS4_INK2_KV, DS4_INK2_KV, DS4_INK2_HIDDEN, DS4_INK2_HIDDEN};
    return ds4_gpu_tensor_view(g->hist[il], offset[stream] * DS4_INK2_HISTORY * sizeof(float),
        width[stream] * DS4_INK2_HISTORY * sizeof(float));
}

static bool inkling2_graph_alloc(ds4_inkling2_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t ctx, uint32_t cap) {
    memset(g, 0, sizeof(*g));
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_INKLING2_VQ || !w->glmvq_gpu || !w->inkling2 || !ctx || !cap ||
        cap > ctx || ctx > INKLING2_CONTEXT) { return false; }
    g->ctx_cap = ctx;
    g->row_cap = cap;
    g->ring_cap = inkling2_ring(cap);
    g->gvq = ds4_gpu_gvq_create_linear(w->glmvq_gpu, inkling2_gvq_scratch(cap), cap);
    g->moe = g->gvq ? ds4_gpu_ink2_moe_create(g->gvq, cap) : NULL;
    if (!g->gvq || !g->moe) { inkling2_graph_free(g); return false; }
#define INKLING2_ALLOC(field, cols, type) do { \
        g->field = ds4_gpu_tensor_alloc((uint64_t)cap * (cols) * sizeof(type)); \
        if (!g->field) { \
            fprintf(stderr, "ds4: Inkling-2 graph allocation failed: %s\n", #field); \
            inkling2_graph_free(g); \
            return false; \
        } \
    } while (0);
    INKLING2_ROW_BUFFERS(INKLING2_ALLOC)
#undef INKLING2_ALLOC
    g->logits = ds4_gpu_tensor_alloc((uint64_t)DS4_N_VOCAB * sizeof(float));
    g->partials = ds4_gpu_tensor_alloc(ds4_gpu_ink2_attn_bytes());
    g->host_embed = xmalloc((size_t)cap * DS4_INK2_HIDDEN * sizeof(uint16_t));
    bool ok = g->logits && g->partials;
    for (uint32_t il = 0; ok && il < INKLING2_LAYERS; il++) {
        g->kv[il] = ds4_gpu_tensor_alloc(inkling2_layer_kv_bytes(il, ctx, cap));
        g->hist[il] = ds4_gpu_tensor_alloc((uint64_t)INKLING2_HIST * sizeof(float));
        ok = g->kv[il] && g->hist[il];
    }
    if (ok && inkling2_spill(ctx)) {
        g->state = ds4_gpu_tensor_alloc(ds4_gpu_ink2_state_bytes(cap));
        ok = g->state && inkling2_kv_open(&g->tier, ctx, cap);
    }
    if (!ok) {
        fprintf(stderr, "ds4: Inkling-2 graph allocation failed\n");
        inkling2_graph_free(g);
        return false;
    }
    for (uint32_t il = 0; il < INKLING2_LAYERS; il++) {
        const ds4_inkling2_layer *l = &w->inkling2->layer[il];
        for (uint32_t e = 0; e < DS4_INK2_EXPERTS; e++) {
            g->matrices[il][e] = inkling2_index(m, l->gate_e[e]);
            g->matrices[il][DS4_INK2_EXPERTS + e] = inkling2_index(m, l->up_e[e]);
            g->matrices[il][2u * DS4_INK2_EXPERTS + e] = inkling2_index(m, l->down_e[e]);
        }
    }
    g->kv_bytes = inkling2_kv_bytes(ctx, cap);
    g->workspace_bytes = inkling2_workspace_bytes(cap);
    g->ready = true;
    if (!inkling2_graph_reset(g)) { inkling2_graph_free(g); return false; }
    ds4_log(stderr, DS4_LOG_TIMING, "ds4: Inkling-2 graph ctx=%u chunk=%u KV=%.2f GiB workspace=%.2f GiB\n",
        ctx, cap, (double)g->kv_bytes / 1073741824.0, (double)g->workspace_bytes / 1073741824.0);
    return true;
}

// Projections of one input (Q4_K and BF16 records, at most four).
static bool inkling2_mm(ds4_inkling2_graph *g, const ds4_model *m, ds4_gpu_tensor *const *outs,
        const ds4_tensor *const *ws, uint32_t count, const ds4_gpu_tensor *x, uint32_t rows) {
    uint32_t index[4];
    for (uint32_t c = 0; c < count; c++) { index[c] = inkling2_index(m, ws[c]); }
    const bool ok = count <= 4 && ds4_gpu_gvq_linear_many(g->gvq, outs, index, count, x, rows, DS4_GVQ_BF16);
    if (!ok) { fprintf(stderr, "ds4: Inkling-2 projection group of %u failed (rows=%u)\n", count, rows); }
    return ok;
}

static bool inkling2_mm1(ds4_inkling2_graph *g, const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *w,
        const ds4_gpu_tensor *x, uint32_t rows, ds4_gvq_math math) {
    const bool ok = ds4_gpu_gvq_linear(g->gvq, out, inkling2_index(m, w), x, rows, math) != 0;
    if (!ok) { fprintf(stderr, "ds4: Inkling-2 linear %.*s failed (rows=%u)\n", (int)w->name.len, w->name.ptr, rows); }
    return ok;
}

/* A full layer's attention under the SSD tier, key segments in order:
 * the resident blocks, the file blocks below the window (read
 * INKLING2_RING at a time), then the window. A forward inside the resident
 * blocks, or with nothing below the window, is one ordinary attention. */
static bool inkling2_tier_attn(ds4_inkling2_graph *g, uint32_t il, uint32_t rows, uint32_t pos) {
    inkling2_kv *kv = &g->tier;
    const uint32_t end = pos + rows, resident_end = kv->resident * DS4_INK2_KV_BLOCK;
    const uint32_t w0 = kv->window_block0;
    ds4_gpu_tensor *window = kv->window_gpu[il / DS4_INK2_BLOCK];
    if (end <= resident_end) {
        return ds4_gpu_ink2_attn(g->heads, g->q, g->rel, g->sink, g->gate, g->kv[il], resident_end, 0u,
            DS4_INK2_KV_NVFP4, g->partials, rows, pos);
    }
    if (w0 == 0) {
        return ds4_gpu_ink2_attn(g->heads, g->q, g->rel, g->sink, g->gate, window,
            kv->window_blocks * DS4_INK2_KV_BLOCK, 0u, DS4_INK2_KV_NVFP4, g->partials, rows, pos);
    }
    int mode = DS4_INK2_SEG_FIRST;
    if (kv->resident) {
        if (!ds4_gpu_ink2_attn_seg(g->heads, g->state, g->q, g->rel, g->sink, g->gate, g->kv[il], 0u,
                resident_end, 0u, g->partials, rows, pos, mode)) { return false; }
        mode = DS4_INK2_SEG_NEXT;
    }
    for (uint32_t b = kv->resident; b < w0; b += INKLING2_RING) {
        const uint32_t n = w0 - b < INKLING2_RING ? w0 - b : INKLING2_RING;
        ds4_gpu_tensor *store = inkling2_kv_read(kv, il, b, n);
        if (!store || !ds4_gpu_ink2_attn_seg(g->heads, g->state, g->q, g->rel, g->sink, g->gate, store,
                b * DS4_INK2_KV_BLOCK, (b + n) * DS4_INK2_KV_BLOCK, b, g->partials, rows, pos, mode) ||
            !inkling2_kv_release(kv)) { return false; }
        mode = DS4_INK2_SEG_NEXT;
    }
    return ds4_gpu_ink2_attn_seg(g->heads, g->state, g->q, g->rel, g->sink, g->gate, window,
        w0 * DS4_INK2_KV_BLOCK, end, w0, g->partials, rows, pos, DS4_INK2_SEG_LAST);
}

/* Attention half: attention-residual inputs, projections, K/V tails and
 * stores, q/r tails and profiles, attention with the sink and gate, o_proj,
 * the output convolution and the residual (restarted at a boundary). */
static bool inkling2_attention(ds4_inkling2_graph *g, const ds4_model *m, const ds4_inkling2_weights *w,
        uint32_t il, uint32_t rows, uint32_t pos, uint32_t nblocks) {
    const ds4_inkling2_layer *l = &w->layer[il];
    const bool full = inkling2_global(il);
    ds4_gpu_tensor *const in3[] = {g->xq, g->xk, g->xv};
    const uint32_t norms[] = {inkling2_index(m, l->norm[0]), inkling2_index(m, l->norm[1]),
        inkling2_index(m, l->norm[2])};
    if (!ds4_gpu_ink2_attn_res(g->gvq, in3, g->blocks, nblocks, g->row_cap, g->x, inkling2_index(m, l->attn_query),
            full ? 3u : 1u, norms, rows)) {
        fprintf(stderr, "ds4: Inkling-2 layer %u attention input failed\n", il);
        return false;
    }
    const ds4_gpu_tensor *xk = full ? g->xk : g->xq, *xv = full ? g->xv : g->xq;
    ds4_gpu_tensor *const qouts[] = {g->q, g->gate, g->r};
    const ds4_tensor *const qws[] = {l->q, l->gate, l->r};
    if (!inkling2_mm(g, m, qouts, qws, 3u, g->xq, rows) ||
        !inkling2_mm1(g, m, g->sink, l->sink, g->xq, rows, DS4_GVQ_BF16)) { return false; }
    if (full) {
        ds4_gpu_tensor *const kouts[] = {g->kin};
        ds4_gpu_tensor *const vouts[] = {g->vin};
        const ds4_tensor *const kw[] = {l->k}, *const vw[] = {l->v};
        if (!inkling2_mm(g, m, kouts, kw, 1u, xk, rows) || !inkling2_mm(g, m, vouts, vw, 1u, xv, rows) ||
            !inkling2_mm1(g, m, g->dynk, l->k_dyn, xk, rows, DS4_GVQ_BF16) ||
            !inkling2_mm1(g, m, g->dynv, l->v_dyn, xv, rows, DS4_GVQ_BF16)) { return false; }
    } else {
        ds4_gpu_tensor *const kvouts[] = {g->kin, g->vin};
        const ds4_tensor *const kvws[] = {l->k, l->v};
        if (!inkling2_mm(g, m, kvouts, kvws, 2u, g->xq, rows)) { return false; }
    }
    const uint32_t taps[] = {inkling2_index(m, l->k_taps), inkling2_index(m, l->v_taps)};
    ds4_gpu_tensor *hk = inkling2_hist(g, il, 0), *hv = inkling2_hist(g, il, 1), *ha = inkling2_hist(g, il, 2);
    const uint32_t cap = full ? g->ctx_cap : g->ring_cap, window = full ? 0u : DS4_INK2_WINDOW;
    const uint32_t format = full ? inkling2_full_format() : DS4_INK2_KV_BF16;
    ds4_ink2_tier tier;
    const bool tiered = full && inkling2_kv_tier(&g->tier, il, &tier);
    const bool ok = hk && hv && ha &&
        ds4_gpu_ink2_kv(g->gvq, g->kv[il], cap, format, g->kin, g->vin, hk, hv, taps,
            full ? g->dynk : NULL, full ? g->dynv : NULL, rows, pos, tiered ? &tier : NULL) &&
        ds4_gpu_ink2_qr(g->gvq, g->q, g->r, g->rel, inkling2_index(m, l->q_scale), inkling2_index(m, l->r_scale),
            inkling2_index(m, l->rel_proj), rows, pos, full) &&
        (tiered ? inkling2_tier_attn(g, il, rows, pos)
                : ds4_gpu_ink2_attn(g->heads, g->q, g->rel, g->sink, g->gate, g->kv[il], cap, window, format,
                    g->partials, rows, pos)) &&
        inkling2_mm1(g, m, g->attn_out, l->o, g->heads, rows, DS4_GVQ_BF16) &&
        // A block boundary stores its input as a block state and restarts
        // the residual at the attention output.
        (!inkling2_boundary(il) || ds4_gpu_ink2_push(g->blocks, nblocks, g->row_cap, g->x, rows)) &&
        ds4_gpu_ink2_conv(g->gvq, g->resid, g->attn_out, ha, inkling2_index(m, l->attn_conv),
            inkling2_boundary(il) ? NULL : g->x, rows, 0);
    ds4_gpu_tensor_free(hk);
    ds4_gpu_tensor_free(hv);
    ds4_gpu_tensor_free(ha);
    if (!ok) { fprintf(stderr, "ds4: Inkling-2 layer %u attention failed (rows=%u pos=%u)\n", il, rows, pos); }
    return ok;
}

/* MLP half: attention-residual input with mlp_norm, the F32 router, the
 * routed experts, the MLP output convolution and the BF16 residual. */
static bool inkling2_mlp(ds4_inkling2_graph *g, const ds4_model *m, const ds4_inkling2_weights *w, uint32_t il,
        uint32_t rows, uint32_t pos, uint32_t nblocks) {
    const ds4_inkling2_layer *l = &w->layer[il];
    ds4_gpu_tensor *const in1[] = {g->xq};
    const uint32_t norm = inkling2_index(m, l->mlp_norm);
    ds4_gpu_tensor *hm = inkling2_hist(g, il, 3);
    const bool ok = hm &&
        ds4_gpu_ink2_attn_res(g->gvq, in1, g->blocks, nblocks, g->row_cap, g->resid, inkling2_index(m, l->mlp_query),
            1u, &norm, rows) &&
        inkling2_mm1(g, m, g->router, l->router, g->xq, rows, DS4_GVQ_F32) &&
        ds4_gpu_ink2_route(g->gvq, g->selected, g->route_w, g->router, inkling2_index(m, l->router_bias), rows) &&
        ds4_gpu_ink2_moe(g->moe, g->moe_out, g->xq, g->selected, g->route_w, g->matrices[il], rows) &&
        ds4_gpu_ink2_conv(g->gvq, g->x, g->moe_out, hm, inkling2_index(m, l->mlp_conv), g->resid, rows, 1);
    ds4_gpu_tensor_free(hm);
    if (!ok) { fprintf(stderr, "ds4: Inkling-2 layer %u MLP failed (rows=%u pos=%u)\n", il, rows, pos); }
    return ok;
}

/* Ask the kernel to read `rows` tokens' embedding rows ahead: the prefill
 * loop issues the next chunk's rows before it runs the current one.
 * Advisory only. */
static void inkling2_embed_prefetch(const ds4_model *m, const ds4_weights *w, const int *tokens, uint32_t rows) {
    const uint64_t row = (uint64_t)DS4_INK2_HIDDEN * sizeof(uint16_t);
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
// page cache); the device normalizes them into x.
static bool inkling2_embed(ds4_inkling2_graph *g, const ds4_model *m, const ds4_weights *w, const int *tokens,
        uint32_t rows) {
    const uint64_t row = (uint64_t)DS4_INK2_HIDDEN * sizeof(uint16_t);
    const uint8_t *table = m->map + w->token_embd->abs_offset;
    if (rows > 1u) { inkling2_embed_prefetch(m, w, tokens, rows); }
    for (uint32_t r = 0; r < rows; r++) {
        memcpy((uint8_t *)g->host_embed + r * row, table + (uint64_t)tokens[r] * row, row);
    }
    return ds4_gpu_tensor_write(g->embed, 0u, g->host_embed, (uint64_t)rows * row) &&
        ds4_gpu_ink2_embed(g->gvq, g->x, g->embed, inkling2_index(m, w->inkling2->embed_norm), rows);
}

// Final attention residual, norm and head of `count` rows from `row` on.
static bool inkling2_head(ds4_inkling2_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t row,
        uint32_t count, ds4_gpu_tensor *logits) {
    const uint64_t hidden = (uint64_t)DS4_INK2_HIDDEN * sizeof(float);
    const uint32_t norm = inkling2_index(m, w->output_norm);
    ds4_gpu_tensor *const out[] = {g->xq};
    // Block states of the same rows: one view per slot would not be
    // contiguous, so the final residual runs over every row up to the last
    // and the head reads the rows it needs.
    bool ok = ds4_gpu_ink2_attn_res(g->gvq, out, g->blocks, DS4_INK2_BLOCKS, g->row_cap, g->x,
        inkling2_index(m, w->inkling2->out_query), 1u, &norm, row + count);
    ds4_gpu_tensor *final = ok ? ds4_gpu_tensor_view(g->xq, (uint64_t)row * hidden, (uint64_t)count * hidden) : NULL;
    ok = ok && final && inkling2_mm1(g, m, logits, w->output, final, count, DS4_GVQ_BF16) &&
        ds4_gpu_ink2_logits(logits, count);
    ds4_gpu_tensor_free(final);
    return ok;
}

#include "ds4_inkling2_draft.inc"

static bool inkling2_forward(ds4_inkling2_graph *g, const ds4_model *m, const ds4_weights *w, const int *tokens,
        uint32_t rows, uint32_t pos, inkling2_head_rows head) {
    if (g->failed || !g->ready || !rows || rows > g->row_cap || pos != g->position || pos > g->ctx_cap ||
        rows > g->ctx_cap - pos) { return false; }
    for (uint32_t i = 0; i < rows; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) { return false; }
    }
    /* A failed enqueue leaves partially written K/V and histories; the
     * session must reset before another forward. */
    g->failed = true;
    if (!inkling2_kv_begin(&g->tier, rows, pos)) { return false; }
    if (!inkling2_embed(g, m, w, tokens, rows)) { return false; }
    const ds4_inkling2_weights *k = w->inkling2;
    for (uint32_t il = 0; il < INKLING2_LAYERS; il++) {
        // Block states before this layer: the inputs of boundaries below it.
        const uint32_t before = (il + DS4_INK2_BLOCK - 1u) / DS4_INK2_BLOCK;
        const uint32_t after = before + (inkling2_boundary(il) ? 1u : 0u);
        if (!inkling2_draft_tap(g, il, rows) || !inkling2_attention(g, m, k, il, rows, pos, before) || !inkling2_mlp(g, m, k, il, rows, pos, after) ||
            !inkling2_draft_keep(g, il, rows)) {
            return false;
        }
    }
    if (head == INKLING2_HEAD_LAST && !inkling2_head(g, m, w, rows - 1u, 1u, g->logits)) { return false; }
    if (head == INKLING2_HEAD_ALL && (!g->draft || rows > DS4_INK2_DRAFT_BLOCK ||
            !inkling2_head(g, m, w, 0u, rows, g->draft->vlogits))) {
        return false;
    }
    // The normalized output of every row (the head computed it above), then
    // the drafter's context; a trial picks its anchor row at commit.
    if (g->draft) {
        ds4_gpu_tensor *const out[] = {g->xq};
        const uint32_t norm = inkling2_index(m, w->output_norm);
        if ((head == INKLING2_HEAD_NONE && !ds4_gpu_ink2_attn_res(g->gvq, out, g->blocks, DS4_INK2_BLOCKS, g->row_cap,
                g->x, inkling2_index(m, w->inkling2->out_query), 1u, &norm, rows)) ||
            !inkling2_draft_context(g, m, rows, pos) ||
            (!g->draft->capture && !inkling2_draft_anchor(g->draft, rows - 1u))) { return false; }
    }
    g->position = pos + rows;
    if (g->high < g->position) { g->high = g->position; }
    g->failed = false;
    return true;
}

/* Resume at `at` below the live position: full-layer rows are positional,
 * the SSD tier's window restarts at the block holding `at` and the drafter's
 * rings drop the rows the abandoned branch displaced. The caller restores
 * the convolution histories copied at `at`. */
static bool inkling2_rewind(ds4_inkling2_graph *g, uint32_t at) {
    if (at > g->position || !inkling2_kv_rewind(&g->tier, at)) { return false; }
    inkling2_draft_rewind(g->draft, at, g->high);
    g->position = at;
    return true;
}
#else
// Inkling 2 is CUDA-only; Metal keeps linkage and refuses the graph.
static bool inkling2_graph_alloc(ds4_inkling2_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t ctx, uint32_t cap) {
    (void)g; (void)m; (void)w; (void)ctx; (void)cap;
    return false;
}

static bool inkling2_forward(ds4_inkling2_graph *g, const ds4_model *m, const ds4_weights *w, const int *tokens,
        uint32_t rows, uint32_t pos, inkling2_head_rows head) {
    (void)g; (void)m; (void)w; (void)tokens; (void)rows; (void)pos; (void)head;
    return false;
}

static void inkling2_embed_prefetch(const ds4_model *m, const ds4_weights *w, const int *tokens, uint32_t rows) {
    (void)m; (void)w; (void)tokens; (void)rows;
}

static bool inkling2_rewind(ds4_inkling2_graph *g, uint32_t at) {
    (void)g; (void)at;
    return false;
}
#endif
