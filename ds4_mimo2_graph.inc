/* MiMo owns asymmetric KV and layer scheduling. Reuse the existing mmap
 * matmul/MMQ paths, without borrowing another family's attention semantics. */
/* Span ceiling matches the 64 MiB raw decode cap. The smallest legal
 * frame is 32x256. Each pair can carry a video span and an audio span,
 * and a request may hold four clips. 8 spans was about 8 seconds. */
enum {
    M2_TRIAL = 8,
    M2_MTP_RING = 256,
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
    ds4_gpu_tensor *kv[MIMO2_LAYERS + MIMO2_DRAFT_LAYERS];
    unsigned kv_cap[MIMO2_LAYERS + MIMO2_DRAFT_LAYERS];
    /* Embedded MTP: the last M2_MTP_RING target rows (before the final norm)
     * and their tokens, by position. Block d's rows below mtp_fill[d] carry
     * committed tokens; its RoPE positions count from mtp_base[d]. */
    ds4_gpu_tensor *mtp_hidden;
    int mtp_tok[M2_MTP_RING];
    unsigned mtp_fill[MIMO2_DRAFT_LAYERS], mtp_base[MIMO2_DRAFT_LAYERS];
    unsigned context, cap, position;
    /* One past the newest row written (drafts included): the SWA ring keeps
     * rows at or after high - ring, so a rewind below it is refused. */
    unsigned high;
    /* A restored snapshot holds one SWA window before this row: no rewind
     * below it, however large the ring. */
    unsigned restored;
    unsigned *host_positions;
    float *verify_logits;
    int trial[M2_TRIAL];
    unsigned trial_n, verify_rows;
    uint64_t bytes, media_tag, checkpoint_tag;
    bool failed, hidden_valid, owns_scratch;
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
    /* VQ release (shared scratch): host-gathered BF16 embedding rows, the
     * SWA sinks widened to F32 (51 x 64), the GVQ pointer-plan context and
     * the routed-expert runtime, which replaces the b_routed_* rows. */
    ds4_gpu_tensor *embed, *sinks;
    uint16_t *host_embed;
    ds4_gvq_ctx *gvq;
    ds4_mimo2vq_moe *moe;
} ds4_mimo2_graph;

static void mimo2_graph_free(ds4_mimo2_graph *g) {
    if (g->owns_scratch) {
        exaone_batch_ws_free(&g->ws);
        ds4_gpu_tensor *all[] = {g->qkv, g->positions, g->tokens, g->freq_full,
                                g->freq_swa, g->rope_full, g->rope_swa, g->logits,
                                g->mtp_hidden};
        for (unsigned i = 0; i < sizeof(all) / sizeof(all[0]); i++) {
            ds4_gpu_tensor_free(all[i]);
        }
    }
    for (unsigned il = 0; il < MIMO2_LAYERS + MIMO2_DRAFT_LAYERS; il++) { ds4_gpu_tensor_free(g->kv[il]); }
    if (g->owns_scratch) {
        free(g->host_positions);
        free(g->verify_logits);
    }
    free(g->spans);
    if (g->owns_scratch) {
        free(g->dflash_pos);
        free(g->dflash_saved);
        ds4_gpu_tensor_free(g->dflash_tap);
        ds4_gpu_tensor_free(g->embed);
        ds4_gpu_tensor_free(g->sinks);
        free(g->host_embed);
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
        ds4_gpu_mimo2vq_moe_free(g->moe);
        ds4_gpu_gvq_free(g->gvq);
#endif
    }
    memset(g, 0, sizeof(*g));
}

/* The VQ release's per-graph scratch: BF16 embedding rows, F32 sinks, the
 * GVQ context and the routed-expert runtime (its memory quote). */
static bool mimo2vq_scratch(ds4_mimo2_graph *g, const ds4_model *m, const ds4_weights *w, unsigned cap) {
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
    enum { SINK_LAYERS = MIMO2_LAYERS + MIMO2_DRAFT_LAYERS, HEADS = 64 };
    float sinks[SINK_LAYERS * HEADS] = {0};
    for (unsigned il = 0; il < SINK_LAYERS; il++) {
        const ds4_tensor *t = w->layer[il].attn_sinks;
        if (!t) { continue; }
        const uint16_t *bf16 = (const uint16_t *)(m->map + t->abs_offset);
        for (unsigned h = 0; h < HEADS; h++) {
            const uint32_t bits = (uint32_t)bf16[h] << 16;
            memcpy(&sinks[il * HEADS + h], &bits, sizeof(float));
        }
    }
    g->embed = ds4_gpu_tensor_alloc((uint64_t)cap * M2_EMBED * sizeof(uint16_t));
    g->sinks = ds4_gpu_tensor_alloc(sizeof(sinks));
    g->host_embed = xmalloc((size_t)cap * M2_EMBED * sizeof(uint16_t));
    g->gvq = ds4_gpu_gvq_create_linear(w->glmvq_gpu, M2VQ_GVQ_SCRATCH, cap);
    g->moe = g->gvq ? ds4_gpu_mimo2vq_moe_create(g->gvq, cap) : NULL;
    if (!g->embed || !g->sinks || !g->moe || !ds4_gpu_tensor_write(g->sinks, 0, sinks, sizeof(sinks))) {
        return false;
    }
    g->bytes += mimo2vq_scratch_bytes(cap);
    return true;
#else
    (void)g; (void)m; (void)w; (void)cap;
    return false;
#endif
}

static bool mimo2_graph_alloc(ds4_mimo2_graph *g, const ds4_model *m, const ds4_weights *w,
                              unsigned ctx, unsigned cap, bool dflash) {
    free(g->spans);
    free(g->dflash_pos);
    free(g->dflash_saved);
    ds4_gpu_tensor_free(g->dflash_tap);
    memset(g, 0, sizeof(*g));
    g->owns_scratch = true;
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
    if (!mimo2_vq()) {
        M2_ALLOC(ws.b_routed_gate, (uint64_t)cap * M2_USED * M2_FF);
        M2_ALLOC(ws.b_routed_up, (uint64_t)cap * M2_USED * M2_FF);
        M2_ALLOC(ws.b_routed_mid, (uint64_t)cap * M2_USED * M2_FF);
        M2_ALLOC(ws.b_routed_down, (uint64_t)cap * M2_USED * M2_EMBED);
    }
    M2_ALLOC(qkv, (uint64_t)cap * M2_QKV);
    M2_ALLOC(positions, cap);
    M2_ALLOC(tokens, cap);
    M2_ALLOC(freq_full, M2_ROT / 2);
    M2_ALLOC(freq_swa, M2_ROT / 2);
    M2_ALLOC(rope_full, (uint64_t)cap * M2_ROT);
    M2_ALLOC(rope_swa, (uint64_t)cap * M2_ROT);
    M2_ALLOC(logits, DS4_N_VOCAB);
#undef M2_ALLOC
    if (mimo2_vq() && !mimo2vq_scratch(g, m, w, cap)) { mimo2_graph_free(g); return false; }
    g->host_positions = xmalloc(cap * sizeof(unsigned));
    float full[M2_ROT / 2], swa[M2_ROT / 2];
    for (unsigned d = 0; d < M2_ROT / 2; d++) {
        full[d] = (float)pow(10000000.0, -2.0 * d / M2_ROT);
        swa[d] = (float)pow(10000.0, -2.0 * d / M2_ROT);
    }
    if (!ds4_gpu_tensor_write(g->freq_full, 0, full, sizeof(full)) ||
        !ds4_gpu_tensor_write(g->freq_swa, 0, swa, sizeof(swa))) { mimo2_graph_free(g); return false; }
    for (unsigned il = 0; il < MIMO2_LAYERS; il++) {
        g->kv_cap[il] = mimo2_kv_capacity(il, ctx, cap);
        const uint64_t bytes = (uint64_t)g->kv_cap[il] * mimo2_kv_row(il);
        g->kv[il] = ds4_gpu_tensor_alloc(bytes);
        if (!g->kv[il]) { mimo2_graph_free(g); return false; }
        g->bytes += bytes;
    }
    if (g->bytes != estimate.total_bytes) { mimo2_graph_free(g); return false; }
    /* Draft KV and the MTP row ring (4 MiB) are outside the trunk estimate. */
    const uint64_t draft_row = 8ull * (192 + 128) * sizeof(uint16_t);
    if (!dflash && w->layer[MIMO2_LAYERS].attn_qkv) {
        g->mtp_hidden = ds4_gpu_tensor_alloc((uint64_t)M2_MTP_RING * M2_EMBED * sizeof(float));
        if (!g->mtp_hidden) { mimo2_graph_free(g); return false; }
    }
    g->verify_logits = xcalloc((size_t)M2_TRIAL * DS4_N_VOCAB, sizeof(float));
    for (unsigned il = 0; il < MIMO2_DRAFT_LAYERS; il++) {
        const unsigned layer = MIMO2_LAYERS + il;
        g->kv_cap[layer] = mimo2_kv_capacity(layer, ctx, cap);
        g->kv[layer] = ds4_gpu_tensor_alloc((uint64_t)g->kv_cap[layer] * draft_row);
        if (!g->kv[layer]) { mimo2_graph_free(g); return false; }
    }
    if (!g->verify_logits) {
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

/* Bank forwards are serialized. Share scratch, logits and RoPE; only KV and
 * position are per bank. This keeps two long contexts inside Spark memory. */
static bool mimo2_graph_clone_bank(ds4_mimo2_graph *g,
                                   const ds4_mimo2_graph *shared) {
    if (!g || !shared || !shared->owns_scratch || shared->dflash_on) { return false; }
    *g = *shared;
    g->owns_scratch = false;
    g->position = 0;
    g->high = 0;
    g->restored = 0;
    g->failed = false;
    g->hidden_valid = false;
    g->spans = NULL;
    g->span_cap = 0;
    g->span_n = 0;
    g->dflash_tap = NULL;
    g->dflash_pos = NULL;
    g->dflash_saved = NULL;
    g->dflash_on = false;
    g->mtp_hidden = NULL;
    memset(g->kv, 0, sizeof(g->kv));
    for (unsigned il = 0; il < MIMO2_LAYERS; il++) {
        const uint64_t bytes = (uint64_t)g->kv_cap[il] * mimo2_kv_row(il);
        g->kv[il] = ds4_gpu_tensor_alloc(bytes);
        if (!g->kv[il]) { mimo2_graph_free(g); return false; }
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

/* Keep route weights after down projection. Prefill may emit the same Q8
 * bytes without materializing b_routed_mid; narrow decode uses its old path. */
static bool mimo2_expert_down(ds4_exaone_batch_ws *b, const ds4_model *m,
                               const ds4_layer_weights *l, unsigned n) {
    if (l->ffn_down_exps->type == DS4_TENSOR_IQ2_XS) {
        const int rc = ds4_gpu_mimo2_down(b->b_routed_down, b->b_routed_gate,
                b->b_routed_up, b->b_router_selected, m->map, m->size,
                l->ffn_down_exps->abs_offset, l->ffn_down_exps->bytes, n);
        if (rc >= 0) { return rc != 0; }
    }
    return ds4_gpu_step37_swiglu(b->b_routed_mid, b->b_routed_gate,
            b->b_routed_up, NULL, M2_FF, n * M2_USED, 0) &&
        ds4_gpu_routed_matmul_guarded_tensor(b->b_routed_down, b->b_routed_mid,
            b->b_router_selected, m->map, m->size,
            l->ffn_down_exps->abs_offset, l->ffn_down_exps->bytes,
            l->ffn_down_exps->type, M2_FF, M2_EMBED, M2_EXPERTS,
            n * M2_USED, 1, n);
}

static bool mimo2_attn_add(ds4_exaone_batch_ws *b, unsigned n) {
    enum { MIN_FUSED_ROWS = 32 };
    if (n >= MIN_FUSED_ROWS) {
        const int rc = ds4_gpu_mimo2_attn_add(b->b_cur, b->b_attn_out, n);
        if (rc >= 0) { return rc != 0; }
    }
    const uint64_t count = (uint64_t)n * M2_EMBED;
    return ds4_gpu_dots3_scale_tensor(b->b_attn_out, 0.707f, count) &&
        ds4_gpu_exaone_add_tensor(b->b_cur, b->b_attn_out, count);
}

/* RMSNorm with the mixed artifact's F32 or the VQ release's BF16 weight. */
static bool mimo2_norm(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_model *m,
                       const ds4_tensor *weight, unsigned rows) {
    if (weight->type == DS4_TENSOR_BF16) {
        return ds4_gpu_exaone_rms_norm_bf16_tensor(out, x, m->map, m->size,
            weight->abs_offset, M2_EMBED, rows, DS4_RMS_EPS) != 0;
    }
    return ds4_gpu_exaone_rms_norm_tensor(out, x, m->map, m->size,
        weight->abs_offset, M2_EMBED, rows, DS4_RMS_EPS) != 0;
}

/* Attention of layer `il` (trunk or MTP block): the VQ release reads its
 * sinks from the widened F32 table, the mixed artifact from the map. */
static bool mimo2_attend(ds4_mimo2_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                         unsigned il, unsigned heads, unsigned n) {
    ds4_exaone_batch_ws *b = &g->ws;
    const unsigned window = heads == 4 ? 0 : M2_WINDOW;
    if (g->sinks) {
        return ds4_gpu_mimo2_attention_sinks(b->b_heads, b->b_q, g->kv[il], g->positions,
            l->attn_sinks ? g->sinks : NULL, il, heads, g->kv_cap[il], n, window, g->position,
            mimo2_fp8_rows(il)) != 0;
    }
    return ds4_gpu_mimo2_attention(b->b_heads, b->b_q, g->kv[il], g->positions,
        l->attn_sinks ? m->map : NULL, m->size, l->attn_sinks ? l->attn_sinks->abs_offset : 0,
        heads, g->kv_cap[il], n, window, g->position) != 0;
}

/* The VQ release's routed experts (GVQ runtime) into b_ffn_out. */
static bool mimo2vq_experts(ds4_mimo2_graph *g, const ds4_weights *w, unsigned il, unsigned n) {
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
    ds4_exaone_batch_ws *b = &g->ws;
    return ds4_gpu_mimo2vq_moe(g->moe, b->b_ffn_out, b->b_norm, b->b_router_selected,
        b->b_router_weights, w->mimo2vq[il - 1u].matrices, n) != 0;
#else
    (void)g; (void)w; (void)il; (void)n;
    return false;
#endif
}

static bool mimo2_layer(ds4_mimo2_graph *g, const ds4_model *m,
                         const ds4_weights *w, unsigned il, unsigned n) {
    ds4_exaone_batch_ws *b = &g->ws;
    const ds4_layer_weights *l = &w->layer[il];
    const bool full = mimo2_is_full(il);
    const unsigned heads = full ? 4 : 8, qkv_rows = M2_Q + heads * (192 + 128);
    if (!mimo2_norm(b->b_norm, b->b_cur, m, l->attn_norm, n) ||
        !plain_graph_matmul_tensor(g->qkv, m, l->attn_qkv, M2_EMBED, qkv_rows, b->b_norm, n) ||
        !ds4_gpu_mimo2_qkv(b->b_q, b->b_k, b->b_v, g->qkv,
            full ? g->rope_full : g->rope_swa, heads, n) ||
        !(mimo2_fp8_rows(il)
            ? ds4_gpu_mimo2_kv_store_fp8(g->kv[il], b->b_k, b->b_v, g->positions, n, g->kv_cap[il])
            : ds4_gpu_mimo2_kv_store(g->kv[il], b->b_k, b->b_v, g->positions, heads, n, g->kv_cap[il])) ||
        !mimo2_attend(g, m, l, il, heads, n) ||
        !plain_graph_matmul_tensor(b->b_attn_out, m, l->attn_output, M2_HEADS, M2_EMBED, b->b_heads, n) ||
        !mimo2_attn_add(b, n) ||
        !mimo2_norm(b->b_norm, b->b_cur, m, l->ffn_norm, n)) { return false; }
    if (g->moe && l->ffn_gate_inp) {
        if (!plain_graph_matmul_tensor(b->b_router_logits, m, l->ffn_gate_inp, M2_EMBED, M2_EXPERTS, b->b_norm, n) ||
            !ds4_gpu_mimo2_router(b->b_router_selected, b->b_router_weights, b->b_router_logits,
                m->map, m->size, l->ffn_exp_probs_b->abs_offset, n) ||
            !mimo2vq_experts(g, w, il, n)) { return false; }
    } else if (!l->ffn_gate_exps) {
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
            !mimo2_expert_down(b, m, l, n)) { return false; }
        const int rc = ds4_gpu_mimo2_sum_add(b->b_cur, b->b_routed_down, b->b_router_weights, n);
        if (rc >= 0) { return rc != 0; }
        if (!ds4_gpu_step37_sum(b->b_ffn_out, b->b_routed_down, b->b_router_weights, M2_EMBED, n)) { return false; }
    }
    return ds4_gpu_exaone_add_tensor(b->b_cur, b->b_ffn_out, (uint64_t)n * M2_EMBED);
}

static bool mimo2_head(ds4_mimo2_graph *g, const ds4_model *m, const ds4_weights *w, unsigned row) {
    ds4_gpu_tensor *last = ds4_gpu_tensor_view(g->ws.b_cur,
        (uint64_t)row * M2_EMBED * sizeof(float), M2_EMBED * sizeof(float));
    const bool ok = last && mimo2_norm(g->ws.b_norm, last, m, w->output_norm, 1) &&
        plain_graph_matmul_tensor(g->logits, m, w->output, M2_EMBED, DS4_N_VOCAB, g->ws.b_norm, 1);
    ds4_gpu_tensor_free(last);
    return ok;
}

/* Verify rows through the head in one matmul, so the 0.62 GiB Q8_0 head
 * streams once rather than once per row. The logits land in the qkv
 * scratch, free after the forward; a cap too small for them keeps the
 * one-row head. */
static bool mimo2_head_rows(ds4_mimo2_graph *g, const ds4_model *m, const ds4_weights *w,
                            unsigned n, float *out) {
    const uint64_t bytes = (uint64_t)n * DS4_N_VOCAB * sizeof(float);
    if (n > 1 && ds4_gpu_tensor_bytes(g->qkv) >= bytes) {
        return mimo2_norm(g->ws.b_norm, g->ws.b_cur, m, w->output_norm, n) &&
            plain_graph_matmul_tensor(g->qkv, m, w->output, M2_EMBED, DS4_N_VOCAB, g->ws.b_norm, n) &&
            ds4_gpu_tensor_read(g->qkv, 0, out, bytes);
    }
    for (unsigned row = 0; row < n; row++) {
        if (!mimo2_head(g, m, w, row) ||
            !ds4_gpu_tensor_read(g->logits, 0, out + (size_t)row * DS4_N_VOCAB,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float))) { return false; }
    }
    return true;
}

/* Ask the kernel to read `rows` tokens' embedding rows ahead: a cold gather
 * faults its rows in from the SSD one page at a time (Beam measured ~66 ms
 * per 2,048 rows). Advisory only. */
static void mimo2_embed_prefetch(const ds4_model *m, const ds4_weights *w, const int *tokens, unsigned rows) {
    const uint64_t row = (uint64_t)M2_EMBED * sizeof(uint16_t);
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint8_t *table = m->map + w->token_embd->abs_offset;
    for (unsigned r = 0; r < rows; r++) {
        if (tokens[r] < 0 || (unsigned)tokens[r] >= DS4_N_VOCAB) { continue; }
        const uintptr_t at = (uintptr_t)(table + (uint64_t)tokens[r] * row);
        const uintptr_t first = at / page * page, last = (at + row + page - 1) / page * page;
        (void)madvise((void *)first, last - first, MADV_WILLNEED);
    }
}

/* Token rows into `out`: the VQ release's host reads BF16 rows from its
 * SSD-backed embedding part (page cache) and the device widens them; the
 * mixed artifact gathers its resident table by the device token ids. */
static bool mimo2_embed(ds4_mimo2_graph *g, const ds4_model *m, const ds4_weights *w,
                        const int *tokens, unsigned n, ds4_gpu_tensor *out) {
    if (!g->embed) {
        return ds4_gpu_embed_tokens_quant_tensor(out, g->tokens, m->map, m->size,
            w->token_embd->abs_offset, w->token_embd->type, DS4_N_VOCAB, n, M2_EMBED) != 0;
    }
    const uint64_t row = (uint64_t)M2_EMBED * sizeof(uint16_t);
    const uint8_t *table = m->map + w->token_embd->abs_offset;
    if (n > 1) { mimo2_embed_prefetch(m, w, tokens, n); }
    for (unsigned r = 0; r < n; r++) {
        memcpy((uint8_t *)g->host_embed + r * row, table + (uint64_t)tokens[r] * row, row);
    }
    return ds4_gpu_tensor_write(g->embed, 0, g->host_embed, n * row) &&
        ds4_gpu_qwen4exp_bf16_to_f32_tensor(out, g->embed, (uint64_t)n * M2_EMBED);
}

/* Restart every MTP block at `pos` (reset or restore): rows before it
 * belong to another prefix or are not in the ring. */
static void mimo2_mtp_rebase(ds4_mimo2_graph *g, unsigned pos) {
    for (unsigned d = 0; d < MIMO2_DRAFT_LAYERS; d++) {
        g->mtp_fill[d] = pos;
        g->mtp_base[d] = pos;
    }
}

typedef enum { M2_RING_STORE, M2_RING_LOAD } mimo2_ring_dir;

/* Rows pos..pos+n-1 between `rows` and the MTP ring: two spans at most. */
static bool mimo2_ring_copy(ds4_mimo2_graph *g, ds4_gpu_tensor *rows, unsigned pos,
                            unsigned n, mimo2_ring_dir dir) {
    const uint64_t row = (uint64_t)M2_EMBED * sizeof(float);
    for (unsigned i = 0; i < n;) {
        const unsigned slot = (pos + i) % M2_MTP_RING;
        unsigned span = M2_MTP_RING - slot;
        if (span > n - i) { span = n - i; }
        const bool ok = dir == M2_RING_STORE
            ? ds4_gpu_tensor_copy(g->mtp_hidden, slot * row, rows, i * row, span * row)
            : ds4_gpu_tensor_copy(rows, i * row, g->mtp_hidden, slot * row, span * row);
        if (!ok) { return false; }
        i += span;
    }
    return true;
}

/* Keep the forward's last rows for the MTP blocks: a draft row reads the
 * target row at its own position, up to a window behind the frontier. */
static bool mimo2_mtp_push(ds4_mimo2_graph *g, const int *tokens, unsigned n, unsigned pos) {
    if (!g->mtp_hidden) { return true; }
    const unsigned skip = n > M2_MTP_RING ? n - M2_MTP_RING : 0;
    const uint64_t row = (uint64_t)M2_EMBED * sizeof(float);
    ds4_gpu_tensor *rows = ds4_gpu_tensor_view(g->ws.b_cur, skip * row, (n - skip) * row);
    const bool ok = rows && mimo2_ring_copy(g, rows, pos + skip, n - skip, M2_RING_STORE);
    ds4_gpu_tensor_free(rows);
    for (unsigned i = skip; i < n; i++) { g->mtp_tok[(pos + i) % M2_MTP_RING] = tokens[i]; }
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
        !mimo2_embed(g, m, w, tokens, n, g->ws.b_cur) ||
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
    if (!mimo2_head(g, m, w, n - 1) || !mimo2_mtp_push(g, tokens, n, pos)) { return false; }
    g->hidden_valid = true;
    g->position = pos + n;
    if (g->high < g->position) { g->high = g->position; }
    g->failed = false;
    return true;
}
