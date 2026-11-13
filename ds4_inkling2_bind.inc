/* Rust validated the Inkling-2-Small VQ release (parts, GVQ1 headers, book
 * links, layouts) before this pointer adapter. Original HF names are kept.
 * The BF16 token table lives in its own `embedding` part, the inventory's
 * last tensor: engine open keeps it out of the device copy and the host
 * reads one 9,216-byte row per token from the mapped part (page cache over
 * SSD). */

static ds4_tensor *inkling2_tensor(const ds4_model *m, uint32_t il, const char *name) {
    char full[160];
    const int n = snprintf(full, sizeof(full), "model.llm.layers.%u.%s", il, name);
    if (n < 0 || (size_t)n >= sizeof(full)) { ds4_die("Inkling-2 tensor name too long"); }
    return required_tensor(m, full);
}

// HF [rows, cols] matrices bind as GGUF [cols, rows]; [C, 1, 4] taps as [4, 1, C].
static ds4_tensor *inkling2_matrix(const ds4_model *m, uint32_t il, const char *name, uint32_t type,
        uint64_t rows, uint64_t cols) {
    ds4_tensor *t = inkling2_tensor(m, il, name);
    tensor_expect_layout(t, type, 2, cols, rows, 0);
    return t;
}

static ds4_tensor *inkling2_vector(const ds4_model *m, uint32_t il, const char *name, uint64_t n) {
    ds4_tensor *t = inkling2_tensor(m, il, name);
    tensor_expect_layout(t, DS4_TENSOR_BF16, 1, n, 0, 0);
    return t;
}

static ds4_tensor *inkling2_taps(const ds4_model *m, uint32_t il, const char *name, uint64_t channels) {
    ds4_tensor *t = inkling2_tensor(m, il, name);
    tensor_expect_layout(t, DS4_TENSOR_BF16, 3, DS4_INK2_TAPS, 1, channels);
    return t;
}

static ds4_tensor *inkling2_query(const ds4_model *m, const char *kind, uint32_t index, uint32_t queries) {
    char name[96];
    snprintf(name, sizeof(name), "model.llm.attn_res.%s.%u", kind, index);
    ds4_tensor *t = required_tensor(m, name);
    if (queries == 1) {
        tensor_expect_layout(t, DS4_TENSOR_F32, 1, DS4_INK2_HIDDEN, 0, 0);
    } else {
        tensor_expect_layout(t, DS4_TENSOR_F32, 2, DS4_INK2_HIDDEN, queries, 0);
    }
    return t;
}

/* Full-attention layers: (l + 1) % 6 == 0; block boundaries: l % 6 == 0. */
static bool inkling2_global(uint32_t il) { return (il + 1u) % DS4_INK2_BLOCK == 0; }
static bool inkling2_boundary(uint32_t il) { return il % DS4_INK2_BLOCK == 0; }

static void inkling2_bind_layer(ds4_inkling2_layer *l, const ds4_model *m, uint32_t il) {
    const uint32_t Q4 = DS4_TENSOR_Q4_K;
    l->q = inkling2_matrix(m, il, "attn.wq_du.weight", Q4, DS4_INK2_Q, DS4_INK2_HIDDEN);
    l->k = inkling2_matrix(m, il, "attn.wk_dv.weight", Q4, DS4_INK2_KV, DS4_INK2_HIDDEN);
    l->v = inkling2_matrix(m, il, "attn.wv_dv.weight", Q4, DS4_INK2_KV, DS4_INK2_HIDDEN);
    l->r = inkling2_matrix(m, il, "attn.wr_du.weight", Q4, DS4_INK2_HEADS * DS4_INK2_REL, DS4_INK2_HIDDEN);
    l->gate = inkling2_matrix(m, il, "attn.gate_attn.weight", Q4, DS4_INK2_Q, DS4_INK2_HIDDEN);
    l->o = inkling2_matrix(m, il, "attn.wo_ud.weight", Q4, DS4_INK2_HIDDEN, DS4_INK2_Q);
    l->sink = inkling2_matrix(m, il, "attn.attn_sink_proj.weight", DS4_TENSOR_BF16, DS4_INK2_HEADS,
        DS4_INK2_HIDDEN);
    l->q_scale = inkling2_matrix(m, il, "attn.q_l2_clip.scale", DS4_TENSOR_BF16, DS4_INK2_HEADS, DS4_INK2_HEAD);
    l->r_scale = inkling2_matrix(m, il, "attn.rel_l2_clip.scale", DS4_TENSOR_BF16, DS4_INK2_HEADS, DS4_INK2_REL);
    l->rel_proj = inkling2_matrix(m, il, "attn.rel_logits_proj.proj", DS4_TENSOR_BF16, DS4_INK2_REL,
        DS4_INK2_EXTENT);
    if (inkling2_global(il)) {
        l->k_taps = inkling2_matrix(m, il, "attn.k_sconv.static_w", DS4_TENSOR_BF16, DS4_INK2_TAPS, DS4_INK2_KV);
        l->v_taps = inkling2_matrix(m, il, "attn.v_sconv.static_w", DS4_TENSOR_BF16, DS4_INK2_TAPS, DS4_INK2_KV);
        l->k_dyn = inkling2_matrix(m, il, "attn.k_sconv.weight_proj", DS4_TENSOR_BF16,
            DS4_INK2_KV_HEADS * DS4_INK2_TAPS, DS4_INK2_HIDDEN);
        l->v_dyn = inkling2_matrix(m, il, "attn.v_sconv.weight_proj", DS4_TENSOR_BF16,
            DS4_INK2_KV_HEADS * DS4_INK2_TAPS, DS4_INK2_HIDDEN);
        l->norm[0] = inkling2_vector(m, il, "q_input_norm.weight", DS4_INK2_HIDDEN);
        l->norm[1] = inkling2_vector(m, il, "k_input_norm.weight", DS4_INK2_HIDDEN);
        l->norm[2] = inkling2_vector(m, il, "v_input_norm.weight", DS4_INK2_HIDDEN);
    } else {
        l->k_taps = inkling2_taps(m, il, "attn.k_sconv.weight", DS4_INK2_KV);
        l->v_taps = inkling2_taps(m, il, "attn.v_sconv.weight", DS4_INK2_KV);
        l->norm[0] = inkling2_vector(m, il, "attn_norm.weight", DS4_INK2_HIDDEN);
    }
    l->attn_conv = inkling2_taps(m, il, "attn_sconv.weight", DS4_INK2_HIDDEN);
    l->mlp_conv = inkling2_taps(m, il, "mlp_sconv.weight", DS4_INK2_HIDDEN);
    l->mlp_norm = inkling2_vector(m, il, "mlp_norm.weight", DS4_INK2_HIDDEN);
    l->router = inkling2_matrix(m, il, "mlp.gate.weight", DS4_TENSOR_BF16, DS4_INK2_EXPERTS, DS4_INK2_HIDDEN);
    l->router_bias = inkling2_tensor(m, il, "mlp.gate.bias");
    tensor_expect_layout(l->router_bias, DS4_TENSOR_F32, 1, DS4_INK2_EXPERTS, 0, 0);
    l->attn_query = il ? inkling2_query(m, "attn_queries", il - 1u, inkling2_global(il) ? 3u : 1u) : NULL;
    l->mlp_query = inkling2_query(m, "mlp_queries", il, 1u);
    l->book = inkling2_tensor(m, il, "mlp.vq_codebook");
    tensor_expect_layout(l->book, DS4_TENSOR_GLM_BOOK, 2, 8, 4096, 0);
    for (uint32_t e = 0; e < DS4_INK2_EXPERTS; e++) {
        char name[112];
        snprintf(name, sizeof(name), "mlp.experts.%u.gate_proj.weight", e);
        l->gate_e[e] = inkling2_matrix(m, il, name, DS4_TENSOR_GLM_VQ, DS4_INK2_FFN, DS4_INK2_HIDDEN);
        snprintf(name, sizeof(name), "mlp.experts.%u.up_proj.weight", e);
        l->up_e[e] = inkling2_matrix(m, il, name, DS4_TENSOR_GLM_VQ, DS4_INK2_FFN, DS4_INK2_HIDDEN);
        snprintf(name, sizeof(name), "mlp.experts.%u.down_proj.weight", e);
        l->down_e[e] = inkling2_matrix(m, il, name, DS4_TENSOR_GLM_VQ, DS4_INK2_HIDDEN, DS4_INK2_FFN);
    }
}

static void inkling2_bind(ds4_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    ds4_inkling2_weights *k = xcalloc(1, sizeof(*k));
    w->inkling2 = k;
    for (uint32_t il = 0; il < INKLING2_LAYERS; il++) { inkling2_bind_layer(&k->layer[il], m, il); }
    k->embed_norm = required_tensor(m, "model.llm.embed_norm.weight");
    tensor_expect_layout(k->embed_norm, DS4_TENSOR_BF16, 1, DS4_INK2_HIDDEN, 0, 0);
    k->out_query = required_tensor(m, "model.llm.attn_res.output_query");
    tensor_expect_layout(k->out_query, DS4_TENSOR_F32, 1, DS4_INK2_HIDDEN, 0, 0);
    w->output_norm = required_tensor(m, "model.llm.norm.weight");
    tensor_expect_layout(w->output_norm, DS4_TENSOR_BF16, 1, DS4_INK2_HIDDEN, 0, 0);
    w->output = required_tensor(m, "model.llm.unembed.weight");
    tensor_expect_layout(w->output, DS4_TENSOR_Q4_K, 2, DS4_INK2_HIDDEN, DS4_N_VOCAB, 0);
    // The embedding part is mapped last (Rust release order).
    w->token_embd = required_tensor(m, "model.llm.embed.weight");
    tensor_expect_layout(w->token_embd, DS4_TENSOR_BF16, 2, DS4_INK2_HIDDEN, DS4_N_VOCAB, 0);
}

/* Graph plan: prefill chunk, K/V and row buffers, one allocation schema for
 * the graph and its memory quote (also CPU builds, which refuse the family
 * before allocating). */
enum {
    INKLING2_CONTEXT = 1048576,
    // Prefill rows per forward; every chunk streams each routed expert once.
    INKLING2_PREFILL_DEFAULT = 2048, INKLING2_PREFILL_MAX = 4096,
    // A K/V row: K then V of the 8 KV heads (BF16).
    INKLING2_KV_ROW = 2 * DS4_INK2_KV * 2,
    // Convolution histories per layer: K and V inputs, then the attention
    // and MLP outputs, three F32 rows each.
    INKLING2_HIST = DS4_INK2_HISTORY * (2 * DS4_INK2_KV + 2 * DS4_INK2_HIDDEN),
};

static uint32_t inkling2_prefill_cap(uint32_t ctx) {
    uint32_t cap = INKLING2_PREFILL_DEFAULT;
    const char *value = getenv("DS4_INKLING2_PREFILL_CHUNK");
    if (value && *value) {
        char *end = NULL;
        const unsigned long parsed = strtoul(value, &end, 10);
        if (end != value && !*end && parsed > 0u && parsed <= INKLING2_PREFILL_MAX) { cap = (uint32_t)parsed; }
    }
    return cap < ctx ? cap : ctx;
}

// Sliding layers keep the window behind a whole chunk; full layers every row.
static uint32_t inkling2_ring(uint32_t cap) { return cap + DS4_INK2_WINDOW - 1u; }

/* Full layers store K/V as NVFP4 pages (the release's K/V contract);
 * DS4_INKLING2_KV_BF16=1 (diagnostic) keeps BF16 rows, the source's own. */
static uint32_t inkling2_full_format(void) {
    static int bf16 = -1;
    if (bf16 < 0) {
        const char *v = getenv("DS4_INKLING2_KV_BF16");
        bf16 = v && !strcmp(v, "1");
    }
    return bf16 ? DS4_INK2_KV_BF16 : DS4_INK2_KV_NVFP4;
}

// One layer's K/V store: sliding rings of BF16 rows, full layers by format
// (NVFP4: 128-position blocks).
static uint64_t inkling2_layer_kv_bytes(uint32_t il, uint32_t ctx, uint32_t cap) {
    if (!inkling2_global(il)) { return (uint64_t)inkling2_ring(cap) * INKLING2_KV_ROW; }
    if (inkling2_full_format() == DS4_INK2_KV_BF16) { return (uint64_t)ctx * INKLING2_KV_ROW; }
    return (uint64_t)((ctx + DS4_INK2_KV_BLOCK - 1u) / DS4_INK2_KV_BLOCK) * DS4_INK2_BLOCK_BYTES;
}

static uint64_t inkling2_kv_bytes(uint32_t ctx, uint32_t cap) {
    uint64_t bytes = (uint64_t)INKLING2_LAYERS * INKLING2_HIST * sizeof(float);
    for (uint32_t il = 0; il < INKLING2_LAYERS; il++) { bytes += inkling2_layer_kv_bytes(il, ctx, cap); }
    return bytes;
}

/* Row buffers. F32 rows hold BF16-rounded values except `resid` (the
 * source's F32 residual sum) and the F32 router logits. */
#define INKLING2_ROW_BUFFERS(X) \
    X(x, DS4_INK2_HIDDEN, float) \
    X(resid, DS4_INK2_HIDDEN, float) \
    X(xq, DS4_INK2_HIDDEN, float) \
    X(xk, DS4_INK2_HIDDEN, float) \
    X(xv, DS4_INK2_HIDDEN, float) \
    X(q, DS4_INK2_Q, float) \
    X(gate, DS4_INK2_Q, float) \
    X(r, DS4_INK2_HEADS * DS4_INK2_REL, float) \
    X(sink, DS4_INK2_HEADS, float) \
    X(kin, DS4_INK2_KV, float) \
    X(vin, DS4_INK2_KV, float) \
    X(dynk, DS4_INK2_KV_HEADS * DS4_INK2_TAPS, float) \
    X(dynv, DS4_INK2_KV_HEADS * DS4_INK2_TAPS, float) \
    X(heads, DS4_INK2_Q, float) \
    X(attn_out, DS4_INK2_HIDDEN, float) \
    X(router, DS4_INK2_EXPERTS, float) \
    X(selected, DS4_INK2_USED, int32_t) \
    X(route_w, DS4_INK2_USED, float) \
    X(moe_out, DS4_INK2_HIDDEN, float) \
    X(embed, DS4_INK2_HIDDEN / 2u, uint32_t) \
    X(rel, DS4_INK2_HEADS * DS4_INK2_EXTENT / 2u, uint32_t) \
    X(blocks, DS4_INK2_BLOCKS * DS4_INK2_HIDDEN / 2u, uint32_t)

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
// GVQ scratch: q4p's prepared FP16 input of the widest Q4_K input (wo_ud,
// 6,144), which bounds every prefill projection, and at least the router
// widened to F32 (the source routes in F32: its cuBLAS path converts the
// BF16 weight once per call).
static uint64_t inkling2_gvq_scratch(uint32_t rows) {
    const uint64_t q4p = (uint64_t)rows * DS4_INK2_Q * sizeof(uint16_t) * 2u;
    const uint64_t router = (uint64_t)DS4_INK2_EXPERTS * DS4_INK2_HIDDEN * sizeof(float);
    return q4p > router ? q4p : router;
}
#endif

static uint64_t inkling2_workspace_bytes(uint32_t rows) {
    uint64_t bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
#define INKLING2_QUOTE(field, cols, type) bytes += (uint64_t)rows * (cols) * sizeof(type);
    INKLING2_ROW_BUFFERS(INKLING2_QUOTE)
#undef INKLING2_QUOTE
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
    bytes += ds4_gpu_ink2_attn_bytes() + inkling2_gvq_scratch(rows) + ds4_gpu_gvq_workspace_linear(rows) +
        ds4_gpu_ink2_moe_bytes(rows);
#endif
    return bytes;
}

static ds4_context_memory inkling2_memory(uint32_t ctx, uint32_t cap) {
    ds4_context_memory m = {0};
    if (!ctx || !cap || cap > ctx || ctx > INKLING2_CONTEXT) { return m; }
    m.raw_cap = ctx;
    m.prefill_cap = cap;
    m.raw_bytes = inkling2_kv_bytes(ctx, cap);
    m.scratch_bytes = inkling2_workspace_bytes(cap);
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
static uint32_t inkling2_index(const ds4_model *m, const ds4_tensor *t) {
    return t ? (uint32_t)(t - m->tensors) : DS4_GVQ_NO_LINK;
}

/* Materialized source pointers belong to the weight owner, not each graph.
 * Expert records link their layer book. The embedding, the inventory's last
 * tensor, has no device copy and no record. */
static bool inkling2_gpu_bind(ds4_weights *w, const ds4_model *m) {
    if (!w->inkling2 || !w->token_embd || m->n_tensors < 2 || m->n_tensors > UINT32_MAX ||
        inkling2_index(m, w->token_embd) + 1u != m->n_tensors) { return false; }
    const uint32_t count = (uint32_t)m->n_tensors - 1u;
    ds4_gvq_record *records = xcalloc(count, sizeof(*records));
    for (uint32_t i = 0; i < count; i++) {
        const ds4_tensor *t = &m->tensors[i];
        const uint64_t cols = t->ndim ? t->dim[0] : 1u;
        const uint64_t rows = cols ? t->elements / cols : 0u;
        if (!cols || cols > UINT32_MAX || !rows || rows > UINT32_MAX) {
            free(records);
            return false;
        }
        records[i] = (ds4_gvq_record){t->abs_offset, t->bytes, t->type,
            (uint32_t)rows, (uint32_t)cols, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK};
    }
    for (uint32_t il = 0; il < INKLING2_LAYERS; il++) {
        const ds4_inkling2_layer *l = &w->inkling2->layer[il];
        const uint32_t book = inkling2_index(m, l->book);
        for (uint32_t e = 0; e < DS4_INK2_EXPERTS; e++) {
            records[inkling2_index(m, l->gate_e[e])].book = book;
            records[inkling2_index(m, l->up_e[e])].book = book;
            records[inkling2_index(m, l->down_e[e])].book = book;
        }
    }
    w->glmvq_gpu = ds4_gpu_gvq_bind(m->map, m->size, records, count);
    free(records);
    return w->glmvq_gpu != NULL;
}
#endif
