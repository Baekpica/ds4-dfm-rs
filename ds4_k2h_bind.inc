/* Rust validated the K2-Horizon hVQ release (60 parts, 13-bit GVQ1 headers,
 * book links, Q8_0 scales, layouts) before this pointer adapter. Original HF
 * names are kept. The Q8_0 token table lives in its own `embedding` part,
 * the inventory's last tensor: engine open keeps it out of the device copy
 * and the host reads one 6,528-byte row per token from the mapped part
 * (page cache over SSD). */

static ds4_tensor *k2h_tensor(const ds4_model *m, uint32_t il, const char *name) {
    char full[160];
    const int n = snprintf(full, sizeof(full), "model.layers.%u.%s", il, name);
    if (n < 0 || (size_t)n >= sizeof(full)) { ds4_die("K2-Horizon tensor name too long"); }
    return required_tensor(m, full);
}

// HF [rows, cols] matrices bind as GGUF [cols, rows].
static ds4_tensor *k2h_matrix(const ds4_model *m, uint32_t il, const char *name, uint32_t type,
        uint64_t rows, uint64_t cols) {
    ds4_tensor *t = k2h_tensor(m, il, name);
    tensor_expect_layout(t, type, 2, cols, rows, 0);
    return t;
}

// Norm weights and the router bias: BF16 source values stored as F32.
static ds4_tensor *k2h_vector(const ds4_model *m, uint32_t il, const char *name, uint64_t n) {
    ds4_tensor *t = k2h_tensor(m, il, name);
    tensor_expect_layout(t, DS4_TENSOR_F32, 1, n, 0, 0);
    return t;
}

/* Graph plan: shapes, row buffers and the memory quote (also CPU builds,
 * which refuse the family before allocating). */
enum {
    // max_position_embeddings; NVFP4 K/V is 34.31 GiB there.
    K2H_CONTEXT = 524288,
    // Every chunk streams each routed expert once: 2,048 rows read half of
    // 1,024's expert bytes per token. Short of memory, rows halve down to
    // the minimum before the context does.
    K2H_PREFILL_DEFAULT = 2048, K2H_PREFILL_MAX = 4096, K2H_PREFILL_MIN = 512,
};

static void k2h_bind_experts(ds4_k2h_layer *e, const ds4_model *m, uint32_t il) {
    e->book = k2h_tensor(m, il, "mlp.vq_codebook");
    tensor_expect_layout(e->book, DS4_TENSOR_GLM_BOOK, 2, 8, 8192, 0);
    for (uint32_t x = 0; x < K2H_EXPERTS; x++) {
        char name[96];
        snprintf(name, sizeof(name), "mlp.experts.%u.gate_proj.weight", x);
        e->gate[x] = k2h_matrix(m, il, name, DS4_TENSOR_GLM_VQ, DS4_K2H_FFN, DS4_K2H_HIDDEN);
        snprintf(name, sizeof(name), "mlp.experts.%u.up_proj.weight", x);
        e->up[x] = k2h_matrix(m, il, name, DS4_TENSOR_GLM_VQ, DS4_K2H_FFN, DS4_K2H_HIDDEN);
        snprintf(name, sizeof(name), "mlp.experts.%u.down_proj.weight", x);
        e->down[x] = k2h_matrix(m, il, name, DS4_TENSOR_GLM_VQ, DS4_K2H_HIDDEN, DS4_K2H_FFN);
    }
}

static void k2h_bind_layer(ds4_layer_weights *l, ds4_k2h_layer *e, const ds4_model *m, uint32_t il) {
    const uint32_t Q8 = DS4_TENSOR_Q8_0, H = DS4_K2H_HIDDEN;
    l->attn_norm = k2h_vector(m, il, "input_layernorm.weight", H);
    l->attn_q = k2h_matrix(m, il, "self_attn.q_proj.weight", Q8, DS4_K2H_Q, H);
    l->attn_k = k2h_matrix(m, il, "self_attn.k_proj.weight", Q8, DS4_K2H_KV, H);
    l->attn_v = k2h_matrix(m, il, "self_attn.v_proj.weight", Q8, DS4_K2H_KV, H);
    l->attn_output = k2h_matrix(m, il, "self_attn.o_proj.weight", Q8, H, DS4_K2H_Q);
    l->ffn_norm = k2h_vector(m, il, "post_attention_layernorm.weight", H);
    if (il < K2H_DENSE) {
        l->ffn_gate = k2h_matrix(m, il, "mlp.gate_proj.weight", Q8, DS4_K2H_DENSE_FF, H);
        l->ffn_up = k2h_matrix(m, il, "mlp.up_proj.weight", Q8, DS4_K2H_DENSE_FF, H);
        l->ffn_down = k2h_matrix(m, il, "mlp.down_proj.weight", Q8, H, DS4_K2H_DENSE_FF);
        return;
    }
    l->ffn_gate_inp = k2h_matrix(m, il, "mlp.gate.weight", DS4_TENSOR_F32, K2H_EXPERTS, H);
    l->ffn_exp_probs_b = k2h_vector(m, il, "mlp.gate.bias", K2H_EXPERTS);
    l->ffn_gate_shexp = k2h_matrix(m, il, "mlp.shared_experts.gate_proj.weight", Q8, DS4_K2H_FFN, H);
    l->ffn_up_shexp = k2h_matrix(m, il, "mlp.shared_experts.up_proj.weight", Q8, DS4_K2H_FFN, H);
    l->ffn_down_shexp = k2h_matrix(m, il, "mlp.shared_experts.down_proj.weight", Q8, H, DS4_K2H_FFN);
    k2h_bind_experts(e, m, il);
}

static void k2h_bind(ds4_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    w->k2h = xcalloc(K2H_LAYERS, sizeof(*w->k2h));
    w->token_embd = required_tensor(m, "model.embed_tokens.weight");
    tensor_expect_layout(w->token_embd, DS4_TENSOR_Q8_0, 2, DS4_K2H_HIDDEN, DS4_N_VOCAB, 0);
    w->output_norm = required_tensor(m, "model.norm.weight");
    tensor_expect_layout(w->output_norm, DS4_TENSOR_F32, 1, DS4_K2H_HIDDEN, 0, 0);
    w->output = required_tensor(m, "lm_head.weight");
    tensor_expect_layout(w->output, DS4_TENSOR_Q8_0, 2, DS4_K2H_HIDDEN, DS4_N_VOCAB, 0);
    for (uint32_t il = 0; il < K2H_LAYERS; il++) { k2h_bind_layer(&w->layer[il], &w->k2h[il], m, il); }
}

/* Row buffers, one allocation schema for the graph and its memory quote.
 * F32 rows hold BF16-rounded values, except the router logits and prefill
 * projections' F32 sums. ff_gate/ff_up hold the shared expert's rows; the
 * dense layers' 16,384-wide rows run in slices over the same buffers
 * (k2h_ff_floats). */
#define K2H_ROW_BUFFERS(X) \
    X(cur, DS4_K2H_HIDDEN, float) \
    X(norm, DS4_K2H_HIDDEN, float) \
    X(q, DS4_K2H_Q, float) \
    X(k, DS4_K2H_KV, float) \
    X(v, DS4_K2H_KV, float) \
    X(heads, DS4_K2H_Q, float) \
    X(attn_out, DS4_K2H_HIDDEN, float) \
    X(shared, DS4_K2H_HIDDEN, float) \
    X(ffn_out, DS4_K2H_HIDDEN, float) \
    X(router, K2H_EXPERTS, float) \
    X(route_ids, DS4_K2H_USED, int32_t) \
    X(route_weights, DS4_K2H_USED, float) \
    X(embed, DS4_K2H_EMBED_ROW / 4u, uint32_t) \
    X(rope, 2u * DS4_K2H_ROPE_PAIRS, float)

// Floats of each FFN buffer: the shared expert's rows, at least one dense row.
static uint64_t k2h_ff_floats(uint32_t rows) {
    const uint64_t shared = (uint64_t)rows * DS4_K2H_FFN;
    return shared > DS4_K2H_DENSE_FF ? shared : DS4_K2H_DENSE_FF;
}

// Dense rows per slice that fit the FFN buffers.
static uint32_t k2h_dense_rows(uint32_t rows) { return (uint32_t)(k2h_ff_floats(rows) / DS4_K2H_DENSE_FF); }

static uint32_t k2h_prefill_cap(uint32_t ctx) {
    uint32_t cap = K2H_PREFILL_DEFAULT;
    const char *value = getenv("DS4_K2H_PREFILL_CHUNK");
    if (value && *value) {
        char *end = NULL;
        const unsigned long parsed = strtoul(value, &end, 10);
        if (end != value && !*end && parsed > 0u && parsed <= K2H_PREFILL_MAX) { cap = (uint32_t)parsed; }
    }
    return cap < ctx ? cap : ctx;
}

// Every layer keeps every position: NVFP4 K/V pages of 128 positions.
static uint64_t k2h_layer_kv_bytes(uint32_t ctx) {
    return (uint64_t)((ctx + DS4_K2H_KV_BLOCK - 1u) / DS4_K2H_KV_BLOCK) * DS4_K2H_BLOCK_BYTES;
}

static uint64_t k2h_kv_bytes(uint32_t ctx) { return (uint64_t)K2H_LAYERS * k2h_layer_kv_bytes(ctx); }

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
// GVQ scratch: the router's F32 GEMM needs none; keep one page for the ctx.
static uint64_t k2h_gvq_scratch(uint32_t rows) {
    (void)rows;
    return 1u << 20;
}
#endif

/* Graph 0's scratch beyond its K/V pages: logits, the row and FFN buffers,
 * decode attention partials and the GVQ/expert workspaces. */
static uint64_t k2h_workspace_bytes(uint32_t rows) {
    uint64_t bytes = (uint64_t)DS4_N_VOCAB * sizeof(float) + 2u * k2h_ff_floats(rows) * sizeof(float);
#define K2H_QUOTE(field, cols, type) bytes += (uint64_t)rows * (cols) * sizeof(type);
    K2H_ROW_BUFFERS(K2H_QUOTE)
#undef K2H_QUOTE
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
    bytes += ds4_gpu_k2h_attn_bytes() + k2h_gvq_scratch(rows) + ds4_gpu_gvq_workspace_linear(rows) +
        ds4_gpu_k2h_moe_bytes(rows);
#endif
    return bytes;
}

static ds4_context_memory k2h_memory(uint32_t ctx, uint32_t cap) {
    ds4_context_memory m = {0};
    if (!ctx || !cap || cap > ctx || ctx > K2H_CONTEXT) { return m; }
    m.raw_cap = ctx;
    m.prefill_cap = cap;
    m.raw_bytes = k2h_kv_bytes(ctx);
    m.scratch_bytes = k2h_workspace_bytes(cap);
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}

/* The largest chunk, halving from `cap` but not below K2H_PREFILL_MIN, whose
 * graph and `headroom` fit `free_b` (the native 512K bank leaves a 128 GB
 * Spark a few GiB beside the trunk). */
static uint32_t k2h_fit_cap(uint32_t ctx, uint32_t cap, uint64_t headroom, uint64_t free_b) {
    while (cap / 2u >= K2H_PREFILL_MIN && k2h_memory(ctx, cap).total_bytes + headroom > free_b) { cap /= 2u; }
    return cap;
}

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
static uint32_t k2h_index(const ds4_model *m, const ds4_tensor *t) {
    return t ? (uint32_t)(t - m->tensors) : DS4_GVQ_NO_LINK;
}

/* Materialized source pointers belong to the weight owner, not each bank.
 * Expert records link their layer's book. The embedding, the inventory's
 * last tensor, has no device copy and no record. */
static bool k2h_gpu_bind(ds4_weights *w, const ds4_model *m) {
    if (!w->k2h || !w->token_embd || m->n_tensors < 2 || m->n_tensors > UINT32_MAX ||
        k2h_index(m, w->token_embd) + 1u != m->n_tensors) { return false; }
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
    for (uint32_t il = K2H_DENSE; il < K2H_LAYERS; il++) {
        const ds4_k2h_layer *e = &w->k2h[il];
        const uint32_t book = k2h_index(m, e->book);
        for (uint32_t x = 0; x < K2H_EXPERTS; x++) {
            records[k2h_index(m, e->gate[x])].book = book;
            records[k2h_index(m, e->up[x])].book = book;
            records[k2h_index(m, e->down[x])].book = book;
        }
    }
    w->glmvq_gpu = ds4_gpu_gvq_bind(m->map, m->size, records, count);
    free(records);
    return w->glmvq_gpu != NULL;
}
#endif
