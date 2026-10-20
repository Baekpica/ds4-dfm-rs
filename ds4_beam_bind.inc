/* Rust validated the Beam-1.0 release (51 parts, GVQ1 headers, book links,
 * layouts) before this pointer adapter. Original HF names are kept. The
 * BF16 token table lives in its own `embedding` part, the inventory's last
 * tensor: engine open keeps it out of the device copy and the host reads
 * one 10,240-byte row per token from the mapped part (page cache over SSD). */

// config.json layer_types: full attention at 5, 11, ..., 47 and the last.
static bool ds4_beam_layer_is_full(uint32_t il) {
    return il % 6u == 5u || il + 1u == BEAM_LAYERS;
}

static ds4_tensor *beam_tensor(const ds4_model *m, uint32_t il, const char *name) {
    char full[160];
    const int n = snprintf(full, sizeof(full), "model.layers.%u.%s", il, name);
    if (n < 0 || (size_t)n >= sizeof(full)) { ds4_die("Beam tensor name too long"); }
    return required_tensor(m, full);
}

// HF [rows, cols] matrices bind as GGUF [cols, rows].
static ds4_tensor *beam_matrix(const ds4_model *m, uint32_t il, const char *name, uint32_t type,
        uint64_t rows, uint64_t cols) {
    ds4_tensor *t = beam_tensor(m, il, name);
    tensor_expect_layout(t, type, 2, cols, rows, 0);
    return t;
}

static ds4_tensor *beam_vector(const ds4_model *m, uint32_t il, const char *name, uint32_t type,
        uint64_t n) {
    ds4_tensor *t = beam_tensor(m, il, name);
    tensor_expect_layout(t, type, 1, n, 0, 0);
    return t;
}

/* Graph plan: shapes, row buffers and the memory quote (also CPU builds,
 * which refuse the family before allocating). */
enum {
    BEAM_EMBD = 5120, BEAM_HEADS = 48, BEAM_KV_HEADS = 8, BEAM_HEAD = 128,
    BEAM_Q = BEAM_HEADS * BEAM_HEAD, BEAM_KV = BEAM_KV_HEADS * BEAM_HEAD,
    BEAM_DENSE_FF = 20480, BEAM_FF = 2560, BEAM_USED = 8, BEAM_WINDOW = 512,
    BEAM_CONTEXT = 1048576,
    // The 1M bank fits beside the 91.6 GiB trunk at 2,048-row chunks; 4,096
    // rows need ~1.4 GiB more workspace (handoff memory-budget.json).
    BEAM_PREFILL_DEFAULT = 2048, BEAM_PREFILL_MAX = 4096,
    // Tokenizer ids; the head's last 255 of 151,936 rows are padding.
    BEAM_TOKENS = 151681,
    // SWA rings keep [K | V] f16 rows (the K-EXAONE layout); full layers keep
    // Solar's FP8 row: E4M3 K and V, then one f16 scale per KV head each.
    BEAM_SWA_ROW = 2 * BEAM_KV * 2, BEAM_FULL_ROW = 2 * BEAM_KV + 2 * BEAM_KV_HEADS * 2,
    // Decode attention splits a full layer's keys into at most this many chunks.
    BEAM_DECODE_SPLITS = 128,
};
static const float BEAM_LOGIT_CAP = 30.0f;

// DS4_BEAM_FULL_KV=f16 keeps the full layers' rows as f16 [K | V] (Solar's
// BF16 layout, 4,096 bytes): a diagnostic of the FP8 rounding.
static ds4_solar_kv_format beam_full_kv(void) {
    static int format = -1;
    if (format < 0) {
        const char *value = getenv("DS4_BEAM_FULL_KV");
        format = value && !strcmp(value, "f16") ? DS4_SOLAR_KV_BF16 : DS4_SOLAR_KV_FP8;
    }
    return (ds4_solar_kv_format)format;
}

static uint32_t beam_full_row(void) {
    return beam_full_kv() == DS4_SOLAR_KV_FP8 ? BEAM_FULL_ROW : BEAM_SWA_ROW;
}

static void beam_bind_experts(ds4_beam_layer *e, const ds4_model *m, uint32_t il) {
    e->book = beam_tensor(m, il, "mlp.vq_codebook");
    // recipe.json: layers 3-10 keep K8192 (13-bit) books, the rest K4096.
    const uint64_t words = il <= 10u ? 8192u : 4096u;
    tensor_expect_layout(e->book, DS4_TENSOR_GLM_BOOK, 2, 8, words, 0);
    for (uint32_t expert = 0; expert < BEAM_EXPERTS; expert++) {
        char name[96];
        snprintf(name, sizeof(name), "mlp.experts.%u.gate_proj.weight", expert);
        e->gate[expert] = beam_matrix(m, il, name, DS4_TENSOR_GLM_VQ, BEAM_FF, BEAM_EMBD);
        snprintf(name, sizeof(name), "mlp.experts.%u.up_proj.weight", expert);
        e->up[expert] = beam_matrix(m, il, name, DS4_TENSOR_GLM_VQ, BEAM_FF, BEAM_EMBD);
        snprintf(name, sizeof(name), "mlp.experts.%u.down_proj.weight", expert);
        e->down[expert] = beam_matrix(m, il, name, DS4_TENSOR_GLM_VQ, BEAM_EMBD, BEAM_FF);
    }
}

static void beam_bind_layer(ds4_layer_weights *l, ds4_beam_layer *b, const ds4_model *m, uint32_t il) {
    // Sandwich norms: input -> attention -> post_attention; pre_feedforward
    // -> FFN -> post_feedforward.
    l->attn_norm = beam_vector(m, il, "input_layernorm.weight", DS4_TENSOR_BF16, BEAM_EMBD);
    l->attn_output_norm = beam_vector(m, il, "post_attention_layernorm.weight", DS4_TENSOR_BF16, BEAM_EMBD);
    l->ffn_norm = beam_vector(m, il, "pre_feedforward_layernorm.weight", DS4_TENSOR_BF16, BEAM_EMBD);
    l->ffn_output_norm = beam_vector(m, il, "post_feedforward_layernorm.weight", DS4_TENSOR_BF16, BEAM_EMBD);
    l->attn_q = beam_matrix(m, il, "self_attn.q_proj.weight", DS4_TENSOR_Q4_K, BEAM_Q, BEAM_EMBD);
    l->attn_k = beam_matrix(m, il, "self_attn.k_proj.weight", DS4_TENSOR_Q4_K, BEAM_KV, BEAM_EMBD);
    l->attn_v = beam_matrix(m, il, "self_attn.v_proj.weight", DS4_TENSOR_Q4_K, BEAM_KV, BEAM_EMBD);
    l->attn_gate = beam_matrix(m, il, "self_attn.gate_proj.weight", DS4_TENSOR_Q4_K, BEAM_Q, BEAM_EMBD);
    l->attn_output = beam_matrix(m, il, "self_attn.o_proj.weight", DS4_TENSOR_Q4_K, BEAM_EMBD, BEAM_Q);
    l->attn_q_norm = beam_vector(m, il, "self_attn.q_norm.weight", DS4_TENSOR_BF16, BEAM_HEAD);
    l->attn_k_norm = beam_vector(m, il, "self_attn.k_norm.weight", DS4_TENSOR_BF16, BEAM_HEAD);
    b->gate_bias = beam_vector(m, il, "self_attn.gate_bias", DS4_TENSOR_BF16, BEAM_HEADS);
    if (il < BEAM_DENSE) {
        l->ffn_gate = beam_matrix(m, il, "mlp.gate_proj.weight", DS4_TENSOR_Q4_K, BEAM_DENSE_FF, BEAM_EMBD);
        l->ffn_up = beam_matrix(m, il, "mlp.up_proj.weight", DS4_TENSOR_Q4_K, BEAM_DENSE_FF, BEAM_EMBD);
        l->ffn_down = beam_matrix(m, il, "mlp.down_proj.weight", DS4_TENSOR_Q4_K, BEAM_EMBD, BEAM_DENSE_FF);
        return;
    }
    l->ffn_gate_inp = beam_matrix(m, il, "mlp.gate.weight", DS4_TENSOR_F32, BEAM_EXPERTS, BEAM_EMBD);
    l->ffn_exp_probs_b = beam_vector(m, il, "mlp.gate.e_score_correction_bias", DS4_TENSOR_F32, BEAM_EXPERTS);
    beam_bind_experts(b, m, il);
}

static void beam_bind(ds4_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    w->beam = xcalloc(BEAM_LAYERS, sizeof(*w->beam));
    w->token_embd = required_tensor(m, "model.embed_tokens.weight");
    tensor_expect_layout(w->token_embd, DS4_TENSOR_BF16, 2, BEAM_EMBD, DS4_N_VOCAB, 0);
    w->output_norm = required_tensor(m, "model.norm.weight");
    tensor_expect_layout(w->output_norm, DS4_TENSOR_BF16, 1, BEAM_EMBD, 0, 0);
    w->output = required_tensor(m, "lm_head.weight");
    tensor_expect_layout(w->output, DS4_TENSOR_Q4_K, 2, BEAM_EMBD, DS4_N_VOCAB, 0);
    for (uint32_t il = 0; il < BEAM_LAYERS; il++) { beam_bind_layer(&w->layer[il], &w->beam[il], m, il); }
}

/* Row buffers, one allocation schema for the graph and its memory quote.
 * F32 rows hold BF16-rounded values, except the F32 router logits. */
#define BEAM_ROW_BUFFERS(X) \
    X(token, 1u, int32_t) \
    X(cur, BEAM_EMBD, float) \
    X(norm, BEAM_EMBD, float) \
    X(q, BEAM_Q, float) \
    X(k, BEAM_KV, float) \
    X(v, BEAM_KV, float) \
    X(gate, BEAM_Q, float) \
    X(heads, BEAM_Q, float) \
    X(attn_out, BEAM_EMBD, float) \
    X(ffn_out, BEAM_EMBD, float) \
    X(ff_gate, BEAM_DENSE_FF, float) \
    X(ff_up, BEAM_DENSE_FF, float) \
    X(router, BEAM_EXPERTS, float) \
    X(selected, BEAM_USED, int32_t) \
    X(route_weights, BEAM_USED, float) \
    X(embed, BEAM_EMBD / 2u, uint32_t)

static uint32_t beam_prefill_cap(uint32_t ctx) {
    uint32_t cap = BEAM_PREFILL_DEFAULT;
    const char *value = getenv("DS4_BEAM_PREFILL_CHUNK");
    if (value && *value) {
        char *end = NULL;
        const unsigned long parsed = strtoul(value, &end, 10);
        if (end != value && !*end && parsed > 0u && parsed <= BEAM_PREFILL_MAX) { cap = (uint32_t)parsed; }
    }
    return cap < ctx ? cap : ctx;
}

/* A prefill chunk stores all of its K/V rows before any row attends, so a
 * sliding ring holds the window plus one chunk (exaone_graph_layer_kv_cap). */
static uint32_t beam_swa_cap(uint32_t ctx, uint32_t cap) {
    const uint64_t rows = (uint64_t)BEAM_WINDOW + cap;
    return rows < ctx ? (uint32_t)rows : ctx;
}

static uint64_t beam_kv_bytes(uint32_t ctx, uint32_t cap) {
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < BEAM_LAYERS; il++) {
        bytes += ds4_beam_layer_is_full(il) ? (uint64_t)ctx * beam_full_row()
                                            : (uint64_t)beam_swa_cap(ctx, cap) * BEAM_SWA_ROW;
    }
    return bytes;
}

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
// GVQ scratch: the q4p prepared FP16 input of the widest Q4_K input (dense
// down, 20,480), which bounds every prefill projection.
static uint64_t beam_gvq_scratch(uint32_t rows) {
    return (uint64_t)rows * BEAM_DENSE_FF * sizeof(uint16_t) * 2u;
}
#endif

static uint64_t beam_workspace_bytes(uint32_t rows) {
    uint64_t bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
#define BEAM_QUOTE(field, cols, type) bytes += (uint64_t)rows * (cols) * sizeof(type);
    BEAM_ROW_BUFFERS(BEAM_QUOTE)
#undef BEAM_QUOTE
    // Split decode attention partials: per head and split, head + max + sum.
    bytes += (uint64_t)BEAM_HEADS * BEAM_DECODE_SPLITS * (BEAM_HEAD + 2u) * sizeof(float);
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
    bytes += beam_gvq_scratch(rows) + ds4_gpu_gvq_workspace_linear(rows) + ds4_gpu_beam_moe_bytes(rows);
#endif
    return bytes;
}

static ds4_context_memory beam_memory(uint32_t ctx, uint32_t cap) {
    ds4_context_memory m = {0};
    if (!ctx || !cap || cap > ctx || ctx > BEAM_CONTEXT) { return m; }
    m.raw_cap = ctx;
    m.prefill_cap = cap;
    m.raw_bytes = beam_kv_bytes(ctx, cap);
    m.scratch_bytes = beam_workspace_bytes(cap);
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
static uint32_t beam_index(const ds4_model *m, const ds4_tensor *t) {
    return t ? (uint32_t)(t - m->tensors) : DS4_GVQ_NO_LINK;
}

/* Materialized source pointers belong to the weight owner, not each bank.
 * Expert records link their layer book. The embedding, the inventory's last
 * tensor, has no device copy and no record. */
static bool beam_gpu_bind(ds4_weights *w, const ds4_model *m) {
    if (!w->beam || !w->token_embd || m->n_tensors < 2 || m->n_tensors > UINT32_MAX ||
        beam_index(m, w->token_embd) + 1u != m->n_tensors) { return false; }
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
    for (uint32_t il = BEAM_DENSE; il < BEAM_LAYERS; il++) {
        const ds4_beam_layer *e = &w->beam[il];
        const uint32_t book = beam_index(m, e->book);
        for (uint32_t expert = 0; expert < BEAM_EXPERTS; expert++) {
            records[beam_index(m, e->gate[expert])].book = book;
            records[beam_index(m, e->up[expert])].book = book;
            records[beam_index(m, e->down[expert])].book = book;
        }
    }
    w->glmvq_gpu = ds4_gpu_gvq_bind(m->map, m->size, records, count);
    free(records);
    return w->glmvq_gpu != NULL;
}
#endif
