/* Rust validated the A.X-K2 release (62 parts, GVQ1 v2 headers, book
 * links, layouts) before this pointer adapter. Original HF names are kept.
 * The BF16 token table lives in its own `embedding` part, the inventory's
 * last tensor: engine open keeps it out of the device copy and the host
 * reads one 14,336-byte row per token from the mapped part (page cache over
 * SSD). */

static ds4_tensor *axk2_tensor(const ds4_model *m, uint32_t il, const char *name) {
    char full[160];
    const int n = snprintf(full, sizeof(full), "model.layers.%u.%s", il, name);
    if (n < 0 || (size_t)n >= sizeof(full)) { ds4_die("A.X-K2 tensor name too long"); }
    return required_tensor(m, full);
}

// HF [rows, cols] matrices bind as GGUF [cols, rows].
static ds4_tensor *axk2_matrix(const ds4_model *m, uint32_t il, const char *name, uint32_t type,
        uint64_t rows, uint64_t cols) {
    ds4_tensor *t = axk2_tensor(m, il, name);
    tensor_expect_layout(t, type, 2, cols, rows, 0);
    return t;
}

static ds4_tensor *axk2_vector(const ds4_model *m, uint32_t il, const char *name, uint32_t type, uint64_t n) {
    ds4_tensor *t = axk2_tensor(m, il, name);
    tensor_expect_layout(t, type, 1, n, 0, 0);
    return t;
}

/* Graph plan: shapes, row buffers and the memory quote (also CPU builds,
 * which refuse the family before allocating). */
enum {
    AXK2_EMBD = 7168, AXK2_HEADS = 64, AXK2_Q_RANK = 1536, AXK2_KV_LORA = 512, AXK2_ROPE = 64,
    AXK2_KEY = 192, AXK2_VALUE = 128, AXK2_KV_ROW = AXK2_KV_LORA + AXK2_ROPE,
    AXK2_QB = AXK2_HEADS * (AXK2_KEY + AXK2_VALUE), AXK2_IDX_HEADS = 64, AXK2_IDX_DIM = 128,
    AXK2_TOPK = 2048, AXK2_GATE_RANK = 16, AXK2_DENSE_FF = 18432, AXK2_FF = 2048, AXK2_USED = 8,
    // max_position_embeddings: 131,072 trained positions under YaRN x 2.
    AXK2_CONTEXT = 262144,
    // One 262K bank fits beside the 98 GiB trunk with 2,048-row chunks
    // (handoff memory-budget.json); 4,096 rows need ~1 GiB more scratch.
    AXK2_PREFILL_DEFAULT = 2048, AXK2_PREFILL_MAX = 4096,
    // Prefill attention runs absorb -> flash -> value over row slices of
    // this height: q_abs and the latent output take 256 KiB per row.
    AXK2_ATTN_ROWS = 256,
    // Index scores of one slice of rows over every causal key, F32.
    AXK2_SCORE_BUDGET = 128 << 20,
};
// (1 + 0.1 ln 2)^2 / sqrt(192): yarn_apply_mscale with mscale_all_dim 1.
static const float AXK2_SCALE = 0.08252024f;

// DS4_AXK2_INDEX_KEYS=bf16 keeps the indexer keys as BF16 (256 bytes each),
// a diagnostic of the FP8 key rounding that the 262K budget cannot afford.
static ds4_axk2_keys axk2_key_format(void) {
    static int format = -1;
    if (format < 0) {
        const char *value = getenv("DS4_AXK2_INDEX_KEYS");
        format = value && !strcmp(value, "bf16") ? DS4_AXK2_KEYS_BF16 : DS4_AXK2_KEYS_FP8;
    }
    return (ds4_axk2_keys)format;
}

static void axk2_bind_experts(ds4_axk2_layer *e, const ds4_model *m, uint32_t il) {
    // recipe.json: layers 16, 20, ..., 40 keep K1024 (10-bit) books, the
    // others K512; each projection has its own book.
    const uint64_t words = il >= 16u && il <= 40u && il % 4u == 0u ? 1024u : 512u;
    e->book = axk2_tensor(m, il, "mlp.vq_codebook");
    e->book_up = axk2_tensor(m, il, "mlp.vq_codebook_up");
    e->book_down = axk2_tensor(m, il, "mlp.vq_codebook_down");
    tensor_expect_layout(e->book, DS4_TENSOR_GLM_BOOK, 2, 8, words, 0);
    tensor_expect_layout(e->book_up, DS4_TENSOR_GLM_BOOK, 2, 8, words, 0);
    tensor_expect_layout(e->book_down, DS4_TENSOR_GLM_BOOK, 2, 8, words, 0);
    for (uint32_t expert = 0; expert < AXK2_EXPERTS; expert++) {
        char name[96];
        snprintf(name, sizeof(name), "mlp.experts.%u.gate_proj.weight", expert);
        e->gate[expert] = axk2_matrix(m, il, name, DS4_TENSOR_GLM_VQ, AXK2_FF, AXK2_EMBD);
        snprintf(name, sizeof(name), "mlp.experts.%u.up_proj.weight", expert);
        e->up[expert] = axk2_matrix(m, il, name, DS4_TENSOR_GLM_VQ, AXK2_FF, AXK2_EMBD);
        snprintf(name, sizeof(name), "mlp.experts.%u.down_proj.weight", expert);
        e->down[expert] = axk2_matrix(m, il, name, DS4_TENSOR_GLM_VQ, AXK2_EMBD, AXK2_FF);
    }
}

static void axk2_bind_layer(ds4_layer_weights *l, ds4_axk2_layer *a, const ds4_model *m, uint32_t il) {
    // Gated RMSNorm before attention (every layer) and after it (MoE layers).
    l->attn_norm = axk2_vector(m, il, "input_layernorm.norm.weight", DS4_TENSOR_BF16, AXK2_EMBD);
    a->in_down = axk2_matrix(m, il, "input_layernorm.W_down.weight", DS4_TENSOR_BF16, AXK2_GATE_RANK, AXK2_EMBD);
    a->in_up = axk2_matrix(m, il, "input_layernorm.W_up.weight", DS4_TENSOR_BF16, AXK2_EMBD, AXK2_GATE_RANK);
    l->attn_q_a = axk2_matrix(m, il, "self_attn.q_a_proj.weight", DS4_TENSOR_Q4_K, AXK2_Q_RANK, AXK2_EMBD);
    l->attn_q_a_norm = axk2_vector(m, il, "self_attn.q_a_layernorm.weight", DS4_TENSOR_BF16, AXK2_Q_RANK);
    l->attn_q_b = axk2_matrix(m, il, "self_attn.q_b_proj.weight", DS4_TENSOR_Q4_K, AXK2_QB, 2u * AXK2_Q_RANK);
    l->attn_kv_a_mqa = axk2_matrix(m, il, "self_attn.kv_a_proj_with_mqa.weight", DS4_TENSOR_Q4_K, AXK2_KV_ROW,
        AXK2_EMBD);
    l->attn_kv_a_norm = axk2_vector(m, il, "self_attn.kv_a_layernorm.weight", DS4_TENSOR_BF16, AXK2_KV_LORA);
    l->attn_kv_b = axk2_matrix(m, il, "self_attn.kv_b_proj.weight", DS4_TENSOR_Q4_K,
        AXK2_HEADS * (AXK2_KEY - AXK2_ROPE + AXK2_VALUE), AXK2_KV_LORA);
    l->attn_output = axk2_matrix(m, il, "self_attn.o_proj.weight", DS4_TENSOR_Q4_K, AXK2_EMBD,
        AXK2_HEADS * AXK2_VALUE);
    l->attn_idx_q_b = axk2_matrix(m, il, "self_attn.indexer.wq_b.weight", DS4_TENSOR_Q8_0,
        AXK2_IDX_HEADS * AXK2_IDX_DIM, AXK2_Q_RANK);
    l->attn_idx_k = axk2_matrix(m, il, "self_attn.indexer.wk.weight", DS4_TENSOR_Q8_0, AXK2_IDX_DIM, AXK2_EMBD);
    l->attn_idx_k_norm = axk2_vector(m, il, "self_attn.indexer.k_norm.weight", DS4_TENSOR_BF16, AXK2_IDX_DIM);
    l->attn_idx_k_norm_bias = axk2_vector(m, il, "self_attn.indexer.k_norm.bias", DS4_TENSOR_BF16, AXK2_IDX_DIM);
    l->attn_idx_w = axk2_matrix(m, il, "self_attn.indexer.weights_proj.weight", DS4_TENSOR_BF16, AXK2_IDX_HEADS,
        AXK2_EMBD);
    if (il < AXK2_DENSE) {
        l->ffn_norm = axk2_vector(m, il, "post_attention_layernorm.weight", DS4_TENSOR_BF16, AXK2_EMBD);
        l->ffn_gate = axk2_matrix(m, il, "mlp.gate_proj.weight", DS4_TENSOR_Q4_K, AXK2_DENSE_FF, AXK2_EMBD);
        l->ffn_up = axk2_matrix(m, il, "mlp.up_proj.weight", DS4_TENSOR_Q4_K, AXK2_DENSE_FF, AXK2_EMBD);
        l->ffn_down = axk2_matrix(m, il, "mlp.down_proj.weight", DS4_TENSOR_Q4_K, AXK2_EMBD, AXK2_DENSE_FF);
        return;
    }
    l->ffn_norm = axk2_vector(m, il, "post_attention_layernorm.norm.weight", DS4_TENSOR_BF16, AXK2_EMBD);
    a->post_down = axk2_matrix(m, il, "post_attention_layernorm.W_down.weight", DS4_TENSOR_BF16, AXK2_GATE_RANK,
        AXK2_EMBD);
    a->post_up = axk2_matrix(m, il, "post_attention_layernorm.W_up.weight", DS4_TENSOR_BF16, AXK2_EMBD,
        AXK2_GATE_RANK);
    l->ffn_gate_inp = axk2_matrix(m, il, "mlp.gate.weight", DS4_TENSOR_BF16, AXK2_EXPERTS, AXK2_EMBD);
    l->ffn_exp_probs_b = axk2_vector(m, il, "mlp.gate.e_score_correction_bias", DS4_TENSOR_F32, AXK2_EXPERTS);
    l->ffn_gate_shexp = axk2_matrix(m, il, "mlp.shared_experts.gate_proj.weight", DS4_TENSOR_Q4_K, AXK2_FF,
        AXK2_EMBD);
    l->ffn_up_shexp = axk2_matrix(m, il, "mlp.shared_experts.up_proj.weight", DS4_TENSOR_Q4_K, AXK2_FF, AXK2_EMBD);
    l->ffn_down_shexp = axk2_matrix(m, il, "mlp.shared_experts.down_proj.weight", DS4_TENSOR_Q4_K, AXK2_EMBD,
        AXK2_FF);
    axk2_bind_experts(a, m, il);
}

/* ds4_gpu_axk2_q8 reads the indexer's raw Q8_0 wq_b (decode GEMV and
 * prefill MMQ): its aligned replica, 0.76 GiB over 61 layers, is never read
 * (MiniMax's rule). */
static bool axk2_raw_q8(const ds4_tensor *t) {
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_AXK2_VQ || t->type != DS4_TENSOR_Q8_0 || t->ndim != 2 ||
        t->dim[0] != AXK2_Q_RANK || t->dim[1] != AXK2_IDX_HEADS * AXK2_IDX_DIM) { return false; }
    for (uint32_t il = 0; il < AXK2_LAYERS; il++) {
        char name[96];
        snprintf(name, sizeof(name), "model.layers.%u.self_attn.indexer.wq_b.weight", il);
        if (ds4_streq(t->name, name)) { return true; }
    }
    return false;
}

static void axk2_catalog(const ds4_model *m) {
    if (!m->tensor_traits) { return; }
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (axk2_raw_q8(&m->tensors[i])) { m->tensor_traits[i] &= (uint8_t)~DS4_TCAT_ARTIFACT_ADDITIVE; }
    }
}

static void axk2_bind(ds4_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    w->axk2 = xcalloc(AXK2_LAYERS, sizeof(*w->axk2));
    w->token_embd = required_tensor(m, "model.embed_tokens.weight");
    tensor_expect_layout(w->token_embd, DS4_TENSOR_BF16, 2, AXK2_EMBD, DS4_N_VOCAB, 0);
    w->output_norm = required_tensor(m, "model.norm.weight");
    tensor_expect_layout(w->output_norm, DS4_TENSOR_BF16, 1, AXK2_EMBD, 0, 0);
    w->output = required_tensor(m, "lm_head.weight");
    tensor_expect_layout(w->output, DS4_TENSOR_Q4_K, 2, AXK2_EMBD, DS4_N_VOCAB, 0);
    for (uint32_t il = 0; il < AXK2_LAYERS; il++) { axk2_bind_layer(&w->layer[il], &w->axk2[il], m, il); }
    axk2_catalog(m);
}

/* Row buffers, one allocation schema for the graph and its memory quote.
 * F32 rows hold BF16-rounded values, except router logits, the indexer's
 * F32 head weights and prefill projections' F32 sums. q_b's rows also hold
 * the dense FFN's gate and the SwiGLU mid row; ff_up the up rows. */
#define AXK2_ROW_BUFFERS(X) \
    X(cur, AXK2_EMBD, float) \
    X(norm, AXK2_EMBD, float) \
    X(q_a, AXK2_Q_RANK, float) \
    X(q_r, AXK2_Q_RANK, float) \
    X(qin, 2u * AXK2_Q_RANK, float) \
    X(kv, AXK2_KV_ROW, float) \
    X(qb, AXK2_QB, float) \
    X(q, AXK2_HEADS * AXK2_KEY, float) \
    X(gate, AXK2_HEADS * AXK2_VALUE, float) \
    X(heads, AXK2_HEADS * AXK2_VALUE, float) \
    X(attn_out, AXK2_EMBD, float) \
    X(idx_q, AXK2_IDX_HEADS * AXK2_IDX_DIM, float) \
    X(idx_k, AXK2_IDX_DIM, float) \
    X(idx_w, AXK2_IDX_HEADS, float) \
    X(selected, AXK2_TOPK, uint32_t) \
    X(ff_up, AXK2_DENSE_FF, float) \
    X(shared, AXK2_EMBD, float) \
    X(ffn_out, AXK2_EMBD, float) \
    X(router, AXK2_EXPERTS, float) \
    X(route_ids, AXK2_USED, int32_t) \
    X(route_weights, AXK2_USED, float) \
    X(embed, AXK2_EMBD / 2u, uint32_t) \
    X(rope, AXK2_ROPE, float) \
    X(gmid, AXK2_GATE_RANK + 1u, float)

static uint32_t axk2_prefill_cap(uint32_t ctx) {
    uint32_t cap = AXK2_PREFILL_DEFAULT;
    const char *value = getenv("DS4_AXK2_PREFILL_CHUNK");
    if (value && *value) {
        char *end = NULL;
        const unsigned long parsed = strtoul(value, &end, 10);
        if (end != value && !*end && parsed > 0u && parsed <= AXK2_PREFILL_MAX) { cap = (uint32_t)parsed; }
    }
    return cap < ctx ? cap : ctx;
}

// Every layer keeps every position: FP8 latent rows, BF16 k_pe, index keys.
static uint64_t axk2_layer_kv_bytes(uint32_t ctx) {
    return ds4_axk2_latent_bytes(ctx) + (uint64_t)ctx * AXK2_ROPE * sizeof(uint16_t) +
        ds4_axk2_index_bytes(ctx, axk2_key_format());
}

static uint64_t axk2_kv_bytes(uint32_t ctx) { return (uint64_t)AXK2_LAYERS * axk2_layer_kv_bytes(ctx); }

static uint32_t axk2_attn_rows(uint32_t cap) { return cap < AXK2_ATTN_ROWS ? cap : AXK2_ATTN_ROWS; }

// Rows whose index scores over `ctx` keys fit the score budget (at least one).
static uint32_t axk2_score_rows(uint32_t ctx, uint32_t cap) {
    const uint64_t fit = AXK2_SCORE_BUDGET / ((uint64_t)ctx * sizeof(float));
    return fit < 1u ? 1u : fit < cap ? (uint32_t)fit : cap;
}

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
// GVQ scratch: the q4p prepared FP16 input of the widest Q4_K input (dense
// down, 18,432), or the F32 copy of the router for its prefill GEMM.
static uint64_t axk2_gvq_scratch(uint32_t rows) {
    const uint64_t q4p = (uint64_t)rows * AXK2_DENSE_FF * sizeof(uint16_t) * 2u;
    const uint64_t router = (uint64_t)AXK2_EXPERTS * AXK2_EMBD * sizeof(float);
    return q4p > router ? q4p : router;
}
#endif

/* Graph 0's scratch beyond its KV rows: logits, the row buffers, the
 * attention slices (q_abs, latent output, absorb operands, decode splits),
 * index scores, the GVQ/expert workspaces and the top-k tree's share of the
 * shared CUDA scratch: past 8,192 keys a row's candidates take fewer than
 * `keys` words, so the score slice bounds it. */
static uint64_t axk2_workspace_bytes(uint32_t ctx, uint32_t rows) {
    uint64_t bytes = (uint64_t)DS4_N_VOCAB * sizeof(float) + AXK2_ROPE / 2u * sizeof(float);
#define AXK2_QUOTE(field, cols, type) bytes += (uint64_t)rows * (cols) * sizeof(type);
    AXK2_ROW_BUFFERS(AXK2_QUOTE)
#undef AXK2_QUOTE
    const uint32_t slice = axk2_attn_rows(rows);
    bytes += 2u * (uint64_t)slice * AXK2_HEADS * AXK2_KV_LORA * sizeof(float);
    bytes += 2u * (uint64_t)axk2_score_rows(ctx, rows) * ctx * sizeof(float);
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
    bytes += ds4_gpu_axk2_absorb_bytes(slice) + ds4_gpu_axk2_flash_bytes() + axk2_gvq_scratch(rows) +
        ds4_gpu_gvq_workspace_linear(rows) + ds4_gpu_axk2_moe_bytes(rows);
#endif
    return bytes;
}

static ds4_context_memory axk2_memory(uint32_t ctx, uint32_t cap) {
    ds4_context_memory m = {0};
    if (!ctx || !cap || cap > ctx || ctx > AXK2_CONTEXT) { return m; }
    m.raw_cap = ctx;
    m.prefill_cap = cap;
    m.raw_bytes = axk2_kv_bytes(ctx);
    m.scratch_bytes = axk2_workspace_bytes(ctx, cap);
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
static uint32_t axk2_index(const ds4_model *m, const ds4_tensor *t) {
    return t ? (uint32_t)(t - m->tensors) : DS4_GVQ_NO_LINK;
}

/* Materialized source pointers belong to the weight owner, not each bank.
 * Expert records link their projection's book. The embedding, the
 * inventory's last tensor, has no device copy and no record. */
static bool axk2_gpu_bind(ds4_weights *w, const ds4_model *m) {
    if (!w->axk2 || !w->token_embd || m->n_tensors < 2 || m->n_tensors > UINT32_MAX ||
        axk2_index(m, w->token_embd) + 1u != m->n_tensors) { return false; }
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
    for (uint32_t il = AXK2_DENSE; il < AXK2_LAYERS; il++) {
        const ds4_axk2_layer *e = &w->axk2[il];
        for (uint32_t expert = 0; expert < AXK2_EXPERTS; expert++) {
            records[axk2_index(m, e->gate[expert])].book = axk2_index(m, e->book);
            records[axk2_index(m, e->up[expert])].book = axk2_index(m, e->book_up);
            records[axk2_index(m, e->down[expert])].book = axk2_index(m, e->book_down);
        }
    }
    w->glmvq_gpu = ds4_gpu_gvq_bind(m->map, m->size, records, count);
    free(records);
    return w->glmvq_gpu != NULL;
}
#endif
