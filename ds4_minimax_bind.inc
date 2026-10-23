/* Rust validated the MiniMax-M3.1 VQ release (parts, GVQ1/GVR8 headers,
 * book and residual links, layouts) before this pointer adapter. Original
 * HF names are kept. The BF16 token table lives in its own `embedding`
 * part, the inventory's last tensor: engine open keeps it out of the device
 * copy and the host reads one 12,288-byte row per token from the mapped
 * part (page cache over SSD). */

static ds4_tensor *minimax_tensor(const ds4_model *m, uint32_t il, const char *name) {
    char full[160];
    const int n = snprintf(full, sizeof(full), "language_model.model.layers.%u.%s", il, name);
    if (n < 0 || (size_t)n >= sizeof(full)) { ds4_die("MiniMax tensor name too long"); }
    return required_tensor(m, full);
}

// HF [rows, cols] matrices bind as GGUF [cols, rows].
static ds4_tensor *minimax_matrix(const ds4_model *m, uint32_t il, const char *name, uint32_t type,
        uint64_t rows, uint64_t cols) {
    ds4_tensor *t = minimax_tensor(m, il, name);
    tensor_expect_layout(t, type, 2, cols, rows, 0);
    return t;
}

static ds4_tensor *minimax_vector(const ds4_model *m, uint32_t il, const char *name, uint32_t type, uint64_t n) {
    ds4_tensor *t = minimax_tensor(m, il, name);
    tensor_expect_layout(t, type, 1, n, 0, 0);
    return t;
}

/* Graph plan: shapes, row buffers and the memory quote (also CPU builds,
 * which refuse the family before allocating). */
enum {
    MINIMAX_EMBD = 6144, MINIMAX_HEADS = 64, MINIMAX_KV_HEADS = 4, MINIMAX_HEAD = 128,
    MINIMAX_Q = MINIMAX_HEADS * MINIMAX_HEAD, MINIMAX_KV = MINIMAX_KV_HEADS * MINIMAX_HEAD,
    MINIMAX_INDEX = 128, MINIMAX_IQ = MINIMAX_KV_HEADS * MINIMAX_INDEX,
    MINIMAX_DENSE_FF = 12288, MINIMAX_FF = 3072, MINIMAX_USED = 4, MINIMAX_ROT = 64,
    MINIMAX_BLOCK = 128, MINIMAX_TOPK = 16, MINIMAX_CONTEXT = 1048576,
    // 2,048-row chunks: the reference's MSA selection is per query row, and
    // the 1M plan leaves no room for 4,096-row expert planes.
    MINIMAX_PREFILL_DEFAULT = 2048, MINIMAX_PREFILL_MAX = 4096,
    // Tokenizer ids; the head's last 3 of 200,064 rows are padding.
    MINIMAX_TOKENS = 200061,
    // NVFP4: per token and layer one 72-byte index key and 4 x 2 x 72 bytes
    // of K/V, the latter in (block, KV head) pages of 18,432 bytes.
    MINIMAX_FP4_ROW = 72, MINIMAX_KV_PAGE = 2 * MINIMAX_BLOCK * MINIMAX_FP4_ROW,
    MINIMAX_KV_BLOCK = MINIMAX_KV_HEADS * MINIMAX_KV_PAGE,
    // Decode attention: one partial (16 heads x (128 + max + sum)) per
    // (KV head, selected block).
    MINIMAX_SPLITS = MINIMAX_KV_HEADS * MINIMAX_TOPK,
};
// Persistent banks share graph 0's scratch (ds4_minimax_batch.inc).
static bool minimax_banks_enabled(void) { return true; }

// Block scores of the indexer, at most this many bytes per pass (a 1M
// context scores 512 rows at a time); a decode row needs 128 KiB at 1M.
static const uint64_t MINIMAX_SCORE_BUDGET = 64ull << 20;

static void minimax_bind_experts(ds4_minimax_layer *e, const ds4_model *m, uint32_t il) {
    e->book = minimax_tensor(m, il, "block_sparse_moe.vq_codebook");
    tensor_expect_layout(e->book, DS4_TENSOR_GLM_BOOK, 2, 8, 8192, 0);
    e->res_book = minimax_layer_has_residual(il) ? minimax_tensor(m, il, "block_sparse_moe.vq_codebook_residual")
                                                 : NULL;
    if (e->res_book) { tensor_expect_layout(e->res_book, DS4_TENSOR_GLM_BOOK, 2, 8, 256, 0); }
    for (uint32_t expert = 0; expert < MINIMAX_EXPERTS; expert++) {
        char name[112];
        snprintf(name, sizeof(name), "block_sparse_moe.experts.%u.w1.weight", expert);
        e->w1[expert] = minimax_matrix(m, il, name, DS4_TENSOR_GLM_VQ, MINIMAX_FF, MINIMAX_EMBD);
        snprintf(name, sizeof(name), "block_sparse_moe.experts.%u.w3.weight", expert);
        e->w3[expert] = minimax_matrix(m, il, name, DS4_TENSOR_GLM_VQ, MINIMAX_FF, MINIMAX_EMBD);
        snprintf(name, sizeof(name), "block_sparse_moe.experts.%u.w2.weight", expert);
        e->w2[expert] = minimax_matrix(m, il, name, DS4_TENSOR_GLM_VQ, MINIMAX_EMBD, MINIMAX_FF);
        if (!e->res_book) { continue; }
        snprintf(name, sizeof(name), "block_sparse_moe.experts.%u.w1.weight_residual", expert);
        e->r1[expert] = minimax_matrix(m, il, name, DS4_TENSOR_GLM_RES, MINIMAX_FF, MINIMAX_EMBD);
        snprintf(name, sizeof(name), "block_sparse_moe.experts.%u.w3.weight_residual", expert);
        e->r3[expert] = minimax_matrix(m, il, name, DS4_TENSOR_GLM_RES, MINIMAX_FF, MINIMAX_EMBD);
        snprintf(name, sizeof(name), "block_sparse_moe.experts.%u.w2.weight_residual", expert);
        e->r2[expert] = minimax_matrix(m, il, name, DS4_TENSOR_GLM_RES, MINIMAX_EMBD, MINIMAX_FF);
    }
}

static void minimax_bind_layer(ds4_layer_weights *l, ds4_minimax_layer *x, const ds4_model *m, uint32_t il) {
    l->attn_norm = minimax_vector(m, il, "input_layernorm.weight", DS4_TENSOR_BF16, MINIMAX_EMBD);
    l->ffn_norm = minimax_vector(m, il, "post_attention_layernorm.weight", DS4_TENSOR_BF16, MINIMAX_EMBD);
    l->attn_q = minimax_matrix(m, il, "self_attn.q_proj.weight", DS4_TENSOR_Q4_K, MINIMAX_Q, MINIMAX_EMBD);
    l->attn_k = minimax_matrix(m, il, "self_attn.k_proj.weight", DS4_TENSOR_Q4_K, MINIMAX_KV, MINIMAX_EMBD);
    l->attn_v = minimax_matrix(m, il, "self_attn.v_proj.weight", DS4_TENSOR_Q4_K, MINIMAX_KV, MINIMAX_EMBD);
    l->attn_output = minimax_matrix(m, il, "self_attn.o_proj.weight", DS4_TENSOR_Q4_K, MINIMAX_EMBD, MINIMAX_Q);
    l->attn_idx_q_b = minimax_matrix(m, il, "self_attn.index_q_proj.weight", DS4_TENSOR_Q8_0, MINIMAX_IQ,
        MINIMAX_EMBD);
    l->attn_idx_k = minimax_matrix(m, il, "self_attn.index_k_proj.weight", DS4_TENSOR_Q8_0, MINIMAX_INDEX,
        MINIMAX_EMBD);
    l->attn_q_norm = minimax_vector(m, il, "self_attn.q_norm.weight", DS4_TENSOR_BF16, MINIMAX_HEAD);
    l->attn_k_norm = minimax_vector(m, il, "self_attn.k_norm.weight", DS4_TENSOR_BF16, MINIMAX_HEAD);
    x->index_q_norm = minimax_vector(m, il, "self_attn.index_q_norm.weight", DS4_TENSOR_BF16, MINIMAX_HEAD);
    x->index_k_norm = minimax_vector(m, il, "self_attn.index_k_norm.weight", DS4_TENSOR_BF16, MINIMAX_HEAD);
    if (il < MINIMAX_DENSE) {
        l->ffn_gate = minimax_matrix(m, il, "mlp.gate_proj.weight", DS4_TENSOR_Q4_K, MINIMAX_DENSE_FF, MINIMAX_EMBD);
        l->ffn_up = minimax_matrix(m, il, "mlp.up_proj.weight", DS4_TENSOR_Q4_K, MINIMAX_DENSE_FF, MINIMAX_EMBD);
        l->ffn_down = minimax_matrix(m, il, "mlp.down_proj.weight", DS4_TENSOR_Q4_K, MINIMAX_EMBD, MINIMAX_DENSE_FF);
        return;
    }
    l->ffn_gate_inp = minimax_matrix(m, il, "block_sparse_moe.gate.weight", DS4_TENSOR_F32, MINIMAX_EXPERTS,
        MINIMAX_EMBD);
    l->ffn_exp_probs_b = minimax_vector(m, il, "block_sparse_moe.e_score_correction_bias", DS4_TENSOR_F32,
        MINIMAX_EXPERTS);
    l->ffn_gate_shexp = minimax_matrix(m, il, "block_sparse_moe.shared_experts.gate_proj.weight", DS4_TENSOR_Q4_K,
        MINIMAX_FF, MINIMAX_EMBD);
    l->ffn_up_shexp = minimax_matrix(m, il, "block_sparse_moe.shared_experts.up_proj.weight", DS4_TENSOR_Q4_K,
        MINIMAX_FF, MINIMAX_EMBD);
    l->ffn_down_shexp = minimax_matrix(m, il, "block_sparse_moe.shared_experts.down_proj.weight", DS4_TENSOR_Q4_K,
        MINIMAX_EMBD, MINIMAX_FF);
    minimax_bind_experts(x, m, il);
}

static void minimax_bind(ds4_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    w->minimax = xcalloc(MINIMAX_LAYERS, sizeof(*w->minimax));
    w->output_norm = required_tensor(m, "language_model.model.norm.weight");
    tensor_expect_layout(w->output_norm, DS4_TENSOR_BF16, 1, MINIMAX_EMBD, 0, 0);
    w->output = required_tensor(m, "language_model.lm_head.weight");
    tensor_expect_layout(w->output, DS4_TENSOR_Q4_K, 2, MINIMAX_EMBD, DS4_N_VOCAB, 0);
    for (uint32_t il = 0; il < MINIMAX_LAYERS; il++) {
        minimax_bind_layer(&w->layer[il], &w->minimax[il], m, il);
    }
    // The embedding part is mapped last (Rust release order).
    w->token_embd = required_tensor(m, "language_model.model.embed_tokens.weight");
    tensor_expect_layout(w->token_embd, DS4_TENSOR_BF16, 2, MINIMAX_EMBD, DS4_N_VOCAB, 0);
}

/* Row buffers, one allocation schema for the graph and its memory quote.
 * F32 rows hold BF16-rounded values, except the F32 router logits. */
#define MINIMAX_ROW_BUFFERS(X) \
    X(token, 1u, int32_t) \
    X(cur, MINIMAX_EMBD, float) \
    X(norm, MINIMAX_EMBD, float) \
    X(q, MINIMAX_Q, float) \
    X(k, MINIMAX_KV, float) \
    X(v, MINIMAX_KV, float) \
    X(iq, MINIMAX_IQ, float) \
    X(ik, MINIMAX_INDEX, float) \
    X(heads, MINIMAX_Q, float) \
    X(attn_out, MINIMAX_EMBD, float) \
    X(ffn_out, MINIMAX_EMBD, float) \
    X(shared, MINIMAX_EMBD, float) \
    X(ff_gate, MINIMAX_DENSE_FF, float) \
    X(ff_up, MINIMAX_DENSE_FF, float) \
    X(router, MINIMAX_EXPERTS, float) \
    X(selected, MINIMAX_USED, int32_t) \
    X(route_weights, MINIMAX_USED, float) \
    X(blocks, MINIMAX_KV_HEADS * MINIMAX_TOPK, int32_t) \
    X(embed, MINIMAX_EMBD / 2u, uint32_t) \
    X(rope, MINIMAX_ROT, float)

static uint32_t minimax_prefill_cap(uint32_t ctx) {
    uint32_t cap = MINIMAX_PREFILL_DEFAULT;
    const char *value = getenv("DS4_MINIMAX_PREFILL_CHUNK");
    if (value && *value) {
        char *end = NULL;
        const unsigned long parsed = strtoul(value, &end, 10);
        if (end != value && !*end && parsed > 0u && parsed <= MINIMAX_PREFILL_MAX) { cap = (uint32_t)parsed; }
    }
    return cap < ctx ? cap : ctx;
}

static uint64_t minimax_blocks(uint32_t ctx) {
    return ((uint64_t)ctx + MINIMAX_BLOCK - 1u) / MINIMAX_BLOCK;
}

// One layer's index keys and K/V pages; every layer keeps the full history
// (MSA's selection saves reads, not storage).
static uint64_t minimax_index_bytes(uint32_t ctx) {
    return (uint64_t)ctx * MINIMAX_FP4_ROW;
}

static uint64_t minimax_page_bytes(uint32_t ctx) {
    return minimax_blocks(ctx) * MINIMAX_KV_BLOCK;
}

static uint64_t minimax_kv_bytes(uint32_t ctx) {
    return (uint64_t)MINIMAX_LAYERS * (minimax_index_bytes(ctx) + minimax_page_bytes(ctx));
}

// Indexer scores: every row of a chunk against every block below it, at
// most MINIMAX_SCORE_BUDGET bytes per pass (one decode row always fits).
static uint64_t minimax_score_bytes(uint32_t ctx, uint32_t rows) {
    const uint64_t row = (uint64_t)MINIMAX_KV_HEADS * minimax_blocks(ctx) * sizeof(float);
    const uint64_t all = (uint64_t)rows * row;
    const uint64_t cap = MINIMAX_SCORE_BUDGET > row ? MINIMAX_SCORE_BUDGET / row * row : row;
    return all < cap ? all : cap;
}

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
// GVQ scratch: the q4p prepared FP16 input of the widest Q4_K input (dense
// down, 12,288), which bounds every prefill projection.
static uint64_t minimax_gvq_scratch(uint32_t rows) {
    return (uint64_t)rows * MINIMAX_DENSE_FF * sizeof(uint16_t) * 2u;
}
#endif

static uint64_t minimax_workspace_bytes(uint32_t ctx, uint32_t rows) {
    uint64_t bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
#define MINIMAX_QUOTE(field, cols, type) bytes += (uint64_t)rows * (cols) * sizeof(type);
    MINIMAX_ROW_BUFFERS(MINIMAX_QUOTE)
#undef MINIMAX_QUOTE
    bytes += (uint64_t)MINIMAX_SPLITS * (MINIMAX_HEADS / MINIMAX_KV_HEADS) * (MINIMAX_HEAD + 2u) * sizeof(float);
    bytes += minimax_score_bytes(ctx, rows);
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
    bytes += minimax_gvq_scratch(rows) + ds4_gpu_gvq_workspace_linear(rows) + ds4_gpu_minimax_moe_bytes(rows);
#endif
    return bytes;
}

static ds4_context_memory minimax_memory(uint32_t ctx, uint32_t cap) {
    ds4_context_memory m = {0};
    if (!ctx || !cap || cap > ctx || ctx > MINIMAX_CONTEXT) { return m; }
    m.raw_cap = ctx;
    m.prefill_cap = cap;
    m.raw_bytes = minimax_kv_bytes(ctx);
    m.scratch_bytes = minimax_workspace_bytes(ctx, cap);
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
static uint32_t minimax_index(const ds4_model *m, const ds4_tensor *t) {
    return t ? (uint32_t)(t - m->tensors) : DS4_GVQ_NO_LINK;
}

/* Materialized source pointers belong to the weight owner, not each bank.
 * Expert records link their layer book and, on R13+8 layers, their GVR8
 * residual (whose record links the residual book). The embedding, the
 * inventory's last tensor, has no device copy and no record. */
static bool minimax_gpu_bind(ds4_weights *w, const ds4_model *m) {
    if (!w->minimax || !w->token_embd || m->n_tensors < 2 || m->n_tensors > UINT32_MAX ||
        minimax_index(m, w->token_embd) + 1u != m->n_tensors) { return false; }
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
    for (uint32_t il = MINIMAX_DENSE; il < MINIMAX_LAYERS; il++) {
        const ds4_minimax_layer *e = &w->minimax[il];
        const uint32_t book = minimax_index(m, e->book), res_book = minimax_index(m, e->res_book);
        for (uint32_t expert = 0; expert < MINIMAX_EXPERTS; expert++) {
            const ds4_tensor *parent[3] = {e->w1[expert], e->w3[expert], e->w2[expert]};
            const ds4_tensor *residual[3] = {e->r1[expert], e->r3[expert], e->r2[expert]};
            for (uint32_t p = 0; p < 3; p++) {
                ds4_gvq_record *r = &records[minimax_index(m, parent[p])];
                r->book = book;
                if (!e->res_book) { continue; }
                r->res = minimax_index(m, residual[p]);
                records[r->res].book = res_book;
            }
        }
    }
    w->glmvq_gpu = ds4_gpu_gvq_bind(m->map, m->size, records, count);
    free(records);
    return w->glmvq_gpu != NULL;
}
#endif
