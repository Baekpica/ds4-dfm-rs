/* Rust validated the Ling-3.1 release (parts, GVQ1 headers, book and FP8
 * scale links, layouts) before this pointer adapter. Original HF names are
 * kept; the MTP block (layer 72) binds only when the mtp feature mapped its
 * two parts. FP8 kv_b stays one head-interleaved matrix [h*(128 k + 128 v)]. */
static bool ds4_ling31_layer_is_mla(uint32_t il) {
    return il == LING31_MTP || (il + 1u) % LING31_LAYER_GROUP == 0u;
}

// Routed clamp for layers 65-71, shared clamp for 64-71; MTP is unclamped.
static float ling31_clamp(uint32_t il, bool shared) {
    const uint32_t first = shared ? 64u : 65u;
    return il >= first && il < LING31_LAYERS ? 10.0f : INFINITY;
}

static ds4_tensor *ling31_tensor(const ds4_model *m, uint32_t il, const char *name) {
    char full[160];
    const int n = snprintf(full, sizeof(full), "model.layers.%u.%s", il, name);
    if (n < 0 || (size_t)n >= sizeof(full)) { ds4_die("Ling-3.1 tensor name too long"); }
    return required_tensor(m, full);
}

static ds4_tensor *ling31_linear(const ds4_model *m, uint32_t il, const char *name) {
    char weight[160];
    const int n = snprintf(weight, sizeof(weight), "%s.weight", name);
    if (n < 0 || (size_t)n >= sizeof(weight)) { ds4_die("Ling-3.1 tensor name too long"); }
    ds4_tensor *t = ling31_tensor(m, il, weight);
    if (t->ndim != 2 || (t->type != DS4_TENSOR_BF16 && t->type != DS4_TENSOR_GLM_FP8 &&
        t->type != DS4_TENSOR_GLM_VQ && t->type != DS4_TENSOR_Q4_K)) {
        ds4_die("invalid Ling-3.1 matrix contract");
    }
    if (t->type != DS4_TENSOR_GLM_FP8) { return t; }
    char scale[176];
    const int k = snprintf(scale, sizeof(scale), "%s.weight_scale_inv", name);
    if (k < 0 || (size_t)k >= sizeof(scale)) { ds4_die("Ling-3.1 FP8 scale name too long"); }
    t->source_scale = ling31_tensor(m, il, scale);
    // Ceil-division 128x128 blocks: [576,4096] carries [5,32] scales.
    enum { SCALE_BLOCK = 128 };
    tensor_expect_layout(t->source_scale, DS4_TENSOR_F32, 2,
        (t->dim[0] + SCALE_BLOCK - 1) / SCALE_BLOCK, (t->dim[1] + SCALE_BLOCK - 1) / SCALE_BLOCK, 0);
    return t;
}

static void ling31_bind_experts(ds4_ling31_experts *e, const ds4_model *m, uint32_t il) {
    e->book = ling31_tensor(m, il, "mlp.vq_codebook");
    if (e->book->type != DS4_TENSOR_GLM_BOOK || e->book->ndim != 2 || e->book->dim[0] != 8 ||
        (e->book->dim[1] != 4096 && e->book->dim[1] != 8192)) {
        ds4_die("invalid Ling-3.1 expert codebook");
    }
    for (uint32_t expert = 0; expert < LING31_EXPERTS; expert++) {
        char name[96];
#define LING31_EXPERT(field, src, cols, rows) do { \
        snprintf(name, sizeof(name), "mlp.experts.%u." src, expert); \
        e->field[expert] = ling31_linear(m, il, name); \
        tensor_expect_layout(e->field[expert], DS4_TENSOR_GLM_VQ, 2, cols, rows, 0); \
    } while (0)
        LING31_EXPERT(gate, "gate_proj", DS4_N_EMBD, DS4_N_FF_EXP);
        LING31_EXPERT(up, "up_proj", DS4_N_EMBD, DS4_N_FF_EXP);
        LING31_EXPERT(down, "down_proj", DS4_N_FF_EXP, DS4_N_EMBD);
#undef LING31_EXPERT
    }
}

static void ling31_bind_layer(ds4_layer_weights *l, const ds4_model *m, uint32_t il) {
#define L31_T(field, name) l->field = ling31_tensor(m, il, name)
#define L31_L(field, name) l->field = ling31_linear(m, il, name)
    L31_T(attn_norm, "input_layernorm.weight");
    L31_T(ffn_norm, "post_attention_layernorm.weight");
    if (!ds4_ling31_layer_is_mla(il)) {
        L31_L(kda_q, "attention.q_proj");
        L31_L(kda_k, "attention.k_proj");
        L31_L(kda_v, "attention.v_proj");
        L31_T(kda_q_conv, "attention.q_conv1d.weight");
        L31_T(kda_k_conv, "attention.k_conv1d.weight");
        L31_T(kda_v_conv, "attention.v_conv1d.weight");
        // no_kda_lora: full-rank decay and output gates (FP8, native scales).
        L31_L(kda_f_a, "attention.f_proj");
        L31_L(kda_g_a, "attention.g_proj");
        L31_T(kda_dt_bias, "attention.dt_bias");
        L31_T(kda_a_log, "attention.A_log");
        L31_L(kda_beta, "attention.b_proj");
        L31_T(kda_o_norm, "attention.o_norm.weight");
        L31_L(kda_output, "attention.o_proj");
    } else {
        L31_L(attn_q_a, "attention.q_a_proj");
        L31_T(attn_q_a_norm, "attention.q_a_layernorm.weight");
        L31_L(attn_q_b, "attention.q_b_proj");
        L31_L(attn_kv_a_mqa, "attention.kv_a_proj_with_mqa");
        L31_T(attn_kv_a_norm, "attention.kv_a_layernorm.weight");
        L31_L(attn_kv_b, "attention.kv_b_proj");
        L31_L(attn_gate, "attention.g_proj");
        L31_L(attn_output, "attention.dense");
    }
    if (il < LING31_DENSE) {
        L31_L(ffn_gate, "mlp.gate_proj");
        L31_L(ffn_up, "mlp.up_proj");
        L31_L(ffn_down, "mlp.down_proj");
    } else {
        L31_L(ffn_gate_inp, "mlp.gate");
        L31_T(ffn_exp_probs_b, "mlp.gate.expert_bias");
        L31_L(ffn_gate_shexp, "mlp.shared_experts.gate_proj");
        L31_L(ffn_up_shexp, "mlp.shared_experts.up_proj");
        L31_L(ffn_down_shexp, "mlp.shared_experts.down_proj");
    }
    if (il == LING31_MTP) {
        L31_L(nextn_eh_proj, "eh_proj");
        L31_T(nextn_enorm, "enorm.weight");
        L31_T(nextn_hnorm, "hnorm.weight");
        L31_T(nextn_shared_head_norm, "final_layernorm.weight");
    }
#undef L31_L
#undef L31_T
}

// Whether the release mapped the optional MTP parts (all-or-none in Rust).
static bool ling31_has_mtp(const ds4_model *m) {
    return model_find_tensor(m, "model.layers.72.input_layernorm.weight") != NULL;
}

static void ling31_bind(ds4_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    w->ling31 = xcalloc(LING31_LAYERS + 1u, sizeof(*w->ling31));
    w->token_embd = required_tensor(m, "model.word_embeddings.weight");
    w->output_norm = required_tensor(m, "model.norm.weight");
    w->output = required_tensor(m, "lm_head.weight");
    const uint32_t layers = ling31_has_mtp(m) ? LING31_LAYERS + 1u : LING31_LAYERS;
    for (uint32_t il = 0; il < layers; il++) {
        ling31_bind_layer(&w->layer[il], m, il);
        if (il >= LING31_DENSE) { ling31_bind_experts(&w->ling31[il], m, il); }
    }
}

/* Graph plan: shapes, row buffers and the memory quote (also CPU builds,
 * which refuse the family before allocating). */
enum {
    L31_EMBD = 4096, L31_HEADS = 64, L31_KDA_HEAD = 128, L31_KDA_DIM = L31_HEADS * L31_KDA_HEAD,
    L31_CONV = 4, L31_Q_RANK = 1536, L31_KV_LORA = 512, L31_ROPE = 64, L31_NOPE = 128,
    L31_KEY = L31_NOPE + L31_ROPE, L31_VALUE = 128, L31_KV_ROW = L31_KV_LORA + L31_ROPE,
    L31_QDIM = L31_HEADS * L31_KEY, L31_VDIM = L31_HEADS * L31_VALUE, L31_LATENT = L31_HEADS * L31_KV_LORA,
    L31_DENSE_FF = 10240, L31_FF = 1280, L31_EXPERTS = 512, L31_USED = 8,
    // 4096-row chunks prefill 10-14% faster once allocated (8K/16K; a cold first
    // allocation stalled them below 2048) but need 1.2 GiB more workspace, which a
    // 128 GB Spark lacks beside the trunk and a bank.
    L31_CONTEXT = 262144, L31_PREFILL_DEFAULT = 2048, L31_PREFILL_MAX = 4096,
    // Absorbed MLA below one expanded query tile (decode, verify, short appends);
    // the tensor-core absorbed kernel starts at 8 rows.
    L31_MLA_EXPAND_MIN = 64, L31_MLA_TC_MIN = 8,
    // Recursive MTP drafts per verify; each costs one 276 MiB KDA journal slot.
    L31_MTP_DRAFTS = 3, L31_MTP_ROWS = L31_MTP_DRAFTS + 1,
    // Tokenizer ids; the head's last 31 of 157,184 rows are padding.
    L31_TOKENS = 157153,
};
static const float L31_ROPE_THETA = 6000000.0f;
// Finite, so NaN checks and logits dumps hold; exp() underflows to zero.
static const float L31_PAD_LOGIT = -1e30f;

/* Row buffers, one allocation schema for the graph and its memory quote.
 * Aliases: dense FFN rows reuse k/v/f (KDA projections are dead by then),
 * expanded-MLA K/V reuse k+v, its merge operand f and both LSE vectors g. */
#define L31_ROW_BUFFERS(X) \
    X(token, 1u, int32_t) \
    X(positions, 3u, int32_t) \
    X(cur, L31_EMBD, float) \
    X(norm, L31_EMBD, float) \
    X(q, L31_QDIM, float) \
    X(k, L31_DENSE_FF, float) \
    X(v, L31_DENSE_FF, float) \
    X(f, L31_DENSE_FF, float) \
    X(g, L31_KDA_DIM, float) \
    X(heads, L31_KDA_DIM, float) \
    X(attn_out, L31_EMBD, float) \
    X(q_rank, L31_Q_RANK, float) \
    X(q_rank_norm, L31_Q_RANK, float) \
    X(kv_raw, L31_KV_ROW, float) \
    X(kv_norm, L31_KV_LORA, float) \
    X(head_gate, L31_HEADS, float) \
    X(beta, L31_HEADS, float) \
    X(ffn_out, L31_EMBD, float) \
    X(router, L31_EXPERTS, float) \
    X(selected, L31_USED, int32_t) \
    X(route_weights, L31_USED, float)


static uint32_t ling31_prefill_cap(uint32_t ctx) {
    uint32_t cap = L31_PREFILL_DEFAULT;
    const char *value = getenv("DS4_LING31_PREFILL_CHUNK");
    if (value && *value) {
        char *end = NULL;
        const unsigned long parsed = strtoul(value, &end, 10);
        if (end != value && !*end && parsed > 0u && parsed <= L31_PREFILL_MAX) { cap = (uint32_t)parsed; }
    }
    return cap < ctx ? cap : ctx;
}

static void ling31_layers(uint32_t *kda, uint32_t *mla) {
    *kda = *mla = 0u;
    for (uint32_t il = 0; il < LING31_LAYERS; il++) {
        if (ds4_ling31_layer_is_mla(il)) { (*mla)++; } else { (*kda)++; }
    }
}

static uint64_t ling31_kda_state_bytes(void) {
    // Recurrent tile, then the three convolution tails GVQ's KDA reads.
    return ((uint64_t)L31_KDA_DIM * L31_KDA_HEAD + 3u * (uint64_t)L31_KDA_DIM * L31_CONV) * sizeof(float);
}

// Recurrent state of every KDA layer plus the carried last hidden row.
static uint64_t ling31_state_bytes(void) {
    uint32_t kda, mla;
    ling31_layers(&kda, &mla);
    return kda * ling31_kda_state_bytes() + (uint64_t)L31_EMBD * sizeof(float);
}

static uint64_t ling31_control_bytes(void) {
    uint32_t kda, mla;
    ling31_layers(&kda, &mla);
    const uint64_t kda_ctl = (3u * (uint64_t)L31_KDA_DIM * L31_CONV + L31_HEADS + L31_KDA_DIM) * sizeof(float);
    // MLA: F32 latent norm weight, BF16 k_b and v_b (MTP's restored on demand).
    const uint64_t mla_ctl = L31_KV_LORA * sizeof(float) +
        2u * (uint64_t)L31_HEADS * L31_KV_LORA * L31_NOPE * sizeof(uint16_t);
    return kda * kda_ctl + (mla + 1u) * mla_ctl;
}

/* FP8 latent rows by default: 641 bytes per token and MLA layer against
 * 1,152 (ds4_l31_latent_bytes). DS4_LING31_LATENT=bf16 keeps BF16 rows, sim
 * stores FP8's values in them; Motif's kernels (DS4_LING31_FLASH=0) read BF16. */
static ds4_l31_latent ling31_latent_mode(void) {
    const char *latent = getenv("DS4_LING31_LATENT"), *flash = getenv("DS4_LING31_FLASH");
    if (latent && !strcmp(latent, "sim")) { return DS4_L31_LATENT_SIM; }
    if ((latent && !strcmp(latent, "bf16")) || (flash && !strcmp(flash, "0"))) { return DS4_L31_LATENT_BF16; }
    return DS4_L31_LATENT_FP8;
}

static uint64_t ling31_latent_bytes(uint32_t ctx) {
    return ds4_l31_latent_bytes(ctx, ling31_latent_mode()) + (uint64_t)ctx * L31_ROPE * sizeof(uint16_t);
}

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
// GVQ scratch: the largest temporary any projection, KDA chunk or q4p input
// needs at `rows` (restore+GEMM paths keep F32/BF16 temporaries).
static uint64_t ling31_gvq_scratch(uint32_t rows) {
    const uint64_t candidates[] = {
        ds4_gpu_gvq_kda_bytes(rows, L31_HEADS),
        ds4_gpu_gvq_linear_bytes(L31_KDA_DIM, L31_EMBD, rows, DS4_TENSOR_GLM_FP8, DS4_GVQ_BF16),
        ds4_gpu_gvq_linear_bytes(L31_EMBD, L31_KDA_DIM, rows, DS4_TENSOR_GLM_FP8, DS4_GVQ_BF16),
        ds4_gpu_gvq_linear_bytes(L31_EXPERTS, L31_EMBD, rows, DS4_TENSOR_BF16, DS4_GVQ_F32),
        ds4_gpu_gvq_linear_bytes(L31_HEADS, L31_EMBD, rows, DS4_TENSOR_BF16, DS4_GVQ_BF16),
        ds4_gpu_gvq_linear_bytes(L31_EMBD, 2u * L31_EMBD, rows, DS4_TENSOR_BF16, DS4_GVQ_BF16),
        // q4p prepared FP16 input of the widest Q4_K input (dense down).
        (uint64_t)rows * L31_DENSE_FF * sizeof(uint16_t) * 2u,
    };
    uint64_t bytes = 0;
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        if (candidates[i] > bytes) { bytes = candidates[i]; }
    }
    return bytes;
}
#endif

static uint64_t ling31_workspace_bytes(uint32_t rows) {
    uint64_t bytes = (uint64_t)DS4_N_VOCAB * sizeof(float) + L31_ROPE / 2u * sizeof(float);
#define L31_QUOTE(field, cols, type) bytes += (uint64_t)rows * (cols) * sizeof(type);
    L31_ROW_BUFFERS(L31_QUOTE)
#undef L31_QUOTE
    const uint32_t absorbed = rows < L31_MLA_EXPAND_MIN ? rows : L31_MLA_EXPAND_MIN;
    bytes += 2u * (uint64_t)absorbed * L31_LATENT * sizeof(float);
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
    bytes += ling31_gvq_scratch(rows) + ds4_gpu_gvq_workspace_linear(rows) + ds4_gpu_l31_moe_bytes(rows);
#endif
    return bytes;
}

// MTP: the draft layer's latent rows, the KDA journal (one slot per draft)
// and the hidden/logit rows of a verify.
static uint64_t ling31_mtp_bytes(uint32_t ctx, uint32_t drafts) {
    if (!drafts) { return 0; }
    return ling31_latent_bytes(ctx) + (uint64_t)drafts * ling31_state_bytes() +
        (L31_MTP_ROWS + 2u) * (uint64_t)L31_EMBD * sizeof(float) +
        (uint64_t)L31_MTP_ROWS * DS4_N_VOCAB * sizeof(float);
}

static ds4_context_memory ling31_memory(uint32_t ctx, uint32_t cap) {
    ds4_context_memory m = {0};
    if (!ctx || !cap || cap > ctx || ctx > L31_CONTEXT) { return m; }
    uint32_t kda, mla;
    ling31_layers(&kda, &mla);
    m.raw_cap = ctx;
    m.prefill_cap = cap;
    m.raw_bytes = (uint64_t)mla * ling31_latent_bytes(ctx) + ling31_state_bytes();
    m.scratch_bytes = ling31_workspace_bytes(cap) + ling31_control_bytes();
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
static uint32_t ling31_index(const ds4_model *m, const ds4_tensor *t) {
    return t ? (uint32_t)(t - m->tensors) : DS4_GVQ_NO_LINK;
}

/* Materialized source pointers belong to the weight owner, not each bank.
 * Expert records link their layer book; FP8 records carry their scale. */
static bool ling31_gpu_bind(ds4_weights *w, const ds4_model *m) {
    if (!w->ling31 || !m->n_tensors || m->n_tensors > UINT32_MAX) { return false; }
    ds4_gvq_record *records = xcalloc((size_t)m->n_tensors, sizeof(*records));
    for (uint32_t i = 0; i < m->n_tensors; i++) {
        const ds4_tensor *t = &m->tensors[i];
        const uint64_t cols = t->ndim ? t->dim[0] : 1u;
        const uint64_t rows = cols ? t->elements / cols : 0u;
        if (!cols || cols > UINT32_MAX || !rows || rows > UINT32_MAX) {
            free(records);
            return false;
        }
        records[i] = (ds4_gvq_record){t->abs_offset, t->bytes, t->type,
            (uint32_t)rows, (uint32_t)cols, ling31_index(m, t->source_scale), DS4_GVQ_NO_LINK,
            DS4_GVQ_NO_LINK};
    }
    for (uint32_t il = LING31_DENSE; il <= LING31_LAYERS; il++) {
        const ds4_ling31_experts *e = &w->ling31[il];
        if (!e->book) { continue; }
        const uint32_t book = ling31_index(m, e->book);
        for (uint32_t expert = 0; expert < LING31_EXPERTS; expert++) {
            records[ling31_index(m, e->gate[expert])].book = book;
            records[ling31_index(m, e->up[expert])].book = book;
            records[ling31_index(m, e->down[expert])].book = book;
        }
    }
    w->glmvq_gpu = ds4_gpu_gvq_bind(m->map, m->size, records, (uint32_t)m->n_tensors);
    free(records);
    return w->glmvq_gpu != NULL;
}
#endif
