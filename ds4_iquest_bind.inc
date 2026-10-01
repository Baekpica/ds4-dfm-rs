/* Canonical custom iquest_q1 tensors. The MTP decoder borrows the main
 * embedding and output head and never opens a second weight mapping. */
enum { IQ_TENSORS = 1254 };
typedef enum { IQ_TARGET_BLOCK, IQ_MTP_BLOCK } iquest_block_kind;

static ds4_tensor *iquest_tensor(const ds4_model *m, const char *name,
        unsigned type, unsigned rank, unsigned d0, unsigned d1, unsigned d2) {
    ds4_tensor *t = required_tensor(m, name);
    tensor_expect_layout(t, type, rank, d0, d1, d2);
    return t;
}

static void iquest_pin(const ds4_model *m, const char *key, const char *expected) {
    ds4_str value = {0};
    if (model_get_string(m, key, &value) && ds4_streq(value, expected)) { return; }
    fprintf(stderr, "ds4: unsupported IQuest-Q1 artifact metadata: %s\n", key);
    exit(1);
}

static void iquest_block_bind(ds4_layer_weights *l, const ds4_model *m,
        const char *prefix, unsigned layer, iquest_block_kind kind) {
    const bool mtp = kind == IQ_MTP_BLOCK;
    const bool dense = !mtp && layer == 0;
    const unsigned qtype = mtp ? DS4_TENSOR_Q8_0 : DS4_TENSOR_Q5_K;
    const unsigned kvtype = mtp ? DS4_TENSOR_Q8_0 : DS4_TENSOR_Q6_K;
    char name[96];
#define IQ_BIND(field, suffix, type, rank, d0, d1, d2) do { \
    snprintf(name, sizeof(name), "%s.%s.weight", prefix, suffix); \
    l->field = iquest_tensor(m, name, type, rank, d0, d1, d2); \
} while (0)
    IQ_BIND(attn_norm, "attn_norm", DS4_TENSOR_F32, 1, IQ_EMBED, 0, 0);
    IQ_BIND(attn_output_norm, "attn_output_norm", DS4_TENSOR_F32, 1, IQ_EMBED, 0, 0);
    IQ_BIND(attn_q, "attn_q", qtype, 2, IQ_EMBED, IQ_HEADS * IQ_HEAD, 0);
    IQ_BIND(attn_k, "attn_k", kvtype, 2, IQ_EMBED, IQ_KV_HEADS * IQ_HEAD, 0);
    IQ_BIND(attn_v, "attn_v", kvtype, 2, IQ_EMBED, IQ_KV_HEADS * IQ_HEAD, 0);
    IQ_BIND(attn_output, "attn_output", qtype, 2, IQ_HEADS * IQ_HEAD, IQ_EMBED, 0);
    IQ_BIND(attn_q_norm, "attn_q_norm", DS4_TENSOR_F32, 1, IQ_HEAD, 0, 0);
    IQ_BIND(attn_k_norm, "attn_k_norm", DS4_TENSOR_F32, 1, IQ_HEAD, 0, 0);
    IQ_BIND(attn_sink_k, "attn_sink_k", DS4_TENSOR_F32, 2, IQ_HEAD, IQ_KV_HEADS, 0);
    IQ_BIND(ffn_norm, "ffn_norm", DS4_TENSOR_F32, 1, IQ_EMBED, 0, 0);
    if (dense || mtp) {
        IQ_BIND(ffn_output_norm, "ffn_output_norm", DS4_TENSOR_F32, 1, IQ_EMBED, 0, 0);
    }
    if (dense) {
        IQ_BIND(ffn_gate, "ffn_gate", qtype, 2, IQ_EMBED, IQ_DENSE, 0);
        IQ_BIND(ffn_up, "ffn_up", qtype, 2, IQ_EMBED, IQ_DENSE, 0);
        IQ_BIND(ffn_down, "ffn_down", qtype, 2, IQ_DENSE, IQ_EMBED, 0);
    } else {
        /* The published artifact promotes 15 independent projections. Gate
         * and up types deliberately differ at layers 67, 78 and 80. */
        const bool pair = (layer >= 62 && layer <= 66) || layer == 68;
        const unsigned gate = mtp ? DS4_TENSOR_Q4_K :
            pair || layer == 67 ? DS4_TENSOR_IQ2_XS : DS4_TENSOR_IQ2_XXS;
        const unsigned up = mtp ? DS4_TENSOR_Q4_K :
            pair || layer == 78 || layer == 80 ? DS4_TENSOR_IQ2_XS : DS4_TENSOR_IQ2_XXS;
        const unsigned down = mtp ? DS4_TENSOR_Q4_K : DS4_TENSOR_IQ2_XXS;
        IQ_BIND(ffn_gate_inp, "ffn_gate_inp", DS4_TENSOR_F32, 2, IQ_EMBED, IQ_EXPERTS, 0);
        IQ_BIND(ffn_gate_exps, "ffn_gate_exps", gate, 3, IQ_EMBED, IQ_FF, IQ_EXPERTS);
        IQ_BIND(ffn_up_exps, "ffn_up_exps", up, 3, IQ_EMBED, IQ_FF, IQ_EXPERTS);
        IQ_BIND(ffn_down_exps, "ffn_down_exps", down, 3, IQ_FF, IQ_EMBED, IQ_EXPERTS);
    }
#undef IQ_BIND
}

static void iquest_bind(ds4_weights *w, const ds4_model *m) {
    if (m->n_tensors != IQ_TENSORS) { ds4_die("IQuest-Q1 tensor count mismatch"); }
    iquest_pin(m, "general.source.huggingface.repository", "IQuestLab/IQuest-Q1");
    iquest_pin(m, "general.source.huggingface.revision", "5c21b0630586ef77d38cff1094b8cf37a417fcd8");
    iquest_pin(m, "iquest_q1.tensor_layout", "canonical-v1-source-fc-gate-up-split");
    iquest_pin(m, "iquest_q1.quantization.minimum", "IQ2_XXS");
    memset(w, 0, sizeof(*w));
    w->token_embd = iquest_tensor(m, "token_embd.weight", DS4_TENSOR_Q6_K, 2, IQ_EMBED, IQ_VOCAB, 0);
    w->output = iquest_tensor(m, "output.weight", DS4_TENSOR_Q8_0, 2, IQ_EMBED, IQ_VOCAB, 0);
    w->output_norm = iquest_tensor(m, "output_norm.weight", DS4_TENSOR_F32, 1, IQ_EMBED, 0, 0);
    for (unsigned il = 0; il < IQ_LAYERS; il++) {
        char prefix[32]; snprintf(prefix, sizeof(prefix), "blk.%u", il);
        iquest_block_bind(&w->layer[il], m, prefix, il, IQ_TARGET_BLOCK);
    }
    iquest_block_bind(&w->iquest_mtp, m, "mtp.0", 0, IQ_MTP_BLOCK);
    w->iquest_eh = iquest_tensor(m, "mtp.0.eh_proj.weight", DS4_TENSOR_Q8_0, 2, 2 * IQ_EMBED, IQ_EMBED, 0);
    w->iquest_enorm = iquest_tensor(m, "mtp.0.enorm.weight", DS4_TENSOR_F32, 1, IQ_EMBED, 0, 0);
    w->iquest_hnorm = iquest_tensor(m, "mtp.0.hnorm.weight", DS4_TENSOR_F32, 1, IQ_EMBED, 0, 0);
    w->iquest_mtp_norm = iquest_tensor(m, "mtp.0.output_norm.weight", DS4_TENSOR_F32, 1, IQ_EMBED, 0, 0);
}

static unsigned iquest_prefill_cap(unsigned ctx) {
    unsigned cap = IQ_PREFILL;
    const char *env = getenv("DS4_IQUEST_PREFILL_CHUNK");
    if (env && env[0]) {
        unsigned parsed = 0;
        if (sscanf(env, "%u", &parsed) == 1 && parsed > 0) { cap = parsed; }
    }
    if (cap > IQ_PREFILL_MAX) { cap = IQ_PREFILL_MAX; }
    return ctx < cap ? ctx : cap;
}

static unsigned iquest_kv_cap(unsigned il, unsigned ctx, unsigned cap) {
    if (iquest_full(il)) { return ctx; }
    const unsigned ring = IQ_WINDOW + cap - 1;
    return ctx < ring ? ctx : ring;
}

static ds4_context_memory iquest_memory(unsigned ctx, unsigned cap) {
    ds4_context_memory m = {0};
    if (!ctx || ctx > IQ_CONTEXT || !cap || cap > ctx || cap > IQ_PREFILL_MAX) { return m; }
    m.raw_cap = ctx; m.prefill_cap = cap;
    for (unsigned il = 0; il < IQ_LAYERS; il++) {
        m.raw_bytes += (uint64_t)iquest_kv_cap(il, ctx, cap) * IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8);
    }
    /* Exact tensor allocation quote, including bounded teacher states and
     * MTP trial journals. No 512k-by-hidden FP32 plane is retained. */
    const uint64_t scratch_floats = (uint64_t)cap * (
        5ull * IQ_EMBED + 2ull * IQ_HEADS * IQ_HEAD + 2ull * IQ_KV_HEADS * IQ_HEAD +
        3ull * IQ_DENSE + IQ_EXPERTS + 2ull * IQ_USED +
        3ull * IQ_USED * IQ_FF + IQ_USED * IQ_EMBED + 2ull + 4ull * IQ_EMBED);
    m.scratch_bytes = scratch_floats * sizeof(float) +
        2ull * IQ_VOCAB * sizeof(float) + 3ull * IQ_EMBED * sizeof(float) +
        2ull * (IQ_DRAFT_WINDOW + IQ_DRAFT_SLOTS) * IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8) +
        (uint64_t)(IQ_DRAFT_SLOTS + 1) * (IQ_LAYERS + 1) * IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8) +
        (uint64_t)(IQ_DRAFT_SLOTS + 1) * IQ_EMBED * sizeof(float);
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}
