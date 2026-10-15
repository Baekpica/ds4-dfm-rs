/* Rust validates the complete GVQ1 release before this pointer adapter.
 * Keep original names, native FP8 scales, and each expert's independent
 * payload. The source kv_b is head-interleaved; it is not MXQ's split K/V. */
static ds4_tensor *glmvq_tensor(const ds4_model *m, uint32_t il, const char *name) {
    char full[256];
    const int n = snprintf(full, sizeof(full), "model.language_model.layers.%u.%s", il, name);
    if (n < 0 || (size_t)n >= sizeof(full)) { ds4_die("GLM VQ tensor name too long"); }
    return required_tensor(m, full);
}

static ds4_tensor *glmvq_linear(const ds4_model *m, uint32_t il, const char *name) {
    ds4_tensor *t = glmvq_tensor(m, il, name);
    if (t->ndim != 2 || (t->type != DS4_TENSOR_BF16 &&
        t->type != DS4_TENSOR_GLM_FP8 && t->type != DS4_TENSOR_GLM_VQ)) {
        ds4_die("invalid GLM VQ matrix contract");
    }
    if (t->type != DS4_TENSOR_GLM_FP8) { return t; }

    char scale[192];
    const int n = snprintf(scale, sizeof(scale), "%s_scale_inv", name);
    if (n < 0 || (size_t)n >= sizeof(scale)) { ds4_die("GLM FP8 scale name too long"); }
    t->source_scale = glmvq_tensor(m, il, scale);
    enum { SCALE_BLOCK = 128 };
    if (t->dim[0] % SCALE_BLOCK || t->dim[1] % SCALE_BLOCK) {
        ds4_die("unaligned original GLM FP8 matrix");
    }
    tensor_expect_layout(t->source_scale, DS4_TENSOR_F32, 2,
        t->dim[0] / SCALE_BLOCK, t->dim[1] / SCALE_BLOCK, 0);
    return t;
}

static void glmvq_bind_experts(ds4_glmvq_experts *e, const ds4_model *m, uint32_t il) {
    const uint32_t type = il == GLMVQ_TRUNK ? DS4_TENSOR_GLM_FP8 : DS4_TENSOR_GLM_VQ;
    if (il < GLMVQ_TRUNK) {
        e->book = required_tensorf(m, "glm_vq.layer.%u.codebook.0", il);
        enum { VECTOR = 8, CODEWORDS = 8192 };
        tensor_expect_layout(e->book, DS4_TENSOR_GLM_BOOK, 2, VECTOR, CODEWORDS, 0);
    }
    for (uint32_t expert = 0; expert < GLMVQ_EXPERTS; expert++) {
        char name[96];
#define GLMVQ_EXPERT(field, src, cols, rows) do { \
        snprintf(name, sizeof(name), "mlp.experts.%u." src ".weight", expert); \
        e->field[expert] = glmvq_linear(m, il, name); \
        tensor_expect_layout(e->field[expert], type, 2, cols, rows, 0); \
    } while (0)
        GLMVQ_EXPERT(gate, "gate_proj", DS4_N_EMBD, DS4_N_FF_EXP);
        GLMVQ_EXPERT(up, "up_proj", DS4_N_EMBD, DS4_N_FF_EXP);
        GLMVQ_EXPERT(down, "down_proj", DS4_N_FF_EXP, DS4_N_EMBD);
#undef GLMVQ_EXPERT
    }
}

static void glmvq_bind_layer(ds4_layer_weights *l, const ds4_model *m, uint32_t il) {
#define GVQ_T(field, name) l->field = glmvq_tensor(m, il, name)
#define GVQ_L(field, name) l->field = glmvq_linear(m, il, name ".weight")
    GVQ_T(attn_norm, "input_layernorm.weight");
    GVQ_T(ffn_norm, "post_attention_layernorm.weight");
    if (il < GLMVQ_TRUNK) {
        GVQ_T(hc_attn_fn, "hc_attn_fn");
        GVQ_T(hc_attn_scale, "hc_attn_scale");
        GVQ_T(hc_attn_base, "hc_attn_base");
        GVQ_T(hc_ffn_fn, "hc_ffn_fn");
        GVQ_T(hc_ffn_scale, "hc_ffn_scale");
        GVQ_T(hc_ffn_base, "hc_ffn_base");
    }
    if (ds4_glm53_layer_is_kda(il)) {
        GVQ_L(kda_q, "self_attn.q_proj");
        GVQ_L(kda_k, "self_attn.k_proj");
        GVQ_L(kda_v, "self_attn.v_proj");
        GVQ_T(kda_q_conv, "self_attn.q_conv1d.weight");
        GVQ_T(kda_k_conv, "self_attn.k_conv1d.weight");
        GVQ_T(kda_v_conv, "self_attn.v_conv1d.weight");
        GVQ_L(kda_f_a, "self_attn.f_a_proj");
        GVQ_L(kda_f_b, "self_attn.f_b_proj");
        GVQ_T(kda_dt_bias, "self_attn.dt_bias");
        GVQ_T(kda_a_log, "self_attn.A_log");
        GVQ_L(kda_beta, "self_attn.b_proj");
        GVQ_L(kda_g_a, "self_attn.g_a_proj");
        GVQ_L(kda_g_b, "self_attn.g_b_proj");
        GVQ_T(kda_o_norm, "self_attn.o_norm.weight");
        GVQ_L(kda_output, "self_attn.o_proj");
    } else {
        GVQ_L(attn_q_a, "self_attn.q_a_proj");
        GVQ_T(attn_q_a_norm, "self_attn.q_a_layernorm.weight");
        GVQ_L(attn_q_b, "self_attn.q_b_proj");
        GVQ_L(attn_kv_a_mqa, "self_attn.kv_a_proj_with_mqa");
        GVQ_T(attn_kv_a_norm, "self_attn.kv_a_layernorm.weight");
        GVQ_L(attn_kv_b, "self_attn.kv_b_proj");
        GVQ_L(attn_output, "self_attn.o_proj");
        GVQ_L(indexer_attn_q_b, "self_attn.indexer.wq_b");
        GVQ_L(indexer_attn_k, "self_attn.indexer.wk");
        GVQ_T(indexer_k_norm, "self_attn.indexer.k_norm.weight");
        GVQ_T(indexer_k_norm_b, "self_attn.indexer.k_norm.bias");
        GVQ_L(indexer_proj, "self_attn.indexer.weights_proj");
        GVQ_T(indexer_compressor_ape, "self_attn.indexer.index_kpool_compress_ape");
        GVQ_T(indexer_compressor_gate, "self_attn.indexer.index_kpool_compress_gate");
    }
    if (il < DS4_N_LEADING_DENSE) {
        GVQ_L(ffn_gate, "mlp.gate_proj");
        GVQ_L(ffn_up, "mlp.up_proj");
        GVQ_L(ffn_down, "mlp.down_proj");
    } else {
        GVQ_L(ffn_gate_inp, "mlp.gate");
        GVQ_T(ffn_exp_probs_b, "mlp.gate.e_score_correction_bias");
        GVQ_L(ffn_gate_shexp, "mlp.shared_experts.gate_proj");
        GVQ_L(ffn_up_shexp, "mlp.shared_experts.up_proj");
        GVQ_L(ffn_down_shexp, "mlp.shared_experts.down_proj");
    }
    if (il == GLMVQ_TRUNK) {
        GVQ_L(nextn_eh_proj, "eh_proj");
        GVQ_T(nextn_enorm, "enorm.weight");
        GVQ_T(nextn_hnorm, "hnorm.weight");
        GVQ_T(nextn_shared_head_norm, "shared_head.norm.weight");
    }
#undef GVQ_L
#undef GVQ_T
}

static void glmvq_bind(ds4_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    w->glmvq = xcalloc(GLMVQ_LAYERS, sizeof(*w->glmvq));
    w->token_embd = required_tensor(m, "model.language_model.embed_tokens.weight");
    w->output_norm = required_tensor(m, "model.language_model.norm.weight");
    w->output = required_tensor(m, "lm_head.weight");
    for (uint32_t il = 0; il < GLMVQ_LAYERS; il++) {
        glmvq_bind_layer(&w->layer[il], m, il);
        if (il >= DS4_N_LEADING_DENSE) { glmvq_bind_experts(&w->glmvq[il], m, il); }
    }
}

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
static uint32_t glmvq_index(const ds4_model *m, const ds4_tensor *t) {
    return t ? (uint32_t)(t - m->tensors) : DS4_GVQ_NO_LINK;
}

/* Materialized source pointers belong to the weight owner, not each bank.
 * Sessions borrow the plan and allocate only arithmetic/state scratch. */
static bool glmvq_gpu_bind(ds4_weights *w, const ds4_model *m) {
    if (!w->glmvq || !m->n_tensors || m->n_tensors > UINT32_MAX) { return false; }
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
            (uint32_t)rows, (uint32_t)cols, glmvq_index(m, t->source_scale), DS4_GVQ_NO_LINK};
    }
    for (uint32_t il = DS4_N_LEADING_DENSE; il < GLMVQ_TRUNK; il++) {
        const ds4_glmvq_experts *e = &w->glmvq[il];
        const uint32_t book = glmvq_index(m, e->book);
        for (uint32_t expert = 0; expert < GLMVQ_EXPERTS; expert++) {
            records[glmvq_index(m, e->gate[expert])].book = book;
            records[glmvq_index(m, e->up[expert])].book = book;
            records[glmvq_index(m, e->down[expert])].book = book;
        }
    }
    w->glmvq_gpu = ds4_gpu_gvq_bind(m->map, m->size, records, (uint32_t)m->n_tensors);
    free(records);
    return w->glmvq_gpu != NULL;
}
#endif
