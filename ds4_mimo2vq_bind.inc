/* The VQ release (Rust validated every part's hash, links and layouts) is the
 * same graph under original HF names: Q8_0 fused qkv ([q | k | v]), output
 * projection, dense FFN and head; BF16 norms, router and SWA sinks; GVQ1
 * experts per layer. The BF16 token table is the inventory's last tensor:
 * engine open keeps it out of the device copy and the host reads one
 * 8,192-byte row per token. The optional MTP part binds blocks 48..50. */
static ds4_tensor *mimo2vq_tensor(const ds4_model *m, uint32_t il, const char *name) {
    char full[160];
    const int n = snprintf(full, sizeof(full), "model.layers.%u.%s", il, name);
    if (n < 0 || (size_t)n >= sizeof(full)) { ds4_die("MiMo tensor name too long"); }
    return required_tensor(m, full);
}

static void mimo2vq_bind_experts(ds4_mimo2vq_layer *e, const ds4_model *m, uint32_t il) {
    char name[128];
    e->book = mimo2vq_tensor(m, il, "mlp.vq_codebook");
    snprintf(name, sizeof(name), "model.layers.%u.mlp.vq_codebook_residual", il);
    e->res_book = model_find_tensor(m, name);
    for (uint32_t x = 0; x < DS4_MIMO2VQ_EXPERTS; x++) {
        snprintf(name, sizeof(name), "mlp.experts.%u.gate_proj.weight", x);
        e->gate[x] = mimo2vq_tensor(m, il, name);
        snprintf(name, sizeof(name), "mlp.experts.%u.up_proj.weight", x);
        e->up[x] = mimo2vq_tensor(m, il, name);
        snprintf(name, sizeof(name), "mlp.experts.%u.down_proj.weight", x);
        e->down[x] = mimo2vq_tensor(m, il, name);
        if (!e->res_book) { continue; }
        snprintf(name, sizeof(name), "mlp.experts.%u.gate_proj.weight_residual", x);
        e->gate_res[x] = mimo2vq_tensor(m, il, name);
        snprintf(name, sizeof(name), "mlp.experts.%u.up_proj.weight_residual", x);
        e->up_res[x] = mimo2vq_tensor(m, il, name);
        snprintf(name, sizeof(name), "mlp.experts.%u.down_proj.weight_residual", x);
        e->down_res[x] = mimo2vq_tensor(m, il, name);
    }
}

static void mimo2vq_bind_block(ds4_layer_weights *l, const ds4_model *m, uint32_t il) {
    memset(l, 0, sizeof(*l));
    l->attn_norm = mimo2vq_tensor(m, il, "input_layernorm.weight");
    l->ffn_norm = mimo2vq_tensor(m, il, "post_attention_layernorm.weight");
    l->attn_qkv = mimo2vq_tensor(m, il, "self_attn.qkv_proj.weight");
    l->attn_output = mimo2vq_tensor(m, il, "self_attn.o_proj.weight");
    if (!mimo2_is_full(il)) { l->attn_sinks = mimo2vq_tensor(m, il, "self_attn.attention_sink_bias"); }
    if (il == 0) {
        l->ffn_gate = mimo2vq_tensor(m, il, "mlp.gate_proj.weight");
        l->ffn_up = mimo2vq_tensor(m, il, "mlp.up_proj.weight");
        l->ffn_down = mimo2vq_tensor(m, il, "mlp.down_proj.weight");
        return;
    }
    l->ffn_gate_inp = mimo2vq_tensor(m, il, "mlp.gate.weight");
    l->ffn_exp_probs_b = mimo2vq_tensor(m, il, "mlp.gate.e_score_correction_bias");
}

/* HF MTP block b is layer[48 + b]: input_layernorm before attention,
 * pre_mlp_layernorm before the MLP, final_layernorm before the shared head. */
static void mimo2vq_bind_mtp(ds4_layer_weights *l, const ds4_model *m, uint32_t block) {
    char name[128];
#define M2VQ_MTP(field, suffix) do { \
    snprintf(name, sizeof(name), "model.mtp.layers.%u." suffix, block); \
    l->field = required_tensor(m, name); \
} while (0)
    memset(l, 0, sizeof(*l));
    M2VQ_MTP(attn_norm, "input_layernorm.weight");
    M2VQ_MTP(attn_qkv, "self_attn.qkv_proj.weight");
    M2VQ_MTP(attn_output, "self_attn.o_proj.weight");
    M2VQ_MTP(attn_sinks, "self_attn.attention_sink_bias");
    M2VQ_MTP(ffn_norm, "pre_mlp_layernorm.weight");
    M2VQ_MTP(ffn_gate, "mlp.gate_proj.weight");
    M2VQ_MTP(ffn_up, "mlp.up_proj.weight");
    M2VQ_MTP(ffn_down, "mlp.down_proj.weight");
    M2VQ_MTP(nextn_eh_proj, "eh_proj.weight");
    M2VQ_MTP(nextn_enorm, "enorm.weight");
    M2VQ_MTP(nextn_hnorm, "hnorm.weight");
    M2VQ_MTP(layer_output_norm, "final_layernorm.weight");
#undef M2VQ_MTP
}

static void mimo2vq_bind(ds4_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    w->mimo2vq = xcalloc(MIMO2_LAYERS - 1u, sizeof(*w->mimo2vq));
    w->token_embd = required_tensor(m, "model.embed_tokens.weight");
    w->output_norm = required_tensor(m, "model.norm.weight");
    w->output = required_tensor(m, "lm_head.weight");
    for (uint32_t il = 0; il < MIMO2_LAYERS; il++) {
        mimo2vq_bind_block(&w->layer[il], m, il);
        if (il) { mimo2vq_bind_experts(&w->mimo2vq[il - 1u], m, il); }
    }
    if (!model_find_tensor(m, "model.mtp.layers.0.eh_proj.weight")) { return; }
    for (uint32_t b = 0; b < MIMO2_DRAFT_LAYERS; b++) { mimo2vq_bind_mtp(&w->layer[MIMO2_LAYERS + b], m, b); }
}

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
static uint32_t mimo2vq_index(const ds4_model *m, const ds4_tensor *t) {
    return t ? (uint32_t)(t - m->tensors) : DS4_GVQ_NO_LINK;
}

/* Materialized source pointers belong to the weight owner, not each bank.
 * Expert records link their layer book and, on R13+8 layers, their GVR8
 * record (linked to the residual book). The embedding, the inventory's last
 * tensor, has no device copy and no record. */
static bool mimo2vq_gpu_bind(ds4_weights *w, const ds4_model *m) {
    if (!w->mimo2vq || !w->token_embd || m->n_tensors < 2 || m->n_tensors > UINT32_MAX ||
        mimo2vq_index(m, w->token_embd) + 1u != m->n_tensors) { return false; }
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
    for (uint32_t il = 1; il < MIMO2_LAYERS; il++) {
        ds4_mimo2vq_layer *e = &w->mimo2vq[il - 1u];
        const uint32_t book = mimo2vq_index(m, e->book), res_book = mimo2vq_index(m, e->res_book);
        for (uint32_t x = 0; x < DS4_MIMO2VQ_EXPERTS; x++) {
            ds4_tensor *const matrix[3] = {e->gate[x], e->up[x], e->down[x]};
            ds4_tensor *const residual[3] = {e->gate_res[x], e->up_res[x], e->down_res[x]};
            for (uint32_t p = 0; p < 3; p++) {
                const uint32_t i = mimo2vq_index(m, matrix[p]);
                records[i].book = book;
                e->matrices[p * DS4_MIMO2VQ_EXPERTS + x] = i;
                if (!e->res_book) { continue; }
                records[i].res = mimo2vq_index(m, residual[p]);
                records[records[i].res].book = res_book;
            }
        }
    }
    w->glmvq_gpu = ds4_gpu_gvq_bind(m->map, m->size, records, count);
    free(records);
    return w->glmvq_gpu != NULL;
}
#endif
