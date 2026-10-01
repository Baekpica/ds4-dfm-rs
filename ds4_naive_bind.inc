/* Host validation owns names/types/layouts. Adapt its mmap descriptors to
 * the native graph without converting split GQA into a fused projection. */
static void naive_bind(ds4_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    w->token_embd = required_tensor(m, "token_embd.weight");
    w->output = required_tensor(m, "output.weight");
    w->output_norm = required_tensor(m, "output_norm.weight");
    for (unsigned il = 0; il < N05_LAYERS; il++) {
        ds4_layer_weights *l = &w->layer[il];
        l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", il);
        l->attn_q = required_tensorf(m, "blk.%u.attn_q.weight", il);
        l->attn_k = required_tensorf(m, "blk.%u.attn_k.weight", il);
        l->attn_v = required_tensorf(m, "blk.%u.attn_v.weight", il);
        l->attn_output = required_tensorf(m, "blk.%u.attn_output.weight", il);
        l->ffn_norm = required_tensorf(m, "blk.%u.ffn_norm.weight", il);
        if (naive_is_dsa(il)) {
            // These are direct hidden projections, not MLA/LoRA stages.
            l->indexer_attn_q_b = required_tensorf(m, "blk.%u.indexer.q_proj.weight", il);
            l->indexer_attn_k = required_tensorf(m, "blk.%u.indexer.k_proj.weight", il);
            l->indexer_proj = required_tensorf(m, "blk.%u.indexer.proj.weight", il);
            l->indexer_k_norm = required_tensorf(m, "blk.%u.indexer.k_norm.weight", il);
            l->indexer_k_norm_b = required_tensorf(m, "blk.%u.indexer.k_norm.bias", il);
        } else {
            l->attn_sinks = required_tensorf(m, "blk.%u.attn_sinks.weight", il);
        }
        if (il == 0) {
            l->ffn_gate = required_tensorf(m, "blk.%u.ffn_gate.weight", il);
            l->ffn_up = required_tensorf(m, "blk.%u.ffn_up.weight", il);
            l->ffn_down = required_tensorf(m, "blk.%u.ffn_down.weight", il);
            continue;
        }
        l->ffn_gate_inp = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
        l->ffn_exp_probs_b = required_tensorf(m, "blk.%u.exp_probs_b.bias", il);
        l->ffn_gate_exps = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
        l->ffn_up_exps = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
        l->ffn_down_exps = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
    }
}

static unsigned naive_prefill_cap(unsigned ctx) {
    unsigned cap = N05_PREFILL;
    const char *env = getenv("DS4_NAIVE_PREFILL_CHUNK");
    if (env && env[0]) {
        unsigned parsed = 0;
        if (sscanf(env, "%u", &parsed) == 1 && parsed > 0) { cap = parsed; }
    }
    if (cap > N05_PREFILL_MAX) { cap = N05_PREFILL_MAX; }
    return ctx < cap ? ctx : cap;
}

static ds4_context_memory naive_context_memory(unsigned ctx, unsigned cap) {
    const ds4_naive_memory plan = naive_memory(ctx, cap);
    ds4_context_memory m = {0};
    if (!plan.scratch) { return m; }
    m.raw_cap = ctx;
    m.prefill_cap = cap;
    m.raw_bytes = plan.dsa + plan.swa + plan.index;
    m.scratch_bytes = plan.scratch;
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}

typedef struct {
    const ds4_model *model;
    const ds4_tensor *fc, *hidden_norm, *output_norm, *mask;
    const ds4_tensor *markov1, *markov2, *conf, *bias;
    const ds4_tensor *attn_norm[N05_DF_LAYERS], *q[N05_DF_LAYERS], *k[N05_DF_LAYERS];
    const ds4_tensor *v[N05_DF_LAYERS], *o[N05_DF_LAYERS], *q_norm[N05_DF_LAYERS], *k_norm[N05_DF_LAYERS];
    const ds4_tensor *ffn_norm[N05_DF_LAYERS], *gate[N05_DF_LAYERS], *up[N05_DF_LAYERS], *down[N05_DF_LAYERS];
} ds4_naive_draft;

static const ds4_tensor *naive_df_tensor(const ds4_model *m, const char *name,
                                         unsigned type, unsigned ndim,
                                         unsigned d0, unsigned d1) {
    const ds4_tensor *t = model_find_tensor(m, name);
    if (!t || t->type != type || t->ndim != ndim || t->dim[0] != d0 ||
        (ndim == 2 && t->dim[1] != d1)) { return NULL; }
    return t;
}

/* The host checks pairing metadata before this narrow tensor adaptation. */
static bool naive_draft_bind(ds4_naive_draft *d, const ds4_model *m) {
    if (!d || !m || m->n_tensors != 63) { return false; }
    memset(d, 0, sizeof(*d));
    d->model = m;
#define N05_DF_BIND(field, name, type, ndim, d0, d1) do { \
    d->field = naive_df_tensor(m, name, type, ndim, d0, d1); \
    if (!d->field) { return false; } \
} while (0)
    N05_DF_BIND(fc, "fc.weight", DS4_TENSOR_Q8_0, 2, N05_DF_SLOT, N05_EMBED);
    N05_DF_BIND(hidden_norm, "enc.output_norm.weight", DS4_TENSOR_F32, 1, N05_EMBED, 0);
    N05_DF_BIND(output_norm, "output_norm.weight", DS4_TENSOR_F32, 1, N05_EMBED, 0);
    N05_DF_BIND(mask, "mask_embedding.weight", DS4_TENSOR_F32, 1, N05_EMBED, 0);
    N05_DF_BIND(markov1, "markov_w1.weight", DS4_TENSOR_Q8_0, 2, N05_DF_RANK, N05_VOCAB);
    N05_DF_BIND(markov2, "markov_w2.weight", DS4_TENSOR_Q8_0, 2, N05_DF_RANK, N05_VOCAB);
    N05_DF_BIND(conf, "confidence_head.weight", DS4_TENSOR_F32, 2, N05_EMBED + N05_DF_RANK, 1);
    N05_DF_BIND(bias, "confidence_head.bias", DS4_TENSOR_F32, 1, 1, 0);
    for (unsigned il = 0; il < N05_DF_LAYERS; il++) {
        char name[64];
#define N05_DF_LAYER(field, suffix, type, ndim, d0, d1) do { \
    snprintf(name, sizeof(name), "blk.%u." suffix, il); \
    N05_DF_BIND(field[il], name, type, ndim, d0, d1); \
} while (0)
        N05_DF_LAYER(attn_norm, "attn_norm.weight", DS4_TENSOR_F32, 1, N05_EMBED, 0);
        N05_DF_LAYER(ffn_norm, "ffn_norm.weight", DS4_TENSOR_F32, 1, N05_EMBED, 0);
        N05_DF_LAYER(q_norm, "attn_q_norm.weight", DS4_TENSOR_F32, 1, N05_DF_DIM, 0);
        N05_DF_LAYER(k_norm, "attn_k_norm.weight", DS4_TENSOR_F32, 1, N05_DF_DIM, 0);
        N05_DF_LAYER(q, "attn_q.weight", DS4_TENSOR_Q8_0, 2, N05_EMBED, N05_DF_HEADS * N05_DF_DIM);
        N05_DF_LAYER(k, "attn_k.weight", DS4_TENSOR_Q8_0, 2, N05_EMBED, N05_DF_KV * N05_DF_DIM);
        N05_DF_LAYER(v, "attn_v.weight", DS4_TENSOR_Q8_0, 2, N05_EMBED, N05_DF_KV * N05_DF_DIM);
        N05_DF_LAYER(o, "attn_output.weight", DS4_TENSOR_Q8_0, 2, N05_EMBED, N05_EMBED);
        N05_DF_LAYER(gate, "ffn_gate.weight", DS4_TENSOR_Q8_0, 2, N05_EMBED, N05_EMBED);
        N05_DF_LAYER(up, "ffn_up.weight", DS4_TENSOR_Q8_0, 2, N05_EMBED, N05_EMBED);
        N05_DF_LAYER(down, "ffn_down.weight", DS4_TENSOR_Q8_0, 2, N05_EMBED, N05_EMBED);
#undef N05_DF_LAYER
    }
#undef N05_DF_BIND
    return true;
}
