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
