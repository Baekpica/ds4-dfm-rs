/* Rust validates the complete mixed artifact before this mmap pointer adapter.
 * MTP stays at source block numbers 48..50 and shares the root embed/head. */
static void mimo2_bind_block(ds4_layer_weights *l, const ds4_model *m, unsigned il) {
    memset(l, 0, sizeof(*l));
    l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", il);
    l->attn_qkv = required_tensorf(m, "blk.%u.attn_qkv.weight", il);
    l->attn_output = required_tensorf(m, "blk.%u.attn_output.weight", il);
    l->ffn_norm = required_tensorf(m, "blk.%u.ffn_norm.weight", il);
    if (!mimo2_is_full(il)) {
        l->attn_sinks = required_tensorf(m, "blk.%u.attn_sinks.weight", il);
    }
    if (il == 0 || il >= MIMO2_LAYERS) {
        l->ffn_gate = required_tensorf(m, "blk.%u.ffn_gate.weight", il);
        l->ffn_up = required_tensorf(m, "blk.%u.ffn_up.weight", il);
        l->ffn_down = required_tensorf(m, "blk.%u.ffn_down.weight", il);
    } else {
        l->ffn_gate_inp = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
        l->ffn_exp_probs_b = required_tensorf(m, "blk.%u.exp_probs_b.bias", il);
        l->ffn_gate_exps = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
        l->ffn_up_exps = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
        l->ffn_down_exps = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
    }
    if (il >= MIMO2_LAYERS) {
        l->nextn_eh_proj = required_tensorf(m, "blk.%u.nextn.eh_proj.weight", il);
        l->nextn_enorm = required_tensorf(m, "blk.%u.nextn.enorm.weight", il);
        l->nextn_hnorm = required_tensorf(m, "blk.%u.nextn.hnorm.weight", il);
        l->layer_output_norm = required_tensorf(m, "blk.%u.layer_output_norm.weight", il);
    }
}

static void mimo2_bind(ds4_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    w->token_embd = required_tensor(m, "token_embd.weight");
    w->output = required_tensor(m, "output.weight");
    w->output_norm = required_tensor(m, "output_norm.weight");
    for (unsigned il = 0; il < MIMO2_LAYERS + MIMO2_DRAFT_LAYERS; il++) {
        mimo2_bind_block(&w->layer[il], m, il);
    }
}
