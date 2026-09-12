/* Real descriptor bindings only: no GPU or weight payload reads. */
#include "../ds4.c"

static unsigned checked;
static unsigned char *seen;

static void check(const ds4_model *m, const ds4_tensor *t, const char *name) {
    if (!t || !ds4_streq(t->name, name) || t < m->tensors ||
        t >= m->tensors + m->n_tensors || seen[t - m->tensors]++) {
        fprintf(stderr, "wrong or duplicate binding: %s\n", name);
        exit(1);
    }
    checked++;
}

static void block(const ds4_model *m, const ds4_layer_weights *b, unsigned il) {
    char name[96];
#define CHECK(field, suffix) do { \
    snprintf(name, sizeof(name), "blk.%u.%s", il, suffix); \
    check(m, b->field, name); \
} while (0)
    CHECK(attn_norm, "attn_norm.weight");
    CHECK(attn_q, "attn_q.weight");
    CHECK(attn_k, "attn_k.weight");
    CHECK(attn_v, "attn_v.weight");
    CHECK(attn_q_norm, "attn_q_norm.weight");
    CHECK(attn_k_norm, "attn_k_norm.weight");
    CHECK(attn_gate, "attn_gate.weight");
    CHECK(attn_output, "attn_output.weight");
    CHECK(ffn_norm, "ffn_norm.weight");
    unsigned heads = il < 45 && il % 4 == 0 ? 64 : 96;
    if (b->attn_q->dim[1] != heads * 128 || b->attn_gate->dim[1] != heads ||
        b->attn_k->dim[1] != 1024 || b->attn_output->dim[0] != heads * 128) {
        ds4_die("Step attention geometry mismatch");
    }
    if (il < 3 || il >= 45) {
        CHECK(ffn_gate, "ffn_gate.weight");
        CHECK(ffn_up, "ffn_up.weight");
        CHECK(ffn_down, "ffn_down.weight");
        if (b->ffn_gate_exps || b->ffn_gate->dim[1] != 11264) {
            ds4_die("wrong Step dense block");
        }
    } else {
        CHECK(ffn_gate_inp, "ffn_gate_inp.weight");
        CHECK(ffn_exp_probs_b, "exp_probs_b.bias");
        CHECK(ffn_gate_exps, "ffn_gate_exps.weight");
        CHECK(ffn_up_exps, "ffn_up_exps.weight");
        CHECK(ffn_down_exps, "ffn_down_exps.weight");
        CHECK(ffn_gate_shexp, "ffn_gate_shexp.weight");
        CHECK(ffn_up_shexp, "ffn_up_shexp.weight");
        CHECK(ffn_down_shexp, "ffn_down_shexp.weight");
        if (b->ffn_gate || b->ffn_gate_exps->dim[2] != 288 ||
            b->ffn_gate_exps->dim[1] != 1280) {
            ds4_die("wrong Step routed block");
        }
    }
    if (il >= 45) {
        CHECK(nextn_eh_proj, "nextn.eh_proj.weight");
        CHECK(nextn_enorm, "nextn.enorm.weight");
        CHECK(nextn_hnorm, "nextn.hnorm.weight");
        CHECK(nextn_shared_head_norm, "nextn.shared_head_norm.weight");
        CHECK(nextn_shared_head_head, "nextn.shared_head_head.weight");
    }
#undef CHECK
}

static void coverage(const ds4_model *m, unsigned count) {
    if (checked != count || m->n_tensors != count) { ds4_die("tensor count mismatch"); }
    for (unsigned i = 0; i < count; i++) {
        if (seen[i] != 1) { ds4_die("unbound Step tensor"); }
    }
    free(seen);
    checked = 0;
}

int main(int argc, char **argv) {
    if (argc != 3) { return 2; }
    const ds4_host_shape host = {.variant = DS4_VARIANT_STEP37_FLASH};
    ds4_host_shape_install(&host);
    model_apply_host_shape();
    ds4_host_shape_clear();
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_STEP37 || DS4_N_LAYER != 45 ||
        DS4_N_NEXTN_PREDICT != 3 || DS4_N_EMBD != 4096 || DS4_N_VOCAB != 128896 ||
        DS4_N_HEAD != 64 || g_ds4_shape.n_swa_head != 96 || DS4_N_HEAD_KV != 8 ||
        DS4_N_EXPERT != 288 || DS4_N_EXPERT_USED != 8 || !DS4_USE_ROPE || !DS4_USE_QK_NORM) {
        ds4_die("wrong Step host shape");
    }
    for (unsigned sidecar = 0; sidecar < 2; sidecar++) {
        ds4_model m;
        model_open(&m, argv[sidecar + 1], false, false);
        ds4_weights w;
        if (sidecar) { step37_bind_draft(&w, &m); }
        else { weights_bind(&w, &m, false, 0, UINT32_MAX, true, false); }
        seen = calloc(m.n_tensors, 1);
        if (!seen || !weights_have_output_head(&w)) { ds4_die("allocation or head failure"); }
        check(&m, w.token_embd, "token_embd.weight");
        check(&m, w.output, "output.weight");
        check(&m, w.output_norm, "output_norm.weight");
        check(&m, w.step37_rope_freqs, "rope_freqs.weight");
        for (unsigned i = sidecar ? 45 : 0; i < (sidecar ? 48 : 45); i++) {
            block(&m, &w.layer[i], i);
            if (sidecar && w.layer[i].nextn_shared_head_head == w.output) {
                ds4_die("draft head incorrectly shares the main head");
            }
        }
        coverage(&m, sidecar ? 55 : 754);
        model_close(&m);
    }
    puts("Step native bindings: MQ83 754 + MTP Q8 55 tensors PASS");
    return 0;
}
