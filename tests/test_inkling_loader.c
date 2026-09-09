/* Real GGUF descriptor gate; no GPU initialization or weight payload reads. */
#include "../ds4.c"

static unsigned checked;
static unsigned char *seen;

static void check_tensor(const ds4_model *m, const ds4_tensor *t,
                         const char *prefix, const char *suffix) {
    char name[192];
    snprintf(name, sizeof(name), "%s.%s", prefix, suffix);
    if (!t || !ds4_streq(t->name, name) || t < m->tensors ||
        t >= m->tensors + m->n_tensors || seen[t - m->tensors]++) {
        fprintf(stderr, "wrong, missing or duplicate binding: %s\n", name);
        exit(1);
    }
    checked++;
}

static void check_block(const ds4_model *m, const ds4_inkling_block *b,
                        const char *prefix, unsigned extent) {
#define CHECK(field, suffix) check_tensor(m, b->field, prefix, suffix)
    CHECK(attn_norm, "attn_norm.weight");
    CHECK(mlp_norm, "mlp_norm.weight");
    CHECK(q, "attn.wq_du.weight");
    CHECK(k, "attn.wk_dv.weight");
    CHECK(v, "attn.wv_dv.weight");
    CHECK(r, "attn.wr_du.weight");
    CHECK(o, "attn.wo_ud.weight");
    CHECK(q_norm, "attn.q_norm.weight");
    CHECK(k_norm, "attn.k_norm.weight");
    CHECK(rel_proj, "attn.rel_logits_proj.proj");
    CHECK(k_conv, "attn.k_sconv.weight");
    CHECK(v_conv, "attn.v_sconv.weight");
    CHECK(attn_conv, "attn_sconv.weight");
    CHECK(mlp_conv, "mlp_sconv.weight");
    if (b->rel_proj->dim[0] != extent || b->rel_proj->dim[1] != 16) {
        ds4_die("wrong relative attention geometry");
    }
    if (b->gate) {
        CHECK(gate, "mlp.gate.weight");
        CHECK(bias, "mlp.gate.bias");
        CHECK(scale, "mlp.gate.global_scale");
        CHECK(w13, "mlp.experts.w13_weight");
        CHECK(w2, "mlp.experts.w2_weight");
        CHECK(shared_w13, "mlp.shared_experts.shared_w13_weight");
        CHECK(shared_w2, "mlp.shared_experts.shared_w2_weight");
        if (b->scale->type != DS4_TENSOR_F32 ||
            b->shared_w13->dim[2] != 2 || b->shared_w2->dim[2] != 2) {
            ds4_die("wrong MoE scale or shared expert binding");
        }
    } else {
        CHECK(w13, "mlp.w13_dn.weight");
        CHECK(w2, "mlp.w2_md.weight");
        CHECK(scale, "mlp.global_scale");
        if (b->bias || b->shared_w13 || b->shared_w2 ||
            b->scale->type != DS4_TENSOR_BF16) {
            ds4_die("dense block bound MoE tensors or wrong scale");
        }
    }
#undef CHECK
}

static void check_coverage(const ds4_model *m, unsigned count) {
    if (checked != count || m->n_tensors != count) {
        ds4_die("tensor count mismatch");
    }
    for (unsigned i = 0; i < count; i++) {
        if (seen[i] != 1) {
            ds4_die("unbound tensor");
        }
    }
    free(seen);
    checked = 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <MQ85GB-first.gguf> <MTP-BF16.gguf>\n", argv[0]);
        return 2;
    }
    /* Simulate the validated Rust host selection, without a new C parser. */
    const ds4_host_shape host = {.variant = DS4_VARIANT_INKLING_SMALL};
    ds4_host_shape_install(&host);
    model_apply_host_shape();
    ds4_host_shape_clear();
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_INKLING || DS4_N_LAYER != 42 ||
        DS4_N_NEXTN_PREDICT != 8 || DS4_N_EMBD != 4096 ||
        DS4_N_VOCAB != 201024 || DS4_N_HEAD != 32 || DS4_N_HEAD_KV != 8 ||
        DS4_N_EXPERT_SHARED != 2 || DS4_USE_ROPE || !DS4_USE_QK_NORM) {
        ds4_die("wrong Inkling host shape");
    }
    ds4_model m;
    model_open(&m, argv[1], false, false);
    ds4_weights w;
    weights_bind(&w, &m, false, 0, UINT32_MAX, true, false);
    seen = calloc(m.n_tensors, 1);
    if (!seen || !weights_have_output_head(&w)) {
        ds4_die("missing output head or allocation failure");
    }
    check_tensor(&m, w.token_embd, "model.llm", "embed.weight");
    check_tensor(&m, w.output_norm, "model.llm", "norm.weight");
    check_tensor(&m, w.output, "model.llm", "unembed.weight");
    check_tensor(&m, w.inkling.embed_norm, "model.llm", "embed_norm.weight");
    for (unsigned i = 0; i < 42; i++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "model.llm.layers.%u", i);
        const ds4_inkling_block *b = &w.inkling.layer[i];
        if ((b->gate != NULL) != (i >= 2)) {
            ds4_die("wrong dense/MoE branch");
        }
        check_block(&m, b, prefix, (i + 1) % 6 ? 512 : 1024);
    }
    for (unsigned i = 0; i < 4; i++) {
        char suffix[64];
        snprintf(suffix, sizeof(suffix), "layers.linear_%u.weight", i);
        check_tensor(&m, w.inkling.image_linear[i], "model.visual", suffix);
        if (i < 3) {
            snprintf(suffix, sizeof(suffix), "layers.norm_%u.weight", i);
            check_tensor(&m, w.inkling.image_norm[i], "model.visual", suffix);
        }
    }
    check_tensor(&m, w.inkling.image_norm[3], "model.visual", "final_norm.weight");
    check_tensor(&m, w.inkling.audio_embed, "model.audio", "encoder.weight");
    check_tensor(&m, w.inkling.audio_norm, "model.audio", "final_norm.weight");
    check_coverage(&m, 888);
    model_close(&m);

    model_open(&m, argv[2], false, false);
    ds4_inkling_draft draft;
    inkling_bind_draft(&draft, &m);
    seen = calloc(m.n_tensors, 1);
    if (!seen) {
        ds4_die("allocation failure");
    }
    for (unsigned i = 0; i < 8; i++) {
        char prefix[96];
        snprintf(prefix, sizeof(prefix), "model.mtp.layers.%u", i);
        check_tensor(&m, draft.embed_norm[i], prefix, "embed_norm.weight");
        check_tensor(&m, draft.hidden_norm[i], prefix, "hidden_norm.weight");
        check_tensor(&m, draft.input_proj[i], prefix, "input_proj.weight");
        snprintf(prefix, sizeof(prefix), "model.mtp.layers.%u.transformer_block", i);
        if (draft.layer[i].gate) {
            ds4_die("MTP bound a routed block");
        }
        check_block(&m, &draft.layer[i], prefix, i == 1 || i == 3 ? 1024 : 512);
    }
    check_coverage(&m, 160);
    model_close(&m);
    puts("Inkling native binding: MQ85GB 888 + MTP-BF16 160 tensors PASS");
    return 0;
}
