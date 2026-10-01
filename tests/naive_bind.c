/* Check the actual native binder against an independent source directory. */
#include "../ds4.c"
#include <assert.h>

static bool seen[613];

static void check_tensor(const ds4_model *m, ds4_tensor *t, const char *name) {
    assert(t && ds4_streq(t->name, name));
    const uint64_t index = (uint64_t)(t - m->tensors);
    assert(index < m->n_tensors && !seen[index]);
    seen[index] = true;
}

static void check_layer(const ds4_model *m, const ds4_layer_weights *l, unsigned il) {
    char name[128];
#define CHECK_FIELD(field, suffix) do { \
    snprintf(name, sizeof(name), "blk.%u." suffix, il); \
    check_tensor(m, l->field, name); \
} while (0)
    CHECK_FIELD(attn_norm, "attn_norm.weight");
    CHECK_FIELD(attn_q, "attn_q.weight");
    CHECK_FIELD(attn_k, "attn_k.weight");
    CHECK_FIELD(attn_v, "attn_v.weight");
    CHECK_FIELD(attn_output, "attn_output.weight");
    CHECK_FIELD(ffn_norm, "ffn_norm.weight");
    assert(!l->attn_qkv);
    if (naive_is_dsa(il)) {
        assert(!l->attn_sinks);
        CHECK_FIELD(indexer_attn_q_b, "indexer.q_proj.weight");
        CHECK_FIELD(indexer_attn_k, "indexer.k_proj.weight");
        CHECK_FIELD(indexer_proj, "indexer.proj.weight");
        CHECK_FIELD(indexer_k_norm, "indexer.k_norm.weight");
        CHECK_FIELD(indexer_k_norm_b, "indexer.k_norm.bias");
    } else {
        CHECK_FIELD(attn_sinks, "attn_sinks.weight");
        assert(!l->indexer_attn_q_b && !l->indexer_attn_k);
    }
    if (il == 0) {
        CHECK_FIELD(ffn_gate, "ffn_gate.weight");
        CHECK_FIELD(ffn_up, "ffn_up.weight");
        CHECK_FIELD(ffn_down, "ffn_down.weight");
        assert(!l->ffn_gate_exps);
    } else {
        CHECK_FIELD(ffn_gate_inp, "ffn_gate_inp.weight");
        CHECK_FIELD(ffn_exp_probs_b, "exp_probs_b.bias");
        CHECK_FIELD(ffn_gate_exps, "ffn_gate_exps.weight");
        CHECK_FIELD(ffn_up_exps, "ffn_up_exps.weight");
        CHECK_FIELD(ffn_down_exps, "ffn_down_exps.weight");
        assert(!l->ffn_gate && !l->ffn_gate_shexp);
    }
#undef CHECK_FIELD
}

int main(void) {
    ds4_model model = {0};
    ds4_weights weights;
    ds4_host_shape host = {.variant = 13};
    g_host_shape = &host;
    model_apply_host_shape();
    assert(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_NAIVE);
    assert(DS4_N_LAYER == 48 && g_ds4_shape.n_nextn_predict == 0);
    assert(DS4_N_INDEXER_HEAD == 16 && DS4_N_INDEXER_TOP_K == 2048);
    assert(DS4_RMS_EPS == 1e-5f);
    model.tensors = xcalloc(613, sizeof(ds4_tensor));
    char line[128];
    while (fgets(line, sizeof(line), stdin)) {
        const size_t len = strcspn(line, "\n");
        line[len] = 0;
        assert(model.n_tensors < 613);
        model.tensors[model.n_tensors++].name = (ds4_str){strdup(line), len};
    }
    assert(model.n_tensors == 613);
    naive_bind(&weights, &model);
    check_tensor(&model, weights.token_embd, "token_embd.weight");
    check_tensor(&model, weights.output, "output.weight");
    check_tensor(&model, weights.output_norm, "output_norm.weight");
    for (unsigned il = 0; il < 48; il++) { check_layer(&model, &weights.layer[il], il); }
    for (unsigned i = 0; i < 613; i++) { assert(seen[i]); free((void *)model.tensors[i].name.ptr); }
    free(model.tensors);
    puts("613 native bindings; split GQA, SWA sinks, DSA indexer");
    return 0;
}
