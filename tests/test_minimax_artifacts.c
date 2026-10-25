/* Model-free guard: only the target index-Q's unused replica is omitted. */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "MiniMax artifacts FAIL %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

static ds4_tensor index_q(const char *name) {
    return (ds4_tensor){
        .name = {name, strlen(name)}, .ndim = 2, .dim = {6144, 512},
        .type = DS4_TENSOR_Q8_0, .elements = 6144u * 512u,
        .bytes = 3342336u,
    };
}

int main(void) {
    enum { EXTRA = 4, COUNT = MINIMAX_LAYERS + EXTRA };
    ds4_tensor tensors[COUNT];
    char names[MINIMAX_LAYERS][96];
    uint64_t skipped_bytes = 0;
    g_ds4_shape = DS4_SHAPE_MINIMAX_VQ;

    for (uint32_t il = 0; il < MINIMAX_LAYERS; il++) {
        snprintf(names[il], sizeof(names[il]),
            "language_model.model.layers.%u.self_attn.index_q_proj.weight", il);
        tensors[il] = index_q(names[il]);
        CHECK(minimax_raw_index_q(&tensors[il]));
        skipped_bytes += tensors[il].bytes;
    }
    CHECK(skipped_bytes == UINT64_C(200540160));

    /* DSpark K/V share the target index-Q shape, but use aligned Q8. */
    tensors[MINIMAX_LAYERS] = index_q(
        "language_model.model.dspark.layers.0.decoder_layer.self_attn.k_proj.weight");
    tensors[MINIMAX_LAYERS + 1] = index_q(
        "language_model.model.dspark.layers.0.decoder_layer.self_attn.v_proj.weight");
    tensors[MINIMAX_LAYERS + 2] = index_q(
        "language_model.model.layers.60.self_attn.index_q_proj.weight");
    tensors[MINIMAX_LAYERS + 3] = index_q(
        "language_model.model.layers.00.self_attn.index_q_proj.weight");
    for (uint32_t i = MINIMAX_LAYERS; i < COUNT; i++) {
        CHECK(!minimax_raw_index_q(&tensors[i]));
    }

    ds4_tensor changed = tensors[0];
    changed.type = DS4_TENSOR_Q4_K;
    CHECK(!minimax_raw_index_q(&changed));
    changed = tensors[0]; changed.ndim = 3;
    CHECK(!minimax_raw_index_q(&changed));
    changed = tensors[0]; changed.dim[0] = 512; changed.dim[1] = 6144;
    CHECK(!minimax_raw_index_q(&changed));
    changed = tensors[0]; changed.bytes--;
    CHECK(!minimax_raw_index_q(&changed));
    changed = tensors[0]; changed.name.len--;
    CHECK(!minimax_raw_index_q(&changed));
    changed = index_q("language_model.model.layers.0.self_attn.index_q_proj.weight.extra");
    CHECK(!minimax_raw_index_q(&changed));
    changed = index_q("language_model.model.layers.0.self_attn.index_k_proj.weight");
    CHECK(!minimax_raw_index_q(&changed));

    ds4_model model = {.n_tensors = COUNT, .tensors = tensors};
    model_build_catalog(&model);
    for (uint32_t i = 0; i < COUNT; i++) {
        CHECK(model.tensor_traits[i] == DS4_TCAT_ARTIFACT_ADDITIVE);
    }
    g_ds4_shape = DS4_SHAPE_GLM53_FLASH;
    CHECK(!minimax_raw_index_q(&tensors[0]));
    minimax_catalog(&model);
    for (uint32_t i = 0; i < COUNT; i++) {
        CHECK(model.tensor_traits[i] == DS4_TCAT_ARTIFACT_ADDITIVE);
    }

    g_ds4_shape = DS4_SHAPE_MINIMAX_VQ;
    minimax_catalog(&model);
    for (uint32_t i = 0; i < COUNT; i++) {
        CHECK(model.tensor_traits[i] == (i < MINIMAX_LAYERS ? 0 : DS4_TCAT_ARTIFACT_ADDITIVE));
        CHECK(tensors[i].bytes == 3342336u);
    }
    CHECK(model.n_tensors == COUNT && model.tensors == tensors);
    free(model.tensor_traits);
    puts("MINIMAX_ARTIFACTS_PASS: 60 raw-only index-Q replicas, 200540160 bytes omitted; DSpark retained");
    return 0;
}
