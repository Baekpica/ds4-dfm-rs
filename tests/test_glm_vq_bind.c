/* Read only manifest metadata. No weight mapping, CUDA or model allocation. */
#include "../ds4.c"
#include <sys/wait.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLMVQ bind FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

enum bad_contract { BAD_SCALE, BAD_BOOK, BAD_MTP };

static void reject(ds4_model *m, enum bad_contract bad) {
    const pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
        (void)freopen("/dev/null", "w", stderr);
        if (bad == BAD_SCALE) {
            glmvq_tensor(m, 0, "mlp.gate_proj.weight_scale_inv")->type = DS4_TENSOR_BF16;
            (void)glmvq_linear(m, 0, "mlp.gate_proj.weight");
        } else {
            ds4_glmvq_experts expert = {0};
            if (bad == BAD_BOOK) {
                required_tensor(m, "glm_vq.layer.3.codebook.0")->dim[1] = 4096;
                glmvq_bind_experts(&expert, m, 3);
            } else {
                glmvq_tensor(m, GLMVQ_TRUNK, "mlp.experts.0.gate_proj.weight")->type = DS4_TENSOR_GLM_VQ;
                glmvq_bind_experts(&expert, m, GLMVQ_TRUNK);
            }
        }
        _exit(0);
    }
    int status = 0;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) != 0);
}

int main(void) {
    ds4_model model = {0};
    unsigned count = 0;
    CHECK(scanf("%u", &count) == 1 && count == 39862);
    model.n_tensors = count;
    model.tensors = xcalloc(count, sizeof(*model.tensors));
    for (unsigned i = 0; i < count; i++) {
        char name[256];
        ds4_tensor *t = &model.tensors[i];
        CHECK(scanf("%255s %u %u %" SCNu64, name, &t->type, &t->ndim, &t->bytes) == 4);
        CHECK(t->ndim > 0 && t->ndim <= DS4_MAX_DIMS);
        t->name = (ds4_str){strdup(name), strlen(name)};
        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            CHECK(scanf("%" SCNu64, &t->dim[d]) == 1);
            t->elements *= t->dim[d];
        }
    }
    const ds4_host_shape shape = {.variant = DS4_VARIANT_GLM53_FLASH_VQ};
    ds4_host_shape_install(&shape);
    model_apply_host_shape();
    ds4_host_shape_clear();
    CHECK(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM53_VQ && DS4_N_LAYER == GLMVQ_LAYERS);
    CHECK(strcmp(g_ds4_shape.name, "GLM-5.3-Flash-VQ") == 0);

    ds4_weights *weights = xcalloc(1, sizeof(*weights));
    glmvq_bind(weights, &model);
    unsigned vq = 0, fp8 = 0, routed = 0, kda = 0, dsa = 0;
    for (unsigned il = 0; il < GLMVQ_LAYERS; il++) {
        const ds4_layer_weights *l = &weights->layer[il];
        if (ds4_glm53_layer_is_kda(il)) {
            CHECK(l->kda_q && !l->attn_kv_b);
            kda++;
        } else {
            CHECK(l->attn_kv_b && !l->attn_k_b && !l->attn_v_b);
            dsa++;
        }
        if (il < DS4_N_LEADING_DENSE) { continue; }
        const ds4_glmvq_experts *e = &weights->glmvq[il];
        CHECK((e->book != NULL) == (il < GLMVQ_TRUNK));
        for (unsigned expert = 0; expert < GLMVQ_EXPERTS; expert++) {
            CHECK(e->gate[expert] && e->up[expert] && e->down[expert]);
            CHECK(e->gate[expert] != e->up[expert] && e->up[expert] != e->down[expert]);
        }
    }
    CHECK(kda == 34 && dsa == 12);
    CHECK(weights->layer[GLMVQ_TRUNK].nextn_eh_proj && !weights->layer[GLMVQ_TRUNK].hc_attn_fn);
    for (unsigned i = 0; i < count; i++) {
        const ds4_tensor *t = &model.tensors[i];
        if (t->type == DS4_TENSOR_GLM_VQ) {
            CHECK(!t->source_scale);
            vq++;
        }
        if (t->type == DS4_TENSOR_GLM_FP8) {
            CHECK(t->source_scale && t->source_scale->type == DS4_TENSOR_F32);
            fp8++;
        }
        const uint32_t traits = ds4_tensor_catalog_classify(t->name.ptr, t->name.len,
            t->ndim, t->type, t->dim, t->bytes);
        CHECK(!(traits & DS4_TCAT_VQ_BLOB));
        if (traits & DS4_TCAT_ROUTED_EXPERT) { routed++; }
    }
    printf("GLMVQ binding counts: %u VQ, %u FP8, %u routed\n", vq, fp8, routed);
    fflush(stdout);
    CHECK(vq == 36288 && fp8 == 1050 && routed == 37152);
    reject(&model, BAD_SCALE);
    reject(&model, BAD_BOOK);
    reject(&model, BAD_MTP);
    weights_free(weights);
    CHECK(!weights->glmvq);
    free(weights);
    for (unsigned i = 0; i < count; i++) { free((void *)model.tensors[i].name.ptr); }
    free(model.tensors);
    puts("GLMVQ_BIND_PASS: 34 KDA, 12 DSA including MTP, 36288 VQ, 1050 FP8, 37152 routed");
    return 0;
}
