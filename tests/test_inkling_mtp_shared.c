/* Actual shared MQ85GB embedding/head with all eight imported BF16 depths. */
#include "../ds4.c"

enum { ROWS = 7 };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static int read_top(ds4_inkling_graph *g, float *values) {
    CHECK(ds4_gpu_tensor_read(g->logits, 0, values, DS4_N_VOCAB * sizeof(float)));
    int top = 0;
    for (unsigned i = 0; i < INKLING_VALID_VOCAB; i++) {
        CHECK(isfinite(values[i]));
        if (values[i] > values[top]) { top = (int)i; }
    }
    for (unsigned i = INKLING_VALID_VOCAB; i < DS4_N_VOCAB; i++) {
        CHECK(values[i] == -INFINITY);
    }
    return top;
}

int main(int argc, char **argv) {
    CHECK(argc == 4);
    const ds4_host_shape host = {.variant = DS4_VARIANT_INKLING_SMALL};
    ds4_host_shape_install(&host); model_apply_host_shape(); ds4_host_shape_clear();
    ds4_model main_model, mtp_model;
    model_open(&main_model, argv[1], false, false);
    model_open(&mtp_model, argv[2], false, false);
    ds4_weights main_weights;
    ds4_inkling_draft mtp_weights;
    weights_bind(&main_weights, &main_model, false, 0, UINT32_MAX, true, false);
    inkling_bind_draft(&mtp_weights, &mtp_model);
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(main_model.map, main_model.size));
    /* Import publishes the sidecar ranges; an auxiliary copy would overlap. */
    CHECK(ds4_gpu_import_model_ipc_manifest(mtp_model.map, mtp_model.size, argv[3], "mtp"));
    ds4_inkling_mtp_graph actual, reference;
    CHECK(inkling_draft_alloc(&actual, &mtp_model, &mtp_weights, 32, ROWS));
    CHECK(inkling_draft_alloc(&reference, &mtp_model, &mtp_weights, 32, ROWS));
    const size_t count = ROWS * IK_HIDDEN, bytes = count * sizeof(float);
    float *input = xmalloc(bytes), *raw = xmalloc(bytes), *got = xmalloc(bytes);
    for (size_t i = 0; i < count; i++) { input[i] = ((int)(i * 37 % 1021) - 510) / 256.0f; }
    ds4_gpu_tensor *hidden = ds4_gpu_tensor_alloc(bytes), *embed = ds4_gpu_tensor_alloc(bytes);
    CHECK(hidden && embed && ds4_gpu_tensor_write(hidden, 0, input, bytes));
    const size_t logit_bytes = DS4_N_VOCAB * sizeof(float);
    float *logits = xmalloc(logit_bytes), *want = xmalloc(logit_bytes);
    int tokens[ROWS] = {200000, 976, 9029, 328, 10128, 382, 12650};
    for (unsigned depth = 0; depth < INKLING_DRAFT_LAYERS; depth++) {
        const ds4_gpu_tensor *h = depth ? actual.graph.buf[IK_X] : hidden;
        CHECK(inkling_draft_tokens(&actual, &main_model, &main_weights,
              &mtp_model, &mtp_weights, depth, h, tokens, ROWS));
        const int top = read_top(&actual.graph, logits);
        CHECK(ds4_gpu_tensor_read(actual.graph.buf[IK_X], 0, raw, bytes));
        /* Independently assemble the existing primitives to catch ordering,
         * wrong source-map and buffer-aliasing errors in the token wrapper. */
        CHECK(ds4_gpu_tensor_write(reference.graph.buf[IK_TOKENS], 0, tokens, sizeof(tokens)));
        CHECK(ds4_gpu_embed_tokens_q8_0_tensor(embed, reference.graph.buf[IK_TOKENS],
              main_model.map, main_model.size, main_weights.token_embd->abs_offset,
              DS4_N_VOCAB, ROWS, IK_HIDDEN));
        CHECK(ds4_gpu_inkling_norm(embed, embed, main_model.map, main_model.size,
              main_weights.inkling.embed_norm->abs_offset, IK_HIDDEN, ROWS));
        CHECK(inkling_draft_forward(&reference, &mtp_model, &mtp_weights, depth, hidden, embed, ROWS));
        CHECK(ds4_gpu_tensor_read(reference.graph.buf[IK_X], 0, got, bytes));
        CHECK(memcmp(raw, got, bytes) == 0);
        ds4_gpu_tensor *last = ds4_gpu_tensor_view(reference.graph.buf[IK_X],
                                                 (ROWS - 1) * IK_HIDDEN * sizeof(float), IK_HIDDEN * sizeof(float));
        CHECK(last && ds4_gpu_inkling_add_scale(reference.graph.buf[IK_NORM], last, NULL,
                                               1.0f / IK_LOGIT_DIVISOR, IK_HIDDEN));
        CHECK(plain_graph_matmul_tensor(reference.graph.logits, &main_model, main_weights.output,
                                        IK_HIDDEN, DS4_N_VOCAB, reference.graph.buf[IK_NORM], 1));
        CHECK(ds4_gpu_tensor_read(reference.graph.logits, 0, want, logit_bytes));
        CHECK(memcmp(logits, want, INKLING_VALID_VOCAB * sizeof(float)) == 0);
        ds4_gpu_tensor_free(last);
        CHECK(ds4_gpu_tensor_write(hidden, 0, got, bytes));
        memmove(tokens, tokens + 1, (ROWS - 1) * sizeof(int)); tokens[ROWS - 1] = top;
        printf("shared draft %u: top=%d, all hidden/logits exact\n", depth, top);
    }
    const unsigned position = actual.positions[0];
    tokens[ROWS - 1] = INKLING_VALID_VOCAB;
    CHECK(!inkling_draft_tokens(&actual, &main_model, &main_weights,
          &mtp_model, &mtp_weights, 0, hidden, tokens, ROWS));
    CHECK(actual.positions[0] == position && !actual.graph.failed);
    CHECK(ds4_gpu_tensor_read(actual.graph.buf[IK_X], 0, got, bytes));
    CHECK(memcmp(raw, got, bytes) == 0);
    CHECK(ds4_gpu_tensor_read(actual.graph.logits, 0, want, logit_bytes));
    CHECK(memcmp(logits, want, logit_bytes) == 0);
    inkling_draft_free(&actual); inkling_draft_free(&reference);
    ds4_gpu_tensor_free(hidden); ds4_gpu_tensor_free(embed);
    ds4_gpu_cleanup(); model_close(&main_model); model_close(&mtp_model);
    free(input); free(raw); free(got); free(logits); free(want);
    puts("Inkling all8 shared embedding/head gate passed");
    return 0;
}
