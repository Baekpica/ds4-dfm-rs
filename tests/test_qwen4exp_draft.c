/* Model-free draft selection: preserve host argmax, including low-ID ties,
 * while returning only one GPU-selected ID at the Qwen vocabulary width. */
#include "../ds4.c"

static int check_pick(ds4_qwen_gpu_graph *g, float *values, float *scratch) {
    const uint64_t bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    int picked = -1;
    if (!ds4_gpu_tensor_write(g->logits, 0, values, bytes) ||
        !qwen_mtp_pick(g, scratch, DS4_N_VOCAB, &picked)) {
        return 1;
    }
    const int expected = sample_argmax(values, DS4_N_VOCAB);
    if (picked != expected) {
        fprintf(stderr, "draft ID mismatch: %d != %d\n", picked, expected);
        return 1;
    }
    return 0;
}

int main(void) {
    g_ds4_shape.n_vocab = 248320u;
    if (!ds4_gpu_init()) {
        return 1;
    }
    const uint64_t bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    float *values = malloc(bytes);
    float *scratch = malloc(bytes);
    ds4_qwen_gpu_graph graph = {0};
    graph.logits = ds4_gpu_tensor_alloc(bytes);
    graph.tokens = ds4_gpu_tensor_alloc(sizeof(int32_t));
    if (!values || !scratch || !graph.logits || !graph.tokens) {
        return 1;
    }
    int failed = 0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        values[i] = sinf((float)i * 0.07f) * 30.0f;
    }
    values[DS4_N_VOCAB - 1u] = 60.0f;
    failed += check_pick(&graph, values, scratch);
    values[1027] = 60.0f;
    failed += check_pick(&graph, values, scratch);
    values[3] = 60.0f;
    values[0] = NAN;
    failed += check_pick(&graph, values, scratch);
    values[DS4_N_VOCAB - 1u] = INFINITY;
    failed += check_pick(&graph, values, scratch);
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        values[i] = -INFINITY;
    }
    failed += check_pick(&graph, values, scratch);
    values[23] = DS4_NEG_INF;
    failed += check_pick(&graph, values, scratch);

    /* A compact head returns an index into the proposal list, including
     * high original token IDs; duplicate observations must not grow it. */
    graph.mtp_vocab_ids = malloc(DS4_N_VOCAB * sizeof(uint32_t));
    graph.mtp_vocab_seen = calloc(DS4_N_VOCAB, 1u);
    if (!graph.mtp_vocab_ids || !graph.mtp_vocab_seen) {
        return 1;
    }
    failed += !qwen_mtp_vocab_add(&graph, 0u);
    failed += !qwen_mtp_vocab_add(&graph, DS4_N_VOCAB - 1u);
    failed += !qwen_mtp_vocab_add(&graph, DS4_N_VOCAB - 1u);
    failed += qwen_mtp_vocab_add(&graph, DS4_N_VOCAB);
    failed += graph.mtp_vocab_count != 2u;
    values[0] = -2.0f;
    values[1] = 4.0f;
    int compact = -1;
    failed += !ds4_gpu_tensor_write(graph.logits, 0, values, 2u * sizeof(float));
    failed += !qwen_mtp_pick(&graph, scratch, 2u, &compact);
    if (compact != 1 || graph.mtp_vocab_ids[compact] != DS4_N_VOCAB - 1u) {
        failed++;
    }
    free(graph.mtp_vocab_seen);
    free(graph.mtp_vocab_ids);
    ds4_gpu_tensor_free(graph.tokens);
    ds4_gpu_tensor_free(graph.logits);
    free(scratch);
    free(values);
    ds4_gpu_cleanup();
    printf("Qwen draft selection: %s\n", failed ? "FAIL" : "PASS");
    return failed != 0;
}
