/* State-only guards must work before allocation and after failed restore. */
#include "../ds4.c"
#include <assert.h>

// Foreign memory estimators link these switches; Naive must ignore them.
int ds4_cuda_fp8_kv_enabled(void) { return 0; }
int ds4_cuda_fp4_index_enabled(void) { return 0; }

int main(void) {
    g_ds4_shape = DS4_SHAPE_NAIVE_N05_FLASH;
    ds4_engine e = {0};
    ds4_session s = {0};
    e.backend = DS4_BACKEND_CUDA;
    s.engine = &e;
    s.logits = calloc(N05_VOCAB, sizeof(float));
    assert(s.logits);
    s.logits[17] = 1;
    assert(ds4_session_argmax(&s) == -1);
    uint64_t rng = 1;
    ds4_token_score top;
    assert(ds4_session_sample(&s, 0, 1, 1, 0, &rng) == -1);
    assert(ds4_session_argmax_excluding(&s, 1) == -1);
    assert(ds4_session_sample_excluding(&s, 0, 1, 1, 0, &rng, 1) == -1);
    assert(!ds4_session_top_logprobs(&s, &top, 1));
    s.checkpoint_valid = true; s.checkpoint.len = 3;
    s.naive_graph.position = 3; s.naive_graph_ready = true;
    assert(ds4_session_argmax(&s) == 17);
    s.naive_graph.failed = true;
    assert(ds4_session_argmax(&s) == -1);
    s.naive_graph.failed = false; s.naive_graph.position--;
    assert(ds4_session_argmax(&s) == -1);
    free(s.logits);

    const unsigned ctx = 262144;
    assert(ds4_engine_session_graph_bytes_estimate(&e, ctx) ==
           naive_context_memory(ctx, naive_prefill_cap(ctx)).total_bytes);
    e.backend = DS4_BACKEND_METAL;
    assert(!ds4_engine_session_graph_bytes_estimate(&e, ctx));
    puts("Naive state guards and serial memory quote pass without model allocation");
    return 0;
}
