/* Model-free native sampling guard. No GPU API is called. */
#include "../ds4.c"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "dots3 guard FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)

static void unavailable(ds4_session *s) {
    uint64_t rng = 1;
    ds4_token_score score;
    CHECK(ds4_session_argmax(s) == -1);
    CHECK(ds4_session_argmax_excluding(s, 3) == -1);
    CHECK(ds4_session_sample(s, 0, 0, 1, 0, &rng) == -1);
    CHECK(ds4_session_top_logprobs(s, &score, 1) == 0);
}

int main(void) {
    g_ds4_shape = DS4_SHAPE_DOTS3_NOTE_PREV;
    ds4_engine engine = {.backend = DS4_BACKEND_CUDA};
    ds4_dots3_spec spec = {0};
    ds4_session s = {.engine = &engine, .dots3_spec = &spec};
    s.logits = xcalloc(DS4_N_VOCAB, sizeof(float));
    s.logits[7] = 1;
    unavailable(&s);
    s.checkpoint_valid = true;
    s.checkpoint.len = 4;
    s.dots3_graph.cache_len = 4;
    CHECK(ds4_session_argmax(&s) == 7);
    spec.trial_n = 3;
    unavailable(&s);
    spec.trial_n = 0;
    s.dots3_graph.cache_len = 7;
    unavailable(&s);
    s.dots3_spec = NULL;
    unavailable(&s);
    s.dots3_graph.cache_len = 4;
    CHECK(ds4_session_argmax(&s) == 7);
    CHECK(dots3_spec_bytes(4096, 128) == UINT64_C(26068040));
    CHECK(dots3_spec_bytes(1024, 32) == UINT64_C(24101960));
    CHECK(dots3_spec_bytes(262144, 4096) == UINT64_C(107332680));
    free(s.logits);
    puts("dots3 model-free sampling guards and allocation quote PASS");
    return 0;
}
