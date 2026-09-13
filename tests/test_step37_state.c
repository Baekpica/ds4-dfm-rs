/* Output readers must not expose stale or uninitialized Step logits. */
#include "../ds4.c"

int main(void) {
    g_ds4_shape = DS4_SHAPE_STEP37_FLASH;
    ds4_engine e = {.backend = DS4_BACKEND_CUDA};
    float *logits = xcalloc(DS4_N_VOCAB, sizeof(float));
    float *copy = xmalloc(DS4_N_VOCAB * sizeof(float));
    logits[17] = 4;
    ds4_session s = {.engine = &e, .logits = logits};
    ds4_token_score score;
    uint64_t rng = 1;
    if (ds4_session_argmax(&s) != -1 || ds4_session_argmax_excluding(&s, 17) != -1 ||
        ds4_session_sample(&s, 0, 0, 1, 0, &rng) != -1 ||
        ds4_session_copy_logits(&s, copy, DS4_N_VOCAB) != 0 ||
        ds4_session_top_logprobs(&s, &score, 1) != 0 ||
        ds4_session_token_logprob(&s, 17, &score) != 0) {
        ds4_die("Step readers exposed invalid checkpoint logits");
    }
    s.checkpoint_valid = true;
    if (ds4_session_argmax(&s) != 17 ||
        ds4_session_copy_logits(&s, copy, DS4_N_VOCAB) != (int)DS4_N_VOCAB ||
        memcmp(logits, copy, DS4_N_VOCAB * sizeof(float))) {
        ds4_die("Step valid logits are inaccessible");
    }
    free(logits); free(copy);
    puts("Step logits readers: invalid frontier refused, valid frontier preserved PASS");
    return 0;
}
