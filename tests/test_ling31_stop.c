/* Compile the production stop helper without a model or GPU backend. */
#define DS4_NO_GPU
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "Ling-3.1 stop FAIL %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

/* Source tokenizer IDs. The host vocab resolves them by string and hands
 * them to the native vocab, so the backend must follow the loaded IDs. */
enum {
    L31_EOS = 156895,         /* <|role_end|> */
    L31_END_OF_TEXT = 156892, /* <|endoftext|>, the second official stop */
    L31_THINK = 156903,
    L31_ROLE = 157151,        /* <role> opens the next turn */
    L31_ROLE_CLOSE = 157152
};

int main(void) {
    ds4_vocab vocab = {
        .n_vocab = 157184, .eos_id = L31_EOS, .eot_id = L31_END_OF_TEXT,
        .im_start_id = L31_ROLE, .user_id = -1, .observation_id = -1,
        .end_of_turn_id = -1, .dots3_endoftext_id = -1
    };
    g_ds4_shape = DS4_SHAPE_LING31_VQ;
    /* Speculative acceptance and bank decoding stop on this predicate: a
     * missed stop commits drafts past it and leaks the next role. */
    CHECK(vocab_token_is_generation_stop(&vocab, L31_EOS));
    CHECK(vocab_token_is_generation_stop(&vocab, L31_END_OF_TEXT));
    CHECK(vocab_token_is_generation_stop(&vocab, L31_ROLE));
    CHECK(!vocab_token_is_generation_stop(&vocab, L31_ROLE_CLOSE));
    CHECK(!vocab_token_is_generation_stop(&vocab, L31_THINK));
    CHECK(!vocab_token_is_generation_stop(&vocab, 42));

    vocab.eot_id = vocab.im_start_id = -1;
    CHECK(!vocab_token_is_generation_stop(&vocab, L31_END_OF_TEXT));
    CHECK(!vocab_token_is_generation_stop(&vocab, L31_ROLE));
    puts("Ling-3.1 stop: EOS, <|endoftext|> and <role> stop; optional IDs follow the vocab");
    return 0;
}
