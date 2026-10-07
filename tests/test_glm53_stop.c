/* Compile the production stop helper without a model or GPU backend. */
#define DS4_NO_GPU
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM stop FAIL %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

enum {
    GLM_EOS = 154820,
    GLM_SYSTEM = 154826,
    GLM_USER = 154827,
    GLM_ASSISTANT = 154828,
    GLM_OBSERVATION = 154829,
    GLM_VOCAB = 154880
};

int main(void) {
    /* IDs match the handoff tokenizer and pinned text EOS list. Native
     * vocab_load resolves these special strings rather than assuming IDs. */
    ds4_vocab vocab = {
        .n_vocab = GLM_VOCAB, .eos_id = GLM_EOS,
        .user_id = GLM_USER, .observation_id = GLM_OBSERVATION,
        .assistant_id = GLM_ASSISTANT, .system_id = GLM_SYSTEM,
        .eot_id = -1, .end_of_turn_id = -1, .im_start_id = -1,
        .dots3_endoftext_id = -1
    };
    g_ds4_shape = DS4_SHAPE_GLM53_FLASH;
    CHECK(vocab_token_is_generation_stop(&vocab, GLM_EOS));
    CHECK(vocab_token_is_generation_stop(&vocab, GLM_USER));
    CHECK(vocab_token_is_generation_stop(&vocab, GLM_OBSERVATION));
    CHECK(!vocab_token_is_generation_stop(&vocab, GLM_SYSTEM));
    CHECK(!vocab_token_is_generation_stop(&vocab, GLM_ASSISTANT));
    CHECK(!vocab_token_is_generation_stop(&vocab, 42));
    CHECK(!vocab_token_is_generation_stop(&vocab, -1));
    CHECK(!vocab_token_is_generation_stop(NULL, GLM_USER));

    /* Missing optional markers must not create a stop. Changing loaded
     * IDs must follow the vocab; the backend cannot hardcode this artifact. */
    vocab.user_id = 7; vocab.observation_id = 9;
    CHECK(vocab_token_is_generation_stop(&vocab, 7));
    CHECK(vocab_token_is_generation_stop(&vocab, 9));
    CHECK(!vocab_token_is_generation_stop(&vocab, GLM_USER));
    CHECK(!vocab_token_is_generation_stop(&vocab, GLM_OBSERVATION));
    vocab.user_id = vocab.observation_id = -1;
    CHECK(!vocab_token_is_generation_stop(&vocab, GLM_USER));
    CHECK(!vocab_token_is_generation_stop(&vocab, GLM_OBSERVATION));

    vocab.user_id = GLM_USER; vocab.observation_id = GLM_OBSERVATION;
    const ds4_shape shapes[] = {
        DS4_SHAPE_FLASH, DS4_SHAPE_SOLAR_OPEN2_250B, DS4_SHAPE_MOTIF3,
        DS4_SHAPE_QWEN38_FLASH_NEXT, DS4_SHAPE_MIMO26_FLASH,
        DS4_SHAPE_LING30_FLASH_VL, DS4_SHAPE_DOTS3_NOTE_PREV,
        DS4_SHAPE_KEXAONE_236B
    };
    for (size_t i = 0u; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        g_ds4_shape = shapes[i];
        CHECK(vocab_token_is_generation_stop(&vocab, GLM_EOS));
        CHECK(!vocab_token_is_generation_stop(&vocab, GLM_OBSERVATION));
        CHECK(vocab_token_is_generation_stop(&vocab, GLM_USER) ==
              (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_MOTIF3));
    }
    puts("GLM stop: loaded EOS/user/observation, optional IDs and family scope passed");
    return 0;
}
