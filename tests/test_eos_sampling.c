#include "../ds4.c"

#define CHECK(x) do { \
    if (!(x)) { \
        fprintf(stderr, "EOS sampler FAIL %d: %s\n", __LINE__, #x); \
        exit(1); \
    } \
} while (0)

int main(void) {
    const float logits[] = {2.0f, 9.0f, 5.0f, 1.0f};
    uint64_t rng = 7;

    CHECK(ds4_sample_logits(logits, 4, 0.0f, 0, 1.0f, 0.0f, &rng) == 1);
    CHECK(sample_top_p_min_p_excluding(logits, 4, 0.0f, 0, 1.0f, 0.0f,
                                       &rng, 1, -1) == 2);
    CHECK(sample_top_p_min_p_excluding(logits, 4, 1.0f, 1, 1.0f, 0.0f,
                                       &rng, 1, -1) == 2);

    for (int i = 0; i < 64; i++) {
        CHECK(sample_top_p_min_p_excluding(logits, 4, 1.0f, 0, 1.0f, 0.0f,
                                           &rng, 1, -1) != 1);
        CHECK(sample_top_p_min_p_excluding(logits, 4, 1.0f, 4, 0.9f, 0.0f,
                                           &rng, 1, -1) != 1);
    }

    rng = 7;
    CHECK(sample_top_p_min_p_excluding(logits, 4, 0.0f, 0, 1.0f, 0.0f,
                                       &rng, -1, -1) == 1);
    for (int top_k = 0; top_k <= 4; top_k += 4) {
        uint64_t baseline_rng = 7;
        uint64_t policy_rng = 7;
        for (int i = 0; i < 64; i++) {
            CHECK(ds4_sample_logits(logits, 4, 0.8f, top_k, 0.9f, 0.0f,
                                    &baseline_rng) ==
                  sample_top_p_min_p_excluding(logits, 4, 0.8f, top_k,
                                                0.9f, 0.0f, &policy_rng, -1, -1));
            CHECK(baseline_rng == policy_rng);
        }
    }
    CHECK(sample_top_p_min_p_override_excluding(logits, 4, 0.0f, 0,
          1.0f, 0.0f, &rng, DS4_SAMPLE_OVERRIDE_TOKEN(1), 1, NULL) == -1);
    CHECK(sample_top_p_min_p_override_excluding(logits, 4, 0.0f, 0,
          1.0f, 0.0f, &rng, DS4_SAMPLE_OVERRIDE_TOKEN(2), 1, NULL) == 2);

    g_ds4_shape = DS4_SHAPE_QWEN38_FLASH_NEXT;
    const ds4_vocab vocab = {.eos_id = 1, .eot_id = 2};
    CHECK(sample_eot_exclusion(&vocab, 1) == 2);
    CHECK(sample_top_p_min_p_override_excluding(logits, 4, 0.0f, 0,
          1.0f, 0.0f, &rng, DS4_SAMPLE_OVERRIDE_NONE, 1, &vocab) == 0);
    CHECK(sample_top_p_min_p_override_excluding(logits, 4, 1.0f, 1,
          1.0f, 0.0f, &rng, DS4_SAMPLE_OVERRIDE_NONE, 1, &vocab) == 0);
    CHECK(sample_top_p_min_p_override_excluding(logits, 4, 0.0f, 0,
          1.0f, 0.0f, &rng, DS4_SAMPLE_OVERRIDE_TOKEN(2), 1, &vocab) == -1);
    CHECK(sample_top_p_min_p_override_excluding(logits, 4, 0.0f, 0,
          1.0f, 0.0f, &rng, DS4_SAMPLE_OVERRIDE_NONE, -1, &vocab) == 1);

    g_ds4_shape = DS4_SHAPE_MIMO26_FLASH;
    const ds4_vocab mimo = {.eos_id = 1, .eot_id = 2, .end_of_turn_id = 3};
    CHECK(vocab_token_is_generation_stop(&mimo, mimo.eot_id));
    CHECK(vocab_token_is_generation_stop(&mimo, mimo.end_of_turn_id));
    CHECK(sample_eot_exclusion(&mimo, mimo.eos_id) == mimo.eot_id);
    CHECK(sample_top_p_min_p_override_excluding(logits, 4, 0.0f, 0,
          1.0f, 0.0f, &rng, DS4_SAMPLE_OVERRIDE_NONE, 1, &mimo) == 0);
    CHECK(sample_top_p_min_p_override_excluding(logits, 4, 0.0f, 0,
          1.0f, 0.0f, &rng, DS4_SAMPLE_OVERRIDE_TOKEN(2), 1, &mimo) == -1);
    CHECK(sample_top_p_min_p_override_excluding(logits, 4, 0.0f, 0,
          1.0f, 0.0f, &rng, DS4_SAMPLE_OVERRIDE_TOKEN(3), 1, &mimo) == 3);

    const float only_eos[] = {-INFINITY, 9.0f, -INFINITY};
    CHECK(sample_top_p_min_p_excluding(only_eos, 3, 0.0f, 0, 1.0f, 0.0f,
                                       &rng, 1, -1) == -1);
    puts("EOS sampler PASS");
    return 0;
}
