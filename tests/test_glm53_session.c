#include "../ds4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const uint8_t png_1x1[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x04, 0x00, 0x00, 0x00, 0xb5, 0x1c, 0x0c, 0x02, 0x00, 0x00, 0x00,
    0x0b, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0x64, 0xf8, 0x0f, 0x00,
    0x01, 0x05, 0x01, 0x01, 0x27, 0x18, 0xe3, 0x66, 0x00, 0x00, 0x00, 0x00,
    0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        fprintf(stderr,
                "usage: %s <GLM-5.3-Flash-Q2.gguf> [vision.gguf]\n",
                argv[0]);
        return 2;
    }
    ds4_engine_options opt = {0};
    opt.model_path = argv[1];
    opt.vision_path = argc == 3 ? argv[2] : NULL;
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.defer_boot_prewarm = true;

    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens prompt = {0};
    ds4_tokens vision_prompt = {0};
    ds4_vision_embedding embedding = {0};
    float *logits = NULL;
    char err[256] = "";
    int failed = 1;

    if (ds4_engine_open(&engine, &opt) != 0 ||
        ds4_engine_layer_count(engine) != 45 ||
        ds4_engine_supports_batching(engine)) {
        fprintf(stderr, "GLM-5.3 engine contract failed\n");
        goto cleanup;
    }
    if (ds4_session_create(&session, engine, argc == 3 ? 32 : 8) != 0 ||
        !ds4_session_graph_pending(session) ||
        ds4_session_prefill_cap(session) != 1) {
        fprintf(stderr, "GLM-5.3 session creation failed\n");
        goto cleanup;
    }
    ds4_tokens_push(&prompt, 0);
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0 ||
        ds4_session_pos(session) != 1) {
        fprintf(stderr, "GLM-5.3 sync failed: %s\n", err);
        goto cleanup;
    }
    const int n_vocab = ds4_engine_vocab_size(engine);
    logits = malloc((size_t)n_vocab * sizeof(*logits));
    if (!logits || ds4_session_copy_logits(session, logits, n_vocab) != n_vocab)
        goto cleanup;
    for (int i = 0; i < n_vocab; i++) {
        if (!isfinite(logits[i])) {
            fprintf(stderr, "GLM-5.3 produced non-finite logits\n");
            goto cleanup;
        }
    }
    const int next = ds4_session_argmax(session);
    if (next < 0 || next >= n_vocab ||
        ds4_session_eval(session, next, err, sizeof(err)) != 0 ||
        ds4_session_pos(session) != 2) {
        fprintf(stderr, "GLM-5.3 decode failed: %s\n", err);
        goto cleanup;
    }
    if (argc == 3) {
        if (!ds4_engine_has_vision(engine) ||
            !ds4_engine_vision_encode_memory(
                engine, png_1x1, sizeof(png_1x1), &embedding,
                err, sizeof(err)) ||
            embedding.token_count != 16) {
            fprintf(stderr, "GLM-5.3 vision encode failed: %s\n", err);
            goto cleanup;
        }
        ds4_tokens_push(&vision_prompt, 154830);
        for (uint32_t i = 0; i < embedding.token_count; i++)
            ds4_tokens_push(&vision_prompt, 154854);
        ds4_tokens_push(&vision_prompt, 154831);
        ds4_vision_span span = {
            .token_start = 1,
            .embedding = embedding,
        };
        if (ds4_session_sync_multimodal(
                session, &vision_prompt, &span, 1, err, sizeof(err)) != 0 ||
            ds4_session_pos(session) != 18 ||
            ds4_session_copy_logits(session, logits, n_vocab) != n_vocab) {
            fprintf(stderr, "GLM-5.3 multimodal sync failed: %s\n", err);
            goto cleanup;
        }
        for (int i = 0; i < n_vocab; i++) {
            if (!isfinite(logits[i])) {
                fprintf(stderr, "GLM-5.3 multimodal logits are non-finite\n");
                goto cleanup;
            }
        }
        fprintf(stderr, "GLM-5.3 Q2 multimodal session: 16 image tokens and finite logits\n");
    }
    failed = 0;
    fprintf(stderr, "GLM-5.3 Q2 session: finite prefill and decode logits\n");

cleanup:
    free(logits);
    ds4_vision_embedding_free(&embedding);
    ds4_tokens_free(&vision_prompt);
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return failed;
}
