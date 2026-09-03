#include "../ds4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <GLM-5.3-Flash-Q2.gguf>\n", argv[0]);
        return 2;
    }
    ds4_engine_options opt = {0};
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.defer_boot_prewarm = true;

    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens prompt = {0};
    float *logits = NULL;
    char err[256] = "";
    int failed = 1;

    if (ds4_engine_open(&engine, &opt) != 0 ||
        ds4_engine_layer_count(engine) != 45 ||
        ds4_engine_supports_batching(engine)) {
        fprintf(stderr, "GLM-5.3 engine contract failed\n");
        goto cleanup;
    }
    if (ds4_session_create(&session, engine, 8) != 0 ||
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
    failed = 0;
    fprintf(stderr, "GLM-5.3 Q2 session: finite prefill and decode logits\n");

cleanup:
    free(logits);
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return failed;
}
