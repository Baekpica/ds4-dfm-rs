/* Public K-EXAONE persistent-bank regression.
 *
 *   ./tests/test_exaone_batch <first-model-shard.gguf>
 *
 * It compares two-row persistent generation with the scalar session path,
 * then proves exact-frontier and partial fork reuse on the common continuous contract.
 */
#include "../ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int serial_greedy(ds4_engine *engine, const ds4_tokens *prompt,
                         int ctx_size, int budget, int *out, int *out_n,
                         char *err, size_t errlen) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0 ||
        ds4_session_sync(session, prompt, err, errlen) != 0) {
        ds4_session_free(session);
        return 1;
    }
    const int eos = ds4_token_eos(engine);
    int n = 0;
    while (n < budget) {
        const int token = ds4_session_argmax(session);
        out[n++] = token;
        if (token == eos || n == budget) break;
        if (ds4_session_eval(session, token, err, errlen) != 0) {
            ds4_session_free(session);
            return 1;
        }
    }
    *out_n = n;
    ds4_session_free(session);
    return 0;
}

static int serial_snapshot_roundtrip(ds4_engine *engine,
                                     const ds4_tokens *prompt,
                                     const int oracle[2],
                                     char *err, size_t errlen) {
    ds4_session *session = NULL;
    ds4_session_snapshot snapshot = {0};
    ds4_session_snapshot corrupt = {0};
    int rc = 1;
    if (ds4_session_create(&session, engine, 256) != 0 ||
        ds4_session_sync(session, prompt, err, errlen) != 0 ||
        ds4_session_save_snapshot(session, &snapshot, err, errlen) != 0 ||
        snapshot.len != ds4_session_payload_bytes(session)) {
        goto done;
    }
    corrupt.ptr = malloc((size_t)snapshot.len);
    if (!corrupt.ptr) goto done;
    memcpy(corrupt.ptr, snapshot.ptr, (size_t)snapshot.len);
    corrupt.len = corrupt.cap = snapshot.len;
    corrupt.ptr[5u * sizeof(uint32_t)] ^= UINT8_C(1);
    if (ds4_session_load_snapshot(session, &corrupt, err, errlen) == 0 ||
        ds4_session_pos(session) != 0) {
        goto done;
    }
    if (ds4_session_load_snapshot(session, &snapshot, err, errlen) != 0 ||
        ds4_session_argmax(session) != oracle[0] ||
        ds4_session_eval(session, oracle[0], err, errlen) != 0 ||
        ds4_session_argmax(session) != oracle[1]) {
        goto done;
    }
    rc = 0;
done:
    ds4_session_snapshot_free(&corrupt);
    ds4_session_snapshot_free(&snapshot);
    ds4_session_free(session);
    return rc;
}

typedef struct {
    const ds4_tokens *prompt;
    int cached;
    int next;
    int source_bank;
    int target_bank;
    int admitted_cached;
    int admitted_computed;
    int admitted_bank;
    int token;
    int n;
    int failed;
} fork_case;

static int fork_admitted(void *ud, void *user, int cached,
                         int computed, int bank) {
    (void)user;
    fork_case *test = ud;
    test->admitted_cached = cached;
    test->admitted_computed = computed;
    test->admitted_bank = bank;
    return 1;
}

static int fork_admit(void *ud, ds4_cont_request *req) {
    fork_case *test = ud;
    if (test->next++) return 0;
    memset(req, 0, sizeof(*req));
    req->tokens = test->prompt->v;
    req->n = test->prompt->len;
    req->max_new = 1;
    req->eos = -1;
    req->n_cached = test->cached >= 0 ? test->cached : test->prompt->len;
    req->fork_bank = test->source_bank + 1;
    req->place_bank = test->target_bank + 1;
    req->on_admitted = fork_admitted;
    req->user = test;
    return 1;
}

static void fork_done(void *ud, void *user, const int *tokens,
                      int n, int finish) {
    (void)user;
    (void)finish;
    fork_case *test = ud;
    if (!tokens || n != 1) {
        test->failed = 1;
        return;
    }
    test->token = tokens[0];
    test->n = n;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <first-model-shard.gguf>\n", argv[0]);
        return 2;
    }
    if (setenv("DS4_SESSION_LAZY_GRAPH", "1", 1) != 0 ||
        setenv("DS4_NO_BOOT_PREWARM", "1", 1) != 0) {
        perror("setenv");
        return 1;
    }

    ds4_engine_options opt = {0};
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.defer_boot_prewarm = true;
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "EXAONE engine open failed\n");
        return 1;
    }

    int failed = 0;
    char err[256] = {0};
    ds4_tokens prompt[2] = {0};
    ds4_tokens restored_prompt = {0};
    ds4_session_payload_file bank_payload = {0};
    ds4_batch_ctx *ctx = NULL;
    ds4_tokenize_text(engine, "The capital of France is", &prompt[0]);
    ds4_tokenize_text(engine, "Two plus two equals", &prompt[1]);
    if (prompt[0].len <= 0 || prompt[1].len <= 0 ||
        !ds4_engine_supports_batching(engine)) {
        fprintf(stderr, "EXAONE tokenizer or batching capability failed\n");
        failed = 1;
        goto cleanup;
    }

    int oracle[2][3] = {{0}};
    int oracle_n[2] = {0};
    for (int i = 0; i < 2; i++) {
        if (serial_greedy(engine, &prompt[i], 256, 3,
                          oracle[i], &oracle_n[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "EXAONE scalar oracle %d failed: %s\n", i, err);
            failed = 1;
            goto cleanup;
        }
    }
    if (serial_snapshot_roundtrip(
            engine, &prompt[0], oracle[0], err, sizeof(err)) != 0) {
        fprintf(stderr, "EXAONE serial snapshot round-trip failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    if (ds4_batch_ctx_create_fit(
            engine, 256, 2, 16, &ctx, err, sizeof(err)) != 0 ||
        !ctx || ds4_batch_ctx_max_seq(ctx) != 2 ||
        !ds4_batch_ctx_supports_partial_reuse(ctx)) {
        fprintf(stderr, "EXAONE batch context failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    const int budget[2] = {2, 2};
    const int eos[2] = {ds4_token_eos(engine), ds4_token_eos(engine)};
    ds4_batch_gen_result got[2] = {0};
    if (ds4_engine_batched_generate_ctx(
            ctx, prompt, 2, budget, eos, got, err, sizeof(err)) != 0) {
        fprintf(stderr, "EXAONE persistent batch failed: %s\n", err);
        failed = 1;
    }
    for (int i = 0; !failed && i < 2; i++) {
        if (got[i].n_tokens != 2 ||
            memcmp(got[i].tokens, oracle[i],
                   2u * sizeof(int)) != 0) {
            fprintf(stderr,
                    "EXAONE row %d differs from scalar: got=%d,%d want=%d,%d\n",
                    i,
                    got[i].n_tokens > 0 ? got[i].tokens[0] : -1,
                    got[i].n_tokens > 1 ? got[i].tokens[1] : -1,
                    oracle_n[i] > 0 ? oracle[i][0] : -1,
                    oracle_n[i] > 1 ? oracle[i][1] : -1);
            failed = 1;
        }
    }
    free(got[0].tokens);
    free(got[1].tokens);

    ds4_tokens fork_prompt = {0};
    if (!failed && oracle_n[0] == 3) {
        fork_prompt.len = prompt[0].len + 1;
        fork_prompt.cap = fork_prompt.len;
        fork_prompt.v = malloc((size_t)fork_prompt.len * sizeof(*fork_prompt.v));
        if (!fork_prompt.v) {
            fprintf(stderr, "EXAONE fork prompt allocation failed\n");
            failed = 1;
        } else {
            memcpy(fork_prompt.v, prompt[0].v,
                   (size_t)prompt[0].len * sizeof(*fork_prompt.v));
            fork_prompt.v[prompt[0].len] = oracle[0][0];
        }
    } else if (!failed) {
        fprintf(stderr, "EXAONE scalar oracle did not reach fork frontier\n");
        failed = 1;
    }

    /* A two-token generation commits prompt + token[0] to KV; token[1] is
     * sampled from that exact frontier but is not itself forwarded yet. */
    fork_case fork = {
        .prompt = &fork_prompt,
        .cached = -1,
        .source_bank = 0,
        .target_bank = 1,
        .admitted_cached = -1,
        .admitted_computed = -1,
        .admitted_bank = -1,
    };
    if (!failed &&
        (ds4_engine_continuous_generate(
             ctx, fork_admit, NULL, fork_done, &fork,
             err, sizeof(err)) != 0 ||
         fork.failed || fork.n != 1 || fork.token != oracle[0][1] ||
         fork.admitted_cached != fork_prompt.len ||
         fork.admitted_computed != 0 || fork.admitted_bank != 1)) {
        fprintf(stderr,
                "EXAONE exact fork failed: %s token=%d want=%d split=%d+%d bank=%d\n",
                err, fork.token, oracle[0][1], fork.admitted_cached,
                fork.admitted_computed, fork.admitted_bank);
        failed = 1;
    }
    const int *committed = NULL;
    const int committed_n = ds4_batch_ctx_bank_committed(ctx, 1, &committed);
    if (!failed &&
        (committed_n != fork_prompt.len ||
         memcmp(committed, fork_prompt.v,
                (size_t)fork_prompt.len * sizeof(*committed)) != 0 ||
         ds4_cont_bank_payload_bytes(ctx, 1u) == 0u ||
         ds4_cont_bank_stage_payload(
             ctx, 1u, &bank_payload, err, sizeof(err)) != 0)) {
        fprintf(stderr, "EXAONE durable bank stage failed: %s\n", err);
        failed = 1;
    }
    FILE *bank_fp = NULL;
    if (!failed) bank_fp = fopen(bank_payload.path, "rb");
    if (!failed &&
        (!bank_fp || ds4_cont_bank_restore_payload(
             ctx, 0u, bank_fp, bank_payload.bytes,
             err, sizeof(err)) != 0)) {
        fprintf(stderr, "EXAONE durable bank restore failed: %s\n", err);
        failed = 1;
    }
    if (bank_fp && fclose(bank_fp) != 0 && !failed) {
        perror("EXAONE durable bank close");
        failed = 1;
    }
    if (!failed) {
        restored_prompt.len = committed_n + 1;
        restored_prompt.cap = restored_prompt.len;
        restored_prompt.v = malloc(
            (size_t)restored_prompt.len * sizeof(*restored_prompt.v));
        if (!restored_prompt.v) {
            fprintf(stderr, "EXAONE restored prompt allocation failed\n");
            failed = 1;
        } else {
            memcpy(restored_prompt.v, committed,
                   (size_t)committed_n * sizeof(*restored_prompt.v));
            restored_prompt.v[committed_n] = oracle[0][1];
        }
    }
    fork_case restored = {
        .prompt = &restored_prompt,
        .cached = committed_n,
        .source_bank = -1,
        .target_bank = 0,
        .admitted_cached = -1,
        .admitted_computed = -1,
        .admitted_bank = -1,
    };
    if (!failed &&
        (ds4_engine_continuous_generate(
             ctx, fork_admit, NULL, fork_done, &restored,
             err, sizeof(err)) != 0 ||
         restored.failed || restored.n != 1 ||
         restored.token != oracle[0][2] ||
         restored.admitted_cached != committed_n ||
         restored.admitted_computed != 1 ||
         restored.admitted_bank != 0)) {
        fprintf(stderr,
                "EXAONE restored bank continuation failed: %s "
                "token=%d want=%d split=%d+%d bank=%d\n",
                err, restored.token, oracle[0][2],
                restored.admitted_cached, restored.admitted_computed,
                restored.admitted_bank);
        failed = 1;
    }
    /* Bank 1 inherited the prompt checkpoint during the exact fork. Restore
     * below its current frontier, then truncate the source in place. */
    for (int target = 0; !failed && target < 2; target++) {
        fork_case partial = {
            .prompt = &prompt[0], .cached = prompt[0].len,
            .source_bank = 1, .target_bank = target,
            .admitted_cached = -1, .admitted_computed = -1,
            .admitted_bank = -1,
        };
        if (ds4_engine_continuous_generate(ctx, fork_admit, NULL, fork_done,
                &partial, err, sizeof(err)) != 0 || partial.failed ||
            partial.n != 1 || partial.token != oracle[0][0] ||
            partial.admitted_cached != prompt[0].len ||
            partial.admitted_computed != 0 || partial.admitted_bank != target) {
            fprintf(stderr, "EXAONE partial fork/truncate failed: %s split=%d+%d bank=%d\n",
                    err, partial.admitted_cached, partial.admitted_computed,
                    partial.admitted_bank);
            failed = 1;
        }
        const int *source = NULL;
        int source_n = ds4_batch_ctx_bank_committed(ctx, 1, &source);
        int expected_n = target == 0 ? fork_prompt.len : prompt[0].len;
        if (source_n != expected_n || memcmp(source, fork_prompt.v, (size_t)expected_n * sizeof(int))) {
            fprintf(stderr, "EXAONE partial source history mismatch\n");
            failed = 1;
        }
    }
    ds4_tokens_free(&fork_prompt);

cleanup:
    ds4_session_payload_file_free(&bank_payload);
    ds4_tokens_free(&restored_prompt);
    ds4_batch_ctx_destroy(ctx);
    ds4_tokens_free(&prompt[0]);
    ds4_tokens_free(&prompt[1]);
    ds4_engine_close(engine);
    printf("%s\n", failed ? "FAILURES" : "all checks passed");
    return failed ? 1 : 0;
}
