/* Motif-3 persistent three-bank regression.
 *
 *   ./tests/test_motif3_batch <model.gguf> [--partial-only]
 *
 * The persistent batch path must match three independent greedy sessions.
 * --partial-only runs just the partial-prefix checkpoint gate: request-
 * boundary and periodic SWA-window checkpoints, a ring-wrapping restore,
 * and cold-oracle greedy parity for every partial fork.
 */
#include "../ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int serial_greedy(ds4_engine *engine, const ds4_tokens *prompt,
                         int budget, int *out, char *err, size_t errlen) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, 256) != 0 ||
        ds4_session_sync(session, prompt, err, errlen) != 0) {
        ds4_session_free(session);
        return 1;
    }
    for (int i = 0; i < budget; i++) {
        out[i] = ds4_session_argmax(session);
        if (i + 1 < budget &&
            ds4_session_eval(session, out[i], err, errlen) != 0) {
            ds4_session_free(session);
            return 1;
        }
    }
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
    int place_bank;
    int next;
    int admitted_cached;
    int admitted_computed;
    int admitted_bank;
    int token;
    int done;
    int failed;
} restore_case;

static int restore_admitted(void *ud, void *user, int cached,
                            int computed, int bank) {
    (void)user;
    restore_case *test = ud;
    test->admitted_cached = cached;
    test->admitted_computed = computed;
    test->admitted_bank = bank;
    return 1;
}

static int restore_admit(void *ud, ds4_cont_request *req) {
    restore_case *test = ud;
    if (test->next++) return 0;
    memset(req, 0, sizeof(*req));
    req->tokens = test->prompt->v;
    req->n = test->prompt->len;
    req->max_new = 1;
    req->eos = -1;
    req->n_cached = test->cached;
    req->place_bank = test->place_bank + 1;
    req->on_admitted = restore_admitted;
    req->user = test;
    return 1;
}

static void restore_done(void *ud, void *user, const int *tokens,
                         int n, int finish) {
    (void)user;
    (void)finish;
    restore_case *test = ud;
    if (!tokens || n != 1) {
        test->failed = 1;
        return;
    }
    test->token = tokens[0];
    test->done = 1;
}

/* ---- partial-prefix checkpoint gate ------------------------------------- */

typedef struct {
    const ds4_tokens *prompt;
    int cached;
    int place_bank;   /* 0-based target; -1 = engine pick */
    int fork_bank;    /* 0-based source; -1 = none */
    ds4_batch_ctx *ctx;
    int checkpoint_at, checkpoints;
    ds4_session_payload_file history;
    int next;
    int admitted_cached;
    int admitted_computed;
    int admitted_bank;
    int token;
    int done;
    int failed;
} partial_case;

static void partial_checkpoint(void *ud, void *user, int bank, int current) {
    (void)user;
    partial_case *test = ud;
    const int *tokens = NULL;
    char err[256] = "";
    if (current != test->checkpoint_at ||
        ds4_batch_ctx_bank_committed(test->ctx, bank, &tokens) != current ||
        !tokens || memcmp(tokens, test->prompt->v, (size_t)current * sizeof(int)) ||
        ds4_cont_bank_stage_payload(test->ctx, (uint32_t)bank,
                                   &test->history, err, sizeof(err))) {
        fprintf(stderr, "Motif-3 history capture failed: %s\n", err);
        test->failed = 1;
    }
    test->checkpoints++;
}

static int partial_admitted(void *ud, void *user, int cached,
                            int computed, int bank) {
    (void)user;
    partial_case *test = ud;
    test->admitted_cached = cached;
    test->admitted_computed = computed;
    test->admitted_bank = bank;
    return 1;
}

static int partial_admit(void *ud, ds4_cont_request *req) {
    partial_case *test = ud;
    if (test->next++) return 0;
    memset(req, 0, sizeof(*req));
    req->tokens = test->prompt->v;
    req->n = test->prompt->len;
    req->max_new = 1;
    req->eos = -1;
    req->n_cached = test->cached;
    req->place_bank = test->place_bank + 1;
    req->fork_bank = test->fork_bank + 1;
    req->checkpoint_at = test->checkpoint_at;
    req->on_checkpoint = test->checkpoint_at ? partial_checkpoint : NULL;
    req->on_admitted = partial_admitted;
    req->user = test;
    return 1;
}

static void partial_done(void *ud, void *user, const int *tokens,
                         int n, int finish) {
    (void)user;
    (void)finish;
    partial_case *test = ud;
    if (!tokens || n != 1) {
        test->failed = 1;
        return;
    }
    test->token = tokens[0];
    test->done = 1;
}

static int run_partial_request(
        ds4_batch_ctx *ctx, const ds4_tokens *prompt, int cached,
        int place_bank, int fork_bank, partial_case *test,
        char *err, size_t errlen) {
    memset(test, 0, sizeof(*test));
    test->prompt = prompt;
    test->cached = cached;
    test->place_bank = place_bank;
    test->fork_bank = fork_bank;
    test->admitted_cached = -1;
    test->admitted_computed = -1;
    test->admitted_bank = -1;
    err[0] = '\0';
    return ds4_engine_continuous_generate(
               ctx, partial_admit, NULL, partial_done, test,
               err, errlen) != 0 ||
           test->failed || !test->done;
}

/* One divergent branch: a cold oracle on cold_bank, then a partial fork of
 * source_bank on partial_bank.  Passes when the fork restored exactly
 * expected_cached tokens, replayed the rest, produced the oracle's greedy
 * token, and left the source frontier untouched. */
static int expect_motif3_partial(
        ds4_batch_ctx *ctx, const ds4_tokens *prompt,
        int source_bank, int source_len, int cut, int expected_cached,
        int cold_bank, int partial_bank, char *err, size_t errlen) {
    partial_case cold;
    if (run_partial_request(
            ctx, prompt, 0, cold_bank, -1, &cold, err, errlen) ||
        cold.admitted_cached != 0 ||
        cold.admitted_computed != prompt->len ||
        cold.admitted_bank != cold_bank) {
        fprintf(stderr,
                "Motif-3 partial cold oracle failed: %s done=%d "
                "split=%d+%d bank=%d\n",
                err, cold.done, cold.admitted_cached,
                cold.admitted_computed, cold.admitted_bank);
        return 1;
    }
    partial_case fork;
    if (run_partial_request(
            ctx, prompt, cut, partial_bank, source_bank, &fork,
            err, errlen) ||
        fork.token != cold.token ||
        fork.admitted_cached != expected_cached ||
        fork.admitted_computed != prompt->len - expected_cached ||
        fork.admitted_bank != partial_bank ||
        ds4_batch_ctx_bank_committed(ctx, source_bank, NULL) != source_len ||
        ds4_batch_ctx_bank_committed(ctx, partial_bank, NULL) !=
            prompt->len) {
        fprintf(stderr,
                "Motif-3 partial fork failed: %s done=%d token=%d/%d "
                "cut=%d split=%d+%d bank=%d source=%d destination=%d\n",
                err, fork.done, fork.token, cold.token, cut,
                fork.admitted_cached, fork.admitted_computed,
                fork.admitted_bank,
                ds4_batch_ctx_bank_committed(ctx, source_bank, NULL),
                ds4_batch_ctx_bank_committed(ctx, partial_bank, NULL));
        return 1;
    }
    fprintf(stderr,
            "Motif-3 partial fork: source=%d cut=%d checkpoint=%d "
            "replay=%d token=%d\n",
            source_len, cut, expected_cached,
            prompt->len - expected_cached, fork.token);
    return 0;
}

/* Grow one source bank through request-boundary checkpoints, then fork two
 * divergent branches against a cold oracle.  `targets` are the successive
 * source frontiers (boundary checkpoints); cuts[i] must restore
 * checkpoints[i]. */
static int run_partial_stage(
        ds4_engine *engine, int ctx_size, const ds4_tokens *seed,
        const int *targets, int n_targets,
        const int *cuts, const int *checkpoints, int n_cuts) {
    ds4_batch_ctx *ctx = NULL;
    ds4_tokens source = {0};
    ds4_tokens branch = {0};
    partial_case req;
    char err[256] = "";
    int failed = 0;

    if (ds4_batch_ctx_create_fit(
            engine, ctx_size, 4, 4 * ctx_size, &ctx,
            err, sizeof(err)) != 0 ||
        !ctx || !ds4_batch_ctx_supports_partial_reuse(ctx)) {
        fprintf(stderr, "Motif-3 partial gate context failed: %s\n", err);
        failed = 1;
        goto done;
    }
    for (int t = 0; t < n_targets; t++) {
        const int cached = source.len;
        while (source.len < targets[t])
            ds4_tokens_push(&source, seed->v[source.len % seed->len]);
        if (run_partial_request(
                ctx, &source, cached, 0, -1, &req, err, sizeof(err)) ||
            req.admitted_cached != cached ||
            req.admitted_computed != targets[t] - cached ||
            ds4_batch_ctx_bank_committed(ctx, 0, NULL) != targets[t]) {
            fprintf(stderr,
                    "Motif-3 partial gate frontier %d failed: %s "
                    "split=%d+%d\n",
                    targets[t], err, req.admitted_cached,
                    req.admitted_computed);
            failed = 1;
            goto done;
        }
    }

    const int source_len = targets[n_targets - 1];
    for (int n = 0; n < n_cuts; n++) {
        ds4_tokens_free(&branch);
        memset(&branch, 0, sizeof(branch));
        for (int i = 0; i < cuts[n]; i++)
            ds4_tokens_push(&branch, source.v[i]);
        ds4_tokens_push(&branch, (source.v[cuts[n]] + 1) %
                                 ds4_engine_vocab_size(engine));
        if (expect_motif3_partial(
                ctx, &branch, 0, source_len, cuts[n], checkpoints[n],
                2, 3, err, sizeof(err)) != 0) {
            failed = 1;
            goto done;
        }
    }

done:
    ds4_tokens_free(&branch);
    ds4_tokens_free(&source);
    ds4_batch_ctx_destroy(ctx);
    return failed;
}

static int same_payload(const ds4_session_payload_file *a,
                        const ds4_session_payload_file *b) {
    if (a->bytes != b->bytes) return 0;
    FILE *left = fopen(a->path, "rb"), *right = fopen(b->path, "rb");
    int same = left && right;
    unsigned char x[4096], y[4096];
    uint64_t remaining = a->bytes;
    while (same && remaining) {
        const size_t n = remaining < sizeof(x) ? (size_t)remaining : sizeof(x);
        same = fread(x, 1, n, left) == n && fread(y, 1, n, right) == n && !memcmp(x, y, n);
        remaining -= n;
    }
    if (left) fclose(left);
    if (right) fclose(right);
    return same;
}

/* A generation-only suffix needs a checkpoint before it, even when no
 * periodic checkpoint fits the short prompt. Its callback payload and the
 * native SWA partial checkpoint must both describe that exact frontier. */
static int run_history_stage(ds4_engine *engine, const ds4_tokens *seed) {
    enum { CTX = 128, SOURCE = 64, HISTORY = 29, CUT = 35 };
    ds4_batch_ctx *ctx = NULL;
    ds4_tokens source = {0}, branch = {0};
    ds4_session_payload_file before = {0}, after = {0};
    partial_case initial = {0}, replay = {0}, cold = {0};
    char err[256] = "";
    int failed = 1;
    if (ds4_batch_ctx_create_fit(engine, CTX, 4, 4 * CTX, &ctx, err, sizeof(err)) || !ctx)
        goto done;
    while (source.len < SOURCE)
        ds4_tokens_push(&source, seed->v[source.len % seed->len]);
    initial = (partial_case){.prompt = &source, .place_bank = 0, .fork_bank = -1,
        .ctx = ctx, .checkpoint_at = HISTORY};
    if (ds4_engine_continuous_generate(ctx, partial_admit, NULL, partial_done,
                                      &initial, err, sizeof(err)) ||
        initial.failed || !initial.done || initial.checkpoints != 1 ||
        initial.admitted_cached != 0 || initial.admitted_computed != SOURCE ||
        ds4_cont_bank_stage_payload(ctx, 0u, &before, err, sizeof(err)))
        goto done;
    for (int i = 0; i < CUT; i++) ds4_tokens_push(&branch, source.v[i]);
    ds4_tokens_push(&branch, (source.v[CUT] + 1) % ds4_engine_vocab_size(engine));
    if (expect_motif3_partial(ctx, &branch, 0, SOURCE, CUT, HISTORY, 2, 3,
                             err, sizeof(err)) ||
        ds4_cont_bank_stage_payload(ctx, 0u, &after, err, sizeof(err)) ||
        !same_payload(&before, &after))
        goto done;
    FILE *saved = fopen(initial.history.path, "rb");
    if (!saved) goto done;
    const int restored = ds4_cont_bank_restore_payload(ctx, 1u, saved,
        initial.history.bytes, err, sizeof(err));
    fclose(saved);
    const int *tokens = NULL;
    if (restored || ds4_batch_ctx_bank_committed(ctx, 1, &tokens) != HISTORY ||
        !tokens || memcmp(tokens, source.v, HISTORY * sizeof(int)))
        goto done;
    if (run_partial_request(ctx, &branch, HISTORY, 1, -1, &replay, err, sizeof(err)) ||
        run_partial_request(ctx, &branch, 0, 2, -1, &cold, err, sizeof(err)) ||
        replay.admitted_cached != HISTORY || replay.token != cold.token)
        goto done;
    failed = 0;
    fprintf(stderr, "Motif-3 history cut: boundary=%d partial=%d source_payload=unchanged "
                    "disk_tokens=exact cold_token=equal\n", HISTORY, HISTORY);
done:
    if (failed) fprintf(stderr, "Motif-3 history frontier gate failed: %s\n", err);
    ds4_session_payload_file_free(&initial.history);
    ds4_session_payload_file_free(&before);
    ds4_session_payload_file_free(&after);
    ds4_tokens_free(&source);
    ds4_tokens_free(&branch);
    ds4_batch_ctx_destroy(ctx);
    return failed;
}

static int run_motif3_partial_gate(ds4_engine *engine) {
    ds4_tokens seed = {0};
    int failed = 0;
    ds4_tokenize_text(engine, "The capital of France is Paris. ", &seed);
    if (seed.len <= 0) {
        fprintf(stderr, "Motif-3 partial gate seed tokenization failed\n");
        ds4_tokens_free(&seed);
        return 1;
    }

    failed = run_history_stage(engine, &seed);

    /* Stage 1: request-boundary checkpoints inside the linear (unwrapped)
     * window regime; mirrors the Solar 16/24/32 gate. */
    if (!failed) {
        const int targets[3] = {16, 24, 32};
        const int cuts[2] = {19, 27};
        const int checkpoints[2] = {16, 24};
        failed |= run_partial_stage(
            engine, 128, &seed, targets, 3, cuts, checkpoints, 2);
    }

    /* Stage 2: 4096-chunk regime.  The 4300 boundary checkpoint's window
     * rows [4172,4300) wrap the 4225-slot SWA ring, and the 4200 cut lands
     * on the PERIODIC chunk-boundary checkpoint at 4096, so both capture
     * kinds and the two-segment restore are exercised.  ctx 8192 keeps the
     * SWA cap (128+1+4096) below the context so the ring actually wraps. */
    if (!failed && setenv("DS4_MOTIF3_PREFILL_CHUNK", "4096", 1) == 0) {
        const int targets[2] = {4300, 4500};
        const int cuts[2] = {4200, 4400};
        const int checkpoints[2] = {4096, 4300};
        failed |= run_partial_stage(
            engine, 8192, &seed, targets, 2, cuts, checkpoints, 2);
        setenv("DS4_MOTIF3_PREFILL_CHUNK", "16", 1);
    }

    ds4_tokens_free(&seed);
    if (!failed)
        fprintf(stderr, "Motif-3 partial reuse checkpoint gate passed\n");
    return failed;
}

int main(int argc, char **argv) {
    const int partial_only = argc == 3 &&
        strcmp(argv[2], "--partial-only") == 0;
    if (argc != 2 && !partial_only) {
        fprintf(stderr, "usage: %s <model.gguf> [--partial-only]\n",
                argv[0]);
        return 2;
    }
    if (setenv("DS4_SESSION_LAZY_GRAPH", "1", 1) != 0 ||
        setenv("DS4_NO_BOOT_PREWARM", "1", 1) != 0 ||
        setenv("DS4_MOTIF3_PREFILL_CHUNK", "16", 1) != 0) {
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
        fprintf(stderr, "Motif-3 engine open failed\n");
        return 1;
    }
    if (ds4_engine_routed_quant_bits(engine) != 2) {
        fprintf(stderr, "Motif-3 routed quant identity is not IQ2/Q2K\n");
        ds4_engine_close(engine);
        return 1;
    }
    if (partial_only) {
        const int failed_gate = run_motif3_partial_gate(engine);
        ds4_engine_close(engine);
        printf("%s\n", failed_gate ? "FAILURES" : "all checks passed");
        return failed_gate ? 1 : 0;
    }

    const char *text[3] = {
        "The capital of France is",
        "Two plus two equals",
        "Write one word for the color of grass:",
    };
    ds4_tokens prompt[3] = {0};
    int oracle[3][5] = {{0}};
    char err[256] = {0};
    int failed = 0;
    ds4_batch_ctx *ctx = NULL;
    ds4_session_payload_file bank_payload = {0};
    ds4_tokens restored_prompt = {0};
    for (int i = 0; i < 3; i++) {
        ds4_tokenize_text(engine, text[i], &prompt[i]);
        if (prompt[i].len <= 0 ||
            serial_greedy(engine, &prompt[i], 5, oracle[i],
                          err, sizeof(err)) != 0) {
            fprintf(stderr, "Motif-3 scalar oracle %d failed: %s\n", i, err);
            failed = 1;
            goto cleanup;
        }
    }
    if (serial_snapshot_roundtrip(
            engine, &prompt[0], oracle[0], err, sizeof(err)) != 0) {
        fprintf(stderr, "Motif-3 serial snapshot round-trip failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    if (!ds4_engine_supports_batching(engine)) {
        fprintf(stderr, "Motif-3 batching capability is disabled\n");
        failed = 1;
        goto cleanup;
    }
    if (ds4_batch_ctx_create_fit(
            engine, 256, 3, 48, &ctx, err, sizeof(err)) != 0 ||
        !ctx || ds4_batch_ctx_max_seq(ctx) != 3 ||
        !ds4_batch_ctx_supports_partial_reuse(ctx)) {
        fprintf(stderr, "Motif-3 batch context failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    /* Four tokens keep bank 0 alive while banks 1 and 2 finish admission;
     * the third scheduler pass therefore executes an actual three-row decode. */
    const int budget[3] = {4, 4, 4};
    const int eos[3] = {
        ds4_token_eos(engine), ds4_token_eos(engine), ds4_token_eos(engine),
    };
    ds4_batch_gen_result got[3] = {0};
    if (ds4_engine_batched_generate_ctx(
            ctx, prompt, 3, budget, eos, got, err, sizeof(err)) != 0) {
        fprintf(stderr, "Motif-3 persistent batch failed: %s\n", err);
        failed = 1;
    }
    for (int i = 0; i < 3; i++) {
        if (!failed && (got[i].n_tokens != 4 ||
            memcmp(got[i].tokens, oracle[i],
                   4u * sizeof(oracle[i][0])) != 0)) {
            fprintf(stderr,
                    "Motif-3 row %d differs (n=%d): "
                    "got=%d,%d,%d,%d want=%d,%d,%d,%d\n",
                    i, got[i].n_tokens,
                    got[i].n_tokens > 0 ? got[i].tokens[0] : -1,
                    got[i].n_tokens > 1 ? got[i].tokens[1] : -1,
                    got[i].n_tokens > 2 ? got[i].tokens[2] : -1,
                    got[i].n_tokens > 3 ? got[i].tokens[3] : -1,
                    oracle[i][0], oracle[i][1],
                    oracle[i][2], oracle[i][3]);
            failed = 1;
        }
        free(got[i].tokens);
    }

    const int *committed = NULL;
    const int committed_n = ds4_batch_ctx_bank_committed(ctx, 0, &committed);
    if (!failed &&
        (committed_n != prompt[0].len + 3 ||
         memcmp(committed, prompt[0].v,
                (size_t)prompt[0].len * sizeof(*committed)) != 0 ||
         memcmp(committed + prompt[0].len, oracle[0],
                3u * sizeof(*committed)) != 0 ||
         ds4_cont_bank_payload_bytes(ctx, 0u) == 0u ||
         ds4_cont_bank_stage_payload(
             ctx, 0u, &bank_payload, err, sizeof(err)) != 0)) {
        fprintf(stderr, "Motif-3 durable bank stage failed: %s\n", err);
        failed = 1;
    }
    FILE *bank_fp = NULL;
    if (!failed) bank_fp = fopen(bank_payload.path, "rb");
    if (!failed &&
        (!bank_fp || ds4_cont_bank_restore_payload(
             ctx, 1u, bank_fp, bank_payload.bytes,
             err, sizeof(err)) != 0)) {
        fprintf(stderr, "Motif-3 durable bank restore failed: %s\n", err);
        failed = 1;
    }
    if (bank_fp && fclose(bank_fp) != 0 && !failed) {
        perror("Motif-3 durable bank close");
        failed = 1;
    }

    if (!failed) {
        restored_prompt.len = committed_n + 1;
        restored_prompt.cap = restored_prompt.len;
        restored_prompt.v = malloc(
            (size_t)restored_prompt.len * sizeof(*restored_prompt.v));
        if (!restored_prompt.v) {
            fprintf(stderr, "Motif-3 restored prompt allocation failed\n");
            failed = 1;
        } else {
            memcpy(restored_prompt.v, committed,
                   (size_t)committed_n * sizeof(*restored_prompt.v));
            restored_prompt.v[committed_n] = oracle[0][3];
        }
    }
    restore_case restored = {
        .prompt = &restored_prompt,
        .cached = committed_n,
        .place_bank = 1,
        .admitted_cached = -1,
        .admitted_computed = -1,
        .admitted_bank = -1,
    };
    if (!failed &&
        (ds4_engine_continuous_generate(
             ctx, restore_admit, NULL, restore_done, &restored,
             err, sizeof(err)) != 0 ||
         restored.failed || !restored.done ||
         restored.token != oracle[0][4] ||
         restored.admitted_cached != committed_n ||
         restored.admitted_computed != 1 ||
         restored.admitted_bank != 1)) {
        fprintf(stderr,
                "Motif-3 restored bank continuation failed: %s "
                "token=%d want=%d split=%d+%d bank=%d\n",
                err, restored.token, oracle[0][4],
                restored.admitted_cached, restored.admitted_computed,
                restored.admitted_bank);
        failed = 1;
    }

cleanup:
    ds4_session_payload_file_free(&bank_payload);
    ds4_tokens_free(&restored_prompt);
    ds4_batch_ctx_destroy(ctx);
    for (int i = 0; i < 3; i++) ds4_tokens_free(&prompt[i]);
    ds4_engine_close(engine);
    printf("%s\n", failed ? "FAILURES" : "all checks passed");
    return failed ? 1 : 0;
}
