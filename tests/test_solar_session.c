/* Public Solar Open2 session-lifecycle regression.
 *
 *   ./tests/test_solar_session <first-model-shard.gguf>
 *
 * It pins both the live-session contract and exact persistence of Solar's
 * recurrent KDA state plus live GQA KV rows.
 */
#include "../ds4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int copy_logits(ds4_session *s, float *out, int n_vocab,
                       const char *where) {
    if (ds4_session_copy_logits(s, out, n_vocab) != n_vocab) {
        fprintf(stderr, "%s: logits copy failed\n", where);
        return 1;
    }
    return 0;
}

static int expect_same_logits(const float *want, const float *got,
                              int n_vocab, const char *where) {
    if (memcmp(want, got, (size_t)n_vocab * sizeof(*want)) != 0) {
        fprintf(stderr, "%s: deterministic replay logits differ\n", where);
        return 1;
    }
    return 0;
}

static double max_abs_diff(const float *a, const float *b, int n,
                           int *nonfinite) {
    double worst = 0.0;
    *nonfinite = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) {
            *nonfinite = 1;
            continue;
        }
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

typedef struct {
    int token;
    int emitted;
    int done;
} generation_capture;

static void capture_token(void *ud, int token) {
    generation_capture *capture = ud;
    capture->token = token;
    capture->emitted++;
}

static void capture_done(void *ud) {
    generation_capture *capture = ud;
    capture->done++;
}

typedef struct solar_cont_test solar_cont_test;

typedef struct {
    solar_cont_test *test;
    int id;
} solar_cont_user;

struct solar_cont_test {
    const ds4_tokens *prompt[2];
    solar_cont_user user[2];
    int n_requests;
    int first_cached;
    int first_max_new;
    int first_place_bank;
    int first_fork_bank;
    int next_admit;
    int allow_second;
    int alive_calls;
    int alive_fail_after;
    int abort_first_token;
    int on_token_calls;
    int sample_override;
    int failed;
    int done;
    int bank_used[2];
    int admitted_cached[2];
    int admitted_computed[2];
    int admitted_bank[2];
    int tokens[2][4];
    int n_tokens[2];
    int finish[2];
};

static int solar_cont_sample_override(void *ud, void *user) {
    (void)ud;
    solar_cont_user *u = user;
    return u->test->sample_override;
}

static int solar_cont_alive(void *ud, void *user) {
    (void)user;
    solar_cont_test *test = ud;
    return test->alive_calls++ < test->alive_fail_after;
}

static int solar_cont_on_admitted(void *ud, void *user, int n_cached,
                                  int n_computed, int bank) {
    (void)ud;
    solar_cont_user *u = user;
    u->test->admitted_cached[u->id] = n_cached;
    u->test->admitted_computed[u->id] = n_computed;
    u->test->admitted_bank[u->id] = bank;
    return 1;
}

static int solar_cont_admit(void *ud, ds4_cont_request *req) {
    solar_cont_test *test = ud;
    if (test->next_admit >= test->n_requests) return 0;
    if (test->next_admit == 1 && !test->allow_second) return 0;
    const int i = test->next_admit++;
    memset(req, 0, sizeof(*req));
    req->tokens = test->prompt[i]->v;
    req->n = test->prompt[i]->len;
    req->max_new = i == 0 ? test->first_max_new : 1;
    req->eos = -1;
    req->user = &test->user[i];
    req->on_admitted = solar_cont_on_admitted;
    req->bank_used = &test->bank_used[i];
    if (test->alive_fail_after > 0) req->alive = solar_cont_alive;
    if (test->sample_override != DS4_SAMPLE_OVERRIDE_NONE)
        req->sample_override = solar_cont_sample_override;
    if (i == 0) {
        req->place_bank = test->first_place_bank;
        req->n_cached = test->first_cached;
        req->fork_bank = test->first_fork_bank;
    }
    return 1;
}

static int solar_cont_on_token(void *ud, void *user, int token) {
    (void)token;
    solar_cont_test *test = ud;
    solar_cont_user *u = user;
    test->on_token_calls++;
    if (u->id == 0) test->allow_second = 1;
    if (u->id == 0 && test->abort_first_token) return 0;
    return 1;
}

static void solar_cont_on_done(void *ud, void *user, const int *tokens,
                               int n, int finish) {
    solar_cont_test *test = ud;
    solar_cont_user *u = user;
    if (!tokens || n < 0 || n > 4) {
        test->failed = 1;
        return;
    }
    memcpy(test->tokens[u->id], tokens, (size_t)n * sizeof(*tokens));
    test->n_tokens[u->id] = n;
    test->finish[u->id] = finish;
    test->done++;
}

static int expect_solar_partial(
        ds4_batch_ctx *ctx, const ds4_tokens *prompt,
        int source_bank, int source_len, int cut, int expected_cached,
        int cold_bank, int partial_bank, char *err, size_t errlen) {
    solar_cont_test cold = {0};
    cold.prompt[0] = prompt;
    cold.n_requests = 1;
    cold.first_max_new = 1;
    cold.first_place_bank = cold_bank + 1;
    cold.user[0].test = &cold;
    cold.user[0].id = 0;
    cold.bank_used[0] = -1;
    cold.admitted_cached[0] = -1;
    cold.admitted_computed[0] = -1;
    cold.admitted_bank[0] = -1;
    err[0] = '\0';
    if (ds4_engine_continuous_generate(
            ctx, solar_cont_admit, solar_cont_on_token,
            solar_cont_on_done, &cold, err, errlen) != 0 ||
        cold.failed || cold.done != 1 || cold.n_tokens[0] != 1 ||
        cold.admitted_cached[0] != 0 ||
        cold.admitted_computed[0] != prompt->len ||
        cold.admitted_bank[0] != cold_bank) {
        fprintf(stderr,
                "Solar partial cold oracle failed: %s done=%d split=%d+%d "
                "bank=%d\n",
                err, cold.done, cold.admitted_cached[0],
                cold.admitted_computed[0], cold.admitted_bank[0]);
        return 1;
    }

    solar_cont_test partial = {0};
    partial.prompt[0] = prompt;
    partial.n_requests = 1;
    partial.first_cached = cut;
    partial.first_max_new = 1;
    partial.first_place_bank = partial_bank + 1;
    partial.first_fork_bank = source_bank + 1;
    partial.user[0].test = &partial;
    partial.user[0].id = 0;
    partial.bank_used[0] = -1;
    partial.admitted_cached[0] = -1;
    partial.admitted_computed[0] = -1;
    partial.admitted_bank[0] = -1;
    err[0] = '\0';
    if (ds4_engine_continuous_generate(
            ctx, solar_cont_admit, solar_cont_on_token,
            solar_cont_on_done, &partial, err, errlen) != 0 ||
        partial.failed || partial.done != 1 || partial.n_tokens[0] != 1 ||
        partial.tokens[0][0] != cold.tokens[0][0] ||
        partial.admitted_cached[0] != expected_cached ||
        partial.admitted_computed[0] != prompt->len - expected_cached ||
        partial.admitted_bank[0] != partial_bank ||
        (source_bank != partial_bank &&
         ds4_batch_ctx_bank_committed(ctx, source_bank, NULL) != source_len) ||
        ds4_batch_ctx_bank_committed(ctx, partial_bank, NULL) != prompt->len) {
        fprintf(stderr,
                "Solar partial fork failed: %s done=%d token=%d/%d "
                "cut=%d split=%d+%d bank=%d source=%d destination=%d\n",
                err, partial.done, partial.tokens[0][0], cold.tokens[0][0],
                cut, partial.admitted_cached[0],
                partial.admitted_computed[0], partial.admitted_bank[0],
                ds4_batch_ctx_bank_committed(ctx, source_bank, NULL),
                ds4_batch_ctx_bank_committed(ctx, partial_bank, NULL));
        return 1;
    }
    fprintf(stderr,
            "Solar partial fork: source=%d cut=%d checkpoint=%d replay=%d "
            "token=%d\n",
            source_len, cut, expected_cached,
            prompt->len - expected_cached, partial.tokens[0][0]);
    return 0;
}

static int run_solar_cont_request(
        ds4_batch_ctx *ctx, const ds4_tokens *prompt, int cached,
        int max_new, int place_bank, int fork_bank,
        solar_cont_test *test, char *err, size_t errlen) {
    memset(test, 0, sizeof(*test));
    test->prompt[0] = prompt;
    test->n_requests = 1;
    test->first_cached = cached;
    test->first_max_new = max_new;
    test->first_place_bank = place_bank + 1;
    test->first_fork_bank = fork_bank + 1;
    test->user[0].test = test;
    test->user[0].id = 0;
    test->bank_used[0] = -1;
    test->admitted_cached[0] = -1;
    test->admitted_computed[0] = -1;
    test->admitted_bank[0] = -1;
    err[0] = '\0';
    return ds4_engine_continuous_generate(
               ctx, solar_cont_admit, solar_cont_on_token,
               solar_cont_on_done, test, err, errlen) != 0 ||
           test->failed || test->done != 1 || test->n_tokens[0] != max_new;
}

static int run_solar_partial_gate(ds4_engine *engine,
                                  const ds4_tokens *seed) {
    ds4_batch_ctx *ctx = NULL;
    ds4_tokens source = {0};
    ds4_tokens branch = {0};
    solar_cont_test req = {0};
    char err[256] = "";
    int failed = 0;

    if (ds4_batch_ctx_create_fit(
            engine, 128, 4, 16, &ctx, err, sizeof(err)) != 0 ||
        !ctx || !ds4_batch_ctx_supports_partial_reuse(ctx)) {
        fprintf(stderr, "Solar partial gate context failed: %s\n", err);
        failed = 1;
        goto done;
    }
    for (int i = 0; i < 16; i++)
        ds4_tokens_push(&source, seed->v[i % seed->len]);
    if (run_solar_cont_request(
            ctx, &source, 0, 1, 0, -1, &req, err, sizeof(err)) ||
        req.admitted_cached[0] != 0 || req.admitted_computed[0] != 16 ||
        ds4_batch_ctx_bank_committed(ctx, 0, NULL) != 16) {
        fprintf(stderr, "Solar partial gate source failed: %s\n", err);
        failed = 1;
        goto done;
    }

    for (int target = 24; target <= 32; target += 8) {
        const int cached = source.len;
        while (source.len < target)
            ds4_tokens_push(&source, seed->v[source.len % seed->len]);
        if (run_solar_cont_request(
                ctx, &source, cached, 1, 0, -1,
                &req, err, sizeof(err)) ||
            req.admitted_cached[0] != cached ||
            req.admitted_computed[0] != target - cached ||
            ds4_batch_ctx_bank_committed(ctx, 0, NULL) != target) {
            fprintf(stderr,
                    "Solar partial gate checkpoint %d failed: %s split=%d+%d\n",
                    target, err, req.admitted_cached[0],
                    req.admitted_computed[0]);
            failed = 1;
            goto done;
        }
    }

    const int cuts[2] = {19, 27};
    const int checkpoints[2] = {16, 24};
    for (int n = 0; n < 2; n++) {
        ds4_tokens_free(&branch);
        for (int i = 0; i < cuts[n]; i++)
            ds4_tokens_push(&branch, source.v[i]);
        int diverge = (source.v[cuts[n]] + 1) %
                      ds4_engine_vocab_size(engine);
        ds4_tokens_push(&branch, diverge);
        if (expect_solar_partial(
                ctx, &branch, 0, 32, cuts[n], checkpoints[n],
                2, 3, err, sizeof(err)) != 0) {
            failed = 1;
            goto done;
        }
    }

    /* One-bank HTTP workers in-place truncate the source (src == dst). */
    ds4_tokens_free(&branch);
    for (int i = 0; i < cuts[0]; i++) {
        ds4_tokens_push(&branch, source.v[i]);
    }
    ds4_tokens_push(
        &branch, (source.v[cuts[0]] + 2) % ds4_engine_vocab_size(engine));
    if (expect_solar_partial(
            ctx, &branch, 0, 32, cuts[0], checkpoints[0],
            1, 0, err, sizeof(err)) != 0) {
        failed = 1;
        goto done;
    }

done:
    ds4_tokens_free(&branch);
    ds4_tokens_free(&source);
    ds4_batch_ctx_destroy(ctx);
    return failed;
}

static int solar_open_engine(ds4_engine **engine, const char *model_path) {
    ds4_engine_options opt = {0};
    opt.model_path = model_path;
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.defer_boot_prewarm = true;
    if (ds4_engine_open(engine, &opt) != 0) {
        fprintf(stderr, "Solar engine open failed\n");
        return 1;
    }
    return 0;
}

static int solar_mkstemp_payload(char *path, size_t path_len) {
    const char *dir = getenv("SOLAR_DISK_KV_DIR");
    if (dir == NULL || dir[0] == '\0') {
        dir = ".";
    }
    if (snprintf(path, path_len, "%s/solar-disk-kv-XXXXXX", dir) >=
        (int)path_len) {
        fprintf(stderr, "Solar disk-KV temp path is too long\n");
        return -1;
    }
    const int fd = mkstemp(path);
    if (fd < 0) {
        perror("Solar disk-KV mkstemp");
        return -1;
    }
    return fd;
}

/* Worker-restart disk KV: save a Solar payload, close the engine, reopen,
 * restore, and continue. Token identity is the contract; the 0.25 snapshot
 * logit bound is not raised. */
static int run_solar_disk_kv_gate(ds4_engine **engine, const char *model_path,
                                  const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    ds4_batch_ctx *ctx = NULL;
    ds4_tokens continued = {0};
    char err[256] = "";
    char serial_path[512] = "";
    char bank_path[512] = "";
    int failed = 0;
    int serial_fd = -1;
    int bank_fd = -1;
    FILE *fp = NULL;

    if (ds4_session_create(&session, *engine, 128) != 0) {
        fprintf(stderr, "Solar disk-KV session creation failed\n");
        return 1;
    }
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar disk-KV cold sync failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    const int serial_next = ds4_session_argmax(session);
    serial_fd = solar_mkstemp_payload(serial_path, sizeof(serial_path));
    if (serial_fd < 0) {
        ds4_session_free(session);
        return 1;
    }
    fp = fdopen(serial_fd, "wb");
    serial_fd = -1;
    if (fp == NULL ||
        ds4_session_save_payload(session, fp, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar disk-KV serial save failed: %s\n", err);
        if (fp != NULL) {
            fclose(fp);
        }
        ds4_session_free(session);
        unlink(serial_path);
        return 1;
    }
    if (fflush(fp) != 0 || fseek(fp, 0, SEEK_END) != 0) {
        perror("Solar disk-KV serial payload size");
        fclose(fp);
        ds4_session_free(session);
        unlink(serial_path);
        return 1;
    }
    const uint64_t serial_bytes = (uint64_t)ftell(fp);
    if (fclose(fp) != 0 || serial_bytes == 0u) {
        fprintf(stderr, "Solar disk-KV serial payload is empty\n");
        ds4_session_free(session);
        unlink(serial_path);
        return 1;
    }
    fp = NULL;
    err[0] = '\0';
    if (ds4_session_eval(session, serial_next, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar disk-KV serial oracle decode failed: %s\n",
                err);
        ds4_session_free(session);
        unlink(serial_path);
        return 1;
    }
    const int serial_decoded = ds4_session_argmax(session);
    ds4_session_free(session);
    session = NULL;
    ds4_engine_close(*engine);
    *engine = NULL;

    if (solar_open_engine(engine, model_path) != 0) {
        unlink(serial_path);
        return 1;
    }
    if (ds4_session_create(&session, *engine, 128) != 0) {
        fprintf(stderr, "Solar disk-KV restarted session failed\n");
        unlink(serial_path);
        return 1;
    }
    fp = fopen(serial_path, "rb");
    err[0] = '\0';
    if (fp == NULL ||
        ds4_session_load_payload(
            session, fp, serial_bytes, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar disk-KV serial load failed: %s\n", err);
        failed = 1;
        goto done;
    }
    if (fclose(fp) != 0) {
        perror("Solar disk-KV serial close");
        fp = NULL;
        failed = 1;
        goto done;
    }
    fp = NULL;
    if (ds4_session_pos(session) != prompt->len ||
        ds4_session_argmax(session) != serial_next) {
        fprintf(stderr,
                "Solar disk-KV serial restart did not reuse the prefix: "
                "pos=%d want=%d token=%d want=%d\n",
                ds4_session_pos(session), prompt->len,
                ds4_session_argmax(session), serial_next);
        failed = 1;
        goto done;
    }
    err[0] = '\0';
    if (ds4_session_eval(session, serial_next, err, sizeof(err)) != 0 ||
        ds4_session_argmax(session) != serial_decoded) {
        fprintf(stderr,
                "Solar disk-KV serial continuation token mismatch: %s "
                "got=%d want=%d\n",
                err, ds4_session_argmax(session), serial_decoded);
        failed = 1;
        goto done;
    }
    fprintf(stderr,
            "Solar disk-KV restart reuse: cached_tokens=%d serial_token=%d\n",
            prompt->len, serial_decoded);

    ds4_session_free(session);
    session = NULL;

    /* Continuous-bank payload is what --kv-disk-dir restores on the server. */
    if (ds4_batch_ctx_create_fit(
            *engine, 128, 4, 16, &ctx, err, sizeof(err)) != 0 ||
        ctx == NULL) {
        fprintf(stderr, "Solar disk-KV bank context failed: %s\n", err);
        failed = 1;
        goto done;
    }

    solar_cont_test source = {0};
    if (run_solar_cont_request(
            ctx, prompt, 0, 3, 0, -1, &source, err, sizeof(err)) ||
        source.admitted_cached[0] != 0 ||
        source.admitted_computed[0] != prompt->len ||
        source.n_tokens[0] != 3) {
        fprintf(stderr, "Solar disk-KV bank source failed: %s split=%d+%d\n",
                err, source.admitted_cached[0], source.admitted_computed[0]);
        failed = 1;
        goto done;
    }
    /* Last sampled token is not in KV. Committed history is prompt plus
     * the two forwarded generations; bank payloads store zeros for logits,
     * so restart reuse prefills that last sampled token as a suffix. */
    const int committed_len =
        ds4_batch_ctx_bank_committed(ctx, 0, NULL);
    if (committed_len != prompt->len + 2) {
        fprintf(stderr, "Solar disk-KV bank committed length %d\n",
                committed_len);
        failed = 1;
        goto done;
    }

    bank_fd = solar_mkstemp_payload(bank_path, sizeof(bank_path));
    if (bank_fd < 0) {
        failed = 1;
        goto done;
    }
    fp = fdopen(bank_fd, "wb");
    bank_fd = -1;
    err[0] = '\0';
    if (fp == NULL ||
        ds4_cont_bank_save_payload(ctx, 0, fp, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar disk-KV bank save failed: %s\n", err);
        failed = 1;
        goto done;
    }
    if (fflush(fp) != 0 || fseek(fp, 0, SEEK_END) != 0) {
        perror("Solar disk-KV bank payload size");
        failed = 1;
        goto done;
    }
    const uint64_t bank_bytes = (uint64_t)ftell(fp);
    if (fclose(fp) != 0 || bank_bytes == 0u) {
        fprintf(stderr, "Solar disk-KV bank payload is empty\n");
        fp = NULL;
        failed = 1;
        goto done;
    }
    fp = NULL;

    for (int i = 0; i < prompt->len; i++) {
        ds4_tokens_push(&continued, prompt->v[i]);
    }
    ds4_tokens_push(&continued, source.tokens[0][0]);
    ds4_tokens_push(&continued, source.tokens[0][1]);
    ds4_tokens_push(&continued, source.tokens[0][2]);
    solar_cont_test oracle = {0};
    if (run_solar_cont_request(
            ctx, &continued, committed_len, 1, 0, -1, &oracle, err,
            sizeof(err)) ||
        oracle.admitted_cached[0] != committed_len ||
        oracle.admitted_computed[0] != 1 ||
        oracle.n_tokens[0] != 1) {
        fprintf(stderr,
                "Solar disk-KV bank oracle failed: %s split=%d+%d n=%d\n",
                err, oracle.admitted_cached[0], oracle.admitted_computed[0],
                oracle.n_tokens[0]);
        failed = 1;
        goto done;
    }
    const int bank_oracle = oracle.tokens[0][0];

    ds4_batch_ctx_destroy(ctx);
    ctx = NULL;
    ds4_engine_close(*engine);
    *engine = NULL;

    if (solar_open_engine(engine, model_path) != 0) {
        failed = 1;
        goto done;
    }
    err[0] = '\0';
    if (ds4_batch_ctx_create_fit(
            *engine, 128, 4, 16, &ctx, err, sizeof(err)) != 0 ||
        ctx == NULL) {
        fprintf(stderr, "Solar disk-KV restarted bank context failed: %s\n",
                err);
        failed = 1;
        goto done;
    }
    fp = fopen(bank_path, "rb");
    err[0] = '\0';
    if (fp == NULL ||
        ds4_cont_bank_restore_payload(
            ctx, 0, fp, bank_bytes, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar disk-KV bank load failed: %s\n", err);
        failed = 1;
        goto done;
    }
    if (fclose(fp) != 0) {
        perror("Solar disk-KV bank close");
        fp = NULL;
        failed = 1;
        goto done;
    }
    fp = NULL;
    if (ds4_batch_ctx_bank_committed(ctx, 0, NULL) != committed_len) {
        fprintf(stderr,
                "Solar disk-KV bank restart committed %d want %d\n",
                ds4_batch_ctx_bank_committed(ctx, 0, NULL), committed_len);
        failed = 1;
        goto done;
    }

    solar_cont_test warm = {0};
    if (run_solar_cont_request(
            ctx, &continued, committed_len, 1, 0, -1, &warm, err,
            sizeof(err)) ||
        warm.admitted_cached[0] != committed_len ||
        warm.admitted_computed[0] != 1 ||
        warm.n_tokens[0] != 1 ||
        warm.tokens[0][0] != bank_oracle) {
        fprintf(stderr,
                "Solar disk-KV bank warm reuse failed: %s split=%d+%d n=%d "
                "token=%d want=%d\n",
                err, warm.admitted_cached[0], warm.admitted_computed[0],
                warm.n_tokens[0],
                warm.n_tokens[0] > 0 ? warm.tokens[0][0] : -1,
                bank_oracle);
        failed = 1;
        goto done;
    }
    fprintf(stderr,
            "Solar disk-KV restart reuse: cached_tokens=%d bank_token=%d\n",
            warm.admitted_cached[0], warm.tokens[0][0]);

done:
    if (fp != NULL) {
        fclose(fp);
    }
    if (serial_fd >= 0) {
        close(serial_fd);
    }
    if (bank_fd >= 0) {
        close(bank_fd);
    }
    if (serial_path[0] != '\0') {
        unlink(serial_path);
    }
    if (bank_path[0] != '\0') {
        unlink(bank_path);
    }
    ds4_tokens_free(&continued);
    if (session != NULL) {
        ds4_session_free(session);
    }
    if (ctx != NULL) {
        ds4_batch_ctx_destroy(ctx);
    }
    return failed;
}

int main(int argc, char **argv) {
    const int partial_only = argc == 3 &&
        strcmp(argv[2], "--partial-only") == 0;
    const int disk_kv_only = argc == 3 &&
        strcmp(argv[2], "--disk-kv-only") == 0;
    if (argc != 2 && !partial_only && !disk_kv_only) {
        fprintf(stderr,
                "usage: %s <first-model-shard.gguf> "
                "[--partial-only|--disk-kv-only]\n",
                argv[0]);
        return 2;
    }
    if (setenv("DS4_METAL_PREFILL_CHUNK", "3", 1) != 0 ||
        setenv("DS4_SOLAR_KV_FORMAT", "hybrid", 1) != 0 ||
        setenv("DS4_SESSION_LAZY_GRAPH", "1", 1) != 0 ||
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
    ds4_session *session = NULL;
    ds4_tokens prompt = {0};
    ds4_tokens prompt_alt = {0};
    ds4_tokens prompt_restored = {0};
    ds4_tokens prompt_interrupted = {0};
    ds4_tokens prompt_retained = {0};
    float *initial = NULL;
    float *decoded = NULL;
    float *warm = NULL;
    float *replayed = NULL;
    ds4_session_snapshot snapshot = {0};
    ds4_session_snapshot restored_snapshot = {0};
    ds4_session_payload_file bank_payload = {0};
    ds4_batch_gen_result primary_batch_oracle = {0};
    ds4_batch_gen_result alternate_batch_oracle = {0};
    char err[256] = "";
    int failed = 0;

    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "Solar engine open failed\n");
        return 1;
    }
    if (ds4_engine_layer_count(engine) != 48 ||
        ds4_engine_hidden_f32_values(engine) != 4096u ||
        ds4_engine_n_hc(engine) != 1) {
        fprintf(stderr,
                "unexpected Solar public shape: layers=%d hidden=%llu n_hc=%d\n",
                ds4_engine_layer_count(engine),
                (unsigned long long)ds4_engine_hidden_f32_values(engine),
                ds4_engine_n_hc(engine));
        failed = 1;
        goto cleanup;
    }

    ds4_tokens_push(&prompt, 128);
    ds4_tokens_push(&prompt, 29497);
    ds4_tokens_push(&prompt, 132);
    ds4_tokens_push(&prompt, 4767);
    ds4_tokens_push(&prompt_alt, 128);
    ds4_tokens_push(&prompt_alt, 29497);
    ds4_tokens_push(&prompt_alt, 132);
    ds4_tokens_push(&prompt_alt, 4768);
    if (partial_only) {
        failed = run_solar_partial_gate(engine, &prompt);
        goto cleanup;
    }
    if (disk_kv_only) {
        failed = run_solar_disk_kv_gate(&engine, argv[1], &prompt);
        goto cleanup;
    }
    if (ds4_session_create(&session, engine, 128) != 0) {
        fprintf(stderr, "Solar session creation failed\n");
        failed = 1;
        goto cleanup;
    }
    if (!ds4_session_graph_pending(session) ||
        ds4_session_prefill_cap(session) != 3) {
        fprintf(stderr, "Solar lazy graph/prefill-cap contract is wrong\n");
        failed = 1;
        goto cleanup;
    }

    const int n_vocab = ds4_engine_vocab_size(engine);
    initial = calloc((size_t)n_vocab, sizeof(*initial));
    decoded = calloc((size_t)n_vocab, sizeof(*decoded));
    warm = calloc((size_t)n_vocab, sizeof(*warm));
    replayed = calloc((size_t)n_vocab, sizeof(*replayed));
    if (!initial || !decoded || !warm || !replayed) {
        fprintf(stderr, "host logits allocation failed\n");
        failed = 1;
        goto cleanup;
    }

    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar cold sync failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    if (ds4_session_graph_pending(session) ||
        ds4_session_pos(session) != prompt.len ||
        copy_logits(session, initial, n_vocab, "cold sync")) {
        fprintf(stderr, "Solar cold-sync state is wrong\n");
        failed = 1;
        goto cleanup;
    }
    if (ds4_session_save_snapshot(session, &snapshot,
                                  err, sizeof(err)) != 0 ||
        snapshot.len != ds4_session_payload_bytes(session)) {
        fprintf(stderr, "Solar snapshot save failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    fprintf(stderr,
            "Solar snapshot: bytes=%llu MiB=%.2f tokens=%d\n",
            (unsigned long long)snapshot.len,
            (double)snapshot.len / 1048576.0,
            prompt.len);

    const uint64_t stable_generation = ds4_session_generation(session);
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0 ||
        ds4_session_pos(session) != prompt.len ||
        ds4_session_generation(session) != stable_generation ||
        copy_logits(session, replayed, n_vocab, "unchanged sync") ||
        expect_same_logits(initial, replayed, n_vocab, "unchanged sync")) {
        fprintf(stderr, "Solar unchanged sync was not a no-op: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    const int next = ds4_session_argmax(session);
    if (ds4_session_eval(session, next, err, sizeof(err)) != 0 ||
        ds4_session_pos(session) != prompt.len + 1 ||
        copy_logits(session, decoded, n_vocab, "first decode")) {
        fprintf(stderr, "Solar decode failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    const int decoded_argmax = ds4_session_argmax(session);

    ds4_session_rewind(session, prompt.len);
    err[0] = '\0';
    if (ds4_session_eval(session, next, err, sizeof(err)) == 0 ||
        ds4_session_pos(session) != prompt.len) {
        fprintf(stderr, "Solar decode unexpectedly accepted a rewound KDA state\n");
        failed = 1;
        goto cleanup;
    }

    if (ds4_session_load_snapshot(session, &snapshot,
                                  err, sizeof(err)) != 0 ||
        ds4_session_pos(session) != prompt.len ||
        ds4_session_argmax(session) != next ||
        copy_logits(session, replayed, n_vocab, "snapshot restore") ||
        expect_same_logits(initial, replayed, n_vocab, "snapshot restore")) {
        fprintf(stderr, "Solar snapshot restore failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    if (ds4_session_save_snapshot(session, &restored_snapshot,
                                  err, sizeof(err)) != 0 ||
        restored_snapshot.len != snapshot.len ||
        memcmp(restored_snapshot.ptr, snapshot.ptr,
               (size_t)snapshot.len) != 0) {
        fprintf(stderr, "Solar restored snapshot bytes differ: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    /* A malformed layout must be rejected before touching GPU state.  Loading
     * failures deliberately invalidate the old logical checkpoint. */
    const size_t layout_magic_offset = 5u * sizeof(uint32_t);
    if (restored_snapshot.len <= layout_magic_offset) {
        fprintf(stderr, "Solar snapshot is shorter than its fixed header\n");
        failed = 1;
        goto cleanup;
    }
    const uint8_t layout_byte = restored_snapshot.ptr[layout_magic_offset];
    restored_snapshot.ptr[layout_magic_offset] ^= UINT8_C(1);
    err[0] = '\0';
    const int corrupt_rc = ds4_session_load_snapshot(
        session, &restored_snapshot, err, sizeof(err));
    restored_snapshot.ptr[layout_magic_offset] = layout_byte;
    if (corrupt_rc == 0 || ds4_session_pos(session) != 0 || err[0] == '\0') {
        fprintf(stderr,
                "Solar corrupted snapshot was not rejected cleanly: %s\n",
                err);
        failed = 1;
        goto cleanup;
    }
    err[0] = '\0';
    if (ds4_session_load_snapshot(session, &snapshot,
                                  err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar recovery after rejected snapshot failed: %s\n",
                err);
        failed = 1;
        goto cleanup;
    }

    err[0] = '\0';
    if (ds4_session_eval(session, next, err, sizeof(err)) != 0 ||
        copy_logits(session, warm, n_vocab, "snapshot warm decode")) {
        fprintf(stderr, "Solar snapshot warm decode failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    const int warm_argmax = ds4_session_argmax(session);
    int nonfinite = 0;
    const double cold_warm_diff =
        max_abs_diff(decoded, warm, n_vocab, &nonfinite);
    fprintf(stderr,
            "Solar snapshot cold/warm decode: max_abs=%.9g argmax=%d/%d\n",
            cold_warm_diff, decoded_argmax, warm_argmax);
    if (nonfinite || cold_warm_diff > 0.25 ||
        warm_argmax != decoded_argmax) {
        fprintf(stderr,
                "Solar cold/warm snapshot decode drift exceeded its bound\n");
        failed = 1;
        goto cleanup;
    }

    if (ds4_session_load_snapshot(session, &snapshot,
                                  err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar second snapshot restore failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    err[0] = '\0';
    if (ds4_session_eval(session, next, err, sizeof(err)) != 0 ||
        copy_logits(session, replayed, n_vocab,
                    "snapshot deterministic replay")) {
        fprintf(stderr, "Solar snapshot replay decode failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    const double warm_replay_diff =
        max_abs_diff(warm, replayed, n_vocab, &nonfinite);
    fprintf(stderr,
            "Solar snapshot warm/replay decode: max_abs=%.9g argmax=%d/%d\n",
            warm_replay_diff, warm_argmax, ds4_session_argmax(session));
    if (nonfinite || warm_replay_diff > 1.0e-5 ||
        ds4_session_argmax(session) != warm_argmax) {
        fprintf(stderr, "Solar restored decode is not deterministic\n");
        failed = 1;
        goto cleanup;
    }

    ds4_session_invalidate(session);
    if (ds4_session_pos(session) != 0 ||
        ds4_session_sync(session, &prompt, err, sizeof(err)) != 0 ||
        copy_logits(session, replayed, n_vocab, "invalidated resync") ||
        expect_same_logits(initial, replayed, n_vocab, "invalidated resync")) {
        fprintf(stderr, "Solar invalidation/resync failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    /* Unsupported DeepSeek-only roots must reject without mutating the live
     * Solar session. */
    const int boundary_pos = ds4_session_pos(session);
    const uint64_t boundary_generation = ds4_session_generation(session);
    err[0] = '\0';
    if (ds4_session_layer_slice_reset(session, err, sizeof(err)) == 0 ||
        ds4_session_pos(session) != boundary_pos ||
        ds4_session_generation(session) != boundary_generation ||
        ds4_session_argmax(session) != next) {
        fprintf(stderr, "Solar layer-slice boundary was not side-effect free: %s\n",
                err);
        failed = 1;
        goto cleanup;
    }
    err[0] = '\0';
    if (ds4_session_output_head_bench(
            session, 1, stderr, err, sizeof(err)) == 0 ||
        ds4_session_pos(session) != boundary_pos) {
        fprintf(stderr, "Solar output-head bench boundary failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    /* The common batch boundary must expose isolated Solar state banks.  Use
     * two different prompts so an accidental shared KDA/KV lane changes at
     * least one bank's greedy seed. */
    ds4_session_invalidate(session);
    if (ds4_session_sync(session, &prompt_alt, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar alternate prompt sync failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    const int alt_next = ds4_session_argmax(session);
    ds4_session_invalidate(session);
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0 ||
        ds4_session_argmax(session) != next) {
        fprintf(stderr, "Solar primary prompt restore failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    if (ds4_session_eval(session, next, err, sizeof(err)) != 0 ||
        ds4_session_argmax(session) != decoded_argmax ||
        ds4_session_eval(session, decoded_argmax, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar scalar three-token oracle failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    const int third_next = ds4_session_argmax(session);
    if (ds4_session_eval(session, third_next, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar scalar fourth-token oracle failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    ds4_session_invalidate(session);
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0 ||
        ds4_session_argmax(session) != next) {
        fprintf(stderr, "Solar scalar oracle restore failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    /* The K=128 D2R prefill tier is value-equivalent, not bit-identical, to
     * the scalar/native path.  Derive the isolation oracle from fresh
     * one-bank batch contexts so this test detects bank-width contamination
     * without conflating it with the documented kernel fold-order change. */
    const int batch_oracle_budget = 6;
    const int one_token_single = 1;
    const int solar_eos = ds4_token_eos(engine);
    err[0] = '\0';
    if (ds4_engine_batched_generate_ex(
            engine, &prompt, 1, 128, &batch_oracle_budget, &solar_eos,
            &primary_batch_oracle, err, sizeof(err)) != 0 ||
        primary_batch_oracle.n_tokens != batch_oracle_budget) {
        fprintf(stderr, "Solar primary batch oracle failed: %s n=%d\n",
                err, primary_batch_oracle.n_tokens);
        failed = 1;
        goto cleanup;
    }
    err[0] = '\0';
    if (ds4_engine_batched_generate_ex(
            engine, &prompt_alt, 1, 128, &one_token_single, &solar_eos,
            &alternate_batch_oracle, err, sizeof(err)) != 0 ||
        alternate_batch_oracle.n_tokens != one_token_single) {
        fprintf(stderr, "Solar alternate batch oracle failed: %s n=%d\n",
                err, alternate_batch_oracle.n_tokens);
        failed = 1;
        goto cleanup;
    }
    if (!ds4_engine_supports_batching(engine)) {
        fprintf(stderr, "Solar batching capability is not enabled\n");
        failed = 1;
        goto cleanup;
    }
    ds4_batch_ctx *batch_ctx = NULL;
    err[0] = '\0';
    if (ds4_batch_ctx_create_fit(
            engine, 128, 4, 16, &batch_ctx, err, sizeof(err)) != 0 ||
        batch_ctx == NULL || ds4_batch_ctx_max_seq(batch_ctx) != 4 ||
        ds4_batch_ctx_seq_cap(batch_ctx) != 128) {
        fprintf(stderr, "Solar batch context creation failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    ds4_tokens batch_prompts[2] = { prompt, prompt_alt };
    ds4_batch_gen_result batch_results[2] = {0};
    const int one_token[2] = {1, 1};
    const int batch_eos[2] = {ds4_token_eos(engine), ds4_token_eos(engine)};
    err[0] = '\0';
    if (ds4_engine_batched_generate_ctx(
            batch_ctx, batch_prompts, 2, one_token, batch_eos,
            batch_results, err, sizeof(err)) != 0 ||
        batch_results[0].n_tokens != 1 ||
        batch_results[1].n_tokens != 1 ||
        batch_results[0].tokens[0] != primary_batch_oracle.tokens[0] ||
        batch_results[1].tokens[0] != alternate_batch_oracle.tokens[0]) {
        fprintf(stderr,
                "Solar persistent batch isolation failed: %s got=%d/%d want=%d/%d\n",
                err,
                batch_results[0].n_tokens ? batch_results[0].tokens[0] : -1,
                batch_results[1].n_tokens ? batch_results[1].tokens[0] : -1,
                primary_batch_oracle.tokens[0],
                alternate_batch_oracle.tokens[0]);
        free(batch_results[0].tokens);
        free(batch_results[1].tokens);
        ds4_batch_ctx_destroy(batch_ctx);
        failed = 1;
        goto cleanup;
    }
    free(batch_results[0].tokens);
    free(batch_results[1].tokens);

    solar_cont_test cont = {0};
    cont.prompt[0] = &prompt;
    cont.prompt[1] = &prompt_alt;
    cont.n_requests = 2;
    cont.first_cached = prompt.len;
    cont.first_max_new = 3;
    cont.first_place_bank = 1;
    cont.user[0].test = &cont;
    cont.user[0].id = 0;
    cont.user[1].test = &cont;
    cont.user[1].id = 1;
    cont.bank_used[0] = cont.bank_used[1] = -1;
    cont.admitted_cached[0] = cont.admitted_cached[1] = -1;
    cont.admitted_computed[0] = cont.admitted_computed[1] = -1;
    cont.admitted_bank[0] = cont.admitted_bank[1] = -1;
    err[0] = '\0';
    if (ds4_engine_continuous_generate(
            batch_ctx, solar_cont_admit, solar_cont_on_token,
            solar_cont_on_done, &cont, err, sizeof(err)) != 0 ||
        cont.failed || cont.done != 2 || cont.next_admit != 2 ||
        cont.n_tokens[0] != 3 || cont.n_tokens[1] != 1 ||
        cont.tokens[0][0] != primary_batch_oracle.tokens[0] ||
        cont.tokens[0][1] != primary_batch_oracle.tokens[1] ||
        cont.tokens[0][2] != primary_batch_oracle.tokens[2] ||
        cont.tokens[1][0] != alternate_batch_oracle.tokens[0] ||
        cont.bank_used[0] != 0 || cont.bank_used[1] != 1 ||
        cont.admitted_cached[0] != prompt.len ||
        cont.admitted_computed[0] != 0 ||
        cont.admitted_cached[1] != 0 ||
        cont.admitted_computed[1] != prompt_alt.len ||
        cont.admitted_bank[0] != 0 || cont.admitted_bank[1] != 1) {
        fprintf(stderr,
                "Solar rolling batch failed: %s done=%d admit=%d "
                "n=%d/%d tok0=%d,%d,%d tok1=%d banks=%d/%d "
                "split=%d+%d/%d+%d\n",
                err, cont.done, cont.next_admit,
                cont.n_tokens[0], cont.n_tokens[1],
                cont.tokens[0][0], cont.tokens[0][1], cont.tokens[0][2],
                cont.tokens[1][0], cont.bank_used[0], cont.bank_used[1],
                cont.admitted_cached[0], cont.admitted_computed[0],
                cont.admitted_cached[1], cont.admitted_computed[1]);
        ds4_batch_ctx_destroy(batch_ctx);
        failed = 1;
        goto cleanup;
    }
    const int *committed = NULL;
    if (ds4_batch_ctx_bank_committed(batch_ctx, 0, &committed) != 6 ||
        !committed || committed[4] != primary_batch_oracle.tokens[0] ||
        committed[5] != primary_batch_oracle.tokens[1] ||
        ds4_batch_ctx_bank_committed(batch_ctx, 1, NULL) != prompt_alt.len ||
        ds4_batch_ctx_bank_generation(batch_ctx, 0) == 0u ||
        ds4_batch_ctx_bank_generation(batch_ctx, 1) == 0u) {
        fprintf(stderr, "Solar rolling bank history contract failed\n");
        ds4_batch_ctx_destroy(batch_ctx);
        failed = 1;
        goto cleanup;
    }
    fprintf(stderr,
            "Solar batching: scalar seeds=%d/%d batch=%d/%d "
            "rolling=%d,%d,%d + %d\n",
            next, alt_next,
            primary_batch_oracle.tokens[0], alternate_batch_oracle.tokens[0],
            primary_batch_oracle.tokens[0], primary_batch_oracle.tokens[1],
            primary_batch_oracle.tokens[2], alternate_batch_oracle.tokens[0]);

    const uint64_t bank_bytes =
        ds4_cont_bank_payload_bytes(batch_ctx, 0);
    err[0] = '\0';
    if (bank_bytes == 0u ||
        ds4_cont_bank_stage_payload(
            batch_ctx, 0, &bank_payload, err, sizeof(err)) != 0 ||
        bank_payload.bytes != bank_bytes || !bank_payload.path) {
        fprintf(stderr, "Solar bank payload stage failed: %s\n", err);
        ds4_batch_ctx_destroy(batch_ctx);
        failed = 1;
        goto cleanup;
    }
    FILE *bank_fp = fopen(bank_payload.path, "rb");
    if (!bank_fp || ds4_cont_bank_restore_payload(
            batch_ctx, 1, bank_fp, bank_payload.bytes,
            err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar bank payload restore failed: %s\n", err);
        if (bank_fp) fclose(bank_fp);
        ds4_batch_ctx_destroy(batch_ctx);
        failed = 1;
        goto cleanup;
    }
    if (fclose(bank_fp) != 0) {
        perror("Solar bank payload close");
        ds4_batch_ctx_destroy(batch_ctx);
        failed = 1;
        goto cleanup;
    }
    ds4_session_payload_file_free(&bank_payload);

    for (int i = 0; i < prompt.len; i++)
        ds4_tokens_push(&prompt_restored, prompt.v[i]);
    ds4_tokens_push(&prompt_restored, primary_batch_oracle.tokens[0]);
    ds4_tokens_push(&prompt_restored, primary_batch_oracle.tokens[1]);
    ds4_tokens_push(&prompt_restored, primary_batch_oracle.tokens[2]);
    solar_cont_test restored = {0};
    restored.prompt[0] = &prompt_restored;
    restored.n_requests = 1;
    restored.first_cached = 6;
    restored.first_max_new = 1;
    restored.first_place_bank = 2;
    restored.user[0].test = &restored;
    restored.user[0].id = 0;
    restored.bank_used[0] = -1;
    restored.admitted_cached[0] = -1;
    restored.admitted_computed[0] = -1;
    restored.admitted_bank[0] = -1;
    err[0] = '\0';
    if (ds4_engine_continuous_generate(
            batch_ctx, solar_cont_admit, solar_cont_on_token,
            solar_cont_on_done, &restored, err, sizeof(err)) != 0 ||
        restored.failed || restored.done != 1 ||
        restored.n_tokens[0] != 1 ||
        restored.tokens[0][0] != primary_batch_oracle.tokens[3] ||
        restored.bank_used[0] != 1 ||
        restored.admitted_cached[0] != 6 ||
        restored.admitted_computed[0] != 1 ||
        restored.admitted_bank[0] != 1 ||
        ds4_batch_ctx_bank_committed(batch_ctx, 1, NULL) != 7) {
        fprintf(stderr,
                "Solar restored bank warm continuation failed: %s "
                "done=%d token=%d/%d bank=%d split=%d+%d\n",
                err, restored.done, restored.tokens[0][0],
                primary_batch_oracle.tokens[3],
                restored.bank_used[0], restored.admitted_cached[0],
                restored.admitted_computed[0]);
        ds4_batch_ctx_destroy(batch_ctx);
        failed = 1;
        goto cleanup;
    }
    fprintf(stderr,
            "Solar bank persistence: bytes=%llu cached=6 computed=1 next=%d\n",
            (unsigned long long)bank_bytes, primary_batch_oracle.tokens[3]);

    /* A warm suffix longer than one prefill chunk advances KDA/GQA state and
     * history before frontier logits exist.  If the client leaves between
     * chunks, an exact zero-suffix retry must not sample the older frontier's
     * logits: it safely cold-replays the retained prefix instead. */
    for (int i = 0; i < prompt.len; i++)
        ds4_tokens_push(&prompt_interrupted, prompt.v[i]);
    /* Two appended tokens reproduce bank0's committed 4->6 frontier; the
     * remaining 17 form the warm suffix under test (total request len 23). */
    for (int i = 0; i < 19; i++)
        ds4_tokens_push(&prompt_interrupted,
                        primary_batch_oracle.tokens[i % 6]);
    for (int i = 0; i < 22; i++)
        ds4_tokens_push(&prompt_retained, prompt_interrupted.v[i]);

    /* Derive the oracle through another bank of this same persistent context
     * so both cold paths use the same 16-token prefill chunk shape. */
    solar_cont_test retained_cold_oracle = {0};
    retained_cold_oracle.prompt[0] = &prompt_retained;
    retained_cold_oracle.n_requests = 1;
    retained_cold_oracle.first_cached = 0;
    retained_cold_oracle.first_max_new = 1;
    retained_cold_oracle.first_place_bank = 3;
    retained_cold_oracle.user[0].test = &retained_cold_oracle;
    retained_cold_oracle.user[0].id = 0;
    retained_cold_oracle.bank_used[0] = -1;
    retained_cold_oracle.admitted_cached[0] = -1;
    retained_cold_oracle.admitted_computed[0] = -1;
    retained_cold_oracle.admitted_bank[0] = -1;
    err[0] = '\0';
    if (ds4_engine_continuous_generate(
            batch_ctx, solar_cont_admit, solar_cont_on_token,
            solar_cont_on_done, &retained_cold_oracle,
            err, sizeof(err)) != 0 ||
        retained_cold_oracle.failed || retained_cold_oracle.done != 1 ||
        retained_cold_oracle.n_tokens[0] != 1 ||
        retained_cold_oracle.admitted_cached[0] != 0 ||
        retained_cold_oracle.admitted_computed[0] != 22 ||
        retained_cold_oracle.admitted_bank[0] != 2) {
        fprintf(stderr,
                "Solar retained-prefix cold oracle failed: %s "
                "done=%d split=%d+%d bank=%d\n",
                err, retained_cold_oracle.done,
                retained_cold_oracle.admitted_cached[0],
                retained_cold_oracle.admitted_computed[0],
                retained_cold_oracle.admitted_bank[0]);
        ds4_batch_ctx_destroy(batch_ctx);
        failed = 1;
        goto cleanup;
    }

    solar_cont_test interrupted = {0};
    interrupted.prompt[0] = &prompt_interrupted;
    interrupted.n_requests = 1;
    interrupted.first_cached = 6;
    interrupted.first_max_new = 1;
    interrupted.first_place_bank = 1;
    interrupted.alive_fail_after = 1; /* run one 16-token chunk, then cancel */
    interrupted.user[0].test = &interrupted;
    interrupted.user[0].id = 0;
    interrupted.bank_used[0] = -1;
    interrupted.admitted_cached[0] = -1;
    interrupted.admitted_computed[0] = -1;
    interrupted.admitted_bank[0] = -1;
    err[0] = '\0';
    if (ds4_engine_continuous_generate(
            batch_ctx, solar_cont_admit, solar_cont_on_token,
            solar_cont_on_done, &interrupted, err, sizeof(err)) != 0 ||
        interrupted.failed || interrupted.done != 1 ||
        interrupted.n_tokens[0] != 0 ||
        interrupted.alive_calls != 2 ||
        interrupted.admitted_cached[0] != 6 ||
        interrupted.admitted_computed[0] != 17 ||
        interrupted.admitted_bank[0] != 0 ||
        ds4_batch_ctx_bank_committed(batch_ctx, 0, NULL) != 22) {
        fprintf(stderr,
                "Solar interrupted warm prefill failed: %s done=%d n=%d "
                "split=%d+%d bank=%d alive=%d committed=%d\n",
                err, interrupted.done, interrupted.n_tokens[0],
                interrupted.admitted_cached[0],
                interrupted.admitted_computed[0],
                interrupted.admitted_bank[0],
                interrupted.alive_calls,
                ds4_batch_ctx_bank_committed(batch_ctx, 0, NULL));
        ds4_batch_ctx_destroy(batch_ctx);
        failed = 1;
        goto cleanup;
    }

    solar_cont_test retained_retry = {0};
    retained_retry.prompt[0] = &prompt_retained;
    retained_retry.n_requests = 1;
    retained_retry.first_cached = 22;
    retained_retry.first_max_new = 1;
    retained_retry.first_place_bank = 1;
    retained_retry.user[0].test = &retained_retry;
    retained_retry.user[0].id = 0;
    retained_retry.bank_used[0] = -1;
    retained_retry.admitted_cached[0] = -1;
    retained_retry.admitted_computed[0] = -1;
    retained_retry.admitted_bank[0] = -1;
    err[0] = '\0';
    if (ds4_engine_continuous_generate(
            batch_ctx, solar_cont_admit, solar_cont_on_token,
            solar_cont_on_done, &retained_retry, err, sizeof(err)) != 0 ||
        retained_retry.failed || retained_retry.done != 1 ||
        retained_retry.n_tokens[0] != 1 ||
        retained_retry.tokens[0][0] != retained_cold_oracle.tokens[0][0] ||
        retained_retry.admitted_cached[0] != 0 ||
        retained_retry.admitted_computed[0] != 22 ||
        retained_retry.admitted_bank[0] != 0) {
        fprintf(stderr,
                "Solar interrupted-prefix cold replay failed: %s "
                "done=%d token=%d/%d split=%d+%d bank=%d\n",
                err, retained_retry.done, retained_retry.tokens[0][0],
                retained_cold_oracle.tokens[0][0],
                retained_retry.admitted_cached[0],
                retained_retry.admitted_computed[0],
                retained_retry.admitted_bank[0]);
        ds4_batch_ctx_destroy(batch_ctx);
        failed = 1;
        goto cleanup;
    }
    fprintf(stderr,
            "Solar interrupted prefill: retained=22 retry=cold token=%d\n",
            retained_cold_oracle.tokens[0][0]);

    /* Aborting from the first sampled-token callback is different: that
     * token has not been forwarded into KDA/GQA state or bank history yet,
     * and the final-prompt logits are current.  The identical retry must keep
     * the exact-frontier fast path and sample the same first token. */
    solar_cont_test first_token_abort = {0};
    int forced_abort_token = retained_cold_oracle.tokens[0][0];
    if (forced_abort_token == solar_eos)
        forced_abort_token = (solar_eos + 1) % ds4_engine_vocab_size(engine);
    first_token_abort.prompt[0] = &prompt_retained;
    first_token_abort.n_requests = 1;
    first_token_abort.first_cached = 22;
    first_token_abort.first_max_new = 2;
    first_token_abort.first_place_bank = 1;
    first_token_abort.abort_first_token = 1;
    first_token_abort.sample_override =
        DS4_SAMPLE_OVERRIDE_TOKEN(forced_abort_token);
    first_token_abort.user[0].test = &first_token_abort;
    first_token_abort.user[0].id = 0;
    first_token_abort.bank_used[0] = -1;
    first_token_abort.admitted_cached[0] = -1;
    first_token_abort.admitted_computed[0] = -1;
    first_token_abort.admitted_bank[0] = -1;
    err[0] = '\0';
    if (ds4_engine_continuous_generate(
            batch_ctx, solar_cont_admit, solar_cont_on_token,
            solar_cont_on_done, &first_token_abort, err, sizeof(err)) != 0 ||
        first_token_abort.failed || first_token_abort.done != 1 ||
        first_token_abort.on_token_calls != 1 ||
        first_token_abort.n_tokens[0] != 1 ||
        first_token_abort.tokens[0][0] != forced_abort_token ||
        first_token_abort.admitted_cached[0] != 22 ||
        first_token_abort.admitted_computed[0] != 0 ||
        ds4_batch_ctx_bank_committed(batch_ctx, 0, NULL) != 22) {
        fprintf(stderr,
                "Solar first-token abort frontier failed: %s done=%d "
                "token=%d/%d calls=%d split=%d+%d committed=%d\n",
                err, first_token_abort.done,
                first_token_abort.tokens[0][0],
                forced_abort_token, first_token_abort.on_token_calls,
                first_token_abort.admitted_cached[0],
                first_token_abort.admitted_computed[0],
                ds4_batch_ctx_bank_committed(batch_ctx, 0, NULL));
        ds4_batch_ctx_destroy(batch_ctx);
        failed = 1;
        goto cleanup;
    }

    solar_cont_test first_token_retry = {0};
    first_token_retry.prompt[0] = &prompt_retained;
    first_token_retry.n_requests = 1;
    first_token_retry.first_cached = 22;
    first_token_retry.first_max_new = 1;
    first_token_retry.first_place_bank = 1;
    first_token_retry.sample_override =
        DS4_SAMPLE_OVERRIDE_TOKEN(forced_abort_token);
    first_token_retry.user[0].test = &first_token_retry;
    first_token_retry.user[0].id = 0;
    first_token_retry.bank_used[0] = -1;
    first_token_retry.admitted_cached[0] = -1;
    first_token_retry.admitted_computed[0] = -1;
    first_token_retry.admitted_bank[0] = -1;
    err[0] = '\0';
    if (ds4_engine_continuous_generate(
            batch_ctx, solar_cont_admit, solar_cont_on_token,
            solar_cont_on_done, &first_token_retry, err, sizeof(err)) != 0 ||
        first_token_retry.failed || first_token_retry.done != 1 ||
        first_token_retry.n_tokens[0] != 1 ||
        first_token_retry.tokens[0][0] != forced_abort_token ||
        first_token_retry.admitted_cached[0] != 22 ||
        first_token_retry.admitted_computed[0] != 0) {
        fprintf(stderr,
                "Solar first-token retry warm reuse failed: %s done=%d "
                "token=%d/%d split=%d+%d\n",
                err, first_token_retry.done,
                first_token_retry.tokens[0][0],
                forced_abort_token,
                first_token_retry.admitted_cached[0],
                first_token_retry.admitted_computed[0]);
        ds4_batch_ctx_destroy(batch_ctx);
        failed = 1;
        goto cleanup;
    }
    fprintf(stderr,
            "Solar first-token abort: retained=22 retry=cached token=%d\n",
            forced_abort_token);

    /* Ramp through 2/3/4-row decode passes.  Chunked admission staggers the
     * four identical banks by one step; every final sequence must still be
     * identical to the independent one-bank batch trajectory. */
    ds4_tokens fused_prompts[4] = { prompt, prompt, prompt, prompt };
    ds4_batch_gen_result fused_results[4] = {0};
    const int six_tokens[4] = {6, 6, 6, 6};
    const int fused_eos[4] = {
        ds4_token_eos(engine), ds4_token_eos(engine),
        ds4_token_eos(engine), ds4_token_eos(engine),
    };
    err[0] = '\0';
    const int fused_rc = ds4_engine_batched_generate_ctx(
        batch_ctx, fused_prompts, 4, six_tokens, fused_eos,
        fused_results, err, sizeof(err));
    int fused_ok = fused_rc == 0;
    for (int i = 0; fused_ok && i < 4; i++) {
        fused_ok = fused_results[i].n_tokens == 6 &&
                   memcmp(fused_results[i].tokens,
                          primary_batch_oracle.tokens,
                          6u * sizeof(int)) == 0 &&
                   (i == 0 || memcmp(
                       fused_results[0].tokens, fused_results[i].tokens,
                       6u * sizeof(int)) == 0);
    }
    if (!fused_ok) {
        fprintf(stderr,
                "Solar fused four-bank decode failed: %s "
                "n=%d/%d/%d/%d first=%d,%d,%d\n",
                err, fused_results[0].n_tokens, fused_results[1].n_tokens,
                fused_results[2].n_tokens, fused_results[3].n_tokens,
                fused_results[0].n_tokens > 0 ? fused_results[0].tokens[0] : -1,
                fused_results[0].n_tokens > 1 ? fused_results[0].tokens[1] : -1,
                fused_results[0].n_tokens > 2 ? fused_results[0].tokens[2] : -1);
        for (int i = 0; i < 4; i++) free(fused_results[i].tokens);
        ds4_batch_ctx_destroy(batch_ctx);
        failed = 1;
        goto cleanup;
    }
    fprintf(stderr,
            "Solar fused decode: banks=4 tokens=%d,%d,%d,...\n",
            primary_batch_oracle.tokens[0], primary_batch_oracle.tokens[1],
            primary_batch_oracle.tokens[2]);
    for (int i = 0; i < 4; i++) free(fused_results[i].tokens);

    /* Final idle-context trim exercises the Solar-specific VMM path.  The
     * implementation must synchronize before unmapping any state/KV slab;
     * after successful release at least one bank generation changes so a
     * stale warm directive cannot survive the destructive reclaim. */
    uint64_t trim_gen_before[4];
    for (int i = 0; i < 4; i++)
        trim_gen_before[i] = ds4_batch_ctx_bank_generation(batch_ctx, i);
    const uint64_t trimmed_bytes =
        ds4_batch_ctx_trim_free(batch_ctx, UINT64_MAX);
    int trim_gen_changed = 0;
    for (int i = 0; i < 4; i++) {
        if (ds4_batch_ctx_bank_generation(batch_ctx, i) != trim_gen_before[i])
            trim_gen_changed++;
    }
    if (trimmed_bytes == 0 || trim_gen_changed == 0) {
        fprintf(stderr,
                "Solar idle-bank trim failed: bytes=%llu generations=%d\n",
                (unsigned long long)trimmed_bytes, trim_gen_changed);
        ds4_batch_ctx_destroy(batch_ctx);
        failed = 1;
        goto cleanup;
    }
    fprintf(stderr,
            "Solar idle-bank trim: %.1f MiB generations=%d\n",
            (double)trimmed_bytes / 1048576.0, trim_gen_changed);
    ds4_batch_ctx_destroy(batch_ctx);

    ds4_batch_gen_result batch_result = {0};
    err[0] = '\0';
    if (ds4_engine_batched_generate_ex(
            engine, &prompt, 1, 128, &one_token_single, &solar_eos,
            &batch_result, err, sizeof(err)) != 0 ||
        batch_result.n_tokens != 1 ||
        batch_result.tokens[0] != primary_batch_oracle.tokens[0]) {
        fprintf(stderr, "Solar per-call batch path failed: %s\n", err);
        free(batch_result.tokens);
        failed = 1;
        goto cleanup;
    }
    free(batch_result.tokens);

    int accepted = -1;
    err[0] = '\0';
    if (ds4_session_eval_speculative_argmax(
            session, next, 4, solar_eos, &accepted, 1,
            err, sizeof(err)) != 1 ||
        accepted != next || ds4_session_pos(session) != prompt.len + 1) {
        fprintf(stderr, "Solar speculative API fallback failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    generation_capture capture = {0};
    if (ds4_engine_generate_argmax(
            engine, &prompt, 1, 128,
            capture_token, capture_done, &capture,
            NULL, NULL) != 0 ||
        capture.emitted != 1 || capture.token != next || capture.done != 1) {
        fprintf(stderr,
                "Solar public engine generation failed: emitted=%d token=%d "
                "want=%d done=%d\n",
                capture.emitted, capture.token, next, capture.done);
        failed = 1;
        goto cleanup;
    }

cleanup:
    free(alternate_batch_oracle.tokens);
    free(primary_batch_oracle.tokens);
    ds4_session_payload_file_free(&bank_payload);
    ds4_session_snapshot_free(&restored_snapshot);
    ds4_session_snapshot_free(&snapshot);
    free(replayed);
    free(warm);
    free(decoded);
    free(initial);
    ds4_session_free(session);
    ds4_tokens_free(&prompt_restored);
    ds4_tokens_free(&prompt_retained);
    ds4_tokens_free(&prompt_interrupted);
    ds4_tokens_free(&prompt_alt);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    if (partial_only) {
        puts(failed ? "Solar partial reuse checkpoint gate FAILED"
                    : "Solar partial reuse checkpoint gate passed");
    } else if (disk_kv_only) {
        puts(failed ? "Solar disk-KV restart reuse FAILED"
                    : "Solar disk-KV restart reuse passed");
    } else {
        puts(failed ? "Solar public session lifecycle FAILED"
                    : "Solar public session lifecycle passed");
    }
    return failed ? 1 : 0;
}
