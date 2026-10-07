/* Actual selective-attention and durable continuation gate. Input IDs come
 * from the metadata-only tokenizer and an independently rendered fixture. */
#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { CTX_8K = 8192, INPUT_8K = 6147, CTX_1M = 1048576, FIRST = 2048, CHECKPOINT = 2049,
       OUTPUT_TOKENS = 64, IO_BYTES = 65536, HOST_THREADS = 8 };
enum long_case { LONG_8K, LONG_1M };
enum long_weights { LONG_SSD, LONG_RESIDENT };
enum decode_pass { ANSWER_PASS, REPLAY_PASS };
static const uint64_t CACHE_BYTES = UINT64_C(24) << 30;
static char error[512];
static const char *output_dir;
static int same_files(FILE *a, uint64_t an, FILE *b, uint64_t bn);

#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr, "GLM long FAIL %d: %s (%s)\n", __LINE__, #x, error); \
    goto cleanup; \
} } while (0)

static double now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { return 0.0; }
    return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}

static void progress(void *ud, const char *event, int current, int total) {
    if (strcmp(event, "prefill_chunk")) { return; }
    fprintf(stderr, "GLM long frontier: %d/%d elapsed_s=%.6f\n",
        current, total, now() - *(double *)ud);
}

static FILE *artifact(const char *name) {
    char path[4096];
    const int n = snprintf(path, sizeof(path), "%s/%s", output_dir, name);
    if (n < 0 || n >= (int)sizeof(path)) { return NULL; }
    return fopen(path, "wb+");
}

static int same_artifacts(const char *left, const char *right) {
    char paths[2][4096];
    const char *names[2] = {left, right};
    FILE *files[2] = {NULL, NULL};
    uint64_t bytes[2] = {0, 0};
    int ok = 1;
    for (unsigned i = 0; i < 2; i++) {
        const int n = snprintf(paths[i], sizeof(paths[i]), "%s/%s", output_dir, names[i]);
        if (n < 0 || n >= (int)sizeof(paths[i])) { ok = 0; break; }
        files[i] = fopen(paths[i], "rb");
        if (!files[i] || fseeko(files[i], 0, SEEK_END) || ftello(files[i]) < 0) { ok = 0; break; }
        bytes[i] = (uint64_t)ftello(files[i]);
    }
    if (ok) { ok = same_files(files[0], bytes[0], files[1], bytes[1]); }
    for (unsigned i = 0; i < 2; i++) {
        if (files[i] && fclose(files[i])) { ok = 0; }
    }
    return ok;
}

static int finite_logits(ds4_session *s, float *out, int vocab) {
    if (ds4_session_copy_logits(s, out, vocab) != vocab) { return 0; }
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(out[i])) { return 0; }
    }
    return 1;
}

static int same_history(ds4_session *s, const ds4_tokens *prompt) {
    const ds4_tokens *history = ds4_session_tokens(s);
    return history && history->len == prompt->len &&
        ds4_session_pos(s) == prompt->len &&
        !memcmp(history->v, prompt->v, (size_t)prompt->len * sizeof(int));
}

static uint64_t save(ds4_session *s, FILE *fp) {
    const uint64_t bytes = ds4_session_payload_bytes(s);
    if (!bytes || ds4_session_save_payload(s, fp, error, sizeof(error)) ||
        fflush(fp) || ftello(fp) < 0 || (uint64_t)ftello(fp) != bytes) { return 0u; }
    return bytes;
}

static int same_files(FILE *a, uint64_t an, FILE *b, uint64_t bn) {
    if (an != bn || fseeko(a, 0, SEEK_SET) || fseeko(b, 0, SEEK_SET)) { return 0; }
    unsigned char left[IO_BYTES], right[IO_BYTES];
    for (uint64_t off = 0; off < an;) {
        const size_t n = an - off < IO_BYTES ? (size_t)(an - off) : IO_BYTES;
        if (fread(left, 1u, n, a) != n || fread(right, 1u, n, b) != n) { return 0; }
        if (memcmp(left, right, n)) {
            for (size_t i = 0; i < n; i++) {
                if (left[i] == right[i]) { continue; }
                fprintf(stderr, "GLM long payload first mismatch at %" PRIu64 "\n", off + i);
                break;
            }
            return 0;
        }
        off += n;
    }
    return 1;
}

static int sync_prefix(ds4_session *s, ds4_tokens *prompt, int n,
                       float *logits, int vocab) {
    ds4_tokens prefix = *prompt;
    prefix.len = n;
    const double start = now();
    fprintf(stderr, "GLM long start: committed=%d target=%d\n", ds4_session_pos(s), n);
    const int ok = !ds4_session_sync(s, &prefix, error, sizeof(error)) &&
        same_history(s, &prefix) && finite_logits(s, logits, vocab);
    fprintf(stderr, "GLM long phase: target=%d committed=%d wall_s=%.6f result=%s\n",
        n, ds4_session_pos(s), now() - start, ok ? "PASS" : "FAIL");
    return ok;
}

static int generate(ds4_engine *e, ds4_session *s, enum decode_pass pass) {
    const char *text_name = pass == ANSWER_PASS ? "answer.txt" : "replay-answer.txt";
    const char *ids_name = pass == ANSWER_PASS ? "answer.tokens.txt" : "replay-answer.tokens.txt";
    FILE *text = artifact(text_name), *ids = artifact(ids_name);
    if (!text || !ids) {
        if (text) { fclose(text); }
        if (ids) { fclose(ids); }
        return 0;
    }
    uint64_t rng = 424242u;
    int ok = 1, count = 0;
    const double start = now();
    for (int i = 0; i < OUTPUT_TOKENS; i++) {
        const int t = ds4_session_sample(s, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (t < 0 || fprintf(ids, "%d\n", t) < 0) { ok = 0; break; }
        count++;
        if (ds4_token_is_stop(e, t)) { break; }
        size_t n = 0;
        char *piece = ds4_token_text(e, t, &n);
        const int written = piece && fwrite(piece, 1u, n, text) == n;
        free(piece);
        if (!written || ds4_session_eval(s, t, error, sizeof(error))) { ok = 0; break; }
    }
    if (fclose(text)) { ok = 0; }
    if (fclose(ids)) { ok = 0; }
    fprintf(stderr, "GLM long decode: tokens=%d wall_s=%.6f result=%s\n",
        count, now() - start, ok ? "PASS" : "FAIL");
    return ok && count > 0;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        fprintf(stderr, "usage: %s MODEL.gguf TOKENS.i32 OUTPUT_DIR [--1m] [--resident]\n", argv[0]);
        return 2;
    }
    enum long_case mode = LONG_8K;
    enum long_weights weights = LONG_SSD;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--1m") && mode == LONG_8K) { mode = LONG_1M; continue; }
        if (!strcmp(argv[i], "--resident") && weights == LONG_SSD) { weights = LONG_RESIDENT; continue; }
        fprintf(stderr, "unknown or repeated option: %s\n", argv[i]);
        return 2;
    }
    // Resident qualification borrows one owner, never another huge weight copy.
    const char *manifest = getenv("DS4_CUDA_WEIGHT_IPC_MANIFEST");
    if (weights == LONG_RESIDENT && (!manifest || !*manifest)) {
        fprintf(stderr, "--resident requires a weight owner manifest\n");
        return 2;
    }
    const int ctx = mode == LONG_1M ? CTX_1M : CTX_8K;
    const int input_tokens = mode == LONG_1M ? CTX_1M - OUTPUT_TOKENS : INPUT_8K;
    output_dir = argv[3];
    ds4_engine *e = NULL;
    ds4_session *s = NULL;
    ds4_tokens prompt = {0};
    FILE *input = NULL, *checkpoint = NULL, *restored = NULL, *full = NULL, *replay = NULL;
    FILE *generated = NULL;
    float *logits = NULL, *reference = NULL;
    int failed = 1;

    input = fopen(argv[2], "rb");
    REQUIRE(input && !fseeko(input, 0, SEEK_END) &&
        ftello(input) == (off_t)(input_tokens * sizeof(int32_t)) && !fseeko(input, 0, SEEK_SET));
    prompt.v = malloc((size_t)input_tokens * sizeof(*prompt.v));
    prompt.len = prompt.cap = input_tokens;
    REQUIRE(prompt.v && fread(prompt.v, sizeof(*prompt.v), (size_t)input_tokens, input) == (size_t)input_tokens);
    const int close_error = fclose(input);
    input = NULL;
    REQUIRE(!close_error);

    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_CUDA,
        .n_threads = HOST_THREADS, .defer_boot_prewarm = true, .mtp_draft_tokens = 0,
        .ssd_streaming = weights == LONG_SSD,
        .ssd_streaming_cache_bytes = weights == LONG_SSD ? CACHE_BYTES : 0u};
    REQUIRE(!ds4_engine_open(&e, &opt) && ds4_engine_layer_count(e) == 45);
    REQUIRE(!ds4_session_create(&s, e, ctx));
    double start = now();
    ds4_session_set_progress(s, progress, &start);
    const int vocab = ds4_engine_vocab_size(e);
    REQUIRE(vocab > 0);
    logits = malloc((size_t)vocab * sizeof(*logits));
    reference = malloc((size_t)vocab * sizeof(*reference));
    REQUIRE(logits && reference);
    REQUIRE(sync_prefix(s, &prompt, FIRST, logits, vocab));
    REQUIRE(sync_prefix(s, &prompt, CHECKPOINT, logits, vocab));
    checkpoint = artifact("checkpoint-2049.payload.bin");
    REQUIRE(checkpoint);
    const uint64_t checkpoint_bytes = save(s, checkpoint);
    REQUIRE(checkpoint_bytes);

    REQUIRE(sync_prefix(s, &prompt, input_tokens, reference, vocab));
    full = artifact(mode == LONG_1M ? "full-1048512.payload.bin" : "full-6147.payload.bin");
    REQUIRE(full);
    const uint64_t full_bytes = save(s, full);
    REQUIRE(full_bytes);
    if (mode == LONG_1M) {
        /* One full prefill is sufficient. Replay only generated tokens after
         * restoring the complete FILE frontier; avoid a second 1M prefill. */
        REQUIRE(generate(e, s, ANSWER_PASS) && finite_logits(s, reference, vocab));
        generated = artifact("generated.payload.bin");
        REQUIRE(generated);
        const uint64_t generated_bytes = save(s, generated);
        REQUIRE(generated_bytes && !fseeko(full, 0, SEEK_SET) &&
            !ds4_session_load_payload(s, full, full_bytes, error, sizeof(error)));
        restored = artifact("restored-1048512.payload.bin");
        REQUIRE(restored);
        const uint64_t restored_bytes = save(s, restored);
        REQUIRE(restored_bytes && same_files(full, full_bytes, restored, restored_bytes));
        REQUIRE(generate(e, s, REPLAY_PASS) && finite_logits(s, logits, vocab));
        replay = artifact("replayed-generation.payload.bin");
        REQUIRE(replay);
        const uint64_t replay_bytes = save(s, replay);
        REQUIRE(replay_bytes && same_files(generated, generated_bytes, replay, replay_bytes) &&
            !memcmp(reference, logits, (size_t)vocab * sizeof(float)) &&
            same_artifacts("answer.txt", "replay-answer.txt") &&
            same_artifacts("answer.tokens.txt", "replay-answer.tokens.txt"));
        fprintf(stderr, "GLM long full-context restore/decode: byte exact; input=%d ctx=%d\n",
            input_tokens, ctx);
        goto verified;
    }

    REQUIRE(!fseeko(checkpoint, 0, SEEK_SET) &&
        !ds4_session_load_payload(s, checkpoint, checkpoint_bytes, error, sizeof(error)));
    restored = artifact("restored-2049.payload.bin");
    REQUIRE(restored);
    const uint64_t restored_bytes = save(s, restored);
    REQUIRE(restored_bytes && same_files(checkpoint, checkpoint_bytes, restored, restored_bytes));
    REQUIRE(sync_prefix(s, &prompt, input_tokens, logits, vocab));
    replay = artifact("replay-6147.payload.bin");
    REQUIRE(replay);
    const uint64_t replay_bytes = save(s, replay);
    REQUIRE(replay_bytes && same_files(full, full_bytes, replay, replay_bytes) &&
        !memcmp(reference, logits, (size_t)vocab * sizeof(float)));
    fprintf(stderr, "GLM long restore/continuation: byte exact; checkpoint=%" PRIu64
        " full=%" PRIu64 "\n", checkpoint_bytes, full_bytes);
    REQUIRE(generate(e, s, ANSWER_PASS));
verified:
    REQUIRE(finite_logits(s, logits, vocab) && !ds4_gpu_mem_census_faults() &&
        !ds4_metrics_get()->memgov_faults);
    fprintf(stderr, "GLM long structural gate: PASS; retrieval answer judged separately\n");
    failed = 0;
cleanup:
    if (input) { fclose(input); }
    if (checkpoint) { fclose(checkpoint); }
    if (restored) { fclose(restored); }
    if (full) { fclose(full); }
    if (replay) { fclose(replay); }
    if (generated) { fclose(generated); }
    free(logits); free(reference);
    ds4_tokens_free(&prompt);
    ds4_session_free(s);
    ds4_engine_close(e);
    return failed;
}
