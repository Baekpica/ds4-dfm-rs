/* Actual-artifact prefix invariant, not draft acceptance or answer quality.
 * One session replays identical candidates through trial/commit and ordinary
 * width-one eval. FILE payloads include KDA, latent pools and MTP frontier. */
#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TEST_CTX = 2048, TRIAL_ROWS = 4, IO_BYTES = 65536 };
static const uint64_t CACHE_BYTES = UINT64_C(24) << 30;
static char error[512];
static const char *output_dir;

#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr, "GLM MTP actual FAIL %d: %s (%s)\n", __LINE__, #x, error); \
    goto cleanup; \
} } while (0)

static FILE *open_stage(const char *stage, const char *suffix) {
    char path[4096];
    const int n = snprintf(path, sizeof(path), "%s/%s.%s", output_dir, stage, suffix);
    if (n < 0 || n >= (int)sizeof(path)) { return NULL; }
    return fopen(path, "wb+");
}

static int close_stage(FILE **stage) {
    FILE *fp = *stage;
    *stage = NULL;
    return !fclose(fp);
}

static char *read_prompt(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp || fseeko(fp, 0, SEEK_END)) {
        if (fp) { fclose(fp); }
        return NULL;
    }
    const off_t n = ftello(fp);
    if (n <= 0 || n > IO_BYTES || fseeko(fp, 0, SEEK_SET)) {
        fclose(fp);
        return NULL;
    }
    char *text = malloc((size_t)n + 1u);
    if (!text || fread(text, 1u, (size_t)n, fp) != (size_t)n) {
        free(text);
        fclose(fp);
        return NULL;
    }
    text[n] = 0;
    fclose(fp);
    return text;
}

static int read_logits(ds4_session *s, float *logits, int vocab) {
    if (ds4_session_copy_logits(s, logits, vocab) != vocab) { return 0; }
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(logits[i])) { return 0; }
    }
    return 1;
}

static uint64_t save_payload(ds4_session *s, FILE *fp) {
    const uint64_t bytes = ds4_session_payload_bytes(s);
    if (!bytes || ds4_session_save_payload(s, fp, error, sizeof(error)) ||
        fflush(fp) || ftello(fp) < 0 || (uint64_t)ftello(fp) != bytes) { return 0u; }
    return bytes;
}

static int load_payload(ds4_session *s, FILE *fp, uint64_t bytes) {
    if (fseeko(fp, 0, SEEK_SET)) { return 0; }
    return !ds4_session_load_payload(s, fp, bytes, error, sizeof(error));
}

static int same_files(FILE *a, uint64_t an, FILE *b, uint64_t bn) {
    if (an != bn || fseeko(a, 0, SEEK_SET) || fseeko(b, 0, SEEK_SET)) { return 0; }
    unsigned char left[IO_BYTES], right[IO_BYTES];
    for (uint64_t off = 0u; off < an;) {
        const size_t n = an - off < IO_BYTES ? (size_t)(an - off) : IO_BYTES;
        if (fread(left, 1u, n, a) != n || fread(right, 1u, n, b) != n) { return 0; }
        if (memcmp(left, right, n)) {
            for (size_t i = 0u; i < n; i++) {
                if (left[i] != right[i]) {
                    fprintf(stderr, "GLM MTP payload first difference at %" PRIu64
                        ": %u/%u\n", off + i, left[i], right[i]);
                    break;
                }
            }
            return 0;
        }
        off += n;
    }
    return 1;
}

static int save_logits(const char *stage, const float *logits, int vocab) {
    FILE *fp = open_stage(stage, "logits.f32");
    if (!fp) { return 0; }
    const int ok = fwrite(logits, sizeof(*logits), (size_t)vocab, fp) == (size_t)vocab;
    return !fclose(fp) && ok;
}

static int same_history(ds4_session *s, const ds4_tokens *prompt,
                         const int *candidate, int keep) {
    const ds4_tokens *history = ds4_session_tokens(s);
    return history && history->len == prompt->len + keep &&
        ds4_session_pos(s) == history->len &&
        !memcmp(history->v, prompt->v, (size_t)prompt->len * sizeof(int)) &&
        !memcmp(history->v + prompt->len, candidate, (size_t)keep * sizeof(int));
}

static int save_candidates(int keep, const int *candidate, const int *target) {
    char stage[32];
    snprintf(stage, sizeof(stage), "keep%d", keep);
    FILE *fp = open_stage(stage, "tokens.txt");
    if (!fp) { return 0; }
    int ok = 1;
    for (int i = 0; i < TRIAL_ROWS; i++) {
        if (fprintf(fp, "%d %d %d\n", i, candidate[i], target[i]) < 0) { ok = 0; }
    }
    return !fclose(fp) && ok;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL.gguf RENDERED_PROMPT.txt OUTPUT_DIR\n", argv[0]);
        return 2;
    }
    output_dir = argv[3];
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_CUDA,
        .n_threads = 8, .defer_boot_prewarm = true, .mtp_draft_tokens = 3,
        .ssd_streaming = true, .ssd_streaming_cache_bytes = CACHE_BYTES};
    ds4_engine *e = NULL;
    ds4_session *s = NULL;
    ds4_tokens prompt = {0};
    FILE *baseline = NULL, *trial_file = NULL, *eval_file = NULL;
    char *text = NULL;
    float *base_logits = NULL, *trial_logits = NULL, *eval_logits = NULL;
    int failed = 1, candidate[TRIAL_ROWS], target[TRIAL_ROWS];
    REQUIRE(!ds4_engine_open(&e, &opt) && ds4_engine_layer_count(e) == 45 && ds4_engine_has_mtp(e));
    REQUIRE(!ds4_session_create(&s, e, TEST_CTX));
    text = read_prompt(argv[2]);
    REQUIRE(text);
    /* The official template has already rendered these bytes. Preserve its
     * literal special tokens; do not add a second chat wrapper. */
    ds4_tokenize_rendered_chat(e, text, &prompt);
    REQUIRE(prompt.len > 0 && prompt.len + TRIAL_ROWS < TEST_CTX);
    REQUIRE(!ds4_session_sync(s, &prompt, error, sizeof(error)));
    const int vocab = ds4_engine_vocab_size(e);
    REQUIRE(vocab > 0);
    base_logits = malloc((size_t)vocab * sizeof(float));
    trial_logits = malloc((size_t)vocab * sizeof(float));
    eval_logits = malloc((size_t)vocab * sizeof(float));
    REQUIRE(base_logits && trial_logits && eval_logits && read_logits(s, base_logits, vocab));
    baseline = open_stage("baseline", "payload.bin");
    REQUIRE(baseline);
    const uint64_t base_bytes = save_payload(s, baseline);
    REQUIRE(base_bytes && save_logits("baseline", base_logits, vocab));
    uint64_t rng = 424242u;
    const int first = ds4_session_sample(s, 0.0f, 0, 1.0f, 0.0f, &rng);
    REQUIRE(first >= 0 && first < vocab);
    fprintf(stderr, "GLM MTP actual: prompt=%d ctx=%d prefill_cap=%d first=%d "
        "cache=%" PRIu64 " baseline=%" PRIu64 "\n", prompt.len, TEST_CTX,
        ds4_session_prefill_cap(s), first, CACHE_BYTES, base_bytes);

    REQUIRE(ds4_session_glm53_trial(s, first, TRIAL_ROWS, candidate, target,
        TRIAL_ROWS, error, sizeof(error)) == TRIAL_ROWS && candidate[0] == first);
    REQUIRE(!ds4_session_glm53_commit(s, 0, error, sizeof(error)));
    trial_file = open_stage("abort", "payload.bin");
    REQUIRE(trial_file && same_history(s, &prompt, candidate, 0));
    const uint64_t abort_bytes = save_payload(s, trial_file);
    REQUIRE(read_logits(s, trial_logits, vocab) &&
        !memcmp(base_logits, trial_logits, (size_t)vocab * sizeof(float)) &&
        same_files(baseline, base_bytes, trial_file, abort_bytes));
    REQUIRE(close_stage(&trial_file));
    fprintf(stderr, "GLM MTP abort: payload/logits/history byte exact\n");

    for (int keep = 1; keep <= TRIAL_ROWS; keep++) {
        REQUIRE(load_payload(s, baseline, base_bytes));
        REQUIRE(ds4_session_glm53_trial(s, first, TRIAL_ROWS, candidate, target,
            TRIAL_ROWS, error, sizeof(error)) == TRIAL_ROWS && candidate[0] == first);
        REQUIRE(save_candidates(keep, candidate, target));
        REQUIRE(!ds4_session_glm53_commit(s, keep, error, sizeof(error)) &&
            same_history(s, &prompt, candidate, keep) && read_logits(s, trial_logits, vocab));
        char trial_stage[32], eval_stage[32];
        snprintf(trial_stage, sizeof(trial_stage), "keep%d-trial", keep);
        snprintf(eval_stage, sizeof(eval_stage), "keep%d-eval", keep);
        trial_file = open_stage(trial_stage, "payload.bin");
        REQUIRE(trial_file);
        const uint64_t trial_bytes = save_payload(s, trial_file);
        REQUIRE(trial_bytes && save_logits(trial_stage, trial_logits, vocab));

        REQUIRE(load_payload(s, baseline, base_bytes));
        for (int i = 0; i < keep; i++) {
            REQUIRE(!ds4_session_eval(s, candidate[i], error, sizeof(error)));
        }
        REQUIRE(same_history(s, &prompt, candidate, keep) && read_logits(s, eval_logits, vocab));
        eval_file = open_stage(eval_stage, "payload.bin");
        REQUIRE(eval_file);
        const uint64_t eval_bytes = save_payload(s, eval_file);
        REQUIRE(eval_bytes && save_logits(eval_stage, eval_logits, vocab) &&
            same_files(trial_file, trial_bytes, eval_file, eval_bytes) &&
            !memcmp(trial_logits, eval_logits, (size_t)vocab * sizeof(float)));
        REQUIRE(close_stage(&eval_file));
        /* A tool close can consume fewer rows than MTP already committed.
         * Restore that journal lane without replaying the recurrent trunk. */
        REQUIRE(load_payload(s, baseline, base_bytes));
        REQUIRE(ds4_session_glm53_trial(s, first, TRIAL_ROWS, candidate, target,
            TRIAL_ROWS, error, sizeof(error)) == TRIAL_ROWS);
        REQUIRE(!ds4_session_glm53_commit(s, TRIAL_ROWS, error, sizeof(error)));
        ds4_session_rewind(s, prompt.len + keep);
        REQUIRE(same_history(s, &prompt, candidate, keep) && read_logits(s, eval_logits, vocab));
        snprintf(eval_stage, sizeof(eval_stage), "keep%d-rewind", keep);
        eval_file = open_stage(eval_stage, "payload.bin");
        REQUIRE(eval_file);
        const uint64_t rewind_bytes = save_payload(s, eval_file);
        REQUIRE(rewind_bytes && same_files(trial_file, trial_bytes, eval_file, rewind_bytes) &&
            !memcmp(trial_logits, eval_logits, (size_t)vocab * sizeof(float)));
        REQUIRE(close_stage(&trial_file));
        REQUIRE(close_stage(&eval_file));
        fprintf(stderr, "GLM MTP keep%d: identical-input width-one "
            "payload/logits/history and journal rewind byte exact (%" PRIu64 " bytes)\n", keep, trial_bytes);
    }
    /* Outside the journal, a rewind has no valid recurrent state or logits. */
    ds4_session_rewind(s, prompt.len - 1);
    REQUIRE(!ds4_session_payload_bytes(s));
    REQUIRE(ds4_session_argmax(s) == -1 && ds4_session_copy_logits(s, eval_logits, vocab) == 0);
    REQUIRE(!ds4_gpu_mem_census_faults() && !ds4_metrics_get()->memgov_faults);
    fprintf(stderr, "GLM MTP actual prefix invariant: PASS; accepted-token/answer gate separate\n");
    failed = 0;
cleanup:
    if (baseline) { fclose(baseline); }
    if (trial_file) { fclose(trial_file); }
    if (eval_file) { fclose(eval_file); }
    free(text); free(base_logits); free(trial_logits); free(eval_logits);
    ds4_tokens_free(&prompt);
    ds4_session_free(s);
    ds4_engine_close(e);
    return failed;
}
