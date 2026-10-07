/* Compare complete long-context MTP state without a second model or a RAM
 * snapshot. Two disk payloads alternate as the accepted-prefix checkpoint. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { IO_BYTES = 65536, TRIAL_ROWS = 4, OUTPUT_TOKENS = 16,
       HEADER_WORDS = 13, MAX_CTX = 1048576, MULTI_MIN = 2 };
typedef enum { STREAM_EXACT, STREAM_MUTATED, STREAM_SHORT, STREAM_EXTRA } stream_case;

typedef struct {
    FILE *expected;
    uint64_t offset;
    int failed;
} compare_stream;

static ssize_t compare_write(void *cookie, const char *data, size_t len) {
    compare_stream *state = cookie;
    unsigned char buffer[IO_BYTES];
    if (state->failed) { return 0; }
    for (size_t off = 0; off < len;) {
        const size_t n = len - off < IO_BYTES ? len - off : IO_BYTES;
        if (fread(buffer, 1, n, state->expected) != n || memcmp(buffer, data + off, n)) {
            state->failed = 1;
            errno = EIO;
            return 0;
        }
        state->offset += n;
        off += n;
    }
    return (ssize_t)len;
}

static FILE *compare_open(compare_stream *state, FILE *expected) {
    if (fseeko(expected, 0, SEEK_SET)) { return NULL; }
    *state = (compare_stream){.expected = expected};
    const cookie_io_functions_t io = {.write = compare_write};
    return fopencookie(state, "wb", io);
}

static int compare_close(FILE *stream, compare_stream *state, uint64_t bytes) {
    const int flushed = fflush(stream) == 0 && !ferror(stream);
    const int closed = fclose(stream) == 0;
    return flushed && closed && !state->failed && state->offset == bytes &&
        fgetc(state->expected) == EOF && !ferror(state->expected);
}

static int compare_case(stream_case mode) {
    unsigned char data[IO_BYTES + 31];
    for (size_t i = 0; i < sizeof(data); i++) { data[i] = (unsigned char)(i * 37u); }
    FILE *expected = tmpfile();
    if (!expected) { return 0; }
    const size_t n = sizeof(data) - 1;
    if (fwrite(data, 1, n, expected) != n || fflush(expected)) {
        fclose(expected);
        return 0;
    }
    if (mode == STREAM_MUTATED) { data[n / 2] ^= 1u; }
    size_t written = n;
    if (mode == STREAM_SHORT) { written--; }
    if (mode == STREAM_EXTRA) { written++; }
    compare_stream state;
    FILE *stream = compare_open(&state, expected);
    if (!stream) {
        fclose(expected);
        return 0;
    }
    const int wrote = fwrite(data, 1, written, stream) == written;
    const int equal = compare_close(stream, &state, n);
    fclose(expected);
    return (wrote && equal) == (mode == STREAM_EXACT);
}

static int compare_selfcheck(void) {
    for (int mode = STREAM_EXACT; mode <= STREAM_EXTRA; mode++) {
        if (!compare_case((stream_case)mode)) { return 1; }
    }
    fprintf(stderr, "GLM streaming comparison: exact/mutated/short/extra PASS\n");
    return 0;
}

#ifdef GLM53_COMPARE_ONLY
int main(void) { return compare_selfcheck(); }
#else
static const uint64_t CACHE_BYTES = UINT64_C(24) << 30;
static char error[512];

#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr, "GLM long MTP FAIL %d: %s (%s)\n", __LINE__, #x, error); \
    goto cleanup; \
} } while (0)

static uint64_t file_bytes(FILE *fp) {
    if (fseeko(fp, 0, SEEK_END)) { return 0; }
    const off_t end = ftello(fp);
    return end > 0 ? (uint64_t)end : 0;
}

static uint64_t save_payload(ds4_session *s, FILE *fp) {
    const uint64_t bytes = ds4_session_payload_bytes(s);
    if (!bytes || fseeko(fp, 0, SEEK_SET) || ftruncate(fileno(fp), 0) ||
        ds4_session_save_payload(s, fp, error, sizeof(error)) || fflush(fp) ||
        file_bytes(fp) != bytes) { return 0; }
    return bytes;
}

static int load_payload(ds4_session *s, FILE *fp, uint64_t bytes) {
    if (fseeko(fp, 0, SEEK_SET)) { return 0; }
    return !ds4_session_load_payload(s, fp, bytes, error, sizeof(error));
}

static int same_payload(ds4_session *s, FILE *expected, uint64_t bytes) {
    if (ds4_session_payload_bytes(s) != bytes) { return 0; }
    compare_stream state;
    FILE *stream = compare_open(&state, expected);
    if (!stream) { return 0; }
    const int saved = !ds4_session_save_payload(s, stream, error, sizeof(error));
    const int equal = compare_close(stream, &state, bytes);
    if (!equal) {
        fprintf(stderr, "GLM long MTP state mismatch near byte %" PRIu64 "\n", state.offset);
    }
    return saved && equal;
}

static int read_logits(ds4_session *s, float *logits, int vocab) {
    if (ds4_session_copy_logits(s, logits, vocab) != vocab) { return 0; }
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(logits[i])) { return 0; }
    }
    return 1;
}

static FILE *open_stage(const char *dir, const char *name) {
    char path[4096];
    const int n = snprintf(path, sizeof(path), "%s/%s.payload.bin", dir, name);
    if (n < 0 || n >= (int)sizeof(path)) { return NULL; }
    return fopen(path, "wb+");
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--self-test")) { return compare_selfcheck(); }
    if ((argc != 4 && argc != 5) || (argc == 5 && strcmp(argv[4], "--resident"))) {
        fprintf(stderr, "usage: %s MODEL PAYLOAD OUTPUT_DIR [--resident]\n", argv[0]);
        return 2;
    }
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    FILE *input = NULL, *stages[2] = {NULL, NULL};
    float *actual = NULL, *replay = NULL;
    int failed = 1, generated[OUTPUT_TOKENS], counts[OUTPUT_TOKENS];
    int total = 0, cycles = 0, max_accepted = 0;
    REQUIRE(!compare_selfcheck());
    const char *manifest = getenv("DS4_CUDA_WEIGHT_IPC_MANIFEST");
    REQUIRE(argc != 5 || (manifest && *manifest));
    input = fopen(argv[2], "rb");
    REQUIRE(input);
    const uint64_t input_bytes = file_bytes(input);
    uint32_t header[HEADER_WORDS];
    REQUIRE(input_bytes && !fseeko(input, 0, SEEK_SET) &&
        fread(header, sizeof(*header), HEADER_WORDS, input) == HEADER_WORDS);
    /* Native payload parsing validates the remaining source-matched layout.
     * Restrict this diagnostic to compact MTP checkpoints beyond the raw window. */
    REQUIRE(header[5] == UINT32_C(0x354d4347) && header[2] <= MAX_CTX &&
        header[7] > 2048 && header[7] + OUTPUT_TOKENS <= header[2]);
    ds4_engine_options options = {.model_path = argv[1], .backend = DS4_BACKEND_CUDA,
        .n_threads = 8, .defer_boot_prewarm = true, .mtp_draft_tokens = TRIAL_ROWS - 1,
        .mtp_margin = 0.0f, .ssd_streaming = argc != 5,
        .ssd_streaming_cache_bytes = argc != 5 ? CACHE_BYTES : 0};
    REQUIRE(!ds4_engine_open(&engine, &options) && ds4_engine_has_mtp(engine) &&
        ds4_engine_layer_count(engine) == 45);
    REQUIRE(!ds4_session_create(&session, engine, (int)header[2]));
    REQUIRE(load_payload(session, input, input_bytes) &&
        ds4_session_pos(session) == (int)header[7] && same_payload(session, input, input_bytes));
    const int vocab = ds4_engine_vocab_size(engine);
    REQUIRE(vocab > 0);
    actual = malloc((size_t)vocab * sizeof(*actual));
    replay = malloc((size_t)vocab * sizeof(*replay));
    stages[0] = open_stage(argv[3], "accepted-a");
    stages[1] = open_stage(argv[3], "accepted-b");
    REQUIRE(actual && replay && stages[0] && stages[1]);
    FILE *baseline = input;
    uint64_t baseline_bytes = input_bytes;
    while (total < OUTPUT_TOKENS) {
        const int before = ds4_session_pos(session);
        const int first = ds4_session_argmax(session);
        REQUIRE(first >= 0);
        int accepted[TRIAL_ROWS];
        const int n = ds4_session_eval_speculative_argmax(session, first,
            OUTPUT_TOKENS - total, ds4_token_eos(engine), accepted, TRIAL_ROWS,
            error, sizeof(error));
        REQUIRE(n > 0 && n <= TRIAL_ROWS && total + n <= OUTPUT_TOKENS &&
            ds4_session_pos(session) == before + n && read_logits(session, actual, vocab));
        FILE *committed = stages[cycles % 2];
        const uint64_t committed_bytes = save_payload(session, committed);
        REQUIRE(committed_bytes);

        /* Restore is tested independently before the width-one target replay.
         * Streaming comparison includes every serialized KV and predictor byte. */
        REQUIRE(load_payload(session, baseline, baseline_bytes) &&
            same_payload(session, baseline, baseline_bytes));
        for (int i = 0; i < n; i++) {
            REQUIRE(ds4_session_argmax(session) == accepted[i] &&
                !ds4_session_eval(session, accepted[i], error, sizeof(error)));
        }
        REQUIRE(same_payload(session, committed, committed_bytes) &&
            read_logits(session, replay, vocab) &&
            !memcmp(actual, replay, (size_t)vocab * sizeof(*actual)));
        const ds4_tokens *history = ds4_session_tokens(session);
        REQUIRE(history && history->len == before + n &&
            !memcmp(history->v + before, accepted, (size_t)n * sizeof(*accepted)));
        memcpy(generated + total, accepted, (size_t)n * sizeof(*accepted));
        total += n;
        counts[cycles++] = n;
        if (n > max_accepted) { max_accepted = n; }
        fprintf(stderr, "GLM long MTP cycle=%d accepted=%d pos=%d bytes=%" PRIu64
            ": target/restore byte exact\n", cycles, n, history->len, committed_bytes);
        baseline = committed;
        baseline_bytes = committed_bytes;
        if (ds4_token_is_stop(engine, accepted[n - 1])) { break; }
    }
    REQUIRE(max_accepted >= MULTI_MIN && !ds4_gpu_mem_census_faults() &&
        !ds4_metrics_get()->memgov_faults);
    printf("{\"result\":\"PASS\",\"context\":%u,\"prompt_tokens\":%u,"
        "\"generated_ids\":[", header[2], header[7]);
    for (int i = 0; i < total; i++) { printf("%s%d", i ? "," : "", generated[i]); }
    printf("],\"accepted_counts\":[");
    for (int i = 0; i < cycles; i++) { printf("%s%d", i ? "," : "", counts[i]); }
    printf("],\"max_accepted\":%d,\"target_state\":\"byte_exact\","
        "\"file_restore\":\"byte_exact\",\"target_logits\":\"all_f32_bits_equal\","
        "\"target_argmax\":\"all_ids_equal\",\"census_faults\":0,\"memgov_faults\":0}\n",
        max_accepted);
    failed = 0;
cleanup:
    if (input) { fclose(input); }
    for (int i = 0; i < 2; i++) {
        if (stages[i]) { fclose(stages[i]); }
    }
    free(actual);
    free(replay);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return failed;
}
#endif
