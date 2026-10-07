/* Bounded full-artifact contract. Each invocation owns one engine; the Python
 * harness runs resident/SSD and row widths sequentially in fresh processes. */
#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CONTRACT_CTX = 2048, STRUCTURAL_CTX = 1048576, GEN_TOKENS = 24,
    MTP_ROWS = 4, VISION_START = 154830, VISION_PAD = 154854, VISION_END = 154831 };
static char error[512];
static const char *output_dir;
static const uint8_t png_1x1[] = {
    0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a,0,0,0,0x0d,
    0x49,0x48,0x44,0x52,0,0,0,1,0,0,0,1,8,4,0,0,0,0xb5,0x1c,0x0c,2,
    0,0,0,0x0b,0x49,0x44,0x41,0x54,0x78,0xda,0x63,0x64,0xf8,0x0f,0,
    1,5,1,1,0x27,0x18,0xe3,0x66,0,0,0,0,0x49,0x45,0x4e,0x44,0xae,0x42,0x60,0x82
};
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "GLM contract FAIL line %d: %s (%s)\n", \
    __LINE__, #x, error); goto cleanup; } } while (0)

static FILE *artifact(const char *stage, const char *suffix) {
    if (!output_dir) { return tmpfile(); }
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/%s.%s", output_dir, stage, suffix) >= (int)sizeof(path)) { return NULL; }
    return fopen(path, "wb+");
}
static int finite_logits(ds4_session *s, float *logits, int vocab) {
    if (ds4_session_copy_logits(s, logits, vocab) != vocab) { return 0; }
    for (int i = 0; i < vocab; i++) { if (!isfinite(logits[i])) { return 0; } }
    return 1;
}
static int dump_stage(ds4_session *s, const char *stage, float *logits, int vocab) {
    if (!finite_logits(s, logits, vocab)) { return 0; }
    FILE *fp = artifact(stage, "logits.f32");
    int ok = fp && fwrite(logits, sizeof(*logits), (size_t)vocab, fp) == (size_t)vocab;
    if (fp && fclose(fp)) { ok = 0; }
    fp = artifact(stage, "payload.bin");
    const uint64_t bytes = ds4_session_payload_bytes(s);
    ok = ok && fp && bytes && !ds4_session_save_payload(s, fp, error, sizeof(error)) &&
         (uint64_t)ftello(fp) == bytes;
    if (fp && fclose(fp)) { ok = 0; }
    fp = artifact(stage, "tokens.txt");
    const ds4_tokens *tokens = ds4_session_tokens(s);
    if (!fp || !tokens) { ok = 0; }
    for (int i = 0; fp && tokens && i < tokens->len; i++) { fprintf(fp, "%d\n", tokens->v[i]); }
    if (fp && fclose(fp)) { ok = 0; }
    return ok;
}
static int snapshot_same(ds4_session *s, const ds4_session_snapshot *snap) {
    FILE *fp = tmpfile();
    if (!fp) { return 0; }
    int ok = ds4_session_payload_bytes(s) == snap->len &&
             !ds4_session_save_payload(s, fp, error, sizeof(error));
    rewind(fp);
    uint8_t buf[65536];
    for (uint64_t off = 0; ok && off < snap->len;) {
        const size_t n = snap->len - off < sizeof(buf) ? (size_t)(snap->len - off) : sizeof(buf);
        ok = fread(buf, 1u, n, fp) == n && !memcmp(buf, snap->ptr + off, n);
        off += n;
    }
    fclose(fp); return ok;
}
static int pending_reject(ds4_session *s, const ds4_tokens *prompt,
                          const ds4_session_snapshot *snap) {
    FILE *fp = tmpfile();
    if (!fp) { return 0; }
    const uint64_t generation = ds4_session_generation(s);
    int ok = ds4_session_sync(s, prompt, error, sizeof(error)) != 0 &&
             ds4_session_save_payload(s, fp, error, sizeof(error)) != 0 &&
             ds4_session_glm53_commit(s, MTP_ROWS + 1, error, sizeof(error)) != 0;
    rewind(fp);
    ok = ok && fwrite(snap->ptr, 1u, (size_t)snap->len, fp) == snap->len;
    rewind(fp);
    ok = ok && ds4_session_load_payload(s, fp, snap->len, error, sizeof(error)) != 0 &&
         ds4_session_generation(s) == generation && ds4_session_pos(s) == prompt->len;
    fclose(fp); return ok;
}
static char *input_text(void) {
    const char *path = getenv("DS4_GLM53_CONTRACT_PROMPT");
    if (!path) { return strdup("A warehouse had 240 notebooks. On Monday it shipped 35, "
        "on Tuesday it received 18, and on Wednesday it shipped 47. Show the calculation "
        "and state how many notebooks remain. Then write a Python function that computes "
        "the remaining count from an initial count and a list of signed changes. "
        "Keep the explanation concise. Before answering, account for every event in order. "
        "A positive change is incoming stock and a negative change is outgoing stock. "
        "Use the same sign convention for the explanation and the function. The function "
        "should return an integer and should not print. Include one assertion for this example. "
        "The audit records each movement once. The Monday shipment left the warehouse "
        "after the opening count was recorded. The Tuesday receipt increased the physical "
        "stock before Wednesday's shipment. There were no returns, losses, adjustments, "
        "or other movements during this period. A previous draft incorrectly added an "
        "outgoing shipment, so check the signs carefully. Do not change any of the event "
        "counts. The final explanation should be understandable to a colleague who has "
        "never seen this ledger. The Python function should also work for an empty list "
        "of changes. Avoid any external dependencies. The assertion should use the initial "
        "count and these three signed changes, and should verify the same result stated "
        "in the explanation. Give the result first, then the calculation and code."); }
    FILE *fp = fopen(path, "rb");
    if (!fp || fseeko(fp, 0, SEEK_END)) { if (fp) { fclose(fp); } return NULL; }
    const off_t n = ftello(fp);
    if (n <= 0 || n > 65536 || fseeko(fp, 0, SEEK_SET)) { fclose(fp); return NULL; }
    char *text = malloc((size_t)n + 1u);
    if (!text || fread(text, 1u, (size_t)n, fp) != (size_t)n) { free(text); fclose(fp); return NULL; }
    text[n] = 0; fclose(fp); return text;
}
static int generate(ds4_engine *e, ds4_session *s, const char *label,
        float temperature, int *tokens, int *count) {
    FILE *text = artifact(label, "text.txt"), *ids = artifact(label, "tokens.txt");
    if (!text || !ids) { if (text) { fclose(text); } if (ids) { fclose(ids); } return 0; }
    uint64_t rng = 424242u;
    int ok = 1; *count = 0;
    for (int i = 0; ok && i < GEN_TOKENS; i++) {
        const int t = ds4_session_sample(s, temperature, 40, 0.95f, 0.0f, &rng);
        if (t < 0) { ok = 0; break; }
        tokens[(*count)++] = t;
        fprintf(ids, "%d\n", t);
        size_t n = 0; char *piece = ds4_token_text(e, t, &n);
        if (!piece || fwrite(piece, 1u, n, text) != n) { ok = 0; }
        free(piece);
        if (ds4_token_is_stop(e, t)) { break; }
        ok = !ds4_session_eval(s, t, error, sizeof(error));
    }
    if (fclose(text) || fclose(ids)) { ok = 0; }
    return ok && *count > 0;
}

typedef struct {
    ds4_cont_request request;
    int admitted, cached, computed, bank, done, count;
    int result[GEN_TOKENS];
} bank_case;
static int bank_admit(void *ud, ds4_cont_request *req) {
    bank_case *c = ud;
    if (c->admitted) { return 0; }
    c->admitted = 1; *req = c->request; req->user = c; return 1;
}
typedef struct { bank_case c[2]; int next; } bank_pair;
static int pair_admit(void *ud, ds4_cont_request *req) {
    bank_pair *pair = ud;
    if (pair->next == 2) { return 0; }
    bank_case *c = &pair->c[pair->next++];
    *req = c->request; req->user = c; return 1;
}
static int bank_placed(void *ud, void *user, int cached, int computed, int bank) {
    (void)ud; bank_case *c = user;
    c->cached = cached; c->computed = computed; c->bank = bank; return 1;
}
static void bank_done(void *ud, void *user, const int *tokens, int n, int finish) {
    (void)ud; (void)finish; bank_case *c = user;
    c->done = tokens && n > 0 && n <= GEN_TOKENS;
    c->count = n;
    if (c->done) { memcpy(c->result, tokens, (size_t)n * sizeof(int)); }
}
static int bank_run(ds4_batch_ctx *ctx, bank_case *c) {
    c->request.on_admitted = bank_placed;
    return !ds4_engine_continuous_generate(ctx, bank_admit, NULL, bank_done, c, error, sizeof(error)) && c->done;
}
static int files_same(FILE *a, FILE *b) {
    uint8_t x[65536], y[65536]; rewind(a); rewind(b);
    for (;;) {
        const size_t n = fread(x, 1u, sizeof(x), a), m = fread(y, 1u, sizeof(y), b);
        if (n != m || memcmp(x, y, n)) { return 0; }
        if (!n) { return !ferror(a) && !ferror(b); }
    }
}
static int bank_payload(ds4_batch_ctx *ctx, uint32_t bank, FILE *fp) {
    const uint64_t n = ds4_cont_bank_payload_bytes(ctx, bank);
    return n && !ds4_cont_bank_save_payload(ctx, bank, fp, error, sizeof(error)) && (uint64_t)ftello(fp) == n;
}
static int banks_contract(ds4_engine *e, const ds4_tokens *prompt, const int *greedy) {
    ds4_batch_ctx *ctx = NULL;
    ds4_session *reference = NULL;
    ds4_tokens fork = {0}, partial = {0}, prefix = {0};
    FILE *before = NULL, *after = NULL, *disk = NULL;
    int ok = 0;
    const int cut = prompt->len / 2;
    if (ds4_batch_ctx_create(e, CONTRACT_CTX, 2, 256, &ctx, error, sizeof(error)) ||
        ds4_batch_ctx_max_seq(ctx) != 2 || !ds4_batch_ctx_supports_partial_reuse(ctx)) { goto cleanup; }
    bank_pair seeds = {0};
    for (int i = 0; i < 2; i++) {
        seeds.c[i].request = (ds4_cont_request){.tokens=prompt->v, .n=prompt->len,
            .max_new=2, .eos=-1, .place_bank=i + 1, .checkpoint_at=cut, .on_admitted=bank_placed};
    }
    if (ds4_engine_continuous_generate(ctx, pair_admit, NULL, bank_done, &seeds, error, sizeof(error))) { goto cleanup; }
    for (int i = 0; i < 2; i++) {
        bank_case *seed = &seeds.c[i];
        if (!seed->done || seed->count != 2 || seed->cached || seed->bank != i ||
            memcmp(seed->result, greedy, 2u * sizeof(int))) { goto cleanup; }
    }
    const int *history = NULL;
    if (ds4_batch_ctx_bank_committed(ctx, 0, &history) != prompt->len + 1 ||
        memcmp(history, prompt->v, (size_t)prompt->len * sizeof(int)) ||
        history[prompt->len] != greedy[0]) { goto cleanup; }
    before = artifact("bank-source", "payload.bin");
    if (!before || !bank_payload(ctx, 0u, before)) { goto cleanup; }
    ds4_tokens_copy(&fork, prompt); ds4_tokens_push(&fork, greedy[0]); ds4_tokens_push(&fork, greedy[1]);
    bank_case copy = {.request = {.tokens=fork.v, .n=fork.len, .max_new=1,
        .eos=-1, .place_bank=2, .fork_bank=1, .n_cached=prompt->len + 1}};
    if (!bank_run(ctx, &copy) || copy.cached != prompt->len + 1 || copy.bank != 1 ||
        copy.result[0] != greedy[2]) { goto cleanup; }
    after = tmpfile();
    if (!after || !bank_payload(ctx, 0u, after) || !files_same(before, after)) { goto cleanup; }
    for (int i = 0; i < cut + 1; i++) { ds4_tokens_push(&partial, prompt->v[i]); }
    ds4_tokenize_text(e, " Give only the integer remaining.", &partial);
    bank_case branch = {.request = {.tokens=partial.v, .n=partial.len, .max_new=1,
        .eos=-1, .place_bank=2, .fork_bank=1, .n_cached=cut + 1}};
    if (!bank_run(ctx, &branch) || branch.cached != cut || branch.computed != partial.len - cut) { goto cleanup; }
    fclose(after); after = tmpfile();
    if (!after || !bank_payload(ctx, 0u, after) || !files_same(before, after)) { goto cleanup; }
    disk = artifact("bank-partial", "payload.bin");
    if (!disk || !bank_payload(ctx, 1u, disk)) { goto cleanup; }
    const uint64_t bytes = (uint64_t)ftello(disk);
    rewind(disk);
    if (ds4_cont_bank_restore_payload(ctx, 0u, disk, bytes, error, sizeof(error))) { goto cleanup; }
    if (ds4_batch_ctx_bank_committed(ctx, 0, &history) != partial.len ||
        memcmp(history, partial.v, (size_t)partial.len * sizeof(int))) { goto cleanup; }
    fclose(after); after = artifact("bank-restored", "payload.bin");
    if (!after || !bank_payload(ctx, 0u, after) || !files_same(disk, after)) { goto cleanup; }
    bank_case warm = {.request = {.tokens=partial.v, .n=partial.len, .max_new=1,
        .eos=-1, .place_bank=1, .n_cached=partial.len}};
    if (!bank_run(ctx, &warm) || warm.cached != partial.len || warm.result[0] != branch.result[0]) { goto cleanup; }
    ds4_batch_ctx_destroy(ctx); ctx = NULL;
    for (int i = 0; i < cut; i++) { ds4_tokens_push(&prefix, partial.v[i]); }
    if (ds4_session_create(&reference, e, CONTRACT_CTX) ||
        ds4_session_sync(reference, &prefix, error, sizeof(error)) ||
        ds4_session_sync(reference, &partial, error, sizeof(error)) ||
        ds4_session_argmax(reference) != branch.result[0]) { goto cleanup; }
    fclose(after); after = artifact("bank-partial-reference", "payload.bin");
    if (!after || ds4_session_save_payload(reference, after, error, sizeof(error)) ||
        !files_same(disk, after)) { goto cleanup; }
    ok = 1;
cleanup:
    if (before) { fclose(before); } if (after) { fclose(after); } if (disk) { fclose(disk); }
    ds4_tokens_free(&fork); ds4_tokens_free(&partial); ds4_tokens_free(&prefix);
    ds4_session_free(reference); ds4_batch_ctx_destroy(ctx); return ok;
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) { fprintf(stderr, "usage: %s <GLM artifact.gguf> [vision.gguf]\n", argv[0]); return 2; }
    output_dir = getenv("DS4_GLM53_CONTRACT_OUT");
    const char *stream = getenv("DS4_GLM53_CONTRACT_SSD");
    const char *mtp = getenv("DS4_GLM53_CONTRACT_MTP");
    ds4_engine_options opt = {.model_path=argv[1], .vision_path=argc == 3 ? argv[2] : NULL,
        .backend=DS4_BACKEND_CUDA, .n_threads=8, .defer_boot_prewarm=true,
        .mtp_draft_tokens=mtp && !strcmp(mtp, "1") ? 3 : 1,
        .ssd_streaming=stream && !strcmp(stream, "1"), .ssd_streaming_cache_experts=1024};
    ds4_engine *e = NULL; ds4_session *s = NULL, *probe = NULL;
    ds4_tokens prompt = {0}, prefix = {0}, image = {0};
    ds4_session_snapshot snap = {0}, accepted = {0};
    ds4_vision_embedding embedding = {0};
    float *logits = NULL; char *text = NULL;
    int failed = 1, greedy[GEN_TOKENS], sampled[GEN_TOKENS], ng = 0, ns = 0;
    REQUIRE(!ds4_engine_open(&e, &opt) && ds4_engine_layer_count(e) == 45 && ds4_engine_supports_batching(e));
    REQUIRE(!ds4_gpu_mem_census_faults() && !ds4_metrics_get()->memgov_faults);
    const char *expanded = getenv("DS4_GLM53_DSA_EXPANDED");
    const int diagnostic = expanded && !strcmp(expanded, "1");
    REQUIRE(diagnostic || ds4_engine_session_graph_bytes_estimate(e, STRUCTURAL_CTX) > 0);
    REQUIRE(ds4_engine_session_graph_bytes_estimate(e, STRUCTURAL_CTX + 1) == 0 &&
        ds4_engine_session_graph_bytes_estimate(e, 0) == 0 &&
        ds4_engine_session_graph_bytes_estimate(NULL, CONTRACT_CTX) == 0);
    REQUIRE(ds4_session_create(&probe, e, STRUCTURAL_CTX + 1) != 0 && !probe);
    if (!diagnostic) { REQUIRE(!ds4_session_create(&probe, e, 2049)); }
    ds4_session_free(probe); probe = NULL;
    REQUIRE(!ds4_session_create(&s, e, CONTRACT_CTX) && ds4_session_graph_pending(s));
    const int rowcap = ds4_session_prefill_cap(s);
    REQUIRE(rowcap == 1 || rowcap == 128 || rowcap == 256);
    text = input_text(); REQUIRE(text);
    ds4_encode_chat_prompt(e, NULL, text, DS4_THINK_NONE, &prompt);
    REQUIRE(prompt.len > 256 && prompt.len + GEN_TOKENS + MTP_ROWS < CONTRACT_CTX);
    const int cut = prompt.len / 2;
    for (int i = 0; i < cut; i++) { ds4_tokens_push(&prefix, prompt.v[i]); }
    REQUIRE(!ds4_session_sync(s, &prefix, error, sizeof(error)));
    const uint64_t generation = ds4_session_generation(s);
    REQUIRE(!ds4_session_sync(s, &prompt, error, sizeof(error)) && ds4_session_pos(s) == prompt.len &&
        ds4_session_generation(s) == generation);
    const int vocab = ds4_engine_vocab_size(e);
    logits = malloc((size_t)vocab * sizeof(*logits)); REQUIRE(logits);
    REQUIRE(dump_stage(s, "prefill", logits, vocab));
    fprintf(stderr, "GLM contract: prefill %d tokens, finite logits saved\n", prompt.len);
    REQUIRE(!ds4_session_save_snapshot(s, &snap, error, sizeof(error)) && snap.len);
    REQUIRE(generate(e, s, "greedy", 0.0f, greedy, &ng) && ng >= 3);
    REQUIRE(dump_stage(s, "greedy-end", logits, vocab));
    REQUIRE(!ds4_session_load_snapshot(s, &snap, error, sizeof(error)) && snapshot_same(s, &snap));
    REQUIRE(dump_stage(s, "restored", logits, vocab));
    REQUIRE(generate(e, s, "sampled", 0.7f, sampled, &ns));
    REQUIRE(dump_stage(s, "sampled-end", logits, vocab));
    fprintf(stderr, "GLM contract: greedy/sample and snapshot restore passed\n");
    REQUIRE(!ds4_session_load_snapshot(s, &snap, error, sizeof(error)));
    if (mtp && !strcmp(mtp, "1")) {
        REQUIRE(ds4_engine_has_mtp(e));
        int draft[MTP_ROWS], target[MTP_ROWS];
        int n = ds4_session_glm53_trial(s, greedy[0], MTP_ROWS, draft, target, MTP_ROWS, error, sizeof(error));
        REQUIRE(n == MTP_ROWS);
        REQUIRE(pending_reject(s, &prompt, &snap));
        REQUIRE(!ds4_session_glm53_commit(s, 0, error, sizeof(error)) && snapshot_same(s, &snap));
        REQUIRE(dump_stage(s, "mtp-abort", logits, vocab));
        n = ds4_session_glm53_trial(s, greedy[0], MTP_ROWS, draft, target, MTP_ROWS, error, sizeof(error));
        REQUIRE(n == MTP_ROWS && !ds4_session_glm53_commit(s, 1, error, sizeof(error)));
        REQUIRE(!ds4_session_save_snapshot(s, &accepted, error, sizeof(error)));
        REQUIRE(dump_stage(s, "mtp-keep1", logits, vocab));
        REQUIRE(!ds4_session_load_snapshot(s, &snap, error, sizeof(error)) &&
            !ds4_session_eval(s, greedy[0], error, sizeof(error)) && snapshot_same(s, &accepted));
        REQUIRE(dump_stage(s, "mtp-base1", logits, vocab));
        fprintf(stderr, "GLM contract: MTP abort/keep1 passed\n");
    }
    ds4_session_free(s); s = NULL; /* Banks share weights, never a second engine. */
    REQUIRE(banks_contract(e, &prompt, greedy));
    fprintf(stderr, "GLM contract: two banks, fork and disk restore passed\n");
    if (argc == 3) {
        REQUIRE(ds4_engine_has_vision(e) && ds4_engine_vision_encode_memory(e, png_1x1, sizeof(png_1x1),
            &embedding, error, sizeof(error)) && embedding.token_count == 16);
        ds4_tokens_push(&image, VISION_START);
        for (uint32_t i = 0; i < embedding.token_count; i++) { ds4_tokens_push(&image, VISION_PAD); }
        ds4_tokens_push(&image, VISION_END);
        ds4_tokens suffix = {0}; ds4_encode_chat_prompt(e, NULL, "Describe the image briefly.", DS4_THINK_NONE, &suffix);
        for (int i = 0; i < suffix.len; i++) { ds4_tokens_push(&image, suffix.v[i]); }
        ds4_tokens_free(&suffix);
        ds4_vision_span span = {.token_start=1, .embedding=embedding};
        REQUIRE(!ds4_session_create(&s, e, CONTRACT_CTX) &&
            !ds4_session_sync_multimodal(s, &image, &span, 1, error, sizeof(error)) &&
            ds4_session_pos(s) == image.len && dump_stage(s, "vision", logits, vocab));
        int nv = 0;
        REQUIRE(generate(e, s, "vision-greedy", 0.0f, sampled, &nv));
        fprintf(stderr, "GLM contract: Vision inference passed\n");
    }
    REQUIRE(!ds4_gpu_mem_census_faults() && !ds4_metrics_get()->memgov_faults);
    FILE *summary = artifact("contract", "json"); REQUIRE(summary);
    fprintf(summary, "{\"model\":\"GLM-5.3-Flash\",\"ctx\":%d,\"structural_ctx\":%d,"
        "\"prefill_rows\":%d,\"prompt_tokens\":%d,\"vocab\":%d,\"ssd\":%s,"
        "\"mtp\":%s,\"vision\":%s,\"diagnostic_expanded\":%s,\"greedy_tokens\":%d,\"sample_tokens\":%d,\"passed\":true}\n",
        CONTRACT_CTX, diagnostic ? CONTRACT_CTX : STRUCTURAL_CTX, rowcap, prompt.len, vocab, opt.ssd_streaming ? "true" : "false",
        mtp && !strcmp(mtp, "1") ? "true" : "false", argc == 3 ? "true" : "false",
        diagnostic ? "true" : "false", ng, ns);
    REQUIRE(!fclose(summary)); failed = 0;
    fprintf(stderr, "GLM actual artifact contract: ctx=%d prefill=%d prompt=%d, state/logits/banks passed\n",
        CONTRACT_CTX, rowcap, prompt.len);
cleanup:
    free(text); free(logits); ds4_vision_embedding_free(&embedding);
    ds4_session_snapshot_free(&snap); ds4_session_snapshot_free(&accepted);
    ds4_tokens_free(&prompt); ds4_tokens_free(&prefix); ds4_tokens_free(&image);
    ds4_session_free(s); ds4_session_free(probe); ds4_engine_close(e); return failed;
}
