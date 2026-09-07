/* Same-process vision-path comparison at the production bank geometry.
 * Includes the native graph only to inspect embeddings and frontier logits;
 * the public serving ABI remains unchanged. Run with a GGUF and 1..4 images. */
#include "../ds4.c"

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d %s: %s\n", \
    __FILE__, __LINE__, #x, error); exit(1); } } while (0)

typedef struct {
    ds4_cont_request request;
    ds4_batch_ctx *ctx;
    float *logits;
    float *features;
    uint64_t feature_count;
    int admitted;
    int bank;
    int token;
    int done;
} vision_case;

static int vision_admit(void *ud, ds4_cont_request *request) {
    vision_case *c = ud;
    if (c->admitted++) { return 0; }
    *request = c->request;
    return 1;
}

static int vision_sample(void *ud, void *user) {
    (void)user;
    vision_case *c = ud;
    memcpy(c->logits, family_banked_logits(c->ctx, c->bank),
           DS4_N_VOCAB * sizeof(float));
    return DS4_SAMPLE_OVERRIDE_GREEDY;
}

static int vision_placed(void *ud, void *user, int cached, int computed, int bank) {
    (void)user; (void)computed;
    vision_case *c = ud;
    if (cached) { fprintf(stderr, "FAIL expected cold placement\n"); exit(1); }
    c->bank = bank;
    const ds4_qwen_gpu_graph *g = &c->ctx->qwen->graph[c->bank];
    if (!g->image_features || !ds4_gpu_tensor_read(g->image_features, 0,
            c->features, c->feature_count * sizeof(float))) {
        fprintf(stderr, "FAIL vision feature readback\n"); exit(1);
    }
    return 1;
}

static void vision_done(void *ud, void *user, const int *tokens, int n, int finish) {
    (void)user; (void)finish;
    vision_case *c = ud;
    if (!tokens || n != 1) { fprintf(stderr, "FAIL vision generation\n"); exit(1); }
    c->token = tokens[0]; c->done++;
}

static double compare(const char *name, const float *a, const float *b, uint64_t n) {
    double err = 0, ref = 0, max = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) { fprintf(stderr, "FAIL finite\n"); exit(1); }
        const double delta = (double)a[i] - b[i];
        err += delta * delta; ref += (double)a[i] * a[i];
        max = fmax(max, fabs(delta));
    }
    const double rms = sqrt(err / fmax(ref, 1e-30));
    printf("%s values=%llu max_abs=%.9g rel_rms=%.9g\n", name,
           (unsigned long long)n, max, rms);
    return rms;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        fprintf(stderr, "usage: %s MODEL IMAGE [IMAGE ...]\n", argv[0]); return 2;
    }
    char error[256] = "";
    REQUIRE(unsetenv("DS4_CUDA_NO_QWEN_VISION_TILE") == 0);
    ds4_engine_options opt = {0};
    opt.model_path = argv[1]; opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8; opt.mtp_draft_tokens = 2; opt.defer_boot_prewarm = true;
    ds4_engine *engine = NULL;
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    vision_case c = {0};
    uint64_t patch_rows = 0;
    ds4_tokens prompt = {0}, tail = {0};
    ds4_tokenize_text(engine, "<|im_start|>user\nBriefly describe the visible content and any task counts.\n", &prompt);
    uint8_t *data[4] = {0};
    for (int i = 2; i < argc; i++) {
        FILE *f = fopen(argv[i], "rb"); REQUIRE(f);
        REQUIRE(fseek(f, 0, SEEK_END) == 0);
        const long len = ftell(f); REQUIRE(len > 0 && len <= 10 * 1024 * 1024);
        rewind(f); data[i - 2] = xmalloc(len);
        REQUIRE(fread(data[i - 2], 1, len, f) == (size_t)len); fclose(f);
        ds4_qwen_image_info info = {0};
        REQUIRE(ds4_qwen_image_probe(data[i - 2], len, &info, error, sizeof(error)) == 0);
        patch_rows += (uint64_t)info.grid_h * info.grid_w;
        ds4_tokens_push(&prompt, DS4_QWEN_VISION_START_TOKEN_ID);
        c.request.images[i - 2] = (ds4_qwen_image_input){
            .data = data[i - 2], .data_len = len, .token_offset = prompt.len,
            .grid_h = info.grid_h, .grid_w = info.grid_w};
        for (uint32_t t = 0; t < info.token_count; t++) {
            ds4_tokens_push(&prompt, DS4_QWEN_IMAGE_PAD_TOKEN_ID);
        }
        ds4_tokens_push(&prompt, DS4_QWEN_VISION_END_TOKEN_ID);
        c.feature_count += (uint64_t)info.token_count * DS4_N_EMBD;
        c.request.image_count++;
    }
    printf("image patch rows=%llu (paired kernel threshold=512)\n",
           (unsigned long long)patch_rows);
    REQUIRE(patch_rows >= 512);
    ds4_tokenize_text(engine, "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", &tail);
    for (int i = 0; i < tail.len; i++) { ds4_tokens_push(&prompt, tail.v[i]); }
    REQUIRE(ds4_batch_ctx_create_fit(engine, 262144, 2, 16384,
                                    &c.ctx, error, sizeof(error)) == 0);
    c.request.tokens = prompt.v; c.request.n = prompt.len; c.request.max_new = 1;
    c.request.eos = -1; c.request.on_admitted = vision_placed;
    c.request.sample_override = vision_sample;
    float *logits[4], *features[4]; int tokens[4];
    for (int pass = 0; pass < 4; pass++) {
        if (pass < 2) { REQUIRE(setenv("DS4_QWEN_VISION_LEGACY", "1", 1) == 0); }
        else { REQUIRE(unsetenv("DS4_QWEN_VISION_LEGACY") == 0); }
        c.logits = logits[pass] = xmalloc(DS4_N_VOCAB * sizeof(float));
        c.features = features[pass] = xmalloc(c.feature_count * sizeof(float));
        c.admitted = c.done = 0;
        REQUIRE(ds4_engine_continuous_generate(c.ctx, vision_admit, NULL,
                     vision_done, &c, error, sizeof(error)) == 0);
        REQUIRE(c.done == 1); tokens[pass] = c.token;
        printf("pass=%d images=%u prompt=%d token=%d\n", pass,
               c.request.image_count, prompt.len, c.token);
        family_banked_reset(c.ctx, c.bank);
    }
    compare("legacy repeat features", features[0], features[1], c.feature_count);
    compare("legacy repeat logits", logits[0], logits[1], DS4_N_VOCAB);
    compare("vision features", features[1], features[2], c.feature_count);
    compare("frontier logits", logits[1], logits[2], DS4_N_VOCAB);
    compare("new repeat features", features[2], features[3], c.feature_count);
    compare("new repeat logits", logits[2], logits[3], DS4_N_VOCAB);
    fflush(stdout);
    REQUIRE(memcmp(features[0], features[1], c.feature_count * sizeof(float)) == 0);
    REQUIRE(memcmp(logits[0], logits[1], DS4_N_VOCAB * sizeof(float)) == 0);
    REQUIRE(memcmp(features[1], features[2], c.feature_count * sizeof(float)) == 0);
    REQUIRE(memcmp(logits[1], logits[2], DS4_N_VOCAB * sizeof(float)) == 0);
    REQUIRE(tokens[0] == tokens[1] && tokens[1] == tokens[2] && tokens[2] == tokens[3]);
    REQUIRE(memcmp(features[2], features[3], c.feature_count * sizeof(float)) == 0);
    REQUIRE(memcmp(logits[2], logits[3], DS4_N_VOCAB * sizeof(float)) == 0);
    for (int p = 0; p < 4; p++) { free(features[p]); free(logits[p]); }
    for (int i = 0; i < 4; i++) { free(data[i]); }
    ds4_batch_ctx_destroy(c.ctx); ds4_tokens_free(&tail); ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    puts("VISION MODEL PASS"); return 0;
}
