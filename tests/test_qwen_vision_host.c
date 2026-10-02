/* Manual Qwen/Darwin numeric gate with Rust-tokenized whole-prompt IDs.
 * See fixtures/qwen-images/README.md for preparation and proof boundaries.
 * Native internals stay test-local; no production ABI is added. */
#include "../ds4.c"

enum {
    GATE_CTX = 262144,
    GATE_BANKS = 1,
    GATE_THREADS = 8,
    GATE_MIN_PATCHES = 512,
    GATE_PREFILL = 8192,
    GATE_TOKENS = 32,
    GATE_IMAGES = 4,
    GATE_PASSES = 4,
    GATE_PAIR_PASSES = 2,
    GATE_TOKEN_CONTROL = 3,
    GATE_TOKEN_USER_DEFINED = 4,
    GATE_IO_BYTES = 1024 * 1024,
    GATE_IMAGE_BYTES = 10 * 1024 * 1024,
    GATE_PATH_BYTES = 4096,
};

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", \
    __FILE__, __LINE__, #x); exit(1); } } while (0)

typedef struct {
    ds4_model map;
    ds4_host_vocab vocab;
    ds4_host_str *tokens;
    ds4_host_str *merges;
    int32_t *user_defined;
} gate_vocab;

typedef struct {
    ds4_cont_request request;
    ds4_batch_ctx *ctx;
    FILE *features;
    FILE *logits;
    uint64_t feature_values;
    int admitted;
    int bank;
    int eos;
    int sampled;
    int done;
    int output[GATE_TOKENS];
} gate_case;

static ds4_host_str *host_strings(const ds4_model *m, const char *name,
                                 uint32_t *count) {
    ds4_array_ref table;
    REQUIRE(model_get_array(m, name, &table));
    REQUIRE(table.type == GGUF_VALUE_STRING && table.len <= UINT32_MAX);
    *count = (uint32_t)table.len;
    ds4_host_str *out = xcalloc(*count, sizeof(*out));
    ds4_cursor cur = cursor_at(m, table.data_pos);
    for (uint32_t i = 0; i < *count; i++) {
        ds4_str s;
        REQUIRE(cursor_string(&cur, &s));
        out[i] = (ds4_host_str){.ptr = s.ptr, .len = s.len};
    }
    return out;
}

static int host_special(const ds4_host_vocab *v, const char *text) {
    const size_t len = strlen(text);
    for (uint32_t i = 0; i < v->n_vocab; i++) {
        if (v->tokens[i].len == len &&
            memcmp(v->tokens[i].ptr, text, len) == 0) {
            return (int)i;
        }
    }
    fprintf(stderr, "missing host special: %s\n", text);
    exit(1);
}

static void host_vocab_open(gate_vocab *host, const char *path) {
    memset(host, 0, sizeof(*host));
    /* Borrow real GGUF strings until engine_close; this mapping uploads no
     * weights. Rust --dump-tokens validates and tokenizes the same metadata. */
    model_open(&host->map, path, false, false);
    ds4_host_vocab *v = &host->vocab;
    *v = (ds4_host_vocab){
        .bos_id = -1, .eos_id = -1, .system_id = -1, .eot_id = -1,
        .im_start_id = -1, .im_content_id = -1, .im_end_id = -1,
        .user_id = -1, .assistant_id = -1, .start_of_turn_id = -1,
        .end_of_turn_id = -1, .tool_id = -1, .reference_id = -1,
        .plan_start_id = -1, .plan_end_id = -1, .observation_id = -1,
        .sop_id = -1, .think_start_id = -1, .think_end_id = -1,
        .tool_call_start_id = -1, .tool_call_end_id = -1,
        .tool_response_start_id = -1, .tool_response_end_id = -1,
        .arg_key_start_id = -1, .arg_key_end_id = -1,
        .arg_value_start_id = -1, .latent_start_id = -1,
        .latent_pad_id = -1, .latent_end_id = -1,
        .tool_schema_start_id = -1, .tool_schema_end_id = -1, .dsml_id = -1,
        .dots3_endofsystem_id = -1, .dots3_endofuser_id = -1,
        .dots3_endoftext_id = -1,
    };
    host->tokens = host_strings(&host->map, "tokenizer.ggml.tokens", &v->n_vocab);
    host->merges = host_strings(&host->map, "tokenizer.ggml.merges", &v->n_merges);
    v->tokens = host->tokens;
    v->merges = host->merges;
    host->user_defined = xcalloc(v->n_vocab, sizeof(*host->user_defined));
    ds4_array_ref types;
    REQUIRE(model_get_array(&host->map, "tokenizer.ggml.token_type", &types));
    REQUIRE((types.type == GGUF_VALUE_INT32 || types.type == GGUF_VALUE_UINT32)
            && types.len == v->n_vocab);
    ds4_str pre = {0};
    const bool added_controls = model_get_string(&host->map, "tokenizer.ggml.pre", &pre)
        && ds4_streq(pre, "qwen4exp");
    ds4_cursor cur = cursor_at(&host->map, types.data_pos);
    for (uint32_t i = 0; i < v->n_vocab; i++) {
        uint32_t type;
        REQUIRE(cursor_u32(&cur, &type));
        if ((type != GATE_TOKEN_USER_DEFINED &&
             !(added_controls && type == GATE_TOKEN_CONTROL)) || !v->tokens[i].len) {
            continue;
        }
        host->user_defined[v->n_user_defined++] = (int32_t)i;
        if (v->tokens[i].len > v->user_defined_max_len) {
            REQUIRE(v->tokens[i].len <= UINT32_MAX);
            v->user_defined_max_len = (uint32_t)v->tokens[i].len;
        }
        v->user_defined_first[(uint8_t)v->tokens[i].ptr[0]] = 1;
    }
    v->user_defined = host->user_defined;
    REQUIRE(model_get_token_id(&host->map, "tokenizer.ggml.bos_token_id", &v->bos_id));
    REQUIRE(model_get_token_id(&host->map, "tokenizer.ggml.eos_token_id", &v->eos_id));
    v->eot_id = host_special(v, "<|endoftext|>");
    v->im_start_id = host_special(v, "<|im_start|>");
    v->im_end_id = host_special(v, "<|im_end|>");
    v->think_start_id = host_special(v, "<think>");
    v->think_end_id = host_special(v, "</think>");
    v->tool_call_start_id = host_special(v, "<tool_call>");
    v->tool_call_end_id = host_special(v, "</tool_call>");
    v->tool_response_start_id = host_special(v, "<tool_response>");
    v->tool_response_end_id = host_special(v, "</tool_response>");
    char error[256] = "";
    REQUIRE(ds4_host_vocab_apply(v, error, sizeof(error)) == 0);
    ds4_host_vocab_install(v);
    printf("real host vocab tokens=%u merges=%u added=%u bos=%d eos=%d eot=%d\n",
           v->n_vocab, v->n_merges, v->n_user_defined,
           v->bos_id, v->eos_id, v->eot_id);
}

static void host_vocab_close(gate_vocab *host) {
    ds4_host_vocab_clear();
    free(host->user_defined);
    free(host->merges);
    free(host->tokens);
    model_close(&host->map);
}

static void tokens_read(const char *path, ds4_tokens *tokens) {
    FILE *fp = fopen(path, "rb");
    REQUIRE(fp && fgetc(fp) == '[');
    for (;;) {
        int ch;
        do { ch = fgetc(fp); } while (ch != EOF && isspace(ch));
        if (ch == ']') { break; }
        REQUIRE(ch != EOF && ungetc(ch, fp) != EOF);
        long token;
        REQUIRE(fscanf(fp, "%ld", &token) == 1);
        REQUIRE(token >= 0 && token <= INT_MAX && tokens->len < GATE_CTX);
        ds4_tokens_push(tokens, (int)token);
        do { ch = fgetc(fp); } while (ch != EOF && isspace(ch));
        REQUIRE(ch == ',' || ch == ']');
        if (ch == ']') { break; }
    }
    REQUIRE(tokens->len > 0);
    REQUIRE(fclose(fp) == 0);
}

static void file_path(char *out, const char *base, int pass, const char *kind) {
    const int n = snprintf(out, GATE_PATH_BYTES, "%s.vision-pass%d.%s", base, pass, kind);
    REQUIRE(n > 0 && n < GATE_PATH_BYTES);
}

static FILE *file_open(const char *base, int pass, const char *kind) {
    char path[GATE_PATH_BYTES];
    file_path(path, base, pass, kind);
    FILE *fp = fopen(path, "wbx");
    REQUIRE(fp);
    return fp;
}

static void tensor_write(FILE *fp, const ds4_gpu_tensor *tensor, uint64_t bytes) {
    uint8_t *buffer = xmalloc(GATE_IO_BYTES);
    for (uint64_t pos = 0; pos < bytes; pos += GATE_IO_BYTES) {
        const uint64_t left = bytes - pos;
        const size_t n = left < GATE_IO_BYTES ? (size_t)left : GATE_IO_BYTES;
        REQUIRE(ds4_gpu_tensor_read(tensor, pos, buffer, n));
        REQUIRE(fwrite(buffer, 1, n, fp) == n);
    }
    free(buffer);
}

static void files_equal(const char *base, int left, int right, const char *kind) {
    char path[GATE_PATH_BYTES];
    file_path(path, base, left, kind);
    FILE *a = fopen(path, "rb");
    file_path(path, base, right, kind);
    FILE *b = fopen(path, "rb");
    REQUIRE(a && b);
    uint8_t *x = xmalloc(GATE_IO_BYTES), *y = xmalloc(GATE_IO_BYTES);
    uint64_t pos = 0;
    for (;;) {
        const size_t an = fread(x, 1, GATE_IO_BYTES, a);
        const size_t bn = fread(y, 1, GATE_IO_BYTES, b);
        REQUIRE(!ferror(a) && !ferror(b) && an == bn);
        if (!an) { break; }
        if (memcmp(x, y, an) != 0) {
            fprintf(stderr, "FAIL %s pass%d/pass%d at block byte %llu\n",
                    kind, left, right, (unsigned long long)pos);
            exit(1);
        }
        pos += an;
    }
    printf("exact %s pass%d/pass%d bytes=%llu\n", kind, left, right,
           (unsigned long long)pos);
    free(y); free(x);
    REQUIRE(fclose(b) == 0 && fclose(a) == 0);
}

static int gate_admit(void *ud, ds4_cont_request *request) {
    gate_case *c = ud;
    if (c->admitted++) { return 0; }
    *request = c->request;
    return 1;
}

static int gate_sample(void *ud, void *user) {
    (void)user;
    gate_case *c = ud;
    /* Capture raw logits before the sampler excludes EOS and Qwen EOT. */
    const float *logits = family_banked_logits(c->ctx, c->bank);
    REQUIRE(logits && c->sampled < GATE_TOKENS);
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) { REQUIRE(isfinite(logits[i])); }
    REQUIRE(fwrite(logits, sizeof(float), DS4_N_VOCAB, c->logits) == DS4_N_VOCAB);
    c->sampled++;
    return DS4_SAMPLE_OVERRIDE_GREEDY;
}

static int gate_exclude(void *ud, void *user) {
    (void)user;
    /* The native sampler also excludes Qwen EOT when EOS is excluded. */
    return ((gate_case *)ud)->eos;
}

static int gate_placed(void *ud, void *user, int cached, int computed, int bank) {
    (void)user; (void)computed;
    gate_case *c = ud;
    REQUIRE(cached == 0 && bank == 0);
    c->bank = bank;
    const ds4_qwen_gpu_graph *g = &c->ctx->qwen->graph[bank];
    REQUIRE(g->image_features && g->image_feature_rows > 0);
    REQUIRE((uint64_t)g->image_feature_rows * DS4_N_EMBD == c->feature_values);
    tensor_write(c->features, g->image_features, c->feature_values * sizeof(float));
    return 1;
}

static void gate_done(void *ud, void *user, const int *tokens, int n, int finish) {
    (void)user;
    gate_case *c = ud;
    REQUIRE(tokens && n == GATE_TOKENS && finish == 0);
    memcpy(c->output, tokens, sizeof(c->output));
    c->done++;
}

static void save_state(const char *base, int pass, const gate_case *c) {
    char error[256] = "";
    FILE *fp = file_open(base, pass, "payload");
    const uint64_t expected = ds4_cont_bank_payload_bytes(c->ctx, c->bank);
    REQUIRE(expected > 0);
    REQUIRE(ds4_cont_bank_save_payload(c->ctx, c->bank, fp, error, sizeof(error)) == 0);
    REQUIRE((uint64_t)ftello(fp) == expected);
    REQUIRE(fclose(fp) == 0);
    fp = file_open(base, pass, "tokens.i32");
    REQUIRE(fwrite(c->output, sizeof(int), GATE_TOKENS, fp) == GATE_TOKENS);
    REQUIRE(fclose(fp) == 0);
    const ds4_qwen_gpu_graph *g = &c->ctx->qwen->graph[c->bank];
    /* The final sampled token has not been fed back into the decoder. */
    REQUIRE(g->length == (uint32_t)c->request.n + GATE_TOKENS - 1u);
    REQUIRE(c->ctx->bank_hist_len[c->bank] == g->length);
    REQUIRE(g->mrope_written == g->length);
    const uint32_t header[] = {
        g->length, (uint32_t)g->mrope_active, (uint32_t)g->mrope_delta,
        g->mrope_written, g->image_count, g->image_feature_rows,
    };
    fp = file_open(base, pass, "mrope");
    REQUIRE(fwrite(header, sizeof(uint32_t), sizeof(header) / sizeof(header[0]), fp)
            == sizeof(header) / sizeof(header[0]));
    REQUIRE(g->mrope_positions && g->mrope_written >= (uint32_t)c->request.n);
    tensor_write(fp, g->mrope_positions,
                 (uint64_t)g->mrope_written * 3u * sizeof(int32_t));
    REQUIRE(fwrite(g->image, sizeof(g->image[0]), g->image_count, fp) == g->image_count);
    REQUIRE(fwrite(&g->ple.hash_state, sizeof(g->ple.hash_state), 1, fp) == 1);
    REQUIRE(fclose(fp) == 0);
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 3 + GATE_IMAGES) {
        fprintf(stderr, "usage: %s MODEL TOKEN_IDS.txt IMAGE [IMAGE ...]\n", argv[0]);
        return 2;
    }
    REQUIRE(sizeof(int) == sizeof(int32_t));
    const char *gate_control = getenv("DS4_QWEN_VISION_GATE_CONTROL");
    if (gate_control) {
        REQUIRE(strcmp(gate_control, "DS4_QWEN_VISION_PACK") == 0);
    }
    gate_vocab host;
    host_vocab_open(&host, argv[1]);
    REQUIRE(unsetenv("DS4_CUDA_NO_QWEN_VISION_TILE") == 0);
    REQUIRE(unsetenv("DS4_QWEN_VISION_LEGACY") == 0);
    char error[256] = "";
    ds4_engine_options opt = {0};
    opt.model_path = argv[1]; opt.backend = DS4_BACKEND_CUDA;
    /* Plain decode exposes one logits frontier per sample callback. MTP2
     * needs a separate serving gate; its verify callback precedes decode. */
    opt.n_threads = GATE_THREADS; opt.mtp_draft_tokens = 1; opt.defer_boot_prewarm = true;
    ds4_engine *engine = NULL;
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    REQUIRE(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_QWEN4EXP);
    REQUIRE((uint32_t)ds4_engine_vocab_size(engine) == host.vocab.n_vocab);
    gate_case c = {.eos = host.vocab.eos_id};
    REQUIRE(c.eos >= 0);
    ds4_tokens raw = {0}, prompt = {0};
    tokens_read(argv[2], &raw);
    uint8_t *data[GATE_IMAGES] = {0};
    uint32_t image_i = 0, patches = 0;
    for (int i = 0; i < raw.len; i++) {
        REQUIRE(raw.v[i] >= 0 && (uint32_t)raw.v[i] < DS4_N_VOCAB);
        if (raw.v[i] != DS4_QWEN_IMAGE_PAD_TOKEN_ID) {
            ds4_tokens_push(&prompt, raw.v[i]);
            continue;
        }
        REQUIRE(image_i < (uint32_t)(argc - 3));
        FILE *fp = fopen(argv[3 + image_i], "rb");
        REQUIRE(fp && fseek(fp, 0, SEEK_END) == 0);
        const long bytes = ftell(fp);
        REQUIRE(bytes > 0 && bytes <= GATE_IMAGE_BYTES);
        rewind(fp); data[image_i] = xmalloc((size_t)bytes);
        REQUIRE(fread(data[image_i], 1, (size_t)bytes, fp) == (size_t)bytes);
        REQUIRE(fclose(fp) == 0);
        ds4_qwen_image_info info = {0};
        REQUIRE(ds4_qwen_image_probe(data[image_i], bytes, &info, error, sizeof(error)) == 0);
        c.request.images[image_i] = (ds4_qwen_image_input){
            .data = data[image_i], .data_len = bytes, .token_offset = prompt.len,
            .grid_h = info.grid_h, .grid_w = info.grid_w,
        };
        for (uint32_t t = 0; t < info.token_count; t++) {
            ds4_tokens_push(&prompt, DS4_QWEN_IMAGE_PAD_TOKEN_ID);
        }
        c.feature_values += (uint64_t)info.token_count * DS4_N_EMBD;
        patches += info.grid_h * info.grid_w;
        image_i++;
    }
    REQUIRE(image_i == (uint32_t)(argc - 3) && prompt.len < GATE_CTX - GATE_TOKENS);
    REQUIRE(patches >= GATE_MIN_PATCHES);
    REQUIRE(ds4_batch_ctx_create_fit(engine, GATE_CTX, GATE_BANKS, GATE_PREFILL,
                                    &c.ctx, error, sizeof(error)) == 0);
    REQUIRE(ds4_batch_ctx_max_seq(c.ctx) == GATE_BANKS && ds4_batch_ctx_seq_cap(c.ctx) == GATE_CTX);
    c.request.tokens = prompt.v; c.request.n = prompt.len;
    c.request.image_count = image_i; c.request.max_new = GATE_TOKENS;
    c.request.eos = c.eos; c.request.on_admitted = gate_placed;
    c.request.sample_override = gate_sample; c.request.sample_exclude = gate_exclude;
    printf("ctx=%d banks=%d prefill=%d images=%u patches=%u prompt=%d "
           "plain_samples=%d excluded_eos=%d excluded_eot=%d\n",
           GATE_CTX, GATE_BANKS, GATE_PREFILL, image_i, patches, prompt.len,
           GATE_TOKENS, c.eos, sample_eot_exclusion(&engine->vocab, c.eos));
    for (int pass = 0; pass < GATE_PASSES; pass++) {
        const char *arm = pass < GATE_PAIR_PASSES ? "0" : "1";
        REQUIRE(setenv("DS4_QWEN_VISION_QUAD", gate_control ? "1" : arm, 1) == 0);
        if (gate_control) {
            /* Compare one diagnostic change while retaining the quad path. */
            REQUIRE(setenv(gate_control, arm, 1) == 0);
            printf("control=%s value=%s\n", gate_control, arm);
        }
        c.features = file_open(argv[2], pass, "features.f32");
        c.logits = file_open(argv[2], pass, "logits.f32");
        c.admitted = c.done = c.sampled = 0;
        REQUIRE(ds4_engine_continuous_generate(c.ctx, gate_admit, NULL,
                    gate_done, &c, error, sizeof(error)) == 0);
        REQUIRE(c.done == 1 && c.sampled == GATE_TOKENS);
        REQUIRE(fclose(c.logits) == 0 && fclose(c.features) == 0);
        save_state(argv[2], pass, &c);
        printf("pass=%d queries=%d sampled=%d committed=%u\n", pass,
               !gate_control && pass < GATE_PAIR_PASSES ? 2 : 4, c.sampled,
               c.ctx->bank_hist_len[c.bank]);
        family_banked_reset(c.ctx, c.bank);
    }
    const char *kinds[] = {"features.f32", "logits.f32", "tokens.i32", "payload", "mrope"};
    for (size_t k = 0; k < sizeof(kinds) / sizeof(kinds[0]); k++) {
        for (int pass = 1; pass < GATE_PASSES; pass++) {
            files_equal(argv[2], 0, pass, kinds[k]);
        }
    }
    ds4_batch_ctx_destroy(c.ctx);
    ds4_tokens_free(&prompt); ds4_tokens_free(&raw);
    for (uint32_t i = 0; i < image_i; i++) { free(data[i]); }
    ds4_engine_close(engine); host_vocab_close(&host);
    puts("QWEN VISION HOST EXACT PASS");
    return 0;
}
