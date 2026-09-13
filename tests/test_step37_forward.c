/* Explicit token fixture, complete vocabulary dump, no tokenizer ambiguity.
 * This is a GPU integration gate, not an independent numerical oracle. */
static void trace(const char *, const void *, unsigned, unsigned, unsigned, unsigned);
#define STEP37_TRACE trace
#include "../ds4.c"

static void trace(const char *name, const void *tensor, unsigned width, unsigned rows,
                  unsigned layer, unsigned pos) {
    const char *dir = getenv("STEP37_TRACE");
    if (!dir || !*dir || pos) { return; }
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s-%u.bin", dir, name, layer);
    FILE *f = fopen(path, "wb");
    const size_t count = (size_t)width * rows;
    float *bytes = xmalloc(count * sizeof(float));
    const char *replay = getenv("STEP37_REPLAY");
    if (replay && !strcmp(name, "attn_norm_in")) {
        char source[1024];
        snprintf(source, sizeof(source), "%s/%s-%u.bin", replay, name, layer);
        FILE *input = fopen(source, "rb");
        if (!input || fread(bytes, sizeof(float), count, input) != count ||
            fgetc(input) != EOF || !ds4_gpu_tensor_write((ds4_gpu_tensor *)tensor,
                0, bytes, count * sizeof(float))) { ds4_die("Step layer replay failed"); }
        fclose(input);
    }
    if (!f || !ds4_gpu_tensor_read(tensor, 0, bytes, count * sizeof(float)) ||
        fwrite(bytes, sizeof(float), count, f) != count) { ds4_die("Step trace write failed"); }
    fclose(f); free(bytes);
}

static void kv_equal(const ds4_step37_graph *ring, const ds4_step37_graph *full) {
    const size_t stride = 2 * S37_KV * sizeof(uint16_t);
    for (unsigned il = 0; il < STEP37_LAYERS; il++) {
        unsigned rows = ring->position < ring->kv_cap[il] ? ring->position : ring->kv_cap[il];
        void *a = xmalloc(rows * stride), *b = xmalloc(full->position * stride);
        if (!ds4_gpu_tensor_read(ring->kv[il], 0, a, rows * stride) ||
            !ds4_gpu_tensor_read(full->kv[il], 0, b, full->position * stride)) {
            ds4_die("Step KV read failed");
        }
        for (unsigned t = ring->position - rows; t < ring->position; t++) {
            if (memcmp((char *)a + (t % ring->kv_cap[il]) * stride,
                       (char *)b + t * stride, stride)) {
                fprintf(stderr, "Step KV mismatch layer=%u position=%u\n", il, t);
                exit(1);
            }
        }
        free(a); free(b);
    }
}

static void kv_proof(ds4_step37_graph *g, const ds4_model *m, const ds4_weights *w,
                     const ds4_tokens *tokens, const float *logits) {
    if ((unsigned)tokens->len <= g->cap || tokens->len % g->cap) {
        ds4_die("Step KV proof requires multiple whole chunks");
    }
    ds4_step37_graph full;
    if (!step37_graph_alloc(&full, m, w, g->context, g->cap)) { ds4_die("Step proof allocation"); }
    // Same window and execution width, with every historical KV row retained.
    for (unsigned il = 0; il < STEP37_LAYERS; il++) {
        if (!step37_sliding(il)) { continue; }
        ds4_gpu_tensor_free(full.kv[il]);
        full.kv[il] = ds4_gpu_tensor_alloc((uint64_t)full.context * 2 * S37_KV * sizeof(uint16_t));
        full.kv_cap[il] = full.context;
        if (!full.kv[il]) { ds4_die("Step full KV allocation"); }
    }
    for (unsigned pos = 0; pos < (unsigned)tokens->len; pos += full.cap) {
        if (!step37_forward(&full, m, w, tokens->v + pos, full.cap, pos)) {
            ds4_die("Step full KV forward");
        }
    }
    float *actual = xmalloc(DS4_N_VOCAB * sizeof(float));
    if (!ds4_gpu_tensor_read(full.logits, 0, actual, DS4_N_VOCAB * sizeof(float)) ||
        memcmp(logits, actual, DS4_N_VOCAB * sizeof(float))) {
        ds4_die("Step ring/full logits differ");
    }
    kv_equal(g, &full);
    const unsigned end = g->position, start = end - g->cap;
    if (step37_rewind(g, end + 1) || (start > 1 && step37_rewind(g, start - 1)) || g->position != end ||
        !step37_rewind(g, start) || !step37_forward(g, m, w, tokens->v + start, g->cap, start) ||
        !ds4_gpu_tensor_read(g->logits, 0, actual, DS4_N_VOCAB * sizeof(float)) ||
        memcmp(logits, actual, DS4_N_VOCAB * sizeof(float))) {
        ds4_die("Step rewind logits differ");
    }
    kv_equal(g, &full);
    free(actual);
    step37_graph_free(&full);
    fprintf(stderr, "Step ring/full KV and chunk rewind: byte-exact logits + 45-layer KV PASS\n");
}

static int run_case(const ds4_model *m, const ds4_weights *w, const char *input_path,
                    const char *output_path, unsigned cap, unsigned decode) {
    fprintf(stderr, "Step case %s -> %s\n", input_path, output_path);
    FILE *input = fopen(input_path, "r");
    if (!input) { return 2; }
    ds4_tokens tokens = {0}; int token;
    while (fscanf(input, "%d", &token) == 1) {
        if (token < 0 || token >= 128896) { return 2; }
        ds4_tokens_push(&tokens, token);
    }
    fclose(input);
    if (!tokens.len) { return 2; }
    ds4_step37_graph g;
    unsigned ctx = tokens.len + decode + 1;
    if (cap > ctx) { cap = ctx; }
    if (!step37_graph_alloc(&g, m, w, ctx, cap)) { return 1; }
    FILE *out = fopen(output_path, "wb");
    if (!out) { return 2; }
    float *logits = xmalloc(DS4_N_VOCAB * sizeof(float));
    const double start = now_sec();
    for (unsigned pos = 0; pos < (unsigned)tokens.len;) {
        unsigned n = tokens.len - pos;
        if (n > cap) { n = cap; }
        if (!step37_forward(&g, m, w, tokens.v + pos, n, pos)) { return 1; }
        pos += n;
    }
    if (!ds4_gpu_tensor_read(g.logits, 0, logits, DS4_N_VOCAB * sizeof(float))) { return 1; }
    const double prefill_end = now_sec();
    fprintf(stderr, "Step prefill %d tokens %.6f seconds %.3f tok/s\n",
            tokens.len, prefill_end - start, tokens.len / (prefill_end - start));
    if (getenv("STEP37_VERIFY_KV")) { kv_proof(&g, m, w, &tokens, logits); }
    const double decode_start = now_sec();
    for (unsigned i = 0; i <= decode; i++) {
        int best = 0;
        for (unsigned j = 0; j < DS4_N_VOCAB; j++) {
            if (!isfinite(logits[j])) { ds4_die("nonfinite Step logits"); }
            if (logits[j] > logits[best]) { best = j; }
        }
        printf("%u %d %.9g\n", i, best, logits[best]);
        if (fwrite(logits, sizeof(float), DS4_N_VOCAB, out) != DS4_N_VOCAB) { return 1; }
        if (i == decode) { break; }
        if (!step37_forward(&g, m, w, &best, 1, tokens.len + i) ||
            !ds4_gpu_tensor_read(g.logits, 0, logits, DS4_N_VOCAB * sizeof(float))) { return 1; }
    }
    fprintf(stderr, "Step decode %u evaluations %.6f seconds\n", decode, now_sec() - decode_start);
    fclose(out); free(logits);
    step37_graph_free(&g);
    free(tokens.v);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s MAIN TOKENS.txt LOGITS.f32 CHUNK DECODE\n"
                        "       %s MAIN @CASES.txt - CHUNK DECODE\n", argv[0], argv[0]);
        return 2;
    }
    unsigned cap = (unsigned)strtoul(argv[4], NULL, 10);
    unsigned decode = (unsigned)strtoul(argv[5], NULL, 10);
    if (!cap || cap > 4096 || decode > 1024) { return 2; }
    ds4_model m;
    model_open(&m, argv[1], false, false);
    const ds4_host_shape host = {.variant = DS4_VARIANT_STEP37_FLASH};
    ds4_host_shape_install(&host);
    model_apply_host_shape();
    ds4_host_shape_clear();
    if (DS4_RMS_EPS != 1e-5f) { ds4_die("wrong Step RMS epsilon"); }
    const ds4_context_memory memory = step37_memory(262144, 512);
    if (memory.raw_bytes != (12ull * 262144 + 33ull * 1024) * 4096 ||
        step37_memory(262145, 512).total_bytes || step37_memory(512, 513).total_bytes) {
        ds4_die("Step context admission geometry mismatch");
    }
    (void)unsetenv("DS4_STEP37_PREFILL_CHUNK");
    if (step37_prefill_cap(4096) != 1024u || step37_prefill_cap(512) != 512u) {
        ds4_die("Step default prefill chunk mismatch");
    }
    if (setenv("DS4_STEP37_PREFILL_CHUNK", "512", 1) != 0 ||
        step37_prefill_cap(4096) != 512u ||
        unsetenv("DS4_STEP37_PREFILL_CHUNK") != 0) {
        ds4_die("Step prefill chunk restore mismatch");
    }
    ds4_weights w;
    weights_bind(&w, &m, false, 0, UINT32_MAX, true, false);
    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(m.map, m.size)) { return 1; }
    int rc;
    if (argv[2][0] != '@') {
        rc = run_case(&m, &w, argv[2], argv[3], cap, decode);
    } else {
        FILE *manifest = fopen(argv[2] + 1, "r");
        if (!manifest) { return 2; }
        char input[1024], output[1024];
        unsigned count = 0;
        rc = 0;
        for (;;) {
            int fields = fscanf(manifest, "%1023s %1023s", input, output);
            if (fields == EOF) { break; }
            if (fields != 2) { rc = 2; break; }
            count++;
            rc = run_case(&m, &w, input, output, cap, decode);
            if (rc) { break; }
        }
        fclose(manifest);
        if (!count) { rc = 2; }
    }
    model_close(&m);
    return rc;
}
