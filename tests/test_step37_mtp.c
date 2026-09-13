/* Three actual Q8 predictors, independent heads, and retained KV rewind. */
static void trace(const char *, const void *, unsigned, unsigned, unsigned, unsigned);
#define STEP37_TRACE trace
#include "../ds4.c"

enum { TEST_ROWS = 7, TEST_CONTEXT = 32 };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "Step MTP FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static void trace(const char *name, const void *tensor, unsigned width, unsigned rows,
                    unsigned layer, unsigned pos) {
    const char *dir = getenv("STEP37_TRACE");
    if (!dir || !*dir || pos) { return; }
    char path[1024];
    CHECK(snprintf(path, sizeof(path), "%s/%s-%u.bin", dir, name, layer) < (int)sizeof(path));
    FILE *fp = fopen(path, "wb");
    const size_t bytes = (size_t)width * rows * sizeof(float);
    void *buffer = xmalloc(bytes);
    CHECK(fp && ds4_gpu_tensor_read(tensor, 0, buffer, bytes) &&
          fwrite(buffer, 1, bytes, fp) == bytes && fclose(fp) == 0);
    free(buffer);
}

static void read_ref(const char *dir, const char *name, float *out, size_t count) {
    char path[1024];
    CHECK(snprintf(path, sizeof(path), "%s/%s", dir, name) < (int)sizeof(path));
    FILE *fp = fopen(path, "rb"); CHECK(fp);
    CHECK(fread(out, sizeof(float), count, fp) == count);
    CHECK(fgetc(fp) == EOF && fclose(fp) == 0);
}

static void compare(const float *got, const float *want, size_t count) {
    double error = 0, reference = 0;
    for (size_t i = 0; i < count; i++) {
        CHECK(isfinite(got[i]) && isfinite(want[i]));
        error += ((double)got[i] - want[i]) * ((double)got[i] - want[i]);
        reference += (double)want[i] * want[i];
    }
    CHECK(reference > 0);
    const double relative = sqrt(error / reference);
    printf("%zu values rel_rms=%g\n", count, relative);
    CHECK(relative < 0.03);
}

static void ring(const ds4_model *m, const ds4_weights *w,
                   ds4_gpu_tensor *h, ds4_gpu_tensor *e, const float *seed) {
    enum { PREFIX = S37_WINDOW + TEST_ROWS + 5, CONTEXT = PREFIX + 2 * TEST_ROWS };
    ds4_step37_draft trial, control;
    CHECK(step37_draft_alloc(&trial, m, w, CONTEXT, TEST_ROWS) &&
          step37_draft_alloc(&control, m, w, CONTEXT, TEST_ROWS));
    const size_t bytes = TEST_ROWS * S37_HIDDEN * sizeof(float);
    CHECK(ds4_gpu_tensor_write(h, 0, seed, bytes));
    for (unsigned pos = 0; pos < PREFIX;) {
        unsigned n = PREFIX - pos < TEST_ROWS ? PREFIX - pos : TEST_ROWS;
        for (unsigned depth = 0; depth < STEP37_DRAFT_LAYERS; depth++) {
            CHECK(step37_draft_forward(&trial, m, w, depth, h, e, n));
        }
        pos += n;
    }
    const unsigned capacity = trial.graph.kv_cap[STEP37_LAYERS];
    const size_t row_bytes = 2 * S37_KV * sizeof(uint16_t);
    const size_t layer_bytes = capacity * row_bytes;
    unsigned char *checkpoint = xmalloc(STEP37_DRAFT_LAYERS * layer_bytes);
    unsigned char *got = xmalloc(layer_bytes), *want = xmalloc(layer_bytes);
    float *changed = xmalloc(bytes), *next = xmalloc(bytes), *next_ref = xmalloc(bytes);
    for (unsigned depth = 0; depth < STEP37_DRAFT_LAYERS; depth++) {
        CHECK(ds4_gpu_tensor_read(trial.graph.kv[STEP37_LAYERS + depth], 0,
                checkpoint + depth * layer_bytes, layer_bytes));
    }
    for (unsigned keep = 0; keep <= TEST_ROWS; keep++) {
        memcpy(changed, seed, bytes);
        for (unsigned i = keep * S37_HIDDEN; i < TEST_ROWS * S37_HIDDEN; i++) { changed[i] *= -3; }
        for (unsigned depth = 0; depth < STEP37_DRAFT_LAYERS; depth++) {
            const unsigned il = STEP37_LAYERS + depth;
            CHECK(ds4_gpu_tensor_write(trial.graph.kv[il], 0, checkpoint + depth * layer_bytes, layer_bytes));
            CHECK(ds4_gpu_tensor_write(control.graph.kv[il], 0, checkpoint + depth * layer_bytes, layer_bytes));
            trial.position[depth] = control.position[depth] = PREFIX;
            trial.high_water[depth] = control.high_water[depth] = PREFIX;
            CHECK(ds4_gpu_tensor_write(h, 0, changed, bytes));
            CHECK(step37_draft_forward(&trial, m, w, depth, h, e, TEST_ROWS));
            CHECK(ds4_gpu_tensor_write(h, 0, seed, bytes));
            CHECK(step37_draft_forward(&control, m, w, depth, h, e, TEST_ROWS));
            CHECK(step37_draft_rewind(&trial, depth, PREFIX + keep) &&
                  step37_draft_rewind(&control, depth, PREFIX + keep));
            CHECK(!step37_draft_logits(&trial, m, w, depth, TEST_ROWS));
            CHECK(!step37_draft_rewind(&trial, depth, PREFIX - 1));
            CHECK(ds4_gpu_tensor_read(trial.graph.kv[il], 0, got, layer_bytes) &&
                  ds4_gpu_tensor_read(control.graph.kv[il], 0, want, layer_bytes));
            /* Future lanes remain scratch; compare every causally live row. */
            for (unsigned pos = PREFIX + keep - S37_WINDOW; pos < PREFIX + keep; pos++) {
                size_t offset = (pos % capacity) * row_bytes;
                CHECK(!memcmp(got + offset, want + offset, row_bytes));
            }
            CHECK(step37_draft_forward(&trial, m, w, depth, h, e, 1));
            CHECK(ds4_gpu_tensor_read(trial.graph.ws.b_cur, 0, next, S37_HIDDEN * sizeof(float)));
            CHECK(step37_draft_forward(&control, m, w, depth, h, e, 1));
            CHECK(ds4_gpu_tensor_read(control.graph.ws.b_cur, 0, next_ref, S37_HIDDEN * sizeof(float)) &&
                  !memcmp(next, next_ref, S37_HIDDEN * sizeof(float)));
        }
        printf("keep %u/7: all three live ring KV and next hidden exact\n", keep);
    }
    step37_draft_free(&trial); step37_draft_free(&control);
    free(checkpoint); free(got); free(want); free(changed); free(next); free(next_ref);
}

int main(int argc, char **argv) {
    CHECK(argc == 3 || (argc == 4 && !strcmp(argv[3], "--local")));
    const bool local = argc == 4;
    const ds4_host_shape host = {.variant = DS4_VARIANT_STEP37_FLASH};
    ds4_host_shape_install(&host); model_apply_host_shape(); ds4_host_shape_clear();
    ds4_model m;
    model_open(&m, argv[1], false, false);
    ds4_weights w;
    step37_bind_draft(&w, &m);
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(m.map, m.size));
    ds4_step37_draft d;
    CHECK(step37_draft_alloc(&d, &m, &w, TEST_CONTEXT, TEST_ROWS));
    const size_t count = TEST_ROWS * S37_HIDDEN, bytes = count * sizeof(float);
    float *hidden = xmalloc(bytes), *embedding = xmalloc(bytes), *got = xmalloc(bytes), *ref = xmalloc(bytes);
    float *logits = xmalloc(DS4_N_VOCAB * sizeof(float)), *logits_ref = xmalloc(DS4_N_VOCAB * sizeof(float));
    ds4_gpu_tensor *h = ds4_gpu_tensor_alloc(bytes), *e = ds4_gpu_tensor_alloc(bytes);
    read_ref(argv[2], "hidden.f32", hidden, count);
    read_ref(argv[2], "embeddings.f32", embedding, count);
    CHECK(h && e && ds4_gpu_tensor_write(h, 0, hidden, bytes) &&
          ds4_gpu_tensor_write(e, 0, embedding, bytes));
    for (unsigned depth = 0; depth < STEP37_DRAFT_LAYERS; depth++) {
        char file[64];
        CHECK(step37_draft_forward(&d, &m, &w, depth, h, e, TEST_ROWS));
        CHECK(ds4_gpu_tensor_read(d.graph.ws.b_cur, 0, got, bytes));
        snprintf(file, sizeof(file), "depth-%u.f32", depth);
        read_ref(argv[2], file, ref, count);
        printf("depth %u hidden: ", depth); compare(got, ref, count);
        CHECK(step37_draft_logits(&d, &m, &w, depth, TEST_ROWS));
        CHECK(ds4_gpu_tensor_read(d.graph.logits, 0, logits, DS4_N_VOCAB * sizeof(float)));
        snprintf(file, sizeof(file), "logits-%u.f32", depth);
        read_ref(argv[2], file, logits_ref, DS4_N_VOCAB);
        printf("depth %u logits: ", depth); compare(logits, logits_ref, DS4_N_VOCAB);
        CHECK(argmax_f32(logits, DS4_N_VOCAB) == argmax_f32(logits_ref, DS4_N_VOCAB));
        /* Rewriting the same prefix must preserve both logits and every KV byte. */
        const size_t kv_bytes = TEST_ROWS * 2 * S37_KV * sizeof(uint16_t);
        void *before = xmalloc(kv_bytes), *after = xmalloc(kv_bytes);
        CHECK(ds4_gpu_tensor_read(d.graph.kv[STEP37_LAYERS + depth], 0, before, kv_bytes));
        CHECK(step37_draft_rewind(&d, depth, 0));
        CHECK(step37_draft_forward(&d, &m, &w, depth, h, e, TEST_ROWS));
        CHECK(ds4_gpu_tensor_read(d.graph.ws.b_cur, 0, ref, bytes) && !memcmp(ref, got, bytes));
        CHECK(ds4_gpu_tensor_read(d.graph.kv[STEP37_LAYERS + depth], 0, after, kv_bytes) &&
              !memcmp(before, after, kv_bytes));
        free(before); free(after);
        /* Local arithmetic uses the same input at each depth. Keep the full
         * quantized chain as a separate, stricter trajectory diagnostic. */
        if (local) {
            snprintf(file, sizeof(file), "depth-%u.f32", depth);
            read_ref(argv[2], file, ref, count);
        }
        CHECK(ds4_gpu_tensor_write(h, 0, local ? ref : got, bytes));
    }
    CHECK(!step37_draft_forward(&d, &m, &w, STEP37_DRAFT_LAYERS, h, e, 1));
    step37_draft_free(&d);
    ring(&m, &w, h, e, hidden);
    ds4_gpu_tensor_free(h); ds4_gpu_tensor_free(e);
    ds4_gpu_cleanup(); model_close(&m);
    free(hidden); free(embedding); free(got); free(ref); free(logits); free(logits_ref);
    puts("Step MTP: three Q8 predictors, separate heads and rewind PASS");
    return 0;
}
