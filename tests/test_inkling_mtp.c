/* All eight actual BF16 draft blocks: source oracle and width/state parity. */
#include "../ds4.c"

enum { TEST_ROWS = 7, TEST_CONTEXT = 32 };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static void read_ref(const char *dir, const char *name, float *out, size_t count) {
    char path[1024];
    CHECK(snprintf(path, sizeof(path), "%s/%s", dir, name) < (int)sizeof(path));
    FILE *fp = fopen(path, "rb"); CHECK(fp);
    CHECK(fread(out, sizeof(float), count, fp) == count);
    CHECK(fgetc(fp) == EOF && fclose(fp) == 0);
}

static void compare_ref(const float *got, const float *want, size_t count, unsigned depth) {
    double err = 0, ref = 0, peak = 0, maximum = 0;
    for (size_t i = 0; i < count; i++) {
        CHECK(isfinite(got[i]) && isfinite(want[i]));
        const double delta = (double)got[i] - want[i];
        err += delta * delta; ref += (double)want[i] * want[i];
        peak = fmax(peak, fabs(want[i])); maximum = fmax(maximum, fabs(delta));
    }
    CHECK(ref > 0 && peak > 0);
    printf("draft %u: %zu values rel_rms=%g max/peak=%g\n",
           depth, count, sqrt(err / ref), maximum / peak);
    CHECK(sqrt(err / ref) < 0.01 && maximum / peak < 0.02);
}

static size_t read_state(const ds4_inkling_mtp_graph *d, unsigned char *out) {
    size_t offset = 0;
    for (unsigned i = 0; i < INKLING_DRAFT_LAYERS; i++) {
        const inkling_layer_state *s = &d->graph.layer[i];
        const unsigned valid = d->positions[i] < s->capacity ? d->positions[i] : s->capacity;
        size_t bytes = (size_t)valid * 2 * IK_KV * sizeof(uint16_t);
        CHECK(ds4_gpu_tensor_read(s->kv, 0, out + offset, bytes)); offset += bytes;
        for (unsigned j = 0; j < 4; j++) {
            bytes = IK_HISTORY * (j < 2 ? IK_KV : IK_HIDDEN) * sizeof(float);
            CHECK(ds4_gpu_tensor_read(s->conv[j], 0, out + offset, bytes)); offset += bytes;
        }
    }
    return offset;
}

static void write_state(ds4_inkling_mtp_graph *d, const unsigned char *in, unsigned rows) {
    size_t offset = 0;
    for (unsigned i = 0; i < INKLING_DRAFT_LAYERS; i++) {
        inkling_layer_state *s = &d->graph.layer[i];
        const unsigned valid = rows < s->capacity ? rows : s->capacity;
        size_t bytes = (size_t)valid * 2 * IK_KV * sizeof(uint16_t);
        CHECK(ds4_gpu_tensor_write(s->kv, 0, in + offset, bytes)); offset += bytes;
        for (unsigned j = 0; j < 4; j++) {
            bytes = IK_HISTORY * (j < 2 ? IK_KV : IK_HIDDEN) * sizeof(float);
            CHECK(ds4_gpu_tensor_write(s->conv[j], 0, in + offset, bytes)); offset += bytes;
        }
        d->positions[i] = rows;
    }
}

static void check_rollback(const ds4_model *m, const ds4_inkling_draft *w,
                            const float *seed, const float *embed) {
    enum { PREFIX = IK_LOCAL - 3, CONTEXT = IK_LOCAL + 17 };
    ds4_inkling_mtp_graph trial, baseline;
    CHECK(inkling_draft_alloc(&trial, m, w, CONTEXT, TEST_ROWS));
    CHECK(inkling_draft_alloc(&baseline, m, w, CONTEXT, TEST_ROWS));
    const size_t bytes = TEST_ROWS * IK_HIDDEN * sizeof(float);
    ds4_gpu_tensor *h = ds4_gpu_tensor_alloc(bytes), *e = ds4_gpu_tensor_alloc(bytes);
    CHECK(h && e && ds4_gpu_tensor_write(h, 0, seed, bytes) &&
          ds4_gpu_tensor_write(e, 0, embed, bytes));
    for (unsigned start = 0; start < PREFIX;) {
        const unsigned n = PREFIX - start < TEST_ROWS ? PREFIX - start : TEST_ROWS;
        for (unsigned i = 0; i < INKLING_DRAFT_LAYERS; i++) {
            CHECK(inkling_draft_forward(&trial, m, w, i, h, e, n));
        }
        start += n;
    }
    const size_t state_cap = INKLING_DRAFT_LAYERS *
        (CONTEXT * 2 * IK_KV * sizeof(uint16_t) +
         IK_HISTORY * (2 * IK_KV + 2 * IK_HIDDEN) * sizeof(float));
    unsigned char *prefix = xmalloc(state_cap), *got = xmalloc(state_cap), *want = xmalloc(state_cap);
    const size_t prefix_bytes = read_state(&trial, prefix);
    float *changed = xmalloc(bytes), *next = xmalloc(IK_HIDDEN * sizeof(float));
    float *next_ref = xmalloc(IK_HIDDEN * sizeof(float));
    CHECK(!inkling_graph_track(&trial.graph, 0));
    CHECK(!inkling_graph_track(&trial.graph, IK_VERIFY_ROWS + 1));
    CHECK(inkling_graph_track(&trial.graph, TEST_ROWS));
    CHECK(inkling_graph_track(&trial.graph, TEST_ROWS));
    CHECK(!inkling_graph_track(&trial.graph, TEST_ROWS - 1));
    for (unsigned keep = 0; keep <= TEST_ROWS; keep++) {
        write_state(&trial, prefix, PREFIX); write_state(&baseline, prefix, PREFIX);
        CHECK(read_state(&trial, got) == prefix_bytes && memcmp(prefix, got, prefix_bytes) == 0);
        CHECK(read_state(&baseline, got) == prefix_bytes && memcmp(prefix, got, prefix_bytes) == 0);
        memcpy(changed, seed, bytes);
        for (unsigned row = keep; row < TEST_ROWS; row++) {
            for (unsigned c = 0; c < IK_HIDDEN; c++) { changed[row * IK_HIDDEN + c] *= -3; }
        }
        for (unsigned i = 0; i < INKLING_DRAFT_LAYERS; i++) {
            CHECK(ds4_gpu_tensor_write(h, 0, changed, bytes));
            CHECK(inkling_draft_forward(&trial, m, w, i, h, e, TEST_ROWS));
            CHECK(ds4_gpu_tensor_write(h, 0, seed, bytes));
            if (keep) { CHECK(inkling_draft_forward(&baseline, m, w, i, h, e, keep)); }
            CHECK(!inkling_draft_keep(&trial, i, TEST_ROWS + 1));
            CHECK(inkling_draft_keep(&trial, i, keep));
            CHECK(!inkling_draft_keep(&trial, i, keep)); /* One journal entry is consumed once. */
            CHECK(trial.positions[i] == PREFIX + keep);
        }
        const size_t want_bytes = read_state(&baseline, want);
        CHECK(read_state(&trial, got) == want_bytes && memcmp(want, got, want_bytes) == 0);
        for (unsigned i = 0; i < INKLING_DRAFT_LAYERS; i++) {
            CHECK(inkling_draft_forward(&trial, m, w, i, h, e, 1));
            CHECK(ds4_gpu_tensor_read(trial.graph.buf[IK_X], 0, next, IK_HIDDEN * sizeof(float)));
            CHECK(inkling_draft_forward(&baseline, m, w, i, h, e, 1));
            CHECK(ds4_gpu_tensor_read(baseline.graph.buf[IK_X], 0, next_ref, IK_HIDDEN * sizeof(float)));
            CHECK(memcmp(next, next_ref, IK_HIDDEN * sizeof(float)) == 0);
        }
        printf("keep %u/7 across ring: all8 state and next hidden exact; changed rejects isolated\n", keep);
    }
    inkling_draft_free(&trial); inkling_draft_free(&baseline);
    ds4_gpu_tensor_free(h); ds4_gpu_tensor_free(e);
    free(prefix); free(got); free(want); free(changed); free(next); free(next_ref);
}

static void check_ring(const ds4_model *m, const ds4_inkling_draft *w,
                        const float *seed, const float *embed) {
    enum { RING_ROWS = IK_LOCAL + 17 };
    ds4_inkling_mtp_graph d;
    CHECK(inkling_draft_alloc(&d, m, w, RING_ROWS, TEST_ROWS));
    const size_t row_bytes = IK_HIDDEN * sizeof(float);
    const size_t bytes = RING_ROWS * row_bytes;
    float *hidden = xmalloc(bytes), *embeddings = xmalloc(bytes);
    float *full = xmalloc(bytes), *got = xmalloc(bytes);
    for (unsigned i = 0; i < RING_ROWS; i++) {
        memcpy(hidden + i * IK_HIDDEN, seed + (i % TEST_ROWS) * IK_HIDDEN, row_bytes);
        memcpy(embeddings + i * IK_HIDDEN, embed + (i % TEST_ROWS) * IK_HIDDEN, row_bytes);
    }
    ds4_gpu_tensor *h = ds4_gpu_tensor_alloc(bytes), *e = ds4_gpu_tensor_alloc(bytes);
    CHECK(h && e && ds4_gpu_tensor_write(h, 0, hidden, bytes) &&
          ds4_gpu_tensor_write(e, 0, embeddings, bytes));
    const size_t state_cap = INKLING_DRAFT_LAYERS *
        (RING_ROWS * 2 * IK_KV * sizeof(uint16_t) +
         IK_HISTORY * (2 * IK_KV + 2 * IK_HIDDEN) * sizeof(float));
    unsigned char *state = xmalloc(state_cap), *new_state = xmalloc(state_cap);
    size_t saved_bytes = 0;
    const unsigned widths[] = {TEST_ROWS, 1};
    for (unsigned test = 0; test < 2; test++) {
        CHECK(inkling_draft_reset(&d));
        for (unsigned start = 0; start < RING_ROWS;) {
            const unsigned n = RING_ROWS - start < widths[test] ? RING_ROWS - start : widths[test];
            ds4_gpu_tensor *hin = ds4_gpu_tensor_view(h, start * row_bytes, n * row_bytes);
            ds4_gpu_tensor *ein = ds4_gpu_tensor_view(e, start * row_bytes, n * row_bytes);
            for (unsigned i = 0; i < INKLING_DRAFT_LAYERS; i++) {
                CHECK(inkling_draft_forward(&d, m, w, i, i ? d.graph.buf[IK_X] : hin, ein, n));
            }
            CHECK(ds4_gpu_tensor_read(d.graph.buf[IK_X], 0, got + start * IK_HIDDEN, n * row_bytes));
            ds4_gpu_tensor_free(hin); ds4_gpu_tensor_free(ein); start += n;
        }
        const size_t read_bytes = read_state(&d, new_state);
        if (!test) {
            memcpy(full, got, bytes); memcpy(state, new_state, read_bytes); saved_bytes = read_bytes;
        } else {
            CHECK(memcmp(full, got, bytes) == 0);
            CHECK(read_bytes == saved_bytes && memcmp(state, new_state, saved_bytes) == 0);
        }
        CHECK(!inkling_draft_forward(&d, m, w, 0, h, e, 1));
        CHECK(read_state(&d, new_state) == saved_bytes && memcmp(state, new_state, saved_bytes) == 0);
        for (unsigned i = 0; i < INKLING_DRAFT_LAYERS; i++) { CHECK(d.positions[i] == RING_ROWS); }
        printf("ring width %u: %u rows, all draft hidden/state exact (%zu bytes), context bound preserved\n",
               widths[test], RING_ROWS, read_bytes);
    }
    inkling_draft_free(&d); ds4_gpu_tensor_free(h); ds4_gpu_tensor_free(e);
    free(hidden); free(embeddings); free(full); free(got); free(state); free(new_state);
}

int main(int argc, char **argv) {
    CHECK(argc == 3);
    const ds4_host_shape host = {.variant = DS4_VARIANT_INKLING_SMALL};
    ds4_host_shape_install(&host); model_apply_host_shape(); ds4_host_shape_clear();
    ds4_model m;
    model_open(&m, argv[1], false, false);
    ds4_inkling_draft w;
    inkling_bind_draft(&w, &m);
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(m.map, m.size));
    ds4_inkling_mtp_graph d;
    CHECK(inkling_draft_alloc(&d, &m, &w, TEST_CONTEXT, TEST_ROWS));
    const size_t count = TEST_ROWS * IK_HIDDEN, bytes = count * sizeof(float);
    float *seed = xmalloc(bytes), *embed = xmalloc(bytes), *want = xmalloc(bytes);
    float *full = xmalloc(bytes), *incremental = xmalloc(bytes);
    read_ref(argv[2], "hidden.f32", seed, count);
    read_ref(argv[2], "embeddings.f32", embed, count);
    ds4_gpu_tensor *h = ds4_gpu_tensor_alloc(bytes), *e = ds4_gpu_tensor_alloc(bytes);
    CHECK(h && e && ds4_gpu_tensor_write(h, 0, seed, bytes) &&
          ds4_gpu_tensor_write(e, 0, embed, bytes));
    for (unsigned i = 0; i < INKLING_DRAFT_LAYERS; i++) {
        CHECK(d.graph.layer[i].capacity == (i == 1 || i == 3 ? TEST_CONTEXT : IK_LOCAL));
        CHECK(inkling_draft_forward(&d, &m, &w, i, h, e, TEST_ROWS));
        CHECK(ds4_gpu_tensor_read(d.graph.buf[IK_X], 0, full, bytes));
        char name[64]; snprintf(name, sizeof(name), "depth-%u.f32", i);
        read_ref(argv[2], name, want, count); compare_ref(full, want, count, i);
        /* Isolate each layer's numerical error from preceding CPU/GPU drift. */
        CHECK(ds4_gpu_tensor_write(h, 0, want, bytes));
    }
    const size_t state_cap = INKLING_DRAFT_LAYERS *
        (TEST_ROWS * 2 * IK_KV * sizeof(uint16_t) +
         IK_HISTORY * (2 * IK_KV + 2 * IK_HIDDEN) * sizeof(float));
    unsigned char *state = xmalloc(state_cap), *got = xmalloc(state_cap);
    const unsigned widths[] = {TEST_ROWS, 1, 2, 3};
    size_t state_bytes = 0;
    for (unsigned test = 0; test < sizeof(widths) / sizeof(widths[0]); test++) {
        CHECK(inkling_draft_reset(&d));
        CHECK(ds4_gpu_tensor_write(h, 0, seed, bytes));
        for (unsigned start = 0; start < TEST_ROWS;) {
            unsigned n = TEST_ROWS - start < widths[test] ? TEST_ROWS - start : widths[test];
            ds4_gpu_tensor *hin = ds4_gpu_tensor_view(h, start * IK_HIDDEN * sizeof(float), n * IK_HIDDEN * sizeof(float));
            ds4_gpu_tensor *ein = ds4_gpu_tensor_view(e, start * IK_HIDDEN * sizeof(float), n * IK_HIDDEN * sizeof(float));
            for (unsigned i = 0; i < INKLING_DRAFT_LAYERS; i++) {
                CHECK(inkling_draft_forward(&d, &m, &w, i,
                      i ? d.graph.buf[IK_X] : hin, ein, n));
            }
            CHECK(ds4_gpu_tensor_read(d.graph.buf[IK_X], 0, incremental + start * IK_HIDDEN,
                                      n * IK_HIDDEN * sizeof(float)));
            ds4_gpu_tensor_free(hin); ds4_gpu_tensor_free(ein); start += n;
        }
        size_t got_bytes = read_state(&d, got);
        if (!test) {
            memcpy(full, incremental, bytes); memcpy(state, got, got_bytes); state_bytes = got_bytes;
        } else {
            CHECK(memcmp(full, incremental, bytes) == 0);
            CHECK(got_bytes == state_bytes && memcmp(state, got, state_bytes) == 0);
        }
        printf("width %u: all eight draft hidden/state chains match (%zu bytes)\n", widths[test], got_bytes);
    }
    CHECK(!inkling_draft_forward(&d, &m, &w, INKLING_DRAFT_LAYERS, h, e, 1));
    CHECK(!inkling_draft_forward(&d, &m, &w, 0, h, e, TEST_ROWS + 1));
    CHECK(read_state(&d, got) == state_bytes && memcmp(state, got, state_bytes) == 0);
    for (unsigned i = 0; i < INKLING_DRAFT_LAYERS; i++) { CHECK(d.positions[i] == TEST_ROWS); }
    inkling_draft_free(&d); ds4_gpu_tensor_free(h); ds4_gpu_tensor_free(e);
    check_ring(&m, &w, seed, embed);
    check_rollback(&m, &w, seed, embed);
    ds4_gpu_cleanup(); model_close(&m);
    free(seed); free(embed); free(want); free(full); free(incremental); free(state); free(got);
    puts("Inkling eight-layer draft component passed");
    return 0;
}
