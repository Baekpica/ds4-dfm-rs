/* Model-free MMX1/MMX2 codec gate: real production codec, host-backed device
 * tensors. Compile with section GC; no CUDA library or model is required. */
#ifdef DS4_NO_GPU
#error "MiniMax codec gate requires production graph declarations"
#endif
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "MiniMax payload FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

struct ds4_gpu_tensor { uint8_t *data; uint64_t bytes; };
static uint64_t writes, largest_transfer;
static int fail_write, fail_sync;

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t ? t->bytes : 0; }
int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t off, void *p, uint64_t n) {
    CHECK(n <= DS4_SESSION_IO_CHUNK);
    if (n > largest_transfer) { largest_transfer = n; }
    if (!t || off > t->bytes || n > t->bytes - off) { return 0; }
    memcpy(p, t->data + off, n);
    return 1;
}
int ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t off, const void *p, uint64_t n) {
    CHECK(n <= DS4_SESSION_IO_CHUNK);
    if (n > largest_transfer) { largest_transfer = n; }
    writes++;
    if (!t || off > t->bytes || n > t->bytes - off || (fail_write && --fail_write == 0)) { return 0; }
    memcpy(t->data + off, p, n);
    return 1;
}
int ds4_gpu_synchronize(void) { return !fail_sync; }

enum { POISON = 0xa5, CAP = 256, TOKENS = 3 };
typedef enum { DRAFT_ABSENT, DRAFT_PRESENT } draft_mode;
typedef enum { STREAM_SEEKABLE, STREAM_NONSEEKABLE } stream_mode;

#ifdef __linux__
static ssize_t stream_read(void *cookie, char *buf, size_t size) {
    FILE *fp = cookie;
    const size_t n = fread(buf, 1, size, fp);
    return n || !ferror(fp) ? (ssize_t)n : -1;
}

static ssize_t stream_write(void *cookie, const char *buf, size_t size) {
    const size_t n = fwrite(buf, 1, size, cookie);
    return n == size ? (ssize_t)n : -1;
}
#endif

static FILE *payload_stream(FILE *fp, const char *mode, stream_mode io) {
    if (io == STREAM_SEEKABLE) { return fp; }
#ifdef __linux__
    // Omit seek support while retaining a bounded, file-backed test stream.
    FILE *stream = fopencookie(fp, mode, (cookie_io_functions_t){.read = stream_read, .write = stream_write});
    CHECK(stream && ftello(stream) < 0);
    return stream;
#else
    (void)mode;
    CHECK(io != STREAM_NONSEEKABLE);
    return fp;
#endif
}

static ds4_gpu_tensor *tensor(uint64_t bytes, uint32_t seed) {
    ds4_gpu_tensor *t = calloc(1, sizeof(*t));
    CHECK(t);
    t->bytes = bytes;
    t->data = malloc(bytes);
    CHECK(t->data);
    for (uint64_t i = 0; i < bytes; i++) { t->data[i] = (uint8_t)((i * 29u + i / 257u + seed * 47u) % 251u); }
    return t;
}

static void release_tensor(ds4_gpu_tensor *t) {
    if (!t) { return; }
    free(t->data);
    free(t);
}

static minimax_draft_rt *draft(void) {
    minimax_draft_rt *d = calloc(1, sizeof(*d));
    CHECK(d);
    const uint64_t bytes = (uint64_t)MINIMAX_DRAFT_RING * MINIMAX_KV * sizeof(uint16_t);
    for (uint32_t l = 0; l < MINIMAX_DRAFT_LAYERS; l++) {
        d->ring_k[l] = tensor(bytes, 2u * l);
        d->ring_v[l] = tensor(bytes, 2u * l + 1u);
    }
    return d;
}

static void poison_draft(minimax_draft_rt *d) {
    if (!d) { return; }
    for (uint32_t l = 0; l < MINIMAX_DRAFT_LAYERS; l++) {
        memset(d->ring_k[l]->data, POISON, d->ring_k[l]->bytes);
        memset(d->ring_v[l]->data, POISON, d->ring_v[l]->bytes);
    }
    d->trial_n = 4;
    d->trial_pos = 19;
    memset(d->trial, POISON, sizeof(d->trial));
    d->draft_ms = d->verify_ms = 1.0;
}

static void release_draft(minimax_draft_rt *d) {
    if (!d) { return; }
    for (uint32_t l = 0; l < MINIMAX_DRAFT_LAYERS; l++) {
        release_tensor(d->ring_k[l]);
        release_tensor(d->ring_v[l]);
    }
    free(d);
}

static void check_ring(const ds4_gpu_tensor *src, const ds4_gpu_tensor *dst, uint32_t first, uint32_t rows) {
    const uint64_t row = (uint64_t)MINIMAX_KV * sizeof(uint16_t);
    for (uint32_t slot = 0; slot < MINIMAX_DRAFT_RING; slot++) {
        const uint8_t *got = dst->data + slot * row;
        const uint32_t distance = (slot + MINIMAX_DRAFT_RING - first % MINIMAX_DRAFT_RING) % MINIMAX_DRAFT_RING;
        if (distance < rows) {
            CHECK(!memcmp(src->data + slot * row, got, row));
            continue;
        }
        for (uint64_t i = 0; i < row; i++) { CHECK(got[i] == POISON); }
    }
}

static void ring_roundtrip(uint32_t n, uint32_t rows) {
    minimax_draft_rt *src = draft(), *dst = draft();
    poison_draft(dst);
    uint8_t *buf = aligned_alloc(4096, DS4_SESSION_IO_CHUNK);
    CHECK(buf);
    FILE *fp = tmpfile();
    CHECK(fp);
    char err[256] = {0};
    CHECK(!minimax_draft_payload(MINIMAX_PAYLOAD_SAVE, fp, src, n, rows, buf, NULL, err, sizeof(err)));
    uint64_t left = (uint64_t)ftello(fp);
    CHECK(left == minimax_payload_draft_bytes(rows));
    CHECK(left <= UINT64_C(40) * 1024 * 1024);
    rewind(fp);
    // Independent logical-row oracle: K then V per layer, never stale slots.
    for (uint32_t l = 0; l < MINIMAX_DRAFT_LAYERS; l++) {
        const ds4_gpu_tensor *rings[] = {src->ring_k[l], src->ring_v[l]};
        for (uint32_t kv = 0; kv < 2; kv++) {
            for (uint32_t p = n - rows; p < n; p++) {
                const size_t row = MINIMAX_KV * sizeof(uint16_t);
                CHECK(fread(buf, 1, row, fp) == row);
                CHECK(!memcmp(buf, rings[kv]->data + (uint64_t)(p % MINIMAX_DRAFT_RING) * row, row));
            }
        }
    }
    CHECK(fgetc(fp) == EOF);
    rewind(fp);
    CHECK(!minimax_draft_payload(MINIMAX_PAYLOAD_LOAD, fp, dst, n, rows, buf, &left, err, sizeof(err)));
    CHECK(left == 0);
    for (uint32_t l = 0; l < MINIMAX_DRAFT_LAYERS; l++) {
        check_ring(src->ring_k[l], dst->ring_k[l], n - rows, rows);
        check_ring(src->ring_v[l], dst->ring_v[l], n - rows, rows);
    }
    rewind(fp);
    left = minimax_payload_draft_bytes(rows);
    CHECK(!minimax_draft_payload(MINIMAX_PAYLOAD_LOAD, fp, NULL, n, rows, buf, &left, err, sizeof(err)));
    CHECK(left == 0 && (uint64_t)ftello(fp) == minimax_payload_draft_bytes(rows));
    writes = 0;
    CHECK(minimax_draft_payload(MINIMAX_PAYLOAD_LOAD, fp, dst, MINIMAX_DRAFT_WINDOW + 1,
        MINIMAX_DRAFT_WINDOW + 1, buf, &left, err, sizeof(err)));
    CHECK(!writes);
    fclose(fp);
    free(buf);
    release_draft(src);
    release_draft(dst);
}

static void graph_init(ds4_minimax_graph *g, draft_mode mode) {
    *g = (ds4_minimax_graph){.ready = true, .ctx_cap = CAP, .row_cap = 128, .position = TOKENS};
    g->logits = tensor((uint64_t)DS4_N_VOCAB * sizeof(float), 0);
    for (uint32_t l = 0; l < MINIMAX_LAYERS; l++) {
        g->index[l] = tensor(minimax_index_bytes(CAP), l);
        g->pages[l] = tensor(minimax_page_bytes(CAP), l);
    }
    if (mode == DRAFT_PRESENT) { g->draft = draft(); }
}

static void graph_free(ds4_minimax_graph *g) {
    release_tensor(g->logits);
    for (uint32_t l = 0; l < MINIMAX_LAYERS; l++) {
        release_tensor(g->index[l]);
        release_tensor(g->pages[l]);
    }
    release_draft(g->draft);
}

static void header(FILE *fp, uint64_t total, uint32_t *h, uint64_t *left) {
    rewind(fp);
    *left = total;
    char err[256] = {0};
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        CHECK(!payload_read_u32(fp, &h[i], left, err, sizeof(err)));
    }
}

static void roundtrip(draft_mode source, draft_mode target, uint32_t first, stream_mode io) {
    ds4_minimax_graph src, dst;
    graph_init(&src, source);
    graph_init(&dst, target);
    src.draft_first = first;
    dst.draft_first = 0;
    dst.row_cap = 1; // Scratch width is not part of the durable layout.
    dst.checkpoint_tag = 42;
    poison_draft(dst.draft);
    for (uint32_t l = 0; l < MINIMAX_LAYERS; l++) {
        memset(dst.index[l]->data, POISON, dst.index[l]->bytes);
        memset(dst.pages[l]->data, POISON, dst.pages[l]->bytes);
    }
    int tokens[TOKENS] = {7, 8, 9}, *got = NULL;
    float *logits = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *restored = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    CHECK(logits && restored);
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) { logits[i] = (float)((int)(i % 17u) - 8) * 0.125f; }
    char err[256] = {0};
    FILE *fp = tmpfile();
    CHECK(fp);
    FILE *stream = payload_stream(fp, "w", io);
    CHECK(!minimax_payload_save(&src, tokens, TOKENS, logits, stream, err, sizeof(err)));
    if (stream != fp) { CHECK(!fclose(stream)); }
    const uint64_t total = (uint64_t)ftello(fp);
    CHECK(total == minimax_payload_bytes(&src, TOKENS));
    uint64_t left;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    header(fp, total, h, &left);
    const uint32_t rows = source == DRAFT_PRESENT ? TOKENS - first : 0;
    CHECK(h[5] == (source == DRAFT_PRESENT ? MINIMAX_PAYLOAD_DRAFT : MINIMAX_PAYLOAD_MAGIC));
    CHECK(h[8] == rows);
    stream = payload_stream(fp, "r", io);
    CHECK(!minimax_payload_restore(&dst, stream, &left, h, &got, restored, err, sizeof(err)));
    if (stream != fp) { CHECK(!fclose(stream)); }
    CHECK(!left && !dst.failed && dst.position == TOKENS && !dst.checkpoint_tag);
    CHECK(dst.draft_first == (target == DRAFT_PRESENT ? TOKENS - rows : TOKENS));
    CHECK(!memcmp(got, tokens, sizeof(tokens)));
    CHECK(!memcmp(restored, logits, (size_t)DS4_N_VOCAB * sizeof(float)));
    CHECK(!memcmp(dst.logits->data, logits, (size_t)DS4_N_VOCAB * sizeof(float)));
    for (uint32_t l = 0; l < MINIMAX_LAYERS; l++) {
        CHECK(!memcmp(src.index[l]->data, dst.index[l]->data, minimax_index_bytes(TOKENS)));
        CHECK(!memcmp(src.pages[l]->data, dst.pages[l]->data, minimax_page_bytes(TOKENS)));
        CHECK(dst.index[l]->data[minimax_index_bytes(TOKENS)] == POISON);
        CHECK(dst.pages[l]->data[minimax_page_bytes(TOKENS)] == POISON);
    }
    if (dst.draft) {
        CHECK(!dst.draft->trial_n && !dst.draft->trial_pos && !dst.draft->draft_ms && !dst.draft->verify_ms);
        for (uint32_t i = 0; i < MINIMAX_DRAFT_BLOCK + 1; i++) { CHECK(!dst.draft->trial[i]); }
        for (uint32_t l = 0; l < MINIMAX_DRAFT_LAYERS; l++) {
            check_ring(src.draft ? src.draft->ring_k[l] : NULL, dst.draft->ring_k[l], first, rows);
            check_ring(src.draft ? src.draft->ring_v[l] : NULL, dst.draft->ring_v[l], first, rows);
        }
    }
    free(got);
    got = NULL;

    // Reject malformed metadata/length before the first device write.
    const uint32_t faults[][2] = {{0, 0}, {1, 0}, {2, TOKENS - 1}, {3, 0}, {4, 59}, {5, 0},
        {6, 2}, {7, 0}, {8, TOKENS + 1}, {8, MINIMAX_DRAFT_WINDOW + 1}, {9, 0}, {10, 0},
        {11, 0}, {12, 0}};
    for (size_t i = 0; i < sizeof(faults) / sizeof(*faults); i++) {
        uint32_t bad[DS4_SESSION_PAYLOAD_U32_FIELDS];
        header(fp, total, bad, &left);
        bad[faults[i][0]] = faults[i][1];
        writes = 0;
        CHECK(minimax_payload_restore(&dst, fp, &left, bad, &got, restored, err, sizeof(err)));
        CHECK(!writes && !got && dst.failed && !dst.position);
    }
    for (int delta = -1; delta <= 1; delta += 2) {
        header(fp, total, h, &left);
        left += delta;
        writes = 0;
        CHECK(minimax_payload_restore(&dst, fp, &left, h, &got, restored, err, sizeof(err)));
        CHECK(!writes && !got);
    }
    header(fp, total, h, &left);
    h[5] = MINIMAX_PAYLOAD_MAGIC;
    h[8] = 1;
    left = minimax_payload_body(TOKENS, 1);
    writes = 0;
    CHECK(minimax_payload_restore(&dst, fp, &left, h, &got, restored, err, sizeof(err)));
    CHECK(!writes && !got);
    if (dst.draft) {
        header(fp, total, h, &left);
        dst.draft->ring_v[MINIMAX_DRAFT_LAYERS - 1]->bytes--;
        writes = 0;
        CHECK(minimax_payload_restore(&dst, fp, &left, h, &got, restored, err, sizeof(err)));
        CHECK(!writes && !got);
        dst.draft->ring_v[MINIMAX_DRAFT_LAYERS - 1]->bytes++;
    }
    if (src.draft) {
        src.draft->trial_n = 1;
        CHECK(!minimax_payload_bytes(&src, TOKENS));
        CHECK(minimax_payload_save(&src, tokens, TOKENS, logits, fp, err, sizeof(err)));
        src.draft->trial_n = 0;
        src.draft_first = TOKENS + 1;
        CHECK(!minimax_payload_bytes(&src, TOKENS));
        src.draft_first = first;
        src.draft->ring_k[0]->bytes--;
        CHECK(!minimax_payload_bytes(&src, TOKENS));
        src.draft->ring_k[0]->bytes++;
    }
    header(fp, total, h, &left);
    fail_write = 1;
    CHECK(minimax_payload_restore(&dst, fp, &left, h, &got, restored, err, sizeof(err)));
    CHECK(dst.failed && !dst.position && !got);
    header(fp, total, h, &left);
    fail_sync = 1;
    writes = 0;
    CHECK(minimax_payload_restore(&dst, fp, &left, h, &got, restored, err, sizeof(err)));
    CHECK(!writes && dst.failed && !got);
    fail_sync = 0;
    CHECK(!ftruncate(fileno(fp), (off_t)total - 1));
    header(fp, total, h, &left);
    CHECK(minimax_payload_restore(&dst, fp, &left, h, &got, restored, err, sizeof(err)));
    CHECK(dst.failed && !dst.position && !got);
    fclose(fp);
    free(logits);
    free(restored);
    graph_free(&src);
    graph_free(&dst);
}

int main(void) {
    g_ds4_shape = DS4_SHAPE_MINIMAX_VQ;
    const uint32_t wrapped = 2u * MINIMAX_DRAFT_RING + 17u;
    ring_roundtrip(19, 7);
    ring_roundtrip(wrapped, MINIMAX_DRAFT_WINDOW);
    ring_roundtrip(wrapped, 25);
    ring_roundtrip(wrapped, 0);
    roundtrip(DRAFT_ABSENT, DRAFT_PRESENT, 0, STREAM_SEEKABLE);
    roundtrip(DRAFT_PRESENT, DRAFT_PRESENT, 1, STREAM_SEEKABLE);
    roundtrip(DRAFT_PRESENT, DRAFT_PRESENT, TOKENS, STREAM_SEEKABLE);
    roundtrip(DRAFT_PRESENT, DRAFT_ABSENT, 0, STREAM_SEEKABLE);
#ifdef __linux__
    roundtrip(DRAFT_PRESENT, DRAFT_PRESENT, 0, STREAM_NONSEEKABLE);
#endif
    CHECK(largest_transfer <= DS4_SESSION_IO_CHUNK);
    puts("MINIMAX_PAYLOAD_PASS: MMX1/MMX2, bounded committed rings, wrap, target-only restore, streams, faults");
    return 0;
}
