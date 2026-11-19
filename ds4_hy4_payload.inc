/* Logical token order makes the state independent of the prefill chunk.
 * Engram history comes from the saved tokens. A partially filled IndexPool
 * group additionally needs its raw operands and original spatial positions. */
enum { HY4_PAYLOAD = 0x31465948 }; /* "HYF1" */
typedef enum { HY4_SAVE, HY4_RESTORE, HY4_COPY } hy4_state_op;
typedef struct {
    hy4_state_op op;
    FILE *fp;
    uint64_t *left;
    uint8_t *buf;
    char *err;
    size_t len;
} hy4_state_io;

static int hy4_span(hy4_state_io *io, ds4_gpu_tensor *src, uint64_t off, ds4_gpu_tensor *dst, uint64_t to,
                    uint64_t n) {
    if (!n) {
        return 0;
    }
    if (io->op == HY4_COPY) {
        return !ds4_gpu_tensor_copy(dst, to, src, off, n);
    }
    if (io->op == HY4_RESTORE) {
        return payload_read_tensor_span(io->fp, dst, to, n, io->buf, DS4_SESSION_IO_CHUNK, io->left,
                                        io->err, io->len);
    }
    return payload_write_tensor_span(io->fp, src, off, n, io->buf, DS4_SESSION_IO_CHUNK, io->err, io->len);
}
static int hy4_ring_span(hy4_state_io *io, ds4_gpu_tensor *src, unsigned src_cap, ds4_gpu_tensor *dst,
                         unsigned dst_cap, unsigned pos, unsigned count, uint64_t width) {
    while (count) {
        const uint32_t from = pos % src_cap, to = pos % dst_cap;
        uint32_t n = src_cap - from;
        if (n > dst_cap - to) {
            n = dst_cap - to;
        }
        if (n > count) {
            n = count;
        }
        if (hy4_span(io, src, from * width, dst, to * width, n * width)) {
            return 1;
        }
        pos += n;
        count -= n;
    }
    return 0;
}

static int hy4_state_transfer(hy4_state_io *io, ds4_hy4_graph *src, ds4_hy4_graph *dst, uint32_t n) {
    if (hy4_span(io, src->seed, 0, dst->seed, 0, HY4_SEED_BYTES)) {
        return 1;
    }
    for (unsigned i = 0; i < HY4_LAYERS; i++) {
        ds4_hy4_cache *s = &src->layer[i], *d = &dst->layer[i];
        const bool sparse = hy4_sparse(i);
        const uint32_t rows = sparse || n < HY4_WINDOW ? n : HY4_WINDOW;
        if (s->fp8 != d->fp8 ||
            hy4_ring_span(io, s->latent, s->cap, d->latent, d->cap, n - rows, rows, s->fp8 ? 512 : 1024) ||
            (s->fp8 &&
             hy4_span(io, s->latent, (uint64_t)s->cap * 512, d->latent, (uint64_t)d->cap * 512, n)) ||
            hy4_ring_span(io, s->rope, s->cap, d->rope, d->cap, n - rows, rows, 128)) {
            return 1;
        }
        if (sparse && (hy4_span(io, s->groups, 0, d->groups, 0, (uint64_t)(n / 4) * 256) ||
                       hy4_ring_span(io, s->recent, s->recent_cap, d->recent, d->recent_cap, n - n % 4,
                                     n % 4, 259 * 4))) {
            return 1;
        }
    }
    return 0;
}

static uint64_t hy4_payload_body(const ds4_hy4_graph *g, uint32_t n) {
    const uint64_t sparse =
        (uint64_t)n * (g->layer[0].fp8 ? 641 : 1152) + (uint64_t)(n / 4) * 256 + (n % 4) * 259 * 4;
    const uint64_t window = (n < HY4_WINDOW ? n : HY4_WINDOW) * UINT64_C(1152);
    return HY4_SEED_BYTES + (uint64_t)n * 4 + 120832 * 4 + 12 * sparse + 30 * window;
}
static uint64_t hy4_payload_bytes(const ds4_hy4_graph *g, uint32_t n) {
    if (!g->ready || g->failed || g->media_count || g->trial_n || !n || n != g->position ||
        n > g->context) {
        return 0;
    }
    return DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) + hy4_payload_body(g, n);
}
static int hy4_payload_save(ds4_hy4_graph *g, const int *tokens, uint32_t n, const float *logits, FILE *fp,
                            char *err, size_t len) {
    if (!fp || !tokens || !logits || !hy4_payload_bytes(g, n)) {
        payload_set_err(err, len, "invalid Hy4 snapshot state");
        return 1;
    }
    for (unsigned i = 0; i < n; i++) {
        if ((unsigned)tokens[i] >= 120832) {
            payload_set_err(err, len, "invalid Hy4 snapshot token");
            return 1;
        }
    }
    for (unsigned i = 0; i < 120832; i++) {
        if (!isfinite(logits[i])) {
            payload_set_err(err, len, "invalid Hy4 snapshot logits");
            return 1;
        }
    }
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!buf || !ds4_gpu_synchronize()) {
        free(buf);
        payload_set_err(err, len, "Hy4 snapshot preparation failed");
        return 1;
    }
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {DS4_SESSION_PAYLOAD_MAGIC,
                                                        DS4_SESSION_PAYLOAD_VERSION,
                                                        g->context,
                                                        g->cap,
                                                        HY4_LAYERS,
                                                        HY4_PAYLOAD,
                                                        (uint32_t)g->layer[0].fp8,
                                                        n,
                                                        0,
                                                        4,
                                                        HY4_WINDOW,
                                                        120832,
                                                        n};
    int rc = 0;
    for (unsigned i = 0; !rc && i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        rc = payload_write_u32(fp, h[i], err, len);
    }
    for (unsigned i = 0; !rc && i < n; i++) {
        rc = payload_write_u32(fp, (uint32_t)tokens[i], err, len);
    }
    if (!rc) {
        rc = payload_write_logits(fp, logits, buf, err, len);
    }
    hy4_state_io io = {HY4_SAVE, fp, NULL, buf, err, len};
    if (!rc) {
        rc = hy4_state_transfer(&io, g, g, n);
    }
    free(buf);
    return rc;
}
static int hy4_payload_restore(ds4_hy4_graph *g, FILE *fp, uint64_t *left, const uint32_t *h,
                               int **tokens_out, float *logits, char *err, size_t len) {
    *tokens_out = NULL;
    g->failed = true;
    g->position = 0;
    g->media_tag = 0;
    g->media_count = 0;
    const uint32_t n = h[7];
    if (!g->ready || !n || n > g->context || h[0] != DS4_SESSION_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_PAYLOAD_VERSION || h[2] < n || h[2] > HY4_CONTEXT || !h[3] || h[3] > h[2] ||
        h[3] > 8192 || h[4] != HY4_LAYERS || h[5] != HY4_PAYLOAD || h[6] != (uint32_t)g->layer[0].fp8 ||
        h[8] || h[9] != 4 || h[10] != HY4_WINDOW || h[11] != 120832 || h[12] != n ||
        *left != hy4_payload_body(g, n)) {
        payload_set_err(err, len, "incompatible Hy4 snapshot header or length");
        return 1;
    }
    int *tokens = malloc((size_t)n * sizeof(*tokens));
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!buf || !tokens || !ds4_gpu_synchronize()) {
        free(buf);
        free(tokens);
        payload_set_err(err, len, "Hy4 restore preparation failed");
        return 1;
    }
    int rc = 0;
    for (unsigned i = 0; !rc && i < n; i++) {
        uint32_t t = 0;
        rc = payload_read_u32(fp, &t, left, err, len);
        if (!rc && t >= 120832) {
            payload_set_err(err, len, "invalid Hy4 snapshot token");
            rc = 1;
        }
        tokens[i] = (int)t;
    }
    if (!rc) {
        rc = payload_read_bytes(fp, logits, 120832 * sizeof(float), left, err, len);
    }
    for (unsigned i = 0; !rc && i < 120832; i++) {
        if (!isfinite(logits[i])) {
            payload_set_err(err, len, "invalid Hy4 snapshot logits");
            rc = 1;
        }
    }
    hy4_state_io io = {HY4_RESTORE, fp, left, buf, err, len};
    if (!rc) {
        rc = hy4_state_transfer(&io, g, g, n);
    }
    if (!rc && (*left || !ds4_gpu_tensor_write(g->logits, 0, logits, 120832 * sizeof(float)) ||
                !ds4_gpu_synchronize())) {
        payload_set_err(err, len, "Hy4 restore finalization failed");
        rc = 1;
    }
    free(buf);
    if (rc) {
        free(tokens);
        return 1;
    }
    *tokens_out = tokens;
    g->position = g->high = g->restored = n;
    g->failed = false;
    return 0;
}
