/* Custom Q8_0 payload includes bounded MTP history and the final target hidden
 * row. Rejected speculative writes never enter a snapshot. */
enum { IQ_PAYLOAD_MAGIC = 0x31515149, IQ_ROW_BYTES = IQ_Q8_ROW_BLOCKS * sizeof(iquest_q8) };
typedef enum { IQ_PAYLOAD_SAVE, IQ_PAYLOAD_LOAD } iquest_payload_dir;

static unsigned iquest_saved_rows(unsigned il, unsigned n) {
    return iquest_full(il) || n < IQ_WINDOW ? n : IQ_WINDOW;
}

static uint64_t iquest_payload_body(unsigned n, iquest_mtp_mode mtp) {
    uint64_t bytes = (uint64_t)n * sizeof(uint32_t) + (uint64_t)DS4_N_VOCAB * sizeof(float);
    for (unsigned il = 0; il < IQ_LAYERS; il++) {
        bytes += (uint64_t)iquest_saved_rows(il, n) *
            IQ_ROW_BYTES;
    }
    bytes += IQ_EMBED * sizeof(float);
    if (mtp == IQ_MTP_ON) { bytes += (uint64_t)((n - 1) < IQ_DRAFT_WINDOW ? n - 1 : IQ_DRAFT_WINDOW) * IQ_ROW_BYTES; }
    return bytes;
}

static bool iquest_payload_layout(const ds4_iquest_graph *g, unsigned n) {
    if (!g || !g->cap || g->cap > IQ_PREFILL_MAX || g->cap > g->context || !n || n > g->context || g->context > IQ_CONTEXT) { return false; }
    for (unsigned il = 0; il < IQ_LAYERS; il++) {
        if (!g->kv[il] || g->kv_cap[il] != iquest_kv_cap(il, g->context, g->cap)) { return false; }
    }
    return g->logits != NULL;
}

static uint64_t iquest_payload_bytes(const ds4_iquest_graph *g, unsigned n) {
    if (!iquest_payload_layout(g, n) || g->failed || g->trial_n || g->position != n || (g->mtp_enabled && g->mtp_position != n - 1)) { return 0; }
    return DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) +
        iquest_payload_body(n, g->mtp_enabled ? IQ_MTP_ON : IQ_MTP_OFF);
}

static int iquest_payload_ring(iquest_payload_dir dir, FILE *fp, ds4_iquest_graph *g,
                              unsigned il, unsigned n, uint8_t *buf,
                              uint64_t *left, char *err, size_t errlen) {
    unsigned rows = iquest_saved_rows(il, n), first = n - rows;
    const uint64_t row = IQ_ROW_BYTES;
    while (rows) {
        const unsigned slot = first % g->kv_cap[il];
        unsigned span = g->kv_cap[il] - slot;
        if (span > rows) { span = rows; }
        const int rc = dir == IQ_PAYLOAD_LOAD
            ? payload_read_tensor_span(fp, g->kv[il], slot * row, span * row,
                                        buf, DS4_SESSION_IO_CHUNK, left, err, errlen)
            : payload_write_tensor_span(fp, g->kv[il], slot * row, span * row,
                                         buf, DS4_SESSION_IO_CHUNK, err, errlen);
        if (rc) { return rc; }
        first += span;
        rows -= span;
    }
    return 0;
}

static int iquest_payload_extra(iquest_payload_dir dir, FILE *fp,
        ds4_iquest_graph *g, unsigned n, uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    int rc = dir == IQ_PAYLOAD_LOAD
        ? payload_read_tensor_span(fp, g->carry, 0, IQ_EMBED * sizeof(float), buf, DS4_SESSION_IO_CHUNK, left, err, errlen)
        : payload_write_tensor_span(fp, g->carry, 0, IQ_EMBED * sizeof(float), buf, DS4_SESSION_IO_CHUNK, err, errlen);
    if (rc || !g->mtp_enabled) { return rc; }
    const unsigned end = n - 1, first = end > IQ_DRAFT_WINDOW ? end - IQ_DRAFT_WINDOW : 0;
    const unsigned capacity = IQ_DRAFT_WINDOW + IQ_DRAFT_SLOTS;
    for (unsigned p = first; !rc && p < end;) {
        const unsigned slot = p % capacity;
        unsigned rows = capacity - slot;
        if (rows > end - p) { rows = end - p; }
        rc = dir == IQ_PAYLOAD_LOAD
            ? payload_read_tensor_span(fp, g->mtp_kv, (uint64_t)slot * IQ_ROW_BYTES,
                (uint64_t)rows * IQ_ROW_BYTES, buf, DS4_SESSION_IO_CHUNK, left, err, errlen)
            : payload_write_tensor_span(fp, g->mtp_kv, (uint64_t)slot * IQ_ROW_BYTES,
                (uint64_t)rows * IQ_ROW_BYTES, buf, DS4_SESSION_IO_CHUNK, err, errlen);
        p += rows;
    }
    return rc;
}

static int iquest_payload_save(ds4_iquest_graph *g, const int *tokens, unsigned n,
                              const float *logits, FILE *fp, char *err, size_t errlen) {
    if (!fp || !tokens || !logits || !iquest_payload_bytes(g, n)) {
        payload_set_err(err, errlen, "invalid IQuest-Q1 snapshot state"); return 1;
    }
    for (unsigned i = 0; i < n; i++) {
        if (tokens[i] < 0 || (unsigned)tokens[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "invalid IQuest-Q1 snapshot token"); return 1;
        }
    }
    for (unsigned i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid IQuest-Q1 snapshot logits"); return 1; }
    }
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!buf || !ds4_gpu_synchronize()) {
        free(buf); payload_set_err(err, errlen, "IQuest-Q1 snapshot allocation or synchronization failed"); return 1;
    }
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, g->context, g->cap,
        IQ_LAYERS, IQ_PAYLOAD_MAGIC, IQ_ROW_BYTES, n, g->mtp_enabled, sizeof(iquest_q8),
        IQ_WINDOW, DS4_N_VOCAB, n,
    };
    int rc = 0;
    for (unsigned i = 0; !rc && i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) { rc = payload_write_u32(fp, h[i], err, errlen); }
    for (unsigned i = 0; !rc && i < n; i++) { rc = payload_write_u32(fp, (uint32_t)tokens[i], err, errlen); }
    if (!rc) { rc = payload_write_logits(fp, logits, buf, err, errlen); }
    for (unsigned il = 0; !rc && il < IQ_LAYERS; il++) {
        rc = iquest_payload_ring(IQ_PAYLOAD_SAVE, fp, g, il, n, buf, NULL, err, errlen);
    }
    if (!rc) { rc = iquest_payload_extra(IQ_PAYLOAD_SAVE, fp, g, n, buf, NULL, err, errlen); }
    free(buf);
    return rc;
}

static void iquest_payload_drop_draft(ds4_iquest_graph *g) {
    g->trial_n = 0; g->mtp_position = 0;
}

static int iquest_payload_restore(ds4_iquest_graph *g, FILE *fp, uint64_t *left,
                                 const uint32_t *h, int **tokens_out, float *logits,
                                 char *err, size_t errlen) {
    if (!g || !fp || !left || !h || !tokens_out || !logits) { return 1; }
    /* A failed/partial restore can never be reused as a committed prefix. */
    g->failed = true;
    g->position = 0;
    iquest_payload_drop_draft(g);
    *tokens_out = NULL;
    const unsigned n = h[7];
    if (!iquest_payload_layout(g, n) || h[0] != DS4_SESSION_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_PAYLOAD_VERSION || h[2] < n || h[2] > IQ_CONTEXT ||
        !h[3] || h[3] > IQ_PREFILL_MAX || h[3] > h[2] || h[4] != IQ_LAYERS || h[5] != IQ_PAYLOAD_MAGIC ||
        h[6] != IQ_ROW_BYTES || h[8] != (unsigned)g->mtp_enabled || h[9] != sizeof(iquest_q8) ||
        h[10] != IQ_WINDOW || h[11] != DS4_N_VOCAB || h[12] != n ||
        *left != iquest_payload_body(n, g->mtp_enabled ? IQ_MTP_ON : IQ_MTP_OFF)) {
        payload_set_err(err, errlen, "incompatible IQuest-Q1 snapshot header or length"); return 1;
    }
    int *tokens = malloc((size_t)n * sizeof(*tokens));
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!tokens || !buf || !ds4_gpu_synchronize()) {
        free(tokens); free(buf); payload_set_err(err, errlen, "IQuest-Q1 restore allocation or synchronization failed"); return 1;
    }
    int rc = 0;
    for (unsigned i = 0; !rc && i < n; i++) {
        uint32_t token = 0;
        rc = payload_read_u32(fp, &token, left, err, errlen);
        if (!rc && token >= DS4_N_VOCAB) { payload_set_err(err, errlen, "invalid IQuest-Q1 snapshot token"); rc = 1; }
        tokens[i] = (int)token;
    }
    if (!rc) { rc = payload_read_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float), left, err, errlen); }
    for (unsigned i = 0; !rc && i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid IQuest-Q1 snapshot logits"); rc = 1; }
    }
    for (unsigned il = 0; !rc && il < IQ_LAYERS; il++) {
        rc = iquest_payload_ring(IQ_PAYLOAD_LOAD, fp, g, il, n, buf, left, err, errlen);
    }
    if (!rc) { rc = iquest_payload_extra(IQ_PAYLOAD_LOAD, fp, g, n, buf, left, err, errlen); }
    if (!rc && (*left || !ds4_gpu_tensor_write(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) ||
                !ds4_gpu_synchronize())) { payload_set_err(err, errlen, "IQuest-Q1 restore finalization failed"); rc = 1; }
    free(buf);
    if (rc) { free(tokens); return 1; }
    g->position = n;
    g->mtp_position = g->mtp_enabled ? n - 1 : 0;
    g->failed = false;
    *tokens_out = tokens;
    return 0;
}
