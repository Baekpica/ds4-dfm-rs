/* Snapshot ("IK21"): tokens, logits, then per layer the full layers' NVFP4
 * blocks [0, ceil(n / 128)) from wherever the SSD tier keeps them (device,
 * file or pinned window) or a sliding layer's last min(n, 1,023) ring rows
 * in position order, and the layer's convolution histories. "IK22" appends
 * the drafter's committed context rows in position order (word 8 per local
 * draft layer, word 9 for the global one), K then V per layer, and f_rnn.
 * Blocks and rows are placed by position, so a payload loads into a graph
 * of any chunk size or resident threshold that holds n; a global draft ring
 * shorter than the saved rows keeps their tail. */
enum { INKLING2_PAYLOAD_MAGIC = 0x31324b49, INKLING2_PAYLOAD_DRAFT = 0x32324b49 }; /* IK21, IK22 */
typedef enum { INKLING2_PAYLOAD_SAVE, INKLING2_PAYLOAD_LOAD } inkling2_payload_dir;

#ifndef __APPLE__
static uint32_t inkling2_payload_window(uint32_t n) { return n < DS4_INK2_WINDOW - 1u ? n : DS4_INK2_WINDOW - 1u; }

static uint32_t inkling2_payload_blocks(uint32_t n) { return (n + DS4_INK2_KV_BLOCK - 1u) / DS4_INK2_KV_BLOCK; }

static uint64_t inkling2_payload_body(uint32_t n, uint32_t magic, const uint32_t draft[2]) {
    uint64_t bytes = (uint64_t)n * sizeof(uint32_t) + (uint64_t)DS4_N_VOCAB * sizeof(float) +
        (uint64_t)INKLING2_FULL_LAYERS * inkling2_payload_blocks(n) * DS4_INK2_BLOCK_BYTES +
        (uint64_t)(INKLING2_LAYERS - INKLING2_FULL_LAYERS) * inkling2_payload_window(n) * INKLING2_KV_ROW +
        (uint64_t)INKLING2_LAYERS * INKLING2_HIST * sizeof(float);
    if (magic == INKLING2_PAYLOAD_DRAFT) {
        const uint64_t rows = (uint64_t)(DS4_INK2_DRAFT_LAYERS - 1u) * draft[0] + draft[1];
        bytes += rows * 2u * DS4_INK2_KV * sizeof(uint16_t) + (uint64_t)DS4_INK2_HIDDEN * sizeof(float);
    }
    return bytes;
}

/* A draft layer's committed context rows the next trial can still read: at
 * or above lo, within the ring less the trial block written first. */
static uint32_t inkling2_draft_kept(const inkling2_draft_rt *d, uint32_t l, uint32_t n) {
    const uint32_t span = d->ring[l] - DS4_INK2_DRAFT_BLOCK;
    uint32_t first = n > span ? n - span : 0u;
    if (first < d->lo[l]) { first = d->lo[l]; }
    return first < n ? n - first : 0u;
}

// Words 8 and 9: rows per local draft layer and of the global one.
static void inkling2_payload_draft_rows(const ds4_inkling2_graph *g, uint32_t n, uint32_t rows[2]) {
    rows[0] = rows[1] = 0u;
    if (!g->draft) { return; }
    rows[0] = UINT32_MAX;
    for (uint32_t l = 0; l < DS4_INK2_DRAFT_LAYERS; l++) {
        const uint32_t kept = inkling2_draft_kept(g->draft, l, n);
        if (!inkling2_draft_local(l)) {
            rows[1] = kept;
        } else if (kept < rows[0]) {
            rows[0] = kept;
        }
    }
}

// Tokens do not name the pixels behind image placeholders: no snapshot.
static uint64_t inkling2_payload_bytes(const ds4_inkling2_graph *g, uint32_t n) {
    if (!g || !g->ready || g->failed || !n || n > g->ctx_cap || g->position != n || g->live_n ||
        (g->draft && g->draft->trial_n)) { return 0; }
    uint32_t rows[2];
    inkling2_payload_draft_rows(g, n, rows);
    return DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) +
        inkling2_payload_body(n, g->draft ? INKLING2_PAYLOAD_DRAFT : INKLING2_PAYLOAD_MAGIC, rows);
}

static int inkling2_payload_span(inkling2_payload_dir dir, FILE *fp, ds4_gpu_tensor *t, uint64_t offset,
        uint64_t bytes, uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    if (!bytes) { return 0; }
    return dir == INKLING2_PAYLOAD_LOAD
        ? payload_read_tensor_span(fp, t, offset, bytes, buf, DS4_SESSION_IO_CHUNK, left, err, errlen)
        : payload_write_tensor_span(fp, t, offset, bytes, buf, DS4_SESSION_IO_CHUNK, err, errlen);
}

// Rows [n - rows, n) of a ring of `cap` slots (position p at p % cap).
static int inkling2_payload_ring(inkling2_payload_dir dir, FILE *fp, ds4_gpu_tensor *t, uint32_t cap, uint64_t row,
        uint32_t n, uint32_t rows, uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    const uint32_t slot = (n - rows) % cap, head = rows < cap - slot ? rows : cap - slot;
    return inkling2_payload_span(dir, fp, t, (uint64_t)slot * row, (uint64_t)head * row, buf, left, err, errlen) ||
        inkling2_payload_span(dir, fp, t, 0, (uint64_t)(rows - head) * row, buf, left, err, errlen);
}

static int inkling2_payload_skip(FILE *fp, uint64_t bytes, uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    while (bytes) {
        const uint64_t n = bytes < DS4_SESSION_IO_CHUNK ? bytes : DS4_SESSION_IO_CHUNK;
        if (payload_read_bytes(fp, buf, n, left, err, errlen)) { return 1; }
        bytes -= n;
    }
    return 0;
}

/* Full layer il's blocks [0, blocks): the device store below the resident
 * count, the tier's file below its window, the pinned window above. A load
 * has placed the window at the block holding n first. */
static int inkling2_payload_blocks_io(inkling2_payload_dir dir, FILE *fp, ds4_inkling2_graph *g, uint32_t il,
        uint32_t blocks, uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    inkling2_kv *kv = &g->tier;
    const uint64_t size = DS4_INK2_BLOCK_BYTES;
    const uint32_t resident = kv->spill && kv->resident < blocks ? kv->resident : blocks;
    if (inkling2_payload_span(dir, fp, g->kv[il], 0, (uint64_t)resident * size, buf, left, err, errlen)) { return 1; }
    if (resident == blocks) { return 0; }
    const uint32_t l = il / DS4_INK2_BLOCK, per = DS4_SESSION_IO_CHUNK / DS4_INK2_BLOCK_BYTES;
    const uint32_t filed = kv->window_block0 < blocks ? kv->window_block0 : blocks;
    for (uint32_t b = resident; b < filed; b += per) {
        const uint32_t count = filed - b < per ? filed - b : per;
        const uint64_t bytes = (uint64_t)count * size, at = inkling2_file_offset(kv, l, b);
        if (dir == INKLING2_PAYLOAD_LOAD) {
            if (payload_read_bytes(fp, buf, bytes, left, err, errlen)) { return 1; }
            if (!inkling2_kv_io(kv, true, buf, bytes, at)) {
                payload_set_err(err, errlen, "Inkling-2 K/V file write failed");
                return 1;
            }
        } else if (!inkling2_kv_io(kv, false, buf, bytes, at) || payload_write_bytes(fp, buf, bytes, err, errlen)) {
            payload_set_err(err, errlen, "Inkling-2 K/V snapshot write failed");
            return 1;
        }
    }
    if (filed == blocks) { return 0; }
    if (blocks - kv->window_block0 > kv->window_blocks) {
        payload_set_err(err, errlen, "Inkling-2 K/V window does not hold the frontier");
        return 1;
    }
    uint8_t *w = kv->window + ((uint64_t)l * kv->window_blocks + (filed - kv->window_block0)) * size;
    const uint64_t bytes = (uint64_t)(blocks - filed) * size;
    return dir == INKLING2_PAYLOAD_LOAD
        ? payload_read_bytes(fp, w, bytes, left, err, errlen)
        : payload_write_bytes(fp, w, bytes, err, errlen);
}

static int inkling2_payload_state(inkling2_payload_dir dir, FILE *fp, ds4_inkling2_graph *g, uint32_t n,
        uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    const uint32_t blocks = inkling2_payload_blocks(n), keep = inkling2_payload_window(n);
    inkling2_kv *kv = &g->tier;
    if (kv->spill) {
        inkling2_io_idle(kv);
        // A load restarts the window at the block holding n, as a rewind to n does.
        const uint32_t frontier = n / DS4_INK2_KV_BLOCK;
        if (dir == INKLING2_PAYLOAD_LOAD) { kv->window_block0 = frontier > kv->resident ? frontier : kv->resident; }
    }
    int rc = 0;
    for (uint32_t il = 0; !rc && il < INKLING2_LAYERS; il++) {
        const off_t start = ftello(fp);
        rc = (inkling2_global(il)
                ? inkling2_payload_blocks_io(dir, fp, g, il, blocks, buf, left, err, errlen)
                : inkling2_payload_ring(dir, fp, g->kv[il], g->ring_cap, INKLING2_KV_ROW, n, keep, buf, left, err,
                    errlen)) ||
            inkling2_payload_span(dir, fp, g->hist[il], 0, (uint64_t)INKLING2_HIST * sizeof(float), buf, left, err,
                errlen);
        // Nonseekable streams still carry the codec; only cache advice needs offsets.
        if (!rc && start >= 0) {
            rc = snapshot_io_trim(fp, (uint64_t)start,
                dir == INKLING2_PAYLOAD_LOAD ? SNAPSHOT_IO_READ : SNAPSHOT_IO_WRITE, err, errlen);
        }
    }
    return rc;
}

/* The drafter's committed context rows: per draft layer K then V of
 * [n - rows, n), then f_rnn. A load keeps what the graph's ring holds and
 * moves lo to the first kept row; a graph without a drafter reads past. */
static int inkling2_payload_draft(inkling2_payload_dir dir, FILE *fp, ds4_inkling2_graph *g, uint32_t n,
        const uint32_t rows[2], uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    inkling2_draft_rt *d = g->draft;
    const uint64_t row = (uint64_t)DS4_INK2_KV * sizeof(uint16_t), hidden = DS4_INK2_HIDDEN * sizeof(float);
    const off_t start = ftello(fp);
    int rc = 0;
    for (uint32_t l = 0; !rc && l < DS4_INK2_DRAFT_LAYERS; l++) {
        const uint32_t saved = rows[inkling2_draft_local(l) ? 0 : 1];
        uint32_t keep = d ? saved : 0u;
        if (d && keep > d->ring[l] - DS4_INK2_DRAFT_BLOCK) { keep = d->ring[l] - DS4_INK2_DRAFT_BLOCK; }
        ds4_gpu_tensor *ring[2] = {d ? d->ring_k[l] : NULL, d ? d->ring_v[l] : NULL};
        for (uint32_t kv = 0; !rc && kv < 2u; kv++) {
            rc = (dir == INKLING2_PAYLOAD_LOAD &&
                    inkling2_payload_skip(fp, (uint64_t)(saved - keep) * row, buf, left, err, errlen)) ||
                (keep && inkling2_payload_ring(dir, fp, ring[kv], d->ring[l], row, n, keep, buf, left, err, errlen));
        }
        if (!rc && d && dir == INKLING2_PAYLOAD_LOAD) { d->lo[l] = n - keep; }
    }
    if (!rc) {
        rc = d ? inkling2_payload_span(dir, fp, d->f_rnn, 0, hidden, buf, left, err, errlen)
               : inkling2_payload_skip(fp, hidden, buf, left, err, errlen);
    }
    if (!rc && start >= 0) {
        rc = snapshot_io_trim(fp, (uint64_t)start,
            dir == INKLING2_PAYLOAD_LOAD ? SNAPSHOT_IO_READ : SNAPSHOT_IO_WRITE, err, errlen);
    }
    return rc;
}

static int inkling2_payload_save(ds4_inkling2_graph *g, const int *tokens, uint32_t n, const float *logits, FILE *fp,
        char *err, size_t errlen) {
    if (!fp || !tokens || !logits || !inkling2_payload_bytes(g, n)) {
        payload_set_err(err, errlen, "invalid Inkling-2 snapshot state");
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB || tokens[i] == INKLING2_IMAGE_PAD) {
            payload_set_err(err, errlen, "invalid Inkling-2 snapshot token");
            return 1;
        }
    }
    for (uint32_t i = 0; i < DS4_INK2_TOKENS; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid Inkling-2 snapshot logits"); return 1; }
    }
    uint8_t *buf = aligned_alloc(4096, DS4_SESSION_IO_CHUNK);
    if (!buf || !ds4_gpu_synchronize()) {
        free(buf);
        payload_set_err(err, errlen, "Inkling-2 snapshot allocation or synchronization failed");
        return 1;
    }
    uint32_t rows[2];
    inkling2_payload_draft_rows(g, n, rows);
    const uint32_t magic = g->draft ? INKLING2_PAYLOAD_DRAFT : INKLING2_PAYLOAD_MAGIC;
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, g->ctx_cap, g->row_cap,
        INKLING2_LAYERS, magic, inkling2_payload_blocks(n), n, rows[0], rows[1],
        DS4_INK2_BLOCK_BYTES, DS4_N_VOCAB, n,
    };
    int rc = 0;
    for (uint32_t i = 0; !rc && i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) { rc = payload_write_u32(fp, h[i], err, errlen); }
    for (uint32_t i = 0; !rc && i < n; i++) { rc = payload_write_u32(fp, (uint32_t)tokens[i], err, errlen); }
    if (!rc) { rc = payload_write_logits(fp, logits, buf, err, errlen); }
    if (!rc) { rc = inkling2_payload_state(INKLING2_PAYLOAD_SAVE, fp, g, n, buf, NULL, err, errlen); }
    if (!rc && g->draft) { rc = inkling2_payload_draft(INKLING2_PAYLOAD_SAVE, fp, g, n, rows, buf, NULL, err, errlen); }
    free(buf);
    return rc;
}

static int inkling2_payload_restore(ds4_inkling2_graph *g, FILE *fp, uint64_t *left, const uint32_t *h,
        int **tokens_out, float *logits, char *err, size_t errlen) {
    if (!g || !fp || !left || !h || !tokens_out || !logits) { return 1; }
    /* A failed or partial restore can never be reused as a committed prefix. */
    g->failed = true;
    g->position = 0;
    g->live_n = 0;
    if (g->draft) {
        g->draft->trial_n = 0;
        g->draft->capture = false;
    }
    *tokens_out = NULL;
    const uint32_t n = h[7], rows[2] = {h[8], h[9]};
    const bool draft = h[5] == INKLING2_PAYLOAD_DRAFT;
    if (!g->ready || !n || n > g->ctx_cap || h[0] != DS4_SESSION_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_PAYLOAD_VERSION || h[2] < n || h[2] > INKLING2_CONTEXT || !h[3] || h[3] > h[2] ||
        h[4] != INKLING2_LAYERS || (!draft && h[5] != INKLING2_PAYLOAD_MAGIC) || h[6] != inkling2_payload_blocks(n) ||
        rows[0] > n || rows[0] > INKLING2_DRAFT_LOCAL_RING - DS4_INK2_DRAFT_BLOCK || rows[1] > n ||
        rows[1] > INKLING2_DRAFT_GLOBAL + 1u || (!draft && (rows[0] || rows[1])) || h[10] != DS4_INK2_BLOCK_BYTES ||
        h[11] != DS4_N_VOCAB || h[12] != n || *left != inkling2_payload_body(n, h[5], rows)) {
        payload_set_err(err, errlen, "incompatible Inkling-2 snapshot header or length");
        return 1;
    }
    int *tokens = malloc((size_t)n * sizeof(*tokens));
    uint8_t *buf = aligned_alloc(4096, DS4_SESSION_IO_CHUNK);
    if (!tokens || !buf || !ds4_gpu_synchronize()) {
        free(tokens);
        free(buf);
        payload_set_err(err, errlen, "Inkling-2 restore allocation or synchronization failed");
        return 1;
    }
    int rc = 0;
    for (uint32_t i = 0; !rc && i < n; i++) {
        uint32_t token = 0;
        rc = payload_read_u32(fp, &token, left, err, errlen);
        if (!rc && token >= DS4_N_VOCAB) { payload_set_err(err, errlen, "invalid Inkling-2 snapshot token"); rc = 1; }
        tokens[i] = (int)token;
    }
    if (!rc) { rc = payload_read_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float), left, err, errlen); }
    for (uint32_t i = 0; !rc && i < DS4_INK2_TOKENS; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid Inkling-2 snapshot logits"); rc = 1; }
    }
    if (!rc) { rc = inkling2_payload_state(INKLING2_PAYLOAD_LOAD, fp, g, n, buf, left, err, errlen); }
    if (!rc && draft) {
        rc = inkling2_payload_draft(INKLING2_PAYLOAD_LOAD, fp, g, n, rows, buf, left, err, errlen);
    } else if (!rc && g->draft) {
        // IK21 carries no context rows: the drafter's context restarts at n.
        for (uint32_t l = 0; l < DS4_INK2_DRAFT_LAYERS; l++) { g->draft->lo[l] = n; }
        if (!ds4_gpu_tensor_fill_f32(g->draft->f_rnn, 0.0f, DS4_INK2_HIDDEN)) {
            payload_set_err(err, errlen, "Inkling-2 drafter reset failed");
            rc = 1;
        }
    }
    if (!rc && (*left || !ds4_gpu_tensor_write(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) ||
                !ds4_gpu_synchronize())) {
        payload_set_err(err, errlen, "Inkling-2 restore finalization failed");
        rc = 1;
    }
    free(buf);
    if (rc) { free(tokens); return 1; }
    g->position = n;
    // The drafter's rings hold rows below n only.
    g->high = n;
    g->failed = false;
    *tokens_out = tokens;
    return 0;
}
#else
static uint64_t inkling2_payload_bytes(const ds4_inkling2_graph *g, uint32_t n) {
    (void)g; (void)n;
    return 0;
}

static int inkling2_payload_save(ds4_inkling2_graph *g, const int *tokens, uint32_t n, const float *logits, FILE *fp,
        char *err, size_t errlen) {
    (void)g; (void)tokens; (void)n; (void)logits; (void)fp;
    payload_set_err(err, errlen, "Inkling-2 snapshots are CUDA-only");
    return 1;
}

static int inkling2_payload_restore(ds4_inkling2_graph *g, FILE *fp, uint64_t *left, const uint32_t *h,
        int **tokens_out, float *logits, char *err, size_t errlen) {
    (void)g; (void)fp; (void)left; (void)h; (void)tokens_out; (void)logits;
    payload_set_err(err, errlen, "Inkling-2 snapshots are CUDA-only");
    return 1;
}
#endif
