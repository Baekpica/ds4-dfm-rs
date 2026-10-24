/* Snapshot ("MMX1"): tokens, logits, then per layer the index keys of
 * positions [0, n) and the K/V pages of blocks [0, ceil(n / 128)). The last
 * page's slots past n hold whatever the graph had; nothing reads them before
 * a later token rewrites them. Pages are position-indexed, so a payload
 * loads into a graph of any chunk size. "MMX2" appends each draft layer's
 * K then V rows in chronological order, exactly [n - h[8], n). The bounded
 * committed tail excludes trial rows and uninitialized ring slots. */
enum { MINIMAX_PAYLOAD_MAGIC = 0x31584d4d, MINIMAX_PAYLOAD_DRAFT = 0x32584d4d }; /* MMX1, MMX2 */
typedef enum { MINIMAX_PAYLOAD_SAVE, MINIMAX_PAYLOAD_LOAD } minimax_payload_dir;

/* The runtime is defined after the session payload code. */
static bool minimax_draft_payload_ready(const minimax_draft_rt *d);
static bool minimax_draft_committed(const minimax_draft_rt *d);
static void minimax_draft_clear_trial(minimax_draft_rt *d);
static int minimax_draft_payload(minimax_payload_dir dir, FILE *fp, minimax_draft_rt *d,
        uint32_t n, uint32_t rows, uint8_t *buf, uint64_t *left, char *err, size_t errlen);

static uint64_t minimax_payload_draft_bytes(uint32_t rows) {
    return (uint64_t)rows * MINIMAX_DRAFT_LAYERS * 2u * MINIMAX_KV * sizeof(uint16_t);
}

static uint32_t minimax_payload_draft_rows(const ds4_minimax_graph *g, uint32_t n) {
    if (!g->draft || g->draft_first >= n) { return 0; }
    const uint32_t rows = n - g->draft_first;
    return rows < MINIMAX_DRAFT_WINDOW ? rows : MINIMAX_DRAFT_WINDOW;
}

static uint64_t minimax_payload_body(uint32_t n, uint32_t draft_rows) {
    return (uint64_t)n * sizeof(uint32_t) + (uint64_t)DS4_N_VOCAB * sizeof(float) +
        (uint64_t)MINIMAX_LAYERS * (minimax_index_bytes(n) + minimax_page_bytes(n)) +
        minimax_payload_draft_bytes(draft_rows);
}

static uint64_t minimax_payload_bytes(const ds4_minimax_graph *g, uint32_t n) {
    if (!g || !g->ready || g->failed || !n || n > g->ctx_cap || g->position != n ||
        (g->draft && (g->draft_first > n || !minimax_draft_committed(g->draft)))) { return 0; }
    return DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) +
        minimax_payload_body(n, minimax_payload_draft_rows(g, n));
}

static int minimax_payload_span(minimax_payload_dir dir, FILE *fp, ds4_gpu_tensor *t, uint64_t bytes, uint8_t *buf,
        uint64_t *left, char *err, size_t errlen) {
    if (!bytes) { return 0; }
    return dir == MINIMAX_PAYLOAD_LOAD
        ? payload_read_tensor_span(fp, t, 0, bytes, buf, DS4_SESSION_IO_CHUNK, left, err, errlen)
        : payload_write_tensor_span(fp, t, 0, bytes, buf, DS4_SESSION_IO_CHUNK, err, errlen);
}

/* Layer il's pages of blocks [0, ceil(n / 128)): device spans in the
 * resident tier, the SSD tier's file otherwise (the caller flushed dirty
 * pages before a save and drops pooled copies after a load). Both tiers
 * write the same bytes. */
static int minimax_payload_pages(minimax_payload_dir dir, FILE *fp, ds4_minimax_graph *g, uint32_t il, uint32_t n,
        uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    if (!g->kv.spill) {
        return minimax_payload_span(dir, fp, g->pages[il], minimax_page_bytes(n), buf, left, err, errlen);
    }
    const uint32_t blocks = (uint32_t)minimax_blocks(n), per = DS4_SESSION_IO_CHUNK / MINIMAX_KV_BLOCK;
    for (uint32_t b = 0; b < blocks; b += per) {
        const uint32_t count = blocks - b < per ? blocks - b : per;
        const uint64_t bytes = (uint64_t)count * MINIMAX_KV_BLOCK;
        if (dir == MINIMAX_PAYLOAD_LOAD) {
            if (payload_read_bytes(fp, buf, bytes, left, err, errlen)) { return 1; }
            if (!minimax_kv_pages(&g->kv, true, il, b, count, buf)) {
                payload_set_err(err, errlen, "MiniMax K/V file write failed");
                return 1;
            }
        } else if (!minimax_kv_pages(&g->kv, false, il, b, count, buf) || fwrite(buf, 1, bytes, fp) != bytes) {
            payload_set_err(err, errlen, "MiniMax K/V snapshot write failed");
            return 1;
        }
    }
    return 0;
}

static int minimax_payload_state(minimax_payload_dir dir, FILE *fp, ds4_minimax_graph *g, uint32_t n,
        uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    if (dir == MINIMAX_PAYLOAD_LOAD) {
        minimax_kv_drop(&g->kv);
    } else if (!minimax_kv_flush(&g->kv)) {
        payload_set_err(err, errlen, "MiniMax K/V flush failed");
        return 1;
    }
    int rc = 0;
    for (uint32_t il = 0; !rc && il < MINIMAX_LAYERS; il++) {
        const off_t start = ftello(fp);
        rc = minimax_payload_span(dir, fp, g->index[il], minimax_index_bytes(n), buf, left, err, errlen) ||
            minimax_payload_pages(dir, fp, g, il, n, buf, left, err, errlen);
        // Nonseekable streams still carry the codec; only cache advice needs offsets.
        if (!rc && start >= 0) { rc = snapshot_io_trim(fp, (uint64_t)start,
            dir == MINIMAX_PAYLOAD_LOAD ? SNAPSHOT_IO_READ : SNAPSHOT_IO_WRITE, err, errlen); }
    }
    return rc;
}

static int minimax_payload_save(ds4_minimax_graph *g, const int *tokens, uint32_t n,
        const float *logits, FILE *fp, char *err, size_t errlen) {
    if (!fp || !tokens || !logits || !minimax_payload_bytes(g, n)) {
        payload_set_err(err, errlen, "invalid MiniMax snapshot state");
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "invalid MiniMax snapshot token");
            return 1;
        }
        // Tokens alone do not name the pixels behind image and video pads.
        if (tokens[i] == 200025 || tokens[i] == 200026) {
            payload_set_err(err, errlen, "MiniMax snapshots refuse media prompts");
            return 1;
        }
    }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid MiniMax snapshot logits"); return 1; }
    }
    uint8_t *buf = aligned_alloc(4096, DS4_SESSION_IO_CHUNK);
    if (!buf || !ds4_gpu_synchronize()) {
        free(buf);
        payload_set_err(err, errlen, "MiniMax snapshot allocation or synchronization failed");
        return 1;
    }
    const uint32_t draft_rows = minimax_payload_draft_rows(g, n);
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, g->ctx_cap, g->row_cap,
        MINIMAX_LAYERS, g->draft ? MINIMAX_PAYLOAD_DRAFT : MINIMAX_PAYLOAD_MAGIC,
        (uint32_t)minimax_blocks(n), n, draft_rows, MINIMAX_FP4_ROW,
        MINIMAX_KV_PAGE, DS4_N_VOCAB, n,
    };
    int rc = 0;
    for (uint32_t i = 0; !rc && i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) { rc = payload_write_u32(fp, h[i], err, errlen); }
    for (uint32_t i = 0; !rc && i < n; i++) { rc = payload_write_u32(fp, (uint32_t)tokens[i], err, errlen); }
    if (!rc) { rc = payload_write_logits(fp, logits, buf, err, errlen); }
    if (!rc) { rc = minimax_payload_state(MINIMAX_PAYLOAD_SAVE, fp, g, n, buf, NULL, err, errlen); }
    if (!rc) { rc = minimax_draft_payload(MINIMAX_PAYLOAD_SAVE, fp, g->draft, n, draft_rows,
        buf, NULL, err, errlen); }
    free(buf);
    return rc;
}

static int minimax_payload_restore(ds4_minimax_graph *g, FILE *fp, uint64_t *left,
        const uint32_t *h, int **tokens_out, float *logits, char *err, size_t errlen) {
    if (!g || !fp || !left || !h || !tokens_out || !logits) { return 1; }
    /* A failed or partial restore can never be reused as a committed prefix. */
    g->failed = true;
    g->position = 0;
    g->checkpoint_tag = 0;
    minimax_draft_clear_trial(g->draft);
    *tokens_out = NULL;
    const uint32_t n = h[7], draft_rows = h[8];
    const bool draft_layout = h[5] == MINIMAX_PAYLOAD_DRAFT;
    if (!g->ready || !n || n > g->ctx_cap || h[0] != DS4_SESSION_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_PAYLOAD_VERSION || h[2] < n || h[2] > MINIMAX_CONTEXT || !h[3] || h[3] > h[2] ||
        h[4] != MINIMAX_LAYERS || (!draft_layout && h[5] != MINIMAX_PAYLOAD_MAGIC) ||
        h[6] != minimax_blocks(n) || (!draft_layout && draft_rows) || draft_rows > n ||
        draft_rows > MINIMAX_DRAFT_WINDOW ||
        h[9] != MINIMAX_FP4_ROW || h[10] != MINIMAX_KV_PAGE || h[11] != DS4_N_VOCAB || h[12] != n ||
        *left != minimax_payload_body(n, draft_rows) ||
        (g->draft && !minimax_draft_payload_ready(g->draft))) {
        payload_set_err(err, errlen, "incompatible MiniMax snapshot header or length");
        return 1;
    }
    int *tokens = malloc((size_t)n * sizeof(*tokens));
    uint8_t *buf = aligned_alloc(4096, DS4_SESSION_IO_CHUNK);
    if (!tokens || !buf || !ds4_gpu_synchronize()) {
        free(tokens);
        free(buf);
        payload_set_err(err, errlen, "MiniMax restore allocation or synchronization failed");
        return 1;
    }
    int rc = 0;
    for (uint32_t i = 0; !rc && i < n; i++) {
        uint32_t token = 0;
        rc = payload_read_u32(fp, &token, left, err, errlen);
        if (!rc && token >= DS4_N_VOCAB) { payload_set_err(err, errlen, "invalid MiniMax snapshot token"); rc = 1; }
        tokens[i] = (int)token;
    }
    if (!rc) { rc = payload_read_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float), left, err, errlen); }
    for (uint32_t i = 0; !rc && i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid MiniMax snapshot logits"); rc = 1; }
    }
    if (!rc) { rc = minimax_payload_state(MINIMAX_PAYLOAD_LOAD, fp, g, n, buf, left, err, errlen); }
    if (!rc) { rc = minimax_draft_payload(MINIMAX_PAYLOAD_LOAD, fp, g->draft, n, draft_rows,
        buf, left, err, errlen); }
    if (!rc && (*left || !ds4_gpu_tensor_write(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) ||
                !ds4_gpu_synchronize())) {
        payload_set_err(err, errlen, "MiniMax restore finalization failed");
        rc = 1;
    }
    free(buf);
    if (rc) { free(tokens); return 1; }
    // MMX1 has no draft context. MMX2 restores only committed rows; later
    // forwards and trial blocks overwrite everything outside this interval.
    g->draft_first = g->draft ? n - draft_rows : n;
    g->position = n;
    g->failed = false;
    *tokens_out = tokens;
    return 0;
}
