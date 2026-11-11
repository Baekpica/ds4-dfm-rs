/* Bonsai (qwen35) session payload.
 *
 * The trunk state is positional fp16 k/v rows on the gated-attention layers
 * (one row per position, no ring) plus, on every gated delta-net layer, the
 * recurrence and its conv history.  Both are whole-prefix functions: the
 * payload describes exactly the tokens it carries, so a restore is valid only
 * for a prompt that extends them -- the session compares the checkpoint's
 * tokens before it reuses one (docs/rust-migration/ARCHITECTURE.md, recurrent
 * state).
 *
 * The grafted MTP block's own k/v prefix and the hidden row it holds travel
 * with the payload: without them a restored session would draft over rows it
 * never wrote.  The block is stored at its rest point (one pending row awaiting
 * its sampled token), which is where a finished request leaves it; a session
 * served without the drafter has no block row and stores none (header word 6
 * is then zero).
 *
 * Header (13 u32, the shared session-payload layout):
 *   0 magic, 1 version, 2 k/v rows the graph allocated, 3 gated-attention
 *   layers, 4 layers, 5 layout magic, 6 block rows saved (0 without a block),
 *   7 tokens, 8 k+v bytes per row, 9 recurrence bytes, 10 conv-history bytes,
 *   11 vocab, 12 tokens.
 * Body: tokens, logits, per gated-attention layer k then v, per gated delta-net
 * layer recurrence then history, then -- when block rows are saved -- the
 * pending flag, the held hidden row and the block's k then v.
 */

enum { QWEN35_PAYLOAD_LAYOUT = UINT32_C(0x314e5342) }; /* "BSN1" */

typedef enum { QWEN35_PAYLOAD_SAVE, QWEN35_PAYLOAD_LOAD } qwen35_payload_dir;

static uint64_t qwen35_payload_kv_half(void) {
    return (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(uint16_t);
}

/* One row of one tensor (k or v) per position. */
static uint64_t qwen35_payload_kv_row(void) {
    return 2u * qwen35_payload_kv_half();
}

static uint32_t qwen35_payload_count(const ds4_qwen35_gpu_graph *g, bool linear) {
    uint32_t n = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (linear ? g->lin_state[il] != NULL : g->k_cache[il] != NULL) { n++; }
    }
    return n;
}

/* The artifact carries the grafted block, so the graph owns its k/v rows. */
static bool qwen35_payload_block(const ds4_qwen35_gpu_graph *g) {
    return g->k_cache[DS4_QWEN35_MTP_LAYER] != NULL;
}

/* Block rows a payload carries: its k/v prefix only exists at the block's rest
 * point (one pending row awaiting its token).  A session that never drafted --
 * MTP off, so the fill never ran -- has written none and stores none. */
static uint32_t qwen35_payload_block_rows(const ds4_qwen35_gpu_graph *g) {
    if (!qwen35_payload_block(g) || !g->mtp_pending || g->mtp_pos + 1u != g->pos) {
        return 0u;
    }
    return g->mtp_pos;
}

/* Whether the live state is a rest point a payload can describe: the trunk
 * ends at `n` and no verify is in flight. */
static bool qwen35_payload_rest(const ds4_qwen35_gpu_graph *g, uint32_t n) {
    return g && g->h && n && g->ctx_cap >= n && g->pos == n && !g->trial_n;
}

static uint64_t qwen35_payload_body(const ds4_qwen35_gpu_graph *g, uint32_t n,
                                    uint32_t rows) {
    uint64_t bytes = (uint64_t)n * sizeof(uint32_t) +
                     (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += (uint64_t)qwen35_payload_count(g, false) * n * qwen35_payload_kv_row();
    bytes += (uint64_t)qwen35_payload_count(g, true) *
             (g->lin_state_bytes + g->lin_hist_bytes);
    if (rows) {
        bytes += sizeof(uint32_t) + (uint64_t)DS4_N_EMBD * sizeof(float) +
                 rows * qwen35_payload_kv_row();
    }
    return bytes;
}

static uint64_t qwen35_payload_bytes(const ds4_qwen35_gpu_graph *g, uint32_t n) {
    if (!qwen35_payload_rest(g, n)) { return 0; }
    return (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) +
           qwen35_payload_body(g, n, qwen35_payload_block_rows(g));
}

static int qwen35_payload_span(qwen35_payload_dir dir, FILE *fp,
                               ds4_gpu_tensor *tensor, uint64_t bytes,
                               uint8_t *buf, uint64_t *left,
                               char *err, size_t errlen) {
    if (!bytes) { return 0; }
    return dir == QWEN35_PAYLOAD_LOAD
        ? payload_read_tensor_span(fp, tensor, 0, bytes, buf, DS4_SESSION_IO_CHUNK,
                                   left, err, errlen)
        : payload_write_tensor_span(fp, tensor, 0, bytes, buf, DS4_SESSION_IO_CHUNK,
                                    err, errlen);
}

/* One layer's k then v, or its recurrence then its conv history. */
static int qwen35_payload_state(qwen35_payload_dir dir, FILE *fp,
                                ds4_qwen35_gpu_graph *g, uint32_t il,
                                uint32_t rows, uint8_t *buf, uint64_t *left,
                                char *err, size_t errlen) {
    if (g->k_cache[il]) {
        const uint64_t live = (uint64_t)rows * qwen35_payload_kv_half();
        int rc = qwen35_payload_span(dir, fp, g->k_cache[il], live, buf, left, err, errlen);
        return rc ? rc : qwen35_payload_span(dir, fp, g->v_cache[il], live, buf, left,
                                             err, errlen);
    }
    int rc = qwen35_payload_span(dir, fp, g->lin_state[il], g->lin_state_bytes, buf,
                                 left, err, errlen);
    return rc ? rc : qwen35_payload_span(dir, fp, g->lin_hist[il], g->lin_hist_bytes,
                                         buf, left, err, errlen);
}

static int qwen35_payload_walk(qwen35_payload_dir dir, FILE *fp,
                               ds4_qwen35_gpu_graph *g, uint32_t n, uint8_t *buf,
                               uint64_t *left, char *err, size_t errlen) {
    int rc = 0;
    for (uint32_t il = 0; !rc && il < DS4_N_LAYER; il++) {
        if (!g->k_cache[il] && !g->lin_state[il]) { continue; }
        rc = qwen35_payload_state(dir, fp, g, il, n, buf, left, err, errlen);
    }
    return rc;
}

/* The block's own prefix: its pending row, the held hidden row and its k/v. */
static int qwen35_payload_walk_block(qwen35_payload_dir dir, FILE *fp,
                                     ds4_qwen35_gpu_graph *g, uint32_t rows,
                                     uint8_t *buf, uint64_t *left,
                                     char *err, size_t errlen) {
    const uint64_t row_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    if (dir == QWEN35_PAYLOAD_SAVE) {
        const uint32_t pending = g->mtp_pending ? 1u : 0u;
        if (payload_write_u32(fp, pending, err, errlen) != 0) { return 1; }
        if (payload_write_tensor_span(fp, g->mtp_hidden, 0, row_bytes, buf,
                                      DS4_SESSION_IO_CHUNK, err, errlen) != 0) {
            return 1;
        }
    } else {
        uint32_t pending = 0;
        if (payload_read_u32(fp, &pending, left, err, errlen) != 0 ||
            pending > 1u) {
            payload_set_err(err, errlen, "invalid Bonsai snapshot block state");
            return 1;
        }
        g->mtp_pending = pending != 0u;
        if (payload_read_tensor_span(fp, g->mtp_hidden, 0, row_bytes, buf,
                                     DS4_SESSION_IO_CHUNK, left, err, errlen) != 0) {
            return 1;
        }
    }
    const uint64_t live = (uint64_t)rows * qwen35_payload_kv_half();
    int rc = qwen35_payload_span(dir, fp, g->k_cache[DS4_QWEN35_MTP_LAYER], live, buf,
                                 left, err, errlen);
    return rc ? rc : qwen35_payload_span(dir, fp, g->v_cache[DS4_QWEN35_MTP_LAYER], live,
                                         buf, left, err, errlen);
}

static int qwen35_payload_save(ds4_qwen35_gpu_graph *g, const int *tokens, uint32_t n,
                               const float *logits, FILE *fp, char *err, size_t errlen) {
    if (!fp || !tokens || !logits || !qwen35_payload_bytes(g, n)) {
        payload_set_err(err, errlen, "invalid Bonsai snapshot state");
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "invalid Bonsai snapshot token");
            return 1;
        }
    }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) {
            payload_set_err(err, errlen, "invalid Bonsai snapshot logits");
            return 1;
        }
    }
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!buf || !ds4_gpu_synchronize()) {
        free(buf);
        payload_set_err(err, errlen, "Bonsai snapshot allocation or synchronization failed");
        return 1;
    }
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, g->ctx_cap,
        qwen35_payload_count(g, false), DS4_N_LAYER, QWEN35_PAYLOAD_LAYOUT,
        qwen35_payload_block_rows(g), n, (uint32_t)qwen35_payload_kv_row(),
        (uint32_t)g->lin_state_bytes, (uint32_t)g->lin_hist_bytes, DS4_N_VOCAB, n,
    };
    int rc = 0;
    for (uint32_t i = 0; !rc && i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        rc = payload_write_u32(fp, h[i], err, errlen);
    }
    for (uint32_t i = 0; !rc && i < n; i++) {
        rc = payload_write_u32(fp, (uint32_t)tokens[i], err, errlen);
    }
    if (!rc) { rc = payload_write_logits(fp, logits, buf, err, errlen); }
    if (!rc) { rc = qwen35_payload_walk(QWEN35_PAYLOAD_SAVE, fp, g, n, buf, NULL, err, errlen); }
    if (!rc && h[6]) {
        rc = qwen35_payload_walk_block(QWEN35_PAYLOAD_SAVE, fp, g, h[6], buf, NULL, err, errlen);
    }
    free(buf);
    return rc;
}

/* A failed restore leaves the graph's state unusable: the caller resets it. */
static int qwen35_payload_restore(ds4_qwen35_gpu_graph *g, FILE *fp, uint64_t *left,
                                  const uint32_t *h, int **tokens_out, float *logits,
                                  char *err, size_t errlen) {
    if (!g || !fp || !left || !h || !tokens_out || !logits) { return 1; }
    *tokens_out = NULL;
    const uint32_t n = h[7];
    const uint32_t rows = h[6];
    g->pos = 0;
    g->mtp_pos = 0;
    g->mtp_pending = false;
    g->trial_n = 0;
    if (h[0] != DS4_SESSION_PAYLOAD_MAGIC || h[1] != DS4_SESSION_PAYLOAD_VERSION ||
        h[5] != QWEN35_PAYLOAD_LAYOUT || h[3] != qwen35_payload_count(g, false) ||
        h[4] != DS4_N_LAYER || h[8] != qwen35_payload_kv_row() ||
        h[9] != g->lin_state_bytes || h[10] != g->lin_hist_bytes ||
        h[11] != DS4_N_VOCAB || h[12] != n || !n || n > h[2] || h[2] > g->ctx_cap ||
        (rows && (!qwen35_payload_block(g) || rows + 1u != n))) {
        payload_set_err(err, errlen,
                        "incompatible Bonsai snapshot header or length");
        return 1;
    }
    if (*left != qwen35_payload_body(g, n, rows)) {
        payload_set_err(err, errlen, "Bonsai snapshot byte count does not match its header");
        return 1;
    }
    int *tokens = xmalloc((size_t)n * sizeof(*tokens));
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!tokens || !buf || !ds4_gpu_synchronize()) {
        free(tokens);
        free(buf);
        payload_set_err(err, errlen, "Bonsai restore allocation or synchronization failed");
        return 1;
    }
    int rc = 0;
    for (uint32_t i = 0; !rc && i < n; i++) {
        uint32_t token = 0;
        rc = payload_read_u32(fp, &token, left, err, errlen);
        if (!rc && token >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "invalid Bonsai snapshot token");
            rc = 1;
        }
        tokens[i] = (int)token;
    }
    if (!rc) {
        rc = payload_read_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                                left, err, errlen);
    }
    for (uint32_t i = 0; !rc && i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) {
            payload_set_err(err, errlen, "invalid Bonsai snapshot logits");
            rc = 1;
        }
    }
    if (!rc) { rc = qwen35_payload_walk(QWEN35_PAYLOAD_LOAD, fp, g, n, buf, left, err, errlen); }
    g->mtp_pending = false;
    if (!rc && rows) {
        rc = qwen35_payload_walk_block(QWEN35_PAYLOAD_LOAD, fp, g, rows, buf, left, err, errlen);
    }
    g->mtp_pos = rows;
    free(buf);
    if (rc) {
        free(tokens);
        return 1;
    }
    if (*left || !ds4_gpu_tensor_write(g->logits, 0, logits,
                                       (uint64_t)DS4_N_VOCAB * sizeof(float)) ||
        !ds4_gpu_synchronize()) {
        payload_set_err(err, errlen, "Bonsai restore finalization failed");
        free(tokens);
        return 1;
    }
    g->pos = n;
    *tokens_out = tokens;
    return 0;
}
