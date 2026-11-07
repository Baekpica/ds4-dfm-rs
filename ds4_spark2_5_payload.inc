/* Spark-X2.6 session payload: the token ids, the next-token logits, then
 * every layer's KV cache whole.
 *
 * The KV is two layouts, so the header carries both row counts and the
 * restore refuses a file whose geometry differs: the 27 sliding layers keep a
 * ring of SPARK_WINDOW + the prefill chunk rows, and the 9 full layers keep
 * ctx rows. The slot of a row is position % ring, which is what the caches
 * already hold, so the restore copies the rings verbatim and only sets the
 * position (ds4_gemma4_payload.inc:1-17, same contract).
 *
 * Header (13 u32, the DS4 session contract; n sits at h[7], the slot the
 * shared session loader reads): magic, version, ctx, cap, layers, tag,
 * full-layer format (0 bf16, 1 FP8), n, window, sliding rows, full rows,
 * vocabulary, reserved.
 * A failed or partial restore marks the graph failed, never a usable prefix. */

enum { SPARK25_PAYLOAD_TAG = 0x324b5653 }; /* "SVK2": BF16/FP8, ring + full */

static uint32_t spark2_5_payload_rows(const ds4_spark2_5_graph *g,
                                      uint32_t il) {
    return spark2_5_kv_rows(il, g->ctx, g->cap);
}

/* One span of a layer: K or V for BF16, the whole K|V|scales row for FP8. */
static uint64_t spark2_5_payload_span(const ds4_spark2_5_graph *g,
                                      uint32_t il) {
    const uint64_t rows = spark2_5_payload_rows(g, il);
    if (g->fkcache[il] != NULL) {
        return rows * spark2_5_fp8_row_bytes();
    }
    return rows * SPARK_KV_DIM * sizeof(uint16_t);
}

static uint64_t spark2_5_payload_body(const ds4_spark2_5_graph *g, uint32_t n) {
    uint64_t body = (uint64_t)n * sizeof(uint32_t) +
        (uint64_t)SPARK_VOCAB * sizeof(float);
    for (uint32_t il = 0; il < SPARK_LAYERS; il++) {
        const uint64_t spans = (g->fkcache[il] != NULL) ? 1u : 2u;
        body += spans * spark2_5_payload_span(g, il);
    }
    return body;
}

static uint64_t spark2_5_payload_bytes(const ds4_spark2_5_graph *g,
                                       uint32_t n) {
    if (!g || !g->ready || g->failed || !n || n > g->ctx ||
        g->position != n) {
        return 0;
    }
    return (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) +
        spark2_5_payload_body(g, n);
}

/* One cache per span: k then v, layer by layer, each whole. */
static int spark2_5_payload_caches(int save, FILE *fp, ds4_spark2_5_graph *g,
        uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    int rc = 0;
    for (uint32_t il = 0; !rc && il < SPARK_LAYERS; il++) {
        const uint64_t bytes = spark2_5_payload_span(g, il);
        if (!bytes) { continue; }
        const bool fp8 = g->fkcache[il] != NULL;
        ds4_gpu_tensor *const spans[2] = {fp8 ? g->fkcache[il] : g->kcache[il],
            fp8 ? NULL : g->vcache[il]};
        const uint32_t count = fp8 ? 1u : 2u;
        for (uint32_t i = 0; !rc && i < count; i++) {
            rc = save
                ? payload_write_tensor_span(fp, spans[i], 0, bytes, buf,
                      DS4_SESSION_IO_CHUNK, err, errlen)
                : payload_read_tensor_span(fp, spans[i], 0, bytes, buf,
                      DS4_SESSION_IO_CHUNK, left, err, errlen);
        }
    }
    return rc;
}

static uint32_t spark2_5_payload_kv_flags(const ds4_spark2_5_graph *g) {
    return (g->full_fp8 ? 1u : 0u) | (g->sliding_fp8 ? 2u : 0u);
}

static int spark2_5_payload_save(ds4_spark2_5_graph *g, const int *tokens,
        uint32_t n, const float *logits, FILE *fp, char *err, size_t errlen) {
    if (!g || !fp || !tokens || !logits || !spark2_5_payload_bytes(g, n)) {
        payload_set_err(err, errlen, "invalid Spark-X2.6 snapshot state");
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= SPARK_VOCAB) {
            payload_set_err(err, errlen, "invalid Spark-X2.6 snapshot token");
            return 1;
        }
    }
    for (uint32_t i = 0; i < SPARK_VOCAB; i++) {
        if (!isfinite(logits[i])) {
            payload_set_err(err, errlen, "invalid Spark-X2.6 snapshot logits");
            return 1;
        }
    }
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!buf || !ds4_gpu_synchronize()) {
        free(buf);
        payload_set_err(err, errlen,
                        "Spark-X2.6 snapshot allocation or synchronization failed");
        return 1;
    }
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, g->ctx, g->cap,
        SPARK_LAYERS, SPARK25_PAYLOAD_TAG, spark2_5_payload_kv_flags(g), n,
        (uint32_t)SPARK_WINDOW, spark2_5_payload_rows(g, 0u),
        spark2_5_payload_rows(g, 3u), SPARK_VOCAB, 0u,
    };
    int rc = 0;
    for (uint32_t i = 0; !rc && i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        rc = payload_write_u32(fp, h[i], err, errlen);
    }
    for (uint32_t i = 0; !rc && i < n; i++) {
        rc = payload_write_u32(fp, (uint32_t)tokens[i], err, errlen);
    }
    if (!rc) { rc = payload_write_logits(fp, logits, buf, err, errlen); }
    if (!rc) { rc = spark2_5_payload_caches(1, fp, g, buf, NULL, err, errlen); }
    free(buf);
    return rc;
}

static int spark2_5_payload_restore(ds4_spark2_5_graph *g, FILE *fp,
        uint64_t *left, const uint32_t *h, int **tokens_out, float *logits,
        char *err, size_t errlen) {
    if (!g || !fp || !left || !h || !tokens_out || !logits) { return 1; }
    /* A failed or partial restore can never be reused as a committed prefix. */
    g->failed = true;
    g->position = g->high = g->restored = 0;
    *tokens_out = NULL;
    const uint32_t n = h[7];
    if (!g->ready || !n || n > g->ctx || h[0] != DS4_SESSION_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_PAYLOAD_VERSION || h[2] != g->ctx ||
        h[3] != g->cap || h[4] != SPARK_LAYERS || h[5] != SPARK25_PAYLOAD_TAG ||
        h[6] != spark2_5_payload_kv_flags(g) || h[8] != (uint32_t)SPARK_WINDOW ||
        h[9] != spark2_5_payload_rows(g, 0u) ||
        h[10] != spark2_5_payload_rows(g, 3u) ||
        h[11] != SPARK_VOCAB || h[12] != 0u ||
        *left != spark2_5_payload_body(g, n)) {
        payload_set_err(err, errlen,
                        "incompatible Spark-X2.6 snapshot header or length");
        return 1;
    }
    int *tokens = malloc((size_t)n * sizeof(*tokens));
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!tokens || !buf || !ds4_gpu_synchronize()) {
        free(tokens);
        free(buf);
        payload_set_err(err, errlen,
                        "Spark-X2.6 restore allocation or synchronization failed");
        return 1;
    }
    int rc = 0;
    for (uint32_t i = 0; !rc && i < n; i++) {
        uint32_t token = 0;
        rc = payload_read_u32(fp, &token, left, err, errlen);
        if (!rc && token >= SPARK_VOCAB) {
            payload_set_err(err, errlen, "invalid Spark-X2.6 snapshot token");
            rc = 1;
        }
        tokens[i] = (int)token;
    }
    if (!rc) {
        rc = payload_read_bytes(fp, logits,
                                (uint64_t)SPARK_VOCAB * sizeof(float),
                                left, err, errlen);
    }
    for (uint32_t i = 0; !rc && i < SPARK_VOCAB; i++) {
        if (!isfinite(logits[i])) {
            payload_set_err(err, errlen, "invalid Spark-X2.6 snapshot logits");
            rc = 1;
        }
    }
    if (!rc) { rc = spark2_5_payload_caches(0, fp, g, buf, left, err, errlen); }
    if (!rc && (*left ||
                !ds4_gpu_tensor_write(g->logits, 0, logits,
                                      (uint64_t)SPARK_VOCAB * sizeof(float)) ||
                !ds4_gpu_synchronize())) {
        payload_set_err(err, errlen, "Spark-X2.6 restore finalization failed");
        rc = 1;
    }
    free(buf);
    if (rc) { free(tokens); return 1; }
    /* The rings come back whole, but rows past an abandoned branch may sit
     * in them: no resume below the restored frontier. */
    g->position = g->high = g->restored = n;
    g->failed = false;
    *tokens_out = tokens;
    return 0;
}
