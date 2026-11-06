/* Gemma-4 session payload: the token ids, the next-token logits, then every
 * layer's K and V cache whole.
 *
 * The KV is two layouts, so the header carries both row counts and the
 * restore refuses a file whose geometry differs: the 25 sliding layers keep a
 * ring of G4_WINDOW + the widest block one forward writes rows, and the 5
 * full layers keep ctx rows. The
 * slot of a row is position % ring, which is what the caches already hold, so
 * the restore copies the rings verbatim and only sets the position.
 *
 * Header (13 u32, the DS4 session contract; n sits at h[7], the slot the
 * shared session loader reads): magic, version, ctx, cap, layers, tag,
 * full-layer format (0 bf16, 1 FP8), n, window, sliding rows, full rows,
 * vocabulary, reserved.
 * A failed or partial restore marks the graph failed, never a usable prefix.
 * The first slice saves each cache whole; a later slice can narrow the ring
 * to the live window if the snapshot file size matters. */

enum { G4_PAYLOAD_TAG = 0x344b5647 }; /* "GVK4": BF16 k/v, ring + full layouts */

static uint32_t gemma4_payload_rows(const ds4_gemma4_graph *g, unsigned il) {
    return ds4_gemma4_kv_rows(il, g->ctx, g->cap);
}

/* One span of a layer: K or V for bf16, the whole K|V|scales row for FP8. */
static uint64_t gemma4_payload_span_bytes(const ds4_gemma4_graph *g, unsigned il) {
    const uint64_t rows = gemma4_payload_rows(g, il);
    if (g->full_fp8 && gemma4_is_full(il)) {
        return rows * ds4_gemma4_full_kv_row();
    }
    return rows * ds4_gemma4_kv_row(il);
}

static uint64_t gemma4_payload_body(const ds4_gemma4_graph *g, uint32_t n) {
    uint64_t body = (uint64_t)n * sizeof(uint32_t) +
        (uint64_t)DS4_N_VOCAB * sizeof(float);
    for (unsigned il = 0; il < G4_LAYERS; il++) {
        const uint32_t spans = (g->full_fp8 && gemma4_is_full(il)) ? 1u : 2u;
        body += spans * gemma4_payload_span_bytes(g, il);
    }
    return body;
}

static uint64_t gemma4_payload_bytes(const ds4_gemma4_graph *g, uint32_t n) {
    if (!g || !g->ready || g->failed || !n || n > g->ctx || g->position != n) {
        return 0;
    }
    return (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) +
        gemma4_payload_body(g, n);
}

/* One cache per span: k then v, layer by layer, each whole. */
static int gemma4_payload_caches(int save, FILE *fp, ds4_gemma4_graph *g,
        uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    int rc = 0;
    for (unsigned il = 0; !rc && il < G4_LAYERS; il++) {
        const uint64_t bytes = gemma4_payload_span_bytes(g, il);
        if (!bytes) { continue; }
        const bool fp8 = g->full_fp8 && gemma4_is_full(il);
        ds4_gpu_tensor *const spans[2] = {fp8 ? g->fkcache[il] : g->kcache[il],
            fp8 ? NULL : g->vcache[il]};
        const unsigned count = fp8 ? 1u : 2u;
        for (unsigned i = 0; !rc && i < count; i++) {
            rc = save
                ? payload_write_tensor_span(fp, spans[i], 0, bytes, buf,
                      DS4_SESSION_IO_CHUNK, err, errlen)
                : payload_read_tensor_span(fp, spans[i], 0, bytes, buf,
                      DS4_SESSION_IO_CHUNK, left, err, errlen);
        }
    }
    return rc;
}

static int gemma4_payload_save(ds4_gemma4_graph *g, const int *tokens,
        uint32_t n, const float *logits, FILE *fp, char *err, size_t errlen) {
    if (!g || !fp || !tokens || !logits || !gemma4_payload_bytes(g, n)) {
        payload_set_err(err, errlen, "invalid Gemma-4 snapshot state");
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "invalid Gemma-4 snapshot token");
            return 1;
        }
    }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) {
            payload_set_err(err, errlen, "invalid Gemma-4 snapshot logits");
            return 1;
        }
    }
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!buf || !ds4_gpu_synchronize()) {
        free(buf);
        payload_set_err(err, errlen,
                        "Gemma-4 snapshot allocation or synchronization failed");
        return 1;
    }
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, g->ctx, g->cap,
        G4_LAYERS, G4_PAYLOAD_TAG, g->full_fp8 ? 1u : 0u, n, (uint32_t)G4_WINDOW,
        gemma4_payload_rows(g, 0u), gemma4_payload_rows(g, 5u), DS4_N_VOCAB, 0u,
    };
    int rc = 0;
    for (uint32_t i = 0; !rc && i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        rc = payload_write_u32(fp, h[i], err, errlen);
    }
    for (uint32_t i = 0; !rc && i < n; i++) {
        rc = payload_write_u32(fp, (uint32_t)tokens[i], err, errlen);
    }
    if (!rc) { rc = payload_write_logits(fp, logits, buf, err, errlen); }
    if (!rc) { rc = gemma4_payload_caches(1, fp, g, buf, NULL, err, errlen); }
    free(buf);
    return rc;
}

static int gemma4_payload_restore(ds4_gemma4_graph *g, FILE *fp,
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
        h[3] != g->cap || h[4] != G4_LAYERS || h[5] != G4_PAYLOAD_TAG ||
        h[6] != (g->full_fp8 ? 1u : 0u) || h[8] != (uint32_t)G4_WINDOW ||
        h[9] != gemma4_payload_rows(g, 0u) ||
        h[10] != gemma4_payload_rows(g, 5u) ||
        h[11] != DS4_N_VOCAB || h[12] != 0u ||
        *left != gemma4_payload_body(g, n)) {
        payload_set_err(err, errlen,
                        "incompatible Gemma-4 snapshot header or length");
        return 1;
    }
    int *tokens = malloc((size_t)n * sizeof(*tokens));
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!tokens || !buf || !ds4_gpu_synchronize()) {
        free(tokens);
        free(buf);
        payload_set_err(err, errlen,
                        "Gemma-4 restore allocation or synchronization failed");
        return 1;
    }
    int rc = 0;
    for (uint32_t i = 0; !rc && i < n; i++) {
        uint32_t token = 0;
        rc = payload_read_u32(fp, &token, left, err, errlen);
        if (!rc && token >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "invalid Gemma-4 snapshot token");
            rc = 1;
        }
        tokens[i] = (int)token;
    }
    if (!rc) {
        rc = payload_read_bytes(fp, logits,
                                (uint64_t)DS4_N_VOCAB * sizeof(float),
                                left, err, errlen);
    }
    for (uint32_t i = 0; !rc && i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) {
            payload_set_err(err, errlen, "invalid Gemma-4 snapshot logits");
            rc = 1;
        }
    }
    if (!rc) { rc = gemma4_payload_caches(0, fp, g, buf, left, err, errlen); }
    if (!rc && (*left ||
                !ds4_gpu_tensor_write(g->logits, 0, logits,
                                      (uint64_t)DS4_N_VOCAB * sizeof(float)) ||
                !ds4_gpu_synchronize())) {
        payload_set_err(err, errlen, "Gemma-4 restore finalization failed");
        rc = 1;
    }
    free(buf);
    if (rc) { free(tokens); return 1; }
    /* The rings come back whole, but rows past an abandoned branch may sit
     * in them: no resume below the restored frontier. */
    g->position = g->high = g->restored = n;
    g->failed = false;
    /* The KV rows and the logits are restored; the drafter's hidden row is
     * not in the payload, so it is republished by the next forward and this
     * cycle decodes one token instead of drafting (ds4_gemma4_mtp.inc). */
    gemma4_mtp_drop_hidden(g);
    *tokens_out = tokens;
    return 0;
}
