/* Mutable KDA state includes the partial indexer pool tails. Immutable
 * controls and SSD cache contents are rebuilt and never enter disk KV. */
#define GLM53_PAYLOAD_COMPACT UINT32_C(0x35434c47) /* "GLC5" */
#define GLM53_PAYLOAD_EXPANDED UINT32_C(0x35454c47) /* "GLE5" */
#define GLM53_PAYLOAD_MTP_COMPACT UINT32_C(0x354d4347) /* "GCM5" */
#define GLM53_PAYLOAD_MTP_EXPANDED UINT32_C(0x354d4547) /* "GEM5" */
typedef enum { GLM53_PAYLOAD_SAVE, GLM53_PAYLOAD_LOAD } glm53_payload_dir;

static uint64_t glm53_payload_row_bytes(const ds4_glm53_gpu_graph *g) {
    const uint64_t values = g->expanded_diag
        ? (uint64_t)DS4_N_HEAD * DS4_N_KEY_MLA * 2u : DS4_N_KV_LORA;
    return values * sizeof(uint16_t);
}

static uint64_t glm53_payload_base_bytes(const ds4_glm53_gpu_graph *g, uint32_t n) {
    if (!g || !g->ready || !n || n > g->ctx_cap ||
        !g->state_bytes || g->state_bytes > UINT32_MAX ||
        ds4_gpu_tensor_bytes(g->state_pool) != g->state_bytes) { return 0u; }
    uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    if (!payload_u64_add(&bytes, (uint64_t)n * sizeof(uint32_t)) ||
        !payload_u64_add(&bytes, (uint64_t)DS4_N_VOCAB * sizeof(float)) ||
        !payload_u64_add(&bytes, g->state_bytes)) { return 0u; }
    const uint64_t latent = (uint64_t)n * glm53_payload_row_bytes(g);
    const uint64_t pooled = (uint64_t)(n / DS4_GLM53_POOL_SIZE) *
        DS4_N_INDEXER_HEAD_DIM * sizeof(uint16_t);
    for (uint32_t il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        if (ds4_glm53_layer_is_kda(il)) { continue; }
        if (latent > ds4_gpu_tensor_bytes(g->layer_kv[il]) ||
            !payload_u64_add(&bytes, latent)) { return 0u; }
        if (!g->expanded_diag && (pooled > ds4_gpu_tensor_bytes(g->layer_pool[il]) ||
            !g->layer_pool[il] || !payload_u64_add(&bytes, pooled))) { return 0u; }
    }
    return bytes;
}

static uint64_t glm53_payload_bytes(const ds4_glm53_gpu_graph *g, uint32_t n) {
    uint64_t bytes = glm53_payload_base_bytes(g, n);
    if (!bytes || g->mtp_trial_n) { return 0u; }
    if (!g->mtp_ready) { return bytes; }
    if (g->mtp_min > g->mtp_len || g->mtp_len >= n ||
        (g->mtp_len == 0u && g->mtp_min != 0u)) { return 0u; }
    const uint64_t row = (uint64_t)DS4_N_KV_LORA * sizeof(uint16_t);
    const uint64_t end = (uint64_t)g->mtp_len * row;
    if (end > ds4_gpu_tensor_bytes(g->mtp_kv) ||
        !payload_u64_add(&bytes, 2u * sizeof(uint32_t)) ||
        !payload_u64_add(&bytes, (uint64_t)(g->mtp_len - g->mtp_min) * row)) { return 0u; }
    return bytes;
}

static int glm53_payload_span(FILE *fp, ds4_gpu_tensor *tensor, uint64_t offset, uint64_t bytes,
        uint8_t *buf, uint64_t *remaining, glm53_payload_dir dir,
        char *err, size_t errlen) {
    return dir == GLM53_PAYLOAD_LOAD
        ? payload_read_tensor_span(fp, tensor, offset, bytes, buf,
            DS4_SESSION_IO_CHUNK, remaining, err, errlen)
        : payload_write_tensor_span(fp, tensor, offset, bytes, buf,
            DS4_SESSION_IO_CHUNK, err, errlen);
}

static int glm53_payload_state(ds4_glm53_gpu_graph *g, FILE *fp, uint32_t n,
        uint64_t *remaining, glm53_payload_dir dir, char *err, size_t errlen) {
    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = glm53_payload_span(fp, g->state_pool, 0u, g->state_bytes, buf,
                                remaining, dir, err, errlen);
    const uint64_t latent = (uint64_t)n * glm53_payload_row_bytes(g);
    const uint64_t pooled = (uint64_t)(n / DS4_GLM53_POOL_SIZE) *
        DS4_N_INDEXER_HEAD_DIM * sizeof(uint16_t);
    for (uint32_t il = 0; !rc && il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        if (ds4_glm53_layer_is_kda(il)) { continue; }
        rc = glm53_payload_span(fp, g->layer_kv[il], 0u, latent, buf,
                                 remaining, dir, err, errlen);
        if (!rc && !g->expanded_diag) {
            rc = glm53_payload_span(fp, g->layer_pool[il], 0u, pooled, buf,
                                     remaining, dir, err, errlen);
        }
    }
    if (!rc && g->mtp_ready && g->mtp_len > g->mtp_min) {
        const uint64_t row = (uint64_t)DS4_N_KV_LORA * sizeof(uint16_t);
        rc = glm53_payload_span(fp, g->mtp_kv, (uint64_t)g->mtp_min * row,
            (uint64_t)(g->mtp_len - g->mtp_min) * row, buf, remaining, dir, err, errlen);
    }
    free(buf);
    return rc;
}

static uint32_t glm53_payload_tag(const ds4_glm53_gpu_graph *g) {
    if (g->mtp_ready) {
        return g->expanded_diag ? GLM53_PAYLOAD_MTP_EXPANDED : GLM53_PAYLOAD_MTP_COMPACT;
    }
    return g->expanded_diag ? GLM53_PAYLOAD_EXPANDED : GLM53_PAYLOAD_COMPACT;
}

static int glm53_payload_save(ds4_glm53_gpu_graph *g, const int *tokens,
        uint32_t n, const float *logits, FILE *fp, char *err, size_t errlen) {
    if (!fp || !tokens || !logits || !glm53_payload_bytes(g, n) || g->cache_len != n) {
        payload_set_err(err, errlen, "GLM-5.3 checkpoint has no complete frontier");
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "GLM-5.3 checkpoint has an invalid token");
            return 1;
        }
    }
    if (!ds4_gpu_synchronize()) {
        payload_set_err(err, errlen, "GLM-5.3 checkpoint synchronization failed");
        return 1;
    }
    uint32_t kda, dsa;
    glm53_graph_layers(&kda, &dsa);
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, g->ctx_cap,
        DS4_N_KV_LORA, dsa, glm53_payload_tag(g), (uint32_t)glm53_payload_row_bytes(g),
        n, DS4_N_LAYER - DS4_N_NEXTN_PREDICT, DS4_N_SSM_CONV,
        DS4_N_INDEXER_HEAD_DIM, DS4_N_VOCAB, (uint32_t)g->state_bytes
    };
    for (unsigned i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, h[i], err, errlen)) { return 1; }
    }
    for (uint32_t i = 0; i < n; i++) {
        if (payload_write_u32(fp, (uint32_t)tokens[i], err, errlen)) { return 1; }
    }
    if (payload_write_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float), err, errlen)) { return 1; }
    if (g->mtp_ready && (payload_write_u32(fp, g->mtp_len, err, errlen) ||
        payload_write_u32(fp, g->mtp_min, err, errlen))) { return 1; }
    return glm53_payload_state(g, fp, n, NULL, GLM53_PAYLOAD_SAVE, err, errlen);
}

static int glm53_payload_restore(ds4_glm53_gpu_graph *g, FILE *fp,
        uint64_t *remaining, const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS],
        int **tokens_out, float *logits, char *err, size_t errlen) {
    uint32_t kda, dsa;
    glm53_graph_layers(&kda, &dsa);
    if (!g || !g->ready || !fp || !remaining || !tokens_out || !logits ||
        h[0] != DS4_SESSION_PAYLOAD_MAGIC || h[1] != DS4_SESSION_PAYLOAD_VERSION ||
        !h[2] || h[2] > g->ctx_cap || h[3] != DS4_N_KV_LORA ||
        h[4] != dsa || h[5] != glm53_payload_tag(g) ||
        h[6] != glm53_payload_row_bytes(g) || !h[7] || h[7] > h[2] ||
        h[8] != DS4_N_LAYER - DS4_N_NEXTN_PREDICT || h[9] != DS4_N_SSM_CONV ||
        h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB || h[12] != g->state_bytes) {
        payload_set_err(err, errlen, "GLM-5.3 checkpoint layout does not match this graph");
        return 1;
    }
    const uint32_t n = h[7];
    const uint64_t total = glm53_payload_base_bytes(g, n);
    const uint64_t header_bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    const uint64_t body_bytes = *remaining;
    const uint64_t min_bytes = total - header_bytes + (g->mtp_ready ? 2u * sizeof(uint32_t) : 0u);
    if (!total || (g->mtp_ready ? body_bytes < min_bytes : body_bytes != min_bytes)) {
        payload_set_err(err, errlen, "GLM-5.3 checkpoint byte count does not match its header");
        return 1;
    }
    int *tokens = xmalloc((size_t)n * sizeof(*tokens));
    int rc = 0;
    for (uint32_t i = 0; !rc && i < n; i++) {
        uint32_t token = 0;
        rc = payload_read_u32(fp, &token, remaining, err, errlen);
        if (!rc && token >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "GLM-5.3 checkpoint has an invalid token");
            rc = 1;
        }
        tokens[i] = (int)token;
    }
    if (!rc) { rc = payload_read_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float), remaining, err, errlen); }
    uint32_t mtp_len = 0u, mtp_min = 0u;
    if (!rc && g->mtp_ready) {
        rc = payload_read_u32(fp, &mtp_len, remaining, err, errlen);
        if (!rc) { rc = payload_read_u32(fp, &mtp_min, remaining, err, errlen); }
        const uint64_t row = (uint64_t)DS4_N_KV_LORA * sizeof(uint16_t);
        if (!rc && (mtp_min > mtp_len || mtp_len >= n ||
            (mtp_len == 0u && mtp_min != 0u) ||
            (uint64_t)mtp_len * row > ds4_gpu_tensor_bytes(g->mtp_kv) ||
            body_bytes != min_bytes + (uint64_t)(mtp_len - mtp_min) * row)) {
            payload_set_err(err, errlen, "GLM-5.3 MTP checkpoint frontier is invalid");
            rc = 1;
        }
    }
    if (!rc && !glm53_graph_reset(g)) {
        payload_set_err(err, errlen, "GLM-5.3 state reset failed before restore");
        rc = 1;
    }
    if (!rc) { g->mtp_len = mtp_len; g->mtp_min = mtp_min; }
    if (!rc) { rc = glm53_payload_state(g, fp, n, remaining, GLM53_PAYLOAD_LOAD, err, errlen); }
    if (!rc && (!ds4_gpu_synchronize() || *remaining)) {
        payload_set_err(err, errlen, "GLM-5.3 checkpoint restore did not complete");
        rc = 1;
    }
    if (rc) {
        free(tokens);
        (void)glm53_graph_reset(g);
        return 1;
    }
    *tokens_out = tokens;
    g->cache_len = n;
    return 0;
}
