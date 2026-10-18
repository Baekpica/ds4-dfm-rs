/* Ling-3.1 MTP (layer 72), HF modeling_bailing_moe_v3.py BailingMoeV3MTPLayer
 * and its training forward: row p pairs the trunk's final-norm hidden p with
 * token p+1 at position p. eh_proj([enorm(embed) | hnorm(hidden)]) feeds an
 * MLA + MoE decoder layer, then final_layernorm and the shared head; the
 * normed output is the next recursive draft's hidden. Prefill, decode and
 * verify fill the draft layer's latent rows from true hidden rows, so draft
 * attention sees the whole prompt. */

/* Draft-layer latent rows pos..pos+n-1 from final-norm `hidden` rows and their
 * next tokens: only the projections that feed the cache. Scratch: token,
 * positions, cur, attn_out, ffn_out, k (concat), heads, kv_raw, kv_norm. */
static bool l31_mtp_kv(ds4_ling31_graph *g, const ds4_model *m, const ds4_weights *w,
        const ds4_gpu_tensor *hidden, const int *tokens, uint32_t n, uint32_t pos) {
    const ds4_layer_weights *l = &w->layer[LING31_MTP];
    return l31_embed(g, m, w, tokens, n, pos) &&
        l31_norm(g, m, g->attn_out, g->cur, l->nextn_enorm, n, L31_EMBD) &&
        l31_norm(g, m, g->ffn_out, hidden, l->nextn_hnorm, n, L31_EMBD) &&
        ds4_gpu_gvq_concat(g->k, g->attn_out, g->ffn_out, n, L31_EMBD) &&
        l31_mm(g, m, g->cur, l->nextn_eh_proj, g->k, n, DS4_GVQ_BF16) &&
        l31_norm(g, m, g->heads, g->cur, l->attn_norm, n, L31_EMBD) &&
        l31_mm(g, m, g->kv_raw, l->attn_kv_a_mqa, g->heads, n, DS4_GVQ_BF16) &&
        ds4_gpu_ling3vl_mrope(g->kv_raw, g->positions, g->inv_freq, n, 1u, L31_KV_ROW,
            L31_KV_LORA, L31_ROPE, L31_ROPE / 2u, 0u, 1.0f) &&
        ds4_gpu_l31_latent_norm(g->kv_norm, g->kv_raw, g->kv_norm_w[LING31_MTP], n, DS4_RMS_EPS) &&
        ds4_gpu_ling3vl_store_latent(g->latent[LING31_MTP], g->k_pe[LING31_MTP], g->kv_norm,
            g->kv_raw, n, pos, g->ctx_cap, L31_KV_ROW, L31_KV_LORA, L31_ROPE);
}

/* Trunk head of an MTP graph: every row's final norm feeds the draft layer.
 * Row pos-1 waited for this forward's first token; this forward's last row
 * waits for the next one in last_hidden. */
static bool l31_mtp_head(ds4_ling31_graph *g, const ds4_model *m, const ds4_weights *w,
        const int *tokens, uint32_t rows, uint32_t pos, bool logits) {
    const uint64_t bytes = (uint64_t)L31_EMBD * sizeof(float);
    if (!l31_norm(g, m, g->norm, g->cur, w->output_norm, rows, L31_EMBD)) { return false; }
    const bool filled = pos ? g->mtp_len == pos - 1u : true;
    if (filled && pos && !l31_mtp_kv(g, m, w, g->last_hidden, tokens, 1u, pos - 1u)) { return false; }
    ds4_gpu_tensor *rows_in = rows > 1u ? ds4_gpu_tensor_view(g->norm, 0u, (rows - 1u) * bytes) : NULL;
    ds4_gpu_tensor *last = ds4_gpu_tensor_view(g->norm, (uint64_t)(rows - 1u) * bytes, bytes);
    const bool ok = last && (!filled || rows < 2u ||
            (rows_in && l31_mtp_kv(g, m, w, rows_in, tokens + 1, rows - 1u, pos))) &&
        ds4_gpu_gvq_copy(g->last_hidden, 0u, last, 0u, bytes, 1u) &&
        (!logits || l31_logits(g, m, w, g->logits, last, 1u));
    ds4_gpu_tensor_free(last);
    ds4_gpu_tensor_free(rows_in);
    if (ok && filled) { g->mtp_len = pos + rows - 1u; }
    return ok;
}

// One recursive draft at `pos`: the full draft layer, then its greedy token.
static bool l31_mtp_step(ds4_ling31_graph *g, const ds4_model *m, const ds4_weights *w,
        const ds4_gpu_tensor *hidden, int token, uint32_t pos, int *next) {
    if (g->mtp_len != pos || pos >= g->ctx_cap) { return false; }
    const ds4_layer_weights *l = &w->layer[LING31_MTP];
    const uint64_t bytes = (uint64_t)L31_EMBD * sizeof(float);
    ds4_gpu_tensor *left = ds4_gpu_tensor_view(g->k, 0u, bytes);
    ds4_gpu_tensor *right = ds4_gpu_tensor_view(g->k, bytes, bytes);
    const bool ok = left && right && l31_embed(g, m, w, &token, 1u, pos) &&
        l31_norm(g, m, left, g->cur, l->nextn_enorm, 1u, L31_EMBD) &&
        l31_norm(g, m, right, hidden, l->nextn_hnorm, 1u, L31_EMBD) &&
        l31_mm(g, m, g->cur, l->nextn_eh_proj, g->k, 1u, DS4_GVQ_BF16) &&
        l31_layer(g, m, w, LING31_MTP, 1u, pos) &&
        l31_norm(g, m, g->norm, g->cur, l->nextn_shared_head_norm, 1u, L31_EMBD) &&
        ds4_gpu_gvq_copy(g->mtp_hidden, 0u, g->norm, 0u, bytes, 1u) &&
        l31_logits(g, m, w, g->logits, g->norm, 1u) &&
        ds4_gpu_argmax_tensor(g->token, g->logits, DS4_N_VOCAB) &&
        ds4_gpu_tensor_read(g->token, 0u, next, sizeof(*next));
    ds4_gpu_tensor_free(right);
    ds4_gpu_tensor_free(left);
    if (ok) { g->mtp_len = pos + 1u; }
    return ok && *next >= 0 && (uint32_t)*next < DS4_N_VOCAB;
}

/* Batched verify of mtp_trial[0..n) at pos: one n-row trunk forward reads
 * every weight once. KDA journals each row's state; every row's hidden and
 * logits are kept, and the draft-layer rows the recursive drafts wrote are
 * rebuilt from the true hidden rows. */
static bool l31_mtp_verify(ds4_ling31_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t n, uint32_t pos, int *target) {
    const uint64_t bytes = (uint64_t)L31_EMBD * sizeof(float);
    if (!ds4_gpu_gvq_copy(g->mtp_rows, 0u, g->last_hidden, 0u, bytes, 1u) ||
        !l31_embed(g, m, w, g->mtp_trial, n, pos)) { return false; }
    g->mtp_capture = true;
    bool ok = l31_trunk(g, m, w, n, pos);
    g->mtp_capture = false;
    ds4_gpu_tensor *rows = ok ? ds4_gpu_tensor_view(g->mtp_rows, bytes, n * bytes) : NULL;
    ds4_gpu_tensor *all = ok ? ds4_gpu_tensor_view(g->mtp_rows, 0u, n * bytes) : NULL;
    ok = ok && rows && all &&
        l31_norm(g, m, rows, g->cur, w->output_norm, n, L31_EMBD) &&
        l31_logits(g, m, w, g->mtp_logits, rows, n) &&
        ds4_gpu_argmax_rows_tensor(g->token, g->mtp_logits, DS4_N_VOCAB, n) &&
        ds4_gpu_tensor_read(g->token, 0u, target, (uint64_t)n * sizeof(*target)) &&
        l31_mtp_kv(g, m, w, all, g->mtp_trial, n, pos - 1u);
    ds4_gpu_tensor_free(all);
    ds4_gpu_tensor_free(rows);
    for (uint32_t i = 0; ok && i < n; i++) { ok = target[i] >= 0 && (uint32_t)target[i] < DS4_N_VOCAB; }
    return ok;
}

/* Frontier after `keep` of the verified rows: KDA state from the journal,
 * then the last kept row's hidden and logits. Latent rows past the frontier
 * are rewritten before any read. */
static bool l31_mtp_land(ds4_ling31_graph *g, uint32_t keep, float *logits) {
    const uint64_t vocab = (uint64_t)DS4_N_VOCAB * sizeof(float), bytes = (uint64_t)L31_EMBD * sizeof(float);
    if (!keep || keep > g->mtp_n) { return false; }
    const uint8_t *pool = (const uint8_t *)ds4_gpu_tensor_ptr(g->state_pool);
    for (uint32_t il = 0; keep < g->mtp_n && il < LING31_LAYERS; il++) {
        if (ds4_ling31_layer_is_mla(il)) { continue; }
        const uint8_t *at = (const uint8_t *)ds4_gpu_tensor_ptr(g->recurrent[il]);
        const uint8_t *end = (const uint8_t *)ds4_gpu_tensor_ptr(g->v_tail[il]) + ds4_gpu_tensor_bytes(g->v_tail[il]);
        const uint64_t offset = (uint64_t)(at - pool);
        if (!ds4_gpu_gvq_copy(g->state_pool, offset, g->mtp_journal,
                (uint64_t)(keep - 1u) * g->state_bytes + offset, (uint64_t)(end - at), 1u)) { return false; }
    }
    if (!ds4_gpu_gvq_copy(g->last_hidden, 0u, g->mtp_rows, keep * bytes, bytes, 1u) ||
        !ds4_gpu_gvq_copy(g->logits, 0u, g->mtp_logits, (keep - 1u) * vocab, vocab, 1u) ||
        !ds4_gpu_tensor_read(g->mtp_logits, (keep - 1u) * vocab, logits, vocab)) { return false; }
    g->position = g->mtp_start + keep;
    g->mtp_len = g->position - 1u;
    g->mtp_n = 0u;
    return true;
}
