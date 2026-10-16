/* SGLang c15edf314d20073630e672eb477cbcc3c58fac57:
 * deepseek_nextn.py:195-245, used by glm5_next_nextn.py. Both target and
 * recursive draft inputs are post-final-norm hidden states. Draft attention
 * is full causal over the draft-layer KV, which prefill fills with the
 * prompt (the source layer selects keys with its own DSA indexer). */
static bool gvq_mtp_block(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_weights *w, const ds4_gpu_tensor *hidden, int token,
        uint32_t pos, uint32_t first) {
    const ds4_layer_weights *l = &w->layer[GLMVQ_TRUNK];
    const uint64_t bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t row = (uint64_t)DS4_N_KV_LORA * sizeof(uint16_t);
    ds4_gpu_tensor *left = ds4_gpu_tensor_view(g->mtp_concat, 0u, bytes);
    ds4_gpu_tensor *right = ds4_gpu_tensor_view(g->mtp_concat, bytes, bytes);
    ds4_gpu_tensor *cache = ds4_gpu_tensor_view(g->mtp_kv, (uint64_t)first * row,
        (uint64_t)(g->ctx_cap - first) * row);
    const bool ok = left && right && cache &&
        ds4_gpu_embed_token_quant_tensor(g->cur, m->map, m->size,
            w->token_embd->abs_offset, w->token_embd->type, DS4_N_VOCAB, (uint32_t)token, DS4_N_EMBD) &&
        gvq_norm(g, m, left, g->cur, l->nextn_enorm, 1u, DS4_N_EMBD) &&
        gvq_norm(g, m, right, hidden, l->nextn_hnorm, 1u, DS4_N_EMBD) &&
        gvq_mm(g, m, g->cur, l->nextn_eh_proj, g->mtp_concat, 1u, DS4_GVQ_BF16) &&
        gvq_norm(g, m, g->norm, g->cur, l->attn_norm, 1u, DS4_N_EMBD) &&
        gvq_mm(g, m, g->q_rank, l->attn_q_a, g->norm, 1u, DS4_GVQ_BF16) &&
        gvq_norm(g, m, g->q_rank_norm, g->q_rank, l->attn_q_a_norm, 1u, DS4_N_LORA_Q) &&
        gvq_mm(g, m, g->q, l->attn_q_b, g->q_rank_norm, 1u, DS4_GVQ_BF16) &&
        gvq_mm(g, m, g->kv_rank, l->attn_kv_a_mqa, g->norm, 1u, DS4_GVQ_BF16) &&
        gvq_norm(g, m, g->kv_rank_norm, g->kv_rank, l->attn_kv_a_norm, 1u, DS4_N_KV_LORA) &&
        ds4_gpu_gvq_store(g->mtp_kv, g->kv_rank_norm, 1u, pos, g->ctx_cap, DS4_N_KV_LORA) &&
        ds4_gpu_gvq_dsa(g->gvq, g->heads, g->q, cache, glmvq_index(m, l->attn_kv_b),
            g->gvq_latent, g->gvq_kv, g->gvq_pos, pos - first, 1u,
            DS4_N_HEAD, DS4_N_KEY_MLA, NULL, 0u) &&
        gvq_mm(g, m, g->attn_out, l->attn_output, g->heads, 1u, DS4_GVQ_BF16) &&
        ds4_gpu_add_tensor(g->mtp_hidden, g->cur, g->attn_out, DS4_N_EMBD) &&
        ds4_gpu_gvq_round(g->mtp_hidden, DS4_N_EMBD) &&
        gvq_norm(g, m, g->norm, g->mtp_hidden, l->ffn_norm, 1u, DS4_N_EMBD) &&
        gvq_ffn(g, m, w, GLMVQ_TRUNK, 1u) &&
        ds4_gpu_add_tensor(g->mtp_hidden, g->mtp_hidden, g->ffn_out, DS4_N_EMBD) &&
        ds4_gpu_gvq_round(g->mtp_hidden, DS4_N_EMBD) &&
        gvq_norm(g, m, g->norm, g->mtp_hidden, l->nextn_shared_head_norm, 1u, DS4_N_EMBD) &&
        ds4_gpu_tensor_copy(g->mtp_hidden, 0u, g->norm, 0u, bytes) &&
        gvq_mm(g, m, g->logits, w->output, g->norm, 1u, DS4_GVQ_BF16);
    ds4_gpu_tensor_free(cache);
    ds4_gpu_tensor_free(right);
    ds4_gpu_tensor_free(left);
    return ok;
}

/* Draft-layer KV rows pos..pos+n-1 from true final-norm hidden rows and
 * their next tokens: only the projections that feed the cache, as in
 * gvq_mtp_block. `a` is bound to at least n rows (its cur, norm, attn_out,
 * ffn_out and kv_rank buffers are scratch); `concat` holds n x 2 rows. */
static bool gvq_mtp_kv(ds4_glm53_gpu_graph *a, const ds4_model *m,
        const ds4_weights *w, const ds4_gpu_tensor *hidden, const int *tokens,
        uint32_t n, uint32_t pos, ds4_gpu_tensor *concat) {
    const ds4_layer_weights *l = &w->layer[GLMVQ_TRUNK];
    return ds4_gpu_tensor_write(a->token, 0u, tokens, (uint64_t)n * sizeof(int32_t)) &&
        ds4_gpu_embed_tokens_quant_tensor(a->cur, a->token, m->map, m->size,
            w->token_embd->abs_offset, w->token_embd->type, DS4_N_VOCAB, n, DS4_N_EMBD) &&
        gvq_norm(a, m, a->attn_out, hidden, l->nextn_hnorm, n, DS4_N_EMBD) &&
        gvq_norm(a, m, a->ffn_out, a->cur, l->nextn_enorm, n, DS4_N_EMBD) &&
        ds4_gpu_gvq_concat(concat, a->ffn_out, a->attn_out, n, DS4_N_EMBD) &&
        gvq_mm(a, m, a->cur, l->nextn_eh_proj, concat, n, DS4_GVQ_BF16) &&
        gvq_norm(a, m, a->norm, a->cur, l->attn_norm, n, DS4_N_EMBD) &&
        gvq_mm(a, m, a->kv_rank, l->attn_kv_a_mqa, a->norm, n, DS4_GVQ_BF16) &&
        gvq_norm(a, m, a->kv_rank_norm, a->kv_rank, l->attn_kv_a_norm, n, DS4_N_KV_LORA) &&
        ds4_gpu_gvq_store(a->mtp_kv, a->kv_rank_norm, n, pos, a->ctx_cap, DS4_N_KV_LORA);
}

/* Prefill fills the draft-layer KV with the prompt so the first drafts see
 * the whole context, not only decoded tokens. Row p pairs hidden p with
 * token p+1, so a chunk's last row waits for the next chunk (its hidden is
 * parked in mtp_rows[0]) or for the first trial's draft step. */
static bool gvq_mtp_prefill(ds4_glm53_gpu_graph *g, ds4_glm53_gpu_graph *a,
        const ds4_model *m, const ds4_weights *w, const int *tokens, uint32_t rows,
        uint32_t first, bool pending) {
    const uint64_t bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    if (pending) {
        ds4_gpu_tensor *hidden = ds4_gpu_tensor_view(g->mtp_rows, 0u, bytes);
        const bool ok = hidden && gvq_mtp_kv(a, m, w, hidden, tokens, 1u, first - 1u, g->mtp_concat);
        ds4_gpu_tensor_free(hidden);
        if (!ok) { return false; }
        g->mtp_len = first;
    }
    if (rows < 2 || (g->mtp_len && g->mtp_len != first)) { return true; }

    // Every row's final norm, as gvq_output computes it for the last row.
    ds4_gpu_tensor *concat = ds4_gpu_tensor_view(a->hc_flat, 0u, 2u * (rows - 1u) * bytes);
    const bool ok = concat &&
        ds4_gpu_hc_weighted_sum_tensor(a->cur, a->hc_cur, a->hc_mean, DS4_N_EMBD, DS4_N_HC) &&
        ds4_gpu_gvq_round(a->cur, (uint64_t)rows * DS4_N_EMBD) &&
        gvq_norm(a, m, a->norm, a->cur, w->output_norm, rows, DS4_N_EMBD) &&
        gvq_mtp_kv(a, m, w, a->norm, tokens + 1, rows - 1u, first, concat);
    ds4_gpu_tensor_free(concat);
    if (!ok) { return false; }
    if (!g->mtp_len) { g->mtp_min = first; }
    g->mtp_len = first + rows - 1u;
    return true;
}
