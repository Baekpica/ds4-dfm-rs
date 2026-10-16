/* SGLang c15edf314d20073630e672eb477cbcc3c58fac57:
 * deepseek_nextn.py:195-245, used by glm5_next_nextn.py. Both target and
 * recursive draft inputs are post-final-norm hidden states. Draft attention
 * currently uses the explicit GLM lazy suffix/full-causal cache policy. */
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
