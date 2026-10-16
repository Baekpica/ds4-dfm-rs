/* Source BF16 graph, independent of MXQ arithmetic. Source citations refer
 * to the pinned Transformers 5.19.0 modeling_glm5_next.py in the handoff.
 * Sparse index pools keep the source BF16 boundaries. Captured execution
 * remains a separate state/position contract. */

static uint64_t gvq_scratch(uint32_t ctx, uint32_t rows) {
    uint64_t bytes = ds4_gpu_gvq_attn_bytes(rows, ctx, DS4_N_HEAD, DS4_N_KEY_MLA);
    // Only the diagnostic Q4 restore path materializes the full output head.
    // Fund it before graph allocation; direct dots need no expanded head.
    const char *direct = getenv("DS4_GLMVQ_Q4_DIRECT");
    const uint32_t head_type = direct && !strcmp(direct, "0") ? DS4_TENSOR_Q4_K : DS4_TENSOR_BF16;
    const uint64_t candidates[] = {
        ds4_gpu_gvq_index_bytes(rows, (ctx + DS4_GLM53_POOL_SIZE - 1u) / DS4_GLM53_POOL_SIZE),
        ds4_gpu_gvq_linear_bytes(DS4_N_FF_DENSE, DS4_N_EMBD, rows,
            DS4_TENSOR_GLM_FP8, DS4_GVQ_BF16),
        ds4_gpu_gvq_linear_bytes(DS4_N_EMBD, DS4_N_HEAD * DS4_N_KEY_MLA, rows,
            DS4_TENSOR_GLM_FP8, DS4_GVQ_BF16),
        ds4_gpu_gvq_linear_bytes(2u * DS4_N_HEAD * DS4_N_KEY_MLA,
            DS4_N_KV_LORA, ctx, DS4_TENSOR_GLM_FP8, DS4_GVQ_BF16),
        ds4_gpu_gvq_linear_bytes(DS4_N_EXPERT, DS4_N_EMBD, rows,
            DS4_TENSOR_BF16, DS4_GVQ_F32),
        ds4_gpu_gvq_linear_bytes(DS4_N_VOCAB, DS4_N_EMBD, 1u,
            head_type, DS4_GVQ_BF16)
    };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        if (candidates[i] > bytes) { bytes = candidates[i]; }
    }
    return bytes;
}

static uint64_t gvq_extra_bytes(uint32_t ctx, uint32_t rows) {
    return gvq_scratch(ctx, rows) + ds4_gpu_gvq_workspace(rows) +
        (uint64_t)ctx * (DS4_N_KV_LORA + 2u * DS4_N_HEAD * DS4_N_KEY_MLA) * sizeof(float) +
        (uint64_t)rows * sizeof(uint32_t);
}

static bool gvq_graph_init(ds4_glm53_gpu_graph *g, const ds4_weights *w) {
    if (g->ctx_cap > DS4_GVQ_CTX_MAX || !w->glmvq_gpu) { return false; }
    g->gvq = ds4_gpu_gvq_create(w->glmvq_gpu, gvq_scratch(g->ctx_cap, g->row_cap), g->row_cap);
    g->gvq_latent = ds4_gpu_tensor_alloc((uint64_t)g->ctx_cap * DS4_N_KV_LORA * sizeof(float));
    g->gvq_kv = ds4_gpu_tensor_alloc((uint64_t)g->ctx_cap * 2u * DS4_N_HEAD * DS4_N_KEY_MLA * sizeof(float));
    g->gvq_pos = ds4_gpu_tensor_alloc((uint64_t)g->row_cap * sizeof(uint32_t));
    g->workspace_bytes += gvq_extra_bytes(g->ctx_cap, g->row_cap);
    return g->gvq && g->gvq_latent && g->gvq_kv && g->gvq_pos;
}

static void gvq_graph_free(ds4_glm53_gpu_graph *g) {
    ds4_gpu_gvq_free(g->gvq);
    ds4_gpu_tensor_free(g->gvq_pos);
    ds4_gpu_tensor_free(g->gvq_kv);
    ds4_gpu_tensor_free(g->gvq_latent);
}

static bool gvq_control(ds4_glm53_gpu_graph *g, const ds4_model *m,
        ds4_gpu_tensor *out, const ds4_tensor *weight, uint64_t bytes) {
    return ds4_gpu_gvq_control(g->gvq, out, glmvq_index(m, weight), bytes / sizeof(float));
}

static bool gvq_mm(ds4_glm53_gpu_graph *g, const ds4_model *m,
        ds4_gpu_tensor *out, const ds4_tensor *w, const ds4_gpu_tensor *x,
        uint32_t rows, ds4_gvq_math math) {
    const bool ok = ds4_gpu_gvq_linear(g->gvq, out, glmvq_index(m, w), x, rows, math);
    if (!ok && w) {
        fprintf(stderr, "ds4: GLM VQ linear %.*s failed (rows=%u)\n", (int)w->name.len, w->name.ptr, rows);
    }
    return ok;
}

static bool gvq_norm(ds4_glm53_gpu_graph *g, const ds4_model *m,
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_tensor *w,
        uint32_t rows, uint32_t width) {
    return ds4_gpu_gvq_rms(g->gvq, out, x, NULL, glmvq_index(m, w),
        rows, width, DS4_RMS_EPS, DS4_GVQ_RMS);
}

static bool gvq_hc(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_tensor *fn, const ds4_tensor *scale, const ds4_tensor *base,
        const ds4_tensor *norm, const ds4_gpu_tensor *residual, uint32_t rows) {
    // Source 286-329: mHC control arithmetic is FP32; collapse is BF16.
    return ds4_gpu_gvq_rms(g->gvq, g->hc_flat, residual, NULL, DS4_GVQ_NO_LINK,
            rows, DS4_N_HC * DS4_N_EMBD, DS4_RMS_EPS, DS4_GVQ_PLAIN_RMS) &&
        gvq_mm(g, m, g->hc_mix, fn, g->hc_flat, rows, DS4_GVQ_F32) &&
        ds4_gpu_gvq_split(g->gvq, g->cur, g->hc_split, g->hc_mix, residual,
            glmvq_index(m, scale), glmvq_index(m, base), rows, DS4_N_EMBD,
            DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS) &&
        gvq_norm(g, m, g->norm, g->cur, norm, rows, DS4_N_EMBD);
}

static bool gvq_kda(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_layer_weights *l, uint32_t il, uint32_t rows) {
    if (!gvq_mm(g, m, g->q, l->kda_q, g->norm, rows, DS4_GVQ_BF16) ||
        !gvq_mm(g, m, g->k, l->kda_k, g->norm, rows, DS4_GVQ_BF16) ||
        !gvq_mm(g, m, g->v, l->kda_v, g->norm, rows, DS4_GVQ_BF16) ||
        !gvq_mm(g, m, g->q_rank, l->kda_f_a, g->norm, rows, DS4_GVQ_BF16) ||
        !gvq_mm(g, m, g->heads, l->kda_f_b, g->q_rank, rows, DS4_GVQ_BF16) ||
        !gvq_mm(g, m, g->kv_rank, l->kda_beta, g->norm, rows, DS4_GVQ_BF16) ||
        !gvq_mm(g, m, g->q_rank, l->kda_g_a, g->norm, rows, DS4_GVQ_BF16) ||
        !gvq_mm(g, m, g->ffn_gate, l->kda_g_b, g->q_rank, rows, DS4_GVQ_BF16)) { return false; }
    return ds4_gpu_gvq_kda(g->ffn_up, g->recurrent[il],
        g->q_conv_state[il], g->k_conv_state[il], g->v_conv_state[il],
        g->q, g->k, g->v, g->heads, g->kv_rank, g->q_conv_weight[il],
        g->k_conv_weight[il], g->v_conv_weight[il], g->decay_scale[il],
        g->dt_bias[il], rows, DS4_N_HEAD, DS4_KDA_GATE_CLAMP_MIN) &&
        ds4_gpu_gvq_rms(g->gvq, g->heads, g->ffn_up, g->ffn_gate,
            glmvq_index(m, l->kda_o_norm), rows * DS4_N_HEAD,
            DS4_N_KDA_HEAD_DIM, DS4_RMS_EPS, DS4_GVQ_GATED_RMS) &&
        gvq_mm(g, m, g->attn_out, l->kda_output, g->heads, rows, DS4_GVQ_BF16);
}

static bool gvq_index(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_layer_weights *l, uint32_t il, uint32_t pos, uint32_t rows) {
    if (g->ctx_cap <= DS4_GLM53_INDEX_TOPK) { return true; }
    // Build pools during the dense prefix too: a later sparse continuation
    // must see all prior keys and the same incomplete four-token tail.
    if (!gvq_mm(g, m, g->idx_k, l->indexer_attn_k, g->norm, rows, DS4_GVQ_BF16) ||
        !gvq_mm(g, m, g->idx_gate, l->indexer_compressor_gate, g->norm, rows, DS4_GVQ_BF16) ||
        !ds4_gpu_gvq_pool(g->gvq, g->layer_pool[il], g->layer_tail_k[il], g->layer_tail_gate[il],
            g->idx_k, g->idx_gate, glmvq_index(m, l->indexer_k_norm),
            glmvq_index(m, l->indexer_k_norm_b), glmvq_index(m, l->indexer_compressor_ape),
            pos, rows, g->ctx_cap)) { return false; }
    if (pos + rows <= DS4_GLM53_INDEX_TOPK) { return true; }

    const uint32_t pools = (pos + rows) / DS4_GLM53_POOL_SIZE;
    const uint32_t topk = DS4_GLM53_INDEX_TOPK / DS4_GLM53_POOL_SIZE;
    return gvq_mm(g, m, g->idx_q, l->indexer_attn_q_b, g->q_rank_norm, rows, DS4_GVQ_BF16) &&
        gvq_mm(g, m, g->idx_weights, l->indexer_proj, g->norm, rows, DS4_GVQ_BF16) &&
        ds4_gpu_gvq_index(g->gvq, g->idx_scores, g->idx_q, g->idx_weights,
            g->layer_pool[il], pools, rows, pos) &&
        ds4_gpu_indexer_topk_tensor(g->idx_pools, g->idx_scores, pools, rows, topk, 0u, UINT32_MAX) &&
        ds4_gpu_glm53_pool_expand(g->idx_selected, g->idx_pools, rows, pos,
            topk, DS4_GLM53_INDEX_TOPK, DS4_GLM53_MAX_SELECTED);
}

static bool gvq_dsa(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_layer_weights *l, uint32_t il, uint32_t pos, uint32_t rows) {
    const uint32_t keys = pos + rows;
    const uint32_t stride = keys > DS4_GLM53_INDEX_TOPK ? DS4_GLM53_MAX_SELECTED : 0u;
    if (keys > DS4_GVQ_CTX_MAX) { return false; }
    return gvq_mm(g, m, g->q_rank, l->attn_q_a, g->norm, rows, DS4_GVQ_BF16) &&
        gvq_norm(g, m, g->q_rank_norm, g->q_rank, l->attn_q_a_norm, rows, DS4_N_LORA_Q) &&
        gvq_mm(g, m, g->q, l->attn_q_b, g->q_rank_norm, rows, DS4_GVQ_BF16) &&
        gvq_mm(g, m, g->kv_rank, l->attn_kv_a_mqa, g->norm, rows, DS4_GVQ_BF16) &&
        gvq_norm(g, m, g->kv_rank_norm, g->kv_rank, l->attn_kv_a_norm, rows, DS4_N_KV_LORA) &&
        gvq_index(g, m, l, il, pos, rows) &&
        ds4_gpu_gvq_store(g->layer_kv[il], g->kv_rank_norm, rows, pos, g->ctx_cap, DS4_N_KV_LORA) &&
        ds4_gpu_gvq_dsa(g->gvq, g->heads, g->q, g->layer_kv[il],
            glmvq_index(m, l->attn_kv_b), g->gvq_latent, g->gvq_kv,
            g->gvq_pos, pos, rows, DS4_N_HEAD, DS4_N_KEY_MLA,
            stride ? g->idx_selected : NULL, stride) &&
        gvq_mm(g, m, g->attn_out, l->attn_output, g->heads, rows, DS4_GVQ_BF16);
}

static bool gvq_mlp(ds4_glm53_gpu_graph *g, const ds4_model *m,
        ds4_gpu_tensor *out, const ds4_tensor *gate, const ds4_tensor *up,
        const ds4_tensor *down, uint32_t rows) {
    return gvq_mm(g, m, g->ffn_gate, gate, g->norm, rows, DS4_GVQ_BF16) &&
        gvq_mm(g, m, g->ffn_up, up, g->norm, rows, DS4_GVQ_BF16) &&
        ds4_gpu_gvq_swiglu(g->ffn_mid, g->ffn_gate, g->ffn_up,
            (uint64_t)rows * gate->dim[1], DS4_SWIGLU_CLAMP_EXP) &&
        gvq_mm(g, m, out, down, g->ffn_mid, rows, DS4_GVQ_BF16);
}

static bool gvq_ffn(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_weights *w, uint32_t il, uint32_t rows) {
    const ds4_layer_weights *l = &w->layer[il];
    if (il < DS4_N_LEADING_DENSE) {
        return gvq_mlp(g, m, g->ffn_out, l->ffn_gate, l->ffn_up, l->ffn_down, rows);
    }
    uint32_t matrices[3 * GLMVQ_EXPERTS];
    const ds4_glmvq_experts *e = &w->glmvq[il];
    for (uint32_t expert = 0; expert < GLMVQ_EXPERTS; expert++) {
        matrices[expert] = glmvq_index(m, e->gate[expert]);
        matrices[GLMVQ_EXPERTS + expert] = glmvq_index(m, e->up[expert]);
        matrices[2 * GLMVQ_EXPERTS + expert] = glmvq_index(m, e->down[expert]);
    }
    return gvq_mm(g, m, g->kv_rank, l->ffn_gate_inp, g->norm, rows, DS4_GVQ_F32) &&
        ds4_gpu_gvq_router(g->gvq, g->selected, g->route_weights, g->kv_rank,
            glmvq_index(m, l->ffn_exp_probs_b), rows, DS4_EXPERT_WEIGHT_SCALE) &&
        ds4_gpu_gvq_moe(g->gvq, g->ffn_out, g->q, g->ffn_gate, g->ffn_up,
            g->ffn_mid, g->routed_down, g->norm, g->selected, g->route_weights, matrices, rows, DS4_SWIGLU_CLAMP_EXP) &&
        gvq_mlp(g, m, g->attn_out, l->ffn_gate_shexp, l->ffn_up_shexp, l->ffn_down_shexp, rows) &&
        ds4_gpu_add_tensor(g->ffn_out, g->ffn_out, g->attn_out, (uint64_t)rows * DS4_N_EMBD) &&
        ds4_gpu_gvq_round(g->ffn_out, (uint64_t)rows * DS4_N_EMBD);
}

static bool gvq_trace_tensor(const char *root, uint32_t il, const char *name,
        const ds4_gpu_tensor *tensor, uint64_t offset, uint64_t bytes) {
    char path[PATH_MAX];
    const int n = snprintf(path, sizeof(path), "%s/layer_%02u_%s.bin", root, il, name);
    if (n < 0 || (size_t)n >= sizeof(path)) { return false; }
    void *buffer = xmalloc((size_t)bytes);
    const bool read = ds4_gpu_tensor_read(tensor, offset, buffer, bytes) != 0;
    FILE *fp = read ? fopen(path, "wb") : NULL;
    bool ok = fp && fwrite(buffer, 1u, (size_t)bytes, fp) == bytes;
    if (fp && fclose(fp)) { ok = false; }
    free(buffer);
    return ok;
}

static bool gvq_trace(ds4_glm53_gpu_graph *g, uint32_t il, uint32_t rows, uint32_t pos) {
    const char *root = getenv("DS4_GLMVQ_TRACE");
    if (!root || !root[0] || pos) { return true; }
    const uint64_t hidden = (uint64_t)DS4_N_HC * DS4_N_EMBD * sizeof(float);
    if (!gvq_trace_tensor(root, il, "last_hidden", g->hc_next, (rows - 1u) * hidden, hidden)) { return false; }
    if (il < DS4_N_LEADING_DENSE) { return true; }
    const uint64_t router = (uint64_t)DS4_N_EXPERT * sizeof(float);
    const uint64_t route = (uint64_t)DS4_N_EXPERT_USED * sizeof(float);
    return gvq_trace_tensor(root, il, "router_logits", g->kv_rank, (rows - 1u) * router, router) &&
        gvq_trace_tensor(root, il, "router_weights", g->route_weights, (rows - 1u) * route, route) &&
        gvq_trace_tensor(root, il, "router_indices", g->selected, (rows - 1u) * route, route);
}

static bool gvq_layer(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_weights *w, uint32_t il, uint32_t rows, uint32_t pos) {
    const ds4_layer_weights *l = &w->layer[il];
    if (!gvq_hc(g, m, l->hc_attn_fn, l->hc_attn_scale, l->hc_attn_base, l->attn_norm, g->hc_cur, rows)) {
        return false;
    }
    const bool attention = ds4_glm53_layer_is_kda(il)
        ? gvq_kda(g, m, l, il, rows) : gvq_dsa(g, m, l, il, pos, rows);
    return attention && ds4_gpu_gvq_expand(g->hc_after_attn, g->attn_out, g->hc_cur,
            g->hc_split, rows, DS4_N_EMBD) &&
        gvq_hc(g, m, l->hc_ffn_fn, l->hc_ffn_scale, l->hc_ffn_base, l->ffn_norm, g->hc_after_attn, rows) &&
        gvq_ffn(g, m, w, il, rows) &&
        ds4_gpu_gvq_expand(g->hc_next, g->ffn_out, g->hc_after_attn, g->hc_split, rows, DS4_N_EMBD) &&
        gvq_trace(g, il, rows, pos);
}

static bool gvq_output(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_weights *w, uint32_t rows, float *logits) {
    const uint64_t bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    ds4_gpu_tensor *hc = ds4_gpu_tensor_view(g->hc_cur,
        (uint64_t)(rows - 1u) * DS4_N_HC * bytes, DS4_N_HC * bytes);
    ds4_gpu_tensor *cur = ds4_gpu_tensor_view(g->cur, (uint64_t)(rows - 1u) * bytes, bytes);
    ds4_gpu_tensor *norm = ds4_gpu_tensor_view(g->norm, (uint64_t)(rows - 1u) * bytes, bytes);
    bool ok = hc && cur && norm &&
        ds4_gpu_hc_weighted_sum_tensor(cur, hc, g->hc_mean, DS4_N_EMBD, DS4_N_HC) &&
        ds4_gpu_gvq_round(cur, DS4_N_EMBD) && ds4_gpu_tensor_copy(g->last_hidden, 0u, cur, 0u, bytes);
    if (ok && logits) {
        ok = gvq_norm(g, m, norm, cur, w->output_norm, 1u, DS4_N_EMBD) &&
            gvq_mm(g, m, g->logits, w->output, norm, 1u, DS4_GVQ_BF16) &&
            ds4_gpu_tensor_read(g->logits, 0u, logits, (uint64_t)DS4_N_VOCAB * sizeof(float));
    }
    ds4_gpu_tensor_free(norm); ds4_gpu_tensor_free(cur); ds4_gpu_tensor_free(hc);
    return ok;
}
