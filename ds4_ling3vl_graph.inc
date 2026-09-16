/* Native GPU state only. Rust owns artifact validation and input processing.
 *
 * Ling-3.0-flash-VL alternates 35 recurrent KDA blocks with 7 latent MLA
 * blocks, so the per-layer state is not uniform: a KDA block owns a fixed
 * 128x128 recurrent tile plus three convolution rings and never grows with
 * context, while an MLA block owns a linear BF16 latent cache of 576 values
 * per token. At 65536 tokens that is 72 MiB of recurrent state against
 * 504 MiB of latent cache for the whole model.
 *
 * The KDA recurrence, the latent attention and the routed-MoE operators are
 * the Solar/GLM, Motif and Step primitives; only the grouped router, the
 * VL M-RoPE, the fused-row latent RMSNorm and the BF16 absorb pair are
 * family-specific. */


typedef struct {
    ds4_exaone_batch_ws ws;
    bool ws_owned;

    ds4_gpu_tensor *positions, *tokens, *inv_freq, *logits;
    ds4_gpu_tensor *kv_raw, *kv_norm, *q_absorbed, *latent_out;
    ds4_gpu_tensor *kda_decay, *kda_gate, *kda_beta, *kda_scratch;
    ds4_gpu_tensor *attn_gate;
    /* Packed RMSNorm activations for the layer's BF16 projections. */
    ds4_gpu_tensor *norm_bf16;

    /* KDA control tensors, uploaded once from the model map. */
    ds4_gpu_tensor *state_pool, *control_pool;
    ds4_gpu_tensor *recurrent[LING3VL_LAYERS];
    ds4_gpu_tensor *q_conv_state[LING3VL_LAYERS];
    ds4_gpu_tensor *k_conv_state[LING3VL_LAYERS];
    ds4_gpu_tensor *v_conv_state[LING3VL_LAYERS];
    ds4_gpu_tensor *q_conv_weight[LING3VL_LAYERS];
    ds4_gpu_tensor *k_conv_weight[LING3VL_LAYERS];
    ds4_gpu_tensor *v_conv_weight[LING3VL_LAYERS];
    ds4_gpu_tensor *decay_scale[LING3VL_LAYERS];
    ds4_gpu_tensor *dt_bias[LING3VL_LAYERS];
    ds4_gpu_tensor *o_norm[LING3VL_LAYERS];

    /* MLA blocks: one linear BF16 latent cache and its rotated key tail. */
    ds4_gpu_tensor *latent_cache[LING3VL_LAYERS];
    ds4_gpu_tensor *k_pe_cache[LING3VL_LAYERS];

    /* Borrowed from the session: image rows already projected to hidden. */
    const ds4_ling3vl_media *media;

    uint32_t context, cap, position, high_water;
    uint64_t state_bytes, kv_bytes, workspace_bytes, bytes;
    int32_t *host_positions;
    bool failed, ready;
} ds4_ling3vl_graph;

static bool ds4_ling3vl_layer_is_kda(uint32_t il) {
    return il < LING3VL_LAYERS && ((il + 1u) % LING3VL_LAYER_GROUP) != 0u;
}

static bool ling3vl_matmul_tensor(
        ds4_gpu_tensor *out, const ds4_model *model, const ds4_tensor *weight,
        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t n_tokens) {
    if (weight && weight->type == DS4_TENSOR_BF16) {
        return ds4_gpu_ling3vl_matmul_bf16(
                   out, model->map, model->size, weight->abs_offset,
                   in_dim, out_dim, x, n_tokens) != 0;
    }
    return plain_graph_matmul_tensor(
        out, model, weight, in_dim, out_dim, x, n_tokens);
}

/* Prefill: several BF16 GEMMs read the same RMSNorm row.  Convert once.
 * Decode n=1 keeps the per-call path (warp GEMV). */
static bool ling3vl_norm_packed(const ds4_ling3vl_graph *g, uint32_t n) {
    if (!g || !g->norm_bf16 || n <= 1u) {
        return false;
    }
    const char *kill = getenv("DS4_LING3VL_NO_BF16_REUSE");
    return !(kill && kill[0] == '1');
}

static bool ling3vl_pack_norm(ds4_ling3vl_graph *g, uint32_t n) {
    if (!ling3vl_norm_packed(g, n)) {
        return true;
    }
    return ds4_gpu_f32_to_bf16(
               g->norm_bf16, g->ws.b_norm, (uint64_t)n * L3V_HIDDEN) != 0;
}

static bool ling3vl_matmul_norm(
        ds4_ling3vl_graph *g, ds4_gpu_tensor *out, const ds4_model *model,
        const ds4_tensor *weight, uint64_t in_dim, uint64_t out_dim,
        uint32_t n) {
    if (weight && weight->type == DS4_TENSOR_BF16 &&
        ling3vl_norm_packed(g, n)) {
        return ds4_gpu_matmul_bf16_input_tensor(
                   out, model->map, model->size, weight->abs_offset,
                   in_dim, out_dim, g->norm_bf16, n) != 0;
    }
    return ling3vl_matmul_tensor(
        out, model, weight, in_dim, out_dim, g->ws.b_norm, n);
}

/* Decode n=1: two BF16 GEMVs share one f32->bf16 convert of b_norm. */
static bool ling3vl_pair_ok(uint32_t n, const ds4_tensor *a,
                            const ds4_tensor *b) {
    const char *kill = getenv("DS4_LING3VL_NO_BF16_PAIR");
    return n == 1u && a && b && a->type == DS4_TENSOR_BF16 &&
        b->type == DS4_TENSOR_BF16 && !(kill && kill[0] == '1');
}

static bool ling3vl_matmul_pair(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1, const ds4_model *model,
        const ds4_tensor *w0, const ds4_tensor *w1, uint64_t in_dim,
        uint64_t out0_dim, uint64_t out1_dim, const ds4_gpu_tensor *x,
        uint32_t n) {
    if (!ling3vl_pair_ok(n, w0, w1)) {
        return false;
    }
    return ds4_gpu_ling3vl_gemv_pair(
               out0, out1, model->map, model->size, w0->abs_offset,
               w1->abs_offset, in_dim, out0_dim, out1_dim, x) != 0;
}

static uint32_t ling3vl_prefill_cap(uint32_t ctx) {
    /* 4096 halves routed-MoE and MLA launches vs 2048 on the 8K campaign.
     * Scratch grows linearly with the cap.  DS4_LING3VL_PREFILL_CHUNK=2048
     * restores the previous width. */
    enum { DEFAULT_CHUNK = 4096u, MAX_CHUNK = 4096u };
    uint32_t cap = DEFAULT_CHUNK;
    const char *value = getenv("DS4_LING3VL_PREFILL_CHUNK");
    if (value && *value) {
        char *end = NULL;
        const unsigned long parsed = strtoul(value, &end, 10);
        if (end != value && !*end && parsed > 0u && parsed <= MAX_CHUNK) {
            cap = (uint32_t)parsed;
        }
    }
    return cap < ctx ? cap : ctx;
}

static ds4_context_memory ling3vl_memory(uint32_t ctx, uint32_t cap) {
    ds4_context_memory m = {0};
    if (!ctx || !cap || cap > ctx || ctx > L3V_CONTEXT) { return m; }
    m.raw_cap = ctx;
    m.prefill_cap = cap;
    uint32_t mla = 0u, kda = 0u;
    for (uint32_t il = 0; il < LING3VL_LAYERS; il++) {
        if (ds4_ling3vl_layer_is_kda(il)) { kda++; } else { mla++; }
    }
    m.raw_bytes = (uint64_t)mla * ctx * L3V_KV_ROW * sizeof(uint16_t);
    /* The recurrent tile and its three convolution rings are context-free. */
    m.raw_bytes += (uint64_t)kda *
        ((uint64_t)L3V_KDA_DIM * L3V_KDA_HEAD + 3ull * L3V_KDA_DIM * L3V_CONV) *
        sizeof(float);
    const uint64_t common = 5ull * L3V_HIDDEN + 2ull * L3V_QDIM +
        2ull * L3V_KDA_DIM + 3ull * L3V_DENSE + 3ull * L3V_EXPERT_FF +
        L3V_EXPERTS + 2ull * L3V_USED + 3ull * L3V_USED * L3V_EXPERT_FF +
        (uint64_t)L3V_USED * L3V_HIDDEN;
    const uint64_t family = (uint64_t)L3V_KV_ROW + L3V_KV_LORA +
        2ull * L3V_LATENT + 2ull * L3V_KDA_DIM + L3V_HEADS + L3V_HEADS + 4ull;
    m.scratch_bytes =
        ((uint64_t)cap * (common + family) + L3V_MROPE_PAIRS + DS4_N_VOCAB) *
        sizeof(float);
    /* Packed RMSNorm BF16 rows reused by the layer's dense projections. */
    m.scratch_bytes += (uint64_t)cap * L3V_HIDDEN * sizeof(uint16_t);
    m.scratch_bytes += ds4_gpu_solar_kda_prefill_scratch_bytes(
        cap, L3V_HEADS, L3V_KDA_HEAD);
    /* control_pool: conv weights, decay, dt_bias, o_norm. Persistent per
     * graph, not in the checkpoint slab. */
    const uint64_t conv_bytes =
        (uint64_t)L3V_KDA_DIM * L3V_CONV * sizeof(float);
    m.scratch_bytes +=
        (3ull * conv_bytes + (uint64_t)L3V_HEADS * sizeof(float) +
         (uint64_t)L3V_KDA_DIM * sizeof(float) +
         (uint64_t)L3V_KDA_HEAD * sizeof(float)) *
        kda;
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}

static void ling3vl_graph_free(ds4_ling3vl_graph *g) {
    if (!g) { return; }
    if (g->ws_owned) { exaone_batch_ws_free(&g->ws); }
    ds4_gpu_tensor *all[] = {
        g->positions, g->tokens, g->inv_freq, g->logits, g->kv_raw, g->kv_norm,
        g->q_absorbed, g->latent_out, g->kda_decay, g->kda_gate, g->kda_beta,
        g->kda_scratch, g->attn_gate, g->norm_bf16,
    };
    for (unsigned i = 0; i < sizeof(all) / sizeof(all[0]); i++) {
        ds4_gpu_tensor_free(all[i]);
    }
    for (uint32_t il = 0; il < LING3VL_LAYERS; il++) {
        /* Recurrent and control tensors are views into the two pools. */
        ds4_gpu_tensor_free(g->latent_cache[il]);
        ds4_gpu_tensor_free(g->k_pe_cache[il]);
    }
    ds4_gpu_tensor_free(g->state_pool);
    ds4_gpu_tensor_free(g->control_pool);
    free(g->host_positions);
    memset(g, 0, sizeof(*g));
}

/* inv_freq[j] = rope_theta^(-2j/rope_dim); the kernel multiplies by the
 * axis position and reduces the angle in double. */
static bool ling3vl_upload_inv_freq(ds4_ling3vl_graph *g) {
    float inv[L3V_MROPE_PAIRS];
    for (uint32_t j = 0; j < L3V_MROPE_PAIRS; j++) {
        inv[j] = (float)pow((double)DS4_ROPE_FREQ_BASE,
                            -2.0 * (double)j / (double)L3V_ROPE);
        if (!isfinite(inv[j]) || inv[j] <= 0.0f) { return false; }
    }
    return ds4_gpu_tensor_write(g->inv_freq, 0, inv, sizeof(inv));
}

static bool ling3vl_scratch_alloc(ds4_ling3vl_graph *g, uint32_t ctx,
                                  uint32_t cap) {
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_LING3VL || !ctx || !cap ||
        cap > ctx || ctx > L3V_CONTEXT) { return false; }
    g->context = ctx;
    g->cap = cap;
    if (!exaone_batch_ws_alloc(&g->ws, cap, L3V_HIDDEN, L3V_QDIM,
                               L3V_KDA_DIM)) {
        return false;
    }
    g->ws_owned = true;
    g->workspace_bytes = g->ws.bytes;
#define L3V_ALLOC(field, elems, type) do {                                    \
    const uint64_t nbytes = (uint64_t)(elems) * sizeof(type);                 \
    g->field = ds4_gpu_tensor_alloc(nbytes);                                  \
    if (!g->field) {                                                          \
        fprintf(stderr, "ds4: Ling graph allocation failed: %s\n", #field);   \
        ling3vl_graph_free(g);                                                \
        return false;                                                         \
    }                                                                         \
    g->workspace_bytes += nbytes;                                             \
} while (0)
    L3V_ALLOC(positions, (uint64_t)cap * 3u, int32_t);
    L3V_ALLOC(tokens, cap, int32_t);
    L3V_ALLOC(inv_freq, L3V_MROPE_PAIRS, float);
    L3V_ALLOC(logits, DS4_N_VOCAB, float);
    L3V_ALLOC(kv_raw, (uint64_t)cap * L3V_KV_ROW, float);
    L3V_ALLOC(kv_norm, (uint64_t)cap * L3V_KV_LORA, float);
    L3V_ALLOC(q_absorbed, (uint64_t)cap * L3V_LATENT, float);
    L3V_ALLOC(latent_out, (uint64_t)cap * L3V_LATENT, float);
    L3V_ALLOC(kda_decay, (uint64_t)cap * L3V_KDA_DIM, float);
    L3V_ALLOC(kda_gate, (uint64_t)cap * L3V_KDA_DIM, float);
    L3V_ALLOC(kda_beta, (uint64_t)cap * L3V_HEADS, float);
    L3V_ALLOC(attn_gate, (uint64_t)cap * L3V_HEADS, float);
    L3V_ALLOC(norm_bf16, (uint64_t)cap * L3V_HIDDEN, uint16_t);
#undef L3V_ALLOC
    const uint64_t scratch_bytes = ds4_gpu_solar_kda_prefill_scratch_bytes(
        cap, L3V_HEADS, L3V_KDA_HEAD);
    if (scratch_bytes) {
        g->kda_scratch = ds4_gpu_tensor_alloc(scratch_bytes);
        if (!g->kda_scratch) { ling3vl_graph_free(g); return false; }
        g->workspace_bytes += scratch_bytes;
    }
    g->host_positions = xmalloc((size_t)cap * 3u * sizeof(*g->host_positions));
    if (!ling3vl_upload_inv_freq(g)) { ling3vl_graph_free(g); return false; }
    return true;
}

static bool ling3vl_graph_reset(ds4_ling3vl_graph *g) {
    if (!g) { return false; }
    if (g->failed && !ds4_gpu_synchronize()) { return false; }
    g->position = 0u;
    g->high_water = 0u;
    g->failed = false;
    /* A recurrent block has no positional index to rewind: the state itself
     * is the history, so a reset has to zero it. */
    return !g->state_pool ||
        ds4_gpu_tensor_fill_f32(g->state_pool, 0.0f,
                                g->state_bytes / sizeof(float));
}

static bool ling3vl_graph_alloc(ds4_ling3vl_graph *g, const ds4_model *model,
                                const ds4_weights *weights, uint32_t ctx,
                                uint32_t cap) {
    memset(g, 0, sizeof(*g));
    if (!model || !weights || !ling3vl_scratch_alloc(g, ctx, cap)) {
        return false;
    }
    const uint64_t recurrent_bytes =
        (uint64_t)L3V_KDA_DIM * L3V_KDA_HEAD * sizeof(float);
    const uint64_t conv_bytes = (uint64_t)L3V_KDA_DIM * L3V_CONV * sizeof(float);
    const uint64_t decay_bytes = (uint64_t)L3V_HEADS * sizeof(float);
    const uint64_t dt_bytes = (uint64_t)L3V_KDA_DIM * sizeof(float);
    const uint64_t norm_bytes = (uint64_t)L3V_KDA_HEAD * sizeof(float);
    uint32_t n_kda = 0u;
    for (uint32_t il = 0; il < LING3VL_LAYERS; il++) {
        if (ds4_ling3vl_layer_is_kda(il)) { n_kda++; }
    }
    g->state_bytes = (recurrent_bytes + 3u * conv_bytes) * n_kda;
    const uint64_t control_bytes =
        (3u * conv_bytes + decay_bytes + dt_bytes + norm_bytes) * n_kda;
    g->state_pool = ds4_gpu_tensor_alloc(g->state_bytes);
    g->control_pool = ds4_gpu_tensor_alloc(control_bytes);
    if (!g->state_pool || !g->control_pool) {
        fprintf(stderr, "ds4: Ling KDA pool allocation failed\n");
        ling3vl_graph_free(g);
        return false;
    }
    uint64_t state_cursor = 0u, control_cursor = 0u;
    for (uint32_t il = 0; il < LING3VL_LAYERS; il++) {
        if (!ds4_ling3vl_layer_is_kda(il)) { continue; }
        const ds4_layer_weights *l = &weights->layer[il];
        const bool ok =
            solar_pool_take(&g->recurrent[il], g->state_pool, &state_cursor,
                            recurrent_bytes) &&
            solar_pool_take(&g->q_conv_state[il], g->state_pool, &state_cursor,
                            conv_bytes) &&
            solar_pool_take(&g->k_conv_state[il], g->state_pool, &state_cursor,
                            conv_bytes) &&
            solar_pool_take(&g->v_conv_state[il], g->state_pool, &state_cursor,
                            conv_bytes) &&
            solar_pool_take(&g->q_conv_weight[il], g->control_pool,
                            &control_cursor, conv_bytes) &&
            solar_pool_take(&g->k_conv_weight[il], g->control_pool,
                            &control_cursor, conv_bytes) &&
            solar_pool_take(&g->v_conv_weight[il], g->control_pool,
                            &control_cursor, conv_bytes) &&
            solar_pool_take(&g->decay_scale[il], g->control_pool,
                            &control_cursor, decay_bytes) &&
            solar_pool_take(&g->dt_bias[il], g->control_pool, &control_cursor,
                            dt_bytes) &&
            solar_pool_take(&g->o_norm[il], g->control_pool, &control_cursor,
                            norm_bytes) &&
            solar_copy_control(g->q_conv_weight[il], model, l->ssm_q_conv,
                               conv_bytes) &&
            solar_copy_control(g->k_conv_weight[il], model, l->ssm_k_conv,
                               conv_bytes) &&
            solar_copy_control(g->v_conv_weight[il], model, l->ssm_v_conv,
                               conv_bytes) &&
            /* ssm_a is already exp(A_log) in this artifact. */
            solar_copy_control(g->decay_scale[il], model, l->ssm_a,
                               decay_bytes) &&
            solar_copy_control(g->dt_bias[il], model, l->ssm_dt_bias,
                               dt_bytes) &&
            solar_copy_control(g->o_norm[il], model, l->ssm_o_norm, norm_bytes);
        if (!ok) {
            fprintf(stderr, "ds4: Ling KDA control bind failed at layer %u\n", il);
            ling3vl_graph_free(g);
            return false;
        }
    }
    if (state_cursor != g->state_bytes || control_cursor != control_bytes) {
        ling3vl_graph_free(g);
        return false;
    }
    for (uint32_t il = 0; il < LING3VL_LAYERS; il++) {
        if (ds4_ling3vl_layer_is_kda(il)) { continue; }
        const uint64_t latent = (uint64_t)ctx * L3V_KV_LORA * sizeof(uint16_t);
        const uint64_t rope = (uint64_t)ctx * L3V_ROPE * sizeof(uint16_t);
        g->latent_cache[il] = ds4_gpu_tensor_alloc(latent);
        g->k_pe_cache[il] = ds4_gpu_tensor_alloc(rope);
        if (!g->latent_cache[il] || !g->k_pe_cache[il]) {
            fprintf(stderr, "ds4: Ling latent KV allocation failed at layer %u\n", il);
            ling3vl_graph_free(g);
            return false;
        }
        g->kv_bytes += latent + rope;
    }
    g->bytes = g->workspace_bytes + g->state_bytes + control_bytes + g->kv_bytes;
    g->ready = true;
    if (!ling3vl_graph_reset(g)) { ling3vl_graph_free(g); return false; }
    ds4_log(stderr, DS4_LOG_TIMING,
            "ds4: Ling graph ctx=%u chunk=%u KV=%.2f GiB state=%.1f MiB "
            "workspace=%.2f GiB\n",
            ctx, cap, (double)g->kv_bytes / 1073741824.0,
            (double)g->state_bytes / 1048576.0,
            (double)g->workspace_bytes / 1073741824.0);
    return true;
}

static bool ling3vl_layer_kda(ds4_ling3vl_graph *g, const ds4_model *m,
                              const ds4_layer_weights *l, uint32_t il,
                              uint32_t n) {
    ds4_exaone_batch_ws *b = &g->ws;
    const bool qk = ling3vl_matmul_pair(
        b->b_q, b->b_k, m, l->attn_q, l->attn_k, L3V_HIDDEN,
        L3V_KDA_DIM, L3V_KDA_DIM, b->b_norm, n);
    const bool vd = ling3vl_matmul_pair(
        b->b_v, g->kda_decay, m, l->attn_v, l->ssm_f_a, L3V_HIDDEN,
        L3V_KDA_DIM, L3V_KDA_DIM, b->b_norm, n);
    const bool gb = ling3vl_matmul_pair(
        g->kda_gate, g->kda_beta, m, l->ssm_g_a, l->ssm_beta, L3V_HIDDEN,
        L3V_KDA_DIM, L3V_HEADS, b->b_norm, n);
    if ((!qk && (!ling3vl_matmul_norm(g, b->b_q, m, l->attn_q, L3V_HIDDEN,
                                          L3V_KDA_DIM, n) ||
                 !ling3vl_matmul_norm(g, b->b_k, m, l->attn_k, L3V_HIDDEN,
                                          L3V_KDA_DIM, n))) ||
        (!vd && (!ling3vl_matmul_norm(g, b->b_v, m, l->attn_v, L3V_HIDDEN,
                                          L3V_KDA_DIM, n) ||
                 !ling3vl_matmul_norm(g, g->kda_decay, m, l->ssm_f_a,
                                          L3V_HIDDEN, L3V_KDA_DIM, n))) ||
        /* no_kda_lora: the decay and output gates are full rank, so each is
         * one projection instead of GLM's low-rank pair. */
        (!gb && (!ling3vl_matmul_norm(g, g->kda_gate, m, l->ssm_g_a,
                                          L3V_HIDDEN, L3V_KDA_DIM, n) ||
                 !ling3vl_matmul_norm(g, g->kda_beta, m, l->ssm_beta,
                                          L3V_HIDDEN, L3V_HEADS, n)))) {
        return false;
    }
    const bool recurrence = n > 1u
        ? ds4_gpu_ling3vl_kda_prefill_tensor(
              b->b_heads, g->kda_scratch, g->recurrent[il], g->q_conv_state[il],
              g->k_conv_state[il], g->v_conv_state[il], b->b_q, b->b_k, b->b_v,
              g->kda_decay, g->kda_beta, g->q_conv_weight[il],
              g->k_conv_weight[il], g->v_conv_weight[il], g->decay_scale[il],
              g->dt_bias[il], n, L3V_HEADS, L3V_KDA_HEAD, L3V_CONV,
              DS4_KDA_GATE_CLAMP_MIN)
        : ds4_gpu_ling3vl_kda_decode_tensor(
              b->b_heads, g->recurrent[il], g->q_conv_state[il],
              g->k_conv_state[il], g->v_conv_state[il], b->b_q, b->b_k, b->b_v,
              g->kda_decay, g->kda_beta, g->q_conv_weight[il],
              g->k_conv_weight[il], g->v_conv_weight[il], g->decay_scale[il],
              g->dt_bias[il], L3V_HEADS, L3V_KDA_HEAD, L3V_CONV,
              DS4_KDA_GATE_CLAMP_MIN);
    return recurrence &&
        /* Per-head RMSNorm over the 128 value dims, then the sigmoid gate.
         * b_q is the only free buffer wide enough for the 4096-value result;
         * the projections that filled it are dead once the recurrence ran. */
        ds4_gpu_solar_head_rms_sigmoid_gate_tensor(
            b->b_q, b->b_heads, g->kda_gate, g->o_norm[il], n,
            L3V_HEADS, L3V_KDA_HEAD, DS4_RMS_EPS) &&
        ling3vl_matmul_tensor(b->b_attn_out, m, l->attn_output, L3V_KDA_DIM,
                                  L3V_HIDDEN, b->b_q, n);
}

static bool ling3vl_layer_mla(ds4_ling3vl_graph *g, const ds4_model *m,
                              const ds4_layer_weights *l, uint32_t il,
                              uint32_t n, uint32_t pos) {
    ds4_exaone_batch_ws *b = &g->ws;
    const float scale = 1.0f / sqrtf((float)L3V_KEY_MLA);
    const bool qkv = ling3vl_matmul_pair(
        b->b_q, g->kv_raw, m, l->attn_q, l->attn_kv_a_mqa, L3V_HIDDEN,
        L3V_QDIM, L3V_KV_ROW, b->b_norm, n);
    if ((!qkv && (!ling3vl_matmul_norm(g, b->b_q, m, l->attn_q, L3V_HIDDEN,
                                           L3V_QDIM, n) ||
                  !ling3vl_matmul_norm(g, g->kv_raw, m, l->attn_kv_a_mqa,
                                           L3V_HIDDEN, L3V_KV_ROW, n))) ||
        !ling3vl_matmul_norm(g, g->attn_gate, m, l->attn_gate, L3V_HIDDEN,
                                 L3V_HEADS, n) ||
        /* Rotate the 64-dim tail of every Q head and of the single shared
         * key row, in place, before either is cached or absorbed. */
        !ds4_gpu_ling3vl_mrope(b->b_q, g->positions, g->inv_freq, n, L3V_HEADS,
                               L3V_KEY_MLA, L3V_NOPE, L3V_ROPE, L3V_SECTION_T,
                               L3V_SECTION_H) ||
        !ds4_gpu_ling3vl_mrope(g->kv_raw, g->positions, g->inv_freq, n, 1u,
                               L3V_KV_ROW, L3V_KV_LORA, L3V_ROPE,
                               L3V_SECTION_T, L3V_SECTION_H) ||
        /* kv_raw is 576-wide; the norm covers only the 512-value latent. */
        !ds4_gpu_ling3vl_rms_norm(g->kv_norm, g->kv_raw, m->map, m->size,
                                  l->attn_kv_a_norm->abs_offset, L3V_KV_LORA,
                                  L3V_KV_ROW, n, DS4_RMS_EPS) ||
        !ds4_gpu_ling3vl_store_latent(g->latent_cache[il], g->k_pe_cache[il],
                                      g->kv_norm, g->kv_raw, n, pos,
                                      g->context, L3V_KV_ROW, L3V_KV_LORA,
                                      L3V_ROPE) ||
        /* Absorb k_b into Q so attention scores against the stored latent. */
        !ds4_gpu_ling3vl_qk_absorb(g->q_absorbed, b->b_q, m->map, m->size,
                                   l->attn_k_b->abs_offset, n, L3V_HEADS,
                                   L3V_KEY_MLA, L3V_NOPE, L3V_KV_LORA)) {
        return false;
    }
    return ds4_gpu_ling3vl_latent_attn(
               g->latent_out, b->b_q, g->q_absorbed, g->latent_cache[il],
               g->k_pe_cache[il], n, pos, g->context, 0u, L3V_HEADS,
               L3V_KV_LORA, L3V_NOPE, L3V_ROPE, scale) &&
        ds4_gpu_ling3vl_value_project(b->b_heads, g->latent_out, m->map,
                                      m->size, l->attn_v_b->abs_offset, n,
                                      L3V_HEADS, L3V_KV_LORA, L3V_VALUE_MLA) &&
        /* head_wise gated attention: one sigmoid per head before the output
         * projection, the same shape Step's 128-wide gate applies. */
        ds4_gpu_step37_gate(b->b_heads, g->attn_gate, L3V_HEADS, n) &&
        ling3vl_matmul_tensor(b->b_attn_out, m, l->attn_output, L3V_VDIM,
                                  L3V_HIDDEN, b->b_heads, n);
}

static bool ling3vl_layer_ffn(ds4_ling3vl_graph *g, const ds4_model *m,
                              const ds4_layer_weights *l, uint32_t il,
                              uint32_t n) {
    ds4_exaone_batch_ws *b = &g->ws;
    if (!l->ffn_gate_exps) {
        return ling3vl_matmul_norm(g, b->b_dense_gate, m, l->ffn_gate,
                                       L3V_HIDDEN, L3V_DENSE, n) &&
            ling3vl_matmul_norm(g, b->b_dense_up, m, l->ffn_up, L3V_HIDDEN,
                                    L3V_DENSE, n) &&
            ds4_gpu_step37_swiglu(b->b_dense_mid, b->b_dense_gate, b->b_dense_up,
                                  NULL, L3V_DENSE, n, 0.0f) &&
            ling3vl_matmul_tensor(b->b_ffn_out, m, l->ffn_down, L3V_DENSE,
                                      L3V_HIDDEN, b->b_dense_mid, n) &&
            ds4_gpu_exaone_add_tensor(b->b_cur, b->b_ffn_out,
                                      (uint64_t)n * L3V_HIDDEN);
    }
    const float clamp = ling3vl_swiglu_clamp_exp(il);
    const float shared_clamp = ling3vl_swiglu_clamp_shexp(il);
    uint64_t gate_in = 0u, gate_out = 0u, gate_row = 0u;
    uint64_t down_in = 0u, down_out = 0u, down_row = 0u;
    (void)tensor_expert_bytes(m, l->ffn_gate_exps, 0u, &gate_in, &gate_out,
                              &gate_row);
    (void)tensor_expert_bytes(m, l->ffn_down_exps, 0u, &down_in, &down_out,
                              &down_row);
    if (gate_in != L3V_HIDDEN || gate_out != L3V_EXPERT_FF ||
        down_in != L3V_EXPERT_FF || down_out != L3V_HIDDEN) { return false; }
    if (!ling3vl_matmul_norm(g, b->b_router_logits, m, l->ffn_gate_inp,
                                 L3V_HIDDEN, L3V_EXPERTS, n) ||
        !ds4_gpu_ling3vl_router(b->b_router_selected, b->b_router_weights,
                                b->b_router_logits, m->map, m->size,
                                l->ffn_exp_probs_b->abs_offset, n,
                                DS4_EXPERT_WEIGHT_SCALE) ||
        !ds4_gpu_routed_gate_up_tensor(
            b->b_routed_gate, b->b_routed_up, b->b_norm, b->b_router_selected,
            m->map, m->size, l->ffn_gate_exps->abs_offset,
            l->ffn_gate_exps->bytes, l->ffn_up_exps->abs_offset,
            l->ffn_up_exps->bytes, l->ffn_gate_exps->type, L3V_HIDDEN,
            L3V_EXPERT_FF, L3V_EXPERTS, n, L3V_USED) ||
        /* No route weights before down: its activation quantizer is nonlinear. */
        !ds4_gpu_step37_swiglu(b->b_routed_mid, b->b_routed_gate, b->b_routed_up,
                               NULL, L3V_EXPERT_FF, n * L3V_USED, clamp) ||
        !ds4_gpu_routed_matmul_guarded_tensor(
            b->b_routed_down, b->b_routed_mid, b->b_router_selected, m->map,
            m->size, l->ffn_down_exps->abs_offset, l->ffn_down_exps->bytes,
            l->ffn_down_exps->type, L3V_EXPERT_FF, L3V_HIDDEN, L3V_EXPERTS,
            n * L3V_USED, 1u, n) ||
        !ds4_gpu_step37_sum(b->b_ffn_out, b->b_routed_down, b->b_router_weights,
                            L3V_HIDDEN, n) ||
        !(ling3vl_matmul_pair(
              b->b_shared_gate, b->b_shared_up, m, l->ffn_gate_shexp,
              l->ffn_up_shexp, L3V_HIDDEN, L3V_EXPERT_FF, L3V_EXPERT_FF,
              b->b_norm, n) ||
          (ling3vl_matmul_norm(g, b->b_shared_gate, m, l->ffn_gate_shexp,
                                   L3V_HIDDEN, L3V_EXPERT_FF, n) &&
           ling3vl_matmul_norm(g, b->b_shared_up, m, l->ffn_up_shexp,
                                   L3V_HIDDEN, L3V_EXPERT_FF, n))) ||
        !ds4_gpu_step37_swiglu(b->b_shared_mid, b->b_shared_gate, b->b_shared_up,
                               NULL, L3V_EXPERT_FF, n, shared_clamp) ||
        !ling3vl_matmul_tensor(b->b_shared_out, m, l->ffn_down_shexp,
                                   L3V_EXPERT_FF, L3V_HIDDEN, b->b_shared_mid, n) ||
        !ds4_gpu_exaone_add_tensor(b->b_ffn_out, b->b_shared_out,
                                   (uint64_t)n * L3V_HIDDEN)) {
        return false;
    }
    return ds4_gpu_exaone_add_tensor(b->b_cur, b->b_ffn_out,
                                     (uint64_t)n * L3V_HIDDEN);
}

static bool ling3vl_layer(ds4_ling3vl_graph *g, const ds4_model *m,
                          const ds4_weights *w, uint32_t il, uint32_t n,
                          uint32_t pos) {
    ds4_exaone_batch_ws *b = &g->ws;
    const ds4_layer_weights *l = &w->layer[il];
    if (!ds4_gpu_exaone_rms_norm_tensor(b->b_norm, b->b_cur, m->map, m->size,
                                        l->attn_norm->abs_offset, L3V_HIDDEN, n,
                                        DS4_RMS_EPS) ||
        !ling3vl_pack_norm(g, n)) {
        return false;
    }
    const bool attention = ds4_ling3vl_layer_is_kda(il)
        ? ling3vl_layer_kda(g, m, l, il, n)
        : ling3vl_layer_mla(g, m, l, il, n, pos);
    return attention &&
        ds4_gpu_exaone_add_tensor(b->b_cur, b->b_attn_out,
                                  (uint64_t)n * L3V_HIDDEN) &&
        ds4_gpu_exaone_rms_norm_tensor(b->b_norm, b->b_cur, m->map, m->size,
                                       l->ffn_norm->abs_offset, L3V_HIDDEN, n,
                                       DS4_RMS_EPS) &&
        ling3vl_pack_norm(g, n) &&
        ling3vl_layer_ffn(g, m, l, il, n);
}

static bool ling3vl_head(ds4_ling3vl_graph *g, const ds4_model *m,
                         const ds4_weights *w, uint32_t row) {
    if (row >= g->cap) { return false; }
    ds4_gpu_tensor *last = ds4_gpu_tensor_view(
        g->ws.b_cur, (uint64_t)row * L3V_HIDDEN * sizeof(float),
        L3V_HIDDEN * sizeof(float));
    const bool ok = last &&
        ds4_gpu_exaone_rms_norm_tensor(g->ws.b_norm, last, m->map, m->size,
                                       w->output_norm->abs_offset, L3V_HIDDEN,
                                       1u, DS4_RMS_EPS) &&
        ling3vl_matmul_tensor(g->logits, m, w->output, L3V_HIDDEN,
                                  DS4_N_VOCAB, g->ws.b_norm, 1u);
    ds4_gpu_tensor_free(last);
    return ok;
}

typedef enum { LING3VL_HEAD_LAST, LING3VL_HEAD_SKIP } ling3vl_head_mode;

static bool ling3vl_forward_impl(ds4_ling3vl_graph *g, const ds4_model *m,
                                 const ds4_weights *w, const int *tokens,
                                 uint32_t n, uint32_t pos,
                                 ling3vl_head_mode head) {
    if (g->failed || !g->ready || !n || n > g->cap || pos != g->position ||
        pos > g->context || n > g->context - pos) { return false; }
    for (uint32_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) { return false; }
    }
    /* Text rows advance all three M-RoPE axes together; an image span keeps
     * its own grid coordinates and the rows after it resume from the span's
     * widest axis. Media, when present, supplies the whole plan. */
    if (!ling3vl_media_positions(g->media, tokens, n, pos, g->host_positions)) {
        return false;
    }
    /* A failed enqueue leaves partially written KV and recurrent state that
     * cannot be resumed; the session must reset before another forward. */
    g->failed = true;
    if (!ds4_gpu_tensor_write(g->tokens, 0, tokens, (uint64_t)n * sizeof(int)) ||
        !ds4_gpu_tensor_write(g->positions, 0, g->host_positions,
                              (uint64_t)n * 3u * sizeof(int32_t)) ||
        !ds4_gpu_embed_tokens_quant_tensor(g->ws.b_cur, g->tokens, m->map,
                                           m->size, w->token_embd->abs_offset,
                                           w->token_embd->type, DS4_N_VOCAB, n,
                                           L3V_HIDDEN) ||
        !ling3vl_media_apply(g->media, g->ws.b_cur, pos, n)) {
        return false;
    }
    for (uint32_t il = 0; il < LING3VL_LAYERS; il++) {
        if (!ling3vl_layer(g, m, w, il, n, pos)) {
            fprintf(stderr, "ds4: Ling forward failed at layer %u\n", il);
            return false;
        }
    }
    const bool ok = head == LING3VL_HEAD_SKIP || ling3vl_head(g, m, w, n - 1u);
    if (ok) {
        g->position = pos + n;
        if (g->high_water < g->position) { g->high_water = g->position; }
        g->failed = false;
    }
    return ok;
}

static bool ling3vl_forward(ds4_ling3vl_graph *g, const ds4_model *m,
                            const ds4_weights *w, const int *tokens,
                            uint32_t n, uint32_t pos) {
    return ling3vl_forward_impl(g, m, w, tokens, n, pos, LING3VL_HEAD_LAST);
}
