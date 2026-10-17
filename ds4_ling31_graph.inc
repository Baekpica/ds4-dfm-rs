/* Native GPU state only. Rust owns artifact validation and input processing.
 *
 * Ling-3.1-flash alternates 63 recurrent KDA blocks with 9 latent MLA blocks
 * (every eighth) over a plain BF16 residual. A KDA block owns a fixed
 * 64 x 128 x 128 F32 recurrent tile plus three convolution tails and never
 * grows with context; an MLA block owns a linear BF16 latent cache of
 * 512 + 64 values per token (10,368 bytes per token over the nine blocks).
 *
 *   embed ─▶ [ rms ─▶ KDA | MLA ─▶ +res ─▶ rms ─▶ dense | MoE+shared ─▶ +res ] x72 ─▶ rms ─▶ head
 *
 * Weights stay in the GVQ pointer plan: Q4_K / FP8 / BF16 projections and
 * the KDA kernels are GLM VQ's, the routed experts Ling's own runtime
 * (ds4_ling31_gpu.cuh), MLA rope/cache/attention Ling-3.0's primitives with
 * k_b/v_b restored once from the FP8 kv_b. */

typedef struct {
    uint32_t ctx_cap, row_cap, position;
    bool ready, failed, borrowed;
    /* Diagnostic switches, default on: fused residual-norm passes and BF16
     * decay/gate/heads rows (DS4_LING31_FUSED=0, bit-identical either way),
     * BF16 q/k/v prefill rows (DS4_LING31_QKV_BF16=0 keeps them F32), the
     * tensor-core decode/verify attention (DS4_LING31_FLASH=0: Motif's). */
    bool fused, qkv_bf16, flash;
    uint64_t workspace_bytes, state_bytes, control_bytes, kv_bytes;
#define L31_FIELD(field, cols, type) ds4_gpu_tensor *field;
    L31_ROW_BUFFERS(L31_FIELD)
#undef L31_FIELD
    ds4_gpu_tensor *q_abs, *latent_out;          // absorbed MLA, L31_MLA_EXPAND_MIN rows
    ds4_gpu_tensor *k_full, *value, *attn_tmp, *lse, *lse_tmp; // views (expanded MLA)
    ds4_gpu_tensor *logits, *inv_freq;
    ds4_gpu_tensor *state_pool, *control_pool, *last_hidden;
    ds4_gpu_tensor *recurrent[DS4_MAX_LAYER], *q_tail[DS4_MAX_LAYER], *k_tail[DS4_MAX_LAYER],
        *v_tail[DS4_MAX_LAYER];
    ds4_gpu_tensor *q_conv[DS4_MAX_LAYER], *k_conv[DS4_MAX_LAYER], *v_conv[DS4_MAX_LAYER],
        *a_log[DS4_MAX_LAYER], *dt_bias[DS4_MAX_LAYER];
    ds4_gpu_tensor *kv_norm_w[DS4_MAX_LAYER], *k_b[DS4_MAX_LAYER], *v_b[DS4_MAX_LAYER];
    ds4_gpu_tensor *latent[DS4_MAX_LAYER], *k_pe[DS4_MAX_LAYER];
    uint32_t matrices[LING31_LAYERS + 1u][3u * L31_EXPERTS];
    int32_t *host_positions;
    /* MTP (mtp_drafts > 0): the recursive draft hidden, a verify's hidden
     * rows (row 0 is the pre-trial hidden) and logits, and the KDA journal
     * (slot r: state after verify row r). Draft-layer rows [0, mtp_len) are
     * filled from true hidden rows. */
    ds4_gpu_tensor *mtp_hidden, *mtp_rows, *mtp_logits, *mtp_journal;
    uint32_t mtp_drafts, mtp_len, mtp_start, mtp_n;
    bool mtp_capture;
    int mtp_trial[L31_MTP_ROWS];
#ifndef __APPLE__
    ds4_gvq_ctx *gvq;
    ds4_l31_moe *moe;
#endif
} ds4_ling31_graph;

/* Every graph owns its state pool (and the views into it) and its latent
 * rows. A bank graph borrows everything else from graph 0: forwards are
 * serialized, so scratch, controls and MTP verify buffers are shared. */
static void ling31_graph_free(ds4_ling31_graph *g) {
    if (!g) { return; }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor *t[] = {g->recurrent[il], g->q_tail[il], g->k_tail[il], g->v_tail[il],
            g->latent[il], g->k_pe[il]};
        for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) { ds4_gpu_tensor_free(t[i]); }
    }
    ds4_gpu_tensor_free(g->last_hidden);
    ds4_gpu_tensor_free(g->state_pool);
    if (g->borrowed) {
        memset(g, 0, sizeof(*g));
        return;
    }
#ifndef __APPLE__
    ds4_gpu_l31_moe_free(g->moe);
    ds4_gpu_gvq_free(g->gvq);
#endif
#define L31_FREE(field, cols, type) ds4_gpu_tensor_free(g->field);
    L31_ROW_BUFFERS(L31_FREE)
#undef L31_FREE
    ds4_gpu_tensor *views[] = {g->k_full, g->value, g->attn_tmp, g->lse, g->lse_tmp};
    for (size_t i = 0; i < sizeof(views) / sizeof(views[0]); i++) { ds4_gpu_tensor_free(views[i]); }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor *t[] = {g->q_conv[il], g->k_conv[il], g->v_conv[il], g->a_log[il],
            g->dt_bias[il], g->kv_norm_w[il], g->k_b[il], g->v_b[il]};
        for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) { ds4_gpu_tensor_free(t[i]); }
    }
    ds4_gpu_tensor_free(g->q_abs);
    ds4_gpu_tensor_free(g->latent_out);
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->inv_freq);
    ds4_gpu_tensor *mtp[] = {g->mtp_hidden, g->mtp_rows, g->mtp_logits, g->mtp_journal};
    for (size_t i = 0; i < sizeof(mtp) / sizeof(mtp[0]); i++) { ds4_gpu_tensor_free(mtp[i]); }
    ds4_gpu_tensor_free(g->control_pool);
    free(g->host_positions);
    memset(g, 0, sizeof(*g));
}

static bool ling31_graph_reset(ds4_ling31_graph *g) {
    if (!g || !g->ready) { return false; }
    if (g->failed && !ds4_gpu_synchronize()) { return false; }
    g->position = 0u;
    g->failed = false;
    g->mtp_len = g->mtp_n = 0u;
    /* A recurrent block's state is its history: a reset zeroes it. */
    return ds4_gpu_tensor_fill_f32(g->state_pool, 0.0f, g->state_bytes / sizeof(float)) != 0;
}

static bool ling31_take(ds4_gpu_tensor **out, ds4_gpu_tensor *pool, uint64_t *cursor, uint64_t bytes) {
    return solar_pool_take(out, pool, cursor, bytes);
}

typedef enum { LING31_HEAD_LAST, LING31_HEAD_SKIP } ling31_head_mode;

#ifndef __APPLE__
static bool ling31_control(ds4_ling31_graph *g, ds4_gpu_tensor *out, const ds4_model *m,
        const ds4_tensor *weight, uint64_t count) {
    return ds4_gpu_gvq_control(g->gvq, out, ling31_index(m, weight), count) != 0;
}

/* Per-graph controls (conv taps, A_log, dt_bias, latent norm as F32; k_b/v_b
 * restored from FP8), uploaded once; banks borrow the owner's pool. */
static bool ling31_graph_controls(ds4_ling31_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t layers) {
    g->control_bytes = ling31_control_bytes();
    g->control_pool = ds4_gpu_tensor_alloc(g->control_bytes);
    if (!g->control_pool) { return false; }
    const uint64_t conv = (uint64_t)L31_KDA_DIM * L31_CONV * sizeof(float);
    const uint64_t half = (uint64_t)L31_HEADS * L31_KV_LORA * L31_NOPE * sizeof(uint16_t);
    uint64_t cursor = 0u;
    for (uint32_t il = 0; il < layers; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        if (ds4_ling31_layer_is_mla(il)) {
            if (!ling31_take(&g->kv_norm_w[il], g->control_pool, &cursor, L31_KV_LORA * sizeof(float)) ||
                !ling31_take(&g->k_b[il], g->control_pool, &cursor, half) ||
                !ling31_take(&g->v_b[il], g->control_pool, &cursor, half) ||
                !ling31_control(g, g->kv_norm_w[il], m, l->attn_kv_a_norm, L31_KV_LORA) ||
                !ds4_gpu_l31_kv_b(g->gvq, g->k_b[il], g->v_b[il], ling31_index(m, l->attn_kv_b))) {
                return false;
            }
            continue;
        }
        if (!ling31_take(&g->q_conv[il], g->control_pool, &cursor, conv) ||
            !ling31_take(&g->k_conv[il], g->control_pool, &cursor, conv) ||
            !ling31_take(&g->v_conv[il], g->control_pool, &cursor, conv) ||
            !ling31_take(&g->a_log[il], g->control_pool, &cursor, L31_HEADS * sizeof(float)) ||
            !ling31_take(&g->dt_bias[il], g->control_pool, &cursor, L31_KDA_DIM * sizeof(float)) ||
            !ling31_control(g, g->q_conv[il], m, l->kda_q_conv, conv / sizeof(float)) ||
            !ling31_control(g, g->k_conv[il], m, l->kda_k_conv, conv / sizeof(float)) ||
            !ling31_control(g, g->v_conv[il], m, l->kda_v_conv, conv / sizeof(float)) ||
            !ling31_control(g, g->a_log[il], m, l->kda_a_log, L31_HEADS) ||
            !ling31_control(g, g->dt_bias[il], m, l->kda_dt_bias, L31_KDA_DIM)) {
            return false;
        }
    }
    return cursor <= g->control_bytes;
}

static bool ling31_graph_state(ds4_ling31_graph *g, uint32_t layers) {
    g->state_bytes = ling31_state_bytes();
    g->state_pool = ds4_gpu_tensor_alloc(g->state_bytes);
    if (!g->state_pool) { return false; }
    const uint64_t tile = (uint64_t)L31_KDA_DIM * L31_KDA_HEAD * sizeof(float);
    const uint64_t tail = (uint64_t)L31_KDA_DIM * L31_CONV * sizeof(float);
    uint64_t cursor = 0u;
    for (uint32_t il = 0; il < LING31_LAYERS; il++) {
        if (ds4_ling31_layer_is_mla(il)) { continue; }
        if (!ling31_take(&g->recurrent[il], g->state_pool, &cursor, tile) ||
            !ling31_take(&g->q_tail[il], g->state_pool, &cursor, tail) ||
            !ling31_take(&g->k_tail[il], g->state_pool, &cursor, tail) ||
            !ling31_take(&g->v_tail[il], g->state_pool, &cursor, tail)) { return false; }
    }
    if (!ling31_take(&g->last_hidden, g->state_pool, &cursor, (uint64_t)L31_EMBD * sizeof(float)) ||
        cursor != g->state_bytes) { return false; }
    for (uint32_t il = 0; il < layers; il++) {
        if (!ds4_ling31_layer_is_mla(il)) { continue; }
        g->latent[il] = ds4_gpu_tensor_alloc((uint64_t)g->ctx_cap * L31_KV_LORA * sizeof(uint16_t));
        g->k_pe[il] = ds4_gpu_tensor_alloc((uint64_t)g->ctx_cap * L31_ROPE * sizeof(uint16_t));
        if (!g->latent[il] || !g->k_pe[il]) { return false; }
        g->kv_bytes += ling31_latent_bytes(g->ctx_cap);
    }
    return true;
}

static bool ling31_rope_table(ds4_ling31_graph *g) {
    // rope_scaling is null: theta 6e6 over the 64 rotary dims at every context.
    float inv[L31_ROPE / 2u];
    for (uint32_t j = 0; j < L31_ROPE / 2u; j++) {
        inv[j] = (float)pow((double)L31_ROPE_THETA, -2.0 * j / L31_ROPE);
    }
    return ds4_gpu_tensor_write(g->inv_freq, 0, inv, sizeof(inv)) != 0;
}

static bool ling31_mtp_alloc(ds4_ling31_graph *g, uint32_t drafts) {
    if (!drafts) { return true; }
    g->mtp_drafts = drafts;
    g->mtp_hidden = ds4_gpu_tensor_alloc((uint64_t)L31_EMBD * sizeof(float));
    g->mtp_rows = ds4_gpu_tensor_alloc((L31_MTP_ROWS + 1u) * (uint64_t)L31_EMBD * sizeof(float));
    g->mtp_logits = ds4_gpu_tensor_alloc((uint64_t)L31_MTP_ROWS * DS4_N_VOCAB * sizeof(float));
    g->mtp_journal = ds4_gpu_tensor_alloc((uint64_t)drafts * g->state_bytes);
    return g->mtp_hidden && g->mtp_rows && g->mtp_logits && g->mtp_journal;
}

// `drafts` > 0 runs the MTP layer (its part must be mapped).
static bool ling31_graph_alloc(ds4_ling31_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t ctx, uint32_t cap, uint32_t drafts) {
    memset(g, 0, sizeof(*g));
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_LING31_VQ || !w->glmvq_gpu || !ctx || !cap ||
        cap > ctx || ctx > L31_CONTEXT || drafts > L31_MTP_DRAFTS ||
        (drafts && !ling31_has_mtp(m))) { return false; }
    g->ctx_cap = ctx;
    g->row_cap = cap;
    const uint32_t layers = drafts ? LING31_LAYERS + 1u : LING31_LAYERS;
    g->gvq = ds4_gpu_gvq_create_linear(w->glmvq_gpu, ling31_gvq_scratch(cap), cap);
    g->moe = g->gvq ? ds4_gpu_l31_moe_create(g->gvq, cap) : NULL;
    if (!g->gvq || !g->moe) { ling31_graph_free(g); return false; }
#define L31_ALLOC(field, cols, type) do { \
        const uint64_t bytes_ = (uint64_t)cap * (cols) * sizeof(type); \
        g->field = ds4_gpu_tensor_alloc(bytes_); \
        if (!g->field) { \
            fprintf(stderr, "ds4: Ling-3.1 graph allocation failed: %s\n", #field); \
            ling31_graph_free(g); \
            return false; \
        } \
    } while (0);
    L31_ROW_BUFFERS(L31_ALLOC)
#undef L31_ALLOC
    const uint32_t absorbed = cap < L31_MLA_EXPAND_MIN ? cap : L31_MLA_EXPAND_MIN;
    g->q_abs = ds4_gpu_tensor_alloc((uint64_t)absorbed * L31_LATENT * sizeof(float));
    g->latent_out = ds4_gpu_tensor_alloc((uint64_t)absorbed * L31_LATENT * sizeof(float));
    g->logits = ds4_gpu_tensor_alloc((uint64_t)DS4_N_VOCAB * sizeof(float));
    g->inv_freq = ds4_gpu_tensor_alloc(L31_ROPE / 2u * sizeof(float));
    // Expanded MLA: one key segment of `cap` rows of BF16 K (192) and V (128)
    // per head fits k+v; the merge operand fits f, both LSE vectors g.
    const uint64_t k_bytes = (uint64_t)cap * L31_QDIM * sizeof(uint16_t);
    const uint64_t v_bytes = (uint64_t)cap * L31_VDIM * sizeof(uint16_t);
    const uint64_t lse_bytes = (uint64_t)cap * L31_HEADS * sizeof(float);
    g->k_full = ds4_gpu_tensor_view(g->k, 0u, k_bytes);
    g->value = ds4_gpu_tensor_view(g->v, 0u, v_bytes);
    g->attn_tmp = ds4_gpu_tensor_view(g->f, 0u, (uint64_t)cap * L31_VDIM * sizeof(float));
    g->lse = ds4_gpu_tensor_view(g->g, 0u, lse_bytes);
    g->lse_tmp = ds4_gpu_tensor_view(g->g, lse_bytes, lse_bytes);
    g->host_positions = xmalloc((size_t)cap * 3u * sizeof(*g->host_positions));
    if (!g->q_abs || !g->latent_out || !g->logits || !g->inv_freq || !g->k_full || !g->value ||
        !g->attn_tmp || !g->lse || !g->lse_tmp || k_bytes > ds4_gpu_tensor_bytes(g->k) ||
        v_bytes > ds4_gpu_tensor_bytes(g->v) || !ling31_rope_table(g) ||
        !ling31_graph_controls(g, m, w, layers) || !ling31_graph_state(g, layers) ||
        !ling31_mtp_alloc(g, drafts)) {
        fprintf(stderr, "ds4: Ling-3.1 graph allocation failed\n");
        ling31_graph_free(g);
        return false;
    }
    for (uint32_t il = LING31_DENSE; il < layers; il++) {
        const ds4_ling31_experts *e = &w->ling31[il];
        for (uint32_t x = 0; x < L31_EXPERTS; x++) {
            g->matrices[il][x] = ling31_index(m, e->gate[x]);
            g->matrices[il][L31_EXPERTS + x] = ling31_index(m, e->up[x]);
            g->matrices[il][2u * L31_EXPERTS + x] = ling31_index(m, e->down[x]);
        }
    }
    g->workspace_bytes = ling31_workspace_bytes(cap);
    const char *fused = getenv("DS4_LING31_FUSED"), *qkv = getenv("DS4_LING31_QKV_BF16");
    const char *flash = getenv("DS4_LING31_FLASH");
    g->fused = !(fused && !strcmp(fused, "0"));
    g->flash = !(flash && !strcmp(flash, "0"));
    g->qkv_bf16 = !(qkv && !strcmp(qkv, "0"));
    g->ready = true;
    if (!ling31_graph_reset(g)) { ling31_graph_free(g); return false; }
    ds4_log(stderr, DS4_LOG_TIMING,
        "ds4: Ling-3.1 graph ctx=%u chunk=%u KV=%.2f GiB state=%.1f MiB control=%.1f MiB "
        "workspace=%.2f GiB\n", ctx, cap, (double)g->kv_bytes / 1073741824.0,
        (double)g->state_bytes / 1048576.0, (double)g->control_bytes / 1048576.0,
        (double)g->workspace_bytes / 1073741824.0);
    return true;
}

// A bank graph: its own state pool and latent rows, everything else borrowed.
static bool ling31_graph_bank(ds4_ling31_graph *g, const ds4_ling31_graph *owner) {
    *g = *owner;
    g->borrowed = true;
    g->state_pool = g->last_hidden = NULL;
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        g->recurrent[il] = g->q_tail[il] = g->k_tail[il] = g->v_tail[il] = NULL;
        g->latent[il] = g->k_pe[il] = NULL;
    }
    g->kv_bytes = 0;
    g->ready = false;
    if (!ling31_graph_state(g, owner->mtp_drafts ? LING31_LAYERS + 1u : LING31_LAYERS)) {
        ling31_graph_free(g);
        return false;
    }
    g->ready = true;
    if (!ling31_graph_reset(g)) { ling31_graph_free(g); return false; }
    return true;
}

static bool l31_mm(ds4_ling31_graph *g, const ds4_model *m, ds4_gpu_tensor *out,
        const ds4_tensor *w, const ds4_gpu_tensor *x, uint32_t rows, ds4_gvq_math math) {
    const bool ok = ds4_gpu_gvq_linear(g->gvq, out, ling31_index(m, w), x, rows, math) != 0;
    if (!ok && w) {
        fprintf(stderr, "ds4: Ling-3.1 linear %.*s failed (rows=%u)\n", (int)w->name.len, w->name.ptr, rows);
    }
    return ok;
}

// Projections of one input: prefill prepares it once (Q8_1 / FP16 / BF16).
static bool l31_mm_many(ds4_ling31_graph *g, const ds4_model *m, ds4_gpu_tensor *const *outs,
        const ds4_tensor *const *ws, uint32_t count, const ds4_gpu_tensor *x, uint32_t rows) {
    uint32_t index[4];
    for (uint32_t c = 0; c < count; c++) { index[c] = ling31_index(m, ws[c]); }
    const bool ok = count <= 4 && ds4_gpu_gvq_linear_many(g->gvq, outs, index, count, x, rows, DS4_GVQ_BF16);
    if (!ok) { fprintf(stderr, "ds4: Ling-3.1 linear group of %u failed (rows=%u)\n", count, rows); }
    return ok;
}

static bool l31_norm(ds4_ling31_graph *g, const ds4_model *m, ds4_gpu_tensor *out,
        const ds4_gpu_tensor *x, const ds4_tensor *w, uint32_t rows, uint32_t width) {
    return ds4_gpu_gvq_rms(g->gvq, out, x, NULL, ling31_index(m, w), rows, width,
        DS4_RMS_EPS, DS4_GVQ_RMS) != 0;
}

// Residual add at the source's BF16 boundary.
static bool l31_residual(ds4_gpu_tensor *cur, const ds4_gpu_tensor *branch, uint32_t rows) {
    const uint64_t count = (uint64_t)rows * L31_EMBD;
    return ds4_gpu_add_tensor(cur, cur, branch, (uint32_t)count) && ds4_gpu_gvq_round(cur, count);
}

// The residual, then the next RMS norm of cur into norm: one pass that
// matches the three kernels bit for bit.
static bool l31_residual_norm(ds4_ling31_graph *g, const ds4_model *m, const ds4_gpu_tensor *branch,
        const ds4_tensor *w, uint32_t rows) {
    if (g->fused) {
        return ds4_gpu_l31_residual_rms(g->gvq, g->cur, branch, g->norm, ling31_index(m, w), rows,
            L31_EMBD, DS4_RMS_EPS) != 0;
    }
    return l31_residual(g->cur, branch, rows) && l31_norm(g, m, g->norm, g->cur, w, rows, L31_EMBD);
}

/* KDA: q/k/v (Q4_K), full-rank decay f and output gate g (FP8), beta (BF16).
 * GVQ's recurrence takes raw projections (it applies conv + SiLU, the l2
 * norms, beta's sigmoid and -5*sigmoid(exp(A_log)*(f+dt_bias)) itself). */
static bool l31_kda(ds4_ling31_graph *g, const ds4_model *m, const ds4_layer_weights *l,
        uint32_t il, uint32_t rows) {
    ds4_gpu_tensor *const qkv[] = {g->q, g->k, g->v};
    const ds4_tensor *const qkv_w[] = {l->kda_q, l->kda_k, l->kda_v};
    ds4_gpu_tensor *const gates[] = {g->f, g->g};
    const ds4_tensor *const gates_w[] = {l->kda_f_a, l->kda_g_a};
    // Prefill chunks read q/k/v (decode's BF16 rounding), the FP8 decay and
    // gate (cuBLAS's BF16 results, unwidened) as BF16 rows, and the output
    // projection reads the gated norm's already rounded BF16 heads.
    const bool fused = !g->mtp_capture && ds4_gpu_gvq_kda_fused(g->gvq, rows);
    const uint32_t qkv_index[] = {ling31_index(m, l->kda_q), ling31_index(m, l->kda_k), ling31_index(m, l->kda_v)};
    const uint32_t gates_index[] = {ling31_index(m, l->kda_f_a), ling31_index(m, l->kda_g_a)};
    const ds4_gvq_rows qkv_rows = fused && g->qkv_bf16 &&
        ds4_gpu_gvq_linear_many_bf16(g->gvq, qkv, qkv_index, 3u, g->norm, rows) ? DS4_GVQ_ROWS_BF16 : DS4_GVQ_ROWS_F32;
    const ds4_gvq_rows gate_rows = fused && g->fused &&
        ds4_gpu_gvq_fp8_many_bf16(g->gvq, gates, gates_index, 2u, g->norm, rows) ? DS4_GVQ_ROWS_BF16 : DS4_GVQ_ROWS_F32;
    if ((qkv_rows == DS4_GVQ_ROWS_F32 && !l31_mm_many(g, m, qkv, qkv_w, 3u, g->norm, rows)) ||
        (gate_rows == DS4_GVQ_ROWS_F32 && !l31_mm_many(g, m, gates, gates_w, 2u, g->norm, rows)) ||
        !l31_mm(g, m, g->beta, l->kda_beta, g->norm, rows, DS4_GVQ_BF16)) { return false; }
    if (fused) {
        const uint32_t output = ling31_index(m, l->kda_output);
        const ds4_gvq_rows heads = g->fused && ds4_gpu_gvq_bf16_input(g->gvq, output, rows)
            ? DS4_GVQ_ROWS_BF16 : DS4_GVQ_ROWS_F32;
        const int done = ds4_gpu_gvq_kda_gated(g->gvq, g->heads, g->recurrent[il],
            g->q_tail[il], g->k_tail[il], g->v_tail[il], g->q, g->k, g->v, qkv_rows,
            g->f, gate_rows, g->beta, g->q_conv[il], g->k_conv[il], g->v_conv[il],
            g->a_log[il], g->dt_bias[il], g->g, gate_rows, ling31_index(m, l->kda_o_norm),
            rows, L31_HEADS, DS4_KDA_GATE_CLAMP_MIN, DS4_RMS_EPS, heads);
        if (done <= 0) { return false; }
        if (heads == DS4_GVQ_ROWS_F32) { return l31_mm(g, m, g->attn_out, l->kda_output, g->heads, rows, DS4_GVQ_BF16); }
        const bool ok = ds4_gpu_gvq_linear_rows(g->gvq, g->attn_out, output, g->heads, heads, rows) != 0;
        if (!ok) { fprintf(stderr, "ds4: Ling-3.1 KDA output projection failed (rows=%u)\n", rows); }
        return ok;
    }
    // MTP verify: journal slot r receives this layer's state after row r
    // (the kernel snapshots every row but the last; slots mirror the pool).
    ds4_gpu_tensor *snap = NULL;
    if (g->mtp_capture) {
        const uint64_t offset = (uint64_t)((const uint8_t *)ds4_gpu_tensor_ptr(g->recurrent[il]) -
            (const uint8_t *)ds4_gpu_tensor_ptr(g->state_pool));
        snap = ds4_gpu_tensor_view(g->mtp_journal, offset, ds4_gpu_tensor_bytes(g->mtp_journal) - offset);
        if (!snap) { return false; }
    }
    // The 128-wide gated norm runs a warp per row that loads before it
    // stores, so it normalizes the recurrence output in place.
    const bool recurred = ds4_gpu_gvq_kda(g->gvq, g->heads, g->recurrent[il], g->q_tail[il],
        g->k_tail[il], g->v_tail[il], g->q, g->k, g->v, g->f, g->beta, g->q_conv[il], g->k_conv[il],
        g->v_conv[il], g->a_log[il], g->dt_bias[il], rows, L31_HEADS, DS4_KDA_GATE_CLAMP_MIN,
        snap, g->state_bytes);
    ds4_gpu_tensor_free(snap);
    return recurred &&
        ds4_gpu_gvq_rms(g->gvq, g->heads, g->heads, g->g, ling31_index(m, l->kda_o_norm),
            rows * L31_HEADS, L31_KDA_HEAD, DS4_RMS_EPS, DS4_GVQ_GATED_RMS) &&
        l31_mm(g, m, g->attn_out, l->kda_output, g->heads, rows, DS4_GVQ_BF16);
}

/* Expanded MLA prefill (Ling-3.0's split): each key segment of the latent
 * cache becomes per-head BF16 K (192) / V (128), the range kernel attends
 * it and segments merge through their log-sum-exp. Prefix segments end at
 * pos so every segment starts at or before the first query. */
static bool l31_mla_expanded(ds4_ling31_graph *g, uint32_t il, uint32_t rows, uint32_t pos, float scale) {
    const uint32_t end = pos + rows;
    bool first = true;
    for (uint32_t seg0 = 0; seg0 < end;) {
        const uint32_t limit = seg0 < pos ? pos : end;
        const uint32_t len = limit - seg0 < g->row_cap ? limit - seg0 : g->row_cap;
        ds4_gpu_tensor *out = first ? g->heads : g->attn_tmp;
        ds4_gpu_tensor *lse = first ? g->lse : g->lse_tmp;
        if (!ds4_gpu_l31_expand(g->k_full, g->value, g->latent[il], g->k_pe[il], g->k_b[il], g->v_b[il],
                seg0, len) ||
            !ds4_gpu_ling3vl_expanded_attn(out, lse, g->q, g->k_full, g->value, rows, pos, len, seg0,
                L31_HEADS, L31_KEY, L31_VALUE, scale, 1) ||
            (!first && !ds4_gpu_motif3_merge_attention_states_tensor(g->heads, g->lse, g->attn_tmp,
                g->lse_tmp, rows, L31_HEADS, L31_VALUE))) { return false; }
        first = false;
        seg0 += len;
    }
    return true;
}

/* Absorbed MLA: k_b folded into Q so scores land on the stored latent row,
 * then v_b expands the latent-basis output (decode, verify, short appends). */
static bool l31_mla_absorbed(ds4_ling31_graph *g, uint32_t il, uint32_t rows, uint32_t pos, float scale) {
    if (!ds4_gpu_l31_absorb(g->q_abs, g->q, g->k_b[il], rows)) { return false; }
    // Decode and verify rows stream every latent key once on tensor cores:
    // 64K keys 1.7 ms -> 0.33 ms per layer against Motif's FP32 kernel.
    if (g->flash && ds4_gpu_l31_flash(g->gvq, g->latent_out, g->q, g->q_abs, g->latent[il], g->k_pe[il],
            rows, pos, scale)) {
        return ds4_gpu_l31_value(g->heads, g->latent_out, g->v_b[il], rows);
    }
    // MTP verify widths below the tensor-core kernel's 8 rows would take
    // Motif's general kernel (1.4 ms per call against 80 us for the split
    // decode kernel): attend each row at its own position, as decode does.
    const uint32_t step = rows < L31_MLA_TC_MIN ? 1u : rows;
    const uint64_t qb = (uint64_t)L31_QDIM * sizeof(float), lb = (uint64_t)L31_LATENT * sizeof(float);
    bool ok = true;
    for (uint32_t r = 0; ok && r < rows; r += step) {
        ds4_gpu_tensor *q = ds4_gpu_tensor_view(g->q, r * qb, step * qb);
        ds4_gpu_tensor *q_abs = ds4_gpu_tensor_view(g->q_abs, r * lb, step * lb);
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(g->latent_out, r * lb, step * lb);
        ok = q && q_abs && out && ds4_gpu_ling3vl_latent_attn(out, q, q_abs, g->latent[il], g->k_pe[il],
            step, pos + r, g->ctx_cap, 0u, L31_HEADS, L31_KV_LORA, L31_NOPE, L31_ROPE, scale);
        ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(q_abs);
        ds4_gpu_tensor_free(q);
    }
    return ok && ds4_gpu_l31_value(g->heads, g->latent_out, g->v_b[il], rows);
}

static bool l31_mla(ds4_ling31_graph *g, const ds4_model *m, const ds4_layer_weights *l,
        uint32_t il, uint32_t rows, uint32_t pos) {
    const float scale = 1.0f / sqrtf((float)L31_KEY);
    ds4_gpu_tensor *const lows[] = {g->q_rank, g->kv_raw};
    const ds4_tensor *const lows_w[] = {l->attn_q_a, l->attn_kv_a_mqa};
    if (!l31_mm_many(g, m, lows, lows_w, 2u, g->norm, rows) ||
        !l31_mm(g, m, g->head_gate, l->attn_gate, g->norm, rows, DS4_GVQ_BF16) ||
        !l31_norm(g, m, g->q_rank_norm, g->q_rank, l->attn_q_a_norm, rows, L31_Q_RANK) ||
        !l31_mm(g, m, g->q, l->attn_q_b, g->q_rank_norm, rows, DS4_GVQ_BF16) ||
        /* Interleaved rope on every head's 64-dim tail and on the shared key
         * tail of the 576-wide latent row, before either is cached. */
        !ds4_gpu_ling3vl_mrope(g->q, g->positions, g->inv_freq, rows, L31_HEADS, L31_KEY,
            L31_NOPE, L31_ROPE, L31_ROPE / 2u, 0u, 1.0f) ||
        !ds4_gpu_ling3vl_mrope(g->kv_raw, g->positions, g->inv_freq, rows, 1u, L31_KV_ROW,
            L31_KV_LORA, L31_ROPE, L31_ROPE / 2u, 0u, 1.0f) ||
        !ds4_gpu_l31_latent_norm(g->kv_norm, g->kv_raw, g->kv_norm_w[il], rows, DS4_RMS_EPS) ||
        !ds4_gpu_ling3vl_store_latent(g->latent[il], g->k_pe[il], g->kv_norm, g->kv_raw, rows, pos,
            g->ctx_cap, L31_KV_ROW, L31_KV_LORA, L31_ROPE)) { return false; }
    const bool attended = rows >= L31_MLA_EXPAND_MIN
        ? l31_mla_expanded(g, il, rows, pos, scale) : l31_mla_absorbed(g, il, rows, pos, scale);
    // head_wise gate: one sigmoid per head on its 128 values, then dense.
    return attended && ds4_gpu_step37_gate(g->heads, g->head_gate, L31_HEADS, rows) &&
        l31_mm(g, m, g->attn_out, l->attn_output, g->heads, rows, DS4_GVQ_BF16);
}

// SwiGLU MLP (dense FFN or shared expert): fused Q4_K gate/up at decode and
// verify widths, separate projections otherwise. `mid` follows `gate`.
static bool l31_mlp(ds4_ling31_graph *g, const ds4_model *m, ds4_gpu_tensor *out,
        const ds4_tensor *gate, const ds4_tensor *up, const ds4_tensor *down,
        uint32_t rows, float clamp) {
    const uint32_t width = (uint32_t)gate->dim[1];
    if (ds4_gpu_l31_gate_up(g->gvq, g->f, ling31_index(m, gate), ling31_index(m, up), g->norm, rows, clamp)) {
        return l31_mm(g, m, out, down, g->f, rows, DS4_GVQ_BF16);
    }
    ds4_gpu_tensor *const outs[] = {g->k, g->v};
    const ds4_tensor *const ws[] = {gate, up};
    return l31_mm_many(g, m, outs, ws, 2u, g->norm, rows) &&
        ds4_gpu_l31_swiglu(g->f, g->k, g->v, rows, width, clamp, DS4_GVQ_ROWS_F32) &&
        l31_mm(g, m, out, down, g->f, rows, DS4_GVQ_BF16);
}

static bool l31_ffn(ds4_ling31_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t il, uint32_t rows) {
    const ds4_layer_weights *l = &w->layer[il];
    if (il < LING31_DENSE) {
        return l31_mlp(g, m, g->ffn_out, l->ffn_gate, l->ffn_up, l->ffn_down, rows, INFINITY);
    }
    // FP32 router logits (BF16 weights widened), grouped sigmoid top-8; the
    // shared expert runs first so the routed reduce adds it in its pass.
    return l31_mm(g, m, g->router, l->ffn_gate_inp, g->norm, rows, DS4_GVQ_F32) &&
        ds4_gpu_ling3vl_router(g->selected, g->route_weights, g->router, m->map, m->size,
            l->ffn_exp_probs_b->abs_offset, rows, DS4_EXPERT_WEIGHT_SCALE) &&
        l31_mlp(g, m, g->attn_out, l->ffn_gate_shexp, l->ffn_up_shexp, l->ffn_down_shexp, rows,
            ling31_clamp(il, true)) &&
        ds4_gpu_l31_moe(g->moe, g->ffn_out, g->norm, g->selected, g->route_weights, g->matrices[il],
            rows, ling31_clamp(il, false), g->attn_out);
}

/* Layer il over norm = attn_norm(cur). Each residual also writes the RMS
 * norm its successor reads: ffn_norm, then `next` (the next layer's
 * attn_norm; NULL leaves the plain residual for the head). */
static bool l31_block(ds4_ling31_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t il, uint32_t rows, uint32_t pos, const ds4_tensor *next) {
    const ds4_layer_weights *l = &w->layer[il];
    const bool ok = (ds4_ling31_layer_is_mla(il) ? l31_mla(g, m, l, il, rows, pos) : l31_kda(g, m, l, il, rows)) &&
        l31_residual_norm(g, m, g->attn_out, l->ffn_norm, rows) &&
        l31_ffn(g, m, w, il, rows) &&
        (next ? l31_residual_norm(g, m, g->ffn_out, next, rows) : l31_residual(g->cur, g->ffn_out, rows));
    if (!ok) { fprintf(stderr, "ds4: Ling-3.1 layer %u failed (rows=%u pos=%u)\n", il, rows, pos); }
    return ok;
}

static bool l31_layer(ds4_ling31_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t il, uint32_t rows, uint32_t pos) {
    return l31_norm(g, m, g->norm, g->cur, w->layer[il].attn_norm, rows, L31_EMBD) &&
        l31_block(g, m, w, il, rows, pos, NULL);
}

// The 72 trunk layers over the embedded rows in cur.
static bool l31_trunk(ds4_ling31_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t rows, uint32_t pos) {
    if (!l31_norm(g, m, g->norm, g->cur, w->layer[0].attn_norm, rows, L31_EMBD)) { return false; }
    for (uint32_t il = 0; il < LING31_LAYERS; il++) {
        const ds4_tensor *next = il + 1u < LING31_LAYERS ? w->layer[il + 1u].attn_norm : NULL;
        if (!l31_block(g, m, w, il, rows, pos, next)) { return false; }
    }
    return true;
}

static bool l31_embed(ds4_ling31_graph *g, const ds4_model *m, const ds4_weights *w,
        const int *tokens, uint32_t rows, uint32_t pos) {
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t a = 0; a < 3u; a++) { g->host_positions[r * 3u + a] = (int32_t)(pos + r); }
    }
    return ds4_gpu_tensor_write(g->token, 0u, tokens, (uint64_t)rows * sizeof(int32_t)) &&
        ds4_gpu_tensor_write(g->positions, 0u, g->host_positions, (uint64_t)rows * 3u * sizeof(int32_t)) &&
        ds4_gpu_embed_tokens_quant_tensor(g->cur, g->token, m->map, m->size, w->token_embd->abs_offset,
            w->token_embd->type, (uint32_t)w->token_embd->dim[1], rows, L31_EMBD);
}

// Final norm and head of `row`; last_hidden keeps the post-norm row (the
// MTP predictor's input, vLLM bailing_moe_v3_mtp.py previous_hidden_states).
static bool l31_head(ds4_ling31_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t row,
        bool logits) {
    const uint64_t bytes = (uint64_t)L31_EMBD * sizeof(float);
    ds4_gpu_tensor *cur = ds4_gpu_tensor_view(g->cur, (uint64_t)row * bytes, bytes);
    ds4_gpu_tensor *norm = ds4_gpu_tensor_view(g->norm, 0u, bytes);
    const bool ok = cur && norm && l31_norm(g, m, norm, cur, w->output_norm, 1u, L31_EMBD) &&
        ds4_gpu_gvq_copy(g->last_hidden, 0u, norm, 0u, bytes, 1u) &&
        (!logits || l31_mm(g, m, g->logits, w->output, norm, 1u, DS4_GVQ_BF16));
    ds4_gpu_tensor_free(norm);
    ds4_gpu_tensor_free(cur);
    return ok;
}

#include "ds4_ling31_mtp.inc"

static bool ling31_forward(ds4_ling31_graph *g, const ds4_model *m, const ds4_weights *w,
        const int *tokens, uint32_t rows, uint32_t pos, ling31_head_mode head) {
    if (g->failed || !g->ready || !rows || rows > g->row_cap || pos != g->position ||
        pos > g->ctx_cap || rows > g->ctx_cap - pos) { return false; }
    for (uint32_t i = 0; i < rows; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) { return false; }
    }
    /* A failed enqueue leaves partially written KV and recurrent state; the
     * session must reset before another forward. */
    g->failed = true;
    if (!l31_embed(g, m, w, tokens, rows, pos) || !l31_trunk(g, m, w, rows, pos)) { return false; }
    const bool logits = head == LING31_HEAD_LAST;
    if (!(g->mtp_drafts ? l31_mtp_head(g, m, w, tokens, rows, pos, logits)
                        : l31_head(g, m, w, rows - 1u, logits))) { return false; }
    g->position = pos + rows;
    g->failed = false;
    return true;
}
#else
// Ling-3.1 is CUDA-only; Metal keeps linkage and refuses the graph.
static bool ling31_graph_alloc(ds4_ling31_graph *g, const ds4_model *m, const ds4_weights *w,
        uint32_t ctx, uint32_t cap, uint32_t drafts) {
    (void)g; (void)m; (void)w; (void)ctx; (void)cap; (void)drafts;
    return false;
}

static bool ling31_forward(ds4_ling31_graph *g, const ds4_model *m, const ds4_weights *w,
        const int *tokens, uint32_t rows, uint32_t pos, ling31_head_mode head) {
    (void)g; (void)m; (void)w; (void)tokens; (void)rows; (void)pos; (void)head;
    return false;
}

static bool ling31_graph_bank(ds4_ling31_graph *g, const ds4_ling31_graph *owner) {
    (void)g; (void)owner;
    return false;
}
#endif
