#include "ds4_glm53_compact.h"
#include "ds4_glm53_attn.h"

/* GLM-5.3's compact graph follows the pinned native DSA/indexer algebra.
 * Layer-major prefill batches the projections and MoE work while each KDA
 * layer carries its sequence state. This is a required execution path;
 * throughput and artifact-level arithmetic gates remain separate evidence. */
enum {
    DS4_GLM53_RESIDENT_CTX_MAX = 1048576u,
    GLM53_DIAG_CTX_MAX = 2048u,
    GLM53_PREFILL_DEFAULT = 2048u,
    GLM53_PREFILL_MAX = 2048u,
    GLM53_PREFILL_WINDOW = 4096u,
    GLM53_QUANT_Q8 = 8u
};
static const float GLM53_POOL_EPS = 1.0e-6f;

static uint64_t glm53_dense_reserve(uint32_t rows, uint32_t ctx) {
    const char *value = getenv("DS4_GLM53_DENSE_GEMM");
    if (!value || strcmp(value, "0") == 0 || rows < GLM53_DENSE_MIN_ROWS) { return 0u; }
    return glm53_dense_bytes(rows, ctx);
}

#define GLM53_MTP_STATE
#include "ds4_glm53_mtp.inc"
#undef GLM53_MTP_STATE

static bool glm53_mtp_step(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_weights *w, const ds4_gpu_tensor *hidden, int token,
        uint32_t pos, int *next);

static uint32_t glm53_graph_row_cap(uint32_t ctx, uint32_t slots) {
    (void)slots; /* Routing is subdivided by its actual expert union. */
    uint32_t rows = GLM53_PREFILL_DEFAULT;
    const char *value = getenv("DS4_GLM53_PREFILL_ROWS");
    if (value && value[0]) {
        char *end = NULL;
        const unsigned long parsed = strtoul(value, &end, 10);
        if (!end || *end || parsed < 1ul || parsed > GLM53_PREFILL_MAX) {
            fprintf(stderr, "ds4: invalid DS4_GLM53_PREFILL_ROWS (use 1..2048)\n");
            return 0u;
        }
        rows = (uint32_t)parsed;
    }
    return rows < ctx ? rows : ctx;
}

static uint32_t glm53_window_cap(uint32_t ctx, uint32_t slots) {
    const char *value = getenv("DS4_GLM53_PREFILL_WINDOW");
    if ((value && strcmp(value, "4096") != 0) || ctx < GLM53_PREFILL_WINDOW ||
        slots < 2u * DS4_N_EXPERT ||
        glm53_graph_row_cap(ctx, slots) < GLM53_DENSE_MIN_ROWS) { return 0u; }
    return GLM53_PREFILL_WINDOW;
}

static uint32_t glm53_prefill_cap(uint32_t ctx, uint32_t slots) {
    const uint32_t window = glm53_window_cap(ctx, slots);
    return window ? window : glm53_graph_row_cap(ctx, slots);
}

static uint64_t glm53_window_bytes(uint32_t window) {
    return 2u * (uint64_t)window * DS4_N_HC * DS4_N_EMBD * sizeof(float);
}

static bool glm53_graph_diag(void) {
    const char *value = getenv("DS4_GLM53_DSA_EXPANDED");
    return value && strcmp(value, "1") == 0;
}

static uint64_t glm53_graph_ffn_cols(void) {
    const uint64_t routed = (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    return DS4_N_FF_DENSE > routed ? DS4_N_FF_DENSE : routed;
}

/* One allocation schema also defines exact views and the admission quote.
 * Do not pass capacity-sized views to mHC, which infers its row count. */
#define GLM53_ROW_BUFFERS(X) \
    X(token, 1u, int32_t) \
    X(cur, DS4_N_EMBD, float) \
    X(norm, DS4_N_EMBD, float) \
    X(hc_cur, (uint64_t)DS4_N_HC * DS4_N_EMBD, float) \
    X(hc_next, (uint64_t)DS4_N_HC * DS4_N_EMBD, float) \
    X(hc_flat, (uint64_t)DS4_N_HC * DS4_N_EMBD, float) \
    X(hc_mix, (uint64_t)DS4_N_HC * (DS4_N_HC + 2u), float) \
    X(hc_split, (uint64_t)DS4_N_HC * (DS4_N_HC + 2u), float) \
    X(hc_after_attn, (uint64_t)DS4_N_HC * DS4_N_EMBD, float) \
    X(hc_mean, DS4_N_HC, float) \
    X(q_rank, DS4_N_LORA_Q, float) \
    X(q_rank_norm, DS4_N_LORA_Q, float) \
    X(kv_rank, DS4_N_KV_LORA, float) \
    X(kv_rank_norm, DS4_N_KV_LORA, float) \
    X(q, (uint64_t)DS4_N_HEAD * DS4_N_KEY_MLA, float) \
    X(k, (uint64_t)DS4_N_HEAD * DS4_N_KEY_MLA, float) \
    X(v, (uint64_t)DS4_N_HEAD * DS4_N_KEY_MLA, float) \
    X(heads, (uint64_t)DS4_N_HEAD * DS4_N_KEY_MLA, float) \
    X(attn_out, DS4_N_EMBD, float) \
    X(ffn_gate, glm53_graph_ffn_cols(), float) \
    X(ffn_up, glm53_graph_ffn_cols(), float) \
    X(ffn_mid, glm53_graph_ffn_cols(), float) \
    X(ffn_out, DS4_N_EMBD, float) \
    X(routed_down, (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD, float) \
    X(selected, DS4_N_EXPERT_USED, int32_t) \
    X(route_weights, DS4_N_EXPERT_USED, float) \
    X(low_q, (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA, float) \
    X(low_out, (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA, float) \
    X(idx_q, (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM, float) \
    X(idx_k, DS4_N_INDEXER_HEAD_DIM, float) \
    X(idx_gate, DS4_N_INDEXER_HEAD_DIM, float) \
    X(idx_weights, DS4_N_INDEXER_HEAD, float) \
    X(idx_scores, g->pool_cap, float) \
    X(idx_pools, DS4_GLM53_INDEX_TOPK / DS4_GLM53_POOL_SIZE, uint32_t) \
    X(idx_selected, DS4_GLM53_MAX_SELECTED, uint32_t)

static void glm53_graph_layers(uint32_t *kda, uint32_t *dsa) {
    *kda = 0u;
    *dsa = 0u;
    const uint32_t layers = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    for (uint32_t il = 0; il < layers; il++) {
        if (ds4_glm53_layer_is_kda(il)) { (*kda)++; }
        else { (*dsa)++; }
    }
}

static uint64_t glm53_graph_state_bytes(void) {
    uint32_t kda, dsa;
    glm53_graph_layers(&kda, &dsa);
    const uint64_t dim = (uint64_t)DS4_N_HEAD * DS4_N_KDA_HEAD_DIM;
    return (dim * DS4_N_KDA_HEAD_DIM + 3u * dim * DS4_N_SSM_CONV) * sizeof(float) * kda +
        2u * (uint64_t)DS4_GLM53_POOL_SIZE * DS4_N_INDEXER_HEAD_DIM * sizeof(float) * dsa +
        2u * (uint64_t)DS4_N_EMBD * sizeof(float);
}

static uint64_t glm53_graph_bytes_plan(uint32_t ctx_cap, uint32_t rows, uint32_t window) {
    if (!ctx_cap || ctx_cap > DS4_GLM53_RESIDENT_CTX_MAX ||
        !rows || rows > ctx_cap || rows > GLM53_PREFILL_MAX ||
        (window && (window != GLM53_PREFILL_WINDOW || window > ctx_cap ||
                    rows < GLM53_DENSE_MIN_ROWS)) ||
        (glm53_graph_diag() && ctx_cap > GLM53_DIAG_CTX_MAX)) { return 0u; }
    ds4_glm53_gpu_graph quote = {0};
    ds4_glm53_gpu_graph *g = &quote;
    g->pool_cap = (ctx_cap + DS4_GLM53_POOL_SIZE - 1u) / DS4_GLM53_POOL_SIZE;
    uint64_t bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
#define GLM53_QUOTE(field, cols, type) bytes += (uint64_t)rows * (cols) * sizeof(type);
    GLM53_ROW_BUFFERS(GLM53_QUOTE)
#undef GLM53_QUOTE
    bytes += ds4_gpu_solar_kda_prefill_scratch_bytes(rows, DS4_N_HEAD, DS4_N_KDA_HEAD_DIM);
    bytes += glm53_dense_reserve(rows, ctx_cap);
    bytes += glm53_window_bytes(window);

    uint32_t kda, dsa;
    glm53_graph_layers(&kda, &dsa);
    const uint64_t kda_dim = (uint64_t)DS4_N_HEAD * DS4_N_KDA_HEAD_DIM;
    const uint64_t conv = kda_dim * DS4_N_SSM_CONV * sizeof(float);
    const uint64_t state = kda_dim * DS4_N_KDA_HEAD_DIM * sizeof(float) + 3u * conv;
    const uint64_t control = 3u * conv + ((uint64_t)DS4_N_HEAD + kda_dim +
                             DS4_N_KDA_HEAD_DIM) * sizeof(float);
    const uint64_t tail = 2u * (uint64_t)DS4_GLM53_POOL_SIZE * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    bytes += (uint64_t)kda * (state + control) + (uint64_t)dsa * tail;
    const uint64_t cache = glm53_graph_diag()
        ? (uint64_t)ctx_cap * DS4_N_HEAD * DS4_N_KEY_MLA * 2u * sizeof(uint16_t)
        : ((uint64_t)ctx_cap * DS4_N_KV_LORA +
           (uint64_t)g->pool_cap * DS4_N_INDEXER_HEAD_DIM) * sizeof(uint16_t);
    return bytes + (uint64_t)dsa * cache + 2u * (uint64_t)DS4_N_EMBD * sizeof(float);
}

static uint64_t glm53_graph_bytes_for(uint32_t ctx_cap, uint32_t slots) {
    return glm53_graph_bytes_plan(ctx_cap, glm53_graph_row_cap(ctx_cap, slots),
                                  glm53_window_cap(ctx_cap, slots));
}

static uint64_t glm53_graph_bytes_estimate(uint32_t ctx_cap) {
    return glm53_graph_bytes_for(ctx_cap, 0u);
}

typedef enum { GLM53_MTP_OFF, GLM53_MTP_ON } glm53_mtp_mode;

static uint64_t glm53_session_bytes(uint32_t ctx, uint32_t rows,
        uint32_t window, glm53_mtp_mode mode) {
    const uint64_t base = glm53_graph_bytes_plan(ctx, rows, window);
    if (!base || mode == GLM53_MTP_OFF) { return base; }
    return base + glm53_mtp_bytes(ctx) + GLM53_MTP_SAVES * glm53_graph_state_bytes();
}

static void glm53_graph_unbind(ds4_glm53_gpu_graph *g) {
#define GLM53_UNBIND(field, cols, type) ds4_gpu_tensor_free(g->field); g->field = NULL;
    GLM53_ROW_BUFFERS(GLM53_UNBIND)
#undef GLM53_UNBIND
}

static bool glm53_graph_bind(ds4_glm53_gpu_graph *active,
                              ds4_glm53_gpu_graph *owner, uint32_t rows) {
    if (!active || !owner || !rows || rows > owner->row_cap) { return false; }
    *active = *owner;
    ds4_glm53_gpu_graph *g = active;
#define GLM53_CLEAR(field, cols, type) g->field = NULL;
    GLM53_ROW_BUFFERS(GLM53_CLEAR)
#undef GLM53_CLEAR
#define GLM53_BIND(field, cols, type) do { \
        g->field = ds4_gpu_tensor_view(owner->field, 0u, (uint64_t)rows * (cols) * sizeof(type)); \
        if (!g->field) { glm53_graph_unbind(g); return false; } \
    } while (0);
    GLM53_ROW_BUFFERS(GLM53_BIND)
#undef GLM53_BIND
    return true;
}

static void glm53_graph_free(ds4_glm53_gpu_graph *g) {
    if (!g) { return; }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor *views[] = {
            g->recurrent[il], g->q_conv_state[il], g->k_conv_state[il], g->v_conv_state[il],
            g->q_conv_weight[il], g->k_conv_weight[il], g->v_conv_weight[il],
            g->decay_scale[il], g->dt_bias[il], g->o_norm[il],
            g->layer_tail_k[il], g->layer_tail_gate[il], g->layer_kv[il], g->layer_pool[il]
        };
        for (size_t i = 0; i < sizeof(views) / sizeof(views[0]); i++) { ds4_gpu_tensor_free(views[i]); }
    }
    glm53_graph_unbind(g);
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->kda_scratch);
    ds4_gpu_tensor_free(g->state_pool);
    ds4_gpu_tensor_free(g->control_pool);
    ds4_gpu_tensor_free(g->last_hidden);
    ds4_gpu_tensor_free(g->mtp_hidden);
    glm53_mtp_free(g);
    memset(g, 0, sizeof(*g));
}

static bool glm53_graph_reset(ds4_glm53_gpu_graph *g) {
    if (!g || !g->ready || !g->state_pool || !g->state_bytes ||
        !ds4_gpu_tensor_fill_f32(g->state_pool, 0.0f, g->state_bytes / sizeof(float))) {
        return false;
    }
    g->cache_len = 0u;
    g->last_rows = 0u;
    g->mtp_len = 0u;
    g->mtp_min = 0u;
    g->mtp_trial_n = 0u;
    g->mtp_rewind_n = 0u;
    g->mtp_rewind_epoch = 0u;
    return true;
}

static bool glm53_graph_state(ds4_glm53_gpu_graph *g,
                               const ds4_model *model, const ds4_weights *weights) {
    const uint64_t kda_dim = (uint64_t)DS4_N_HEAD * DS4_N_KDA_HEAD_DIM;
    const uint64_t recurrent_bytes = kda_dim * DS4_N_KDA_HEAD_DIM * sizeof(float);
    const uint64_t conv_bytes = kda_dim * DS4_N_SSM_CONV * sizeof(float);
    const uint64_t decay_bytes = (uint64_t)DS4_N_HEAD * sizeof(float);
    const uint64_t dt_bytes = kda_dim * sizeof(float);
    const uint64_t norm_bytes = (uint64_t)DS4_N_KDA_HEAD_DIM * sizeof(float);
    const uint64_t tail_bytes = (uint64_t)DS4_GLM53_POOL_SIZE * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    uint32_t kda, dsa;
    glm53_graph_layers(&kda, &dsa);
    const uint64_t hidden_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    g->state_bytes = (recurrent_bytes + 3u * conv_bytes) * kda + 2u * tail_bytes * dsa + 2u * hidden_bytes;
    g->control_bytes = (3u * conv_bytes + decay_bytes + dt_bytes + norm_bytes) * kda;
    g->state_pool = ds4_gpu_tensor_alloc(g->state_bytes);
    g->control_pool = ds4_gpu_tensor_alloc(g->control_bytes);
    if (!g->state_pool || !g->control_pool) { return false; }
    uint64_t state_cursor = 0u, control_cursor = 0u;
    const uint32_t layers = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    for (uint32_t il = 0; il < layers; il++) {
        if (!ds4_glm53_layer_is_kda(il)) {
            if (!solar_pool_take(&g->layer_tail_k[il], g->state_pool, &state_cursor, tail_bytes) ||
                !solar_pool_take(&g->layer_tail_gate[il], g->state_pool, &state_cursor, tail_bytes)) { return false; }
            const uint64_t kv_bytes = g->expanded_diag
                ? (uint64_t)g->ctx_cap * DS4_N_HEAD * DS4_N_KEY_MLA * 2u * sizeof(uint16_t)
                : (uint64_t)g->ctx_cap * DS4_N_KV_LORA * sizeof(uint16_t);
            g->layer_kv[il] = ds4_gpu_tensor_alloc(kv_bytes);
            if (!g->layer_kv[il]) { return false; }
            g->kv_bytes += kv_bytes;
            if (!g->expanded_diag) {
                const uint64_t pool_bytes = (uint64_t)g->pool_cap * DS4_N_INDEXER_HEAD_DIM * sizeof(uint16_t);
                g->layer_pool[il] = ds4_gpu_tensor_alloc(pool_bytes);
                if (!g->layer_pool[il]) { return false; }
                g->kv_bytes += pool_bytes;
            }
            continue;
        }
        const ds4_layer_weights *l = &weights->layer[il];
        const bool ok =
            solar_pool_take(&g->recurrent[il], g->state_pool, &state_cursor, recurrent_bytes) &&
            solar_pool_take(&g->q_conv_state[il], g->state_pool, &state_cursor, conv_bytes) &&
            solar_pool_take(&g->k_conv_state[il], g->state_pool, &state_cursor, conv_bytes) &&
            solar_pool_take(&g->v_conv_state[il], g->state_pool, &state_cursor, conv_bytes) &&
            solar_pool_take(&g->q_conv_weight[il], g->control_pool, &control_cursor, conv_bytes) &&
            solar_pool_take(&g->k_conv_weight[il], g->control_pool, &control_cursor, conv_bytes) &&
            solar_pool_take(&g->v_conv_weight[il], g->control_pool, &control_cursor, conv_bytes) &&
            solar_pool_take(&g->decay_scale[il], g->control_pool, &control_cursor, decay_bytes) &&
            solar_pool_take(&g->dt_bias[il], g->control_pool, &control_cursor, dt_bytes) &&
            solar_pool_take(&g->o_norm[il], g->control_pool, &control_cursor, norm_bytes) &&
            solar_copy_control(g->q_conv_weight[il], model, l->kda_q_conv, conv_bytes) &&
            solar_copy_control(g->k_conv_weight[il], model, l->kda_k_conv, conv_bytes) &&
            solar_copy_control(g->v_conv_weight[il], model, l->kda_v_conv, conv_bytes) &&
            solar_copy_control(g->decay_scale[il], model, l->kda_a_log, decay_bytes) &&
            solar_copy_control(g->dt_bias[il], model, l->kda_dt_bias, dt_bytes) &&
            solar_copy_control(g->o_norm[il], model, l->kda_o_norm, norm_bytes);
        if (!ok) { return false; }
    }
    return solar_pool_take(&g->last_hidden, g->state_pool, &state_cursor, hidden_bytes) &&
        solar_pool_take(&g->mtp_hidden, g->state_pool, &state_cursor, hidden_bytes) &&
        state_cursor == g->state_bytes && control_cursor == g->control_bytes;
}

static bool glm53_graph_alloc_plan(ds4_glm53_gpu_graph *g,
        const ds4_model *model, const ds4_weights *weights, uint32_t ctx_cap,
        uint32_t rows, uint32_t window, ds4_glm53_stream *stream) {
    if (!g || !model || !weights || (stream && stream->count < DS4_N_EXPERT_USED) ||
        DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM53 ||
        (window && (!stream || stream->count < 2u * DS4_N_EXPERT)) ||
        !glm53_graph_bytes_plan(ctx_cap, rows, window)) { return false; }
    memset(g, 0, sizeof(*g));
    g->ctx_cap = ctx_cap;
    g->row_cap = rows;
    g->window_cap = window;
    g->pool_cap = (ctx_cap + DS4_GLM53_POOL_SIZE - 1u) / DS4_GLM53_POOL_SIZE;
    g->expanded_diag = glm53_graph_diag();
    g->stream = stream;
#define GLM53_ALLOC(field, cols, type) do { \
        const uint64_t bytes_ = (uint64_t)g->row_cap * (cols) * sizeof(type); \
        g->field = ds4_gpu_tensor_alloc(bytes_); \
        if (!g->field) { glm53_graph_free(g); return false; } \
        g->workspace_bytes += bytes_; \
    } while (0);
    GLM53_ROW_BUFFERS(GLM53_ALLOC)
#undef GLM53_ALLOC
    const uint64_t logit_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    g->logits = ds4_gpu_tensor_alloc(logit_bytes);
    g->workspace_bytes += logit_bytes;
    // The CUDA temporary arena retains the largest dense GEMM slab.
    g->workspace_bytes += glm53_dense_reserve(g->row_cap, ctx_cap);
    g->workspace_bytes += glm53_window_bytes(window);
    const uint64_t scratch_bytes = ds4_gpu_solar_kda_prefill_scratch_bytes(
        g->row_cap, DS4_N_HEAD, DS4_N_KDA_HEAD_DIM);
    if (scratch_bytes) {
        g->kda_scratch = ds4_gpu_tensor_alloc(scratch_bytes);
        g->workspace_bytes += scratch_bytes;
    }
    if (!g->logits || (scratch_bytes && !g->kda_scratch) || !glm53_graph_state(g, model, weights)) {
        glm53_graph_free(g);
        return false;
    }
    g->ready = true;
    if (!ds4_gpu_tensor_fill_f32(g->hc_mean, 1.0f / DS4_N_HC,
                               (uint64_t)g->row_cap * DS4_N_HC) || !glm53_graph_reset(g)) {
        glm53_graph_free(g);
        return false;
    }
    ds4_log(stderr, DS4_LOG_TIMING,
        "ds4: GLM-5.3 %s graph: ctx %u, prefill %u, KV %.2f GiB, "
        "state %.2f MiB, workspace %.2f MiB\n",
        g->expanded_diag ? "expanded diagnostic" : "compact", ctx_cap, g->row_cap,
        (double)g->kv_bytes / 1073741824.0, (double)g->state_bytes / 1048576.0,
        (double)g->workspace_bytes / 1048576.0);
    return true;
}

static bool glm53_graph_alloc(ds4_glm53_gpu_graph *g,
        const ds4_model *model, const ds4_weights *weights, uint32_t ctx_cap,
        ds4_glm53_stream *stream) {
    const uint32_t slots = stream ? stream->count : 0u;
    return glm53_graph_alloc_plan(g, model, weights, ctx_cap,
        glm53_graph_row_cap(ctx_cap, slots), glm53_window_cap(ctx_cap, slots), stream);
}

static bool glm53_graph_hc_pre(ds4_glm53_gpu_graph *g,
        const ds4_model *model, const ds4_tensor *fn, const ds4_tensor *scale,
        const ds4_tensor *base, const ds4_tensor *norm_weight,
        const ds4_gpu_tensor *residual_hc, uint32_t rows) {
    if (!g || !model || !fn || !scale || !base || !norm_weight || !residual_hc) { return false; }
    const uint32_t hc_dim = DS4_N_HC * DS4_N_EMBD;
    const uint32_t mix_dim = DS4_N_HC * (DS4_N_HC + 2u);
    return ds4_gpu_rms_norm_plain_rows_tensor(g->hc_flat, residual_hc, hc_dim, rows, DS4_RMS_EPS) &&
        plain_graph_matmul_tensor(g->hc_mix, model, fn, hc_dim, mix_dim, g->hc_flat, rows) &&
        ds4_gpu_hc_split_weighted_sum_tensor(g->cur, g->hc_split, g->hc_mix, residual_hc,
            model->map, model->size, scale->abs_offset, base->abs_offset,
            DS4_N_EMBD, DS4_N_HC, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS) &&
        ds4_gpu_rms_norm_weight_rows_tensor(g->norm, g->cur, model->map, model->size,
            norm_weight->abs_offset, DS4_N_EMBD, rows, DS4_RMS_EPS);
}

static bool glm53_graph_kda(ds4_glm53_gpu_graph *g, const ds4_model *model,
        const ds4_layer_weights *l, uint32_t il, uint32_t rows) {
    const uint32_t projection = DS4_N_HEAD * DS4_N_KDA_HEAD_DIM;
    if (!plain_graph_matmul_tensor(g->q, model, l->kda_q, DS4_N_EMBD, projection, g->norm, rows) ||
        !plain_graph_matmul_tensor(g->k, model, l->kda_k, DS4_N_EMBD, projection, g->norm, rows) ||
        !plain_graph_matmul_tensor(g->v, model, l->kda_v, DS4_N_EMBD, projection, g->norm, rows) ||
        !plain_graph_matmul_tensor(g->q_rank, model, l->kda_f_a, DS4_N_EMBD, DS4_N_KDA_HEAD_DIM, g->norm, rows) ||
        !plain_graph_matmul_tensor(g->heads, model, l->kda_f_b, DS4_N_KDA_HEAD_DIM, projection, g->q_rank, rows) ||
        !plain_graph_matmul_tensor(g->kv_rank, model, l->kda_beta, DS4_N_EMBD, DS4_N_HEAD, g->norm, rows) ||
        !plain_graph_matmul_tensor(g->q_rank, model, l->kda_g_a, DS4_N_EMBD, DS4_N_KDA_HEAD_DIM, g->norm, rows) ||
        !plain_graph_matmul_tensor(g->ffn_gate, model, l->kda_g_b, DS4_N_KDA_HEAD_DIM, projection, g->q_rank, rows)) {
        return false;
    }
    const bool recurrence = rows == 1u
        ? ds4_gpu_glm53_kda_decode_tensor(g->ffn_up, g->recurrent[il],
            g->q_conv_state[il], g->k_conv_state[il], g->v_conv_state[il],
            g->q, g->k, g->v, g->heads, g->kv_rank, g->q_conv_weight[il],
            g->k_conv_weight[il], g->v_conv_weight[il], g->decay_scale[il],
            g->dt_bias[il], DS4_N_HEAD, DS4_N_KDA_HEAD_DIM, DS4_N_SSM_CONV, DS4_KDA_GATE_CLAMP_MIN)
        : ds4_gpu_glm53_kda_prefill_tensor(g->ffn_up, g->kda_scratch,
            g->recurrent[il], g->q_conv_state[il], g->k_conv_state[il], g->v_conv_state[il],
            g->q, g->k, g->v, g->heads, g->kv_rank, g->q_conv_weight[il],
            g->k_conv_weight[il], g->v_conv_weight[il], g->decay_scale[il],
            g->dt_bias[il], rows, DS4_N_HEAD, DS4_N_KDA_HEAD_DIM, DS4_N_SSM_CONV, DS4_KDA_GATE_CLAMP_MIN);
    return recurrence && ds4_gpu_solar_head_rms_sigmoid_gate_tensor(g->heads,
        g->ffn_up, g->ffn_gate, g->o_norm[il], rows, DS4_N_HEAD, DS4_N_KDA_HEAD_DIM, DS4_RMS_EPS) &&
        plain_graph_matmul_tensor(g->attn_out, model, l->kda_output,
            projection, DS4_N_EMBD, g->heads, rows);
}

static bool glm53_graph_dsa_diag(ds4_glm53_gpu_graph *g, const ds4_model *model,
        const ds4_layer_weights *l, uint32_t il, uint32_t pos0, uint32_t rows) {
    const uint32_t qdim = DS4_N_HEAD * DS4_N_KEY_MLA;
    if (!ds4_gpu_glm53_k_b_project_tensor(g->k, g->kv_rank_norm, model->map, model->size,
            l->attn_k_b->abs_offset, rows, DS4_N_KV_LORA, DS4_N_KEY_MLA, DS4_N_HEAD) ||
        !plain_graph_matmul_tensor(g->v, model, l->attn_v_b, DS4_N_KV_LORA, qdim, g->kv_rank_norm, rows) ||
        !ds4_gpu_exaone_kv_store_tensor(g->layer_kv[il], g->k, g->v, qdim, rows, pos0, g->ctx_cap)) {
        return false;
    }
    for (uint32_t t = 0; t < rows; t++) {
        const uint64_t bytes = (uint64_t)qdim * sizeof(float);
        ds4_gpu_tensor *q = ds4_gpu_tensor_view(g->q, (uint64_t)t * bytes, bytes);
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(g->heads, (uint64_t)t * bytes, bytes);
        const bool ok = q && out && ds4_gpu_exaone_attention_decode_tensor(out, q,
            g->layer_kv[il], DS4_N_HEAD, DS4_N_HEAD, DS4_N_KEY_MLA, g->ctx_cap, pos0 + t, 0u);
        ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(q);
        if (!ok) { return false; }
    }
    return plain_graph_matmul_tensor(g->attn_out, model, l->attn_output, qdim, DS4_N_EMBD, g->heads, rows);
}

static bool glm53_graph_dsa(ds4_glm53_gpu_graph *g, const ds4_model *model,
        const ds4_layer_weights *l, uint32_t il, uint32_t pos0, uint32_t rows) {
    const uint32_t qdim = DS4_N_HEAD * DS4_N_KEY_MLA;
    if (!plain_graph_matmul_tensor(g->q_rank, model, l->attn_q_a,
            DS4_N_EMBD, DS4_N_LORA_Q, g->norm, rows) ||
        !ds4_gpu_rms_norm_weight_rows_tensor(g->q_rank_norm, g->q_rank, model->map, model->size,
            l->attn_q_a_norm->abs_offset, DS4_N_LORA_Q, rows, DS4_RMS_EPS) ||
        !plain_graph_matmul_tensor(g->q, model, l->attn_q_b, DS4_N_LORA_Q, qdim, g->q_rank_norm, rows) ||
        !plain_graph_matmul_tensor(g->kv_rank, model, l->attn_kv_a_mqa,
            DS4_N_EMBD, DS4_N_KV_LORA, g->norm, rows) ||
        !ds4_gpu_rms_norm_weight_rows_tensor(g->kv_rank_norm, g->kv_rank, model->map, model->size,
            l->attn_kv_a_norm->abs_offset, DS4_N_KV_LORA, rows, DS4_RMS_EPS)) { return false; }
    if (g->expanded_diag) { return glm53_graph_dsa_diag(g, model, l, il, pos0, rows); }

    /* Pool every key, including the dense prefix, so sparse continuation has
     * the same completed pools and partial tail as an uninterrupted prompt. */
    if (!plain_graph_matmul_tensor(g->idx_k, model, l->indexer_attn_k,
            DS4_N_EMBD, DS4_N_INDEXER_HEAD_DIM, g->norm, rows) ||
        !plain_graph_matmul_tensor(g->idx_gate, model, l->indexer_compressor_gate,
            DS4_N_EMBD, DS4_N_INDEXER_HEAD_DIM, g->norm, rows) ||
        !ds4_gpu_glm53_pool_key(g->layer_pool[il], g->layer_tail_k[il], g->layer_tail_gate[il],
            g->idx_k, g->idx_gate, model->map, model->size, l->indexer_k_norm->abs_offset,
            l->indexer_k_norm_b->abs_offset, l->indexer_compressor_ape->abs_offset,
            pos0, rows, g->ctx_cap, GLM53_POOL_EPS) ||
        !ds4_gpu_glm53_store_low(g->layer_kv[il], g->kv_rank_norm, rows, pos0, g->ctx_cap, DS4_N_KV_LORA) ||
        !ds4_gpu_glm53_absorb_q(g->low_q, g->q, model->map, model->size,
            l->attn_k_b->abs_offset, rows, DS4_N_HEAD, DS4_N_KV_LORA, DS4_N_KEY_MLA)) {
        return false;
    }
    ds4_gpu_tensor *selected = NULL;
    uint32_t stride = 0u;
    if (pos0 + rows > DS4_GLM53_INDEX_TOPK) {
        const uint32_t pools = (pos0 + rows) / DS4_GLM53_POOL_SIZE;
        const uint32_t topk = DS4_GLM53_INDEX_TOPK / DS4_GLM53_POOL_SIZE;
        const float scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM));
        if (!plain_graph_matmul_tensor(g->idx_q, model, l->indexer_attn_q_b, DS4_N_LORA_Q,
                (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM, g->q_rank_norm, rows) ||
            !plain_graph_matmul_tensor(g->idx_weights, model, l->indexer_proj,
                DS4_N_EMBD, DS4_N_INDEXER_HEAD, g->norm, rows) ||
            !ds4_gpu_glm53_pool_score(g->idx_scores, g->idx_q, g->idx_weights,
                g->layer_pool[il], pools, rows, pos0, DS4_N_INDEXER_HEAD, scale) ||
            !ds4_gpu_indexer_topk_tensor(g->idx_pools, g->idx_scores, pools, rows, topk, 0u, UINT32_MAX) ||
            !ds4_gpu_glm53_pool_expand(g->idx_selected, g->idx_pools, rows, pos0,
                topk, DS4_GLM53_INDEX_TOPK, DS4_GLM53_MAX_SELECTED)) { return false; }
        selected = g->idx_selected;
        stride = DS4_GLM53_MAX_SELECTED;
    }
    return ds4_gpu_glm53_attn_low(g->low_out, g->low_q, g->layer_kv[il], selected,
        stride, rows, pos0, g->ctx_cap, DS4_N_HEAD, DS4_N_KV_LORA, DS4_N_KEY_MLA) &&
        ds4_gpu_glm53_proj_v(g->heads, g->low_out, model->map, model->size,
            l->attn_v_b->abs_offset, rows, DS4_N_HEAD, DS4_N_KV_LORA, DS4_N_VALUE_MLA) &&
        plain_graph_matmul_tensor(g->attn_out, model, l->attn_output,
            qdim, DS4_N_EMBD, g->heads, rows);
}

static bool glm53_graph_routed(ds4_glm53_gpu_graph *g, const ds4_model *model,
        const ds4_layer_weights *l, uint32_t il, uint32_t rows) {
    uint64_t gate_in, gate_out, gate_row, down_in, down_out, down_row;
    if (!tensor_expert_bytes(model, l->ffn_gate_exps, 0u, &gate_in, &gate_out, &gate_row) ||
        !tensor_expert_bytes(model, l->ffn_down_exps, 0u, &down_in, &down_out, &down_row) ||
        gate_in != DS4_N_EMBD || gate_out != DS4_N_FF_EXP ||
        down_in != DS4_N_FF_EXP || down_out != DS4_N_EMBD) { return false; }
    if (!g->stream) {
        bool mid_is_f16 = false;
        return ds4_gpu_routed_moe_batch_tensor(g->ffn_out, g->ffn_gate, g->ffn_up,
            g->ffn_mid, g->routed_down, model->map, model->size,
            l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset, l->ffn_down_exps->abs_offset,
            l->ffn_gate_exps->type, l->ffn_down_exps->type, gate_out * gate_row, gate_row,
            down_out * down_row, down_row, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD,
            g->selected, g->route_weights, DS4_N_EXPERT, DS4_N_EXPERT_USED,
            DS4_SWIGLU_CLAMP_EXP, g->norm, il, rows, &mid_is_f16) != 0;
    }
    ds4_glm53_stream *s = g->stream;
    if (!glm53_stream_routes(s, g->selected, rows)) { return false; }
    for (uint32_t done = 0; done < rows;) {
        int32_t *ids = s->ids + (uint64_t)done * DS4_N_EXPERT_USED;
        const uint32_t chunk = glm53_cache_rows(ids, rows - done,
            DS4_N_EXPERT_USED, DS4_N_EXPERT, s->count);
        if (!chunk) { return false; }
        /* Cache admission changes supply only; projection widths are kept
         * whenever the actual route union fits, including duplicate routes. */
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(g->ffn_out,
            (uint64_t)done * DS4_N_EMBD * sizeof(float), (uint64_t)chunk * DS4_N_EMBD * sizeof(float));
        ds4_gpu_tensor *x = ds4_gpu_tensor_view(g->norm,
            (uint64_t)done * DS4_N_EMBD * sizeof(float), (uint64_t)chunk * DS4_N_EMBD * sizeof(float));
        const uint64_t routed_cols = (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP;
        ds4_gpu_tensor *gate = ds4_gpu_tensor_view(g->ffn_gate,
            (uint64_t)done * routed_cols * sizeof(float), (uint64_t)chunk * routed_cols * sizeof(float));
        ds4_gpu_tensor *up = ds4_gpu_tensor_view(g->ffn_up,
            (uint64_t)done * routed_cols * sizeof(float), (uint64_t)chunk * routed_cols * sizeof(float));
        ds4_gpu_tensor *mid = ds4_gpu_tensor_view(g->ffn_mid,
            (uint64_t)done * routed_cols * sizeof(float), (uint64_t)chunk * routed_cols * sizeof(float));
        const uint64_t down_cols = (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD;
        ds4_gpu_tensor *down = ds4_gpu_tensor_view(g->routed_down,
            (uint64_t)done * down_cols * sizeof(float), (uint64_t)chunk * down_cols * sizeof(float));
        ds4_gpu_tensor *picked = ds4_gpu_tensor_view(g->selected,
            (uint64_t)done * DS4_N_EXPERT_USED * sizeof(int32_t), (uint64_t)chunk * DS4_N_EXPERT_USED * sizeof(int32_t));
        ds4_gpu_tensor *weights = ds4_gpu_tensor_view(g->route_weights,
            (uint64_t)done * DS4_N_EXPERT_USED * sizeof(float), (uint64_t)chunk * DS4_N_EXPERT_USED * sizeof(float));
        const bool ok = out && x && gate && up && mid && down && picked && weights &&
            glm53_stream_stage(s, model, l, il, ids, chunk) &&
            ds4_gpu_glm53_moe_owned(out, gate, up, mid, down, s->gate, s->up, s->down,
                l->ffn_gate_exps->type, l->ffn_down_exps->type, s->gate_stride, s->down_stride,
                DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD, s->selected, weights, s->count,
                DS4_N_EXPERT_USED, chunk, DS4_SWIGLU_CLAMP_EXP, x);
        ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(picked); ds4_gpu_tensor_free(down);
        ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(gate);
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out);
        if (!ok) { return false; }
        done += chunk;
    }
    return true;
}

static bool glm53_graph_ffn(ds4_glm53_gpu_graph *g, const ds4_model *model,
        const ds4_layer_weights *l, uint32_t il, uint32_t rows) {
    if (il < DS4_N_LEADING_DENSE) {
        const uint32_t hidden = (uint32_t)l->ffn_gate->dim[1];
        return plain_graph_matmul_tensor(g->ffn_gate, model, l->ffn_gate, DS4_N_EMBD, hidden, g->norm, rows) &&
            plain_graph_matmul_tensor(g->ffn_up, model, l->ffn_up, DS4_N_EMBD, hidden, g->norm, rows) &&
            ds4_gpu_swiglu_tensor(g->ffn_mid, g->ffn_gate, g->ffn_up, rows * hidden, DS4_SWIGLU_CLAMP_EXP, 1.0f) &&
            plain_graph_matmul_tensor(g->ffn_out, model, l->ffn_down, hidden, DS4_N_EMBD, g->ffn_mid, rows);
    }
    if (!plain_graph_matmul_tensor(g->kv_rank, model, l->ffn_gate_inp,
            DS4_N_EMBD, DS4_N_EXPERT, g->norm, rows) ||
        !ds4_gpu_sigmoid_topk_router_tensor(g->selected, g->route_weights, g->kv_rank,
            model->map, model->size, l->ffn_exp_probs_b->abs_offset, 1, DS4_N_EXPERT,
            DS4_N_EXPERT_USED, rows, DS4_EXPERT_WEIGHT_SCALE) ||
        !glm53_graph_routed(g, model, l, il, rows)) { return false; }

    /* Decode shares the Q8_1 input and writes clamped mid without gate/up scratch. */
    int shared = 0;
    if (rows == 1u && l->ffn_gate_shexp->type == GLM53_QUANT_Q8 &&
        l->ffn_up_shexp->type == GLM53_QUANT_Q8) {
        shared = ds4_gpu_glm53_shared_q8(g->ffn_mid, model->map, model->size,
            l->ffn_gate_shexp->abs_offset, l->ffn_up_shexp->abs_offset,
            DS4_N_EMBD, DS4_N_FF_EXP, g->norm, DS4_SWIGLU_CLAMP_EXP);
    }
    if (shared < 0) { return false; }

    const bool ready = shared > 0 || (plain_graph_matmul_tensor(g->ffn_gate, model, l->ffn_gate_shexp,
            DS4_N_EMBD, DS4_N_FF_EXP, g->norm, rows) &&
        plain_graph_matmul_tensor(g->ffn_up, model, l->ffn_up_shexp,
            DS4_N_EMBD, DS4_N_FF_EXP, g->norm, rows) &&
        ds4_gpu_swiglu_tensor(g->ffn_mid, g->ffn_gate, g->ffn_up,
            rows * DS4_N_FF_EXP, DS4_SWIGLU_CLAMP_EXP, 1.0f));
    return ready && plain_graph_matmul_tensor(g->attn_out, model, l->ffn_down_shexp,
            DS4_N_FF_EXP, DS4_N_EMBD, g->ffn_mid, rows) &&
        ds4_gpu_add_tensor(g->ffn_out, g->ffn_out, g->attn_out, (uint64_t)rows * DS4_N_EMBD);
}

static bool glm53_graph_output(ds4_glm53_gpu_graph *g, const ds4_model *model,
        const ds4_weights *weights, uint32_t rows, float *host_logits) {
    const uint64_t bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    ds4_gpu_tensor *hc = ds4_gpu_tensor_view(g->hc_cur,
        (uint64_t)(rows - 1u) * DS4_N_HC * bytes, DS4_N_HC * bytes);
    ds4_gpu_tensor *cur = ds4_gpu_tensor_view(g->cur, (uint64_t)(rows - 1u) * bytes, bytes);
    ds4_gpu_tensor *norm = ds4_gpu_tensor_view(g->norm, (uint64_t)(rows - 1u) * bytes, bytes);
    bool ok = hc && cur && norm &&
        ds4_gpu_hc_weighted_sum_tensor(cur, hc, g->hc_mean, DS4_N_EMBD, DS4_N_HC) &&
        ds4_gpu_tensor_copy(g->last_hidden, 0u, cur, 0u, bytes);
    if (ok && host_logits) {
        ok = ds4_gpu_rms_norm_weight_tensor(norm, cur, model->map, model->size,
            weights->output_norm->abs_offset, DS4_N_EMBD, DS4_RMS_EPS) &&
        plain_graph_matmul_tensor(g->logits, model, weights->output, DS4_N_EMBD, DS4_N_VOCAB, norm, 1u) &&
        ds4_gpu_tensor_read(g->logits, 0u, host_logits, (uint64_t)DS4_N_VOCAB * sizeof(float));
    }
    ds4_gpu_tensor_free(norm); ds4_gpu_tensor_free(cur); ds4_gpu_tensor_free(hc);
    return ok;
}

static bool glm53_graph_embed(ds4_glm53_gpu_graph *g, const ds4_model *model,
        const ds4_weights *weights, const int *tokens, uint32_t rows,
        const float *embeddings) {
    const bool embedded = embeddings
        ? ds4_gpu_tensor_write(g->cur, 0u, embeddings, (uint64_t)rows * DS4_N_EMBD * sizeof(float)) != 0
        : ds4_gpu_tensor_write(g->token, 0u, tokens, (uint64_t)rows * sizeof(int32_t)) &&
          ds4_gpu_embed_tokens_quant_tensor(g->cur, g->token, model->map, model->size,
            weights->token_embd->abs_offset, weights->token_embd->type,
            (uint32_t)weights->token_embd->dim[1], rows, DS4_N_EMBD);
    if (!embedded || !ds4_gpu_repeat_hc_rows_tensor(g->hc_cur, g->cur, DS4_N_EMBD, DS4_N_HC, rows)) { return false; }
    return true;
}

static bool glm53_graph_layer(ds4_glm53_gpu_graph *g, const ds4_model *model,
        const ds4_weights *weights, uint32_t il, uint32_t rows, uint32_t pos0) {
    const ds4_layer_weights *l = &weights->layer[il];
    if (!glm53_graph_hc_pre(g, model, l->hc_attn_fn, l->hc_attn_scale,
            l->hc_attn_base, l->attn_norm, g->hc_cur, rows)) {
        fprintf(stderr, "ds4: GLM-5.3 layer %u attention mHC failed\n", il);
        return false;
    }
    const bool attention = ds4_glm53_layer_is_kda(il)
        ? glm53_graph_kda(g, model, l, il, rows)
        : glm53_graph_dsa(g, model, l, il, pos0, rows);
    if (!attention || !ds4_gpu_hc_expand_split_tensor(g->hc_after_attn,
            g->attn_out, g->hc_cur, g->hc_split, DS4_N_EMBD, DS4_N_HC) ||
        !glm53_graph_hc_pre(g, model, l->hc_ffn_fn, l->hc_ffn_scale,
            l->hc_ffn_base, l->ffn_norm, g->hc_after_attn, rows) ||
        !glm53_graph_ffn(g, model, l, il, rows) ||
        !ds4_gpu_hc_expand_split_tensor(g->hc_next, g->ffn_out,
            g->hc_after_attn, g->hc_split, DS4_N_EMBD, DS4_N_HC)) {
        fprintf(stderr, "ds4: GLM-5.3 layer %u batch execution failed\n", il);
        return false;
    }
    return true;
}

static bool glm53_graph_batch(ds4_glm53_gpu_graph *g, const ds4_model *model,
        const ds4_weights *weights, const int *tokens, uint32_t rows,
        uint32_t pos0, const float *embeddings, float *host_logits) {
    if (!glm53_graph_embed(g, model, weights, tokens, rows, embeddings)) { return false; }
    const uint32_t layers = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    for (uint32_t il = 0; il < layers; il++) {
        if (!glm53_graph_layer(g, model, weights, il, rows, pos0)) { return false; }
        ds4_gpu_tensor *swap = g->hc_cur;
        g->hc_cur = g->hc_next;
        g->hc_next = swap;
    }
    return glm53_graph_output(g, model, weights, rows, host_logits);
}

/* Keep each layer's KDA/pool/causal-KV updates in token order. Scratch rows
 * stay <=2048; only HC rows survive across chunks within a checkpoint window. */
static bool glm53_graph_window(ds4_glm53_gpu_graph *g, const ds4_model *model,
        const ds4_weights *weights, const int *tokens, uint32_t rows,
        uint32_t pos0, const float *embeddings, float *host_logits) {
    const uint64_t row_bytes = (uint64_t)DS4_N_HC * DS4_N_EMBD * sizeof(float);
    ds4_gpu_tensor *cur = ds4_gpu_tensor_alloc(rows * row_bytes);
    ds4_gpu_tensor *next = ds4_gpu_tensor_alloc(rows * row_bytes);
    bool ok = cur && next;
    const char *value = getenv("DS4_GLM53_HOT_LAST");
    // Only the last window feeds Decode. Earlier hot copies are overwritten
    // before use; 0 keeps them for a matched diagnostic comparison.
    const enum glm53_hot_copy keep = host_logits || (value && strcmp(value, "0") == 0)
        ? GLM53_HOT_KEEP : GLM53_HOT_SKIP;
    for (uint32_t done = 0; ok && done < rows;) {
        const uint32_t chunk = rows - done < g->row_cap ? rows - done : g->row_cap;
        ds4_glm53_gpu_graph active;
        if (!glm53_graph_bind(&active, g, chunk)) { ok = false; break; }
        ok = glm53_graph_embed(&active, model, weights, tokens + done, chunk,
            embeddings ? embeddings + (uint64_t)done * DS4_N_EMBD : NULL) &&
            ds4_gpu_tensor_copy(cur, done * row_bytes, active.hc_cur, 0u, chunk * row_bytes);
        glm53_graph_unbind(&active);
        done += chunk;
    }
    const uint32_t layers = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    for (uint32_t il = 0; ok && il < layers; il++) {
        ok = glm53_stream_begin(g->stream, model, weights, il);
        for (uint32_t done = 0; ok && done < rows;) {
            const uint32_t chunk = rows - done < g->row_cap ? rows - done : g->row_cap;
            ds4_glm53_gpu_graph active;
            if (!glm53_graph_bind(&active, g, chunk)) { ok = false; break; }
            ok = ds4_gpu_tensor_copy(active.hc_cur, 0u, cur, done * row_bytes, chunk * row_bytes) &&
                glm53_graph_layer(&active, model, weights, il, chunk, pos0 + done) &&
                ds4_gpu_tensor_copy(next, done * row_bytes, active.hc_next, 0u, chunk * row_bytes);
            glm53_graph_unbind(&active);
            done += chunk;
        }
        if (ok) { ok = glm53_stream_end(g->stream, il, keep); }
        ds4_gpu_tensor *swap = cur; cur = next; next = swap;
    }
    const uint32_t last = (rows - 1u) % g->row_cap + 1u;
    if (ok) {
        ds4_glm53_gpu_graph active;
        ok = glm53_graph_bind(&active, g, last);
        if (ok) {
            ok = ds4_gpu_tensor_copy(active.hc_cur, 0u, cur, (rows - last) * row_bytes, last * row_bytes) &&
                glm53_graph_output(&active, model, weights, last, host_logits);
            glm53_graph_unbind(&active);
        }
    }
    ds4_gpu_tensor_free(cur); ds4_gpu_tensor_free(next);
    if (!ok) {
        glm53_stream_finish(g->stream);
        g->cache_len = UINT32_MAX;
        return false;
    }
    // No partially evaluated window is a reusable prefix. Restore/reset must
    // replace a failed graph; success commits every layer at the same frontier.
    g->cache_len = pos0 + rows;
    g->last_rows = last;
    return true;
}

static bool glm53_graph_prefill(ds4_glm53_gpu_graph *g, const ds4_model *model,
        const ds4_weights *weights, const int *tokens, uint32_t rows,
        uint32_t pos0, const float *host_embeddings, float *host_logits) {
    if (!g || !g->ready || !model || !weights || !tokens || !rows ||
        pos0 != g->cache_len || pos0 > g->ctx_cap || rows > g->ctx_cap - pos0) { return false; }
    for (uint32_t t = 0; t < rows; t++) {
        if (tokens[t] < 0 || (uint32_t)tokens[t] >= DS4_N_VOCAB) { return false; }
    }
    const glm53_stream_stats before = glm53_stream_snap(g->stream);
    glm53_stream_hot_begin(g->stream, rows);
    bool staged = false;
    for (uint32_t done = 0; done < rows;) {
        const uint32_t window = g->window_cap
            ? g->window_cap - (pos0 + done) % g->window_cap : 0u;
        const uint32_t span = rows - done < window ? rows - done : window;
        if (span > g->row_cap) {
            if (!glm53_graph_window(g, model, weights, tokens + done, span, pos0 + done,
                    host_embeddings ? host_embeddings + (uint64_t)done * DS4_N_EMBD : NULL,
                    done + span == rows ? host_logits : NULL)) {
                glm53_stream_finish(g->stream);
                g->cache_len = UINT32_MAX;
                return false;
            }
            staged = true;
            done += span;
            continue;
        }
        const uint32_t left = window ? span : rows - done;
        const uint32_t chunk = left < g->row_cap ? left : g->row_cap;
        ds4_glm53_gpu_graph active;
        if (!glm53_graph_bind(&active, g, chunk)) {
            glm53_stream_finish(g->stream);
            return false;
        }
        const bool ok = glm53_graph_batch(&active, model, weights, tokens + done, chunk, pos0 + done,
            host_embeddings ? host_embeddings + (uint64_t)done * DS4_N_EMBD : NULL,
            done + chunk == rows ? host_logits : NULL);
        if (ok && ds4_gpu_tensor_ptr(active.hc_cur) != ds4_gpu_tensor_ptr(g->hc_cur)) {
            ds4_gpu_tensor *swap = g->hc_cur;
            g->hc_cur = g->hc_next;
            g->hc_next = swap;
        }
        glm53_graph_unbind(&active);
        if (!ok) {
            glm53_stream_finish(g->stream);
            g->cache_len = UINT32_MAX;
            return false;
        }
        done += chunk;
        g->cache_len = pos0 + done;
        g->last_rows = chunk;
    }
    if (staged) { glm53_stream_finish(g->stream); }
    glm53_stream_hot_end(g->stream);
    glm53_stream_delta(g->stream, before, pos0, rows);
    return true;
}

static bool glm53_graph_forward_token(ds4_glm53_gpu_graph *g, const ds4_model *model,
        const ds4_weights *weights, int token, uint32_t pos,
        const float *host_embedding, float *host_logits) {
    if (!g || pos != g->cache_len || pos >= g->ctx_cap) { return false; }
    g->mtp_rewind_n = 0u;
    if (g->mtp_ready && pos) {
        if (g->mtp_len && g->mtp_len != pos - 1u) { g->mtp_len = g->mtp_min = 0u; }
        int unused = -1;
        if (!glm53_mtp_step(g, model, weights, g->last_hidden, token, pos - 1u, &unused)) {
            return false;
        }
    }
    return glm53_graph_prefill(g, model, weights, &token, 1u, pos, host_embedding, host_logits);
}

#define GLM53_MTP_STEP
#include "ds4_glm53_mtp.inc"
#undef GLM53_MTP_STEP

#define GLM53_MTP_TRIAL
#include "ds4_glm53_mtp.inc"
#undef GLM53_MTP_TRIAL

#undef GLM53_ROW_BUFFERS
