/* Spark-X2.6-4B VQ graph (host side), included by ds4.c after the weight
 * bind and after the CUDA entries (ds4_spark2_5_vq.h).
 *
 * The forward is modeling_spark.py's, layer by layer, with its rounding
 * points reproduced where an existing entry makes that free:
 *
 *   Spark2_5DecoderLayer.forward :253-278   (pre-norm, residual F32)
 *   Spark2_5Attention.forward    :158-208   (fused qkv, g_proj, GQA, gate)
 *   Spark2_5MLP.forward          :126-128   (gelu(gate) * up)
 *   Spark2_5Model.forward        :399-412   (F32 residual stream, final norm
 *                                            cast to BF16 before the tied head)
 *
 * Named deviations, all measured by the logits gate rather than assumed:
 *  1. The reference rounds the projection outputs to BF16; this graph asks
 *     the VQ/GEMM entries for F32 and rounds explicitly where the next
 *     operation's rounding would otherwise differ (q, k, attention output,
 *     the gate product, each FFN stage). The VQ weights are approximate by
 *     construction, so bit identity is not a goal (AGENTS.md "Never chase
 *     bit identity").
 *  2. The head gate's sigmoid is applied as F32 by the shared dots3 kernel;
 *     the reference casts it to BF16 before the multiply
 *     (modeling_spark.py:197-204). The multiply stays F32 here.
 *  3. GELU is the exact erf form the reference selects through
 *     ACT2FN["gelu"]; its intermediate is not rounded to BF16 the way a
 *     BF16 tensor op would round it.
 *  4. The attention scale 1/sqrt(head_dim) = 2^-4 (modeling_spark.py:146,
 *     applied at :84) is folded into q once after the rope instead of into
 *     the scores: g4_attn takes no scale (Gemma's own attention has none).
 *     A power of two keeps the fold exact, so q_scaled * k is bit-identical
 *     to (q * k) * scale. The reference then rounds the scaled score to
 *     BF16; the shared kernel keeps its F32 score (deviation 1's class). */

#ifndef DS4_NO_GPU

/* ---------------- shape and memory ---------------- */

/* A sliding layer keeps window + chunk rows so a prefill chunk never
 * overwrites a slot one of its own rows still reads; a full layer keeps the
 * whole context, indexed by absolute position (ring 0). */
static uint32_t spark2_5_kv_rows(uint32_t il, uint32_t ctx, uint32_t cap) {
    if (spark2_5_layer_full(il)) { return ctx; }
    const uint32_t need = SPARK_WINDOW + cap;
    return ctx < need ? ctx : need;
}

static uint64_t spark2_5_kv_bytes(uint32_t ctx, uint32_t cap) {
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < SPARK_LAYERS; il++) {
        bytes += 2u * (uint64_t)spark2_5_kv_rows(il, ctx, cap) *
                 SPARK_KV_DIM * sizeof(uint16_t);
    }
    return bytes;
}

static uint32_t spark2_5_prefill_cap(uint32_t ctx) {
    uint32_t cap = ctx < 512u ? ctx : 512u;
    const char *env = getenv("DS4_SPARK2_5_PREFILL_CHUNK");
    if (env && env[0]) {
        const long asked = strtol(env, NULL, 10);
        if (asked > 0 && asked <= SPARK_PREFILL_MAX) { cap = (uint32_t)asked; }
    }
    return cap < ctx ? cap : ctx;
}

/* The VQ linear restores a whole matrix into scratch before its GEMM
 * (cuda/glm_vq_linear.cuh:51-96: [restored | x | y]), and the family's widest
 * matrix is the 10,240 x 2,560 FFN pair, whose restored BF16 image is 52.4 MB
 * at any row count. ds4_gpu_gvq_workspace_linear's fixed 32 MiB covers
 * Gemma-4's 2,112/2,816-wide matrices but not this one, so the graph sizes its
 * own scratch and asks glmvq_linear_bytes for the term. */
static uint64_t spark2_5_linear_workspace(uint32_t rows) {
    uint64_t need = ds4_gpu_gvq_workspace_linear(rows);
    const uint64_t widest = ds4_gpu_gvq_linear_bytes(
        SPARK_FF, SPARK_EMBD, rows, DS4_TENSOR_GLM_VQ, DS4_GVQ_BF16);
    return widest > need ? widest : need;
}

static uint64_t spark2_5_activation_bytes(uint32_t rows) {
    const uint64_t f32 = sizeof(float);
    return (uint64_t)rows * f32 * (3u * SPARK_EMBD + SPARK_QKV_DIM +
        2u * SPARK_Q_DIM + 2u * SPARK_KV_DIM + SPARK_HEADS +
        3u * SPARK_FF + SPARK_EMBD) +
        (uint64_t)rows * (sizeof(int32_t) + sizeof(uint32_t)) +
        (uint64_t)SPARK_VOCAB * f32 +
        (uint64_t)(SPARK_ROT_SLIDING + SPARK_ROT_FULL) / 2u * f32;
}

static ds4_context_memory spark2_5_session_memory(uint32_t ctx, uint32_t cap) {
    ds4_context_memory m = {0};
    if (!ctx || !cap || cap > ctx || ctx > SPARK_CONTEXT ||
        cap > SPARK_PREFILL_MAX) {
        return m;
    }
    m.prefill_cap = cap;
    m.raw_cap = ctx;
    m.raw_bytes = spark2_5_kv_bytes(ctx, cap);
    m.scratch_bytes = spark2_5_activation_bytes(cap) +
        spark2_5_linear_workspace(cap);
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}

/* The fit gate the other graph families use: the session's own priced bytes
 * against the device's free memory, with the same fail-open legs. */
static bool spark2_5_session_fit(const ds4_engine *e, uint32_t ctx, uint32_t cap,
                                 ds4_session_graph_fit_quote *q) {
    const uint64_t need = spark2_5_session_memory(ctx, cap).total_bytes;
    if (q) {
        memset(q, 0, sizeof(*q));
        q->need_bytes = need;
    }
    if (e->backend != DS4_BACKEND_CUDA || !need) { return false; }
    uint64_t available = 0, total = 0;
    if (ds4_session_graph_fit_off() || ds4_gpu_mem_info(&available, &total) != 0) {
        if (q) {
            q->fits = 1;
            q->fail_open = 1;
        }
        return true;
    }
    const uint64_t substrate = ds4_gpu_substrate_outstanding();
    available = available > substrate ? available - substrate : 0;
    const uint64_t margin = ds4_session_graph_headroom_bytes();
    const uint64_t ask = need > UINT64_MAX - margin ? UINT64_MAX : need + margin;
    const bool fits = available >= ask;
    if (q) {
        q->fits = fits;
        q->avail_bytes = available;
        q->headroom_bytes = margin;
        q->deficit_bytes = fits ? 0 : ask - available;
    }
    return fits;
}

/* ---------------- weight plan ---------------- */

static uint32_t spark2_5_index(const ds4_model *m, const ds4_tensor *t) {
    return t ? (uint32_t)(t - m->tensors) : DS4_GVQ_NO_LINK;
}

/* Materialized source pointers belong to the weight owner. Each matrix links
 * its own 8192-word book; its GVR8 plane links the 256-word residual book
 * (the link ivq checks at bind time), which is what makes the R13+8 decode
 * exact. */
static bool spark2_5_gpu_bind(ds4_weights *w, const ds4_model *m) {
    if (!w->spark || !m->n_tensors || m->n_tensors > UINT32_MAX) { return false; }
    const uint32_t count = (uint32_t)m->n_tensors;
    ds4_gvq_record *records = xcalloc(count, sizeof(*records));
    for (uint32_t i = 0; i < count; i++) {
        const ds4_tensor *t = &m->tensors[i];
        const uint64_t cols = t->ndim ? t->dim[0] : 1u;
        const uint64_t rows = cols ? t->elements / cols : 0u;
        if (!cols || cols > UINT32_MAX || !rows || rows > UINT32_MAX) {
            free(records);
            return false;
        }
        records[i] = (ds4_gvq_record){t->abs_offset, t->bytes, t->type,
            (uint32_t)rows, (uint32_t)cols, spark2_5_index(m, t->source_scale),
            DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK};
    }
    for (uint32_t il = 0; il < SPARK_LAYERS; il++) {
        const ds4_spark2_5_layer *l = &w->spark[il];
        const ds4_spark2_5_matrix *mx[5] = {
            &l->qkv, &l->out, &l->ffn_gate, &l->ffn_up, &l->ffn_down
        };
        for (uint32_t k = 0; k < 5; k++) {
            if (!mx[k]->w || !mx[k]->book || !mx[k]->res ||
                !mx[k]->res_book) {
                free(records);
                return false;
            }
            const uint32_t w_i = spark2_5_index(m, mx[k]->w);
            const uint32_t res_i = spark2_5_index(m, mx[k]->res);
            records[w_i].book = spark2_5_index(m, mx[k]->book);
            records[w_i].res = res_i;
            records[res_i].book = spark2_5_index(m, mx[k]->res_book);
        }
    }
    w->glmvq_gpu = ds4_gpu_gvq_bind(m->map, m->size, records, count);
    free(records);
    if (!w->glmvq_gpu) { return false; }
    /* One BF16 device image of the VQ matrices for the whole engine: a graph
     * per session must not duplicate it. DS4_SPARK2_5_RESTORE=0 keeps the
     * per-call restore for an A/B or a device too small to hold it. */
    const char *restore_env = getenv("DS4_SPARK2_5_RESTORE");
    if (!restore_env || strcmp(restore_env, "0") != 0) {
        uint64_t bytes = 0;
        w->spark_restored = ds4_gpu_gvq_restore_device(w->glmvq_gpu,
            &w->spark_restored_store, &bytes);
        if (w->spark_restored) {
            fprintf(stderr, "ds4: Spark-X2.6 restored %.2f GiB of VQ weights to device BF16\n",
                (double)bytes / 1073741824.0);
        } else {
            fprintf(stderr, "ds4: Spark-X2.6 device restore unavailable; the linear "
                "path restores each matrix per call\n");
        }
    }
    return true;
}

/* ---------------- graph ---------------- */

typedef struct ds4_spark2_5_graph {
    ds4_gvq_model *model;
    ds4_gvq_ctx *gvq;
    uint32_t ctx, cap, rows, position;
    ds4_gpu_tensor *cur, *norm, *branch;         /* [rows][EMBD] */
    ds4_gpu_tensor *qkv;                          /* [rows][QKV_DIM] */
    ds4_gpu_tensor *q, *attn;                     /* [rows][Q_DIM] */
    ds4_gpu_tensor *k, *v;                        /* [rows][KV_DIM] */
    ds4_gpu_tensor *gate;                         /* [rows][HEADS] */
    ds4_gpu_tensor *ffn_gate, *ffn_up, *ffn_mid;  /* [rows][FF] */
    ds4_gpu_tensor *ffn_out;                      /* [rows][EMBD] */
    ds4_gpu_tensor *logits;                       /* [VOCAB] */
    ds4_gpu_tensor *tokens;                       /* [rows] i32 */
    ds4_gpu_tensor *positions;                    /* [rows] u32 */
    ds4_gpu_tensor *inv_sliding, *inv_full;       /* rope tables */
    ds4_gpu_tensor *kcache[SPARK_LAYERS], *vcache[SPARK_LAYERS];
    uint64_t kv_bytes;
    bool ready;
    bool failed;
} ds4_spark2_5_graph;

static void spark2_5_graph_clear(ds4_spark2_5_graph *g) {
    if (!g) { return; }
    for (uint32_t il = 0; il < SPARK_LAYERS; il++) {
        ds4_gpu_tensor_free(g->kcache[il]);
        ds4_gpu_tensor_free(g->vcache[il]);
    }
    ds4_gpu_gvq_free(g->gvq);
    ds4_gpu_tensor_free(g->cur);
    ds4_gpu_tensor_free(g->norm);
    ds4_gpu_tensor_free(g->branch);
    ds4_gpu_tensor_free(g->qkv);
    ds4_gpu_tensor_free(g->q);
    ds4_gpu_tensor_free(g->attn);
    ds4_gpu_tensor_free(g->k);
    ds4_gpu_tensor_free(g->v);
    ds4_gpu_tensor_free(g->gate);
    ds4_gpu_tensor_free(g->ffn_gate);
    ds4_gpu_tensor_free(g->ffn_up);
    ds4_gpu_tensor_free(g->ffn_mid);
    ds4_gpu_tensor_free(g->ffn_out);
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->tokens);
    ds4_gpu_tensor_free(g->positions);
    ds4_gpu_tensor_free(g->inv_sliding);
    ds4_gpu_tensor_free(g->inv_full);
    memset(g, 0, sizeof(*g));
}

static void spark2_5_graph_release(ds4_spark2_5_graph *g) {
    if (!g) { return; }
    spark2_5_graph_clear(g);
    free(g);
}

static bool spark2_5_graph_reset(ds4_spark2_5_graph *g) {
    g->position = 0;
    g->failed = false;
    return true;
}

/* inv_freq over `rot` rotary dims at `theta`, the reference's
 * (modeling_spark.py:31-34): exponent 2i/rope_head_dim. `pairs` entries. */
static void spark2_5_rope_table(uint32_t pairs, uint32_t rot, double theta,
                                float *inv) {
    for (uint32_t i = 0; i < pairs; i++) {
        inv[i] = (float)(1.0 / pow(theta, (double)(2u * i) / (double)rot));
    }
}

static ds4_spark2_5_graph *spark2_5_graph_create(ds4_weights *w,
        uint32_t ctx, uint32_t cap) {
    if (!w->spark || !w->glmvq_gpu || !ctx || !cap || cap > ctx ||
        ctx > SPARK_CONTEXT || cap > SPARK_PREFILL_MAX) {
        return NULL;
    }
    ds4_spark2_5_graph *g = xcalloc(1, sizeof(*g));
    g->ctx = ctx;
    g->cap = cap;
    g->rows = cap;
    const uint32_t rows = g->rows;
    /* The restored BF16 image is the engine's (spark2_5_gpu_bind); a graph
     * only owns its KV state and activation set. */
    g->model = w->spark_restored ? w->spark_restored : w->glmvq_gpu;
    g->gvq = ds4_gpu_gvq_create_linear(g->model,
        spark2_5_linear_workspace(rows), rows);
    if (!g->gvq) {
        fprintf(stderr, "ds4: Spark-X2.6 GVQ context creation failed\n");
        spark2_5_graph_release(g);
        return NULL;
    }
    const uint64_t f32 = sizeof(float);
#define SPARK_ALLOC(field, bytes) do {                                        \
        g->field = ds4_gpu_tensor_alloc(bytes);                               \
        if (!g->field) {                                                      \
            fprintf(stderr, "ds4: Spark-X2.6 graph allocation failed: %s\n",  \
                #field);                                                      \
            spark2_5_graph_release(g);                                        \
            return NULL;                                                      \
        }                                                                     \
    } while (0)
    SPARK_ALLOC(cur, (uint64_t)rows * SPARK_EMBD * f32);
    SPARK_ALLOC(norm, (uint64_t)rows * SPARK_EMBD * f32);
    SPARK_ALLOC(branch, (uint64_t)rows * SPARK_EMBD * f32);
    SPARK_ALLOC(qkv, (uint64_t)rows * SPARK_QKV_DIM * f32);
    SPARK_ALLOC(q, (uint64_t)rows * SPARK_Q_DIM * f32);
    SPARK_ALLOC(attn, (uint64_t)rows * SPARK_Q_DIM * f32);
    SPARK_ALLOC(k, (uint64_t)rows * SPARK_KV_DIM * f32);
    SPARK_ALLOC(v, (uint64_t)rows * SPARK_KV_DIM * f32);
    SPARK_ALLOC(gate, (uint64_t)rows * SPARK_HEADS * f32);
    SPARK_ALLOC(ffn_gate, (uint64_t)rows * SPARK_FF * f32);
    SPARK_ALLOC(ffn_up, (uint64_t)rows * SPARK_FF * f32);
    SPARK_ALLOC(ffn_mid, (uint64_t)rows * SPARK_FF * f32);
    SPARK_ALLOC(ffn_out, (uint64_t)rows * SPARK_EMBD * f32);
    SPARK_ALLOC(logits, (uint64_t)SPARK_VOCAB * f32);
    SPARK_ALLOC(tokens, (uint64_t)rows * sizeof(int32_t));
    SPARK_ALLOC(positions, (uint64_t)rows * sizeof(uint32_t));
    SPARK_ALLOC(inv_sliding, (uint64_t)(SPARK_ROT_SLIDING / 2u) * f32);
    SPARK_ALLOC(inv_full, (uint64_t)(SPARK_ROT_FULL / 2u) * f32);
#undef SPARK_ALLOC
    float inv[SPARK_ROT_SLIDING / 2u];
    spark2_5_rope_table(SPARK_ROT_SLIDING / 2u, SPARK_ROT_SLIDING,
                        SPARK_THETA_SLIDING, inv);
    if (!ds4_gpu_tensor_write(g->inv_sliding, 0, inv,
            (uint64_t)(SPARK_ROT_SLIDING / 2u) * f32)) {
        spark2_5_graph_release(g);
        return NULL;
    }
    spark2_5_rope_table(SPARK_ROT_FULL / 2u, SPARK_ROT_FULL,
                        SPARK_THETA_FULL, inv);
    if (!ds4_gpu_tensor_write(g->inv_full, 0, inv,
            (uint64_t)(SPARK_ROT_FULL / 2u) * f32)) {
        spark2_5_graph_release(g);
        return NULL;
    }
    for (uint32_t il = 0; il < SPARK_LAYERS; il++) {
        const uint32_t kv_rows = spark2_5_kv_rows(il, ctx, cap);
        const uint64_t bytes = (uint64_t)kv_rows * SPARK_KV_DIM * sizeof(uint16_t);
        g->kcache[il] = ds4_gpu_tensor_alloc(bytes);
        g->vcache[il] = ds4_gpu_tensor_alloc(bytes);
        if (!g->kcache[il] || !g->vcache[il]) {
            fprintf(stderr, "ds4: Spark-X2.6 KV allocation failed at layer %u\n", il);
            spark2_5_graph_release(g);
            return NULL;
        }
        g->kv_bytes += 2u * bytes;
    }
    ds4_log(stderr, DS4_LOG_TIMING,
        "ds4: Spark-X2.6 graph ctx=%u chunk=%u KV=%.2f GiB workspace=%.1f MiB%s\n",
        ctx, cap, (double)g->kv_bytes / 1073741824.0,
        (double)spark2_5_session_memory(ctx, cap).scratch_bytes / 1048576.0,
        w->spark_restored ? " (shared restored weights)" : " (per-call restore)");
    g->ready = spark2_5_graph_reset(g);
    return g;
}

/* ---------------- forward ---------------- */

/* Diagnostic only (DS4_SPARK2_5_DUMP_HIDDEN=<prefix>): writes the last row of
 * `x` as F32 to <prefix>.<tag>.f32 so a parity run can compare each stage
 * against the reference's output_hidden_states. One host readback per stage,
 * never on the production path. */
static bool spark2_5_dump_on(void) {
    static const char *prefix = NULL;
    static int checked = 0;
    if (!checked) {
        prefix = getenv("DS4_SPARK2_5_DUMP_HIDDEN");
        checked = 1;
    }
    return prefix && prefix[0];
}

static void spark2_5_dump_row(const ds4_gpu_tensor *x, uint64_t stride_elems,
                              const char *tag) {
    if (!spark2_5_dump_on()) { return; }
    const char *prefix = getenv("DS4_SPARK2_5_DUMP_HIDDEN");
    char path[1024];
    const int n = snprintf(path, sizeof(path), "%s.%s.f32", prefix, tag);
    if (n < 0 || (size_t)n >= sizeof(path)) { return; }
    float row[SPARK_EMBD];
    if (!ds4_gpu_tensor_read(x, stride_elems * sizeof(float), row, sizeof(row))) {
        return;
    }
    FILE *fp = fopen(path, "wb");
    if (!fp) { return; }
    (void)fwrite(row, sizeof(float), SPARK_EMBD, fp);
    (void)fclose(fp);
}

/* One decoder layer over `rows` activation rows starting at position pos0.
 * Every buffer is the graph's own; the layer differs from the next only in
 * its weights and its rope geometry. */
static bool spark2_5_layer(ds4_spark2_5_graph *g, const ds4_model *m,
        const ds4_weights *w, uint32_t il, uint32_t rows, uint32_t pos0) {
    const ds4_spark2_5_layer *l = &w->spark[il];
    const bool full = spark2_5_layer_full(il);
    const uint32_t pairs = spark2_5_rot(il) / 2u;
    const uint32_t kv_rows = spark2_5_kv_rows(il, g->ctx, g->cap);
    const uint32_t ring = full ? 0u : kv_rows;
    const uint64_t acts = (uint64_t)rows * SPARK_EMBD;
#define SPARK_STEP(expr, stage) do {                                          \
        if (!(expr)) {                                                        \
            fprintf(stderr, "ds4: Spark-X2.6 layer %u failed at %s\n",        \
                il, stage);                                                   \
            return false;                                                     \
        }                                                                     \
    } while (0)

    /* Pre-norm, then the fused q|k|v projection and the per-head gate
     * projection from the same normed row (modeling_spark.py:170-176). */
    SPARK_STEP(ds4_gpu_rms_norm_weight_rows_tensor(g->norm, g->cur, m->map,
        m->size, l->attn_norm->abs_offset, SPARK_EMBD, rows,
        SPARK_RMS_EPS), "attention norm");
    if (spark2_5_dump_on()) {
        char tag[32];
        snprintf(tag, sizeof(tag), "norm_attn%u", il);
        spark2_5_dump_row(g->norm, (uint64_t)(rows - 1u) * SPARK_EMBD, tag);
    }
    SPARK_STEP(ds4_gpu_gvq_linear(g->gvq, g->qkv,
        spark2_5_index(m, l->qkv.w), g->norm, rows, DS4_GVQ_BF16), "qkv projection");
    SPARK_STEP(ds4_gpu_matmul_bf16_tensor(g->gate, m->map, m->size,
        l->g_proj->abs_offset, SPARK_EMBD, SPARK_HEADS, g->norm, rows),
        "gate projection");
    SPARK_STEP(ds4_gpu_spark_qkv_split(g->q, g->k, g->v, g->qkv, rows),
        "qkv split");

    /* q and k are BF16 in the reference (the linear's dtype), rotated in F32
     * and stored BF16 again (modeling_spark.py:43-56). */
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->q, g->q,
        (uint64_t)rows * SPARK_Q_DIM), "q rounding");
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->k, g->k,
        (uint64_t)rows * SPARK_KV_DIM), "k rounding");
    SPARK_STEP(ds4_gpu_spark_rope(g->q, g->positions,
        full ? g->inv_full : g->inv_sliding, rows, SPARK_HEADS,
        SPARK_HEAD_DIM, pairs), "query rope");
    SPARK_STEP(ds4_gpu_spark_rope(g->k, g->positions,
        full ? g->inv_full : g->inv_sliding, rows, SPARK_KV_HEADS,
        SPARK_HEAD_DIM, pairs), "key rope");
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->q, g->q,
        (uint64_t)rows * SPARK_Q_DIM), "q rounding after rope");
    /* Folded attention scale (see deviation 4): 2^-4 scales q exactly. */
    SPARK_STEP(ds4_gpu_dots3_scale_tensor(g->q, SPARK_ATTN_SCALE,
        (uint64_t)rows * SPARK_Q_DIM), "query scale");
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->k, g->k,
        (uint64_t)rows * SPARK_KV_DIM), "k rounding after rope");
    SPARK_STEP(ds4_gpu_g4_kv_store(g->kcache[il], g->k, g->positions, rows,
        SPARK_KV_DIM, ring, kv_rows), "key cache store");
    SPARK_STEP(ds4_gpu_g4_kv_store(g->vcache[il], g->v, g->positions, rows,
        SPARK_KV_DIM, ring, kv_rows), "value cache store");

    /* 4:1 grouped causal attention, sliding on 27 layers and full on 9
     * (modeling_spark.py:60-96; config.json sliding_window 512). */
    SPARK_STEP(ds4_gpu_g4_attn(g->attn, g->q, g->kcache[il], g->vcache[il],
        g->positions, rows, pos0, SPARK_HEADS, SPARK_KV_HEADS,
        SPARK_HEAD_DIM, full ? 0u : SPARK_WINDOW, ring, 0u), "attention");
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->attn, g->attn,
        (uint64_t)rows * SPARK_Q_DIM), "attention output rounding");

    /* Headwise sigmoid output gate, then the output projection
     * (modeling_spark.py:194-205). */
    SPARK_STEP(ds4_gpu_dots3_gate_mul_tensor(g->attn, g->gate, rows,
        SPARK_HEADS, SPARK_HEAD_DIM), "headwise gate");
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->attn, g->attn,
        (uint64_t)rows * SPARK_Q_DIM), "gated output rounding");
    SPARK_STEP(ds4_gpu_gvq_linear(g->gvq, g->branch,
        spark2_5_index(m, l->out.w), g->attn, rows, DS4_GVQ_BF16),
        "output projection");
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->branch, g->branch,
        acts), "output projection rounding");
    SPARK_STEP(ds4_gpu_add_tensor(g->cur, g->cur, g->branch,
        (uint32_t)acts), "attention residual");

    /* Dense GELU FFN, 10,240 wide (modeling_spark.py:126-128, :207-212). */
    SPARK_STEP(ds4_gpu_rms_norm_weight_rows_tensor(g->norm, g->cur, m->map,
        m->size, l->ffn_norm->abs_offset, SPARK_EMBD, rows,
        SPARK_RMS_EPS), "FFN norm");
    if (spark2_5_dump_on()) {
        char tag[32];
        snprintf(tag, sizeof(tag), "norm_ffn%u", il);
        spark2_5_dump_row(g->norm, (uint64_t)(rows - 1u) * SPARK_EMBD, tag);
    }
    {
        const uint32_t proj[2] = {
            spark2_5_index(m, l->ffn_gate.w),
            spark2_5_index(m, l->ffn_up.w),
        };
        ds4_gpu_tensor *outs[2] = {g->ffn_gate, g->ffn_up};
        SPARK_STEP(ds4_gpu_gvq_linear_many(g->gvq, outs, proj, 2u, g->norm,
            rows, DS4_GVQ_BF16), "FFN gate/up projection");
    }
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->ffn_gate, g->ffn_gate,
        (uint64_t)rows * SPARK_FF), "FFN gate rounding");
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->ffn_up, g->ffn_up,
        (uint64_t)rows * SPARK_FF), "FFN up rounding");
    SPARK_STEP(ds4_gpu_spark_gelu_mul(g->ffn_mid, g->ffn_gate, g->ffn_up,
        (uint64_t)rows * SPARK_FF), "GELU product");
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->ffn_mid, g->ffn_mid,
        (uint64_t)rows * SPARK_FF), "FFN mid rounding");
    SPARK_STEP(ds4_gpu_gvq_linear(g->gvq, g->ffn_out,
        spark2_5_index(m, l->ffn_down.w), g->ffn_mid, rows, DS4_GVQ_BF16),
        "FFN down projection");
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->ffn_out, g->ffn_out,
        acts), "FFN down rounding");
    SPARK_STEP(ds4_gpu_add_tensor(g->cur, g->cur, g->ffn_out,
        (uint32_t)acts), "FFN residual");
#undef SPARK_STEP
    return true;
}

/* The tied head over the BF16 token table, on the last row only: the normal
 * forward returns logits for the row a sampler reads. */
static bool spark2_5_head(ds4_spark2_5_graph *g, const ds4_model *m,
        const ds4_weights *w, uint32_t rows) {
    ds4_gpu_tensor *row = ds4_gpu_tensor_view(g->cur,
        (uint64_t)(rows - 1u) * SPARK_EMBD * sizeof(float),
        (uint64_t)SPARK_EMBD * sizeof(float));
    if (!row) { return false; }
    const bool ok =
        ds4_gpu_rms_norm_weight_rows_tensor(g->norm, row, m->map, m->size,
            w->output_norm->abs_offset, SPARK_EMBD, 1u, SPARK_RMS_EPS) != 0 &&
        ds4_gpu_motif3_round_bf16_tensor(g->norm, g->norm,
            SPARK_EMBD) != 0;
    if (ok) { spark2_5_dump_row(g->norm, 0, "final"); }
    const bool head_ok = ok &&
        ds4_gpu_matmul_bf16_tensor(g->logits, m->map, m->size,
            w->output->abs_offset, SPARK_EMBD, SPARK_VOCAB, g->norm, 1u) != 0;
    ds4_gpu_tensor_free(row);
    return head_ok;
}

static bool spark2_5_forward(ds4_spark2_5_graph *g, const ds4_model *m,
        const ds4_weights *w, const uint32_t *tokens, uint32_t rows,
        uint32_t pos0, bool head) {
    if (!g || !g->ready || g->failed || !tokens || !rows || rows > g->cap ||
        pos0 > g->ctx || rows > g->ctx - pos0) {
        return false;
    }
    uint32_t positions[SPARK_PREFILL_MAX];
    for (uint32_t r = 0; r < rows; r++) { positions[r] = pos0 + r; }
    if (!ds4_gpu_tensor_write(g->tokens, 0, tokens,
            (uint64_t)rows * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(g->positions, 0, positions,
            (uint64_t)rows * sizeof(uint32_t))) {
        return false;
    }
    /* Input embedding: the table is the checkpoint's BF16 table, cast to the
     * F32 residual stream (modeling_spark.py:338, :390). */
    if (!ds4_gpu_embed_tokens_quant_tensor(g->cur, g->tokens, m->map, m->size,
            w->token_embd->abs_offset, DS4_TENSOR_BF16, SPARK_VOCAB, rows,
            SPARK_EMBD)) {
        return false;
    }
    spark2_5_dump_row(g->cur, (uint64_t)(rows - 1u) * SPARK_EMBD, "embed");
    for (uint32_t il = 0; il < SPARK_LAYERS; il++) {
        if (!spark2_5_layer(g, m, w, il, rows, pos0)) {
            g->failed = true;
            return false;
        }
        char tag[32];
        snprintf(tag, sizeof(tag), "layer%u", il + 1u);
        spark2_5_dump_row(g->cur, (uint64_t)(rows - 1u) * SPARK_EMBD, tag);
    }
    if (head && !spark2_5_head(g, m, w, rows)) {
        g->failed = true;
        return false;
    }
    g->position = pos0 + rows;
    return true;
}

#endif
