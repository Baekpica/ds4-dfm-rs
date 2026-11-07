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
        const uint32_t rows = spark2_5_kv_rows(il, ctx, cap);
        /* A FP8 layer keeps K, V and their scales in one row
         * (cuda/spark2_5.cuh); a BF16 layer keeps two BF16 planes.
         * The allocator and the host quote share this function. */
        if (spark2_5_layer_kv_fp8(il)) {
            bytes += (uint64_t)rows * spark2_5_fp8_row_bytes();
        } else {
            bytes += 2u * (uint64_t)rows * SPARK_KV_DIM * sizeof(uint16_t);
        }
    }
    return bytes;
}

static uint32_t spark2_5_prefill_cap(uint32_t ctx) {
    uint32_t cap = ctx < SPARK_PREFILL_MAX ? ctx : SPARK_PREFILL_MAX;
    const char *env = getenv("DS4_SPARK2_5_PREFILL_CHUNK");
    if (env && env[0]) {
        const long asked = strtol(env, NULL, 10);
        if (asked > 0 && asked <= SPARK_PREFILL_MAX) { cap = (uint32_t)asked; }
    }
    return cap < ctx ? cap : ctx;
}

/* The widest matrix is the 10,240 x 2,560 FFN pair: one restored FP8 or BF16
 * row set. With the shared image on (the default) a graph needs no weight
 * scratch at all; DS4_SPARK2_5_RESTORE=0 restores each matrix into this
 * buffer before the GEMM. */
static uint64_t spark2_5_widest_matrix_bytes(void) {
    return spark2_5_weights_fp8()
        ? spark2_5_fp8_matrix_bytes(SPARK_FF, SPARK_EMBD)
        : (uint64_t)SPARK_FF * SPARK_EMBD * sizeof(uint16_t);
}

static uint64_t spark2_5_linear_workspace(void) {
    const char *restore_env = getenv("DS4_SPARK2_5_RESTORE");
    const int image = !restore_env || strcmp(restore_env, "0") != 0;
    return image ? 0 : spark2_5_widest_matrix_bytes();
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
        spark2_5_linear_workspace();
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

/* The 180 VQ matrices as one BF16 image owned by ds4_weights and shared by
 * every session's graph. `spark2_5_dims` is the (input, output) pair of each
 * of the five matrices, in the order the layers bind them: q_k_v_proj,
 * out_proj, gate_proj, up_proj, down_proj (modeling_spark.py:152-212). */
static const uint32_t spark2_5_dims[5][2] = {
    {SPARK_EMBD, SPARK_QKV_DIM}, {SPARK_Q_DIM, SPARK_EMBD},
    {SPARK_EMBD, SPARK_FF}, {SPARK_EMBD, SPARK_FF}, {SPARK_FF, SPARK_EMBD},
};

static ds4_spark2_5_matrix *spark2_5_matrix(ds4_spark2_5_layer *l, uint32_t k) {
    switch (k) {
    case 0: return &l->qkv;
    case 1: return &l->out;
    case 2: return &l->ffn_gate;
    case 3: return &l->ffn_up;
    default: return &l->ffn_down;
    }
}

/* One matrix into `dst` (out rows x in cols FP8 E4M3+scales or BF16). The
 * stage-2 stream is present only on a container version 3 release. */
static bool spark2_5_restore_matrix(const ds4_model *m,
        const ds4_spark2_5_matrix *mx, ds4_gpu_tensor *dst,
        uint32_t out, uint32_t in) {
    const uint8_t *map = (const uint8_t *)m->map;
    if (spark2_5_weights_fp8()) {
        return ds4_gpu_spark_restore_fp8(dst,
            map + mx->w->abs_offset, mx->w->bytes,
            mx->stage2 ? map + mx->stage2->abs_offset : NULL,
            mx->stage2 ? mx->stage2->bytes : 0,
            map + mx->book->abs_offset,
            mx->stage2_book ? map + mx->stage2_book->abs_offset : NULL,
            map + mx->res->abs_offset, mx->res->bytes,
            map + mx->res_book->abs_offset, out, in) != 0;
    }
    return ds4_gpu_spark_restore(dst,
        map + mx->w->abs_offset, mx->w->bytes,
        mx->stage2 ? map + mx->stage2->abs_offset : NULL,
        mx->stage2 ? mx->stage2->bytes : 0,
        map + mx->book->abs_offset,
        mx->stage2_book ? map + mx->stage2_book->abs_offset : NULL,
        map + mx->res->abs_offset, mx->res->bytes,
        map + mx->res_book->abs_offset, out, in) != 0;
}

/* DS4_SPARK2_5_CODES=1: no decoded image; the artifact's codewords are
 * copied to the device once and every projection decodes them in the GEMM
 * (cuda/spark2_5_vq_gemm.cuh). */
static bool spark2_5_codes_on(void) {
    const char *env = getenv("DS4_SPARK2_5_CODES");
    return env && env[0] == '1' && env[1] == '\0';
}

/* Build the engine's BF16 image. The restore runs the family's own kernel
 * (cuda/spark2_5.cuh spark_restore_kernel) over both container versions, so
 * the image is the same whether the release carries one 13-bit stream or two.
 * DS4_SPARK2_5_RESTORE=0 binds no image: each linear restores into the
 * graph's scratch instead. */
static bool spark2_5_gpu_bind(ds4_weights *w, const ds4_model *m) {
    if (!w->spark) { return false; }
    if (spark2_5_codes_on()) {
        /* A codeword read from the host-registered map crosses PCIe every
         * time; one copy of the artifact keeps the fused path on device. */
        w->spark_codes = ds4_gpu_tensor_alloc(m->size);
        if (!w->spark_codes ||
            !ds4_gpu_spark_codes_copy(w->spark_codes, m->map, m->size)) {
            return false;
        }
        fprintf(stderr,
            "ds4: Spark-X2.6 copied %.2f GiB of codewords to the device"
            " (fused VQ projection, no BF16 image)\n",
            (double)m->size / 1073741824.0);
        const uint8_t *map = (const uint8_t *)m->map;
        const void *arena = ds4_gpu_spark_codes_base(w->spark_codes);
        for (uint32_t il = 0; il < SPARK_LAYERS; il++) {
            for (uint32_t k = 0; k < 5; k++) {
                ds4_spark2_5_matrix *mx = spark2_5_matrix(&w->spark[il], k);
                if (!ds4_gpu_spark_vq_resolve(&mx->vq, map, arena,
                        map + mx->w->abs_offset, mx->w->bytes,
                        mx->stage2 ? map + mx->stage2->abs_offset : NULL,
                        mx->stage2 ? mx->stage2->bytes : 0,
                        map + mx->book->abs_offset,
                        mx->stage2_book
                            ? map + mx->stage2_book->abs_offset : NULL,
                        map + mx->res->abs_offset, mx->res->bytes,
                        map + mx->res_book->abs_offset, spark2_5_dims[k][1],
                        spark2_5_dims[k][0])) {
                    fprintf(stderr,
                        "ds4: Spark-X2.6 fused stream resolve failed at"
                        " layer %u\n", il);
                    return false;
                }
            }
        }
        /* Diagnostic (DS4_SPARK2_5_DUMP_WEIGHT): decode layer 0's down_proj
         * with the fused arithmetic, for a diff against the image path. */
        const char *dump = getenv("DS4_SPARK2_5_DUMP_WEIGHT");
        if (dump && dump[0]) {
            const ds4_spark2_5_matrix *mx = &w->spark[0].ffn_down;
            ds4_gpu_tensor *tmp = ds4_gpu_tensor_alloc(
                (uint64_t)mx->vq.rows * mx->vq.cols * sizeof(uint16_t));
            uint16_t head[16] = {0};
            if (tmp && ds4_gpu_spark_vq_probe(tmp, &mx->vq)) {
                FILE *fp = fopen(dump, "wb");
                if (fp && ds4_gpu_tensor_read(tmp, 0, head, sizeof(head))) {
                    (void)fwrite(head, sizeof(head), 1, fp);
                    fprintf(stderr,
                        "ds4: Spark-X2.6 fused decode dumped to %s\n", dump);
                }
                if (fp) { (void)fclose(fp); }
            }
            ds4_gpu_tensor_free(tmp);
        }
        return true;
    }
    const char *restore_env = getenv("DS4_SPARK2_5_RESTORE");
    if (restore_env && strcmp(restore_env, "0") == 0) { return true; }

    const int fp8_weights = spark2_5_weights_fp8();
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < SPARK_LAYERS; il++) {
        for (uint32_t k = 0; k < 5; k++) {
            const ds4_spark2_5_matrix *mx = spark2_5_matrix(&w->spark[il], k);
            if (!mx->w || !mx->book || !mx->res || !mx->res_book ||
                (mx->stage2 && !mx->stage2_book)) {
                return false;
            }
            const uint32_t in = spark2_5_dims[k][0], out = spark2_5_dims[k][1];
            bytes += fp8_weights
                ? spark2_5_fp8_matrix_bytes(in, out)
                : (uint64_t)in * out * sizeof(uint16_t);
        }
        if (fp8_weights) {
            bytes += spark2_5_fp8_matrix_bytes(SPARK_EMBD, SPARK_HEADS);
            bytes += 2ull * SPARK_EMBD * sizeof(float);
        }
    }
    if (fp8_weights) {
        bytes += (uint64_t)SPARK_EMBD * sizeof(float);
        bytes += spark2_5_fp8_matrix_bytes(SPARK_EMBD, SPARK_VOCAB);
    }
    w->spark_image = ds4_gpu_tensor_alloc(bytes);
    if (!w->spark_image) { return false; }

    const uint8_t *map = (const uint8_t *)m->map;
    uint64_t off = 0;
    for (uint32_t il = 0; il < SPARK_LAYERS; il++) {
        ds4_spark2_5_layer *l = &w->spark[il];
        for (uint32_t k = 0; k < 5; k++) {
            ds4_spark2_5_matrix *mx = spark2_5_matrix(l, k);
            const uint32_t in = spark2_5_dims[k][0], out = spark2_5_dims[k][1];
            const uint64_t slice = fp8_weights
                ? spark2_5_fp8_matrix_bytes(in, out)
                : (uint64_t)in * out * sizeof(uint16_t);
            ds4_gpu_tensor *view = ds4_gpu_tensor_view(w->spark_image, off, slice);
            if (!view) { return false; }
            const bool ok = spark2_5_restore_matrix(m, mx, view, out, in);
            ds4_gpu_tensor_free(view);
            if (!ok) { return false; }
            mx->image_off = off;
            off += slice;
        }
        if (fp8_weights) {
            const uint64_t g_slice = spark2_5_fp8_matrix_bytes(SPARK_EMBD, SPARK_HEADS);
            ds4_gpu_tensor *g_view = ds4_gpu_tensor_view(w->spark_image, off, g_slice);
            if (!g_view) { return false; }
            const int g_ok = ds4_gpu_spark_bf16_to_fp8(
                g_view, map + l->g_proj->abs_offset, SPARK_HEADS, SPARK_EMBD);
            ds4_gpu_tensor_free(g_view);
            if (!g_ok) { return false; }
            l->g_proj_off = off;
            off += g_slice;

            const uint64_t norm_bytes = (uint64_t)SPARK_EMBD * sizeof(float);
            if (!ds4_gpu_tensor_write(w->spark_image, off,
                    map + l->attn_norm->abs_offset, norm_bytes)) {
                return false;
            }
            l->attn_norm_off = off;
            off += norm_bytes;

            if (!ds4_gpu_tensor_write(w->spark_image, off,
                    map + l->ffn_norm->abs_offset, norm_bytes)) {
                return false;
            }
            l->ffn_norm_off = off;
            off += norm_bytes;
        }
    }
    if (fp8_weights) {
        const uint64_t norm_bytes = (uint64_t)SPARK_EMBD * sizeof(float);
        if (!ds4_gpu_tensor_write(w->spark_image, off,
                map + w->output_norm->abs_offset, norm_bytes)) {
            return false;
        }
        w->spark_output_norm_off = off;
        off += norm_bytes;

        const uint64_t out_slice = spark2_5_fp8_matrix_bytes(SPARK_EMBD, SPARK_VOCAB);
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(w->spark_image, off, out_slice);
        if (!out_view) { return false; }
        const int out_ok = ds4_gpu_spark_bf16_to_fp8(
            out_view, map + w->output->abs_offset, SPARK_VOCAB, SPARK_EMBD);
        ds4_gpu_tensor_free(out_view);
        if (!out_ok) { return false; }
        w->spark_output_off = off;
        off += out_slice;
    }
    fprintf(stderr, "ds4: Spark-X2.6 restored %.2f GiB of VQ + head weights to one %s image\n",
        (double)bytes / 1073741824.0, fp8_weights ? "FP8" : "BF16");
    /* Diagnostic only (DS4_SPARK2_5_DUMP_WEIGHT=<file>): the first 16 BF16
     * values of layer 0's down_proj, so a parity run can check the restore
     * against a hand decode of the container (stage 1 + residual + stage 2). */
    const char *dump = getenv("DS4_SPARK2_5_DUMP_WEIGHT");
    if (dump && dump[0]) {
        uint16_t head[16] = {0};
        FILE *fp = fopen(dump, "wb");
        if (fp && ds4_gpu_tensor_read(w->spark_image,
                w->spark[0].ffn_down.image_off, head, sizeof(head))) {
            (void)fwrite(head, sizeof(head), 1, fp);
        }
        if (fp) { (void)fclose(fp); }
    }
    return true;
}

/* ---------------- graph ---------------- */

typedef struct ds4_spark2_5_graph {
    /* Weight scratch only when DS4_SPARK2_5_RESTORE=0: one restored matrix
     * (the widest, 52.4 MB) before its GEMM. Otherwise the engine's BF16
     * image (ds4_weights spark_image) holds every weight. */
    ds4_gpu_tensor *wbf16;
    uint32_t ctx, cap, rows, position;
    /* The highest position written since the last reset, and the position a
     * restore brought back: a resume is exact only where the sliding ring
     * still holds the window (spark2_5_graph_keeps, ds4.c ring_holds). */
    uint32_t high, restored;
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
    /* A FP8 layer's combined K|V|scales rows; kcache/vcache stay NULL
     * for those layers (cuda/spark2_5.cuh). */
    ds4_gpu_tensor *fkcache[SPARK_LAYERS];
    int full_fp8;
    int sliding_fp8;
    int weights_fp8;
    uint64_t kv_bytes;
    /* A bank graph borrows graph 0's activations, rope tables, token and
     * position rows and logits; it owns its KV caches only, so clearing it
     * must not free the owner's buffers (spark2_5_graph_bank). */
    bool borrowed;
    bool ready;
    bool failed;
} ds4_spark2_5_graph;

static void spark2_5_graph_clear(ds4_spark2_5_graph *g) {
    if (!g) { return; }
    for (uint32_t il = 0; il < SPARK_LAYERS; il++) {
        ds4_gpu_tensor_free(g->kcache[il]);
        ds4_gpu_tensor_free(g->vcache[il]);
        ds4_gpu_tensor_free(g->fkcache[il]);
    }
    if (g->borrowed) {
        memset(g, 0, sizeof(*g));
        return;
    }
    ds4_gpu_tensor_free(g->wbf16);
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
    g->high = g->restored = 0;
    g->failed = false;
    return true;
}

/* ds4.c's serial ring rule (defined with the reuse helpers, below this
 * include): a ring of `ring` rows written up to `high` still holds the
 * `window` rows under `k`. Declared here so the bank lane, which is included
 * above it, can ask the same question. */
static bool ring_holds(uint32_t ring, uint32_t window, uint32_t high,
                       uint32_t restored, uint32_t k);

/* Whether `g` can resume at position k: the full layers keep every position,
 * the sliding ring (layer 0 slides) must still hold the window under k. */
static bool spark2_5_graph_keeps(const ds4_spark2_5_graph *g, uint32_t k) {
    return ring_holds(spark2_5_kv_rows(0u, g->ctx, g->cap), SPARK_WINDOW,
                      g->high, g->restored, k);
}

/* A bank graph: graph 0's activations, rope tables, token and position rows
 * and logits, plus this bank's own KV caches of both layouts. Bank forwards
 * are serialized under the engine generation lock, so sharing the scratch is
 * exact (ds4_gemma4_graph.inc:282-322, the same split). */
static bool spark2_5_graph_bank(ds4_spark2_5_graph *g,
                                const ds4_spark2_5_graph *owner) {
    if (!g || !owner || !owner->ready) { return false; }
    *g = *owner;
    g->borrowed = true;
    g->ready = false;
    g->failed = false;
    g->position = 0;
    g->high = g->restored = 0;
    g->kv_bytes = 0;
    for (uint32_t il = 0; il < SPARK_LAYERS; il++) {
        g->kcache[il] = g->vcache[il] = NULL;
        g->fkcache[il] = NULL;
    }
    for (uint32_t il = 0; il < SPARK_LAYERS; il++) {
        const uint32_t rows = spark2_5_kv_rows(il, g->ctx, g->cap);
        const bool layer_fp8 = spark2_5_layer_full(il)
            ? (g->full_fp8 != 0) : (g->sliding_fp8 != 0);
        if (layer_fp8) {
            const uint64_t bytes = (uint64_t)rows * spark2_5_fp8_row_bytes();
            g->fkcache[il] = ds4_gpu_tensor_alloc(bytes);
            if (!g->fkcache[il]) {
                fprintf(stderr,
                    "ds4: Spark-X2.6 bank FP8 KV allocation failed at layer %u\n", il);
                spark2_5_graph_clear(g);
                return false;
            }
            g->kv_bytes += bytes;
            continue;
        }
        const uint64_t bytes = (uint64_t)rows * SPARK_KV_DIM * sizeof(uint16_t);
        g->kcache[il] = ds4_gpu_tensor_alloc(bytes);
        g->vcache[il] = ds4_gpu_tensor_alloc(bytes);
        if (!g->kcache[il] || !g->vcache[il]) {
            fprintf(stderr, "ds4: Spark-X2.6 bank KV allocation failed at layer %u\n", il);
            spark2_5_graph_clear(g);
            return false;
        }
        g->kv_bytes += 2u * bytes;
    }
    g->ready = true;
    return spark2_5_graph_reset(g);
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
    if (!w->spark || !ctx || !cap || cap > ctx ||
        ctx > SPARK_CONTEXT || cap > SPARK_PREFILL_MAX) {
        return NULL;
    }
    ds4_spark2_5_graph *g = xcalloc(1, sizeof(*g));
    g->ctx = ctx;
    g->cap = cap;
    g->rows = cap;
    const uint32_t rows = g->rows;
    /* The restored BF16 image is the engine's (spark2_5_gpu_bind); a graph
     * only owns its KV state, its activation set and, without the image, one
     * matrix of scratch. */
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
    if (!w->spark_image && !w->spark_codes) { SPARK_ALLOC(wbf16, spark2_5_widest_matrix_bytes()); }
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
    g->full_fp8 = spark2_5_full_kv_fp8();
    g->sliding_fp8 = spark2_5_sliding_kv_fp8();
    g->weights_fp8 = spark2_5_weights_fp8();
    for (uint32_t il = 0; il < SPARK_LAYERS; il++) {
        const uint32_t kv_rows = spark2_5_kv_rows(il, ctx, cap);
        /* One combined E4M3 row per position on a FP8 layer; two BF16
         * planes everywhere else. */
        const bool layer_fp8 = spark2_5_layer_full(il)
            ? (g->full_fp8 != 0) : (g->sliding_fp8 != 0);
        if (layer_fp8) {
            const uint64_t bytes =
                (uint64_t)kv_rows * spark2_5_fp8_row_bytes();
            g->fkcache[il] = ds4_gpu_tensor_alloc(bytes);
            if (!g->fkcache[il]) {
                fprintf(stderr,
                    "ds4: Spark-X2.6 FP8 KV allocation failed at layer %u\n", il);
                spark2_5_graph_release(g);
                return NULL;
            }
            g->kv_bytes += bytes;
            continue;
        }
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
        "ds4: Spark-X2.6 graph ctx=%u chunk=%u KV=%.2f GiB workspace=%.1f MiB"
        " full_kv=%s sliding_kv=%s weights=%s%s\n",
        ctx, cap, (double)g->kv_bytes / 1073741824.0,
        (double)spark2_5_session_memory(ctx, cap).scratch_bytes / 1048576.0,
        g->full_fp8 ? "fp8" : "bf16",
        g->sliding_fp8 ? "fp8" : "bf16",
        w->spark_codes ? "vq_codes" : (g->weights_fp8 ? "fp8" : "bf16"),
        w->spark_codes ? " (fused VQ projection over the copied artifact)"
        : w->spark_image ? " (shared restored image)" : " (per-call restore)");
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

/* One VQ projection as a FP8 or BF16 GEMM/GEMV over the engine's restored
 * image, or over a scratch restore when the image is off
 * (DS4_SPARK2_5_RESTORE=0). */
static bool spark2_5_linear(ds4_spark2_5_graph *g, const ds4_weights *w,
        const ds4_model *m, const ds4_spark2_5_matrix *mx,
        ds4_gpu_tensor *out, const ds4_gpu_tensor *in,
        uint32_t rows, uint32_t in_dim, uint32_t out_dim) {
    if (w->spark_codes) {
        return ds4_gpu_spark_vq_linear(out, &mx->vq, in, rows) != 0;
    }
    if (!w->spark_image) {
        if (!g->wbf16 || !spark2_5_restore_matrix(m, mx, g->wbf16, out_dim, in_dim)) {
            return false;
        }
        return (g->weights_fp8
            ? ds4_gpu_spark_matmul_fp8(out, g->wbf16, in_dim, out_dim, in, rows)
            : ds4_gpu_v41_matmul_bf16_dev_tensor(out, g->wbf16, in_dim,
                out_dim, in, rows)) != 0;
    }
    const uint64_t slice = g->weights_fp8
        ? spark2_5_fp8_matrix_bytes(in_dim, out_dim)
        : (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    ds4_gpu_tensor *view = ds4_gpu_tensor_view(w->spark_image, mx->image_off, slice);
    if (!view) { return false; }
    const int ok = g->weights_fp8
        ? ds4_gpu_spark_matmul_fp8(out, view, in_dim, out_dim, in, rows)
        : ds4_gpu_v41_matmul_bf16_dev_tensor(out, view, in_dim, out_dim, in, rows);
    ds4_gpu_tensor_free(view);
    return ok != 0;
}

static bool spark2_5_rms_norm(const ds4_spark2_5_graph *g, const ds4_weights *w,
        const ds4_model *m, const ds4_tensor *norm_t, uint64_t image_off,
        ds4_gpu_tensor *out, const ds4_gpu_tensor *in, uint32_t rows) {
    if (w->spark_image && g->weights_fp8) {
        ds4_gpu_tensor *view = ds4_gpu_tensor_view(
            w->spark_image, image_off, (uint64_t)SPARK_EMBD * sizeof(float));
        if (!view) { return false; }
        const int ok = ds4_gpu_v41_rms_norm_dev_tensor(
            out, in, view, SPARK_EMBD, rows, SPARK_RMS_EPS);
        ds4_gpu_tensor_free(view);
        return ok != 0;
    }
    return ds4_gpu_rms_norm_weight_rows_tensor(out, in, m->map, m->size,
        norm_t->abs_offset, SPARK_EMBD, rows, SPARK_RMS_EPS) != 0;
}

static bool spark2_5_fp8_or_bf16_proj(const ds4_spark2_5_graph *g,
        const ds4_weights *w, const ds4_model *m, const ds4_tensor *bf16_t,
        uint64_t fp8_off, ds4_gpu_tensor *out, const ds4_gpu_tensor *in,
        uint32_t rows, uint32_t in_dim, uint32_t out_dim) {
    if (w->spark_image && g->weights_fp8) {
        const uint64_t slice = spark2_5_fp8_matrix_bytes(in_dim, out_dim);
        ds4_gpu_tensor *view = ds4_gpu_tensor_view(w->spark_image, fp8_off, slice);
        if (!view) { return false; }
        const int ok = ds4_gpu_spark_matmul_fp8(out, view, in_dim, out_dim, in, rows);
        ds4_gpu_tensor_free(view);
        return ok != 0;
    }
    return ds4_gpu_matmul_bf16_tensor(out, m->map, m->size, bf16_t->abs_offset,
        in_dim, out_dim, in, rows) != 0;
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
    SPARK_STEP(spark2_5_rms_norm(g, w, m, l->attn_norm, l->attn_norm_off,
        g->norm, g->cur, rows), "attention norm");
    if (spark2_5_dump_on()) {
        char tag[32];
        snprintf(tag, sizeof(tag), "norm_attn%u", il);
        spark2_5_dump_row(g->norm, (uint64_t)(rows - 1u) * SPARK_EMBD, tag);
    }
    SPARK_STEP(spark2_5_linear(g, w, m, &l->qkv, g->qkv, g->norm, rows,
        SPARK_EMBD, SPARK_QKV_DIM), "qkv projection");
    SPARK_STEP(spark2_5_fp8_or_bf16_proj(g, w, m, l->g_proj, l->g_proj_off,
        g->gate, g->norm, rows, SPARK_EMBD, SPARK_HEADS),
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
    /* Full and sliding layers keep E4M3 rows with one f16 scale per KV head
     * by default (DS4_SPARK2_5_FULL_KV=fp8); DS4_SPARK2_5_FULL_KV=bf16 keeps
     * BF16 planes. Both lanes share the attention geometry below. */
    if (g->fkcache[il]) {
        SPARK_STEP(ds4_gpu_spark_kv_store_fp8(g->fkcache[il], g->k, g->v,
            g->positions, rows, SPARK_KV_DIM, SPARK_KV_HEADS, SPARK_HEAD_DIM,
            ring, kv_rows), "FP8 key/value cache store");
        SPARK_STEP(ds4_gpu_spark_attn_fp8(g->attn, g->q, g->fkcache[il],
            g->positions, rows, pos0, SPARK_HEADS, SPARK_KV_HEADS,
            SPARK_HEAD_DIM, full ? 0u : SPARK_WINDOW, ring, 0u,
            spark2_5_fp8_row_bytes()),
            "FP8 attention");
    } else {
        SPARK_STEP(ds4_gpu_g4_kv_store(g->kcache[il], g->k, g->positions, rows,
            SPARK_KV_DIM, ring, kv_rows), "key cache store");
        SPARK_STEP(ds4_gpu_g4_kv_store(g->vcache[il], g->v, g->positions, rows,
            SPARK_KV_DIM, ring, kv_rows), "value cache store");

        /* 4:1 grouped causal attention, sliding on 27 layers and full on 9
         * (modeling_spark.py:60-96; config.json sliding_window 512). */
        SPARK_STEP(ds4_gpu_g4_attn(g->attn, g->q, g->kcache[il], g->vcache[il],
            g->positions, rows, pos0, SPARK_HEADS, SPARK_KV_HEADS,
            SPARK_HEAD_DIM, full ? 0u : SPARK_WINDOW, ring, 0u), "attention");
    }
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->attn, g->attn,
        (uint64_t)rows * SPARK_Q_DIM), "attention output rounding");

    /* Headwise sigmoid output gate, then the output projection
     * (modeling_spark.py:194-205). */
    SPARK_STEP(ds4_gpu_dots3_gate_mul_tensor(g->attn, g->gate, rows,
        SPARK_HEADS, SPARK_HEAD_DIM), "headwise gate");
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->attn, g->attn,
        (uint64_t)rows * SPARK_Q_DIM), "gated output rounding");
    SPARK_STEP(spark2_5_linear(g, w, m, &l->out, g->branch, g->attn, rows,
        SPARK_Q_DIM, SPARK_EMBD), "output projection");
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->branch, g->branch,
        acts), "output projection rounding");
    SPARK_STEP(ds4_gpu_add_tensor(g->cur, g->cur, g->branch,
        (uint32_t)acts), "attention residual");

    /* Dense GELU FFN, 10,240 wide (modeling_spark.py:126-128, :207-212). */
    SPARK_STEP(spark2_5_rms_norm(g, w, m, l->ffn_norm, l->ffn_norm_off,
        g->norm, g->cur, rows), "FFN norm");
    if (spark2_5_dump_on()) {
        char tag[32];
        snprintf(tag, sizeof(tag), "norm_ffn%u", il);
        spark2_5_dump_row(g->norm, (uint64_t)(rows - 1u) * SPARK_EMBD, tag);
    }
    SPARK_STEP(spark2_5_linear(g, w, m, &l->ffn_gate, g->ffn_gate, g->norm,
        rows, SPARK_EMBD, SPARK_FF), "FFN gate projection");
    SPARK_STEP(spark2_5_linear(g, w, m, &l->ffn_up, g->ffn_up, g->norm,
        rows, SPARK_EMBD, SPARK_FF), "FFN up projection");
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->ffn_gate, g->ffn_gate,
        (uint64_t)rows * SPARK_FF), "FFN gate rounding");
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->ffn_up, g->ffn_up,
        (uint64_t)rows * SPARK_FF), "FFN up rounding");
    SPARK_STEP(ds4_gpu_spark_gelu_mul(g->ffn_mid, g->ffn_gate, g->ffn_up,
        (uint64_t)rows * SPARK_FF), "GELU product");
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->ffn_mid, g->ffn_mid,
        (uint64_t)rows * SPARK_FF), "FFN mid rounding");
    SPARK_STEP(spark2_5_linear(g, w, m, &l->ffn_down, g->ffn_out, g->ffn_mid,
        rows, SPARK_FF, SPARK_EMBD), "FFN down projection");
    SPARK_STEP(ds4_gpu_motif3_round_bf16_tensor(g->ffn_out, g->ffn_out,
        acts), "FFN down rounding");
    SPARK_STEP(ds4_gpu_add_tensor(g->cur, g->cur, g->ffn_out,
        (uint32_t)acts), "FFN residual");
#undef SPARK_STEP
    return true;
}

/* The tied head over the FP8/BF16 token table, on the last row only: the normal
 * forward returns logits for the row a sampler reads. */
static bool spark2_5_head(ds4_spark2_5_graph *g, const ds4_model *m,
        const ds4_weights *w, uint32_t rows) {
    ds4_gpu_tensor *row = ds4_gpu_tensor_view(g->cur,
        (uint64_t)(rows - 1u) * SPARK_EMBD * sizeof(float),
        (uint64_t)SPARK_EMBD * sizeof(float));
    if (!row) { return false; }
    const bool ok =
        spark2_5_rms_norm(g, w, m, w->output_norm, w->spark_output_norm_off,
            g->norm, row, 1u) &&
        ds4_gpu_motif3_round_bf16_tensor(g->norm, g->norm,
            SPARK_EMBD) != 0;
    if (ok) { spark2_5_dump_row(g->norm, 0, "final"); }
    const bool head_ok = ok &&
        spark2_5_fp8_or_bf16_proj(g, w, m, w->output, w->spark_output_off,
            g->logits, g->norm, 1u, SPARK_EMBD, SPARK_VOCAB);
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
    /* Input embedding: GPU-resident FP8 table by default, or BF16 checkpoint
     * table when DS4_SPARK2_5_WEIGHTS=bf16. */
    if (w->spark_image && g->weights_fp8) {
        const uint64_t out_slice = spark2_5_fp8_matrix_bytes(SPARK_EMBD, SPARK_VOCAB);
        ds4_gpu_tensor *tbl = ds4_gpu_tensor_view(
            w->spark_image, w->spark_output_off, out_slice);
        if (!tbl) { return false; }
        const int emb_ok = ds4_gpu_spark_embed_fp8(
            g->cur, g->tokens, tbl, SPARK_VOCAB, rows, SPARK_EMBD);
        ds4_gpu_tensor_free(tbl);
        if (!emb_ok) { return false; }
    } else if (!ds4_gpu_embed_tokens_quant_tensor(g->cur, g->tokens, m->map, m->size,
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
    if (g->position > g->high) { g->high = g->position; }
    return true;
}

#endif
