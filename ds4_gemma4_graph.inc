/* Gemma-4-26B-A4B VQ graph (host side). Included by ds4.c after the weight
 * bind and after the CUDA entry points in ds4_gemma4_gpu.cuh.
 *
 * The forward pass is docs/gemma4-reference-semantics.md section 3, layer by
 * layer, with every rounding the reference makes: seven norms, the routed
 * experts with their float32 weights, the dense FFN, the tanh-30 vocabulary
 * cap and the block's layer_scalar. No fused kernels: this is the correctness
 * path, and the sibling families' fused forms are a later measurement. */

#define G4_RMS_EPS 1e-6f
#define G4_ROUTER_ROOT (1.0f / 53.065996645686f)

/* Defined by ds4_gemma4_mtp.inc: the forward publishes the final-norm row the
 * drafter extrapolates from, and a rewrite drops it. */
struct ds4_gemma4_graph;
struct ds4_gemma4_mtp_runtime;
static bool gemma4_mtp_publish(struct ds4_gemma4_graph *g, uint32_t row);
static void gemma4_mtp_drop_hidden(struct ds4_gemma4_graph *g);
static void gemma4_mtp_runtime_free(struct ds4_gemma4_mtp_runtime *r);

typedef struct ds4_gemma4_graph {
    ds4_gvq_model *model;
    ds4_gvq_ctx *gvq;
    ds4_g4_moe *moe;
    /* Engine-owned SSD expert cache; NULL unless --ssd-streaming is on. */
    ds4_gemma4_stream *stream;
    /* `cap` is the prefill chunk (and the KV ring's margin); `rows` is the
     * activation row count, the wider of the chunk and one draft round's
     * verify block (gemma4_token_rows). */
    uint32_t ctx, cap, rows, position;
    ds4_gpu_tensor *cur, *norm, *branch, *tmp, *exp_acc;   /* [rows][2816] */
    ds4_gpu_tensor *q, *k, *v, *attn;                      /* [rows][heads*dim] */
    ds4_gpu_tensor *dense_g, *dense_u, *dense_mid;         /* [rows][2112] */
    ds4_gpu_tensor *router_logits;                         /* [rows][128] */
    ds4_gpu_tensor *selected, *weights;                    /* [rows*8] */
    ds4_gpu_tensor *logits;                                /* [vocab] */
    ds4_gpu_tensor *tokens;                                /* [rows] i32 */
    ds4_gpu_tensor *positions;                             /* [rows] u32 */
    ds4_gpu_tensor *inv_sliding, *inv_full;                /* rope tables */
    ds4_gpu_tensor *kcache[G4_LAYERS], *vcache[G4_LAYERS];
    /* The five full layers' combined E4M3 rows (K | V | scales) when
     * DS4_GEMMA4_FULL_KV=fp8; kcache/vcache stay NULL for those layers. */
    ds4_gpu_tensor *fkcache[G4_LAYERS];
    bool full_fp8;
    float scalar[G4_LAYERS];
    uint64_t kv_bytes;
    /* The drafter's device state, owned by graph 0 when `--mtp` armed it. */
    struct ds4_gemma4_mtp_runtime *mtp;
    bool ready;
    bool failed;
    /* A bank graph owns its KV caches only; every other pointer, the GVQ
     * context and the expert scratch belong to graph 0. Bank forwards are
     * serialized under the engine generation lock, so sharing them is
     * exact (ds4_gemma4_batch.inc). */
    bool borrowed;
} ds4_gemma4_graph;

/* The K/V the drafter consumes is the trunk's own, never a projection of its
 * own: modeling_gemma4.py:1219-1239 takes `shared_kv_states[layer_type]` for
 * a shared layer. The producer of each type is the last non-consumer trunk
 * layer of that type (configuration_gemma4.py:236-251), which for this trunk
 * is layer 28 (sliding, the ring) and layer 29 (full, the whole context) -
 * see G4_SHARED_SLIDING/G4_SHARED_FULL. The rows are already stored by the
 * ordinary forward, post k/v norm and post rope, so publishing is a read. */
static const ds4_gpu_tensor *gemma4_shared_k(const ds4_gemma4_graph *g, bool full) {
    /* The full layer is a combined FP8 row; the drafter reads bf16, so the
     * FP8 option refuses --mtp instead of publishing a row it cannot read. */
    if (full && g->full_fp8) { return NULL; }
    return g->kcache[gemma4_shared_layer(full)];
}

static const ds4_gpu_tensor *gemma4_shared_v(const ds4_gemma4_graph *g, bool full) {
    if (full && g->full_fp8) { return NULL; }
    return g->vcache[gemma4_shared_layer(full)];
}

static uint32_t gemma4_index(const ds4_model *m, const ds4_tensor *t) {
    return t ? (uint32_t)(t - m->tensors) : DS4_GVQ_NO_LINK;
}

/* Materialized source pointers belong to the weight owner. Each expert
 * matrix links its layer's book, as the container records it. */
static bool gemma4_gpu_bind(ds4_weights *w, const ds4_model *m) {
    if (!w->gemma4 || !m->n_tensors || m->n_tensors > UINT32_MAX) { return false; }
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
            (uint32_t)rows, (uint32_t)cols, gemma4_index(m, t->source_scale),
            DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK};
    }
    for (uint32_t il = 0; il < G4_LAYERS; il++) {
        const ds4_gemma4_layer *l = &w->gemma4[il];
        const uint32_t down = gemma4_index(m, l->book_down);
        const uint32_t gate_up = gemma4_index(m, l->book_gate_up);
        /* R13+8: each expert matrix points at its GVR8 plane and that plane's
         * record at the 256-entry residual book, the link gvq_link_res
         * validates at bind time. */
        const uint32_t res_down = gemma4_index(m, l->res_book_down);
        const uint32_t res_gate_up = gemma4_index(m, l->res_book_gate_up);
        for (uint32_t e = 0; e < G4_EXPERTS; e++) {
            records[gemma4_index(m, l->gate[e])].book = gate_up;
            records[gemma4_index(m, l->up[e])].book = gate_up;
            records[gemma4_index(m, l->down[e])].book = down;
            if (res_down != DS4_GVQ_NO_LINK && res_gate_up != DS4_GVQ_NO_LINK) {
                records[gemma4_index(m, l->gate[e])].res = gemma4_index(m, l->gate_res[e]);
                records[gemma4_index(m, l->up[e])].res = gemma4_index(m, l->up_res[e]);
                records[gemma4_index(m, l->down[e])].res = gemma4_index(m, l->down_res[e]);
                records[gemma4_index(m, l->gate_res[e])].book = res_gate_up;
                records[gemma4_index(m, l->up_res[e])].book = res_gate_up;
                records[gemma4_index(m, l->down_res[e])].book = res_down;
            }
        }
    }
    /* Discrete SSD streaming (ds4_gemma4_map.inc): the binder gathers every
     * VQ/RES record's own GVQ header with a device kernel
     * (ds4_glmvq_gpu.cuh gvq_headers), and a discrete card cannot read a
     * pageable host pointer from a kernel. The cold expert records therefore
     * go in header-sized and plain -- their 128 bytes are a real device range
     * published by ds4_gpu_g4_stream_plan_cold -- and the truthful record is
     * put back right after the bind. ds4_gpu_g4_stream_rebind re-validates
     * each of them from the pread staging before its first routed read, so
     * nothing is lost: the same rows/cols/book/res checks run there. */
    const bool discrete_stream = ds4_gpu_g4_stream_discrete_plan() != 0 &&
                                 (w->gemma4_stream != NULL || gemma4_cpu_moe_on());
    ds4_gvq_record *truth = NULL;
    uint32_t *cold = NULL;
    uint32_t cold_n = 0;
    if (discrete_stream) {
        truth = xmalloc((size_t)count * sizeof(*truth));
        memcpy(truth, records, (size_t)count * sizeof(*truth));
        cold = xmalloc((size_t)6u * G4_LAYERS * G4_EXPERTS * sizeof(*cold));
        cold_n = gemma4_stream_cold_records(w, m, cold);
        uint64_t *offsets = xmalloc((size_t)cold_n * sizeof(*offsets));
        uint64_t *sizes = xmalloc((size_t)cold_n * sizeof(*sizes));
        uint8_t *headers = xmalloc((size_t)cold_n * G4_STREAM_HEADER);
        bool ok = cold_n != 0 && m->map != NULL;
        for (uint32_t k = 0; ok && k < cold_n; k++) {
            const ds4_gvq_record *r = &records[cold[k]];
            if (r->bytes < G4_STREAM_HEADER || r->offset > m->size ||
                G4_STREAM_HEADER > m->size - r->offset) {
                ok = false;
                break;
            }
            memcpy(headers + (size_t)k * G4_STREAM_HEADER,
                   (const uint8_t *)m->map + r->offset, G4_STREAM_HEADER);
            offsets[k] = r->offset;
            sizes[k] = G4_STREAM_HEADER;
        }
        if (ok) {
            ok = ds4_gpu_g4_stream_plan_cold(m->map, m->size, offsets, sizes,
                                             cold_n, headers,
                                             G4_STREAM_HEADER) != 0;
        }
        free(headers);
        free(offsets);
        free(sizes);
        if (!ok) {
            fprintf(stderr, "ds4: Gemma-4 SSD cold header plan failed\n");
            free(cold);
            free(truth);
            free(records);
            return false;
        }
        for (uint32_t k = 0; k < cold_n; k++) {
            records[cold[k]].type = DS4_TENSOR_BF16;
            records[cold[k]].bytes = G4_STREAM_HEADER;
        }
    }
    w->glmvq_gpu = ds4_gpu_gvq_bind(m->map, m->size, records, count);
    free(records);
    if (w->glmvq_gpu && discrete_stream) {
        if (!ds4_gpu_g4_stream_plan_restore(w->glmvq_gpu, truth, cold, cold_n)) {
            fprintf(stderr, "ds4: Gemma-4 SSD stream plan restore failed\n");
            ds4_gpu_gvq_unbind(w->glmvq_gpu);
            w->glmvq_gpu = NULL;
        }
    }
    free(cold);
    free(truth);
    return w->glmvq_gpu != NULL;
}

static bool gemma4_rope_table(uint32_t pairs, uint32_t dim, double theta, float *inv) {
    for (uint32_t i = 0; i < dim / 2u; i++) {
        inv[i] = i < pairs ? (float)(1.0 / pow(theta, (double)(2u * i) / dim)) : 0.0f;
    }
    return true;
}

static bool gemma4_scalars(ds4_gemma4_graph *g, const ds4_model *m, const ds4_weights *w) {
    for (uint32_t il = 0; il < G4_LAYERS; il++) {
        uint16_t bits = 0;
        const ds4_tensor *t = w->gemma4[il].layer_scalar;
        if (!t || t->bytes != sizeof(bits) ||
            m->map == NULL || t->abs_offset + t->bytes > m->size) {
            return false;
        }
        memcpy(&bits, (const uint8_t *)m->map + t->abs_offset, sizeof(bits));
        float value;
        /* bf16 -> f32 keeps the stored bits, which is the multiply's operand. */
        const uint32_t wide = (uint32_t)bits << 16;
        memcpy(&value, &wide, sizeof(value));
        g->scalar[il] = value;
    }
    return true;
}

static bool gemma4_graph_reset(ds4_gemma4_graph *g);

static uint64_t gemma4_graph_bytes(uint32_t ctx, uint32_t cap) {
    const uint32_t rows = gemma4_token_rows(cap);
    /* The MoE workspace follows the session chunk. It used to be capped at
     * 512 rows, which silently made any wider chunk fail in the experts
     * (items_cap overflow) after the graph had already opened. */
    return gemma4_activation_bytes(cap) + ds4_gemma4_kv_bytes(ctx, cap) +
        ds4_gpu_gvq_workspace_linear(rows) + ds4_gpu_g4_moe_bytes(rows);
}

/* Frees what the graph owns. A borrowed bank graph owns only its KV rows;
 * the rest is graph 0's and outlives it. */
static void gemma4_graph_clear(ds4_gemma4_graph *g) {
    if (!g) { return; }
    for (uint32_t il = 0; il < G4_LAYERS; il++) {
        ds4_gpu_tensor_free(g->kcache[il]);
        ds4_gpu_tensor_free(g->vcache[il]);
        ds4_gpu_tensor_free(g->fkcache[il]);
    }
    if (g->borrowed) {
        memset(g, 0, sizeof(*g));
        return;
    }
    gemma4_mtp_runtime_free(g->mtp);
    g->mtp = NULL;
    ds4_gpu_g4_moe_free(g->moe);
    ds4_gpu_gvq_free(g->gvq);
    ds4_gpu_tensor_free(g->cur);
    ds4_gpu_tensor_free(g->norm);
    ds4_gpu_tensor_free(g->branch);
    ds4_gpu_tensor_free(g->tmp);
    ds4_gpu_tensor_free(g->exp_acc);
    ds4_gpu_tensor_free(g->q);
    ds4_gpu_tensor_free(g->k);
    ds4_gpu_tensor_free(g->v);
    ds4_gpu_tensor_free(g->attn);
    ds4_gpu_tensor_free(g->dense_g);
    ds4_gpu_tensor_free(g->dense_u);
    ds4_gpu_tensor_free(g->dense_mid);
    ds4_gpu_tensor_free(g->router_logits);
    ds4_gpu_tensor_free(g->selected);
    ds4_gpu_tensor_free(g->weights);
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->tokens);
    ds4_gpu_tensor_free(g->positions);
    ds4_gpu_tensor_free(g->inv_sliding);
    ds4_gpu_tensor_free(g->inv_full);
    memset(g, 0, sizeof(*g));
}

static void gemma4_graph_release(ds4_gemma4_graph *g) {
    if (!g) { return; }
    gemma4_graph_clear(g);
    free(g);
}

/* A bank graph: its own KV rows, everything else borrowed from graph 0. The
 * row count is the same function the allocator, the payload and the serving
 * quote use, so the ring is exactly as wide as the forward writes it. */
static bool gemma4_graph_bank(ds4_gemma4_graph *g, const ds4_gemma4_graph *owner) {
    if (!g || !owner || !owner->ready) { return false; }
    *g = *owner;
    g->borrowed = true;
    g->ready = false;
    g->failed = false;
    g->position = 0;
    g->kv_bytes = 0;
    /* The drafter belongs to graph 0: the serial lane drafts, the banked lane
     * does not, and a bank must not free the owner's runtime. */
    g->mtp = NULL;
    for (uint32_t il = 0; il < G4_LAYERS; il++) {
        g->kcache[il] = g->vcache[il] = NULL;
        g->fkcache[il] = NULL;
    }
    for (uint32_t il = 0; il < G4_LAYERS; il++) {
        const uint32_t rows = ds4_gemma4_kv_rows(il, g->ctx, g->cap);
        if (g->full_fp8 && gemma4_is_full(il)) {
            const uint64_t bytes = (uint64_t)rows * ds4_gemma4_full_kv_row();
            g->fkcache[il] = ds4_gpu_tensor_alloc(bytes);
            if (!g->fkcache[il]) {
                fprintf(stderr, "ds4: Gemma-4 bank FP8 KV allocation failed at layer %u\n", il);
                gemma4_graph_clear(g);
                return false;
            }
            g->kv_bytes += bytes;
            continue;
        }
        const uint64_t bytes = (uint64_t)rows * ds4_gemma4_kv_row(il);
        g->kcache[il] = ds4_gpu_tensor_alloc(bytes);
        g->vcache[il] = ds4_gpu_tensor_alloc(bytes);
        if (!g->kcache[il] || !g->vcache[il]) {
            fprintf(stderr, "ds4: Gemma-4 bank KV allocation failed at layer %u\n", il);
            gemma4_graph_clear(g);
            return false;
        }
        g->kv_bytes += 2u * bytes;
    }
    g->ready = true;
    return gemma4_graph_reset(g);
}

static ds4_gemma4_graph *gemma4_graph_create(ds4_model *m, ds4_weights *w,
        uint32_t ctx, uint32_t cap) {
    if (!w->gemma4 || !w->glmvq_gpu || !ctx || !cap || cap > ctx ||
        ctx > G4_CONTEXT || cap > G4_PREFILL_MAX) {
        return NULL;
    }
    ds4_gemma4_graph *g = xcalloc(1, sizeof(*g));
    g->ctx = ctx;
    g->cap = cap;
    g->full_fp8 = gemma4_full_kv_fp8();
    /* The activation rows cover one draft round's verify block as well as a
     * prefill chunk (gemma4_token_rows); the KV ring margin stays `cap`. */
    g->rows = gemma4_token_rows(cap);
    const uint32_t rows = g->rows;
    g->model = w->glmvq_gpu;
    g->stream = w->gemma4_stream;
    g->gvq = ds4_gpu_gvq_create_linear(g->model, ds4_gpu_gvq_workspace_linear(rows), rows);
    if (!g->gvq) {
        fprintf(stderr, "ds4: Gemma-4 GVQ context creation failed\n");
        gemma4_graph_release(g);
        return NULL;
    }
    const uint64_t act = (uint64_t)rows * G4_EMBED * sizeof(float);
    const uint64_t wide = (uint64_t)rows * G4_HEADS * G4_GLOBAL_HEAD_DIM * sizeof(float);
    const uint64_t narrow = (uint64_t)rows * G4_HEADS * G4_HEAD_DIM * sizeof(float);
    const uint64_t kvwidest = (uint64_t)rows * G4_KV_HEADS * G4_HEAD_DIM * sizeof(float);
#define G4_ALLOC(field, bytes) do { \
        g->field = ds4_gpu_tensor_alloc(bytes); \
        if (!g->field) { \
            fprintf(stderr, "ds4: Gemma-4 graph allocation failed: %s\n", #field); \
            gemma4_graph_release(g); \
            return NULL; \
        } \
    } while (0);
    G4_ALLOC(cur, act);
    G4_ALLOC(norm, act);
    G4_ALLOC(branch, act);
    G4_ALLOC(tmp, act);
    G4_ALLOC(exp_acc, act);
    G4_ALLOC(q, wide);
    G4_ALLOC(k, kvwidest);
    G4_ALLOC(v, kvwidest);
    G4_ALLOC(attn, wide);
    G4_ALLOC(dense_g, (uint64_t)rows * G4_DENSE * sizeof(float));
    G4_ALLOC(dense_u, (uint64_t)rows * G4_DENSE * sizeof(float));
    G4_ALLOC(dense_mid, (uint64_t)rows * G4_DENSE * sizeof(float));
    G4_ALLOC(router_logits, (uint64_t)rows * G4_EXPERTS * sizeof(float));
    G4_ALLOC(selected, (uint64_t)rows * G4_USED * sizeof(uint32_t));
    G4_ALLOC(weights, (uint64_t)rows * G4_USED * sizeof(float));
    G4_ALLOC(logits, (uint64_t)G4_VOCAB * sizeof(float));
    G4_ALLOC(tokens, (uint64_t)rows * sizeof(int32_t));
    G4_ALLOC(positions, (uint64_t)rows * sizeof(uint32_t));
    G4_ALLOC(inv_sliding, G4_HEAD_DIM / 2u * sizeof(float));
    G4_ALLOC(inv_full, G4_GLOBAL_HEAD_DIM / 2u * sizeof(float));
#undef G4_ALLOC
    (void)narrow;
    float inv[G4_GLOBAL_HEAD_DIM / 2u];
    if (!gemma4_rope_table(G4_ROT, G4_HEAD_DIM, 10000.0, inv) ||
        !ds4_gpu_tensor_write(g->inv_sliding, 0, inv, G4_HEAD_DIM / 2u * sizeof(float))) {
        gemma4_graph_release(g);
        return NULL;
    }
    if (!gemma4_rope_table(G4_GLOBAL_ROT, G4_GLOBAL_HEAD_DIM, 1000000.0, inv) ||
        !ds4_gpu_tensor_write(g->inv_full, 0, inv, G4_GLOBAL_HEAD_DIM / 2u * sizeof(float))) {
        gemma4_graph_release(g);
        return NULL;
    }
    for (uint32_t il = 0; il < G4_LAYERS; il++) {
        const uint32_t rows = ds4_gemma4_kv_rows(il, ctx, cap);
        if (g->full_fp8 && gemma4_is_full(il)) {
            const uint64_t bytes = (uint64_t)rows * ds4_gemma4_full_kv_row();
            g->fkcache[il] = ds4_gpu_tensor_alloc(bytes);
            if (!g->fkcache[il]) {
                fprintf(stderr, "ds4: Gemma-4 FP8 KV allocation failed at layer %u\n", il);
                gemma4_graph_release(g);
                return NULL;
            }
            g->kv_bytes += bytes;
            continue;
        }
        const uint64_t bytes = (uint64_t)rows * ds4_gemma4_kv_row(il);
        g->kcache[il] = ds4_gpu_tensor_alloc(bytes);
        g->vcache[il] = ds4_gpu_tensor_alloc(bytes);
        if (!g->kcache[il] || !g->vcache[il]) {
            fprintf(stderr, "ds4: Gemma-4 KV allocation failed at layer %u\n", il);
            gemma4_graph_release(g);
            return NULL;
        }
        g->kv_bytes += 2u * bytes;
    }
    g->moe = ds4_gpu_g4_moe_create(g->gvq, rows);
    if (!g->moe || !gemma4_scalars(g, m, w) || !gemma4_graph_reset(g)) {
        gemma4_graph_release(g);
        return NULL;
    }
    g->ready = true;
    if (getenv("DS4_GEMMA4_DEBUG")) {
        /* The norm weight the graph will read, straight from the map, so a bad
         * pointer or a bad type cannot masquerade as a kernel fault. This is
         * the check that pinned the P6 nan: the map held 1.70..30.6 and the
         * kernel was reading it as float32. */
        const ds4_tensor *t = w->gemma4[0].norm[0];
        if (t && m->map && t->abs_offset + t->bytes <= m->size && t->bytes >= 2u * G4_EMBED) {
            const uint16_t *bits = (const uint16_t *)((const uint8_t *)m->map + t->abs_offset);
            float low = INFINITY, high = -INFINITY, peak = 0.0f;
            uint32_t peak_at = 0, bad = 0;
            for (uint32_t d = 0; d < G4_EMBED; d++) {
                const uint32_t wide = (uint32_t)bits[d] << 16;
                float value;
                memcpy(&value, &wide, sizeof(value));
                if (!isfinite(value)) { bad++; continue; }
                if (value < low) { low = value; }
                if (value > high) { high = value; }
                if (fabsf(value) > fabsf(peak)) { peak = value; peak_at = d; }
            }
            fprintf(stderr, "ds4: G4 norm0 abs=%llu type=%u bytes=%llu "
                "range=[%.6g, %.6g] peak=%u:%.6g bad=%u\n",
                (unsigned long long)t->abs_offset, t->type, (unsigned long long)t->bytes,
                (double)low, (double)high, peak_at, (double)peak, bad);
        }
    }
    ds4_log(stderr, DS4_LOG_TIMING,
        "ds4: Gemma-4 graph ctx=%u chunk=%u KV=%.2f GiB activation=%.1f MiB\n",
        ctx, cap, (double)g->kv_bytes / 1073741824.0, (double)act * 6.0 / 1048576.0);
    ds4_log(stderr, DS4_LOG_TIMING, "ds4: Gemma-4 full-layer KV: %s\n",
        g->full_fp8 ? "E4M3 rows" : "bf16 rows");
    return g;
}

static bool gemma4_graph_reset(ds4_gemma4_graph *g) {
    g->position = 0;
    g->failed = false;
    /* The published hidden belongs to a forward this reset discards. */
    gemma4_mtp_drop_hidden(g);
    return true;
}

/* Development probe: with DS4_GEMMA4_DEBUG=1, print the leading values of a
 * stage so a numeric break names its stage. Silent in normal runs. */
static void gemma4_probe_at(const ds4_gpu_tensor *t, uint32_t il, const char *stage) {
    if (!getenv("DS4_GEMMA4_DEBUG") || !t) { return; }
    float host[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (!ds4_gpu_tensor_read(t, 0, host, sizeof(host))) {
        fprintf(stderr, "ds4: G4 L%u %s: readback failed\n", il, stage);
        return;
    }
    fprintf(stderr, "ds4: G4 L%u %s: %.6g %.6g %.6g %.6g\n", il, stage,
        (double)host[0], (double)host[1], (double)host[2], (double)host[3]);
}

/* Whole-buffer companion to gemma4_probe_at: counts non-finite values and
 * names the first, so a bad row cannot hide behind a good first float. */
static void gemma4_scan(const ds4_gpu_tensor *t, uint64_t count, uint32_t il, const char *stage) {
    if (!getenv("DS4_GEMMA4_DEBUG") || !t || !count || count > 268435456u) { return; }
    float *host = (float *)malloc(count * sizeof(float));
    if (!host) { return; }
    if (!ds4_gpu_tensor_read(t, 0, host, count * sizeof(float))) {
        fprintf(stderr, "ds4: G4 L%u %s: scan readback failed\n", il, stage);
        free(host);
        return;
    }
    uint64_t bad = 0, first = 0;
    uint64_t where[4] = {0, 0, 0, 0};
    float there[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float low = INFINITY, high = -INFINITY;
    for (uint64_t i = 0; i < count; i++) {
        const float v = host[i];
        if (!isfinite(v)) {
            if (!bad) { first = i; }
            if (bad < 4u) { where[bad] = i; there[bad] = v; }
            bad++;
            continue;
        }
        if (v < low) { low = v; }
        if (v > high) { high = v; }
    }
    fprintf(stderr, "ds4: G4 L%u %s: %llu/%llu non-finite first=%llu range=[%.6g, %.6g]\n",
        il, stage, (unsigned long long)bad, (unsigned long long)count,
        (unsigned long long)first, (double)low, (double)high);
    for (uint32_t i = 0; i < 4u; i++) {
        if (i >= bad) { break; }
        fprintf(stderr, "ds4: G4 L%u %s: bad[%u]=[%llu] %.6g\n", il, stage, i,
            (unsigned long long)where[i], (double)there[i]);
    }
    free(host);
}

/* Same probe at one row of a row-major buffer, so a multi-row launch can be
 * compared against the same token computed alone. */
static void gemma4_probe_row(const ds4_gpu_tensor *t, uint32_t il, const char *stage,
        uint32_t row, uint32_t width) {
    if (!getenv("DS4_GEMMA4_DEBUG") || !t || !width) { return; }
    float host[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (!ds4_gpu_tensor_read(t, (uint64_t)row * width * sizeof(float), host, sizeof(host))) {
        fprintf(stderr, "ds4: G4 L%u %s row %u: readback failed\n", il, stage, row);
        return;
    }
    fprintf(stderr, "ds4: G4 L%u %s row %u: %.6g %.6g %.6g %.6g\n", il, stage, row,
        (double)host[0], (double)host[1], (double)host[2], (double)host[3]);
}

/* Full-row fingerprint: a 32-bit FNV-1a over the row's bytes, so a whole row
 * can be compared between two launch shapes without printing 2816 values. */
static void gemma4_row_hash(const ds4_gpu_tensor *t, uint32_t il, const char *stage,
        uint32_t row, uint32_t width) {
    if (!getenv("DS4_GEMMA4_DEBUG") || !t || !width || width > 65536u) { return; }
    float *host = (float *)malloc((size_t)width * sizeof(float));
    if (!host) { return; }
    if (!ds4_gpu_tensor_read(t, (uint64_t)row * width * sizeof(float), host,
            (uint64_t)width * sizeof(float))) {
        fprintf(stderr, "ds4: G4 L%u %s row %u: hash readback failed\n", il, stage, row);
        free(host);
        return;
    }
    const uint8_t *bytes = (const uint8_t *)host;
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < (size_t)width * sizeof(float); i++) {
        hash ^= bytes[i];
        hash *= 16777619u;
    }
    fprintf(stderr, "ds4: G4 L%u %s row %u hash=%08x\n", il, stage, row, hash);
    free(host);
}

/* Stage hashes are on at layer 0. DS4_GEMMA4_HASH_LAYER=N adds the same
 * stage set at layer N (localizing a row-count divergence to one block), and
 * DS4_GEMMA4_HASH_LAYERS=all hashes every layer's block output (finding the
 * first layer a row count changes). Both are silent without the variable. */
static uint32_t gemma4_hash_layer(void) {
    const char *value = getenv("DS4_GEMMA4_HASH_LAYER");
    if (!value || !*value) { return G4_LAYERS; }
    char *end = NULL;
    const unsigned long parsed = strtoul(value, &end, 10);
    return end != value && !*end && parsed < G4_LAYERS ? (uint32_t)parsed : G4_LAYERS;
}

static bool gemma4_hash_layers(void) {
    const char *value = getenv("DS4_GEMMA4_HASH_LAYERS");
    return value && !strcmp(value, "all");
}

/* P6a kept a grouped prefill bit-identical to the one-row path by repeating
 * the one-row call per row. That makes prefill read every weight once per
 * token. The sibling VQ families batch instead (Ling-3.1's l31_mm_many,
 * A.X-K2's axk2_mm_many: one ds4_gpu_gvq_linear_many over all rows), so the
 * weight is restored once per chunk and the GEMM amortizes it. The grouped
 * Q4 kernel has its own FMA order from two rows up (cuda/glm_vq_q4m.cuh),
 * which is the price of the batch. DS4_GEMMA4_EXACT_PREFILL=1 restores the
 * one-row calls for the chunk-size exactness gate. */
static bool gemma4_exact_prefill(void) {
    return getenv("DS4_GEMMA4_EXACT_PREFILL") != NULL;
}

/* Gemma's own Q4_K decode (cuda/gemma4_q4_decode.cuh), off by default
 * (DS4_GEMMA4_Q4_DECODE=1). One token only; the shared dispatcher, and its
 * scalar fallback under 4 MiB, stays the reference. */
static bool g4_q4_decode(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_Q4_DECODE");
        on = v && v[0] == '1' && !v[1];
    }
    return on != 0;
}

/* Q4_K projections of one input: the batched prefill path, or the one-row
 * path the exactness gate measures. One token takes the Gemma Q4_K decode
 * when it is enabled and the matrices are all Q4_K; every other shape falls
 * through to the shared dispatcher. */
static bool gemma4_projection_rows(ds4_gemma4_graph *g, ds4_gpu_tensor *const *outs,
        const uint32_t *weights, const uint32_t *widths, uint32_t count,
        const ds4_gpu_tensor *in, uint32_t rows, uint32_t cols) {
    if (rows == 1u && g4_q4_decode() &&
        ds4_gpu_g4_q4_many(g->gvq, outs, weights, count, in, rows) != 0) {
        return true;
    }
    if (rows == 1 || !gemma4_exact_prefill()) {
        return ds4_gpu_gvq_linear_many(g->gvq, outs, weights, count, in, rows, DS4_GVQ_BF16) != 0;
    }
    for (uint32_t r = 0; r < rows; r++) {
        ds4_gpu_tensor *in_row = ds4_gpu_tensor_view(in, (uint64_t)r * cols * sizeof(float),
            (uint64_t)cols * sizeof(float));
        if (!in_row) { return false; }
        for (uint32_t c = 0; c < count; c++) {
            ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(outs[c],
                (uint64_t)r * widths[c] * sizeof(float), (uint64_t)widths[c] * sizeof(float));
            const bool ok = out_row && ds4_gpu_gvq_linear(g->gvq, out_row, weights[c], in_row,
                1u, DS4_GVQ_BF16) != 0;
            ds4_gpu_tensor_free(out_row);
            if (!ok) {
                ds4_gpu_tensor_free(in_row);
                return false;
            }
        }
        ds4_gpu_tensor_free(in_row);
    }
    return true;
}

/* Decode-only grouped-KV attention, off by default. */
static bool g4_attn_fast(void) {
    static int fast = -1;
    if (fast < 0) {
        const char *v = getenv("DS4_GEMMA4_ATTN_FAST");
        fast = v && v[0] == '1' && !v[1];
    }
    return fast != 0;
}

/* Prefill (rows > 1) grouped-KV attention with the shared K/V tile, off by
 * default. Separate from DS4_GEMMA4_ATTN_FAST so the two can be measured
 * apart; the oracle stays for both until they are folded together. */
static bool g4_attn_prefill(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_ATTN_PREFILL");
        on = v && v[0] == '1' && !v[1];
    }
    return on != 0;
}

/* Tensor-core prefill attention (cuda/gemma4_attn_mma.cuh), off by default
 * (DS4_GEMMA4_ATTN_MMA=1). It is the parity-gated path, not the bit-exact
 * one: the MTP verify widths stay on the tile kernel. */
static bool g4_attn_mma(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_ATTN_MMA");
        on = v && v[0] == '1' && !v[1];
    }
    return on != 0;
}

/* Split-K decode attention (cuda/gemma4_attn_split.cuh), off by default
 * (DS4_GEMMA4_ATTN_SPLIT=1). The per-key arithmetic is the fast kernel's;
 * the combine order differs. */
static bool g4_attn_split(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_ATTN_SPLIT");
        on = v && v[0] == '1' && !v[1];
    }
    return on != 0;
}

/* Fused RMSNorm + RoPE, off by default (DS4_GEMMA4_FUSE_ROPE=1). The
 * arithmetic is the two-kernel sequence's; the option exists to A/B it. */
static bool g4_fuse_rope(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_FUSE_ROPE");
        on = v && v[0] == '1' && !v[1];
    }
    return on != 0;
}

/* The family's own BF16 vocabulary head, off by default
 * (DS4_GEMMA4_HEAD_GEMV=1); the shared linear stays the fallback. */
static bool g4_head_gemv(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_HEAD_GEMV");
        on = v && v[0] == '1' && !v[1];
    }
    return on != 0;
}

/* The family's own router projection, off by default
 * (DS4_GEMMA4_ROUTER_DOT=1); the shared linear stays the fallback. */
static bool g4_router_dot(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_ROUTER_DOT");
        on = v && v[0] == '1' && !v[1];
    }
    return on != 0;
}

/* Dense down through the batch warp8 Q8_0 kernel, off by default
 * (DS4_GEMMA4_Q8_BATCH=1). */
static bool gemma4_q8_batch(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_Q8_BATCH");
        on = v && v[0] == '1' && !v[1];
    }
    return on != 0;
}

/* Persistent layer-book MoE decode (cuda/gemma4_moe_persist.cuh), off by
 * default (DS4_GEMMA4_MOE_PERSIST=1). One token only; the per-expert loop
 * stays the reference and the fallback. */
static bool g4_moe_persist(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_MOE_PERSIST");
        on = v && v[0] == '1' && !v[1];
    }
    return on != 0;
}

/* One layer, reference order (semantics 3.6). Returns false on the first
 * failed launch, naming the step: the wrappers report the CUDA error, this
 * names which stage of the block produced it. */
#define G4_STEP(call, step) do { \
    if (!(call)) { \
        fprintf(stderr, "ds4: Gemma-4 layer %u %s failed\n", il, step); \
        return false; \
    } \
} while (0)

/* Cache-miss fallback (DS4_GEMMA4_MOE_MISS=cpu): a route whose expert is
 * absent while the slot pool is full is computed on the host instead of
 * evicting a resident one. The GPU chunk sees a zero-weight stand-in for it
 * (a no-op in the ordered scatter), and the host contribution is added with
 * the same rounding right after the chunk. */
static bool gemma4_moe_miss_cpu(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_MOE_MISS");
        on = v && strcmp(v, "cpu") == 0;
    }
    return on != 0;
}

static bool gemma4_moe_run_mix(ds4_gemma4_graph *g, const ds4_model *m,
        const ds4_weights *w, uint32_t il, const uint32_t *experts) {
    static ds4_gpu_tensor *mix_ids, *mix_w, *mix_d;
    static float hx[G4_EMBED], hd[G4_EMBED];
    int32_t ids[G4_USED], gids[G4_USED];
    float ws[G4_USED], gws[G4_USED];
    uint32_t cpu[G4_USED], ncpu = 0;
    if (!g->stream ||
        !ds4_gpu_tensor_read(g->selected, 0, ids, sizeof(ids)) ||
        !ds4_gpu_tensor_read(g->weights, 0, ws, sizeof(ws))) {
        return false;
    }
    int32_t stand_in = -1;
    for (uint32_t i = 0; i < G4_USED; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= G4_EXPERTS) { return false; }
        if (gemma4_stream_slot_of(g->stream, il, (uint32_t)ids[i]) >= 0 ||
            gemma4_stream_has_free(g->stream)) {
            gids[i] = ids[i];
            gws[i] = ws[i];
            continue;
        }
        if (stand_in < 0) { stand_in = gemma4_stream_any_resident(g->stream, il); }
        if (stand_in < 0) {
            gids[i] = ids[i];
            gws[i] = ws[i];
            continue;
        }
        gids[i] = stand_in;
        gws[i] = 0.0f;
        cpu[ncpu++] = i;
    }
    if (!mix_ids) {
        mix_ids = ds4_gpu_tensor_alloc((uint64_t)G4_USED * sizeof(int32_t));
        mix_w = ds4_gpu_tensor_alloc((uint64_t)G4_USED * sizeof(float));
        mix_d = ds4_gpu_tensor_alloc((uint64_t)G4_EMBED * sizeof(float));
        if (!mix_ids || !mix_w || !mix_d) { return false; }
    }
    if (!ds4_gpu_tensor_write(mix_ids, 0, gids, sizeof(gids)) ||
        !ds4_gpu_tensor_write(mix_w, 0, gws, sizeof(gws))) {
        return false;
    }
    const uint32_t chunk = gemma4_stream_stage(g->stream, m, w, il, mix_ids, 0, 1);
    if (!chunk ||
        !ds4_gpu_g4_moe_run_ids(g->moe, g->exp_acc, g->norm, mix_w, experts,
                                g->stream->ids, chunk)) {
        return false;
    }
    if (!ncpu) { return true; }
    if (!ds4_gpu_tensor_read(g->norm, 0, hx, (uint64_t)G4_EMBED * sizeof(float))) {
        return false;
    }
    for (uint32_t k = 0; k < ncpu; k++) {
        const uint32_t i = cpu[k];
        if (!ds4_gpu_g4_cpu_moe_expert(g->gvq, experts, (uint32_t)ids[i], hx, hd) ||
            !ds4_gpu_tensor_write(mix_d, 0, hd, (uint64_t)G4_EMBED * sizeof(float)) ||
            !ds4_gpu_g4_moe_add_one(g->exp_acc, mix_d, ws[i], G4_EMBED)) {
            return false;
        }
    }
    return true;
}

/* The MoE over one layer's routes. Without the stream this is the single
 * grouped call. With it, the stage admits the largest row prefix whose
 * expert union fits the cache and the same grouped kernel runs over that
 * prefix with the host-side route ids the stage already read. */
static bool gemma4_moe_run(ds4_gemma4_graph *g, const ds4_model *m,
        const ds4_weights *w, uint32_t il, const uint32_t *experts, uint32_t rows) {
    if (!g->stream) {
        if (rows == 1u && g4_moe_persist() &&
            ds4_gpu_g4_moe_run_persist(g->moe, g->exp_acc, g->norm, g->selected,
                g->weights, experts, il) != 0) {
            return true;
        }
        return ds4_gpu_g4_moe_run(g->moe, g->exp_acc, g->norm, g->selected,
            g->weights, experts, rows) != 0;
    }
    if (rows == 1u && gemma4_moe_miss_cpu()) {
        return gemma4_moe_run_mix(g, m, w, il, experts);
    }
    for (uint32_t done = 0; done < rows;) {
        const uint32_t chunk = gemma4_stream_stage(g->stream, m, w, il,
            g->selected, done, rows - done);
        if (!chunk) { return false; }
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(g->exp_acc,
            (uint64_t)done * G4_EMBED * sizeof(float),
            (uint64_t)chunk * G4_EMBED * sizeof(float));
        ds4_gpu_tensor *in = ds4_gpu_tensor_view(g->norm,
            (uint64_t)done * G4_EMBED * sizeof(float),
            (uint64_t)chunk * G4_EMBED * sizeof(float));
        ds4_gpu_tensor *wt = ds4_gpu_tensor_view(g->weights,
            (uint64_t)done * G4_USED * sizeof(float),
            (uint64_t)chunk * G4_USED * sizeof(float));
        const bool ok = out && in && wt &&
            ds4_gpu_g4_moe_run_ids(g->moe, out, in, wt, experts,
                g->stream->ids, chunk) != 0;
        ds4_gpu_tensor_free(wt);
        ds4_gpu_tensor_free(in);
        ds4_gpu_tensor_free(out);
        if (!ok) { return false; }
        done += chunk;
    }
    return true;
}

static bool gemma4_layer(ds4_gemma4_graph *g, ds4_model *m, ds4_weights *w,
        uint32_t il, uint32_t rows, uint32_t first) {
    const ds4_gemma4_layer *l = &w->gemma4[il];
    const uint32_t hash_layer = gemma4_hash_layer();
    const bool hash_stages = il == 0 || il == hash_layer;
    const uint32_t dim = gemma4_head_dim(il);
    const uint32_t kv = gemma4_kv_heads(il);
    const bool full = gemma4_is_full(il);
    const uint32_t heads_width = G4_HEADS * dim;
    const uint32_t kv_width = kv * dim;
    const uint32_t ring = full ? 0u : ds4_gemma4_kv_rows(il, g->ctx, g->cap);
    const uint32_t capacity = ds4_gemma4_kv_rows(il, g->ctx, g->cap);
    const ds4_gpu_tensor *inv = full ? g->inv_full : g->inv_sliding;
    const uint32_t pairs = full ? G4_GLOBAL_ROT : G4_ROT;

    gemma4_probe_at(g->cur, il, "cur in");
    G4_STEP(ds4_gpu_g4_rms(g->gvq, g->norm, g->cur, gemma4_index(m, l->norm[0]),
        rows, G4_EMBED, G4_RMS_EPS), "input norm");
    gemma4_probe_at(g->norm, il, "norm in");
    if (hash_stages) { gemma4_scan(g->norm, (uint64_t)rows * G4_EMBED, il, "norm in scan"); }
    uint32_t proj[3] = {gemma4_index(m, l->q), gemma4_index(m, l->k),
                        gemma4_index(m, l->v)};
    ds4_gpu_tensor *outs[3] = {g->q, g->k, g->v};
    const uint32_t count = full ? 2u : 3u;
    const uint32_t widths[3] = {heads_width, kv_width, kv_width};
    if (!gemma4_projection_rows(g, outs, proj, widths, count, g->norm, rows, G4_EMBED)) {
        fprintf(stderr, "ds4: Gemma-4 q/k/v projection failed at layer %u\n", il);
        return false;
    }
    if (hash_stages) {
        gemma4_scan(g->q, (uint64_t)rows * heads_width, il, "q scan");
        gemma4_scan(g->cur, (uint64_t)rows * G4_EMBED, il, "cur in scan");
    }
    gemma4_probe_at(g->q, il, "q proj");
    gemma4_probe_at(g->k, il, "k proj");
    if (hash_stages) { for (uint32_t r = 0; r < rows; r++) { gemma4_row_hash(g->q, il, "q proj", r, heads_width); } }
    if (hash_stages && rows > 1) {
        gemma4_probe_row(g->q, il, "q proj", rows - 1u, heads_width);
        gemma4_probe_row(g->k, il, "k proj", rows - 1u, kv_width);
    }
    if (full) {
        G4_STEP(ds4_gpu_tensor_copy(g->v, 0, g->k, 0,
            (uint64_t)rows * kv_width * sizeof(float)), "attention_k_eq_v copy");
    }
    if (g4_fuse_rope()) {
        G4_STEP(ds4_gpu_g4_rms_rope(g->gvq, g->q, g->q, gemma4_index(m, l->attn_norm[0]),
            g->positions, inv, rows, G4_HEADS, dim, pairs, G4_RMS_EPS), "query norm + rope");
        G4_STEP(ds4_gpu_g4_rms_rope(g->gvq, g->k, g->k, gemma4_index(m, l->attn_norm[1]),
            g->positions, inv, rows, kv, dim, pairs, G4_RMS_EPS), "key norm + rope");
        G4_STEP(ds4_gpu_g4_rms(g->gvq, g->v, g->v, DS4_GVQ_NO_LINK,
            rows * kv, dim, G4_RMS_EPS), "value norm");
    } else {
        G4_STEP(ds4_gpu_g4_rms(g->gvq, g->q, g->q, gemma4_index(m, l->attn_norm[0]),
            rows * G4_HEADS, dim, G4_RMS_EPS), "query norm");
        G4_STEP(ds4_gpu_g4_rms(g->gvq, g->k, g->k, gemma4_index(m, l->attn_norm[1]),
            rows * kv, dim, G4_RMS_EPS), "key norm");
        G4_STEP(ds4_gpu_g4_rms(g->gvq, g->v, g->v, DS4_GVQ_NO_LINK,
            rows * kv, dim, G4_RMS_EPS), "value norm");
        G4_STEP(ds4_gpu_g4_rope(g->q, g->positions, inv, rows, G4_HEADS, dim, pairs),
            "query rope");
        G4_STEP(ds4_gpu_g4_rope(g->k, g->positions, inv, rows, kv, dim, pairs),
            "key rope");
    }
    gemma4_probe_at(g->q, il, "q normed roped");
    gemma4_probe_at(g->k, il, "k normed roped");
    gemma4_probe_at(g->v, il, "v normed");
    if (full && g->full_fp8) {
        G4_STEP(ds4_gpu_g4_kv_store_fp8(g->fkcache[il], g->k, g->v, g->positions, rows,
            kv_width, kv, dim, ring, capacity), "key/value FP8 cache store");
        G4_STEP(ds4_gpu_g4_attn_fp8(g->attn, g->q, g->fkcache[il], g->positions, rows,
            first, G4_HEADS, kv, dim, 0u, ring, 0u, ds4_gemma4_full_kv_row()), "attention");
    } else {
        G4_STEP(ds4_gpu_g4_kv_store(g->kcache[il], g->k, g->positions, rows, kv_width,
            ring, capacity), "key cache store");
        G4_STEP(ds4_gpu_g4_kv_store(g->vcache[il], g->v, g->positions, rows, kv_width,
            ring, capacity), "value cache store");
        if (rows == 1u && g4_attn_split()) {
            G4_STEP(ds4_gpu_g4_attn_split(g->attn, g->q, g->kcache[il], g->vcache[il],
                g->positions, rows, first, G4_HEADS, kv, dim, full ? 0u : G4_WINDOW,
                ring, 0u), "attention");
        } else if (rows == 1u && g4_attn_fast()) {
            G4_STEP(ds4_gpu_g4_attn_fast(g->attn, g->q, g->kcache[il], g->vcache[il],
                g->positions, rows, first, G4_HEADS, kv, dim, full ? 0u : G4_WINDOW,
                ring, 0u), "attention");
        } else if (rows >= 16u && g4_attn_mma()) {
            G4_STEP(ds4_gpu_g4_attn_mma(g->attn, g->q, g->kcache[il], g->vcache[il],
                g->positions, rows, first, G4_HEADS, kv, dim, full ? 0u : G4_WINDOW,
                ring, 0u), "attention");
            if (getenv("DS4_GEMMA4_ATTN_CMP") && (il == 0u || il == 5u)) {
                /* One-layer parity probe: run the tile kernel on the same Q/K/V
                 * and report the attention output's distance. The model gate
                 * (frontier logits) cannot say whether a drift is the kernel
                 * or its propagation through 30 layers. */
                const uint64_t bytes = (uint64_t)rows * heads_width * sizeof(float);
                ds4_gpu_tensor *cmp = ds4_gpu_tensor_alloc(bytes);
                if (cmp && ds4_gpu_g4_attn_tile(cmp, g->q, g->kcache[il], g->vcache[il],
                        g->positions, rows, first, G4_HEADS, kv, dim,
                        full ? 0u : G4_WINDOW, ring, 0u)) {
                    float *a = (float *)malloc((size_t)bytes);
                    float *b = (float *)malloc((size_t)bytes);
                    if (a && b && ds4_gpu_tensor_read(g->attn, 0, a, bytes) &&
                        ds4_gpu_tensor_read(cmp, 0, b, bytes)) {
                        double num = 0.0, den = 0.0, mx = 0.0;
                        for (uint64_t i = 0; i < (uint64_t)rows * heads_width; i++) {
                            const double d = (double)a[i] - (double)b[i];
                            num += d * d;
                            den += (double)b[i] * b[i];
                            if (fabs(d) > mx) { mx = fabs(d); }
                        }
                        fprintf(stderr, "ds4: G4 L%u attn mma-vs-tile relL2 %.6g max %.6g\n",
                            il, sqrt(num / (den > 0.0 ? den : 1.0)), mx);
                    }
                    free(a);
                    free(b);
                }
                if (cmp) { ds4_gpu_tensor_free(cmp); }
            }
        } else if (rows > 1u && g4_attn_prefill()) {
            G4_STEP(ds4_gpu_g4_attn_tile(g->attn, g->q, g->kcache[il], g->vcache[il],
                g->positions, rows, first, G4_HEADS, kv, dim, full ? 0u : G4_WINDOW,
                ring, 0u), "attention");
        } else {
            G4_STEP(ds4_gpu_g4_attn(g->attn, g->q, g->kcache[il], g->vcache[il], g->positions,
                rows, first, G4_HEADS, kv, dim, full ? 0u : G4_WINDOW, ring, 0u), "attention");
        }
    }
    gemma4_probe_at(g->attn, il, "attn out");
    if (hash_stages) { for (uint32_t r = 0; r < rows; r++) { gemma4_row_hash(g->attn, il, "attn out", r, heads_width); } }
    if (hash_stages && rows > 1) { gemma4_probe_row(g->attn, il, "attn out", rows - 1u, heads_width); }
    uint32_t o_proj[1] = {gemma4_index(m, l->o)};
    const uint32_t o_width[1] = {G4_EMBED};
    G4_STEP(gemma4_projection_rows(g, &g->branch, o_proj, o_width, 1u, g->attn, rows,
        heads_width), "o projection");
    G4_STEP(ds4_gpu_g4_rms(g->gvq, g->branch, g->branch, gemma4_index(m, l->norm[1]),
        rows, G4_EMBED, G4_RMS_EPS), "post-attention norm");
    G4_STEP(ds4_gpu_add_tensor(g->cur, g->cur, g->branch,
        (uint32_t)((uint64_t)rows * G4_EMBED)), "attention residual");
    G4_STEP(ds4_gpu_gvq_round(g->cur, (uint64_t)rows * G4_EMBED), "attention round");
    gemma4_probe_at(g->cur, il, "cur after attn");
    if (hash_stages) { for (uint32_t r = 0; r < rows; r++) { gemma4_row_hash(g->cur, il, "cur after attn", r, G4_EMBED); } }

    /* Dense branch: pre-FFN norm, GELU(product), down (Q8_0 from the map),
     * then post_feedforward_layernorm_1 (list index 5). */
    G4_STEP(ds4_gpu_g4_rms(g->gvq, g->norm, g->cur, gemma4_index(m, l->norm[2]),
        rows, G4_EMBED, G4_RMS_EPS), "pre-FFN norm");
    uint32_t dense[2] = {gemma4_index(m, l->dense_gate), gemma4_index(m, l->dense_up)};
    const uint32_t dense_width[2] = {G4_DENSE, G4_DENSE};
    ds4_gpu_tensor *dense_out[2] = {g->dense_g, g->dense_u};
    bool dense_ok;
    if (rows == 1u && g4_q4_decode() &&
        ds4_gpu_g4_q4_glu(g->gvq, g->dense_mid, dense[0], dense[1], g->norm, rows) != 0) {
        dense_ok = true;
    } else {
        dense_ok = gemma4_projection_rows(g, dense_out, dense, dense_width, 2u, g->norm, rows,
                G4_EMBED) &&
            ds4_gpu_g4_gelu(g->dense_mid, g->dense_g, g->dense_u, (uint64_t)rows * G4_DENSE);
    }
    if (dense_ok && rows > 1u && gemma4_q8_batch() && !gemma4_exact_prefill()) {
        /* Gemma's dense down is K = 8*256 + 64; the shared dispatcher's
         * tile path refuses it and falls to the per-element preq kernel.
         * The batch warp8 entry keeps every thread busy (see ds4_cuda.cu). */
        dense_ok = ds4_gpu_matmul_q8_0_dense_batch(g->branch, m->map, m->size,
            l->dense_down->abs_offset, G4_DENSE, G4_EMBED, g->dense_mid, rows) != 0;
    } else if (dense_ok && (rows == 1 || !gemma4_exact_prefill())) {
        /* The Q8_0 dense kernel takes the token count directly; one call
         * over the chunk restores the matrix once, like the Q4 projections
         * above. */
        dense_ok = ds4_gpu_matmul_q8_0_tensor(g->branch, m->map, m->size,
            l->dense_down->abs_offset, G4_DENSE, G4_EMBED, g->dense_mid, rows) != 0;
    } else if (dense_ok) {
        /* Exactness gate: the one-row call per row, as P6a measured. */
        for (uint32_t r = 0; r < rows; r++) {
            ds4_gpu_tensor *in_row = ds4_gpu_tensor_view(g->dense_mid,
                (uint64_t)r * G4_DENSE * sizeof(float), (uint64_t)G4_DENSE * sizeof(float));
            ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(g->branch,
                (uint64_t)r * G4_EMBED * sizeof(float), (uint64_t)G4_EMBED * sizeof(float));
            dense_ok = in_row && out_row && ds4_gpu_matmul_q8_0_tensor(out_row, m->map, m->size,
                l->dense_down->abs_offset, G4_DENSE, G4_EMBED, in_row, 1u) != 0;
            ds4_gpu_tensor_free(in_row);
            ds4_gpu_tensor_free(out_row);
            if (!dense_ok) { break; }
        }
    }
    if (!dense_ok ||
        !ds4_gpu_g4_rms(g->gvq, g->branch, g->branch, gemma4_index(m, l->norm[5]), rows, G4_EMBED, G4_RMS_EPS)) {
        fprintf(stderr, "ds4: Gemma-4 dense FFN failed at layer %u\n", il);
        return false;
    }
    gemma4_probe_at(g->dense_mid, il, "dense mid");
    gemma4_probe_at(g->branch, il, "dense down normed");
    if (hash_stages) { for (uint32_t r = 0; r < rows; r++) { gemma4_row_hash(g->branch, il, "dense down normed", r, G4_EMBED); } }

    /* Router reads the post-attention residual, not the FFN input. */
    uint32_t router[1] = {gemma4_index(m, l->router)};
    const uint32_t router_width[1] = {G4_EXPERTS};
    bool router_ok = ds4_gpu_g4_router_input(g->gvq, g->tmp, g->cur,
        gemma4_index(m, l->router_scale), rows, G4_EMBED, G4_RMS_EPS, G4_ROUTER_ROOT);
    if (router_ok) {
        const bool own_dot = g4_router_dot() && rows == 1u;
        router_ok = own_dot
            ? ds4_gpu_g4_router_logits(g->gvq, g->router_logits,
                gemma4_index(m, l->router), g->tmp, rows) != 0
            : gemma4_projection_rows(g, &g->router_logits, router, router_width,
                1u, g->tmp, rows, G4_EMBED);
        if (!router_ok && own_dot) {
            router_ok = gemma4_projection_rows(g, &g->router_logits, router,
                router_width, 1u, g->tmp, rows, G4_EMBED);
        }
    }
    if (!router_ok ||
        !ds4_gpu_g4_router(g->gvq, g->selected, g->weights, g->router_logits,
            gemma4_index(m, l->router_per_expert), rows)) {
        fprintf(stderr, "ds4: Gemma-4 router failed at layer %u\n", il);
        return false;
    }
    gemma4_probe_at(g->tmp, il, "router input");
    if (hash_stages) { for (uint32_t r = 0; r < rows; r++) { gemma4_row_hash(g->tmp, il, "router input", r, G4_EMBED); } }
    gemma4_probe_at(g->router_logits, il, "router logits");
    if (hash_stages && getenv("DS4_GEMMA4_DEBUG")) {
        /* The route is discontinuous: a 1-ULP input change can swap an expert,
         * so the selection is printed beside the logits. */
        uint32_t sel[G4_USED] = {0};
        float w[G4_USED] = {0.0f};
        const uint64_t at = (uint64_t)(rows - 1u) * G4_USED * sizeof(uint32_t);
        if (ds4_gpu_tensor_read(g->selected, at, sel, sizeof(sel)) &&
            ds4_gpu_tensor_read(g->weights, at, w, sizeof(w))) {
            fprintf(stderr, "ds4: G4 L%u route row %u:", il, rows - 1u);
            for (uint32_t i = 0; i < G4_USED; i++) {
                fprintf(stderr, " %u/%.4g", sel[i], (double)w[i]);
            }
            fprintf(stderr, "\n");
        }
    }
    uint32_t experts[3u * G4_EXPERTS];
    for (uint32_t e = 0; e < G4_EXPERTS; e++) {
        experts[e] = gemma4_index(m, l->gate[e]);
        experts[G4_EXPERTS + e] = gemma4_index(m, l->up[e]);
        experts[2u * G4_EXPERTS + e] = gemma4_index(m, l->down[e]);
    }
    if (!ds4_gpu_g4_rms(g->gvq, g->norm, g->cur, gemma4_index(m, l->norm[4]), rows, G4_EMBED, G4_RMS_EPS) ||
        !gemma4_moe_run(g, m, w, il, experts, rows)) {
        fprintf(stderr, "ds4: Gemma-4 experts failed at layer %u\n", il);
        return false;
    }
    gemma4_probe_at(g->exp_acc, il, "expert acc raw");
    if (getenv("DS4_GEMMA4_GPU_MOE_DUMP") && il == 0) {
        static int dumped;
        if (!dumped) {
            dumped = 1;
            float *h = (float *)malloc((size_t)G4_EMBED * sizeof(float));
            if (h && ds4_gpu_tensor_read(g->exp_acc, 0, h, (size_t)G4_EMBED * sizeof(float))) {
                FILE *fp = fopen("/tmp/g4-gpu-out.bin", "wb");
                if (fp) {
                    fwrite(h, sizeof(float), G4_EMBED, fp);
                    fclose(fp);
                }
            }
            free(h);
        }
    }
    if (!ds4_gpu_g4_rms(g->gvq, g->exp_acc, g->exp_acc, gemma4_index(m, l->norm[6]), rows, G4_EMBED, G4_RMS_EPS)) {
        fprintf(stderr, "ds4: Gemma-4 experts failed at layer %u\n", il);
        return false;
    }
    gemma4_probe_at(g->norm, il, "expert input");
    gemma4_probe_at(g->exp_acc, il, "expert acc normed");
    if (hash_stages) { for (uint32_t r = 0; r < rows; r++) { gemma4_row_hash(g->exp_acc, il, "expert acc normed", r, G4_EMBED); } }
    if (!ds4_gpu_add_tensor(g->branch, g->branch, g->exp_acc,
            (uint32_t)((uint64_t)rows * G4_EMBED)) ||
        !ds4_gpu_g4_rms(g->gvq, g->branch, g->branch, gemma4_index(m, l->norm[3]), rows, G4_EMBED, G4_RMS_EPS) ||
        !ds4_gpu_add_tensor(g->cur, g->cur, g->branch, (uint32_t)((uint64_t)rows * G4_EMBED)) ||
        !ds4_gpu_gvq_round(g->cur, (uint64_t)rows * G4_EMBED) ||
        !ds4_gpu_g4_layer_scale(g->cur, (uint64_t)rows * G4_EMBED, g->scalar[il])) {
        fprintf(stderr, "ds4: Gemma-4 block scaling failed at layer %u\n", il);
        return false;
    }
    (void)heads_width;
    return true;
}


/* tokens at absolute positions first..first+rows-1; logits for the last row. */
static bool gemma4_forward(ds4_gemma4_graph *g, ds4_model *m, ds4_weights *w,
        const uint32_t *tokens, uint32_t rows, uint32_t first, bool want_logits) {
    if (!g->ready || !rows || first + rows > g->ctx || rows > g->rows) { return false; }
    int32_t ids[G4_PREFILL_MAX];
    uint32_t pos[G4_PREFILL_MAX];
    if (rows > G4_PREFILL_MAX) { return false; }
    for (uint32_t r = 0; r < rows; r++) {
        ids[r] = (int32_t)tokens[r];
        pos[r] = first + r;
    }
    if (!ds4_gpu_tensor_write(g->tokens, 0, ids, (uint64_t)rows * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(g->positions, 0, pos, (uint64_t)rows * sizeof(uint32_t))) {
        fprintf(stderr, "ds4: Gemma-4 token/position upload failed at row %u\n", first);
        return false;
    }
    if (!ds4_gpu_embed_tokens_quant_tensor(g->cur, g->tokens, m->map, m->size,
            w->token_embd->abs_offset, w->token_embd->type, G4_VOCAB, rows, G4_EMBED)) {
        fprintf(stderr, "ds4: Gemma-4 embedding gather failed (type %d) at row %u\n",
            (int)w->token_embd->type, first);
        return false;
    }
    gemma4_probe_at(g->cur, G4_LAYERS, "embed");
    if (!ds4_gpu_g4_embed_scale(g->cur, (uint64_t)rows * G4_EMBED)) {
        fprintf(stderr, "ds4: Gemma-4 embedding scale failed at row %u\n", first);
        return false;
    }
    const bool hash_layers = gemma4_hash_layers();
    for (uint32_t il = 0; il < G4_LAYERS; il++) {
        if (!gemma4_layer(g, m, w, il, rows, first)) { return false; }
        gemma4_probe_at(g->cur, il, "layer out");
    if (il == 0 && rows > 1) { gemma4_probe_row(g->cur, il, "layer out", rows - 1u, G4_EMBED); }
    if (il == 0) { for (uint32_t r = 0; r < rows; r++) { gemma4_row_hash(g->cur, il, "layer out", r, G4_EMBED); } }
    if (hash_layers && il) { for (uint32_t r = 0; r < rows; r++) { gemma4_row_hash(g->cur, il, "layer out", r, G4_EMBED); } }
    }
    if (!ds4_gpu_g4_rms(g->gvq, g->norm, g->cur, gemma4_index(m, w->output_norm),
            rows, G4_EMBED, G4_RMS_EPS)) {
        fprintf(stderr, "ds4: Gemma-4 final norm failed\n");
        return false;
    }
    gemma4_probe_at(g->norm, G4_LAYERS, "final norm");
    g->position = first + rows;
    /* The drafter's hidden half is this forward's last final-norm row
     * (candidate_generator.py:1321-1324, and modeling_gemma4_assistant.py:176
     * consumes it as `last_hidden_state`); only the last row is ever needed. */
    if (g->mtp && !gemma4_mtp_publish(g, rows - 1u)) { return false; }
    if (!want_logits) { return true; }
    ds4_gpu_tensor *last = ds4_gpu_tensor_view(g->norm,
        (uint64_t)(rows - 1u) * G4_EMBED * sizeof(float), (uint64_t)G4_EMBED * sizeof(float));
    if (!last) {
        fprintf(stderr, "ds4: Gemma-4 logits view failed\n");
        return false;
    }
    bool ok = false;
    if (g4_head_gemv()) {
        ok = ds4_gpu_g4_head(g->gvq, g->logits, gemma4_index(m, w->token_embd),
            last, G4_SOFTCAP) != 0;
    }
    if (!ok) {
        ok = ds4_gpu_gvq_linear(g->gvq, g->logits,
            gemma4_index(m, w->token_embd), last, 1u, DS4_GVQ_BF16) &&
            ds4_gpu_g4_softcap(g->logits, G4_VOCAB);
    }
    ds4_gpu_tensor_free(last);
    if (!ok) { fprintf(stderr, "ds4: Gemma-4 vocabulary head failed\n"); }
    gemma4_probe_at(g->logits, G4_LAYERS, "logits");
    return ok;
}
