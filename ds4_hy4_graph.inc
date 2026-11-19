/* All tensor weights are borrowed from the Rust-validated host inventory.
 * Per-bank state consists of MLA keys, complete pooled groups and the tail
 * of the unpooled index. Engram history is the bank's accepted token prefix. */
typedef struct {
    uint32_t context, cap, position, high, restored;
    bool ready, failed, borrowed, prefill_queued;
    ds4_hy4_span media[4];
    uint32_t media_count;
    uint64_t media_tag;
    ds4_hy4_cache layer[HY4_LAYERS];
#define HY4_FIELD(field, cols, type) ds4_gpu_tensor *field;
    HY4_BUFFERS(HY4_FIELD)
#undef HY4_FIELD
    ds4_gpu_tensor *logits, *seed;
    float *verify_logits;
    uint32_t trial_n;
    int trial[HY4_DRAFT_BLOCK];
    bool verifying;
    uint8_t *host_records;
    uint16_t *host_embed;
    int32_t *host_positions;
} ds4_hy4_graph;

static void hy4_graph_free(ds4_hy4_graph *g) {
    for (unsigned i = 0; i < HY4_LAYERS; i++) {
        ds4_hy4_cache *c = &g->layer[i];
        ds4_gpu_tensor_free(c->latent);
        ds4_gpu_tensor_free(c->rope);
        ds4_gpu_tensor_free(c->groups);
        ds4_gpu_tensor_free(c->recent);
    }
    ds4_gpu_tensor_free(g->seed);
    free(g->verify_logits);
    if (!g->borrowed) {
#define HY4_FREE(field, cols, type) ds4_gpu_tensor_free(g->field);
        HY4_BUFFERS(HY4_FREE)
#undef HY4_FREE
        ds4_gpu_tensor_free(g->logits);
        free(g->host_records);
        free(g->host_embed);
        free(g->host_positions);
    }
    memset(g, 0, sizeof(*g));
}

static void hy4_reset(ds4_hy4_graph *g) {
    g->failed = g->failed && !ds4_gpu_synchronize();
    g->position = g->high = g->restored = g->trial_n = 0;
    g->verifying = false;
    g->prefill_queued = false;
}

#ifndef __APPLE__
static bool hy4_graph_alloc(ds4_hy4_graph *g, uint32_t ctx, uint32_t cap, ds4_hy4_graph *owner) {
    if (!hy4_memory(ctx, cap).total_bytes) {
        return false;
    }
    g->context = ctx;
    g->cap = cap;
    const uint64_t seeds = (1 + HY4_DRAFT_BLOCK) * HY4_SEED_BYTES;
    g->seed = ds4_gpu_tensor_alloc(seeds);
    if (!g->seed || !ds4_gpu_tensor_fill_f32(g->seed, 0.f, seeds / 4)) {
        return false;
    }
    if (owner) {
        if (!owner->ready || owner->cap != cap) {
            return false;
        }
        g->borrowed = true;
#define HY4_BORROW(field, cols, type) g->field = owner->field;
        HY4_BUFFERS(HY4_BORROW)
#undef HY4_BORROW
        g->logits = owner->logits;
        g->host_records = owner->host_records;
        g->host_embed = owner->host_embed;
        g->host_positions = owner->host_positions;
    } else {
#define HY4_ALLOC(field, cols, type)                                                                       \
    if (!(g->field = ds4_gpu_tensor_alloc((uint64_t)cap * (cols) * sizeof(type))))                         \
        return false;
        HY4_BUFFERS(HY4_ALLOC)
#undef HY4_ALLOC
        g->logits = ds4_gpu_tensor_alloc(120832 * sizeof(float));
        if (!g->logits) {
            return false;
        }
        g->host_records = malloc((size_t)cap * 4 * HY4_ENGRAM_BYTES);
        g->host_embed = malloc((size_t)cap * 3072 * 2);
        g->host_positions = malloc((size_t)cap * 3 * sizeof(int32_t));
        if (!g->host_records || !g->host_embed || !g->host_positions) {
            return false;
        }
    }
    for (uint32_t i = 0; i < HY4_LAYERS; i++) {
        ds4_hy4_cache *c = &g->layer[i];
        bool sparse = hy4_sparse(i);
        c->cap = sparse ? ctx : hy4_ring(ctx, cap);
        c->fp8 = sparse && hy4_fp8();
        c->latent = ds4_gpu_tensor_alloc((uint64_t)c->cap * (c->fp8 ? 513 : 1024));
        c->rope = ds4_gpu_tensor_alloc((uint64_t)c->cap * 128);
        if (!c->latent || !c->rope) {
            return false;
        }
        if (sparse) {
            c->recent_cap = hy4_recent(ctx, cap);
            c->groups = ds4_gpu_tensor_alloc((((uint64_t)ctx + 3) / 4) * 256);
            c->recent = ds4_gpu_tensor_alloc((uint64_t)c->recent_cap * 259 * sizeof(float));
            if (!c->groups || !c->recent) {
                return false;
            }
        }
    }
    g->ready = true;
    return true;
}

// Verify widths of a BF16 projection; -1 leaves other calls to the caller.
static int hy4_rows(ds4_gpu_tensor *out, const ds4_model *m, const ds4_tensor *w, const ds4_gpu_tensor *x,
                    ds4_hy4_rows input, uint32_t n) {
    if (w->type != DS4_TENSOR_BF16) {
        return -1;
    }
    return ds4_gpu_hy4_rows_bf16(out, m->map, m->size, w->abs_offset, (uint32_t)w->dim[0], (uint32_t)w->dim[1],
                                 x, input, n);
}
static bool hy4_mm(ds4_gpu_tensor *out, const ds4_model *m, const ds4_tensor *w, const ds4_gpu_tensor *x,
                   uint32_t n) {
    const int rows = hy4_rows(out, m, w, x, DS4_HY4_ROWS_F32, n);
    if (rows >= 0) {
        return rows != 0;
    }
    return ling3vl_matmul_tensor(out, m, w, w->dim[0], w->dim[1], x, n);
}
static const float *hy4_weight(const ds4_model *m, const ds4_tensor *w) {
    return ds4_gpu_hy4_weight(m->map, m->size, w->abs_offset, w->bytes);
}
static bool hy4_norm(ds4_gpu_tensor *out, const ds4_model *m, const ds4_tensor *w, const ds4_gpu_tensor *x,
                     uint32_t n) {
    return ds4_gpu_motif3_rms_norm_round_bf16_rows_tensor(out, x, m->map, m->size, w->abs_offset,
                                                          (uint32_t)w->dim[0], n, 1e-5f);
}
// Multi-row norms keep a BF16 copy for their BF16 GEMMs; the values are
// BF16-rounded already, so the copy is exact and the GEMMs are unchanged.
static bool hy4_pre_norm(ds4_hy4_graph *g, const ds4_model *m, const ds4_tensor *w, uint32_t n) {
    return hy4_norm(g->norm, m, w, g->input, n) &&
           (n == 1 || ds4_gpu_f32_to_bf16(g->norm_bf16, g->norm, (uint64_t)n * 3072));
}
static bool hy4_hc_pre(ds4_hy4_graph *g, const ds4_model *m, const ds4_tensor *fn, const ds4_tensor *scale,
                       const ds4_tensor *base, const ds4_tensor *norm, uint32_t n) {
    return hy4_mm(g->mix, m, fn, g->streams, n) &&
           ds4_gpu_hy4_hc(g->input, g->post, g->streams, g->mix, hy4_weight(m, scale), hy4_weight(m, base),
                          n) &&
           hy4_pre_norm(g, m, norm, n);
}
/* Whether a branch's input step first applies the preceding branch. */
typedef enum { HY4_FRESH, HY4_CARRY } hy4_carry;
typedef enum { HY4_HC_SPLIT, HY4_HC_FUSED } hy4_hc_mode;
// Residual, iHC and RMSNorm for the next branch: one pass (HY4_HC_FUSED) or
// the residual, TF32 mix GEMM, combine and norm launches (DS4_HY4_HC_FUSED=0).
static bool hy4_branch_in(ds4_hy4_graph *g, const ds4_model *m, const ds4_tensor *fn, const ds4_tensor *scale,
                          const ds4_tensor *base, const ds4_tensor *norm, uint32_t n, hy4_carry carry,
                          hy4_hc_mode mode) {
    if (mode == HY4_HC_FUSED) {
        return ds4_gpu_hy4_hc_step(g->streams, g->post, carry == HY4_CARRY ? g->branch : NULL, hy4_weight(m, fn),
                                   hy4_weight(m, scale), hy4_weight(m, base), hy4_weight(m, norm), g->norm,
                                   n > 1 ? g->norm_bf16 : NULL, n);
    }
    return (carry == HY4_FRESH || ds4_gpu_hy4_residual(g->streams, g->branch, g->post, n)) &&
           hy4_hc_pre(g, m, fn, scale, base, norm, n);
}
static bool hy4_mm_norm(ds4_hy4_graph *g, const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *w, uint32_t n) {
    if (n == 1 || w->type != DS4_TENSOR_BF16) {
        return hy4_mm(out, m, w, g->norm, n);
    }
    const int rows = hy4_rows(out, m, w, g->norm_bf16, DS4_HY4_ROWS_BF16, n);
    if (rows >= 0) {
        return rows != 0;
    }
    return ds4_gpu_matmul_bf16_input_tensor(out, m->map, m->size, w->abs_offset, w->dim[0], w->dim[1],
                                            g->norm_bf16, n) != 0;
}
static bool hy4_run_index(ds4_hy4_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                          ds4_hy4_cache *c, uint32_t n, uint32_t pos) {
    if (!c->groups) {
        return true;
    }
    if (!hy4_mm_norm(g, m, g->index_key, l->indexer_attn_k, n) ||
        !hy4_norm(g->index_key, m, l->indexer_k_norm, g->index_key, n) ||
        !hy4_mm(g->index_v, m, l->indexer_compressor_kv, g->index_key, n) ||
        !hy4_mm(g->index_gate, m, l->indexer_compressor_gate, g->index_key, n) ||
        !ds4_gpu_hy4_pool(c, g->index_v, g->index_gate, g->positions,
                          hy4_weight(m, l->indexer_compressor_ape),
                          hy4_weight(m, l->indexer_compressor_norm), pos, n)) {
        return false;
    }
    return pos + n <= 2048 ||
           (hy4_mm(g->index_q, m, l->indexer_attn_q_b, g->qa, n) &&
            ds4_gpu_hy4_rope(g->index_q, g->positions, n, 16, 128, 64, 1e7f) &&
            hy4_mm(g->index_w, m, l->indexer_proj, g->norm, n) &&
            ds4_gpu_hy4_index(g->groups, g->tile, g->best, g->index_q, g->index_w, c, pos, n));
}

static bool hy4_attention(ds4_hy4_graph *g, const ds4_model *m, const ds4_layer_weights *l, uint32_t il,
                          uint32_t n, uint32_t pos, const ds4_gpu_tensor *draft_groups) {
    const bool sparse = hy4_sparse(il);
    const uint32_t h = sparse ? 32 : 64;
    const float theta = sparse ? 1e7f : 1e4f;
    ds4_hy4_cache *c = &g->layer[il];
    // The tiled kernel rounds q_abs to BF16 as it loads it.
    const ds4_hy4_head_output q_abs = ds4_gpu_hy4_flash(n, draft_groups != NULL) ? DS4_HY4_HEAD_OUT_BF16
                                                                                  : DS4_HY4_HEAD_OUT_F32;
    if (!hy4_mm_norm(g, m, g->qa, l->attn_q_a, n) || !hy4_norm(g->qa, m, l->attn_q_a_norm, g->qa, n) ||
        !hy4_mm(g->q, m, l->attn_q_b, g->qa, n) || !hy4_mm_norm(g, m, g->kv, l->attn_kv_a_mqa, n) ||
        !ds4_gpu_hy4_rope(g->q, g->positions, n, h, 256, 192, theta) ||
        !ds4_gpu_hy4_rope(g->kv, g->positions, n, 1, 576, 512, theta) ||
        !ds4_gpu_hy4_store(c, g->kv, hy4_weight(m, l->attn_kv_a_norm), pos, n) ||
        (sparse && !draft_groups && !hy4_run_index(g, m, l, c, n, pos)) ||
        !ds4_gpu_hy4_head_mm(g->absorbed, g->packed, g->q, m->map, m->size, l->attn_k_b->abs_offset, n, h, 192,
                             256, 512, DS4_HY4_HEAD_F32, q_abs)) {
        return false;
    }
    // Prefill attention leaves the v_b operand packed as BF16.
    const int attended = ds4_gpu_hy4_attention(g->attended, g->packed, g->absorbed, g->q, c,
                                               draft_groups ? draft_groups : g->groups,
                                               hy4_weight(m, l->attn_sinks), pos, n, h, draft_groups != NULL);
    return attended >= 0 &&
           ds4_gpu_hy4_head_mm(g->q, g->packed, g->attended, m->map, m->size, l->attn_v_b->abs_offset, n, h,
                               512, 512, 256, (ds4_hy4_head_input)attended, DS4_HY4_HEAD_OUT_F32) &&
           hy4_mm(g->attn_gate, m, l->attn_gate, g->norm, n) &&
           ds4_gpu_hy4_gate(g->q, g->attn_gate, n, h * 256) &&
           hy4_mm(g->branch, m, l->attn_output, g->q, n) &&
           hy4_norm(g->branch, m, l->attn_output_norm, g->branch, n);
}
static bool hy4_dense(ds4_hy4_graph *g, const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *gate,
                      const ds4_tensor *up, const ds4_tensor *down, uint32_t n) {
    return hy4_mm_norm(g, m, g->ff_gate, gate, n) && hy4_mm_norm(g, m, g->ff_up, up, n) &&
           ds4_gpu_hy4_swiglu(g->ff_mid, g->ff_gate, g->ff_up, n * (uint32_t)gate->dim[1], INFINITY) &&
           hy4_mm(out, m, down, g->ff_mid, n);
}
/* Prefill and verify quantize the clamped SwiGLU straight into the down's
 * Q8 operand; one-row decode keeps the F32 mid (DS4_HY4_SWIGLU_Q8=0: always). */
static bool hy4_routed_down(ds4_hy4_graph *g, const ds4_model *m, const ds4_tensor *down, uint32_t n) {
    const int fused = ds4_gpu_hy4_down(g->ff_gate, g->ff_gate, g->ff_up, g->experts, m->map, m->size,
                                       down->abs_offset, down->bytes, down->type, n);
    if (fused >= 0) {
        return fused != 0;
    }
    return ds4_gpu_hy4_swiglu(g->ff_mid, g->ff_gate, g->ff_up, n * 16 * 1536, 10.f) &&
           ds4_gpu_routed_matmul_guarded_tensor(g->ff_gate, g->ff_mid, g->experts, m->map, m->size,
                                                down->abs_offset, down->bytes, down->type, 1536, 1536, 512,
                                                n * 16, 1, n);
}
/* Prefill and verify rows run the experts on FP16 tensor cores; one decode
 * row (and DS4_HY4_FP16_MOE=0) takes MMQ. The down rows land in ff_gate and
 * their weighted sum in latent. */
static bool hy4_routed(ds4_hy4_graph *g, const ds4_model *m, const ds4_tensor *gate, const ds4_tensor *up,
                       const ds4_tensor *down, uint32_t n) {
    const ds4_hy4_experts w = {{gate->abs_offset, up->abs_offset, down->abs_offset},
                               {gate->bytes, up->bytes, down->bytes},
                               {gate->type, up->type, down->type}};
    const int fp16 = ds4_gpu_hy4_experts(g->ff_gate, g->ff_up, g->ff_mid, g->latent, g->experts, m->map, m->size,
                                         &w, n);
    if (fp16 >= 0) {
        return fp16 != 0 && ds4_gpu_hy4_moe_sum(g->latent, g->ff_gate, g->weights, n, (ds4_hy4_rows)fp16);
    }
    return ds4_gpu_routed_gate_up_tensor(g->ff_gate, g->ff_up, g->latent, g->experts, m->map, m->size,
                                         gate->abs_offset, gate->bytes, up->abs_offset, up->bytes, gate->type,
                                         1536, 1536, 512, n, 16) &&
           hy4_routed_down(g, m, down, n) &&
           ds4_gpu_hy4_moe_sum(g->latent, g->ff_gate, g->weights, n, DS4_HY4_ROWS_F32);
}
static bool hy4_ffn(ds4_hy4_graph *g, const ds4_model *m, const ds4_layer_weights *l, uint32_t n) {
    if (!l->ffn_gate_exps) {
        return hy4_dense(g, m, g->branch, l->ffn_gate, l->ffn_up, l->ffn_down, n) &&
               hy4_norm(g->branch, m, l->ffn_output_norm, g->branch, n);
    }
    const ds4_tensor *gate = l->ffn_gate_exps, *up = l->ffn_up_exps, *down = l->ffn_down_exps;
    return hy4_mm(g->router, m, l->ffn_gate_inp, g->norm, n) &&
           ds4_gpu_sigmoid_topk_router_tensor(g->experts, g->weights, g->router, m->map, m->size,
                                              l->ffn_exp_probs_b->abs_offset, 1, 512, 16, n, 3.998f) &&
           hy4_mm_norm(g, m, g->latent, l->ffn_routed_down, n) &&
           hy4_routed(g, m, gate, up, down, n) &&
           hy4_mm(g->branch, m, l->ffn_routed_up, g->latent, n) &&
           hy4_dense(g, m, g->shared, l->ffn_gate_shexp, l->ffn_up_shexp, l->ffn_down_shexp, n) &&
           ds4_gpu_exaone_add_tensor(g->branch, g->shared, (uint64_t)n * 3072) &&
           hy4_norm(g->branch, m, l->ffn_output_norm, g->branch, n);
}
static bool hy4_memory_branch(ds4_hy4_graph *g, const ds4_model *m, const ds4_layer_weights *l, uint32_t il,
                              uint32_t n) {
    return ds4_gpu_hy4_unpack(g->latent, g->records, hy4_weight(m, l->engram_lookup_norm), il / 10, n) &&
           hy4_mm(g->ff_gate, m, l->engram_wkv, g->latent, n) &&
           ds4_gpu_hy4_engram(g->streams, g->ff_gate, g->ids, hy4_weight(m, l->engram_kv_bias),
                              hy4_weight(m, l->engram_k), hy4_weight(m, l->engram_q),
                              hy4_weight(m, l->engram_pre_gamma), hy4_weight(m, l->engram_hc_scale), n);
}

// Prompt rows and rotary positions diverge after a visual span. The geometry
// is kept for decoding; pixel projections are borrowed only during the sync.
static bool hy4_input_row(ds4_hy4_graph *g, int token, uint32_t row, const uint8_t *embedding,
                          uint16_t *out, int32_t *coords) {
    uint32_t cursor = row;
    for (uint32_t i = 0; i < g->media_count; i++) {
        const ds4_hy4_span *s = &g->media[i];
        const uint32_t cols = s->width + 1, step = s->height > cols ? s->height : cols;
        const uint32_t per = s->height * cols, count = per * s->frames;
        if (row < s->start) {
            break;
        }
        if (row - s->start < count) {
            const uint32_t idx = row - s->start, frame = idx / per, cell = idx % per,
                           base = cursor - idx + frame * step;
            if (s->token != token || !s->features) {
                return false;
            }
            coords[0] = (int32_t)base;
            coords[1] = (int32_t)(base + cell / cols);
            coords[2] = (int32_t)(base + cell % cols);
            for (unsigned j = 0; j < 3072; j++) {
                uint32_t b;
                memcpy(&b, s->features + (uint64_t)idx * 3072 + j, 4);
                out[j] = (uint16_t)(b >> 16);
            }
            return true;
        }
        cursor -= count - s->frames * step;
    }
    coords[0] = coords[1] = coords[2] = (int32_t)cursor;
    memcpy(out, embedding + (uint64_t)token * 6144, 6144);
    return true;
}

static bool hy4_head_row(ds4_hy4_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t row,
                         uint32_t seedslot) {
    const uint64_t bytes = 12288 * sizeof(float);
    ds4_gpu_tensor *hc = ds4_gpu_tensor_view(g->streams, (uint64_t)row * bytes, bytes);
    const bool ok = hc && hy4_mm(g->mix, m, w->output_hc_fn, hc, 1) &&
                    ds4_gpu_hy4_hc(g->input, NULL, hc, g->mix, hy4_weight(m, w->output_hc_scale),
                                   hy4_weight(m, w->output_hc_base), 1) &&
                    hy4_norm(g->norm, m, w->output_norm, g->input, 1) &&
                    ds4_gpu_tensor_copy(g->seed, seedslot * HY4_SEED_BYTES, g->norm, 0, 3072 * 4) &&
                    hy4_mm(g->logits, m, w->output, g->norm, 1);
    ds4_gpu_tensor_free(hc);
    return ok;
}
static bool hy4_verify_head(ds4_hy4_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t n) {
    const char *env = getenv("DS4_HY4_BATCH_HEAD");
    const uint64_t bytes = (uint64_t)n * 120832 * sizeof(float);
    if ((env && strcmp(env, "0") == 0) || ds4_gpu_tensor_bytes(g->ff_gate) < bytes) {
        for (uint32_t r = 0; r < n; r++) {
            if (!hy4_head_row(g, m, w, r, r + 1) ||
                !ds4_gpu_tensor_read(g->logits, 0, g->verify_logits + (size_t)r * 120832, 120832 * 4)) {
                return false;
            }
        }
        return true;
    }
    if (!hy4_mm(g->mix, m, w->output_hc_fn, g->streams, n) ||
        !ds4_gpu_hy4_hc(g->input, NULL, g->streams, g->mix, hy4_weight(m, w->output_hc_scale),
                        hy4_weight(m, w->output_hc_base), n) ||
        !hy4_norm(g->norm, m, w->output_norm, g->input, n) ||
        !hy4_mm(g->ff_gate, m, w->output, g->norm, n)) {
        return false;
    }
    for (uint32_t r = 0; r < n; r++) {
        if (!ds4_gpu_tensor_copy(g->seed, (r + 1) * HY4_SEED_BYTES, g->norm, (uint64_t)r * 3072 * 4, 3072 * 4)) {
            return false;
        }
    }
    return ds4_gpu_tensor_read(g->ff_gate, 0, g->verify_logits, bytes);
}
static bool hy4_capture_groups(ds4_hy4_graph *g, uint32_t il, uint32_t n, uint32_t pos) {
    if (il < 25 || (il - 25) % 4 || pos + n <= 2048) {
        return true;
    }
    for (uint32_t r = g->verifying ? 0 : n - 1; r < n; r++) {
        if (pos + r < 2048) {
            continue;
        }
        const uint32_t slot = g->verifying ? r + 1 : 0;
        const uint32_t off = slot * HY4_SEED_BYTES + 3072 * 4 + (il - 25) / 4 * 511 * 4;
        if (!ds4_gpu_tensor_copy(g->seed, off, g->groups, (uint64_t)r * 512 * 4, 511 * 4)) {
            return false;
        }
    }
    return true;
}
static bool hy4_read_engram(ds4_hy4_graph *g, const ds4_engine *e, const int *prev, unsigned np,
                            const int *tokens, uint32_t n, uint32_t first, uint32_t layers) {
    const size_t stride = (size_t)n * HY4_ENGRAM_BYTES, bytes = layers * stride;
    return e->hy4.rows(e->hy4.user, prev, np, tokens, n, first, layers, g->host_records, bytes) &&
           ds4_gpu_tensor_write(g->records, first * stride, g->host_records, bytes);
}
static bool hy4_forward(ds4_hy4_graph *g, const ds4_engine *e, const int *prev, unsigned np,
                        const int *tokens, uint32_t n, uint32_t pos, bool head) {
    if (!g->ready || g->failed || !n || n > g->cap || pos != g->position || n > g->context - pos ||
        !e->hy4.rows) {
        return false;
    }
    const ds4_model *m = &e->model;
    const ds4_weights *w = &e->weights;
    const char *layer_reads = getenv("DS4_HY4_ENGRAM_LAYER_READS");
    // The preceding unfinished chunk can hide one bulk read of all four tables.
    // A decode or verify step starts idle: layer 0 hides the first table.
    const bool staged = !g->prefill_queued && (!layer_reads || strcmp(layer_reads, "0") != 0);
    const uint8_t *emb = m->map + w->token_embd->abs_offset;
    if (!staged && !hy4_read_engram(g, e, prev, np, tokens, n, 0, 4)) {
        return false;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || tokens[i] >= 120832) {
            return false;
        }
        if (!hy4_input_row(g, tokens[i], pos + i, emb, g->host_embed + (size_t)i * 3072,
                           g->host_positions + 3 * i)) {
            return false;
        }
    }
    if (!ds4_gpu_tensor_write(g->embed, 0, g->host_embed, (uint64_t)n * 6144) ||
        !ds4_gpu_tensor_write(g->ids, 0, tokens, n * sizeof(int)) ||
        !ds4_gpu_tensor_write(g->positions, 0, g->host_positions, n * 3 * sizeof(int32_t)) ||
        !ds4_gpu_hy4_embed(g->streams, g->embed, n)) {
        return false;
    }
    // One CTA per row: prefill widths only; a decode row ran 3 % slower fused.
    enum { HY4_HC_FUSED_MIN_ROWS = 16 };
    const char *hc_env = getenv("DS4_HY4_HC_FUSED");
    const hy4_hc_mode mode = n < HY4_HC_FUSED_MIN_ROWS || (hc_env && strcmp(hc_env, "0") == 0) ? HY4_HC_SPLIT
                                                                                                : HY4_HC_FUSED;
    for (uint32_t il = 0; il < HY4_LAYERS; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        // A layer's attention input carries the preceding FFN unless Engram
        // reads the streams in between.
        const hy4_carry carry = il && !hy4_engram(il) ? HY4_CARRY : HY4_FRESH;
        if (il && carry == HY4_FRESH && !ds4_gpu_hy4_residual(g->streams, g->branch, g->post, n)) {
            return false;
        }
        // Later SSD reads overlap kernels already queued by the preceding layers.
        if (staged && hy4_engram(il) && !hy4_read_engram(g, e, prev, np, tokens, n, il / 10, 1)) {
            return false;
        }
        if ((hy4_engram(il) && !hy4_memory_branch(g, m, l, il, n)) ||
            !hy4_branch_in(g, m, l->hc_attn_fn, l->hc_attn_scale, l->hc_attn_base, l->attn_norm, n, carry, mode) ||
            !hy4_attention(g, m, l, il, n, pos, NULL) || (head && !hy4_capture_groups(g, il, n, pos)) ||
            !hy4_branch_in(g, m, l->hc_ffn_fn, l->hc_ffn_scale, l->hc_ffn_base, l->ffn_norm, n, HY4_CARRY,
                           mode) ||
            !hy4_ffn(g, m, l, n)) {
            fprintf(stderr, "ds4: Hy4 forward failed at layer %u, position %u, rows %u\n", il, pos, n);
            return false;
        }
    }
    if (!ds4_gpu_hy4_residual(g->streams, g->branch, g->post, n)) {
        return false;
    }
    if (head && !(g->verifying ? hy4_verify_head(g, m, w, n) : hy4_head_row(g, m, w, n - 1, 0))) {
        return false;
    }
    g->prefill_queued = !head;
    g->position = pos + n;
    if (g->position > g->high) {
        g->high = g->position;
    }
    return true;
}
#else
static bool hy4_graph_alloc(ds4_hy4_graph *g, uint32_t ctx, uint32_t cap, ds4_hy4_graph *owner) {
    (void)owner;
    (void)g;
    (void)ctx;
    (void)cap;
    return false;
}
static bool hy4_forward(ds4_hy4_graph *g, const ds4_engine *e, const int *p, unsigned np, const int *ids,
                        uint32_t n, uint32_t pos, bool head) {
    (void)g;
    (void)e;
    (void)p;
    (void)np;
    (void)ids;
    (void)n;
    (void)pos;
    (void)head;
    return false;
}
#endif
