/* Inkling-2 DSpark speculative decoding in the serial session (the release's
 * `drafter` part; TokenSpeed inkling2_mtp.py at c485a46a):
 *
 *   target forward ─▶ taps: the partial state entering layers 5, 11, ..., 41
 *        and the normalized output (BF16)
 *        ─▶ per draft layer: dflare_norm(sum_j alpha_lj tap_j) ─▶ k_ctx, v_ctx
 *        ─▶ k_norm_ctx + RoPE ─▶ that layer's ring; the Markov stage's mixture
 *        of the last committed row is f_rnn
 *   trial  [anchor embedding, mask x 7] at p.. ─▶ 5 x [ norm ─▶ q/k/v ─▶ q/k
 *          norms + RoPE ─▶ ring ─▶ attention over the context and the whole
 *          block (layers 0, 1, 3, 4: keys at or above q_pos - 1030) ─▶ o ─▶
 *          add-norm ─▶ SwiGLU 18,432 ─▶ add-norm ] ─▶ target head, rows 1..7
 *          ─▶ rnn_hidden: z = rnn_proj[state, w1[prev], hidden_k, f_rnn],
 *          state = GRU(z), proposal k = argmax softcap(base_k + fold(tanh z_o))
 *   verify the target forwards the anchor and the proposals once; commit keeps
 *          the accepted prefix and rolls the convolution histories back to it
 *          (saved before the trial, recomputed from the kept rows' inputs).
 *
 * Rejected positions' target K/V and draft ring rows are rewritten before
 * any read: the next trial's block covers them. fold = head . hidden_out
 * folds the head into the recurrence's correction (TokenSpeed's FP64 fold,
 * here the Q4_K head's F32 sums, BF16 stored). The global draft layer keeps
 * the last INKLING2_DRAFT_GLOBAL positions, TokenSpeed all of them: a named
 * deviation that changes proposals only. CUDA only: graph.inc includes this
 * file inside its GPU section. */

struct inkling2_draft_rt {
    const ds4_inkling2_draft *w;
    uint32_t rows, ring[DS4_INK2_DRAFT_LAYERS];
    uint32_t lo[DS4_INK2_DRAFT_LAYERS];   // lowest position whose context row each ring still holds
    ds4_gpu_tensor *alpha, *taps, *mixed, *ck, *cv, *markov, *f_rnn;
    ds4_gpu_tensor *ring_k[DS4_INK2_DRAFT_LAYERS], *ring_v[DS4_INK2_DRAFT_LAYERS];
    ds4_gpu_tensor *h, *x, *res, *q, *k, *v, *heads, *o, *gate, *up, *base, *logits, *zin, *pinv, *z, *state, *latent;
    ds4_gpu_tensor *ids, *fold_w, *fold_b, *vlogits, *vids, *hist_save, *trial_rows;
    uint64_t bytes;
    uint32_t trial_n, trial_pos;
    int trial[DS4_INK2_DRAFT_BLOCK];
    bool capture;
    double draft_ms, verify_ms;
};

static bool inkling2_draft_local(uint32_t l) { return l != INKLING2_DRAFT_GLOBAL_LAYER; }

// A new sequence: no pending trial; the rings refill from position 0.
static void inkling2_draft_reset(inkling2_draft_rt *d) {
    if (!d) { return; }
    d->trial_n = 0;
    d->capture = false;
    memset(d->lo, 0, sizeof(d->lo));
}

/* A rewind to `at` after rows up to `high` were written: a ring row below
 * `at` survives only above high - ring (the abandoned rows took the slots
 * below); the context restarts at `at` when none does. */
static void inkling2_draft_rewind(inkling2_draft_rt *d, uint32_t at, uint32_t high) {
    if (!d) { return; }
    d->trial_n = 0;
    for (uint32_t l = 0; l < DS4_INK2_DRAFT_LAYERS; l++) {
        const uint32_t lost = high > d->ring[l] ? high - d->ring[l] : 0u;
        if (d->lo[l] < lost) { d->lo[l] = lost; }
        if (d->lo[l] > at) { d->lo[l] = at; }
    }
}

static void inkling2_draft_free(inkling2_draft_rt *d) {
    if (!d) { return; }
    for (uint32_t l = 0; l < DS4_INK2_DRAFT_LAYERS; l++) {
        ds4_gpu_tensor_free(d->ring_k[l]);
        ds4_gpu_tensor_free(d->ring_v[l]);
    }
    ds4_gpu_tensor *all[] = {d->alpha, d->taps, d->mixed, d->ck, d->cv, d->markov, d->f_rnn, d->h, d->x, d->res, d->q,
        d->k, d->v, d->heads, d->o, d->gate, d->up, d->base, d->logits, d->zin, d->pinv, d->z, d->state, d->latent,
        d->ids, d->fold_w, d->fold_b, d->vlogits, d->vids, d->hist_save, d->trial_rows};
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) { ds4_gpu_tensor_free(all[i]); }
    free(d);
}

/* softmax(fuse_weight) per stage, rounded to BF16 (the reference's alpha). */
static bool inkling2_draft_alpha(inkling2_draft_rt *d, const ds4_model *m) {
    const uint16_t *fw = (const uint16_t *)(m->map + d->w->fuse->abs_offset);
    float alpha[(DS4_INK2_DRAFT_LAYERS + 1) * DS4_INK2_DRAFT_TAPS];
    for (uint32_t s = 0; s <= DS4_INK2_DRAFT_LAYERS; s++) {
        float v[DS4_INK2_DRAFT_TAPS], top = -INFINITY, sum = 0.0f;
        for (uint32_t j = 0; j < DS4_INK2_DRAFT_TAPS; j++) {
            uint32_t bits = (uint32_t)fw[s * DS4_INK2_DRAFT_TAPS + j] << 16;
            memcpy(&v[j], &bits, sizeof(float));
            top = fmaxf(top, v[j]);
        }
        for (uint32_t j = 0; j < DS4_INK2_DRAFT_TAPS; j++) { sum += expf(v[j] - top); }
        for (uint32_t j = 0; j < DS4_INK2_DRAFT_TAPS; j++) {
            const float a = expf(v[j] - top) / sum;
            uint32_t bits;
            memcpy(&bits, &a, sizeof(bits));
            bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
            memcpy(&alpha[s * DS4_INK2_DRAFT_TAPS + j], &bits, sizeof(float));
        }
    }
    return ds4_gpu_tensor_write(d->alpha, 0, alpha, sizeof(alpha)) != 0;
}

/* fold_w[v] = head_v . hidden_out (256 columns), fold_b[v] = head_v .
 * hidden_out.bias: the head maps hidden_out's columns and bias as 257 rows. */
static bool inkling2_draft_fold(ds4_gvq_ctx *gvq, inkling2_draft_rt *d, const ds4_model *m, const ds4_weights *w) {
    const uint32_t rows = DS4_INK2_RANK + 1u;
    const uint64_t in_row = (uint64_t)DS4_INK2_HIDDEN * sizeof(float), out_row = (uint64_t)DS4_INK2_VOCAB * sizeof(float);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(rows * in_row);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(rows * out_row);
    bool ok = x && out && ds4_gpu_ink2_draft_fold_x(gvq, x, inkling2_index(m, d->w->out_w), inkling2_index(m, d->w->out_b));
    // The head's projection takes at most a chunk of rows per call.
    for (uint32_t r0 = 0; ok && r0 < rows; r0 += d->rows) {
        const uint32_t n = rows - r0 < d->rows ? rows - r0 : d->rows;
        ds4_gpu_tensor *xi = ds4_gpu_tensor_view(x, r0 * in_row, n * in_row);
        ds4_gpu_tensor *oi = ds4_gpu_tensor_view(out, r0 * out_row, n * out_row);
        ok = xi && oi && ds4_gpu_gvq_linear(gvq, oi, inkling2_index(m, w->output), xi, n, DS4_GVQ_BF16);
        ds4_gpu_tensor_free(xi);
        ds4_gpu_tensor_free(oi);
    }
    ok = ok && ds4_gpu_ink2_draft_fold_pack(d->fold_w, d->fold_b, out) && ds4_gpu_synchronize();
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    return ok;
}

static inkling2_draft_rt *inkling2_draft_alloc(ds4_gvq_ctx *gvq, const ds4_model *m, const ds4_weights *w, uint32_t ctx,
        uint32_t rows) {
    if (!w->inkling2_draft || !rows || rows < DS4_INK2_DRAFT_BLOCK) { return NULL; }
    inkling2_draft_rt *d = xcalloc(1, sizeof(*d));
    d->w = w->inkling2_draft;
    d->rows = rows;
    bool ok = true;
#define INKLING2_DRAFT_ALLOC(field, size) do { \
        d->field = ds4_gpu_tensor_alloc(size); \
        ok = ok && d->field; \
        d->bytes += (size); \
    } while (0)
    const uint64_t f = sizeof(float), b = DS4_INK2_DRAFT_BLOCK, steps = b - 1u, kv = DS4_INK2_KV;
    INKLING2_DRAFT_ALLOC(alpha, (uint64_t)(DS4_INK2_DRAFT_LAYERS + 1u) * DS4_INK2_DRAFT_TAPS * f);
    INKLING2_DRAFT_ALLOC(taps, (uint64_t)rows * DS4_INK2_DRAFT_TAPS * DS4_INK2_HIDDEN * sizeof(uint16_t));
    INKLING2_DRAFT_ALLOC(mixed, (uint64_t)rows * DS4_INK2_HIDDEN * f);
    INKLING2_DRAFT_ALLOC(markov, (uint64_t)rows * DS4_INK2_HIDDEN * f);
    INKLING2_DRAFT_ALLOC(ck, (uint64_t)rows * kv * f);
    INKLING2_DRAFT_ALLOC(cv, (uint64_t)rows * kv * f);
    INKLING2_DRAFT_ALLOC(f_rnn, DS4_INK2_HIDDEN * f);
    for (uint32_t l = 0; l < DS4_INK2_DRAFT_LAYERS; l++) {
        d->ring[l] = inkling2_draft_local(l) ? INKLING2_DRAFT_LOCAL_RING
            : (ctx < INKLING2_DRAFT_GLOBAL ? ctx : INKLING2_DRAFT_GLOBAL) + DS4_INK2_DRAFT_BLOCK + 1u;
        INKLING2_DRAFT_ALLOC(ring_k[l], (uint64_t)d->ring[l] * kv * sizeof(uint16_t));
        INKLING2_DRAFT_ALLOC(ring_v[l], (uint64_t)d->ring[l] * kv * sizeof(uint16_t));
    }
    INKLING2_DRAFT_ALLOC(h, b * DS4_INK2_HIDDEN * f);
    INKLING2_DRAFT_ALLOC(x, b * DS4_INK2_HIDDEN * f);
    INKLING2_DRAFT_ALLOC(res, b * DS4_INK2_HIDDEN * f);
    INKLING2_DRAFT_ALLOC(o, b * DS4_INK2_HIDDEN * f);
    INKLING2_DRAFT_ALLOC(q, b * DS4_INK2_Q * f);
    INKLING2_DRAFT_ALLOC(heads, b * DS4_INK2_Q * f);
    INKLING2_DRAFT_ALLOC(k, b * kv * f);
    INKLING2_DRAFT_ALLOC(v, b * kv * f);
    INKLING2_DRAFT_ALLOC(gate, b * DS4_INK2_DRAFT_FF * f);
    INKLING2_DRAFT_ALLOC(up, b * DS4_INK2_DRAFT_FF * f);
    INKLING2_DRAFT_ALLOC(base, steps * DS4_INK2_VOCAB * f);
    INKLING2_DRAFT_ALLOC(logits, steps * DS4_INK2_VOCAB * f);
    INKLING2_DRAFT_ALLOC(zin, steps * DS4_INK2_RNN_IN * f);
    INKLING2_DRAFT_ALLOC(pinv, steps * 3u * DS4_INK2_RANK * f);
    INKLING2_DRAFT_ALLOC(z, 3u * DS4_INK2_RANK * f);
    INKLING2_DRAFT_ALLOC(state, DS4_INK2_RANK * f);
    INKLING2_DRAFT_ALLOC(latent, DS4_INK2_RANK * f);
    INKLING2_DRAFT_ALLOC(ids, b * sizeof(int32_t));
    INKLING2_DRAFT_ALLOC(fold_w, (uint64_t)DS4_INK2_VOCAB * DS4_INK2_RANK * sizeof(uint16_t));
    INKLING2_DRAFT_ALLOC(fold_b, (uint64_t)DS4_INK2_VOCAB * sizeof(uint16_t));
    INKLING2_DRAFT_ALLOC(vlogits, b * DS4_INK2_VOCAB * f);
    INKLING2_DRAFT_ALLOC(vids, b * sizeof(int32_t));
    INKLING2_DRAFT_ALLOC(hist_save, (uint64_t)INKLING2_LAYERS * INKLING2_HIST * f);
    INKLING2_DRAFT_ALLOC(trial_rows, (uint64_t)INKLING2_LAYERS * b * INKLING2_TRIAL_ROW * f);
#undef INKLING2_DRAFT_ALLOC
    if (!ok || d->bytes != inkling2_draft_bytes(ctx, rows) || !inkling2_draft_alpha(d, m) ||
        !inkling2_draft_fold(gvq, d, m, w)) {
        fprintf(stderr, "ds4: Inkling-2 DSpark drafter allocation failed\n");
        inkling2_draft_free(d);
        return NULL;
    }
    fprintf(stderr, "ds4: Inkling-2 DSpark drafter: %.2f GiB runtime, global draft layer over %u positions\n",
        (double)d->bytes / 1073741824.0, d->ring[INKLING2_DRAFT_GLOBAL_LAYER] - DS4_INK2_DRAFT_BLOCK - 1u);
    return d;
}

// Tap `il / 6` of the rows: the partial state entering a tapped layer.
static bool inkling2_draft_tap(ds4_inkling2_graph *g, uint32_t il, uint32_t rows) {
    if (!g->draft || !inkling2_global(il) || il + 1u >= INKLING2_LAYERS) { return true; }
    return ds4_gpu_ink2_draft_tap(g->draft->taps, g->x, il / DS4_INK2_BLOCK, rows) != 0;
}

// The trial forward's convolution inputs of layer il (see the header).
static bool inkling2_draft_keep(ds4_inkling2_graph *g, uint32_t il, uint32_t rows) {
    inkling2_draft_rt *d = g->draft;
    if (!d || !d->capture) { return true; }
    const uint64_t at = (uint64_t)il * DS4_INK2_DRAFT_BLOCK * INKLING2_TRIAL_ROW, b = DS4_INK2_DRAFT_BLOCK;
    return ds4_gpu_ink2_draft_copy(d->trial_rows, at, g->kin, (uint64_t)rows * DS4_INK2_KV) &&
        ds4_gpu_ink2_draft_copy(d->trial_rows, at + b * DS4_INK2_KV, g->vin, (uint64_t)rows * DS4_INK2_KV) &&
        ds4_gpu_ink2_draft_copy(d->trial_rows, at + b * 2u * DS4_INK2_KV, g->attn_out, (uint64_t)rows * DS4_INK2_HIDDEN) &&
        ds4_gpu_ink2_draft_copy(d->trial_rows, at + b * (2u * DS4_INK2_KV + DS4_INK2_HIDDEN), g->moe_out,
            (uint64_t)rows * DS4_INK2_HIDDEN);
}

/* After the target layers (g->xq holds the normalized output of every row):
 * the last tap, each draft layer's context K/V into its ring, and the Markov
 * stage's mixture of every row. */
static bool inkling2_draft_context(ds4_inkling2_graph *g, const ds4_model *m, uint32_t rows, uint32_t pos) {
    inkling2_draft_rt *d = g->draft;
    if (!d) { return true; }
    const ds4_inkling2_draft *w = d->w;
    if (!ds4_gpu_ink2_draft_tap(d->taps, g->xq, DS4_INK2_DRAFT_TAPS - 1u, rows)) { return false; }
    for (uint32_t l = 0; l < DS4_INK2_DRAFT_LAYERS; l++) {
        const ds4_inkling2_draft_layer *L = &w->layer[l];
        if (!ds4_gpu_ink2_draft_mix(g->gvq, d->mixed, d->taps, d->alpha, l, inkling2_index(m, L->dflare), 0, rows) ||
            !plain_graph_matmul_tensor(d->ck, m, L->k_ctx, DS4_INK2_HIDDEN, DS4_INK2_KV, d->mixed, rows) ||
            !plain_graph_matmul_tensor(d->cv, m, L->v_ctx, DS4_INK2_HIDDEN, DS4_INK2_KV, d->mixed, rows) ||
            !ds4_gpu_ink2_draft_qk(g->gvq, d->ck, inkling2_index(m, L->k_norm_ctx), DS4_INK2_KV_HEADS, rows, pos) ||
            !ds4_gpu_ink2_draft_ring(d->ring_k[l], d->ring_v[l], d->ck, d->cv, rows, pos, d->ring[l])) {
            fprintf(stderr, "ds4: Inkling-2 DSpark context of layer %u failed (rows=%u pos=%u)\n", l, rows, pos);
            return false;
        }
    }
    return ds4_gpu_ink2_draft_mix(g->gvq, d->markov, d->taps, d->alpha, DS4_INK2_DRAFT_LAYERS,
        inkling2_index(m, w->markov_norm), 0, rows) != 0;
}

// f_rnn = the Markov mixture of row `row` of the last forward.
static bool inkling2_draft_anchor(inkling2_draft_rt *d, uint32_t row) {
    ds4_gpu_tensor *src = ds4_gpu_tensor_view(d->markov, (uint64_t)row * DS4_INK2_HIDDEN * sizeof(float),
        DS4_INK2_HIDDEN * sizeof(float));
    const bool ok = src && ds4_gpu_ink2_draft_copy(d->f_rnn, 0, src, DS4_INK2_HIDDEN);
    ds4_gpu_tensor_free(src);
    return ok;
}

// The five layers over the block rows at p..p+7; x ends as norm(h + res).
static bool inkling2_draft_layers(ds4_inkling2_graph *g, const ds4_model *m, uint32_t p) {
    inkling2_draft_rt *d = g->draft;
    const ds4_inkling2_draft *w = d->w;
    const uint32_t n = DS4_INK2_DRAFT_BLOCK;
    if (!ds4_gpu_tensor_fill_f32(d->res, 0.0f, (uint64_t)n * DS4_INK2_HIDDEN)) { return false; }
    for (uint32_t l = 0; l < DS4_INK2_DRAFT_LAYERS; l++) {
        const ds4_inkling2_draft_layer *L = &w->layer[l];
        const uint32_t window = inkling2_draft_local(l) ? DS4_INK2_DRAFT_WINDOW : 0u;
        if (!ds4_gpu_ink2_draft_add_norm(g->gvq, d->x, d->res, d->h, inkling2_index(m, L->ln1), n) ||
            !plain_graph_matmul_tensor(d->q, m, L->q, DS4_INK2_HIDDEN, DS4_INK2_Q, d->x, n) ||
            !plain_graph_matmul_tensor(d->k, m, L->k, DS4_INK2_HIDDEN, DS4_INK2_KV, d->x, n) ||
            !plain_graph_matmul_tensor(d->v, m, L->v, DS4_INK2_HIDDEN, DS4_INK2_KV, d->x, n) ||
            !ds4_gpu_ink2_draft_qk(g->gvq, d->q, inkling2_index(m, L->q_norm), DS4_INK2_HEADS, n, p) ||
            !ds4_gpu_ink2_draft_qk(g->gvq, d->k, inkling2_index(m, L->k_norm), DS4_INK2_KV_HEADS, n, p) ||
            !ds4_gpu_ink2_draft_ring(d->ring_k[l], d->ring_v[l], d->k, d->v, n, p, d->ring[l]) ||
            !ds4_gpu_ink2_draft_attn(d->heads, g->partials, d->q, d->ring_k[l], d->ring_v[l], d->ring[l], n, p, d->lo[l],
                window) ||
            !plain_graph_matmul_tensor(d->o, m, L->o, DS4_INK2_Q, DS4_INK2_HIDDEN, d->heads, n) ||
            !ds4_gpu_ink2_draft_add_norm(g->gvq, d->x, d->res, d->o, inkling2_index(m, L->ln2), n) ||
            !plain_graph_matmul_tensor(d->gate, m, L->gate, DS4_INK2_HIDDEN, DS4_INK2_DRAFT_FF, d->x, n) ||
            !plain_graph_matmul_tensor(d->up, m, L->up, DS4_INK2_HIDDEN, DS4_INK2_DRAFT_FF, d->x, n) ||
            !ds4_gpu_ink2_draft_silu(d->gate, d->gate, d->up, (uint64_t)n * DS4_INK2_DRAFT_FF) ||
            !plain_graph_matmul_tensor(d->h, m, L->down, DS4_INK2_DRAFT_FF, DS4_INK2_HIDDEN, d->gate, n)) {
            fprintf(stderr, "ds4: Inkling-2 DSpark layer %u failed (pos=%u)\n", l, p);
            return false;
        }
    }
    return ds4_gpu_ink2_draft_add_norm(g->gvq, d->x, d->res, d->h, inkling2_index(m, w->norm), n) != 0;
}

/* Proposals tokens[1..n): the block from anchor tokens[0] at g->position,
 * the head on rows 1..n-1, then the recurrence, argmax on the device. */
static bool inkling2_draft_propose(ds4_inkling2_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t n,
        int *tokens) {
    inkling2_draft_rt *d = g->draft;
    const ds4_inkling2_draft *dw = d->w;
    const uint32_t p = g->position, steps = n - 1u;
    const uint64_t hidden = (uint64_t)DS4_INK2_HIDDEN * sizeof(float), vrow = (uint64_t)DS4_INK2_VOCAB * sizeof(float);
    int32_t ids[DS4_INK2_DRAFT_BLOCK] = {tokens[0]};
    ds4_gpu_tensor *rows = ds4_gpu_tensor_view(d->x, hidden, (uint64_t)steps * hidden);
    bool ok = rows && inkling2_embed(g, m, w, tokens, 1u) && ds4_gpu_ink2_draft_copy(d->h, 0, g->x, DS4_INK2_HIDDEN) &&
        ds4_gpu_ink2_draft_mask(g->gvq, d->h, inkling2_index(m, dw->mask), DS4_INK2_DRAFT_BLOCK) &&
        inkling2_draft_layers(g, m, p) && inkling2_mm1(g, m, d->base, w->output, rows, steps, DS4_GVQ_BF16) &&
        ds4_gpu_ink2_draft_markov_in(d->zin, d->x, d->f_rnn, steps) &&
        plain_graph_matmul_tensor(d->pinv, m, dw->rnn_w, DS4_INK2_RNN_IN, 3u * DS4_INK2_RANK, d->zin, steps) &&
        ds4_gpu_tensor_fill_f32(d->state, 0.0f, DS4_INK2_RANK) && ds4_gpu_tensor_write(d->ids, 0, ids, sizeof(int32_t));
    ds4_gpu_tensor_free(rows);
    for (uint32_t k = 0; ok && k < steps; k++) {
        ds4_gpu_tensor *logits = ds4_gpu_tensor_view(d->logits, (uint64_t)k * vrow, vrow);
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(d->ids, (uint64_t)(k + 1u) * sizeof(int32_t), sizeof(int32_t));
        ok = logits && out && ds4_gpu_ink2_draft_markov(g->gvq, logits, d->state, d->latent, d->z, d->pinv, d->base,
                d->fold_w, d->fold_b, d->ids, inkling2_index(m, dw->rnn_w), inkling2_index(m, dw->rnn_b),
                inkling2_index(m, dw->w1), k) &&
            ds4_gpu_argmax_rows_tensor(out, logits, DS4_INK2_VOCAB, 1u);
        ds4_gpu_tensor_free(logits);
        ds4_gpu_tensor_free(out);
    }
    if (!ok || !ds4_gpu_tensor_read(d->ids, 0, ids, (uint64_t)n * sizeof(int32_t))) { return false; }
    for (uint32_t i = 1; i < n; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= DS4_INK2_TOKENS) { return false; }
        tokens[i] = ids[i];
    }
    return true;
}

/* A rejected tail: each layer's histories become the saved ones advanced by
 * the kept rows' inputs. */
static bool inkling2_draft_rollback(ds4_inkling2_graph *g, uint32_t keep) {
    inkling2_draft_rt *d = g->draft;
    static const uint32_t width[INKLING2_DRAFT_STREAMS] = {DS4_INK2_KV, DS4_INK2_KV, DS4_INK2_HIDDEN, DS4_INK2_HIDDEN};
    static const uint32_t start[INKLING2_DRAFT_STREAMS] = {0, DS4_INK2_KV, 2u * DS4_INK2_KV,
        2u * DS4_INK2_KV + DS4_INK2_HIDDEN};
    const uint64_t b = DS4_INK2_DRAFT_BLOCK;
    bool ok = true;
    for (uint32_t il = 0; ok && il < INKLING2_LAYERS; il++) {
        ds4_gpu_tensor *saved = ds4_gpu_tensor_view(d->hist_save, (uint64_t)il * INKLING2_HIST * sizeof(float),
            (uint64_t)INKLING2_HIST * sizeof(float));
        ok = saved && ds4_gpu_ink2_draft_copy(g->hist[il], 0, saved, INKLING2_HIST);
        ds4_gpu_tensor_free(saved);
        for (uint32_t s = 0; ok && s < INKLING2_DRAFT_STREAMS; s++) {
            ds4_gpu_tensor *hist = inkling2_hist(g, il, s);
            ds4_gpu_tensor *rows = ds4_gpu_tensor_view(d->trial_rows,
                ((uint64_t)il * b * INKLING2_TRIAL_ROW + b * start[s]) * sizeof(float), (uint64_t)keep * width[s] * sizeof(float));
            ok = hist && rows && ds4_gpu_ink2_history(hist, rows, width[s], keep);
            ds4_gpu_tensor_free(hist);
            ds4_gpu_tensor_free(rows);
        }
    }
    return ok;
}
