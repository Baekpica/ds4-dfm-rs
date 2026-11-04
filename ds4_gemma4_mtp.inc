/* Gemma-4 drafter (`gemma4_assistant`): the P8 speculative part.
 *
 * The reference is transformers' `Gemma4AssistantForCausalLM` driving
 * `SinglePositionMultiTokenCandidateGenerator`; every step below cites it.
 *
 * The forward is the assistant's own (modeling_gemma4_assistant.py:166-194):
 *
 *   inputs  = cat([ trunk_embed(last_token), trunk_last_hidden ])   2816 + 2816
 *   h       = pre_projection(inputs)                                1024
 *   h       = four Gemma4TextDecoderLayer blocks, every one a shared layer
 *   hn      = mtp.model.norm(h)                                     1024
 *   logits  = lm_head(hn)   (the tied 262144 x 1024 table)          no softcap
 *   hidden  = post_projection(hn)                                   2816, fed back
 *
 * and the draft round is candidate_generator.py:1329-1351: up to
 * `num_assistant_tokens` steps of that forward, argmax between steps, with
 * `position_ids` held at `input_ids.shape[1] - 1` (line 1327) so the query
 * never advances inside the round.
 *
 * Four details decide the numbers and each is taken from the reference, not
 * from a plausible alternative:
 *
 * 1. The embedding half comes from the TARGET's table
 *    (candidate_generator.py:1336 `target_model_input_embeddings`, a
 *    Gemma4TextScaledWordEmbedding at the backbone width), never from the
 *    drafter's own 1024-wide table. That table is the tied head and nothing
 *    else (modeling_gemma4_assistant.py:110, 193). The two widths settle it.
 * 2. The K/V is the trunk's: modeling_gemma4.py:1219-1239 reads
 *    `shared_kv_states[layer_type]` for a shared layer and never projects
 *    locally; configuration_gemma4.py:236-251 names the last non-consumer
 *    layer of each type as the producer, which for this trunk is layer 28
 *    (sliding) and layer 29 (full) - see G4_SHARED_SLIDING/G4_SHARED_FULL.
 * 3. That K/V is truncated to the committed length
 *    (candidate_generator.py:1316-1319) while the query sits at the position
 *    of the token being committed, so the query's own row is NOT in the keys
 *    (the `keys` argument of ds4_gpu_g4_attn).
 * 4. The block order is the reference's, not the trunk's MoE order
 *    (modeling_gemma4.py:1379-1421): residual -> input norm -> attention ->
 *    post_attention norm -> add -> pre-FFN norm -> MLP -> post-FFN norm ->
 *    add -> * layer_scalar. The assistant has no experts
 *    (configuration_gemma4_assistant.py:88-89 refuses enable_moe_block), so
 *    only the dense MLP runs.

 * A draft round is a trial the spec lane verifies: the trunk then runs ONE
 * forward over the whole candidate block, so every drafted position is
 * checked by the model that defines the output stream, and only the longest
 * matching prefix is committed (ds4_session_eval_speculative_argmax). */

typedef struct {
    uint32_t norm[4], q, q_norm, o, gate, up, down, layer_scalar;
} ds4_gemma4_mtp_index;

typedef struct ds4_gemma4_mtp_runtime {
    ds4_gemma4_graph *g;         /* the trunk graph: its KV and its hidden row */
    ds4_gvq_model *model;        /* the part's pointer plan */
    ds4_gvq_ctx *gvq;            /* its own linear context (one-row shapes) */
    uint32_t drafts;             /* requested draft steps, clamped to G4_MTP_DRAFTS */
    uint32_t embed, norm, pre, post;
    ds4_gemma4_mtp_index layer[G4_MTP_LAYERS];
    ds4_gpu_tensor *hidden;      /* [2816] the trunk's final-norm row (the head's input half) */
    ds4_gpu_tensor *emb;         /* [2816] the trunk's token row, scaled */
    ds4_gpu_tensor *cat;         /* [5632] [emb | hidden] */
    ds4_gpu_tensor *x, *xn, *o;  /* [1024] residual, normed, branch */
    ds4_gpu_tensor *q, *attn;    /* [16][512] at the full layer's width */
    ds4_gpu_tensor *mlp_g, *mlp_u, *mlp_m;  /* [8192] */
    ds4_gpu_tensor *logits;      /* [262144] the drafter's head, one row */
    ds4_gpu_tensor *verify;      /* [(drafts + 1)][262144] the trunk's verify rows */
    ds4_gpu_tensor *tokens;      /* [drafts + 1] i32 */
    ds4_gpu_tensor *positions;   /* [1] u32: the round's fixed position */
    ds4_gpu_tensor *next;        /* [1] i32 */
    float *host;                 /* [(drafts + 1)][262144] verify logits on the host */
    bool hidden_valid;           /* `hidden` matches the last committed row */
    /* A verified block is pending: `trial[0..trial_n)` hold the spec lane's
     * candidates and g->position is already past them. The commit lands the
     * accepted prefix, exactly as ling31's mtp_n does. */
    uint32_t trial_n, trial_start;
    int trial[G4_MTP_DRAFTS + 1];
} ds4_gemma4_mtp_runtime;

/* One record per weight the drafter multiplies, indexed the way
 * ds4_gpu_gvq_linear wants. The layout is the GGUF one: `cols` is the input
 * width (dim[0]), so the record mirrors gemma4_gpu_bind's own build. */
static uint32_t gemma4_mtp_record(ds4_gvq_record **records, uint32_t *count,
                                  const ds4_tensor *t) {
    const uint64_t cols = t->ndim ? t->dim[0] : 1u;
    const uint64_t rows = cols ? t->elements / cols : 0u;
    if (!cols || !rows || cols > UINT32_MAX || rows > UINT32_MAX) {
        ds4_die("Gemma-4 drafter tensor has an unusable shape");
    }
    ds4_gvq_record *grown = realloc(*records, (size_t)(*count + 1u) * sizeof(grown[0]));
    if (!grown) { ds4_die("out of memory building the Gemma-4 drafter plan"); }
    *records = grown;
    grown[*count] = (ds4_gvq_record){t->abs_offset, t->bytes, t->type,
        (uint32_t)rows, (uint32_t)cols, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK,
        DS4_GVQ_NO_LINK};
    return (*count)++;
}

static void gemma4_mtp_runtime_free(ds4_gemma4_mtp_runtime *r) {
    if (!r) { return; }
    if (r->gvq) { ds4_gpu_gvq_free(r->gvq); }
    if (r->model) { ds4_gpu_gvq_unbind(r->model); }
    ds4_gpu_tensor_free(r->hidden);
    ds4_gpu_tensor_free(r->emb);
    ds4_gpu_tensor_free(r->cat);
    ds4_gpu_tensor_free(r->x);
    ds4_gpu_tensor_free(r->xn);
    ds4_gpu_tensor_free(r->o);
    ds4_gpu_tensor_free(r->q);
    ds4_gpu_tensor_free(r->attn);
    ds4_gpu_tensor_free(r->mlp_g);
    ds4_gpu_tensor_free(r->mlp_u);
    ds4_gpu_tensor_free(r->mlp_m);
    ds4_gpu_tensor_free(r->logits);
    ds4_gpu_tensor_free(r->verify);
    ds4_gpu_tensor_free(r->tokens);
    ds4_gpu_tensor_free(r->positions);
    ds4_gpu_tensor_free(r->next);
    free(r->host);
    free(r);
}

/* The trunk publishes its final-norm row; the drafter consumes it as the
 * hidden half of its input (modeling_gemma4_assistant.py:176, and
 * candidate_generator.py:1321-1324 picks the last validated row). */
static bool gemma4_mtp_publish(ds4_gemma4_graph *g, uint32_t row) {
    if (!g || !g->mtp || !g->norm) { return true; }
    ds4_gemma4_mtp_runtime *r = g->mtp;
    const uint64_t bytes = (uint64_t)G4_EMBED * sizeof(float);
    if (!ds4_gpu_gvq_copy(r->hidden, 0u, g->norm, (uint64_t)row * bytes, bytes, 1u)) {
        fprintf(stderr, "ds4: Gemma-4 drafter hidden publish failed at row %u\n", row);
        return false;
    }
    r->hidden_valid = true;
    return true;
}

/* A rewrite (prefix reuse, payload restore) drops the published row: the KV
 * rows are still exact, but the trunk row the drafter would extrapolate from
 * belongs to a forward this graph no longer holds. A pending verified block
 * dies with it, for the same reason. */
static void gemma4_mtp_drop_hidden(ds4_gemma4_graph *g) {
    if (!g || !g->mtp) { return; }
    g->mtp->hidden_valid = false;
    g->mtp->trial_n = 0u;
}

/* Build the drafter's device state over an existing trunk graph. `w` is the
 * bound part (e->mtp_weights.gemma4_mtp), `m` its model. */
static ds4_gemma4_mtp_runtime *gemma4_mtp_create(ds4_gemma4_graph *g,
        const ds4_gemma4_mtp *w, const ds4_model *m, uint32_t drafts) {
    if (!g || !w || !m || !g->ready || !drafts) { return NULL; }
    if (drafts > G4_MTP_DRAFTS) { drafts = G4_MTP_DRAFTS; }
    ds4_gemma4_mtp_runtime *r = xcalloc(1, sizeof(*r));
    r->g = g;
    r->drafts = drafts;
    ds4_gvq_record *records = NULL;
    uint32_t count = 0;
    r->embed = gemma4_mtp_record(&records, &count, w->embed);
    r->norm = gemma4_mtp_record(&records, &count, w->norm);
    r->pre = gemma4_mtp_record(&records, &count, w->pre);
    r->post = gemma4_mtp_record(&records, &count, w->post);
    for (uint32_t il = 0; il < G4_MTP_LAYERS; il++) {
        ds4_gemma4_mtp_index *ix = &r->layer[il];
        for (uint32_t i = 0; i < 4; i++) {
            ix->norm[i] = gemma4_mtp_record(&records, &count, w->layer[il].norm[i]);
        }
        ix->q = gemma4_mtp_record(&records, &count, w->layer[il].q);
        ix->q_norm = gemma4_mtp_record(&records, &count, w->layer[il].q_norm);
        ix->o = gemma4_mtp_record(&records, &count, w->layer[il].o);
        ix->gate = gemma4_mtp_record(&records, &count, w->layer[il].gate);
        ix->up = gemma4_mtp_record(&records, &count, w->layer[il].up);
        ix->down = gemma4_mtp_record(&records, &count, w->layer[il].down);
        ix->layer_scalar = gemma4_mtp_record(&records, &count, w->layer[il].layer_scalar);
    }
    r->model = ds4_gpu_gvq_bind(m->map, m->size, records, count);
    free(records);
    if (!r->model ||
        !(r->gvq = ds4_gpu_gvq_create_linear(r->model,
            ds4_gpu_gvq_workspace_linear(1u), 1u))) {
        fprintf(stderr, "ds4: Gemma-4 drafter plan creation failed\n");
        gemma4_mtp_runtime_free(r);
        return NULL;
    }
    const uint32_t rows = drafts + 1u;
    r->hidden = ds4_gpu_tensor_alloc((uint64_t)G4_EMBED * sizeof(float));
    r->emb = ds4_gpu_tensor_alloc((uint64_t)G4_EMBED * sizeof(float));
    r->cat = ds4_gpu_tensor_alloc((uint64_t)2u * G4_EMBED * sizeof(float));
    r->x = ds4_gpu_tensor_alloc((uint64_t)G4_MTP_EMBED * sizeof(float));
    r->xn = ds4_gpu_tensor_alloc((uint64_t)G4_MTP_EMBED * sizeof(float));
    r->o = ds4_gpu_tensor_alloc((uint64_t)G4_MTP_EMBED * sizeof(float));
    r->q = ds4_gpu_tensor_alloc((uint64_t)G4_MTP_HEADS * G4_GLOBAL_HEAD_DIM * sizeof(float));
    r->attn = ds4_gpu_tensor_alloc((uint64_t)G4_MTP_HEADS * G4_GLOBAL_HEAD_DIM * sizeof(float));
    r->mlp_g = ds4_gpu_tensor_alloc((uint64_t)G4_MTP_FFN * sizeof(float));
    r->mlp_u = ds4_gpu_tensor_alloc((uint64_t)G4_MTP_FFN * sizeof(float));
    r->mlp_m = ds4_gpu_tensor_alloc((uint64_t)G4_MTP_FFN * sizeof(float));
    r->logits = ds4_gpu_tensor_alloc((uint64_t)G4_VOCAB * sizeof(float));
    r->verify = ds4_gpu_tensor_alloc((uint64_t)rows * G4_VOCAB * sizeof(float));
    r->tokens = ds4_gpu_tensor_alloc((uint64_t)rows * sizeof(int32_t));
    r->positions = ds4_gpu_tensor_alloc(sizeof(uint32_t));
    r->next = ds4_gpu_tensor_alloc((uint64_t)rows * sizeof(int32_t));
    r->host = xmalloc((size_t)rows * G4_VOCAB * sizeof(float));
    if (!r->hidden || !r->emb || !r->cat || !r->x || !r->xn || !r->o || !r->q ||
        !r->attn || !r->mlp_g || !r->mlp_u || !r->mlp_m || !r->logits ||
        !r->verify || !r->tokens || !r->positions || !r->next || !r->host) {
        fprintf(stderr, "ds4: Gemma-4 drafter allocation failed\n");
        gemma4_mtp_runtime_free(r);
        return NULL;
    }
    g->mtp = r;
    return r;
}

#define G4M_STEP(call, step) do { \
    if (!(call)) { \
        fprintf(stderr, "ds4: Gemma-4 drafter %s failed\n", step); \
        return false; \
    } \
} while (0)

/* One drafter forward: the token and the trunk hidden row in, the next
 * token and the post-projected hidden out. `pos` is the round's fixed
 * position, the index of the token being committed. */
static bool gemma4_mtp_step(ds4_gemma4_mtp_runtime *r, const ds4_model *m,
        const ds4_weights *w, const ds4_gemma4_mtp *b, int token, uint32_t pos,
        int *next) {
    ds4_gemma4_graph *g = r->g;
    const uint32_t start = pos;
    if (!g || !m || !w || !b || !r->hidden_valid || pos < 1u || pos > g->position) {
        return false;
    }
    /* candidate_generator.py:1336: the target's own embedding table, with the
     * trunk's scaled word embedding, is the input half. */
    const int32_t id = (int32_t)token;
    G4M_STEP(ds4_gpu_tensor_write(r->tokens, 0, &id, sizeof(id)), "token upload");
    G4M_STEP(ds4_gpu_embed_tokens_quant_tensor(r->emb, r->tokens, m->map, m->size,
        w->token_embd->abs_offset, w->token_embd->type, G4_VOCAB, 1u, G4_EMBED),
        "trunk embedding gather");
    G4M_STEP(ds4_gpu_g4_embed_scale(r->emb, G4_EMBED), "embedding scale");
    /* candidate_generator.py:1337 `cat([embedding, hidden])`, then
     * modeling_gemma4_assistant.py:176 `pre_projection`. */
    G4M_STEP(ds4_gpu_gvq_concat(r->cat, r->emb, r->hidden, 1u, G4_EMBED), "input concat");
    G4M_STEP(ds4_gpu_gvq_linear(r->gvq, r->x, r->pre, r->cat, 1u, DS4_GVQ_BF16),
        "pre_projection");
    for (uint32_t il = 0; il < G4_MTP_LAYERS; il++) {
        const ds4_gemma4_mtp_index *ix = &r->layer[il];
        const bool full = gemma4_mtp_is_full(il);
        const uint32_t dim = gemma4_mtp_head_dim(il);
        const uint32_t shared = gemma4_shared_layer(full);
        const uint32_t kv_heads = gemma4_kv_heads(shared);
        const uint32_t ring = full ? 0u : ds4_gemma4_kv_rows(shared, g->ctx, g->cap);
        /* modeling_gemma4.py:1379-1392: attention in, norm, attention, norm,
         * add. A shared layer's attention takes q from q_proj + q_norm + rope
         * and k/v from the shared states (lines 1213-1222). */
        G4M_STEP(ds4_gpu_g4_rms(r->gvq, r->xn, r->x, ix->norm[0], 1u, G4_MTP_EMBED,
            G4_RMS_EPS), "input norm");
        G4M_STEP(ds4_gpu_gvq_linear(r->gvq, r->q, ix->q, r->xn, 1u, DS4_GVQ_BF16),
            "q projection");
        G4M_STEP(ds4_gpu_g4_rms(r->gvq, r->q, r->q, ix->q_norm, G4_MTP_HEADS, dim,
            G4_RMS_EPS), "query norm");
        G4M_STEP(ds4_gpu_g4_rope(r->q, r->positions, full ? g->inv_full : g->inv_sliding,
            1u, G4_MTP_HEADS, dim, gemma4_mtp_rot(il)), "query rope");
        /* `keys = start`: the trunk holds rows 0..start-1 (candidate_generator
         * .py:1316-1319), the window is relative to the query at `start`. */
        G4M_STEP(ds4_gpu_g4_attn(r->attn, r->q, gemma4_shared_k(g, full),
            gemma4_shared_v(g, full), g->positions, 1u, start, G4_MTP_HEADS, kv_heads,
            dim, full ? 0u : G4_WINDOW, ring, start), "shared attention");
        G4M_STEP(ds4_gpu_gvq_linear(r->gvq, r->o, ix->o, r->attn, 1u, DS4_GVQ_BF16),
            "o projection");
        G4M_STEP(ds4_gpu_g4_rms(r->gvq, r->o, r->o, ix->norm[1], 1u, G4_MTP_EMBED,
            G4_RMS_EPS), "post-attention norm");
        G4M_STEP(ds4_gpu_add_tensor(r->x, r->x, r->o, G4_MTP_EMBED), "attention residual");
        G4M_STEP(ds4_gpu_gvq_round(r->x, G4_MTP_EMBED), "attention residual rounding");
        /* modeling_gemma4.py:1395-1412: pre-FFN norm, the dense MLP (no MoE
         * block in this checkpoint), post-FFN norm, add. */
        G4M_STEP(ds4_gpu_g4_rms(r->gvq, r->xn, r->x, ix->norm[2], 1u, G4_MTP_EMBED,
            G4_RMS_EPS), "pre-FFN norm");
        G4M_STEP(ds4_gpu_gvq_linear(r->gvq, r->mlp_g, ix->gate, r->xn, 1u, DS4_GVQ_BF16),
            "FFN gate");
        G4M_STEP(ds4_gpu_gvq_linear(r->gvq, r->mlp_u, ix->up, r->xn, 1u, DS4_GVQ_BF16),
            "FFN up");
        G4M_STEP(ds4_gpu_g4_gelu(r->mlp_m, r->mlp_g, r->mlp_u, G4_MTP_FFN), "FFN gelu");
        G4M_STEP(ds4_gpu_gvq_linear(r->gvq, r->o, ix->down, r->mlp_m, 1u, DS4_GVQ_BF16),
            "FFN down");
        G4M_STEP(ds4_gpu_g4_rms(r->gvq, r->o, r->o, ix->norm[3], 1u, G4_MTP_EMBED,
            G4_RMS_EPS), "post-FFN norm");
        G4M_STEP(ds4_gpu_add_tensor(r->x, r->x, r->o, G4_MTP_EMBED), "FFN residual");
        G4M_STEP(ds4_gpu_gvq_round(r->x, G4_MTP_EMBED), "FFN residual rounding");
        /* modeling_gemma4.py:1422: the block output is scaled by the bf16
         * layer_scalar, not the branch. */
        G4M_STEP(ds4_gpu_g4_layer_scale(r->x, G4_MTP_EMBED, b->layer[il].scalar),
            "layer scalar");
    }
    /* modeling_gemma4.py:1687 the text model's final norm, then
     * modeling_gemma4_assistant.py:190-194: post_projection for the next
     * step's hidden half, and the head over the SAME normed row. */
    G4M_STEP(ds4_gpu_g4_rms(r->gvq, r->xn, r->x, r->norm, 1u, G4_MTP_EMBED, G4_RMS_EPS),
        "final norm");
    G4M_STEP(ds4_gpu_gvq_linear(r->gvq, r->logits, r->embed, r->xn, 1u, DS4_GVQ_BF16),
        "tied head");
    /* No final_logit_softcapping in the assistant's text config: the
     * checkpoint declares null, so the head is unwrapped (line 194). */
    G4M_STEP(ds4_gpu_argmax_tensor(r->next, r->logits, G4_VOCAB), "argmax");
    G4M_STEP(ds4_gpu_tensor_read(r->next, 0, next, sizeof(*next)), "argmax readback");
    G4M_STEP(ds4_gpu_gvq_linear(r->gvq, r->hidden, r->post, r->xn, 1u, DS4_GVQ_BF16),
        "post_projection");
    return *next >= 0 && (uint32_t)*next < G4_VOCAB;
}

#undef G4M_STEP

/* The trunk's verdict for every candidate row: one forward over the block,
 * then the tied head on each row. `target[i]` is the token the trunk would
 * have emitted at candidate i, exactly as its own single-row decode would
 * (the graph keeps one-row call shapes per row, so the block forward is
 * bit-identical to decoding the rows one at a time - P6a's measured rule). */
static bool gemma4_mtp_verify(ds4_gemma4_mtp_runtime *r, ds4_model *m, ds4_weights *w,
        const int *tokens, uint32_t n, uint32_t pos, int *target) {
    ds4_gemma4_graph *g = r->g;
    if (!g || !tokens || !n || n > r->drafts + 1u || pos < 1u || pos + n > g->ctx) {
        return false;
    }
    uint32_t ids[G4_MTP_DRAFTS + 1];
    for (uint32_t i = 0; i < n; i++) { ids[i] = (uint32_t)tokens[i]; }
    if (!gemma4_forward(g, m, w, ids, n, pos, false)) {
        fprintf(stderr, "ds4: Gemma-4 drafter verify forward failed\n");
        return false;
    }
    for (uint32_t row = 0; row < n; row++) {
        ds4_gpu_tensor *in = ds4_gpu_tensor_view(g->norm,
            (uint64_t)row * G4_EMBED * sizeof(float), (uint64_t)G4_EMBED * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(r->verify,
            (uint64_t)row * G4_VOCAB * sizeof(float), (uint64_t)G4_VOCAB * sizeof(float));
        /* The target's own head: the trunk's tied table and the trunk's
         * tanh-30 vocabulary cap, exactly as gemma4_forward's last row. The
         * drafter's 1024-wide table is never the target's head. */
        const bool ok = in && out &&
            ds4_gpu_gvq_linear(g->gvq, out, gemma4_index(m, w->token_embd), in, 1u,
                DS4_GVQ_BF16) &&
            ds4_gpu_g4_softcap(out, G4_VOCAB);
        ds4_gpu_tensor_free(in);
        ds4_gpu_tensor_free(out);
        if (!ok) {
            fprintf(stderr, "ds4: Gemma-4 verify head failed at row %u\n", row);
            return false;
        }
    }
    if (!ds4_gpu_tensor_read(r->verify, 0, r->host,
            (uint64_t)n * G4_VOCAB * sizeof(float))) {
        fprintf(stderr, "ds4: Gemma-4 verify logits readback failed\n");
        return false;
    }
    for (uint32_t i = 0; i < n; i++) {
        target[i] = sample_argmax(r->host + (uint64_t)i * G4_VOCAB, G4_VOCAB);
        if (target[i] < 0) { return false; }
    }
    if (getenv("DS4_MTP_SPEC_LOG")) {
        fprintf(stderr, "ds4: Gemma-4 verify pos=%u draft=", pos);
        for (uint32_t i = 0; i < n; i++) {
            fprintf(stderr, "%s%d", i ? "," : "", tokens[i]);
        }
        fprintf(stderr, " target=");
        for (uint32_t i = 0; i < n; i++) {
            fprintf(stderr, "%s%d", i ? "," : "", target[i]);
        }
        fprintf(stderr, "\n");
    }
    return true;
}

/* MTP trial: the reference's draft round, then a batched verify. The caller
 * keeps the longest candidate prefix whose rows match the trunk's own
 * tokens and commits it. */
int ds4_session_gemma4_trial(ds4_session *s, int first, int max_tokens,
        int *tokens, int *target, int capacity, char *err, size_t errlen) {
    if (!ds4_session_is_gemma4(s) || !tokens || !target || capacity < 1 ||
        first < 0 || (uint32_t)first >= DS4_N_VOCAB) {
        payload_set_err(err, errlen, "invalid Gemma-4 draft trial");
        return -1;
    }
    ds4_gemma4_graph *g = s->gemma4_graph;
    ds4_gemma4_mtp_runtime *r = s->gemma4_mtp;
    if (max_tokens < 2 || !s->gemma4_graph_ready || !g || !r || r->drafts < 2) { return 0; }
    /* Nothing to extrapolate from yet: a rewritten or restored session has the
     * KV rows but not the trunk row the drafter reads (it is republished by
     * the next forward), so this cycle decodes one token and drafts after. */
    if (!r->hidden_valid || !g->position) { return 0; }
    if (!s->checkpoint_valid || g->failed || r->trial_n ||
        g->position != (uint32_t)s->checkpoint.len) {
        payload_set_err(err, errlen, "Gemma-4 drafting needs a committed checkpoint");
        return -1;
    }
    uint32_t n = (uint32_t)max_tokens;
    if (n > (uint32_t)capacity) { n = (uint32_t)capacity; }
    if (n > r->drafts + 1u) { n = r->drafts + 1u; }
    if (n > g->ctx - g->position) { n = g->ctx - g->position; }
    if (n < 2u) { return 0; }
    ds4_model *m = &s->engine->model;
    ds4_weights *w = &s->engine->weights;
    const ds4_gemma4_mtp *b = s->engine->mtp_weights.gemma4_mtp;
    if (!b) { return 0; }
    const uint32_t start = g->position;
    /* candidate_generator.py:1327: one position for the whole round. */
    if (!ds4_gpu_tensor_write(r->positions, 0, &start, sizeof(start))) {
        gemma4_session_fail(s, err, errlen);
        return -1;
    }
    tokens[0] = first;
    for (uint32_t i = 0; i + 1u < n; i++) {
        if (!gemma4_mtp_step(r, m, w, b, tokens[i], start, &tokens[i + 1u])) {
            gemma4_session_fail(s, err, errlen);
            return -1;
        }
    }
    if (!gemma4_mtp_verify(r, &s->engine->model, w, tokens, n, start, target)) {
        gemma4_session_fail(s, err, errlen);
        return -1;
    }
    /* A sampling request draws its targets from the same verify rows with the
     * host's sampler (sample-and-match); a greedy request keeps the argmax
     * rows. Inkling's trial tail is the reference (ds4.c:78386): the rows the
     * verify read back are the plain decode's own, so the draw at row i is the
     * draw plain sampling would make there. */
    if (spec_sampling(&s->spec)) {
        spec_sample_rows(&s->spec, r->host, G4_VOCAB, G4_VOCAB, tokens, n, target);
    }
    g->position = start + n;
    r->trial_start = start;
    r->trial_n = n;
    if (n > (uint32_t)(sizeof(r->trial) / sizeof(r->trial[0]))) {
        gemma4_session_fail(s, err, errlen);
        return -1;
    }
    memcpy(r->trial, tokens, (size_t)n * sizeof(*tokens));
    ds4_metric_add(&ds4_metrics_get()->spec_drafts, n - 1u);
    return (int)n;
}

/* Commit `keep` of the verified rows: the KV rows are already written and
 * position-indexed, so only the frontier, the drafter's hidden and the
 * session's logits move. Every row past `keep` stays stale and is overwritten
 * before it is read (the session's own reuse rule). */
int ds4_session_gemma4_commit(ds4_session *s, int keep, char *err, size_t errlen) {
    ds4_gemma4_mtp_runtime *r = s->gemma4_mtp;
    if (!ds4_session_is_gemma4(s) || !s->checkpoint_valid || !r || !r->trial_n ||
        keep < 1 || (uint32_t)keep > r->trial_n ||
        (uint32_t)s->checkpoint.len != r->trial_start) {
        payload_set_err(err, errlen, "invalid Gemma-4 accepted prefix");
        return 1;
    }
    ds4_gemma4_graph *g = s->gemma4_graph;
    g->position = r->trial_start + (uint32_t)keep;
    if (!gemma4_mtp_publish(g, (uint32_t)keep - 1u)) {
        return gemma4_session_fail(s, err, errlen);
    }
    memcpy(s->logits, r->host + (size_t)(keep - 1) * G4_VOCAB,
        (size_t)G4_VOCAB * sizeof(float));
    for (int i = 0; i < keep; i++) { token_vec_push(&s->checkpoint, r->trial[i]); }
    if (keep > 1) { ds4_metric_add(&ds4_metrics_get()->spec_hits, (uint64_t)keep - 1u); }
    r->trial_n = 0u;
    s->mtp_draft_valid = false;
    if (getenv("DS4_MTP_SPEC_LOG")) {
        fprintf(stderr, "ds4: Gemma-4 MTP keep=%d pos=%u\n", keep, g->position);
    }
    return 0;
}
