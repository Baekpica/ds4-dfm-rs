/* Gemma-4 VQ binding. Rust validated the release (release.json, files.json,
 * every GVQ1 header, every book link and the 12177-tensor layout), so this is
 * the pointer adapter: original HF names, stored types and GGUF dim order
 * (dim[0] is the input width, the transpose of the checkpoint's [out, in]).
 *
 * The trunk part carries no lm_head: the head is the token table
 * (`tie_word_embeddings`), so output borrows token_embd.
 *
 * The two expert stacks are one book each: `down_proj.vq_codebook` links the
 * 128 down matrices and `gate_up_proj.vq_codebook` links the gate and up
 * matrices (the checkpoint stores the gate and up projections as one stack).
 */
static ds4_tensor *gemma4_named(const ds4_model *m, const char *fmt,
                                unsigned il, unsigned expert, int has_expert) {
    char name[160];
    const int n = has_expert
        ? snprintf(name, sizeof(name), fmt, il, expert)
        : snprintf(name, sizeof(name), fmt, il);
    if (n < 0 || (size_t)n >= sizeof(name)) {
        ds4_die("Gemma-4 tensor name too long");
    }
    return required_tensor(m, name);
}

/* An R13+8-only tensor: null on the plain artifact, which the caller treats
 * as "this stack has no residual stage". */
static ds4_tensor *gemma4_optional(const ds4_model *m, const char *fmt,
                                   unsigned il, unsigned expert, int has_expert) {
    char name[160];
    const int n = has_expert
        ? snprintf(name, sizeof(name), fmt, il, expert)
        : snprintf(name, sizeof(name), fmt, il);
    if (n < 0 || (size_t)n >= sizeof(name)) {
        ds4_die("Gemma-4 tensor name too long");
    }
    return model_find_tensor(m, name);
}

static void gemma4_bind(ds4_weights *w, const ds4_model *m) {
    w->token_embd = required_tensor(m, "model.embed_tokens.weight");
    tensor_expect_layout(w->token_embd, DS4_TENSOR_BF16, 2, G4_EMBED, G4_VOCAB, 0);
    w->output_norm = required_tensor(m, "model.norm.weight");
    tensor_expect_layout(w->output_norm, DS4_TENSOR_BF16, 1, G4_EMBED, 0, 0);
    w->output = w->token_embd;

    w->gemma4 = calloc(G4_LAYERS, sizeof(*w->gemma4));
    if (!w->gemma4) { ds4_die("out of memory binding Gemma-4 layers"); }

    for (unsigned il = 0; il < G4_LAYERS; il++) {
        ds4_gemma4_layer *l = &w->gemma4[il];
        const unsigned dim = gemma4_head_dim(il);
        const unsigned kv = gemma4_kv_heads(il);
        const bool full = gemma4_is_full(il);

        /* The seven layer norms, in the order the checkpoint runs them:
         * attention in/out, the dense FFN pair, then the expert FFN pair. */
        static const char *const norms[] = {
            "model.layers.%u.input_layernorm.weight",
            "model.layers.%u.post_attention_layernorm.weight",
            "model.layers.%u.pre_feedforward_layernorm.weight",
            "model.layers.%u.post_feedforward_layernorm.weight",
            "model.layers.%u.pre_feedforward_layernorm_2.weight",
            "model.layers.%u.post_feedforward_layernorm_1.weight",
            "model.layers.%u.post_feedforward_layernorm_2.weight",
        };
        for (unsigned i = 0; i < sizeof(norms) / sizeof(norms[0]); i++) {
            l->norm[i] = gemma4_named(m, norms[i], il, 0, 0);
            tensor_expect_layout(l->norm[i], DS4_TENSOR_BF16, 1, G4_EMBED, 0, 0);
        }
        l->layer_scalar = gemma4_named(m, "model.layers.%u.layer_scalar", il, 0, 0);
        tensor_expect_layout(l->layer_scalar, DS4_TENSOR_BF16, 1, 1, 0, 0);

        l->q = gemma4_named(m, "model.layers.%u.self_attn.q_proj.weight", il, 0, 0);
        tensor_expect_layout(l->q, DS4_TENSOR_Q4_K, 2, G4_EMBED, G4_HEADS * dim, 0);
        l->k = gemma4_named(m, "model.layers.%u.self_attn.k_proj.weight", il, 0, 0);
        tensor_expect_layout(l->k, DS4_TENSOR_Q4_K, 2, G4_EMBED, kv * dim, 0);
        /* attention_k_eq_v: the full layers have no v_proj; their value
         * projection is the key projection. */
        l->v = full ? l->k
                    : gemma4_named(m, "model.layers.%u.self_attn.v_proj.weight",
                                   il, 0, 0);
        tensor_expect_layout(l->v, DS4_TENSOR_Q4_K, 2, G4_EMBED, kv * dim, 0);
        l->o = gemma4_named(m, "model.layers.%u.self_attn.o_proj.weight", il, 0, 0);
        tensor_expect_layout(l->o, DS4_TENSOR_Q4_K, 2, G4_HEADS * dim, G4_EMBED, 0);
        for (unsigned i = 0; i < 2; i++) {
            l->attn_norm[i] = gemma4_named(
                m, i == 0 ? "model.layers.%u.self_attn.q_norm.weight"
                          : "model.layers.%u.self_attn.k_norm.weight",
                il, 0, 0);
            tensor_expect_layout(l->attn_norm[i], DS4_TENSOR_BF16, 1, dim, 0, 0);
        }

        l->dense_gate = gemma4_named(m, "model.layers.%u.mlp.gate_proj.weight", il, 0, 0);
        tensor_expect_layout(l->dense_gate, DS4_TENSOR_Q4_K, 2, G4_EMBED, G4_DENSE, 0);
        l->dense_up = gemma4_named(m, "model.layers.%u.mlp.up_proj.weight", il, 0, 0);
        tensor_expect_layout(l->dense_up, DS4_TENSOR_Q4_K, 2, G4_EMBED, G4_DENSE, 0);
        l->dense_down = gemma4_named(m, "model.layers.%u.mlp.down_proj.weight", il, 0, 0);
        /* 2112 columns are not a multiple of Q4_K's 256-block: Q8_0. */
        tensor_expect_layout(l->dense_down, DS4_TENSOR_Q8_0, 2, G4_DENSE, G4_EMBED, 0);

        l->router = gemma4_named(m, "model.layers.%u.mlp.gate.weight", il, 0, 0);
        tensor_expect_layout(l->router, DS4_TENSOR_BF16, 2, G4_EMBED, G4_EXPERTS, 0);
        l->router_scale = gemma4_named(m, "model.layers.%u.mlp.gate.scale", il, 0, 0);
        tensor_expect_layout(l->router_scale, DS4_TENSOR_BF16, 1, G4_EMBED, 0, 0);
        l->router_per_expert =
            gemma4_named(m, "model.layers.%u.mlp.gate.per_expert", il, 0, 0);
        tensor_expect_layout(l->router_per_expert, DS4_TENSOR_BF16, 1, G4_EXPERTS, 0, 0);

        l->book_down =
            gemma4_named(m, "model.layers.%u.experts.down_proj.vq_codebook", il, 0, 0);
        l->book_gate_up =
            gemma4_named(m, "model.layers.%u.experts.gate_up_proj.vq_codebook", il, 0, 0);
        tensor_expect_layout(l->book_down, DS4_TENSOR_GLM_BOOK, 2,
                             G4_VECTOR, G4_WORDS, 0);
        tensor_expect_layout(l->book_gate_up, DS4_TENSOR_GLM_BOOK, 2,
                             G4_VECTOR, G4_WORDS, 0);
        for (unsigned e = 0; e < G4_EXPERTS; e++) {
            l->down[e] = gemma4_named(
                m, "model.layers.%u.experts.%u.down_proj", il, e, 1);
            l->gate[e] = gemma4_named(
                m, "model.layers.%u.experts.%u.gate_proj", il, e, 1);
            l->up[e] = gemma4_named(
                m, "model.layers.%u.experts.%u.up_proj", il, e, 1);
            tensor_expect_layout(l->down[e], DS4_TENSOR_GLM_VQ, 2, G4_FF, G4_EMBED, 0);
            tensor_expect_layout(l->gate[e], DS4_TENSOR_GLM_VQ, 2, G4_EMBED, G4_FF, 0);
            tensor_expect_layout(l->up[e], DS4_TENSOR_GLM_VQ, 2, G4_EMBED, G4_FF, 0);
        }

        /* The residual stage is all-or-none per artifact: an R13+8 build
         * carries the two 256-entry books and a plane for every one of the
         * 384 expert matrices, and a plain build carries none of them. */
        l->res_book_down = gemma4_optional(
            m, "model.layers.%u.experts.down_proj.vq_residual_book", il, 0, 0);
        l->res_book_gate_up = gemma4_optional(
            m, "model.layers.%u.experts.gate_up_proj.vq_residual_book", il, 0, 0);
        bool any_res = l->res_book_down || l->res_book_gate_up;
        for (unsigned e = 0; e < G4_EXPERTS; e++) {
            l->down_res[e] = gemma4_optional(
                m, "model.layers.%u.experts.%u.down_proj.res", il, e, 1);
            l->gate_res[e] = gemma4_optional(
                m, "model.layers.%u.experts.%u.gate_proj.res", il, e, 1);
            l->up_res[e] = gemma4_optional(
                m, "model.layers.%u.experts.%u.up_proj.res", il, e, 1);
            any_res = any_res || l->down_res[e] || l->gate_res[e] || l->up_res[e];
        }
        if (any_res) {
            if (!l->res_book_down || !l->res_book_gate_up) {
                ds4_die("Gemma-4 residual books are missing from an R13+8 part");
            }
            tensor_expect_layout(l->res_book_down, DS4_TENSOR_GLM_BOOK, 2,
                                 G4_VECTOR, 256, 0);
            tensor_expect_layout(l->res_book_gate_up, DS4_TENSOR_GLM_BOOK, 2,
                                 G4_VECTOR, 256, 0);
            for (unsigned e = 0; e < G4_EXPERTS; e++) {
                if (!l->down_res[e] || !l->gate_res[e] || !l->up_res[e]) {
                    ds4_die("Gemma-4 residual stage is partial");
                }
                tensor_expect_layout(l->down_res[e], DS4_TENSOR_GLM_RES, 2,
                                     G4_FF, G4_EMBED, 0);
                tensor_expect_layout(l->gate_res[e], DS4_TENSOR_GLM_RES, 2,
                                     G4_EMBED, G4_FF, 0);
                tensor_expect_layout(l->up_res[e], DS4_TENSOR_GLM_RES, 2,
                                     G4_EMBED, G4_FF, 0);
            }
        }
    }
}

#ifndef DS4_NO_GPU
/* Prefill chunk: the sliding-window ring is sized to hold a whole chunk on top
 * of the window, so the cap is the only knob. Default 2048 keeps the
 * ring at 3072 rows per sliding layer.
 *
 * P6a: batched widths used to order their sums differently from the one-row
 * launches (measured: a 2-row batch moved this model's logits by 2.4 nats,
 * first at the layer-0 o projection). The graph now repeats the one-row call
 * per row for the Q4_K projections, the Q8_0 dense down, the router and the
 * expert routes, so every width is bit-identical to one row (measured: a
 * 41-token prompt's logits are byte-identical at chunks 1, 2, 8, 9 and 22).
 * One row stays the default because it is the measured path: a wider chunk
 * repeats the same launches and buys nothing. */
static uint32_t gemma4_prefill_cap(uint32_t ctx) {
    uint32_t cap = 1u;
    const char *value = getenv("DS4_GEMMA4_PREFILL_CHUNK");
    if (value && *value) {
        char *end = NULL;
        const unsigned long parsed = strtoul(value, &end, 10);
        if (end != value && !*end && parsed > 0u && parsed <= G4_PREFILL_MAX) {
            cap = (uint32_t)parsed;
        }
    }
    return cap < ctx ? cap : ctx;
}

/* The session's device footprint, term for term the graph's own allocation:
 * `gemma4_activation_bytes` (shared with gemma4_graph_bytes), the KV layout
 * from ds4_gemma4.h, the linear workspace and the routed-expert scratch. */
static ds4_context_memory gemma4_session_memory(uint32_t ctx, uint32_t cap) {
    ds4_context_memory m = {0};
    if (!ctx || !cap || cap > ctx || ctx > G4_CONTEXT || cap > G4_PREFILL_MAX) {
        return m;
    }
    m.prefill_cap = cap;
    m.raw_cap = ctx;
    m.raw_bytes = ds4_gemma4_kv_bytes(ctx, cap);
    m.scratch_bytes = gemma4_activation_bytes(cap) +
        ds4_gpu_gvq_workspace_linear(cap) + ds4_gpu_g4_moe_bytes(cap < 512u ? cap : 512u);
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}

#endif
