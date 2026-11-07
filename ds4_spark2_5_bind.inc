/* Spark-X2.6 VQ binding. Rust validated the release (release.json,
 * files.json, every GVQ1 header, every book and residual link, and the
 * 830-tensor layout), so this is the pointer adapter: the checkpoint's own
 * names, stored types and GGUF dim order (dim[0] is the input width, the
 * transpose of the checkpoint's [out, in]).
 *
 * The artifact carries no lm_head: `tie_word_embeddings` is true and the head
 * is the token table, so output borrows token_embd.
 *
 * Every matrix is one R13+8 container: the GVQ1 index plane, its 8192-word
 * codebook, the 256-word residual book and the GVR8 residual plane. The data
 * files hold one such group per matrix, so the book and the residual plane are
 * bound by the matrix's own name rather than shared across a layer.
 */

static ds4_tensor *spark2_5_named(const ds4_model *m, const char *fmt,
                                  unsigned il) {
    char name[192];
    const int n = snprintf(name, sizeof(name), fmt, il);
    if (n < 0 || (size_t)n >= sizeof(name)) {
        ds4_die("Spark-X2.6 tensor name too long");
    }
    return required_tensor(m, name);
}

/* One GVQ1 matrix and the three tensors it links: its 8192-word book, the
 * residual book and the matrix's GVR8 plane. The plane's own `book` link is
 * made in spark2_5_gpu_bind, where the record table exists. */
static ds4_tensor *spark2_5_matrix_tensor(const ds4_model *m, unsigned il,
                                          const char *stem,
                                          const char *suffix) {
    char name[192];
    const int n = snprintf(name, sizeof(name), "model.layers.%u.%s%s",
                           il, stem, suffix);
    if (n < 0 || (size_t)n >= sizeof(name)) {
        ds4_die("Spark-X2.6 tensor name too long");
    }
    return required_tensor(m, name);
}

static void spark2_5_bind_matrix(const ds4_model *m, ds4_spark2_5_matrix *mx,
                                 const char *stem, unsigned il,
                                 uint64_t in, uint64_t out) {
    mx->w = spark2_5_matrix_tensor(m, il, stem, ".weight");
    tensor_expect_layout(mx->w, DS4_TENSOR_GLM_VQ, 2, in, out, 0);
    mx->book = spark2_5_matrix_tensor(m, il, stem, ".weight.codebook");
    tensor_expect_layout(mx->book, DS4_TENSOR_GLM_BOOK, 2,
                         SPARK_VECTOR, SPARK_WORDS, 0);
    mx->res_book = spark2_5_matrix_tensor(m, il, stem,
                                          ".weight.vq_residual_book");
    tensor_expect_layout(mx->res_book, DS4_TENSOR_GLM_BOOK, 2,
                         SPARK_VECTOR, SPARK_RES_WORDS, 0);
    mx->res = spark2_5_matrix_tensor(m, il, stem, ".weight.res");
    tensor_expect_layout(mx->res, DS4_TENSOR_GLM_RES, 2, in, out, 0);
}

static void spark2_5_bind(ds4_weights *w, const ds4_model *m) {
    w->token_embd = required_tensor(m, "model.embedding.weight");
    tensor_expect_layout(w->token_embd, DS4_TENSOR_BF16, 2,
                         SPARK_EMBD, SPARK_VOCAB, 0);
    w->output_norm = required_tensor(m, "model.norm.weight");
    tensor_expect_layout(w->output_norm, DS4_TENSOR_F32, 2, 1, SPARK_EMBD, 0);
    w->output = w->token_embd;

    w->spark = calloc(SPARK_LAYERS, sizeof(*w->spark));
    if (!w->spark) { ds4_die("out of memory binding Spark-X2.6 layers"); }

    for (unsigned il = 0; il < SPARK_LAYERS; il++) {
        ds4_spark2_5_layer *l = &w->spark[il];
        l->attn_norm = spark2_5_named(
            m, "model.layers.%u.input_layernorm.weight", il);
        tensor_expect_layout(l->attn_norm, DS4_TENSOR_F32, 2, 1, SPARK_EMBD, 0);
        l->ffn_norm = spark2_5_named(
            m, "model.layers.%u.post_attention_layernorm.weight", il);
        tensor_expect_layout(l->ffn_norm, DS4_TENSOR_F32, 2, 1, SPARK_EMBD, 0);
        l->g_proj = spark2_5_named(
            m, "model.layers.%u.self_attn.g_proj.weight", il);
        tensor_expect_layout(l->g_proj, DS4_TENSOR_BF16, 2,
                             SPARK_EMBD, SPARK_HEADS, 0);

        /* The fused q/k/v projection (modeling_spark.py:152-176). */
        spark2_5_bind_matrix(m, &l->qkv, "self_attn.q_k_v_proj",
                             il, SPARK_EMBD, SPARK_QKV_DIM);
        spark2_5_bind_matrix(m, &l->out, "self_attn.out_proj",
                             il, SPARK_Q_DIM, SPARK_EMBD);
        spark2_5_bind_matrix(m, &l->ffn_gate, "mlp.gate_proj",
                             il, SPARK_EMBD, SPARK_FF);
        spark2_5_bind_matrix(m, &l->ffn_up, "mlp.up_proj",
                             il, SPARK_EMBD, SPARK_FF);
        spark2_5_bind_matrix(m, &l->ffn_down, "mlp.down_proj",
                             il, SPARK_FF, SPARK_EMBD);
    }
}
