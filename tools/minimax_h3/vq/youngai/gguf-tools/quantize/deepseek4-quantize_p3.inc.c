static void hot_mask_load(hot_mask *hm, const char *path) {
    memset(hm, 0, sizeof(*hm));
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open hot mask", path);
    unsigned char magic[4];
    if (fread(magic, 1, 4, fp) != 4) die("hot mask: short header");
    if (memcmp(magic, "DSXM", 4) != 0) die("hot mask: bad magic, expected DSXM");
    int32_t version = read_i32_fp(fp, "hot-mask version");
    if (version != 1) die("hot mask: unsupported version");
    int32_t n_layers = read_i32_fp(fp, "hot-mask n_layers");
    int32_t n_expert = read_i32_fp(fp, "hot-mask n_expert");
    (void)read_i32_fp(fp, "hot-mask reserved");
    if (n_layers < 1 || n_layers > 4096) die("hot mask: unreasonable n_layers");
    if (n_expert < 1 || n_expert > 65536) die("hot mask: unreasonable n_expert");
    size_t n_bits = (size_t)n_layers * (size_t)n_expert;
    size_t n_bytes = (n_bits + 7) / 8;
    uint8_t *bits = xmalloc(n_bytes);
    if (fread(bits, 1, n_bytes, fp) != n_bytes) die("hot mask: short bit body");
    fclose(fp);
    hm->active = true;
    hm->n_layers = n_layers;
    hm->n_expert = n_expert;
    hm->kept_counts = xcalloc((size_t)n_layers, sizeof(int));
    hm->kept_ids = xcalloc((size_t)n_layers, sizeof(int *));
    for (int l = 0; l < n_layers; l++) {
        int cnt = 0;
        for (int e = 0; e < n_expert; e++) {
            size_t idx = (size_t)l * (size_t)n_expert + (size_t)e;
            if (bits[idx >> 3] & (1u << (idx & 7))) cnt++;
        }
        hm->kept_counts[l] = cnt;
        hm->kept_ids[l] = xmalloc((size_t)(cnt > 0 ? cnt : 1) * sizeof(int));
        int s = 0;
        for (int e = 0; e < n_expert; e++) {
            size_t idx = (size_t)l * (size_t)n_expert + (size_t)e;
            if (bits[idx >> 3] & (1u << (idx & 7))) hm->kept_ids[l][s++] = e;
        }
        hm->total_kept += cnt;
    }
    free(bits);
    fprintf(stderr, "loaded hot mask %s: layers=%d n_expert=%d kept=%d/%d (%.1f%%)\n",
            path, hm->n_layers, hm->n_expert, hm->total_kept,
            hm->n_layers * hm->n_expert,
            100.0 * (double)hm->total_kept / ((double)hm->n_layers * (double)hm->n_expert));
}

static void hot_mask_free(hot_mask *hm) {
    if (!hm->active) return;
    for (int l = 0; l < hm->n_layers; l++) free(hm->kept_ids[l]);
    free(hm->kept_ids);
    free(hm->kept_counts);
    memset(hm, 0, sizeof(*hm));
}

/* =====
 * GGUF tensor mapping and quantization policy
 */

typedef enum { EXP_NONE, EXP_W1, EXP_W2, EXP_W3 } expert_part;

typedef struct {
    bool is_expert;
    bool is_mtp;   /* tensor lives in an MTP draft module (HF prefix mtp.<layer>. not layers.<layer>.) */
    int layer;     /* block index for blk.N / MTP module index for mtp.N */
    expert_part part;
} expert_tensor;

static expert_tensor parse_expert_tensor(const char *name) {
    expert_tensor e = {0};
    int layer = -1;
    char kind[16];
    int rest = 0;
    bool is_mtp = false;
    if (!(sscanf(name, "blk.%d.ffn_%15[^_]_exps.weight%n", &layer, kind, &rest) == 2
          && rest == (int)strlen(name))) {
        is_mtp = true;
        if (!(sscanf(name, "mtp.%d.ffn_%15[^_]_exps.weight%n", &layer, kind, &rest) == 2
              && rest == (int)strlen(name)))
            return e;
    }
    if (strcmp(kind, "gate") == 0 || strcmp(kind, "down") == 0 || strcmp(kind, "up") == 0) {
        e.is_expert = true;
        e.is_mtp = is_mtp;
        e.layer = layer;
        e.part = strcmp(kind, "gate") == 0 ? EXP_W1 : strcmp(kind, "down") == 0 ? EXP_W2 : EXP_W3;
    }
    return e;
}

static const char *expert_part_name(expert_part p) {
    switch (p) {
        case EXP_W1: return "w1";
        case EXP_W2: return "w2";
        case EXP_W3: return "w3";
        default: die("bad expert part");
    }
    return "";
}

typedef struct {
    const char *gguf;
    const char *hf;
} name_map;

static const name_map top_map[] = {
    { "token_embd.weight",      "embed.weight" },
    { "output_norm.weight",     "norm.weight" },
    { "output.weight",          "head.weight" },
    { "output_hc_base.weight",  "hc_head_base" },
    { "output_hc_fn.weight",    "hc_head_fn" },
    { "output_hc_scale.weight", "hc_head_scale" },
};

static const name_map layer_map[] = {
    { "hc_attn_base.weight",              "hc_attn_base" },
    { "hc_attn_fn.weight",                "hc_attn_fn" },
    { "hc_attn_scale.weight",             "hc_attn_scale" },
    { "hc_ffn_base.weight",               "hc_ffn_base" },
    { "hc_ffn_fn.weight",                 "hc_ffn_fn" },
    { "hc_ffn_scale.weight",              "hc_ffn_scale" },
    { "attn_sinks.weight",                "attn.attn_sink" },
    { "attn_q_a.weight",                  "attn.wq_a.weight" },
    { "attn_q_b.weight",                  "attn.wq_b.weight" },
    { "attn_q_a_norm.weight",             "attn.q_norm.weight" },
    { "attn_kv.weight",                   "attn.wkv.weight" },
    { "attn_kv_a_norm.weight",            "attn.kv_norm.weight" },
    { "attn_output_a.weight",             "attn.wo_a.weight" },
    { "attn_output_b.weight",             "attn.wo_b.weight" },
    { "attn_compressor_ape.weight",       "attn.compressor.ape" },
    { "attn_compressor_kv.weight",        "attn.compressor.wkv.weight" },
    { "attn_compressor_gate.weight",      "attn.compressor.wgate.weight" },
    { "attn_compressor_norm.weight",      "attn.compressor.norm.weight" },
    { "indexer.attn_q_b.weight",          "attn.indexer.wq_b.weight" },
    { "indexer.proj.weight",              "attn.indexer.weights_proj.weight" },
    { "indexer_compressor_ape.weight",    "attn.indexer.compressor.ape" },
    { "indexer_compressor_kv.weight",     "attn.indexer.compressor.wkv.weight" },
    { "indexer_compressor_gate.weight",   "attn.indexer.compressor.wgate.weight" },
    { "indexer_compressor_norm.weight",   "attn.indexer.compressor.norm.weight" },
    { "attn_norm.weight",                 "attn_norm.weight" },
    { "ffn_norm.weight",                  "ffn_norm.weight" },
    { "ffn_gate_shexp.weight",            "ffn.shared_experts.w1.weight" },
    { "ffn_up_shexp.weight",              "ffn.shared_experts.w3.weight" },
    { "ffn_down_shexp.weight",            "ffn.shared_experts.w2.weight" },
    { "ffn_gate_inp.weight",              "ffn.gate.weight" },
    { "exp_probs_b.bias",                 "ffn.gate.bias" },
    { "ffn_gate_tid2eid.weight",          "ffn.gate.tid2eid" },
};

/* MTP-only suffixes (after the mtp.<N>. prefix). The MTP module reuses most
 * regular layer tensor names, so anything not listed here falls back to
 * layer_map below; these are the ones that differ or have no layer analogue
 * (the eagle e/h projection + norms, and the hc head correction tensors, which
 * for a regular layer live under output_hc_* in top_map instead). */
static const name_map mtp_map[] = {
    { "hc_head_base.weight",   "hc_head_base" },
    { "hc_head_fn.weight",     "hc_head_fn" },
    { "hc_head_scale.weight",  "hc_head_scale" },
    /* 0731 DSpark drafter 头(2026-08-18): target-hidden 融合投影 + 置信/markov 头 */
    { "main_proj.weight",       "main_proj.weight" },
    /* mtp 层是 bias-topk 路由(非 hash), 但层内模板 blk.0 是 hash 层没有 exp_probs_b
     * 条目 ⇒ 必须走 specials 注入, 否则 drafter 路由丢 bias(2026-08-20 命中率 5% 根因) */
    { "exp_probs_b.bias",       "ffn.gate.bias" },
    { "main_norm.weight",       "main_norm.weight" },
    { "confidence_proj.weight", "confidence_head.proj.weight" },
    { "markov_w1.weight",       "markov_head.markov_w1.weight" },
    { "markov_w2.weight",       "markov_head.markov_w2.weight" },
    /* 旧 EAGLE 形态(pre-0731 checkpoint)保留兼容 */
    { "e_proj.weight",         "e_proj.weight" },
    { "h_proj.weight",         "h_proj.weight" },
    { "enorm.weight",          "enorm.weight" },
    { "hnorm.weight",          "hnorm.weight" },
    { "norm.weight",           "norm.weight" },
};

static char *hf_name_for_regular(const char *gguf_name) {
    for (size_t i = 0; i < sizeof(top_map) / sizeof(top_map[0]); i++) {
        if (strcmp(gguf_name, top_map[i].gguf) == 0) return xstrdup(top_map[i].hf);
    }
    int layer = -1;
    const char *p = gguf_name;
    if (sscanf(p, "mtp.%d.", &layer) == 1) {
        const char *rest = strchr(p + 4, '.');
        if (!rest) die("bad mtp tensor name");
        rest++;
        char buf[512];
        for (size_t i = 0; i < sizeof(mtp_map) / sizeof(mtp_map[0]); i++) {
            if (strcmp(rest, mtp_map[i].gguf) == 0) {
                snprintf(buf, sizeof(buf), "mtp.%d.%s", layer, mtp_map[i].hf);
                return xstrdup(buf);
            }
        }
        for (size_t i = 0; i < sizeof(layer_map) / sizeof(layer_map[0]); i++) {
            if (strcmp(rest, layer_map[i].gguf) == 0) {
                snprintf(buf, sizeof(buf), "mtp.%d.%s", layer, layer_map[i].hf);
                return xstrdup(buf);
            }
        }
        fprintf(stderr, "error: cannot map GGUF tensor to HF tensor: %s\n", gguf_name);
        exit(1);
    }
    if (sscanf(p, "blk.%d.", &layer) != 1) {
        fprintf(stderr, "error: cannot map GGUF tensor to HF tensor: %s\n", gguf_name);
        exit(1);
    }
    const char *rest = strchr(p + 4, '.');
    if (!rest) die("bad layer tensor name");
    rest++;
    for (size_t i = 0; i < sizeof(layer_map) / sizeof(layer_map[0]); i++) {
        if (strcmp(rest, layer_map[i].gguf) == 0) {
            char buf[512];
            snprintf(buf, sizeof(buf), "layers.%d.%s", layer, layer_map[i].hf);
            return xstrdup(buf);
        }
    }
    fprintf(stderr, "error: cannot map GGUF tensor to HF tensor: %s\n", gguf_name);
    exit(1);
}

typedef struct {
    char *prefix;
    ds4q_type type;
} type_override;

typedef struct {
    ds4q_type routed_w1, routed_w2, routed_w3;
    ds4q_type hash_w1, hash_w2, hash_w3;  /* layers < n_hash_layers: hash-routed, non-specialty */
    int n_hash_layers;
    /* --go1b-layers A:B — 逐层量化测试: 只有 layer∈[A:B] 的 routed 专家取 routed_w*(go1b),
     * 范围外的专家取 go1b_outside(默认 q8_0 高精)。go1b_hi<0 = 禁用(全按 routed_w*)。 */
    int go1b_lo, go1b_hi;
    ds4q_type go1b_outside;
    ds4q_type attention_proj, attention, shared, embedding, output, dense;
    type_override *overrides;
    int n_overrides;
} quant_policy;

static bool is_attention_projection(const char *name) {
    /* indexer 的 attn_q_b(blk.L.indexer.attn_q_b.weight)名字里也含 ".attn_q_b.weight", 但它属于
     * 引擎的 f16 专线家族(装载 expect f16, 只认 Q2_K 影子): 2026-09-06 --attention-proj q8_0 把它
     * 量成 q8_0 ⇒ "tensor blk.2.indexer.attn_q_b.weight has type q8_0, expected f16" 加载即死。
     * 模版驱动的默认路(tmpl->type=f16)从没踩到, 显式旗标才露馅。indexer 归 --attention 档。 */
    if (strstr(name, "indexer")) return false;
    return strstr(name, ".attn_kv.weight") || strstr(name, ".attn_q_a.weight") ||
           strstr(name, ".attn_q_b.weight") || strstr(name, ".attn_output_a.weight") ||
           strstr(name, ".attn_output_b.weight");
}

static bool is_attention_tensor(const char *name) {
    return strstr(name, ".attn") || strstr(name, "attn_") || strstr(name, ".indexer") || strstr(name, "indexer_");
}

static bool is_shared_expert(const char *name) {
    return strstr(name, "_shexp.") != NULL;
}

static bool is_output_tensor(const char *name) {
    return str_starts(name, "output.");
}

/* Tensors that ds4 loads through the plain-layout fast path and REQUIRES to be
 * F16/F32 — one tensor_expect_plain_layout() call each in ds4.c:
 *   - the "hc" head/attn/ffn projections (blk.N / mtp.N .hc_{head,attn,ffn}_fn,
 *     their 1-D _base/_scale siblings, and the main-model output_hc_*);
 *   - the MoE router gate ffn_gate_inp.weight (2-D F16/F32).
 * Quantizing any of these makes the model fail to load with
 * "tensor ... has type qN_k, expected F16 or F32". The 1-D base/scale are also
 * kept by the n_dims<=1 rule; the 2-D fn projections and the router gate need
 * this guard so an aggressive --dense/--attention policy can't reach them.
 * ffn_gate_inp matched precisely (not ffn_gate_exps/_shexp, which DO quantize). */
static bool is_plain_layout_weight(const char *name) {
    return strstr(name, "hc_head_") != NULL ||
           strstr(name, "hc_attn_") != NULL ||
           strstr(name, "hc_ffn_")  != NULL ||
           strstr(name, "_hc_fn")    != NULL ||
           strstr(name, "_hc_base")  != NULL ||
           strstr(name, "_hc_scale") != NULL ||
           strstr(name, "ffn_gate_inp") != NULL;
}

typedef struct {
    char *name;
    int n_dims;
    int64_t ne[DS4Q_MAX_DIMS];
    ds4q_type type;
    uint64_t old_offset;
    uint64_t new_offset;
    size_t size;
} tensor_meta;

static int tensor_n_dims(const tensor_meta *t) {
    int n = t->n_dims;
    while (n > 1 && t->ne[n - 1] == 1) n--;
    return n;
}

static ds4q_type policy_type(const quant_policy *p, const char *name, const tensor_meta *tmpl) {
    for (int i = 0; i < p->n_overrides; i++) {
        if (strcmp(name, p->overrides[i].prefix) == 0 || str_starts(name, p->overrides[i].prefix)) {
            return p->overrides[i].type;
        }
    }
    expert_tensor e = parse_expert_tensor(name);
    if (e.is_expert) {
        /* Hash layers (0..n_hash_layers-1) are hash-routed and non-specialty; give
         * them a separate (smaller) type so the size budget goes to the specialty
         * routed experts. Falls through to routed_w* when --hash-w* is unset. */
        if (!e.is_mtp && e.layer < p->n_hash_layers) {
            if (e.part == EXP_W1 && p->hash_w1 != DS4Q_TYPE_COUNT) return p->hash_w1;
            if (e.part == EXP_W2 && p->hash_w2 != DS4Q_TYPE_COUNT) return p->hash_w2;
            if (e.part == EXP_W3 && p->hash_w3 != DS4Q_TYPE_COUNT) return p->hash_w3;
        }
        /* 逐层量化测试: 范围外的专家用高精 go1b_outside(默认 q8_0) */
        if (p->go1b_hi >= 0 && !e.is_mtp && (e.layer < p->go1b_lo || e.layer > p->go1b_hi))
            return p->go1b_outside;
        if (e.part == EXP_W1 && p->routed_w1 != DS4Q_TYPE_COUNT) return p->routed_w1;
        if (e.part == EXP_W2 && p->routed_w2 != DS4Q_TYPE_COUNT) return p->routed_w2;
        if (e.part == EXP_W3 && p->routed_w3 != DS4Q_TYPE_COUNT) return p->routed_w3;
        return tmpl->type;
    }
    if (tmpl->type != DS4Q_TYPE_F32 && tmpl->type != DS4Q_TYPE_F16 &&
        tmpl->type != DS4Q_TYPE_BF16 && !ds4q_can_quantize(tmpl->type)) {
        return tmpl->type;
    }
    if (tensor_n_dims(tmpl) <= 1) return tmpl->type;
    if (is_plain_layout_weight(name)) return tmpl->type;  /* plain-layout F16/F32, never quantize */
    if (strcmp(name, "token_embd.weight") == 0 && p->embedding != DS4Q_TYPE_COUNT) return p->embedding;
    if (is_output_tensor(name) && p->output != DS4Q_TYPE_COUNT) return p->output;
    if (is_shared_expert(name) && p->shared != DS4Q_TYPE_COUNT) return p->shared;
    if (is_attention_projection(name) && p->attention_proj != DS4Q_TYPE_COUNT) return p->attention_proj;
    if (is_attention_tensor(name) && p->attention != DS4Q_TYPE_COUNT) return p->attention;
    if (p->dense != DS4Q_TYPE_COUNT) return p->dense;
    return tmpl->type;
}

static ds4q_type parse_type(const char *raw) {
    char wanted[64];
    size_t n = 0;
    for (const char *p = raw; *p && n + 1 < sizeof(wanted); p++) {
        if (*p != '-' && *p != '_') wanted[n++] = (char)tolower((unsigned char)*p);
    }
    wanted[n] = '\0';
    if (strcmp(wanted, "copy") == 0 || strcmp(wanted, "template") == 0) return DS4Q_TYPE_COUNT;
    for (int i = 0; i < DS4Q_TYPE_COUNT; i++) {
        char name[64];
        size_t m = 0;
        const char *tn = ds4q_type_name((ds4q_type)i);
        if (!tn) continue;
        for (const char *p = tn; *p && m + 1 < sizeof(name); p++) {
            if (*p != '-' && *p != '_') name[m++] = (char)tolower((unsigned char)*p);
        }
        name[m] = '\0';
        if (strcmp(name, wanted) == 0) return (ds4q_type)i;
    }
    fprintf(stderr, "error: unknown quant type: %s\n", raw);
    exit(1);
}

static bool is_quantizable_target(ds4q_type type) {
    return type == DS4Q_TYPE_F32 || type == DS4Q_TYPE_F16 || type == DS4Q_TYPE_BF16 || ds4q_can_quantize(type);
}

/* =====
 * Tensor generation
 */

typedef struct {
    uint8_t *data;
    size_t size;
} byte_buf;

static byte_buf f32_to_type(const float *src, int64_t n, ds4q_type type, int64_t ncols, const float *imat) {
    if (ncols <= 0 || n % ncols != 0) die("bad ncols for tensor conversion");
    byte_buf out = {0};
    if (type == DS4Q_TYPE_F32) {
        out.size = (size_t)n * sizeof(float);
        out.data = xmalloc(out.size);
        memcpy(out.data, src, out.size);
        return out;
    }
    if (type == DS4Q_TYPE_F16) {
        out.size = (size_t)n * sizeof(uint16_t);
        out.data = xmalloc(out.size);
        ds4q_f32_to_f16_row(src, (uint16_t *)out.data, n);
        return out;
    }
    if (type == DS4Q_TYPE_BF16) {
        out.size = (size_t)n * sizeof(uint16_t);
        out.data = xmalloc(out.size);
        ds4q_f32_to_bf16_row(src, (uint16_t *)out.data, n);
        return out;
    }
    if (!ds4q_can_quantize(type)) die("unsupported quant target type");
    if (ncols % ds4q_block_size(type) != 0) die("ncols is not divisible by quant block size");
    const int64_t nrows = n / ncols;
    out.size = (size_t)nrows * ds4q_row_size(type, ncols);
    out.data = xmalloc(out.size);

    float *synthetic = NULL;
    const float *im_ptr = imat;
    if (!im_ptr && ds4q_requires_imatrix(type)) {
        synthetic = xcalloc((size_t)ncols, sizeof(float));
        for (int64_t r = 0; r < nrows; r++) {
            const float *row = src + (size_t)r * (size_t)ncols;
            for (int64_t c = 0; c < ncols; c++) synthetic[c] += row[c] * row[c];
        }
        im_ptr = synthetic;
    }
    size_t written = ds4q_quantize_chunk(type, src, out.data, 0, nrows, ncols, im_ptr);
    free(synthetic);
    if (written != out.size) die("ds4q_quantize_chunk wrote unexpected byte count");
    return out;
}

