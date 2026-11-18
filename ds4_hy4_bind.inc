/* Rust validates every part, dtype and dimension before the native adapter. */
enum {
    HY4_LAYERS = 42,
    HY4_HIDDEN = 3072,
    HY4_HC = 4,
    HY4_CONTEXT = 1048576,
    HY4_WINDOW = 512,
    HY4_ENGRAM_BYTES = 16 * 98,
    HY4_USED = 16,
    HY4_FF = 1536
};

enum { HY4_DRAFT_BLOCK = 6, HY4_TAPS = 5, HY4_SEED_BYTES = (3072 + HY4_TAPS * 511) * 4 };
typedef struct {
    ds4_layer_weights layer[HY4_TAPS];
    const ds4_tensor *enorm, *hnorm, *fusion, *gate, *hidden_norm, *embed_norm, *correct_gate, *correct_up,
        *correct_down, *norm;
} ds4_hy4_draft;
static void hy4_draft_bind(ds4_hy4_draft *d, const ds4_model *m) {
    for (uint32_t i = 0; i < HY4_TAPS; i++) {
        ds4_layer_weights *l = &d->layer[i];
#define HY4_DW(field) l->field = required_tensorf(m, "dfly.blk.%u." #field ".weight", i)
        HY4_DW(attn_norm);
        HY4_DW(attn_q_a);
        HY4_DW(attn_q_a_norm);
        HY4_DW(attn_q_b);
        HY4_DW(attn_kv_a_mqa);
        HY4_DW(attn_kv_a_norm);
        HY4_DW(attn_k_b);
        HY4_DW(attn_v_b);
        HY4_DW(attn_output);
        HY4_DW(attn_gate);
        HY4_DW(attn_sinks);
        l->attn_output_norm = required_tensorf(m, "dfly.blk.%u.post_attention_norm.weight", i);
        HY4_DW(ffn_norm);
        HY4_DW(ffn_gate);
        HY4_DW(ffn_up);
        HY4_DW(ffn_down);
#undef HY4_DW
    }
#define HY4_DW(field, name) d->field = required_tensor(m, "dfly." name ".weight")
    HY4_DW(enorm, "fusion.enorm");
    HY4_DW(hnorm, "fusion.hnorm");
    HY4_DW(fusion, "fusion.eh_proj");
    HY4_DW(gate, "fusion.gate");
    HY4_DW(hidden_norm, "correction.hidden_norm");
    HY4_DW(embed_norm, "correction.embed_norm");
    HY4_DW(correct_gate, "correction.gate_proj");
    HY4_DW(correct_up, "correction.up_proj");
    HY4_DW(correct_down, "correction.down_proj");
    HY4_DW(norm, "output_norm");
#undef HY4_DW
}

static bool hy4_sparse(uint32_t layer) { return layer == 0 || layer % 4 == 1; }
static bool hy4_engram(uint32_t layer) { return layer % 10 == 1 && layer <= 31; }

static void hy4_bind(ds4_weights *w, const ds4_model *m) {
    w->token_embd = required_tensor(m, "token_embd.weight");
    w->output = required_tensor(m, "output.weight");
    w->output_norm = required_tensor(m, "output_norm.weight");
    w->output_hc_fn = required_tensor(m, "output_hc_fn.weight");
    w->output_hc_scale = required_tensor(m, "output_hc_scale.weight");
    w->output_hc_base = required_tensor(m, "output_hc_base.weight");
    for (uint32_t i = 0; i < HY4_LAYERS; i++) {
        ds4_layer_weights *l = &w->layer[i];
#define HY4_WEIGHT(field) l->field = required_tensorf(m, "blk.%u." #field ".weight", i)
#define HY4_NAMED(field, name) l->field = required_tensorf(m, "blk.%u." name, i)
        HY4_WEIGHT(hc_attn_fn);
        HY4_WEIGHT(hc_attn_scale);
        HY4_WEIGHT(hc_attn_base);
        HY4_WEIGHT(hc_ffn_fn);
        HY4_WEIGHT(hc_ffn_scale);
        HY4_WEIGHT(hc_ffn_base);
        HY4_WEIGHT(attn_norm);
        HY4_WEIGHT(ffn_norm);
        HY4_NAMED(attn_output_norm, "post_attention_norm.weight");
        HY4_NAMED(ffn_output_norm, "post_ffw_norm.weight");
        HY4_WEIGHT(attn_q_a);
        HY4_WEIGHT(attn_q_a_norm);
        HY4_WEIGHT(attn_q_b);
        HY4_WEIGHT(attn_kv_a_mqa);
        HY4_WEIGHT(attn_kv_a_norm);
        HY4_WEIGHT(attn_k_b);
        HY4_WEIGHT(attn_v_b);
        HY4_WEIGHT(attn_sinks);
        HY4_WEIGHT(attn_gate);
        HY4_WEIGHT(attn_output);
        if (hy4_sparse(i)) {
            HY4_NAMED(indexer_attn_q_b, "indexer.attn_q_b.weight");
            HY4_NAMED(indexer_attn_k, "indexer.attn_k.weight");
            HY4_NAMED(indexer_k_norm, "indexer.k_norm.weight");
            HY4_NAMED(indexer_proj, "indexer.proj.weight");
            HY4_NAMED(indexer_compressor_kv, "indexer_compressor_value.weight");
            HY4_WEIGHT(indexer_compressor_gate);
            HY4_WEIGHT(indexer_compressor_ape);
            HY4_WEIGHT(indexer_compressor_norm);
        }
        if (hy4_engram(i)) {
            HY4_NAMED(engram_wkv, "engram_kv.weight");
            HY4_NAMED(engram_kv_bias, "engram_kv_b.weight");
            HY4_NAMED(engram_k, "engram_norm_key.weight");
            HY4_NAMED(engram_q, "engram_norm_query.weight");
            HY4_WEIGHT(engram_hc_scale);
            HY4_WEIGHT(engram_pre_gamma);
            HY4_WEIGHT(engram_lookup_norm);
        }
        if (!i) {
            HY4_WEIGHT(ffn_gate);
            HY4_WEIGHT(ffn_up);
            HY4_WEIGHT(ffn_down);
        } else {
            HY4_WEIGHT(ffn_gate_inp);
            HY4_NAMED(ffn_exp_probs_b, "exp_probs_b.bias");
            HY4_WEIGHT(ffn_routed_down);
            HY4_WEIGHT(ffn_routed_up);
            HY4_WEIGHT(ffn_gate_shexp);
            HY4_WEIGHT(ffn_up_shexp);
            HY4_WEIGHT(ffn_down_shexp);
            HY4_WEIGHT(ffn_gate_exps);
            HY4_WEIGHT(ffn_up_exps);
            HY4_WEIGHT(ffn_down_exps);
        }
#undef HY4_WEIGHT
#undef HY4_NAMED
    }
}

/* Same schema prices and allocates scratch, independent of context length. */
#define HY4_BUFFERS(X) \
    X(streams, 12288, float) \
    X(embed, 3072, uint16_t) \
    X(records, HY4_ENGRAM_BYTES * 4, uint8_t) \
    X(ids, 1, int32_t) \
    X(positions, 3, int32_t) \
    X(mix, 8, float) \
    X(post, 4, float) \
    X(input, 3072, float) \
    X(norm, 3072, float) \
    X(branch, 3072, float) \
    X(qa, 1024, float) \
    X(q, 64 * 256, float) \
    X(kv, 576, float) \
    X(attn_gate, 64 * 256, float) \
    X(absorbed, 64 * 512, float) \
    X(attended, 64 * 512, float) \
    X(packed, 64 * 512, uint16_t) \
    X(index_key, 128, float) \
    X(index_v, 128, float) \
    X(index_gate, 128, float) \
    X(index_q, 16 * 128, float) \
    X(index_w, 16, float) \
    X(groups, 512, int32_t) \
    X(tile, 1024, float) \
    X(best, 512, float) \
    X(router, 512, float) \
    X(experts, 16, int32_t) \
    X(weights, 16, float) \
    X(latent, 1536, float) \
    X(ff_gate, 16 * 1536, float) \
    X(ff_up, 16 * 1536, float) \
    X(ff_mid, 16 * 1536, float) \
    X(shared, 3072, float)

static uint32_t hy4_prefill_cap(uint32_t ctx) {
    uint32_t cap = 512;
    const char *value = getenv("DS4_HY4_PREFILL_CHUNK");
    if (value && *value) {
        char *end = NULL;
        unsigned long n = strtoul(value, &end, 10);
        if (end != value && !*end && n && n <= 8192) {
            cap = (uint32_t)n;
        }
    }
    return cap < ctx ? cap : ctx;
}

static bool hy4_fp8(void) {
    const char *mode = getenv("DS4_HY4_KV");
    return !(mode && strcmp(mode, "bf16") == 0);
}
static uint32_t hy4_ring(uint32_t ctx, uint32_t cap) {
    return ctx < cap + HY4_WINDOW - 1 ? ctx : cap + HY4_WINDOW - 1;
}
static uint32_t hy4_recent(uint32_t ctx, uint32_t cap) {
    return ctx < cap + HY4_WINDOW + 3 ? ctx : cap + HY4_WINDOW + 3;
}
static ds4_context_memory hy4_memory(uint32_t ctx, uint32_t cap) {
    ds4_context_memory m = {0};
    if (!ctx || ctx > HY4_CONTEXT || !cap || cap > ctx || cap > 8192) {
        return m;
    }
    m.raw_cap = ctx;
    m.prefill_cap = cap;
    m.raw_bytes = (uint64_t)30 * hy4_ring(ctx, cap) * 1152;
    m.compressed_bytes = (uint64_t)12 * ctx * (hy4_fp8() ? 641 : 1152);
    m.compressed_bytes +=
        (uint64_t)12 * (((uint64_t)ctx + 3) / 4 * 256 + (uint64_t)hy4_recent(ctx, cap) * 259 * 4);
#define HY4_PRICE(field, cols, type) m.scratch_bytes += (uint64_t)cap * (cols) * sizeof(type);
    HY4_BUFFERS(HY4_PRICE)
#undef HY4_PRICE
    m.raw_bytes += (1 + HY4_DRAFT_BLOCK) * HY4_SEED_BYTES;
    m.scratch_bytes += (uint64_t)120832 * sizeof(float);
    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}
