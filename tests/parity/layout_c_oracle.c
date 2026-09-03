/* Host weights_validate_layout expected table. Do not include ds4.c.
 * SPEC / COUNT / LAYOUT lines match crates/ds4-core/src/layout.rs. */

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    T_F32 = 0,
    T_F16 = 1,
    T_Q4_0 = 2,
    T_Q5_0 = 6,
    T_Q8_0 = 8,
    T_Q2_K = 10,
    T_Q3_K = 11,
    T_Q4_K = 12,
    T_Q5_K = 13,
    T_Q6_K = 14,
    T_IQ2_XXS = 16,
    T_I32 = 26,
    T_I64 = 27,
    T_BF16 = 30
};

enum {
    FAM_DEEPSEEK4 = 0,
    FAM_SOLAR = 1,
    FAM_MOTIF3 = 2,
    FAM_EXAONE = 3,
    FAM_DOTS3 = 4,
    FAM_QWEN = 5,
    FAM_GLM53 = 6
};

enum {
    VAR_FLASH = 0,
    VAR_PRO = 1,
    VAR_SOLAR = 2,
    VAR_MOTIF3 = 3,
    VAR_KEXAONE = 4,
    VAR_DOTS3 = 5,
    VAR_QWEN = 6,
    VAR_GLM53 = 7
};

typedef enum {
    CLS_EXACT,
    CLS_OPTIONAL,
    CLS_PLAIN,
    CLS_MOTIF_PROJ,
    CLS_ROUTED,
    CLS_EXAONE,
    CLS_SOLAR_GATEUP,
    CLS_SOLAR_DOWN,
    CLS_SOLAR_CONV,
    CLS_SOLAR_DECAY,
    CLS_QWEN_MATRIX,
    CLS_QWEN_PLAIN,
    CLS_QWEN_MTP_ROUTED,
    CLS_GLM_DENSE
} type_class;

typedef struct {
    const char *name;
    uint32_t family;
    uint32_t variant;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_noise_head;
    uint32_t n_head_dim;
    uint32_t n_value_dim;
    uint32_t n_rot;
    uint32_t n_out_group;
    uint32_t n_lora_q;
    uint32_t n_lora_o;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_ff_exp;
    uint32_t n_ff_dense;
    uint32_t n_ff_shexp;
    uint32_t n_hash_layer;
    uint32_t n_swa_period;
    uint32_t n_indexer_head;
    uint32_t n_indexer_head_dim;
    uint32_t n_hc;
    uint32_t n_nextn_predict;
    uint32_t n_leading_dense;
    uint32_t n_kv_lora;
    uint32_t n_key_mla;
    uint32_t n_value_mla;
    uint32_t n_swa_head;
    uint32_t n_swa_kv_lora;
    uint32_t n_swa_key_mla;
    uint32_t n_kda_head_dim;
    uint32_t n_ssm_conv;
    bool use_qk_norm;
} layout_shape;

static const layout_shape SHAPE_FLASH = {
    "DeepSeek V4 Flash", FAM_DEEPSEEK4, VAR_FLASH,
    43, 4096, 129280, 64, 1, 0, 512, 512, 64, 8, 1024, 1024,
    256, 6, 2048, 0, 0, 3, 0, 64, 128, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false
};
static const layout_shape SHAPE_PRO = {
    "DeepSeek V4 Pro", FAM_DEEPSEEK4, VAR_PRO,
    61, 7168, 129280, 128, 1, 0, 512, 512, 64, 16, 1536, 1024,
    384, 6, 3072, 0, 0, 3, 0, 64, 128, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false
};
static const layout_shape SHAPE_SOLAR = {
    "Solar Open2 250B", FAM_SOLAR, VAR_SOLAR,
    48, 4096, 196608, 64, 8, 0, 128, 128, 0, 0, 0, 0,
    320, 8, 1280, 10240, 1280, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 4, false
};
static const layout_shape SHAPE_MOTIF3 = {
    "Motif-3", FAM_MOTIF3, VAR_MOTIF3,
    53, 4096, 220160, 80, 16, 16, 192, 128, 64, 0, 1024, 0,
    384, 8, 1280, 12288, 0, 0, 4, 0, 0, 4, 1, 2, 512, 192, 128, 0, 0, 0, 0, 0, false
};
static const layout_shape SHAPE_KEXAONE = {
    "K-EXAONE 236B A23B", FAM_EXAONE, VAR_KEXAONE,
    49, 6144, 153600, 64, 8, 0, 128, 128, 128, 0, 0, 0,
    128, 8, 2048, 18432, 2048, 0, 4, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, true
};
static const layout_shape SHAPE_DOTS3 = {
    "dots3-note-prev", FAM_DOTS3, VAR_DOTS3,
    47, 5120, 152064, 128, 128, 0, 192, 128, 64, 0, 1024, 0,
    256, 8, 1536, 13824, 0, 0, 4, 64, 128, 0, 1, 1, 512, 192, 128, 64, 1024, 256, 0, 0, false
};
static const layout_shape SHAPE_QWEN = {
    .name = "Qwen3.8-Flash-Next",
    .family = FAM_QWEN,
    .variant = VAR_QWEN,
    .n_layer = 48,
    .n_embd = 2560,
    .n_vocab = 248320,
    .n_head = 24,
    .n_head_kv = 2,
    .n_head_dim = 256,
    .n_value_dim = 256,
    .n_rot = 64,
    .n_expert = 512,
    .n_expert_used = 10,
    .n_ff_exp = 640,
    .n_ff_shexp = 640,
    .n_swa_period = 4,
    .n_indexer_head = 4,
    .n_indexer_head_dim = 128,
    .n_hc = 4,
    .n_nextn_predict = 1,
    .n_kda_head_dim = 128,
    .n_ssm_conv = 4,
    .use_qk_norm = true
};
static const layout_shape SHAPE_GLM53 = {
    .name = "GLM 5.3 Flash",
    .family = FAM_GLM53,
    .variant = VAR_GLM53,
    .n_layer = 46,
    .n_embd = 4096,
    .n_vocab = 154880,
    .n_head = 64,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 256,
    .n_lora_q = 1536,
    .n_expert = 288,
    .n_expert_used = 8,
    .n_ff_exp = 2048,
    .n_ff_dense = 12288,
    .n_indexer_head = 32,
    .n_indexer_head_dim = 128,
    .n_hc = 4,
    .n_nextn_predict = 1,
    .n_leading_dense = 3,
    .n_kv_lora = 512,
    .n_key_mla = 256,
    .n_value_mla = 256,
    .n_kda_head_dim = 128,
    .n_ssm_conv = 4
};

static uint32_t g_n;

static const char *type_name(uint32_t t)
{
    static const char *n[] = {
        "f32", "f16", "q4_0", "q4_1", NULL, NULL, "q5_0", "q5_1",
        "q8_0", "q8_1", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k",
        "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "iq4_nl", "iq3_s", "iq2_s",
        "iq4_xs", "i8", "i16", "i32", "i64", "f64", "iq1_m", "bf16"
    };
    if (t >= sizeof(n) / sizeof(n[0]) || !n[t]) return "unknown";
    return n[t];
}

static void spec(const char *name, type_class cls, uint32_t typ, uint32_t ndim,
                 uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3)
{
    char tok[64];
    const char *ct;
    uint32_t nprint = ndim;
    uint64_t ds[4] = {d0, d1, d2, d3};
    uint32_t i;
    switch (cls) {
    case CLS_EXACT:
        snprintf(tok, sizeof(tok), "exact:%s", type_name(typ));
        ct = tok;
        break;
    case CLS_OPTIONAL:
        snprintf(tok, sizeof(tok), "optional:%s", type_name(typ));
        ct = tok;
        break;
    case CLS_PLAIN:
        ct = "plain";
        break;
    case CLS_MOTIF_PROJ:
        ct = "motif-proj";
        break;
    case CLS_ROUTED:
        ct = "routed";
        break;
    case CLS_EXAONE:
        ct = "exaone-quant";
        break;
    case CLS_SOLAR_GATEUP:
        ct = "solar-gateup";
        break;
    case CLS_SOLAR_DOWN:
        ct = "solar-down";
        break;
    case CLS_SOLAR_CONV:
        ct = "solar-conv";
        break;
    case CLS_SOLAR_DECAY:
        ct = "solar-decay";
        break;
    case CLS_QWEN_MATRIX:
        ct = "qwen-matrix";
        break;
    case CLS_QWEN_PLAIN:
        ct = "qwen-plain";
        break;
    case CLS_QWEN_MTP_ROUTED:
        ct = "qwen-mtp-routed";
        break;
    case CLS_GLM_DENSE:
        ct = "glm-dense";
        break;
    default:
        ct = "unknown";
        break;
    }
    if (cls == CLS_SOLAR_CONV || cls == CLS_SOLAR_DECAY) {
        uint32_t m = ndim < 2 ? 2 : ndim;
        nprint = m < 3 ? m : 3;
    }
    printf("SPEC %s %s %u ", name, ct, ndim);
    for (i = 0; i < nprint; i++) {
        if (i) putchar(',');
        printf("%" PRIu64, ds[i]);
    }
    putchar('\n');
    g_n++;
}

static void spec5(const char *name, type_class cls, uint32_t typ,
                  uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3,
                  uint64_t d4)
{
    const char *ct = cls == CLS_EXACT ? type_name(typ) : "unknown";
    printf("SPEC %s exact:%s 5 %" PRIu64 ",%" PRIu64 ",%" PRIu64
           ",%" PRIu64 ",%" PRIu64 "\n",
           name, ct, d0, d1, d2, d3, d4);
    g_n++;
}

static void specf(const char *fmt, uint32_t il, type_class cls, uint32_t typ,
                  uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3)
{
    char name[128];
    snprintf(name, sizeof(name), fmt, il);
    spec(name, cls, typ, ndim, d0, d1, d2, d3);
}

static uint32_t expected_compress_ratio(const layout_shape *s, uint32_t il)
{
    if (il >= s->n_layer) return 0;
    if (s->variant == VAR_FLASH) {
        if (il < 2) return 0;
        return (il & 1u) == 0 ? 4u : 128u;
    }
    if (s->variant == VAR_PRO) {
        if (il < 2) return 128u;
        return (il & 1u) == 0 ? 4u : 128u;
    }
    return 0;
}

static bool solar_layer_is_gqa(const layout_shape *s, uint32_t il)
{
    return s->family == FAM_SOLAR && il < s->n_layer && (il % 4u) == 0u;
}

static bool dots3_layer_is_full_attention(const layout_shape *s, uint32_t il)
{
    if (s->family != FAM_DOTS3 || il >= s->n_layer) return false;
    if (s->n_nextn_predict != 0 && il + s->n_nextn_predict >= s->n_layer) return false;
    return il == 0u || (s->n_swa_period != 0 && (il % s->n_swa_period) == 1u);
}

static bool qwen4exp_layer_is_full_attention(const layout_shape *s, uint32_t il)
{
    return s->family == FAM_QWEN && il < s->n_layer &&
           s->n_swa_period != 0 && (il % s->n_swa_period) == 3u;
}

static bool is_nextn(const layout_shape *s, uint32_t il)
{
    return s->n_nextn_predict != 0 && il + s->n_nextn_predict >= s->n_layer;
}

static void motif_mhc(const char *prefix, const layout_shape *s)
{
    uint64_t e = s->n_embd;
    uint64_t hc = s->n_hc;
    uint64_t hc_dim = e * hc;
    char n[160];
    snprintf(n, sizeof(n), "%s.rms_norm.weight", prefix);
    spec(n, CLS_EXACT, T_F32, 1, hc_dim, 0, 0, 0);
    snprintf(n, sizeof(n), "%s.proj_pre.weight", prefix);
    spec(n, CLS_MOTIF_PROJ, 0, 2, hc_dim, hc, 0, 0);
    snprintf(n, sizeof(n), "%s.proj_post.weight", prefix);
    spec(n, CLS_MOTIF_PROJ, 0, 2, hc_dim, hc, 0, 0);
    snprintf(n, sizeof(n), "%s.proj_res.weight", prefix);
    spec(n, CLS_MOTIF_PROJ, 0, 2, hc_dim, hc * hc, 0, 0);
    snprintf(n, sizeof(n), "%s.alpha_pre", prefix);
    spec(n, CLS_EXACT, T_F32, 1, 1, 0, 0, 0);
    snprintf(n, sizeof(n), "%s.alpha_post", prefix);
    spec(n, CLS_EXACT, T_F32, 1, 1, 0, 0, 0);
    snprintf(n, sizeof(n), "%s.alpha_res", prefix);
    spec(n, CLS_EXACT, T_F32, 1, 1, 0, 0, 0);
    snprintf(n, sizeof(n), "%s.bias_pre", prefix);
    spec(n, CLS_EXACT, T_F32, 1, hc, 0, 0, 0);
    snprintf(n, sizeof(n), "%s.bias_post", prefix);
    spec(n, CLS_EXACT, T_F32, 1, hc, 0, 0, 0);
    snprintf(n, sizeof(n), "%s.bias_res", prefix);
    spec(n, CLS_EXACT, T_F32, 2, hc, hc, 0, 0);
}

static void motif_attention(const char *prefix, const layout_shape *s, bool include_norm)
{
    uint64_t e = s->n_embd;
    uint64_t q_dim = (uint64_t)s->n_head * s->n_head_dim;
    uint64_t signal_heads = (uint64_t)(s->n_head - s->n_noise_head);
    uint64_t signal_value_dim = signal_heads * s->n_value_dim;
    uint64_t kv_b_dim = (uint64_t)s->n_head_kv *
                        ((s->n_head_dim - s->n_rot) + s->n_value_dim);
    char n[160];
    if (include_norm) {
        snprintf(n, sizeof(n), "%sattn_norm.weight", prefix);
        spec(n, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    }
    snprintf(n, sizeof(n), "%sattn_q_a.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, e, s->n_lora_q, 0, 0);
    snprintf(n, sizeof(n), "%sattn_q_a_norm.weight", prefix);
    spec(n, CLS_EXACT, T_F32, 1, s->n_lora_q, 0, 0, 0);
    snprintf(n, sizeof(n), "%sattn_q_b.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, s->n_lora_q, q_dim, 0, 0);
    snprintf(n, sizeof(n), "%sattn_q_gate.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, s->n_lora_q, signal_value_dim, 0, 0);
    snprintf(n, sizeof(n), "%sattn_kv_a.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, e, (uint64_t)s->n_kv_lora + s->n_rot, 0, 0);
    snprintf(n, sizeof(n), "%sattn_kv_a_norm.weight", prefix);
    spec(n, CLS_EXACT, T_F32, 1, s->n_kv_lora, 0, 0, 0);
    snprintf(n, sizeof(n), "%sattn_kv_b.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, s->n_kv_lora, kv_b_dim, 0, 0);
    snprintf(n, sizeof(n), "%sattn_lambda.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, e, signal_heads, 0, 0);
    snprintf(n, sizeof(n), "%sattn_output.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, signal_value_dim, e, 0, 0);
}

static void motif_dense_ffn(const char *prefix, const layout_shape *s)
{
    uint64_t e = s->n_embd;
    uint64_t ff = s->n_ff_dense;
    char n[160];
    snprintf(n, sizeof(n), "%sffn_gate.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, e, ff, 0, 0);
    snprintf(n, sizeof(n), "%sffn_up.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, e, ff, 0, 0);
    snprintf(n, sizeof(n), "%sffn_down.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, ff, e, 0, 0);
    snprintf(n, sizeof(n), "%sffn_polynorm.weight", prefix);
    spec(n, CLS_EXACT, T_F32, 1, 3, 0, 0, 0);
    snprintf(n, sizeof(n), "%sffn_polynorm.bias", prefix);
    spec(n, CLS_EXACT, T_F32, 1, 1, 0, 0, 0);
}

static void expected_motif3(const layout_shape *s)
{
    uint64_t e = s->n_embd;
    uint32_t il;
    spec("token_embd.weight", CLS_EXACT, T_Q8_0, 2, e, s->n_vocab, 0, 0);
    spec("output_norm.weight", CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    spec("output.weight", CLS_EXACT, T_Q8_0, 2, e, s->n_vocab, 0, 0);
    for (il = 0; il < s->n_layer; il++) {
        char p[32], mhc[40];
        snprintf(p, sizeof(p), "blk.%u.", il);
        snprintf(mhc, sizeof(mhc), "blk.%u.mhc_attn", il);
        motif_mhc(mhc, s);
        motif_attention(p, s, true);
        snprintf(mhc, sizeof(mhc), "blk.%u.mhc_ffn", il);
        motif_mhc(mhc, s);
        specf("blk.%u.ffn_norm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
        if (il < s->n_leading_dense) {
            motif_dense_ffn(p, s);
        } else {
            specf("blk.%u.ffn_gate_inp.weight", il, CLS_EXACT, T_F32, 2, e, s->n_expert, 0, 0);
            specf("blk.%u.exp_probs_b.bias", il, CLS_EXACT, T_F32, 1, s->n_expert, 0, 0, 0);
            specf("blk.%u.ffn_gate_exps.weight", il, CLS_EXACT, T_IQ2_XXS, 3, e, s->n_ff_exp, s->n_expert, 0);
            specf("blk.%u.ffn_up_exps.weight", il, CLS_EXACT, T_IQ2_XXS, 3, e, s->n_ff_exp, s->n_expert, 0);
            specf("blk.%u.ffn_down_exps.weight", il, CLS_EXACT, T_Q2_K, 3, s->n_ff_exp, e, s->n_expert, 0);
            specf("blk.%u.ffn_polynorm_exps.weight", il, CLS_EXACT, T_F32, 2, 3, s->n_expert, 0, 0);
            specf("blk.%u.ffn_polynorm_exps.bias", il, CLS_EXACT, T_F32, 2, 1, s->n_expert, 0, 0);
            specf("blk.%u.ffn_gate_shexp.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_ff_exp, 0, 0);
            specf("blk.%u.ffn_up_shexp.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_ff_exp, 0, 0);
            specf("blk.%u.ffn_down_shexp.weight", il, CLS_EXACT, T_Q8_0, 2, s->n_ff_exp, e, 0, 0);
            specf("blk.%u.ffn_polynorm_shexp.weight", il, CLS_EXACT, T_F32, 1, 3, 0, 0, 0);
            specf("blk.%u.ffn_polynorm_shexp.bias", il, CLS_EXACT, T_F32, 1, 1, 0, 0, 0);
        }
    }
    spec("mtp.0.embed_norm.weight", CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    spec("mtp.0.input_layernorm.weight", CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    spec("mtp.0.input_proj.weight", CLS_EXACT, T_Q8_0, 2, 2 * e, e, 0, 0);
    spec("mtp.0.final_layernorm.weight", CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    motif_attention("mtp.0.", s, false);
    spec("mtp.0.post_attention_layernorm.weight", CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    motif_dense_ffn("mtp.0.", s);
}

static void expected_dots3(const layout_shape *s)
{
    uint64_t e = s->n_embd;
    uint32_t il;
    spec("token_embd.weight", CLS_EXACT, T_Q8_0, 2, e, s->n_vocab, 0, 0);
    spec("output_norm.weight", CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    spec("output.weight", CLS_EXACT, T_Q8_0, 2, e, s->n_vocab, 0, 0);
    for (il = 0; il < s->n_layer; il++) {
        bool full = dots3_layer_is_full_attention(s, il);
        uint64_t heads = full ? s->n_head : s->n_swa_head;
        uint64_t kv_lora = full ? s->n_kv_lora : s->n_swa_kv_lora;
        uint64_t qk_dim = full ? s->n_key_mla : s->n_swa_key_mla;
        uint64_t nope = qk_dim - s->n_rot;
        uint64_t v_dim = s->n_value_mla;
        specf("blk.%u.attn_norm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
        specf("blk.%u.attn_q_a.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_lora_q, 0, 0);
        specf("blk.%u.attn_q_a_norm.weight", il, CLS_EXACT, T_Q8_0, 1, s->n_lora_q, 0, 0, 0);
        specf("blk.%u.attn_q_b.weight", il, CLS_EXACT, T_Q8_0, 2, s->n_lora_q, heads * qk_dim, 0, 0);
        specf("blk.%u.attn_kv_a_mqa.weight", il, CLS_EXACT, T_Q8_0, 2, e, kv_lora + s->n_rot, 0, 0);
        specf("blk.%u.attn_kv_a_norm.weight", il, CLS_EXACT, T_Q8_0, 1, kv_lora, 0, 0, 0);
        specf("blk.%u.attn_kv_b.weight", il, CLS_EXACT, T_Q8_0, 2, kv_lora, heads * (nope + v_dim), 0, 0);
        specf("blk.%u.attn_k_rope_norm.weight", il, CLS_EXACT, T_Q8_0, 1, s->n_rot, 0, 0, 0);
        specf("blk.%u.attn_gate.weight", il, CLS_EXACT, T_Q8_0, 2, e, heads, 0, 0);
        specf("blk.%u.attn_output.weight", il, CLS_EXACT, T_Q8_0, 2, heads * v_dim, e, 0, 0);
        if (full) {
            specf("blk.%u.attn_idx_q_b.weight", il, CLS_EXACT, T_Q8_0, 2,
                  s->n_lora_q, (uint64_t)s->n_indexer_head * s->n_indexer_head_dim, 0, 0);
            specf("blk.%u.attn_idx_k.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_indexer_head_dim, 0, 0);
            specf("blk.%u.attn_idx_w.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_indexer_head, 0, 0);
            specf("blk.%u.attn_idx_k_norm.weight", il, CLS_EXACT, T_F32, 1, s->n_indexer_head_dim, 0, 0, 0);
            specf("blk.%u.attn_idx_k_norm.bias", il, CLS_EXACT, T_F32, 1, s->n_indexer_head_dim, 0, 0, 0);
        }
        specf("blk.%u.ffn_norm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
        if (il < s->n_leading_dense || is_nextn(s, il)) {
            specf("blk.%u.ffn_gate.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_ff_dense, 0, 0);
            specf("blk.%u.ffn_up.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_ff_dense, 0, 0);
            specf("blk.%u.ffn_down.weight", il, CLS_EXACT, T_Q8_0, 2, s->n_ff_dense, e, 0, 0);
        } else {
            specf("blk.%u.ffn_gate_inp.weight", il, CLS_EXACT, T_F32, 2, e, s->n_expert, 0, 0);
            specf("blk.%u.exp_probs_b.bias", il, CLS_EXACT, T_F32, 1, s->n_expert, 0, 0, 0);
            specf("blk.%u.ffn_gate_exps.weight", il, CLS_EXACT, T_IQ2_XXS, 3, e, s->n_ff_exp, s->n_expert, 0);
            specf("blk.%u.ffn_up_exps.weight", il, CLS_EXACT, T_IQ2_XXS, 3, e, s->n_ff_exp, s->n_expert, 0);
            specf("blk.%u.ffn_down_exps.weight", il, CLS_EXACT, T_Q2_K, 3, s->n_ff_exp, e, s->n_expert, 0);
            specf("blk.%u.ffn_gate_shexp.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_ff_exp, 0, 0);
            specf("blk.%u.ffn_up_shexp.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_ff_exp, 0, 0);
            specf("blk.%u.ffn_down_shexp.weight", il, CLS_EXACT, T_Q8_0, 2, s->n_ff_exp, e, 0, 0);
        }
        if (is_nextn(s, il)) {
            specf("blk.%u.eh_proj.weight", il, CLS_EXACT, T_Q8_0, 2, 2 * e, e, 0, 0);
            specf("blk.%u.enorm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
            specf("blk.%u.hnorm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
            specf("blk.%u.shared_head_norm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
        }
    }
    spec("token_embd_mtp.weight", CLS_EXACT, T_Q8_0, 2, e, s->n_vocab, 0, 0);
}

static void expected_solar(const layout_shape *s)
{
    uint64_t e = s->n_embd;
    uint64_t q_dim = (uint64_t)s->n_head * s->n_head_dim;
    uint64_t kv_dim = (uint64_t)s->n_head_kv * s->n_head_dim;
    uint64_t kda_dim = (uint64_t)s->n_head * s->n_kda_head_dim;
    uint32_t il;
    spec("token_embd.weight", CLS_EXACT, T_Q8_0, 2, e, s->n_vocab, 0, 0);
    spec("output_norm.weight", CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    spec("output.weight", CLS_EXACT, T_Q8_0, 2, e, s->n_vocab, 0, 0);
    for (il = 0; il < s->n_layer; il++) {
        specf("blk.%u.attn_norm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
        specf("blk.%u.ffn_norm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
        if (solar_layer_is_gqa(s, il)) {
            specf("blk.%u.attn_q.weight", il, CLS_EXACT, T_Q8_0, 2, e, q_dim, 0, 0);
            specf("blk.%u.attn_k.weight", il, CLS_EXACT, T_Q8_0, 2, e, kv_dim, 0, 0);
            specf("blk.%u.attn_v.weight", il, CLS_EXACT, T_Q8_0, 2, e, kv_dim, 0, 0);
            specf("blk.%u.attn_gate.weight", il, CLS_EXACT, T_Q8_0, 2, e, q_dim, 0, 0);
            specf("blk.%u.attn_output.weight", il, CLS_EXACT, T_Q8_0, 2, q_dim, e, 0, 0);
        } else {
            specf("blk.%u.attn_q.weight", il, CLS_EXACT, T_Q8_0, 2, e, kda_dim, 0, 0);
            specf("blk.%u.attn_k.weight", il, CLS_EXACT, T_Q8_0, 2, e, kda_dim, 0, 0);
            specf("blk.%u.attn_v.weight", il, CLS_EXACT, T_Q8_0, 2, e, kda_dim, 0, 0);
            specf("blk.%u.ssm_conv1d_q.weight", il, CLS_SOLAR_CONV, T_F32, 3, s->n_ssm_conv, 1, kda_dim, 0);
            specf("blk.%u.ssm_conv1d_k.weight", il, CLS_SOLAR_CONV, T_F32, 3, s->n_ssm_conv, 1, kda_dim, 0);
            specf("blk.%u.ssm_conv1d_v.weight", il, CLS_SOLAR_CONV, T_F32, 3, s->n_ssm_conv, 1, kda_dim, 0);
            specf("blk.%u.ssm_f_a.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_kda_head_dim, 0, 0);
            specf("blk.%u.ssm_f_b.weight", il, CLS_EXACT, T_Q8_0, 2, s->n_kda_head_dim, kda_dim, 0, 0);
            specf("blk.%u.ssm_beta.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_head, 0, 0);
            specf("blk.%u.ssm_a", il, CLS_SOLAR_DECAY, T_F32, 2, 1, s->n_head, 0, 0);
            specf("blk.%u.ssm_dt.bias", il, CLS_EXACT, T_F32, 1, kda_dim, 0, 0, 0);
            specf("blk.%u.ssm_g_a.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_kda_head_dim, 0, 0);
            specf("blk.%u.ssm_g_b.weight", il, CLS_EXACT, T_Q8_0, 2, s->n_kda_head_dim, kda_dim, 0, 0);
            specf("blk.%u.ssm_norm.weight", il, CLS_EXACT, T_F32, 1, s->n_kda_head_dim, 0, 0, 0);
            specf("blk.%u.attn_output.weight", il, CLS_EXACT, T_Q8_0, 2, kda_dim, e, 0, 0);
        }
        specf("blk.%u.ffn_gate_inp.weight", il, CLS_EXACT, T_F32, 2, e, s->n_expert, 0, 0);
        specf("blk.%u.exp_probs_b.bias", il, CLS_EXACT, T_F32, 1, s->n_expert, 0, 0, 0);
        specf("blk.%u.ffn_gate_exps.weight", il, CLS_SOLAR_GATEUP, 0, 3, e, s->n_ff_exp, s->n_expert, 0);
        specf("blk.%u.ffn_up_exps.weight", il, CLS_SOLAR_GATEUP, 0, 3, e, s->n_ff_exp, s->n_expert, 0);
        specf("blk.%u.ffn_down_exps.weight", il, CLS_SOLAR_DOWN, 0, 3, s->n_ff_exp, e, s->n_expert, 0);
        specf("blk.%u.ffn_gate_shexp.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_ff_shexp, 0, 0);
        specf("blk.%u.ffn_up_shexp.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_ff_shexp, 0, 0);
        specf("blk.%u.ffn_down_shexp.weight", il, CLS_EXACT, T_Q8_0, 2, s->n_ff_shexp, e, 0, 0);
    }
}

static void expected_exaone(const layout_shape *s)
{
    uint64_t e = s->n_embd;
    uint64_t q_dim = (uint64_t)s->n_head * s->n_head_dim;
    uint64_t kv_dim = (uint64_t)s->n_head_kv * s->n_head_dim;
    uint32_t il;
    spec("token_embd.weight", CLS_EXAONE, 0, 2, e, s->n_vocab, 0, 0);
    spec("output_norm.weight", CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    spec("output.weight", CLS_EXAONE, 0, 2, e, s->n_vocab, 0, 0);
    for (il = 0; il < s->n_layer; il++) {
        specf("blk.%u.attn_norm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
        specf("blk.%u.attn_q.weight", il, CLS_EXAONE, 0, 2, e, q_dim, 0, 0);
        specf("blk.%u.attn_k.weight", il, CLS_EXAONE, 0, 2, e, kv_dim, 0, 0);
        specf("blk.%u.attn_v.weight", il, CLS_EXAONE, 0, 2, e, kv_dim, 0, 0);
        specf("blk.%u.attn_output.weight", il, CLS_EXAONE, 0, 2, q_dim, e, 0, 0);
        if (s->use_qk_norm) {
            specf("blk.%u.attn_q_norm.weight", il, CLS_EXACT, T_F32, 1, s->n_head_dim, 0, 0, 0);
            specf("blk.%u.attn_k_norm.weight", il, CLS_EXACT, T_F32, 1, s->n_head_dim, 0, 0, 0);
        }
        specf("blk.%u.ffn_norm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
        if (il < s->n_leading_dense || is_nextn(s, il)) {
            specf("blk.%u.ffn_gate.weight", il, CLS_EXAONE, 0, 2, e, s->n_ff_dense, 0, 0);
            specf("blk.%u.ffn_up.weight", il, CLS_EXAONE, 0, 2, e, s->n_ff_dense, 0, 0);
            specf("blk.%u.ffn_down.weight", il, CLS_EXAONE, 0, 2, s->n_ff_dense, e, 0, 0);
        } else {
            specf("blk.%u.ffn_gate_inp.weight", il, CLS_EXACT, T_F32, 2, e, s->n_expert, 0, 0);
            specf("blk.%u.exp_probs_b.bias", il, CLS_EXACT, T_F32, 1, s->n_expert, 0, 0, 0);
            specf("blk.%u.ffn_gate_exps.weight", il, CLS_EXAONE, 0, 3, e, s->n_ff_exp, s->n_expert, 0);
            specf("blk.%u.ffn_up_exps.weight", il, CLS_EXAONE, 0, 3, e, s->n_ff_exp, s->n_expert, 0);
            specf("blk.%u.ffn_down_exps.weight", il, CLS_EXAONE, 0, 3, s->n_ff_exp, e, s->n_expert, 0);
            specf("blk.%u.ffn_gate_shexp.weight", il, CLS_EXAONE, 0, 2, e, s->n_ff_shexp, 0, 0);
            specf("blk.%u.ffn_up_shexp.weight", il, CLS_EXAONE, 0, 2, e, s->n_ff_shexp, 0, 0);
            specf("blk.%u.ffn_down_shexp.weight", il, CLS_EXAONE, 0, 2, s->n_ff_shexp, e, 0, 0);
        }
        if (is_nextn(s, il)) {
            specf("blk.%u.nextn.eh_proj.weight", il, CLS_EXAONE, 0, 2, 2 * e, e, 0, 0);
            specf("blk.%u.nextn.enorm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
            specf("blk.%u.nextn.hnorm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
            specf("blk.%u.nextn.shared_head_norm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
        }
    }
}

static void qwen_spec(const char *prefix, const char *suffix,
                      type_class cls, uint32_t typ, uint32_t ndim,
                      uint64_t d0, uint64_t d1, uint64_t d2)
{
    char name[160];
    snprintf(name, sizeof(name), "%s.%s", prefix, suffix);
    spec(name, cls, typ, ndim, d0, d1, d2, 0);
}

static void qwen_hc(const char *prefix, const layout_shape *s, bool inject)
{
    uint64_t hc_dim = (uint64_t)s->n_embd * s->n_hc;
    qwen_spec(prefix, "norm.weight", CLS_QWEN_PLAIN, 0, 1, hc_dim, 0, 0);
    qwen_spec(prefix, "mix_down.weight", CLS_EXACT, T_BF16, 2, hc_dim, 320, 0);
    qwen_spec(prefix, "mix_up.weight", CLS_EXACT, T_BF16, 2, 320, hc_dim, 0);
    if (inject)
        qwen_spec(prefix, "inject.weight", CLS_EXACT, T_BF16, 2,
                  hc_dim, s->n_hc, 0);
}

static void qwen_linear_attention(const char *prefix, const layout_shape *s)
{
    uint64_t e = s->n_embd;
    uint64_t key_dim = 16u * s->n_kda_head_dim;
    uint64_t value_dim = 48u * s->n_kda_head_dim;
    uint64_t conv_dim = 2u * key_dim + value_dim;
    qwen_spec(prefix, "a_log", CLS_QWEN_PLAIN, 0, 1, 48, 0, 0);
    qwen_spec(prefix, "conv.weight", CLS_EXACT, T_BF16, 3,
              s->n_ssm_conv, 1, conv_dim);
    qwen_spec(prefix, "dt_bias", CLS_QWEN_PLAIN, 0, 1, 48, 0, 0);
    qwen_spec(prefix, "in_a.weight", CLS_QWEN_MATRIX, 0, 2, e, 48, 0);
    qwen_spec(prefix, "in_b.weight", CLS_QWEN_MATRIX, 0, 2, e, 48, 0);
    qwen_spec(prefix, "qkv.weight", CLS_QWEN_MATRIX, 0, 2, e, conv_dim, 0);
    qwen_spec(prefix, "z.weight", CLS_QWEN_MATRIX, 0, 2, e, value_dim, 0);
    qwen_spec(prefix, "norm.weight", CLS_QWEN_PLAIN, 0, 1,
              s->n_kda_head_dim, 0, 0);
    qwen_spec(prefix, "out.weight", CLS_QWEN_MATRIX, 0, 2, value_dim, e, 0);
}

static void qwen_qsa(const char *prefix, const layout_shape *s)
{
    uint64_t e = s->n_embd;
    uint64_t index_dim = (uint64_t)(s->n_indexer_head + 1u) * s->n_indexer_head_dim;
    uint64_t q_dim = 2u * (uint64_t)s->n_head * s->n_head_dim;
    uint64_t kv_dim = (uint64_t)s->n_head_kv * s->n_head_dim;
    uint64_t value_dim = (uint64_t)s->n_head * s->n_head_dim;
    qwen_spec(prefix, "attn_index_qk.weight", CLS_QWEN_MATRIX, 0, 2, e, index_dim, 0);
    qwen_spec(prefix, "attn_index_q_norm.weight", CLS_QWEN_PLAIN, 0, 1,
              s->n_indexer_head_dim, 0, 0);
    qwen_spec(prefix, "attn_index_k_norm.weight", CLS_QWEN_PLAIN, 0, 1,
              s->n_indexer_head_dim, 0, 0);
    qwen_spec(prefix, "attn_q.weight", CLS_QWEN_MATRIX, 0, 2, e, q_dim, 0);
    qwen_spec(prefix, "attn_q_norm.weight", CLS_QWEN_PLAIN, 0, 1,
              s->n_head_dim, 0, 0);
    qwen_spec(prefix, "attn_k.weight", CLS_QWEN_MATRIX, 0, 2, e, kv_dim, 0);
    qwen_spec(prefix, "attn_k_norm.weight", CLS_QWEN_PLAIN, 0, 1,
              s->n_head_dim, 0, 0);
    qwen_spec(prefix, "attn_v.weight", CLS_QWEN_MATRIX, 0, 2, e, kv_dim, 0);
    qwen_spec(prefix, "attn_output.weight", CLS_QWEN_MATRIX, 0, 2,
              value_dim, e, 0);
}

static void qwen_moe(const char *prefix, const layout_shape *s, bool mtp)
{
    type_class routed = mtp ? CLS_QWEN_MTP_ROUTED : CLS_QWEN_MATRIX;
    uint64_t e = s->n_embd, experts = s->n_expert;
    qwen_spec(prefix, "ffn_gate_inp.weight", CLS_QWEN_PLAIN, 0, 2, e, experts, 0);
    qwen_spec(prefix, "ffn_gate_exps.weight", routed, 0, 3,
              e, s->n_ff_exp, experts);
    qwen_spec(prefix, "ffn_up_exps.weight", routed, 0, 3,
              e, s->n_ff_exp, experts);
    qwen_spec(prefix, "ffn_down_exps.main.weight", CLS_QWEN_MATRIX, 0, 3,
              512, e, experts);
    qwen_spec(prefix, "ffn_down_exps.tail.weight", CLS_QWEN_MATRIX, 0, 3,
              128, e, experts);
    qwen_spec(prefix, "ffn_gate_shexp.weight", CLS_QWEN_MATRIX, 0, 2,
              e, s->n_ff_shexp, 0);
    qwen_spec(prefix, "ffn_up_shexp.weight", CLS_QWEN_MATRIX, 0, 2,
              e, s->n_ff_shexp, 0);
    qwen_spec(prefix, "ffn_down_shexp.weight", CLS_QWEN_MATRIX, 0, 2,
              s->n_ff_shexp, e, 0);
    qwen_spec(prefix, "ffn_shexp_gate_inp.weight", CLS_QWEN_PLAIN, 0, 2,
              e, 1, 0);
}

static void qwen_ple(const layout_shape *s)
{
    uint64_t e = s->n_embd;
    uint64_t hc_dim = e * s->n_hc;
    spec("blk.1.ple.conv.weight", CLS_EXACT, T_BF16, 3,
         s->n_ssm_conv, 1, hc_dim, 0);
    spec("blk.1.ple.key.weight", CLS_QWEN_MATRIX, 0, 2, e, hc_dim, 0, 0);
    spec("blk.1.ple.value.weight", CLS_QWEN_MATRIX, 0, 2, e, e, 0, 0);
    spec("blk.1.ple.conv_norm.weight", CLS_QWEN_PLAIN, 0, 1, hc_dim, 0, 0, 0);
    spec("blk.1.ple.key_norm.weight", CLS_QWEN_PLAIN, 0, 1, hc_dim, 0, 0, 0);
    spec("blk.1.ple.query_norm.weight", CLS_QWEN_PLAIN, 0, 1, hc_dim, 0, 0, 0);
    spec("blk.1.ple.layer_multipliers", CLS_EXACT, T_I64, 1, 3, 0, 0, 0);
    spec("blk.1.ple.head_offsets", CLS_EXACT, T_I64, 1, 16, 0, 0, 0);
    spec("blk.1.ple.head_vocab_sizes", CLS_EXACT, T_I64, 1, 16, 0, 0, 0);
}

static void qwen_vision(const layout_shape *s)
{
    const uint64_t h = 1152, ff = 4304, merged = 4u * 1152u;
    uint32_t il;
    spec5("vision.patch_embed.weight", CLS_EXACT, T_BF16, 16, 16, 2, 3, h);
    spec("vision.patch_embed.bias", CLS_EXACT, T_F32, 1, h, 0, 0, 0);
    spec("vision.position_embd.weight", CLS_EXACT, T_Q8_0, 2, h, 2304, 0, 0);
    for (il = 0; il < 27u; il++) {
        specf("vblk.%u.norm1.weight", il, CLS_EXACT, T_F32, 1, h, 0, 0, 0);
        specf("vblk.%u.norm1.bias", il, CLS_EXACT, T_F32, 1, h, 0, 0, 0);
        specf("vblk.%u.attn_qkv.weight", il, CLS_EXACT, T_Q8_0, 2, h, 3u * h, 0, 0);
        specf("vblk.%u.attn_qkv.bias", il, CLS_EXACT, T_F32, 1, 3u * h, 0, 0, 0);
        specf("vblk.%u.attn_output.weight", il, CLS_EXACT, T_Q8_0, 2, h, h, 0, 0);
        specf("vblk.%u.attn_output.bias", il, CLS_EXACT, T_F32, 1, h, 0, 0, 0);
        specf("vblk.%u.norm2.weight", il, CLS_EXACT, T_F32, 1, h, 0, 0, 0);
        specf("vblk.%u.norm2.bias", il, CLS_EXACT, T_F32, 1, h, 0, 0, 0);
        specf("vblk.%u.ffn_up.weight", il, CLS_EXACT, T_Q8_0, 2, h, ff, 0, 0);
        specf("vblk.%u.ffn_up.bias", il, CLS_EXACT, T_F32, 1, ff, 0, 0, 0);
        specf("vblk.%u.ffn_down.weight", il, CLS_EXACT, T_BF16, 2, ff, h, 0, 0);
        specf("vblk.%u.ffn_down.bias", il, CLS_EXACT, T_F32, 1, h, 0, 0, 0);
    }
    spec("vision.merger.norm.weight", CLS_EXACT, T_F32, 1, h, 0, 0, 0);
    spec("vision.merger.norm.bias", CLS_EXACT, T_F32, 1, h, 0, 0, 0);
    spec("vision.merger.ffn_up.weight", CLS_EXACT, T_Q8_0, 2,
         merged, merged, 0, 0);
    spec("vision.merger.ffn_up.bias", CLS_EXACT, T_F32, 1, merged, 0, 0, 0);
    spec("vision.merger.ffn_down.weight", CLS_EXACT, T_Q8_0, 2,
         merged, s->n_embd, 0, 0);
    spec("vision.merger.ffn_down.bias", CLS_EXACT, T_F32, 1,
         s->n_embd, 0, 0, 0);
}

static void expected_qwen(const layout_shape *s)
{
    uint64_t e = s->n_embd;
    uint32_t il;
    char prefix[64];
    spec("token_embd.weight", CLS_QWEN_MATRIX, 0, 2, e, s->n_vocab, 0, 0);
    spec("output.weight", CLS_QWEN_MATRIX, 0, 2, e, s->n_vocab, 0, 0);
    qwen_hc("hc_input", s, false);
    qwen_vision(s);
    for (il = 0; il < s->n_layer; il++) {
        snprintf(prefix, sizeof(prefix), "blk.%u.hc_attn", il);
        qwen_hc(prefix, s, true);
        snprintf(prefix, sizeof(prefix), "blk.%u", il);
        if (qwen4exp_layer_is_full_attention(s, il)) qwen_qsa(prefix, s);
        else {
            snprintf(prefix, sizeof(prefix), "blk.%u.linear_attn", il);
            qwen_linear_attention(prefix, s);
        }
        snprintf(prefix, sizeof(prefix), "blk.%u", il);
        qwen_moe(prefix, s, false);
        snprintf(prefix, sizeof(prefix), "blk.%u.hc_ffn", il);
        qwen_hc(prefix, s, true);
        if (il == 1u) qwen_ple(s);
    }
    spec("mtp.fc_embedding.weight", CLS_QWEN_MATRIX, 0, 2, e, e, 0, 0);
    spec("mtp.fc_hidden.weight", CLS_QWEN_MATRIX, 0, 2, e, e, 0, 0);
    spec("mtp.fc_embedding_norm.weight", CLS_QWEN_PLAIN, 0, 1, e, 0, 0, 0);
    spec("mtp.fc_hidden_norm.weight", CLS_QWEN_PLAIN, 0, 1,
         e * s->n_hc, 0, 0, 0);
    qwen_hc("mtp.hc_input", s, false);
    qwen_hc("mtp.blk.0.hc_attn", s, true);
    qwen_qsa("mtp.blk.0", s);
    qwen_moe("mtp.blk.0", s, true);
    qwen_hc("mtp.blk.0.hc_ffn", s, true);
}

static bool glm53_layer_is_kda(const layout_shape *s, uint32_t il)
{
    return !is_nextn(s, il) && (il % 4u) != 3u;
}

static void expected_glm53(const layout_shape *s)
{
    const uint64_t e = s->n_embd;
    const uint64_t hc_dim = (uint64_t)s->n_hc * e;
    const uint64_t hc_mix = 2u * s->n_hc + (uint64_t)s->n_hc * s->n_hc;
    const uint64_t kda = (uint64_t)s->n_head * s->n_kda_head_dim;
    const uint64_t q_dim = (uint64_t)s->n_head * s->n_key_mla;
    const uint64_t index_q = (uint64_t)s->n_indexer_head * s->n_indexer_head_dim;
    uint32_t il;

    spec("token_embd.weight", CLS_GLM_DENSE, 0, 2, e, s->n_vocab, 0, 0);
    spec("output_norm.weight", CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    spec("output.weight", CLS_GLM_DENSE, 0, 2, e, s->n_vocab, 0, 0);

    for (il = 0; il < s->n_layer; il++) {
        specf("blk.%u.attn_norm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
        specf("blk.%u.ffn_norm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
        if (!is_nextn(s, il)) {
            specf("blk.%u.hc_attn_fn.weight", il, CLS_EXACT, T_BF16, 2, hc_dim, hc_mix, 0, 0);
            specf("blk.%u.hc_attn_scale.weight", il, CLS_EXACT, T_F32, 1, 3, 0, 0, 0);
            specf("blk.%u.hc_attn_base.weight", il, CLS_EXACT, T_F32, 1, hc_mix, 0, 0, 0);
            specf("blk.%u.hc_ffn_fn.weight", il, CLS_EXACT, T_BF16, 2, hc_dim, hc_mix, 0, 0);
            specf("blk.%u.hc_ffn_scale.weight", il, CLS_EXACT, T_F32, 1, 3, 0, 0, 0);
            specf("blk.%u.hc_ffn_base.weight", il, CLS_EXACT, T_F32, 1, hc_mix, 0, 0, 0);
        }

        if (glm53_layer_is_kda(s, il)) {
            specf("blk.%u.kda_q.weight", il, CLS_GLM_DENSE, 0, 2, e, kda, 0, 0);
            specf("blk.%u.kda_k.weight", il, CLS_GLM_DENSE, 0, 2, e, kda, 0, 0);
            specf("blk.%u.kda_v.weight", il, CLS_GLM_DENSE, 0, 2, e, kda, 0, 0);
            specf("blk.%u.kda_f_a.weight", il, CLS_GLM_DENSE, 0, 2, e, s->n_kda_head_dim, 0, 0);
            specf("blk.%u.kda_f_b.weight", il, CLS_GLM_DENSE, 0, 2, s->n_kda_head_dim, kda, 0, 0);
            specf("blk.%u.kda_beta.weight", il, CLS_GLM_DENSE, 0, 2, e, s->n_head, 0, 0);
            specf("blk.%u.kda_g_a.weight", il, CLS_GLM_DENSE, 0, 2, e, s->n_kda_head_dim, 0, 0);
            specf("blk.%u.kda_g_b.weight", il, CLS_GLM_DENSE, 0, 2, s->n_kda_head_dim, kda, 0, 0);
            specf("blk.%u.kda_output.weight", il, CLS_GLM_DENSE, 0, 2, kda, e, 0, 0);
            specf("blk.%u.kda_q_conv.weight", il, CLS_EXACT, T_F32, 3, s->n_ssm_conv, 1, kda, 0);
            specf("blk.%u.kda_k_conv.weight", il, CLS_EXACT, T_F32, 3, s->n_ssm_conv, 1, kda, 0);
            specf("blk.%u.kda_v_conv.weight", il, CLS_EXACT, T_F32, 3, s->n_ssm_conv, 1, kda, 0);
            specf("blk.%u.kda_dt_bias.weight", il, CLS_EXACT, T_F32, 1, kda, 0, 0, 0);
            specf("blk.%u.kda_a_log.weight", il, CLS_EXACT, T_F32, 1, s->n_head, 0, 0, 0);
            specf("blk.%u.kda_o_norm.weight", il, CLS_EXACT, T_F32, 1, s->n_kda_head_dim, 0, 0, 0);
        } else {
            specf("blk.%u.attn_q_a.weight", il, CLS_GLM_DENSE, 0, 2, e, s->n_lora_q, 0, 0);
            specf("blk.%u.attn_q_b.weight", il, CLS_GLM_DENSE, 0, 2, s->n_lora_q, q_dim, 0, 0);
            specf("blk.%u.attn_kv_a_mqa.weight", il, CLS_GLM_DENSE, 0, 2, e, s->n_kv_lora, 0, 0);
            specf("blk.%u.attn_k_b.weight", il, CLS_GLM_DENSE, 0, 3, s->n_key_mla, s->n_kv_lora, s->n_head, 0);
            specf("blk.%u.attn_v_b.weight", il, CLS_GLM_DENSE, 0, 3, s->n_kv_lora, s->n_value_mla, s->n_head, 0);
            specf("blk.%u.attn_output.weight", il, CLS_GLM_DENSE, 0, 2, (uint64_t)s->n_head * s->n_value_mla, e, 0, 0);
            specf("blk.%u.indexer.attn_q_b.weight", il, CLS_GLM_DENSE, 0, 2, s->n_lora_q, index_q, 0, 0);
            specf("blk.%u.indexer.attn_k.weight", il, CLS_GLM_DENSE, 0, 2, e, s->n_indexer_head_dim, 0, 0);
            specf("blk.%u.indexer.proj.weight", il, CLS_GLM_DENSE, 0, 2, e, s->n_indexer_head, 0, 0);
            specf("blk.%u.indexer.k_norm.weight", il, CLS_EXACT, T_F32, 1, s->n_indexer_head_dim, 0, 0, 0);
            specf("blk.%u.indexer.k_norm.bias", il, CLS_EXACT, T_F32, 1, s->n_indexer_head_dim, 0, 0, 0);
            specf("blk.%u.indexer.pool_ape.weight", il, CLS_EXACT, T_BF16, 2, s->n_indexer_head_dim, 4, 0, 0);
            specf("blk.%u.indexer.pool_gate.weight", il, CLS_EXACT, T_BF16, 2, e, s->n_indexer_head_dim, 0, 0);
            specf("blk.%u.attn_q_a_norm.weight", il, CLS_EXACT, T_F32, 1, s->n_lora_q, 0, 0, 0);
            specf("blk.%u.attn_kv_a_norm.weight", il, CLS_EXACT, T_F32, 1, s->n_kv_lora, 0, 0, 0);
        }

        if (il < s->n_leading_dense) {
            specf("blk.%u.ffn_gate.weight", il, CLS_GLM_DENSE, 0, 2, e, s->n_ff_dense, 0, 0);
            specf("blk.%u.ffn_up.weight", il, CLS_GLM_DENSE, 0, 2, e, s->n_ff_dense, 0, 0);
            specf("blk.%u.ffn_down.weight", il, CLS_GLM_DENSE, 0, 2, s->n_ff_dense, e, 0, 0);
        } else {
            specf("blk.%u.ffn_gate_inp.weight", il, CLS_EXACT, T_F32, 2, e, s->n_expert, 0, 0);
            specf("blk.%u.exp_probs_b.bias", il, CLS_EXACT, T_F32, 1, s->n_expert, 0, 0, 0);
            specf("blk.%u.ffn_gate_exps.weight", il, CLS_ROUTED, 0, 3, e, s->n_ff_exp, s->n_expert, 0);
            specf("blk.%u.ffn_up_exps.weight", il, CLS_ROUTED, 0, 3, e, s->n_ff_exp, s->n_expert, 0);
            specf("blk.%u.ffn_down_exps.weight", il, CLS_ROUTED, 0, 3, s->n_ff_exp, e, s->n_expert, 0);
            specf("blk.%u.ffn_gate_shexp.weight", il, CLS_GLM_DENSE, 0, 2, e, s->n_ff_exp, 0, 0);
            specf("blk.%u.ffn_up_shexp.weight", il, CLS_GLM_DENSE, 0, 2, e, s->n_ff_exp, 0, 0);
            specf("blk.%u.ffn_down_shexp.weight", il, CLS_GLM_DENSE, 0, 2, s->n_ff_exp, e, 0, 0);
        }
        if (is_nextn(s, il)) {
            specf("blk.%u.nextn.eh_proj.weight", il, CLS_GLM_DENSE, 0, 2, 2u * e, e, 0, 0);
            specf("blk.%u.nextn.enorm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
            specf("blk.%u.nextn.hnorm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
            specf("blk.%u.nextn.shared_head_norm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
        }
    }
}

static void expected_deepseek(const layout_shape *s)
{
    uint64_t e = s->n_embd;
    uint64_t hc = s->n_hc;
    uint64_t hc_dim = e * hc;
    uint64_t hc_mix = 2 * hc + hc * hc;
    uint64_t q_dim = (uint64_t)s->n_head * s->n_head_dim;
    uint64_t out_low = (uint64_t)s->n_out_group * s->n_lora_o;
    uint32_t il;
    spec("token_embd.weight", CLS_EXACT, T_F16, 2, e, s->n_vocab, 0, 0);
    spec("output_hc_base.weight", CLS_EXACT, T_F32, 1, hc, 0, 0, 0);
    spec("output_hc_fn.weight", CLS_EXACT, T_F16, 2, hc_dim, hc, 0, 0);
    spec("output_hc_scale.weight", CLS_EXACT, T_F32, 1, 1, 0, 0, 0);
    spec("output_norm.weight", CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    spec("output.weight", CLS_EXACT, T_Q8_0, 2, e, s->n_vocab, 0, 0);
    for (il = 0; il < s->n_layer; il++) {
        uint32_t ratio = expected_compress_ratio(s, il);
        specf("blk.%u.hc_attn_fn.weight", il, CLS_EXACT, T_F16, 2, hc_dim, hc_mix, 0, 0);
        specf("blk.%u.hc_attn_scale.weight", il, CLS_EXACT, T_F32, 1, 3, 0, 0, 0);
        specf("blk.%u.hc_attn_base.weight", il, CLS_EXACT, T_F32, 1, hc_mix, 0, 0, 0);
        specf("blk.%u.attn_norm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
        specf("blk.%u.attn_q_a.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_lora_q, 0, 0);
        specf("blk.%u.attn_q_a_norm.weight", il, CLS_EXACT, T_F32, 1, s->n_lora_q, 0, 0, 0);
        specf("blk.%u.attn_q_b.weight", il, CLS_EXACT, T_Q8_0, 2, s->n_lora_q, q_dim, 0, 0);
        specf("blk.%u.attn_kv.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_head_dim, 0, 0);
        specf("blk.%u.attn_kv_a_norm.weight", il, CLS_EXACT, T_F32, 1, s->n_head_dim, 0, 0, 0);
        specf("blk.%u.attn_sinks.weight", il, CLS_EXACT, T_F32, 1, s->n_head, 0, 0, 0);
        specf("blk.%u.attn_output_a.weight", il, CLS_EXACT, T_Q8_0, 2,
              (uint64_t)s->n_head_dim * (s->n_head / s->n_out_group), out_low, 0, 0);
        specf("blk.%u.attn_output_b.weight", il, CLS_EXACT, T_Q8_0, 2, out_low, e, 0, 0);
        if (ratio != 0) {
            uint64_t coff = ratio == 4 ? 2 : 1;
            uint64_t comp_width = coff * s->n_head_dim;
            specf("blk.%u.attn_compressor_ape.weight", il, CLS_EXACT, T_F16, 2, comp_width, ratio, 0, 0);
            specf("blk.%u.attn_compressor_kv.weight", il, CLS_EXACT, T_F16, 2, e, comp_width, 0, 0);
            specf("blk.%u.attn_compressor_gate.weight", il, CLS_EXACT, T_F16, 2, e, comp_width, 0, 0);
            specf("blk.%u.attn_compressor_norm.weight", il, CLS_EXACT, T_F32, 1, s->n_head_dim, 0, 0, 0);
        }
        if (ratio == 4) {
            uint64_t index_q = (uint64_t)s->n_indexer_head * s->n_indexer_head_dim;
            uint64_t index_width = 2u * s->n_indexer_head_dim;
            specf("blk.%u.indexer.attn_q_b.weight", il, CLS_EXACT, T_F16, 2, s->n_lora_q, index_q, 0, 0);
            specf("blk.%u.indexer.proj.weight", il, CLS_EXACT, T_F16, 2, e, s->n_indexer_head, 0, 0);
            specf("blk.%u.indexer_compressor_ape.weight", il, CLS_EXACT, T_F16, 2, index_width, 4, 0, 0);
            specf("blk.%u.indexer_compressor_kv.weight", il, CLS_EXACT, T_F16, 2, e, index_width, 0, 0);
            specf("blk.%u.indexer_compressor_gate.weight", il, CLS_EXACT, T_F16, 2, e, index_width, 0, 0);
            specf("blk.%u.indexer_compressor_norm.weight", il, CLS_EXACT, T_F32, 1, s->n_indexer_head_dim, 0, 0, 0);
        }
        specf("blk.%u.hc_ffn_fn.weight", il, CLS_EXACT, T_F16, 2, hc_dim, hc_mix, 0, 0);
        specf("blk.%u.hc_ffn_scale.weight", il, CLS_EXACT, T_F32, 1, 3, 0, 0, 0);
        specf("blk.%u.hc_ffn_base.weight", il, CLS_EXACT, T_F32, 1, hc_mix, 0, 0, 0);
        specf("blk.%u.ffn_norm.weight", il, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
        specf("blk.%u.ffn_gate_inp.weight", il, CLS_EXACT, T_F16, 2, e, s->n_expert, 0, 0);
        specf("blk.%u.exp_probs_b.bias", il, CLS_OPTIONAL, T_F32, 1, s->n_expert, 0, 0, 0);
        specf("blk.%u.ffn_gate_exps.weight", il, CLS_ROUTED, 0, 3, e, s->n_ff_exp, s->n_expert, 0);
        specf("blk.%u.ffn_up_exps.weight", il, CLS_ROUTED, 0, 3, e, s->n_ff_exp, s->n_expert, 0);
        specf("blk.%u.ffn_down_exps.weight", il, CLS_ROUTED, 0, 3, s->n_ff_exp, e, s->n_expert, 0);
        specf("blk.%u.ffn_gate_shexp.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_ff_exp, 0, 0);
        specf("blk.%u.ffn_up_shexp.weight", il, CLS_EXACT, T_Q8_0, 2, e, s->n_ff_exp, 0, 0);
        specf("blk.%u.ffn_down_shexp.weight", il, CLS_EXACT, T_Q8_0, 2, s->n_ff_exp, e, 0, 0);
        if (il < s->n_hash_layer) {
            specf("blk.%u.ffn_gate_tid2eid.weight", il, CLS_EXACT, T_I32, 2, s->n_expert_used, s->n_vocab, 0, 0);
        }
    }
}

static void dump_shape(const layout_shape *s)
{
    g_n = 0;
    printf("LAYOUT name=%s family=%u variant=%u n_layer=%u\n",
           s->name, s->family, s->variant, s->n_layer);
    switch (s->family) {
    case FAM_GLM53:
        expected_glm53(s);
        break;
    case FAM_QWEN:
        expected_qwen(s);
        break;
    case FAM_MOTIF3:
        expected_motif3(s);
        break;
    case FAM_DOTS3:
        expected_dots3(s);
        break;
    case FAM_SOLAR:
        expected_solar(s);
        break;
    case FAM_EXAONE:
        expected_exaone(s);
        break;
    default:
        expected_deepseek(s);
        break;
    }
    printf("COUNT n=%u\n", g_n);
}

typedef struct {
    const char *name;
    type_class cls;
    uint32_t typ;
    uint32_t ndim;
    uint64_t dim[4];
} expect_spec;

typedef struct {
    const char *name;
    uint32_t typ;
    uint32_t ndim;
    uint64_t dim[8];
} fake_t;

static bool type_ok2(const expect_spec *sp, uint32_t typ)
{
    switch (sp->cls) {
    case CLS_EXACT:
    case CLS_OPTIONAL:
        return typ == sp->typ;
    case CLS_PLAIN:
        return typ == T_F16 || typ == T_F32;
    case CLS_MOTIF_PROJ:
        return typ == T_F16 || typ == T_BF16;
    case CLS_ROUTED:
        return typ == T_IQ2_XXS || typ == T_Q2_K || typ == T_Q4_K;
    case CLS_EXAONE:
        return typ == T_Q8_0 || typ == T_Q6_K || typ == T_Q5_K || typ == T_Q4_K ||
               typ == T_Q3_K || typ == T_Q2_K || typ == T_IQ2_XXS || typ == T_F16 ||
               typ == T_F32;
    case CLS_SOLAR_GATEUP:
        return typ == T_Q4_K || typ == T_Q2_K || typ == T_IQ2_XXS;
    case CLS_SOLAR_DOWN:
        return typ == T_Q4_K || typ == T_Q3_K || typ == T_Q2_K;
    case CLS_SOLAR_CONV:
    case CLS_SOLAR_DECAY:
        return typ == T_F32;
    case CLS_QWEN_MATRIX:
        return typ == T_BF16 || typ == T_Q8_0 || typ == T_Q6_K ||
               typ == T_Q5_K || typ == T_Q5_0 || typ == T_Q4_K ||
               typ == T_Q4_0 || typ == T_Q3_K || typ == T_Q2_K ||
               typ == T_IQ2_XXS;
    case CLS_QWEN_PLAIN:
        return typ == T_F32 || typ == T_BF16;
    case CLS_QWEN_MTP_ROUTED:
        return typ == T_Q8_0 || typ == T_BF16;
    case CLS_GLM_DENSE:
        return typ == T_BF16 || typ == T_Q8_0 || typ == T_Q4_K || typ == T_Q4_0;
    }
    return false;
}

static const char *check_one(const expect_spec *sp, const fake_t *t)
{
    static char buf[128];
    uint32_t i;
    if (!t) {
        if (sp->cls == CLS_OPTIONAL) return "ok";
        snprintf(buf, sizeof(buf), "missing %s", sp->name);
        return buf;
    }
    if (!type_ok2(sp, t->typ)) {
        snprintf(buf, sizeof(buf), "type %s", sp->name);
        return buf;
    }
    if (sp->cls == CLS_SOLAR_CONV) {
        uint64_t d_inner = sp->dim[2];
        if (t->ndim == 4 && t->dim[0] == sp->dim[0] && t->dim[1] == 1 &&
            t->dim[2] == d_inner && t->dim[3] == 1) {
            return "ok";
        }
        if (t->ndim == 3 && t->dim[0] == sp->dim[0] && t->dim[1] == 1 &&
            t->dim[2] == d_inner) {
            return "ok";
        }
        snprintf(buf, sizeof(buf), "dim %s", sp->name);
        return buf;
    }
    if (sp->cls == CLS_SOLAR_DECAY) {
        if (t->ndim == 4 && t->dim[0] == 1 && t->dim[1] == sp->dim[1] &&
            t->dim[2] == 1 && t->dim[3] == 1) {
            return "ok";
        }
        if (t->ndim == 2 && t->dim[0] == 1 && t->dim[1] == sp->dim[1]) {
            return "ok";
        }
        snprintf(buf, sizeof(buf), "dim %s", sp->name);
        return buf;
    }
    if (t->ndim != sp->ndim) {
        snprintf(buf, sizeof(buf), "ndim %s", sp->name);
        return buf;
    }
    for (i = 0; i < sp->ndim; i++) {
        if (t->dim[i] != sp->dim[i]) {
            snprintf(buf, sizeof(buf), "dim %s", sp->name);
            return buf;
        }
    }
    return "ok";
}

static void dump_check(void)
{
    expect_spec embd = {"token_embd.weight", CLS_EXACT, T_Q8_0, 2, {4, 8, 0, 0}};
    expect_spec opt = {"exp_probs_b.bias", CLS_OPTIONAL, T_F32, 1, {4, 0, 0, 0}};
    expect_spec conv = {"ssm_conv.weight", CLS_SOLAR_CONV, T_F32, 3, {4, 1, 16, 0}};
    fake_t bad_ty = {"token_embd.weight", T_F32, 2, {4, 8}};
    fake_t bad_nd = {"token_embd.weight", T_Q8_0, 1, {4}};
    fake_t bad_dim = {"token_embd.weight", T_Q8_0, 2, {4, 7}};
    fake_t ok = {"token_embd.weight", T_Q8_0, 2, {4, 8}};
    fake_t conv3 = {"ssm_conv.weight", T_F32, 3, {4, 1, 16}};
    fake_t conv4 = {"ssm_conv.weight", T_F32, 4, {4, 1, 16, 1}};
    printf("missing %s\n", check_one(&embd, NULL));
    printf("optional %s\n", check_one(&opt, NULL));
    printf("type %s\n", check_one(&embd, &bad_ty));
    printf("ndim %s\n", check_one(&embd, &bad_nd));
    printf("dim %s\n", check_one(&embd, &bad_dim));
    printf("ok %s\n", check_one(&embd, &ok));
    printf("conv3 %s\n", check_one(&conv, &conv3));
    printf("conv4 %s\n", check_one(&conv, &conv4));
}

static void deepseek_block(const char *prefix, const layout_shape *s)
{
    uint64_t e = s->n_embd;
    uint64_t hc = s->n_hc;
    uint64_t hc_dim = e * hc;
    uint64_t hc_mix = 2 * hc + hc * hc;
    uint64_t q_dim = (uint64_t)s->n_head * s->n_head_dim;
    uint64_t out_low = (uint64_t)s->n_out_group * s->n_lora_o;
    char n[160];
    snprintf(n, sizeof(n), "%shc_attn_fn.weight", prefix);
    spec(n, CLS_PLAIN, 0, 2, hc_dim, hc_mix, 0, 0);
    snprintf(n, sizeof(n), "%shc_attn_scale.weight", prefix);
    spec(n, CLS_EXACT, T_F32, 1, 3, 0, 0, 0);
    snprintf(n, sizeof(n), "%shc_attn_base.weight", prefix);
    spec(n, CLS_EXACT, T_F32, 1, hc_mix, 0, 0, 0);
    snprintf(n, sizeof(n), "%sattn_norm.weight", prefix);
    spec(n, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    snprintf(n, sizeof(n), "%sattn_q_a.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, e, s->n_lora_q, 0, 0);
    snprintf(n, sizeof(n), "%sattn_q_a_norm.weight", prefix);
    spec(n, CLS_EXACT, T_F32, 1, s->n_lora_q, 0, 0, 0);
    snprintf(n, sizeof(n), "%sattn_q_b.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, s->n_lora_q, q_dim, 0, 0);
    snprintf(n, sizeof(n), "%sattn_kv.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, e, s->n_head_dim, 0, 0);
    snprintf(n, sizeof(n), "%sattn_kv_a_norm.weight", prefix);
    spec(n, CLS_EXACT, T_F32, 1, s->n_head_dim, 0, 0, 0);
    snprintf(n, sizeof(n), "%sattn_sinks.weight", prefix);
    spec(n, CLS_EXACT, T_F32, 1, s->n_head, 0, 0, 0);
    snprintf(n, sizeof(n), "%sattn_output_a.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2,
         (uint64_t)s->n_head_dim * (s->n_head / s->n_out_group), out_low, 0, 0);
    snprintf(n, sizeof(n), "%sattn_output_b.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, out_low, e, 0, 0);
    snprintf(n, sizeof(n), "%shc_ffn_fn.weight", prefix);
    spec(n, CLS_PLAIN, 0, 2, hc_dim, hc_mix, 0, 0);
    snprintf(n, sizeof(n), "%shc_ffn_scale.weight", prefix);
    spec(n, CLS_EXACT, T_F32, 1, 3, 0, 0, 0);
    snprintf(n, sizeof(n), "%shc_ffn_base.weight", prefix);
    spec(n, CLS_EXACT, T_F32, 1, hc_mix, 0, 0, 0);
    snprintf(n, sizeof(n), "%sffn_norm.weight", prefix);
    spec(n, CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    snprintf(n, sizeof(n), "%sffn_gate_inp.weight", prefix);
    spec(n, CLS_PLAIN, 0, 2, e, s->n_expert, 0, 0);
    snprintf(n, sizeof(n), "%sexp_probs_b.bias", prefix);
    spec(n, CLS_EXACT, T_F32, 1, s->n_expert, 0, 0, 0);
    snprintf(n, sizeof(n), "%sffn_gate_exps.weight", prefix);
    spec(n, CLS_ROUTED, 0, 3, e, s->n_ff_exp, s->n_expert, 0);
    snprintf(n, sizeof(n), "%sffn_up_exps.weight", prefix);
    spec(n, CLS_ROUTED, 0, 3, e, s->n_ff_exp, s->n_expert, 0);
    snprintf(n, sizeof(n), "%sffn_down_exps.weight", prefix);
    spec(n, CLS_ROUTED, 0, 3, s->n_ff_exp, e, s->n_expert, 0);
    snprintf(n, sizeof(n), "%sffn_gate_shexp.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, e, s->n_ff_exp, 0, 0);
    snprintf(n, sizeof(n), "%sffn_up_shexp.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, e, s->n_ff_exp, 0, 0);
    snprintf(n, sizeof(n), "%sffn_down_shexp.weight", prefix);
    spec(n, CLS_EXACT, T_Q8_0, 2, s->n_ff_exp, e, 0, 0);
}

static void dump_mtp(const layout_shape *s)
{
    uint64_t e = s->n_embd;
    uint64_t hc = s->n_hc;
    uint64_t hc_dim = e * hc;
    g_n = 0;
    printf("LAYOUT kind=mtp name=%s family=%u variant=%u\n",
           s->name, s->family, s->variant);
    spec("mtp.0.hc_head_base.weight", CLS_EXACT, T_F32, 1, hc, 0, 0, 0);
    spec("mtp.0.hc_head_fn.weight", CLS_PLAIN, 0, 2, hc_dim, hc, 0, 0);
    spec("mtp.0.hc_head_scale.weight", CLS_EXACT, T_F32, 1, 1, 0, 0, 0);
    spec("mtp.0.e_proj.weight", CLS_EXACT, T_Q8_0, 2, e, e, 0, 0);
    spec("mtp.0.h_proj.weight", CLS_EXACT, T_Q8_0, 2, e, e, 0, 0);
    spec("mtp.0.enorm.weight", CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    spec("mtp.0.hnorm.weight", CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    spec("mtp.0.norm.weight", CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    deepseek_block("mtp.0.", s);
    printf("COUNT n=%u\n", g_n);
}

static void dump_dspark(const layout_shape *s)
{
    uint64_t e = s->n_embd;
    uint64_t hc = s->n_hc;
    uint64_t hc_dim = e * hc;
    uint64_t rank = 256;
    uint32_t il;
    g_n = 0;
    printf("LAYOUT kind=dspark name=%s family=%u variant=%u markov_rank=256 n_layer=3\n",
           s->name, s->family, s->variant);
    spec("dspark.main_proj.weight", CLS_EXACT, T_Q8_0, 2, 3 * e, e, 0, 0);
    spec("dspark.main_norm.weight", CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    spec("dspark.markov_w1.weight", CLS_EXACT, T_F16, 2, rank, s->n_vocab, 0, 0);
    spec("dspark.markov_w2.weight", CLS_EXACT, T_F16, 2, rank, s->n_vocab, 0, 0);
    spec("dspark.conf_proj.weight", CLS_EXACT, T_F32, 2, e + rank, 1, 0, 0);
    spec("dspark.hc_head_fn.weight", CLS_PLAIN, 0, 2, hc_dim, hc, 0, 0);
    spec("dspark.hc_head_base.weight", CLS_EXACT, T_F32, 1, hc, 0, 0, 0);
    spec("dspark.hc_head_scale.weight", CLS_EXACT, T_F32, 1, 1, 0, 0, 0);
    spec("dspark.norm.weight", CLS_EXACT, T_F32, 1, e, 0, 0, 0);
    for (il = 0; il < 3; il++) {
        char p[32];
        snprintf(p, sizeof(p), "dspark.%u.", il);
        deepseek_block(p, s);
    }
    printf("COUNT n=%u\n", g_n);
}

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "check") == 0) {
        dump_check();
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "support") == 0) {
        dump_mtp(&SHAPE_FLASH);
        dump_dspark(&SHAPE_FLASH);
        dump_mtp(&SHAPE_PRO);
        dump_dspark(&SHAPE_PRO);
        return 0;
    }
    dump_shape(&SHAPE_FLASH);
    dump_shape(&SHAPE_PRO);
    dump_shape(&SHAPE_SOLAR);
    dump_shape(&SHAPE_MOTIF3);
    dump_shape(&SHAPE_KEXAONE);
    dump_shape(&SHAPE_DOTS3);
    dump_shape(&SHAPE_QWEN);
    dump_shape(&SHAPE_GLM53);
    return 0;
}
