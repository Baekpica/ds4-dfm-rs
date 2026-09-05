/* C g_ds4_shape catalog + DeepSeek select + architecture dispatch.
 * Copied from the v0.6.5-dfm Qwen golden so Rust can compare without linking
 * ds4.o. */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define DS4_DEFAULT_RMS_EPS (1.0e-6f)
#define DS4_DEFAULT_HC_EPS (1.0e-6f)
#define DS4_DEFAULT_SWIGLU_CLAMP_EXP (10.0f)
#define DS4_DEFAULT_ROPE_FREQ_BASE (10000.0f)
#define DS4_DEFAULT_ROPE_SCALE_FACTOR (16.0f)
#define DS4_DEFAULT_ROPE_YARN_BETA_FAST (32.0f)
#define DS4_DEFAULT_ROPE_YARN_BETA_SLOW (1.0f)
#define DS4_DEFAULT_COMPRESS_ROPE_FREQ_BASE (160000.0f)
#define DS4_DEFAULT_ROPE_ORIG_CTX UINT64_C(65536)

typedef enum {
    DS4_MODEL_FAMILY_DEEPSEEK4 = 0,
    DS4_MODEL_FAMILY_SOLAR_OPEN2 = 1,
    DS4_MODEL_FAMILY_MOTIF3 = 2,
    DS4_MODEL_FAMILY_EXAONE_MOE = 3,
    DS4_MODEL_FAMILY_DOTS3_NOTE = 4,
    DS4_MODEL_FAMILY_QWEN4EXP = 5,
    DS4_MODEL_FAMILY_GLM53 = 6,
} ds4_model_family;

typedef enum {
    DS4_VARIANT_FLASH = 0,
    DS4_VARIANT_PRO = 1,
    DS4_VARIANT_SOLAR_OPEN2_250B = 2,
    DS4_VARIANT_MOTIF3 = 3,
    DS4_VARIANT_KEXAONE_236B = 4,
    DS4_VARIANT_DOTS3_NOTE_PREV = 5,
    DS4_VARIANT_QWEN38_FLASH_NEXT = 6,
    DS4_VARIANT_GLM53_FLASH = 7,
    DS4_VARIANT_K2_HORIZON_375B = 8,
} ds4_variant;

typedef struct {
    const char *name;
    ds4_model_family family;
    ds4_variant variant;
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
    uint32_t n_expert_shared;
    uint32_t n_ff_exp;
    uint32_t n_ff_dense;
    uint32_t n_ff_shexp;
    uint32_t n_hash_layer;
    uint32_t n_swa;
    uint32_t n_swa_period;
    uint32_t n_indexer_head;
    uint32_t n_indexer_head_dim;
    uint32_t n_indexer_top_k;
    uint32_t n_hc;
    uint32_t n_hc_sinkhorn_iter;
    uint32_t n_nextn_predict;
    uint32_t n_leading_dense;
    uint32_t n_kv_lora;
    uint32_t n_key_mla;
    uint32_t n_value_mla;
    uint32_t n_swa_head;
    uint32_t n_swa_kv_lora;
    uint32_t n_swa_key_mla;
    uint32_t n_full_attn_count;
    uint32_t n_kda_head_dim;
    uint32_t n_ssm_conv;
    bool use_rope;
    bool use_qk_norm;
    float rms_eps;
    float kda_l2_eps;
    float kda_gate_clamp_min;
    float hc_eps;
    float expert_weight_scale;
    float swiglu_clamp_exp;
    float rope_freq_base;
    float rope_freq_base_swa;
    float rope_scale_factor;
    float rope_yarn_beta_fast;
    float rope_yarn_beta_slow;
    float compress_rope_freq_base;
    uint64_t rope_orig_ctx;
} ds4_shape;

static const ds4_shape DS4_SHAPE_FLASH = {
    .name = "DeepSeek V4 Flash",
    .family = DS4_MODEL_FAMILY_DEEPSEEK4,
    .variant = DS4_VARIANT_FLASH,
    .n_layer = 43,
    .n_embd = 4096,
    .n_vocab = 129280,
    .n_head = 64,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 512,
    .n_rot = 64,
    .n_out_group = 8,
    .n_lora_q = 1024,
    .n_lora_o = 1024,
    .n_expert = 256,
    .n_expert_used = 6,
    .n_expert_shared = 1,
    .n_ff_exp = 2048,
    .n_hash_layer = 3,
    .n_swa = 128,
    .n_indexer_head = 64,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 512,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .use_rope = true,
    .rms_eps = DS4_DEFAULT_RMS_EPS,
    .hc_eps = DS4_DEFAULT_HC_EPS,
    .expert_weight_scale = 1.5f,
    .swiglu_clamp_exp = DS4_DEFAULT_SWIGLU_CLAMP_EXP,
    .rope_freq_base = DS4_DEFAULT_ROPE_FREQ_BASE,
    .rope_scale_factor = DS4_DEFAULT_ROPE_SCALE_FACTOR,
    .rope_yarn_beta_fast = DS4_DEFAULT_ROPE_YARN_BETA_FAST,
    .rope_yarn_beta_slow = DS4_DEFAULT_ROPE_YARN_BETA_SLOW,
    .compress_rope_freq_base = DS4_DEFAULT_COMPRESS_ROPE_FREQ_BASE,
    .rope_orig_ctx = DS4_DEFAULT_ROPE_ORIG_CTX,
};

static const ds4_shape DS4_SHAPE_PRO = {
    .name = "DeepSeek V4 Pro",
    .family = DS4_MODEL_FAMILY_DEEPSEEK4,
    .variant = DS4_VARIANT_PRO,
    .n_layer = 61,
    .n_embd = 7168,
    .n_vocab = 129280,
    .n_head = 128,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 512,
    .n_rot = 64,
    .n_out_group = 16,
    .n_lora_q = 1536,
    .n_lora_o = 1024,
    .n_expert = 384,
    .n_expert_used = 6,
    .n_expert_shared = 1,
    .n_ff_exp = 3072,
    .n_hash_layer = 3,
    .n_swa = 128,
    .n_indexer_head = 64,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 1024,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .use_rope = true,
    .rms_eps = DS4_DEFAULT_RMS_EPS,
    .hc_eps = DS4_DEFAULT_HC_EPS,
    .expert_weight_scale = 2.5f,
    .swiglu_clamp_exp = DS4_DEFAULT_SWIGLU_CLAMP_EXP,
    .rope_freq_base = DS4_DEFAULT_ROPE_FREQ_BASE,
    .rope_scale_factor = DS4_DEFAULT_ROPE_SCALE_FACTOR,
    .rope_yarn_beta_fast = DS4_DEFAULT_ROPE_YARN_BETA_FAST,
    .rope_yarn_beta_slow = DS4_DEFAULT_ROPE_YARN_BETA_SLOW,
    .compress_rope_freq_base = DS4_DEFAULT_COMPRESS_ROPE_FREQ_BASE,
    .rope_orig_ctx = DS4_DEFAULT_ROPE_ORIG_CTX,
};

static const ds4_shape DS4_SHAPE_MOTIF3 = {
    .name = "Motif-3",
    .family = DS4_MODEL_FAMILY_MOTIF3,
    .variant = DS4_VARIANT_MOTIF3,
    .n_layer = 53,
    .n_embd = 4096,
    .n_vocab = 220160,
    .n_head = 80,
    .n_head_kv = 16,
    .n_noise_head = 16,
    .n_head_dim = 192,
    .n_value_dim = 128,
    .n_rot = 64,
    .n_out_group = 0,
    .n_lora_q = 1024,
    .n_lora_o = 0,
    .n_expert = 384,
    .n_expert_used = 8,
    .n_expert_shared = 1,
    .n_ff_exp = 1280,
    .n_ff_dense = 12288,
    .n_hash_layer = 0,
    .n_swa = 128,
    .n_swa_period = 4,
    .n_indexer_head = 0,
    .n_indexer_head_dim = 0,
    .n_indexer_top_k = 0,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .n_nextn_predict = 1,
    .n_leading_dense = 2,
    .n_kv_lora = 512,
    .n_key_mla = 192,
    .n_value_mla = 128,
    .rms_eps = 1.0e-5f,
    .hc_eps = 1.0e-6f,
    .expert_weight_scale = 2.0f,
    .swiglu_clamp_exp = 0.0f,
    .rope_freq_base = 10000.0f,
    .rope_scale_factor = 64.0f,
    .rope_yarn_beta_fast = 32.0f,
    .rope_yarn_beta_slow = 1.0f,
    .compress_rope_freq_base = 0.0f,
    .rope_orig_ctx = 4096,
};

static const ds4_shape DS4_SHAPE_SOLAR_OPEN2_250B = {
    .name = "Solar Open2 250B",
    .family = DS4_MODEL_FAMILY_SOLAR_OPEN2,
    .variant = DS4_VARIANT_SOLAR_OPEN2_250B,
    .n_layer = 48,
    .n_embd = 4096,
    .n_vocab = 196608,
    .n_head = 64,
    .n_head_kv = 8,
    .n_head_dim = 128,
    .n_value_dim = 128,
    .n_rot = 0,
    .n_expert = 320,
    .n_expert_used = 8,
    .n_expert_shared = 1,
    .n_ff_exp = 1280,
    .n_ff_dense = 10240,
    .n_ff_shexp = 1280,
    .n_kda_head_dim = 128,
    .n_ssm_conv = 4,
    .use_rope = false,
    .rms_eps = 1.0e-5f,
    .kda_l2_eps = 1.0e-6f,
    .kda_gate_clamp_min = -5.0f,
    .expert_weight_scale = 1.0f,
    .rope_freq_base = DS4_DEFAULT_ROPE_FREQ_BASE,
    .rope_scale_factor = 1.0f,
    .rope_orig_ctx = UINT64_C(1048576),
};

static const ds4_shape DS4_SHAPE_KEXAONE_236B = {
    .name = "K-EXAONE 236B A23B",
    .family = DS4_MODEL_FAMILY_EXAONE_MOE,
    .variant = DS4_VARIANT_KEXAONE_236B,
    .n_layer = 49,
    .n_embd = 6144,
    .n_vocab = 153600,
    .n_head = 64,
    .n_head_kv = 8,
    .n_head_dim = 128,
    .n_value_dim = 128,
    .n_rot = 128,
    .n_out_group = 0,
    .n_lora_q = 0,
    .n_lora_o = 0,
    .n_expert = 128,
    .n_expert_used = 8,
    .n_expert_shared = 1,
    .n_ff_exp = 2048,
    .n_ff_shexp = 2048,
    .n_ff_dense = 18432,
    .n_hash_layer = 0,
    .n_swa = 128,
    .n_swa_period = 4,
    .use_qk_norm = true,
    .n_indexer_head = 0,
    .n_indexer_head_dim = 0,
    .n_indexer_top_k = 0,
    .n_hc = 0,
    .n_hc_sinkhorn_iter = 0,
    .n_nextn_predict = 1,
    .n_leading_dense = 1,
    .n_kv_lora = 0,
    .n_key_mla = 0,
    .n_value_mla = 0,
    .rms_eps = 1.0e-5f,
    .hc_eps = 0.0f,
    .expert_weight_scale = 2.5f,
    .swiglu_clamp_exp = 0.0f,
    .rope_freq_base = 1000000.0f,
    .rope_scale_factor = 1.0f,
    .rope_yarn_beta_fast = 0.0f,
    .rope_yarn_beta_slow = 0.0f,
    .compress_rope_freq_base = 0.0f,
    .rope_orig_ctx = 262144,
};

static const ds4_shape DS4_SHAPE_DOTS3_NOTE_PREV = {
    .name = "dots3-note-prev",
    .family = DS4_MODEL_FAMILY_DOTS3_NOTE,
    .variant = DS4_VARIANT_DOTS3_NOTE_PREV,
    .n_layer = 47,
    .n_embd = 5120,
    .n_vocab = 152064,
    .n_head = 128,
    .n_head_kv = 128,
    .n_head_dim = 192,
    .n_value_dim = 128,
    .n_rot = 64,
    .n_out_group = 0,
    .n_lora_q = 1024,
    .n_lora_o = 0,
    .n_expert = 256,
    .n_expert_used = 8,
    .n_expert_shared = 1,
    .n_ff_exp = 1536,
    .n_ff_dense = 13824,
    .n_hash_layer = 0,
    .n_swa = 513,
    .n_swa_period = 4,
    .n_indexer_head = 64,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 2048,
    .n_hc = 0,
    .n_hc_sinkhorn_iter = 0,
    .n_nextn_predict = 1,
    .n_leading_dense = 1,
    .n_kv_lora = 512,
    .n_key_mla = 192,
    .n_value_mla = 128,
    .n_swa_head = 64,
    .n_swa_kv_lora = 1024,
    .n_swa_key_mla = 256,
    .n_full_attn_count = 13,
    .use_rope = true,
    .rms_eps = 1.0e-5f,
    .hc_eps = 0.0f,
    .expert_weight_scale = 1.0f,
    .swiglu_clamp_exp = 0.0f,
    .rope_freq_base = 80000000.0f,
    .rope_freq_base_swa = 50000.0f,
    .rope_scale_factor = 1.0f,
    .rope_yarn_beta_fast = 0.0f,
    .rope_yarn_beta_slow = 0.0f,
    .compress_rope_freq_base = 0.0f,
    .rope_orig_ctx = 524288,
};

static const ds4_shape DS4_SHAPE_QWEN38_FLASH_NEXT = {
    .name = "Qwen3.8-Flash-Next",
    .family = DS4_MODEL_FAMILY_QWEN4EXP,
    .variant = DS4_VARIANT_QWEN38_FLASH_NEXT,
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
    .n_expert_shared = 1,
    .n_ff_exp = 640,
    .n_ff_shexp = 640,
    .n_swa_period = 4,
    .n_indexer_head = 4,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 2048,
    .n_hc = 4,
    .n_nextn_predict = 1,
    .n_full_attn_count = 12,
    .n_kda_head_dim = 128,
    .n_ssm_conv = 4,
    .use_rope = true,
    .use_qk_norm = true,
    .rms_eps = 1.0e-6f,
    .hc_eps = 1.0e-6f,
    .expert_weight_scale = 1.0f,
    .rope_freq_base = 10000000.0f,
    .rope_scale_factor = 1.0f,
    .rope_orig_ctx = UINT64_C(262144),
};

static const ds4_shape DS4_SHAPE_GLM53_FLASH = {
    .name = "GLM 5.3 Flash",
    .family = DS4_MODEL_FAMILY_GLM53,
    .variant = DS4_VARIANT_GLM53_FLASH,
    .n_layer = 46,
    .n_embd = 4096,
    .n_vocab = 154880,
    .n_head = 64,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 256,
    .n_rot = 0,
    .n_lora_q = 1536,
    .n_expert = 288,
    .n_expert_used = 8,
    .n_expert_shared = 1,
    .n_ff_exp = 2048,
    .n_ff_dense = 12288,
    .n_indexer_head = 32,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 2048,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .n_nextn_predict = 1,
    .n_leading_dense = 3,
    .n_kv_lora = 512,
    .n_key_mla = 256,
    .n_value_mla = 256,
    .n_kda_head_dim = 128,
    .n_ssm_conv = 4,
    .use_rope = false,
    .rms_eps = 1.0e-5f,
    .kda_l2_eps = 1.0e-6f,
    .kda_gate_clamp_min = -5.0f,
    .hc_eps = 1.0e-6f,
    .expert_weight_scale = 2.5f,
    .swiglu_clamp_exp = 10.0f,
    .rope_scale_factor = 1.0f,
    .rope_orig_ctx = UINT64_C(1048576),
};

static const ds4_shape DS4_SHAPE_K2_HORIZON_375B = {
    .name = "K2-Horizon 375B A23B",
    .family = DS4_MODEL_FAMILY_EXAONE_MOE,
    .variant = DS4_VARIANT_K2_HORIZON_375B,
    .n_layer = 61,
    .n_embd = 6144,
    .n_vocab = 250624,
    .n_head = 48,
    .n_head_kv = 8,
    .n_head_dim = 128,
    .n_value_dim = 128,
    .n_rot = 64,
    .n_expert = 192,
    .n_expert_used = 8,
    .n_expert_shared = 1,
    .n_ff_exp = 1792,
    .n_ff_dense = 16384,
    .n_ff_shexp = 1792,
    .n_full_attn_count = 61,
    .n_nextn_predict = 0,
    .n_leading_dense = 3,
    .use_rope = true,
    .use_qk_norm = false,
    .rms_eps = 1.0e-6f,
    .expert_weight_scale = 2.5f,
    .rope_freq_base = 10000000.0f,
    .rope_scale_factor = 1.0f,
    .rope_orig_ctx = 524288,
};

static void dump_shape(const char *tag, const ds4_shape *s)
{
    uint32_t bits;
    printf("SHAPE\t%s\t%s\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u",
           tag, s->name,
           (unsigned)s->family, (unsigned)s->variant,
           s->n_layer, s->n_embd, s->n_vocab, s->n_head, s->n_head_kv,
           s->n_noise_head, s->n_head_dim, s->n_value_dim, s->n_rot,
           s->n_out_group, s->n_lora_q, s->n_lora_o, s->n_expert,
           s->n_expert_used, s->n_expert_shared, s->n_ff_exp, s->n_ff_dense,
           s->n_ff_shexp, s->n_hash_layer, s->n_swa, s->n_swa_period,
           s->n_indexer_head, s->n_indexer_head_dim, s->n_indexer_top_k,
           s->n_hc, s->n_hc_sinkhorn_iter, s->n_nextn_predict,
           s->n_leading_dense, s->n_kv_lora, s->n_key_mla, s->n_value_mla,
           s->n_swa_head, s->n_swa_kv_lora, s->n_swa_key_mla,
           s->n_full_attn_count, s->n_kda_head_dim, s->n_ssm_conv,
           (unsigned)s->use_rope, (unsigned)s->use_qk_norm);
#define DUMPF(field) do { memcpy(&bits, &s->field, 4); printf("\t%08x", bits); } while (0)
    DUMPF(rms_eps);
    DUMPF(kda_l2_eps);
    DUMPF(kda_gate_clamp_min);
    DUMPF(hc_eps);
    DUMPF(expert_weight_scale);
    DUMPF(swiglu_clamp_exp);
    DUMPF(rope_freq_base);
    DUMPF(rope_freq_base_swa);
    DUMPF(rope_scale_factor);
    DUMPF(rope_yarn_beta_fast);
    DUMPF(rope_yarn_beta_slow);
    DUMPF(compress_rope_freq_base);
#undef DUMPF
    printf("\t%llu\n", (unsigned long long)s->rope_orig_ctx);
}

static bool shape_matches(const ds4_shape *s,
                          uint32_t n_layer, uint32_t n_embd, uint32_t n_vocab,
                          uint32_t n_head, uint32_t n_head_kv, uint32_t n_head_dim,
                          uint32_t n_value_dim, uint32_t n_rot, uint32_t n_lora_q,
                          uint32_t n_lora_o, uint32_t n_out_group, uint32_t n_expert,
                          uint32_t n_expert_used, uint32_t n_ff_exp,
                          uint32_t n_expert_shared, uint32_t n_hash_layer,
                          uint32_t n_swa, uint32_t n_indexer_head,
                          uint32_t n_indexer_head_dim, uint32_t n_indexer_top_k,
                          uint32_t n_hc, uint32_t n_hc_sinkhorn_iter)
{
    return s->n_layer == n_layer && s->n_embd == n_embd && s->n_vocab == n_vocab &&
           s->n_head == n_head && s->n_head_kv == n_head_kv &&
           s->n_head_dim == n_head_dim && s->n_value_dim == n_value_dim &&
           s->n_rot == n_rot && s->n_lora_q == n_lora_q && s->n_lora_o == n_lora_o &&
           s->n_out_group == n_out_group && s->n_expert == n_expert &&
           s->n_expert_used == n_expert_used && s->n_ff_exp == n_ff_exp &&
           s->n_expert_shared == n_expert_shared && s->n_hash_layer == n_hash_layer &&
           s->n_swa == n_swa && s->n_indexer_head == n_indexer_head &&
           s->n_indexer_head_dim == n_indexer_head_dim &&
           s->n_indexer_top_k == n_indexer_top_k && s->n_hc == n_hc &&
           s->n_hc_sinkhorn_iter == n_hc_sinkhorn_iter;
}

static const char *select_name(const ds4_shape *probe)
{
    if (shape_matches(probe,
                      probe->n_layer, probe->n_embd, probe->n_vocab, probe->n_head,
                      probe->n_head_kv, probe->n_head_dim, probe->n_value_dim,
                      probe->n_rot, probe->n_lora_q, probe->n_lora_o,
                      probe->n_out_group, probe->n_expert, probe->n_expert_used,
                      probe->n_ff_exp, probe->n_expert_shared, probe->n_hash_layer,
                      probe->n_swa, probe->n_indexer_head, probe->n_indexer_head_dim,
                      probe->n_indexer_top_k, probe->n_hc, probe->n_hc_sinkhorn_iter)) {
        /* always true for probe vs itself — use catalog compare below */
    }
    if (shape_matches(&DS4_SHAPE_FLASH,
                      probe->n_layer, probe->n_embd, probe->n_vocab, probe->n_head,
                      probe->n_head_kv, probe->n_head_dim, probe->n_value_dim,
                      probe->n_rot, probe->n_lora_q, probe->n_lora_o,
                      probe->n_out_group, probe->n_expert, probe->n_expert_used,
                      probe->n_ff_exp, probe->n_expert_shared, probe->n_hash_layer,
                      probe->n_swa, probe->n_indexer_head, probe->n_indexer_head_dim,
                      probe->n_indexer_top_k, probe->n_hc, probe->n_hc_sinkhorn_iter)) {
        return DS4_SHAPE_FLASH.name;
    }
    if (shape_matches(&DS4_SHAPE_PRO,
                      probe->n_layer, probe->n_embd, probe->n_vocab, probe->n_head,
                      probe->n_head_kv, probe->n_head_dim, probe->n_value_dim,
                      probe->n_rot, probe->n_lora_q, probe->n_lora_o,
                      probe->n_out_group, probe->n_expert, probe->n_expert_used,
                      probe->n_ff_exp, probe->n_expert_shared, probe->n_hash_layer,
                      probe->n_swa, probe->n_indexer_head, probe->n_indexer_head_dim,
                      probe->n_indexer_top_k, probe->n_hc, probe->n_hc_sinkhorn_iter)) {
        return DS4_SHAPE_PRO.name;
    }
    return "unsupported";
}

static const char *arch_route(const char *arch)
{
    if (!arch || strcmp(arch, "deepseek4") == 0) return "deepseek-select";
    if (strcmp(arch, "exaone-moe") == 0) return DS4_SHAPE_KEXAONE_236B.name;
    if (strcmp(arch, "solar-open2") == 0) return DS4_SHAPE_SOLAR_OPEN2_250B.name;
    if (strcmp(arch, "motif3") == 0) return DS4_SHAPE_MOTIF3.name;
    if (strcmp(arch, "dots3-note") == 0) return DS4_SHAPE_DOTS3_NOTE_PREV.name;
    if (strcmp(arch, "qwen4exp") == 0) return DS4_SHAPE_QWEN38_FLASH_NEXT.name;
    if (strcmp(arch, "glm5-next") == 0) return DS4_SHAPE_GLM53_FLASH.name;
    if (strcmp(arch, "k2-horizon") == 0) return DS4_SHAPE_K2_HORIZON_375B.name;
    return "unsupported";
}

int main(void)
{
    ds4_shape miss;
    printf("FAMILY\t0\t1\t2\t3\t4\t5\t6\n");
    printf("VARIANT\t0\t1\t2\t3\t4\t5\t6\t7\t8\n");
    dump_shape("DEFAULT", &DS4_SHAPE_FLASH);
    dump_shape("FLASH", &DS4_SHAPE_FLASH);
    dump_shape("PRO", &DS4_SHAPE_PRO);
    dump_shape("MOTIF3", &DS4_SHAPE_MOTIF3);
    dump_shape("SOLAR", &DS4_SHAPE_SOLAR_OPEN2_250B);
    dump_shape("KEXAONE", &DS4_SHAPE_KEXAONE_236B);
    dump_shape("DOTS3", &DS4_SHAPE_DOTS3_NOTE_PREV);
    dump_shape("QWEN38", &DS4_SHAPE_QWEN38_FLASH_NEXT);
    dump_shape("GLM53", &DS4_SHAPE_GLM53_FLASH);
    dump_shape("K2HORIZON", &DS4_SHAPE_K2_HORIZON_375B);
    printf("SELECT\tflash\t%s\n", select_name(&DS4_SHAPE_FLASH));
    printf("SELECT\tpro\t%s\n", select_name(&DS4_SHAPE_PRO));
    miss = DS4_SHAPE_FLASH;
    miss.n_layer = 1;
    printf("SELECT\tmiss\t%s\n", select_name(&miss));
    printf("ARCH\tmissing\t%s\n", arch_route(NULL));
    printf("ARCH\tdeepseek4\t%s\n", arch_route("deepseek4"));
    printf("ARCH\texaone-moe\t%s\n", arch_route("exaone-moe"));
    printf("ARCH\tsolar-open2\t%s\n", arch_route("solar-open2"));
    printf("ARCH\tmotif3\t%s\n", arch_route("motif3"));
    printf("ARCH\tdots3-note\t%s\n", arch_route("dots3-note"));
    printf("ARCH\tqwen4exp\t%s\n", arch_route("qwen4exp"));
    printf("ARCH\tglm5-next\t%s\n", arch_route("glm5-next"));
    printf("ARCH\tk2-horizon\t%s\n", arch_route("k2-horizon"));
    printf("ARCH\tglm-dsa\t%s\n", arch_route("glm-dsa"));
    return 0;
}
