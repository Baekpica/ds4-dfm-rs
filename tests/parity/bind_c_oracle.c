/* weights_bind name catalog + bind-plan check/match oracle from ds4.c.
 * Standalone: do not include ds4.c. */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    DS4_MODEL_FAMILY_DEEPSEEK4 = 0,
    DS4_MODEL_FAMILY_SOLAR_OPEN2 = 1,
    DS4_MODEL_FAMILY_MOTIF3 = 2,
    DS4_MODEL_FAMILY_EXAONE_MOE = 3,
    DS4_MODEL_FAMILY_DOTS3_NOTE = 4,
    DS4_MODEL_FAMILY_QWEN4EXP = 5,
    DS4_MODEL_FAMILY_GLM53 = 6
} ds4_model_family;

typedef enum {
    DS4_VARIANT_FLASH = 0,
    DS4_VARIANT_PRO = 1,
    DS4_VARIANT_SOLAR_OPEN2_250B = 2,
    DS4_VARIANT_MOTIF3 = 3,
    DS4_VARIANT_KEXAONE_236B = 4,
    DS4_VARIANT_DOTS3_NOTE_PREV = 5,
    DS4_VARIANT_QWEN38_FLASH_NEXT = 6,
    DS4_VARIANT_GLM53_FLASH = 7
} ds4_variant;

typedef struct {
    const char *name;
    ds4_model_family family;
    ds4_variant variant;
    uint32_t n_layer;
    uint32_t n_hash_layer;
    uint32_t n_swa_period;
    uint32_t n_nextn_predict;
    uint32_t n_leading_dense;
    bool use_qk_norm;
} bind_shape;

static const bind_shape SHAPE_FLASH = {
    "DeepSeek V4 Flash", DS4_MODEL_FAMILY_DEEPSEEK4, DS4_VARIANT_FLASH,
    43, 3, 0, 0, 0, false
};
static const bind_shape SHAPE_PRO = {
    "DeepSeek V4 Pro", DS4_MODEL_FAMILY_DEEPSEEK4, DS4_VARIANT_PRO,
    61, 3, 0, 0, 0, false
};
static const bind_shape SHAPE_SOLAR = {
    "Solar Open2 250B", DS4_MODEL_FAMILY_SOLAR_OPEN2, DS4_VARIANT_SOLAR_OPEN2_250B,
    48, 0, 0, 0, 0, false
};
static const bind_shape SHAPE_MOTIF3 = {
    "Motif-3", DS4_MODEL_FAMILY_MOTIF3, DS4_VARIANT_MOTIF3,
    53, 0, 4, 1, 2, false
};
static const bind_shape SHAPE_KEXAONE = {
    "K-EXAONE 236B A23B", DS4_MODEL_FAMILY_EXAONE_MOE, DS4_VARIANT_KEXAONE_236B,
    49, 0, 4, 1, 1, true
};
static const bind_shape SHAPE_DOTS3 = {
    "dots3-note-prev", DS4_MODEL_FAMILY_DOTS3_NOTE, DS4_VARIANT_DOTS3_NOTE_PREV,
    47, 0, 4, 1, 1, false
};
static const bind_shape SHAPE_QWEN = {
    "Qwen3.8-Flash-Next", DS4_MODEL_FAMILY_QWEN4EXP, DS4_VARIANT_QWEN38_FLASH_NEXT,
    48, 0, 4, 1, 0, true
};
static const bind_shape SHAPE_GLM53 = {
    "GLM 5.3 Flash", DS4_MODEL_FAMILY_GLM53, DS4_VARIANT_GLM53_FLASH,
    46, 0, 0, 1, 3, false
};

static uint32_t g_n, g_req, g_opt;

static uint32_t expected_compress_ratio(const bind_shape *s, uint32_t il) {
    if (il >= s->n_layer) return 0;
    if (s->variant == DS4_VARIANT_FLASH) {
        if (il < 2) return 0;
        return (il & 1u) == 0 ? 4u : 128u;
    }
    if (s->variant == DS4_VARIANT_PRO) {
        if (il < 2) return 128u;
        return (il & 1u) == 0 ? 4u : 128u;
    }
    return 0;
}

static bool solar_layer_is_gqa(const bind_shape *s, uint32_t il) {
    return s->family == DS4_MODEL_FAMILY_SOLAR_OPEN2 && il < s->n_layer && (il % 4u) == 0u;
}

static bool dots3_layer_is_full_attention(const bind_shape *s, uint32_t il) {
    if (s->family != DS4_MODEL_FAMILY_DOTS3_NOTE || il >= s->n_layer) return false;
    if (s->n_nextn_predict != 0 && il + s->n_nextn_predict >= s->n_layer) return false;
    return il == 0u || (s->n_swa_period != 0 && (il % s->n_swa_period) == 1u);
}

static bool qwen4exp_layer_is_full_attention(const bind_shape *s, uint32_t il) {
    return s->family == DS4_MODEL_FAMILY_QWEN4EXP && il < s->n_layer &&
           s->n_swa_period != 0 && (il % s->n_swa_period) == 3u;
}

static void emit(const char *name, int required);
static void emitf(const char *fmt, uint32_t il, int required);

static bool is_nextn(const bind_shape *s, uint32_t il) {
    return s->n_nextn_predict != 0 && il + s->n_nextn_predict >= s->n_layer;
}

static bool glm53_layer_is_kda(const bind_shape *s, uint32_t il) {
    return s->family == DS4_MODEL_FAMILY_GLM53 && il < s->n_layer &&
           !is_nextn(s, il) && (il % 4u) != 3u;
}

static void bind_glm53_layer(const bind_shape *s, uint32_t il) {
    static const char *const hc[] = {
        "hc_attn_fn.weight", "hc_attn_scale.weight", "hc_attn_base.weight",
        "hc_ffn_fn.weight", "hc_ffn_scale.weight", "hc_ffn_base.weight"
    };
    static const char *const kda[] = {
        "kda_q.weight", "kda_k.weight", "kda_v.weight",
        "kda_q_conv.weight", "kda_k_conv.weight", "kda_v_conv.weight",
        "kda_f_a.weight", "kda_f_b.weight", "kda_dt_bias.weight",
        "kda_a_log.weight", "kda_beta.weight", "kda_g_a.weight",
        "kda_g_b.weight", "kda_o_norm.weight", "kda_output.weight"
    };
    static const char *const dsa[] = {
        "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
        "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_k_b.weight",
        "attn_v_b.weight", "attn_output.weight", "indexer.attn_q_b.weight",
        "indexer.attn_k.weight", "indexer.k_norm.weight", "indexer.k_norm.bias",
        "indexer.proj.weight", "indexer.pool_ape.weight", "indexer.pool_gate.weight"
    };
    static const char *const dense[] = {
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"
    };
    static const char *const moe[] = {
        "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_exps.weight",
        "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight", "ffn_down_shexp.weight"
    };
    static const char *const nextn[] = {
        "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
        "nextn.shared_head_norm.weight"
    };
    char name[128];
    size_t i;

    emitf("blk.%u.attn_norm.weight", il, 1);
    emitf("blk.%u.ffn_norm.weight", il, 1);
    if (!is_nextn(s, il)) {
        for (i = 0; i < sizeof(hc) / sizeof(hc[0]); i++) {
            snprintf(name, sizeof(name), "blk.%u.%s", il, hc[i]);
            emit(name, 1);
        }
    }
    const char *const *attn = glm53_layer_is_kda(s, il) ? kda : dsa;
    for (i = 0; i < sizeof(kda) / sizeof(kda[0]); i++) {
        snprintf(name, sizeof(name), "blk.%u.%s", il, attn[i]);
        emit(name, 1);
    }
    const char *const *ffn = il < s->n_leading_dense ? dense : moe;
    const size_t ffn_count = il < s->n_leading_dense
        ? sizeof(dense) / sizeof(dense[0])
        : sizeof(moe) / sizeof(moe[0]);
    for (i = 0; i < ffn_count; i++) {
        snprintf(name, sizeof(name), "blk.%u.%s", il, ffn[i]);
        emit(name, 1);
    }
    if (is_nextn(s, il)) {
        for (i = 0; i < sizeof(nextn) / sizeof(nextn[0]); i++) {
            snprintf(name, sizeof(name), "blk.%u.%s", il, nextn[i]);
            emit(name, 1);
        }
    }
}

static void emit(const char *name, int required) {
    printf("NAME %s %s\n", name, required ? "REQ" : "OPT");
    g_n++;
    if (required) g_req++; else g_opt++;
}

static void emitf(const char *fmt, uint32_t il, int required) {
    char name[128];
    snprintf(name, sizeof(name), fmt, il);
    emit(name, required);
}

static void bind_motif3_layer(const bind_shape *s, uint32_t il) {
    emitf("blk.%u.attn_norm.weight", il, 1);
    emitf("blk.%u.mhc_attn.rms_norm.weight", il, 1);
    emitf("blk.%u.mhc_attn.proj_pre.weight", il, 1);
    emitf("blk.%u.mhc_attn.proj_post.weight", il, 1);
    emitf("blk.%u.mhc_attn.proj_res.weight", il, 1);
    emitf("blk.%u.mhc_attn.alpha_pre", il, 1);
    emitf("blk.%u.mhc_attn.alpha_post", il, 1);
    emitf("blk.%u.mhc_attn.alpha_res", il, 1);
    emitf("blk.%u.mhc_attn.bias_pre", il, 1);
    emitf("blk.%u.mhc_attn.bias_post", il, 1);
    emitf("blk.%u.mhc_attn.bias_res", il, 1);
    emitf("blk.%u.attn_q_a.weight", il, 1);
    emitf("blk.%u.attn_q_a_norm.weight", il, 1);
    emitf("blk.%u.attn_q_b.weight", il, 1);
    emitf("blk.%u.attn_q_gate.weight", il, 1);
    emitf("blk.%u.attn_kv_a.weight", il, 1);
    emitf("blk.%u.attn_kv_a_norm.weight", il, 1);
    emitf("blk.%u.attn_kv_b.weight", il, 1);
    emitf("blk.%u.attn_lambda.weight", il, 1);
    emitf("blk.%u.attn_output.weight", il, 1);
    emitf("blk.%u.mhc_ffn.rms_norm.weight", il, 1);
    emitf("blk.%u.mhc_ffn.proj_pre.weight", il, 1);
    emitf("blk.%u.mhc_ffn.proj_post.weight", il, 1);
    emitf("blk.%u.mhc_ffn.proj_res.weight", il, 1);
    emitf("blk.%u.mhc_ffn.alpha_pre", il, 1);
    emitf("blk.%u.mhc_ffn.alpha_post", il, 1);
    emitf("blk.%u.mhc_ffn.alpha_res", il, 1);
    emitf("blk.%u.mhc_ffn.bias_pre", il, 1);
    emitf("blk.%u.mhc_ffn.bias_post", il, 1);
    emitf("blk.%u.mhc_ffn.bias_res", il, 1);
    emitf("blk.%u.ffn_norm.weight", il, 1);
    if (il < s->n_leading_dense) {
        emitf("blk.%u.ffn_gate.weight", il, 1);
        emitf("blk.%u.ffn_up.weight", il, 1);
        emitf("blk.%u.ffn_down.weight", il, 1);
        emitf("blk.%u.ffn_polynorm.weight", il, 1);
        emitf("blk.%u.ffn_polynorm.bias", il, 1);
        return;
    }
    emitf("blk.%u.ffn_gate_inp.weight", il, 1);
    emitf("blk.%u.exp_probs_b.bias", il, 1);
    emitf("blk.%u.ffn_gate_exps.weight", il, 1);
    emitf("blk.%u.ffn_up_exps.weight", il, 1);
    emitf("blk.%u.ffn_down_exps.weight", il, 1);
    emitf("blk.%u.ffn_polynorm_exps.weight", il, 1);
    emitf("blk.%u.ffn_polynorm_exps.bias", il, 1);
    emitf("blk.%u.ffn_gate_shexp.weight", il, 1);
    emitf("blk.%u.ffn_up_shexp.weight", il, 1);
    emitf("blk.%u.ffn_down_shexp.weight", il, 1);
    emitf("blk.%u.ffn_polynorm_shexp.weight", il, 1);
    emitf("blk.%u.ffn_polynorm_shexp.bias", il, 1);
}

static void bind_motif3_mtp(void) {
    emit("mtp.0.embed_norm.weight", 1);
    emit("mtp.0.input_layernorm.weight", 1);
    emit("mtp.0.input_proj.weight", 1);
    emit("mtp.0.final_layernorm.weight", 1);
    emit("mtp.0.attn_q_a.weight", 1);
    emit("mtp.0.attn_q_a_norm.weight", 1);
    emit("mtp.0.attn_q_b.weight", 1);
    emit("mtp.0.attn_q_gate.weight", 1);
    emit("mtp.0.attn_kv_a.weight", 1);
    emit("mtp.0.attn_kv_a_norm.weight", 1);
    emit("mtp.0.attn_kv_b.weight", 1);
    emit("mtp.0.attn_lambda.weight", 1);
    emit("mtp.0.attn_output.weight", 1);
    emit("mtp.0.post_attention_layernorm.weight", 1);
    emit("mtp.0.ffn_gate.weight", 1);
    emit("mtp.0.ffn_up.weight", 1);
    emit("mtp.0.ffn_down.weight", 1);
    emit("mtp.0.ffn_polynorm.weight", 1);
    emit("mtp.0.ffn_polynorm.bias", 1);
}

static void bind_exaone_moe_layer(const bind_shape *s, uint32_t il) {
    emitf("blk.%u.attn_norm.weight", il, 1);
    emitf("blk.%u.attn_q.weight", il, 1);
    emitf("blk.%u.attn_k.weight", il, 1);
    emitf("blk.%u.attn_v.weight", il, 1);
    emitf("blk.%u.attn_output.weight", il, 1);
    if (s->use_qk_norm) {
        emitf("blk.%u.attn_q_norm.weight", il, 1);
        emitf("blk.%u.attn_k_norm.weight", il, 1);
    }
    emitf("blk.%u.ffn_norm.weight", il, 1);
    if (il < s->n_leading_dense || is_nextn(s, il)) {
        emitf("blk.%u.ffn_gate.weight", il, 1);
        emitf("blk.%u.ffn_up.weight", il, 1);
        emitf("blk.%u.ffn_down.weight", il, 1);
    } else {
        emitf("blk.%u.ffn_gate_inp.weight", il, 1);
        emitf("blk.%u.exp_probs_b.bias", il, 1);
        emitf("blk.%u.ffn_gate_exps.weight", il, 1);
        emitf("blk.%u.ffn_up_exps.weight", il, 1);
        emitf("blk.%u.ffn_down_exps.weight", il, 1);
        emitf("blk.%u.ffn_gate_shexp.weight", il, 1);
        emitf("blk.%u.ffn_up_shexp.weight", il, 1);
        emitf("blk.%u.ffn_down_shexp.weight", il, 1);
    }
    if (is_nextn(s, il)) {
        emitf("blk.%u.nextn.eh_proj.weight", il, 1);
        emitf("blk.%u.nextn.enorm.weight", il, 1);
        emitf("blk.%u.nextn.hnorm.weight", il, 1);
        emitf("blk.%u.nextn.shared_head_norm.weight", il, 1);
    }
}

static void bind_dots3_note_layer(const bind_shape *s, uint32_t il) {
    emitf("blk.%u.attn_norm.weight", il, 1);
    emitf("blk.%u.attn_q_a.weight", il, 1);
    emitf("blk.%u.attn_q_a_norm.weight", il, 1);
    emitf("blk.%u.attn_q_b.weight", il, 1);
    emitf("blk.%u.attn_kv_a_mqa.weight", il, 1);
    emitf("blk.%u.attn_kv_a_norm.weight", il, 1);
    emitf("blk.%u.attn_kv_b.weight", il, 1);
    emitf("blk.%u.attn_k_rope_norm.weight", il, 1);
    emitf("blk.%u.attn_gate.weight", il, 1);
    emitf("blk.%u.attn_output.weight", il, 1);
    if (dots3_layer_is_full_attention(s, il)) {
        emitf("blk.%u.attn_idx_q_b.weight", il, 1);
        emitf("blk.%u.attn_idx_k.weight", il, 1);
        emitf("blk.%u.attn_idx_w.weight", il, 1);
        emitf("blk.%u.attn_idx_k_norm.weight", il, 1);
        emitf("blk.%u.attn_idx_k_norm.bias", il, 1);
    }
    emitf("blk.%u.ffn_norm.weight", il, 1);
    if (il < s->n_leading_dense || is_nextn(s, il)) {
        emitf("blk.%u.ffn_gate.weight", il, 1);
        emitf("blk.%u.ffn_up.weight", il, 1);
        emitf("blk.%u.ffn_down.weight", il, 1);
    } else {
        emitf("blk.%u.ffn_gate_inp.weight", il, 1);
        emitf("blk.%u.exp_probs_b.bias", il, 1);
        emitf("blk.%u.ffn_gate_exps.weight", il, 1);
        emitf("blk.%u.ffn_up_exps.weight", il, 1);
        emitf("blk.%u.ffn_down_exps.weight", il, 1);
        emitf("blk.%u.ffn_gate_shexp.weight", il, 1);
        emitf("blk.%u.ffn_up_shexp.weight", il, 1);
        emitf("blk.%u.ffn_down_shexp.weight", il, 1);
    }
    if (is_nextn(s, il)) {
        emitf("blk.%u.eh_proj.weight", il, 1);
        emitf("blk.%u.enorm.weight", il, 1);
        emitf("blk.%u.hnorm.weight", il, 1);
        emitf("blk.%u.shared_head_norm.weight", il, 1);
    }
}

static void bind_qwen4exp_layer(const bind_shape *s, uint32_t il) {
    static const char *const hc_attn[] = {
        "blk.%u.hc_attn.norm.weight", "blk.%u.hc_attn.mix_down.weight",
        "blk.%u.hc_attn.mix_up.weight", "blk.%u.hc_attn.inject.weight"
    };
    static const char *const qsa[] = {
        "blk.%u.attn_index_qk.weight", "blk.%u.attn_index_q_norm.weight",
        "blk.%u.attn_index_k_norm.weight", "blk.%u.attn_q.weight",
        "blk.%u.attn_q_norm.weight", "blk.%u.attn_k.weight",
        "blk.%u.attn_k_norm.weight", "blk.%u.attn_v.weight",
        "blk.%u.attn_output.weight"
    };
    static const char *const gdn[] = {
        "blk.%u.linear_attn.a_log", "blk.%u.linear_attn.conv.weight",
        "blk.%u.linear_attn.dt_bias", "blk.%u.linear_attn.in_a.weight",
        "blk.%u.linear_attn.in_b.weight", "blk.%u.linear_attn.qkv.weight",
        "blk.%u.linear_attn.z.weight", "blk.%u.linear_attn.norm.weight",
        "blk.%u.linear_attn.out.weight"
    };
    static const char *const tail[] = {
        "blk.%u.ffn_gate_inp.weight", "blk.%u.ffn_gate_exps.weight",
        "blk.%u.ffn_up_exps.weight", "blk.%u.ffn_down_exps.main.weight",
        "blk.%u.ffn_down_exps.tail.weight", "blk.%u.ffn_gate_shexp.weight",
        "blk.%u.ffn_up_shexp.weight", "blk.%u.ffn_down_shexp.weight",
        "blk.%u.ffn_shexp_gate_inp.weight", "blk.%u.hc_ffn.norm.weight",
        "blk.%u.hc_ffn.mix_down.weight", "blk.%u.hc_ffn.mix_up.weight",
        "blk.%u.hc_ffn.inject.weight"
    };
    static const char *const ple[] = {
        "blk.1.ple.conv.weight", "blk.1.ple.key.weight",
        "blk.1.ple.value.weight", "blk.1.ple.conv_norm.weight",
        "blk.1.ple.key_norm.weight", "blk.1.ple.query_norm.weight",
        "blk.1.ple.layer_multipliers", "blk.1.ple.head_offsets",
        "blk.1.ple.head_vocab_sizes"
    };
    size_t i;
    for (i = 0; i < sizeof(hc_attn) / sizeof(hc_attn[0]); i++) emitf(hc_attn[i], il, 1);
    if (qwen4exp_layer_is_full_attention(s, il)) {
        for (i = 0; i < sizeof(qsa) / sizeof(qsa[0]); i++) emitf(qsa[i], il, 1);
    } else {
        for (i = 0; i < sizeof(gdn) / sizeof(gdn[0]); i++) emitf(gdn[i], il, 1);
    }
    for (i = 0; i < sizeof(tail) / sizeof(tail[0]); i++) emitf(tail[i], il, 1);
    if (il == 1u)
        for (i = 0; i < sizeof(ple) / sizeof(ple[0]); i++) emit(ple[i], 1);
}

static void bind_qwen4exp_mtp(void) {
    static const char *const names[] = {
        "mtp.fc_embedding.weight", "mtp.fc_hidden.weight",
        "mtp.fc_embedding_norm.weight", "mtp.fc_hidden_norm.weight",
        "mtp.hc_input.norm.weight", "mtp.hc_input.mix_down.weight",
        "mtp.hc_input.mix_up.weight", "mtp.blk.0.hc_attn.norm.weight",
        "mtp.blk.0.hc_attn.mix_down.weight", "mtp.blk.0.hc_attn.mix_up.weight",
        "mtp.blk.0.hc_attn.inject.weight", "mtp.blk.0.attn_index_qk.weight",
        "mtp.blk.0.attn_index_q_norm.weight", "mtp.blk.0.attn_index_k_norm.weight",
        "mtp.blk.0.attn_q.weight", "mtp.blk.0.attn_q_norm.weight",
        "mtp.blk.0.attn_k.weight", "mtp.blk.0.attn_k_norm.weight",
        "mtp.blk.0.attn_v.weight", "mtp.blk.0.attn_output.weight",
        "mtp.blk.0.ffn_gate_inp.weight", "mtp.blk.0.ffn_gate_exps.weight",
        "mtp.blk.0.ffn_up_exps.weight", "mtp.blk.0.ffn_down_exps.main.weight",
        "mtp.blk.0.ffn_down_exps.tail.weight", "mtp.blk.0.ffn_gate_shexp.weight",
        "mtp.blk.0.ffn_up_shexp.weight", "mtp.blk.0.ffn_down_shexp.weight",
        "mtp.blk.0.ffn_shexp_gate_inp.weight", "mtp.blk.0.hc_ffn.norm.weight",
        "mtp.blk.0.hc_ffn.mix_down.weight", "mtp.blk.0.hc_ffn.mix_up.weight",
        "mtp.blk.0.hc_ffn.inject.weight"
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) emit(names[i], 1);
}

static void bind_qwen4exp_vision(void) {
    static const char *const layer[] = {
        "vblk.%u.norm1.weight", "vblk.%u.norm1.bias",
        "vblk.%u.attn_qkv.weight", "vblk.%u.attn_qkv.bias",
        "vblk.%u.attn_output.weight", "vblk.%u.attn_output.bias",
        "vblk.%u.norm2.weight", "vblk.%u.norm2.bias",
        "vblk.%u.ffn_up.weight", "vblk.%u.ffn_up.bias",
        "vblk.%u.ffn_down.weight", "vblk.%u.ffn_down.bias"
    };
    static const char *const merger[] = {
        "vision.merger.norm.weight", "vision.merger.norm.bias",
        "vision.merger.ffn_up.weight", "vision.merger.ffn_up.bias",
        "vision.merger.ffn_down.weight", "vision.merger.ffn_down.bias"
    };
    emit("vision.patch_embed.weight", 1);
    emit("vision.patch_embed.bias", 1);
    emit("vision.position_embd.weight", 1);
    for (uint32_t il = 0; il < 27u; il++)
        for (size_t i = 0; i < sizeof(layer) / sizeof(layer[0]); i++)
            emitf(layer[i], il, 1);
    for (size_t i = 0; i < sizeof(merger) / sizeof(merger[0]); i++) emit(merger[i], 1);
}

static void bind_solar_open2_layer(const bind_shape *s, uint32_t il) {
    emitf("blk.%u.attn_norm.weight", il, 1);
    emitf("blk.%u.attn_q.weight", il, 1);
    emitf("blk.%u.attn_k.weight", il, 1);
    emitf("blk.%u.attn_v.weight", il, 1);
    emitf("blk.%u.attn_output.weight", il, 1);
    if (solar_layer_is_gqa(s, il)) {
        emitf("blk.%u.attn_gate.weight", il, 1);
    } else {
        emitf("blk.%u.ssm_conv1d_q.weight", il, 1);
        emitf("blk.%u.ssm_conv1d_k.weight", il, 1);
        emitf("blk.%u.ssm_conv1d_v.weight", il, 1);
        emitf("blk.%u.ssm_f_a.weight", il, 1);
        emitf("blk.%u.ssm_f_b.weight", il, 1);
        emitf("blk.%u.ssm_beta.weight", il, 1);
        emitf("blk.%u.ssm_a", il, 1);
        emitf("blk.%u.ssm_dt.bias", il, 1);
        emitf("blk.%u.ssm_g_a.weight", il, 1);
        emitf("blk.%u.ssm_g_b.weight", il, 1);
        emitf("blk.%u.ssm_norm.weight", il, 1);
    }
    emitf("blk.%u.ffn_norm.weight", il, 1);
    emitf("blk.%u.ffn_gate_inp.weight", il, 1);
    emitf("blk.%u.exp_probs_b.bias", il, 1);
    emitf("blk.%u.ffn_gate_exps.weight", il, 1);
    emitf("blk.%u.ffn_up_exps.weight", il, 1);
    emitf("blk.%u.ffn_down_exps.weight", il, 1);
    emitf("blk.%u.ffn_gate_shexp.weight", il, 1);
    emitf("blk.%u.ffn_up_shexp.weight", il, 1);
    emitf("blk.%u.ffn_down_shexp.weight", il, 1);
}

static void bind_deepseek_layer(const bind_shape *s, uint32_t il) {
    uint32_t compress_ratio = expected_compress_ratio(s, il);
    emitf("blk.%u.hc_attn_fn.weight", il, 1);
    emitf("blk.%u.hc_attn_scale.weight", il, 1);
    emitf("blk.%u.hc_attn_base.weight", il, 1);
    emitf("blk.%u.attn_norm.weight", il, 1);
    emitf("blk.%u.attn_q_a.weight", il, 1);
    emitf("blk.%u.attn_q_a_norm.weight", il, 1);
    emitf("blk.%u.attn_q_b.weight", il, 1);
    emitf("blk.%u.attn_kv.weight", il, 1);
    emitf("blk.%u.attn_kv_a_norm.weight", il, 1);
    emitf("blk.%u.attn_sinks.weight", il, 1);
    emitf("blk.%u.attn_output_a.weight", il, 1);
    emitf("blk.%u.attn_output_b.weight", il, 1);
    if (compress_ratio != 0) {
        emitf("blk.%u.attn_compressor_ape.weight", il, 1);
        emitf("blk.%u.attn_compressor_kv.weight", il, 1);
        emitf("blk.%u.attn_compressor_gate.weight", il, 1);
        emitf("blk.%u.attn_compressor_norm.weight", il, 1);
    }
    if (compress_ratio == 4) {
        emitf("blk.%u.indexer.attn_q_b.weight", il, 1);
        emitf("blk.%u.indexer.proj.weight", il, 1);
        emitf("blk.%u.indexer_compressor_ape.weight", il, 1);
        emitf("blk.%u.indexer_compressor_kv.weight", il, 1);
        emitf("blk.%u.indexer_compressor_gate.weight", il, 1);
        emitf("blk.%u.indexer_compressor_norm.weight", il, 1);
    }
    emitf("blk.%u.hc_ffn_fn.weight", il, 1);
    emitf("blk.%u.hc_ffn_scale.weight", il, 1);
    emitf("blk.%u.hc_ffn_base.weight", il, 1);
    emitf("blk.%u.ffn_norm.weight", il, 1);
    emitf("blk.%u.ffn_gate_inp.weight", il, 1);
    emitf("blk.%u.exp_probs_b.bias", il, 0);
    emitf("blk.%u.ffn_gate_exps.weight", il, 1);
    emitf("blk.%u.ffn_up_exps.weight", il, 1);
    emitf("blk.%u.ffn_down_exps.weight", il, 1);
    emitf("blk.%u.ffn_gate_shexp.weight", il, 1);
    emitf("blk.%u.ffn_up_shexp.weight", il, 1);
    emitf("blk.%u.ffn_down_shexp.weight", il, 1);
    if (il < s->n_hash_layer) emitf("blk.%u.ffn_gate_tid2eid.weight", il, 1);
}

static void dump_shape(const bind_shape *s) {
    uint32_t il;
    g_n = g_req = g_opt = 0;
    printf("BIND name=%s family=%u variant=%u n_layer=%u\n",
           s->name, (unsigned)s->family, (unsigned)s->variant, s->n_layer);
    if (s->family == DS4_MODEL_FAMILY_GLM53) {
        emit("token_embd.weight", 1);
        emit("output_norm.weight", 1);
        emit("output.weight", 1);
        for (il = 0; il < s->n_layer; il++) bind_glm53_layer(s, il);
    } else if (s->family == DS4_MODEL_FAMILY_QWEN4EXP) {
        emit("token_embd.weight", 1);
        emit("output.weight", 1);
        emit("hc_input.norm.weight", 1);
        emit("hc_input.mix_down.weight", 1);
        emit("hc_input.mix_up.weight", 1);
        for (il = 0; il < s->n_layer; il++) bind_qwen4exp_layer(s, il);
        bind_qwen4exp_mtp();
        bind_qwen4exp_vision();
    } else if (s->family == DS4_MODEL_FAMILY_MOTIF3) {
        emit("token_embd.weight", 1);
        emit("output_norm.weight", 1);
        emit("output.weight", 1);
        for (il = 0; il < s->n_layer; il++) bind_motif3_layer(s, il);
        bind_motif3_mtp();
    } else if (s->family == DS4_MODEL_FAMILY_DOTS3_NOTE) {
        emit("token_embd.weight", 1);
        emit("output_norm.weight", 1);
        emit("output.weight", 1);
        for (il = 0; il < s->n_layer; il++) bind_dots3_note_layer(s, il);
        emit("token_embd_mtp.weight", 1);
    } else if (s->family == DS4_MODEL_FAMILY_SOLAR_OPEN2) {
        emit("token_embd.weight", 1);
        emit("output_norm.weight", 1);
        emit("output.weight", 1);
        for (il = 0; il < s->n_layer; il++) bind_solar_open2_layer(s, il);
    } else if (s->family == DS4_MODEL_FAMILY_EXAONE_MOE) {
        emit("token_embd.weight", 1);
        emit("output_norm.weight", 1);
        emit("output.weight", 1);
        for (il = 0; il < s->n_layer; il++) bind_exaone_moe_layer(s, il);
    } else {
        emit("token_embd.weight", 1);
        emit("output_hc_base.weight", 1);
        emit("output_hc_fn.weight", 1);
        emit("output_hc_scale.weight", 1);
        emit("output_norm.weight", 1);
        emit("output.weight", 1);
        for (il = 0; il < s->n_layer; il++) bind_deepseek_layer(s, il);
    }
    printf("COUNT n=%u req=%u opt=%u\n", g_n, g_req, g_opt);
}

/* Bind-plan check/match tokens match native/bridge/ds4_bridge.c. */
#define DS4_MAX_DIMS 8

typedef struct {
    const char *name;
    uint32_t required;
    uint32_t ndim;
    uint64_t dim[8];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t bytes;
    uint32_t shard;
    uint32_t found;
} ds4_bridge_bind_slot;

typedef struct {
    const char *path;
    uint64_t size;
    uint64_t base;
} ds4_bridge_shard;

typedef struct {
    uint32_t n_slots;
    const ds4_bridge_bind_slot *slots;
    uint32_t n_shards;
    const ds4_bridge_shard *shards;
    uint64_t data_pos;
    uint64_t alignment;
    uint64_t page;
} ds4_bridge_bind_plan;

static void set_err(char *err, size_t errlen, const char *msg) {
    if (!err || errlen == 0) return;
    snprintf(err, errlen, "%s", msg ? msg : "unknown error");
}

static int bind_plan_check(const ds4_bridge_bind_plan *plan, char *err, size_t errlen) {
    uint32_t i;
    if (!plan) {
        set_err(err, errlen, "plan-null");
        return 1;
    }
    if (plan->n_slots > 0 && !plan->slots) {
        set_err(err, errlen, "slots-null");
        return 1;
    }
    if (plan->n_shards > 0 && !plan->shards) {
        set_err(err, errlen, "shards-null");
        return 1;
    }
    for (i = 0; i < plan->n_slots; i++) {
        const ds4_bridge_bind_slot *s = &plan->slots[i];
        if (!s->name || !s->name[0]) {
            set_err(err, errlen, "name-empty");
            return 1;
        }
        if (s->required && !s->found) {
            snprintf(err, errlen, "missing %s", s->name);
            return 1;
        }
        if (s->found && (s->ndim == 0 || s->ndim > DS4_MAX_DIMS)) {
            set_err(err, errlen, "bad-ndim");
            return 1;
        }
    }
    return 0;
}

static int bind_plan_match(const ds4_bridge_bind_plan *host,
                           const ds4_bridge_bind_plan *native,
                           char *err, size_t errlen) {
    uint32_t i, d;
    if (!host || !native) {
        set_err(err, errlen, "plan-null");
        return 1;
    }
    if (host->n_slots != native->n_slots) {
        set_err(err, errlen, "count-mismatch");
        return 1;
    }
    for (i = 0; i < host->n_slots; i++) {
        const ds4_bridge_bind_slot *h = &host->slots[i];
        const ds4_bridge_bind_slot *n = &native->slots[i];
        if (!h->name || !n->name || strcmp(h->name, n->name) != 0) {
            set_err(err, errlen, "name-mismatch");
            return 1;
        }
        if (h->required != n->required) {
            set_err(err, errlen, "need-mismatch");
            return 1;
        }
        if (h->found != n->found) {
            set_err(err, errlen, "found-mismatch");
            return 1;
        }
        if (!h->found) continue;
        if (h->type != n->type) {
            set_err(err, errlen, "type-mismatch");
            return 1;
        }
        if (h->ndim != n->ndim) {
            set_err(err, errlen, "dim-mismatch");
            return 1;
        }
        for (d = 0; d < h->ndim && d < DS4_MAX_DIMS; d++) {
            if (h->dim[d] != n->dim[d]) {
                set_err(err, errlen, "dim-mismatch");
                return 1;
            }
        }
        if (h->rel_offset != n->rel_offset || h->abs_offset != n->abs_offset) {
            set_err(err, errlen, "offset-mismatch");
            return 1;
        }
        if (h->bytes != n->bytes) {
            set_err(err, errlen, "bytes-mismatch");
            return 1;
        }
        if (h->shard != n->shard) {
            set_err(err, errlen, "shard-mismatch");
            return 1;
        }
    }
    if (host->n_shards != native->n_shards ||
        host->data_pos != native->data_pos ||
        host->alignment != native->alignment ||
        host->page != native->page) {
        set_err(err, errlen, "data-mismatch");
        return 1;
    }
    return 0;
}

static void print_check(int rc, const char *err) {
    if (rc == 0) printf("CHECK OK\n");
    else printf("CHECK %s\n", err);
}

static void print_match(int rc, const char *err) {
    if (rc == 0) printf("MATCH OK\n");
    else printf("MATCH %s\n", err);
}

static void run_check_cases(void) {
    ds4_bridge_bind_slot ok = {
        .name = "token_embd.weight",
        .required = 1,
        .ndim = 2,
        .dim = {4, 8},
        .type = 8,
        .rel_offset = 0,
        .abs_offset = 32,
        .bytes = 272,
        .shard = 0,
        .found = 1
    };
    ds4_bridge_bind_slot miss = ok;
    ds4_bridge_bind_slot empty = ok;
    ds4_bridge_bind_slot opt = ok;
    ds4_bridge_bind_slot badn = ok;
    ds4_bridge_shard shard = { .path = "/tmp/a.gguf", .size = 100, .base = 0 };
    ds4_bridge_bind_plan plan;
    char err[128];

    memset(&plan, 0, sizeof(plan));
    print_check(bind_plan_check(NULL, err, sizeof(err)), err);

    plan.n_slots = 1;
    plan.slots = NULL;
    print_check(bind_plan_check(&plan, err, sizeof(err)), err);

    plan.slots = &ok;
    plan.n_shards = 1;
    plan.shards = NULL;
    print_check(bind_plan_check(&plan, err, sizeof(err)), err);

    plan.shards = &shard;
    print_check(bind_plan_check(&plan, err, sizeof(err)), err);

    miss.found = 0;
    plan.slots = &miss;
    print_check(bind_plan_check(&plan, err, sizeof(err)), err);

    empty.name = "";
    plan.slots = &empty;
    print_check(bind_plan_check(&plan, err, sizeof(err)), err);

    opt.required = 0;
    opt.found = 0;
    opt.ndim = 0;
    plan.slots = &opt;
    print_check(bind_plan_check(&plan, err, sizeof(err)), err);

    badn.ndim = 0;
    plan.slots = &badn;
    print_check(bind_plan_check(&plan, err, sizeof(err)), err);
}

static void run_match_cases(void) {
    ds4_bridge_bind_slot a = {
        .name = "token_embd.weight",
        .required = 1,
        .ndim = 2,
        .dim = {4, 8},
        .type = 8,
        .rel_offset = 0,
        .abs_offset = 32,
        .bytes = 272,
        .shard = 0,
        .found = 1
    };
    ds4_bridge_bind_slot b = a;
    ds4_bridge_shard shard = { .path = "/tmp/a.gguf", .size = 100, .base = 0 };
    ds4_bridge_bind_plan host = {
        .n_slots = 1, .slots = &a, .n_shards = 1, .shards = &shard,
        .data_pos = 32, .alignment = 32, .page = 4096
    };
    ds4_bridge_bind_plan native = host;
    native.slots = &b;
    char err[128];

    print_match(bind_plan_match(&host, &native, err, sizeof(err)), err);

    native.n_slots = 2;
    print_match(bind_plan_match(&host, &native, err, sizeof(err)), err);
    native.n_slots = 1;

    b.name = "other.weight";
    print_match(bind_plan_match(&host, &native, err, sizeof(err)), err);
    b.name = a.name;

    b.type = 0;
    print_match(bind_plan_match(&host, &native, err, sizeof(err)), err);
    b.type = a.type;

    b.bytes = 1;
    print_match(bind_plan_match(&host, &native, err, sizeof(err)), err);
    b.bytes = a.bytes;

    native.data_pos = 99;
    print_match(bind_plan_match(&host, &native, err, sizeof(err)), err);
}

static void dump_names_header(const char *header)
{
    g_n = 0;
    g_req = 0;
    g_opt = 0;
    printf("%s", header);
}

static void dump_names_count(void)
{
    printf("COUNT n=%u req=%u opt=%u\n", g_n, g_req, g_opt);
}

static void dump_mtp(const bind_shape *s)
{
    char header[160];
    snprintf(header, sizeof(header), "BIND kind=mtp name=%s family=%u variant=%u\n",
             s->name, (unsigned)s->family, (unsigned)s->variant);
    dump_names_header(header);
    emit("mtp.0.hc_head_base.weight", 1);
    emit("mtp.0.hc_head_fn.weight", 1);
    emit("mtp.0.hc_head_scale.weight", 1);
    emit("mtp.0.e_proj.weight", 1);
    emit("mtp.0.h_proj.weight", 1);
    emit("mtp.0.enorm.weight", 1);
    emit("mtp.0.hnorm.weight", 1);
    emit("mtp.0.norm.weight", 1);
    emit("mtp.0.hc_attn_fn.weight", 1);
    emit("mtp.0.hc_attn_scale.weight", 1);
    emit("mtp.0.hc_attn_base.weight", 1);
    emit("mtp.0.attn_norm.weight", 1);
    emit("mtp.0.attn_q_a.weight", 1);
    emit("mtp.0.attn_q_a_norm.weight", 1);
    emit("mtp.0.attn_q_b.weight", 1);
    emit("mtp.0.attn_kv.weight", 1);
    emit("mtp.0.attn_kv_a_norm.weight", 1);
    emit("mtp.0.attn_sinks.weight", 1);
    emit("mtp.0.attn_output_a.weight", 1);
    emit("mtp.0.attn_output_b.weight", 1);
    emit("mtp.0.hc_ffn_fn.weight", 1);
    emit("mtp.0.hc_ffn_scale.weight", 1);
    emit("mtp.0.hc_ffn_base.weight", 1);
    emit("mtp.0.ffn_norm.weight", 1);
    emit("mtp.0.ffn_gate_inp.weight", 1);
    emit("mtp.0.exp_probs_b.bias", 1);
    emit("mtp.0.ffn_gate_exps.weight", 1);
    emit("mtp.0.ffn_up_exps.weight", 1);
    emit("mtp.0.ffn_down_exps.weight", 1);
    emit("mtp.0.ffn_gate_shexp.weight", 1);
    emit("mtp.0.ffn_up_shexp.weight", 1);
    emit("mtp.0.ffn_down_shexp.weight", 1);
    dump_names_count();
}

static void dump_dspark(const bind_shape *s)
{
    char header[160];
    uint32_t il;
    snprintf(header, sizeof(header),
             "BIND kind=dspark name=%s family=%u variant=%u n_layer=3\n",
             s->name, (unsigned)s->family, (unsigned)s->variant);
    dump_names_header(header);
    emit("dspark.main_proj.weight", 1);
    emit("dspark.main_norm.weight", 1);
    emit("dspark.markov_w1.weight", 1);
    emit("dspark.markov_w2.weight", 1);
    emit("dspark.conf_proj.weight", 1);
    emit("dspark.hc_head_fn.weight", 1);
    emit("dspark.hc_head_base.weight", 1);
    emit("dspark.hc_head_scale.weight", 1);
    emit("dspark.norm.weight", 1);
    for (il = 0; il < 3; il++) {
        emitf("dspark.%u.hc_attn_fn.weight", il, 1);
        emitf("dspark.%u.hc_attn_scale.weight", il, 1);
        emitf("dspark.%u.hc_attn_base.weight", il, 1);
        emitf("dspark.%u.attn_norm.weight", il, 1);
        emitf("dspark.%u.attn_q_a.weight", il, 1);
        emitf("dspark.%u.attn_q_a_norm.weight", il, 1);
        emitf("dspark.%u.attn_q_b.weight", il, 1);
        emitf("dspark.%u.attn_kv.weight", il, 1);
        emitf("dspark.%u.attn_kv_a_norm.weight", il, 1);
        emitf("dspark.%u.attn_sinks.weight", il, 1);
        emitf("dspark.%u.attn_output_a.weight", il, 1);
        emitf("dspark.%u.attn_output_b.weight", il, 1);
        emitf("dspark.%u.hc_ffn_fn.weight", il, 1);
        emitf("dspark.%u.hc_ffn_scale.weight", il, 1);
        emitf("dspark.%u.hc_ffn_base.weight", il, 1);
        emitf("dspark.%u.ffn_norm.weight", il, 1);
        emitf("dspark.%u.ffn_gate_inp.weight", il, 1);
        emitf("dspark.%u.exp_probs_b.bias", il, 1);
        emitf("dspark.%u.ffn_gate_exps.weight", il, 1);
        emitf("dspark.%u.ffn_up_exps.weight", il, 1);
        emitf("dspark.%u.ffn_down_exps.weight", il, 1);
        emitf("dspark.%u.ffn_gate_shexp.weight", il, 1);
        emitf("dspark.%u.ffn_up_shexp.weight", il, 1);
        emitf("dspark.%u.ffn_down_shexp.weight", il, 1);
    }
    dump_names_count();
}

int main(int argc, char **argv) {
    const char *cmd = argc > 1 ? argv[1] : "names";
    if (strcmp(cmd, "names") == 0) {
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
    if (strcmp(cmd, "support") == 0) {
        dump_mtp(&SHAPE_FLASH);
        dump_dspark(&SHAPE_FLASH);
        dump_mtp(&SHAPE_PRO);
        dump_dspark(&SHAPE_PRO);
        return 0;
    }
    if (strcmp(cmd, "check") == 0) {
        run_check_cases();
        return 0;
    }
    if (strcmp(cmd, "match") == 0) {
        run_match_cases();
        return 0;
    }
    fprintf(stderr, "usage: bind_c_oracle names|support|check|match\n");
    return 2;
}
