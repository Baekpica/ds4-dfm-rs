/* Qwen3.8-Flash-Next pinned metadata, hybrid schedule, and SSD-PLE contract
 * smoke.
 *
 *   ./tests/test_qwen4exp_loader <first BF16/Q8/Mixed GGUF shard>
 */
#include "../ds4.c"

static int check_ssd_precision_maps(void) {
    const ds4_str q6 = {
        "MQ-Q6-SSD-PLE-BF16", sizeof("MQ-Q6-SSD-PLE-BF16") - 1u
    };
    const ds4_str q5 = {
        "MQ-Q5-SSD-PLE-BF16", sizeof("MQ-Q5-SSD-PLE-BF16") - 1u
    };
    const ds4_str unknown = {
        "MQ-Q4-SSD-PLE-BF16", sizeof("MQ-Q4-SSD-PLE-BF16") - 1u
    };
    uint32_t edge = 0, interior = 0, down = 0, tail = 0;

    if (!qwen4exp_ssd_precision_types(q6, &edge, &interior, &down, &tail) ||
        edge != DS4_TENSOR_Q6_K || interior != DS4_TENSOR_Q5_K ||
        down != DS4_TENSOR_Q6_K || tail != DS4_TENSOR_Q5_0)
        return 1;
    if (!qwen4exp_ssd_precision_types(q5, &edge, &interior, &down, &tail) ||
        edge != DS4_TENSOR_Q5_K || interior != DS4_TENSOR_Q4_K ||
        down != DS4_TENSOR_Q5_K || tail != DS4_TENSOR_Q5_0)
        return 1;
    return qwen4exp_ssd_precision_types(
               unknown, &edge, &interior, &down, &tail) ? 1 : 0;
}

static int check_source_revisions(void) {
    const ds4_str upstream = {
        "f5d08274bafd880402bd16f5e3e6c514136ec06c",
        sizeof("f5d08274bafd880402bd16f5e3e6c514136ec06c") - 1u
    };
    const ds4_str uncensored = {
        "8336e613ea508b13c2159bd0f68965d97a606b95",
        sizeof("8336e613ea508b13c2159bd0f68965d97a606b95") - 1u
    };
    const ds4_str unknown = {"unknown", sizeof("unknown") - 1u};

    return !qwen4exp_source_revision_supported(upstream) ||
           !qwen4exp_source_revision_supported(uncensored) ||
           qwen4exp_source_revision_supported(unknown);
}

int main(int argc, char **argv) {
    if (check_ssd_precision_maps() != 0) {
        fprintf(stderr, "Qwen4Exp SSD-PLE precision-map recognition failed\n");
        return 1;
    }
    if (check_source_revisions() != 0) {
        fprintf(stderr, "Qwen4Exp source-revision allowlist failed\n");
        return 1;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: %s <qwen4exp.gguf>\n", argv[0]);
        return 2;
    }

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_QWEN4EXP ||
        DS4_MODEL_VARIANT != DS4_VARIANT_QWEN38_FLASH_NEXT ||
        DS4_N_LAYER != 48 || DS4_N_EMBD != 2560 ||
        DS4_N_VOCAB != 248320 || DS4_N_HEAD != 24 ||
        DS4_N_HEAD_KV != 2 || DS4_N_HEAD_DIM != 256 ||
        DS4_N_ROT != 64 || DS4_N_EXPERT != 512 ||
        DS4_N_EXPERT_USED != 10 || DS4_N_FF_EXP != 640 ||
        DS4_N_FF_SHEXP != 640 || DS4_N_HC != 4 ||
        DS4_N_INDEXER_HEAD != 4 || DS4_N_INDEXER_HEAD_DIM != 128 ||
        DS4_N_INDEXER_TOP_K != 2048 || DS4_N_FULL_ATTN_COUNT != 12) {
        fprintf(stderr, "Qwen4Exp profile selection produced unexpected constants\n");
        model_close(&model);
        return 1;
    }

    uint32_t full = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const bool expected = (il % 4u) == 3u;
        if (ds4_qwen4exp_layer_is_full_attention(il) != expected) {
            fprintf(stderr, "Qwen4Exp layer %u has the wrong attention branch\n", il);
            model_close(&model);
            return 1;
        }
        if (expected) full++;
    }
    if (full != 12u) {
        fprintf(stderr, "Qwen4Exp full-attention layer count is %u, expected 12\n", full);
        model_close(&model);
        return 1;
    }

    const uint64_t short_plan = qwen4exp_graph_bytes_estimate(4096u, 256u);
    const uint64_t long_plan = qwen4exp_graph_bytes_estimate(262144u, 256u);
    const uint64_t wide_plan = qwen4exp_graph_bytes_estimate(262144u, 512u);
    if (short_plan == 0u || long_plan <= short_plan || wide_plan <= long_plan ||
        long_plan < (UINT64_C(12) << 30) || long_plan > (UINT64_C(18) << 30)) {
        fprintf(stderr,
                "Qwen4Exp graph memory plan is invalid: short=%.2f long=%.2f wide=%.2f GiB\n",
                (double)short_plan / 1073741824.0,
                (double)long_plan / 1073741824.0,
                (double)wide_plan / 1073741824.0);
        model_close(&model);
        return 1;
    }
    if (!qwen4exp_ple_cache_mb_valid(512u) ||
        !qwen4exp_ple_cache_mb_valid(1024u) ||
        !qwen4exp_ple_cache_mb_valid(2048u) ||
        qwen4exp_ple_cache_mb_valid(511u) ||
        qwen4exp_ple_cache_mb_valid(513u) ||
        qwen4exp_ple_cache_mb_valid(4096u)) {
        fprintf(stderr, "Qwen4Exp PLE cache-size contract is invalid\n");
        model_close(&model);
        return 1;
    }

    ds4_str quantization = {0};
    const bool external_ple =
        model_get_string(&model, "general.quantization", &quantization) &&
        qwen4exp_ssd_precision_types(
            quantization, NULL, NULL, NULL, NULL);
    if (external_ple) {
        if (!config_validate_qwen4exp_external_ple(&model)) {
            fprintf(stderr, "SSD-PLE variant was not recognized as external\n");
            model_close(&model);
            return 1;
        }
        printf("Qwen3.8-Flash-Next SSD-PLE contract: valid "
               "(BF16 sidecar, 128 resident PLE tables absent, I64 controls present)\n");
    }

    ds4_weights weights;
    weights_bind(&weights, &model, false, 0, DS4_N_LAYER - 1u,
                 true, false);
    if (!weights.token_embd || !weights.output ||
        !weights.qwen_input_hc.norm ||
        !weights.qwen_mtp_fc_embedding ||
        !weights.qwen_mtp.qwen_qsa.output) {
        fprintf(stderr, "Qwen4Exp top-level or MTP weight binding is incomplete\n");
        model_close(&model);
        return 1;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &weights.layer[il];
        if (!l->qwen_attn_hc.inject || !l->qwen_ffn_hc.inject ||
            !l->ffn_gate_exps || !l->ffn_down_exps ||
            !l->ffn_down_exps_tail || !l->ffn_shexp_gate_inp) {
            fprintf(stderr, "Qwen4Exp layer %u weight binding is incomplete\n", il);
            model_close(&model);
            return 1;
        }
        if (ds4_qwen4exp_layer_is_full_attention(il)) {
            if (!l->qwen_qsa.output || l->qwen_linear_attn.out) {
                fprintf(stderr, "Qwen4Exp layer %u bound the wrong attention branch\n", il);
                model_close(&model);
                return 1;
            }
        } else if (!l->qwen_linear_attn.out || l->qwen_qsa.output) {
            fprintf(stderr, "Qwen4Exp layer %u bound the wrong attention branch\n", il);
            model_close(&model);
            return 1;
        }
        if ((il == 1u) != (l->qwen_ple.key != NULL)) {
            fprintf(stderr, "Qwen4Exp layer %u has the wrong PLE binding\n", il);
            model_close(&model);
            return 1;
        }
    }

    printf("Qwen3.8-Flash-Next metadata: valid "
           "(48 layers, 36 GDN + 12 QSA, 512E top-10, PLE+vision+MTP present)\n");
    printf("Qwen3.8-Flash-Next text weights: valid "
           "(48 attention/HC/MoE blocks, PLE projections+controls, MTP bound)\n");
    model_close(&model);
    return 0;
}
