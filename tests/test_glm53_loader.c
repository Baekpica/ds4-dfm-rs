/* GLM 5.3 Flash metadata smoke. This maps only the GGUF header and directory. */
#include "../ds4.c"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <GLM-5.3-Flash-Q2.gguf>\n", argv[0]);
        return 2;
    }

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);

    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM53 ||
        DS4_MODEL_VARIANT != DS4_VARIANT_GLM53_FLASH ||
        DS4_N_LAYER != 46 || DS4_N_EMBD != 4096 || DS4_N_VOCAB != 154880 ||
        DS4_N_HEAD != 64 || DS4_N_HEAD_KV != 1 || DS4_N_HEAD_DIM != 512 ||
        DS4_N_VALUE_DIM != 256 || DS4_N_ROT != 0 || DS4_N_LORA_Q != 1536 ||
        DS4_N_EXPERT != 288 || DS4_N_EXPERT_USED != 8 ||
        DS4_N_FF_EXP != 2048 || DS4_N_FF_DENSE != 12288 ||
        DS4_N_LEADING_DENSE != 3 || DS4_N_NEXTN_PREDICT != 1 ||
        DS4_N_INDEXER_HEAD != 32 || DS4_N_INDEXER_HEAD_DIM != 128 ||
        DS4_N_INDEXER_TOP_K != 2048 || DS4_N_HC != 4 ||
        DS4_N_HC_SINKHORN_ITER != 20 || DS4_N_KV_LORA != 512 ||
        DS4_N_KEY_MLA != 256 || DS4_N_VALUE_MLA != 256 ||
        DS4_N_KDA_HEAD_DIM != 128 || DS4_N_SSM_CONV != 4 || DS4_USE_ROPE ||
        fabsf(DS4_RMS_EPS - 1.0e-5f) > 1.0e-12f ||
        fabsf(DS4_HC_EPS - 1.0e-6f) > 1.0e-12f ||
        fabsf(DS4_KDA_GATE_CLAMP_MIN - (-5.0f)) > 1.0e-6f) {
        fprintf(stderr, "GLM 5.3 profile selection produced unexpected constants\n");
        model_close(&model);
        return 1;
    }

    uint32_t kda = 0, dsa = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (ds4_glm53_layer_is_kda(il)) kda++;
        else dsa++;
    }
    if (kda != 34 || dsa != 12 || ds4_glm53_layer_is_kda(45)) {
        fprintf(stderr, "GLM 5.3 hybrid layer schedule is wrong\n");
        model_close(&model);
        return 1;
    }

    printf("GLM 5.3 metadata: valid (%u shards, 34 KDA, 12 DSA)\n",
           model.split_count);
    model_close(&model);
    return 0;
}
