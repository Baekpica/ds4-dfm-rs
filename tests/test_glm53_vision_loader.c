/* Strict loader check for the supported GLM-5.3 vision sidecar. */
#include "../ds4.c"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <GLM-5.3-Flash-Vision-Encoder.gguf>\n",
                argv[0]);
        return 2;
    }

    ds4_model model;
    ds4_glm53_vision_weights weights;
    model_open(&model, argv[1], false, false);
    glm53_vision_weights_bind(&weights, &model);

    const uint32_t image = required_u32(
            &model, "glm5-next-vision.image_token_id");
    const uint32_t start = required_u32(
            &model, "glm5-next-vision.image_start_token_id");
    const uint32_t end = required_u32(
            &model, "glm5-next-vision.image_end_token_id");
    if (image != 154854u || start != 154830u || end != 154831u ||
        weights.patch_weight == 0u || weights.merger_down == 0u ||
        weights.layer[0].qkv_weight == 0u ||
        weights.layer[23].down_weight == 0u) {
        fprintf(stderr, "GLM-5.3 vision binding is incomplete\n");
        model_close(&model);
        return 1;
    }

    printf("GLM-5.3 vision encoder: valid (347 BF16 tensors, 24 layers)\n");
    model_close(&model);
    return 0;
}
