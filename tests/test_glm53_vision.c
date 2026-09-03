/* Exact-sidecar CUDA smoke for the GLM-5.3 vision encoder. */
#include "../ds4.c"

static const uint8_t png_1x1[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x04, 0x00, 0x00, 0x00, 0xb5, 0x1c, 0x0c, 0x02, 0x00, 0x00, 0x00,
    0x0b, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0x64, 0xf8, 0x0f, 0x00,
    0x01, 0x05, 0x01, 0x01, 0x27, 0x18, 0xe3, 0x66, 0x00, 0x00, 0x00, 0x00,
    0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <GLM-5.3-Flash-Vision-Encoder.gguf>\n",
                argv[0]);
        return 2;
    }

    ds4_model model;
    ds4_glm53_vision_weights weights;
    model_open(&model, argv[1], true, false);
    glm53_vision_weights_bind(&weights, &model);
    if (!ds4_gpu_init() || !ds4_gpu_set_aux_model_map_range(
            model.map, model.size, model.tensor_data_pos,
            model.size - model.tensor_data_pos)) {
        fprintf(stderr, "GLM-5.3 vision CUDA setup failed\n");
        model_close(&model);
        return 1;
    }

    char error[160] = {0};
    ds4_glm53_vision_host host = {0};
    if (!glm53_vision_host_prepare(png_1x1, sizeof(png_1x1),
                                   &host, error, sizeof(error))) {
        fprintf(stderr, "GLM-5.3 image preprocessing failed: %s\n", error);
        ds4_gpu_cleanup();
        model_close(&model);
        return 1;
    }
    const uint64_t output_values = (uint64_t)host.info.token_count * 4096u;
    float *out = xmalloc((size_t)output_values * sizeof(*out));

    int ok = ds4_gpu_glm53_vision_encode(
            out, host.patches, host.info.grid_height, host.info.grid_width,
            model.map, model.size, &weights);
    double energy = 0.0;
    for (uint64_t i = 0; ok && i < output_values; i++) {
        if (!isfinite(out[i])) ok = 0;
        energy += fabs((double)out[i]);
    }
    if (!ok || !(energy > 0.0))
        fprintf(stderr, "GLM-5.3 vision CUDA output is invalid\n");
    else
        printf("GLM-5.3 vision CUDA: %u finite image embeddings (energy %.6f)\n",
               host.info.token_count, energy);

    free(out);
    glm53_vision_host_free(&host);
    ds4_gpu_cleanup();
    model_close(&model);
    return ok ? 0 : 1;
}
