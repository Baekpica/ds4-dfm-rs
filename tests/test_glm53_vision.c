/* Exact-sidecar CUDA smoke for the GLM-5.3 vision encoder. */
#include "../ds4.c"

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

    float *patches = xmalloc(4u * 1176u * sizeof(*patches));
    float *out = xmalloc(4096u * sizeof(*out));
    for (uint32_t i = 0; i < 4u * 1176u; i++)
        patches[i] = (float)((int)(i % 31u) - 15) / 31.0f;

    int ok = ds4_gpu_glm53_vision_encode(
            out, patches, 2u, 2u, model.map, model.size, &weights);
    double energy = 0.0;
    for (uint32_t i = 0; ok && i < 4096u; i++) {
        if (!isfinite(out[i])) ok = 0;
        energy += fabs((double)out[i]);
    }
    if (!ok || !(energy > 0.0))
        fprintf(stderr, "GLM-5.3 vision CUDA output is invalid\n");
    else
        printf("GLM-5.3 vision CUDA: finite embedding (energy %.6f)\n", energy);

    free(out);
    free(patches);
    ds4_gpu_cleanup();
    model_close(&model);
    return ok ? 0 : 1;
}
