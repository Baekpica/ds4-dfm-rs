/* GLM-5.3 image geometry and patch-layout smoke. */
#include "../ds4.c"

static const uint8_t png_1x1[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x04, 0x00, 0x00, 0x00, 0xb5, 0x1c, 0x0c, 0x02, 0x00, 0x00, 0x00,
    0x0b, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0x64, 0xf8, 0x0f, 0x00,
    0x01, 0x05, 0x01, 0x01, 0x27, 0x18, 0xe3, 0x66, 0x00, 0x00, 0x00, 0x00,
    0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

int main(void) {
    char error[160] = {0};
    ds4_vision_image_info info = {0};
    if (!glm53_vision_probe(png_1x1, sizeof(png_1x1), &info,
                            error, sizeof(error)) ||
        info.source_width != 1u || info.source_height != 1u ||
        info.padded_width != 112u || info.padded_height != 112u ||
        info.grid_width != 8u || info.grid_height != 8u ||
        info.token_count != 16u) {
        fprintf(stderr, "GLM-5.3 image probe failed: %s\n", error);
        return 1;
    }

    ds4_glm53_vision_host host = {0};
    if (!glm53_vision_host_prepare(png_1x1, sizeof(png_1x1),
                                   &host, error, sizeof(error))) {
        fprintf(stderr, "GLM-5.3 image preprocessing failed: %s\n", error);
        return 1;
    }
    int ok = host.patches != NULL;
    for (uint32_t i = 0; ok && i < 196u; i++) {
        ok = isfinite(host.patches[i]) &&
             host.patches[i] == host.patches[196u + i];
    }
    glm53_vision_host_free(&host);
    if (!ok || glm53_vision_probe((const uint8_t *)"bad", 3u, &info,
                                  error, sizeof(error))) {
        fprintf(stderr, "GLM-5.3 image patch layout validation failed\n");
        return 1;
    }
    puts("GLM-5.3 image preprocessing: valid (16 tokens)");
    return 0;
}
