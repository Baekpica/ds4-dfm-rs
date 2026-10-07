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

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM image FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static void check_exif(unsigned orientation) {
    char path[80], error[160] = {0};
    snprintf(path, sizeof(path), "tests/fixtures/glm53/exif%u.jpg", orientation);
    FILE *fp = fopen(path, "rb");
    CHECK(fp && fseek(fp, 0, SEEK_END) == 0);
    const long bytes = ftell(fp);
    CHECK(bytes > 0 && fseek(fp, 0, SEEK_SET) == 0);
    uint8_t *data = xmalloc((size_t)bytes);
    CHECK(fread(data, 1u, (size_t)bytes, fp) == (size_t)bytes);
    fclose(fp);
    ds4_vision_image_info info;
    CHECK(glm53_vision_probe(data, (size_t)bytes, &info, error, sizeof(error)));
    CHECK(info.source_width == 224u && info.source_height == 112u);
    CHECK(info.content_width == 224u && info.content_height == 112u);
    CHECK(info.grid_width == 16u && info.grid_height == 8u && info.token_count == 32u);
    ds4_glm53_vision_host host = {0};
    CHECK(glm53_vision_host_prepare(data, (size_t)bytes, &host, error, sizeof(error)));
    int width, height, channels;
    uint8_t *raw = stbi_load_from_memory(data, (int)bytes, &width, &height, &channels, 3);
    CHECK(raw && width == 112 && height == 224);
    /* First two 2x2 groups are TL,TR,BL,BR in display coordinates. */
    const unsigned xy[8][2] = {{0,0},{1,0},{0,1},{1,1},{2,0},{3,0},{2,1},{3,1}};
    const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float stddev[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    for (unsigned row = 0u; row < 8u; row++) {
        const unsigned dx = xy[row][0] * 14u + 7u, dy = xy[row][1] * 14u + 7u;
        const unsigned sx = orientation == 6u ? dy : 111u - dy;
        const unsigned sy = orientation == 6u ? 223u - dx : dx;
        for (unsigned c = 0u; c < 3u; c++) {
            const float expected = ((float)raw[((size_t)sy * 112u + sx) * 3u + c] / 255.0f - mean[c]) / stddev[c];
            for (unsigned t = 0u; t < 2u; t++) {
                const size_t at = (size_t)row * 1176u + c * 392u + t * 196u + 7u * 14u + 7u;
                CHECK(host.patches[at] == expected);
            }
        }
    }
    stbi_image_free(raw); free(data); glm53_vision_host_free(&host);
}

static void check_orphans(void) {
    char error[160] = {0};
    int data[40] = {0}; float dummy = 0.0f;
    for (unsigned i = 2u; i < 18u; i++) { data[i] = 154854; }
    for (unsigned i = 20u; i < 36u; i++) { data[i] = 154854; }
    ds4_tokens prompt = {.v=data, .len=40, .cap=40};
    ds4_vision_span spans[2] = {
        {.token_start=2, .embedding={.data=&dummy,.token_count=16,.grid_width=8,.grid_height=8}},
        {.token_start=20, .embedding={.data=&dummy,.token_count=16,.grid_width=8,.grid_height=8}}
    };
    CHECK(glm53_vision_spans_validate(&prompt, spans, 2u, 154854, error, sizeof(error)));
    const unsigned orphan[] = {0u,18u,39u};
    for (unsigned i = 0u; i < 3u; i++) {
        data[orphan[i]] = 154854;
        CHECK(!glm53_vision_spans_validate(&prompt, spans, 2u, 154854, error, sizeof(error)));
        data[orphan[i]] = 0;
    }
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "exif") == 0) {
        check_exif(6u); check_exif(8u);
        puts("GLM image: EXIF6/8 display geometry and grouped temporal RGB patches passed"); return 0;
    }
    if (argc == 2 && strcmp(argv[1], "spans") == 0) {
        check_orphans(); puts("GLM image: orphan markers before/between/after spans rejected"); return 0;
    }
    check_exif(6u); check_exif(8u); check_orphans();
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

    int token_data[20] = {0};
    for (uint32_t i = 2u; i < 18u; i++) token_data[i] = 154854;
    ds4_tokens tokens = {.v = token_data, .len = 20, .cap = 20};
    float dummy = 0.0f;
    ds4_vision_span span = {
        .token_start = 2u,
        .embedding = {
            .data = &dummy,
            .token_count = 16u,
            .grid_width = 8u,
            .grid_height = 8u,
        },
    };
    if (!glm53_vision_spans_validate(
            &tokens, &span, 1u, 154854, error, sizeof(error))) {
        fprintf(stderr, "GLM-5.3 vision span validation failed: %s\n", error);
        return 1;
    }
    token_data[17] = 1;
    if (glm53_vision_spans_validate(
            &tokens, &span, 1u, 154854, error, sizeof(error))) {
        fprintf(stderr, "GLM-5.3 vision span accepted a non-image token\n");
        return 1;
    }
    puts("GLM-5.3 image preprocessing: valid (16 tokens)");
    return 0;
}
