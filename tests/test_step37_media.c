/* GPU feature splicing at text/prefill/MTP-shift boundaries, no weights. */
#include "../ds4.c"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "Step media FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)
int main(void) {
    CHECK(ds4_gpu_init());
    g_ds4_shape = DS4_SHAPE_STEP37_FLASH;
    const size_t pixel_count = 3 * 504 * 504;
    float *pixels = xcalloc(pixel_count, sizeof(float));
    int tokens[171];
    for (unsigned i = 0; i < 171; i++) { tokens[i] = 7; }
    for (unsigned i = 3; i < 84; i++) { tokens[i] = S37_IMAGE_TOKEN; }
    for (unsigned i = 89; i < 170; i++) { tokens[i] = S37_IMAGE_TOKEN; }
    ds4_tokens prompt = {.v = tokens, .len = 171, .cap = 171};
    ds4_step37_pixels crops[] = {
        {.pixels = pixels, .pixel_count = pixel_count, .token_offset = 3, .token_count = 81, .edge = 504},
        {.pixels = pixels, .pixel_count = pixel_count, .token_offset = 89, .token_count = 81, .edge = 504},
    };
    ds4_step37_media checked = {0};
    CHECK(step37_media_check(&prompt, crops, 2, &checked));
    CHECK(checked.rows == 162 && checked.count == 2 && checked.spans[1].feature == 81);
    CHECK(!step37_media_check(&prompt, crops, 1, &checked));
    CHECK(!step37_media_check(&prompt, crops, 0, &checked));
    crops[1].token_offset = 4; CHECK(!step37_media_check(&prompt, crops, 2, &checked));
    crops[1].token_offset = UINT32_MAX; CHECK(!step37_media_check(&prompt, crops, 2, &checked));
    crops[1].token_offset = 89; crops[1].edge = 728;
    CHECK(!step37_media_check(&prompt, crops, 2, &checked));
    crops[1].edge = 504; crops[1].pixel_count--;
    CHECK(!step37_media_check(&prompt, crops, 2, &checked));
    crops[1].pixel_count++;
    tokens[0] = S37_IMAGE_TOKEN; CHECK(!step37_media_check(&prompt, crops, 2, &checked));
    tokens[0] = -1; CHECK(!step37_media_check(&prompt, crops, 2, &checked));
    tokens[0] = 7; tokens[50] = 7; CHECK(!step37_media_check(&prompt, crops, 2, &checked));
    tokens[50] = S37_IMAGE_TOKEN; pixels[pixel_count - 1] = NAN;
    CHECK(!step37_media_check(&prompt, crops, 2, &checked));
    free(pixels);
    enum { ROWS = 8, FEATURES = 7 };
    float *input = xmalloc(ROWS * S37_HIDDEN * sizeof(float));
    float *features = xmalloc(FEATURES * S37_HIDDEN * sizeof(float));
    float *got = xmalloc(ROWS * S37_HIDDEN * sizeof(float));
    for (unsigned i = 0; i < ROWS * S37_HIDDEN; i++) { input[i] = (float)(i % 83); }
    for (unsigned i = 0; i < FEATURES * S37_HIDDEN; i++) { features[i] = -(float)(1 + i % 131); }
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(ROWS * S37_HIDDEN * sizeof(float));
    ds4_step37_media media = {0};
    media.features = ds4_gpu_tensor_alloc(FEATURES * S37_HIDDEN * sizeof(float));
    media.rows = FEATURES; media.count = 2;
    media.spans[0] = (ds4_step37_image_span){.position = 3, .rows = 5, .feature = 0};
    media.spans[1] = (ds4_step37_image_span){.position = 11, .rows = 2, .feature = 5};
    CHECK(out && media.features && ds4_gpu_tensor_write(media.features, 0, features, FEATURES * S37_HIDDEN * sizeof(float)));
    // Every possible width/position covers inside, outside, clipping both
    // ends, two spans and the +1/+2/+3 shifted predictor embedding positions.
    for (unsigned position = 0; position < 16; position++) {
        for (unsigned n = 1; n <= ROWS; n++) {
            CHECK(ds4_gpu_tensor_write(out, 0, input, ROWS * S37_HIDDEN * sizeof(float)));
            CHECK(step37_media_apply(&media, out, position, n));
            CHECK(ds4_gpu_tensor_read(out, 0, got, ROWS * S37_HIDDEN * sizeof(float)));
            for (unsigned row = 0; row < ROWS; row++) {
                const unsigned absolute = position + row;
                const float *want = input + row * S37_HIDDEN;
                if (row < n && absolute >= 3 && absolute < 8) { want = features + (absolute - 3) * S37_HIDDEN; }
                if (row < n && absolute >= 11 && absolute < 13) { want = features + (absolute - 11 + 5) * S37_HIDDEN; }
                CHECK(!memcmp(got + row * S37_HIDDEN, want, S37_HIDDEN * sizeof(float)));
            }
        }
    }
    CHECK(step37_media_apply(NULL, out, 0, ROWS));
    media.count = S37_MEDIA_CROPS + 1;
    CHECK(!step37_media_apply(&media, out, 0, ROWS));
    // Reused vision scratch must replace attention's segment limit as well
    // as the visible crop shape, including the small-to-large transition.
    ds4_step37_vision vision;
    CHECK(step37_vision_alloc(&vision, 728));
    const unsigned edges[] = {504, 728, 504};
    for (unsigned i = 0; i < 3; i++) {
        CHECK(step37_vision_shape(&vision, edges[i]));
        const unsigned n = (edges[i] / 14) * (edges[i] / 14);
        int *segments = xmalloc(n * sizeof(int));
        CHECK(ds4_gpu_tensor_read(vision.start, 0, segments, n * sizeof(int)));
        for (unsigned j = 0; j < n; j++) { CHECK(segments[j] == 0); }
        CHECK(ds4_gpu_tensor_read(vision.end, 0, segments, n * sizeof(int)));
        for (unsigned j = 0; j < n; j++) { CHECK(segments[j] == (int)n); }
        CHECK(vision.edge == edges[i] && vision.rows == n && vision.output_rows == n / 16);
        CHECK(vision.bytes == step37_vision_bytes(728));
        free(segments);
    }
    CHECK(!step37_vision_shape(&vision, 729));
    step37_vision_free(&vision);
    CHECK(step37_vision_alloc(&vision, 504));
    CHECK(!step37_vision_shape(&vision, 728));
    CHECK(vision.edge == 504 && vision.bytes == step37_vision_bytes(504));
    step37_vision_free(&vision);
    media.count = 2; media.spans[1].feature = FEATURES;
    CHECK(!step37_media_apply(&media, out, 0, ROWS));
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(media.features); ds4_gpu_cleanup();
    free(input); free(features); free(got);
    puts("Step media: all 128 boundary/width splices preserve every float byte PASS");
}
