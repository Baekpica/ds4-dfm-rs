/* GPU feature splicing at text/prefill/MTP-shift boundaries, no weights. */
#include "../ds4.c"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "Step media FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)
int main(void) {
    CHECK(ds4_gpu_init());
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
    media.count = 2; media.spans[1].feature = FEATURES;
    CHECK(!step37_media_apply(&media, out, 0, ROWS));
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(media.features); ds4_gpu_cleanup();
    free(input); free(features); free(got);
    puts("Step media: all 128 boundary/width splices preserve every float byte PASS");
}
