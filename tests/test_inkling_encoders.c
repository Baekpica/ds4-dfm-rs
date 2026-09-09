/* Actual MQ85GB media weights versus an independent CPU BF16 oracle. */
#include "../ds4.c"

enum { REF_IMAGES = 3, REF_AUDIO = 5, PIXELS = 2 * 40 * 40 * 3,
       AUDIO_BINS = 80, WIDTH = 4096, WEIGHTS = 10,
       MANY_IMAGES = 17, MANY_AUDIO = 65, IMAGE_SCRATCH = 16384 };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static void *read_file(const char *dir, const char *name, size_t *bytes) {
    char path[1024];
    CHECK(snprintf(path, sizeof(path), "%s/%s", dir, name) < (int)sizeof(path));
    FILE *fp = fopen(path, "rb"); CHECK(fp);
    CHECK(fseek(fp, 0, SEEK_END) == 0);
    long length = ftell(fp); CHECK(length > 0);
    rewind(fp);
    void *data = malloc((size_t)length); CHECK(data);
    CHECK(fread(data, 1, (size_t)length, fp) == (size_t)length);
    CHECK(fclose(fp) == 0); *bytes = (size_t)length;
    return data;
}

static void check_ref(const char *dir, const char *name, const float *got, size_t count) {
    size_t bytes;
    float *want = read_file(dir, name, &bytes); CHECK(bytes == count * sizeof(float));
    double e2 = 0, r2 = 0, maximum = 0, peak = 0;
    for (size_t i = 0; i < count; i++) {
        CHECK(isfinite(got[i]) && isfinite(want[i]));
        uint32_t bits; memcpy(&bits, &got[i], sizeof(bits)); CHECK((bits & 0xffffu) == 0);
        double diff = (double)got[i] - want[i];
        e2 += diff * diff; r2 += (double)want[i] * want[i];
        maximum = fmax(maximum, fabs(diff)); peak = fmax(peak, fabs(want[i]));
    }
    CHECK(r2 > 0 && peak > 0);
    const double rel = sqrt(e2 / r2);
    printf("%s: %zu values rel_rms=%g max_abs=%g max/peak=%g\n", name, count, rel, maximum, maximum / peak);
    /* BF16 stage boundaries amplify different FP32/FP64 reduction orders. */
    CHECK(rel < 0.005 && maximum / peak < 0.01);
    free(want);
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    size_t bytes;
    void *map = read_file(argv[1], "weights.bin", &bytes);
    CHECK(bytes >= 4096 && memcmp(map, "INKMEDIA", 8) == 0);
    uint32_t header[2]; memcpy(header, (char *)map + 8, sizeof(header));
    CHECK(header[0] == 1 && header[1] == WEIGHTS);
    ds4_model m = {0}; m.map = map; m.size = bytes;
    ds4_tensor t[WEIGHTS] = {0};
    for (unsigned i = 0; i < WEIGHTS; i++) {
        uint64_t entry[4]; memcpy(entry, (char *)map + 16 + i * sizeof(entry), sizeof(entry));
        t[i].abs_offset = entry[0]; t[i].dim[0] = entry[1]; t[i].dim[1] = entry[2];
        t[i].bytes = entry[3]; t[i].type = DS4_TENSOR_BF16;
        t[i].ndim = entry[2] == 1 ? 1 : 2;
        CHECK(entry[0] < bytes && entry[3] <= bytes - entry[0]);
    }
    ds4_inkling_weights w = {0};
    for (unsigned i = 0; i < INKLING_IMAGE_STAGES; i++) {
        w.image_linear[i] = &t[i]; w.image_norm[i] = &t[i + 4];
    }
    w.audio_embed = &t[8]; w.audio_norm = &t[9];
    float *pixels = read_file(argv[1], "pixels.f32", &bytes);
    CHECK(bytes == REF_IMAGES * PIXELS * sizeof(float));
    int32_t *ids = read_file(argv[1], "audio.i32", &bytes);
    CHECK(bytes == REF_AUDIO * AUDIO_BINS * sizeof(int32_t));
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(map, m.size));

    const size_t scratch = REF_IMAGES * IMAGE_SCRATCH * sizeof(float);
    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(scratch), *b = ds4_gpu_tensor_alloc(scratch);
    float *stage = malloc(scratch); CHECK(a && b && stage);
    CHECK(ds4_gpu_tensor_write(a, 0, pixels, REF_IMAGES * PIXELS * sizeof(float)));
    const unsigned lengths[] = {2 * 8 * 8 * 128, 2 * 4 * 4 * 320, 2 * 4800, WIDTH};
    for (unsigned i = 0; i < INKLING_IMAGE_STAGES; i++) {
        CHECK(inkling_image_stage(a, b, a, &m, &w, i, REF_IMAGES));
        size_t count = REF_IMAGES * lengths[i];
        CHECK(ds4_gpu_tensor_read(a, 0, stage, count * sizeof(float)));
        char name[64]; snprintf(name, sizeof(name), "image-stage-%u.f32", i);
        check_ref(argv[1], name, stage, count);
    }
    float encoded[REF_IMAGES * WIDTH], audio[REF_AUDIO * WIDTH];
    CHECK(inkling_image_encode(encoded, pixels, REF_IMAGES, &m, &w));
    CHECK(memcmp(encoded, stage, sizeof(encoded)) == 0);
    CHECK(inkling_audio_encode(audio, ids, REF_AUDIO, &m, &w));
    check_ref(argv[1], "audio-output.f32", audio, REF_AUDIO * WIDTH);

    /* Cross both workspace chunk boundaries; each item remains independent. */
    float *many = malloc(MANY_IMAGES * PIXELS * sizeof(float));
    float *result = malloc(MANY_AUDIO * WIDTH * sizeof(float));
    int32_t *codes = malloc(MANY_AUDIO * AUDIO_BINS * sizeof(int32_t));
    CHECK(many && result && codes);
    for (unsigned i = 0; i < MANY_IMAGES; i++) {
        memcpy(many + i * PIXELS, pixels + (i % REF_IMAGES) * PIXELS, PIXELS * sizeof(float));
    }
    CHECK(inkling_image_encode(result, many, MANY_IMAGES, &m, &w));
    for (unsigned i = 0; i < MANY_IMAGES; i++) {
        CHECK(memcmp(result + i * WIDTH, encoded + (i % REF_IMAGES) * WIDTH, WIDTH * sizeof(float)) == 0);
    }
    for (unsigned i = 0; i < MANY_AUDIO; i++) {
        memcpy(codes + i * AUDIO_BINS, ids + (i % REF_AUDIO) * AUDIO_BINS, AUDIO_BINS * sizeof(int32_t));
    }
    CHECK(inkling_audio_encode(result, codes, MANY_AUDIO, &m, &w));
    for (unsigned i = 0; i < MANY_AUDIO; i++) {
        CHECK(memcmp(result + i * WIDTH, audio + (i % REF_AUDIO) * WIDTH, WIDTH * sizeof(float)) == 0);
    }
    CHECK(!inkling_image_encode(result, pixels, 0, &m, &w));
    CHECK(!inkling_audio_encode(result, ids, 0, &m, &w));
    codes[MANY_AUDIO * AUDIO_BINS - 1] = 16;
    result[0] = 123.0f;
    CHECK(!inkling_audio_encode(result, codes, MANY_AUDIO, &m, &w));
    CHECK(result[0] == 123.0f); /* Invalid later frame must fail before work. */
    many[MANY_IMAGES * PIXELS - 1] = NAN;
    CHECK(!inkling_image_encode(result, many, MANY_IMAGES, &m, &w));
    CHECK(result[0] == 123.0f);
    ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b);
    ds4_gpu_cleanup();
    free(stage); free(many); free(result); free(codes); free(pixels); free(ids); free(map);
    puts("Inkling real-weight encoders and cross-chunk item order passed");
    return 0;
}
