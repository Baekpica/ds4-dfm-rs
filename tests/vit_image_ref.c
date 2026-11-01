/* The C image path that C hosts still run (stb_image decode, Pillow-style
 * resize, the Qwen/Ling/GLM packers), driven per file so the Rust port in
 * crates/ds4-core/src/vit_image.rs can be compared with it.
 *
 *   vit_image_ref probe qwen|ling|glm FILE   -> plan line
 *   vit_image_ref hash FILE                  -> Qwen pixel hash
 *   vit_image_ref decode FILE OUT            -> RGB8 bytes
 *   vit_image_ref pack qwen|ling|glm FILE OUT -> f32 patch rows
 *   vit_image_ref bench qwen|ling|glm FILE N  -> median ms per pack, peak RSS
 */
#include <sys/resource.h>
#include "../ds4.c"

static uint8_t *read_all(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = n > 0 ? malloc((size_t)n) : NULL;
    if (!data || fread(data, 1, (size_t)n, f) != (size_t)n) {
        free(data);
        data = NULL;
    }
    fclose(f);
    *len = (size_t)n;
    return data;
}

static int write_all(const char *path, const void *data, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    const size_t n = fwrite(data, 1, len, f);
    fclose(f);
    return n == len ? 0 : 1;
}

static int probe(const char *tower, const uint8_t *data, size_t len,
                 ds4_vision_image_info *out) {
    char err[256] = {0};
    if (!strcmp(tower, "glm")) {
        if (!glm53_vision_probe(data, len, out, err, sizeof(err))) goto fail;
        return 0;
    }
    ds4_qwen_image_info q = {0};
    const int rc = !strcmp(tower, "qwen")
        ? ds4_qwen_image_probe(data, len, &q, err, sizeof(err))
        : ds4_ling3vl_image_probe(data, len, (ds4_ling3vl_image_info *)&q, err, sizeof(err));
    if (rc != 0) goto fail;
    *out = (ds4_vision_image_info){
        .source_width = q.source_width, .source_height = q.source_height,
        .content_width = q.resized_width, .content_height = q.resized_height,
        .padded_width = q.resized_width, .padded_height = q.resized_height,
        .grid_height = q.grid_h, .grid_width = q.grid_w, .token_count = q.token_count};
    return 0;
fail:
    fprintf(stderr, "probe: %s\n", err);
    return 1;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Ling encodes on device; its pack is the CPU half of ling3vl_vision_encode_one. */
static float *pack_ling(const uint8_t *data, size_t len, const ds4_vision_image_info *info) {
    int w = 0, h = 0, c = 0;
    uint8_t *rgb = stbi_load_from_memory(data, (int)len, &w, &h, &c, 3);
    if (!rgb) return NULL;
    uint8_t *resized = qwen4exp_resize_rgb_pillow(rgb, (uint32_t)w, (uint32_t)h,
                                                  info->content_width, info->content_height);
    stbi_image_free(rgb);
    const uint32_t rows = info->grid_height * info->grid_width;
    float *patches = xmalloc((size_t)rows * L3V_VIT_PATCH_DIM * sizeof(float));
    int32_t *idx = xmalloc((size_t)rows * 4u * sizeof(int32_t));
    float *wts = xmalloc((size_t)rows * 4u * sizeof(float));
    int32_t *ys = xmalloc((size_t)rows * sizeof(int32_t));
    int32_t *xs = xmalloc((size_t)rows * sizeof(int32_t));
    const bool ok = resized && ling3vl_vision_pack(resized, info->content_width,
                                                   info->grid_height, info->grid_width,
                                                   patches, idx, wts, ys, xs);
    free(resized); free(idx); free(wts); free(ys); free(xs);
    if (!ok) { free(patches); return NULL; }
    return patches;
}

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    const char *cmd = argv[1];
    const bool tower_cmd = !strcmp(cmd, "probe") || !strcmp(cmd, "pack") ||
                           !strcmp(cmd, "bench");
    const char *file = tower_cmd ? argv[3] : argv[2];
    size_t len = 0;
    uint8_t *data = file ? read_all(file, &len) : NULL;
    if (!data) { fprintf(stderr, "cannot read %s\n", file ? file : "(none)"); return 2; }

    if (!strcmp(cmd, "hash")) {
        uint64_t h = 0;
        char err[256] = {0};
        if (ds4_qwen_image_pixel_hash(data, len, &h, err, sizeof(err)) != 0) {
            fprintf(stderr, "hash: %s\n", err);
            return 1;
        }
        printf("%" PRIu64 "\n", h);
        return 0;
    }
    if (!strcmp(cmd, "decode") && argc == 4) {
        int w = 0, h = 0, c = 0;
        uint8_t *rgb = stbi_load_from_memory(data, (int)len, &w, &h, &c, 3);
        if (!rgb) return 1;
        printf("%d %d\n", w, h);
        return write_all(argv[3], rgb, (size_t)w * h * 3u);
    }

    ds4_vision_image_info info = {0};
    if (!tower_cmd || argc < 4 || probe(argv[2], data, len, &info) != 0) return 1;
    printf("%u %u %u %u %u %u %u %u %u\n", info.source_width, info.source_height,
           info.content_width, info.content_height, info.padded_width,
           info.padded_height, info.grid_height, info.grid_width, info.token_count);
    if (!strcmp(cmd, "probe")) return 0;
    if (argc != 5) return 2;

    if (!strcmp(cmd, "bench")) {
        /* Decode, resize and pack as the native prefill does, N times. */
        const int n = atoi(argv[4]);
        double *ms = calloc((size_t)n, sizeof(double));
        for (int i = 0; i < n; i++) {
            const double t0 = now_ms();
            if (!strcmp(argv[2], "ling")) {
                free(pack_ling(data, len, &info));
            } else if (!strcmp(argv[2], "glm")) {
                ds4_glm53_vision_host host = {0};
                char err[256] = {0};
                glm53_vision_host_prepare(data, len, &host, err, sizeof(err));
                glm53_vision_host_free(&host);
            } else {
                ds4_qwen_image_input image = {.data = data, .data_len = len,
                                              .grid_h = info.grid_height, .grid_w = info.grid_width};
                ds4_qwen_vision_host host = {0};
                char err[256] = {0};
                qwen4exp_vision_host_prepare(&image, 1u, &host, err, sizeof(err));
                qwen4exp_vision_host_free(&host);
            }
            ms[i] = now_ms() - t0;
        }
        qsort(ms, (size_t)n, sizeof(double), cmp_double);
        struct rusage ru;
        getrusage(RUSAGE_SELF, &ru);
        printf("c %s median %.2f ms min %.2f ms peak_rss %ld KiB\n", argv[2], ms[n / 2], ms[0], ru.ru_maxrss);
        return 0;
    }

    const uint32_t rows = info.grid_height * info.grid_width;
    if (!strcmp(argv[2], "ling")) {
        float *p = pack_ling(data, len, &info);
        return !p || write_all(argv[4], p, (size_t)rows * L3V_VIT_PATCH_DIM * sizeof(float));
    }
    if (!strcmp(argv[2], "glm")) {
        ds4_glm53_vision_host host = {0};
        char err[256] = {0};
        if (!glm53_vision_host_prepare(data, len, &host, err, sizeof(err))) {
            fprintf(stderr, "pack: %s\n", err);
            return 1;
        }
        return write_all(argv[4], host.patches, (size_t)rows * 1176u * sizeof(float));
    }
    ds4_qwen_image_input image = {.data = data, .data_len = len,
                                  .grid_h = info.grid_height, .grid_w = info.grid_width};
    ds4_qwen_vision_host host = {0};
    char err[256] = {0};
    if (!qwen4exp_vision_host_prepare(&image, 1u, &host, err, sizeof(err))) {
        fprintf(stderr, "pack: %s\n", err);
        return 1;
    }
    return write_all(argv[4], host.patches,
                     (size_t)rows * DS4_QWEN_VISION_PATCH_DIM * sizeof(float));
}
