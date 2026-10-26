/*
 * h3_vision_probe — run one reference Qwen vision forward and hash everything
 * the port must reproduce.
 *
 * The engine compiled here is `/data/h3.c` at eae5d54, unmodified; only this
 * driver is ours (`h3_vision_encoder.h`).
 *
 * Usage:
 *   h3_vision_probe <text_encoder_dir> <image> [--height H] [--width W]
 *
 * Output:
 *   GRID <h>x<w> rows=<patch rows> merged=<merged rows>
 *   PARITY tokens= width= values= fnv= first= last=        the final merger
 *   DEEPSTACK <i> fnv= first= last=                        one per deepstack
 *   SECONDS <wall clock>
 *
 * The hash covers the reference's own BF16 payload as it hands it back
 * (`h3_vision_output.merged` / `.deepstack[]`), so it is byte-for-byte the
 * convention the port's `h3_vision_gate` uses. The image is decoded by the
 * reference's FFmpeg reader (`h3_ffmpeg.c:143-283`), channel-major F32, so
 * both sides see the same pixels when they are given the same file and shape.
 */

#include "h3_ffmpeg.h"
#include "h3_vision_encoder.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Same seed and prime as tools/minimax_h3/text_probe.c:37-38, which records
 * why: the probe family hashes with the reference's nonstandard seed. */
static const uint64_t FNV_SEED = UINT64_C(1469598103934665603);
static const uint64_t FNV_PRIME = UINT64_C(1099511628211);

static uint64_t fnv(const void *data, size_t bytes) {
    const unsigned char *values = data;
    uint64_t hash = FNV_SEED;

    for (size_t index = 0; index < bytes; index++) {
        hash ^= values[index];
        hash *= FNV_PRIME;
    }

    return hash;
}

static double seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static int parse_int(const char *text, int *value) {
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (!end || *end || parsed < INT32_MIN || parsed > INT32_MAX) return 0;
    *value = (int)parsed;
    return 1;
}

static void usage(void) {
    fprintf(stderr,
            "usage: h3_vision_probe <text_encoder_dir> <image> "
            "[--height H] [--width W]\n");
}

/* Eight BF16 words from `offset`, as the gate prints them. */
static void edges(char *buffer, size_t size, const uint16_t *values,
                  size_t count, size_t offset) {
    size_t end = offset + 8 < count ? offset + 8 : count;
    size_t used = 0;

    buffer[0] = '\0';
    for (size_t index = offset; index < end; index++)
        used += (size_t)snprintf(buffer + used, size - used, "%s%04x",
                                 index == offset ? "" : ".", values[index]);
}

static void print_tensor(const char *label, const uint16_t *values,
                         size_t elements, size_t tokens, size_t width) {
    char first[128], last[128];
    edges(first, sizeof(first), values, elements, 0);
    edges(last, sizeof(last), values, elements,
          elements > 8 ? elements - 8 : 0);
    printf("%s tokens=%zu width=%zu values=%zu fnv=%016llx first=%s last=%s\n",
           label, tokens, width, elements,
           (unsigned long long)fnv(values, elements * sizeof(uint16_t)),
           first, last);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage();
        return 2;
    }

    const char *weights = argv[1];
    const char *image = argv[2];
    int height = 512;
    int width = 192;

    for (int index = 3; index + 1 < argc; index += 2) {
        if (!strcmp(argv[index], "--height")) {
            if (!parse_int(argv[index + 1], &height)) return 2;
        } else if (!strcmp(argv[index], "--width")) {
            if (!parse_int(argv[index + 1], &width)) return 2;
        } else {
            usage();
            return 2;
        }
    }

    char error[1024];
    float *pixels = NULL;

    if (!h3_ffmpeg_read_image_f32(image, width, height, H3_IMAGE_FIT_STRETCH,
                                  &pixels, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }

    double started = seconds();
    h3_vision_output output;
    int ok = h3_vision_encode_bf16(weights, "h3_shaders.metal", pixels, 1,
                                   height, width, NULL, NULL, &output, error,
                                   sizeof(error));
    double elapsed = seconds() - started;
    free(pixels);

    if (!ok) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }

    size_t tokens = output.tokens;
    size_t values = tokens * H3_VISION_OUTPUT_WIDTH;

    printf("GRID %dx%d rows=%zu merged=%zu\n", output.grid_h, output.grid_w,
           (size_t)output.grid_h * (size_t)output.grid_w, tokens);
    print_tensor("PARITY", output.merged, values, tokens,
                 H3_VISION_OUTPUT_WIDTH);

    for (unsigned index = 0; index < H3_VISION_DEEPSTACKS; index++) {
        char label[32];
        snprintf(label, sizeof(label), "DEEPSTACK %u", index);
        print_tensor(label, output.deepstack[index], values, tokens,
                     H3_VISION_OUTPUT_WIDTH);
    }

    printf("SECONDS %.3f\n", elapsed);
    h3_vision_output_free(&output);
    return 0;
}
