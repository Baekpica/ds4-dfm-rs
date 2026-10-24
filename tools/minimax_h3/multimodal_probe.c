/*
 * h3_multimodal_probe — run the reference's released `<Picture 1>: `
 * presentation and Qwen decoder, and hash what the port must reproduce.
 *
 * The engine compiled here is `/data/h3.c` at eae5d54, unmodified; only this
 * driver is ours (`h3_multimodal.h`, `h3_vision_encoder.h`).
 *
 * Usage:
 *   h3_multimodal_probe <text_encoder_dir> <image> <prompt>
 *                       [--height H] [--width W]
 *
 * Output:
 *   GRID <h>x<w> merged=<merged rows>
 *   TOKENS <rows> spans=1 zeros=<tag zeros>
 *   PARITY tokens= width= values= fnv= first= last=     the BF16 rows
 *   TAGS tokens= fnv=                                   the modality tags
 *   SECONDS <wall clock, vision tower included>
 *
 * The hash conventions are the probe family's: BF16 payload bytes for PARITY,
 * the tag bytes as stored for TAGS, and the nonstandard FNV seed recorded in
 * tools/minimax_h3/text_probe.c:37-38.
 */

#include "h3_ffmpeg.h"
#include "h3_multimodal.h"
#include "h3_tokenizer.h"
#include "h3_vision_encoder.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
            "usage: h3_multimodal_probe <text_encoder_dir> <image> <prompt> "
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

int main(int argc, char **argv) {
    if (argc < 4) {
        usage();
        return 2;
    }

    const char *weights = argv[1];
    const char *image = argv[2];
    const char *prompt = argv[3];
    int height = 512;
    int width = 384;

    for (int index = 4; index + 1 < argc; index += 2) {
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
    char tokenizer_path[1024];
    /* The artifact keeps the tokenizer beside the shard it belongs to:
     * `FL2VA/tokenizer/tokenizer.json` next to `FL2VA/text_encoder/`. */
    int path_length = snprintf(tokenizer_path, sizeof(tokenizer_path),
                               "%s/../tokenizer/tokenizer.json", weights);
    if (path_length < 0 || (size_t)path_length >= sizeof(tokenizer_path)) {
        fprintf(stderr, "tokenizer path is too long\n");
        return 1;
    }

    h3_tokenizer *tokenizer =
        h3_tokenizer_load(tokenizer_path, error, sizeof(error));
    if (!tokenizer) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }

    float *pixels = NULL;
    if (!h3_ffmpeg_read_image_f32(image, width, height, H3_IMAGE_FIT_STRETCH,
                                  &pixels, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        h3_tokenizer_free(tokenizer);
        return 1;
    }

    double started = seconds();
    h3_vision_output vision;
    h3_text_embedding text;
    memset(&text, 0, sizeof(text));

    if (!h3_vision_encode_bf16(weights, "h3_shaders.metal", pixels, 1, height,
                               width, NULL, NULL, &vision, error,
                               sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        free(pixels);
        h3_tokenizer_free(tokenizer);
        return 1;
    }
    free(pixels);

    int ok = h3_multimodal_encode_fl2va_bf16(
        tokenizer, weights, "h3_shaders.metal", prompt, &vision, 1, NULL, NULL,
        &text, error, sizeof(error));
    double elapsed = seconds() - started;

    if (!ok) {
        fprintf(stderr, "%s\n", error);
        h3_vision_output_free(&vision);
        h3_tokenizer_free(tokenizer);
        return 1;
    }

    size_t tokens = text.tokens;
    size_t width_hidden = text.width;
    size_t values = tokens * width_hidden;
    size_t zeros = 0;

    for (size_t index = 0; index < tokens; index++) {
        if (text.tags && text.tags[index] == 0) zeros++;
    }

    char first[128], last[128];
    edges(first, sizeof(first), text.values, values, 0);
    edges(last, sizeof(last), text.values, values, values > 8 ? values - 8 : 0);

    printf("GRID %dx%d merged=%zu\n", vision.grid_h, vision.grid_w,
           vision.tokens);
    printf("TOKENS %zu spans=1 zeros=%zu\n", tokens, zeros);
    printf("PARITY tokens=%zu width=%zu values=%zu fnv=%016llx first=%s "
           "last=%s\n",
           tokens, width_hidden, values,
           (unsigned long long)fnv(text.values, values * sizeof(uint16_t)),
           first, last);
    printf("TAGS tokens=%zu fnv=%016llx\n", tokens,
           (unsigned long long)fnv(text.tags, tokens));
    printf("SECONDS %.3f\n", elapsed);

    h3_text_embedding_free(&text);
    h3_vision_output_free(&vision);
    h3_tokenizer_free(tokenizer);
    return 0;
}
