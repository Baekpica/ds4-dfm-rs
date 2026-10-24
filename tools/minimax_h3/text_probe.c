/*
 * h3_text_probe — run the reference prompt encoder and hash its output.
 *
 * The forward unit's gate replays this golden with the port's own encoder.
 * The encoder compiled here is `/data/h3.c` at eae5d54 (`h3_text_encoder.c`),
 * unmodified; only this driver is ours. On CUDA the shader path is ignored
 * (`h3_gpu_cuda.cu:274`), and the engine passes the same name
 * (`h3.c:1427`).
 *
 * Usage: h3_text_probe <FL2VA/text_encoder> <tokenizer.json> <prompt>
 * Output: the prompt ids, then one TEXT line with the shape, an FNV-1a hash
 * over the BF16 values, both edges, and the wall time.
 */

#include "h3_text_encoder.h"
#include "h3_tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t fnv(const void *data, size_t bytes) {
    const unsigned char *values = data;
    uint64_t hash = UINT64_C(1469598103934665603);

    for (size_t index = 0; index < bytes; index++) {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
    }

    return hash;
}

static double seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static void edges(const char *label, const uint16_t *values, size_t count) {
    printf(" %s=", label);
    for (size_t index = 0; index < count; index++) {
        printf("%s%04x", index ? "." : "", values[index]);
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: h3_text_probe <FL2VA/text_encoder> <tokenizer.json> "
                "<prompt>\n");
        return 2;
    }

    char error[512];
    h3_tokenizer *tokenizer = h3_tokenizer_load(argv[2], error, sizeof(error));
    if (!tokenizer) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }

    uint32_t *ids = NULL;
    size_t count = 0;
    if (!h3_tokenizer_encode(tokenizer, argv[3], 1, &ids, &count, error,
                             sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }

    printf("TOKENS");
    for (size_t index = 0; index < count; index++) {
        printf(" %u", ids[index]);
    }
    putchar('\n');

    h3_text_embedding text;
    memset(&text, 0, sizeof(text));
    double start = seconds();

    if (!h3_text_encode_bf16(argv[1], "h3_shaders.metal", ids, count, NULL,
                             NULL, &text, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }

    double elapsed = seconds() - start;
    size_t values = text.tokens * text.width;

    printf("TEXT tokens=%zu width=%zu values=%zu fnv=%016llx",
           text.tokens, text.width, values,
           (unsigned long long)fnv(text.values, values * sizeof(uint16_t)));
    edges("first", text.values, 8);
    edges("last", text.values + values - 8, 8);
    printf(" seconds=%.3f\n", elapsed);

    h3_text_embedding_free(&text);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    return 0;
}
