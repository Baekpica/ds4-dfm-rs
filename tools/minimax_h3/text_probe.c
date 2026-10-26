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
 *        h3_text_probe --rope <tokens>
 * Output: the prompt ids, then one TEXT line with the shape, an FNV-1a hash
 * over the BF16 values, both edges, and the wall time.
 *
 * `--rope` prints the encoder's own RoPE tables for <tokens> positions, one
 * `T <position> <64 cos words> <64 sin words>` line each, as raw float32 bit
 * patterns. It runs no layer; it is the E4 Phase A gate against
 * `minimax_h3::text_rope`.
 */

#include "h3_text_encoder.h"
#include "h3_tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* `h3_text_encoder.c:20-27`: file-static there, absent from the header. */
enum { TEXT_HEAD_DIM = 128, TEXT_ROPE_HALF = TEXT_HEAD_DIM / 2 };
static const float TEXT_ROPE_THETA = 5000000.0f;

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

/* `h3_text_encoder.c:543-567` with position_ids == NULL: the float32 tables
 * the encoder uploads, so the port's host-side build can be gated bit for bit
 * before any layer runs. */
static void rope_dump(size_t token_count) {
    float inverse_frequency[TEXT_ROPE_HALF];
    for (size_t index = 0; index < TEXT_ROPE_HALF; index++) {
        inverse_frequency[index] = 1.0f /
            powf(TEXT_ROPE_THETA,
                 (float)(index * 2) / (float)TEXT_HEAD_DIM);
    }

    printf("ROPE tokens=%zu half=%d\n", token_count, TEXT_ROPE_HALF);
    for (size_t position = 0; position < token_count; position++) {
        float cosines[TEXT_ROPE_HALF];
        float sines[TEXT_ROPE_HALF];

        for (size_t index = 0; index < TEXT_ROPE_HALF; index++) {
            float coordinate = (float)position;
            float angle = coordinate * inverse_frequency[index];
            cosines[index] = cosf(angle);
            sines[index] = sinf(angle);
        }

        printf("T %zu", position);
        for (size_t index = 0; index < TEXT_ROPE_HALF; index++) {
            uint32_t bits;
            memcpy(&bits, &cosines[index], sizeof(bits));
            printf(" %08x", bits);
        }
        for (size_t index = 0; index < TEXT_ROPE_HALF; index++) {
            uint32_t bits;
            memcpy(&bits, &sines[index], sizeof(bits));
            printf(" %08x", bits);
        }
        putchar('\n');
    }
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "--rope") == 0) {
        char *end = NULL;
        unsigned long tokens = strtoul(argv[2], &end, 10);
        if (!end || *end || tokens == 0) {
            fprintf(stderr, "usage: h3_text_probe --rope <tokens>\n");
            return 2;
        }
        rope_dump((size_t)tokens);
        return 0;
    }

    if (argc < 4) {
        fprintf(stderr,
                "usage: h3_text_probe <FL2VA/text_encoder> <tokenizer.json> "
                "<prompt>\n"
                "       h3_text_probe --rope <tokens>\n");
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
