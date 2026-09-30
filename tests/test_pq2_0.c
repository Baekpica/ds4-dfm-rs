/*
 * PQ2_0 block format test: the Prism-private 2-bit format used by the Bonsai
 * GGUFs (ggml type 142: 128 weights per block, fp16 scale then 32 code bytes).
 *
 * The block bytes and the expected f32 checksums below were produced by the
 * reference dequantizer in the PrismML llama.cpp fork
 * (ggml_get_type_traits(GGML_TYPE_PQ2_0)->to_float) over the same bytes; the
 * generator is tests/pq2_0/pq2_ref_generator.c.  A checksum mismatch means this
 * implementation no longer reads the file the way the reference does.
 *
 * Build: make pq2-0-test
 */

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define QK_PQ2_0    128
#define PQ2_0_BYTES 34

typedef struct {
    uint16_t d;                    /* fp16 scale */
    uint8_t  qs[QK_PQ2_0 / 4];     /* 2 bits per weight, 4 weights per byte */
} block_pq2_0;

typedef struct {
    const char *   label;
    uint8_t        raw[PQ2_0_BYTES];
    uint64_t       checksum;
    float          head[3];
} pq2_case;

static uint64_t fnv1a_f32(const float * x, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    const uint8_t * b = (const uint8_t *) x;
    for (size_t i = 0; i < n * sizeof(float); i++) {
        h ^= b[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

/* fp16 -> fp32, matching ds4's f16_to_f32 (denormals and infinities included). */
static float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const int32_t  exp  = (h >> 10) & 0x1f;
    const uint32_t frac = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (frac == 0) { bits = sign; }
        else {
            /* subnormal: normalize */
            int e = -1;
            uint32_t f = frac;
            do { f <<= 1; e++; } while ((f & 0x400u) == 0);
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((f & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (frac << 13);
    } else {
        bits = sign | ((uint32_t)(exp - 15 + 127) << 23) | (frac << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/* The ds4 reading of one PQ2_0 block: element j sits in byte j/4 at bits
 * (j%4)*2, LSB first, and the 2-bit code c reconstructs the level c-1. */
static void pq2_0_dequant_block(const block_pq2_0 * blk, float * out) {
    const float d = h2f(blk->d);
    for (int j = 0; j < QK_PQ2_0; j++) {
        const uint8_t code = (uint8_t)((blk->qs[j / 4] >> (2 * (j % 4))) & 0x3u);
        out[j] = (float)((int)code - 1) * d;
    }
}

/* Reference dot product of one PQ2_0 block against a float row: the
 * double-accumulated form ds4's reference path uses. */
static double pq2_0_dot_block(const block_pq2_0 * blk, const float * x) {
    float row[QK_PQ2_0];
    pq2_0_dequant_block(blk, row);
    double acc = 0.0;
    for (int j = 0; j < QK_PQ2_0; j++) acc += (double) row[j] * x[j];
    return acc;
}

static const pq2_case g_cases[] = {
    {
        "uniform codes",
        {
            0x00, 0x28, 0x28, 0xd4, 0x2c, 0xec, 0x20, 0x14, 0x66, 0x97, 0xed,
            0xbc, 0x97, 0x2e, 0x8e, 0x06, 0xab, 0x5f, 0x60, 0x26, 0x96, 0xca,
            0x2b, 0xfb, 0x7d, 0x89, 0xcb, 0xc2, 0x7b, 0x48, 0xce, 0x3c, 0xa6,
            0xb0,
        },
        0x796b7f1eb861cc55ull,
        { -0.03125f, 0.03125f, 0.03125f },
    },
    {
        "ternary codes, negative scale",
        {
            0x00, 0xa0, 0x6a, 0x56, 0x59, 0x56, 0x96, 0x65, 0xa6, 0xaa, 0x6a,
            0x66, 0xa5, 0x96, 0xa6, 0x9a, 0xa9, 0x5a, 0xa5, 0x69, 0x65, 0xa5,
            0xaa, 0x95, 0x5a, 0x9a, 0xa6, 0xa9, 0xa9, 0x65, 0x6a, 0x5a, 0x69,
            0x95,
        },
        0x84a94abdecd49cc1ull,
        { -0.0078125f, -0.0078125f, -0.0078125f },
    },
    {
        "all code 3",
        {
            0x00, 0x3c, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff,
        },
        0x205816f530654b25ull,
        { 2.0f, 2.0f, 2.0f },
    },
    {
        "all code 0",
        {
            0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00,
        },
        0x888be5f3ccc2fb25ull,
        { -0.0009765625f, -0.0009765625f, -0.0009765625f },
    },
    {
        "zero scale",
        {
            0x00, 0x00, 0x14, 0x01, 0x21, 0xd8, 0x92, 0xa1, 0x95, 0x75, 0x09,
            0x55, 0x64, 0x24, 0x71, 0xd2, 0x6b, 0x90, 0xe7, 0xae, 0xfb, 0x94,
            0x0f, 0x9d, 0x80, 0xb2, 0x95, 0x95, 0x34, 0x51, 0x9f, 0x37, 0xa8,
            0x58,
        },
        0x84a6a1ece5f83425ull,
        { -0.0f, 0.0f, 0.0f },
    },
    {
        "subnormal scale",
        {
            0x03, 0x00, 0x43, 0x32, 0xff, 0x33, 0xf2, 0xc8, 0x20, 0xaa, 0x74,
            0xca, 0x37, 0xe6, 0x77, 0xa1, 0x30, 0x48, 0xe4, 0xb7, 0x67, 0xe3,
            0xaf, 0xee, 0x26, 0xb7, 0x6a, 0xad, 0x22, 0x36, 0x1d, 0x4d, 0x93,
            0x2c,
        },
        0xf324b5f5db33ffe9ull,
        { 3.57627869e-07f, -1.78813934e-07f, -1.78813934e-07f },
    },
};

static int check_dequant(void) {
    int failures = 0;
    for (size_t c = 0; c < sizeof(g_cases) / sizeof(g_cases[0]); c++) {
        const pq2_case * tc = &g_cases[c];
        block_pq2_0 blk;
        memcpy(&blk, tc->raw, sizeof(blk));

        float out[QK_PQ2_0];
        pq2_0_dequant_block(&blk, out);

        const uint64_t sum = fnv1a_f32(out, QK_PQ2_0);
        if (sum != tc->checksum) {
            printf("FAIL %-32s checksum %016" PRIx64 " expected %016" PRIx64 "\n",
                   tc->label, sum, tc->checksum);
            failures++;
        }
        const int head_idx[3] = { 0, 1, QK_PQ2_0 - 1 };
        for (int i = 0; i < 3; i++) {
            const float got = out[head_idx[i]];
            if (memcmp(&got, &tc->head[i], sizeof(float)) != 0) {
                printf("FAIL %-32s value[%d] %.9g expected %.9g\n",
                       tc->label, head_idx[i], got, tc->head[i]);
                failures++;
            }
        }
    }
    return failures;
}

/* The four codes must map to -d, 0, +d, +2d in that order, element 4b in the
 * low bits of byte b. */
static int check_code_mapping(void) {
    block_pq2_0 blk;
    memset(&blk, 0, sizeof(blk));
    blk.d = 0x3c00;               /* 1.0 */
    blk.qs[0] = 0xe4;             /* codes for elements 0..3: 0, 1, 2, 3 */
    float out[QK_PQ2_0];
    pq2_0_dequant_block(&blk, out);

    const float expect[4] = { -1.0f, 0.0f, 1.0f, 2.0f };
    int failures = 0;
    for (int i = 0; i < 4; i++) {
        if (memcmp(&out[i], &expect[i], sizeof(float)) != 0) {
            printf("FAIL code mapping: element %d = %.9g expected %.9g\n", i, out[i], expect[i]);
            failures++;
        }
    }
    return failures;
}

/* A row is a whole number of blocks; the row reader must use one scale per
 * 128 weights and never carry state across blocks. */
static int check_row_stepping(void) {
    block_pq2_0 row[3];
    memset(row, 0, sizeof(row));
    row[0].d = 0x3c00; row[0].qs[0] = 0x02;   /* element 0 = +1 */
    row[1].d = 0x3c00; row[1].qs[0] = 0xe4;   /* codes 0,1,2,3 again */
    row[2].d = 0x0000; row[2].qs[0] = 0xff;

    float out[3 * QK_PQ2_0];
    for (int b = 0; b < 3; b++) pq2_0_dequant_block(&row[b], out + b * QK_PQ2_0);

    int failures = 0;
    if (out[0] != 1.0f || out[1] != -1.0f || out[QK_PQ2_0 - 1] != -1.0f) {
        printf("FAIL row stepping: block 0 = %.9g %.9g ... %.9g\n",
               out[0], out[1], out[QK_PQ2_0 - 1]);
        failures++;
    }
    const float expect[4] = { -1.0f, 0.0f, 1.0f, 2.0f };
    for (int i = 0; i < 4; i++) {
        if (memcmp(&out[QK_PQ2_0 + i], &expect[i], sizeof(float)) != 0) {
            printf("FAIL row stepping: block 1 element %d = %.9g expected %.9g\n",
                   i, out[QK_PQ2_0 + i], expect[i]);
            failures++;
        }
    }
    for (int i = 0; i < QK_PQ2_0; i++) {
        if (out[2 * QK_PQ2_0 + i] != 0.0f) {
            printf("FAIL row stepping: zero scale block produced %.9g\n", out[2 * QK_PQ2_0 + i]);
            failures++;
            break;
        }
    }
    return failures;
}

/* The dot product must use the same levels as the dequantizer; a ternary row
 * (codes 1 and 2 only) is exactly sum(x) - sum(x over the code-1 positions). */
static int check_dot(void) {
    block_pq2_0 blk;
    memset(&blk, 0, sizeof(blk));
    blk.d = 0x3c00;                   /* 1.0 */
    for (int b = 0; b < QK_PQ2_0 / 4; b++) blk.qs[b] = 0xaa;   /* codes 2,2,2,2 */
    float x[QK_PQ2_0];
    for (int i = 0; i < QK_PQ2_0; i++) x[i] = (float) (i % 7) * 0.25f - 0.75f;

    const double got = pq2_0_dot_block(&blk, x);
    double want = 0.0;
    for (int i = 0; i < QK_PQ2_0; i++) want += x[i];           /* every code is 2 -> +1 * d */

    int failures = 0;
    if (fabs(got - want) > 1e-9) {
        printf("FAIL dot: %.12g expected %.12g\n", got, want);
        failures++;
    }
    /* and the all-zero-code block must give minus that sum */
    for (int b = 0; b < QK_PQ2_0 / 4; b++) blk.qs[b] = 0x00;
    const double got_neg = pq2_0_dot_block(&blk, x);
    if (fabs(got_neg + want) > 1e-9) {
        printf("FAIL dot: all-code-0 %.12g expected %.12g\n", got_neg, -want);
        failures++;
    }
    return failures;
}

int main(void) {
    if (sizeof(block_pq2_0) != PQ2_0_BYTES) {
        printf("FAIL block size %zu expected %d\n", sizeof(block_pq2_0), PQ2_0_BYTES);
        return 1;
    }
    int failures = 0;
    failures += check_dequant();
    failures += check_code_mapping();
    failures += check_row_stepping();
    failures += check_dot();

    if (failures) {
        printf("pq2_0: %d failure(s)\n", failures);
        return 1;
    }
    printf("pq2_0: all checks passed (%zu reference blocks, %d bytes/block, %.3f bpw)\n",
           sizeof(g_cases) / sizeof(g_cases[0]), PQ2_0_BYTES,
           8.0 * (double) PQ2_0_BYTES / (double) QK_PQ2_0);
    return 0;
}
