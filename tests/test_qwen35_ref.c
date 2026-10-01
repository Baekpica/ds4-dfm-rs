/* The dense FFN down projection has no recurrent state. UBSan catches the
 * null member access that optimized builds can otherwise discard. */
#include "../ds4.c"

enum { ROW_WIDTH = 128, BLOCK_BYTES = 34 };

int main(void) {
    uint8_t block[BLOCK_BYTES] = {0};
    const uint16_t scale = 0x3c00; /* FP16 one. */
    memcpy(block, &scale, sizeof(scale));
    memset(block + sizeof(scale), 0xaa, BLOCK_BYTES - sizeof(scale));

    ds4_model model = {.map = block, .size = sizeof(block)};
    ds4_tensor weight = {
        .ndim = 2,
        .dim = {ROW_WIDTH, 1},
        .type = DS4_TENSOR_PQ2_0,
        .bytes = sizeof(block),
    };
    float input[ROW_WIDTH] = {1.0f};
    float output;

    /* Every stored weight is one; an unfolded unit input has dot product one. */
    ds4_qwen35_ref_matvec_folded(&model, &weight, input, &output, false, NULL);
    if (output != 1.0f) {
        fprintf(stderr, "unfolded stateless projection: %g, expected 1\n", output);
        return 1;
    }

    /* H times a unit input is constant 1/sqrt(width), with no state needed. */
    g_hadamard.enabled = true;
    g_hadamard.block_size = ROW_WIDTH;
    ds4_qwen35_ref_matvec_folded(&model, &weight, input, &output, false, NULL);
    const float expected = sqrtf((float)ROW_WIDTH);
    if (!isfinite(output) || fabsf(output - expected) > 1e-5f) {
        fprintf(stderr, "folded stateless projection: %g, expected %g\n",
                output, expected);
        return 1;
    }

    puts("qwen35 reference: stateless dense projections PASS");
    return 0;
}
