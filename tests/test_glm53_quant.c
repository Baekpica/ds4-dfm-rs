/* Model-free GLM recipe and GGUF byte-range checks. */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM quant FAIL %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

int main(void) {
    uint64_t bytes = 0u;
    CHECK(tensor_nbytes(DS4_TENSOR_IQ2_XS, 256u, &bytes));
    CHECK(bytes == 74u);
    CHECK(tensor_nbytes(DS4_TENSOR_IQ2_XS, 257u, &bytes));
    CHECK(bytes == 148u);
    CHECK(tensor_nbytes(DS4_TENSOR_IQ2_XS, UINT64_MAX, &bytes));
    CHECK(bytes == UINT64_C(5332261958806667264));
    CHECK(!tensor_nbytes(DS4_TENSOR_F32, UINT64_MAX, &bytes));

    g_ds4_shape = DS4_SHAPE_GLM53_FLASH;
    CHECK(tensor_is_routed_expert_type(DS4_TENSOR_IQ2_XS));
    g_ds4_shape = DS4_SHAPE_FLASH;
    CHECK(!tensor_is_routed_expert_type(DS4_TENSOR_IQ2_XS));
    g_ds4_shape = DS4_SHAPE_QWEN38_FLASH_NEXT;
    CHECK(!tensor_is_routed_expert_type(DS4_TENSOR_IQ2_XS));

    puts("GLM quant: IQ2_XS scoped admission and byte rounding passed");
    return 0;
}
