/* All immutable source matrices must fit the admission scratch quote.
 * Metadata only: link the production size functions without initializing CUDA. */
#include "../ds4.c"

int main(void) {
    const ds4_host_shape shape = {.variant = DS4_VARIANT_GLM53_FLASH_VQ};
    ds4_host_shape_install(&shape);
    model_apply_host_shape();
    ds4_host_shape_clear();
    const uint32_t cases[][2] = {{1, 1}, {256, 4}, {256, 256},
        {2048, 1}, {2048, 512}, {2048, 2048}};
    unsigned count = 0, matrices = 0;
    if (scanf("%u", &count) != 1 || count != 39862) { return 1; }
    for (unsigned i = 0; i < count; i++) {
        char name[256];
        uint32_t type, ndim;
        uint64_t bytes, dims[DS4_MAX_DIMS];
        if (scanf("%255s %u %u %" SCNu64, name, &type, &ndim, &bytes) != 4 ||
            ndim > DS4_MAX_DIMS) { return 1; }
        for (uint32_t d = 0; d < ndim; d++) {
            if (scanf("%" SCNu64, &dims[d]) != 1 || dims[d] > UINT32_MAX) { return 1; }
        }
        if (ndim != 2 || strstr(name, "embed_tokens.weight") ||
            (type != DS4_TENSOR_BF16 && type != DS4_TENSOR_GLM_FP8 && type != DS4_TENSOR_GLM_VQ)) {
            continue;
        }
        const ds4_gvq_math math = strstr(name, ".hc_attn_fn") || strstr(name, ".hc_ffn_fn") ||
            strstr(name, ".mlp.gate.weight") ? DS4_GVQ_F32 : DS4_GVQ_BF16;
        for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
            const uint32_t ctx = cases[c][0], rows = cases[c][1];
            const uint32_t tokens = strcmp(name, "lm_head.weight") == 0 ? 1u :
                strstr(name, ".self_attn.kv_b_proj.weight") ? ctx : rows;
            const uint64_t need = ds4_gpu_gvq_linear_bytes((uint32_t)dims[1], (uint32_t)dims[0], tokens, type, math);
            const uint64_t quoted = gvq_scratch(ctx, rows);
            if (need > quoted) {
                fprintf(stderr, "GVQ scratch FAIL %s ctx=%u rows=%u need=%llu quote=%llu\n",
                    name, ctx, rows, (unsigned long long)need, (unsigned long long)quoted);
                return 1;
            }
        }
        matrices++;
    }
    printf("GLM VQ scratch PASS: %u matrices, six context/width cases\n", matrices);
    return 0;
}
