/* Model-free embedded-MTP routing; no weights or GPU allocation. */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GVQ MTP mode FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

int main(void) {
    const ds4_host_shape shape = {.variant = DS4_VARIANT_GLM53_FLASH_VQ};
    ds4_host_shape_install(&shape);
    model_apply_host_shape();
    ds4_host_shape_clear();
    ds4_engine *e = xcalloc(1u, sizeof(*e));
    e->backend = DS4_BACKEND_CUDA;
    e->mtp_draft_tokens = GLM53_MTP_DRAFTS;
    CHECK(!ds4_engine_has_mtp(e));
    e->glm53_mtp = true;
    CHECK(ds4_engine_has_mtp(e));
    CHECK(ds4_engine_mtp_draft_tokens(e) == GLM53_MTP_DRAFTS);
    e->backend = DS4_BACKEND_CPU;
    CHECK(!ds4_engine_has_mtp(e));
    free(e);
    puts("GVQ_MTP_MODE_PASS: embedded CUDA, disabled and CPU routes");
    return 0;
}
