/* Model-free attention frontier gate; persistent rows have no top-k cap. */
#include "../ds4_glm53_attn.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM attention FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

int main(void) {
    const uint32_t ctx = 1048576u;
    CHECK(glm53_attn_frontier(1u, 2050u, ctx, 0u, GLM53_ATTN_ALL));
    CHECK(glm53_attn_frontier(1u, 2051u, ctx, 0u, GLM53_ATTN_ALL));
    CHECK(glm53_attn_frontier(3u, 4096u, ctx, 0u, GLM53_ATTN_ALL));
    CHECK(glm53_attn_frontier(1u, ctx - 1u, ctx, 0u, GLM53_ATTN_ALL));
    CHECK(!glm53_attn_frontier(0u, 0u, ctx, 0u, GLM53_ATTN_ALL));
    CHECK(!glm53_attn_frontier(2u, UINT32_MAX - 1u, UINT32_MAX, 0u, GLM53_ATTN_ALL));
    CHECK(!glm53_attn_frontier(1u, ctx, ctx, 0u, GLM53_ATTN_ALL));
    CHECK(!glm53_attn_frontier(1u, 0u, ctx, 1u, GLM53_ATTN_ALL));
    CHECK(glm53_attn_frontier(1u, ctx - 1u, ctx, DS4_GLM53_MAX_SELECTED, GLM53_ATTN_SELECTED));
    CHECK(!glm53_attn_frontier(1u, 0u, ctx, DS4_GLM53_MAX_SELECTED + 1u, GLM53_ATTN_SELECTED));
    CHECK(!glm53_attn_frontier(1u, 0u, ctx, 0u, GLM53_ATTN_SELECTED));
    puts("GLM attention: full context frontier and selected bounds passed");
    return 0;
}
