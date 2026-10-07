#include "../ds4_glm53_cache.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM cache FAIL %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

int main(void) {
    ds4_glm53_cache_slot slots[3] = {0};
    CHECK(glm53_cache_find(slots, 3, 3, 287) < 0);
    CHECK(glm53_cache_victim(slots, 3, 1) == 0);
    slots[0] = (ds4_glm53_cache_slot){3, 287, 1, 0};
    slots[1] = (ds4_glm53_cache_slot){43, 287, 2, 0};
    slots[2] = (ds4_glm53_cache_slot){3, 0, 3, 0};
    CHECK(glm53_cache_find(slots, 3, 3, 287) == 0);
    CHECK(glm53_cache_find(slots, 3, 43, 287) == 1);
    CHECK(glm53_cache_find(slots, 3, 3, 0) == 2);
    CHECK(glm53_cache_victim(slots, 3, 4) == 0);

    slots[0].pinned = 4;
    CHECK(glm53_cache_victim(slots, 3, 4) == 1);
    slots[1].pinned = 4;
    CHECK(glm53_cache_victim(slots, 3, 4) == 2);
    slots[2].pinned = 4;
    CHECK(glm53_cache_victim(slots, 3, 4) < 0);
    CHECK(glm53_cache_victim(slots, 3, 5) == 0);
    CHECK(glm53_cache_victim(slots, 0, 5) < 0);
    puts("GLM cache: layer/expert identity, LRU and submission pins passed");
    return 0;
}
