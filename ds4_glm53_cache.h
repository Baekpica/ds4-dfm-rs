#ifndef DS4_GLM53_CACHE_H
#define DS4_GLM53_CACHE_H

#include <stdint.h>

/* A routing batch pins hits before admitting misses. This prevents an early
 * miss from evicting an expert consumed later in the same GPU submission. */
typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint64_t used;
    uint64_t pinned;
} ds4_glm53_cache_slot;

static int glm53_cache_find(const ds4_glm53_cache_slot *slots,
                            uint32_t count, uint32_t layer,
                            uint32_t expert) {
    for (uint32_t i = 0; i < count; i++) {
        if (slots[i].used && slots[i].layer == layer &&
            slots[i].expert == expert) {
            return (int)i;
        }
    }
    return -1;
}

static int glm53_cache_victim(const ds4_glm53_cache_slot *slots,
                              uint32_t count, uint64_t epoch) {
    int victim = -1;
    for (uint32_t i = 0; i < count; i++) {
        if (slots[i].pinned == epoch) { continue; }
        if (!slots[i].used) { return (int)i; }
        if (victim < 0 || slots[i].used < slots[victim].used) {
            victim = (int)i;
        }
    }
    return victim;
}

#endif
