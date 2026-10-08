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

enum { GLM53_CACHE_EXPERTS_MAX = 288u };
static const uint64_t GLM53_CACHE_HOT_PIN = UINT64_MAX;

/* Return a whole-row prefix whose route union fits. Repeated top-k IDs
 * consume one slot, so a small cache need not force every projection narrow. */
static uint32_t glm53_cache_rows(const int32_t *ids, uint32_t rows,
        uint32_t used, uint32_t experts, uint32_t capacity) {
    if (!ids || !used || used > experts || experts > GLM53_CACHE_EXPERTS_MAX ||
        !capacity || rows > UINT32_MAX / used) { return 0u; }
    uint8_t seen[GLM53_CACHE_EXPERTS_MAX] = {0};
    uint32_t unique = 0u;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t i = 0; i < used; i++) {
            const int32_t id = ids[(uint64_t)row * used + i];
            if (id < 0 || (uint32_t)id >= experts) { return 0u; }
            if (seen[id]) { continue; }
            seen[id] = 1u;
            if (++unique > capacity) { return row; }
        }
    }
    return rows;
}

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
        if (slots[i].pinned == epoch || slots[i].pinned == GLM53_CACHE_HOT_PIN) { continue; }
        if (!slots[i].used) { return (int)i; }
        if (victim < 0 || slots[i].used < slots[victim].used) {
            victim = (int)i;
        }
    }
    return victim;
}

#endif
